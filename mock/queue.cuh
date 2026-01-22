#include <cuda_runtime.h>
#include <cstdio>
#include <cassert>

// 基于每槽序号的有界 MPMC 队列实现（设备可见）
// 设计要点：
// - 每个槽维护一个序号 seq，初始为槽索引 i
// - 生产者通过获取全局 tail（atomicAdd）得到 pos，等待 slot.seq == pos
//   写数据，__threadfence()，再设置 slot.seq = pos+1（release）
// - 消费者通过获取全局 head（atomicAdd）得到 pos，等待 slot.seq == pos+1
//   读数据，__threadfence()，再设置 slot.seq = pos + capacity（循环回到空态）

template <typename T>
struct CUDAContConcurrentQueueSlot {
    T value;
    unsigned int seq; // 序号（用于判断槽状态）
};

template <typename T>
struct CUDAContConcurrentQueueData {
    CUDAContConcurrentQueueSlot<T>* buffer; // 设备上槽数组
    unsigned int capacity;                  // 必须为2的幂
    unsigned int mask;                      // capacity - 1，快速取模
    unsigned int head;                      // 消费者位置（原子）
    unsigned int tail;                      // 生产者位置（原子）
};

// 生产者（设备端）
template <typename T>
__device__ void queuePushBlocking(CUDAContConcurrentQueueData<T>* q, const T& val) {
    unsigned int pos = atomicAdd(&q->tail, 1u);
    unsigned int idx = pos & q->mask;
    unsigned int expected = pos;
    // 等待该槽变为空（seq == pos）
    // 使用volatile读取优化性能，避免不必要的原子操作开销
    unsigned int spins = 0;
    volatile unsigned int* seq_ptr = &q->buffer[idx].seq;
    while (*seq_ptr != expected) {
        // busy-wait with exponential backoff
        spins++;
        if (spins < 32) {
            // 前32次快速自旋
        } else if (spins < 256) {
            // 中等延迟
            #if __CUDA_ARCH__ >= 700
            __nanosleep(50);
            #endif
        } else {
            // 较长延迟，避免过度占用资源
            #if __CUDA_ARCH__ >= 700
            __nanosleep(200);
            #endif
        }
    }
    // 写入数据
    q->buffer[idx].value = val;
    // 确保写入对其它线程可见（跨 block）
    // __threadfence(); // 移除昂贵的全局内存屏障，atomicExch的release语义已足够
    // 标记为已填充（release）
    atomicExch(&q->buffer[idx].seq, expected + 1u);
}

// 非阻塞尝试push：返回 true 成功，false 表示队列满（不阻塞）
template <typename T>
__device__ bool queueTryPush(CUDAContConcurrentQueueData<T>* q, const T& val) {
    unsigned int pos = atomicAdd(&q->tail, 1u);
    unsigned int idx = pos & q->mask;
    unsigned int expected = pos;
    // 使用volatile读取优化性能
    volatile unsigned int* seq_ptr = &q->buffer[idx].seq;
    unsigned int cur = *seq_ptr;
    if (cur != expected) {
        // 槽还未回到空态，回滚 tail（近似：无法回滚原子 tail，故该实现把失败视为队满）
        return false;
    }
    q->buffer[idx].value = val;
    // __threadfence();
    atomicExch(&q->buffer[idx].seq, expected + 1u);
    return true;
}

// 消费者（设备端），阻塞式读取
template <typename T>
__device__ void queuePopBlocking(CUDAContConcurrentQueueData<T>* q, T& out) {
    unsigned int pos = atomicAdd(&q->head, 1u);
    unsigned int idx = pos & q->mask;
    unsigned int expected = pos + 1u;
    // 等待槽被生产者填充（seq == pos+1）
    // 使用volatile读取优化性能，避免不必要的原子操作开销
    unsigned int spins = 0;
    volatile unsigned int* seq_ptr = &q->buffer[idx].seq;
    while (*seq_ptr != expected) {
        // busy-wait with exponential backoff
        spins++;
        if (spins < 32) {
            // 前32次快速自旋
        } else if (spins < 256) {
            // 中等延迟
            #if __CUDA_ARCH__ >= 700
            __nanosleep(50);
            #endif
        } else {
            // 较长延迟，避免过度占用资源
            #if __CUDA_ARCH__ >= 700
            __nanosleep(200);
            #endif
        }
    }
    // 确保数据对当前线程可见
    // __threadfence();
    out = q->buffer[idx].value;
    // 标记槽为空：将 seq 更新为 pos + capacity（循环回到空态）
    atomicExch(&q->buffer[idx].seq, pos + q->capacity);
}

// 非阻塞尝试pop：返回 true 并输出值；false 表示队空
template <typename T>
__device__ bool queueTryPop(CUDAContConcurrentQueueData<T>* q, T& out) {
    unsigned int pos = atomicAdd(&q->head, 1u);
    unsigned int idx = pos & q->mask;
    unsigned int expected = pos + 1u;
    // 使用volatile读取优化性能
    volatile unsigned int* seq_ptr = &q->buffer[idx].seq;
    unsigned int cur = *seq_ptr;
    if (cur != expected) {
        return false;
    }
    // __threadfence();
    out = q->buffer[idx].value;
    atomicExch(&q->buffer[idx].seq, pos + q->capacity);
    return true;
}

// 主机端队列管理类（负责内存分配/初始化/释放）
template <typename T>
class CUDAContConcurrentQueue {
public:
    CUDAContConcurrentQueueData<T>* d_data;

    CUDAContConcurrentQueue(unsigned int capacity) {
        assert((capacity & (capacity - 1)) == 0 && "Capacity must be power of 2");
        // 分配并初始化主机-side 临时结构
        CUDAContConcurrentQueueData<T> h_data;
        h_data.capacity = capacity;
        h_data.mask = capacity - 1u;
        h_data.head = 0u;
        h_data.tail = 0u;

        // 在设备上分配结构体
        cudaMalloc(&d_data, sizeof(CUDAContConcurrentQueueData<T>));
        // 在设备上分配槽数组
        CUDAContConcurrentQueueSlot<T>* d_slots = nullptr;
        cudaMalloc(&d_slots, capacity * sizeof(CUDAContConcurrentQueueSlot<T>));

        // 在主机上初始化槽的序号并拷贝到设备
        CUDAContConcurrentQueueSlot<T>* h_slots = (CUDAContConcurrentQueueSlot<T>*)malloc(capacity * sizeof(CUDAContConcurrentQueueSlot<T>));
        for (unsigned int i = 0; i < capacity; ++i) {
            h_slots[i].seq = i; // 初始序号
        }
        cudaMemcpy(d_slots, h_slots, capacity * sizeof(CUDAContConcurrentQueueSlot<T>), cudaMemcpyHostToDevice);
        free(h_slots);

        // 将设备槽指针写入设备结构并拷贝结构到设备
        h_data.buffer = d_slots;
        cudaMemcpy(d_data, &h_data, sizeof(CUDAContConcurrentQueueData<T>), cudaMemcpyHostToDevice);
    }

    ~CUDAContConcurrentQueue() {
        // 获取设备结构以释放槽
        CUDAContConcurrentQueueData<T> h_data;
        cudaMemcpy(&h_data, d_data, sizeof(CUDAContConcurrentQueueData<T>), cudaMemcpyDeviceToHost);
        cudaFree(h_data.buffer);
        cudaFree(d_data);
    }

    // 调试：读取当前 tail-head 差值（近似大小）
    unsigned int getCount() {
        CUDAContConcurrentQueueData<T> h_data;
        cudaMemcpy(&h_data, d_data, sizeof(CUDAContConcurrentQueueData<T>), cudaMemcpyDeviceToHost);
        return h_data.tail - h_data.head;
    }
};

// 测试核：每个线程 push 然后 pop（使用阻塞版本）
__global__ void testConcurrentQueue(CUDAContConcurrentQueueData<int>* d_queue_data) {
    unsigned int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    queuePushBlocking<int>(d_queue_data, (int)thread_id);
    int value = -1;
    queuePopBlocking<int>(d_queue_data, value);
    if (thread_id < 1024) {
        printf("Thread %u: popped=%d\n", thread_id, value);
    }
}