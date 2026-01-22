#include <cuda_runtime.h>
#include <cstdio>
#include <cassert>

// CUDA并发队列的设备端数据结构（GPU可见）
template <typename T>
struct CUDAContConcurrentQueueData {
    T* buffer;          // 队列缓冲区
    int capacity;       // 队列容量（必须是2的幂）
    int head;           // 读指针（pop位置）
    int tail;           // 写指针（push位置）
    int count;          // 当前元素数量
};

// 独立的设备端push函数（线程安全）
template <typename T>
__device__ bool queuePush(CUDAContConcurrentQueueData<T>* data, const T& value) {
    // 原子增加count，判断是否队列已满
    int old_count = atomicAdd(&data->count, 1);
    if (old_count >= data->capacity) {
        // 队列已满，回滚count并返回失败
        atomicSub(&data->count, 1);
        return false;
    }

    // 计算写入位置（tail是环形指针，位运算替代取模）
    int pos = atomicAdd(&data->tail, 1) & (data->capacity - 1);
    data->buffer[pos] = value;
    
    return true;
}

// 独立的设备端pop函数（线程安全）
template <typename T>
__device__ bool queuePop(CUDAContConcurrentQueueData<T>* data, T& value) {
    // 原子减少count，判断是否队列已空
    int old_count = atomicSub(&data->count, 1);
    if (old_count <= 0) {
        // 队列已空，回滚count并返回失败
        atomicAdd(&data->count, 1);
        return false;
    }

    // 计算读取位置（head是环形指针）
    int pos = atomicAdd(&data->head, 1) & (data->capacity - 1);
    value = data->buffer[pos];
    
    return true;
}

// 主机端队列管理类（仅负责内存分配/释放）
template <typename T>
class CUDAContConcurrentQueue {
public:
    CUDAContConcurrentQueueData<T>* d_data;

    // 构造函数：初始化GPU队列
    CUDAContConcurrentQueue(int capacity) {
        // 确保容量是2的幂（简化环形缓冲区的取模操作）
        assert((capacity & (capacity - 1)) == 0 && "Capacity must be power of 2");
        
        // 主机端分配并初始化设备数据结构
        CUDAContConcurrentQueueData<T> h_data;
        h_data.capacity = capacity;
        h_data.head = 0;
        h_data.tail = 0;
        h_data.count = 0;

        // 分配设备端数据结构内存
        cudaMalloc(&d_data, sizeof(CUDAContConcurrentQueueData<T>));
        // 分配设备端缓冲区内存
        cudaMalloc(&h_data.buffer, capacity * sizeof(T));
        // 将初始化好的主机数据拷贝到设备
        cudaMemcpy(d_data, &h_data, sizeof(CUDAContConcurrentQueueData<T>), cudaMemcpyHostToDevice);
    }

    // 析构函数：释放GPU内存（仅主机端执行）
    ~CUDAContConcurrentQueue() {
        // 先获取缓冲区指针并释放
        CUDAContConcurrentQueueData<T> h_data;
        cudaMemcpy(&h_data, d_data, sizeof(CUDAContConcurrentQueueData<T>), cudaMemcpyDeviceToHost);
        cudaFree(h_data.buffer);
        // 释放数据结构本身
        cudaFree(d_data);
    }

    // 主机端获取当前队列大小（仅供调试）
    int getCount() {
        CUDAContConcurrentQueueData<T> h_data;
        cudaMemcpy(&h_data, d_data, sizeof(CUDAContConcurrentQueueData<T>), cudaMemcpyDeviceToHost);
        return h_data.count;
    }
};

// 测试核函数：多线程push和pop（无对象创建/析构）
__global__ void testConcurrentQueue(CUDAContConcurrentQueueData<int>* d_queue_data) {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 每个线程先push一个值（线程ID）
    bool push_ok = queuePush(d_queue_data, thread_id);
    while (!push_ok) {
        // 如果push失败（队列满），尝试重新push
        push_ok = queuePush(d_queue_data, thread_id);
    }
    
    // 同步所有线程，确保push完成后再pop
    // __syncthreads();
    
    // 每个线程尝试pop一个值
    int value = -1;
    bool pop_ok = queuePop(d_queue_data, value);
    
    // 输出结果（仅前32个线程输出，避免输出过多）
    if (thread_id < 16*64) {
        printf("Thread %d: push_ok=%d, pop_ok=%d, pop_value=%d\n", 
               thread_id, push_ok, pop_ok, value);
    }
}

// 主函数：测试并发队列
int main() {
    // 队列容量（2的幂）
    const int QUEUE_CAPACITY = 1024 / 2;
    // 线程配置：16个block，每个block 64个thread（共1024线程）
    const int BLOCKS = 16;
    const int THREADS_PER_BLOCK = 64;

    // 创建并发队列（仅主机端管理内存）
    CUDAContConcurrentQueue<int> queue(QUEUE_CAPACITY);

    // 启动测试核函数
    testConcurrentQueue<<<BLOCKS, THREADS_PER_BLOCK>>>(queue.d_data);
    
    // 等待核函数执行完成
    cudaDeviceSynchronize();
    
    // 检查队列最终大小（应该为0，因为push和pop数量相等）
    printf("Final queue count: %d\n", queue.getCount());

    // 检查CUDA错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return -1;
    }

    return 0;
}