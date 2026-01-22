#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>


// 一个槽位：ticket 用来判定该槽是否可写/可读
template <typename T>
struct __align__(16) QueueSlot {
    unsigned long long ticket; // 状态序号
    T value;
};

// MPSC：多生产者 push，单消费者 pop
template <typename T>
struct GpuMpscQueue {
    QueueSlot<T>* slots;
    unsigned int capacity;      // 必须是 2 的幂
    unsigned int mask;          // capacity - 1
    unsigned long long head;    // 消费者用（出队序号）
    unsigned long long tail;    // 生产者用（入队序号）

    __device__ __forceinline__ void init_device(QueueSlot<T>* s, unsigned int cap) {
        slots = s;
        capacity = cap;
        mask = cap - 1;
        head = 0;
        tail = 0;
        // 初始化每个槽位的 ticket = index
        for (unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
            i < cap;
            i += blockDim.x * gridDim.x) {
            slots[i].ticket = (unsigned long long)i;
        }
    }

    // push: 多生产者并发安全
    // 返回 true 表示成功入队；false 表示队列满（自旋失败）
    __device__ __forceinline__ bool push(const T& v) {
        unsigned long long t = atomicAdd((unsigned long long*)&tail, 1ULL);
        QueueSlot<T>* slot = &slots[(unsigned int)t & mask];

        // slot->ticket == t 时可写
        // 如果队列满，会看到 ticket < t（还没被消费者推进到下一个周期）
        // 这里做有限自旋，避免死等
        int spins = 0;
        while (true) {
            unsigned long long tk = slot->ticket;
            long long diff = (long long)tk - (long long)t;
            if (diff == 0) {
                // 抢到可写位置：写 value，再把 ticket 置为 t+1（可读）
                slot->value = v;
                __threadfence();                 // 确保 value 对其他线程可见
                slot->ticket = t + 1ULL;
                return true;
            } else if (diff < 0) {
                // tk < t：槽位还没被释放到这个序号 => 队列满/落后
                // 退让：你也可以改成更激进的等待策略
                if (++spins > 1024) return false;
            }
            // 轻量 pause
#if __CUDA_ARCH__ >= 700
            __nanosleep(50);
#endif
        }
    }

    // pop: 单消费者安全（只能一个线程/warp/block来调用）
    // 返回 true 表示成功出队；false 表示队列空
    __device__ __forceinline__ bool pop(T& out) {
        unsigned long long h = head;
        QueueSlot<T>* slot = &slots[(unsigned int)h & mask];

        // slot->ticket == h+1 时可读
        unsigned long long tk = slot->ticket;
        long long diff = (long long)tk - (long long)(h + 1ULL);
        if (diff == 0) {
            out = slot->value;
            __threadfence(); // 防止重排（可选但建议保守）
            // 释放该槽：ticket = h + capacity（进入下一个周期可写）
            slot->ticket = h + (unsigned long long)capacity;
            head = h + 1ULL;
            return true;
        }
        return false; // 空
    }
};


__global__ void demo_queue_kernel_push(GpuMpscQueue<int>* q, int nPush, int* out, int* outCount) {
    // 多生产者：所有线程 push 自己的值
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < nPush) {
        // 生产一个简单值
        int v = tid;
        bool success = q->push(v);
        while (!success) {
            // 自旋直到成功
            success = q->push(v);
        }
        printf("pushed %d\n", v);
    }

    // __syncthreads();

    // // 单消费者：用一个线程把能 pop 的全 pop 出来
    // if (blockIdx.x == 0 && threadIdx.x == 0) {
    //     int cnt = 0;
    //     int v;
    //     // 尝试 pop nPush 次（队列可能没满就足够）
    //     for (int i = 0; i < nPush; i++) {
    //         if (q->pop(v)) {
    //             printf("popped %d\n", v);
    //             out[cnt++] = v;
    //         }
    //     }
    //     *outCount = cnt;
    // }
}

__global__ void demo_queue_kernel_pop(GpuMpscQueue<int>* q, int nPush, int* out, int* outCount) {
    // 单消费者：用一个线程把能 pop 的全 pop 出来
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        int cnt = 0;
        int totalpopped = 0;
        int v;
        // 尝试 pop nPush 次（队列可能没满就足够）
        while (totalpopped < nPush) {
            if (q->pop(v)) {
                totalpopped++;
                printf("popped %d\n", v);
                out[cnt++] = v;
            }
        }
        *outCount = cnt;
    }
}

template <typename T>
__global__ void init_queue_kernel(GpuMpscQueue<T>* q, QueueSlot<T>* slots, unsigned int cap) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        q->slots = slots;
        q->capacity = cap;
        q->mask = cap - 1;
        q->head = 0;
        q->tail = 0;
    }
    __syncthreads();
    // 初始化 tickets
    for (unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
        i < cap;
        i += blockDim.x * gridDim.x) {
        slots[i].ticket = (unsigned long long)i;
    }
}


int main() {
    const unsigned int CAP = 1u << 8; // 65536，2的幂
    const int nPush = 4096;

    // device 内存
    GpuMpscQueue<int>* d_q;
    QueueSlot<int>* d_slots;
    int *d_out, *d_outCount;

    cudaMalloc(&d_q, sizeof(GpuMpscQueue<int>));
    cudaMalloc(&d_slots, sizeof(QueueSlot<int>) * CAP);
    cudaMalloc(&d_out, sizeof(int) * nPush);
    cudaMalloc(&d_outCount, sizeof(int));

    init_queue_kernel<int><<<128, 256>>>(d_q, d_slots, CAP);
    cudaDeviceSynchronize();

    cudaStream_t stream_a, stream_b;
    cudaStreamCreate(&stream_a);
    cudaStreamCreate(&stream_b);

    demo_queue_kernel_push<<<nPush / 256, 256, 0, stream_a>>>(d_q, nPush, d_out, d_outCount);
    demo_queue_kernel_pop<<<1, 1, 0, stream_b>>>(d_q, nPush, d_out, d_outCount);
    cudaDeviceSynchronize();

    int outCount = 0;
    cudaMemcpy(&outCount, d_outCount, sizeof(int), cudaMemcpyDeviceToHost);

    // printf("popped = %d\n", outCount);

    cudaFree(d_outCount);
    cudaFree(d_out);
    cudaFree(d_slots);
    cudaFree(d_q);
    return 0;
}
