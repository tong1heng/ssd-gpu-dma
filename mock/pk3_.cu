#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <unistd.h>
#include "queue.cuh"


struct Context {
    int value = 0;
    int state = 0;
    uint64_t timer_start = 0;
    uint64_t timer_end = 0;
};

__device__ __forceinline__ unsigned long long globaltimer_ns() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

__device__ __forceinline__ dim3 get_3d_idx(int idx, dim3 dim) {
  dim3 result;
  result.x = idx % dim.x;
  result.y = idx / dim.x % dim.y;
  result.z = idx / (dim.x * dim.y);
  return result;
}

__device__ void do_kernel(Context* ctx, int task_id) {
    Context* task_ctx = &ctx[task_id];
    if (task_ctx->state == 0) {
        // atomicAdd(&task_ctx->value, 1);
        if (threadIdx.x + threadIdx.y + threadIdx.z == 0) {
            task_ctx->value += 1;
            // atomicAdd(&task_ctx->value, 1);
            task_ctx->state = 1;
            task_ctx->timer_start = globaltimer_ns();
        }
    } else if (task_ctx->state == 1) {
        // atomicAdd(&task_ctx->value, 10);
        if (threadIdx.x + threadIdx.y + threadIdx.z == 0) {
            task_ctx->value += 10;
            // atomicAdd(&task_ctx->value, 1);
            task_ctx->state = 2;
            task_ctx->timer_end = globaltimer_ns();
        }
    } else {
        // atomicAdd(&task_ctx->value, 100);
        if (threadIdx.x + threadIdx.y + threadIdx.z == 0) {
            task_ctx->value += 100;
            // atomicAdd(&task_ctx->value, 1);
            task_ctx->state = 3;
        }
    }
    // printf("Task %d executed state %d by block (%d,%d,%d) thread (%d,%d,%d), new value: %d\n", 
    //        task_id, task_ctx->state - 1, blockIdx.x, blockIdx.y, blockIdx.z, 
    //        thread_idx.x, thread_idx.y, thread_idx.z, task_ctx->value);
}

__global__ void persistent_kernel(int task_num, int task_offset, 
                                  CUDAContConcurrentQueueData<int>* task_queue, 
                                  unsigned int* stop_signal, Context* ctx) {
    unsigned int warp_idx = threadIdx.x % 32;
    unsigned int stop_sig;

    // dim3 block_idx = dim3(blockIdx.x, blockIdx.y, blockIdx.z);
    // dim3 thread_idx = dim3(threadIdx.x, threadIdx.y, threadIdx.z);

    while (true) {
        if (warp_idx == 0) {
            stop_sig = *((volatile unsigned int *)stop_signal);
        }
        stop_sig = __shfl_sync(0xFFFFFFFF, stop_sig, 0);
        if (stop_sig) {
            printf("Persistent kernel received stop signal.\n");
            break;
        }

        __shared__ int task_id;
        __shared__ bool pop_ok;
        
        if (threadIdx.x + threadIdx.y + threadIdx.z == 0) {
            // first thread of the block
            pop_ok = queueTryPop(task_queue, task_id);
            // printf("Popped task_id %d by thread block: %d\n", task_id, blockIdx.x);
        }
        __syncthreads();
        if (!pop_ok) {
            continue;
        }

        do_kernel(ctx, task_id);
        __syncthreads();

        __shared__ bool task_done;
        if (threadIdx.x + threadIdx.y + threadIdx.z == 0) {
            // first thread of the block
            if (ctx[task_id].state >= 3) {
                task_done = true;
                // printf("Task %d completed all states, state = %d.\n", task_id, ctx[task_id].state);
            } else {
                task_done = false;
            }
            if (!task_done) {
                // re-enqueue the task if not done
                // printf("Re-enqueueing task %d with state %d.\n", task_id, ctx[task_id].state);

                uint64_t t1 = globaltimer_ns();
                queuePushBlocking(task_queue, task_id);
                uint64_t t2 = globaltimer_ns();
                printf("push overhead = %lu ns\n", t2 - t1);
            }
        }
    }
}

__global__ void init_kernel(CUDAContConcurrentQueueData<int>* io_queue_data, int task_num, int task_offset, Context* ctx) {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < task_num) {
        // bool push_ok = queueTryPush(io_queue_data, thread_id + task_offset);
        queuePushBlocking(io_queue_data, thread_id + task_offset);
        // printf("Initialized task %d, push_ok=%d\n", thread_id + task_offset, push_ok);
        ctx[thread_id + task_offset] = Context();
        // printf("ctx[%d] initialized: value=%d, state=%d\n", thread_id + task_offset, ctx[thread_id + task_offset].value, ctx[thread_id + task_offset].state);
    }
}

__global__ void test_kernel(int task_num, int task_offset, 
                            CUDAContConcurrentQueueData<int>* task_queue, 
                            unsigned int* stop_signal, Context* ctx, uint64_t* timer,
                            unsigned int *iterations) {
    while (*iterations < task_num * 2) {
        __shared__ int task_id;
        __shared__ bool pop_ok;
        
        if (threadIdx.x + threadIdx.y + threadIdx.z == 0) {
            // first thread of the block
            pop_ok = queueTryPop(task_queue, task_id);
            // printf("Popped task_id %d by thread block: %d\n", task_id, blockIdx.x);
        }
        __syncthreads();
        if (!pop_ok) {
            continue;
        }

        do_kernel(ctx, task_id);
        __syncthreads();

        __shared__ bool task_done;
        if (threadIdx.x + threadIdx.y + threadIdx.z == 0) {
            // first thread of the block
            if (ctx[task_id].state >= 3) {
                task_done = true;
                // printf("Task %d completed all states, state = %d.\n", task_id, ctx[task_id].state);
            } else {
                task_done = false;
            }
            if (!task_done) {
                uint64_t t1 = globaltimer_ns();
                queuePushBlocking(task_queue, task_id);
                uint64_t t2 = globaltimer_ns();
                // printf("push overhead = %lu ns\n", t2 - t1);
                timer[*iterations] = t2 - t1;
                atomicAdd(iterations, 1u);
            }
        }
    }
}



int main() {
    const int BLOCKS = 16;
    const int THREADS_PER_BLOCK = 256;
    const int QUEUE_CAPACITY = BLOCKS * THREADS_PER_BLOCK * 128;

    // 每个slot存一个int任务ID，作为ctx数组的索引
    CUDAContConcurrentQueue<int> task_queue(QUEUE_CAPACITY);

    unsigned int* persistent_io_stop_signal;
    // cudaMallocManaged((void**)&persistent_io_stop_signal, sizeof(unsigned int));
    cudaHostAlloc((void**)&persistent_io_stop_signal, sizeof(unsigned int), cudaHostAllocMapped);
    *persistent_io_stop_signal = 0;

    const int task_num = BLOCKS * THREADS_PER_BLOCK;
    int task_offset = 0;    // unused
    
    // context
    Context* ctx;
    cudaMalloc((void**)&ctx, sizeof(Context) * task_num);

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    init_kernel<<<BLOCKS, THREADS_PER_BLOCK>>>(task_queue.d_data, task_num, task_offset, ctx);
    printf("Initialized IO queue with %d tasks.\n", task_num);
    cudaDeviceSynchronize();

    // persistent_kernel<<<BLOCKS, THREADS_PER_BLOCK, 0, stream>>>(task_num, task_offset, task_queue.d_data, persistent_io_stop_signal, ctx);
    
    uint64_t* timer;
    cudaMalloc((void**)&timer, sizeof(uint64_t) * task_num * 10);
    unsigned int* iterations;
    cudaMalloc((void**)&iterations, sizeof(unsigned int));
    cudaMemset(iterations, 0, sizeof(unsigned int));
    test_kernel<<<BLOCKS, THREADS_PER_BLOCK, 0, stream>>>(task_num, task_offset, task_queue.d_data, persistent_io_stop_signal, ctx, timer, iterations);
    
    
    sleep(5);
    *persistent_io_stop_signal = 1;
    printf("Signaled persistent kernel to stop.\n");

    // cudaStreamSynchronize(stream);
    // cudaStreamDestroy(stream);
    cudaDeviceSynchronize();

    printf("Completed all tasks.\n");

    printf("queue size after processing: %d\n", task_queue.getCount());

    // for (int i = 0; i < task_num; i++) {
    //     Context h_ctx;
    //     cudaMemcpy(&h_ctx, &ctx[i], sizeof(Context), cudaMemcpyDeviceToHost);
    //     printf("Final state of task %d: value=%d, state=%d, duration=%lu ns\n", i, h_ctx.value, h_ctx.state, h_ctx.timer_end - h_ctx.timer_start);
    // }

    // print timer
    uint64_t h_timer[task_num * 10];
    cudaMemcpy(h_timer, timer, sizeof(uint64_t) * task_num * 10, cudaMemcpyDeviceToHost);
    for (int i = 0; i < task_num * 10; i++) {
        printf("timer[%d] = %lu ns\n", i, h_timer[i]);
    }
    cudaFree(timer);


    cudaFree(ctx);
    cudaFree(persistent_io_stop_signal);

    return 0;
}