#include <cuda_runtime.h>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>
#include <unistd.h>
#include "queue4.cuh"


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



__global__ void init_kernel(CUDAQueue<int>** d_task_queues, int task_num, Context* ctx) {
    if (threadIdx.x + threadIdx.y + threadIdx.z == 0) {
        int block_id = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
        CUDAQueue<int>* io_queue = d_task_queues[block_id];
        // for each thread in this block
        for (int t = 0; t < blockDim.x * blockDim.y * blockDim.z; t++) {
            int task_id = block_id * (blockDim.x * blockDim.y * blockDim.z) + t;
            bool push_ok = queuePush<int>(io_queue, task_id);
            printf("Initialized task %d, push_ok=%d\n", task_id, push_ok);
            ctx[task_id] = Context();
            // printf("ctx[%d] initialized: value=%d, state=%d\n", task_id, ctx[task_id].value, ctx[task_id].state);
        }
    }
}

__global__ void persistent_kernel(int task_num, 
                                  CUDAQueue<int>** d_task_queues, 
                                  Context* ctx,
                                  uint64_t* timer,
                                  unsigned int* iterations) {
    while (atomicAdd(iterations, 0) < task_num) {
        __shared__ int task_id;
        __shared__ bool pop_ok;
        
        if (threadIdx.x + threadIdx.y + threadIdx.z == 0) { // first thread of the block
            // find the task queue according to the block id
            int block_id = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
            CUDAQueue<int>* task_queue = d_task_queues[block_id];
            pop_ok = queuePop(task_queue, task_id);
            // printf("Popped task_id %d by thread block: %d\n", task_id, blockIdx.x);
        }
        __syncthreads();
        if (!pop_ok) {
            printf("warning: no task to pop for block %d\n", 
                   blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y);
            continue;
        }

        do_kernel(ctx, task_id);
        __syncthreads();

        if (threadIdx.x + threadIdx.y + threadIdx.z == 0) { // first thread of the block
            if (ctx[task_id].state >= 3) { // task done
                atomicAdd(iterations, 1);
                // printf("Task %d completed all states, state = %d.\n", task_id, ctx[task_id].state);
            } else {
                queuePush(d_task_queues[blockIdx.x], task_id);
            }
        }
    }
}



int main() {
    const int BLOCKS = 16;
    const int THREADS_PER_BLOCK = 256;
    const int QUEUE_CAPACITY = BLOCKS * THREADS_PER_BLOCK * 128;

    std::vector<CUDAQueue<int>*> h_task_queues;
    for (int i = 0; i < BLOCKS; i++) {
        CUDAQueue<int>* d_task_queue;
        cudaMalloc((void**)&d_task_queue, sizeof(CUDAQueue<int>));
        queueInit<int>(d_task_queue, QUEUE_CAPACITY / BLOCKS);
        h_task_queues.push_back(d_task_queue);
    }
    CUDAQueue<int>** d_task_queues;
    cudaMalloc((void**)&d_task_queues, sizeof(CUDAQueue<int>*) * BLOCKS);
    cudaMemcpy(d_task_queues, h_task_queues.data(), sizeof(CUDAQueue<int>*) * BLOCKS, cudaMemcpyHostToDevice);

    const int task_num = BLOCKS * THREADS_PER_BLOCK;
    
    // context
    Context* ctx;
    cudaMalloc((void**)&ctx, sizeof(Context) * task_num);

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    init_kernel<<<BLOCKS, THREADS_PER_BLOCK>>>(d_task_queues, task_num, ctx);
    printf("Initialized IO queue with %d tasks.\n", task_num);
    cudaDeviceSynchronize();

    uint64_t* timer;
    cudaMalloc((void**)&timer, sizeof(uint64_t) * task_num * 2);
    unsigned int* iterations;
    cudaMalloc((void**)&iterations, sizeof(unsigned int));
    cudaMemset(iterations, 0, sizeof(unsigned int));
    persistent_kernel<<<BLOCKS, THREADS_PER_BLOCK, 0, stream>>>(task_num, d_task_queues, ctx, timer, iterations);
    cudaDeviceSynchronize();

    printf("Completed all tasks.\n");

    for (int i = 0; i < task_num; i++) {
        Context h_ctx;
        cudaMemcpy(&h_ctx, &ctx[i], sizeof(Context), cudaMemcpyDeviceToHost);
        printf("Final state of task %d: value=%d, state=%d, duration=%lu ns\n", i, h_ctx.value, h_ctx.state, h_ctx.timer_end - h_ctx.timer_start);
    }

    // print timer
    // uint64_t h_timer[task_num * 2];
    // cudaMemcpy(h_timer, timer, sizeof(uint64_t) * task_num * 2, cudaMemcpyDeviceToHost);
    // for (int i = 0; i < task_num * 2; i++) {
    //     printf("timer[%d] = %lu ns\n", i, h_timer[i]);
    // }

    cudaFree(timer);
    cudaFree(ctx);

    return 0;
}