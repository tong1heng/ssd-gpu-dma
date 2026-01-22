#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include "queue.cuh"


enum nvm_io_command_set {
    NVM_IO_FLUSH,
    NVM_IO_WRITE,
    NVM_IO_READ,
    NVM_IO_WRITE_ZEROES
};

struct IoArgs {
    // nvm_cmd_header
    nvm_io_command_set opcode;
    uint16_t cid;       // command identifier, usually thread id
    uint32_t ns_id;
    // nvm cmd data ptr
    uint64_t prp1;
    uint64_t prp2;
    // nvm cmd rw blks
    uint64_t starting_lba;
    uint64_t n_blocks;
};

struct PollingArgs {
    void* cq;
    uint16_t cid;
};

struct ComputeArgs {
    int iteratiions;
};


__device__ IoArgs* d_io_task_queue;
__device__ PollingArgs* d_polling_task_queue;
__device__ ComputeArgs* d_compute_task_queue;


__device__ __forceinline__ dim3 get_3d_idx(int idx, dim3 dim) {
  dim3 result;
  result.x = idx % dim.x;
  result.y = idx / dim.x % dim.y;
  result.z = idx / (dim.x * dim.y);
  return result;
}

__device__ void io_kernel(dim3 task_idx, dim3 thread_idx) {
    // perform io task
    // ...

    // TODO: push to polling task queue
}

__device__ void polling_kernel(dim3 task_idx, dim3 thread_idx) {
    // perform polling task
    
    // push to compute task queue

}

__device__ void compute_kernel(dim3 task_idx, dim3 thread_idx) {
    // perform compute task

    // submit io task

    // push to polling task queue
    
}

// submit io task
__global__ void persistent_io(int task_num, int task_offset, int* task_slot) {
    dim3 task_dim = dim3(200,1,1);
    dim3 thread_idx = dim3(threadIdx.x, threadIdx.y, threadIdx.z);
    __shared__ int idx[1];
    while (true) {
        if (threadIdx.x + threadIdx.y + threadIdx.z == 0) {
            // first thread of the block
            int temp = atomicAdd(task_slot, 1);
            idx[0] = temp + task_offset;
        }
        __syncthreads();
        if (idx[0] >= task_num) return;
        dim3 task_idx = get_3d_idx(idx[0], task_dim);
        io_kernel(task_idx, thread_idx);

    }
}

// polling io task
__global__ void persistent_polling(int task_num, int task_offset, int* task_slot) {
    dim3 task_dim = dim3(200,1,1);
    dim3 thread_idx = dim3(threadIdx.x, threadIdx.y, threadIdx.z);
    __shared__ int idx[1];
    while (true) {
        if (threadIdx.x + threadIdx.y + threadIdx.z == 0) {
            // first thread of the block
            int temp = atomicAdd(task_slot, 1);
            idx[0] = temp + task_offset;
        }
        __syncthreads();
        if (idx[0] >= task_num) return;
        dim3 task_idx = get_3d_idx(idx[0], task_dim);
        polling_kernel(task_idx, thread_idx);
    }
}

// compute task
__global__ void persistent_compute(int task_num, int task_offset, int* task_slot) {
    dim3 task_dim = dim3(200,1,1);
    dim3 thread_idx = dim3(threadIdx.x, threadIdx.y, threadIdx.z);
    __shared__ int idx[1];
    while (true) {
        if (threadIdx.x + threadIdx.y + threadIdx.z == 0) {
            // first thread of the block
            int temp = atomicAdd(task_slot, 1);
            idx[0] = temp + task_offset;
        }
        __syncthreads();
        if (idx[0] >= task_num) return;
        dim3 task_idx = get_3d_idx(idx[0], task_dim);
        compute_kernel(task_idx, thread_idx);
    }
}



int main() {
    const int BLOCKS = 16;
    const int THREADS_PER_BLOCK = 64;
    const int QUEUE_CAPACITY = BLOCKS * THREADS_PER_BLOCK;
    
    CUDAContConcurrentQueue<int> queue(QUEUE_CAPACITY);

    const int task_num = QUEUE_CAPACITY;
    int task_offset = 0;

    int io_task_slot = 0;
    int polling_task_slot = 0;
    int compute_task_slot = 0;



    cudaStream_t stream_a, stream_b, stream_c;
    cudaStreamCreate(&stream_a);
    cudaStreamCreate(&stream_b);
    cudaStreamCreate(&stream_c);

    persistent_io<<<16, 256, 0, stream_a>>>(task_num, task_offset, &io_task_slot);
    persistent_polling<<<16,256, 0, stream_b>>>(task_num, task_offset, &polling_task_slot);
    persistent_compute<<<16, 256, 0, stream_c>>>(task_num, task_offset, &compute_task_slot);

    cudaDeviceSynchronize();


    return 0;
}