#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <unistd.h>
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

__device__ __forceinline__ dim3 get_3d_idx(int idx, dim3 dim) {
  dim3 result;
  result.x = idx % dim.x;
  result.y = idx / dim.x % dim.y;
  result.z = idx / (dim.x * dim.y);
  return result;
}

__device__ void io_kernel(int* value) {
    // perform io submission task
    // ...
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    atomicAdd(value, 1);
    printf("IO Kernel executed by thread %d for value %d\n", thread_id, *value);
}

__device__ void polling_kernel(int* value) {
    // perform polling task
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    atomicAdd(value, 1);
    printf("Polling Kernel executed by thread %d for value %d\n", thread_id, *value);
}

__device__ void compute_kernel(int* value) {
    // perform compute task
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    atomicAdd(value, 1);
    printf("Compute Kernel executed by thread %d for value %d\n", thread_id, *value);
}

// submit io task
__global__ void persistent_io(int task_num, int task_offset, CUDAContConcurrentQueueData<int>* io_queue_data, CUDAContConcurrentQueueData<int>* polling_queue_data, unsigned int* stop_signal) {
    unsigned int warp_idx = threadIdx.x % 32;
    unsigned int stop_sig;
    while (true) {
        if (warp_idx == 0) {
            stop_sig = *((volatile unsigned int *)stop_signal);
        }
        stop_sig = __shfl_sync(0xFFFFFFFF, stop_sig, 0);
        if (stop_sig) {
            printf("Persistent IO kernel received stop signal.\n");
            break;
        }

        __shared__ int value;
        __shared__ bool pop_ok;
        if (threadIdx.x + threadIdx.y + threadIdx.z == 0) {
            // first thread of the block
            pop_ok = queuePop(io_queue_data, value);
        }

        __syncthreads();
        if (!pop_ok) {
            continue;
        }

        io_kernel(&value);
        if (threadIdx.x + threadIdx.y + threadIdx.z == 0) {
            // first thread of the block
            queuePush(polling_queue_data, value);
        }
    }
}

// polling io task
__global__ void persistent_polling(int task_num, int task_offset, CUDAContConcurrentQueueData<int>* polling_queue_data, CUDAContConcurrentQueueData<int>* compute_queue_data, unsigned int* stop_signal) {
    unsigned int warp_idx = threadIdx.x % 32;
    unsigned int stop_sig;
    while (true) {
        if (warp_idx == 0) {
            stop_sig = *((volatile unsigned int *)stop_signal);
        }
        stop_sig = __shfl_sync(0xFFFFFFFF, stop_sig, 0);
        if (stop_sig) {
            printf("Persistent polling kernel received stop signal.\n");
            break;
        }

        __shared__ int value;
        __shared__ bool pop_ok;
        if (threadIdx.x + threadIdx.y + threadIdx.z == 0) {
            // first thread of the block
            pop_ok = queuePop(polling_queue_data, value);
            // if (pop_ok) {
            //     printf("Polling got value %d\n", value);
            // } else {
            //     // printf("Polling queue empty.\n");
            // }
        }

        __syncthreads();
        if (!pop_ok) {
            continue;
        }

        polling_kernel(&value);
        if (threadIdx.x + threadIdx.y + threadIdx.z == 0) {
            // first thread of the block
            queuePush(compute_queue_data, value);
        }
    }
}

// compute task
__global__ void persistent_compute(int task_num, int task_offset, CUDAContConcurrentQueueData<int>* compute_queue_data, unsigned int* stop_signal) {
    unsigned int warp_idx = threadIdx.x % 32;
    unsigned int stop_sig;
    while (true) {
        if (warp_idx == 0) {
            stop_sig = *((volatile unsigned int *)stop_signal);
        }
        stop_sig = __shfl_sync(0xFFFFFFFF, stop_sig, 0);
        if (stop_sig) {
            printf("Persistent compute kernel received stop signal.\n");
            break;
        }

        __shared__ int value;
        __shared__ bool pop_ok;
        if (threadIdx.x + threadIdx.y + threadIdx.z == 0) {
            // first thread of the block
            pop_ok = queuePop(compute_queue_data, value);
        }

        __syncthreads();
        if (!pop_ok) {
            continue;
        }

        compute_kernel(&value);
    }
}

__global__ void init_kernel(CUDAContConcurrentQueueData<int>* io_queue_data, int task_num, int task_offset) {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < task_num) {
        queuePush(io_queue_data, thread_id + task_offset);
        printf("Initialized task %d in IO queue.\n", thread_id + task_offset);
    }
}



int main() {
    const int BLOCKS = 2;
    const int THREADS_PER_BLOCK =32;
    const int QUEUE_CAPACITY = BLOCKS * THREADS_PER_BLOCK;

    CUDAContConcurrentQueue<int> io_queue(QUEUE_CAPACITY);
    CUDAContConcurrentQueue<int> polling_queue(QUEUE_CAPACITY);
    CUDAContConcurrentQueue<int> compute_queue(QUEUE_CAPACITY);

    unsigned int* persistent_io_stop_signal;
    // cudaMallocManaged((void**)&persistent_io_stop_signal, sizeof(unsigned int));
    cudaHostAlloc((void**)&persistent_io_stop_signal, sizeof(unsigned int), cudaHostAllocMapped);
    *persistent_io_stop_signal = 0;
    unsigned int* persistent_polling_stop_signal;
    // cudaMallocManaged((void**)&persistent_polling_stop_signal, sizeof(unsigned int));
    cudaHostAlloc((void**)&persistent_polling_stop_signal, sizeof(unsigned int), cudaHostAllocMapped);
    *persistent_polling_stop_signal = 0;
    unsigned int* persistent_compute_stop_signal;
    // cudaMallocManaged((void**)&persistent_compute_stop_signal, sizeof(unsigned int));
    cudaHostAlloc((void**)&persistent_compute_stop_signal, sizeof(unsigned int), cudaHostAllocMapped);
    *persistent_compute_stop_signal = 0;

    const int task_num = QUEUE_CAPACITY;
    int task_offset = 0;

    cudaStream_t stream_a, stream_b, stream_c;
    cudaStreamCreate(&stream_a);
    cudaStreamCreate(&stream_b);
    cudaStreamCreate(&stream_c);

    init_kernel<<<BLOCKS, THREADS_PER_BLOCK>>>(io_queue.d_data, task_num, task_offset);
    printf("Initialized IO queue with %d tasks.\n", task_num);
    cudaDeviceSynchronize();

    persistent_io<<<BLOCKS, THREADS_PER_BLOCK, 0, stream_a>>>(task_num, task_offset, io_queue.d_data, polling_queue.d_data, persistent_io_stop_signal);
    persistent_polling<<<BLOCKS, THREADS_PER_BLOCK, 0, stream_b>>>(task_num, task_offset, polling_queue.d_data, compute_queue.d_data, persistent_polling_stop_signal);
    persistent_compute<<<BLOCKS, THREADS_PER_BLOCK, 0, stream_c>>>(task_num, task_offset, compute_queue.d_data, persistent_compute_stop_signal);


    sleep(5);

    *persistent_io_stop_signal = 1;
    *persistent_polling_stop_signal = 1;
    *persistent_compute_stop_signal = 1;
    printf("Signaled all persistent kernels to stop.\n");

    // cudaStreamSynchronize(stream_a);
    // cudaStreamSynchronize(stream_b);
    // cudaStreamSynchronize(stream_c);
    // cudaStreamDestroy(stream_a);
    // cudaStreamDestroy(stream_b);
    // cudaStreamDestroy(stream_c);
    cudaDeviceSynchronize();

    printf("Completed all tasks.\n");

    return 0;
}