#include <cstddef>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdio>
#include <cassert>
#include <cstdint>
#include <sys/time.h>

#define N 16
#define ITER 1

__device__ __forceinline__ unsigned long long globaltimer_ns() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

__device__ __forceinline__ void busy_wait_ns(unsigned long long target_ns) {
    unsigned long long start = globaltimer_ns();
    while (globaltimer_ns() - start < target_ns) {
        asm volatile("");
    }
}

__global__ void testKernel(const float* A, const float* B, float* C, int n, int iter)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    tid = tid % 32;
    int wid = tid / 32;
    int total = n * n;

    // I/O wait simulation
    __nanosleep(10*1000);

    A += wid * n * n;
    B += wid * n * n;
    C += wid * n * n;
    auto start = globaltimer_ns();
    for (int idx = tid; idx < total; idx += blockDim.x * gridDim.x) {
        int row = idx / n;
        int col = idx % n;
        if (row < n && col < n) {
            for (int it = 0; it < iter; ++it) {
                float sum = 0.0f;
                for (int k = 0; k < n; ++k) {
                    sum += A[row * n + k] * B[k * n + col];
                }
                C[row * n + col] = sum;
            }
        }
    }
    // __syncthreads();
    auto end = globaltimer_ns();
    printf("Thread %d completed in %llu ns\n", tid, end - start);
}

int main() {
    cudaDeviceProp properties;
    if (cudaGetDeviceProperties(&properties, 0) != cudaSuccess)
    {
        fprintf(stderr, "Failed to get CUDA device properties\n");
        return 1;
    }

    auto err = cudaSetDevice(0);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to set CUDA device\n");
        return 1;
    }

    size_t numWarps = 2;
    size_t numThreads = numWarps * 32;

    // 初始化矩阵
    size_t bytes = N * N * sizeof(float) * numWarps;
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);

    for (int i = 0; i < N * N; ++i) {
        h_A[i] = 1.0f; // 可自定义
        h_B[i] = 2.0f;
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    testKernel<<<1, numThreads>>>(d_A, d_B, d_C, N, ITER);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float elapsed_ms = 0;
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    printf("Kernel time: %.3f ms\n", elapsed_ms);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);


    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}