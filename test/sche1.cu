#include <cstddef>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdio>
#include <cassert>
#include <cstdint>

#define N    16
#define ITER 1

__device__ __forceinline__ unsigned long long globaltimer_ns() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

__global__ void testKernel(const float* A, const float* B, float* C,
                           int n, int iter, int numWarps)
{
    // 仅启动一个 block，按 threadIdx.x 计算 lane/warp
    const int t    = threadIdx.x;
    const int lane = t & 31;     // 0..31
    const int wid  = t >> 5;     // 0..(numWarps-1)（每 32 线程一个 warp）

    if (wid >= numWarps) return; // 多开的线程（非 32 倍数时）直接退出
    const int total = n * n;

    // 每个 warp 处理自己那一张矩阵（第 wid 张）
    const float* Aw = A + wid * total;
    const float* Bw = B + wid * total;
    float*       Cw = C + wid * total;


    // 可选的 I/O wait 模拟（只让 lane0 执行，避免所有线程都 sleep）
    if (lane == 0) {
        __nanosleep(10 * 1000);
    }
    __syncwarp(); // 同步该 warp


    unsigned long long start = globaltimer_ns();

    // 该 warp 的 32 个线程并行遍历当前矩阵的所有元素
    for (int idx = lane; idx < total; idx += 32) {
        const int row = idx / n;
        const int col = idx % n;

        float sum = 0.0f;
        // 朴素 GEMM：C = A * B
        #pragma unroll
        for (int k = 0; k < n; ++k) {
            sum += Aw[row * n + k] * Bw[k * n + col];
        }
        // ITER 次重复计算（若需“加速计时”，也可以把 sum 写回多次）
        for (int it = 0; it < iter; ++it) {
            Cw[row * n + col] = sum;
        }
    }

    unsigned long long end = globaltimer_ns();
    if (lane == 0) {
        printf("[warp %d] completed in %llu ns\n", wid, end - start);
    }
}

int main() {
    int dev = 0;
    cudaDeviceProp properties;
    if (cudaGetDeviceProperties(&properties, dev) != cudaSuccess) {
        fprintf(stderr, "Failed to get CUDA device properties\n");
        return 1;
    }
    if (cudaSetDevice(dev) != cudaSuccess) {
        fprintf(stderr, "Failed to set CUDA device\n");
        return 1;
    }

    // ---- 可调并发度：numThreads 必须是 32 的倍数（每 32 线程一个 warp）----
    int numWarps   = 1;
    int numThreads = numWarps * 32;  // 单 block 的线程数
    assert(numThreads % 32 == 0);

    // ---- 准备 numWarps 份矩阵 ----
    const size_t perMatElems = N * N;
    const size_t bytesAll = perMatElems * sizeof(float) * numWarps;

    float *h_A = (float*)malloc(bytesAll);
    float *h_B = (float*)malloc(bytesAll);
    float *h_C = (float*)malloc(bytesAll);

    // 为每个 warp 填一份（这里所有矩阵内容相同，也可以按 warp 定制）
    for (int w = 0; w < numWarps; ++w) {
        float* Ap = h_A + w * perMatElems;
        float* Bp = h_B + w * perMatElems;
        for (int i = 0; i < (int)perMatElems; ++i) {
            Ap[i] = 1.0f;
            Bp[i] = 2.0f;
        }
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytesAll);
    cudaMalloc(&d_B, bytesAll);
    cudaMalloc(&d_C, bytesAll);

    cudaMemcpy(d_A, h_A, bytesAll, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytesAll, cudaMemcpyHostToDevice);

    // ---- 启动：只启动一个 block，线程数 = numThreads ----
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    testKernel<<<1, numThreads>>>(d_A, d_B, d_C, N, ITER, numWarps);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float elapsed_ms = 0.f;
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    printf("Kernel time: %.3f ms\n", elapsed_ms);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(h_C, d_C, bytesAll, cudaMemcpyDeviceToHost);

    // // 简单校验：由于 A=1，B=2，C 元素应为 2*N
    // bool ok = true;
    // for (int w = 0; w < numWarps; ++w) {
    //     float* Cp = h_C + w * perMatElems;
    //     for (int i = 0; i < (int)perMatElems; ++i) {
    //         if (fabs(Cp[i] - (2.0f * N)) > 1e-3f) {
    //             ok = false; break;
    //         }
    //     }
    //     if (!ok) break;
    // }
    // printf("Check: %s\n", ok ? "PASS" : "FAIL");

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    return 0;
}
