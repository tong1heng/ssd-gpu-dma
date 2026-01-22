#include <stdio.h>
#include <cuda_runtime.h>
#include <cstdint>

#define NUM_BLOCKS 16    // 设置线程块数目
#define NUM_THREADS 256  // 每个线程块中的线程数

__device__ __forceinline__ unsigned long long globaltimer_ns() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

// 计算atomicAdd延迟的核函数
__global__ void atomicAddTest(int *d_result) {
    // 每个线程向同一个整数变量执行atomicAdd操作
    for (int i = 0; i < 1000; i++) {  // 循环多次以增加延迟
        uint64_t start = globaltimer_ns();
        atomicAdd(d_result, 1);  // 使用atomicAdd对d_result加1
        __threadfence();
        uint64_t end = globaltimer_ns();
        printf("Block %d Thread %d atomicAdd latency: %lu ns\n", 
               blockIdx.x, threadIdx.x, end - start);
    }

}

int main() {
    int h_result = 0;    // 主机上的结果变量
    int *d_result;       // 设备上的结果变量指针

    // 在设备上分配内存
    cudaMalloc((void **)&d_result, sizeof(int));

    // 初始化设备变量
    cudaMemset(d_result, 0, sizeof(int));

    // 启动内核：多个线程块执行atomicAdd
    atomicAddTest<<<NUM_BLOCKS, NUM_THREADS>>>(d_result);

    // 检查核函数是否执行成功
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // 等待所有线程完成
    cudaDeviceSynchronize();

    // 从设备复制结果回主机
    cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);

    printf("Result: %d\n", h_result);

    // 释放设备内存
    cudaFree(d_result);

    return 0;
}