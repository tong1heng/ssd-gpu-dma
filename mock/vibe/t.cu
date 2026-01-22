#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>

// 检查 CUDA 错误的辅助函数
#define CHECK_CUDA_ERROR(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA Error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// Device 核函数：直接访问 Host 固定内存
__global__ void modify_pinned_memory(int* d_ptr, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        d_ptr[idx] *= 2; // Device 直接修改 Host 固定内存的数据
    }
}

int main() {
    const int size = 5;
    int* h_pinned = nullptr;
    int* d_pinned_ptr = nullptr; // 存储设备端可访问的指针

    // 1. 分配主机固定内存（Pinned Memory）
    CHECK_CUDA_ERROR(cudaHostAlloc((void**)&h_pinned, size * sizeof(int), cudaHostAllocMapped));
    printf("h_pinned 地址: %p\n", h_pinned);
    // 2. Host 初始化固定内存数据
    for (int i = 0; i < size; i++) {
        h_pinned[i] = i + 1; // 初始值：1,2,3,4,5
    }
    printf("初始化后的数据：");
    for (int i = 0; i < size; i++) printf("%d ", h_pinned[i]);
    printf("\n");

    // 3. 核心操作：获取固定内存对应的设备端指针
    CHECK_CUDA_ERROR(cudaHostGetDevicePointer((void**)&d_pinned_ptr, h_pinned, 0));
    printf("d_pinned_ptr 地址: %p\n", d_pinned_ptr);

    // 4. 启动核函数：Device 直接操作 Host 固定内存
    modify_pinned_memory<<<1, size>>>(h_pinned, size);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize()); // 等待核函数执行完成

    // 5. Host 直接读取修改后的数据（无需拷贝）
    printf("Device 修改后的数据：");
    for (int i = 0; i < size; i++) printf("%d ", h_pinned[i]); // 输出：2,4,6,8,10
    printf("\n");

    // 6. 释放资源
    CHECK_CUDA_ERROR(cudaFreeHost(h_pinned));
    return 0;
}