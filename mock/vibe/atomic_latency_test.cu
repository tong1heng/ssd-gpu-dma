#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>

__device__ __forceinline__ unsigned long long globaltimer_ns() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

// 测试1: atomicAdd延迟（高竞争）- 多次测量取平均
__global__ void test_atomicAdd_high_contention(unsigned int* counter, uint64_t* latencies) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t total = 0;
    for (int i = 0; i < 100; i++) {
        uint64_t start = globaltimer_ns();
        atomicAdd(counter, 1u);
        uint64_t end = globaltimer_ns();
        total += (end - start);
    }
    latencies[tid] = total / 100;
}

// 测试2: atomicExch延迟（高竞争）- 多次测量取平均
__global__ void test_atomicExch_high_contention(unsigned int* target, uint64_t* latencies) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t total = 0;
    for (int i = 0; i < 100; i++) {
        uint64_t start = globaltimer_ns();
        atomicExch(target, tid + i);
        uint64_t end = globaltimer_ns();
        total += (end - start);
    }
    latencies[tid] = total / 100;
}

// 测试3: 普通内存读取延迟（无竞争）
__global__ void test_normal_read(unsigned int* data, uint64_t* latencies) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t start = globaltimer_ns();
    volatile unsigned int val = data[tid % 1024];  // 读取不同位置，避免竞争
    uint64_t end = globaltimer_ns();
    latencies[tid] = end - start;
    (void)val;  // 避免未使用变量警告
}

// 测试4: volatile读取延迟（低竞争）
__global__ void test_volatile_read(unsigned int* data, uint64_t* latencies) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t start = globaltimer_ns();
    volatile unsigned int* ptr = &data[tid % 1024];
    volatile unsigned int val = *ptr;
    uint64_t end = globaltimer_ns();
    latencies[tid] = end - start;
    (void)val;
}

// 测试5: 分散的atomicAdd（低竞争）- 多次测量取平均
__global__ void test_atomicAdd_low_contention(unsigned int* counters, uint64_t* latencies) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = tid % 1024;  // 分散到不同位置
    uint64_t total = 0;
    for (int i = 0; i < 100; i++) {
        uint64_t start = globaltimer_ns();
        atomicAdd(&counters[idx], 1u);
        uint64_t end = globaltimer_ns();
        total += (end - start);
    }
    latencies[tid] = total / 100;
}

void print_stats(const char* name, uint64_t* latencies, int count) {
    uint64_t min = UINT64_MAX, max = 0, sum = 0;
    for (int i = 0; i < count; i++) {
        if (latencies[i] < min) min = latencies[i];
        if (latencies[i] > max) max = latencies[i];
        sum += latencies[i];
    }
    printf("%s:\n", name);
    printf("  最小: %lu ns (%.2f us)\n", min, min / 1000.0);
    printf("  最大: %lu ns (%.2f us)\n", max, max / 1000.0);
    printf("  平均: %.0f ns (%.2f us)\n", sum / (double)count, (sum / (double)count) / 1000.0);
    printf("\n");
}

int main() {
    const int BLOCKS = 8;
    const int THREADS_PER_BLOCK = 256;
    const int TOTAL_THREADS = BLOCKS * THREADS_PER_BLOCK;
    
    // 分配设备内存
    unsigned int* d_counter;
    unsigned int* d_counters_array;
    unsigned int* d_data;
    uint64_t* d_latencies;
    
    cudaMalloc(&d_counter, sizeof(unsigned int));
    cudaMalloc(&d_counters_array, 1024 * sizeof(unsigned int));
    cudaMalloc(&d_data, 1024 * sizeof(unsigned int));
    cudaMalloc(&d_latencies, TOTAL_THREADS * sizeof(uint64_t));
    
    uint64_t* h_latencies = (uint64_t*)malloc(TOTAL_THREADS * sizeof(uint64_t));
    
    printf("=== GPU显存原子操作延迟分析 ===\n\n");
    
    // 测试1: atomicAdd高竞争
    cudaMemset(d_counter, 0, sizeof(unsigned int));
    test_atomicAdd_high_contention<<<BLOCKS, THREADS_PER_BLOCK>>>(d_counter, d_latencies);
    cudaDeviceSynchronize();
    cudaMemcpy(h_latencies, d_latencies, TOTAL_THREADS * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    print_stats("1. atomicAdd (高竞争 - 所有线程竞争同一位置)", h_latencies, TOTAL_THREADS);
    
    // 测试2: atomicExch高竞争
    cudaMemset(d_counter, 0, sizeof(unsigned int));
    test_atomicExch_high_contention<<<BLOCKS, THREADS_PER_BLOCK>>>(d_counter, d_latencies);
    cudaDeviceSynchronize();
    cudaMemcpy(h_latencies, d_latencies, TOTAL_THREADS * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    print_stats("2. atomicExch (高竞争 - 所有线程竞争同一位置)", h_latencies, TOTAL_THREADS);
    
    // 测试3: 普通读取（无竞争）
    test_normal_read<<<BLOCKS, THREADS_PER_BLOCK>>>(d_data, d_latencies);
    cudaDeviceSynchronize();
    cudaMemcpy(h_latencies, d_latencies, TOTAL_THREADS * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    print_stats("3. 普通内存读取 (无竞争)", h_latencies, TOTAL_THREADS);
    
    // 测试4: volatile读取（低竞争）
    test_volatile_read<<<BLOCKS, THREADS_PER_BLOCK>>>(d_data, d_latencies);
    cudaDeviceSynchronize();
    cudaMemcpy(h_latencies, d_latencies, TOTAL_THREADS * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    print_stats("4. volatile读取 (低竞争)", h_latencies, TOTAL_THREADS);
    
    // 测试5: atomicAdd低竞争（分散到不同位置）
    cudaMemset(d_counters_array, 0, 1024 * sizeof(unsigned int));
    test_atomicAdd_low_contention<<<BLOCKS, THREADS_PER_BLOCK>>>(d_counters_array, d_latencies);
    cudaDeviceSynchronize();
    cudaMemcpy(h_latencies, d_latencies, TOTAL_THREADS * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    print_stats("5. atomicAdd (低竞争 - 分散到不同位置)", h_latencies, TOTAL_THREADS);
    
    printf("=== 结论 ===\n");
    printf("原子操作延迟高的主要原因：\n");
    printf("1. 缓存一致性协议开销：需要保证所有SM的缓存一致性\n");
    printf("2. 串行化：多个线程的原子操作必须串行执行\n");
    printf("3. 显存访问延迟：原子操作需要访问全局显存\n");
    printf("4. 竞争加剧延迟：多个线程竞争同一位置时延迟显著增加\n");
    
    // 清理
    cudaFree(d_counter);
    cudaFree(d_counters_array);
    cudaFree(d_data);
    cudaFree(d_latencies);
    free(h_latencies);
    
    return 0;
}

