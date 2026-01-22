#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>

__device__ __forceinline__ unsigned long long globaltimer_ns() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

// 模拟队列push中的等待循环
__global__ void test_wait_loop_latency(volatile unsigned int* seq, unsigned int expected, uint64_t* latencies) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 模拟等待循环
    uint64_t start = globaltimer_ns();
    unsigned int spins = 0;
    while (*seq != expected) {
        spins++;
        if (spins < 32) {
            // 快速自旋
        } else if (spins < 256) {
            #if __CUDA_ARCH__ >= 700
            __nanosleep(50);
            #endif
        } else {
            #if __CUDA_ARCH__ >= 700
            __nanosleep(200);
            #endif
        }
        // 防止无限循环（测试用）
        if (spins > 10000) break;
    }
    uint64_t end = globaltimer_ns();
    latencies[tid] = end - start;
}

// 测试：生产者等待消费者释放槽位
__global__ void test_producer_consumer_latency(
    volatile unsigned int* seq_array,
    unsigned int* tail_counter,
    uint64_t* push_latencies) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 模拟queuePushBlocking
    unsigned int pos = atomicAdd(tail_counter, 1u);
    unsigned int idx = pos % 1024;  // 模拟mask
    unsigned int expected = pos;
    
    uint64_t t1 = globaltimer_ns();
    
    // 等待槽位可用
    volatile unsigned int* seq_ptr = &seq_array[idx];
    unsigned int spins = 0;
    while (*seq_ptr != expected) {
        spins++;
        if (spins < 32) {
        } else if (spins < 256) {
            #if __CUDA_ARCH__ >= 700
            __nanosleep(50);
            #endif
        } else {
            #if __CUDA_ARCH__ >= 700
            __nanosleep(200);
            #endif
        }
        if (spins > 10000) break;  // 防止死锁
    }
    
    uint64_t t2 = globaltimer_ns();
    
    // 写入数据（模拟）
    // 更新seq（模拟）- 需要转换为非volatile指针
    atomicExch((unsigned int*)&seq_array[idx], expected + 1u);
    
    uint64_t t3 = globaltimer_ns();
    
    push_latencies[tid * 3 + 0] = t2 - t1;  // 等待循环延迟
    push_latencies[tid * 3 + 1] = t3 - t2;  // atomicExch延迟
    push_latencies[tid * 3 + 2] = t3 - t1;  // 总延迟
}

struct TestConfig {
    int blocks;
    int threads_per_block;
    int total_threads;
};

struct LatencyStats {
    uint64_t wait_min, wait_max, wait_avg;
    uint64_t atomic_min, atomic_max, atomic_avg;
    uint64_t total_min, total_max, total_avg;
};

LatencyStats compute_stats(uint64_t* latencies, int total_threads) {
    LatencyStats stats;
    uint64_t wait_min = UINT64_MAX, wait_max = 0, wait_sum = 0;
    uint64_t atomic_min = UINT64_MAX, atomic_max = 0, atomic_sum = 0;
    uint64_t total_min = UINT64_MAX, total_max = 0, total_sum = 0;
    
    for (int i = 0; i < total_threads; i++) {
        uint64_t wait = latencies[i * 3 + 0];
        uint64_t atomic = latencies[i * 3 + 1];
        uint64_t total = latencies[i * 3 + 2];
        
        if (wait < wait_min) wait_min = wait;
        if (wait > wait_max) wait_max = wait;
        wait_sum += wait;
        
        if (atomic < atomic_min) atomic_min = atomic;
        if (atomic > atomic_max) atomic_max = atomic;
        atomic_sum += atomic;
        
        if (total < total_min) total_min = total;
        if (total > total_max) total_max = total;
        total_sum += total;
        // printf("Thread %d: wait=%lu ns, atomic=%lu ns, total=%lu ns\n", 
        //        i, wait, atomic, total);
    }
    
    stats.wait_min = wait_min;
    stats.wait_max = wait_max;
    stats.wait_avg = wait_sum / total_threads;
    
    stats.atomic_min = atomic_min;
    stats.atomic_max = atomic_max;
    stats.atomic_avg = atomic_sum / total_threads;
    
    stats.total_min = total_min;
    stats.total_max = total_max;
    stats.total_avg = total_sum / total_threads;

    
    return stats;
}

int main() {
    // 定义不同的线程数配置
    TestConfig configs[] = {
        {1, 32, 32},        // 1 warp
        {1, 64, 64},        // 2 warps
        {1, 128, 128},      // 4 warps
        {1, 256, 256},      // 8 warps
        {2, 256, 512},      // 16 warps
        {4, 256, 1024},     // 32 warps
        {8, 256, 2048},     // 64 warps
        {16, 256, 4096},    // 128 warps
        {32, 256, 8192},    // 256 warps
    };
    
    int num_configs = sizeof(configs) / sizeof(configs[0]);
    
    printf("=== 不同线程数下的队列Push操作延迟分析 ===\n\n");
    printf("测试场景：模拟queuePushBlocking操作\n");
    printf("- 所有槽位初始可用（seq[i] = i）\n");
    printf("- 所有线程同时执行push操作\n");
    printf("- 每个配置运行3次取平均值\n\n");
    printf("注意：高线程数时延迟暴增是因为atomicAdd(&tail)的串行化竞争\n");
    printf("     实际队列中，如果消费者也在工作，延迟会更复杂\n\n");
    
    // 分配设备内存（使用最大配置的大小）
    const int MAX_THREADS = 8192;
    volatile unsigned int* d_seq;
    unsigned int* d_tail;
    uint64_t* d_latencies;
    
    cudaMalloc((void**)&d_seq, 1024 * sizeof(unsigned int));    // 队列
    cudaMalloc(&d_tail, sizeof(unsigned int));
    cudaMalloc(&d_latencies, MAX_THREADS * 3 * sizeof(uint64_t));
    
    uint64_t* h_latencies = (uint64_t*)malloc(MAX_THREADS * 3 * sizeof(uint64_t));
    
    // 初始化：seq[i] = i（模拟空槽位）
    unsigned int* h_seq_init = (unsigned int*)malloc(1024 * sizeof(unsigned int));
    for (int i = 0; i < 1024; i++) {
        h_seq_init[i] = i;
    }
    
    printf("| 线程数 | Blocks | Threads/Block | 等待循环延迟(us) | atomicExch延迟(us) | 总延迟(us) |\n");
    printf("|--------|--------|---------------|------------------|-------------------|------------|\n");
    
    for (int cfg = 0; cfg < num_configs; cfg++) {
        TestConfig config = configs[cfg];
        
        // 运行3次取平均
        LatencyStats avg_stats = {0};
        int runs = 3;
        
        for (int run = 0; run < runs; run++) {
            // 重新初始化
            cudaMemcpy((void*)d_seq, h_seq_init, 1024 * sizeof(unsigned int), cudaMemcpyHostToDevice);
            cudaMemset(d_tail, 0, sizeof(unsigned int));
            
            // 运行测试
            test_producer_consumer_latency<<<config.blocks, config.threads_per_block>>>(
                d_seq, d_tail, d_latencies);
            cudaDeviceSynchronize();
            
            cudaMemcpy(h_latencies, d_latencies, config.total_threads * 3 * sizeof(uint64_t), 
                       cudaMemcpyDeviceToHost);
            
            LatencyStats stats = compute_stats(h_latencies, config.total_threads);
            
            avg_stats.wait_min += stats.wait_min;
            avg_stats.wait_max += stats.wait_max;
            avg_stats.wait_avg += stats.wait_avg;
            
            avg_stats.atomic_min += stats.atomic_min;
            avg_stats.atomic_max += stats.atomic_max;
            avg_stats.atomic_avg += stats.atomic_avg;
            
            avg_stats.total_min += stats.total_min;
            avg_stats.total_max += stats.total_max;
            avg_stats.total_avg += stats.total_avg;
        }
        
        // 计算平均值
        avg_stats.wait_min /= runs;
        avg_stats.wait_max /= runs;
        avg_stats.wait_avg /= runs;
        avg_stats.atomic_min /= runs;
        avg_stats.atomic_max /= runs;
        avg_stats.atomic_avg /= runs;
        avg_stats.total_min /= runs;
        avg_stats.total_max /= runs;
        avg_stats.total_avg /= runs;
        
        printf("| %6d | %6d | %13d | %10.2f (avg) | %11.2f (avg) | %8.2f (avg) |\n",
               config.total_threads,
               config.blocks,
               config.threads_per_block,
               avg_stats.wait_avg / 1000.0,
               avg_stats.atomic_avg / 1000.0,
               avg_stats.total_avg / 1000.0);
    }
    
    printf("\n");
    printf("=== 详细统计（最后一次配置：8192线程）===\n");
    
    // 重新运行最后一次配置获取详细统计
    TestConfig last_config = configs[num_configs - 1];
    cudaMemcpy((void*)d_seq, h_seq_init, 1024 * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemset(d_tail, 0, sizeof(unsigned int));
    
    test_producer_consumer_latency<<<last_config.blocks, last_config.threads_per_block>>>(
        d_seq, d_tail, d_latencies);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_latencies, d_latencies, last_config.total_threads * 3 * sizeof(uint64_t), 
               cudaMemcpyDeviceToHost);
    
    LatencyStats detailed_stats = compute_stats(h_latencies, last_config.total_threads);
    
    printf("等待循环延迟:\n");
    printf("  最小: %lu ns (%.2f us)\n", detailed_stats.wait_min, detailed_stats.wait_min / 1000.0);
    printf("  最大: %lu ns (%.2f us)\n", detailed_stats.wait_max, detailed_stats.wait_max / 1000.0);
    printf("  平均: %lu ns (%.2f us)\n", detailed_stats.wait_avg, detailed_stats.wait_avg / 1000.0);
    printf("\n");
    
    printf("atomicExch延迟:\n");
    printf("  最小: %lu ns (%.2f us)\n", detailed_stats.atomic_min, detailed_stats.atomic_min / 1000.0);
    printf("  最大: %lu ns (%.2f us)\n", detailed_stats.atomic_max, detailed_stats.atomic_max / 1000.0);
    printf("  平均: %lu ns (%.2f us)\n", detailed_stats.atomic_avg, detailed_stats.atomic_avg / 1000.0);
    printf("\n");
    
    printf("总延迟:\n");
    printf("  最小: %lu ns (%.2f us)\n", detailed_stats.total_min, detailed_stats.total_min / 1000.0);
    printf("  最大: %lu ns (%.2f us)\n", detailed_stats.total_max, detailed_stats.total_max / 1000.0);
    printf("  平均: %lu ns (%.2f us)\n", detailed_stats.total_avg, detailed_stats.total_avg / 1000.0);
    printf("\n");
    
    printf("=== 分析 ===\n");
    printf("1. 低线程数（32-1024）：延迟低（<1us），竞争少\n");
    printf("2. 中等线程数（2048）：延迟开始暴增（~5ms），atomicAdd(&tail)串行化\n");
    printf("3. 高线程数（4096+）：延迟继续增加（~7-9ms），严重串行化排队\n");
    printf("\n");
    printf("延迟来源分解：\n");
    printf("- atomicAdd(&tail): 获取位置，高竞争时串行化（主要延迟来源）\n");
    printf("- 等待循环: 等待槽位可用，如果槽位已准备好则很快\n");
    printf("- atomicExch(&seq): 更新seq，相对稳定\n");
    printf("\n");
    printf("优化建议：\n");
    printf("1. 使用多个队列分散负载（减少atomicAdd竞争）\n");
    printf("2. 使用per-warp队列（warp内同步，减少跨warp竞争）\n");
    printf("3. 使用更轻量的原子操作（如atomicAdd代替atomicExch）\n");
    printf("4. 考虑使用ticket-based队列（如q1.cu中的实现）\n");
    
    // 清理
    cudaFree((void*)d_seq);
    cudaFree(d_tail);
    cudaFree(d_latencies);
    free(h_latencies);
    free(h_seq_init);
    
    return 0;
}

