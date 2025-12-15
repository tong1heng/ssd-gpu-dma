#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <sys/time.h>

#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <thread>

using namespace nvcuda::wmma;

#define CUDA_RT(call)                                                       \
  do {                                                                      \
    cudaError_t _err = (call);                                              \
    if (cudaSuccess != _err) {                                              \
      fprintf(stderr, "CUDA error in file '%s' at line %i: %s\n", __FILE__, \
              __LINE__, cudaGetErrorString(_err));                          \
      return _err;                                                          \
    }                                                                       \
  } while (0)

#define CUDA_DRV(call)                                                      \
  do {                                                                      \
    CUresult _status = (call);                                              \
    if (CUDA_SUCCESS != _status) {                                          \
      fprintf(stderr, "CUDA error in file '%s' at line %i: %i\n", __FILE__, \
              __LINE__, _status);                                           \
      return _status;                                                       \
    }                                                                       \
  } while (0)

__device__ unsigned int getsmid() {
  unsigned int smid;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
  return smid;
}

__device__ void sleep_s(int seconds) {
  for (uint64_t _ = 0; _ < seconds; _++) {
    for (int i = 0; i < 1e3; i++) __nanosleep(1LL * 1e6);
  }
}

#define next_rand() (seed = seed * 1664525u + 1013904223u)

__global__  //__launch_bounds__(1024, 2)
    void
    nopKernel() {
  int thread_id = threadIdx.x;

  // // if (thread_id == 0) printf("nopKernel running on SM %u\n", getsmid());
  // for (int iter = 0; iter < 10000; ++iter) {
  //   // 使用简单 LCG 在 device 上生成伪随机数
  //   uint32_t seed =
  //       (threadIdx.x + blockIdx.x * blockDim.x) * 1664525u + 1013904223u;

  //   fragment<matrix_a, 16, 16, 16, __half, row_major> a_frag;
  //   fragment<matrix_b, 16, 16, 16, __half, row_major> b_frag;
  //   fragment<accumulator, 16, 16, 16, float> c_frag;

  //   // 初始化临时矩阵并随机赋值
  //   __half A[16 * 16];
  //   __half B[16 * 16];
  //   for (int i = 0; i < 16 * 16; ++i) {
  //     // 生成 0..1 范围的随机浮点数并转换为 __half
  //     float ra = (next_rand() & 0x7FFF) / float(0x7FFF);
  //     float rb = (next_rand() & 0x7FFF) / float(0x7FFF);
  //     A[i] = __float2half(ra);
  //     B[i] = __float2half(rb);
  //   }

  //   // 加载到 fragment
  //   nvcuda::wmma::load_matrix_sync(a_frag, A, 16);
  //   nvcuda::wmma::load_matrix_sync(b_frag, B, 16);

  //   // 初始化累加器为 0
  //   nvcuda::wmma::fill_fragment(c_frag, 0.0f);

  //   // 执行 10000 次 WMMA 指令
  //   for (int iter = 0; iter < 10000; ++iter) {
  //     nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
  //   }

  //   // 防止编译器优化掉计算：让线程0 打印一个累加结果（可选）
  //   if (thread_id == 0) {
  //     printf("wmma finished, c_frag[0]=%f\n", c_frag.x[0]);
  //   }
  // }

  sleep_s(3);

  // if (thread_id == 0) printf("nopKernel done on SM %u\n", getsmid());
}

__global__  //__launch_bounds__(1024, 2)
    void
    activeKernelA(int *ptr) {
  *ptr = 1;
  int thread_id = threadIdx.x;
  // if (thread_id == 0) printf("activeKernelA running on SM %u\n", getsmid());
  // for (uint64_t _ = 0; _ < (1ULL << 63) - 1; _++) {
  //   if (thread_id == 0) {
  //     printf("activeKernelA is running\n");
  //   }
  // sleep_s(1);
  // }
  // sleep_s(1);
  // if (thread_id == 0) printf("activeKernelA done on SM %u\n", getsmid());
}

__global__  //__launch_bounds__(1,32)
    void
    activeKernelB() {
  int thread_id = threadIdx.x;
  if (thread_id == 0) printf("activeKernelB running on SM %u\n", getsmid());
  while (1) {
    if (thread_id == 0) {
      printf("activeKernelB is running\n");
    }
    __nanosleep(1LL * 1e6);
  }
}

int main() {
  cudaStream_t streamA;
  cudaStream_t streamB;
  cudaStream_t streamC;

  cudaStreamCreate(&streamA);
  cudaStreamCreate(&streamB);
  cudaStreamCreate(&streamC);

  int *ptr;
  cudaHostAlloc((void **)&ptr, sizeof(int), cudaHostAllocPortable);

  *ptr = 0;

  printf("Launching nop kernels\n");
  // nopKernel<<<56 * 2 - 1, 1024, 0, (cudaStream_t)streamA>>>();
  nopKernel<<<1, 32, 20 * 1024, (cudaStream_t)streamA>>>();


  printf("Launching A\n");
  activeKernelA<<<1, 32, 0, (cudaStream_t)streamB>>>(ptr);

  for (int _ = 0; _ < 20; _++) {
    printf("After launching A %d\n", *ptr);
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
  }

  // std::this_thread::sleep_for(std::chrono::seconds(1));

  // printf("Launching B\n");
  // activeKernelB<<<1, 32, 0, (cudaStream_t)streamC>>>();

  cudaDeviceSynchronize();

  //   CUDA_RT(cudaGetLastError());

  return 0;
}