#include <cuda.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <thread>


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

__global__ __launch_bounds__(1024, 2)
void nopKernel() {
  int thread_id = threadIdx.x;
  if (thread_id == 0) printf("nopKernel running on SM %u\n", getsmid());
  while (1) {
		// if (thread_id == 0) {
		// 	printf("nopKernel is running\n");
		// }
    __nanosleep(1000000);
  }
}

__global__ __launch_bounds__(1024, 2)
 void activeKernelA() {
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (thread_id == 0) printf("activeKernelA running on SM %u\n", getsmid());
  while (1) {
    if (thread_id == 0) {
      printf("activeKernelA is running\n");
    }
    size_t time = 10LL * 1e9;
    __nanosleep(time);
  }
}

__global__ void activeKernelB() {
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (thread_id == 0) printf("activeKernelB running on SM %u\n", getsmid());
  while (1) {
    if (thread_id == 0) {
      printf("activeKernelB is running\n");
    }
    size_t time = 10LL * 1e9;
    __nanosleep(time);
  }
}

int main() {
  CUgreenCtx gctx[2];
  CUdevResourceDesc desc[2];
  CUdevResource input;
  CUdevResource resources[2];
  CUstream streamA;
  CUstream streamB;
  CUstream streamC;

  unsigned int nbGroups = 1;
  unsigned int minCount = 0;

  // Initialize device 0
  CUDA_RT(cudaInitDevice(0, 0, 0));

  // Query input SMs
  CUDA_DRV(cuDeviceGetDevResource((CUdevice)0, &input, CU_DEV_RESOURCE_TYPE_SM));
  minCount = 4;
  printf("Device 0 has %u SMs, requesting %u for green context\n",
         input.sm.smCount, minCount);

  // Split my resources
  CUDA_DRV(cuDevSmResourceSplitByCount(&resources[0], &nbGroups, &input,
                                       &resources[1], 0, minCount));
  CUDA_DRV(cuDevResourceGenerateDesc(&desc[0], &resources[0], 1));
  CUDA_DRV(cuGreenCtxCreate(&gctx[0], desc[0], (CUdevice)0,
                            CU_GREEN_CTX_DEFAULT_STREAM));
  CUDA_DRV(
      cuGreenCtxStreamCreate(&streamA, gctx[0], CU_STREAM_NON_BLOCKING, 0));
  CUDA_DRV(
      cuGreenCtxStreamCreate(&streamB, gctx[0], CU_STREAM_NON_BLOCKING, 0));
  CUDA_DRV(
      cuGreenCtxStreamCreate(&streamC, gctx[0], CU_STREAM_NON_BLOCKING, 0));

  CUdevResource i1;
  CUDA_DRV(cuGreenCtxGetDevResource(gctx[0], &i1, CU_DEV_RESOURCE_TYPE_SM));
  printf("Green context 0 has %u SMs\n", i1.sm.smCount);
  printf("Remaining has %u SMs\n", resources[1].sm.smCount);


  // launch kernels

  printf("Launching nop kernels\n");
  nopKernel<<<6, 1024, 0, (cudaStream_t)streamA>>>();
  std::this_thread::sleep_for(std::chrono::seconds(2));

  printf("Launching A\n");
  activeKernelA<<<1, 1, 0, (cudaStream_t)streamB>>>();

  // int err = cuLaunchKernel((CUfunction)activeKernelA, 1, 1, 1, 1024, 1, 1, 0,
  //                          streamB, nullptr, nullptr);
	// auto err = cudaLaunchKernel(activeKernelA, dim3(1, 1, 1), dim3(1024, 1, 1), nullptr, 0,
	// 											 (cudaStream_t)streamB);

  // printf("Launching A ret = %d\n", err);

  std::this_thread::sleep_for(std::chrono::seconds(2));

  printf("Launching B\n");
  activeKernelB<<<1, 32, 0, (cudaStream_t)streamC>>>();
	// err = cudaLaunchKernel(activeKernelB, dim3(1, 1, 1), dim3(32, 1, 1), nullptr, 0,
	// 											 (cudaStream_t)streamC);
	// printf("Launching B ret = %d\n", err);

  CUDA_RT(cudaGetLastError());
  cudaDeviceSynchronize();

  return 0;
}