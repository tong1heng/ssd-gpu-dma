cmake_minimum_required (VERSION 3.1)
project (libnvm-benchmarks)

set (THREADS_PREFER_PTHREAD_FLAG ON)
find_package (Threads REQUIRED)

find_package (CUDA 8.0 REQUIRED)

make_sisci_benchmark (mock-benchmark-smartio mock-bench "main.cu;buffer.cu;ctrl.cc;settings.cu;queue.cu")

make_benchmark (mock-benchmark-module mock-bench "main.cu;buffer.cu;ctrl.cc;settings.cu;queue.cu")

make_benchmark_choice (mock-benchmark mock-benchmark-smartio mock-benchmark-module)
