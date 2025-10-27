#!/bin/bash
threads=(1 2 4 8 16 32)

mkdir -p ./test/double_buffer_compute/

for thread in "${threads[@]}"; do
    echo "Running nvm-cuda-bench with $thread threads..."
    sudo ./build/bin/nvm-cuda-bench  --ctrl="/dev/libnvm0" --stats=1 --threads="$thread" --chunks=32 --double-buffer=1 > ./test/double_buffer_compute/thread_$thread.log 2>&1
done