#!/bin/bash
threads=(1 2 4 8 16 32)

mkdir -p ./test/thread/

for thread in "${threads[@]}"; do
    echo "Running nvm-cuda-bench with $thread threads..."
    sudo ./build/bin/nvm-cuda-bench  --ctrl="/dev/libnvm0" --stats=1 --threads="$thread" --chunks=32 > ./test/thread/thread_$thread.log 2>&1
done