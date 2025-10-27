#!/bin/bash
chunks=(32 64 128 256 512 1024)

mkdir -p ./test/chunk/

for chunk in "${chunks[@]}"; do
    echo "Running nvm-cuda-bench with $chunk chunks..."
    sudo ./build/bin/nvm-cuda-bench  --ctrl="/dev/libnvm0" --stats=1 --threads=32 --chunks=$chunk > ./test/chunk/chunk_$chunk.log 2>&1
done