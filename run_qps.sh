#!/bin/bash
qps=(1 2 4 8 16 32 64 128)

mkdir -p ./test/qps

for qp in "${qps[@]}"; do
    echo "Running nvm-cuda-bench with $qp QPs..."
    sudo ./build/bin/nvm-cuda-bench  --ctrl="/dev/libnvm0" --stats=1 --threads=32 --qpcount=$qp --chunks=32 > ./test/qps/qp_$qp.log 2>&1
done