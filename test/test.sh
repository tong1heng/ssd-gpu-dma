#!/bin/bash

# nvcc gctx.cu -o gctx -lcuda -arch=sm_80
# nvcc gctx.cu -g -o gctx -lcuda -arch=sm_80
# cuda-gdb ./gctx

nvcc stream.cu --ptxas-options=-v -O0 -o stream -lcuda -arch=sm_80

./stream