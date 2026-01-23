#include <cuda_runtime.h>
#include <cstdio>
#include <cassert>

template <typename T>
struct CUDAQueue {
    T* buffer;      // 队列缓冲区
    int capacity;   // 队列容量
    int head;       // 读指针（pop位置）
    int tail;       // 写指针（push位置）
    int count;      // 当前队列中的元素数量
};

template <typename T>
__host__ void queueInit(CUDAQueue<T>* d_queue, int capacity) {
    CUDAQueue<T> h_queue;
    h_queue.capacity = capacity;
    h_queue.head = 0;
    h_queue.tail = 0;
    h_queue.count = 0;

    T* d_buffer = nullptr;
    cudaMalloc(&d_buffer, capacity * sizeof(T));
    h_queue.buffer = d_buffer;
    cudaMemcpy(d_queue, &h_queue, sizeof(CUDAQueue<T>), cudaMemcpyHostToDevice);
}

template <typename T>
__device__ bool queuePush(CUDAQueue<T>* queue, const T& value) {
    if (queue->count == queue->capacity) {
        // 队列满，无法插入
        return false;
    }

    // 在尾部插入元素
    queue->buffer[queue->tail] = value;

    // 更新tail和count
    queue->tail = (queue->tail + 1) % queue->capacity;
    queue->count++;

    return true;
}

template <typename T>
__device__ bool queuePop(CUDAQueue<T>* queue, T& value) {
    if (queue->count == 0) {
        // 队列为空，无法弹出元素
        return false;
    }

    // 从头部弹出元素
    value = queue->buffer[queue->head];

    // 更新head和count
    queue->head = (queue->head + 1) % queue->capacity;
    queue->count--;

    return true;
}