#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "hip/hip_runtime.h"

#define BLOCKDIM 64

/* Macro for checking GPU API return values */
#define HIP_ASSERT(call)                                                                           \
    do                                                                                             \
    {                                                                                              \
        hipError_t gpuErr = call;                                                                  \
        if(hipSuccess != gpuErr)                                                                   \
        {                                                                                          \
            printf(                                                                                \
                "GPU API Error - %s:%d: '%s'\n", __FILE__, __LINE__, hipGetErrorString(gpuErr));   \
            exit(1);                                                                               \
        }                                                                                          \
    } while(0)

// HIP kernel. Each thread takes care of one element of input
__global__ void
cube(double* input, double* output, size_t offset, size_t elements_per_stream)
{
    size_t tid     = blockIdx.x * blockDim.x + threadIdx.x;
    size_t gstride = blockDim.x * gridDim.x;

    // Span all elements assigned to this stream
    for(size_t id = tid + offset; id < offset + elements_per_stream; id += gstride)
        for(size_t i = 0; i < 1000; ++i)
            output[id] = input[id] * input[id] * input[id];
}

int
main()
{
    // number of streams
    const int num_streams = 8;

    // Number of threads in each thread block
    const int blockSize = 512;

    // Size of vectors
    int n                   = 100000000;
    int elements_per_stream = n / num_streams;
    int bytes_per_stream    = elements_per_stream * sizeof(double);

    // Host input vectors
    double* h_input1{nullptr};
    // Host output vector
    double* h_output1{nullptr};

    // Device input vectors
    double* d_input1{nullptr};
    // Device output vector
    double* d_output1{nullptr};

    // Creating events for timers
    hipEvent_t start{}, stop{};
    HIP_ASSERT(hipEventCreate(&start));
    HIP_ASSERT(hipEventCreate(&stop));

    // Creating streams
    hipStream_t streams[num_streams];
    for(int i = 0; i < num_streams; ++i)
    {
        HIP_ASSERT(hipStreamCreate(&streams[i]));
    }

    // Size, in bytes, of each vector
    size_t bytes = n * sizeof(double);

    // Allocate page locked memory for these vectors on host
    HIP_ASSERT(hipHostMalloc(&h_input1, bytes));
    HIP_ASSERT(hipHostMalloc(&h_output1, bytes));

    // Allocate memory for each vector on GPU
    HIP_ASSERT(hipMalloc(&d_input1, bytes));
    HIP_ASSERT(hipMalloc(&d_output1, bytes));

    // Initialize vectors on host
    for(int i = 0; i < n; i++)
    {
        h_input1[i] = sin(i);
    }

    // Number of thread blocks in grid
    const int gridSizePerStream = 104;  //(int)ceil((float)elements_per_stream/blockSize);

    HIP_ASSERT(hipEventRecord(start));
    // split H2D copies and kernel calls into separate loops
    for(int i = 0; i < num_streams; i++)
    {
        int offset = i * elements_per_stream;
        HIP_ASSERT(hipMemcpyAsync(&d_input1[offset],
                                  &h_input1[offset],
                                  bytes_per_stream,
                                  hipMemcpyHostToDevice,
                                  streams[i]));
    }
    for(int i = 0; i < num_streams; i++)
    {
        int offset = i * elements_per_stream;
        cube<<<gridSizePerStream, blockSize, 0, streams[i]>>>(
            d_input1, d_output1, offset, elements_per_stream);
    }
    for(int i = 0; i < num_streams; i++)
    {
        int offset = i * elements_per_stream;
        HIP_ASSERT(hipMemcpyAsync(&h_output1[offset],
                                  &d_output1[offset],
                                  bytes_per_stream,
                                  hipMemcpyDeviceToHost,
                                  streams[i]));
    }

    HIP_ASSERT(hipEventRecord(stop));
    HIP_ASSERT(hipEventSynchronize(stop));

    float milliseconds = 0;
    HIP_ASSERT(hipEventElapsedTime(&milliseconds, start, stop));

    // Release device memory
    HIP_ASSERT(hipFree(d_input1));
    HIP_ASSERT(hipFree(d_output1));

    // Release host memory
    HIP_ASSERT(hipHostFree(h_input1));
    HIP_ASSERT(hipHostFree(h_output1));

    // Destroy streams
    for(int i = 0; i < num_streams; ++i)
    {
        HIP_ASSERT(hipStreamDestroy(streams[i]));
    }

    return 0;
}
