#include <hip/hip_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define BLOCKDIM 64

/* Macro for checking GPU API return values */
#define HIP_ASSERT(call)                                                                 \
    do{                                                                                  \
        hipError_t gpuErr = call;                                                        \
        if(hipSuccess != gpuErr)                                                         \
        {                                                                                \
            printf("GPU API Error - %s:%d: '%s'\n", __FILE__, __LINE__,                  \
                   hipGetErrorString(gpuErr));                                           \
            exit(1);                                                                     \
        }                                                                                           \
    } while(0)


// HIP kernel. Each thread takes care of one element of input
__global__ void
cube(double *input, double *output,int offset, int elements_per_stream)
{
    size_t tid     = blockIdx.x*blockDim.x+threadIdx.x;
    size_t gstride = blockDim.x * gridDim.x;

    // Span all elements assigned to this stream
    for(size_t id = tid + offset; id < offset + elements_per_stream; id += gstride)
        for(size_t i = 0; i < 1000; ++i)
            output[id] = input[id]*input[id]*input[id];
}

void
usage()
{
    printf(
        "Usage: ./compute_comm_overlap <nstreams> <blockSize (optional, default=64)>\n");
    exit(1);
}

int
main(int argc, char* argv[])
{
    // Read input arg
    if(argc < 2) usage();
    int num_streams = atoi(argv[1]);

    // Number of threads in each thread block
    int blockSize = BLOCKDIM;
    if(argc == 3) blockSize = atoi(argv[2]);

    // Size of vectors
    int n                   = 100000000;
    int elements_per_stream = (n % num_streams == 0) ? n / num_streams : -1;
    if(elements_per_stream == -1)
    {
        printf("ERROR: input value of num_streams does not evenly devide the array size \n");
        printf("Please provide an input value for num_streams that evenly divides %d\n", n);
        exit(1);
    }
    int bytes_per_stream = elements_per_stream * sizeof(double);
    printf("bytes/stream: %f (MB)\n", bytes_per_stream * 1.e-6);


    // Host input vectors
    double* h_input1;
    // Host output vector
    double* h_output1;
    // Host output vector for verification
    double* h_verify1;

    // Device input vectors
    double* d_input1;
    // Device output vector
    double* d_output1;

    // Creating events for timers
    hipEvent_t start, stop;
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


    h_verify1 = (double*) malloc(bytes);


    printf("Finished allocating vectors on the CPU\n");
    // Allocate memory for each vector on GPU
    HIP_ASSERT(hipMalloc(&d_input1, bytes));
    HIP_ASSERT(hipMalloc(&d_output1, bytes));

    printf("Finished allocating vectors on the GPU\n");

    int i;
    // Initialize vectors on host
    for(i = 0; i < n; i++)
    {
        h_input1[i] = sin(i);
    }

    // Number of thread blocks in grid
#ifdef MI300A_RUN
    int gridSizePerStream = 228;  //(int)ceil((float)elements_per_stream/blockSize);
#else
    int gridSizePerStream = 104;  //(int)ceil((float)elements_per_stream/blockSize);
#endif
    printf("gridSizePerStream: %d\n", gridSizePerStream);

    HIP_ASSERT(hipEventRecord(start));
    // Copy and execute in loop

#ifndef SPLIT_DATACOPY_KERNEL_CALLS
    for(int i = 0; i < num_streams; i++)
    {
        int offset = i * elements_per_stream;
        HIP_ASSERT(hipMemcpyAsync(&d_input1[offset], &h_input1[offset], bytes_per_stream,
                                  hipMemcpyHostToDevice,streams[i]));

        HIP_ASSERT(hipEventRecord(start, streams[i]));

        cube<<<gridSizePerStream, blockSize, 0, streams[i]>>>(d_input1, d_output1, offset,
                                                              elements_per_stream);

        HIP_ASSERT(hipEventRecord(stop, streams[i]));

        HIP_ASSERT(hipMemcpyAsync(&h_output1[offset], &d_output1[offset],
                                  bytes_per_stream, hipMemcpyDeviceToHost,streams[i]));
    }
#else
    // split H2D copies and kernel calls into separate loops
    for(int i = 0; i < num_streams; i++)
    {
        int offset = i * elements_per_stream;
        HIP_ASSERT(hipMemcpyAsync(&d_input1[offset], &h_input1[offset], bytes_per_stream,
                                  hipMemcpyHostToDevice,streams[i]));
    }
    for(int i = 0; i < num_streams; i++)
    {
        int offset = i * elements_per_stream;
        HIP_ASSERT(hipEventRecord(start, streams[i]));
        cube<<<gridSizePerStream, blockSize, 0, streams[i]>>>(d_input1, d_output1, offset,
                                                              elements_per_stream);
        HIP_ASSERT(hipEventRecord(stop, streams[i]));
    }
    for(int i = 0; i < num_streams; i++)
    {
        int offset = i * elements_per_stream;
        HIP_ASSERT(hipMemcpyAsync(&h_output1[offset], &d_output1[offset],
                                  bytes_per_stream, hipMemcpyDeviceToHost, streams[i]));
    }
#endif //SPLIT_DATACOPY_KERNEL_CALLS

    HIP_ASSERT(hipEventRecord(stop));
    HIP_ASSERT(hipEventSynchronize(stop));

    float milliseconds = 0;
    HIP_ASSERT(hipEventElapsedTime(&milliseconds, start, stop));
    printf("Time required total (ms) %f\n", milliseconds);
    printf("Finished copying the output vector from the GPU to the CPU\n");

    // Compute for CPU.
    for(i=0; i < n; i++)
    {
        h_verify1[i] = h_input1[i] * h_input1[i] * h_input1[i];
    }


    // Verify results
    for(i=0; i < n; i++)
    {
        if(abs(h_verify1[i] - h_output1[i]) > 1e-5)
        {
            printf("Error at position i %d, Expected: %f, Found: %f \n", i, h_verify1[i],
                   h_output1[i]);
        }
    }

    printf("Releasing GPU memory\n");

    // Release device memory
    HIP_ASSERT(hipFree(d_input1));
    HIP_ASSERT(hipFree(d_output1));

    // Release host memory
    printf("Releasing CPU memory\n");
    HIP_ASSERT(hipHostFree(h_input1));
    HIP_ASSERT(hipHostFree(h_output1));
    free(h_verify1);

    return 0;
}
