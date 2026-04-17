#include <iostream>

// hip header file
#include <hip/hip_runtime.h>

#define WIDTH 1024

#define NUM (WIDTH * WIDTH)

#define THREADS_PER_BLOCK_X 8
#define THREADS_PER_BLOCK_Y 8
#define THREADS_PER_BLOCK_Z 1

#define HIP_API_CALL(CALL)                                                                         \
    {                                                                                              \
        hipError_t error_ = (CALL);                                                                \
        if(error_ != hipSuccess)                                                                   \
        {                                                                                          \
            fprintf(stderr,                                                                        \
                    "%s:%d :: HIP error : %s\n",                                                   \
                    __FILE__,                                                                      \
                    __LINE__,                                                                      \
                    hipGetErrorString(error_));                                                    \
            throw std::runtime_error("hip_api_call");                                              \
        }                                                                                          \
    }

// Device (Kernel) function, it must be void
__global__ void
matrixTranspose(float* out, float* in, const int width)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    out[y * width + x] = in[x * width + y];
}

// CPU implementation of matrix transpose
void
matrixTransposeCPUReference(float* output, float* input, const unsigned int width)
{
    for(unsigned int j = 0; j < width; j++)
    {
        for(unsigned int i = 0; i < width; i++)
        {
            output[i * width + j] = input[j * width + i];
        }
    }
}

int
main()
{
    float* Matrix;
    float* TransposeMatrix;
    float* cpuTransposeMatrix;

    float* gpuMatrix;
    float* gpuTransposeMatrix;

    hipDeviceProp_t devProp;
    HIP_API_CALL(hipGetDeviceProperties(&devProp, 0));

    std::cout << "Device name " << devProp.name << std::endl;

    int i;
    int errors;

    Matrix             = (float*) malloc(NUM * sizeof(float));
    TransposeMatrix    = (float*) malloc(NUM * sizeof(float));
    cpuTransposeMatrix = (float*) malloc(NUM * sizeof(float));

    // initialize the input data
    for(i = 0; i < NUM; i++)
    {
        Matrix[i] = (float) i * 10.0f;
    }

    // allocate the memory on the device side
    HIP_API_CALL(hipMalloc((void**) &gpuMatrix, NUM * sizeof(float)));
    HIP_API_CALL(hipMalloc((void**) &gpuTransposeMatrix, NUM * sizeof(float)));

    // Memory transfer from host to device
    HIP_API_CALL(hipMemcpy(gpuMatrix, Matrix, NUM * sizeof(float), hipMemcpyHostToDevice));

    // Memory transfer that should be hidden by profiling tool
    HIP_API_CALL(
        hipMemcpy(gpuTransposeMatrix, gpuMatrix, NUM * sizeof(float), hipMemcpyDeviceToDevice));

    // Lauching kernel from host
    hipLaunchKernelGGL(matrixTranspose,
                       dim3(WIDTH / THREADS_PER_BLOCK_X, WIDTH / THREADS_PER_BLOCK_Y),
                       dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y),
                       0,
                       0,
                       gpuTransposeMatrix,
                       gpuMatrix,
                       WIDTH);

    // Memory transfer from device to host
    HIP_API_CALL(
        hipMemcpy(TransposeMatrix, gpuTransposeMatrix, NUM * sizeof(float), hipMemcpyDeviceToHost));

    // CPU MatrixTranspose computation
    matrixTransposeCPUReference(cpuTransposeMatrix, Matrix, WIDTH);

    // verify the results
    errors     = 0;
    double eps = 1.0E-6;
    for(i = 0; i < NUM; i++)
    {
        if(std::abs(TransposeMatrix[i] - cpuTransposeMatrix[i]) > eps)
        {
            errors++;
        }
    }
    if(errors != 0)
    {
        printf("FAILED: %d errors\n", errors);
    }
    else
    {
        printf("PASSED!\n");
    }

    // free the resources on device side
    HIP_API_CALL(hipFree(gpuMatrix));
    HIP_API_CALL(hipFree(gpuTransposeMatrix));

    // free the resources on host side
    free(Matrix);
    free(TransposeMatrix);
    free(cpuTransposeMatrix);

    return errors;
}
