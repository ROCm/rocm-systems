#include <thread>
#include <vector>

#include "hip/hip_runtime.h"

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

static void
copy_to_dev(const hipStream_t stream)
{
    unsigned int n   = (32 * 1024);  // 32KB
    double*      A_h = nullptr;
    double*      A_d = nullptr;

    HIP_ASSERT(hipHostMalloc(&A_h, n * sizeof(double)));
    HIP_ASSERT(hipMalloc(&A_d, n * sizeof(double)));

    for(unsigned int i = 0; i < n; ++i)
    {
        A_h[i] = 123.5;
    }
    HIP_ASSERT(hipMemcpyAsync(A_d, A_h, n * sizeof(double), hipMemcpyHostToDevice, stream));
    // Repeat to make sure streams remain the same
    HIP_ASSERT(hipMemcpyAsync(A_d, A_h, n * sizeof(double), hipMemcpyHostToDevice, stream));

    // Release device memory
    HIP_ASSERT(hipFree(A_d));
    // Release host memory
    HIP_ASSERT(hipHostFree(A_h));
}

int
main()
{
    // Test hipStreamPerThread with multiple threads
    const size_t             num_streams = 3;
    const size_t             thread_cnt  = 9;
    std::vector<std::thread> threads(thread_cnt);
    // Creating streams
    hipStream_t streams[num_streams];
    threads[0] = std::thread(copy_to_dev, nullptr);
    threads[1] = std::thread(copy_to_dev, hipStreamPerThread);
    HIP_ASSERT(hipStreamCreate(&streams[0]));
    threads[2] = std::thread(copy_to_dev, streams[0]);
    threads[3] = std::thread(copy_to_dev, hipStreamLegacy);
    threads[4] = std::thread(copy_to_dev, hipStreamPerThread);
    HIP_ASSERT(hipStreamCreate(&streams[1]));
    threads[5] = std::thread(copy_to_dev, streams[1]);
    threads[6] = std::thread(copy_to_dev, hipStreamLegacy);
    threads[7] = std::thread(copy_to_dev, hipStreamPerThread);
    HIP_ASSERT(hipStreamCreate(&streams[2]));
    threads[8] = std::thread(copy_to_dev, streams[2]);

    for(auto& thread : threads)
    {
        thread.join();
    }
    return 0;
}
