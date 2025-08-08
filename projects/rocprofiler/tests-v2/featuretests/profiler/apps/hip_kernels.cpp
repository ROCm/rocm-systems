#include <hip/hip_runtime.h>
#include <cassert>
#include <vector>

#ifdef NDEBUG
#define HIP_ASSERT(x) x
#else
#define HIP_ASSERT(x) (assert((x) == hipSuccess))
#endif

#define WIDTH 1024
#define HEIGHT 1024

#define NUM (WIDTH * HEIGHT)

#define THREADS_PER_BLOCK_X 16
#define THREADS_PER_BLOCK_Y 16
#define THREADS_PER_BLOCK_Z 1

__device__ int counter = 0;
// empty kernel
__global__ void kernel() {}

// vector add kernel
__global__ void vectoradd_float(float* __restrict__ a, const float* __restrict__ b,
                                const float* __restrict__ c, int width, int height)

{
  int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

  int i = y * width + x;
  if (i < (width * height)) {
    a[i] = b[i] + c[i];
  }
}

__global__ void add(int n, float* x, float* y) {

  if(__hip_atomic_load(&counter, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT) != 0){
    abort();
  }
  __hip_atomic_fetch_add(&counter, 1, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride) y[i] = x[i] + y[i];
   __hip_atomic_fetch_add(&counter, -1, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);

}

// launches an empty kernel in profiler context
void KernelLaunch() {
  // run empty kernel
  kernel<<<1, 1>>>();
  hipDeviceSynchronize();
}
void LaunchMultiStreamKernels() {
  int N = 1 << 4;
  float* x = new float[N];
  float* y = new float[N];
  float* d_x;
  float* d_y;
  //   Allocate Unified Memory -- accessible from CPU or GPU
  HIP_ASSERT(hipMallocManaged(&d_x, N * sizeof(float)));
  HIP_ASSERT(hipMallocManaged(&d_y, N * sizeof(float)));

  //   initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }
  std::vector< hipStream_t> hip_streams;
  for(int i = 0; i < 100; i++) {
    hipStream_t stream;
    hipStreamCreate	(&stream);
    hip_streams.push_back(stream);

  }
  HIP_ASSERT(hipMemcpy(d_x, x, N * sizeof(float), hipMemcpyHostToDevice));
  HIP_ASSERT(hipMemcpy(d_y, y, N * sizeof(float), hipMemcpyHostToDevice));

  // Launch kernel on 1M elements on the GPU
  int blockSize = 64;
  // This Kernel will always be launched with one wave
  int numBlocks = 1;

  for(int i = 0; i < 100; i++) {
    for(int j = 0; j < hip_streams.size(); j++)
       hipLaunchKernelGGL(add, numBlocks, blockSize, 0, hip_streams[j], N, d_x, d_y);
    HIP_ASSERT(hipDeviceSynchronize());
  }

  //Wait for GPU to finish before accessing on host
  HIP_ASSERT(hipDeviceSynchronize());

  HIP_ASSERT(hipMemcpy(x, d_x, N * sizeof(float), hipMemcpyDeviceToHost));
  HIP_ASSERT(hipMemcpy(y, d_y, N * sizeof(float), hipMemcpyDeviceToHost));

    //   Free memory
  HIP_ASSERT(hipFree(d_x));
  HIP_ASSERT(hipFree(d_y));

  delete[] x;
  delete[] y;
}
int LaunchVectorAddKernel() {
  float* hostA;
  float* hostB;
  float* hostC;

  float* deviceA;
  float* deviceB;
  float* deviceC;

  hipDeviceProp_t devProp;
  hipGetDeviceProperties(&devProp, 0);

  int i;
  int errors;

  hostA = (float*)malloc(NUM * sizeof(float));
  hostB = (float*)malloc(NUM * sizeof(float));
  hostC = (float*)malloc(NUM * sizeof(float));

  // initialize the input data
  for (i = 0; i < NUM; i++) {
    hostB[i] = (float)i;
    hostC[i] = (float)i * 100.0f;
  }

  HIP_ASSERT(hipMalloc((void**)&deviceA, NUM * sizeof(float)));
  HIP_ASSERT(hipMalloc((void**)&deviceB, NUM * sizeof(float)));
  HIP_ASSERT(hipMalloc((void**)&deviceC, NUM * sizeof(float)));

  HIP_ASSERT(hipMemcpy(deviceB, hostB, NUM * sizeof(float), hipMemcpyHostToDevice));
  HIP_ASSERT(hipMemcpy(deviceC, hostC, NUM * sizeof(float), hipMemcpyHostToDevice));


  hipLaunchKernelGGL(vectoradd_float,
                     dim3(WIDTH / THREADS_PER_BLOCK_X, HEIGHT / THREADS_PER_BLOCK_Y),
                     dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y), 0, 0, deviceA, deviceB,
                     deviceC, WIDTH, HEIGHT);


  HIP_ASSERT(hipMemcpy(hostA, deviceA, NUM * sizeof(float), hipMemcpyDeviceToHost));

  // verify the results
  errors = 0;
  for (i = 0; i < NUM; i++) {
    if (hostA[i] != (hostB[i] + hostC[i])) {
      errors++;
    }
  }
  if (errors != 0) {
    printf("FAILED: %d errors\n", errors);
  }

  HIP_ASSERT(hipFree(deviceA));
  HIP_ASSERT(hipFree(deviceB));
  HIP_ASSERT(hipFree(deviceC));

  free(hostA);
  free(hostB);
  free(hostC);

  // hipResetDefaultAccelerator();

  return errors;
}