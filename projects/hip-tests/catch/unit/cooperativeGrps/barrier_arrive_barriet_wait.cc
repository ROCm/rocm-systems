
#include "cooperative_groups_common.hh"
//#include "cg_common_kernels.hh"

#include <cpu_grid.h>
#include <optional>
#include <resource_guards.hh>
#include <utils.hh>

#include <cmd_options.hh>

static __global__ void coopKernel_1D_Grid_sleep(uint32_t interval,
                                                const uint32_t ticks_per_ms) {
  printf("gridDim.x = %d , gridDim.y = %d, gridDim.z = %d \n", gridDim.x,
         gridDim.y, gridDim.z);
  printf("blockIdx.z = %d , blockIdx.y = %d , blockIdx.x = %d \n", blockIdx.z,
         blockIdx.y, blockIdx.x);

  cooperative_groups::grid_group grid = cooperative_groups::this_grid();

  // Hold all threads >= 2
  if (blockIdx.x >= 64) {
    printf("PreProcessing started in block %d\n", blockIdx.x);

    while (interval--) {
      uint64_t start = clock64();
      while (clock64() - start < ticks_per_ms) {
        __builtin_amdgcn_s_sleep(10);
      }
    }
    printf("PreProcessing ended in block %d\n", blockIdx.x);
  }

  auto token = grid.barrier_arrive();

  printf("MainProcessing started in block %d\n", blockIdx.x);


  printf("MainProcessing ended in block %d\n", blockIdx.x);

  grid.barrier_wait(std::move(token));

  printf("FinalProcessing started in block %d\n", blockIdx.x);

  printf("FinalProcessing ended in block %d\n", blockIdx.x);
}

static __global__ void coopKernel_2D_Grid_sleep(uint32_t interval,
                                                const uint32_t ticks_per_ms) {
  printf("gridDim.x = %d , gridDim.y = %d, gridDim.z = %d \n", gridDim.x,
         gridDim.y, gridDim.z);
  printf("blockIdx.z = %d , blockIdx.y = %d , blockIdx.x = %d \n", blockIdx.z,
         blockIdx.y, blockIdx.x);

  cooperative_groups::grid_group grid = cooperative_groups::this_grid();

  // Hold all threads in y direction
  if (blockIdx.y >= 1) {
    printf("PreProcessing in started blockIdx.y = %d , blockIdx.x = %d \n",
           blockIdx.y, blockIdx.x);

    while (interval--) {
      uint64_t start = clock64();
      while (clock64() - start < ticks_per_ms) {
        __builtin_amdgcn_s_sleep(10);
      }
    }
    printf("PreProcessing ended in blockIdx.y = %d , blockIdx.x = %d \n",
           blockIdx.y, blockIdx.x);
  }

  auto token = grid.barrier_arrive();

  printf("MainProcessing started in blockIdx.y = %d , blockIdx.x = %d \n",
         blockIdx.y, blockIdx.x);

  printf("MainProcessing ended in blockIdx.y = %d , blockIdx.x = %d \n",
         blockIdx.y, blockIdx.x);

  grid.barrier_wait(std::move(token));

  printf("FinalProcessing Doing in blockIdx.y = %d , blockIdx.x = %d \n",
         blockIdx.y, blockIdx.x);
}

static __global__ void coopKernel_3D_Grid_z_sleep(uint32_t interval,
                                                  const uint32_t ticks_per_ms) {
  printf("gridDim.x = %d , gridDim.y = %d, gridDim.z = %d \n", gridDim.x,
         gridDim.y, gridDim.z);
  printf("blockIdx.z = %d , blockIdx.y = %d , blockIdx.x = %d \n", blockIdx.z,
         blockIdx.y, blockIdx.x);

  cooperative_groups::grid_group grid = cooperative_groups::this_grid();

  // Hold all threads in z direction
  if (blockIdx.z >= 1) {
    printf("PreProcessing in started blockIdx.z = %d , blockIdx.y = %d , "
           "blockIdx.x = %d \n",
           blockIdx.z, blockIdx.y, blockIdx.x);

    while (interval--) {
      uint64_t start = clock64();
      while (clock64() - start < ticks_per_ms) {
        __builtin_amdgcn_s_sleep(10);
      }
    }
    printf("PreProcessing ended in blockIdx.z = %d , blockIdx.y = %d , "
           "blockIdx.x = %d \n",
           blockIdx.z, blockIdx.y, blockIdx.x);
  }

  auto token = grid.barrier_arrive();

  printf("MainProcessing started in blockIdx.z = %d , blockIdx.y = %d , "
         "blockIdx.x = %d \n",
         blockIdx.z, blockIdx.y, blockIdx.x);

  printf("MainProcessing ended in blockIdx.z = %d , blockIdx.y = %d , "
         "blockIdx.x = %d \n",
         blockIdx.z, blockIdx.y, blockIdx.x);

  grid.barrier_wait(std::move(token));

  printf("FinalProcessing Doing in blockIdx.z = %d , blockIdx.y = %d , "
         "blockIdx.x = %d \n",
         blockIdx.z, blockIdx.y, blockIdx.x);
}

static __global__ void
coopKernel_3D_Grid_yz_sleep(uint32_t interval, const uint32_t ticks_per_ms) {
  printf("gridDim.x = %d , gridDim.y = %d, gridDim.z = %d \n", gridDim.x,
         gridDim.y, gridDim.z);
  printf("blockIdx.z = %d , blockIdx.y = %d , blockIdx.x = %d \n", blockIdx.z,
         blockIdx.y, blockIdx.x);

  cooperative_groups::grid_group grid = cooperative_groups::this_grid();

  // Hold all threads in y & z direction
  if (blockIdx.y >= 1 || blockIdx.z >= 1) {
    printf("PreProcessing in started blockIdx.z = %d , blockIdx.y = %d , "
           "blockIdx.x = %d \n",
           blockIdx.z, blockIdx.y, blockIdx.x);

    while (interval--) {
      uint64_t start = clock64();
      while (clock64() - start < ticks_per_ms) {
        __builtin_amdgcn_s_sleep(10);
      }
    }
    printf("PreProcessing ended in blockIdx.z = %d , blockIdx.y = %d , "
           "blockIdx.x = %d \n",
           blockIdx.z, blockIdx.y, blockIdx.x);
  }

  auto token = grid.barrier_arrive();

  printf("MainProcessing started in blockIdx.z = %d , blockIdx.y = %d , "
         "blockIdx.x = %d \n",
         blockIdx.z, blockIdx.y, blockIdx.x);

  printf("MainProcessing ended in blockIdx.z = %d , blockIdx.y = %d , "
         "blockIdx.x = %d \n",
         blockIdx.z, blockIdx.y, blockIdx.x);

  grid.barrier_wait(std::move(token));

  printf("FinalProcessing Doing in blockIdx.z = %d , blockIdx.y = %d , "
         "blockIdx.x = %d \n",
         blockIdx.z, blockIdx.y, blockIdx.x);
}

TEST_CASE("Unit_barrier_wait_barrier_arrive_Grid_sleep") {
  int cooperativeLaunchSupported;
  HIP_CHECK(hipDeviceGetAttribute(&cooperativeLaunchSupported,
                                  hipDeviceAttributeCooperativeLaunch, 0));

  if (!cooperativeLaunchSupported) {
    HipTest::HIP_SKIP_TEST("Skipping test as CooperativeLaunch not supported");
  }

  int ticks_per_ms = 0;
  HIP_CHECK(
      hipDeviceGetAttribute(&ticks_per_ms, hipDeviceAttributeWallClockRate, 0));
  std::cout << "ticks_per_ms : " << ticks_per_ms << std::endl;

  // const std::chrono::duration<uint64_t, std::milli> delay = 1000000; // 100
  // sec
  std::chrono::milliseconds delay = std::chrono::milliseconds(1000000);
  std::cout << "delay : " << delay.count() << std::endl;

  ///////////////// Case 1D grid - Hold all threads >= 2
  {
    constexpr int N = 128;
    dim3 gridDim = dim3{N, 1, 1};
    dim3 blockDim = dim3{1, 1, 1};

    uint32_t interval = delay.count();
    void *params[2];
    params[0] = &interval;
    params[1] = &ticks_per_ms;

    HIP_CHECK(hipLaunchCooperativeKernel((void *)(coopKernel_1D_Grid_sleep),
                                         gridDim, blockDim, params, 0, 0));
    HIP_CHECK(hipDeviceSynchronize());
  }

  ///////////////// Case 2D grid  - Hold all threads in y direction
  {
    constexpr int N = 2;
    dim3 gridDim = dim3{N, N, 1};
    dim3 blockDim = dim3{1, 1, 1};

    uint32_t interval = delay.count();
    void *params[2];
    params[0] = &interval;
    params[1] = &ticks_per_ms;

    HIP_CHECK(hipLaunchCooperativeKernel((void *)(coopKernel_2D_Grid_sleep),
                                         gridDim, blockDim, params, 0, 0));
    HIP_CHECK(hipDeviceSynchronize());
  }

  ///////////////// Case 3D grid  - Hold all threads in z directions
  {
    constexpr int N = 2;
    dim3 gridDim = dim3{N, N, N};
    dim3 blockDim = dim3{1, 1, 1};

    uint32_t interval = delay.count();
    void *params[2];
    params[0] = &interval;
    params[1] = &ticks_per_ms;

    HIP_CHECK(hipLaunchCooperativeKernel((void *)(coopKernel_3D_Grid_z_sleep),
                                         gridDim, blockDim, params, 0, 0));
    HIP_CHECK(hipDeviceSynchronize());
  }

  ///////////////// Case 3D grid  - Hold all threads in y & z directions
  {
    constexpr int N = 2;
    dim3 gridDim = dim3{N, N, N};
    dim3 blockDim = dim3{1, 1, 1};

    uint32_t interval = delay.count();
    void *params[2];
    params[0] = &interval;
    params[1] = &ticks_per_ms;

    HIP_CHECK(hipLaunchCooperativeKernel((void *)(coopKernel_3D_Grid_yz_sleep),
                                         gridDim, blockDim, params, 0, 0));
    HIP_CHECK(hipDeviceSynchronize());
  }
}

static __global__ void coopKernel_1D_Block_sleep(uint32_t interval,
                                                 const uint32_t ticks_per_ms) {
  printf("gridDim.x = %d , gridDim.y = %d, gridDim.z = %d \n", gridDim.x,
         gridDim.y, gridDim.z);
  printf("blockDim.x = %d , blockDim.y = %d, blockDim.z = %d \n", blockDim.x,
         blockDim.y, blockDim.z);

  printf("blockIdx.z = %d , blockIdx.y = %d , blockIdx.x = %d \n", blockIdx.z,
         blockIdx.y, blockIdx.x);
  printf("threadIdx.z = %d , threadIdx.y = %d , threadIdx.x = %d \n",
         threadIdx.z, threadIdx.y, threadIdx.x);

  cooperative_groups::thread_block block =
      cooperative_groups::this_thread_block();

  // Hold all threads >= 2
  // if(threadIdx.x >= 2) {
  // if(threadIdx.x >= 120) { // for case 1
  if (threadIdx.x >= 30) { // for case 1

    printf("PreProcessing started in thread %d\n", threadIdx.x);

    while (interval--) {
      uint64_t start = clock64();
      while (clock64() - start < ticks_per_ms) {
        __builtin_amdgcn_s_sleep(10);
      }
    }
    printf("PreProcessing ended in thread %d\n", threadIdx.x);
  }

  auto token = block.barrier_arrive();

  printf("MainProcessing started in thread %d\n", threadIdx.x);

  printf("MainProcessing ended in thread %d\n", threadIdx.x);

  block.barrier_wait(std::move(token));

  printf("FinalProcessing Doing in thread %d\n", threadIdx.x);
}

static __global__ void coopKernel_2D_Block_sleep(uint32_t interval,
                                                 const uint32_t ticks_per_ms) {
  printf("gridDim.x = %d , gridDim.y = %d, gridDim.z = %d \n", gridDim.x,
         gridDim.y, gridDim.z);
  printf("blockDim.x = %d , blockDim.y = %d, blockDim.z = %d \n", blockDim.x,
         blockDim.y, blockDim.z);

  printf("blockIdx.z = %d , blockIdx.y = %d , blockIdx.x = %d \n", blockIdx.z,
         blockIdx.y, blockIdx.x);
  printf("threadIdx.z = %d , threadIdx.y = %d , threadIdx.x = %d \n",
         threadIdx.z, threadIdx.y, threadIdx.x);

  cooperative_groups::thread_block block =
      cooperative_groups::this_thread_block();

  // Hold all threads in y direction
  if (threadIdx.y >= 1) {
    printf("PreProcessing in started threadIdx.y = %d , threadIdx.x = %d \n",
           threadIdx.y, threadIdx.x);

    while (interval--) {
      uint64_t start = clock64();
      while (clock64() - start < ticks_per_ms) {
        __builtin_amdgcn_s_sleep(10);
      }
    }
    printf("PreProcessing ended in threadIdx.y = %d , threadIdx.x = %d \n",
           threadIdx.y, threadIdx.x);
  }

  auto token = block.barrier_arrive();

  printf("MainProcessing started in threadIdx.y = %d , threadIdx.x = %d \n",
         threadIdx.y, threadIdx.x);

  printf("MainProcessing ended in threadIdx.y = %d , threadIdx.x = %d \n",
         threadIdx.y, threadIdx.x);

  block.barrier_wait(std::move(token));

  printf("FinalProcessing Doing in threadIdx.y = %d , threadIdx.x = %d \n",
         threadIdx.y, threadIdx.x);
}

static __global__ void
coopKernel_3D_Block_z_sleep(uint32_t interval, const uint32_t ticks_per_ms) {
  printf("gridDim.x = %d , gridDim.y = %d, gridDim.z = %d \n", gridDim.x,
         gridDim.y, gridDim.z);
  printf("blockDim.x = %d , blockDim.y = %d, blockDim.z = %d \n", blockDim.x,
         blockDim.y, blockDim.z);

  printf("blockIdx.z = %d , blockIdx.y = %d , blockIdx.x = %d \n", blockIdx.z,
         blockIdx.y, blockIdx.x);
  printf("threadIdx.z = %d , threadIdx.y = %d , threadIdx.x = %d \n",
         threadIdx.z, threadIdx.y, threadIdx.x);

  cooperative_groups::thread_block block =
      cooperative_groups::this_thread_block();

  // Hold all threads in z direction
  if (threadIdx.z >= 1) {
    printf("PreProcessing in started threadIdx.z = %d , threadIdx.y = %d , "
           "threadIdx.x = %d \n",
           threadIdx.z, threadIdx.y, threadIdx.x);

    while (interval--) {
      uint64_t start = clock64();
      while (clock64() - start < ticks_per_ms) {
        __builtin_amdgcn_s_sleep(10);
      }
    }
    printf("PreProcessing ended in threadIdx.z = %d , threadIdx.y = %d , "
           "threadIdx.x = %d \n",
           threadIdx.z, threadIdx.y, threadIdx.x);
  }

  auto token = block.barrier_arrive();

  printf("MainProcessing started in threadIdx.z = %d , threadIdx.y = %d , "
         "threadIdx.x = %d \n",
         threadIdx.z, threadIdx.y, threadIdx.x);

  printf("MainProcessing ended in threadIdx.z = %d , threadIdx.y = %d , "
         "threadIdx.x = %d \n",
         threadIdx.z, threadIdx.y, threadIdx.x);

  block.barrier_wait(std::move(token));

  printf("FinalProcessing Doing in threadIdx.z = %d , threadIdx.y = %d , "
         "threadIdx.x = %d \n",
         threadIdx.z, threadIdx.y, threadIdx.x);
}

static __global__ void
coopKernel_3D_Block_yz_sleep(uint32_t interval, const uint32_t ticks_per_ms) {
  printf("gridDim.x = %d , gridDim.y = %d, gridDim.z = %d \n", gridDim.x,
         gridDim.y, gridDim.z);
  printf("blockDim.x = %d , blockDim.y = %d, blockDim.z = %d \n", blockDim.x,
         blockDim.y, blockDim.z);

  printf("blockIdx.z = %d , blockIdx.y = %d , blockIdx.x = %d \n", blockIdx.z,
         blockIdx.y, blockIdx.x);
  printf("threadIdx.z = %d , threadIdx.y = %d , threadIdx.x = %d \n",
         threadIdx.z, threadIdx.y, threadIdx.x);

  cooperative_groups::thread_block block =
      cooperative_groups::this_thread_block();

  // Hold all threads in y & z direction
  if (threadIdx.y >= 1 || threadIdx.z >= 1) {
    printf("PreProcessing in started threadIdx.z = %d , threadIdx.y = %d , "
           "threadIdx.x = %d \n",
           threadIdx.z, threadIdx.y, threadIdx.x);

    while (interval--) {
      uint64_t start = clock64();
      while (clock64() - start < ticks_per_ms) {
        __builtin_amdgcn_s_sleep(10);
      }
    }
    printf("PreProcessing ended in threadIdx.z = %d , threadIdx.y = %d , "
           "threadIdx.x = %d \n",
           threadIdx.z, threadIdx.y, threadIdx.x);
  }

  auto token = block.barrier_arrive();

  printf("MainProcessing started in threadIdx.z = %d , threadIdx.y = %d , "
         "threadIdx.x = %d \n",
         threadIdx.z, threadIdx.y, threadIdx.x);

  printf("MainProcessing ended in threadIdx.z = %d , threadIdx.y = %d , "
         "threadIdx.x = %d \n",
         threadIdx.z, threadIdx.y, threadIdx.x);

  block.barrier_wait(std::move(token));

  printf("FinalProcessing Doing in threadIdx.z = %d , threadIdx.y = %d , "
         "threadIdx.x = %d \n",
         threadIdx.z, threadIdx.y, threadIdx.x);
}

TEST_CASE("Unit_barrier_wait_barrier_arrive_Block_sleep") {

  int cooperativeLaunchSupported;
  HIP_CHECK(hipDeviceGetAttribute(&cooperativeLaunchSupported,
                                  hipDeviceAttributeCooperativeLaunch, 0));

  if (!cooperativeLaunchSupported) {
    HipTest::HIP_SKIP_TEST("Skipping test as CooperativeLaunch not supported");
  }

  int ticks_per_ms = 0;
  HIP_CHECK(
      hipDeviceGetAttribute(&ticks_per_ms, hipDeviceAttributeWallClockRate, 0));
  std::cout << "ticks_per_ms : " << ticks_per_ms << std::endl;

  // const std::chrono::duration<uint64_t, std::milli> delay = 1000000; // 100
  // sec
  std::chrono::milliseconds delay = std::chrono::milliseconds(1000000);

  std::cout << "delay : " << delay.count() << std::endl;

#if 1

  ///////////////// Case 1D grid - Hold all threads >= 2
  {
    constexpr int N = 128;
    dim3 gridDim = dim3{1, 1, 1};
    dim3 blockDim = dim3{N, 1, 1};

    uint32_t interval = delay.count();
    void *params[2];
    params[0] = &interval;
    params[1] = &ticks_per_ms;

    HIP_CHECK(hipLaunchCooperativeKernel((void *)(coopKernel_1D_Block_sleep),
                                         gridDim, blockDim, params, 0, 0));
    HIP_CHECK(hipDeviceSynchronize());
  }

  ///////////////// Case 2D grid  - Hold all threads in y direction
  {
    constexpr int N = 128;
    dim3 gridDim = dim3{1, 1, 1};
    dim3 blockDim = dim3{N, 2, 1};

    uint32_t interval = delay.count();
    void *params[2];
    params[0] = &interval;
    params[1] = &ticks_per_ms;

    HIP_CHECK(hipLaunchCooperativeKernel((void *)(coopKernel_2D_Block_sleep),
                                         gridDim, blockDim, params, 0, 0));
    HIP_CHECK(hipDeviceSynchronize());
  }

  ///////////////// Case 3D grid  - Hold all threads in z directions
  {
    constexpr int N = 2;
    dim3 gridDim = dim3{1, 1, 1};
    dim3 blockDim = dim3{N, N, N};

    uint32_t interval = delay.count();
    void *params[2];
    params[0] = &interval;
    params[1] = &ticks_per_ms;

    HIP_CHECK(hipLaunchCooperativeKernel((void *)(coopKernel_3D_Block_z_sleep),
                                         gridDim, blockDim, params, 0, 0));
    HIP_CHECK(hipDeviceSynchronize());
  }

  ///////////////// Case 3D grid  - Hold all threads in y & z directions
  {
    constexpr int N = 2;
    dim3 gridDim = dim3{1, 1, 1};
    dim3 blockDim = dim3{N, N, N};

    uint32_t interval = delay.count();
    void *params[2];
    params[0] = &interval;
    params[1] = &ticks_per_ms;

    HIP_CHECK(hipLaunchCooperativeKernel((void *)(coopKernel_3D_Block_yz_sleep),
                                         gridDim, blockDim, params, 0, 0));
    HIP_CHECK(hipDeviceSynchronize());
  }

#endif

  hipDeviceProp_t deviceProp;
  HIP_CHECK(hipGetDeviceProperties(&deviceProp, 0));

  std::cout << "Device " << 0 << ": " << deviceProp.name << std::endl;
  std::cout << "  Warp Size: " << deviceProp.warpSize << std::endl;

  ///////////////// Case 1D grid - Hold all threads >= 2
  {
    constexpr int N = 32;
    dim3 gridDim = dim3{2, 1, 1};
    dim3 blockDim = dim3{N, 1, 1};

    uint32_t interval = delay.count();
    void *params[2];
    params[0] = &interval;
    params[1] = &ticks_per_ms;

    HIP_CHECK(hipLaunchCooperativeKernel((void *)(coopKernel_1D_Block_sleep),
                                         gridDim, blockDim, params, 0, 0));
    HIP_CHECK(hipDeviceSynchronize());
  }
}

static constexpr int N = 1024;
static __device__ int devArr[N];

static __device__ void fillData(int *local_data) {
  for (int i = 0; i < N; i++) {
    local_data[i] = i + 1;
  }
}

static __device__ void elementWiseDouble(int *local_data) {
  for (int i = 0; i < N; i++) {
    local_data[i] = local_data[i] + local_data[i];
  }
}

static __device__ int localDataSum(int *local_data) {
  int sum = 0;
  for (int i = 0; i < N; i++) {
    sum += local_data[i];
  }
  return sum;
}

static __device__ void fillData_WF(int *local_data, int N) {
  for (int i = 0; i < N; i++) {
    local_data[i] = i + 1;
  }
}

static __device__ void elementWiseDouble_WF(int *local_data, int N) {
  for (int i = 0; i < N; i++) {
    local_data[i] = local_data[i] + local_data[i];
  }
}

static __device__ int localDataSum_WF(int *local_data, int N) {
  int sum = 0;
  for (int i = 0; i < N; i++) {
    sum += local_data[i];
  }
  return sum;
}

static __global__ void coopKernel_1D_Grid_data(int *output) {
  printf("gridDim.x = %d , gridDim.y = %d, gridDim.z = %d \n", gridDim.x,
         gridDim.y, gridDim.z);
  printf("blockIdx.z = %d , blockIdx.y = %d , blockIdx.x = %d \n", blockIdx.z,
         blockIdx.y, blockIdx.x);

  cooperative_groups::grid_group grid = cooperative_groups::this_grid();

  if (blockIdx.x >= 1000) {
    printf("PreProcessing started in block %d\n", blockIdx.x);
    devArr[blockIdx.x] = 100;
    printf("PreProcessing ended in block %d\n", blockIdx.x);
  }
  auto token = grid.barrier_arrive();

  printf("MainProcessing started in block %d\n", blockIdx.x);

  int sum = 0;
  int local_data[N];
  fillData(local_data);
  elementWiseDouble(local_data);
  sum = localDataSum(local_data);

  printf("MainProcessing ended in block %d\n", blockIdx.x);

  grid.barrier_wait(std::move(token));

  printf("FinalProcessing Doing in block %d\n", blockIdx.x);

  int devArrSum = 0;
  for (int i = 0; i < N; i++) {
    devArrSum = devArrSum + devArr[i];
  }
  output[blockIdx.x] = sum + devArrSum;
}

static __global__ void coopKernel_2D_Grid_data(int *output) {
  printf("gridDim.x = %d , gridDim.y = %d, gridDim.z = %d \n", gridDim.x,
         gridDim.y, gridDim.z);
  printf("blockIdx.z = %d , blockIdx.y = %d , blockIdx.x = %d \n", blockIdx.z,
         blockIdx.y, blockIdx.x);

  int idx = gridDim.x * blockIdx.y + blockIdx.x;
  printf("idx = %d \n", idx);

  cooperative_groups::grid_group grid = cooperative_groups::this_grid();

  // Hold all threads in y direction
  if (blockIdx.y >= 1) {
    printf("PreProcessing in started blockIdx.y = %d , blockIdx.x = %d \n",
           blockIdx.y, blockIdx.x);
    devArr[idx] = 100;
    printf("PreProcessing ended in blockIdx.y = %d , blockIdx.x = %d \n",
           blockIdx.y, blockIdx.x);
  }

  auto token = grid.barrier_arrive();

  printf("MainProcessing started in blockIdx.y = %d , blockIdx.x = %d \n",
         blockIdx.y, blockIdx.x);

  int sum = 0;
  int local_data[N];
  fillData(local_data);
  elementWiseDouble(local_data);
  sum = localDataSum(local_data);

  printf("MainProcessing ended in blockIdx.y = %d , blockIdx.x = %d \n",
         blockIdx.y, blockIdx.x);

  grid.barrier_wait(std::move(token));

  printf("FinalProcessing Doing in blockIdx.y = %d , blockIdx.x = %d \n",
         blockIdx.y, blockIdx.x);

  int devArrSum = 0;
  for (int i = 0; i < N; i++) {
    devArrSum = devArrSum + devArr[i];
  }
  output[idx] = sum + devArrSum;
}

static __global__ void coopKernel_3D_Grid_z_data(int *output) {
  printf("gridDim.x = %d , gridDim.y = %d, gridDim.z = %d \n", gridDim.x,
         gridDim.y, gridDim.z);
  printf("blockIdx.z = %d , blockIdx.y = %d , blockIdx.x = %d \n", blockIdx.z,
         blockIdx.y, blockIdx.x);

  int idx =
      gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x;

  printf("idx = %d \n", idx);

  cooperative_groups::grid_group grid = cooperative_groups::this_grid();

  // Hold all threads in z direction
  if (blockIdx.z >= 1) {

    printf("PreProcessing in started blockIdx.z = %d , blockIdx.y = %d , "
           "blockIdx.x = %d \n",
           blockIdx.z, blockIdx.y, blockIdx.x);

    for (int i = 0; i < N; i++) {
      int local_data_[N];
      fillData(local_data_);
      elementWiseDouble(local_data_);
      int sum_ = localDataSum(local_data_);
      sum_ += sum_;
    }

    devArr[idx] = 100;

    printf("PreProcessing ended in blockIdx.z = %d , blockIdx.y = %d , "
           "blockIdx.x = %d \n",
           blockIdx.z, blockIdx.y, blockIdx.x);
  }

  auto token = grid.barrier_arrive();

  printf("MainProcessing started in blockIdx.z = %d , blockIdx.y = %d , "
         "blockIdx.x = %d \n",
         blockIdx.z, blockIdx.y, blockIdx.x);

  int sum = 0;
  int local_data[N];
  fillData(local_data);
  elementWiseDouble(local_data);
  sum = localDataSum(local_data);

  printf("MainProcessing ended in blockIdx.z = %d , blockIdx.y = %d , "
         "blockIdx.x = %d \n",
         blockIdx.z, blockIdx.y, blockIdx.x);

  grid.barrier_wait(std::move(token));

  printf("FinalProcessing Doing in blockIdx.z = %d , blockIdx.y = %d , "
         "blockIdx.x = %d \n",
         blockIdx.z, blockIdx.y, blockIdx.x);

  int devArrSum = 0;
  for (int i = 0; i < N; i++) {
    devArrSum = devArrSum + devArr[i];
  }
  output[idx] = sum + devArrSum;
}

static __global__ void coopKernel_3D_Grid_yz_data(int *output) {
  printf("gridDim.x = %d , gridDim.y = %d, gridDim.z = %d \n", gridDim.x,
         gridDim.y, gridDim.z);
  printf("blockIdx.z = %d , blockIdx.y = %d , blockIdx.x = %d \n", blockIdx.z,
         blockIdx.y, blockIdx.x);

  int idx =
      gridDim.x * gridDim.y * blockIdx.z + blockIdx.y * gridDim.x + blockIdx.x;

  printf("idx = %d \n", idx);

  cooperative_groups::grid_group grid = cooperative_groups::this_grid();

  // Hold all threads in y and z direction
  if (blockIdx.y >= 1 || blockIdx.z >= 1) {

    printf("PreProcessing in started blockIdx.z = %d , blockIdx.y = %d , "
           "blockIdx.x = %d \n",
           blockIdx.z, blockIdx.y, blockIdx.x);

    for (int i = 0; i < N; i++) {
      int local_data_[N];
      fillData(local_data_);
      elementWiseDouble(local_data_);
      int sum_ = localDataSum(local_data_);
      sum_ += sum_;
    }

    devArr[idx] = 100;

    printf("PreProcessing ended in blockIdx.z = %d , blockIdx.y = %d , "
           "blockIdx.x = %d \n",
           blockIdx.z, blockIdx.y, blockIdx.x);
  }

  auto token = grid.barrier_arrive();

  printf("MainProcessing started in blockIdx.z = %d , blockIdx.y = %d , "
         "blockIdx.x = %d \n",
         blockIdx.z, blockIdx.y, blockIdx.x);

  int sum = 0;
  int local_data[N];
  fillData(local_data);
  elementWiseDouble(local_data);
  sum = localDataSum(local_data);

  printf("MainProcessing ended in blockIdx.z = %d , blockIdx.y = %d , "
         "blockIdx.x = %d \n",
         blockIdx.z, blockIdx.y, blockIdx.x);

  grid.barrier_wait(std::move(token));

  printf("FinalProcessing Doing in blockIdx.z = %d , blockIdx.y = %d , "
         "blockIdx.x = %d \n",
         blockIdx.z, blockIdx.y, blockIdx.x);

  int devArrSum = 0;
  for (int i = 0; i < N; i++) {
    devArrSum = devArrSum + devArr[i];
  }
  output[idx] = sum + devArrSum;
}

TEST_CASE("Unit_barrier_wait_barrier_arrive_Grid_Data") {

  int cooperativeLaunchSupported;
  HIP_CHECK(hipDeviceGetAttribute(&cooperativeLaunchSupported,
                                  hipDeviceAttributeCooperativeLaunch, 0));

  if (!cooperativeLaunchSupported) {
    HipTest::HIP_SKIP_TEST("Skipping test as CooperativeLaunch not supported");
  }

  ///////////////// Case 1D grid - Hold all threads >= 1000
  {
    int hostMem[N];
    for (int i = 0; i < N; i++) {
      hostMem[i] = 0;
    }

    int *devMem = nullptr;
    HIP_CHECK(hipMalloc(&devMem, N * sizeof(int)));
    REQUIRE(devMem != nullptr);
    HIP_CHECK(
        hipMemcpy(devMem, hostMem, N * sizeof(int), hipMemcpyHostToDevice));

    dim3 gridDim = dim3{N, 1, 1};
    dim3 blockDim = dim3{1, 1, 1};

    void *params[1];
    params[0] = &devMem;

    HIP_CHECK(hipLaunchCooperativeKernel((void *)(coopKernel_1D_Grid_data),
                                         gridDim, blockDim, params, 0, 0));
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipMemcpy(hostMem, devMem, N * sizeof(int), hipMemcpyDefault));

    /*
    threads            0 1 2 ... 1022 1023
    fill data          1 2 3 ... 1023 1024
    elementWiseDouble  2 4 6 ... 2046 2048
    Sum of all Above numbers = Double of (N*N+1/2) 1024*1025/2 = Double of
524800 = 1049600 In kernel 100 will be there for 24 elements 2400 Final result =
1049600 + 2400 = 1052000
    */

    int expectedResult = 1052000;

    for (int i = 0; i < N; i++) {
      if (hostMem[i] != expectedResult) {
        std::cout << " i = " << i << "hostMem : " << hostMem[i] << std::endl;
        REQUIRE(false);
      }
    }

    HIP_CHECK(hipFree(devMem));
  }

  ///////////////// Case 2D grid  - Hold all threads in y direction
  {
    int hostMem[N];
    for (int i = 0; i < N; i++) {
      hostMem[i] = 0;
    }

    int *devMem = nullptr;
    HIP_CHECK(hipMalloc(&devMem, N * sizeof(int)));
    REQUIRE(devMem != nullptr);
    HIP_CHECK(
        hipMemcpy(devMem, hostMem, N * sizeof(int), hipMemcpyHostToDevice));

    dim3 gridDim = dim3{N / 2, N / 512, 1}; // {512, 2, 1}
    dim3 blockDim = dim3{1, 1, 1};

    void *params[1];
    params[0] = &devMem;

    HIP_CHECK(hipLaunchCooperativeKernel((void *)(coopKernel_2D_Grid_data),
                                         gridDim, blockDim, params, 0, 0));
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipMemcpy(hostMem, devMem, N * sizeof(int), hipMemcpyDefault));

    /*
    Threads            0 1 2 ... 1022 1023
    fill data          1 2 3 ... 1023 1024
    elementWiseDouble  2 4 6 ... 2046 2048
    Sum of all Above numbers = Double of (N*N+1/2) 1024*1025/2 = Double of
524800 = 1049600 In kernel 100 will be there for blocks in y dimention (512) =
51200 Final result = 	1049600 + 51200 = 1100800
    */

    int expectedResult = 1100800;

    for (int i = 0; i < N; i++) {
      if (hostMem[i] != expectedResult) {
        std::cout << " i = " << i << "hostMem : " << hostMem[i] << std::endl;
        REQUIRE(false);
      }
    }

    HIP_CHECK(hipFree(devMem));
  }

  ///////////////// Case 3D grid  - Hold all threads in z directions
  {
    int hostMem[N];
    for (int i = 0; i < N; i++) {
      hostMem[i] = 0;
    }

    int *devMem = nullptr;
    HIP_CHECK(hipMalloc(&devMem, N * sizeof(int)));
    REQUIRE(devMem != nullptr);
    HIP_CHECK(
        hipMemcpy(devMem, hostMem, N * sizeof(int), hipMemcpyHostToDevice));

    dim3 gridDim = dim3{N / 512, N / 256, N / 8}; // {2, 4, 128}
    dim3 blockDim = dim3{1, 1, 1};

    void *params[1];
    params[0] = &devMem;

    HIP_CHECK(hipLaunchCooperativeKernel((void *)(coopKernel_3D_Grid_z_data),
                                         gridDim, blockDim, params, 0, 0));
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipMemcpy(hostMem, devMem, N * sizeof(int), hipMemcpyDefault));

    /*
    Threads            0 1 2 ... 1022 1023
    fill data          1 2 3 ... 1023 1024
    elementWiseDouble  2 4 6 ... 2046 2048
    Sum of all Above numbers = Double of (N*N+1/2) 1024*1025/2 = Double of
524800 = 1049600 In kernel 100 will be there for blocks in z dimention > 1
(1024-8 = 1016) = 101600 Final result = 	1049600 + 101600 = 1151200
    */

    int expectedResult = 1151200;

    for (int i = 0; i < N; i++) {
      if (hostMem[i] != expectedResult) {
        std::cout << " i = " << i << "hostMem : " << hostMem[i] << std::endl;
        REQUIRE(false);
      }
    }

    HIP_CHECK(hipFree(devMem));
  }

  ///////////////// Case 3D grid  - Hold all threads in y & z directions
  {
    int hostMem[N];
    for (int i = 0; i < N; i++) {
      hostMem[i] = 0;
    }

    int *devMem = nullptr;
    HIP_CHECK(hipMalloc(&devMem, N * sizeof(int)));
    REQUIRE(devMem != nullptr);
    HIP_CHECK(
        hipMemcpy(devMem, hostMem, N * sizeof(int), hipMemcpyHostToDevice));

    dim3 gridDim = dim3{N / 512, N / 256, N / 8}; // {2, 4, 128}
    dim3 blockDim = dim3{1, 1, 1};

    void *params[1];
    params[0] = &devMem;

    HIP_CHECK(hipLaunchCooperativeKernel((void *)(coopKernel_3D_Grid_yz_data),
                                         gridDim, blockDim, params, 0, 0));
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipMemcpy(hostMem, devMem, N * sizeof(int), hipMemcpyDefault));

    /*
    Threads            0 1 2 ... 1022 1023
    fill data          1 2 3 ... 1023 1024
    elementWiseDouble  2 4 6 ... 2046 2048
    Sum of all Above numbers = Double of (N*N+1/2) 1024*1025/2 = Double of
524800 = 1049600 In kernel 100 will be there for blocks in y dimention > 1 & y
dimention > 1 (1024-2 = 1022) = 102200 Final result = 	1049600 + 102200 =
1151800
    */

    int expectedResult = 1151800;

    for (int i = 0; i < N; i++) {
      if (hostMem[i] != expectedResult) {
        std::cout << " i = " << i << "hostMem : " << hostMem[i] << std::endl;
        REQUIRE(false);
      }
    }

    HIP_CHECK(hipFree(devMem));
  }

  std::cout << " ==== Program END" << std::endl;
}

static __global__ void coopKernel_1D_Block_data(int *output) {
  printf("gridDim.x = %d , gridDim.y = %d, gridDim.z = %d \n", gridDim.x,
         gridDim.y, gridDim.z);
  printf("blockDim.x = %d , blockDim.y = %d, blockDim.z = %d \n", blockDim.x,
         blockDim.y, blockDim.z);

  printf("blockIdx.z = %d , blockIdx.y = %d , blockIdx.x = %d \n", blockIdx.z,
         blockIdx.y, blockIdx.x);
  printf("threadIdx.z = %d , threadIdx.y = %d , threadIdx.x = %d \n",
         threadIdx.z, threadIdx.y, threadIdx.x);

  cooperative_groups::thread_block block =
      cooperative_groups::this_thread_block();

  if (threadIdx.x >= 1000) {
    printf("PreProcessing started in block %d\n", threadIdx.x);
    devArr[threadIdx.x] = 100;
    printf("PreProcessing ended in block %d\n", threadIdx.x);
  }
  auto token = block.barrier_arrive();

  int sum = 0;

  printf("MainProcessing started in block %d\n", threadIdx.x);

  int local_data[N];
  fillData(local_data);
  elementWiseDouble(local_data);
  sum = localDataSum(local_data);

  printf("MainProcessing ended in block %d\n", threadIdx.x);

  block.barrier_wait(std::move(token));

  printf("FinalProcessing Doing in block %d\n", threadIdx.x);

  int devArrSum = 0;
  for (int i = 0; i < N; i++) {
    devArrSum = devArrSum + devArr[i];
  }
  output[threadIdx.x] = sum + devArrSum;
}

static __global__ void coopKernel_2D_Block_data(int *output) {
  printf("gridDim.x = %d , gridDim.y = %d, gridDim.z = %d \n", gridDim.x,
         gridDim.y, gridDim.z);
  printf("blockDim.x = %d , blockDim.y = %d, blockDim.z = %d \n", blockDim.x,
         blockDim.y, blockDim.z);

  printf("blockIdx.z = %d , blockIdx.y = %d , blockIdx.x = %d \n", blockIdx.z,
         blockIdx.y, blockIdx.x);
  printf("threadIdx.z = %d , threadIdx.y = %d , threadIdx.x = %d \n",
         threadIdx.z, threadIdx.y, threadIdx.x);

  int idx = blockDim.x * threadIdx.y + threadIdx.x;
  printf("idx = %d \n", idx);

  cooperative_groups::thread_block block =
      cooperative_groups::this_thread_block();

  // Hold all threads in y direction
  if (threadIdx.y >= 1) {
    printf("PreProcessing in started threadIdx.y = %d , threadIdx.x = %d \n",
           threadIdx.y, threadIdx.x);
    devArr[idx] = 100;
    printf("PreProcessing ended in threadIdx.y = %d , threadIdx.x = %d \n",
           threadIdx.y, threadIdx.x);
  }

  auto token = block.barrier_arrive();

  printf("MainProcessing started in threadIdx.y = %d , threadIdx.x = %d \n",
         threadIdx.y, threadIdx.x);

  int sum = 0;
  int local_data[N];
  fillData(local_data);
  elementWiseDouble(local_data);
  sum = localDataSum(local_data);

  printf("MainProcessing ended in threadIdx.y = %d , threadIdx.x = %d \n",
         threadIdx.y, threadIdx.x);

  block.barrier_wait(std::move(token));

  printf("FinalProcessing Doing in threadIdx.y = %d , threadIdx.x = %d \n",
         threadIdx.y, threadIdx.x);

  int devArrSum = 0;
  for (int i = 0; i < N; i++) {
    devArrSum = devArrSum + devArr[i];
  }
  output[idx] = sum + devArrSum;
}

static __global__ void coopKernel_3D_Block_z_data(int *output) {
  printf("gridDim.x = %d , gridDim.y = %d, gridDim.z = %d \n", gridDim.x,
         gridDim.y, gridDim.z);
  printf("blockDim.x = %d , blockDim.y = %d, blockDim.z = %d \n", blockDim.x,
         blockDim.y, blockDim.z);

  printf("blockIdx.z = %d , blockIdx.y = %d , blockIdx.x = %d \n", blockIdx.z,
         blockIdx.y, blockIdx.x);
  printf("threadIdx.z = %d , threadIdx.y = %d , threadIdx.x = %d \n",
         threadIdx.z, threadIdx.y, threadIdx.x);

  int idx = blockDim.x * blockDim.y * threadIdx.z + threadIdx.y * blockDim.x +
            threadIdx.x;

  printf("idx = %d \n", idx);

  cooperative_groups::thread_block block =
      cooperative_groups::this_thread_block();

  // Hold all threads in z direction
  if (threadIdx.z >= 1) {

    printf("PreProcessing in started threadIdx.z = %d , threadIdx.y = %d , "
           "threadIdx.x = %d \n",
           threadIdx.z, threadIdx.y, threadIdx.x);

    devArr[idx] = 100;

    printf("PreProcessing ended in threadIdx.z = %d , threadIdx.y = %d , "
           "threadIdx.x = %d \n",
           threadIdx.z, threadIdx.y, threadIdx.x);
  }

  auto token = block.barrier_arrive();

  printf("MainProcessing started in threadIdx.z = %d , threadIdx.y = %d , "
         "threadIdx.x = %d \n",
         threadIdx.z, threadIdx.y, threadIdx.x);

  int sum = 0;
  int local_data[N];
  fillData(local_data);
  elementWiseDouble(local_data);
  sum = localDataSum(local_data);

  printf("MainProcessing ended in threadIdx.z = %d , threadIdx.y = %d , "
         "threadIdx.x = %d \n",
         threadIdx.z, threadIdx.y, threadIdx.x);

  block.barrier_wait(std::move(token));

  printf("FinalProcessing Doing in threadIdx.z = %d , threadIdx.y = %d , "
         "threadIdx.x = %d \n",
         threadIdx.z, threadIdx.y, threadIdx.x);

  int devArrSum = 0;
  for (int i = 0; i < N; i++) {
    devArrSum = devArrSum + devArr[i];
  }
  output[idx] = sum + devArrSum;
}

static __global__ void coopKernel_3D_Block_yz_data(int *output) {
  printf("gridDim.x = %d , gridDim.y = %d, gridDim.z = %d \n", gridDim.x,
         gridDim.y, gridDim.z);
  printf("blockDim.x = %d , blockDim.y = %d, blockDim.z = %d \n", blockDim.x,
         blockDim.y, blockDim.z);

  printf("blockIdx.z = %d , blockIdx.y = %d , blockIdx.x = %d \n", blockIdx.z,
         blockIdx.y, blockIdx.x);
  printf("threadIdx.z = %d , threadIdx.y = %d , threadIdx.x = %d \n",
         threadIdx.z, threadIdx.y, threadIdx.x);

  int idx = blockDim.x * blockDim.y * threadIdx.z + threadIdx.y * blockDim.x +
            threadIdx.x;

  printf("idx = %d \n", idx);

  cooperative_groups::thread_block block =
      cooperative_groups::this_thread_block();

  // Hold all threads in y and z direction
  if (threadIdx.y >= 1 || threadIdx.z >= 1) {

    printf("PreProcessing in started threadIdx.z = %d , threadIdx.y = %d , "
           "threadIdx.x = %d \n",
           threadIdx.z, threadIdx.y, threadIdx.x);

    devArr[idx] = 100;

    printf("PreProcessing ended in threadIdx.z = %d , threadIdx.y = %d , "
           "threadIdx.x = %d \n",
           threadIdx.z, threadIdx.y, threadIdx.x);
  }

  auto token = block.barrier_arrive();

  printf("MainProcessing started in threadIdx.z = %d , threadIdx.y = %d , "
         "threadIdx.x = %d \n",
         threadIdx.z, threadIdx.y, threadIdx.x);

  int sum = 0;
  int local_data[N];
  fillData(local_data);
  elementWiseDouble(local_data);
  sum = localDataSum(local_data);

  printf("MainProcessing ended in threadIdx.z = %d , threadIdx.y = %d , "
         "threadIdx.x = %d \n",
         threadIdx.z, threadIdx.y, threadIdx.x);

  block.barrier_wait(std::move(token));

  printf("FinalProcessing Doing in threadIdx.z = %d , threadIdx.y = %d , "
         "threadIdx.x = %d \n",
         threadIdx.z, threadIdx.y, threadIdx.x);

  int devArrSum = 0;
  for (int i = 0; i < N; i++) {
    devArrSum = devArrSum + devArr[i];
  }
  output[idx] = sum + devArrSum;
}

static __global__ void coopKernel_1D_Block_2_warp(int *output, int N) {
  printf("gridDim.x = %d , gridDim.y = %d, gridDim.z = %d \n", gridDim.x,
         gridDim.y, gridDim.z);
  printf("blockDim.x = %d , blockDim.y = %d, blockDim.z = %d \n", blockDim.x,
         blockDim.y, blockDim.z);

  printf("blockIdx.z = %d , blockIdx.y = %d , blockIdx.x = %d \n", blockIdx.z,
         blockIdx.y, blockIdx.x);
  printf("threadIdx.z = %d , threadIdx.y = %d , threadIdx.x = %d \n",
         threadIdx.z, threadIdx.y, threadIdx.x);

  cooperative_groups::thread_block block =
      cooperative_groups::this_thread_block();
  auto gid = blockDim.x * blockIdx.x + threadIdx.x;

  __shared__ int dev_shared_mem[32];

  if (threadIdx.x >= 30) {
    printf("PreProcessing started in block %d\n", threadIdx.x);
    // devArr[threadIdx.x] = 100;
    dev_shared_mem[threadIdx.x] = 100 + blockIdx.x;
    printf("PreProcessing ended in block %d\n", threadIdx.x);
  }
  auto token = block.barrier_arrive();

  int sum = 0;

  printf("MainProcessing started in block %d\n", threadIdx.x);

  int *local_data = new int[N];
  fillData_WF(local_data, N);
  elementWiseDouble_WF(local_data, N);
  sum = localDataSum_WF(local_data, N);

  printf("MainProcessing ended in block %d\n", threadIdx.x);

  block.barrier_wait(std::move(token));

  printf("FinalProcessing Doing in block %d\n", threadIdx.x);

  int devArrSum = 0;
  for (int i = 0; i < N; i++) {
    devArrSum = devArrSum + dev_shared_mem[i];
  }
  // output[threadIdx.x] = sum + devArrSum;
  output[gid] = sum + devArrSum;
  delete[] local_data;
}

TEST_CASE("Unit_barrier_wait_barrier_arrive_Block_Data") {

  int cooperativeLaunchSupported;
  HIP_CHECK(hipDeviceGetAttribute(&cooperativeLaunchSupported,
                                  hipDeviceAttributeCooperativeLaunch, 0));

  if (!cooperativeLaunchSupported) {
    HipTest::HIP_SKIP_TEST("Skipping test as CooperativeLaunch not supported");
  }

#if 1

  ///////////////// Case 1D grid - Hold all threads >= 1000
  {
    int hostMem[N];
    for (int i = 0; i < N; i++) {
      hostMem[i] = 0;
    }

    int *devMem = nullptr;
    HIP_CHECK(hipMalloc(&devMem, N * sizeof(int)));
    REQUIRE(devMem != nullptr);
    HIP_CHECK(
        hipMemcpy(devMem, hostMem, N * sizeof(int), hipMemcpyHostToDevice));

    dim3 gridDim = dim3{1, 1, 1};
    dim3 blockDim = dim3{N, 1, 1};

    void *params[1];
    params[0] = &devMem;

    HIP_CHECK(hipLaunchCooperativeKernel((void *)(coopKernel_1D_Block_data),
                                         gridDim, blockDim, params, 0, 0));
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipMemcpy(hostMem, devMem, N * sizeof(int), hipMemcpyDefault));

    /*
    threads            0 1 2 ... 1022 1023
    fill data          1 2 3 ... 1023 1024
    elementWiseDouble  2 4 6 ... 2046 2048
    Sum of all Above numbers = Double of (N*N+1/2) 1024*1025/2 = Double of
524800 = 1049600 In kernel 100 will be there for 24 elements 2400 Final result =
1049600 + 2400 = 1052000
    */

    int expectedResult = 1052000;

    for (int i = 0; i < N; i++) {
      if (hostMem[i] != expectedResult) {
        std::cout << " i = " << i << "hostMem : " << hostMem[i] << std::endl;
        REQUIRE(false);
      }
    }

    HIP_CHECK(hipFree(devMem));
  }

  ///////////////// Case 2D grid  - Hold all threads in y direction
  {
    int hostMem[N];
    for (int i = 0; i < N; i++) {
      hostMem[i] = 0;
    }

    int *devMem = nullptr;
    HIP_CHECK(hipMalloc(&devMem, N * sizeof(int)));
    REQUIRE(devMem != nullptr);
    HIP_CHECK(
        hipMemcpy(devMem, hostMem, N * sizeof(int), hipMemcpyHostToDevice));

    dim3 gridDim = dim3{1, 1, 1};
    dim3 blockDim = dim3{N / 2, N / 512, 1}; // {512, 2, 1}

    void *params[1];
    params[0] = &devMem;

    HIP_CHECK(hipLaunchCooperativeKernel((void *)(coopKernel_2D_Block_data),
                                         gridDim, blockDim, params, 0, 0));
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipMemcpy(hostMem, devMem, N * sizeof(int), hipMemcpyDefault));

    /*
    Threads            0 1 2 ... 1022 1023
    fill data          1 2 3 ... 1023 1024
    elementWiseDouble  2 4 6 ... 2046 2048
    Sum of all Above numbers = Double of (N*N+1/2) 1024*1025/2 = Double of
524800 = 1049600 In kernel 100 will be there for blocks in y dimention (512) =
51200 Final result = 	1049600 + 51200 = 1100800
    */

    int expectedResult = 1100800;

    for (int i = 0; i < N; i++) {
      if (hostMem[i] != expectedResult) {
        std::cout << " i = " << i << "hostMem : " << hostMem[i] << std::endl;
        REQUIRE(false);
      }
    }

    HIP_CHECK(hipFree(devMem));
  }

  ///////////////// Case 3D grid  - Hold all threads in z directions
  {
    int hostMem[N];
    for (int i = 0; i < N; i++) {
      hostMem[i] = 0;
    }

    int *devMem = nullptr;
    HIP_CHECK(hipMalloc(&devMem, N * sizeof(int)));
    REQUIRE(devMem != nullptr);
    HIP_CHECK(
        hipMemcpy(devMem, hostMem, N * sizeof(int), hipMemcpyHostToDevice));

    dim3 gridDim = dim3{1, 1, 1};
    dim3 blockDim = dim3{N / 512, N / 256, N / 8}; // {2, 4, 128}

    void *params[1];
    params[0] = &devMem;

    HIP_CHECK(hipLaunchCooperativeKernel((void *)(coopKernel_3D_Block_z_data),
                                         gridDim, blockDim, params, 0, 0));
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipMemcpy(hostMem, devMem, N * sizeof(int), hipMemcpyDefault));

    /*
    Threads            0 1 2 ... 1022 1023
    fill data          1 2 3 ... 1023 1024
    elementWiseDouble  2 4 6 ... 2046 2048
    Sum of all Above numbers = Double of (N*N+1/2) 1024*1025/2 = Double of
524800 = 1049600 In kernel 100 will be there for blocks in z dimention > 1
(1024-8 = 1016) = 101600 Final result = 	1049600 + 101600 = 1151200
    */

    int expectedResult = 1151200;

    for (int i = 0; i < N; i++) {
      if (hostMem[i] != expectedResult) {
        std::cout << " i = " << i << "hostMem : " << hostMem[i] << std::endl;
        REQUIRE(false);
      }
    }

    HIP_CHECK(hipFree(devMem));
  }

  ///////////////// Case 3D grid  - Hold all threads in y & z directions
  {
    int hostMem[N];
    for (int i = 0; i < N; i++) {
      hostMem[i] = 0;
    }

    int *devMem = nullptr;
    HIP_CHECK(hipMalloc(&devMem, N * sizeof(int)));
    REQUIRE(devMem != nullptr);
    HIP_CHECK(
        hipMemcpy(devMem, hostMem, N * sizeof(int), hipMemcpyHostToDevice));

    dim3 gridDim = dim3{1, 1, 1};
    dim3 blockDim = dim3{N / 512, N / 256, N / 8}; // {2, 4, 128}

    void *params[1];
    params[0] = &devMem;

    HIP_CHECK(hipLaunchCooperativeKernel((void *)(coopKernel_3D_Block_yz_data),
                                         gridDim, blockDim, params, 0, 0));
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipMemcpy(hostMem, devMem, N * sizeof(int), hipMemcpyDefault));

    /*
    Threads            0 1 2 ... 1022 1023
    fill data          1 2 3 ... 1023 1024
    elementWiseDouble  2 4 6 ... 2046 2048
    Sum of all Above numbers = Double of (N*N+1/2) 1024*1025/2 = Double of
524800 = 1049600 In kernel 100 will be there for blocks in y dimention > 1 & z
dimention > 1 (1024-2 = 1022) = 102200 Final result = 	1049600 + 102200 =
1151800
    */

    int expectedResult = 1151800;

    for (int i = 0; i < N; i++) {
      if (hostMem[i] != expectedResult) {
        std::cout << " i = " << i << "hostMem : " << hostMem[i] << std::endl;
        REQUIRE(false);
      }
    }

    HIP_CHECK(hipFree(devMem));
  }

#endif

  ///////////////// Case 1D block with warp size - Hold all threads >= 1000
  {
    // Get warp id
    hipDeviceProp_t deviceProp;
    HIP_CHECK(hipGetDeviceProperties(&deviceProp, 0));

    std::cout << "Device " << 0 << ": " << deviceProp.name << std::endl;
    std::cout << "Warp Size: " << deviceProp.warpSize << std::endl;

    int N = deviceProp.warpSize;

    std::vector<int> hostMem;
    for (int i = 0; i < N; i++) {
      hostMem.push_back(0);
    }

    std::cout << "At : " << __LINE__ << std::endl;

    int *devMem = nullptr;
    HIP_CHECK(hipMalloc(&devMem, N * sizeof(int)));
    REQUIRE(devMem != nullptr);
    HIP_CHECK(hipMemcpy(devMem, hostMem.data(), N * sizeof(int),
                        hipMemcpyHostToDevice));

    std::cout << "At : " << __LINE__ << std::endl;

    dim3 gridDim = dim3{2, 1, 1};
    dim3 blockDim = dim3{32, 1, 1}; // modify this later

    int N_half = N / 2;

    void *params[2];
    params[0] = &devMem;
    params[1] = &N_half;

    std::cout << "At : " << __LINE__ << std::endl;

    HIP_CHECK(hipLaunchCooperativeKernel((void *)(coopKernel_1D_Block_2_warp),
                                         gridDim, blockDim, params, 0, 0));
    HIP_CHECK(hipDeviceSynchronize());

    std::cout << "At : " << __LINE__ << std::endl;

    HIP_CHECK(
        hipMemcpy(hostMem.data(), devMem, N * sizeof(int), hipMemcpyDefault));

    /*
    threads            0 1 2 ... 29 30 31
    fill data          1 2 3 ... 30 31 32
    elementWiseDouble  2 4 6 ... 60 62 64
    Sum of all Above numbers = Double of (N*N+1/2) 32*33/2 = Double of 528 =
1056 In kernel 100 will be there for 2 elements 200 Final result = 	1056 +
200 = 1256
    */

    int expectedResult = 1256;

    for (int i = 0; i < N / 2; i++) {
      if (hostMem[i] != expectedResult) {
        std::cout << " i = " << i << "hostMem : " << hostMem[i] << std::endl;
        REQUIRE(false);
      }
    }
    expectedResult = 1258;
    for (int i = N / 2; i < N; i++) {
      if (hostMem[i] != expectedResult) {
        std::cout << " i = " << i << "hostMem : " << hostMem[i] << std::endl;
        REQUIRE(false);
      }
    }

    HIP_CHECK(hipFree(devMem));
  }

  std::cout << " ==== Program END" << std::endl;
}

static __global__ void
coopKernel_1D_Grid_1D_Block_sleep(uint32_t interval,
                                  const uint32_t ticks_per_ms) {
  printf("gridDim.x = %d , gridDim.y = %d, gridDim.z = %d \n", gridDim.x,
         gridDim.y, gridDim.z);
  printf("blockIdx.z = %d , blockIdx.y = %d , blockIdx.x = %d \n", blockIdx.z,
         blockIdx.y, blockIdx.x);

  cooperative_groups::grid_group grid = cooperative_groups::this_grid();

  if (blockIdx.x == 0) {
    printf("PreProcessing started in block %d\n", blockIdx.x);

    while (interval--) {
      uint64_t start = clock64();
      while (clock64() - start < ticks_per_ms) {
        __builtin_amdgcn_s_sleep(10);
      }
    }
    printf("PreProcessing ended in block %d\n", blockIdx.x);
  }

  auto token1 = grid.barrier_arrive();

  printf("MainProcessing started in block %d\n", blockIdx.x);

  if (blockIdx.x == 1) {
    cooperative_groups::thread_block block =
        cooperative_groups::this_thread_block();

    if (threadIdx.x >= 2) {
      printf("PreProcessing started in thread %d\n", threadIdx.x);

      while (interval--) {
        uint64_t start = clock64();
        while (clock64() - start < ticks_per_ms) {
          __builtin_amdgcn_s_sleep(10);
        }
      }
      printf("PreProcessing ended in thread %d\n", threadIdx.x);
    }

    auto token2 = block.barrier_arrive();

    printf("MainProcessing started in thread %d\n", threadIdx.x);

    printf("MainProcessing ended in thread %d\n", threadIdx.x);

    block.barrier_wait(std::move(token2));
  }

  printf("MainProcessing ended in block %d\n", blockIdx.x);

  grid.barrier_wait(std::move(token1));

  printf("FinalProcessing Doing in block %d\n", blockIdx.x);
}

TEST_CASE("Unit_barrier_wait_barrier_arrive_GridAndBlock_Sleep") {
  int cooperativeLaunchSupported;
  HIP_CHECK(hipDeviceGetAttribute(&cooperativeLaunchSupported,
                                  hipDeviceAttributeCooperativeLaunch, 0));

  if (!cooperativeLaunchSupported) {
    HipTest::HIP_SKIP_TEST("Skipping test as CooperativeLaunch not supported");
  }

  int ticks_per_ms = 0;
  HIP_CHECK(
      hipDeviceGetAttribute(&ticks_per_ms, hipDeviceAttributeWallClockRate, 0));
  std::cout << "ticks_per_ms : " << ticks_per_ms << std::endl;

  // const std::chrono::duration<uint64_t, std::milli> delay = 1000000; // 100
  // sec
  std::chrono::milliseconds delay = std::chrono::milliseconds(1000000);

  std::cout << "delay : " << delay.count() << std::endl;

  ///////////////// Case 1D grid
  {
    constexpr int N = 2;
    dim3 gridDim = dim3{N, 1, 1};
    dim3 blockDim = dim3{N * N, 1, 1};

    uint32_t interval = delay.count();
    void *params[2];
    params[0] = &interval;
    params[1] = &ticks_per_ms;

    HIP_CHECK(
        hipLaunchCooperativeKernel((void *)(coopKernel_1D_Grid_1D_Block_sleep),
                                   gridDim, blockDim, params, 0, 0));
    HIP_CHECK(hipDeviceSynchronize());
  }
}

static __global__ void coopKernel_1D_Grid_1D_Block_data(int *output) {
  printf("gridDim.x = %d , gridDim.y = %d, gridDim.z = %d \n", gridDim.x,
         gridDim.y, gridDim.z);
  printf("blockIdx.z = %d , blockIdx.y = %d , blockIdx.x = %d \n", blockIdx.z,
         blockIdx.y, blockIdx.x);

  cooperative_groups::grid_group grid = cooperative_groups::this_grid();

  int index = blockDim.x * blockIdx.x + threadIdx.x;

  printf("index = %d \n", index);

  int sum1 = 0;
  if (blockIdx.x == 1) {
    printf("PreProcessing started in block %d\n", blockIdx.x);
    devArr[index] = 100;
    printf("PreProcessing ended in block %d\n", blockIdx.x);
  }

  auto token1 = grid.barrier_arrive();

  printf("MainProcessing started in block %d\n", blockIdx.x);

  if (blockIdx.x == 0) {
    cooperative_groups::thread_block block =
        cooperative_groups::this_thread_block();

    if (threadIdx.x >= 256) {
      printf("PreProcessing started in thread %d\n", threadIdx.x);

      devArr[index] = 100;

      printf("PreProcessing ended in thread %d\n", threadIdx.x);
    }

    auto token2 = block.barrier_arrive();

    printf("MainProcessing started in thread %d\n", threadIdx.x);

    int local_data[N];
    fillData(local_data);
    sum1 = localDataSum(local_data);

    printf("MainProcessing ended in thread %d\n", threadIdx.x);

    block.barrier_wait(std::move(token2));
  }

  int sum2 = 0;
  int local_data[N];
  fillData(local_data);
  sum2 = localDataSum(local_data);
  int sum = sum1 + sum2;

  printf("MainProcessing ended in block %d\n", blockIdx.x);

  grid.barrier_wait(std::move(token1));

  printf("FinalProcessing Doing in block %d\n", blockIdx.x);

  int devArrSum = 0;
  for (int i = 0; i < N; i++) {
    devArrSum = devArrSum + devArr[i];
  }
  output[index] = sum + devArrSum;
}

TEST_CASE("Unit_barrier_wait_barrier_arrive_GridAndBlock_Data") {

  int cooperativeLaunchSupported;
  HIP_CHECK(hipDeviceGetAttribute(&cooperativeLaunchSupported,
                                  hipDeviceAttributeCooperativeLaunch, 0));

  if (!cooperativeLaunchSupported) {
    HipTest::HIP_SKIP_TEST("Skipping test as CooperativeLaunch not supported");
  }

  {
    int hostMem[N];
    for (int i = 0; i < N; i++) {
      hostMem[i] = 0;
    }

    int *devMem = nullptr;
    HIP_CHECK(hipMalloc(&devMem, N * sizeof(int)));
    REQUIRE(devMem != nullptr);
    HIP_CHECK(
        hipMemcpy(devMem, hostMem, N * sizeof(int), hipMemcpyHostToDevice));

    dim3 gridDim = dim3{2, 1, 1};
    dim3 blockDim = dim3{512, 1, 1};

    void *params[1];
    params[0] = &devMem;

    HIP_CHECK(
        hipLaunchCooperativeKernel((void *)(coopKernel_1D_Grid_1D_Block_data),
                                   gridDim, blockDim, params, 0, 0));
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipMemcpy(hostMem, devMem, N * sizeof(int), hipMemcpyDefault));

    /*
    threads                     0 1 2 ... 255 256 ... 511 512 513 ... 1022 1023
    fill data                   1 2 3 ... 256 257 ... 512 513 514 ... 1023 1024
                                <-----Block  0 -------------> <----- Block 1--->
                                            dev arr <---  0 ---->
<---100-------> <----- 100 ------> sum 1   <-------- 524800 -----------> <------
0 -------> sum 2   <-------- 524800 -----------> <---  524800 ----> sum
<-------- 1049600 ----------> <---- 524800 ----> output
<--------1126400------------> <---- 601600 ----> Notes : Sum of all Above
numbers = (N*N+1/2) 1024*1025/2 = 524800 In kernel, 100 will be there for all
second block and half of threads in first block (512+256 = 768) = 76800 Final result = 	1049600 + 76800 = 1126400  <--- 0 to 511 524800  + 76800 =
601600  <--- 512 to 1023
    */

    int expectedResult_1 = 1126400;
    int expectedResult_2 = 601600;

    for (int i = 0; i < N / 2; i++) {
      if (hostMem[i] != expectedResult_1) {
        std::cout << " i = " << i << "hostMem : " << hostMem[i] << std::endl;
        REQUIRE(false);
      }
    }

    for (int i = N / 2; i < N; i++) {
      if (hostMem[i] != expectedResult_2) {
        std::cout << " i = " << i << "hostMem : " << hostMem[i] << std::endl;
        REQUIRE(false);
      }
    }

    HIP_CHECK(hipFree(devMem));
  }
}
