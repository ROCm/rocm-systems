/*
Copyright (c) 2020 - 2022 Advanced Micro Devices, Inc. All rights reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
IMPLIED, INNCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANNY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
// Test Description:
/*The general idea of the application is to test how multi-GPU Cooperative
Groups kernel launches to a stream interact with other things that may be
simultaneously running in the same streams.

The HIP specification says that a multi-GPU cooperative launch will wait
until all of the streams it's using finish their work. Only then will the
cooperative kernel be launched to all of the devices. Then no other work
can take part in the any of the streams until all of the multi-GPU
cooperative work is done.

However, there are flags that allow you to disable each of these
serialization points: hipCooperativeLaunchMultiDeviceNoPreSync and
hipCooperativeLaunchMultiDeviceNoPostSync.

As such, this benchmark tests the following five situations launching
to two GPUs (and thus two streams):

    1. Normal multi-GPU cooperative kernel:
        This should result in the following pattern:
        Stream 0: Cooperative
        Stream 1: Cooperative
    2. Regular kernel launches and multi-GPU cooperative kernel launches
       with the default flags, resulting in the following pattern:
        Stream 0: Regular --> Cooperative
        Stream 1:         --> Cooperative --> Regular

    3. Regular kernel launches and multi-GPU cooperative kernel launches
       that turn off "pre-sync". This should allow a cooperative kernel
       to launch even if work is already in a stream pointing to
       another GPU.
        This should result in the following pattern:
        Stream 0: Regular --> Cooperative
        Stream 1: Cooperative            --> Regular

    4. Regular kernel launches and multi-GPU cooperative kernel launches
       that turn off "post-sync". This should allow a new kernel to enter
       a GPU even if another GPU still has a cooperative kernel on it.
        This should result in the following pattern:
        Stream 0: Regular --> Cooperative
        Stream 1:         --> Cooperative--> Regular

    5. Regular kernel launches and multi-GPU cooperative kernel launches
       that turn off both pre- and post-sync. This should allow any of
       the kernels to launch to their GPU regardless of the status of
       other kernels in other multi-GPU stream groups.
        This should result in the following pattern:
        Stream 0: Regular --> Cooperative
        Stream 1: Cooperative --> Regular

We time how long it takes to run each of these benchmarks and print it as
the output of the benchmark. The kernels themselves are just useless time-
wasting code so that the kernel takes a meaningful amount of time on the
GPU before it exits. We only launch a single wavefront for each kernel, so
any serialization should not be because of GPU occupancy concerns.

If tests 2, 3, and 4 take roughly 3x as long as #1, that implies that
cooperative kernels are serialized as expected.

If test #5 takes roughly twice as long as #1, that implies that the
overlap-allowing flags work as expected.
*/

#include <hip_test_common.hh>
#include <hip/hip_cooperative_groups.h>

namespace cg = cooperative_groups;

static constexpr size_t kBufferLen = 1024 * 1024;

__global__ void test_gws(uint* buf, uint buf_size, long* tmp_buf, long* result) {
  extern __shared__ long tmp[];
  uint groups = gridDim.x;
  uint group_id = blockIdx.x;
  uint local_id = threadIdx.x;
  uint chunk = gridDim.x * blockDim.x;

  uint i = group_id * blockDim.x + local_id;
  long sum = 0;
  while (i < buf_size) {
    sum += buf[i];
    i += chunk;
  }
  tmp[local_id] = sum;
  __syncthreads();
  i = 0;
  if (local_id == 0) {
    sum = 0;
    while (i < blockDim.x) {
      sum += tmp[i];
      i++;
    }
    tmp_buf[group_id] = sum;
  }
  // wait
  cg::this_grid().sync();

  if (((blockIdx.x * blockDim.x) + threadIdx.x) == 0) {
    for (uint i = 1; i < groups; ++i) {
      sum += tmp_buf[i];
    }
    //*result = sum;
    result[1 + cg::this_multi_grid().grid_rank()] = sum;
  }
  cg::this_multi_grid().sync();
  if (cg::this_multi_grid().grid_rank() == 0) {
    sum = 0;
    for (uint i = 1; i <= cg::this_multi_grid().num_grids(); ++i) {
      sum += result[i];
    }
    *result = sum;
  }
}

__global__ void test_coop_kernel(unsigned int loops, long long* array, int fast_gpu) {
  cg::multi_grid_group mgrid = cg::this_multi_grid();
  unsigned int rank = blockIdx.x * blockDim.x + threadIdx.x;

  if (mgrid.grid_rank() == fast_gpu) {
    return;
  }

  for (int i = 0; i < loops; i++) {
    long long time_diff = 0;
    long long last_clock = clock64();
    do {
      long long cur_clock = clock64();
      if (cur_clock > last_clock) {
        time_diff += (cur_clock - last_clock);
      }
      // If it rolls over, we don't know how much to add to catch up.
      // So just ignore those slipped cycles.
      last_clock = cur_clock;
    } while (time_diff < 1000000);
    array[rank] += clock64();
  }
}

__global__ void test_coop_kernel_gfx11(unsigned int loops, long long* array, int fast_gpu) {
#if HT_AMD
  cg::multi_grid_group mgrid = cg::this_multi_grid();
  unsigned int rank = blockIdx.x * blockDim.x + threadIdx.x;

  if (mgrid.grid_rank() == fast_gpu) {
    return;
  }

  for (int i = 0; i < loops; i++) {
    long long time_diff = 0;
    long long last_clock = clock_function();
    do {
      long long cur_clock = clock_function();
      if (cur_clock > last_clock) {
        time_diff += (cur_clock - last_clock);
      }
      // If it rolls over, we don't know how much to add to catch up.
      // So just ignore those slipped cycles.
      last_clock = cur_clock;
    } while (time_diff < 1000000);
    array[rank] += clock_function();
  }
#endif
}

__global__ void test_kernel(uint32_t loops, unsigned long long* array) {
  unsigned int rank = blockIdx.x * blockDim.x + threadIdx.x;

  for (int i = 0; i < loops; i++) {
    long long time_diff = 0;
    long long last_clock = clock64();
    do {
      long long cur_clock = clock64();
      if (cur_clock > last_clock) {
        time_diff += (cur_clock - last_clock);
      }
      // If it rolls over, we don't know how much to add to catch up.
      // So just ignore those slipped cycles.
      last_clock = cur_clock;
    } while (time_diff < 1000000);
    array[rank] += clock64();
  }
}

__global__ void test_kernel_gfx11(uint32_t loops, unsigned long long* array) {
#if HT_AMD
  unsigned int rank = blockIdx.x * blockDim.x + threadIdx.x;

  for (int i = 0; i < loops; i++) {
    long long time_diff = 0;
    long long last_clock = clock_function();
    do {
      long long cur_clock = clock_function();
      if (cur_clock > last_clock) {
        time_diff += (cur_clock - last_clock);
      }
      // If it rolls over, we don't know how much to add to catch up.
      // So just ignore those slipped cycles.
      last_clock = cur_clock;
    } while (time_diff < 1000000);
    array[rank] += clock_function();
  }
#endif
}

static void verify_time(double single_kernel_time, double multi_kernel_time, float low_bound,
                        float high_bound) {
  // Test that multiple kernel times are inside expected boundaries
  REQUIRE(multi_kernel_time >= low_bound * single_kernel_time);
  REQUIRE(multi_kernel_time <= high_bound * single_kernel_time);
}

void test_multigrid_streams(int device_num) {
  uint32_t loops = 2000;
  int32_t fast_gpu = -1;

  // We will launch enough waves to fill up all of the GPU
  int warp_sizes[2];
  int num_sms[2];
  hipDeviceProp_t device_properties[2];
  int warp_size = INT_MAX;
  int num_sm = INT_MAX;
  for (int dev = 0; dev < (device_num - 1); ++dev) {
    for (int i = 0; i < 2; i++) {
      HIP_CHECK(hipGetDeviceProperties(&device_properties[i], (dev + i)));
      warp_sizes[i] = device_properties[i].warpSize;
      if (warp_sizes[i] < warp_size) {
        warp_size = warp_sizes[i];
      }
      num_sms[i] = device_properties[i].multiProcessorCount;
      if (num_sms[i] < num_sm) {
        num_sm = num_sms[i];
      }
    }

    // Calculate the device occupancy to know how many blocks can be run.
    int max_blocks_per_sm_arr[2];
    int max_blocks_per_sm = INT_MAX;
    for (int i = 0; i < 2; i++) {
      HIP_CHECK(hipSetDevice(dev + i));
      auto test_kernel_used = IsGfx11() ? test_kernel_gfx11 : test_kernel;
      HIP_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_sm_arr[i],
                                                             test_kernel_used, warp_size, 0));
      if (max_blocks_per_sm_arr[i] < max_blocks_per_sm) {
        max_blocks_per_sm = max_blocks_per_sm_arr[i];
      }
    }
    int desired_blocks = 1;

    if (desired_blocks > max_blocks_per_sm * num_sm) {
      INFO("The requested number of blocks will not fit on the GPU");
      REQUIRE(desired_blocks < max_blocks_per_sm * num_sm);
      return;
    }

    // Create the streams we will use in this test
    hipStream_t streams[2];
    for (int i = 0; i < 2; i++) {
      HIP_CHECK(hipSetDevice(dev + i));
      HIP_CHECK(hipStreamCreate(&streams[i]));
    }

    // Set up data to pass into the kernel
    // Alocate the host input buffer, and two device-focused buffers that we
    // will use for our test.
    unsigned long long* dev_array[2];
    for (int i = 0; i < 2; i++) {
      int good_size = desired_blocks * warp_size * sizeof(long long);
      HIP_CHECK(hipSetDevice(dev + i));
      HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&dev_array[i]), good_size));
      HIP_CHECK(hipMemsetAsync(dev_array[i], 0, good_size, streams[i]));
    }
    for (int i = 0; i < 2; i++) {
      HIP_CHECK(hipSetDevice(dev + i));
      HIP_CHECK(hipDeviceSynchronize());
    }

    /* Launch the kernels ****************************************************/
    void* dev_params[2][3];
    hipLaunchParams md_params[2];
    std::chrono::time_point<std::chrono::system_clock> start_time[2];
    std::chrono::time_point<std::chrono::system_clock> end_time[2];

    // Test 0: Launching a multi-GPU cooperative kernel
    // Both GPUs launch a long cooperative kernel
    INFO("GPU " << dev << ": Long Coop Kernel");
    INFO("GPU " << (dev + 1) << ": Long Coop Kernel");

    auto test_coop_kernel_used = IsGfx11() ? test_coop_kernel_gfx11 : test_coop_kernel;
    for (int i = 0; i < 2; i++) {
      dev_params[i][0] = reinterpret_cast<void*>(&loops);
      dev_params[i][1] = reinterpret_cast<void*>(&dev_array[i]);
      dev_params[i][2] = reinterpret_cast<void*>(&fast_gpu);
      md_params[i].func = reinterpret_cast<void*>(test_coop_kernel_used);
      md_params[i].gridDim = desired_blocks;
      md_params[i].blockDim = warp_size;
      md_params[i].sharedMem = 0;
      md_params[i].stream = streams[i];
      md_params[i].args = dev_params[i];
    }

    start_time[0] = std::chrono::system_clock::now();
    HIP_CHECK(hipLaunchCooperativeKernelMultiDevice(md_params, 2, 0));
    for (int i = 0; i < 2; i++) {
      HIP_CHECK(hipSetDevice(dev + i));
      HIP_CHECK(hipDeviceSynchronize());
    }
    end_time[0] = std::chrono::system_clock::now();

    std::chrono::duration<double> single_kernel_time = (end_time[0] - start_time[0]);
    INFO("A single kernel on both GPUs took: " << single_kernel_time.count() << " seconds");

    SECTION("GPU1 - Standard/ Long Coop, GPU2 - Coop/Standard") {
      INFO("GPU " << dev << ": Standard/Long Coop");
      INFO("GPU " << (dev + 1) << ": Coop/Standard");
      fast_gpu = 1;
      start_time[1] = std::chrono::system_clock::now();
      HIP_CHECK(hipSetDevice(dev));
      auto test_kernel_used = IsGfx11() ? test_kernel_gfx11 : test_kernel;
      hipLaunchKernelGGL(test_kernel_used, dim3(desired_blocks), dim3(warp_size), 0, streams[0],
                         loops, dev_array[0]);
      HIP_CHECK(hipGetLastError());
      HIP_CHECK(hipLaunchCooperativeKernelMultiDevice(md_params, 2, 0));
      HIP_CHECK(hipSetDevice(dev + 1));
      test_kernel_used = IsGfx11() ? test_kernel_gfx11 : test_kernel;
      hipLaunchKernelGGL(test_kernel_used, dim3(desired_blocks), dim3(warp_size), 0, streams[1],
                         loops, dev_array[1]);
      HIP_CHECK(hipGetLastError());
      for (int i = 0; i < 2; i++) {
        HIP_CHECK(hipSetDevice(dev + i));
        HIP_CHECK(hipDeviceSynchronize());
      }
      end_time[1] = std::chrono::system_clock::now();
      std::chrono::duration<double> serialized_gpu0_time = (end_time[1] - start_time[1]);
      INFO("Serialized set of three kernels with GPU0 being long took: "
           << serialized_gpu0_time.count() << " seconds");

      verify_time(single_kernel_time.count(), serialized_gpu0_time.count(), 2.7f, 3.3f);
    }

    SECTION("GPU1 - Standard/Coop, GPU2 - Long Coop/Standard") {
      INFO("GPU " << dev << ": Standard/Coop");
      INFO("GPU " << (dev + 1) << ": Long Coop/Standard");
      fast_gpu = 0;
      start_time[1] = std::chrono::system_clock::now();
      HIP_CHECK(hipSetDevice(dev));
      auto test_kernel_used = IsGfx11() ? test_kernel_gfx11 : test_kernel;
      hipLaunchKernelGGL(test_kernel_used, dim3(desired_blocks), dim3(warp_size), 0, streams[0],
                         loops, dev_array[0]);
      HIP_CHECK(hipGetLastError());
      HIP_CHECK(hipLaunchCooperativeKernelMultiDevice(md_params, 2, 0));
      HIP_CHECK(hipSetDevice(dev + 1));
      test_kernel_used = IsGfx11() ? test_kernel_gfx11 : test_kernel;
      hipLaunchKernelGGL(test_kernel_used, dim3(desired_blocks), dim3(warp_size), 0, streams[1],
                         loops, dev_array[1]);
      HIP_CHECK(hipGetLastError());
      for (int i = 0; i < 2; i++) {
        HIP_CHECK(hipSetDevice(dev + i));
        HIP_CHECK(hipDeviceSynchronize());
      }
      end_time[1] = std::chrono::system_clock::now();
      std::chrono::duration<double> serialized_gpu1_time = (end_time[1] - start_time[1]);
      INFO("Serialized set of three kernels with GPU1 being long took: "
           << serialized_gpu1_time.count() << " seconds");

      verify_time(single_kernel_time.count(), serialized_gpu1_time.count(), 2.7f, 3.3f);
    }

    SECTION(
        "GPU1 - Standard/Coop, GPU2 - Long Coop/Standard - regular and coop kernel overlap at "
        "beginning") {
      INFO("GPU " << dev << ": Standard/Coop with multi device no pre sync");
      INFO("GPU " << (dev + 1) << ": Long Coop/Standard with multi device no pre sync");
      fast_gpu = 0;
      start_time[1] = std::chrono::system_clock::now();
      HIP_CHECK(hipSetDevice(dev));
      auto test_kernel_used = IsGfx11() ? test_kernel_gfx11 : test_kernel;
      hipLaunchKernelGGL(test_kernel_used, dim3(desired_blocks), dim3(warp_size), 0, streams[0],
                         loops, dev_array[0]);
      HIP_CHECK(hipGetLastError());
      HIP_CHECK(hipLaunchCooperativeKernelMultiDevice(md_params, 2,
                                                      hipCooperativeLaunchMultiDeviceNoPreSync));
      HIP_CHECK(hipSetDevice(dev + 1));
      test_kernel_used = IsGfx11() ? test_kernel_gfx11 : test_kernel;
      hipLaunchKernelGGL(test_kernel_used, dim3(desired_blocks), dim3(warp_size), 0, streams[1],
                         loops, dev_array[1]);
      HIP_CHECK(hipGetLastError());
      for (int i = 0; i < 2; i++) {
        HIP_CHECK(hipSetDevice(dev + i));
        HIP_CHECK(hipDeviceSynchronize());
      }
      end_time[1] = std::chrono::system_clock::now();
      std::chrono::duration<double> pre_overlapped_time = (end_time[1] - start_time[1]);
      INFO("Multiple kernels with pre-overlap allowed took: " << pre_overlapped_time.count()
                                                              << " seconds");

      verify_time(single_kernel_time.count(), pre_overlapped_time.count(), 1.7f, 2.3f);
    }

    SECTION(
        "GPU1 - Standard/Long Coop, GPU2 - Coop/Standard - regular and coop kernel overlap at "
        "end") {
      INFO("GPU " << dev << ": Standard/Long Coop with multi device no post sync");
      INFO("GPU " << (dev + 1) << ": Coop/Standard with multi device no post sync");
      fast_gpu = 1;
      start_time[1] = std::chrono::system_clock::now();
      HIP_CHECK(hipSetDevice(dev));
      auto test_kernel_used = IsGfx11() ? test_kernel_gfx11 : test_kernel;
      hipLaunchKernelGGL(test_kernel_used, dim3(desired_blocks), dim3(warp_size), 0, streams[0],
                         loops, dev_array[0]);
      HIP_CHECK(hipGetLastError());
      HIP_CHECK(hipLaunchCooperativeKernelMultiDevice(md_params, 2,
                                                      hipCooperativeLaunchMultiDeviceNoPostSync));
      HIP_CHECK(hipSetDevice(dev + 1));
      test_kernel_used = IsGfx11() ? test_kernel_gfx11 : test_kernel;
      hipLaunchKernelGGL(test_kernel_used, dim3(desired_blocks), dim3(warp_size), 0, streams[1],
                         loops, dev_array[1]);
      for (int i = 0; i < 2; i++) {
        HIP_CHECK(hipSetDevice(dev + i));
        HIP_CHECK(hipDeviceSynchronize());
      }
      end_time[1] = std::chrono::system_clock::now();
      std::chrono::duration<double> post_overlapped_time = (end_time[1] - start_time[1]);
      INFO("Multiple kernels with post-overlap allowed took: " << post_overlapped_time.count()
                                                               << " seconds");

      verify_time(single_kernel_time.count(), post_overlapped_time.count(), 1.7f, 2.3f);
    }

    SECTION(
        "GPU1 - Standard/Long Coop, GPU2 - Long Coop/Standard - regular and coop kernel overlap") {
      INFO("GPU " << dev << ": Standard/Long Coop with multi device no pre or post sync");
      INFO("GPU " << (dev + 1) << ": Long Coop/Standard with multi device no pre or post sync");
      start_time[1] = std::chrono::system_clock::now();
      HIP_CHECK(hipSetDevice(dev));
      auto test_kernel_used = IsGfx11() ? test_kernel_gfx11 : test_kernel;
      hipLaunchKernelGGL(test_kernel_used, dim3(desired_blocks), dim3(warp_size), 0, streams[0],
                         loops, dev_array[0]);
      HIP_CHECK(hipGetLastError());
      HIP_CHECK(hipLaunchCooperativeKernelMultiDevice(
          md_params, 2,
          hipCooperativeLaunchMultiDeviceNoPreSync | hipCooperativeLaunchMultiDeviceNoPostSync));
      HIP_CHECK(hipSetDevice(dev + 1));
      test_kernel_used = IsGfx11() ? test_kernel_gfx11 : test_kernel;
      hipLaunchKernelGGL(test_kernel_used, dim3(desired_blocks), dim3(warp_size), 0, streams[1],
                         loops, dev_array[1]);
      HIP_CHECK(hipGetLastError());
      for (int i = 0; i < 2; i++) {
        HIP_CHECK(hipSetDevice(dev + i));
        HIP_CHECK(hipDeviceSynchronize());
      }
      end_time[1] = std::chrono::system_clock::now();
      std::chrono::duration<double> overlapped_time = (end_time[1] - start_time[1]);
      INFO("Multiple kernels with overlap allowed took: " << overlapped_time.count() << " seconds");

      verify_time(single_kernel_time.count(), overlapped_time.count(), 1.8f, 2.2f);
    }

    for (int k = 0; k < 2; ++k) {
      HIP_CHECK(hipFree(dev_array[k]));
      HIP_CHECK(hipStreamDestroy(streams[k]));
    }
  }
}

TEST_CASE("Unit_hipLaunchCooperativeKernelMultiDevice_Basic") {
  constexpr uint num_kernel_args = 4;

  int device_num = 0;
  HIP_CHECK(hipGetDeviceCount(&device_num));

  size_t buffer_size = kBufferLen * sizeof(int);

  int* A_h = reinterpret_cast<int*>(malloc(buffer_size * device_num));
  for (uint32_t i = 0; i < kBufferLen * device_num; ++i) {
    A_h[i] = static_cast<int>(i);
  }

  std::vector<int*> A_d(device_num);
  std::vector<long*> B_d(device_num);
  long* C_d;
  std::vector<hipStream_t> stream(device_num);

  std::vector<hipDeviceProp_t> device_properties(device_num);

  for (int i = 0; i < device_num; i++) {
    HIP_CHECK(hipSetDevice(i));

    // Calculate the device occupancy to know how many blocks can be run concurrently
    HIP_CHECK(hipGetDeviceProperties(&device_properties[i], 0));
    if (!device_properties[i].cooperativeMultiDeviceLaunch) {
      HipTest::HIP_SKIP_TEST("Device doesn't support cooperative launch!");
      return;
    }

    HIP_CHECK(hipMalloc(&A_d[i], buffer_size));
    HIP_CHECK(hipMemcpy(A_d[i], &A_h[i * kBufferLen], buffer_size, hipMemcpyHostToDevice));
    if (i == 0) {
      HIP_CHECK(hipHostMalloc(&C_d, (device_num + 1) * sizeof(long)));
    }

    HIP_CHECK(hipStreamCreate(&stream[i]));
    HIP_CHECK(hipDeviceSynchronize());
  }

  dim3 dimBlock;
  dim3 dimGrid;
  dimGrid.x = 1;
  dimGrid.y = 1;
  dimGrid.z = 1;
  dimBlock.x = 64;
  dimBlock.y = 1;
  dimBlock.z = 1;

  int num_blocks = 0;
  uint workgroup = GENERATE(64, 128, 256);

  hipLaunchParams* launch_params_list = new hipLaunchParams[device_num];
  std::vector<void*> args(device_num * num_kernel_args);

  for (int i = 0; i < device_num; i++) {
    HIP_CHECK(hipSetDevice(i));

    dimBlock.x = workgroup;
    HIP_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(
        &num_blocks, test_gws, dimBlock.x * dimBlock.y * dimBlock.z, dimBlock.x * sizeof(long)));

    INFO("GPU" << i << " has block size = " << dimBlock.x << " and num blocks per CU " << num_blocks
               << "\n");

    dimGrid.x = device_properties[i].multiProcessorCount * std::min(num_blocks, 32);

    HIP_CHECK(hipMalloc(&B_d[i], dimGrid.x * sizeof(long)));

    args[i * num_kernel_args] = (void*)&A_d[i];
    args[i * num_kernel_args + 1] = (void*)&kBufferLen;
    args[i * num_kernel_args + 2] = (void*)&B_d[i];
    args[i * num_kernel_args + 3] = (void*)&C_d;

    launch_params_list[i].func = reinterpret_cast<void*>(test_gws);
    launch_params_list[i].gridDim = dimGrid;
    launch_params_list[i].blockDim = dimBlock;
    launch_params_list[i].sharedMem = dimBlock.x * sizeof(long);
    launch_params_list[i].stream = stream[i];
    launch_params_list[i].args = &args[i * num_kernel_args];
  }

  HIP_CHECK(hipLaunchCooperativeKernelMultiDevice(launch_params_list, device_num, 0));
  for (int i = 0; i < device_num; i++) {
    HIP_CHECK(hipStreamSynchronize(stream[i]));
  }

  size_t processed_Dwords = kBufferLen * device_num;
  REQUIRE(*C_d == (((long)(processed_Dwords) * (processed_Dwords - 1)) / 2));

  delete[] launch_params_list;

  HIP_CHECK(hipSetDevice(0));
  HIP_CHECK(hipHostFree(C_d));
  for (int i = 0; i < device_num; i++) {
    HIP_CHECK(hipSetDevice(i));
    HIP_CHECK(hipFree(A_d[i]));
    HIP_CHECK(hipFree(B_d[i]));
    HIP_CHECK(hipStreamDestroy(stream[i]));
  }

  free(A_h);
}

TEST_CASE("Unit_hipLaunchCooperativeKernelMultiDevice_Streams") {
  int device_num = 0;
  HIP_CHECK(hipGetDeviceCount(&device_num));

  if (device_num < 2) {
    HipTest::HIP_SKIP_TEST("Skipping because devices < 2");
    return;
  }

  hipDeviceProp_t device_properties;
  for (int i = 0; i < device_num; i++) {
    HIP_CHECK(hipGetDeviceProperties(&device_properties, i));
    if (!device_properties.cooperativeMultiDeviceLaunch) {
      HipTest::HIP_SKIP_TEST("Device doesn't support cooperative launch!");
      return;
    }
  }

  test_multigrid_streams(device_num);
}
