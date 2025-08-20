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
