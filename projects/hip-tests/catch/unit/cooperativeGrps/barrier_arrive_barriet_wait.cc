/*
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include "cooperative_groups_common.hh"
#include <hip_test_common.hh>

static constexpr int N = 1024;
static __device__ int devArr[N];

/**
 * Kernel to reset the Global device memory devArr
 */
static __global__ void resetGlobalDevArr() {
  for (int i = 0; i < N; i++) {
    devArr[i] = 0;
  }
}

/**
 * Device function to fill data in given array.
 * 0 to size-1 indexes filled with 1 to size data
 */
static __device__ void fillData(int *arr, int size) {
  for (int i = 0; i < size; i++) {
    arr[i] = i + 1;
  }
}

/**
 * Device function to find the sum of elements of array
 */
static __device__ int getSumOfArrayElements(int *arr, int size) {
  int sum = 0;
  for (int i = 0; i < size; i++) {
    sum += arr[i];
  }
  return sum;
}

/**
 * Kernel function perform below operation :
 * Threads  : 0   1     2     3     ... 1022         1023
 * devArr   : 100 0     0     0     ...    0         100
 *                <------------------------> (1 to 1022 Threads continue)
 * sum      : 0   1*2/2 2*3/2 3*4/2 ... 1022*1023/2  0
 *          : 0   1     3     6     ... 522753       0 <- All threads wait here
 * outputArr: 200 201   203   206   ... 522953       200
 * devArrSum is 200 (2 elements in devArr filled with 100)
 */
static __global__ void coopKernelWithBlock_1D(int *outputArr) {
  cooperative_groups::thread_block block =
      cooperative_groups::this_thread_block();

  if (threadIdx.x == 0 || threadIdx.x == (N - 1)) { // for 0 and 1023
    devArr[threadIdx.x] = 100;
  }
  auto token = block.barrier_arrive();

  int sum = 0;

  if (!(threadIdx.x == 0 || threadIdx.x == (N - 1))) { // for 1 to 1022
    int *local_data = new int[threadIdx.x];
    fillData(local_data, threadIdx.x);
    sum = getSumOfArrayElements(local_data, threadIdx.x);
    delete[] local_data;
  }

  block.barrier_wait(std::move(token));

  int devArrSum = 0;
  for (int i = 0; i < N; i++) {
    devArrSum = devArrSum + devArr[i];
  }
  outputArr[threadIdx.x] = sum + devArrSum;
}

/**
 * Kernel function perform below operation :
 * Threads  : 0     1      ...    511     512 513 ... 1022 1023
 *            <------- y=0, x ------>     <------- y=1, x ---->
 * devArr   : 0     0      ...    0       100 100 ... 100  100
 *              <-------------------> (1 to 511 Threads continue)
 * sum      : 0     1*2/2  ... 511*512/2  0   0   ... 0    0
 *          : 0     1      ... 130816     0   0   ... 0    0 <- All threads wait
 * outputArr: 51200 51201  ... 182016    <------ 51200  ------->
 * devArrSum is 51200 ( 512 elements in devArr filled with 100)
 */
static __global__ void coopKernelWithBlock_2D(int *outputArr) {
  int index = blockDim.x * threadIdx.y + threadIdx.x;
  cooperative_groups::thread_block block =
      cooperative_groups::this_thread_block();

  // All threads in y direction
  if (threadIdx.y == 1) {
    devArr[index] = 100;
  }

  auto token = block.barrier_arrive();

  int sum = 0;
  // All threads in x direction except 0
  if ((threadIdx.y == 0) && (threadIdx.x != 0)) {
    int *local_data = new int[index];
    fillData(local_data, index);
    sum = getSumOfArrayElements(local_data, index);
    delete[] local_data;
  }

  block.barrier_wait(std::move(token));

  int devArrSum = 0;
  for (int i = 0; i < N; i++) {
    devArrSum = devArrSum + devArr[i];
  }
  outputArr[index] = sum + devArrSum;
}

/**
 * Kernel function perform below operation :
 * Threads  :  0     1       2   3   ...   14  15   16  ... 1023
 *           <-z=0,y=0,x--> <--  z=0, y, x ------>  <-- z>0 --->
 * devArr   :  0     0       0   0   ...    0   0   <-- 100 --->
 *                 <-----------------------------> (1 to 15 Threads continue)
 * sum      :  0    1*2/2   2*3/2  ...........  7*8/2  0  ... 0
 *          :  0     1        3    ...........  28     0  ... 0 All threads wait
 * outputArr: 100800 100801 100803 ........... 100828  <-- 100800 -->
 * devArrSum is 100800 ( 1016 elements in devArr filled with 100)
 */
static __global__ void coopKernelWithBlock_Z_3D(int *outputArr) {
  int index = blockDim.x * blockDim.y * threadIdx.z + threadIdx.y * blockDim.x +
              threadIdx.x;
  cooperative_groups::thread_block block =
      cooperative_groups::this_thread_block();

  // All threads in z direction
  if (threadIdx.z >= 1) {
    devArr[index] = 100;
  }

  auto token = block.barrier_arrive();

  int sum = 0;
  // All threads in x & y direction except index 0
  if ((threadIdx.z == 0) && (index != 0)) {
    int *local_data = new int[index];
    fillData(local_data, index);
    sum = getSumOfArrayElements(local_data, index);
    delete[] local_data;
  }

  block.barrier_wait(std::move(token));

  int devArrSum = 0;
  for (int i = 0; i < N; i++) {
    devArrSum = devArrSum + devArr[i];
  }
  outputArr[index] = sum + devArrSum;
}

/**
 * Kernel function perform below operation :
 * Threads  :  0     1       2   3   ...   14  15   16  ... 1023
 *           <-z=0,y=0,x--> <--  z=0, y, x  ---->  <-- z>0 -->
 * devArr   :  0     0      <---------------- 100 --------------->
 *                 <----> (Only 1 thread continue where z=0, y=0, x=1)
 * sum      :  0    1*2/2    0   0   0   0   0   0   0  ...  0
 *          :  0     1       0   0   0   0   0   0   0  ...  0 All threads wait
 * outputArr: 102200 102201 <-------------- 102200 -------------->
 * devArrSum is 102200 ( 1022 elements in devArr filled with 100 )
 */
static __global__ void coopKernelWithBlock_YZ_3D(int *outputArr) {
  int index = blockDim.x * blockDim.y * threadIdx.z + threadIdx.y * blockDim.x +
              threadIdx.x;

  cooperative_groups::thread_block block =
      cooperative_groups::this_thread_block();

  // Hold all threads in y and z direction
  if (threadIdx.y >= 1 || threadIdx.z >= 1) {
    devArr[index] = 100;
  }

  auto token = block.barrier_arrive();

  int sum = 0;
  // All threads in x direction except index 0
  if ((!(threadIdx.y >= 1 || threadIdx.z >= 1)) && (index != 0)) {
    int *local_data = new int[index];
    fillData(local_data, index);
    sum = getSumOfArrayElements(local_data, index);
    delete[] local_data;
  }

  block.barrier_wait(std::move(token));

  int devArrSum = 0;
  for (int i = 0; i < N; i++) {
    devArrSum = devArrSum + devArr[i];
  }
  outputArr[index] = sum + devArrSum;
}

/**
 * Test Description
 * ------------------------
 *  - This test case checks the following scenarios and validates the
 *  - behavior of barrier_arrive & barrier_wait.
 *  - Launches kernel with different below combinations and
 *  - validates the output
 *  - 1) With 1D Block with N blocks, with different work in some threads
 *  - 2) With 2D Block, with different work in blocks which are in y direction
 *  - 3) With 3D Block, with different work in blocks which are in z direction
 *  - 4) With 3D Block, with different work in blocks which are in y&z direction
 * Test source
 * ------------------------
 *  - unit/cooperativeGrps/barrier_arrive_barrier_wait.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.1
 */
TEST_CASE("Unit_barrier_wait_barrier_arrive_Block") {
  int cooperativeLaunchSupported;
  HIP_CHECK(hipDeviceGetAttribute(&cooperativeLaunchSupported,
                                  hipDeviceAttributeCooperativeLaunch, 0));

  if (!cooperativeLaunchSupported) {
    HipTest::HIP_SKIP_TEST("Skipping test as CooperativeLaunch not supported");
  }

  int hostMem[N];
  for (int i = 0; i < N; i++) {
    hostMem[i] = 0;
  }
  resetGlobalDevArr<<<1, 1>>>();
  HIP_CHECK(hipDeviceSynchronize());

  resetGlobalDevArr<<<1, 1>>>();
  HIP_CHECK(hipDeviceSynchronize());

  int *devMem = nullptr;
  HIP_CHECK(hipMalloc(&devMem, N * sizeof(int)));
  REQUIRE(devMem != nullptr);
  HIP_CHECK(hipMemcpy(devMem, hostMem, N * sizeof(int), hipMemcpyHostToDevice));

  void *params[1];
  params[0] = &devMem;

  SECTION("With 1D Block - Perform sum operation in 0-1022 threads") {
    dim3 gridDim = dim3{1, 1, 1};
    dim3 blockDim = dim3{N, 1, 1};

    HIP_CHECK(hipLaunchCooperativeKernel((void *)(coopKernelWithBlock_1D),
                                         gridDim, blockDim, params, 0, 0));
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipMemcpy(hostMem, devMem, N * sizeof(int), hipMemcpyDefault));

    REQUIRE(hostMem[0] == 200);
    REQUIRE(hostMem[N - 1] == 200);

    int expectedResult;
    for (int i = 1; i <= N - 2; i++) { // 1 to 1022
      expectedResult = 200 + (i * (i + 1) / 2);
      INFO("At i : " << i << " Got : " << hostMem[i]
                     << " Expected : " << expectedResult);
      REQUIRE(hostMem[i] == expectedResult);
    }
  }

  SECTION("With 2D Block - Perform sum operation in x direction threads") {
    dim3 gridDim = dim3{1, 1, 1};
    dim3 blockDim = dim3{N / 2, N / 512, 1}; // {512, 2, 1}

    HIP_CHECK(hipLaunchCooperativeKernel((void *)(coopKernelWithBlock_2D),
                                         gridDim, blockDim, params, 0, 0));
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipMemcpy(hostMem, devMem, N * sizeof(int), hipMemcpyDefault));

    int expectedResult = 51200;

    REQUIRE(hostMem[0] == expectedResult);
    for (int i = N / 2; i < N; i++) { // 512 to 1023
      INFO("At i : " << i << " Got : " << hostMem[i]
                     << " Expected : " << expectedResult);
      REQUIRE(hostMem[i] == expectedResult);
    }

    for (int i = 1; i < N / 2; i++) { // 1 to 511
      expectedResult = 51200 + (i * (i + 1) / 2);
      INFO("At i : " << i << " Got : " << hostMem[i]
                     << " Expected : " << expectedResult);
      REQUIRE(hostMem[i] == expectedResult);
    }
  }

  SECTION("With 3D Block  - Perform sum operation in x & y direction threads") {
    dim3 gridDim = dim3{1, 1, 1};
    dim3 blockDim = dim3{N / 512, N / 128, N / 16}; // {2, 8, 64}

    HIP_CHECK(hipLaunchCooperativeKernel((void *)(coopKernelWithBlock_Z_3D),
                                         gridDim, blockDim, params, 0, 0));
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipMemcpy(hostMem, devMem, N * sizeof(int), hipMemcpyDefault));

    int expectedResult = 100800;

    REQUIRE(hostMem[0] == expectedResult);
    for (int i = (blockDim.x * blockDim.y); i < N; i++) { // 8 to 1023
      INFO("At i : " << i << " Got : " << hostMem[i]
                     << " Expected : " << expectedResult);
      REQUIRE(hostMem[i] == expectedResult);
    }

    for (int i = 1; i < (blockDim.x * blockDim.y); i++) { // 1 to 7
      expectedResult = 100800 + (i * (i + 1) / 2);
      INFO("At i : " << i << " Got : " << hostMem[i]
                     << " Expected : " << expectedResult);
      REQUIRE(hostMem[i] == expectedResult);
    }
  }

  SECTION(
      "With 3D Block  - Perform sum operation in only x direction threads") {
    dim3 gridDim = dim3{1, 1, 1};
    dim3 blockDim = dim3{N / 512, N / 128, N / 16}; // {2, 8, 64}

    HIP_CHECK(hipLaunchCooperativeKernel((void *)(coopKernelWithBlock_YZ_3D),
                                         gridDim, blockDim, params, 0, 0));
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipMemcpy(hostMem, devMem, N * sizeof(int), hipMemcpyDefault));

    int expectedResult = 102200;

    REQUIRE(hostMem[0] == expectedResult);
    for (int i = 2; i < N; i++) { // 2 to 1023
      INFO("At i : " << i << " Got : " << hostMem[i]
                     << " Expected : " << expectedResult);
      REQUIRE(hostMem[i] == expectedResult);
    }

    REQUIRE(hostMem[1] == expectedResult + 1);
  }

  HIP_CHECK(hipFree(devMem));
}

/**
 * Kernel function does below operation.
 * Blocks   : 0   1     2     3     ... 1022          1023
 * devArr   : 100 0     0     0     ...    0          100
 *                <------------------------> (1 to 1022 blocks continue)
 * sum      : 0   1*2/2 2*3/2 3*4/2 ... 1022*1023/2   0
 *          : 0   1     3     6     ... 522753        0 <- All blocks wait here
 * outputArr: 200 201   203   206   ... 522953        200
 * devArrSum is 200 (2 elements in devArr filled with 100)
 */
static __global__ void coopKernelWithGrid_1D(int *outputArr) {
  cooperative_groups::grid_group grid = cooperative_groups::this_grid();

  if (blockIdx.x == 0 || blockIdx.x == (N - 1)) { // for 0 and 1023
    devArr[blockIdx.x] = 100;
  }
  auto token = grid.barrier_arrive();

  int sum = 0;
  if (!(blockIdx.x == 0 || blockIdx.x == (N - 1))) { // for 1 to 1022
    int *local_data = new int[blockIdx.x];
    fillData(local_data, blockIdx.x);
    sum = getSumOfArrayElements(local_data, blockIdx.x);
    delete[] local_data;
  }

  grid.barrier_wait(std::move(token));

  int devArrSum = 0;
  for (int i = 0; i < N; i++) {
    devArrSum = devArrSum + devArr[i];
  }
  outputArr[blockIdx.x] = sum + devArrSum;
}

/**
 * Kernel function does below operation.
 * Blocks   : 0     1      ...    511     512 513 ... 1022 1023
 *            <------- y=0, x ------>     <------- y=1, x ---->
 * devArr   : 0     0      ...    0       100 100 ... 100  100
 *              <-------------------> (1 to 511 blocks continue)
 * sum      : 0     1*2/2  ... 511*512/2  0   0   ... 0    0
 *          : 0     1      ... 130816     0   0   ... 0    0 <- All blocks wait
 * outputArr: 51200 51201  ... 182016    <------ 51200  ------->
 * devArrSum is 51200 ( 512 elements in devArr filled with 100)
 */
static __global__ void coopKernelWithGrid_2D(int *outputArr) {
  int index = gridDim.x * blockIdx.y + blockIdx.x;
  cooperative_groups::grid_group grid = cooperative_groups::this_grid();

  // All threads in y direction
  if (blockIdx.y >= 1) {
    devArr[index] = 100;
  }

  auto token = grid.barrier_arrive();

  int sum = 0;
  // All threads in x direction except 0
  if ((blockIdx.y == 0) && (blockIdx.x != 0)) {
    int *local_data = new int[index];
    fillData(local_data, index);
    sum = getSumOfArrayElements(local_data, index);
    delete[] local_data;
  }

  grid.barrier_wait(std::move(token));

  int devArrSum = 0;
  for (int i = 0; i < N; i++) {
    devArrSum = devArrSum + devArr[i];
  }
  outputArr[index] = sum + devArrSum;
}

/**
 * Kernel function does below operation.
 * Blocks   :  0     1       2   3   4   5   6   7     8  ... 1023
 *           <-z=0,y=0,x--> <--  z=0, y=1-3, x -->     <-- z>0 -->
 * devArr   :  0     0       0   0   0   0   0   0     <-- 100 -->
 *                 <-----------------------------> (1 to 7 blocks continue)
 * sum      :  0    1*2/2   2*3/2  ...........  7*8/2  0  ... 0
 *          :  0     1        3    ...........  28     0  ... 0 <- All blocks
 *                                                                 wait here
 * outputArr: 101600 101601 101603 ........... 101628  <-- 101600 -->
 * devArrSum is 101600 ( 1016 elements in devArr filled with 100)
 */
static __global__ void coopKernelWithGrid_Z_3D(int *outputArr) {
  int index =
      gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x;

  cooperative_groups::grid_group grid = cooperative_groups::this_grid();

  // All threads in z direction
  if (blockIdx.z >= 1) {
    devArr[index] = 100;
  }

  auto token = grid.barrier_arrive();

  int sum = 0;
  // All threads in x & y direction except index 0
  if ((blockIdx.z == 0) && (index != 0)) {
    int *local_data = new int[index];
    fillData(local_data, index);
    sum = getSumOfArrayElements(local_data, index);
    delete[] local_data;
  }

  grid.barrier_wait(std::move(token));

  int devArrSum = 0;
  for (int i = 0; i < N; i++) {
    devArrSum = devArrSum + devArr[i];
  }
  outputArr[index] = sum + devArrSum;
}

/**
 * Kernel function does below operation.
 * Blocks   :  0     1       2   3   4   5   6   7   8  ... 1023
 *           <-z=0,y=0,x--> <--  z=0, y=>1, x -->    <-- z>0 -->
 * devArr   :  0     0      <---------------- 100 --------------->
 *                 <----> (Only 1 blocks continue where z=0, y=0, x=1)
 * sum      :  0    1*2/2    0   0   0   0   0   0   0  ...  0
 *          :  0     1       0   0   0   0   0   0   0  ...  0 <- All blocks
 *                                                                wait here
 * devArrSum is 102200 ( 1022 elements in devArr filled with 100 )
 * outputArr: 102200 102201 <-------------- 102200 -------------->
 */
static __global__ void coopKernelWithGrid_YZ_3D(int *outputArr) {
  int index =
      gridDim.x * gridDim.y * blockIdx.z + blockIdx.y * gridDim.x + blockIdx.x;

  cooperative_groups::grid_group grid = cooperative_groups::this_grid();

  // All threads in y and z direction
  if (blockIdx.y >= 1 || blockIdx.z >= 1) {
    devArr[index] = 100;
  }

  auto token = grid.barrier_arrive();

  int sum = 0;
  // All threads in x direction except index 0
  if ((!(blockIdx.y >= 1 || blockIdx.z >= 1)) && (index != 0)) {
    int *local_data = new int[index];
    fillData(local_data, index);
    sum = getSumOfArrayElements(local_data, index);
    delete[] local_data;
  }

  grid.barrier_wait(std::move(token));

  int devArrSum = 0;
  for (int i = 0; i < N; i++) {
    devArrSum = devArrSum + devArr[i];
  }
  outputArr[index] = sum + devArrSum;
}

/**
 * Test Description
 * ------------------------
 *  - This test case checks the following scenarios and validates the
 *  - behavior of barrier_arrive & barrier_wait.
 *  - Launches kernel with different below combinations and
 *  - validates the output
 *  - 1) With 1D Grid with N blocks, with different work in some threads
 *  - 2) With 2D Grid, with different work in blocks which are in y direction
 *  - 3) With 3D Grid, with different work in blocks which are in z direction
 *  - 4) With 3D Grid, with different work in blocks which are in y&z direction
 * Test source
 * ------------------------
 *  - unit/cooperativeGrps/barrier_arrive_barrier_wait.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.1
 */
TEST_CASE("Unit_barrier_wait_barrier_arrive_Grid") {
  int cooperativeLaunchSupported;
  HIP_CHECK(hipDeviceGetAttribute(&cooperativeLaunchSupported,
                                  hipDeviceAttributeCooperativeLaunch, 0));

  if (!cooperativeLaunchSupported) {
    HipTest::HIP_SKIP_TEST("Skipping test as CooperativeLaunch not supported");
  }

  int hostMem[N];
  for (int i = 0; i < N; i++) {
    hostMem[i] = 0;
  }
  resetGlobalDevArr<<<1, 1>>>();
  HIP_CHECK(hipDeviceSynchronize());

  int *devMem = nullptr;
  HIP_CHECK(hipMalloc(&devMem, N * sizeof(int)));
  REQUIRE(devMem != nullptr);
  HIP_CHECK(hipMemcpy(devMem, hostMem, N * sizeof(int), hipMemcpyHostToDevice));

  void *params[1];
  params[0] = &devMem;

  SECTION("With 1D Grid - Perform sum operation in 0-1022 blocks") {
    dim3 gridDim = dim3{N, 1, 1};
    dim3 blockDim = dim3{1, 1, 1};

    HIP_CHECK(hipLaunchCooperativeKernel((void *)(coopKernelWithGrid_1D),
                                         gridDim, blockDim, params, 0, 0));
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipMemcpy(hostMem, devMem, N * sizeof(int), hipMemcpyDefault));

    REQUIRE(hostMem[0] == 200);
    REQUIRE(hostMem[N - 1] == 200);

    int expectedResult;
    for (int i = 1; i <= N - 2; i++) { // 1 to 1022
      expectedResult = 200 + (i * (i + 1) / 2);
      INFO("At i : " << i << " Got : " << hostMem[i]
                     << " Expected : " << expectedResult);
      REQUIRE(hostMem[i] == expectedResult);
    }
  }

  SECTION("With 2D Grid - Perform sum operation in x direction blocks") {
    dim3 gridDim = dim3{N / 2, N / 512, 1}; // {512, 2, 1}
    dim3 blockDim = dim3{1, 1, 1};

    HIP_CHECK(hipLaunchCooperativeKernel((void *)(coopKernelWithGrid_2D),
                                         gridDim, blockDim, params, 0, 0));
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipMemcpy(hostMem, devMem, N * sizeof(int), hipMemcpyDefault));

    int expectedResult = 51200;

    REQUIRE(hostMem[0] == expectedResult);
    for (int i = N / 2; i < N; i++) { // 512 to 1023
      INFO("At i : " << i << " Got : " << hostMem[i]
                     << " Expected : " << expectedResult);
      REQUIRE(hostMem[i] == expectedResult);
    }

    for (int i = 1; i < N / 2; i++) { // 1 to 511
      expectedResult = 51200 + (i * (i + 1) / 2);
      INFO("At i : " << i << " Got : " << hostMem[i]
                     << " Expected : " << expectedResult);
      REQUIRE(hostMem[i] == expectedResult);
    }
  }

  SECTION("With 3D Grid - Perform sum operation in x&y direction blocks") {
    dim3 gridDim = dim3{N / 512, N / 256, N / 8}; // {2, 4, 128}
    dim3 blockDim = dim3{1, 1, 1};

    HIP_CHECK(hipLaunchCooperativeKernel((void *)(coopKernelWithGrid_Z_3D),
                                         gridDim, blockDim, params, 0, 0));
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipMemcpy(hostMem, devMem, N * sizeof(int), hipMemcpyDefault));

    int expectedResult = 101600;

    REQUIRE(hostMem[0] == expectedResult);
    for (int i = (gridDim.x * gridDim.y); i < N; i++) { // 8 to 1023
      INFO("At i : " << i << " Got : " << hostMem[i]
                     << " Expected : " << expectedResult);
      REQUIRE(hostMem[i] == expectedResult);
    }

    for (int i = 1; i < (gridDim.x * gridDim.y); i++) { // 1 to 7
      expectedResult = 101600 + (i * (i + 1) / 2);
      INFO("At i : " << i << " Got : " << hostMem[i]
                     << " Expected : " << expectedResult);
      REQUIRE(hostMem[i] == expectedResult);
    }
  }

  SECTION("With 3D Grid - Perform sum operation in only x direction blocks") {
    dim3 gridDim = dim3{N / 512, N / 256, N / 8}; // {2, 4, 128}
    dim3 blockDim = dim3{1, 1, 1};

    HIP_CHECK(hipLaunchCooperativeKernel((void *)(coopKernelWithGrid_YZ_3D),
                                         gridDim, blockDim, params, 0, 0));
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipMemcpy(hostMem, devMem, N * sizeof(int), hipMemcpyDefault));

    int expectedResult = 102200;

    REQUIRE(hostMem[0] == expectedResult);
    REQUIRE(hostMem[1] == expectedResult + 1);
    for (int i = 2; i < N; i++) { // 2 to 1023
      INFO("At i : " << i << " Got : " << hostMem[i]
                     << " Expected : " << expectedResult);
      REQUIRE(hostMem[i] == expectedResult);
    }
  }

  HIP_CHECK(hipFree(devMem));
}

/**
 * Kernel function does below operation. (Considering Wavefront size 64)
 * Blocks      : <----------- 0 -------------->  <----------- 1 ----------->
 * Threads     :  0  1  2  3   ....   30    31   32 33 34 35   ....   63  64
 * threadIdx.x :  0  1  2  3   ....   30    31   0  1  2  3    ....   30  31
 * shared_mem  : <|  | at 0, 101, at 1 102 ---> <|  | at 0, 101, at 1 102 --->
 * sum         :  0  0  2*3/2  ........  30*31/2 0  0  2*3/2  ........  30*31/2
 *             :  0  0  3      ........    465   0  0  3      ........    465
 * devArrSum is 203 ( 101+102 in each block )
 * outputArr   :203 203 206    -------     668  203 203 206    -------     668
 */
static __global__ void coopKernelToProcessWavefront(int *outputArr, int N) {
  cooperative_groups::thread_block block =
      cooperative_groups::this_thread_block();
  auto index = blockDim.x * blockIdx.x + threadIdx.x;

  __shared__ int dev_shared_mem[2];

  if (threadIdx.x < 2) {
    dev_shared_mem[threadIdx.x] = 100 + threadIdx.x + 1;
  }

  auto token = block.barrier_arrive();

  int sum = 0;
  if ((threadIdx.x >= 2) && (threadIdx.x < N)) {
    int *local_data = new int[threadIdx.x];
    fillData(local_data, threadIdx.x);
    sum = getSumOfArrayElements(local_data, threadIdx.x);
    delete[] local_data;
  }

  block.barrier_wait(std::move(token));

  int devArrSum = 0;
  for (int i = 0; i < 2; i++) {
    devArrSum = devArrSum + dev_shared_mem[i];
  }
  outputArr[index] = sum + devArrSum;
}

/**
 * Test Description
 * ------------------------
 *  - This test case checks validates the behavior of
 *  - barrier_arrive & barrier_wait in block by launching the kernel
 *  - with total number of threads across the blocks is equal to the
 *  - Wavefront size and validates the output.
 * Test source
 * ------------------------
 *  - unit/cooperativeGrps/barrier_arrive_barrier_wait.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.1
 */
TEST_CASE("Unit_barrier_wait_barrier_arrive_Block_WaveFront") {

  int cooperativeLaunchSupported;
  HIP_CHECK(hipDeviceGetAttribute(&cooperativeLaunchSupported,
                                  hipDeviceAttributeCooperativeLaunch, 0));

  if (!cooperativeLaunchSupported) {
    HipTest::HIP_SKIP_TEST("Skipping test as CooperativeLaunch not supported");
  }

  hipDeviceProp_t deviceProp;
  HIP_CHECK(hipGetDeviceProperties(&deviceProp, 0));

  INFO("Warp Size: " << deviceProp.warpSize);

  int N = deviceProp.warpSize;
  uint32_t blockCount = 2;
  uint32_t threadCount = N / 2;

  std::vector<int> hostMem;
  for (int i = 0; i < N; i++) {
    hostMem.push_back(0);
  }

  int *devMem = nullptr;
  HIP_CHECK(hipMalloc(&devMem, N * sizeof(int)));
  REQUIRE(devMem != nullptr);
  HIP_CHECK(hipMemcpy(devMem, hostMem.data(), N * sizeof(int),
                      hipMemcpyHostToDevice));

  dim3 gridDim = dim3{blockCount, 1, 1};
  dim3 blockDim = dim3{threadCount, 1, 1};

  void *params[2];
  params[0] = &devMem;
  params[1] = &threadCount;

  HIP_CHECK(hipLaunchCooperativeKernel((void *)(coopKernelToProcessWavefront),
                                       gridDim, blockDim, params, 0, 0));
  HIP_CHECK(hipDeviceSynchronize());

  HIP_CHECK(
      hipMemcpy(hostMem.data(), devMem, N * sizeof(int), hipMemcpyDefault));

  REQUIRE(hostMem[0] == 203);               //  0
  REQUIRE(hostMem[1] == 203);               //  1
  REQUIRE(hostMem[threadCount] == 203);     //  32
  REQUIRE(hostMem[threadCount + 1] == 203); //  33

  int expectedResult;
  for (int i = 2; i < threadCount; i++) {
    expectedResult = 203 + (i * (i + 1) / 2);
    INFO("At i : " << i << " Got : " << hostMem[i]
                   << " Expected : " << expectedResult);
    REQUIRE(hostMem[i] == expectedResult); // 2 to 31

    INFO("At i : " << threadCount + i << " Got : " << hostMem[threadCount + i]
                   << " Expected : " << expectedResult); // 34 to 63
    REQUIRE(hostMem[threadCount + i] == expectedResult);
  }

  HIP_CHECK(hipFree(devMem));
}

/**
 * Kernel function does below operation.
 * Blocks :<--------------- 0 --------------> <----------- 1 ----------------->
 * Threads: 0   1     -> 500       501 -> 511 512   513   -> 1012    1013 ->1023
 * tIdx.x : 0   1     -> 500       501 -> 511 0     1     -> 500      501 -> 511
 * devArr : 0   0     -> 200       200 -> 200 100   100   -> 100      100 -> 100
 * sum1   : 0   1*2/2 -> 500*501/2 0   -> 0
 *        : 0   1     -> 125250    0   -> 0
 * sum2   :                                   0     1*2/2 -> 500*501/2 0  -> 0
 *        :                                   0     1     -> 125250    0  -> 0
 * outArr :2400 2401  -> 127650    2400->2400 53600 53601 -> 178850 53600->53600
 *
 * devArrSum1 is 2400 (12 threads fill 200, 12*200 = 2400)
 * devArrSum2 is 53600(12 threads fill 200 = 2400, 512 threads fill 100 = 51200)
 */
static __global__ void coopKernel_Grid_1D_Block_1D(int *outputArr) {
  int index = blockDim.x * blockIdx.x + threadIdx.x;
  cooperative_groups::grid_group grid = cooperative_groups::this_grid();

  int sum1 = 0;
  if (blockIdx.x == 1) {
    devArr[index] = 100;
  }

  auto token1 = grid.barrier_arrive();

  if (blockIdx.x == 0) {
    cooperative_groups::thread_block block =
        cooperative_groups::this_thread_block();

    if (threadIdx.x >= 500) {
      devArr[index] = 200;
    }

    auto token2 = block.barrier_arrive();

    if ((threadIdx.x < 500) && (threadIdx.x != 0)) {
      int *local_data = new int[threadIdx.x];
      fillData(local_data, threadIdx.x);
      sum1 = getSumOfArrayElements(local_data, threadIdx.x);
      delete[] local_data;
    }

    block.barrier_wait(std::move(token2));

    int devArrSum1 = 0;
    for (int i = 0; i < 512; i++) {
      devArrSum1 = devArrSum1 + devArr[i];
    }
    outputArr[index] = sum1 + devArrSum1;
  }

  int sum2 = 0;
  if (blockIdx.x == 1) {
    if ((threadIdx.x < 500) && (threadIdx.x != 0)) {
      int *local_data = new int[threadIdx.x];
      fillData(local_data, threadIdx.x);
      sum2 = getSumOfArrayElements(local_data, threadIdx.x);
      delete[] local_data;
    }
  }

  grid.barrier_wait(std::move(token1));

  int devArrSum2 = 0;
  for (int i = 0; i < N; i++) {
    devArrSum2 = devArrSum2 + devArr[i];
  }

  if (outputArr[index] == 0) {
    outputArr[index] = sum2 + devArrSum2;
  }
}

/**
 * Test Description
 * ------------------------
 *  - This test case checks validates the behavior of
 *  - barrier_arrive & barrier_wait in Grid and block within
 *  - the same kernel and validates the output.
 * Test source
 * ------------------------
 *  - unit/cooperativeGrps/barrier_arrive_barrier_wait.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.1
 */
TEST_CASE("Unit_barrier_wait_barrier_arrive_GridAndBlock_1D") {
  int cooperativeLaunchSupported;
  HIP_CHECK(hipDeviceGetAttribute(&cooperativeLaunchSupported,
                                  hipDeviceAttributeCooperativeLaunch, 0));

  if (!cooperativeLaunchSupported) {
    HipTest::HIP_SKIP_TEST("Skipping test as CooperativeLaunch not supported");
  }

  int hostMem[N];
  for (int i = 0; i < N; i++) {
    hostMem[i] = 0;
  }

  resetGlobalDevArr<<<1, 1>>>();
  HIP_CHECK(hipDeviceSynchronize());

  int *devMem = nullptr;
  HIP_CHECK(hipMalloc(&devMem, N * sizeof(int)));
  REQUIRE(devMem != nullptr);
  HIP_CHECK(hipMemcpy(devMem, hostMem, N * sizeof(int), hipMemcpyHostToDevice));

  dim3 gridDim = dim3{2, 1, 1};
  dim3 blockDim = dim3{512, 1, 1};

  void *params[1];
  params[0] = &devMem;

  HIP_CHECK(hipLaunchCooperativeKernel((void *)(coopKernel_Grid_1D_Block_1D),
                                       gridDim, blockDim, params, 0, 0));
  HIP_CHECK(hipDeviceSynchronize());

  HIP_CHECK(hipMemcpy(hostMem, devMem, N * sizeof(int), hipMemcpyDefault));

  int expectedResultBlock1 = 2400;
  int expectedResultBlock2 = 53600;

  REQUIRE(hostMem[0] == expectedResultBlock1);
  REQUIRE(hostMem[N / 2] == expectedResultBlock2);

  for (int i = 1; i < (N / 2) - 12; i++) {
    INFO("At i : " << i << " Got : " << hostMem[i] << " Expected : "
                   << expectedResultBlock1 + (i * (i + 1) / 2));

    REQUIRE(hostMem[i] == expectedResultBlock1 + (i * (i + 1) / 2)); // 1 to 499

    INFO("At i : " << N / 2 + i << " Got : " << hostMem[N / 2 + i]
                   << " Expected : "
                   << expectedResultBlock2 + (i * (i + 1) / 2));

    REQUIRE(hostMem[N / 2 + i] ==
            expectedResultBlock2 + (i * (i + 1) / 2)); // 513 to 1011
  }

  for (int i = (N / 2) - 12; i < (N / 2); i++) {
    INFO("At i : " << i << " Got : " << hostMem[i]
                   << " Expected : " << expectedResultBlock1);

    REQUIRE(hostMem[i] == expectedResultBlock1); // 500 to 511

    INFO("At i : " << i << " Got : " << hostMem[i]
                   << " Expected : " << expectedResultBlock2);

    REQUIRE(hostMem[N / 2 + i] == expectedResultBlock2); // 1012 to 1023
  }

  HIP_CHECK(hipFree(devMem));
}
