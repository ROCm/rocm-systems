/*
Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.

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

#include "execution_control_common.hh"

#include <hip_test_common.hh>
#include <resource_guards.hh>

TEST_CASE("Unit_hipLaunchByPtr_Positive_Basic") {
  LinearAllocGuard<int> alloc(LinearAllocs::hipMallocManaged, 4);

  SECTION("hipConfigureCall") { HIP_CHECK(hipConfigureCall(dim3{1}, dim3{1}, 0, nullptr)); }

  SECTION("__hipPushCallConfiguration") {
    HIP_CHECK(__hipPushCallConfiguration(dim3{1}, dim3{1}, 0, nullptr));
  }

  int* arg = alloc.ptr();
  HIP_CHECK(hipSetupArgument(&arg, sizeof(int*), 0));

  HIP_CHECK(hipLaunchByPtr(reinterpret_cast<void*>(kernel_42)));
  HIP_CHECK(hipDeviceSynchronize());

  REQUIRE(alloc.ptr()[0] == 42);
}

TEST_CASE("Unit_hipLaunchByPtr_Negative_Parameters") {
  HIP_CHECK(hipConfigureCall(dim3{1}, dim3{1}, 0, nullptr));
  HIP_CHECK_ERROR(hipLaunchByPtr(nullptr), hipErrorInvalidDeviceFunction);
}

TEST_CASE("Unit___hipPushCallConfiguration_Positive_Basic") {
  StreamGuard stream_guard(Streams::created);
  HIP_CHECK(__hipPushCallConfiguration(dim3{1, 2, 3}, dim3{3, 2, 1}, 1024, stream_guard.stream()));

  dim3 grid;
  dim3 block;
  size_t shmem;
  hipStream_t stream;
  HIP_CHECK(__hipPopCallConfiguration(&grid, &block, &shmem, &stream));

  REQUIRE(grid.x == 1);
  REQUIRE(grid.y == 2);
  REQUIRE(grid.z == 3);
  REQUIRE(block.x == 3);
  REQUIRE(block.y == 2);
  REQUIRE(block.z == 1);
  REQUIRE(shmem == 1024);
  REQUIRE(stream == stream_guard.stream());
}