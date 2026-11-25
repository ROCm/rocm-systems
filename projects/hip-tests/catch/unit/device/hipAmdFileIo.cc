/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <resource_guards.hh>
#include <hip/amd_detail/hip_storage.h>
#include <fcntl.h>
#include <unistd.h>
#include <algorithm>
#include <cstdlib>
#include <vector>

/**
 * hipError_t hipAmdFileRead(hipAmdFileHandle_t handle, void* devicePtr, uint64_t size,
 *                           int64_t file_offset, uint64_t* size_copied, int32_t* status);
 * hipError_t hipAmdFileWrite(hipAmdFileHandle_t handle, void* devicePtr, uint64_t size,
 *                            int64_t file_offset, uint64_t* size_copied, int32_t* status);
 */

using FileIoFn = hipError_t (*)(hipAmdFileHandle_t, void*, uint64_t, int64_t, uint64_t*,
                                int32_t*);

static constexpr uint64_t kIoSize = 4096;

static FileIoFn GetFileIoFn(const char* name) {
  int version = 0;
  HIP_CHECK(hipRuntimeGetVersion(&version));
  void* fn = nullptr;
  HIP_CHECK(hipGetProcAddress(name, &fn, version, 0, nullptr));
  REQUIRE(fn != nullptr);
  return reinterpret_cast<FileIoFn>(fn);
}

// Unlinked right away so a failed REQUIRE cannot leave the file behind.
static int CreateTempFile() {
  char path[] = "/tmp/hipAmdFileIoXXXXXX";
  int fd = mkstemp(path);
  REQUIRE(fd >= 0);
  REQUIRE(unlink(path) == 0);
  REQUIRE(ftruncate(fd, kIoSize) == 0);
  return fd;
}

static void CheckLastErrorState(hipError_t expected) {
  // Peek rewrites the per-call error: read it first, clear it last.
  REQUIRE(hipExtGetLastError() == expected);
  REQUIRE(hipPeekAtLastError() == expected);
  REQUIRE(hipGetLastError() == expected);
  REQUIRE(hipGetLastError() == hipSuccess);
  static_cast<void>(hipExtGetLastError());
}

/**
 * Test Description
 * ------------------------
 *  - Calls hipAmdFileRead/hipAmdFileWrite with an invalid file descriptor and
 *    verifies the failure is returned and recorded as the last error.
 * Test source
 * ------------------------
 *  - unit/device/hipAmdFileIo.cc
 * Test requirements
 * ------------------------
 *  - Linux
 */
HIP_TEST_CASE(Unit_hipAmdFileIo_Negative_InvalidFd) {
  const char* api_name = nullptr;
  SECTION("read") { api_name = "hipAmdFileRead"; }
  SECTION("write") { api_name = "hipAmdFileWrite"; }
  FileIoFn file_io = GetFileIoFn(api_name);

  LinearAllocGuard<unsigned char> buf(LinearAllocs::hipMalloc, kIoSize);

  hipAmdFileHandle_t handle{};
  handle.fd = -1;
  uint64_t size_copied = 0;
  int32_t status = 0;
  hipError_t ret = file_io(handle, buf.ptr(), kIoSize, 0, &size_copied, &status);

  REQUIRE(ret == hipErrorUnknown);
  CheckLastErrorState(hipErrorUnknown);
}

/**
 * Test Description
 * ------------------------
 *  - Calls hipAmdFileRead/hipAmdFileWrite with a null device pointer and
 *    verifies the failure is returned and recorded as the last error.
 * Test source
 * ------------------------
 *  - unit/device/hipAmdFileIo.cc
 * Test requirements
 * ------------------------
 *  - Linux
 */
HIP_TEST_CASE(Unit_hipAmdFileIo_Negative_NullDevicePtr) {
  const char* api_name = nullptr;
  SECTION("read") { api_name = "hipAmdFileRead"; }
  SECTION("write") { api_name = "hipAmdFileWrite"; }
  FileIoFn file_io = GetFileIoFn(api_name);

  int fd = CreateTempFile();

  hipAmdFileHandle_t handle{};
  handle.fd = fd;
  uint64_t size_copied = 0;
  int32_t status = 0;
  hipError_t ret = file_io(handle, nullptr, kIoSize, 0, &size_copied, &status);

  REQUIRE(ret == hipErrorUnknown);
  CheckLastErrorState(hipErrorUnknown);

  REQUIRE(close(fd) == 0);
}

/**
 * Test Description
 * ------------------------
 *  - Calls hipAmdFileRead/hipAmdFileWrite with valid arguments. Hosts without
 *    AIS support may fail the transfer; any failure must match the recorded
 *    last error.
 * Test source
 * ------------------------
 *  - unit/device/hipAmdFileIo.cc
 * Test requirements
 * ------------------------
 *  - Linux
 */
HIP_TEST_CASE(Unit_hipAmdFileIo_ValidArgs_ErrorStateConsistent) {
  const char* api_name = nullptr;
  SECTION("read") { api_name = "hipAmdFileRead"; }
  SECTION("write") { api_name = "hipAmdFileWrite"; }
  FileIoFn file_io = GetFileIoFn(api_name);

  int fd = CreateTempFile();
  LinearAllocGuard<unsigned char> buf(LinearAllocs::hipMalloc, kIoSize);
  HIP_CHECK(hipGetLastError());

  hipAmdFileHandle_t handle{};
  handle.fd = fd;
  uint64_t size_copied = 0;
  int32_t status = 0;
  hipError_t ret = file_io(handle, buf.ptr(), kIoSize, 0, &size_copied, &status);

  if (ret == hipSuccess) {
    REQUIRE(size_copied <= kIoSize);
    REQUIRE(hipExtGetLastError() == hipSuccess);
  } else {
    REQUIRE(hipPeekAtLastError() == ret);
    REQUIRE(hipGetLastError() == ret);
  }

  REQUIRE(close(fd) == 0);
}

/**
 * Test Description
 * ------------------------
 *  - Writes a device buffer to an O_DIRECT file with hipAmdFileWrite, reads it
 *    back into a second device buffer with hipAmdFileRead and verifies the
 *    data and that no last error is recorded.
 * Test source
 * ------------------------
 *  - unit/device/hipAmdFileIo.cc
 * Test requirements
 * ------------------------
 *  - Linux
 *  - AIS-capable GPU and storage
 */
HIP_TEST_CASE(Unit_hipAmdFileIo_Positive_WriteRead) {
  FileIoFn file_write = GetFileIoFn("hipAmdFileWrite");
  FileIoFn file_read = GetFileIoFn("hipAmdFileRead");

  constexpr unsigned char kPattern = 0xA5;
  int fd = CreateTempFile();
  REQUIRE(fcntl(fd, F_SETFL, fcntl(fd, F_GETFL) | O_DIRECT) == 0);

  LinearAllocGuard<unsigned char> src(LinearAllocs::hipMalloc, kIoSize);
  LinearAllocGuard<unsigned char> dst(LinearAllocs::hipMalloc, kIoSize);
  HIP_CHECK(hipMemset(src.ptr(), kPattern, kIoSize));
  HIP_CHECK(hipMemset(dst.ptr(), 0, kIoSize));
  HIP_CHECK(hipDeviceSynchronize());

  hipAmdFileHandle_t handle{};
  handle.fd = fd;
  uint64_t size_copied = 0;
  int32_t status = 0;
  hipError_t ret = file_write(handle, src.ptr(), kIoSize, 0, &size_copied, &status);
  REQUIRE(ret == hipSuccess);
  REQUIRE(size_copied == kIoSize);
  REQUIRE(hipGetLastError() == hipSuccess);

  size_copied = 0;
  status = 0;
  ret = file_read(handle, dst.ptr(), kIoSize, 0, &size_copied, &status);
  REQUIRE(ret == hipSuccess);
  REQUIRE(size_copied == kIoSize);
  REQUIRE(hipGetLastError() == hipSuccess);

  std::vector<unsigned char> host(kIoSize, 0);
  HIP_CHECK(hipMemcpy(host.data(), dst.ptr(), kIoSize, hipMemcpyDeviceToHost));
  REQUIRE(std::all_of(host.begin(), host.end(), [](unsigned char b) { return b == kPattern; }));

  REQUIRE(close(fd) == 0);
}
