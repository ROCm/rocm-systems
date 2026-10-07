/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>

#if defined(__linux__) && defined(__HIP_PLATFORM_AMD__)
#include <hip_test_process.hh>
#include <algorithm>
#include <array>
#include <fstream>

namespace {
constexpr size_t kBytes = 4096;
constexpr unsigned char kInitial = 0x35;
constexpr unsigned char kWritten = 0x79;
__global__ void writeImported(unsigned char* data) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < kBytes) data[i] = kWritten;
}
}  // namespace

// Run only through SpawnProc, so HIP initializes after the visibility mask is set.
TEST_CASE("hipIpcMemVisibilityImporter", "[.]") {
  alarm(60);  // Bound a stuck runtime call in the isolated importer.
  const char* path = getenv("HIP_TEST_IPC_VISIBILITY_HANDLE");
  REQUIRE(path != nullptr);
  std::ifstream input(path, std::ios::binary);
  hipIpcMemHandle_t handle{};
  REQUIRE(input.read(reinterpret_cast<char*>(&handle), sizeof(handle)).good());
  int count = 0;
  HIP_CHECK(hipGetDeviceCount(&count));
  const char* expected = getenv("HIP_TEST_IPC_VISIBILITY_COUNT");
  REQUIRE(expected != nullptr);
  REQUIRE(count == std::stoi(expected));
  HIP_CHECK(hipSetDevice(0));
  void* imported = nullptr;
  HIP_CHECK(hipIpcOpenMemHandle(&imported, handle, hipIpcMemLazyEnablePeerAccess));
  // Exercise the cached-handle path, including its retain/release pair.
  void* again = nullptr;
  HIP_CHECK(hipIpcOpenMemHandle(&again, handle, hipIpcMemLazyEnablePeerAccess));
  REQUIRE(again == imported);
  HIP_CHECK(hipIpcCloseMemHandle(again));
  std::array<unsigned char, kBytes> data{};
  HIP_CHECK(hipMemcpy(data.data(), imported, kBytes, hipMemcpyDeviceToHost));
  REQUIRE(std::all_of(data.begin(), data.end(), [](auto v) { return v == kInitial; }));
  writeImported<<<kBytes / 256, 256>>>(static_cast<unsigned char*>(imported));
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipIpcCloseMemHandle(imported));
}

/**
 * Test Description
 * ------------------------
 * Export on ordinal 1 and import in fresh processes with fewer or reordered
 * visible devices. Check data in both directions without enabling peer access.
 * Preserve ROCR_VISIBLE_DEVICES so the test stays inside scheduler assignments.
 * Test requirements
 * ------------------------
 * Linux, AMD, two GPUs with peer access. HIP_VISIBLE_DEVICES and
 * CUDA_VISIBLE_DEVICES must be unset. Use ROCR_VISIBLE_DEVICES for isolation.
 */
HIP_TEST_CASE(Unit_hipIpcOpenMemHandle_VisibleDevices) {
  for (const char* name : {"HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"}) {
    const char* value = getenv(name);
    if (value != nullptr && *value != '\0') {
      HIP_SKIP_TEST("Use ROCR_VISIBLE_DEVICES to restrict this test's GPUs");
      return;
    }
  }
  int count = 0;
  HIP_CHECK(hipGetDeviceCount(&count));
  if (count < 2) {
    HIP_SKIP_TEST("Requires two GPUs");
    return;
  }
  int can_access = 0;
  HIP_CHECK(hipDeviceCanAccessPeer(&can_access, 0, 1));
  if (!can_access) {
    HIP_SKIP_TEST("Requires peer access from GPU 0 to GPU 1");
    return;
  }
  int previous = 0;
  HIP_CHECK(hipGetDevice(&previous));
  struct Resources {
    int previous;
    void* allocation = nullptr;
    std::string path;
    ~Resources() {
      if (allocation) (void)hipFree(allocation);
      if (!path.empty()) unlink(path.c_str());
      (void)hipSetDevice(previous);
    }
  } resources{previous, nullptr, {}};
  HIP_CHECK(hipSetDevice(1));
  HIP_CHECK(hipMalloc(&resources.allocation, kBytes));
  hipIpcMemHandle_t handle{};
  HIP_CHECK(hipIpcGetMemHandle(&handle, resources.allocation));
  char path[] = "/tmp/hip-ipc-visibility-XXXXXX";
  int fd = mkstemp(path);
  REQUIRE(fd >= 0);
  resources.path = path;
  bool written = hip::writeAll(fd, &handle, sizeof(handle));
  close(fd);
  REQUIRE(written);

  // Out-of-range ordinal on another GPU, out-of-range ordinal on the same
  // physical GPU, and an in-range ordinal that names the wrong physical GPU.
  for (const char* mask : {"0", "1", "1,0"}) {
    INFO("Importer HIP_VISIBLE_DEVICES=" << mask);
    HIP_CHECK(hipMemset(resources.allocation, kInitial, kBytes));
    HIP_CHECK(hipDeviceSynchronize());
    hip::SpawnProc importer(getSelfExePath());
    importer.setEnv("HIP_VISIBLE_DEVICES", mask);
    importer.setEnv("HIP_TEST_IPC_VISIBILITY_HANDLE", resources.path);
    importer.setEnv("HIP_TEST_IPC_VISIBILITY_COUNT", std::string(mask) == "1,0" ? "2" : "1");
    REQUIRE(importer.run("hipIpcMemVisibilityImporter") == 0);
    std::array<unsigned char, kBytes> data{};
    HIP_CHECK(hipMemcpy(data.data(), resources.allocation, kBytes, hipMemcpyDeviceToHost));
    REQUIRE(std::all_of(data.begin(), data.end(), [](auto v) { return v == kWritten; }));
  }
}
#endif
