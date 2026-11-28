/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>

#include <vector>

namespace hipHostUnregisterTests {
constexpr unsigned int allFlags = hipHostRegisterDefault |   // 0
                                  hipHostRegisterPortable |  // 1
                                  hipHostRegisterMapped |    // 2
                                  hipHostRegisterIoMemory    // 4
#if HT_NVIDIA
                                  | cudaHostRegisterReadOnly;  // 8
#else
    ;
#endif

inline void hipHostRegisterSupported() {
#if HT_NVIDIA
  // unable to query for cudaDevAttrHostRegisterSupported equivalent

  HIP_SKIP_TEST("hipHostRegister is not supported on this device.");
#endif
}


HIP_TEST_CASE(Unit_hipHostUnregister_MemoryNotAccessibleAfterUnregister) {
  hipHostRegisterSupported();

  // try all combinations of flags
  for (unsigned int flag = 0; flag <= allFlags; ++flag) {
#if defined(_WIN32)
    // hipHostRegisterIoMemory not supported on Windows; skip flags 4-7
    if (flag & hipHostRegisterIoMemory) continue;
#endif
      DYNAMIC_SECTION("Using flag: " << flag) {
      auto x = std::unique_ptr<int>(new int);
      HIP_CHECK(hipHostRegister(x.get(), sizeof(int), flag));

      void* device_memory;
      HIP_CHECK(hipHostGetDevicePointer(&device_memory, x.get(), 0));

      HIP_CHECK(hipHostUnregister(x.get()));
      HIP_CHECK_ERROR(hipHostGetDevicePointer(&device_memory, x.get(), 0), hipErrorInvalidValue);
    }
  }
}

HIP_TEST_CASE(Unit_hipHostUnregister_NullPtr) {
  HIP_CHECK_ERROR(hipHostUnregister(nullptr), hipErrorInvalidValue);
}

HIP_TEST_CASE(Unit_hipHostUnregister_Ptr_Different_Than_Specified_To_Register) {
  std::vector<int> alloc(2);
  HIP_CHECK(hipHostRegister(alloc.data(), alloc.size() * sizeof(int), 0));
  HIP_CHECK_ERROR(hipHostUnregister(&alloc.data()[1]), hipErrorHostMemoryNotRegistered);
  HIP_CHECK(hipHostUnregister(alloc.data()));
}

// An address inside a registration resolves to it, but only the registered address may unregister
// it; anything else must fail without touching the registration (it used to abort the process).
HIP_TEST_CASE(Unit_hipHostUnregister_InteriorPtr) {
  hipHostRegisterSupported();
  constexpr size_t kSize = 64 * 1024;
  std::vector<char> buffer(kSize);
  char* host = buffer.data();
  HIP_CHECK(hipHostRegister(host, kSize, 0));

  HIP_CHECK_ERROR(hipHostUnregister(host + 4096), hipErrorHostMemoryNotRegistered);
  HIP_CHECK_ERROR(hipHostUnregister(host + kSize - 1), hipErrorHostMemoryNotRegistered);

  // Still registered and usable.
  void* device_memory = nullptr;
  HIP_CHECK(hipHostGetDevicePointer(&device_memory, host, 0));
  HIP_CHECK(hipMemset(device_memory, 1, kSize));
  HIP_CHECK(hipDeviceSynchronize());
  REQUIRE(host[kSize - 1] == 1);
  HIP_CHECK(hipHostUnregister(host));
}

#if HT_AMD
// Each device can map registered memory at a VA other than the host address (on Windows always,
// from its own VA space). Passing that VA to hipHostUnregister or hipFree must fail without
// touching the registration (it aborted the process on Windows, and released the registration on
// Linux, leaving the host address pointing at a freed object).
HIP_TEST_CASE(Unit_hipHostUnregister_DevicePtr) {
  hipHostRegisterSupported();
  constexpr size_t kSize = 64 * 1024;
  std::vector<char> buffer(kSize);
  char* host = buffer.data();
  HIP_CHECK(hipHostRegister(host, kSize, hipHostRegisterPortable));
  const int num_devices = HipTest::getDeviceCount();
  for (int dev = 0; dev < num_devices; ++dev) {
    HIP_CHECK(hipSetDevice(dev));
    void* device_memory = nullptr;
    HIP_CHECK(hipHostGetDevicePointer(&device_memory, host, 0));
    if (device_memory == host) {
      continue;
    }
    INFO("device " << dev << " VA " << device_memory << " host " << static_cast<void*>(host));
    HIP_CHECK_ERROR(hipHostUnregister(device_memory), hipErrorHostMemoryNotRegistered);
    HIP_CHECK_ERROR(hipFree(device_memory), hipErrorInvalidValue);
    HIP_CHECK(hipMemset(device_memory, dev + 1, kSize));
    HIP_CHECK(hipDeviceSynchronize());
    REQUIRE(host[0] == dev + 1);
  }
  HIP_CHECK(hipSetDevice(0));
  HIP_CHECK(hipHostUnregister(host));
}
#endif

// Registered memory gets a VA on every device, from per-device VA spaces. Once the devices have
// used different amounts of VA, one buffer's VA on a device can lie inside another buffer's VA
// range on another device. Unregistering such a buffer while the other one is still registered
// aborted with "Memobj map does not have ptr" (ROCm/TheRock#6224, ComfyUI with a dGPU and an iGPU).
HIP_TEST_CASE(Unit_hipHostUnregister_MultiGpuOverlappingDeviceVA) {
  hipHostRegisterSupported();
  const int num_devices = HipTest::getDeviceCount();
  if (num_devices < 2) {
    HIP_SKIP_TEST(HipTest::SkipReason::kFewerThanTwoGpus);
  }
  // Use more device VA on the last device than on device 0 (streams and their queues), so the
  // per-device VAs of the buffers registered below do not line up across devices.
  HIP_CHECK(hipSetDevice(num_devices - 1));
  std::vector<hipStream_t> streams(8);
  char* scratch = nullptr;
  HIP_CHECK(hipMalloc(&scratch, 1 << 20));
  for (auto& stream : streams) {
    HIP_CHECK(hipStreamCreate(&stream));
    HIP_CHECK(hipMemsetAsync(scratch, 0, 1 << 20, stream));
  }
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipSetDevice(0));

  const size_t sizes[] = {64 << 20, 4 << 20, 16 << 20, 2 << 20, 32 << 20, 1 << 20};
  std::vector<std::vector<char>> buffers;
  for (size_t size : sizes) {
    buffers.emplace_back(size);
    HIP_CHECK(hipHostRegister(buffers.back().data(), size, hipHostRegisterPortable));
  }
  for (int dev = 0; dev < num_devices; ++dev) {
    HIP_CHECK(hipSetDevice(dev));
    for (auto& buffer : buffers) {
      void* device_memory = nullptr;
      HIP_CHECK(hipHostGetDevicePointer(&device_memory, buffer.data(), 0));
      HIP_CHECK(hipMemset(device_memory, dev + 1, 4096));
    }
    HIP_CHECK(hipDeviceSynchronize());
  }
  // Count how often one buffer's VA on a device falls inside another buffer's VA range on another
  // device: the layout that broke a shared pointer map. Hardware with a different VA layout may
  // produce none, and then this test cannot catch the bug.
  const size_t count = buffers.size();
  std::vector<std::vector<uintptr_t>> vas(num_devices, std::vector<uintptr_t>(count));
  for (int dev = 0; dev < num_devices; ++dev) {
    HIP_CHECK(hipSetDevice(dev));
    for (size_t i = 0; i < count; ++i) {
      void* device_memory = nullptr;
      HIP_CHECK(hipHostGetDevicePointer(&device_memory, buffers[i].data(), 0));
      vas[dev][i] = reinterpret_cast<uintptr_t>(device_memory);
    }
  }
  int overlaps = 0;
  for (int a = 0; a < num_devices; ++a) {
    for (int b = 0; b < num_devices; ++b) {
      for (size_t i = 0; i < count && a != b; ++i) {
        for (size_t j = 0; j < count; ++j) {
          if (i != j && vas[a][i] >= vas[b][j] && vas[a][i] < vas[b][j] + sizes[j]) {
            ++overlaps;
          }
        }
      }
    }
  }
  if (overlaps == 0) {
    WARN("no cross-device VA overlap produced; this run does not exercise ROCm/TheRock#6224");
  } else {
    INFO("cross-device VA overlaps: " << overlaps);
  }

  HIP_CHECK(hipSetDevice(0));
  // Newest first for every other buffer, then the rest oldest first.
  for (size_t i = count; i-- > 0;) {
    if ((count - 1 - i) % 2 == 0) {
      HIP_CHECK(hipHostUnregister(buffers[i].data()));
    }
  }
  for (size_t i = 0; i < count; ++i) {
    if ((count - 1 - i) % 2 != 0) {
      HIP_CHECK(hipHostUnregister(buffers[i].data()));
    }
  }
  for (auto& buffer : buffers) {
    REQUIRE(buffer[0] == num_devices);
    HIP_CHECK_ERROR(hipHostUnregister(buffer.data()), hipErrorHostMemoryNotRegistered);
  }

  HIP_CHECK(hipSetDevice(num_devices - 1));
  for (auto& stream : streams) {
    HIP_CHECK(hipStreamDestroy(stream));
  }
  HIP_CHECK(hipFree(scratch));
  HIP_CHECK(hipSetDevice(0));
}

HIP_TEST_CASE(Unit_hipHostUnregister_NotRegisteredPointer) {
  auto x = std::unique_ptr<int>(new int);
  HIP_CHECK_ERROR(hipHostUnregister(x.get()), hipErrorHostMemoryNotRegistered);
}

HIP_TEST_CASE(Unit_hipHostUnregister_AlreadyUnregisteredPointer) {
  hipHostRegisterSupported();

  // try all combinations of flags
  for (unsigned int flag = 0; flag <= allFlags; ++flag) {
#if defined(_WIN32)
    // hipHostRegisterIoMemory not supported on Windows; skip flags 4-7
    if (flag & hipHostRegisterIoMemory) continue;
#endif
    DYNAMIC_SECTION("Using flag: " << flag) {
      auto x = std::unique_ptr<int>(new int);
      HIP_CHECK(hipHostRegister(x.get(), sizeof(int), flag));
      HIP_CHECK(hipHostUnregister(x.get()));
      HIP_CHECK_ERROR(hipHostUnregister(x.get()), hipErrorHostMemoryNotRegistered);
    }
  }
}

HIP_TEST_CASE(Unit_hipHostUnregister_Capture) {
  constexpr size_t kBufferSize = 1024;
  auto buffer = std::make_unique<int[]>(kBufferSize);
  hipError_t capture_error = hipSuccess;

  HIP_CHECK_ERROR(hipHostRegister(buffer.get(), kBufferSize * sizeof(int), 0), capture_error);

  constexpr bool kRelaxedModeAllowed = true;
  BEGIN_CAPTURE_SYNC(capture_error, kRelaxedModeAllowed);
  HIP_CHECK_ERROR(hipHostUnregister(buffer.get()), capture_error);
  END_CAPTURE_SYNC(capture_error);

  if (capture_error != hipSuccess) {
    HIP_CHECK(hipHostUnregister(buffer.get()));
  }
}

}  // namespace hipHostUnregisterTests
