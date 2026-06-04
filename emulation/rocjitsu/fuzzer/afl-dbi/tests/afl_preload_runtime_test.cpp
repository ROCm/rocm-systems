// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu_fuzzer/afl_runtime.h"

#include <gtest/gtest.h>

#include <dlfcn.h>
#include <sys/shm.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>

#ifndef ROCJITSU_AFL_PRELOAD_PATH
#error "ROCJITSU_AFL_PRELOAD_PATH must point to librocjitsu_afl_preload.so"
#endif

#ifndef ROCJITSU_AFL_FAKE_HIP_RUNTIME_PATH
#error "ROCJITSU_AFL_FAKE_HIP_RUNTIME_PATH must point to the fake HIP runtime"
#endif

namespace {

using rocjitsu::fuzzer::afl::kCoverageSlots;
using rocjitsu::fuzzer::afl::kDeviceStart;
using rocjitsu::fuzzer::afl::kMapSize;

using PersistentFn = int (*)();
using DeviceCountersFn = uint32_t *(*)();
using DeviceCounterBytesFn = size_t (*)();
using DeviceSynchronizeCallsFn = int (*)();

class DlHandle {
public:
  explicit DlHandle(const char *path, int flags) : handle_(dlopen(path, flags)) {}
  DlHandle(const DlHandle &) = delete;
  DlHandle &operator=(const DlHandle &) = delete;
  ~DlHandle() {
    if (handle_ != nullptr)
      dlclose(handle_);
  }

  void *symbol(const char *name) const { return dlsym(handle_, name); }
  void *get() const { return handle_; }

private:
  void *handle_ = nullptr;
};

class SharedMemory {
public:
  SharedMemory() : id_(shmget(IPC_PRIVATE, kMapSize, IPC_CREAT | 0600)) {
    EXPECT_GE(id_, 0);
    if (id_ >= 0) {
      data_ = static_cast<uint8_t *>(shmat(id_, nullptr, 0));
      EXPECT_NE(data_, reinterpret_cast<uint8_t *>(-1));
      if (data_ != reinterpret_cast<uint8_t *>(-1))
        std::memset(data_, 0, kMapSize);
    }
  }

  SharedMemory(const SharedMemory &) = delete;
  SharedMemory &operator=(const SharedMemory &) = delete;

  ~SharedMemory() {
    if (data_ != nullptr && data_ != reinterpret_cast<uint8_t *>(-1))
      shmdt(data_);
    if (id_ >= 0)
      shmctl(id_, IPC_RMID, nullptr);
  }

  int id() const { return id_; }
  uint8_t *data() const { return data_; }

private:
  int id_ = -1;
  uint8_t *data_ = nullptr;
};

} // namespace

TEST(RocjitsuAflPreloadRuntimeTest, PersistentBeginEndMergesBasicBlockCounters) {
  SharedMemory shm;
  ASSERT_GE(shm.id(), 0);
  ASSERT_NE(shm.data(), nullptr);
  setenv("__AFL_SHM_ID", std::to_string(shm.id()).c_str(), 1);
  setenv("ROCJITSU_AFL_HIP_RUNTIME_PATH", ROCJITSU_AFL_FAKE_HIP_RUNTIME_PATH, 1);

  DlHandle preload(ROCJITSU_AFL_PRELOAD_PATH, RTLD_LAZY | RTLD_LOCAL);
  ASSERT_NE(preload.get(), nullptr) << dlerror();

  auto persistent_begin =
      reinterpret_cast<PersistentFn>(preload.symbol("rocjitsu_afl_persistent_begin"));
  auto persistent_end =
      reinterpret_cast<PersistentFn>(preload.symbol("rocjitsu_afl_persistent_end"));
  ASSERT_NE(persistent_begin, nullptr);
  ASSERT_NE(persistent_end, nullptr);

  ASSERT_EQ(persistent_begin(), 0);

  DlHandle fake_hip(ROCJITSU_AFL_FAKE_HIP_RUNTIME_PATH, RTLD_LAZY | RTLD_LOCAL);
  ASSERT_NE(fake_hip.get(), nullptr) << dlerror();
  auto fake_device_counters =
      reinterpret_cast<DeviceCountersFn>(fake_hip.symbol("rocfuzz_fake_device_counters"));
  auto fake_device_counter_bytes =
      reinterpret_cast<DeviceCounterBytesFn>(fake_hip.symbol("rocfuzz_fake_device_counter_bytes"));
  auto fake_device_synchronize_calls = reinterpret_cast<DeviceSynchronizeCallsFn>(
      fake_hip.symbol("rocfuzz_fake_device_synchronize_calls"));
  ASSERT_NE(fake_device_counters, nullptr);
  ASSERT_NE(fake_device_counter_bytes, nullptr);
  ASSERT_NE(fake_device_synchronize_calls, nullptr);

  uint32_t *device_counters = fake_device_counters();
  ASSERT_NE(device_counters, nullptr);
  EXPECT_EQ(fake_device_counter_bytes(), sizeof(uint32_t) * kCoverageSlots);
  for (uint32_t i = 0; i < kCoverageSlots; ++i)
    ASSERT_EQ(device_counters[i], 0u);

  device_counters[0] = 1;
  device_counters[1] = 3;
  device_counters[2] = 512;
  ASSERT_EQ(persistent_end(), 0);

  EXPECT_EQ(shm.data()[kDeviceStart], 1u);
  EXPECT_EQ(shm.data()[kDeviceStart + 1], 3u);
  EXPECT_EQ(shm.data()[kDeviceStart + 2], 255u);
  EXPECT_GE(fake_device_synchronize_calls(), 1);

  unsetenv("ROCJITSU_AFL_HIP_RUNTIME_PATH");
  unsetenv("__AFL_SHM_ID");
}
