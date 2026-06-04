#pragma once

#include <hip/hip_runtime_api.h>

#include <dlfcn.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

#ifndef __AFL_FUZZ_INIT
#define __AFL_FUZZ_INIT()
#endif

#ifndef __AFL_INIT
#define __AFL_INIT() ((void)0)
#endif

#ifndef __AFL_LOOP
#define __AFL_LOOP(_count) rj_fuzz::fallback_afl_loop_once()
#endif

extern "C" {
int rocjitsu_afl_persistent_begin() __attribute__((weak));
int rocjitsu_afl_persistent_end() __attribute__((weak));
}

namespace rj_fuzz {

inline bool fallback_afl_loop_once() {
  static bool done = false;
  if (done)
    return false;
  done = true;
  return true;
}

inline std::vector<uint8_t> read_input(int argc, char **argv) {
  std::vector<uint8_t> bytes;
  if (argc > 1) {
    std::ifstream file(argv[1], std::ios::binary);
    bytes.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
  } else {
    bytes.assign(std::istreambuf_iterator<char>(std::cin), std::istreambuf_iterator<char>());
  }

  if (bytes.empty())
    bytes = {0, 1, 2, 3, 5, 8, 13, 21};
  return bytes;
}

class ByteStream {
public:
  explicit ByteStream(std::vector<uint8_t> bytes) : bytes_(std::move(bytes)) {
    if (bytes_.empty())
      bytes_ = {0, 1, 2, 3, 5, 8, 13, 21};
  }

  uint8_t next() {
    const uint8_t value = bytes_[offset_ % bytes_.size()];
    ++offset_;
    return value;
  }

  template <typename T, size_t N> T pick(const std::array<T, N> &values) {
    return values[next() % values.size()];
  }

  float next_float() {
    static constexpr std::array<float, 16> kInterestingValues = {
        -4.0f, -2.0f, -1.0f, -0.5f, -0.0f, 0.0f,   0.125f, 0.5f,
        1.0f,  2.0f,  4.0f,  8.0f,  16.0f, -16.0f, 0.25f,  -0.25f};
    return pick(kInterestingValues);
  }

private:
  std::vector<uint8_t> bytes_;
  size_t offset_ = 0;
};

inline int interesting_dim(ByteStream &stream) {
  static constexpr std::array<int, 20> kDims = {1,  2,  3,  4,  5,  7,  8,  15, 16,  17,
                                                31, 32, 33, 48, 63, 64, 65, 96, 127, 128};
  return stream.pick(kDims);
}

inline void fill_floats(std::vector<float> &values, ByteStream &stream) {
  for (float &value : values)
    value = stream.next_float();
}

inline void crash_on_hip_error(const char *what, hipError_t status) {
  if (status == hipSuccess)
    return;
  std::fprintf(stderr, "%s failed: %s\n", what, hipGetErrorString(status));
  std::abort();
}

inline int persistent_iteration_begin() {
  if (rocjitsu_afl_persistent_begin != nullptr)
    return rocjitsu_afl_persistent_begin();
  return 0;
}

inline int persistent_iteration_end() {
  if (rocjitsu_afl_persistent_end != nullptr)
    return rocjitsu_afl_persistent_end();

  const hipError_t status = hipDeviceSynchronize();
  if (status == hipSuccess)
    return 0;

  std::fprintf(stderr, "hipDeviceSynchronize failed: %s\n", hipGetErrorString(status));
  return static_cast<int>(status);
}

template <typename T> class DeviceBuffer {
public:
  DeviceBuffer() = default;
  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;

  ~DeviceBuffer() {
    if (data_ != nullptr)
      (void)hipFree(data_);
  }

  bool allocate(size_t count) {
    count_ = count;
    return hipMalloc(reinterpret_cast<void **>(&data_), sizeof(T) * count_) == hipSuccess;
  }

  bool copy_from_host(const std::vector<T> &host) {
    return hipMemcpy(data_, host.data(), sizeof(T) * host.size(), hipMemcpyHostToDevice) ==
           hipSuccess;
  }

  bool copy_to_host(std::vector<T> &host) const {
    return hipMemcpy(host.data(), data_, sizeof(T) * host.size(), hipMemcpyDeviceToHost) ==
           hipSuccess;
  }

  T *get() const { return data_; }
  size_t count() const { return count_; }

private:
  T *data_ = nullptr;
  size_t count_ = 0;
};

inline bool have_hip_device() {
  int device_count = 0;
  return hipGetDeviceCount(&device_count) == hipSuccess && device_count > 0;
}

using PersistentHook = int (*)();

inline PersistentHook load_persistent_hook(const char *name) {
  return reinterpret_cast<PersistentHook>(dlsym(RTLD_DEFAULT, name));
}

inline bool persistent_hooks_required() {
  return std::getenv("ROCJITSU_AFL_REQUIRE_PERSISTENT_HOOKS") != nullptr;
}

inline int call_persistent_hook(PersistentHook hook, const char *name) {
  if (hook == nullptr)
    return 0;
  const int rc = hook();
  if (rc != 0)
    std::fprintf(stderr, "%s failed with rc=%d\n", name, rc);
  return rc;
}

} // namespace rj_fuzz
