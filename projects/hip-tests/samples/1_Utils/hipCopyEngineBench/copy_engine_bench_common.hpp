/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <hip/hip_ext.h>
#include <hip/hip_ext_copy_engine.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#define HIP_CHECK(expr)                                                                            \
  do {                                                                                             \
    const hipError_t hip_check_status = (expr);                                                    \
    if (hip_check_status != hipSuccess) {                                                          \
      std::fprintf(stderr, "HIP error %s (%d) at %s:%d: %s\n", hipGetErrorName(hip_check_status),  \
                   static_cast<int>(hip_check_status), __FILE__, __LINE__, #expr);                 \
      std::exit(EXIT_FAILURE);                                                                     \
    }                                                                                              \
  } while (0)

namespace copy_engine_bench {

constexpr uint32_t kMagicValue = 0xDEADBEEF;
// Device-side bound on a signal wait, in spins; far above any healthy copy.
constexpr uint64_t kMaxSpins = 1ull << 34;

//===----------------------------------------------------------------------===//
// Command line
//===----------------------------------------------------------------------===//

class ArgParser {
 public:
  ArgParser(std::string program, std::string summary)
      : program_(std::move(program)), summary_(std::move(summary)) {}

  // |names| is a comma separated list such as "-b,--minCopySize".
  template <typename T>
  void AddOption(const std::string& names, T* value, const std::string& help) {
    Entry entry;
    entry.names = Split(names);
    entry.help = help + " [" + Show(*value) + "]";
    entry.takes_value = true;
    entry.set = [value](const std::string& text) {
      if constexpr (std::is_same_v<T, std::string>) {
        *value = text;
        return true;
      } else {
        if (std::is_unsigned_v<T> && !text.empty() && text[0] == '-') return false;
        std::istringstream in(text);
        T parsed{};
        in >> parsed;
        if (in.fail() || !in.eof()) return false;
        *value = parsed;
        return true;
      }
    };
    entries_.push_back(std::move(entry));
  }

  void AddFlag(const std::string& names, bool* value, const std::string& help) {
    Entry entry;
    entry.names = Split(names);
    entry.help = help;
    entry.takes_value = false;
    entry.set = [value](const std::string&) {
      *value = true;
      return true;
    };
    entries_.push_back(std::move(entry));
  }

  // Returns true when the program should continue; otherwise |exit_code| is what main returns.
  bool Parse(int argc, char** argv, int* exit_code) const {
    for (int i = 1; i < argc; ++i) {
      std::string arg = argv[i];
      if (arg == "-h" || arg == "--help") {
        PrintUsage(std::cout);
        *exit_code = 0;
        return false;
      }
      std::string inline_value;
      bool has_inline_value = false;
      const size_t equals = arg.find('=');
      if (arg.rfind("--", 0) == 0 && equals != std::string::npos) {
        inline_value = arg.substr(equals + 1);
        arg = arg.substr(0, equals);
        has_inline_value = true;
      }
      const Entry* entry = Find(arg);
      if (entry == nullptr) return Fail("unknown option '" + arg + "'", exit_code);
      std::string value;
      if (entry->takes_value) {
        if (has_inline_value) {
          value = inline_value;
        } else if (i + 1 < argc) {
          value = argv[++i];
        } else {
          return Fail("option '" + arg + "' needs a value", exit_code);
        }
      }
      if (!entry->set(value)) {
        return Fail("invalid value '" + value + "' for option '" + arg + "'", exit_code);
      }
    }
    return true;
  }

  void PrintUsage(std::ostream& out) const {
    out << summary_ << "\n\nUsage: " << program_ << " [options]\n\nOptions:\n";
    for (const Entry& entry : entries_) {
      std::string names;
      for (const std::string& name : entry.names) names += (names.empty() ? "" : ",") + name;
      if (entry.takes_value) names += " <value>";
      out << "  " << names;
      for (size_t pad = names.size(); pad < 40; ++pad) out << ' ';
      out << ' ' << entry.help << '\n';
    }
    out << "  -h,--help" << std::string(31, ' ') << " Show this message\n";
  }

 private:
  struct Entry {
    std::vector<std::string> names;
    std::string help;
    bool takes_value = false;
    std::function<bool(const std::string&)> set;
  };

  static std::vector<std::string> Split(const std::string& names) {
    std::vector<std::string> out;
    std::istringstream in(names);
    std::string name;
    while (std::getline(in, name, ',')) out.push_back(name);
    return out;
  }

  template <typename T> static std::string Show(const T& value) {
    std::ostringstream out;
    out << value;
    return out.str();
  }

  const Entry* Find(const std::string& name) const {
    for (const Entry& entry : entries_) {
      if (std::find(entry.names.begin(), entry.names.end(), name) != entry.names.end()) {
        return &entry;
      }
    }
    return nullptr;
  }

  bool Fail(const std::string& message, int* exit_code) const {
    std::cerr << "error: " << message << "\n\n";
    PrintUsage(std::cerr);
    *exit_code = 2;
    return false;
  }

  std::string program_;
  std::string summary_;
  std::vector<Entry> entries_;
};

//===----------------------------------------------------------------------===//
// CSV
//===----------------------------------------------------------------------===//

class CsvWriter {
 public:
  explicit CsvWriter(const std::string& path) : stream_(path) {
    if (!stream_) {
      std::cerr << "error: cannot open output file '" << path << "'\n";
      std::exit(EXIT_FAILURE);
    }
  }

  void WriteHeader(const std::vector<std::string>& columns) {
    bool first = true;
    for (const std::string& column : columns) WriteValue(column, first);
    stream_ << '\n';
  }

  template <typename... Values> void WriteRow(const Values&... values) {
    bool first = true;
    (WriteValue(values, first), ...);
    stream_ << '\n';
  }

 private:
  void WriteValue(const std::string& value, bool& first) {
    WriteSeparator(first);
    const bool quoted = value.find_first_of(",\"\n") != std::string::npos;
    if (quoted) stream_ << '"';
    for (char character : value) {
      if (character == '"') stream_ << '"';
      stream_ << character;
    }
    if (quoted) stream_ << '"';
  }

  template <typename Value> void WriteValue(const Value& value, bool& first) {
    WriteSeparator(first);
    stream_ << value;
  }

  void WriteSeparator(bool& first) {
    if (!first) stream_ << ',';
    first = false;
  }

  std::ofstream stream_;
};

//===----------------------------------------------------------------------===//
// Statistics
//===----------------------------------------------------------------------===//

inline std::pair<double, double> MeanStd(const std::vector<double>& values) {
  if (values.empty()) return {0.0, 0.0};
  double sum = 0.0;
  for (double value : values) sum += value;
  const double mean = sum / static_cast<double>(values.size());
  double squares = 0.0;
  for (double value : values) squares += (value - mean) * (value - mean);
  return {mean, std::sqrt(squares / static_cast<double>(values.size()))};
}

//===----------------------------------------------------------------------===//
// Device setup
//===----------------------------------------------------------------------===//

// Where the per-copy completion words live. Each submitted copy needs a word of its own;
// hipMallocSignalMemory only hands out single 8-byte words, so arrays are plain allocations.
enum class SignalMemory { kDevice, kHost };

inline bool ParseSignalMemory(const std::string& text, SignalMemory* out) {
  if (text == "device") {
    *out = SignalMemory::kDevice;
  } else if (text == "host") {
    *out = SignalMemory::kHost;
  } else {
    return false;
  }
  return true;
}

inline uint64_t* AllocSignals(size_t count, SignalMemory kind) {
  const size_t bytes = std::max<size_t>(count, 1) * sizeof(uint64_t);
  void* ptr = nullptr;
  if (kind == SignalMemory::kDevice) {
    HIP_CHECK(hipExtMallocWithFlags(&ptr, bytes, hipDeviceMallocUncached));
    HIP_CHECK(hipMemset(ptr, 0, bytes));
  } else {
    HIP_CHECK(hipHostMalloc(&ptr, bytes, hipHostMallocCoherent));
    std::memset(ptr, 0, bytes);
  }
  return static_cast<uint64_t*>(ptr);
}

inline void FreeSignals(uint64_t* signals, SignalMemory kind) {
  if (kind == SignalMemory::kDevice) {
    HIP_CHECK(hipFree(signals));
  } else {
    HIP_CHECK(hipHostFree(signals));
  }
}

inline void* AllocUncached(size_t bytes) {
  void* ptr = nullptr;
  HIP_CHECK(hipExtMallocWithFlags(&ptr, bytes, hipDeviceMallocUncached));
  return ptr;
}

constexpr char kCopyEngineUnavailable[] =
    "kernel-side copies are off (set DEBUG_CLR_ENABLE_ABCE=1 and GPU_MAX_SDMA_DEVICE_QUEUES=16) "
    "or unsupported on this GPU";

// Makes |src_device| current. Returns false, after saying why, when it is not a device.
inline bool SelectSourceDevice(int src_device) {
  int num_devices = 0;
  HIP_CHECK(hipGetDeviceCount(&num_devices));
  if (src_device < 0 || src_device >= num_devices) {
    std::cerr << "error: --srcGpu " << src_device << " is out of range (" << num_devices
              << " devices)\n";
    return false;
  }
  HIP_CHECK(hipSetDevice(src_device));
  return true;
}

// Most handles one kernel can hold: they travel by value, as one launch argument.
constexpr size_t kMaxHandles = 64;

struct HandleSet {
  hipExtCopyEngineDeviceHandle handles[kMaxHandles];
};

// Asks for |count| handles a kernel on |src_device| submits through for copies from |src_device|
// memory to |dst_device| memory, best engine first, and appends them to |out|. Returns false,
// after saying why, when the runtime has fewer: kernel-side copies are off, or the pair has fewer
// engines (or the device's ring limit is lower) than asked for.
inline bool GetCopyEngineHandles(int src_device, int dst_device, int count,
                                 std::vector<hipExtCopyEngineDeviceHandle>* out) {
  std::vector<hipExtCopyEngineDeviceHandle> handles(count);
  int num_handles = 0;
  const hipError_t status = hipExtCopyEngineGetDeviceHandles(src_device, src_device, dst_device,
                                                             count, handles.data(), &num_handles);
  if (status == hipErrorNotSupported) {
    std::cerr << "error: " << kCopyEngineUnavailable << "\n";
    return false;
  }
  HIP_CHECK(status);
  if (num_handles < count) {
    std::cerr << "error: GPU " << src_device << " to GPU " << dst_device << " has " << num_handles
              << " copy engine(s), " << count
              << " asked for (raise GPU_MAX_SDMA_DEVICE_QUEUES, or ask for fewer)\n";
    return false;
  }
  out->insert(out->end(), handles.begin(), handles.begin() + count);
  return true;
}

// Lets copies submitted on |src_device| reach memory on |dst_device|.
inline bool EnablePeerAccess(int src_device, int dst_device) {
  HIP_CHECK(hipSetDevice(src_device));
  int can_access = 0;
  HIP_CHECK(hipDeviceCanAccessPeer(&can_access, src_device, dst_device));
  if (!can_access) {
    std::cerr << "error: GPU " << src_device << " cannot access GPU " << dst_device << "\n";
    return false;
  }
  const hipError_t status = hipDeviceEnablePeerAccess(dst_device, 0);
  if (status != hipErrorPeerAccessAlreadyEnabled) HIP_CHECK(status);
  return true;
}

// The first |count| devices other than |src_device|, in ordinal order. Empty when there are
// not that many.
inline std::vector<int> PickDestinations(int src_device, size_t count) {
  int num_devices = 0;
  HIP_CHECK(hipGetDeviceCount(&num_devices));
  std::vector<int> destinations;
  for (int device = 0; device < num_devices && destinations.size() < count; ++device) {
    if (device != src_device) destinations.push_back(device);
  }
  if (destinations.size() < count) {
    std::cerr << "error: " << count << " destination(s) requested but only " << destinations.size()
              << " other GPU(s) exist\n";
    destinations.clear();
  }
  return destinations;
}

// Wall-clock ticks per microsecond on |device|; the reference tools assume 100.
inline double WallClockTicksPerUs(int device) {
  int khz = 0;
  HIP_CHECK(hipDeviceGetAttribute(&khz, hipDeviceAttributeWallClockRate, device));
  return khz > 0 ? khz / 1000.0 : 100.0;
}

inline bool ValidateSweep(size_t min_size, size_t max_size) {
  if (min_size == 0 || max_size < min_size) {
    std::cerr << "error: need 0 < --minCopySize <= --maxCopySize\n";
    return false;
  }
  return true;
}

// Every word of |bytes| bytes of |device_buffer| must still hold kMagicValue.
inline bool VerifyBuffer(const void* device_buffer, size_t bytes, int device) {
  std::vector<uint32_t> host(bytes / sizeof(uint32_t));
  HIP_CHECK(hipSetDevice(device));
  HIP_CHECK(
      hipMemcpy(host.data(), device_buffer, host.size() * sizeof(uint32_t), hipMemcpyDeviceToHost));
  size_t reported = 0;
  size_t errors = 0;
  for (size_t word = 0; word < host.size(); ++word) {
    if (host[word] == kMagicValue) continue;
    ++errors;
    if (reported++ < 10) {
      std::cerr << "mismatch on GPU " << device << " word " << word << ": got 0x" << std::hex
                << host[word] << std::dec << "\n";
    }
  }
  return errors == 0;
}

inline void FillMagic(void* device_buffer, size_t bytes) {
  std::vector<uint32_t> host(bytes / sizeof(uint32_t), kMagicValue);
  HIP_CHECK(
      hipMemcpy(device_buffer, host.data(), host.size() * sizeof(uint32_t), hipMemcpyHostToDevice));
}

// Aborts when any copy a kernel submitted or waited on failed.
inline void CheckStatuses(const hipError_t* statuses, size_t count) {
  for (size_t i = 0; i < count; ++i) {
    if (statuses[i] == hipSuccess) continue;
    std::fprintf(stderr, "kernel-side copy failed on warp %zu: %s (%d)\n", i,
                 hipGetErrorName(statuses[i]), static_cast<int>(statuses[i]));
    std::exit(EXIT_FAILURE);
  }
}

}  // namespace copy_engine_bench

//===----------------------------------------------------------------------===//
// Device side
//===----------------------------------------------------------------------===//

#if defined(__HIPCC__)

namespace copy_engine_bench {

// Waits until |*word| reaches |value|, with relaxed loads. Returns false after |max_spins|.
//
// No acquire follows: the benchmark's kernels read nothing a copy wrote. A kernel that consumes the
// copy must acquire at system scope before reading it.
__device__ inline bool WaitForSignal(const uint64_t* word, uint64_t value, uint64_t max_spins) {
  for (uint64_t spin = 0; spin <= max_spins; ++spin) {
    if (__hip_atomic_load(word, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM) >= value) return true;
    __builtin_amdgcn_s_sleep(1);
  }
  return false;
}

// Submits |num_commands| back-to-back copies of |copy_size| bytes on |handle|, from one lane of
// the wave, then waits for them. Only the last carries a completion: it sets |*signal| to |epoch|,
// and a signal lands after every earlier operation on the engine, so reaching |epoch| means every
// copy is done. |epoch| must exceed every value |*signal| has held, which spares the host from
// resetting it. When |submit_end_clock| is not null it receives the time the last submit returned.
// Nothing is released: the benchmark's buffers were written before the kernel launched. Returns the
// first failure, or hipSuccess.
__device__ inline hipError_t SubmitBatchAndWait(const hipExtCopyEngineDeviceHandle* handle,
                                                char* dst, const char* src, uint64_t copy_size,
                                                uint32_t num_commands, uint64_t* signal,
                                                uint64_t epoch, int64_t* submit_end_clock) {
  const hipExtCopyEngineSignal completion{reinterpret_cast<uint64_t>(signal), epoch,
                                          hipExtCopyEngineSignalSet, 0};
  hipError_t result = hipSuccess;
  for (uint32_t command = 0; command < num_commands && result == hipSuccess; ++command) {
    // Two calls with constant arguments, not one with a selected pointer: a pointer that may
    // be null keeps |completion| in memory and costs the submit about 1.7 us.
    if (command + 1 == num_commands) {
      result = hipExtCopyEngineSubmitLinearOneLane(
          handle, dst + command * copy_size, src + command * copy_size, copy_size, &completion);
    } else {
      result = hipExtCopyEngineSubmitLinearOneLane(handle, dst + command * copy_size,
                                                   src + command * copy_size, copy_size, nullptr);
    }
  }
  if (submit_end_clock != nullptr) *submit_end_clock = wall_clock64();
  if (result != hipSuccess) return result;
  return WaitForSignal(signal, epoch, kMaxSpins) ? hipSuccess : hipErrorNotReady;
}

}  // namespace copy_engine_bench

#endif  // __HIPCC__
