// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file native_workloads.hip
/// @brief Deterministic native HIP controls for the rocjitsu benchmark suite.

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

constexpr std::size_t kCopyBytes = 32ULL * 1024ULL * 1024ULL;
constexpr std::size_t kCopyElements = kCopyBytes / sizeof(float);
constexpr std::size_t kVectorElements = 1'048'583;
constexpr unsigned int kBlockSize = 256;
constexpr unsigned int kNoopBlocks = 1;
constexpr unsigned int kNoopThreads = 64;
constexpr unsigned int kCopyGridSize = 256;
constexpr std::size_t kVectorBytes = 16;
constexpr std::string_view kBuildTarget = ROCJITSU_BENCHMARK_BUILD_TARGET;

using Clock = std::chrono::steady_clock;

struct Options {
  std::string case_id;
  std::string target;
  std::string output = "-";
  unsigned int warmups = 3;
  unsigned int samples = 21;
};

struct Result {
  Options options;
  std::vector<std::uint64_t> timings_ns;
};

[[noreturn]] void fail(std::string message) { throw std::runtime_error(std::move(message)); }

void check_hip(hipError_t status, std::string_view expression) {
  if (status == hipSuccess)
    return;
  std::ostringstream message;
  message << expression << " failed: " << hipGetErrorString(status) << " ("
          << static_cast<int>(status) << ')';
  fail(message.str());
}

#define HIP_CHECK(expression) check_hip((expression), #expression)

template <typename T> class DeviceBuffer {
public:
  explicit DeviceBuffer(std::size_t elements) : elements_(elements) {
    HIP_CHECK(hipMalloc(reinterpret_cast<void **>(&pointer_), elements * sizeof(T)));
  }

  ~DeviceBuffer() {
    if (pointer_ != nullptr)
      (void)hipFree(pointer_);
  }

  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;

  [[nodiscard]] T *get() const { return pointer_; }
  [[nodiscard]] std::size_t bytes() const { return elements_ * sizeof(T); }

private:
  T *pointer_ = nullptr;
  std::size_t elements_ = 0;
};

extern "C" __global__ void launch_noop_kernel() {}

extern "C" __global__ void copy_u128_grid_stride_kernel(const uint4 *__restrict__ source,
                                                        uint4 *__restrict__ destination,
                                                        std::uint64_t count) {
  std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::uint64_t stride = static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
  for (; index < count; index += stride)
    destination[index] = source[index];
}

extern "C" __global__ void vector_add_tail_kernel(const float *__restrict__ left,
                                                  const float *__restrict__ right,
                                                  float *__restrict__ output, std::uint64_t count) {
  const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count)
    output[index] = left[index] + right[index];
}

std::string usage(std::string_view executable) {
  std::ostringstream output;
  output << "Usage: " << executable << " --case CASE --target GFX [options]\n\n"
         << "Cases:\n"
         << "  hip.launch_noop\n"
         << "  hip.copy_fp32_32m\n"
         << "  hip.vector_add_fp32_tail\n\n"
         << "Options:\n"
         << "  --warmups N          Untimed synchronized launches (default: 3)\n"
         << "  --samples N          Timed synchronized launches (default: 21)\n"
         << "  --output PATH        JSON output path; '-' writes stdout\n"
         << "  --target GFX         Required runtime architecture\n"
         << "  --help               Show this help\n";
  return output.str();
}

template <typename T> T parse_unsigned(std::string_view value, std::string_view option) {
  T parsed = 0;
  const char *begin = value.data();
  const char *end = value.data() + value.size();
  const auto [position, error] = std::from_chars(begin, end, parsed);
  if (value.empty() || error != std::errc{} || position != end)
    fail("invalid value for " + std::string(option) + ": " + std::string(value));
  return parsed;
}

std::string_view require_value(int &index, int argc, char **argv, std::string_view option) {
  if (++index >= argc)
    fail("missing value for " + std::string(option));
  return argv[index];
}

Options parse_options(int argc, char **argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument = argv[index];
    if (argument == "--help") {
      std::cout << usage(argv[0]);
      std::exit(0);
    }
    if (argument == "--case") {
      options.case_id = require_value(index, argc, argv, argument);
    } else if (argument == "--target") {
      options.target = require_value(index, argc, argv, argument);
    } else if (argument == "--warmups") {
      options.warmups =
          parse_unsigned<unsigned int>(require_value(index, argc, argv, argument), argument);
    } else if (argument == "--samples") {
      options.samples =
          parse_unsigned<unsigned int>(require_value(index, argc, argv, argument), argument);
    } else if (argument == "--output") {
      options.output = require_value(index, argc, argv, argument);
    } else {
      fail("unknown option: " + std::string(argument));
    }
  }

  constexpr std::array<std::string_view, 3> cases = {"hip.launch_noop", "hip.copy_fp32_32m",
                                                     "hip.vector_add_fp32_tail"};
  if (std::find(cases.begin(), cases.end(), options.case_id) == cases.end())
    fail("--case must name one of the documented native workloads");
  if (options.target.empty())
    fail("--target is required");
  if (options.samples == 0)
    fail("--samples must be positive");
  if (options.output.empty())
    fail("--output must not be empty");
  return options;
}

bool target_matches(std::string_view reported, std::string_view expected) {
  if (!reported.starts_with(expected))
    return false;
  return reported.size() == expected.size() || reported[expected.size()] == ':';
}

template <typename Launch>
std::vector<std::uint64_t> measure(const Options &options, Launch launch) {
  for (unsigned int iteration = 0; iteration < options.warmups; ++iteration) {
    launch();
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
  }

  std::vector<std::uint64_t> durations;
  durations.reserve(options.samples);
  for (unsigned int sample = 0; sample < options.samples; ++sample) {
    const auto begin = Clock::now();
    launch();
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    const auto end = Clock::now();
    const auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
    if (duration <= 0)
      fail("measured a non-positive dispatch duration");
    durations.push_back(static_cast<std::uint64_t>(duration));
  }
  return durations;
}

void run_noop(Result &result) {
  result.timings_ns = measure(
      result.options, []() { launch_noop_kernel<<<dim3(kNoopBlocks), dim3(kNoopThreads)>>>(); });
}

void run_copy(Result &result) {
  static_assert(kCopyBytes % sizeof(uint4) == 0);
  static_assert(sizeof(uint4) == kVectorBytes);
  DeviceBuffer<float> device_source(kCopyElements);
  DeviceBuffer<float> device_output(kCopyElements);
  HIP_CHECK(hipMemset(device_source.get(), 0, device_source.bytes()));
  HIP_CHECK(hipDeviceSynchronize());

  const auto launch = [&]() {
    copy_u128_grid_stride_kernel<<<dim3(kCopyGridSize), dim3(kBlockSize)>>>(
        reinterpret_cast<const uint4 *>(device_source.get()),
        reinterpret_cast<uint4 *>(device_output.get()), kCopyBytes / sizeof(uint4));
  };
  result.timings_ns = measure(result.options, launch);
}

void run_vector_add(Result &result) {
  DeviceBuffer<float> device_left(kVectorElements);
  DeviceBuffer<float> device_right(kVectorElements);
  DeviceBuffer<float> device_output(kVectorElements);
  HIP_CHECK(hipMemset(device_left.get(), 0, device_left.bytes()));
  HIP_CHECK(hipMemset(device_right.get(), 0, device_right.bytes()));
  HIP_CHECK(hipDeviceSynchronize());

  constexpr std::size_t grid_size = (kVectorElements + kBlockSize - 1) / kBlockSize;
  const auto launch = [&]() {
    vector_add_tail_kernel<<<dim3(grid_size), dim3(kBlockSize)>>>(
        device_left.get(), device_right.get(), device_output.get(), kVectorElements);
  };
  result.timings_ns = measure(result.options, launch);
}

std::string json_escape(std::string_view value) {
  std::ostringstream output;
  for (const unsigned char character : value) {
    switch (character) {
    case '"':
      output << "\\\"";
      break;
    case '\\':
      output << "\\\\";
      break;
    case '\n':
      output << "\\n";
      break;
    case '\r':
      output << "\\r";
      break;
    case '\t':
      output << "\\t";
      break;
    default:
      if (character < 0x20)
        output << "\\u" << std::hex << std::setw(4) << std::setfill('0')
               << static_cast<unsigned int>(character) << std::dec;
      else
        output << character;
    }
  }
  return output.str();
}

void write_parameters(std::ostream &output, std::string_view case_id) {
  if (case_id == "hip.launch_noop") {
    output << "{\"blocks\": " << kNoopBlocks << ", \"threads_per_block\": " << kNoopThreads << '}';
  } else if (case_id == "hip.copy_fp32_32m") {
    output << "{\"block_size\": " << kBlockSize << ", \"bytes\": " << kCopyBytes
           << ", \"dtype\": \"fp32\", \"elements\": " << kCopyElements
           << ", \"grid_size\": " << kCopyGridSize << ", \"vector_bytes\": " << kVectorBytes << '}';
  } else {
    output << "{\"block_size\": " << kBlockSize
           << ", \"dtype\": \"fp32\", \"elements\": " << kVectorElements << "}";
  }
}

void write_json(std::ostream &output, const Result &result) {
  const auto &samples = result.timings_ns;
  output << "{\n"
         << "  \"schema\": \"rocjitsu.benchmark.workload.v1\",\n"
         << "  \"case\": \"" << json_escape(result.options.case_id) << "\",\n"
         << "  \"target\": \"" << json_escape(result.options.target) << "\",\n"
         << "  \"provider\": \"hip\",\n"
         << "  \"parameters\": ";
  write_parameters(output, result.options.case_id);
  output << ",\n"
         << "  \"timings_ns\": [";
  for (std::size_t index = 0; index < samples.size(); ++index) {
    if (index != 0)
      output << ", ";
    output << samples[index];
  }
  output << "]\n}\n";
}

Result run(const Options &options) {
  if (!target_matches(kBuildTarget, options.target))
    fail("binary target '" + std::string(kBuildTarget) + "' does not match expected target '" +
         options.target + "'");

  HIP_CHECK(hipSetDevice(0));
  hipDeviceProp_t properties{};
  HIP_CHECK(hipGetDeviceProperties(&properties, 0));

  Result result;
  result.options = options;
  if (!target_matches(properties.gcnArchName, options.target))
    fail("runtime reported target '" + std::string(properties.gcnArchName) + "', expected '" +
         options.target + "'");

  if (options.case_id == "hip.launch_noop")
    run_noop(result);
  else if (options.case_id == "hip.copy_fp32_32m")
    run_copy(result);
  else
    run_vector_add(result);

  (void)hipDeviceReset();
  return result;
}

} // namespace

int main(int argc, char **argv) {
  try {
    const Options options = parse_options(argc, argv);
    const Result result = run(options);
    if (options.output == "-") {
      write_json(std::cout, result);
    } else {
      std::ofstream output(options.output, std::ios::trunc);
      if (!output)
        fail("could not open output file: " + options.output);
      write_json(output, result);
    }
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "rocjitsu native benchmark: " << error.what() << '\n';
    return 2;
  }
}
