// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file vgpr_saturation.hip
/// @brief Standalone rocjitsu benchmark for demand-paged VGPR scaling.
///
/// Each template instance keeps Count independent uint32_t values live across
/// a runtime-sized update loop. The final checksum makes every value
/// observable. On gfx942 with the ROCm SDK compiler used by rocjitsu, the
/// instances below use Count + 1 VGPRs and zero VGPR spills. The program also
/// queries and reports the compiler's actual register count at runtime.

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iterator>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

constexpr uint32_t WAVE_SIZE = 64;
constexpr int DEFAULT_SAMPLES = 7;
constexpr int DEFAULT_WARMUPS = 2;

constexpr int SUPPORTED_COUNTS[] = {
#define VGPR_SATURATION_COUNT(N) N,
#include "vgpr_saturation_counts.def"
#undef VGPR_SATURATION_COUNT
};

template <int Count>
__global__ __launch_bounds__(WAVE_SIZE, 1) void vgpr_saturation_kernel(uint32_t *output,
                                                                       uint32_t iterations) {
  const uint32_t global_id = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t values[Count];

#pragma unroll
  for (int i = 0; i < Count; ++i) {
    values[i] = global_id + 0x9e3779b9u * static_cast<uint32_t>(i + 1);
  }

  for (uint32_t iter = 0; iter < iterations; ++iter) {
#pragma unroll
    for (int i = 0; i < Count; ++i) {
      values[i] = (values[i] ^ values[(i + 1) % Count]) * 1664525u + 1013904223u;
    }
  }

  uint32_t checksum = 0;
#pragma unroll
  for (int i = 0; i < Count; ++i) {
    checksum ^= values[i] + static_cast<uint32_t>(i);
  }
  output[global_id] = checksum;
}

struct Options {
  std::vector<int> counts{std::begin(SUPPORTED_COUNTS), std::end(SUPPORTED_COUNTS)};
  std::vector<uint32_t> iterations = {0, 1, 4, 16, 64};
  uint32_t waves = 1;
  int samples = DEFAULT_SAMPLES;
  int warmups = DEFAULT_WARMUPS;
  bool csv = false;
};

struct Result {
  int requested_vgprs = 0;
  int reported_vgprs = 0;
  uint32_t iterations = 0;
  uint32_t waves = 0;
  int samples = 0;
  double median_us = 0.0;
  double min_us = 0.0;
  double max_us = 0.0;
  uint32_t checksum = 0;
};

[[noreturn]] void fail(const std::string &message) {
  std::fprintf(stderr, "vgpr_saturation: %s\n", message.c_str());
  std::exit(1);
}

void check_hip(hipError_t error, std::string_view operation) {
  if (error != hipSuccess) {
    fail(std::string(operation) + ": " + hipGetErrorString(error));
  }
}

template <typename Value>
std::vector<Value> parse_list(std::string_view value, const char *option) {
  std::vector<Value> parsed;
  size_t begin = 0;
  while (begin < value.size()) {
    const size_t end = value.find(',', begin);
    const std::string token(
        value.substr(begin, end == std::string_view::npos ? value.size() - begin : end - begin));
    char *parse_end = nullptr;
    const unsigned long raw = std::strtoul(token.c_str(), &parse_end, 10);
    if (token.empty() || *parse_end != '\0' || raw > std::numeric_limits<Value>::max()) {
      fail(std::string("invalid value for ") + option + ": " + token);
    }
    parsed.push_back(static_cast<Value>(raw));
    if (end == std::string_view::npos)
      break;
    begin = end + 1;
  }
  if (parsed.empty())
    fail(std::string(option) + " requires at least one value");
  return parsed;
}

unsigned long parse_scalar(std::string_view value, const char *option) {
  char *parse_end = nullptr;
  const std::string token(value);
  const unsigned long raw = std::strtoul(token.c_str(), &parse_end, 10);
  if (token.empty() || *parse_end != '\0')
    fail(std::string("invalid value for ") + option + ": " + token);
  return raw;
}

void usage(const char *program) {
  std::string supported_counts;
  for (int count : SUPPORTED_COUNTS) {
    if (!supported_counts.empty())
      supported_counts += ',';
    supported_counts += std::to_string(count);
  }
  std::printf("Usage: %s [options]\n"
              "  --counts N,N,...      Live-value counts (supported: %s)\n"
              "  --iterations N,N,...  Update-loop iterations (default: 0,1,4,16,64)\n"
              "  --waves N              Wave64 workgroups per launch (default: 1)\n"
              "  --samples N            Timed launches per point (default: 7)\n"
              "  --warmups N            Untimed runtime warmups per point (default: 2)\n"
              "  --csv                  Emit machine-readable CSV\n"
              "  --help                 Show this help\n",
              program, supported_counts.c_str());
}

Options parse_options(int argc, char **argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string_view argument(argv[i]);
    auto next_value = [&](const char *option) -> std::string_view {
      if (++i >= argc)
        fail(std::string(option) + " requires a value");
      return argv[i];
    };

    if (argument == "--counts") {
      options.counts = parse_list<int>(next_value("--counts"), "--counts");
    } else if (argument == "--iterations") {
      options.iterations = parse_list<uint32_t>(next_value("--iterations"), "--iterations");
    } else if (argument == "--waves") {
      const unsigned long value = parse_scalar(next_value("--waves"), "--waves");
      if (value == 0 || value > std::numeric_limits<uint32_t>::max() / WAVE_SIZE)
        fail("--waves is too large for the 32-bit global thread index");
      options.waves = static_cast<uint32_t>(value);
    } else if (argument == "--samples") {
      const unsigned long value = parse_scalar(next_value("--samples"), "--samples");
      if (value == 0 || value > std::numeric_limits<int>::max())
        fail("--samples must be in [1, INT_MAX]");
      options.samples = static_cast<int>(value);
    } else if (argument == "--warmups") {
      const unsigned long value = parse_scalar(next_value("--warmups"), "--warmups");
      if (value > std::numeric_limits<int>::max())
        fail("--warmups must be in [0, INT_MAX]");
      options.warmups = static_cast<int>(value);
    } else if (argument == "--csv") {
      options.csv = true;
    } else if (argument == "--help") {
      usage(argv[0]);
      std::exit(0);
    } else {
      fail("unknown option: " + std::string(argument));
    }
  }
  return options;
}

template <int Count> uint32_t reference_checksum(uint32_t global_id, uint32_t iterations) {
  uint32_t values[Count];
  for (int i = 0; i < Count; ++i)
    values[i] = global_id + 0x9e3779b9u * static_cast<uint32_t>(i + 1);
  for (uint32_t iter = 0; iter < iterations; ++iter)
    for (int i = 0; i < Count; ++i)
      values[i] = (values[i] ^ values[(i + 1) % Count]) * 1664525u + 1013904223u;
  uint32_t checksum = 0;
  for (int i = 0; i < Count; ++i)
    checksum ^= values[i] + static_cast<uint32_t>(i);
  return checksum;
}

template <int Count>
Result run_point(uint32_t *device_output, uint32_t iterations, uint32_t waves, int warmups,
                 int samples) {
  hipFuncAttributes attributes{};
  check_hip(hipFuncGetAttributes(&attributes,
                                 reinterpret_cast<const void *>(vgpr_saturation_kernel<Count>)),
            "hipFuncGetAttributes");

  auto launch = [&] {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(vgpr_saturation_kernel<Count>), dim3(waves), dim3(WAVE_SIZE),
                       0, nullptr, device_output, iterations);
    check_hip(hipGetLastError(), "hipLaunchKernelGGL");
    check_hip(hipDeviceSynchronize(), "hipDeviceSynchronize");
  };

  for (int i = 0; i < warmups; ++i)
    launch();

  std::vector<double> elapsed_us;
  elapsed_us.reserve(samples);
  for (int i = 0; i < samples; ++i) {
    const auto begin = Clock::now();
    launch();
    const auto end = Clock::now();
    elapsed_us.push_back(std::chrono::duration<double, std::micro>(end - begin).count());
  }
  std::sort(elapsed_us.begin(), elapsed_us.end());

  const size_t output_elements = static_cast<size_t>(waves) * WAVE_SIZE;
  std::vector<uint32_t> checksums(output_elements);
  check_hip(hipMemcpy(checksums.data(), device_output, checksums.size() * sizeof(uint32_t),
                      hipMemcpyDeviceToHost),
            "hipMemcpy(checksums)");
  for (size_t global_id = 0; global_id < checksums.size(); ++global_id) {
    const uint32_t expected =
        reference_checksum<Count>(static_cast<uint32_t>(global_id), iterations);
    if (checksums[global_id] != expected) {
      fail("checksum mismatch for Count=" + std::to_string(Count) + ", iterations=" +
           std::to_string(iterations) + ", global_id=" + std::to_string(global_id));
    }
  }

  Result result;
  result.requested_vgprs = Count;
  result.reported_vgprs = attributes.numRegs;
  result.iterations = iterations;
  result.waves = waves;
  result.samples = samples;
  result.median_us = elapsed_us[elapsed_us.size() / 2];
  result.min_us = elapsed_us.front();
  result.max_us = elapsed_us.back();
  result.checksum = checksums.front();
  return result;
}

template <int Count>
void run_count(const Options &options, uint32_t *device_output, std::vector<Result> &results) {
  for (uint32_t iterations : options.iterations) {
    results.push_back(run_point<Count>(device_output, iterations, options.waves, options.warmups,
                                       options.samples));
  }
}

void dispatch_count(int count, const Options &options, uint32_t *device_output,
                    std::vector<Result> &results) {
#define VGPR_SATURATION_COUNT(N)                                                                   \
  case N:                                                                                          \
    run_count<N>(options, device_output, results);                                                 \
    return;
  switch (count) {
#include "vgpr_saturation_counts.def"
  default:
    fail("unsupported --counts value: " + std::to_string(count));
  }
#undef VGPR_SATURATION_COUNT
}

void print_results(const Options &options, const std::vector<Result> &results) {
  if (options.csv) {
    std::printf("requested_vgprs,reported_vgprs,iterations,waves,samples,median_us,min_us,max_us,"
                "checksum\n");
    for (const Result &result : results) {
      std::printf("%d,%d,%u,%u,%d,%.3f,%.3f,%.3f,0x%08x\n", result.requested_vgprs,
                  result.reported_vgprs, result.iterations, result.waves, result.samples,
                  result.median_us, result.min_us, result.max_us, result.checksum);
    }
    return;
  }

  std::printf("rocjitsu VGPR saturation benchmark (wave64, %u wave%s/launch)\n", options.waves,
              options.waves == 1 ? "" : "s");
  std::printf("requested  actual  iterations  median_us      min_us      max_us  checksum\n");
  for (const Result &result : results) {
    std::printf("%9d  %6d  %10u  %9.3f  %10.3f  %10.3f  %08x\n", result.requested_vgprs,
                result.reported_vgprs, result.iterations, result.median_us, result.min_us,
                result.max_us, result.checksum);
  }
}

} // namespace

int main(int argc, char **argv) {
  const Options options = parse_options(argc, argv);
  check_hip(hipSetDevice(0), "hipSetDevice");

  const size_t output_elements = static_cast<size_t>(options.waves) * WAVE_SIZE;
  if (output_elements > std::numeric_limits<size_t>::max() / sizeof(uint32_t))
    fail("output allocation size overflow");

  uint32_t *device_output = nullptr;
  check_hip(hipMalloc(&device_output, output_elements * sizeof(uint32_t)), "hipMalloc");

  std::vector<Result> results;
  results.reserve(options.counts.size() * options.iterations.size());
  for (int count : options.counts)
    dispatch_count(count, options, device_output, results);

  check_hip(hipFree(device_output), "hipFree");
  print_results(options, results);
  return 0;
}
