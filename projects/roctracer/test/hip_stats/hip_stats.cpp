/* Copyright (c) 2022 Advanced Micro Devices, Inc.

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
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

#include "roctracer.h"
#include "roctracer_hip.h"

#include <cstdint>
#include <cstdlib>
#include <experimental/filesystem>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <numeric>
#include <set>
#include <string>
#include <sstream>
#include <unordered_map>
#include <utility>

#define CHECK_ROCTRACER(call)                                                                      \
  do {                                                                                             \
    roctracer_status_t status = call;                                                              \
    if (status != ROCTRACER_STATUS_SUCCESS) {                                                      \
      std::cerr << roctracer_error_string() << std::endl;                                          \
      abort();                                                                                     \
    }                                                                                              \
  } while (false)

namespace {

constexpr uint64_t NextPowerOf2(uint64_t v) {
  v += (v == 0);
  v--;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  v |= v >> 32;
  return ++v;
}

constexpr size_t KiB = 1024;
constexpr size_t MiB = KiB * KiB;
constexpr size_t GiB = KiB * KiB * KiB;

std::string HumanReadableSize(size_t size, int precision) {
  std::stringstream ss;
  if (size < KiB)
    ss << size;
  else if (size < MiB)
    ss << std::fixed << std::setprecision(precision) << (double)size / KiB << "K";
  else if (size < GiB)
    ss << std::fixed << std::setprecision(precision) << (double)size / MiB << "M";
  else
    ss << std::fixed << std::setprecision(precision) << (double)size / GiB << "G";
  return ss.str();
}

struct FunctionStats {
  uint64_t total_time_ns;
  uint64_t count;
  void Accumulate(uint64_t time_ns) {
    total_time_ns += time_ns;
    ++count;
  }
};

struct MemCopyStats {
  uint64_t total_time_ns;
  uint64_t total_byte_size;
  uint64_t count;
  void Accumulate(uint64_t time_ns, uint64_t byte_size) {
    total_time_ns += time_ns;
    total_byte_size += byte_size;
    ++count;
  }
};

struct pair_hash {
  template <typename T1, typename T2> std::size_t operator()(const std::pair<T1, T2>& pair) const {
    return std::hash<T1>()(pair.first) ^ std::hash<T2>()(pair.second);
  }
};

std::unordered_map<decltype(roctracer_record_t::op), FunctionStats> hip_api_stats;
std::unordered_map<std::string, FunctionStats> kernel_stats;
std::unordered_map<std::pair<decltype(roctracer_record_t::kind), size_t>, MemCopyStats, pair_hash>
    memcpy_stats;

void CollectStatistics(const char* begin, const char* end, void* /* user_arg */) {
  const auto* record = reinterpret_cast<const roctracer_record_t*>(begin);
  while (record < reinterpret_cast<const roctracer_record_t*>(end)) {
    auto elapsed_time_ns = record->end_ns - record->begin_ns;

    if (record->domain == ACTIVITY_DOMAIN_HIP_OPS && record->op == HIP_OP_ID_DISPATCH) {
      const char* kernel_name = record->kernel_name;
      if (kernel_name == nullptr) kernel_name = "Unknown kernels";
      kernel_stats[kernel_name].Accumulate(elapsed_time_ns);
    } else if (record->domain == ACTIVITY_DOMAIN_HIP_OPS && record->op == HIP_OP_ID_COPY)
      memcpy_stats[std::make_pair(record->kind, NextPowerOf2(record->bytes))].Accumulate(
          elapsed_time_ns, record->bytes);
    else if (record->domain == ACTIVITY_DOMAIN_HIP_API)
      hip_api_stats[record->op].Accumulate(elapsed_time_ns);

    CHECK_ROCTRACER(roctracer_next_record(record, &record));
  }
}

namespace fs = std::experimental::filesystem;

void DumpStatistics() {
  CHECK_ROCTRACER(roctracer_close_pool());

  fs::path output_dir = []() {
    const char* env_var = getenv("ROCP_OUTPUT_DIR");
    return env_var != nullptr ? env_var : "";
  }();

  std::ofstream out;

  if (output_dir.empty()) {
    // If an output directory was not specified, then print the statistics to stdout.
    out.copyfmt(std::cout);
    out.clear(std::cout.rdstate());
    out.basic_ios<char>::rdbuf(std::cout.rdbuf());
  } else {
    if (auto status = fs::status(output_dir); !fs::exists(status) || !fs::is_directory(status)) {
      std::cerr << "error: ROCP_OUTPUT_DIR=" << output_dir << " is not a directory" << std::endl;
      return;
    }
  }

  auto compare = [](const auto& x, const auto& y) {
    return x.second.total_time_ns > y.second.total_time_ns;
  };

  // Print the HIP API statistics sorted by descending total inclusive time.
  if (!hip_api_stats.empty()) {
    const char* filename = "hip_api_stats.csv";
    if (!output_dir.empty()) out = std::ofstream(output_dir / filename);

    if (out.good()) {
      std::cout << "Dumping HIP API statistics." << std::endl;

      uint64_t total_hip_api_time_ns =
          std::accumulate(hip_api_stats.begin(), hip_api_stats.end(), 0,
                          [](uint64_t total_time_ns, const auto& stats) {
                            return total_time_ns + stats.second.total_time_ns;
                          });

      out << "\"Name\",\"Calls\",\"TotalDurationNs\",\"AverageNs\",\"Percentage\"" << std::endl;
      for (auto&& [op, stats] : std::set<decltype(hip_api_stats)::value_type, decltype(compare)>(
               hip_api_stats.begin(), hip_api_stats.end(), compare))
        out << "\"" << roctracer_op_string(ACTIVITY_DOMAIN_HIP_API, op, 0) << "\"," << stats.count
            << "," << stats.total_time_ns << "," << stats.total_time_ns / stats.count << ","
            << std::fixed << std::setprecision(4)
            << (double)stats.total_time_ns / total_hip_api_time_ns * 100 << std::endl;
    } else {
      std::cerr << "warning: could not open " << output_dir / filename << std::endl;
    }
  }

  // Print the HIP kernel dispatch statistics sorted by descending execution time.

  if (!kernel_stats.empty()) {
    const char* filename = "hip_kernel_stats.csv";
    if (!output_dir.empty()) out = std::ofstream(output_dir / filename);

    if (out.good()) {
      std::cout << "Dumping HIP kernel dispatch statistics." << std::endl;

      uint64_t total_kernel_time_ns =
          std::accumulate(kernel_stats.begin(), kernel_stats.end(), 0,
                          [](uint64_t total_time_ns, const auto& stats) {
                            return total_time_ns + stats.second.total_time_ns;
                          });

      out << "\"Name\",\"Calls\",\"TotalDurationNs\",\"AverageNs\",\"Percentage\"" << std::endl;
      for (auto&& [name, stats] : std::set<decltype(kernel_stats)::value_type, decltype(compare)>(
               kernel_stats.begin(), kernel_stats.end(), compare))
        out << "\"" << name << "\"," << stats.count << "," << stats.total_time_ns << ","
            << stats.total_time_ns / stats.count << "," << std::fixed << std::setprecision(4)
            << (double)stats.total_time_ns / total_kernel_time_ns * 100 << std::endl;
    } else {
      std::cerr << "warning: could not open " << output_dir / filename << std::endl;
    }
  }

  // Print the HIP memory copy statistics sorted by descending transfer time.

  if (!memcpy_stats.empty()) {
    const char* filename = "hip_copy_stats.csv";
    if (!output_dir.empty()) out = std::ofstream(output_dir / filename);

    if (out.good()) {
      std::cout << "Dumping HIP memory copy statistics." << std::endl;

      uint64_t total_memory_copy_time_ns =
          std::accumulate(memcpy_stats.begin(), memcpy_stats.end(), 0,
                          [](uint64_t total_time_ns, const auto& stats) {
                            return total_time_ns + stats.second.total_time_ns;
                          });

      out << "\"Name\",\"Calls\",\"TotalBytes\",\"TotalDurationNs\",\"AverageNs\",\"Percentage\""
          << std::endl;
      for (auto&& [kind, stats] : std::set<decltype(memcpy_stats)::value_type, decltype(compare)>(
               memcpy_stats.begin(), memcpy_stats.end(), compare))
        out << "\"" << roctracer_op_string(ACTIVITY_DOMAIN_HIP_OPS, HIP_OP_ID_COPY, kind.first)
            << "(" << HumanReadableSize(kind.second >> 1, 0) << "-"
            << HumanReadableSize(kind.second, 0) << ")"
            << "\"," << stats.count << "," << stats.total_byte_size << "," << stats.total_time_ns
            << "," << stats.total_time_ns / stats.count << "," << std::fixed << std::setprecision(4)
            << (double)stats.total_time_ns / total_memory_copy_time_ns * 100 << std::endl;
    } else {
      std::cerr << "warning: could not open " << output_dir / filename << std::endl;
    }
  }
}

}  // namespace

#include <hsa/hsa_api_trace.h>

extern "C" ROCTRACER_EXPORT bool OnLoad(HsaApiTable* /* table */, uint64_t /* runtime_version */,
                                        uint64_t /* failed_tool_count */,
                                        const char* const* /* failed_tool_names */) {
  roctracer_properties_t properties{};
  properties.buffer_size = sizeof(roctracer_record_t) * 10000;
  properties.buffer_callback_fun = CollectStatistics;
  properties.buffer_callback_arg = nullptr;

  CHECK_ROCTRACER(roctracer_open_pool(&properties));
  CHECK_ROCTRACER(roctracer_enable_domain_activity(ACTIVITY_DOMAIN_HIP_API));
  CHECK_ROCTRACER(roctracer_enable_op_activity(ACTIVITY_DOMAIN_HIP_OPS, HIP_OP_ID_DISPATCH));
  CHECK_ROCTRACER(roctracer_enable_op_activity(ACTIVITY_DOMAIN_HIP_OPS, HIP_OP_ID_COPY));

  std::atexit([]() { DumpStatistics(); });
  return true;
}

extern "C" ROCTRACER_EXPORT void OnUnload() {}
