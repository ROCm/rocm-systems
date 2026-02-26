// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef ROCP_COUNTER_SAMPLER_H_
#define ROCP_COUNTER_SAMPLER_H_

#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <cstdint>
#include <map>
#include <memory>
#include <unordered_map>
#include <vector>

#define ROCP_ERROR 0
#define ROCP_INFO 1
#define ROCP_DEBUG 2

#define MAX_COUNTER_PER_PROFILE 10

#define ROCP_LOG(debug_level, msg)                                                             \
  do {                                                                                        \
      std::clog << __FILE__ << ":" <<  __LINE__ << " " << msg                                 \
                           << std::endl;                                                      \
  } while (0)

typedef struct metric {
  double value;
} metric;

typedef struct gpu_metric {
  struct metric *list;
} gpu_metric;

namespace amd {
namespace rocp {
class CounterSampler {
 public:
  explicit CounterSampler(rocprofiler_agent_id_t agent);
  ~CounterSampler();

  const std::string& decode_record_name(const rocprofiler_record_counter_t& rec) const;

  std::unordered_map<std::string, size_t> get_record_dimensions(
      const rocprofiler_record_counter_t& rec);

  void sample_counter_values(const std::vector<std::string>& counters,
                             std::vector<rocprofiler_record_counter_t>& out, uint64_t duration);

  rocprofiler_agent_id_t get_agent() const { return agent_; }

  struct ProfileSet {
    struct Profile {
      rocprofiler_counter_config_id_t config;
      std::vector<std::string> counter_names;
      size_t expected_size;
    };
    std::vector<Profile> profiles;
  };

  void sample_counters_with_packing(const std::vector<std::string>& counters,
                                    std::map<std::string, double>& out_values,
                                    uint64_t duration,
                                    uint32_t delay_ms = 0);

  static std::unordered_map<std::string, rocprofiler_counter_id_t> get_supported_counters(
      rocprofiler_agent_id_t agent);

  static std::vector<rocprofiler_agent_v0_t> get_available_agents();

  static std::vector<std::shared_ptr<CounterSampler>>& get_samplers();
  static int runSample(std::vector<std::string> &metric_fields, uint64_t duration, uint32_t delay_ms = 0);

 private:
  rocprofiler_agent_id_t agent_ = {};
  rocprofiler_context_id_t ctx_ = {};
  rocprofiler_profile_config_id_t profile_ = {.handle = 0};
  rocprofiler_counter_config_id_t counter_ = {.handle = 0};

  std::map<std::vector<std::string>, rocprofiler_profile_config_id_t> cached_profiles_;
  std::map<uint64_t, uint64_t> profile_sizes_;
  std::map<std::vector<std::string>, rocprofiler_counter_config_id_t> cached_counter_;
  std::map<uint64_t, uint64_t> counter_sizes_;
  std::map<std::vector<std::string>, ProfileSet> cached_profile_sets_;
  mutable std::map<uint64_t, std::string> id_to_name_;

  void set_profile(rocprofiler_context_id_t ctx, rocprofiler_agent_set_profile_callback_t cb) const;

  size_t get_counter_size(rocprofiler_counter_id_t counter);

  std::vector<rocprofiler_record_dimension_info_t> get_counter_dimensions(
      rocprofiler_counter_id_t counter);

  ProfileSet create_profiles_for_counters(const std::vector<std::string>& counters);

  static std::vector<std::shared_ptr<CounterSampler>> samplers_;
};

}  // namespace rocp
}  // namespace amd

#endif  // ROCP_COUNTER_SAMPLER_H_
