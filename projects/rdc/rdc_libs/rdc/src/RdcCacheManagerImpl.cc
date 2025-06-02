/*
Copyright (c) 2020 - present Advanced Micro Devices, Inc. All rights reserved.

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
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
#include "rdc_lib/impl/RdcCacheManagerImpl.h"

#include <sys/time.h>

#include <cmath>
#include <ctime>
#include <sstream>

#include "rdc_lib/RdcLogger.h"
#include "rdc_lib/rdc_common.h"

namespace amd {
namespace rdc {

rdc_status_t RdcCacheManagerImpl::rdc_field_get_value_since(uint32_t gpu_index,
                                                            rdc_field_t field_id,
                                                            uint64_t since_time_stamp,
                                                            uint64_t* next_since_time_stamp,
                                                            rdc_field_value* value) {
  if (!next_since_time_stamp || !value) {
    return RDC_ST_BAD_PARAMETER;
  }

  std::lock_guard<std::mutex> guard(cache_mutex_);
  RdcFieldKey field{gpu_index, field_id};
  auto cache_samples_ite = cache_samples_.find(field);
  if (cache_samples_ite == cache_samples_.end() || cache_samples_ite->second.size() == 0) {
    return RDC_ST_NOT_FOUND;
  }

  // TODO(bill_liu): Optimize it using the binary search
  auto cache_values = cache_samples_ite->second;
  for (auto cache_value = cache_values.begin(); cache_value != cache_values.end(); cache_value++) {
    if (cache_value->last_time >= since_time_stamp) {
      // move to next potential timestamp
      auto next_iter = std::next(cache_value);
      if (next_iter != cache_values.end()) {
        *next_since_time_stamp = next_iter->last_time;
      } else {  // Last item, set it to the future by adding 1us
        *next_since_time_stamp = cache_value->last_time + 1;
      }
      value->ts = cache_value->last_time;
      value->type = cache_value->type;

      if (value->type == STRING) {
        strncpy_with_null(value->value.str, cache_value->value.str, RDC_MAX_STR_LENGTH);
      } else {
        value->value.l_int = cache_value->value.l_int;
      }
      value->field_id = field_id;
      return RDC_ST_OK;
    }
  }

  *next_since_time_stamp = since_time_stamp;
  return RDC_ST_NOT_FOUND;
}

rdc_status_t RdcCacheManagerImpl::evict_cache(uint32_t gpu_index, rdc_field_t field_id,
                                              uint64_t max_keep_samples, double max_keep_age) {
  std::lock_guard<std::mutex> guard(cache_mutex_);

  RdcFieldKey field{gpu_index, field_id};
  auto cache_samples_ite = cache_samples_.find(field);
  if (cache_samples_ite == cache_samples_.end() || cache_samples_ite->second.size() == 0) {
    return RDC_ST_NOT_FOUND;
  }

  // Check max_keep_samples
  auto& cache_values = cache_samples_ite->second;
  int item_remove = cache_values.size() - max_keep_samples;
  if (item_remove > 0) {
    cache_values.erase(cache_values.begin(), cache_values.begin() + item_remove);
  }

  // Check max_keep_age
  struct timeval tv;
  gettimeofday(&tv, NULL);
  uint64_t now = static_cast<uint64_t>(tv.tv_sec) * 1000 + tv.tv_usec / 1000;

  auto ite = cache_values.begin();
  while (ite != cache_values.end()) {
    if (ite->last_time + max_keep_age * 1000 >= now) {
      break;
    } else {
      ite = cache_values.erase(ite);
    }
  }

  return RDC_ST_OK;
}

rdc_status_t RdcCacheManagerImpl::rdc_field_get_latest_value(uint32_t gpu_index,
                                                             rdc_field_t field_id,
                                                             rdc_field_value* value) {
  if (!value) {
    return RDC_ST_BAD_PARAMETER;
  }

  std::lock_guard<std::mutex> guard(cache_mutex_);
  RdcFieldKey field{gpu_index, field_id};
  auto cache_samples_ite = cache_samples_.find(field);
  if (cache_samples_ite == cache_samples_.end() || cache_samples_ite->second.size() == 0) {
    return RDC_ST_NOT_FOUND;
  }

  auto& cache_value = cache_samples_ite->second.back();
  value->ts = cache_value.last_time;
  value->type = cache_value.type;
  value->value = cache_value.value;
  value->field_id = field_id;

  return RDC_ST_OK;
}

std::string RdcCacheManagerImpl::get_cache_stats() {
  std::stringstream strstream;
  std::lock_guard<std::mutex> guard(cache_mutex_);

  strstream << "Cache samples:";
  auto cache_samples_ite = cache_samples_.begin();
  for (; cache_samples_ite != cache_samples_.end(); cache_samples_ite++) {
    strstream << "<" << cache_samples_ite->first.first << "," << cache_samples_ite->first.second
              << ":" << cache_samples_ite->second.size() << "> ";
  }

  strstream << " Job caches:";
  auto job_ite = cache_jobs_.begin();
  for (; job_ite != cache_jobs_.end(); job_ite++) {
    strstream << "<" << job_ite->first << ":" << job_ite->second.gpu_stats.size() << "> ";
  }

  return strstream.str();
}

rdc_status_t RdcCacheManagerImpl::rdc_update_cache(uint32_t gpu_index,
                                                   const rdc_field_value& value) {
  RdcCacheEntry entry;
  entry.last_time = value.ts;
  entry.value = value.value;
  entry.type = value.type;

  std::lock_guard<std::mutex> guard(cache_mutex_);
  RdcFieldKey field{gpu_index, value.field_id};
  auto cache_samples_ite = cache_samples_.find(field);
  if (cache_samples_ite == cache_samples_.end()) {
    std::vector<RdcCacheEntry> ve;
    ve.push_back(entry);
    cache_samples_.insert({field, ve});
  } else {
    cache_samples_ite->second.push_back(entry);
  }

  return RDC_ST_OK;
}

rdc_status_t RdcCacheManagerImpl::rdc_job_remove(const char job_id[64]) {
  std::lock_guard<std::mutex> guard(cache_mutex_);
  cache_jobs_.erase(job_id);
  return RDC_ST_OK;
}

rdc_status_t RdcCacheManagerImpl::rdc_job_remove_all() {
  std::lock_guard<std::mutex> guard(cache_mutex_);
  cache_jobs_.clear();
  return RDC_ST_OK;
}

rdc_status_t RdcCacheManagerImpl::rdc_update_job_stats(uint32_t gpu_index,
                                                       const std::string& job_id,
                                                       const rdc_field_value& value) {
  std::lock_guard<std::mutex> guard(cache_mutex_);
  auto job_iter = cache_jobs_.find(job_id);
  if (job_iter == cache_jobs_.end()) {
    return RDC_ST_NOT_FOUND;
  }

  auto& je = cache_jobs_.at(job_id);

  // handle process‐start/stop
  if (value.field_id == RDC_EVNT_NOTIF_PROCESS_START ||
      value.field_id == RDC_EVNT_NOTIF_PROCESS_END) {
    rdc_process_status_info_t info{};
    sscanf(value.value.str, "PID: %u  task: %255s", &info.pid, info.process_name);

    uint64_t ts_us = value.ts * 1000;
    if (value.field_id == RDC_EVNT_NOTIF_PROCESS_START) {
      info.start_time = ts_us;
    } else {
      info.stop_time = ts_us;
    }

    auto pit = je.pid_to_index.find(info.pid);
    if (pit == je.pid_to_index.end()) {
      uint32_t slot = je.num_processes;
      if (slot < RDC_MAX_NUM_PROCESSES_STATUS) {
        je.processes[slot] = info;
        je.pid_to_index[info.pid] = slot;
        je.num_processes = slot + 1;
      }
    } else {
      auto& entry = je.processes[pit->second];
      if (value.field_id == RDC_EVNT_NOTIF_PROCESS_START) {
        entry.start_time = info.start_time;
      } else {
        entry.stop_time = info.stop_time;
      }
    }
    return RDC_ST_OK;
  }

  auto gpu_iter = job_iter->second.gpu_stats.find(gpu_index);
  if (gpu_iter == job_iter->second.gpu_stats.end()) {
    return RDC_ST_NOT_FOUND;
  }

  auto fsummary = gpu_iter->second.field_summaries.find(value.field_id);
  if (fsummary == gpu_iter->second.field_summaries.end()) {
    return RDC_ST_NOT_FOUND;
  }
  if (fsummary->second.count == 0) {  // first item
    fsummary->second.count = 1;
    fsummary->second.max_value = value.value.l_int;
    fsummary->second.min_value = value.value.l_int;
    fsummary->second.total_value = value.value.l_int;
    fsummary->second.last_time = value.ts;
    if (value.field_id == RDC_FI_POWER_USAGE) {
      gpu_iter->second.energy_last_time = value.ts;
    }

    // https://www.johndcook.com/blog/standard_deviation/
    fsummary->second.old_s = 0;
    fsummary->second.old_m = fsummary->second.new_m = value.value.l_int;
    return RDC_ST_OK;
  }
  if (value.field_id == RDC_FI_POWER_USAGE) {
    uint64_t time_elapsed = value.ts - gpu_iter->second.energy_last_time;
    // Stored in cache as microseconds and microwats
    gpu_iter->second.energy_consumed += (time_elapsed * value.value.l_int) / (1000.0 * 1000000);
    gpu_iter->second.energy_last_time = value.ts;
  }
  fsummary->second.max_value =
      std::max(fsummary->second.max_value, static_cast<int64_t>(value.value.l_int));
  fsummary->second.min_value =
      std::min(fsummary->second.min_value, static_cast<int64_t>(value.value.l_int));
  fsummary->second.total_value += value.value.l_int;
  fsummary->second.last_time = value.ts;
  fsummary->second.count++;

  // https://www.johndcook.com/blog/standard_deviation/
  fsummary->second.new_m = fsummary->second.old_m +
                           (value.value.l_int - fsummary->second.old_m) / fsummary->second.count;
  fsummary->second.new_s =
      fsummary->second.old_s +
      (value.value.l_int - fsummary->second.old_m) * (value.value.l_int - fsummary->second.new_m);
  fsummary->second.old_m = fsummary->second.new_m;
  fsummary->second.old_s = fsummary->second.new_s;

  return RDC_ST_OK;
}

void RdcCacheManagerImpl::set_summary(const FieldSummaryStats& stats, rdc_stats_summary_t& gpu,
                                      rdc_stats_summary_t& summary, unsigned int adjuster) {
  if (stats.count == 0) {
    gpu.min_value = std::numeric_limits<uint64_t>::max();
    gpu.max_value = gpu.average = 0;
    return;
  }

  gpu.max_value = stats.max_value / adjuster;
  gpu.min_value = stats.min_value / adjuster;
  gpu.average = stats.total_value / stats.count / adjuster;
  summary.max_value = std::max(summary.max_value, gpu.max_value);
  summary.min_value = std::min(summary.min_value, gpu.min_value);
  //< save total for future average calculation.
  summary.average += gpu.average;

  //< calculate the sample variance
  gpu.standard_deviation =
      std::sqrt((stats.count > 1) ? stats.new_s / (stats.count - 1) : 0.0) / adjuster;
  summary.standard_deviation += gpu.standard_deviation;
}

rdc_status_t RdcCacheManagerImpl::rdc_job_get_stats(const char jobId[64],
                                                    const rdc_gpu_gauges_t& gpu_gauges,
                                                    rdc_job_info_t* p_job_info) {
  std::lock_guard<std::mutex> guard(cache_mutex_);
  auto job_stats = cache_jobs_.find(jobId);

  if (job_stats == cache_jobs_.end()) {
    return RDC_ST_NOT_FOUND;
  }

  //< Init the summary info
  bool is_job_stopped = (job_stats->second.end_time != 0);
  RDC_LOG(RDC_DEBUG, "rdc_job_get_stats for job " << jobId);
  auto& summary_info = p_job_info->summary;
  summary_info.start_time = job_stats->second.start_time;
  if (job_stats->second.end_time == 0) {
    summary_info.end_time = time(nullptr);
  } else {
    summary_info.end_time = job_stats->second.end_time;
  }
  summary_info.energy_consumed = 0;
  summary_info.max_gpu_memory_used = 0;
  summary_info.ecc_correct = 0;
  summary_info.ecc_uncorrect = 0;
  summary_info.power_usage = {0, std::numeric_limits<uint64_t>::max(), 0, 0};
  summary_info.pcie_tx = {0, std::numeric_limits<uint64_t>::max(), 0, 0};
  summary_info.pcie_rx = {0, std::numeric_limits<uint64_t>::max(), 0, 0};
  summary_info.pcie_total = {0, std::numeric_limits<uint64_t>::max(), 0, 0};
  summary_info.gpu_temperature = {0, std::numeric_limits<uint64_t>::max(), 0, 0};
  summary_info.memory_clock = {0, std::numeric_limits<uint64_t>::max(), 0, 0};
  summary_info.gpu_clock = {0, std::numeric_limits<uint64_t>::max(), 0, 0};
  summary_info.gpu_utilization = {0, std::numeric_limits<uint64_t>::max(), 0, 0};
  summary_info.memory_utilization = {0, std::numeric_limits<uint64_t>::max(), 0, 0};

  p_job_info->num_gpus = job_stats->second.gpu_stats.size();

  //< Populate information for each GPUs

  auto gpus = job_stats->second.gpu_stats.begin();
  for (; gpus != job_stats->second.gpu_stats.end(); gpus++) {
    auto& gpu_info = p_job_info->gpus[gpus->first];
    gpu_info.gpu_id = gpus->first;  // Explicit assignment
    gpu_info.start_time = summary_info.start_time;
    gpu_info.end_time = summary_info.end_time;
    gpu_info.energy_consumed = gpus->second.energy_consumed;
    summary_info.energy_consumed += gpu_info.energy_consumed;

    if (is_job_stopped) {
      gpu_info.ecc_correct = gpus->second.ecc_correct_init;
      summary_info.ecc_correct += gpu_info.ecc_correct;
    } else if (gpu_gauges.find({gpus->first, RDC_FI_ECC_CORRECT_TOTAL}) != gpu_gauges.end()) {
      gpu_info.ecc_correct =
          gpu_gauges.at({gpus->first, RDC_FI_ECC_CORRECT_TOTAL}) - gpus->second.ecc_correct_init;
      summary_info.ecc_correct += gpu_info.ecc_correct;
    } else {
      gpu_info.ecc_correct = 0;
    }

    if (is_job_stopped) {
      gpu_info.ecc_uncorrect = gpus->second.ecc_uncorrect_init;
      summary_info.ecc_uncorrect += gpu_info.ecc_uncorrect;
    } else if (gpu_gauges.find({gpus->first, RDC_FI_ECC_UNCORRECT_TOTAL}) != gpu_gauges.end()) {
      gpu_info.ecc_uncorrect = gpu_gauges.at({gpus->first, RDC_FI_ECC_UNCORRECT_TOTAL}) -
                               gpus->second.ecc_uncorrect_init;
      summary_info.ecc_uncorrect += gpu_info.ecc_uncorrect;
    } else {
      gpu_info.ecc_uncorrect = 0;
    }

    if (gpu_gauges.find({gpus->first, RDC_FI_GPU_MEMORY_TOTAL}) == gpu_gauges.end()) {
      RDC_LOG(RDC_ERROR, "Cannot find the total memory");
      return RDC_ST_BAD_PARAMETER;
    }
    uint64_t tmemory = gpu_gauges.at({gpus->first, RDC_FI_GPU_MEMORY_TOTAL});

    auto ite = gpus->second.field_summaries.begin();
    for (; ite != gpus->second.field_summaries.end(); ite++) {
      if (ite->first == RDC_FI_POWER_USAGE) {
        set_summary(ite->second, gpu_info.power_usage, summary_info.power_usage, 1000000);
      } else if (ite->first == RDC_FI_GPU_MEMORY_USAGE) {
        set_summary(ite->second, gpu_info.memory_utilization, summary_info.memory_utilization,
                    tmemory / 100);
        gpu_info.max_gpu_memory_used = ite->second.max_value;
        summary_info.max_gpu_memory_used =
            std::max(summary_info.max_gpu_memory_used, gpu_info.max_gpu_memory_used);
      } else if (ite->first == RDC_FI_GPU_CLOCK) {
        set_summary(ite->second, gpu_info.gpu_clock, summary_info.gpu_clock, 1000000);
      } else if (ite->first == RDC_FI_GPU_UTIL) {
        set_summary(ite->second, gpu_info.gpu_utilization, summary_info.gpu_utilization, 1);
      } else if (ite->first == RDC_FI_GPU_TEMP) {
        set_summary(ite->second, gpu_info.gpu_temperature, summary_info.gpu_temperature, 1000);
      } else if (ite->first == RDC_FI_MEM_CLOCK) {
        set_summary(ite->second, gpu_info.memory_clock, summary_info.memory_clock, 1000000);
      } else if (ite->first == RDC_FI_PCIE_TX) {
        set_summary(ite->second, gpu_info.pcie_tx, summary_info.pcie_tx, 1024 * 1024);
      } else if (ite->first == RDC_FI_PCIE_RX) {
        set_summary(ite->second, gpu_info.pcie_rx, summary_info.pcie_rx, 1024 * 1024);
      } else if (ite->first == RDC_FI_PCIE_BANDWIDTH) {
        set_summary(ite->second, gpu_info.pcie_total, summary_info.pcie_total, 1);
      }
    }
  }
  // Set the average of the summary
  set_average_summary(summary_info.power_usage, p_job_info->num_gpus);
  set_average_summary(summary_info.gpu_clock, p_job_info->num_gpus);
  set_average_summary(summary_info.gpu_utilization, p_job_info->num_gpus);
  set_average_summary(summary_info.memory_utilization, p_job_info->num_gpus);
  set_average_summary(summary_info.pcie_tx, p_job_info->num_gpus);
  set_average_summary(summary_info.pcie_rx, p_job_info->num_gpus);
  set_average_summary(summary_info.gpu_temperature, p_job_info->num_gpus);
  set_average_summary(summary_info.memory_clock, p_job_info->num_gpus);

  // Set process start/stop info
  auto& je = cache_jobs_[jobId];
  p_job_info->num_processes = je.num_processes;
  for (uint32_t i = 0; i < je.num_processes; ++i) {
    p_job_info->processes[i] = je.processes[i];
  }

  return RDC_ST_OK;
}

void RdcCacheManagerImpl::set_average_summary(rdc_stats_summary_t& summary, uint32_t num_gpus) {
  summary.average = summary.average / num_gpus;
  summary.standard_deviation = summary.standard_deviation / num_gpus;
}

rdc_status_t RdcCacheManagerImpl::rdc_job_start_stats(const char job_id[64],
                                                      const rdc_group_info_t& ginfo,
                                                      const rdc_field_group_info_t& finfo,
                                                      const rdc_gpu_gauges_t& gpu_gauges) {
  RdcJobStatsCacheEntry cacheEntry{};
  cacheEntry.start_time = std::time(nullptr);
  cacheEntry.end_time = 0;
  cacheEntry.num_processes = 0;
  cacheEntry.pid_to_index.clear();
  for (auto& p : cacheEntry.processes) p = {};
  for (unsigned int i = 0; i < ginfo.count; i++) {  // GPUs
    GpuSummaryStats gstats;
    gstats.energy_consumed = 0;
    gstats.energy_last_time = 0;
    for (unsigned int j = 0; j < finfo.count; j++) {  // init fields
      FieldSummaryStats s;
      s.count = 0;
      s.max_value = s.min_value = s.total_value = 0;
      gstats.field_summaries.insert({finfo.field_ids[j], s});
    }

    gstats.ecc_correct_init = 0;
    if (gpu_gauges.find({ginfo.entity_ids[i], RDC_FI_ECC_CORRECT_TOTAL}) != gpu_gauges.end()) {
      gstats.ecc_correct_init = gpu_gauges.at({ginfo.entity_ids[i], RDC_FI_ECC_CORRECT_TOTAL});
    }

    gstats.ecc_uncorrect_init = 0;
    if (gpu_gauges.find({ginfo.entity_ids[i], RDC_FI_ECC_UNCORRECT_TOTAL}) != gpu_gauges.end()) {
      gstats.ecc_uncorrect_init = gpu_gauges.at({ginfo.entity_ids[i], RDC_FI_ECC_UNCORRECT_TOTAL});
    }

    cacheEntry.gpu_stats.insert({ginfo.entity_ids[i], gstats});
  }

  std::lock_guard<std::mutex> guard(cache_mutex_);
  // Remove the old stats if it exists
  cache_jobs_.erase(job_id);
  cache_jobs_.insert({job_id, cacheEntry});
  return RDC_ST_OK;
}

rdc_status_t RdcCacheManagerImpl::rdc_job_stop_stats(const char job_id[64],
                                                     const rdc_gpu_gauges_t& gpu_gauges) {
  std::lock_guard<std::mutex> guard(cache_mutex_);
  auto job_stats = cache_jobs_.find(job_id);

  if (job_stats == cache_jobs_.end()) {
    return RDC_ST_NOT_FOUND;
  }

  job_stats->second.end_time = std::time(nullptr);

  // update the ecc errors
  auto gpus = job_stats->second.gpu_stats.begin();
  for (; gpus != job_stats->second.gpu_stats.end(); gpus++) {
    if (gpu_gauges.find({gpus->first, RDC_FI_ECC_CORRECT_TOTAL}) != gpu_gauges.end()) {
      gpus->second.ecc_correct_init =
          gpu_gauges.at({gpus->first, RDC_FI_ECC_CORRECT_TOTAL}) - gpus->second.ecc_correct_init;
    }

    if (gpu_gauges.find({gpus->first, RDC_FI_ECC_UNCORRECT_TOTAL}) != gpu_gauges.end()) {
      gpus->second.ecc_uncorrect_init = gpu_gauges.at({gpus->first, RDC_FI_ECC_UNCORRECT_TOTAL}) -
                                        gpus->second.ecc_uncorrect_init;
    }
  }

  return RDC_ST_OK;
}

rdc_status_t RdcCacheManagerImpl::rdc_health_set(rdc_gpu_group_t group_id, uint32_t gpu_index,
                                                 const rdc_field_value& value) {
  std::lock_guard<std::mutex> guard(cache_mutex_);
  RdcFieldKey field{gpu_index, value.field_id};

  // Set initial values
  RdcCacheEntry entry;
  entry.last_time = value.ts;
  entry.value = value.value;
  entry.type = value.type;

  auto health_ite = cache_health_.find(group_id);
  if (health_ite == cache_health_.end()) {
    std::vector<RdcCacheEntry> ve;
    ve.push_back(entry);

    RdcCacheSamples cache_sample;
    cache_sample.insert({field, ve});

    cache_health_.insert({group_id, cache_sample});
  } else {
    auto samples_ite = health_ite->second.find(field);
    if (samples_ite == health_ite->second.end()) {
      std::vector<RdcCacheEntry> ve;
      ve.push_back(entry);

      health_ite->second.insert({field, ve});
    } else {
      samples_ite->second.push_back(entry);
    }
  }

  return RDC_ST_OK;
}

rdc_status_t RdcCacheManagerImpl::rdc_health_get_values(
    rdc_gpu_group_t group_id, uint32_t gpu_index, rdc_field_t field_id, uint64_t start_timestamp,
    uint64_t end_timestamp, rdc_field_value* start_value, rdc_field_value* end_value) {
  if (!start_value && !end_value) return RDC_ST_BAD_PARAMETER;

  std::lock_guard<std::mutex> guard(cache_mutex_);
  auto health_ite = cache_health_.find(group_id);
  if (health_ite == cache_health_.end()) return RDC_ST_NOT_FOUND;

  RdcFieldKey field{gpu_index, field_id};
  auto samples_ite = health_ite->second.find(field);
  if (samples_ite == health_ite->second.end() || samples_ite->second.size() == 0)
    return RDC_ST_NOT_FOUND;

  auto cache_values = samples_ite->second;
  rdc_status_t result = RDC_ST_OK;
  if (start_value != nullptr) {
    // get start value
    result = RDC_ST_NOT_FOUND;
    for (auto entry = cache_values.begin(); entry != cache_values.end(); entry++) {
      if (entry->last_time >= start_timestamp) {
        start_value->field_id = field_id;
        start_value->ts = entry->last_time;
        start_value->type = entry->type;
        if (entry->type == STRING)
          strncpy_with_null(start_value->value.str, entry->value.str, RDC_MAX_STR_LENGTH);
        else
          start_value->value.l_int = entry->value.l_int;
        result = RDC_ST_OK;
        break;
      }
    }  // end for
  }    // end if

  if ((RDC_ST_OK == result) && (end_value != nullptr)) {
    // get end value
    result = RDC_ST_NOT_FOUND;
    for (auto entry = cache_values.rbegin(); entry != cache_values.rend(); entry++) {
      if (entry->last_time <= end_timestamp) {
        end_value->field_id = field_id;
        end_value->ts = entry->last_time;
        end_value->type = entry->type;
        if (entry->type == STRING)
          strncpy_with_null(end_value->value.str, entry->value.str, RDC_MAX_STR_LENGTH);
        else
          end_value->value.l_int = entry->value.l_int;
        result = RDC_ST_OK;
        break;
      }
    }  // end for
  }    // end if

  return result;
}

rdc_status_t RdcCacheManagerImpl::rdc_health_clear(rdc_gpu_group_t group_id) {
  std::lock_guard<std::mutex> guard(cache_mutex_);
  cache_health_.erase(group_id);
  return RDC_ST_OK;
}

rdc_status_t RdcCacheManagerImpl::rdc_update_health_stats(rdc_gpu_group_t group_id,
                                                          uint32_t gpu_index,
                                                          const rdc_field_value& value) {
  std::lock_guard<std::mutex> guard(cache_mutex_);
  auto health_ite = cache_health_.find(group_id);
  if (health_ite == cache_health_.end()) {
    return RDC_ST_NOT_FOUND;
  }

  RdcFieldKey field{gpu_index, value.field_id};
  auto samples_ite = health_ite->second.find(field);
  if (samples_ite == health_ite->second.end()) {
    return RDC_ST_NOT_FOUND;
  }

  // Check HEALTH_MAX_KEEP_SAMPLES
  auto& cache_values = samples_ite->second;
  int item_remove = cache_values.size() - HEALTH_MAX_KEEP_SAMPLES + 1;
  if (item_remove > 0) {
    cache_values.erase(cache_values.begin(), cache_values.begin() + item_remove);
  }

  RdcCacheEntry entry;
  entry.last_time = value.ts;
  entry.value = value.value;
  entry.type = value.type;

  cache_values.push_back(entry);

  return RDC_ST_OK;
}

}  // namespace rdc
}  // namespace amd
