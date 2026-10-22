/*
Copyright (c) 2021 - present Advanced Micro Devices, Inc. All rights reserved.

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
#include "rdc_lib/impl/RdcSmiDiagnosticImpl.h"

#include <amd_smi/amdsmi.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <limits>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "rdc/rdc.h"
#include "rdc_lib/RdcLogger.h"
#include "rdc_lib/impl/SmiUtils.h"
#include "rdc_lib/rdc_common.h"

namespace amd {
namespace rdc {
namespace {

// The most processes the compute process check reads: a list of at most 2 MB.
constexpr uint32_t kMaxProcesses = 65536;
// Room for the processes that start between the count call and the fill call.
constexpr uint32_t kProcessesStarting = 16;
// Reads of the process list (each a count call and a fill call) before the check gives up.
constexpr uint32_t kProcessListReads = 3;
// Fills a process's GPU list before amd-smi writes it: no GPU has this index.
constexpr uint32_t kNoGpu = std::numeric_limits<uint32_t>::max();

// A check's text, built from units (sentences, topology pairs, process IDs) joined by single
// spaces in at most MAX_DIAG_MSG_LENGTH - 1 bytes. The fixed start, if any, is always kept. Units
// are kept while they fit and the rest only counted; a cut text ends after a whole unit with
// " (N more not shown)".
class DiagText {
 public:
  explicit DiagText(std::string start = {}) : text_(std::move(start)) {}

  void add(const std::string& unit) {
    const size_t space = text_.empty() ? 0 : 1;
    if (more_ == 0 && text_.size() + space + unit.size() <= kBudget) {
      unit_starts_.push_back(text_.size());
      text_.append(space, ' ');
      text_ += unit;
    } else {
      more_++;
    }
  }

  bool has_units() const { return !unit_starts_.empty() || more_ != 0; }

  void copy_to(char (&out)[MAX_DIAG_MSG_LENGTH]) const {
    std::string text = text_;
    size_t kept = unit_starts_.size();
    size_t more = more_;
    while (more != 0) {
      const std::string suffix =
          (text.empty() ? "(" : " (") + std::to_string(more) + " more not shown)";
      if (text.size() + suffix.size() <= kBudget || kept == 0) {
        text += suffix;
        break;
      }
      text.resize(unit_starts_[--kept]);
      more++;
    }
    strncpy_with_null(out, text.c_str(), sizeof(out));
  }

 private:
  static constexpr size_t kBudget = MAX_DIAG_MSG_LENGTH - 1;
  std::string text_;
  std::vector<size_t> unit_starts_;  // where each kept unit, with the space before it, starts
  size_t more_ = 0;                  // units not kept
};

rdc_status_t skip_empty_group(rdc_diag_test_result_t* result) {
  result->status = RDC_DIAG_RESULT_SKIP;
  strncpy_with_null(result->info, "The group has no GPU.", MAX_DIAG_MSG_LENGTH);
  return RDC_ST_OK;
}

// amd-smi reports a value it cannot read as its type's maximum.
template <typename T>
std::string value_or_na(T value) {
  return value == std::numeric_limits<T>::max() ? "n/a" : std::to_string(value);
}

}  // namespace

RdcSmiDiagnosticImpl::RdcSmiDiagnosticImpl() {}

rdc_status_t RdcSmiDiagnosticImpl::check_smi_process_info(uint32_t gpu_index[RDC_MAX_NUM_DEVICES],
                                                          uint32_t gpu_count,
                                                          rdc_diag_test_result_t* result) {
  if (result == nullptr) {
    return RDC_ST_BAD_PARAMETER;
  }
  *result = {};
  result->test_case = RDC_DIAG_COMPUTE_PROCESS;
  result->status = RDC_DIAG_RESULT_SKIP;
  result->per_gpu_result_count = 0;
  if (gpu_count == 0) {
    return skip_empty_group(result);
  }
  // gpu_index and result->gpu_results hold RDC_MAX_NUM_DEVICES entries.
  gpu_count = std::min<uint32_t>(gpu_count, RDC_MAX_NUM_DEVICES);

  // A process that starts between the count call and the fill call can leave the fill without
  // room; amd-smi then fills in process IDs only, so the list is read again.
  std::vector<amdsmi_process_info_t> procs;
  uint32_t num_items = 0;
  for (uint32_t attempt = 1;; attempt++) {
    uint32_t count = 0;
    amdsmi_status_t err = amdsmi_get_gpu_compute_process_info(nullptr, &count);
    if (err != AMDSMI_STATUS_SUCCESS) {
      RDC_LOG(RDC_DEBUG, "Fail to get process information: " << err);
      strncpy_with_null(result->info, "Fail to retrieve process information from amd_smi_lib",
                        MAX_DIAG_MSG_LENGTH);
      return Smi2RdcError(err);
    }

    // No process found
    if (count == 0) {
      result->status = RDC_DIAG_RESULT_PASS;
      strncpy_with_null(result->info, "No processes running on any devices.", MAX_DIAG_MSG_LENGTH);
      return RDC_ST_OK;
    }
    if (count > kMaxProcesses) {
      RDC_LOG(RDC_DEBUG, "Too many processes to check: " << count);
      strncpy_with_null(result->info, "Too many processes to check.", MAX_DIAG_MSG_LENGTH);
      return RDC_ST_OK;
    }

    // An entry amd-smi reports but does not write has a CU occupancy of 0 and is skipped below.
    const uint32_t room = std::min(count + kProcessesStarting, kMaxProcesses);
    procs.assign(room, amdsmi_process_info_t{});
    num_items = room;
    err = amdsmi_get_gpu_compute_process_info(procs.data(), &num_items);
    if (err == AMDSMI_STATUS_SUCCESS) {
      num_items = std::min(num_items, room);
      break;
    }
    if (err != AMDSMI_STATUS_INSUFFICIENT_SIZE) {
      RDC_LOG(RDC_DEBUG, "Fail to get process detail information: " << err);
      strncpy_with_null(result->info, "Fail to retrieve process information from amd_smi_lib",
                        MAX_DIAG_MSG_LENGTH);
      return Smi2RdcError(err);
    }
    if (attempt == kProcessListReads) {
      RDC_LOG(RDC_DEBUG,
              "The process list changed during each of " << kProcessListReads << " reads");
      strncpy_with_null(result->info, "The process list changed while it was read.",
                        MAX_DIAG_MSG_LENGTH);
      return RDC_ST_OK;
    }
  }

  // The positions of each of the group's GPUs in the group.
  std::map<uint32_t, std::vector<uint32_t>> group_positions;
  for (uint32_t i = 0; i < gpu_count; i++) {
    group_positions[gpu_index[i]].push_back(i);
  }

  // Per group GPU, the IDs of the processes running on it.
  std::vector<DiagText> running(gpu_count, DiagText("Running process:"));
  std::vector<bool> on_group(num_items, false);
  uint32_t unreadable = 0;
  uint32_t on_many_gpus = 0;
  amdsmi_status_t last_error = AMDSMI_STATUS_SUCCESS;
  std::array<uint32_t, RDC_MAX_NUM_DEVICES> gpus;
  // The process that runs the check is left out: rdcd (or the program that embeds RDC) can stay
  // listed on a GPU after ROCr's checks have used it. amd-smi's process IDs are compared with
  // getpid(), which holds while amd-smi reports them in this process's PID namespace.
  const uint32_t self = static_cast<uint32_t>(getpid());
  for (uint32_t i = 0; i < num_items; i++) {
    if (procs[i].process_id == self) continue;

    // Skip the process does not occupy any GPUs. The hsa_shutdown()
    // will not clear /proc sys file until the process is terminated.
    if (procs[i].cu_occupancy == 0) continue;

    // The GPUs the process runs on, through a list: amd-smi rejects the NULL list that would ask
    // for their count. An entry still holding kNoGpu was not written.
    gpus.fill(kNoGpu);
    uint32_t num_devices = RDC_MAX_NUM_DEVICES;
    const amdsmi_status_t err =
        amdsmi_get_gpu_compute_process_gpus(procs[i].process_id, gpus.data(), &num_devices);
    if (err == AMDSMI_STATUS_NOT_FOUND) {
      continue;  // on no GPU, as when it exited after the process list was read
    }
    if (err == AMDSMI_STATUS_INSUFFICIENT_SIZE) {
      on_many_gpus++;
    } else if (err != AMDSMI_STATUS_SUCCESS) {
      last_error = err;
      unreadable++;
      continue;
    }

    const std::string pid = std::to_string(procs[i].process_id);
    for (uint32_t j = 0; j < std::min<uint32_t>(num_devices, RDC_MAX_NUM_DEVICES); j++) {
      const auto positions = group_positions.find(gpus[j]);
      if (gpus[j] == kNoGpu || positions == group_positions.end()) continue;
      for (uint32_t position : positions->second) {
        running[position].add(pid);
      }
      on_group[i] = true;
    }
  }

  // One line each, not one per process: a client can run the check.
  if (unreadable != 0) {
    RDC_LOG(RDC_DEBUG,
            "Fail to get the GPUs of " << unreadable << " processes, the last with " << last_error);
  }
  if (on_many_gpus != 0) {
    RDC_LOG(RDC_DEBUG,
            on_many_gpus << " processes run on more than " << RDC_MAX_NUM_DEVICES << " GPUs");
  }

  // Set per GPU information
  for (uint32_t i = 0; i < gpu_count; i++) {
    if (!running[i].has_units()) continue;
    auto& per_gpu = result->gpu_results[result->per_gpu_result_count++];
    per_gpu.gpu_index = gpu_index[i];
    per_gpu.gpu_result.code = 0;
    running[i].copy_to(per_gpu.gpu_result.msg);
  }

  const std::string unreadable_text =
      unreadable == 0 ? ""
                      : "Could not read the GPUs of " + std::to_string(unreadable) +
                            (unreadable == 1 ? " process." : " processes.");
  if (result->per_gpu_result_count == 0) {
    if (unreadable != 0) {
      strncpy_with_null(result->info, unreadable_text.c_str(), MAX_DIAG_MSG_LENGTH);
    } else {
      result->status = RDC_DIAG_RESULT_PASS;
      strncpy_with_null(result->info, "No processes running on the group's GPUs.",
                        MAX_DIAG_MSG_LENGTH);
    }
    return RDC_ST_OK;
  }

  result->status = RDC_DIAG_RESULT_FAIL;
  DiagText info(unreadable_text);
  for (uint32_t i = 0; i < num_items; i++) {
    if (!on_group[i]) continue;
    info.add("Process: " + std::to_string(procs[i].process_id) +
             ", vram_usage: " + value_or_na(procs[i].vram_usage) +
             ", sdma_usage: " + value_or_na(procs[i].sdma_usage) +
             ", cu_occupancy: " + value_or_na(procs[i].cu_occupancy) + ".");
  }
  info.copy_to(result->info);
  return RDC_ST_OK;
}

std::string RdcSmiDiagnosticImpl::get_temperature_string(amdsmi_temperature_type_t type) const {
  switch (type) {
    case AMDSMI_TEMPERATURE_TYPE_EDGE:
      return "Edge";
    case AMDSMI_TEMPERATURE_TYPE_JUNCTION:
      return "Junction";
    case AMDSMI_TEMPERATURE_TYPE_VRAM:
      return "Memory";
    default:
      return "Unknown";
  }
}

std::string RdcSmiDiagnosticImpl::get_voltage_string(amdsmi_voltage_type_t type) const {
  switch (type) {
    case AMDSMI_VOLT_TYPE_VDDGFX:
      return "Vddgfx voltage";
    default:
      return "Unknown";
  }
}

// Show topology type
rdc_status_t RdcSmiDiagnosticImpl::check_smi_topo_info(uint32_t gpu_index[RDC_MAX_NUM_DEVICES],
                                                       uint32_t gpu_count,
                                                       rdc_diag_test_result_t* result) {
  if (result == nullptr) {
    return RDC_ST_BAD_PARAMETER;
  }
  *result = {};
  result->test_case = RDC_DIAG_NODE_TOPOLOGY;
  result->status = RDC_DIAG_RESULT_SKIP;
  result->per_gpu_result_count = 0;
  if (gpu_count == 0) {
    return skip_empty_group(result);
  }
  // gpu_index holds RDC_MAX_NUM_DEVICES entries.
  gpu_count = std::min<uint32_t>(gpu_count, RDC_MAX_NUM_DEVICES);

  DiagText info;
  uint32_t pairs = 0;
  for (uint32_t i = 0; i < gpu_count; i++) {
    for (uint32_t j = 0; j < gpu_count; j++) {
      if (gpu_index[i] == gpu_index[j]) continue;
      std::pair<amdsmi_processor_handle, amdsmi_processor_handle> ph{nullptr, nullptr};
      if (get_processor_handle_from_id(gpu_index[i], &ph.first) != AMDSMI_STATUS_SUCCESS ||
          get_processor_handle_from_id(gpu_index[j], &ph.second) != AMDSMI_STATUS_SUCCESS) {
        continue;  // an unavailable GPU has no links to check
      }

      uint64_t weight = 0;
      const amdsmi_status_t err = amdsmi_topo_get_link_weight(ph.first, ph.second, &weight);
      if (err != AMDSMI_STATUS_SUCCESS) {
        result->status = RDC_DIAG_RESULT_FAIL;
        result->details.code = err;
        const std::string err_info = "amdsmi_topo_get_link_weight(" + std::to_string(gpu_index[i]) +
                                     "," + std::to_string(gpu_index[j]) + ", &weight) fail";
        strncpy_with_null(result->details.msg, err_info.c_str(), MAX_DIAG_MSG_LENGTH);
        strncpy_with_null(result->info, err_info.c_str(), MAX_DIAG_MSG_LENGTH);
        return RDC_ST_SMI_ERROR;
      }

      pairs++;
      info.add(std::to_string(gpu_index[i]) + "=>" + std::to_string(gpu_index[j]) +
               " weight:" + std::to_string(weight));
    }
  }
  if (pairs == 0) {
    strncpy_with_null(result->info, "No pair of available GPUs to check.", MAX_DIAG_MSG_LENGTH);
    return RDC_ST_OK;
  }

  info.copy_to(result->info);
  result->status = RDC_DIAG_RESULT_PASS;
  return RDC_ST_OK;
}

static bool rdc_diag_temperature_is_relevant(amdsmi_temperature_type_t type) {
  return type == AMDSMI_TEMPERATURE_TYPE_EDGE || type == AMDSMI_TEMPERATURE_TYPE_JUNCTION ||
         type == AMDSMI_TEMPERATURE_TYPE_VRAM;
}

rdc_status_t RdcSmiDiagnosticImpl::check_smi_param_info(uint32_t gpu_index[RDC_MAX_NUM_DEVICES],
                                                        uint32_t gpu_count,
                                                        rdc_diag_test_result_t* result) {
  if (result == nullptr) {
    return RDC_ST_BAD_PARAMETER;
  }
  *result = {};
  result->test_case = RDC_DIAG_GPU_PARAMETERS;
  if (gpu_count == 0) {
    return skip_empty_group(result);
  }
  // gpu_index and result->gpu_results hold RDC_MAX_NUM_DEVICES entries.
  gpu_count = std::min<uint32_t>(gpu_count, RDC_MAX_NUM_DEVICES);

  result->status = RDC_DIAG_RESULT_PASS;
  DiagText info;

  for (uint32_t i = 0; i < gpu_count; i++) {
    GpuLimits gpu;
    amdsmi_processor_handle processor_handle = nullptr;
    if (get_processor_handle_from_id(gpu_index[i], &processor_handle) == AMDSMI_STATUS_SUCCESS) {
      // temperature
      for (amdsmi_temperature_type_t sensor_type = AMDSMI_TEMPERATURE_TYPE_FIRST;
           sensor_type < AMDSMI_TEMPERATURE_TYPE__MAX;
           sensor_type = static_cast<amdsmi_temperature_type_t>(sensor_type + 1)) {
        if (rdc_diag_temperature_is_relevant(sensor_type)) {
          check_temperature_level(processor_handle, sensor_type, &gpu);
        }
      }

      // Voltage
      for (amdsmi_voltage_type_t sensor_type = AMDSMI_VOLT_TYPE_FIRST;
           sensor_type < AMDSMI_VOLT_TYPE_LAST;
           sensor_type = static_cast<amdsmi_voltage_type_t>(sensor_type + 1)) {
        check_voltage_level(processor_handle, sensor_type, &gpu);
      }
    }

    auto& per_gpu = result->gpu_results[i];
    per_gpu.gpu_index = gpu_index[i];
    const std::string name = "GPU " + std::to_string(gpu_index[i]) + ": ";
    const char* limits = gpu.compared == 1 ? " limit" : " limits";
    if (gpu.compared == 0) {
      gpu.status = RDC_DIAG_RESULT_SKIP;
      info.add(name + "no limit could be compared.");
      strncpy_with_null(per_gpu.gpu_result.msg,
                        "No temperature or voltage limit could be compared.", MAX_DIAG_MSG_LENGTH);
    } else if (gpu.crossed == 0) {
      info.add(name + std::to_string(gpu.compared) + limits + " in range.");
    } else {
      info.add(name + std::to_string(gpu.crossed) + " of " + std::to_string(gpu.compared) + limits +
               " crossed.");
      DiagText messages;
      for (const std::string& message : gpu.messages) {
        messages.add(message);
      }
      messages.copy_to(per_gpu.gpu_result.msg);
    }
    // Set to higher error level
    result->status = std::max(result->status, gpu.status);
    result->per_gpu_result_count++;
  }

  info.copy_to(result->info);
  return RDC_ST_OK;
}

void RdcSmiDiagnosticImpl::GpuLimits::compare(const std::string& sensor, int64_t current,
                                              int64_t limit, bool upper, const char* name,
                                              rdc_diag_result_t level) {
  compared++;
  if (upper ? current < limit : current > limit) return;
  crossed++;
  status = std::max(status, level);
  messages.push_back(sensor + " " + std::to_string(current) +
                     (upper ? " is at or above its " : " is at or below its ") + name + " " +
                     std::to_string(limit) + ".");
}

void RdcSmiDiagnosticImpl::check_temperature_level(amdsmi_processor_handle processor_handle,
                                                   amdsmi_temperature_type_t type,
                                                   GpuLimits* gpu) const {
  int64_t current_temp = 0;
  if (amdsmi_get_temp_metric(processor_handle, type, AMDSMI_TEMP_CURRENT, &current_temp) !=
      AMDSMI_STATUS_SUCCESS) {
    return;
  }

  struct Limit {
    amdsmi_temperature_metric_t metric;
    bool upper;  // crossed at or above it, else at or below it
    const char* name;
    rdc_diag_result_t level;
  };
  static constexpr Limit kLimits[] = {
      {AMDSMI_TEMP_MAX, true, "maximum", RDC_DIAG_RESULT_WARN},
      {AMDSMI_TEMP_MIN, false, "minimum", RDC_DIAG_RESULT_WARN},
      {AMDSMI_TEMP_CRITICAL, true, "critical limit", RDC_DIAG_RESULT_FAIL},
      {AMDSMI_TEMP_EMERGENCY, true, "emergency limit", RDC_DIAG_RESULT_FAIL},
      {AMDSMI_TEMP_CRIT_MIN, false, "critical minimum", RDC_DIAG_RESULT_FAIL},
  };
  const std::string sensor = get_temperature_string(type) + " temperature";
  for (const Limit& l : kLimits) {
    int64_t limit = 0;
    if (amdsmi_get_temp_metric(processor_handle, type, l.metric, &limit) != AMDSMI_STATUS_SUCCESS) {
      continue;
    }
    // amd-smi reports 0 for a limit it does not have; a minimum may be below 0.
    if (l.upper ? limit <= 0 : limit == 0) continue;
    gpu->compare(sensor, current_temp, limit, l.upper, l.name, l.level);
  }
}

void RdcSmiDiagnosticImpl::check_voltage_level(amdsmi_processor_handle processor_handle,
                                               amdsmi_voltage_type_t type, GpuLimits* gpu) const {
  int64_t current_voltage = 0;
  if (amdsmi_get_gpu_volt_metric(processor_handle, type, AMDSMI_VOLT_CURRENT, &current_voltage) !=
      AMDSMI_STATUS_SUCCESS) {
    return;
  }

  struct Limit {
    amdsmi_voltage_metric_t metric;
    bool upper;  // crossed at or above it, else at or below it
    const char* name;
    rdc_diag_result_t level;
  };
  static constexpr Limit kLimits[] = {
      {AMDSMI_VOLT_MAX, true, "maximum", RDC_DIAG_RESULT_WARN},
      {AMDSMI_VOLT_MIN, false, "minimum", RDC_DIAG_RESULT_WARN},
      {AMDSMI_VOLT_MAX_CRIT, true, "critical maximum", RDC_DIAG_RESULT_FAIL},
      {AMDSMI_VOLT_MIN_CRIT, false, "critical minimum", RDC_DIAG_RESULT_FAIL},
  };
  const std::string sensor = get_voltage_string(type);
  for (const Limit& l : kLimits) {
    int64_t limit = 0;
    if (amdsmi_get_gpu_volt_metric(processor_handle, type, l.metric, &limit) !=
        AMDSMI_STATUS_SUCCESS) {
      continue;
    }
    // amd-smi reports 0 for a limit it does not have.
    if (limit <= 0) continue;
    gpu->compare(sensor, current_voltage, limit, l.upper, l.name, l.level);
  }
}

}  // namespace rdc
}  // namespace amd
