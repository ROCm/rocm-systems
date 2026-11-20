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
#include "rdc_lib/DiagText.h"
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

// amd-smi reports a value it cannot read as its type's maximum.
template <typename T>
std::string value_or_na(T value) {
  return value == std::numeric_limits<T>::max() ? "n/a" : std::to_string(value);
}

// Whether amd-smi reports evidence that the process uses a GPU: GPU memory it holds, or compute
// units it occupies. A field at its own type's maximum is a value amd-smi could not read, which
// is no evidence either way.
bool uses_a_gpu(const amdsmi_process_info_t& proc) {
  const bool holds_memory =
      proc.vram_usage > 0 && proc.vram_usage < std::numeric_limits<uint64_t>::max();
  const bool occupies_cus =
      proc.cu_occupancy > 0 && proc.cu_occupancy < std::numeric_limits<uint32_t>::max();
  return holds_memory || occupies_cus;
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
  // gpu_index[] holds the ids the caller passed (the dispatcher passes at most
  // RDC_GROUP_MAX_ENTITIES); RDC_MAX_NUM_DEVICES is the bound of result->gpu_results[].
  gpu_count = std::min<uint32_t>(gpu_count, RDC_MAX_NUM_DEVICES);

  // Map first: each distinct entity of the group gives the slot it holds in RDC's flat GPU table,
  // which is the number amd-smi reports a process's GPUs by. Positions that hold the same entity
  // share its mapping, and an entity that maps to nothing gets no row and no sentence.
  std::map<uint32_t, uint32_t> entity_slots;
  // The group's positions by slot: two entities can name one slot, a physical GPU and a partition
  // of it, and both get the row of a process placed there.
  std::map<uint32_t, std::vector<uint32_t>> slot_positions;
  for (uint32_t i = 0; i < gpu_count; i++) {
    auto mapped = entity_slots.find(gpu_index[i]);
    if (mapped == entity_slots.end()) {
      uint32_t slot = kNoGpu;
      if (get_gpu_slot_from_entity(gpu_index[i], &slot) != AMDSMI_STATUS_SUCCESS) {
        slot = kNoGpu;  // no GPU of the table: this entity is left out of the check
      }
      mapped = entity_slots.emplace(gpu_index[i], slot).first;
    }
    if (mapped->second != kNoGpu) {
      slot_positions[mapped->second].push_back(i);
    }
  }
  // A group of which no GPU can be found is skipped before the process list is read, so that its
  // result does not depend on processes elsewhere on the host.
  if (slot_positions.empty()) {
    strncpy_with_null(result->info, "No GPU of the group could be found.", MAX_DIAG_MSG_LENGTH);
    return RDC_ST_OK;
  }

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
      // The Skip result carries the failure; rdcd sends no result for any other status.
      return RDC_ST_OK;
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

    // An entry amd-smi reports but does not write is all zero, which is no evidence that its
    // process uses a GPU, so it is left out below.
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
      // The Skip result carries the failure; rdcd sends no result for any other status.
      return RDC_ST_OK;
    }
    if (attempt == kProcessListReads) {
      RDC_LOG(RDC_DEBUG,
              "The process list changed during each of " << kProcessListReads << " reads");
      strncpy_with_null(result->info, "The process list changed while it was read.",
                        MAX_DIAG_MSG_LENGTH);
      return RDC_ST_OK;
    }
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

    // A process counts only on evidence that it uses a GPU, so that one which has a GPU open
    // without using it is left out: a process that has called hsa_shutdown() and released its
    // GPU memory stays listed in /proc until it ends, and an idle server keeps its GPUs open.
    if (!uses_a_gpu(procs[i])) continue;

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
      // The numbers amd-smi reports are rocm_smi device indexes, which are the table's slots.
      const auto positions = slot_positions.find(gpus[j]);
      if (gpus[j] == kNoGpu || positions == slot_positions.end()) continue;
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
  // gpu_index[] holds the ids the caller passed (the dispatcher passes at most
  // RDC_GROUP_MAX_ENTITIES); RDC_MAX_NUM_DEVICES is the most this check reads of them.
  gpu_count = std::min<uint32_t>(gpu_count, RDC_MAX_NUM_DEVICES);

  // One lookup per position of the group, before the pairs are walked: a lookup of a partition
  // entity lists amd-smi's sockets and processors. A GPU that cannot be found keeps its nullptr,
  // and the handles stay valid while the caller's amd-smi session lives.
  std::array<amdsmi_processor_handle, RDC_MAX_NUM_DEVICES> handles{};
  for (uint32_t i = 0; i < gpu_count; i++) {
    amdsmi_processor_handle handle = nullptr;
    if (get_processor_handle_from_id(gpu_index[i], &handle) == AMDSMI_STATUS_SUCCESS) {
      handles[i] = handle;
    }
  }

  DiagText info;
  uint32_t pairs = 0;
  for (uint32_t i = 0; i < gpu_count; i++) {
    for (uint32_t j = 0; j < gpu_count; j++) {
      if (gpu_index[i] == gpu_index[j]) continue;
      if (handles[i] == nullptr || handles[j] == nullptr) {
        continue;  // an unavailable GPU has no links to check
      }

      uint64_t weight = 0;
      const amdsmi_status_t err = amdsmi_topo_get_link_weight(handles[i], handles[j], &weight);
      if (err != AMDSMI_STATUS_SUCCESS) {
        result->status = RDC_DIAG_RESULT_FAIL;
        result->details.code = err;
        const std::string err_info = "amdsmi_topo_get_link_weight(" + std::to_string(gpu_index[i]) +
                                     "," + std::to_string(gpu_index[j]) + ", &weight) fail";
        strncpy_with_null(result->details.msg, err_info.c_str(), MAX_DIAG_MSG_LENGTH);
        strncpy_with_null(result->info, err_info.c_str(), MAX_DIAG_MSG_LENGTH);
        // The Fail result carries the failure; rdcd sends no result for any other status.
        return RDC_ST_OK;
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
  // gpu_index[] holds the ids the caller passed (the dispatcher passes at most
  // RDC_GROUP_MAX_ENTITIES); RDC_MAX_NUM_DEVICES is the bound of result->gpu_results[].
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
    // Each check's result is the highest of its GPUs' outcomes, which rests on the order
    // rdc.h declares rdc_diag_result_t in.
    static_assert(RDC_DIAG_RESULT_PASS < RDC_DIAG_RESULT_SKIP &&
                      RDC_DIAG_RESULT_SKIP < RDC_DIAG_RESULT_WARN &&
                      RDC_DIAG_RESULT_WARN < RDC_DIAG_RESULT_FAIL,
                  "rdc_diag_result_t must run from Pass to Fail for std::max to take the highest "
                  "outcome");
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
