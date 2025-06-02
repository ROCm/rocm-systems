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

#include <map>
#include <string>
#include <vector>

#include "rdc/rdc.h"
#include "rdc_lib/RdcLogger.h"
#include "rdc_lib/impl/SmiUtils.h"
#include "rdc_lib/rdc_common.h"

namespace amd {
namespace rdc {
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
  amdsmi_status_t err = AMDSMI_STATUS_SUCCESS;
  uint32_t num_items = 0;
  err = amdsmi_get_gpu_compute_process_info(nullptr, &num_items);
  if (err != AMDSMI_STATUS_SUCCESS) {
    RDC_LOG(RDC_ERROR, "Fail to get process information: " << err);
    strncpy_with_null(result->info, "Fail to retreive process information from amd_smi_lib",
                      MAX_DIAG_MSG_LENGTH);
    return Smi2RdcError(err);
  }

  // No process found
  if (num_items == 0) {
    result->status = RDC_DIAG_RESULT_PASS;
    result->per_gpu_result_count = 0;
    strncpy_with_null(result->info, "No processes running on any devices.", MAX_DIAG_MSG_LENGTH);
    return RDC_ST_OK;
  }

  std::string info;
  // Find details of the process running on each GPU
  std::vector<amdsmi_process_info_t> procs(num_items);
  err = amdsmi_get_gpu_compute_process_info(reinterpret_cast<amdsmi_process_info_t*>(&procs[0]),
                                            &num_items);
  if (err != AMDSMI_STATUS_SUCCESS) {
    RDC_LOG(RDC_INFO, "Fail to get process detail information: " << err);
    strncpy_with_null(result->info, info.c_str(), MAX_DIAG_MSG_LENGTH);
    return Smi2RdcError(err);
  }

  std::map<uint32_t, std::vector<uint32_t>> pids_per_gpu;
  for (uint32_t i = 0; i < num_items; i++) {
    // Skip the process does not occupy any GPUs. The hsa_shutdown()
    // will not clear /proc sys file until the process is terminated.
    if (procs[i].cu_occupancy == 0) continue;
    info += " Process: " + std::to_string(procs[i].process_id) +=
        ", vram_usage: " + std::to_string(procs[i].vram_usage) +=
        ", sdma_usage: " + std::to_string(procs[i].sdma_usage) +=
        ", cu_occupancy: " + std::to_string(procs[i].cu_occupancy) += ".";

    // Get the num_devices the process is running
    uint32_t num_devices = 0;
    amdsmi_status_t err;
    err = amdsmi_get_gpu_compute_process_gpus(procs[i].process_id, nullptr, &num_devices);
    if (err != AMDSMI_STATUS_SUCCESS || num_devices == 0) {
      RDC_LOG(RDC_INFO, "Fail to get process GPUs detail information: " << err);
      continue;
    }

    // Get the details of devices
    std::vector<uint32_t> device_details(num_devices);
    err = amdsmi_get_gpu_compute_process_gpus(
        procs[i].process_id, reinterpret_cast<uint32_t*>(&device_details[0]), &num_devices);
    if (err != AMDSMI_STATUS_SUCCESS) {
      RDC_LOG(RDC_INFO, "Fail to get process GPUs detail information: " << err);
      continue;
    }

    // Add process information in per GPU structure
    for (uint32_t j = 0; j < num_devices; j++) {
      // return the value if exists
      auto ite = pids_per_gpu.insert(
          std::pair<uint32_t, std::vector<uint32_t>>(device_details[j], std::vector<uint32_t>()));
      ite.first->second.push_back(procs[i].process_id);
    }
  }  // end for (uint32_t i=0 ...)

  result->status = RDC_DIAG_RESULT_PASS;  // pass by default
  if (pids_per_gpu.size() == 0) {
    result->per_gpu_result_count = 0;
    strncpy_with_null(result->info, "No processes running on any devices.", MAX_DIAG_MSG_LENGTH);
    return RDC_ST_OK;
  }

  // Mark as fail
  for (uint32_t i = 0; i < gpu_count; i++) {
    if (pids_per_gpu.find(gpu_index[i]) != pids_per_gpu.end()) {
      result->status = RDC_DIAG_RESULT_FAIL;
      break;
    }
  }

  // Set per GPU information
  strncpy_with_null(result->info, info.c_str(), MAX_DIAG_MSG_LENGTH);
  for (auto ite = pids_per_gpu.begin(); ite != pids_per_gpu.end(); ++ite) {
    auto& per_gpu = result->gpu_results[result->per_gpu_result_count];
    per_gpu.gpu_index = ite->first;
    per_gpu.gpu_result.code = 0;
    std::string per_gpu_msg = "Running process:";
    for (uint32_t k = 0; k < ite->second.size(); k++) {
      per_gpu_msg += " " + std::to_string(ite->second[k]);
    }

    strncpy_with_null(per_gpu.gpu_result.msg, per_gpu_msg.c_str(), MAX_DIAG_MSG_LENGTH);

    result->per_gpu_result_count++;
    if (result->per_gpu_result_count >= RDC_MAX_NUM_DEVICES) {
      RDC_LOG(RDC_ERROR, "Found more GPUs than " << RDC_MAX_NUM_DEVICES);
      break;
    }
  }

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

  const std::map<amdsmi_link_type_t, std::string> link_to_string = {
      {AMDSMI_LINK_TYPE_UNKNOWN, "Undefined"},
      {AMDSMI_LINK_TYPE_PCIE, "PCI Express"},
      {AMDSMI_LINK_TYPE_XGMI, "XGMI"}};

  result->status = RDC_DIAG_RESULT_SKIP;
  result->per_gpu_result_count = 0;
  amdsmi_status_t err = AMDSMI_STATUS_SUCCESS;
  std::string info = "";

  for (uint32_t i = 0; i < gpu_count; i++) {
    for (uint32_t j = 0; j < gpu_count; j++) {
      if (gpu_index[i] == gpu_index[j]) continue;
      std::pair<amdsmi_processor_handle, amdsmi_processor_handle> ph;
      err = get_processor_handle_from_id(gpu_index[i], &ph.first);
      err = get_processor_handle_from_id(gpu_index[i], &ph.second);

      uint64_t weight;
      err = amdsmi_topo_get_link_weight(ph.first, ph.second, &weight);
      if (err != AMDSMI_STATUS_SUCCESS) {
        result->status = RDC_DIAG_RESULT_FAIL;
        result->details.code = err;
        std::string err_info = "rsmi_topo_get_link_weight(";
        err_info += std::to_string(i) + ",";
        err_info += std::to_string(j) + ", &weight)";
        err_info += " fail";
        strncpy_with_null(result->details.msg, err_info.c_str(), MAX_DIAG_MSG_LENGTH);
        strncpy_with_null(result->info, err_info.c_str(), MAX_DIAG_MSG_LENGTH);
        return RDC_ST_SMI_ERROR;
      }

      info += std::to_string(i) + "=>";
      info += std::to_string(j) + " weight:";
      info += std::to_string(weight) + " ";
    }
  }
  if (info != "") {
    strncpy_with_null(result->info, info.c_str(), MAX_DIAG_MSG_LENGTH);
  } else {
    strncpy_with_null(result->info, "No link detected.", MAX_DIAG_MSG_LENGTH);
  }

  result->status = RDC_DIAG_RESULT_PASS;
  return RDC_ST_OK;
}

rdc_status_t RdcSmiDiagnosticImpl::check_smi_param_info(uint32_t gpu_index[RDC_MAX_NUM_DEVICES],
                                                        uint32_t gpu_count,
                                                        rdc_diag_test_result_t* result) {
  if (result == nullptr) {
    return RDC_ST_BAD_PARAMETER;
  }
  *result = {};
  result->test_case = RDC_DIAG_GPU_PARAMETERS;

  result->status = RDC_DIAG_RESULT_PASS;
  std::string info = "";

  for (uint32_t i = 0; i < gpu_count; i++) {
    // temperature
    for (amdsmi_temperature_type_t sensor_type = AMDSMI_TEMPERATURE_TYPE_FIRST;
         sensor_type < AMDSMI_TEMPERATURE_TYPE__MAX;) {
      auto status = check_temperature_level(gpu_index[i], sensor_type, result->info,
                                            result->gpu_results[i].gpu_result.msg);
      // Set to higher error level
      if (status > result->status) {
        result->status = status;
      }
      sensor_type = static_cast<amdsmi_temperature_type_t>(sensor_type + 1);
    }

    // Voltage
    for (amdsmi_voltage_type_t sensor_type = AMDSMI_VOLT_TYPE_FIRST;
         sensor_type < AMDSMI_VOLT_TYPE_LAST;) {
      auto status = check_voltage_level(gpu_index[i], sensor_type, result->info,
                                        result->gpu_results[i].gpu_result.msg);
      // Set to higher error level
      if (status > result->status) {
        result->status = status;
      }
      sensor_type = static_cast<amdsmi_voltage_type_t>(sensor_type + 1);
    }
    result->gpu_results->gpu_index = gpu_index[i];
    result->per_gpu_result_count++;
  }
  return RDC_ST_OK;
}

rdc_diag_result_t RdcSmiDiagnosticImpl::check_temperature_level(
    uint32_t gpu_index, amdsmi_temperature_type_t type, char msg[MAX_DIAG_MSG_LENGTH],
    char per_gpu_msg[MAX_DIAG_MSG_LENGTH]) {
  rdc_diag_result_t result = RDC_DIAG_RESULT_PASS;
  amdsmi_temperature_metric_t met = AMDSMI_TEMP_CURRENT;
  amdsmi_status_t err = AMDSMI_STATUS_SUCCESS;
  int64_t current_temp = 0;
  std::string info = msg;
  std::string per_gpu_info = per_gpu_msg;
  amdsmi_processor_handle processor_handle;
  get_processor_handle_from_id(gpu_index, &processor_handle);

  err = amdsmi_get_temp_metric(processor_handle, type, met, &current_temp);
  if (err != AMDSMI_STATUS_SUCCESS) return result;

  // Max temperature
  met = AMDSMI_TEMP_MAX;
  int64_t max_temp = 0;
  err = amdsmi_get_temp_metric(processor_handle, type, met, &max_temp);
  if (err == AMDSMI_STATUS_SUCCESS) {
    if (current_temp >= max_temp) {
      result = RDC_DIAG_RESULT_WARN;
      per_gpu_info += "Max ";
      per_gpu_info += get_temperature_string(type);
      per_gpu_info += " temperature ";
      per_gpu_info += std::to_string(max_temp);
      per_gpu_info += " greater than current temperature ";
      per_gpu_info += std::to_string(current_temp) + ". ";
      info += get_temperature_string(type) + ": ";
      info += "GPU " + std::to_string(gpu_index);
      info += " max ";
      info += get_temperature_string(type);
      info += " temperature exceeds. ";
    } else {
      info += "GPU " + std::to_string(gpu_index);
      info += " max ";
      info += get_temperature_string(type);
      info += " temperature in range. ";
    }
  }

  met = AMDSMI_TEMP_MIN;
  int64_t min_temp = 0;
  err = amdsmi_get_temp_metric(processor_handle, type, met, &min_temp);
  if (err == AMDSMI_STATUS_SUCCESS) {
    if (current_temp <= min_temp) {
      result = RDC_DIAG_RESULT_WARN;
      per_gpu_info += "Min ";
      per_gpu_info += get_temperature_string(type);
      per_gpu_info += " temperature ";
      per_gpu_info += std::to_string(min_temp);
      per_gpu_info += " less than current temperature ";
      per_gpu_info += std::to_string(current_temp) + ". ";
      info += "GPU " + std::to_string(gpu_index);
      info += " min ";
      info += get_temperature_string(type);
      info += " temperature exceeds. ";
    } else {
      info += "GPU " + std::to_string(gpu_index);
      info += " min ";
      info += get_temperature_string(type);
      info += " temperature in range. ";
    }
  }

  met = AMDSMI_TEMP_CRITICAL;
  int64_t critical_temp = 0;
  err = amdsmi_get_temp_metric(processor_handle, type, met, &critical_temp);
  if (err == AMDSMI_STATUS_SUCCESS) {
    if (current_temp >= critical_temp) {
      result = RDC_DIAG_RESULT_FAIL;
      per_gpu_info += "Critical ";
      per_gpu_info += get_temperature_string(type);
      per_gpu_info += " temperature ";
      per_gpu_info += std::to_string(critical_temp);
      per_gpu_info += " greater than current temperature ";
      per_gpu_info += std::to_string(current_temp) + ". ";
      info += "GPU " + std::to_string(gpu_index);
      info += " Critical ";
      info += get_temperature_string(type);
      info += " temperature exceeds. ";
    } else {
      info += "GPU " + std::to_string(gpu_index);
      info += " Critical ";
      info += get_temperature_string(type);
      info += " temperature in range. ";
    }
  }

  met = AMDSMI_TEMP_EMERGENCY;
  int64_t emergency_temp = 0;
  err = amdsmi_get_temp_metric(processor_handle, type, met, &emergency_temp);
  if (err == AMDSMI_STATUS_SUCCESS) {
    if (current_temp >= critical_temp) {
      result = RDC_DIAG_RESULT_FAIL;
      per_gpu_info += "Emergency ";
      per_gpu_info += get_temperature_string(type);
      per_gpu_info += " temperature ";
      per_gpu_info += std::to_string(emergency_temp);
      per_gpu_info += " greater than current temperature ";
      per_gpu_info += std::to_string(current_temp) + ". ";
      info += "GPU " + std::to_string(gpu_index);
      info += " Emergency ";
      info += get_temperature_string(type);
      info += " temperature exceeds. ";
    } else {
      info += "GPU " + std::to_string(gpu_index);
      info += " Emergency ";
      info += get_temperature_string(type);
      info += " temperature in range. ";
    }
  }

  met = AMDSMI_TEMP_CRIT_MIN;
  int64_t critical_min_temp = 0;
  err = amdsmi_get_temp_metric(processor_handle, type, met, &critical_min_temp);
  if (err == AMDSMI_STATUS_SUCCESS) {
    if (current_temp <= critical_min_temp) {
      result = RDC_DIAG_RESULT_FAIL;
      per_gpu_info += "Critical Min ";
      per_gpu_info += get_temperature_string(type);
      per_gpu_info += " temperature ";
      per_gpu_info += std::to_string(critical_min_temp);
      per_gpu_info += " less than current temperature ";
      per_gpu_info += std::to_string(current_temp) + ". ";
      info += "GPU " + std::to_string(gpu_index);
      info += " Critical Min ";
      info += get_temperature_string(type);
      info += " temperature exceeds. ";
    } else {
      info += "GPU " + std::to_string(gpu_index);
      info += " Critical Min ";
      info += get_temperature_string(type);
      info += " temperature in range. ";
    }
  }

  strncpy_with_null(msg, info.c_str(), MAX_DIAG_MSG_LENGTH);
  strncpy_with_null(per_gpu_msg, per_gpu_info.c_str(), MAX_DIAG_MSG_LENGTH);

  return result;
}

rdc_diag_result_t RdcSmiDiagnosticImpl::check_voltage_level(uint32_t gpu_index,
                                                            amdsmi_voltage_type_t type,
                                                            char msg[MAX_DIAG_MSG_LENGTH],
                                                            char per_gpu_msg[MAX_DIAG_MSG_LENGTH]) {
  rdc_diag_result_t result = RDC_DIAG_RESULT_PASS;
  amdsmi_voltage_metric_t met = AMDSMI_VOLT_CURRENT;
  amdsmi_status_t err = AMDSMI_STATUS_SUCCESS;
  int64_t current_voltage = 0;
  std::string info = msg;
  std::string per_gpu_info = per_gpu_msg;
  amdsmi_processor_handle processor_handle;
  get_processor_handle_from_id(gpu_index, &processor_handle);

  err = amdsmi_get_gpu_volt_metric(processor_handle, type, met, &current_voltage);
  if (err != AMDSMI_STATUS_SUCCESS) return result;

  // Max voltage
  met = AMDSMI_VOLT_MAX;
  int64_t max_volt = 0;
  err = amdsmi_get_gpu_volt_metric(processor_handle, type, met, &max_volt);
  if (err == AMDSMI_STATUS_SUCCESS) {
    if (current_voltage >= max_volt) {
      result = RDC_DIAG_RESULT_WARN;
      per_gpu_info += "Max ";
      per_gpu_info += get_voltage_string(type);
      per_gpu_info += " voltage ";
      per_gpu_info += std::to_string(max_volt);
      per_gpu_info += " greater than current voltage ";
      per_gpu_info += std::to_string(current_voltage) + ". ";
      info += get_voltage_string(type) + ": ";
      info += "GPU " + std::to_string(gpu_index);
      info += " max ";
      info += get_voltage_string(type);
      info += " voltage exceeds. ";
    } else {
      info += "GPU " + std::to_string(gpu_index);
      info += " max ";
      info += get_voltage_string(type);
      info += " voltage in range. ";
    }
  }

  // Min voltage
  met = AMDSMI_VOLT_MIN;
  int64_t min_volt = 0;
  err = amdsmi_get_gpu_volt_metric(processor_handle, type, met, &min_volt);
  if (err == AMDSMI_STATUS_SUCCESS) {
    if (current_voltage <= min_volt) {
      result = RDC_DIAG_RESULT_WARN;
      per_gpu_info += "Min ";
      per_gpu_info += get_voltage_string(type);
      per_gpu_info += " voltage ";
      per_gpu_info += std::to_string(min_volt);
      per_gpu_info += " less than current voltage ";
      per_gpu_info += std::to_string(current_voltage) + ". ";
      info += get_voltage_string(type) + ": ";
      info += "GPU " + std::to_string(gpu_index);
      info += " min ";
      info += get_voltage_string(type);
      info += " voltage exceeds. ";
    } else {
      info += "GPU " + std::to_string(gpu_index);
      info += " min ";
      info += get_voltage_string(type);
      info += " voltage in range. ";
    }
  }

  // Max Critical voltage
  met = AMDSMI_VOLT_MAX_CRIT;
  int64_t critical_max_volt = 0;
  err = amdsmi_get_gpu_volt_metric(nullptr, type, met, &critical_max_volt);
  if (err == AMDSMI_STATUS_SUCCESS) {
    if (current_voltage >= critical_max_volt) {
      result = RDC_DIAG_RESULT_FAIL;
      per_gpu_info += "Critical Max ";
      per_gpu_info += get_voltage_string(type);
      per_gpu_info += " voltage ";
      per_gpu_info += std::to_string(critical_max_volt);
      per_gpu_info += " greater than current voltage ";
      per_gpu_info += std::to_string(current_voltage) + ". ";
      info += get_voltage_string(type) + ": ";
      info += "GPU " + std::to_string(gpu_index);
      info += " Critical  max ";
      info += get_voltage_string(type);
      info += " voltage exceeds. ";
    } else {
      info += "GPU " + std::to_string(gpu_index);
      info += " Critical  max ";
      info += get_voltage_string(type);
      info += " voltage in range. ";
    }
  }

  // Min Critical voltage
  met = AMDSMI_VOLT_MIN_CRIT;
  int64_t critical_min_volt = 0;
  err = amdsmi_get_gpu_volt_metric(nullptr, type, met, &critical_min_volt);
  if (err == AMDSMI_STATUS_SUCCESS) {
    if (current_voltage <= critical_min_volt) {
      result = RDC_DIAG_RESULT_FAIL;
      per_gpu_info += "Critical Min ";
      per_gpu_info += get_voltage_string(type);
      per_gpu_info += " voltage ";
      per_gpu_info += std::to_string(critical_min_volt);
      per_gpu_info += " less than current voltage ";
      per_gpu_info += std::to_string(current_voltage) + ". ";
      info += get_voltage_string(type) + ": ";
      info += "GPU " + std::to_string(gpu_index);
      info += " Critical min ";
      info += get_voltage_string(type);
      info += " voltage exceeds. ";
    } else {
      info += "GPU " + std::to_string(gpu_index);
      info += " Critical min ";
      info += get_voltage_string(type);
      info += " voltage in range. ";
    }
  }

  strncpy_with_null(msg, info.c_str(), MAX_DIAG_MSG_LENGTH);
  strncpy_with_null(per_gpu_msg, per_gpu_info.c_str(), MAX_DIAG_MSG_LENGTH);

  return result;
}

}  // namespace rdc
}  // namespace amd
