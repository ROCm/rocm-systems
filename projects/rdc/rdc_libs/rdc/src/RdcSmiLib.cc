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
#include "rdc_lib/impl/RdcSmiLib.h"

#include <amd_smi/amdsmi.h>
#include <stdlib.h>
#include <strings.h>

#include <array>
#include <chrono>
#include <string>
#include <vector>

#include "rdc/rdc.h"
#include "rdc_lib/RdcLogger.h"
#include "rdc_lib/impl/RdcPollLatencyTest.h"
#include "rdc_lib/rdc_common.h"

namespace amd {
namespace rdc {

RdcSmiLib::RdcSmiLib(const RdcMetricFetcherPtr& mf)
    : metric_fetcher_(mf),
      bulk_fetch_enabled_(false),
      smi_diag_(std::make_shared<RdcSmiDiagnosticImpl>()) {
  char* bulk_env = getenv("RDC_BULK_FETCH_ENABLED");
  if (bulk_env != nullptr && strcasecmp(bulk_env, "true") == 0) {
    RDC_LOG(RDC_DEBUG, "Bulk fetch enabled.");
    bulk_fetch_enabled_ = true;
  } else {
    RDC_LOG(RDC_DEBUG, "Bulk fetch disabled.");
  }
}

// Bulk fetch wrapper for the amd_smi_lib. This will be replaced after
// amd_smi_lib can support bulk fetch.
rdc_status_t RdcSmiLib::rdc_telemetry_fields_value_get(rdc_gpu_field_t* fields,
                                                       uint32_t fields_count,
                                                       rdc_field_value_f callback,
                                                       void* user_data) {
  std::lock_guard<std::mutex> guard(fetch_mutex_);
  return read_fields_locked(fields, fields_count, callback, user_data);
}

rdc_status_t RdcSmiLib::read_fields_locked(rdc_gpu_field_t* fields, uint32_t fields_count,
                                           rdc_field_value_f callback, void* user_data) {
  if (fields == nullptr) {
    return RDC_ST_BAD_PARAMETER;
  }

  RDC_LOG(RDC_DEBUG, "Fetch " << fields_count << " fields from amd_smi_lib.");

  // Bulk fetch fields
  std::vector<rdc_gpu_field_value_t> bulk_results;
  if (bulk_fetch_enabled_) {
    rdc_status_t status =
        metric_fetcher_->bulk_fetch_smi_fields(fields, fields_count, bulk_results);
    RDC_LOG(RDC_DEBUG, "Bulk fetched " << bulk_results.size()
                                       << " fields from amd_smi_lib which return " << status);
    if (bulk_results.size() > 0) {
      rdc_status_t status = callback(&bulk_results[0], bulk_results.size(), user_data);
      if (status != RDC_ST_OK) {
        return status;
      }
    }
  }

  // Fetch it one by one for left fields
  const int BULK_FIELDS_MAX = 16;
  rdc_gpu_field_value_t values[BULK_FIELDS_MAX];
  uint32_t bulk_count = 0;
  for (uint32_t i = 0; i < fields_count; i++) {
    bool is_fetched = false;
    for (std::size_t j = 0; j < bulk_results.size(); j++) {
      if (bulk_results[j].gpu_index == fields[i].gpu_index &&
          bulk_results[j].field_value.field_id == fields[i].field_id) {
        is_fetched = true;
        break;
      }
    }
    if (is_fetched) continue;
    if (bulk_count >= BULK_FIELDS_MAX) {
      rdc_status_t status = callback(values, bulk_count, user_data);
      // When the callback returns errors, stop processing and return.
      if (status != RDC_ST_OK) {
        return status;
      }
      bulk_count = 0;
    }
    rdc_gpu_field_value_t& slot = values[bulk_count];
    slot = {};
    slot.gpu_index = fields[i].gpu_index;
    const rdc_status_t fetch_status = metric_fetcher_->fetch_smi_field(
        fields[i].gpu_index, static_cast<rdc_field_t>(fields[i].field_id), &slot.field_value);
    if (slot.field_value.ts == 0) {
      // fetch_smi_field() returned before it filled in the value.
      slot.field_value.field_id = static_cast<rdc_field_t>(fields[i].field_id);
      slot.field_value.status = (fetch_status == RDC_ST_NOT_SUPPORTED)
                                    ? static_cast<int>(AMDSMI_STATUS_NOT_SUPPORTED)
                                    : static_cast<int>(AMDSMI_STATUS_UNKNOWN_ERROR);
    }
    bulk_count++;
  }
  if (bulk_count != 0) {
    rdc_status_t status = callback(values, bulk_count, user_data);
    if (status != RDC_ST_OK) {
      return status;
    }
    bulk_count = 0;
  }

  return RDC_ST_OK;
}

rdc_status_t RdcSmiLib::rdc_telemetry_fields_watch(rdc_gpu_field_t* fields, uint32_t fields_count) {
  rdc_status_t ret;

  if (fields == nullptr) {
    return RDC_ST_BAD_PARAMETER;
  }

  for (uint32_t i = 0; i < fields_count; i++) {
    ret = metric_fetcher_->acquire_smi_handle({fields[i].gpu_index, fields[i].field_id});
    if (ret != RDC_ST_OK) {
      RDC_LOG(RDC_ERROR, "Failed to acquire amd_smi handle for field.");
    }
  }
  RDC_LOG(RDC_DEBUG, "acquire " << fields_count << " field handles from amd_smi_lib");

  return RDC_ST_OK;
}

rdc_status_t RdcSmiLib::rdc_telemetry_fields_unwatch(rdc_gpu_field_t* fields,
                                                     uint32_t fields_count) {
  if (fields == nullptr) {
    return RDC_ST_BAD_PARAMETER;
  }

  for (uint32_t i = 0; i < fields_count; i++) {
    metric_fetcher_->delete_smi_handle({fields[i].gpu_index, fields[i].field_id});
  }
  RDC_LOG(RDC_DEBUG, "delete " << fields_count << " field handles from amd_smi_lib");

  return RDC_ST_OK;
}

rdc_status_t RdcSmiLib::rdc_telemetry_fields_query(uint32_t field_ids[MAX_NUM_FIELDS],
                                                   uint32_t* field_count) {
  if (field_count == nullptr) {
    return RDC_ST_BAD_PARAMETER;
  }

  // List of fields supported by amd_smi_lib
  // clang-format off
  const std::vector<uint32_t> fields{
      RDC_FI_GPU_COUNT,                RDC_FI_DEV_NAME,               RDC_FI_GPU_CLOCK,
      RDC_FI_MEM_CLOCK,                RDC_FI_GPU_CLOCK_MIN,          RDC_FI_GPU_CLOCK_MAX,
      RDC_FI_SOC_CLOCK,                RDC_FI_VCLK0,                  RDC_FI_DCLK0,
      RDC_FI_MEMORY_TEMP,              RDC_FI_GPU_TEMP,               RDC_FI_GPU_JUNCTION_TEMP,
      RDC_FI_POWER_USAGE,              RDC_FI_GPU_ENERGY,             RDC_FI_PCIE_TX,
      RDC_FI_PCIE_RX,
      RDC_FI_PCIE_BANDWIDTH,           RDC_FI_PCIE_LC_PERF_OTHER_END_RECOVERY, RDC_FI_PCIE_NAK_RCVD_COUNT_ACC,
      RDC_FI_PCIE_NAK_SENT_COUNT_ACC,  RDC_FI_PCIE_SPEED,             RDC_FI_PCIE_MAX_SPEED,
      RDC_FI_PCIE_REPLAY_ROLLOVER,     RDC_FI_PCIE_BANDWIDTH_BIDIR,
      RDC_FI_GPU_UTIL,                 RDC_FI_GPU_MEMORY_USAGE,
      RDC_FI_GPU_MEMORY_TOTAL,         RDC_FI_GPU_MEMORY_FREE,
      RDC_FI_GPU_VIS_VRAM_TOTAL,       RDC_FI_GPU_VIS_VRAM_USED,      RDC_FI_GPU_VIS_VRAM_FREE,
      RDC_FI_GPU_GTT_TOTAL,            RDC_FI_GPU_GTT_USED,           RDC_FI_GPU_GTT_FREE,
      RDC_FI_GPU_GFX_BUSY_INST,        RDC_FI_GPU_VCN_BUSY_INST,      RDC_FI_GPU_JPEG_BUSY_INST,
      RDC_FI_ECC_CORRECT_TOTAL,        RDC_FI_ECC_UNCORRECT_TOTAL,
      RDC_FI_ECC_SDMA_CE,              RDC_FI_ECC_SDMA_UE,            RDC_FI_ECC_GFX_CE,
      RDC_FI_ECC_GFX_UE,               RDC_FI_ECC_MMHUB_CE,           RDC_FI_ECC_MMHUB_UE,
      RDC_FI_ECC_ATHUB_CE,             RDC_FI_ECC_ATHUB_UE,           RDC_FI_ECC_PCIE_BIF_CE,
      RDC_FI_ECC_PCIE_BIF_UE,          RDC_FI_ECC_HDP_CE,             RDC_FI_ECC_HDP_UE,
      RDC_FI_ECC_XGMI_WAFL_CE,         RDC_FI_ECC_XGMI_WAFL_UE,       RDC_FI_ECC_DF_CE,
      RDC_FI_ECC_DF_UE,                RDC_FI_ECC_SMN_CE,             RDC_FI_ECC_SMN_UE,
      RDC_FI_ECC_SEM_CE,               RDC_FI_ECC_SEM_UE,             RDC_FI_ECC_MP0_CE,
      RDC_FI_ECC_MP0_UE,               RDC_FI_ECC_MP1_CE,             RDC_FI_ECC_MP1_UE,
      RDC_FI_ECC_FUSE_CE,              RDC_FI_ECC_FUSE_UE,            RDC_FI_ECC_UMC_CE,
      RDC_FI_ECC_UMC_UE,               RDC_FI_ECC_MCA_CE,             RDC_FI_ECC_MCA_UE,
      RDC_FI_ECC_VCN_CE,               RDC_FI_ECC_VCN_UE,             RDC_FI_ECC_JPEG_CE,
      RDC_FI_ECC_JPEG_UE,              RDC_FI_ECC_IH_CE,              RDC_FI_ECC_IH_UE,
      RDC_FI_ECC_MPIO_CE,              RDC_FI_ECC_MPIO_UE,
      RDC_FI_ECC_SDMA_DE,              RDC_FI_ECC_GFX_DE,             RDC_FI_ECC_MMHUB_DE,
      RDC_FI_ECC_ATHUB_DE,             RDC_FI_ECC_PCIE_BIF_DE,        RDC_FI_ECC_HDP_DE,
      RDC_FI_ECC_XGMI_WAFL_DE,         RDC_FI_ECC_DF_DE,              RDC_FI_ECC_SMN_DE,
      RDC_FI_ECC_SEM_DE,               RDC_FI_ECC_MP0_DE,             RDC_FI_ECC_MP1_DE,
      RDC_FI_ECC_FUSE_DE,              RDC_FI_ECC_UMC_DE,             RDC_FI_ECC_MCA_DE,
      RDC_FI_ECC_VCN_DE,               RDC_FI_ECC_JPEG_DE,            RDC_FI_ECC_IH_DE,
      RDC_FI_ECC_MPIO_DE,              RDC_FI_ECC_DEFERRED_TOTAL,     RDC_FI_AFID,
      RDC_FI_XGMI_0_READ_KB,
      RDC_FI_XGMI_1_READ_KB,           RDC_FI_XGMI_2_READ_KB,         RDC_FI_XGMI_3_READ_KB,
      RDC_FI_XGMI_4_READ_KB,           RDC_FI_XGMI_5_READ_KB,         RDC_FI_XGMI_6_READ_KB,
      RDC_FI_XGMI_7_READ_KB,           RDC_FI_XGMI_TOTAL_READ_KB,     RDC_FI_XGMI_0_WRITE_KB,
      RDC_FI_XGMI_1_WRITE_KB,          RDC_FI_XGMI_2_WRITE_KB,        RDC_FI_XGMI_3_WRITE_KB,
      RDC_FI_XGMI_4_WRITE_KB,          RDC_FI_XGMI_5_WRITE_KB,        RDC_FI_XGMI_6_WRITE_KB,
      RDC_FI_XGMI_7_WRITE_KB,          RDC_FI_XGMI_TOTAL_WRITE_KB,
      RDC_EVNT_XGMI_0_NOP_TX,          RDC_EVNT_XGMI_0_REQ_TX,        RDC_EVNT_XGMI_0_RESP_TX,
      RDC_EVNT_XGMI_0_BEATS_TX,        RDC_EVNT_XGMI_1_NOP_TX,        RDC_EVNT_XGMI_1_REQ_TX,
      RDC_EVNT_XGMI_1_RESP_TX,         RDC_EVNT_XGMI_1_BEATS_TX,      RDC_EVNT_XGMI_0_THRPUT,
      RDC_EVNT_XGMI_1_THRPUT,          RDC_EVNT_XGMI_2_THRPUT,        RDC_EVNT_XGMI_3_THRPUT,
      RDC_EVNT_XGMI_4_THRPUT,          RDC_EVNT_XGMI_5_THRPUT,        RDC_FI_OAM_ID,
      RDC_FI_GPU_MM_ENC_UTIL,          RDC_FI_GPU_MM_DEC_UTIL,        RDC_FI_GPU_MEMORY_ACTIVITY,
      RDC_HEALTH_XGMI_ERROR,           RDC_HEALTH_PCIE_REPLAY_COUNT,  RDC_HEALTH_RETIRED_PAGE_NUM,
      RDC_HEALTH_PENDING_PAGE_NUM,     RDC_HEALTH_RETIRED_PAGE_LIMIT, RDC_HEALTH_EEPROM_CONFIG_VALID,
      RDC_HEALTH_POWER_THROTTLE_TIME,  RDC_HEALTH_THERMAL_THROTTLE_TIME,
      RDC_HEALTH_VIOLATION_COUNTER,    RDC_HEALTH_PROCHOT_RESIDENCY_ACC,
      RDC_HEALTH_PROCHOT_RESIDENCY_PCT,RDC_HEALTH_PPT_RESIDENCY_PCT,
      RDC_HEALTH_SOCKET_THRM_ACC,      RDC_HEALTH_SOCKET_THRM_PCT,
      RDC_HEALTH_VR_THRM_ACC,          RDC_HEALTH_VR_THRM_PCT,
      RDC_HEALTH_HBM_THRM_ACC,         RDC_HEALTH_HBM_THRM_PCT,
      RDC_HEALTH_GFX_CLK_LMT_PWR_ACC,  RDC_HEALTH_GFX_CLK_LMT_PWR_PCT,
      RDC_HEALTH_GFX_CLK_LMT_THM_ACC,  RDC_HEALTH_GFX_CLK_LMT_THM_PCT,
      RDC_HEALTH_LOW_UTIL_ACC,         RDC_HEALTH_LOW_UTIL_PCT,
      RDC_HEALTH_GFX_CLK_LMT_TOTAL_ACC,RDC_HEALTH_GFX_CLK_LMT_TOTAL_PCT,
      RDC_FI_GPU_MEMORY_MAX_BANDWIDTH, RDC_FI_GPU_MEMORY_CUR_BANDWIDTH,
      RDC_FI_GPU_BUSY_PERCENT,         RDC_FI_GFX_ACTIVITY_ACC,        RDC_FI_MEM_ACTIVITY_ACC,
      RDC_FI_ACCUMULATION_COUNTER,     RDC_FI_GPU_PAGE_RETRIED,
      RDC_FI_DEV_ID,                   RDC_FI_REV_ID,                 RDC_FI_TARGET_GRAPHICS_VERSION,
      RDC_FI_NUM_OF_COMPUTE_UNITS,     RDC_FI_UUID,                   RDC_FI_GPU_PARTITION_COUNT,
      RDC_FI_KFD_ID,                   RDC_FI_CPU_SKT_COUNT,          RDC_FI_CPU_MODEL,
      RDC_FI_CPU_MODEL_ID,             RDC_FI_CPU_FAMILY,             RDC_FI_CPU_CORES_PER_SKT,
      RDC_FI_CPU_SKT_ENERGY,           RDC_FI_CPU_HSMP_DRIVER_VERSION,RDC_FI_CPU_SMU_FW_VERSION,
      RDC_FI_CPU_HSMP_PROTO_VERSION,   RDC_FI_CPU_FCLK_FREQUENCY,     RDC_FI_CPU_MCLK_FREQUENCY,
      RDC_FI_CPU_CCLK_LIMIT,           RDC_FI_CPU_SKT_ACTIVE_FREQ_LIMIT,
      RDC_FI_CPU_SKT_FREQ_LIMIT_SRC,   RDC_FI_CPU_SKT_FREQ_RANGE_MAX, RDC_FI_CPU_SKT_FREQ_RANGE_MIN,
      RDC_FI_CPU_SKT_C0_RESIDENCY,     RDC_FI_CPU_SKT_LCLK_DPM_LEVEL,
  };
  // clang-format on

  std::copy(fields.begin(), fields.end(), field_ids);
  *field_count = fields.size();

  return RDC_ST_OK;
}

rdc_status_t RdcSmiLib::rdc_diag_test_cases_query(rdc_diag_test_cases_t test_cases[MAX_TEST_CASES],
                                                  uint32_t* test_case_count) {
  if (test_case_count == nullptr) {
    return RDC_ST_BAD_PARAMETER;
  }

  const std::vector<rdc_diag_test_cases_t> tests{RDC_DIAG_COMPUTE_PROCESS, RDC_DIAG_NODE_TOPOLOGY,
                                                 RDC_DIAG_GPU_PARAMETERS, RDC_DIAG_POLL_LATENCY};
  std::copy(tests.begin(), tests.end(), test_cases);
  *test_case_count = tests.size();
  return RDC_ST_OK;
}

// Run a specific test case
rdc_status_t RdcSmiLib::rdc_test_case_run(rdc_diag_test_cases_t test_case,
                                          uint32_t gpu_index[RDC_MAX_NUM_DEVICES],
                                          uint32_t gpu_count, const char* config,
                                          size_t config_size, rdc_diag_test_result_t* result,
                                          rdc_diag_callback_t* callback) {
  if (result == nullptr) {
    return RDC_ST_BAD_PARAMETER;
  }
  switch (test_case) {
    case RDC_DIAG_COMPUTE_PROCESS:
      return smi_diag_->check_smi_process_info(gpu_index, gpu_count, result);
    case RDC_DIAG_NODE_TOPOLOGY:
      return smi_diag_->check_smi_topo_info(gpu_index, gpu_count, result);
    case RDC_DIAG_GPU_PARAMETERS:
      return smi_diag_->check_smi_param_info(gpu_index, gpu_count, result);
    case RDC_DIAG_POLL_LATENCY:
      return run_poll_latency(gpu_index, gpu_count, config, config_size, result);
    default:
      return RDC_ST_NOT_SUPPORTED;
  }
}

namespace {

// The fields the poll latency check reads: the clock, temperatures, power,
// utilization, VRAM usage and ECC error counts that GPU monitoring polls.
constexpr std::array<rdc_field_t, 9> kPollLatencyFields = {
    RDC_FI_GPU_CLOCK,         RDC_FI_MEMORY_TEMP,       RDC_FI_GPU_TEMP,
    RDC_FI_GPU_JUNCTION_TEMP, RDC_FI_POWER_USAGE,       RDC_FI_GPU_UTIL,
    RDC_FI_GPU_MEMORY_USAGE,  RDC_FI_ECC_CORRECT_TOTAL, RDC_FI_ECC_UNCORRECT_TOTAL};

rdc_status_t count_poll_values(rdc_gpu_field_value_t* values, uint32_t num_values,
                               void* user_data) {
  auto* counts = static_cast<PollValueCounts*>(user_data);
  for (uint32_t i = 0; i < num_values; i++) {
    const int status = values[i].field_value.status;
    if (status == AMDSMI_STATUS_SUCCESS) {
      counts->ok++;
    } else if (status == AMDSMI_STATUS_NOT_SUPPORTED ||
               status == AMDSMI_STATUS_NOT_YET_IMPLEMENTED) {
      counts->not_supported++;
    } else {
      counts->failed++;
    }
  }
  return RDC_ST_OK;
}

}  // namespace

rdc_status_t RdcSmiLib::run_poll_latency(uint32_t gpu_index[RDC_MAX_NUM_DEVICES],
                                         uint32_t gpu_count, const char* config, size_t config_size,
                                         rdc_diag_test_result_t* result) {
  PollLatencyConfig cfg;
  std::string error;
  if (parse_poll_latency_config(config, config_size, &cfg, &error) != RDC_ST_OK) {
    result->status = RDC_DIAG_RESULT_FAIL;
    result->test_case = RDC_DIAG_POLL_LATENCY;
    result->per_gpu_result_count = 0;
    result->details.code = 0;
    result->details.msg[0] = '\0';
    strncpy_with_null(result->info, error.c_str(), MAX_DIAG_MSG_LENGTH);
    return RDC_ST_BAD_PARAMETER;
  }

  // The check reads GPU fields, so it skips the CPUs in the group.
  std::vector<uint32_t> gpus;
  for (uint32_t i = 0; i < gpu_count && i < RDC_MAX_NUM_DEVICES; i++) {
    if (rdc_get_info_from_entity_index(gpu_index[i]).device_type == RDC_DEVICE_TYPE_GPU) {
      gpus.push_back(gpu_index[i]);
    }
  }

  auto read = [this](uint32_t gpu, uint64_t* elapsed_us, PollValueCounts* counts) {
    std::array<rdc_gpu_field_t, kPollLatencyFields.size()> fields;
    for (size_t i = 0; i < fields.size(); i++) {
      fields[i] = {gpu, kPollLatencyFields[i]};
    }
    std::lock_guard<std::mutex> guard(fetch_mutex_);
    const auto start = std::chrono::steady_clock::now();
    const rdc_status_t status = read_fields_locked(
        fields.data(), static_cast<uint32_t>(fields.size()), count_poll_values, counts);
    *elapsed_us = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
                                            std::chrono::steady_clock::now() - start)
                                            .count());
    return status;
  };
  return run_poll_latency_check(gpus.data(), static_cast<uint32_t>(gpus.size()),
                                static_cast<uint32_t>(kPollLatencyFields.size()), cfg, read,
                                real_poll_clock(), result);
}

rdc_status_t RdcSmiLib::rdc_diagnostic_run(const rdc_group_info_t&, rdc_diag_level_t, const char*,
                                           size_t, rdc_diag_response_t*, rdc_diag_callback_t*) {
  return RDC_ST_NOT_SUPPORTED;
}

rdc_status_t RdcSmiLib::rdc_diag_init(uint64_t) { return RDC_ST_OK; }

rdc_status_t RdcSmiLib::rdc_diag_destroy() { return RDC_ST_OK; }

}  // namespace rdc
}  // namespace amd
