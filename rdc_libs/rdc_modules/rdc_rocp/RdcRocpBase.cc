/*
Copyright (c) 2022 - present Advanced Micro Devices, Inc. All rights reserved.

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

#include "rdc_modules/rdc_rocp/RdcRocpBase.h"

#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <stdexcept>
#include <vector>

// #include "hsa.h"
#include "amd_smi/amdsmi.h"
#include "rdc/rdc.h"
#include "rdc_lib/RdcLogger.h"
#include "rdc_lib/RdcTelemetryLibInterface.h"
#include "rdc_lib/impl/SmiUtils.h"
#include "rdc_lib/rdc_common.h"
#include "rdc_modules/rdc_rocp/RdcRocpCounterSampler.h"

namespace amd {
namespace rdc {

static const std::map<rdc_field_t, const char*> temp_field_map_k = {
    {RDC_FI_PROF_OCCUPANCY_PERCENT, "OccupancyPercent"},
    {RDC_FI_PROF_ACTIVE_CYCLES, "GRBM_GUI_ACTIVE"},
    {RDC_FI_PROF_ACTIVE_WAVES, "SQ_WAVES"},
    {RDC_FI_PROF_ELAPSED_CYCLES, "GRBM_COUNT"},
    {RDC_FI_PROF_TENSOR_ACTIVE_PERCENT,
     "MfmaUtil"},  // same as TENSOR_ACTIVE but available for more GPUs
    {RDC_FI_PROF_GPU_UTIL_PERCENT, "GPU_UTIL"},  // metric is divided by 100 to get percent
    // metrics below are divided by time passed
    {RDC_FI_PROF_EVAL_MEM_R_BW, "FETCH_SIZE"},
    {RDC_FI_PROF_EVAL_MEM_W_BW, "WRITE_SIZE"},
    {RDC_FI_PROF_EVAL_FLOPS_16, "TOTAL_16_OPS"},
    {RDC_FI_PROF_EVAL_FLOPS_32, "TOTAL_32_OPS"},
    {RDC_FI_PROF_EVAL_FLOPS_64, "TOTAL_64_OPS"},
    {RDC_FI_PROF_EVAL_FLOPS_16_PERCENT, "RDC_OPS_16_PER_SIMDCYCLE"},
    {RDC_FI_PROF_EVAL_FLOPS_32_PERCENT, "RDC_OPS_32_PER_SIMDCYCLE"},
    {RDC_FI_PROF_EVAL_FLOPS_64_PERCENT, "RDC_OPS_64_PER_SIMDCYCLE"},
    // metrics below are not divided by time passed
    {RDC_FI_PROF_VALU_PIPE_ISSUE_UTIL, "ValuPipeIssueUtil"},
    {RDC_FI_PROF_SM_ACTIVE, "VALUBusy"},
    {RDC_FI_PROF_OCC_PER_ACTIVE_CU, "MeanOccupancyPerActiveCU"},
    {RDC_FI_PROF_OCC_ELAPSED,
     "GRBM_GUI_ACTIVE"},  // this metric is derived from OCC_PER_ACTIVE_CU and ACTIVE_CYCLES
    {RDC_FI_PROF_CPC_CPC_STAT_BUSY, "CPC_CPC_STAT_BUSY"},
    {RDC_FI_PROF_CPC_CPC_STAT_IDLE, "CPC_CPC_STAT_IDLE"},
    {RDC_FI_PROF_CPC_CPC_STAT_STALL, "CPC_CPC_STAT_STALL"},
    {RDC_FI_PROF_CPC_CPC_TCIU_BUSY, "CPC_CPC_TCIU_BUSY"},
    {RDC_FI_PROF_CPC_CPC_TCIU_IDLE, "CPC_CPC_TCIU_IDLE"},
    {RDC_FI_PROF_CPC_CPC_UTCL2IU_BUSY, "CPC_CPC_UTCL2IU_BUSY"},
    {RDC_FI_PROF_CPC_CPC_UTCL2IU_IDLE, "CPC_CPC_UTCL2IU_IDLE"},
    {RDC_FI_PROF_CPC_CPC_UTCL2IU_STALL, "CPC_CPC_UTCL2IU_STALL"},
    {RDC_FI_PROF_CPC_ME1_BUSY_FOR_PACKET_DECODE, "CPC_ME1_BUSY_FOR_PACKET_DECODE"},
    {RDC_FI_PROF_CPC_ME1_DC0_SPI_BUSY, "CPC_ME1_DC0_SPI_BUSY"},
    {RDC_FI_PROF_CPC_UTCL1_STALL_ON_TRANSLATION, "CPC_UTCL1_STALL_ON_TRANSLATION"},
    {RDC_FI_PROF_CPC_ALWAYS_COUNT, "CPC_ALWAYS_COUNT"},
    {RDC_FI_PROF_CPC_ADC_VALID_CHUNK_NOT_AVAIL, "CPC_ADC_VALID_CHUNK_NOT_AVAIL"},
    {RDC_FI_PROF_CPC_ADC_DISPATCH_ALLOC_DONE, "CPC_ADC_DISPATCH_ALLOC_DONE"},
    {RDC_FI_PROF_CPC_ADC_VALID_CHUNK_END, "CPC_ADC_VALID_CHUNK_END"},
    {RDC_FI_PROF_CPC_SYNC_FIFO_FULL_LEVEL, "CPC_SYNC_FIFO_FULL_LEVEL"},
    {RDC_FI_PROF_CPC_SYNC_FIFO_FULL, "CPC_SYNC_FIFO_FULL"},
    {RDC_FI_PROF_CPC_GD_BUSY, "CPC_GD_BUSY"},
    {RDC_FI_PROF_CPC_TG_SEND, "CPC_TG_SEND"},
    {RDC_FI_PROF_CPC_WALK_NEXT_CHUNK, "CPC_WALK_NEXT_CHUNK"},
    {RDC_FI_PROF_CPC_STALLED_BY_SE0_SPI, "CPC_STALLED_BY_SE0_SPI"},
    {RDC_FI_PROF_CPC_STALLED_BY_SE1_SPI, "CPC_STALLED_BY_SE1_SPI"},
    {RDC_FI_PROF_CPC_STALLED_BY_SE2_SPI, "CPC_STALLED_BY_SE2_SPI"},
    {RDC_FI_PROF_CPC_STALLED_BY_SE3_SPI, "CPC_STALLED_BY_SE3_SPI"},
    {RDC_FI_PROF_CPC_LTE_ALL, "CPC_LTE_ALL"},
    {RDC_FI_PROF_CPC_SYNC_WRREQ_FIFO_BUSY, "CPC_SYNC_WRREQ_FIFO_BUSY"},
    {RDC_FI_PROF_CPC_CANE_BUSY, "CPC_CANE_BUSY"},
    {RDC_FI_PROF_CPC_CANE_STALL, "CPC_CANE_STALL"},
    {RDC_FI_PROF_CPF_CMP_UTCL1_STALL_ON_TRANSLATION, "CPF_CMP_UTCL1_STALL_ON_TRANSLATION"},
    {RDC_FI_PROF_CPF_CPF_STAT_BUSY, "CPF_CPF_STAT_BUSY"},
    {RDC_FI_PROF_CPF_CPF_STAT_IDLE, "CPF_CPF_STAT_IDLE"},
    {RDC_FI_PROF_CPF_CPF_STAT_STALL, "CPF_CPF_STAT_STALL"},
    {RDC_FI_PROF_CPF_CPF_TCIU_BUSY, "CPF_CPF_TCIU_BUSY"},
    {RDC_FI_PROF_CPF_CPF_TCIU_IDLE, "CPF_CPF_TCIU_IDLE"},
    {RDC_FI_PROF_CPF_CPF_TCIU_STALL, "CPF_CPF_TCIU_STALL"},
    {RDC_FI_PROF_SIMD_UTILIZATION, "SIMD_UTILIZATION"},
    {RDC_FI_PROF_UUID, "SQ_WAVES"},    // dummy value,
    {RDC_FI_PROF_KFD_ID, "SQ_WAVES"},  // dummy value,
};

double RdcRocpBase::run_profiler(uint32_t agent_index, rdc_field_t field) {
  thread_local std::vector<rocprofiler_record_counter_t> records;

  auto counter_sampler = CounterSampler::get_samplers()[agent_index];
  if (!counter_sampler) {
    RDC_LOG(RDC_ERROR, "Error: Counter sampler not found for GPU index " << agent_index);
    return RDC_ST_BAD_PARAMETER;
  }

  auto field_it = field_to_metric.find(field);
  if (field_it == field_to_metric.end()) {
    RDC_LOG(RDC_ERROR, "Error: Field " << field << " not found in field_to_metric map.");
    return RDC_ST_BAD_PARAMETER;
  }
  const std::string& metric_id = field_it->second;

  try {
    counter_sampler->sample_counter_values({metric_id}, records, collection_duration_us_k);
  } catch (const std::exception& e) {
    RDC_LOG(RDC_ERROR, "Error while sampling counter values: " << e.what());
    return RDC_ST_BAD_PARAMETER;
  }

  // Aggregate counter values. Rocprof v1/v2 summed values across dimensions.
  double value = 0.0;
  for (auto& record : records) {
    value += record.counter_value;  // Summing up values from all dimensions.
  }

  return value;
}

const char* RdcRocpBase::get_field_id_from_name(rdc_field_t field) {
  auto it = field_to_metric.find(field);
  if (it == field_to_metric.end()) {
    RDC_LOG(RDC_ERROR, "Error: Field ID " << field << " not found in field_to_metric map.");
    return "";
  }

  return field_to_metric.at(field);
}

const std::vector<rdc_field_t> RdcRocpBase::get_field_ids() {
  std::vector<rdc_field_t> field_ids;
  for (auto& [k, v] : field_to_metric) {
    field_ids.push_back(k);
  }
  return field_ids;
}

rocprofiler_uuid_t asic_serial_to_uuid(const char* asic_serial) {
  rocprofiler_uuid_t uuid = {0};
  // have to cast to stoull as a workaround for amdsmi ignoring leading zeroes
  uuid.value = std::stoull(asic_serial, nullptr, 16);
  return uuid;
}

std::string uuid_to_string(const uint64_t uuid) {
  std::ostringstream oss;
  oss << "0x" << std::hex << std::setw(16) << std::setfill('0') << uuid;
  return oss.str();
}

std::string uuid_to_string(const rocprofiler_uuid_t& uuid) { return uuid_to_string(uuid.value); }

rdc_status_t RdcRocpBase::map_entity_to_profiler() {
  // std::map<uint32_t, uint32_t> entity_to_index_map;
  // kfd_id_t is only used inside this function
  typedef uint64_t kfd_id_t;
  std::map<uint32_t, kfd_id_t> prof_kfd_map;

  // populate profiler map
  for (uint32_t prof_gpu_index = 0; prof_gpu_index < agents.size(); prof_gpu_index++) {
    prof_kfd_map.insert({prof_gpu_index, agents[prof_gpu_index].gpu_id});
  }

  std::vector<amdsmi_socket_handle> sockets;
  auto amdsmi_status = get_socket_handles(sockets);
  if (amdsmi_status != AMDSMI_STATUS_SUCCESS) {
    RDC_LOG(RDC_ERROR, "Failed to get socket handles: " << amdsmi_status);
    return Smi2RdcError(amdsmi_status);
  }

  for (int socket_index = 0; socket_index < sockets.size(); socket_index++) {
    auto* socket = sockets[socket_index];
    std::vector<amdsmi_processor_handle> processors;
    amdsmi_status = get_processor_handles(socket, processors);
    if (amdsmi_status != AMDSMI_STATUS_SUCCESS) {
      RDC_LOG(RDC_ERROR, "Failed to get processor handles for socket " << socket_index << ": "
                                                                       << amdsmi_status);
      return Smi2RdcError(amdsmi_status);
    }

    for (int processor_index = 0; processor_index < processors.size(); processor_index++) {
      auto* processor = processors[processor_index];
      processor_type_t processor_type = AMDSMI_PROCESSOR_TYPE_UNKNOWN;
      amdsmi_status = amdsmi_get_processor_type(processor, &processor_type);
      if (amdsmi_status != AMDSMI_STATUS_SUCCESS) {
        RDC_LOG(RDC_ERROR, "Failed to get processor type for processor "
                               << processor_index << " on socket " << socket_index << ": "
                               << amdsmi_status);
        return Smi2RdcError(amdsmi_status);
      }
      if (processor_type != AMDSMI_PROCESSOR_TYPE_AMD_GPU) {
        continue;
      }

      amdsmi_kfd_info_t kfd_info;
      amdsmi_status = amdsmi_get_gpu_kfd_info(processor, &kfd_info);
      if (amdsmi_status != AMDSMI_STATUS_SUCCESS) {
        RDC_LOG(RDC_ERROR, "Failed to get KFD info for processor "
                               << processor_index << " on socket " << socket_index << ": "
                               << amdsmi_status);
        return Smi2RdcError(amdsmi_status);
      }

      rdc_entity_info_t entity_info = {
          .device_index = static_cast<uint32_t>(socket_index),
          .instance_index = static_cast<uint32_t>(processor_index),
          .entity_role = RDC_DEVICE_ROLE_PHYSICAL,
          .device_type = RDC_DEVICE_TYPE_GPU,
      };

      uint32_t entity_index = rdc_get_entity_index_from_info(entity_info);

      for (const auto& [prof_index, prof_id] : prof_kfd_map) {
        if (std::memcmp(&kfd_info.kfd_id, &prof_id, sizeof(kfd_id_t)) == 0) {
          // match found
          // clang-format off
          RDC_LOG(RDC_DEBUG, "SMI[" << entity_index << "] <-> Profiler[" << prof_index << "] = KFD_ID[" << prof_id << "]");
          // clang-format on
          if (entity_info.entity_role == RDC_DEVICE_ROLE_PHYSICAL) {
            entity_index = rdc_get_entity_index_from_info(entity_info);
            entity_to_prof_map.insert({entity_index, prof_index});
          }
          if (processors.size() > 1) {
            // if there are multiple processors, also add entity with partition instance type
            entity_info.entity_role = RDC_DEVICE_ROLE_PARTITION_INSTANCE;
            entity_index = rdc_get_entity_index_from_info(entity_info);
            entity_to_prof_map.insert({entity_index, prof_index});
          }
          break;
        }
      }
    }
  }
  return RDC_ST_OK;
}

void RdcRocpBase::init_rocp_if_not() {
  if (m_is_initialized) {
    return;
  }

  // ensure initialization is attempted only once, even if it fails
  m_is_initialized = true;

  hsa_status_t status = hsa_init();
  if (status != HSA_STATUS_SUCCESS) {
    const char* errstr = nullptr;
    hsa_status_string(status, &errstr);
    throw std::runtime_error("hsa error code: " + std::to_string(status) + " " + errstr);
  }

  // check rocprofiler
  if (int rocp_status = 0;
      rocprofiler_is_initialized(&rocp_status) == ROCPROFILER_STATUS_SUCCESS && rocp_status != 1) {
    throw std::runtime_error("Rocprofiler is not initialized. status: " +
                             std::to_string(rocp_status));
  }

  std::vector<std::string> all_fields;
  std::vector<std::string> checked_fields;

  // populate list of agents
  agents = CounterSampler::get_available_agents();
  RDC_LOG(RDC_DEBUG, "Agent count: " << agents.size());
  samplers = CounterSampler::get_samplers();

  map_entity_to_profiler();

  // find intersection of supported and requested fields
  uint32_t agent_index = 0;
  auto& cs = *samplers[agent_index];
  RDC_LOG(RDC_DEBUG, "agent_index[" << agent_index << "] location_id["
                                    << agents[agent_index].location_id << "]");

  for (auto& [str, id] : CounterSampler::get_supported_counters(cs.get_agent())) {
    checked_fields.emplace_back(str);
  }

  for (const auto& [k, v] : temp_field_map_k) {
    auto found = std::find(checked_fields.begin(), checked_fields.end(), v);
    if (found != checked_fields.end()) {
      field_to_metric.insert({k, v});
    }
  }

  // populate fields
  for (const auto& [k, v] : temp_field_map_k) {
    all_fields.emplace_back(v);
  }

  RDC_LOG(RDC_DEBUG, "Profiler supports " << field_to_metric.size() << " fields");
}

RdcRocpBase::RdcRocpBase() {
  // To verify if a field is actually supported by rocprofiler,
  // initialization and agent querying are required.
  // This initialization is deferred until the first call to rocp_lookup.
  // Here, we define the potential fields that rocprofiler may support,
  // allowing get_field_ids() to return them.
  for (const auto& [k, v] : temp_field_map_k) {
    field_to_metric.insert({k, v});
  }

  RDC_LOG(RDC_DEBUG, "Rocprofiler by default supports " << field_to_metric.size() << " fields");
}

RdcRocpBase::~RdcRocpBase() {
  hsa_status_t status = HSA_STATUS_SUCCESS;
  status = hsa_shut_down();
  assert(status == HSA_STATUS_SUCCESS);
  status = hsa_shut_down();
  assert(status == HSA_STATUS_ERROR_NOT_INITIALIZED);
}

rdc_status_t RdcRocpBase::rocp_lookup(rdc_gpu_field_t gpu_field, rdc_field_value_data* data,
                                      rdc_field_type_t* type) {
  // default type
  *type = DOUBLE;

  // convert from entity to flat index
  uint32_t agent_index = entity_to_prof_map[gpu_field.gpu_index];
  const auto& field = gpu_field.field_id;

  if (data == nullptr) {
    return RDC_ST_BAD_PARAMETER;
  }

  init_rocp_if_not();

  const bool is_eval_field = (eval_fields.find(field) != eval_fields.end());

  const auto start_time = std::chrono::high_resolution_clock::now();
  // direct read from rocprofiler
  const double read_dbl = run_profiler(agent_index, field);
  const auto stop_time = std::chrono::high_resolution_clock::now();
  const double elapsed = std::chrono::duration<double, std::milli>(stop_time - start_time).count();
  // divide by elapsed time if needed
  double divided_dbl = NAN;

  if (is_eval_field) {
    if (elapsed != 0.0) {
      divided_dbl = read_dbl / (elapsed / 1000.0);
    } else {
      RDC_LOG(RDC_ERROR, "Error: Elapsed time is zero. Cannot divide by zero.");
      return RDC_ST_BAD_PARAMETER;
    }
  }

  switch (field) {
    case RDC_FI_PROF_GPU_UTIL_PERCENT:
      // RDC_FI_PROF_GPU_UTIL_PERCENT is mapped to GPU_UTIL
      // GPU_UTIL metric is available on more GPUs than ENGINE_ACTIVE.
      // ENGINE_ACTIVE = GPU_UTIL/100, so do the math ourselves
      data->dbl = read_dbl / 100.0F;
      break;
    case RDC_FI_PROF_OCC_ELAPSED: {
      // RDC_FI_PROF_OCC_ELAPSED is mapped to GRBM_GUI_ACTIVE, the read happens earlier in this
      // function
      const double active_cycles_val = read_dbl;
      if (active_cycles_val != 0.0) {
        // read second value from profiler
        const double occupancy_val = run_profiler(agent_index, RDC_FI_PROF_OCC_PER_ACTIVE_CU);
        data->dbl = occupancy_val / active_cycles_val;
      } else {
        return RDC_ST_BAD_PARAMETER;
      }
    } break;
    case RDC_FI_PROF_EVAL_FLOPS_16_PERCENT: {
      if (!is_eval_field) {
        RDC_LOG(RDC_ERROR, "Field expected to be in the eval_fields list but it isn't!");
        return RDC_ST_BAD_PARAMETER;
      }
      // 1024, 2048, and 256 are taken from "INTRODUCING AMD CDNA 3 ARCHITECTURE" white paper
      const std::string target_version = agents[agent_index].name;
      // TODO: Design a lookup table for other GPUs
      const bool isMI200 = (target_version.find("gfx90a") != std::string::npos);
      // FLOPS/clock/CU
      if (isMI200) {
        data->dbl = divided_dbl / (1024.0F / static_cast<double>(agents[agent_index].simd_per_cu));
      } else {  // Assume mi300
        data->dbl = divided_dbl / (2048.0F / static_cast<double>(agents[agent_index].simd_per_cu));
      }
    } break;
    case RDC_FI_PROF_EVAL_FLOPS_32_PERCENT:
    case RDC_FI_PROF_EVAL_FLOPS_64_PERCENT:
      if (!is_eval_field) {
        RDC_LOG(RDC_ERROR, "Field expected to be in the eval_fields list but it isn't!");
        return RDC_ST_BAD_PARAMETER;
      }
      // FLOPS/clock/CU
      data->dbl = divided_dbl / (256.0F / static_cast<double>(agents[agent_index].simd_per_cu));
      break;
    case RDC_FI_PROF_UUID: {
      // do not care what RDC_FI_PROF_UUID is mapped to. read value from agents
      *type = STRING;
      std::string uuid_str = uuid_to_string(agents[agent_index].uuid);
      strncpy_with_null(data->str, uuid_str.c_str(), uuid_str.length());
      break;
    }
    case RDC_FI_PROF_KFD_ID: {
      // do not care what RDC_FI_PROF_UUID is mapped to. read value from agents
      *type = INTEGER;
      data->l_int = agents[agent_index].gpu_id;
      break;
    }
    default:
      // only support default fallback for doubles
      assert(*type == DOUBLE);
      if (is_eval_field) {
        data->dbl = divided_dbl;
      } else {
        data->dbl = read_dbl;
      }
      break;
  }

  return RDC_ST_OK;
}

}  // namespace rdc
}  // namespace amd
