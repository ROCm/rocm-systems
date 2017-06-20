// MIT License
//
// Copyright (c) 2017-2025 Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "core/aql_profile.hpp"
#include "core/include/aql_profile_v2.h"

#include <cstdint>
#include <future>
#include <map>
#include <string>
#include <vector>
#include <mutex>

#include "core/counter_dimensions.hpp"

#include "core/logger.h"
#include "core/pm4_factory.h"
#include "pm4/cmd_builder.h"
#include "pm4/pmc_builder.h"
#include "pm4/spm_builder.h"
#include "pm4/sqtt_builder.h"

#include "core/commandbuffermgr.hpp"

#define CONSTRUCTOR_API __attribute__((constructor))
#define DESTRUCTOR_API __attribute__((destructor))
#define ERR_CHECK(cond, err, msg) \
  {                               \
    if (cond) {                   \
      ERR_LOGGING << msg;         \
      return err;                 \
    }                             \
  }

// Getting SPM data using driver API
namespace spm_kfd_namespace {
hsa_status_t spm_iterate_data(const hsa_ven_amd_aqlprofile_profile_t* profile,
                              hsa_ven_amd_aqlprofile_data_callback_t callback, void* data);
}

// PC sampling callback data
struct pcsmp_callback_data_t {
  const char* kernel_name;  // sampled kernel name
  void* data_buffer;        // host buffer for tracing data
  uint64_t id;              // sample id
  uint64_t cycle;           // sample cycle
  uint64_t pc;              // sample PC
};

std::atomic<int> ATT_TARGET_CU{0};

namespace aql_profile {
// Command buffer partitioning manager
// Supports Pre/Post commands partitioning
// and prefix control partition

static std::unordered_map<void*, pm4_builder::TraceConfig> configs;
static std::mutex config_mut;

static inline pm4_builder::counters_vector CountersVec(const profile_t* profile,
                                                       const Pm4Factory* pm4_factory) {
  pm4_builder::counters_vector vec;
  std::map<block_des_t, uint32_t, lt_block_des> index_map;
  for (const hsa_ven_amd_aqlprofile_event_t* p = profile->events;
       p < profile->events + profile->event_count; ++p) {
    const GpuBlockInfo* block_info = pm4_factory->GetBlockInfo(p);
    const block_des_t block_des = {pm4_factory->GetBlockInfo(p)->id, p->block_index};
    // Counting counter register index per block
    const auto ret = index_map.insert({block_des, 0});
    uint32_t& reg_index = ret.first->second;

    if (pm4_builder::SPISkip(block_info->attr, p->counter_id)) {
      vec.push_back({p->counter_id, reg_index, block_des, block_info});
      continue;
    }

    if (reg_index >= block_info->counter_count) {
      throw event_exception("Event is out of block counter registers number limit, ", *p);
    }

    vec.push_back({p->counter_id, reg_index, block_des, block_info});

    ++reg_index;
  }

  if (pm4_factory->IsGFX10() && (vec.get_attr() & CounterBlockGRBMAttr) == 0 && !vec.empty()) {
    event_t grbm_event{
        .block_name = HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GRBM, .block_index = 0, .counter_id = 0};
    const GpuBlockInfo* block_info = pm4_factory->GetBlockInfo(&grbm_event);
    if (block_info == nullptr) return vec;
    const block_des_t block_des = {block_info->id, 0};
    const auto ret = index_map.insert({block_des, 0});
    uint32_t& reg_index = ret.first->second;
    vec.push_back({0, reg_index, block_des, block_info});
    reg_index++;
  }
  return vec;
}

static inline bool IsEventMatch(const event_t& event1, const event_t& event2) {
  return (event1.block_name == event2.block_name) && (event1.block_index == event2.block_index) &&
         (event1.counter_id == event2.counter_id);
}

hsa_status_t DefaultPmcdataCallback(hsa_ven_amd_aqlprofile_info_type_t info_type,
                                    hsa_ven_amd_aqlprofile_info_data_t* info_data,
                                    void* callback_data) {
  hsa_status_t status = HSA_STATUS_SUCCESS;
  hsa_ven_amd_aqlprofile_info_data_t* passed_data =
      reinterpret_cast<hsa_ven_amd_aqlprofile_info_data_t*>(callback_data);

  if (info_type == HSA_VEN_AMD_AQLPROFILE_INFO_PMC_DATA) {
    if (IsEventMatch(info_data->pmc_data.event, passed_data->pmc_data.event)) {
      if (passed_data->sample_id == UINT32_MAX) {
        passed_data->pmc_data.result += info_data->pmc_data.result;
      } else if (passed_data->sample_id == info_data->sample_id) {
        passed_data->pmc_data.result = info_data->pmc_data.result;
        status = HSA_STATUS_INFO_BREAK;
      }
    }
  }

  return status;
}

hsa_status_t DefaultTracedataCallback(hsa_ven_amd_aqlprofile_info_type_t info_type,
                                      hsa_ven_amd_aqlprofile_info_data_t* info_data,
                                      void* callback_data) {
  hsa_status_t status = HSA_STATUS_SUCCESS;
  hsa_ven_amd_aqlprofile_info_data_t* passed_data =
      reinterpret_cast<hsa_ven_amd_aqlprofile_info_data_t*>(callback_data);

  if (info_type == HSA_VEN_AMD_AQLPROFILE_INFO_TRACE_DATA) {
    if (info_data->sample_id == passed_data->sample_id) {
      passed_data->trace_data = info_data->trace_data;
      status = HSA_STATUS_INFO_BREAK;
    }
  }

  return status;
}

Logger::mutex_t Logger::mutex_;
Logger* Logger::instance_ = NULL;
bool Pm4Factory::concurrent_create_mode_ = false;
bool Pm4Factory::spm_kfd_mode_ = false;
Pm4Factory::mutex_t Pm4Factory::mutex_;
Pm4Factory::instances_t* Pm4Factory::instances_ = NULL;
bool read_api_enabled = true;

CONSTRUCTOR_API void constructor() {
  const char* read_api_enabled_str = getenv("AQLPROFILE_READ_API");
  if (read_api_enabled_str != NULL) {
    if (atoi(read_api_enabled_str) == 0) read_api_enabled = false;
  }
}

DESTRUCTOR_API void destructor() {
  Logger::Destroy();
  Pm4Factory::Destroy();
}

}  // namespace aql_profile

extern "C" {

// Return library major/minor version
PUBLIC_API uint32_t hsa_ven_amd_aqlprofile_version_major() { return HSA_AQLPROFILE_VERSION_MAJOR; }
PUBLIC_API uint32_t hsa_ven_amd_aqlprofile_version_minor() { return HSA_AQLPROFILE_VERSION_MINOR; }

// Returns the last error message
PUBLIC_API hsa_status_t hsa_ven_amd_aqlprofile_error_string(const char** str) {
  *str = aql_profile::Logger::LastMessage().c_str();
  return HSA_STATUS_SUCCESS;
}

// Check if event is valid for the specific GPU
PUBLIC_API hsa_status_t hsa_ven_amd_aqlprofile_validate_event(
    hsa_agent_t agent, const hsa_ven_amd_aqlprofile_event_t* event, bool* result) {
  hsa_status_t status = HSA_STATUS_SUCCESS;
  *result = false;

  try {
    aql_profile::Pm4Factory* pm4_factory = aql_profile::Pm4Factory::Create(agent);
    if (pm4_factory->GetBlockInfo(event) != NULL) *result = true;
  } catch (aql_profile::event_exception& e) {
    INFO_LOGGING << e.what();
  } catch (std::exception& e) {
    ERR_LOGGING << e.what();
    status = HSA_STATUS_ERROR;
  }

  return status;
}

// Method to populate the provided AQL packet with profiling start commands
PUBLIC_API hsa_status_t hsa_ven_amd_aqlprofile_start(hsa_ven_amd_aqlprofile_profile_t* profile,
                                                     aql_profile::packet_t* aql_start_packet) {
  try {
    pm4_builder::CmdBuffer commands;
    aql_profile::CommandBufferMgr cmd_buffer_mgr(profile->command_buffer.ptr, UINT_MAX);

    aql_profile::Pm4Factory* pm4_factory = aql_profile::Pm4Factory::Create(profile);
    const bool is_concurrent = pm4_factory->IsConcurrent();
    const pm4_builder::counters_vector countersVec = CountersVec(profile, pm4_factory);

    if (profile->type == HSA_VEN_AMD_AQLPROFILE_EVENT_TYPE_PMC) {
      pm4_builder::PmcBuilder* pmc_builder = pm4_factory->GetPmcBuilder();

      // Generate read commands
      auto data_size = pmc_builder->Read(&commands, countersVec, profile->output_buffer.ptr);
      if (!aql_profile::read_api_enabled) commands.Clear();
      cmd_buffer_mgr.SetRdSize(commands.Size());

      // Copy generated read commands
      if (profile->command_buffer.ptr != NULL) {
        const aql_profile::descriptor_t rd_descr = cmd_buffer_mgr.GetRdDescr();
        memcpy(rd_descr.ptr, commands.Data(), commands.Size());
        commands.Clear();
      }

      // Generate start commands
      pmc_builder->Start(&commands, countersVec);
      cmd_buffer_mgr.SetPreSize(commands.Size());

      // Generate stop commands
      if (!aql_profile::read_api_enabled)
        pmc_builder->Read(&commands, countersVec, profile->output_buffer.ptr);
      pmc_builder->Stop(&commands, countersVec);

      if (profile->output_buffer.size < data_size) {
        profile->output_buffer.size = data_size;
        if (profile->output_buffer.ptr != NULL) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
      }
    } else if (profile->type == HSA_VEN_AMD_AQLPROFILE_EVENT_TYPE_TRACE) {
      pm4_builder::TraceConfig trace_config{};
      const uint64_t se_number_total = pm4_factory->GetShaderEnginesNumber();

      if (profile->parameters) {
        for (const hsa_ven_amd_aqlprofile_parameter_t* p = profile->parameters;
             p < (profile->parameters + profile->parameter_count); ++p) {
          switch (p->parameter_name) {
            case HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_SE_MASK:
              trace_config.se_mask = p->value & ((1ull << se_number_total) - 1);
              break;
            case HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_COMPUTE_UNIT_TARGET:
              if (p->value > 15)
                throw aql_profile::aql_profile_exc_val<uint32_t>(
                    "ThreadTraceConfig: CuId must be between 0 and 15, TargetCu", p->value);
              trace_config.targetCu = p->value;
              ATT_TARGET_CU.store(p->value);
              break;
            case HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_VM_ID_MASK:
              trace_config.vmIdMask = p->value;
              break;
            case HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_MASK:
              if ((p->value & 0x50) != 0)
                throw aql_profile::aql_profile_exc_val<uint32_t>(
                    "ThreadTraceConfig: Mask should have bits [4,6] set to Zero, Mask", p->value);
              trace_config.deprecated_mask = p->value;
              trace_config.targetCu = p->value & 0xF;
              ATT_TARGET_CU.store(trace_config.targetCu);
              break;
            case HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_TOKEN_MASK:
              if ((p->value & 0xFF000000) != 0)
                throw aql_profile::aql_profile_exc_val<uint32_t>(
                    "ThreadTraceConfig: TokenMask should have bits [31:25] set to Zero, TokenMask",
                    p->value);
              trace_config.deprecated_tokenMask = p->value;
              break;
            case HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_TOKEN_MASK2:
              trace_config.deprecated_tokenMask2 = p->value;
              break;
            case HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_SAMPLE_RATE:
              trace_config.sampleRate = p->value;
              break;
            case HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_K_CONCURRENT:
              trace_config.concurrent = p->value;
              break;
            case HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_SIMD_SELECTION:
              trace_config.simd_sel = p->value & 0xF;
              break;
            case HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_OCCUPANCY_MODE:
              trace_config.occupancy_mode = p->value ? 1 : 0;
              break;
            case HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_ATT_BUFFER_SIZE:
              break;
            case HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_PERFCOUNTER_MASK:
              trace_config.perfMASK = p->value;
              break;
            case HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_PERFCOUNTER_CTRL:
              trace_config.perfCTRL = ((p->value & 0x1F) << 8) | 0xFFFF007F;
              break;
            case HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_PERFCOUNTER_NAME:
              if (trace_config.perfcounters.size() < 8)
                trace_config.perfcounters.push_back({p->value, 0xF});
              break;
            default:
              ERR_LOGGING << "Bad trace parameter name (" << p->parameter_name << ")";
              return HSA_STATUS_ERROR_INVALID_ARGUMENT;
          }
        }
      }

      auto control_size = sizeof(pm4_builder::TraceControl) * se_number_total;
      char* prefix_ptr = cmd_buffer_mgr.AddPrefix(control_size);
      auto* control_ptr = reinterpret_cast<pm4_builder::TraceControl*>(prefix_ptr);

      trace_config.control_buffer_ptr = control_ptr;
      trace_config.control_buffer_size = control_size;
      trace_config.data_buffer_ptr = profile->output_buffer.ptr;
      trace_config.data_buffer_size = profile->output_buffer.size;

      if (countersVec.size() == 0) {
        pm4_builder::SqttBuilder* sqtt_builder = pm4_factory->GetSqttBuilder();
        // Generate start commands
        sqtt_builder->Begin(&commands, &trace_config);
        cmd_buffer_mgr.SetPreSize(commands.Size());
        // Generate stop commands
        sqtt_builder->End(&commands, &trace_config);
      } else {
        const char* sz_sampling_rate = getenv("AQLPROFILE_SPM_SAMPLE_RATE");
        if (sz_sampling_rate != NULL) trace_config.sampleRate = atoi(sz_sampling_rate);

        pm4_builder::SpmBuilder* spm_builder = pm4_factory->GetSpmBuilder();
        // Generate start commands
        spm_builder->Begin(&commands, &trace_config, countersVec);
        cmd_buffer_mgr.SetPreSize(commands.Size());
        // Generate stop commands
        spm_builder->End(&commands, &trace_config);
      }
      aql_profile::configs[profile->command_buffer.ptr] = trace_config;
    } else {
      ERR_LOGGING << "Bad profile type (" << profile->type << ")";
      return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    }

    cmd_buffer_mgr.Finalize(commands.Size());
    const uint32_t cmd_size = (cmd_buffer_mgr.GetSize() + 0x1800) & ~0xFFF;
    if (profile->command_buffer.size < cmd_size) {
      profile->command_buffer.size = cmd_size;
      if (profile->command_buffer.ptr != NULL) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    }
    if (profile->command_buffer.ptr != NULL) {
      // Copy generated commands
      const aql_profile::descriptor_t pre_descr = cmd_buffer_mgr.GetPreDescr();
      const aql_profile::descriptor_t post_descr = cmd_buffer_mgr.GetPostDescr();
      memcpy(pre_descr.ptr, commands.Data(), pre_descr.size);
      memcpy(post_descr.ptr, reinterpret_cast<const char*>(commands.Data()) + pre_descr.size,
             post_descr.size);
      // Populate start aql packet
      pm4_builder::CmdBuilder* cmd_writer = pm4_factory->GetCmdBuilder();
      aql_profile::PopulateAql(pre_descr.ptr, pre_descr.size, cmd_writer, aql_start_packet);
    }
  } catch (std::exception& e) {
    ERR_LOGGING << e.what();
    return HSA_STATUS_ERROR;
  }

  return HSA_STATUS_SUCCESS;
}

// Method to populate the provided AQL packet with profiling stop commands
PUBLIC_API hsa_status_t hsa_ven_amd_aqlprofile_stop(const hsa_ven_amd_aqlprofile_profile_t* profile,
                                                    aql_profile::packet_t* aql_stop_packet) {
  try {
    // Populate stop aql packet
    aql_profile::Pm4Factory* pm4_factory = aql_profile::Pm4Factory::Create(profile);
    pm4_builder::CmdBuilder* cmd_writer = pm4_factory->GetCmdBuilder();
    aql_profile::CommandBufferMgr cmd_buffer_mgr(profile);
    const aql_profile::descriptor_t post_descr = cmd_buffer_mgr.GetPostDescr();
    aql_profile::PopulateAql(post_descr.ptr, post_descr.size, cmd_writer, aql_stop_packet);
  } catch (std::exception& e) {
    ERR_LOGGING << e.what();
    return HSA_STATUS_ERROR;
  }

  return HSA_STATUS_SUCCESS;
}

// Method to populate the provided AQL packet with profiling read commands
PUBLIC_API hsa_status_t hsa_ven_amd_aqlprofile_read(const hsa_ven_amd_aqlprofile_profile_t* profile,
                                                    aql_profile::packet_t* aql_read_packet) {
  if (!aql_profile::read_api_enabled) return HSA_STATUS_ERROR;
  try {
    // Populate read aql packet
    aql_profile::Pm4Factory* pm4_factory = aql_profile::Pm4Factory::Create(profile);
    const bool is_concurrent = pm4_factory->IsConcurrent();
    pm4_builder::CmdBuilder* cmd_writer = pm4_factory->GetCmdBuilder();
    aql_profile::CommandBufferMgr cmd_buffer_mgr(profile);

    const aql_profile::descriptor_t rd_descr =
        (is_concurrent == false) ? cmd_buffer_mgr.GetRdDescr() : cmd_buffer_mgr.FetchRdDescr();
    aql_profile::PopulateAql(rd_descr.ptr, rd_descr.size, cmd_writer, aql_read_packet);
  } catch (std::exception& e) {
    ERR_LOGGING << e.what();
    return HSA_STATUS_ERROR;
  }
  return HSA_STATUS_SUCCESS;
}

// Legacy devices, converting of the profiling AQL packet to PM4 packet blob
PUBLIC_API hsa_status_t
hsa_ven_amd_aqlprofile_legacy_get_pm4(const aql_profile::packet_t* aql_packet, void* data) {
  return HSA_STATUS_ERROR;
}

// Method for getting the profile info
PUBLIC_API hsa_status_t
hsa_ven_amd_aqlprofile_get_info(const hsa_ven_amd_aqlprofile_profile_t* profile,
                                hsa_ven_amd_aqlprofile_info_type_t attribute, void* value) {
  hsa_status_t status = HSA_STATUS_SUCCESS;

  const uint32_t attr_op = (uint32_t)attribute;
  const uint32_t begin_op = (uint32_t)HSA_VEN_AMD_AQLPROFILE_INFO_ENABLE_CMD;
  if (attr_op >= begin_op) attribute = (hsa_ven_amd_aqlprofile_info_type_t)begin_op;

  if (profile == NULL) {
    ERR_LOGGING << "NULL argument 'profile'";
    return HSA_STATUS_ERROR;
  }
  if (attribute != HSA_VEN_AMD_AQLPROFILE_INFO_ENABLE_CMD) {
    if (value == NULL) {
      ERR_LOGGING << "NULL argument 'value'";
      return HSA_STATUS_ERROR;
    }
  }

  try {
    aql_profile::Pm4Factory* pm4_factory = aql_profile::Pm4Factory::Create(profile);
    switch (attribute) {
      case HSA_VEN_AMD_AQLPROFILE_INFO_COMMAND_BUFFER_SIZE:
        *(uint32_t*)value = 0x2000;  // a current approximation as 4K is big enough
        break;
      case HSA_VEN_AMD_AQLPROFILE_INFO_PMC_DATA_SIZE:
        *(uint32_t*)value = 0x1800;  // a current approximation as 4K is big enough
        break;
      case HSA_VEN_AMD_AQLPROFILE_INFO_PMC_DATA:
        reinterpret_cast<hsa_ven_amd_aqlprofile_info_data_t*>(value)->pmc_data.result = 0;
        status = hsa_ven_amd_aqlprofile_iterate_data(profile, aql_profile::DefaultPmcdataCallback,
                                                     value);
        break;
      case HSA_VEN_AMD_AQLPROFILE_INFO_TRACE_DATA:
        status = hsa_ven_amd_aqlprofile_iterate_data(profile, aql_profile::DefaultTracedataCallback,
                                                     value);
        break;
      case HSA_VEN_AMD_AQLPROFILE_INFO_BLOCK_COUNTERS:
        *reinterpret_cast<uint32_t*>(value) =
            pm4_factory->GetBlockInfo(&(profile->events[0]))->counter_count;
        break;
      case HSA_VEN_AMD_AQLPROFILE_INFO_BLOCK_ID: {
        hsa_ven_amd_aqlprofile_id_query_t* query =
            reinterpret_cast<hsa_ven_amd_aqlprofile_id_query_t*>(value);
        const uint32_t block = pm4_factory->FindBlock(query->name);
        const GpuBlockInfo* info = pm4_factory->GetBlockInfo(block);
        status = (info == NULL) ? HSA_STATUS_ERROR : HSA_STATUS_SUCCESS;
        if (status == HSA_STATUS_SUCCESS) {
          query->id = block;
          query->instance_count = info->instance_count;
        }
        break;
      }
      case HSA_VEN_AMD_AQLPROFILE_INFO_ENABLE_CMD: {
        aql_profile::Pm4Factory* pm4_factory = aql_profile::Pm4Factory::Create(profile);
        pm4_builder::PmcBuilder* pmc_builder = pm4_factory->GetPmcBuilder();
        pm4_builder::CmdBuilder* cmd_writer = pm4_factory->GetCmdBuilder();
        pm4_builder::CmdBuffer commands;

        const uint32_t op = attr_op - begin_op;
        switch (op) {
          case 0:
            pmc_builder->Enable(&commands);
            break;
          case 1:
            pmc_builder->Disable(&commands);
            break;
          case 2:
            pmc_builder->WaitIdle(&commands);
            break;
          default:
            ERR_LOGGING << "get_info, not supported op (" << op << ")";
            status = HSA_STATUS_ERROR;
        }

        if (profile->command_buffer.ptr == NULL) {
          const_cast<hsa_ven_amd_aqlprofile_profile_t*>(profile)->command_buffer.size =
              commands.Size();
          break;
        }

        if (profile->command_buffer.size != commands.Size()) {
          ERR_LOGGING << "get_info, wrong profile cmd size";
          status = HSA_STATUS_ERROR;
          break;
        }
        if (value == NULL) {
          ERR_LOGGING << "NULL argument 'value'";
          status = HSA_STATUS_ERROR;
          break;
        }

        memcpy(profile->command_buffer.ptr, commands.Data(), profile->command_buffer.size);
        aql_profile::PopulateAql(profile->command_buffer.ptr, profile->command_buffer.size,
                                 cmd_writer, reinterpret_cast<aql_profile::packet_t*>(value));

        break;
      }
      default:
        status = HSA_STATUS_ERROR_INVALID_ARGUMENT;
        ERR_LOGGING << "Invalid attribute (" << attribute << ")";
    }
  } catch (std::exception& e) {
    ERR_LOGGING << e.what();
    return HSA_STATUS_ERROR;
  }

  return status;
}

PUBLIC_API hsa_status_t
hsa_ven_amd_aqlprofile_iterate_event_ids(hsa_ven_amd_aqlprofile_eventname_callback_t callback) {
  try {
    EventDimension::init();
    for (auto& [name, id] : EventDimension::dimension_table) callback(id, name.c_str());
  } catch (...) {
    return HSA_STATUS_ERROR;
  }
  return HSA_STATUS_SUCCESS;
}

PUBLIC_API hsa_status_t hsa_ven_amd_aqlprofile_iterate_event_coord(
    hsa_agent_t agent, hsa_ven_amd_aqlprofile_event_t event, uint32_t sample_id,
    hsa_ven_amd_aqlprofile_coordinate_callback_t callback, void* userdata) {
  try {
    const EventAttribDimension& attrib = EventAttribDimension::get(agent, event.block_name);

    if (!attrib.get_num()) return HSA_STATUS_ERROR;

    std::vector<uint8_t> coord;
    coord.resize(attrib.get_num());
    attrib.get_coordinates(coord.data(),
                           sample_id * attrib.get_num_instances() + event.block_index);

    for (size_t i = 0; i < attrib.get_num(); i++) {
      EventDimension dim = attrib.get_dim(i);
      callback(i, dim.id, dim.extent, coord[i], dim.name.data(), userdata);
    }
  } catch (...) {
    return HSA_STATUS_ERROR;
  }
  return HSA_STATUS_SUCCESS;
}

// Method for iterating the events output data
PUBLIC_API hsa_status_t
hsa_ven_amd_aqlprofile_iterate_data(const hsa_ven_amd_aqlprofile_profile_t* profile,
                                    hsa_ven_amd_aqlprofile_data_callback_t callback, void* data) {
  hsa_status_t status = HSA_STATUS_SUCCESS;

  try {
    aql_profile::Pm4Factory* pm4_factory = aql_profile::Pm4Factory::Create(profile);
    const bool is_concurrent = pm4_factory->IsConcurrent();
    const uint32_t xcc_num = pm4_factory->GetXccNumber();
    const uint32_t se_number = pm4_factory->GetShaderEnginesNumber() / xcc_num;

    if (profile->type == HSA_VEN_AMD_AQLPROFILE_EVENT_TYPE_PMC) {
      uint64_t* samples = reinterpret_cast<uint64_t*>(profile->output_buffer.ptr);

      for (const hsa_ven_amd_aqlprofile_event_t* p = profile->events;
           p < profile->events + profile->event_count; ++p) {
        if ((char*)samples >= (char*)profile->output_buffer.ptr + profile->output_buffer.size)
          return HSA_STATUS_ERROR;

        if (!(pm4_factory->GetBlockInfo(p)->attr & CounterBlockAidAttr)) continue;

        // Process an MI300 UMC event for XCC 0 ONLY
        auto sample_id = p->block_index;  // sample id is the event block_index or the UMCCH id
        hsa_ven_amd_aqlprofile_info_data_t sample_info;
        sample_info.sample_id = sample_id;
        sample_info.pmc_data.event = *p;
        sample_info.pmc_data.result = *samples;
#if DEBUG_TRACE == 2
        printf(
            "DATA: sample index(%u) id(%u) bloc id(%u) index(%u) counter id(%u) "
            "res(%lu)\n",
            sample_index, sample_id, p->block_name, p->block_index, p->counter_id,
            samples[sample_index]);
#endif

        status = callback(HSA_VEN_AMD_AQLPROFILE_INFO_PMC_DATA, &sample_info, data);
        if (status == HSA_STATUS_INFO_BREAK) {
          status = HSA_STATUS_SUCCESS;
          break;
        }
        if (status != HSA_STATUS_SUCCESS) break;
        samples++;
      }
      for (uint32_t xcc_index = 0; xcc_index < xcc_num; xcc_index++) {
        for (const hsa_ven_amd_aqlprofile_event_t* p = profile->events;
             p < profile->events + profile->event_count; ++p) {
          // this check needs to be the first check as it takes care of a corner case
          // in which a UMC event is the last event in profile->events
          if (pm4_factory->GetBlockInfo(p)->attr & CounterBlockAidAttr) continue;

          if ((char*)samples > (char*)profile->output_buffer.ptr + profile->output_buffer.size)
            return HSA_STATUS_ERROR;

          // non-MI300A-AID counter event.
          uint32_t block_samples_count = 1;
          if (pm4_factory->GetBlockInfo(p)->attr & CounterBlockSeAttr)
            block_samples_count *= se_number;
          if (pm4_factory->GetBlockInfo(p)->attr & CounterBlockSaAttr)
            block_samples_count *= 2;
          if (pm4_factory->GetBlockInfo(p)->attr & CounterBlockWgpAttr)
            block_samples_count *= pm4_factory->GetNumWGPs();
          if (pm4_factory->GetBlockInfo(p)->attr & CounterBlockSqAttr && pm4_factory->IsGFX11())
            block_samples_count *= pm4_factory->GetNumWGPs();

          for (uint32_t blk = 0; blk < block_samples_count; ++blk) {
            hsa_ven_amd_aqlprofile_info_data_t sample_info;
            sample_info.sample_id = blk;
            sample_info.pmc_data.event = *p;
#if DEBUG_TRACE == 2
            printf("DATA: xcc(%u) id(%u) bloc id(%u) index(%u) counter id(%u) res(%lu)\n",
                   xcc_index, blk, p->block_name, p->block_index, p->counter_id, *samples);
#endif

            sample_info.pmc_data.result = *samples;
            status = callback(HSA_VEN_AMD_AQLPROFILE_INFO_PMC_DATA, &sample_info, data);
            if (status == HSA_STATUS_INFO_BREAK) {
              status = HSA_STATUS_SUCCESS;
              break;
            }
            if (status != HSA_STATUS_SUCCESS) break;
            samples++;
          }
        }
      }
    } else if (profile->type == HSA_VEN_AMD_AQLPROFILE_EVENT_TYPE_TRACE) {
      uint32_t mode = 2;
      switch (profile->event_count) {
        case 0:
          mode = 0;
          break;
        case UINT32_MAX:
          const_cast<hsa_ven_amd_aqlprofile_profile_t*>(profile)->event_count = 0;
          mode = 1;
          break;
      }

      if (mode != 2) {  // SQTT trace data, or SQTT pc sampling
        auto& trace_config = aql_profile::configs.at(profile->command_buffer.ptr);
        pm4_builder::SqttBuilder* sqttbuilder = pm4_factory->GetSqttBuilder();
        const uint64_t se_number_total = pm4_factory->GetShaderEnginesNumber();
        // Control buffer was allocated as the CmdBuffer prefix partition
        aql_profile::CommandBufferMgr cmd_buffer_mgr(profile);

        auto* control_ptr =
            reinterpret_cast<pm4_builder::TraceControl*>(cmd_buffer_mgr.GetPrefix1());
        // Check if SQTT buffer was wrapped
        for (size_t se_index = 0; se_index < se_number_total; se_index++) {
          if (control_ptr[se_index].status & sqttbuilder->GetUTCErrorMask()) {
            ERR_LOGGING << "SQTT memory error received, SE(" << se_index << ")";
            status = HSA_STATUS_ERROR_EXCEPTION;
          } else if (control_ptr[se_index].status & sqttbuilder->GetBufferFullMask()) {
            ERR2_LOGGING << "SQTT data buffer full, SE(" << se_index << ")";
            if (status == HSA_STATUS_SUCCESS) status = HSA_STATUS_ERROR_OUT_OF_RESOURCES;
          }
        }

        // The samples sizes are returned in the control buffer
        for (size_t se_index = 0; se_index < se_number_total; se_index++) {
          bool bMaskedIn = trace_config.GetTargetCU(se_index) >= 0;
          uint64_t sample_capacity = trace_config.GetCapacity(se_index);
          void* sample_ptr = reinterpret_cast<void*>(trace_config.GetSEBaseAddr(se_index));

          // WPTR specifies the index in thread trace buffer where next token will be
          // written by hardware. The index is incremented by size of 32 bytes.
          size_t sample_size = (control_ptr[se_index].wptr & sqttbuilder->GetWritePtrMask()) *
                               sqttbuilder->GetWritePtrBlk();

          if (pm4_factory->GetGpuId() == aql_profile::GFX11_GPU_ID) {
            sample_size = sample_size - reinterpret_cast<uint64_t>(sample_ptr);
            sample_size &= (1ull << 29) - 1;
          }

          if (sample_size >= sample_capacity) {
            ERR_LOGGING << "SQTT data out of bounds, sample_id(" << se_index << ") size("
                        << sample_size << "/" << sample_capacity << ")";
            sample_size = sample_capacity;
            if (status == HSA_STATUS_SUCCESS) status = HSA_STATUS_ERROR_OUT_OF_RESOURCES;
          }
          hsa_status_t call_status;
          if (mode == 0) {  // SQTT trace
            if (bMaskedIn) {
              hsa_ven_amd_aqlprofile_info_data_t info;
              info.sample_id = se_index;
              info.trace_data.ptr = sample_ptr;
              info.trace_data.size = sample_size;

              status = callback(HSA_VEN_AMD_AQLPROFILE_INFO_TRACE_DATA, &info, data);
            }
          } else {  // PC sampling
            pcsmp_callback_data_t* pcsmp_data = reinterpret_cast<pcsmp_callback_data_t*>(data);
            pcsmp_data->id = se_index;
            pcsmp_data->cycle = 333;
            pcsmp_data->pc = 0x333;
            call_status = callback(HSA_VEN_AMD_AQLPROFILE_INFO_TRACE_DATA, NULL, data);
          }
        }
      } else {  // SPM trace data
        if (pm4_factory->SpmKfdMode() == false) {
          const uint32_t tnumber = 1;
          void* sample_ptr = profile->output_buffer.ptr;
          const uint32_t sample_size = profile->output_buffer.size;
          const uint32_t sample_capacity = (profile->output_buffer.size / tnumber);

          for (unsigned i = 0; i < tnumber; ++i) {
            hsa_ven_amd_aqlprofile_info_data_t sample_info;
            sample_info.sample_id = i;
            sample_info.trace_data.ptr = sample_ptr;
            sample_info.trace_data.size = sample_size;
            status = callback(HSA_VEN_AMD_AQLPROFILE_INFO_TRACE_DATA, &sample_info, data);
            if (status == HSA_STATUS_INFO_BREAK) {
              status = HSA_STATUS_SUCCESS;
              break;
            }
            if (status != HSA_STATUS_SUCCESS) {
              ERR_LOGGING << "SQTT data callback error, sample_id(" << i << ") status(" << status
                          << ")";
              break;
            }
            sample_ptr = reinterpret_cast<char*>(sample_ptr) + sample_capacity;
          }
        } else {
          status = spm_kfd_namespace::spm_iterate_data(profile, callback, data);
        }
      }
    } else {
      ERR_LOGGING << "Bad profile type (" << profile->type << ")";
      status = HSA_STATUS_ERROR_INVALID_ARGUMENT;
    }
  } catch (std::exception& e) {
    ERR_LOGGING << e.what();
    return HSA_STATUS_ERROR;
  }

  return status;
}

// Method to populate the provided AQL packet with ATT Markers
PUBLIC_API hsa_status_t hsa_ven_amd_aqlprofile_att_marker(
    hsa_ven_amd_aqlprofile_profile_t* profile, aql_profile::packet_t* aql_marker_packet,
    uint32_t data, hsa_ven_amd_aqlprofile_att_marker_channel_t channel) {
  assert(profile->type == HSA_VEN_AMD_AQLPROFILE_EVENT_TYPE_TRACE);

  aql_profile::Pm4Factory* pm4_factory = aql_profile::Pm4Factory::Create(profile);
  pm4_builder::SqttBuilder* sqtt_builder = pm4_factory->GetSqttBuilder();
  pm4_builder::CmdBuilder* cmd_writer = pm4_factory->GetCmdBuilder();
  pm4_builder::CmdBuffer commands;

  // Generate start commands
  auto status = sqtt_builder->InsertMarker(&commands, data, channel);
  if (status != HSA_STATUS_SUCCESS) return status;
  aql_profile::descriptor_t& cmdbuffer = profile->command_buffer;

  size_t cmd_size = cmdbuffer.size;
  cmdbuffer.size = commands.Size();

  if (cmdbuffer.ptr == NULL) return HSA_STATUS_SUCCESS;
  if (cmd_size < commands.Size()) return HSA_STATUS_ERROR_OUT_OF_RESOURCES;

  // Populate stop aql packet
  memcpy(cmdbuffer.ptr, commands.Data(), commands.Size());
  aql_profile::PopulateAql(cmdbuffer.ptr, commands.Size(), cmd_writer, aql_marker_packet);

  return HSA_STATUS_SUCCESS;
}

}  // extern "C"
