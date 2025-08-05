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
#include "aqlprofile-sdk/aql_profile_v2.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <future>
#include <map>
#include <string>
#include <vector>

#include "core/counter_dimensions.hpp"

#include "core/logger.h"
#include "core/pm4_factory.h"
#include "pm4/cmd_builder.h"
#include "pm4/pmc_builder.h"
#include "pm4/spm_builder.h"
#include "pm4/sqtt_builder.h"

#include "core/commandbuffermgr.hpp"
#include "memorymanager.hpp"

#define PUBLIC_API __attribute__((visibility("default")))
#define CONSTRUCTOR_API __attribute__((constructor))
#define DESTRUCTOR_API __attribute__((destructor))
#define ERR_CHECK(cond, err, msg) \
  {                               \
    if (cond) {                   \
      ERR_LOGGING << msg;         \
      return err;                 \
    }                             \
  }

#define HSA_TRY_WRAP try {
#define HSA_CATCH_WRAP         \
  }                            \
  catch (std::exception & e) { \
    return HSA_STATUS_ERROR;   \
  }

std::vector<std::string> EventDimension::dimension_list;
std::unordered_map<std::string, size_t> EventDimension::dimension_table;

namespace aql_profile_v2 {
// Command buffer partitioning manager
// Supports Pre/Post commands partitioning
// and prefix control partition

using aql_profile::event_exception;
using aql_profile::event_t;
using ::aql_profile::Pm4Factory;

uint32_t HandleSQFlagsBlock(Pm4Factory* pm4_factory, const aqlprofile_pmc_event_t& event) {
  auto visible_id = event.event_id;
  if (event.flags.sq_flags.accum == AQLPROFILE_ACCUMULATION_LO_RES)
    visible_id = pm4_factory->GetAccumLowID();
  if (event.flags.sq_flags.accum == AQLPROFILE_ACCUMULATION_HI_RES)
    visible_id = pm4_factory->GetAccumHiID();
  return visible_id;
}

counter_des_t GetCounter(Pm4Factory* pm4_factory, EventRequest& event,
                         std::map<block_des_t, uint32_t, lt_block_des>& index_map) {
  const GpuBlockInfo* block_info = pm4_factory->GetBlockInfo(event.block_name);
  const block_des_t block_des = {block_info->id, event.block_index};
  const auto ret = index_map.insert({block_des, 0});
  auto reg_index = ret.first->second;
  auto visible_id = event.event_id;

  if (pm4_builder::SPISkip(block_info->attr, visible_id)) {
    event.bInternal = true;
    return {visible_id, reg_index, block_des, block_info};
  }

  if (reg_index >= block_info->counter_count)
    throw std::string("Event is out of block counter registers number limit");

  if (event.flags.raw) {
    if (event.block_name == HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SQ) {
      visible_id = HandleSQFlagsBlock(pm4_factory, event);
    } else {
      throw HSA_STATUS_ERROR_INVALID_ARGUMENT;
    }
  }

  ret.first->second++;
  return {visible_id, reg_index, block_des, block_info};
}

pm4_builder::counters_vector CountersVec(std::vector<EventRequest>& events,
                                         Pm4Factory* pm4_factory) {
  pm4_builder::counters_vector vec;
  std::map<block_des_t, uint32_t, lt_block_des> index_map;

  for (auto& event : events) vec.push_back(GetCounter(pm4_factory, event, index_map));

  if (pm4_factory->IsGFX10() && (vec.get_attr() & CounterBlockGRBMAttr) == 0) {
    EventRequest grbm_event{0};
    grbm_event.block_name = HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GRBM;
    vec.push_back(GetCounter(pm4_factory, grbm_event, index_map));
  }
  return vec;
}

// Method for iterating the events output data
hsa_status_t _internal_aqlprofile_pmc_iterate_data(aqlprofile_handle_t handle,
                                                   aqlprofile_pmc_data_callback_t callback,
                                                   void* userdata) {
  auto counter_memorymgr = MemoryManager::GetManager(handle.handle);
  auto agent = counter_memorymgr->AgentHandle();
  CounterMemoryManager* memorymgr = dynamic_cast<CounterMemoryManager*>(counter_memorymgr.get());
  if (!memorymgr) return HSA_STATUS_ERROR_INVALID_ARGUMENT;

  aql_profile::Pm4Factory* pm4_factory = aql_profile::Pm4Factory::Create(memorymgr->AgentHandle());
  const uint32_t xcc_num = pm4_factory->GetXccNumber();

  uint64_t* samples = reinterpret_cast<uint64_t*>(memorymgr->GetOutputBuf());
  uint64_t* buffer_end_location = samples + memorymgr->GetOutputBufSize() / sizeof(uint64_t);
  auto& events = memorymgr->GetEvents();

  size_t umc_sample_id = 0;
  if (xcc_num > 1)
    for (auto& event : events) {
      if (samples >= buffer_end_location) return HSA_STATUS_ERROR;

      if (!(pm4_factory->GetBlockInfo(event.block_name)->attr & CounterBlockUmcAttr)) continue;

#if DEBUG_TRACE == 2
      printf("DATA: sample index(%u) id(%u) bloc id(%u) index(%u) counter id(%u) res(%lu)\n",
             sample_index, sample_id, p->block_name, p->block_index, p->counter_id, *samples);
#endif

      hsa_status_t status = callback(event, event.block_index, *samples, userdata);
      samples++;
      umc_sample_id++;

      if (status == HSA_STATUS_INFO_BREAK) return HSA_STATUS_SUCCESS;
      if (status != HSA_STATUS_SUCCESS) return status;
    }

  for (uint32_t xcc_index = 0; xcc_index < xcc_num; xcc_index++)
    for (auto& event : events) {
      if (samples >= buffer_end_location) return HSA_STATUS_ERROR;

      if (pm4_factory->GetBlockInfo(event.block_name)->attr & CounterBlockUmcAttr) continue;

      // non-MI300A-AID counter event.
      uint32_t block_samples_count = pm4_factory->GetNumEvents(event.block_name);
      const EventAttribDimension& attrib = EventAttribDimension::get(agent, event.block_name);
      if (!attrib.get_num()) return HSA_STATUS_ERROR;
      size_t xcc_sample_count = attrib.get_num_instances() * block_samples_count;
      for (uint32_t blk = 0; blk < block_samples_count; ++blk) {
#if DEBUG_TRACE == 2
        printf("DATA: xcc(%u) blk(%u) bloc id(%u) index(%u) counter id(%u) res(%lu)\n", xcc_index,
               blk, event.block_name, event.block_index, event.event_id, *samples);
#endif
        size_t xcc_sample_id = xcc_sample_count * xcc_index +
                               static_cast<size_t>(event.block_index) * block_samples_count + blk;

        if (!event.bInternal) {
          hsa_status_t status = callback(event, xcc_sample_id, *samples, userdata);
          if (status == HSA_STATUS_INFO_BREAK)
            return HSA_STATUS_SUCCESS;
          else if (status != HSA_STATUS_SUCCESS)
            return status;
        }

        samples++;
      }
    }

  return HSA_STATUS_SUCCESS;
}

hsa_status_t _internal_aqlprofile_pmc_create_packets(
    aqlprofile_handle_t* handle, aqlprofile_pmc_aql_packets_t* packets,
    aqlprofile_pmc_profile_t profile, aqlprofile_memory_alloc_callback_t alloc_cb,
    aqlprofile_memory_dealloc_callback_t dealloc_cb, aqlprofile_memory_copy_t memcpy_cb,
    void* userdata) {
  pm4_builder::CmdBuffer commands;
  auto memorymgr =
      std::make_shared<CounterMemoryManager>(profile.agent, alloc_cb, dealloc_cb, userdata);
  MemoryManager::RegisterManager(memorymgr);
  memorymgr->CopyEvents(profile.events, profile.event_count);

  pm4_builder::CmdBuffer read_cmd;
  pm4_builder::CmdBuffer start_cmd;
  pm4_builder::CmdBuffer stop_cmd;

  aql_profile::Pm4Factory* pm4_factory = aql_profile::Pm4Factory::Create(profile.agent);
  const pm4_builder::counters_vector countersVec = CountersVec(memorymgr->GetEvents(), pm4_factory);

  pm4_builder::PmcBuilder* pmc_builder = pm4_factory->GetPmcBuilder();

  // Start outputbuf ptr
  size_t output_bytes = 8;  // Extra space for GRBM block on gfx10
  for (auto& event : memorymgr->GetEvents())
    output_bytes += pm4_factory->GetBytesNeeded(event.block_name);
  memorymgr->CreateOutputBuf(output_bytes);
  // Generate read commands
  size_t data_size = pmc_builder->Read(&read_cmd, countersVec, memorymgr->GetOutputBuf());
  // Generate start commands
  pmc_builder->Start(&start_cmd, countersVec);
  // Generate stop commands
  pmc_builder->Stop(&stop_cmd, countersVec);

  ERR_CHECK(data_size == 0, HSA_STATUS_ERROR, "PMC Builder Stop(): data size set to zero");
  if (memorymgr->GetOutputBufSize() < data_size) return HSA_STATUS_ERROR_OUT_OF_RESOURCES;

  // Copy generated commands
  size_t start_size = aql_profile::CommandBufferMgr::Align(start_cmd.Size());
  size_t stop_size = aql_profile::CommandBufferMgr::Align(stop_cmd.Size());
  size_t read_size = aql_profile::CommandBufferMgr::Align(read_cmd.Size());
  memorymgr->CreateCmdBuf(start_size + stop_size + read_size);

  handle->handle = memorymgr->GetHandler();
  pm4_builder::CmdBuilder* cmd_writer = pm4_factory->GetCmdBuilder();
  uint8_t* cmdbuf = reinterpret_cast<uint8_t*>(memorymgr->GetCmdBuf());

  memcpy_cb(cmdbuf, read_cmd.Data(), read_cmd.Size(), userdata);
  aql_profile::PopulateAql(cmdbuf, read_cmd.Size(), cmd_writer, &packets->read_packet);
  cmdbuf += read_size;
  memcpy_cb(cmdbuf, start_cmd.Data(), start_cmd.Size(), userdata);
  aql_profile::PopulateAql(cmdbuf, start_cmd.Size(), cmd_writer, &packets->start_packet);
  cmdbuf += start_size;
  memcpy_cb(cmdbuf, stop_cmd.Data(), stop_cmd.Size(), userdata);
  aql_profile::PopulateAql(cmdbuf, stop_cmd.Size(), cmd_writer, &packets->stop_packet);

  return HSA_STATUS_SUCCESS;
}

}  // namespace aql_profile_v2

extern "C" {

PUBLIC_API hsa_status_t aqlprofile_pmc_create_packets(
    aqlprofile_handle_t* handle, aqlprofile_pmc_aql_packets_t* packets,
    aqlprofile_pmc_profile_t profile, aqlprofile_memory_alloc_callback_t alloc_cb,
    aqlprofile_memory_dealloc_callback_t dealloc_cb, aqlprofile_memory_copy_t memcpy_cb,
    void* userdata) {
  try {
    return aql_profile_v2::_internal_aqlprofile_pmc_create_packets(
        handle, packets, profile, alloc_cb, dealloc_cb, memcpy_cb, userdata);
  } catch (hsa_status_t err) {
    ERR_LOGGING << err;
    return err;
  } catch (std::exception& e) {
    ERR_LOGGING << e.what();
    return HSA_STATUS_ERROR;
  } catch (...) {
    return HSA_STATUS_ERROR;
  }
}

PUBLIC_API void aqlprofile_pmc_delete_packets(aqlprofile_handle_t handle) {
  try {
    MemoryManager::DeleteManager(handle.handle);
  } catch (std::exception& e) {
    return;
  } catch (...) {
    return;
  }
}

PUBLIC_API hsa_status_t aqlprofile_pmc_iterate_data(aqlprofile_handle_t handle,
                                                    aqlprofile_pmc_data_callback_t callback,
                                                    void* userdata) {
  try {
    return aql_profile_v2::_internal_aqlprofile_pmc_iterate_data(handle, callback, userdata);
  } catch (hsa_status_t err) {
    ERR_LOGGING << err;
    return err;
  } catch (std::exception& e) {
    ERR_LOGGING << e.what();
    return HSA_STATUS_ERROR;
  } catch (...) {
    return HSA_STATUS_ERROR;
  }
}

PUBLIC_API hsa_status_t aqlprofile_iterate_event_ids(aqlprofile_eventname_callback_t callback,
                                                     void* user_data) {
  try {
    EventDimension::init();
    for (auto& [name, id] : EventDimension::dimension_table) {
      if (auto ret = callback(id, name.c_str(), user_data); ret != HSA_STATUS_SUCCESS) {
        return ret;
      }
    }
  } catch (hsa_status_t err) {
    ERR_LOGGING << err;
    return err;
  } catch (...) {
    return HSA_STATUS_ERROR;
  }
  return HSA_STATUS_SUCCESS;
}

PUBLIC_API hsa_status_t aqlprofile_iterate_event_coord(aqlprofile_agent_handle_t agent,
                                                       aqlprofile_pmc_event_t event,
                                                       uint64_t counter_id,
                                                       aqlprofile_coordinate_callback_t callback,
                                                       void* userdata) {
  try {
    const EventAttribDimension& attrib = EventAttribDimension::get(agent, event.block_name);

    if (!attrib.get_num()) return HSA_STATUS_ERROR;

    std::array<uint8_t, 32> coord;
    assert(attrib.get_num() < coord.size());
    attrib.get_coordinates(coord.data(), counter_id);

    for (size_t i = 0; i < attrib.get_num(); i++) {
      EventDimension dim = attrib.get_dim(i);
      callback(i, dim.id, dim.extent, coord.at(i), dim.name.data(), userdata);
    }
  } catch (hsa_status_t err) {
    ERR_LOGGING << err;
    return err;
  } catch (...) {
    return HSA_STATUS_ERROR;
  }
  return HSA_STATUS_SUCCESS;
}

PUBLIC_API hsa_status_t aqlprofile_register_agent(aqlprofile_agent_handle_t* agent_id,
                                                  const aqlprofile_agent_info_t* agent_info) {
  return aqlprofile_register_agent_info(agent_id, agent_info, AQLPROFILE_AGENT_VERSION_V0);
}

PUBLIC_API hsa_status_t aqlprofile_register_agent_info(aqlprofile_agent_handle_t* agent_id,
                                                       const void* agent_info,
                                                       aqlprofile_agent_version_t version) {
  try {
    if (agent_info == NULL) return HSA_STATUS_ERROR_INVALID_ARGUMENT;

    switch (version) {
      case AQLPROFILE_AGENT_VERSION_V0: {
        const auto* info = static_cast<const aqlprofile_agent_info_t*>(agent_info);
        aqlprofile_agent_info_v1_t info_v1 = {
            .agent_gfxip = info->agent_gfxip,
            .xcc_num = info->xcc_num,
            .se_num = info->se_num,
            .cu_num = info->cu_num,
            .shader_arrays_per_se = info->shader_arrays_per_se,
            .domain = 0,
            .location_id = 0,
        };
        *agent_id = aql_profile::RegisterAgent(&info_v1);
      } break;
      case AQLPROFILE_AGENT_VERSION_V1: {
        *agent_id =
            aql_profile::RegisterAgent(static_cast<const aqlprofile_agent_info_v1_t*>(agent_info));
      } break;
      case AQLPROFILE_AGENT_VERSION_NONE:
      case AQLPROFILE_AGENT_VERSION_LAST:
        return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    }
  } catch (hsa_status_t err) {
    ERR_LOGGING << err;
    return err;
  } catch (...) {
    return HSA_STATUS_ERROR;
  }

  return HSA_STATUS_SUCCESS;
}

// Check if event is valid for the specific GPU
PUBLIC_API hsa_status_t aqlprofile_validate_pmc_event(aqlprofile_agent_handle_t agent,
                                                      const aqlprofile_pmc_event_t* event,
                                                      bool* result) {
  hsa_status_t status = HSA_STATUS_SUCCESS;
  *result = false;

  try {
    aql_profile::Pm4Factory* pm4_factory = aql_profile::Pm4Factory::Create(agent);
    if (pm4_factory->GetBlockInfo(event) != NULL) *result = true;
  } catch (hsa_status_t err) {
    ERR_LOGGING << err;
    return err;
  } catch (...) {
    return HSA_STATUS_ERROR;
  }

  return status;
}

PUBLIC_API hsa_status_t aqlprofile_get_pmc_info(const aqlprofile_pmc_profile_t* profile,
                                                aqlprofile_pmc_info_type_t attribute, void* value) {
  if (!profile) return HSA_STATUS_ERROR;

  try {
    aql_profile::Pm4Factory* pm4_factory = aql_profile::Pm4Factory::Create(profile->agent);

    switch (attribute) {
      case AQLPROFILE_INFO_BLOCK_ID: {
        hsa_ven_amd_aqlprofile_id_query_t* query =
            reinterpret_cast<hsa_ven_amd_aqlprofile_id_query_t*>(value);
        const uint32_t block = pm4_factory->FindBlock(query->name);
        const GpuBlockInfo* info = pm4_factory->GetBlockInfo(block);
        if (!info) return HSA_STATUS_ERROR;

        const auto& attrib =
            EventAttribDimension::get(profile->agent, (hsa_ven_amd_aqlprofile_block_name_t)block);
        if (!attrib.get_num()) return HSA_STATUS_ERROR;

        query->id = block;
        query->instance_count = attrib.get_num_instances();
      } break;
      case AQLPROFILE_INFO_BLOCK_COUNTERS: {
        *reinterpret_cast<uint32_t*>(value) =
            pm4_factory->GetBlockInfo(&profile->events[0])->counter_count;
      } break;
      default:
        return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    }

  } catch (hsa_status_t err) {
    ERR_LOGGING << err;
    return err;
  } catch (...) {
    return HSA_STATUS_ERROR;
  }
  return HSA_STATUS_SUCCESS;
}
}  // extern "C"
