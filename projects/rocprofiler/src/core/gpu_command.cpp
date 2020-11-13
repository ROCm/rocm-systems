/******************************************************************************
Copyright (c) 2018 Advanced Micro Devices, Inc. All rights reserved.

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
*******************************************************************************/

#include "gpu_command.h"

#include <hsa.h>

#include <map>

#include "core/profile.h"
#include "util/exception.h"
#include "util/hsa_rsrc_factory.h"

namespace rocprofiler {
size_t CreateGpuCommand(gpu_cmd_op_t op,
                        const rocprofiler::util::AgentInfo* agent_info,
                        packet_t* command,
                        const size_t& slot_count) {
  if (op >= NUMBER_GPU_CMD_OP) EXC_RAISING(HSA_STATUS_ERROR, "bad op value (" << op << ")");

  const bool is_legacy = (strncmp(agent_info->name, "gfx8", 4) == 0);
  const size_t packet_count = (is_legacy) ? Profile::LEGACY_SLOT_SIZE_PKT : 1;

  rocprofiler::util::HsaRsrcFactory* hsa_rsrc = &rocprofiler::util::HsaRsrcFactory::Instance();

  if (packet_count > slot_count) EXC_RAISING(HSA_STATUS_ERROR, "packet_count > slot_count");

  // AQLprofile object
  hsa_ven_amd_aqlprofile_profile_t profile{};
  profile.agent = agent_info->dev_id;
  // Query for cmd buffer size
  hsa_ven_amd_aqlprofile_info_type_t info_type =
    (hsa_ven_amd_aqlprofile_info_type_t)((int)HSA_VEN_AMD_AQLPROFILE_INFO_ENABLE_CMD + (int)op);
  hsa_status_t status = hsa_rsrc->AqlProfileApi()->hsa_ven_amd_aqlprofile_get_info(&profile, info_type, NULL);
  if (status != HSA_STATUS_SUCCESS) EXC_RAISING(status, "get_info(ENABLE_CMD ).size exc, op(" << int(op) << ")");
  if (profile.command_buffer.size == 0) EXC_RAISING(status, "get_info(ENABLE_CMD).size == 0");
  // Allocate cmd buffer
  const size_t aligment_mask = 0x100 - 1;
  profile.command_buffer.ptr =
    hsa_rsrc->AllocateSysMemory(agent_info, profile.command_buffer.size);
  if ((reinterpret_cast<uintptr_t>(profile.command_buffer.ptr) & aligment_mask) != 0) {
    EXC_RAISING(status, "profile.command_buffer.ptr bad alignment");
  }

  // Generating cmd packet
  if (is_legacy) {
    packet_t packet{};

    // Query for cmd buffer data
    status = hsa_rsrc->AqlProfileApi()->hsa_ven_amd_aqlprofile_get_info(&profile, info_type, &packet);
    if (status != HSA_STATUS_SUCCESS) EXC_RAISING(status, "get_info(ENABLE_CMD).data exc");

    // Check for legacy GFXIP
    status = hsa_rsrc->AqlProfileApi()->hsa_ven_amd_aqlprofile_legacy_get_pm4(&packet, command);
    if (status != HSA_STATUS_SUCCESS) AQL_EXC_RAISING(status, "hsa_ven_amd_aqlprofile_legacy_get_pm4");
  } else {
    // Query for cmd buffer data
    status = hsa_rsrc->AqlProfileApi()->hsa_ven_amd_aqlprofile_get_info(&profile, info_type, command);
    if (status != HSA_STATUS_SUCCESS) EXC_RAISING(status, "get_info(ENABLE_CMD).data exc");
  }

  // Return cmd packet data size
  return (packet_count * sizeof(packet_t));
}

struct gpu_cmd_entry_t {
  packet_t command[Profile::LEGACY_SLOT_SIZE_PKT];
  uint32_t size;
};
struct gpu_cmd_key_t {
  gpu_cmd_op_t op;
  uint32_t chip_id;
};
struct gpu_cmd_fncomp_t {
  bool operator() (const gpu_cmd_key_t& a, const gpu_cmd_key_t& b) const {
    return (a.op < b.op) || ((a.op == b.op) && (a.chip_id < b.chip_id));
  }
};
typedef std::map<gpu_cmd_key_t, gpu_cmd_entry_t, gpu_cmd_fncomp_t> gpu_cmd_map_t;

typedef std::mutex gpu_cmd_mutex_t;
gpu_cmd_mutex_t gpu_cmd_mutex;

size_t GetGpuCommand(gpu_cmd_op_t op,
                       const rocprofiler::util::AgentInfo* agent_info,
                       packet_t** command_out) {
  static gpu_cmd_map_t* map = NULL;

  // Getting chip-id
  uint32_t chip_id = 0;
  hsa_agent_info_t attribute = static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_CHIP_ID);
  hsa_status_t status = hsa_agent_get_info(agent_info->dev_id, attribute, &chip_id);
  if (status != HSA_STATUS_SUCCESS) EXC_RAISING(status, "hsa_agent_get_info failed");

  // Query/create a command
  std::lock_guard<gpu_cmd_mutex_t> lck(gpu_cmd_mutex);
  if (map == NULL) map = new gpu_cmd_map_t;
  auto ret = map->insert({gpu_cmd_key_t{op, chip_id}, gpu_cmd_entry_t{}});
  gpu_cmd_map_t::iterator it = ret.first;
  if (ret.second) {
    it->second.size = CreateGpuCommand(op, agent_info, it->second.command, Profile::LEGACY_SLOT_SIZE_PKT);
  }

  *command_out = it->second.command;
  return it->second.size;
}

}  // namespace rocprofiler
