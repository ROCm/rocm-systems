// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
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

#pragma once

#if ROCPROFSYS_USE_ROCM > 0
#    include <nlohmann/json.hpp>
#    include <rocprofiler-sdk/agent.h>

namespace rocprofsys
{
namespace agent_info
{

inline std::string
to_json_string(const rocprofiler_agent_v0_t& agent_data)
{
    nlohmann::json data;

    data["size"]                       = agent_data.size;
    data["id"]["handle"]               = agent_data.id.handle;
    data["type"]                       = agent_data.type;
    data["cpu_cores_count"]            = agent_data.cpu_cores_count;
    data["simd_count"]                 = agent_data.simd_count;
    data["mem_banks_count"]            = agent_data.mem_banks_count;
    data["caches_count"]               = agent_data.caches_count;
    data["io_links_count"]             = agent_data.io_links_count;
    data["cpu_core_id_base"]           = agent_data.cpu_core_id_base;
    data["simd_id_base"]               = agent_data.simd_id_base;
    data["max_waves_per_simd"]         = agent_data.max_waves_per_simd;
    data["lds_size_in_kb"]             = agent_data.lds_size_in_kb;
    data["gds_size_in_kb"]             = agent_data.gds_size_in_kb;
    data["num_gws"]                    = agent_data.num_gws;
    data["wave_front_size"]            = agent_data.wave_front_size;
    data["num_xcc"]                    = agent_data.num_xcc;
    data["cu_count"]                   = agent_data.cu_count;
    data["array_count"]                = agent_data.array_count;
    data["num_shader_banks"]           = agent_data.num_shader_banks;
    data["simd_arrays_per_engine"]     = agent_data.simd_arrays_per_engine;
    data["cu_per_simd_array"]          = agent_data.cu_per_simd_array;
    data["simd_per_cu"]                = agent_data.simd_per_cu;
    data["max_slots_scratch_cu"]       = agent_data.max_slots_scratch_cu;
    data["gfx_target_version"]         = agent_data.gfx_target_version;
    data["vendor_id"]                  = agent_data.vendor_id;
    data["device_id"]                  = agent_data.device_id;
    data["location_id"]                = agent_data.location_id;
    data["domain"]                     = agent_data.domain;
    data["drm_render_minor"]           = agent_data.drm_render_minor;
    data["num_sdma_engines"]           = agent_data.num_sdma_engines;
    data["num_sdma_xgmi_engines"]      = agent_data.num_sdma_xgmi_engines;
    data["num_sdma_queues_per_engine"] = agent_data.num_sdma_queues_per_engine;
    data["num_cp_queues"]              = agent_data.num_cp_queues;
    data["max_engine_clk_ccompute"]    = agent_data.max_engine_clk_ccompute;
    data["max_engine_clk_fcompute"]    = agent_data.max_engine_clk_fcompute;

    data["sdma_fw_version"]["uCodeSDMA"] = agent_data.sdma_fw_version.uCodeSDMA;
    data["sdma_fw_version"]["uCodeRes"]  = agent_data.sdma_fw_version.uCodeRes;

    data["fw_version"]["uCode"]    = agent_data.fw_version.ui32.uCode;
    data["fw_version"]["Major"]    = agent_data.fw_version.ui32.Major;
    data["fw_version"]["Minor"]    = agent_data.fw_version.ui32.Minor;
    data["fw_version"]["Stepping"] = agent_data.fw_version.ui32.Stepping;

    data["capability"]["HotPluggable"]  = agent_data.capability.ui32.HotPluggable;
    data["capability"]["HSAMMUPresent"] = agent_data.capability.ui32.HSAMMUPresent;
    data["capability"]["SharedWithGraphics"] =
        agent_data.capability.ui32.SharedWithGraphics;
    data["capability"]["QueueSizePowerOfTwo"] =
        agent_data.capability.ui32.QueueSizePowerOfTwo;
    data["capability"]["QueueSize32bit"] = agent_data.capability.ui32.QueueSize32bit;
    data["capability"]["QueueIdleEvent"] = agent_data.capability.ui32.QueueIdleEvent;
    data["capability"]["VALimit"]        = agent_data.capability.ui32.VALimit;
    data["capability"]["WatchPointsSupported"] =
        agent_data.capability.ui32.WatchPointsSupported;
    data["capability"]["WatchPointsTotalBits"] =
        agent_data.capability.ui32.WatchPointsTotalBits;
    data["capability"]["DoorbellType"] = agent_data.capability.ui32.DoorbellType;
    data["capability"]["AQLQueueDoubleMap"] =
        agent_data.capability.ui32.AQLQueueDoubleMap;
    data["capability"]["DebugTrapSupported"] =
        agent_data.capability.ui32.DebugTrapSupported;
    data["capability"]["WaveLaunchTrapOverrideSupported"] =
        agent_data.capability.ui32.WaveLaunchTrapOverrideSupported;
    data["capability"]["WaveLaunchModeSupported"] =
        agent_data.capability.ui32.WaveLaunchModeSupported;
    data["capability"]["PreciseMemoryOperationsSupported"] =
        agent_data.capability.ui32.PreciseMemoryOperationsSupported;
    data["capability"]["DEPRECATED_SRAM_EDCSupport"] =
        agent_data.capability.ui32.DEPRECATED_SRAM_EDCSupport;
    data["capability"]["Mem_EDCSupport"]  = agent_data.capability.ui32.Mem_EDCSupport;
    data["capability"]["RASEventNotify"]  = agent_data.capability.ui32.RASEventNotify;
    data["capability"]["ASICRevision"]    = agent_data.capability.ui32.ASICRevision;
    data["capability"]["SRAM_EDCSupport"] = agent_data.capability.ui32.SRAM_EDCSupport;
    data["capability"]["SVMAPISupported"] = agent_data.capability.ui32.SVMAPISupported;
    data["capability"]["CoherentHostAccess"] =
        agent_data.capability.ui32.CoherentHostAccess;
    data["capability"]["DebugSupportedFirmware"] =
        agent_data.capability.ui32.DebugSupportedFirmware;

    data["cu_per_engine"]      = agent_data.cu_per_engine;
    data["max_waves_per_cu"]   = agent_data.max_waves_per_cu;
    data["family_id"]          = agent_data.family_id;
    data["workgroup_max_size"] = agent_data.workgroup_max_size;
    data["grid_max_size"]      = agent_data.grid_max_size;
    data["local_mem_size"]     = agent_data.local_mem_size;
    data["hive_id"]            = agent_data.hive_id;
    data["gpu_id"]             = agent_data.gpu_id;

    data["workgroup_max_dim"]["x"] = agent_data.workgroup_max_dim.x;
    data["workgroup_max_dim"]["y"] = agent_data.workgroup_max_dim.y;
    data["workgroup_max_dim"]["z"] = agent_data.workgroup_max_dim.z;

    data["grid_max_dim"]["x"] = agent_data.grid_max_dim.x;
    data["grid_max_dim"]["y"] = agent_data.grid_max_dim.y;
    data["grid_max_dim"]["z"] = agent_data.grid_max_dim.z;

    data["name"] = agent_data.name ? std::string(agent_data.name) : "";
    data["vendor_name"] =
        agent_data.vendor_name ? std::string(agent_data.vendor_name) : "";
    data["product_name"] =
        agent_data.product_name ? std::string(agent_data.product_name) : "";
    data["model_name"] = agent_data.model_name ? std::string(agent_data.model_name) : "";

    data["node_id"]              = agent_data.node_id;
    data["logical_node_id"]      = agent_data.logical_node_id;
    data["logical_node_type_id"] = agent_data.logical_node_type_id;

#    if(ROCPROFILER_VERSION >= 600)
    data["runtime_visibility"]["hsa"]       = agent_data.runtime_visibility.hsa;
    data["runtime_visibility"]["hip"]       = agent_data.runtime_visibility.hip;
    data["runtime_visibility"]["rccl"]      = agent_data.runtime_visibility.rccl;
    data["runtime_visibility"]["rocdecode"] = agent_data.runtime_visibility.rocdecode;
#    endif

#    if(ROCPROFILER_VERSION >= 700)
    auto& uuid_bytes = data["uuid"]["bytes"];
    for(size_t i = 0; i < 16; ++i)
    {
        uuid_bytes[std::string("value") + std::to_string(i)] = agent_data.uuid.bytes[i];
    }
#    endif

    data["mem_banks"] = nlohmann::json::array();
    for(uint32_t i = 0; i < agent_data.mem_banks_count; ++i)
    {
        nlohmann::json bank;
        bank["heap_type"]             = agent_data.mem_banks[i].heap_type;
        bank["flags"]["HotPluggable"] = agent_data.mem_banks[i].flags.ui32.HotPluggable;
        bank["flags"]["NonVolatile"]  = agent_data.mem_banks[i].flags.ui32.NonVolatile;
        bank["width"]                 = agent_data.mem_banks[i].width;
        bank["mem_clk_max"]           = agent_data.mem_banks[i].mem_clk_max;
        bank["size_in_bytes"]         = agent_data.mem_banks[i].size_in_bytes;
        data["mem_banks"].push_back(bank);
    }

    data["caches"] = nlohmann::json::array();
    for(uint32_t i = 0; i < agent_data.caches_count; ++i)
    {
        nlohmann::json cache;
        cache["processor_id_low"]    = agent_data.caches[i].processor_id_low;
        cache["size"]                = agent_data.caches[i].size;
        cache["level"]               = agent_data.caches[i].level;
        cache["cache_line_size"]     = agent_data.caches[i].cache_line_size;
        cache["cache_lines_per_tag"] = agent_data.caches[i].cache_lines_per_tag;
        cache["association"]         = agent_data.caches[i].association;
        cache["latency"]             = agent_data.caches[i].latency;
        cache["type"]["Data"]        = agent_data.caches[i].type.ui32.Data;
        cache["type"]["Instruction"] = agent_data.caches[i].type.ui32.Instruction;
        cache["type"]["CPU"]         = agent_data.caches[i].type.ui32.CPU;
        cache["type"]["HSACU"]       = agent_data.caches[i].type.ui32.HSACU;
        data["caches"].push_back(cache);
    }

    data["io_links"] = nlohmann::json::array();
    for(uint32_t i = 0; i < agent_data.io_links_count; ++i)
    {
        nlohmann::json link;
        link["type"]          = agent_data.io_links[i].type;
        link["version_major"] = agent_data.io_links[i].version_major;
        link["version_minor"] = agent_data.io_links[i].version_minor;
        link["node_from"]     = agent_data.io_links[i].node_from;
        link["node_to"]       = agent_data.io_links[i].node_to;
        link["weight"]        = agent_data.io_links[i].weight;
        link["min_latency"]   = agent_data.io_links[i].min_latency;
        link["max_latency"]   = agent_data.io_links[i].max_latency;
        link["min_bandwidth"] = agent_data.io_links[i].min_bandwidth;
        link["max_bandwidth"] = agent_data.io_links[i].max_bandwidth;
        link["recommended_transfer_size"] =
            agent_data.io_links[i].recommended_transfer_size;
        link["flags"]["Override"]    = agent_data.io_links[i].flags.ui32.Override;
        link["flags"]["NonCoherent"] = agent_data.io_links[i].flags.ui32.NonCoherent;
        link["flags"]["NoAtomics32bit"] =
            agent_data.io_links[i].flags.ui32.NoAtomics32bit;
        link["flags"]["NoAtomics64bit"] =
            agent_data.io_links[i].flags.ui32.NoAtomics64bit;
        link["flags"]["NoPeerToPeerDMA"] =
            agent_data.io_links[i].flags.ui32.NoPeerToPeerDMA;
        data["io_links"].push_back(link);
    }

    data["gpu_index"] = (agent_data.type == ROCPROFILER_AGENT_TYPE_GPU)
                            ? agent_data.logical_node_type_id
                            : -1;

    // Normalize JSON string by escaping quotes to prepare for SQL insertion
    auto normalize_json_string = [](nlohmann::json& json_data) {
        auto   json_str = json_data.dump();
        size_t pos      = 0;
        while((pos = json_str.find('"', pos)) != std::string::npos)
        {
            json_str.replace(pos, 1, "\"\"");
            pos += 2;
        }
        return json_str;
    };

    return normalize_json_string(data);
}

}  // namespace agent_info
}  // namespace rocprofsys

#endif  // ROCPROFSYS_USE_ROCM
