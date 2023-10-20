// MIT License
//
// Copyright (c) 2023 ROCm Developer Tools
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

#include <rocprofiler/agent.h>
#include <rocprofiler/fwd.h>
#include <rocprofiler/rocprofiler.h>

#include "lib/rocprofiler/agent.hpp"
#include "lib/rocprofiler/hsa/agent_cache.hpp"

#include <fmt/core.h>
#include <glog/logging.h>
#include <hsa/hsa_api_trace.h>
#include <libdrm/amdgpu.h>
#include <xf86drm.h>

#include <filesystem>
#include <fstream>
#include <limits>
#include <regex>
#include <sstream>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace rocprofiler
{
namespace agent
{
namespace
{
namespace fs = ::std::filesystem;

struct cpu_info
{
    long        processor   = -1;
    long        family      = -1;
    long        model       = -1;
    long        physical_id = -1;
    long        core_id     = -1;
    long        apicid      = -1;
    std::string vendor_id   = {};
    std::string model_name  = {};

    bool is_valid() const
    {
        return !(processor < 0 || family < 0 || model < 0 || physical_id < 0 || core_id < 0 ||
                 apicid < 0 || vendor_id.empty() || model_name.empty());
    }
};

auto
parse_cpu_info()
{
    auto ifs  = std::ifstream{"/proc/cpuinfo"};
    auto data = std::vector<cpu_info>{};
    if(!ifs) return data;

    auto read_blocks = [&ifs]() {
        auto blocks        = std::vector<std::vector<std::string>>{};
        auto current_block = std::vector<std::string>{};
        auto line          = std::string{};
        while(std::getline(ifs, line))
        {
            if(ifs.eof())
            {
                if(!current_block.empty()) blocks.emplace_back(std::move(current_block));
                break;
            }

            if(line.empty())
            {
                if(!current_block.empty()) blocks.emplace_back(std::move(current_block));
                current_block.clear();
            }
            else
            {
                current_block.emplace_back(line);
            }
        }
        return blocks;
    };

    auto processor_blocks = read_blocks();
    auto processor_info   = std::vector<cpu_info>{};
    processor_info.reserve(processor_blocks.size());

    for(const auto& bitr : processor_blocks)
    {
        auto info_v = cpu_info{};
        for(const auto& itr : bitr)
        {
            auto             match = std::smatch{};
            const std::regex re{".*: (.*)$"};
            if(std::regex_match(itr, match, re))
            {
                if(match.size() == 2)
                {
                    std::ssub_match value = match[1];

                    if(itr.find("vendor_id") == 0)
                        info_v.vendor_id = value.str();
                    else if(itr.find("model name") == 0)
                        info_v.model_name = value.str();
                    else if(itr.find("processor") == 0)
                        info_v.processor = std::stol(value.str());
                    else if(itr.find("cpu family") == 0)
                        info_v.family = std::stol(value.str());
                    else if(itr.find("model") == 0 && itr.find("model name") != 0)
                        info_v.model = std::stol(value.str());
                    else if(itr.find("physical id") == 0)
                        info_v.physical_id = std::stol(value.str());
                    else if(itr.find("core id") == 0)
                        info_v.core_id = std::stol(value.str());
                    else if(itr.find("apicid") == 0)
                        info_v.apicid = std::stol(value.str());
                }
            }
        }
        if(info_v.is_valid())
            processor_info.emplace_back(info_v);
        else
        {
            LOG(ERROR) << "Invalid processor info: "
                       << fmt::format("processor={}, vendor={}, family={}, model={}, name={}, "
                                      "physical id={}, core id={}, apicid={}",
                                      info_v.processor,
                                      info_v.vendor_id,
                                      info_v.family,
                                      info_v.model,
                                      info_v.model_name,
                                      info_v.physical_id,
                                      info_v.core_id,
                                      info_v.apicid);
        }
    }

    return processor_info;
}

auto&
get_cpu_info()
{
    static auto _v = parse_cpu_info();
    return _v;
}

// check to see if the file is readable
bool
is_readable(const fs::path& fpath)
{
    auto ec    = std::error_code{};
    auto perms = fs::status(fpath, ec).permissions();
    LOG_IF(ERROR, ec) << fmt::format(
        "Error getting status for file '{}': {}", fpath.string(), ec.message());
    return (!ec && (perms & fs::perms::owner_read) != fs::perms::none);
}

auto
read_file(const std::string& fname)
{
    auto data = std::vector<std::string>{};

    if(!is_readable(fs::path{fname}))
        throw std::runtime_error{fmt::format("file '{}' cannot be read", fname)};

    auto ifs = std::ifstream{fname};
    if(!ifs || !ifs.good())
        throw std::runtime_error{fmt::format("file '{}' cannot be read", fname)};

    while(true)
    {
        auto value = std::string{};
        ifs >> value;
        if(ifs.eof() || value.empty()) break;

        data.emplace_back(value);
    }

    return data;
}

auto
read_map(const std::string& fname)
{
    auto data = std::unordered_map<std::string, std::string>{};

    if(!is_readable(fs::path{fname}))
        throw std::runtime_error{fmt::format("file '{}' cannot be read", fname)};

    auto ifs = std::ifstream{fname};
    if(!ifs || !ifs.good())
        throw std::runtime_error{fmt::format("file '{}' cannot be read", fname)};

    auto last_label = std::string{};
    while(true)
    {
        auto label = std::string{};
        ifs >> label;
        if(ifs.eof() || label.empty()) break;

        auto entry = std::string{};
        ifs >> entry;
        if(ifs.eof())
            throw std::runtime_error{
                fmt::format("unexpected file format in '{}' at {}", fname, label)};

        auto ret = data.emplace(label, entry);
        if(!ret.second)
            throw std::runtime_error{
                fmt::format("duplicate entry in '{}': '{}' (='{}'). last label was '{}'",
                            fname,
                            label,
                            entry,
                            last_label)};

        if(!label.empty()) last_label = std::move(label);
    }

    return data;
}

template <typename MapT, typename Tp>
void
read_property(const MapT& data, const std::string& label, Tp& value)
{
    using mutable_type = std::remove_const_t<Tp>;

    if constexpr(std::is_enum<Tp>::value)
    {
        using value_type = std::underlying_type_t<mutable_type>;
        // never expect this to be true but it does guard against infinite recursion
        static_assert(!std::is_enum<value_type>::value, "Expected non-enum type");

        auto value_v = static_cast<value_type>(value);
        read_property(data, label, value_v);
        if constexpr(std::is_const<Tp>::value)
            const_cast<mutable_type&>(value) = static_cast<mutable_type>(value_v);
        else
            value = static_cast<Tp>(value_v);
    }
    else
    {
        static_assert(std::is_integral<Tp>::value, "Expected integral type");
        using value_type = std::conditional_t<std::is_signed<Tp>::value, intmax_t, uintmax_t>;

        if(data.find(label) == data.end())
        {
            LOG(ERROR) << "agent properties map missing " << label << " entry";
            return;
        }

        auto       iss = std::istringstream{data.at(label)};
        value_type local_value;
        iss >> local_value;

        // verify that we have used the correct data sizes
        constexpr auto min_value = std::numeric_limits<Tp>::min();
        constexpr auto max_value = std::numeric_limits<Tp>::max();
        if(local_value < min_value)
        {
            throw std::runtime_error{
                fmt::format("data with label {} has a value (={}) which is less "
                            "than the min value for the type (={})",
                            label,
                            local_value,
                            min_value)};
        }
        else if(local_value > max_value)
        {
            throw std::runtime_error{fmt::format("data with label {} has a value (={}) which is "
                                                 "greater "
                                                 "than the max value for the type (={})",
                                                 label,
                                                 local_value,
                                                 max_value)};
        }

        if constexpr(std::is_const<Tp>::value)
            const_cast<mutable_type&>(value) = static_cast<mutable_type>(local_value);
        else
            value = static_cast<Tp>(local_value);
    }
}

constexpr auto
compute_version(uint32_t major_v, uint32_t minor_v, uint32_t patch_v)
{
    return (major_v * 10000) + (minor_v * 100) + patch_v;
}

auto
read_topology()
{
    using unique_agent_t = std::unique_ptr<rocprofiler_agent_t, void (*)(rocprofiler_agent_t*)>;

    auto sysfs_nodes_path = fs::path{"/sys/class/kfd/kfd/topology/nodes/"};
    if(!fs::exists(sysfs_nodes_path))
        throw std::runtime_error{
            fmt::format("sysfs nodes path '{}' does not exist", sysfs_nodes_path.string())};

    using pc_sampling_config_vec_t = std::vector<rocprofiler_pc_sampling_configuration_t>;

    static auto mi200_pc_sampling_config = pc_sampling_config_vec_t{
        rocprofiler_pc_sampling_configuration_t{ROCPROFILER_PC_SAMPLING_METHOD_HOST_TRAP,
                                                ROCPROFILER_PC_SAMPLING_UNIT_TIME,
                                                1UL,
                                                1000000000UL,
                                                0}};

    const auto& cpu_info_v = get_cpu_info();
    auto        data       = std::vector<unique_agent_t>{};
    uint64_t    idcount    = 0;
    uint64_t    nodecount  = 0;

    while(true)
    {
        auto idx       = idcount++;
        auto node_path = sysfs_nodes_path / std::to_string(idx);
        // assumes that nodes are monotonically increasing and thus once we are missing a node
        // folder for a number, there are no more nodes
        if(!fs::exists(node_path)) break;
        // skip if we don't have permission to read the file
        if(!is_readable(node_path)) continue;

        auto properties  = std::unordered_map<std::string, std::string>{};
        auto name_prop   = std::vector<std::string>{};
        auto gpu_id_prop = std::vector<std::string>{};
        try
        {
            properties  = read_map(node_path / "properties");
            name_prop   = read_file(node_path / "name");
            gpu_id_prop = read_file(node_path / "gpu_id");
        } catch(std::runtime_error& e)
        {
            LOG(ERROR) << "Error reading '" << (node_path / "properties").string()
                       << "' :: " << e.what();
            continue;
        }

        // we may have been able to open the properties file but if it was empty, we ignore it
        if(properties.empty()) continue;

        auto agent_info = rocprofiler_agent_t{};
        memset(&agent_info, 0, sizeof(agent_info));

        agent_info.size      = sizeof(rocprofiler_agent_t);
        agent_info.id.handle = idx;
        agent_info.type      = ROCPROFILER_AGENT_TYPE_NONE;
        agent_info.node_id   = nodecount++;

        if(!name_prop.empty())
            agent_info.model_name = strdup(name_prop.front().c_str());
        else
            agent_info.model_name = "";

        if(!gpu_id_prop.empty()) agent_info.gpu_id = std::stoull(gpu_id_prop.front());

        read_property(properties, "cpu_cores_count", agent_info.cpu_cores_count);
        read_property(properties, "simd_count", agent_info.simd_count);

        if(agent_info.cpu_cores_count > 0)
            agent_info.type = ROCPROFILER_AGENT_TYPE_CPU;
        else if(agent_info.simd_count > 0)
            agent_info.type = ROCPROFILER_AGENT_TYPE_GPU;

        read_property(properties, "mem_banks_count", agent_info.mem_banks_count);
        read_property(properties, "caches_count", agent_info.caches_count);
        read_property(properties, "io_links_count", agent_info.io_links_count);
        read_property(properties, "cpu_core_id_base", agent_info.cpu_core_id_base);
        read_property(properties, "simd_id_base", agent_info.simd_id_base);
        read_property(properties, "max_waves_per_simd", agent_info.max_waves_per_simd);
        read_property(properties, "lds_size_in_kb", agent_info.lds_size_in_kb);
        read_property(properties, "gds_size_in_kb", agent_info.gds_size_in_kb);
        read_property(properties, "num_gws", agent_info.num_gws);
        read_property(properties, "wave_front_size", agent_info.wave_front_size);
        read_property(properties, "array_count", agent_info.array_count);
        read_property(properties, "simd_arrays_per_engine", agent_info.simd_arrays_per_engine);
        read_property(properties, "cu_per_simd_array", agent_info.cu_per_simd_array);
        read_property(properties, "simd_per_cu", agent_info.simd_per_cu);
        read_property(properties, "max_slots_scratch_cu", agent_info.max_slots_scratch_cu);
        read_property(properties, "gfx_target_version", agent_info.gfx_target_version);
        read_property(properties, "vendor_id", agent_info.vendor_id);
        read_property(properties, "device_id", agent_info.device_id);
        read_property(properties, "location_id", agent_info.location_id);
        read_property(properties, "domain", agent_info.domain);
        read_property(properties, "drm_render_minor", agent_info.drm_render_minor);
        read_property(properties, "hive_id", agent_info.hive_id);
        read_property(properties, "num_sdma_engines", agent_info.num_sdma_engines);
        read_property(properties, "num_sdma_xgmi_engines", agent_info.num_sdma_xgmi_engines);
        read_property(
            properties, "num_sdma_queues_per_engine", agent_info.num_sdma_queues_per_engine);
        read_property(properties, "num_cp_queues", agent_info.num_cp_queues);
        read_property(properties, "max_engine_clk_ccompute", agent_info.max_engine_clk_ccompute);

        agent_info.name         = "";
        agent_info.product_name = "";
        agent_info.vendor_name  = "";
        if(agent_info.type == ROCPROFILER_AGENT_TYPE_GPU)
        {
            constexpr auto workgrp_max = 1024;
            constexpr auto grid_max    = std::numeric_limits<uint32_t>::max();

            read_property(
                properties, "max_engine_clk_fcompute", agent_info.max_engine_clk_fcompute);
            read_property(properties, "local_mem_size", agent_info.local_mem_size);
            read_property(properties, "fw_version", agent_info.fw_version.Value);
            read_property(properties, "capability", agent_info.capability.Value);
            read_property(properties, "sdma_fw_version", agent_info.sdma_fw_version.Value);
            agent_info.fw_version.Value &= 0x3ff;
            agent_info.sdma_fw_version.Value &= 0x3ff;
            agent_info.workgroup_max_size = workgrp_max;  // hardcoded in hsa-runtime
            agent_info.workgroup_max_dim  = {workgrp_max, workgrp_max, workgrp_max};
            agent_info.grid_max_size      = grid_max;  // hardcoded in hsa-runtime
            agent_info.grid_max_dim       = {grid_max, grid_max, grid_max};
            agent_info.cu_count           = agent_info.simd_count / agent_info.simd_per_cu;

            if(int drm_fd = 0; (drm_fd = drmOpenRender(agent_info.drm_render_minor)) >= 0)
            {
                uint32_t major_version = 0;
                uint32_t minor_version = 0;
                auto*    device_handle = amdgpu_device_handle{};
                if(amdgpu_device_initialize(
                       drm_fd, &major_version, &minor_version, &device_handle) == 0)
                {
                    auto major = (agent_info.gfx_target_version / 10000) % 100;
                    auto minor = (agent_info.gfx_target_version / 100) % 100;
                    auto step  = (agent_info.gfx_target_version % 100);

                    agent_info.name =
                        strdup(fmt::format("gfx{}{}{:x}", major, minor, step).c_str());
                    agent_info.product_name = strdup(amdgpu_get_marketing_name(device_handle));
                    agent_info.vendor_name  = strdup("AMD");

                    amdgpu_gpu_info gpu_info = {};
                    if(amdgpu_query_gpu_info(device_handle, &gpu_info) == 0)
                    {
                        agent_info.family_id = gpu_info.family_id;
                    }
                    amdgpu_device_deinitialize(device_handle);
                }
                drmClose(drm_fd);
            }

            // TODO(jomadsen): make contingent on whether this process acquired the PC sampling
            // device lock
            {
                constexpr auto gfx90a_version = compute_version(9, 0, 10);

                if(agent_info.gfx_target_version >= gfx90a_version)
                {
                    agent_info.pc_sampling_configs     = mi200_pc_sampling_config.data();
                    agent_info.num_pc_sampling_configs = mi200_pc_sampling_config.size();
                }
            }
        }
        else if(agent_info.type == ROCPROFILER_AGENT_TYPE_CPU)
        {
            agent_info.cu_count    = agent_info.cpu_cores_count;
            agent_info.vendor_name = strdup("CPU");
            for(const auto& itr : cpu_info_v)
            {
                if(agent_info.cpu_core_id_base == itr.apicid)
                {
                    agent_info.name         = strdup(itr.model_name.c_str());
                    agent_info.product_name = strdup(agent_info.name);
                    agent_info.family_id    = itr.family;
                    break;
                }
            }
        }

        if(properties.count("num_xcc") > 0)
            read_property(properties, "num_xcc", agent_info.num_xcc);
        else
            agent_info.num_xcc = 1;

        agent_info.max_waves_per_cu = agent_info.simd_per_cu * agent_info.max_waves_per_simd;

        if(agent_info.simd_arrays_per_engine > 0)
        {
            agent_info.num_shader_banks =
                agent_info.array_count / agent_info.simd_arrays_per_engine;

            // depends on above
            if(agent_info.num_shader_banks * agent_info.simd_arrays_per_engine > 0)
            {
                agent_info.cu_per_engine =
                    (agent_info.simd_count / agent_info.simd_per_cu) /
                    (agent_info.num_shader_banks * agent_info.simd_arrays_per_engine);
            }
        }

        agent_info.mem_banks = nullptr;
        agent_info.caches    = nullptr;
        agent_info.io_links  = nullptr;

        if(agent_info.mem_banks_count > 0)
        {
            agent_info.mem_banks = new rocprofiler_agent_mem_bank_t[agent_info.mem_banks_count];

            for(uint32_t i = 0; i < agent_info.mem_banks_count; ++i)
            {
                auto subproperties =
                    read_map(node_path / "mem_banks" / std::to_string(i) / "properties");

                read_property(subproperties, "heap_type", agent_info.mem_banks[i].heap_type);
                read_property(
                    subproperties, "size_in_bytes", agent_info.mem_banks[i].size_in_bytes);
                read_property(subproperties, "flags", agent_info.mem_banks[i].flags.MemoryProperty);
                read_property(subproperties, "width", agent_info.mem_banks[i].width);
                read_property(subproperties, "mem_clk_max", agent_info.mem_banks[i].mem_clk_max);
            }
        }

        if(agent_info.caches_count > 0)
        {
            agent_info.caches = new rocprofiler_agent_cache_t[agent_info.caches_count];

            for(uint32_t i = 0; i < agent_info.caches_count; ++i)
            {
                auto subproperties =
                    read_map(node_path / "caches" / std::to_string(i) / "properties");

                read_property(
                    subproperties, "processor_id_low", agent_info.caches[i].processor_id_low);
                read_property(subproperties, "level", agent_info.caches[i].level);
                read_property(subproperties, "size", agent_info.caches[i].size);
                read_property(
                    subproperties, "cache_line_size", agent_info.caches[i].cache_line_size);
                read_property(
                    subproperties, "cache_lines_per_tag", agent_info.caches[i].cache_lines_per_tag);
                read_property(subproperties, "association", agent_info.caches[i].association);
                read_property(subproperties, "latency", agent_info.caches[i].latency);
                read_property(subproperties, "type", agent_info.caches[i].type.Value);
            }
        }

        if(agent_info.io_links_count > 0)
        {
            agent_info.io_links = new rocprofiler_agent_io_link_t[agent_info.io_links_count];

            for(uint32_t i = 0; i < agent_info.io_links_count; ++i)
            {
                auto subproperties =
                    read_map(node_path / "io_links" / std::to_string(i) / "properties");

                read_property(subproperties, "type", agent_info.io_links[i].type);
                read_property(subproperties, "version_major", agent_info.io_links[i].version_major);
                read_property(subproperties, "version_minor", agent_info.io_links[i].version_minor);
                read_property(subproperties, "node_from", agent_info.io_links[i].node_from);
                read_property(subproperties, "node_to", agent_info.io_links[i].node_to);
                read_property(subproperties, "weight", agent_info.io_links[i].weight);
                read_property(subproperties, "min_latency", agent_info.io_links[i].min_latency);
                read_property(subproperties, "max_latency", agent_info.io_links[i].max_latency);
                read_property(subproperties, "min_bandwidth", agent_info.io_links[i].min_bandwidth);
                read_property(subproperties, "max_bandwidth", agent_info.io_links[i].max_bandwidth);
                read_property(subproperties,
                              "recommended_transfer_size",
                              agent_info.io_links[i].recommended_transfer_size);
                read_property(subproperties, "flags", agent_info.io_links[i].flags.LinkProperty);
            }
        }

        data.emplace_back(new rocprofiler_agent_t{agent_info}, [](rocprofiler_agent_t* ptr) {
            if(ptr)
            {
                auto free_cstring = [](const char*& val) {
                    if(val && ::strnlen(val, 1) > 0) ::free(const_cast<char*>(val));
                    val = "";
                };

                delete[] ptr->mem_banks;
                delete[] ptr->caches;
                delete[] ptr->io_links;
                free_cstring(ptr->name);
                free_cstring(ptr->vendor_name);
                free_cstring(ptr->product_name);
                free_cstring(ptr->model_name);
            }
            delete ptr;
        });
    }
    return data;
}

auto&
get_agent_topology()
{
    static auto _v = read_topology();
    return _v;
}

auto&
get_agent_caches()
{
    static auto _v = std::vector<hsa::AgentCache>{};
    return _v;
}
}  // namespace

std::vector<const rocprofiler_agent_t*>
get_agents()
{
    auto& agents   = rocprofiler::agent::get_agent_topology();
    auto  pointers = std::vector<const rocprofiler_agent_t*>{};
    pointers.reserve(agents.size());
    for(auto& agent : agents)
    {
        pointers.emplace_back(agent.get());
    }
    return pointers;
}

void
construct_agent_cache(::HsaApiTable* table)
{
    if(!table) return;

    auto rocp_agents = agent::get_agents();
    auto hsa_agents  = std::vector<hsa_agent_t>{};

    // Get HSA Agents
    table->core_->hsa_iterate_agents_fn(
        [](hsa_agent_t agent, void* data) {
            CHECK_NOTNULL(static_cast<std::vector<hsa_agent_t>*>(data))->emplace_back(agent);
            return HSA_STATUS_SUCCESS;
        },
        &hsa_agents);

    LOG_IF(FATAL, rocp_agents.size() != hsa_agents.size())
        << "Found " << rocp_agents.size() << " rocprofiler agents and " << hsa_agents.size()
        << " HSA agents";

    auto hsa_agent_node_map = std::unordered_map<uint32_t, hsa_agent_t>{};
    for(const auto& itr : hsa_agents)
    {
        if(uint32_t node_id = 0;
           table->core_->hsa_agent_get_info_fn(
               itr, static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_DRIVER_NODE_ID), &node_id) ==
           HSA_STATUS_SUCCESS)
        {
            hsa_agent_node_map[node_id] = itr;
        }
    }

    auto agent_map =
        std::unordered_map<uint32_t, std::tuple<const rocprofiler_agent_t*, hsa_agent_t>>{};
    for(const auto* ritr : rocp_agents)
    {
        for(auto hitr : hsa_agents)
        {
            if(uint32_t node_id = 0;
               table->core_->hsa_agent_get_info_fn(
                   hitr,
                   static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_DRIVER_NODE_ID),
                   &node_id) == HSA_STATUS_SUCCESS)
            {
                if(ritr->node_id == node_id)
                {
                    agent_map.emplace(ritr->node_id, std::make_tuple(ritr, hitr));
                    break;
                }
            }
        }
    }

    LOG_IF(ERROR, agent_map.size() != hsa_agents.size())
        << "rocprofiler was only able to map " << agent_map.size()
        << " rocprofiler agents to HSA agents, expected " << hsa_agents.size();

// For Pre-ROCm 6.0 releases
#if ROCPROFILER_HSA_RUNTIME_VERSION <= 100900
#    define HSA_AMD_AGENT_INFO_NEAREST_CPU 0xA113
#endif

    auto find_nearest_hsa_cpu_agent = [&table, &agent_map](uint32_t node_id) {
        auto _nearest_cpu = hsa_agent_t{.handle = 0};
        auto _hsa_agent   = std::get<1>(agent_map.at(node_id));
        if(table->core_->hsa_agent_get_info_fn(
               _hsa_agent,
               static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_NEAREST_CPU),
               &_nearest_cpu) != HSA_STATUS_SUCCESS)
        {
            const auto* _rocp_agent  = std::get<0>(agent_map.at(node_id));
            auto        distance_min = std::numeric_limits<int32_t>::max();
            for(uint32_t i = 0; i < _rocp_agent->io_links_count; ++i)
            {
                const auto& io_link = _rocp_agent->io_links[i];
                auto        _from   = io_link.node_from;
                auto        _to     = io_link.node_to;

                LOG_IF(FATAL, _from != node_id)
                    << "unexpected condition for node_id=" << node_id << ". io_link[" << i
                    << "].node_from=" << _from
                    << ". Expected this to match the node_id (node_to=" << _to << ")";

                if(agent_map.find(_to) == agent_map.end())
                {
                    LOG(WARNING) << "no agent mapping for io_link[" << i << "].node_to=" << _to
                                 << " in rocprofiler agent " << node_id;
                    continue;
                }

                auto [_to_rocp_agent, _to_hsa_agent] = agent_map.at(_to);
                auto _distance                       = std::abs(static_cast<int32_t>(_from - _to));
                if(_distance > 0 && _distance < distance_min &&
                   _to_rocp_agent->type == ROCPROFILER_AGENT_TYPE_CPU)
                {
                    distance_min = _distance;
                    _nearest_cpu = _to_hsa_agent;
                }
            }
        }
        return _nearest_cpu;
    };

    auto is_duplicate = [](const auto* agent_v) {
        for(const auto& itr : get_agent_caches())
        {
            if(itr == agent_v) return true;
        }
        return false;
    };

    // Generate supported agents
    for(const auto& itr : agent_map)
    {
        const auto* rocp_agent = std::get<0>(itr.second);
        auto        hsa_agent  = std::get<1>(itr.second);
        if(is_duplicate(rocp_agent)) continue;

        // AgentCache is only for GPU agents
        if(rocp_agent->type != ROCPROFILER_AGENT_TYPE_GPU) continue;

        auto _nearest_cpu = find_nearest_hsa_cpu_agent(itr.first);
        try
        {
            get_agent_caches().emplace_back(
                rocp_agent, hsa_agent, itr.first, _nearest_cpu, *table->amd_ext_);
        } catch(std::runtime_error& err)
        {
            if(rocp_agent->type == ROCPROFILER_AGENT_TYPE_GPU)
            {
                LOG(ERROR) << fmt::format("rocprofiler agent <-> HSA agent mapping failed: {} ({})",
                                          rocp_agent->node_id,
                                          err.what());
            }
        }
    }
}

std::optional<hsa_agent_t>
get_hsa_agent(const rocprofiler_agent_t* agent)
{
    for(const auto& itr : get_agent_caches())
    {
        if(itr == agent) return itr.get_hsa_agent();
    }

    return std::nullopt;
}

const rocprofiler_agent_t*
get_rocprofiler_agent(hsa_agent_t agent)
{
    for(const auto& itr : get_agent_caches())
    {
        if(itr == agent) return &itr.get_rocp_agent();
    }

    return nullptr;
}

std::optional<hsa::AgentCache>
get_agent_cache(const rocprofiler_agent_t* agent)
{
    for(const auto& itr : get_agent_caches())
    {
        if(itr == agent) return itr;
    }

    return std::nullopt;
}

std::optional<hsa::AgentCache>
get_agent_cache(hsa_agent_t agent)
{
    for(const auto& itr : get_agent_caches())
    {
        if(itr == agent) return itr;
    }

    return std::nullopt;
}
}  // namespace agent
}  // namespace rocprofiler

extern "C" {
rocprofiler_status_t
rocprofiler_query_available_agents(rocprofiler_available_agents_cb_t callback,
                                   size_t                            agent_size,
                                   void*                             user_data)
{
    if(agent_size > sizeof(rocprofiler_agent_t))
    {
        LOG(ERROR) << "rocprofiler_agent_t used by caller is ABI-incompatible with "
                      "rocprofiler_agent_t in rocprofiler";
        return ROCPROFILER_STATUS_ERROR_INCOMPATIBLE_ABI;
    }

    auto&& pointers = rocprofiler::agent::get_agents();
    return callback(pointers.data(), pointers.size(), user_data);
}
}
