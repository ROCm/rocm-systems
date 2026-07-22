// MIT License
//
// Copyright (c) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
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
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "lib/rocprofiler-sdk/platform/gnulinux/agent.hpp"

#include "lib/common/environment.hpp"
#include "lib/common/filesystem.hpp"
#include "lib/common/logging.hpp"
#include "lib/common/string_entry.hpp"
#include "lib/common/utility.hpp"
#include "lib/rocprofiler-sdk/agent.hpp"

#include <rocprofiler-sdk/agent.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/cxx/details/tokenize.hpp>

#include <fmt/core.h>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <libdrm/amdgpu.h>
#include <xf86drm.h>

#include <array>
#include <cstring>
#include <fstream>
#include <limits>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace rocprofiler
{
namespace platform
{
namespace gnulinux
{
namespace
{
namespace fs = ::rocprofiler::common::filesystem;

using ::rocprofiler::agent::get_agent_available_properties;
using ::rocprofiler::agent::update_agent_runtime_visibility;
using ::rocprofiler::agent::uuid_view_t;

// Random per-process offset applied to rocprofiler_agent_id_t.handle so that
// agent IDs are not stable integers across runs. Helps catch consumers that
// accidentally hard-code an ID instead of looking it up via the agent table.
uint64_t
get_agent_offset()
{
    static uint64_t _v = []() {
        auto gen = std::mt19937{std::random_device{}()};
        auto rng = std::uniform_int_distribution<uint64_t>{std::numeric_limits<uint8_t>::max(),
                                                           std::numeric_limits<uint16_t>::max()};
        return rng(gen);
    }();
    return _v;
}

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

            line = sdk::parse::strip(std::move(line), " \t\n\v\f\r");
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
            auto match = sdk::parse::tokenize(itr, std::vector<std::string_view>{": "});
            if(match.size() >= 2)
            {
                auto get_stol = [_label = std::string_view{itr}](const auto& _value) -> long {
                    try
                    {
                        return std::stol(_value);
                    } catch(std::exception& e)
                    {
                        ROCP_CI_LOG(WARNING) << fmt::format("rocprofiler-sdk agent encountered "
                                                            "error while parsing CPU info '{}': {}",
                                                            _label,
                                                            e.what());
                    }
                    return 0;
                };

                // For cases with multiple colons, join all tokens after the first one
                // e.g. "model name : AMD EPYC : 100-000000248" split into
                // ["model name", "AMD EPYC", "100-000000248"] with the last two tokens joined
                // back together with ": "
                std::string value;
                if(match.size() == 2)
                {
                    value = match.back();
                }
                else
                {
                    // Join all tokens after the first one with ": " separator
                    for(size_t i = 1; i < match.size(); ++i)
                    {
                        if(i > 1) value += ": ";
                        value += match[i];
                    }
                }

                if(itr.find("vendor_id") == 0)
                    info_v.vendor_id = value;
                else if(itr.find("model name") == 0)
                {
                    info_v.model_name = value;
                    // Remove leading and trailing whitespaces
                    info_v.model_name =
                        sdk::parse::strip(std::string{info_v.model_name}, " \t\n\v\f\r");
                }
                else if(itr.find("processor") == 0)
                    info_v.processor = get_stol(value);
                else if(itr.find("cpu family") == 0)
                    info_v.family = get_stol(value);
                else if(itr.find("model") == 0 && itr.find("model name") != 0)
                    info_v.model = get_stol(value);
                else if(itr.find("physical id") == 0)
                    info_v.physical_id = get_stol(value);
                else if(itr.find("core id") == 0)
                    info_v.core_id = get_stol(value);
                else if(itr.find("apicid") == 0)
                    info_v.apicid = get_stol(value);
            }
            else
            {
                // Each processor_block is grouped by the presence of an empty line in /proc/cpuinfo
                // so no checks for empty lines are performed inside this loop. If an empty line is
                // found, that should be considered an error. Entries like "power management:" with
                // no info (i.e. where the ":" is the last character on the line) can be ignored
                auto last_colon_pos = itr.find_last_of(':');
                ROCP_CI_LOG_IF(
                    INFO, last_colon_pos < itr.length() && (last_colon_pos + 1) != itr.length())
                    << fmt::format("Encountered unexpected /proc/cpuinfo line format: '{}'", itr);
            }
        }

        if(info_v.is_valid())
            processor_info.emplace_back(info_v);
        else
        {
            ROCP_ERROR << "Invalid processor info: "
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
    ROCP_ERROR_IF(ec) << fmt::format(
        "Error getting status for file '{}': {}", fpath.string(), ec.message());
    return (!ec && (perms & fs::perms::owner_read) != fs::perms::none);
}

auto
read_file(const std::string& fname)
{
    auto data = std::vector<std::string>{};

    if(!is_readable(fs::path{fname}))
    {
        ROCP_CI_LOG(WARNING) << fmt::format("file '{}' cannot be read", fname);
        return data;
    }

    auto ifs = std::ifstream{fname};
    if(!ifs || !ifs.good())
    {
        ROCP_CI_LOG(WARNING) << fmt::format("file '{}' cannot be read", fname);
        return data;
    }

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
    {
        ROCP_CI_LOG(WARNING) << fmt::format("file '{}' cannot be read", fname);
        return data;
    }

    auto ifs = std::ifstream(fname);

    if(!ifs || !ifs.good())
    {
        ROCP_CI_LOG(WARNING) << fmt::format("file '{}' cannot be read", fname);
        return data;
    }
    auto last_label = std::string{};
    while(ifs && ifs.good())
    {
        auto label = std::string{};
        ifs >> label;
        if(ifs.fail() || ifs.eof() || label.empty()) break;

        auto entry = std::string{};
        ifs >> entry;
        if(ifs.fail() || ifs.eof())
        {
            ROCP_CI_LOG(WARNING) << fmt::format(
                "unexpected file format in '{}' at {}", fname, label);
            continue;
        }

        auto ret = data.emplace(label, entry);
        if(!ret.second)
        {
            ROCP_CI_LOG(WARNING) << fmt::format(
                "duplicate entry in '{}': '{}' (='{}'). last label was '{}'",
                fname,
                label,
                entry,
                last_label);
            continue;
        }

        if(!label.empty()) last_label = std::move(label);
    }

    return data;
}

template <typename MapT, typename Tp>
void
read_property(const MapT& data, const std::string& label, Tp& value)
{
    using mutable_type = std::remove_const_t<Tp>;

    get_agent_available_properties().insert(label);
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
            ROCP_ERROR << "agent properties map missing " << label << " entry";
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
            ROCP_CI_LOG(WARNING) << fmt::format(
                "data with label {} has a value (={}) which is less "
                "than the min value for the type (={})",
                label,
                local_value,
                min_value);
            return;
        }
        else if(local_value > max_value)
        {
            ROCP_CI_LOG(WARNING) << fmt::format("data with label {} has a value (={}) which is "
                                                "greater than the max value for the type (={})",
                                                label,
                                                local_value,
                                                max_value);
            return;
        }

        if constexpr(std::is_const<Tp>::value)
            const_cast<mutable_type&>(value) = static_cast<mutable_type>(local_value);
        else
            value = static_cast<Tp>(local_value);
    }
}

// Candidate locations for the KFD sysfs topology root, in priority order.
// Env-var overrides come first so a test fixture or alternate mount can
// supersede the canonical /sys paths without rebuilding. Returned by value;
// callers iterate it directly so the temporary strings outlive the loop.
std::array<std::string, 5>
sysfs_node_candidates()
{
    return {
        common::get_env("ROCPROFILER_KFD_TOPOLOGY", ""),  // rocprofiler-sdk specific
        common::get_env("AMD_KFD_TOPOLOGY", ""),          // universal AMD KFD topology env var
        common::get_env("HSA_MODEL_TOPOLOGY", ""),        // HSA specific env var
        std::string{"/sys/devices/virtual/kfd/kfd/topology/nodes"},
        std::string{"/sys/class/kfd/kfd/topology/nodes"},
    };
}
}  // namespace

bool
is_available()
{
    for(const std::string& p : sysfs_node_candidates())
    {
        if(!p.empty() && fs::exists(fs::path{p}) && fs::is_directory(fs::path{p})) return true;
    }
    return false;
}

std::vector<unique_agent_t>
enumerate()
{
    auto data = std::vector<unique_agent_t>{};

    auto _sysfs_nodes_path = std::optional<fs::path>{};
    for(const auto& itr : sysfs_node_candidates())
    {
        if(!itr.empty() && fs::exists(fs::path{itr}) && fs::is_directory(fs::path{itr}))
        {
            if(!_sysfs_nodes_path)
            {
                ROCP_INFO << fmt::format("Using KFD topology path '{}'", itr);
                _sysfs_nodes_path = fs::path{itr};
            }
        }
    }

    auto sysfs_nodes_path =
        _sysfs_nodes_path.value_or(fs::path{"/sys/class/kfd/kfd/topology/nodes"});

    if(!fs::exists(sysfs_nodes_path))
    {
        ROCP_WARNING << fmt::format("sysfs nodes path '{}' does not exist",
                                    sysfs_nodes_path.string());

        return data;
    }

    const auto& cpu_info_v = get_cpu_info();
    uint64_t    idcount    = 0;
    uint64_t    nodecount  = 0;
    uint64_t    cpucount   = 0;
    uint64_t    gpucount   = 0;
    uint64_t    unkcount   = 0;

    while(true)
    {
        auto node_id   = nodecount++;
        auto node_path = sysfs_nodes_path / std::to_string(node_id);
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
            ROCP_ERROR << "Error reading '" << (node_path / "properties").string()
                       << "' :: " << e.what();
            continue;
        }

        // we may have been able to open the properties file but if it was empty, we ignore it
        if(properties.empty()) continue;

        auto agent_info                 = common::init_public_api_struct(rocprofiler_agent_t{});
        agent_info.type                 = ROCPROFILER_AGENT_TYPE_NONE;
        agent_info.logical_node_id      = idcount++;
        agent_info.node_id              = node_id;
        agent_info.id.handle            = (agent_info.logical_node_id) + get_agent_offset();
        agent_info.logical_node_type_id = -1;

        if(!name_prop.empty())
            agent_info.model_name =
                common::get_string_entry(fmt::format("{}", fmt::join(name_prop, " ")))->c_str();
        else
            agent_info.model_name = "";

        if(!gpu_id_prop.empty())
        {
            try
            {
                agent_info.gpu_id = std::stoull(gpu_id_prop.front());
            } catch(std::exception& e)
            {
                ROCP_CI_LOG(WARNING) << fmt::format("rocprofiler-sdk agent encountered error while "
                                                    "parsing gpu id property '{}': {}",
                                                    gpu_id_prop.front(),
                                                    e.what());
            }
        }

        read_property(properties, "cpu_cores_count", agent_info.cpu_cores_count);
        read_property(properties, "simd_count", agent_info.simd_count);

        if(agent_info.cpu_cores_count > 0 && agent_info.simd_count == 0)
        {
            // CPU cores and no SIMDs indicates a CPU agent
            agent_info.type = ROCPROFILER_AGENT_TYPE_CPU;
        }
        else if(agent_info.simd_count > 0 && agent_info.cpu_cores_count == 0)
        {
            // SIMDs and no CPU cores indicates a (discrete) GPU agent
            agent_info.type = ROCPROFILER_AGENT_TYPE_GPU;
        }
        else if(agent_info.cpu_cores_count > 0 && agent_info.simd_count > 0)
        {
            // SIMDs and CPU cores indicates an APU but we are marking this as an GPU for the
            // purposes of rocprofiler-sdk.
            agent_info.type = ROCPROFILER_AGENT_TYPE_GPU;
        }
        else
        {
            ROCP_CI_LOG(WARNING) << fmt::format(
                "agent-{} is neither a CPU nor a GPU. CPU cores: {}, SIMD count: {}",
                agent_info.node_id,
                agent_info.cpu_cores_count,
                agent_info.simd_count);
        }

        if(agent_info.type == ROCPROFILER_AGENT_TYPE_CPU)
            agent_info.logical_node_type_id = cpucount++;
        else if(agent_info.type == ROCPROFILER_AGENT_TYPE_GPU)
            agent_info.logical_node_type_id = gpucount++;
        else
            agent_info.logical_node_type_id = unkcount++;

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
        memset(&agent_info.uuid.bytes, 0, sizeof(agent_info.uuid.bytes));
        if(agent_info.type == ROCPROFILER_AGENT_TYPE_GPU)
        {
            constexpr auto workgrp_max = 1024;
            constexpr auto grid_max    = std::numeric_limits<int32_t>::max();
            constexpr auto grid_max_x  = std::numeric_limits<int32_t>::max();
            constexpr auto grid_max_y  = std::numeric_limits<uint16_t>::max();
            constexpr auto grid_max_z  = std::numeric_limits<uint16_t>::max();

            auto     _uuid    = uuid_view_t{};
            uint64_t uuid_val = 0;
            read_property(properties, "unique_id", uuid_val);
            _uuid.value64[0] = uuid_val;
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
            agent_info.grid_max_dim       = {grid_max_x, grid_max_y, grid_max_z};
            agent_info.cu_count           = agent_info.simd_count / agent_info.simd_per_cu;

            // fallback in case drmOpenRender or amdgpu_device_initialize fails
            auto _set_default_agent_names = [&agent_info]() {
                if(agent_info.gfx_target_version >= 10000)
                {
                    ROCP_INFO << fmt::format(
                        "Setting default agent names for agent-{} with gfx target version {}",
                        agent_info.node_id,
                        agent_info.gfx_target_version);

                    auto major = (agent_info.gfx_target_version / 10000) % 100;
                    auto minor = (agent_info.gfx_target_version / 100) % 100;
                    auto step  = (agent_info.gfx_target_version % 100);
                    agent_info.name =
                        common::get_string_entry(fmt::format("gfx{}{}{:x}", major, minor, step))
                            ->c_str();
                    agent_info.product_name = common::get_string_entry("unknown")->c_str();
                    agent_info.vendor_name  = common::get_string_entry("AMD")->c_str();
                }
                else
                {
                    ROCP_INFO << fmt::format(
                        "Failed to set default agent names for agent-{} with gfx target version "
                        "{}. Requires gfx target version >= 10000.",
                        agent_info.node_id,
                        agent_info.gfx_target_version);
                }
            };

            agent_info.uuid = static_cast<rocprofiler_uuid_t>(_uuid);
            if(int drm_fd = 0; (drm_fd = drmOpenRender(agent_info.drm_render_minor)) >= 0)
            {
                ROCP_TRACE << fmt::format(
                    "Successful drmOpenRender for agent-{} using drm render minor {}... fd={}",
                    agent_info.node_id,
                    agent_info.drm_render_minor,
                    drm_fd);

                uint32_t major_version = 0;
                uint32_t minor_version = 0;
                auto*    device_handle = amdgpu_device_handle{};
                if(amdgpu_device_initialize(
                       drm_fd, &major_version, &minor_version, &device_handle) == 0)
                {
                    ROCP_INFO << fmt::format(
                        "Initialized amdgpu device handle for agent-{} using drm render minor {} "
                        "(fd={}). amdgpu driver version {}.{}",
                        agent_info.node_id,
                        agent_info.drm_render_minor,
                        drm_fd,
                        major_version,
                        minor_version);

                    auto major = (agent_info.gfx_target_version / 10000) % 100;
                    auto minor = (agent_info.gfx_target_version / 100) % 100;
                    auto step  = (agent_info.gfx_target_version % 100);

                    agent_info.name =
                        common::get_string_entry(fmt::format("gfx{}{}{:x}", major, minor, step))
                            ->c_str();

                    const char* marketing_name = amdgpu_get_marketing_name(device_handle);
                    if(marketing_name == nullptr) marketing_name = "unknown";

                    agent_info.product_name = common::get_string_entry(marketing_name)->c_str();
                    agent_info.vendor_name  = common::get_string_entry("AMD")->c_str();

                    amdgpu_gpu_info gpu_info = {};
                    if(amdgpu_query_gpu_info(device_handle, &gpu_info) == 0)
                    {
                        agent_info.family_id = gpu_info.family_id;
                    }
                    amdgpu_device_deinitialize(device_handle);
                }
                else
                {
                    _set_default_agent_names();
                }
                drmClose(drm_fd);
            }
            else
            {
                _set_default_agent_names();
            }
        }
        else if(agent_info.type == ROCPROFILER_AGENT_TYPE_CPU)
        {
            agent_info.cu_count    = agent_info.cpu_cores_count;
            agent_info.vendor_name = common::get_string_entry("CPU")->c_str();
            for(const auto& itr : cpu_info_v)
            {
                if(agent_info.cpu_core_id_base == itr.apicid)
                {
                    agent_info.name         = common::get_string_entry(itr.model_name)->c_str();
                    agent_info.product_name = common::get_string_entry(agent_info.name)->c_str();
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
            if(agent_info.num_shader_banks > 0)
            {
                agent_info.cu_per_engine = (agent_info.simd_count / agent_info.simd_per_cu) /
                                           (agent_info.num_shader_banks);
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

        update_agent_runtime_visibility(agent_info);

        data.emplace_back(new rocprofiler_agent_t{agent_info}, [](rocprofiler_agent_t* ptr) {
            if(ptr)
            {
                delete[] ptr->mem_banks;
                delete[] ptr->caches;
                delete[] ptr->io_links;
            }
            delete ptr;
        });
    }
    return data;
}

}  // namespace gnulinux
}  // namespace platform
}  // namespace rocprofiler
