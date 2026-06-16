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

#include "lib/rocprofiler-sdk/agent.hpp"
#include "lib/aqlprofile/aqlprofile.hpp"
#include "lib/common/environment.hpp"
#include "lib/common/logging.hpp"
#include "lib/common/scope_destructor.hpp"
#include "lib/common/static_object.hpp"
#include "lib/common/string_entry.hpp"
#include "lib/common/utility.hpp"
#include "lib/rocprofiler-sdk/hsa/agent_cache.hpp"
#include "lib/rocprofiler-sdk/platform/agent.hpp"
#ifdef _WIN32
#    include "lib/rocprofiler-sdk/platform/windows/agent.hpp"
#else
#    include "lib/rocprofiler-sdk/platform/gnulinux/agent.hpp"
#    include "lib/rocprofiler-sdk/platform/wsl/agent.hpp"
#endif

#include <rocprofiler-sdk/agent.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/cxx/details/tokenize.hpp>

#include <fmt/core.h>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <hsa/hsa.h>
#include <hsa/hsa_api_trace.h>

#include <iomanip>
#include <limits>
#include <set>
#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace rocprofiler
{
namespace agent
{
namespace
{
struct bdf_info
{
    uint32_t domain{};
    uint8_t  bus{};
    uint8_t  device{};
    uint8_t  function{};
};

}  // namespace

void
update_agent_runtime_visibility(rocprofiler_agent_t& agent_info)
{
    //
    //      https://rocm.docs.amd.com/en/latest/conceptual/gpu-isolation.html
    //
    //
    // ROCR_VISIBLE_DEVICES
    //
    //      A list of device indices or UUIDs that will be exposed to applications.
    //
    //      Runtime : ROCm Software Runtime. Applies to all applications using the user mode
    //      ROCm software stack.
    //
    //      Example to expose the 1. device and a device based on UUID.
    //          export ROCR_VISIBLE_DEVICES="0,GPU-DEADBEEFDEADBEEF"
    //
    // GPU_DEVICE_ORDINAL
    //      Devices indices exposed to OpenCL and HIP applications.
    //
    //      Runtime : ROCm Compute Language Runtime (ROCclr). Applies to applications and
    //      runtimes using the ROCclr abstraction layer including HIP and OpenCL applications.
    //
    //      Example to expose the 1. and 3. device in the system.
    //          export GPU_DEVICE_ORDINAL="0,2"
    //
    // HIP_VISIBLE_DEVICES
    //      Device indices exposed to HIP applications.
    //
    //      Runtime: HIP runtime. Applies only to applications using HIP on the AMD platform.
    //
    //      Example to expose the 1. and 3. devices in the system.
    //          export HIP_VISIBLE_DEVICES="0,2"
    //
    // CUDA_VISIBLE_DEVICES
    //      Provided for CUDA compatibility, has the same effect as HIP_VISIBLE_DEVICES on the
    //      AMD platform.
    //
    //      Runtime : HIP or CUDA Runtime. Applies to HIP applications on the AMD or NVIDIA
    //      platform and CUDA applications.
    //
    // OMP_DEFAULT_DEVICE
    //      Default device used for OpenMP target offloading.
    //
    //      Runtime : OpenMP Runtime. Applies only to applications using OpenMP offloading.
    //
    //      Example on setting the default device to the third device.
    //          export OMP_DEFAULT_DEVICE="2"
    //

    struct parse_result
    {
        bool    value = false;
        int32_t index = -1;

        operator bool() const { return (value && index >= 0); }
    };

    constexpr auto zero_visibility = rocprofiler_agent_runtime_visiblity_t{
        .hsa = 0, .hip = 0, .rccl = 0, .rocdecode = 0, .reserved = 0};
    constexpr auto full_visibility = rocprofiler_agent_runtime_visiblity_t{
        .hsa = 1, .hip = 1, .rccl = 1, .rocdecode = 1, .reserved = 0};

    agent_info.runtime_visibility = zero_visibility;

    if(agent_info.type == ROCPROFILER_AGENT_TYPE_CPU)
    {
        agent_info.runtime_visibility = full_visibility;
    }
    else if(agent_info.type == ROCPROFILER_AGENT_TYPE_GPU)
    {
        auto set_hip_visibility = [&agent_info](bool is_hip_visible) {
            if(is_hip_visible && agent_info.runtime_visibility.hsa == 0)
            {
                ROCP_WARNING << fmt::format(
                    "Attempt to enable hip visibility for agent-{} which is "
                    "not visible to HSA (ROCR)",
                    agent_info.node_id);
                return;
            }

            ROCP_INFO << "agent-" << agent_info.node_id
                      << " ::  HIP_VISIBLE_DEVICE = " << std::boolalpha << is_hip_visible;
            agent_info.runtime_visibility.hip       = is_hip_visible;
            agent_info.runtime_visibility.rccl      = is_hip_visible;
            agent_info.runtime_visibility.rocdecode = is_hip_visible;
        };

        auto set_hsa_visibility = [&agent_info, &set_hip_visibility](bool is_hsa_visible) {
            ROCP_INFO << "agent-" << agent_info.node_id
                      << " :: ROCR_VISIBLE_DEVICE = " << std::boolalpha << is_hsa_visible;
            agent_info.runtime_visibility.hsa = is_hsa_visible;
            if(!is_hsa_visible) set_hip_visibility(false);
        };

        auto parse_env_visible = [&agent_info](std::string_view env_varname,
                                               int32_t env_node_id) -> std::optional<parse_result> {
            constexpr auto uuid_prefix = std::string_view{"GPU-"};
            auto           env_value   = common::get_env(env_varname, "");
            if(env_value.empty()) return std::nullopt;

            ROCP_INFO << "Found visibility environment variable :: " << env_varname << " = "
                      << env_value;
            int32_t idx = 0;
            for(const auto& itr : rocprofiler::sdk::parse::tokenize(env_value, ", "))
            {
                if(itr.empty()) continue;

                ROCP_TRACE << "Processing " << env_varname << " token: " << itr;

                auto _idx_v = idx++;
                if(itr.find_first_not_of("0123456789") == std::string::npos)
                {
                    auto _ordinal = std::stoll(itr);
                    if(_ordinal == env_node_id) return parse_result{true, _idx_v};
                }
                else if(itr.find(uuid_prefix) == 0 && itr.length() > uuid_prefix.length())
                {
                    auto _uuid =
                        std::strtoull(itr.substr(uuid_prefix.length()).c_str(), nullptr, 16);
                    auto uuid_view = uuid_view_t{agent_info.uuid};
                    if(_uuid == uuid_view.value64[0]) return parse_result{true, _idx_v};
                }
                else
                {
                    ROCP_CI_LOG(WARNING)
                        << fmt::format("Sequence '{}' in {}={} not recognized. Expected device "
                                       "ordinal or GPU-XXX where XXX is the hexadecimal UUID",
                                       itr,
                                       env_varname,
                                       env_value);
                }
            }
            return parse_result{false, agent_info.logical_node_type_id};
        };

        static_assert(
            ROCPROFILER_LIBRARY_LAST == ROCPROFILER_ROCJPEG_LIBRARY,
            "Since a new library was added to rocprofiler_runtime_library_t, please make sure "
            "rocprofiler_agent_runtime_visiblity_t has an entry for this library (if "
            "necessary) and make the necessary updates to the logic below has been updated");

        std::string_view hip_visible_envvar = "HIP_VISIBLE_DEVICES";

        auto rocr_visible =
            parse_env_visible("ROCR_VISIBLE_DEVICES", agent_info.logical_node_type_id);

        auto rocr_index =
            (rocr_visible && *rocr_visible) ? rocr_visible->index : agent_info.logical_node_type_id;

        ROCP_INFO << fmt::format("agent-{} (GPU {}) has a rocr index = {}",
                                 agent_info.node_id,
                                 agent_info.logical_node_type_id,
                                 rocr_index);

        auto hip_visible = parse_env_visible(hip_visible_envvar, rocr_index);

        auto parse_hip_visible_alt = [&hip_visible, &agent_info, &rocr_index, &parse_env_visible](
                                         std::string_view env_primary,
                                         std::string_view env_secondary) {
            auto secondary_visible = parse_env_visible(env_secondary, rocr_index);
            if(secondary_visible && !hip_visible)
            {
                hip_visible = secondary_visible;
                return env_secondary;
            }
            else if(secondary_visible && hip_visible && *secondary_visible != *hip_visible)
            {
                ROCP_CI_LOG(WARNING) << fmt::format("Conflicting visibility of agent-{} between "
                                                    "{} and {}. Assuming {} supersedes {}",
                                                    agent_info.node_id,
                                                    env_primary,
                                                    env_secondary,
                                                    env_primary,
                                                    env_secondary);
            }
            return env_primary;
        };

        // if HIP_VISIBLE_DEVICES is not set, fall back on these
        hip_visible_envvar = parse_hip_visible_alt(hip_visible_envvar, "CUDA_VISIBLE_DEVICES");
        hip_visible_envvar = parse_hip_visible_alt(hip_visible_envvar, "GPU_DEVICE_ORDINAL");

        if(!hip_visible && !rocr_visible)
        {
            set_hsa_visibility(true);
            set_hip_visibility(true);
        }
        else
        {
            ROCP_INFO << "agent-" << agent_info.node_id
                      << " :: logical node type id: " << agent_info.logical_node_type_id;

            if(rocr_visible)
                set_hsa_visibility(*rocr_visible);
            else
                set_hsa_visibility(true);

            if(hip_visible)
                set_hip_visibility(*hip_visible);
            else
                set_hip_visibility((rocr_visible) ? rocr_visible->value : true);
        }
    }
    else
    {
        ROCP_CI_LOG(WARNING) << "Agent-" << agent_info.node_id
                             << " has unexpected agent type value " << agent_info.type
                             << " passed to " << __FUNCTION__;
    }
}

namespace
{
using unique_agent_t = ::rocprofiler::platform::unique_agent_t;

// Selects the platform enumerator at runtime.
//
// Linux builds compile only gnulinux/ and wsl/ (the WIN32 branch of
// platform/CMakeLists.txt swaps in windows/), so only the enumerators built
// into this binary are candidates here.
//
// Precedence on Linux:
//   1. ROCPROFILER_FORCE_PLATFORM={gnulinux|wsl} overrides autodetect.
//      Values not built into this binary are logged and ignored.
//   2. gnulinux::is_available() (KFD sysfs present) - the common bare-metal
//      Linux case, kept first to preserve existing behaviour byte-for-byte.
//   3. wsl::is_available() (/dev/dxg + libdxcore.so) - WSL guest with the
//      DXCore driver shimmed in.
//   4. Fallback: gnulinux enumerator (will log and return an empty vector).
//
// Windows builds collapse to a single candidate (platform::windows).
std::vector<unique_agent_t>
enumerate_platform_agents()
{
    const auto forced = common::get_env("ROCPROFILER_FORCE_PLATFORM", std::string{});
    if(!forced.empty())
    {
#ifndef _WIN32
        if(forced == "gnulinux")
        {
            ROCP_INFO << "agent topology: forced gnulinux via ROCPROFILER_FORCE_PLATFORM";
            return platform::gnulinux::enumerate();
        }
        if(forced == "wsl")
        {
            ROCP_INFO << "agent topology: forced wsl via ROCPROFILER_FORCE_PLATFORM";
            return platform::wsl::enumerate();
        }
#endif
#ifdef _WIN32
        if(forced == "windows")
        {
            ROCP_INFO << "agent topology: forced windows via ROCPROFILER_FORCE_PLATFORM";
            return platform::windows::enumerate();
        }
#endif
        ROCP_WARNING << fmt::format(
            "agent topology: ROCPROFILER_FORCE_PLATFORM='{}' is not built into this binary "
            "(expected gnulinux|wsl on Linux, windows on Windows); falling back to autodetect",
            forced);
    }

#ifndef _WIN32
    if(platform::gnulinux::is_available())
    {
        ROCP_INFO << "agent topology: selected " << platform::gnulinux::name << " (sysfs present)";
        return platform::gnulinux::enumerate();
    }
    if(platform::wsl::is_available())
    {
        ROCP_INFO << "agent topology: selected " << platform::wsl::name
                  << " (libdxcore.so present)";
        return platform::wsl::enumerate();
    }
    ROCP_WARNING << "agent topology: no platform matched; falling back to "
                 << platform::gnulinux::name << " (will return empty)";
    return platform::gnulinux::enumerate();
#else
    if(platform::windows::is_available())
    {
        ROCP_INFO << "agent topology: selected " << platform::windows::name;
        return platform::windows::enumerate();
    }
    ROCP_WARNING << "agent topology: no platform matched; falling back to "
                 << platform::windows::name << " (will return empty)";
    return platform::windows::enumerate();
#endif
}

auto&
get_agent_topology()
{
    static auto*& _v =
        common::static_object<std::vector<unique_agent_t>>::construct(enumerate_platform_agents());
    return *CHECK_NOTNULL(_v);
}

auto&
get_agent_caches()
{
    static auto*& _v = common::static_object<std::vector<hsa::AgentCache>>::construct();
    return *CHECK_NOTNULL(_v);
}

struct agent_pair
{
    const rocprofiler_agent_t* rocp_agent = nullptr;
    hsa_agent_t                hsa_agent  = {};
};

auto&
get_agent_mapping()
{
    static auto*& _v = common::static_object<std::vector<agent_pair>>::construct();
    return *CHECK_NOTNULL(_v);
}

bdf_info
get_bdf_info(const rocprofiler_agent_t* agent)
{
    // location_id encodes PCI Bus/Device/Function (BDF) as a 16-bit value:
    //   bits [15:8] = bus number
    //   bits  [7:3] = device number
    //   bits  [2:0] = function number
    return {.domain   = agent->domain,
            .bus      = static_cast<uint8_t>((agent->location_id >> 8) & 0xFF),
            .device   = static_cast<uint8_t>((agent->location_id >> 3) & 0x1F),
            .function = static_cast<uint8_t>(agent->location_id & 0x07)};
}

const std::vector<aqlprofile_agent_handle_t>&
get_aql_handles()
{
    static auto*& _v =
        common::static_object<std::vector<aqlprofile_agent_handle_t>>::construct([]() {
            std::vector<aqlprofile_agent_handle_t> agent_handles;

            for(auto& agent : get_agents())
            {
                aqlprofile_agent_handle_t handle = {.handle = 0};

                const auto bdf = get_bdf_info(agent);
                common::consume_args(bdf);

#if ROCPROFILER_EXTERNAL_AQLPROFILE
                ROCP_TRACE << fmt::format(
                    "Registering agent {} with external aqlprofile (libhsa-amd-aqlprofile64.so)",
                    agent->name);

                aqlprofile_agent_info_t agent_info = {
                    .agent_gfxip          = agent->name,
                    .xcc_num              = agent->num_xcc,
                    .se_num               = agent->num_shader_banks,
                    .cu_num               = agent->cu_count,
                    .shader_arrays_per_se = agent->simd_arrays_per_engine};

                if(aqlprofile_register_agent(&handle, &agent_info) != HSA_STATUS_SUCCESS)
                {
                    ROCP_WARNING << "Failed to register agent " << agent->name;
                }
#else

                ROCP_TRACE << fmt::format(
                    "Registering agent {:04x}:{:02x}:{:02x}.{:x} :: {} with IP discovery",
                    bdf.domain,
                    bdf.bus,
                    bdf.device,
                    bdf.function,
                    agent->name);

                aqlprofile_agent_info_v1_t agent_info = {
                    .agent_gfxip          = agent->name,
                    .xcc_num              = agent->num_xcc,
                    .se_num               = agent->num_shader_banks,
                    .cu_num               = agent->cu_count,
                    .shader_arrays_per_se = agent->simd_arrays_per_engine,
                    .domain               = agent->domain,
                    .location_id          = agent->location_id,
                };

                if(aqlprofile_register_agent_info(
                       &handle, &agent_info, AQLPROFILE_AGENT_VERSION_V1) != HSA_STATUS_SUCCESS)
                {
                    ROCP_WARNING << fmt::format(
                        "Failed to register agent {:04x}:{:02x}:{:02x}.{:x} :: {}",
                        bdf.domain,
                        bdf.bus,
                        bdf.device,
                        bdf.function,
                        agent->name);
                }
#endif
                agent_handles.push_back(handle);
            }
            return agent_handles;
        }());

    return *CHECK_NOTNULL(_v);
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

const rocprofiler_agent_t*
get_agent(rocprofiler_agent_id_t id)
{
    for(const auto& itr : get_agents())
    {
        if(itr && itr->id.handle == id.handle) return itr;
    }
    return nullptr;
}

const aqlprofile_agent_handle_t*
get_aql_agent(rocprofiler_agent_id_t id)
{
    size_t pos = 0;
    for(const auto& itr : get_agents())
    {
        if(itr && itr->id.handle == id.handle)
        {
            return &get_aql_handles().at(pos);
        }
        pos++;
    }
    return nullptr;
}

void
construct_agent_cache(::HsaApiTable* table)
{
    if(!table) return;

    auto rocp_agents = agent::get_agents();
    auto hsa_agents  = std::vector<hsa_agent_t>{};

    auto get_hsa_status_string = [table](hsa_status_t _status) -> std::string_view {
        const char* _status_msg = nullptr;
        return (table->core_->hsa_status_string_fn(_status, &_status_msg) == HSA_STATUS_SUCCESS &&
                _status_msg)
                   ? std::string_view{_status_msg}
                   : std::string_view{"(unknown HSA error)"};
    };

    {
        auto _hsa_agents = std::vector<hsa_agent_t>{};

        // Get HSA Agents
        table->core_->hsa_iterate_agents_fn(
            [](hsa_agent_t agent, void* data) {
                CHECK_NOTNULL(static_cast<std::vector<hsa_agent_t>*>(data))->emplace_back(agent);
                return HSA_STATUS_SUCCESS;
            },
            &_hsa_agents);

        // remove agents that are not CPU or GPU
        for(const auto& _agent : _hsa_agents)
        {
            hsa_device_type_t agent_type{};

            auto ret = table->core_->hsa_agent_get_info_fn(
                _agent, hsa_agent_info_t{HSA_AGENT_INFO_DEVICE}, &agent_type);

            if(ret != HSA_STATUS_SUCCESS)
            {
                ROCP_CI_LOG(ERROR) << fmt::format(
                    "hsa_agent_get_info(hsa_agent_t={}, "
                    "HSA_AGENT_INFO_DEVICE, ...) returned {} :: {}, skipping this agent",
                    _agent.handle,
                    static_cast<std::underlying_type_t<hsa_status_t>>(ret),
                    get_hsa_status_string(ret));
                continue;
            }

            if(agent_type == HSA_DEVICE_TYPE_CPU || agent_type == HSA_DEVICE_TYPE_GPU)
            {
                hsa_agents.emplace_back(_agent);
            }
        }
    }

    ROCP_CI_LOG_IF(ERROR, hsa_agents.empty()) << fmt::format("Did not detect any HSA agents");

    auto rocp_hsa_agent_node_ids = std::set<uint32_t>{};
    if(rocp_agents.size() != hsa_agents.size())
    {
        for(auto hitr : hsa_agents)
        {
            auto internal_node_id = std::numeric_limits<uint32_t>::max();
            auto ret              = table->core_->hsa_agent_get_info_fn(
                hitr,
                static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_DRIVER_NODE_ID),
                &internal_node_id);

            ROCP_ERROR_IF(ret != HSA_STATUS_SUCCESS)
                << "hsa_agent_get_info(hsa_agent_t=" << hitr.handle
                << ", HSA_AMD_AGENT_INFO_DRIVER_NODE_ID, ...) returned " << ret
                << " :: " << get_hsa_status_string(ret);

            if(ret == HSA_STATUS_SUCCESS)
            {
                {
                    auto ret_emplace = rocp_hsa_agent_node_ids.emplace(internal_node_id).second;
                    ROCP_WARNING_IF(!ret_emplace)
                        << "duplicate internal node id " << internal_node_id;
                }

                for(const auto* ritr : rocp_agents)
                {
                    // TODO(aelwazir): To be changed back to use node id once ROCR fixes
                    // the hsa_agents to use the real node id
                    if(ritr->logical_node_id == static_cast<int64_t>(internal_node_id))
                    {
                        rocp_hsa_agent_node_ids.erase(internal_node_id);
                        break;
                    }
                }
            }
        }
    }

    ROCP_FATAL_IF(!rocp_hsa_agent_node_ids.empty())
        << "Found " << rocp_agents.size() << " rocprofiler agents and " << hsa_agents.size()
        << " HSA agents. HSA agents contained " << rocp_hsa_agent_node_ids.size()
        << " internal node ids not found by rocprofiler: "
        << fmt::format(
               "{}",
               fmt::join(rocp_hsa_agent_node_ids.begin(), rocp_hsa_agent_node_ids.end(), ", "));

    get_agent_caches().clear();
    get_agent_mapping().clear();
    get_agent_mapping().reserve(get_agent_mapping().size() + rocp_agents.size());

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
                // TODO(aelwazir): To be changed back to use node id once ROCR fixes
                // the hsa_agents to use the real node id
                if(ritr->logical_node_id == static_cast<int64_t>(node_id))
                {
                    agent_map.emplace(ritr->logical_node_id, std::make_tuple(ritr, hitr));
                    get_agent_mapping().emplace_back(agent_pair{ritr, hitr});
                    break;
                }
            }
        }
    }

    ROCP_INFO << "# agent node maps: " << hsa_agent_node_map.size();

    ROCP_FATAL_IF(agent_map.size() != hsa_agents.size())
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

                ROCP_FATAL_IF(_from != node_id)
                    << "unexpected condition for node_id=" << node_id << ". io_link[" << i
                    << "].node_from=" << _from
                    << ". Expected this to match the node_id (node_to=" << _to << ")";

                if(agent_map.find(_to) == agent_map.end())
                {
                    ROCP_WARNING << "no agent mapping for io_link[" << i << "].node_to=" << _to
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
        for(const auto& aitr : get_agent_caches())
        {
            if(aitr == agent_v) return true;
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
                rocp_agent, hsa_agent, itr.first, _nearest_cpu, *table->amd_ext_, *table->core_);
        } catch(std::runtime_error& err)
        {
            if(rocp_agent->type == ROCPROFILER_AGENT_TYPE_GPU)
            {
                // TODO(aelwazir): To be changed back to use node id once ROCR fixes
                // the hsa_agents to use the real node id
                ROCP_ERROR << fmt::format("rocprofiler agent <-> HSA agent mapping failed: {} ({})",
                                          rocp_agent->logical_node_id,
                                          err.what());
            }
        }
    }
}

std::optional<hsa_agent_t>
get_hsa_agent(const rocprofiler_agent_t* agent)
{
    for(const auto& itr : get_agent_mapping())
    {
        if(itr.rocp_agent->id.handle == agent->id.handle) return itr.hsa_agent;
    }

    return std::nullopt;
}

std::optional<hsa_agent_t>
get_hsa_agent(rocprofiler_agent_id_t agent_id)
{
    if(const auto* _agent = get_agent(agent_id); _agent) return get_hsa_agent(_agent);
    return std::nullopt;
}

const rocprofiler_agent_t*
get_rocprofiler_agent(hsa_agent_t agent)
{
    for(const auto& itr : get_agent_mapping())
    {
        if(itr.hsa_agent.handle == agent.handle) return itr.rocp_agent;
    }

    return nullptr;
}

const hsa::AgentCache*
get_agent_cache(const rocprofiler_agent_t* agent)
{
    for(const auto& itr : get_agent_caches())
    {
        if(itr == agent) return &itr;
    }

    return nullptr;
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

std::unordered_set<std::string>&
get_agent_available_properties()
{
    static std::unordered_set<std::string> _prop;
    return _prop;
}

void
internal_refresh_topology()
{
    auto _updated_topology = enumerate_platform_agents();
    std::swap(get_agent_topology(), _updated_topology);
}
}  // namespace agent
}  // namespace rocprofiler

extern "C" {
rocprofiler_status_t
rocprofiler_query_available_agents(rocprofiler_agent_version_t             version,
                                   rocprofiler_query_available_agents_cb_t callback,
                                   size_t                                  agent_size,
                                   void*                                   user_data)
{
    // only support version 0 for now
    if(version != ROCPROFILER_AGENT_INFO_VERSION_0)
        return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;

    // this will need to be updated for new versions
    if(version == ROCPROFILER_AGENT_INFO_VERSION_0)
    {
        if(agent_size > sizeof(rocprofiler_agent_v0_t))
        {
            ROCP_ERROR << "size of rocprofiler agent struct used by caller is ABI-incompatible "
                          "with rocprofiler_agent_v0_t in rocprofiler";
            return ROCPROFILER_STATUS_ERROR_INCOMPATIBLE_ABI;
        }
    }
    else
    {
        ROCP_FATAL << "rocprofiler-sdk does not support given agent info version";
    }

    auto&& pointers   = rocprofiler::agent::get_agents();
    auto   v_pointers = std::vector<const void*>{};
    v_pointers.reserve(pointers.size());
    for(const auto& itr : pointers)
        v_pointers.emplace_back(itr);
    return callback(version, v_pointers.data(), pointers.size(), user_data);
}
}
