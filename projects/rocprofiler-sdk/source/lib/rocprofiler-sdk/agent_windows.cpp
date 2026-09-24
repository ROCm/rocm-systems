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

// Windows-only companion to agent.cpp.  Provides:
//   1. enumerate_platform_agents_impl() — selects the Windows platform enumerator,
//      split into its own TU to avoid the MSVC C1001 ICE that fires when the heavy
//      fmt/abseil template chains in agent.cpp share the same translation unit with
//      std::integer_sequence<enum> specializations or deep static_object<T> chains.
//   2. get_agent_topology() / get_agent_caches() — static_object<T>::construct calls
//      that also trigger the ICE in agent.cpp's context; safe here because this TU
//      does not pull in aqlprofile or libdrm headers.
//   3. Lightweight stubs for HSA-agent-mapping functions that are no-ops on Windows
//      (HSA agent cache population is driven by construct_agent_cache which is a
//      no-op on Windows until full HSA integration is complete).

#include "lib/common/environment.hpp"
#include "lib/common/logging.hpp"
#include "lib/common/static_object.hpp"
#include "lib/rocprofiler-sdk/agent.hpp"
#include "lib/rocprofiler-sdk/hsa/agent_cache.hpp"
#include "lib/rocprofiler-sdk/platform/agent.hpp"
#include "lib/rocprofiler-sdk/platform/windows/agent.hpp"

#include <rocprofiler-sdk/agent.h>
#include <rocprofiler-sdk/cxx/details/tokenize.hpp>

#include <fmt/format.h>
#include <hsa/hsa.h>
#include <hsa/hsa_api_trace.h>

#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

namespace rocprofiler
{
using unique_agent_t = platform::unique_agent_t;

// ---------------------------------------------------------------------------
// 1. Platform enumerator selection (called from agent.cpp::get_agent_topology)
// ---------------------------------------------------------------------------
std::vector<unique_agent_t>
enumerate_platform_agents_impl()
{
    const auto forced = common::get_env("ROCPROFILER_FORCE_PLATFORM", std::string{});

    if(!forced.empty())
    {
        if(forced == "windows")
        {
            ROCP_INFO << "agent topology: forced windows via ROCPROFILER_FORCE_PLATFORM";
            return platform::windows::enumerate();
        }
        ROCP_WARNING << fmt::format(
            "agent topology: ROCPROFILER_FORCE_PLATFORM='{}' is not built into this binary "
            "(expected windows on Windows); falling back to autodetect",
            forced);
    }
    if(platform::windows::is_available())
    {
        ROCP_INFO << "agent topology: selected " << platform::windows::name;
        return platform::windows::enumerate();
    }
    ROCP_WARNING << "agent topology: no platform matched; falling back to "
                 << platform::windows::name << " (will return empty)";
    return platform::windows::enumerate();
}

// ---------------------------------------------------------------------------
// 2. Static storage helpers — defined here to avoid MSVC ICE in agent.cpp
// ---------------------------------------------------------------------------
namespace agent
{
namespace
{
struct agent_pair
{
    const rocprofiler_agent_t* rocp_agent = nullptr;
    hsa_agent_t                hsa_agent  = {};
};

std::vector<unique_agent_t>&
get_agent_topology()
{
    static auto*& _v = common::static_object<std::vector<unique_agent_t>>::construct(
        ::rocprofiler::enumerate_platform_agents_impl());
    return *CHECK_NOTNULL(_v);
}

std::vector<hsa::AgentCache>&
get_agent_caches()
{
    static auto*& _v = common::static_object<std::vector<hsa::AgentCache>>::construct();
    return *CHECK_NOTNULL(_v);
}

std::vector<agent_pair>&
get_agent_mapping()
{
    static auto*& _v = common::static_object<std::vector<agent_pair>>::construct();
    return *CHECK_NOTNULL(_v);
}
}  // namespace

// ---------------------------------------------------------------------------
// 3. Public agent API — Windows stubs
// ---------------------------------------------------------------------------

std::vector<const rocprofiler_agent_t*>
get_agents()
{
    auto& agents   = get_agent_topology();
    auto  pointers = std::vector<const rocprofiler_agent_t*>{};
    pointers.reserve(agents.size());
    for(auto& agent : agents)
        pointers.emplace_back(&agent->public_info);
    return pointers;
}

const rocprofiler_agent_t*
get_agent(rocprofiler_agent_id_t id)
{
    for(const auto* itr : get_agents())
    {
        if(itr && itr->id.handle == id.handle) return itr;
    }
    return nullptr;
}

const platform::agent_info*
get_agent_info(rocprofiler_agent_id_t id)
{
    for(const auto& itr : get_agent_topology())
    {
        if(itr && itr->public_info.id.handle == id.handle) return itr.get();
    }
    return nullptr;
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

// construct_agent_cache is a no-op on Windows: HSA agent cache population
// requires HSA runtime infrastructure not yet available on Windows.
void
construct_agent_cache(::HsaApiTable* table)
{
    (void) table;
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
    // no-op on Windows — used only by Linux platform tests
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
    if(version != ROCPROFILER_AGENT_INFO_VERSION_0)
        return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;

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
