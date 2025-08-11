// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#pragma once

#include <rocprofiler-sdk/agent.h>

#include "lib/rocprofiler-sdk/aql/aql_profile_v2.h"
#include "lib/rocprofiler-sdk/hsa/agent_cache.hpp"

#include <hsa/hsa_api_trace.h>

#include <optional>
#include <unordered_set>
#include <vector>

namespace rocprofiler
{
namespace agent
{
std::vector<const rocprofiler_agent_t*>
get_agents();

const rocprofiler_agent_t*
get_agent(rocprofiler_agent_id_t id);

void
construct_agent_cache(::HsaApiTable* table);

std::optional<hsa_agent_t>
get_hsa_agent(const rocprofiler_agent_t* agent);

std::optional<hsa_agent_t>
get_hsa_agent(rocprofiler_agent_id_t agent_id);

const rocprofiler_agent_t*
get_rocprofiler_agent(hsa_agent_t agent);

const hsa::AgentCache*
get_agent_cache(const rocprofiler_agent_t* agent);

std::optional<hsa::AgentCache>
get_agent_cache(hsa_agent_t agent);

/**
 * @brief A set containing all properties that may have been
 *        set during decoding of the properties file.
 *
 * @return std::unordered_set<std::string> of all property names
 */
std::unordered_set<std::string>&
get_agent_available_properties();

const aqlprofile_agent_handle_t*
get_aql_agent(rocprofiler_agent_id_t id);

void
construct_agent_cache(::HsaApiTable* table);

void
internal_refresh_topology();  // only for internal testing
}  // namespace agent
}  // namespace rocprofiler
