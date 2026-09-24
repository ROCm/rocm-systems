// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

// Windows stand-ins for the AQL sources excluded from this build. Every entry point
// here ultimately calls into aqlprofile to translate counter metrics into AQL packets;
// aqlprofile is not built on Windows, so a packet can never be constructed and
// can_collect() reports the counters as uncollectable.

#include "lib/rocprofiler-sdk/aql/helpers.hpp"
#include "lib/rocprofiler-sdk/aql/packet_construct.hpp"

#include <rocprofiler-sdk/fwd.h>

#include <cstdint>
#include <map>
#include <memory>
#include <vector>

namespace rocprofiler
{
namespace aql
{
// stands in for packet_construct.cpp
CounterPacketConstruct::CounterPacketConstruct(rocprofiler_agent_id_t               agent,
                                               const std::vector<counters::Metric>& metrics)
: _agent{agent}
{
    _metrics.reserve(metrics.size());
    for(const auto& itr : metrics)
        _metrics.emplace_back(AQLProfileMetric{itr, {}, {}});
}

std::unique_ptr<hsa::CounterAQLPacket>
CounterPacketConstruct::construct_packet(const CoreApiTable&, const AmdExtTable&)
{
    return nullptr;
}

const std::vector<aqlprofile_pmc_event_t>&
CounterPacketConstruct::get_counter_events(const counters::Metric&) const
{
    return _events;
}

rocprofiler_status_t
CounterPacketConstruct::can_collect()
{
    return ROCPROFILER_STATUS_ERROR_NOT_IMPLEMENTED;
}

// stands in for helpers.cpp
rocprofiler_status_t
get_dim_info(rocprofiler_agent_id_t, aqlprofile_pmc_event_t, uint32_t, std::map<int, uint64_t>&)
{
    return ROCPROFILER_STATUS_ERROR_NOT_IMPLEMENTED;
}
}  // namespace aql
}  // namespace rocprofiler
