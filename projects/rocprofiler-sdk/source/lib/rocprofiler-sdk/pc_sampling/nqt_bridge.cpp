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

#include "lib/rocprofiler-sdk/pc_sampling/nqt_bridge.hpp"
#include "lib/rocprofiler-sdk/pc_sampling/defines.hpp"

#if ROCPROFILER_SDK_HSA_PC_SAMPLING > 0

#    include "lib/rocprofiler-sdk/pc_sampling/parser/pc_record_interface.hpp"
#    include "lib/rocprofiler-sdk/pc_sampling/parser/rocr.h"
#    include "lib/rocprofiler-sdk/pc_sampling/service.hpp"
#    include "lib/rocprofiler-sdk/pc_sampling/types.hpp"

namespace rocprofiler
{
namespace pc_sampling
{
namespace
{
PCSAgentSession*
find_agent_session(rocprofiler_agent_id_t agent_id)
{
    auto* service = get_configured_pc_sampling_service().load();
    if(!service) return nullptr;

    auto it = service->agent_sessions.find(agent_id);
    if(it == service->agent_sessions.end()) return nullptr;

    return it->second.get();
}
}  // namespace

void
nqt_register_dispatch(rocprofiler_agent_id_t             agent_id,
                      uint64_t                           doorbell_id,
                      uint64_t                           write_index,
                      uint64_t                           queue_size,
                      rocprofiler_dispatch_id_t          dispatch_id,
                      rocprofiler_async_correlation_id_t correlation_id)
{
    auto* session = find_agent_session(agent_id);
    if(!session || !session->parser) return;

    dispatch_pkt_id_t pkt{};
    pkt.type           = AMD_DISPATCH_PKT_ID;
    pkt.device         = device_handle{static_cast<uint32_t>(session->agent->id.handle)};
    pkt.doorbell_id    = static_cast<uint32_t>(doorbell_id);
    pkt.queue_size     = queue_size;
    pkt.write_index    = write_index;
    pkt.read_index     = 0;
    pkt.correlation_id = correlation_id;
    pkt.dispatch_id    = dispatch_id;

    session->parser->newDispatch(pkt);
}

void
nqt_complete_dispatch(rocprofiler_agent_id_t agent_id,
                      uint64_t               internal_correlation_id)
{
    auto* session = find_agent_session(agent_id);
    if(!session || !session->parser) return;

    session->parser->completeDispatch(internal_correlation_id);
}

}  // namespace pc_sampling
}  // namespace rocprofiler

#else

namespace rocprofiler
{
namespace pc_sampling
{
void
nqt_register_dispatch(rocprofiler_agent_id_t,
                      uint64_t,
                      uint64_t,
                      uint64_t,
                      rocprofiler_dispatch_id_t,
                      rocprofiler_async_correlation_id_t)
{}

void
nqt_complete_dispatch(rocprofiler_agent_id_t, uint64_t)
{}

}  // namespace pc_sampling
}  // namespace rocprofiler

#endif
