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

#include "lib/rocprofiler-sdk/range_replay/range_state.hpp"
#include "lib/common/logging.hpp"
#include "lib/common/static_object.hpp"
#include "lib/common/synchronized.hpp"
#include "lib/common/utility.hpp"
#include "lib/rocprofiler-sdk/code_object/code_object.hpp"
#include "lib/rocprofiler-sdk/code_object/hsa/code_object.hpp"
#include "lib/rocprofiler-sdk/hsa/agent_cache.hpp"
#include "lib/rocprofiler-sdk/hsa/queue.hpp"

#include <rocprofiler-sdk/agent.h>
#include <rocprofiler-sdk/fwd.h>

#include <fmt/format.h>
#include <hsa/hsa.h>

#include <atomic>
#include <cstring>
#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

namespace rocprofiler
{
namespace range_replay
{
namespace
{
// Number of ranges open across the process. The dispatch path reads this before taking any lock.
std::atomic<int64_t>&
open_range_counter()
{
    static auto _v = std::atomic<int64_t>{0};
    return _v;
}

// One entry per open range. Small (bounded by the number of threads inside a range), so a vector
// scanned under a read lock is cheaper than a map. Foreign dispatches and device-writing copies
// look up by agent; `agent_key` is 0 until the range binds to its first dispatch, and an unbound
// range cannot be interfered with because it has recorded nothing yet.
struct registry_entry_t
{
    uint64_t                            agent_key = 0;
    uint64_t                            thread_id = 0;
    std::shared_ptr<external_decline_t> external  = {};
    const range_context_t*              owner     = nullptr;
};

using registry_t = std::vector<registry_entry_t>;

common::Synchronized<registry_t>&
registry()
{
    static auto*& _v = common::static_object<common::Synchronized<registry_t>>::construct();
    return *_v;
}

thread_local std::unique_ptr<range_context_t> tl_range     = {};
thread_local bool                             tl_replaying = false;

// kernel_object -> (kernarg segment size, alignment), memoized. Resolving it means walking the
// loaded code objects' symbols, which is far too slow to repeat per recorded dispatch.
struct kernarg_layout_t
{
    uint32_t size      = 0;
    uint32_t alignment = 0;
};

using kernarg_cache_t = std::unordered_map<uint64_t, kernarg_layout_t>;

std::optional<kernarg_layout_t>
kernarg_layout(uint64_t kernel_object)
{
    static auto*& cache = common::static_object<common::Synchronized<kernarg_cache_t>>::construct();
    if(cache == nullptr) return std::nullopt;

    auto hit = cache->rlock([&](const kernarg_cache_t& map) -> std::optional<kernarg_layout_t> {
        auto itr = map.find(kernel_object);
        if(itr == map.end()) return std::nullopt;
        return itr->second;
    });
    if(hit) return hit;

    auto found = std::optional<kernarg_layout_t>{};
    code_object::iterate_loaded_code_objects([&](const code_object::hsa::code_object& co) {
        if(found) return;
        for(const auto& symbol : co.symbols)
        {
            if(!symbol) continue;
            const auto& data = symbol->rocp_data;
            if(data.kernel_object != kernel_object) continue;
            found = kernarg_layout_t{data.kernarg_segment_size, data.kernarg_segment_alignment};
            break;
        }
    });

    if(found) cache->wlock([&](kernarg_cache_t& map) { map[kernel_object] = *found; });
    return found;
}

bool
is_dispatch_packet(const hsa::rocprofiler_packet& packet)
{
    const auto type = (packet.kernel_dispatch.header >> HSA_PACKET_HEADER_TYPE) &
                      ((1U << HSA_PACKET_HEADER_WIDTH_TYPE) - 1U);

    if(type == HSA_PACKET_TYPE_KERNEL_DISPATCH) return true;
#if HSA_AMD_EXT_API_TABLE_STEP_VERSION >= 0x0D
    if(type == HSA_PACKET_TYPE_VENDOR_SPECIFIC &&
       packet.ext_kernel_dispatch.amd_format == HSA_AMD_PACKET_TYPE_EXT_KERNEL_DISPATCH)
        return true;
#endif
    return false;
}

// kernarg_address and kernel_object sit at the same offsets in both dispatch packet layouts (the
// ext packet is a superset), which the static_asserts in hsa/queue.cpp establish for
// completion_signal; these two are read through the base layout for the same reason.
void*
kernarg_address(const hsa::rocprofiler_packet& packet)
{
    return packet.kernel_dispatch.kernarg_address;
}

void
set_kernarg_address(hsa::rocprofiler_packet& packet, void* address)
{
    packet.kernel_dispatch.kernarg_address = address;
}
}  // namespace

bool
range_record_t::eligible() const
{
    return m_status == ROCPROFILER_RANGE_REPLAY_STATUS_REPLAYED;
}

void
range_record_t::decline(rocprofiler_range_replay_status_t reason)
{
    if(reason == ROCPROFILER_RANGE_REPLAY_STATUS_REPLAYED) return;
    if(!eligible()) return;  // first reason wins

    m_status = reason;
    m_dispatches.clear();
    m_dispatches.shrink_to_fit();
}

bool
range_record_t::bind(uint64_t queue_key, uint64_t agent_key)
{
    if(!eligible()) return false;

    if(m_agent_key == 0 && m_queue_key == 0)
    {
        m_agent_key = agent_key;
        m_queue_key = queue_key;
        return true;
    }

    if(m_agent_key != agent_key)
    {
        decline(ROCPROFILER_RANGE_REPLAY_STATUS_MULTI_AGENT);
        return false;
    }

    if(m_queue_key != queue_key)
    {
        decline(ROCPROFILER_RANGE_REPLAY_STATUS_MULTI_QUEUE);
        return false;
    }

    return true;
}

bool
range_record_t::add_dispatch(recorded_dispatch_t&& dispatch)
{
    if(!eligible()) return false;

    if(m_dispatches.size() >= kMaxRecordedDispatches)
    {
        decline(ROCPROFILER_RANGE_REPLAY_STATUS_PROGRAM_TOO_LARGE);
        return false;
    }

    m_dispatches.emplace_back(std::move(dispatch));
    ++m_observed;
    return true;
}

void
publish_external_decline(external_decline_t& channel, rocprofiler_range_replay_status_t reason)
{
    auto expected = uint32_t{ROCPROFILER_RANGE_REPLAY_STATUS_REPLAYED};
    channel.reason.compare_exchange_strong(
        expected, static_cast<uint32_t>(reason), std::memory_order_relaxed);
}

void
fold_external_decline(range_context_t& ctx)
{
    if(!ctx.external) return;
    const auto reason = ctx.external->reason.load(std::memory_order_relaxed);
    if(reason != ROCPROFILER_RANGE_REPLAY_STATUS_REPLAYED)
        ctx.record.decline(static_cast<rocprofiler_range_replay_status_t>(reason));
}

bool
any_range_open()
{
    return open_range_counter().load(std::memory_order_relaxed) > 0;
}

bool
this_thread_replaying()
{
    return tl_replaying;
}

void
set_this_thread_replaying(bool replaying)
{
    tl_replaying = replaying;
}

range_context_t*
current_range()
{
    return tl_range.get();
}

bool
open_range(uint64_t range_id)
{
    if(tl_range) return false;

    auto ctx      = std::make_unique<range_context_t>();
    ctx->record   = range_record_t{range_id};
    ctx->external = std::make_shared<external_decline_t>();

    registry().wlock([&](registry_t& entries) {
        entries.emplace_back(registry_entry_t{0, common::get_tid(), ctx->external, ctx.get()});
    });

    tl_range = std::move(ctx);
    open_range_counter().fetch_add(1, std::memory_order_relaxed);
    return true;
}

bool
take_range(range_context_t& out)
{
    if(!tl_range) return false;

    // Unregister before folding: once the entry is gone no other thread can publish, so the fold
    // below sees the final value.
    const auto* self = tl_range.get();
    registry().wlock([&](registry_t& entries) {
        for(auto itr = entries.begin(); itr != entries.end(); ++itr)
        {
            if(itr->owner == self)
            {
                entries.erase(itr);
                break;
            }
        }
    });
    open_range_counter().fetch_sub(1, std::memory_order_relaxed);

    fold_external_decline(*tl_range);
    out = std::move(*tl_range);
    tl_range.reset();
    return true;
}

void
note_submission(const hsa::Queue&              queue,
                const hsa::rocprofiler_packet* packets,
                size_t                         packet_count,
                bool                           graph_launch_active)
{
    auto* ctx = current_range();
    if(ctx == nullptr) return;

    fold_external_decline(*ctx);
    if(!ctx->record.eligible()) return;

    if(graph_launch_active)
    {
        ctx->record.decline(ROCPROFILER_RANGE_REPLAY_STATUS_GRAPH_LAUNCH);
        return;
    }

    const auto* rocp_agent = queue.get_agent().get_rocp_agent();
    if(rocp_agent == nullptr) return;

    const auto agent_key = rocp_agent->id.handle;
    if(!ctx->record.bind(queue.get_id().handle, agent_key)) return;

    if(ctx->queue == nullptr)
    {
        ctx->queue     = &queue;
        ctx->hsa_agent = queue.get_agent().get_hsa_agent();
        ctx->agent_id  = rocp_agent->id;

        // Publish the agent so a foreign dispatch or a device-writing copy on it can find us.
        const auto* self = ctx;
        registry().wlock([&](registry_t& entries) {
            for(auto& entry : entries)
                if(entry.owner == self) entry.agent_key = agent_key;
        });
    }

    for(size_t i = 0; i < packet_count; ++i)
    {
        // Barrier and marker packets are dropped: replayed passes are serialized dispatch by
        // dispatch, so nothing they could have been waiting for is still outstanding.
        if(!is_dispatch_packet(packets[i])) continue;

        auto dispatch          = recorded_dispatch_t{};
        dispatch.packet        = packets[i];
        dispatch.kernel_object = dispatch.packet.kernel_dispatch.kernel_object;
        dispatch.kernel_id     = code_object::get_kernel_id(dispatch.kernel_object);

        const auto layout = kernarg_layout(dispatch.kernel_object);
        if(!layout)
        {
            ctx->record.decline(ROCPROFILER_RANGE_REPLAY_STATUS_UNKNOWN_KERNARG_SIZE);
            return;
        }

        // The application's completion signal must never be re-fired: it has already been consumed
        // (and may have been reused or destroyed) by the time the range closes. Replayed passes get
        // their own signals from the interceptor.
        dispatch.packet.kernel_dispatch.completion_signal = hsa_signal_t{.handle = 0};

        if(layout->size > 0)
        {
            const auto* args = static_cast<const uint8_t*>(kernarg_address(dispatch.packet));
            if(args == nullptr)
            {
                ctx->record.decline(ROCPROFILER_RANGE_REPLAY_STATUS_UNKNOWN_KERNARG_SIZE);
                return;
            }
            // Kernarg memory is host-accessible fine-grained memory, so the bytes can be copied
            // directly. This must happen now: HIP recycles a kernarg block as soon as its kernel
            // completes, so a later launch in the same range can overwrite these arguments.
            dispatch.kernarg.assign(args, args + layout->size);
        }
        set_kernarg_address(dispatch.packet, nullptr);

        if(!ctx->record.add_dispatch(std::move(dispatch))) return;
    }
}

void
note_foreign_dispatch(uint64_t agent_key)
{
    if(!any_range_open()) return;

    const auto tid = common::get_tid();
    registry().rlock([&](const registry_t& entries) {
        for(const auto& entry : entries)
        {
            if(entry.agent_key != agent_key || entry.thread_id == tid) continue;
            if(entry.external)
                publish_external_decline(*entry.external,
                                         ROCPROFILER_RANGE_REPLAY_STATUS_CONCURRENT_DISPATCH);
        }
    });
}

void
note_device_write(uint64_t agent_key)
{
    if(!any_range_open()) return;

    registry().rlock([&](const registry_t& entries) {
        for(const auto& entry : entries)
        {
            if(entry.agent_key != agent_key) continue;
            if(entry.external)
                publish_external_decline(*entry.external,
                                         ROCPROFILER_RANGE_REPLAY_STATUS_MEMORY_COPY_IN_RANGE);
        }
    });
}
}  // namespace range_replay
}  // namespace rocprofiler
