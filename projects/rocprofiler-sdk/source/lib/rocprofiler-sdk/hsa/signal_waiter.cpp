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

#include "lib/rocprofiler-sdk/hsa/signal_waiter.hpp"

#include "lib/common/logging.hpp"
#include "lib/rocprofiler-sdk/context/context.hpp"
#include "lib/rocprofiler-sdk/details/kfd_ioctl.h"
#include "lib/rocprofiler-sdk/hsa/hsa.hpp"
#include "lib/rocprofiler-sdk/hsa/queue.hpp"
#include "lib/rocprofiler-sdk/hsa/queue_controller.hpp"
#include "lib/rocprofiler-sdk/internal_threading.hpp"
#include "lib/rocprofiler-sdk/kernel_dispatch/tracing.hpp"
#include "lib/rocprofiler-sdk/registration.hpp"

#include <hsa/amd_hsa_signal.h>
#include <hsa/hsa.h>

#include <fmt/format.h>

#include <fcntl.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstring>
#include <thread>
#include <vector>

namespace rocprofiler
{
namespace hsa
{
namespace
{
amd_signal_t*
get_amd_signal(hsa_signal_t signal)
{
    // hsa_signal_t::handle points to amd_signal_t::value (offset 8 from base).
    // Subtract the offset to obtain the amd_signal_t pointer.
    return reinterpret_cast<amd_signal_t*>(reinterpret_cast<char*>(signal.handle) -
                                           offsetof(amd_signal_t, value));
}
}  // namespace

SignalWaiter::SignalWaiter()
{
    _kfd_fd = ::open("/dev/kfd", O_RDWR | O_CLOEXEC);
    ROCP_FATAL_IF(_kfd_fd < 0) << fmt::format("SignalWaiter: failed to open /dev/kfd: {}",
                                              std::strerror(errno));

    internal_threading::notify_pre_internal_thread_create(ROCPROFILER_LIBRARY);
    _thread = std::thread{&SignalWaiter::run, this};
    internal_threading::notify_post_internal_thread_create(ROCPROFILER_LIBRARY);
}

SignalWaiter::~SignalWaiter() { stop(); }

void
SignalWaiter::stop()
{
    bool expected = false;
    if(!_stopped.compare_exchange_strong(expected, true)) return;

    if(_thread.joinable()) _thread.join();

    if(_kfd_fd >= 0)
    {
        ::close(_kfd_fd);
        _kfd_fd = -1;
    }
}

void
SignalWaiter::enqueue(std::shared_ptr<queue_info_session_t> session)
{
    std::lock_guard<std::mutex> lock(_mutex);
    _pending.emplace_back(std::move(session));
}

void
SignalWaiter::run()
{
    // Active sessions being monitored for completion.
    std::vector<std::shared_ptr<queue_info_session_t>> active;

    // Scratch buffer for the ioctl call, reused across iterations.
    std::vector<kfd_event_data> event_data_buf;

    while(!_stopped.load(std::memory_order_relaxed))
    {
        // 1. Drain pending queue into the active set.
        {
            std::lock_guard<std::mutex> lock(_mutex);
            if(!_pending.empty())
            {
                active.insert(active.end(),
                              std::make_move_iterator(_pending.begin()),
                              std::make_move_iterator(_pending.end()));
                _pending.clear();
            }
        }

        // 2. If nothing to wait on, sleep briefly and retry.
        if(active.empty())
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        // 3-4. Collect event_ids from every packet in every active session.
        event_data_buf.clear();

        for(size_t si = 0; si < active.size(); ++si)
        {
            auto& session = *active[si];
            for(size_t pi = 0; pi < session.packet_data.size(); ++pi)
            {
                auto& packet = session.packet_data[pi];
                if(packet.completion_signal.handle == 0) continue;

                auto*    amd_sig = get_amd_signal(packet.completion_signal);
                uint32_t eid     = amd_sig->event_id;
                if(eid == 0) continue;

                kfd_event_data ev     = {};
                ev.event_id           = eid;
                ev.kfd_event_data_ext = 0;
                event_data_buf.emplace_back(ev);
            }
        }

        // If no valid event_ids were found (shouldn't normally happen), sleep and retry.
        if(event_data_buf.empty())
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        // 5. Wait for any signal to complete via KFD ioctl.
        kfd_ioctl_wait_events_args args = {};
        args.events_ptr                 = reinterpret_cast<uint64_t>(event_data_buf.data());
        args.num_events                 = static_cast<uint32_t>(event_data_buf.size());
        args.wait_for_all               = 0;     // wait for ANY
        args.timeout                    = 1000;  // 1 second timeout

        int ret;
        do
        {
            ret = ::ioctl(_kfd_fd, AMDKFD_IOC_WAIT_EVENTS, &args);
        } while(ret == -1 && (errno == EINTR || errno == EAGAIN));

        // On timeout or error, loop back to check _stopped and try again.
        // Timeout is not an error — just means no signals completed within the window.

        // 6-8. Check each signal value and process completions.
        // Track which sessions have at least one completed packet.
        // We use a bitmask per session to know when ALL packets are done.
        std::vector<bool> session_fully_complete(active.size(), true);

        // First pass: mark each session as fully complete, then check packets.
        for(size_t si = 0; si < active.size(); ++si)
        {
            auto& session  = *active[si];
            bool  all_done = true;

            for(size_t pi = 0; pi < session.packet_data.size(); ++pi)
            {
                auto& packet = session.packet_data[pi];
                if(packet.completion_signal.handle == 0) continue;

                auto* amd_sig = get_amd_signal(packet.completion_signal);
                auto  value   = __atomic_load_n(&amd_sig->value, __ATOMIC_ACQUIRE);

                if(value > 0)
                {
                    // Not yet completed.
                    all_done = false;
                    continue;
                }

                // Signal completed (value <= 0). Process this packet.

                if(registration::get_fini_status() == 0)
                {
                    auto dispatch_time = kernel_dispatch::get_dispatch_time(session, packet);
                    kernel_dispatch::dispatch_complete(session, packet, dispatch_time);

                    auto session_ptr = active[si];
                    session.queue.signal_callback([&](const auto& map) {
                        for(const auto& [client_id, cb_data] : map)
                        {
                            cb_data.signal_completion(session.queue,
                                                      packet.kernel_packet,
                                                      session_ptr,
                                                      packet,
                                                      packet.instrumentation_packets,
                                                      dispatch_time);
                        }
                    });

                    if(packet.is_serialized)
                    {
                        CHECK_NOTNULL(hsa::get_queue_controller())
                            ->serializer(&session.queue)
                            .wlock([&](auto& serializer) {
                                serializer.kernel_completion_signal(session.queue);
                            });
                    }
                }

                // Always clean up signal lifetime and correlation IDs,
                // even during finalization.
                auto* ext_table = hsa::get_amd_ext_table();
                if(ext_table && ext_table->hsa_amd_signal_waiting_dec_fn)
                    ext_table->hsa_amd_signal_waiting_dec_fn(packet.completion_signal);

                auto* _corr_id = session.correlation_id;
                if(_corr_id)
                {
                    ROCP_FATAL_IF(_corr_id->get_ref_count() == 0)
                        << "reference counter for correlation id " << _corr_id->internal
                        << " from thread " << _corr_id->thread_idx << " has no reference count";
                    _corr_id->sub_kern_count();
                    _corr_id->sub_ref_count();
                }

                packet.completion_signal.handle = 0;
            }

            session_fully_complete[si] = all_done;
        }

        // 8-9. Remove fully-completed sessions and call async_complete for each.
        for(size_t si = active.size(); si-- > 0;)
        {
            if(session_fully_complete[si])
            {
                active[si]->queue.async_complete();
                active.erase(active.begin() + static_cast<ptrdiff_t>(si));
            }
        }
    }

    // Shutdown: clean up any remaining active sessions.
    for(auto& session : active)
    {
        if(!session) continue;

        for(auto& packet : session->packet_data)
        {
            if(packet.completion_signal.handle == 0) continue;

            auto* ext_table = hsa::get_amd_ext_table();
            if(ext_table && ext_table->hsa_amd_signal_waiting_dec_fn)
                ext_table->hsa_amd_signal_waiting_dec_fn(packet.completion_signal);

            auto* _corr_id = session->correlation_id;
            if(_corr_id)
            {
                _corr_id->sub_kern_count();
                _corr_id->sub_ref_count();
            }
        }

        session->queue.async_complete();
    }
    active.clear();
}

}  // namespace hsa
}  // namespace rocprofiler
