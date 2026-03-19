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

#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include "lib/rocprofiler-sdk/hsa/queue.hpp"

#include <atomic>
#include <mutex>
#include <vector>

namespace rocprofiler
{
namespace hsa
{

struct NativeQueueEntry
{
    hsa_queue_t*           queue       = nullptr;
    hsa_agent_t            agent       = {};
    std::atomic<uint64_t>  last_index  = {0};
    uint64_t               doorbell_id = 0;
    std::unique_ptr<Queue> rocp_queue;
};

struct PendingDispatch
{
    Queue*                                              queue;
    hsa_signal_t                                        signal;
    hsa_agent_t                                         hsa_agent;
    rocprofiler_kernel_id_t                             kernel_id;
    rocprofiler_dispatch_id_t                           dispatch_id;
    rocprofiler_thread_id_t                             tid;
    rocprofiler_timestamp_t                             enqueue_ts;
    rocprofiler_callback_tracing_kernel_dispatch_data_t callback_record;
    tracing::tracing_data                               tracing_data;
    context::correlation_id*                            corr_id = nullptr;
};

/// Tracks non-intercepted HSA queues for late-attach profiling.
///
/// Discovers pre-existing queues via hsa_amd_queue_iterate, enables
/// the MEC profiling bit so firmware writes timestamps into completion
/// signals.  With a HIP change that forces signal allocation on
/// profiling-enabled queues, on_hip_api_exit() detects new dispatches
/// and reads timestamps directly from the amd_signal_t struct.
class NativeQueueTracker
{
public:
    NativeQueueTracker() = default;
    ~NativeQueueTracker();

    void discover_queues(const CoreApiTable& core_api,
                         const AmdExtTable&  ext_api);

    /// Called from the HIP API ENTER callback to snapshot write indices
    /// so that EXIT can scan only the freshly-written packets.
    void on_hip_api_enter();

    /// Called from the HIP API EXIT callback.  Drains completed dispatches
    /// then scans for new kernel dispatch packets on native queues.
    void on_hip_api_exit();

    /// Force-drain all pending dispatches with a short timeout.
    /// Called during finalization before buffers are flushed.
    void flush(uint64_t timeout_ms = 200);

    bool   is_initialized() const { return _initialized.load(std::memory_order_relaxed); }
    size_t queue_count() const { return _entries.size(); }

private:
    void try_rediscover();
    void drain_completed();
    void discard_stale(uint64_t max_age_ns);

    static hsa_status_t queue_iterate_cb(hsa_queue_t* queue,
                                         hsa_agent_t  agent,
                                         void*        data);

    std::mutex                                     _mutex;
    std::vector<std::unique_ptr<NativeQueueEntry>> _entries;
    std::vector<std::unique_ptr<PendingDispatch>>  _pending;
    CoreApiTable                                   _core_api      = {};
    AmdExtTable                                    _ext_api       = {};
    std::atomic<bool>                              _initialized   = {false};
    std::atomic<bool>                              _rediscovered  = {false};
};

NativeQueueTracker*
get_native_queue_tracker();

}  // namespace hsa
}  // namespace rocprofiler
