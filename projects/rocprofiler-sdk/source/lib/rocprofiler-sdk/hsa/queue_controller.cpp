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

#include "lib/rocprofiler-sdk/hsa/queue_controller.hpp"
#include "lib/common/static_object.hpp"
#include "lib/rocprofiler-sdk/agent.hpp"
#include "lib/rocprofiler-sdk/context/context.hpp"
#include "lib/rocprofiler-sdk/hsa/agent_cache.hpp"
#include "lib/rocprofiler-sdk/hsa/queue.hpp"
#include "lib/rocprofiler-sdk/hsa/queue_interposition.hpp"
#include "lib/rocprofiler-sdk/kfd/kfd_correlation.hpp"
#include "lib/rocprofiler-sdk/kfd/kfd_profiler.hpp"
#include "lib/rocprofiler-sdk/kfd/signal_less.hpp"
#include "lib/rocprofiler-sdk/kfd/signal_less_gate.hpp"

#include <hsa/amd_hsa_queue.h>
#include <hsa/amd_hsa_signal.h>
#include <hsa/hsa_ext_amd.h>

#include <rocprofiler-sdk/fwd.h>
#include <unistd.h>
#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>

namespace rocprofiler
{
namespace hsa
{
namespace
{
// Read AGENT's own GPU-clock counter -- the same tick domain as fw_record::ts for
// records emitted by AGENT. Called exactly twice per queue lifetime, never
// per dispatch. Returns 0 on any failure or on an untrustworthy sentinel; the
// caller then fails closed (poison / signal path). `core` is the saved core table
// the controller already holds, so this needs no free-function table accessor.
uint64_t
gpu_tick_now(const CoreApiTable& core, hsa_agent_t agent)
{
// HSA_AMD_AGENT_INFO_CLOCK_COUNTERS / hsa_amd_clock_counters_t were added at HSA
// AMD interface 1.11 (ROCm 7.0). Older installed headers (e.g. the 6.2/6.3/6.4
// release-compatibility builds) don't declare them at all, so this whole path is
// compiled out there; the caller's existing clock-failure handling (poison the
// slot / take the signal path) is exactly the right degraded behavior when the
// query itself can't be made.
#if defined(HSA_AMD_INTERFACE_VERSION_MAJOR) &&                                                    \
    (HSA_AMD_INTERFACE_VERSION_MAJOR > 1 ||                                                        \
     (HSA_AMD_INTERFACE_VERSION_MAJOR == 1 && HSA_AMD_INTERFACE_VERSION_MINOR >= 11))
    hsa_amd_clock_counters_t c{};
    if(core.hsa_agent_get_info_fn == nullptr) return 0;
    if(core.hsa_agent_get_info_fn(agent,
                                  static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_CLOCK_COUNTERS),
                                  &c) != HSA_STATUS_SUCCESS)
        return 0;
    // Reject 0 and the kWindowOpen sentinel so no sentinel is ever stored as a
    // real t_open/t_close and boundary reasoning never sees a wrapped tick.
    if(c.gpu_clock_counter == 0 || c.gpu_clock_counter == kfd::kWindowOpen) return 0;
    return c.gpu_clock_counter;
#else
    (void) core;
    (void) agent;
    return 0;
#endif
}

// HSA Intercept Functions (create_queue/destroy_queue)
hsa_status_t
create_queue(hsa_agent_t        agent,
             uint32_t           size,
             hsa_queue_type32_t type,
             void (*callback)(hsa_status_t status, hsa_queue_t* source, void* data),
             void*         data,
             uint32_t      private_segment_size,
             uint32_t      group_segment_size,
             hsa_queue_t** queue)
{
    auto* controller = CHECK_NOTNULL(get_queue_controller());
    for(const auto& [_, agent_info] : controller->get_supported_agents())
    {
        if(agent_info.get_hsa_agent().handle == agent.handle)
        {
            std::unique_ptr<Queue> new_queue;
            if(queue_interposition::supports_queue_interposition())
            {
                ROCP_INFO << "[queue-intercept] creating queue via INLINE path for agent "
                          << agent.handle;
                auto status = controller->get_core_table().hsa_queue_create_fn(agent,
                                                                               size,
                                                                               type,
                                                                               callback,
                                                                               data,
                                                                               private_segment_size,
                                                                               group_segment_size,
                                                                               queue);
                if(status != HSA_STATUS_SUCCESS) return status;

                new_queue = std::make_unique<Queue>(agent_info,
                                                    controller->get_core_table(),
                                                    controller->get_ext_table(),
                                                    *queue,
                                                    [](write_interceptor_t, void*) {});
            }
            else
            {
                ROCP_INFO << "[queue-intercept] creating queue via LEGACY path for agent "
                          << agent.handle;
                new_queue = std::make_unique<Queue>(agent_info,
                                                    size,
                                                    type,
                                                    callback,
                                                    data,
                                                    private_segment_size,
                                                    group_segment_size,
                                                    controller->get_core_table(),
                                                    controller->get_ext_table(),
                                                    queue);
            }

            controller->serializer(new_queue.get()).wlock([&](auto& serializer) {
                serializer.add_queue(queue, *new_queue);
            });
            controller->add_queue(*queue, std::move(new_queue));
            ROCP_INFO << "created queue for HSA agent handle " << agent.handle;
            return HSA_STATUS_SUCCESS;
        }
    }
    ROCP_FATAL << "Could not find agent - " << agent.handle;
    return HSA_STATUS_ERROR_FATAL;
}

hsa_status_t
destroy_queue(hsa_queue_t* hsa_queue)
{
    if(get_queue_controller()) get_queue_controller()->destroy_queue(hsa_queue);
    return HSA_STATUS_SUCCESS;
}

#if defined(HSA_AMD_EXT_API_TABLE_STEP_VERSION) && HSA_AMD_EXT_API_TABLE_STEP_VERSION >= 0x10
hsa_status_t
create_amd_queue(hsa_agent_t agent, hsa_amd_queue_create_desc_t* descs, uint32_t num_descs)
{
    auto* controller = CHECK_NOTNULL(get_queue_controller());
    auto  status     = controller->get_ext_table().hsa_amd_queue_create_fn(agent, descs, num_descs);

    // hsa_amd_queue_create permits partial batch success: it may return an error for the first
    // failing descriptor while leaving earlier descs[i].queue entries valid.  Process every
    // non-null queue so those successful queues are still registered with rocprofiler-sdk (Queue,
    // serializer entry, QueueState).  The original status is returned afterward.
    const bool inline_intercept = queue_interposition::supports_queue_interposition();

    for(uint32_t desc_idx = 0; desc_idx < num_descs; ++desc_idx)
    {
        hsa_queue_t* queue = descs[desc_idx].queue;
        if(!queue) continue;

        for(const auto& [_, agent_info] : controller->get_supported_agents())
        {
            if(agent_info.get_hsa_agent().handle != agent.handle) continue;

            std::unique_ptr<Queue> new_queue;
            if(inline_intercept)
            {
                // Inline write-index wrappers intercept this plain queue directly (via the
                // per-queue state registered in add_queue), so no ROCr-level packet
                // interceptor needs to be attached here.
                ROCP_INFO << "[queue-intercept] registering hsa_amd_queue_create queue via "
                             "INLINE path for agent "
                          << agent.handle;
                new_queue = std::make_unique<Queue>(agent_info,
                                                    controller->get_core_table(),
                                                    controller->get_ext_table(),
                                                    queue,
                                                    [](write_interceptor_t, void*) {});
            }
            else if(descs[desc_idx].engine_type == HSA_AMD_QUEUE_ENGINE_COMPUTE)
            {
                // A queue created by hsa_amd_queue_create is a plain queue that cannot carry a
                // ROCr packet interceptor (hsa_amd_queue_intercept_register only accepts an
                // InterceptQueue). In non-inline mode we therefore discard the plain queue and
                // build a ROCr InterceptQueue that mirrors the descriptor's compute parameters,
                // then hand that back to the caller in place of the original queue.
                //
                // NOTE: the InterceptQueue is created via the classic hsa_queue_create path, so
                // the descriptor's device-memory ring-buffer flag is not honored while a
                // non-inline profiling context (e.g. --sys-trace / HIP graph tracing) is active;
                // the queue falls back to a system-memory ring.
                //
                // Known limitation: the replacement InterceptQueue does not preserve the
                // descriptor's priority or CU-mask settings. A CU-partitioned or high-priority
                // stream may behave differently while profiled. Passing these attributes through
                // (e.g. via SetPriority / SetCUMasking on the replacement queue) is deferred to a
                // follow-up change.
                ROCP_WARNING
                    << "[queue-intercept] device-memory ring-buffer requested but profiling "
                       "requires system-memory InterceptQueue; falling back to system-memory "
                       "ring for agent "
                    << agent.handle
                    << " (priority and CU-mask from the descriptor are not preserved)";

                const auto&    compute_params = descs[desc_idx].engine.compute;
                const uint32_t ring_packets   = static_cast<uint32_t>(
                    descs[desc_idx].queue_size_bytes / sizeof(hsa_kernel_dispatch_packet_t));

                // Release the plain queue that hsa_amd_queue_create just handed us; the
                // InterceptQueue built below replaces it. get_core_table() holds the real ROCr
                // functions (saved before the interception wrappers were installed).
                controller->get_core_table().hsa_queue_destroy_fn(queue);

                hsa_queue_t* intercept_queue = nullptr;
                new_queue                    = std::make_unique<Queue>(
                    agent_info,
                    ring_packets,
                    compute_params.type,
                    descs[desc_idx].callback,
                    descs[desc_idx].callback_data,
                    compute_params.private_segment_size,
                    // Descriptor v1 does not expose group_segment_size; UINT32_MAX requests the
                    // runtime default, matching the classic hsa_queue_create behavior.
                    UINT32_MAX,
                    controller->get_core_table(),
                    controller->get_ext_table(),
                    &intercept_queue);

                queue                 = intercept_queue;
                descs[desc_idx].queue = intercept_queue;
            }
            else
            {
                // Non-compute engines (e.g. SDMA) do not dispatch kernels; keep the plain queue
                // and register it without a packet interceptor.  Skip inline QueueState
                // registration because SDMA queue sizes are byte counts and use a different
                // packet format incompatible with AQL interposition.
                ROCP_INFO << "[queue-intercept] registering non-compute hsa_amd_queue_create "
                             "queue (no packet interception) for agent "
                          << agent.handle;
                new_queue = std::make_unique<Queue>(agent_info,
                                                    controller->get_core_table(),
                                                    controller->get_ext_table(),
                                                    queue,
                                                    [](write_interceptor_t, void*) {});
            }

            const bool is_compute = (descs[desc_idx].engine_type == HSA_AMD_QUEUE_ENGINE_COMPUTE);
            controller->serializer(new_queue.get()).wlock([&](auto& serializer) {
                serializer.add_queue(&queue, *new_queue);
            });
            controller->add_queue(queue, std::move(new_queue), is_compute);
            ROCP_INFO << "created queue (hsa_amd_queue_create) for HSA agent handle "
                      << agent.handle;
            break;
        }
    }
    return status;
}
#endif

constexpr rocprofiler_agent_t default_agent =
    rocprofiler_agent_t{.size                       = sizeof(rocprofiler_agent_t),
                        .id                         = rocprofiler_agent_id_t{.handle = 0},
                        .type                       = ROCPROFILER_AGENT_TYPE_NONE,
                        .cpu_cores_count            = 0,
                        .simd_count                 = 0,
                        .mem_banks_count            = 0,
                        .caches_count               = 0,
                        .io_links_count             = 0,
                        .cpu_core_id_base           = 0,
                        .simd_id_base               = 0,
                        .max_waves_per_simd         = 0,
                        .lds_size_in_kb             = 0,
                        .gds_size_in_kb             = 0,
                        .num_gws                    = 0,
                        .wave_front_size            = 0,
                        .num_xcc                    = 0,
                        .cu_count                   = 0,
                        .array_count                = 0,
                        .num_shader_banks           = 0,
                        .simd_arrays_per_engine     = 0,
                        .cu_per_simd_array          = 0,
                        .simd_per_cu                = 0,
                        .max_slots_scratch_cu       = 0,
                        .gfx_target_version         = 0,
                        .vendor_id                  = 0,
                        .device_id                  = 0,
                        .location_id                = 0,
                        .domain                     = 0,
                        .drm_render_minor           = 0,
                        .num_sdma_engines           = 0,
                        .num_sdma_xgmi_engines      = 0,
                        .num_sdma_queues_per_engine = 0,
                        .num_cp_queues              = 0,
                        .max_engine_clk_ccompute    = 0,
                        .max_engine_clk_fcompute    = 0,
                        .sdma_fw_version            = {},
                        .fw_version                 = {},
                        .capability                 = {},
                        .cu_per_engine              = 0,
                        .max_waves_per_cu           = 0,
                        .family_id                  = 0,
                        .workgroup_max_size         = 0,
                        .grid_max_size              = 0,
                        .local_mem_size             = 0,
                        .hive_id                    = 0,
                        .gpu_id                     = 0,
                        .workgroup_max_dim          = {.x = 0, .y = 0, .z = 0},
                        .grid_max_dim               = {.x = 0, .y = 0, .z = 0},
                        .mem_banks                  = nullptr,
                        .caches                     = nullptr,
                        .io_links                   = nullptr,
                        .name                       = nullptr,
                        .vendor_name                = nullptr,
                        .product_name               = nullptr,
                        .model_name                 = nullptr,
                        .node_id                    = 0,
                        .logical_node_id            = 0,
                        .logical_node_type_id       = 0,
                        .runtime_visibility         = {0, 0, 0, 0, 0},
                        .uuid = static_cast<rocprofiler_uuid_t>(agent::uuid_view_t{})};

RocAttachDispatchTable**
get_attach_table()
{
    static auto* table = common::static_object<RocAttachDispatchTable*>::construct();
    return table;
}

void
queue_controller_iterate_attach_queue(hsa_queue_t* queue, hsa_agent_t agent, void*)
{
    auto* qc                    = CHECK_NOTNULL(get_queue_controller());
    bool  registration_consumed = false;

    auto set_write_interceptor = [&queue](write_interceptor_t wi, void* data) {
        CHECK_NOTNULL(*(get_attach_table()))
            ->rocprofiler_attach_set_write_interceptor(queue, wi, data);
    };

    for(const auto& [_, agent_info] : qc->get_supported_agents())
    {
        if(agent_info.get_hsa_agent().handle == agent.handle)
        {
            auto new_queue = std::make_unique<rocprofiler::hsa::Queue>(agent_info,
                                                                       qc->get_core_table(),
                                                                       qc->get_ext_table(),
                                                                       queue,
                                                                       set_write_interceptor);

            qc->serializer(new_queue.get()).wlock([&](auto& serializer) {
                serializer.add_queue(&queue, *new_queue);
            });
            // is_compute is an ASSUMPTION here (the attach callback carries no
            // engine type); is_attach is a fact about this call site. Both
            // written out. Attach opens no window and latches the process-wide
            // disable, so its dispatches take the signal path.
            qc->add_queue(queue, std::move(new_queue), /*is_compute=*/true, /*is_attach=*/true);
            registration_consumed = true;
            ROCP_INFO << "Adding queue from queue registration for HSA agent handle "
                      << agent.handle;
            break;
        }
    }
    if(!registration_consumed)
    {
        ROCP_FATAL << "Could not find agent " << agent.handle << " for queue registration";
    }
}

void
queue_controller_attach_queue_event(hsa_queue_t*                     queue,
                                    hsa_agent_t                      agent,
                                    rocprofiler_attach_queue_phase_t phase,
                                    void* /*data*/)
{
    if(phase == ROCPROFILER_ATTACH_QUEUE_CREATED)
    {
        queue_controller_iterate_attach_queue(queue, agent, nullptr);
    }
    else if(auto* qc = get_queue_controller())
    {
        qc->destroy_queue(queue);
    }
}

void
queue_controller_load_attach_queues()
{
    auto* attach_table = CHECK_NOTNULL(*(get_attach_table()));

    attach_table->rocprofiler_attach_add_queue_cb(queue_controller_attach_queue_event, nullptr);
}

}  // namespace

void
QueueController::add_queue(hsa_queue_t*           id,
                           std::unique_ptr<Queue> queue,
                           bool                   is_compute,
                           bool                   is_attach)
{
    CHECK(queue);
    const auto agent_id = queue->get_agent().get_rocp_agent()->id;

    _callback_cache.wlock([&](auto& callbacks) {
        _queues.wlock([&](auto& map) {
            map[id] = std::move(queue);
            for(const auto& [cbid, cb_data] : callbacks)
            {
                auto& [agent, cb] = cb_data;
                if(agent.id == default_agent.id || agent.id == agent_id)
                {
                    map[id]->register_callback(cbid, cb);
                }
            }
        });
    });

    // signal-less live-queue bookkeeping and window open. Gated on
    // is_compute -- only a compute queue's doorbell can source a CP dispatch-log
    // record -- and on fork safety. Inert with the feature off. Every live
    // compute queue registers ownership (a queue that predates the session or never
    // dispatches still owns its slot); the clock is read once, before the map lock.
    if(is_compute && kfd::signal_less_feature_enabled() && !kfd::signal_less_child_stale())
    {
        if(const auto* _q = get_queue(*id);
           _q && kfd::gpu_supports_dispatch_log(
                     static_cast<uint32_t>(_q->get_agent().get_rocp_agent()->gpu_id)))
        {
            // T-CLK: gate all window/registry bookkeeping on this GPU's dispatch-log
            // capability FIRST. A compute queue on an unsupported/attached GPU is not a
            // dispatch-log participant, so it must not open windows, register ownership,
            // or trip signal_less_disable_permanently() process-wide.
            // Open the owner window on any dispatch_log_stream_format-capable compute
            // agent: the record tick and gpu_tick_now are the same free-running GPU
            // clock counter, so the capability probe already covers the clock domain.
            const auto* _rocp    = _q->get_agent().get_rocp_agent();
            const auto  _gpu     = static_cast<uint32_t>(_rocp->gpu_id);
            auto        _slot    = std::optional<uint32_t>{};
            bool        _poison  = false;
            bool        _disable = is_attach;  // an adopted queue's history is unseen
            if(auto _db = capture_doorbell_key(_q->intercept_queue()))
            {
                _slot = *_db;  // a live owner, always registered with its real slot
                if(!is_attach)
                {
                    const uint64_t _tick =
                        gpu_tick_now(get_core_table(), _q->get_agent().get_hsa_agent());
                    if(_tick == 0)
                        _poison = true;  // clock failure: slot-scoped poison
                    else
                        _poison = kfd::doorbell_map()
                                      .open_window(_gpu, _q->get_id(), *_slot, _tick)
                                      .overlapped;  // two live owners
                }
            }
            else
            {
                _disable = true;  // capture failed: an unwindowed owner exists
            }
            kfd::add_live_queue(_q->get_id().handle, _gpu, _slot);
            if(_poison && _slot) kfd::poison_slot(_gpu, *_slot);
            if(_disable) kfd::signal_less_disable_permanently();
        }
    }

    // Interposition-state creation wants the same answer as the signal-less gate:
    // only a compute queue's AQL ring can be interposed.
    if(is_compute)
    {
        queue_interposition::create_queue_state(id);
    }
}

void
QueueController::destroy_queue(hsa_queue_t* id)
{
    if(!id) return;

    const auto* queue = get_queue(*id);

    // return if queue does not exist
    if(!queue) return;

    const auto _queue_token = queue->get_id().handle;

    // close this queue's owner window. Inert with the feature off and
    // gated for fork safety. Holds at most one of {gate, DoorbellMap, hub, registry}
    // at any instant, so no lock cycle exists. Never blocks on the reader.
    if(kfd::signal_less_feature_enabled() && !kfd::signal_less_child_stale())
    {
        // F1: hold drain_mu across the whole close/drain/close-window sequence, so
        // finalization's concurrent drain either wins the mutex (and blocks us before
        // the runtime queue is freed) or observes rdid_valid==false and skips. The
        // state shared_ptr keeps QueueState (and drain_mu) alive across the erase.
        auto _state    = queue_interposition::lookup_queue_state(id, /*create_if_missing=*/false);
        auto _drain_lk = _state ? std::unique_lock<std::mutex>{_state->drain_mu}
                                : std::unique_lock<std::mutex>{};

        // Step 0: latch admission AND snapshot next_submit_pos in ONE gate_lock
        // section. The latch precedes the snapshot; both are ordered
        // against every publishing critical section by that lock, so no separate
        // fence is needed and every already-registered packet is <= P.
        const uint64_t _P =
            _state ? queue_interposition::close_admission_and_snapshot_locked(*_state) : 0;

        // Step 0b: derive (gpu, slot) BEFORE step 6 destroys the mapping; skip the
        // window work entirely when this queue never resolved a slot.
        const auto _slot = kfd::owner_registry().slot_of(_queue_token);
        const auto _gpu  = kfd::owner_registry().gpu_of(_queue_token);
        if(_slot && _gpu)
        {
            // Step 2: HW drain against the snapshot P, unconditionally -- this
            // queue's UNREGISTERED dispatches also produce firmware records that an
            // unanchored t_close could misattribute. Bounded by the
            // per-close budget; the aggregate pool is deleted.
            const uint64_t _deadline = kfd::steady_now_ns() + kfd::close_drain_budget_ns();
            const bool     _drained =
                _state ? queue_interposition::wait_queue_hw_drained_locked(*_state, _P, _deadline)
                           : true;
            // Step 3: read t_close AFTER the drain, before any lock, from this
            // queue's agent.
            const uint64_t _tick =
                gpu_tick_now(get_core_table(), queue->get_agent().get_hsa_agent());
            // Step 4: a truncated close or a clock failure leaves t_close unable to
            // bound anything, so poison the slot.
            if(!_drained || _tick == 0) kfd::poison_slot(*_gpu, *_slot);
            // Step 5: stamp t_close and the GC deadline (nullptr if no window).
            kfd::doorbell_map().close_window(
                queue->get_id(), _tick, kfd::steady_now_ns() + kfd::close_drain_budget_ns());
        }

        // Step 6: drop ownership so a surviving co-owner becomes injective again.
        kfd::remove_live_queue(_queue_token);

        // Last action under drain_mu: invalidate the interlock BEFORE releasing it,
        // so finalization can never load a real_rdid the runtime is about to free.
        if(_state)
        {
            _state->rdid_valid = false;
            _state->real_rdid  = nullptr;
        }
        // Release drain_mu here (end of the `if` scope) -- BEFORE destroy_queue_state/
        // sync/erase, i.e. before anything can free amd_queue_t.
    }

    queue_interposition::destroy_queue_state(id);
    queue->sync();
    if(queue->block_signal.handle != 0) get_core_table().hsa_signal_destroy_fn(queue->block_signal);
    _queues.wlock([&](auto& map) { map.erase(id); });
}

ClientID
QueueController::add_callback(std::optional<rocprofiler_agent_t> agent, queue_callbacks_t callbacks)
{
    static auto client_id = std::atomic<ClientID>{1};
    ClientID    return_id = -1;
    _callback_cache.wlock([&](auto& cb_cache) {
        return_id = client_id;
        if(agent)
        {
            cb_cache[client_id] = std::make_tuple(*agent, callbacks);
        }
        else
        {
            cb_cache[client_id] = std::make_tuple(default_agent, callbacks);
        }
        client_id++;

        _queues.wlock([&](auto& map) {
            for(auto& [_, queue] : map)
            {
                if(!agent || queue->get_agent().get_rocp_agent()->id.handle == agent->id.handle)
                {
                    queue->register_callback(return_id, callbacks);
                }
            }
        });
    });
    return return_id;
}

void
QueueController::remove_callback(ClientID id)
{
    _callback_cache.wlock([&](auto& cb_cache) {
        cb_cache.erase(id);
        _queues.wlock([&](auto& map) {
            for(auto& [_, queue] : map)
            {
                queue->remove_callback(id);
            }
        });
    });
}

void
QueueController::init(CoreApiTable& core_table, AmdExtTable& ext_table)
{
    _core_table = core_table;
    _ext_table  = ext_table;

    auto agents = agent::get_agents();

    // Generate supported agents
    for(const auto* itr : agents)
    {
        const auto* cached_agent = agent::get_agent_cache(itr);
        ROCP_TRACE << fmt::format(
            "RocP Agent {:x} has Cache Agent? {}", itr->id.handle, cached_agent ? "yes" : "no");
        if(cached_agent)
        {
            ROCP_TRACE << fmt::format("RocP Agent {:x} Type {}",
                                      itr->id.handle,
                                      (int) cached_agent->get_rocp_agent()->type);
        }

        if(cached_agent && cached_agent->get_rocp_agent()->type == ROCPROFILER_AGENT_TYPE_GPU)
        {
            ROCP_TRACE << fmt::format("RocP Agent {:x} is added to cache", itr->id.handle);
            get_supported_agents().emplace(cached_agent->index(), *cached_agent);
        }
    }

    if(enable_queue_intercept())
    {
        if(*(get_attach_table()))
        {
            // Attach table was previously registered, so we need to
            // - Load and instrument queues that the attach library captured
            // - NOT instrument the HSA API as the attach library has already done so
            queue_controller_load_attach_queues();
        }
        else
        {
            core_table.hsa_queue_create_fn  = hsa::create_queue;
            core_table.hsa_queue_destroy_fn = hsa::destroy_queue;
#if defined(HSA_AMD_EXT_API_TABLE_STEP_VERSION) && HSA_AMD_EXT_API_TABLE_STEP_VERSION >= 0x10
            ext_table.hsa_amd_queue_create_fn = hsa::create_amd_queue;
#endif
        }
    }
}

const Queue*
QueueController::get_queue(const hsa_queue_t& _hsa_queue) const
{
    return _queues.rlock(
        [](const queue_map_t& _data, const hsa_queue_t& _inp) -> const Queue* {
            for(const auto& itr : _data)
            {
                if(itr.first->id == _inp.id) return itr.second.get();
            }
            return nullptr;
        },
        _hsa_queue);
}

common::Synchronized<hsa::profiler_serializer>&
QueueController::serializer(const Queue* queue)
{
    CHECK(queue);
    common::Synchronized<hsa::profiler_serializer>* ret = nullptr;
    _profiler_serializer.ulock(
        [&](const auto& m) {
            if(auto ptr = m.find(queue->get_agent().get_rocp_agent()->id); ptr != m.end())
            {
                ret = ptr->second.get();
                return true;
            }
            return false;
        },
        [&](auto& m) {
            ret = m.emplace(queue->get_agent().get_rocp_agent()->id,
                            std::make_shared<common::Synchronized<hsa::profiler_serializer>>())
                      .first->second.get();
            if(_serialized_enabled.load() == true)
            {
                ret->wlock([&](auto& serializer) { serializer.enable({}); });
            }
            return true;
        });
    return *ret;
}

namespace
{
std::unordered_map<rocprofiler_agent_id_t, hsa_barrier::queue_map_ptr_t>
per_dev_map(const QueueController::queue_map_t& _queues_v)
{
    std::unordered_map<rocprofiler_agent_id_t, hsa_barrier::queue_map_ptr_t> dmap;
    for(const auto& [k, v] : _queues_v)
    {
        dmap[v->get_agent().get_rocp_agent()->id][k] = v.get();
    }
    return dmap;
}
};  // namespace

void
QueueController::disable_serialization()
{
    _queues.rlock([&](const queue_map_t& _queues_v) {
        _serialized_enabled.store(false);
        auto pd_map = per_dev_map(_queues_v);
        _profiler_serializer.wlock([&](auto& m) {
            for(auto& [k, v] : m)
            {
                if(auto it = pd_map.find(k); it != pd_map.end())
                {
                    v->wlock([&](auto& serializer) { serializer.disable(it->second); });
                }
                else
                {
                    v->wlock([&](auto& serializer) { serializer.disable({}); });
                }
            }
        });
    });
}

void
QueueController::enable_serialization()
{
    _queues.rlock([&](const queue_map_t& _queues_v) {
        _serialized_enabled.store(true);
        auto pd_map = per_dev_map(_queues_v);
        _profiler_serializer.wlock([&](auto& m) {
            for(auto& [k, v] : m)
            {
                if(auto it = pd_map.find(k); it != pd_map.end())
                {
                    v->wlock([&](auto& serializer) { serializer.enable(it->second); });
                }
                else
                {
                    v->wlock([&](auto& serializer) { serializer.enable({}); });
                }
            }
        });
    });
}

void
QueueController::print_debug_signals() const
{
#if !defined(NDEBUG)
    _debug_signals.rlock([&](const auto& signals) {
        for(const auto& [id, signal] : signals)
        {
            ROCP_ERROR << "Signal " << signal.handle << " "
                       << get_core_table().hsa_signal_load_scacquire_fn(signal);
        }
    });
#endif

    _queues.rlock([&](const auto& queues) {
        for(const auto& [_, queue] : queues)
        {
            ROCP_ERROR << "Queue " << queue->get_id().handle << " " << queue->ready_signal.handle
                       << ":" << get_core_table().hsa_signal_load_scacquire_fn(queue->ready_signal)
                       << " " << queue->block_signal.handle << ":"
                       << get_core_table().hsa_signal_load_scacquire_fn(queue->block_signal);
        }
    });
}

void
QueueController::set_queue_state(queue_state state, hsa_queue_t* hsa_queue)
{
    _queues.wlock([&](auto& map) { map[hsa_queue]->set_state(state); });
}

void
QueueController::iterate_queues(const queue_iterator_cb_t& cb) const
{
    _queues.rlock([&cb](const queue_map_t& _queues_v) {
        for(const auto& itr : _queues_v)
        {
            if(itr.second) cb(itr.second.get());
        }
    });
}

void
QueueController::iterate_callbacks(const callback_iterator_cb_t& cb) const
{
    _callback_cache.rlock([&cb](const auto& map) {
        for(const auto& [cid, tuple] : map)
        {
            cb(cid, tuple);
        }
    });
}

const QueueController::agent_cache_map_t&
QueueController::get_supported_agents() const
{
    return _supported_agents;
}

QueueController::agent_cache_map_t&
QueueController::get_supported_agents()
{
    return _supported_agents;
}

QueueController*
get_queue_controller()
{
    static auto*& controller = common::static_object<QueueController>::construct();
    return controller;
}

bool
enable_queue_intercept()
{
    for(const auto& itr : context::get_registered_contexts())
    {
        constexpr auto expected_context_size = 224UL;
        static_assert(
            sizeof(context::context) == expected_context_size,
            "If you added a new field to context struct, make sure there is a check here if it "
            "requires queue interception. Once you have done so, increment expected_context_size");

        bool has_kernel_tracing = itr->is_tracing(ROCPROFILER_CALLBACK_TRACING_KERNEL_DISPATCH) ||
                                  itr->is_tracing(ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH);

        bool has_scratch_reporting = itr->is_tracing(ROCPROFILER_CALLBACK_TRACING_SCRATCH_MEMORY) ||
                                     itr->is_tracing(ROCPROFILER_BUFFER_TRACING_SCRATCH_MEMORY);

        // Keep interception active for HIP_GRAPH subscribers (drives kernel_dispatch_count).
        bool has_hip_graph_tracing = itr->is_tracing(ROCPROFILER_BUFFER_TRACING_HIP_GRAPH);

        // Kernel replay drives its multi-pass loop from WriteInterceptor, so it needs the queue
        // interceptor even when no other service is configured.
        bool has_kernel_replay = itr->is_tracing(ROCPROFILER_CALLBACK_TRACING_KERNEL_REPLAY);

        if(itr->dispatch_counter_collection || itr->pc_sampler || has_kernel_tracing ||
           itr->dispatch_spm || has_scratch_reporting || itr->device_counter_collection ||
           (itr->device_thread_trace && itr->device_thread_trace->requires_queue_intercept()) ||
           itr->dispatch_thread_trace || has_hip_graph_tracing || has_kernel_replay)
            return true;
    }
    return false;
}

bool
context_needs_queue_interposition_tracing(const context::context* ctx)
{
    return ctx != nullptr && ctx->is_tracing_one_of(ROCPROFILER_CALLBACK_TRACING_KERNEL_DISPATCH,
                                                    ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH,
                                                    ROCPROFILER_CALLBACK_TRACING_SCRATCH_MEMORY,
                                                    ROCPROFILER_BUFFER_TRACING_SCRATCH_MEMORY,
                                                    ROCPROFILER_CALLBACK_TRACING_KERNEL_REPLAY);
}

void
queue_controller_init(HsaApiTable* table)
{
    CHECK_NOTNULL(get_queue_controller())->init(*table->core_, *table->amd_ext_);

    if(enable_queue_intercept()) queue_init();
}

void
queue_controller_sync()
{
    // sync the queue interceptor
    queue_interposition::interposition_sync();

    if(get_queue_controller())
        get_queue_controller()->iterate_queues([](const Queue* _queue) { _queue->sync(); });
}

void
queue_controller_fini()
{
    // synchronize first
    queue_controller_sync();

    // finalize queue data (e.g. clean up signal pool)
    if(enable_queue_intercept()) queue_fini();

    queue_interposition::interposition_fini();
}

void
queue_controller_init(RocAttachDispatchTable* attach_table)
{
    // We need to save the attach table for later, when the queue controller receives the HSA table
    // and is initialized. We must get the attach table before HSA for correct behavior. This is
    // guaranteed by rocprofiler-register.
    if(get_queue_controller())
    {
        ROCP_ERROR_IF(get_queue_controller()->get_core_table().version.major_id != 0)
            << "Queue controller was initialized before attach table was provided. Future queues "
               "may not be instrumented correctly.";
    }
    *(get_attach_table()) = attach_table;

    if(enable_queue_intercept()) queue_init();
}

std::optional<uint32_t>
capture_doorbell_key(const hsa_queue_t* intercept_queue)
{
    // Extract the queue's hardware doorbell pointer from its intercept queue's
    // doorbell signal (HSA-internal amd_signal_t layout; same pattern as
    // hsa/async_copy.cpp). nullopt if unavailable -> caller falls back to HSA.
    uint64_t hwptr = 0;
    if(intercept_queue != nullptr && intercept_queue->doorbell_signal.handle != 0)
    {
        // hsa_signal_t::handle IS the address of the amd_signal_t in the AMD HSA
        // ABI, so the int-to-ptr conversion is the only way to reach it; same
        // construct as queue_interposition.cpp's lookup_queue_state_by_doorbell.
        const uint64_t _h = intercept_queue->doorbell_signal.handle;
        // NOLINTNEXTLINE(performance-no-int-to-ptr)
        const auto* sig = reinterpret_cast<const amd_signal_t*>(_h);
        // hardware_doorbell_ptr aliases other union members for non-doorbell kinds.
        if(sig->kind == AMD_SIGNAL_KIND_DOORBELL || sig->kind == AMD_SIGNAL_KIND_LEGACY_DOORBELL)
            hwptr = reinterpret_cast<uint64_t>(sig->hardware_doorbell_ptr);
    }
    if(hwptr == 0) return std::nullopt;

    // Page-relative doorbell slot; must match what the reader derives from each
    // firmware record (kfd::doorbell_off_to_page_slot). The 4 KiB / 1024-dword
    // mask is baked in -- no sysconf, no bind (open_window binds now).
    return kfd::doorbell_ptr_to_page_slot(hwptr);
}

}  // namespace hsa
}  // namespace rocprofiler
