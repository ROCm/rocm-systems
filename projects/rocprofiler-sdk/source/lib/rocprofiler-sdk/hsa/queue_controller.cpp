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

#include <hsa/amd_hsa_queue.h>

#include <rocprofiler-sdk/fwd.h>
#include <algorithm>
#include <cstddef>
#include <memory>
#include <vector>

namespace rocprofiler
{
namespace hsa
{
namespace
{
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
    if(auto* controller = get_queue_controller())
        return destroy_queue_with_runtime_fallback(
            *controller, hsa_queue, controller->get_core_table().hsa_queue_destroy_fn);
    return HSA_STATUS_SUCCESS;
}

#if HSA_AMD_EXT_API_TABLE_STEP_VERSION >= 0x10
bool
is_supported_agent(const QueueController& controller,
                   hsa_agent_t            agent,
                   const AgentCache**     agent_info)
{
    for(const auto& [_, itr] : controller.get_supported_agents())
    {
        if(itr.get_hsa_agent().handle == agent.handle)
        {
            if(agent_info) *agent_info = &itr;
            return true;
        }
    }
    return false;
}

bool
is_compute_desc(const hsa_amd_queue_create_desc_t& desc)
{
    return desc.engine_type == HSA_AMD_QUEUE_ENGINE_COMPUTE;
}

bool
is_zero(const void* data, size_t size)
{
    const auto* bytes = static_cast<const uint8_t*>(data);
    return std::all_of(bytes, bytes + size, [](uint8_t byte) { return byte == 0; });
}

bool
is_valid_old_queue_type(hsa_queue_type32_t type)
{
    return type == HSA_QUEUE_TYPE_MULTI || type == HSA_QUEUE_TYPE_SINGLE ||
           type == HSA_QUEUE_TYPE_COOPERATIVE;
}

void
register_queue(QueueController& controller, hsa_queue_t** queue_ptr, std::unique_ptr<Queue> queue)
{
    controller.serializer(queue.get()).wlock([&](auto& serializer) {
        serializer.add_queue(queue_ptr, *queue);
    });
    controller.add_queue(*queue_ptr, std::move(queue));
}

hsa_status_t
create_amd_queue(hsa_agent_t agent, hsa_amd_queue_create_desc_t* descs, uint32_t num_descs)
{
    auto* controller = CHECK_NOTNULL(get_queue_controller());
    auto  raw_create = controller->get_ext_table().hsa_amd_queue_create_fn;
    if(!raw_create) return HSA_STATUS_ERROR_NOT_INITIALIZED;

    const AgentCache* agent_info = nullptr;
    if(!is_supported_agent(*controller, agent, &agent_info))
    {
        if(descs)
        {
            for(uint32_t i = 0; i < num_descs; ++i)
                descs[i].queue = nullptr;
        }

        auto status = raw_create(agent, descs, num_descs);
        if(descs && queue_interposition::supports_queue_interposition())
        {
            for(uint32_t i = 0; i < num_descs; ++i)
            {
                if(descs[i].queue) queue_interposition::ignore_queue_state(descs[i].queue);
            }
        }
        return status;
    }

    if(!descs || num_descs == 0) return raw_create(agent, descs, num_descs);

    if(queue_interposition::supports_queue_interposition())
    {
        auto old_intercept_compatible_desc = std::vector<bool>{};
        old_intercept_compatible_desc.reserve(num_descs);
        for(uint32_t i = 0; i < num_descs; ++i)
        {
            descs[i].queue = nullptr;
            old_intercept_compatible_desc.emplace_back(
                !should_ignore_inline_amd_queue_create_desc(descs[i]));
        }

        auto status = raw_create(agent, descs, num_descs);
        for(uint32_t i = 0; i < num_descs; ++i)
        {
            auto& desc = descs[i];
            if(!desc.queue) continue;

            if(!old_intercept_compatible_desc[i])
            {
                queue_interposition::ignore_queue_state(desc.queue);
                if(is_compute_desc(desc))
                    ROCP_WARNING
                        << "hsa_amd_queue_create compute descriptor is not in the tested old "
                           "intercept compatibility envelope; creating it without queue "
                           "interception";
                continue;
            }

            // TODO: Expand inline adoption after descriptor features outside the old
            // intercept API envelope have tests and compatibility guarantees.
            auto new_queue = std::make_unique<Queue>(*agent_info,
                                                     controller->get_core_table(),
                                                     controller->get_ext_table(),
                                                     desc.queue,
                                                     [](write_interceptor_t, void*) {});
            register_queue(*controller, &desc.queue, std::move(new_queue));
            ROCP_INFO << "created queue for HSA agent handle " << agent.handle;
        }
        return status;
    }

    auto first_error  = HSA_STATUS_SUCCESS;
    auto record_error = [&first_error](hsa_status_t status) {
        if(first_error == HSA_STATUS_SUCCESS && status != HSA_STATUS_SUCCESS) first_error = status;
    };

    for(uint32_t i = 0; i < num_descs; ++i)
    {
        auto& desc = descs[i];
        desc.queue = nullptr;

        uint32_t   queue_size_packets = 0;
        const auto old_intercept_compatible =
            is_amd_queue_create_desc_old_intercept_compatible(desc, &queue_size_packets);

        if(old_intercept_compatible)
        {
            // TODO: The legacy Queue constructor still fatals on intercept-create/profiler
            // registration failures. Fully preserving per-descriptor batch partial-failure
            // semantics requires a non-fatal factory or ROCR descriptor-aware intercept-create.
            auto new_queue = std::make_unique<Queue>(*agent_info,
                                                     queue_size_packets,
                                                     desc.engine.compute.type,
                                                     desc.callback,
                                                     desc.callback_data,
                                                     desc.engine.compute.private_segment_size,
                                                     UINT32_MAX,
                                                     controller->get_core_table(),
                                                     controller->get_ext_table(),
                                                     &desc.queue);
            register_queue(*controller, &desc.queue, std::move(new_queue));
            ROCP_INFO << "created queue for HSA agent handle " << agent.handle;
            continue;
        }

        if(is_compute_desc(desc))
        {
            ROCP_WARNING << "hsa_amd_queue_create compute descriptor is not representable by the "
                            "legacy intercept API; creating it without queue interception";
        }
        auto status = raw_create(agent, &desc, 1);
        if(status == HSA_STATUS_SUCCESS && desc.queue)
            queue_interposition::ignore_queue_state(desc.queue);
        record_error(status);
    }

    return first_error;
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
            qc->add_queue(queue, std::move(new_queue));
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
QueueController::add_queue(hsa_queue_t* id, std::unique_ptr<Queue> queue)
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

    // Register queue state for SDK-level write pointer interception
    queue_interposition::create_queue_state(id);
}

bool
QueueController::destroy_queue(hsa_queue_t* id)
{
    if(!id) return false;

    const auto* queue = get_queue(*id);

    // return if queue does not exist
    if(!queue) return false;

    if(queue_interposition::supports_queue_interposition())
        queue_interposition::interposition_sync();
    queue_interposition::destroy_queue_state(id);
    queue->sync();
    if(queue->block_signal.handle != 0) get_core_table().hsa_signal_destroy_fn(queue->block_signal);
    _queues.wlock([&](auto& map) { map.erase(id); });
    return true;
}

hsa_status_t
destroy_queue_with_runtime_fallback(QueueController& controller,
                                    hsa_queue_t*     hsa_queue,
                                    hsa_status_t (*raw_destroy)(hsa_queue_t*))
{
    if(controller.destroy_queue(hsa_queue))
    {
        if(queue_interposition::supports_queue_interposition() && raw_destroy)
            return raw_destroy(hsa_queue);
        return HSA_STATUS_SUCCESS;
    }

    queue_interposition::unignore_queue_state(hsa_queue);
    if(raw_destroy) return raw_destroy(hsa_queue);
    return HSA_STATUS_SUCCESS;
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
#if HSA_AMD_EXT_API_TABLE_STEP_VERSION >= 0x10
            constexpr auto hsa_amd_queue_create_offset =
                offsetof(AmdExtTable, hsa_amd_queue_create_fn);
            if(ext_table.version.minor_id >=
                   hsa_amd_queue_create_offset + sizeof(ext_table.hsa_amd_queue_create_fn) &&
               ext_table.hsa_amd_queue_create_fn)
                ext_table.hsa_amd_queue_create_fn = hsa::create_amd_queue;
#endif
        }
    }
}

#if HSA_AMD_EXT_API_TABLE_STEP_VERSION >= 0x10
bool
is_amd_queue_create_desc_old_intercept_compatible(const hsa_amd_queue_create_desc_t& desc,
                                                  uint32_t* queue_size_packets)
{
    // Structural filter only. Runtime/intercept-create still validate agent-specific queue
    // constraints, including whether a given compute queue type is supported for this agent.
    if(desc.version != HSA_AMD_QUEUE_CREATE_DESC_VERSION) return false;
    if(desc.flags != HSA_AMD_QUEUE_CREATE_SYSTEM_MEM) return false;
    if(desc.engine_type != HSA_AMD_QUEUE_ENGINE_COMPUTE) return false;
    if(!is_zero(desc.reserved_header, sizeof(desc.reserved_header))) return false;
    if(!is_zero(desc.reserved, sizeof(desc.reserved))) return false;
    if(desc.traffic_class != 0) return false;
    if(desc.priority != HSA_AMD_QUEUE_PRIORITY_NORMAL) return false;

    const auto& compute = desc.engine.compute;
    if(!is_valid_old_queue_type(compute.type)) return false;
    if(compute.cu_mask != nullptr || compute.cu_mask_count != 0) return false;
    if(!is_zero(compute.reserved, sizeof(compute.reserved))) return false;

    if(desc.queue_size_bytes == 0 || (desc.queue_size_bytes & (desc.queue_size_bytes - 1)) != 0)
        return false;
    if(desc.queue_size_bytes % sizeof(hsa_kernel_dispatch_packet_t) != 0) return false;

    const auto packet_count = desc.queue_size_bytes / sizeof(hsa_kernel_dispatch_packet_t);
    if(packet_count < 3) return false;

    if(queue_size_packets) *queue_size_packets = packet_count;
    return true;
}

bool
should_ignore_inline_amd_queue_create_desc(const hsa_amd_queue_create_desc_t& desc)
{
    return !is_amd_queue_create_desc_old_intercept_compatible(desc);
}
#endif

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

        if(itr->dispatch_counter_collection || itr->pc_sampler || has_kernel_tracing ||
           itr->dispatch_spm || has_scratch_reporting || itr->device_counter_collection ||
           itr->device_thread_trace || itr->dispatch_thread_trace)
            return true;
    }
    return false;
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

}  // namespace hsa
}  // namespace rocprofiler
