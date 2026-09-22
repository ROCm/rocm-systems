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

#include "lib/rocprofiler-sdk/context/context.hpp"
#include "lib/common/container/small_vector.hpp"
#include "lib/common/container/stable_vector.hpp"
#include "lib/common/scope_destructor.hpp"
#include "lib/common/static_object.hpp"
#include "lib/common/synchronized.hpp"
#include "lib/common/utility.hpp"
#include "lib/rocprofiler-sdk/buffer.hpp"
#include "lib/rocprofiler-sdk/counters/core.hpp"
#include "lib/rocprofiler-sdk/hsa/queue_interposition.hpp"
#include "lib/rocprofiler-sdk/kfd/kfd_profiler.hpp"
#include "lib/rocprofiler-sdk/pc_sampling/service.hpp"
#include "lib/rocprofiler-sdk/thread_trace/core.hpp"

#include <rocprofiler-sdk/buffer_tracing.h>
#include <rocprofiler-sdk/fwd.h>

#include <unistd.h>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <random>
#include <unordered_set>
#include <utility>
#include <vector>

namespace rocprofiler
{
namespace context
{
// Owns the contexts it names. A reader holding a snapshot keeps them alive for the duration of
// the read, so deregister_client_contexts() dropping the registry's reference cannot destroy a
// context out from under an in-flight completion. The parallel raw-pointer array is what the
// snapshot iterates, so walking one costs no reference counting.
struct registered_context_storage
{
    std::vector<std::shared_ptr<context>> owned = {};
    std::vector<const context*>           view  = {};
    // Positional, unlike `view`: index i is registry slot i, holding nullptr where the slot is
    // empty. Resolving a context id is an index operation, so it needs the slot numbering that
    // allocate_context() assigns, which `view` loses by skipping empty slots.
    std::vector<context*> slots = {};
};

namespace
{
using reserve_size_t         = common::container::reserve_size;
using context_ptr_t          = std::shared_ptr<context>;
using stable_context_vec_t   = common::container::stable_vector<context_ptr_t, 8>;
using active_context_vec_t   = common::container::stable_vector<std::atomic<const context*>, 8>;
using context_snapshot_ptr_t = std::shared_ptr<const registered_context_storage>;

constexpr auto invalid_client_idx = std::numeric_limits<uint32_t>::max();

uint64_t
get_contexts_offset()
{
    static uint64_t _v = []() {
        auto gen = std::mt19937{std::random_device{}()};
        auto rng = std::uniform_int_distribution<uint64_t>{std::numeric_limits<uint8_t>::max(),
                                                           std::numeric_limits<uint16_t>::max()};
        return rng(gen);
    }();
    return _v;
}

auto&
get_client_index()
{
    static auto _v = invalid_client_idx;
    return _v;
}

auto&
get_num_active_contexts()
{
    static auto _v = std::atomic<int64_t>{0};
    return _v;
}

active_context_vec_t&
get_active_contexts_impl()
{
    static auto* _v = new active_context_vec_t{reserve_size_t{active_context_vec_t::chunk_size}};
    return *_v;
}

// C++20 replaces these with std::atomic<std::shared_ptr<T>>; the free functions are the C++17
// spelling and are deprecated there, so select on the feature macro rather than leaving a
// deprecation warning to trip -Werror on a future standard bump.
#if defined(__cpp_lib_atomic_shared_ptr) && __cpp_lib_atomic_shared_ptr >= 201711L
using published_snapshot_t = std::atomic<context_snapshot_ptr_t>;

context_snapshot_ptr_t
load_snapshot(published_snapshot_t& _v)
{
    return _v.load(std::memory_order_acquire);
}

void
store_snapshot(published_snapshot_t& _v, context_snapshot_ptr_t&& _new)
{
    _v.store(std::move(_new), std::memory_order_release);
}
#else
using published_snapshot_t = context_snapshot_ptr_t;

context_snapshot_ptr_t
load_snapshot(published_snapshot_t& _v)
{
    return std::atomic_load_explicit(&_v, std::memory_order_acquire);
}

void
store_snapshot(published_snapshot_t& _v, context_snapshot_ptr_t&& _new)
{
    std::atomic_store_explicit(&_v, std::move(_new), std::memory_order_release);
}
#endif

// The registry and the snapshot published from it live in one static object rather than two.
// common::destroy_static_objects() tears static objects down in reverse construction order, so two
// objects would be ordered by whichever happened to be touched first -- and if the snapshot
// outlived the registry it would be holding the last reference to every context, destroying them
// after the HSA tables that a context destructor calls into are already gone. As members, the
// snapshot is destroyed before the registry by the ordinary reverse-member rule.
struct registry_state
{
    stable_context_vec_t registry =
        stable_context_vec_t{reserve_size_t{stable_context_vec_t::chunk_size}};
    published_snapshot_t published = {};
    // Contexts whose client has deregistered. They are unreachable through the registry and the
    // snapshot, but they are not destroyed here: get_registered_contexts() and
    // get_active_contexts() hand out raw pointers that outlive the snapshot they were read from,
    // and those pointers are spread across every tracing service. Holding the last reference until
    // static teardown gives them the lifetime they had when the registry stored the contexts
    // by value, which is the lifetime all of those callers were written against.
    std::vector<context_ptr_t> retired = {};
};

registry_state*
get_registry_state()
{
    static auto*& _v = common::static_object<registry_state>::construct();
    return _v;
}

stable_context_vec_t*
get_registered_contexts_impl()
{
    auto* _state = get_registry_state();
    return (_state) ? &_state->registry : nullptr;
}

published_snapshot_t*
get_published_snapshot()
{
    auto* _state = get_registry_state();
    return (_state) ? &_state->published : nullptr;
}

std::vector<context_ptr_t>*
get_retired_contexts()
{
    auto* _state = get_registry_state();
    return (_state) ? &_state->retired : nullptr;
}

context_snapshot_ptr_t
current_registered_contexts()
{
    auto* _pub = get_published_snapshot();
    if(!_pub) return {};
    return load_snapshot(*_pub);
}

// Rebuilds the published snapshot from the registry. Must be called with get_contexts_mutex()
// held, i.e. by whichever function just mutated the registry.
void
publish_registered_contexts()
{
    auto* _pub = get_published_snapshot();
    if(!_pub) return;

    auto _data = std::make_shared<registered_context_storage>();

    if(auto* _impl = get_registered_contexts_impl())
    {
        _data->owned.reserve(_impl->size());
        _data->view.reserve(_impl->size());
        _data->slots.reserve(_impl->size());
        for(const auto& itr : *_impl)
        {
            _data->slots.emplace_back(itr.get());
            if(!itr) continue;
            _data->owned.emplace_back(itr);
            _data->view.emplace_back(itr.get());
        }
    }

    store_snapshot(*_pub, std::move(_data));
}

// The registry is index-addressed: allocate_context() derives context_idx from the slot position,
// so a context id maps straight back to its slot.
//
// Requires get_contexts_mutex(). The registry is a stable_vector, which keeps element addresses
// stable but reallocates the chunk index that at() walks, so an unlocked reader can index a freed
// buffer. Callers that cannot take the mutex must go through lookup_registered_context() instead.
context_ptr_t
get_registered_context_slot(uint64_t handle)
{
    auto* _impl = get_registered_contexts_impl();
    if(!_impl || handle < get_contexts_offset()) return {};

    auto _idx = handle - get_contexts_offset();
    if(_idx >= _impl->size()) return {};

    return _impl->at(_idx);
}

// Resolves a context id off the published snapshot, for the callers that hold no lock. The
// snapshot is immutable once published and owns the contexts it names, so indexing it neither
// races allocate_context() nor touches a reference count.
context*
lookup_registered_context(uint64_t handle)
{
    if(handle < get_contexts_offset()) return nullptr;

    auto _snapshot = current_registered_contexts();
    if(!_snapshot) return nullptr;

    auto _idx = handle - get_contexts_offset();
    if(_idx >= _snapshot->slots.size()) return nullptr;

    return _snapshot->slots[_idx];
}

// Context ids whose stop is in progress. stop_context() releases get_contexts_mutex() across the
// GPU drain, so for that window the context is still in the active array with no lock held. This
// set is what keeps start_context() and a second stop_context() from acting on that half-stopped
// state. Guarded by get_contexts_mutex().
std::unordered_set<uint64_t>&
get_stopping_contexts()
{
    static auto _v = std::unordered_set<uint64_t>{};
    return _v;
}

std::condition_variable&
get_contexts_cv()
{
    static auto _v = std::condition_variable{};
    return _v;
}

}  // namespace

// Waits until no context is mid-stop, which restores the property the old single-lock
// stop_context() had: no other lifecycle operation ever observes a context whose services have
// been torn down but whose active slot is still populated.
void
wait_for_stopping_contexts(std::unique_lock<std::mutex>& _lk)
{
    get_contexts_cv().wait(_lk, []() { return get_stopping_contexts().empty(); });
}

std::mutex&
get_contexts_mutex()
{
    static auto _v = std::mutex{};
    return _v;
}

bool
dispatch_counter_collection_service::intersects(
    const dispatch_counter_collection_service& rhs) const
{
    if(agents.empty() || rhs.agents.empty()) return true;
    const auto& small = (agents.size() < rhs.agents.size()) ? agents : rhs.agents;
    const auto& large = (agents.size() < rhs.agents.size()) ? rhs.agents : agents;
    for(const auto& agent_id : small)
    {
        if(large.count(agent_id) > 0) return true;
    }
    return false;
}

registered_contexts_snapshot::registered_contexts_snapshot(context_snapshot_ptr_t&& data)
: m_data{std::move(data)}
{
    if(m_data)
    {
        m_begin = m_data->view.data();
        m_end   = m_begin + m_data->view.size();
    }
}

registered_contexts_snapshot
get_registered_contexts_snapshot()
{
    return registered_contexts_snapshot{current_registered_contexts()};
}

context_array_t&
get_registered_contexts(context_array_t& data, context_filter_t filter)
{
    data.clear();

    auto snapshot = get_registered_contexts_snapshot();
    if(snapshot.empty()) return data;

    data.reserve(snapshot.size());
    for(const auto* ctx : snapshot)
    {
        if(!filter || (filter && filter(ctx))) data.emplace_back(ctx);
    }
    return data;
}

context_array_t
get_registered_contexts(context_filter_t filter)
{
    auto data = context_array_t{};
    get_registered_contexts(data, filter);
    return data;
}

context_array_t&
get_active_contexts(context_array_t& data, context_filter_t filter)
{
    data.clear();
    auto num_ctx = get_num_active_contexts().load(std::memory_order_acquire);
    if(num_ctx <= 0) return data;

    data.reserve(num_ctx);
    for(auto& itr : get_active_contexts_impl())
    {
        const auto* ctx = itr.load(std::memory_order_acquire);
        if(ctx)
        {
            if(!filter || (filter && filter(ctx))) data.emplace_back(ctx);
        }
        if(static_cast<int64_t>(data.size()) == num_ctx)
        {
            // if the number of active contexts changed, restart
            if(num_ctx != get_num_active_contexts().load(std::memory_order_relaxed))
            {
                data.clear();
                return get_active_contexts(data, filter);
            }
            break;
        }
    }
    return data;
}

context_array_t
get_active_contexts(context_filter_t filter)
{
    auto data = context_array_t{};
    get_active_contexts(data, filter);
    return data;
}

const context*
get_active_context(rocprofiler_context_id_t id)
{
    if(get_num_active_contexts().load(std::memory_order_acquire) > 0)
    {
        for(auto& itr : get_active_contexts_impl())
        {
            const auto* ctx = itr.load(std::memory_order_acquire);
            if(ctx && ctx->context_idx == id.handle) return ctx;
        }
    }
    return nullptr;
}

// set the client index needs to be called before allocate_context()
void
push_client(uint32_t value)
{
    LOG_ASSERT(get_client_index() == invalid_client_idx)
        << " rocprofiler client index is currently " << get_client_index()
        << "... which means that a new client is initializing before the last client finished "
           "initializing. This is an internal error, please file a bug report with a reproducer";
    get_client_index() = value;
}

// remove the client index
void
pop_client(uint32_t value)
{
    LOG_ASSERT(get_client_index() == value)
        << " rocprofiler client index is currently not " << value
        << "... which means that a new client was initialized before this client finished "
           "initializing. This is an internal error, please file a bug report with a reproducer";
    get_client_index() = invalid_client_idx;
}

std::optional<rocprofiler_context_id_t>
allocate_context()
{
    // ... allocate any internal space needed to handle another context ...
    auto _lk = std::unique_lock<std::mutex>{get_contexts_mutex()};

    // initial context identifier number
    auto _idx = get_registered_contexts_impl()->size() + get_contexts_offset();

    // create an entry in the registered
    auto& _cfg_v = get_registered_contexts_impl()->emplace_back(std::make_shared<context>());
    auto* _cfg   = _cfg_v.get();
    // ...

    if(!_cfg) return std::nullopt;

    _cfg->size        = sizeof(context);
    _cfg->context_idx = _idx;
    _cfg->client_idx  = get_client_index();

    LOG_ASSERT(_cfg->client_idx != invalid_client_idx)
        << " rocprofiler internal error: a context was allocated without an associated tool client "
           "identifier";

    publish_registered_contexts();

    return rocprofiler_context_id_t{_idx};
}

context*
get_mutable_registered_context(rocprofiler_context_id_t id)
{
    return lookup_registered_context(id.handle);
}

const context*
get_registered_context(rocprofiler_context_id_t id)
{
    return get_mutable_registered_context(id);
}

rocprofiler_status_t
validate_context(const context* cfg)
{
    return (cfg) ? ROCPROFILER_STATUS_SUCCESS : ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_FOUND;
}

rocprofiler_status_t
start_context(rocprofiler_context_id_t context_id)
{
    if(context_id.handle < get_contexts_offset()) return ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_FOUND;

    // Held for the rest of the function: the service starts below run without the contexts mutex,
    // and deregister_client_contexts() may drop the registry's reference while they do.
    auto           _owner = context_ptr_t{};
    const context* cfg    = nullptr;
    {
        auto _lk = std::unique_lock<std::mutex>{get_contexts_mutex()};
        _owner   = get_registered_context_slot(context_id.handle);
        cfg      = _owner.get();
    }

    if(!cfg) return ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_FOUND;

    if(validate_context(cfg) != ROCPROFILER_STATUS_SUCCESS)
        return ROCPROFILER_STATUS_ERROR_CONTEXT_INVALID;

    uint64_t rocp_tot_contexts = 0;
    auto     idx               = rocp_tot_contexts;
    auto&    active_contexts   = get_active_contexts_impl();
    {
        // hold a lock here to prevent multiple threads from finding the same nullptr slot
        auto _lk = std::unique_lock<std::mutex>{get_contexts_mutex()};

        // Sized under the lock: size() walks the registry's chunk index, which allocate_context()
        // reallocates as it grows.
        rocp_tot_contexts = get_registered_contexts_impl()->size();
        idx               = rocp_tot_contexts;

        // A context that is mid-stop is still in the active array while its GPU drain runs, so
        // scanning against that state would either report a conflict against a context that is on
        // its way out, or hand back a slot that the stop is about to clear.
        wait_for_stopping_contexts(_lk);

        auto current_contexts = context_array_t{};
        for(const auto* itr : get_active_contexts(current_contexts))
        {
            if(cfg->context_idx == itr->context_idx)
            {
                return ROCPROFILER_STATUS_SUCCESS;
            }
            else if(cfg->dispatch_counter_collection && itr->dispatch_counter_collection &&
                    cfg->dispatch_counter_collection->intersects(*itr->dispatch_counter_collection))
            {
                // Conflicting context. Two counter-collection contexts can run concurrently as
                // long as they target disjoint sets of GPU agents -- the hardware counters they
                // program are per-agent, so contexts that never touch the same agent cannot
                // contend. A context with no agent restriction claims every agent and therefore
                // still conflicts with any other counter-collection context.
                return ROCPROFILER_STATUS_ERROR_CONTEXT_CONFLICT;
            }
        }

        // try to find a nullptr slot first
        for(size_t i = 0; i < active_contexts.size(); ++i)
        {
            const auto* itr = active_contexts.at(i).load(std::memory_order_relaxed);
            if(itr == nullptr)
            {
                idx = i;
                break;
            }
            else if(context_id.handle == itr->context_idx)
            {
                return ROCPROFILER_STATUS_SUCCESS;
            }
        }

        // if no nullptr slot was found, then create one while lock is held
        if(idx == rocp_tot_contexts)
        {
            idx = active_contexts.size();
            active_contexts.emplace_back();
        }

        get_num_active_contexts().fetch_add(1, std::memory_order_release);
    }

    rocprofiler::hsa::queue_interposition::notify_queue_interposition_consumer_context_started(cfg);

    // atomic swap the pointer into the "active" array used internally
    const context* _expected = nullptr;
    bool           success   = active_contexts.at(idx).compare_exchange_strong(_expected, cfg);

    if(!success)
    {
        rocprofiler::hsa::queue_interposition::notify_queue_interposition_consumer_context_stopped(
            cfg);
        get_num_active_contexts().fetch_sub(1, std::memory_order_release);
        return ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_STARTED;
    }

    auto status = ROCPROFILER_STATUS_SUCCESS;

    // A context that traces kernel dispatch is the only reason to arm the KFD
    // dispatch-log ring, so arm it here rather than at startup. Idempotent, and
    // early enough that the firmware is recording before the first dispatch.
    if(cfg->is_tracing_one_of(ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH,
                              ROCPROFILER_CALLBACK_TRACING_KERNEL_DISPATCH))
        rocprofiler::kfd::arm_dispatch_log_sessions();

    if(cfg->dispatch_counter_collection) rocprofiler::counters::start_context(cfg);
    if(cfg->dispatch_spm) status = rocprofiler::spm::start_context(cfg);
    if(cfg->device_thread_trace) cfg->device_thread_trace->start_context();
    if(cfg->dispatch_thread_trace) cfg->dispatch_thread_trace->start_context();
    if(cfg->device_counter_collection) status = rocprofiler::counters::start_agent_ctx(cfg);
#if ROCPROFILER_SDK_HSA_PC_SAMPLING > 0
    if(cfg->pc_sampler) status = rocprofiler::pc_sampling::start_service(cfg);
#endif

    return status;
}

rocprofiler_status_t
stop_context(rocprofiler_context_id_t idx)
{
    std::atomic<const context*>* slot      = nullptr;
    const context*               _expected = nullptr;
    // Held for the rest of the function. The teardown below runs without the contexts mutex, so
    // without this the registry's reference is the only thing keeping the context alive and
    // deregister_client_contexts() could destroy it out from under the drain.
    auto _owner = context_ptr_t{};

    // Phase one, locked: claim the context. Only the search of the active array needs the lock;
    // the teardown below must not hold it.
    {
        auto _lk = std::unique_lock<std::mutex>{get_contexts_mutex()};

        // another thread may already be tearing this context down
        wait_for_stopping_contexts(_lk);

        for(auto& itr : get_active_contexts_impl())
        {
            const context* _ctx = itr.load(std::memory_order_acquire);
            if(_ctx && _ctx->context_idx == idx.handle)
            {
                slot      = &itr;
                _expected = _ctx;
                break;
            }
        }

        if(!slot) return ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_FOUND;

        _owner = get_registered_context_slot(idx.handle);
        get_stopping_contexts().emplace(idx.handle);
    }

    auto _unclaim = common::scope_destructor{[&idx]() {
        {
            auto _lk = std::unique_lock<std::mutex>{get_contexts_mutex()};
            get_stopping_contexts().erase(idx.handle);
        }
        get_contexts_cv().notify_all();
    }};

    // Phase two, unlocked: the service teardowns below call hsa::queue_controller_sync(), an
    // unbounded wait on in-flight GPU work. Holding get_contexts_mutex() across it stalls every
    // context lifecycle operation in the process behind one context's dispatches, and it puts the
    // mutex on the far side of a wait that the completion path has to get through -- so any future
    // completion-path read that took the mutex would deadlock rather than merely block.
    //
    // The active slot stays populated for the whole phase, which is deliberate:
    // kernel_dispatch_phase_enter_hook has to keep seeing the context so the serialized ->
    // unserialized transition stays coordinated until disable_serialization() has run. Clearing
    // the slot first opens the opposite window, in which the enter hook sees no active context and
    // a dispatch is submitted without serializer packets while the serializer is still enabled.
    // get_stopping_contexts() is what keeps other lifecycle callers from reading that state as a
    // running context.
    if(_expected->dispatch_counter_collection)
    {
        rocprofiler::counters::stop_context(const_cast<context*>(_expected));
    }

    if(_expected->dispatch_spm) rocprofiler::spm::stop_context(const_cast<context*>(_expected));

    if(_expected->device_thread_trace) _expected->device_thread_trace->stop_context();
    if(_expected->dispatch_thread_trace) _expected->dispatch_thread_trace->stop_context();

    // Phase three, relocked: retire the slot. The element address is stable across phase two --
    // stable_vector never moves an element once constructed -- so slot is still the entry the
    // search found.
    {
        auto _lk = std::unique_lock<std::mutex>{get_contexts_mutex()};

        if(!slot->compare_exchange_strong(_expected, nullptr))
        {
            // Not reachable as written: deactivate_client_contexts() is the only other writer of
            // this slot and the stopping marker holds it off until phase four returns. Kept
            // because the pre-phasing code bailed out the same way on a lost exchange.
            return ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_FOUND;
        }

        auto nactive = get_num_active_contexts().load(std::memory_order_acquire);
        if(nactive > 0) get_num_active_contexts().fetch_sub(1, std::memory_order_release);
    }

    // Phase four, unlocked: the remaining services are not queue-interposed, so they only need
    // the slot to be cleared first, not the lock to be held. They wait on the GPU too --
    // stop_agent_ctx() waits for its stop packet and pc_sampling::stop_service() drains the
    // device buffers -- and the stopping marker above, not the mutex, is what keeps a concurrent
    // start_context() from running before they finish.
    rocprofiler::hsa::queue_interposition::notify_queue_interposition_consumer_context_stopped(
        _expected);

    if(_expected->device_counter_collection)
    {
        rocprofiler::counters::stop_agent_ctx(const_cast<context*>(_expected));
    }

#if ROCPROFILER_SDK_HSA_PC_SAMPLING > 0
    if(_expected->pc_sampler)
    {
        rocprofiler::pc_sampling::stop_service(_expected);
    }
#endif

    return ROCPROFILER_STATUS_SUCCESS;
}

context_id_array_t
get_client_contexts(rocprofiler_client_id_t id)
{
    auto _data = context_id_array_t{};

    for(const auto* itr : get_registered_contexts_snapshot())
    {
        if(itr->client_idx == id.handle)
        {
            _data.emplace_back(rocprofiler_context_id_t{.handle = itr->context_idx});
        }
    }
    return _data;
}

rocprofiler_status_t
stop_client_contexts(rocprofiler_client_id_t client_id)
{
    if(!get_registered_contexts_impl()) return ROCPROFILER_STATUS_ERROR;

    auto ret = ROCPROFILER_STATUS_SUCCESS;

    // Held across the loop: stop_context() may block on a GPU drain, and a snapshot is what keeps
    // the contexts it is iterating alive for that long.
    auto snapshot = get_registered_contexts_snapshot();
    for(const auto* itr : snapshot)
    {
        if(itr->client_idx == client_id.handle)
        {
            auto status = stop_context(rocprofiler_context_id_t{itr->context_idx});
            if(status != ROCPROFILER_STATUS_SUCCESS &&
               status != ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_FOUND)
                ret = status;
        }
    }
    return ret;
}

void
deactivate_client_contexts(rocprofiler_client_id_t client_id)
{
    auto _lk = std::unique_lock<std::mutex>{get_contexts_mutex()};

    // a context mid-stop is still in the active array; let its teardown finish rather than
    // clearing the slot underneath it
    wait_for_stopping_contexts(_lk);

    for(auto& itr : get_active_contexts_impl())
    {
        const context* itr_v = itr.load(std::memory_order_acquire);
        if(itr_v && itr_v->client_idx == client_id.handle)
        {
            if(itr.compare_exchange_strong(itr_v, nullptr))
            {
                rocprofiler::hsa::queue_interposition::
                    notify_queue_interposition_consumer_context_stopped(itr_v);
            }
        }
    }
}

void
deregister_client_contexts(rocprofiler_client_id_t client_id)
{
    if(!get_registered_contexts_impl()) return;

    {
        // Mutates the registry, so it needs the same lock allocate_context() takes.
        auto _lk = std::unique_lock<std::mutex>{get_contexts_mutex()};

        // a context mid-stop is still being torn down through a raw pointer; retiring its registry
        // entry now would pull the object out from under that teardown
        wait_for_stopping_contexts(_lk);

        for(auto& itr : *get_registered_contexts_impl())
        {
            if(!itr) continue;

            if(itr->client_idx == client_id.handle && buffer::get_buffers())
            {
                for(auto& bitr : *buffer::get_buffers())
                {
                    if(bitr && bitr->context_id == itr->context_idx) bitr.reset();
                }
                // Moved to the retired list rather than reset: dropping the registry's reference
                // here would destroy the context while dispatch and completion paths may still
                // hold raw pointers to it, which is what made the anytime-tool-config tests
                // segfault when one tool finalized while another tool's kernels were in flight.
                if(auto* _retired = get_retired_contexts())
                    _retired->emplace_back(std::move(itr));
                else
                    itr.reset();
            }
        }

        publish_registered_contexts();
    }
}

template <typename KindT>
bool
context::is_tracing(KindT _kind) const
{
    constexpr auto is_callback_tracing =
        std::is_same<KindT, rocprofiler_callback_tracing_kind_t>::value;
    constexpr auto is_buffered_tracing =
        std::is_same<KindT, rocprofiler_buffer_tracing_kind_t>::value;
    static_assert(is_callback_tracing || is_buffered_tracing, "Unsupported domain type");

    if constexpr(is_callback_tracing)
        return (callback_tracer && callback_tracer->domains(_kind));
    else if constexpr(is_buffered_tracing)
        return (buffered_tracer && buffered_tracer->domains(_kind));
}

template <typename KindT>
bool
context::is_tracing(KindT _kind, uint32_t _operation) const
{
    constexpr auto is_callback_tracing =
        std::is_same<KindT, rocprofiler_callback_tracing_kind_t>::value;
    constexpr auto is_buffered_tracing =
        std::is_same<KindT, rocprofiler_buffer_tracing_kind_t>::value;
    static_assert(is_callback_tracing || is_buffered_tracing, "Unsupported domain type");

    if constexpr(is_callback_tracing)
        return (callback_tracer && callback_tracer->domains(_kind, _operation));
    else if constexpr(is_buffered_tracing)
        return (buffered_tracer && buffered_tracer->domains(_kind, _operation));
}

// explicitly instantiate
template bool context::is_tracing(rocprofiler_callback_tracing_kind_t) const;
template bool context::is_tracing(rocprofiler_buffer_tracing_kind_t) const;

template bool context::is_tracing(rocprofiler_callback_tracing_kind_t, uint32_t) const;
template bool context::is_tracing(rocprofiler_buffer_tracing_kind_t, uint32_t) const;
}  // namespace context
}  // namespace rocprofiler
