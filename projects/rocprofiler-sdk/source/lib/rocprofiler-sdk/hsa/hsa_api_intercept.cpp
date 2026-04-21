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

#include "lib/rocprofiler-sdk/hsa/hsa_api_intercept.hpp"
#include "lib/rocprofiler-sdk/hsa/queue_intercept.hpp"
#include "lib/rocprofiler-sdk/hsa/queue_state_registry.hpp"
#include "lib/rocprofiler-sdk/registration.hpp"

#include <atomic>

namespace rocprofiler
{
namespace hsa
{
namespace queue_intercept
{
namespace
{
std::atomic<bool> s_intercept_installed = false;

// Saved next-in-chain function pointers (tracing functors or raw HSA, depending on
// when install_intercept is called). Our wrappers chain through these for untracked
// queues and for the final doorbell ring on tracked queues.
CoreApiTable s_next_table = {};

bool
should_bypass_inline_intercept()
{
    return (!s_intercept_installed.load(std::memory_order_acquire) ||
            registration::get_fini_status() > 0);
}

// --- add_write_index wrappers (4) ---

uint64_t
wrap_add_write_index_relaxed(const hsa_queue_t* q, uint64_t v)
{
    if(should_bypass_inline_intercept())
        return s_next_table.hsa_queue_add_write_index_relaxed_fn(q, v);

    auto s = lookup_queue_state(q);
    if(s) return add_write_index_impl(s.get(), v, std::memory_order_relaxed);
    return s_next_table.hsa_queue_add_write_index_relaxed_fn(q, v);
}

uint64_t
wrap_add_write_index_scacq_screl(const hsa_queue_t* q, uint64_t v)
{
    if(should_bypass_inline_intercept())
        return s_next_table.hsa_queue_add_write_index_scacq_screl_fn(q, v);

    auto s = lookup_queue_state(q);
    if(s) return add_write_index_impl(s.get(), v, std::memory_order_acq_rel);
    return s_next_table.hsa_queue_add_write_index_scacq_screl_fn(q, v);
}

uint64_t
wrap_add_write_index_scacquire(const hsa_queue_t* q, uint64_t v)
{
    if(should_bypass_inline_intercept())
        return s_next_table.hsa_queue_add_write_index_scacquire_fn(q, v);

    auto s = lookup_queue_state(q);
    if(s) return add_write_index_impl(s.get(), v, std::memory_order_acquire);
    return s_next_table.hsa_queue_add_write_index_scacquire_fn(q, v);
}

uint64_t
wrap_add_write_index_screlease(const hsa_queue_t* q, uint64_t v)
{
    if(should_bypass_inline_intercept())
        return s_next_table.hsa_queue_add_write_index_screlease_fn(q, v);

    auto s = lookup_queue_state(q);
    if(s) return add_write_index_impl(s.get(), v, std::memory_order_release);
    return s_next_table.hsa_queue_add_write_index_screlease_fn(q, v);
}

// --- store_write_index wrappers (2) ---

void
wrap_store_write_index_relaxed(const hsa_queue_t* q, uint64_t v)
{
    if(should_bypass_inline_intercept())
    {
        s_next_table.hsa_queue_store_write_index_relaxed_fn(q, v);
        return;
    }

    auto s = lookup_queue_state(q);
    if(s)
    {
        store_write_index_impl(s.get(), v, std::memory_order_relaxed);
        return;
    }
    s_next_table.hsa_queue_store_write_index_relaxed_fn(q, v);
}

void
wrap_store_write_index_screlease(const hsa_queue_t* q, uint64_t v)
{
    if(should_bypass_inline_intercept())
    {
        s_next_table.hsa_queue_store_write_index_screlease_fn(q, v);
        return;
    }

    auto s = lookup_queue_state(q);
    if(s)
    {
        store_write_index_impl(s.get(), v, std::memory_order_release);
        return;
    }
    s_next_table.hsa_queue_store_write_index_screlease_fn(q, v);
}

// --- cas_write_index wrappers (4) ---

uint64_t
wrap_cas_write_index_relaxed(const hsa_queue_t* q, uint64_t expected, uint64_t value)
{
    if(should_bypass_inline_intercept())
        return s_next_table.hsa_queue_cas_write_index_relaxed_fn(q, expected, value);

    auto s = lookup_queue_state(q);
    if(s) return cas_write_index_impl(s.get(), expected, value, std::memory_order_relaxed);
    return s_next_table.hsa_queue_cas_write_index_relaxed_fn(q, expected, value);
}

uint64_t
wrap_cas_write_index_scacq_screl(const hsa_queue_t* q, uint64_t expected, uint64_t value)
{
    if(should_bypass_inline_intercept())
        return s_next_table.hsa_queue_cas_write_index_scacq_screl_fn(q, expected, value);

    auto s = lookup_queue_state(q);
    if(s) return cas_write_index_impl(s.get(), expected, value, std::memory_order_acq_rel);
    return s_next_table.hsa_queue_cas_write_index_scacq_screl_fn(q, expected, value);
}

uint64_t
wrap_cas_write_index_scacquire(const hsa_queue_t* q, uint64_t expected, uint64_t value)
{
    if(should_bypass_inline_intercept())
        return s_next_table.hsa_queue_cas_write_index_scacquire_fn(q, expected, value);

    auto s = lookup_queue_state(q);
    if(s) return cas_write_index_impl(s.get(), expected, value, std::memory_order_acquire);
    return s_next_table.hsa_queue_cas_write_index_scacquire_fn(q, expected, value);
}

uint64_t
wrap_cas_write_index_screlease(const hsa_queue_t* q, uint64_t expected, uint64_t value)
{
    if(should_bypass_inline_intercept())
        return s_next_table.hsa_queue_cas_write_index_screlease_fn(q, expected, value);

    auto s = lookup_queue_state(q);
    if(s) return cas_write_index_impl(s.get(), expected, value, std::memory_order_release);
    return s_next_table.hsa_queue_cas_write_index_screlease_fn(q, expected, value);
}

// --- load_write_index wrappers (2) ---

uint64_t
wrap_load_write_index_relaxed(const hsa_queue_t* q)
{
    if(should_bypass_inline_intercept())
        return s_next_table.hsa_queue_load_write_index_relaxed_fn(q);

    auto s = lookup_queue_state(q);
    if(s) return load_write_index_impl(s.get(), std::memory_order_relaxed);
    return s_next_table.hsa_queue_load_write_index_relaxed_fn(q);
}

uint64_t
wrap_load_write_index_scacquire(const hsa_queue_t* q)
{
    if(should_bypass_inline_intercept())
        return s_next_table.hsa_queue_load_write_index_scacquire_fn(q);

    auto s = lookup_queue_state(q);
    if(s) return load_write_index_impl(s.get(), std::memory_order_acquire);
    return s_next_table.hsa_queue_load_write_index_scacquire_fn(q);
}

// --- signal_store wrappers (2) ---

void
wrap_signal_store_relaxed(hsa_signal_t sig, hsa_signal_value_t val)
{
    if(should_bypass_inline_intercept())
    {
        s_next_table.hsa_signal_store_relaxed_fn(sig, val);
        return;
    }

    auto s = lookup_queue_state_by_doorbell(sig);
    if(s)
    {
        process_doorbell_impl(s, val, [](hsa_signal_t db, hsa_signal_value_t v) {
            s_next_table.hsa_signal_store_relaxed_fn(db, v);
        });
        return;
    }
    s_next_table.hsa_signal_store_relaxed_fn(sig, val);
}

void
wrap_signal_store_screlease(hsa_signal_t sig, hsa_signal_value_t val)
{
    if(should_bypass_inline_intercept())
    {
        s_next_table.hsa_signal_store_screlease_fn(sig, val);
        return;
    }

    auto s = lookup_queue_state_by_doorbell(sig);
    if(s)
    {
        process_doorbell_impl(s, val, [](hsa_signal_t db, hsa_signal_value_t v) {
            s_next_table.hsa_signal_store_screlease_fn(db, v);
        });
        return;
    }
    s_next_table.hsa_signal_store_screlease_fn(sig, val);
}

// --- signal_silent_store wrappers (2, bug #12) ---

void
wrap_signal_silent_store_relaxed(hsa_signal_t sig, hsa_signal_value_t val)
{
    if(should_bypass_inline_intercept())
    {
        s_next_table.hsa_signal_silent_store_relaxed_fn(sig, val);
        return;
    }

    auto s = lookup_queue_state_by_doorbell(sig);
    if(s)
    {
        process_doorbell_impl(s, val, [](hsa_signal_t db, hsa_signal_value_t v) {
            s_next_table.hsa_signal_silent_store_relaxed_fn(db, v);
        });
        return;
    }
    s_next_table.hsa_signal_silent_store_relaxed_fn(sig, val);
}

void
wrap_signal_silent_store_screlease(hsa_signal_t sig, hsa_signal_value_t val)
{
    if(should_bypass_inline_intercept())
    {
        s_next_table.hsa_signal_silent_store_screlease_fn(sig, val);
        return;
    }

    auto s = lookup_queue_state_by_doorbell(sig);
    if(s)
    {
        process_doorbell_impl(s, val, [](hsa_signal_t db, hsa_signal_value_t v) {
            s_next_table.hsa_signal_silent_store_screlease_fn(db, v);
        });
        return;
    }
    s_next_table.hsa_signal_silent_store_screlease_fn(sig, val);
}

}  // namespace

bool
is_intercepting_inline()
{
    return s_intercept_installed.load(std::memory_order_acquire);
}

void
shutdown_intercept()
{
    s_intercept_installed.store(false, std::memory_order_release);

    get_queue_registry().wlock([](auto& map) { map.clear(); });
    get_doorbell_map().wlock([](auto& map) { map.clear(); });
}

void
install_intercept(CoreApiTable& core_table)
{
    // Save current table entries as our next-in-chain (tracing functors when called
    // after update_table, or raw HSA functions otherwise)
    s_next_table = core_table;

    core_table.hsa_queue_add_write_index_relaxed_fn     = wrap_add_write_index_relaxed;
    core_table.hsa_queue_add_write_index_scacq_screl_fn = wrap_add_write_index_scacq_screl;
    core_table.hsa_queue_add_write_index_scacquire_fn   = wrap_add_write_index_scacquire;
    core_table.hsa_queue_add_write_index_screlease_fn   = wrap_add_write_index_screlease;

    core_table.hsa_queue_store_write_index_relaxed_fn   = wrap_store_write_index_relaxed;
    core_table.hsa_queue_store_write_index_screlease_fn = wrap_store_write_index_screlease;

    core_table.hsa_queue_cas_write_index_relaxed_fn     = wrap_cas_write_index_relaxed;
    core_table.hsa_queue_cas_write_index_scacq_screl_fn = wrap_cas_write_index_scacq_screl;
    core_table.hsa_queue_cas_write_index_scacquire_fn   = wrap_cas_write_index_scacquire;
    core_table.hsa_queue_cas_write_index_screlease_fn   = wrap_cas_write_index_screlease;

    core_table.hsa_queue_load_write_index_relaxed_fn   = wrap_load_write_index_relaxed;
    core_table.hsa_queue_load_write_index_scacquire_fn = wrap_load_write_index_scacquire;

    core_table.hsa_signal_store_relaxed_fn   = wrap_signal_store_relaxed;
    core_table.hsa_signal_store_screlease_fn = wrap_signal_store_screlease;

    core_table.hsa_signal_silent_store_relaxed_fn   = wrap_signal_silent_store_relaxed;
    core_table.hsa_signal_silent_store_screlease_fn = wrap_signal_silent_store_screlease;

    s_intercept_installed.store(true, std::memory_order_release);
}

}  // namespace queue_intercept
}  // namespace hsa
}  // namespace rocprofiler
