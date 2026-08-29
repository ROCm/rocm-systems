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

#pragma once

#include "lib/rocprofiler-sdk/kfd/correlation_types.hpp"
#include "lib/rocprofiler-sdk/kfd/doorbell_map.hpp"

#include <atomic>
#include <cstdint>
#include <iterator>
#include <mutex>
#include <optional>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

// DispatchHub: the pending-completion registry for signal-less kernel dispatch.
//
// An inline batch registers one entry per dispatch BEFORE its packets publish.
// The reader later proves completion from a firmware EOP and takes ownership of
// the payload; a loss event (overrun, reader death, quarantine, close, teardown)
// instead leaks it. Those two outcomes compete for a single winner under one
// lock:
//
//     ABSENT --register_batch--> PENDING(start_ticks: none|present)
//     PENDING --record_kernel_end (loss-free drain)--> EOP_PROVEN (payload handed out)
//     PENDING --leak/poison/quarantine/close/teardown--> LEAKED (payload handed
//                                                                out, NOT retired)
//
// Once an entry leaves PENDING through record_kernel_end() it can never become
// LEAKED, which is why proven entries leave the map entirely -- ownership moves
// to the caller and the hub cannot hand it out twice.
//
// THREADING: exactly one mutex, and the hub NEVER invokes caller code while
// holding it -- every operation returns owned values the caller acts on
// afterwards. PayloadT is only default-constructed, moved and destroyed, so a
// payload destructor never runs under the hub lock.
//
// PayloadT is a template parameter so the hub is unit-testable with a fake
// payload and carries no dependency on the HSA queue session types.

namespace rocprofiler
{
namespace kfd
{
// Gates eligibility only; registration and completion are deliberately
// unconditional.
enum class session_mode
{
    running = 0,
    stopping,
    child_stale,  // post-fork child
};

// dispatches vs correlation_ids are counted separately for the loud warning,
// because a batch shares one correlation id with one reference per dispatch.
struct loss_stats
{
    uint64_t dispatches      = 0;
    uint64_t correlation_ids = 0;
};

template <typename PayloadT>
class DispatchHub
{
public:
    // `correlation_id` is a value, never dereferenced by the hub: it counts unique
    // ids in a loss report and populates the ledger finalize must skip. `window`
    // is the owner_window this dispatch was registered against; the resolution
    // rule selects an entry by containing the record's START tick in it.
    struct registration
    {
        correlation_key key            = {};
        uint64_t        correlation_id = 0;
        window_ptr      window         = {};
        PayloadT        payload        = {};
    };

    // Ownership handed to the reader on a proven completion. correlation_id is
    // carried so the abandon path can ledger a proven that was already erased from
    // the map but dropped instead of emitted (see ledger_abandoned).
    struct proven
    {
        correlation_key         key            = {};
        uint64_t                correlation_id = 0;
        std::optional<uint64_t> start_ticks    = {};  // absent -> COMPLETED_NO_TIMING
        uint64_t                end_ticks      = 0;
        PayloadT                payload        = {};
    };

    // Ownership handed back on a loss. The payload is released; the correlation id
    // is deliberately NOT retired (P1).
    struct leaked
    {
        correlation_key key            = {};
        uint64_t        correlation_id = 0;
        PayloadT        payload        = {};
    };

    DispatchHub()  = default;
    ~DispatchHub() = default;

    DispatchHub(const DispatchHub&) = delete;
    DispatchHub& operator=(const DispatchHub&) = delete;

    // --- enqueue side -----------------------------------------------------

    // Validates and inserts a whole batch, all or none; caller falls back to the
    // signal path on false. Not mode-gated: eligibility already committed this
    // batch, so refusing here would leave dispatches that skipped their signals
    // with nothing to complete them.
    bool register_batch(std::vector<registration>&& batch)
    {
        if(m_abandoned.load(std::memory_order_acquire)) return false;

        auto lk = std::lock_guard<std::mutex>{m_mu};
        // reject on the process-wide disable latch, checked under
        // m_mu (NOT m_mode, which is teardown-only and must keep admitting
        // in-flight batches). The mutex chain makes this race-free against
        // signal_less_disable_permanently()'s drain: a batch that raced past it is
        // swept by the drain; one that acquires m_mu after it observes the latch.
        if(signal_less_disabled()) return false;

        for(size_t i = 0; i < batch.size(); ++i)
        {
            if(!key_admissible_locked(batch[i].key)) return false;
            for(size_t j = 0; j < i; ++j)
                if(batch[j].key == batch[i].key) return false;
        }

        for(auto& reg : batch)
        {
            auto e           = entry{};
            e.correlation_id = reg.correlation_id;
            e.window         = std::move(reg.window);
            e.payload        = std::move(reg.payload);
            m_entries.emplace(reg.key, std::move(e));
        }
        return true;
    }

    // Would register_batch() accept these keys right now? Advisory only: the
    // authoritative check is register_batch() itself, under the same lock.
    bool can_register_batch(const std::vector<correlation_key>& keys) const
    {
        if(m_abandoned.load(std::memory_order_acquire)) return false;

        auto lk = std::lock_guard<std::mutex>{m_mu};
        if(signal_less_disabled()) return false;
        if(m_mode != session_mode::running) return false;
        for(size_t i = 0; i < keys.size(); ++i)
        {
            if(!key_admissible_locked(keys[i])) return false;
            for(size_t j = 0; j < i; ++j)
                if(keys[j] == keys[i]) return false;
        }
        return true;
    }

    // --- reader side ------------------------------------------------------

    // Time-as-generation resolution. The drain already paired START<->EOP
    // and hands one record here, so there is one hub call and one lock per record.
    // With a known START tick, select the entry whose window STRICTLY contains it
    // (containment applied even for a sole candidate -- a stale pending entry could
    // be the sole candidate while the record's true owner is an unregistered later
    // dispatch on a colliding low-32 id). With no START (shape ii), accept only a
    // sole candidate on a first_owner, non-superseded window. Otherwise drop.
    //
    // NO session-mode gate: an EOP arriving during the teardown drain still proves
    // its kernel finished. A key with no live entry is REJECTED, never cached.
    std::optional<proven> record_kernel_end(const correlation_key&         key,
                                            const std::optional<uint64_t>& start_ticks_opt,
                                            uint64_t                       end_ticks)
    {
        if(m_abandoned.load(std::memory_order_acquire)) return std::nullopt;

        auto lk = std::lock_guard<std::mutex>{m_mu};

        auto [first, last] = m_entries.equal_range(key);
        if(first == last) return std::nullopt;

        auto take = [&](typename pending_entry_map::iterator it) {
            auto out           = proven{};
            out.key            = it->first;
            out.correlation_id = it->second.correlation_id;
            out.start_ticks    = start_ticks_opt;
            out.end_ticks      = end_ticks;
            out.payload        = std::move(it->second.payload);
            m_entries.erase(it);
            return out;
        };

        if(start_ticks_opt)
        {
            const uint64_t t = *start_ticks_opt;
            for(auto it = first; it != last; ++it)
            {
                const auto& w = it->second.window;
                if(w && w->t_open < t && t < w->t_close.load(std::memory_order_acquire))
                {
                    if(end_ticks < t) break;  // free sanity check -> drop
                    return take(it);
                }
            }
            return std::nullopt;  // no containing window
        }

        // Unknown START: sole candidate on a first_owner, non-superseded window.
        if(std::next(first) != last) return std::nullopt;
        const auto& w = first->second.window;
        if(!w || !w->first_owner) return std::nullopt;
        if(w->superseded.load(std::memory_order_acquire)) return std::nullopt;
        return take(first);
    }

    // --- loss side --------------------------------------------------------

    // Structural-ambiguity poison ONLY: two live owners on one
    // slot, a truncated close, or a clock failure. Permanent for the process.
    std::vector<leaked> quarantine_slot(uint32_t gpu_id, uint32_t doorbell_slot)
    {
        if(m_abandoned.load(std::memory_order_acquire)) return {};

        auto lk = std::lock_guard<std::mutex>{m_mu};
        m_quarantined.insert({gpu_id, doorbell_slot});
        auto out = std::vector<leaked>{};
        for(auto it = m_entries.begin(); it != m_entries.end();)
        {
            if(it->first.gpu_id == gpu_id && it->first.doorbell_off == doorbell_slot)
            {
                out.emplace_back(leak_locked(it));
                it = m_entries.erase(it);
            }
            else
            {
                ++it;
            }
        }
        return out;
    }

    // deferred GC: leak+ledger every entry of a window that closed at least
    // close_grace_ns ago. Called on the processor's periodic tick. Returns the
    // payloads for the caller to release OUTSIDE the lock (the hub's standing
    // contract), plus the loss stats.
    std::pair<std::vector<leaked>, loss_stats> gc_closed_windows(uint64_t now_ns)
    {
        if(m_abandoned.load(std::memory_order_acquire)) return {};

        auto lk  = std::lock_guard<std::mutex>{m_mu};
        auto out = std::vector<leaked>{};
        auto ids = correlation_id_set{};
        for(auto it = m_entries.begin(); it != m_entries.end();)
        {
            const auto& w = it->second.window;
            if(w && w->t_close.load(std::memory_order_acquire) != kWindowOpen &&
               now_ns >= w->gc_deadline_ns)
            {
                ids.insert(it->second.correlation_id);
                out.emplace_back(leak_locked(it));
                it = m_entries.erase(it);
            }
            else
            {
                ++it;
            }
        }
        auto stats            = loss_stats{};
        stats.dispatches      = out.size();
        stats.correlation_ids = ids.size();
        return {std::move(out), stats};
    }

    // Teardown step 5: everything still PENDING becomes LEAKED before the task
    // group is joined and correlation ids are finalized.
    std::pair<std::vector<leaked>, loss_stats> drain_for_teardown()
    {
        if(m_abandoned.load(std::memory_order_acquire)) return {};

        auto lk = std::lock_guard<std::mutex>{m_mu};
        if(m_mode == session_mode::running) m_mode = session_mode::stopping;
        return leak_all_locked();
    }

    // --- queries ----------------------------------------------------------

    // correlation_id_finalize() must NOT force-retire a leaked id: its kernel may
    // still be running. Once the ledger saturates this returns true
    // unconditionally -- the conservative "do not force-retire" answer -- at O(1)
    // memory, so a pathological loss stream cannot grow the set forever.
    bool is_ledgered(uint64_t correlation_id) const
    {
        if(m_abandoned.load(std::memory_order_acquire)) return false;
        auto lk = std::lock_guard<std::mutex>{m_mu};
        if(m_ledger_saturated) return true;
        return m_ledger.count(correlation_id) != 0;
    }

    // Ledger a correlation id whose PROVEN entry was already erased from the map by
    // record_kernel_end() but then dropped instead of emitted (abandon-on-timeout).
    // drain_for_teardown() cannot reach it -- it is no longer in m_entries -- so the
    // abandon path ledgers it here, keeping correlation_id_finalize() from treating
    // it as a dangling id. NOT gated on m_mode: an abandon can happen while running.
    void ledger_abandoned(uint64_t correlation_id)
    {
        if(m_abandoned.load(std::memory_order_acquire)) return;
        auto lk = std::lock_guard<std::mutex>{m_mu};
        ledger_locked(correlation_id);
    }

    // Total live PENDING entries. Introspection for the state-bound tests;
    // production reasons per-window, never per-slot, so no slot-scoped count exists.
    size_t pending_count() const
    {
        if(m_abandoned.load(std::memory_order_acquire)) return 0;
        auto lk = std::lock_guard<std::mutex>{m_mu};
        return m_entries.size();
    }

    session_mode mode() const
    {
        if(m_abandoned.load(std::memory_order_acquire)) return session_mode::child_stale;
        auto lk = std::lock_guard<std::mutex>{m_mu};
        return m_mode;
    }

    void set_mode(session_mode m)
    {
        if(m_abandoned.load(std::memory_order_acquire)) return;
        auto lk = std::lock_guard<std::mutex>{m_mu};
        m_mode  = m;
    }

    // --- fork -------------------------------------------------------------

    // pthread_atfork child handler. Async-signal-safe: ONE atomic store, no mutex,
    // allocation, map access or logging. EVERY operation tests this BEFORE it
    // would take m_mu, so a child never touches an inherited mutex a vanished
    // thread may have held locked. One-way: nothing un-abandons a child.
    void abandon_in_child() { m_abandoned.store(true, std::memory_order_release); }

private:
    struct entry
    {
        uint64_t   correlation_id = 0;
        window_ptr window         = {};
        PayloadT   payload        = {};
    };

    // A doorbell slot is only unique per GPU, so every slot-keyed container is
    // keyed by the pair, never the slot alone. A key3 can be held by several
    // windows at once -- that IS the recycled-doorbell collision, now representable
    // instead of refused -- so this is a multimap.
    using slot_key          = std::pair<uint32_t, uint32_t>;  // (gpu_id, doorbell_slot)
    using slot_set          = std::set<slot_key>;
    using pending_entry_map = std::unordered_multimap<correlation_key, entry, correlation_key_hash>;
    using correlation_id_set = std::unordered_set<uint64_t>;

    // Cap on the loss ledger. Past it, is_ledgered() answers true always.
    static constexpr size_t kLedgerCap = 1u << 20;

    // Caller holds m_mu. A key may be registered onto a quarantined slot never,
    // and onto a slot that already has an OPEN-window entry for this key3 never
    // (a 2^32 low-32 wrap on one live queue). ACROSS windows -- a recycled
    // doorbell -- recurrence is exactly the case this design handles, so it is
    // admitted. Session mode gates eligibility, not registration.
    bool key_admissible_locked(const correlation_key& key) const
    {
        if(m_quarantined.count({key.gpu_id, key.doorbell_off}) != 0) return false;
        auto [first, last] = m_entries.equal_range(key);
        for(auto it = first; it != last; ++it)
        {
            const auto& w = it->second.window;
            if(w && w->t_close.load(std::memory_order_acquire) == kWindowOpen) return false;
        }
        return true;
    }

    // Caller holds m_mu. Insert into the bounded ledger; once it saturates,
    // is_ledgered() answers true unconditionally at O(1) memory.
    void ledger_locked(uint64_t correlation_id)
    {
        if(m_ledger_saturated) return;
        m_ledger.insert(correlation_id);
        if(m_ledger.size() >= kLedgerCap) m_ledger_saturated = true;
    }

    // Caller holds m_mu. Does NOT erase; callers that iterate erase themselves.
    leaked leak_locked(typename pending_entry_map::iterator it)
    {
        auto out           = leaked{};
        out.key            = it->first;
        out.correlation_id = it->second.correlation_id;
        out.payload        = std::move(it->second.payload);
        ledger_locked(out.correlation_id);
        return out;
    }

    std::pair<std::vector<leaked>, loss_stats> leak_all_locked()
    {
        auto out = std::vector<leaked>{};
        auto ids = correlation_id_set{};
        for(auto it = m_entries.begin(); it != m_entries.end(); ++it)
        {
            ids.insert(it->second.correlation_id);
            out.emplace_back(leak_locked(it));
        }
        m_entries.clear();
        auto stats            = loss_stats{};
        stats.dispatches      = out.size();
        stats.correlation_ids = ids.size();
        return {std::move(out), stats};
    }

    mutable std::mutex m_mu = {};
    // Checked before m_mu on every operation, so a forked child never touches the
    // inherited mutex or map.
    std::atomic<bool> m_abandoned = {false};

    session_mode       m_mode             = session_mode::running;
    pending_entry_map  m_entries          = {};
    correlation_id_set m_ledger           = {};
    bool               m_ledger_saturated = false;
    // A single GPU's queue destroy must not quarantine another GPU's live slot.
    slot_set m_quarantined = {};
};
}  // namespace kfd
}  // namespace rocprofiler
