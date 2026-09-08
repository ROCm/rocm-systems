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
#include "lib/rocprofiler-sdk/kfd/env_parse.hpp"

#include <algorithm>
#include <atomic>
#include <cstddef>
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

// Sole bound on hub growth for entries whose window never closes (gc_closed_windows
// only reaps closed windows; F3's start_max_age_ns ages pending_starts, not the hub
// -- the two are unrelated). Env ROCPROFILER_KFD_DISPATCH_LOG_MAX_PENDING_PER_GPU,
// default 2000000, accepted range [1, 4194304], read once (F3 pattern). Reject-and-
// default via D8's parse contract; the upper limit keeps cap far below SIZE_MAX so
// the register_batch comparisons cannot be defeated by an absurd value.
//
// Sizing (empirically validated, not the original ~100 B "few MB" estimate): a
// graph-launch burst workload (hipGraphLaunch, e.g. 1024-node graph x 1000 iters)
// legitimately keeps hundreds of thousands of dispatches in flight relative to the
// reader/processor retirement rate -- an order of magnitude above the old 65536
// default, which refused 16-41% of dispatches and forced the slower signal path,
// erasing most of the signal-less speedup. At ~100 B/entry the 2M default costs
// ~190-200 MB per GPU at worst. Why not a "medium" cap: collect_cap_victims_locked
// is an O(live-entries-for-that-GPU) scan+sort under the single hub mutex, so a cap
// a realistic workload still exceeds is WORSE than a small cap (cheap scans) or a
// large one (scans essentially never fire) -- a mid cap that trips constantly cost
// +408% in the benchmark. Residual (accepted, not re-engineered here): a workload
// whose genuine peak concurrency exceeds 2M, or a real EOP-loss leak growing
// unboundedly toward the cap, still pays that eviction-scan cost; a proper fix needs
// a per-GPU secondary index for eligible-victim lookup, out of scope for this fix.
inline size_t
hub_max_pending_per_gpu()
{
    static const size_t _v = []() -> size_t {
        constexpr long _dflt = 2'000'000;
        auto           _parsed =
            env_long_in_range("ROCPROFILER_KFD_DISPATCH_LOG_MAX_PENDING_PER_GPU", 1, 4'194'304);
        return static_cast<size_t>(_parsed.value_or(_dflt));
    }();
    return _v;
}

// Exact per-GPU cap shortfall: how many entries must be evicted for `batch` new
// entries to fit under `cap` given `live` already present, or 0 if none. All
// three are size_t; the checked branches never form `live + batch` (overflows) or
// subtract into a wrap on a healthy GPU (the underflow the review flagged). Pure
// and header-exposed so the D9 arithmetic is a deterministic unit seam.
inline size_t
hub_cap_need(size_t live, size_t batch, size_t cap)
{
    if(live > cap) return (live - cap) + batch;          // live-cap guarded by live>cap
    if(batch > cap - live) return batch - (cap - live);  // cap-live safe: live <= cap
    return 0;                                            // fits: no shortfall
}

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
    //
    // Cap-evicted payloads leave via evicted_out for the caller to release AFTER
    // it drops m_mu (the hub's standing no-destroy-under-lock contract). On false
    // the map is UNMUTATED across every GPU and evicted_out is empty.
    bool register_batch(std::vector<registration>&& batch, std::vector<leaked>& evicted_out)
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

        // Per-GPU cap, evaluated for the WHOLE batch in this one critical section
        // so two batches cannot each observe headroom only one has. Eligibility uses
        // the same clock domain (steady_now_ns) as gc_deadline_ns / gc_closed_windows.
        const size_t   cap          = hub_max_pending_per_gpu();
        const uint64_t now_ns       = steady_now_ns();
        auto           batch_by_gpu = std::unordered_map<uint32_t, size_t>{};
        for(const auto& reg : batch)
            ++batch_by_gpu[reg.key.gpu_id];

        // Read-only victim selection across ALL GPUs first; evict only once every
        // GPU's shortfall is satisfied, so a shortfall on a later GPU leaves the
        // map unmutated. Checked comparisons only -- never live+batch-cap, which
        // overflows the add and underflows a healthy GPU into a huge `need`.
        auto victims = std::vector<typename pending_entry_map::iterator>{};
        for(const auto& [gpu, bcount] : batch_by_gpu)
        {
            const size_t live = pending_count_for_gpu_locked(gpu);
            const size_t need = hub_cap_need(live, bcount, cap);
            if(need == 0) continue;  // fits: no shortfall for this GPU

            if(!collect_cap_victims_locked(gpu, need, now_ns, victims))
                return false;  // insufficient eligible victims -> refuse, map unmutated
        }

        for(auto it : victims)
        {
            evicted_out.emplace_back(leak_locked(it));
            erase_entry_locked(it);
        }

        for(auto& reg : batch)
        {
            auto e           = entry{};
            e.correlation_id = reg.correlation_id;
            e.window         = std::move(reg.window);
            e.payload        = std::move(reg.payload);
            e.seq            = m_next_seq++;
            m_entries.emplace(reg.key, std::move(e));
            ++m_pending_by_gpu[reg.key.gpu_id];
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
    // sole candidate on a first_owner, non-superseded window whose t_open the EOP
    // postdates. Otherwise drop. EVERY acceptance test runs before the entry is
    // taken, so a rejected record never consumes or retires anything.
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
            erase_entry_locked(it);
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
        // Terminal-time sanity, applied BEFORE take(): a kernel registered in this
        // window cannot have ended at or before the window opened, so such an EOP
        // belongs to an earlier dispatch on this raw key whose START was lost.
        // Taking it would consume and retire a NEWER dispatch's entry against a
        // foreign record -- the finalizer's staleness bound runs too late to undo
        // that. Rejected here the entry stays PENDING for its own EOP. Same clock
        // domain as the containment test above: raw agent GPU ticks, so no
        // conversion and no host clock is involved. There is deliberately no upper
        // bound: t_close bounds STARTs, not ends -- a kernel legitimately outlives
        // its window -- and the hub holds no tick-domain `now` to bound the future
        // with, so an implausibly-future EOP is still taken here and rejected by
        // the finalizer's kMaxFutureNs test (a no-timing record, not a misattributed
        // one).
        if(end_ticks <= w->t_open) return std::nullopt;
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
            if(it->first.gpu_id == gpu_id && it->first.doorbell_slot == doorbell_slot)
            {
                out.emplace_back(leak_locked(it));
                it = erase_entry_locked(it);
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
                it = erase_entry_locked(it);
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

    // Next insertion seq to be assigned. D9 cap eviction orders victims by it. Under m_mu.
    uint64_t current_seq() const
    {
        if(m_abandoned.load(std::memory_order_acquire)) return 0;
        auto lk = std::lock_guard<std::mutex>{m_mu};
        return m_next_seq;
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
        // Monotonic insertion order, assigned under m_mu. Ages the cap's eviction
        // selection (oldest-first).
        uint64_t seq = 0;
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
        if(m_quarantined.count({key.gpu_id, key.doorbell_slot}) != 0) return false;
        auto [first, last] = m_entries.equal_range(key);
        for(auto it = first; it != last; ++it)
        {
            const auto& w = it->second.window;
            if(w && w->t_close.load(std::memory_order_acquire) == kWindowOpen) return false;
        }
        return true;
    }

    // Caller holds m_mu. Live PENDING entries on one GPU (0 if none).
    size_t pending_count_for_gpu_locked(uint32_t gpu_id) const
    {
        auto it = m_pending_by_gpu.find(gpu_id);
        return it == m_pending_by_gpu.end() ? 0 : it->second;
    }

    // Caller holds m_mu. Collect up to `need` oldest-by-seq ELIGIBLE victims on one
    // GPU into `out` (appended). Eligible == gc_closed_windows()'s own predicate:
    // window closed AND now_ns >= gc_deadline_ns -- never open-window, never in-grace,
    // so no admission guard is lost and no still-running kernel is ledgered. Returns
    // false (collecting nothing usable for the caller to act on) if fewer than `need`
    // eligible entries exist; the caller then refuses and leaves the map unmutated.
    bool collect_cap_victims_locked(uint32_t                                           gpu_id,
                                    size_t                                             need,
                                    uint64_t                                           now_ns,
                                    std::vector<typename pending_entry_map::iterator>& out)
    {
        auto eligible = std::vector<typename pending_entry_map::iterator>{};
        for(auto it = m_entries.begin(); it != m_entries.end(); ++it)
        {
            if(it->first.gpu_id != gpu_id) continue;
            const auto& w = it->second.window;
            if(w && w->t_close.load(std::memory_order_acquire) != kWindowOpen &&
               now_ns >= w->gc_deadline_ns)
                eligible.push_back(it);
        }
        if(eligible.size() < need) return false;
        std::sort(eligible.begin(), eligible.end(), [](const auto& a, const auto& b) {
            return a->second.seq < b->second.seq;  // oldest first
        });
        out.insert(
            out.end(), eligible.begin(), eligible.begin() + static_cast<std::ptrdiff_t>(need));
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
        m_pending_by_gpu.clear();
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
    // Next insertion seq, and live PENDING count per gpu_id (the cap is per-GPU).
    // Incremented at insert, decremented on every erase path via erase_entry_locked.
    uint64_t                             m_next_seq       = 0;
    std::unordered_map<uint32_t, size_t> m_pending_by_gpu = {};

    // Caller holds m_mu. Erase an entry and keep m_pending_by_gpu in step; the
    // single removal point every path funnels through so the per-GPU count cannot
    // drift. Returns the iterator following the erased element.
    typename pending_entry_map::iterator erase_entry_locked(typename pending_entry_map::iterator it)
    {
        auto _g = m_pending_by_gpu.find(it->first.gpu_id);
        if(_g != m_pending_by_gpu.end() && _g->second > 0)
        {
            if(--_g->second == 0) m_pending_by_gpu.erase(_g);
        }
        return m_entries.erase(it);
    }
};
}  // namespace kfd
}  // namespace rocprofiler
