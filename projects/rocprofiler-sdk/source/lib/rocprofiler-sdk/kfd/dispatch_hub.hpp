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

#include <atomic>
#include <cstdint>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

// DispatchHub: the pending-completion registry for signal-less kernel dispatch
// (design plan "Rendezvous invariants (hub)").
//
// An inline batch registers one entry per dispatch BEFORE its packets publish.
// The KFD reader later proves completion from a firmware EOP record and takes
// ownership of the entry's payload; a loss event (ring overrun, reader death,
// slot quarantine, queue close, teardown) instead leaks it. Those two outcomes
// compete for a single winner under one lock:
//
//     ABSENT --register_batch--> PENDING(start_ticks: none|present)
//     PENDING --prove_eop (loss-free drain)--> EOP_PROVEN   (payload handed out)
//     PENDING --leak/poison/quarantine/close/teardown--> LEAKED (payload handed
//                                                                out, NOT retired)
//
// EOP_PROVEN, RESULT_READY and COMPLETED_NO_TIMING are all completion outcomes:
// once an entry leaves PENDING through prove_eop() it can never become LEAKED,
// which is why proven entries leave the map entirely -- ownership moves to the
// caller (task group / retry owner) and the hub cannot hand it out twice. The
// worker's later choice between RESULT_READY and COMPLETED_NO_TIMING is a
// property of the payload, not of the hub.
//
// THREADING: exactly one mutex, and the hub NEVER invokes caller code while
// holding it -- every operation returns owned values and the caller acts on them
// afterwards (invariant 11). PayloadT is only default-constructed, moved, and
// destroyed; a payload destructor therefore never runs under the hub lock except
// for the entry being overwritten, which cannot happen (no-overwrite, invariant 3).
//
// PayloadT is a template parameter so the hub is unit-testable with a fake
// payload and carries no dependency on the HSA queue session types (test seam S2).

namespace rocprofiler
{
namespace kfd
{
enum class entry_state : uint8_t
{
    pending = 0,
    eop_proven,           // completion proven, payload owned by the caller
    result_ready,         // worker: converted + sane -> emit KFD record, retire
    completed_no_timing,  // worker: no start_ticks or convert/sanity failed -> retire
    leaked,               // loss: no record, corr-id deliberately NOT retired
};

// Session lifecycle. No terminal mode returns to `running` without recreating
// the stream.
enum class session_mode : uint8_t
{
    running = 0,
    // Retained for the tombstone cap and for reader death -- a ring overrun no
    // longer poisons the session. An overrun's data is already lost; taking the
    // whole feature down for the rest of the process turned a bounded loss into a
    // total one, so the reader now reports it and keeps collecting.
    loss_poisoned,
    reader_dead,
    stopping,
    child_stale,  // post-fork child
};

// Counted separately for the loud warning: a batch shares one correlation id
// with one reference per dispatch (U15).
struct loss_stats
{
    uint64_t dispatches      = 0;
    uint64_t correlation_ids = 0;
};

template <typename PayloadT>
class DispatchHub
{
public:
    // What a caller registers for one dispatch. `correlation_id` is a value, never
    // dereferenced by the hub: it exists to count unique ids in a loss report and
    // to populate the loss ledger that correlation_id_finalize() must skip.
    struct registration
    {
        correlation_key key            = {};
        uint64_t        correlation_id = 0;
        uint64_t        queue_token    = 0;
        uint64_t        submit_index   = 0;
        PayloadT        payload        = {};
    };

    // Ownership handed to the reader on a proven completion.
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

    // Tombstones are unbounded in principle, so cap them. Exceeding the cap
    // poisons the session rather than evicting a tombstone: forgetting one would
    // let a recurring low-32 dispatch id reactivate a leaked key (U11), and
    // poisoning merely turns signal-less off for the process.
    static constexpr size_t kMaxTombstones = 8192;

    DispatchHub()  = default;
    ~DispatchHub() = default;

    DispatchHub(const DispatchHub&) = delete;
    DispatchHub& operator=(const DispatchHub&) = delete;

    // --- enqueue side -----------------------------------------------------

    // Whole-batch atomic registration (invariant 1): validates every entry first
    // and inserts either all or none. Rejects when the session is not running, a
    // key is already live or tombstoned, a slot is quarantined, or the batch
    // contains a duplicate key -- no overwrite semantics anywhere (invariant 3).
    // The caller must fall back to the signal path for the WHOLE batch on false.
    bool register_batch(std::vector<registration>&& batch)
    {
        if(m_abandoned.load(std::memory_order_acquire)) return false;

        auto lk = std::lock_guard<std::mutex>{m_mu};
        if(m_mode != session_mode::running) return false;

        for(size_t i = 0; i < batch.size(); ++i)
        {
            if(!key_admissible_locked(batch[i].key)) return false;
            for(size_t j = 0; j < i; ++j)
                if(batch[j].key == batch[i].key) return false;
        }

        for(auto& reg : batch)
        {
            ++m_outstanding[reg.queue_token];
            auto& e          = m_entries[reg.key];
            e.state          = entry_state::pending;
            e.correlation_id = reg.correlation_id;
            e.queue_token    = reg.queue_token;
            e.submit_index   = reg.submit_index;
            e.payload        = std::move(reg.payload);
        }
        return true;
    }

    // Would register_batch() accept these keys right now? Used by the enqueue-side
    // eligibility decision, which must be final BEFORE any packet is modified: a
    // batch that skipped its completion signals cannot be un-skipped once the
    // packets are staged. The inline path holds the queue's gate_lock, and keys
    // carry the queue's own submit index, so no other thread can claim these keys
    // between this check and the registration.
    bool can_register_batch(const std::vector<correlation_key>& keys) const
    {
        if(m_abandoned.load(std::memory_order_acquire)) return false;

        auto lk = std::lock_guard<std::mutex>{m_mu};
        for(size_t i = 0; i < keys.size(); ++i)
        {
            if(!key_admissible_locked(keys[i])) return false;
            for(size_t j = 0; j < i; ++j)
                if(keys[j] == keys[i]) return false;
        }
        return true;
    }

    // --- reader side ------------------------------------------------------

    // A firmware START record. Stored on the live entry and cleared only by a
    // terminal transition -- never aged out, so a legitimately long kernel is not
    // stranded (correctness requirement 2, U8). An unmatched START is dropped.
    bool note_start(const correlation_key& key, uint64_t start_ticks)
    {
        if(m_abandoned.load(std::memory_order_acquire)) return false;

        auto lk = std::lock_guard<std::mutex>{m_mu};
        if(m_mode != session_mode::running) return false;
        auto it = m_entries.find(key);
        if(it == m_entries.end() || it->second.state != entry_state::pending) return false;
        it->second.start_ticks = start_ticks;
        return true;
    }

    // A firmware EOP proves the kernel completed (G3). Takes ownership of the
    // entry exactly once; the loser of a race against a loss transition gets
    // nullopt. `drain_loss_free` is the reader's wptr verdict for the drain that
    // produced this record: an EOP observed under an overrun proves nothing about
    // WHICH dispatch it belongs to, so it completes nothing (U9).
    //
    // A key with no live entry is REJECTED, never cached (U3b): a result with no
    // pending owner is stale or ambiguous and must not be applied to a later
    // same-key dispatch.
    std::optional<proven> prove_eop(const correlation_key& key,
                                    uint64_t               end_ticks,
                                    bool                   drain_loss_free)
    {
        if(m_abandoned.load(std::memory_order_acquire)) return std::nullopt;
        if(!drain_loss_free) return std::nullopt;

        auto lk = std::lock_guard<std::mutex>{m_mu};
        if(m_mode != session_mode::running) return std::nullopt;

        auto it = m_entries.find(key);
        if(it == m_entries.end() || it->second.state != entry_state::pending) return std::nullopt;

        auto out           = proven{};
        out.key            = key;
        out.correlation_id = it->second.correlation_id;
        out.start_ticks    = it->second.start_ticks;
        out.end_ticks      = end_ticks;
        out.payload        = std::move(it->second.payload);
        release_locked(it->second.queue_token);
        m_entries.erase(it);
        ++m_proven;
        return out;
    }

    // --- loss side --------------------------------------------------------

    // Single-winner loss transition for one dispatch.
    std::optional<leaked> leak(const correlation_key& key)
    {
        if(m_abandoned.load(std::memory_order_acquire)) return std::nullopt;

        auto lk = std::lock_guard<std::mutex>{m_mu};
        auto it = m_entries.find(key);
        if(it == m_entries.end()) return std::nullopt;
        auto out = leak_locked(it);
        m_entries.erase(it);
        return out;
    }

    // Ring overrun / permanent loss: leak every still-matchable entry, latch
    // LOSS_POISONED so nothing registers or completes again, and report the counts
    // for the single loud warning (P1 leak-and-shout, U15). Payloads are returned
    // so the caller releases them off-lock; correlation ids are NOT retired.
    std::pair<std::vector<leaked>, loss_stats> poison(session_mode terminal_mode)
    {
        if(m_abandoned.load(std::memory_order_acquire)) return {};

        auto lk = std::lock_guard<std::mutex>{m_mu};
        m_mode  = terminal_mode;
        return leak_all_locked();
    }

    // Queue destroy has begun for this slot. Eligibility must stop RESERVING on it
    // immediately, but a batch that already passed eligibility and skipped its
    // completion signals must still be able to register -- the destroy path fences
    // those in flight before it leaks and quarantines. So this deliberately does
    // NOT make register_batch() fail; only the eligibility query consults it.
    void mark_slot_closing(uint32_t doorbell_slot)
    {
        if(m_abandoned.load(std::memory_order_acquire)) return;
        auto lk = std::lock_guard<std::mutex>{m_mu};
        m_closing.insert(doorbell_slot);
    }

    bool is_closing(uint32_t doorbell_slot) const
    {
        if(m_abandoned.load(std::memory_order_acquire)) return false;
        auto lk = std::lock_guard<std::mutex>{m_mu};
        return m_closing.count(doorbell_slot) != 0;
    }

    // Slot collision (requirement 3) or queue close / generation bump
    // (requirement 4): leak everything still pending on the slot and make it
    // permanently unusable, so no later owner or reused generation can register
    // against it again (invariant 9, U12/U13).
    std::vector<leaked> quarantine_slot(uint32_t doorbell_slot)
    {
        if(m_abandoned.load(std::memory_order_acquire)) return {};

        auto lk = std::lock_guard<std::mutex>{m_mu};
        m_quarantined.insert(doorbell_slot);
        m_closing.insert(doorbell_slot);
        auto out = std::vector<leaked>{};
        for(auto it = m_entries.begin(); it != m_entries.end();)
        {
            if(it->first.doorbell_off == doorbell_slot)
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

    // Queue destroy: leak whatever that queue still has outstanding. Paired with
    // quarantine_slot() by the caller, which owns the doorbell generation bump.
    std::vector<leaked> close_queue(uint64_t queue_token)
    {
        if(m_abandoned.load(std::memory_order_acquire)) return {};

        auto lk  = std::lock_guard<std::mutex>{m_mu};
        auto out = std::vector<leaked>{};
        for(auto it = m_entries.begin(); it != m_entries.end();)
        {
            if(it->second.queue_token == queue_token)
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

    // Teardown step 5: everything still PENDING becomes LEAKED before the task
    // group is joined and correlation ids are finalized (requirement 7, U18).
    std::pair<std::vector<leaked>, loss_stats> drain_for_teardown()
    {
        if(m_abandoned.load(std::memory_order_acquire)) return {};

        auto lk = std::lock_guard<std::mutex>{m_mu};
        if(m_mode == session_mode::running) m_mode = session_mode::stopping;
        return leak_all_locked();
    }

    // --- queries ----------------------------------------------------------

    // correlation_id_finalize() must NOT force-retire a leaked id (requirement 6,
    // U14): its kernel may still be running and its references were deliberately
    // not dropped.
    bool is_ledgered(uint64_t correlation_id) const
    {
        if(m_abandoned.load(std::memory_order_acquire)) return false;
        auto lk = std::lock_guard<std::mutex>{m_mu};
        return m_ledger.count(correlation_id) != 0;
    }

    bool is_quarantined(uint32_t doorbell_slot) const
    {
        if(m_abandoned.load(std::memory_order_acquire)) return false;
        auto lk = std::lock_guard<std::mutex>{m_mu};
        return m_quarantined.count(doorbell_slot) != 0;
    }

    // ++ on register, -- on any terminal transition (invariant 7).
    size_t outstanding(uint64_t queue_token) const
    {
        if(m_abandoned.load(std::memory_order_acquire)) return 0;
        auto lk = std::lock_guard<std::mutex>{m_mu};
        auto it = m_outstanding.find(queue_token);
        return (it == m_outstanding.end()) ? 0 : it->second;
    }

    size_t live_entries() const
    {
        if(m_abandoned.load(std::memory_order_acquire)) return 0;
        auto lk = std::lock_guard<std::mutex>{m_mu};
        return m_entries.size();
    }

    size_t tombstones() const
    {
        if(m_abandoned.load(std::memory_order_acquire)) return 0;
        auto lk = std::lock_guard<std::mutex>{m_mu};
        return m_tombstones.size();
    }

    uint64_t proven_count() const { return m_proven.load(std::memory_order_relaxed); }

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

    // pthread_atfork child handler (requirement 8). Async-signal-safe: ONE atomic
    // store, no mutex, no allocation, no map access, no logging. EVERY operation
    // -- mutating and query alike -- tests this BEFORE it would take m_mu, so a
    // child never touches an inherited mutex that a vanished thread may have held
    // locked at the moment of the fork.
    //
    // The abandoned state is deliberately one-way: nothing un-abandons a child.
    void abandon_in_child() { m_abandoned.store(true, std::memory_order_release); }

    bool abandoned() const { return m_abandoned.load(std::memory_order_acquire); }

private:
    struct entry
    {
        entry_state             state          = entry_state::pending;
        uint64_t                correlation_id = 0;
        uint64_t                queue_token    = 0;
        uint64_t                submit_index   = 0;
        std::optional<uint64_t> start_ticks    = {};
        PayloadT                payload        = {};
    };

    using map_t = std::unordered_map<correlation_key, entry, correlation_key_hash>;

    // Caller holds m_mu. A key may be registered only into a running session, only
    // once, never onto a tombstone (U11), and never onto a quarantined slot.
    bool key_admissible_locked(const correlation_key& key) const
    {
        return m_mode == session_mode::running && m_entries.count(key) == 0 &&
               m_tombstones.count(key) == 0 && m_quarantined.count(key.doorbell_off) == 0;
    }

    // Caller holds m_mu. Does NOT erase; callers that iterate erase themselves.
    leaked leak_locked(typename map_t::iterator it)
    {
        auto out           = leaked{};
        out.key            = it->first;
        out.correlation_id = it->second.correlation_id;
        out.payload        = std::move(it->second.payload);
        it->second.state   = entry_state::leaked;
        release_locked(it->second.queue_token);
        tombstone_locked(it->first);
        m_ledger.insert(out.correlation_id);
        return out;
    }

    std::pair<std::vector<leaked>, loss_stats> leak_all_locked()
    {
        auto out = std::vector<leaked>{};
        auto ids = std::unordered_set<uint64_t>{};
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

    void tombstone_locked(const correlation_key& key)
    {
        m_tombstones.insert(key);
        // Bounded: poison rather than forget a tombstone (see kMaxTombstones).
        if(m_tombstones.size() > kMaxTombstones && m_mode == session_mode::running)
            m_mode = session_mode::loss_poisoned;
    }

    void release_locked(uint64_t queue_token)
    {
        auto it = m_outstanding.find(queue_token);
        if(it == m_outstanding.end()) return;
        if(--it->second == 0) m_outstanding.erase(it);
    }

    mutable std::mutex m_mu = {};
    // Checked before m_mu on every operation, so a forked child never touches the
    // inherited mutex or map.
    std::atomic<bool>     m_abandoned = {false};
    std::atomic<uint64_t> m_proven    = {0};

    session_mode                                              m_mode        = session_mode::running;
    map_t                                                     m_entries     = {};
    std::unordered_set<correlation_key, correlation_key_hash> m_tombstones  = {};
    std::unordered_set<uint64_t>                              m_ledger      = {};
    std::unordered_set<uint32_t>                              m_quarantined = {};
    std::unordered_set<uint32_t>                              m_closing     = {};
    std::unordered_map<uint64_t, size_t>                      m_outstanding = {};
};
}  // namespace kfd
}  // namespace rocprofiler
