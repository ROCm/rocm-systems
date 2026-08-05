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
//     PENDING --prove_eop (loss-free drain)--> EOP_PROVEN   (payload handed out)
//     PENDING --leak/poison/quarantine/close/teardown--> LEAKED (payload handed
//                                                                out, NOT retired)
//
// Once an entry leaves PENDING through prove_eop() it can never become LEAKED,
// which is why proven entries leave the map entirely -- ownership moves to the
// caller and the hub cannot hand it out twice.
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
// Session lifecycle. Gates eligibility only; registration and completion are
// deliberately unconditional.
enum class session_mode : uint8_t
{
    running = 0,
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
    // `correlation_id` is a value, never dereferenced by the hub: it counts unique
    // ids in a loss report and populates the ledger finalize must skip.
    struct registration
    {
        correlation_key key            = {};
        uint64_t        correlation_id = 0;
        PayloadT        payload        = {};
    };

    // Ownership handed to the reader on a proven completion.
    struct proven
    {
        correlation_key         key         = {};
        std::optional<uint64_t> start_ticks = {};  // absent -> COMPLETED_NO_TIMING
        uint64_t                end_ticks   = 0;
        PayloadT                payload     = {};
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

    // Whole-batch atomic: validates every entry first and inserts all or none.
    // No overwrite semantics anywhere. The caller must fall back to the signal
    // path for the WHOLE batch on false.
    // NO session-mode gate: eligibility already committed this batch to the
    // signal-less path, so refusing it here would leave dispatches that have
    // skipped their completion signals with nothing to complete them. Teardown
    // step 5 leaks anything that registers late, which retires nothing.
    bool register_batch(std::vector<registration>&& batch)
    {
        if(m_abandoned.load(std::memory_order_acquire)) return false;

        auto lk = std::lock_guard<std::mutex>{m_mu};

        for(size_t i = 0; i < batch.size(); ++i)
        {
            if(!key_admissible_locked(batch[i].key)) return false;
            for(size_t j = 0; j < i; ++j)
                if(batch[j].key == batch[i].key) return false;
        }

        for(auto& reg : batch)
        {
            auto& e          = m_entries[reg.key];
            e.correlation_id = reg.correlation_id;
            e.payload        = std::move(reg.payload);
        }
        return true;
    }

    // Would register_batch() accept these keys right now? Advisory only: the
    // authoritative check is register_batch() itself, under the same lock.
    bool can_register_batch(const std::vector<correlation_key>& keys) const
    {
        if(m_abandoned.load(std::memory_order_acquire)) return false;

        auto lk = std::lock_guard<std::mutex>{m_mu};
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

    // Cleared only by the entry leaving PENDING. No session-mode gate, for the
    // same reason as prove_eop: a START seen during the teardown drain is what
    // gives that drain's EOP an interval instead of a no-timing completion.
    bool note_start(const correlation_key& key, uint64_t start_ticks)
    {
        if(m_abandoned.load(std::memory_order_acquire)) return false;

        auto lk = std::lock_guard<std::mutex>{m_mu};
        auto it = m_entries.find(key);
        if(it == m_entries.end()) return false;
        it->second.start_ticks = start_ticks;
        return true;
    }

    // NO session-mode gate: the final teardown drain runs after the mode is
    // already stopping, and an EOP that arrives then still proves its kernel
    // finished, so completing it is strictly better than leaking it.
    //
    // A firmware EOP proves the kernel completed. Takes ownership exactly once;
    // the loser of a race against a loss transition gets nullopt.
    // `drain_loss_free` is the reader's wptr verdict: an EOP observed under an
    // overrun proves nothing about WHICH dispatch it belongs to, so it completes
    // nothing. A key with no live entry is REJECTED, never cached -- a result
    // with no pending owner must not be applied to a later same-key dispatch.
    std::optional<proven> prove_eop(const correlation_key& key,
                                    uint64_t               end_ticks,
                                    bool                   drain_loss_free)
    {
        if(m_abandoned.load(std::memory_order_acquire)) return std::nullopt;
        if(!drain_loss_free) return std::nullopt;

        auto lk = std::lock_guard<std::mutex>{m_mu};

        auto it = m_entries.find(key);
        if(it == m_entries.end()) return std::nullopt;

        auto out        = proven{};
        out.key         = key;
        out.start_ticks = it->second.start_ticks;
        out.end_ticks   = end_ticks;
        out.payload     = std::move(it->second.payload);
        m_entries.erase(it);
        return out;
    }

    // --- loss side --------------------------------------------------------

    // Eligibility must stop RESERVING the slot before the queue's gate_lock is
    // fenced, or a batch can slip in behind the fence.
    void mark_slot_closing(uint32_t gpu_id, uint32_t doorbell_slot)
    {
        if(m_abandoned.load(std::memory_order_acquire)) return;
        auto lk = std::lock_guard<std::mutex>{m_mu};
        m_closing.insert({gpu_id, doorbell_slot});
    }

    bool is_closing(uint32_t gpu_id, uint32_t doorbell_slot) const
    {
        if(m_abandoned.load(std::memory_order_acquire)) return false;
        auto lk = std::lock_guard<std::mutex>{m_mu};
        return m_closing.count({gpu_id, doorbell_slot}) != 0;
    }

    // Slot collision or queue close / generation bump.
    std::vector<leaked> quarantine_slot(uint32_t gpu_id, uint32_t doorbell_slot)
    {
        if(m_abandoned.load(std::memory_order_acquire)) return {};

        auto lk = std::lock_guard<std::mutex>{m_mu};
        m_quarantined.insert({gpu_id, doorbell_slot});
        m_closing.insert({gpu_id, doorbell_slot});
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

    // correlation_id_finalize() must NOT force-retire a leaked id: its kernel may
    // still be running.
    bool is_ledgered(uint64_t correlation_id) const
    {
        if(m_abandoned.load(std::memory_order_acquire)) return false;
        auto lk = std::lock_guard<std::mutex>{m_mu};
        return m_ledger.count(correlation_id) != 0;
    }

    // Exactly what the close drain waits on.
    size_t pending_for_slot(uint32_t gpu_id, uint32_t doorbell_slot) const
    {
        if(m_abandoned.load(std::memory_order_acquire)) return 0;
        auto   lk      = std::lock_guard<std::mutex>{m_mu};
        size_t pending = 0;
        for(const auto& itr : m_entries)
        {
            if(itr.first.gpu_id == gpu_id && itr.first.doorbell_off == doorbell_slot) ++pending;
        }
        return pending;
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
        uint64_t                correlation_id = 0;
        std::optional<uint64_t> start_ticks    = {};
        PayloadT                payload        = {};
    };

    using map_t = std::unordered_map<correlation_key, entry, correlation_key_hash>;

    // Caller holds m_mu. A key may be registered only once and never onto a
    // quarantined slot -- which, being permanent, is what keeps a leaked key from
    // ever matching again. Session mode is deliberately NOT part of this: it
    // gates eligibility, not registration.
    bool key_admissible_locked(const correlation_key& key) const
    {
        return m_entries.count(key) == 0 &&
               m_quarantined.count({key.gpu_id, key.doorbell_off}) == 0;
    }

    // Caller holds m_mu. Does NOT erase; callers that iterate erase themselves.
    leaked leak_locked(typename map_t::iterator it)
    {
        auto out           = leaked{};
        out.key            = it->first;
        out.correlation_id = it->second.correlation_id;
        out.payload        = std::move(it->second.payload);
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

    mutable std::mutex m_mu = {};
    // Checked before m_mu on every operation, so a forked child never touches the
    // inherited mutex or map.
    std::atomic<bool> m_abandoned = {false};

    session_mode                 m_mode    = session_mode::running;
    map_t                        m_entries = {};
    std::unordered_set<uint64_t> m_ledger  = {};
    // Keyed by (gpu_id, slot): slot numbers repeat across GPUs, so a single set
    // would let one GPU's queue destroy quarantine another GPU's live slot.
    std::set<std::pair<uint32_t, uint32_t>> m_quarantined = {};
    std::set<std::pair<uint32_t, uint32_t>> m_closing     = {};
};
}  // namespace kfd
}  // namespace rocprofiler
