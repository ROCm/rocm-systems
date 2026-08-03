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

#include <cstddef>
#include <cstdint>
#include <map>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <utility>

// Reverse doorbell-owner registry (design requirement 3) and the lazy
// HW-profiling bookkeeping.
//
// A firmware record identifies its queue only by a page-relative doorbell slot,
// so selecting a record for a dispatch is only sound when exactly ONE live queue
// owns that slot. This tracks ownership for EVERY live compute queue the SDK
// knows about -- populated at queue creation, so a queue that existed before the
// dispatch-log session was established is already in it, and a queue that has
// never dispatched still counts as an owner.
//
// Both types are free of the HSA and tracing headers so the ownership rules are
// unit-testable with an injected (queue, slot) resolver (test seam S4).
//
// LOCK ORDERING: this registry's mutex is NEVER held while the hub's mutex is
// taken, or vice versa. Every operation returns a verdict and the caller acts on
// it after releasing the lock, so the two locks are never nested in either
// direction and cannot deadlock.

namespace rocprofiler
{
namespace kfd
{
class OwnerRegistry
{
public:
    enum class add_result
    {
        sole_owner,    // this queue is the only live owner of its slot
        collision,     // a second live owner appeared: the caller quarantines the slot
        slot_unknown,  // the queue's doorbell could not be resolved
    };

    OwnerRegistry()  = default;
    ~OwnerRegistry() = default;

    OwnerRegistry(const OwnerRegistry&) = delete;
    OwnerRegistry& operator=(const OwnerRegistry&) = delete;

    // Register a live compute queue. `slot` is nullopt when the queue's doorbell
    // could not be resolved -- such a queue could be the second owner of ANY slot,
    // so it disables signal-less for its whole GPU while it lives.
    //
    // Re-registering an existing token replaces its previous ownership.
    add_result add_queue(uint64_t queue_token, uint32_t gpu_id, std::optional<uint32_t> slot)
    {
        auto lk = std::lock_guard<std::mutex>{m_mu};
        remove_locked(queue_token);

        m_by_queue[queue_token] = queue_entry{gpu_id, slot};
        if(!slot)
        {
            ++m_unresolved[gpu_id];
            return add_result::slot_unknown;
        }

        auto _owners = ++m_owners[{gpu_id, *slot}];
        return (_owners > 1) ? add_result::collision : add_result::sole_owner;
    }

    void remove_queue(uint64_t queue_token)
    {
        auto lk = std::lock_guard<std::mutex>{m_mu};
        remove_locked(queue_token);
    }

    // Exactly one live owner for this slot on this GPU, and every live queue on
    // the GPU has a known slot. A queue whose slot is unknown could be the second
    // owner, so its presence makes every slot on that GPU non-injective.
    //
    // Note this deliberately says nothing about quarantine: a slot that once
    // collided stays quarantined in the hub for the rest of the process even after
    // one of the colliding queues dies and ownership looks injective again.
    bool is_injective(uint32_t gpu_id, uint32_t slot) const
    {
        auto lk = std::lock_guard<std::mutex>{m_mu};
        if(unresolved_locked(gpu_id) != 0) return false;
        return owners_locked(gpu_id, slot) == 1;
    }

    size_t owners_of(uint32_t gpu_id, uint32_t slot) const
    {
        auto lk = std::lock_guard<std::mutex>{m_mu};
        return owners_locked(gpu_id, slot);
    }

    size_t unresolved_queues(uint32_t gpu_id) const
    {
        auto lk = std::lock_guard<std::mutex>{m_mu};
        return unresolved_locked(gpu_id);
    }

    size_t live_queues() const
    {
        auto lk = std::lock_guard<std::mutex>{m_mu};
        return m_by_queue.size();
    }

private:
    struct queue_entry
    {
        uint32_t                gpu_id = 0;
        std::optional<uint32_t> slot   = {};
    };

    void remove_locked(uint64_t queue_token)
    {
        auto it = m_by_queue.find(queue_token);
        if(it == m_by_queue.end()) return;

        if(it->second.slot)
        {
            auto owner_it = m_owners.find({it->second.gpu_id, *it->second.slot});
            if(owner_it != m_owners.end() && --owner_it->second == 0) m_owners.erase(owner_it);
        }
        else
        {
            auto unres_it = m_unresolved.find(it->second.gpu_id);
            if(unres_it != m_unresolved.end() && --unres_it->second == 0)
                m_unresolved.erase(unres_it);
        }
        m_by_queue.erase(it);
    }

    size_t owners_locked(uint32_t gpu_id, uint32_t slot) const
    {
        auto it = m_owners.find({gpu_id, slot});
        return (it == m_owners.end()) ? 0 : it->second;
    }

    size_t unresolved_locked(uint32_t gpu_id) const
    {
        auto it = m_unresolved.find(gpu_id);
        return (it == m_unresolved.end()) ? 0 : it->second;
    }

    mutable std::mutex                                m_mu         = {};
    std::unordered_map<uint64_t, queue_entry>         m_by_queue   = {};
    std::map<std::pair<uint32_t, uint32_t>, size_t>   m_owners     = {};
    std::unordered_map<uint32_t, size_t>              m_unresolved = {};
};

// Which queues have already had HW profiling enabled, so the lazy path enables it
// exactly once per queue and only for a queue that actually takes the signal path.
class ProfilingEnableTracker
{
public:
    // True exactly once per queue: the caller then enables profiling on it.
    bool mark(uint64_t queue_token)
    {
        auto lk = std::lock_guard<std::mutex>{m_mu};
        return m_enabled.insert(queue_token).second;
    }

    void forget(uint64_t queue_token)
    {
        auto lk = std::lock_guard<std::mutex>{m_mu};
        m_enabled.erase(queue_token);
    }

    bool enabled(uint64_t queue_token) const
    {
        auto lk = std::lock_guard<std::mutex>{m_mu};
        return m_enabled.count(queue_token) != 0;
    }

    size_t size() const
    {
        auto lk = std::lock_guard<std::mutex>{m_mu};
        return m_enabled.size();
    }

private:
    mutable std::mutex             m_mu      = {};
    std::unordered_set<uint64_t>   m_enabled = {};
};
}  // namespace kfd
}  // namespace rocprofiler
