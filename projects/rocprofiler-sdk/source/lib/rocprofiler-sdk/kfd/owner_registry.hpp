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

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <map>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <utility>

// Reverse doorbell-owner registry and the lazy HW-profiling bookkeeping.
//
// A firmware record identifies its queue only by a page-relative doorbell slot,
// so selecting a record is only sound when exactly ONE live queue owns that
// slot. Ownership is tracked for EVERY live compute queue, populated at queue
// creation, so a queue that predates the session or has never dispatched still
// counts as an owner.
//
// LOCK ORDERING: this mutex is NEVER held while the hub's is taken, or vice
// versa. Every operation returns a verdict the caller acts on after releasing
// the lock, so the two are never nested in either direction.

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

    // `slot` is nullopt when the queue's doorbell has not resolved yet.
    add_result add_queue(uint64_t queue_token, uint32_t gpu_id, std::optional<uint32_t> slot)
    {
        if(m_abandoned.load(std::memory_order_acquire)) return add_result::slot_unknown;
        auto lk = std::lock_guard<std::mutex>{m_mu};
        remove_locked(queue_token);

        m_by_queue[queue_token] = queue_entry{gpu_id, slot};
        if(!slot)
        {
            ++m_unresolved[gpu_id];
            return add_result::slot_unknown;
        }

        auto _owner_count = ++m_owners[{gpu_id, *slot}];
        return (_owner_count > 1) ? add_result::collision : add_result::sole_owner;
    }

    void remove_queue(uint64_t queue_token)
    {
        if(m_abandoned.load(std::memory_order_acquire)) return;
        auto lk = std::lock_guard<std::mutex>{m_mu};
        remove_locked(queue_token);
    }

    // Exactly one live owner for this slot on this GPU. False also when the slot
    // is unknown, so an unresolved doorbell is never treated as injective.
    bool slot_uniquely_owned(uint32_t gpu_id, uint32_t slot) const
    {
        if(m_abandoned.load(std::memory_order_acquire)) return false;
        auto lk = std::lock_guard<std::mutex>{m_mu};
        if(unresolved_locked(gpu_id) != 0) return false;
        return owners_locked(gpu_id, slot) == 1;
    }

    // The slot this queue owns, if it is live and its doorbell resolved.
    std::optional<uint32_t> slot_of(uint64_t queue_token) const
    {
        if(m_abandoned.load(std::memory_order_acquire)) return std::nullopt;
        auto lk = std::lock_guard<std::mutex>{m_mu};
        auto it = m_by_queue.find(queue_token);
        if(it == m_by_queue.end()) return std::nullopt;
        return it->second.slot;
    }

    // The GPU this queue lives on, if it is still live. The close path needs it to
    // address the hub's per-(gpu, slot) quarantine.
    std::optional<uint32_t> gpu_of(uint64_t queue_token) const
    {
        if(m_abandoned.load(std::memory_order_acquire)) return std::nullopt;
        auto lk = std::lock_guard<std::mutex>{m_mu};
        auto it = m_by_queue.find(queue_token);
        if(it == m_by_queue.end()) return std::nullopt;
        return it->second.gpu_id;
    }

    size_t owners_of(uint32_t gpu_id, uint32_t slot) const
    {
        if(m_abandoned.load(std::memory_order_acquire)) return 0;
        auto lk = std::lock_guard<std::mutex>{m_mu};
        return owners_locked(gpu_id, slot);
    }

    size_t unresolved_queues(uint32_t gpu_id) const
    {
        if(m_abandoned.load(std::memory_order_acquire)) return 0;
        auto lk = std::lock_guard<std::mutex>{m_mu};
        return unresolved_locked(gpu_id);
    }

    size_t live_queues() const
    {
        if(m_abandoned.load(std::memory_order_acquire)) return 0;
        auto lk = std::lock_guard<std::mutex>{m_mu};
        return m_by_queue.size();
    }

    // pthread_atfork child handler. One atomic store; every operation checks it
    // before taking the mutex.
    void abandon_in_child() { m_abandoned.store(true, std::memory_order_release); }

    bool abandoned() const { return m_abandoned.load(std::memory_order_acquire); }

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

    // A doorbell slot is only unique per GPU, so ownership is keyed by the pair.
    using slot_key = std::pair<uint32_t, uint32_t>;  // (gpu_id, doorbell_slot)
    // queue_token -> the GPU and slot that queue owns.
    using queue_entry_map = std::unordered_map<uint64_t, queue_entry>;
    // slot_key -> how many live queues own that slot (1 == uniquely owned).
    using slot_owner_count_map = std::map<slot_key, size_t>;
    // gpu_id -> how many live queues on it have no resolved doorbell yet.
    using unresolved_count_map = std::unordered_map<uint32_t, size_t>;

    std::atomic<bool>    m_abandoned  = {false};
    mutable std::mutex   m_mu         = {};
    queue_entry_map      m_by_queue   = {};
    slot_owner_count_map m_owners     = {};
    unresolved_count_map m_unresolved = {};
};
}  // namespace kfd
}  // namespace rocprofiler
