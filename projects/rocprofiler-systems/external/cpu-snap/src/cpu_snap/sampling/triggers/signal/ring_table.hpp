#pragma once

/**
 * @file triggers/signal/ring_table.hpp
 * @brief Lock-free tid -> snapshot_ring* table shared between the sampler worker
 *        (single writer) and the signal handlers (many async-signal readers).
 *
 * Open-addressed with linear probing, bounded to k_max_probe slots from a tid's
 * home slot: an entry is never placed farther away, so a handler lookup costs at
 * most k_max_probe atomic loads no matter how much tid churn the table has seen.
 * Detached entries leave a tombstone that is turned back into an empty slot as
 * soon as the probe chain behind it ends, so chains do not grow with churn.
 *
 * Every reader-visible transition goes from one valid state to another, so the
 * handler needs no lock. Header-only. C++20.
 */

#include <atomic>
#include <array>
#include <cstddef>
#include <cstdint>

#include <sys/types.h>  // NOLINT(misc-include-cleaner): pid_t (include-cleaner mis-attributes)

#include "cpu_snap/sampling/triggers/signal/snapshot.hpp"

namespace cpu_snap::sampling::triggers::signal
{

class ring_table
{
public:
    /** Number of slots; a power of two. */
    static constexpr std::size_t k_slots = 1024;
    /** Maximum distance of an entry from its home slot, and so the lookup bound. */
    static constexpr std::size_t k_max_probe = 64;
    /** Returned by retire() when the tid was not present. */
    static constexpr std::size_t k_npos = ~std::size_t{ 0 };

    /** @brief First slot probed for @p tid (Fibonacci hashing spreads sequential tids). */
    [[nodiscard]] static constexpr std::size_t home_slot(pid_t tid) noexcept
    {
        constexpr std::uint32_t k_hash_mult = 0x9E3779B1U;
        const std::uint32_t     mixed       = static_cast<std::uint32_t>(tid) * k_hash_mult;
        return static_cast<std::size_t>(mixed) & (k_slots - 1);
    }

    /**
     * @brief Insert or update the mapping for @p tid. Worker only.
     * @return False when no free slot exists within k_max_probe of the home slot;
     *         the table is unchanged.
     */
    [[nodiscard]] bool publish(pid_t tid, snapshot_ring* ring) noexcept
    {
        std::size_t idx        = home_slot(tid);
        std::size_t first_free = k_npos;
        for(std::size_t probe = 0; probe < k_max_probe; ++probe, idx = next(idx))
        {
            slot&       s   = m_slots[idx];
            const pid_t cur = s.tid.load(std::memory_order_acquire);
            if(cur == tid)
            {
                s.ring.store(ring, std::memory_order_release);
                return true;
            }
            if(cur == k_tid_tombstone && first_free == k_npos)
            {
                first_free = idx;
            }
            else if(cur == k_tid_empty)
            {
                if(first_free == k_npos)
                {
                    first_free = idx;
                }
                break;
            }
        }
        if(first_free == k_npos)
        {
            return false;
        }
        slot& s = m_slots[first_free];
        s.ring.store(ring, std::memory_order_release);
        s.tid.store(tid, std::memory_order_release);
        return true;
    }

    /**
     * @brief Remove the mapping for @p tid. Worker only.
     * @return The slot index the entry occupied, or k_npos if @p tid was absent.
     */
    std::size_t retire(pid_t tid) noexcept
    {
        std::size_t idx = home_slot(tid);
        for(std::size_t probe = 0; probe < k_max_probe; ++probe, idx = next(idx))
        {
            slot&       s   = m_slots[idx];
            const pid_t cur = s.tid.load(std::memory_order_acquire);
            if(cur == tid)
            {
                s.ring.store(nullptr, std::memory_order_seq_cst);
                s.tid.store(k_tid_tombstone, std::memory_order_seq_cst);
                reclaim_tombstones(idx);
                return idx;
            }
            if(cur == k_tid_empty)
            {
                break;
            }
        }
        return k_npos;
    }

    /** @brief Ring published for @p tid, or nullptr. ASYNC-SIGNAL-SAFE. */
    [[nodiscard]] snapshot_ring* lookup(pid_t tid) const noexcept
    {
        std::size_t idx = home_slot(tid);
        for(std::size_t probe = 0; probe < k_max_probe; ++probe, idx = next(idx))
        {
            const slot& s   = m_slots[idx];
            const pid_t cur = s.tid.load(std::memory_order_acquire);
            if(cur == tid)
            {
                return s.ring.load(std::memory_order_acquire);
            }
            if(cur == k_tid_empty)
            {
                return nullptr;
            }
        }
        return nullptr;
    }

    /**
     * @brief Handler-side entry into the slot of @p tid. ASYNC-SIGNAL-SAFE.
     *
     * Marks the slot busy before re-validating it, so a worker that retires the
     * entry and then sees the slot idle() knows no handler can still hold the
     * ring. Every successful enter() must be paired with leave().
     *
     * @param[out] slot_idx Slot to pass to leave(); meaningful only on success.
     * @return The ring for @p tid, or nullptr (and the slot is not busy).
     */
    [[nodiscard]] snapshot_ring* enter(pid_t tid, std::size_t& slot_idx) noexcept
    {
        std::size_t idx = home_slot(tid);
        for(std::size_t probe = 0; probe < k_max_probe; ++probe, idx = next(idx))
        {
            slot&       s   = m_slots[idx];
            const pid_t cur = s.tid.load(std::memory_order_acquire);
            if(cur == k_tid_empty)
            {
                return nullptr;
            }
            if(cur != tid)
            {
                continue;
            }
            s.busy.fetch_add(1, std::memory_order_seq_cst);
            snapshot_ring* ring = nullptr;
            if(s.tid.load(std::memory_order_seq_cst) == tid)
            {
                ring = s.ring.load(std::memory_order_acquire);
            }
            if(ring == nullptr)
            {
                s.busy.fetch_sub(1, std::memory_order_release);
                return nullptr;
            }
            slot_idx = idx;
            return ring;
        }
        return nullptr;
    }

    /** @brief Leave a slot entered with enter(). ASYNC-SIGNAL-SAFE. */
    void leave(std::size_t slot_idx) noexcept
    {
        m_slots[slot_idx].busy.fetch_sub(1, std::memory_order_release);
    }

    /**
     * @brief Whether no handler is currently inside @p slot_idx. Worker only.
     *
     * Once a retired entry's slot is idle, no handler holds its ring and the ring
     * may be freed: handlers that enter afterwards no longer find the entry.
     */
    [[nodiscard]] bool idle(std::size_t slot_idx) const noexcept
    {
        return m_slots[slot_idx].busy.load(std::memory_order_seq_cst) == 0;
    }

    /** @brief Drop every entry. Only valid when no handler can run (after fork). */
    void clear() noexcept
    {
        for(slot& s : m_slots)
        {
            s.ring.store(nullptr, std::memory_order_relaxed);
            s.tid.store(k_tid_empty, std::memory_order_relaxed);
            s.busy.store(0, std::memory_order_relaxed);
        }
    }

    /** @brief Slots that are not empty (live entries plus tombstones). */
    [[nodiscard]] std::size_t occupied_slots() const noexcept
    {
        std::size_t count = 0;
        for(const slot& s : m_slots)
        {
            count += (s.tid.load(std::memory_order_relaxed) != k_tid_empty) ? 1U : 0U;
        }
        return count;
    }

private:
    static constexpr pid_t k_tid_empty     = 0;
    static constexpr pid_t k_tid_tombstone = -1;

    struct slot
    {
        std::atomic<pid_t>          tid{ k_tid_empty };
        std::atomic<snapshot_ring*> ring{ nullptr };
        std::atomic<std::uint32_t>  busy{ 0 };
    };

    [[nodiscard]] static constexpr std::size_t next(std::size_t idx) noexcept
    {
        return (idx + 1) & (k_slots - 1);
    }

    [[nodiscard]] static constexpr std::size_t prev(std::size_t idx) noexcept
    {
        return (idx + k_slots - 1) & (k_slots - 1);
    }

    // A tombstone only has to keep later entries of a probe chain reachable. When
    // the slot after it is empty nothing lies behind it, so it, and every
    // tombstone directly in front of it, can become empty again.
    void reclaim_tombstones(std::size_t idx) noexcept
    {
        if(m_slots[next(idx)].tid.load(std::memory_order_acquire) != k_tid_empty)
        {
            return;
        }
        for(std::size_t steps = 0; steps < k_slots; ++steps, idx = prev(idx))
        {
            slot& s = m_slots[idx];
            if(s.tid.load(std::memory_order_acquire) != k_tid_tombstone)
            {
                return;
            }
            s.tid.store(k_tid_empty, std::memory_order_release);
        }
    }

    std::array<slot, k_slots> m_slots;
};

}  // namespace cpu_snap::sampling::triggers::signal
