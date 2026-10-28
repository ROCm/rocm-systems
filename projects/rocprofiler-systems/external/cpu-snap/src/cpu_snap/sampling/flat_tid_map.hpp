#pragma once

/**
 * @file flat_tid_map.hpp
 * @brief Per-tid associative store optimized for the recorder hot path.
 *
 * A direct-mapped 256-slot table keyed by (tid mod 256) absorbs almost
 * all lookups without the bucket -> node -> {key, value} pointer
 * chase that std::unordered_map performs. An overflow
 * std::unordered_map catches the rare workloads where the slot table
 * fills.
 */

#include <array>
#include <cstddef>
#include <cstdint>
#include <sys/types.h>
#include <unordered_map>

namespace cpu_snap::sampling
{

/**
 * @brief Per-tid associative store with a fast-path direct-mapped table of
 *        256 slots keyed by (tid mod 256) + linear probe; growth fallback
 *        to std::unordered_map<pid_t, T> when the slot cluster fills.
 *
 * Tuned for the recorder's hot path where most workloads have fewer
 * than 256 concurrent tids and tids cluster around a small range
 * (sequential allocations from /proc/self/task).
 *
 * @note Not thread-safe; intended for single-worker hot path.
 */
template <typename T>
class flat_tid_map
{
public:
    static constexpr std::size_t k_slot_count = 256;

    /**
     * @brief Get-or-insert the entry for @p tid.
     * @param tid Linux thread id key.
     * @return Reference to the stored value. Valid until the next
     *         insert that triggers an overflow-map rehash.
     */
    T& operator[](pid_t tid)
    {
        const std::size_t home       = bucket_for(tid);
        std::size_t       insert_idx = k_slot_count;
        for(std::size_t step = 0; step < k_probe_limit; ++step)
        {
            const std::size_t idx       = (home + step) & (k_slot_count - 1);
            slot&             candidate = m_slots[idx];
            if(candidate.state == slot_state::occupied)
            {
                if(candidate.tid == tid) return candidate.value;
                continue;
            }
            if(insert_idx == k_slot_count) insert_idx = idx;
            if(candidate.state == slot_state::empty) break;
        }
        if(auto it = m_overflow.find(tid); it != m_overflow.end())
        {
            return it->second;
        }
        if(insert_idx != k_slot_count)
        {
            slot& target = m_slots[insert_idx];
            target.state = slot_state::occupied;
            target.tid   = tid;
            target.value = T{};
            return target.value;
        }
        return m_overflow[tid];
    }

    /**
     * @brief Look up @p tid without insertion.
     * @param tid Linux thread id key.
     * @return Pointer to the value if present, nullptr otherwise.
     */
    [[nodiscard]] T* find(pid_t tid) noexcept
    {
        const std::size_t home = bucket_for(tid);
        for(std::size_t step = 0; step < k_probe_limit; ++step)
        {
            const std::size_t idx = (home + step) & (k_slot_count - 1);
            if(m_slots[idx].state == slot_state::empty) break;
            if(m_slots[idx].state == slot_state::occupied && m_slots[idx].tid == tid)
            {
                return &m_slots[idx].value;
            }
        }
        if(auto it = m_overflow.find(tid); it != m_overflow.end())
        {
            return &it->second;
        }
        return nullptr;
    }

    /**
     * @brief Const lookup overload.
     * @param tid Linux thread id key.
     * @return Pointer to the value if present, nullptr otherwise.
     * @see find
     */
    [[nodiscard]] const T* find(pid_t tid) const noexcept
    {
        const std::size_t home = bucket_for(tid);
        for(std::size_t step = 0; step < k_probe_limit; ++step)
        {
            const std::size_t idx = (home + step) & (k_slot_count - 1);
            if(m_slots[idx].state == slot_state::empty) break;
            if(m_slots[idx].state == slot_state::occupied && m_slots[idx].tid == tid)
            {
                return &m_slots[idx].value;
            }
        }
        if(auto it = m_overflow.find(tid); it != m_overflow.end())
        {
            return &it->second;
        }
        return nullptr;
    }

    /**
     * @brief Erase the entry for @p tid (no-op if absent).
     * @param tid Linux thread id key.
     */
    void erase(pid_t tid)
    {
        const std::size_t home = bucket_for(tid);
        for(std::size_t step = 0; step < k_probe_limit; ++step)
        {
            const std::size_t idx = (home + step) & (k_slot_count - 1);
            if(m_slots[idx].state == slot_state::empty) break;
            if(m_slots[idx].state == slot_state::occupied && m_slots[idx].tid == tid)
            {
                m_slots[idx]       = {};
                m_slots[idx].state = slot_state::deleted;
                return;
            }
        }
        m_overflow.erase(tid);
    }

    /**
     * @brief Number of tracked entries across both tables.
     * @return Total entry count.
     */
    [[nodiscard]] std::size_t size() const noexcept
    {
        std::size_t count = 0;
        for(const auto& slot_entry : m_slots)
        {
            if(slot_entry.state == slot_state::occupied) ++count;
        }
        return count + m_overflow.size();
    }

    /**
     * @brief Number of entries currently in the overflow map.
     * @return Overflow-map size.
     */
    [[nodiscard]] std::size_t overflow_size() const noexcept
    {
        return m_overflow.size();
    }

    /**
     * @brief Visit every occupied entry.
     *
     * Iteration order is unspecified.
     * @param callback Callable with signature `void(pid_t tid, T& value)`.
     */
    template <typename Fn>
    void for_each(Fn&& callback)
    {
        for(auto& slot_entry : m_slots)
        {
            if(slot_entry.state == slot_state::occupied)
            {
                std::forward<Fn>(callback)(slot_entry.tid, slot_entry.value);
            }
        }
        for(auto& [tid, value] : m_overflow)
        {
            std::forward<Fn>(callback)(tid, value);
        }
    }

    /**
     * @brief Const overload of for_each.
     * @param callback Callable with signature `void(pid_t tid, const T& value)`.
     */
    template <typename Fn>
    void for_each(Fn&& callback) const
    {
        for(const auto& slot_entry : m_slots)
        {
            if(slot_entry.state == slot_state::occupied)
            {
                std::forward<Fn>(callback)(slot_entry.tid, slot_entry.value);
            }
        }
        for(const auto& [tid, value] : m_overflow)
        {
            std::forward<Fn>(callback)(tid, value);
        }
    }

private:
    static constexpr std::size_t k_probe_limit = 8;

    // tombstone keeps probe chains intact
    enum class slot_state : std::uint8_t
    {
        empty,
        occupied,
        deleted,
    };

    struct slot
    {
        slot_state state = slot_state::empty;
        pid_t      tid   = 0;
        T          value{};
    };

    // Cheap hash: low bits of tid mod 256. Linux assigns tids
    // sequentially so adjacent threads land in adjacent slots; the
    // linear probe absorbs short collision runs from that pattern.
    [[nodiscard]] static std::size_t bucket_for(pid_t tid) noexcept
    {
        return static_cast<std::size_t>(tid) & (k_slot_count - 1);
    }

    std::array<slot, k_slot_count> m_slots{};
    std::unordered_map<pid_t, T>   m_overflow;
};

}  // namespace cpu_snap::sampling
