#pragma once

/**
 * @file snapshot.hpp
 * @brief Async-signal-safe per-thread snapshot record and SPSC ring used by
 *        the signal/timer capture backend.
 *
 * One snapshot is produced inside the timer-delivered signal handler (single
 * producer, the sampled thread) and consumed off-handler by the sampler worker
 * (single consumer). The record carries everything the off-handler libdwfl
 * unwind needs (saved GPRs + a bounded copy of the user stack + the stack base
 * address).
 *
 * The ring is a fixed-capacity, preallocated, lock-free single-producer /
 * single-consumer queue. ALL storage is allocated when the ring is constructed
 * (in attach_tid, off the signal path); the handler only fills a slot and
 * publishes it with a release store. No allocation, lock, or throw is reachable
 * from push().
 *
 * Header-only. C++20. x86_64.
 */

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include <sys/types.h>  // NOLINT(misc-include-cleaner): pid_t (include-cleaner mis-attributes)

#include "cpu_snap/sampling/unwind/perf_regs.hpp"

namespace cpu_snap::sampling::triggers::signal
{

/**
 * @brief One sampled snapshot captured inside the signal handler.
 *
 * Fixed-size POD. The register array is packed in the same PERF_REG_X86_* slot
 * order every regs_and_stack unwinder consumes (see k_perf_dwarf_n_regs_x86_64),
 * so a snapshot can be fed straight into dwarf_unwinder::unwind or
 * fp_copy_unwinder::unwind without repacking. The stack copy is a bounded
 * slab starting at the interrupted RSP.
 *
 * @note The handler fills regs + stack_size bytes of stack + the scalar fields;
 *       indices [stack_size, capacity) of stack are unspecified.
 */
struct snapshot
{
    /** Sample timestamp in nanoseconds of the configured sampler clock (matches the perf path). */
    std::uint64_t ts_ns = 0;
    /** Base address (interrupted RSP) of the captured stack slab in the thread VAS. */
    std::uint64_t stack_base_addr = 0;
    /** Number of valid bytes in @ref stack. */
    std::uint32_t stack_size = 0;
    /** Linux tid of the sampled thread. */
    pid_t tid = -1;  // NOLINT(misc-include-cleaner): pid_t via <sys/types.h>
    /** Saved GPRs packed in PERF_REG_X86_* slot order (RIP at slot 8, RSP at slot 7). */
    std::array<std::uint64_t, unwind::k_perf_dwarf_n_regs_x86_64> regs{};
    /**
     * Bounded copy of the user stack starting at @ref stack_base_addr.
     *
     * Heap-allocated once at ring construction and reused per slot; capacity is
     * fixed at snapshot_ring construction (stack_snapshot_bytes).
     */
    std::unique_ptr<std::uint8_t[]> stack;  // NOLINT(cppcoreguidelines-avoid-c-arrays)
};

/**
 * @brief Fixed-capacity lock-free SPSC ring of snapshot records.
 *
 * Bounded ring buffer with a power-of-two slot count. The producer (signal
 * handler) advances a head index; the consumer (worker) advances a tail index.
 * Both are std::atomic with acquire/release ordering, giving a wait-free,
 * allocation-free push() that is safe to call from a signal handler.
 *
 * Overflow policy: when the ring is full, push() drops the new snapshot and
 * increments an atomic lost counter (surfaced like perf PERF_RECORD_LOST). The
 * producer never blocks and never overwrites unconsumed data.
 *
 * @note Single producer, single consumer only. The producer is exactly one
 *       thread (the sampled thread, in-handler); the consumer is exactly one
 *       thread (the sampler worker).
 */
class snapshot_ring
{
public:
    /**
     * @brief Construct and preallocate every slot.
     *
     * @param capacity        Number of slots. Rounded up to a power of two,
     *                         minimum 2. All slots are allocated here.
     * @param stack_capacity  Bytes of stack storage per slot.
     */
    snapshot_ring(std::size_t capacity, std::uint32_t stack_capacity)
    : m_mask{ round_up_pow2(capacity) - 1 }
    , m_stack_capacity{ stack_capacity }
    , m_slots(m_mask + 1)
    {
        for(auto& slot : m_slots)
        {
            slot.stack = std::make_unique<std::uint8_t[]>(  // NOLINT(cppcoreguidelines-avoid-c-arrays)
                stack_capacity);
        }
    }

    snapshot_ring(const snapshot_ring&)            = delete;
    snapshot_ring& operator=(const snapshot_ring&) = delete;
    snapshot_ring(snapshot_ring&&)                 = delete;
    snapshot_ring& operator=(snapshot_ring&&)      = delete;
    ~snapshot_ring()                                      = default;

    /** @brief Per-slot stack copy capacity in bytes. */
    [[nodiscard]] std::uint32_t stack_capacity() const noexcept
    {
        return m_stack_capacity;
    }

    /**
     * @brief Reserve the next writable slot, or nullptr if the ring is full.
     *
     * ASYNC-SIGNAL-SAFE. Reads both indices, computes free space; returns a
     * pointer to the slot the producer should fill, WITHOUT advancing the head.
     * The producer fills the slot then calls commit() to publish it. On a full
     * ring this increments the lost counter and returns nullptr.
     *
     * @return Slot to fill, or nullptr when the ring is full.
     */
    [[nodiscard]] snapshot* acquire() noexcept
    {
        const std::size_t head = m_head.load(std::memory_order_relaxed);
        const std::size_t tail = m_tail.load(std::memory_order_acquire);
        if(head - tail > m_mask)
        {
            m_lost.fetch_add(1, std::memory_order_relaxed);
            return nullptr;
        }
        return &m_slots[head & m_mask];
    }

    /**
     * @brief Publish the slot returned by the most recent acquire().
     *
     * ASYNC-SIGNAL-SAFE. Release-stores the advanced head so the consumer sees
     * a fully written slot. Must be called exactly once per non-null acquire().
     *
     * @return Number of committed, not yet consumed slots including this one.
     */
    std::size_t commit() noexcept
    {
        const std::size_t head = m_head.load(std::memory_order_relaxed) + 1;
        m_head.store(head, std::memory_order_release);
        return head - m_tail.load(std::memory_order_acquire);
    }

    /**
     * @brief Fill depth at which the producer should wake the consumer.
     *
     * A quarter of the capacity (at least 1): the consumer's own timeout covers
     * a ring that fills more slowly, so only crossing this depth needs a wake.
     */
    [[nodiscard]] std::size_t wake_depth() const noexcept
    {
        return std::max<std::size_t>((m_mask + 1) / 4, 1);
    }

    /**
     * @brief Pop the oldest committed slot, or nullptr if empty.
     *
     * Consumer side (worker). The returned pointer is valid until the next
     * pop(); the consumer must finish reading before calling pop() again.
     *
     * @return Oldest committed snapshot, or nullptr when the ring is empty.
     */
    [[nodiscard]] const snapshot* pop() noexcept
    {
        const std::size_t tail = m_tail.load(std::memory_order_relaxed);
        const std::size_t head = m_head.load(std::memory_order_acquire);
        if(tail == head)
        {
            return nullptr;
        }
        return &m_slots[tail & m_mask];
    }

    /**
     * @brief Release the slot returned by the most recent pop().
     *
     * Consumer side. Advances the tail, freeing the slot for the producer.
     */
    void consume() noexcept
    {
        m_tail.store(m_tail.load(std::memory_order_relaxed) + 1,
                     std::memory_order_release);
    }

    /**
     * @brief Record the alternate signal stack the handler mapped for the
     *        sampled thread.
     *
     * ASYNC-SIGNAL-SAFE (a single atomic store). The worker reads it back via
     * @ref altstack once the handler can no longer run against this ring.
     */
    void set_altstack(void* base) noexcept
    {
        m_altstack.store(base, std::memory_order_release);
    }

    /** @brief Alternate stack mapped for the sampled thread, or nullptr. */
    [[nodiscard]] void* altstack() const noexcept
    {
        return m_altstack.load(std::memory_order_acquire);
    }

    /**
     * @brief Number of snapshots dropped on overflow since construction.
     * @return Cumulative dropped count.
     */
    [[nodiscard]] std::uint64_t lost() const noexcept
    {
        return m_lost.load(std::memory_order_relaxed);
    }

private:
    static constexpr std::size_t round_up_pow2(std::size_t n) noexcept
    {
        const std::size_t p = std::bit_ceil(n);
        return p < 2 ? std::size_t{ 2 } : p;
    }

    std::atomic<std::size_t>     m_head{ 0 };
    std::atomic<std::size_t>     m_tail{ 0 };
    std::atomic<std::uint64_t>   m_lost{ 0 };
    std::atomic<void*>           m_altstack{ nullptr };
    std::size_t                  m_mask           = 0;
    std::uint32_t                m_stack_capacity = 0;
    std::vector<snapshot> m_slots;
};

}  // namespace cpu_snap::sampling::triggers::signal
