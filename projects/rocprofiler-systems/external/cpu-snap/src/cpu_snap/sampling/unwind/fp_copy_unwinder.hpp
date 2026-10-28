#pragma once

/**
 * @file fp_copy_unwinder.hpp
 * @brief Classic rbp-chain unwinder over an already-copied stack snapshot.
 *
 * Walks the frame-pointer chain entirely over the bytes already captured
 * into a machine_snapshot::stack slab: no libdw, no live-process memory
 * access. This is what makes signal + frame_pointer possible: the signal
 * backend can only ever hand an unwinder a regs_and_stack snapshot (it has
 * no kernel-walked-callchain equivalent), and until this type existed the
 * only unwinder that could consume that snapshot was dwarf_unwinder.
 *
 * Header-only declaration; unwind() body lives in fp_copy_unwinder.cpp
 * (real stack-walking / bounds-checking logic, not header-only-trivial).
 * C++20. x86_64.
 */

#include <cstddef>
#include <span>

#include "cpu_snap/sampling/unwind/machine_snapshot.hpp"

namespace cpu_snap::sampling::unwind
{

/**
 * @brief Walks the rbp chain over a bounded copy of the user stack.
 *
 * No internal state; stateless and safe to share across snapshots. Never
 * reads outside the copied stack window, even against corrupt or
 * adversarial stack contents.
 */
class fp_copy_unwinder
{
public:
    /** @brief capture_kind this unwinder needs from its trigger. */
    [[nodiscard]] static constexpr capture_kind required_capture() noexcept
    {
        return capture_kind::regs_and_stack;
    }

    /**
     * @brief Walk the rbp chain starting from @p snap's registers.
     *
     * out[0] is the leaf IP (the interrupted RIP), taken directly from
     * @p snap.regs, unadjusted. Each subsequent slot is the return address
     * read from the caller's saved frame, also unadjusted (no ip-1
     * call-site normalization; matches every other unwinder in this
     * codebase today).
     *
     * @param snap Snapshot with regs / stack_addr / stack populated
     *             (capture_kind::regs_and_stack).
     * @param out  Destination buffer.
     * @return Number of frames written (at least 1 when snap.regs is
     *         non-null and @p out is non-empty; 0 otherwise).
     */
    [[nodiscard]] std::size_t unwind(const machine_snapshot& snap,
                                     std::span<std::uint64_t> out) const noexcept;
};

static_assert(Unwinder<fp_copy_unwinder>);

}  // namespace cpu_snap::sampling::unwind
