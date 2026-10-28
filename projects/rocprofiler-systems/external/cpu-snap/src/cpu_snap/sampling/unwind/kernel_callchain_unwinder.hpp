#pragma once

/**
 * @file kernel_callchain_unwinder.hpp
 * @brief Passthrough unwinder for already kernel-walked callchains.
 *
 * When a trigger already delivers a fully-walked callchain (perf's
 * frame-pointer PERF_SAMPLE_CALLCHAIN mode), no unwinding is needed: this
 * type formalizes that no-op as a real Unwinder so the sampler's closed
 * unwinder variant can represent it uniformly alongside dwarf_unwinder.
 *
 * Header-only. C++20.
 */

#include <algorithm>
#include <cstddef>
#include <span>

#include "cpu_snap/sampling/unwind/machine_snapshot.hpp"

namespace cpu_snap::sampling::unwind
{

/**
 * @brief Copies an already-walked callchain, turning return addresses into
 *        call-site addresses.
 *
 * Entry 0 is the interrupted instruction and stays exact. Every later entry
 * is a return address, i.e. the instruction AFTER the call; subtracting one
 * lands inside the call so symbolization and line lookup attribute the frame
 * to the call and not to whatever follows it.
 */
class kernel_callchain_unwinder
{
public:
    [[nodiscard]] static constexpr capture_kind required_capture() noexcept
    {
        return capture_kind::kernel_callchain;
    }

    /**
     * @brief Copy @p snap.ready_frames into @p out, normalising non-leaf
     *        frames to call-site addresses (a zero entry stays zero).
     * @param snap Snapshot whose ready_frames holds the kernel-walked chain.
     * @param out  Destination buffer.
     * @return Number of frames copied (bounded by both spans' sizes).
     */
    [[nodiscard]] std::size_t unwind(const machine_snapshot& snap,
                                     std::span<std::uint64_t> out) const noexcept
    {
        const std::size_t n = std::min(snap.ready_frames.size(), out.size());
        std::copy_n(snap.ready_frames.begin(), n, out.begin());
        for(std::size_t i = 1; i < n; ++i)
        {
            if(out[i] != 0)
            {
                --out[i];
            }
        }
        return n;
    }
};

static_assert(Unwinder<kernel_callchain_unwinder>);

}  // namespace cpu_snap::sampling::unwind
