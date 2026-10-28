#pragma once

#include <cstdint>

namespace cpu_snap::sampling
{

/**
 * @brief Stack-walking strategy.
 *
 * @note frame_pointer  rbp-chain walking: perf's kernel-walked
 *                      PERF_SAMPLE_CALLCHAIN, or (signal trigger)
 *                      fp_copy_unwinder walking a copied stack in the
 *                      worker. Misses frames when the target was built
 *                      without -fno-omit-frame-pointer.
 * @note dwarf          uses PERF_SAMPLE_REGS_USER + PERF_SAMPLE_STACK_USER +
 *                      libdwfl (elfutils). Works on any binary (no FPs
 *                      needed), at the cost of higher per-sample overhead
 *                      vs a frame-pointer walk.
 * @note auto_best      resolves to dwarf on every backend: correct for any
 *                      binary. frame_pointer is the cheaper explicit choice
 *                      when the target and its libraries keep frame pointers.
 */
enum class unwind_mode : std::uint8_t
{
    auto_best,
    frame_pointer,
    dwarf,
};

}  // namespace cpu_snap::sampling
