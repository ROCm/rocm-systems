#pragma once

/**
 * @file machine_snapshot.hpp
 * @brief Shared shape every alternative of the closed `unwinder` variant
 *        consumes, plus the concept that verifies it at compile time.
 *
 * Header-only. C++20.
 */

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <span>

namespace cpu_snap::sampling::unwind
{

/**
 * @brief What kind of interrupted-thread state a trigger hands to an unwinder.
 *
 * @note kernel_callchain the kernel already walked the stack (perf's
 *                        frame-pointer PERF_SAMPLE_CALLCHAIN mode); the
 *                        unwinder is a passthrough.
 * @note regs_and_stack   raw register file + a bounded copy of the user
 *                        stack; the unwinder must walk it itself.
 */
enum class capture_kind : std::uint8_t
{
    kernel_callchain,
    regs_and_stack,
};

/**
 * @brief One decoded sample, in whichever shape its capture_kind produces.
 *
 * Exactly one of the two payloads is populated by the producing trigger,
 * matching the capture_kind it reports: @ref ready_frames for
 * kernel_callchain, @ref regs / @ref stack_addr / @ref stack for
 * regs_and_stack. Every field is a non-owning view into caller-owned
 * scratch buffers, valid only for the duration of the unwind() call.
 */
struct machine_snapshot
{
    /** kernel_callchain: already-walked IPs, innermost frame first. */
    std::span<const std::uint64_t> ready_frames;

    /** regs_and_stack: register snapshot, packed in PERF_REG_X86_* slot order. */
    const std::uint64_t* regs = nullptr;
    /** regs_and_stack: address of stack[0] in the sampled thread's VAS. */
    std::uint64_t stack_addr = 0;
    /** regs_and_stack: bounded copy of the user stack starting at stack_addr. */
    std::span<const std::byte> stack;
};

/**
 * @brief Shape every alternative of the closed `unwinder` variant satisfies.
 *
 * unwind() is called from inside a loop that is instantiated once per
 * drain batch against a single concrete type (see
 * triggers::perf::perf_sample_decoder::process_sample); it must stay a concrete,
 * inlinable call - never virtual, never std::function.
 */
template <typename U>
concept Unwinder = requires(U& u, const machine_snapshot& snap, std::span<std::uint64_t> out) {
    { u.unwind(snap, out) } -> std::convertible_to<std::size_t>;
    { U::required_capture() } -> std::convertible_to<capture_kind>;
};

}  // namespace cpu_snap::sampling::unwind
