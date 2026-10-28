#pragma once

/**
 * @file triggers/perf/ring_sizing.hpp
 * @brief Sizing rules tying the per-tid perf ring to the sample record size.
 *
 * Header-only. C++20.
 */

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>

#include "cpu_snap/sampling/unwind/perf_regs.hpp"
#include "cpu_snap/sampling/unwind_mode.hpp"

namespace cpu_snap::sampling::triggers::perf
{

/** Kernel limit for PERF_SAMPLE_STACK_USER (USHRT_MAX rounded down to 8). */
inline constexpr std::uint32_t k_max_dwarf_stack_bytes = 65528;

/** Ring pages used for frame-pointer sampling when the user does not choose. */
inline constexpr std::uint32_t k_default_fp_ring_pages = 8;

/** Records a default-sized ring must hold. */
inline constexpr std::size_t k_default_ring_records = 8;

/** Upper bound on perf_event_attr::wakeup_events. */
inline constexpr std::uint32_t k_max_wakeup_events = 32;

/** Kernel default for the callchain depth (perf_event_max_stack). */
inline constexpr std::size_t k_kernel_max_callchain = 127;

/**
 * @brief Largest PERF_RECORD_SAMPLE the kernel can emit for the given mode.
 * @param mode              Resolved unwind mode (frame_pointer or dwarf).
 * @param dwarf_stack_bytes PERF_SAMPLE_STACK_USER size; ignored unless dwarf.
 */
[[nodiscard]] constexpr std::size_t
max_sample_record_bytes(unwind_mode mode, std::uint32_t dwarf_stack_bytes) noexcept
{
    constexpr std::size_t word       = sizeof(std::uint64_t);
    constexpr std::size_t header_ip_tid_time = 4 * word;
    if(mode == unwind_mode::dwarf)
    {
        // abi + regs + stack size + stack data + dyn_size
        return header_ip_tid_time + word + word * unwind::k_perf_dwarf_n_regs_x86_64 +
               word + dwarf_stack_bytes + word;
    }
    return header_ip_tid_time + word + word * k_kernel_max_callchain;
}

/**
 * @brief Ring data pages used when the user left the size to the library.
 *
 * Power of two holding at least k_default_ring_records of the largest
 * record, never below the frame-pointer default.
 */
[[nodiscard]] constexpr std::uint32_t
default_ring_pages(unwind_mode mode, std::uint32_t dwarf_stack_bytes,
                   std::size_t page_bytes) noexcept
{
    const std::size_t want_bytes = k_default_ring_records *
                                   max_sample_record_bytes(mode, dwarf_stack_bytes);
    const std::size_t want_pages = (want_bytes + page_bytes - 1) / page_bytes;
    return std::max(k_default_fp_ring_pages,
                    static_cast<std::uint32_t>(std::bit_ceil(want_pages)));
}

/**
 * @brief Whether a ring of @p pages data pages can hold one maximal record.
 */
[[nodiscard]] constexpr bool
ring_holds_a_record(unwind_mode mode, std::uint32_t dwarf_stack_bytes, std::uint32_t pages,
                    std::size_t page_bytes) noexcept
{
    return static_cast<std::size_t>(pages) * page_bytes >=
           max_sample_record_bytes(mode, dwarf_stack_bytes);
}

/**
 * @brief perf_event_attr::wakeup_events for a ring of @p pages data pages.
 *
 * A quarter of the records the ring holds, so a wakeup leaves the worker
 * time to drain before the ring fills; at most k_max_wakeup_events.
 */
[[nodiscard]] constexpr std::uint32_t
wakeup_events_for(unwind_mode mode, std::uint32_t dwarf_stack_bytes, std::uint32_t pages,
                  std::size_t page_bytes) noexcept
{
    const std::size_t records =
        static_cast<std::size_t>(pages) * page_bytes /
        max_sample_record_bytes(mode, dwarf_stack_bytes);
    return static_cast<std::uint32_t>(
        std::clamp<std::size_t>(records / 4, 1, k_max_wakeup_events));
}

}  // namespace cpu_snap::sampling::triggers::perf
