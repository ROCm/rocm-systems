#pragma once

/**
 * @file dwarf_unwinder.hpp
 * @brief libdwfl-based DWARF stack unwinder for perf_event_open
 *        PERF_SAMPLE_REGS_USER + PERF_SAMPLE_STACK_USER snapshots.
 *
 * The earlier libunwind custom-accessor approach aborted inside
 * libunwind because tdep_find_proc_info requires
 * `identity == unw_local_addr_space`. We switched to elfutils libdwfl,
 * which exposes a dedicated remote-thread API:
 *
 *   1. dwfl_begin + dwfl_report_elf (one call per object known to the
 *      dynamic loader) populate the Dwfl with the current process's
 *      modules / .eh_frame; the Dwfl is rebuilt when the loader changes.
 *   2. dwfl_attach_state(... Dwfl_Thread_Callbacks ...) plugs in our
 *      register source (saved perf snapshot) and memory source
 *      (saved stack slab + process_vm_readv fallback).
 *   3. dwfl_getthread_frames drives the unwind; each frame's IP is
 *      extracted with dwfl_frame_pc.
 *
 * Tested against libdw / elfutils 0.190 on Ubuntu (libdw-dev).
 *
 * Use:
 *   dwarf_unwinder dw;
 *   std::array<std::uint64_t, N> out_frames{};
 *   std::uint32_t n = dw.unwind(snap, out_frames.data(), N);
 *
 * @note x86_64 only.
 */

#include <cstddef>
#include <cstdint>
#include <optional>
#include <memory>
#include <span>

#include "cpu_snap/sampling/unwind/machine_snapshot.hpp"
#include "cpu_snap/sampling/unwind/perf_regs.hpp"

namespace cpu_snap::sampling::unwind
{


/**
 * @brief One sampled snapshot: register values + slab of stack memory.
 *
 * Pointers reference scratch buffers owned by the caller; this struct
 * does not own the data.
 */
struct dwarf_snapshot
{
    /** Register values, packed in PERF_REG_X86_* slot order. */
    const std::uint64_t* regs;
    /** Address of byte 0 of `stack_data` in the sampled thread's VAS. */
    std::uint64_t stack_addr;
    /** Pointer to the snapshot stack bytes. */
    const std::uint8_t* stack_data;
    /** Length of `stack_data`. */
    std::uint64_t stack_size;
};

/**
 * @brief Offset of the 8-byte word at @p addr inside the snapshot stack.
 *
 * Overflow-safe: an address near the top of the address space, below the
 * snapshot base, or a word that would run past the end yields nullopt.
 */
[[nodiscard]] constexpr std::optional<std::uint64_t>
snapshot_word_offset(const dwarf_snapshot& snap, std::uint64_t addr) noexcept
{
    constexpr std::uint64_t k_word = sizeof(std::uint64_t);
    if(snap.stack_size < k_word || addr < snap.stack_addr)
    {
        return std::nullopt;
    }
    const std::uint64_t offset = addr - snap.stack_addr;
    if(offset > snap.stack_size - k_word)
    {
        return std::nullopt;
    }
    return offset;
}

/**
 * @brief RAII wrapper around a libdwfl session bound to the current
 *        process's mappings.
 *
 * Each `unwind()` call attaches a fresh thread state pointing at the
 * caller's snapshot, walks the frames, and returns.
 *
 * Not thread-safe: each worker that needs to unwind must own its own
 * instance.
 */
class dwarf_unwinder
{
public:
    /** @brief Construct and report the current process's modules into libdwfl. */
    dwarf_unwinder() noexcept;

    ~dwarf_unwinder();

    dwarf_unwinder(const dwarf_unwinder&)            = delete;
    dwarf_unwinder& operator=(const dwarf_unwinder&) = delete;
    dwarf_unwinder(dwarf_unwinder&&)                 = delete;
    dwarf_unwinder& operator=(dwarf_unwinder&&)      = delete;

    /**
     * @brief Unwind from @p snap.
     *
     * Writes up to @p max_frames IPs to @p out (innermost-first,
     * matching the perf_event CALLCHAIN order).
     *
     * @param snap       Snapshot of caller-saved regs + stack bytes.
     * @param out        Output buffer, at least @p max_frames slots.
     * @param max_frames Maximum number of frames to write.
     * @return Number of frames written. Zero on any failure.
     */
    std::uint32_t unwind(const dwarf_snapshot& snap, std::uint64_t* out,
                         std::uint32_t max_frames) noexcept;

    /**
     * @brief Number of unwinds that read at least one word outside the
     *        snapshot window.
     *
     * Such a "live read" falls back to process_vm_readv on the sampled
     * process's current memory, which may differ from the memory at sample
     * time. A high count relative to the number of unwinds means parts of
     * the recovered stacks may be stale. The remedy is a larger snapshot
     * window: `perf.dwarf_stack_bytes` (perf backend) or
     * `signal.stack_snapshot_bytes` (signal backend).
     *
     * Safe to call from any thread while another thread unwinds; the value
     * is a relaxed snapshot.
     *
     * @return Cumulative count of unwind() calls that used a live read.
     */
    [[nodiscard]] std::uint64_t live_stack_read_samples() const noexcept;

    /** @brief capture_kind this unwinder needs from its trigger. */
    [[nodiscard]] static constexpr capture_kind required_capture() noexcept
    {
        return capture_kind::regs_and_stack;
    }

    /**
     * @brief Unwinder-concept-shaped entry point; adapts @p snap to
     *        @ref dwarf_snapshot and forwards to the overload above.
     * @param snap Machine snapshot (regs / stack_addr / stack populated).
     * @param out  Destination buffer.
     * @return Number of frames written. Zero on any failure.
     */
    [[nodiscard]] std::size_t unwind(const machine_snapshot& snap,
                                     std::span<std::uint64_t> out) noexcept
    {
        const dwarf_snapshot legacy{ snap.regs, snap.stack_addr,
                                    reinterpret_cast<const std::uint8_t*>(snap.stack.data()),
                                    static_cast<std::uint64_t>(snap.stack.size()) };
        return unwind(legacy, out.data(), static_cast<std::uint32_t>(out.size()));
    }

private:
    class impl;
    std::unique_ptr<impl> m_impl;
};

static_assert(Unwinder<dwarf_unwinder>);

}  // namespace cpu_snap::sampling::unwind
