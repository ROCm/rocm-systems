#pragma once

/**
 * @file perf_regs.hpp
 * @brief x86_64 PERF_SAMPLE_REGS_USER register mask and packed-slot indices.
 */

#include <bit>
#include <cstdint>

#include <asm/perf_regs.h>

namespace cpu_snap::sampling::unwind
{

// NOLINTBEGIN(readability-identifier-naming): project-wide k_* constant naming.
/**
 * @brief Subset of x86_64 registers requested via PERF_SAMPLE_REGS_USER.
 *
 * Mask = all GPRs + RIP + FLAGS. The kernel delivers the snapshot as a
 * packed u64 array in the order of set bits (low bit first).
 */
inline constexpr std::uint64_t k_perf_dwarf_regs_mask_x86_64 =
    (1ULL << PERF_REG_X86_AX) | (1ULL << PERF_REG_X86_BX) | (1ULL << PERF_REG_X86_CX) |
    (1ULL << PERF_REG_X86_DX) | (1ULL << PERF_REG_X86_SI) | (1ULL << PERF_REG_X86_DI) |
    (1ULL << PERF_REG_X86_BP) | (1ULL << PERF_REG_X86_SP) | (1ULL << PERF_REG_X86_IP) |
    (1ULL << PERF_REG_X86_FLAGS) | (1ULL << PERF_REG_X86_R8) | (1ULL << PERF_REG_X86_R9) |
    (1ULL << PERF_REG_X86_R10) | (1ULL << PERF_REG_X86_R11) | (1ULL << PERF_REG_X86_R12) |
    (1ULL << PERF_REG_X86_R13) | (1ULL << PERF_REG_X86_R14) | (1ULL << PERF_REG_X86_R15);

/** @brief Number of u64 slots delivered by the kernel for the mask above. */
inline constexpr std::uint32_t k_perf_dwarf_n_regs_x86_64 =
    static_cast<std::uint32_t>(std::popcount(k_perf_dwarf_regs_mask_x86_64));

/** @brief Index of a PERF_REG_X86_* register in the packed snapshot array. */
constexpr std::uint32_t
perf_slot_of(unsigned reg) noexcept
{
    return static_cast<std::uint32_t>(
        std::popcount(k_perf_dwarf_regs_mask_x86_64 & ((1ULL << reg) - 1)));
}

inline constexpr std::uint32_t k_perf_slot_rbp = perf_slot_of(PERF_REG_X86_BP);
inline constexpr std::uint32_t k_perf_slot_rip = perf_slot_of(PERF_REG_X86_IP);
inline constexpr std::uint32_t k_perf_slot_rsp = perf_slot_of(PERF_REG_X86_SP);
// NOLINTEND(readability-identifier-naming)

static_assert(k_perf_slot_rbp == 6, "kernel perf register ABI changed");
static_assert(k_perf_slot_rip == 8, "kernel perf register ABI changed");
static_assert(k_perf_dwarf_n_regs_x86_64 == 18, "kernel perf register ABI changed");

}  // namespace cpu_snap::sampling::unwind
