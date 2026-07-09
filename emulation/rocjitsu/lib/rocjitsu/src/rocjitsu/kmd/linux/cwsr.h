// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file cwsr.h
/// @brief Serialization of stopped-wave state into the KFD context-save-restore
/// (CWSR) area in the exact layout rocm-dbgapi parses.
///
/// @details rocm-dbgapi never reads wave registers through an ioctl. It reads
/// them by `pread`-ing the inferior's /proc/<pid>/mem at the queue's
/// context-save-restore GPU virtual address, interpreting a control stack and a
/// per-wave save area whose byte layout is defined by the CWSR ABI. The
/// emulator hosts wave state in the daemon, so on a wave stop it must write that
/// state into the (memfd-shared) CWSR area at exactly the offsets dbgapi
/// computes. This module reproduces the gfx9.4 layout used by CDNA3/CDNA4
/// (gfx942/gfx950), cross-checked against projects/rocdbgapi/src/architecture.cpp
/// (gfx90a/gfx9_4/mi cwsr_record_t). Earlier gfx9 generations differ: gfx908
/// saves an ACC-VGPR block, and gfx908/gfx90a keep the packet id in TTMP6.

#ifndef ROCJITSU_KMD_LINUX_CWSR_H_
#define ROCJITSU_KMD_LINUX_CWSR_H_

#include <array>
#include <cstdint>
#include <functional>
#include <vector>

namespace rocjitsu {
namespace kmd {

/// @brief The saved architectural state of one stopped wave.
///
/// @details Register values are the live wave state; the serializer places them
/// at the CWSR offsets rocm-dbgapi expects and synthesizes the trap-temporary
/// registers (TTMP4-11) that carry the wave's debugger metadata.
struct CwsrWaveState {
  uint64_t pc = 0;   ///< Program counter (past the s_trap on a breakpoint).
  uint64_t exec = 0; ///< EXEC mask.
  uint64_t vcc = 0;  ///< VCC (placed into its aliased SGPR slot).
  /// FLAT_SCRATCH base register (gfx9_4 architected flat scratch), placed into
  /// its aliased SGPR slot. rocm-dbgapi validates the scratch base it computes
  /// from COMPUTE_TMPRING_SIZE against this register (wave.cpp
  /// scratch_memory_region); a mismatch disables private-memory access.
  uint64_t flat_scratch = 0;
  uint32_t status = 0;  ///< STATUS register.
  uint32_t trapsts = 0; ///< TRAPSTS register.
  uint32_t mode = 0;    ///< MODE register.
  uint32_t m0 = 0;      ///< M0 register.

  uint64_t wave_id = 0; ///< Stable, unique wave id (TTMP4:5) dbgapi reads as its own.
  std::array<uint32_t, 3> group_ids{}; ///< Workgroup coordinates (TTMP8/9/10).
  uint32_t wave_in_group = 0;          ///< Wave index within the workgroup (TTMP11[0:5]).
  uint32_t queue_packet_id = 0;        ///< Dispatch packet id (TTMP11[6:30]).
  uint32_t trap_id = 0;                ///< Trap id from the s_trap (TTMP6[25:28]).
  bool wave_stopped = true;            ///< Whether the wave is stopped (TTMP6[30]).
  bool saved_status_halt = false;      ///< Saved STATUS.HALT (TTMP6[29]).
  /// Whether the SPI initialized the dispatch-bookkeeping TTMPs (group ids in
  /// TTMP8-10, packet id in TTMP11). Mirrors kfd_runtime_info.ttmp_setup; when
  /// false, TTMP6[31] (spi_ttmps_setup_disabled) is set and rocm-dbgapi skips
  /// packet/workgroup correlation for the wave.
  bool spi_ttmps_setup = false;

  uint32_t num_sgprs = 0; ///< Meaningful scalar registers (at most s0-s105).
  uint32_t num_vgprs = 0; ///< Meaningful wave64 vector registers (at most 256).
  /// Scalar register values, index = sgpr number.
  std::vector<uint32_t> sgprs;
  /// Vector register values, index = vgpr_number * 64 + lane.
  std::vector<uint32_t> vgprs;
};

/// @brief The CWSR geometry chosen for a serialized wave save area.
///
/// @details Returned so callers (and tests) can locate the per-register offsets
/// dbgapi will compute. All offsets are relative to the ctx-save base.
struct CwsrLayout {
  uint32_t control_stack_offset = 0; ///< Offset to the first control-stack word.
  uint32_t control_stack_size = 0;   ///< Control-stack size in bytes.
  /// Offset to the high end (one past the last byte) of the wave-state area.
  uint32_t wave_state_offset = 0;
  uint32_t wave_state_size = 0; ///< Size extending down from @ref wave_state_offset.
  uint32_t debug_offset = 0;    ///< Offset of the debugger displaced-step region.
  uint32_t debug_size = 0;      ///< Size of the debugger displaced-step region.
  bool ok = false;              ///< False if the waves do not fit in the ctx-save area.
};

/// @brief Serialize a queue's stopped waves into its CWSR area.
///
/// @param ctx_base Context-save-restore GPU virtual address (from the queue).
/// @param area_size Context-save-restore area size in bytes (from the queue).
/// @param waves The stopped waves, in the order they should appear (the first
///        owns the workgroup's LDS region at the top of the save area).
/// @param write32 Callback that writes one dword to a GPU virtual address.
/// @returns The chosen layout; @ref CwsrLayout::ok is false if the waves do not
///          fit, in which case nothing is written.
///
/// The written layout satisfies rocm-dbgapi's invariants: a header at the base,
/// a control stack (two skipped PM4 words, one state word, one wave word per
/// wave) contiguous with the wave save area, and per-wave register blocks laid
/// out high-to-low as [TTMP|HWREG] / [SGPR] / [VGPR] at the offsets
/// gfx9_architecture_t::cwsr_record_t::register_address computes.
///
/// This gfx9.4 serializer requires a dword-aligned base, no more than 106
/// meaningful SGPRs or 256 VGPRs per wave, and a non-null writer. The caller
/// must keep the queue suspended and wave/register storage alive for the whole
/// call, serialize against a stable snapshot, and provide a writer that
/// publishes directly to the inferior-visible coherent CWSR mapping. The
/// serializer writes payload before the header, but cross-thread/process
/// exclusion and any required cache maintenance remain the caller's
/// responsibility.
CwsrLayout serialize_queue_cwsr(uint64_t ctx_base, uint32_t area_size,
                                const std::vector<CwsrWaveState> &waves,
                                const std::function<void(uint64_t, uint32_t)> &write32);

/// @brief Read wave register state back from a serialized CWSR area.
///
/// @details The inverse of @ref serialize_queue_cwsr. rocm-dbgapi writes a
/// stopped wave's registers (PC, EXEC, VCC, STATUS, MODE, SGPRs, VGPRs, and the
/// TTMP6 run/stop and MODE.debug_en bits) straight into the CWSR area via
/// /proc/<pid>/mem; on resume the driver reads them back to apply the debugger's
/// edits to the live wave and to learn the requested run vs. single-step state.
///
/// @param ctx_base Context-save-restore GPU virtual address (from the queue).
/// @param area_size Context-save-restore area size in bytes (from the queue).
/// @param waves In: each element supplies the wave geometry (@ref
///        CwsrWaveState::num_sgprs / @ref CwsrWaveState::num_vgprs) and ordering
///        used to reproduce the exact layout; Out: filled with the values
///        currently stored in the area. The count and geometry must match the
///        @ref serialize_queue_cwsr call that produced the area.
/// @param read32 Callback that reads one dword from a GPU virtual address.
/// @returns True on success; false if the waves do not fit the area (in which
///          case @p waves is left unchanged).
bool deserialize_queue_cwsr(uint64_t ctx_base, uint32_t area_size,
                            std::vector<CwsrWaveState> &waves,
                            const std::function<uint32_t(uint64_t)> &read32);

} // namespace kmd
} // namespace rocjitsu

#endif // ROCJITSU_KMD_LINUX_CWSR_H_
