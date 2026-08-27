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
/// (gfx942/gfx950) and the gfx12.5 layout used by CDNA5 (gfx1250),
/// cross-checked against projects/rocdbgapi/src/architecture.cpp.

#ifndef ROCJITSU_KMD_LINUX_CWSR_H_
#define ROCJITSU_KMD_LINUX_CWSR_H_

#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/kmd/linux/amdgpu_properties.h"

#include <array>
#include <cstdint>
#include <functional>
#include <span>
#include <vector>

namespace rocjitsu {
namespace kmd {

/// @brief Context-save record layouts implemented by this codec.
enum class CwsrLayoutKind : uint8_t {
  Unsupported,
  Gfx9_4,
  Gfx12_5,
};

/// @brief Select the exact context-save ABI implemented for @p arch.
/// @details This is deliberately fail-closed. Wave width, register count, and
/// TTMP placement are insufficient to infer compatibility with a dbgapi record.
constexpr CwsrLayoutKind cwsr_layout_kind(rj_code_arch_t arch) {
  // gfx908 adds an ACC-VGPR block and gfx908/gfx90a keep the packet id in
  // TTMP6, so CDNA1/CDNA2 are deliberately excluded despite also being gfx9.
  switch (arch) {
  case ROCJITSU_CODE_ARCH_CDNA3:
  case ROCJITSU_CODE_ARCH_CDNA4:
    return CwsrLayoutKind::Gfx9_4;
  case ROCJITSU_CODE_ARCH_CDNA5:
    return CwsrLayoutKind::Gfx12_5;
  default:
    return CwsrLayoutKind::Unsupported;
  }
}

/// @brief Whether this module models the CWSR record layout for @p arch.
/// @details The codec reproduces the gfx9.4 and gfx12.5 layouts. Differences on
/// other generations are not confined to one field: the control stack carries a
/// different number of state words, COMPUTE_RELAUNCH packs different bits, the
/// VGPR stride assumes wave64, the SGPR alias slots sit elsewhere, and the
/// dispatch identity lives in different TTMPs. Serializing a wave from an
/// unmodelled architecture would hand rocm-dbgapi an image it decodes against
/// its own layout, which is worse than refusing: the debugger reads plausible
/// nonsense instead of reporting that the target is unsupported.
/// @param arch Architecture of the GPU whose queue would be serialized.
/// @returns True if serialize/deserialize produce an image dbgapi can decode.
constexpr bool cwsr_layout_modelled(rj_code_arch_t arch) {
  return cwsr_layout_kind(arch) != CwsrLayoutKind::Unsupported;
}

/// @brief The same question as cwsr_layout_modelled(), asked of a GPU named the
/// way KFD names one.
/// @details The two predicates exist because their callers hold different
/// identities. The DBG_TRAP paths have a SoC and so an arch enum; the KFD
/// topology paths only ever see a @c gfx_target_version, which they translate to
/// the GC hardware IP version the amdkfd driver keys on. Kept adjacent so the
/// two architecture spellings cannot be updated apart --
/// KfdTopologyTest.ArchAndGcSpellingsOfTheCwsrGateAgree pins them together.
/// \NPI update this alongside cwsr_layout_modelled() when the codec learns a
/// new record layout.
/// @param gc_ip_version Packed GC hardware IP version, as
///        gc_ip_version_for_gfx_target_version() produces.
/// @returns True if serialize/deserialize produce an image dbgapi can decode.
constexpr CwsrLayoutKind cwsr_layout_kind_for_gc_ip_version(uint32_t gc_ip_version) {
  if (gc_ip_version == make_gc_ip_version(9, 4, 3) || // gfx942 (CDNA3)
      gc_ip_version == make_gc_ip_version(9, 5, 0))   // gfx950 (CDNA4)
    return CwsrLayoutKind::Gfx9_4;
  if (gc_ip_version == make_gc_ip_version(12, 1, 0)) // gfx1250 (CDNA5)
    return CwsrLayoutKind::Gfx12_5;
  return CwsrLayoutKind::Unsupported;
}

constexpr bool cwsr_layout_modelled_for_gc_ip_version(uint32_t gc_ip_version) {
  return cwsr_layout_kind_for_gc_ip_version(gc_ip_version) != CwsrLayoutKind::Unsupported;
}

/// @brief Number of scalar slots in a saved SGPR block (gfx9.4 sgpr_count).
/// @details The block is a fixed 112 slots regardless of how many scalars the
/// dispatch actually allocated. It is also an input to the COMPUTE_RELAUNCH
/// state word (sgprs_field = kCwsrSavedSgprSlots / 16, which must stay inside
/// three bits), not just to the alias arithmetic below.
inline constexpr uint32_t kCwsrSavedSgprSlots = 112;
/// @brief Number of saved scalar slots in a gfx12.5 record.
/// @details The last 16 slots carry TTMP0-15 rather than ordinary SGPRs.
inline constexpr uint32_t kCwsrGfx1250SavedSgprSlots = 128;
/// @brief gfx9.4 scalar_register_count(): the architected scalars s0..s101.
/// @details Selectors at or above this alias VCC, FLAT_SCRATCH and XNACK_MASK
/// rather than naming an SGPR.
inline constexpr uint32_t kCwsrArchScalarRegisters = 102;
/// @brief One past the last aliased slot (+ gfx9.4 scalar_alias_count() of 6).
inline constexpr uint32_t kCwsrAliasedSgprEnd = kCwsrArchScalarRegisters + 6; // 108
/// @brief Slot holding VCC_LO; VCC_HI is the next one up.
inline constexpr uint32_t kCwsrVccLoSlot = kCwsrAliasedSgprEnd - 2; // 106
/// @brief Slot holding FLAT_SCRATCH_LO; FLAT_SCRATCH_HI is the next one up.
inline constexpr uint32_t kCwsrFlatScratchLoSlot = kCwsrAliasedSgprEnd - 6; // 102
static_assert(kCwsrAliasedSgprEnd <= kCwsrSavedSgprSlots,
              "aliased slots must land inside the saved SGPR block");

/// @brief Whether a saved SGPR slot is occupied by an aliased register.
///
/// @details gfx9.4 stores FLAT_SCRATCH and VCC in slots near the top of the
/// block. gfx12.5 stores FLAT_SCRATCH in its HWREG block, retains the VCC
/// aliases, and reserves the final 16 scalar slots for TTMP0-15. Every path
/// that walks the block by slot index must consult this so a round trip does
/// not read an alias or TTMP back as an SGPR. On gfx9.4 the aliases are not a
/// contiguous top range: s104 and s105 sit between FLAT_SCRATCH and VCC and
/// are ordinary registers, so a `slot < 102` bound is not equivalent.
/// @param slot Index into the saved SGPR block.
/// @returns True if the slot holds an alias rather than SGPR @p slot.
constexpr bool cwsr_sgpr_slot_is_aliased(uint32_t slot, rj_code_arch_t arch) {
  if (cwsr_layout_kind(arch) == CwsrLayoutKind::Gfx12_5)
    return slot == kCwsrVccLoSlot || slot == kCwsrVccLoSlot + 1 ||
           slot >= kCwsrGfx1250SavedSgprSlots - 16;
  return slot == kCwsrVccLoSlot || slot == kCwsrVccLoSlot + 1 || slot == kCwsrFlatScratchLoSlot ||
         slot == kCwsrFlatScratchLoSlot + 1;
}

/// @brief The saved architectural state of one stopped wave.
///
/// @details Register values are the live wave state; the serializer places them
/// at the CWSR offsets rocm-dbgapi expects and synthesizes the trap-temporary
/// registers (TTMP4-11) that carry the wave's debugger metadata.
struct CwsrWaveState {
  uint64_t pc = 0;   ///< Program counter (past the s_trap on a breakpoint).
  uint64_t exec = 0; ///< EXEC mask.
  uint64_t vcc = 0;  ///< VCC (placed into its aliased SGPR slot).
  /// FLAT_SCRATCH base register. gfx9.4 places it in aliased SGPR slots;
  /// gfx12.5 places it in the dedicated HWREG block. rocm-dbgapi validates the
  /// scratch base it computes from COMPUTE_TMPRING_SIZE against this register
  /// (wave.cpp scratch_memory_region); a mismatch disables private-memory
  /// access.
  uint64_t flat_scratch = 0;
  uint32_t status = 0;  ///< STATUS register.
  uint32_t trapsts = 0; ///< TRAPSTS register.
  uint32_t mode = 0;    ///< MODE register.
  uint32_t m0 = 0;      ///< M0 register.

  /// gfx12.5 split the gfx9 STATUS/TRAPSTS/MODE debug fields across dedicated
  /// saved registers. These fields are ignored by the gfx9.4 codec.
  uint32_t state_priv = 0;
  uint32_t excp_flag_priv = 0;
  uint32_t excp_flag_user = 0;
  uint32_t trap_ctrl = 0;
  uint32_t xnack_state_priv = 0;
  uint32_t xnack_mask = 0;

  uint64_t wave_id = 0; ///< Stable, unique wave id (TTMP4:5) dbgapi reads as its own.
  /// Workgroup coordinates: gfx9.4 TTMP8/9/10; gfx12.5 X in TTMP9 and Y/Z in TTMP7.
  std::array<uint32_t, 3> group_ids{};
  /// Wave index: gfx9.4 TTMP11[0:5]; gfx12.5 TTMP8[25:29].
  uint32_t wave_in_group = 0;
  /// Dispatch packet id: gfx9.4 TTMP11[6:30]; gfx12.5 TTMP8[0:24].
  uint32_t queue_packet_id = 0;
  /// Whether this wave is the first / last of its workgroup in the control stack.
  /// rocm-dbgapi groups consecutive control-stack waves into a workgroup: the
  /// first wave becomes the group leader and following waves must share its
  /// group ids until the last wave closes the group (rocdbgapi queue.cpp
  /// update_waves). These must therefore be set per workgroup, not per queue.
  bool is_first_in_group = true;
  bool is_last_in_group = true;
  /// This wave's slot in the queue's scratch allocation (COMPUTE_RELAUNCH wave
  /// word bits [0:8] on gfx9.4 and [0:9] on gfx12.5). rocm-dbgapi multiplies it
  /// by the per-wave scratch size to locate the wave's private memory, so it
  /// must match the wave's scratch base.
  uint32_t scratch_scoreboard_id = 0;
  /// Physical shader engine containing the wave. gfx12.5 records this in the
  /// COMPUTE_RELAUNCH wave word so scratch slots are interpreted per SE.
  uint32_t shader_engine_id = 0;
  /// Trap id from s_trap: gfx9.4 TTMP6[25:28]; gfx12.5 TTMP11[28:31].
  uint32_t trap_id = 0;
  bool wave_stopped = true;       ///< Whether the wave is stopped (TTMP6[30]).
  bool saved_status_halt = false; ///< Saved STATUS.HALT (TTMP6[29]).
  /// Whether the gfx9.4 SPI initialized the dispatch-bookkeeping TTMPs. Mirrors
  /// kfd_runtime_info.ttmp_setup; when false, TTMP6[31]
  /// (spi_ttmps_setup_disabled) is set and rocm-dbgapi skips packet/workgroup
  /// correlation. gfx12.5 uses the validity/debug markers in TTMP8 instead.
  bool spi_ttmps_setup = false;

  /// Meaningful scalar registers, at most @ref kCwsrSavedSgprSlots. Slots for
  /// which @ref cwsr_sgpr_slot_is_aliased holds do not round trip: they carry
  /// VCC and FLAT_SCRATCH, which travel in their own fields.
  uint32_t num_sgprs = 0;
  /// Meaningful vector registers: at most 256 for gfx9.4 or 1024 for gfx12.5.
  uint32_t num_vgprs = 0;
  /// Scalar register values, index = sgpr number.
  std::vector<uint32_t> sgprs;
  /// Vector register values, index = vgpr_number * 64 + lane.
  std::vector<uint32_t> vgprs;
  /// Workgroup LDS bytes. Present only on the first wave in each workgroup.
  std::vector<uint8_t> lds;
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
/// The written layout satisfies rocm-dbgapi's architecture-specific invariants:
/// a header at the base, a control stack contiguous with the wave save area,
/// and per-wave register blocks at the offsets the selected architecture's
/// cwsr_record_t computes. gfx9.4 uses one relaunch state word and saves
/// wave64 VGPRs; gfx12.5 uses two state words, saves wave32 VGPRs, and moves
/// TTMPs into the final 16 SGPR slots.
///
/// The serializer requires a dword-aligned base, register counts within the
/// selected architecture's saved blocks, and a non-null writer. The caller
/// must keep the queue suspended and wave/register storage alive for the whole
/// call, serialize against a stable snapshot, and provide a writer that
/// publishes directly to the inferior-visible coherent CWSR mapping. The
/// serializer writes payload before the header, but cross-thread/process
/// exclusion and any required cache maintenance remain the caller's
/// responsibility.
CwsrLayout serialize_queue_cwsr(uint64_t ctx_base, uint32_t area_size,
                                const std::vector<CwsrWaveState> &waves,
                                const std::function<void(uint64_t, uint32_t)> &write32,
                                rj_code_arch_t arch);

/// @brief Serialize a queue using two block writes: payload first, then header.
///
/// @details This is the production-oriented equivalent of
/// @ref serialize_queue_cwsr. It constructs the same byte-exact image but
/// publishes the contiguous control-stack/wave payload with one callback and
/// the 40-byte ABI header with a second callback. Keeping the header write last
/// preserves the CWSR publication contract while avoiding a GPU page-table
/// lookup and lock acquisition for every dword of every wave register.
///
/// @param write_block Callback receiving a target address and contiguous bytes.
/// @returns The chosen layout; @ref CwsrLayout::ok is false and no callback is
///          made when the image is invalid or does not fit.
CwsrLayout serialize_queue_cwsr_bulk(
    uint64_t ctx_base, uint32_t area_size, const std::vector<CwsrWaveState> &waves,
    const std::function<void(uint64_t, std::span<const uint8_t>)> &write_block,
    rj_code_arch_t arch);

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
                            const std::function<uint32_t(uint64_t)> &read32, rj_code_arch_t arch);

/// @brief Read and deserialize a queue CWSR image using block transfers.
///
/// @details Validates the ABI header against the geometry implied by @p waves,
/// reads the contiguous control-stack/wave payload once, then applies the same
/// decoder as @ref deserialize_queue_cwsr. The queue must remain suspended for
/// the complete operation.
bool deserialize_queue_cwsr_bulk(
    uint64_t ctx_base, uint32_t area_size, std::vector<CwsrWaveState> &waves,
    const std::function<void(uint64_t, std::span<uint8_t>)> &read_block, rj_code_arch_t arch);

} // namespace kmd
} // namespace rocjitsu

#endif // ROCJITSU_KMD_LINUX_CWSR_H_
