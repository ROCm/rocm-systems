// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file relocation_function_table.h
/// @brief Discovery and CFG modeling of loader-relocated device-function tables.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace rocjitsu {

class AmdGpuCodeObject;
class BasicBlock;

/// @brief One populated device-function pointer in a relocation-backed table.
///
/// @details Linked AMDGPU code objects represent a function pointer stored in
/// non-executable data with a symbol-less `R_AMDGPU_RELATIVE64` relocation. The
/// loader writes `load_bias + r_addend` at `r_offset`; when the addend names
/// `.text`, the resulting value is a callable device address. DBT retains both
/// sides of that relationship so moving the target instruction can update the
/// relocation addend without interpreting the table's source-language type.
struct RelocationFunctionPointer {
  /// Virtual address of the table slot and `R_AMDGPU_RELATIVE64` relocation place.
  uint64_t slot_vaddr = 0;

  /// Original `.text`-relative byte offset encoded by the relocation addend.
  uint64_t target_text_offset = 0;
};

/// @brief One finite data object populated with relocated device-function pointers.
///
/// @details The object is identified structurally: an allocated, non-executable
/// `STT_OBJECT` contains one or more aligned slots with symbol-less
/// `R_AMDGPU_RELATIVE64` addends into `.text`. Code may materialize the table
/// address directly or load it through a GOT slot. Consequently GOT references
/// strengthen discovery but are optional and are recorded separately from the
/// populated entries.
struct RelocationFunctionTable {
  /// Original virtual address from the defining object's `st_value`.
  uint64_t table_vaddr = 0;

  /// Object extent from `st_size`, used to associate relocation places with this table.
  uint64_t table_size = 0;

  /// GOT relocation places whose `R_AMDGPU_ABS64` value resolves to this object.
  /// Empty when translated code addresses the table directly.
  std::vector<uint64_t> got_slot_vaddrs;

  /// Populated slots, sorted by `slot_vaddr`. Slots without a qualifying text
  /// relocation are not possible callees and do not appear here.
  std::vector<RelocationFunctionPointer> entries;
};

/// @brief One dynamically indexed table load feeding a call-like scalar PC swap.
///
/// @details This record connects source CFG recovery with final text patching.
/// `table_index` supplies the finite callee set, while the source offsets locate
/// the call and the PC-relative address builder that must remain pointed at the
/// same data object after `.text` grows.
struct RelocationTableDispatch {
  /// Index into the table vector passed to `discover_relocation_table_dispatches()`.
  size_t table_index = 0;

  /// Original `.text`-relative offset of the `s_swap_pc_i64` call instruction.
  uint64_t source_call_offset = 0;

  /// Low SGPR of the pair that receives the architectural return PC.
  uint16_t return_sreg = 0;

  /// Original `.text`-relative offset of the address builder's `s_get_pc_i64`.
  uint64_t source_getpc_offset = 0;

  /// Original `.text`-relative offset of the in-place literal64 address add.
  uint64_t source_address_add_offset = 0;

  /// Original non-executable virtual address materialized by getpc plus the literal.
  ///
  /// This is either a GOT slot containing the table base or the table address
  /// itself. DBT keeps the same section-relative data target when relocated
  /// text changes the PC observed by `s_get_pc_i64`.
  uint64_t source_table_address_vaddr = 0;
};

/// @brief Discover finite device-call tables from ELF symbols and relocations.
///
/// @details A candidate must be a non-empty `STT_OBJECT` whose size is a
/// multiple of eight bytes. It must reside in an allocated, non-executable
/// section and contain at least one aligned
/// symbol-less `R_AMDGPU_RELATIVE64` relocation whose addend lands in the code
/// object's single `.text` section. `R_AMDGPU_ABS64` references to the object
/// are recorded as optional GOT slots. Discovery deliberately depends on ELF
/// structure rather than table or application symbol names.
///
/// @returns Tables sorted by `table_vaddr`. Invalid, ambiguous, and unsupported
/// ELF records are ignored; an object without a qualifying populated entry is
/// not returned.
[[nodiscard]] std::vector<RelocationFunctionTable>
discover_relocation_function_tables(const AmdGpuCodeObject &object);

/// @brief Resolve decoded dynamic calls back to relocation-discovered tables.
///
/// @details The analysis propagates only a small SGPR-pair lattice:
/// `s_get_pc_i64 + s_add_nc_u64 literal64` produces an address. That address
/// may name the table directly or a GOT slot whose zero-offset load produces
/// the table base. A subsequent indexed `s_load_b64` produces an entry value,
/// and a dispatch is reported only when that value reaches `s_swap_pc_i64`.
/// Writes to either SGPR half kill the fact, and CFG joins retain only facts
/// that agree on every initialized predecessor, so unproven calls remain
/// unresolved rather than acquiring speculative table edges.
[[nodiscard]] std::vector<RelocationTableDispatch>
discover_relocation_table_dispatches(std::span<const std::unique_ptr<BasicBlock>> blocks,
                                     std::span<const RelocationFunctionTable> tables,
                                     uint64_t text_vaddr);

} // namespace rocjitsu
