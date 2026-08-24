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
  /// Index into the table span passed to `analyze_relocation_pairs()`.
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

/// @brief One `s_get_pc_i64` plus literal add that materializes an address.
///
/// @details The pair is the only way AMDGPU names a non-`.text` address from code, and it is
/// position-dependent: the literal is the distance from the instruction after the getpc to the
/// target. DBT relocates bodies, so the getpc observes a different PC and the same literal reaches
/// a different address. Recording both halves lets the patcher recompute the literal from the
/// getpc's final placement, which is what keeps the target fixed.
struct PcRelativeAddressBuilder {
  /// `.text`-relative offset of the `s_get_pc_i64`.
  uint64_t source_getpc_offset = 0;

  /// `.text`-relative offset of the in-place literal64 address add.
  uint64_t source_address_add_offset = 0;

  /// Virtual address the pair produces in the source object.
  uint64_t target_vaddr = 0;
};

/// @brief Results collected by the shared SGPR-pair dataflow.
struct RelocationPairAnalysis {
  /// Dispatches sorted by `(source_call_offset, table_index)` and unique on that key.
  std::vector<RelocationTableDispatch> dispatches;

  /// Address builders sorted and unique by `source_address_add_offset`.
  /// This complete set is independent of the supplied relocation tables.
  std::vector<PcRelativeAddressBuilder> address_builders;
};

/// @brief Discover table dispatches and address builders in one dataflow pass.
///
/// @details The analysis propagates a small SGPR-pair lattice. An
/// `s_get_pc_i64` plus literal add produces an address builder; a builder is
/// reported before its value can be reclassified as a known table base, so
/// `tables` affects only `dispatches`. A chained second add, a write to either
/// half, or a CFG join whose predecessors disagree leaves the pair unreported.
/// A table or GOT address followed by an indexed load and `s_swap_pc_i64`
/// produces a dispatch only when every step is proven.
[[nodiscard]] RelocationPairAnalysis
analyze_relocation_pairs(std::span<const std::unique_ptr<BasicBlock>> blocks,
                         std::span<const RelocationFunctionTable> tables, uint64_t text_vaddr);

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

/// @brief `.text`-relative offsets of every sized `STT_FUNC` symbol in the code object.
///
/// @details These are function entries, which makes them block leaders. A separately compiled
/// device library reaches most of its functions only through a pointer, so nothing in the decoded
/// instruction stream names them and CFG construction would otherwise let the padding before such
/// a function fall through into its body -- leaving the entry in the middle of a block, where it
/// cannot be adopted as a root or seeded with the ABI's architectural entry state.
///
/// Like the relocation tables above, and unlike anything derived from recovered dataflow, this set
/// is a fixed property of the input: the same symbols are found whether or not the object has
/// already been translated, so using it to partition `.text` stays a fixed point.
///
/// @returns Sorted, unique offsets. Empty when the object has no symbol table, more than one
/// `.text`, or no qualifying symbol.
[[nodiscard]] std::vector<uint64_t>
discover_text_function_symbol_offsets(const AmdGpuCodeObject &object);

/// @brief Whether every sized `STT_FUNC` in `.text` is an AMDHSA kernel.
///
/// @details A kernel is `<name>` with a companion `<name>.kd` descriptor object; a device function
/// has no companion. When this holds the object defines no device-function body, so a pointer
/// reaching an indirect transfer can only name a body in another code object -- and this
/// translation has none of its own to adopt as a root, retarget, or grow a descriptor for.
///
/// This is a scope fence, not a soundness argument. It bounds what admitting an unproven transfer
/// would own; the proof that such a target is already relocated has to come from the transfer's
/// own operand. It is vacuously true for an object with no such symbol, so it must never be the
/// only gate.
///
/// @returns True when no sized `.text` `STT_FUNC` lacks a `<name>.kd`. False when the symbol
/// tables cannot be read -- unlike its siblings here, an object this cannot inspect must not earn
/// the fence.
[[nodiscard]] bool object_defines_only_kernels(const AmdGpuCodeObject &object);

/// @brief Sized `.text` `STT_FUNC` offsets whose symbol a host could resolve.
///
/// @details When a translation relies on the whole-object relocation permission it also promises
/// that every externally resolvable `.text` symbol still names its body afterwards, because such a
/// symbol is exactly what an outside holder of a code address looked it up through. Those bodies
/// therefore have to be emitted, and a body no kernel reaches is emitted only if something adopts
/// it. Like the other symbol-derived sets here this is a fixed property of the input, so adopting
/// from it leaves the scope partition a fixed point.
///
/// @returns Sorted, unique offsets. Empty when the object has no symbol table or no such symbol.
[[nodiscard]] std::vector<uint64_t>
discover_externally_resolvable_text_function_offsets(const AmdGpuCodeObject &object);

/// @brief Function entries named by an `R_AMDGPU_RELATIVE64` addend landing in `.text`.
///
/// @details Each such addend is a stored function pointer, so its target is an address-taken body
/// that must be emitted and whose new placement the addend is rewritten to. discover_relocation_
/// function_tables() finds only the subset held by a qualifying `STT_OBJECT`; a compiler-anonymous
/// pointer array carries no such symbol, and its targets are just as dereferenced. Addends are ELF
/// data, so like symbols they are a fixed property of the input and safe to partition `.text` by.
///
/// @param function_entry_offsets Sorted function-entry offsets, from
///        discover_text_function_symbol_offsets(). A target is reported only when it is one of
///        these. Callers make every target a block leader and an adopted root seeded with the
///        ABI's entry state, which is only meaningful at a function boundary, so an addend that
///        names a mid-function label is omitted instead of guessed at -- leaving it unadopted, and
///        its object refused by relocate_relative_text_addends().
[[nodiscard]] std::vector<uint64_t>
discover_relative_text_addend_targets(const AmdGpuCodeObject &object,
                                      std::span<const uint64_t> function_entry_offsets);

} // namespace rocjitsu
