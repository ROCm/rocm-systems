// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file relocation_function_table.h
/// @brief Discovery of loader-relocated device-function pointer tables.

#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace rocjitsu {

class AmdGpuCodeObject;
class BasicBlock;

/// @brief One populated pointer slot in a relocation-backed function table.
struct RelocationFunctionPointer {
  uint64_t slot_vaddr = 0;
  uint64_t target_text_offset = 0;
};

/// @brief One data object reached through the GOT and populated with text pointers.
struct RelocationFunctionTable {
  uint64_t table_vaddr = 0;
  uint64_t table_size = 0;
  std::vector<uint64_t> got_slot_vaddrs;
  std::vector<RelocationFunctionPointer> entries;
};

/// @brief One dynamically indexed table load feeding a call-like scalar PC swap.
struct RelocationTableDispatch {
  size_t table_index = 0;
  uint64_t source_call_offset = 0;
  uint16_t return_sreg = 0;
  uint64_t source_getpc_offset = 0;
  uint64_t source_address_add_offset = 0;
  uint64_t got_slot_vaddr = 0;
};

/// @brief Discover finite device-call tables from ELF symbols and relocations.
///
/// @details A candidate must be an STT_OBJECT, be referenced by an
/// R_AMDGPU_ABS64 GOT-style relocation, and contain aligned
/// R_AMDGPU_RELATIVE64 slots whose addends land in `.text`. The requirements
/// deliberately avoid symbol-name conventions such as `ncclDevFuncTable`.
[[nodiscard]] std::vector<RelocationFunctionTable>
discover_relocation_function_tables(const AmdGpuCodeObject &object);

/// @brief Resolve decoded dynamic calls back to relocation-discovered tables.
///
/// @details The analysis propagates only a small SGPR-pair lattice:
/// `s_get_pc_i64 + s_add_nc_u64 literal64` produces an address, loading a
/// discovered GOT slot produces a table base, and loading through that base
/// produces a table entry. A call is reported only when that entry reaches an
/// `s_swap_pc_i64`; conflicting CFG paths erase the fact.
[[nodiscard]] std::vector<RelocationTableDispatch>
discover_relocation_table_dispatches(
    std::span<const std::unique_ptr<BasicBlock>> blocks,
    std::span<const RelocationFunctionTable> tables, uint64_t text_vaddr);

} // namespace rocjitsu
