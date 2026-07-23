// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/relocation_function_table.h"

#include "rocjitsu/analysis/def_use_chain.h"
#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/code/basic_block.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/isa/operand.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <optional>
#include <queue>
#include <span>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace rocjitsu {
namespace {

template <typename T>
[[nodiscard]] bool read_object(std::span<const uint8_t> image, uint64_t offset, T &value) {
  if (offset > image.size() || sizeof(T) > image.size() - offset)
    return false;
  std::memcpy(&value, image.data() + offset, sizeof(T));
  return true;
}

[[nodiscard]] bool range_in_image(std::span<const uint8_t> image, uint64_t offset, uint64_t size) {
  return offset <= image.size() && size <= image.size() - offset;
}

struct ObjectCandidate {
  uint64_t vaddr = 0;
  uint64_t size = 0;
  std::vector<uint64_t> got_slots;
  std::vector<RelocationFunctionPointer> entries;
};

enum class PairValueKind {
  Address,
  TableBase,
  TableEntry,
};

struct PairValue {
  PairValueKind kind = PairValueKind::Address;
  uint64_t value = 0;
  uint64_t source_getpc_offset = 0;
  uint64_t source_address_add_offset = 0;
  uint64_t source_table_address_vaddr = 0;

  friend bool operator==(const PairValue &, const PairValue &) = default;
};

using PairState = std::unordered_map<uint16_t, PairValue>;

[[nodiscard]] std::optional<uint16_t> sgpr_pair(const Operand *operand) {
  if (operand == nullptr)
    return std::nullopt;
  const auto ref = operand->to_register_ref();
  if (!ref || ref->cls != RegClass::SGPR || ref->width < 2 ||
      static_cast<size_t>(ref->index) + 1 >= REGISTER_SET_MAX_SGPRS) {
    return std::nullopt;
  }
  return ref->index;
}

void kill_defined_pairs(PairState &state, const Instruction &inst) {
  if (state.empty())
    return;
  const InstDefUse def_use(inst);
  std::erase_if(state, [&](const auto &item) {
    const uint16_t pair = item.first;
    return def_use.defs.contains(RegisterRef{RegClass::SGPR, pair, 1}) ||
           def_use.defs.contains(RegisterRef{RegClass::SGPR, static_cast<uint16_t>(pair + 1), 1});
  });
}

[[nodiscard]] std::optional<uint64_t> literal64(const Operand *operand) {
  return operand == nullptr ? std::nullopt : operand->literal64_value();
}

[[nodiscard]] std::optional<std::pair<size_t, uint64_t>>
table_for_got_slot(std::span<const RelocationFunctionTable> tables, uint64_t vaddr) {
  for (size_t table_index = 0; table_index < tables.size(); ++table_index) {
    if (std::ranges::find(tables[table_index].got_slot_vaddrs, vaddr) !=
        tables[table_index].got_slot_vaddrs.end()) {
      return std::pair{table_index, vaddr};
    }
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<size_t>
table_at_address(std::span<const RelocationFunctionTable> tables, uint64_t vaddr) {
  const auto table = std::ranges::find(tables, vaddr, &RelocationFunctionTable::table_vaddr);
  if (table == tables.end())
    return std::nullopt;
  return static_cast<size_t>(table - tables.begin());
}

void transfer_instruction(PairState &state, const Instruction &inst,
                          std::span<const RelocationFunctionTable> tables, uint64_t text_vaddr,
                          std::vector<RelocationTableDispatch> *dispatches) {
  const std::string_view mnemonic = inst.mnemonic();
  // Only getpc can create a tracked fact from an empty state. Large linked
  // objects contain millions of unrelated instructions, so avoid decoding
  // operand register classes or constructing a full def/use set for them.
  if (state.empty() && mnemonic != "s_get_pc_i64" && mnemonic != "s_getpc_b64")
    return;
  const auto dst_pair = sgpr_pair(inst.dst_operand(0));
  const auto src0_pair = sgpr_pair(inst.src_operand(0));

  if (mnemonic == "s_swap_pc_i64" || mnemonic == "s_swappc_b64") {
    if (dispatches != nullptr && dst_pair && src0_pair) {
      const auto value = state.find(*src0_pair);
      if (value != state.end() && value->second.kind == PairValueKind::TableEntry) {
        dispatches->push_back(
            {.table_index = static_cast<size_t>(value->second.value),
             .source_call_offset = inst.src_loc(),
             .return_sreg = *dst_pair,
             .source_getpc_offset = value->second.source_getpc_offset,
             .source_address_add_offset = value->second.source_address_add_offset,
             .source_table_address_vaddr = value->second.source_table_address_vaddr});
      }
    }
    kill_defined_pairs(state, inst);
    return;
  }

  std::optional<PairValue> result;
  if ((mnemonic == "s_get_pc_i64" || mnemonic == "s_getpc_b64") && dst_pair) {
    if (inst.src_loc() <= std::numeric_limits<uint64_t>::max() - text_vaddr &&
        static_cast<uint64_t>(inst.size()) <=
            std::numeric_limits<uint64_t>::max() - text_vaddr - inst.src_loc()) {
      result = PairValue{.kind = PairValueKind::Address,
                         .value = text_vaddr + inst.src_loc() + static_cast<uint64_t>(inst.size()),
                         .source_getpc_offset = inst.src_loc()};
    }
  } else if (mnemonic == "s_add_nc_u64" && dst_pair) {
    const auto src1_pair = sgpr_pair(inst.src_operand(1));
    std::optional<uint16_t> address_pair;
    std::optional<uint64_t> addend;
    if (src0_pair && *src0_pair == *dst_pair) {
      address_pair = src0_pair;
      addend = literal64(inst.src_operand(1));
    } else if (src1_pair && *src1_pair == *dst_pair) {
      address_pair = src1_pair;
      addend = literal64(inst.src_operand(0));
    }
    if (address_pair && addend) {
      const auto address = state.find(*address_pair);
      // Only track a SINGLE address add. getpc produces an Address with
      // source_address_add_offset == 0; the first add sets it to this add's offset.
      // A second add would chain (target + addend1 + addend2), but the patcher only
      // rewrites the one recorded source_address_add_offset literal, so the earlier
      // addend would still execute. Refuse to track a second add and leave the pair
      // untracked below (fail closed) rather than record a value the relocation
      // cannot faithfully reproduce.
      if (address != state.end() && address->second.kind == PairValueKind::Address &&
          address->second.source_address_add_offset == 0) {
        result = address->second;
        // s_add_nc_u64 is modulo-2^64. A table or GOT below .text is addressed
        // with a two's-complement negative literal whose add wraps around; that is
        // the architecturally correct result, not an error. Add modulo and let the
        // table/GOT membership check below decide whether the result is a real
        // base — a wrapped value that matches no table simply stays an untracked
        // Address.
        result->value += *addend;
        result->source_address_add_offset = inst.src_loc();
        // RCCL materializes ncclDevFuncTable_{1,2,4} directly with getpc plus
        // a literal and then performs an indexed s_load_b64 from that base.
        if (const auto table = table_at_address(tables, result->value)) {
          result->kind = PairValueKind::TableBase;
          result->value = *table;
          result->source_table_address_vaddr = tables[*table].table_vaddr;
        }
      }
    }
  } else if (mnemonic == "s_load_b64" && dst_pair && src0_pair) {
    const auto base = state.find(*src0_pair);
    if (base != state.end()) {
      if (base->second.kind == PairValueKind::Address && inst.num_src_operands() >= 2 &&
          inst.src_operand(1) != nullptr && inst.src_operand(1)->encoding_value() == 0) {
        if (const auto table = table_for_got_slot(tables, base->second.value)) {
          result = PairValue{.kind = PairValueKind::TableBase,
                             .value = table->first,
                             .source_getpc_offset = base->second.source_getpc_offset,
                             .source_address_add_offset = base->second.source_address_add_offset,
                             .source_table_address_vaddr = table->second};
        }
      } else if (base->second.kind == PairValueKind::TableBase) {
        result = base->second;
        result->kind = PairValueKind::TableEntry;
      }
    }
  }

  kill_defined_pairs(state, inst);
  if (dst_pair && result)
    state[*dst_pair] = *result;
}

[[nodiscard]] PairState
meet_predecessors(const BasicBlock &block,
                  const std::unordered_map<const BasicBlock *, size_t> &positions,
                  const std::vector<PairState> &out, const std::vector<bool> &initialized) {
  PairState result;
  bool have_predecessor = false;
  for (const BasicBlock *predecessor : block.predecessors()) {
    const auto position = positions.find(predecessor);
    if (position == positions.end() || !initialized[position->second])
      continue;
    if (!have_predecessor) {
      result = out[position->second];
      have_predecessor = true;
      continue;
    }
    std::erase_if(result, [&](const auto &item) {
      const auto other = out[position->second].find(item.first);
      return other == out[position->second].end() || other->second != item.second;
    });
  }
  return result;
}

[[nodiscard]] ObjectCandidate *containing_candidate(std::vector<ObjectCandidate> &candidates,
                                                    uint64_t vaddr) {
  for (ObjectCandidate &candidate : candidates) {
    if (vaddr >= candidate.vaddr && vaddr - candidate.vaddr < candidate.size)
      return &candidate;
  }
  return nullptr;
}

} // namespace

std::vector<RelocationFunctionTable>
discover_relocation_function_tables(const AmdGpuCodeObject &object) {
  const auto *bytes = reinterpret_cast<const uint8_t *>(object.image_data());
  const std::span<const uint8_t> image(bytes, object.image_size());
  Elf64_Ehdr ehdr{};
  if (!read_object(image, 0, ehdr) || ehdr.e_shentsize != sizeof(Elf64_Shdr) ||
      !range_in_image(image, ehdr.e_shoff,
                      static_cast<uint64_t>(ehdr.e_shnum) * sizeof(Elf64_Shdr)))
    return {};

  std::vector<Elf64_Shdr> sections(ehdr.e_shnum);
  std::memcpy(sections.data(), image.data() + ehdr.e_shoff, sections.size() * sizeof(Elf64_Shdr));

  if (object.text_sections().size() != 1)
    return {};
  const Section &text = *object.text_sections().front();
  const uint64_t text_vaddr = text.vaddr();
  const uint64_t text_size = text.size();

  std::vector<ObjectCandidate> candidates;
  for (const Elf64_Shdr &symtab : sections) {
    if ((symtab.sh_type != SHT_SYMTAB && symtab.sh_type != SHT_DYNSYM) ||
        symtab.sh_entsize != sizeof(Elf64_Sym) ||
        !range_in_image(image, symtab.sh_offset, symtab.sh_size))
      continue;
    const size_t count = symtab.sh_size / sizeof(Elf64_Sym);
    for (size_t index = 0; index < count; ++index) {
      Elf64_Sym symbol{};
      if (!read_object(image, symtab.sh_offset + index * sizeof(Elf64_Sym), symbol))
        continue;
      if (elf_symbol_type(symbol.st_info) != kElfSymbolTypeObject || symbol.st_size == 0 ||
          (symbol.st_size % sizeof(uint64_t)) != 0 || symbol.st_shndx >= sections.size())
        continue;
      const Elf64_Shdr &section = sections[symbol.st_shndx];
      if ((section.sh_flags & SHF_ALLOC) == 0 || (section.sh_flags & SHF_EXECINSTR) != 0)
        continue;
      const auto duplicate = std::ranges::find_if(candidates, [&](const ObjectCandidate &item) {
        return item.vaddr == symbol.st_value && item.size == symbol.st_size;
      });
      if (duplicate == candidates.end())
        candidates.push_back(
            {.vaddr = symbol.st_value, .size = symbol.st_size, .got_slots = {}, .entries = {}});
    }
  }

  for (const Elf64_Shdr &relocations : sections) {
    if (relocations.sh_type != SHT_RELA || relocations.sh_entsize != sizeof(Elf64_Rela) ||
        !range_in_image(image, relocations.sh_offset, relocations.sh_size))
      continue;
    const Elf64_Shdr *linked_symbols =
        relocations.sh_link < sections.size() ? &sections[relocations.sh_link] : nullptr;
    const size_t count = relocations.sh_size / sizeof(Elf64_Rela);
    for (size_t index = 0; index < count; ++index) {
      Elf64_Rela rela{};
      if (!read_object(image, relocations.sh_offset + index * sizeof(Elf64_Rela), rela))
        continue;
      const uint32_t type = elf_reloc_type(rela.r_info);
      if (type == R_AMDGPU_RELATIVE64) {
        ObjectCandidate *candidate = containing_candidate(candidates, rela.r_offset);
        if (candidate == nullptr || ((rela.r_offset - candidate->vaddr) % sizeof(uint64_t)) != 0 ||
            rela.r_addend < 0)
          continue;
        const uint64_t target = static_cast<uint64_t>(rela.r_addend);
        if (target < text_vaddr || target - text_vaddr >= text_size)
          continue;
        candidate->entries.push_back(
            {.slot_vaddr = rela.r_offset, .target_text_offset = target - text_vaddr});
        continue;
      }
      if (type != R_AMDGPU_ABS64 || linked_symbols == nullptr ||
          (linked_symbols->sh_type != SHT_SYMTAB && linked_symbols->sh_type != SHT_DYNSYM) ||
          linked_symbols->sh_entsize != sizeof(Elf64_Sym) ||
          !range_in_image(image, linked_symbols->sh_offset, linked_symbols->sh_size))
        continue;
      const uint32_t symbol_index = elf_reloc_sym(rela.r_info);
      if (symbol_index >= linked_symbols->sh_size / sizeof(Elf64_Sym))
        continue;
      Elf64_Sym symbol{};
      if (!read_object(image,
                       linked_symbols->sh_offset +
                           static_cast<uint64_t>(symbol_index) * sizeof(Elf64_Sym),
                       symbol) ||
          elf_symbol_type(symbol.st_info) != kElfSymbolTypeObject)
        continue;
      uint64_t target = symbol.st_value;
      if (rela.r_addend >= 0) {
        const uint64_t addend = static_cast<uint64_t>(rela.r_addend);
        if (addend > std::numeric_limits<uint64_t>::max() - target)
          continue;
        target += addend;
      } else {
        // Avoid negating INT64_MIN in signed arithmetic.
        const uint64_t magnitude = uint64_t{0} - static_cast<uint64_t>(rela.r_addend);
        if (magnitude > target)
          continue;
        target -= magnitude;
      }
      const auto candidate = std::ranges::find_if(
          candidates, [&](const ObjectCandidate &item) { return item.vaddr == target; });
      if (candidate != candidates.end())
        candidate->got_slots.push_back(rela.r_offset);
    }
  }

  std::vector<RelocationFunctionTable> tables;
  for (ObjectCandidate &candidate : candidates) {
    // RCCL addresses its relocation-backed function tables directly from
    // executable text, so a GOT reference is useful evidence but not required.
    if (candidate.entries.empty())
      continue;
    std::ranges::sort(candidate.entries, {}, &RelocationFunctionPointer::slot_vaddr);
    std::ranges::sort(candidate.got_slots);
    candidate.got_slots.erase(std::ranges::unique(candidate.got_slots).begin(),
                              candidate.got_slots.end());
    tables.push_back({.table_vaddr = candidate.vaddr,
                      .table_size = candidate.size,
                      .got_slot_vaddrs = std::move(candidate.got_slots),
                      .entries = std::move(candidate.entries)});
  }
  std::ranges::sort(tables, {}, &RelocationFunctionTable::table_vaddr);
  return tables;
}

std::vector<RelocationTableDispatch>
discover_relocation_table_dispatches(std::span<const std::unique_ptr<BasicBlock>> blocks,
                                     std::span<const RelocationFunctionTable> tables,
                                     uint64_t text_vaddr) {
  if (blocks.empty() || tables.empty())
    return {};

  std::unordered_map<const BasicBlock *, size_t> positions;
  positions.reserve(blocks.size());
  for (size_t i = 0; i < blocks.size(); ++i) {
    if (blocks[i] != nullptr)
      positions.emplace(blocks[i].get(), i);
  }

  std::vector<PairState> in(blocks.size());
  std::vector<PairState> out(blocks.size());
  std::vector<bool> initialized(blocks.size(), false);
  std::vector<bool> queued(blocks.size(), false);
  std::queue<size_t> worklist;
  auto seed = [&](size_t i) {
    if (i < blocks.size() && blocks[i] != nullptr && !queued[i]) {
      worklist.push(i);
      queued[i] = true;
    }
  };
  for (size_t i = 0; i < blocks.size(); ++i) {
    if (blocks[i] != nullptr && blocks[i]->predecessors().empty())
      seed(i);
  }
  // Always seed the entry block. A kernel whose entry has an in-edge (e.g. a
  // back-edge from a loop that re-enters the first block) has no zero-predecessor
  // block, so the loop above would seed nothing and the analysis would silently
  // find no dispatches. blocks[0] is the entry by construction.
  seed(0);

  while (!worklist.empty()) {
    const size_t index = worklist.front();
    worklist.pop();
    queued[index] = false;
    BasicBlock *block = blocks[index].get();
    if (block == nullptr)
      continue;

    PairState next_in = meet_predecessors(*block, positions, out, initialized);
    PairState next_out = next_in;
    for (const Instruction &inst : block->instructions())
      transfer_instruction(next_out, inst, tables, text_vaddr, nullptr);

    const bool changed = !initialized[index] || next_in != in[index] || next_out != out[index];
    initialized[index] = true;
    in[index] = std::move(next_in);
    out[index] = std::move(next_out);
    if (!changed)
      continue;

    for (BasicBlock *successor : block->successors()) {
      const auto position = positions.find(successor);
      if (position != positions.end() && !queued[position->second]) {
        worklist.push(position->second);
        queued[position->second] = true;
      }
    }
  }

  std::vector<RelocationTableDispatch> dispatches;
  for (size_t i = 0; i < blocks.size(); ++i) {
    if (blocks[i] == nullptr || !initialized[i])
      continue;
    PairState state = in[i];
    for (const Instruction &inst : blocks[i]->instructions()) {
      transfer_instruction(state, inst, tables, text_vaddr, &dispatches);
    }
  }
  std::ranges::sort(dispatches, [](const auto &lhs, const auto &rhs) {
    if (lhs.source_call_offset != rhs.source_call_offset)
      return lhs.source_call_offset < rhs.source_call_offset;
    return lhs.table_index < rhs.table_index;
  });
  dispatches.erase(std::ranges::unique(dispatches, {},
                                       [](const RelocationTableDispatch &item) {
                                         return std::pair{item.source_call_offset,
                                                          item.table_index};
                                       })
                       .begin(),
                   dispatches.end());
  return dispatches;
}

} // namespace rocjitsu
