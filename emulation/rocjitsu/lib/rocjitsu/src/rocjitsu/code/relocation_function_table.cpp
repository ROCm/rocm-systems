// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/relocation_function_table.h"

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/code/analysis/def_use_chain.h"
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
#include <unordered_set>
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

/// @details Matched by containment rather than by an exact base, because the address code holds is
/// not always the object's first byte. The Itanium ABI's vptr points sixteen bytes into the vtable,
/// past the offset-to-top and typeinfo words, so a dispatch materializing a vptr names the table
/// from the middle. The callee set is the whole table either way -- the slot index is not tracked
/// -- so widening the match cannot admit a callee that an exact match would have excluded.
[[nodiscard]] std::optional<size_t>
table_at_address(std::span<const RelocationFunctionTable> tables, uint64_t vaddr) {
  for (size_t table_index = 0; table_index < tables.size(); ++table_index) {
    const RelocationFunctionTable &table = tables[table_index];
    if (vaddr >= table.table_vaddr && vaddr - table.table_vaddr < table.table_size)
      return table_index;
  }
  return std::nullopt;
}

void transfer_instruction(PairState &state, const Instruction &inst,
                          std::span<const RelocationFunctionTable> tables, uint64_t text_vaddr,
                          std::vector<RelocationTableDispatch> *dispatches,
                          std::vector<PcRelativeAddressBuilder> *address_builders) {
  const std::string_view mnemonic = inst.mnemonic();
  // Only getpc can create a tracked fact from an empty state. Large linked
  // objects contain millions of unrelated instructions, so avoid decoding
  // operand register classes or constructing a full def/use set for them.
  if (state.empty() && mnemonic != "s_get_pc_i64" && mnemonic != "s_getpc_b64")
    return;
  // Report the completed address before the table check below can reclassify the pair, so a
  // builder is described the same way whether or not its target happens to be a known table.
  const auto report_address_builder = [&](const PairValue &value) {
    if (address_builders == nullptr || value.source_address_add_offset == 0)
      return;
    address_builders->push_back({.source_getpc_offset = value.source_getpc_offset,
                                 .source_address_add_offset = value.source_address_add_offset,
                                 .target_vaddr = value.value});
  };
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
        report_address_builder(*result);
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

/// @brief Shared getpc/add fixed point that collects both result kinds after convergence.
void run_pair_dataflow(std::span<const std::unique_ptr<BasicBlock>> blocks,
                       std::span<const RelocationFunctionTable> tables, uint64_t text_vaddr,
                       std::vector<RelocationTableDispatch> *dispatches,
                       std::vector<PcRelativeAddressBuilder> *address_builders);

/// @brief Whether @p vaddr falls inside a section the loader maps.
///
/// @details `SHT_NOBITS` still counts: `.bss` occupies no file bytes but is allocated and
/// writable at run time, so a relocation destined there is a real stored pointer.
[[nodiscard]] bool vaddr_in_allocated_section(std::span<const Elf64_Shdr> sections,
                                              uint64_t vaddr) {
  for (const Elf64_Shdr &section : sections) {
    if ((section.sh_flags & SHF_ALLOC) == 0 || section.sh_size == 0)
      continue;
    if (vaddr >= section.sh_addr && vaddr - section.sh_addr < section.sh_size)
      return true;
  }
  return false;
}

/// @brief Section-header index of the code object's single `.text`.
///
/// @details Symbol values are only interpretable against the section a symbol belongs to, so every
/// symbol walk here needs the index to compare `st_shndx` against. The section header table is the
/// authority: match the header whose file offset, address and size are all the ones the parsed
/// `.text` reports.
[[nodiscard]] std::optional<size_t> text_section_index(std::span<const Elf64_Shdr> sections,
                                                       const Section &text) {
  for (size_t index = 0; index < sections.size(); ++index) {
    const Elf64_Shdr &section = sections[index];
    if (section.sh_offset == text.sectionOffset() && section.sh_addr == text.vaddr() &&
        section.sh_size == text.size())
      return index;
  }
  return std::nullopt;
}

} // namespace

bool object_defines_only_kernels(const AmdGpuCodeObject &object) {
  // The suffix AMDHSA gives a kernel's descriptor object, matching amdgpu_code_object.cpp and
  // kernel_symbol.cpp rather than restating the contract in a third place.
  constexpr std::string_view kKernelDescriptorSymbolSuffix = ".kd";
  const auto *bytes = reinterpret_cast<const uint8_t *>(object.image_data());
  const std::span<const uint8_t> image(bytes, object.image_size());
  Elf64_Ehdr ehdr{};
  if (!read_object(image, 0, ehdr) || ehdr.e_shentsize != sizeof(Elf64_Shdr) ||
      !range_in_image(image, ehdr.e_shoff,
                      static_cast<uint64_t>(ehdr.e_shnum) * sizeof(Elf64_Shdr)))
    return false;

  std::vector<Elf64_Shdr> sections(ehdr.e_shnum);
  std::memcpy(sections.data(), image.data() + ehdr.e_shoff, sections.size() * sizeof(Elf64_Shdr));

  if (object.text_sections().size() != 1)
    return false;
  const Section &text = *object.text_sections().front();
  const auto text_index = text_section_index(sections, text);
  if (!text_index)
    return false;

  // One symbol's name, bounded at the end of its string table so an unterminated string cannot
  // run past it.
  const auto name_at = [&](const Elf64_Shdr &strtab, uint32_t offset) -> std::string_view {
    if (strtab.sh_type != SHT_STRTAB || !range_in_image(image, strtab.sh_offset, strtab.sh_size) ||
        offset >= strtab.sh_size)
      return {};
    const char *first = reinterpret_cast<const char *>(image.data() + strtab.sh_offset + offset);
    const size_t bound = static_cast<size_t>(strtab.sh_size - offset);
    return std::string_view(first, ::strnlen(first, bound));
  };

  // The two tables have different string tables, so gather every descriptor name across both
  // before asking whether a function has one: a function is a kernel if a descriptor for it
  // exists anywhere in the object, not merely in the table that happens to name the function.
  std::unordered_set<std::string_view> descriptor_names;
  std::vector<std::string_view> function_names;
  bool saw_symbol_table = false;
  for (const Elf64_Shdr &symtab : sections) {
    if ((symtab.sh_type != SHT_SYMTAB && symtab.sh_type != SHT_DYNSYM) ||
        symtab.sh_entsize != sizeof(Elf64_Sym) ||
        !range_in_image(image, symtab.sh_offset, symtab.sh_size))
      continue;
    if (symtab.sh_link >= sections.size())
      return false;
    const Elf64_Shdr &strtab = sections[symtab.sh_link];
    saw_symbol_table = true;
    const size_t count = symtab.sh_size / sizeof(Elf64_Sym);
    for (size_t index = 0; index < count; ++index) {
      Elf64_Sym symbol{};
      if (!read_object(image, symtab.sh_offset + index * sizeof(Elf64_Sym), symbol))
        return false;
      const std::string_view name = name_at(strtab, symbol.st_name);
      if (name.empty())
        continue;
      if (name.ends_with(kKernelDescriptorSymbolSuffix)) {
        descriptor_names.insert(name.substr(0, name.size() - kKernelDescriptorSymbolSuffix.size()));
        continue;
      }
      if (elf_symbol_type(symbol.st_info) == kElfSymbolTypeFunc && symbol.st_size != 0 &&
          symbol.st_shndx == *text_index) {
        function_names.push_back(name);
      }
    }
  }
  if (!saw_symbol_table)
    return false;

  return std::ranges::all_of(
      function_names, [&](std::string_view name) { return descriptor_names.contains(name); });
}

static std::vector<uint64_t> discover_text_function_symbol_offsets(const AmdGpuCodeObject &object,
                                                                   bool externally_resolvable_only);

std::vector<uint64_t>
discover_externally_resolvable_text_function_offsets(const AmdGpuCodeObject &object) {
  std::vector<uint64_t> offsets;
  for (uint64_t offset :
       discover_text_function_symbol_offsets(object, /*externally_resolvable_only=*/true))
    offsets.push_back(offset);
  return offsets;
}

std::vector<uint64_t> discover_text_function_symbol_offsets(const AmdGpuCodeObject &object) {
  return discover_text_function_symbol_offsets(object, /*externally_resolvable_only=*/false);
}

static std::vector<uint64_t>
discover_text_function_symbol_offsets(const AmdGpuCodeObject &object,
                                      bool externally_resolvable_only) {
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
  // Without the section index a symbol value cannot be interpreted. Refuse to guess: returning
  // nothing costs the caller its extra split points, while guessing would invent them.
  const auto text_index = text_section_index(sections, text);
  if (!text_index)
    return {};

  std::vector<uint64_t> offsets;
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
      if (elf_symbol_type(symbol.st_info) != kElfSymbolTypeFunc || symbol.st_size == 0)
        continue;
      if (externally_resolvable_only &&
          !elf_symbol_is_externally_resolvable(symbol.st_info, symbol.st_other))
        continue;
      // The symbol has to belong to `.text` before its value means anything here. This is what
      // makes the ET_REL branch below safe: st_value is section-relative there, so a sized
      // function in some other section would otherwise be read as an offset near the start of
      // `.text` and split a real block. SHN_UNDEF and the other reserved indices are excluded by
      // the same test, since none of them equal the text index.
      if (symbol.st_shndx != *text_index)
        continue;
      // ET_REL symbol values are already section-relative; every other kind is a virtual address.
      uint64_t offset = symbol.st_value;
      if (ehdr.e_type != ET_REL) {
        if (symbol.st_value < text_vaddr)
          continue;
        offset = symbol.st_value - text_vaddr;
      }
      if (offset >= text_size || (offset % sizeof(uint32_t)) != 0)
        continue;
      offsets.push_back(offset);
    }
  }
  std::ranges::sort(offsets);
  offsets.erase(std::ranges::unique(offsets).begin(), offsets.end());
  return offsets;
}

std::vector<uint64_t>
discover_relative_text_addend_targets(const AmdGpuCodeObject &object,
                                      std::span<const uint64_t> function_entry_offsets) {
  const auto *bytes = reinterpret_cast<const uint8_t *>(object.image_data());
  const std::span<const uint8_t> image(bytes, object.image_size());
  Elf64_Ehdr ehdr{};
  if (!read_object(image, 0, ehdr) || ehdr.e_type != ET_DYN ||
      ehdr.e_shentsize != sizeof(Elf64_Shdr) ||
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

  std::vector<uint64_t> targets;
  for (const Elf64_Shdr &relocs : sections) {
    if (relocs.sh_type != SHT_RELA || relocs.sh_entsize != sizeof(Elf64_Rela) ||
        !range_in_image(image, relocs.sh_offset, relocs.sh_size))
      continue;
    const size_t count = relocs.sh_size / sizeof(Elf64_Rela);
    for (size_t index = 0; index < count; ++index) {
      Elf64_Rela rela{};
      if (!read_object(image, relocs.sh_offset + index * sizeof(Elf64_Rela), rela))
        continue;
      // RELATIVE64 is the symbol-less form: the loader stores load_bias + r_addend, so a nonzero
      // symbol index means some other relocation kind that reuses the type number, not a stored
      // pointer whose value this pass can predict.
      if (elf_reloc_type(rela.r_info) != R_AMDGPU_RELATIVE64 || elf_reloc_sym(rela.r_info) != 0 ||
          rela.r_addend < 0)
        continue;
      // The destination has to be memory the loader actually writes. A relocation carried only by
      // non-loaded metadata never produces a runtime code pointer, so admitting it would let the
      // code-address audit believe a dynamic transfer is accounted for when nothing holds its
      // target -- exactly the claim that permits an otherwise unresolved indirect branch.
      if (!vaddr_in_allocated_section(sections, rela.r_offset))
        continue;
      const auto addend = static_cast<uint64_t>(rela.r_addend);
      if (addend < text_vaddr)
        continue;
      const uint64_t offset = addend - text_vaddr;
      if (offset >= text_size || (offset % sizeof(uint32_t)) != 0)
        continue;
      // Only a function entry may be reported. Callers turn each target into a block leader and an
      // adopted root, and an adopted root is seeded with the ABI's architectural entry state --
      // both of which are claims about a function boundary, and neither of which is true of a
      // mid-function label a pointer happens to name. A target with no sized STT_FUNC symbol at
      // exactly that offset is therefore left out rather than guessed at, which keeps it
      // unadopted; relocate_relative_text_addends() then finds no placement for its addend and
      // refuses the object, which is the conservative outcome.
      if (!std::ranges::binary_search(function_entry_offsets, offset))
        continue;
      targets.push_back(offset);
    }
  }
  std::ranges::sort(targets);
  targets.erase(std::ranges::unique(targets).begin(), targets.end());
  return targets;
}

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
      if (!elf_relocation_place_is_allocated(ehdr, sections, relocations, rela.r_offset))
        continue;
      const uint32_t type = elf_reloc_type(rela.r_info);
      if (type == R_AMDGPU_RELATIVE64) {
        if (rela.r_addend < 0)
          continue;
        const uint64_t target = static_cast<uint64_t>(rela.r_addend);
        if (target >= text_vaddr && target - text_vaddr < text_size) {
          ObjectCandidate *candidate = containing_candidate(candidates, rela.r_offset);
          if (candidate == nullptr || ((rela.r_offset - candidate->vaddr) % sizeof(uint64_t)) != 0)
            continue;
          candidate->entries.push_back(
              {.slot_vaddr = rela.r_offset, .target_text_offset = target - text_vaddr});
          continue;
        }
        // A slot the loader initializes to the address of another candidate names that object, the
        // same way a GOT slot does. A C++ vptr is exactly this shape: an eight-byte word in a
        // statically constructed object, relocated to its class's vtable. Recording it is what lets
        // a dispatch that loads the pointer out of the object resolve to the vtable's callees;
        // without it the load produces no fact and the call is refused as unrecoverable.
        if (ObjectCandidate *pointed = containing_candidate(candidates, target))
          pointed->got_slots.push_back(rela.r_offset);
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

RelocationPairAnalysis analyze_relocation_pairs(std::span<const std::unique_ptr<BasicBlock>> blocks,
                                                std::span<const RelocationFunctionTable> tables,
                                                uint64_t text_vaddr) {
  RelocationPairAnalysis result;
  if (blocks.empty())
    return result;

  run_pair_dataflow(blocks, tables, text_vaddr, &result.dispatches, &result.address_builders);
  std::ranges::sort(result.dispatches, [](const auto &lhs, const auto &rhs) {
    if (lhs.source_call_offset != rhs.source_call_offset)
      return lhs.source_call_offset < rhs.source_call_offset;
    return lhs.table_index < rhs.table_index;
  });
  result.dispatches.erase(std::ranges::unique(result.dispatches, {},
                                              [](const RelocationTableDispatch &item) {
                                                return std::pair{item.source_call_offset,
                                                                 item.table_index};
                                              })
                              .begin(),
                          result.dispatches.end());

  std::ranges::sort(result.address_builders, {},
                    &PcRelativeAddressBuilder::source_address_add_offset);
  result.address_builders.erase(
      std::ranges::unique(result.address_builders, {},
                          &PcRelativeAddressBuilder::source_address_add_offset)
          .begin(),
      result.address_builders.end());
  return result;
}

namespace {

void run_pair_dataflow(std::span<const std::unique_ptr<BasicBlock>> blocks,
                       std::span<const RelocationFunctionTable> tables, uint64_t text_vaddr,
                       std::vector<RelocationTableDispatch> *dispatches,
                       std::vector<PcRelativeAddressBuilder> *address_builders) {
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
      transfer_instruction(next_out, inst, tables, text_vaddr, nullptr, nullptr);

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

  for (size_t i = 0; i < blocks.size(); ++i) {
    if (blocks[i] == nullptr || !initialized[i])
      continue;
    PairState state = in[i];
    for (const Instruction &inst : blocks[i]->instructions())
      transfer_instruction(state, inst, tables, text_vaddr, dispatches, address_builders);
  }
}

} // namespace

} // namespace rocjitsu
