// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/analysis/waitcheck.h"

#include "rocjitsu/analysis/def_use_chain.h"
#include "rocjitsu/base/rj_compiler.h"
#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/code/basic_block.h"
#include "rocjitsu/code/code_object.h"
#include "rocjitsu/isa/arch/amdgpu/cdna4/machine_insts.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/isa/operand.h"
#include "rocjitsu/vm/amdgpu/vgpr_msb.h"
#include "util/except.h"

RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include "hsa/AMDHSAKernelDescriptor.h"
RJ_DIAGNOSTIC_POP

#include <algorithm>
#include <array>
#include <bitset>
#include <charconv>
#include <cstring>
#include <iomanip>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <utility>

namespace rocjitsu {
namespace {

constexpr size_t kCounterCount = static_cast<size_t>(WaitCounterKind::Count);
constexpr uint32_t kMaxEncodedSgpr = 105;
using KernelDescriptor = rocr::llvm::amdhsa::kernel_descriptor_t;

static_assert(sizeof(KernelDescriptor) == 64, "AMDHSA kernel descriptor size changed");

[[nodiscard]] bool is_supported_waitcheck_arch(rj_code_arch_t arch) {
  return arch == ROCJITSU_CODE_ARCH_CDNA3 || arch == ROCJITSU_CODE_ARCH_CDNA4 ||
         arch == ROCJITSU_CODE_ARCH_RDNA3 || arch == ROCJITSU_CODE_ARCH_RDNA4 ||
         arch == ROCJITSU_CODE_ARCH_GFX1250;
}

[[nodiscard]] bool starts_with(std::string_view text, std::string_view prefix) {
  return text.size() >= prefix.size() && text.substr(0, prefix.size()) == prefix;
}

enum class WaitcntModel { LegacyNoVscnt, LegacyVscnt, SplitGfx12 };

[[nodiscard]] WaitcntModel waitcnt_model(rj_code_arch_t arch) {
  switch (arch) {
  case ROCJITSU_CODE_ARCH_CDNA3:
  case ROCJITSU_CODE_ARCH_CDNA4:
    return WaitcntModel::LegacyNoVscnt;
  case ROCJITSU_CODE_ARCH_RDNA3:
    return WaitcntModel::LegacyVscnt;
  default:
    return WaitcntModel::SplitGfx12;
  }
}

[[nodiscard]] bool uses_legacy_waitcnt(rj_code_arch_t arch) {
  return waitcnt_model(arch) != WaitcntModel::SplitGfx12;
}

[[nodiscard]] bool has_legacy_vscnt(rj_code_arch_t arch) {
  return waitcnt_model(arch) == WaitcntModel::LegacyVscnt;
}

[[nodiscard]] bool supports_expert_scheduling(rj_code_arch_t arch) {
  return arch == ROCJITSU_CODE_ARCH_RDNA4 || arch == ROCJITSU_CODE_ARCH_GFX1250;
}

[[nodiscard]] bool tracks_gfx12_sgpr_hazards(rj_code_arch_t arch) {
  return arch == ROCJITSU_CODE_ARCH_RDNA4;
}

[[nodiscard]] WaitCounterKind smem_wait_counter(rj_code_arch_t arch) {
  return uses_legacy_waitcnt(arch) ? WaitCounterKind::Ds : WaitCounterKind::Km;
}

[[nodiscard]] WaitCounterKind vmem_store_wait_counter(rj_code_arch_t arch) {
  return waitcnt_model(arch) == WaitcntModel::LegacyNoVscnt ? WaitCounterKind::Load
                                                            : WaitCounterKind::Store;
}

[[nodiscard]] WaitCounterKind image_sample_wait_counter(rj_code_arch_t arch) {
  return uses_legacy_waitcnt(arch) ? WaitCounterKind::Load : WaitCounterKind::Sample;
}

[[nodiscard]] WaitCounterKind image_bvh_wait_counter(rj_code_arch_t arch) {
  return uses_legacy_waitcnt(arch) ? WaitCounterKind::Load : WaitCounterKind::Bvh;
}

struct LegacyWaitcnt {
  uint32_t vmcnt = 0;
  uint32_t expcnt = 0;
  uint32_t lgkmcnt = 0;
};

[[nodiscard]] LegacyWaitcnt decode_legacy_waitcnt(uint32_t value) {
  return {
      (value & 0xFu) | (((value >> 14u) & 0x3u) << 4u),
      (value >> 4u) & 0x7u,
      (value >> 8u) & 0xFu,
  };
}

[[nodiscard]] LegacyWaitcnt decode_gfx11_waitcnt(uint32_t value) {
  return {
      (value >> 10u) & 0x3Fu,
      value & 0x7u,
      (value >> 4u) & 0x3Fu,
  };
}

[[nodiscard]] LegacyWaitcnt decode_legacy_waitcnt(uint32_t value, rj_code_arch_t arch) {
  return arch == ROCJITSU_CODE_ARCH_RDNA3 ? decode_gfx11_waitcnt(value)
                                          : decode_legacy_waitcnt(value);
}

[[nodiscard]] std::string wait_expression(WaitCounterKind counter, uint32_t required_count,
                                          rj_code_arch_t arch) {
  std::ostringstream os;
  if (uses_legacy_waitcnt(arch)) {
    switch (counter) {
    case WaitCounterKind::Load:
      os << "s_waitcnt vmcnt(" << required_count << ")";
      return os.str();
    case WaitCounterKind::Ds:
      os << "s_waitcnt lgkmcnt(" << required_count << ")";
      return os.str();
    case WaitCounterKind::Exp:
      os << "s_waitcnt expcnt(" << required_count << ")";
      return os.str();
    case WaitCounterKind::Store:
      if (has_legacy_vscnt(arch)) {
        os << "s_waitcnt_vscnt null, " << required_count;
        return os.str();
      }
      break;
    default:
      break;
    }
  }
  if (counter == WaitCounterKind::VmVsrc) {
    os << "s_wait_alu depctr_vm_vsrc(" << required_count << ")";
  } else {
    os << "s_wait_" << wait_counter_name(counter) << " <= " << required_count;
  }
  return os.str();
}

[[nodiscard]] bool fits_in_image(uint64_t offset, uint64_t size, size_t image_size) {
  return offset <= image_size && size <= image_size - offset;
}

[[nodiscard]] bool is_kernel_descriptor_symbol(const Elf64_Sym &sym, const char *strtab,
                                               size_t strtab_size) {
  if (sym.st_size != sizeof(KernelDescriptor))
    return false;
  if (elf_symbol_type(sym.st_info) != kElfSymbolTypeObject ||
      elf_symbol_bind(sym.st_info) != kElfSymbolBindGlobal)
    return false;
  if (strtab == nullptr || strtab_size == 0 || sym.st_name >= strtab_size)
    return false;

  const char *name = strtab + sym.st_name;
  const size_t max_len = strtab_size - sym.st_name;
  const size_t len = strnlen(name, max_len);
  return len > 3 && std::string_view(name + len - 3, 3) == ".kd";
}

[[nodiscard]] std::vector<uint64_t> kernel_entry_offsets(const CodeObject &code_object) {
  if (code_object.text_sections().size() != 1)
    return {};

  const Section &text = *code_object.text_sections().front();
  const auto *image = reinterpret_cast<const uint8_t *>(code_object.image_data());
  const size_t image_size = code_object.image_size();
  if (image == nullptr || image_size < sizeof(Elf64_Ehdr))
    return {};

  const auto *ehdr = reinterpret_cast<const Elf64_Ehdr *>(image);
  if (std::memcmp(ehdr->e_ident, EI_MAGIC, EI_MAGIC_SIZE) != 0 ||
      ehdr->e_ident[EI_CLASS] != ELFCLASS64 || ehdr->e_shentsize != sizeof(Elf64_Shdr))
    return {};
  if (!fits_in_image(ehdr->e_shoff, static_cast<uint64_t>(ehdr->e_shnum) * sizeof(Elf64_Shdr),
                     image_size))
    return {};

  const auto *shdrs = reinterpret_cast<const Elf64_Shdr *>(image + ehdr->e_shoff);
  const uint64_t text_vaddr = text.vaddr();
  const uint64_t text_size = text.size();
  if (text_size == 0)
    return {};

  std::set<uint64_t> entries;
  std::set<uint64_t> seen_descriptors;
  for (uint16_t i = 0; i < ehdr->e_shnum; ++i) {
    const Elf64_Shdr &symtab_shdr = shdrs[i];
    if (symtab_shdr.sh_type != SHT_SYMTAB && symtab_shdr.sh_type != SHT_DYNSYM)
      continue;
    if (symtab_shdr.sh_entsize != sizeof(Elf64_Sym))
      continue;
    if (!fits_in_image(symtab_shdr.sh_offset, symtab_shdr.sh_size, image_size))
      continue;
    if (symtab_shdr.sh_link >= ehdr->e_shnum)
      continue;

    const Elf64_Shdr &strtab_shdr = shdrs[symtab_shdr.sh_link];
    if (!fits_in_image(strtab_shdr.sh_offset, strtab_shdr.sh_size, image_size))
      continue;

    const char *strtab = reinterpret_cast<const char *>(image + strtab_shdr.sh_offset);
    const auto *syms = reinterpret_cast<const Elf64_Sym *>(image + symtab_shdr.sh_offset);
    const size_t symbol_count = symtab_shdr.sh_size / symtab_shdr.sh_entsize;
    for (size_t sym_index = 0; sym_index < symbol_count; ++sym_index) {
      const Elf64_Sym &sym = syms[sym_index];
      if (!is_kernel_descriptor_symbol(sym, strtab, strtab_shdr.sh_size))
        continue;
      if (sym.st_shndx >= ehdr->e_shnum)
        continue;

      const Elf64_Shdr &descriptor_section = shdrs[sym.st_shndx];
      if (sym.st_value < descriptor_section.sh_addr)
        continue;

      const uint64_t descriptor_file_offset =
          descriptor_section.sh_offset + (sym.st_value - descriptor_section.sh_addr);
      if (!fits_in_image(descriptor_file_offset, sizeof(KernelDescriptor), image_size))
        continue;
      if (!seen_descriptors.insert(descriptor_file_offset).second)
        continue;

      KernelDescriptor descriptor{};
      std::memcpy(&descriptor, image + descriptor_file_offset, sizeof(descriptor));
      const int64_t entry_vaddr_signed =
          static_cast<int64_t>(sym.st_value) + descriptor.kernel_code_entry_byte_offset;
      if (entry_vaddr_signed < 0)
        continue;

      const uint64_t entry_vaddr = static_cast<uint64_t>(entry_vaddr_signed);
      if (entry_vaddr < text_vaddr || entry_vaddr >= text_vaddr + text_size)
        continue;
      entries.insert(entry_vaddr - text_vaddr);
    }
  }

  return {entries.begin(), entries.end()};
}

enum class WaitEventKind {
  Unknown,
  VmemNoSamplerLoad,
  FlatLoad,
  VmemStore,
  FlatStore,
  Ds,
  Smem,
  Sample,
  Bvh,
  Export,
  SccWrite,
  SqMessage,
  GlobalInv,
  GlobalWb,
  LdsDirect,
};

enum class TrackedRegisterSource {
  None,
  Defs,
  Uses,
  VectorUses,
  StoreDataUses,
};

struct ClassifiedEvent {
  ClassifiedEvent(WaitCounterKind counter = WaitCounterKind::Load,
                  WaitEventKind kind = WaitEventKind::Unknown,
                  TrackedRegisterSource registers = TrackedRegisterSource::Defs,
                  bool check_uses = true, bool check_defs = true, bool check_exec_defs = false,
                  std::optional<RegisterRef> special_reg = std::nullopt,
                  std::optional<int64_t> barrier_id = std::nullopt, bool check_memory_order = false,
                  bool check_program_end = false)
      : counter(counter), kind(kind), registers(registers), check_uses(check_uses),
        check_defs(check_defs), check_exec_defs(check_exec_defs), special_reg(special_reg),
        barrier_id(barrier_id), check_memory_order(check_memory_order),
        check_program_end(check_program_end) {}

  WaitCounterKind counter = WaitCounterKind::Load;
  WaitEventKind kind = WaitEventKind::Unknown;
  TrackedRegisterSource registers = TrackedRegisterSource::Defs;
  bool check_uses = true;
  bool check_defs = true;
  bool check_exec_defs = false;
  std::optional<RegisterRef> special_reg;
  std::optional<int64_t> barrier_id;
  bool check_memory_order = false;
  bool check_program_end = false;
};

struct PendingEvent {
  WaitCounterKind counter = WaitCounterKind::Load;
  WaitEventKind kind = WaitEventKind::Unknown;
  RegisterSet regs;
  std::optional<RegisterRef> special_reg;
  std::optional<int64_t> barrier_id;
  bool check_uses = true;
  bool check_defs = true;
  bool check_exec_defs = false;
  std::string section_name;
  uint64_t section_offset = 0;
  uint64_t file_offset = 0;
  std::string instruction;
  bool check_memory_order = false;
  bool check_program_end = false;
  uint32_t min_younger = 0;

  bool operator==(const PendingEvent &) const = default;
};

struct SgprHazardProducer {
  std::string section_name;
  uint64_t section_offset = 0;
  uint64_t file_offset = 0;
  std::string instruction;

  bool operator==(const SgprHazardProducer &) const = default;
};

constexpr uint8_t kSgprHazardSalu = 1u << 0u;
constexpr uint8_t kSgprHazardValu = 1u << 1u;

struct SgprHazardState {
  std::bitset<64> tracked_pairs;
  bool tracked_vcc = false;
  std::bitset<128> salu_hazards;
  std::bitset<128> valu_hazards;
  uint8_t vcc_hazard = 0;
  std::array<std::optional<SgprHazardProducer>, 128> salu_producers;
  std::array<std::optional<SgprHazardProducer>, 128> valu_producers;
  std::optional<SgprHazardProducer> salu_vcc_producer;
  std::optional<SgprHazardProducer> valu_vcc_producer;
  uint8_t consecutive_ds_nops = 0;

  bool operator==(const SgprHazardState &) const = default;
};

struct VaVdstHazard {
  uint8_t age = 0;
  bool trans_since = false;
  SgprHazardProducer producer;

  bool operator==(const VaVdstHazard &) const = default;
};

struct VaVdstHazardState {
  std::array<std::optional<VaVdstHazard>, REGISTER_SET_MAX_VGPRS> hazards;

  bool operator==(const VaVdstHazardState &) const = default;
};

struct VgprMsbState {
  uint8_t mode = 0;
  bool known = true;

  [[nodiscard]] uint32_t for_role(amdgpu::VgprMsbRole role) const {
    switch (role) {
    case amdgpu::VgprMsbRole::Src0:
      return mode & 0x3u;
    case amdgpu::VgprMsbRole::Src1:
      return (mode >> 2u) & 0x3u;
    case amdgpu::VgprMsbRole::Src2:
      return (mode >> 4u) & 0x3u;
    case amdgpu::VgprMsbRole::Dst:
      return (mode >> 6u) & 0x3u;
    case amdgpu::VgprMsbRole::None:
      return 0;
    }
    return 0;
  }

  bool operator==(const VgprMsbState &) const = default;
};

enum class DelayAluEffect : uint8_t {
  None,
  Valu,
  Salu,
};

struct PendingDelayAlu {
  uint8_t countdown = 0;
  DelayAluEffect effect = DelayAluEffect::None;

  bool operator==(const PendingDelayAlu &) const = default;
};

struct ExpertSchedulingState {
  bool enabled = false;
  bool known = true;

  bool operator==(const ExpertSchedulingState &) const = default;
};

struct PendingState {
  std::array<std::vector<PendingEvent>, kCounterCount> pending;
  std::array<bool, kCounterCount> uncertain_order{};
  SgprHazardState sgpr_hazards;
  VaVdstHazardState va_vdst_hazards;
  VgprMsbState vgpr_msb;
  std::vector<PendingDelayAlu> delay_alu;
  ExpertSchedulingState expert_scheduling;

  bool operator==(const PendingState &) const = default;
};

struct CfgInstructionView {
  uint64_t section_offset = 0;
  const Instruction *instruction = nullptr;
};

struct CfgBlockView {
  const BasicBlock *block = nullptr;
  std::vector<CfgInstructionView> instructions;
};

struct ScalarValueConstraint {
  std::optional<int64_t> equal;
  std::set<int64_t> not_equal;
};

using ScalarConstraints = std::map<uint16_t, ScalarValueConstraint>;

struct SccPredicate {
  enum class Kind : uint8_t { EqImm };

  Kind kind = Kind::EqImm;
  uint16_t sgpr = 0;
  int64_t value = 0;
};

struct Analyzer {
  Analyzer(WaitcheckReport &report, WaitcheckOptions options)
      : report_(report), options_(options) {}

  void analyze_stream(std::span<const uint32_t> words, rj_code_arch_t arch,
                      std::string section_name, uint64_t file_offset_base) {
    auto decoder = Decoder::create(arch);
    if (!decoder) {
      report_.supported = false;
      return;
    }

    PendingState state;
    state.expert_scheduling.enabled = supports_expert_scheduling(arch);
    std::vector<uint32_t> padded(words.begin(), words.end());
    padded.resize(padded.size() + 2);

    size_t word_index = 0;
    while (word_index < words.size()) {
      std::unique_ptr<Instruction> inst;
      try {
        inst.reset(decoder->decode(&padded[word_index]));
      } catch (const util::Exception &ex) {
        set_analysis_error(section_name, word_index * sizeof(uint32_t), ex);
        return;
      }
      if (!inst || inst->size() <= 0) {
        break;
      }

      const size_t inst_words = static_cast<size_t>(inst->size()) / sizeof(uint32_t);
      if (inst_words == 0 || word_index + inst_words > words.size()) {
        break;
      }

      const auto section_offset = static_cast<uint64_t>(word_index * sizeof(uint32_t));
      const auto file_offset = file_offset_base + section_offset;
      analyze_instruction(state, *inst, section_name, section_offset, file_offset, arch, true);
      ++report_.instructions_analyzed;
      word_index += inst_words;
      if (should_stop_after_diagnostic())
        break;
    }
  }

  void analyze_cfg(std::vector<std::unique_ptr<BasicBlock>> &blocks,
                   const std::string &section_name, uint64_t file_offset_base,
                   rj_code_arch_t arch) {
    if (blocks.empty())
      return;

    std::unordered_map<const BasicBlock *, size_t> block_index;
    block_index.reserve(blocks.size());
    for (size_t i = 0; i < blocks.size(); ++i)
      block_index.emplace(blocks[i].get(), i);

    std::vector<PendingState> in(blocks.size());
    std::vector<PendingState> out(blocks.size());

    bool changed = true;
    size_t iterations = 0;
    const size_t max_iterations = blocks.size() * 8 + 32;
    while (changed && iterations++ < max_iterations) {
      changed = false;
      for (size_t i = 0; i < blocks.size(); ++i) {
        PendingState merged = merge_predecessors(*blocks[i], block_index, out);
        PendingState next_out =
            analyze_block(*blocks[i], merged, section_name, file_offset_base, arch, false);
        if (!(merged == in[i]) || !(next_out == out[i])) {
          in[i] = std::move(merged);
          out[i] = std::move(next_out);
          changed = true;
        }
      }
    }

    if (changed) {
      report_.supported = false;
      report_.analysis_error = "waitcheck CFG dataflow did not converge";
      return;
    }

    prepare_cfg_path_filter(blocks);
    for (size_t i = 0; i < blocks.size(); ++i) {
      (void)analyze_block(*blocks[i], in[i], section_name, file_offset_base, arch, true);
      if (should_stop_after_diagnostic())
        break;
    }
    clear_cfg_path_filter();
  }

private:
  [[nodiscard]] static size_t counter_index(WaitCounterKind counter) {
    return static_cast<size_t>(counter);
  }

  [[nodiscard]] static bool same_event_identity(const PendingEvent &lhs, const PendingEvent &rhs) {
    PendingEvent lhs_key = lhs;
    PendingEvent rhs_key = rhs;
    lhs_key.min_younger = 0;
    rhs_key.min_younger = 0;
    return lhs_key == rhs_key;
  }

  [[nodiscard]] static auto find_event(std::vector<PendingEvent> &events,
                                       const PendingEvent &event) {
    return std::ranges::find_if(
        events, [&](const PendingEvent &existing) { return same_event_identity(existing, event); });
  }

  [[nodiscard]] static bool contains_event(const std::vector<PendingEvent> &events,
                                           const PendingEvent &event) {
    return std::ranges::any_of(
        events, [&](const PendingEvent &existing) { return same_event_identity(existing, event); });
  }

  [[nodiscard]] bool diagnostics_available() const { return !report_.diagnostics_truncated; }

  [[nodiscard]] bool should_stop_after_diagnostic() const { return report_.stopped_early; }

  void prepare_cfg_path_filter(const std::vector<std::unique_ptr<BasicBlock>> &blocks) {
    cfg_views_.clear();
    cfg_view_index_.clear();
    cfg_block_by_start_offset_.clear();
    feasible_path_cache_.clear();

    cfg_views_.reserve(blocks.size());
    for (const auto &block : blocks) {
      CfgBlockView view;
      view.block = block.get();
      uint64_t section_offset = block->start_offset();
      for (const Instruction &inst : block->instructions()) {
        view.instructions.push_back({section_offset, &inst});
        section_offset += static_cast<uint64_t>(inst.size());
      }
      cfg_view_index_.emplace(view.block, cfg_views_.size());
      cfg_block_by_start_offset_.emplace(block->start_offset(), cfg_views_.size());
      cfg_views_.push_back(std::move(view));
    }
  }

  void clear_cfg_path_filter() {
    cfg_views_.clear();
    cfg_view_index_.clear();
    cfg_block_by_start_offset_.clear();
    feasible_path_cache_.clear();
  }

  void record_diagnostic(WaitcheckDiagnostic diag) {
    ++report_.diagnostics_observed;
    if (options_.stop_after_first_diagnostic) {
      report_.stopped_early = true;
      report_.diagnostics_truncated = true;
    }
    if (report_.diagnostics.size() < options_.max_diagnostics) {
      report_.diagnostics.push_back(std::move(diag));
    } else {
      report_.diagnostics_truncated = true;
      return;
    }

    if (options_.max_diagnostics != std::numeric_limits<size_t>::max() &&
        report_.diagnostics.size() >= options_.max_diagnostics) {
      report_.diagnostics_truncated = true;
    }
  }

  static void merge_va_vdst_hazards(VaVdstHazardState &dst, const VaVdstHazardState &src) {
    for (size_t i = 0; i < dst.hazards.size(); ++i) {
      const auto &src_hazard = src.hazards[i];
      if (!src_hazard)
        continue;
      auto &dst_hazard = dst.hazards[i];
      if (!dst_hazard) {
        dst_hazard = src_hazard;
        continue;
      }
      if (*dst_hazard == *src_hazard)
        continue;

      dst_hazard->trans_since = dst_hazard->trans_since || src_hazard->trans_since;
      dst_hazard->age = dst_hazard->trans_since ? 0 : std::min(dst_hazard->age, src_hazard->age);
    }
  }

  static void merge_into(PendingState &dst, const PendingState &src) {
    for (size_t i = 0; i < kCounterCount; ++i) {
      dst.uncertain_order[i] = dst.uncertain_order[i] || src.uncertain_order[i];
      for (const auto &event : src.pending[i]) {
        auto existing = find_event(dst.pending[i], event);
        if (existing == dst.pending[i].end()) {
          dst.pending[i].push_back(event);
        } else {
          existing->min_younger = std::min(existing->min_younger, event.min_younger);
        }
      }
    }
    merge_sgpr_hazards(dst.sgpr_hazards, src.sgpr_hazards);
    merge_va_vdst_hazards(dst.va_vdst_hazards, src.va_vdst_hazards);
    for (const PendingDelayAlu &delay : src.delay_alu) {
      if (std::find(dst.delay_alu.begin(), dst.delay_alu.end(), delay) == dst.delay_alu.end())
        dst.delay_alu.push_back(delay);
    }
  }

  [[nodiscard]] static PendingState
  merge_predecessors(const BasicBlock &block,
                     const std::unordered_map<const BasicBlock *, size_t> &block_index,
                     const std::vector<PendingState> &outputs) {
    PendingState merged;
    if (block.predecessors().empty())
      return merged;

    std::array<std::optional<std::vector<PendingEvent>>, kCounterCount> first_source;
    std::optional<VgprMsbState> first_vgpr_msb;
    std::optional<ExpertSchedulingState> first_expert_scheduling;
    for (const BasicBlock *pred : block.predecessors()) {
      const auto pred_it = block_index.find(pred);
      if (pred_it == block_index.end())
        continue;
      const PendingState &pred_out = outputs[pred_it->second];
      if (!first_expert_scheduling) {
        first_expert_scheduling = pred_out.expert_scheduling;
      } else if (*first_expert_scheduling != pred_out.expert_scheduling) {
        first_expert_scheduling->known = false;
        first_expert_scheduling->enabled = true;
      }
      if (!first_vgpr_msb) {
        first_vgpr_msb = pred_out.vgpr_msb;
      } else if (*first_vgpr_msb != pred_out.vgpr_msb) {
        first_vgpr_msb->known = false;
        first_vgpr_msb->mode = 0;
      }
      for (size_t i = 0; i < kCounterCount; ++i) {
        if (!pred_out.pending[i].empty()) {
          if (!first_source[i]) {
            first_source[i] = pred_out.pending[i];
          } else if (*first_source[i] != pred_out.pending[i]) {
            merged.uncertain_order[i] = true;
          }
        }
      }
      merge_into(merged, pred_out);
    }
    if (first_vgpr_msb)
      merged.vgpr_msb = *first_vgpr_msb;
    if (first_expert_scheduling)
      merged.expert_scheduling = *first_expert_scheduling;
    return merged;
  }

  [[nodiscard]] PendingState analyze_block(BasicBlock &block, const PendingState &input,
                                           const std::string &section_name,
                                           uint64_t file_offset_base, rj_code_arch_t arch,
                                           bool emit_diagnostics) {
    PendingState state = input;
    uint64_t section_offset = block.start_offset();
    for (const Instruction &inst : block.instructions()) {
      analyze_instruction(state, inst, section_name, section_offset,
                          file_offset_base + section_offset, arch, emit_diagnostics);
      if (emit_diagnostics)
        ++report_.instructions_analyzed;
      section_offset += static_cast<uint64_t>(inst.size());
      if (emit_diagnostics && should_stop_after_diagnostic())
        break;
    }
    return state;
  }

  static void apply_wait(PendingState &state, WaitCounterKind counter, uint32_t count) {
    const size_t idx = counter_index(counter);
    auto &pending = state.pending[idx];
    if (count == 0) {
      pending.clear();
      state.uncertain_order[idx] = false;
      return;
    }
    std::erase_if(pending,
                  [count](const PendingEvent &event) { return event.min_younger >= count; });
    if (pending.empty())
      state.uncertain_order[idx] = false;
  }

  static void apply_kmcnt_wait(PendingState &state, uint32_t count) {
    if (count == 0) {
      apply_wait(state, WaitCounterKind::Km, count);
      return;
    }

    // RDNA4 scalar-memory requests may return out of order, including relative
    // to other scalar-memory requests.  A nonzero KMCNT therefore cannot prove
    // that any particular SMEM destination has retired.  Other KMCNT event
    // kinds retain the normal counter-order behavior.
    const size_t idx = counter_index(WaitCounterKind::Km);
    auto &pending = state.pending[idx];
    std::erase_if(pending, [count](const PendingEvent &event) {
      return event.kind != WaitEventKind::Smem && event.min_younger >= count;
    });
    if (pending.empty())
      state.uncertain_order[idx] = false;
  }

  [[nodiscard]] static bool vm_vsrc_event_implied_by_wait(WaitEventKind kind,
                                                          WaitCounterKind counter) {
    switch (counter) {
    case WaitCounterKind::Load:
      return kind == WaitEventKind::VmemNoSamplerLoad || kind == WaitEventKind::FlatLoad;
    case WaitCounterKind::Store:
      return kind == WaitEventKind::VmemStore || kind == WaitEventKind::FlatStore;
    case WaitCounterKind::Ds:
      return kind == WaitEventKind::Ds || kind == WaitEventKind::FlatLoad ||
             kind == WaitEventKind::FlatStore;
    case WaitCounterKind::Sample:
      return kind == WaitEventKind::Sample;
    case WaitCounterKind::Bvh:
      return kind == WaitEventKind::Bvh;
    default:
      return false;
    }
  }

  template <typename Predicate>
  static void apply_filtered_vm_vsrc_wait(PendingState &state, Predicate is_implied,
                                          uint32_t count) {
    const size_t idx = counter_index(WaitCounterKind::VmVsrc);
    auto &pending = state.pending[idx];
    if (pending.empty())
      return;

    const size_t matching = static_cast<size_t>(std::ranges::count_if(pending, is_implied));
    if (matching == 0)
      return;

    if (count == 0) {
      pending.erase(std::remove_if(pending.begin(), pending.end(), is_implied), pending.end());
      if (pending.empty())
        state.uncertain_order[idx] = false;
      return;
    }

    if (state.uncertain_order[idx] || count >= matching)
      return;

    size_t to_remove = matching - count;
    pending.erase(std::remove_if(pending.begin(), pending.end(),
                                 [&](const PendingEvent &event) {
                                   if (!is_implied(event) || to_remove == 0)
                                     return false;
                                   --to_remove;
                                   return true;
                                 }),
                  pending.end());
  }

  static void apply_implied_vm_vsrc_wait(PendingState &state, WaitCounterKind counter,
                                         uint32_t count) {
    apply_filtered_vm_vsrc_wait(
        state,
        [&](const PendingEvent &event) {
          return vm_vsrc_event_implied_by_wait(event.kind, counter);
        },
        count);
  }

  [[nodiscard]] static bool vm_vsrc_event_implied_by_xcnt(WaitEventKind kind) {
    return kind == WaitEventKind::VmemNoSamplerLoad || kind == WaitEventKind::FlatLoad ||
           kind == WaitEventKind::VmemStore || kind == WaitEventKind::FlatStore;
  }

  static void apply_xcnt_wait(PendingState &state, uint32_t count) {
    apply_filtered_vm_vsrc_wait(
        state, [](const PendingEvent &event) { return vm_vsrc_event_implied_by_xcnt(event.kind); },
        count);
  }

  static void apply_memory_wait(PendingState &state, WaitCounterKind counter, uint32_t count) {
    apply_wait(state, counter, count);
    apply_implied_vm_vsrc_wait(state, counter, count);
  }

  [[nodiscard]] static bool expert_waits_enabled(const PendingState &state) {
    return !state.expert_scheduling.known || state.expert_scheduling.enabled;
  }

  static void clear_expert_wait_state(PendingState &state) {
    state.pending[counter_index(WaitCounterKind::VmVsrc)].clear();
    state.uncertain_order[counter_index(WaitCounterKind::VmVsrc)] = false;
    clear_va_vdst_hazards(state.va_vdst_hazards);
  }

  static void clear_salu_sgpr_hazards(SgprHazardState &state) {
    state.salu_hazards.reset();
    for (auto &producer : state.salu_producers)
      producer.reset();
    state.vcc_hazard = static_cast<uint8_t>(state.vcc_hazard & ~kSgprHazardSalu);
    state.salu_vcc_producer.reset();
  }

  static void clear_valu_sgpr_hazards(SgprHazardState &state) {
    state.valu_hazards.reset();
    for (auto &producer : state.valu_producers)
      producer.reset();
  }

  static void clear_valu_vcc_hazard(SgprHazardState &state) {
    state.vcc_hazard = static_cast<uint8_t>(state.vcc_hazard & ~kSgprHazardValu);
    state.valu_vcc_producer.reset();
  }

  static void clear_all_sgpr_hazards(SgprHazardState &state) {
    clear_salu_sgpr_hazards(state);
    clear_valu_sgpr_hazards(state);
    clear_valu_vcc_hazard(state);
  }

  static void merge_lane_producers(std::array<std::optional<SgprHazardProducer>, 128> &dst,
                                   const std::array<std::optional<SgprHazardProducer>, 128> &src) {
    for (size_t i = 0; i < dst.size(); ++i) {
      if (!dst[i] && src[i])
        dst[i] = src[i];
    }
  }

  static void merge_sgpr_hazards(SgprHazardState &dst, const SgprHazardState &src) {
    dst.tracked_pairs |= src.tracked_pairs;
    dst.tracked_vcc = dst.tracked_vcc || src.tracked_vcc;
    dst.salu_hazards |= src.salu_hazards;
    dst.valu_hazards |= src.valu_hazards;
    dst.vcc_hazard |= src.vcc_hazard;
    merge_lane_producers(dst.salu_producers, src.salu_producers);
    merge_lane_producers(dst.valu_producers, src.valu_producers);
    if (!dst.salu_vcc_producer && src.salu_vcc_producer)
      dst.salu_vcc_producer = src.salu_vcc_producer;
    if (!dst.valu_vcc_producer && src.valu_vcc_producer)
      dst.valu_vcc_producer = src.valu_vcc_producer;
  }

  [[nodiscard]] static uint32_t depctr_field(uint32_t value, uint32_t shift, uint32_t width) {
    return (value >> shift) & ((1u << width) - 1u);
  }

  static void apply_sgpr_hazard_wait(SgprHazardState &state, uint32_t depctr) {
    constexpr uint32_t kDepctrSaSdstShift = 0;
    constexpr uint32_t kDepctrVaVccShift = 1;
    constexpr uint32_t kDepctrVaSdstShift = 9;
    constexpr uint32_t kDepctrSaSdstWidth = 1;
    constexpr uint32_t kDepctrVaVccWidth = 1;
    constexpr uint32_t kDepctrVaSdstWidth = 3;

    if (depctr_field(depctr, kDepctrSaSdstShift, kDepctrSaSdstWidth) == 0)
      clear_salu_sgpr_hazards(state);
    if (depctr_field(depctr, kDepctrVaVccShift, kDepctrVaVccWidth) == 0)
      clear_valu_vcc_hazard(state);
    if (depctr_field(depctr, kDepctrVaSdstShift, kDepctrVaSdstWidth) == 0)
      clear_valu_sgpr_hazards(state);
  }

  [[nodiscard]] static DelayAluEffect decode_delay_alu_effect(uint32_t instid) {
    if (instid >= 1 && instid <= 8)
      return DelayAluEffect::Valu;
    if (instid >= 9 && instid <= 11)
      return DelayAluEffect::Salu;
    return DelayAluEffect::None;
  }

  static void append_delay_alu(PendingState &state, uint32_t countdown, DelayAluEffect effect) {
    if (effect == DelayAluEffect::None)
      return;
    PendingDelayAlu delay{static_cast<uint8_t>(countdown), effect};
    if (std::find(state.delay_alu.begin(), state.delay_alu.end(), delay) == state.delay_alu.end())
      state.delay_alu.push_back(delay);
  }

  static void apply_s_delay_alu(PendingState &state, const Instruction &inst) {
    constexpr uint32_t kInstidMask = 0xfu;
    constexpr uint32_t kInstskipShift = 4;
    constexpr uint32_t kInstskipMask = 0x7u;
    constexpr uint32_t kInstid1Shift = 7;

    const Operand *op = inst.src_operand(0);
    const uint32_t value = op ? static_cast<uint32_t>(op->encoding_value()) : 0;
    append_delay_alu(state, 0, decode_delay_alu_effect(value & kInstidMask));
    append_delay_alu(state, (value >> kInstskipShift) & kInstskipMask,
                     decode_delay_alu_effect((value >> kInstid1Shift) & kInstidMask));
    state.sgpr_hazards.consecutive_ds_nops = 0;
  }

  static void apply_due_delay_alu(PendingState &state) {
    for (const PendingDelayAlu &delay : state.delay_alu) {
      if (delay.countdown != 0)
        continue;
      switch (delay.effect) {
      case DelayAluEffect::Valu:
        clear_valu_sgpr_hazards(state.sgpr_hazards);
        clear_valu_vcc_hazard(state.sgpr_hazards);
        break;
      case DelayAluEffect::Salu:
        clear_salu_sgpr_hazards(state.sgpr_hazards);
        break;
      case DelayAluEffect::None:
        break;
      }
    }
    std::erase_if(state.delay_alu,
                  [](const PendingDelayAlu &delay) { return delay.countdown == 0; });
  }

  static void advance_delay_alu(PendingState &state) {
    for (PendingDelayAlu &delay : state.delay_alu) {
      if (delay.countdown > 0)
        --delay.countdown;
    }
  }

  void apply_waitcnt(PendingState &state, const Instruction &inst, rj_code_arch_t arch) {
    const auto mnemonic = inst.mnemonic();
    if (mnemonic == "s_wait_idle") {
      VgprMsbState vgpr_msb = state.vgpr_msb;
      ExpertSchedulingState expert_scheduling = state.expert_scheduling;
      state = {};
      state.vgpr_msb = vgpr_msb;
      state.expert_scheduling = expert_scheduling;
      return;
    }

    auto src_encoding = [&](int operand_index) {
      const Operand *op = inst.src_operand(operand_index);
      return op ? static_cast<uint32_t>(op->encoding_value()) : 0;
    };
    const bool legacy_sopk_wait = mnemonic == "s_waitcnt_vscnt" || mnemonic == "s_waitcnt_vmcnt" ||
                                  mnemonic == "s_waitcnt_expcnt" || mnemonic == "s_waitcnt_lgkmcnt";
    const uint32_t value = legacy_sopk_wait ? src_encoding(1) : src_encoding(0);

    if (uses_legacy_waitcnt(arch) && mnemonic == "s_waitcnt") {
      const LegacyWaitcnt wait = decode_legacy_waitcnt(value, arch);
      apply_memory_wait(state, WaitCounterKind::Load, wait.vmcnt);
      apply_wait(state, WaitCounterKind::Exp, wait.expcnt);
      apply_memory_wait(state, WaitCounterKind::Ds, wait.lgkmcnt);
    } else if (uses_legacy_waitcnt(arch) && mnemonic == "s_waitcnt_vmcnt") {
      apply_memory_wait(state, WaitCounterKind::Load, value);
    } else if (has_legacy_vscnt(arch) && mnemonic == "s_waitcnt_vscnt") {
      apply_memory_wait(state, WaitCounterKind::Store, value);
    } else if (uses_legacy_waitcnt(arch) && mnemonic == "s_waitcnt_expcnt") {
      apply_wait(state, WaitCounterKind::Exp, value);
    } else if (uses_legacy_waitcnt(arch) && mnemonic == "s_waitcnt_lgkmcnt") {
      apply_memory_wait(state, WaitCounterKind::Ds, value);
    } else if (mnemonic == "s_wait_alu") {
      apply_sgpr_hazard_wait(state.sgpr_hazards, value);
      constexpr uint32_t kDepctrVmVsrcShift = 2;
      constexpr uint32_t kDepctrVmVsrcWidth = 3;
      apply_wait(state, WaitCounterKind::VmVsrc,
                 depctr_field(value, kDepctrVmVsrcShift, kDepctrVmVsrcWidth));
    } else if (mnemonic == "s_wait_loadcnt") {
      apply_memory_wait(state, WaitCounterKind::Load, value);
    } else if (mnemonic == "s_wait_storecnt") {
      apply_memory_wait(state, WaitCounterKind::Store, value);
    } else if (mnemonic == "s_wait_dscnt") {
      apply_memory_wait(state, WaitCounterKind::Ds, value);
    } else if (mnemonic == "s_wait_kmcnt") {
      apply_kmcnt_wait(state, value);
    } else if (mnemonic == "s_wait_samplecnt") {
      apply_memory_wait(state, WaitCounterKind::Sample, value);
    } else if (mnemonic == "s_wait_bvhcnt") {
      apply_memory_wait(state, WaitCounterKind::Bvh, value);
    } else if (mnemonic == "s_wait_expcnt") {
      apply_wait(state, WaitCounterKind::Exp, value);
    } else if (mnemonic == "s_wait_xcnt") {
      apply_xcnt_wait(state, value);
    } else if (mnemonic == "s_wait_loadcnt_dscnt") {
      apply_memory_wait(state, WaitCounterKind::Load, (value >> 4) & 0xFu);
      apply_memory_wait(state, WaitCounterKind::Ds, value & 0xFu);
    } else if (mnemonic == "s_wait_storecnt_dscnt") {
      apply_memory_wait(state, WaitCounterKind::Store, (value >> 4) & 0xFu);
      apply_memory_wait(state, WaitCounterKind::Ds, value & 0xFu);
    }
  }

  [[nodiscard]] static std::optional<int64_t> first_operand_value(const Instruction &inst) {
    const Operand *op = inst.src_operand(0);
    if (!op)
      return std::nullopt;
    return static_cast<int64_t>(static_cast<int32_t>(op->encoding_value()));
  }

  static bool apply_vgpr_msb_mode(PendingState &state, const Instruction &inst,
                                  rj_code_arch_t arch) {
    if (arch != ROCJITSU_CODE_ARCH_GFX1250 || inst.mnemonic() != "s_set_vgpr_msb")
      return false;

    const auto value = first_operand_value(inst);
    state.vgpr_msb.mode = value ? static_cast<uint8_t>(*value) : 0;
    state.vgpr_msb.known = true;
    return true;
  }

  static void apply_expert_scheduling_mode(PendingState &state, const Instruction &inst,
                                           rj_code_arch_t arch) {
    if (arch != ROCJITSU_CODE_ARCH_GFX1250 && arch != ROCJITSU_CODE_ARCH_RDNA4)
      return;
    if (inst.mnemonic() != "s_setreg_imm32_b32" && inst.mnemonic() != "s_setreg_b32")
      return;

    constexpr uint32_t kHwRegWaveSchedMode = 26;
    constexpr uint32_t kSchedModeValue = 2;
    const Operand *hwreg = inst.dst_operand(0);
    if (hwreg == nullptr)
      return;

    const uint32_t value = static_cast<uint32_t>(hwreg->encoding_value());
    const uint32_t reg_id = value & 0x3fu;
    const uint32_t offset = (value >> 6u) & 0x1fu;
    const uint32_t size = ((value >> 11u) & 0x1fu) + 1u;
    if (reg_id != kHwRegWaveSchedMode || offset >= 2)
      return;

    if (inst.mnemonic() != "s_setreg_imm32_b32" || inst.raw_encoding() == nullptr ||
        inst.size() < 2 * static_cast<int>(sizeof(uint32_t)) || offset != 0 || size < 2) {
      state.expert_scheduling.known = false;
      state.expert_scheduling.enabled = true;
      return;
    }

    state.expert_scheduling.known = true;
    state.expert_scheduling.enabled =
        ((inst.raw_encoding()[1] >> offset) & 0x3u) == kSchedModeValue;
    if (!state.expert_scheduling.enabled)
      clear_expert_wait_state(state);
  }

  [[nodiscard]] static bool is_exec_masked_def(RegisterRef ref) {
    return ref.cls == RegClass::VGPR || ref.cls == RegClass::ACC_VGPR;
  }

  static void expand_vgpr_msb_ref(RegisterSet &regs, RegisterRef ref, const Operand &op,
                                  const VgprMsbState &state, rj_code_arch_t arch) {
    if (arch != ROCJITSU_CODE_ARCH_GFX1250 || ref.cls != RegClass::VGPR ||
        op.vgpr_msb_role() == amdgpu::VgprMsbRole::None) {
      regs.expand(ref);
      return;
    }

    if (!state.known) {
      for (uint32_t high = 0; high < 4; ++high) {
        RegisterRef banked = ref;
        banked.index = static_cast<uint16_t>(ref.index + high * 256u);
        regs.expand(banked);
      }
      return;
    }

    ref.index = static_cast<uint16_t>(ref.index + state.for_role(op.vgpr_msb_role()) * 256u);
    regs.expand(ref);
  }

  static void add_def(InstDefUse &du, RegisterRef ref, const Operand &op, const VgprMsbState &state,
                      rj_code_arch_t arch) {
    expand_vgpr_msb_ref(du.defs, ref, op, state, arch);
    if (is_exec_masked_def(ref))
      du.has_exec_masked_vector_def = true;
  }

  [[nodiscard]] static bool has_enabled_scalar_address(const Instruction &inst) {
    const std::string_view mnemonic = inst.mnemonic();
    if (!starts_with(mnemonic, "global_") && !starts_with(mnemonic, "scratch_"))
      return false;

    for (int i = 1; i < inst.num_src_operands(); ++i) {
      const Operand *op = inst.src_operand(i);
      if (!op)
        continue;
      if (auto ref = op->to_register_ref(); ref && ref->cls == RegClass::SGPR)
        return true;
    }
    return false;
  }

  [[nodiscard]] static bool is_cdna4_buffer_vaddr_operand(const Instruction &inst,
                                                          int operand_index, rj_code_arch_t arch) {
    if (arch != ROCJITSU_CODE_ARCH_CDNA4)
      return false;

    const std::string_view mnemonic = inst.mnemonic();
    if (starts_with(mnemonic, "buffer_store") || starts_with(mnemonic, "tbuffer_store"))
      return operand_index == 1;
    if (starts_with(mnemonic, "buffer_") || starts_with(mnemonic, "tbuffer_"))
      return operand_index == 0;
    return false;
  }

  [[nodiscard]] static std::optional<uint8_t> cdna4_buffer_vaddr_width(const Instruction &inst) {
    if (inst.raw_encoding() == nullptr || inst.size() < 2 * static_cast<int>(sizeof(uint32_t)))
      return std::nullopt;

    constexpr uint32_t kOffenBit = 12;
    constexpr uint32_t kIdxenBit = 13;
    const uint32_t word0 = inst.raw_encoding()[0];
    const bool offen = (word0 & (1u << kOffenBit)) != 0;
    const bool idxen = (word0 & (1u << kIdxenBit)) != 0;
    if (offen && idxen)
      return 2;
    if (offen || idxen)
      return 1;
    return 0;
  }

  [[nodiscard]] static bool add_adjusted_source_use(InstDefUse &du, const Instruction &inst,
                                                    const Operand &op, int operand_index,
                                                    const VgprMsbState &state,
                                                    rj_code_arch_t arch) {
    const std::string_view mnemonic = inst.mnemonic();
    if (operand_index == 0 && starts_with(mnemonic, "scratch_") && op.encoding_value() == 0)
      return true;

    if (operand_index == 0 && starts_with(mnemonic, "s_buffer_")) {
      if (const int encoded = op.encoding_value(); encoded >= 0) {
        const uint32_t base = static_cast<uint32_t>(encoded) * 2u;
        if (base <= kMaxEncodedSgpr)
          du.uses.expand({RegClass::SGPR, static_cast<uint16_t>(base), 4});
      }
      return true;
    }

    if (is_cdna4_buffer_vaddr_operand(inst, operand_index, arch)) {
      if (const auto width = cdna4_buffer_vaddr_width(inst)) {
        if (*width == 0)
          return true;
        if (auto ref = op.to_register_ref(); ref && ref->cls == RegClass::VGPR) {
          ref->width = *width;
          expand_vgpr_msb_ref(du.uses, *ref, op, state, arch);
          return true;
        }
      }
    }

    if (arch == ROCJITSU_CODE_ARCH_CDNA4 &&
        (starts_with(mnemonic, "buffer_") || starts_with(mnemonic, "tbuffer_"))) {
      if (auto ref = op.to_register_ref(); ref && ref->cls == RegClass::SGPR && ref->width == 4) {
        if (const int encoded = op.encoding_value(); encoded >= 0) {
          const uint32_t base = static_cast<uint32_t>(encoded) * 4u;
          if (base <= kMaxEncodedSgpr)
            du.uses.expand({RegClass::SGPR, static_cast<uint16_t>(base), 4});
        }
        return true;
      }
    }

    if (operand_index == 0 && has_enabled_scalar_address(inst)) {
      if (auto ref = op.to_register_ref(); ref && ref->cls == RegClass::VGPR) {
        ref->width = 1;
        expand_vgpr_msb_ref(du.uses, *ref, op, state, arch);
        return true;
      }
    }

    return false;
  }

  [[nodiscard]] static bool is_cdna4_mubuf_lds_load(const Instruction &inst, rj_code_arch_t arch) {
    if (arch != ROCJITSU_CODE_ARCH_CDNA4 || !starts_with(inst.mnemonic(), "buffer_load") ||
        inst.raw_encoding() == nullptr || inst.size() < 2 * static_cast<int>(sizeof(uint32_t)))
      return false;

    constexpr uint32_t kCdna4MubufLdsBit = 1u << 16u;
    return (inst.raw_encoding()[0] & kCdna4MubufLdsBit) != 0;
  }

  [[nodiscard]] static std::optional<cdna4::Vop3pMfmaMachineInst>
  cdna4_vop3p_mfma_encoding(const Instruction &inst, rj_code_arch_t arch) {
    if (arch != ROCJITSU_CODE_ARCH_CDNA4 || !inst.is_mfma() || inst.raw_encoding() == nullptr ||
        inst.size() < static_cast<int>(sizeof(cdna4::Vop3pMfmaMachineInst)))
      return std::nullopt;

    cdna4::Vop3pMfmaMachineInst encoding{};
    std::memcpy(&encoding, inst.raw_encoding(), sizeof(encoding));
    constexpr uint32_t kCdna4Vop3pMfmaEncoding = 0x1a7;
    if (encoding.encoding != kCdna4Vop3pMfmaEncoding)
      return std::nullopt;
    return encoding;
  }

  static void remap_vgpr_ref_to_acc(RegisterSet &regs, const Operand *op) {
    if (op == nullptr)
      return;
    auto ref = op->to_register_ref();
    if (!ref || ref->cls != RegClass::VGPR)
      return;

    regs.erase(*ref);
    ref->cls = RegClass::ACC_VGPR;
    regs.expand(*ref);
  }

  static void adjust_cdna4_mfma_acc_cd_def_use(InstDefUse &du, const Instruction &inst,
                                               rj_code_arch_t arch) {
    const auto encoding = cdna4_vop3p_mfma_encoding(inst, arch);
    if (!encoding || encoding->acc_cd == 0)
      return;

    remap_vgpr_ref_to_acc(du.defs, inst.dst_operand(0));
    if (starts_with(inst.mnemonic(), "v_mfma_"))
      remap_vgpr_ref_to_acc(du.uses, inst.src_operand(2));
  }

  [[nodiscard]] static InstDefUse inst_def_use_for_waitcheck(const Instruction &inst,
                                                             const VgprMsbState &state,
                                                             rj_code_arch_t arch) {
    InstDefUse du(inst);
    du.defs = {};
    du.uses = {};
    du.has_exec_masked_vector_def = false;
    du.has_predicated_def = inst.flags() & PREDICATED_DEF;

    if (!is_cdna4_mubuf_lds_load(inst, arch)) {
      for (int i = 0; i < inst.num_dst_operands(); ++i) {
        const Operand *op = inst.dst_operand(i);
        if (op == nullptr)
          continue;
        if (auto ref = op->to_register_ref())
          add_def(du, *ref, *op, state, arch);
      }
    }
    inst.implicit_defs(du.defs);

    for (int i = 0; i < inst.num_src_operands(); ++i) {
      const Operand *op = inst.src_operand(i);
      if (op == nullptr)
        continue;
      if (add_adjusted_source_use(du, inst, *op, i, state, arch))
        continue;
      if (auto ref = op->to_register_ref())
        expand_vgpr_msb_ref(du.uses, *ref, *op, state, arch);
    }
    inst.implicit_uses(du.uses);
    adjust_cdna4_mfma_acc_cd_def_use(du, inst, arch);

    return du;
  }

  static void clear_matching_barrier_scc_write(PendingState &state, const Instruction &inst) {
    if (inst.mnemonic() != "s_barrier_wait")
      return;
    const auto barrier_id = first_operand_value(inst);
    if (!barrier_id)
      return;

    auto &events = state.pending[counter_index(WaitCounterKind::Km)];
    std::erase_if(events, [&](const PendingEvent &event) {
      return event.kind == WaitEventKind::SccWrite && event.barrier_id == barrier_id;
    });
  }

  [[nodiscard]] static std::optional<uint32_t> vinterp_wait_exp(const Instruction &inst) {
    if (!starts_with(inst.mnemonic(), "v_interp_") || inst.raw_encoding() == nullptr ||
        inst.size() < static_cast<int>(sizeof(uint32_t)))
      return std::nullopt;
    return (inst.raw_encoding()[0] >> 8u) & 0x7u;
  }

  [[nodiscard]] static bool is_dsdir(std::string_view mnemonic) {
    return mnemonic == "ds_param_load" || mnemonic == "ds_direct_load";
  }

  [[nodiscard]] static std::optional<uint32_t> dsdir_wait_va_vdst(const Instruction &inst) {
    if (!is_dsdir(inst.mnemonic()) || inst.raw_encoding() == nullptr ||
        inst.size() < static_cast<int>(sizeof(uint32_t)))
      return std::nullopt;
    return (inst.raw_encoding()[0] >> 16u) & 0xFu;
  }

  [[nodiscard]] static std::optional<uint32_t> dsdir_wait_vm_vsrc(const Instruction &inst) {
    const auto mnemonic = inst.mnemonic();
    if (!is_dsdir(mnemonic) || inst.raw_encoding() == nullptr ||
        inst.size() < static_cast<int>(sizeof(uint32_t)))
      return std::nullopt;
    return (inst.raw_encoding()[0] >> 23u) & 0x1u;
  }

  static void apply_embedded_waitcnt(PendingState &state, const Instruction &inst) {
    if (const auto wait_exp = vinterp_wait_exp(inst))
      apply_wait(state, WaitCounterKind::Exp, *wait_exp);
    if (const auto wait_vm_vsrc = dsdir_wait_vm_vsrc(inst); wait_vm_vsrc && *wait_vm_vsrc == 0)
      apply_wait(state, WaitCounterKind::VmVsrc, 0);
  }

  [[nodiscard]] static bool is_scalar_memory_op(std::string_view mnemonic) {
    return starts_with(mnemonic, "s_load") || starts_with(mnemonic, "s_buffer_load");
  }

  [[nodiscard]] static bool is_nonflat_vmem_op(std::string_view mnemonic) {
    return starts_with(mnemonic, "global_") || starts_with(mnemonic, "buffer_") ||
           starts_with(mnemonic, "tbuffer_") || starts_with(mnemonic, "image_");
  }

  static void apply_sgpr_hazard_memory_cull(SgprHazardState &state, std::string_view mnemonic) {
    if (!is_scalar_memory_op(mnemonic) && !is_nonflat_vmem_op(mnemonic))
      return;
    clear_all_sgpr_hazards(state);
  }

  [[nodiscard]] static bool apply_sgpr_hazard_ds_nop_cull(SgprHazardState &state,
                                                          std::string_view mnemonic) {
    constexpr uint8_t kWave32DsNopCullCount = 4;
    if (mnemonic != "ds_nop") {
      state.consecutive_ds_nops = 0;
      return false;
    }

    if (state.consecutive_ds_nops < kWave32DsNopCullCount)
      ++state.consecutive_ds_nops;
    if (state.consecutive_ds_nops >= kWave32DsNopCullCount) {
      state.tracked_pairs.reset();
      state.tracked_vcc = false;
    }
    return true;
  }

  [[nodiscard]] static bool is_scalar_alu(const Instruction &inst) {
    const auto mnemonic = inst.mnemonic();
    return starts_with(mnemonic, "s_") && !inst.is_waitcnt() && !inst.is_barrier() &&
           !inst.is_branch() && !is_scalar_memory_op(mnemonic);
  }

  [[nodiscard]] static bool is_vector_alu(const Instruction &inst) {
    const auto mnemonic = inst.mnemonic();
    return starts_with(mnemonic, "v_") && !inst.is_memory_op();
  }

  [[nodiscard]] static bool is_trans_valu(const Instruction &inst) {
    const auto mnemonic = inst.mnemonic();
    return starts_with(mnemonic, "v_exp_") || starts_with(mnemonic, "v_log_") ||
           starts_with(mnemonic, "v_rcp_") || starts_with(mnemonic, "v_rsq_") ||
           starts_with(mnemonic, "v_sqrt_") || starts_with(mnemonic, "v_sin_") ||
           starts_with(mnemonic, "v_cos_");
  }

  [[nodiscard]] static bool is_vmem_store(std::string_view mnemonic) {
    return starts_with(mnemonic, "global_store") || starts_with(mnemonic, "scratch_store") ||
           starts_with(mnemonic, "buffer_store") || starts_with(mnemonic, "tbuffer_store") ||
           starts_with(mnemonic, "image_store");
  }

  [[nodiscard]] static bool is_image_atomic(std::string_view mnemonic) {
    return starts_with(mnemonic, "image_atomic");
  }

  [[nodiscard]] static bool is_scratch_store(std::string_view mnemonic) {
    return starts_with(mnemonic, "scratch_store");
  }

  [[nodiscard]] static bool is_ds_store(std::string_view mnemonic) {
    return starts_with(mnemonic, "ds_store") || starts_with(mnemonic, "ds_write") ||
           starts_with(mnemonic, "ds_cmpstore") || starts_with(mnemonic, "ds_mskor");
  }

  [[nodiscard]] static bool is_memory_ordering_consumer(std::string_view mnemonic) {
    return starts_with(mnemonic, "global_") || starts_with(mnemonic, "flat_") ||
           starts_with(mnemonic, "scratch_") || starts_with(mnemonic, "buffer_") ||
           starts_with(mnemonic, "tbuffer_") || starts_with(mnemonic, "image_") ||
           starts_with(mnemonic, "ds_") || starts_with(mnemonic, "s_load") ||
           starts_with(mnemonic, "s_buffer_load") || mnemonic == "export";
  }

  [[nodiscard]] static bool is_va_vdst_expiring_instruction(std::string_view mnemonic) {
    const bool vmem = starts_with(mnemonic, "flat_") || starts_with(mnemonic, "global_") ||
                      starts_with(mnemonic, "scratch_") || starts_with(mnemonic, "buffer_") ||
                      starts_with(mnemonic, "tbuffer_") || starts_with(mnemonic, "image_");
    const bool ds = starts_with(mnemonic, "ds_") && !is_dsdir(mnemonic);
    return vmem || ds || mnemonic == "export";
  }

  [[nodiscard]] static bool is_program_end(std::string_view mnemonic) {
    return mnemonic == "s_endpgm" || mnemonic == "s_endpgm_saved";
  }

  [[nodiscard]] static std::vector<ClassifiedEvent> classify_events(const Instruction &inst,
                                                                    rj_code_arch_t arch) {
    std::vector<ClassifiedEvent> events;
    const bool expert = supports_expert_scheduling(arch);
    const auto mnemonic = inst.mnemonic();
    if (starts_with(mnemonic, "flat_load")) {
      events.push_back({WaitCounterKind::Load, WaitEventKind::FlatLoad});
      events.push_back({WaitCounterKind::Ds, WaitEventKind::Ds});
      if (expert)
        events.push_back({WaitCounterKind::VmVsrc, WaitEventKind::FlatLoad,
                          TrackedRegisterSource::VectorUses, false, true});
      return events;
    }

    if (mnemonic == "global_inv") {
      events.emplace_back(WaitCounterKind::Load, WaitEventKind::GlobalInv,
                          TrackedRegisterSource::None, false, false, false, std::nullopt,
                          std::nullopt, true);
      return events;
    }

    if (mnemonic == "global_wb" || mnemonic == "global_wbinv") {
      events.emplace_back(vmem_store_wait_counter(arch), WaitEventKind::GlobalWb,
                          TrackedRegisterSource::None, false, false, false, std::nullopt,
                          std::nullopt, true);
      return events;
    }

    if (is_cdna4_mubuf_lds_load(inst, arch)) {
      events.push_back({WaitCounterKind::Load, WaitEventKind::LdsDirect,
                        TrackedRegisterSource::None, false, false});
      return events;
    }

    if (starts_with(mnemonic, "global_load") || starts_with(mnemonic, "scratch_load") ||
        starts_with(mnemonic, "buffer_load") || starts_with(mnemonic, "tbuffer_load") ||
        starts_with(mnemonic, "image_load")) {
      events.push_back({WaitCounterKind::Load, WaitEventKind::VmemNoSamplerLoad});
      if (expert)
        events.push_back({WaitCounterKind::VmVsrc, WaitEventKind::VmemNoSamplerLoad,
                          TrackedRegisterSource::VectorUses, false, true});
      return events;
    }

    if (is_image_atomic(mnemonic)) {
      events.push_back({WaitCounterKind::Load, WaitEventKind::VmemNoSamplerLoad});
      if (!uses_legacy_waitcnt(arch))
        events.push_back({WaitCounterKind::Exp, WaitEventKind::VmemStore,
                          TrackedRegisterSource::StoreDataUses, false, true});
      if (expert)
        events.push_back({WaitCounterKind::VmVsrc, WaitEventKind::VmemNoSamplerLoad,
                          TrackedRegisterSource::VectorUses, false, true});
      return events;
    }

    if (starts_with(mnemonic, "flat_store")) {
      events.push_back({vmem_store_wait_counter(arch), WaitEventKind::FlatStore,
                        TrackedRegisterSource::None, false, false});
      events.push_back(
          {WaitCounterKind::Ds, WaitEventKind::Ds, TrackedRegisterSource::None, false, false});
      if (expert)
        events.push_back({WaitCounterKind::VmVsrc, WaitEventKind::FlatStore,
                          TrackedRegisterSource::VectorUses, false, true});
      return events;
    }

    if (is_vmem_store(mnemonic)) {
      events.push_back({vmem_store_wait_counter(arch), WaitEventKind::VmemStore,
                        TrackedRegisterSource::None, false, false});
      if (expert)
        events.push_back({WaitCounterKind::VmVsrc, WaitEventKind::VmemStore,
                          TrackedRegisterSource::VectorUses, false, true});
      return events;
    }

    if (starts_with(mnemonic, "ds_load") || is_ds_store(mnemonic) ||
        starts_with(mnemonic, "ds_read") || starts_with(mnemonic, "ds_bpermute") ||
        starts_with(mnemonic, "ds_permute")) {
      events.push_back({WaitCounterKind::Ds, WaitEventKind::Ds});
      if (expert)
        events.push_back({WaitCounterKind::VmVsrc, WaitEventKind::Ds,
                          TrackedRegisterSource::VectorUses, false, true});
      return events;
    }

    if (mnemonic == "ds_param_load" || mnemonic == "ds_direct_load") {
      events.push_back({WaitCounterKind::Exp, WaitEventKind::LdsDirect});
      return events;
    }

    if (starts_with(mnemonic, "s_load") || starts_with(mnemonic, "s_buffer_load")) {
      events.push_back({smem_wait_counter(arch), WaitEventKind::Smem});
      return events;
    }

    if (mnemonic == "s_sendmsg_rtn_b32" || mnemonic == "s_sendmsg_rtn_b64") {
      events.push_back({smem_wait_counter(arch), WaitEventKind::SqMessage});
      return events;
    }

    if (mnemonic == "image_msaa_load" || starts_with(mnemonic, "image_sample") ||
        starts_with(mnemonic, "image_gather")) {
      events.push_back({image_sample_wait_counter(arch), WaitEventKind::Sample});
      if (expert)
        events.push_back({WaitCounterKind::VmVsrc, WaitEventKind::Sample,
                          TrackedRegisterSource::VectorUses, false, true});
      return events;
    }

    if (starts_with(mnemonic, "image_bvh")) {
      events.push_back({image_bvh_wait_counter(arch), WaitEventKind::Bvh});
      if (expert)
        events.push_back({WaitCounterKind::VmVsrc, WaitEventKind::Bvh,
                          TrackedRegisterSource::VectorUses, false, true});
      return events;
    }

    if (mnemonic == "export") {
      events.push_back({WaitCounterKind::Exp, WaitEventKind::Export, TrackedRegisterSource::Uses,
                        false, true, true});
      return events;
    }

    if (mnemonic == "s_barrier_signal_isfirst") {
      if (!uses_legacy_waitcnt(arch))
        events.push_back({WaitCounterKind::Km, WaitEventKind::SccWrite, TrackedRegisterSource::None,
                          true, true, false, RegisterRef{RegClass::SCC, 0, 1},
                          first_operand_value(inst)});
      return events;
    }

    return events;
  }

  [[nodiscard]] static std::optional<RegisterRef> first_intersection(const RegisterSet &lhs,
                                                                     const RegisterSet &rhs) {
    std::optional<RegisterRef> result;
    lhs.for_each([&](RegisterRef ref) {
      if (!result && rhs.contains(ref))
        result = ref;
    });
    return result;
  }

  [[nodiscard]] static std::string reg_name(RegisterRef ref) {
    const char *prefix = "?";
    switch (ref.cls) {
    case RegClass::SGPR:
      prefix = "s";
      break;
    case RegClass::VGPR:
      prefix = "v";
      break;
    case RegClass::ACC_VGPR:
      prefix = "acc";
      break;
    case RegClass::EXEC:
      return "exec";
    case RegClass::VCC:
      return "vcc";
    case RegClass::SCC:
      return "scc";
    case RegClass::PC:
      return "memory operation";
    default:
      break;
    }
    if (ref.width <= 1)
      return std::string(prefix) + std::to_string(ref.index);
    return std::string(prefix) + "[" + std::to_string(ref.index) + ":" +
           std::to_string(ref.index + ref.width - 1) + "]";
  }

  void emit_diagnostic(const Instruction &inst, const PendingEvent &event, RegisterRef reg,
                       WaitcheckAccessKind access, uint32_t required_count, uint64_t section_offset,
                       uint64_t file_offset, rj_code_arch_t arch) {
    if (!should_emit_cfg_diagnostic(event, section_offset, arch))
      return;

    WaitcheckDiagnostic diag;
    diag.counter = event.counter;
    diag.access = access;
    diag.reg = reg;
    diag.section_name = event.section_name;
    diag.section_offset = section_offset;
    diag.file_offset = file_offset;
    diag.instruction = inst.disassemble();
    diag.producer_section_offset = event.section_offset;
    diag.producer_file_offset = event.file_offset;
    diag.producer_instruction = event.instruction;
    diag.required_count = required_count;

    std::ostringstream msg;
    msg << "missing " << wait_expression(event.counter, required_count, arch) << " before ";
    if (access == WaitcheckAccessKind::MemoryOrder) {
      msg << "memory operation";
    } else if (access == WaitcheckAccessKind::ProgramEnd) {
      msg << "program end";
    } else {
      msg << (access == WaitcheckAccessKind::Use ? "use" : "def") << " of " << reg_name(reg);
    }
    diag.message = msg.str();
    record_diagnostic(std::move(diag));
  }

  void emit_sgpr_hazard_diagnostic(const Instruction &inst,
                                   const std::optional<SgprHazardProducer> &producer,
                                   RegisterRef reg, std::string_view depctr_field,
                                   uint64_t section_offset, uint64_t file_offset) {
    WaitcheckDiagnostic diag;
    diag.counter = WaitCounterKind::Depctr;
    diag.access = WaitcheckAccessKind::Use;
    diag.reg = reg;
    diag.section_offset = section_offset;
    diag.file_offset = file_offset;
    diag.instruction = inst.disassemble();
    diag.required_count = 0;
    if (producer) {
      diag.section_name = producer->section_name;
      diag.producer_section_offset = producer->section_offset;
      diag.producer_file_offset = producer->file_offset;
      diag.producer_instruction = producer->instruction;
    }

    std::ostringstream msg;
    msg << "missing s_wait_alu " << depctr_field << "(0) before use of " << reg_name(reg);
    diag.message = msg.str();
    record_diagnostic(std::move(diag));
  }

  void emit_va_vdst_hazard_diagnostic(const Instruction &inst, const VaVdstHazard &hazard,
                                      RegisterRef reg, uint32_t encoded_wait,
                                      uint32_t required_wait, uint64_t section_offset,
                                      uint64_t file_offset) {
    WaitcheckDiagnostic diag;
    diag.counter = WaitCounterKind::VaVdst;
    diag.access = WaitcheckAccessKind::Def;
    diag.reg = reg;
    diag.section_name = hazard.producer.section_name;
    diag.section_offset = section_offset;
    diag.file_offset = file_offset;
    diag.instruction = inst.disassemble();
    diag.producer_section_offset = hazard.producer.section_offset;
    diag.producer_file_offset = hazard.producer.file_offset;
    diag.producer_instruction = hazard.producer.instruction;
    diag.required_count = required_wait;

    std::ostringstream msg;
    msg << "missing wait_va_vdst <= " << required_wait << " before def of " << reg_name(reg)
        << " (encoded wait_va_vdst=" << encoded_wait << ")";
    diag.message = msg.str();
    record_diagnostic(std::move(diag));
  }

  void check_va_vdst_hazard(const VaVdstHazardState &state, const Instruction &inst,
                            const InstDefUse &du, uint64_t section_offset, uint64_t file_offset) {
    const auto encoded_wait = dsdir_wait_va_vdst(inst);
    if (!encoded_wait)
      return;

    bool emitted = false;
    du.defs.for_each([&](RegisterRef ref) {
      if (emitted || ref.cls != RegClass::VGPR || ref.index >= state.hazards.size())
        return;
      const auto &hazard = state.hazards[ref.index];
      if (!hazard)
        return;

      const uint32_t required_wait = hazard->trans_since ? 0u : hazard->age;
      if (*encoded_wait <= required_wait)
        return;

      emit_va_vdst_hazard_diagnostic(inst, *hazard, ref, *encoded_wait, required_wait,
                                     section_offset, file_offset);
      emitted = true;
    });
  }

  static void age_va_vdst_hazards(VaVdstHazardState &state, bool trans_seen) {
    constexpr uint8_t kNoHazardWaitStates = 15;
    for (auto &hazard : state.hazards) {
      if (!hazard)
        continue;
      if (hazard->age < kNoHazardWaitStates)
        ++hazard->age;
      if (hazard->age >= kNoHazardWaitStates) {
        hazard.reset();
        continue;
      }
      hazard->trans_since = hazard->trans_since || trans_seen;
    }
  }

  static void clear_va_vdst_hazards(VaVdstHazardState &state) {
    for (auto &hazard : state.hazards)
      hazard.reset();
  }

  static void set_va_vdst_hazard_for_regs(VaVdstHazardState &state, const RegisterSet &regs,
                                          bool trans_seen, const SgprHazardProducer &producer) {
    regs.for_each([&](RegisterRef ref) {
      if (ref.cls != RegClass::VGPR || ref.index >= state.hazards.size())
        return;
      state.hazards[ref.index] = VaVdstHazard{0, trans_seen, producer};
    });
  }

  static void update_va_vdst_hazards(VaVdstHazardState &state, const Instruction &inst,
                                     const InstDefUse &du, const std::string &section_name,
                                     uint64_t section_offset, uint64_t file_offset) {
    if (is_va_vdst_expiring_instruction(inst.mnemonic())) {
      clear_va_vdst_hazards(state);
      return;
    }
    if (!is_vector_alu(inst))
      return;

    const bool trans_seen = is_trans_valu(inst);
    age_va_vdst_hazards(state, trans_seen);

    const SgprHazardProducer producer{section_name, section_offset, file_offset,
                                      inst.disassemble()};
    set_va_vdst_hazard_for_regs(state, du.uses, trans_seen, producer);
    set_va_vdst_hazard_for_regs(state, du.defs, trans_seen, producer);
  }

  [[nodiscard]] static bool ordered_waw(const PendingEvent &event,
                                        std::span<const ClassifiedEvent> current_events) {
    switch (event.kind) {
    case WaitEventKind::VmemNoSamplerLoad:
    case WaitEventKind::Sample:
    case WaitEventKind::Bvh:
      break;
    default:
      return false;
    }
    return std::ranges::any_of(current_events, [&](const ClassifiedEvent &current_event) {
      return event.kind == current_event.kind;
    });
  }

  [[nodiscard]] std::optional<size_t> cfg_block_index_containing(uint64_t section_offset) const {
    for (size_t i = 0; i < cfg_views_.size(); ++i) {
      const BasicBlock *block = cfg_views_[i].block;
      if (block != nullptr && section_offset >= block->start_offset() &&
          section_offset < block->end_offset())
        return i;
    }
    return std::nullopt;
  }

  [[nodiscard]] bool instruction_adds_younger_event(const Instruction &inst,
                                                    const PendingEvent &event,
                                                    rj_code_arch_t arch) const {
    const auto events = classify_events(inst, arch);
    return std::ranges::any_of(events, [&](const ClassifiedEvent &classification) {
      return classification.counter == event.counter;
    });
  }

  void apply_path_pre_dependency_waits(const Instruction &inst, const PendingEvent &event,
                                       rj_code_arch_t arch, bool &pending, uint32_t &age) {
    if (!pending)
      return;

    PendingEvent path_event = event;
    path_event.min_younger = age;
    PendingState state;
    state.pending[counter_index(event.counter)].push_back(path_event);

    if (inst.is_waitcnt()) {
      apply_waitcnt(state, inst, arch);
    } else {
      apply_embedded_waitcnt(state, inst);
    }

    auto &events = state.pending[counter_index(event.counter)];
    auto found = find_event(events, path_event);
    if (found == events.end()) {
      pending = false;
      age = 0;
      return;
    }
    age = found->min_younger;
  }

  [[nodiscard]] static std::optional<int64_t> parse_operand_immediate(const Operand *op) {
    if (op == nullptr || op->to_register_ref())
      return std::nullopt;

    std::string name = op->name();
    int base = 10;
    std::string_view text(name);
    if (text.size() > 2 && text[0] == '0' && (text[1] == 'x' || text[1] == 'X')) {
      text.remove_prefix(2);
      base = 16;
    }
    int64_t value = 0;
    auto [ptr, ec] = std::from_chars(text.data(), text.data() + text.size(), value, base);
    if (ec != std::errc{} || ptr != text.data() + text.size())
      return std::nullopt;
    return value;
  }

  [[nodiscard]] static std::optional<SccPredicate> scalar_scc_predicate(const Instruction &inst) {
    if (inst.mnemonic() != "s_cmp_eq_u32")
      return std::nullopt;
    const Operand *lhs = inst.src_operand(0);
    const Operand *rhs = inst.src_operand(1);
    if (lhs == nullptr || rhs == nullptr)
      return std::nullopt;

    auto lhs_ref = lhs->to_register_ref();
    auto rhs_imm = parse_operand_immediate(rhs);
    if (!lhs_ref || lhs_ref->cls != RegClass::SGPR || lhs_ref->width != 1 || !rhs_imm)
      return std::nullopt;

    return SccPredicate{SccPredicate::Kind::EqImm, lhs_ref->index, *rhs_imm};
  }

  [[nodiscard]] static bool is_scc_conditional_branch(std::string_view mnemonic) {
    return mnemonic == "s_cbranch_scc0" || mnemonic == "s_cbranch_scc1";
  }

  [[nodiscard]] static bool
  apply_scc_branch_constraint(ScalarConstraints &constraints,
                              const std::optional<SccPredicate> &predicate,
                              bool branch_condition_true) {
    if (!predicate)
      return true;

    switch (predicate->kind) {
    case SccPredicate::Kind::EqImm: {
      ScalarValueConstraint &constraint = constraints[predicate->sgpr];
      if (branch_condition_true) {
        if (constraint.equal && *constraint.equal != predicate->value)
          return false;
        if (constraint.not_equal.contains(predicate->value))
          return false;
        constraint.equal = predicate->value;
        return true;
      }
      if (constraint.equal && *constraint.equal == predicate->value)
        return false;
      constraint.not_equal.insert(predicate->value);
      return true;
    }
    }
    return true;
  }

  [[nodiscard]] static std::string constraints_key(const ScalarConstraints &constraints,
                                                   const std::optional<SccPredicate> &predicate) {
    std::ostringstream os;
    for (const auto &[sgpr, constraint] : constraints) {
      os << 's' << sgpr << '=';
      if (constraint.equal)
        os << *constraint.equal;
      os << '!';
      for (int64_t value : constraint.not_equal)
        os << value << ',';
      os << ';';
    }
    if (predicate)
      os << "scc:eq:s" << predicate->sgpr << ':' << predicate->value;
    return os.str();
  }

  static void invalidate_redefined_scalar_constraints(ScalarConstraints &constraints,
                                                      const InstDefUse &du) {
    std::vector<uint16_t> redefined;
    du.defs.for_each([&](RegisterRef ref) {
      if (ref.cls == RegClass::SGPR)
        redefined.push_back(ref.index);
    });
    for (uint16_t sgpr : redefined)
      constraints.erase(sgpr);
  }

  [[nodiscard]] bool has_cfg_path_with_event_pending(const PendingEvent &event,
                                                     uint64_t consumer_offset,
                                                     rj_code_arch_t arch) {
    if (cfg_views_.empty())
      return true;

    using CacheKey = std::tuple<uint64_t, uint64_t, WaitCounterKind>;
    const CacheKey cache_key{event.section_offset, consumer_offset, event.counter};
    if (auto cached = feasible_path_cache_.find(cache_key); cached != feasible_path_cache_.end())
      return cached->second;

    const auto producer_block_index = cfg_block_index_containing(event.section_offset);
    const auto consumer_block_index = cfg_block_index_containing(consumer_offset);
    if (!producer_block_index) {
      feasible_path_cache_[cache_key] = true;
      return true;
    }
    if (!consumer_block_index) {
      feasible_path_cache_[cache_key] = true;
      return true;
    }
    std::set<size_t> can_reach_consumer;
    std::vector<size_t> reverse_worklist{*consumer_block_index};
    while (!reverse_worklist.empty()) {
      const size_t block_index = reverse_worklist.back();
      reverse_worklist.pop_back();
      if (!can_reach_consumer.insert(block_index).second)
        continue;
      const BasicBlock *block = cfg_views_[block_index].block;
      if (block == nullptr)
        continue;
      for (const BasicBlock *predecessor : block->predecessors()) {
        const auto predecessor_it = cfg_view_index_.find(predecessor);
        if (predecessor_it != cfg_view_index_.end())
          reverse_worklist.push_back(predecessor_it->second);
      }
    }
    if (!can_reach_consumer.contains(*producer_block_index)) {
      feasible_path_cache_[cache_key] = false;
      return false;
    }

    std::set<size_t> can_reach_producer;
    reverse_worklist.push_back(*producer_block_index);
    while (!reverse_worklist.empty()) {
      const size_t block_index = reverse_worklist.back();
      reverse_worklist.pop_back();
      if (!can_reach_producer.insert(block_index).second)
        continue;
      const BasicBlock *block = cfg_views_[block_index].block;
      if (block == nullptr)
        continue;
      for (const BasicBlock *predecessor : block->predecessors()) {
        const auto predecessor_it = cfg_view_index_.find(predecessor);
        if (predecessor_it != cfg_view_index_.end())
          reverse_worklist.push_back(predecessor_it->second);
      }
    }

    struct SearchState {
      size_t block_index = 0;
      size_t inst_index = 0;
      bool pending = false;
      uint32_t age = 0;
      ScalarConstraints constraints;
      std::optional<SccPredicate> scc_predicate;
    };

    constexpr uint32_t kMaxTrackedAge = 255;
    constexpr size_t kMaxPathSearchStates = 200000;
    std::vector<SearchState> worklist;
    for (size_t i = 0; i < cfg_views_.size(); ++i) {
      const BasicBlock *block = cfg_views_[i].block;
      if (block != nullptr && block->predecessors().empty() && can_reach_producer.contains(i))
        worklist.push_back({i, 0, false, 0, {}, std::nullopt});
    }
    if (worklist.empty())
      worklist.push_back({*producer_block_index, 0, false, 0, {}, std::nullopt});
    std::set<std::tuple<size_t, size_t, bool, uint32_t, std::string>> visited;
    size_t states_processed = 0;

    auto can_continue_from = [&](size_t block_index, bool pending) {
      return pending ? can_reach_consumer.contains(block_index)
                     : can_reach_producer.contains(block_index);
    };

    while (!worklist.empty()) {
      SearchState state = worklist.back();
      worklist.pop_back();
      state.age = std::min(state.age, kMaxTrackedAge);
      const std::string constraint_key = constraints_key(state.constraints, state.scc_predicate);
      if (!visited
               .insert(
                   {state.block_index, state.inst_index, state.pending, state.age, constraint_key})
               .second)
        continue;
      if (++states_processed > kMaxPathSearchStates) {
        feasible_path_cache_[cache_key] = true;
        return true;
      }

      const CfgBlockView &block_view = cfg_views_[state.block_index];
      if (state.inst_index >= block_view.instructions.size()) {
        if (block_view.block == nullptr)
          continue;
        for (const BasicBlock *successor : block_view.block->successors()) {
          const auto successor_it = cfg_view_index_.find(successor);
          if (successor_it != cfg_view_index_.end() &&
              can_continue_from(successor_it->second, state.pending))
            worklist.push_back({successor_it->second, 0, state.pending, state.age,
                                state.constraints, state.scc_predicate});
        }
        continue;
      }

      const CfgInstructionView &inst_view = block_view.instructions[state.inst_index];
      const Instruction &inst = *inst_view.instruction;
      apply_path_pre_dependency_waits(inst, event, arch, state.pending, state.age);

      if (inst_view.section_offset == consumer_offset && state.pending) {
        feasible_path_cache_[cache_key] = true;
        return true;
      }

      if (inst_view.section_offset == event.section_offset) {
        state.pending = true;
        state.age = 0;
      } else if (state.pending && !inst.is_waitcnt() && is_program_end(inst.mnemonic())) {
        state.pending = false;
        state.age = 0;
      } else if (state.pending && !inst.is_waitcnt() &&
                 instruction_adds_younger_event(inst, event, arch)) {
        state.age = std::min<uint32_t>(state.age + 1, kMaxTrackedAge);
      }

      const InstDefUse du = inst_def_use_for_waitcheck(inst, VgprMsbState{}, arch);
      invalidate_redefined_scalar_constraints(state.constraints, du);

      if (const auto predicate = scalar_scc_predicate(inst)) {
        state.scc_predicate = predicate;
      } else if (is_scc_conditional_branch(inst.mnemonic())) {
        const auto branch_delta = inst.branch_offset_bytes();
        const auto fallthrough_offset =
            inst_view.section_offset + static_cast<uint64_t>(inst.size());
        if (branch_delta) {
          const int64_t target =
              static_cast<int64_t>(fallthrough_offset) + static_cast<int64_t>(*branch_delta);
          if (target >= 0) {
            const auto target_it = cfg_block_by_start_offset_.find(static_cast<uint64_t>(target));
            if (target_it != cfg_block_by_start_offset_.end() &&
                can_continue_from(target_it->second, state.pending)) {
              ScalarConstraints taken_constraints = state.constraints;
              const bool branch_takes_on_scc_true = inst.mnemonic() == "s_cbranch_scc1";
              if (apply_scc_branch_constraint(taken_constraints, state.scc_predicate,
                                              branch_takes_on_scc_true)) {
                worklist.push_back({target_it->second, 0, state.pending, state.age,
                                    std::move(taken_constraints), std::nullopt});
              }
            }
          }
        }

        ScalarConstraints fallthrough_constraints = state.constraints;
        const bool fallthrough_on_scc_true = inst.mnemonic() == "s_cbranch_scc0";
        if (apply_scc_branch_constraint(fallthrough_constraints, state.scc_predicate,
                                        fallthrough_on_scc_true)) {
          const auto fallthrough_it = cfg_block_by_start_offset_.find(fallthrough_offset);
          if (fallthrough_it != cfg_block_by_start_offset_.end()) {
            if (can_continue_from(fallthrough_it->second, state.pending)) {
              worklist.push_back({fallthrough_it->second, 0, state.pending, state.age,
                                  std::move(fallthrough_constraints), std::nullopt});
            }
          } else if (state.inst_index + 1 < block_view.instructions.size()) {
            worklist.push_back({state.block_index, state.inst_index + 1, state.pending, state.age,
                                std::move(fallthrough_constraints), std::nullopt});
          }
        }
        continue;
      } else if (instruction_defines_special(inst, RegisterRef{RegClass::SCC, 0, 1}) ||
                 instruction_uses_special(inst, RegisterRef{RegClass::SCC, 0, 1})) {
        state.scc_predicate.reset();
      }

      worklist.push_back({state.block_index, state.inst_index + 1, state.pending, state.age,
                          std::move(state.constraints), state.scc_predicate});
    }

    feasible_path_cache_[cache_key] = false;
    return false;
  }

  [[nodiscard]] bool should_emit_cfg_diagnostic(const PendingEvent &event, uint64_t consumer_offset,
                                                rj_code_arch_t arch) {
    return has_cfg_path_with_event_pending(event, consumer_offset, arch);
  }

  [[nodiscard]] static bool instruction_uses_special(const Instruction &inst, RegisterRef ref) {
    if (ref.cls != RegClass::SCC)
      return false;

    const std::string_view mnemonic = inst.mnemonic();
    return mnemonic == "s_cselect_b32" || mnemonic == "s_cselect_b64" ||
           mnemonic == "s_cbranch_scc0" || mnemonic == "s_cbranch_scc1" ||
           mnemonic == "s_addc_u32" || mnemonic == "s_subb_u32";
  }

  [[nodiscard]] static bool instruction_defines_special(const Instruction &inst, RegisterRef ref) {
    if (ref.cls != RegClass::SCC)
      return false;

    const std::string_view mnemonic = inst.mnemonic();
    return mnemonic == "s_barrier_signal_isfirst" || starts_with(mnemonic, "s_cmp_") ||
           starts_with(mnemonic, "s_bitcmp") || starts_with(mnemonic, "s_addk_co_") ||
           starts_with(mnemonic, "s_add_co_") || starts_with(mnemonic, "s_sub_co_") ||
           starts_with(mnemonic, "s_addc_") || starts_with(mnemonic, "s_subb_");
  }

  void check_dependencies(const PendingState &state, const Instruction &inst, const InstDefUse &du,
                          std::span<const ClassifiedEvent> current_events, uint64_t section_offset,
                          uint64_t file_offset, rj_code_arch_t arch) {
    for (size_t counter_idx = 0; counter_idx < state.pending.size(); ++counter_idx) {
      const auto &events = state.pending[counter_idx];
      for (size_t i = 0; i < events.size(); ++i) {
        const PendingEvent &event = events[i];
        std::optional<RegisterRef> reg;
        WaitcheckAccessKind access = WaitcheckAccessKind::Use;
        if (event.check_uses) {
          reg = first_intersection(event.regs, du.uses);
        }
        if (!reg && event.check_defs) {
          reg = first_intersection(event.regs, du.defs);
          access = WaitcheckAccessKind::Def;
        }
        if (!reg && event.special_reg) {
          if (event.check_uses && instruction_uses_special(inst, *event.special_reg)) {
            reg = event.special_reg;
            access = WaitcheckAccessKind::Use;
          } else if (event.check_defs && instruction_defines_special(inst, *event.special_reg)) {
            reg = event.special_reg;
            access = WaitcheckAccessKind::Def;
          }
        }
        if (!reg && event.check_memory_order && is_memory_ordering_consumer(inst.mnemonic())) {
          reg = RegisterRef{RegClass::PC, 0, 1};
          access = WaitcheckAccessKind::MemoryOrder;
        }
        if (!reg && event.check_program_end && is_program_end(inst.mnemonic())) {
          reg = RegisterRef{RegClass::PC, 0, 1};
          access = WaitcheckAccessKind::ProgramEnd;
        }
        if (!reg)
          continue;
        if (access == WaitcheckAccessKind::Def && ordered_waw(event, current_events))
          continue;

        const auto required_count = event.kind == WaitEventKind::Smem ? 0u : event.min_younger;
        emit_diagnostic(inst, event, *reg, access, required_count, section_offset, file_offset,
                        arch);
      }
    }

    if (!instruction_defines_exec(inst))
      return;

    for (size_t counter_idx = 0; counter_idx < state.pending.size(); ++counter_idx) {
      const auto &events = state.pending[counter_idx];
      for (size_t i = 0; i < events.size(); ++i) {
        const PendingEvent &event = events[i];
        if (!event.check_exec_defs)
          continue;
        const auto required_count = event.kind == WaitEventKind::Smem ? 0u : event.min_younger;
        emit_diagnostic(inst, event, RegisterRef{RegClass::EXEC, 0, 1}, WaitcheckAccessKind::Def,
                        required_count, section_offset, file_offset, arch);
      }
    }
  }

  [[nodiscard]] static RegisterSet
  registers_for_event(const Instruction &inst, const InstDefUse &du, ClassifiedEvent classification,
                      const VgprMsbState &state, rj_code_arch_t arch) {
    switch (classification.registers) {
    case TrackedRegisterSource::None:
      return {};
    case TrackedRegisterSource::Defs:
      return du.defs;
    case TrackedRegisterSource::Uses:
      return du.uses;
    case TrackedRegisterSource::VectorUses:
      return vector_use_registers(du);
    case TrackedRegisterSource::StoreDataUses:
      return store_data_registers(inst, state, arch);
    }
    return {};
  }

  [[nodiscard]] static RegisterSet vector_use_registers(const InstDefUse &du) {
    RegisterSet regs;
    du.uses.for_each([&](RegisterRef ref) {
      if (ref.cls == RegClass::VGPR || ref.cls == RegClass::ACC_VGPR)
        regs.expand(ref);
    });
    return regs;
  }

  [[nodiscard]] static RegisterSet
  store_data_registers(const Instruction &inst, const VgprMsbState &state, rj_code_arch_t arch) {
    RegisterSet regs;
    const auto mnemonic = inst.mnemonic();
    if (is_ds_store(mnemonic)) {
      for (int i = 0; i < inst.num_src_operands(); ++i) {
        const Operand *op = inst.src_operand(i);
        if (!op)
          continue;
        if (auto ref = op->to_register_ref())
          expand_vgpr_msb_ref(regs, *ref, *op, state, arch);
      }
      return regs;
    }

    const int data_operand_index =
        starts_with(mnemonic, "buffer_store") || starts_with(mnemonic, "tbuffer_store") ||
                starts_with(mnemonic, "image_store") || is_image_atomic(mnemonic)
            ? 0
            : 1;
    const Operand *op = inst.src_operand(data_operand_index);
    if (op) {
      if (auto ref = op->to_register_ref())
        expand_vgpr_msb_ref(regs, *ref, *op, state, arch);
    }
    return regs;
  }

  [[nodiscard]] static bool instruction_defines_exec(const Instruction &inst) {
    for (int i = 0; i < inst.num_dst_operands(); ++i) {
      const Operand *op = inst.dst_operand(i);
      if (!op)
        continue;
      const std::string name = op->name();
      if (name == "exec" || name == "exec_lo" || name == "exec_hi")
        return true;
    }

    const std::string_view mnemonic = inst.mnemonic();
    return mnemonic.find("_saveexec_") != std::string_view::npos ||
           mnemonic.find("_wrexec_") != std::string_view::npos;
  }

  [[nodiscard]] static uint16_t sgpr_pair(RegisterRef ref) {
    return static_cast<uint16_t>((ref.index >> 1u) & 0x3fu);
  }

  [[nodiscard]] static bool has_tracked_sgpr_use(const SgprHazardState &state,
                                                 const RegisterSet &uses) {
    bool result = false;
    uses.for_each([&](RegisterRef ref) {
      if (!result && ref.cls == RegClass::SGPR && state.tracked_pairs.test(sgpr_pair(ref)))
        result = true;
    });
    return result;
  }

  static void track_sgpr_uses(SgprHazardState &state, const RegisterSet &uses) {
    uses.for_each([&](RegisterRef ref) {
      if (ref.cls == RegClass::SGPR)
        state.tracked_pairs.set(sgpr_pair(ref));
    });
  }

  [[nodiscard]] static bool operand_is_vcc(const Operand *op) {
    if (op == nullptr)
      return false;
    const std::string name = op->name();
    return name == "vcc" || name == "vcc_lo" || name == "vcc_hi";
  }

  [[nodiscard]] static bool instruction_uses_vcc_explicit(const Instruction &inst) {
    for (int i = 0; i < inst.num_src_operands(); ++i) {
      if (operand_is_vcc(inst.src_operand(i)))
        return true;
    }
    return false;
  }

  [[nodiscard]] static bool instruction_defines_vcc_explicit(const Instruction &inst) {
    for (int i = 0; i < inst.num_dst_operands(); ++i) {
      if (operand_is_vcc(inst.dst_operand(i)))
        return true;
    }
    return false;
  }

  [[nodiscard]] static bool has_sgpr_def(const RegisterSet &defs) {
    bool result = false;
    defs.for_each([&](RegisterRef ref) {
      if (ref.cls == RegClass::SGPR)
        result = true;
    });
    return result;
  }

  [[nodiscard]] static bool instruction_uses_vcc_implicit(const Instruction &inst) {
    const std::string_view mnemonic = inst.mnemonic();
    constexpr uint16_t kVopdXyEncodingId = 0x32;
    constexpr uint16_t kVopdCndmaskOp = 9;
    const bool vopd_xy_uses_vcc =
        inst.encoding_id() == kVopdXyEncodingId &&
        ((inst.opcode() >> 8u) == kVopdCndmaskOp || (inst.opcode() & 0xffu) == kVopdCndmaskOp);
    return mnemonic == "v_cndmask_b32_e32" || mnemonic == "v_add_co_ci_u32_e32" ||
           mnemonic == "v_sub_co_ci_u32_e32" || mnemonic == "v_subrev_co_ci_u32_e32" ||
           vopd_xy_uses_vcc;
  }

  [[nodiscard]] static bool instruction_defines_vcc_implicit(const Instruction &inst,
                                                             const InstDefUse &du) {
    const std::string_view mnemonic = inst.mnemonic();
    if (starts_with(mnemonic, "v_cmp_") && !has_sgpr_def(du.defs))
      return true;
    return mnemonic == "v_add_co_ci_u32_e32" || mnemonic == "v_sub_co_ci_u32_e32" ||
           mnemonic == "v_subrev_co_ci_u32_e32";
  }

  [[nodiscard]] static bool instruction_uses_vcc(const Instruction &inst) {
    return instruction_uses_vcc_explicit(inst) || instruction_uses_vcc_implicit(inst);
  }

  [[nodiscard]] static bool instruction_defines_vcc(const Instruction &inst, const InstDefUse &du) {
    return instruction_defines_vcc_explicit(inst) || instruction_defines_vcc_implicit(inst, du);
  }

  static void set_sgpr_hazard(SgprHazardState &state, RegisterRef ref, bool is_valu,
                              const SgprHazardProducer &producer) {
    if (ref.cls != RegClass::SGPR || ref.index >= 128)
      return;
    if (!state.tracked_pairs.test(sgpr_pair(ref)))
      return;

    if (is_valu) {
      state.valu_hazards.set(ref.index);
      state.valu_producers[ref.index] = producer;
    } else {
      state.salu_hazards.set(ref.index);
      state.salu_producers[ref.index] = producer;
    }
  }

  static void set_vcc_hazard(SgprHazardState &state, bool is_valu,
                             const SgprHazardProducer &producer) {
    if (!state.tracked_vcc)
      return;

    state.vcc_hazard = is_valu ? kSgprHazardValu : kSgprHazardSalu;
    if (is_valu) {
      state.valu_vcc_producer = producer;
      state.salu_vcc_producer.reset();
    } else {
      state.salu_vcc_producer = producer;
      state.valu_vcc_producer.reset();
    }
  }

  void check_sgpr_hazard_uses(const SgprHazardState &state, const Instruction &inst,
                              const RegisterSet &uses, bool is_valu, uint64_t section_offset,
                              uint64_t file_offset) {
    uses.for_each([&](RegisterRef ref) {
      if (ref.cls != RegClass::SGPR || ref.index >= 128)
        return;
      if (!state.tracked_pairs.test(sgpr_pair(ref)))
        return;

      if (state.salu_hazards.test(ref.index))
        emit_sgpr_hazard_diagnostic(inst, state.salu_producers[ref.index], ref, "depctr_sa_sdst",
                                    section_offset, file_offset);
      if (is_valu && state.valu_hazards.test(ref.index))
        emit_sgpr_hazard_diagnostic(inst, state.valu_producers[ref.index], ref, "depctr_va_sdst",
                                    section_offset, file_offset);
    });
  }

  void check_vcc_hazard_use(const SgprHazardState &state, const Instruction &inst, bool is_valu,
                            uint64_t section_offset, uint64_t file_offset) {
    if (!state.tracked_vcc)
      return;

    RegisterRef vcc{RegClass::VCC, 0, 1};
    if (state.vcc_hazard & kSgprHazardSalu)
      emit_sgpr_hazard_diagnostic(inst, state.salu_vcc_producer, vcc, "depctr_sa_sdst",
                                  section_offset, file_offset);
    if (is_valu && (state.vcc_hazard & kSgprHazardValu))
      emit_sgpr_hazard_diagnostic(inst, state.valu_vcc_producer, vcc, "depctr_va_vcc",
                                  section_offset, file_offset);
  }

  void update_sgpr_hazards(SgprHazardState &state, const Instruction &inst, const InstDefUse &du,
                           const std::string &section_name, uint64_t section_offset,
                           uint64_t file_offset, bool emit_diagnostics) {
    const bool is_salu = is_scalar_alu(inst);
    const bool is_valu = is_vector_alu(inst);
    if (!is_salu && !is_valu)
      return;

    const bool uses_vcc = instruction_uses_vcc(inst);
    const bool defines_vcc = instruction_defines_vcc(inst, du);
    if (is_salu && has_tracked_sgpr_use(state, du.uses))
      clear_valu_sgpr_hazards(state);
    if (is_salu && uses_vcc)
      clear_valu_vcc_hazard(state);
    if (emit_diagnostics) {
      check_sgpr_hazard_uses(state, inst, du.uses, is_valu, section_offset, file_offset);
      if (uses_vcc)
        check_vcc_hazard_use(state, inst, is_valu, section_offset, file_offset);
    }

    if (is_valu) {
      track_sgpr_uses(state, du.uses);
      if (uses_vcc)
        state.tracked_vcc = true;
    }

    SgprHazardProducer producer{section_name, section_offset, file_offset, inst.disassemble()};
    du.defs.for_each([&](RegisterRef ref) { set_sgpr_hazard(state, ref, is_valu, producer); });
    if (defines_vcc)
      set_vcc_hazard(state, is_valu, producer);
  }

  void add_event(PendingState &state, ClassifiedEvent classification, const Instruction &inst,
                 const InstDefUse &du, std::string section_name, uint64_t section_offset,
                 uint64_t file_offset, std::string instruction, rj_code_arch_t arch,
                 bool record_stats) {
    PendingEvent event;
    event.counter = classification.counter;
    event.kind = classification.kind;
    event.regs = registers_for_event(inst, du, classification, state.vgpr_msb, arch);
    event.special_reg = classification.special_reg;
    event.barrier_id = classification.barrier_id;
    event.check_uses = classification.check_uses;
    event.check_defs = classification.check_defs;
    event.check_exec_defs = classification.check_exec_defs;
    event.section_name = std::move(section_name);
    event.section_offset = section_offset;
    event.file_offset = file_offset;
    event.instruction = std::move(instruction);
    event.check_memory_order = classification.check_memory_order;
    event.check_program_end = classification.check_program_end;
    const size_t idx = counter_index(classification.counter);
    for (PendingEvent &pending_event : state.pending[idx]) {
      if (pending_event.min_younger != std::numeric_limits<uint32_t>::max())
        ++pending_event.min_younger;
    }
    if (auto existing = find_event(state.pending[idx], event);
        existing != state.pending[idx].end()) {
      existing->min_younger = 0;
    } else {
      state.pending[idx].push_back(std::move(event));
    }
    if (record_stats)
      ++report_.memory_events_tracked;
  }

  void analyze_instruction(PendingState &state, const Instruction &inst,
                           const std::string &section_name, uint64_t section_offset,
                           uint64_t file_offset, rj_code_arch_t arch, bool emit_diagnostics) {
    const bool record_stats = emit_diagnostics;
    const bool emit_report_diagnostics = emit_diagnostics && diagnostics_available();
    if (inst.mnemonic() == "s_delay_alu") {
      apply_s_delay_alu(state, inst);
      return;
    }
    apply_due_delay_alu(state);
    auto finish_instruction = [&]() { advance_delay_alu(state); };
    if (apply_sgpr_hazard_ds_nop_cull(state.sgpr_hazards, inst.mnemonic())) {
      clear_va_vdst_hazards(state.va_vdst_hazards);
      finish_instruction();
      return;
    }
    if (inst.is_waitcnt()) {
      apply_waitcnt(state, inst, arch);
      finish_instruction();
      return;
    }
    if (apply_vgpr_msb_mode(state, inst, arch)) {
      finish_instruction();
      return;
    }
    apply_expert_scheduling_mode(state, inst, arch);
    clear_matching_barrier_scc_write(state, inst);
    if (tracks_gfx12_sgpr_hazards(arch))
      apply_sgpr_hazard_memory_cull(state.sgpr_hazards, inst.mnemonic());
    apply_embedded_waitcnt(state, inst);

    InstDefUse du = inst_def_use_for_waitcheck(inst, state.vgpr_msb, arch);
    auto events = classify_events(inst, arch);
    if (!expert_waits_enabled(state)) {
      std::erase_if(events, [](const ClassifiedEvent &event) {
        return event.counter == WaitCounterKind::VmVsrc || event.counter == WaitCounterKind::VaVdst;
      });
    }
    if (emit_report_diagnostics) {
      check_dependencies(state, inst, du, events, section_offset, file_offset, arch);
      if (expert_waits_enabled(state))
        check_va_vdst_hazard(state.va_vdst_hazards, inst, du, section_offset, file_offset);
    }
    if (tracks_gfx12_sgpr_hazards(arch)) {
      update_sgpr_hazards(state.sgpr_hazards, inst, du, section_name, section_offset, file_offset,
                          emit_report_diagnostics);
    }
    if (expert_waits_enabled(state))
      update_va_vdst_hazards(state.va_vdst_hazards, inst, du, section_name, section_offset,
                             file_offset);

    if (is_program_end(inst.mnemonic())) {
      state = {};
      return;
    }

    for (const ClassifiedEvent &event : events) {
      add_event(state, event, inst, du, section_name, section_offset, file_offset,
                inst.disassemble(), arch, record_stats);
    }
    finish_instruction();
  }

  void set_analysis_error(std::string_view section_name, uint64_t section_offset,
                          const util::Exception &ex) {
    report_.supported = false;
    std::ostringstream os;
    os << "decode failed in " << section_name << "+0x" << std::hex << section_offset << ": "
       << ex.what();
    report_.analysis_error = os.str();
  }

  WaitcheckReport &report_;
  WaitcheckOptions options_;
  std::vector<CfgBlockView> cfg_views_;
  std::unordered_map<const BasicBlock *, size_t> cfg_view_index_;
  std::unordered_map<uint64_t, size_t> cfg_block_by_start_offset_;
  std::map<std::tuple<uint64_t, uint64_t, WaitCounterKind>, bool> feasible_path_cache_;
};

} // namespace

std::string_view wait_counter_name(WaitCounterKind counter) {
  switch (counter) {
  case WaitCounterKind::Load:
    return "loadcnt";
  case WaitCounterKind::Store:
    return "storecnt";
  case WaitCounterKind::Ds:
    return "dscnt";
  case WaitCounterKind::Km:
    return "kmcnt";
  case WaitCounterKind::Sample:
    return "samplecnt";
  case WaitCounterKind::Bvh:
    return "bvhcnt";
  case WaitCounterKind::Exp:
    return "expcnt";
  case WaitCounterKind::VmVsrc:
    return "depctr_vm_vsrc";
  case WaitCounterKind::VaVdst:
    return "wait_va_vdst";
  case WaitCounterKind::Depctr:
    return "depctr";
  case WaitCounterKind::Count:
    break;
  }
  return "unknown";
}

rj_code_arch_t waitcheck_arch_for_target(rj_code_target_id_t target) {
  switch (target) {
  case ROCJITSU_CODE_TARGET_GFX942:
    return ROCJITSU_CODE_ARCH_CDNA3;
  case ROCJITSU_CODE_TARGET_GFX1100:
    return ROCJITSU_CODE_ARCH_RDNA3;
  case ROCJITSU_CODE_TARGET_GFX1200:
  case ROCJITSU_CODE_TARGET_GFX1201:
    return ROCJITSU_CODE_ARCH_RDNA4;
  case ROCJITSU_CODE_TARGET_GFX1250:
    return ROCJITSU_CODE_ARCH_GFX1250;
  case ROCJITSU_CODE_TARGET_GFX950:
    return ROCJITSU_CODE_ARCH_CDNA4;
  default:
    return ROCJITSU_CODE_ARCH_INVALID;
  }
}

WaitcheckReport analyze_waitcnts(std::span<const uint32_t> words, rj_code_arch_t arch,
                                 WaitcheckOptions options) {
  WaitcheckReport report;
  report.arch = arch;
  if (!is_supported_waitcheck_arch(arch)) {
    report.supported = false;
    report.analysis_error = "unsupported architecture";
    return report;
  }

  Analyzer analyzer(report, options);
  analyzer.analyze_stream(words, arch, ".text", 0);
  return report;
}

WaitcheckReport analyze_waitcnts(const CodeObject &code_object, rj_code_arch_t arch,
                                 WaitcheckOptions options) {
  WaitcheckReport report;
  report.arch = arch;
  if (!is_supported_waitcheck_arch(arch)) {
    report.supported = false;
    report.analysis_error = "unsupported architecture";
    return report;
  }

  Analyzer analyzer(report, options);
  auto decoder = Decoder::create(arch);
  if (!decoder) {
    report.supported = false;
    report.analysis_error = "failed to create decoder";
    return report;
  }

  const auto text_file_offset = code_object.text_sections().size() == 1
                                    ? code_object.text_sections().front()->sectionOffset()
                                    : 0;
  try {
    const auto entry_offsets = kernel_entry_offsets(code_object);
    if (entry_offsets.empty()) {
      std::vector<std::unique_ptr<BasicBlock>> blocks =
          BasicBlock::build(code_object, *decoder, arch);
      analyzer.analyze_cfg(blocks, ".text", text_file_offset, arch);
    } else if (options.stop_after_first_diagnostic) {
      for (uint64_t entry_offset : entry_offsets) {
        const std::array<uint64_t, 1> entry{entry_offset};
        std::vector<std::unique_ptr<BasicBlock>> blocks =
            BasicBlock::build_reachable(code_object, *decoder, entry);
        analyzer.analyze_cfg(blocks, ".text", text_file_offset, arch);
        if (!report.supported || report.stopped_early)
          break;
      }
    } else {
      std::vector<std::unique_ptr<BasicBlock>> blocks =
          BasicBlock::build_reachable(code_object, *decoder, entry_offsets);
      analyzer.analyze_cfg(blocks, ".text", text_file_offset, arch);
    }
  } catch (const util::Exception &ex) {
    report.supported = false;
    report.analysis_error = std::string("decode failed while building CFG: ") + ex.what();
    return report;
  }
  if (!report.supported)
    return report;
  if (report.stopped_early)
    return report;

  for (const auto &owned_section : code_object.all_sections()) {
    const Section *section = owned_section.get();
    if ((section->flags() & SHF_EXECINSTR) == 0)
      continue;
    if (std::ranges::find(code_object.text_sections(), section) !=
        code_object.text_sections().end())
      continue;
    const auto *words = reinterpret_cast<const uint32_t *>(section->data());
    const size_t word_count = section->size() / sizeof(uint32_t);
    analyzer.analyze_stream(std::span<const uint32_t>(words, word_count), arch, section->name(),
                            section->sectionOffset());
  }
  return report;
}

} // namespace rocjitsu
