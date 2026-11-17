// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/analysis/waitcheck.h"

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/code/amdhsa_metadata.h"
#include "rocjitsu/code/analysis/def_use_chain.h"
#include "rocjitsu/code/analysis/waitcheck/byte_interval.h"
#include "rocjitsu/code/analysis/waitcheck/state.h"
#include "rocjitsu/code/basic_block.h"
#include "rocjitsu/code/code_object.h"
#include "rocjitsu/code/kernel_descriptor_scan.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/encodings.h"
#include "rocjitsu/isa/arch/amdgpu/generated/shared/isa_properties.h"
#include "rocjitsu/isa/arch/amdgpu/shared/accvgpr_layout.h"
#include "rocjitsu/isa/arch/amdgpu/shared/dpp_lane.h"
#include "rocjitsu/isa/arch/amdgpu/shared/instruction_encoding.h"
#include "rocjitsu/isa/arch/amdgpu/shared/scalar_operand_selectors.h"
#include "rocjitsu/isa/arch/amdgpu/shared/vgpr_msb.h"
#include "rocjitsu/isa/arch/amdgpu/shared/wait_counter_policy.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/isa/operand.h"
#include "rocjitsu/isa/target_registry.h"

#include <algorithm>
#include <array>
#include <bit>
#include <bitset>
#include <charconv>
#include <cstring>
#include <deque>
#include <functional>
#include <iomanip>
#include <iterator>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <ranges>
#include <set>
#include <sstream>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace rocjitsu {
namespace {

using namespace waitcheck_detail;
[[nodiscard]] bool is_supported_waitcheck_arch(rj_code_arch_t arch) {
  return arch == ROCJITSU_CODE_ARCH_CDNA3 || arch == ROCJITSU_CODE_ARCH_CDNA4 ||
         arch == ROCJITSU_CODE_ARCH_RDNA3 || arch == ROCJITSU_CODE_ARCH_RDNA3_5 ||
         arch == ROCJITSU_CODE_ARCH_RDNA4 || arch == ROCJITSU_CODE_ARCH_CDNA5;
}

[[nodiscard]] uint32_t default_wavefront_size(rj_code_arch_t arch) {
  return arch == ROCJITSU_CODE_ARCH_RDNA3 || arch == ROCJITSU_CODE_ARCH_RDNA3_5 ||
                 arch == ROCJITSU_CODE_ARCH_RDNA4 || arch == ROCJITSU_CODE_ARCH_CDNA5
             ? 32
             : 64;
}

[[nodiscard]] bool starts_with(std::string_view text, std::string_view prefix) {
  return text.size() >= prefix.size() && text.substr(0, prefix.size()) == prefix;
}

[[nodiscard]] bool tracks_gfx12_sgpr_hazards(rj_code_arch_t arch) {
  return arch == ROCJITSU_CODE_ARCH_RDNA4;
}

[[nodiscard]] bool fits_in_image(uint64_t offset, uint64_t size, size_t image_size) {
  return offset <= image_size && size <= image_size - offset;
}

[[nodiscard]] std::map<uint64_t, uint64_t>
find_waitcheck_function_sizes(const CodeObject &code_object) {
  if (code_object.text_sections().size() != 1)
    return {};

  const Section &text = *code_object.text_sections().front();
  const auto *image = reinterpret_cast<const uint8_t *>(code_object.image_data());
  const size_t image_size = code_object.image_size();
  if (image == nullptr || image_size < sizeof(Elf64_Ehdr))
    return {};

  const auto *ehdr = reinterpret_cast<const Elf64_Ehdr *>(image);
  if (std::memcmp(ehdr->e_ident, EI_MAGIC, EI_MAGIC_SIZE) != 0 ||
      ehdr->e_ident[EI_CLASS] != ELFCLASS64 || ehdr->e_shentsize != sizeof(Elf64_Shdr) ||
      !fits_in_image(ehdr->e_shoff, static_cast<uint64_t>(ehdr->e_shnum) * sizeof(Elf64_Shdr),
                     image_size)) {
    return {};
  }

  const auto *shdrs = reinterpret_cast<const Elf64_Shdr *>(image + ehdr->e_shoff);
  const uint64_t text_vaddr = text.vaddr();
  const uint64_t text_size = text.size();
  std::map<uint64_t, uint64_t> function_sizes;
  for (uint16_t i = 0; i < ehdr->e_shnum; ++i) {
    const Elf64_Shdr &symtab_shdr = shdrs[i];
    if ((symtab_shdr.sh_type != SHT_SYMTAB && symtab_shdr.sh_type != SHT_DYNSYM) ||
        symtab_shdr.sh_entsize != sizeof(Elf64_Sym) ||
        !fits_in_image(symtab_shdr.sh_offset, symtab_shdr.sh_size, image_size)) {
      continue;
    }

    const auto *syms = reinterpret_cast<const Elf64_Sym *>(image + symtab_shdr.sh_offset);
    const size_t symbol_count = symtab_shdr.sh_size / symtab_shdr.sh_entsize;
    for (size_t sym_index = 0; sym_index < symbol_count; ++sym_index) {
      const Elf64_Sym &sym = syms[sym_index];
      if (elf_symbol_type(sym.st_info) != kElfSymbolTypeFunc || sym.st_size == 0 ||
          sym.st_shndx >= ehdr->e_shnum || (shdrs[sym.st_shndx].sh_flags & SHF_EXECINSTR) == 0 ||
          sym.st_value < text_vaddr || sym.st_value >= text_vaddr + text_size) {
        continue;
      }
      function_sizes.insert_or_assign(sym.st_value - text_vaddr, sym.st_size);
    }
  }
  return function_sizes;
}

// Older hand-written objects can supply the kernel argument extent solely in the
// AMDHSA metadata consumed by the loader. Require an unambiguous descriptor symbol.
std::map<std::string, uint32_t> metadata_kernarg_sizes(std::span<const uint8_t> image) {
  std::map<std::string, uint32_t> sizes;
  const auto document = amdhsa::MetadataDocument::parse(image);
  if (document.failed())
    return sizes;
  const auto *kernels = document.value().root().field("amdhsa.kernels");
  if (!kernels)
    return sizes;
  for (const auto &entry : kernels->array) {
    const auto *symbol = entry.field(".symbol");
    if (!symbol || symbol->kind != amdhsa::MetadataValue::Kind::String ||
        !symbol->string.ends_with(".kd"))
      continue;
    const auto name = symbol->string.substr(0, symbol->string.size() - 3);
    auto [item, unique] = sizes.try_emplace(
        std::string(name), entry.integer_field(".kernarg_segment_size").value_or(0));
    if (!unique)
      item->second = 0;
  }
  return sizes;
}

[[nodiscard]] std::vector<WaitcheckKernelInfo> find_waitcheck_kernels(const CodeObject &code_object,
                                                                      rj_code_arch_t arch) {
  if (code_object.text_sections().size() != 1)
    return {};

  const Section &text = *code_object.text_sections().front();
  const auto image = std::span(reinterpret_cast<const uint8_t *>(code_object.image_data()),
                               code_object.image_size());
  if (image.size() < sizeof(Elf64_Ehdr))
    return {};
  const auto function_sizes = find_waitcheck_function_sizes(code_object);
  std::map<std::pair<uint64_t, uint32_t>, size_t> entry_modes;
  std::vector<WaitcheckKernelInfo> kernels;
  for (auto &info : scan_kernel_descriptors(image, text.sectionOffset(), text.size(),
                                            text.sectionHeaderIndex())) {
    const auto size = function_sizes.find(info.entry_text_offset);
    auto exact = waitcheck_kernel_from_descriptor(
        std::move(info.kernel_name), info.descriptor_vaddr, info.entry_text_offset,
        size == function_sizes.end() ? 0 : size->second, arch, info.descriptor);
    const auto [it, inserted] =
        entry_modes.try_emplace({exact.entry_offset, exact.wavefront_size}, kernels.size());
    if (!inserted) {
      auto &kernel = kernels[it->second];
      kernel.alias_descriptor_vaddrs.push_back(exact.descriptor_vaddr);
      kernel.min_vgprs = std::min(kernel.min_vgprs, exact.min_vgprs);
      kernel.max_vgprs = std::max(kernel.max_vgprs, exact.max_vgprs);
      kernel.min_sgprs = std::min(kernel.min_sgprs, exact.min_sgprs);
      kernel.max_sgprs = std::max(kernel.max_sgprs, exact.max_sgprs);
      kernel.max_memory_vgprs = std::max(kernel.max_memory_vgprs, exact.max_memory_vgprs);
      kernel.max_memory_acc_vgprs =
          std::max(kernel.max_memory_acc_vgprs, exact.max_memory_acc_vgprs);
      continue;
    }
    kernels.push_back(std::move(exact));
  }
  std::ranges::sort(kernels, {}, &WaitcheckKernelInfo::entry_offset);
  return kernels;
}

[[nodiscard]] std::vector<uint64_t> find_waitcheck_function_entries(const CodeObject &code_object) {
  const auto function_sizes = find_waitcheck_function_sizes(code_object);
  std::vector<uint64_t> entries;
  entries.reserve(function_sizes.size());
  for (const auto &[entry, size] : function_sizes) {
    (void)size;
    entries.push_back(entry);
  }
  return entries;
}
// A literal can resemble an instruction, so inspect every possible aligned
// instruction start. Reject register-sourced MODE writes and any immediate
// whose original or shifted bits might set VGPR_MSB on either gfx1250 revision.
bool image_preserves_zero_vgpr_banks(std::span<const uint8_t> text) {
  if (text.size() % sizeof(uint32_t) != 0)
    return false;
  const auto word_at = [&](size_t offset) {
    uint32_t word = 0;
    std::memcpy(&word, text.data() + offset, sizeof(word));
    return word;
  };
  constexpr auto set_msb = cdna5::build_sopp(cdna5::kSSetVgprMsbSopp)[0] >> 16;
  constexpr auto setreg = cdna5::build_sopk(cdna5::kSSetregB32Sopk)[0] >> 23;
  constexpr auto setreg_imm = cdna5::build_sopk(cdna5::kSSetregImm32B32Sopk)[0] >> 23;
  for (size_t offset = 0; offset < text.size(); offset += sizeof(uint32_t)) {
    const uint32_t word = word_at(offset);
    if ((word >> 16) == set_msb && (word & 0xffu) != 0)
      return false;
    const uint32_t opcode = word >> 23;
    if (opcode != setreg && opcode != setreg_imm)
      continue;
    const auto hwreg = static_cast<uint16_t>(word);
    if (amdgpu::decode_vgpr_msb_hwreg(hwreg).id != amdgpu::MODE_HWREG)
      continue;
    if (opcode == setreg || text.size() - offset < 2 * sizeof(uint32_t))
      return false;
    const uint32_t literal = word_at(offset + sizeof(uint32_t));
    auto banks = amdgpu::unpack_vgpr_msb_banks(uint8_t{0});
    amdgpu::apply_vgpr_msb_mode_write(banks, hwreg, literal);
    if ((literal & amdgpu::VGPR_MSB_MODE_MASK) != 0 ||
        amdgpu::pack_vgpr_msb_banks(banks) != std::optional<uint8_t>{0})
      return false;
  }
  return true;
}

bool needs_indirect_call_target(const BasicBlock &block) {
  const auto *term = block.terminator();
  if (block.successor_issue() != BasicBlock::SuccessorIssue::IndirectControlFlow || !term ||
      term->mnemonic() != "s_swap_pc_i64")
    return false;
  const auto &targets = block.static_indirect_call_fixups();
  return targets.empty() || std::ranges::any_of(targets, &IndirectCallFixup::source_incomplete);
}

bool has_closed_device_transfers(const BasicBlock &block) {
  if (block.successor_issue() == BasicBlock::SuccessorIssue::None)
    return true;
  const auto *term = block.terminator();
  if (block.successor_issue() != BasicBlock::SuccessorIssue::IndirectControlFlow || !term)
    return false;
  const auto &targets = block.static_indirect_call_fixups();
  if (!targets.empty())
    return std::ranges::none_of(targets, &IndirectCallFixup::source_incomplete);
  // The ordinary context graph separately verifies that this transfer matches
  // the active incoming call's return register before following it.
  return term->mnemonic() == "s_set_pc_i64" && term->raw_encoding() &&
         (term->raw_encoding()[0] & 0xffu) == 30;
}

void retry_gfx1250_indirect_calls(const CodeObject &code_object, Decoder &decoder,
                                  std::span<const uint64_t> entry, uint32_t wavefront_size,
                                  std::vector<std::unique_ptr<BasicBlock>> &blocks) {
  if (std::ranges::none_of(blocks,
                           [](const auto &block) { return needs_indirect_call_target(*block); }) ||
      code_object.text_sections().size() != 1)
    return;
  const auto &section = *code_object.text_sections().front();
  const auto text = std::span(reinterpret_cast<const uint8_t *>(section.data()), section.size());
  if (!image_preserves_zero_vgpr_banks(text))
    return;
  const auto symbols = find_waitcheck_function_sizes(code_object);
  std::set<uint64_t> decoded;
  for (const auto &block : blocks)
    decoded.insert(block->start_offset());
  // Seed decoding of referenced symbol bodies. A seed adds no call edge. The
  // normal target analysis must still establish the value at every call site.
  constexpr size_t kMaxAdditionalFunctions = 64;
  std::set<uint64_t> candidates;
  for (const auto &block : blocks) {
    for (const auto &builder : block->static_pc_address_builders()) {
      if (!builder.resolved || builder.poisoned || builder.source_target_offset < 0)
        continue;
      const auto target = static_cast<uint64_t>(builder.source_target_offset);
      if (symbols.contains(target) && !decoded.contains(target))
        candidates.insert(target);
      if (candidates.size() > kMaxAdditionalFunctions)
        return;
    }
  }
  const std::vector<uint64_t> seeds(candidates.begin(), candidates.end());
  auto candidate = BasicBlock::build_cfg(code_object, decoder, ROCJITSU_CODE_ARCH_CDNA5,
                                         {.entries = entry,
                                          .decode_seeds = seeds,
                                          .wave_size = static_cast<uint8_t>(wavefront_size),
                                          .verify_closed_zero_bank_calls = true});
  if (candidate.succeeded() && std::ranges::all_of(candidate.value(), [](const auto &block) {
        return has_closed_device_transfers(*block);
      }))
    blocks = std::move(candidate.value());
}

constexpr size_t kBoundaryLookaheadLimit = 256;
constexpr size_t kParityLookaheadLimit = 200000;

struct CounterParityRequirement {
  PendingEvent event;
  RegisterRef reg{RegClass::VGPR, 0, 1};
  WaitcheckAccessKind access = WaitcheckAccessKind::Use;
  uint32_t required_count = 0;
  const Instruction *consumer = nullptr;
  uint64_t consumer_section_offset = 0;
  uint64_t consumer_file_offset = 0;
  bool witness_unknown = false;

  static bool can_improve(const std::optional<CounterParityRequirement> &current, uint32_t count) {
    return !current || count < current->required_count ||
           (count == current->required_count && current->witness_unknown);
  }
};

constexpr size_t kMaxParityAlternatives = 8;
struct CounterParityAlternatives {
  std::array<std::vector<CounterParityRequirement>, kCounterCount> values;
};

struct PendingWaitGroup {
  PendingState state_before;
  std::array<std::optional<uint32_t>, kCounterCount> emitted_counts;
  std::string section_name;
  uint64_t section_offset = 0;
  uint64_t file_offset = 0;
  std::string instructions;
  bool non_entry_function_prolog = false;
  uint64_t last_wait_section_offset = 0;
  std::optional<size_t> cfg_block;
  std::vector<WaitFields> joint_flat_waits{};
  std::vector<WaitFields> vm_vsrc_companions{};
  std::unique_ptr<PendingState> vm_vsrc_without_field{};
};

struct CfgInstructionView {
  uint64_t section_offset = 0;
  const Instruction *instruction = nullptr;
  // Exclusive end of a straight-line run that leaves path-search state intact.
  size_t neutral_run_end = 0;
  size_t predicate_index = std::numeric_limits<size_t>::max();
  uint64_t issued_counters = 0;
  uint64_t issued_scalar_result_counters = 0;
  uint64_t issued_scc_write_counters = 0;
  uint64_t zero_wait_counters = 0;
  std::vector<uint16_t> scalar_result_dest_sgprs;
  bool scalar_result_dest_exec = false;
  bool scalar_result_dest_vcc = false;
  bool scalar_result_dest_scc = false;
  bool scalar_result_dest_unknown = false;
  bool invalidates_scc = false;
  bool invalidates_exec = false;
  bool invalidates_vcc_condition = false;
  VgprMsbState vgpr_msb;
  bool enters_resolved_callee = false;
  std::optional<bool> vector_skip = false;
  std::optional<ExpertSchedulingState> expert_mode_write;
  bool valu_issues_with_empty_exec = false;
  bool scalar_destination_pending = false;
  std::optional<uint16_t> scratch_relative_sgpr;
};

struct PathQueryStart {
  uint64_t offset = 0;
  size_t contexts = 0;
};

struct CfgBlockView {
  const BasicBlock *block = nullptr;
  std::vector<CfgInstructionView> instructions;
  std::vector<size_t> predecessors;
  std::vector<size_t> successors;
  std::optional<size_t> exec_nop_join;
  std::optional<size_t> repeated_exec_filter_arm;
};

struct ScalarValueConstraint {
  std::optional<uint32_t> equal;
  std::set<uint32_t> not_equal;
  std::optional<uint32_t> unsigned_min;
  std::optional<uint32_t> unsigned_max;
};

using ScalarConstraints = std::map<uint16_t, ScalarValueConstraint>;

struct SccPredicate {
  enum class Kind : uint8_t { EqImm, GeUnsignedImm, LeUnsignedImm, Constant };

  Kind kind = Kind::EqImm;
  uint16_t sgpr = 0;
  uint32_t value = 0;
  bool operator==(const SccPredicate &) const = default;
};

struct MaskedSccPredicate {
  uint16_t source;
  uint32_t mask;
  uint32_t value;
  auto operator<=>(const MaskedSccPredicate &) const = default;
};

// Substitute this counter only in the physical wait group. A threshold replay
// can also tighten its encoded minimum or test that exact same value.
struct WaitRange {
  uint64_t first;
  uint64_t last;
  uint32_t count;
  std::optional<uint32_t> original_minimum = {};
  auto operator<=>(const WaitRange &) const = default;
};

struct LoopParity {
  uint16_t saved, live;
  auto operator<=>(const LoopParity &) const = default;
};

struct ParityStep {
  uint8_t pairs;
  uint8_t on_true;
  uint8_t on_false;
};

struct Analyzer : private WaitcheckStateOps {
  Analyzer(WaitcheckReport &report, WaitcheckOptions options, uint32_t elf_flags = 0,
           std::span<const uint8_t> image = {}, uint8_t hsa_abi = 0)
      : report_(report), options_(options),
        tracks_d16_halves_(tracks_d16_subregisters(report.arch, elf_flags, hsa_abi)),
        image_(image) {
    const auto &registry = options.registry.get();
    if (const auto *arch = registry.find(report.arch)) {
      const auto bound = std::ranges::find(arch->gpu_targets, elf_flags & EF_AMDGPU_MACH,
                                           &IsaGpuTargetDescription::elf_machine);
      const auto *gpu_target =
          bound != arch->gpu_targets.end() ? &*bound : registry.find_default_gpu_target(*arch);
      target_ = gpu_target ? gpu_target->public_id : ROCJITSU_CODE_TARGET_INVALID;
      setreg_vgpr_msb_fixup_ = gpu_target && gpu_target->capabilities.setreg_vgpr_msb_fixup;
      decoder_ = gpu_target ? Decoder::create(registry, gpu_target->public_id)
                            : Decoder::create(registry, report.arch);
    }
    if (!decoder_) {
      report_.supported = false;
      report_.analysis_error = "failed to create decoder";
    }
  }

  Decoder &decoder() const { return *decoder_; }

  void analyze_stream(std::span<const uint32_t> words, rj_code_arch_t arch,
                      std::string section_name, uint64_t file_offset_base) {
    current_kernel_ = nullptr;
    source_allocation_may_change_ = false;
    stream_words_ = words;
    track_sgpr_offsets_ = true;
    wavefront_size_ = default_wavefront_size(arch);

    PendingState state;
    state.expert_scheduling.enabled = amdgpu::supports_expert_scheduling(arch);
    std::optional<PendingWaitGroup> pending_wait_group;
    size_t word_index = 0;
    while (word_index < words.size()) {
      std::unique_ptr<Instruction> inst;
      {
        const auto emit_decode_error = [&](std::string_view message) {
          set_analysis_error(section_name, word_index * sizeof(uint32_t), message);
        };
        DecodeResult decoded =
            decoder_->decode_window(words.subspan(word_index), word_index * sizeof(uint32_t),
                                    DecodeErrorEmitter(emit_decode_error));
        if (decoded.failed())
          return;
        inst = std::move(decoded).value();
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
      source_allocation_may_change_ |= inst->mnemonic() == "s_alloc_vgpr";
      update_counter_parity_group(pending_wait_group, state, *inst, section_name, section_offset,
                                  file_offset, arch);
      analyze_instruction(state, *inst, section_name, section_offset, file_offset, arch, true);
      ++report_.instructions_analyzed;
      word_index += inst_words;
      if (should_stop_after_diagnostic())
        break;
    }
    finish_counter_parity_block(pending_wait_group, state, arch);
  }

  void analyze_cfg(std::vector<std::unique_ptr<BasicBlock>> &blocks,
                   const std::string &section_name, uint64_t file_offset_base, rj_code_arch_t arch,
                   uint32_t wavefront_size, uint64_t text_size,
                   std::span<const uint64_t> entry_offsets = {}, bool non_entry_function = false) {
    if (blocks.empty())
      return;
    wavefront_size_ = wavefront_size;
    path_wait_facts_.clear();
    private_flat_addresses_.clear();
    rdna3_flat_result_counters_.clear();
    scratch_prologue_cache_.clear();
    scratch_writer_cache_.clear();
    scratch_callee_cache_.clear();
    rdna3_scalar_apertures_.clear();
    rdna3_has_aperture_ = false;
    private_proof_roots_.clear();
    relative_cfg_waits_.clear();
    relative_cfg_m0_.clear();

    std::unordered_map<const BasicBlock *, size_t> block_index;
    block_index.reserve(blocks.size());
    for (size_t i = 0; i < blocks.size(); ++i)
      block_index.emplace(blocks[i].get(), i);

    // A shared helper may be called from many sites. Keep the active return
    // continuation in the analysis-node identity so an s_setpc_b64 return can
    // resume only the call that created that context. Without this expansion,
    // later calls can return into earlier, mutually exclusive continuations and
    // manufacture impossible pending-event paths.
    // Keep the callee entry as well as the return continuation. Generated
    // device helpers can recurse; re-entering an already active callee is
    // summarized at that call's continuation instead of enumerating an
    // exponential family of finite stacks up to the hard depth limit.
    // The optional third component identifies the first block executed after
    // one specific call returns. Keeping it in the node identity consumes the
    // link-register seed after that block, so a later loop backedge does not
    // manufacture another return from the call.
    // A matching s_setpc source is a return only while it still contains the
    // caller's link. Track one explicit SGPR-pair save so a helper may use the
    // link pair for a local indirect jump and restore it before returning.
    using CallFrame =
        std::tuple<size_t, uint16_t, size_t, uint64_t, bool, std::optional<uint16_t>>;
    using CallContinuationSeed = std::pair<uint16_t, uint64_t>;
    using AnalysisNodeKey =
        std::tuple<size_t, std::vector<CallFrame>, std::optional<CallContinuationSeed>>;
    constexpr size_t kMaxCallDepth = 32;
    // PyTorch's incomplete-gamma kernels have over 220,000 distinct
    // block/return-stack pairs at depth four, with under 1,500 physical
    // blocks. Bound the expansion independently of physical block count.
    constexpr size_t kCallContextNodeAllowance = 262144;
    const size_t max_analysis_nodes =
        std::max(blocks.size() * 32, blocks.size() + kCallContextNodeAllowance);

    std::map<AnalysisNodeKey, size_t> analysis_node_index;
    std::vector<AnalysisNodeKey> analysis_node_keys;
    std::vector<BasicBlock *> analysis_blocks;
    std::vector<std::vector<size_t>> cfg_successors;
    std::vector<uint8_t> conservative_sgpr_entries;
    std::vector<size_t> expansion_worklist;

    auto get_or_add_node = [&](AnalysisNodeKey key) -> std::optional<size_t> {
      if (const auto it = analysis_node_index.find(key); it != analysis_node_index.end())
        return it->second;
      if (analysis_node_keys.size() >= max_analysis_nodes)
        return std::nullopt;
      const size_t index = analysis_node_keys.size();
      analysis_node_index.emplace(key, index);
      analysis_blocks.push_back(blocks[std::get<0>(key)].get());
      analysis_node_keys.push_back(std::move(key));
      cfg_successors.emplace_back();
      conservative_sgpr_entries.push_back(0);
      expansion_worklist.push_back(index);
      return index;
    };

    auto set_call_context_limit_error = [&] {
      size_t deepest_stack = 0;
      size_t setpc_nodes = 0;
      size_t matching_return_nodes = 0;
      for (const auto &key : analysis_node_keys)
        deepest_stack = std::max(deepest_stack, std::get<1>(key).size());
      for (size_t i = 0; i < analysis_node_keys.size(); ++i) {
        const Instruction *term = analysis_blocks[i]->terminator();
        if (term == nullptr ||
            (term->mnemonic() != "s_setpc_b64" && term->mnemonic() != "s_set_pc_i64"))
          continue;
        ++setpc_nodes;
        if (!std::get<1>(analysis_node_keys[i]).empty() && term->raw_encoding() != nullptr &&
            static_cast<uint16_t>(term->raw_encoding()[0] & 0xffu) ==
                std::get<1>(std::get<1>(analysis_node_keys[i]).back()))
          ++matching_return_nodes;
      }
      report_.analysis_error = "waitcheck call-context graph exceeded node limit: nodes=" +
                               std::to_string(analysis_node_keys.size()) +
                               " blocks=" + std::to_string(blocks.size()) +
                               " deepest_call_stack=" + std::to_string(deepest_stack) +
                               " setpc_nodes=" + std::to_string(setpc_nodes) +
                               " matching_return_nodes=" + std::to_string(matching_return_nodes);
    };

    std::vector<size_t> root_blocks;
    if (entry_offsets.empty()) {
      // Symbol-less section analysis has no distinguished entry point.
      root_blocks.resize(blocks.size());
      std::iota(root_blocks.begin(), root_blocks.end(), 0);
    } else {
      // Reachable kernel CFGs must start with an empty call stack only at the
      // real entry points. Seeding every helper as another empty-stack root
      // duplicates its graph and analyzes returns without their callers.
      for (uint64_t entry_offset : entry_offsets) {
        const auto it = std::ranges::find_if(
            blocks, [&](const auto &block) { return block->start_offset() == entry_offset; });
        if (it == blocks.end()) {
          report_.supported = false;
          report_.analysis_error = "waitcheck CFG entry block is missing";
          return;
        }
        root_blocks.push_back(static_cast<size_t>(std::distance(blocks.begin(), it)));
      }
    }
    const bool rdna3 = arch == ROCJITSU_CODE_ARCH_RDNA3 || arch == ROCJITSU_CODE_ARCH_RDNA3_5;
    if (rdna3) {
      rdna3_has_aperture_ = std::ranges::any_of(blocks, [](const auto &block) {
        return std::ranges::any_of(block->instructions(), [](const Instruction &inst) {
          return rdna3_scalar_aperture(inst).has_value();
        });
      });
    }
    if (rdna3_has_aperture_ || arch == ROCJITSU_CODE_ARCH_CDNA5) {
      for (size_t root : root_blocks)
        private_proof_roots_.insert(blocks[root].get());
      for (const auto &block : blocks) {
        for (const BasicBlock::CallEdge &call : block->call_edges())
          private_proof_roots_.insert(call.callee);
      }
    }
    for (size_t i : root_blocks) {
      const auto root = get_or_add_node({i, {}, std::nullopt});
      if (!root) {
        report_.supported = false;
        set_call_context_limit_error();
        return;
      }
      if (tracks_gfx12_sgpr_hazards(arch) && (entry_offsets.empty() || non_entry_function))
        conservative_sgpr_entries[*root] = 1;
    }

    std::optional<uint64_t> missing_successor;
    for (size_t work_index = 0; work_index < expansion_worklist.size(); ++work_index) {
      const size_t node_index = expansion_worklist[work_index];
      const AnalysisNodeKey node_key = analysis_node_keys[node_index];
      const BasicBlock &block = *blocks[std::get<0>(node_key)];

      std::vector<CallFrame> call_stack_after_block = std::get<1>(node_key);
      if (!call_stack_after_block.empty()) {
        for (const Instruction &inst : block.instructions()) {
          const InstDefUse du(inst, wavefront_size);
          const bool move_pair = inst.mnemonic() == "s_mov_b64" && inst.num_dst_operands() == 1 &&
                                 inst.num_src_operands() == 1;
          const auto dst = move_pair ? inst.dst_operand(0)->to_register_ref() : std::nullopt;
          const auto src = move_pair ? inst.src_operand(0)->to_register_ref() : std::nullopt;
          const bool sgpr_pair = dst && src && dst->cls == RegClass::SGPR &&
                                 src->cls == RegClass::SGPR && dst->width == 2 && src->width == 2;
          const bool relative_sgpr_write = inst.mnemonic().starts_with("s_movreld");
          for (auto &frame : call_stack_after_block) {
            const uint16_t link = std::get<1>(frame);
            const auto saved = std::get<5>(frame);
            const bool disjoint_link = sgpr_pair &&
                                       (dst->index + 2 <= link || dst->index >= link + 2);
            const bool copy_link = disjoint_link && std::get<4>(frame) && src->index == link;
            const bool copy_saved = sgpr_pair && saved && src->index == *saved;
            if (relative_sgpr_write || du.defs.intersects(RegisterRef{RegClass::SGPR, link, 2}))
              std::get<4>(frame) = false;
            if (relative_sgpr_write ||
                (saved && du.defs.intersects(RegisterRef{RegClass::SGPR, *saved, 2})))
              std::get<5>(frame).reset();
            if (copy_link || (copy_saved && disjoint_link))
              std::get<5>(frame) = dst->index;
            if (copy_saved && dst->index == link)
              std::get<4>(frame) = true;
          }
        }
      }

      auto add_context_edge = [&](size_t target_block_index, std::vector<CallFrame> call_stack,
                                  bool conservative_sgpr_entry = false,
                                  std::optional<CallContinuationSeed> call_continuation =
                                      std::nullopt) -> bool {
        const auto target = get_or_add_node(
            {target_block_index, std::move(call_stack), std::move(call_continuation)});
        if (!target)
          return false;
        if (conservative_sgpr_entry && tracks_gfx12_sgpr_hazards(arch))
          conservative_sgpr_entries[*target] = 1;
        auto &successors = cfg_successors[node_index];
        if (std::ranges::find(successors, *target) == successors.end())
          successors.push_back(*target);
        return true;
      };

      const Instruction *term = block.terminator();
      if (!call_stack_after_block.empty() && std::get<4>(call_stack_after_block.back()) &&
          term != nullptr &&
          term->size() == static_cast<int>(sizeof(uint32_t)) &&
          (term->mnemonic() == "s_setpc_b64" || term->mnemonic() == "s_set_pc_i64") &&
          block.static_indirect_call_fixups().empty() && term->raw_encoding() != nullptr &&
          static_cast<uint16_t>(term->raw_encoding()[0] & 0xffu) ==
              std::get<1>(call_stack_after_block.back())) {
        std::vector<CallFrame> caller_stack = std::move(call_stack_after_block);
        const size_t continuation_index = std::get<0>(caller_stack.back());
        const uint16_t return_sreg = std::get<1>(caller_stack.back());
        const uint64_t source_call_offset = std::get<3>(caller_stack.back());
        caller_stack.pop_back();
        if (!add_context_edge(continuation_index, std::move(caller_stack),
                              /*conservative_sgpr_entry=*/true,
                              CallContinuationSeed{return_sreg, source_call_offset})) {
          report_.supported = false;
          set_call_context_limit_error();
          return;
        }
        continue;
      }

      if (!missing_successor && block.successor_issue() != BasicBlock::SuccessorIssue::None) {
        const auto &fixups = block.static_indirect_call_fixups();
        const bool recovered =
            block.successor_issue() == BasicBlock::SuccessorIssue::IndirectControlFlow &&
            !fixups.empty() && std::ranges::none_of(fixups, &IndirectCallFixup::source_incomplete);
        // Standalone function analysis has no caller context for the ABI return.
        const bool external_return =
            non_entry_function && std::get<1>(node_key).empty() && term &&
            (term->mnemonic() == "s_setpc_b64" || term->mnemonic() == "s_set_pc_i64") &&
            term->raw_encoding() && (term->raw_encoding()[0] & 0xffu) == 30;
        const bool end_of_region =
            block.successor_issue() == BasicBlock::SuccessorIssue::MissingFallthrough &&
            block.end_offset() == text_size && term && !term->is_branch();
        if (!recovered && !external_return && !end_of_region)
          missing_successor = term ? term->src_loc() : block.start_offset();
      }
      for (const BasicBlock *successor : block.successors()) {
        const auto successor_it = block_index.find(successor);
        if (successor_it == block_index.end())
          continue;
        const bool is_call_fallthrough =
            std::ranges::any_of(block.call_edges(), [&](const BasicBlock::CallEdge &call) {
              return call.continuation == successor;
            });
        std::optional<CallContinuationSeed> unresolved_call_continuation;
        if (!is_call_fallthrough && term != nullptr && (term->flags() & INDIRECT_CALL) != 0) {
          const Operand *return_operand = term->dst_operand(0);
          const auto return_ref =
              return_operand == nullptr ? std::nullopt : return_operand->to_register_ref();
          if (return_ref && return_ref->cls == RegClass::SGPR) {
            const uint64_t source_call_offset =
                block.end_offset() - static_cast<uint64_t>(term->size());
            unresolved_call_continuation =
                CallContinuationSeed{return_ref->index, source_call_offset};
          }
        }
        if (!is_call_fallthrough && !add_context_edge(successor_it->second, call_stack_after_block,
                                                      unresolved_call_continuation.has_value(),
                                                      unresolved_call_continuation)) {
          report_.supported = false;
          set_call_context_limit_error();
          return;
        }
      }

      for (const BasicBlock::CallEdge &call : block.call_edges()) {
        const auto callee_it = block_index.find(call.callee);
        const auto continuation_it = block_index.find(call.continuation);
        if (callee_it == block_index.end() || continuation_it == block_index.end())
          continue;
        if (std::get<1>(node_key).size() >= kMaxCallDepth) {
          report_.supported = false;
          report_.analysis_error = "waitcheck call depth exceeds supported limit";
          return;
        }
        const bool recursive_reentry =
            std::ranges::any_of(std::get<1>(node_key), [&](const CallFrame &frame) {
              return std::get<2>(frame) == callee_it->second;
            });
        if (recursive_reentry) {
          // A returning recursive call can write registers in its callee.
          // Skipping directly to the continuation loses those writes.
          report_.complete = false;
          report_.analysis_error = "recursive call exceeds waitcheck call-context model";
          return;
        }
        std::vector<CallFrame> callee_stack = call_stack_after_block;
        callee_stack.emplace_back(continuation_it->second, call.return_sreg, callee_it->second,
                                  call.source_call_offset, true, std::nullopt);
        if (!add_context_edge(callee_it->second, std::move(callee_stack),
                              /*conservative_sgpr_entry=*/true)) {
          report_.supported = false;
          set_call_context_limit_error();
          return;
        }
      }
    }

    std::vector<std::vector<size_t>> cfg_predecessors(analysis_blocks.size());
    for (size_t from = 0; from < cfg_successors.size(); ++from) {
      for (size_t to : cfg_successors[from])
        cfg_predecessors[to].push_back(from);
    }

    // Dynamic allocation may release even the lowest VGPR block. Without a
    // closed entry graph and a fixed allocation, a returning VMEM can use effective EXEC=0.
    source_allocation_may_change_ =
        amdgpu::supports_expert_scheduling(arch) && wavefront_size == 32 &&
        (missing_successor || std::ranges::any_of(analysis_blocks, [](const BasicBlock *block) {
           return std::ranges::any_of(block->instructions(), [](const Instruction &inst) {
             return inst.mnemonic() == "s_alloc_vgpr";
           });
         }));

    // On a guarded edge the instruction itself tests EXEC. No scalar register
    // or pending operand can substitute for that test at a CFG merge.
    std::vector<bool> expert_entry_nonzero(analysis_blocks.size());
    if (amdgpu::supports_expert_scheduling(arch) && !missing_successor) {
      for (size_t i = 1; i < analysis_blocks.size(); ++i) {
        const auto dest = analysis_blocks[i]->start_offset();
        const auto nonempty_edge = [&](size_t predecessor) {
          const auto *pred = analysis_blocks[predecessor];
          const auto *branch = pred->terminator();
          if (!branch ||
              (branch->mnemonic() != "s_cbranch_execz" && branch->mnemonic() != "s_cbranch_execnz"))
            return false;
          const auto delta = branch->branch_offset_bytes();
          if (!delta || *delta == 0)
            return false;
          const auto nonempty = branch->mnemonic() == "s_cbranch_execz"
                                    ? pred->end_offset()
                                    : pred->end_offset() + *delta;
          return dest == nonempty;
        };
        const auto &preds = cfg_predecessors[i];
        expert_entry_nonzero[i] = !preds.empty() && std::ranges::all_of(preds, nonempty_edge);
      }
    }
    std::vector<PendingState> in(analysis_blocks.size());
    std::vector<PendingState> out(analysis_blocks.size());
    std::vector<uint8_t> out_initialized(analysis_blocks.size());
    std::unordered_map<uint64_t, const Instruction *> instruction_by_offset;
    track_sgpr_offsets_ = false;
    for (const auto &block : blocks) {
      for (const Instruction &inst : block->instructions()) {
        instruction_by_offset.emplace(inst.src_loc(), &inst);
        track_sgpr_offsets_ |=
            relative_wait_operand(inst, arch) || inst.mnemonic().starts_with("s_movrel") ||
            inst.mnemonic().starts_with("v_movrel") || inst.mnemonic().starts_with("v_swaprel");
      }
    }

    // Revisit only successors whose merged input may have changed. Large
    // generated kernels otherwise spend most of their time rescanning stable
    // blocks on every fixed-point iteration.
    std::deque<size_t> dataflow_worklist;
    std::vector<uint8_t> queued(analysis_blocks.size(), 1);
    for (size_t i = 0; i < analysis_blocks.size(); ++i)
      dataflow_worklist.push_back(i);
    size_t node_visits = 0;
    size_t last_changed_node = 0;
    std::string last_changed_components;
    auto differing_components = [](const PendingState &lhs, const PendingState &rhs) {
      std::string result;
      auto append = [&](std::string_view component) {
        if (!result.empty())
          result += ',';
        result += component;
      };
      for (size_t counter = 0; counter < kCounterCount; ++counter) {
        if (lhs.pending[counter] != rhs.pending[counter])
          append(wait_counter_name(static_cast<WaitCounterKind>(counter)));
        if (lhs.pending_event_ages[counter] != rhs.pending_event_ages[counter])
          append("pending-event-kinds");
        if (lhs.pending_smem[counter] != rhs.pending_smem[counter])
          append("pending-smem");
        if (lhs.uncertain_order[counter] != rhs.uncertain_order[counter])
          append("uncertain-order");
      }
      if (lhs.sgpr_hazards != rhs.sgpr_hazards)
        append("sgpr-hazards");
      if (lhs.va_vdst_hazards != rhs.va_vdst_hazards)
        append("va-vdst");
      if (lhs.m0 != rhs.m0 || lhs.sgpr_offsets != rhs.sgpr_offsets)
        append("scalar-offsets");
      if (lhs.expert_exec_empty != rhs.expert_exec_empty)
        append("expert-empty-exec");
      if (lhs.expert_valu_may_skip != rhs.expert_valu_may_skip)
        append("expert-valu-may-skip");
      if (lhs.expert_exec_nonzero != rhs.expert_exec_nonzero ||
          lhs.expert_saved_nonzero_exec != rhs.expert_saved_nonzero_exec)
        append("expert-saved-mask");
      if (lhs.exec_nonzero != rhs.exec_nonzero)
        append("exec-nonzero");
      if (lhs.vgpr_msb != rhs.vgpr_msb)
        append("vgpr-msb");
      if (lhs.vgpr_msb_setreg_hazard != rhs.vgpr_msb_setreg_hazard)
        append("vgpr-msb-setreg-hazard");
      if (lhs.previous_vm_vsrc_zero_wait != rhs.previous_vm_vsrc_zero_wait)
        append("previous-vm-vsrc-zero-wait");
      if (lhs.async_barrier_post_wait != rhs.async_barrier_post_wait)
        append("async-barrier-post-wait");
      if (lhs.delay_alu != rhs.delay_alu)
        append("delay-alu");
      if (lhs.expert_delay != rhs.expert_delay)
        append("expert-valu-delay");
      if (lhs.expert_scheduling != rhs.expert_scheduling)
        append("expert-scheduling");
      return result;
    };
    const size_t max_node_visits = analysis_blocks.size() * 64 + 1024;
    while (!dataflow_worklist.empty() && node_visits++ < max_node_visits) {
      const size_t i = dataflow_worklist.front();
      dataflow_worklist.pop_front();
      queued[i] = 0;

      // Accumulate inputs across visits so pending events and conservative
      // ordering facts cannot disappear as earlier states circulate in a loop.
      // An unvisited input is bottom, not an initialized empty state.
      PendingState merged = merge_predecessors(cfg_predecessors[i], out, out_initialized).value();
      if (flat_issues_both_counters(arch) && i == 0 && current_kernel_ &&
          cfg_predecessors[i].empty()) {
        merged.exec_nonzero = true;
        merged.expert_exec_nonzero = amdgpu::supports_expert_scheduling(arch);
      }
      if (out_initialized[i])
        merge_into(merged, in[i]);
      if (expert_entry_nonzero[i]) {
        merged.expert_exec_nonzero = true;
        merged.expert_exec_empty = false;
        merged.expert_valu_may_skip = false;
      }
      if (conservative_sgpr_entries[i] != 0) {
        // Linked code objects do not retain whether LLVM's boundary-cull
        // option was enabled. Model its default disabled behavior: every
        // tracked pair and VCC can be live across an externally visible
        // function boundary or a call whose target cannot be recovered.
        merged.sgpr_hazards.tracked_pairs.set();
        merged.sgpr_hazards.tracked_vcc = true;
      }
      if (const auto &continuation = std::get<2>(analysis_node_keys[i])) {
        const auto [return_sreg, source_call_offset] = *continuation;
        const auto source = instruction_by_offset.find(source_call_offset);
        const SgprHazardProducer producer{
            .section_name = section_name,
            .section_offset = source_call_offset,
            .file_offset = file_offset_base + source_call_offset,
            .instruction = source == instruction_by_offset.end() ? std::string{}
                                                                 : source->second->disassemble(),
        };
        set_sgpr_hazard(merged.sgpr_hazards, RegisterRef{RegClass::SGPR, return_sreg, 1},
                        /*is_valu=*/false, producer);
        if (return_sreg < 127) {
          set_sgpr_hazard(merged.sgpr_hazards,
                          RegisterRef{RegClass::SGPR, static_cast<uint16_t>(return_sreg + 1), 1},
                          /*is_valu=*/false, producer);
        }
      }
      PendingState next_out = analyze_block(*analysis_blocks[i], merged, section_name,
                                            file_offset_base, arch, false, i);
      const bool input_changed = !(merged == in[i]);
      const bool output_changed = out_initialized[i] == 0 || !(next_out == out[i]);
      if (!input_changed && !output_changed)
        continue;

      last_changed_node = i;
      last_changed_components.clear();
      if (input_changed)
        last_changed_components = "in:" + differing_components(merged, in[i]);
      if (output_changed) {
        if (!last_changed_components.empty())
          last_changed_components += ';';
        last_changed_components += "out:" + differing_components(next_out, out[i]);
      }
      in[i] = std::move(merged);
      out[i] = std::move(next_out);
      out_initialized[i] = 1;

      if (output_changed) {
        for (size_t successor : cfg_successors[i]) {
          if (queued[successor] != 0)
            continue;
          queued[successor] = 1;
          dataflow_worklist.push_back(successor);
        }
      }
    }

    if (!dataflow_worklist.empty()) {
      report_.supported = false;
      std::ostringstream os;
      os << "waitcheck CFG dataflow did not converge at .text+0x" << std::hex
         << analysis_blocks[last_changed_node]->start_offset();
      if (!last_changed_components.empty())
        os << " (" << last_changed_components << ')';
      report_.analysis_error = os.str();
      return;
    }

    cfg_has_unknown_successor_ = missing_successor.has_value();
    prepare_cfg_path_filter(analysis_blocks, cfg_predecessors, cfg_successors, in, arch);
    cfg_outputs_ = out;
    relative_entry_proven_ = current_kernel_ && !missing_successor;
    for (size_t i = 0; i < analysis_blocks.size(); ++i) {
      current_cfg_view_index_ = i;
      const auto &stack = std::get<1>(analysis_node_keys[i]);
      current_function_entry_.reset();
      current_call_return_sreg_.reset();
      if (!stack.empty()) {
        current_function_entry_ = blocks[std::get<2>(stack.back())]->start_offset();
        current_call_return_sreg_ = std::get<1>(stack.back());
      } else if (non_entry_function && !entry_offsets.empty()) {
        current_function_entry_ = entry_offsets.front();
        current_call_return_sreg_ = 30;
      }
      (void)analyze_block(*analysis_blocks[i], in[i], section_name, file_offset_base, arch, true);
      if (should_stop_after_diagnostic())
        break;
    }
    // One physical wait protects all paths that reach this instruction. A
    // completed proof in one caller rules out a weaker setting even if
    // another caller reaches the same wait with no dependency.
    for (auto &entry : shared_parity_) {
      auto &result = entry.second;
      const uint32_t emitted_count = std::get<3>(entry.first);
      if (result.too_weak)
        continue;
      if (result.exact && (emitted_count == 0 || !result.cannot_relax)) {
        ++report_.counter_parity_exact;
        continue;
      }
      if (result.cannot_relax || !result.diagnostic)
        continue;
      // Another caller can stop at a partial drain before its first
      // dependent instruction. Its surviving requests could tighten the
      // concrete threshold found in this caller.
      if (result.diagnostic->has_required_dependency && result.partial_policy) {
        result.diagnostic = std::move(result.partial_policy);
        result.diagnostic->message += " (strongest required count across partial waits in other "
                                      "callers is unresolved)";
      }
      record_counter_underaccounting(std::move(*result.diagnostic));
    }
    shared_parity_.clear();
    cfg_outputs_ = {};
    current_cfg_view_index_.reset();
    current_function_entry_.reset();
    current_call_return_sreg_.reset();
    clear_cfg_path_filter();
    if (missing_successor) {
      report_.complete = false;
      std::ostringstream message;
      message << "incomplete control flow in " << section_name << "+0x" << std::hex
              << *missing_successor;
      report_.analysis_error = message.str();
    }
  }

  void set_kernel_context(const WaitcheckKernelInfo *kernel) { current_kernel_ = kernel; }

private:
  [[nodiscard]] static bool is_counter_token_only(const PendingEvent &event) {
    // Events without a dependency payload exist only to advance the age of
    // older events on the same hardware counter. Once those ages have been
    // advanced, retaining one static PendingEvent per token adds no
    // information and makes large store-heavy CFGs quadratic in memory.
    return event.regs.size() == 0 && !event.special_reg && !event.barrier_id && !event.check_uses &&
           !event.check_defs && !event.check_exec_defs && !event.check_memory_order &&
           !event.check_program_end && !event.check_counter_parity_order;
  }

  [[nodiscard]] static auto find_event(auto &events, const PendingEvent &event) {
    auto position = std::ranges::lower_bound(events, event, event_identity_less);
    if (position != events.end() && same_event_identity(*position, event))
      return position;
    return events.end();
  }

  [[nodiscard]] static bool contains_event(const std::vector<PendingEvent> &events,
                                           const PendingEvent &event) {
    return std::ranges::any_of(
        events, [&](const PendingEvent &existing) { return same_event_identity(existing, event); });
  }

  [[nodiscard]] bool should_stop_after_diagnostic() const { return report_.stopped_early; }

  struct KernargRead {
    ByteInterval bytes;
    RegisterRef destination;
  };

  std::optional<KernargRead> kernarg_read(const Instruction &inst, RegisterRef base) const {
    if (!inst.mnemonic().starts_with("s_load_dword") || inst.num_dst_operands() != 1 ||
        inst.num_src_operands() != 3 || inst.src_operand(0)->to_register_ref() != base ||
        !inst.raw_encoding() || inst.size() != sizeof(amdgpu::SmemMachineInst))
      return std::nullopt;
    amdgpu::SmemMachineInst smem{};
    std::memcpy(&smem, inst.raw_encoding(), sizeof(smem));
    const auto dst = inst.dst_operand(0)->to_register_ref();
    if (!dst || dst->cls != RegClass::SGPR ||
        dst->index + dst->width > current_kernel_->min_sgprs || !smem.imm || smem.soffset_en ||
        smem.offset >= (1u << 20) || smem.offset % 4)
      return std::nullopt;
    // A positive CDNA3 SMEM immediate has only 20 bits. The last byte fits in uint32_t.
    return KernargRead{{smem.offset, smem.offset + static_cast<uint32_t>(dst->width) * 4}, *dst};
  }

  bool repeats_kernarg_read(uint64_t producer, const Instruction &consumer) {
    if (report_.arch != ROCJITSU_CODE_ARCH_CDNA3 || !current_kernel_ ||
        !current_kernel_->kernarg_sgpr || !current_kernel_->alias_descriptor_vaddrs.empty() ||
        image_.size() < sizeof(Elf64_Ehdr) || image_[EI_OSABI] != ELFOSABI_AMDGPU_HSA ||
        cfg_has_unknown_successor_ || current_cfg_view_index_ != 0 || cfg_views_.empty())
      return false;
    const auto &entry = cfg_views_.front();
    if (!entry.predecessors.empty() || entry.instructions.empty() ||
        entry.block->start_offset() != current_kernel_->entry_offset ||
        path_query_starts_.at(entry.block->start_offset()).contexts != 1)
      return false;

    const RegisterRef base{RegClass::SGPR, *current_kernel_->kernarg_sgpr, 2};
    const auto later = kernarg_read(consumer, base);
    if (!later)
      return false;
    // Only prove a straight-line prefix of the ABI entry. Everything past the
    // budget keeps its original overwrite diagnostic.
    constexpr size_t max_preceding_instructions = 32;
    auto prefix = std::span(entry.instructions)
                      .first(std::min(entry.instructions.size(), max_preceding_instructions + 1));
    auto pos = std::ranges::lower_bound(prefix, consumer.src_loc(), {},
                                        &CfgInstructionView::section_offset);
    if (pos == prefix.end() || pos->instruction != &consumer)
      return false;
    auto prior = std::ranges::lower_bound(prefix.begin(), pos, producer, {},
                                          &CfgInstructionView::section_offset);
    if (prior == pos || prior->section_offset != producer)
      return false;
    const auto earlier = kernarg_read(*prior->instruction, base);
    if (!earlier || earlier->destination != later->destination ||
        compare_intervals(earlier->bytes, later->bytes) != IntervalRelation::Identical)
      return false;

    const RegisterRef high{RegClass::SGPR, static_cast<uint16_t>(base.index + 1), 1};
    while (pos != prefix.begin()) {
      --pos;
      const auto &inst = *pos->instruction;
      const auto &defs = path_scalar_defs_.at(&inst);
      if (std::ranges::find(defs, base.index) == defs.end() &&
          std::ranges::find(defs, high.index) == defs.end())
        continue;
      // CDNA3 GPUVM uses 48 address bits. Clearing higher bits preserves the address.
      if (inst.mnemonic() == "s_and_b32" && inst.dst_operand(0)->to_register_ref() == high &&
          inst.src_operand(0)->to_register_ref() == high &&
          inst.src_operand(1)->const_value() == 0xffffu)
        continue;
      return false;
    }

    if (!metadata_sizes_)
      metadata_sizes_ = metadata_kernarg_sizes(image_);
    uint32_t extent = current_kernel_->kernarg_size;
    const auto metadata = metadata_sizes_->find(current_kernel_->name);
    if (metadata != metadata_sizes_->end())
      extent = extent ? std::min(extent, metadata->second) : metadata->second;
    return later->bytes.end <= extent;
  }

  uint32_t minimum_source_vgprs(rj_code_arch_t arch) const {
    if (source_allocation_may_change_)
      return 0;
    const auto floor =
        descriptor_vgpr_count_granule_for_wavefront(arch, wavefront_size_).value_or(0);
    if (current_kernel_ && (arch == ROCJITSU_CODE_ARCH_CDNA5 || wavefront_size_ == 64))
      return std::max(floor, current_kernel_->min_vgprs);
    return floor;
  }

  // A still-live source proves its original EXEC contained a lane. Record a
  // set of wave32 SGPRs that each retain all those lanes. Explicit writes
  // invalidate the corresponding copy; relative writes and calls invalidate all.
  void transfer_saved_source_mask(const Instruction &inst, bool writes_exec, bool km_pending,
                                  std::span<const uint16_t> defs, bool &active, bool &expanded,
                                  std::bitset<128> &saved) const {
    const auto name = inst.mnemonic();
    const bool indirect = (inst.flags() & (INDIRECT_CALL | INDIRECT_BRANCH)) ||
                          name.starts_with("s_call") || name.starts_with("s_rfe");
    if (indirect || name.starts_with("s_movrel"))
      saved.reset();
    for (const auto reg : defs)
      if (reg < saved.size())
        saved.reset(reg);
    if (!writes_exec && !indirect)
      return;
    bool restored = false;
    if (name == "s_and_saveexec_b32" && active && !indirect) {
      const auto dst = inst.dst_operand(0)->to_register_ref();
      if (dst && dst->cls == RegClass::SGPR && dst->width == 1 && dst->index < 128 &&
          dst->index < current_kernel_->min_sgprs && !km_pending &&
          !valu_source_mask_writes_.test(dst->index))
        saved.set(dst->index);
    } else if (saved.any() && name == "s_or_b32" &&
               low_selector(inst.dst_operand(0), amdgpu::kExecSelectorFirst)) {
      for (int i = 0; i < inst.num_src_operands(); ++i) {
        const auto ref = inst.src_operand(i)->to_register_ref();
        restored |= ref && ref->cls == RegClass::SGPR && ref->width == 1 &&
                    ref->index < saved.size() && saved.test(ref->index);
      }
    }
    active = restored || (active && preserves_exec_lanes(inst));
    expanded = active;
  }

  void index_repeated_exec_guards() {
    for (size_t guard = 0; guard < cfg_views_.size(); ++guard) {
      const auto &first = cfg_views_[guard];
      if (first.instructions.size() < 2 || first.successors.size() != 2)
        continue;
      const auto &branch = first.instructions.back();
      const auto &save = *first.instructions[first.instructions.size() - 2].instruction;
      if (branch.instruction->mnemonic() != "s_cbranch_execz" ||
          save.mnemonic() != "s_and_saveexec_b32")
        continue;
      const auto saved = save.dst_operand(0)->to_register_ref();
      const auto mask = save.src_operand(0)->to_register_ref();
      if (!saved || !mask || saved->cls != RegClass::SGPR || mask->cls != RegClass::SGPR ||
          saved->index == mask->index || !current_kernel_ ||
          std::max(saved->index, mask->index) >= current_kernel_->min_sgprs)
        continue;
      size_t arm = first.successors[0], merge = first.successors[1];
      const auto next = branch.section_offset + branch.instruction->size();
      if (cfg_views_[arm].block->start_offset() != next)
        std::swap(arm, merge);
      const auto &body = cfg_views_[arm];
      const auto &join = cfg_views_[merge];
      const auto target = branch.instruction->branch_offset_bytes();
      if (!target || *target <= 0 || body.block->start_offset() != next ||
          join.block->start_offset() != next + *target ||
          body.block->end_offset() != next + *target || body.predecessors.size() != 1 ||
          body.predecessors.front() != guard || body.successors.size() != 1 ||
          body.successors.front() != merge || join.predecessors.size() != 2 ||
          join.instructions.size() != 3 ||
          join.instructions.back().instruction->mnemonic() != "s_cbranch_execz")
        continue;
      const auto &restore = *join.instructions[0].instruction;
      const auto &repeat = *join.instructions[1].instruction;
      if (restore.mnemonic() != "s_or_b32" ||
          !low_selector(restore.dst_operand(0), amdgpu::kExecSelectorFirst) ||
          !low_selector(restore.src_operand(0), amdgpu::kExecSelectorFirst) ||
          restore.src_operand(1)->to_register_ref() != saved ||
          repeat.mnemonic() != "s_and_saveexec_b32" ||
          repeat.dst_operand(0)->to_register_ref() != saved ||
          repeat.src_operand(0)->to_register_ref() != mask)
        continue;
      const bool clobber = std::ranges::any_of(body.instructions, [&](const auto &view) {
        const auto &inst = *view.instruction;
        const auto &defs = path_scalar_defs_.at(&inst);
        return view.invalidates_exec || inst.is_branch() || inst.flags() & INDIRECT_CALL ||
               inst.mnemonic().starts_with("s_movrel") || inst.mnemonic().starts_with("s_call") ||
               std::ranges::find(defs, saved->index) != defs.end() ||
               std::ranges::find(defs, mask->index) != defs.end();
      });
      if (!clobber) {
        // Both incoming edges carry E & M. The two adjacent instructions compute
        // ((E & M) | E) & M, restoring that exact mask before it is tested again.
        cfg_views_[merge].repeated_exec_filter_arm = arm;
        cfg_has_repeated_exec_filter_ = true;
      }
    }
  }

  [[nodiscard]] bool is_wave64_exec_scc_guard(const CfgInstructionView &ones_view,
                                              const Instruction &mask,
                                              const Instruction &branch) const {
    const Instruction &ones = *ones_view.instruction;
    if (!is_scc_conditional_branch(branch.mnemonic()) || mask.mnemonic() != "s_and_b64" ||
        ones.mnemonic() != "s_mov_b64" || ones_view.scalar_destination_pending ||
        mask.is_vskip_affected() || ones.is_vskip_affected() || branch.is_vskip_affected())
      return false;
    const auto allocated_pair = [&](const Operand *op) -> std::optional<RegisterRef> {
      if (!op)
        return {};
      const auto reg = op->to_register_ref();
      if (reg && reg->cls == RegClass::SGPR && reg->width == 2 && reg->index <= 100 &&
          reg->index % 2 == 0 && reg->index + reg->width <= current_kernel_->min_sgprs)
        return reg;
      return {};
    };
    const auto all_ones = allocated_pair(ones.dst_operand(0));
    if (!all_ones || !inline_minus_one(ones.src_operand(0)) || !allocated_pair(mask.dst_operand(0)))
      return false;
    for (unsigned operand = 0; operand < 2; ++operand) {
      if (low_selector(mask.src_operand(operand), amdgpu::kExecSelectorFirst) &&
          allocated_pair(mask.src_operand(1 - operand)) == all_ones)
        return true;
    }
    return false;
  }

  [[nodiscard]] static bool inline_minus_one(const Operand *operand) {
    constexpr int kInlineMinusOne = 193;
    return operand && operand->encoding_value() == kInlineMinusOne;
  }

  [[nodiscard]] bool wave64_guard_drains_load(size_t block_index, size_t producer,
                                              rj_code_arch_t arch) const {
    const auto *block = &cfg_views_[block_index];
    size_t index = producer;
    const auto advance = [&](int32_t displacement = 0) {
      const auto &current = block->instructions[index];
      const uint64_t next = current.section_offset + current.instruction->size() + displacement;
      if (++index == block->instructions.size()) {
        if (block->successors.size() != 1)
          return false;
        block = &cfg_views_[block->successors.front()];
        index = 0;
      }
      return !block->instructions.empty() && block->instructions[index].section_offset == next;
    };
    if (!advance())
      return false;
    const auto &after_load = *block->instructions[index].instruction;
    if (after_load.mnemonic() == "s_branch") {
      const auto target = after_load.branch_offset_bytes();
      if (!target || !advance(*target))
        return false;
    }
    // Other predecessors can join between these instructions. The path from
    // this producer still executes the adjacent all-ones write before S_AND.
    const auto &ones = block->instructions[index];
    if (!advance())
      return false;
    const auto &mask = *block->instructions[index].instruction;
    if (!advance() || index + 1 != block->instructions.size() || block->successors.size() != 2)
      return false;
    const auto &branch = *block->instructions[index].instruction;
    if (!is_wave64_exec_scc_guard(ones, mask, branch))
      return false;
    const auto end = block->block->end_offset();
    const auto delta = branch.branch_offset_bytes();
    if (!delta || !*delta)
      return false;
    const auto nonempty_offset = branch.mnemonic() == "s_cbranch_scc0" ? end : end + *delta;
    const auto empty_offset = nonempty_offset == end ? end + *delta : end;
    if (std::ranges::none_of(block->successors, [&](size_t successor) {
          return cfg_views_[successor].block->start_offset() == empty_offset;
        }))
      return false;
    for (size_t successor : block->successors) {
      const auto &next = cfg_views_[successor];
      if (next.block->start_offset() != nonempty_offset)
        continue;
      // The guard tests the same EXEC that controlled the load. The empty
      // edge issued no result. The other edge drains before any VGPR access.
      block = &next;
      index = 0;
      bool crossed_prefix_boundary = false;
      while (index < block->instructions.size()) {
        const auto &view = block->instructions[index];
        const auto &inst = *view.instruction;
        if (inst.mnemonic() == "s_waitcnt") {
          const auto fields = explicit_wait_fields(inst, arch).value();
          return fields && (*fields)[counter_index(WaitCounterKind::Load)] == 0;
        }
        if (inst.mnemonic() != "s_mov_b32" || view.invalidates_exec)
          break;
        const auto dst = inst.dst_operand(0)->to_register_ref();
        if (!dst || dst->cls != RegClass::SGPR || dst->width != 1)
          break;
        if (index + 1 == block->instructions.size()) {
          // Another edge may target the wait directly and split this prefix.
          if (crossed_prefix_boundary)
            break;
          crossed_prefix_boundary = true;
        }
        if (!advance())
          break;
      }
      return false;
    }
    return false;
  }

  void index_wave64_guarded_loads(rj_code_arch_t arch) {
    std::unordered_map<uint64_t, bool> drains;
    for (size_t block = 0; block < cfg_views_.size(); ++block) {
      const auto &instructions = cfg_views_[block].instructions;
      for (size_t i = 0; i < instructions.size(); ++i) {
        const auto &view = instructions[i];
        if (view.instruction->mnemonic() != "global_load_dwordx2")
          continue;
        const auto *issue = view.instruction->amdgpu_memory_issue_info();
        const bool drained =
            issue && issue->exec_masked && wave64_guard_drains_load(block, i, arch);
        auto [entry, inserted] = drains.try_emplace(view.section_offset, drained);
        if (!inserted)
          entry->second &= drained;
      }
    }
    for (const auto &[offset, drained] : drains)
      if (drained)
        wave64_guarded_loads_.insert(offset);
  }

  void prepare_cfg_path_filter(const std::vector<BasicBlock *> &blocks,
                               const std::vector<std::vector<size_t>> &predecessors,
                               const std::vector<std::vector<size_t>> &successors,
                               std::span<const PendingState> inputs, rj_code_arch_t arch) {
    cfg_views_.clear();
    feasible_path_cache_.clear();
    copy_lane_cache_.clear();
    path_scalar_defs_.clear();
    path_predicates_.clear();
    masked_predicates_.clear();
    repeated_masked_predicate_.reset();
    indexed_masked_predicates_ = false;
    indexed_loop_parity_ = false;
    loop_parity_.reset();
    path_mask_inputs_.clear();
    cfg_has_scratch_zero_exit_ = false;
    path_exec_inputs_.clear();
    exec_masked_results_.clear();
    cfg_has_mask_branches_ = false;
    cfg_has_saved_source_exec_ = false;
    source_save_destinations_.reset();
    valu_source_mask_writes_.reset();
    cfg_has_constant_scc_ = false;
    cfg_has_exec_guards_ = false;
    cfg_has_salu_exec_write_ = false;
    cfg_has_scalar_prefetch_ = false;
    cfg_has_expert_delay_ = false;
    cfg_has_repeated_exec_filter_ = false;
    wave64_guarded_loads_.clear();
    path_query_starts_.clear();
    reverse_reachability_cache_.clear();
    forward_reachability_cache_.clear();
    reachability_cache_bytes_ = 0;
    cfg_component_.clear();
    dominator_preorder_.clear();
    dominator_subtree_end_.clear();

    const bool track_wave32_vcc_lo =
        wavefront_size_ == 32 && amdgpu::supports_expert_scheduling(arch) &&
        std::ranges::any_of(blocks, [](const BasicBlock *block) {
          const auto *terminator = block->terminator();
          return terminator && mask_branch_class(terminator->mnemonic()) == RegClass::VCC;
        });
    const bool gfx11_wave32 = current_kernel_ && !cfg_has_unknown_successor_ &&
                              track_sgpr_offsets_ && wavefront_size_ == 32 &&
                              (arch == ROCJITSU_CODE_ARCH_RDNA3 || arch == ROCJITSU_CODE_ARCH_RDNA3_5);
    const bool track_wave64_exec_scc = current_kernel_ && !cfg_has_unknown_successor_ &&
                                     arch == ROCJITSU_CODE_ARCH_CDNA4 && wavefront_size_ == 64;
    bool may_defer_guard_mask_write = false;
    bool has_wave64_ones = false;
    bool may_defer_m0_write = false;
    cfg_views_.reserve(blocks.size());
    for (size_t block_index = 0; block_index < blocks.size(); ++block_index) {
      BasicBlock *block = blocks[block_index];
      CfgBlockView view;
      view.block = block;
      view.predecessors = predecessors[block_index];
      view.successors = successors[block_index];
      const bool track_banks = arch == ROCJITSU_CODE_ARCH_CDNA5;
      PendingState banks;
      banks.vgpr_msb = inputs[block_index].vgpr_msb;
      banks.vgpr_msb_setreg_hazard = inputs[block_index].vgpr_msb_setreg_hazard;
      const bool track_saved_exec =
          current_kernel_ && wavefront_size_ == 32 && amdgpu::supports_expert_scheduling(arch);
      std::bitset<128> pending_scalar_writes;
      const bool track_pending_scalar = track_saved_exec || track_wave64_exec_scc;
      const auto scalar_counter = track_wave64_exec_scc ? WaitCounterKind::Ds : WaitCounterKind::Km;
      if (track_pending_scalar) {
        for (const auto &event : inputs[block_index].pending[counter_index(scalar_counter)]) {
          if (event.kind == WaitEventKind::Smem || event.kind == WaitEventKind::SqMessage)
            event.regs.for_each([&](RegisterRef r) {
              if (r.cls == RegClass::SGPR && r.index < pending_scalar_writes.size())
                pending_scalar_writes.set(r.index);
            });
        }
      }
      auto vector_skip = inputs[block_index].vector_skip;
      uint64_t section_offset = block->start_offset();
      for (const Instruction &inst : block->instructions()) {
        if (options_.check_counter_parity && track_banks && is_scalar_prefetch(inst.mnemonic()))
          cfg_has_scalar_prefetch_ = true;
        cfg_has_expert_delay_ |=
            amdgpu::supports_expert_scheduling(arch) && inst.mnemonic() == "s_delay_alu";
        const auto predicate = scalar_scc_predicate(inst, arch);
        const size_t predicate_index =
            predicate ? path_predicates_.size() : std::numeric_limits<size_t>::max();
        if (predicate) {
          path_predicates_.emplace_back(block_index, section_offset, *predicate);
          cfg_has_constant_scc_ |= predicate->kind == SccPredicate::Kind::Constant;
        }
        const InstDefUse du =
            inst_def_use_for_waitcheck(inst, banks.vgpr_msb, arch, wavefront_size_);
        auto [defs, inserted] = path_scalar_defs_.try_emplace(&inst);
        if (inserted) {
          du.defs.for_each([&](RegisterRef ref) {
            if (ref.cls == RegClass::SGPR)
              defs->second.push_back(ref.index);
          });
        }
        static_assert(kCounterCount <= 64);
        uint64_t issued_counters = 0;
        uint64_t issued_scalar_result_counters = 0;
        uint64_t issued_scc_write_counters = 0;
        bool scalar_result_from_operand = false;
        bool scalar_result_dest_scc = false;
        const auto classified = effective_events(inst, banks.vgpr_msb, arch);
        may_defer_m0_write |= gfx11_wave32 && du.defs.contains({RegClass::M0, 0, 1}) &&
                              std::ranges::any_of(classified, [](const ClassifiedEvent &event) {
                                return is_completion_counter(event.counter) &&
                                       event.registers == TrackedRegisterSource::Defs;
                              });
        bool scalar_destination_pending = false;
        if (track_pending_scalar) {
          const bool save = track_saved_exec && inst.mnemonic() == "s_and_saveexec_b32";
          cfg_has_saved_source_exec_ |= save;
          if (track_saved_exec &&
              (save || (options_.check_counter_parity && inst.mnemonic() == "s_mov_b32"))) {
            const auto dst = inst.dst_operand(0)->to_register_ref();
            if (dst && dst->cls == RegClass::SGPR && dst->index < pending_scalar_writes.size()) {
              scalar_destination_pending = pending_scalar_writes.test(dst->index);
              if (save && !scalar_destination_pending)
                source_save_destinations_.set(dst->index);
            }
          }
          if (track_wave64_exec_scc && inst.mnemonic() == "s_mov_b64" &&
              inline_minus_one(inst.src_operand(0))) {
            const auto dst = inst.dst_operand(0)->to_register_ref();
            if (dst && dst->cls == RegClass::SGPR && dst->width == 2 && dst->index < 127) {
              has_wave64_ones = true;
              scalar_destination_pending =
                  pending_scalar_writes.test(dst->index) || pending_scalar_writes.test(dst->index + 1);
            }
          }
          // A full scalar memory wait drains returning memory and messages.
          // Other waits conservatively retain these destinations within this block.
          if (inst.is_waitcnt()) {
            const auto fields = explicit_wait_fields(inst, arch).value();
            if (fields && (*fields)[counter_index(scalar_counter)] == 0)
              pending_scalar_writes.reset();
          }
          const bool deferred = std::ranges::any_of(classified, [](const ClassifiedEvent &event) {
            return (event.kind == WaitEventKind::Smem || event.kind == WaitEventKind::SqMessage) &&
                   event.registers == TrackedRegisterSource::Defs;
          });
          may_defer_guard_mask_write |= track_wave64_exec_scc && deferred &&
                                        (du.defs.contains({RegClass::EXEC, 0, 1}) ||
                                         du.defs.contains({RegClass::SCC, 0, 1}));
          if (deferred || (tracks_gfx12_sgpr_hazards(arch) && (inst.flags() & VALU))) {
            for (const auto reg : defs->second) {
              if (reg < pending_scalar_writes.size()) {
                if (deferred)
                  pending_scalar_writes.set(reg);
                else
                  valu_source_mask_writes_.set(reg);
              }
            }
          }
        }
        for (const auto &event : classified)
          issued_counters |= uint64_t{1} << counter_index(event.counter);
        if (wavefront_size_ == 32 &&
            (arch == ROCJITSU_CODE_ARCH_RDNA3 || arch == ROCJITSU_CODE_ARCH_RDNA3_5) &&
            std::ranges::any_of(classified, [](const ClassifiedEvent &event) {
              return (event.counter == WaitCounterKind::Ds ||
                      event.counter == WaitCounterKind::Load) &&
                     event.registers == TrackedRegisterSource::Defs;
            })) {
          const auto *issue = inst.amdgpu_memory_issue_info();
          if (issue && issue->exec_masked) {
            amdgpu::RegisterModifiers mods;
            inst.amdgpu_register_modifiers(mods);
            // An exact EXEC subset is safe for per-lane memory results.
            // Whole-quad and whole-wave result writes need a wider mask.
            if (!mods.exec_whole_quads && !mods.exec_all_if_nonzero)
              exec_masked_results_.insert(section_offset);
          }
        }
        for (const auto &event : classified) {
          if (event.registers == TrackedRegisterSource::Defs &&
              (event.kind == WaitEventKind::Smem || event.kind == WaitEventKind::SqMessage)) {
            issued_scalar_result_counters |= uint64_t{1} << counter_index(event.counter);
            scalar_result_from_operand = true;
          }
          if (event.kind == WaitEventKind::SccWrite &&
              event.special_reg == RegisterRef{RegClass::SCC, 0, 1}) {
            issued_scalar_result_counters |= uint64_t{1} << counter_index(event.counter);
            issued_scc_write_counters |= uint64_t{1} << counter_index(event.counter);
            scalar_result_dest_scc = true;
          }
        }
        std::vector<uint16_t> scalar_result_dest_sgprs;
        bool scalar_result_dest_exec = false, scalar_result_dest_vcc = false,
             scalar_result_dest_unknown = false;
        if (scalar_result_from_operand) {
          const auto *dst = inst.dst_operand(0);
          const auto dest = dst ? dst->to_register_ref() : std::nullopt;
          scalar_result_dest_unknown = !dest;
          const auto mark_destination = [&](RegisterRef ref) {
            if (ref.cls == RegClass::SGPR) {
              for (uint16_t reg = ref.index; reg < ref.index + ref.width; ++reg)
                scalar_result_dest_sgprs.push_back(reg);
            } else if (ref.cls == RegClass::EXEC) {
              scalar_result_dest_exec = true;
            } else if (ref.cls == RegClass::VCC) {
              scalar_result_dest_vcc = true;
            } else if (ref.cls == RegClass::SCC) {
              scalar_result_dest_scc = true;
            }
          };
          if (dest)
            mark_destination(*dest);
          du.defs.for_each(mark_destination);
        }
        uint64_t zero_wait_counters = 0;
        if (inst.is_waitcnt()) {
          if (const auto fields = explicit_wait_fields(inst, arch).value(); fields) {
            for (size_t i = 0; i < kCounterCount; ++i) {
              if ((*fields)[i] == 0)
                zero_wait_counters |= uint64_t{1} << i;
            }
          }
        }
        const bool invalidates_scc = du.defs.contains({RegClass::SCC, 0, 1}) ||
                                     instruction_defines_special(inst, {RegClass::SCC, 0, 1}) ||
                                     instruction_uses_special(inst, {RegClass::SCC, 0, 1});
        const bool invalidates_exec =
            writes_exec_mask(inst) || du.defs.contains({RegClass::EXEC, 0, 1});
        cfg_has_salu_exec_write_ |= invalidates_exec && inst.mnemonic().starts_with("s_");
        const bool invalidates_vcc = instruction_defines_vcc(inst, du);
        const bool invalidates_vcc_condition =
            invalidates_vcc && (!track_wave32_vcc_lo || !only_defines_vcc_hi(inst, du, arch));
        if (mask_branch_class(inst.mnemonic()) == RegClass::EXEC ||
            ((inst.mnemonic() == "s_mov_b64" || inst.mnemonic() == "s_mov_b32") &&
             low_selector(inst.dst_operand(0), amdgpu::kExecSelectorFirst)))
          cfg_has_exec_guards_ = true;
        if (mask_branch_class(inst.mnemonic()))
          cfg_has_mask_branches_ = true;
        if (wavefront_size_ == 32 &&
            (arch == ROCJITSU_CODE_ARCH_RDNA3 || arch == ROCJITSU_CODE_ARCH_RDNA3_5) &&
            inst.mnemonic() == "s_mov_b32" &&
            low_selector(inst.dst_operand(0), amdgpu::kExecSelectorFirst)) {
          const auto ref = inst.src_operand(0)->to_register_ref();
          if (ref && ref->cls == RegClass::SGPR && ref->width == 1)
            path_exec_inputs_.insert(ref->index);
        }
        if (wavefront_size_ == 32 && inst.mnemonic() == "s_and_b32" &&
            operand_is_vcc(inst.dst_operand(0)) &&
            (low_selector(inst.src_operand(0), amdgpu::kExecSelectorFirst) ||
             low_selector(inst.src_operand(1), amdgpu::kExecSelectorFirst))) {
          cfg_has_exec_guards_ = true;
          for (int i = 0; i < inst.num_src_operands(); ++i) {
            const auto ref = inst.src_operand(i)->to_register_ref();
            if (ref && ref->cls == RegClass::SGPR && ref->width == 1)
              path_mask_inputs_.insert(ref->index);
          }
        }
        const auto mode_write = expert_scheduling_mode_write(inst, arch);
        const bool valu_issues_with_empty_exec =
            (issued_counters & (uint64_t{1} << counter_index(WaitCounterKind::VaVdst))) &&
            (invalidates_vcc || expert_valu_issues_without_active_lanes(inst, du.defs));
        const bool neutral =
            !invalidates_exec && !invalidates_vcc && defs->second.empty() && !predicate &&
            !inst.is_branch() && !inst.is_waitcnt() && !is_program_end(inst.mnemonic()) &&
            !releases_gfx10_scalar_sources(inst, arch) && !vinterp_wait_exp(inst) &&
            dsdir_wait_vm_vsrc(inst) != 0 && !invalidates_scc && issued_counters == 0;
        const size_t index = view.instructions.size();
        const auto scratch_relative_sgpr =
            gfx11_wave32 && index
                ? resolve_scratch_relative_sgpr(inst, *view.instructions.back().instruction, arch)
                : std::nullopt;
        view.instructions.push_back(
            {section_offset, &inst, index + (neutral ? 1 : 0), predicate_index, issued_counters,
             issued_scalar_result_counters, issued_scc_write_counters, zero_wait_counters,
             std::move(scalar_result_dest_sgprs), scalar_result_dest_exec, scalar_result_dest_vcc,
             scalar_result_dest_scc, scalar_result_dest_unknown, invalidates_scc,
             invalidates_exec, invalidates_vcc_condition, banks.vgpr_msb, false, vector_skip,
             mode_write, valu_issues_with_empty_exec, scalar_destination_pending,
             scratch_relative_sgpr});
        update_vector_skip(vector_skip, inst, arch);
        if ((inst.flags() & INDIRECT_CALL || inst.mnemonic().starts_with("s_call")) &&
            (arch == ROCJITSU_CODE_ARCH_RDNA3 || arch == ROCJITSU_CODE_ARCH_RDNA3_5) &&
            wavefront_size_ == 32 && !view.successors.empty()) {
          // Unresolved calls have a continuation edge.
          // A fully traversed callee accounts for its own register writes.
          view.instructions.back().enters_resolved_callee =
              std::ranges::all_of(view.successors, [&](size_t succ) {
                bool enters_callee = false;
                for (const auto &edge : block->call_edges()) {
                  if (edge.continuation == blocks[succ])
                    return false;
                  enters_callee |= edge.callee == blocks[succ];
                }
                return enters_callee;
              });
        }
        if (track_banks) {
          const bool previous_mode_write = std::exchange(banks.vgpr_msb_setreg_hazard, false);
          apply_vgpr_msb_mode(banks, inst, arch, previous_mode_write);
        }
        section_offset += static_cast<uint64_t>(inst.size());
      }
      if (wavefront_size_ == 32 &&
          (arch == ROCJITSU_CODE_ARCH_RDNA3 || arch == ROCJITSU_CODE_ARCH_RDNA3_5) &&
          view.successors.size() == 2 && !view.instructions.empty()) {
        const CfgInstructionView &last = view.instructions.back();
        const auto &branch = *last.instruction;
        const auto delta = branch.branch_offset_bytes();
        if (mask_branch_class(branch.mnemonic()) == RegClass::EXEC && delta && *delta > 0) {
          const auto next = last.section_offset + static_cast<uint64_t>(branch.size());
          const auto target = next + *delta;
          size_t fallthrough = view.successors[0], join = view.successors[1];
          if (blocks[fallthrough]->start_offset() != next)
            std::swap(fallthrough, join);
          const BasicBlock &arm = *blocks[fallthrough];
          // Check the actual context edges. The skipped instructions must preserve
          // every path fact, issue age, and lane mask until that same join.
          if (arm.start_offset() == next && arm.end_offset() == target &&
              blocks[join]->start_offset() == target && successors[fallthrough].size() == 1 &&
              successors[fallthrough].front() == join &&
              std::ranges::all_of(arm.instructions(),
                                  [](const Instruction &i) { return i.mnemonic() == "s_nop"; }))
            view.exec_nop_join = join;
        }
      }
      for (size_t index = view.instructions.size(); index-- > 0;) {
        auto &inst = view.instructions[index];
        if (inst.neutral_run_end == index + 1 && index + 1 < view.instructions.size())
          inst.neutral_run_end = view.instructions[index + 1].neutral_run_end;
      }
      // Track scalar inputs to an EXEC-derived SCC only when that SCC guards
      // control flow before another instruction can replace it. Unrelated
      // scalar ANDs are common in large elementwise kernels and would multiply
      // every lane-feasibility search through their constant assignments.
      if (wavefront_size_ == 32) {
        for (size_t index = 0; index < view.instructions.size(); ++index) {
          const Instruction &mask = *view.instructions[index].instruction;
          if (mask.mnemonic() != "s_and_b32")
            continue;
          const auto *dst_operand = mask.dst_operand(0);
          const auto dst = dst_operand ? dst_operand->to_register_ref() : std::nullopt;
          if (!dst || dst->cls != RegClass::SGPR || dst->width != 1 ||
              (!low_selector(mask.src_operand(0), amdgpu::kExecSelectorFirst) &&
               !low_selector(mask.src_operand(1), amdgpu::kExecSelectorFirst)))
            continue;
          bool guarded = false;
          for (size_t next = index + 1;
               next < view.instructions.size() && next <= index + 16; ++next) {
            const auto &following = view.instructions[next];
            if (is_scc_conditional_branch(following.instruction->mnemonic())) {
              guarded = true;
              break;
            }
            if (following.invalidates_scc || (following.instruction->flags() & INDIRECT_CALL))
              break;
          }
          if (!guarded)
            continue;
          cfg_has_exec_guards_ = true;
          for (int source = 0; source < mask.num_src_operands(); ++source) {
            const auto ref = mask.src_operand(source)->to_register_ref();
            if (ref && ref->cls == RegClass::SGPR && ref->width == 1)
              path_mask_inputs_.insert(ref->index);
          }
        }
      }
      if (!cfg_has_scratch_zero_exit_ && wavefront_size_ == 32 &&
          (arch == ROCJITSU_CODE_ARCH_RDNA3 || arch == ROCJITSU_CODE_ARCH_RDNA3_5) &&
          view.instructions.size() >= 2 &&
          view.instructions.back().instruction->mnemonic() == "s_cbranch_vccz") {
        const auto &mask = *view.instructions[view.instructions.size() - 2].instruction;
        const auto offset = view.instructions.back().instruction->branch_offset_bytes();
        if (offset && mask.mnemonic() == "s_and_b32" &&
            low_selector(mask.dst_operand(0), amdgpu::kVccSelectorFirst)) {
          for (unsigned source = 0; source < 2; ++source) {
            if (!low_selector(mask.src_operand(source), amdgpu::kExecSelectorFirst))
              continue;
            const auto input = mask.src_operand(1 - source)->to_register_ref();
            if (!input || input->cls != RegClass::SGPR || input->width != 1)
              continue;
            const auto exit = static_cast<uint64_t>(block->end_offset() + *offset);
            cfg_has_scratch_zero_exit_ = std::ranges::any_of(view.successors, [&](size_t succ) {
              const auto &target = *blocks[succ];
              const auto first = target.instructions().begin();
              return target.start_offset() == exit && first != target.instructions().end() &&
                     std::next(first) == target.instructions().end() &&
                     is_program_end((*first).mnemonic());
            });
          }
        }
      }
      size_t run_begin = 0;
      for (size_t index = 0; index < view.instructions.size(); ++index) {
        const auto &inst = view.instructions[index];
        uint64_t query_start = inst.section_offset;
        if (inst.neutral_run_end > index)
          query_start = view.instructions[run_begin].section_offset;
        else
          run_begin = index + 1;
        auto [entry, inserted] =
            path_query_starts_.try_emplace(inst.section_offset, PathQueryStart{query_start, 0});
        ++entry->second.contexts;
        // Overlapping blocks or call contexts must agree on the whole run.
        if (entry->second.offset != query_start)
          entry->second.offset = inst.section_offset;
      }
      cfg_views_.push_back(std::move(view));
    }

    // A deferred write to M0 may complete between the literal and the
    // relative move. Keep the general transfer in that case.
    cfg_can_resolve_scratch_relative_ = gfx11_wave32 && !may_defer_m0_write;
    if (has_wave64_ones && !may_defer_guard_mask_write)
      index_wave64_guarded_loads(arch);

    if (amdgpu::supports_expert_scheduling(arch) && wavefront_size_ == 32 &&
        !cfg_has_unknown_successor_)
      index_repeated_exec_guards();

    // A pending event whose block dominates the current wait is structurally
    // guaranteed to have a path to that wait. Cache a compact dominator tree
    // so the path-sensitive scalar-predicate search remains reserved for
    // branch-correlated events. Large generated kernels otherwise repeat the
    // same whole-CFG predicate scan for thousands of ordinary wait fields.
    const size_t node_count = cfg_views_.size();
    const size_t virtual_root = node_count;
    std::vector<std::vector<size_t>> augmented_successors(node_count + 1);
    std::vector<uint8_t> virtual_root_child(node_count);
    for (size_t node = 0; node < node_count; ++node) {
      augmented_successors[node] = cfg_views_[node].successors;
      if (cfg_views_[node].predecessors.empty()) {
        augmented_successors[virtual_root].push_back(node);
        virtual_root_child[node] = 1;
      }
    }

    std::vector<uint8_t> visited(node_count + 1);
    std::vector<size_t> postorder;
    auto append_postorder = [&](size_t root) {
      std::vector<std::pair<size_t, size_t>> stack{{root, 0}};
      visited[root] = 1;
      while (!stack.empty()) {
        auto &[node, successor_index] = stack.back();
        if (successor_index < augmented_successors[node].size()) {
          const size_t successor = augmented_successors[node][successor_index++];
          if (!visited[successor]) {
            visited[successor] = 1;
            stack.emplace_back(successor, 0);
          }
          continue;
        }
        postorder.push_back(node);
        stack.pop_back();
      }
    };
    append_postorder(virtual_root);
    // Defensive support for malformed/disconnected executable CFG islands.
    // Treat each island as another child of the virtual root.
    bool added_island_root = false;
    for (size_t node = 0; node < node_count; ++node) {
      if (!visited[node]) {
        augmented_successors[virtual_root].push_back(node);
        virtual_root_child[node] = 1;
        added_island_root = true;
      }
    }
    if (added_island_root) {
      std::ranges::fill(visited, 0);
      postorder.clear();
      append_postorder(virtual_root);
    }

    std::vector<size_t> reverse_postorder(postorder.rbegin(), postorder.rend());
    // A predecessor closes a loop exactly when both ends of its edge
    // belong to one strongly connected component. The first DFS above
    // already produced the order needed for Kosaraju's reverse traversal.
    cfg_component_.assign(node_count, node_count);
    for (size_t root : reverse_postorder) {
      if (root == virtual_root || cfg_component_[root] != node_count)
        continue;
      std::vector<size_t> stack{root};
      cfg_component_[root] = root;
      while (!stack.empty()) {
        const size_t block = stack.back();
        stack.pop_back();
        for (size_t pred : cfg_views_[block].predecessors) {
          if (cfg_component_[pred] == node_count) {
            cfg_component_[pred] = root;
            stack.push_back(pred);
          }
        }
      }
    }
    std::vector<size_t> rpo_index(node_count + 1, node_count + 1);
    for (size_t index = 0; index < reverse_postorder.size(); ++index)
      rpo_index[reverse_postorder[index]] = index;
    std::vector<size_t> immediate_dominator(node_count + 1, node_count + 1);
    immediate_dominator[virtual_root] = virtual_root;
    auto intersect = [&](size_t lhs, size_t rhs) {
      while (lhs != rhs) {
        while (rpo_index[lhs] > rpo_index[rhs])
          lhs = immediate_dominator[lhs];
        while (rpo_index[rhs] > rpo_index[lhs])
          rhs = immediate_dominator[rhs];
      }
      return lhs;
    };
    bool changed = true;
    while (changed) {
      changed = false;
      for (size_t node : reverse_postorder) {
        if (node == virtual_root)
          continue;
        size_t new_dominator = node_count + 1;
        auto merge_predecessor = [&](size_t predecessor) {
          if (immediate_dominator[predecessor] == node_count + 1)
            return;
          new_dominator =
              new_dominator == node_count + 1 ? predecessor : intersect(new_dominator, predecessor);
        };
        if (virtual_root_child[node])
          merge_predecessor(virtual_root);
        for (size_t predecessor : cfg_views_[node].predecessors)
          merge_predecessor(predecessor);
        if (new_dominator != node_count + 1 && immediate_dominator[node] != new_dominator) {
          immediate_dominator[node] = new_dominator;
          changed = true;
        }
      }
    }

    std::vector<std::vector<size_t>> dominator_children(node_count + 1);
    for (size_t node = 0; node < node_count; ++node) {
      const size_t parent = immediate_dominator[node];
      if (parent <= node_count)
        dominator_children[parent].push_back(node);
    }
    dominator_preorder_.resize(node_count);
    dominator_subtree_end_.resize(node_count);
    size_t next_preorder = 0;
    std::vector<std::tuple<size_t, size_t, bool>> stack{{virtual_root, 0, false}};
    while (!stack.empty()) {
      auto &[node, child_index, entered] = stack.back();
      if (!entered) {
        entered = true;
        if (node != virtual_root)
          dominator_preorder_[node] = next_preorder++;
      }
      if (child_index < dominator_children[node].size()) {
        stack.emplace_back(dominator_children[node][child_index++], 0, false);
        continue;
      }
      if (node != virtual_root)
        dominator_subtree_end_[node] = next_preorder;
      stack.pop_back();
    }
  }

  void clear_cfg_path_filter() {
    cfg_has_repeated_exec_filter_ = false;
    cfg_views_.clear();
    feasible_path_cache_.clear();
    copy_lane_cache_.clear();
    path_scalar_defs_.clear();
    path_predicates_.clear();
    masked_predicates_.clear();
    repeated_masked_predicate_.reset();
    indexed_masked_predicates_ = false;
    indexed_loop_parity_ = false;
    loop_parity_.reset();
    path_query_starts_.clear();
    reverse_reachability_cache_.clear();
    forward_reachability_cache_.clear();
    reachability_cache_bytes_ = 0;
    cfg_component_.clear();
    dominator_preorder_.clear();
    dominator_subtree_end_.clear();
  }

  [[nodiscard]] bool cfg_block_dominates(size_t dominator, size_t node) const {
    if (dominator >= dominator_preorder_.size() || node >= dominator_preorder_.size())
      return false;
    return dominator_preorder_[dominator] <= dominator_preorder_[node] &&
           dominator_preorder_[node] < dominator_subtree_end_[dominator];
  }

  void record_diagnostic(WaitcheckDiagnostic diag) {
    if (current_kernel_) {
      diag.has_kernel = true;
      diag.kernel_name = current_kernel_->name;
      diag.kernel_entry_offset = current_kernel_->entry_offset;
    }
    const auto key =
        std::make_tuple(diag.counter, diag.access, diag.reg.cls, diag.reg.index, diag.reg.width,
                        diag.section_name, diag.section_offset, diag.file_offset,
                        diag.producer_section_offset, diag.producer_file_offset,
                        diag.required_count, diag.instruction, diag.producer_instruction);
    if (!diagnostic_keys_.insert(key).second)
      return;

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
  }

  using ResolvedCfgWait = std::pair<size_t, const Instruction *>;

  [[nodiscard]] std::optional<WaitFields>
  cfg_explicit_wait_fields(size_t block_index, const Instruction &inst, rj_code_arch_t arch) const {
    if (auto it = relative_cfg_waits_.find({block_index, &inst}); it != relative_cfg_waits_.end())
      return it->second;
    return explicit_wait_fields(inst, arch).value();
  }

  [[nodiscard]] std::optional<uint32_t> cfg_counter_wait(size_t block_index,
                                                         const Instruction &inst,
                                                         WaitCounterKind counter,
                                                         rj_code_arch_t arch) const {
    const size_t index = counter_index(counter);
    if (counter == WaitCounterKind::VmVsrc && releases_gfx10_scalar_sources(inst, arch))
      return 0;
    if (const auto fields = cfg_explicit_wait_fields(block_index, inst, arch);
        fields && (*fields)[index])
      return (*fields)[index];
    if (const auto fields = embedded_wait_fields(inst, arch).value(); fields && (*fields)[index])
      return (*fields)[index];
    return std::nullopt;
  }

  static void apply_counter_parity_wait(PendingState &state, WaitCounterKind counter,
                                        uint32_t count, rj_code_arch_t arch) {
    switch (counter) {
    case WaitCounterKind::Load:
    case WaitCounterKind::Store:
    case WaitCounterKind::Ds:
    case WaitCounterKind::Sample:
    case WaitCounterKind::Bvh:
    case WaitCounterKind::Async:
      (void)apply_memory_wait(state, counter, count, arch);
      break;
    case WaitCounterKind::Km:
      (void)apply_kmcnt_wait(state, count, arch);
      break;
    case WaitCounterKind::X:
      apply_xcnt_wait(state, count);
      break;
    case WaitCounterKind::Exp:
    case WaitCounterKind::Tensor:
    case WaitCounterKind::VmVsrc:
    case WaitCounterKind::VaVdst:
      (void)apply_counter_wait(state, counter, count, arch);
      break;
    case WaitCounterKind::Depctr:
    case WaitCounterKind::Count:
      break;
    }
  }

  // Preserve encoded joint waits when comparing a field against its omission.
  // Replaying the individual counters loses FLAT completion and its X prefix.
  static void apply_joint_parity_waits(PendingState &state, const PendingWaitGroup &group,
                                       WaitCounterKind audited, bool include_audited,
                                       rj_code_arch_t arch) {
    for (auto fields : group.joint_flat_waits) {
      if (!include_audited)
        fields[counter_index(audited)].reset();
      (void)apply_wait_fields(state, fields, arch);
    }
  }

  static bool is_vm_vsrc_primary_counter(WaitCounterKind counter) {
    return counter == WaitCounterKind::Load || counter == WaitCounterKind::Store ||
           counter == WaitCounterKind::Ds || counter == WaitCounterKind::Sample ||
           counter == WaitCounterKind::Bvh;
  }

  static void record_vm_vsrc_parity_companions(PendingWaitGroup &group, const WaitFields &fields,
                                               rj_code_arch_t arch) {
    if (!amdgpu::supports_expert_scheduling(arch) ||
        group.state_before.pending[counter_index(WaitCounterKind::VmVsrc)].empty())
      return;
    WaitFields primary{};
    for (size_t i = 0; i < fields.size(); ++i)
      if (is_vm_vsrc_primary_counter(static_cast<WaitCounterKind>(i)))
        primary[i] = fields[i];
    if (std::ranges::any_of(primary, [](auto count) { return count.has_value(); }))
      group.vm_vsrc_companions.push_back(primary);
  }

  [[nodiscard]] static bool has_counter_parity_candidate(const PendingState &state,
                                                         WaitCounterKind counter) {
    if (counter == WaitCounterKind::VaVdst)
      return !state.pending[counter_index(counter)].empty() ||
             !state.va_vdst_hazards.hazards.empty();
    if (counter == WaitCounterKind::Depctr || counter == WaitCounterKind::Count)
      return false;
    if (!state.pending[counter_index(counter)].empty())
      return true;

    // Split memory-counter waits may also retire an older VM_VSRC event, and
    // LOADCNT/KMCNT may imply an X_CNT wait. A future dependency can justify
    // the current wait only when one of those events already existed before
    // the wait. Events issued after it cannot retroactively make it useful.
    switch (counter) {
    case WaitCounterKind::Load:
    case WaitCounterKind::Store:
    case WaitCounterKind::Ds:
    case WaitCounterKind::Sample:
    case WaitCounterKind::Bvh:
      return !state.pending[counter_index(WaitCounterKind::VmVsrc)].empty() ||
             !state.pending[counter_index(WaitCounterKind::X)].empty();
    case WaitCounterKind::Km:
    case WaitCounterKind::Async:
      return !state.pending[counter_index(WaitCounterKind::X)].empty();
    case WaitCounterKind::Exp:
    case WaitCounterKind::X:
    case WaitCounterKind::Tensor:
    case WaitCounterKind::VmVsrc:
    case WaitCounterKind::VaVdst:
    case WaitCounterKind::Depctr:
    case WaitCounterKind::Count:
      return false;
    }
    return false;
  }

  [[nodiscard]] static bool counter_may_retire_event(WaitCounterKind emitted_counter,
                                                     const PendingEvent &event) {
    if (event.counter == emitted_counter)
      return true;
    if (event.counter == WaitCounterKind::VmVsrc) {
      return emitted_counter == WaitCounterKind::Load ||
             emitted_counter == WaitCounterKind::Store || emitted_counter == WaitCounterKind::Ds ||
             emitted_counter == WaitCounterKind::Sample || emitted_counter == WaitCounterKind::Bvh;
    }
    return event.counter == WaitCounterKind::X &&
           (emitted_counter == WaitCounterKind::Load || emitted_counter == WaitCounterKind::Store ||
            emitted_counter == WaitCounterKind::Ds || emitted_counter == WaitCounterKind::Sample ||
            emitted_counter == WaitCounterKind::Bvh || emitted_counter == WaitCounterKind::Km ||
            emitted_counter == WaitCounterKind::Async);
  }

  [[nodiscard]] static std::optional<uint32_t>
  implied_counter_requirement(const PendingState &state, const PendingEvent &event,
                              WaitCounterKind emitted_counter, rj_code_arch_t arch) {
    const auto no_wait = counter_no_wait_value(arch, emitted_counter).value();
    if (!no_wait || event.counter == emitted_counter ||
        !counter_may_retire_event(emitted_counter, event))
      return std::nullopt;

    auto is_sufficient = [&](uint32_t count) {
      PendingState after = state;
      apply_counter_parity_wait(after, emitted_counter, count, arch);
      const auto &pending = after.pending[counter_index(event.counter)];
      return !std::ranges::any_of(pending, [&](const PendingEvent &candidate) {
        return same_event_identity(candidate, event);
      });
    };
    if (!is_sufficient(0))
      return std::nullopt;

    // Sufficiency is monotonic: a lower counter is always at least as strong.
    // Find the weakest sufficient value without copying a potentially large
    // pending state once for every representable counter value.
    uint32_t lower = 0;
    uint32_t upper = *no_wait;
    while (lower + 1 < upper) {
      const uint32_t middle = lower + (upper - lower) / 2;
      if (is_sufficient(middle))
        lower = middle;
      else
        upper = middle;
    }
    return lower;
  }

  void record_counter_underaccounting(WaitcheckCounterUnderaccountingDiagnostic diag) {
    if (current_kernel_) {
      diag.has_kernel = true;
      diag.kernel_name = current_kernel_->name;
      diag.kernel_entry_offset = current_kernel_->entry_offset;
    }
    const auto key =
        std::make_tuple(diag.kernel_name, diag.counter, diag.wait_section_offset,
                        diag.consumer_section_offset, diag.producer_section_offset,
                        diag.emitted_count, diag.required_count, diag.has_required_dependency);
    if (!counter_underaccounting_keys_.insert(key).second)
      return;

    ++report_.counter_underaccounting_observed;
    if (!diag.has_required_dependency)
      ++report_.counter_unmodeled_wait_observed;
    if (report_.counter_underaccounting_diagnostics.size() <
        options_.max_counter_parity_diagnostics) {
      report_.counter_underaccounting_diagnostics.push_back(std::move(diag));
    } else {
      report_.counter_parity_diagnostics_truncated = true;
    }
  }

  static bool cdna5_device_cache_op(const Instruction &instruction, std::string_view opcode) {
    if (instruction.mnemonic() != opcode || !instruction.raw_encoding() ||
        instruction.size() != sizeof(cdna5::VglobalMachineInst))
      return false;
    cdna5::VglobalMachineInst encoding{};
    std::memcpy(&encoding, instruction.raw_encoding(), sizeof(encoding));
    return encoding.scope >= 2; // SCOPE_DEV or SCOPE_SYS
  }

  bool cdna5_cache_wait_matches(const PendingWaitGroup &group, const Instruction &consumer,
                                WaitCounterKind counter, rj_code_arch_t arch) const {
    if (arch != ROCJITSU_CODE_ARCH_CDNA5 ||
        (counter != WaitCounterKind::Load && counter != WaitCounterKind::Store) ||
        !consumer.parent())
      return false;
    const auto idx = counter_index(counter);
    const auto &ages = group.state_before.pending_event_ages[idx].values;
    if (std::ranges::all_of(ages, [](uint8_t age) { return age == kNoPendingEventAge; }))
      return false;

    // Cache scope and the affected operation both matter. The compact queue
    // state alone cannot distinguish a scratch store from a global store,
    // or a returning atomic from an ordinary read on LOADCNT.
    const auto &instructions = consumer.parent()->instructions();
    auto it = decltype(instructions.begin())(&consumer);
    const auto end = instructions.end();
    auto transparent = [](const Instruction &inst) {
      if (!(inst.flags() & (VALU | SALU)) ||
          (inst.flags() &
           (MEMORY_OP | MEMORY_WAIT_PRODUCER | EMBEDDED_MEMORY_WAIT | BARRIER | BRANCH |
            COND_BRANCH | INDIRECT_CALL | INDIRECT_BRANCH | PROGRAM_TERMINATOR | WAITCNT)))
        return false;
      const auto op = inst.mnemonic();
      return !op.starts_with("s_setreg") && !op.starts_with("s_sendmsg") &&
             !op.starts_with("s_trap") && !op.starts_with("s_wait");
    };

    unsigned steps = 0;
    for (; it != end && steps < 32; ++it, ++steps) {
      if ((*it).flags() & WAITCNT) {
        const auto fields = explicit_wait_fields(*it, arch).value();
        if (fields && !(*fields)[idx])
          continue; // An independent wait cannot retire this cache operation.
      }
      if (!transparent(*it))
        break;
    }
    if (it == end)
      return false;
    const auto next = (*it).mnemonic();
    const bool writeback = cdna5_device_cache_op(*it, "global_wb");
    const bool follow_load = next.starts_with("global_load_") || next.starts_with("buffer_load_") ||
                             next.starts_with("tbuffer_load_");
    if (!writeback && !(counter == WaitCounterKind::Load && follow_load))
      return false;

    // Work backwards within this basic block, across scalar/vector arithmetic
    // and the wait group itself. A prior wait on this field, branch, or memory
    // operation for which we have no ordering proof ends the search.
    it = decltype(instructions.begin())(&consumer);
    for (unsigned step = 0; step < 64 && it != instructions.begin(); ++step) {
      const Instruction &previous = *--it;
      if (previous.src_loc() >= group.section_offset) {
        // An embedded wait can share the consumer, but memory between a wait
        // group and its first consumer is never part of that wait group.
        if (!(previous.flags() & WAITCNT))
          return false;
        continue;
      }
      if (transparent(previous))
        continue;
      if (previous.flags() & WAITCNT) {
        const auto fields = explicit_wait_fields(previous, arch).value();
        if (!fields || (*fields)[idx])
          return false;
        continue;
      }
      const auto op = previous.mnemonic();
      if (!writeback)
        return cdna5_device_cache_op(previous, "global_inv");
      if (!(op.starts_with("global_") || op.starts_with("buffer_") || op.starts_with("tbuffer_")))
        return false;
      if (!is_vmem_store(op) && !is_vmem_atomic(op))
        return false;
      // Global reads never dirty L2. Only writes and read-modify-writes
      // have data that a release writeback needs to publish.
      const auto events = effective_events(previous, {.known = false}, arch);
      return std::ranges::any_of(
          events, [counter](const ClassifiedEvent &event) { return event.counter == counter; });
    }
    return false;
  }

  bool legacy_buffer_wbl2_wait_matches(const PendingWaitGroup &group, WaitCounterKind counter,
                                       rj_code_arch_t arch) {
    if ((arch != ROCJITSU_CODE_ARCH_CDNA2 && arch != ROCJITSU_CODE_ARCH_CDNA3 &&
         arch != ROCJITSU_CODE_ARCH_CDNA4) ||
        counter != WaitCounterKind::Load)
      return false;
    // BUFFER_WBL2 raises legacy VM_CNT and its ACK must complete before a
    // release can proceed. A register dependency on an older load may need
    // only vmcnt(1), but cannot justify weakening this vmcnt(0).
    return std::ranges::any_of(group.state_before.pending[counter_index(counter)],
                               [&](const PendingEvent &event) {
                                 return event.kind == WaitEventKind::GlobalWb &&
                                        event.instruction.starts_with("buffer_wbl2") &&
                                        event_can_reach_wait(event, group.section_offset, arch);
                               });
  }

  bool needs_constant_overwrite_parity_path(const CounterParityRequirement &candidate,
                                            const Instruction &immediate_consumer,
                                            WaitCounterKind counter, uint32_t emitted_count,
                                            rj_code_arch_t arch) const {
    if (candidate.required_count != emitted_count || candidate.event.counter != counter ||
        candidate.event.present_on_all_paths || candidate.consumer != &immediate_consumer ||
        candidate.access != WaitcheckAccessKind::Def || candidate.reg.cls != RegClass::VGPR ||
        immediate_consumer.mnemonic() != "v_mov_b32_e32" ||
        !access_requires_active_lane(immediate_consumer, candidate.reg) ||
        guarded_lane_operand(candidate.event, immediate_consumer, candidate.reg,
                             candidate.access, arch) ||
        has_guarded_scratch_result(candidate.event))
      return false;
    const auto *source = immediate_consumer.src_operand(0);
    return source && source->const_value().has_value();
  }

  void collect_counter_parity_requirements(
      const PendingWaitGroup &group, const Instruction &consumer, const InstDefUse &du,
      uint64_t consumer_section_offset, uint64_t consumer_file_offset, rj_code_arch_t arch,
      std::array<std::optional<CounterParityRequirement>, kCounterCount> &requirements,
      bool past_xcnt_drain = false, bool forward_search = false,
      CounterParityAlternatives *alternatives = nullptr) {
    const PendingState *requirement_state = &group.state_before;
    auto current_events = effective_events(consumer, requirement_state->vgpr_msb, arch);
    if (!expert_waits_enabled(*requirement_state)) {
      std::erase_if(current_events, [](const ClassifiedEvent &event) {
        return event.requires_expert_scheduling();
      });
    }
    const bool prefetch =
        arch == ROCJITSU_CODE_ARCH_CDNA5 && is_scalar_prefetch(consumer.mnemonic());
    if (prefetch && requirement_state->prefetch_enabled == false)
      return;
    std::optional<PendingState> xcnt_adjusted_state;
    if (!prefetch && std::ranges::any_of(current_events, [](const ClassifiedEvent &event) {
          return event.counter == WaitCounterKind::X;
        })) {
      xcnt_adjusted_state.emplace(group.state_before);
      apply_implicit_xcnt_ordering(*xcnt_adjusted_state, consumer, du, current_events, arch);
      requirement_state = &*xcnt_adjusted_state;
    }
    std::optional<PendingState> delay_adjusted_state;
    if (!forward_search && (consumer.flags() & VALU) && requirement_state->expert_delay.pending &&
        expert_waits_enabled(*requirement_state)) {
      delay_adjusted_state.emplace(*requirement_state);
      apply_expert_valu_delay(*delay_adjusted_state, consumer, arch,
                              requirement_state->expert_valu_may_skip &&
                                  !expert_valu_issues_without_active_lanes(consumer, du.defs));
      requirement_state = &*delay_adjusted_state;
    }
    std::optional<PendingState> x_without_field;
    if (arch == ROCJITSU_CODE_ARCH_CDNA5 &&
        group.emitted_counts[counter_index(WaitCounterKind::X)] &&
        !requirement_state->pending[counter_index(WaitCounterKind::X)].empty()) {
      for (size_t i = 0; i < group.emitted_counts.size(); ++i) {
        if (i == counter_index(WaitCounterKind::X) || !group.emitted_counts[i])
          continue;
        if (!x_without_field) {
          x_without_field = *requirement_state;
          apply_joint_parity_waits(*x_without_field, group, WaitCounterKind::X, false, arch);
        }
        apply_counter_parity_wait(*x_without_field, static_cast<WaitCounterKind>(i),
                                  *group.emitted_counts[i], arch);
      }
    }
    // If the companion waits leave any prefetch translation live, X must
    // still protect it. Keep that encoded X count when deciding whether
    // a companion is independently needed for the other translations.
    const bool prefetch_needs_x =
        prefetch && x_without_field &&
        std::ranges::any_of(x_without_field->pending[counter_index(WaitCounterKind::X)],
                            [&](const PendingEvent &event) {
                              return event_can_reach_wait(event, group.section_offset, arch);
                            });
    auto record_dependency = [&](const PendingEvent &event, RegisterRef reg,
                                 WaitcheckAccessKind access, uint32_t required_count) {
      // A branch, bank update, or other implicit X wait retires replay
      // sources before this instruction observes or changes registers.
      if (event.counter == WaitCounterKind::X &&
          (past_xcnt_drain || (arch == ROCJITSU_CODE_ARCH_CDNA5 && is_xcnt_drain(consumer))))
        return;
      bool can_improve = CounterParityRequirement::can_improve(
          requirements[counter_index(event.counter)], required_count);
      const auto can_retain_equal = [&](WaitCounterKind counter, uint32_t count) {
        const auto index = counter_index(counter);
        const auto &current = requirements[index];
        return alternatives && counter == event.counter && group.emitted_counts[index] &&
               current && current->required_count == count &&
               needs_constant_overwrite_parity_path(*current, consumer, counter,
                                                    *group.emitted_counts[index], arch);
      };
      can_improve |= can_retain_equal(event.counter, required_count);
      for (size_t counter_idx = 0; !can_improve && counter_idx < group.emitted_counts.size();
           ++counter_idx) {
        if (!group.emitted_counts[counter_idx])
          continue;
        const auto emitted_counter = static_cast<WaitCounterKind>(counter_idx);
        const auto &current = requirements[counter_idx];
        can_improve = counter_may_retire_event(emitted_counter, event) &&
                      (!current || current->required_count != 0);
      }
      if (!can_improve)
        return;
      if (!event_can_reach_wait(event, group.section_offset, arch))
        return;
      auto record_requirement = [&](WaitCounterKind counter, uint32_t count) {
        const auto index = counter_index(counter);
        auto &strongest = requirements[index];
        if (!CounterParityRequirement::can_improve(strongest, count) &&
            can_retain_equal(counter, count)) {
          auto &pool = alternatives->values[index];
          if (pool.size() < kMaxParityAlternatives)
            pool.push_back(CounterParityRequirement{event, reg, access, count, &consumer,
                                                    consumer_section_offset, consumer_file_offset});
          return;
        }
        if (CounterParityRequirement::can_improve(strongest, count)) {
          CounterParityRequirement candidate{
              event, reg, access, count, &consumer, consumer_section_offset, consumer_file_offset};
          if (counter == event.counter) {
            if (counter == WaitCounterKind::VmVsrc && group.vm_vsrc_without_field) {
              const auto &sources = group.vm_vsrc_without_field->pending[counter_index(counter)];
              if (find_event(sources, event) == sources.end())
                return;
            }
            const auto witness = cross_lane_parity_witness(group, candidate, arch);
            if (witness == false)
              return;
            candidate.witness_unknown = !witness.has_value();
            if (!refine_guarded_load_count(group, arch, candidate))
              return;
          }
          if (CounterParityRequirement::can_improve(strongest, candidate.required_count)) {
            strongest = std::move(candidate);
            if (alternatives)
              alternatives->values[index].clear();
          }
        }
      };
      const bool translation_order = access == WaitcheckAccessKind::MemoryOrder &&
                                     event.counter == WaitCounterKind::X &&
                                     (prefetch || is_vmem_atomic(consumer.mnemonic()));
      const bool direct =
          event.counter != WaitCounterKind::X || !x_without_field ||
          std::ranges::any_of(
              x_without_field->pending[counter_index(WaitCounterKind::X)],
              [&](const PendingEvent &remaining) { return same_event_identity(remaining, event); });
      if (direct)
        record_requirement(event.counter, required_count);
      if (translation_order && (prefetch_needs_x || !prefetch) &&
          group.emitted_counts[counter_index(WaitCounterKind::X)].value_or(UINT32_MAX) <=
              required_count)
        return;

      // A split GFX12 memory-counter wait may also retire the
      // corresponding VM_VSRC or X_CNT event. Attribute that
      // dependency to the emitted field so a compiler using
      // the implied form compares against its effective
      // semantic strength rather than looking redundant.
      for (size_t counter_idx = 0; counter_idx < group.emitted_counts.size(); ++counter_idx) {
        if (!group.emitted_counts[counter_idx])
          continue;
        const auto emitted_counter = static_cast<WaitCounterKind>(counter_idx);
        if (const auto implied =
                implied_counter_requirement(*requirement_state, event, emitted_counter, arch)) {
          record_requirement(emitted_counter, *implied);
        }
      }
    };
    if (prefetch && !past_xcnt_drain && !forward_search) {
      for (const auto &event : requirement_state->pending[counter_index(WaitCounterKind::X)])
        record_dependency(event, {RegClass::PC, 0, 1}, WaitcheckAccessKind::MemoryOrder,
                          dependency_required_count(*requirement_state, event, arch).value());
    }
    visit_dependencies(*requirement_state, consumer, du, current_events, arch, record_dependency);

    if (arch == ROCJITSU_CODE_ARCH_RDNA3) {
      auto &requirement = requirements[counter_index(WaitCounterKind::VaVdst)];
      if (!requirement || requirement->required_count != 0) {
        if (auto trans = trans_use_requirement(requirement_state->trans_use_hazards, consumer, du,
                                               consumer_section_offset, consumer_file_offset))
          requirement = std::move(trans);
      }
    }

    if (expert_waits_enabled(*requirement_state) && dsdir_wait_va_vdst(consumer)) {
      auto &strongest = requirements[counter_index(WaitCounterKind::VaVdst)];
      du.defs.for_each([&](RegisterRef ref) {
        if (ref.cls != RegClass::VGPR || ref.index >= REGISTER_SET_MAX_VGPRS)
          return;
        const auto [hazard, required_count] =
            requirement_state->va_vdst_hazards.requirement(ref.index);
        if (!hazard || (strongest && strongest->required_count <= required_count))
          return;

        PendingEvent producer;
        producer.counter = WaitCounterKind::VaVdst;
        producer.section_name = hazard->producer.section_name;
        producer.section_offset = hazard->producer.section_offset;
        producer.file_offset = hazard->producer.file_offset;
        producer.instruction = hazard->producer.instruction;
        strongest =
            CounterParityRequirement{std::move(producer), ref,       WaitcheckAccessKind::Def,
                                     required_count,      &consumer, consumer_section_offset,
                                     consumer_file_offset};
      });
    }

    if (amdgpu::uses_legacy_waitcnt(amdgpu::waitcnt_model(arch).value()) &&
        consumer.mnemonic() == "s_barrier") {
      // LLVM synchronization lowering may attach a local-memory ordering
      // requirement to a barrier even when the final instruction has no
      // explicit register operands. Attribute a legacy LGKM drain to the
      // strongest outstanding LGKM event instead of losing that final-ISA
      // intent as an unmodeled wait.
      auto &strongest = requirements[counter_index(WaitCounterKind::Ds)];
      for (const PendingEvent &event :
           requirement_state->pending[counter_index(WaitCounterKind::Ds)]) {
        if (!event_can_reach_wait(event, group.section_offset, arch))
          continue;
        const uint32_t required_count =
            dependency_required_count(*requirement_state, event, arch).value();
        if (!strongest || required_count < strongest->required_count) {
          strongest = CounterParityRequirement{event,
                                               RegisterRef{RegClass::PC, 0, 1},
                                               WaitcheckAccessKind::MemoryOrder,
                                               required_count,
                                               &consumer,
                                               consumer_section_offset,
                                               consumer_file_offset};
        }
      }
    }

    if (!amdgpu::uses_legacy_waitcnt(amdgpu::waitcnt_model(arch).value()) &&
        starts_with(consumer.mnemonic(), "s_barrier_signal")) {
      // Split barriers do not encode an implicit DS drain. LLVM-produced
      // synchronization sequences therefore place s_wait_dscnt 0 before the
      // signal when LDS operations are outstanding. Attribute that field to
      // the barrier's memory-order requirement; otherwise forward lookahead
      // can incorrectly attach the deliberately strong wait to a register
      // dependency after the matching barrier wait.
      auto &strongest = requirements[counter_index(WaitCounterKind::Ds)];
      for (const PendingEvent &event :
           requirement_state->pending[counter_index(WaitCounterKind::Ds)]) {
        if (!event_can_reach_wait(event, group.section_offset, arch))
          continue;
        strongest = CounterParityRequirement{event,
                                             RegisterRef{RegClass::PC, 0, 1},
                                             WaitcheckAccessKind::MemoryOrder,
                                             0,
                                             &consumer,
                                             consumer_section_offset,
                                             consumer_file_offset};
        break;
      }
    }

    if (!amdgpu::uses_legacy_waitcnt(amdgpu::waitcnt_model(arch).value()) &&
        consumer.mnemonic() == "global_inv") {
      // LLVM emits zero waits before GLOBAL_INV for a cross-address-space acquire.
      constexpr std::array fence_counters{WaitCounterKind::Load, WaitCounterKind::Store,
                                          WaitCounterKind::Ds, WaitCounterKind::Sample,
                                          WaitCounterKind::Bvh};
      for (WaitCounterKind counter : fence_counters) {
        const size_t counter_idx = counter_index(counter);
        if (!group.emitted_counts[counter_idx])
          continue;
        auto &strongest = requirements[counter_idx];
        for (const PendingEvent &event : requirement_state->pending[counter_idx]) {
          if (!event_can_reach_wait(event, group.section_offset, arch))
            continue;
          strongest = CounterParityRequirement{event,
                                               RegisterRef{RegClass::PC, 0, 1},
                                               WaitcheckAccessKind::MemoryOrder,
                                               0,
                                               &consumer,
                                               consumer_section_offset,
                                               consumer_file_offset};
          break;
        }
      }
    }

    if (consumer.mnemonic() == "s_barrier" || starts_with(consumer.mnemonic(), "ds_")) {
      for (size_t counter_idx = 0; counter_idx < requirement_state->pending.size(); ++counter_idx) {
        auto &strongest = requirements[counter_idx];
        for (const PendingEvent &event : requirement_state->pending[counter_idx]) {
          if (!event.check_counter_parity_order ||
              !event_can_reach_wait(event, group.section_offset, arch)) {
            continue;
          }
          const uint32_t required_count =
              dependency_required_count(*requirement_state, event, arch).value();
          if (!strongest || required_count < strongest->required_count) {
            strongest = CounterParityRequirement{event,
                                                 RegisterRef{RegClass::PC, 0, 1},
                                                 WaitcheckAccessKind::MemoryOrder,
                                                 required_count,
                                                 &consumer,
                                                 consumer_section_offset,
                                                 consumer_file_offset};
          }
        }
      }
    }
  }

  [[nodiscard]] std::optional<CounterParityRequirement> find_forward_counter_parity_requirement(
      const PendingWaitGroup &group, const Instruction &first_consumer,
      uint64_t first_consumer_section_offset, WaitCounterKind counter, uint32_t emitted_count,
      rj_code_arch_t arch, bool function_boundary, bool &complete, bool &reached_partial_wait,
      bool &reached_expert_completion) {
    if (!current_cfg_view_index_ || *current_cfg_view_index_ >= cfg_views_.size())
      return std::nullopt;

    const CfgBlockView &start_block = cfg_views_[*current_cfg_view_index_];
    const auto start_it = std::ranges::find_if(start_block.instructions, [&](const auto &view) {
      return view.instruction == &first_consumer &&
             view.section_offset == first_consumer_section_offset;
    });
    if (start_it == start_block.instructions.end())
      return std::nullopt;

    RegisterSet possible_uses;
    RegisterSet possible_defs;
    bool may_define_exec = false;
    bool may_use_special = false;
    bool may_define_special = false;
    bool may_require_memory_order = false;
    bool may_require_program_end = false;
    bool may_require_counter_order = false;
    for (const auto &events : group.state_before.pending) {
      for (const PendingEvent &event : events) {
        if (!counter_may_retire_event(counter, event))
          continue;
        if (event.check_uses)
          possible_uses |= event.regs;
        if (event.check_defs)
          possible_defs |= event.regs;
        may_define_exec |= event.check_exec_defs;
        may_use_special |= event.check_uses && event.special_reg.has_value();
        may_define_special |= event.check_defs && event.special_reg.has_value();
        may_require_memory_order |= event.check_memory_order;
        may_require_program_end |= event.check_program_end;
        may_require_counter_order |= event.check_counter_parity_order;
      }
    }

    const bool may_track_xcnt =
        arch == ROCJITSU_CODE_ARCH_CDNA5 &&
        !group.state_before.pending[counter_index(WaitCounterKind::X)].empty();
    const bool may_retire_valu = counter == WaitCounterKind::VaVdst &&
                                 expert_waits_enabled(group.state_before) && cfg_has_expert_delay_;
    const bool may_retire_source = counter == WaitCounterKind::VmVsrc &&
                                   amdgpu::supports_expert_scheduling(arch) &&
                                   expert_waits_enabled(group.state_before);
    reached_expert_completion |= may_retire_valu && group.state_before.expert_delay.pending;
    using Position = std::tuple<size_t, size_t, bool>;
    // At ABI boundaries spend the short search budget on both sides of a
    // branch before a long path can hide a dependency in a nearby caller.
    const size_t max_lookahead_positions =
        function_boundary ? kBoundaryLookaheadLimit : kParityLookaheadLimit;
    std::deque<Position> worklist{{
        *current_cfg_view_index_,
        static_cast<size_t>(std::distance(start_block.instructions.begin(), start_it)),
        false,
    }};
    std::set<Position> visited;
    std::set<std::tuple<const Instruction *, bool, bool, uint8_t>> dependencies_checked;
    std::optional<CounterParityRequirement> strongest;
    while (!worklist.empty() && visited.size() < max_lookahead_positions) {
      const auto [block_index, instruction_index, past_xcnt_drain] =
          function_boundary ? worklist.front() : worklist.back();
      if (function_boundary)
        worklist.pop_front();
      else
        worklist.pop_back();
      if (!visited.insert({block_index, instruction_index, past_xcnt_drain}).second)
        continue;

      const CfgBlockView &block = cfg_views_[block_index];
      if (instruction_index >= block.instructions.size()) {
        for (size_t successor : block.successors)
          worklist.emplace_back(successor, 0, past_xcnt_drain);
        continue;
      }

      const CfgInstructionView &view = block.instructions[instruction_index];
      const Instruction &instruction = *view.instruction;
      if (instruction.is_vskip_affected() && view.vector_skip != false) {
        if (!view.vector_skip.has_value()) {
          complete = false;
          return std::nullopt;
        }
        worklist.emplace_back(block_index, instruction_index + 1, past_xcnt_drain);
        continue;
      }
      // The explicit wait separates the first looked-ahead bank change from
      // MODE writes. Any later adjacent MODE write itself already drains X.
      const bool xcnt_drained = past_xcnt_drain || (may_track_xcnt && is_xcnt_drain(instruction));
      if (counter == WaitCounterKind::X && xcnt_drained)
        continue;
      const auto wait_count = cfg_counter_wait(block_index, instruction, counter, arch);
      if (wait_count) {
        reached_partial_wait |= *wait_count != 0;
        if (!relative_wait_operand(instruction, arch))
          continue;
      }
      if (!reached_expert_completion) {
        if (may_retire_valu && instruction.mnemonic() == "s_delay_alu") {
          reached_expert_completion = true;
        } else if (may_retire_source && instruction.is_waitcnt()) {
          if (const auto fields = cfg_explicit_wait_fields(block_index, instruction, arch)) {
            for (size_t i = 0; i < fields->size(); ++i)
              reached_expert_completion |=
                  (*fields)[i] && is_vm_vsrc_primary_counter(static_cast<WaitCounterKind>(i));
          }
        }
      }

      // A physical VGPR operand uses the bank selected at this instruction. A
      // helper reached through two calls can have different incoming banks.
      if (dependencies_checked
              .emplace(&instruction, xcnt_drained, view.vgpr_msb.known, view.vgpr_msb.mode)
              .second) {
        const InstDefUse du =
            inst_def_use_for_waitcheck(instruction, view.vgpr_msb, arch, wavefront_size_);
        const bool may_depend_on_register =
            possible_uses.intersects(du.uses) || possible_defs.intersects(du.defs) ||
            (possible_uses.contains({RegClass::VCC, 0, 1}) &&
             vcc_access_halves(instruction, du, WaitcheckAccessKind::Use, arch, wavefront_size_)) ||
            (possible_defs.contains({RegClass::VCC, 0, 1}) &&
             vcc_access_halves(instruction, du, WaitcheckAccessKind::Def, arch, wavefront_size_));
        const bool may_depend_on_special =
            (may_use_special &&
             instruction_uses_special(instruction, RegisterRef{RegClass::SCC, 0, 1})) ||
            (may_define_special &&
             instruction_defines_special(instruction, RegisterRef{RegClass::SCC, 0, 1}));
        const bool may_depend_on_exec = may_define_exec && writes_exec_mask(instruction);
        const bool is_barrier = instruction.mnemonic() == "s_barrier";
        const bool is_ds = starts_with(instruction.mnemonic(), "ds_");
        const bool may_depend_on_order =
            may_require_memory_order ||
            (may_require_program_end && is_program_end(instruction.mnemonic())) ||
            (may_require_counter_order && (is_barrier || is_ds)) ||
            (counter == WaitCounterKind::Ds &&
             instruction.mnemonic().starts_with("s_barrier_signal")) ||
            (may_track_xcnt && counter == WaitCounterKind::X &&
             (is_scalar_prefetch(instruction.mnemonic()) ||
              is_vmem_atomic(instruction.mnemonic()))) ||
            instruction.mnemonic() == "global_inv" ||
            (amdgpu::uses_legacy_waitcnt(amdgpu::waitcnt_model(arch).value()) &&
             counter == WaitCounterKind::Ds && is_barrier) ||
            (counter == WaitCounterKind::VaVdst && dsdir_wait_va_vdst(instruction));
        if (may_depend_on_register || may_depend_on_special || may_depend_on_exec ||
            may_depend_on_order) {
          std::array<std::optional<CounterParityRequirement>, kCounterCount> requirements;
          collect_counter_parity_requirements(
              group, instruction, du, view.section_offset,
              group.file_offset - group.section_offset + view.section_offset, arch, requirements,
              xcnt_drained, /*forward_search=*/true);
          if (const auto &requirement = requirements[counter_index(counter)]) {
            if (CounterParityRequirement::can_improve(strongest, requirement->required_count))
              strongest = requirement;
            // Counter-parity auditing is intentionally one-sided. As soon as
            // one guarded dependency requires a value no weaker than the
            // emitted field, this wait cannot be an N > M under-accounting
            // candidate. Avoid scanning the rest of a large CFG solely to
            // discover an even stronger ordinary-hazard requirement.
            if (!strongest->witness_unknown && strongest->required_count <= emitted_count)
              return strongest;
          }
        }
      }
      if (wait_count || is_program_end(instruction.mnemonic()))
        continue;
      worklist.emplace_back(block_index, instruction_index + 1, xcnt_drained);
    }
    if (!worklist.empty())
      complete = false;
    return strongest;
  }

  [[nodiscard]] bool explicit_va_vdst_zero(size_t block, const Instruction &inst,
                                           rj_code_arch_t arch) const {
    const auto fields = cfg_explicit_wait_fields(block, inst, arch);
    return fields && (*fields)[counter_index(WaitCounterKind::VaVdst)] == 0u;
  }

  [[nodiscard]] bool tracks_partial_forwarding(rj_code_arch_t arch) const {
    return wavefront_size_ == 64 &&
           (arch == ROCJITSU_CODE_ARCH_RDNA3 || arch == ROCJITSU_CODE_ARCH_RDNA3_5);
  }

  static constexpr uint8_t kMaxPartialForwardingWriterAge = 7;

  // An RDNA3 wave64 can partially forward two VGPR operands across a scalar
  // EXEC write. Walk backwards on individual CFG paths: combining register
  // writers or EXEC from different predecessors would produce a false hazard.
  [[nodiscard]] std::optional<CounterParityRequirement>
  partial_forwarding_requirement(const Instruction &consumer, uint64_t section_offset,
                                 uint64_t file_offset, const std::string &section_name,
                                 rj_code_arch_t arch, size_t consumer_block,
                                 const PendingWaitGroup *wait = nullptr) {
    if (!tracks_partial_forwarding(arch) || !cfg_has_salu_exec_write_ || !is_vector_alu(consumer))
      return std::nullopt;
    struct Source {
      RegisterRef reg;
      uint8_t bytes;
      bool operator==(const Source &) const = default;
    };
    std::vector<Source> sources;
    for (int i = 0; i < consumer.num_src_operands(); ++i) {
      const auto *op = consumer.src_operand(i);
      const auto reg = op ? op->to_register_ref() : std::nullopt;
      if (!reg || reg->cls != RegClass::VGPR || !consumer.tracks_explicit_source(i))
        continue;
      const Source source{*reg, valu_operand_bytes(consumer, *op, i)};
      if (std::ranges::find(sources, source) == sources.end())
        sources.push_back(source);
    }
    if (sources.size() < 2)
      return std::nullopt;

    const auto &block = cfg_views_[consumer_block];
    const auto pos = std::ranges::lower_bound(block.instructions, section_offset, {},
                                              &CfgInstructionView::section_offset);
    if (pos == block.instructions.end() || pos->instruction != &consumer)
      return std::nullopt;
    // At most eight preceding VALUs can participate. For each source keep its
    // most recent overlapping VALU writer. LLVM's window is <=4 VALUs after
    // the younger writer and <=2 *between* the two writers.
    constexpr uint8_t unseen = 0xff;
    struct History {
      uint8_t age = 0;
      uint8_t exec = unseen;
      std::vector<uint8_t> writers;
      bool crossed_wait = false;
      auto operator<=>(const History &) const = default;
    };
    struct Cursor {
      size_t block;
      size_t end;
      History history;
    };
    std::vector<Cursor> work{{consumer_block,
                              size_t(pos - block.instructions.begin()),
                              {0, unseen, std::vector<uint8_t>(sources.size(), unseen), false}}};
    std::set<std::pair<size_t, History>> at_entry;
    while (!work.empty()) {
      auto [block_index, end, history] = std::move(work.back());
      work.pop_back();
      const auto &view = cfg_views_[block_index];
      bool expired = false;
      while (end && !expired) {
        const auto &entry = view.instructions[--end];
        const Instruction &inst = *entry.instruction;
        if (wait && block_index == wait->cfg_block && entry.section_offset == wait->section_offset)
          history.crossed_wait = true;
        if (is_va_vdst_expiring_instruction(inst.mnemonic())) {
          expired = true;
          break;
        }
        if (((inst.is_waitcnt() && explicit_va_vdst_zero(block_index, inst, arch)) ||
             dsdir_wait_va_vdst(inst) == 0u) &&
            !(wait && block_index == wait->cfg_block &&
              entry.section_offset >= wait->section_offset &&
              entry.section_offset <= wait->last_wait_section_offset)) {
          expired = true;
          break;
        }
        if (!is_vector_alu(inst)) {
          if (history.exec == unseen && inst.mnemonic().starts_with("s_") &&
              entry.invalidates_exec &&
              std::ranges::any_of(history.writers, [=](uint8_t a) { return a != unseen; }))
            history.exec = history.age;
          continue;
        }
        const auto du = inst_def_use_for_waitcheck(inst, entry.vgpr_msb, arch, wavefront_size_);
        for (size_t i = 0; i < sources.size(); ++i) {
          if (history.writers[i] != unseen || !du.defs.intersects(sources[i].reg) ||
              (sources[i].bytes != 0xf &&
               !(trans_operand_bytes(inst, sources[i].reg, false) & sources[i].bytes)))
            continue;
          history.writers[i] = history.age;
          if (history.exec == unseen || (wait && !history.crossed_wait))
            continue;
          const auto post = std::ranges::min(history.writers);
          if (post >= history.exec || post > 4 || history.age - post > 3)
            continue;
          PendingEvent producer;
          producer.counter = WaitCounterKind::VaVdst;
          producer.section_name = section_name;
          producer.section_offset = entry.section_offset;
          producer.file_offset = file_offset - section_offset + entry.section_offset;
          producer.instruction = inst.disassemble();
          return CounterParityRequirement{
              std::move(producer), sources[i].reg, WaitcheckAccessKind::Use, 0, &consumer,
              section_offset,      file_offset};
        }
        expired = ++history.age > kMaxPartialForwardingWriterAge ||
                  (history.age > 4 &&
                   std::ranges::all_of(history.writers, [=](uint8_t a) { return a == unseen; }));
      }
      if (end || expired || !at_entry.emplace(block_index, history).second)
        continue;
      for (size_t pred : view.predecessors)
        work.push_back({pred, cfg_views_[pred].instructions.size(), history});
    }
    return std::nullopt;
  }

  // A wait can guard any consumer reachable while its oldest possible writer
  // is still in the eight-VALU window. Reconstruct each candidate on one path
  // through this exact wait, so a producer on another arm cannot justify it.
  std::optional<CounterParityRequirement>
  find_partial_forwarding_after_wait(const PendingWaitGroup &group,
                                     const Instruction &first_consumer, rj_code_arch_t arch) {
    if (!current_cfg_view_index_ || !cfg_has_salu_exec_write_)
      return std::nullopt;
    const auto &start = cfg_views_[*current_cfg_view_index_].instructions;
    const auto it = std::ranges::find(start, &first_consumer, &CfgInstructionView::instruction);
    if (it == start.end())
      return std::nullopt;
    struct Cursor {
      size_t block;
      size_t index;
      unsigned age;
    };
    std::vector<Cursor> work{{*current_cfg_view_index_, size_t(it - start.begin()), 0}};
    std::map<size_t, unsigned> visited;
    while (!work.empty()) {
      auto [block, index, age] = work.back();
      work.pop_back();
      if (index == 0) {
        const auto [pos, inserted] = visited.try_emplace(block, age);
        if (!inserted && pos->second <= age)
          continue;
        pos->second = age;
      }
      const auto &view = cfg_views_[block];
      for (; index < view.instructions.size(); ++index) {
        const auto &entry = view.instructions[index];
        const auto &inst = *entry.instruction;
        if (is_va_vdst_expiring_instruction(inst.mnemonic()) ||
            (inst.is_waitcnt() && explicit_va_vdst_zero(block, inst, arch)) ||
            dsdir_wait_va_vdst(inst) == 0u)
          break;
        if (is_vector_alu(inst)) {
          const auto file_offset = group.file_offset - group.section_offset + entry.section_offset;
          if (auto hazard = partial_forwarding_requirement(inst, entry.section_offset, file_offset,
                                                           group.section_name, arch, block, &group))
            return hazard;
          if (++age > kMaxPartialForwardingWriterAge)
            break;
        }
      }
      if (index == view.instructions.size())
        for (size_t successor : view.successors)
          work.push_back({successor, 0, age});
    }
    return std::nullopt;
  }

  // Follow only the original TRANS destinations while hypothetically removing
  // this wait. A later TRANS ages the old results but cannot justify an
  // earlier wait by introducing a new destination.
  std::optional<CounterParityRequirement>
  find_trans_use_after_wait(const PendingWaitGroup &group, const Instruction &first_consumer) {
    if (!current_cfg_view_index_ || group.state_before.trans_use_hazards.empty())
      return std::nullopt;
    const auto &block = cfg_views_[*current_cfg_view_index_];
    const auto start =
        std::ranges::find(block.instructions, &first_consumer, &CfgInstructionView::instruction);
    if (start == block.instructions.end())
      return std::nullopt;

    using Position = std::pair<size_t, size_t>;
    struct Search {
      Position position;
      TransUseHazards hazards;
    };
    std::vector<Search> work{
        {{*current_cfg_view_index_, size_t(start - block.instructions.begin())},
         group.state_before.trans_use_hazards}};
    // Scalar loops do not advance either issue window. Memoize each original
    // VGPR's least age at a block entry for its exact intervening TRANS count.
    std::map<size_t, TransUseHazards> seen;
    while (!work.empty()) {
      auto [position, hazards] = std::move(work.back());
      work.pop_back();
      const auto &[block_index, index] = position;
      const auto &view = cfg_views_[block_index];
      if (index == 0) {
        auto &prior = seen[block_index];
        std::erase_if(hazards, [&](const auto &item) {
          auto [it, inserted] = prior.try_emplace(item.first, item.second);
          if (inserted)
            return false;
          if (it->second.valu_age <= item.second.valu_age)
            return true;
          it->second = item.second;
          return false;
        });
      }
      for (size_t i = index; i < view.instructions.size() && !hazards.empty(); ++i) {
        const auto &entry = view.instructions[i];
        const auto &inst = *entry.instruction;
        const bool current_group = entry.section_offset >= group.section_offset &&
                                   entry.section_offset <= group.last_wait_section_offset;
        if (!current_group &&
            ((inst.is_waitcnt() &&
              explicit_va_vdst_zero(block_index, inst, ROCJITSU_CODE_ARCH_RDNA3)) ||
             dsdir_wait_va_vdst(inst) == 0u)) {
          hazards.clear();
          break;
        }
        if (is_vector_alu(inst)) {
          const auto offset = relative_cfg_m0_.find({block_index, &inst});
          const auto du = inst_def_use_for_waitcheck(
              inst, entry.vgpr_msb, ROCJITSU_CODE_ARCH_RDNA3, wavefront_size_,
              offset == relative_cfg_m0_.end() ? ScalarOffsetValues{} : offset->second);
          const auto file_offset = group.file_offset - group.section_offset + entry.section_offset;
          if (auto result =
                  trans_use_requirement(hazards, inst, du, entry.section_offset, file_offset))
            return result;
        }
        advance_trans_use_hazards(hazards, inst);
      }
      if (!hazards.empty())
        for (size_t successor : view.successors)
          work.push_back({{successor, 0}, hazards});
    }
    return std::nullopt;
  }

  using DependencyObserver =
      std::function<void(const PendingEvent &, RegisterRef, WaitcheckAccessKind, uint32_t)>;

  enum class SourceUseComparison { Preserved, AddedUse, Incomplete };

  // Keeping an older VM source kind live can prevent a later partial VM_VSRC
  // wait from retiring even sources issued after this field. Check the proposed
  // thresholds through their common suffix, holding the other fields fixed.
  SourceUseComparison compare_vm_vsrc_uses(const PendingWaitGroup &group,
                                           const Instruction &consumer, uint64_t consumer_offset,
                                           WaitCounterKind field, uint32_t reference_count,
                                           uint32_t candidate_count, rj_code_arch_t arch) {
    constexpr auto source = WaitCounterKind::VmVsrc;
    const auto source_idx = counter_index(source);
    if (group.state_before.pending[source_idx].empty() &&
        std::ranges::all_of(group.state_before.pending_event_ages[source_idx].values,
                            [](uint8_t age) { return age == kNoPendingEventAge; }))
      return SourceUseComparison::Preserved;
    PendingState reference = group.state_before;
    PendingState candidate = reference;
    auto fields = group.emitted_counts;
    fields[counter_index(field)] = reference_count;
    (void)apply_wait_fields(reference, fields, arch);
    fields[counter_index(field)] = candidate_count;
    (void)apply_wait_fields(candidate, fields, arch);
    const bool loses_explicit_drain =
        field == source && reference_count == 0 && candidate_count != 0;
    if (reference == candidate)
      return SourceUseComparison::Preserved;

    auto block_index = current_cfg_view_index_;
    const CfgBlockView *block = block_index ? &cfg_views_[*block_index] : nullptr;
    // The ordinary checker merges all incoming source kinds before applying
    // a partial VM_VSRC wait. Both trials need the unaffected arms of a join.
    // A backedge can depend on the edited field and needs a new fixed point.
    const auto enter_block = [&](size_t from, size_t to) {
      if (group.cfg_block == to)
        return false;
      for (size_t pred : cfg_views_[to].predecessors) {
        if (pred == from)
          continue;
        if (cfg_component_[pred] == cfg_component_[to])
          return false;
        merge_into(reference, cfg_outputs_[pred]);
        merge_into(candidate, cfg_outputs_[pred]);
      }
      block = &cfg_views_[to];
      block_index = to;
      return true;
    };
    size_t pos = consumer_offset / sizeof(uint32_t);
    if (block) {
      if (cfg_has_unknown_successor_)
        return SourceUseComparison::Incomplete;
      if (group.cfg_block && group.cfg_block != current_cfg_view_index_ &&
          !enter_block(*group.cfg_block, *current_cfg_view_index_))
        return SourceUseComparison::Incomplete;
      auto start =
          std::ranges::find(block->instructions, &consumer, &CfgInstructionView::instruction);
      if (start == block->instructions.end() || start->section_offset != consumer_offset)
        return SourceUseComparison::Incomplete;
      pos = static_cast<size_t>(start - block->instructions.begin());
    }
    for (size_t steps = 0; steps < 256; ++steps) {
      uint64_t pc = 0;
      std::unique_ptr<Instruction> decoded;
      const Instruction *inst = nullptr;
      if (block) {
        if (pos == block->instructions.size()) {
          if (block->successors.empty())
            return SourceUseComparison::Preserved;
          if (block->successors.size() != 1)
            return SourceUseComparison::Incomplete;
          if (!enter_block(*block_index, block->successors.front()))
            return SourceUseComparison::Incomplete;
          pos = 0;
          continue;
        }
        const auto &view = block->instructions[pos++];
        pc = view.section_offset;
        inst = view.instruction;
      } else {
        if (stream_words_.empty())
          return SourceUseComparison::Incomplete;
        if (pos >= stream_words_.size())
          return SourceUseComparison::Preserved;
        pc = pos * sizeof(uint32_t);
        auto result = decoder_->decode_window(stream_words_.subspan(pos), pc);
        if (result.failed() || !(decoded = std::move(result).value()) || decoded->size() <= 0)
          return SourceUseComparison::Incomplete;
        inst = decoded.get();
        pos += static_cast<size_t>(decoded->size()) / sizeof(uint32_t);
      }
      // Returning earlier requires equality of all live events and ages,
      // including the counter-only credits used to decide partial retirement.
      if (loses_explicit_drain && pc == consumer_offset &&
          inst->mnemonic() == "ds_atomic_async_barrier_arrive_b64")
        return SourceUseComparison::AddedUse;
      auto transfer = [&](PendingState &state) {
        using Use = std::tuple<uint64_t, RegClass, uint16_t, uint16_t, WaitcheckAccessKind>;
        std::map<Use, uint32_t> uses;
        const DependencyObserver observe = [&](const PendingEvent &event, RegisterRef reg,
                                               WaitcheckAccessKind access, uint32_t required) {
          if (event.counter == source) {
            const Use use{event.section_offset, reg.cls, reg.index, reg.width, access};
            auto [it, inserted] = uses.try_emplace(use, required);
            if (!inserted)
              it->second = std::min(it->second, required);
          }
        };
        analyze_instruction(state, *inst, group.section_name, pc,
                            group.file_offset - group.section_offset + pc, arch, false, &observe);
        return uses;
      };
      const auto prior_uses = transfer(reference);
      const auto new_uses = transfer(candidate);
      for (const auto &[use, required] : new_uses) {
        auto prior = prior_uses.find(use);
        if (prior == prior_uses.end() || required < prior->second)
          return SourceUseComparison::AddedUse;
      }
      if (reference == candidate || is_program_end(inst->mnemonic()))
        return SourceUseComparison::Preserved;
      if (!block && inst->is_branch())
        return SourceUseComparison::Incomplete;
    }
    return SourceUseComparison::Incomplete;
  }

  enum class StagedWait { OmitField, CheckQueueAdmission, ApplyField };

  [[nodiscard]] std::optional<CounterParityRequirement> find_staged_counter_parity_requirement(
      const PendingWaitGroup &group, const Instruction &first_consumer,
      uint64_t first_consumer_section_offset, WaitCounterKind counter, uint32_t emitted_count,
      rj_code_arch_t arch, size_t step_limit, bool &complete,
      bool *guarded_flat_dependency = nullptr, std::optional<uint32_t> weakened_flat_count = {},
      StagedWait mode = StagedWait::OmitField) {
    const bool actual = mode == StagedWait::ApplyField;
    // Mark later issues from backedges separately from requests that crossed
    // the wait. Their counter credits and all companion facets remain live.
    const std::string issued_after_wait =
        actual ? group.section_name + " (after wait)" : group.section_name;
    const bool staged_vm_vsrc = counter == WaitCounterKind::VmVsrc && group.vm_vsrc_without_field;
    PendingState initial_state = staged_vm_vsrc ? *group.vm_vsrc_without_field : group.state_before;
    if (!staged_vm_vsrc)
      apply_joint_parity_waits(initial_state, group, counter, actual, arch);
    for (size_t i = 0; i < group.emitted_counts.size(); ++i) {
      const auto companion = static_cast<WaitCounterKind>(i);
      if (companion != counter && group.emitted_counts[i] &&
          !(staged_vm_vsrc && is_vm_vsrc_primary_counter(companion)))
        apply_counter_parity_wait(initial_state, companion, *group.emitted_counts[i], arch);
    }
    if (actual) {
      WaitFields field{};
      field[counter_index(counter)] = group.emitted_counts[counter_index(counter)];
      (void)apply_wait_fields(initial_state, field, arch);
    }
    const auto &translations = initial_state.pending[counter_index(WaitCounterKind::X)];
    const bool prefetch_companion = cfg_has_scalar_prefetch_ && counter != WaitCounterKind::X &&
                                    std::ranges::any_of(translations, [&](const auto &e) {
                                      return counter_may_retire_event(counter, e);
                                    });
    auto eligible = [&](const PendingEvent &event) {
      return event.counter == counter ||
             ((actual || (prefetch_companion && event.counter == WaitCounterKind::X)) &&
              counter_may_retire_event(counter, event));
    };
    auto original_event = [&](const PendingEvent &event) -> const PendingEvent * {
      if (!eligible(event))
        return nullptr;
      const auto &originals = initial_state.pending[counter_index(event.counter)];
      auto it = find_event(originals, event);
      return it == originals.end() ? nullptr : &*it;
    };
    auto has_original = [&](const PendingState &state) {
      if (counter == WaitCounterKind::VaVdst &&
          (!state.va_vdst_hazards.hazards.empty() || !state.trans_use_hazards.empty()))
        return true;
      for (auto kind : {counter, WaitCounterKind::X, WaitCounterKind::VmVsrc})
        if (std::ranges::any_of(state.pending[counter_index(kind)], original_event))
          return true;
      return false;
    };
    if (!has_original(initial_state))
      return std::nullopt;
    using Position = std::pair<size_t, size_t>;
    const bool stream = actual && !current_cfg_view_index_ && !stream_words_.empty();
    Position start_position{0, first_consumer_section_offset / sizeof(uint32_t)};
    if (!stream) {
      if (!current_cfg_view_index_ || *current_cfg_view_index_ >= cfg_views_.size() ||
          (counter == WaitCounterKind::VaVdst && !actual &&
           initial_state.pending[counter_index(counter)].empty())) {
        if (weakened_flat_count || mode != StagedWait::OmitField)
          complete = false;
        return std::nullopt;
      }
      const auto &start = cfg_views_[*current_cfg_view_index_].instructions;
      auto it = std::ranges::find(start, &first_consumer, &CfgInstructionView::instruction);
      if (it == start.end() || it->section_offset != first_consumer_section_offset) {
        if (weakened_flat_count || mode != StagedWait::OmitField)
          complete = false;
        return std::nullopt;
      }
      start_position = {*current_cfg_view_index_, static_cast<size_t>(it - start.begin())};
      if (actual && cfg_has_unknown_successor_)
        complete = false;
      if (actual && counter == WaitCounterKind::VaVdst && tracks_partial_forwarding(arch)) {
        if (auto hazard = find_partial_forwarding_after_wait(group, first_consumer, arch))
          return hazard;
      }
    }
    // Interior instructions have a single predecessor. Carry their state forward
    // and retain it only at CFG entries, where incoming paths can meet again.
    std::map<Position, PendingState> states;
    std::vector<Position> worklist;
    std::optional<std::pair<Position, PendingState>> linear;
    if (stream || start_position.second != 0)
      linear.emplace(start_position, initial_state);
    else {
      states.emplace(start_position, initial_state);
      worklist.push_back(start_position);
    }
    std::optional<CounterParityRequirement> strongest;
    size_t steps = 0;

    auto enqueue = [&](Position position, const PendingState &state) {
      auto [it, inserted] = states.try_emplace(position, state);
      if (inserted) {
        worklist.push_back(position);
        return;
      }
      PendingState merged = it->second;
      merge_into(merged, state);
      if (merged != it->second) {
        it->second = std::move(merged);
        worklist.push_back(position);
      }
    };

    while ((linear || !worklist.empty()) && steps++ < step_limit) {
      Position position;
      PendingState state;
      if (linear) {
        position = linear->first;
        state = std::move(linear->second);
        linear.reset();
      } else {
        position = worklist.back();
        worklist.pop_back();
        state = states.at(position);
      }
      const auto [block_index, instruction_index] = position;
      std::unique_ptr<Instruction> decoded;
      CfgInstructionView view;
      Position next{block_index, instruction_index + 1};
      if (stream) {
        if (instruction_index >= stream_words_.size())
          continue;
        view.section_offset = instruction_index * sizeof(uint32_t);
        auto result =
            decoder_->decode_window(stream_words_.subspan(instruction_index), view.section_offset);
        if (result.failed() || !(decoded = std::move(result).value()) || decoded->size() <= 0) {
          complete = false;
          continue;
        }
        view.instruction = decoded.get();
        next.second = instruction_index + static_cast<size_t>(decoded->size()) / sizeof(uint32_t);
      } else {
        const CfgBlockView &block = cfg_views_[block_index];
        if (instruction_index >= block.instructions.size()) {
          for (size_t successor : block.successors)
            enqueue({successor, 0}, state);
          continue;
        }
        view = block.instructions[instruction_index];
      }
      const Instruction &instruction = *view.instruction;
      if (instruction.is_vskip_affected() && state.vector_skip != false) {
        if (!state.vector_skip.has_value()) {
          complete = false;
          return std::nullopt;
        }
        linear.emplace(next, std::move(state));
        continue;
      }
      if (!actual && view.section_offset == group.section_offset && position != start_position &&
          cfg_counter_wait(block_index, instruction, counter, arch)) {
        // A loop can reach the same physical field again. While testing a
        // weakened field, keep a superset of its original events around
        // the backedge. The path query applies each actual field and its
        // companions on every visit before admitting a consumer.
        if (weakened_flat_count || mode == StagedWait::CheckQueueAdmission)
          linear.emplace(next, std::move(state));
        continue;
      }

      if (actual) {
        // Embedded fields retire their requests before this instruction uses its operands.
        (void)apply_embedded_waitcnt(state, instruction, arch);
      }
      const PendingState *dependency_state = &state;
      const InstDefUse du = inst_def_use_for_waitcheck(instruction, dependency_state->vgpr_msb,
                                                       arch, wavefront_size_, dependency_state->m0);
      auto current_events = effective_events(instruction, state.vgpr_msb, arch);
      if (!expert_waits_enabled(*dependency_state)) {
        std::erase_if(current_events, [](const ClassifiedEvent &event) {
          return event.requires_expert_scheduling();
        });
      }
      if (mode != StagedWait::OmitField)
        apply_admission(state, instruction, current_events, arch, false);
      const bool prefetch =
          arch == ROCJITSU_CODE_ARCH_CDNA5 && is_scalar_prefetch(instruction.mnemonic());
      if (prefetch && state.prefetch_enabled == false) {
        analyze_instruction(state, instruction, issued_after_wait, view.section_offset,
                            group.file_offset - group.section_offset + view.section_offset, arch,
                            /*emit_diagnostics=*/false);
        linear.emplace(next, std::move(state));
        continue;
      }
      std::optional<PendingState> xcnt_adjusted_state;
      if (!prefetch && std::ranges::any_of(current_events, [](const ClassifiedEvent &event) {
            return event.counter == WaitCounterKind::X;
          })) {
        xcnt_adjusted_state.emplace(state);
        apply_implicit_xcnt_ordering(*xcnt_adjusted_state, instruction, du, current_events, arch);
        dependency_state = &*xcnt_adjusted_state;
      }
      // The selected ALU delay takes effect before the receiving instruction
      // reads and writes its operands. Preserve the original state so that
      // analyze_instruction consumes the tag exactly once on the way forward.
      std::optional<PendingState> delay_adjusted_state;
      if (counter == WaitCounterKind::VaVdst && (instruction.flags() & VALU) &&
          dependency_state->expert_delay.pending && expert_waits_enabled(*dependency_state)) {
        delay_adjusted_state.emplace(*dependency_state);
        apply_expert_valu_delay(*delay_adjusted_state, instruction, arch,
                                state.expert_valu_may_skip &&
                                    !expert_valu_issues_without_active_lanes(instruction, du.defs));
        dependency_state = &*delay_adjusted_state;
      }
      if (actual && counter == WaitCounterKind::VaVdst) {
        if (arch == ROCJITSU_CODE_ARCH_RDNA3) {
          auto hazard =
              trans_use_requirement(state.trans_use_hazards, instruction, du, view.section_offset,
                                    group.file_offset - group.section_offset + view.section_offset);
          if (hazard)
            return hazard;
        }
        if (expert_waits_enabled(state) && dsdir_wait_va_vdst(instruction)) {
          du.defs.for_each([&](RegisterRef ref) {
            if (ref.cls != RegClass::VGPR)
              return;
            for (bool trans : {true, false}) {
              const auto &originals = initial_state.va_vdst_hazards.hazards;
              const auto it = originals.find({ref.index, trans});
              if (it == originals.end() ||
                  !state.va_vdst_hazards.contains_producer(ref.index, it->second.producer))
                continue;
              const uint32_t count = trans ? 0u : it->second.age;
              if (strongest && strongest->required_count <= count)
                continue;
              PendingEvent source;
              source.counter = counter;
              source.section_offset = it->second.producer.section_offset;
              strongest = CounterParityRequirement{source,
                                                   ref,
                                                   WaitcheckAccessKind::Def,
                                                   count,
                                                   &instruction,
                                                   view.section_offset,
                                                   group.file_offset - group.section_offset +
                                                       view.section_offset};
            }
          });
        }
      }
      auto record_if_original = [&](const PendingEvent &event, RegisterRef reg,
                                    WaitcheckAccessKind access) {
        const PendingEvent *original = original_event(event);
        if (!original || (actual && reg.cls == RegClass::VGPR &&
                          scratch_access_excludes_flat_result(event, instruction, arch)))
          return;
        std::optional<uint32_t> threshold;
        if (original->counter == counter)
          threshold = dependency_required_count(initial_state, *original, arch).value();
        else
          threshold = implied_counter_requirement(initial_state, *original, counter, arch);
        if (!threshold)
          return;
        const uint32_t required_count = *threshold;
        if (CounterParityRequirement::can_improve(strongest, required_count)) {
          CounterParityRequirement candidate{*original,
                                             reg,
                                             access,
                                             required_count,
                                             &instruction,
                                             view.section_offset,
                                             group.file_offset - group.section_offset +
                                                 view.section_offset};
          if (weakened_flat_count) {
            const auto witness =
                guarded_flat_wait_witness(group, candidate, arch, *weakened_flat_count);
            if (!witness)
              complete = false;
            else if (*witness)
              strongest = std::move(candidate);
            return;
          }
          // A merged zero can belong to an older conditional FLAT. Check this
          // event before stopping, so another operand or later partial wait
          // can still establish a real zero-count dependency.
          const auto witness =
              stream   ? std::optional<bool>{true}
              : actual ? has_cfg_path_with_event_pending(
                             event, view.section_offset, arch,
                             relative_wait_operand(instruction, arch) != nullptr,
                             access_requires_active_lane(instruction, reg), 0, {},
                             guarded_lane_operand(event, instruction, reg, access, arch))
                       : cross_lane_parity_witness(group, candidate, arch);
          if (stream)
            candidate.consumer = nullptr; // The decoded stream instruction is temporary.
          if (actual && !witness)
            complete = false;
          if (witness == false ||
              prolog_flat_lacks_zero_witness(group, counter, emitted_count, arch, candidate)) {
            if (guarded_flat_dependency)
              *guarded_flat_dependency = true;
          } else {
            candidate.witness_unknown = !witness.has_value();
            if (!stream && counter == event.counter &&
                !refine_guarded_load_count(group, arch, candidate))
              return;
            if (actual && candidate.witness_unknown)
              complete = false;
            if (CounterParityRequirement::can_improve(strongest, candidate.required_count))
              strongest = std::move(candidate);
          }
        }
      };
      if (prefetch && (counter == WaitCounterKind::X || prefetch_companion)) {
        for (const auto &event : state.pending[counter_index(WaitCounterKind::X)])
          record_if_original(event, {RegClass::PC, 0, 1}, WaitcheckAccessKind::MemoryOrder);
      }
      visit_dependencies(
          *dependency_state, instruction, du, current_events, arch,
          [&](const PendingEvent &event, RegisterRef reg, WaitcheckAccessKind access, uint32_t) {
            if ((actual || event.counter == counter) &&
                !(event.counter == WaitCounterKind::X && arch == ROCJITSU_CODE_ARCH_CDNA5 &&
                  is_xcnt_drain(instruction, state.vgpr_msb_setreg_hazard)))
              record_if_original(event, reg, access);
          });

      // Compiler barrier/fence policy can explain the strength of an adjacent wait.
      // It does not create a later operand use of each outstanding register result.
      if (!actual && amdgpu::uses_legacy_waitcnt(amdgpu::waitcnt_model(arch).value()) &&
          instruction.mnemonic() == "s_barrier" && counter == WaitCounterKind::Ds) {
        for (const PendingEvent &event :
             dependency_state->pending[counter_index(WaitCounterKind::Ds)]) {
          if (event.kind != WaitEventKind::Smem) {
            record_if_original(event, RegisterRef{RegClass::PC, 0, 1},
                               WaitcheckAccessKind::MemoryOrder);
          }
        }
      }
      if (instruction.mnemonic() == "s_barrier" || starts_with(instruction.mnemonic(), "ds_")) {
        for (const PendingEvent &event : dependency_state->pending[counter_index(counter)]) {
          if (event.check_counter_parity_order) {
            record_if_original(event, RegisterRef{RegClass::PC, 0, 1},
                               WaitcheckAccessKind::MemoryOrder);
          }
        }
      }
      if (strongest && !strongest->witness_unknown &&
          (weakened_flat_count || strongest->required_count <= emitted_count))
        return strongest;

      if (actual) {
        if (auto fields = relative_cfg_waits_.find({block_index, &instruction});
            fields != relative_cfg_waits_.end())
          (void)apply_wait_fields(state, fields->second, arch);
      }
      analyze_instruction(state, instruction, issued_after_wait, view.section_offset,
                          group.file_offset - group.section_offset + view.section_offset, arch,
                          /*emit_diagnostics=*/false);
      if (is_program_end(instruction.mnemonic()))
        continue;

      if (actual && counter == WaitCounterKind::VaVdst) {
        std::erase_if(state.va_vdst_hazards.hazards, [&](const auto &entry) {
          return !initial_state.va_vdst_hazards.contains_producer(entry.first.first,
                                                                  entry.second.producer);
        });
        std::erase_if(state.trans_use_hazards, [&](const auto &entry) {
          return std::ranges::none_of(initial_state.trans_use_hazards, [&](const auto &original) {
            return original.first.first == entry.first.first &&
                   original.second.producer == entry.second.producer;
          });
        });
      }
      if (!has_original(state))
        continue;
      if (stream && instruction.is_branch()) {
        complete = false;
        continue;
      }
      linear.emplace(next, std::move(state));
    }
    if (linear || !worklist.empty())
      complete = false;
    return strongest;
  }

  [[nodiscard]] bool guarded_gfx11_flat_result(WaitCounterKind counter, rj_code_arch_t arch,
                                               const CounterParityRequirement &requirement) const {
    return current_kernel_ && cfg_has_exec_guards_ &&
           (arch == ROCJITSU_CODE_ARCH_RDNA3 || arch == ROCJITSU_CODE_ARCH_RDNA3_5) &&
           (counter == WaitCounterKind::Ds ||
            (counter == WaitCounterKind::Load && wavefront_size_ == 32)) &&
           requirement.event.counter == counter &&
           requirement.event.kind == WaitEventKind::FlatLoad &&
           requirement.reg.cls == RegClass::VGPR;
  }

  std::optional<bool> guarded_flat_wait_witness(const PendingWaitGroup &group,
                                                const CounterParityRequirement &candidate,
                                                rj_code_arch_t arch, uint32_t count) {
    return has_cfg_path_with_event_pending(
        candidate.event, candidate.consumer_section_offset, arch, false,
        access_requires_active_lane(*candidate.consumer, candidate.reg), 0,
        WaitRange{group.section_offset, group.last_wait_section_offset, count},
        guarded_lane_operand(candidate.event, *candidate.consumer, candidate.reg, candidate.access,
                             arch));
  }

  // A VCCZ fallthrough cannot consume a producer whose every route to the
  // branch sets the captured wave32 mask to zero. Qualify the field at its
  // incoming edge, before any counter threshold or consumer operand applies.
  bool producer_skips_masked_parity_wait(const PendingWaitGroup &group, const PendingEvent &event,
                                         rj_code_arch_t arch) {
    if (!current_kernel_ || !amdgpu::supports_expert_scheduling(arch) || wavefront_size_ != 32 ||
        event.counter != WaitCounterKind::VaVdst || cfg_has_unknown_successor_ ||
        path_mask_inputs_.empty() || !group.cfg_block || *group.cfg_block == 0 ||
        *group.cfg_block >= cfg_views_.size())
      return false;
    const auto &wait = cfg_views_[*group.cfg_block];
    if (wait.predecessors.size() != 1)
      return false;
    const auto guard_index = wait.predecessors.front();
    const auto &guard = cfg_views_[guard_index];
    if (guard_index == 0 || guard.successors.size() != 2 || guard.instructions.empty() ||
        guard.instructions.size() > 64 || guard.block->end_offset() != wait.block->start_offset() ||
        guard.instructions.back().instruction->mnemonic() != "s_cbranch_vccz" ||
        guard.predecessors.empty() || guard.predecessors.size() > 8)
      return false;
    // Only the last VCC write matters. The scalar input may be overwritten
    // after the AND without changing the branch.
    const auto writer = std::ranges::find_if(guard.instructions | std::views::reverse,
                                             &CfgInstructionView::invalidates_vcc_condition);
    if (writer == guard.instructions.rend())
      return false;
    const auto &mask = *writer->instruction;
    if (mask.mnemonic() != "s_and_b32" ||
        !low_selector(mask.dst_operand(0), amdgpu::kVccSelectorFirst))
      return false;
    const bool exec_first = low_selector(mask.src_operand(0), amdgpu::kExecSelectorFirst);
    if (!exec_first && !low_selector(mask.src_operand(1), amdgpu::kExecSelectorFirst))
      return false;
    const auto input = mask.src_operand(exec_first ? 1 : 0)->to_register_ref();
    if (!input || input->cls != RegClass::SGPR || input->width != 1 ||
        input->index >= valu_source_mask_writes_.size() ||
        valu_source_mask_writes_.test(input->index))
      return false;
    const auto clobbers = [](const CfgInstructionView &view) {
      const auto &inst = *view.instruction;
      return (inst.flags() & INDIRECT_CALL) || inst.mnemonic().starts_with("s_call") ||
             inst.mnemonic().starts_with("s_movrel");
    };
    const auto writes_input = [&](const CfgInstructionView &view) {
      const auto &defs = path_scalar_defs_.at(view.instruction);
      return std::ranges::find(defs, input->index) != defs.end();
    };
    if (std::ranges::any_of(guard.instructions, clobbers) ||
        std::any_of(guard.instructions.begin(), writer.base() - 1, writes_input))
      return false;
    const auto blocks = cfg_block_indices_containing(event.section_offset);
    if (blocks.empty() || std::ranges::find(blocks, guard_index) != blocks.end() ||
        std::ranges::find(blocks, *group.cfg_block) != blocks.end())
      return false;
    const auto reachable = reachable_blocks(blocks, true);
    for (size_t pred : guard.predecessors) {
      if (!(*reachable)[pred])
        continue;
      const auto &arm = cfg_views_[pred];
      if (arm.instructions.size() > 64)
        return false;
      bool zero = false;
      for (const auto &view : arm.instructions | std::views::reverse) {
        if (clobbers(view))
          return false;
        if (writes_input(view)) {
          const auto &inst = *view.instruction;
          const auto *src = inst.src_operand(0);
          zero = !view.scalar_destination_pending && inst.mnemonic() == "s_mov_b32" && src &&
                 src->const_value() == 0;
          break;
        }
      }
      if (!zero)
        return false;
    }
    return true;
  }

  bool wait_in_repeated_exec_region(const PendingWaitGroup &group, rj_code_arch_t arch) const {
    if (!cfg_has_repeated_exec_filter_ || !group.cfg_block)
      return false;
    const auto first_arm_has_wait = [&](size_t arm) {
      return std::ranges::any_of(cfg_views_[arm].instructions, [&](const auto &view) {
        return view.instruction->mnemonic() == "s_wait_alu" &&
               cfg_counter_wait(arm, *view.instruction, WaitCounterKind::VaVdst, arch).has_value();
      });
    };
    size_t block = *group.cfg_block;
    const auto &measured = cfg_views_[block];
    if (measured.successors.size() == 1 &&
        cfg_views_[measured.successors.front()].repeated_exec_filter_arm == block)
      return first_arm_has_wait(block);

    // A branch-only block with one incoming and one outgoing edge forwards
    // the guarded fallthrough without changing its state. Bound cyclic walks.
    for (size_t hops = 0; hops < cfg_views_.size(); ++hops) {
      const auto &current = cfg_views_[block];
      if (current.predecessors.size() != 1)
        return false;
      const size_t predecessor = current.predecessors.front();
      const auto &previous = cfg_views_[predecessor];
      if (previous.repeated_exec_filter_arm &&
          previous.block->end_offset() == current.block->start_offset())
        return first_arm_has_wait(*previous.repeated_exec_filter_arm);
      if (previous.instructions.size() != 1 || previous.successors.size() != 1 ||
          previous.instructions.front().instruction->mnemonic() != "s_branch")
        return false;
      block = predecessor;
    }
    return false;
  }

  std::optional<bool> cross_lane_parity_witness(const PendingWaitGroup &group,
                                                const CounterParityRequirement &requirement,
                                                rj_code_arch_t arch) {
    const auto operand = guarded_lane_operand(requirement.event, *requirement.consumer,
                                              requirement.reg, requirement.access, arch);
    if (!group.emitted_counts[counter_index(requirement.event.counter)])
      return true;
    if (producer_skips_masked_parity_wait(group, requirement.event, arch))
      return false;
    const bool scratch =
        requirement.reg.cls == RegClass::VGPR && has_guarded_scratch_result(requirement.event);
    const bool active =
        operand.has_value() || access_requires_active_lane(*requirement.consumer, requirement.reg);
    // Reapplying the same EXEC filter can correlate the two guarded arms.
    // Replay paths that execute the measured VA field with that field disabled.
    const bool repeated_exec_wait = active &&
                                    requirement.event.counter == WaitCounterKind::VaVdst &&
                                    wait_in_repeated_exec_region(group, arch);
    if (!operand && !scratch && !repeated_exec_wait)
      return true;
    // The emitted wait can itself retire the producer. Remove just this
    // field when testing its candidate consumer; all later waits still apply.
    const auto pending = has_cfg_path_with_event_pending(
        requirement.event, requirement.consumer_section_offset, arch, false, active, 0,
        WaitRange{group.section_offset, group.last_wait_section_offset,
                  *counter_no_wait_value(arch, requirement.event.counter).value()},
        operand);
    // Keep the original witness if the extra repeated-EXEC proof was bounded.
    if (!operand && !scratch)
      return pending.value_or(true);
    return pending;
  }

  [[nodiscard]] bool has_guarded_scratch_result(const PendingEvent &event) const {
    return cfg_has_scratch_zero_exit_ && event.counter == WaitCounterKind::Load &&
           event.instruction.starts_with("scratch_load");
  }

  [[nodiscard]] bool prolog_flat_lacks_zero_witness(const PendingWaitGroup &group,
                                                    WaitCounterKind counter, uint32_t emitted_count,
                                                    rj_code_arch_t arch,
                                                    const CounterParityRequirement &requirement) {
    return group.non_entry_function_prolog && emitted_count == 0 &&
           requirement.required_count == 0 &&
           guarded_gfx11_flat_result(counter, arch, requirement) &&
           !has_cfg_path_with_event_pending(
                requirement.event, requirement.consumer_section_offset, arch,
                /*before_target=*/false,
                access_requires_active_lane(*requirement.consumer, requirement.reg),
                /*minimum_younger=*/0,
                WaitRange{group.section_offset, group.last_wait_section_offset, 1},
                guarded_lane_operand(requirement.event, *requirement.consumer, requirement.reg,
                                     requirement.access, arch))
                .value_or(false);
  }

  bool refine_guarded_flat_requirement(const PendingWaitGroup &group,
                                       const Instruction &first_consumer,
                                       uint64_t first_consumer_offset, WaitCounterKind counter,
                                       uint32_t emitted_count, rj_code_arch_t arch,
                                       std::optional<CounterParityRequirement> &requirement) {
    if (!guarded_gfx11_flat_result(counter, arch, *requirement) ||
        group.state_before.pending[counter_index(counter)].size() != 1 ||
        group.last_wait_section_offset != group.section_offset ||
        requirement->required_count <= emitted_count)
      return true;
    if (wavefront_size_ != 32 && (requirement->consumer_section_offset != first_consumer_offset ||
                                  requirement->access != WaitcheckAccessKind::Use))
      return true;

    const uint32_t no_wait = *counter_no_wait_value(arch, counter).value();
    uint32_t safe = requirement->required_count;
    // With one original result, every consumer is safe if the weakened
    // field still retires this result on every path reaching the wait.
    auto retires = [&](uint32_t count) {
      return has_cfg_path_with_event_pending(requirement->event, group.section_offset, arch,
                                             /*before_target=*/true, /*requires_active_lane=*/true,
                                             count) == false;
    };
    if (safe + 1 < no_wait && retires(safe + 1)) {
      ++safe;
      uint32_t upper = no_wait;
      while (safe + 1 < upper) {
        const uint32_t middle = safe + (upper - safe) / 2;
        if (retires(middle))
          safe = middle;
        else
          upper = middle;
      }
      requirement->required_count = safe;
    }
    if (wavefront_size_ != 32 ||
        guarded_flat_wait_witness(group, *requirement, arch, safe + 1) != false)
      return true;

    // This particular access cannot see the result at safe+1. Another
    // register use, overwrite or ordering constraint may still require it.
    // Follow the original issue through later waits and every successor.
    auto witness = [&](uint32_t count, bool &complete) {
      return find_staged_counter_parity_requirement(
          group, first_consumer, first_consumer_offset, counter, emitted_count, arch,
          kParityLookaheadLimit, complete, nullptr, count);
    };
    uint32_t lower = safe;
    uint32_t upper = no_wait;
    bool complete = true;
    auto surviving = witness(upper, complete);
    if (!surviving) {
      if (!complete)
        return false;
      requirement.reset();
      return true;
    }
    while (lower + 1 < upper) {
      const uint32_t middle = lower + (upper - lower) / 2;
      complete = true;
      if (auto found = witness(middle, complete)) {
        upper = middle;
        surviving = std::move(found);
      } else if (complete) {
        lower = middle;
      } else {
        break;
      }
    }
    surviving->required_count = lower;
    requirement = std::move(surviving);
    return true;
  }

  void audit_counter_parity_group(PendingWaitGroup &group, const Instruction &consumer,
                                  uint64_t consumer_section_offset, uint64_t consumer_file_offset,
                                  const PendingState &consumer_input, rj_code_arch_t arch) {
    ++report_.counter_parity_wait_groups;
    if (consumer.is_vskip_affected() && group.state_before.vector_skip != false) {
      ++report_.counter_parity_indeterminate_groups;
      return;
    }
    // Keep the producer ages from before the candidate field. The delay
    // selector follows actual instruction positions across all encoded waits,
    // even when one of their counter fields is omitted in a replay.
    group.state_before.expert_delay = consumer_input.expert_delay;

    // Only direct VM_VSRC attribution uses the state with its field omitted.
    // Apply the primary waits in encoded order to preserve FLAT route checks.
    if (group.emitted_counts[counter_index(WaitCounterKind::VmVsrc)] &&
        !group.vm_vsrc_companions.empty()) {
      group.vm_vsrc_without_field = std::make_unique<PendingState>(group.state_before);
      for (const auto &fields : group.vm_vsrc_companions)
        (void)apply_wait_fields(*group.vm_vsrc_without_field, fields, arch);
    }

    std::array<std::optional<CounterParityRequirement>, kCounterCount> strongest_requirements;
    CounterParityAlternatives alternatives;
    const InstDefUse du = inst_def_use_for_waitcheck(consumer, group.state_before.vgpr_msb, arch,
                                                     wavefront_size_, group.state_before.m0);
    collect_counter_parity_requirements(group, consumer, du, consumer_section_offset,
                                        consumer_file_offset, arch, strongest_requirements,
                                        false, false, &alternatives);

    bool group_indeterminate = false;
    for (size_t counter_idx = 0; counter_idx < group.emitted_counts.size(); ++counter_idx) {
      if (!group.emitted_counts[counter_idx])
        continue;
      const auto counter = static_cast<WaitCounterKind>(counter_idx);
      const auto no_wait = counter_no_wait_value(arch, counter).value();
      const bool ordered_result_counter =
          (arch == ROCJITSU_CODE_ARCH_CDNA4 && counter == WaitCounterKind::Ds) ||
          (arch == ROCJITSU_CODE_ARCH_CDNA5 && counter == WaitCounterKind::Load);
      if (!no_wait)
        continue;
      const uint32_t emitted_count = *group.emitted_counts[counter_idx];

      auto &requirement = strongest_requirements[counter_idx];
      // LLVM inserts zero waits on legacy result counters at non-entry
      // boundaries. Its split-counter ABI also excludes STORE, X, ASYNC and
      // TENSOR. Only LLVM's actual entry and return fields get the short
      // boundary search and policy attribution.
      const bool function_boundary_policy =
          emitted_count == 0 && counter != WaitCounterKind::Store &&
          counter != WaitCounterKind::X && counter != WaitCounterKind::Async &&
          counter != WaitCounterKind::Tensor &&
          (group.non_entry_function_prolog || is_non_entry_function_return(consumer));
      // The caller's continuation can reach the same native instruction both
      // after a return and from an ordinary branch. One qualifying path can
      // require the encoded field even if another path arrived with no work.
      const auto start = path_query_starts_.find(group.section_offset);
      const bool shared = function_boundary_policy ||
                          (start != path_query_starts_.end() && start->second.contexts > 1);
      const auto key =
          std::tuple{group.section_offset, group.last_wait_section_offset, counter, emitted_count};
      if (!shared || shared_parity_.try_emplace(key).second)
        ++report_.counter_parity_fields_checked;
      auto mark_exact = [&] {
        if (shared)
          shared_parity_.at(key).exact = true;
        else
          ++report_.counter_parity_exact;
      };
      // A previous global write/atomic may need to reach L2 before WB, and
      // a following global read may need to wait for INV itself to complete.
      // Legacy BUFFER_WBL2 also needs a zero wait for its own ACK.
      if (emitted_count == 0 && (!requirement || requirement->required_count != 0) &&
          (cdna5_cache_wait_matches(group, consumer, counter, arch) ||
           legacy_buffer_wbl2_wait_matches(group, counter, arch))) {
        mark_exact();
        continue;
      }
      if (!requirement && counter == WaitCounterKind::VaVdst && tracks_partial_forwarding(arch))
        requirement = find_partial_forwarding_after_wait(group, consumer, arch);
      if (!requirement && arch == ROCJITSU_CODE_ARCH_RDNA3 && counter == WaitCounterKind::VaVdst)
        requirement = find_trans_use_after_wait(group, consumer);
      bool boundary_policy_fallback = false;
      bool guarded_flat_dependency = false;
      bool unexamined_partial = false;
      bool reached_expert_completion = false;
      if (has_counter_parity_candidate(group.state_before, counter) &&
          (!requirement || requirement->witness_unknown ||
           requirement->required_count > emitted_count)) {
        bool lookahead_complete = true;
        bool reached_partial_wait = false;
        // GFX1250 implicitly drains the preceding translation class at a
        // VMEM/SMEM switch. Completion waits can also retire a prefix of X.
        // Carry the original events through those instructions before naming
        // a later register as the reason for the explicit X field.
        const bool track_translations =
            arch == ROCJITSU_CODE_ARCH_CDNA5 && counter == WaitCounterKind::X;
        const auto forward_requirement =
            track_translations
                ? find_staged_counter_parity_requirement(group, consumer, consumer_section_offset,
                                                         counter, emitted_count, arch,
                                                         kParityLookaheadLimit, lookahead_complete)
                : find_forward_counter_parity_requirement(
                      group, consumer, consumer_section_offset, counter, emitted_count, arch,
                      function_boundary_policy, lookahead_complete, reached_partial_wait,
                      reached_expert_completion);
        if (!lookahead_complete) {
          if (function_boundary_policy) {
            boundary_policy_fallback = true;
          } else {
            if (shared)
              shared_parity_[key].cannot_relax = true;
            group_indeterminate = true;
            continue;
          }
        }
        if (forward_requirement && CounterParityRequirement::can_improve(
                                       requirement, forward_requirement->required_count)) {
          requirement = forward_requirement;
        }
        // For an emitted zero, only a reachable partial wait can hide an
        // additional dependency beyond the forward search's stopping point.
        // A bounded ABI search without a prefix dependency retains its policy
        // diagnostic. A different caller may still propose a weaker count.
        const bool guarded_prolog_partial =
            !requirement && reached_partial_wait && group.non_entry_function_prolog &&
            (counter == WaitCounterKind::Ds ||
             (counter == WaitCounterKind::Load && wavefront_size_ == 32)) &&
            current_kernel_ && cfg_has_exec_guards_ &&
            (arch == ROCJITSU_CODE_ARCH_RDNA3 || arch == ROCJITSU_CODE_ARCH_RDNA3_5) &&
            std::ranges::any_of(group.state_before.pending[counter_idx], [](const PendingEvent &e) {
              return e.kind == WaitEventKind::FlatLoad && !e.present_on_all_paths;
            });
        unexamined_partial = function_boundary_policy && reached_partial_wait && !requirement &&
                             !boundary_policy_fallback;
        const bool prefetch_companion =
            cfg_has_scalar_prefetch_ && !track_translations &&
            std::ranges::any_of(
                group.state_before.pending[counter_index(WaitCounterKind::X)],
                [&](const PendingEvent &e) { return counter_may_retire_event(counter, e); });
        if (!track_translations &&
            (!function_boundary_policy ||
             ((requirement || guarded_prolog_partial) && !boundary_policy_fallback)) &&
            (emitted_count != 0 || reached_partial_wait || prefetch_companion) &&
            (!requirement || requirement->witness_unknown ||
             requirement->required_count > emitted_count)) {
          const auto staged_requirement = find_staged_counter_parity_requirement(
              group, consumer, consumer_section_offset, counter, emitted_count, arch,
              function_boundary_policy ? kBoundaryLookaheadLimit : kParityLookaheadLimit,
              lookahead_complete, &guarded_flat_dependency, {},
              ordered_result_counter ? StagedWait::CheckQueueAdmission : StagedWait::OmitField);
          if (!lookahead_complete) {
            if (function_boundary_policy) {
              boundary_policy_fallback = true;
            } else {
              if (shared)
                shared_parity_[key].cannot_relax = true;
              group_indeterminate = true;
              continue;
            }
          }
          if (function_boundary_policy && staged_requirement &&
              !staged_requirement->witness_unknown &&
              staged_requirement->required_count == emitted_count) {
            shared_parity_.at(key).exact = true;
            continue;
          }
          if (guarded_flat_dependency)
            boundary_policy_fallback = true;
          // A prefix with no known dependency retains the ABI policy unless
          // the staged search establishes that one count weaker is unsafe.
          if (!guarded_prolog_partial && staged_requirement &&
              CounterParityRequirement::can_improve(requirement,
                                                    staged_requirement->required_count)) {
            requirement = staged_requirement;
          }
        }
      }
      if (reached_expert_completion && requirement) {
        // The initial lookahead identifies potential uses of the original
        // events. Revisit them with the field omitted and replay later
        // instructions that independently retire those operands.
        bool complete = true;
        requirement = find_staged_counter_parity_requirement(
            group, consumer, consumer_section_offset, counter, emitted_count, arch,
            function_boundary_policy ? kBoundaryLookaheadLimit : kParityLookaheadLimit, complete);
        if (!complete) {
          if (shared)
            shared_parity_.at(key).cannot_relax = true;
          group_indeterminate = true;
          continue;
        }
      }
      // A bounded ABI search can find a local dependency without proving
      // that this is the strongest requirement on every path.
      const bool cyclic_cfg_fixed_point_policy =
          has_loop_carried_events(counter, group.state_before, group.section_offset, arch);
      if (requirement &&
          prolog_flat_lacks_zero_witness(group, counter, emitted_count, arch, *requirement)) {
        // The first merged zero was disproved on its producer paths. Follow
        // the original events through other operands, instructions and partial
        // waits before falling back to the unknown-input ABI policy.
        bool complete = true;
        const auto other = find_staged_counter_parity_requirement(
            group, consumer, consumer_section_offset, counter, emitted_count, arch,
            kBoundaryLookaheadLimit, complete);
        if (other && !other->witness_unknown && other->required_count == emitted_count) {
          shared_parity_.at(key).exact = true;
          continue;
        }
        boundary_policy_fallback = true;
      } else if (requirement && !function_boundary_policy &&
                 !refine_guarded_flat_requirement(group, consumer, consumer_section_offset, counter,
                                                  emitted_count, arch, requirement)) {
        group_indeterminate = true;
        continue;
      }
      if (requirement && requirement->witness_unknown) {
        if (!function_boundary_policy) {
          if (shared)
            shared_parity_[key].cannot_relax = true;
          group_indeterminate = true;
          continue;
        }
        boundary_policy_fallback = true;
      }
      // Queue admission can retire an operand at its consumer. A fence orders
      // every applicable request, even when no later register use observes it.
      if (ordered_result_counter && requirement &&
          requirement->access != WaitcheckAccessKind::MemoryOrder &&
          requirement->event.counter == counter &&
          (requirement->event.kind == WaitEventKind::Ds ||
           requirement->event.kind == WaitEventKind::VmemNoSamplerLoad)) {
        // A non-memory consumer accesses the very same pending state that an
        // adjacent wait tested. An issuing consumer may retire an older
        // result at admission, before reading even its own address operand.
        const bool immediate = requirement->consumer_section_offset == consumer_section_offset;
        bool verify = !immediate;
        if (immediate) {
          const auto events = effective_events(consumer, group.state_before.vgpr_msb, arch);
          const auto capacity = *no_wait;
          verify = std::ranges::any_of(events, [&](const ClassifiedEvent &event) {
            if (event.counter != counter)
              return false;
            const uint32_t incoming = issue_units(consumer, event, arch);
            return std::ranges::any_of(group.state_before.pending[counter_idx],
                                       [&](const PendingEvent &pending) {
                                         return pending.min_younger + incoming >= capacity;
                                       }) ||
                   std::ranges::any_of(
                       group.state_before.pending_event_ages[counter_idx].values, [&](uint8_t age) {
                         return age != kNoPendingEventAge && age + incoming >= capacity;
                       });
          });
        }
        if (verify) {
          // A prefix that matches the emitted count can also point to a
          // retired result. Replay all pending events and competing uses.
          bool complete = true;
          requirement = find_staged_counter_parity_requirement(
              group, consumer, consumer_section_offset, counter, emitted_count, arch,
              function_boundary_policy ? kBoundaryLookaheadLimit : kParityLookaheadLimit, complete,
              nullptr, {}, StagedWait::CheckQueueAdmission);
          if (!complete) {
            if (function_boundary_policy) {
              boundary_policy_fallback = true;
            } else {
              if (shared)
                shared_parity_.at(key).cannot_relax = true;
              group_indeterminate = true;
              continue;
            }
          }
        }
      }
      const bool known_boundary_dependency =
          function_boundary_policy && (requirement.has_value() || guarded_flat_dependency);
      const bool needs_boundary_policy =
          function_boundary_policy && (boundary_policy_fallback || !requirement);
      const bool affects_source_wait =
          arch == ROCJITSU_CODE_ARCH_CDNA5 &&
          (counter == WaitCounterKind::VmVsrc || counter == WaitCounterKind::Load ||
           counter == WaitCounterKind::Store || counter == WaitCounterKind::Ds ||
           counter == WaitCounterKind::Sample || counter == WaitCounterKind::Bvh);
      if (affects_source_wait && requirement && requirement->required_count > emitted_count &&
          !needs_boundary_policy) {
        // Positive counts can leave the same later source locked across several
        // thresholds. Keep all companions and use the zero field as the reference.
        const auto comparison = compare_vm_vsrc_uses(group, consumer, consumer_section_offset,
                                                     counter, 0, requirement->required_count, arch);
        if (comparison != SourceUseComparison::Preserved) {
          if (shared)
            shared_parity_.at(key).cannot_relax = true;
          group_indeterminate = true;
          continue;
        }
      }
      if (needs_boundary_policy)
        requirement.reset();
      // The fast parity path can name a conditional vector producer because
      // it reaches the wait on one CFG route, even when no active lane can
      // carry that producer through the measured field to its consumer.
      // Check only an otherwise exact attribution; the full path replay is
      // considerably more expensive than the counter dataflow pass.
      const auto needs_path = [&](const CounterParityRequirement &candidate) {
        return needs_constant_overwrite_parity_path(candidate, consumer, counter, emitted_count,
                                                    arch);
      };
      if (requirement && needs_path(*requirement)) {
        const auto has_witness = [&](CounterParityRequirement &candidate) {
          const auto lane = guarded_lane_operand(candidate.event, *candidate.consumer,
                                                 candidate.reg, candidate.access, arch);
          const auto witness = has_cfg_path_with_event_pending(
              candidate.event, candidate.consumer_section_offset, arch, false,
              lane.has_value() || access_requires_active_lane(*candidate.consumer, candidate.reg),
              0, WaitRange{group.section_offset, group.last_wait_section_offset, *no_wait}, lane);
          return witness == true && refine_guarded_load_count(group, arch, candidate) &&
                 candidate.required_count == emitted_count;
        };
        bool validated = has_witness(*requirement);
        if (!validated) {
          for (auto candidate : alternatives.values[counter_idx]) {
            if (has_witness(candidate)) {
              requirement = std::move(candidate);
              validated = true;
              break;
            }
          }
        }
        // The candidate pool retains only equal-count producers. A different
        // producer or weaker requirement may still justify this field, so a
        // failed or bounded proof cannot establish an excessive wait.
        if (!validated) {
          if (shared)
            shared_parity_.at(key).cannot_relax = true;
          group_indeterminate = true;
          continue;
        }
      }
      const uint32_t required_count = requirement ? requirement->required_count : *no_wait;
      if (requirement && required_count == emitted_count) {
        if (emitted_count != 0) {
          if (affects_source_wait) {
            // A tighter field can discharge a younger source that is still live
            // with the emitted count, even when the first use agrees with it.
            const auto comparison = compare_vm_vsrc_uses(group, consumer, consumer_section_offset,
                                                         counter, 0, emitted_count, arch);
            if (comparison != SourceUseComparison::Preserved) {
              if (shared) {
                if (comparison == SourceUseComparison::AddedUse)
                  shared_parity_.at(key).too_weak = true;
                else
                  shared_parity_.at(key).cannot_relax = true;
              }
              group_indeterminate |= comparison == SourceUseComparison::Incomplete;
              continue;
            }
          }
          // The first matching use rules out relaxation. Check whether the
          // encoded field leaves another original request live at a later use.
          bool complete = true;
          const auto stricter = find_staged_counter_parity_requirement(
              group, consumer, consumer_section_offset, counter, emitted_count - 1, arch,
              kParityLookaheadLimit, complete, nullptr, {}, StagedWait::ApplyField);
          if (stricter && !stricter->witness_unknown && stricter->required_count < emitted_count) {
            if (shared)
              shared_parity_.at(key).too_weak = true;
            continue;
          }
          if (!complete || (stricter && stricter->witness_unknown)) {
            if (shared)
              shared_parity_.at(key).cannot_relax = true;
            group_indeterminate = true;
            continue;
          }
        }
        mark_exact();
      }
      if (required_count <= emitted_count) {
        if (shared && required_count < emitted_count)
          shared_parity_.at(key).too_weak = true;
        continue;
      }

      WaitcheckCounterUnderaccountingDiagnostic diag;
      diag.counter = counter;
      diag.emitted_count = emitted_count;
      diag.required_count = required_count;
      diag.has_required_dependency = requirement.has_value();
      diag.section_name = group.section_name;
      diag.wait_section_offset = group.section_offset;
      diag.wait_file_offset = group.file_offset;
      diag.wait_instruction = group.instructions;
      diag.consumer_section_offset =
          requirement ? requirement->consumer_section_offset : consumer_section_offset;
      diag.consumer_file_offset =
          requirement ? requirement->consumer_file_offset : consumer_file_offset;
      diag.consumer_instruction =
          requirement ? requirement->consumer->disassemble() : consumer.disassemble();
      if (requirement) {
        diag.access = requirement->access;
        diag.reg = requirement->reg;
        diag.producer_section_offset = requirement->event.section_offset;
        diag.producer_file_offset = requirement->event.file_offset;
        diag.producer_instruction = requirement->event.instruction;
      } else {
        diag.reg = RegisterRef{RegClass::PC, 0, 1};
      }
      std::ostringstream message;
      message << "emitted " << wait_expression(counter, emitted_count, arch).value();
      if (requirement)
        message << " is stronger than the modeled dependency requires ("
                << wait_expression(counter, required_count, arch).value() << ")";
      else if (needs_boundary_policy) {
        message << " follows function-boundary policy";
        if (boundary_policy_fallback)
          message << (known_boundary_dependency ? "; a dependency was found, but its strongest "
                                                  "required count is unresolved"
                                                : "; the bounded dependency search was incomplete");
      } else
        message << " has no attributable modeled dependency";
      message << " before " << diag.consumer_instruction;
      if (cyclic_cfg_fixed_point_policy && requirement)
        message << " (loop-carried requests may also affect the emitted count)";
      if (function_boundary_policy && requirement)
        message << " (the function ABI may require the stronger emitted count)";
      diag.message = message.str();
      if (shared) {
        auto &result = shared_parity_[key];
        if (unexamined_partial && !result.partial_policy)
          result.partial_policy = diag;
        // An incomplete context prevents advertising a weaker threshold
        // across all callers. Otherwise keep the tightest concrete witness.
        if (boundary_policy_fallback || !result.diagnostic ||
            (!result.incomplete && requirement &&
             (!result.diagnostic->has_required_dependency ||
              required_count < result.diagnostic->required_count)))
          result.diagnostic = std::move(diag);
        result.incomplete |= boundary_policy_fallback;
      } else {
        record_counter_underaccounting(std::move(diag));
      }
    }
    if (group_indeterminate)
      ++report_.counter_parity_indeterminate_groups;
  }

  void update_counter_parity_group(std::optional<PendingWaitGroup> &pending_group,
                                   const PendingState &state, const Instruction &inst,
                                   const std::string &section_name, uint64_t section_offset,
                                   uint64_t file_offset, rj_code_arch_t arch) {
    if (!options_.check_counter_parity)
      return;
    if (inst.is_vskip_affected() && state.vector_skip == true)
      return;

    if (pending_group && relative_wait_operand(inst, arch)) {
      // The selector observes the state after earlier waits in this group
      // and before this instruction applies its own counter field.
      audit_counter_parity_group(*pending_group, inst, section_offset, file_offset, state, arch);
      pending_group.reset();
    }
    const auto emitted_fields = resolved_wait_fields(state, inst, arch).value();
    if (emitted_fields) {
      const bool has_wait =
          std::ranges::any_of(*emitted_fields, [](const auto &value) { return value.has_value(); });
      if (!pending_group && has_wait) {
        pending_group = PendingWaitGroup{state,
                                         {},
                                         section_name,
                                         section_offset,
                                         file_offset,
                                         {},
                                         is_non_entry_function_prolog(section_offset),
                                         0,
                                         current_cfg_view_index_};
      }
      if (!pending_group)
        return;
      pending_group->last_wait_section_offset = section_offset;
      if (arch == ROCJITSU_CODE_ARCH_CDNA5 &&
          (*emitted_fields)[counter_index(WaitCounterKind::Ds)] &&
          ((*emitted_fields)[counter_index(WaitCounterKind::Load)] ||
           (*emitted_fields)[counter_index(WaitCounterKind::Store)]))
        pending_group->joint_flat_waits.push_back(*emitted_fields);
      record_vm_vsrc_parity_companions(*pending_group, *emitted_fields, arch);
      for (size_t i = 0; i < emitted_fields->size(); ++i) {
        if (!(*emitted_fields)[i])
          continue;
        auto &combined = pending_group->emitted_counts[i];
        combined = combined ? std::min(*combined, *(*emitted_fields)[i]) : (*emitted_fields)[i];
      }
      if (!pending_group->instructions.empty())
        pending_group->instructions += "; ";
      pending_group->instructions += inst.disassemble();
      return;
    }

    if (const auto embedded_fields = embedded_wait_fields(inst, arch).value()) {
      if (!pending_group) {
        pending_group = PendingWaitGroup{state,
                                         {},
                                         section_name,
                                         section_offset,
                                         file_offset,
                                         {},
                                         is_non_entry_function_prolog(section_offset),
                                         0,
                                         current_cfg_view_index_};
      }
      pending_group->last_wait_section_offset = section_offset;
      record_vm_vsrc_parity_companions(*pending_group, *embedded_fields, arch);
      for (size_t i = 0; i < embedded_fields->size(); ++i) {
        if (!(*embedded_fields)[i])
          continue;
        auto &combined = pending_group->emitted_counts[i];
        combined = combined ? std::min(*combined, *(*embedded_fields)[i]) : (*embedded_fields)[i];
      }
      if (!pending_group->instructions.empty())
        pending_group->instructions += "; ";
      pending_group->instructions += inst.disassemble();
      audit_counter_parity_group(*pending_group, inst, section_offset, file_offset, state, arch);
      pending_group.reset();
      return;
    }

    if (!pending_group)
      return;
    const std::string_view mnemonic = inst.mnemonic();
    const bool known_non_counter_wait =
        mnemonic == "s_waitcnt_depctr" || mnemonic == "s_wait_event";
    if (inst.is_waitcnt() && !known_non_counter_wait) {
      ++report_.counter_parity_indeterminate_groups;
    } else {
      audit_counter_parity_group(*pending_group, inst, section_offset, file_offset, state, arch);
    }
    pending_group.reset();
  }

  void finish_counter_parity_block(std::optional<PendingWaitGroup> &pending_group,
                                   const PendingState &state, rj_code_arch_t arch) {
    if (!pending_group)
      return;

    // A basic block can end at a label even when the final instruction is not
    // a control-flow instruction. In that case an emitted wait at the block
    // boundary still guards the first instruction in its sole fallthrough
    // successor. Continue the pending wait group across that structural split.
    // A block ending in a wait cannot itself branch, so multiple successors
    // remain explicitly indeterminate.
    if (current_cfg_view_index_ && *current_cfg_view_index_ < cfg_views_.size()) {
      const CfgBlockView &block = cfg_views_[*current_cfg_view_index_];
      if (block.successors.size() == 1) {
        const size_t successor_index = block.successors.front();
        if (successor_index < cfg_views_.size() &&
            !cfg_views_[successor_index].instructions.empty()) {
          const CfgInstructionView consumer = cfg_views_[successor_index].instructions.front();
          bool overlapping_counter_wait = false;
          if (const auto fields =
                  cfg_explicit_wait_fields(successor_index, *consumer.instruction, arch)) {
            for (size_t i = 0; i < fields->size(); ++i) {
              overlapping_counter_wait |=
                  (*fields)[i].has_value() && pending_group->emitted_counts[i].has_value();
            }
          }
          if ((!overlapping_counter_wait || relative_wait_operand(*consumer.instruction, arch)) &&
              !embedded_wait_fields(*consumer.instruction, arch).value()) {
            const size_t saved_block_index = *current_cfg_view_index_;
            current_cfg_view_index_ = successor_index;
            audit_counter_parity_group(*pending_group, *consumer.instruction,
                                       consumer.section_offset,
                                       pending_group->file_offset - pending_group->section_offset +
                                           consumer.section_offset,
                                       state, arch);
            current_cfg_view_index_ = saved_block_index;
            pending_group.reset();
            return;
          }
        }
      }
    }
    ++report_.counter_parity_indeterminate_groups;
    pending_group.reset();
  }

  [[nodiscard]] bool is_non_entry_function_prolog(uint64_t section_offset) const {
    return current_function_entry_ == section_offset;
  }

  [[nodiscard]] bool is_non_entry_function_return(const Instruction &instruction) const {
    if (!current_call_return_sreg_ || instruction.size() != static_cast<int>(sizeof(uint32_t)) ||
        (instruction.mnemonic() != "s_setpc_b64" && instruction.mnemonic() != "s_set_pc_i64") ||
        instruction.raw_encoding() == nullptr) {
      return false;
    }
    return static_cast<uint16_t>(instruction.raw_encoding()[0] & 0xffu) ==
           *current_call_return_sreg_;
  }

  [[nodiscard]] bool has_loop_carried_events(WaitCounterKind counter, const PendingState &at_wait,
                                             uint64_t wait_offset, rj_code_arch_t arch) {
    if (!current_cfg_view_index_ || *current_cfg_view_index_ >= cfg_views_.size())
      return false;
    const size_t block_index = *current_cfg_view_index_;
    const size_t index = counter_index(counter);
    if (at_wait.pending[index].empty())
      return false;
    return std::ranges::any_of(cfg_views_[block_index].predecessors, [&](size_t predecessor) {
      if (cfg_component_[predecessor] != cfg_component_[block_index] ||
          cfg_outputs_[predecessor].pending[index].empty())
        return false;
      // Replay the block's waits on the incoming requests alone. A full drain
      // before this group separates fresh requests from the previous iteration.
      PendingState incoming = cfg_outputs_[predecessor];
      for (const auto &view : cfg_views_[block_index].instructions) {
        if (view.section_offset >= wait_offset)
          break;
        if (view.instruction->is_waitcnt())
          (void)apply_waitcnt(incoming, *view.instruction, arch);
        else
          (void)apply_embedded_waitcnt(incoming, *view.instruction, arch);
        if (incoming.pending[index].empty())
          return false;
      }
      return std::ranges::any_of(incoming.pending[index], [&](const PendingEvent &carried) {
        return std::ranges::any_of(at_wait.pending[index], [&](const PendingEvent &event) {
          return same_event_identity(carried, event);
        });
      });
    });
  }

  [[nodiscard]] PendingState analyze_block(BasicBlock &block, const PendingState &input,
                                           const std::string &section_name,
                                           uint64_t file_offset_base, rj_code_arch_t arch,
                                           bool emit_diagnostics, size_t block_index = 0) {
    PendingState state = input;
    std::optional<PendingWaitGroup> pending_wait_group;
    uint64_t section_offset = block.start_offset();
    for (const Instruction &inst : block.instructions()) {
      if (!emit_diagnostics && options_.check_counter_parity && arch == ROCJITSU_CODE_ARCH_RDNA3 &&
          (inst.mnemonic().starts_with("v_movrel") || inst.mnemonic().starts_with("v_swaprel"))) {
        relative_cfg_m0_[{block_index, &inst}] = state.m0;
      }
      if (!emit_diagnostics && options_.check_counter_parity && relative_wait_operand(inst, arch)) {
        // The worklist revisits this contextual block whenever its input changes.
        // Retain its final incoming value for lookahead across calls and loops.
        relative_cfg_waits_[{block_index, &inst}] =
            *resolved_wait_fields(state, inst, arch).value();
      }
      if (!emit_diagnostics && inst.is_waitcnt()) {
        // The path search tracks a single event. Keep the hardware ordering
        // facts for this wait across all its incoming analysis contexts.
        auto &facts = path_wait_facts_[&inst];
        for (const auto counter :
             {WaitCounterKind::Load, WaitCounterKind::Store, WaitCounterKind::Ds}) {
          const uint64_t bit = uint64_t{1} << counter_index(counter);
          if (counter_has_event_kind(state, counter, WaitEventKind::FlatLoad))
            facts.flat_load_routes |= bit;
          if (counter_has_event_kind(state, counter, WaitEventKind::FlatStore))
            facts.flat_store_routes |= bit;
        }
        for (size_t i = 0; i < state.pending.size(); ++i) {
          if (state.pending[i].empty())
            continue;
          auto mixed = counter_out_of_order(state, static_cast<WaitCounterKind>(i), arch);
          if (state.uncertain_order[i] || mixed.failed() || mixed.value())
            facts.unordered |= uint64_t{1} << i;
        }
      }
      if (emit_diagnostics) {
        update_counter_parity_group(pending_wait_group, state, inst, section_name, section_offset,
                                    file_offset_base + section_offset, arch);
      }
      analyze_instruction(state, inst, section_name, section_offset,
                          file_offset_base + section_offset, arch, emit_diagnostics);
      if (emit_diagnostics)
        ++report_.instructions_analyzed;
      section_offset += static_cast<uint64_t>(inst.size());
      if (emit_diagnostics && should_stop_after_diagnostic())
        break;
    }
    if (emit_diagnostics)
      finish_counter_parity_block(pending_wait_group, state, arch);
    return state;
  }

  [[nodiscard]] static bool expert_waits_enabled(const PendingState &state) {
    return !state.expert_scheduling.known || state.expert_scheduling.enabled;
  }

  static void clear_expert_wait_state(PendingState &state) {
    state.expert_delay = {};
    for (const auto counter : {WaitCounterKind::VmVsrc, WaitCounterKind::VaVdst}) {
      const auto i = counter_index(counter);
      state.pending[i].clear();
      state.pending_event_ages[i] = {};
      state.uncertain_order[i] = false;
    }
    clear_va_vdst_hazards(state.va_vdst_hazards);
  }

  static void clear_all_sgpr_hazards(SgprHazardState &state) {
    clear_salu_sgpr_hazards(state);
    clear_valu_sgpr_hazards(state);
    clear_valu_vcc_hazard(state);
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

  [[nodiscard]] static bool is_vm_vsrc_zero_wait(const Instruction &inst) {
    if (inst.mnemonic() != "s_wait_alu")
      return false;
    const Operand *op = inst.src_operand(0);
    if (!op)
      return false;
    constexpr uint32_t kDepctrVmVsrcShift = 2;
    constexpr uint32_t kDepctrVmVsrcWidth = 3;
    return depctr_field(static_cast<uint32_t>(op->encoding_value()), kDepctrVmVsrcShift,
                        kDepctrVmVsrcWidth) == 0;
  }

  [[nodiscard]] static std::optional<int64_t> first_operand_value(const Instruction &inst) {
    const Operand *op = inst.src_operand(0);
    if (!op)
      return std::nullopt;
    return static_cast<int64_t>(static_cast<int32_t>(op->encoding_value()));
  }

  [[nodiscard]] static bool is_xcnt_drain(const Instruction &inst,
                                          bool immediately_after_mode_setreg = false) {
    if ((inst.flags() &
         (BRANCH | COND_BRANCH | INDIRECT_BRANCH | INDIRECT_CALL | PROGRAM_TERMINATOR)) != 0)
      return true;

    const std::string_view mnemonic = inst.mnemonic();
    return mnemonic == "s_trap" || mnemonic == "s_getreg_b32" ||
           starts_with(mnemonic, "s_setreg") || starts_with(mnemonic, "s_sendmsg") ||
           starts_with(mnemonic, "s_barrier_wait") || starts_with(mnemonic, "s_barrier_signal") ||
           (mnemonic == "s_set_vgpr_msb" && !immediately_after_mode_setreg);
  }

  static void apply_xcnt_drain(PendingState &state, const Instruction &inst, rj_code_arch_t arch,
                               bool immediately_after_mode_setreg) {
    if (arch == ROCJITSU_CODE_ARCH_CDNA5 && is_xcnt_drain(inst, immediately_after_mode_setreg))
      apply_xcnt_wait(state, 0);
  }

  bool apply_vgpr_msb_mode(PendingState &state, const Instruction &inst, rj_code_arch_t arch,
                           bool immediately_after_mode_setreg) const {
    if (arch != ROCJITSU_CODE_ARCH_CDNA5)
      return false;

    const std::string_view mnemonic = inst.mnemonic();
    if (mnemonic == "s_set_vgpr_msb") {
      // GFX1250 silently drops this instruction when it immediately follows
      // S_SETREG_IMM32_B32 targeting MODE. AMDGPULowerVGPREncoding inserts an
      // S_NOP for exactly this hazard.
      if (!immediately_after_mode_setreg) {
        const auto value = first_operand_value(inst);
        state.vgpr_msb.mode = value ? static_cast<uint8_t>(*value) : 0;
        state.vgpr_msb.known = true;
      }
      return true;
    }

    if (mnemonic != "s_setreg_imm32_b32" && mnemonic != "s_setreg_b32")
      return false;
    const Operand *hwreg = inst.dst_operand(0);
    if (hwreg == nullptr)
      return false;
    constexpr uint32_t kHwRegMode = 1;
    if ((static_cast<uint32_t>(hwreg->encoding_value()) & 0x3fu) != kHwRegMode)
      return false;

    const bool immediate = mnemonic == "s_setreg_imm32_b32" && inst.raw_encoding() &&
                           inst.size() >= 2 * static_cast<int>(sizeof(uint32_t));
    const auto value = immediate ? std::optional{inst.raw_encoding()[1]} : std::nullopt;
    const auto slice = amdgpu::decode_vgpr_msb_hwreg(hwreg->encoding_value());
    if (slice.begin <= 24 && slice.begin + slice.width > 24) {
      state.prefetch_enabled =
          value ? std::optional{((*value >> (24 - slice.begin)) & 1u) != 0} : std::nullopt;
    }
    if (setreg_vgpr_msb_fixup_) {
      // gfx1250 reads all four VGPR banks from the unshifted source bits
      // [19:12], even for MODE writes to unrelated fields such as REPLAY_MODE.
      const auto layout = static_cast<uint8_t>(value.value_or(0) >> amdgpu::VGPR_MSB_MODE_SHIFT);
      state.vgpr_msb = {.mode = amdgpu::mode_layout_to_set_vgpr_msb(layout),
                        .known = value.has_value()};
      state.vgpr_msb_setreg_hazard = immediate;
    } else {
      auto banks = amdgpu::unpack_vgpr_msb_banks(
          state.vgpr_msb.known ? std::optional{state.vgpr_msb.mode} : std::nullopt);
      amdgpu::apply_vgpr_msb_mode_write(banks, hwreg->encoding_value(), value);
      const auto packed = amdgpu::pack_vgpr_msb_banks(banks);
      state.vgpr_msb = {.mode = packed.value_or(0), .known = packed.has_value()};
    }
    return false;
  }

  static void update_vector_skip(std::optional<bool> &skip, const Instruction &inst,
                                 rj_code_arch_t arch) {
    const auto name = inst.mnemonic();
    if (name == "s_setvskip") {
      const auto source = inst.src_operand(0)->const_value();
      const auto bit = inst.src_operand(1)->const_value();
      if (source == 0)
        skip = false;
      else if (source && bit)
        skip = ((static_cast<uint32_t>(*source) >> (*bit & 31u)) & 1u) != 0;
      else
        skip.reset();
      return;
    }
    if (name != "s_setreg_b32" && name != "s_setreg_imm32_b32")
      return;
    if (arch != ROCJITSU_CODE_ARCH_CDNA1 && arch != ROCJITSU_CODE_ARCH_CDNA2 &&
        arch != ROCJITSU_CODE_ARCH_CDNA3 && arch != ROCJITSU_CODE_ARCH_CDNA4)
      return;
    const auto *hwreg = inst.dst_operand(0);
    if (!hwreg)
      return;
    const auto slice = amdgpu::decode_vgpr_msb_hwreg(hwreg->encoding_value());
    constexpr unsigned skip_bit = 28;
    if (slice.id != amdgpu::MODE_HWREG || slice.begin > skip_bit ||
        slice.begin + slice.width <= skip_bit)
      return;
    if (name == "s_setreg_imm32_b32" && inst.raw_encoding() && inst.size() == 8)
      skip = ((inst.raw_encoding()[1] >> (skip_bit - slice.begin)) & 1u) != 0;
    else
      skip.reset();
  }

  static std::optional<ExpertSchedulingState> expert_scheduling_mode_write(const Instruction &inst,
                                                                           rj_code_arch_t arch) {
    if (arch != ROCJITSU_CODE_ARCH_CDNA5 && arch != ROCJITSU_CODE_ARCH_RDNA4)
      return std::nullopt;
    if (inst.mnemonic() != "s_setreg_imm32_b32" && inst.mnemonic() != "s_setreg_b32")
      return std::nullopt;

    constexpr uint32_t kHwRegWaveSchedMode = 26;
    constexpr uint32_t kSchedModeValue = 2;
    const Operand *hwreg = inst.dst_operand(0);
    if (hwreg == nullptr)
      return std::nullopt;

    const uint32_t value = static_cast<uint32_t>(hwreg->encoding_value());
    const uint32_t reg_id = value & 0x3fu;
    const uint32_t offset = (value >> 6u) & 0x1fu;
    const uint32_t size = ((value >> 11u) & 0x1fu) + 1u;
    if (reg_id != kHwRegWaveSchedMode || offset >= 2)
      return std::nullopt;

    if (inst.mnemonic() != "s_setreg_imm32_b32" || inst.raw_encoding() == nullptr ||
        inst.size() < 2 * static_cast<int>(sizeof(uint32_t)) || offset != 0 || size < 2) {
      return ExpertSchedulingState{.enabled = true, .known = false};
    }
    return ExpertSchedulingState{.enabled = (inst.raw_encoding()[1] & 0x3u) == kSchedModeValue};
  }

  [[nodiscard]] static bool is_exec_masked_def(RegisterRef ref) {
    return ref.cls == RegClass::VGPR || ref.cls == RegClass::ACC_VGPR;
  }

  static void expand_vgpr_msb_ref(RegisterSet &regs, RegisterRef ref, const Operand &op,
                                  const VgprMsbState &state, rj_code_arch_t arch) {
    if (arch != ROCJITSU_CODE_ARCH_CDNA5 || ref.cls != RegClass::VGPR ||
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

  [[nodiscard]] static bool is_vop3_carry_in(std::string_view mnemonic) {
    return mnemonic == "v_add_co_ci_u32" || mnemonic == "v_sub_co_ci_u32" ||
           mnemonic == "v_subrev_co_ci_u32";
  }

  [[nodiscard]] static bool add_adjusted_source_use(InstDefUse &du, const Instruction &inst,
                                                    const Operand &op, int operand_index,
                                                    uint32_t wavefront_size) {
    const std::string_view mnemonic = inst.mnemonic();
    const bool is_vop3_mask =
        (starts_with(mnemonic, "v_cndmask_b32") || starts_with(mnemonic, "v_cndmask_b16")) &&
        operand_index == 2;
    const bool is_vopd_mask = mnemonic.find("v_dual_cndmask_b32") != std::string_view::npos &&
                              op.vgpr_msb_role() == amdgpu::VgprMsbRole::Src2;
    const bool is_vop3_carry_mask = operand_index == 2 && is_vop3_carry_in(mnemonic);
    if (is_vop3_mask || is_vopd_mask || is_vop3_carry_mask) {
      if (auto ref = op.to_register_ref(); ref && ref->cls == RegClass::SGPR) {
        // LLVM's BoolRC predicate is one physical SGPR in Wave32 and a pair in
        // Wave64. VOP3/VOPD keep the maximum-width encoded operand in the
        // generated decoder, just like VOPC destinations above.
        ref->width = wavefront_size == 32 ? 1 : 2;
        du.uses.expand(*ref);
        return true;
      }
    }
    if (operand_index == 0 && starts_with(mnemonic, "s_buffer_")) {
      if (auto ref = op.to_register_ref(); ref && ref->cls == RegClass::SGPR) {
        ref->width = 4;
        du.uses.expand(*ref);
      }
      return true;
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

  void update_scalar_offsets(PendingState &state, const Instruction &inst, const InstDefUse &du,
                             rj_code_arch_t arch) const {
    if (inst.flags() & INDIRECT_CALL) {
      state.m0 = {};
      state.sgpr_offsets.clear();
      return;
    }
    const RegisterRef m0{RegClass::M0, 0, 1};
    const bool writes_m0 = du.defs.contains(m0);
    if (!writes_m0 && (!track_sgpr_offsets_ ||
                       !du.defs.intersects({RegClass::SGPR, 0, amdgpu::kVccSelectorFirst})))
      return;

    std::optional<RegisterRef> dst;
    ScalarOffsetValues result;
    const std::string_view name = inst.mnemonic();
    const bool add = name == "s_add_i32" || name == "s_add_u32";
    const bool mask = track_sgpr_offsets_ && name == "s_and_b32";
    const bool bit_xor = track_sgpr_offsets_ && name == "s_xor_b32";
    const bool bit_or = track_sgpr_offsets_ && name == "s_or_b32";
    const bool shift = track_sgpr_offsets_ && name == "s_lshl_b32";
    if (name == "s_mov_b32" || add || mask || bit_xor || bit_or || shift) {
      dst = register_operand(*inst.dst_operand(0), arch);
      const Operand *src = inst.src_operand(0);
      const Operand *rhs = inst.num_src_operands() > 1 ? inst.src_operand(1) : nullptr;
      if ((mask || bit_xor || bit_or) && src->const_value())
        std::swap(src, rhs);
      if (const auto value = src->const_value()) {
        result = {{static_cast<uint32_t>(*value)}, 1};
      } else if (const auto reg = register_operand(*src, arch)) {
        if (reg->cls == RegClass::M0)
          result = state.m0;
        else if (reg->cls == RegClass::SGPR) {
          const auto known = state.sgpr_offsets.find(reg->index);
          if (known != state.sgpr_offsets.end())
            result = known->second;
        }
      }
      if (rhs) {
        const auto literal = rhs->const_value();
        if (!literal) {
          result = {};
        } else {
          const uint32_t value = static_cast<uint32_t>(*literal);
          if (mask && result.count == 0 &&
              std::popcount(value) <= std::countr_zero(result.literals.size())) {
            // All submasks, including zero, are possible with an unknown source.
            uint32_t submask = 0;
            do {
              result.literals[result.count++] = submask;
              submask = (submask - value) & value;
            } while (submask != 0);
          } else {
            for (unsigned i = 0; i < result.count; ++i) {
              uint32_t &offset = result.literals[i];
              if (mask)
                offset &= value;
              else if (bit_xor)
                offset ^= value;
              else if (bit_or)
                offset |= value;
              else if (shift)
                offset <<= value & 31u;
              else
                offset += value;
            }
            auto first = result.literals.begin();
            auto end = first + result.count;
            std::sort(first, end);
            result.count = static_cast<unsigned>(std::unique(first, end) - first);
            std::fill(first + result.count, end, 0u);
          }
        }
      }
    }
    if (writes_m0)
      state.m0 = {};
    std::erase_if(state.sgpr_offsets, [&](const auto &entry) {
      return du.defs.contains({RegClass::SGPR, entry.first, 1});
    });
    if (!dst || result.count == 0 || (dst->cls != RegClass::SGPR && dst->cls != RegClass::M0))
      return;
    // A prior asynchronous write may still replace this scalar move's result.
    for (const auto &queue : state.pending)
      for (const auto &event : queue)
        if (event.check_uses && event.regs.contains(*dst))
          return;
    if (dst->cls == RegClass::M0) {
      state.m0 = result;
    } else if (track_sgpr_offsets_) {
      const unsigned guaranteed_sgprs =
          current_kernel_ && current_kernel_->min_sgprs
              ? current_kernel_->min_sgprs
              : (isa_properties(arch).descriptor_sgpr_count_encoded ? 16u
                                                                    : amdgpu::kVccSelectorFirst);
      if (dst->index < guaranteed_sgprs)
        state.sgpr_offsets.emplace(dst->index, result);
    }
  }

  [[nodiscard]] static std::optional<uint16_t>
  resolve_scratch_relative_sgpr(const Instruction &inst, const Instruction &previous,
                                rj_code_arch_t arch) {
    if (inst.mnemonic() != "s_movreld_b32" || previous.mnemonic() != "s_mov_b32" ||
        inst.is_vskip_affected() || previous.is_vskip_affected() ||
        !low_selector(previous.dst_operand(0), amdgpu::scalar_m0_selector(arch)))
      return std::nullopt;
    const auto m0 = previous.src_operand(0)->const_value();
    const auto *dst = inst.dst_operand(0);
    if (!m0 || !dst || !dst->has_register_selector())
      return std::nullopt;
    // Both SALU instructions are unmasked and share a block. A destination
    // outside the guaranteed allocation can suppress the write, so retaining
    // its resolved SGPR as a possible def is conservative.
    const auto selector = amdgpu::relative_scalar_selector(
        arch, dst->encoding_value(), static_cast<uint32_t>(*m0), 1, true, false);
    if (!selector || *selector >= amdgpu::kVccSelectorFirst)
      return std::nullopt;
    return static_cast<uint16_t>(*selector);
  }

  // A MOVREL indexes a physical register after bank selection. Resolve every
  // possible M0 value, or all addressable targets if its value is unknown.
  void relative_def_use(InstDefUse &du, const Instruction &inst, const ScalarOffsetValues &m0,
                        const VgprMsbState &banks, rj_code_arch_t arch) const {
    const std::string_view name = inst.mnemonic();
    const bool vector = name.starts_with("v_");
    const bool swap = name.starts_with("v_swaprel");
    const bool split = swap || name.find("movrelsd_2") != std::string_view::npos;
    const bool relative_src = swap || name.find("movrels") != std::string_view::npos;
    const bool relative_dst = swap || name.find("movreld") != std::string_view::npos ||
                              name.find("movrelsd") != std::string_view::npos;
    const auto &props = isa_properties(arch);
    const unsigned min_vgprs =
        current_kernel_ && current_kernel_->min_vgprs
            ? current_kernel_->min_vgprs
            : descriptor_vgpr_count_granule_for_wavefront(arch, wavefront_size_).value_or(0);
    const unsigned max_vgprs =
        current_kernel_ && current_kernel_->max_vgprs
            ? std::min(current_kernel_->max_vgprs, props.max_addressable_vgprs_per_wf)
            : props.max_addressable_vgprs_per_wf;
    const unsigned min_sgprs =
        current_kernel_ && current_kernel_->min_sgprs
            ? current_kernel_->min_sgprs
            : (props.descriptor_sgpr_count_encoded ? 16u : amdgpu::kVccSelectorFirst);
    const unsigned max_sgprs = current_kernel_ && current_kernel_->max_sgprs
                                   ? current_kernel_->max_sgprs
                                   : amdgpu::kVccSelectorFirst;

    du.uses.expand({RegClass::M0, 0, 1});
    const auto resolve = [&](const Operand &op, bool relative, bool dst, RegisterSet &regs) {
      if (!relative) {
        if (auto ref = register_operand(op, arch))
          expand_vgpr_msb_ref(regs, *ref, op, banks, arch);
        return true;
      }
      const unsigned width = std::max(1, op.size_bits() / 32);
      bool valid = false;
      bool may_read_zero = m0.count == 0;
      if (vector) {
        const auto ref = op.to_register_ref();
        if (ref && ref->cls == RegClass::VGPR) {
          const bool banked =
              arch == ROCJITSU_CODE_ARCH_CDNA5 && op.vgpr_msb_role() != amdgpu::VgprMsbRole::None;
          const unsigned first = banked && banks.known ? banks.for_role(op.vgpr_msb_role()) : 0;
          const unsigned last = banked && !banks.known ? 3 : first;
          for (unsigned bank = first; bank <= last; ++bank) {
            const unsigned base = bank * 256 + ref->index;
            const auto add = [&](uint32_t offset) {
              auto index = amdgpu::relative_vgpr_index(base, offset, width, max_vgprs, arch);
              if (index) {
                valid = true;
                regs.expand({RegClass::VGPR, static_cast<uint16_t>(*index), ref->width});
              }
              may_read_zero |= !index || *index + width > min_vgprs;
            };
            if (m0.count != 0)
              for (uint32_t value : m0.values())
                add(split ? (value >> (dst ? 16 : 0)) & 0x3ff : value);
            else
              for (unsigned offset = 0; base + offset < max_vgprs; ++offset)
                add(offset);
          }
        } else {
          may_read_zero = true;
        }
      } else {
        const auto add = [&](uint32_t m0_value) {
          const auto selector = amdgpu::relative_scalar_selector(arch, op.encoding_value(),
                                                                 m0_value, width, dst, split);
          if (selector) {
            const auto range = amdgpu::resolve_scalar_storage_range(arch, *selector, width);
            if (range->storage != amdgpu::ScalarRegisterStorage::SGPR ||
                range->index + width <= max_sgprs) {
              valid = true;
              if (auto ref = range->register_ref())
                regs.expand(*ref);
              may_read_zero |= range->storage == amdgpu::ScalarRegisterStorage::SGPR &&
                               range->index + width > min_sgprs;
              return;
            }
          }
          may_read_zero = true;
        };
        if (m0.count != 0)
          for (uint32_t value : m0.values())
            add(value);
        else
          for (unsigned offset = 0; offset <= amdgpu::kTtmpSelectorLast; ++offset)
            add(split && dst ? offset << 16 : offset);
      }
      // An out-of-allocation source falls back to register zero. SWAPREL
      // suppresses both sides when either indexed register is unavailable.
      if (!dst && !swap && may_read_zero)
        regs.expand({vector ? RegClass::VGPR : RegClass::SGPR, 0, static_cast<uint8_t>(width)});
      return valid;
    };

    const auto &dst = *inst.dst_operand(0);
    const auto &src = *(swap ? inst.dst_operand(1) : inst.src_operand(0));
    RegisterSet writes, reads;
    if (!resolve(dst, relative_dst, true, writes))
      return;
    if (!resolve(src, relative_src, false, reads) && swap)
      return;
    du.defs |= writes;
    du.uses |= reads;
    if (swap) {
      du.defs |= reads;
      du.uses |= writes;
    }
    du.has_exec_masked_vector_def = vector;
  }

  // A proven out-of-allocation result nullifies its vector memory operation.
  // CDNA2–4 still occupy a counter position. GFX12 can reserve a conditional
  // empty-EXEC position while an older request is outstanding.
  [[nodiscard]] bool rejects_memory_destination(const Instruction &inst, const VgprMsbState &banks,
                                                rj_code_arch_t arch) const {
    if (!current_kernel_ || !current_kernel_->max_memory_vgprs || source_allocation_may_change_ ||
        !inst.is_memory_op())
      return false;
    const bool gfx12 = arch == ROCJITSU_CODE_ARCH_CDNA5 || arch == ROCJITSU_CODE_ARCH_RDNA4;
    bool buffer_load = false;
    if (gfx12) {
      const auto name = inst.mnemonic();
      buffer_load = name.starts_with("buffer_load_") || name.starts_with("tbuffer_load_");
      if (!(inst.flags() & SIMPLE_MEMORY_RESULT) && !name.starts_with("flat_") && !buffer_load &&
          !(is_vmem_atomic(name) && (name.starts_with("global_") || name.starts_with("buffer_"))))
        return false;
    }
    amdgpu::RegisterModifiers modifiers;
    if (buffer_load)
      inst.amdgpu_register_modifiers(modifiers);
    for (int i = 0; i < inst.num_dst_operands(); ++i) {
      const auto *dst = inst.dst_operand(i);
      const auto reg = dst->decoded_vgpr();
      if (!reg)
        continue;
      const bool acc = !gfx12 && reg->index >= amdgpu::ACC_VGPR_OFFSET;
      const uint32_t base = acc ? amdgpu::ACC_VGPR_OFFSET : 0;
      const uint32_t count =
          acc ? current_kernel_->max_memory_acc_vgprs : current_kernel_->max_memory_vgprs;
      const uint32_t bank = arch == ROCJITSU_CODE_ARCH_CDNA5 && banks.known
                                ? banks.for_role(dst->vgpr_msb_role()) * 256u
                                : 0;
      const uint32_t result_width =
          gfx12 && wavefront_size_ == 64 && (inst.flags() & COMPACT_WAVE64_MEMORY_RESULT)
              ? std::max(1u, uint32_t{reg->width} / 2)
              : reg->width;
      const uint32_t width = result_width + (buffer_load && modifiers.buffer_tfe);
      // If the bank is unknown, bank zero is the smallest possible address.
      // Crossing the limit there proves rejection for every possible bank.
      if (reg->index - base + bank + width > count)
        return true;
    }
    return false;
  }

  [[nodiscard]] std::vector<ClassifiedEvent>
  effective_events(const Instruction &inst, const VgprMsbState &banks, rj_code_arch_t arch) const {
    auto events = classify_events(inst, arch).value();
    if ((arch == ROCJITSU_CODE_ARCH_CDNA5 || arch == ROCJITSU_CODE_ARCH_RDNA4) &&
        rejects_memory_destination(inst, banks, arch)) {
      // The nullified instruction reads no address and returns no value,
      // but an older outstanding request can still force its counter slots.
      for (auto &event : events) {
        event.registers = TrackedRegisterSource::None;
        event.check_uses = false;
        event.check_defs = false;
        event.check_exec_defs = false;
        event.check_memory_order = false;
        event.check_program_end = false;
      }
    }
    return events;
  }

  [[nodiscard]] InstDefUse inst_def_use_for_waitcheck(const Instruction &inst,
                                                      const VgprMsbState &state,
                                                      rj_code_arch_t arch, uint32_t wavefront_size,
                                                      const ScalarOffsetValues &m0 = {}) const {
    InstDefUse du(inst, wavefront_size);
    du.defs = {};
    du.uses = {};
    du.has_exec_masked_vector_def = false;
    du.has_predicated_def = inst.flags() & PREDICATED_DEF;
    if (rejects_memory_destination(inst, state, arch))
      return du;

    const bool relative = inst.mnemonic().starts_with("s_movrel") ||
                          inst.mnemonic().starts_with("v_movrel") ||
                          inst.mnemonic().starts_with("v_swaprel");
    if (relative)
      relative_def_use(du, inst, m0, state, arch);
    if (!relative && !is_cdna4_mubuf_lds_load(inst, arch)) {
      for (int i = 0; i < inst.num_dst_operands(); ++i) {
        const Operand *op = inst.dst_operand(i);
        if (op == nullptr)
          continue;
        if (auto ref = register_operand(*op, arch))
          add_def(du, inst.wave_sized_destination(*op, *ref, wavefront_size), *op, state, arch);
      }
    }
    inst.implicit_defs(du.defs);

    for (int i = 0; !relative && i < inst.num_src_operands(); ++i) {
      const Operand *op = inst.src_operand(i);
      if (op == nullptr)
        continue;
      if (add_adjusted_source_use(du, inst, *op, i, wavefront_size))
        continue;
      if (auto ref = register_operand(*op, arch))
        expand_vgpr_msb_ref(du.uses, *ref, *op, state, arch);
    }
    RegisterSet implicit_uses;
    inst.implicit_uses(implicit_uses);
    if (arch == ROCJITSU_CODE_ARCH_CDNA5) {
      // The flat hook loses each operand's VGPR-MSB role. Keep its non-VGPR
      // reads (including encoded-field SGPRs), and resolve implicit VGPR reads
      // through their operands, just as for explicit sources above.
      implicit_uses.clear_class(RegClass::VGPR);
      std::vector<const Operand *> operands;
      inst.implicit_use_operands(operands);
      for (const Operand *op : operands) {
        if (op == nullptr)
          continue;
        if (auto ref = op->to_register_ref(); ref && ref->cls == RegClass::VGPR)
          expand_vgpr_msb_ref(implicit_uses, *ref, *op, state, arch);
      }
    }
    // D16 memory loads preserve the opposite half for liveness, but the two
    // asynchronous half writes do not consume one another at issue time.
    if (const auto partial = partial_d16_load_def(inst, tracks_d16_halves_))
      implicit_uses.erase(partial->reg);
    du.uses |= implicit_uses;
    return du;
  }

  static void clear_matching_barrier_scc_write(PendingState &state, const Instruction &inst) {
    if (inst.mnemonic() != "s_barrier_wait")
      return;
    const auto barrier_id = first_barrier_id(inst);
    if (!barrier_id)
      return;

    auto &events = state.pending[counter_index(WaitCounterKind::Km)];
    std::erase_if(events, [&](const PendingEvent &event) {
      return event.kind == WaitEventKind::SccWrite && event.barrier_id == barrier_id;
    });
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

  static bool has_implicit_registers(const Instruction &inst) {
    RegisterSet registers;
    inst.implicit_uses(registers);
    inst.implicit_defs(registers);
    return !registers.none();
  }

  [[nodiscard]] static bool instruction_has_register_operand(const Instruction &inst) {
    if (has_implicit_registers(inst) || inst.mnemonic() == "s_setreg_imm32_b32")
      return true;
    for (int i = 0; i < inst.num_dst_operands(); ++i) {
      const Operand *operand = inst.dst_operand(i);
      if (operand != nullptr &&
          (operand->to_register_ref() || operand->to_special_reg_class() ||
           (operand->has_register_selector() && operand->encoding_value() >= 235 &&
            operand->encoding_value() <= 239)))
        return true;
    }
    for (int i = 0; i < inst.num_src_operands(); ++i) {
      const Operand *operand = inst.src_operand(i);
      if (operand != nullptr &&
          (operand->to_register_ref() || operand->to_special_reg_class() ||
           (operand->has_register_selector() && operand->encoding_value() >= 235 &&
            operand->encoding_value() <= 239)))
        return true;
    }
    return false;
  }

  [[nodiscard]] static bool is_sgpr_hazard_control_transfer(const Instruction &inst) {
    constexpr uint64_t kControlTransferFlags = INDIRECT_CALL | INDIRECT_BRANCH;
    return (inst.flags() & kControlTransferFlags) != 0;
  }

  [[nodiscard]] static bool instruction_waits_for_valu_sgpr_writes(const Instruction &inst) {
    // Scalar memory is handled by apply_sgpr_hazard_memory_cull(). For SALU,
    // model the issue stall used by LLVM's
    // AMDGPUInsertDelayAlu::instructionWaitsForSGPRWrites: any explicit or
    // implicit register operand waits for VA_SDST==0, even when unrelated to
    // the outstanding destination. This deliberately differs from the more
    // conservative data-dependency repair in AMDGPUWaitSGPRHazards; see
    // llvm-project#131111 and llvm-project#145728.
    return !is_scalar_memory_op(inst.mnemonic()) && starts_with(inst.mnemonic(), "s_") &&
           instruction_has_register_operand(inst);
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

  [[nodiscard]] static bool is_gfx1250_smem(std::string_view mnemonic) {
    return is_scalar_memory_op(mnemonic);
  }

  [[nodiscard]] static bool is_scratch_store(std::string_view mnemonic) {
    return starts_with(mnemonic, "scratch_store");
  }

  [[nodiscard]] static bool is_ds_store(std::string_view mnemonic) {
    return starts_with(mnemonic, "ds_store") || starts_with(mnemonic, "ds_write") ||
           starts_with(mnemonic, "ds_cmpstore") || starts_with(mnemonic, "ds_mskor");
  }

  [[nodiscard]] static bool is_async_or_tensor_producer(std::string_view mnemonic) {
    return is_async_lds_load(mnemonic) || is_async_lds_store(mnemonic) ||
           mnemonic == "ds_atomic_async_barrier_arrive_b64" || is_tensor_lds_load(mnemonic) ||
           is_tensor_lds_store(mnemonic);
  }

  [[nodiscard]] static bool is_async_ordering_consumer(const PendingEvent &event,
                                                       std::string_view mnemonic) {
    if (event.kind == WaitEventKind::AsyncBarrier)
      return starts_with(mnemonic, "s_barrier_wait");
    if (is_async_or_tensor_producer(mnemonic))
      return false;
    return starts_with(mnemonic, "ds_") || starts_with(mnemonic, "s_barrier_wait");
  }

  [[nodiscard]] static bool is_tensor_ordering_consumer(const PendingEvent &event,
                                                        std::string_view mnemonic) {
    (void)event;
    if (is_async_or_tensor_producer(mnemonic))
      return false;
    return starts_with(mnemonic, "ds_") || starts_with(mnemonic, "s_barrier_wait");
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

  [[nodiscard]] static uint8_t memory_source_mask(const Instruction &inst, RegisterRef word) {
    // A store can name the same VGPR as both data and an address component.
    // Union the bytes actually read by each decoded operand before narrowing.
    uint8_t bytes = 0;
    for (int i = 0; i < inst.num_src_operands(); ++i) {
      const Operand *op = inst.src_operand(i);
      if (!op || !inst.tracks_explicit_source(i))
        continue;
      const auto ref = op->to_register_ref();
      if (ref && ref->cls == word.cls && word.index >= ref->index &&
          word.index < ref->index + ref->width)
        bytes |= op->register_byte_mask();
    }
    // An implicit use has no decoded subregister selection.
    if (!bytes)
      return kVgprFull32Mask;
    return ((bytes & 0x3) ? kVgprLow16Mask : 0) | ((bytes & 0xc) ? kVgprHigh16Mask : 0);
  }

  [[nodiscard]] uint8_t instruction_register_mask(const Instruction &inst, RegisterRef ref,
                                                  WaitcheckAccessKind access) const {
    if (access == WaitcheckAccessKind::Use && ref.cls == RegClass::VGPR && tracks_d16_halves_ &&
        inst.is_memory_op())
      return memory_source_mask(inst, ref);
    const auto partial = access == WaitcheckAccessKind::Def
                             ? partial_d16_load_def(inst, tracks_d16_halves_)
                             : std::nullopt;
    return partial && partial->reg == ref ? partial->mask : kVgprFull32Mask;
  }

  [[nodiscard]] static uint8_t vcc_access_halves(const Instruction &inst, const InstDefUse &du,
                                                 WaitcheckAccessKind access, rj_code_arch_t arch,
                                                 uint32_t wave_size,
                                                 const ScalarOffsetValues &m0 = {}) {
    const bool read = access == WaitcheckAccessKind::Use;
    const auto name = inst.mnemonic();
    const bool relative_scalar =
        read ? name.starts_with("s_movrels")
             : name.starts_with("s_movreld") || name.starts_with("s_movrelsd");
    if (relative_scalar) {
      if (!(read ? du.uses : du.defs).contains({RegClass::VCC, 0, 1}))
        return 0;
      if (m0.count == 0)
        return 3;
      const Operand *op = read ? inst.src_operand(0) : inst.dst_operand(0);
      const unsigned width = std::max(1, op->size_bits() / 32);
      const bool split = name.starts_with("s_movrelsd_2");
      uint8_t halves = 0;
      for (uint32_t value : m0.values()) {
        const auto selector = amdgpu::relative_scalar_selector(arch, op->encoding_value(), value,
                                                               width, !read, split);
        if (selector &&
            (*selector == amdgpu::kVccSelectorFirst || *selector == amdgpu::kVccSelectorLast))
          halves |= static_cast<uint8_t>(((1u << std::min(2u, width)) - 1u)
                                         << (*selector - amdgpu::kVccSelectorFirst)) &
                    3u;
      }
      return halves;
    }
    const uint8_t wave_mask = wave_size == 32 ? 1u : 3u;
    uint8_t halves = 0;
    const int count = read ? inst.num_src_operands() : inst.num_dst_operands();
    for (int i = 0; i < count; ++i) {
      const Operand *op = read ? inst.src_operand(i) : inst.dst_operand(i);
      if (!op || (read && !inst.tracks_explicit_source(i)))
        continue;
      // VCCZ and the fieldless condition operand observe the active wave's
      // mask. Explicit scalar accesses to VCC_HI still select that dword.
      if ((read && op->has_register_selector() && op->encoding_value() == 251) ||
          (op->is_fieldless() && op->to_special_reg_class() == RegClass::VCC)) {
        halves |= wave_mask;
      } else if (const auto ref = register_operand(*op, arch); ref && ref->cls == RegClass::VCC) {
        auto sized = read ? *ref : inst.wave_sized_destination(*op, *ref, wave_size);
        const bool mask_src =
            read &&
            (inst.mnemonic().starts_with("v_cndmask") || is_vop3_carry_in(inst.mnemonic())) &&
            i == 2;
        if (mask_src || op->is_wave_mask())
          sized.width = wave_size / 32;
        halves |=
            static_cast<uint8_t>(((1u << std::min<unsigned>(sized.width, 2)) - 1u) << sized.index) &
            3u;
      }
    }
    if (read ? instruction_uses_vcc_implicit(inst) ||
                   mask_branch_class(inst.mnemonic()) == RegClass::VCC
             : instruction_defines_vcc_implicit(inst, du))
      halves |= wave_mask;
    return halves;
  }

  [[nodiscard]] static bool only_defines_vcc_hi(const Instruction &inst, const InstDefUse &du,
                                                  rj_code_arch_t arch) {
    // Wave32 VCCZ depends on VCC_LO. Special, deferred, or indirect writes
    // may affect the condition even when their encoded destination is VCC_HI.
    if (inst.is_memory_op() || inst.mnemonic().starts_with("s_sendmsg") ||
        inst.mnemonic().starts_with("s_movrel"))
      return false;
    if (vcc_access_halves(inst, du, WaitcheckAccessKind::Def, arch, 32) != 2)
      return false;
    RegisterSet implicit_defs;
    inst.implicit_defs(implicit_defs);
    if (implicit_defs.contains({RegClass::VCC, 0, 1}))
      return false;
    for (int i = 0; i < inst.num_dst_operands(); ++i) {
      const auto *op = inst.dst_operand(i);
      if (low_selector(op, amdgpu::kVccSelectorLast) && op->size_bits() == 32)
        return true;
    }
    return false;
  }

  [[nodiscard]] std::optional<RegisterRef>
  first_dependency_intersection(const PendingEvent &event, const Instruction &inst,
                                const InstDefUse &du, WaitcheckAccessKind access,
                                rj_code_arch_t arch, const ScalarOffsetValues &m0) const {
    const RegisterSet &current_regs = access == WaitcheckAccessKind::Use ? du.uses : du.defs;
    std::optional<RegisterRef> result;
    event.regs.for_each([&](RegisterRef ref) {
      if (!result && ref.cls == RegClass::VCC) {
        const auto overlap =
            event.vcc_halves & vcc_access_halves(inst, du, access, arch, wavefront_size_, m0);
        if (overlap)
          result = RegisterRef{RegClass::VCC, static_cast<uint16_t>(overlap & 1 ? 0 : 1), 1};
      } else if (!result && current_regs.contains(ref) &&
                 (!event.partial_reg || *event.partial_reg != ref ||
                  (event.partial_reg_mask & instruction_register_mask(inst, ref, access))))
        result = ref;
    });
    return result;
  }

  void apply_implicit_xcnt_ordering(PendingState &state, const Instruction &inst,
                                    const InstDefUse &du,
                                    std::span<const ClassifiedEvent> current_events,
                                    rj_code_arch_t arch) {
    const bool current_vmem = std::ranges::any_of(current_events, [](const ClassifiedEvent &event) {
      return event.counter == WaitCounterKind::X && is_xcnt_vmem_kind(event.kind);
    });
    const bool current_smem = std::ranges::any_of(current_events, [](const ClassifiedEvent &event) {
      return event.counter == WaitCounterKind::X && event.kind == WaitEventKind::Smem;
    });
    if (!current_vmem && !current_smem)
      return;

    const size_t idx = counter_index(WaitCounterKind::X);
    const auto &pending = state.pending[idx];
    const bool opposite_group = current_vmem ? std::ranges::any_of(pending, is_xcnt_smem_event)
                                             : std::ranges::any_of(pending, is_xcnt_vmem_event);
    if (opposite_group) {
      if (!is_vmem_atomic(inst.mnemonic())) {
        apply_xcnt_group_wait(state, 0);
        return;
      }
      // A joined VMEM path can still carry earlier replay candidates when
      // another predecessor is SMEM. Only the scalar group is guaranteed to
      // drain along the edge entering this atomic.
      apply_xcnt_smem_group_wait(state);
    }

    if (!current_vmem)
      return;

    // VMEM address translations are ordered. If this VMEM instruction would
    // overwrite a source of an older VMEM operation, LLVM internally applies
    // just the X_CNT wait required for that dependency instead of emitting an
    // instruction. Preserve younger, unrelated translations.
    const bool single_access = is_vmem_atomic(inst.mnemonic());
    const auto overwrites_source = [&](const PendingEvent &event) {
      if (!is_xcnt_vmem_event(event) ||
          !first_dependency_intersection(event, inst, du, WaitcheckAccessKind::Def, arch, state.m0))
        return false;
      if (!inst.mnemonic().starts_with("flat_"))
        return true;
      // An LDS result cannot order earlier VMEM translations. A proven scratch
      // load uses VMEM; private returning atomics additionally require a dword.
      for (int i = 0; i < inst.num_dst_operands(); ++i) {
        const auto reg = inst.dst_operand(i)->to_register_ref();
        if (reg && reg->cls == RegClass::VGPR)
          return (!single_access || reg->width == 1) && cdna5_flat_uses_scratch(inst, state);
      }
      return false;
    };
    std::optional<uint32_t> required_count;
    for (const PendingEvent &event : state.pending[idx]) {
      if (!overwrites_source(event) || (single_access && !event.present_on_all_paths))
        continue;
      required_count = required_count ? std::min(*required_count, event.min_younger)
                                      : std::optional<uint32_t>(event.min_younger);
    }
    if (required_count)
      apply_xcnt_group_wait(state, *required_count);
    // A conditional witness cannot retire unrelated translations on the
    // alternate path, but whenever the witness exists its own source is safe.
    if (single_access)
      std::erase_if(state.pending[idx], overwrites_source);
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
      return ref.index == 1 ? "vcc_hi" : "vcc";
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

  static bool has_guarded_matrix_input(const Instruction &inst, rj_code_arch_t arch) {
    // CDNA3 software pipelines can unpack the buffer result before MFMA.
    return (inst.flags() & MATRIX_REGISTER_ACCESSES) ||
           (arch == ROCJITSU_CODE_ARCH_CDNA3 && inst.mnemonic() == "v_perm_b32");
  }

  // Scalar guards can control younger buffer loads, their matrix input
  // preparation, or an accumulator write.
  std::optional<bool> guarded_load_witness(const PendingEvent &event, const Instruction &inst,
                                           uint64_t offset, rj_code_arch_t arch,
                                           std::optional<WaitRange> field = {}) {
    if (event.counter != WaitCounterKind::Load)
      return true;
    auto query = [&](bool younger, std::optional<MaskedSccPredicate> mask,
                     std::optional<LoopParity> parity) {
      return has_cfg_path_with_event_pending(event, offset, arch, false, false, 0, field, {}, false,
                                             younger, mask, parity);
    };
    if (has_guarded_matrix_input(inst, arch)) {
      auto witness = query(true, {}, {});
      if (witness != false && arch == ROCJITSU_CODE_ARCH_CDNA3) {
        if (const auto pair = loop_parity()) {
          const auto other = query(false, {}, pair);
          if (other != true)
            return other;
        }
      }
      return witness;
    }
    if (inst.mnemonic().starts_with("v_accvgpr_write")) {
      index_masked_predicates();
      if (repeated_masked_predicate_)
        return query(false, repeated_masked_predicate_, {});
    }
    return true;
  }

  // The merged age is a lower bound. Establish the first count that can
  // leave this producer live at this consumer before comparing other uses.
  void refine_masked_expert_count(const PendingEvent &event, const Instruction &inst,
                                  RegisterRef reg, uint64_t consumer, rj_code_arch_t arch,
                                  uint32_t &count, std::optional<WaitRange> field = {}) {
    const bool requires_lane = access_requires_active_lane(inst, reg);
    const bool repeated_valu = cfg_has_repeated_exec_filter_ &&
                               event.counter == WaitCounterKind::VaVdst &&
                               event.kind == WaitEventKind::ValuCore && requires_lane;
    const bool nested_vm = event.counter == WaitCounterKind::VmVsrc &&
                           !cfg_has_unknown_successor_ && source_save_destinations_.count() >= 2;
    if (!repeated_valu && !nested_vm)
      return;
    const auto no_wait = *counter_no_wait_value(arch, event.counter).value();
    for (uint32_t next = count + 1; next < no_wait; ++next) {
      if (field)
        field->count = next;
      const auto pending = has_cfg_path_with_event_pending(event, consumer, arch, false,
                                                           requires_lane, field ? 0 : next, field);
      if (pending != false)
        break;
      count = next;
    }
  }

  bool refine_guarded_load_count(const PendingWaitGroup &group, rj_code_arch_t arch,
                                 CounterParityRequirement &candidate) {
    const auto &event = candidate.event;
    const auto &inst = *candidate.consumer;
    const auto encoded = group.emitted_counts[counter_index(event.counter)];
    if (encoded && group.cfg_block)
      refine_masked_expert_count(
          event, inst, candidate.reg, candidate.consumer_section_offset, arch,
          candidate.required_count,
          WaitRange{group.section_offset, group.last_wait_section_offset, 0, encoded});
    if (!encoded || !group.cfg_block || event.counter != WaitCounterKind::Load ||
        (!has_guarded_matrix_input(inst, arch) && !inst.mnemonic().starts_with("v_accvgpr_write")))
      return true;
    const auto no_wait = *counter_no_wait_value(arch, event.counter).value();
    auto pending = [&](uint32_t count) {
      return guarded_load_witness(
          event, inst, candidate.consumer_section_offset, arch,
          WaitRange{group.section_offset, group.last_wait_section_offset, count, encoded});
    };
    uint32_t safe = candidate.required_count;
    if (safe >= no_wait)
      return false;
    auto first = pending(safe + 1);
    if (first != false) {
      candidate.witness_unknown |= !first.has_value();
      return true;
    }
    ++safe;
    if (safe == no_wait)
      return false;
    candidate.required_count = safe;
    auto last = pending(no_wait);
    if (last == false)
      return false;
    if (!last) {
      candidate.witness_unknown = true;
      return true;
    }
    uint32_t upper = no_wait;
    while (safe + 1 < upper) {
      const uint32_t middle = safe + (upper - safe) / 2;
      const auto reaches = pending(middle);
      if (!reaches) {
        candidate.witness_unknown = true;
        return true;
      }
      if (*reaches)
        upper = middle;
      else
        safe = middle;
    }
    candidate.required_count = safe;
    return true;
  }

  void emit_diagnostic(const Instruction &inst, const PendingEvent &event, RegisterRef reg,
                       WaitcheckAccessKind access, uint32_t required_count, uint64_t section_offset,
                       uint64_t file_offset, rj_code_arch_t arch) {
    if (!should_emit_cfg_diagnostic(event, inst, reg, access, section_offset, arch))
      return;
    if (guarded_load_witness(event, inst, section_offset, arch) == false)
      return;
    // An outstanding dedicated VMEM producer can be older than the merged
    // queue state shows. Use the same loop proof to refine a wait inserted
    // immediately before this matrix operand is read.
    if (required_count == 0 && (inst.flags() & MATRIX_REGISTER_ACCESSES) &&
        arch == ROCJITSU_CODE_ARCH_CDNA3 && event.counter == WaitCounterKind::Load &&
        event.kind == WaitEventKind::VmemNoSamplerLoad && !cfg_has_unknown_successor_ &&
        has_cfg_path_with_event_pending(event, section_offset, arch, /*before_target=*/true,
                                        /*requires_active_lane=*/false, /*minimum_younger=*/1, {},
                                        {}, /*scalar_full_drains_only=*/false,
                                        /*track_younger_predicates=*/true) == false)
      required_count = 1;

    refine_masked_expert_count(event, inst, reg, section_offset, arch, required_count);

    WaitcheckDiagnostic diag;
    diag.kind = WaitcheckDiagnosticKind::WaitCounter;
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
    msg << "missing " << wait_expression(event.counter, required_count, arch).value() << " before ";
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

  void emit_sgpr_hazard_diagnostic(const Instruction &inst, const SgprHazardProducer *producer,
                                   RegisterRef reg, std::string_view depctr_field,
                                   uint64_t section_offset, uint64_t file_offset,
                                   WaitcheckAccessKind access = WaitcheckAccessKind::Use) {
    WaitcheckDiagnostic diag;
    diag.kind = WaitcheckDiagnosticKind::SgprDepctr;
    diag.counter = WaitCounterKind::Depctr;
    diag.access = access;
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
    msg << "missing s_wait_alu " << depctr_field << "(0) before ";
    if (access == WaitcheckAccessKind::ControlTransfer)
      msg << "control transfer with outstanding " << reg_name(reg);
    else
      msg << "use of " << reg_name(reg);
    diag.message = msg.str();
    record_diagnostic(std::move(diag));
  }

  void emit_async_barrier_pipe_diagnostic(const Instruction &inst,
                                          const SgprHazardProducer &barrier,
                                          bool missing_after_barrier, uint64_t section_offset,
                                          uint64_t file_offset) {
    WaitcheckDiagnostic diag;
    diag.kind = missing_after_barrier ? WaitcheckDiagnosticKind::AsyncBarrierPostWait
                                      : WaitcheckDiagnosticKind::AsyncBarrierPreWait;
    diag.counter = WaitCounterKind::VmVsrc;
    diag.access = WaitcheckAccessKind::MemoryOrder;
    diag.reg = RegisterRef{RegClass::PC, 0, 1};
    diag.section_name = barrier.section_name;
    diag.section_offset = section_offset;
    diag.file_offset = file_offset;
    diag.instruction = inst.disassemble();
    diag.producer_section_offset = barrier.section_offset;
    diag.producer_file_offset = barrier.file_offset;
    diag.producer_instruction = barrier.instruction;
    diag.required_count = 0;
    diag.message = std::string("missing s_wait_alu depctr_vm_vsrc(0) immediately ") +
                   (missing_after_barrier ? "after" : "before") +
                   " ds_atomic_async_barrier_arrive_b64";
    record_diagnostic(std::move(diag));
  }

  void emit_va_vdst_hazard_diagnostic(const Instruction &inst, const VaVdstHazard &hazard,
                                      RegisterRef reg, uint32_t encoded_wait,
                                      uint32_t required_wait, uint64_t section_offset,
                                      uint64_t file_offset) {
    WaitcheckDiagnostic diag;
    diag.kind = WaitcheckDiagnosticKind::VaVdst;
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
      if (emitted || ref.cls != RegClass::VGPR || ref.index >= REGISTER_SET_MAX_VGPRS)
        return;
      const auto [hazard, required_wait] = state.requirement(ref.index);
      if (!hazard || *encoded_wait <= required_wait)
        return;

      emit_va_vdst_hazard_diagnostic(inst, *hazard, ref, *encoded_wait, required_wait,
                                     section_offset, file_offset);
      emitted = true;
    });
  }

  // VOP3 encodes the high-half select outside the decoded VGPR operand.
  static uint8_t valu_operand_bytes(const Instruction &inst, const Operand &op, int index) {
    uint8_t mask = op.register_byte_mask();
    if (mask == 0xf && op.size_bits() == 16 && inst.raw_encoding() &&
        inst.size() >= static_cast<int>(sizeof(rdna3::Vop3MachineInst)) &&
        (inst.encoding_id() >> 3) == (rdna3::encoding::kVop3 >> 3)) {
      rdna3::Vop3MachineInst encoding;
      std::memcpy(&encoding, inst.raw_encoding(), sizeof(encoding));
      // A tied accumulator names VDST even when listed as a source.
      const int selector = &op == inst.dst_operand(0) ? 3 : index;
      mask = amdgpu::true16_source_byte_mask(amdgpu::vop3_opsel(encoding), selector);
    }
    return mask;
  }

  static uint8_t trans_operand_bytes(const Instruction &inst, RegisterRef word, bool read) {
    uint8_t bytes = 0;
    for (int i = 0; i < (read ? inst.num_src_operands() : inst.num_dst_operands()); ++i) {
      const auto *op = read ? inst.src_operand(i) : inst.dst_operand(i);
      if (!op || (read && !inst.tracks_explicit_source(i)))
        continue;
      const auto ref = op->to_register_ref();
      if (!ref || ref->cls != RegClass::VGPR || word.index < ref->index ||
          word.index >= ref->index + ref->width)
        continue;
      bytes |= valu_operand_bytes(inst, *op, i);
    }
    if (bytes)
      return bytes;
    // A direct true16 VALU preserves the unwritten destination half for
    // liveness. Only its source operands (including tied accumulators) read
    // VGPR data for the TRANS hazard. DPP can also read an old destination.
    if (read && (inst.flags() & DIRECT_REGISTER_ACCESSES)) {
      const auto *dst = inst.dst_operand(0);
      if (dst && dst->size_bits() == 16 && dst->to_register_ref() == word) {
        amdgpu::RegisterModifiers modifiers;
        inst.amdgpu_register_modifiers(modifiers);
        if (!modifiers.dpp && !modifiers.dpp8 && !modifiers.sdwa)
          return 0;
      }
    }
    return 0xf;
  }

  [[nodiscard]] static std::optional<CounterParityRequirement>
  trans_use_requirement(const TransUseHazards &hazards, const Instruction &consumer,
                        const InstDefUse &du, uint64_t section_offset, uint64_t file_offset) {
    if (hazards.empty() || !is_vector_alu(consumer))
      return std::nullopt;
    for (const auto &[key, hazard] : hazards) {
      const RegisterRef reg{RegClass::VGPR, static_cast<uint16_t>(key.first / 2), 1};
      const uint8_t half = key.first % 2 ? 0xc : 0x3;
      if (!du.uses.contains(reg) || !(trans_operand_bytes(consumer, reg, true) & half))
        continue;
      PendingEvent event;
      event.counter = WaitCounterKind::VaVdst;
      event.section_name = hazard.producer.section_name;
      event.section_offset = hazard.producer.section_offset;
      event.file_offset = hazard.producer.file_offset;
      event.instruction = hazard.producer.instruction;
      return CounterParityRequirement{
          std::move(event), reg,        WaitcheckAccessKind::Use, 0, &consumer,
          section_offset,   file_offset};
    }
    return std::nullopt;
  }

  static void advance_trans_use_hazards(TransUseHazards &hazards, const Instruction &inst) {
    if (hazards.empty())
      return;
    if (is_va_vdst_expiring_instruction(inst.mnemonic())) {
      hazards.clear();
      return;
    }
    if (!is_vector_alu(inst))
      return;
    const bool trans = is_trans_valu(inst);
    for (auto it = hazards.begin(); it != hazards.end();) {
      if (++it->second.valu_age > 5 || (trans && it->first.second == 1))
        it = hazards.erase(it);
      else
        ++it;
    }
    if (trans) {
      TransUseHazards older;
      while (!hazards.empty()) {
        auto node = hazards.extract(hazards.begin());
        node.key().second = 1;
        older.insert(std::move(node));
      }
      hazards.swap(older);
    }
  }

  static void age_va_vdst_hazards(VaVdstHazardState &state, bool trans_seen) {
    constexpr uint8_t kNoHazardWaitStates = 15;
    for (auto it = state.hazards.begin(); it != state.hazards.end();) {
      if (++it->second.age >= kNoHazardWaitStates)
        it = state.hazards.erase(it);
      else
        ++it;
    }
    if (!trans_seen)
      return;
    // All surviving paths have now issued a TRANS, with their actual ages.
    for (auto it = state.hazards.begin(); it != state.hazards.end();) {
      if (it->first.second) {
        ++it;
        continue;
      }
      auto node = state.hazards.extract(it++);
      state.merge_hazard({node.key().first, true}, node.mapped());
    }
  }

  static void clear_va_vdst_hazards(VaVdstHazardState &state) { state.hazards.clear(); }

  static void set_va_vdst_hazard_for_regs(VaVdstHazardState &state, const RegisterSet &regs,
                                          bool trans_seen, const SgprHazardProducer &producer) {
    regs.for_each([&](RegisterRef ref) {
      if (ref.cls != RegClass::VGPR || ref.index >= REGISTER_SET_MAX_VGPRS)
        return;
      state.hazards.erase({ref.index, !trans_seen});
      state.hazards.insert_or_assign({ref.index, trans_seen}, VaVdstHazard{0, producer});
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

  using Reachability = std::vector<uint8_t>;
  using SharedReachability = std::shared_ptr<const Reachability>;

  template <typename Cache>
  [[nodiscard]] SharedReachability
  cache_reachability(Cache &cache, const std::vector<size_t> &roots, Reachability reachable) {
    const size_t bytes =
        reachable.size() * sizeof(Reachability::value_type) + roots.size() * sizeof(size_t);
    auto result = std::make_shared<const Reachability>(std::move(reachable));
    if (bytes <= options_.max_reachability_cache_bytes -
                     std::min(reachability_cache_bytes_, options_.max_reachability_cache_bytes)) {
      reachability_cache_bytes_ += bytes;
      cache.emplace(roots, result);
    }
    return result;
  }

  // A physical instruction may occur in many expanded call contexts. A
  // multi-source search computes the union of their reachability in one walk.
  [[nodiscard]] SharedReachability reachable_blocks(const std::vector<size_t> &roots,
                                                    bool forward) {
    auto &cache = forward ? forward_reachability_cache_ : reverse_reachability_cache_;
    if (auto cached = cache.find(roots); cached != cache.end())
      return cached->second;
    Reachability reachable(cfg_views_.size());
    std::vector<size_t> worklist(roots);
    for (size_t root : roots)
      reachable[root] = 1;
    while (!worklist.empty()) {
      const size_t block = worklist.back();
      worklist.pop_back();
      const auto &edges = forward ? cfg_views_[block].successors : cfg_views_[block].predecessors;
      for (size_t next : edges) {
        if (reachable[next] == 0) {
          reachable[next] = 1;
          worklist.push_back(next);
        }
      }
    }
    return cache_reachability(cache, roots, std::move(reachable));
  }

  [[nodiscard]] std::vector<size_t> cfg_block_indices_containing(uint64_t section_offset) const {
    std::vector<size_t> result;
    for (size_t i = 0; i < cfg_views_.size(); ++i) {
      const BasicBlock *block = cfg_views_[i].block;
      if (block != nullptr && section_offset >= block->start_offset() &&
          section_offset < block->end_offset())
        result.push_back(i);
    }
    return result;
  }

  // Keep the eight delay slots out of ordinary CFG search states. A small
  // ID represents one immutable history on a path after this producer.
  struct PathDelayHistories {
    std::vector<ExpertValuDelayState> values{ExpertValuDelayState{}};
    std::unordered_map<std::string, uint16_t> ids;

    std::optional<uint16_t> intern(const ExpertValuDelayState &history, uint16_t previous) {
      if (history == values[previous])
        return previous;
      if (history == values.front())
        return 0;
      std::string key;
      key.reserve(18);
      key.push_back(static_cast<char>(history.pending));
      key.push_back(static_cast<char>(history.pending >> 8));
      for (const auto &group : history.history)
        for (const auto &slot : group) {
          key.push_back(static_cast<char>(slot.kind));
          key.push_back(static_cast<char>(slot.units));
        }
      if (auto found = ids.find(key); found != ids.end())
        return found->second;
      if (values.size() >= 4096)
        return std::nullopt;
      const auto next = static_cast<uint16_t>(values.size());
      ids.emplace(std::move(key), next);
      values.push_back(history);
      return next;
    }
  };

  // These counts belong to the replay prefix, independent of the dataflow
  // event supplied at the consumer. In particular, a later core VALU cannot
  // help a VA_VDST wait that precedes it.
  struct PathProgress {
    uint32_t age = 0;
    uint8_t store_age = 0;
    uint8_t core = 0;
    uint16_t delay_id = 0;
    bool expert_enabled = false;
    bool source_implies_nonempty_exec = false;
    bool crossed_exec_expansion = false;
    uint8_t younger_source_primaries = 0;
    std::bitset<128> saved_source_exec;
    bool unknown_delay_issue = false;
    bool operator==(const PathProgress &) const = default;
  };

  void apply_path_pre_dependency_waits(const Instruction &inst, const PendingEvent &event,
                                       rj_code_arch_t arch, bool &pending, PathProgress &progress,
                                       bool expert, const WaitFields *replacement = nullptr) {
    if (!pending)
      return;
    // Most path instructions cannot retire an event. Avoid constructing and
    // copying a full pending state merely to apply an identity transition.
    const auto wait_vm_vsrc = dsdir_wait_vm_vsrc(inst);
    if (!replacement && !inst.is_waitcnt() && !releases_gfx10_scalar_sources(inst, arch) &&
        !vinterp_wait_exp(inst) && (!wait_vm_vsrc || *wait_vm_vsrc != 0))
      return;

    PendingEvent path_event = event;
    const auto ordering = path_wait_facts_.find(&inst);
    const bool ordered =
        ordering != path_wait_facts_.end() &&
        !(ordering->second.unordered & (uint64_t{1} << counter_index(event.counter)));
    path_event.min_younger =
        is_completion_counter(event.counter) || expert || ordered ? progress.age : 0;
    path_event.min_younger_core = progress.core;
    path_event.younger_source_primaries = progress.younger_source_primaries;
    PendingState state;
    path_event.min_younger_primary = progress.store_age;
    state.pending[counter_index(event.counter)].push_back(path_event);
    if ((event.counter == WaitCounterKind::VmVsrc || event.counter == WaitCounterKind::X) &&
        ordering != path_wait_facts_.end() &&
        (event.kind == WaitEventKind::FlatLoad || event.kind == WaitEventKind::FlatStore)) {
      const uint64_t routes = event.kind == WaitEventKind::FlatLoad
                                  ? ordering->second.flat_load_routes
                                  : ordering->second.flat_store_routes;
      for (const auto counter :
           {WaitCounterKind::Load, WaitCounterKind::Store, WaitCounterKind::Ds}) {
        const auto index = counter_index(counter);
        if (routes & (uint64_t{1} << index))
          state.pending_event_ages[index].values[static_cast<size_t>(event.kind)] = 0;
      }
    }

    if (replacement) {
      (void)apply_wait_fields(state, *replacement, arch);
    } else if (inst.is_waitcnt()) {
      (void)apply_waitcnt(state, inst, arch);
    } else {
      (void)apply_embedded_waitcnt(state, inst, arch);
    }

    auto &events = state.pending[counter_index(event.counter)];
    auto found = find_event(events, path_event);
    if (found == events.end()) {
      pending = false;
      progress.age = 0;
      progress.store_age = 0;
      progress.core = 0;
      return;
    }
    progress.age = found->min_younger;
    progress.younger_source_primaries = found->younger_source_primaries;
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

  // A saved trip count and its decreasing copy can select opposite matrix
  // epilogues. Track their low bits only if a kernel both copies the count
  // and tests the saved low bit. One pair bounds the search to four states.
  std::optional<LoopParity> loop_parity() {
    if (indexed_loop_parity_)
      return loop_parity_;
    indexed_loop_parity_ = true;
    std::set<LoopParity> copies;
    std::set<uint16_t> tested, decremented;
    for (const auto &block : cfg_views_) {
      for (const auto &view : block.instructions) {
        const auto &inst = *view.instruction;
        const auto name = inst.mnemonic();
        if (name != "s_mov_b32" && name != "s_and_b32" && name != "s_sub_u32")
          continue;
        const auto dst = inst.dst_operand(0)->to_register_ref();
        const auto src = inst.src_operand(0)->to_register_ref();
        if (!dst || !src || dst->cls != RegClass::SGPR || src->cls != RegClass::SGPR ||
            dst->width != 1 || src->width != 1)
          continue;
        if (name == "s_mov_b32") {
          if (dst != src)
            copies.insert({dst->index, src->index});
        } else if (inst.src_operand(1)->const_value() == 1) {
          if (name == "s_and_b32")
            tested.insert(src->index);
          else if (dst == src)
            decremented.insert(src->index);
        }
      }
    }
    for (auto pair : copies) {
      if (tested.contains(pair.saved) && decremented.contains(pair.live)) {
        loop_parity_ = pair;
        break;
      }
    }
    return loop_parity_;
  }

  static ParityStep step_parity(const Instruction &inst, LoopParity pair, uint8_t pairs,
                                std::span<const uint16_t> defs,
                                const ScalarConstraints &constraints) {
    unsigned written = 0;
    for (auto reg : defs)
      written |= reg == pair.saved ? 2 : reg == pair.live ? 1 : 0;
    auto source = [](const Operand *operand) {
      return operand ? operand->to_register_ref() : std::optional<RegisterRef>{};
    };
    const auto lhs = source(inst.src_operand(0));
    const auto dst = source(inst.dst_operand(0));
    const auto name = inst.mnemonic();
    const auto *rhs = inst.src_operand(1);
    const auto constant = rhs ? rhs->const_value() : std::nullopt;
    const bool live =
        lhs && lhs->cls == RegClass::SGPR && lhs->width == 1 && lhs->index == pair.live;
    const bool saved =
        lhs && lhs->cls == RegClass::SGPR && lhs->width == 1 && lhs->index == pair.saved;
    const bool writes_saved =
        dst && dst->cls == RegClass::SGPR && dst->width == 1 && dst->index == pair.saved;
    const bool writes_live =
        dst && dst->cls == RegClass::SGPR && dst->width == 1 && dst->index == pair.live;
    const bool copy = name == "s_mov_b32" && live && writes_saved;
    const bool subtract = name == "s_sub_u32" && live && writes_live && constant;
    const bool equals = (name == "s_cmp_eq_i32" || name == "s_cmp_eq_u32") && live && constant;
    // The source bit is tested before the AND writes its destination.
    const bool test_saved = name == "s_and_b32" && saved && constant == 1;
    unsigned bounded_parities = 3;
    if (live && name == "s_cmp_le_u32" && constant && *constant <= 8) {
      // A pipeline can test 0 and 1 before the small unroll limit. Retain
      // only low bits actually admitted by those earlier feasible branches.
      bounded_parities = 0;
      const auto known = constraints.find(pair.live);
      for (uint32_t x = 0; x <= *constant; ++x) {
        if (known == constraints.end() ||
            ((!known->second.equal || *known->second.equal == x) &&
             !known->second.not_equal.contains(x) &&
             (!known->second.unsigned_min || *known->second.unsigned_min <= x) &&
             (!known->second.unsigned_max || *known->second.unsigned_max >= x)))
          bounded_parities |= 1u << (x & 1);
      }
    }
    ParityStep step{};
    for (unsigned before = 0; before < 4; ++before) {
      if (!(pairs & (1u << before)))
        continue;
      for (unsigned after = 0; after < 4; ++after) {
        if (((before ^ after) & ~written) || (copy && (after >> 1) != (before & 1)) ||
            (subtract && (after & 1) != ((before ^ *constant) & 1)))
          continue;
        const uint8_t mask = 1u << after;
        step.pairs |= mask;
        if ((bounded_parities & (1u << (before & 1))) &&
            (!equals || (before & 1) == (*constant & 1)) && (!test_saved || (before & 2)))
          step.on_true |= mask;
        // A mismatched full value can have either parity.
        if (!test_saved || !(before & 2))
          step.on_false |= mask;
      }
    }
    return step;
  }

  // Only an ordinary load-to-ACC diagnostic needs these comparisons. Intern
  // equal expressions in a stable order and inspect each instruction once.
  void index_masked_predicates() {
    if (indexed_masked_predicates_)
      return;
    indexed_masked_predicates_ = true;
    std::set<MaskedSccPredicate> seen;
    for (const auto &block : cfg_views_) {
      for (size_t i = 0; i < block.instructions.size(); ++i) {
        const auto *inst = block.instructions[i].instruction;
        if (masked_predicates_.contains(inst))
          continue;
        const auto guard = masked_scc_predicate(*inst, std::span(block.instructions).first(i));
        if (guard) {
          masked_predicates_.emplace(inst, *guard);
          if (!seen.insert(*guard).second &&
              (!repeated_masked_predicate_ || *guard < *repeated_masked_predicate_))
            repeated_masked_predicate_ = guard;
        }
      }
    }
  }

  // Recover (sN & mask) == value from a short, straight-line chain. The
  // optional subtraction is modulo 2^32, as is equality for either spelling
  // of S_CMP_EQ. Never take a definition from another block or across a call.
  std::optional<MaskedSccPredicate>
  masked_scc_predicate(const Instruction &compare, std::span<const CfgInstructionView> prior) {
    if (compare.mnemonic() != "s_cmp_eq_i32" && compare.mnemonic() != "s_cmp_eq_u32")
      return std::nullopt;
    auto compared = compare.src_operand(0)->to_register_ref();
    const auto imm = compare.src_operand(1)->const_value();
    if (!compared || compared->cls != RegClass::SGPR || compared->width != 1 || !imm)
      return std::nullopt;
    uint32_t value = static_cast<uint32_t>(*imm);
    bool subtracted = false;
    const size_t first = prior.size() - std::min<size_t>(prior.size(), 8);
    std::set<uint16_t> overwritten;
    for (size_t i = prior.size(); i-- > first;) {
      const auto &inst = *prior[i].instruction;
      const auto name = inst.mnemonic();
      if (inst.is_branch() || name.starts_with("s_movrel") || name.starts_with("s_call"))
        return std::nullopt;
      const auto &defs = path_scalar_defs_.at(&inst);
      if (std::ranges::find(defs, compared->index) == defs.end()) {
        overwritten.insert(defs.begin(), defs.end());
        continue;
      }
      if (!inst.dst_operand(0) || inst.dst_operand(0)->to_register_ref() != compared)
        return std::nullopt;
      if (name != "s_and_b32" && name != "s_sub_i32" && name != "s_sub_u32")
        return std::nullopt;
      const auto src = inst.src_operand(0)->to_register_ref();
      const auto literal = inst.src_operand(1)->const_value();
      if (!src || src->cls != RegClass::SGPR || src->width != 1 || !literal)
        return std::nullopt;
      if (name == "s_and_b32") {
        // The same physical root must still hold the sampled value at compare.
        if (src == compared || overwritten.contains(src->index))
          return std::nullopt;
        return MaskedSccPredicate{src->index, static_cast<uint32_t>(*literal), value};
      }
      if (subtracted)
        return std::nullopt;
      overwritten.insert(compared->index);
      compared = src;
      value += static_cast<uint32_t>(*literal);
      subtracted = true;
    }
    return std::nullopt;
  }

  [[nodiscard]] static std::optional<SccPredicate> scalar_scc_predicate(const Instruction &inst,
                                                                        rj_code_arch_t arch) {
    SccPredicate::Kind kind;
    if (inst.mnemonic() == "s_cmp_eq_u32")
      kind = SccPredicate::Kind::EqImm;
    else if (inst.mnemonic() == "s_cmp_ge_u32")
      kind = SccPredicate::Kind::GeUnsignedImm;
    else if (arch == ROCJITSU_CODE_ARCH_CDNA3 && inst.mnemonic() == "s_cmp_le_u32")
      kind = SccPredicate::Kind::LeUnsignedImm;
    else
      return std::nullopt;
    const Operand *lhs = inst.src_operand(0);
    const Operand *rhs = inst.src_operand(1);
    if (lhs == nullptr || rhs == nullptr)
      return std::nullopt;

    auto lhs_ref = lhs->to_register_ref();
    if ((arch == ROCJITSU_CODE_ARCH_RDNA3 || arch == ROCJITSU_CODE_ARCH_RDNA3_5) && lhs_ref &&
        lhs_ref->cls == RegClass::SGPR && lhs_ref->width == 1 && lhs_ref == rhs->to_register_ref())
      return SccPredicate{SccPredicate::Kind::Constant, 0, 1};
    auto rhs_imm = parse_operand_immediate(rhs);
    if (!lhs_ref || lhs_ref->cls != RegClass::SGPR || lhs_ref->width != 1 || !rhs_imm ||
        *rhs_imm < std::numeric_limits<int32_t>::min() ||
        *rhs_imm > std::numeric_limits<uint32_t>::max())
      return std::nullopt;

    return SccPredicate{kind, lhs_ref->index, static_cast<uint32_t>(*rhs_imm)};
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
    if (predicate->kind == SccPredicate::Kind::Constant)
      return branch_condition_true == (predicate->value != 0);

    ScalarValueConstraint &constraint = constraints[predicate->sgpr];
    auto is_satisfiable = [&]() {
      if (constraint.unsigned_min && constraint.unsigned_max &&
          *constraint.unsigned_min > *constraint.unsigned_max)
        return false;
      if (constraint.equal) {
        if (constraint.not_equal.contains(*constraint.equal))
          return false;
        if (constraint.unsigned_min && *constraint.equal < *constraint.unsigned_min)
          return false;
        if (constraint.unsigned_max && *constraint.equal > *constraint.unsigned_max)
          return false;
      }
      if (constraint.unsigned_min && constraint.unsigned_max &&
          *constraint.unsigned_min == *constraint.unsigned_max &&
          constraint.not_equal.contains(*constraint.unsigned_min))
        return false;
      return true;
    };

    switch (predicate->kind) {
    case SccPredicate::Kind::Constant:
      break; // Handled before looking up an SGPR.
    case SccPredicate::Kind::EqImm:
      if (branch_condition_true) {
        if (constraint.equal && *constraint.equal != predicate->value)
          return false;
        if (constraint.not_equal.contains(predicate->value))
          return false;
        constraint.equal = predicate->value;
        return is_satisfiable();
      }
      if (constraint.equal && *constraint.equal == predicate->value)
        return false;
      constraint.not_equal.insert(predicate->value);
      return is_satisfiable();
    case SccPredicate::Kind::LeUnsignedImm:
    case SccPredicate::Kind::GeUnsignedImm: {
      uint32_t bound = predicate->value;
      if (predicate->kind == SccPredicate::Kind::LeUnsignedImm) {
        if (bound == UINT32_MAX)
          return branch_condition_true;
        ++bound; // x <= n is the complement of x >= n+1.
        branch_condition_true = !branch_condition_true;
      }
      if (branch_condition_true) {
        if (!constraint.unsigned_min || *constraint.unsigned_min < bound)
          constraint.unsigned_min = bound;
      } else {
        if (bound == 0)
          return false;
        const uint32_t upper_bound = bound - 1;
        if (!constraint.unsigned_max || *constraint.unsigned_max > upper_bound)
          constraint.unsigned_max = upper_bound;
      }
      return is_satisfiable();
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
      for (uint32_t value : constraint.not_equal)
        os << value << ',';
      if (constraint.unsigned_min)
        os << ">=" << *constraint.unsigned_min;
      if (constraint.unsigned_max)
        os << "<=" << *constraint.unsigned_max;
      os << ';';
    }
    if (predicate)
      os << "scc:" << static_cast<unsigned>(predicate->kind) << ":s" << predicate->sgpr << ':'
         << predicate->value;
    return os.str();
  }

  [[nodiscard]] static std::optional<RegClass> mask_branch_class(std::string_view mnemonic) {
    if (mnemonic == "s_cbranch_execz" || mnemonic == "s_cbranch_execnz")
      return RegClass::EXEC;
    if (mnemonic == "s_cbranch_vccz" || mnemonic == "s_cbranch_vccnz")
      return RegClass::VCC;
    return std::nullopt;
  }

  struct Wave32Lanes {
    // Any participating lane must be in this set. An unknown literal mask
    // therefore starts with all lanes possible.
    uint32_t possible = UINT32_MAX;
    std::optional<uint16_t> source{};
    bool inverted = false;
    bool exact = false;

    bool operator==(const Wave32Lanes &) const = default;

    bool overlaps(const Wave32Lanes &other) const {
      return (possible & other.possible) != 0 &&
             !(source && source == other.source && inverted != other.inverted);
    }

    void include(const Wave32Lanes &other) {
      exact &= other.exact && possible == other.possible;
      possible |= other.possible;
      // Repeated issues at the same PC may write different lanes. Preserve a
      // symbolic mask only when it describes both outstanding issues.
      if (source != other.source || inverted != other.inverted) {
        source.reset();
        inverted = false;
      }
    }

    void append_key(std::string &key, char tag) const {
      key.push_back(tag);
      key += std::to_string(possible);
      if (source) {
        key.push_back(inverted ? '~' : '=');
        key += std::to_string(*source);
      }
    }
  };

  struct PathMasks {
    // Nonempty is weaker than full. A partially populated workgroup can start
    // with only some lanes active.
    std::optional<bool> exec;
    std::optional<bool> vcc;
    std::optional<bool> scc;
    bool vcc_is_exec = false;
    bool scc_is_exec = false;
    Wave32Lanes exec_lanes{};
    Wave32Lanes issue_lanes{};
    uint64_t scalar_pending_counters = 0;
    std::set<uint16_t> scalar_pending_sgprs;
    bool scalar_pending_exec = false;
    bool scalar_pending_vcc = false;
    bool scalar_pending_scc = false;
    bool scalar_pending_unknown = false;
    uint64_t pending_scc_write_counters = 0;
    bool operator==(const PathMasks &) const = default;
  };

  [[nodiscard]] static bool path_operand_has_pending_scalar_result(const Operand *op,
                                                                   const PathMasks &masks) {
    if (masks.scalar_pending_unknown)
      return true;
    const auto ref = op ? op->to_register_ref() : std::nullopt;
    if (!ref || ref->cls != RegClass::SGPR)
      return false;
    for (uint16_t reg = ref->index; reg < ref->index + ref->width; ++reg) {
      if (masks.scalar_pending_sgprs.contains(reg))
        return true;
    }
    return false;
  }

  // Reuse equalities from the SCC path filter. Only literals and SGPRs feeding
  // a mask or the queried lane selector are materialized, so unrelated scalar
  // computations do not multiply the search states.
  [[nodiscard]] static std::optional<uint32_t> path_dword(const Operand *op,
                                                          const ScalarConstraints &facts) {
    if (op == nullptr)
      return std::nullopt;
    if (const auto value = op->const_value())
      return static_cast<uint32_t>(*value);
    if (const auto ref = op->to_register_ref();
        ref && ref->cls == RegClass::SGPR && ref->width == 1) {
      if (auto found = facts.find(ref->index); found != facts.end())
        return found->second.equal;
    }
    return std::nullopt;
  }

  [[nodiscard]] static bool low_selector(const Operand *op, uint32_t selector) {
    return op && op->has_register_selector() && op->encoding_value() == static_cast<int>(selector);
  }

  static void update_path_masks(PathMasks &masks, const CfgInstructionView &view,
                                const ScalarConstraints &facts, uint32_t wave_size,
                                bool track_exec_source, bool writes_only_sgpr) {
    const Instruction &inst = *view.instruction;
    // A summarized call may change every mask and scalar that describes it.
    if ((inst.flags() & INDIRECT_CALL ||
         (track_exec_source && inst.mnemonic().starts_with("s_call"))) &&
        !(track_exec_source && view.enters_resolved_callee)) {
      masks = {};
      masks.scalar_pending_counters = UINT64_MAX;
      masks.scalar_pending_unknown = true;
      return;
    }
    masks.scalar_pending_counters &= ~view.zero_wait_counters;
    if (masks.scalar_pending_counters == 0) {
      masks.scalar_pending_sgprs.clear();
      masks.scalar_pending_exec = false;
      masks.scalar_pending_vcc = false;
      masks.scalar_pending_scc = false;
      masks.scalar_pending_unknown = false;
    }
    masks.scalar_pending_counters |= view.issued_scalar_result_counters;
    for (uint16_t reg : view.scalar_result_dest_sgprs)
      masks.scalar_pending_sgprs.insert(reg);
    masks.scalar_pending_exec |= view.scalar_result_dest_exec;
    masks.scalar_pending_vcc |= view.scalar_result_dest_vcc;
    masks.scalar_pending_scc |= view.scalar_result_dest_scc;
    masks.scalar_pending_unknown |= view.scalar_result_dest_unknown;
    if (masks.scalar_pending_unknown) {
      masks.exec.reset();
      masks.vcc.reset();
      masks.scc.reset();
      masks.vcc_is_exec = false;
      masks.scc_is_exec = false;
      masks.exec_lanes = {};
    }
    if (masks.scalar_pending_scc) {
      masks.scc.reset();
      masks.scc_is_exec = false;
    }
    if (view.scalar_result_dest_exec) {
      masks.exec.reset();
      masks.exec_lanes = {};
      if (masks.vcc_is_exec)
        masks.vcc.reset();
      if (masks.scc_is_exec)
        masks.scc.reset();
      masks.vcc_is_exec = false;
      masks.scc_is_exec = false;
    }
    if (view.scalar_result_dest_vcc) {
      masks.vcc.reset();
      masks.vcc_is_exec = false;
    }
    const Operand *dst = inst.dst_operand(0);
    if (view.invalidates_exec && !writes_only_sgpr) {
      if (masks.vcc_is_exec) {
        masks.vcc = masks.exec;
        masks.vcc_is_exec = false;
      }
      if (masks.scc_is_exec) {
        masks.scc = masks.exec;
        masks.scc_is_exec = false;
      }
    }
    std::optional<bool> and_exec_result;
    bool and_exec_relation = false;
    if (wave_size == 32 && !masks.scalar_pending_exec && !masks.scalar_pending_unknown &&
        inst.mnemonic() == "s_and_b32") {
      for (unsigned i = 0; i < 2; ++i) {
        if (!low_selector(inst.src_operand(i), amdgpu::kExecSelectorFirst) ||
            path_operand_has_pending_scalar_result(inst.src_operand(1 - i), masks))
          continue;
        const auto mask = path_dword(inst.src_operand(1 - i), facts);
        if (masks.exec == false || mask == 0)
          and_exec_result = false;
        else if (mask == UINT32_MAX) {
          and_exec_result = masks.exec;
          and_exec_relation = !masks.exec.has_value();
        }
      }
    }
    if (view.invalidates_vcc_condition && !writes_only_sgpr) {
      const bool tracks_result = low_selector(dst, amdgpu::kVccSelectorFirst);
      masks.vcc = tracks_result && !masks.scalar_pending_vcc ? and_exec_result : std::nullopt;
      masks.vcc_is_exec = tracks_result && !masks.scalar_pending_vcc && and_exec_relation;
    }
    if (view.invalidates_scc && !is_scc_conditional_branch(inst.mnemonic())) {
      masks.scc = masks.scalar_pending_scc ? std::nullopt : and_exec_result;
      masks.scc_is_exec = !masks.scalar_pending_scc && and_exec_relation;
    }
    if (view.invalidates_exec && !writes_only_sgpr) {
      masks.exec.reset();
      masks.exec_lanes = {};
      if (!masks.scalar_pending_exec && !masks.scalar_pending_unknown &&
          low_selector(dst, amdgpu::kExecSelectorFirst)) {
        if (inst.mnemonic() == "s_mov_b64") {
          if (const auto value = inst.src_operand(0)->const_value()) {
            masks.exec = (wave_size == 32 ? static_cast<uint32_t>(*value) : *value) != 0;
            if (track_exec_source) {
              masks.exec_lanes.possible = static_cast<uint32_t>(*value);
              masks.exec_lanes.exact = true;
            }
          }
        } else if (wave_size == 32 && (inst.mnemonic() == "s_mov_b32" ||
                                       (track_exec_source && inst.mnemonic() == "s_not_b32"))) {
          const auto op = inst.src_operand(0);
          const bool inverted = inst.mnemonic() == "s_not_b32";
          const auto source = op->to_register_ref();
          const bool pending_source = path_operand_has_pending_scalar_result(op, masks);
          if (track_exec_source && !pending_source && source && source->cls == RegClass::SGPR &&
              source->width == 1) {
            masks.exec_lanes.source = source->index;
            masks.exec_lanes.inverted = inverted;
          }
          if (!pending_source) {
            if (const auto value = path_dword(op, facts)) {
              const auto lanes = inverted ? ~*value : *value;
              masks.exec = lanes != 0;
              if (track_exec_source) {
                masks.exec_lanes.possible = lanes;
                masks.exec_lanes.exact = true;
              }
            }
          }
          if (!masks.exec && masks.exec_lanes.source && !pending_source) {
            const auto known = facts.find(*masks.exec_lanes.source);
            if (known != facts.end() && known->second.not_equal.contains(inverted ? UINT32_MAX : 0))
              masks.exec = true;
          }
        }
      }
    }
  }

  struct LaneOperand {
    uint16_t index;
    WaitcheckAccessKind access;
    std::array<std::optional<uint16_t>, 2> selector_sgprs{};
    auto operator<=>(const LaneOperand &) const = default;
  };

  static const Operand *single_lane_selector(const Instruction &inst) {
    const int last = inst.num_src_operands() - 1;
    const auto *op = inst.src_operand(last);
    return op == inst.dst_operand(0) ? inst.src_operand(last - 1) : op;
  }

  static bool has_cross_lane_access(const Instruction &inst, const amdgpu::RegisterModifiers &m) {
    const auto name = inst.mnemonic();
    return m.dpp || m.dpp8 || m.exec_whole_quads ||
           m.valu_permutation != amdgpu::ValuPermutation::None ||
           m.ds_permutation != amdgpu::DsPermutation::None || name == "lds_param_load" ||
           name.starts_with("v_interp_") || name.starts_with("v_readlane") ||
           name.starts_with("v_readfirstlane") || name.starts_with("v_writelane");
  }

  bool has_gfx11_exec_masked_result(const PendingEvent &event) const {
    return (event.counter == WaitCounterKind::Ds || event.counter == WaitCounterKind::Load) &&
           exec_masked_results_.contains(event.section_offset);
  }

  std::optional<LaneOperand> guarded_lane_operand(const PendingEvent &event,
                                                  const Instruction &inst, RegisterRef reg,
                                                  WaitcheckAccessKind access,
                                                  rj_code_arch_t arch) const {
    if (!current_kernel_ || !cfg_has_exec_guards_ || wavefront_size_ != 32 ||
        (arch != ROCJITSU_CODE_ARCH_RDNA3 && arch != ROCJITSU_CODE_ARCH_RDNA3_5) ||
        !has_gfx11_exec_masked_result(event) || reg.cls != RegClass::VGPR)
      return {};
    amdgpu::RegisterModifiers m;
    inst.amdgpu_register_modifiers(m);
    if (!has_cross_lane_access(inst, m))
      return {};
    LaneOperand result{reg.index, access};
    auto selector_sgpr = [](const Operand *op) -> std::optional<uint16_t> {
      const auto ref = op ? op->to_register_ref() : std::nullopt;
      if (ref && ref->cls == RegClass::SGPR && ref->width == 1)
        return ref->index;
      return {};
    };
    const auto name = inst.mnemonic();
    if (name.starts_with("v_readlane") || name.starts_with("v_writelane")) {
      result.selector_sgprs[0] = selector_sgpr(single_lane_selector(inst));
    } else if (m.valu_permutation != amdgpu::ValuPermutation::None &&
               m.valu_permutation != amdgpu::ValuPermutation::Perm64) {
      result.selector_sgprs = {selector_sgpr(inst.src_operand(1)),
                               selector_sgpr(inst.src_operand(2))};
    }
    return result;
  }

  // Resolve the physical VGPR lanes for this particular operand. An unknown
  // selector can justify an ordinary warning, but cannot establish an exact
  // zero for a compiler ABI wait. Other matching operands can still prove it.
  static std::optional<bool> physical_lane_witness(const Instruction &inst, LaneOperand operand,
                                                   const PathMasks &masks,
                                                   const ScalarConstraints &facts,
                                                   rj_code_arch_t arch) {
    using namespace amdgpu;
    const bool write = operand.access == WaitcheckAccessKind::Def;
    const auto name = inst.mnemonic();
    const uint32_t active = masks.exec == false ? 0 : masks.exec_lanes.possible;
    const uint32_t produced = masks.issue_lanes.possible;
    const bool complementary = masks.exec_lanes.source &&
                               masks.exec_lanes.source == masks.issue_lanes.source &&
                               masks.exec_lanes.inverted != masks.issue_lanes.inverted;
    const auto lane_bit = [](unsigned lane) { return uint32_t{1} << lane; };
    auto matches = [&](const Operand *op) {
      const auto reg = op ? register_operand(*op, arch) : std::nullopt;
      return reg && reg->cls == RegClass::VGPR && operand.index >= reg->index &&
             operand.index - reg->index < reg->width;
    };
    if (name.starts_with("v_readlane") || name.starts_with("v_writelane")) {
      const auto lane = path_dword(single_lane_selector(inst), facts);
      return lane ? std::optional<bool>{bool(produced & lane_bit(*lane & 31))} : std::nullopt;
    }
    if (name.starts_with("v_readfirstlane")) {
      if (!active || masks.exec_lanes.exact)
        return bool(produced & lane_bit(active ? std::countr_zero(active) : 0));
      if (!masks.exec && (produced & 1u))
        return std::nullopt;
      if (complementary || !(produced & active))
        return false;
      return std::nullopt;
    }
    if (!active)
      return false;

    RegisterModifiers m;
    inst.amdgpu_register_modifiers(m);
    const bool dpp = m.dpp || m.dpp8;
    amdgpu::dpp::DppPlan plan;
    if (dpp)
      plan = m.dpp
                 ? amdgpu::dpp::make_dpp_plan(32, m.control, m.row_mask, m.bank_mask, m.bound_ctrl,
                                              m.fi, active, m.inactive_uses_bound_ctrl)
                 : amdgpu::dpp::make_dpp8_plan(32, m.control, m.fi, active);
    bool unknown = false;
    for (unsigned dest = 0; dest < 32; ++dest) {
      const uint32_t lane = lane_bit(dest);
      if (!(active & lane))
        continue;
      // A complementary mask excludes a same-lane access. It can still
      // select a different, inactive source when fetch-inactive is enabled.
      auto hit = [&](uint32_t sources, bool require_active, bool proven = true) {
        if (require_active)
          sources = complementary ? 0 : sources & active;
        else if (complementary)
          sources &= ~lane;
        if (!(produced & sources))
          return false;
        if (!proven)
          unknown = true;
        return proven;
      };
      if (m.exec_whole_quads || name == "lds_param_load") {
        // Both opcodes write complete active quads. LDS_DIRECT_LOAD also
        // requires a valid data type and alignment in M0.
        if (hit(0xfu << (dest & ~3u), false, name == "lds_param_load"))
          return true;
        continue;
      }
      const auto kind = m.valu_permutation;
      if (kind != ValuPermutation::None) {
        if (kind == ValuPermutation::Perm64)
          continue; // XOR 32 is outside wave32.
        const unsigned row = (dest & ~15u) ^ (kind == ValuPermutation::X16 ? 16u : 0u);
        const auto *control = inst.src_operand((dest & 15) < 8 ? 1 : 2);
        if (!write && matches(control) && hit(1u, false))
          return true;
        const auto selector = path_dword(control, facts);
        const uint32_t sources =
            selector ? lane_bit(valu_permutation_source(
                           kind, dest, (*selector >> ((dest & 7) * 4)) & 15, 16, 32))
                     : 0xffffu << row;
        if (write) {
          if ((m.fi || m.bound_ctrl || (sources & active)) &&
              hit(lane, true, m.fi || m.bound_ctrl || selector.has_value()))
            return true;
        } else if (matches(inst.src_operand(0)) && hit(sources, !m.fi, selector.has_value()))
          return true;
        continue;
      }
      if (write) {
        if ((!dpp || (plan.row_bank_mask & plan.source_write_mask & lane)) && hit(lane, true))
          return true;
        continue;
      }
      const auto ds = m.ds_permutation;
      if (ds != DsPermutation::None) {
        if (matches(m.src1) && hit(lane, true))
          return true;
        if (matches(m.src0)) {
          if (ds == DsPermutation::Swizzle &&
              hit(lane_bit(ds_swizzle_lane(dest, m.ds_offset)), true))
            return true;
          if (ds == DsPermutation::Permute && hit(lane, true))
            return true;
          if (ds == DsPermutation::Bpermute)
            hit(active, true, false);
        }
        continue;
      }
      for (int i = 0; i < inst.num_src_operands(); ++i) {
        const auto *op = inst.src_operand(i);
        if (!inst.tracks_explicit_source(i) || !matches(op))
          continue;
        if (dpp && op == m.src0) {
          if ((plan.physical_read_dest_mask & lane) &&
              hit(lane_bit(plan.source_lanes[dest]), !m.fi))
            return true;
        } else if (dpp && op == m.dpp_masked_compare_rhs) {
          if ((plan.row_bank_mask & plan.source_write_mask & lane) && hit(lane, true))
            return true;
        } else if (name.starts_with("v_interp_") &&
                   (i == 0 || (i == 2 && !name.starts_with("v_interp_p2")))) {
          const unsigned part = i == 2 ? 0 : name.starts_with("v_interp_p2") ? 2 : 1;
          if (hit(lane_bit((dest & ~3u) + part), false))
            return true;
        } else if (hit(lane, true))
          return true;
      }
    }
    return unknown ? std::nullopt : std::optional<bool>{false};
  }

  [[nodiscard]] std::optional<bool> has_cfg_path_with_event_pending(
      const PendingEvent &event, uint64_t consumer_offset, rj_code_arch_t arch,
      bool before_target = false, bool requires_active_lane = false, uint32_t minimum_younger = 0,
      std::optional<WaitRange> weakened = {}, std::optional<LaneOperand> lane_operand = {},
      bool scalar_full_drains_only = false, bool track_younger_predicates = false,
      std::optional<MaskedSccPredicate> masked_guard = {}, std::optional<LoopParity> parity = {}) {
    requires_active_lane |= lane_operand.has_value();
    if (cfg_views_.empty())
      return std::nullopt;

    // A guarded barrier arrive ages an older ASYNC load only on a nonempty
    // EXEC edge. Its LDS memory-order dependency has no masked register operand.
    const bool track_async_issue =
        event.counter == WaitCounterKind::Async && cfg_has_exec_guards_ &&
        (event.kind == WaitEventKind::AsyncLdsLoad || event.kind == WaitEventKind::AsyncBarrier);

    const bool expert = amdgpu::supports_expert_scheduling(arch) &&
                        (event.counter == WaitCounterKind::VaVdst ||
                         (event.counter == WaitCounterKind::VmVsrc &&
                          event.kind != WaitEventKind::Gfx10ScalarSource));
    const bool scratch_guard = !scalar_full_drains_only && has_guarded_scratch_result(event);
    const bool resolve_scratch_relative = scratch_guard && cfg_can_resolve_scratch_relative_;
    // Every consumer in a neutral run observes the same pending-event state.
    // Reuse one path query for the run instead of repeating the CFG search for
    // every arithmetic use of the same asynchronous load. A producer between
    // the run's start and the consumer, or additional overlapping contexts,
    // prevents this equivalence. Producers later in the block are safe too:
    // their loop-carried state is unchanged throughout this prefix of the run.
    if (!before_target && !requires_active_lane && !weakened && !expert && !scratch_guard) {
      const auto query = path_query_starts_.find(consumer_offset);
      if (query != path_query_starts_.end() && query->second.offset < consumer_offset &&
          (event.section_offset < query->second.offset || event.section_offset > consumer_offset)) {
        const auto start = path_query_starts_.find(query->second.offset);
        if (start != path_query_starts_.end() && start->second.contexts == query->second.contexts)
          consumer_offset = query->second.offset;
      }
    }
    const auto cache_key = std::tuple{event.section_offset,
                                      consumer_offset,
                                      event.counter,
                                      before_target,
                                      requires_active_lane,
                                      minimum_younger,
                                      weakened,
                                      lane_operand,
                                      scalar_full_drains_only,
                                      track_younger_predicates,
                                      masked_guard,
                                      parity,
                                      expert ? event.kind : WaitEventKind::Unknown,
                                      expert && !event.regs.none(),
                                      expert ? event.wmma_wait_states : uint8_t{0}};
    if (auto cached = feasible_path_cache_.find(cache_key); cached != feasible_path_cache_.end())
      return cached->second;

    // The dataflow state passed to the current instruction already proves
    // that an earlier event in the same straight-line block is still pending.
    // Avoid launching the path-sensitive CFG search for the overwhelmingly
    // common compiler pattern of a wait shortly after its producer.
    if (!requires_active_lane && !track_async_issue && !weakened && !scratch_guard &&
        current_cfg_view_index_ &&
        *current_cfg_view_index_ < cfg_views_.size()) {
      const BasicBlock *current_block = cfg_views_[*current_cfg_view_index_].block;
      if (current_block != nullptr && event.section_offset < consumer_offset &&
          event.section_offset >= current_block->start_offset() &&
          consumer_offset < current_block->end_offset()) {
        feasible_path_cache_[cache_key] = true;
        return true;
      }
    }

    const std::vector<size_t> producer_block_indices =
        cfg_block_indices_containing(event.section_offset);
    if (before_target && minimum_younger == 0 && !track_async_issue && current_cfg_view_index_ &&
        *current_cfg_view_index_ < cfg_views_.size() &&
        std::ranges::any_of(
            producer_block_indices,
            [&](size_t producer_block_index) {
              return cfg_block_dominates(producer_block_index, *current_cfg_view_index_);
            })) {
      feasible_path_cache_[cache_key] = true;
      return true;
    }
    const std::vector<size_t> consumer_block_indices =
        cfg_block_indices_containing(consumer_offset);
    if (producer_block_indices.empty() || consumer_block_indices.empty())
      return std::nullopt;

    const SharedReachability consumer_reaching = reachable_blocks(consumer_block_indices, false);
    const auto &can_reach_consumer = *consumer_reaching;
    const bool producer_can_reach_consumer =
        std::ranges::any_of(producer_block_indices, [&](size_t producer_block_index) {
          return can_reach_consumer[producer_block_index] != 0;
        });
    if (!producer_can_reach_consumer) {
      feasible_path_cache_[cache_key] = false;
      return false;
    }

    const SharedReachability producer_reaching = reachable_blocks(producer_block_indices, false);
    const SharedReachability producer_reachable = reachable_blocks(producer_block_indices, true);
    const auto &can_reach_producer = *producer_reaching;
    const auto &reachable_from_producer = *producer_reachable;

    // An earlier predicate can select the producer. A repeated predicate after
    // issue can select younger ordered operations and the consuming matrix path.
    // Track the second group only on the additional check of a reported hazard.
    using PredicateKey = std::tuple<SccPredicate::Kind, uint16_t, uint32_t>;
    std::set<PredicateKey> predicates_before_producer;
    std::set<PredicateKey> predicates_after_producer;
    std::set<uint16_t> split_counts;
    // The CFG is immutable during these searches. Index scalar comparisons
    // once rather than re-decoding every instruction for every event pair.
    std::set<PredicateKey> relevant_predicates;
    // A single SCC comparison can guard two visits to a VMEM producer.
    // Only track literal assignments for one equality register in its loop.
    const bool find_cyclic_vmem_guard = track_younger_predicates && current_kernel_ &&
                                        cfg_has_exec_guards_ && arch == ROCJITSU_CODE_ARCH_CDNA3 &&
                                        event.counter == WaitCounterKind::Load &&
                                        event.kind == WaitEventKind::VmemNoSamplerLoad;
    std::set<uint16_t> cyclic_guard_registers;
    for (const auto &[block_index, section_offset, predicate] : path_predicates_) {
      const PredicateKey key{predicate.kind, predicate.sgpr, predicate.value};
      if (predicate.kind == SccPredicate::Kind::LeUnsignedImm && !track_younger_predicates)
        continue;
      if (can_reach_producer[block_index] != 0 && section_offset < event.section_offset)
        predicates_before_producer.insert(key);
      if (reachable_from_producer[block_index] != 0 && can_reach_consumer[block_index] != 0 &&
          section_offset >= event.section_offset) {
        const bool repeated = !predicates_after_producer.insert(key).second;
        if (track_younger_predicates && section_offset <= consumer_offset) {
          if (repeated)
            relevant_predicates.insert(key);
          if (predicate.kind == SccPredicate::Kind::LeUnsignedImm)
            split_counts.insert(predicate.sgpr);
          if (find_cyclic_vmem_guard && predicate.kind == SccPredicate::Kind::EqImm &&
              std::ranges::any_of(producer_block_indices, [&](size_t producer) {
                return cfg_component_[producer] == cfg_component_[block_index] &&
                       std::ranges::any_of(cfg_views_[producer].predecessors, [&](size_t pred) {
                         return cfg_component_[pred] == cfg_component_[producer];
                       });
              }))
            cyclic_guard_registers.insert(predicate.sgpr);
        }
      }
    }
    // The same count can skip younger issues at equality and select its
    // consuming unroll stage with an upper bound. Use the existing scalar
    // constraints on both arms, without tracing arithmetic across a write.
    if (!split_counts.empty()) {
      for (const auto &[kind, reg, value] : predicates_after_producer) {
        if (split_counts.contains(reg) &&
            (kind == SccPredicate::Kind::EqImm || kind == SccPredicate::Kind::LeUnsignedImm))
          relevant_predicates.emplace(kind, reg, value);
      }
    }
    const std::optional<uint16_t> cyclic_guard =
        cyclic_guard_registers.size() == 1
            ? std::optional<uint16_t>{*cyclic_guard_registers.begin()}
            : std::nullopt;
    if (track_younger_predicates && !cyclic_guard &&
        (relevant_predicates.empty() || (before_target && minimum_younger != 0)))
      return feasible_path_cache_[cache_key] = true;
    std::ranges::set_intersection(predicates_before_producer, predicates_after_producer,
                                  std::inserter(relevant_predicates, relevant_predicates.end()));
    // Scalar memory is unordered, so a zero wait discharges its result on a
    // particular path. Wave32 EXEC_LO and VCC_LO each describe the whole wave.
    const bool track_masks = requires_active_lane || weakened || track_async_issue ||
                             cyclic_guard || expert ||
                             scratch_guard ||
                             // GFX10 memory sources issue even with empty EXEC. Track branch
                             // masks to reject infeasible paths without requiring active lanes.
                             (event.kind == WaitEventKind::Gfx10ScalarSource &&
                              current_kernel_ && wavefront_size_ == 32 && cfg_has_exec_guards_) ||
                             (!before_target && event.kind == WaitEventKind::Smem &&
                              event.counter != WaitCounterKind::X && current_kernel_ &&
                              wavefront_size_ == 32 && cfg_has_mask_branches_);
    const bool track_exec_source =
        track_masks && wavefront_size_ == 32 &&
        (arch == ROCJITSU_CODE_ARCH_RDNA3 || arch == ROCJITSU_CODE_ARCH_RDNA3_5) &&
        has_gfx11_exec_masked_result(event);
    if (relevant_predicates.empty() && !cfg_has_constant_scc_ && !track_masks && !masked_guard &&
        !parity) {
      // Without a supported repeated scalar predicate there is no correlation
      // for the path filter to prove. The converged dataflow state already
      // established a structural path on which this event remains pending.
      feasible_path_cache_[cache_key] = true;
      return true;
    }

    std::optional<PathDelayHistories> delay_histories;
    if (expert && event.counter == WaitCounterKind::VaVdst && cfg_has_expert_delay_)
      delay_histories.emplace();

    struct SearchState {
      size_t block_index = 0;
      size_t inst_index = 0;
      bool pending = false;
      PathProgress progress;
      ScalarConstraints constraints;
      std::optional<SccPredicate> scc_predicate;
      std::optional<uint32_t> constraint_id = std::nullopt;
      PathMasks masks{};
      bool crossed_wait = false;
      ParityStep parity{15, 15, 15};
    };

    constexpr uint32_t kMaxTrackedAge = 255;
    constexpr size_t kMaxPathSearchStates = 200000;
    const bool cdna3_vmem_pending =
        arch == ROCJITSU_CODE_ARCH_CDNA3 && event.counter == WaitCounterKind::Load;
    const uint32_t cdna3_vmem_capacity =
        cdna3_vmem_pending ? maximum_dependency_wait(arch, event.counter).value() + 1 : 0;
    bool unknown_lane_witness = false;
    std::vector<SearchState> worklist;
    for (size_t i = 0; i < cfg_views_.size(); ++i) {
      const BasicBlock *block = cfg_views_[i].block;
      if ((track_masks || scalar_full_drains_only) && current_kernel_) {
        // A descriptor-backed kernel is analyzed individually; its first
        // context is its dispatch entry even if that entry has a backedge.
        if (i == 0 && can_reach_producer[i] != 0) {
          PathMasks masks;
          masks.exec = true;
          worklist.push_back({i, 0, false, {}, {}, std::nullopt, std::nullopt, masks});
        }
      } else if (block != nullptr && cfg_views_[i].predecessors.empty() &&
                 can_reach_producer[i] != 0) {
        PathMasks masks;
        if (track_masks) {
          masks.scalar_pending_counters = UINT64_MAX;
          masks.scalar_pending_unknown = true;
        }
        worklist.push_back({i, 0, false, {}, {}, std::nullopt, std::nullopt, masks});
      }
    }
    if (worklist.empty() && track_masks && current_kernel_) {
      feasible_path_cache_[cache_key] = true;
      return true;
    }
    if (worklist.empty()) {
      if (weakened)
        return std::nullopt;
      for (size_t producer_block_index : producer_block_indices) {
        PathMasks masks;
        masks.pending_scc_write_counters = UINT64_MAX;
        if (track_masks) {
          masks.scalar_pending_counters = UINT64_MAX;
          masks.scalar_pending_unknown = true;
        }
        worklist.push_back(
            {producer_block_index, 0, false, {}, {}, std::nullopt, std::nullopt, masks});
      }
    }
    if (expert && !current_kernel_)
      for (auto &root : worklist)
        root.progress.expert_enabled = true;
    struct SearchKey {
      size_t block_index;
      size_t inst_index;
      bool pending;
      PathProgress progress;
      uint32_t constraint_id;
      bool crossed_wait;
      bool operator==(const SearchKey &) const = default;
    };
    struct SearchKeyHash {
      size_t operator()(const SearchKey &key) const {
        size_t hash = key.block_index * 2 + key.crossed_wait;
        for (size_t value :
             {key.inst_index, static_cast<size_t>(key.pending),
              static_cast<size_t>(key.progress.age), static_cast<size_t>(key.progress.core),
              static_cast<size_t>(key.progress.store_age),
              static_cast<size_t>(key.progress.expert_enabled),
              static_cast<size_t>(key.progress.source_implies_nonempty_exec),
              static_cast<size_t>(key.progress.crossed_exec_expansion),
              static_cast<size_t>(key.progress.younger_source_primaries),
              std::hash<std::bitset<128>>{}(key.progress.saved_source_exec),
              static_cast<size_t>(key.progress.delay_id),
              static_cast<size_t>(key.progress.unknown_delay_issue),
              static_cast<size_t>(key.constraint_id)})
          hash ^= value + 0x9e3779b9u + (hash << 6) + (hash >> 2);
        return hash;
      }
    };
    // Most instructions preserve scalar facts. Intern them only when they
    // change, and keep visited-state keys numeric instead of repeatedly
    // formatting and comparing the same constraint strings.
    std::unordered_map<std::string, uint32_t> constraint_ids;
    std::unordered_set<SearchKey, SearchKeyHash> visited;
    size_t states_processed = 0;

    auto can_continue_from = [&](size_t block_index, bool pending) {
      return pending ? can_reach_consumer[block_index] != 0 : can_reach_producer[block_index] != 0;
    };

    while (!worklist.empty()) {
      SearchState state = worklist.back();
      worklist.pop_back();
      state.progress.age = std::min(state.progress.age, kMaxTrackedAge);
      if (!state.constraint_id) {
        auto key = constraints_key(state.constraints, state.scc_predicate);
        key += ":scc-pending:" + std::to_string(state.masks.pending_scc_write_counters) + ';';
        if (parity) {
          key.push_back(static_cast<char>(state.parity.pairs));
          key.push_back(static_cast<char>(state.parity.on_true));
          key.push_back(static_cast<char>(state.parity.on_false));
        }
        if (track_masks) {
          for (auto value : {state.masks.exec, state.masks.vcc, state.masks.scc})
            key.push_back(!value ? '?' : *value ? '1' : '0');
          key.push_back(state.masks.vcc_is_exec ? '1' : '0');
          key.push_back(state.masks.scc_is_exec ? '1' : '0');
          key += ':' + std::to_string(state.masks.scalar_pending_counters);
          for (uint16_t reg : state.masks.scalar_pending_sgprs)
            key += ':' + std::to_string(reg);
          key.push_back(';');
          key.push_back(state.masks.scalar_pending_exec ? '1' : '0');
          key.push_back(state.masks.scalar_pending_vcc ? '1' : '0');
          key.push_back(state.masks.scalar_pending_scc ? '1' : '0');
          key.push_back(state.masks.scalar_pending_unknown ? '1' : '0');
          if (track_exec_source) {
            state.masks.exec_lanes.append_key(key, 'e');
            if (lane_operand && state.masks.exec_lanes.exact)
              key.push_back('!');
            if (state.pending)
              state.masks.issue_lanes.append_key(key, 'p');
          }
        }
        const auto [entry, inserted] = constraint_ids.try_emplace(
            std::move(key), static_cast<uint32_t>(constraint_ids.size()));
        state.constraint_id = entry->second;
      }
      if (!visited
               .insert({state.block_index, state.inst_index, state.pending, state.progress,
                        *state.constraint_id, state.crossed_wait})
               .second)
        continue;
      if (++states_processed > kMaxPathSearchStates) {
        feasible_path_cache_[cache_key] = std::nullopt;
        return std::nullopt;
      }

      const CfgBlockView &block_view = cfg_views_[state.block_index];
      if (state.inst_index >= block_view.instructions.size()) {
        if (block_view.block == nullptr)
          continue;
        for (size_t successor : block_view.successors) {
          if (can_continue_from(successor, state.pending))
            worklist.push_back({successor, 0, state.pending, state.progress, state.constraints,
                                state.scc_predicate, state.constraint_id, state.masks,
                                state.crossed_wait, state.parity});
        }
        continue;
      }

      const CfgInstructionView &inst_view = block_view.instructions[state.inst_index];
      const bool scratch_sgpr_write = resolve_scratch_relative && inst_view.scratch_relative_sgpr;
      size_t run_end =
          delay_histories && state.pending ? state.inst_index : inst_view.neutral_run_end;
      if (expert) {
        // A scheduling-mode write is neutral to ordinary path queries. An
        // expert event cannot survive a switch back to automatic scheduling.
        for (size_t i = state.inst_index; i < run_end; ++i) {
          if (block_view.instructions[i].expert_mode_write) {
            run_end = i;
            break;
          }
        }
      }
      if ((masked_guard || parity || cyclic_guard || state.progress.saved_source_exec.any()) &&
          run_end > state.inst_index) {
        // Relative SGPR writes may select the root even if the encoded register
        // differs. Inspect them before skipping any cached neutral operations.
        for (size_t i = state.inst_index; i < run_end; ++i) {
          const auto &view = block_view.instructions[i];
          if (!view.instruction->mnemonic().starts_with("s_movrel"))
            continue;
          if (!(resolve_scratch_relative && view.scratch_relative_sgpr))
            return feasible_path_cache_[cache_key] = std::nullopt;
          run_end = i;
          break;
        }
      }
      if (run_end > state.inst_index) {
        // A neutral run has one entry and changes no pending-event, age, or
        // scalar-predicate state. Its entry's visited key therefore covers its
        // interior. Keep query endpoints explicit and charge every skipped
        // instruction to the original conservative search budget.
        for (uint64_t endpoint : {event.section_offset, consumer_offset}) {
          const auto begin = block_view.instructions.begin() + state.inst_index;
          const auto end = block_view.instructions.begin() + run_end;
          const auto target = std::lower_bound(begin, end, endpoint,
                                               [](const CfgInstructionView &view, uint64_t offset) {
                                                 return view.section_offset < offset;
                                               });
          if (target != end && target->section_offset == endpoint)
            run_end = static_cast<size_t>(target - block_view.instructions.begin());
        }
        if (run_end > state.inst_index) {
          const size_t additional_states = run_end - state.inst_index - 1;
          if (additional_states > kMaxPathSearchStates - states_processed) {
            feasible_path_cache_[cache_key] = std::nullopt;
            return std::nullopt;
          }
          states_processed += additional_states;
          state.inst_index = run_end;
          worklist.push_back(std::move(state));
          continue;
        }
      }
      const Instruction &inst = *inst_view.instruction;
      if (inst.is_vskip_affected() && inst_view.vector_skip != false) {
        if (!inst_view.vector_skip.has_value())
          return feasible_path_cache_[cache_key] = std::nullopt;
        ++state.inst_index;
        worklist.push_back(std::move(state));
        continue;
      }
      if (before_target && inst_view.section_offset == consumer_offset && state.pending &&
          (!minimum_younger || state.progress.age < minimum_younger)) {
        if (!state.progress.unknown_delay_issue)
          return feasible_path_cache_[cache_key] = true;
        unknown_lane_witness = true;
      }
      std::optional<WaitFields> replacement;
      if (weakened && state.pending && inst_view.section_offset >= weakened->first &&
          inst_view.section_offset <= weakened->last) {
        replacement = inst.is_waitcnt() ? cfg_explicit_wait_fields(state.block_index, inst, arch)
                                        : embedded_wait_fields(inst, arch).value();
        const auto field =
            replacement ? (*replacement)[counter_index(event.counter)] : std::nullopt;
        if (field && (*field < weakened->count || weakened->original_minimum == field)) {
          (*replacement)[counter_index(event.counter)] = weakened->count;
          state.crossed_wait = true;
        } else {
          replacement.reset();
        }
      }
      if (expert && inst_view.expert_mode_write) {
        if (!inst_view.expert_mode_write->known || !inst_view.expert_mode_write->enabled)
          state.progress.delay_id = 0;
        state.progress.expert_enabled = inst_view.expert_mode_write->enabled;
        if (!state.progress.expert_enabled)
          state.pending = false;
      }
      std::optional<ExpertValuDelayState> delay;
      if (delay_histories && state.pending) {
        delay = delay_histories->values[state.progress.delay_id];
        const auto selected = delay->before(inst, arch);
        const bool certain_issue =
            state.masks.exec == true || inst_view.valu_issues_with_empty_exec;
        if (certain_issue) {
          for (const auto [kind, younger] : selected) {
            if ((kind == event.kind && state.progress.age >= younger) ||
                (kind == WaitEventKind::ValuCore && event.wmma_wait_states &&
                 state.progress.core > event.wmma_wait_states + younger))
              state.pending = false;
          }
        } else if (std::ranges::any_of(
                       selected, [](auto slot) { return slot.kind != WaitEventKind::Unknown; })) {
          state.progress.unknown_delay_issue = true;
        }
      }
      if (scalar_full_drains_only) {
        // Allow every lane to issue, skip and consume. Only an explicit zero
        // on this counter can prove completion independently of all EXEC paths.
        if (inst.flags() & INDIRECT_CALL || inst.mnemonic().starts_with("s_call") ||
            inst.mnemonic().starts_with("s_movrel"))
          return feasible_path_cache_[cache_key] = std::nullopt;
        if (state.pending && inst.is_waitcnt()) {
          const auto fields = explicit_wait_fields(inst, arch).value();
          if (fields && (*fields)[counter_index(event.counter)] == 0)
            state.pending = false;
        }
      } else {
        apply_path_pre_dependency_waits(inst, event, arch, state.pending, state.progress, expert,
                                        replacement ? &*replacement : nullptr);
      }
      if (!state.pending) {
        state.crossed_wait = false;
        state.progress.age = 0;
        state.progress.store_age = 0;
        state.progress.core = 0;
        state.progress.source_implies_nonempty_exec = false;
        state.progress.crossed_exec_expansion = false;
        state.progress.saved_source_exec.reset();
        state.progress.younger_source_primaries = 0;
        state.progress.delay_id = 0;
        state.progress.unknown_delay_issue = false;
      }

      if (!before_target && inst_view.section_offset == consumer_offset && state.pending &&
          (!minimum_younger || state.progress.age < minimum_younger) &&
          (!weakened || state.crossed_wait)) {
        const std::optional<bool> reaches =
            lane_operand
                ? physical_lane_witness(inst, *lane_operand, state.masks, state.constraints, arch)
                : std::optional<bool>{!requires_active_lane ||
                                      (state.masks.exec != false &&
                                       (!track_exec_source ||
                                        state.masks.issue_lanes.overlaps(state.masks.exec_lanes)))};
        if (reaches == true && !state.progress.unknown_delay_issue)
          return feasible_path_cache_[cache_key] = true;
        unknown_lane_witness |=
            !reaches.has_value() || (reaches == true && state.progress.unknown_delay_issue);
      }

      const auto *issue = inst.amdgpu_memory_issue_info();
      const bool vector_issue_masked =
          (issue && issue->exec_masked) || (expert && event.counter == WaitCounterKind::VaVdst &&
                                            !inst_view.valu_issues_with_empty_exec);
      const bool may_skip_lanes = (requires_active_lane || weakened || cyclic_guard || expert) &&
                                  state.masks.exec != true && vector_issue_masked;
      const bool same_producer = inst_view.section_offset == event.section_offset;
      const bool no_result = may_skip_lanes && state.masks.exec == false;
      // CDNA3 dedicated VMEM still occupies VMCNT with empty EXEC while the
      // tracked producer is outstanding. VSKIP was handled above.
      if (same_producer &&
          (!no_result ||
           (!cdna3_vmem_pending && !(expert && event.counter == WaitCounterKind::VmVsrc))) &&
          (!expert || state.progress.expert_enabled)) {
        // A masked issue with empty EXEC cannot replace the original result.
        if (!no_result) {
          if (track_exec_source) {
            if (state.pending)
              state.masks.issue_lanes.include(state.masks.exec_lanes);
            else
              state.masks.issue_lanes = state.masks.exec_lanes;
            state.constraint_id.reset();
          }
          state.progress.crossed_exec_expansion = false;
          state.progress.saved_source_exec.reset();
          state.progress.younger_source_primaries = 0;
          // The live physical source implies its original mask had a lane.
          // This evidence applies only to the pending event, never to a CFG edge.
          state.progress.source_implies_nonempty_exec =
              expert && !event.regs.none() &&
              (event.counter == WaitCounterKind::VmVsrc ||
               (cfg_has_repeated_exec_filter_ && event.kind == WaitEventKind::ValuCore &&
                !inst_view.valu_issues_with_empty_exec));
          state.pending = true;
          state.progress.age = 0;
          state.progress.store_age = 0;
          state.progress.core = 0;
        } else if (state.pending) {
          state.progress.age = 0;
          state.progress.store_age = 0;
          state.progress.core = 0;
        }
      } else if (state.pending && !inst.is_waitcnt() && is_program_end(inst.mnemonic())) {
        state.pending = false;
        state.progress.age = 0;
        state.progress.store_age = 0;
      } else if (!scalar_full_drains_only && state.pending && !inst.is_waitcnt() &&
                 (expert || !may_skip_lanes || cdna3_vmem_pending) &&
                 (inst_view.issued_counters & (uint64_t{1} << counter_index(event.counter))) != 0) {
        if (expert) {
          const auto model = amdgpu::waitcnt_model(arch).value();
          const auto order = normalized_hardware_event_kind(event.counter, event.kind, model);
          const bool vm_source = event.counter == WaitCounterKind::VmVsrc;
          const bool dedicated = vm_source && dedicated_source_issues(inst);
          const bool allocated =
              !vm_source || vm_source_results_allocated(inst, arch, inst_view.vgpr_msb,
                                                        minimum_source_vgprs(arch));
          const bool active =
              allocated &&
              state.masks.exec.value_or(state.progress.source_implies_nonempty_exec &&
                                        (!state.progress.crossed_exec_expansion || dedicated));
          const auto current = effective_events(inst, inst_view.vgpr_msb, arch);
          for (const auto &issued : current) {
            if (vm_source && event.kind == WaitEventKind::VmemStore &&
                issued.counter == WaitCounterKind::Store && issued.kind == WaitEventKind::VmemStore)
              state.progress.store_age = std::min<unsigned>(
                  state.progress.store_age + issue_units(inst, issued, arch), kMaxTrackedAge);
            if (issued.counter != event.counter)
              continue;
            const auto units = issue_units(inst, issued, arch);
            if (!units)
              continue;
            const bool source_issues =
                event.counter == WaitCounterKind::VmVsrc &&
                (pending_source_forces_issue(event.kind, issued.kind, state.masks.exec == false) ||
                 (state.progress.younger_source_primaries & dedicated_source_primary(issued.kind)));
            if (!active && !source_issues && !inst_view.valu_issues_with_empty_exec)
              continue;
            if (order &&
                order == normalized_hardware_event_kind(issued.counter, issued.kind, model)) {
              state.progress.age = std::min(state.progress.age + units, kMaxTrackedAge);
              if (dedicated && active && dedicated_source_primary(issued.kind)) {
                const auto du =
                    inst_def_use_for_waitcheck(inst, inst_view.vgpr_msb, arch, wavefront_size_);
                if (!registers_for_event(inst, du, issued, inst_view.vgpr_msb, arch).none())
                  state.progress.younger_source_primaries |= dedicated_source_primary(issued.kind);
              }
            }
            if (event.wmma_wait_states && issued.kind == WaitEventKind::ValuCore)
              state.progress.core =
                  std::min<unsigned>(state.progress.core + 1, event.wmma_wait_states + 16);
          }
        } else if (is_completion_counter(event.counter)) {
          const auto order = completion_order_kind(event.counter, event.kind, arch);
          const auto current = effective_events(inst, inst_view.vgpr_msb, arch);
          for (const auto &issued : current) {
            const bool dedicated_vmem = issued.kind == WaitEventKind::VmemNoSamplerLoad ||
                                        issued.kind == WaitEventKind::VmemStore;
            if (issued.counter == event.counter && order && (!may_skip_lanes || dedicated_vmem) &&
                guaranteed_issue_order(issued.counter, issued.kind, arch,
                                       state.masks.exec == true) == order) {
              state.progress.age =
                  std::min(state.progress.age + issue_units(inst, issued, arch), kMaxTrackedAge);
              // If the tracked result survives, every guaranteed younger
              // operation in its FIFO must also occupy a VMCNT position.
              if (cdna3_vmem_pending && state.progress.age >= cdna3_vmem_capacity)
                state.pending = false;
            }
          }
        } else if (event.counter != WaitCounterKind::X ||
                   (!is_async_transfer_kind(event.kind) && !is_async_lds_load(inst.mnemonic()) &&
                    !is_async_lds_store(inst.mnemonic()))) {
          state.progress.age = std::min<uint32_t>(state.progress.age + 1, kMaxTrackedAge);
        }
      }

      if (delay_histories && state.pending) {
        // Only a known issue may prove queue progress. Although skipped VALUs
        // count toward INSTID, they need not have completed any registers.
        // Forget the prefix if its issue was ambiguous, on opaque calls, and
        // on reissue (earlier histories cannot establish this new producer).
        if (inst_view.section_offset == event.section_offset || !delay) {
          delay.emplace();
          state.progress.unknown_delay_issue = false;
        }
        if ((inst.flags() & INDIRECT_CALL) || inst.mnemonic().starts_with("s_call")) {
          delay.emplace();
          state.progress.unknown_delay_issue = true;
        }
        if (const auto kind = expert_valu_kind(inst, arch)) {
          if (state.masks.exec == true || inst_view.valu_issues_with_empty_exec) {
            delay->issue(inst, *kind,
                         issue_units(inst, ClassifiedEvent{event.counter, *kind}, arch), arch);
          } else if (state.masks.exec == false) {
            delay->skip(inst, *kind, arch);
          } else {
            delay.emplace();
            state.progress.unknown_delay_issue = true;
          }
        }
        const auto id = delay_histories->intern(*delay, state.progress.delay_id);
        if (!id)
          return feasible_path_cache_[cache_key] = std::nullopt;
        state.progress.delay_id = *id;
      }
      if (state.pending && expert && event.counter == WaitCounterKind::VmVsrc &&
          cfg_has_saved_source_exec_ && !cfg_has_unknown_successor_) {
        transfer_saved_source_mask(
            inst, inst_view.invalidates_exec, inst_view.scalar_destination_pending,
            path_scalar_defs_.at(&inst), state.progress.source_implies_nonempty_exec,
            state.progress.crossed_exec_expansion, state.progress.saved_source_exec);
      } else if (inst_view.invalidates_exec || (inst.flags() & INDIRECT_CALL)) {
        if (event.counter == WaitCounterKind::VmVsrc &&
            state.progress.source_implies_nonempty_exec && preserves_exec_lanes(inst))
          state.progress.crossed_exec_expansion = true;
        else
          state.progress.source_implies_nonempty_exec = false;
      }

      // This extra query tracks just one masked expression. Its Boolean value
      // uses a constraint slot outside the physical SGPR range.
      constexpr uint16_t kMaskedGuard = std::numeric_limits<uint16_t>::max();
      if ((masked_guard || parity || cyclic_guard) &&
          ((inst.flags() & INDIRECT_CALL) || inst.mnemonic().starts_with("s_call") ||
           (inst.mnemonic().starts_with("s_movrel") && !scratch_sgpr_write)))
        return feasible_path_cache_[cache_key] = std::nullopt;
      std::span<const uint16_t> scalar_defs = path_scalar_defs_.at(&inst);
      if (scratch_sgpr_write)
        scalar_defs = std::span{&*inst_view.scratch_relative_sgpr, 1};
      if (parity && !is_scc_conditional_branch(inst.mnemonic()) &&
          (inst_view.invalidates_scc || !scalar_defs.empty())) {
        // An independent SGPR definition can change a tracked bit before the
        // SCC branch. Widen both arms if the new definition is unsupported.
        state.parity = step_parity(inst, *parity, state.parity.pairs, scalar_defs,
                                   state.constraints);
        state.constraint_id.reset();
      }
      const auto previous_predicate = state.scc_predicate;
      bool constraints_changed = false;
      std::optional<ScalarValueConstraint> assigned_constraint;
      std::optional<RegisterRef> assigned_reg;
      uint64_t pending_scc_write_counters = state.masks.pending_scc_write_counters;
      pending_scc_write_counters &= ~inst_view.zero_wait_counters;
      pending_scc_write_counters |= inst_view.issued_scc_write_counters;
      if ((inst.flags() & INDIRECT_CALL ||
           (track_exec_source && inst.mnemonic().starts_with("s_call"))) &&
          !(track_exec_source && inst_view.enters_resolved_callee))
        pending_scc_write_counters = UINT64_MAX;
      if (track_masks) {
        const auto previous_masks = state.masks;
        update_path_masks(state.masks, inst_view, state.constraints, wavefront_size_,
                          track_exec_source, scratch_sgpr_write);
        if (expert && block_view.repeated_exec_filter_arm && state.inst_index < 2)
          state.masks.exec = previous_masks.exec;
        if (state.masks != previous_masks)
          state.constraint_id.reset();
        if ((inst.flags() & INDIRECT_CALL ||
             (track_exec_source && inst.mnemonic().starts_with("s_call"))) &&
            !(track_exec_source && inst_view.enters_resolved_callee)) {
          state.constraints.clear();
          state.scc_predicate.reset();
          state.constraint_id.reset();
        }
        if (inst.mnemonic() == "s_mov_b32" && inst.dst_operand(0)) {
          assigned_reg = inst.dst_operand(0)->to_register_ref();
          if (assigned_reg && assigned_reg->cls == RegClass::SGPR &&
              (path_mask_inputs_.contains(assigned_reg->index) ||
               assigned_reg->index == cyclic_guard ||
               (track_exec_source && path_exec_inputs_.contains(assigned_reg->index)) ||
               (lane_operand &&
                std::ranges::find(lane_operand->selector_sgprs, assigned_reg->index) !=
                    lane_operand->selector_sgprs.end())) &&
              !path_operand_has_pending_scalar_result(inst.dst_operand(0), state.masks) &&
              !path_operand_has_pending_scalar_result(inst.src_operand(0), state.masks)) {
            if (const auto value = path_dword(inst.src_operand(0), state.constraints)) {
              assigned_constraint.emplace().equal = *value;
            } else if (const auto source = inst.src_operand(0)->to_register_ref();
                       track_exec_source && source && source->cls == RegClass::SGPR &&
                       source->width == 1) {
              const auto known = state.constraints.find(source->index);
              if (known != state.constraints.end())
                assigned_constraint = known->second;
            }
          }
        }
      }
      if (state.masks.pending_scc_write_counters != pending_scc_write_counters) {
        state.masks.pending_scc_write_counters = pending_scc_write_counters;
        state.constraint_id.reset();
      }
      if (pending_scc_write_counters) {
        state.scc_predicate.reset();
        if (parity)
          state.parity.on_true = state.parity.on_false = state.parity.pairs;
      }
      for (uint16_t reg : scalar_defs) {
        constraints_changed |= state.constraints.erase(reg) != 0;
        if (masked_guard && reg == masked_guard->source) {
          constraints_changed |= state.constraints.erase(kMaskedGuard) != 0;
          if (state.scc_predicate && state.scc_predicate->sgpr == kMaskedGuard)
            state.scc_predicate.reset();
        }
        for (Wave32Lanes *lanes : {&state.masks.exec_lanes, &state.masks.issue_lanes}) {
          if (lanes->source == reg) {
            lanes->source.reset();
            lanes->inverted = false;
            state.constraint_id.reset();
          }
        }
        if (state.scc_predicate && state.scc_predicate->kind != SccPredicate::Kind::Constant &&
            state.scc_predicate->sgpr == reg)
          state.scc_predicate.reset();
      }

      if (assigned_constraint) {
        state.constraints[assigned_reg->index] = std::move(*assigned_constraint);
        constraints_changed = true;
      }
      if (const auto branch = mask_branch_class(inst.mnemonic()); track_masks && branch) {
        if (track_exec_source && block_view.exec_nop_join) {
          // Either edge arrives with the same incoming state. Separately retaining
          // each edge's EXEC constraint can multiply identical NOP-only paths.
          const size_t join = *block_view.exec_nop_join;
          if (can_continue_from(join, state.pending)) {
            state.block_index = join;
            state.inst_index = 0;
            worklist.push_back(std::move(state));
          }
          continue;
        }
        const bool pending_write = state.masks.scalar_pending_unknown ||
                                   (*branch == RegClass::EXEC ? state.masks.scalar_pending_exec
                                                              : state.masks.scalar_pending_vcc);
        const auto nonempty = pending_write               ? std::optional<bool>{}
                              : *branch == RegClass::EXEC ? state.masks.exec
                              : state.masks.vcc_is_exec   ? state.masks.exec
                                                          : state.masks.vcc;
        const auto delta = inst.branch_offset_bytes();
        if (delta) {
          const auto next = inst_view.section_offset + static_cast<uint64_t>(inst.size());
          const auto target = next + *delta;
          if (target != next) {
            for (size_t succ : block_view.successors) {
              const auto dest = cfg_views_[succ].block->start_offset();
              if (!can_continue_from(succ, state.pending))
                continue;
              auto successor = state;
              successor.block_index = succ;
              successor.inst_index = 0;
              if (dest == target || dest == next) {
                const bool edge_nonempty = (dest == target) == inst.mnemonic().ends_with("nz");
                if (nonempty && *nonempty != edge_nonempty)
                  continue;
                auto &mask = *branch == RegClass::EXEC ? successor.masks.exec : successor.masks.vcc;
                mask = pending_write ? std::optional<bool>{} : std::optional<bool>{edge_nonempty};
                if (!pending_write && *branch == RegClass::VCC &&
                    successor.masks.vcc_is_exec) {
                  successor.masks.exec = edge_nonempty;
                  successor.masks.vcc_is_exec = false;
                }
                if (!pending_write && *branch == RegClass::EXEC &&
                    successor.masks.exec_lanes.source &&
                    !apply_scc_branch_constraint(
                        successor.constraints,
                        SccPredicate{SccPredicate::Kind::EqImm, *successor.masks.exec_lanes.source,
                                     successor.masks.exec_lanes.inverted ? UINT32_MAX : 0},
                        !edge_nonempty))
                  continue;
                successor.constraint_id.reset();
              }
              worklist.push_back(std::move(successor));
            }
            continue;
          }
        }
      }
      const auto selected_mask =
          masked_guard ? masked_predicates_.find(&inst) : masked_predicates_.end();
      if (selected_mask != masked_predicates_.end() && selected_mask->second == *masked_guard &&
          !pending_scc_write_counters) {
        state.scc_predicate = SccPredicate{SccPredicate::Kind::EqImm, kMaskedGuard, 1};
      } else if (inst_view.predicate_index != std::numeric_limits<size_t>::max()) {
        const auto &predicate = std::get<2>(path_predicates_[inst_view.predicate_index]);
        if (!pending_scc_write_counters && !state.masks.scalar_pending_unknown &&
            !state.masks.scalar_pending_sgprs.contains(predicate.sgpr) &&
            (predicate.kind == SccPredicate::Kind::Constant ||
             (parity && predicate.sgpr == parity->live) || predicate.sgpr == cyclic_guard ||
             relevant_predicates.contains({predicate.kind, predicate.sgpr, predicate.value}) ||
             (track_exec_source && path_exec_inputs_.contains(predicate.sgpr))))
          state.scc_predicate = predicate;
        else
          state.scc_predicate.reset();
      } else if (is_scc_conditional_branch(inst.mnemonic())) {
        const auto scc_edge_masks = [&](bool scc_true) -> std::optional<PathMasks> {
          PathMasks masks = state.masks;
          if (!track_masks)
            return masks;
          const bool pending_write = masks.scalar_pending_scc ||
                                     masks.pending_scc_write_counters;
          const auto known = pending_write ? std::optional<bool>{}
                             : masks.scc_is_exec ? masks.exec
                                                 : masks.scc;
          if (known && *known != scc_true)
            return std::nullopt;
          if (!pending_write && masks.scc_is_exec) {
            masks.exec = scc_true;
            masks.scc_is_exec = false;
          }
          if (!pending_write)
            masks.scc = scc_true;
          return masks;
        };
        const auto branch_delta = inst.branch_offset_bytes();
        const auto fallthrough_offset =
            inst_view.section_offset + static_cast<uint64_t>(inst.size());
        if (branch_delta) {
          const int64_t target =
              static_cast<int64_t>(fallthrough_offset) + static_cast<int64_t>(*branch_delta);
          if (target >= 0) {
            for (size_t successor : block_view.successors) {
              const BasicBlock *successor_block = cfg_views_[successor].block;
              if (successor_block == nullptr ||
                  successor_block->start_offset() != static_cast<uint64_t>(target) ||
                  !can_continue_from(successor, state.pending))
                continue;
              ScalarConstraints taken_constraints = state.constraints;
              const bool branch_takes_on_scc_true = inst.mnemonic() == "s_cbranch_scc1";
              const uint8_t pairs =
                  branch_takes_on_scc_true ? state.parity.on_true : state.parity.on_false;
              if (pairs && apply_scc_branch_constraint(taken_constraints, state.scc_predicate,
                                                       branch_takes_on_scc_true)) {
                auto masks = scc_edge_masks(branch_takes_on_scc_true);
                if (!masks)
                  continue;
                worklist.push_back({successor,
                                    0,
                                    state.pending,
                                    state.progress,
                                    std::move(taken_constraints),
                                    std::nullopt,
                                    std::nullopt,
                                    *masks,
                                    state.crossed_wait,
                                    {pairs, pairs, pairs}});
              }
            }
          }
        }

        ScalarConstraints fallthrough_constraints = state.constraints;
        const bool fallthrough_on_scc_true = inst.mnemonic() == "s_cbranch_scc0";
        const uint8_t pairs =
            fallthrough_on_scc_true ? state.parity.on_true : state.parity.on_false;
        if (pairs && apply_scc_branch_constraint(fallthrough_constraints, state.scc_predicate,
                                                 fallthrough_on_scc_true)) {
          auto masks = scc_edge_masks(fallthrough_on_scc_true);
          if (!masks)
            continue;
          bool found_fallthrough = false;
          for (size_t successor : block_view.successors) {
            const BasicBlock *successor_block = cfg_views_[successor].block;
            if (successor_block == nullptr || successor_block->start_offset() != fallthrough_offset)
              continue;
            found_fallthrough = true;
            if (can_continue_from(successor, state.pending)) {
              worklist.push_back({successor,
                                  0,
                                  state.pending,
                                  state.progress,
                                  fallthrough_constraints,
                                  std::nullopt,
                                  std::nullopt,
                                  *masks,
                                  state.crossed_wait,
                                  {pairs, pairs, pairs}});
            }
          }
          if (!found_fallthrough && state.inst_index + 1 < block_view.instructions.size()) {
            worklist.push_back({state.block_index,
                                state.inst_index + 1,
                                state.pending,
                                state.progress,
                                std::move(fallthrough_constraints),
                                std::nullopt,
                                std::nullopt,
                                *masks,
                                state.crossed_wait,
                                {pairs, pairs, pairs}});
          }
        }
        continue;
      } else if (inst_view.invalidates_scc) {
        state.scc_predicate.reset();
      }

      if (constraints_changed || previous_predicate != state.scc_predicate)
        state.constraint_id.reset();
      worklist.push_back({state.block_index, state.inst_index + 1, state.pending, state.progress,
                          std::move(state.constraints), state.scc_predicate, state.constraint_id,
                          state.masks, state.crossed_wait, state.parity});
    }

    return feasible_path_cache_[cache_key] =
               unknown_lane_witness ? std::nullopt : std::optional<bool>{false};
  }

  [[nodiscard]] static bool is_exec_masked_lane_access(const Instruction &consumer,
                                                       RegisterRef reg) {
    const auto mnemonic = consumer.mnemonic();
    const auto *issue = consumer.amdgpu_memory_issue_info();
    // Lane transfers and matrix instructions may inspect inactive VGPRs.
    // Scalar operands and unconditional memory accesses retain their checks.
    return is_exec_masked_def(reg) &&
           !(consumer.flags() & (IGNORES_EXEC | MATRIX_REGISTER_ACCESSES)) &&
           !mnemonic.starts_with("v_readlane") && !mnemonic.starts_with("v_readfirstlane") &&
           !mnemonic.starts_with("v_writelane") &&
           (is_vector_alu(consumer) || (issue && issue->exec_masked));
  }

  [[nodiscard]] bool access_requires_active_lane(const Instruction &consumer,
                                                 RegisterRef reg) const {
    return current_kernel_ && cfg_has_exec_guards_ && is_exec_masked_lane_access(consumer, reg);
  }

  // If this masked VGPR access executes, earlier FLAT loads under the same
  // EXEC also issue on the LDS FIFO. Count them backwards from a covering
  // wait, stopping before any possible reissue of the reported producer.
  bool consumer_exec_proves_flat_wait(const PendingEvent &event, const Instruction &consumer,
                                     rj_code_arch_t arch) const {
    if ((arch != ROCJITSU_CODE_ARCH_RDNA3 && arch != ROCJITSU_CODE_ARCH_RDNA3_5) ||
        !current_kernel_ || wavefront_size_ != 32 || cfg_has_unknown_successor_ ||
        !current_cfg_view_index_ || event.kind != WaitEventKind::FlatLoad ||
        event.counter != WaitCounterKind::Ds || event.section_offset == consumer.src_loc())
      return false;
    const auto &instructions = cfg_views_[*current_cfg_view_index_].instructions;
    auto pos = std::ranges::lower_bound(instructions, consumer.src_loc(), {},
                                        &CfgInstructionView::section_offset);
    if (pos == instructions.end() || pos->instruction != &consumer)
      return false;
    std::optional<uint32_t> remaining;
    for (unsigned distance = 0; pos != instructions.begin() && distance < 16; ++distance) {
      --pos;
      const Instruction &inst = *pos->instruction;
      if (pos->section_offset == event.section_offset || pos->invalidates_exec)
        break;
      if (inst.is_waitcnt()) {
        if (const auto wait = cfg_counter_wait(*current_cfg_view_index_, inst, event.counter, arch))
          remaining = std::min(remaining.value_or(*wait), *wait);
        if (remaining == 0)
          return true;
        continue;
      }
      if (inst.mnemonic() == "flat_load_b32" && remaining) {
        amdgpu::RegisterModifiers mods;
        inst.amdgpu_register_modifiers(mods);
        const auto allocated = [&](const Operand *op, unsigned width) {
          const auto ref = op ? op->to_register_ref() : std::nullopt;
          return ref && ref->cls == RegClass::VGPR && ref->width == width &&
                 ref->index + width <= current_kernel_->min_vgprs;
        };
        if (!mods.flat_scalar_address && allocated(mods.flat_address, 2) &&
            allocated(inst.dst_operand(0), 1) && --*remaining == 0)
          return true;
        continue;
      }
      // Limit the suffix to instructions with known mask and control effects.
      if (inst.mnemonic() != "s_clause" && inst.mnemonic() != "s_nop" &&
          inst.mnemonic() != "s_delay_alu" && inst.mnemonic() != "s_mov_b32")
        break;
    }
    return false;
  }

  [[nodiscard]] bool should_emit_cfg_diagnostic(const PendingEvent &event,
                                                const Instruction &consumer, RegisterRef reg,
                                                WaitcheckAccessKind access,
                                                uint64_t consumer_offset, rj_code_arch_t arch) {
    const auto lane_operand = guarded_lane_operand(event, consumer, reg, access, arch);
    const bool active = access_requires_active_lane(consumer, reg);
    const auto *memory_issue = consumer.amdgpu_memory_issue_info();
    if (reg.cls == RegClass::VGPR && event.counter == WaitCounterKind::Load &&
        wave64_guarded_loads_.contains(event.section_offset) &&
        is_exec_masked_lane_access(consumer, reg))
      return false;
    if (reg.cls == RegClass::VGPR && memory_issue && memory_issue->exec_masked &&
        consumer_exec_proves_flat_wait(event, consumer, arch))
      return false;
    // An SGPR relative-wait selector is consumed before this instruction's drain.
    const bool before_wait = relative_wait_operand(consumer, arch) != nullptr;
    const auto reaches = has_cfg_path_with_event_pending(event, consumer_offset, arch, before_wait,
                                                         active, 0, {}, lane_operand);
    if (reaches || !active || cfg_has_unknown_successor_ || !is_completion_counter(event.counter))
      return reaches.value_or(true);
    // A lane search may exceed its bound even if scalar branches force every
    // producer-to-consumer path through a full drain. Ignore lane facts and
    // partial waits in this independent, conservative disproof.
    return has_cfg_path_with_event_pending(event, consumer_offset, arch, false, false, 0, {}, {},
                                           /*scalar_full_drains_only=*/true)
        .value_or(true);
  }

  [[nodiscard]] bool event_can_reach_wait(const PendingEvent &event, uint64_t wait_offset,
                                          rj_code_arch_t arch) {
    return has_cfg_path_with_event_pending(event, wait_offset, arch, true).value_or(true);
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

    for (int i = 0; i < inst.num_dst_operands(); ++i) {
      const auto *operand = inst.dst_operand(i);
      if (operand && operand->to_special_reg_class() == ref.cls)
        return true;
    }
    const std::string_view mnemonic = inst.mnemonic();
    return mnemonic == "s_barrier_signal_isfirst" || starts_with(mnemonic, "s_cmp_") ||
           starts_with(mnemonic, "s_bitcmp") || starts_with(mnemonic, "s_addk_co_") ||
           starts_with(mnemonic, "s_add_co_") || starts_with(mnemonic, "s_sub_co_") ||
           starts_with(mnemonic, "s_addc_") || starts_with(mnemonic, "s_subb_");
  }

  // Loop backedges may lose an induction variable's precise value after the
  // load has already drained. Recheck MOVREL on the incoming edges on which
  // this specific load is still pending. Unknown offsets retain all aliases.
  bool relative_load_disjoint_on_incoming_edges(const PendingEvent &event,
                                                const Instruction &consumer, rj_code_arch_t arch) {
    if (!current_cfg_view_index_ || cfg_outputs_.empty() || current_function_entry_.has_value() ||
        !relative_entry_proven_)
      return false;
    const auto &block = cfg_views_[*current_cfg_view_index_];
    if (block.predecessors.size() < 2 || block.predecessors.size() > 8 ||
        block.instructions.size() > 256 ||
        (event.section_offset >= block.block->start_offset() &&
         event.section_offset < block.block->end_offset()))
      return false;
    const auto position =
        std::ranges::find(block.instructions, &consumer, &CfgInstructionView::instruction);
    if (position == block.instructions.end())
      return false;

    bool reached = false;
    for (const auto predecessor : block.predecessors) {
      const auto &output = cfg_outputs_[predecessor];
      const auto &pending = output.pending[counter_index(event.counter)];
      if (!std::ranges::any_of(pending, [&](const PendingEvent &candidate) {
            return same_event_identity(candidate, event);
          }))
        continue;
      reached = true;
      PendingState incoming = output;
      for (auto it = block.instructions.begin(); it != position; ++it)
        analyze_instruction(incoming, *it->instruction, event.section_name, it->section_offset,
                            it->section_offset, arch, false);
      if (incoming.m0.count == 0)
        return false;
      const auto operands = inst_def_use_for_waitcheck(consumer, incoming.vgpr_msb, arch,
                                                       wavefront_size_, incoming.m0);
      if ((event.check_uses &&
           first_dependency_intersection(event, consumer, operands, WaitcheckAccessKind::Use, arch,
                                         incoming.m0)) ||
          (event.check_defs &&
           first_dependency_intersection(event, consumer, operands, WaitcheckAccessKind::Def, arch,
                                         incoming.m0)))
        return false;
    }
    return reached;
  }

  // Track one arbitrary issuing lane. A bitwise full-width operation applies
  // to either half of a wave64. Any partial SGPR write invalidates its pair.
  struct LaneFacts {
    static constexpr unsigned exec = 128;
    std::array<std::optional<bool>, exec + 1> bits{};
    unsigned younger_flat = 0; // Guaranteed later GFX11 DS FIFO credits.
    bool operator==(const LaneFacts &) const = default;
    void join(const LaneFacts &other) {
      younger_flat = std::min(younger_flat, other.younger_flat);
      for (size_t i = 0; i != bits.size(); ++i)
        if (bits[i] != other.bits[i])
          bits[i].reset();
    }
  };

  std::optional<unsigned> lane_mask_register(const Operand &operand) const {
    const auto ref = operand.to_register_ref();
    if (operand.size_bits() != static_cast<int>(wavefront_size_))
      return std::nullopt;
    if (low_selector(&operand, amdgpu::kExecSelectorFirst))
      return LaneFacts::exec;
    if (!ref)
      return std::nullopt;
    if (ref->cls != RegClass::SGPR || ref->width != wavefront_size_ / 32 ||
        (ref->width == 2 && (ref->index & 1)) || !current_kernel_ ||
        ref->index + ref->width > current_kernel_->min_sgprs || ref->index >= 128)
      return std::nullopt;
    return ref->index;
  }

  void update_lane_facts(LaneFacts &state, const Instruction &inst, rj_code_arch_t arch) {
    const auto input = [&](unsigned n) -> std::optional<bool> {
      const auto &op = *inst.src_operand(n);
      if (const auto pair = lane_mask_register(op))
        return state.bits[*pair];
      if (const auto value = op.const_value()) {
        if (*value == 0)
          return false;
        if (static_cast<int64_t>(*value) == -1)
          return true;
      }
      return std::nullopt;
    };
    const auto neg = [](std::optional<bool> a) -> std::optional<bool> {
      return a ? std::optional{!*a} : std::nullopt;
    };
    const auto both = [](std::optional<bool> a, std::optional<bool> b) -> std::optional<bool> {
      if (a == false || b == false)
        return false;
      return a && b ? std::optional{true} : std::nullopt;
    };
    const auto old_exec = state.bits[LaneFacts::exec];
    const auto a = inst.num_src_operands() ? input(0) : std::nullopt;
    const auto b = inst.num_src_operands() > 1 ? input(1) : std::nullopt;
    const auto du = inst_def_use_for_waitcheck(inst, {}, arch, wavefront_size_);
    du.defs.for_each([&](const RegisterRef &ref) {
      if (ref.cls == RegClass::SGPR) {
        for (unsigned i = ref.index; i < ref.index + ref.width && i < 128; ++i)
          state.bits[wavefront_size_ == 32 ? i : i & ~1u].reset();
      } else if (ref.cls == RegClass::EXEC) {
        state.bits[LaneFacts::exec].reset();
      }
    });
    if (writes_exec_mask(inst) || (inst.num_dst_operands() &&
                                   (low_selector(inst.dst_operand(0), amdgpu::kExecSelectorFirst) ||
                                    low_selector(inst.dst_operand(0), amdgpu::kExecSelectorLast))))
      state.bits[LaneFacts::exec].reset();
    if (inst.mnemonic().starts_with("v_cmpx") && old_exec == false)
      state.bits[LaneFacts::exec] = false;
    if (!inst.num_dst_operands())
      return;
    const auto name = inst.mnemonic();
    const auto dst = lane_mask_register(*inst.dst_operand(0));
    const auto op = name.substr(0, name.rfind('_'));

    if (op == "s_and_saveexec" && dst) {
      state.bits[*dst] = old_exec;
      state.bits[LaneFacts::exec] = both(old_exec, a);
    } else if (op == "s_and_not1_saveexec" && dst) {
      state.bits[*dst] = old_exec;
      state.bits[LaneFacts::exec] = both(a, neg(old_exec));
    } else if (dst) {
      if (op == "s_mov")
        state.bits[*dst] = a;
      else if (op == "s_and")
        state.bits[*dst] = both(a, b);
      else if (op == "s_or")
        state.bits[*dst] = neg(both(neg(a), neg(b)));
      else if (op == "s_orn2" || op == "s_or_not1")
        state.bits[*dst] = neg(both(neg(a), b));
      else if (op == "s_andn2" || op == "s_and_not1")
        state.bits[*dst] = both(a, neg(b));
      else if (op == "s_xor" && a && b)
        state.bits[*dst] = *a != *b;
    }
  }

  struct LanePosition {
    size_t block;
    size_t index;
    auto operator<=>(const LanePosition &) const = default;
  };

  // Walk until a covering LGKM wait or program end. At joins discard
  // conflicting lane facts and take the lower queue age. A loop that repeats
  // the capture or producer, a call, or an unresolved exit invalidates the proof.
  bool excluded_copy_lanes(LanePosition start, LanePosition stop, LaneFacts initial,
                           std::optional<LanePosition> capture, unsigned saved,
                           const std::set<size_t> &reachable, rj_code_arch_t arch,
                           unsigned &budget) {
    std::map<LanePosition, LaneFacts> inputs;
    std::vector<std::pair<LanePosition, LaneFacts>> todo{{start, initial}};
    bool reached = false;
    const bool track_fifo = !capture && arch == ROCJITSU_CODE_ARCH_RDNA3;
    while (!todo.empty()) {
      auto [pos, state] = todo.back();
      todo.pop_back();
      auto [it, first] = inputs.try_emplace(pos, state);
      if (!first) {
        state.join(it->second);
        if (state == it->second)
          continue;
        it->second = state;
      }
      const auto &block = cfg_views_[pos.block];
      for (size_t i = pos.index; i != block.instructions.size(); ++i) {
        if (!budget)
          return false;
        --budget;
        const auto &item = block.instructions[i];
        const auto &inst = *item.instruction;
        if (pos.block == stop.block && i == stop.index) {
          if (state.bits[LaneFacts::exec] != false)
            return false;
          reached = true;
          if (capture)
            break;
        }
        if (capture) {
          // A new invocation, call or overwrite would invalidate the saved bit.
          if (pos.block == capture->block && i == capture->index)
            return false;
          const auto du = inst_def_use_for_waitcheck(inst, {}, arch, wavefront_size_);
          if (du.defs.intersects({RegClass::SGPR, static_cast<uint16_t>(saved),
                                  static_cast<uint8_t>(wavefront_size_ / 32)}))
            return false;
        } else {
          const auto wait = cfg_counter_wait(pos.block, inst, WaitCounterKind::Ds, arch);
          if (is_program_end(inst.mnemonic()) || (wait && *wait <= state.younger_flat))
            break;
          // A lane that issued the original FLAT proves a younger ordinary
          // FLAT issues even if other lanes or the VM queue are idle.
          const auto op = inst.mnemonic();
          if (track_fifo && !(pos.block == start.block && i == start.index) &&
              state.bits[LaneFacts::exec] == true && op == "flat_load_b32" &&
              inst.num_dst_operands() == 1 && inst.num_src_operands() && state.younger_flat < 15) {
            const auto allocated_vgpr = [&](const Operand &operand, unsigned width) {
              const auto ref = operand.to_register_ref();
              return ref && ref->cls == RegClass::VGPR && ref->width == width &&
                     ref->index + width <= current_kernel_->min_vgprs;
            };
            if (allocated_vgpr(*inst.dst_operand(0), 1) && allocated_vgpr(*inst.src_operand(0), 2))
              ++state.younger_flat;
          }
        }
        if ((inst.flags() & INDIRECT_CALL) || inst.mnemonic().starts_with("s_call") ||
            inst.mnemonic().starts_with("s_setpc") || inst.mnemonic().starts_with("s_swappc") ||
            inst.mnemonic().starts_with("s_rfe"))
          return false;
        update_lane_facts(state, inst, arch);
        if (i + 1 != block.instructions.size())
          continue;
        if (block.successors.empty())
          return false;
        for (size_t next : block.successors) {
          // Before the producer only paths reaching it matter.
          if (capture && !reachable.contains(next))
            continue;
          if ((capture && next == capture->block) || (!capture && next == start.block))
            return false;
          todo.push_back({{next, 0}, state});
        }
      }
    }
    return reached || !capture;
  }

  bool scratch_access_excludes_flat_result(const PendingEvent &event, const Instruction &consumer,
                                           rj_code_arch_t arch) {
    if (!((arch == ROCJITSU_CODE_ARCH_CDNA4 && wavefront_size_ == 64) ||
          (arch == ROCJITSU_CODE_ARCH_RDNA3 && wavefront_size_ == 32)) ||
        !current_kernel_ || !current_cfg_view_index_ || cfg_has_unknown_successor_ ||
        event.kind != WaitEventKind::FlatLoad || event.counter != WaitCounterKind::Ds ||
        (!consumer.mnemonic().starts_with("scratch_store") &&
         !(arch == ROCJITSU_CODE_ARCH_RDNA3 && consumer.mnemonic().starts_with("scratch_load"))))
      return false;
    const auto key = std::tuple{event.section_offset, *current_cfg_view_index_, consumer.src_loc()};
    // The current CFG block can contain multiple scratch consumers.
    auto [cached, fresh] = copy_lane_cache_.try_emplace(key, false);
    if (!fresh)
      return cached->second;
    const auto consumer_block = *current_cfg_view_index_;
    const auto producers = cfg_block_indices_containing(event.section_offset);
    if (producers.empty() || producers.size() > 16 || consumer.src_loc() <= event.section_offset ||
        consumer.src_loc() - event.section_offset > 2048)
      return false;
    const auto position = [&](size_t b, uint64_t offset) {
      const auto &instructions = cfg_views_[b].instructions;
      return LanePosition{
          b, static_cast<size_t>(std::ranges::lower_bound(instructions, offset, {},
                                                          &CfgInstructionView::section_offset) -
                                 instructions.begin())};
    };
    const size_t pb = producers.front();
    const auto source = position(pb, event.section_offset);
    const auto sink = position(consumer_block, consumer.src_loc());
    unsigned budget = 4096;
    if (arch == ROCJITSU_CODE_ARCH_RDNA3 && consumer.mnemonic().starts_with("scratch_load")) {
      LaneFacts issued;
      issued.bits[LaneFacts::exec] = true;
      return cached->second = std::ranges::all_of(producers, [&](size_t b) {
               return excluded_copy_lanes(position(b, event.section_offset), sink, issued,
                                          std::nullopt, 0, {}, arch, budget);
             });
    }
    if (producers.size() != 1 || cfg_block_indices_containing(consumer.src_loc()).size() != 1)
      return false;
    // Look for a dominating SAVEEXEC close to this producer. The pre-issue
    // proof excludes lanes that were inactive at that save. From issue onward
    // the saved bit is therefore one for every lane that could still be pending.
    std::vector<LanePosition> captures;
    std::set<size_t> reachable;
    std::vector<size_t> work{pb};
    while (!work.empty()) {
      const size_t b = work.back();
      work.pop_back();
      if (!reachable.insert(b).second)
        continue;
      if (reachable.size() > 64)
        return false;
      const auto &block = cfg_views_[b];
      for (size_t i = 0; i < block.instructions.size(); ++i) {
        const auto &item = block.instructions[i];
        const auto &inst = *item.instruction;
        const bool captures_exec =
            inst.mnemonic() ==
                (wavefront_size_ == 64 ? "s_and_saveexec_b64" : "s_and_saveexec_b32") ||
            (inst.mnemonic() == (wavefront_size_ == 64 ? "s_mov_b64" : "s_mov_b32") &&
             lane_mask_register(*inst.src_operand(0)) == LaneFacts::exec);
        if (captures_exec && item.section_offset < event.section_offset &&
            event.section_offset - item.section_offset <= 512)
          captures.push_back({b, i});
      }
      for (size_t pred : block.predecessors) {
        const auto address = cfg_views_[pred].block->end_offset();
        if (address <= event.section_offset && event.section_offset - address <= 512)
          work.push_back(pred);
      }
    }
    std::set<const Instruction *> tested;
    for (auto c : captures) {
      const auto &save = *cfg_views_[c.block].instructions[c.index].instruction;
      if (!tested.insert(&save).second)
        continue;
      const auto pair = lane_mask_register(*save.dst_operand(0));
      if (!pair || *pair == LaneFacts::exec)
        continue;
      // Build the complete inverse slice from this specific producer to its
      // dominating capture. A detour that leaves the window and returns must
      // either be included here or make the proof fail.
      reachable.clear();
      std::vector<size_t> pending{pb}, origins;
      bool valid = true;
      while (!pending.empty()) {
        const auto b = pending.back();
        pending.pop_back();
        if (!reachable.insert(b).second)
          continue;
        // A conditional call can create several analysis contexts for the
        // same native continuation. Every predecessor path must cross this
        // physical capture before it can reach the issuing instruction.
        if (cfg_views_[b].block == cfg_views_[c.block].block) {
          origins.push_back(b);
          continue;
        }
        if (reachable.size() > 32 || cfg_views_[b].predecessors.empty()) {
          valid = false;
          break;
        }
        for (auto pred : cfg_views_[b].predecessors)
          pending.push_back(pred);
      }
      if (!valid || origins.empty())
        continue;
      LaneFacts before, after;
      before.bits[LaneFacts::exec] = before.bits[*pair] = false;
      after.bits[LaneFacts::exec] = after.bits[*pair] = true;
      // The capture saved an inactive lane. Begin at its successor instruction.
      // A capture at a block terminator would require starting at its edges.
      if (c.index + 1 >= cfg_views_[c.block].instructions.size())
        continue;
      if (std::ranges::all_of(origins,
                              [&](size_t origin) {
                                return excluded_copy_lanes({origin, c.index + 1}, source, before,
                                                           LanePosition{origin, c.index}, *pair,
                                                           reachable, arch, budget);
                              }) &&
          excluded_copy_lanes(source, sink, after, std::nullopt, *pair, reachable, arch, budget))
        return cached->second = true;
    }
    return false;
  }

  template <typename Visitor>
  void visit_dependencies(const PendingState &state, const Instruction &inst, const InstDefUse &du,
                          std::span<const ClassifiedEvent> current_events, rj_code_arch_t arch,
                          Visitor &&visit) {
    const bool single_access = arch == ROCJITSU_CODE_ARCH_CDNA5 && is_vmem_atomic(inst.mnemonic());
    for (size_t counter_idx = 0; counter_idx < state.pending.size(); ++counter_idx) {
      const auto &events = state.pending[counter_idx];
      for (size_t i = 0; i < events.size(); ++i) {
        const PendingEvent &event = events[i];
        // VALU-to-VALU hazards are interlocked in expert scheduling mode.
        if ((event.counter == WaitCounterKind::VaVdst &&
             ((inst.flags() & VALU) || dsdir_wait_va_vdst(inst))) ||
            !checks_scalar_source(event.kind, inst))
          continue;
        std::optional<RegisterRef> reg;
        WaitcheckAccessKind access = WaitcheckAccessKind::Use;
        // An earlier value or a later synchronous write does not make a pending
        // physical destination ready. Keep both RAW and WAW dependencies until
        // an explicit wait or a qualified ordering rule retires the event.
        if (event.check_uses) {
          reg = first_dependency_intersection(event, inst, du, WaitcheckAccessKind::Use, arch,
                                              state.m0);
        }
        if (!reg && event.check_defs) {
          reg = first_dependency_intersection(event, inst, du, WaitcheckAccessKind::Def, arch,
                                              state.m0);
          access = WaitcheckAccessKind::Def;
        }
        if (reg && guarded_lane_operand(event, inst, *reg, access, arch)) {
          // A tuple can overlap a routed source, an ordinary source and the
          // destination in different lanes. Proving the first one disjoint
          // does not discharge the other physical operands of this event.
          const auto count = dependency_required_count(state, event, arch).value();
          event.regs.for_each([&](RegisterRef part) {
            if (part.cls != RegClass::VGPR)
              return;
            for (const auto kind : {WaitcheckAccessKind::Use, WaitcheckAccessKind::Def}) {
              const bool read = kind == WaitcheckAccessKind::Use;
              if (read ? !event.check_uses || !du.uses.contains(part)
                       : !event.check_defs || !du.defs.contains(part) ||
                             ordered_waw(event, current_events, arch))
                continue;
              if (event.partial_reg == part &&
                  !(event.partial_reg_mask & instruction_register_mask(inst, part, kind)))
                continue;
              if (read && inst.mnemonic().starts_with("v_writelane") && du.defs.contains(part))
                continue;
              visit(event, part, kind, count);
            }
          });
          continue;
        }
        if (reg && reg->cls == RegClass::VGPR && state.m0.count == 0 &&
            event.counter == WaitCounterKind::Load && inst.mnemonic().starts_with("v_movrel") &&
            relative_load_disjoint_on_incoming_edges(event, inst, arch))
          continue;
        if (!reg && event.special_reg) {
          if (event.check_uses && instruction_uses_special(inst, *event.special_reg)) {
            reg = event.special_reg;
            access = WaitcheckAccessKind::Use;
          } else if (event.check_defs && (du.defs.contains(*event.special_reg) ||
                                          instruction_defines_special(inst, *event.special_reg))) {
            reg = event.special_reg;
            access = WaitcheckAccessKind::Def;
          }
        }
        // VMEM atomics must execute once across a page fault. A run of RMWs
        // is replay-safe after its first drain; an intervening non-atomic
        // translation can introduce a new replay point. Source-level volatile
        // loads and atomic stores have no unique ordinary load/store encoding.
        if (!reg && single_access && event.counter == WaitCounterKind::X &&
            !is_vmem_atomic(event.instruction)) {
          reg = RegisterRef{RegClass::PC, 0, 1};
          access = WaitcheckAccessKind::MemoryOrder;
        }
        if (!reg && event.check_memory_order) {
          const bool is_consumer = event.counter == WaitCounterKind::Async
                                       ? is_async_ordering_consumer(event, inst.mnemonic())
                                   : event.counter == WaitCounterKind::Tensor
                                       ? is_tensor_ordering_consumer(event, inst.mnemonic())
                                       : is_memory_ordering_consumer(inst.mnemonic());
          if (is_consumer) {
            reg = RegisterRef{RegClass::PC, 0, 1};
            access = WaitcheckAccessKind::MemoryOrder;
          }
        }
        if (!reg && event.check_program_end && is_program_end(inst.mnemonic())) {
          reg = RegisterRef{RegClass::PC, 0, 1};
          access = WaitcheckAccessKind::ProgramEnd;
        }
        if (!reg)
          continue;
        // Writelane's destination input preserves the other lanes for whole-
        // register dataflow; it is not a hardware read. A pending definition
        // overlapping this destination conflicts with the selected lane write.
        if (access == WaitcheckAccessKind::Use && event.check_defs &&
            inst.mnemonic() == "v_writelane_b32" && du.defs.contains(*reg))
          access = WaitcheckAccessKind::Def;
        // The HSA argument segment is immutable throughout dispatch. Two loads
        // writing identical bytes into the same dwords can finish in either order.
        // Both results remain outstanding for reads and other overwrites.
        if (access == WaitcheckAccessKind::Def && event.counter == WaitCounterKind::Ds &&
            event.kind == WaitEventKind::Smem && repeats_kernarg_read(event.section_offset, inst))
          continue;
        if (access == WaitcheckAccessKind::Def &&
            !(event.counter == WaitCounterKind::Load &&
              inst.mnemonic().find("atomic") != std::string_view::npos) &&
            ordered_waw(event, current_events, arch))
          continue;

        const auto required_count = dependency_required_count(state, event, arch).value();
        visit(event, *reg, access, required_count);
      }
    }

    if (!writes_exec_mask(inst))
      return;

    for (size_t counter_idx = 0; counter_idx < state.pending.size(); ++counter_idx) {
      const auto &events = state.pending[counter_idx];
      for (size_t i = 0; i < events.size(); ++i) {
        const PendingEvent &event = events[i];
        if (!event.check_exec_defs || !checks_scalar_source(event.kind, inst))
          continue;
        const auto required_count = dependency_required_count(state, event, arch).value();
        visit(event, RegisterRef{RegClass::EXEC, 0, 1}, WaitcheckAccessKind::Def, required_count);
      }
    }
  }

  void check_dependencies(const PendingState &state, const Instruction &inst, const InstDefUse &du,
                          std::span<const ClassifiedEvent> current_events, uint64_t section_offset,
                          uint64_t file_offset, rj_code_arch_t arch) {
    visit_dependencies(state, inst, du, current_events, arch,
                       [&](const PendingEvent &event, RegisterRef reg, WaitcheckAccessKind access,
                           uint32_t required_count) {
                         // The bounded walk uses actual emitted waits. Parity evaluates
                         // counterfactual executions with the measured wait removed.
                         if (reg.cls == RegClass::VGPR &&
                             scratch_access_excludes_flat_result(event, inst, arch))
                           return;
                         emit_diagnostic(inst, event, reg, access, required_count, section_offset,
                                         file_offset, arch);
                       });
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
      return vector_registers(du.uses);
    case TrackedRegisterSource::VectorDefs:
      return vector_registers(du.defs);
    case TrackedRegisterSource::StoreDataUses:
      return store_data_registers(inst, state, arch);
    }
    return {};
  }

  [[nodiscard]] static RegisterSet vector_registers(const RegisterSet &all) {
    RegisterSet regs;
    all.for_each([&](RegisterRef ref) {
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

  [[nodiscard]] static bool writes_exec_mask(const Instruction &inst) {
    for (int i = 0; i < inst.num_dst_operands(); ++i) {
      const Operand *op = inst.dst_operand(i);
      if (!op)
        continue;
      const auto ref = op->to_register_ref();
      if ((ref && ref->cls == RegClass::EXEC) ||
          (!ref && op->to_special_reg_class() == RegClass::EXEC))
        return true;
    }

    const std::string_view mnemonic = inst.mnemonic();
    return starts_with(mnemonic, "v_cmpx_") ||
           mnemonic.find("_saveexec_") != std::string_view::npos ||
           mnemonic.find("_wrexec_") != std::string_view::npos;
  }

  [[nodiscard]] static uint16_t sgpr_pair(RegisterRef ref) {
    return static_cast<uint16_t>((ref.index >> 1u) & 0x3fu);
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
    const auto ref = op->to_register_ref();
    return (ref && ref->cls == RegClass::VCC) || op->to_special_reg_class() == RegClass::VCC ||
           (op->has_register_selector() && (op->encoding_value() == amdgpu::kVccSelectorFirst ||
                                            op->encoding_value() == amdgpu::kVccSelectorLast));
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
    RegisterSet uses;
    inst.implicit_uses(uses);
    return uses.contains({RegClass::VCC, 0, 1}) || instruction_uses_vcc_explicit(inst) ||
           instruction_uses_vcc_implicit(inst);
  }

  [[nodiscard]] static bool instruction_defines_vcc(const Instruction &inst, const InstDefUse &du) {
    return du.defs.contains({RegClass::VCC, 0, 1}) || instruction_defines_vcc_explicit(inst) ||
           instruction_defines_vcc_implicit(inst, du);
  }

  static void set_sgpr_hazard(SgprHazardState &state, RegisterRef ref, bool is_valu,
                              const SgprHazardProducer &producer) {
    if (ref.cls != RegClass::SGPR || ref.index >= 128)
      return;
    if (!state.tracked_pairs.test(sgpr_pair(ref)))
      return;

    if (is_valu) {
      state.valu_hazards.set(ref.index);
      state.valu_producers.insert_or_assign(ref.index, producer);
    } else {
      state.salu_hazards.set(ref.index);
      state.salu_producers.insert_or_assign(ref.index, producer);
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

  [[nodiscard]] static std::optional<uint16_t>
  first_sgpr_hazard_lane(const std::bitset<128> &hazards) {
    for (uint16_t lane = 0; lane < hazards.size(); ++lane) {
      if (hazards.test(lane))
        return lane;
    }
    return std::nullopt;
  }

  // Branch relaxation emits a getpc-relative SETPC after LLVM's SGPR hazard pass.
  // A scalar target pair does not read VCC. With only program exits as targets,
  // no instruction in a caller or callee can subsequently observe that VCC write.
  static bool branches_directly_to_program_end(const Instruction &inst) {
    const BasicBlock *block = inst.parent();
    if (!block || inst.mnemonic() != "s_setpc_b64" || inst.num_src_operands() != 1 ||
        block->terminator() != &inst || !block->call_edges().empty() ||
        block->successor_issue() != BasicBlock::SuccessorIssue::IndirectControlFlow)
      return false;
    const auto pair = inst.src_operand(0)->to_register_ref();
    if (!pair || pair->cls != RegClass::SGPR || pair->width != 2)
      return false;
    const auto &fixups = block->static_indirect_call_fixups();
    return !fixups.empty() && std::ranges::all_of(fixups, [&](const IndirectCallFixup &fixup) {
      if (fixup.source_incomplete || fixup.source_is_call || fixup.source_call_sreg != pair->index)
        return false;
      return std::ranges::any_of(block->successors(), [&](const BasicBlock *target) {
        return target->start_offset() == fixup.source_target_offset && target->terminator() &&
               target->terminator()->src_loc() == target->start_offset() &&
               target->terminator()->mnemonic() == "s_endpgm";
      });
    });
  }

  void check_sgpr_control_transfer_hazards(const SgprHazardState &state, const Instruction &inst,
                                           uint64_t section_offset, uint64_t file_offset) {
    // One all-zero depctr field repairs every register in that hazard class.
    // Emit its lowest-numbered representative instead of one diagnostic per
    // outstanding lane; VCC represents the field only when no SGPR lane does.
    const bool check_vcc = state.vcc_hazard && !branches_directly_to_program_end(inst);
    if (const auto lane = first_sgpr_hazard_lane(state.salu_hazards)) {
      const auto producer = state.salu_producers.find(*lane);
      emit_sgpr_hazard_diagnostic(
          inst, producer == state.salu_producers.end() ? nullptr : &producer->second,
          RegisterRef{RegClass::SGPR, *lane, 1}, "depctr_sa_sdst", section_offset, file_offset,
          WaitcheckAccessKind::ControlTransfer);
    } else if (check_vcc && (state.vcc_hazard & kSgprHazardSalu) != 0) {
      emit_sgpr_hazard_diagnostic(
          inst, state.salu_vcc_producer ? &*state.salu_vcc_producer : nullptr,
          RegisterRef{RegClass::VCC, 0, 1}, "depctr_sa_sdst", section_offset, file_offset,
          WaitcheckAccessKind::ControlTransfer);
    }

    if (const auto lane = first_sgpr_hazard_lane(state.valu_hazards)) {
      const auto producer = state.valu_producers.find(*lane);
      emit_sgpr_hazard_diagnostic(
          inst, producer == state.valu_producers.end() ? nullptr : &producer->second,
          RegisterRef{RegClass::SGPR, *lane, 1}, "depctr_va_sdst", section_offset, file_offset,
          WaitcheckAccessKind::ControlTransfer);
    }
    if (check_vcc && (state.vcc_hazard & kSgprHazardValu) != 0) {
      emit_sgpr_hazard_diagnostic(inst,
                                  state.valu_vcc_producer ? &*state.valu_vcc_producer : nullptr,
                                  RegisterRef{RegClass::VCC, 0, 1}, "depctr_va_vcc", section_offset,
                                  file_offset, WaitcheckAccessKind::ControlTransfer);
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

      if (is_valu && state.salu_hazards.test(ref.index)) {
        const auto producer = state.salu_producers.find(ref.index);
        emit_sgpr_hazard_diagnostic(
            inst, producer == state.salu_producers.end() ? nullptr : &producer->second, ref,
            "depctr_sa_sdst", section_offset, file_offset);
      }
      if (is_valu && state.valu_hazards.test(ref.index)) {
        const auto producer = state.valu_producers.find(ref.index);
        emit_sgpr_hazard_diagnostic(
            inst, producer == state.valu_producers.end() ? nullptr : &producer->second, ref,
            "depctr_va_sdst", section_offset, file_offset);
      }
    });
  }

  void check_vcc_hazard_use(const SgprHazardState &state, const Instruction &inst, bool is_valu,
                            uint64_t section_offset, uint64_t file_offset) {
    if (!state.tracked_vcc)
      return;

    RegisterRef vcc{RegClass::VCC, 0, 1};
    if (is_valu && (state.vcc_hazard & kSgprHazardSalu))
      emit_sgpr_hazard_diagnostic(inst,
                                  state.salu_vcc_producer ? &*state.salu_vcc_producer : nullptr,
                                  vcc, "depctr_sa_sdst", section_offset, file_offset);
    if (is_valu && (state.vcc_hazard & kSgprHazardValu))
      emit_sgpr_hazard_diagnostic(inst,
                                  state.valu_vcc_producer ? &*state.valu_vcc_producer : nullptr,
                                  vcc, "depctr_va_vcc", section_offset, file_offset);
  }

  void update_sgpr_hazards(SgprHazardState &state, const Instruction &inst, const InstDefUse &du,
                           const std::string &section_name, uint64_t section_offset,
                           uint64_t file_offset, bool emit_diagnostics) {
    if (is_sgpr_hazard_control_transfer(inst)) {
      if (emit_diagnostics)
        check_sgpr_control_transfer_hazards(state, inst, section_offset, file_offset);
      // Calls, returns, and indirect branches do not carry the hardware hazard
      // chain across the transfer. Report every required field at the source,
      // then begin the target context without stale producer state. CFG edge
      // construction separately marks callees and return continuations as
      // conservatively tracked and injects a call's link-register write only
      // into that call's matching continuation.
      clear_all_sgpr_hazards(state);
      return;
    }

    if (instruction_waits_for_valu_sgpr_writes(inst))
      clear_valu_sgpr_hazards(state);

    // SOPP VCC branches read the VALU result before either successor issues.
    if (mask_branch_class(inst.mnemonic()) == RegClass::VCC) {
      clear_valu_vcc_hazard(state);
      return;
    }

    const bool is_salu = is_scalar_alu(inst);
    const bool is_valu = is_vector_alu(inst);
    if (!is_salu && !is_valu)
      return;

    const bool uses_vcc = instruction_uses_vcc(inst);
    const bool defines_vcc = instruction_defines_vcc(inst, du);
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

  // Both VOPD formats concatenate the X and Y source lists. Use the
  // physical destination to select the slot, including when both opcodes match.
  static std::optional<int> vopd_source_begin(const Instruction &inst, RegisterRef dst,
                                              std::string_view op, int arity) {
    constexpr unsigned vopd_xy = 0x32, vopd3 = 0xcf;
    const bool extended = inst.encoding_id() == vopd3;
    // The extended shift has the same two inputs in either slot. Other
    // extended opcodes can have a third input, e.g. CNDMASK's explicit mask.
    if ((extended ? op != "v_dual_lshlrev_b32" : inst.encoding_id() != vopd_xy) ||
        (inst.flags() & INVALID_VOPD_OPERANDS) || inst.num_dst_operands() != 2 ||
        inst.num_src_operands() <= arity)
      return std::nullopt;
    const auto name = inst.mnemonic();
    const auto sep = name.find(" :: ");
    if (sep != std::string_view::npos) {
      // Require zero NEG bits for the selected slot of an extended encoding.
      const auto modifiers = extended ? inst.raw_encoding()[1] : 0u;
      if (name.substr(0, sep) == op && inst.dst_operand(0)->to_register_ref() == dst &&
          !(modifiers & (7u << 9)))
        return 0;
      if (name.substr(sep + 4) == op && inst.dst_operand(1)->to_register_ref() == dst &&
          !(modifiers & (7u << 12)))
        return inst.num_src_operands() - arity;
    }
    return std::nullopt;
  }

  static std::optional<WaitCounterKind> rdna3_scalar_aperture(const Instruction &inst) {
    if (inst.mnemonic() != "s_mov_b64" || inst.num_src_operands() != 1 ||
        !inst.src_operand(0)->has_register_selector())
      return std::nullopt;
    const unsigned selector = inst.src_operand(0)->encoding_value();
    if (selector == 237) // SRC_PRIVATE_BASE
      return WaitCounterKind::Load;
    if (selector == 235) // SRC_SHARED_BASE
      return WaitCounterKind::Ds;
    return std::nullopt;
  }

  // A scalar high word can outlive a local vector copy by many blocks. Require
  // every reaching definition to select the same aperture. EXEC cannot alter an
  // SGPR copy. Keep the lane-sensitive vector part inside the local proof.
  std::optional<WaitCounterKind> rdna3_saved_aperture(const BasicBlock *start,
                                                      const Instruction *before, uint16_t sgpr,
                                                      rj_code_arch_t arch) {
    if (!rdna3_has_aperture_)
      return std::nullopt;
    using BlockWord = std::pair<const BasicBlock *, uint16_t>;
    std::vector<BlockWord> work{{start, sgpr}};
    std::set<BlockWord> seen;
    std::optional<WaitCounterKind> result;
    unsigned budget = 16384;
    while (!work.empty()) {
      auto [block, word] = work.back();
      work.pop_back();
      const auto &instructions = block->instructions();
      auto pos = before ? decltype(instructions.begin())(before) : instructions.end();
      if (!std::exchange(before, nullptr)) {
        const BlockWord key{block, word};
        if (!seen.insert(key).second)
          continue;
        if (auto it = rdna3_scalar_apertures_.find(key); it != rdna3_scalar_apertures_.end()) {
          if (result && *result != it->second)
            return std::nullopt;
          result = it->second;
          continue;
        }
      }
      bool defined = false;
      while (pos != instructions.begin()) {
        if (!budget--)
          return std::nullopt;
        const Instruction &inst = *--pos;
        const auto name = inst.mnemonic();
        if ((inst.flags() & INDIRECT_CALL) || name.starts_with("s_call") ||
            name.starts_with("s_swappc") || name.starts_with("s_setpc") ||
            name.starts_with("s_setreg") || name.find("movrel") != std::string_view::npos)
          return std::nullopt;
        const RegisterRef wanted{RegClass::SGPR, word, 1};
        if (!inst_def_use_for_waitcheck(inst, {}, arch, wavefront_size_).defs.intersects(wanted))
          continue;
        if (inst.num_dst_operands() != 1 || inst.num_src_operands() != 1)
          return std::nullopt;
        const auto dst = inst.dst_operand(0)->to_register_ref();
        if (!dst || dst->cls != RegClass::SGPR ||
            dst->index + dst->width > current_kernel_->min_sgprs)
          return std::nullopt;
        if (const auto aperture = rdna3_scalar_aperture(inst)) {
          if (dst->width != 2 || word != dst->index + 1 || (result && result != aperture))
            return std::nullopt;
          result = aperture;
          defined = true;
          break;
        }
        const auto src = inst.src_operand(0)->to_register_ref();
        if ((name != "s_mov_b32" && name != "s_mov_b64") || !src || src->cls != RegClass::SGPR ||
            src->width != dst->width || src->index + src->width > current_kernel_->min_sgprs)
          return std::nullopt;
        word = src->index + word - dst->index;
      }
      if (defined)
        continue;
      if (private_proof_roots_.contains(block) || block->predecessors().empty())
        return std::nullopt;
      for (const BasicBlock *pred : block->predecessors())
        work.emplace_back(pred, word);
    }
    if (result)
      for (const auto &key : seen)
        rdna3_scalar_apertures_.emplace(key, *result);
    return result;
  }

  // On RDNA3 the FLAT segment is selected from the unoffset input pointer.
  // A proven private or shared aperture selects the only path that writes
  // a result. Unknown pointers keep both dependencies, including at joins.
  std::optional<WaitCounterKind> rdna3_flat_result_counter(const Instruction &load,
                                                           rj_code_arch_t arch) {
    if ((arch != ROCJITSU_CODE_ARCH_RDNA3 && arch != ROCJITSU_CODE_ARCH_RDNA3_5) ||
        !current_kernel_ || !load.parent())
      return std::nullopt;
    if (auto it = rdna3_flat_result_counters_.find(&load); it != rdna3_flat_result_counters_.end())
      return it->second;
    auto entry = rdna3_flat_result_counters_.emplace(&load, std::nullopt).first;
    amdgpu::RegisterModifiers modifiers;
    load.amdgpu_register_modifiers(modifiers);
    if (!modifiers.flat_address || modifiers.flat_scalar_address)
      return std::nullopt;
    const auto address = modifiers.flat_address->to_register_ref();
    if (!address || address->cls != RegClass::VGPR || address->width != 2 ||
        address->index + 2u > current_kernel_->min_vgprs)
      return std::nullopt;
    RegisterRef wanted{RegClass::VGPR, static_cast<uint16_t>(address->index + 1), 1};
    const auto &instructions = load.parent()->instructions();
    auto position = decltype(instructions.begin())(&load);
    for (unsigned i = 0; i < 64 && position != instructions.begin(); ++i) {
      const Instruction &inst = *--position;
      const auto du = inst_def_use_for_waitcheck(inst, {}, arch, wavefront_size_);
      if (wanted.cls == RegClass::VGPR &&
          (writes_exec_mask(inst) || du.defs.intersects({RegClass::EXEC, 0, 1})))
        return std::nullopt;
      if (!du.defs.intersects(wanted))
        continue;
      const auto name = inst.mnemonic();
      const Operand *source = nullptr;
      if (name == "v_mov_b32_e32" && inst.num_src_operands() == 1 && inst.num_dst_operands() == 1) {
        source = inst.src_operand(0);
      } else if (const auto begin = vopd_source_begin(inst, wanted, "v_dual_mov_b32", 1)) {
        source = inst.src_operand(*begin);
      }
      if (!source)
        return std::nullopt;
      const auto ref = source->to_register_ref();
      if (!ref || (ref->cls != RegClass::SGPR && ref->cls != RegClass::VGPR) || ref->width != 1)
        return std::nullopt;
      if (ref->index >=
          (ref->cls == RegClass::VGPR ? current_kernel_->min_vgprs : current_kernel_->min_sgprs))
        return std::nullopt;
      if (ref->cls == RegClass::SGPR) {
        entry->second = rdna3_saved_aperture(load.parent(), &inst, ref->index, arch);
        return entry->second;
      }
      wanted = *ref;
    }
    return entry->second;
  }

  // A wave32 scratch pointer has a private aperture prefix and zero lane bits
  // in [56:52]. Adding a 32-bit byte offset and a lane number in [56:52]
  // cannot carry into [63:58], even when the low 52 bits of the base are full.
  enum class ScratchWord { FlatHigh, Address, BaseHigh, OffsetHigh, LaneIndex };

  // One bit per physical dword whose value is a subset of the EXEC at the
  // scratch-offset writer. Joint predecessors retain only facts true on both.
  using ScratchMaskFacts = std::bitset<129>;
  static constexpr size_t scratch_exec_bit = 128;

  static ScratchMaskFacts transfer_scratch_masks(const Instruction &inst, ScratchMaskFacts facts,
                                                 const InstDefUse &du) {
    const auto name = inst.mnemonic();
    const auto index = [](RegisterRef ref) -> std::optional<size_t> {
      if (ref.cls == RegClass::EXEC)
        return scratch_exec_bit;
      if (ref.cls == RegClass::SGPR && ref.width == 1 && ref.index < scratch_exec_bit)
        return ref.index;
      return std::nullopt;
    };
    // RegisterSet tracks EXEC as a singleton. A wave32 proof describes only
    // the low dword; an explicit high-half operand has independent contents.
    const auto low_exec = [](const Operand &op) {
      return op.has_register_selector() &&
             op.encoding_value() == amdgpu::kExecSelectorFirst;
    };
    const auto known = [&](int i) {
      if (i >= inst.num_src_operands())
        return false;
      const auto &op = *inst.src_operand(i);
      if (op.const_value() == 0)
        return true;
      const auto reg = op.to_register_ref();
      const auto bit = reg ? index(*reg) : std::nullopt;
      return bit && (*bit != scratch_exec_bit || low_exec(op)) && facts[*bit];
    };
    const bool old_exec = facts[scratch_exec_bit];
    const bool a = known(0), b = known(1);
    const Operand *destination = inst.num_dst_operands() ? inst.dst_operand(0) : nullptr;
    const auto dest = destination ? destination->to_register_ref() : std::nullopt;
    const auto bit = dest ? index(*dest) : std::nullopt;
    const bool high_only_move = name == "s_mov_b32" && bit == scratch_exec_bit &&
                                destination->has_register_selector() &&
                                destination->encoding_value() == amdgpu::kExecSelectorLast;
    du.defs.for_each([&](RegisterRef r) {
      if (const auto k = index(r); k && (*k != scratch_exec_bit || !high_only_move))
        facts.reset(*k);
    });
    if (high_only_move)
      return facts;
    if (name.starts_with("v_cmpx")) {
      facts[scratch_exec_bit] = old_exec;
    } else if (name.find("saveexec") != std::string_view::npos) {
      // NOT1 complements the old EXEC, NOT0 complements the explicit source.
      if (!bit || *bit == scratch_exec_bit)
        return {};
      facts[*bit] = old_exec;
      facts[scratch_exec_bit] =
          name == "s_and_saveexec_b32" ? old_exec || a :
          name == "s_and_not1_saveexec_b32" ? a :
          name == "s_and_not0_saveexec_b32" ? old_exec :
          (name == "s_or_saveexec_b32" || name == "s_xor_saveexec_b32") && old_exec && a;
    } else if (bit) {
      // Only explicit scalar destination operands select a low or high dword.
      // V_CMPX writes its implicit EXEC as a subset of the previous mask.
      if (bit == scratch_exec_bit && !low_exec(*destination))
        return facts;
      if (name == "s_mov_b32")
        facts[*bit] = a;
      else if (name == "s_and_b32")
        facts[*bit] = a || b;
      else if (name == "s_and_not1_b32")
        facts[*bit] = a;
      else if (name == "s_and_not0_b32")
        facts[*bit] = b;
      else if (name == "s_or_b32" || name == "s_xor_b32" || name == "s_cselect_b32")
        facts[*bit] = a && b;
    }
    return facts;
  }

  bool cdna5_prologue_word(const Instruction &use, RegisterRef high, bool address) {
    const auto key = std::tuple{&use, high.index, address};
    auto [saved, fresh] = scratch_prologue_cache_.try_emplace(key, false);
    if (!fresh)
      return saved->second;
    // Find the predecessor region before the scratch add. Crossing another
    // function, exhausting the bound, or losing the sole root leaves it unknown.
    std::vector<const BasicBlock *> work{use.parent()};
    std::set<const BasicBlock *> region;
    const BasicBlock *root = nullptr;
    while (!work.empty()) {
      const BasicBlock *block = work.back();
      work.pop_back();
      if (!region.insert(block).second)
        continue;
      if (region.size() > 2048 || !block->call_edges().empty())
        return false;
      if (private_proof_roots_.contains(block) || block->predecessors().empty()) {
        if (root && root != block)
          return false;
        root = block;
        continue;
      }
      for (const BasicBlock *pred : block->predecessors())
        work.push_back(pred);
    }
    if (!root || !root->predecessors().empty())
      return false;

    const RegisterRef exec{RegClass::EXEC, 0, 1};
    const Instruction *writer = nullptr;
    unsigned prologue_budget = 96;
    for (const Instruction &inst : root->instructions()) {
      if (!prologue_budget--)
        break;
      const auto name = inst.mnemonic();
      const auto du = inst_def_use_for_waitcheck(inst, {}, ROCJITSU_CODE_ARCH_CDNA5, 32);
      if (du.defs.intersects(exec) || writes_exec_mask(inst) || name.find("movrel") != std::string_view::npos || name == "s_set_vgpr_msb" || (inst.flags() & INDIRECT_CALL))
        break;
      if (!du.defs.intersects(high))
        continue;
      if (address) {
        const auto *raw = inst.raw_encoding();
        const auto pair = inst.num_dst_operands() ? inst.dst_operand(0)->to_register_ref() : std::nullopt;
        if (name != "v_add_nc_u64_e32" || inst.size() != 4 || !raw ||
            (raw[0] & 0xfe0001ffU) != 0x500000e6U || !pair || pair->width != 2 ||
            pair->cls != RegClass::VGPR || pair->index + 1 != high.index)
          continue;
        const auto offset = inst.src_operand(1)->to_register_ref();
        if (!offset || offset->cls != RegClass::VGPR || offset->width != 2)
          continue;
        const RegisterRef offset_high{RegClass::VGPR, static_cast<uint16_t>(offset->index + 1), 1};
        if (cdna5_prologue_word(inst, offset_high, false)) {
          writer = &inst;
          break;
        }
        continue;
      }
      const auto begin = vopd_source_begin(inst, high, "v_dual_lshlrev_b32", 2);
      if (!begin && (name != "v_lshlrev_b32_e32" || inst.size() != 4 ||
                     inst.dst_operand(0)->to_register_ref() != high))
        continue;
      const auto source = begin.value_or(0);
      if (inst.src_operand(source)->const_value() != 20)
        continue;
      const auto lane = inst.src_operand(source + 1)->to_register_ref();
      if (!lane || lane->cls != RegClass::VGPR || lane->width != 1)
        continue;
      for (auto pos = decltype(root->instructions().begin())(&inst); pos != root->instructions().begin();) {
        const Instruction &prior = *--pos;
        if (!inst_def_use_for_waitcheck(prior, {}, ROCJITSU_CODE_ARCH_CDNA5, 32).defs.intersects(*lane))
          continue;
        const auto *raw = prior.raw_encoding();
        if (prior.mnemonic() == "v_mbcnt_lo_u32_b32" && prior.size() == 8 && raw &&
            (raw[0] & 0xffff0000U) == 0xd71f0000U && raw[1] == 0x020100c1U &&
            prior.dst_operand(0)->to_register_ref() == lane)
          writer = &inst;
        break;
      }
      if (writer)
        break;
    }
    if (!writer)
      return false;

    std::map<const BasicBlock *, ScratchMaskFacts> incoming;
    std::deque<const BasicBlock *> queue{root};
    incoming[root].set(scratch_exec_bit);
    bool reached = false;
    unsigned budget = 65536;
    while (!queue.empty()) {
      const BasicBlock *block = queue.front();
      queue.pop_front();
      auto facts = incoming.at(block);
      bool passed_writer = block != root;
      for (const Instruction &inst : block->instructions()) {
        if (!budget--)
          return false;
        const auto name = inst.mnemonic();
        if ((inst.flags() & INDIRECT_CALL) || name.starts_with("s_call") ||
            name.starts_with("s_swappc") || (name.starts_with("s_setreg") && passed_writer) ||
            name == "s_set_vgpr_msb" || name.find("movrel") != std::string_view::npos)
          return false;
        if (&inst == &use) {
          if (!passed_writer || !facts[scratch_exec_bit])
            return false;
          reached = true;
          break;
        }
        const auto du = inst_def_use_for_waitcheck(inst, {}, ROCJITSU_CODE_ARCH_CDNA5, 32);
        if (passed_writer && du.defs.intersects(high))
          return false;
        passed_writer |= &inst == writer;
        facts = transfer_scratch_masks(inst, facts, du);
      }
      // Re-entering this use after a backedge must also preserve its value.
      // The bounded certificate intentionally excludes a use on a cycle.
      if (block == use.parent()) {
        for (const BasicBlock *succ : block->successors())
          if (region.contains(succ))
            return false;
        continue;
      }
      for (const BasicBlock *succ : block->successors()) {
        if (!region.contains(succ))
          continue;
        if (succ == root)
          return false;
        auto [it, inserted] = incoming.emplace(succ, facts);
        if (!inserted && (it->second & facts) == it->second)
          continue;
        it->second &= facts;
        queue.push_back(succ);
      }
    }
    return saved->second = reached;
  }

  struct ScratchCallEffect {
    ScratchMaskFacts clobber;
    bool preserves_word = false;
  };

  // A proven direct call may preserve this particular VGPR without promising
  // to preserve any caller-saved SGPR. Check the decoded body and every exit.
  ScratchCallEffect cdna5_scratch_callee(const BasicBlock::CallEdge &edge, RegisterRef word) {
    const auto key = std::tuple{edge.callee, edge.return_sreg, word.index};
    auto [saved, inserted] = scratch_callee_cache_.try_emplace(key);
    if (!inserted)
      return saved->second;
    ScratchCallEffect effect;
    if (!edge.callee)
      return effect;
    std::map<const BasicBlock *, ScratchMaskFacts> incoming;
    std::deque<const BasicBlock *> work{edge.callee};
    incoming[edge.callee].set(scratch_exec_bit);
    unsigned budget = 32768;
    bool returned = false;
    const RegisterRef link{RegClass::SGPR, edge.return_sreg, 2};
    while (!work.empty()) {
      const auto *block = work.front();
      work.pop_front();
      auto facts = incoming.at(block);
      if (!block->call_edges().empty())
        return {};
      bool ret = false;
      for (const auto &inst : block->instructions()) {
        if (!budget--)
          return {};
        const auto name = inst.mnemonic();
        if (name == "s_set_pc_i64" || name == "s_setpc_b64") {
          if (&inst != block->terminator() || !inst.num_src_operands() ||
              inst.src_operand(0)->to_register_ref() != link || !facts[scratch_exec_bit])
            return {};
          ret = true;
          break;
        }
        if ((inst.flags() & (INDIRECT_BRANCH | INDIRECT_CALL | PROGRAM_TERMINATOR)) ||
            name.starts_with("s_call") || name.starts_with("s_setreg") ||
            name == "s_set_vgpr_msb" || name.find("movrel") != std::string_view::npos)
          return {};
        const auto du = inst_def_use_for_waitcheck(inst, {}, ROCJITSU_CODE_ARCH_CDNA5, 32);
        if (du.defs.intersects(word) || du.defs.intersects(link))
          return {};
        du.defs.for_each([&](RegisterRef r) {
          if (r.cls == RegClass::SGPR)
            for (size_t i = r.index; i < r.index + r.width && i < scratch_exec_bit; ++i)
              effect.clobber.set(i);
        });
        facts = transfer_scratch_masks(inst, facts, du);
      }
      if (ret) {
        returned = true;
        continue;
      }
      if (block->successor_issue() != BasicBlock::SuccessorIssue::None || block->successors().empty())
        return {};
      for (const auto *succ : block->successors()) {
        auto [it, fresh] = incoming.emplace(succ, facts);
        if (!fresh && (it->second & facts) == it->second)
          continue;
        it->second &= facts;
        work.push_back(succ);
      }
    }
    effect.preserves_word = returned;
    return saved->second = effect;
  }

  ScratchCallEffect cdna5_scratch_call(const BasicBlock &caller, const Instruction &inst,
                                       RegisterRef word) {
    ScratchCallEffect effect;
    if (&inst != caller.terminator() || caller.call_edges().empty() ||
        (caller.successor_issue() != BasicBlock::SuccessorIssue::None &&
         caller.successor_issue() != BasicBlock::SuccessorIssue::IndirectControlFlow))
      return {};
    if (caller.successors().size() != 1)
      return {};
    if (std::ranges::any_of(caller.call_edges(), [](const BasicBlock::CallEdge &edge) {
          return edge.kind == BasicBlock::CallEdgeKind::IndirectSwapPc;
        })) {
      const auto &fixups = caller.static_indirect_call_fixups();
      if (fixups.empty() || std::ranges::any_of(fixups, [&](const IndirectCallFixup &fixup) {
            return !fixup.source_is_call || fixup.source_incomplete ||
                   std::ranges::none_of(caller.call_edges(), [&](const BasicBlock::CallEdge &edge) {
                     return edge.callee && edge.callee->start_offset() == fixup.source_target_offset;
                   });
          }))
        return {};
    }
    for (const auto &edge : caller.call_edges()) {
      if (edge.continuation != caller.successors().front())
        return {};
      const auto summary = cdna5_scratch_callee(edge, word);
      if (!summary.preserves_word)
        return {};
      effect.clobber |= summary.clobber;
      if (edge.return_sreg + 1u >= scratch_exec_bit)
        return {};
      effect.clobber.set(edge.return_sreg);
      effect.clobber.set(edge.return_sreg + 1);
    }
    effect.preserves_word = true;
    return effect;
  }

  // Find the only reaching physical writer, admitting both masked and
  // unmasked writes as possible clobbers. Then show that every issuing
  // consumer lane was active after that writer. A repeated definition begins
  // a new proof interval; a path reaching a function entry invalidates it.
  bool cdna5_covered_scratch_word(const Instruction &use, RegisterRef word, ScratchWord goal) {
    const auto key = std::tuple{&use, word.index, goal};
    auto [saved, fresh] = scratch_writer_cache_.try_emplace(key, false);
    if (!fresh)
      return saved->second;
    using Position = std::pair<const BasicBlock *, const Instruction *>;
    const auto next = [](const BasicBlock *block, const Instruction &inst) {
      auto pos = decltype(block->instructions().begin())(&inst);
      return ++pos == block->instructions().end() ? nullptr : &*pos;
    };
    unsigned budget = 65536;
    std::vector<Position> work{{use.parent(), &use}};
    std::set<Position> region;
    const Instruction *writer = nullptr;
    std::map<const Instruction *, ScratchMaskFacts> call_clobbers;
    while (!work.empty()) {
      auto [block, after] = work.back();
      work.pop_back();
      if (!region.emplace(block, after).second)
        continue;
      if (!budget--)
        return false;
      const auto &instructions = block->instructions();
      auto pos = after ? decltype(instructions.begin())(after) : instructions.end();
      if (pos == instructions.begin()) {
        if (private_proof_roots_.contains(block) || block->predecessors().empty())
          return false;
        for (const auto *pred : block->predecessors())
          work.emplace_back(pred, nullptr);
        continue;
      }
      const Instruction &inst = *--pos;
      const auto name = inst.mnemonic();
      if ((inst.flags() & INDIRECT_CALL) || name.starts_with("s_call") || name.starts_with("s_swappc") ||
          name == "s_swap_pc_i64") {
        const auto effect = cdna5_scratch_call(*block, inst, word);
        if (!effect.preserves_word)
          return false;
        call_clobbers[&inst] = effect.clobber;
      }
      if (name.starts_with("s_setreg") || name == "s_set_vgpr_msb" ||
          name.find("movrel") != std::string_view::npos)
        return false;
      if (!inst_def_use_for_waitcheck(inst, {}, ROCJITSU_CODE_ARCH_CDNA5, 32).defs.intersects(word)) {
        work.emplace_back(block, &inst);
        continue;
      }
      if (writer && writer != &inst)
        return false;
      writer = &inst;
    }
    if (!writer)
      return false;
    const auto *raw = writer->raw_encoding();
    const auto name = writer->mnemonic();
    if (goal == ScratchWord::LaneIndex) {
      if (name != "v_mbcnt_lo_u32_b32" || writer->size() != 8 || !raw ||
          (raw[0] & 0xffff0000U) != 0xd71f0000U || raw[1] != 0x020100c1U ||
          writer->dst_operand(0)->to_register_ref() != word)
        return false;
    } else if (goal == ScratchWord::OffsetHigh) {
      const auto begin = vopd_source_begin(*writer, word, "v_dual_lshlrev_b32", 2);
      if (!begin && (name != "v_lshlrev_b32_e32" || writer->size() != 4 ||
                     writer->dst_operand(0)->to_register_ref() != word))
        return false;
      const auto i = begin.value_or(0);
      const auto lane = writer->src_operand(i + 1)->to_register_ref();
      if (writer->src_operand(i)->const_value() != 20 || !lane || lane->cls != RegClass::VGPR ||
          lane->width != 1 || !cdna5_covered_scratch_word(*writer, *lane, ScratchWord::LaneIndex))
        return false;
    } else {
      const auto dst = writer->num_dst_operands() ? writer->dst_operand(0)->to_register_ref() : std::nullopt;
      if (goal != ScratchWord::Address || name != "v_add_nc_u64_e32" || writer->size() != 4 ||
          !raw || (raw[0] & 0xfe0001ffU) != 0x500000e6U || !dst || dst->width != 2 ||
          dst->index + 1 != word.index)
        return false;
      const auto offset = writer->src_operand(1)->to_register_ref();
      if (!offset || offset->cls != RegClass::VGPR || offset->width != 2)
        return false;
      const RegisterRef high{RegClass::VGPR, static_cast<uint16_t>(offset->index + 1), 1};
      if (!cdna5_covered_scratch_word(*writer, high, ScratchWord::OffsetHigh))
        return false;
    }

    // Facts represent subsets of the mask at this particular writer. Join by
    // intersection; forgetting a mask can never promote a previously inactive
    // lane. Propagate only on paths between that writer and the consuming use.
    std::map<Position, ScratchMaskFacts> in;
    std::deque<Position> queue;
    const auto enqueue = [&](Position p, ScratchMaskFacts facts) {
      if (!region.contains(p))
        return;
      auto [it, ins] = in.emplace(p, facts);
      if (!ins && (it->second & facts) == it->second)
        return;
      it->second &= facts;
      queue.push_back(p);
    };
    ScratchMaskFacts initial;
    initial.set(scratch_exec_bit);
    // An SGPR saved just before the writer can cover exactly its issuing
    // lanes. Replay only a straight-line suffix with an unchanged EXEC.
    const auto &writer_block = writer->parent()->instructions();
    const auto end = decltype(writer_block.begin())(writer);
    auto start = end;
    for (unsigned n = 0; n != 64 && start != writer_block.begin(); ++n) {
      auto previous = start;
      const Instruction &inst = *--previous;
      const auto name = inst.mnemonic();
      const auto du = inst_def_use_for_waitcheck(inst, {}, ROCJITSU_CODE_ARCH_CDNA5, 32);
      if (writes_exec_mask(inst) || du.defs.intersects({RegClass::EXEC, 0, 1}) ||
          (inst.flags() & INDIRECT_CALL) || name.starts_with("s_call") || name.starts_with("s_swappc") ||
          name.starts_with("s_setreg") || name == "s_set_vgpr_msb" ||
          name.find("movrel") != std::string_view::npos)
        break;
      start = previous;
    }
    for (; start != end; ++start) {
      const auto du = inst_def_use_for_waitcheck(*start, {}, ROCJITSU_CODE_ARCH_CDNA5, 32);
      initial = transfer_scratch_masks(*start, initial, du);
    }
    // The writer may itself define an SGPR in a dual-issue slot.
    initial = transfer_scratch_masks(*writer, initial,
        inst_def_use_for_waitcheck(*writer, {}, ROCJITSU_CODE_ARCH_CDNA5, 32));
    enqueue({writer->parent(), next(writer->parent(), *writer)}, initial);
    bool reached = false;
    while (!queue.empty()) {
      auto [block, inst] = queue.front();
      queue.pop_front();
      if (!budget--)
        return false;
      const auto facts = in.at({block, inst});
      if (inst == &use) {
        if (!facts[scratch_exec_bit])
          return false;
        reached = true;
      }
      if (!inst) {
        for (const auto *succ : block->successors())
          if (!succ->instructions().empty())
            enqueue({succ, &*succ->instructions().begin()}, facts);
      } else {
        if (const auto call = call_clobbers.find(inst); call != call_clobbers.end()) {
          enqueue({block, next(block, *inst)}, facts & ~call->second);
        } else {
          const auto du = inst_def_use_for_waitcheck(*inst, {}, ROCJITSU_CODE_ARCH_CDNA5, 32);
          enqueue({block, next(block, *inst)}, transfer_scratch_masks(*inst, facts, du));
        }
      }
    }
    return saved->second = reached;
  }

  bool cdna5_scratch_definition(const Instruction &inst, RegisterRef word, ScratchWord goal,
                                unsigned &budget) {
    const auto name = inst.mnemonic();
    const auto scratch_base = [](const Operand &source) {
      return source.has_register_selector() &&
             source.encoding_value() == amdgpu::kFlatScratchBaseSelectorFirst;
    };
    const auto trace = [&](const Operand &source, ScratchWord next) {
      const auto ref = source.to_register_ref();
      return ref && ref->width == 1 && cdna5_scratch_word(inst, *ref, next, budget);
    };
    if (goal == ScratchWord::FlatHigh) {
      if (const auto begin = vopd_source_begin(inst, word, "v_dual_cndmask_b32", 2)) {
        // Zero cannot select an enabled LDS aperture. The low word and mask
        // value are immaterial when both possible high words exclude LDS.
        return inst.src_operand(*begin)->const_value() == 0 &&
               trace(*inst.src_operand(*begin + 1), ScratchWord::Address);
      }
      goal = ScratchWord::Address;
    }
    if (goal == ScratchWord::OffsetHigh) {
      const auto begin = vopd_source_begin(inst, word, "v_dual_lshlrev_b32", 2);
      if (begin || (name == "v_lshlrev_b32_e32" && inst.size() == 4 &&
                    inst.dst_operand(0)->to_register_ref() == word)) {
        const auto i = begin.value_or(0);
        return inst.src_operand(i)->const_value() == 20 &&
               trace(*inst.src_operand(i + 1), ScratchWord::LaneIndex);
      }
      return false;
    }
    if (!inst.num_dst_operands())
      return false;
    const auto dst = inst.dst_operand(0)->to_register_ref();
    if (!dst || dst->cls != RegClass::VGPR)
      return false;
    if (goal == ScratchWord::LaneIndex) {
      return name == "v_mbcnt_lo_u32_b32" && *dst == word && inst.size() == 8 &&
             inst.src_operand(0)->const_value() &&
             static_cast<uint32_t>(*inst.src_operand(0)->const_value()) == UINT32_MAX &&
             inst.src_operand(1)->const_value() == 0;
    }
    const bool high_of_pair = dst->width == 2 && word.index == dst->index + 1;
    if (goal == ScratchWord::BaseHigh)
      return name == "v_mov_b64_e32" && inst.size() == 4 && high_of_pair &&
             scratch_base(*inst.src_operand(0));
    if (name == "v_add_co_ci_u32" && *dst == word && inst.size() == 8) {
      // The carry into this high-word addition is one bit. Even an arbitrary
      // carry plus the shifted wave32 lane stays below the aperture tag.
      return trace(*inst.src_operand(0), ScratchWord::OffsetHigh) &&
             trace(*inst.src_operand(1), ScratchWord::BaseHigh);
    }
    if (name != "v_add_nc_u64_e32" || inst.size() != 4 || !high_of_pair ||
        !scratch_base(*inst.src_operand(0)))
      return false;
    const auto offset = inst.src_operand(1)->to_register_ref();
    if (!offset || offset->cls != RegClass::VGPR || offset->width != 2)
      return false;
    const RegisterRef high{RegClass::VGPR, static_cast<uint16_t>(offset->index + 1), 1};
    return cdna5_scratch_word(inst, high, ScratchWord::OffsetHigh, budget) ||
           cdna5_prologue_word(inst, high, false) ||
           cdna5_covered_scratch_word(inst, high, ScratchWord::OffsetHigh);
  }

  bool cdna5_scratch_word(const Instruction &use, RegisterRef word, ScratchWord goal,
                          unsigned &budget) {
    if (!current_kernel_ || word.cls != RegClass::VGPR || word.index >= current_kernel_->min_vgprs)
      return false;
    using Position = std::tuple<const BasicBlock *, const Instruction *, bool>;
    std::vector<Position> work{{use.parent(), &use, true}};
    std::set<Position> seen;
    bool found = false;
    while (!work.empty()) {
      const auto [block, before, active] = work.back();
      work.pop_back();
      if (!block || !seen.emplace(block, before, active).second)
        continue;
      const auto &instructions = block->instructions();
      auto pos = before ? decltype(instructions.begin())(before) : instructions.end();
      if (pos == instructions.begin()) {
        if (private_proof_roots_.contains(block) || block->predecessors().empty())
          return false;
        for (const BasicBlock *pred : block->predecessors())
          work.emplace_back(pred, nullptr, active);
        continue;
      }
      if (!budget)
        return false;
      --budget;
      const Instruction &inst = *--pos;
      const auto name = inst.mnemonic();
      // A call, register-bank switch or indirect register write can invalidate
      // the assumed physical word; avoid searching across any of them.
      if ((inst.flags() & INDIRECT_CALL) || !block->call_edges().empty() ||
          name.starts_with("s_setreg") || name == "s_set_vgpr_msb" ||
          name.find("movrel") != std::string_view::npos)
        return false;
      const auto du = inst_def_use_for_waitcheck(inst, {}, ROCJITSU_CODE_ARCH_CDNA5, 32);
      const bool defines_word = du.defs.intersects(word);
      // Ordinary VALU writes are lane-masked. Other operations, such as
      // WRITELANE, can change a VGPR even while that lane is inactive.
      if (!active && defines_word && name != "v_add_nc_u64_e32" && name != "v_lshlrev_b32_e32" &&
          name != "v_mbcnt_lo_u32_b32" && name != "v_mov_b32_e32")
        return false;
      if (active && defines_word) {
        if (!cdna5_scratch_definition(inst, word, goal, budget))
          return false;
        found = true;
        continue;
      }
      if (writes_exec_mask(inst) || du.defs.intersects({RegClass::EXEC, 0, 1})) {
        // Inactive prior lanes cannot become active through an AND/compare.
        // An arbitrary EXEC write may restore them, in which case their old
        // VGPR value survived intervening masked writes.
        const auto exec_source = [&](int i) {
          const Operand &source = *inst.src_operand(i);
          return source.has_register_selector() &&
                 source.encoding_value() == amdgpu::kExecSelectorFirst;
        };
        const bool narrowing = name.starts_with("v_cmpx") || name.starts_with("s_and_saveexec") ||
                               (name == "s_and_b32" && inst.num_src_operands() == 2 &&
                                (exec_source(0) || exec_source(1)));
        if (!narrowing || !active)
          work.emplace_back(block, &inst, !active);
      }
      work.emplace_back(block, &inst, active);
    }
    return found;
  }

  // Starting in a lane that issued the earlier read, follow bit-preserving
  // EXEC operations until the later use. NOT1 excludes that lane after an OR
  // retained it. All paths to the use must retain the resulting exclusion.
  bool cdna5_disjoint_flat_read(const Instruction &load, const Instruction &use,
                                const std::unordered_set<const BasicBlock *> &region) {
    enum class Membership { Unknown, Included, Excluded };
    using Point = std::tuple<const BasicBlock *, const Instruction *, Membership>;
    std::vector<Point> work{{load.parent(), &load, Membership::Included}};
    std::set<Point> seen;
    bool reached = false;
    while (!work.empty()) {
      const auto [block, after, known] = work.back();
      work.pop_back();
      if (!seen.emplace(block, after, known).second)
        continue;
      if (seen.size() > 64 || !block->call_edges().empty())
        return false;
      const auto &instructions = block->instructions();
      auto it = after ? std::next(decltype(instructions.begin())(after)) : instructions.begin();
      Membership mask = known;
      for (; it != instructions.end(); ++it) {
        const Instruction &inst = *it;
        if (&inst == &use) {
          if (mask != Membership::Excluded)
            return false;
          reached = true;
          break;
        }
        const auto name = inst.mnemonic();
        if ((inst.flags() & (INDIRECT_CALL | INDIRECT_BRANCH)) || name.starts_with("s_call") ||
            name.starts_with("s_setreg") || name == "s_set_vgpr_msb")
          return false;
        const auto du = inst_def_use_for_waitcheck(inst, {}, ROCJITSU_CODE_ARCH_CDNA5, 32);
        if (!writes_exec_mask(inst) && !du.defs.intersects({RegClass::EXEC, 0, 1}))
          continue;
        const auto low = [](const Operand &op) {
          return op.has_register_selector() && op.encoding_value() == amdgpu::kExecSelectorFirst;
        };
        if (name == "s_or_b32" && low(*inst.dst_operand(0)) &&
            (low(*inst.src_operand(0)) || low(*inst.src_operand(1))) && mask == Membership::Included)
          continue;
        if (name == "s_and_not1_saveexec_b32" && mask == Membership::Included)
          mask = Membership::Excluded;
        else if (mask != Membership::Excluded ||
                 !(name.starts_with("v_cmpx") || name == "s_and_saveexec_b32" ||
                   name == "s_and_not0_saveexec_b32"))
          mask = Membership::Unknown;
      }
      if (it != instructions.end())
        continue;
      if (block->successor_issue() != BasicBlock::SuccessorIssue::None)
        return false;
      for (const BasicBlock *next : block->successors()) {
        if (!region.contains(next))
          continue;
        if (next->start_offset() <= block->start_offset())
          return false;
        work.emplace_back(next, nullptr, mask);
      }
    }
    return reached;
  }

  // For multiple physical definitions, track the union of lanes covered by
  // any authenticated scratch constructor. An unknown incoming dword starts
  // with empty coverage. Alternate writes are clobbers unless the later FLAT
  // provably issues in disjoint lanes.
  bool cdna5_multiple_scratch_addresses(const Instruction &use, RegisterRef high) {
    std::unordered_set<const BasicBlock *> region;
    std::vector<const BasicBlock *> pending{use.parent()};
    std::deque<const BasicBlock *> queue;
    std::map<const BasicBlock *, ScratchMaskFacts> incoming;
    std::unordered_set<const Instruction *> constructors;
    unsigned count = 0;
    while (!pending.empty()) {
      const BasicBlock *block = pending.back();
      pending.pop_back();
      if (!region.insert(block).second)
        continue;
      if ((count += block->num_instructions()) > 512)
        return false;
      bool fresh = private_proof_roots_.contains(block) || block->predecessors().empty();
      if (!fresh) {
        for (const auto *pred : block->predecessors()) {
          if (pred->call_edges().empty())
            pending.push_back(pred);
          else
            fresh = true;
        }
      }
      if (fresh) {
        incoming.emplace(block, ScratchMaskFacts{});
        queue.push_back(block);
      }
    }
    // Cut only at true call continuations. An unknown incoming physical value
    // is conservative and a later constructor can establish fresh coverage.
    for (const auto *block : region) {
      if (!block->call_edges().empty() || block->successor_issue() != BasicBlock::SuccessorIssue::None)
        return false;
      for (const auto &inst : block->instructions()) {
        if (&inst == &use)
          break;
        const auto name = inst.mnemonic();
        const bool replay_control = name == "s_setreg_imm32_b32" && inst.size() == 8 &&
                                    inst.raw_encoding() && inst.raw_encoding()[0] == 0xb9800641U;
        if ((inst.flags() & (INDIRECT_BRANCH | INDIRECT_CALL)) || name.starts_with("s_call") ||
            (name.starts_with("s_setreg") && !replay_control) || name == "s_set_vgpr_msb" ||
            name.find("movrel") != std::string_view::npos)
          return false;
        const auto du = inst_def_use_for_waitcheck(inst, {}, ROCJITSU_CODE_ARCH_CDNA5, 32);
        if (!du.defs.intersects(high))
          continue;
        const auto dst = inst.num_dst_operands() ? inst.dst_operand(0)->to_register_ref() : std::nullopt;
        const auto raw = inst.raw_encoding();
        if (name == "v_add_nc_u64_e32" && raw && (raw[0] & 0xfe0001ffU) == 0x500000e6U &&
            dst && dst->cls == RegClass::VGPR && dst->width == 2 && dst->index + 1 == high.index) {
          const auto offset = inst.src_operand(1)->to_register_ref();
          if (!offset || offset->cls != RegClass::VGPR || offset->width != 2)
            return false;
          const RegisterRef upper{RegClass::VGPR, static_cast<uint16_t>(offset->index + 1), 1};
          if (!cdna5_covered_scratch_word(inst, upper, ScratchWord::OffsetHigh))
            return false;
          constructors.insert(&inst);
        } else if (!(name == "flat_load_b64" && cdna5_disjoint_flat_read(inst, use, region))) {
          return false;
        }
      }
    }
    if (constructors.size() < 2)
      return false;
    bool reached = false;
    unsigned budget = 8192;
    while (!queue.empty()) {
      const auto *block = queue.front();
      queue.pop_front();
      auto facts = incoming.at(block);
      for (const auto &inst : block->instructions()) {
        if (!budget--)
          return false;
        if (&inst == &use) {
          if (!facts[scratch_exec_bit])
            return false;
          reached = true;
          break;
        }
        facts = transfer_scratch_masks(inst, facts,
            inst_def_use_for_waitcheck(inst, {}, ROCJITSU_CODE_ARCH_CDNA5, 32));
        if (constructors.contains(&inst))
          facts.set(scratch_exec_bit);
      }
      for (const auto *succ : block->successors()) {
        if (!region.contains(succ))
          continue;
        if (block == use.parent())
          return false;
        auto [it, fresh] = incoming.emplace(succ, facts);
        if (!fresh && (it->second & facts) == it->second)
          continue;
        it->second &= facts;
        queue.push_back(succ);
      }
    }
    return reached;
  }

  bool cdna5_flat_uses_scratch(const Instruction &inst, const PendingState &state) {
    if (!current_kernel_ || wavefront_size_ != 32 || !inst.parent() || !state.vgpr_msb.known ||
        state.vgpr_msb.mode != 0)
      return false;
    if (auto it = private_flat_addresses_.find(&inst); it != private_flat_addresses_.end())
      return it->second;
    auto cache = private_flat_addresses_.emplace(&inst, false).first;
    amdgpu::RegisterModifiers modifiers;
    inst.amdgpu_register_modifiers(modifiers);
    if (!modifiers.flat_address || modifiers.flat_scalar_address || modifiers.flat_offset)
      return false;
    const auto address = modifiers.flat_address->to_register_ref();
    if (!address || address->cls != RegClass::VGPR || address->width != 2)
      return false;
    unsigned budget = 1500;
    // With no instruction displacement, only the high word can select LDS.
    RegisterRef hi{RegClass::VGPR, static_cast<uint16_t>(address->index + 1), 1};
    cache->second = cdna5_scratch_word(inst, hi, ScratchWord::FlatHigh, budget) ||
                    cdna5_prologue_word(inst, hi, true) ||
                    cdna5_covered_scratch_word(inst, hi, ScratchWord::Address) ||
                    cdna5_multiple_scratch_addresses(inst, hi);
    return cache->second;
  }

  void add_event(PendingState &state, ClassifiedEvent classification, const Instruction &inst,
                 const InstDefUse &du, std::string section_name, uint64_t section_offset,
                 uint64_t file_offset, std::string instruction, rj_code_arch_t arch,
                 bool record_stats) {
    // A scratch-routed FLAT translation completes on the ordinary VM queue.
    // Its DS issue credit remains independent of this translation obligation.
    if (arch == ROCJITSU_CODE_ARCH_CDNA5 && classification.counter == WaitCounterKind::X &&
        (classification.kind == WaitEventKind::FlatLoad ||
         classification.kind == WaitEventKind::FlatStore) &&
        cdna5_flat_uses_scratch(inst, state))
      classification.kind = classification.kind == WaitEventKind::FlatLoad
                                ? WaitEventKind::VmemNoSamplerLoad
                                : WaitEventKind::VmemStore;
    // An issued RDNA3 FLAT still occupies both queues. When its tag is
    // proven, only its selected pipeline can supply destination register data.
    if ((classification.counter == WaitCounterKind::Ds ||
         classification.counter == WaitCounterKind::Load) &&
        classification.kind == WaitEventKind::FlatLoad &&
        classification.registers == TrackedRegisterSource::Defs) {
      const auto result_counter = rdna3_flat_result_counter(inst, arch);
      if (result_counter && classification.counter != *result_counter) {
        classification.registers = TrackedRegisterSource::None;
        classification.check_uses = false;
        classification.check_defs = false;
      }
    }
    if (rejects_memory_destination(inst, state.vgpr_msb, arch)) {
      classification.registers = TrackedRegisterSource::None;
      classification.check_uses = false;
      classification.check_defs = false;
    }
    PendingEvent event;
    event.counter = classification.counter;
    event.kind = classification.kind;
    if (event.kind == WaitEventKind::ValuMatrix)
      event.wmma_wait_states = WaitcheckTarget::wmma_valu_wait_states(inst, arch, target_);
    event.regs = registers_for_event(inst, du, classification, state.vgpr_msb, arch);
    if (classification.registers == TrackedRegisterSource::Defs ||
        classification.registers == TrackedRegisterSource::Uses) {
      const auto access = classification.registers == TrackedRegisterSource::Defs
                              ? WaitcheckAccessKind::Def
                              : WaitcheckAccessKind::Use;
      event.vcc_halves = vcc_access_halves(inst, du, access, arch, wavefront_size_, state.m0);
      if (event.vcc_halves)
        event.regs.expand({RegClass::VCC, 0, 1});
    }
    if (classification.registers == TrackedRegisterSource::Defs) {
      if (const auto partial = partial_d16_load_def(inst, tracks_d16_halves_);
          partial && event.regs.contains(partial->reg)) {
        event.partial_reg = partial->reg;
        event.partial_reg_mask = partial->mask;
      }
    }
    event.special_reg = classification.special_reg;
    event.barrier_id = classification.barrier_id;
    event.check_uses = classification.check_uses;
    event.check_defs = classification.check_defs;
    if ((arch == ROCJITSU_CODE_ARCH_RDNA3 || arch == ROCJITSU_CODE_ARCH_RDNA3_5) &&
        event.counter == WaitCounterKind::Ds &&
        classification.registers == TrackedRegisterSource::Defs &&
        (event.kind == WaitEventKind::Ds || event.kind == WaitEventKind::FlatLoad)) {
      const auto *issue = inst.amdgpu_memory_issue_info();
      event.implies_active_exec = issue && issue->exec_masked;
    }
    if (event.counter == WaitCounterKind::VmVsrc &&
        event.kind != WaitEventKind::Gfx10ScalarSource) {
      if (state.expert_exec_empty)
        event.regs = {};
      event.implies_active_exec = !event.regs.none();
    }
    event.check_exec_defs = classification.check_exec_defs;
    event.section_name = std::move(section_name);
    event.section_offset = section_offset;
    event.file_offset = file_offset;
    event.instruction = std::move(instruction);
    event.check_memory_order = classification.check_memory_order;
    event.check_program_end = classification.check_program_end;
    event.check_counter_parity_order = classification.check_counter_parity_order;
    const size_t idx = counter_index(classification.counter);
    if (classification.kind == WaitEventKind::Smem)
      state.pending_smem[idx] = true;
    const auto units = track_issue_units(state, inst, classification, du.defs, arch);
    if (!units)
      return;
    const bool vm_source =
        event.counter == WaitCounterKind::VmVsrc && amdgpu::supports_expert_scheduling(arch);
    const bool allocated = !vm_source || vm_source_results_allocated(inst, arch, state.vgpr_msb,
                                                                     minimum_source_vgprs(arch));
    age_events(state, event, arch, maximum_dependency_wait(arch, event.counter).value(), *units,
               vm_source && dedicated_source_issues(inst), allocated);
    if (record_stats && event.counter != WaitCounterKind::VaVdst)
      ++report_.memory_events_tracked;
    if (event.counter == WaitCounterKind::VaVdst && tracks_d16_halves_) {
      const bool read = classification.registers == TrackedRegisterSource::VectorUses;
      RegisterSet full;
      event.regs.for_each([&](RegisterRef ref) {
        const uint8_t bytes = trans_operand_bytes(inst, ref, read);
        if (bytes == 0xf) {
          full.expand(ref);
        } else if (bytes) {
          auto part = event;
          part.regs = {};
          part.regs.expand(ref);
          part.partial_reg = ref;
          part.partial_reg_mask =
              ((bytes & 3) ? kVgprLow16Mask : 0) | ((bytes & 12) ? kVgprHigh16Mask : 0);
          materialize_event(state.pending[idx], std::move(part));
        }
      });
      event.regs = std::move(full);
    }
    if (is_counter_token_only(event))
      return;
    materialize_event(state.pending[idx], std::move(event));
  }

  static void materialize_event(std::vector<PendingEvent> &pending, PendingEvent event) {
    auto position = std::ranges::lower_bound(pending, event, event_identity_less);
    if (position != pending.end() && same_event_identity(*position, event))
      *position = std::move(event);
    else
      pending.insert(position, std::move(event));
  }

  void analyze_instruction(PendingState &state, const Instruction &inst,
                           const std::string &section_name, uint64_t section_offset,
                           uint64_t file_offset, rj_code_arch_t arch, bool emit_diagnostics,
                           const DependencyObserver *observe = nullptr) {
    update_vector_skip(state.vector_skip, inst, arch);
    if (inst.is_vskip_affected() && state.vector_skip != false) {
      if (state.vector_skip == true)
        return;
      // A dynamic MODE bit can bypass this entire instruction, including its
      // counter credit, register effects and embedded wait. Retain both paths.
      PendingState skipped = state;
      state.vector_skip = false;
      analyze_instruction(state, inst, section_name, section_offset, file_offset, arch,
                          emit_diagnostics, observe);
      skipped.vector_skip = true;
      merge_into(state, skipped);
      return;
    }
    const bool record_stats = emit_diagnostics;
    // The cap limits stored details, not detection or the observed count.
    const bool emit_report_diagnostics = emit_diagnostics;
    const bool immediately_after_mode_setreg = std::exchange(state.vgpr_msb_setreg_hazard, false);
    const bool previous_vm_vsrc_zero_wait = std::exchange(state.previous_vm_vsrc_zero_wait, false);
    const bool current_vm_vsrc_zero_wait = is_vm_vsrc_zero_wait(inst);
    if (state.async_barrier_post_wait) {
      if (!current_vm_vsrc_zero_wait && emit_report_diagnostics)
        emit_async_barrier_pipe_diagnostic(inst, *state.async_barrier_post_wait,
                                           /*missing_after_barrier=*/true, section_offset,
                                           file_offset);
      state.async_barrier_post_wait.reset();
    }
    if (expert_waits_enabled(state) && amdgpu::supports_expert_scheduling(arch)) {
      const bool may_skip =
          state.expert_valu_may_skip && state.expert_delay.pending && (inst.flags() & VALU) &&
          !expert_valu_issues_without_active_lanes(
              inst,
              inst_def_use_for_waitcheck(inst, state.vgpr_msb, arch, wavefront_size_, state.m0)
                  .defs);
      apply_expert_valu_delay(state, inst, arch, may_skip);
    }
    if (inst.mnemonic() == "s_delay_alu") {
      apply_s_delay_alu(state, inst);
      return;
    }
    apply_due_delay_alu(state);
    auto finish_instruction = [&]() { advance_delay_alu(state); };
    const bool prefetch = arch == ROCJITSU_CODE_ARCH_CDNA5 && is_scalar_prefetch(inst.mnemonic());
    if (prefetch && state.prefetch_enabled == false) {
      finish_instruction();
      return;
    }
    if (apply_sgpr_hazard_ds_nop_cull(state.sgpr_hazards, inst.mnemonic()))
      clear_va_vdst_hazards(state.va_vdst_hazards);
    apply_xcnt_drain(state, inst, arch, immediately_after_mode_setreg);
    if (inst.is_waitcnt()) {
      if ((emit_report_diagnostics || observe) && relative_wait_operand(inst, arch)) {
        const auto uses =
            inst_def_use_for_waitcheck(inst, state.vgpr_msb, arch, wavefront_size_, state.m0);
        if (observe)
          visit_dependencies(state, inst, uses, {}, arch, *observe);
        if (emit_report_diagnostics)
          check_dependencies(state, inst, uses, {}, section_offset, file_offset, arch);
      }
      (void)apply_waitcnt(state, inst, arch);
      state.previous_vm_vsrc_zero_wait = current_vm_vsrc_zero_wait;
      finish_instruction();
      return;
    }
    if (apply_vgpr_msb_mode(state, inst, arch, immediately_after_mode_setreg)) {
      finish_instruction();
      return;
    }
    if (const auto mode = expert_scheduling_mode_write(inst, arch)) {
      state.expert_scheduling = *mode;
      if (!mode->enabled)
        clear_expert_wait_state(state);
    }
    clear_matching_barrier_scc_write(state, inst);
    // LLVM's AMDGPUWaitSGPRHazards pass models this RDNA4 hardware hazard
    // independently of expert scheduling mode. Do not couple SA_SDST,
    // VA_SDST, or VA_VCC tracking to the expert-only VM/VALU counters.
    if (tracks_gfx12_sgpr_hazards(arch))
      apply_sgpr_hazard_memory_cull(state.sgpr_hazards, inst.mnemonic());
    (void)apply_embedded_waitcnt(state, inst, arch);

    InstDefUse du =
        inst_def_use_for_waitcheck(inst, state.vgpr_msb, arch, wavefront_size_, state.m0);
    auto events = effective_events(inst, state.vgpr_msb, arch);
    apply_admission(state, inst, events, arch, false);
    const bool is_async_barrier =
        arch == ROCJITSU_CODE_ARCH_CDNA5 && inst.mnemonic() == "ds_atomic_async_barrier_arrive_b64";
    if (is_async_barrier && !previous_vm_vsrc_zero_wait && emit_report_diagnostics) {
      const SgprHazardProducer barrier{section_name, section_offset, file_offset,
                                       inst.disassemble()};
      emit_async_barrier_pipe_diagnostic(inst, barrier, /*missing_after_barrier=*/false,
                                         section_offset, file_offset);
    }
    if (!expert_waits_enabled(state)) {
      std::erase_if(
          events, [](const ClassifiedEvent &event) { return event.requires_expert_scheduling(); });
    }
    if (prefetch) {
      if (emit_report_diagnostics) {
        for (const auto &event : state.pending[counter_index(WaitCounterKind::X)])
          emit_diagnostic(inst, event, {RegClass::PC, 0, 1}, WaitcheckAccessKind::MemoryOrder, 0,
                          section_offset, file_offset, arch);
      }
    } else {
      apply_implicit_xcnt_ordering(state, inst, du, events, arch);
    }
    if (observe)
      visit_dependencies(state, inst, du, events, arch, *observe);
    if (emit_report_diagnostics) {
      check_dependencies(state, inst, du, events, section_offset, file_offset, arch);
      if (current_cfg_view_index_) {
        if (auto hazard = partial_forwarding_requirement(
                inst, section_offset, file_offset, section_name, arch, *current_cfg_view_index_))
          emit_diagnostic(inst, hazard->event, hazard->reg, WaitcheckAccessKind::Use, 0,
                          section_offset, file_offset, arch);
      }
      if (expert_waits_enabled(state))
        check_va_vdst_hazard(state.va_vdst_hazards, inst, du, section_offset, file_offset);
    }
    apply_admission(state, inst, events, arch, true);
    if (arch == ROCJITSU_CODE_ARCH_RDNA3) {
      if (emit_report_diagnostics) {
        if (auto hazard = trans_use_requirement(state.trans_use_hazards, inst, du, section_offset,
                                                file_offset))
          emit_diagnostic(inst, hazard->event, hazard->reg, WaitcheckAccessKind::Use, 0,
                          section_offset, file_offset, arch);
      }
      advance_trans_use_hazards(state.trans_use_hazards, inst);
      if (is_trans_valu(inst)) {
        const SgprHazardProducer producer{section_name, section_offset, file_offset,
                                          inst.disassemble()};
        du.defs.for_each([&](RegisterRef ref) {
          if (ref.cls != RegClass::VGPR)
            return;
          const uint8_t bytes = trans_operand_bytes(inst, ref, false);
          for (uint16_t half = 0; half < 2; ++half) {
            if (bytes & (half ? 0xc : 0x3))
              state.trans_use_hazards[{static_cast<uint16_t>(2 * ref.index + half), 0}] =
                  TransUseHazard{0, producer};
          }
        });
      }
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
    update_scalar_offsets(state, inst, du, arch);
    const bool writes_exec = writes_exec_mask(inst) || du.defs.contains({RegClass::EXEC, 0, 1}) ||
                             (inst.flags() & INDIRECT_CALL);
    if (amdgpu::supports_expert_scheduling(arch) &&
        (writes_exec || state.expert_saved_nonzero_exec.any() ||
         (state.expert_exec_nonzero && inst.mnemonic() == "s_mov_b32")))
      update_expert_valu_mask(state, inst, du.defs, wavefront_size_,
                              current_kernel_ && current_kernel_->min_sgprs
                                  ? current_kernel_->min_sgprs
                                  : amdgpu::kVccSelectorFirst,
                              writes_exec);
    if (flat_issues_both_counters(arch) && writes_exec) {
      if (arch == ROCJITSU_CODE_ARCH_RDNA3 || arch == ROCJITSU_CODE_ARCH_RDNA3_5)
        for (auto &pending : state.pending[counter_index(WaitCounterKind::Ds)])
          pending.implies_active_exec = false;
      state.exec_nonzero = constant_exec_nonzero(inst, 32) == true;
      if (amdgpu::supports_expert_scheduling(arch)) {
        const bool preserves = preserves_exec_lanes(inst);
        for (auto &source : state.pending[counter_index(WaitCounterKind::VmVsrc)]) {
          source.implies_active_exec &= preserves;
          source.crossed_exec_expansion = source.implies_active_exec;
        }
      }
    }
    if (is_async_barrier) {
      state.async_barrier_post_wait =
          SgprHazardProducer{section_name, section_offset, file_offset, inst.disassemble()};
    }
    finish_instruction();
  }

  void set_analysis_error(std::string_view section_name, uint64_t section_offset,
                          std::string_view error) {
    report_.supported = false;
    std::ostringstream os;
    os << "decode failed in " << section_name << "+0x" << std::hex << section_offset << ": "
       << error;
    report_.analysis_error = os.str();
  }

  struct SharedParity {
    bool cannot_relax = false;
    bool exact = false;
    bool too_weak = false;
    bool incomplete = false;
    std::optional<WaitcheckCounterUnderaccountingDiagnostic> partial_policy;
    std::optional<WaitcheckCounterUnderaccountingDiagnostic> diagnostic;
  };
  std::map<std::tuple<uint64_t, uint64_t, WaitCounterKind, uint32_t>, SharedParity> shared_parity_;
  WaitcheckReport &report_;
  WaitcheckOptions options_;
  std::unique_ptr<Decoder> decoder_;
  bool track_sgpr_offsets_ = true;
  rj_code_target_id_t target_ = ROCJITSU_CODE_TARGET_INVALID;
  bool setreg_vgpr_msb_fixup_ = false;
  bool tracks_d16_halves_ = false;
  // Borrowed from the current linear analysis for on-demand parity lookahead.
  std::span<const uint32_t> stream_words_;
  const WaitcheckKernelInfo *current_kernel_ = nullptr;
  uint32_t wavefront_size_ = 64;
  std::set<
      std::tuple<WaitCounterKind, WaitcheckAccessKind, RegClass, uint16_t, uint16_t, std::string,
                 uint64_t, uint64_t, uint64_t, uint64_t, uint32_t, std::string, std::string>>
      diagnostic_keys_;
  std::set<std::tuple<std::string, WaitCounterKind, uint64_t, uint64_t, uint64_t, uint32_t,
                      uint32_t, bool>>
      counter_underaccounting_keys_;
  std::span<const uint8_t> image_;
  std::optional<std::map<std::string, uint32_t>> metadata_sizes_;
  bool cfg_has_unknown_successor_ = false;
  std::unordered_map<const Instruction *, std::vector<uint16_t>> path_scalar_defs_;
  std::unordered_map<const Instruction *, MaskedSccPredicate> masked_predicates_;
  std::optional<MaskedSccPredicate> repeated_masked_predicate_;
  bool indexed_masked_predicates_ = false;
  bool indexed_loop_parity_ = false;
  std::optional<LoopParity> loop_parity_;
  std::set<uint16_t> path_mask_inputs_;
  bool cfg_has_scratch_zero_exit_ = false;
  bool cfg_can_resolve_scratch_relative_ = false;
  std::set<uint16_t> path_exec_inputs_;
  std::unordered_set<uint64_t> exec_masked_results_;
  bool cfg_has_mask_branches_ = false;
  bool cfg_has_constant_scc_ = false;
  bool cfg_has_exec_guards_ = false;
  bool cfg_has_salu_exec_write_ = false;
  std::bitset<128> source_save_destinations_;
  bool source_allocation_may_change_ = false;
  bool cfg_has_saved_source_exec_ = false;
  std::bitset<128> valu_source_mask_writes_;
  bool cfg_has_scalar_prefetch_ = false;
  bool cfg_has_expert_delay_ = false;
  bool cfg_has_repeated_exec_filter_ = false;
  std::unordered_set<uint64_t> wave64_guarded_loads_;
  std::vector<std::tuple<size_t, uint64_t, SccPredicate>> path_predicates_;
  std::unordered_map<uint64_t, PathQueryStart> path_query_starts_;
  struct PathWaitFacts {
    uint64_t unordered = 0;
    uint64_t flat_load_routes = 0;
    uint64_t flat_store_routes = 0;
  };
  std::unordered_map<const Instruction *, PathWaitFacts> path_wait_facts_;
  std::map<ResolvedCfgWait, WaitFields> relative_cfg_waits_;
  std::map<ResolvedCfgWait, ScalarOffsetValues> relative_cfg_m0_;
  std::map<std::tuple<const Instruction *, uint16_t, ScratchWord>, bool> scratch_writer_cache_;
  std::map<std::tuple<const BasicBlock *, uint16_t, uint16_t>, ScratchCallEffect> scratch_callee_cache_;
  std::map<std::tuple<const Instruction *, uint16_t, bool>, bool> scratch_prologue_cache_;
  std::unordered_map<const Instruction *, bool> private_flat_addresses_;
  std::unordered_map<const Instruction *, std::optional<WaitCounterKind>>
      rdna3_flat_result_counters_;
  std::map<std::pair<const BasicBlock *, uint16_t>, WaitCounterKind> rdna3_scalar_apertures_;
  bool rdna3_has_aperture_ = false;
  std::unordered_set<const BasicBlock *> private_proof_roots_;
  std::vector<CfgBlockView> cfg_views_;
  std::map<std::tuple<uint64_t, size_t, uint64_t>, bool> copy_lane_cache_;
  std::optional<size_t> current_cfg_view_index_;
  std::span<const PendingState> cfg_outputs_;
  bool relative_entry_proven_ = false;
  std::optional<uint64_t> current_function_entry_;
  std::optional<uint16_t> current_call_return_sreg_;
  std::map<std::tuple<uint64_t, uint64_t, WaitCounterKind, bool, bool, uint32_t,
                      std::optional<WaitRange>, std::optional<LaneOperand>, bool, bool,
                      std::optional<MaskedSccPredicate>, std::optional<LoopParity>, WaitEventKind,
                      bool, uint8_t>,
           std::optional<bool>>
      feasible_path_cache_;
  std::map<std::vector<size_t>, SharedReachability> reverse_reachability_cache_;
  std::map<std::vector<size_t>, SharedReachability> forward_reachability_cache_;
  size_t reachability_cache_bytes_ = 0;
  std::vector<size_t> cfg_component_;
  std::vector<size_t> dominator_preorder_;
  std::vector<size_t> dominator_subtree_end_;
};

} // namespace

WaitcheckKernelInfo
waitcheck_kernel_from_descriptor(std::string name, uint64_t descriptor_vaddr, uint64_t entry_offset,
                                 uint64_t code_size, rj_code_arch_t arch,
                                 const amdhsa::kernel_descriptor_t &descriptor) {
  const auto wave = kernel_wavefront_size(arch, descriptor);
  const auto rsrc1 = descriptor.compute_pgm_rsrc1;
  const uint32_t vgprs =
      (1 + RJ_AMDHSA_BITS_GET(rsrc1, amdhsa::COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT)) *
      descriptor_vgpr_count_granule_for_wavefront(arch, wave).value_or(0);
  // CDNA SGPR allocation is rounded to 16. The gfx950 dispatch ABI reserves s32/s33.
  uint32_t sgprs = amdgpu::kVccSelectorFirst;
  if (isa_properties(arch).descriptor_sgpr_count_encoded) {
    const uint32_t requested =
        (1 + RJ_AMDHSA_BITS_GET(rsrc1, amdhsa::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT)) *
        8;
    sgprs = (std::max(requested, arch == ROCJITSU_CODE_ARCH_CDNA4 ? 34u : 0u) + 15) & ~15u;
  }
  uint16_t ordinary = 0, accumulators = 0;
  if (arch == ROCJITSU_CODE_ARCH_CDNA2 || arch == ROCJITSU_CODE_ARCH_CDNA3 ||
      arch == ROCJITSU_CODE_ARCH_CDNA4) {
    ordinary = std::min(vgprs, amdhsa::gfx90a_accum_vgpr_base(descriptor.compute_pgm_rsrc3));
    accumulators = std::min(vgprs - ordinary, amdgpu::ACC_VGPR_OFFSET);
  } else if (arch == ROCJITSU_CODE_ARCH_CDNA5 || arch == ROCJITSU_CODE_ARCH_RDNA4) {
    ordinary = vgprs;
  }
  return {std::move(name),
          descriptor_vaddr,
          entry_offset,
          code_size,
          wave,
          {},
          vgprs,
          vgprs,
          sgprs,
          sgprs,
          kernarg_segment_ptr_sgpr(descriptor),
          descriptor.kernarg_size,
          ordinary,
          accumulators};
}

rj_code_arch_t waitcheck_arch_for_target(rj_code_target_id_t target) {
  switch (target) {
  case ROCJITSU_CODE_TARGET_GFX942:
    return ROCJITSU_CODE_ARCH_CDNA3;
  case ROCJITSU_CODE_TARGET_GFX1100:
    return ROCJITSU_CODE_ARCH_RDNA3;
  case ROCJITSU_CODE_TARGET_GFX1150:
  case ROCJITSU_CODE_TARGET_GFX1151:
    return ROCJITSU_CODE_ARCH_RDNA3_5;
  case ROCJITSU_CODE_TARGET_GFX1200:
  case ROCJITSU_CODE_TARGET_GFX1201:
    return ROCJITSU_CODE_ARCH_RDNA4;
  case ROCJITSU_CODE_TARGET_GFX1250:
  case ROCJITSU_CODE_TARGET_GFX1251:
    return ROCJITSU_CODE_ARCH_CDNA5;
  case ROCJITSU_CODE_TARGET_GFX950:
    return ROCJITSU_CODE_ARCH_CDNA4;
  default:
    return ROCJITSU_CODE_ARCH_INVALID;
  }
}

std::vector<WaitcheckKernelInfo> waitcheck_kernels(const AmdGpuCodeObject &code_object) {
  return find_waitcheck_kernels(code_object, waitcheck_arch_for_target(code_object.target_id()));
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
  if (report.supported)
    analyzer.analyze_stream(words, arch, ".text", 0);
  return report;
}

namespace {

WaitcheckReport analyze_code_object(const CodeObject &code_object, rj_code_arch_t arch,
                                    std::optional<uint64_t> selected_kernel_entry,
                                    std::span<const WaitcheckKernelInfo> known_kernels,
                                    bool kernels_known, WaitcheckOptions options) {
  WaitcheckReport report;
  report.arch = arch;
  if (!is_supported_waitcheck_arch(arch)) {
    report.supported = false;
    report.analysis_error = "unsupported architecture";
    return report;
  }

  uint32_t elf_flags = 0;
  uint8_t hsa_abi = 0;
  if (code_object.image_size() >= sizeof(Elf64_Ehdr)) {
    Elf64_Ehdr ehdr{};
    std::memcpy(&ehdr, code_object.image_data(), sizeof(ehdr));
    if (std::memcmp(ehdr.e_ident, EI_MAGIC, EI_MAGIC_SIZE) == 0) {
      if (ehdr.e_type == ET_REL) {
        report.supported = false;
        report.analysis_error = "relocatable AMDGPU objects are not final loadable code objects";
        return report;
      }
      if (ehdr.e_machine == EM_AMDGPU) {
        elf_flags = ehdr.e_flags;
        if (ehdr.e_ident[EI_OSABI] == ELFOSABI_AMDGPU_HSA)
          hsa_abi = ehdr.e_ident[EI_ABIVERSION];
      }
    }
  }

  for (const auto &owned_section : code_object.all_sections()) {
    const Section *section = owned_section.get();
    if ((section->flags() & SHF_EXECINSTR) == 0 || section->size() == 0)
      continue;
    if (std::ranges::find(code_object.text_sections(), section) !=
        code_object.text_sections().end())
      continue;
    // CFG construction only covers text sections. A linear pass through
    // another executable section could miss a branch that skips a wait.
    report.complete = false;
    report.analysis_error = "executable section outside waitcheck CFG coverage: " +
                            std::string(section->name());
    return report;
  }

  const auto image = std::span(reinterpret_cast<const uint8_t *>(code_object.image_data()),
                               code_object.image_size());
  Analyzer analyzer(report, options, elf_flags, image, hsa_abi);
  if (!report.supported)
    return report;

  const auto text_file_offset = code_object.text_sections().size() == 1
                                    ? code_object.text_sections().front()->sectionOffset()
                                    : 0;
  const auto text_size =
      code_object.text_sections().size() == 1 ? code_object.text_sections().front()->size() : 0;
  const auto cfg_error = [&](std::string_view message) {
    report.supported = false;
    report.analysis_error = "decode failed while building CFG: " + std::string(message);
  };
  {
    std::vector<WaitcheckKernelInfo> discovered_kernels;
    if (!kernels_known)
      discovered_kernels = find_waitcheck_kernels(code_object, arch);
    const std::span<const WaitcheckKernelInfo> kernels =
        kernels_known ? known_kernels : std::span<const WaitcheckKernelInfo>{discovered_kernels};
    report.kernels_discovered = kernels.size();
    const auto function_entries = !kernels_known && kernels.empty()
                                      ? find_waitcheck_function_entries(code_object)
                                      : std::vector<uint64_t>{};
    if (kernels_known && kernels.empty())
      return report;
    if (selected_kernel_entry &&
        std::ranges::find(kernels, *selected_kernel_entry, &WaitcheckKernelInfo::entry_offset) ==
            kernels.end() &&
        std::ranges::find(function_entries, *selected_kernel_entry) == function_entries.end()) {
      report.supported = false;
      report.analysis_error = "kernel entry offset is not present in the code object";
      return report;
    }
    if (kernels.empty() && function_entries.empty()) {
      auto blocks =
          BasicBlock::build(code_object, analyzer.decoder(), arch, DecodeErrorEmitter(cfg_error));
      if (blocks.failed())
        return report;
      analyzer.set_kernel_context(nullptr);
      analyzer.analyze_cfg(blocks.value(), ".text", text_file_offset, arch,
                           default_wavefront_size(arch), text_size);
    } else {
      // Analyze each kernel independently. Building one combined CFG for a
      // multi-kernel library keeps two large wait states per basic block alive
      // at once; generated Tensile libraries can contain hundreds of kernels.
      const auto analyze_entry = [&](uint64_t entry_offset, const WaitcheckKernelInfo *kernel,
                                     uint32_t wavefront_size) {
        const std::array<uint64_t, 1> entry{entry_offset};
        const auto start = std::chrono::steady_clock::now();
        auto blocks = BasicBlock::build_cfg(
            code_object, analyzer.decoder(), arch,
            {.entries = entry, .wave_size = static_cast<uint8_t>(wavefront_size)},
            DecodeErrorEmitter(cfg_error));
        if (blocks.failed())
          return;
        if (arch == ROCJITSU_CODE_ARCH_CDNA5)
          retry_gfx1250_indirect_calls(code_object, analyzer.decoder(), entry, wavefront_size,
                                       blocks.value());
        analyzer.set_kernel_context(kernel);
        analyzer.analyze_cfg(blocks.value(), ".text", text_file_offset, arch, wavefront_size,
                             text_size, entry,
                             /*non_entry_function=*/kernel == nullptr);
        if (kernel && report.supported && !report.stopped_early) {
          ++report.kernels_analyzed;
          if (options.kernel_analyzed_callback)
            options.kernel_analyzed_callback();
          if (options.kernel_timing_callback)
            options.kernel_timing_callback(*kernel,
                                           std::chrono::duration_cast<std::chrono::nanoseconds>(
                                               std::chrono::steady_clock::now() - start));
        }
      };
      for (const WaitcheckKernelInfo &kernel : kernels) {
        if (selected_kernel_entry && kernel.entry_offset != *selected_kernel_entry)
          continue;
        analyze_entry(kernel.entry_offset, &kernel, kernel.wavefront_size);
        if (!report.supported || report.stopped_early)
          break;
      }
      if (report.supported && !report.stopped_early) {
        for (uint64_t entry_offset : function_entries) {
          if (selected_kernel_entry && entry_offset != *selected_kernel_entry)
            continue;
          analyze_entry(entry_offset, nullptr, default_wavefront_size(arch));
          if (!report.supported || report.stopped_early)
            break;
        }
      }
    }
  }
  if (!report.supported)
    return report;
  if (report.stopped_early)
    return report;
  if (selected_kernel_entry || kernels_known)
    return report;

  return report;
}

} // namespace

WaitcheckReport analyze_waitcnts(const CodeObject &code_object, rj_code_arch_t arch,
                                 WaitcheckOptions options) {
  return analyze_code_object(code_object, arch, std::nullopt, {}, false, options);
}

WaitcheckReport analyze_waitcnts_for_kernel(const CodeObject &code_object, rj_code_arch_t arch,
                                            uint64_t kernel_entry_offset,
                                            WaitcheckOptions options) {
  return analyze_code_object(code_object, arch, kernel_entry_offset, {}, false, options);
}

WaitcheckReport analyze_waitcnts_for_kernel(const CodeObject &code_object, rj_code_arch_t arch,
                                            const WaitcheckKernelInfo &kernel,
                                            WaitcheckOptions options) {
  return analyze_code_object(code_object, arch, kernel.entry_offset,
                             std::span<const WaitcheckKernelInfo>{&kernel, 1}, true, options);
}

WaitcheckReport analyze_waitcnts_for_kernels(const CodeObject &code_object, rj_code_arch_t arch,
                                             std::span<const WaitcheckKernelInfo> kernels,
                                             WaitcheckOptions options) {
  return analyze_code_object(code_object, arch, std::nullopt, kernels, true, options);
}

} // namespace rocjitsu
