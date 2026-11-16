// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/analysis/waitcheck.h"

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/code/analysis/def_use_chain.h"
#include "rocjitsu/code/analysis/waitcheck/state.h"
#include "rocjitsu/code/basic_block.h"
#include "rocjitsu/code/code_object.h"
#include "rocjitsu/code/kernel_descriptor_scan.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/generated/shared/isa_properties.h"
#include "rocjitsu/isa/arch/amdgpu/shared/instruction_encoding.h"
#include "rocjitsu/isa/arch/amdgpu/shared/scalar_operand_selectors.h"
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
#include <iomanip>
#include <iterator>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
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
struct CounterParityRequirement {
  PendingEvent event;
  RegisterRef reg{RegClass::VGPR, 0, 1};
  WaitcheckAccessKind access = WaitcheckAccessKind::Use;
  uint32_t required_count = 0;
  const Instruction *consumer = nullptr;
  uint64_t consumer_section_offset = 0;
  uint64_t consumer_file_offset = 0;
};

struct PendingWaitGroup {
  PendingState state_before;
  std::array<std::optional<uint32_t>, kCounterCount> emitted_counts;
  std::string section_name;
  uint64_t section_offset = 0;
  uint64_t file_offset = 0;
  std::string instructions;
  bool non_entry_function_prolog = false;
};

struct CfgInstructionView {
  uint64_t section_offset = 0;
  const Instruction *instruction = nullptr;
  // Exclusive end of a straight-line run that leaves path-search state intact.
  size_t neutral_run_end = 0;
  size_t predicate_index = std::numeric_limits<size_t>::max();
  uint64_t issued_counters = 0;
  bool invalidates_scc = false;
  bool invalidates_exec = false;
  bool invalidates_vcc = false;
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
};

struct ScalarValueConstraint {
  std::optional<uint32_t> equal;
  std::set<uint32_t> not_equal;
  std::optional<uint32_t> unsigned_min;
  std::optional<uint32_t> unsigned_max;
};

using ScalarConstraints = std::map<uint16_t, ScalarValueConstraint>;

struct SccPredicate {
  enum class Kind : uint8_t { EqImm, GeUnsignedImm };

  Kind kind = Kind::EqImm;
  uint16_t sgpr = 0;
  uint32_t value = 0;
  bool operator==(const SccPredicate &) const = default;
};

struct Analyzer : private WaitcheckStateOps {
  Analyzer(WaitcheckReport &report, WaitcheckOptions options, uint32_t elf_machine = 0)
      : report_(report), options_(options) {
    const auto &registry = options.registry.get();
    if (const auto *arch = registry.find(report.arch)) {
      const auto bound =
          std::ranges::find(arch->gpu_targets, elf_machine, &IsaGpuTargetDescription::elf_machine);
      const auto *gpu_target =
          bound != arch->gpu_targets.end() ? &*bound : registry.find_default_gpu_target(*arch);
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
      update_counter_parity_group(pending_wait_group, state, *inst, section_name, section_offset,
                                  file_offset, arch);
      analyze_instruction(state, *inst, section_name, section_offset, file_offset, arch, true);
      ++report_.instructions_analyzed;
      word_index += inst_words;
      if (should_stop_after_diagnostic())
        break;
    }
    finish_counter_parity_block(pending_wait_group, arch);
  }

  void analyze_cfg(std::vector<std::unique_ptr<BasicBlock>> &blocks,
                   const std::string &section_name, uint64_t file_offset_base, rj_code_arch_t arch,
                   uint32_t wavefront_size, uint64_t text_size,
                   std::span<const uint64_t> entry_offsets = {}, bool non_entry_function = false) {
    if (blocks.empty())
      return;
    wavefront_size_ = wavefront_size;
    path_wait_facts_.clear();

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
    using CallFrame = std::tuple<size_t, uint16_t, size_t, uint64_t>;
    using CallContinuationSeed = std::pair<uint16_t, uint64_t>;
    using AnalysisNodeKey =
        std::tuple<size_t, std::vector<CallFrame>, std::optional<CallContinuationSeed>>;
    constexpr size_t kMaxCallDepth = 32;
    // Real code objects can have tens of thousands of distinct depth-two
    // contexts when many wrappers share a small set of helpers. Keep a hard
    // bound, but do not reject that finite graph merely because the underlying
    // CFG is small.
    constexpr size_t kCallContextNodeAllowance = 65536;
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
      if (!std::get<1>(node_key).empty() && term != nullptr &&
          term->size() == static_cast<int>(sizeof(uint32_t)) &&
          (term->mnemonic() == "s_setpc_b64" || term->mnemonic() == "s_set_pc_i64") &&
          term->raw_encoding() != nullptr &&
          static_cast<uint16_t>(term->raw_encoding()[0] & 0xffu) ==
              std::get<1>(std::get<1>(node_key).back())) {
        std::vector<CallFrame> caller_stack = std::get<1>(node_key);
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
        if (!is_call_fallthrough && !add_context_edge(successor_it->second, std::get<1>(node_key),
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
          if (!add_context_edge(continuation_it->second, std::get<1>(node_key),
                                /*conservative_sgpr_entry=*/true,
                                CallContinuationSeed{call.return_sreg, call.source_call_offset})) {
            report_.supported = false;
            set_call_context_limit_error();
            return;
          }
          continue;
        }
        std::vector<CallFrame> callee_stack = std::get<1>(node_key);
        callee_stack.emplace_back(continuation_it->second, call.return_sreg, callee_it->second,
                                  call.source_call_offset);
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
          cfg_predecessors[i].empty())
        merged.exec_nonzero = true;
      if (out_initialized[i])
        merge_into(merged, in[i]);
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
      PendingState next_out =
          analyze_block(*analysis_blocks[i], merged, section_name, file_offset_base, arch, false);
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

    prepare_cfg_path_filter(analysis_blocks, cfg_predecessors, cfg_successors, arch);
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

  [[nodiscard]] static auto find_event(std::vector<PendingEvent> &events,
                                       const PendingEvent &event) {
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

  void prepare_cfg_path_filter(const std::vector<BasicBlock *> &blocks,
                               const std::vector<std::vector<size_t>> &predecessors,
                               const std::vector<std::vector<size_t>> &successors,
                               rj_code_arch_t arch) {
    cfg_views_.clear();
    feasible_path_cache_.clear();
    path_scalar_defs_.clear();
    path_predicates_.clear();
    path_mask_inputs_.clear();
    cfg_has_mask_branches_ = false;
    cfg_has_exec_guards_ = false;
    path_query_starts_.clear();
    reverse_reachability_cache_.clear();
    forward_reachability_cache_.clear();
    reachability_cache_bytes_ = 0;
    dominator_preorder_.clear();
    dominator_subtree_end_.clear();

    cfg_views_.reserve(blocks.size());
    for (size_t block_index = 0; block_index < blocks.size(); ++block_index) {
      BasicBlock *block = blocks[block_index];
      CfgBlockView view;
      view.block = block;
      view.predecessors = predecessors[block_index];
      view.successors = successors[block_index];
      uint64_t section_offset = block->start_offset();
      for (const Instruction &inst : block->instructions()) {
        const auto predicate = scalar_scc_predicate(inst);
        const size_t predicate_index =
            predicate ? path_predicates_.size() : std::numeric_limits<size_t>::max();
        if (predicate)
          path_predicates_.emplace_back(block_index, section_offset, *predicate);
        const InstDefUse du =
            inst_def_use_for_waitcheck(inst, VgprMsbState{}, arch, wavefront_size_);
        auto [defs, inserted] = path_scalar_defs_.try_emplace(&inst);
        if (inserted) {
          du.defs.for_each([&](RegisterRef ref) {
            if (ref.cls == RegClass::SGPR)
              defs->second.push_back(ref.index);
          });
        }
        static_assert(kCounterCount <= 64);
        uint64_t issued_counters = 0;
        const auto classified = classify_events(inst, arch).value();
        for (const auto &event : classified)
          issued_counters |= uint64_t{1} << counter_index(event.counter);
        const bool invalidates_scc = du.defs.contains({RegClass::SCC, 0, 1}) ||
                                     instruction_defines_special(inst, {RegClass::SCC, 0, 1}) ||
                                     instruction_uses_special(inst, {RegClass::SCC, 0, 1});
        const bool invalidates_exec =
            writes_exec_mask(inst) || du.defs.contains({RegClass::EXEC, 0, 1});
        const bool invalidates_vcc = instruction_defines_vcc(inst, du);
        if (mask_branch_class(inst.mnemonic()) == RegClass::EXEC ||
            ((inst.mnemonic() == "s_mov_b64" || inst.mnemonic() == "s_mov_b32") &&
             low_selector(inst.dst_operand(0), amdgpu::kExecSelectorFirst)))
          cfg_has_exec_guards_ = true;
        if (mask_branch_class(inst.mnemonic()))
          cfg_has_mask_branches_ = true;
        if (inst.mnemonic() == "s_and_b32" && operand_is_vcc(inst.dst_operand(0))) {
          for (int i = 0; i < inst.num_src_operands(); ++i) {
            const auto ref = inst.src_operand(i)->to_register_ref();
            if (ref && ref->cls == RegClass::SGPR && ref->width == 1)
              path_mask_inputs_.insert(ref->index);
          }
        }
        const bool neutral =
            !invalidates_exec && !invalidates_vcc && defs->second.empty() && !predicate &&
            !inst.is_branch() && !inst.is_waitcnt() && !is_program_end(inst.mnemonic()) &&
            !releases_gfx10_scalar_sources(inst, arch) && !vinterp_wait_exp(inst) &&
            dsdir_wait_vm_vsrc(inst) != 0 && !invalidates_scc && issued_counters == 0;
        const size_t index = view.instructions.size();
        view.instructions.push_back({section_offset, &inst, index + (neutral ? 1 : 0),
                                     predicate_index, issued_counters, invalidates_scc,
                                     invalidates_exec, invalidates_vcc});
        section_offset += static_cast<uint64_t>(inst.size());
      }
      for (size_t index = view.instructions.size(); index-- > 0;) {
        auto &inst = view.instructions[index];
        if (inst.neutral_run_end == index + 1 && index + 1 < view.instructions.size())
          inst.neutral_run_end = view.instructions[index + 1].neutral_run_end;
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
    cfg_views_.clear();
    feasible_path_cache_.clear();
    path_scalar_defs_.clear();
    path_predicates_.clear();
    path_query_starts_.clear();
    reverse_reachability_cache_.clear();
    forward_reachability_cache_.clear();
    reachability_cache_bytes_ = 0;
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

  [[nodiscard]] static bool instruction_emits_counter_wait(const Instruction &inst,
                                                           WaitCounterKind counter,
                                                           rj_code_arch_t arch) {
    const size_t index = counter_index(counter);
    if (counter == WaitCounterKind::VmVsrc && releases_gfx10_scalar_sources(inst, arch))
      return true;
    if (const auto fields = explicit_wait_fields(inst, arch).value(); fields && (*fields)[index])
      return true;
    if (const auto fields = embedded_wait_fields(inst, arch).value(); fields && (*fields)[index])
      return true;
    return false;
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
      (void)apply_counter_wait(state, counter, count, arch);
      break;
    case WaitCounterKind::VaVdst:
    case WaitCounterKind::Depctr:
    case WaitCounterKind::Count:
      break;
    }
  }

  [[nodiscard]] static bool has_counter_parity_candidate(const PendingState &state,
                                                         WaitCounterKind counter) {
    if (counter == WaitCounterKind::VaVdst)
      return !state.va_vdst_hazards.hazards.empty();
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

  void collect_counter_parity_requirements(
      const PendingWaitGroup &group, const Instruction &consumer, uint64_t consumer_section_offset,
      uint64_t consumer_file_offset, rj_code_arch_t arch,
      std::array<std::optional<CounterParityRequirement>, kCounterCount> &requirements,
      bool past_xcnt_drain = false) {
    const PendingState *requirement_state = &group.state_before;
    const InstDefUse du = inst_def_use_for_waitcheck(consumer, requirement_state->vgpr_msb, arch,
                                                     wavefront_size_, requirement_state->m0);
    auto current_events = classify_events(consumer, arch).value();
    if (!expert_waits_enabled(*requirement_state)) {
      std::erase_if(current_events, [](const ClassifiedEvent &event) {
        return event.requires_expert_scheduling();
      });
    }
    std::optional<PendingState> xcnt_adjusted_state;
    if (std::ranges::any_of(current_events, [](const ClassifiedEvent &event) {
          return event.counter == WaitCounterKind::X;
        })) {
      xcnt_adjusted_state.emplace(group.state_before);
      apply_implicit_xcnt_ordering(*xcnt_adjusted_state, consumer, du, current_events, arch);
      requirement_state = &*xcnt_adjusted_state;
    }
    visit_dependencies(
        *requirement_state, consumer, du, current_events, arch,
        [&](const PendingEvent &event, RegisterRef reg, WaitcheckAccessKind access,
            uint32_t required_count) {
          // A branch, bank update, or other implicit X wait retires replay
          // sources before this instruction observes or changes registers.
          if (event.counter == WaitCounterKind::X &&
              (past_xcnt_drain || (arch == ROCJITSU_CODE_ARCH_CDNA5 && is_xcnt_drain(consumer))))
            return;
          bool can_improve =
              !requirements[counter_index(event.counter)] ||
              required_count < requirements[counter_index(event.counter)]->required_count;
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
            auto &strongest = requirements[counter_index(counter)];
            if (!strongest || count < strongest->required_count) {
              strongest = CounterParityRequirement{event,
                                                   reg,
                                                   access,
                                                   count,
                                                   &consumer,
                                                   consumer_section_offset,
                                                   consumer_file_offset};
            }
          };
          record_requirement(event.counter, required_count);

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
        });

    if (expert_waits_enabled(*requirement_state) && dsdir_wait_va_vdst(consumer)) {
      auto &strongest = requirements[counter_index(WaitCounterKind::VaVdst)];
      du.defs.for_each([&](RegisterRef ref) {
        if (ref.cls != RegClass::VGPR || ref.index >= REGISTER_SET_MAX_VGPRS)
          return;
        const auto hazard_it = requirement_state->va_vdst_hazards.hazards.find(ref.index);
        if (hazard_it == requirement_state->va_vdst_hazards.hazards.end())
          return;
        const VaVdstHazard &hazard = hazard_it->second;
        const uint32_t required_count = hazard.trans_since ? 0u : hazard.age;
        if (strongest && strongest->required_count <= required_count)
          return;

        PendingEvent producer;
        producer.counter = WaitCounterKind::VaVdst;
        producer.section_name = hazard.producer.section_name;
        producer.section_offset = hazard.producer.section_offset;
        producer.file_offset = hazard.producer.file_offset;
        producer.instruction = hazard.producer.instruction;
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
      // SIMemoryLegalizer lowers an acquire fence that crosses LDS and global
      // address spaces to soft all-zero waits followed by GLOBAL_INV.  The
      // cache instruction itself is not an ordinary register dependency, but
      // it is the object-visible marker that gives the preceding wait group
      // its memory-order meaning.  Attribute only counter fields that the
      // GFX12 fence lowering can emit; otherwise forward lookahead can attach
      // a fence drain to an unrelated register dependency after GLOBAL_INV.
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
      rj_code_arch_t arch, size_t max_lookahead_positions, bool &complete) {
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
    using Position = std::tuple<size_t, size_t, bool>;
    std::vector<Position> worklist{{
        *current_cfg_view_index_,
        static_cast<size_t>(std::distance(start_block.instructions.begin(), start_it)),
        false,
    }};
    std::set<Position> visited;
    std::set<std::pair<const Instruction *, bool>> dependencies_checked;
    std::optional<CounterParityRequirement> strongest;
    while (!worklist.empty() && visited.size() < max_lookahead_positions) {
      const auto [block_index, instruction_index, past_xcnt_drain] = worklist.back();
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
      // The explicit wait separates the first looked-ahead bank change from
      // MODE writes. Any later adjacent MODE write itself already drains X.
      const bool xcnt_drained = past_xcnt_drain || (may_track_xcnt && is_xcnt_drain(instruction));
      if ((counter == WaitCounterKind::X && xcnt_drained) ||
          instruction_emits_counter_wait(instruction, counter, arch)) {
        continue;
      }

      if (dependencies_checked.emplace(&instruction, xcnt_drained).second) {
        const InstDefUse du = inst_def_use_for_waitcheck(instruction, group.state_before.vgpr_msb,
                                                         arch, wavefront_size_);
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
            (amdgpu::uses_legacy_waitcnt(amdgpu::waitcnt_model(arch).value()) &&
             counter == WaitCounterKind::Ds && is_barrier) ||
            (counter == WaitCounterKind::VaVdst && dsdir_wait_va_vdst(instruction));
        if (may_depend_on_register || may_depend_on_special || may_depend_on_exec ||
            may_depend_on_order) {
          std::array<std::optional<CounterParityRequirement>, kCounterCount> requirements;
          collect_counter_parity_requirements(group, instruction, view.section_offset,
                                              group.file_offset - group.section_offset +
                                                  view.section_offset,
                                              arch, requirements, xcnt_drained);
          if (const auto &requirement = requirements[counter_index(counter)]) {
            if (!strongest || requirement->required_count < strongest->required_count)
              strongest = requirement;
            // Counter-parity auditing is intentionally one-sided. As soon as
            // one guarded dependency requires a value no weaker than the
            // emitted field, this wait cannot be an N > M under-accounting
            // candidate. Avoid scanning the rest of a large CFG solely to
            // discover an even stronger ordinary-hazard requirement.
            if (strongest->required_count <= emitted_count)
              return strongest;
          }
        }
      }
      if (is_program_end(instruction.mnemonic()))
        continue;
      worklist.emplace_back(block_index, instruction_index + 1, xcnt_drained);
    }
    if (!worklist.empty())
      complete = false;
    return strongest;
  }

  [[nodiscard]] std::optional<CounterParityRequirement> find_staged_counter_parity_requirement(
      const PendingWaitGroup &group, const Instruction &first_consumer,
      uint64_t first_consumer_section_offset, WaitCounterKind counter, uint32_t emitted_count,
      rj_code_arch_t arch, bool &complete) {
    if (!current_cfg_view_index_ || *current_cfg_view_index_ >= cfg_views_.size() ||
        counter == WaitCounterKind::VaVdst) {
      return std::nullopt;
    }

    const CfgBlockView &start_block = cfg_views_[*current_cfg_view_index_];
    const auto start_it = std::ranges::find_if(start_block.instructions, [&](const auto &view) {
      return view.instruction == &first_consumer &&
             view.section_offset == first_consumer_section_offset;
    });
    if (start_it == start_block.instructions.end())
      return std::nullopt;

    PendingState initial_state = group.state_before;
    for (size_t counter_idx = 0; counter_idx < group.emitted_counts.size(); ++counter_idx) {
      if (counter_idx == counter_index(counter) || !group.emitted_counts[counter_idx])
        continue;
      apply_counter_parity_wait(initial_state, static_cast<WaitCounterKind>(counter_idx),
                                *group.emitted_counts[counter_idx], arch);
    }

    const std::vector<PendingEvent> original_events = initial_state.pending[counter_index(counter)];
    if (original_events.empty())
      return std::nullopt;

    auto original_event = [&](const PendingEvent &event) -> const PendingEvent * {
      const auto found = std::ranges::find_if(original_events, [&](const PendingEvent &original) {
        return same_event_identity(original, event);
      });
      return found == original_events.end() ? nullptr : &*found;
    };

    using Position = std::pair<size_t, size_t>;
    const Position start_position{
        *current_cfg_view_index_,
        static_cast<size_t>(std::distance(start_block.instructions.begin(), start_it)),
    };
    std::map<Position, PendingState> states;
    states.emplace(start_position, initial_state);
    std::vector<Position> worklist{start_position};
    std::optional<CounterParityRequirement> strongest;
    constexpr size_t kMaxSimulationSteps = 200000;
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

    while (!worklist.empty() && steps++ < kMaxSimulationSteps) {
      const Position position = worklist.back();
      worklist.pop_back();
      const auto [block_index, instruction_index] = position;
      PendingState state = states.at(position);
      const CfgBlockView &block = cfg_views_[block_index];
      if (instruction_index >= block.instructions.size()) {
        for (size_t successor : block.successors)
          enqueue({successor, 0}, state);
        continue;
      }

      const CfgInstructionView &view = block.instructions[instruction_index];
      const Instruction &instruction = *view.instruction;
      if (view.section_offset == group.section_offset && position != start_position &&
          instruction_emits_counter_wait(instruction, counter, arch)) {
        continue;
      }

      const PendingState *dependency_state = &state;
      const InstDefUse du = inst_def_use_for_waitcheck(instruction, dependency_state->vgpr_msb,
                                                       arch, wavefront_size_, dependency_state->m0);
      auto current_events = classify_events(instruction, arch).value();
      if (!expert_waits_enabled(*dependency_state)) {
        std::erase_if(current_events, [](const ClassifiedEvent &event) {
          return event.requires_expert_scheduling();
        });
      }
      std::optional<PendingState> xcnt_adjusted_state;
      if (std::ranges::any_of(current_events, [](const ClassifiedEvent &event) {
            return event.counter == WaitCounterKind::X;
          })) {
        xcnt_adjusted_state.emplace(state);
        apply_implicit_xcnt_ordering(*xcnt_adjusted_state, instruction, du, current_events, arch);
        dependency_state = &*xcnt_adjusted_state;
      }
      auto record_if_original = [&](const PendingEvent &event, RegisterRef reg,
                                    WaitcheckAccessKind access) {
        const PendingEvent *original = original_event(event);
        if (!original)
          return;
        const uint32_t required_count =
            dependency_required_count(initial_state, *original, arch).value();
        if (!strongest || required_count < strongest->required_count) {
          strongest = CounterParityRequirement{*original,
                                               reg,
                                               access,
                                               required_count,
                                               &instruction,
                                               view.section_offset,
                                               group.file_offset - group.section_offset +
                                                   view.section_offset};
        }
      };
      visit_dependencies(
          *dependency_state, instruction, du, current_events, arch,
          [&](const PendingEvent &event, RegisterRef reg, WaitcheckAccessKind access, uint32_t) {
            if (event.counter == counter &&
                !(counter == WaitCounterKind::X && arch == ROCJITSU_CODE_ARCH_CDNA5 &&
                  is_xcnt_drain(instruction, state.vgpr_msb_setreg_hazard)))
              record_if_original(event, reg, access);
          });

      if (amdgpu::uses_legacy_waitcnt(amdgpu::waitcnt_model(arch).value()) &&
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
      if (strongest && strongest->required_count <= emitted_count)
        return strongest;

      analyze_instruction(state, instruction, group.section_name, view.section_offset,
                          group.file_offset - group.section_offset + view.section_offset, arch,
                          /*emit_diagnostics=*/false);
      if (is_program_end(instruction.mnemonic()))
        continue;

      const auto &pending = state.pending[counter_index(counter)];
      const bool has_original = std::ranges::any_of(
          pending, [&](const PendingEvent &event) { return original_event(event) != nullptr; });
      if (!has_original)
        continue;
      enqueue({block_index, instruction_index + 1}, state);
    }
    if (!worklist.empty())
      complete = false;
    return strongest;
  }

  void audit_counter_parity_group(const PendingWaitGroup &group, const Instruction &consumer,
                                  uint64_t consumer_section_offset, uint64_t consumer_file_offset,
                                  rj_code_arch_t arch) {
    ++report_.counter_parity_wait_groups;

    std::array<std::optional<CounterParityRequirement>, kCounterCount> strongest_requirements;
    collect_counter_parity_requirements(group, consumer, consumer_section_offset,
                                        consumer_file_offset, arch, strongest_requirements);

    bool group_indeterminate = false;
    for (size_t counter_idx = 0; counter_idx < group.emitted_counts.size(); ++counter_idx) {
      if (!group.emitted_counts[counter_idx])
        continue;
      const auto counter = static_cast<WaitCounterKind>(counter_idx);
      const auto no_wait = counter_no_wait_value(arch, counter).value();
      if (!no_wait)
        continue;
      const uint32_t emitted_count = *group.emitted_counts[counter_idx];

      ++report_.counter_parity_fields_checked;
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
      bool boundary_policy_fallback = false;
      if (has_counter_parity_candidate(group.state_before, counter) &&
          (!requirement || requirement->required_count > emitted_count)) {
        bool lookahead_complete = true;
        constexpr size_t kMaxGeneralLookaheadPositions = 200000;
        constexpr size_t kMaxFunctionBoundaryLookaheadPositions = 256;
        const auto forward_requirement = find_forward_counter_parity_requirement(
            group, consumer, consumer_section_offset, counter, emitted_count, arch,
            function_boundary_policy ? kMaxFunctionBoundaryLookaheadPositions
                                     : kMaxGeneralLookaheadPositions,
            lookahead_complete);
        if (!lookahead_complete) {
          if (function_boundary_policy) {
            boundary_policy_fallback = true;
          } else {
            group_indeterminate = true;
            continue;
          }
        }
        if (forward_requirement &&
            (!requirement || forward_requirement->required_count < requirement->required_count)) {
          requirement = forward_requirement;
        }
        if (!function_boundary_policy && emitted_count != 0 &&
            (!requirement || requirement->required_count > emitted_count)) {
          const auto staged_requirement = find_staged_counter_parity_requirement(
              group, consumer, consumer_section_offset, counter, emitted_count, arch,
              lookahead_complete);
          if (!lookahead_complete) {
            group_indeterminate = true;
            continue;
          }
          if (staged_requirement &&
              (!requirement || staged_requirement->required_count < requirement->required_count)) {
            requirement = staged_requirement;
          }
        }
      }
      // A stronger boundary wait is ABI policy rather than evidence that LLVM
      // calculated a tighter dependency count, so keep it in the unmodeled
      // catalog. The fast path above intentionally permits an immediate exact
      // dependency to remain an ordinary parity match.
      // SIInsertWaitcnts reaches a fixed point over loop score brackets while
      // retaining waits inserted by an earlier iteration. At a cyclic block,
      // that process can conservatively strengthen a partial wait (often by
      // one, or all the way to zero) even when the final-ISA dependency has a
      // larger safe count. The encoded count is then loop policy, not a direct
      // dependency score. Exact loop waits remain ordinary parity matches.
      const bool cyclic_cfg_fixed_point_policy =
          has_loop_carried_events(counter, group.state_before, group.section_offset, arch);
      const bool needs_boundary_policy =
          function_boundary_policy &&
          (boundary_policy_fallback || !requirement || requirement->required_count > emitted_count);
      if (needs_boundary_policy || (cyclic_cfg_fixed_point_policy && requirement &&
                                    requirement->required_count > emitted_count)) {
        requirement.reset();
      }
      const uint32_t required_count = requirement ? requirement->required_count : *no_wait;
      if (requirement && required_count == emitted_count)
        ++report_.counter_parity_exact;
      if (required_count <= emitted_count)
        continue;

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
      message << "emitted " << wait_expression(counter, emitted_count, arch).value()
              << " is stronger than waitcheck requirement "
              << wait_expression(counter, required_count, arch).value() << " before "
              << diag.consumer_instruction;
      diag.message = message.str();
      record_counter_underaccounting(std::move(diag));
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
                                         is_non_entry_function_prolog(section_offset)};
      }
      if (!pending_group)
        return;
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
                                         is_non_entry_function_prolog(section_offset)};
      }
      for (size_t i = 0; i < embedded_fields->size(); ++i) {
        if (!(*embedded_fields)[i])
          continue;
        auto &combined = pending_group->emitted_counts[i];
        combined = combined ? std::min(*combined, *(*embedded_fields)[i]) : (*embedded_fields)[i];
      }
      if (!pending_group->instructions.empty())
        pending_group->instructions += "; ";
      pending_group->instructions += inst.disassemble();
      audit_counter_parity_group(*pending_group, inst, section_offset, file_offset, arch);
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
      audit_counter_parity_group(*pending_group, inst, section_offset, file_offset, arch);
    }
    pending_group.reset();
  }

  void finish_counter_parity_block(std::optional<PendingWaitGroup> &pending_group,
                                   rj_code_arch_t arch) {
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
          if (const auto fields = explicit_wait_fields(*consumer.instruction, arch).value()) {
            for (size_t i = 0; i < fields->size(); ++i) {
              overlapping_counter_wait |=
                  (*fields)[i].has_value() && pending_group->emitted_counts[i].has_value();
            }
          }
          if (!overlapping_counter_wait &&
              !embedded_wait_fields(*consumer.instruction, arch).value()) {
            const size_t saved_block_index = *current_cfg_view_index_;
            current_cfg_view_index_ = successor_index;
            audit_counter_parity_group(*pending_group, *consumer.instruction,
                                       consumer.section_offset,
                                       pending_group->file_offset - pending_group->section_offset +
                                           consumer.section_offset,
                                       arch);
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
    const SharedReachability reachable = blocks_reachable_from(block_index);
    return std::ranges::any_of(cfg_views_[block_index].predecessors, [&](size_t predecessor) {
      if (predecessor >= reachable->size() || !(*reachable)[predecessor])
        return false;
      const auto index = counter_index(counter);
      if (cfg_outputs_[predecessor].pending[index].empty() || at_wait.pending[index].empty())
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
                                           bool emit_diagnostics) {
    PendingState state = input;
    std::optional<PendingWaitGroup> pending_wait_group;
    uint64_t section_offset = block.start_offset();
    for (const Instruction &inst : block.instructions()) {
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
      finish_counter_parity_block(pending_wait_group, arch);
    return state;
  }

  [[nodiscard]] static bool expert_waits_enabled(const PendingState &state) {
    return !state.expert_scheduling.known || state.expert_scheduling.enabled;
  }

  static void clear_expert_wait_state(PendingState &state) {
    state.pending[counter_index(WaitCounterKind::VmVsrc)].clear();
    state.uncertain_order[counter_index(WaitCounterKind::VmVsrc)] = false;
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

  static void apply_expert_scheduling_mode(PendingState &state, const Instruction &inst,
                                           rj_code_arch_t arch) {
    if (arch != ROCJITSU_CODE_ARCH_CDNA5 && arch != ROCJITSU_CODE_ARCH_RDNA4)
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

  [[nodiscard]] static bool add_adjusted_source_use(InstDefUse &du, const Instruction &inst,
                                                    const Operand &op, int operand_index,
                                                    const VgprMsbState &state, rj_code_arch_t arch,
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
    const bool add_literal = inst.mnemonic() == "s_add_i32" || inst.mnemonic() == "s_add_u32";
    if (inst.mnemonic() == "s_mov_b32" || add_literal) {
      dst = register_operand(*inst.dst_operand(0), arch);
      const Operand &src = *inst.src_operand(0);
      if (const auto value = src.const_value()) {
        result = {{static_cast<uint32_t>(*value)}, 1};
      } else if (const auto reg = register_operand(src, arch)) {
        if (reg->cls == RegClass::M0)
          result = state.m0;
        else if (reg->cls == RegClass::SGPR) {
          const auto known = state.sgpr_offsets.find(reg->index);
          if (known != state.sgpr_offsets.end())
            result = known->second;
        }
      }
      if (add_literal) {
        const auto increment = inst.src_operand(1)->const_value();
        if (!increment)
          result = {};
        else
          for (unsigned n = 0; n < result.count; ++n)
            result.literals[n] += static_cast<uint32_t>(*increment);
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

  [[nodiscard]] InstDefUse inst_def_use_for_waitcheck(const Instruction &inst,
                                                      const VgprMsbState &state,
                                                      rj_code_arch_t arch, uint32_t wavefront_size,
                                                      const ScalarOffsetValues &m0 = {}) const {
    InstDefUse du(inst, wavefront_size);
    du.defs = {};
    du.uses = {};
    du.has_exec_masked_vector_def = false;
    du.has_predicated_def = inst.flags() & PREDICATED_DEF;

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
      if (add_adjusted_source_use(du, inst, *op, i, state, arch, wavefront_size))
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
    if (const auto partial = partial_d16_load_def(inst, arch))
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

  [[nodiscard]] static bool tracks_d16_subregisters(rj_code_arch_t arch) {
    // These families preserve the unselected half at memory writeback. VALU
    // operand uses remain conservative when their selection is unknown.
    return arch == ROCJITSU_CODE_ARCH_RDNA3 || arch == ROCJITSU_CODE_ARCH_RDNA3_5 ||
           arch == ROCJITSU_CODE_ARCH_RDNA4;
  }

  [[nodiscard]] static std::optional<PartialRegisterAccess>
  partial_d16_load_def(const Instruction &inst, rj_code_arch_t arch) {
    if (!tracks_d16_subregisters(arch) || !inst.is_memory_op())
      return std::nullopt;

    const std::string_view mnemonic = inst.mnemonic();
    if (mnemonic.find("load") == std::string_view::npos ||
        mnemonic.find("d16") == std::string_view::npos)
      return std::nullopt;

    // Packed format XY produces two half values in one full destination VGPR.
    // Image D16 results depend on the component mask and resource format.
    if (mnemonic.starts_with("image_") || mnemonic.find("format_xy") != std::string_view::npos ||
        mnemonic.find("format_d16_xy") != std::string_view::npos)
      return std::nullopt;

    const Operand *dst = inst.dst_operand(0);
    if (!dst)
      return std::nullopt;
    auto ref = dst->to_register_ref();
    if (!ref || ref->cls != RegClass::VGPR || ref->width != 1)
      return std::nullopt;

    const uint8_t mask =
        mnemonic.find("d16_hi") != std::string_view::npos ? kVgprHigh16Mask : kVgprLow16Mask;
    return PartialRegisterAccess{*ref, mask};
  }

  [[nodiscard]] static bool is_low_half_memory_store(std::string_view mnemonic) {
    return mnemonic.find("store_b8") != std::string_view::npos ||
           mnemonic.find("store_b16") != std::string_view::npos ||
           mnemonic.find("store_byte") != std::string_view::npos ||
           mnemonic.find("store_short") != std::string_view::npos ||
           mnemonic.find("write_b8") != std::string_view::npos ||
           mnemonic.find("write_b16") != std::string_view::npos;
  }

  [[nodiscard]] static std::optional<PartialRegisterAccess>
  partial_memory_store_use(const Instruction &inst, rj_code_arch_t arch) {
    if (!tracks_d16_subregisters(arch) || !inst.is_memory_op())
      return std::nullopt;

    const std::string_view mnemonic = inst.mnemonic();
    if (mnemonic.find("store") == std::string_view::npos &&
        mnemonic.find("write") == std::string_view::npos)
      return std::nullopt;

    uint8_t mask = kVgprFull32Mask;
    if (mnemonic.find("d16_hi") != std::string_view::npos)
      mask = kVgprHigh16Mask;
    else if (is_low_half_memory_store(mnemonic))
      mask = kVgprLow16Mask;
    else
      return std::nullopt;

    const int data_operand_index = starts_with(mnemonic, "buffer_store") ||
                                           starts_with(mnemonic, "tbuffer_store") ||
                                           starts_with(mnemonic, "image_store")
                                       ? 0
                                       : 1;
    const Operand *data = inst.src_operand(data_operand_index);
    if (!data)
      return std::nullopt;
    auto ref = data->to_register_ref();
    if (!ref || ref->cls != RegClass::VGPR || ref->width != 1)
      return std::nullopt;
    return PartialRegisterAccess{*ref, mask};
  }

  [[nodiscard]] static uint8_t pending_register_mask(const PendingEvent &event, RegisterRef ref) {
    return event.partial_reg && *event.partial_reg == ref ? event.partial_reg_mask
                                                          : kVgprFull32Mask;
  }

  [[nodiscard]] static uint8_t instruction_register_mask(const Instruction &inst, RegisterRef ref,
                                                         WaitcheckAccessKind access,
                                                         rj_code_arch_t arch) {
    const auto partial = access == WaitcheckAccessKind::Def   ? partial_d16_load_def(inst, arch)
                         : access == WaitcheckAccessKind::Use ? partial_memory_store_use(inst, arch)
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
                 (pending_register_mask(event, ref) &
                  instruction_register_mask(inst, ref, access, arch)) != 0)
        result = ref;
    });
    return result;
  }

  void apply_implicit_xcnt_ordering(PendingState &state, const Instruction &inst,
                                    const InstDefUse &du,
                                    std::span<const ClassifiedEvent> current_events,
                                    rj_code_arch_t arch) const {
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
    const bool pending_vmem = std::ranges::any_of(pending, is_xcnt_vmem_event);
    const bool pending_smem = std::ranges::any_of(pending, is_xcnt_smem_event);

    // Hardware places an implicit X_CNT drain between interleaved SMEM and
    // VMEM translations. LLVM models the switch by retaining only the new
    // group's event.
    if ((current_vmem && pending_smem) || (current_smem && pending_vmem)) {
      apply_xcnt_group_wait(state, 0);
      return;
    }

    if (!current_vmem)
      return;

    // VMEM address translations are ordered. If this VMEM instruction would
    // overwrite a source of an older VMEM operation, LLVM internally applies
    // just the X_CNT wait required for that dependency instead of emitting an
    // instruction. Preserve younger, unrelated translations.
    std::optional<uint32_t> required_count;
    for (const PendingEvent &event : state.pending[idx]) {
      if (!is_xcnt_vmem_event(event) ||
          !first_dependency_intersection(event, inst, du, WaitcheckAccessKind::Def, arch, state.m0))
        continue;
      required_count = required_count ? std::min(*required_count, event.min_younger)
                                      : std::optional<uint32_t>(event.min_younger);
    }
    if (required_count)
      apply_xcnt_group_wait(state, *required_count);
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

  void emit_diagnostic(const Instruction &inst, const PendingEvent &event, RegisterRef reg,
                       WaitcheckAccessKind access, uint32_t required_count, uint64_t section_offset,
                       uint64_t file_offset, rj_code_arch_t arch) {
    if (!should_emit_cfg_diagnostic(event, inst, reg, section_offset, arch))
      return;

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
      const auto hazard_it = state.hazards.find(ref.index);
      if (hazard_it == state.hazards.end())
        return;
      const VaVdstHazard &hazard = hazard_it->second;

      const uint32_t required_wait = hazard.trans_since ? 0u : hazard.age;
      if (*encoded_wait <= required_wait)
        return;

      emit_va_vdst_hazard_diagnostic(inst, hazard, ref, *encoded_wait, required_wait,
                                     section_offset, file_offset);
      emitted = true;
    });
  }

  static void age_va_vdst_hazards(VaVdstHazardState &state, bool trans_seen) {
    constexpr uint8_t kNoHazardWaitStates = 15;
    for (auto it = state.hazards.begin(); it != state.hazards.end();) {
      VaVdstHazard &hazard = it->second;
      if (hazard.age < kNoHazardWaitStates)
        ++hazard.age;
      if (hazard.age >= kNoHazardWaitStates) {
        it = state.hazards.erase(it);
        continue;
      }
      hazard.trans_since = hazard.trans_since || trans_seen;
      ++it;
    }
  }

  static void clear_va_vdst_hazards(VaVdstHazardState &state) { state.hazards.clear(); }

  static void set_va_vdst_hazard_for_regs(VaVdstHazardState &state, const RegisterSet &regs,
                                          bool trans_seen, const SgprHazardProducer &producer) {
    regs.for_each([&](RegisterRef ref) {
      if (ref.cls != RegClass::VGPR || ref.index >= REGISTER_SET_MAX_VGPRS)
        return;
      state.hazards.insert_or_assign(ref.index, VaVdstHazard{0, trans_seen, producer});
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
  [[nodiscard]] SharedReachability cache_reachability(Cache &cache, size_t block_index,
                                                      Reachability reachable) {
    const size_t bytes = reachable.size() * sizeof(Reachability::value_type);
    auto result = std::make_shared<const Reachability>(std::move(reachable));
    if (bytes <= options_.max_reachability_cache_bytes -
                     std::min(reachability_cache_bytes_, options_.max_reachability_cache_bytes)) {
      reachability_cache_bytes_ += bytes;
      cache.emplace(block_index, result);
    }
    return result;
  }

  [[nodiscard]] SharedReachability blocks_reaching(size_t target_block_index) {
    if (auto cached = reverse_reachability_cache_.find(target_block_index);
        cached != reverse_reachability_cache_.end())
      return cached->second;

    Reachability reachable(cfg_views_.size());
    std::vector<size_t> worklist{target_block_index};
    while (!worklist.empty()) {
      const size_t block_index = worklist.back();
      worklist.pop_back();
      if (reachable[block_index] != 0)
        continue;
      reachable[block_index] = 1;
      for (size_t predecessor : cfg_views_[block_index].predecessors)
        worklist.push_back(predecessor);
    }
    return cache_reachability(reverse_reachability_cache_, target_block_index,
                              std::move(reachable));
  }

  [[nodiscard]] SharedReachability blocks_reachable_from(size_t source_block_index) {
    if (auto cached = forward_reachability_cache_.find(source_block_index);
        cached != forward_reachability_cache_.end())
      return cached->second;

    Reachability reachable(cfg_views_.size());
    std::vector<size_t> worklist{source_block_index};
    while (!worklist.empty()) {
      const size_t block_index = worklist.back();
      worklist.pop_back();
      if (reachable[block_index] != 0)
        continue;
      reachable[block_index] = 1;
      for (size_t successor : cfg_views_[block_index].successors)
        worklist.push_back(successor);
    }
    return cache_reachability(forward_reachability_cache_, source_block_index,
                              std::move(reachable));
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

  void apply_path_pre_dependency_waits(const Instruction &inst, const PendingEvent &event,
                                       rj_code_arch_t arch, bool &pending, uint32_t &age) {
    if (!pending)
      return;
    // Most path instructions cannot retire an event. Avoid constructing and
    // copying a full pending state merely to apply an identity transition.
    const auto wait_vm_vsrc = dsdir_wait_vm_vsrc(inst);
    if (!inst.is_waitcnt() && !releases_gfx10_scalar_sources(inst, arch) &&
        !vinterp_wait_exp(inst) && (!wait_vm_vsrc || *wait_vm_vsrc != 0))
      return;

    PendingEvent path_event = event;
    const auto ordering = path_wait_facts_.find(&inst);
    const bool ordered =
        ordering != path_wait_facts_.end() &&
        !(ordering->second.unordered & (uint64_t{1} << counter_index(event.counter)));
    path_event.min_younger = is_completion_counter(event.counter) || ordered ? age : 0;
    PendingState state;
    path_event.min_younger_primary = 0;
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

    if (inst.is_waitcnt()) {
      (void)apply_waitcnt(state, inst, arch);
    } else {
      (void)apply_embedded_waitcnt(state, inst, arch);
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
    SccPredicate::Kind kind;
    if (inst.mnemonic() == "s_cmp_eq_u32")
      kind = SccPredicate::Kind::EqImm;
    else if (inst.mnemonic() == "s_cmp_ge_u32")
      kind = SccPredicate::Kind::GeUnsignedImm;
    else
      return std::nullopt;
    const Operand *lhs = inst.src_operand(0);
    const Operand *rhs = inst.src_operand(1);
    if (lhs == nullptr || rhs == nullptr)
      return std::nullopt;

    auto lhs_ref = lhs->to_register_ref();
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
    case SccPredicate::Kind::GeUnsignedImm:
      if (branch_condition_true) {
        if (!constraint.unsigned_min || *constraint.unsigned_min < predicate->value)
          constraint.unsigned_min = predicate->value;
      } else {
        if (predicate->value == 0)
          return false;
        const uint32_t upper_bound = predicate->value - 1;
        if (!constraint.unsigned_max || *constraint.unsigned_max > upper_bound)
          constraint.unsigned_max = upper_bound;
      }
      return is_satisfiable();
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

  struct PathMasks {
    // Nonempty is weaker than full. A partially populated workgroup can start
    // with only some lanes active.
    std::optional<bool> exec;
    std::optional<bool> vcc;
    bool operator==(const PathMasks &) const = default;
  };

  // Reuse equalities from the SCC path filter. Only literals and SGPRs feeding
  // a mask operation are materialized, so unrelated scalar computations do not
  // multiply the search states.
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
                                const ScalarConstraints &facts, uint32_t wave_size) {
    const Instruction &inst = *view.instruction;
    // A summarized call may change every mask and scalar that describes it.
    if (inst.flags() & INDIRECT_CALL) {
      masks = {};
      return;
    }
    const Operand *dst = inst.dst_operand(0);
    if (view.invalidates_vcc) {
      std::optional<bool> next;
      if (wave_size == 32 && inst.mnemonic() == "s_and_b32" &&
          low_selector(dst, amdgpu::kVccSelectorFirst)) {
        for (unsigned i = 0; i < 2; ++i) {
          if (!low_selector(inst.src_operand(i), amdgpu::kExecSelectorFirst))
            continue;
          const auto mask = path_dword(inst.src_operand(1 - i), facts);
          if (masks.exec == false || mask == 0)
            next = false;
          else if (mask == UINT32_MAX)
            next = masks.exec;
        }
      }
      masks.vcc = next;
    }
    if (view.invalidates_exec) {
      masks.exec.reset();
      if (low_selector(dst, amdgpu::kExecSelectorFirst)) {
        if (inst.mnemonic() == "s_mov_b64") {
          if (const auto value = inst.src_operand(0)->const_value())
            masks.exec = (wave_size == 32 ? static_cast<uint32_t>(*value) : *value) != 0;
        } else if (wave_size == 32 && inst.mnemonic() == "s_mov_b32") {
          if (const auto value = path_dword(inst.src_operand(0), facts))
            masks.exec = *value != 0;
        }
      }
    }
  }

  [[nodiscard]] bool has_cfg_path_with_event_pending(const PendingEvent &event,
                                                     uint64_t consumer_offset, rj_code_arch_t arch,
                                                     bool before_target = false,
                                                     bool requires_active_lane = false) {
    if (cfg_views_.empty())
      return true;

    // A guarded barrier arrive ages an older ASYNC load only on a nonempty
    // EXEC edge. Its LDS memory-order dependency has no masked register operand.
    const bool track_async_issue =
        event.counter == WaitCounterKind::Async && cfg_has_exec_guards_ &&
        (event.kind == WaitEventKind::AsyncLdsLoad || event.kind == WaitEventKind::AsyncBarrier);

    // Every consumer in a neutral run observes the same pending-event state.
    // Reuse one path query for the run instead of repeating the CFG search for
    // every arithmetic use of the same asynchronous load. A producer between
    // the run's start and the consumer, or additional overlapping contexts,
    // prevents this equivalence. Producers later in the block are safe too:
    // their loop-carried state is unchanged throughout this prefix of the run.
    if (!before_target && !requires_active_lane) {
      const auto query = path_query_starts_.find(consumer_offset);
      if (query != path_query_starts_.end() && query->second.offset < consumer_offset &&
          (event.section_offset < query->second.offset || event.section_offset > consumer_offset)) {
        const auto start = path_query_starts_.find(query->second.offset);
        if (start != path_query_starts_.end() && start->second.contexts == query->second.contexts)
          consumer_offset = query->second.offset;
      }
    }
    using CacheKey = std::tuple<uint64_t, uint64_t, WaitCounterKind, bool, bool>;
    const CacheKey cache_key{event.section_offset, consumer_offset, event.counter, before_target,
                             requires_active_lane};
    if (auto cached = feasible_path_cache_.find(cache_key); cached != feasible_path_cache_.end())
      return cached->second;

    // The dataflow state passed to the current instruction already proves
    // that an earlier event in the same straight-line block is still pending.
    // Avoid launching the path-sensitive CFG search for the overwhelmingly
    // common compiler pattern of a wait shortly after its producer.
    if (!requires_active_lane && !track_async_issue && current_cfg_view_index_ &&
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
    if (before_target && !track_async_issue && current_cfg_view_index_ &&
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
    if (producer_block_indices.empty()) {
      feasible_path_cache_[cache_key] = true;
      return true;
    }
    if (consumer_block_indices.empty()) {
      feasible_path_cache_[cache_key] = true;
      return true;
    }

    std::vector<uint8_t> can_reach_consumer(cfg_views_.size());
    for (size_t consumer_block_index : consumer_block_indices) {
      const SharedReachability reachable = blocks_reaching(consumer_block_index);
      for (size_t i = 0; i < reachable->size(); ++i)
        can_reach_consumer[i] = static_cast<uint8_t>(can_reach_consumer[i] | (*reachable)[i]);
    }
    std::vector<uint8_t> can_reach_producer(cfg_views_.size());
    std::vector<uint8_t> reachable_from_producer(cfg_views_.size());
    for (size_t producer_block_index : producer_block_indices) {
      const SharedReachability reaching = blocks_reaching(producer_block_index);
      const SharedReachability reachable = blocks_reachable_from(producer_block_index);
      for (size_t i = 0; i < cfg_views_.size(); ++i) {
        can_reach_producer[i] = static_cast<uint8_t>(can_reach_producer[i] | (*reaching)[i]);
        reachable_from_producer[i] =
            static_cast<uint8_t>(reachable_from_producer[i] | (*reachable)[i]);
      }
    }
    const bool producer_can_reach_consumer =
        std::ranges::any_of(producer_block_indices, [&](size_t producer_block_index) {
          return can_reach_consumer[producer_block_index] != 0;
        });
    if (!producer_can_reach_consumer) {
      feasible_path_cache_[cache_key] = false;
      return false;
    }

    // Only scalar predicates repeated after the producer can correlate the path
    // that selected a producer with the path to its consumer. Tracking unrelated
    // comparisons across large dispatch CFGs creates a combinatorial number of
    // equivalent search states without improving feasibility precision.
    using PredicateKey = std::tuple<SccPredicate::Kind, uint16_t, uint32_t>;
    std::set<PredicateKey> predicates_before_producer;
    std::set<PredicateKey> predicates_after_producer;
    // The CFG is immutable during these searches. Index scalar comparisons
    // once rather than re-decoding every instruction for every event pair.
    for (const auto &[block_index, section_offset, predicate] : path_predicates_) {
      const PredicateKey key{predicate.kind, predicate.sgpr, predicate.value};
      if (can_reach_producer[block_index] != 0 && section_offset < event.section_offset)
        predicates_before_producer.insert(key);
      if (reachable_from_producer[block_index] != 0 && can_reach_consumer[block_index] != 0 &&
          section_offset >= event.section_offset)
        predicates_after_producer.insert(key);
    }
    std::set<PredicateKey> relevant_predicates;
    std::ranges::set_intersection(predicates_before_producer, predicates_after_producer,
                                  std::inserter(relevant_predicates, relevant_predicates.end()));
    // Scalar memory is unordered, so a zero wait discharges its result on a
    // particular path. Wave32 EXEC_LO and VCC_LO each describe the whole wave.
    const bool track_masks = requires_active_lane || track_async_issue ||
                             (!before_target && event.kind == WaitEventKind::Smem &&
                              event.counter != WaitCounterKind::X && current_kernel_ &&
                              wavefront_size_ == 32 && cfg_has_mask_branches_);
    if (relevant_predicates.empty() && !track_masks) {
      // Without a supported repeated scalar predicate there is no correlation
      // for the path filter to prove. The converged dataflow state already
      // established a structural path on which this event remains pending.
      feasible_path_cache_[cache_key] = true;
      return true;
    }

    struct SearchState {
      size_t block_index = 0;
      size_t inst_index = 0;
      bool pending = false;
      uint32_t age = 0;
      ScalarConstraints constraints;
      std::optional<SccPredicate> scc_predicate;
      std::optional<uint32_t> constraint_id = std::nullopt;
      PathMasks masks{};
    };

    constexpr uint32_t kMaxTrackedAge = 255;
    constexpr size_t kMaxPathSearchStates = 200000;
    std::vector<SearchState> worklist;
    for (size_t i = 0; i < cfg_views_.size(); ++i) {
      const BasicBlock *block = cfg_views_[i].block;
      if (track_masks && current_kernel_) {
        // A descriptor-backed kernel is analyzed individually; its first
        // context is its dispatch entry even if that entry has a backedge.
        if (i == 0 && can_reach_producer[i] != 0)
          worklist.push_back(
              {i, 0, false, 0, {}, std::nullopt, std::nullopt, {true, std::nullopt}});
      } else if (block != nullptr && cfg_views_[i].predecessors.empty() &&
                 can_reach_producer[i] != 0) {
        worklist.push_back({i, 0, false, 0, {}, std::nullopt});
      }
    }
    if (worklist.empty()) {
      for (size_t producer_block_index : producer_block_indices)
        worklist.push_back({producer_block_index, 0, false, 0, {}, std::nullopt});
    }
    struct SearchKey {
      size_t block_index;
      size_t inst_index;
      bool pending;
      uint32_t age;
      uint32_t constraint_id;
      bool operator==(const SearchKey &) const = default;
    };
    struct SearchKeyHash {
      size_t operator()(const SearchKey &key) const {
        size_t hash = key.block_index;
        for (size_t value : {key.inst_index, static_cast<size_t>(key.pending),
                             static_cast<size_t>(key.age), static_cast<size_t>(key.constraint_id)})
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
      state.age = std::min(state.age, kMaxTrackedAge);
      if (!state.constraint_id) {
        auto key = constraints_key(state.constraints, state.scc_predicate);
        if (track_masks) {
          for (auto value : {state.masks.exec, state.masks.vcc})
            key.push_back(!value ? '?' : *value ? '1' : '0');
        }
        const auto [entry, inserted] = constraint_ids.try_emplace(
            std::move(key), static_cast<uint32_t>(constraint_ids.size()));
        state.constraint_id = entry->second;
      }
      if (!visited
               .insert({state.block_index, state.inst_index, state.pending, state.age,
                        *state.constraint_id})
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
        for (size_t successor : block_view.successors) {
          if (can_continue_from(successor, state.pending))
            worklist.push_back({successor, 0, state.pending, state.age, state.constraints,
                                state.scc_predicate, state.constraint_id, state.masks});
        }
        continue;
      }

      const CfgInstructionView &inst_view = block_view.instructions[state.inst_index];
      size_t run_end = inst_view.neutral_run_end;
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
            feasible_path_cache_[cache_key] = true;
            return true;
          }
          states_processed += additional_states;
          state.inst_index = run_end;
          worklist.push_back(std::move(state));
          continue;
        }
      }
      const Instruction &inst = *inst_view.instruction;
      if (before_target && inst_view.section_offset == consumer_offset && state.pending) {
        feasible_path_cache_[cache_key] = true;
        return true;
      }
      apply_path_pre_dependency_waits(inst, event, arch, state.pending, state.age);

      if (!before_target && inst_view.section_offset == consumer_offset && state.pending &&
          !(requires_active_lane && state.masks.exec == false)) {
        feasible_path_cache_[cache_key] = true;
        return true;
      }

      if (inst_view.section_offset == event.section_offset) {
        // A masked vector issue with no participating lanes cannot create a
        // register result. A pending result from an earlier iteration survives.
        const auto *issue = inst.amdgpu_memory_issue_info();
        if (!(requires_active_lane && state.masks.exec == false && issue && issue->exec_masked)) {
          state.pending = true;
          state.age = 0;
        } else if (state.pending) {
          state.age = 0;
        }
      } else if (state.pending && !inst.is_waitcnt() && is_program_end(inst.mnemonic())) {
        state.pending = false;
        state.age = 0;
      } else if (state.pending && !inst.is_waitcnt() &&
                 !(requires_active_lane && state.masks.exec != true &&
                   inst.amdgpu_memory_issue_info() &&
                   inst.amdgpu_memory_issue_info()->exec_masked) &&
                 (inst_view.issued_counters & (uint64_t{1} << counter_index(event.counter))) != 0) {
        if (is_completion_counter(event.counter)) {
          const auto order = completion_order_kind(event.counter, event.kind, arch);
          const auto current = classify_events(inst, arch).value();
          for (const auto &issued : current) {
            if (issued.counter == event.counter && order &&
                guaranteed_issue_order(issued.counter, issued.kind, arch,
                                       state.masks.exec == true) == order)
              state.age = std::min(state.age + issue_units(inst, issued, arch), kMaxTrackedAge);
          }
        } else if (event.counter != WaitCounterKind::X ||
                   (!is_async_transfer_kind(event.kind) && !is_async_lds_load(inst.mnemonic()) &&
                    !is_async_lds_store(inst.mnemonic()))) {
          state.age = std::min<uint32_t>(state.age + 1, kMaxTrackedAge);
        }
      }

      const auto previous_predicate = state.scc_predicate;
      bool constraints_changed = false;
      std::optional<uint32_t> assigned_value;
      std::optional<RegisterRef> assigned_reg;
      if (track_masks) {
        const auto previous_masks = state.masks;
        update_path_masks(state.masks, inst_view, state.constraints, wavefront_size_);
        if (state.masks != previous_masks)
          state.constraint_id.reset();
        if (inst.flags() & INDIRECT_CALL) {
          state.constraints.clear();
          state.scc_predicate.reset();
          state.constraint_id.reset();
        }
        if (inst.mnemonic() == "s_mov_b32" && inst.dst_operand(0)) {
          assigned_reg = inst.dst_operand(0)->to_register_ref();
          if (assigned_reg && assigned_reg->cls == RegClass::SGPR &&
              path_mask_inputs_.contains(assigned_reg->index))
            assigned_value = path_dword(inst.src_operand(0), state.constraints);
        }
      }
      for (uint16_t reg : path_scalar_defs_.at(&inst)) {
        constraints_changed |= state.constraints.erase(reg) != 0;
        if (state.scc_predicate && state.scc_predicate->sgpr == reg)
          state.scc_predicate.reset();
      }

      if (assigned_value) {
        state.constraints[assigned_reg->index].equal = *assigned_value;
        constraints_changed = true;
      }
      if (const auto branch = mask_branch_class(inst.mnemonic()); track_masks && branch) {
        const auto nonempty = *branch == RegClass::EXEC ? state.masks.exec : state.masks.vcc;
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
                mask = edge_nonempty;
                successor.constraint_id.reset();
              }
              worklist.push_back(std::move(successor));
            }
            continue;
          }
        }
      }
      if (inst_view.predicate_index != std::numeric_limits<size_t>::max()) {
        const auto &predicate = std::get<2>(path_predicates_[inst_view.predicate_index]);
        if (relevant_predicates.contains({predicate.kind, predicate.sgpr, predicate.value}))
          state.scc_predicate = predicate;
        else
          state.scc_predicate.reset();
      } else if (is_scc_conditional_branch(inst.mnemonic())) {
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
              if (apply_scc_branch_constraint(taken_constraints, state.scc_predicate,
                                              branch_takes_on_scc_true)) {
                worklist.push_back({successor, 0, state.pending, state.age,
                                    std::move(taken_constraints), std::nullopt, std::nullopt,
                                    state.masks});
              }
            }
          }
        }

        ScalarConstraints fallthrough_constraints = state.constraints;
        const bool fallthrough_on_scc_true = inst.mnemonic() == "s_cbranch_scc0";
        if (apply_scc_branch_constraint(fallthrough_constraints, state.scc_predicate,
                                        fallthrough_on_scc_true)) {
          bool found_fallthrough = false;
          for (size_t successor : block_view.successors) {
            const BasicBlock *successor_block = cfg_views_[successor].block;
            if (successor_block == nullptr || successor_block->start_offset() != fallthrough_offset)
              continue;
            found_fallthrough = true;
            if (can_continue_from(successor, state.pending)) {
              worklist.push_back({successor, 0, state.pending, state.age, fallthrough_constraints,
                                  std::nullopt, std::nullopt, state.masks});
            }
          }
          if (!found_fallthrough && state.inst_index + 1 < block_view.instructions.size()) {
            worklist.push_back({state.block_index, state.inst_index + 1, state.pending, state.age,
                                std::move(fallthrough_constraints), std::nullopt, std::nullopt,
                                state.masks});
          }
        }
        continue;
      } else if (inst_view.invalidates_scc) {
        state.scc_predicate.reset();
      }

      if (constraints_changed || previous_predicate != state.scc_predicate)
        state.constraint_id.reset();
      worklist.push_back({state.block_index, state.inst_index + 1, state.pending, state.age,
                          std::move(state.constraints), state.scc_predicate, state.constraint_id,
                          state.masks});
    }

    feasible_path_cache_[cache_key] = false;
    return false;
  }

  [[nodiscard]] bool should_emit_cfg_diagnostic(const PendingEvent &event,
                                                const Instruction &consumer, RegisterRef reg,
                                                uint64_t consumer_offset, rj_code_arch_t arch) {
    const auto mnemonic = consumer.mnemonic();
    const auto *issue = consumer.amdgpu_memory_issue_info();
    // Lane transfers and matrix instructions may inspect inactive VGPRs.
    // Scalar operands and unconditional memory accesses retain their checks.
    const bool masked = current_kernel_ && cfg_has_exec_guards_ && is_exec_masked_def(reg) &&
                        !(consumer.flags() & (IGNORES_EXEC | MATRIX_REGISTER_ACCESSES)) &&
                        !mnemonic.starts_with("v_readlane") &&
                        !mnemonic.starts_with("v_readfirstlane") &&
                        !mnemonic.starts_with("v_writelane") &&
                        (is_vector_alu(consumer) || (issue && issue->exec_masked));
    return has_cfg_path_with_event_pending(event, consumer_offset, arch, false, masked);
  }

  [[nodiscard]] bool event_can_reach_wait(const PendingEvent &event, uint64_t wait_offset,
                                          rj_code_arch_t arch) {
    return has_cfg_path_with_event_pending(event, wait_offset, arch, true);
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
                                                const Instruction &consumer,
                                                rj_code_arch_t arch) {
    if (!current_cfg_view_index_ || cfg_outputs_.empty() || current_function_entry_.has_value() ||
        !relative_entry_proven_)
      return false;
    const auto &block = cfg_views_[*current_cfg_view_index_];
    if (block.predecessors.size() < 2 || block.predecessors.size() > 8 ||
        block.instructions.size() > 256 || (event.section_offset >= block.block->start_offset() &&
         event.section_offset < block.block->end_offset()))
      return false;
    const auto position = std::ranges::find(block.instructions, &consumer,
                                            &CfgInstructionView::instruction);
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
      if ((event.check_uses && first_dependency_intersection(event, consumer, operands,
                               WaitcheckAccessKind::Use, arch, incoming.m0)) ||
          (event.check_defs && first_dependency_intersection(event, consumer, operands,
                               WaitcheckAccessKind::Def, arch, incoming.m0)))
        return false;
    }
    return reached;
  }

  template <typename Visitor>
  void visit_dependencies(const PendingState &state, const Instruction &inst, const InstDefUse &du,
                          std::span<const ClassifiedEvent> current_events, rj_code_arch_t arch,
                          Visitor &&visit) {
    for (size_t counter_idx = 0; counter_idx < state.pending.size(); ++counter_idx) {
      const auto &events = state.pending[counter_idx];
      for (size_t i = 0; i < events.size(); ++i) {
        const PendingEvent &event = events[i];
        if (!checks_scalar_source(event.kind, inst))
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

  void check_sgpr_control_transfer_hazards(const SgprHazardState &state, const Instruction &inst,
                                           uint64_t section_offset, uint64_t file_offset) {
    // One all-zero depctr field repairs every register in that hazard class.
    // Emit its lowest-numbered representative instead of one diagnostic per
    // outstanding lane; VCC represents the field only when no SGPR lane does.
    if (const auto lane = first_sgpr_hazard_lane(state.salu_hazards)) {
      const auto producer = state.salu_producers.find(*lane);
      emit_sgpr_hazard_diagnostic(
          inst, producer == state.salu_producers.end() ? nullptr : &producer->second,
          RegisterRef{RegClass::SGPR, *lane, 1}, "depctr_sa_sdst", section_offset, file_offset,
          WaitcheckAccessKind::ControlTransfer);
    } else if ((state.vcc_hazard & kSgprHazardSalu) != 0) {
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
    if ((state.vcc_hazard & kSgprHazardValu) != 0) {
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

  void add_event(PendingState &state, ClassifiedEvent classification, const Instruction &inst,
                 const InstDefUse &du, std::string section_name, uint64_t section_offset,
                 uint64_t file_offset, std::string instruction, rj_code_arch_t arch,
                 bool record_stats) {
    PendingEvent event;
    event.counter = classification.counter;
    event.kind = classification.kind;
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
      if (const auto partial = partial_d16_load_def(inst, arch);
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
    age_events(state, event, arch, maximum_dependency_wait(arch, event.counter).value(),
               issue_units(inst, classification, arch));
    if (record_stats)
      ++report_.memory_events_tracked;
    if (is_counter_token_only(event))
      return;
    auto position = std::ranges::lower_bound(state.pending[idx], event, event_identity_less);
    if (position != state.pending[idx].end() && same_event_identity(*position, event)) {
      *position = std::move(event);
    } else {
      state.pending[idx].insert(position, std::move(event));
    }
  }

  void analyze_instruction(PendingState &state, const Instruction &inst,
                           const std::string &section_name, uint64_t section_offset,
                           uint64_t file_offset, rj_code_arch_t arch, bool emit_diagnostics) {
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
      (void)apply_waitcnt(state, inst, arch);
      state.previous_vm_vsrc_zero_wait = current_vm_vsrc_zero_wait;
      finish_instruction();
      return;
    }
    if (apply_vgpr_msb_mode(state, inst, arch, immediately_after_mode_setreg)) {
      finish_instruction();
      return;
    }
    apply_expert_scheduling_mode(state, inst, arch);
    clear_matching_barrier_scc_write(state, inst);
    // LLVM's AMDGPUWaitSGPRHazards pass models this RDNA4 hardware hazard
    // independently of expert scheduling mode. Do not couple SA_SDST,
    // VA_SDST, or VA_VCC tracking to the expert-only VM/VALU counters.
    if (tracks_gfx12_sgpr_hazards(arch))
      apply_sgpr_hazard_memory_cull(state.sgpr_hazards, inst.mnemonic());
    (void)apply_embedded_waitcnt(state, inst, arch);

    InstDefUse du =
        inst_def_use_for_waitcheck(inst, state.vgpr_msb, arch, wavefront_size_, state.m0);
    auto events = classify_events(inst, arch).value();
    apply_admission(state, inst, events, arch);
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
    update_scalar_offsets(state, inst, du, arch);
    if (flat_issues_both_counters(arch) &&
        (writes_exec_mask(inst) || du.defs.contains({RegClass::EXEC, 0, 1}) ||
         (inst.flags() & INDIRECT_CALL))) {
      if (arch == ROCJITSU_CODE_ARCH_RDNA3 || arch == ROCJITSU_CODE_ARCH_RDNA3_5)
        for (auto &pending : state.pending[counter_index(WaitCounterKind::Ds)])
          pending.implies_active_exec = false;
      state.exec_nonzero = false;
      if ((inst.mnemonic() == "s_mov_b32" || inst.mnemonic() == "s_mov_b64") &&
          low_selector(inst.dst_operand(0), amdgpu::kExecSelectorFirst)) {
        const auto value = inst.src_operand(0)->const_value();
        state.exec_nonzero = value && static_cast<uint32_t>(*value) != 0;
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

  WaitcheckReport &report_;
  WaitcheckOptions options_;
  std::unique_ptr<Decoder> decoder_;
  bool track_sgpr_offsets_ = true;
  bool setreg_vgpr_msb_fixup_ = false;
  const WaitcheckKernelInfo *current_kernel_ = nullptr;
  uint32_t wavefront_size_ = 64;
  std::set<
      std::tuple<WaitCounterKind, WaitcheckAccessKind, RegClass, uint16_t, uint16_t, std::string,
                 uint64_t, uint64_t, uint64_t, uint64_t, uint32_t, std::string, std::string>>
      diagnostic_keys_;
  std::set<std::tuple<std::string, WaitCounterKind, uint64_t, uint64_t, uint64_t, uint32_t,
                      uint32_t, bool>>
      counter_underaccounting_keys_;
  std::unordered_map<const Instruction *, std::vector<uint16_t>> path_scalar_defs_;
  std::set<uint16_t> path_mask_inputs_;
  bool cfg_has_mask_branches_ = false;
  bool cfg_has_exec_guards_ = false;
  std::vector<std::tuple<size_t, uint64_t, SccPredicate>> path_predicates_;
  std::unordered_map<uint64_t, PathQueryStart> path_query_starts_;
  struct PathWaitFacts {
    uint64_t unordered = 0;
    uint64_t flat_load_routes = 0;
    uint64_t flat_store_routes = 0;
  };
  std::unordered_map<const Instruction *, PathWaitFacts> path_wait_facts_;
  std::vector<CfgBlockView> cfg_views_;
  std::optional<size_t> current_cfg_view_index_;
  std::span<const PendingState> cfg_outputs_;
  bool relative_entry_proven_ = false;
  std::optional<uint64_t> current_function_entry_;
  std::optional<uint16_t> current_call_return_sreg_;
  std::map<std::tuple<uint64_t, uint64_t, WaitCounterKind, bool, bool>, bool> feasible_path_cache_;
  std::map<size_t, SharedReachability> reverse_reachability_cache_;
  std::map<size_t, SharedReachability> forward_reachability_cache_;
  size_t reachability_cache_bytes_ = 0;
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
  return {std::move(name),
          descriptor_vaddr,
          entry_offset,
          code_size,
          wave,
          {},
          vgprs,
          vgprs,
          sgprs,
          sgprs};
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

  uint32_t elf_machine = 0;
  if (code_object.image_size() >= sizeof(Elf64_Ehdr)) {
    Elf64_Ehdr ehdr{};
    std::memcpy(&ehdr, code_object.image_data(), sizeof(ehdr));
    if (std::memcmp(ehdr.e_ident, EI_MAGIC, EI_MAGIC_SIZE) == 0) {
      if (ehdr.e_type == ET_REL) {
        report.supported = false;
        report.analysis_error = "relocatable AMDGPU objects are not final loadable code objects";
        return report;
      }
      if (ehdr.e_machine == EM_AMDGPU)
        elf_machine = ehdr.e_flags & EF_AMDGPU_MACH;
    }
  }

  Analyzer analyzer(report, options, elf_machine);
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
