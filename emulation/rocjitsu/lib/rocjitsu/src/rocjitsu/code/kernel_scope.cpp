// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/kernel_scope.h"

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/code/analysis/control_flow.h"
#include "rocjitsu/code/analysis/def_use_chain.h"
#include "rocjitsu/code/basic_block.h"
#include "rocjitsu/code/kernel_descriptor_scan.h"
#include "rocjitsu/isa/arch/amdgpu/cdna3/sgpr_layout.h"
#include "rocjitsu/isa/arch/amdgpu/shared/scalar_operand_selectors.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/target_registry.h"
#include "util/diagnostic.h"

#include <algorithm>
#include <array>
#include <bitset>
#include <cassert>
#include <cstring>
#include <functional>
#include <limits>
#include <optional>
#include <set>

namespace rocjitsu {

BlockOffsetIndex build_block_offset_index(const std::vector<std::unique_ptr<BasicBlock>> &blocks) {
  BlockOffsetIndex index;
  index.reserve(blocks.size());
  for (const auto &block : blocks) {
    if (block != nullptr) {
      index.emplace_back(block->start_offset(), block.get());
    }
  }
  std::ranges::sort(index, {}, &std::pair<uint64_t, BasicBlock *>::first);
  return index;
}

BlockPositionIndex
build_block_position_index(const std::vector<std::unique_ptr<BasicBlock>> &blocks) {
  BlockPositionIndex index;
  index.reserve(blocks.size());
  for (size_t i = 0; i < blocks.size(); ++i) {
    if (blocks[i] != nullptr) {
      index.emplace(blocks[i].get(), i);
    }
  }
  return index;
}

BasicBlock *block_for_offset(const BlockOffsetIndex &index, uint64_t offset) {
  auto it = std::ranges::upper_bound(index, offset, std::less<>{},
                                     &std::pair<uint64_t, BasicBlock *>::first);
  if (it == index.begin()) {
    return nullptr;
  }
  --it;

  BasicBlock *block = it->second;
  if (block == nullptr || offset >= block->end_offset()) {
    return nullptr;
  }
  return block;
}

std::vector<BasicBlock *> reachable_kernel_blocks(
    const std::vector<std::unique_ptr<BasicBlock>> &blocks, const BlockOffsetIndex &block_index,
    const BlockPositionIndex &block_positions, BasicBlock &entry, const KernelScopeSpec &spec) {
  std::vector<uint8_t> reachable(blocks.size(), 0);
  std::vector<size_t> reached_indices;
  std::vector<size_t> stack;
  auto push_block = [&](BasicBlock *block) {
    auto it = block_positions.find(block);
    if (it != block_positions.end()) {
      stack.push_back(it->second);
    }
  };
  push_block(&entry);
  for (const uint64_t own_entry : spec.own_entries) {
    if (own_entry == entry.start_offset()) {
      continue;
    }
    if (BasicBlock *extra_entry = block_for_offset(block_index, own_entry);
        extra_entry != nullptr && extra_entry != &entry) {
      push_block(extra_entry);
    }
  }

  while (!stack.empty()) {
    const size_t block_idx = stack.back();
    stack.pop_back();
    if (block_idx >= blocks.size() || reachable[block_idx]) {
      continue;
    }
    reachable[block_idx] = 1;
    reached_indices.push_back(block_idx);
    BasicBlock *block = blocks[block_idx].get();
    assert(block != nullptr && "reachable walk stack should contain only decoded blocks");

    for (BasicBlock *succ : block->successors()) {
      assert(succ != nullptr && "BasicBlock successors should never be null");
      if (!spec.own_entries.contains(succ->start_offset()) &&
          spec.kernel_entries.contains(succ->start_offset())) {
        continue;
      }
      push_block(succ);
    }
    // Ordinary CFG successors describe control that always follows from the
    // current program counter: fallthroughs, conditional targets, direct branch
    // targets, and recovered non-returning setpc targets. Call edges are tracked
    // separately because a shared callee block can return to different
    // continuations depending on which call site entered it. Reachability for
    // translation still has to include the callee body, but later liveness gets
    // explicit call/return edges rather than treating every possible return as a
    // global CFG successor.
    for (const BasicBlock::CallEdge &call : block->call_edges()) {
      BasicBlock *callee = call.callee;
      assert(callee != nullptr && "BasicBlock call edges should always have a callee");
      // A body whose address is taken is emitted exactly once, as an adopted root, and every
      // pointer to it names that one copy. Cloning it into each caller's scope would put the same
      // source offset at several placements and leave relocate_relative_text_addends() choosing
      // between them -- which is the reasoning kernel_translation_scopes() already documents for
      // adopted roots, applied here to the callees a relocation-table dispatch reaches.
      //
      // Not doing so is a divergence, not merely waste: the addend is rewritten to the canonical
      // clone, so a later translation gives every dispatch site a call edge to a clone owned by
      // some other scope, which that scope then clones again. Each pass adds another copy.
      //
      // Only the indirect edges may be dropped. A direct s_call_b64 to the same body leaves a
      // BranchFixup naming that source offset in this scope, and patch_direct_branch_fixups()
      // resolves it against the kernel-local layout, so dropping the body makes that call
      // unresolvable. Nothing rewrites a direct call through the addend, so a local clone cannot
      // create the placement ambiguity the indirect case has.
      const bool address_taken_indirect_callee =
          call.kind == BasicBlock::CallEdgeKind::IndirectSwapPc &&
          spec.address_taken_entries.contains(callee->start_offset());
      if (!spec.own_entries.contains(callee->start_offset()) &&
          (spec.kernel_entries.contains(callee->start_offset()) || address_taken_indirect_callee)) {
        continue;
      }
      push_block(callee);
    }
  }

  std::ranges::sort(reached_indices);
  std::vector<BasicBlock *> ordered;
  ordered.reserve(reached_indices.size());
  for (size_t block_idx : reached_indices) {
    if (blocks[block_idx]) {
      ordered.push_back(blocks[block_idx].get());
    }
  }
  return ordered;
}

KernelOwnership::KernelOwnership() = default;
KernelOwnership::~KernelOwnership() = default;
KernelOwnership::KernelOwnership(KernelOwnership &&) noexcept = default;
KernelOwnership &KernelOwnership::operator=(KernelOwnership &&) noexcept = default;

bool KernelOwnership::belongs_to(const AmdGpuCodeObject &source) const {
  const std::shared_ptr<const AmdGpuCodeObject> locked = source_.lock();
  return locked && locked.get() == &source;
}

std::optional<std::span<const KernelExecutionOwner>>
KernelOwnership::owners_at(uint64_t offset) const {
  const std::shared_ptr<const AmdGpuCodeObject> locked = source_.lock();
  if (!locked) {
    return std::nullopt;
  }
  const BasicBlock *block = block_for_offset(offsets_, offset);
  if (!block || std::ranges::none_of(block->instructions(), [&](const Instruction &inst) {
        return inst.src_loc() == offset;
      })) {
    return std::nullopt;
  }
  const std::unordered_map<const BasicBlock *, std::vector<KernelExecutionOwner>>::const_iterator
      found = owners_.find(block);
  if (found == owners_.end()) {
    return std::nullopt;
  }
  return std::span<const KernelExecutionOwner>(found->second);
}

std::optional<std::span<const KernelDescriptorInfo>> KernelOwnership::kernels() const {
  const std::shared_ptr<const AmdGpuCodeObject> locked = source_.lock();
  if (!locked) {
    return std::nullopt;
  }
  return std::span<const KernelDescriptorInfo>(kernels_);
}

std::optional<KernelOwnerFlow> KernelOwnership::owner_flow(size_t kernel_index) const {
  const std::shared_ptr<const AmdGpuCodeObject> locked = source_.lock();
  if (!locked || kernel_index >= flows_.size() || locked->text_sections().size() != 1) {
    return std::nullopt;
  }
  const FlowStorage &flow = flows_[kernel_index];
  const Section &text = *locked->text_sections().front();
  return KernelOwnerFlow{.blocks = flow.blocks,
                         .roots = flow.roots,
                         .call_return_edges = flow.edges,
                         .text = {reinterpret_cast<const uint8_t *>(text.data()), text.size()},
                         .target = locked->target_id(),
                         .wave_size = flow.wave_size};
}

const Instruction *KernelOwnership::instruction_at(size_t kernel_index, uint64_t offset) const {
  const std::shared_ptr<const AmdGpuCodeObject> locked = source_.lock();
  if (!locked || kernel_index >= flows_.size()) {
    return nullptr;
  }
  const std::optional<std::span<const KernelExecutionOwner>> owners = owners_at(offset);
  if (!owners || std::ranges::none_of(*owners, [&](const KernelExecutionOwner &owner) {
        return owner.kernel_index == kernel_index;
      })) {
    return nullptr;
  }
  const BasicBlock *block = block_for_offset(offsets_, offset);
  for (const Instruction &instruction : block->instructions()) {
    if (instruction.src_loc() == offset) {
      return &instruction;
    }
  }
  return nullptr;
}

static bool has_descriptor_sgpr_aliases(IsaFamily arch) {
  return arch == IsaFamily::Cdna1 || arch == IsaFamily::Cdna2 || arch == IsaFamily::Cdna3 ||
         arch == IsaFamily::Cdna4;
}

static std::optional<uint16_t> descriptor_sgpr_alias_base(IsaFamily arch, uint32_t alias_end,
                                                          RegClass cls) {
  if (cls == RegClass::VCC) {
    return cdna3::vcc_base(alias_end);
  }
  // The VCC allocation rule is shared by CDNA1-4. Only the CDNA3/4
  // FLAT_SCRATCH backing has been qualified for this ownership proof.
  if (cls == RegClass::FLAT_SCRATCH && (arch == IsaFamily::Cdna3 || arch == IsaFamily::Cdna4)) {
    return cdna3::flat_scratch_base(alias_end);
  }
  return std::nullopt;
}

static bool overlaps_descriptor_sgpr_alias(IsaFamily arch, uint32_t sgpr_count, uint16_t pair) {
  const uint32_t alias_end = cdna3::sgpr_alias_end(sgpr_count);
  for (RegClass cls : {RegClass::VCC, RegClass::FLAT_SCRATCH}) {
    const std::optional<uint16_t> base = descriptor_sgpr_alias_base(arch, alias_end, cls);
    if (base && pair < *base + 2u && pair + 2u > *base) {
      return true;
    }
  }
  return false;
}

struct DescriptorAliasWrites {
  bool vcc = false;
  bool flat_scratch = false;
};

static DescriptorAliasWrites descriptor_alias_writes(const Instruction &instruction) {
  DescriptorAliasWrites writes = {};
  const auto record = [&](RegClass cls) {
    writes.vcc = writes.vcc || cls == RegClass::VCC;
    writes.flat_scratch = writes.flat_scratch || cls == RegClass::FLAT_SCRATCH;
  };
  // Ordinary def/use projects selector-encoded specials out. Ownership also
  // needs named, fieldless and implicit writes to their physical SGPR backing.
  for (int destination = 0; destination < instruction.num_dst_operands(); ++destination) {
    const Operand *operand = instruction.dst_operand(destination);
    if (!operand) {
      continue;
    }
    const std::optional<RegisterRef> ref = operand->to_register_ref();
    if (ref) {
      record(ref->cls);
    }
    const std::optional<RegClass> special = operand->to_special_reg_class();
    if (special) {
      record(*special);
    }
    if (operand->has_register_selector()) {
      const int selector = operand->encoding_value();
      if (selector >= static_cast<int>(amdgpu::kVccSelectorFirst) &&
          selector <= static_cast<int>(amdgpu::kVccSelectorLast)) {
        record(RegClass::VCC);
      } else if (selector >= static_cast<int>(amdgpu::kFlatScratchSelectorFirst) &&
                 selector <= static_cast<int>(amdgpu::kFlatScratchSelectorLast)) {
        record(RegClass::FLAT_SCRATCH);
      }
    }
  }
  RegisterSet implicit_defs;
  instruction.implicit_defs(implicit_defs);
  implicit_defs.for_each([&](RegisterRef ref) { record(ref.cls); });
  return writes;
}

/// Prove that a setpc pair contains a scoped call continuation on every incoming
/// path. Syntactic return matching is sufficient for liveness edges, but cannot
/// establish ownership: hardware entries start with unknown registers, and any
/// write to either half invalidates the pair, including conditional writes.
///
/// Keep the exact continuation stored by each call, separating states with
/// different link values. A shared leaf can have an enclosing saved link in one
/// caller and no such link in another: merging them would invent return paths
/// that lose the enclosing link. The ordinary caller-to-continuation edge skips
/// callee effects and is excluded unless it is also an actual call target.
/// Repeated states terminate loops; a bounded number of extra contexts prevents
/// pathological state growth. Register save/restore idioms are not value-tracked.
struct ProvedReturns {
  std::unordered_set<uint64_t> offsets;
  std::vector<ScopedCfgEdge> edges;
};

static FailureOr<ProvedReturns>
proved_call_return_offsets(std::span<BasicBlock *const> scope, std::span<const uint64_t> roots,
                           const BlockOffsetIndex &index, IsaFamily arch,
                           uint32_t kernel_sgpr_count, uint8_t wave_size) {
  using LinkPairs = std::bitset<REGISTER_SET_MAX_SGPRS>;
  struct FlowState {
    LinkPairs killed;
    std::optional<std::pair<size_t, size_t>> saved_link;
    std::optional<size_t> return_pair;
    std::vector<size_t> successors;
  };
  std::unordered_map<const BasicBlock *, size_t> positions;
  std::array<size_t, REGISTER_SET_MAX_SGPRS> pair_positions;
  pair_positions.fill(REGISTER_SET_MAX_SGPRS);
  std::vector<uint16_t> return_registers;
  for (size_t position = 0; position < scope.size(); ++position) {
    positions.emplace(scope[position], position);
    for (const BasicBlock::CallEdge &call : scope[position]->call_edges()) {
      if (call.return_sreg + 1u < pair_positions.size() &&
          pair_positions[call.return_sreg] == REGISTER_SET_MAX_SGPRS) {
        pair_positions[call.return_sreg] = return_registers.size();
        return_registers.push_back(call.return_sreg);
      }
    }
  }
  if (return_registers.empty()) {
    return ProvedReturns{};
  }
  const bool descriptor_aliases = has_descriptor_sgpr_aliases(arch);
  const uint32_t alias_end = descriptor_aliases ? cdna3::sgpr_alias_end(kernel_sgpr_count) : 0;
  std::vector<FlowState> states(scope.size());
  for (size_t position = 0; position < scope.size(); ++position) {
    BasicBlock &block = *scope[position];
    FlowState &state = states[position];
    const auto invalidate_special_alias = [&](RegClass cls) {
      if (!descriptor_aliases) {
        return;
      }
      const std::optional<uint16_t> base = descriptor_sgpr_alias_base(arch, alias_end, cls);
      if (!base) {
        // An unqualified backing cannot establish preservation of any link.
        state.killed.set();
        return;
      }
      // A selector may name only one half. Killing the whole alias is safe for
      // this proof, including conditional writes and implicit definitions.
      for (size_t sgpr = *base; sgpr < static_cast<size_t>(*base) + 2; ++sgpr) {
        if (sgpr < state.killed.size()) {
          state.killed.set(sgpr);
        }
        if (sgpr != 0 && sgpr - 1 < state.killed.size()) {
          state.killed.set(sgpr - 1);
        }
      }
    };
    auto add_successor = [&](const BasicBlock *successor) {
      const std::unordered_map<const BasicBlock *, size_t>::const_iterator found =
          positions.find(successor);
      if (found != positions.end()) {
        state.successors.push_back(found->second);
      }
    };
    for (const Instruction &instruction : block.instructions()) {
      const std::string_view mnemonic = instruction.mnemonic();
      // Relative scalar destinations depend on M0, not just the encoded SDST.
      if (mnemonic.starts_with("s_movreld") || mnemonic.starts_with("s_movrelsd")) {
        state.killed.set();
      }
      InstDefUse(instruction, wave_size).defs.for_each([&](RegisterRef ref) {
        if (ref.cls != RegClass::SGPR) {
          return;
        }
        state.killed.set(ref.index);
        if (ref.index != 0) {
          state.killed.set(ref.index - 1);
        }
      });
      if (descriptor_aliases) {
        const DescriptorAliasWrites writes = descriptor_alias_writes(instruction);
        if (writes.vcc) {
          invalidate_special_alias(RegClass::VCC);
        }
        if (writes.flat_scratch) {
          invalidate_special_alias(RegClass::FLAT_SCRATCH);
        }
      }
    }
    const Instruction *term = block.terminator();
    if (!term) {
      continue;
    }
    if ((term->mnemonic() == "s_setpc_b64" || term->mnemonic() == "s_set_pc_i64") &&
        term->num_src_operands() != 0) {
      const std::optional<RegisterRef> source = term->src_operand(0)->to_register_ref();
      if (source && source->cls == RegClass::SGPR && source->width == 2 &&
          source->index + 1u < pair_positions.size() &&
          pair_positions[source->index] != REGISTER_SET_MAX_SGPRS) {
        state.return_pair = pair_positions[source->index];
      }
    }
    const BasicBlock *call_continuation = nullptr;
    for (const BasicBlock::CallEdge &call : block.call_edges()) {
      call_continuation = call.continuation;
      // Only ordinary SGPR destinations are tracked. Special scalar selectors
      // do not have the lane-wise def semantics needed for this proof.
      const std::unordered_map<const BasicBlock *, size_t>::const_iterator continuation =
          positions.find(call.continuation);
      if (call.return_sreg + 1u < pair_positions.size() && continuation != positions.end() &&
          InstDefUse(*term, wave_size).defs.contains({RegClass::SGPR, call.return_sreg, 2})) {
        state.saved_link = {pair_positions[call.return_sreg], continuation->second};
      }
      add_successor(call.callee);
    }
    for (const BasicBlock *successor : block.successors()) {
      if (successor == call_continuation) {
        const std::optional<int64_t> delta = term->branch_offset_bytes();
        const bool direct_target = delta && static_cast<int64_t>(block.end_offset()) + *delta ==
                                                static_cast<int64_t>(successor->start_offset());
        const bool indirect_target = std::ranges::any_of(
            block.static_indirect_call_fixups(), [&](const IndirectCallFixup &fixup) {
              return fixup.source_target_offset == successor->start_offset();
            });
        if (!direct_target && !indirect_target) {
          continue;
        }
      }
      add_successor(successor);
    }
  }
  // Store only registers used as call links. Unknown is outside the block index.
  const size_t unknown = scope.size();
  using ReturnContext = std::pair<size_t, std::vector<size_t>>;
  std::set<ReturnContext> visited;
  std::vector<std::set<ReturnContext>::const_iterator> pending;
  auto enqueue = [&](size_t position, const std::vector<size_t> &links) {
    const std::pair<std::set<ReturnContext>::iterator, bool> inserted =
        visited.emplace(position, links);
    if (inserted.second) {
      pending.push_back(inserted.first);
    }
  };
  for (uint64_t root : roots) {
    const BasicBlock *root_block = block_for_offset(index, root);
    const std::unordered_map<const BasicBlock *, size_t>::const_iterator position =
        positions.find(root_block);
    if (position == positions.end()) {
      return Result::failure();
    }
    enqueue(position->second, std::vector<size_t>(return_registers.size(), unknown));
  }
  ProvedReturns returns;
  std::unordered_set<uint64_t> unproved_returns;
  // Allow one state per block plus 64K additional calling contexts per kernel.
  // Exhaustion rejects ownership; it must never publish a partial proof.
  constexpr size_t kMaxExtraReturnContexts = 65536;
  while (!pending.empty()) {
    if (visited.size() > scope.size() && visited.size() - scope.size() > kMaxExtraReturnContexts) {
      return Result::failure();
    }
    const ReturnContext &context = *pending.back();
    pending.pop_back();
    const size_t position = context.first;
    const FlowState &state = states[position];
    std::vector<size_t> outgoing = context.second;
    for (size_t pair = 0; pair < return_registers.size(); ++pair) {
      if (state.killed.test(return_registers[pair])) {
        outgoing[pair] = unknown;
      }
    }
    if (state.saved_link) {
      outgoing[state.saved_link->first] = state.saved_link->second;
    }
    if (state.return_pair) {
      const uint64_t offset = scope[position]->terminator()->src_loc();
      const size_t continuation = outgoing[*state.return_pair];
      if (continuation == unknown) {
        unproved_returns.insert(offset);
      } else {
        returns.offsets.insert(offset);
        const ScopedCfgEdge edge = {.from = scope[position], .to = scope[continuation]};
        if (std::ranges::none_of(returns.edges, [&](const ScopedCfgEdge &existing) {
              return existing.from == edge.from && existing.to == edge.to;
            })) {
          returns.edges.push_back(edge);
        }
        enqueue(continuation, outgoing);
      }
    }
    for (size_t successor : state.successors) {
      enqueue(successor, outgoing);
    }
  }
  for (uint64_t offset : unproved_returns) {
    returns.offsets.erase(offset);
  }
  std::erase_if(returns.edges, [&](const ScopedCfgEdge &edge) {
    return !edge.from->terminator() ||
           !returns.offsets.contains(edge.from->terminator()->src_loc());
  });
  return returns;
}

namespace {

enum class PcBuilderInputProof { Proved, MissingPath, CrossesCall };

PcBuilderInputProof
pc_builder_input_proof(const std::unordered_set<const BasicBlock *> &producers,
                       const BasicBlock &consumer,
                       const std::unordered_set<uint64_t> &hardware_entries,
                       const std::unordered_set<const BasicBlock *> &call_entries) {
  // Recovery keeps useful builder edges across syntactic call continuations,
  // but ownership additionally needs proof that the callee preserved the pair.
  // Stop at any complete producer: a shared target may have one per kernel.
  // Executing getpc again refreshes the value, including
  // on loop backedges. Inspect predecessor edges so the consumer's own call
  // does not invalidate the target it reads before entering its callee.
  std::vector<const BasicBlock *> pending = {&consumer};
  std::unordered_set<const BasicBlock *> visited;
  bool reached_producer = false;
  while (!pending.empty()) {
    const BasicBlock *block = pending.back();
    pending.pop_back();
    if (producers.contains(block)) {
      reached_producer = true;
      continue;
    }
    if (!visited.insert(block).second) {
      continue;
    }
    // Call edges are separate from ordinary predecessors, even when a callee
    // also has a loop backedge. Losing that incoming path is not a proof.
    if (call_entries.contains(block)) {
      return PcBuilderInputProof::CrossesCall;
    }
    if (hardware_entries.contains(block->start_offset()) || block->predecessors().empty()) {
      return PcBuilderInputProof::MissingPath;
    }
    for (const BasicBlock *predecessor : block->predecessors()) {
      if (!predecessor->call_edges().empty()) {
        return PcBuilderInputProof::CrossesCall;
      }
      pending.push_back(predecessor);
    }
  }
  return reached_producer ? PcBuilderInputProof::Proved : PcBuilderInputProof::MissingPath;
}

std::optional<uint64_t> file_backed_address(std::span<const Elf64_Phdr> segments, uint64_t offset,
                                            uint64_t size, uint32_t required_flags) {
  if (size == 0) {
    return std::nullopt;
  }
  const Elf64_Phdr *mapping = nullptr;
  uint64_t address = 0;
  for (const Elf64_Phdr &segment : segments) {
    if (segment.p_type != PT_LOAD || offset < segment.p_offset) {
      continue;
    }
    const uint64_t delta = offset - segment.p_offset;
    if (delta > segment.p_filesz || size > segment.p_filesz - delta) {
      continue;
    }
    if (mapping || (segment.p_flags & required_flags) != required_flags ||
        segment.p_vaddr > std::numeric_limits<uint64_t>::max() - delta) {
      return std::nullopt;
    }
    mapping = &segment;
    address = segment.p_vaddr + delta;
  }
  if (!mapping || address > std::numeric_limits<uint64_t>::max() - size ||
      offset > std::numeric_limits<uint64_t>::max() - size) {
    return std::nullopt;
  }
  for (const Elf64_Phdr &segment : segments) {
    if (&segment == mapping || segment.p_type != PT_LOAD) {
      continue;
    }
    if (segment.p_filesz != 0 && segment.p_offset < offset + size &&
        (segment.p_offset >= offset || segment.p_filesz > offset - segment.p_offset)) {
      return std::nullopt;
    }
    if (segment.p_memsz != 0 && segment.p_vaddr < address + size &&
        (segment.p_vaddr >= address || segment.p_memsz > address - segment.p_vaddr)) {
      return std::nullopt;
    }
  }
  return address;
}

bool validate_owner_descriptors(const AmdGpuCodeObject &source, const Section &text,
                                std::span<const KernelDescriptorInfo> kernels,
                                std::vector<uint64_t> &declared_functions, std::string &error) {
  const std::span<const uint8_t> image(reinterpret_cast<const uint8_t *>(source.image_data()),
                                       source.image_size());
  const auto fail = [&](const char *message) {
    error = message;
    return false;
  };
  if (image.size() < sizeof(Elf64_Ehdr) || kernels.empty() || !text.sectionHeaderIndex() ||
      (text.flags() & (SHF_ALLOC | SHF_EXECINSTR)) != (SHF_ALLOC | SHF_EXECINSTR)) {
    return fail("missing executable text or descriptor inventory");
  }
  Elf64_Ehdr header = {};
  std::memcpy(&header, image.data(), sizeof(header));
  const IsaTargetBinding *target = default_isa_target_registry().find_target(source.target_id());
  if (!target || std::memcmp(header.e_ident, EI_MAGIC, EI_MAGIC_SIZE) != 0 ||
      header.e_ident[EI_CLASS] != ELFCLASS64 || header.e_ident[EI_DATA] != ELFDATA2LSB ||
      header.e_ident[EI_OSABI] != ELFOSABI_AMDGPU_HSA ||
      header.e_ident[EI_ABIVERSION] < ELFABIVERSION_AMDGPU_HSA_V3 ||
      header.e_ident[EI_ABIVERSION] > ELFABIVERSION_AMDGPU_HSA_V6 || header.e_type != ET_DYN ||
      header.e_machine != EM_AMDGPU || (header.e_flags & EF_AMDGPU_MACH) != target->elf_machine ||
      (header.e_flags & ~uint32_t{0xfff}) != 0) {
    return fail("invalid source target or ELF identity");
  }
  const std::optional<std::vector<Elf64_Shdr>> sections =
      read_elf_table<Elf64_Shdr>(image, header.e_shoff, header.e_shnum, header.e_shentsize);
  const std::optional<std::vector<Elf64_Phdr>> segments =
      read_elf_table<Elf64_Phdr>(image, header.e_phoff, header.e_phnum, header.e_phentsize);
  if (!sections || !segments) {
    return fail("invalid ELF tables");
  }
  const std::optional<uint64_t> text_address =
      file_backed_address(*segments, text.sectionOffset(), text.size(), PF_X);
  if (!text_address || *text_address != text.vaddr()) {
    return fail("text mapping is not uniquely file-backed");
  }
  for (const KernelDescriptorInfo &kernel : kernels) {
    const std::optional<uint64_t> address =
        file_backed_address(*segments, kernel.descriptor_file_offset, sizeof(kernel.descriptor), 0);
    const std::optional<uint64_t> entry =
        address ? add_signed_offset(*address, kernel.descriptor.kernel_code_entry_byte_offset)
                : std::nullopt;
    if (!entry || kernel.entry_text_offset > std::numeric_limits<uint64_t>::max() - text.vaddr() ||
        *entry != text.vaddr() + kernel.entry_text_offset) {
      return fail("descriptor mapping does not resolve to its source entry");
    }
  }
  for (const Elf64_Shdr &section : *sections) {
    if (section.sh_type == SHT_REL ||
        classify_rocr_relocation_section(header, *sections, section) ==
            RocrRelocationSectionMode::Dynamic ||
        classify_rocr_relocation_section(header, *sections, section) ==
            RocrRelocationSectionMode::Malformed) {
      return fail("dynamic or malformed relocations require additional ownership proof");
    }
    if (section.sh_type != SHT_SYMTAB && section.sh_type != SHT_DYNSYM) {
      continue;
    }
    if (section.sh_entsize != sizeof(Elf64_Sym) || section.sh_size % sizeof(Elf64_Sym) != 0 ||
        section.sh_link >= sections->size()) {
      return fail("invalid owner symbol table");
    }
    const std::optional<std::vector<Elf64_Sym>> symbols = read_elf_table<Elf64_Sym>(
        image, section.sh_offset, section.sh_size / sizeof(Elf64_Sym), section.sh_entsize);
    const Elf64_Shdr &strings = (*sections)[section.sh_link];
    if (!symbols || strings.sh_type != SHT_STRTAB || strings.sh_offset > image.size() ||
        strings.sh_size > image.size() - strings.sh_offset) {
      return fail("invalid owner symbol strings");
    }
    for (const Elf64_Sym &symbol : *symbols) {
      if (symbol.st_name >= strings.sh_size) {
        return fail("invalid owner symbol name");
      }
      const char *name =
          reinterpret_cast<const char *>(image.data() + strings.sh_offset + symbol.st_name);
      const void *end = std::memchr(name, 0, strings.sh_size - symbol.st_name);
      if (!end) {
        return fail("unterminated owner symbol name");
      }
      const std::string_view name_view(name, static_cast<const char *>(end) - name);
      const uint8_t type = elf_symbol_type(symbol.st_info);
      if (!name_view.ends_with(".kd")) {
        if (symbol.st_shndx != *text.sectionHeaderIndex()) {
          if ((type == kElfSymbolTypeFunc || type == kElfSymbolTypeAmdGpuHsaKernel ||
               elf_symbol_bind(symbol.st_info) != kElfSymbolBindLocal) &&
              symbol.st_shndx < sections->size() &&
              ((*sections)[symbol.st_shndx].sh_flags & (SHF_ALLOC | SHF_EXECINSTR)) ==
                  (SHF_ALLOC | SHF_EXECINSTR)) {
            return fail("executable symbol is outside analyzed text");
          }
          continue;
        }
        if (type == kElfSymbolTypeObject || type == kElfSymbolTypeAmdGpuHsaKernel) {
          return fail("unsupported executable object or kernel symbol");
        }
        if (type != kElfSymbolTypeFunc && elf_symbol_bind(symbol.st_info) != kElfSymbolBindLocal &&
            (symbol.st_value < text.vaddr() ||
             !std::ranges::any_of(kernels, [&](const KernelDescriptorInfo &kernel) {
               return kernel.entry_text_offset == symbol.st_value - text.vaddr();
             }))) {
          return fail("external executable symbol needs additional owner proof");
        }
        if (type == kElfSymbolTypeFunc) {
          if (symbol.st_value < text.vaddr() || symbol.st_value - text.vaddr() >= text.size()) {
            return fail("function entry is outside analyzed text");
          }
          const uint64_t offset = symbol.st_value - text.vaddr();
          if (section.sh_type == SHT_DYNSYM &&
              elf_symbol_bind(symbol.st_info) != kElfSymbolBindLocal &&
              !std::ranges::any_of(kernels, [&](const KernelDescriptorInfo &kernel) {
                return kernel.entry_text_offset == offset;
              })) {
            return fail("external dynamic function needs additional owner proof");
          }
          declared_functions.push_back(offset);
        }
        continue;
      }
      if (symbol.st_shndx >= sections->size() ||
          symbol.st_size != sizeof(amdhsa::kernel_descriptor_t) ||
          elf_symbol_type(symbol.st_info) != kElfSymbolTypeObject ||
          elf_symbol_bind(symbol.st_info) != kElfSymbolBindGlobal) {
        return fail("malformed kernel descriptor symbol");
      }
      const Elf64_Shdr &owner = (*sections)[symbol.st_shndx];
      if (owner.sh_type == SHT_NOBITS || symbol.st_value < owner.sh_addr ||
          symbol.st_value - owner.sh_addr > owner.sh_size ||
          symbol.st_size > owner.sh_size - (symbol.st_value - owner.sh_addr) ||
          owner.sh_offset > image.size() || owner.sh_size > image.size() - owner.sh_offset) {
        return fail("invalid kernel descriptor symbol range");
      }
      const uint64_t file_offset = owner.sh_offset + (symbol.st_value - owner.sh_addr);
      if (!std::ranges::any_of(kernels, [&](const KernelDescriptorInfo &kernel) {
            return kernel.kernel_name == name_view.substr(0, name_view.size() - 3) &&
                   kernel.descriptor_file_offset == file_offset;
          })) {
        return fail("descriptor was omitted from owner inventory");
      }
    }
  }
  return true;
}
} // namespace

FailureOr<KernelOwnership>
KernelOwnership::build(std::shared_ptr<const AmdGpuCodeObject> source_handle,
                       rj_code_target_id_t expected_target, std::string &error) {
  auto reject = [&](const char *message) {
    error = message;
    return Result::failure();
  };
  error.clear();
  if (!source_handle) {
    return reject("missing immutable source");
  }
  const AmdGpuCodeObject &source = *source_handle;
  const IsaTargetRegistry &registry = default_isa_target_registry();
  const IsaTargetDescriptor *architecture = registry.find(expected_target);
  if (!source.is_valid() || expected_target != source.target_id() || !architecture ||
      !registry.find_target(expected_target) || source.text_sections().size() != 1) {
    return reject("invalid source or mismatched concrete target");
  }
  const Section &section = *source.text_sections().front();
  const std::span<const uint8_t> image(reinterpret_cast<const uint8_t *>(source.image_data()),
                                       source.image_size());
  const std::span<const uint8_t> text(reinterpret_cast<const uint8_t *>(section.data()),
                                      section.size());
  const std::vector<KernelDescriptorInfo> kernels = scan_kernel_descriptors(
      image, section.sectionOffset(), section.size(), section.sectionHeaderIndex());
  std::vector<uint64_t> declared_functions;
  if (!validate_owner_descriptors(source, section, kernels, declared_functions, error)) {
    return Result::failure();
  }
  std::vector<std::vector<uint64_t>> kernel_roots;
  std::vector<uint8_t> kernel_wave_sizes;
  std::vector<uint64_t> entries;
  for (const KernelDescriptorInfo &kernel : kernels) {
    const std::optional<std::vector<uint64_t>> roots =
        kernel_hardware_entry_candidates(kernel, architecture->family, section.size());
    if (!roots) {
      return reject("descriptor hardware root is out of range");
    }
    entries.insert(entries.end(), roots->begin(), roots->end());
    kernel_roots.push_back(*roots);
    kernel_wave_sizes.push_back(kernel_wavefront_size(architecture->family, kernel.descriptor));
  }
  std::unique_ptr<Decoder> decoder = Decoder::create(registry, expected_target);
  if (!decoder) {
    return reject("concrete target decoder unavailable");
  }
  util::StringDiagnostic diagnostic;
  FailureOr<std::vector<std::unique_ptr<BasicBlock>>> decoded =
      BasicBlock::build_reachable(source, *decoder, architecture->family, entries,
                                  diagnostic.emitter(), {}, {}, {}, expected_target);
  if (decoded.failed()) {
    error = diagnostic.message();
    return Result::failure();
  }
  KernelOwnership result;
  result.source_ = source_handle;
  result.kernels_ = kernels;
  result.blocks_ = std::move(decoded).value();
  const std::vector<std::unique_ptr<BasicBlock>> &blocks = result.blocks_;
  const BlockOffsetIndex index = build_block_offset_index(blocks);
  const BlockPositionIndex positions = build_block_position_index(blocks);
  result.offsets_ = index;
  std::unordered_set<uint64_t> all_entries;
  std::unordered_set<const BasicBlock *> call_entries;
  for (const std::unique_ptr<BasicBlock> &block : blocks) {
    for (const BasicBlock::CallEdge &call : block->call_edges()) {
      call_entries.insert(call.callee);
    }
  }
  for (const std::vector<uint64_t> &roots : kernel_roots) {
    if (roots.empty()) {
      return reject("kernel has no hardware roots");
    }
    for (uint64_t entry : roots) {
      const BasicBlock *root = block_for_offset(index, entry);
      if (!all_entries.insert(entry).second) {
        return reject("kernel descriptors alias one hardware entry");
      }
      if (entry >= text.size() || !root || root->start_offset() != entry) {
        return reject("hardware entry is not a decoded block start");
      }
    }
  }
  for (size_t kernel_index = 0; kernel_index < kernel_roots.size(); ++kernel_index) {
    const std::vector<uint64_t> &roots = kernel_roots[kernel_index];
    const KernelScopeSpec spec = {.kernel_entries = all_entries,
                                  .own_entries = {roots.begin(), roots.end()},
                                  .address_taken_entries = {}};
    std::vector<BasicBlock *> scope = reachable_kernel_blocks(
        blocks, index, positions, *block_for_offset(index, roots.front()), spec);
    const std::optional<uint32_t> kernel_sgpr_count = AmdGpuCodeObject::min_kernel_sgpr_count(
        architecture->family,
        std::span<const KernelDescriptorInfo>(kernels).subspan(kernel_index, 1));
    if (!kernel_sgpr_count) {
      return reject("kernel SGPR allocation is unavailable");
    }
    const FailureOr<ProvedReturns> returns =
        proved_call_return_offsets(scope, roots, index, architecture->family, *kernel_sgpr_count,
                                   kernel_wave_sizes[kernel_index]);
    if (returns.failed()) {
      return reject("return ownership exceeds context budget");
    }
    const bool unproved_flat_scratch =
        (architecture->family == IsaFamily::Cdna1 || architecture->family == IsaFamily::Cdna2) &&
        std::ranges::any_of(scope, [](const BasicBlock *block) {
          return std::ranges::any_of(block->instructions(), [](const Instruction &instruction) {
            return descriptor_alias_writes(instruction).flat_scratch;
          });
        });
    for (BasicBlock *block : scope) {
      if (block->has_implicit_terminator() ||
          (block->successor_issue() != BasicBlock::SuccessorIssue::None &&
           block->successor_issue() != BasicBlock::SuccessorIssue::IndirectControlFlow)) {
        return reject("kernel ownership requires a closed CFG");
      }
      auto crosses_entry = [&](const BasicBlock *destination) {
        return destination && all_entries.contains(destination->start_offset()) &&
               !spec.own_entries.contains(destination->start_offset());
      };
      for (const BasicBlock *successor : block->successors()) {
        if (crosses_entry(successor)) {
          return reject("control flow crosses another kernel entry");
        }
      }
      for (const BasicBlock::CallEdge &call : block->call_edges()) {
        if (crosses_entry(call.callee) || crosses_entry(call.continuation)) {
          return reject("call crosses another kernel entry");
        }
      }
      for (const Instruction &instruction : block->instructions()) {
        // The CFG closes abort traps and retains other traps' fallthrough.
        // Their register definitions still participate in the return proof.
        if (!is_control_flow_transfer(instruction) || is_program_path_terminator(instruction) ||
            (instruction.flags() & SOFTWARE_TRAP) != 0 || instruction.branch_offset_bytes()) {
          continue;
        }
        if (returns.value().offsets.contains(instruction.src_loc())) {
          continue;
        }
        bool recovered = false;
        std::unordered_set<const BasicBlock *> producers;
        for (const IndirectCallFixup &fixup : block->static_indirect_call_fixups()) {
          if (fixup.source_call_offset != instruction.src_loc()) {
            continue;
          }
          const BasicBlock *target = block_for_offset(index, fixup.source_target_offset);
          if (fixup.source_incomplete || !target ||
              target->start_offset() != fixup.source_target_offset) {
            return reject("incomplete indirect ownership");
          }
          if (unproved_flat_scratch) {
            return reject("recovered PC ownership requires proved FLAT_SCRATCH backing");
          }
          // Recovery has no descriptor allocation with which to model the
          // special registers' physical backing. Its producer and consumer
          // must both name ordinary, unaliased pairs in this owner's layout.
          if (has_descriptor_sgpr_aliases(architecture->family) &&
              (fixup.source_builder_sreg + 1u >= REGISTER_SET_ALLOCATABLE_SGPRS ||
               fixup.source_call_sreg + 1u >= REGISTER_SET_ALLOCATABLE_SGPRS ||
               overlaps_descriptor_sgpr_alias(architecture->family, *kernel_sgpr_count,
                                              fixup.source_builder_sreg) ||
               overlaps_descriptor_sgpr_alias(architecture->family, *kernel_sgpr_count,
                                              fixup.source_call_sreg))) {
            return reject("recovered PC ownership requires unaliased ordinary SGPR pairs");
          }
          const BasicBlock *producer = block_for_offset(index, fixup.source_getpc_offset);
          if (!producer) {
            return reject("recovered target lacks incoming-path proof");
          }
          producers.insert(producer);
          recovered = true;
        }
        if (!recovered) {
          return reject("unresolved indirect ownership");
        }
        const PcBuilderInputProof proof =
            pc_builder_input_proof(producers, *block, all_entries, call_entries);
        if (proof == PcBuilderInputProof::CrossesCall) {
          return reject("call-carried PC builder requires callee preservation proof");
        }
        if (proof != PcBuilderInputProof::Proved) {
          return reject("recovered target lacks incoming-path proof");
        }
      }
    }
    FlowStorage flow = {.blocks = scope,
                        .roots = {},
                        .edges = returns.value().edges,
                        .wave_size = kernel_wave_sizes[kernel_index]};
    for (uint64_t root : roots) {
      flow.roots.push_back(block_for_offset(index, root));
    }
    for (BasicBlock *block : scope) {
      for (const BasicBlock::CallEdge &call : block->call_edges()) {
        if (!call.callee || !call.continuation ||
            std::ranges::find(scope, call.callee) == scope.end() ||
            std::ranges::find(scope, call.continuation) == scope.end()) {
          return reject("incomplete scoped call edge");
        }
        flow.edges.push_back({.from = block, .to = call.callee});
      }
    }
    result.flows_.push_back(std::move(flow));
    // Propagate the strongest required call proof along each path. Ordinary
    // successors include call continuations, so helpers do not taint caller-local sites.
    std::unordered_map<const BasicBlock *, KernelOwnerProof> proofs;
    std::vector<std::pair<BasicBlock *, KernelOwnerProof>> pending;
    for (uint64_t root : roots) {
      pending.emplace_back(block_for_offset(index, root), KernelOwnerProof::KernelLocal);
    }
    while (!pending.empty()) {
      const std::pair<BasicBlock *, KernelOwnerProof> current = pending.back();
      pending.pop_back();
      BasicBlock *block = current.first;
      const KernelOwnerProof proof = current.second;
      const std::unordered_map<const BasicBlock *, KernelOwnerProof>::const_iterator old =
          proofs.find(block);
      if (old != proofs.end() && old->second >= proof) {
        continue;
      }
      proofs[block] = proof;
      for (BasicBlock *successor : block->successors()) {
        pending.emplace_back(successor, proof);
      }
      for (const BasicBlock::CallEdge &call : block->call_edges()) {
        const KernelOwnerProof edge = call.kind == BasicBlock::CallEdgeKind::DirectCall
                                          ? KernelOwnerProof::DirectCall
                                          : KernelOwnerProof::RecoveredIndirectCall;
        pending.emplace_back(call.callee, std::max(proof, edge));
      }
    }
    for (const BasicBlock *block : scope) {
      const std::unordered_map<const BasicBlock *, KernelOwnerProof>::const_iterator proof =
          proofs.find(block);
      if (proof == proofs.end()) {
        return reject("scope block has no owner proof");
      }
      result.owners_[block].push_back({.kernel_index = kernel_index, .proof = proof->second});
    }
  }
  if (result.owners_.size() != blocks.size()) {
    return reject("unresolved code ownership");
  }
  for (const std::unique_ptr<BasicBlock> &block : blocks) {
    if (result.owners_.at(block.get()).size() < 2) {
      continue;
    }
    // A shared dispatch's recovered edges contain the union of all incoming
    // builders. Without kernel-specific value provenance, following that union
    // for each owner would invent owners at its distinct destinations.
    std::unordered_map<uint64_t, uint64_t> targets;
    for (const IndirectCallFixup &fixup : block->static_indirect_call_fixups()) {
      const std::pair<std::unordered_map<uint64_t, uint64_t>::iterator, bool> inserted =
          targets.emplace(fixup.source_call_offset, fixup.source_target_offset);
      if (!inserted.second && inserted.first->second != fixup.source_target_offset) {
        return reject("shared indirect targets require kernel-specific proof");
      }
    }
  }
  for (uint64_t function : declared_functions) {
    const BasicBlock *block = block_for_offset(result.offsets_, function);
    if (!block || block->start_offset() != function || !result.owners_at(function)) {
      return reject("declared function has unresolved ownership");
    }
  }
  return result;
}

} // namespace rocjitsu
