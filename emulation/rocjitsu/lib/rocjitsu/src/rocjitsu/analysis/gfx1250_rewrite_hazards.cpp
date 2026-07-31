// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/analysis/gfx1250_rewrite_hazards.h"

#include "rocjitsu/analysis/gfx1250_vgpr_msb.h"
#include "rocjitsu/code/basic_block.h"
#include "rocjitsu/isa/arch/amdgpu/gfx1250/machine_insts.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/isa/operand.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <set>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace rocjitsu {

RegisterSet Gfx1250GeneratedValuHazard::avoid_destination_vgprs_in_bank(uint8_t bank) const {
  RegisterSet physical;
  for (uint8_t nops = 1; nops <= kMaxNops; ++nops)
    physical |= defs_requiring_nops[nops];

  RegisterSet result;
  if (bank >= 4)
    return result;
  constexpr uint16_t kVgprsPerBank = 256;
  physical.for_each([&](RegisterRef ref) {
    if (ref.cls == RegClass::VGPR && ref.index / kVgprsPerBank == bank) {
      result.expand({RegClass::VGPR, static_cast<uint16_t>(ref.index % kVgprsPerBank), ref.width});
    }
  });
  return result;
}

uint8_t Gfx1250GeneratedValuHazard::required_nops(const RegisterSet &defs,
                                                  const RegisterSet &uses) const {
  for (uint8_t nops = kMaxNops; nops != 0; --nops) {
    if (defs.intersects(defs_requiring_nops[nops]) || uses.intersects(uses_requiring_nops[nops])) {
      return nops;
    }
  }
  return 0;
}

namespace {

[[nodiscard]] bool is_valu(const Instruction &inst) { return inst.mnemonic().starts_with("v_"); }

[[nodiscard]] bool is_wmma(const Instruction &inst) {
  return inst.mnemonic().starts_with("v_wmma_") || inst.mnemonic().starts_with("v_swmmac_");
}

[[nodiscard]] bool is_swmmac(const Instruction &inst) {
  return inst.mnemonic().starts_with("v_swmmac_");
}

[[nodiscard]] bool is_trans(const Instruction &inst) {
  const std::string_view mnemonic = inst.mnemonic();
  constexpr std::array<std::string_view, 8> prefixes = {
      "v_exp_", "v_log_", "v_rcp_", "v_rsq_", "v_sqrt_", "v_sin_", "v_cos_", "v_tanh_",
  };
  return std::ranges::any_of(prefixes,
                             [&](std::string_view prefix) { return mnemonic.starts_with(prefix); });
}

[[nodiscard]] const uint32_t *wmma_matrix_words(const Instruction &inst) {
  if (inst.raw_encoding() == nullptr)
    return nullptr;
  const std::string_view mnemonic = inst.mnemonic();
  if ((mnemonic.starts_with("v_wmma_scale_") || mnemonic.starts_with("v_wmma_scale16_")) &&
      inst.size() >= 4 * static_cast<int>(sizeof(uint32_t))) {
    return inst.raw_encoding() + 2;
  }
  if (inst.size() >= 2 * static_cast<int>(sizeof(uint32_t)))
    return inst.raw_encoding();
  return nullptr;
}

/// @brief Decode the MATRIX_A_FMT field from a gfx1250 VOP3P WMMA body.
[[nodiscard]] uint8_t wmma_matrix_a_format(const gfx1250::Vop3pMachineInst &matrix) {
  return static_cast<uint8_t>(matrix.opsel);
}

/// @brief Decode the split MATRIX_B_FMT field from a gfx1250 VOP3P WMMA body.
///
/// @details The generated machine struct names reserved encoding bit 14
/// `pad_14`, but gfx1250 WMMA assigns that bit as MATRIX_B_FMT[2]. Keep the
/// encoding-specific assembly in one helper so latency classification cannot
/// silently omit the high format bit.
[[nodiscard]] uint8_t wmma_matrix_b_format(const gfx1250::Vop3pMachineInst &matrix) {
  return static_cast<uint8_t>((matrix.pad_14 << 2u) | matrix.opsel_hi);
}

/// @brief LLVM gfx1250 WMMA-to-ordinary-VALU distance for one producer.
[[nodiscard]] uint8_t wmma_to_valu_distance(const Instruction &inst) {
  const std::string_view mnemonic = inst.mnemonic();
  if (is_swmmac(inst))
    return mnemonic.find("_iu8") != std::string_view::npos ? 4 : 2;
  if (mnemonic.find("_iu8") != std::string_view::npos)
    return 8;

  // Mixed F8F6F4 is the only gfx1250 dense class whose latency depends on
  // encoded formats. Either F8 input selects the eight-slot category.
  if (mnemonic.find("f8f6f4") != std::string_view::npos) {
    const uint32_t *matrix_words = wmma_matrix_words(inst);
    if (matrix_words == nullptr)
      return 8;
    gfx1250::Vop3pMachineInst matrix{};
    std::memcpy(&matrix, matrix_words, sizeof(matrix));
    return wmma_matrix_a_format(matrix) <= 1 || wmma_matrix_b_format(matrix) <= 1 ? 8 : 4;
  }
  return 4;
}

void expand_operand(RegisterSet &set, const Instruction &inst, const Operand &operand,
                    const Gfx1250VgprMsbAnalysis &vgpr_msb) {
  auto ref = operand.to_register_ref();
  if (!ref || ref->cls != RegClass::VGPR)
    return;
  if (const auto bank = vgpr_msb.bank_before(inst, operand.vgpr_msb_role())) {
    ref->index = static_cast<uint16_t>(ref->index + static_cast<uint16_t>(*bank) * 256u);
    set.expand(*ref);
    return;
  }

  // A join with different MODE.VGPR_MSB values can select any one bank.
  // Timing exclusions are may-access facts, so retain all four possibilities.
  for (uint16_t bank = 0; bank < 4; ++bank) {
    RegisterRef possible = *ref;
    possible.index = static_cast<uint16_t>(possible.index + bank * 256u);
    set.expand(possible);
  }
}

struct PhysicalVgprAccesses {
  RegisterSet defs;
  RegisterSet uses;
};

[[nodiscard]] PhysicalVgprAccesses physical_vgpr_accesses(const Instruction &inst,
                                                          const Gfx1250VgprMsbAnalysis &vgpr_msb) {
  PhysicalVgprAccesses result;
  for (int index = 0; index < inst.num_dst_operands(); ++index) {
    if (const Operand *operand = inst.dst_operand(index))
      expand_operand(result.defs, inst, *operand, vgpr_msb);
  }
  for (int index = 0; index < inst.num_src_operands(); ++index) {
    if (const Operand *operand = inst.src_operand(index))
      expand_operand(result.uses, inst, *operand, vgpr_msb);
  }
  std::vector<const Operand *> implicit_uses;
  inst.implicit_use_operands(implicit_uses);
  for (const Operand *operand : implicit_uses) {
    if (operand != nullptr)
      expand_operand(result.uses, inst, *operand, vgpr_msb);
  }
  return result;
}

struct WmmaCoexecutionAccesses {
  RegisterSet definition;
  RegisterSet definition_conflicts;
  RegisterSet inputs;
};

[[nodiscard]] WmmaCoexecutionAccesses
wmma_coexecution_accesses(const Instruction &inst, const Gfx1250VgprMsbAnalysis &vgpr_msb) {
  WmmaCoexecutionAccesses result;
  if (const Operand *destination = inst.dst_operand(0)) {
    expand_operand(result.definition, inst, *destination, vgpr_msb);
    expand_operand(result.definition_conflicts, inst, *destination, vgpr_msb);
  }
  for (int index = 0; index < std::min(inst.num_src_operands(), 2); ++index) {
    if (const Operand *operand = inst.src_operand(index)) {
      expand_operand(result.definition_conflicts, inst, *operand, vgpr_msb);
      expand_operand(result.inputs, inst, *operand, vgpr_msb);
    }
  }
  if (is_swmmac(inst)) {
    if (const Operand *index = inst.src_operand(2)) {
      expand_operand(result.definition_conflicts, inst, *index, vgpr_msb);
      expand_operand(result.inputs, inst, *index, vgpr_msb);
    }
  }
  return result;
}

} // namespace

bool gfx1250_is_rewrite_hazard_producer(const Instruction &inst) {
  return is_trans(inst) || is_wmma(inst);
}

bool gfx1250_scope_has_rewrite_hazard_producer(KernelBlockScope blocks) {
  for (BasicBlock *block : blocks) {
    if (block == nullptr)
      continue;
    for (const Instruction &inst : block->instructions()) {
      if (gfx1250_is_rewrite_hazard_producer(inst))
        return true;
    }
  }
  return false;
}

class Gfx1250RewriteHazardAnalysis::Impl {
public:
  Impl(KernelBlockScope blocks, BasicBlock *entry, std::span<const ScopedCfgEdge> extra_edges,
       std::span<const uint8_t> text)
      : vgpr_msb_(blocks, entry, extra_edges, text) {
    std::unordered_map<const BasicBlock *, size_t> block_index;
    block_index.reserve(blocks.size());
    predecessors_.resize(blocks.size());
    successors_.resize(blocks.size());
    instructions_.resize(blocks.size());
    for (size_t index = 0; index < blocks.size(); ++index) {
      BasicBlock *block = blocks[index];
      if (block == nullptr)
        continue;
      block_index.emplace(block, index);
      auto &instructions = instructions_[index];
      instructions.reserve(block->num_instructions());
      for (const Instruction &inst : block->instructions()) {
        instruction_position_.emplace(&inst, std::pair{index, instructions.size()});
        instructions.push_back(&inst);
      }
    }

    const auto add_edge = [&](const BasicBlock *from, const BasicBlock *to) {
      const auto from_it = block_index.find(from);
      const auto to_it = block_index.find(to);
      if (from_it == block_index.end() || to_it == block_index.end())
        return;
      auto &successors = successors_[from_it->second];
      if (std::ranges::find(successors, to_it->second) == successors.end())
        successors.push_back(to_it->second);
      auto &predecessors = predecessors_[to_it->second];
      if (std::ranges::find(predecessors, from_it->second) == predecessors.end())
        predecessors.push_back(from_it->second);
    };
    for (const BasicBlock *block : blocks) {
      if (block == nullptr)
        continue;
      for (const BasicBlock *successor : block->successors())
        add_edge(block, successor);
    }
    for (const ScopedCfgEdge &edge : extra_edges)
      add_edge(edge.from, edge.to);
  }

  [[nodiscard]] Gfx1250GeneratedValuHazard before_generated_valu(const Instruction &inst) const {
    Gfx1250GeneratedValuHazard result;
    const auto initial = instruction_position_.find(&inst);
    if (initial == instruction_position_.end())
      return result;

    struct Cursor {
      size_t block = 0;
      size_t exclusive_index = 0;
      uint8_t valu_distance = 0;
      bool nearest_valu_seen = false;
    };
    std::vector<Cursor> worklist = {
        {.block = initial->second.first, .exclusive_index = initial->second.second}};
    std::set<std::tuple<size_t, size_t, uint8_t, bool>> visited;

    while (!worklist.empty()) {
      Cursor cursor = worklist.back();
      worklist.pop_back();
      if (!visited
               .emplace(cursor.block, cursor.exclusive_index, cursor.valu_distance,
                        cursor.nearest_valu_seen)
               .second) {
        continue;
      }

      const auto &instructions = instructions_[cursor.block];
      size_t index = cursor.exclusive_index;
      bool expired = false;
      while (index != 0) {
        const Instruction &previous = *instructions[--index];
        if (!is_valu(previous))
          continue;

        if (!cursor.nearest_valu_seen) {
          cursor.nearest_valu_seen = true;
          if (is_trans(previous)) {
            const PhysicalVgprAccesses accesses = physical_vgpr_accesses(previous, vgpr_msb_);
            // TRANS write -> generated VALU read, or TRANS read -> generated
            // VALU write. A write/write pair is not a co-execution conflict.
            result.uses_requiring_nops[1] |= accesses.defs;
            result.defs_requiring_nops[1] |= accesses.uses;
          }
        }

        if (is_wmma(previous)) {
          const uint8_t required = wmma_to_valu_distance(previous);
          if (cursor.valu_distance < required) {
            const uint8_t missing = static_cast<uint8_t>(required - cursor.valu_distance);
            const WmmaCoexecutionAccesses accesses = wmma_coexecution_accesses(previous, vgpr_msb_);
            result.uses_requiring_nops[missing] |= accesses.definition;
            result.defs_requiring_nops[missing] |= accesses.definition_conflicts;
          }
        }

        ++cursor.valu_distance;
        if (cursor.valu_distance >= Gfx1250GeneratedValuHazard::kMaxNops) {
          expired = true;
          break;
        }
      }
      if (expired)
        continue;

      for (size_t predecessor : predecessors_[cursor.block]) {
        worklist.push_back({.block = predecessor,
                            .exclusive_index = instructions_[predecessor].size(),
                            .valu_distance = cursor.valu_distance,
                            .nearest_valu_seen = cursor.nearest_valu_seen});
      }
    }
    return result;
  }

  [[nodiscard]] std::vector<Gfx1250InstructionHazard> find_instruction_hazards() const {
    struct ConsumerHazards {
      std::optional<Gfx1250InstructionHazard> trans;
      std::optional<Gfx1250InstructionHazard> wmma;
    };
    std::unordered_map<const Instruction *, ConsumerHazards> hazards_by_consumer;
    std::unordered_map<const Instruction *, PhysicalVgprAccesses> physical_access_cache;
    std::unordered_map<const Instruction *, WmmaCoexecutionAccesses> wmma_access_cache;

    const auto physical_accesses = [&](const Instruction &inst) -> const PhysicalVgprAccesses & {
      auto [it, inserted] = physical_access_cache.try_emplace(&inst);
      if (inserted)
        it->second = physical_vgpr_accesses(inst, vgpr_msb_);
      return it->second;
    };
    const auto wmma_accesses = [&](const Instruction &inst) -> const WmmaCoexecutionAccesses & {
      auto [it, inserted] = wmma_access_cache.try_emplace(&inst);
      if (inserted)
        it->second = wmma_coexecution_accesses(inst, vgpr_msb_);
      return it->second;
    };
    const auto record = [](std::optional<Gfx1250InstructionHazard> &destination,
                           Gfx1250InstructionHazard candidate) {
      if (!destination || candidate.missing_nops() > destination->missing_nops())
        destination = candidate;
    };

    struct Cursor {
      size_t block = 0;
      size_t next_index = 0;
      uint8_t valu_distance = 0;
    };
    std::vector<Cursor> worklist;
    // The generated-VALU maximum bounds the reachable forward state today.
    // Retain one defensive state beyond it so a future consumer-specific
    // distance cannot make the worklist index depend on scan-loop ordering.
    std::vector<std::array<uint32_t, Gfx1250GeneratedValuHazard::kMaxNops + 2>> visited_epochs(
        instructions_.size());
    uint32_t traversal_epoch = 0;

    // Timing windows are sparse in real code. Walk forward from each producer
    // for its bounded lifetime instead of walking backward from every VALU in
    // the kernel.
    for (size_t producer_block = 0; producer_block < instructions_.size(); ++producer_block) {
      const auto &block_instructions = instructions_[producer_block];
      for (size_t producer_index = 0; producer_index < block_instructions.size();
           ++producer_index) {
        const Instruction *producer_ptr = block_instructions[producer_index];
        if (producer_ptr == nullptr)
          continue;
        const Instruction &producer = *producer_ptr;
        const bool producer_is_trans = is_trans(producer);
        const bool producer_is_wmma = is_wmma(producer);
        if (!producer_is_trans && !producer_is_wmma)
          continue;

        const uint8_t max_distance =
            producer_is_wmma ? static_cast<uint8_t>(wmma_to_valu_distance(producer) + 1u) : 1u;
        const auto scan_block = [&](Cursor &cursor) {
          const auto &instructions = instructions_[cursor.block];
          size_t index = cursor.next_index;
          while (index < instructions.size()) {
            const Instruction &current = *instructions[index++];
            if (!is_valu(current))
              continue;

            const bool current_is_wmma = is_wmma(current);
            const bool current_is_trans = is_trans(current);
            if (producer_is_trans) {
              if (!current_is_wmma && !current_is_trans) {
                const PhysicalVgprAccesses &producer_accesses = physical_accesses(producer);
                const PhysicalVgprAccesses &current_accesses = physical_accesses(current);
                if (current_accesses.uses.intersects(producer_accesses.defs) ||
                    current_accesses.defs.intersects(producer_accesses.uses)) {
                  record(hazards_by_consumer[&current].trans,
                         {.kind = Gfx1250InstructionHazardKind::TransCoexecution,
                          .producer_offset = producer.src_loc(),
                          .consumer_offset = current.src_loc(),
                          .required_slots = 1,
                          .existing_valu_slots = 0});
                }
              }
              return true;
            }

            if (!current_is_trans || current_is_wmma) {
              const uint8_t required = static_cast<uint8_t>(wmma_to_valu_distance(producer) +
                                                            (current_is_wmma ? 1u : 0u));
              const WmmaCoexecutionAccesses &producer_accesses = wmma_accesses(producer);
              const bool overlaps =
                  current_is_wmma
                      ? producer_accesses.definition.intersects(wmma_accesses(current).inputs)
                      : (physical_accesses(current).uses.intersects(producer_accesses.definition) ||
                         physical_accesses(current).defs.intersects(
                             producer_accesses.definition_conflicts));
              if (overlaps && cursor.valu_distance < required) {
                record(hazards_by_consumer[&current].wmma,
                       {.kind = Gfx1250InstructionHazardKind::WmmaCoexecution,
                        .producer_offset = producer.src_loc(),
                        .consumer_offset = current.src_loc(),
                        .required_slots = required,
                        .existing_valu_slots = cursor.valu_distance});
              }
            }

            ++cursor.valu_distance;
            if (cursor.valu_distance >= max_distance)
              return true;
          }
          return false;
        };

        // Most timing windows expire in their producer's block. Avoid allocating
        // and maintaining a CFG worklist for that common case.
        Cursor initial = {
            .block = producer_block, .next_index = producer_index + 1u, .valu_distance = 0};
        if (scan_block(initial))
          continue;

        if (++traversal_epoch == 0) {
          for (auto &epochs : visited_epochs)
            epochs.fill(0);
          traversal_epoch = 1;
        }
        worklist.clear();
        for (size_t successor : successors_[producer_block])
          worklist.push_back(
              {.block = successor, .next_index = 0, .valu_distance = initial.valu_distance});

        while (!worklist.empty()) {
          Cursor cursor = worklist.back();
          worklist.pop_back();
          uint32_t &visited = visited_epochs[cursor.block][cursor.valu_distance];
          if (visited == traversal_epoch)
            continue;
          visited = traversal_epoch;
          if (scan_block(cursor))
            continue;
          for (size_t successor : successors_[cursor.block]) {
            worklist.push_back(
                {.block = successor, .next_index = 0, .valu_distance = cursor.valu_distance});
          }
        }
      }
    }

    std::vector<Gfx1250InstructionHazard> hazards;
    for (const auto &block_instructions : instructions_) {
      for (const Instruction *inst : block_instructions) {
        const auto found = hazards_by_consumer.find(inst);
        if (found == hazards_by_consumer.end())
          continue;
        if (found->second.trans)
          hazards.push_back(*found->second.trans);
        if (found->second.wmma)
          hazards.push_back(*found->second.wmma);
      }
    }
    return hazards;
  }

private:
  Gfx1250VgprMsbAnalysis vgpr_msb_;
  std::vector<std::vector<size_t>> predecessors_;
  std::vector<std::vector<size_t>> successors_;
  std::vector<std::vector<const Instruction *>> instructions_;
  std::unordered_map<const Instruction *, std::pair<size_t, size_t>> instruction_position_;
};

Gfx1250RewriteHazardAnalysis::Gfx1250RewriteHazardAnalysis(
    KernelBlockScope blocks, BasicBlock *entry, std::span<const ScopedCfgEdge> extra_edges,
    std::span<const uint8_t> text)
    : impl_(std::make_unique<Impl>(blocks, entry, extra_edges, text)) {}

Gfx1250RewriteHazardAnalysis::~Gfx1250RewriteHazardAnalysis() = default;
Gfx1250RewriteHazardAnalysis::Gfx1250RewriteHazardAnalysis(
    Gfx1250RewriteHazardAnalysis &&) noexcept = default;
Gfx1250RewriteHazardAnalysis &
Gfx1250RewriteHazardAnalysis::operator=(Gfx1250RewriteHazardAnalysis &&) noexcept = default;

Gfx1250GeneratedValuHazard
Gfx1250RewriteHazardAnalysis::before_generated_valu(const Instruction &inst) const {
  return impl_->before_generated_valu(inst);
}

std::vector<Gfx1250InstructionHazard>
Gfx1250RewriteHazardAnalysis::find_instruction_hazards() const {
  return impl_->find_instruction_hazards();
}

} // namespace rocjitsu
