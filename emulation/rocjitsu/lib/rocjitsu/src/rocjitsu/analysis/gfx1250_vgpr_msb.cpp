// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/analysis/gfx1250_vgpr_msb.h"

#include "rocjitsu/code/basic_block.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/opcodes.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/isa/operand.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

namespace rocjitsu {

namespace {

constexpr size_t kSrc0 = 0;
constexpr size_t kSrc1 = 1;
constexpr size_t kSrc2 = 2;
constexpr size_t kDst = 3;

enum class AdjacentSetVgprMsbHazard : uint8_t { Clear, Armed, Maybe };

/// @brief Abstract state immediately before or after an instruction.
struct VgprMsbState {
  bool reachable = false;
  // nullopt is the top value: more than one bank can reach this point.
  amdgpu::VgprMsbBanks banks{};
  AdjacentSetVgprMsbHazard hazard = AdjacentSetVgprMsbHazard::Clear;

  bool operator==(const VgprMsbState &) const = default;
};

/// @brief Architectural VGPR_MSB state at a function entry: all four banks zero.
///
/// @details This is an ABI guarantee rather than an assumption. LLVM documents it in
/// AMDGPULowerVGPREncoding ("the ABI is set to expect all 4 MSBs to be zero on entry") and
/// enforces it by resetting the mode before every call and every terminator, so a callee is
/// always entered with the banks cleared. The same state therefore seeds a kernel entry and any
/// address-taken device function adopted as an additional root.
[[nodiscard]] VgprMsbState entry_state() {
  VgprMsbState state;
  state.reachable = true;
  state.banks.fill(uint8_t{0});
  return state;
}

[[nodiscard]] std::optional<size_t> role_index(amdgpu::VgprMsbRole role) {
  switch (role) {
  case amdgpu::VgprMsbRole::Src0:
    return kSrc0;
  case amdgpu::VgprMsbRole::Src1:
    return kSrc1;
  case amdgpu::VgprMsbRole::Src2:
    return kSrc2;
  case amdgpu::VgprMsbRole::Dst:
    return kDst;
  case amdgpu::VgprMsbRole::None:
    return std::nullopt;
  }
  return std::nullopt;
}

/// @brief Join @p incoming into @p destination.
[[nodiscard]] bool merge_state(VgprMsbState &destination, const VgprMsbState &incoming) {
  if (!incoming.reachable)
    return false;
  if (!destination.reachable) {
    destination = incoming;
    return true;
  }

  bool changed = false;
  for (size_t i = 0; i < destination.banks.size(); ++i) {
    if (destination.banks[i] == incoming.banks[i])
      continue;
    if (destination.banks[i].has_value()) {
      destination.banks[i] = std::nullopt;
      changed = true;
    }
  }
  if (destination.hazard != incoming.hazard &&
      destination.hazard != AdjacentSetVgprMsbHazard::Maybe) {
    destination.hazard = AdjacentSetVgprMsbHazard::Maybe;
    changed = true;
  }
  return changed;
}

/// @brief Read a 32-bit little-endian word from the .text image at @p offset.
/// @returns nullopt when the four bytes are not fully present.
[[nodiscard]] std::optional<uint32_t> text_word_at(std::span<const uint8_t> text, uint64_t offset) {
  if (text.empty() || offset + sizeof(uint32_t) > text.size())
    return std::nullopt;
  uint32_t word = 0;
  std::memcpy(&word, text.data() + offset, sizeof(uint32_t));
  return word;
}

void set_banks_from_set_layout(amdgpu::VgprMsbBanks &banks, uint8_t value) {
  banks[kSrc0] = value & 0x3u;
  banks[kSrc1] = (value >> 2) & 0x3u;
  banks[kSrc2] = (value >> 4) & 0x3u;
  banks[kDst] = (value >> 6) & 0x3u;
}

void merge_banks_with_set_layout(amdgpu::VgprMsbBanks &banks, uint8_t value) {
  const amdgpu::VgprMsbBanks applied = amdgpu::unpack_vgpr_msb_banks(value);
  for (size_t i = 0; i < banks.size(); ++i) {
    if (banks[i] != applied[i])
      banks[i] = std::nullopt;
  }
}

void transfer_instruction(VgprMsbState &state, const Instruction &inst,
                          std::span<const uint8_t> text, bool setreg_vgpr_msb_fixup) {
  const AdjacentSetVgprMsbHazard incoming_hazard = state.hazard;
  state.hazard = AdjacentSetVgprMsbHazard::Clear;
  if (inst.opcode() == cdna5::kSSetVgprMsbSopp && inst.mnemonic() == "s_set_vgpr_msb") {
    if (setreg_vgpr_msb_fixup && incoming_hazard == AdjacentSetVgprMsbHazard::Armed)
      return;
    const Operand *immediate = inst.src_operand(0);
    if (immediate == nullptr) {
      state.banks.fill(std::nullopt);
      return;
    }
    const uint8_t value = static_cast<uint8_t>(immediate->encoding_value() & 0xff);
    if (setreg_vgpr_msb_fixup && incoming_hazard == AdjacentSetVgprMsbHazard::Maybe)
      merge_banks_with_set_layout(state.banks, value);
    else
      set_banks_from_set_layout(state.banks, value);
    return;
  }

  if (inst.mnemonic() != "s_setreg_b32" && inst.mnemonic() != "s_setreg_imm32_b32")
    return;
  const Operand *hwreg_operand = inst.dst_operand(0);
  if (hwreg_operand == nullptr)
    return;
  const uint16_t hwreg = static_cast<uint16_t>(hwreg_operand->encoding_value());
  const bool writes_mode = amdgpu::decode_vgpr_msb_hwreg(hwreg).id == amdgpu::MODE_HWREG;

  if (setreg_vgpr_msb_fixup && writes_mode) {
    if (inst.mnemonic() == "s_setreg_b32") {
      state.banks.fill(std::nullopt);
      return;
    }
    const std::optional<uint32_t> literal = text_word_at(text, inst.src_loc() + sizeof(uint32_t));
    if (!literal || inst.size() < 2 * static_cast<int>(sizeof(uint32_t))) {
      state.banks.fill(std::nullopt);
    } else {
      const uint8_t mode_layout = static_cast<uint8_t>((*literal & amdgpu::VGPR_MSB_MODE_MASK) >>
                                                       amdgpu::VGPR_MSB_MODE_SHIFT);
      set_banks_from_set_layout(state.banks, amdgpu::mode_layout_to_set_vgpr_msb(mode_layout));
    }
    state.hazard = AdjacentSetVgprMsbHazard::Armed;
    return;
  }

  if (inst.mnemonic() == "s_setreg_b32") {
    // The source SGPR is runtime data. Only bank fields intersecting the write
    // become unknown; disjoint MODE fields retain their proven values.
    amdgpu::apply_vgpr_msb_mode_write(state.banks, hwreg, std::nullopt);
    return;
  }

  // Read the 32-bit immediate from the text image at src_loc()+4. The literal
  // follows the encoding word in the instruction stream, and the text image is
  // the authoritative input for this analysis.
  const std::optional<uint32_t> literal = text_word_at(text, inst.src_loc() + sizeof(uint32_t));
  if (!literal || inst.size() < 2 * static_cast<int>(sizeof(uint32_t))) {
    amdgpu::apply_vgpr_msb_mode_write(state.banks, hwreg, std::nullopt);
    return;
  }
  amdgpu::apply_vgpr_msb_mode_write(state.banks, hwreg, *literal);
}

} // namespace

class Gfx1250VgprMsbAnalysis::Impl {
public:
  Impl(KernelBlockScope blocks, BasicBlock *entry, std::span<const ScopedCfgEdge> extra_edges,
       std::span<const uint8_t> text, std::span<BasicBlock *const> additional_entries,
       bool setreg_vgpr_msb_fixup)
      : text_(text), setreg_vgpr_msb_fixup_(setreg_vgpr_msb_fixup) {
    analyze(blocks, entry, extra_edges, additional_entries);
  }

  [[nodiscard]] std::optional<uint8_t> bank_before(const Instruction &inst,
                                                   amdgpu::VgprMsbRole role) const {
    // Operands tagged None are explicitly outside VGPR_MSB addressing. Their
    // encoded low byte therefore always names bank zero.
    if (role == amdgpu::VgprMsbRole::None)
      return uint8_t{0};
    const auto role_id = role_index(role);
    const auto state = before_.find(&inst);
    if (!role_id || state == before_.end() || !state->second.reachable)
      return std::nullopt;
    return state->second.banks[*role_id];
  }

private:
  void analyze(KernelBlockScope blocks, BasicBlock *entry,
               std::span<const ScopedCfgEdge> extra_edges,
               std::span<BasicBlock *const> additional_entries) {
    std::unordered_map<const BasicBlock *, size_t> block_index;
    block_index.reserve(blocks.size());
    for (size_t i = 0; i < blocks.size(); ++i) {
      if (blocks[i] != nullptr)
        block_index.emplace(blocks[i], i);
    }
    const auto entry_it = block_index.find(entry);
    if (entry_it == block_index.end())
      return;

    std::vector<std::vector<size_t>> successors(blocks.size());
    auto add_edge = [&](const BasicBlock *from, const BasicBlock *to) {
      const auto from_it = block_index.find(from);
      const auto to_it = block_index.find(to);
      if (from_it == block_index.end() || to_it == block_index.end())
        return;
      auto &edges = successors[from_it->second];
      if (std::ranges::find(edges, to_it->second) == edges.end())
        edges.push_back(to_it->second);
    };
    for (BasicBlock *block : blocks) {
      if (block == nullptr)
        continue;
      for (BasicBlock *successor : block->successors())
        add_edge(block, successor);
    }
    for (const ScopedCfgEdge &edge : extra_edges)
      add_edge(edge.from, edge.to);

    std::vector<VgprMsbState> in(blocks.size());
    std::vector<VgprMsbState> out(blocks.size());
    in[entry_it->second] = entry_state();

    std::deque<size_t> worklist;
    std::vector<bool> queued(blocks.size(), false);
    auto enqueue = [&](size_t index) {
      if (index >= queued.size() || queued[index])
        return;
      queued[index] = true;
      worklist.push_back(index);
    };
    enqueue(entry_it->second);

    // An adopted root is entered by a call through its address, never by an edge from this scope,
    // so the fixed point below would otherwise never give it a state.
    for (BasicBlock *additional : additional_entries) {
      const auto it = block_index.find(additional);
      if (it == block_index.end())
        continue;
      (void)merge_state(in[it->second], entry_state());
      enqueue(it->second);
    }

    while (!worklist.empty()) {
      const size_t index = worklist.front();
      worklist.pop_front();
      queued[index] = false;
      BasicBlock *block = blocks[index];
      if (block == nullptr || !in[index].reachable)
        continue;

      VgprMsbState state = in[index];
      for (const Instruction &inst : block->instructions())
        transfer_instruction(state, inst, text_, setreg_vgpr_msb_fixup_);
      if (state == out[index])
        continue;
      out[index] = state;
      for (size_t successor : successors[index]) {
        if (merge_state(in[successor], state))
          enqueue(successor);
      }
    }

    // Materialize per-instruction state only after the block fixed point is
    // reached, so queries never observe a transient predecessor ordering.
    for (size_t index = 0; index < blocks.size(); ++index) {
      BasicBlock *block = blocks[index];
      if (block == nullptr || !in[index].reachable)
        continue;
      VgprMsbState state = in[index];
      for (const Instruction &inst : block->instructions()) {
        before_.emplace(&inst, state);
        transfer_instruction(state, inst, text_, setreg_vgpr_msb_fixup_);
      }
    }
  }

  std::span<const uint8_t> text_;
  bool setreg_vgpr_msb_fixup_ = true;
  std::unordered_map<const Instruction *, VgprMsbState> before_;
};

Gfx1250VgprMsbAnalysis::Gfx1250VgprMsbAnalysis(KernelBlockScope blocks, BasicBlock *entry,
                                               std::span<const ScopedCfgEdge> extra_edges,
                                               std::span<const uint8_t> text,
                                               std::span<BasicBlock *const> additional_entries,
                                               bool setreg_vgpr_msb_fixup)
    : impl_(std::make_unique<Impl>(blocks, entry, extra_edges, text, additional_entries,
                                   setreg_vgpr_msb_fixup)) {}

Gfx1250VgprMsbAnalysis::~Gfx1250VgprMsbAnalysis() = default;
Gfx1250VgprMsbAnalysis::Gfx1250VgprMsbAnalysis(Gfx1250VgprMsbAnalysis &&) noexcept = default;
Gfx1250VgprMsbAnalysis &
Gfx1250VgprMsbAnalysis::operator=(Gfx1250VgprMsbAnalysis &&) noexcept = default;

std::optional<uint8_t> Gfx1250VgprMsbAnalysis::bank_before(const Instruction &inst,
                                                           amdgpu::VgprMsbRole role) const {
  return impl_->bank_before(inst, role);
}

} // namespace rocjitsu
