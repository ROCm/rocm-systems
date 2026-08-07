// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/analysis/gfx1250_vgpr_msb.h"

#include "rocjitsu/code/basic_block.h"
#include "rocjitsu/isa/arch/amdgpu/gfx1250/opcodes.h"
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
constexpr size_t kRoleCount = 4;

/// @brief Abstract state immediately before or after an instruction.
struct VgprMsbState {
  bool reachable = false;
  // nullopt is the top value: more than one bank can reach this point.
  std::array<std::optional<uint8_t>, kRoleCount> banks{};

  bool operator==(const VgprMsbState &) const = default;
};

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

/// @brief MODE bit offset of each two-bit bank field.
///
/// MODE[19:12] is ordered {SRC2,SRC1,SRC0,DST}, unlike the immediate
/// S_SET_VGPR_MSB byte, which is ordered {DST,SRC2,SRC1,SRC0}.
constexpr std::array<uint8_t, kRoleCount> kModeBitOffset = {14, 16, 18, 12};

/// @brief The MODE hardware register id in an S_SETREG* HWREG immediate.
constexpr uint16_t kModeHwreg = 1;

/// @brief Decoded fields of an S_SETREG* HWREG immediate: register id, and the
/// [begin, begin+width) bit slice it writes.
struct HwregSlice {
  uint16_t id;
  uint16_t begin;
  uint16_t width;
};

/// @brief Decode the HWREG immediate fields (id[5:0], offset[10:6], size-1[15:11]).
[[nodiscard]] constexpr HwregSlice decode_hwreg(uint16_t hwreg) {
  return HwregSlice{.id = static_cast<uint16_t>(hwreg & 0x3f),
                    .begin = static_cast<uint16_t>((hwreg >> 6) & 0x1f),
                    .width = static_cast<uint16_t>(((hwreg >> 11) & 0x1f) + 1)};
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
  return changed;
}

/// @brief Apply a scalar write to one WAVE_MODE bit slice.
void apply_mode_write(VgprMsbState &state, uint16_t hwreg, std::optional<uint32_t> value) {
  const HwregSlice slice = decode_hwreg(hwreg);
  const uint16_t begin = slice.begin;
  if (slice.id != kModeHwreg || begin >= 32)
    return;
  const uint16_t width = std::min<uint16_t>(slice.width, static_cast<uint16_t>(32 - begin));
  const uint16_t end = static_cast<uint16_t>(begin + width);

  for (size_t role = 0; role < kRoleCount; ++role) {
    const uint16_t field_begin = kModeBitOffset[role];
    const uint16_t field_end = static_cast<uint16_t>(field_begin + 2);
    if (begin >= field_end || field_begin >= end)
      continue;

    if (!value) {
      state.banks[role] = std::nullopt;
      continue;
    }

    const uint16_t overlap_begin = std::max(begin, field_begin);
    const uint16_t overlap_end = std::min(end, field_end);
    // A literal write covering both bits determines the bank even if the
    // incoming state was ambiguous. A partial write can retain the untouched
    // bit only when that incoming state is known.
    if (overlap_begin == field_begin && overlap_end == field_end) {
      const uint8_t source_bit = static_cast<uint8_t>(field_begin - begin);
      state.banks[role] = static_cast<uint8_t>((*value >> source_bit) & 0x3u);
      continue;
    }
    if (!state.banks[role])
      continue;
    uint8_t bank = *state.banks[role];
    for (uint16_t mode_bit = overlap_begin; mode_bit < overlap_end; ++mode_bit) {
      const uint8_t field_bit = static_cast<uint8_t>(mode_bit - field_begin);
      const uint8_t source_bit = static_cast<uint8_t>(mode_bit - begin);
      const uint8_t bit = static_cast<uint8_t>((*value >> source_bit) & 1u);
      bank = static_cast<uint8_t>((bank & ~(uint8_t{1} << field_bit)) | (bit << field_bit));
    }
    state.banks[role] = bank;
  }
}

/// @brief Model gfx1250's unmasked VGPR-MSB side effect for immediate MODE writes.
///
/// S_SETREG_IMM32_B32 targeting MODE updates MODE[19:12] from the same literal
/// bits even when those fields are outside the instruction's requested bit
/// slice. The intended banks are carried in literal bits [19:12]. Model that
/// result rather than the architectural mask.
void apply_immediate_mode_vgpr_msb_side_effect(VgprMsbState &state, uint16_t hwreg,
                                               uint32_t literal) {
  if (decode_hwreg(hwreg).id != kModeHwreg)
    return;
  for (size_t role = 0; role < kRoleCount; ++role)
    state.banks[role] = static_cast<uint8_t>((literal >> kModeBitOffset[role]) & 0x3u);
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

void transfer_instruction(VgprMsbState &state, const Instruction &inst,
                          std::span<const uint8_t> text) {
  if (inst.opcode() == gfx1250::kSSetVgprMsbSopp && inst.mnemonic() == "s_set_vgpr_msb") {
    const Operand *immediate = inst.src_operand(0);
    if (immediate == nullptr) {
      state.banks.fill(std::nullopt);
      return;
    }
    const uint8_t value = static_cast<uint8_t>(immediate->encoding_value() & 0xff);
    state.banks[kSrc0] = value & 0x3u;
    state.banks[kSrc1] = (value >> 2) & 0x3u;
    state.banks[kSrc2] = (value >> 4) & 0x3u;
    state.banks[kDst] = (value >> 6) & 0x3u;
    return;
  }

  if (inst.mnemonic() != "s_setreg_b32" && inst.mnemonic() != "s_setreg_imm32_b32")
    return;
  const Operand *hwreg_operand = inst.dst_operand(0);
  if (hwreg_operand == nullptr)
    return;
  const uint16_t hwreg = static_cast<uint16_t>(hwreg_operand->encoding_value());

  if (inst.mnemonic() == "s_setreg_b32") {
    // The source SGPR is runtime data. Only bank fields intersecting the write
    // become unknown; disjoint MODE fields retain their proven values.
    apply_mode_write(state, hwreg, std::nullopt);
    return;
  }

  // Read the 32-bit immediate from the text image at src_loc()+4 rather than
  // raw_encoding()[1]: raw_encoding() points at the 4-byte SOPK inst_ subobject,
  // so [1] indexes past it and only happens to land on the adjacent literal_
  // member — a layout-dependent out-of-bounds read. The literal follows the
  // encoding word in the instruction stream, so the text image is authoritative.
  const std::optional<uint32_t> literal = text_word_at(text, inst.src_loc() + sizeof(uint32_t));
  if (!literal || inst.size() < 2 * static_cast<int>(sizeof(uint32_t))) {
    apply_mode_write(state, hwreg, std::nullopt);
    if (decode_hwreg(hwreg).id == kModeHwreg)
      state.banks.fill(std::nullopt);
    return;
  }
  apply_mode_write(state, hwreg, *literal);
  apply_immediate_mode_vgpr_msb_side_effect(state, hwreg, *literal);
}

} // namespace

class Gfx1250VgprMsbAnalysis::Impl {
public:
  Impl(KernelBlockScope blocks, BasicBlock *entry, std::span<const ScopedCfgEdge> extra_edges,
       std::span<const uint8_t> text)
      : text_(text) {
    analyze(blocks, entry, extra_edges);
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
               std::span<const ScopedCfgEdge> extra_edges) {
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

    while (!worklist.empty()) {
      const size_t index = worklist.front();
      worklist.pop_front();
      queued[index] = false;
      BasicBlock *block = blocks[index];
      if (block == nullptr || !in[index].reachable)
        continue;

      VgprMsbState state = in[index];
      for (const Instruction &inst : block->instructions())
        transfer_instruction(state, inst, text_);
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
        transfer_instruction(state, inst, text_);
      }
    }
  }

  std::span<const uint8_t> text_;
  std::unordered_map<const Instruction *, VgprMsbState> before_;
};

Gfx1250VgprMsbAnalysis::Gfx1250VgprMsbAnalysis(KernelBlockScope blocks, BasicBlock *entry,
                                               std::span<const ScopedCfgEdge> extra_edges,
                                               std::span<const uint8_t> text)
    : impl_(std::make_unique<Impl>(blocks, entry, extra_edges, text)) {}

Gfx1250VgprMsbAnalysis::~Gfx1250VgprMsbAnalysis() = default;
Gfx1250VgprMsbAnalysis::Gfx1250VgprMsbAnalysis(Gfx1250VgprMsbAnalysis &&) noexcept = default;
Gfx1250VgprMsbAnalysis &
Gfx1250VgprMsbAnalysis::operator=(Gfx1250VgprMsbAnalysis &&) noexcept = default;

std::optional<uint8_t> Gfx1250VgprMsbAnalysis::bank_before(const Instruction &inst,
                                                           amdgpu::VgprMsbRole role) const {
  return impl_->bank_before(inst, role);
}

} // namespace rocjitsu
