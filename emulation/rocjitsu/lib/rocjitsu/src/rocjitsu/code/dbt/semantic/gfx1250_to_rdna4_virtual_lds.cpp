// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/dbt/semantic/gfx1250_to_rdna4_virtual_lds.h"

#include "rocjitsu/analysis/liveness.h"
#include "rocjitsu/code/dbt/virtual_lds_abi.h"
#include "rocjitsu/code/patch/instruction_builder.h"
#include "rocjitsu/isa/arch/amdgpu/gfx1250/encodings.h"
#include "rocjitsu/isa/arch/amdgpu/gfx1250/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/gfx1250/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/rdna4/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/rdna4/opcodes.h"
#include "rocjitsu/isa/instruction.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <optional>
#include <string>
#include <vector>

namespace rocjitsu {
namespace {

constexpr uint8_t kRdna4VglobalEncoding = 0xEE;
constexpr uint8_t kRdna4SmemEncoding = 0x3D;
constexpr uint8_t kRdna4SmemSoffsetNone = 0x7C;
constexpr uint16_t kInlineZero = 128;
constexpr uint16_t kInlineShift16 = 144;
constexpr uint16_t kVgprSrcBase = 256;

struct VirtualDsOp {
  uint16_t target_op = 0;
  bool load = false;
  uint8_t vgprs = 1;
};

[[nodiscard]] bool is_gfx1250_vds(const Instruction &inst) {
  return inst.encoding_id() >= gfx1250::encoding::kVds &&
         inst.encoding_id() <= gfx1250::encoding::kVdsOpHi7;
}

[[nodiscard]] std::optional<VirtualDsOp> virtual_ds_op(uint16_t op) {
  switch (op) {
  case gfx1250::kDsStoreB8Vds:
    return VirtualDsOp{rdna4::kGlobalStoreB8Vglobal, false, 1};
  case gfx1250::kDsStoreB16Vds:
    return VirtualDsOp{rdna4::kGlobalStoreB16Vglobal, false, 1};
  case gfx1250::kDsStoreB32Vds:
    return VirtualDsOp{rdna4::kGlobalStoreB32Vglobal, false, 1};
  case gfx1250::kDsStoreB64Vds:
    return VirtualDsOp{rdna4::kGlobalStoreB64Vglobal, false, 2};
  case gfx1250::kDsStoreB96Vds:
    return VirtualDsOp{rdna4::kGlobalStoreB96Vglobal, false, 3};
  case gfx1250::kDsStoreB128Vds:
    return VirtualDsOp{rdna4::kGlobalStoreB128Vglobal, false, 4};
  case gfx1250::kDsLoadI8Vds:
    return VirtualDsOp{rdna4::kGlobalLoadI8Vglobal, true, 1};
  case gfx1250::kDsLoadU8Vds:
    return VirtualDsOp{rdna4::kGlobalLoadU8Vglobal, true, 1};
  case gfx1250::kDsLoadI16Vds:
    return VirtualDsOp{rdna4::kGlobalLoadI16Vglobal, true, 1};
  case gfx1250::kDsLoadU16Vds:
    return VirtualDsOp{rdna4::kGlobalLoadU16Vglobal, true, 1};
  case gfx1250::kDsLoadB32Vds:
    return VirtualDsOp{rdna4::kGlobalLoadB32Vglobal, true, 1};
  case gfx1250::kDsLoadB64Vds:
    return VirtualDsOp{rdna4::kGlobalLoadB64Vglobal, true, 2};
  case gfx1250::kDsLoadB96Vds:
    return VirtualDsOp{rdna4::kGlobalLoadB96Vglobal, true, 3};
  case gfx1250::kDsLoadB128Vds:
    return VirtualDsOp{rdna4::kGlobalLoadB128Vglobal, true, 4};
  default:
    return std::nullopt;
  }
}

void append_smem_load(std::vector<uint32_t> &words, uint16_t sdata, uint16_t sbase, uint32_t dwords,
                      uint32_t byte_offset) {
  uint8_t op = 0;
  switch (dwords) {
  case 2:
    op = 1;
    break;
  case 4:
    op = 2;
    break;
  default:
    break;
  }
  // RDNA4 SMEM stores the base as an SGPR-pair index, not an absolute SGPR.
  words.push_back(static_cast<uint32_t>((sbase / 2u) & 0x3Fu) |
                  (static_cast<uint32_t>(sdata & 0x7Fu) << 6) | (static_cast<uint32_t>(op) << 13) |
                  (static_cast<uint32_t>(kRdna4SmemEncoding) << 26));
  words.push_back((byte_offset & 0x00FF'FFFFu) |
                  (static_cast<uint32_t>(kRdna4SmemSoffsetNone) << 25));
}

void append_salu(std::vector<uint32_t> &words, uint32_t word) {
  words.push_back(word);
  words.push_back(build_s_delay_alu(kDelayAluSaluDep1, ROCJITSU_CODE_ARCH_RDNA4));
}

void append_stride(std::vector<uint32_t> &words, uint16_t backing_base, uint16_t state_base,
                   uint16_t temp, uint16_t workgroup, uint32_t state_offset) {
  append_smem_load(words, temp, state_base, 1, state_offset);
  words.push_back(pack_sopp(rdna4::kSWaitLoadcntSopp, 0));
  words.push_back(pack_sopp(rdna4::kSWaitKmcntSopp, 0));
  append_salu(words, pack_sop2(rdna4::kSMulI32Sop2, temp, workgroup, temp));
  append_salu(words, pack_sop2(rdna4::kSAddCoU32Sop2, backing_base, backing_base, temp));
  append_salu(words,
              pack_sop2(rdna4::kSAddCoCiU32Sop2, backing_base + 1, backing_base + 1, kInlineZero));
}

[[nodiscard]] constexpr uint32_t build_vop1(uint8_t op, uint8_t vdst, uint16_t src0) {
  return (src0 & 0x1FFu) | ((op & 0x7Fu) << 9) | ((vdst & 0xFFu) << 17) | (0x3Fu << 25);
}

[[nodiscard]] bool overlaps(uint16_t lhs, uint16_t lhs_count, uint16_t rhs, uint16_t rhs_count) {
  return lhs < rhs + rhs_count && rhs < lhs + lhs_count;
}

[[nodiscard]] std::optional<uint8_t> find_address_pair(const Instruction &inst,
                                                       const LivenessAnalysis &liveness,
                                                       const gfx1250::VdsMachineInst &src,
                                                       const VirtualDsOp &op) {
  uint16_t search_start = 0;
  while (const auto candidate = liveness.find_free_run(&inst, 2, search_start, 2)) {
    if (*candidate + 1u > UINT8_MAX)
      return std::nullopt;
    const bool overlaps_address = overlaps(*candidate, 2, src.addr, 1);
    const uint16_t value_base = op.load ? src.vdst : src.data0;
    const bool overlaps_value = overlaps(*candidate, 2, value_base, op.vgprs);
    if (!overlaps_address && !overlaps_value)
      return static_cast<uint8_t>(*candidate);
    search_start = static_cast<uint16_t>(*candidate + 2u);
  }
  return std::nullopt;
}

[[nodiscard]] std::vector<uint32_t> encode_global(const gfx1250::VdsMachineInst &src,
                                                  const VirtualDsOp &op, uint16_t base_sgpr,
                                                  uint8_t address_pair) {
  constexpr uint8_t kVMovB32 = 1;
  std::vector<uint32_t> words{
      build_vop1(kVMovB32, address_pair, static_cast<uint16_t>(kVgprSrcBase + src.addr)),
      build_vop1(kVMovB32, static_cast<uint8_t>(address_pair + 1u), kInlineZero),
      build_s_wait_alu(kWaitAluDepctrVaVdstVmVsrc0, ROCJITSU_CODE_ARCH_RDNA4),
  };

  rdna4::VglobalMachineInst dst{};
  dst.encoding = kRdna4VglobalEncoding;
  dst.op = op.target_op;
  dst.saddr = base_sgpr;
  dst.vaddr = address_pair;
  dst.ioffset = (static_cast<uint32_t>(src.offset1) << 8u) | src.offset0;
  // A WGP-mode workgroup can span sibling CUs with distinct L0 caches.  SE
  // scope is therefore the narrowest safe replacement for workgroup-visible
  // LDS traffic on GFX12; CU scope can return stale data across those waves.
  dst.scope = 1;
  dst.th = 0;
  if (op.load)
    dst.vdst = src.vdst;
  else
    dst.vsrc = src.data0;

  std::array<uint32_t, sizeof(dst) / sizeof(uint32_t)> encoded{};
  std::memcpy(encoded.data(), &dst, sizeof(dst));
  if (op.load) {
    // The source DS wait/barrier sequence provides a workgroup acquire. In
    // GFX12 WGP mode sibling waves can execute on CUs with distinct L0s, so
    // SE scope on the load is not sufficient by itself: invalidate the
    // workgroup-visible cache domain before consuming virtual-LDS data.
    rdna4::VglobalMachineInst invalidate{};
    invalidate.encoding = kRdna4VglobalEncoding;
    invalidate.op = rdna4::kGlobalInvVglobal;
    invalidate.scope = 1;
    // GLOBAL_INV is the short two-dword VGLOBAL form; the ordinary memory
    // operations below use the three-dword form.
    std::array<uint32_t, 2> invalidate_encoded{};
    std::memcpy(invalidate_encoded.data(), &invalidate, sizeof(invalidate_encoded));
    words.insert(words.end(), invalidate_encoded.begin(), invalidate_encoded.end());
    words.push_back(pack_sopp(rdna4::kSWaitLoadcntSopp, 0));
  }
  words.insert(words.end(), encoded.begin(), encoded.end());
  // Make each replacement self-synchronizing. Source DS waits account against
  // LGKM counters, whereas the replacement accounts against VM counters.
  words.push_back(pack_sopp(op.load ? rdna4::kSWaitLoadcntSopp : rdna4::kSWaitStorecntSopp, 0));
  if (op.load) {
    words.push_back(build_s_wait_alu(kWaitAluDepctrVaVdstVmVsrc0, ROCJITSU_CODE_ARCH_RDNA4));
  }
  return words;
}

} // namespace

bool gfx1250_to_rdna4_supports_virtual_lds() { return true; }

bool gfx1250_source_instruction_uses_virtualizable_lds(const Instruction &inst) {
  return is_gfx1250_vds(inst) && virtual_ds_op(inst.opcode()).has_value();
}

std::optional<VirtualLdsBaseSgprReservation>
reserve_rdna4_virtual_lds_base_sgpr_pair(TranslationContext &context,
                                         const KdTranslation &translation) {
  uint32_t floor = translation.target_source_sgpr_count;
  if (translation.rdna4_grid_x_sgpr >= 0)
    floor = std::max(floor, static_cast<uint32_t>(translation.rdna4_grid_x_sgpr + 1));
  if (translation.rdna4_grid_yz_sgpr >= 0)
    floor = std::max(floor, static_cast<uint32_t>(translation.rdna4_grid_yz_sgpr + 1));
  const uint32_t base = (floor + 1u) & ~1u;
  // Four registers hold the backing pointer plus one arithmetic temp pair.
  if (base + 4u > translation.target_sgpr_count || base + 4u > 106u)
    return std::nullopt;
  context.require_sgprs(base + 4u);
  return VirtualLdsBaseSgprReservation{.base = static_cast<uint16_t>(base),
                                       .prologue_temp = static_cast<uint16_t>(base + 2u)};
}

bool append_rdna4_virtual_lds_entry_prologue(KdTranslation &translation) {
  if (!translation.needs_virtual_lds_buffer || !translation.source_has_kernarg_segment_ptr)
    return false;
  const uint16_t base = translation.virtual_lds_lowering.base_sgpr;
  const uint16_t temp = translation.virtual_lds_lowering.prologue_temp_sgpr;
  const uint16_t kernarg = translation.kernarg_segment_ptr_sgpr;
  if ((base & 1u) != 0 || base + 3u > 105u || kernarg + 1u > 105u ||
      translation.rdna4_grid_x_sgpr < 0 || translation.rdna4_grid_yz_sgpr < 0)
    return false;

  auto &words = translation.prologue_words;
  const uint32_t state = translation.virtual_lds_kernarg_pointer_offset;
  append_smem_load(words, base, kernarg, 2, state + kVirtualLdsStateBackingBaseOffset);
  words.push_back(pack_sopp(rdna4::kSWaitLoadcntSopp, 0));
  words.push_back(pack_sopp(rdna4::kSWaitKmcntSopp, 0));

  append_stride(words, base, kernarg, temp, static_cast<uint16_t>(translation.rdna4_grid_x_sgpr),
                state + kVirtualLdsStateStrideXOffset);

  const uint16_t packed_yz = static_cast<uint16_t>(translation.rdna4_grid_yz_sgpr);
  append_salu(words, build_s_mov_b32(temp + 1, packed_yz, ROCJITSU_CODE_ARCH_RDNA4));
  append_salu(words,
              build_s_lshl_b32(temp + 1, temp + 1, kInlineShift16, ROCJITSU_CODE_ARCH_RDNA4));
  append_salu(words,
              build_s_lshr_b32(temp + 1, temp + 1, kInlineShift16, ROCJITSU_CODE_ARCH_RDNA4));
  append_stride(words, base, kernarg, temp, temp + 1, state + kVirtualLdsStateStrideYOffset);

  append_salu(words,
              build_s_lshr_b32(temp + 1, packed_yz, kInlineShift16, ROCJITSU_CODE_ARCH_RDNA4));
  append_stride(words, base, kernarg, temp, temp + 1, state + kVirtualLdsStateStrideZOffset);

  // The guest body must observe its original pointer, not the wrapper pointer.
  append_smem_load(words, kernarg, kernarg, 2, translation.kernarg_wrapper_original_pointer_offset);
  words.push_back(pack_sopp(rdna4::kSWaitLoadcntSopp, 0));
  words.push_back(pack_sopp(rdna4::kSWaitKmcntSopp, 0));
  return true;
}

ExpandResult lower_gfx1250_to_rdna4_virtual_lds_instruction(const Instruction &inst,
                                                            const LivenessAnalysis &liveness,
                                                            TranslationContext &context) {
  if (!context.virtualize_lds || !is_gfx1250_vds(inst))
    return ExpandResult::not_handled();
  const uint32_t *raw = inst.raw_encoding();
  if (raw == nullptr || static_cast<size_t>(inst.size()) < sizeof(gfx1250::VdsMachineInst))
    return ExpandResult::failed(std::string(inst.mnemonic()) +
                                ": missing GFX1250 VDS source encoding");
  gfx1250::VdsMachineInst src{};
  std::memcpy(&src, raw, sizeof(src));
  const auto op = virtual_ds_op(src.op);
  if (!op) {
    if ((inst.flags() & MEMORY_OP) == 0)
      return ExpandResult::not_handled();
    return ExpandResult::failed(
        std::string(inst.mnemonic()) +
            ": full LDS virtualization does not support this storage operation",
        {"Add an explicit GFX1250 VDS-to-RDNA4 VGLOBAL lowering for this opcode."});
  }
  if ((context.virtual_lds_base_sgpr & 1u) != 0 || context.virtual_lds_base_sgpr > 105)
    return ExpandResult::failed(std::string(inst.mnemonic()) +
                                ": virtual LDS base SGPR pair is not encodable");
  const auto address_pair = find_address_pair(inst, liveness, src, *op);
  if (!address_pair) {
    return ExpandResult::failed(
        std::string(inst.mnemonic()) +
            ": full LDS virtualization cannot reserve a 64-bit VGLOBAL address pair",
        {"Reserve or spill an even RDNA4 VGPR pair for the zero-extended LDS address."});
  }
  context.require_vgprs(static_cast<uint32_t>(*address_pair) + 2u);
  context.require_vgprs(static_cast<uint32_t>(src.addr) + 1u);
  if (op->load)
    context.require_vgprs(static_cast<uint32_t>(src.vdst) + op->vgprs);
  else
    context.require_vgprs(static_cast<uint32_t>(src.data0) + op->vgprs);
  return ExpandResult::success(
      encode_global(src, *op, context.virtual_lds_base_sgpr, *address_pair));
}

} // namespace rocjitsu
