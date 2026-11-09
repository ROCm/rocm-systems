// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "generated_kernel.h"

#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/code/builders/smem_builders.h"
#include "rocjitsu/code/builders/vector_builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna1/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna1/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna2/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna2/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna3/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna3/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/operand_types.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna1/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna1/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna1/operand_types.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna2/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna2/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna2/operand_types.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/operand_types.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3_5/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3_5/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3_5/operand_types.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/operand_types.h"

#include <algorithm>
#include <array>
#include <format>
#include <initializer_list>
#include <random>
#include <stdexcept>
#include <vector>

namespace rocjitsu::execution_fuzz {
namespace {

constexpr uint8_t kGfx9FlatDisabledSaddr = 0x7f;

// The older FLAT and newer VFLAT layouts both take a VGPR address pair when
// the scalar base is disabled. Keep the address and data register banks disjoint.
template <size_t N, typename Fields>
std::array<uint32_t, N> flat_access(std::array<uint32_t, N> (*encode)(uint16_t, Fields),
                                    uint16_t opcode, uint8_t null_saddr, uint8_t addr, uint8_t reg,
                                    uint16_t offset, bool store) {
  Fields fields{};
  fields.saddr = null_saddr;
  if constexpr (requires { fields.vaddr; }) {
    fields.vaddr = addr;
    fields.vsrc = store ? reg : 0;
    fields.ioffset = offset;
  } else {
    fields.addr = addr;
    fields.data = store ? reg : 0;
    fields.offset = offset;
  }
  fields.vdst = store ? 0 : reg;
  return encode(opcode, fields);
}

enum class KernelOp {
  VectorAlu,
  ScalarAlu,
  FlatLoad,
  FlatStore,
  ScalarLoad,
  Compare,
  RestoreExec,
  Branch,
};

struct ForwardBranch {
  size_t word_index;
  size_t target_op_index;
  uint16_t opcode;
};

template <typename Vop2, typename Vopc, size_t FlatWords, typename FlatFields>
struct KernelEncoding {
  Vop2 build_vop2;
  Vopc build_vopc;
  std::array<uint32_t, FlatWords> (*build_flat)(uint16_t, FlatFields);
  std::array<uint16_t, 8> alu_ops;
  uint16_t v_and;
  uint16_t v_or;
  uint16_t cmp_lt;
  uint16_t cmp_eq;
  std::array<uint16_t, 4> branch_ops;
  uint16_t branch_scc1;
  uint16_t flat_load;
  uint16_t flat_store;
  uint8_t null_saddr;
  std::initializer_list<uint32_t> wait_load;
  std::initializer_list<uint32_t> wait_store;
};

template <typename Vop2, typename Vopc, size_t FlatWords, typename FlatFields>
void emit(std::span<uint8_t> payload, rj_code_arch_t arch, uint64_t seed,
          const KernelEncoding<Vop2, Vopc, FlatWords, FlatFields> &encoding) {
  std::mt19937_64 random(seed);
  const auto address_pair = [&] {
    return static_cast<uint8_t>(2 * (random() % kAddressPairCount));
  };
  constexpr uint8_t kTempRegisterCount = 16;
  const auto temp_register = [&] {
    return static_cast<uint8_t>(kAddressPairRegisterCount + random() % kTempRegisterCount);
  };
  uint8_t last_vgpr = kAddressPairRegisterCount;
  uint8_t last_sgpr = kAddressPairRegisterCount;
  // Prefer real producer/consumer dependencies while sampling other registers.
  // Values skipped by a branch or masked off have defined incoming values.
  const auto vector_input = [&] { return random() % 4 != 0 ? last_vgpr : temp_register(); };
  const auto scalar_input = [&] { return random() % 4 != 0 ? last_sgpr : temp_register(); };
  const uint16_t exec = scalar_operand_exec_lo(arch).value();
  constexpr uint8_t kSavedExec = 40;
  constexpr uint8_t kLoopCounter = 42;
  constexpr uint8_t kComputedAddr = kAddressPairRegisterCount + kTempRegisterCount;
  const uint32_t wait_scalar = build_wait_scalar_loads_complete(arch).value();
  std::vector<uint32_t> words{build_s_mov_b64(kSavedExec, exec, arch).value()};
  const size_t capacity = payload.size() / sizeof(uint32_t);
  words.reserve(capacity);
  auto append = [&](auto encoded) { words.insert(words.end(), encoded.begin(), encoded.end()); };
  const auto branch_word = [&](uint16_t opcode, size_t from_word, size_t to_word) {
    const int displacement = static_cast<int>(to_word) - static_cast<int>(from_word) - 1;
    return build_sopp_encoding(arch, opcode, static_cast<uint16_t>(displacement)).value();
  };
  const size_t op_count = 24 + random() % 41;
  // At most one backward edge. Its counter is initialized before any branch.
  const bool loop = random() % 2 == 0;
  const size_t loop_end = 7 + random() % (op_count - 7);
  if (loop)
    words.push_back(
        build_s_mov_b32(kLoopCounter, vop3_inline_uint(2 + random() % 2), arch).value());
  constexpr std::array kOps = {KernelOp::VectorAlu,  KernelOp::VectorAlu, KernelOp::VectorAlu,
                               KernelOp::ScalarAlu,  KernelOp::FlatLoad,  KernelOp::FlatStore,
                               KernelOp::ScalarLoad, KernelOp::Compare,   KernelOp::RestoreExec,
                               KernelOp::Branch};
  std::vector<size_t> op_offsets;
  std::vector<ForwardBranch> forward_branches;
  // A FLAT access with a computed address and its waits is the longest operation.
  // Leave one word for s_endpgm and make every branch target an operation start.
  constexpr size_t kComputedAddressWords = 3;
  const size_t max_op_words = kComputedAddressWords + FlatWords +
                              std::max(encoding.wait_load.size(), encoding.wait_store.size());
  for (size_t op_index = 0; op_index < op_count && words.size() + max_op_words + 1 <= capacity;
       ++op_index) {
    op_offsets.push_back(words.size());
    if (loop && op_index == loop_end) {
      const uint16_t minus_one = scalar_inline_neg_one(arch).value();
      words.push_back(build_s_add_u32(kLoopCounter, kLoopCounter, minus_one, arch).value());
      words.push_back(build_s_cmp_lg_u32(kLoopCounter, vop3_inline_uint(0), arch).value());
      const size_t target_op_index = op_index - (1 + random() % 6);
      words.push_back(branch_word(encoding.branch_scc1, words.size(), op_offsets[target_op_index]));
      continue;
    }
    const KernelOp op = kOps[random() % kOps.size()];
    switch (op) {
    case KernelOp::VectorAlu: {
      const uint16_t opcode = encoding.alu_ops[random() % encoding.alu_ops.size()];
      const uint16_t src = random() % 2 ? scalar_input() : 256 + temp_register();
      const uint8_t value = vector_input();
      const uint8_t dst = temp_register();
      append(encoding.build_vop2(opcode, {.src0 = src, .vsrc1 = value, .vdst = dst}));
      last_vgpr = dst;
      break;
    }
    case KernelOp::ScalarAlu: {
      const uint8_t lhs = scalar_input();
      const uint8_t rhs = temp_register();
      const uint8_t dst = temp_register();
      words.push_back(build_s_add_u32(dst, lhs, rhs, arch).value());
      last_sgpr = dst;
      break;
    }
    case KernelOp::FlatLoad:
    case KernelOp::FlatStore: {
      uint8_t addr = address_pair();
      if (random() % 4 == 0) {
        // Fold some result bits into the mapped address without losing alignment.
        const uint8_t index = vector_input();
        append(encoding.build_vop2(
            encoding.v_and, {.src0 = vop3_inline_uint(60), .vsrc1 = index, .vdst = kComputedAddr}));
        append(encoding.build_vop2(encoding.v_or, {.src0 = static_cast<uint16_t>(256 + addr),
                                                   .vsrc1 = kComputedAddr,
                                                   .vdst = kComputedAddr}));
        words.push_back(build_v_mov_b32_src(kComputedAddr + 1, vop3_inline_uint(0), arch).value());
        addr = kComputedAddr;
      }
      const bool is_store = op == KernelOp::FlatStore;
      const uint8_t reg = is_store ? vector_input() : temp_register();
      const uint16_t offset = 4 * (random() % 64);
      append(flat_access(encoding.build_flat, is_store ? encoding.flat_store : encoding.flat_load,
                         encoding.null_saddr, addr, reg, offset, is_store));
      // Drain both vector memory and LDS counters before the next branch target.
      append(is_store ? encoding.wait_store : encoding.wait_load);
      if (!is_store)
        last_vgpr = reg;
      break;
    }
    case KernelOp::ScalarLoad: {
      const uint8_t addr = address_pair();
      const uint32_t offset = 4 * (random() % 128);
      const uint8_t dst = temp_register() & ~1u;
      append(build_s_load_dwordx2(dst, addr, offset, arch).value());
      words.push_back(wait_scalar);
      last_sgpr = dst + (random() % 2);
      break;
    }
    case KernelOp::Compare: {
      const uint16_t opcode = random() % 4 == 0 ? encoding.cmp_eq : encoding.cmp_lt;
      const uint8_t scalar = scalar_input();
      const uint8_t vector = vector_input();
      append(encoding.build_vopc(opcode, {.src0 = scalar, .vsrc1 = vector}));
      break;
    }
    case KernelOp::RestoreExec:
      words.push_back(build_s_mov_b64(exec, kSavedExec, arch).value());
      break;
    case KernelOp::Branch: {
      const uint16_t opcode = encoding.branch_ops[random() % encoding.branch_ops.size()];
      forward_branches.push_back({words.size(), op_index + 2 + random() % 7, opcode});
      words.push_back(0);
      break;
    }
    }
  }
  op_offsets.push_back(words.size());
  for (const ForwardBranch &branch : forward_branches)
    words[branch.word_index] =
        branch_word(branch.opcode, branch.word_index,
                    op_offsets[std::min(branch.target_op_index, op_offsets.size() - 1)]);
  words.push_back(build_s_endpgm(arch).value());
  if (words.size() > capacity)
    throw std::logic_error("generated kernel exceeded its instruction payload");
  size_t offset = 0;
  for (uint32_t word : words)
    for (unsigned byte = 0; byte < sizeof(word); ++byte)
      payload[offset++] = static_cast<uint8_t>(word >> (8 * byte));
}

} // namespace

void write_generated_kernel(std::span<uint8_t> payload, rj_code_arch_t arch, uint64_t seed) {
  if (payload.size() < kMinGeneratedKernelBytes)
    throw std::invalid_argument(std::format(
        "generated kernels require at least {} instruction bytes", kMinGeneratedKernelBytes));
  // Pick opcodes and waits from the generated definitions for each target.
  switch (arch) {
  case ROCJITSU_CODE_ARCH_CDNA1: {
    namespace isa = cdna1;
    const uint32_t wait_all = isa::build_sopp(isa::kSWaitcntSopp)[0];
    return emit(payload, arch, seed,
                KernelEncoding{
                    .build_vop2 = isa::build_vop2,
                    .build_vopc = isa::build_vopc,
                    .build_flat = isa::build_flat,
                    .alu_ops = {isa::kVXorB32Vop2, isa::kVOrB32Vop2, isa::kVAndB32Vop2,
                                isa::kVLshlrevB32Vop2, isa::kVLshrrevB32Vop2, isa::kVMulU32U24Vop2,
                                isa::kVMinU32Vop2, isa::kVMaxU32Vop2},
                    .v_and = isa::kVAndB32Vop2,
                    .v_or = isa::kVOrB32Vop2,
                    .cmp_lt = isa::kVCmpxLtU32Vopc,
                    .cmp_eq = isa::kVCmpxEqU32Vopc,
                    .branch_ops = {isa::kSCbranchExeczSopp, isa::kSCbranchExecnzSopp,
                                   isa::kSCbranchScc0Sopp, isa::kSCbranchScc1Sopp},
                    .branch_scc1 = isa::kSCbranchScc1Sopp,
                    .flat_load = isa::kFlatLoadDwordFlat,
                    .flat_store = isa::kFlatStoreDwordFlat,
                    .null_saddr = kGfx9FlatDisabledSaddr,
                    .wait_load = {wait_all},
                    .wait_store = {wait_all},
                });
  }
  case ROCJITSU_CODE_ARCH_CDNA2: {
    namespace isa = cdna2;
    const uint32_t wait_all = isa::build_sopp(isa::kSWaitcntSopp)[0];
    return emit(payload, arch, seed,
                KernelEncoding{
                    .build_vop2 = isa::build_vop2,
                    .build_vopc = isa::build_vopc,
                    .build_flat = isa::build_flat,
                    .alu_ops = {isa::kVXorB32Vop2, isa::kVOrB32Vop2, isa::kVAndB32Vop2,
                                isa::kVLshlrevB32Vop2, isa::kVLshrrevB32Vop2, isa::kVMulU32U24Vop2,
                                isa::kVMinU32Vop2, isa::kVMaxU32Vop2},
                    .v_and = isa::kVAndB32Vop2,
                    .v_or = isa::kVOrB32Vop2,
                    .cmp_lt = isa::kVCmpxLtU32Vopc,
                    .cmp_eq = isa::kVCmpxEqU32Vopc,
                    .branch_ops = {isa::kSCbranchExeczSopp, isa::kSCbranchExecnzSopp,
                                   isa::kSCbranchScc0Sopp, isa::kSCbranchScc1Sopp},
                    .branch_scc1 = isa::kSCbranchScc1Sopp,
                    .flat_load = isa::kFlatLoadDwordFlat,
                    .flat_store = isa::kFlatStoreDwordFlat,
                    .null_saddr = kGfx9FlatDisabledSaddr,
                    .wait_load = {wait_all},
                    .wait_store = {wait_all},
                });
  }
  case ROCJITSU_CODE_ARCH_CDNA3: {
    namespace isa = cdna3;
    const uint32_t wait_all = isa::build_sopp(isa::kSWaitcntSopp)[0];
    return emit(payload, arch, seed,
                KernelEncoding{
                    .build_vop2 = isa::build_vop2,
                    .build_vopc = isa::build_vopc,
                    .build_flat = isa::build_flat,
                    .alu_ops = {isa::kVXorB32Vop2, isa::kVOrB32Vop2, isa::kVAndB32Vop2,
                                isa::kVLshlrevB32Vop2, isa::kVLshrrevB32Vop2, isa::kVMulU32U24Vop2,
                                isa::kVMinU32Vop2, isa::kVMaxU32Vop2},
                    .v_and = isa::kVAndB32Vop2,
                    .v_or = isa::kVOrB32Vop2,
                    .cmp_lt = isa::kVCmpxLtU32Vopc,
                    .cmp_eq = isa::kVCmpxEqU32Vopc,
                    .branch_ops = {isa::kSCbranchExeczSopp, isa::kSCbranchExecnzSopp,
                                   isa::kSCbranchScc0Sopp, isa::kSCbranchScc1Sopp},
                    .branch_scc1 = isa::kSCbranchScc1Sopp,
                    .flat_load = isa::kFlatLoadDwordFlat,
                    .flat_store = isa::kFlatStoreDwordFlat,
                    .null_saddr = kGfx9FlatDisabledSaddr,
                    .wait_load = {wait_all},
                    .wait_store = {wait_all},
                });
  }
  case ROCJITSU_CODE_ARCH_CDNA4: {
    namespace isa = cdna4;
    const uint32_t wait_all = isa::build_sopp(isa::kSWaitcntSopp)[0];
    return emit(payload, arch, seed,
                KernelEncoding{
                    .build_vop2 = isa::build_vop2,
                    .build_vopc = isa::build_vopc,
                    .build_flat = isa::build_flat,
                    .alu_ops = {isa::kVXorB32Vop2, isa::kVOrB32Vop2, isa::kVAndB32Vop2,
                                isa::kVLshlrevB32Vop2, isa::kVLshrrevB32Vop2, isa::kVMulU32U24Vop2,
                                isa::kVMinU32Vop2, isa::kVMaxU32Vop2},
                    .v_and = isa::kVAndB32Vop2,
                    .v_or = isa::kVOrB32Vop2,
                    .cmp_lt = isa::kVCmpxLtU32Vopc,
                    .cmp_eq = isa::kVCmpxEqU32Vopc,
                    .branch_ops = {isa::kSCbranchExeczSopp, isa::kSCbranchExecnzSopp,
                                   isa::kSCbranchScc0Sopp, isa::kSCbranchScc1Sopp},
                    .branch_scc1 = isa::kSCbranchScc1Sopp,
                    .flat_load = isa::kFlatLoadDwordFlat,
                    .flat_store = isa::kFlatStoreDwordFlat,
                    .null_saddr = kGfx9FlatDisabledSaddr,
                    .wait_load = {wait_all},
                    .wait_store = {wait_all},
                });
  }
  case ROCJITSU_CODE_ARCH_RDNA1: {
    namespace isa = rdna1;
    const uint32_t wait_all = isa::build_sopp(isa::kSWaitcntSopp)[0];
    return emit(payload, arch, seed,
                KernelEncoding{
                    .build_vop2 = isa::build_vop2,
                    .build_vopc = isa::build_vopc,
                    .build_flat = isa::build_flat,
                    .alu_ops = {isa::kVXorB32Vop2, isa::kVOrB32Vop2, isa::kVAndB32Vop2,
                                isa::kVLshlrevB32Vop2, isa::kVLshrrevB32Vop2, isa::kVMulU32U24Vop2,
                                isa::kVMinU32Vop2, isa::kVMaxU32Vop2},
                    .v_and = isa::kVAndB32Vop2,
                    .v_or = isa::kVOrB32Vop2,
                    .cmp_lt = isa::kVCmpxLtU32Vopc,
                    .cmp_eq = isa::kVCmpxEqU32Vopc,
                    .branch_ops = {isa::kSCbranchExeczSopp, isa::kSCbranchExecnzSopp,
                                   isa::kSCbranchScc0Sopp, isa::kSCbranchScc1Sopp},
                    .branch_scc1 = isa::kSCbranchScc1Sopp,
                    .flat_load = isa::kFlatLoadDwordFlat,
                    .flat_store = isa::kFlatStoreDwordFlat,
                    .null_saddr = isa::OPR_SDST_NULL,
                    .wait_load = {wait_all},
                    .wait_store = {isa::build_sopk(isa::kSWaitcntVscntSopk,
                                                   {.sdst = isa::OPR_SDST_NULL})[0],
                                   wait_all},
                });
  }
  case ROCJITSU_CODE_ARCH_RDNA2: {
    namespace isa = rdna2;
    const uint32_t wait_all = isa::build_sopp(isa::kSWaitcntSopp)[0];
    return emit(payload, arch, seed,
                KernelEncoding{
                    .build_vop2 = isa::build_vop2,
                    .build_vopc = isa::build_vopc,
                    .build_flat = isa::build_flat,
                    .alu_ops = {isa::kVXorB32Vop2, isa::kVOrB32Vop2, isa::kVAndB32Vop2,
                                isa::kVLshlrevB32Vop2, isa::kVLshrrevB32Vop2, isa::kVMulU32U24Vop2,
                                isa::kVMinU32Vop2, isa::kVMaxU32Vop2},
                    .v_and = isa::kVAndB32Vop2,
                    .v_or = isa::kVOrB32Vop2,
                    .cmp_lt = isa::kVCmpxLtU32Vopc,
                    .cmp_eq = isa::kVCmpxEqU32Vopc,
                    .branch_ops = {isa::kSCbranchExeczSopp, isa::kSCbranchExecnzSopp,
                                   isa::kSCbranchScc0Sopp, isa::kSCbranchScc1Sopp},
                    .branch_scc1 = isa::kSCbranchScc1Sopp,
                    .flat_load = isa::kFlatLoadDwordFlat,
                    .flat_store = isa::kFlatStoreDwordFlat,
                    .null_saddr = isa::OPR_SDST_NULL,
                    .wait_load = {wait_all},
                    .wait_store = {isa::build_sopk(isa::kSWaitcntVscntSopk,
                                                   {.sdst = isa::OPR_SDST_NULL})[0],
                                   wait_all},
                });
  }
  case ROCJITSU_CODE_ARCH_RDNA3: {
    namespace isa = rdna3;
    const uint32_t wait_all = isa::build_sopp(isa::kSWaitcntSopp)[0];
    return emit(payload, arch, seed,
                KernelEncoding{
                    .build_vop2 = isa::build_vop2,
                    .build_vopc = isa::build_vopc,
                    .build_flat = isa::build_flat,
                    .alu_ops = {isa::kVXorB32Vop2, isa::kVOrB32Vop2, isa::kVAndB32Vop2,
                                isa::kVLshlrevB32Vop2, isa::kVLshrrevB32Vop2, isa::kVMulU32U24Vop2,
                                isa::kVMinU32Vop2, isa::kVMaxU32Vop2},
                    .v_and = isa::kVAndB32Vop2,
                    .v_or = isa::kVOrB32Vop2,
                    .cmp_lt = isa::kVCmpxLtU32Vopc,
                    .cmp_eq = isa::kVCmpxEqU32Vopc,
                    .branch_ops = {isa::kSCbranchExeczSopp, isa::kSCbranchExecnzSopp,
                                   isa::kSCbranchScc0Sopp, isa::kSCbranchScc1Sopp},
                    .branch_scc1 = isa::kSCbranchScc1Sopp,
                    .flat_load = isa::kFlatLoadB32Flat,
                    .flat_store = isa::kFlatStoreB32Flat,
                    .null_saddr = isa::OPR_SDST_NULL,
                    .wait_load = {wait_all},
                    .wait_store = {isa::build_sopk(isa::kSWaitcntVscntSopk,
                                                   {.sdst = isa::OPR_SDST_NULL})[0],
                                   wait_all},
                });
  }
  case ROCJITSU_CODE_ARCH_RDNA3_5: {
    namespace isa = rdna3_5;
    const uint32_t wait_all = isa::build_sopp(isa::kSWaitcntSopp)[0];
    return emit(payload, arch, seed,
                KernelEncoding{
                    .build_vop2 = isa::build_vop2,
                    .build_vopc = isa::build_vopc,
                    .build_flat = isa::build_flat,
                    .alu_ops = {isa::kVXorB32Vop2, isa::kVOrB32Vop2, isa::kVAndB32Vop2,
                                isa::kVLshlrevB32Vop2, isa::kVLshrrevB32Vop2, isa::kVMulU32U24Vop2,
                                isa::kVMinU32Vop2, isa::kVMaxU32Vop2},
                    .v_and = isa::kVAndB32Vop2,
                    .v_or = isa::kVOrB32Vop2,
                    .cmp_lt = isa::kVCmpxLtU32Vopc,
                    .cmp_eq = isa::kVCmpxEqU32Vopc,
                    .branch_ops = {isa::kSCbranchExeczSopp, isa::kSCbranchExecnzSopp,
                                   isa::kSCbranchScc0Sopp, isa::kSCbranchScc1Sopp},
                    .branch_scc1 = isa::kSCbranchScc1Sopp,
                    .flat_load = isa::kFlatLoadB32Flat,
                    .flat_store = isa::kFlatStoreB32Flat,
                    .null_saddr = isa::OPR_SDST_NULL,
                    .wait_load = {wait_all},
                    .wait_store = {isa::build_sopk(isa::kSWaitcntVscntSopk,
                                                   {.sdst = isa::OPR_SDST_NULL})[0],
                                   wait_all},
                });
  }
  case ROCJITSU_CODE_ARCH_RDNA4: {
    namespace isa = rdna4;
    return emit(payload, arch, seed,
                KernelEncoding{
                    .build_vop2 = isa::build_vop2,
                    .build_vopc = isa::build_vopc,
                    .build_flat = isa::build_vflat,
                    .alu_ops = {isa::kVXorB32Vop2, isa::kVOrB32Vop2, isa::kVAndB32Vop2,
                                isa::kVLshlrevB32Vop2, isa::kVLshrrevB32Vop2, isa::kVMulU32U24Vop2,
                                isa::kVMinU32Vop2, isa::kVMaxU32Vop2},
                    .v_and = isa::kVAndB32Vop2,
                    .v_or = isa::kVOrB32Vop2,
                    .cmp_lt = isa::kVCmpxLtU32Vopc,
                    .cmp_eq = isa::kVCmpxEqU32Vopc,
                    .branch_ops = {isa::kSCbranchExeczSopp, isa::kSCbranchExecnzSopp,
                                   isa::kSCbranchScc0Sopp, isa::kSCbranchScc1Sopp},
                    .branch_scc1 = isa::kSCbranchScc1Sopp,
                    .flat_load = isa::kFlatLoadB32Vflat,
                    .flat_store = isa::kFlatStoreB32Vflat,
                    .null_saddr = isa::OPR_SDST_NULL,
                    .wait_load = {isa::build_sopp(isa::kSWaitLoadcntDscntSopp)[0]},
                    .wait_store = {isa::build_sopp(isa::kSWaitStorecntDscntSopp)[0]},
                });
  }
  case ROCJITSU_CODE_ARCH_CDNA5: {
    namespace isa = cdna5;
    return emit(payload, arch, seed,
                KernelEncoding{
                    .build_vop2 = isa::build_vop2,
                    .build_vopc = isa::build_vopc,
                    .build_flat = isa::build_vflat,
                    .alu_ops = {isa::kVXorB32Vop2, isa::kVOrB32Vop2, isa::kVAndB32Vop2,
                                isa::kVLshlrevB32Vop2, isa::kVLshrrevB32Vop2, isa::kVMulU32U24Vop2,
                                isa::kVMinU32Vop2, isa::kVMaxU32Vop2},
                    .v_and = isa::kVAndB32Vop2,
                    .v_or = isa::kVOrB32Vop2,
                    .cmp_lt = isa::kVCmpxLtU32Vopc,
                    .cmp_eq = isa::kVCmpxEqU32Vopc,
                    .branch_ops = {isa::kSCbranchExeczSopp, isa::kSCbranchExecnzSopp,
                                   isa::kSCbranchScc0Sopp, isa::kSCbranchScc1Sopp},
                    .branch_scc1 = isa::kSCbranchScc1Sopp,
                    .flat_load = isa::kFlatLoadB32Vflat,
                    .flat_store = isa::kFlatStoreB32Vflat,
                    .null_saddr = isa::OPR_SDST_NULL,
                    .wait_load = {isa::build_sopp(isa::kSWaitLoadcntDscntSopp)[0]},
                    .wait_store = {isa::build_sopp(isa::kSWaitStorecntDscntSopp)[0]},
                });
  }
  default:
    throw std::logic_error("no generated kernel encoding for target");
  }
}

} // namespace rocjitsu::execution_fuzz
