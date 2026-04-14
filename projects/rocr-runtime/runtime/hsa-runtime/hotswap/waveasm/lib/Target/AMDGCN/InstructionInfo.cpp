// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "waveasm/Target/AMDGCN/InstructionInfo.h"
#include "waveasm/Target/AMDGCN/RegisterInfo.h"

#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

namespace waveasm {

//===----------------------------------------------------------------------===//
// Instruction Registry Implementation
//===----------------------------------------------------------------------===//

InstructionRegistry::InstructionRegistry() {
  // Register common instructions
  // This is a subset - the full registry would be populated from YAML or
  // generated

  // Vector ALU instructions
  registerInstruction(InstrDesc{
      .name = "v_add_u32",
      .category = InstrCategory::VectorALU,
      .defs = {{.name = "vdst", .type = OperandType::VGPR, .isDef = true}},
      .uses = {{.name = "src0", .type = OperandType::VGPR},
               {.name = "src1", .type = OperandType::VGPR}},
      .latency = 1,
  });

  registerInstruction(InstrDesc{
      .name = "v_add_f32",
      .category = InstrCategory::VectorALU,
      .defs = {{.name = "vdst", .type = OperandType::VGPR, .isDef = true}},
      .uses = {{.name = "src0", .type = OperandType::VGPR},
               {.name = "src1", .type = OperandType::VGPR}},
      .latency = 1,
  });

  registerInstruction(InstrDesc{
      .name = "v_mul_f32",
      .category = InstrCategory::VectorALU,
      .defs = {{.name = "vdst", .type = OperandType::VGPR, .isDef = true}},
      .uses = {{.name = "src0", .type = OperandType::VGPR},
               {.name = "src1", .type = OperandType::VGPR}},
      .latency = 1,
  });

  registerInstruction(InstrDesc{
      .name = "v_fma_f32",
      .category = InstrCategory::VectorALU,
      .defs = {{.name = "vdst", .type = OperandType::VGPR, .isDef = true}},
      .uses = {{.name = "src0", .type = OperandType::VGPR},
               {.name = "src1", .type = OperandType::VGPR},
               {.name = "src2", .type = OperandType::VGPR}},
      .latency = 4,
  });

  registerInstruction(InstrDesc{
      .name = "v_mov_b32",
      .category = InstrCategory::VectorALU,
      .defs = {{.name = "vdst", .type = OperandType::VGPR, .isDef = true}},
      .uses = {{.name = "src0", .type = OperandType::VGPR}},
      .latency = 1,
  });

  // Scalar ALU instructions
  registerInstruction(InstrDesc{
      .name = "s_add_u32",
      .category = InstrCategory::ScalarALU,
      .defs = {{.name = "sdst", .type = OperandType::SGPR, .isDef = true}},
      .uses = {{.name = "src0", .type = OperandType::SGPR},
               {.name = "src1", .type = OperandType::SGPR}},
      .latency = 1,
  });

  registerInstruction(InstrDesc{
      .name = "s_mov_b32",
      .category = InstrCategory::ScalarALU,
      .defs = {{.name = "sdst", .type = OperandType::SGPR, .isDef = true}},
      .uses = {{.name = "src0", .type = OperandType::SGPR}},
      .latency = 1,
  });

  registerInstruction(InstrDesc{
      .name = "s_mov_b64",
      .category = InstrCategory::ScalarALU,
      .defs = {{.name = "sdst",
                .type = OperandType::SGPR_Pair,
                .size = 2,
                .isDef = true}},
      .uses = {{.name = "src0", .type = OperandType::SGPR_Pair, .size = 2}},
      .latency = 1,
  });

  registerInstruction(InstrDesc{
      .name = "s_cmp_eq_u32",
      .category = InstrCategory::ScalarALU,
      .defs = {{.name = "scc", .type = OperandType::SCC, .isDef = true}},
      .uses = {{.name = "src0", .type = OperandType::SGPR},
               {.name = "src1", .type = OperandType::SGPR}},
      .latency = 1,
  });

  // Vector memory instructions
  registerInstruction(InstrDesc{
      .name = "global_load_dword",
      .category = InstrCategory::VectorMem,
      .defs = {{.name = "vdst", .type = OperandType::VGPR, .isDef = true}},
      .uses = {{.name = "vaddr", .type = OperandType::VGPR_Pair, .size = 2}},
      .latency = 100,
      .mayLoad = true,
      .incrementsVmcnt = true,
  });

  registerInstruction(InstrDesc{
      .name = "global_load_dwordx2",
      .category = InstrCategory::VectorMem,
      .defs = {{.name = "vdst",
                .type = OperandType::VGPR_Pair,
                .size = 2,
                .isDef = true}},
      .uses = {{.name = "vaddr", .type = OperandType::VGPR_Pair, .size = 2}},
      .latency = 100,
      .mayLoad = true,
      .incrementsVmcnt = true,
  });

  registerInstruction(InstrDesc{
      .name = "global_load_dwordx4",
      .category = InstrCategory::VectorMem,
      .defs = {{.name = "vdst",
                .type = OperandType::VGPR_Quad,
                .size = 4,
                .isDef = true}},
      .uses = {{.name = "vaddr", .type = OperandType::VGPR_Pair, .size = 2}},
      .latency = 100,
      .mayLoad = true,
      .incrementsVmcnt = true,
  });

  registerInstruction(InstrDesc{
      .name = "global_store_dword",
      .category = InstrCategory::VectorMem,
      .defs = {},
      .uses = {{.name = "vaddr", .type = OperandType::VGPR_Pair, .size = 2},
               {.name = "vdata", .type = OperandType::VGPR}},
      .latency = 100,
      .mayStore = true,
      .incrementsVmcnt = true,
  });

  // LDS instructions
  registerInstruction(InstrDesc{
      .name = "ds_read_b32",
      .category = InstrCategory::LDS,
      .defs = {{.name = "vdst", .type = OperandType::VGPR, .isDef = true}},
      .uses = {{.name = "vaddr", .type = OperandType::VGPR}},
      .latency = 20,
      .mayLoad = true,
      .incrementsLgkmcnt = true,
  });

  registerInstruction(InstrDesc{
      .name = "ds_read_b64",
      .category = InstrCategory::LDS,
      .defs = {{.name = "vdst",
                .type = OperandType::VGPR_Pair,
                .size = 2,
                .isDef = true}},
      .uses = {{.name = "vaddr", .type = OperandType::VGPR}},
      .latency = 20,
      .mayLoad = true,
      .incrementsLgkmcnt = true,
  });

  registerInstruction(InstrDesc{
      .name = "ds_write_b32",
      .category = InstrCategory::LDS,
      .defs = {},
      .uses = {{.name = "vaddr", .type = OperandType::VGPR},
               {.name = "vdata", .type = OperandType::VGPR}},
      .latency = 20,
      .mayStore = true,
      .incrementsLgkmcnt = true,
  });

  registerInstruction(InstrDesc{
      .name = "ds_write_b8",
      .category = InstrCategory::LDS,
      .defs = {},
      .uses = {{.name = "vaddr", .type = OperandType::VGPR},
               {.name = "vdata", .type = OperandType::VGPR}},
      .latency = 20,
      .mayStore = true,
      .incrementsLgkmcnt = true,
  });

  registerInstruction(InstrDesc{
      .name = "ds_write_b16",
      .category = InstrCategory::LDS,
      .defs = {},
      .uses = {{.name = "vaddr", .type = OperandType::VGPR},
               {.name = "vdata", .type = OperandType::VGPR}},
      .latency = 20,
      .mayStore = true,
      .incrementsLgkmcnt = true,
  });

  registerInstruction(InstrDesc{
      .name = "ds_read_u8",
      .category = InstrCategory::LDS,
      .defs = {{.name = "vdst", .type = OperandType::VGPR, .isDef = true}},
      .uses = {{.name = "vaddr", .type = OperandType::VGPR}},
      .latency = 20,
      .mayLoad = true,
      .incrementsLgkmcnt = true,
  });

  registerInstruction(InstrDesc{
      .name = "ds_read_u16",
      .category = InstrCategory::LDS,
      .defs = {{.name = "vdst", .type = OperandType::VGPR, .isDef = true}},
      .uses = {{.name = "vaddr", .type = OperandType::VGPR}},
      .latency = 20,
      .mayLoad = true,
      .incrementsLgkmcnt = true,
  });

  // MFMA instructions
  registerInstruction(InstrDesc{
      .name = "v_mfma_f32_32x32x8_f16",
      .category = InstrCategory::MFMA,
      .defs = {{.name = "vdst",
                .type = OperandType::AGPR_Range,
                .size = 16,
                .alignment = 4,
                .isDef = true}},
      .uses = {{.name = "src0", .type = OperandType::VGPR_Range, .size = 4},
               {.name = "src1", .type = OperandType::VGPR_Range, .size = 4},
               {.name = "src2", .type = OperandType::AGPR_Range, .size = 16}},
      .latency = 64,
  });

  registerInstruction(InstrDesc{
      .name = "v_mfma_f32_16x16x16_f16",
      .category = InstrCategory::MFMA,
      .defs = {{.name = "vdst",
                .type = OperandType::AGPR_Range,
                .size = 4,
                .alignment = 4,
                .isDef = true}},
      .uses = {{.name = "src0", .type = OperandType::VGPR_Range, .size = 4},
               {.name = "src1", .type = OperandType::VGPR_Range, .size = 4},
               {.name = "src2", .type = OperandType::AGPR_Range, .size = 4}},
      .latency = 32,
  });

  // v_mfma_f32_16x16x32_f16 - gfx950+ (MI350), uses 8xf16 inputs (4 VGPRs each)
  registerInstruction(InstrDesc{
      .name = "v_mfma_f32_16x16x32_f16",
      .category = InstrCategory::MFMA,
      .defs = {{.name = "vdst",
                .type = OperandType::AGPR_Range,
                .size = 4,
                .alignment = 4,
                .isDef = true}},
      .uses = {{.name = "src0",
                .type = OperandType::VGPR_Range,
                .size = 8}, // 8xf16 = 4 VGPRs
               {.name = "src1",
                .type = OperandType::VGPR_Range,
                .size = 8}, // 8xf16 = 4 VGPRs
               {.name = "src2", .type = OperandType::AGPR_Range, .size = 4}},
      .latency = 32,
  });

  // v_mfma_f32_16x16x32_bf16 - gfx950+ (MI350), uses 8xbf16 inputs (4 VGPRs
  // each)
  registerInstruction(InstrDesc{
      .name = "v_mfma_f32_16x16x32_bf16",
      .category = InstrCategory::MFMA,
      .defs = {{.name = "vdst",
                .type = OperandType::AGPR_Range,
                .size = 4,
                .alignment = 4,
                .isDef = true}},
      .uses = {{.name = "src0",
                .type = OperandType::VGPR_Range,
                .size = 8}, // 8xbf16 = 4 VGPRs
               {.name = "src1",
                .type = OperandType::VGPR_Range,
                .size = 8}, // 8xbf16 = 4 VGPRs
               {.name = "src2", .type = OperandType::AGPR_Range, .size = 4}},
      .latency = 32,
  });

  // Branch instructions
  registerInstruction(InstrDesc{
      .name = "s_branch",
      .category = InstrCategory::Branch,
      .defs = {},
      .uses = {{.name = "target", .type = OperandType::Label}},
      .isBranch = true,
      .isTerminator = true,
  });

  registerInstruction(InstrDesc{
      .name = "s_cbranch_scc0",
      .category = InstrCategory::Branch,
      .defs = {},
      .uses = {{.name = "target", .type = OperandType::Label}},
      .isBranch = true,
      .isConditionalBranch = true,
      .isTerminator = true,
  });

  registerInstruction(InstrDesc{
      .name = "s_cbranch_scc1",
      .category = InstrCategory::Branch,
      .defs = {},
      .uses = {{.name = "target", .type = OperandType::Label}},
      .isBranch = true,
      .isConditionalBranch = true,
      .isTerminator = true,
  });

  registerInstruction(InstrDesc{
      .name = "s_cbranch_vccz",
      .category = InstrCategory::Branch,
      .defs = {},
      .uses = {{.name = "target", .type = OperandType::Label}},
      .isBranch = true,
      .isConditionalBranch = true,
      .isTerminator = true,
  });

  registerInstruction(InstrDesc{
      .name = "s_cbranch_vccnz",
      .category = InstrCategory::Branch,
      .defs = {},
      .uses = {{.name = "target", .type = OperandType::Label}},
      .isBranch = true,
      .isConditionalBranch = true,
      .isTerminator = true,
  });

  // Miscellaneous
  registerInstruction(InstrDesc{
      .name = "s_waitcnt",
      .category = InstrCategory::Misc,
      .defs = {},
      .uses = {},
      .hasSideEffects = true,
      .isBarrier = true,
  });

  registerInstruction(InstrDesc{
      .name = "s_barrier",
      .category = InstrCategory::Misc,
      .defs = {},
      .uses = {},
      .hasSideEffects = true,
      .isBarrier = true,
  });

  registerInstruction(InstrDesc{
      .name = "s_nop",
      .category = InstrCategory::Misc,
      .defs = {},
      .uses = {{.name = "simm16", .type = OperandType::Immediate}},
  });

  registerInstruction(InstrDesc{
      .name = "s_endpgm",
      .category = InstrCategory::Misc,
      .defs = {},
      .uses = {},
      .hasSideEffects = true,
      .isTerminator = true,
  });
}

InstructionRegistry::~InstructionRegistry() = default;

const InstrDesc *InstructionRegistry::getInstrDesc(llvm::StringRef name) const {
  auto it = instructions.find(name);
  if (it != instructions.end())
    return &it->second;
  return nullptr;
}

bool InstructionRegistry::hasInstruction(llvm::StringRef name) const {
  return instructions.contains(name);
}

llvm::SmallVector<llvm::StringRef>
InstructionRegistry::getInstructionNames() const {
  llvm::SmallVector<llvm::StringRef> names;
  for (const auto &entry : instructions) {
    names.push_back(entry.first());
  }
  return names;
}

llvm::SmallVector<llvm::StringRef>
InstructionRegistry::getInstructionsByCategory(InstrCategory category) const {
  llvm::SmallVector<llvm::StringRef> names;
  for (const auto &entry : instructions) {
    if (entry.second.category == category) {
      names.push_back(entry.first());
    }
  }
  return names;
}

void InstructionRegistry::registerInstruction(const InstrDesc &desc) {
  instructions[desc.name] = desc;
}

InstructionRegistry &InstructionRegistry::get() {
  static InstructionRegistry instance;
  return instance;
}

//===----------------------------------------------------------------------===//
// Format Helpers
//===----------------------------------------------------------------------===//

std::string formatRegister(OperandType type, int64_t baseIndex, int64_t size) {
  switch (type) {
  case OperandType::VGPR:
  case OperandType::VGPR_Pair:
  case OperandType::VGPR_Quad:
  case OperandType::VGPR_Range:
    return formatVGPRRange(baseIndex, size);

  case OperandType::SGPR:
  case OperandType::SGPR_Pair:
  case OperandType::SGPR_Quad:
  case OperandType::SGPR_Range:
    return formatSGPRRange(baseIndex, size);

  case OperandType::AGPR:
  case OperandType::AGPR_Range:
    return formatAGPRRange(baseIndex, size);

  case OperandType::VCC:
    return "vcc";
  case OperandType::EXEC:
    return "exec";
  case OperandType::SCC:
    return "scc";
  case OperandType::M0:
    return "m0";

  default:
    return "<unknown>";
  }
}

std::string formatImmediate(int64_t value) {
  if (value >= 0 && value <= 64) {
    // Inline constant
    return std::to_string(value);
  }
  if (value >= -16 && value < 0) {
    // Negative inline constant
    return std::to_string(value);
  }
  // Literal constant
  std::string hex;
  llvm::raw_string_ostream os(hex);
  os << llvm::format_hex(static_cast<uint64_t>(value), 0);
  return os.str();
}

std::string formatLabel(llvm::StringRef label) { return label.str(); }

std::optional<std::pair<OperandType, int64_t>>
parseRegister(llvm::StringRef regStr) {
  // Parse simple register like "v5", "s10"
  if (regStr.size() >= 2) {
    char prefix = regStr[0];
    llvm::StringRef indexStr = regStr.substr(1);
    int64_t index;
    if (!indexStr.getAsInteger(10, index)) {
      if (prefix == 'v')
        return std::make_pair(OperandType::VGPR, index);
      if (prefix == 's')
        return std::make_pair(OperandType::SGPR, index);
      if (prefix == 'a')
        return std::make_pair(OperandType::AGPR, index);
    }
  }

  // Check for special registers
  if (regStr == "vcc")
    return std::make_pair(OperandType::VCC, 0);
  if (regStr == "exec")
    return std::make_pair(OperandType::EXEC, 0);
  if (regStr == "scc")
    return std::make_pair(OperandType::SCC, 0);
  if (regStr == "m0")
    return std::make_pair(OperandType::M0, 0);

  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// RegisterInfo Helpers
//===----------------------------------------------------------------------===//

std::optional<std::tuple<bool, int64_t, int64_t>>
parseRegisterString(llvm::StringRef regStr) {
  // Parse "v5" -> (true, 5, 1)
  // Parse "v[0:3]" -> (true, 0, 4)
  // Parse "s[0:1]" -> (false, 0, 2)

  if (regStr.empty())
    return std::nullopt;

  bool isVGPR = regStr.starts_with("v");
  bool isSGPR = regStr.starts_with("s");

  if (!isVGPR && !isSGPR)
    return std::nullopt;

  regStr = regStr.substr(1); // Remove prefix

  // Check for range syntax
  if (regStr.starts_with("[")) {
    // Parse "[N:M]"
    regStr = regStr.substr(1); // Remove "["
    size_t colonPos = regStr.find(':');
    if (colonPos == llvm::StringRef::npos)
      return std::nullopt;

    llvm::StringRef startStr = regStr.substr(0, colonPos);
    llvm::StringRef endStr = regStr.substr(colonPos + 1);

    // Remove trailing "]"
    if (endStr.ends_with("]"))
      endStr = endStr.drop_back(1);

    int64_t startIdx, endIdx;
    if (startStr.getAsInteger(10, startIdx) || endStr.getAsInteger(10, endIdx))
      return std::nullopt;

    int64_t size = endIdx - startIdx + 1;
    return std::make_tuple(isVGPR, startIdx, size);
  }

  // Single register
  int64_t index;
  if (regStr.getAsInteger(10, index))
    return std::nullopt;

  return std::make_tuple(isVGPR, index, 1);
}

bool isSpecialRegister(llvm::StringRef regStr) {
  return regStr == "vcc" || regStr == "vcc_lo" || regStr == "vcc_hi" ||
         regStr == "exec" || regStr == "exec_lo" || regStr == "exec_hi" ||
         regStr == "scc" || regStr == "m0" || regStr == "flat_scratch" ||
         regStr == "flat_scratch_lo" || regStr == "flat_scratch_hi";
}

} // namespace waveasm
