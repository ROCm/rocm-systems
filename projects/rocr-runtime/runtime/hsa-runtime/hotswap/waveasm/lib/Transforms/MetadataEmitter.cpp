// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
// Instruction Formatter and Metadata Emitter
//
// InstructionFormatter: formats individual AMDGCN assembly instructions.
// MetadataEmitter: emits kernel prologue (.text, symbol), epilogue (kernel
//   descriptor, YAML metadata) for AMDGCN assembly output.
//===----------------------------------------------------------------------===//

#include "waveasm/Dialect/WaveASMOps.h"
#include "waveasm/Dialect/WaveASMTypes.h"
#include "waveasm/Transforms/AssemblyEmitter.h"

using namespace mlir;

namespace waveasm {

//===----------------------------------------------------------------------===//
// Instruction Formatter Implementation
//===----------------------------------------------------------------------===//

std::string InstructionFormatter::format(llvm::StringRef name,
                                         llvm::ArrayRef<std::string> operands) {
  std::string result = "  " + name.str();

  if (!operands.empty()) {
    result += " ";
    for (size_t i = 0; i < operands.size(); ++i) {
      if (i > 0)
        result += ", ";
      result += operands[i];
    }
  }

  return result;
}

std::string InstructionFormatter::formatLabel(llvm::StringRef label) {
  return label.str() + ":";
}

std::string InstructionFormatter::formatComment(llvm::StringRef text) {
  return "  ; " + text.str();
}

std::string InstructionFormatter::formatWaitcnt(std::optional<int64_t> vmcnt,
                                                std::optional<int64_t> lgkmcnt,
                                                std::optional<int64_t> expcnt) {
  std::string result = "  s_waitcnt";
  llvm::SmallVector<std::string> counts;

  if (vmcnt.has_value()) {
    counts.push_back("vmcnt(" + std::to_string(*vmcnt) + ")");
  }
  if (lgkmcnt.has_value()) {
    counts.push_back("lgkmcnt(" + std::to_string(*lgkmcnt) + ")");
  }
  if (expcnt.has_value()) {
    counts.push_back("expcnt(" + std::to_string(*expcnt) + ")");
  }

  if (!counts.empty()) {
    result += " ";
    for (size_t i = 0; i < counts.size(); ++i) {
      if (i > 0)
        result += " & ";
      result += counts[i];
    }
  }

  return result;
}

std::string InstructionFormatter::formatBarrier() { return "  s_barrier"; }

std::string InstructionFormatter::formatEndpgm() { return "  s_endpgm"; }

std::string InstructionFormatter::formatRaw(llvm::StringRef text) {
  return "  " + text.str();
}

//===----------------------------------------------------------------------===//
// Metadata Emitter Implementation
//===----------------------------------------------------------------------===//

MetadataEmitter::MetadataEmitter(ProgramOp program, TargetAttrInterface target)
    : program(program), target(target) {}

llvm::SmallVector<std::string> MetadataEmitter::emitPrologue() {
  llvm::SmallVector<std::string> lines;

  lines.push_back(target.getTargetDirective().str());
  lines.push_back("");
  lines.push_back(".text");
  lines.push_back("");

  std::string symName = getKernelName(program).str();
  lines.push_back(".protected " + symName);
  lines.push_back(".globl " + symName);
  lines.push_back(".p2align 8");
  lines.push_back(".type " + symName + ",@function");
  lines.push_back(symName + ":");

  return lines;
}

llvm::SmallVector<std::string> MetadataEmitter::emitEpilogue(int64_t peakVGPRs,
                                                             int64_t peakSGPRs,
                                                             int64_t peakAGPRs,
                                                             int64_t ldsSize) {
  llvm::SmallVector<std::string> lines;

  auto descriptor =
      emitKernelDescriptor(peakVGPRs, peakSGPRs, peakAGPRs, ldsSize);
  lines.append(descriptor.begin(), descriptor.end());
  lines.push_back("");

  auto metadata = emitMetadataYAML(peakVGPRs, peakSGPRs, peakAGPRs, ldsSize);
  lines.append(metadata.begin(), metadata.end());

  return lines;
}

/// Scan program operations to detect system register usage
static void scanSystemRegisterUsage(ProgramOp program, bool &usesWorkgroupIdX,
                                    bool &usesWorkgroupIdY,
                                    bool &usesWorkgroupIdZ,
                                    bool &usesWorkitemId) {
  usesWorkgroupIdX = false;
  usesWorkgroupIdY = false;
  usesWorkgroupIdZ = false;
  usesWorkitemId = false;

  auto targetAttr = program.getTarget();
  auto targetKind = targetAttr.getTargetKind();
  bool isGfx950 = llvm::isa<GFX950TargetAttr>(targetKind);

  int64_t numArgs = 2;
  if (auto numArgsAttr =
          program->getAttrOfType<IntegerAttr>("num_kernel_args")) {
    numArgs = numArgsAttr.getInt();
  }

  int64_t userSgprCount = 2;
  if (isGfx950) {
    // Hardware limits user SGPRs to 16 on gfx950.
    userSgprCount = std::min(int64_t(16), 2 + numArgs * 2);
  }

  int64_t wgIdXIndex = userSgprCount;
  int64_t wgIdYIndex = userSgprCount + 1;
  int64_t wgIdZIndex = userSgprCount + 2;

  program.walk([&](Operation *op) {
    if (auto precolored = dyn_cast<PrecoloredSRegOp>(op)) {
      int64_t idx = precolored.getIndex();
      int64_t size = precolored.getSize();
      if (size == 1) {
        if (idx == wgIdXIndex)
          usesWorkgroupIdX = true;
        else if (idx == wgIdYIndex)
          usesWorkgroupIdY = true;
        else if (idx == wgIdZIndex)
          usesWorkgroupIdZ = true;
      }
    }
  });
}

llvm::SmallVector<std::string>
MetadataEmitter::emitKernelDescriptor(int64_t peakVGPRs, int64_t peakSGPRs,
                                      int64_t peakAGPRs, int64_t ldsSize) {
  llvm::SmallVector<std::string> lines;
  std::string symName = getKernelName(program).str();

  bool usesWorkgroupIdX, usesWorkgroupIdY, usesWorkgroupIdZ, usesWorkitemId;
  scanSystemRegisterUsage(program, usesWorkgroupIdX, usesWorkgroupIdY,
                          usesWorkgroupIdZ, usesWorkitemId);

  lines.push_back("");
  lines.push_back(".section .rodata,#alloc");
  lines.push_back(".p2align 6");
  lines.push_back(".amdhsa_kernel " + symName);

  int64_t ldsBlocks = (ldsSize + 127) / 128;
  lines.push_back("  .amdhsa_group_segment_fixed_size " +
                  std::to_string(ldsBlocks * 128));
  lines.push_back("  .amdhsa_private_segment_fixed_size 0");

  // Kernarg size must account for all args the runtime packs: buffer ptrs +
  // dynamic dims + stride args. Use 2x base to safely accommodate strides.
  {
    int64_t numArgsForSize = 2;
    if (auto numArgsAttr =
            program->getAttrOfType<IntegerAttr>("num_kernel_args")) {
      numArgsForSize = numArgsAttr.getInt();
    }
    int64_t kernargSize = numArgsForSize * 8 * 2;
    lines.push_back("  .amdhsa_kernarg_size " + std::to_string(kernargSize));
  }

  auto targetAttr = program.getTarget();
  auto targetKind = targetAttr.getTargetKind();
  int64_t preloadLength = program.getKernargPreloadLength();
  bool usePreloading = llvm::isa<GFX950TargetAttr>(targetKind);

  if (usePreloading && preloadLength == 0) {
    int64_t numArgs = 2;
    if (auto numArgsAttr =
            program->getAttrOfType<IntegerAttr>("num_kernel_args")) {
      numArgs = numArgsAttr.getInt();
    }
    // Hardware limits user SGPRs to 16 on gfx950 (2 for kernarg ptr + 14 max
    // preloaded). Overflow args are loaded via explicit s_load in the prologue.
    preloadLength = std::min(int64_t(14), numArgs * 2);
  }

  int64_t userSgprCount = 2;
  if (usePreloading && preloadLength > 0) {
    userSgprCount = 2 + preloadLength;
  }

  lines.push_back("  .amdhsa_user_sgpr_count " + std::to_string(userSgprCount));
  lines.push_back("  .amdhsa_user_sgpr_dispatch_ptr 0");
  lines.push_back("  .amdhsa_user_sgpr_queue_ptr 0");
  lines.push_back("  .amdhsa_user_sgpr_kernarg_segment_ptr 1");
  lines.push_back("  .amdhsa_user_sgpr_dispatch_id 0");

  if (usePreloading && preloadLength > 0) {
    lines.push_back("  .amdhsa_user_sgpr_kernarg_preload_length " +
                    std::to_string(preloadLength));
    lines.push_back("  .amdhsa_user_sgpr_kernarg_preload_offset 0");
  }

  lines.push_back("  .amdhsa_user_sgpr_private_segment_size 0");
  lines.push_back("  .amdhsa_uses_dynamic_stack 0");
  lines.push_back("  .amdhsa_enable_private_segment 0");

  int64_t vgprGranularity = 4;
  if (llvm::isa<GFX942TargetAttr, GFX950TargetAttr>(targetKind)) {
    vgprGranularity = 8;
  }
  int64_t nextFreeVGPR =
      ((peakVGPRs + vgprGranularity - 1) / vgprGranularity) * vgprGranularity;
  int64_t nextFreeAGPR =
      ((peakAGPRs + vgprGranularity - 1) / vgprGranularity) * vgprGranularity;

  int64_t sgprGranularity = 8;
  int64_t nextFreeSGPR =
      ((peakSGPRs + sgprGranularity - 1) / sgprGranularity) * sgprGranularity;
  if (llvm::isa<GFX942TargetAttr, GFX950TargetAttr>(targetKind)) {
    nextFreeSGPR = std::min(nextFreeSGPR, int64_t(102));
  }

  if (llvm::isa<GFX942TargetAttr, GFX950TargetAttr>(targetKind)) {
    int64_t accumOffset = std::max(int64_t(4), ((nextFreeVGPR + 3) / 4) * 4);
    accumOffset = std::min(accumOffset, int64_t(256));
    lines.push_back("  .amdhsa_accum_offset " + std::to_string(accumOffset));
    if (nextFreeAGPR > 0) {
      // On CDNA targets AGPRs share the unified file after accum_offset.
      // next_free_vgpr encodes unified usage when AGPRs are present.
      nextFreeVGPR = accumOffset + nextFreeAGPR;
    }
  }

  lines.push_back("  .amdhsa_next_free_vgpr " + std::to_string(nextFreeVGPR));
  lines.push_back("  .amdhsa_next_free_sgpr " + std::to_string(nextFreeSGPR));

  // Always enable all workgroup IDs when any is used.
  // This matches the real LLVM backend and ensures the SGPR layout
  // is predictable (base+0=x, base+1=y, base+2=z) without gaps.
  bool anyWgId = usesWorkgroupIdX || usesWorkgroupIdY || usesWorkgroupIdZ;
  lines.push_back("  .amdhsa_system_sgpr_workgroup_id_x " +
                  std::to_string(anyWgId ? 1 : 0));
  lines.push_back("  .amdhsa_system_sgpr_workgroup_id_y " +
                  std::to_string(anyWgId ? 1 : 0));
  lines.push_back("  .amdhsa_system_sgpr_workgroup_id_z " +
                  std::to_string(anyWgId ? 1 : 0));

  // Derive system_vgpr_workitem_id from workgroup dimensions.
  // When wgZ > 1, hardware provides thread IDs in v0 (x), v1 (y), v2 (z).
  // When wgY > 1 (and wgZ == 1), hardware provides v0 (x), v1 (y).
  int64_t wgY = 1, wgZ = 1;
  auto wgAttr = program.getWorkgroupSizeAttr();
  if (wgAttr) {
    if (wgAttr.size() >= 2)
      if (auto intAttr = dyn_cast<IntegerAttr>(wgAttr[1]))
        wgY = intAttr.getInt();
    if (wgAttr.size() >= 3)
      if (auto intAttr = dyn_cast<IntegerAttr>(wgAttr[2]))
        wgZ = intAttr.getInt();
  }
  int64_t systemVgprWorkitemId = 0;
  if (wgZ > 1) {
    systemVgprWorkitemId = 2;
  } else if (wgY > 1) {
    systemVgprWorkitemId = 1;
  }
  lines.push_back("  .amdhsa_system_vgpr_workitem_id " +
                  std::to_string(systemVgprWorkitemId));

  lines.push_back("  .amdhsa_float_denorm_mode_32 3");
  lines.push_back("  .amdhsa_float_denorm_mode_16_64 3");

  lines.push_back(".end_amdhsa_kernel");

  return lines;
}

llvm::SmallVector<std::string>
MetadataEmitter::emitMetadataYAML(int64_t peakVGPRs, int64_t peakSGPRs,
                                  int64_t peakAGPRs, int64_t ldsSize) {
  llvm::SmallVector<std::string> lines;
  std::string symName = getKernelName(program).str();

  lines.push_back(".amdgpu_metadata");
  lines.push_back("---");
  lines.push_back("amdhsa.version:");
  lines.push_back("  - 1");
  lines.push_back("  - 2");
  lines.push_back("amdhsa.kernels:");
  lines.push_back("  - .name: " + symName);
  lines.push_back("    .symbol: " + symName + ".kd");

  int64_t numArgs = 2;
  if (auto numArgsAttr =
          program->getAttrOfType<IntegerAttr>("num_kernel_args")) {
    numArgs = numArgsAttr.getInt();
  } else {
    int64_t kernargPreload = program.getKernargPreloadLength();
    if (kernargPreload > 0) {
      numArgs = kernargPreload / 2;
    }
  }
  // The runtime packs: buffer_ptrs (8B each) + dynamic_dims (8B each) +
  // stride_args (8B each). Strides are appended by the runtime for each 2D+
  // tensor binding. Use 2x the base size to safely accommodate strides.
  int64_t kernargSize = numArgs * 8 * 2;
  lines.push_back("    .args:");
  for (int64_t i = 0; i < numArgs; ++i) {
    lines.push_back("      - .name:       arg" + std::to_string(i) + "_ptr");
    lines.push_back("        .offset:     " + std::to_string(i * 8));
    lines.push_back("        .size:       8");
    lines.push_back("        .value_kind: global_buffer");
    lines.push_back("        .value_type: 'i8*'");
  }

  lines.push_back("    .kernarg_segment_size: " + std::to_string(kernargSize));
  lines.push_back("    .group_segment_fixed_size: " + std::to_string(ldsSize));
  lines.push_back("    .private_segment_fixed_size: 0");
  lines.push_back("    .kernarg_segment_align: 8");

  int64_t waveSizeVal = target.getDefaultWaveSize();
  lines.push_back("    .wavefront_size: " + std::to_string(waveSizeVal));
  lines.push_back("    .sgpr_count: " + std::to_string(peakSGPRs));
  lines.push_back("    .vgpr_count: " + std::to_string(peakVGPRs));
  if (peakAGPRs > 0)
    lines.push_back("    .agpr_count: " + std::to_string(peakAGPRs));

  auto workgroupSize = program.getWorkgroupSize();
  int64_t maxFlatSize = 256;
  if (workgroupSize.has_value()) {
    maxFlatSize = 1;
    for (Attribute attr : *workgroupSize) {
      if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
        maxFlatSize *= intAttr.getInt();
      }
    }
  }
  lines.push_back("    .max_flat_workgroup_size: " +
                  std::to_string(maxFlatSize));

  lines.push_back("...");
  lines.push_back(".end_amdgpu_metadata");

  return lines;
}

} // namespace waveasm
