// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef WaveASM_DIALECT_WAVEASMATTRS
#define WaveASM_DIALECT_WAVEASMATTRS

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "waveasm/Dialect/WaveASMDialect.td"
include "waveasm/Dialect/WaveASMInterfaces.td"

// Base class for WAVEASM attributes
class WAVEASMAttr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<WaveASMDialect, name, traits> {
  let mnemonic = attrMnemonic;
}

//===----------------------------------------------------------------------===//
// Kernel ABI Attribute
//===----------------------------------------------------------------------===//

def WaveASM_KernelABIAttr : WAVEASMAttr<"KernelABI", "abi"> {
  let summary = "ABI bindings for precolored registers";
  let description = [{
    Defines the GPU ABI for kernel entry. These registers are precolored
    and reserved from allocation.

    Parameters:
    - tid: VGPR index for thread ID (typically 0)
    - kernarg: SGPR index for kernarg pointer low bits (typically 0, pair uses 0:1)
    - wg_id_x, wg_id_y, wg_id_z: Optional SGPR indices for workgroup IDs

    Example:
    ```mlir
    waveasm.program @kernel
      abi = #waveasm.abi<tid = 0, kernarg = 0, wg_id_x = 2>
    ```
  }];

  let parameters = (ins
    DefaultValuedParameter<"int64_t", "0">:$tid,
    DefaultValuedParameter<"int64_t", "0">:$kernarg,
    OptionalParameter<"std::optional<int64_t>">:$wg_id_x,
    OptionalParameter<"std::optional<int64_t>">:$wg_id_y,
    OptionalParameter<"std::optional<int64_t>">:$wg_id_z
  );

  let assemblyFormat = "`<` struct(params) `>`";

  let extraClassDeclaration = [{
    // Get reserved VGPR indices
    ::llvm::SmallVector<int64_t> getReservedVGPRs() const {
      return {getTid()};
    }

    // Get reserved SGPR indices
    ::llvm::SmallVector<int64_t> getReservedSGPRs() const {
      ::llvm::SmallVector<int64_t> reserved;
      // Kernarg pointer is a pair (low, high)
      reserved.push_back(getKernarg());
      reserved.push_back(getKernarg() + 1);
      if (auto x = getWgIdX()) reserved.push_back(*x);
      if (auto y = getWgIdY()) reserved.push_back(*y);
      if (auto z = getWgIdZ()) reserved.push_back(*z);
      return reserved;
    }
  }];
}

//===----------------------------------------------------------------------===//
// Target Attribute
//===----------------------------------------------------------------------===//

// TODO: consider extending this to be a proper target description instead of
// inlined functions below. This will require a bit of tablegen backend.
def WaveASM_TargetKind
    : I32EnumAttr<"TargetKind", "Supported GPU targets",
                  [I32EnumAttrCase<"GFX942", 0, "gfx942">,
                   I32EnumAttrCase<"GFX950", 1, "gfx950">,
                   I32EnumAttrCase<"GFX1250", 2, "gfx1250">]> {
  let summary = "Supported GPU targets";
  let description = [{
    - gfx942: AMD CDNA3 (MI300 series)
    - gfx950: AMD CDNA3+ (future MI series)
    - gfx1250: AMD RDNA4
  }];
}

class WaveASM_TargetKindAttr<string className, string mnemonic,
                             list<Trait> traits = []>
    : WAVEASMAttr<className, mnemonic,
                  !listconcat([WaveASM_TargetAttrInterface], traits)> {

  // TODO: make into enum.
  string archGeneration = ?;
  // TODO: make into enum.
  string computeArch = ?;
  int maxVGPRs = ?;
  int maxSGPRs = ?;
  int maxAGPRs = ?;
  int defaultWaveSize = ?;
  list<int> supportedWaveSizes = ?;
  int maxLDSSize = ?;
  int LDSBankCount = ?;
  int LDSBankWidth = ?;
  int globalLoadLatency = ?;
  int LDSLoadLatency = ?;
  int maxVmcnt = ?;
  int maxLgkmcnt = ?;
  int maxExpcnt = ?;
  int defaultCodeObjectVersion = ?;
  list<int> supportedCodeObjectVersions = ?;
  string targetDirective = ?;
  string ABIVersion = ?;

  string commonClassDeclaration = [{
    ::llvm::StringRef getArchGeneration() const {
      return "}]#archGeneration#[{";
    }

    ::llvm::StringRef getComputeArch() const {
      return "}]#computeArch#[{";
    }

    int64_t getMaxVGPRs() const {
      return }]#maxVGPRs#[{;
    }

    int64_t getMaxSGPRs() const {
      return }]#maxSGPRs#[{;
    }

    int64_t getMaxAGPRs() const {
      return }]#maxAGPRs#[{;
    }

    int64_t getDefaultWaveSize() const {
      return }]#defaultWaveSize#[{;
    }

    ::llvm::SmallVector<int64_t> getSupportedWaveSizes() const {
      return {}]#!interleave(supportedWaveSizes, ", ")#[{};
    }

    int64_t getMaxLDSSize() const {
      return }]#maxLDSSize#[{;
    }

    int64_t getLDSBankCount() const {
      return }]#LDSBankCount#[{;
    }

    int64_t getLDSBankWidth() const {
      return }]#LDSBankWidth#[{;
    }

    int64_t getGlobalLoadLatency() const {
      return }]#globalLoadLatency#[{;
    }

    int64_t getLDSLoadLatency() const {
      return }]#LDSLoadLatency#[{;
    }

    int64_t getMaxVmcnt() const {
      return }]#maxVmcnt#[{;
    }

    int64_t getMaxLgkmcnt() const {
      return }]#maxLgkmcnt#[{;
    }

    int64_t getMaxExpcnt() const {
      return }]#maxExpcnt#[{;
    }

    int64_t getDefaultCodeObjectVersion() const {
      return }]#defaultCodeObjectVersion#[{;
    }

    ::llvm::SmallVector<int64_t> getSupportedCodeObjectVersions() const {
      return {}]#!interleave(supportedCodeObjectVersions, ", ")#[{};
    }

    ::llvm::StringRef getTargetDirective() const {
      return "}]#targetDirective#[{";
    }

    ::llvm::StringRef getABIVersion() const {
      return "}]#ABIVersion#[{";
    }
  }];
}

def WaveASM_TargetKindAttr_GFX942
    : WaveASM_TargetKindAttr<"GFX942Target", "gfx942"> {
  let archGeneration = "GFX9";
  let computeArch = "CDNA3";
  let maxVGPRs = 512;
  let maxSGPRs = 106;
  let maxAGPRs = 512;
  let defaultWaveSize = 64;
  let supportedWaveSizes = [64];
  let maxLDSSize = 65536;
  let LDSBankCount = 32;
  let LDSBankWidth = 4;
  let globalLoadLatency = 100;
  let LDSLoadLatency = 20;
  let maxVmcnt = 63;
  let maxLgkmcnt = 15;
  let maxExpcnt = 7;
  let defaultCodeObjectVersion = 5;
  let supportedCodeObjectVersions = [4, 5];
  let targetDirective = ".amdgcn_target \\\"amdgcn-amd-amdhsa--gfx942\\\"";
  let ABIVersion = "amdhsa";

  let extraClassDeclaration = [{
    // TODO: turn this into a bitmask enum.
    TargetFeature getFeatures() const {
      return TargetFeature::HasMFMA | TargetFeature::HasFP8 |
            TargetFeature::HasWave64 | TargetFeature::HasAtomicFAdd |
            TargetFeature::HasFlatScratch | TargetFeature::HasAGPRs |
            TargetFeature::HasScaledMFMA;
    }

    // TODO: wrap into tablegen somehow.
    int64_t getMFMALatency(::llvm::StringRef instrName) const {
      // MFMA latencies for GFX942 (approximate)
      if (instrName.contains("f32_32x32"))
        return 64;
      if (instrName.contains("f32_16x16"))
        return 32;
      if (instrName.contains("f16_32x32"))
        return 64;
      if (instrName.contains("f16_16x16"))
        return 32;
      if (instrName.contains("bf16"))
        return 32;
      if (instrName.contains("fp8") || instrName.contains("f8"))
        return 32;
      return 16; // Default
    }

    bool supportsInstruction(::llvm::StringRef instrName) const {
      // GFX942 supports most instructions.
      // Add specific checks for unsupported instructions if needed.
      return true;
    }

    std::optional<std::string> getTargetInstructionName(::llvm::StringRef genericName) const {
      // No aliasing needed for GFX942.
      return std::nullopt;
    }
  }]#commonClassDeclaration;
}

def WaveASM_TargetKindAttr_GFX950
    : WaveASM_TargetKindAttr<"GFX950Target", "gfx950"> {
  let archGeneration = "GFX9";
  let computeArch = "CDNA3.5";
  let maxVGPRs = 512;
  let maxSGPRs = 106;
  let maxAGPRs = 512;
  let defaultWaveSize = 64;
  let supportedWaveSizes = [64];
  let maxLDSSize = 65536;
  let LDSBankCount = 32;
  let LDSBankWidth = 4;
  let globalLoadLatency = 100;
  let LDSLoadLatency = 20;
  let maxVmcnt = 63;
  let maxLgkmcnt = 15;
  let maxExpcnt = 7;
  let defaultCodeObjectVersion = 5;
  let supportedCodeObjectVersions = [4, 5];
  let targetDirective = ".amdgcn_target \\\"amdgcn-amd-amdhsa--gfx950\\\"";
  let ABIVersion = "amdhsa";

  let extraClassDeclaration = [{
    TargetFeature getFeatures() const {
      return TargetFeature::HasMFMA | TargetFeature::HasFP8 |
            TargetFeature::HasWave64 | TargetFeature::HasAtomicFAdd |
            TargetFeature::HasFlatScratch | TargetFeature::HasAGPRs |
            TargetFeature::HasScaledMFMA | TargetFeature::HasXF32;
    }

    int64_t getMFMALatency(llvm::StringRef instrName) const {
      // GFX950 has improved MFMA throughput
      if (instrName.contains("f32_32x32"))
        return 64;
      if (instrName.contains("f32_16x16"))
        return 32;
      if (instrName.contains("f16_32x32"))
        return 32; // Improved over GFX942
      if (instrName.contains("f16_16x16"))
        return 16;
      if (instrName.contains("bf16"))
        return 16;
      if (instrName.contains("fp8") || instrName.contains("f8"))
        return 16;
      return 16;
    }

    bool supportsInstruction(::llvm::StringRef instrName) const {
      // GFX950 supports most GFX9 instructions plus new ones.
      return true;
    }

    std::optional<std::string> getTargetInstructionName(::llvm::StringRef genericName) const {
      // No aliasing needed.
      return std::nullopt;
    }
  }]#commonClassDeclaration;
}

def WaveASM_TargetKindAttr_GFX1250
    : WaveASM_TargetKindAttr<"GFX1250Target", "gfx1250"> {
  let archGeneration = "GFX12";
  let computeArch = "RDNA4";
  let maxVGPRs = 256;
  let maxSGPRs = 106;
  let maxAGPRs = 0;
  let defaultWaveSize = 32;
  let supportedWaveSizes = [32, 64];
  let maxLDSSize = 65536;
  let LDSBankCount = 32;
  let LDSBankWidth = 4;
  let globalLoadLatency = 80;
  let LDSLoadLatency = 16;
  let maxVmcnt = 63;
  let maxLgkmcnt = 63;
  let maxExpcnt = 7;
  let defaultCodeObjectVersion = 5;
  let supportedCodeObjectVersions = [4, 5];
  let targetDirective = ".amdgcn_target \\\"amdgcn-amd-amdhsa--gfx1250\\\"";
  let ABIVersion = "amdhsa";

  let extraClassDeclaration = [{
    TargetFeature getFeatures() const {
      return TargetFeature::HasWave32 | TargetFeature::HasWave64 |
            TargetFeature::HasAtomicFAdd | TargetFeature::HasFlatScratch |
            TargetFeature::HasPackedFP32 | TargetFeature::HasGlobalLoadLDS;
    }

    int64_t getMFMALatency(::llvm::StringRef instrName) const {
      // GFX1250 uses WMMA instead of MFMA
      if (instrName.contains("wmma"))
        return 16;
      return 0; // No traditional MFMA support
    }

    bool supportsInstruction(llvm::StringRef instrName) const {
      // GFX1250 (RDNA4) doesn't support traditional MFMA
      if (instrName.contains("v_mfma_"))
        return false;

      // Check for CDNA-specific instructions
      if (instrName.contains("_accvgpr_"))
        return false;

      return true;
    }

    std::optional<std::string>
    getTargetInstructionName(llvm::StringRef genericName) const {
      // Some instructions may have different names on RDNA
      // Add aliasing here if needed
      return std::nullopt;
    }
  }]#commonClassDeclaration;
}

def WaveASM_TargetAttr : WAVEASMAttr<"Target", "target"> {
  let summary = "Target GPU architecture";
  let description = [{
    Specifies the target GPU architecture and code object version.

    Code object versions:
    - 4: Standard code object
    - 5: With kernel argument preloading support

    Example:
    ```mlir
    waveasm.program @kernel target = #waveasm.target<#waveasm.gfx942, 5>
    ```
  }];

  let parameters = (ins "TargetAttrInterface":$target_kind,
      DefaultValuedParameter<"int64_t", "5">:$code_object_version);

  let assemblyFormat = "`<` $target_kind (`,` $code_object_version^)? `>`";
}

#endif // WaveASM_DIALECT_WAVEASMATTRS
