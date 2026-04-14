////////////////////////////////////////////////////////////////////////////////
//
// MLIR-Based Transpilation Pipeline
//
// Orchestrates the full transformation:
//   1. Lift binary (MCInst → waveasm MLIR)
//   2. Cross-target mnemonic mapping
//   3. Wave width translation (wave32 → wave64)
//   4. Emit assembly text
//   5. Assemble to bytes via LLVM MC
//
////////////////////////////////////////////////////////////////////////////////

#ifndef ROCR_HOTSWAP_PIPELINE_HPP
#define ROCR_HOTSWAP_PIPELINE_HPP

#include <cstdint>
#include <string>
#include <vector>

namespace hotswap {

struct PipelineStats {
  uint64_t totalInstructions = 0;
  uint64_t liftedInstructions = 0;
  uint64_t rawFallbacks = 0;
  uint64_t failedDisassembly = 0;
  uint64_t mnemonicsRemapped = 0;
};

struct PipelineResult {
  bool success = false;
  std::string assemblyText;
  std::vector<uint8_t> bytes;
  PipelineStats stats;
  std::string errorMessage;
};

/// Run the full MLIR transpilation pipeline on raw instruction bytes.
///
/// @param sourceBytes  Raw machine code bytes from the source ISA
/// @param sourceISA    Source ISA name (e.g., "gfx1250")
/// @param targetISA    Target ISA name (e.g., "gfx942")
/// @param kernelName   Name for the kernel in the IR
/// @return PipelineResult with assembly text and optionally assembled bytes
PipelineResult runPipeline(const uint8_t *sourceBytes, size_t sourceSize,
                           const std::string &sourceISA,
                           const std::string &targetISA,
                           const std::string &kernelName = "kernel");

/// Assemble AMDGPU assembly text to machine code bytes using LLVM MC.
///
/// @param asmText     Assembly text to assemble
/// @param targetISA   Target ISA for the assembler (e.g., "gfx942")
/// @return The assembled bytes, or empty vector on failure
std::vector<uint8_t> assembleToBytes(const std::string &asmText,
                                     const std::string &targetISA);

} // namespace hotswap

#endif // ROCR_HOTSWAP_PIPELINE_HPP
