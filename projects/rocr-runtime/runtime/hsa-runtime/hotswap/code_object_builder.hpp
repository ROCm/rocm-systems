////////////////////////////////////////////////////////////////////////////////
//
// Code Object Builder
//
// Utilities for extracting instruction bytes from AMDGCN ELF code objects and
// rebuilding code objects after the MLIR pipeline has re-emitted assembly.
//
////////////////////////////////////////////////////////////////////////////////

#ifndef ROCR_HOTSWAP_CODE_OBJECT_BUILDER_HPP
#define ROCR_HOTSWAP_CODE_OBJECT_BUILDER_HPP

#include <cstdint>
#include <string>
#include <vector>

namespace hotswap {

struct TextSection {
  std::vector<uint8_t> bytes;
  uint64_t offset = 0;
  uint64_t size = 0;
  bool valid = false;
};

/// Extract the .text section bytes from an AMDGCN ELF code object.
TextSection extractTextSection(const std::vector<uint8_t> &elfData);

/// Splice re-emitted instruction assembly into a device assembly template.
///
/// The template is the full device assembly from hipcc -S (containing kernel
/// descriptor, metadata, alignment directives, etc.). This function replaces
/// only the instruction body between the kernel function label and s_endpgm
/// (inclusive) with the new instructions, preserving all metadata.
///
/// @param deviceAsmTemplate  The full device assembly template from hipcc -S
/// @param newInstructions    Re-emitted instruction assembly from the pipeline
/// @param kernelSymbol       Mangled kernel symbol name
/// @return Complete assembly text ready for llvm-mc
std::string spliceInstructions(const std::string &deviceAsmTemplate,
                               const std::string &newInstructions,
                               const std::string &kernelSymbol);

/// Rebuild a code object from assembly text.
///
/// Runs llvm-mc to assemble and ld.lld to link, producing a shared-object
/// ELF code object suitable for hipModuleLoadData.
///
/// @param assemblyText  Complete assembly text (with metadata)
/// @param targetISA     Target ISA (e.g., "gfx942")
/// @param llvmBinDir    Path to LLVM bin directory (e.g., "/opt/rocm/llvm/bin")
/// @return The code object bytes, or empty on failure
std::vector<uint8_t> rebuildCodeObject(const std::string &assemblyText,
                                       const std::string &targetISA,
                                       const std::string &llvmBinDir);

/// Splice only the "core" computation (between s_cbranch_execz and the branch
/// target label) from translated assembly into a device assembly template.
///
/// Used for cross-ISA translation where the preamble (ABI/workgroup ID setup)
/// comes from the target-ISA template and only the computational body is
/// replaced with translated instructions.
///
/// @param deviceAsmTemplate  The full gfx942 device assembly (hipcc -S)
/// @param translatedCore     The core instructions to insert (no preamble)
/// @param kernelSymbol       Mangled kernel symbol name
/// @return Complete assembly text ready for llvm-mc
std::string spliceCoreInstructions(const std::string &deviceAsmTemplate,
                                   const std::string &translatedCore,
                                   const std::string &kernelSymbol);

/// Extract only the "core" instructions from full pipeline output.
///
/// Returns the lines between the first s_cbranch_execz and the label
/// immediately before s_endpgm (exclusive on both ends).
std::string extractCoreFromPipelineOutput(const std::string &fullAssembly);

/// Read an entire file into a byte vector.
std::vector<uint8_t> readFile(const std::string &path);

/// Read an entire file into a string.
std::string readFileAsString(const std::string &path);

} // namespace hotswap

#endif // ROCR_HOTSWAP_CODE_OBJECT_BUILDER_HPP
