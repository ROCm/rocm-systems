//===-- aegisbit/CodeObjectParser.h - ELF Parsing ---------------*- C++ -*-===//
//
// Part of the AegisBit Project
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Parser for AMD GPU ELF code objects.
/// Extracts sections, symbols, and kernel information from AMDGPU ELFs.
///
//===----------------------------------------------------------------------===//

#ifndef AEGISBIT_CODE_OBJECT_PARSER_H
#define AEGISBIT_CODE_OBJECT_PARSER_H

#include "aegisbit/Types.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include <cstdint>
#include <string>
#include <vector>

namespace aegisbit {

/// Parser for AMD GPU ELF code objects.
///
/// AMD GPU kernels are packaged as ELF64 little-endian files with:
/// - e_machine = 0xE0 (EM_AMDGPU)
/// - e_flags containing EF_AMDGPU_MACH_* for ISA version
/// - .text section with CDNA ISA machine code
/// - .rodata section with 64-byte kernel descriptors
/// - .note section with YAML metadata
/// - Symbol table with kernel names and .kd descriptors
class CodeObjectParser {
public:
  /// Parse an AMD GPU code object from raw bytes.
  /// \param Bytes Raw ELF file contents
  /// \return Parsed code object or error
  static llvm::Expected<ParsedCodeObject> parse(llvm::ArrayRef<uint8_t> Bytes);

  /// Parse an AMD GPU code object from a file.
  /// \param Path Path to the ELF file
  /// \return Parsed code object or error
  static llvm::Expected<ParsedCodeObject> parseFile(llvm::StringRef Path);

  /// Check if bytes represent an AMDGPU code object.
  /// Validates ELF magic and e_machine field.
  /// \param Bytes Raw file contents
  /// \return true if this appears to be an AMDGPU ELF
  static bool isAMDGPUCodeObject(llvm::ArrayRef<uint8_t> Bytes);

  /// Get GPU architecture string from ELF flags.
  /// \param EFlags e_flags value from ELF header
  /// \return Architecture string (e.g., "gfx942", "gfx1100")
  static std::string getGPUArch(uint32_t EFlags);

  /// ELF constants for AMDGPU
  static constexpr uint16_t EM_AMDGPU = 0xE0;
  static constexpr size_t ELF_HEADER_SIZE = 64;

  /// EF_AMDGPU_MACH values for common architectures
  /// Values from llvm/include/llvm/BinaryFormat/ELF.h (EF_AMDGPU_MACH_AMDGCN_*)
  static constexpr uint32_t EF_AMDGPU_MACH_MASK = 0xFF;

  // GFX9 family (Vega, CDNA)
  static constexpr uint32_t EF_AMDGPU_MACH_GFX900 = 0x02c;  // Vega 10
  static constexpr uint32_t EF_AMDGPU_MACH_GFX906 = 0x02f;  // Vega 20
  static constexpr uint32_t EF_AMDGPU_MACH_GFX908 = 0x030;  // MI100 (CDNA1)
  static constexpr uint32_t EF_AMDGPU_MACH_GFX90A = 0x03f;  // MI200 (CDNA2)
  static constexpr uint32_t EF_AMDGPU_MACH_GFX940 = 0x04a;  // MI300A
  static constexpr uint32_t EF_AMDGPU_MACH_GFX941 = 0x04b;  // MI300X variant
  static constexpr uint32_t EF_AMDGPU_MACH_GFX942 = 0x04c;  // MI300X (CDNA3)
  static constexpr uint32_t EF_AMDGPU_MACH_GFX950 = 0x04f;  // MI350X (CDNA4)

  // GFX10 family (RDNA1/2)
  static constexpr uint32_t EF_AMDGPU_MACH_GFX1010 = 0x033;  // Navi 10
  static constexpr uint32_t EF_AMDGPU_MACH_GFX1030 = 0x036;  // Navi 21

  // GFX11 family (RDNA3)
  static constexpr uint32_t EF_AMDGPU_MACH_GFX1100 = 0x041;  // Navi 31

  // GFX12 family (RDNA4 / future CDNA)
  static constexpr uint32_t EF_AMDGPU_MACH_GFX1200 = 0x048;
  static constexpr uint32_t EF_AMDGPU_MACH_GFX1250 = 0x049;  // MI450 (CDNA5)

private:
  /// Extract section by name from parsed ELF
  static llvm::Expected<std::vector<uint8_t>>
  extractSection(llvm::ArrayRef<uint8_t> ELF, llvm::StringRef Name,
                 uint16_t& SectionIndex);

  /// Parse symbol table and identify kernels
  static llvm::Error parseSymbols(llvm::ArrayRef<uint8_t> ELF,
                                   ParsedCodeObject& Result);
};

} // namespace aegisbit

#endif // AEGISBIT_CODE_OBJECT_PARSER_H
