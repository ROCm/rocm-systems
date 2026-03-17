////////////////////////////////////////////////////////////////////////////////
//
// ROCm HotSwap — Cross-Family ISA Transpiler
//
// Translates gfx1250 (RDNA4) GPU binaries to run on gfx950 (CDNA4) hardware.
// Uses LLVM MC text round-trip with semantic translation rules.
//
////////////////////////////////////////////////////////////////////////////////

#ifndef HOTSWAP_TRANSPILER_HPP
#define HOTSWAP_TRANSPILER_HPP

#include "hotswap.hpp"
#include <string>
#include <vector>

namespace rocr {
namespace hotswap {

/// Statistics from a transpile operation.
struct TranspileStats {
  uint32_t total_instructions = 0;
  uint32_t translated_passthrough = 0;  // Same mnemonic, no change needed
  uint32_t translated_renamed = 0;      // Mnemonic renamed (e.g., global_load_b32 → global_load_dword)
  uint32_t translated_waitcnt = 0;      // Wait counter merged
  uint32_t translated_exec = 0;         // EXEC widened for wave32→wave64
  uint32_t unsupported_skipped = 0;     // Instructions with no translation (NOPped)
  uint32_t assembly_errors = 0;         // Instructions that failed to assemble
};

/// Check if a cross-family transpile is needed (e.g., gfx1250 → gfx950).
/// Returns true if source and target are from different ISA families
/// (RDNA vs CDNA) and transpilation is supported.
bool NeedsTranspile(const std::string& source_isa, const std::string& target_isa);

/// Transpile a code object from one ISA family to another.
///
/// This performs full disassemble→translate→reassemble:
/// 1. Disassemble source .text using LLVM MC (source ISA decoder)
/// 2. Apply semantic translation rules (mnemonic renaming, wait counter
///    merging, EXEC widening, etc.)
/// 3. Assemble translated text using LLVM MC (target ISA assembler)
/// 4. Patch .text, kernel descriptors, ELF metadata
///
/// The output may be larger than the input (wave size adaptation adds
/// instructions). The elf_data buffer is reallocated if needed.
///
/// @param elf_data   Pointer to ELF buffer (may be reallocated)
/// @param elf_size   Size of ELF buffer (updated if reallocated)
/// @param source_isa Source ISA string (e.g., "amdgcn-amd-amdhsa--gfx1250")
/// @param target_isa Target ISA string (e.g., "amdgcn-amd-amdhsa--gfx950")
/// @param stats      Output: translation statistics
/// @return RewriteResult with status
RewriteResult TranspileCodeObject(void** elf_data, size_t* elf_size,
                                  const std::string& source_isa,
                                  const std::string& target_isa,
                                  TranspileStats* stats = nullptr);

/// Translate a single assembly line from source ISA to target ISA.
/// Returns the translated line(s). May return multiple lines if an
/// instruction expands (e.g., EXEC widening adds a second instruction).
///
/// @param asm_line   Assembly line from MCInstPrinter (source ISA)
/// @param source_cpu Source CPU name (e.g., "gfx1250")
/// @param target_cpu Target CPU name (e.g., "gfx950")
/// @return Translated assembly lines for target ISA
std::vector<std::string> TranslateInstruction(const std::string& asm_line,
                                               const std::string& source_cpu,
                                               const std::string& target_cpu);

}  // namespace hotswap
}  // namespace rocr

#endif  // HOTSWAP_TRANSPILER_HPP
