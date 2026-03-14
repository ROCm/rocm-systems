////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2014-2025, Advanced Micro Devices, Inc. All rights reserved.
//
// Developed by:
//
//                 AMD Research and AMD HSA Software Development
//
//                 Advanced Micro Devices, Inc.
//
//                 www.amd.com
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal with the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
//  - Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimers.
//  - Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimers in
//    the documentation and/or other materials provided with the distribution.
//  - Neither the names of Advanced Micro Devices, Inc,
//    nor the names of its contributors may be used to endorse or promote
//    products derived from this Software without specific prior written
//    permission.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
// OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
// ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS WITH THE SOFTWARE.
//
////////////////////////////////////////////////////////////////////////////////

#ifndef ROCR_HOTSWAP_HPP
#define ROCR_HOTSWAP_HPP

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "hsa.h"

namespace rocr {
namespace hotswap {

/// Information about a decoded instruction within the .text section.
struct DecodedInst {
  uint64_t offset;         // Byte offset within .text
  uint32_t size;           // Instruction size in bytes (4 or 8 for AMDGPU)
  std::string mnemonic;    // Decoded mnemonic string
  std::vector<uint8_t> bytes; // Raw instruction bytes

  // MCInst is stored internally during rewriting but not exposed in the
  // public interface to avoid LLVM header dependencies.
};

/// Result of a rewrite operation, returned by RewriteCodeObject.
struct RewriteResult {
  hsa_status_t status;
  uint32_t rules_matched;     // Number of rule matches applied
  uint32_t trampolines_added; // Number of trampolines created
};

/// Check if hotswap is enabled (HSA_HOTSWAP_RULES env var is set).
bool IsEnabled();

/// Check if ISA override is enabled (HSA_HOTSWAP_ISA_OVERRIDE env var is set).
/// When enabled, the loader skips the ISA compatibility check, allowing code
/// objects compiled for one GPU (e.g. gfx950) to be loaded on another (e.g.
/// gfx942). The hotswap rewriter is responsible for replacing any incompatible
/// instructions before the code reaches the GPU.
bool IsIsaOverrideEnabled();

/// Patch the ELF's ISA metadata (e_flags and .note sections) to match the
/// target agent's ISA, so that downstream loader code accepts the code object.
///
/// @param elf_data       Mutable ELF buffer
/// @param elf_size       Size of the ELF buffer
/// @param target_isa     Target ISA string (the agent's native ISA)
/// @return               true if patching succeeded
bool PatchElfIsa(void* elf_data, size_t elf_size,
                 const std::string& target_isa);

/// Retarget the .text section from one ISA to another by disassembling each
/// instruction with the source ISA and re-assembling it for the target ISA.
/// Instructions that exist on both ISAs (same mnemonic) are re-encoded.
/// Instructions that don't exist on the target produce an error and are
/// left as-is (caller should apply rewrite rules for those).
///
/// @param elf_data       Mutable ELF buffer
/// @param elf_size       Size of the ELF buffer
/// @param source_isa     Source ISA (e.g. "amdgcn-amd-amdhsa--gfx950")
/// @param target_isa     Target ISA (e.g. "amdgcn-amd-amdhsa--gfx942")
/// @return               RewriteResult with status and count of retargeted instructions
RewriteResult RetargetCodeObject(void* elf_data, size_t elf_size,
                                 const std::string& source_isa,
                                 const std::string& target_isa);

/// Apply ISA rewrite rules to an in-memory ELF code object buffer.
///
/// The buffer is modified in-place. If size-changing rewrites require
/// trampolines, the buffer may be reallocated (via the provided realloc
/// callback or by growing an internal copy).
///
/// @param elf_data    Pointer to mutable ELF buffer
/// @param elf_size    Size of the ELF buffer
/// @param isa_name    Target ISA string (e.g. "amdgcn-amd-amdhsa--gfx1201")
/// @return            RewriteResult with status and statistics
RewriteResult RewriteCodeObject(void* elf_data, size_t elf_size,
                                const std::string& isa_name);

/// Apply ISA rewrite rules, potentially growing the buffer.
/// Returns a new buffer (caller must free with free()) if the buffer grew,
/// or nullptr if the original buffer was modified in-place.
///
/// @param elf_data     Pointer to ELF buffer (read-only input)
/// @param elf_size     Size of the input ELF buffer
/// @param out_data     Output: pointer to rewritten buffer (may == elf_data)
/// @param out_size     Output: size of rewritten buffer
/// @param isa_name     Target ISA string
/// @return             RewriteResult with status and statistics
RewriteResult RewriteCodeObjectGrow(const void* elf_data, size_t elf_size,
                                    void** out_data, size_t* out_size,
                                    const std::string& isa_name);

} // namespace hotswap
} // namespace rocr

#endif // ROCR_HOTSWAP_HPP
