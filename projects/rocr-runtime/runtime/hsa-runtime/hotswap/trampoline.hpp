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

#ifndef ROCR_HOTSWAP_TRAMPOLINE_HPP
#define ROCR_HOTSWAP_TRAMPOLINE_HPP

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace llvm {
class MCSubtargetInfo;
class MCInstrInfo;
class MCRegisterInfo;
class MCAsmInfo;
class MCContext;
class MCCodeEmitter;
} // namespace llvm

namespace rocr {
namespace hotswap {

/// A trampoline: replacement instruction bytes + branch back to original code.
struct Trampoline {
  uint64_t original_offset;   // Offset of the replaced instruction in .text
  uint32_t original_size;     // Size of the original instruction
  std::vector<uint8_t> bytes; // Assembled trampoline bytes (replacement + branch back)
};

/// Build a trampoline for a size-changing rewrite.
///
/// The trampoline contains:
///   1. The replacement instruction sequence (assembled from asm strings)
///   2. An s_branch back to (original_offset + original_size)
///
/// The original instruction site will be overwritten with an s_branch
/// to the trampoline (at trampoline_text_offset).
///
/// @param asm_lines             Assembly lines for the replacement sequence
/// @param original_offset       Byte offset of the original instruction in .text
/// @param original_size         Byte size of the original instruction
/// @param trampoline_text_offset  Where the trampoline will be placed in .text
/// @param cpu                   Target CPU string (e.g. "gfx1201")
/// @param STI                   LLVM subtarget info
/// @param MCII                  LLVM instruction info
/// @param MRI                   LLVM register info
/// @param MAI                   LLVM asm info
/// @param Ctx                   LLVM MC context
/// @param CE                    LLVM code emitter
/// @return                      Assembled trampoline, or empty on error
Trampoline BuildTrampoline(const std::vector<std::string>& asm_lines,
                           uint64_t original_offset,
                           uint32_t original_size,
                           uint64_t trampoline_text_offset,
                           const std::string& cpu,
                           llvm::MCSubtargetInfo* STI,
                           llvm::MCInstrInfo* MCII,
                           llvm::MCRegisterInfo* MRI,
                           const llvm::MCAsmInfo* MAI,
                           llvm::MCContext* Ctx,
                           llvm::MCCodeEmitter* CE);

/// Encode an s_branch instruction to a relative target.
/// The branch offset is in dwords, relative to the PC after the branch
/// instruction (PC + 4).
///
/// @param from_offset  Byte offset of the branch instruction in .text
/// @param to_offset    Byte offset of the branch target in .text
/// @param out_bytes    Output: 4 bytes of the encoded s_branch
/// @return             true if the offset fits in 16-bit signed range
bool EncodeSBranch(uint64_t from_offset, uint64_t to_offset,
                   uint8_t out_bytes[4]);

/// Encode an s_nop 0 instruction (4 bytes).
void EncodeSNop(uint8_t out_bytes[4]);

} // namespace hotswap
} // namespace rocr

#endif // ROCR_HOTSWAP_TRAMPOLINE_HPP
