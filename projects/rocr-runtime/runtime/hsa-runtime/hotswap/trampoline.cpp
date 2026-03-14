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

#include "trampoline.hpp"

#include <cstring>
#include <iostream>
#include <string>

#include <llvm/MC/MCAsmBackend.h>
#include <llvm/MC/MCAsmInfo.h>
#include <llvm/MC/MCCodeEmitter.h>
#include <llvm/MC/MCContext.h>
#include <llvm/MC/MCInstrInfo.h>
#include <llvm/MC/MCObjectFileInfo.h>
#include <llvm/MC/MCObjectWriter.h>
#include <llvm/MC/MCParser/MCTargetAsmParser.h>
#include <llvm/MC/MCRegisterInfo.h>
#include <llvm/MC/MCStreamer.h>
#include <llvm/MC/MCSubtargetInfo.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/raw_ostream.h>
#if LLVM_VERSION_MAJOR > 13
#include <llvm/MC/TargetRegistry.h>
#else
#include <llvm/Support/TargetRegistry.h>
#endif

namespace rocr {
namespace hotswap {

// s_branch encoding: SOPP format
// Bits [31:23] = 0b10111111 10 = 0xBF82 (s_branch opcode for GFX9+)
// Bits [15:0]  = signed 16-bit offset in dwords (relative to PC after branch)
//
// For GFX10+/GFX11+/GFX12, the s_branch encoding is the same SOPP format.
static constexpr uint32_t S_BRANCH_OPCODE = 0xBF820000u;
static constexpr uint32_t S_NOP_OPCODE    = 0xBF800000u;

bool EncodeSBranch(uint64_t from_offset, uint64_t to_offset,
                   uint8_t out_bytes[4]) {
  // Branch offset is in dwords, relative to (from_offset + 4).
  // target_pc = (from_offset + 4) + offset_dwords * 4
  // offset_dwords = (to_offset - from_offset - 4) / 4
  int64_t byte_delta = static_cast<int64_t>(to_offset) -
                       static_cast<int64_t>(from_offset) - 4;

  if (byte_delta % 4 != 0) {
    std::cerr << "hotswap: branch offset not dword-aligned\n";
    return false;
  }

  int64_t dword_offset = byte_delta / 4;

  // s_branch uses a signed 16-bit immediate
  if (dword_offset < -32768 || dword_offset > 32767) {
    std::cerr << "hotswap: branch offset out of range (" << dword_offset
              << " dwords)\n";
    return false;
  }

  uint32_t encoded = S_BRANCH_OPCODE | (static_cast<uint16_t>(dword_offset) & 0xFFFF);
  std::memcpy(out_bytes, &encoded, 4);
  return true;
}

void EncodeSNop(uint8_t out_bytes[4]) {
  uint32_t encoded = S_NOP_OPCODE;
  std::memcpy(out_bytes, &encoded, 4);
}

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
                           llvm::MCCodeEmitter* CE) {
  Trampoline result;
  result.original_offset = original_offset;
  result.original_size = original_size;

  // Build assembly source from lines
  std::string asm_source;
  for (auto& line : asm_lines) {
    asm_source += line + "\n";
  }

  // Use LLVM MC to assemble the replacement sequence.
  // We need a full assembly pipeline here to handle the instruction encoding.

  llvm::StringRef asm_ref(asm_source);
  auto buf = llvm::MemoryBuffer::getMemBuffer(asm_ref, "", false);
  if (!buf->getBufferSize()) {
    std::cerr << "hotswap: trampoline assembly source is empty\n";
    return result;
  }

  llvm::SourceMgr src_mgr;
  src_mgr.AddNewSourceBuffer(std::move(buf), llvm::SMLoc());

  // Output stream for assembled bytes
  std::string data;
  auto data_stream = std::make_unique<llvm::raw_string_ostream>(data);
  auto bos = std::make_unique<llvm::buffer_ostream>(*data_stream);
  llvm::raw_pwrite_stream* os = bos.get();

  // Look up the AMDGPU target
  std::string error;
  llvm::Triple triple("amdgcn-amd-amdhsa");
  const llvm::Target* target =
      llvm::TargetRegistry::lookupTarget("amdgcn", triple, error);
  if (!target) {
    std::cerr << "hotswap: trampoline target lookup failed: " << error << "\n";
    return result;
  }

  // Create fresh code emitter and backend for this assembly
  llvm::MCTargetOptions mc_opts;
#if LLVM_VERSION_MAJOR > 14
  llvm::MCCodeEmitter* ce = target->createMCCodeEmitter(*MCII, *Ctx);
#else
  llvm::MCCodeEmitter* ce = target->createMCCodeEmitter(*MCII, *MRI, *Ctx);
#endif
  llvm::MCAsmBackend* mab = target->createMCAsmBackend(*STI, *MRI, mc_opts);
  if (!ce || !mab) {
    std::cerr << "hotswap: failed to create code emitter/backend\n";
    return result;
  }

#if LLVM_VERSION_MAJOR > 20
  auto streamer = std::unique_ptr<llvm::MCStreamer>(
      target->createMCObjectStreamer(
          triple, *Ctx,
          std::unique_ptr<llvm::MCAsmBackend>(mab),
          mab->createObjectWriter(*os),
          std::unique_ptr<llvm::MCCodeEmitter>(ce), *STI));
#else
  auto streamer = std::unique_ptr<llvm::MCStreamer>(
      target->createMCObjectStreamer(
          triple, *Ctx,
          std::unique_ptr<llvm::MCAsmBackend>(mab),
          mab->createObjectWriter(*os),
          std::unique_ptr<llvm::MCCodeEmitter>(ce), *STI,
          mc_opts.MCRelaxAll, mc_opts.MCIncrementalLinkerCompatible, false));
#endif

  if (!streamer) {
    std::cerr << "hotswap: failed to create MC streamer\n";
    return result;
  }

  auto parser = std::unique_ptr<llvm::MCAsmParser>(
      llvm::createMCAsmParser(src_mgr, *Ctx, *streamer, *MAI));
  auto tap = std::unique_ptr<llvm::MCTargetAsmParser>(
      target->createMCAsmParser(*STI, *parser, *MCII, mc_opts));
  if (!tap) {
    std::cerr << "hotswap: failed to create target asm parser\n";
    return result;
  }
  parser->setTargetParser(*tap);

  if (parser->Run(true)) {
    std::cerr << "hotswap: trampoline assembly failed\n";
    return result;
  }

  bos.reset();
  data_stream->flush();

  // Extract .text from the assembled ELF object.
  // The output is a minimal ELF — find the .text section.
  if (data.size() < 64) {
    std::cerr << "hotswap: assembled output too small\n";
    return result;
  }

  // Simple ELF .text extraction: parse ELF header to find section headers,
  // then find the section named ".text".
  const uint8_t* elf = reinterpret_cast<const uint8_t*>(data.data());

  // Verify ELF magic
  if (elf[0] != 0x7f || elf[1] != 'E' || elf[2] != 'L' || elf[3] != 'F') {
    std::cerr << "hotswap: assembled output is not ELF\n";
    return result;
  }

  // Parse as 64-bit ELF (AMDGPU is always 64-bit)
  uint8_t ei_class = elf[4];
  if (ei_class != 2) {
    std::cerr << "hotswap: expected 64-bit ELF\n";
    return result;
  }

  // ELF64 header fields
  uint64_t e_shoff;
  uint16_t e_shentsize, e_shnum, e_shstrndx;
  std::memcpy(&e_shoff, elf + 40, 8);
  std::memcpy(&e_shentsize, elf + 58, 2);
  std::memcpy(&e_shnum, elf + 60, 2);
  std::memcpy(&e_shstrndx, elf + 62, 2);

  if (e_shoff == 0 || e_shnum == 0 || e_shstrndx >= e_shnum) {
    std::cerr << "hotswap: invalid ELF section headers\n";
    return result;
  }

  // Read section string table
  const uint8_t* shstrtab_hdr = elf + e_shoff + e_shstrndx * e_shentsize;
  uint64_t strtab_offset, strtab_size;
  std::memcpy(&strtab_offset, shstrtab_hdr + 24, 8);
  std::memcpy(&strtab_size, shstrtab_hdr + 32, 8);

  if (strtab_offset + strtab_size > data.size()) {
    std::cerr << "hotswap: section string table out of bounds\n";
    return result;
  }

  const char* strtab = reinterpret_cast<const char*>(elf + strtab_offset);

  // Find .text section
  for (uint16_t i = 0; i < e_shnum; ++i) {
    const uint8_t* sh = elf + e_shoff + i * e_shentsize;
    uint32_t sh_name;
    std::memcpy(&sh_name, sh, 4);

    if (sh_name < strtab_size &&
        std::strcmp(strtab + sh_name, ".text") == 0) {
      uint64_t sh_offset, sh_size;
      std::memcpy(&sh_offset, sh + 24, 8);
      std::memcpy(&sh_size, sh + 32, 8);

      if (sh_offset + sh_size > data.size()) {
        std::cerr << "hotswap: .text section out of bounds\n";
        return result;
      }

      // Copy the assembled .text bytes
      result.bytes.assign(elf + sh_offset, elf + sh_offset + sh_size);
      break;
    }
  }

  if (result.bytes.empty()) {
    std::cerr << "hotswap: no .text section in assembled output\n";
    return result;
  }

  // Append s_branch back to (original_offset + original_size)
  uint64_t branch_back_from = trampoline_text_offset + result.bytes.size();
  uint64_t branch_back_to = original_offset + original_size;

  uint8_t branch_bytes[4];
  if (!EncodeSBranch(branch_back_from, branch_back_to, branch_bytes)) {
    std::cerr << "hotswap: failed to encode return branch for trampoline\n";
    result.bytes.clear();
    return result;
  }

  result.bytes.insert(result.bytes.end(), branch_bytes, branch_bytes + 4);
  return result;
}

} // namespace hotswap
} // namespace rocr
