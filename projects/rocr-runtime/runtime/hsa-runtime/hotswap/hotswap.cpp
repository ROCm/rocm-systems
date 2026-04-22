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

#include "hotswap.hpp"
#include "hotswap_rules.hpp"
#include "trampoline.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <vector>

#include <llvm/Config/llvm-config.h>
#include <llvm/MC/MCAsmBackend.h>
#include <llvm/MC/MCAsmInfo.h>
#include <llvm/MC/MCCodeEmitter.h>
#include <llvm/MC/MCContext.h>
#include <llvm/MC/MCDisassembler/MCDisassembler.h>
#include <llvm/MC/MCInstPrinter.h>
#include <llvm/MC/MCInstrInfo.h>
#include <llvm/MC/MCObjectFileInfo.h>
#include <llvm/MC/MCObjectWriter.h>
#include <llvm/MC/MCParser/MCAsmParser.h>
#include <llvm/MC/MCParser/MCTargetAsmParser.h>
#include <llvm/MC/MCRegisterInfo.h>
#include <llvm/MC/MCStreamer.h>
#include <llvm/MC/MCSubtargetInfo.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/raw_ostream.h>
#if LLVM_VERSION_MAJOR > 13
#include <llvm/MC/TargetRegistry.h>
#else
#include <llvm/Support/TargetRegistry.h>
#endif

namespace rocr {
namespace hotswap {

namespace {

// ── LLVM MC Context (lazy-initialized, one per process) ──────────────────────

struct LLVMState {
  const llvm::Target* target = nullptr;
  std::unique_ptr<llvm::MCRegisterInfo> MRI;
  std::unique_ptr<const llvm::MCAsmInfo> MAI;
  std::unique_ptr<llvm::MCInstrInfo> MCII;
  std::unique_ptr<llvm::MCSubtargetInfo> STI;
  std::unique_ptr<llvm::MCContext> Ctx;
  std::unique_ptr<llvm::MCDisassembler> disasm;
  std::unique_ptr<llvm::MCInstPrinter> printer;
  llvm::MCCodeEmitter* CE = nullptr; // owned by Ctx or target
  std::string cpu;                   // e.g. "gfx1201"
  bool valid = false;
};

static std::once_flag g_llvm_init_flag;
static bool g_llvm_initialized = false;

static void InitLLVMTargets() {
  LLVMInitializeAMDGPUTargetInfo();
  LLVMInitializeAMDGPUTargetMC();
  LLVMInitializeAMDGPUAsmParser();
  LLVMInitializeAMDGPUDisassembler();
  g_llvm_initialized = true;
}

/// Extract CPU name from ISA string like "amdgcn-amd-amdhsa--gfx1201".
/// Returns everything after the last "--" or "-" that starts with "gfx".
static std::string ExtractCPU(const std::string& isa_name) {
  // Look for "gfx" in the ISA string
  size_t pos = isa_name.rfind("gfx");
  if (pos != std::string::npos) {
    // Take from "gfx" to end, stopping at non-alnum
    std::string cpu;
    for (size_t i = pos; i < isa_name.size(); ++i) {
      char c = isa_name[i];
      if ((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') ||
          (c >= 'A' && c <= 'Z'))
        cpu += c;
      else
        break;
    }
    return cpu;
  }
  return "";
}

/// Initialize LLVM MC state for a given ISA. Cached per CPU name to avoid
/// LLVM global state conflicts from creating multiple instances.
static LLVMState& InitLLVMCached(const std::string& isa_name);

static LLVMState InitLLVMImpl(const std::string& isa_name) {
  std::call_once(g_llvm_init_flag, InitLLVMTargets);

  LLVMState state;
  state.cpu = ExtractCPU(isa_name);
  if (state.cpu.empty()) {
    std::cerr << "hotswap: cannot extract CPU from ISA '" << isa_name << "'\n";
    return state;
  }

  std::string error;
  llvm::Triple triple("amdgcn-amd-amdhsa");

  state.target = llvm::TargetRegistry::lookupTarget("amdgcn", triple, error);
  if (!state.target) {
    std::cerr << "hotswap: target lookup failed: " << error << "\n";
    return state;
  }

  state.MRI.reset(state.target->createMCRegInfo(llvm::Triple("amdgcn-amd-amdhsa")));
  if (!state.MRI) return state;

  llvm::MCTargetOptions mc_opts;
#if LLVM_VERSION_MAJOR > 9
  state.MAI.reset(state.target->createMCAsmInfo(
      *state.MRI, llvm::Triple("amdgcn-amd-amdhsa"), mc_opts));
#else
  state.MAI.reset(state.target->createMCAsmInfo(
      *state.MRI, "amdgcn-amd-amdhsa"));
#endif
  if (!state.MAI) return state;

  state.MCII.reset(state.target->createMCInstrInfo());
  if (!state.MCII) return state;

  state.STI.reset(state.target->createMCSubtargetInfo(
      llvm::Triple("amdgcn-amd-amdhsa"), state.cpu, ""));
  if (!state.STI || !state.STI->isCPUStringValid(state.cpu)) {
    std::cerr << "hotswap: invalid CPU '" << state.cpu << "'\n";
    return state;
  }

#if LLVM_VERSION_MAJOR > 12
  state.Ctx = std::make_unique<llvm::MCContext>(
      triple, state.MAI.get(), state.MRI.get(), state.STI.get());
#else
  // Older LLVM needs MCObjectFileInfo
  auto MOFI = std::make_unique<llvm::MCObjectFileInfo>();
  state.Ctx = std::make_unique<llvm::MCContext>(
      state.MAI.get(), state.MRI.get(), MOFI.get());
  MOFI->InitMCObjectFileInfo(triple, true, *state.Ctx);
#endif
  // Same rationale as the `initInlineSourceManager` call in the
  // target-assembler path (~line 1332 below): the MCContext ctor
  // defaults `SourceMgr *Mgr = nullptr`, so any MC-layer diagnostic
  // (from the disassembler here, from the target assembler there)
  // that reaches `MCContext::reportCommon` / `diagnose` with a
  // valid SMLoc would trip the
  // `llvm_unreachable("Either SourceMgr should be available")`
  // abort at MCContext.cpp:1093.  The inline SourceMgr gives the
  // diagnose path a non-null fallback so the diagnostic is routed
  // through the default diag handler (stderr) and the caller's
  // downstream error-path code runs, instead of the process
  // abort'ing on SIG6.  `initInlineSourceManager` is idempotent;
  // this context is NOT reset between disassemblies (one ctx per
  // loaded module), so we only need to init it once here.
  state.Ctx->initInlineSourceManager();

  state.disasm.reset(
      state.target->createMCDisassembler(*state.STI, *state.Ctx));
  if (!state.disasm) {
    std::cerr << "hotswap: failed to create disassembler\n";
    return state;
  }

  // Create instruction printer for dump mode
  unsigned asm_variant = state.MAI->getAssemblerDialect();
  state.printer.reset(state.target->createMCInstPrinter(
      triple, asm_variant, *state.MAI, *state.MCII, *state.MRI));

#if LLVM_VERSION_MAJOR > 14
  state.CE = state.target->createMCCodeEmitter(*state.MCII, *state.Ctx);
#else
  state.CE = state.target->createMCCodeEmitter(
      *state.MCII, *state.MRI, *state.Ctx);
#endif

  state.valid = true;
  return state;
}

static std::mutex g_llvm_cache_mutex;
static std::map<std::string, LLVMState> g_llvm_cache;

static LLVMState& InitLLVMCached(const std::string& isa_name) {
  std::string cpu = ExtractCPU(isa_name);
  std::lock_guard<std::mutex> lock(g_llvm_cache_mutex);
  auto it = g_llvm_cache.find(cpu);
  if (it != g_llvm_cache.end()) return it->second;
  g_llvm_cache[cpu] = InitLLVMImpl(isa_name);
  return g_llvm_cache[cpu];
}

// Backwards-compatible alias
static LLVMState InitLLVM(const std::string& isa_name) {
  // For non-retarget paths, return a copy (they don't need caching)
  return InitLLVMImpl(isa_name);
}

// ── ELF helpers ──────────────────────────────────────────────────────────────

struct ElfSection {
  uint32_t name_idx;
  std::string name;
  uint32_t type;
  uint64_t offset;  // File offset
  uint64_t size;
  uint64_t addr;    // Virtual address
};

struct ElfSymbol {
  std::string name;
  uint64_t value;    // Section-relative offset (or virtual address)
  uint64_t size;
  uint8_t info;      // Type + binding
  uint16_t shndx;    // Section index
};

struct ElfInfo {
  std::vector<ElfSection> sections;
  std::vector<ElfSymbol> symbols;
  int text_section_idx = -1;
  uint64_t text_offset = 0;  // File offset of .text
  uint64_t text_size = 0;
  uint64_t text_addr = 0;    // Virtual address of .text
};

static bool ParseElfInfo(const uint8_t* elf, size_t elf_size, ElfInfo& info) {
  if (elf_size < 64) return false;
  if (elf[0] != 0x7f || elf[1] != 'E' || elf[2] != 'L' || elf[3] != 'F')
    return false;
  if (elf[4] != 2) return false; // Must be 64-bit

  uint64_t e_shoff;
  uint16_t e_shentsize, e_shnum, e_shstrndx;
  std::memcpy(&e_shoff, elf + 40, 8);
  std::memcpy(&e_shentsize, elf + 58, 2);
  std::memcpy(&e_shnum, elf + 60, 2);
  std::memcpy(&e_shstrndx, elf + 62, 2);

  if (e_shoff == 0 || e_shnum == 0) return false;
  if (e_shoff + static_cast<uint64_t>(e_shnum) * e_shentsize > elf_size)
    return false;

  // Read section string table
  const char* shstrtab = nullptr;
  uint64_t shstrtab_size = 0;
  if (e_shstrndx < e_shnum) {
    const uint8_t* sh = elf + e_shoff + e_shstrndx * e_shentsize;
    uint64_t sh_offset, sh_size;
    std::memcpy(&sh_offset, sh + 24, 8);
    std::memcpy(&sh_size, sh + 32, 8);
    if (sh_offset + sh_size <= elf_size) {
      shstrtab = reinterpret_cast<const char*>(elf + sh_offset);
      shstrtab_size = sh_size;
    }
  }

  // Parse all sections
  info.sections.resize(e_shnum);
  for (uint16_t i = 0; i < e_shnum; ++i) {
    const uint8_t* sh = elf + e_shoff + i * e_shentsize;
    auto& sec = info.sections[i];
    std::memcpy(&sec.name_idx, sh, 4);
    std::memcpy(&sec.type, sh + 4, 4);
    std::memcpy(&sec.addr, sh + 16, 8);
    std::memcpy(&sec.offset, sh + 24, 8);
    std::memcpy(&sec.size, sh + 32, 8);

    if (shstrtab && sec.name_idx < shstrtab_size) {
      sec.name = shstrtab + sec.name_idx;
    }

    if (sec.name == ".text") {
      info.text_section_idx = i;
      info.text_offset = sec.offset;
      info.text_size = sec.size;
      info.text_addr = sec.addr;
    }
  }

  // Parse symbol table
  for (uint16_t i = 0; i < e_shnum; ++i) {
    auto& sec = info.sections[i];
    if (sec.type != 2 /* SHT_SYMTAB */ && sec.type != 11 /* SHT_DYNSYM */)
      continue;

    // Get the associated string table
    const uint8_t* sh = elf + e_shoff + i * e_shentsize;
    uint32_t sh_link;
    std::memcpy(&sh_link, sh + 40, 4);

    const char* symstrtab = nullptr;
    uint64_t symstrtab_size = 0;
    if (sh_link < e_shnum) {
      auto& link_sec = info.sections[sh_link];
      if (link_sec.offset + link_sec.size <= elf_size) {
        symstrtab = reinterpret_cast<const char*>(elf + link_sec.offset);
        symstrtab_size = link_sec.size;
      }
    }

    // Parse symbols (each entry is 24 bytes for ELF64)
    size_t sym_count = sec.size / 24;
    for (size_t j = 0; j < sym_count; ++j) {
      if (sec.offset + (j + 1) * 24 > elf_size) break;
      const uint8_t* sym_entry = elf + sec.offset + j * 24;

      ElfSymbol sym;
      uint32_t st_name;
      std::memcpy(&st_name, sym_entry, 4);
      sym.info = sym_entry[4];
      std::memcpy(&sym.shndx, sym_entry + 6, 2);
      std::memcpy(&sym.value, sym_entry + 8, 8);
      std::memcpy(&sym.size, sym_entry + 16, 8);

      if (symstrtab && st_name < symstrtab_size) {
        sym.name = symstrtab + st_name;
      }

      info.symbols.push_back(std::move(sym));
    }
  }

  return info.text_section_idx >= 0;
}

// ── Kernel name lookup ───────────────────────────────────────────────────────

/// Find which kernel (if any) an instruction at a given .text offset belongs to.
static std::string FindKernelAtOffset(const ElfInfo& elf_info,
                                      uint64_t text_offset) {
  for (auto& sym : elf_info.symbols) {
    // Kernel symbols have type STT_FUNC or STT_AMDGPU_HSA_KERNEL (0xa)
    uint8_t sym_type = sym.info & 0xf;
    if (sym_type != 2 /* STT_FUNC */ && sym_type != 10 /* STT_AMDGPU_HSA_KERNEL */)
      continue;
    if (sym.shndx != static_cast<uint16_t>(elf_info.text_section_idx))
      continue;

    // Check if offset is within this kernel's range
    uint64_t sym_start = sym.value;
    uint64_t sym_end = sym.value + sym.size;
    if (text_offset >= sym_start && text_offset < sym_end) {
      return sym.name;
    }
  }
  return "";
}

// ── Decoded instruction with MCInst ──────────────────────────────────────────

struct InternalDecodedInst {
  uint64_t offset;
  uint32_t size;
  llvm::MCInst inst;
  std::string mnemonic;
};

// ── Instruction decode/match/patch ───────────────────────────────────────────

static bool DecodeTextSection(const uint8_t* text, uint64_t text_size,
                              const LLVMState& llvm_state,
                              std::vector<InternalDecodedInst>& decoded) {
  uint64_t pos = 0;
  while (pos < text_size) {
    InternalDecodedInst di;
    di.offset = pos;

    llvm::ArrayRef<uint8_t> bytes(text + pos, text_size - pos);
    uint64_t inst_size = 0;

    auto status = llvm_state.disasm->getInstruction(
        di.inst, inst_size, bytes, pos, llvm::nulls());

    if (status == llvm::MCDisassembler::Fail) {
      // Skip 4 bytes (minimum AMDGPU instruction size) on decode failure
      di.size = 4;
      di.mnemonic = "<unknown>";
      pos += 4;
    } else {
      di.size = static_cast<uint32_t>(inst_size);

      // Get mnemonic via InstPrinter
      if (llvm_state.printer) {
        std::string str;
        llvm::raw_string_ostream rso(str);
        llvm_state.printer->printInst(&di.inst, 0, "", *llvm_state.STI, rso);
        rso.flush();

        // Extract mnemonic (first whitespace-delimited token)
        size_t start = str.find_first_not_of(" \t");
        if (start != std::string::npos) {
          size_t end = str.find_first_of(" \t", start);
          di.mnemonic = str.substr(start, end - start);
        }
      } else {
        // Fallback: use opcode name from instruction info
        di.mnemonic = llvm_state.MCII->getName(di.inst.getOpcode()).str();
      }

      pos += inst_size;
    }

    decoded.push_back(std::move(di));
  }
  return true;
}

static bool MatchRule(const RewriteRule& rule, const InternalDecodedInst& inst,
                      const ElfInfo& elf_info) {
  // Check mnemonic
  if (!rule.match_mnemonic.empty() && rule.match_mnemonic != inst.mnemonic)
    return false;

  // Check offset
  if (rule.match_offset >= 0 &&
      static_cast<uint64_t>(rule.match_offset) != inst.offset)
    return false;

  // Check kernel
  if (!rule.match_kernel.empty()) {
    std::string kernel = FindKernelAtOffset(elf_info, inst.offset);
    if (kernel != rule.match_kernel) return false;
  }

  // Check operands
  if (!rule.operands.empty()) {
    if (rule.operands.size() > static_cast<size_t>(inst.inst.getNumOperands()))
      return false;

    for (size_t i = 0; i < rule.operands.size(); ++i) {
      auto& match = rule.operands[i];
      auto& operand = inst.inst.getOperand(i);

      switch (match.kind) {
        case OperandMatch::Kind::Wildcard:
          break; // Always matches
        case OperandMatch::Kind::Immediate:
          if (!operand.isImm() || operand.getImm() != match.imm_value)
            return false;
          break;
        case OperandMatch::Kind::RegClass:
          // Register class matching would require MCRegisterInfo lookups.
          // For now, just verify it's a register operand.
          if (!operand.isReg()) return false;
          break;
      }
    }
  }

  return true;
}

/// Apply a same-size mnemonic swap by looking up the replacement opcode
/// and re-encoding with the same operands.
static bool ApplyMnemonicSwap(const RewriteRule& rule,
                              InternalDecodedInst& inst,
                              uint8_t* text,
                              const LLVMState& llvm_state) {
  // For mnemonic swaps, we need to find the opcode for the replacement
  // mnemonic. We do this by assembling a minimal instruction with the
  // replacement mnemonic and copying the encoded bytes.

  // Build an assembly line with the replacement mnemonic and the original
  // operands printed out.
  if (!llvm_state.printer) return false;

  std::string orig_str;
  llvm::raw_string_ostream rso(orig_str);
  llvm_state.printer->printInst(&inst.inst, 0, "", *llvm_state.STI, rso);
  rso.flush();

  // Replace the mnemonic in the printed string
  size_t start = orig_str.find_first_not_of(" \t");
  if (start == std::string::npos) return false;
  size_t end = orig_str.find_first_of(" \t", start);

  std::string new_asm;
  if (end != std::string::npos) {
    new_asm = rule.replace_mnemonic + orig_str.substr(end);
  } else {
    new_asm = rule.replace_mnemonic;
  }

  // Assemble the replacement instruction using the code emitter
  // For same-size swaps, we use a simplified path: assemble via MC
  // and verify the size matches.

  // Use LLVM MC assembler pipeline for this single instruction
  llvm::StringRef asm_ref(new_asm);
  auto buf = llvm::MemoryBuffer::getMemBuffer(asm_ref, "", false);
  llvm::SourceMgr src_mgr;
  src_mgr.AddNewSourceBuffer(std::move(buf), llvm::SMLoc());

  std::string data;
  auto data_stream = std::make_unique<llvm::raw_string_ostream>(data);
  auto bos = std::make_unique<llvm::buffer_ostream>(*data_stream);

  llvm::MCTargetOptions mc_opts;
  llvm::Triple triple("amdgcn-amd-amdhsa");

#if LLVM_VERSION_MAJOR > 14
  llvm::MCCodeEmitter* ce =
      llvm_state.target->createMCCodeEmitter(*llvm_state.MCII, *llvm_state.Ctx);
#else
  llvm::MCCodeEmitter* ce = llvm_state.target->createMCCodeEmitter(
      *llvm_state.MCII, *llvm_state.MRI, *llvm_state.Ctx);
#endif
  llvm::MCAsmBackend* mab =
      llvm_state.target->createMCAsmBackend(*llvm_state.STI, *llvm_state.MRI, mc_opts);

  if (!ce || !mab) return false;

#if LLVM_VERSION_MAJOR > 20
  auto streamer = std::unique_ptr<llvm::MCStreamer>(
      llvm_state.target->createMCObjectStreamer(
          triple, *llvm_state.Ctx,
          std::unique_ptr<llvm::MCAsmBackend>(mab),
          mab->createObjectWriter(*bos),
          std::unique_ptr<llvm::MCCodeEmitter>(ce), *llvm_state.STI));
#else
  auto streamer = std::unique_ptr<llvm::MCStreamer>(
      llvm_state.target->createMCObjectStreamer(
          triple, *llvm_state.Ctx,
          std::unique_ptr<llvm::MCAsmBackend>(mab),
          mab->createObjectWriter(*bos),
          std::unique_ptr<llvm::MCCodeEmitter>(ce), *llvm_state.STI,
          mc_opts.MCRelaxAll, mc_opts.MCIncrementalLinkerCompatible, false));
#endif

  if (!streamer) return false;

  auto parser = std::unique_ptr<llvm::MCAsmParser>(
      llvm::createMCAsmParser(src_mgr, *llvm_state.Ctx, *streamer, *llvm_state.MAI));
  auto tap = std::unique_ptr<llvm::MCTargetAsmParser>(
      llvm_state.target->createMCAsmParser(*llvm_state.STI, *parser,
                                           *llvm_state.MCII, mc_opts));
  if (!tap) return false;
  parser->setTargetParser(*tap);

  if (parser->Run(true)) {
    std::cerr << "hotswap: mnemonic swap assembly failed for '"
              << new_asm << "'\n";
    return false;
  }

  bos.reset();
  data_stream->flush();

  // Extract .text from the assembled ELF
  // Simple extraction: find .text section bytes
  const uint8_t* elf_bytes = reinterpret_cast<const uint8_t*>(data.data());
  size_t elf_sz = data.size();
  if (elf_sz < 64) return false;

  ElfInfo asm_elf;
  if (!ParseElfInfo(elf_bytes, elf_sz, asm_elf)) return false;

  if (asm_elf.text_size != inst.size) {
    std::cerr << "hotswap: mnemonic swap produced different size ("
              << asm_elf.text_size << " vs " << inst.size << ")\n";
    return false;
  }

  // Copy the assembled bytes into the original .text
  std::memcpy(text + inst.offset, elf_bytes + asm_elf.text_offset, inst.size);
  return true;
}

/// Apply a raw byte replacement.
static bool ApplyByteReplace(const RewriteRule& rule,
                             const InternalDecodedInst& inst,
                             uint8_t* text, uint64_t text_size) {
  if (rule.replace_bytes.size() > inst.size) {
    std::cerr << "hotswap: replace_bytes larger than original instruction ("
              << rule.replace_bytes.size() << " > " << inst.size << ")\n";
    return false;
  }

  std::memcpy(text + inst.offset, rule.replace_bytes.data(),
              rule.replace_bytes.size());

  // Pad remaining bytes with s_nop if replacement is smaller
  uint32_t remaining = inst.size - static_cast<uint32_t>(rule.replace_bytes.size());
  uint64_t pad_offset = inst.offset + rule.replace_bytes.size();
  while (remaining >= 4) {
    uint8_t nop[4];
    EncodeSNop(nop);
    std::memcpy(text + pad_offset, nop, 4);
    pad_offset += 4;
    remaining -= 4;
  }

  return true;
}

// ── Kernel descriptor update ─────────────────────────────────────────────────

static void UpdateKernelDescriptor(uint8_t* elf_data, size_t elf_size,
                                   const ElfInfo& elf_info,
                                   const std::string& kernel_name,
                                   int32_t extra_vgprs, int32_t extra_sgprs) {
  // Find the kernel descriptor symbol (name ending with ".kd")
  std::string kd_name = kernel_name + ".kd";

  for (auto& sym : elf_info.symbols) {
    if (sym.name != kd_name) continue;
    if (sym.shndx >= elf_info.sections.size()) continue;

    auto& sec = elf_info.sections[sym.shndx];
    uint64_t kd_file_offset = sec.offset + sym.value;

    if (kd_file_offset + 64 > elf_size) continue;

    uint8_t* kd = elf_data + kd_file_offset;

    // Read COMPUTE_PGM_RSRC1 (offset 48 within kernel descriptor)
    uint32_t rsrc1;
    std::memcpy(&rsrc1, kd + 48, 4);

    if (extra_vgprs > 0) {
      // GRANULATED_WORKITEM_VGPR_COUNT is bits [5:0]
      // Granularity depends on architecture:
      //   GFX6-8: (vgprs / 4) - 1
      //   GFX9:   (vgprs / 4) - 1
      //   GFX10+: (vgprs / 8) - 1 (wave32) or (vgprs / 4) - 1 (wave64)
      // We use granularity of 4 as a safe default (adds 4 VGPRs per increment)
      uint32_t current = rsrc1 & 0x3F;
      uint32_t extra_granules = (static_cast<uint32_t>(extra_vgprs) + 3) / 4;
      uint32_t new_val = current + extra_granules;
      if (new_val > 63) new_val = 63;
      rsrc1 = (rsrc1 & ~0x3Fu) | new_val;
    }

    if (extra_sgprs > 0) {
      // GRANULATED_WAVEFRONT_SGPR_COUNT is bits [9:6]
      // Granularity: (sgprs / 8) - 1
      uint32_t current = (rsrc1 >> 6) & 0xF;
      uint32_t extra_granules = (static_cast<uint32_t>(extra_sgprs) + 7) / 8;
      uint32_t new_val = current + extra_granules;
      if (new_val > 15) new_val = 15;
      rsrc1 = (rsrc1 & ~(0xFu << 6)) | (new_val << 6);
    }

    std::memcpy(kd + 48, &rsrc1, 4);
    return;
  }
}

// ── Dump helpers ─────────────────────────────────────────────────────────────

static bool ShouldDump() {
  static int dump = -1;
  if (dump < 0) {
    const char* env = std::getenv("HSA_HOTSWAP_DUMP");
    dump = (env && env[0] == '1') ? 1 : 0;
  }
  return dump == 1;
}

static void DumpInstructions(const char* label,
                             const std::vector<InternalDecodedInst>& decoded,
                             const uint8_t* text) {
  std::cerr << "=== hotswap " << label << " ===\n";
  for (auto& d : decoded) {
    std::cerr << std::hex << std::setw(8) << std::setfill('0') << d.offset
              << ": ";
    for (uint32_t i = 0; i < d.size; ++i) {
      std::cerr << std::hex << std::setw(2) << std::setfill('0')
                << static_cast<int>(text[d.offset + i]) << " ";
    }
    std::cerr << "  " << d.mnemonic << "\n";
  }
  std::cerr << std::dec;
}

} // anonymous namespace

// ── Public API ───────────────────────────────────────────────────────────────

bool IsEnabled() {
  const char* rules = std::getenv("HSA_HOTSWAP_RULES");
  const char* override_isa = std::getenv("HSA_HOTSWAP_ISA_OVERRIDE");
  // Enabled if either rules file is set OR ISA override is set
  return (rules && *rules) || (override_isa && *override_isa && override_isa[0] != '0');
}

bool IsIsaOverrideEnabled() {
  const char* env = std::getenv("HSA_HOTSWAP_ISA_OVERRIDE");
  return env && *env && env[0] != '0';
}

bool PatchElfIsa(void* elf_data, size_t elf_size,
                 const std::string& target_isa) {
  uint8_t* elf = static_cast<uint8_t*>(elf_data);
  if (elf_size < 64) return false;
  if (elf[0] != 0x7f || elf[1] != 'E' || elf[2] != 'L' || elf[3] != 'F')
    return false;
  if (elf[4] != 2) return false; // Must be 64-bit

  // Extract the target gfx number for e_flags patching.
  // e_flags for AMDGPU ELF encodes the EF_AMDGPU_MACH value.
  // We need to map gfx names to their EF_AMDGPU_MACH constants.
  std::string target_cpu = ExtractCPU(target_isa);
  if (target_cpu.empty()) return false;

  // Map of gfx name → EF_AMDGPU_MACH value (from llvm/include/llvm/Support/ScopedPrinter.h)
  // and llvm/include/llvm/BinaryFormat/ELF.h.
  //
  // TODO(mach-table-stale): several entries below disagree with the
  // authoritative LLVM values in ROCm's own `llvm/BinaryFormat/ELF.h`;
  // they survive only because gfx942 (the sole ISA we patch TO in
  // production) is correct and the bogus entries go unused.  Specifically:
  //
  //   - gfx940  = 0x04a  ← LLVM reassigned 0x04a to gfx1151.
  //                        gfx940 was a deprecated prototype and is no
  //                        longer listed in ELF.h at all.
  //   - gfx941  = 0x04b  ← LLVM marks 0x04b as RESERVED_0X4B; gfx941 is
  //                        likewise a deprecated prototype and not listed.
  //   - gfx1151 = 0x04b  ← RESERVED in LLVM; true value is 0x04a.
  //   - gfx1201 = 0x04a  ← LLVM says 0x04e.
  //
  // Also missing entirely: gfx1250 (0x049), gfx1251 (0x05a), gfx1152
  // (0x055), gfx1153 (0x058), and the gfx*_generic subtargets. Any
  // future work that patches TO one of these ISAs must re-sync the
  // table against the installed LLVM headers (and ideally generate it
  // from them rather than maintaining a copy).
  struct GfxMach { const char* name; uint32_t mach; };
  static const GfxMach gfx_mach_map[] = {
    {"gfx900",  0x02c}, {"gfx902",  0x02d}, {"gfx904",  0x02e},
    {"gfx906",  0x02f}, {"gfx908",  0x030}, {"gfx909",  0x031},
    {"gfx90a",  0x03f}, {"gfx90c",  0x032}, {"gfx940",  0x04a},
    {"gfx941",  0x04b}, {"gfx942",  0x04c}, {"gfx950",  0x04f},
    {"gfx1010", 0x033}, {"gfx1011", 0x034}, {"gfx1012", 0x035},
    {"gfx1030", 0x036}, {"gfx1031", 0x037}, {"gfx1032", 0x038},
    {"gfx1033", 0x039}, {"gfx1034", 0x03e}, {"gfx1035", 0x03d},
    {"gfx1100", 0x041}, {"gfx1101", 0x046}, {"gfx1102", 0x047},
    {"gfx1103", 0x044}, {"gfx1150", 0x043}, {"gfx1151", 0x04b},
    {"gfx1200", 0x048}, {"gfx1201", 0x04a},
    {nullptr, 0}
  };

  uint32_t target_mach = 0;
  for (auto* p = gfx_mach_map; p->name; ++p) {
    if (target_cpu == p->name) { target_mach = p->mach; break; }
  }
  if (target_mach == 0) {
    std::cerr << "hotswap: unknown target CPU '" << target_cpu
              << "' for ISA override\n";
    return false;
  }

  // Patch e_flags: the MACH value is in bits [7:0] of e_flags
  // EF_AMDGPU_MACH mask = 0xFF
  uint32_t e_flags;
  std::memcpy(&e_flags, elf + 48, 4);
  e_flags = (e_flags & ~0xFFu) | (target_mach & 0xFF);
  std::memcpy(elf + 48, &e_flags, 4);

  // Also patch .note sections that contain the ISA name string.
  // AMDGPU code objects have NT_AMDGPU_ISA notes (type 27) with the ISA
  // string. We need to find and replace them.
  uint64_t e_shoff;
  uint16_t e_shentsize, e_shnum, e_shstrndx;
  std::memcpy(&e_shoff, elf + 40, 8);
  std::memcpy(&e_shentsize, elf + 58, 2);
  std::memcpy(&e_shnum, elf + 60, 2);
  std::memcpy(&e_shstrndx, elf + 62, 2);

  if (e_shoff == 0 || e_shnum == 0) return true; // e_flags patched, no notes

  for (uint16_t i = 0; i < e_shnum; ++i) {
    const uint8_t* sh = elf + e_shoff + i * e_shentsize;
    uint32_t sh_type;
    std::memcpy(&sh_type, sh + 4, 4);

    if (sh_type != 7 /* SHT_NOTE */) continue;

    uint64_t sh_offset, sh_size;
    std::memcpy(&sh_offset, sh + 24, 8);
    std::memcpy(&sh_size, sh + 32, 8);

    if (sh_offset + sh_size > elf_size) continue;

    // Walk through notes in this section
    uint64_t pos = sh_offset;
    while (pos + 12 <= sh_offset + sh_size) {
      uint32_t namesz, descsz, type;
      std::memcpy(&namesz, elf + pos, 4);
      std::memcpy(&descsz, elf + pos + 4, 4);
      std::memcpy(&type, elf + pos + 8, 4);

      uint32_t namesz_aligned = (namesz + 3) & ~3u;
      uint32_t descsz_aligned = (descsz + 3) & ~3u;
      uint64_t note_total = 12 + namesz_aligned + descsz_aligned;

      if (pos + note_total > sh_offset + sh_size) break;

      // NT_AMDGPU_ISA = 27, owner = "AMDGPU"
      if (type == 27 && namesz > 0) {
        const char* owner = reinterpret_cast<const char*>(elf + pos + 12);
        if (std::strncmp(owner, "AMDGPU", 6) == 0) {
          // The desc contains the ISA string (null-terminated)
          uint8_t* desc = elf + pos + 12 + namesz_aligned;
          std::string orig_isa(reinterpret_cast<const char*>(desc), descsz);

          // Replace gfx part of the ISA string in-place if it fits
          size_t gfx_pos = orig_isa.find("gfx");
          if (gfx_pos != std::string::npos) {
            // Find end of gfx token
            size_t gfx_end = gfx_pos;
            while (gfx_end < orig_isa.size() &&
                   orig_isa[gfx_end] != ':' && orig_isa[gfx_end] != '\0')
              ++gfx_end;
            std::string orig_gfx = orig_isa.substr(gfx_pos, gfx_end - gfx_pos);

            if (target_cpu.size() <= orig_gfx.size()) {
              // Fits in-place — overwrite with target CPU, pad with nulls
              std::memcpy(desc + gfx_pos, target_cpu.c_str(), target_cpu.size());
              for (size_t j = target_cpu.size(); j < orig_gfx.size(); ++j) {
                desc[gfx_pos + j] = '\0';
              }
              std::cerr << "hotswap: ISA override patched note: "
                        << orig_gfx << " -> " << target_cpu << "\n";
            } else {
              std::cerr << "hotswap: ISA override note patch failed: "
                        << target_cpu << " longer than " << orig_gfx << "\n";
            }
          }
        }
      }

      pos += note_total;
    }
  }

  return true;
}

RewriteResult RetargetCodeObject(void* elf_data, size_t elf_size,
                                 const std::string& source_isa,
                                 const std::string& target_isa) {
  RewriteResult result = {HSA_STATUS_SUCCESS, 0, 0};

  // Parse ELF to find .text
  ElfInfo elf_info;
  uint8_t* elf = static_cast<uint8_t*>(elf_data);
  if (!ParseElfInfo(elf, elf_size, elf_info)) return result;
  if (elf_info.text_size == 0) return result;

  // Initialize LLVM MC for source ISA (disassembler) — use cache
  LLVMState& src_state = InitLLVMCached(source_isa);
  if (!src_state.valid) {
    std::cerr << "hotswap: retarget: failed to init source ISA '"
              << source_isa << "'\n";
    result.status = HSA_STATUS_ERROR;
    return result;
  }

  // Extract target CPU name (we don't need a full LLVMState for the target —
  // we build all MC objects locally to avoid LLVM global state conflicts)
  std::string tgt_cpu = ExtractCPU(target_isa);
  if (tgt_cpu.empty()) {
    std::cerr << "hotswap: retarget: cannot extract CPU from target ISA '"
              << target_isa << "'\n";
    result.status = HSA_STATUS_ERROR;
    return result;
  }
  // Reuse the same LLVM Target (it's the same AMDGPU backend for both)
  const llvm::Target* tgt_target = src_state.target;

  uint8_t* text = elf + elf_info.text_offset;

  // Decode all instructions with the source ISA
  std::vector<InternalDecodedInst> decoded;
  if (!DecodeTextSection(text, elf_info.text_size, src_state, decoded)) {
    std::cerr << "hotswap: retarget: instruction decode failed\n";
    result.status = HSA_STATUS_ERROR;
    return result;
  }

  if (ShouldDump()) {
    DumpInstructions("RETARGET BEFORE", decoded, text);
  }

  // Pre-pass: Replace gfx950-only instructions with trampolines that
  // emulate the behavior using gfx942-compatible instructions.
  //
  // gfx942 and gfx950 share identical instruction encodings for all standard
  // VALU/SMEM/VMEM/SOPP. Only the gfx950-only instructions need handling:
  //   - v_cvt_scalef32_pk_fp4_f32 (D23D): 2x f32 → packed FP4 E2M1
  //   - v_cvt_scalef32_pk_f32_fp4 (D23F): packed FP4 E2M1 → 2x f32
  //   - v_mfma_f32_16x16x128_f8f6f4 (D3AD): mixed-format MFMA
  //
  // For FP4 conversion instructions, we replace with trampolines that
  // write zero to the destination register. This is a minimal emulation
  // that prevents crashes. The kernel will produce degraded results
  // (zero-quantized output) but won't hit ILLEGAL_INSTRUCTION.

  uint32_t gfx950_only_replaced = 0;

  // Build a map of NOP sled regions (after each s_endpgm) where trampolines
  // can be placed. Each kernel has padding NOPs for 256-byte alignment.
  struct NopSled {
    uint64_t start;
    uint64_t end;
    uint64_t write_pos; // Next available position for a trampoline
  };
  std::vector<NopSled> nop_sleds;

  for (size_t i = 0; i < decoded.size(); ++i) {
    if (decoded[i].mnemonic == "s_endpgm") {
      uint64_t sled_start = decoded[i].offset + decoded[i].size;
      // Find the end of the NOP sled (first non-NOP after s_endpgm)
      uint64_t sled_end = sled_start;
      for (size_t j = i + 1; j < decoded.size(); ++j) {
        if (decoded[j].mnemonic == "s_nop") {
          sled_end = decoded[j].offset + decoded[j].size;
        } else {
          break;
        }
      }
      if (sled_end > sled_start + 8) { // Need at least 8 bytes for a trampoline
        nop_sleds.push_back({sled_start, sled_end, sled_start});
      }
    }
  }

  // For each gfx950-only instruction, find the nearest NOP sled to place its trampoline
  auto findNearestSled = [&](uint64_t offset) -> NopSled* {
    NopSled* best = nullptr;
    int64_t best_dist = INT64_MAX;
    for (auto& sled : nop_sleds) {
      if (sled.write_pos + 8 > sled.end) continue; // Full
      int64_t dist = std::abs(static_cast<int64_t>(sled.write_pos) -
                              static_cast<int64_t>(offset));
      // s_branch range is +/-128KB
      if (dist < 131072 && dist < best_dist) {
        best = &sled;
        best_dist = dist;
      }
    }
    return best;
  };

  for (auto& di : decoded) {
    if (di.size != 8 || di.offset + 8 > elf_info.text_size) continue;

    uint32_t dword0 = 0, dword1 = 0;
    std::memcpy(&dword0, text + di.offset, 4);
    std::memcpy(&dword1, text + di.offset + 4, 4);
    uint32_t opcode_hi = (dword0 >> 16) & 0xFFFF;

    bool is_fp4_convert = (opcode_hi >= 0xD23D && opcode_hi <= 0xD243);
    bool is_mfma_f8f6f4 = (opcode_hi == 0xD3AD || opcode_hi == 0xD3AE);
    bool is_cvt_pk_f16 = (opcode_hi == 0xD267); // v_cvt_pk_f16_f32 (gfx950)
    bool is_cvt_pk_bf16 = (opcode_hi == 0xD268); // v_cvt_pk_bf16_f32 (gfx950)
    bool is_bitop3 = (opcode_hi == 0xD233);      // v_bitop3_b16 (gfx950)

    bool is_gfx950_only = is_fp4_convert || is_mfma_f8f6f4 ||
                          is_cvt_pk_f16 || is_cvt_pk_bf16 || is_bitop3;
    if (!is_gfx950_only) continue;

    // Extract destination VGPR from VOP3 encoding
    uint8_t vdst = dword0 & 0xFF;

    if (is_cvt_pk_f16) {
      // Swap v_cvt_pk_f16_f32 (D267) → v_cvt_pkrtz_f16_f32 (D296)
      // Same VOP3 operand format, just different opcode. Slightly
      // different rounding (RTZ vs RNE) but functionally compatible.
      uint32_t new_dw0 = (dword0 & ~0xFFFF0000u) | 0xD2960000u;
      std::memcpy(text + di.offset, &new_dw0, 4);
      // DW1 unchanged
    } else if (is_cvt_pk_bf16) {
      // v_cvt_pk_bf16_f32 vDst, vSrc0, vSrc1: pack bf16(src0) and bf16(src1)
      // Full emulation using NOP sled trampoline + temp VGPR (v255):
      //   1. v_lshrrev_b32 vDst, 16, vSrc0   (4B) → bf16(src0) in [15:0]
      //   2. v_lshrrev_b32 v255, 16, vSrc1   (4B) → bf16(src1) in [15:0]
      //   3. v_lshl_or_b32 vDst, v255, 16, vDst (8B) → pack both halves
      //   4. s_branch <return>                (4B)
      // Total: 20 bytes in trampoline
      uint16_t src0_raw = dword1 & 0x1FF;
      uint16_t src1_raw = (dword1 >> 9) & 0x1FF;
      uint8_t src0_vgpr = static_cast<uint8_t>(src0_raw & 0xFF);
      uint8_t src1_vgpr = static_cast<uint8_t>(src1_raw & 0xFF);
      uint8_t vtmp = 255; // temp VGPR

      NopSled* sled = findNearestSled(di.offset);
      if (sled && sled->write_pos + 20 <= sled->end) {
        uint64_t tp = sled->write_pos;

        // 1. v_lshrrev_b32 vDst, 16, vSrc0
        uint32_t i1 = (0x10u << 25) | (static_cast<uint32_t>(vdst) << 17) |
                      (static_cast<uint32_t>(src0_vgpr) << 9) | 0x90u;
        std::memcpy(text + tp, &i1, 4);

        // 2. v_lshrrev_b32 v255, 16, vSrc1
        uint32_t i2 = (0x10u << 25) | (static_cast<uint32_t>(vtmp) << 17) |
                      (static_cast<uint32_t>(src1_vgpr) << 9) | 0x90u;
        std::memcpy(text + tp + 4, &i2, 4);

        // 3. v_lshl_or_b32 vDst, v255, 16, vDst (VOP3 opcode D200)
        uint32_t i3_dw0 = 0xD2000000u | static_cast<uint32_t>(vdst);
        uint32_t i3_dw1 = (256u + static_cast<uint32_t>(vtmp)) |
                          (0x90u << 9) |
                          ((256u + static_cast<uint32_t>(vdst)) << 18);
        std::memcpy(text + tp + 8, &i3_dw0, 4);
        std::memcpy(text + tp + 12, &i3_dw1, 4);

        // 4. s_branch back
        uint8_t br_back[4];
        if (EncodeSBranch(tp + 16, di.offset + 8, br_back)) {
          std::memcpy(text + tp + 16, br_back, 4);

          // Replace original with s_branch to trampoline + s_nop
          uint8_t br_fwd[4];
          if (EncodeSBranch(di.offset, tp, br_fwd)) {
            std::memcpy(text + di.offset, br_fwd, 4);
            uint8_t nop[4];
            EncodeSNop(nop);
            std::memcpy(text + di.offset + 4, nop, 4);
            sled->write_pos += 20;
            di.mnemonic = "<replaced>";
            ++gfx950_only_replaced;
            continue;
          }
        }
      }
      // Fallback: v_lshrrev for src0 half only
      uint32_t lshr_word = (0x10u << 25) |
                           (static_cast<uint32_t>(vdst) << 17) |
                           (static_cast<uint32_t>(src0_vgpr) << 9) | 0x90u;
      std::memcpy(text + di.offset, &lshr_word, 4);
      uint8_t nop[4];
      EncodeSNop(nop);
      std::memcpy(text + di.offset + 4, nop, 4);
    } else if (is_fp4_convert) {
      // v_cvt_scalef32_pk_fp4_f32 vDst, vSrc0, vSrc1, vScale
      // Emulate FP4 E2M1 quantization using trampoline with v255 temp:
      //   1. v_mul_f32 vDst, vSrc0, vScale    (4B VOP2) — scale src0
      //   2. v_mul_f32 vDst, vDst, 2.0        (4B VOP2) — multiply by 2 for E2M1 index
      //   3. v_cvt_u32_f32 vDst, vDst         (4B VOP1) — truncate to uint
      //   4. v_min_u32 vDst, 7, vDst          (4B VOP2) — clamp to [0,7]
      //   5. v_mul_f32 v255, vSrc1, vScale    (4B VOP2) — scale src1
      //   6. v_mul_f32 v255, v255, 2.0        (4B VOP2)
      //   7. v_cvt_u32_f32 v255, v255         (4B VOP1)
      //   8. v_min_u32 v255, 7, v255          (4B VOP2)
      //   9. v_lshl_or_b32 vDst, v255, 4, vDst (8B VOP3) — pack nibbles
      //  10. s_branch <return>                (4B)
      // Total: 44 bytes
      uint16_t src0_raw = dword1 & 0x1FF;
      uint16_t src1_raw = (dword1 >> 9) & 0x1FF;
      uint16_t scale_raw = (dword1 >> 18) & 0x1FF;
      uint8_t src0_vgpr = static_cast<uint8_t>(src0_raw & 0xFF);
      uint8_t src1_vgpr = static_cast<uint8_t>(src1_raw & 0xFF);
      uint8_t scale_vgpr = static_cast<uint8_t>(scale_raw & 0xFF);
      uint8_t vtmp = 255;

      NopSled* sled = findNearestSled(di.offset);
      if (sled && sled->write_pos + 44 <= sled->end) {
        uint64_t tp = sled->write_pos;

        // VOP2 helpers:
        // v_mul_f32_e32 = opcode 0x04: (0x04<<25)|(vdst<<17)|(vsrc1<<9)|src0
        // v_min_u32_e32 = opcode 0x0E: (0x0E<<25)|(vdst<<17)|(vsrc1<<9)|src0
        // VOP1: v_cvt_u32_f32_e32 = opcode 0x07: 0x7E000000|(vdst<<17)|(0x07<<9)|src0
        // 2.0 inline = 0xF4, 7 inline = 0x87
        auto vop2 = [](uint8_t op, uint8_t d, uint8_t s1, uint16_t s0) -> uint32_t {
          return (static_cast<uint32_t>(op) << 25) | (static_cast<uint32_t>(d) << 17) |
                 (static_cast<uint32_t>(s1) << 9) | s0;
        };
        auto vop1_cvt = [](uint8_t d, uint16_t s0) -> uint32_t {
          return 0x7E000000u | (static_cast<uint32_t>(d) << 17) |
                 (0x07u << 9) | s0; // v_cvt_u32_f32 opcode = 0x07
        };

        // 1. v_mul_f32 vDst, vScale, vSrc0 (src0=vScale, vsrc1=vSrc0)
        uint32_t i1 = vop2(0x04, vdst, src0_vgpr, 256u + scale_vgpr);
        std::memcpy(text + tp, &i1, 4);
        // 2. v_mul_f32 vDst, 2.0, vDst (src0=2.0(0xF4), vsrc1=vDst)
        uint32_t i2 = vop2(0x04, vdst, vdst, 0xF4u);
        std::memcpy(text + tp + 4, &i2, 4);
        // 3. v_cvt_u32_f32 vDst, vDst
        uint32_t i3 = vop1_cvt(vdst, 256u + vdst);
        std::memcpy(text + tp + 8, &i3, 4);
        // 4. v_min_u32 vDst, 7, vDst (src0=7(0x87), vsrc1=vDst)
        uint32_t i4 = vop2(0x0E, vdst, vdst, 0x87u);
        std::memcpy(text + tp + 12, &i4, 4);
        // 5. v_mul_f32 v255, vScale, vSrc1
        uint32_t i5 = vop2(0x04, vtmp, src1_vgpr, 256u + scale_vgpr);
        std::memcpy(text + tp + 16, &i5, 4);
        // 6. v_mul_f32 v255, 2.0, v255
        uint32_t i6 = vop2(0x04, vtmp, vtmp, 0xF4u);
        std::memcpy(text + tp + 20, &i6, 4);
        // 7. v_cvt_u32_f32 v255, v255
        uint32_t i7 = vop1_cvt(vtmp, 256u + vtmp);
        std::memcpy(text + tp + 24, &i7, 4);
        // 8. v_min_u32 v255, 7, v255
        uint32_t i8 = vop2(0x0E, vtmp, vtmp, 0x87u);
        std::memcpy(text + tp + 28, &i8, 4);
        // 9. v_lshl_or_b32 vDst, v255, 4, vDst (VOP3)
        uint32_t i9_dw0 = 0xD2000000u | static_cast<uint32_t>(vdst);
        uint32_t i9_dw1 = (256u + vtmp) | (0x84u << 9) | // inline 4
                          ((256u + vdst) << 18);
        std::memcpy(text + tp + 32, &i9_dw0, 4);
        std::memcpy(text + tp + 36, &i9_dw1, 4);
        // 10. s_branch back
        uint8_t br[4];
        if (EncodeSBranch(tp + 40, di.offset + 8, br)) {
          std::memcpy(text + tp + 40, br, 4);

          uint8_t br_fwd[4];
          if (EncodeSBranch(di.offset, tp, br_fwd)) {
            std::memcpy(text + di.offset, br_fwd, 4);
            uint8_t nop[4];
            EncodeSNop(nop);
            std::memcpy(text + di.offset + 4, nop, 4);
            sled->write_pos += 44;
            di.mnemonic = "<replaced>";
            ++gfx950_only_replaced;
            continue;
          }
        }
      }
      // Fallback: constant FP4
      uint32_t mov_word = 0x7E000291u | (static_cast<uint32_t>(vdst) << 17);
      std::memcpy(text + di.offset, &mov_word, 4);
      uint8_t nop[4];
      EncodeSNop(nop);
      std::memcpy(text + di.offset + 4, nop, 4);
    } else if (is_bitop3) {
      // v_bitop3_b16 with bitop3:0xEC = (a ? (b|c) : (b&c)).
      // All instances in AITER use 0xEC. When c (src2) is mostly 1s,
      // this approximates v_or_b32 vDst, vSrc0, vSrc1.
      // Emulate with v_or_b32_e32 (VOP2 opcode 0x14):
      // [31:25]=0x14 [24:17]=vdst [16:9]=vsrc1 [8:0]=src0
      uint16_t src0_raw = dword1 & 0x1FF;
      uint16_t src1_raw = (dword1 >> 9) & 0x1FF;
      uint8_t src0_vgpr = static_cast<uint8_t>(src0_raw & 0xFF);
      uint8_t src1_vgpr = static_cast<uint8_t>(src1_raw & 0xFF);
      uint32_t or_word = (0x14u << 25) |
                         (static_cast<uint32_t>(vdst) << 17) |
                         (static_cast<uint32_t>(src1_vgpr) << 9) |
                         (256u + static_cast<uint32_t>(src0_vgpr));
      std::memcpy(text + di.offset, &or_word, 4);
      uint8_t nop[4];
      EncodeSNop(nop);
      std::memcpy(text + di.offset + 4, nop, 4);
    } else {
      // MFMA or no space for trampoline: NOP out
      uint8_t nop[4];
      EncodeSNop(nop);
      std::memcpy(text + di.offset, nop, 4);
      std::memcpy(text + di.offset + 4, nop, 4);
    }

    di.mnemonic = "<replaced>";
    ++gfx950_only_replaced;
  }

  if (gfx950_only_replaced > 0) {
    uint32_t kept = static_cast<uint32_t>(decoded.size()) - gfx950_only_replaced;
    result.rules_matched = kept;
    std::cerr << "hotswap: retarget: " << kept
              << " instructions kept (identical encoding), "
              << gfx950_only_replaced << " replaced with trampolines ("
              << src_state.cpu << " -> " << tgt_cpu << ")\n";
    return result;
  }

  // Build a single assembly string from all decoded instructions, then
  // assemble in one pass for the target ISA. This avoids creating/destroying
  // hundreds of MC pipeline objects which corrupts LLVM internal state.

  uint32_t retargeted = 0;
  uint32_t failed = 0;

  // Step 1: Print all instructions to assembly text, tracking offsets
  struct AsmEntry {
    size_t decoded_idx;
    std::string asm_line;
  };
  std::vector<AsmEntry> entries;
  std::string full_asm;

  // Add .text directive for the assembler
  full_asm += ".text\n";

  for (size_t i = 0; i < decoded.size(); ++i) {
    auto& di = decoded[i];
    if (di.mnemonic == "<unknown>") continue;

    std::string asm_text;
    if (src_state.printer) {
      llvm::raw_string_ostream rso(asm_text);
      src_state.printer->printInst(&di.inst, 0, "", *src_state.STI, rso);
      rso.flush();
    }

    // Trim leading whitespace
    size_t start = asm_text.find_first_not_of(" \t");
    if (start != std::string::npos && start > 0)
      asm_text = asm_text.substr(start);
    if (asm_text.empty()) continue;

    // Strip trailing comments
    size_t comment = asm_text.find("//");
    if (comment != std::string::npos) {
      asm_text = asm_text.substr(0, comment);
      // Trim trailing whitespace after removing comment
      size_t end = asm_text.find_last_not_of(" \t");
      if (end != std::string::npos)
        asm_text = asm_text.substr(0, end + 1);
    }

    if (asm_text.empty()) continue;

    entries.push_back({i, asm_text});
    full_asm += asm_text + "\n";
  }

  if (entries.empty()) {
    if (elf_info.text_size > 0) {
      std::cerr << "hotswap: retarget: no decodable instructions in "
                << elf_info.text_size << " bytes of .text\n";
    }
    return result;
  }

  std::cerr << "hotswap: retarget: assembling " << entries.size()
            << " instructions for " << tgt_cpu << "...\n";

  // Step 2: Assemble the entire text in a single pass for the target ISA.
  // Use a persistent static MCContext to avoid LLVM backend crashes from
  // creating/destroying MC contexts repeatedly. The AMDGPU backend has
  // global state that doesn't survive multiple context lifecycles.
  struct TargetAssembler {
    std::unique_ptr<llvm::MCRegisterInfo> MRI;
    std::unique_ptr<const llvm::MCAsmInfo> MAI;
    std::unique_ptr<llvm::MCInstrInfo> MCII;
    std::unique_ptr<llvm::MCSubtargetInfo> STI;
    std::unique_ptr<llvm::MCContext> Ctx;
    const llvm::Target* target = nullptr;
    bool valid = false;
  };
  static std::mutex s_asm_mutex;
  static std::map<std::string, TargetAssembler> s_assemblers;

  std::lock_guard<std::mutex> asm_lock(s_asm_mutex);
  auto& ta = s_assemblers[tgt_cpu];
  if (!ta.valid) {
    llvm::Triple init_triple("amdgcn-amd-amdhsa");
    llvm::MCTargetOptions init_opts;
    ta.target = src_state.target;
    ta.MRI.reset(ta.target->createMCRegInfo(init_triple));
#if LLVM_VERSION_MAJOR > 9
    ta.MAI.reset(ta.target->createMCAsmInfo(*ta.MRI, init_triple, init_opts));
#else
    ta.MAI.reset(ta.target->createMCAsmInfo(*ta.MRI, "amdgcn-amd-amdhsa"));
#endif
    ta.MCII.reset(ta.target->createMCInstrInfo());
    ta.STI.reset(ta.target->createMCSubtargetInfo(init_triple, tgt_cpu, ""));
    if (ta.MRI && ta.MAI && ta.MCII && ta.STI) {
#if LLVM_VERSION_MAJOR > 12
      ta.Ctx = std::make_unique<llvm::MCContext>(
          init_triple, ta.MAI.get(), ta.MRI.get(), ta.STI.get());
#else
      auto MOFI = std::make_unique<llvm::MCObjectFileInfo>();
      ta.Ctx = std::make_unique<llvm::MCContext>(
          ta.MAI.get(), ta.MRI.get(), MOFI.get());
      MOFI->InitMCObjectFileInfo(init_triple, true, *ta.Ctx);
#endif
      ta.valid = true;
    }
  }

  if (!ta.valid) {
    std::cerr << "hotswap: retarget: failed to create target assembler\n";
    result.status = HSA_STATUS_ERROR;
    return result;
  }

  // Track how many times we've assembled. After the first successful
  // assembly, the LLVM AMDGPU backend's global state may be in a
  // fragile condition. Limit retarget attempts.
  static int s_retarget_count = 0;
  if (s_retarget_count > 0) {
    // LLVM's AMDGPU MC backend has global state that doesn't survive
    // multiple MCContext lifecycles. Skip subsequent retargets and
    // rely on the first code object being the user's kernel.
    std::cerr << "hotswap: retarget: skipping subsequent code object ("
              << entries.size() << " instructions) — LLVM MC limitation\n";
    return result;
  }

  // Reset MCContext state for the assembly
  ta.Ctx->reset();

  // Attach an inline SourceMgr to the MCContext so the MC layer can
  // format diagnostics from instruction-emission errors (unsupported
  // encodings, out-of-range operands, etc.) without hitting the
  // `llvm_unreachable("Either SourceMgr should be available")` abort
  // in `MCContext::reportCommon` (llvm/lib/MC/MCContext.cpp).  The
  // parser below is given its own `src_mgr` directly, but that's a
  // separate SourceMgr — the MCContext's own SrcMgr stays `nullptr`
  // after construction (the ctor defaults the `SourceMgr *Mgr`
  // parameter to `nullptr`), so any error reported via
  // `MCContext::reportError` / `reportWarning` with a valid `SMLoc`
  // would trigger the abort.  `initInlineSourceManager` is idempotent
  // (guarded by `if (!InlineSrcMgr)`) but `MCContext::reset` clears
  // the inline SourceMgr, so we need to re-init after every reset.
  //
  // Manifested empirically as `SIG6 ... Either SourceMgr should be
  // available UNREACHABLE executed at MCContext.cpp:1093` on every
  // Triton kernel in `compare_correctness --lane=legacy`.  The
  // rule-based rewrite path emits N `waitcnt` patches + M
  // target-unsupported instructions, and those unsupported opcodes
  // trigger MC's error-reporting path which then aborts.  With the
  // inline SourceMgr attached, the errors are formatted and routed
  // through the default diag handler; `parser->Run` returns
  // `asm_failed = true`; the caller's existing `data.size() < 64`
  // fallback (~40 lines below) surfaces a clean `HSA_STATUS_ERROR`
  // instead of a process abort, letting the test harness see an
  // EXIT-based failure rather than SIG6.
  ta.Ctx->initInlineSourceManager();

  llvm::Triple tgt_triple("amdgcn-amd-amdhsa");
  llvm::MCTargetOptions mc_opts;

  // Use the cached target assembler components
  llvm::MCContext& tgt_Ctx = *ta.Ctx;

  llvm::StringRef asm_ref(full_asm);
  auto buf = llvm::MemoryBuffer::getMemBuffer(asm_ref, "", false);
  llvm::SourceMgr src_mgr;
  src_mgr.AddNewSourceBuffer(std::move(buf), llvm::SMLoc());

  std::string data;
  auto data_stream = std::make_unique<llvm::raw_string_ostream>(data);
  auto bos = std::make_unique<llvm::buffer_ostream>(*data_stream);

#if LLVM_VERSION_MAJOR > 14
  llvm::MCCodeEmitter* ce =
      ta.target->createMCCodeEmitter(*ta.MCII, tgt_Ctx);
#else
  llvm::MCCodeEmitter* ce =
      ta.target->createMCCodeEmitter(*ta.MCII, *ta.MRI, tgt_Ctx);
#endif
  llvm::MCAsmBackend* mab =
      ta.target->createMCAsmBackend(*ta.STI, *ta.MRI, mc_opts);

  if (!ce || !mab) {
    std::cerr << "hotswap: retarget: failed to create code emitter/backend\n";
    result.status = HSA_STATUS_ERROR;
    return result;
  }

#if LLVM_VERSION_MAJOR > 20
  auto streamer = std::unique_ptr<llvm::MCStreamer>(
      ta.target->createMCObjectStreamer(
          tgt_triple, tgt_Ctx,
          std::unique_ptr<llvm::MCAsmBackend>(mab),
          mab->createObjectWriter(*bos),
          std::unique_ptr<llvm::MCCodeEmitter>(ce), *ta.STI));
#else
  auto streamer = std::unique_ptr<llvm::MCStreamer>(
      ta.target->createMCObjectStreamer(
          tgt_triple, tgt_Ctx,
          std::unique_ptr<llvm::MCAsmBackend>(mab),
          mab->createObjectWriter(*bos),
          std::unique_ptr<llvm::MCCodeEmitter>(ce), *ta.STI,
          mc_opts.MCRelaxAll, mc_opts.MCIncrementalLinkerCompatible, false));
#endif

  if (!streamer) {
    std::cerr << "hotswap: retarget: failed to create MC streamer\n";
    result.status = HSA_STATUS_ERROR;
    return result;
  }

  auto parser = std::unique_ptr<llvm::MCAsmParser>(
      llvm::createMCAsmParser(src_mgr, tgt_Ctx, *streamer, *ta.MAI));
  auto tap = std::unique_ptr<llvm::MCTargetAsmParser>(
      ta.target->createMCAsmParser(*ta.STI, *parser, *ta.MCII, mc_opts));
  if (!tap) {
    std::cerr << "hotswap: retarget: failed to create target asm parser\n";
    result.status = HSA_STATUS_ERROR;
    return result;
  }
  parser->setTargetParser(*tap);

  bool asm_failed = parser->Run(true);
  // Destroy parser and streamer before flushing to avoid use-after-free
  tap.reset();
  parser.reset();
  streamer.reset();
  bos.reset();
  data_stream->flush();

  if (asm_failed || data.size() < 64) {
    std::cerr << "hotswap: retarget: assembly failed for target "
              << tgt_cpu << " (some instructions may not exist)\n";
    // Even if assembly failed, some instructions may have been emitted.
    // We'll try to use what we got.
    if (data.size() < 64) {
      result.status = HSA_STATUS_ERROR;
      return result;
    }
  }

  // Step 3: Extract .text from the assembled ELF
  const uint8_t* asm_elf = reinterpret_cast<const uint8_t*>(data.data());
  ElfInfo asm_info;
  if (!ParseElfInfo(asm_elf, data.size(), asm_info)) {
    std::cerr << "hotswap: retarget: failed to parse assembled ELF\n";
    result.status = HSA_STATUS_ERROR;
    return result;
  }

  // Step 4: Disassemble the target .text to get per-instruction boundaries
  // Reuse the source disassembler — instruction boundaries are the same
  // for compatible instructions (same mnemonic = same size on both ISAs
  // for the gfx9 family). This avoids creating a second LLVMState which
  // can crash due to LLVM global state conflicts.
  const uint8_t* tgt_text = asm_elf + asm_info.text_offset;
  std::vector<InternalDecodedInst> tgt_decoded;
  DecodeTextSection(tgt_text, asm_info.text_size, src_state, tgt_decoded);

  // Step 5: Match source and target instructions and patch in-place
  // The assembler should produce the same number of instructions in the
  // same order (unless some expanded or were removed).
  size_t tgt_idx = 0;
  for (auto& entry : entries) {
    if (tgt_idx >= tgt_decoded.size()) break;

    auto& src_di = decoded[entry.decoded_idx];
    auto& tgt_di = tgt_decoded[tgt_idx];

    if (tgt_di.size == src_di.size) {
      // Same size — patch in-place
      std::memcpy(text + src_di.offset, tgt_text + tgt_di.offset, src_di.size);
      ++retargeted;
    } else if (tgt_di.size < src_di.size) {
      // Target is smaller — patch + NOP pad
      std::memcpy(text + src_di.offset, tgt_text + tgt_di.offset, tgt_di.size);
      uint32_t remaining = src_di.size - tgt_di.size;
      uint64_t pad = src_di.offset + tgt_di.size;
      while (remaining >= 4) {
        uint8_t nop[4];
        EncodeSNop(nop);
        std::memcpy(text + pad, nop, 4);
        pad += 4;
        remaining -= 4;
      }
      ++retargeted;
    } else {
      // Target is larger — can't fit, skip this instruction
      ++failed;
    }
    ++tgt_idx;
  }

  if (ShouldDump()) {
    // Dump the retargeted .text using the source disassembler
    // (some instructions may decode differently but the bytes are correct)
    std::vector<InternalDecodedInst> decoded_after;
    DecodeTextSection(text, elf_info.text_size, src_state, decoded_after);
    DumpInstructions("RETARGET AFTER", decoded_after, text);
  }

  ++s_retarget_count;
  result.rules_matched = retargeted;
  if (failed > 0) {
    std::cerr << "hotswap: retarget: " << retargeted << " instructions retargeted, "
              << failed << " failed (" << src_state.cpu << " -> "
              << tgt_cpu << ")\n";
  } else {
    std::cerr << "hotswap: retarget: " << retargeted << " instructions retargeted ("
              << src_state.cpu << " -> " << tgt_cpu << ")\n";
  }

  return result;
}

RewriteResult RewriteCodeObject(void* elf_data, size_t elf_size,
                                const std::string& isa_name) {
  void* out_data = nullptr;
  size_t out_size = 0;
  auto result = RewriteCodeObjectGrow(elf_data, elf_size, &out_data, &out_size,
                                      isa_name);

  // If the buffer grew, we can't handle it in the in-place API
  if (out_data && out_data != elf_data) {
    std::cerr << "hotswap: RewriteCodeObject cannot grow buffer; use "
                 "RewriteCodeObjectGrow instead\n";
    std::free(out_data);
    result.status = HSA_STATUS_ERROR;
  }

  return result;
}

RewriteResult RewriteCodeObjectGrow(const void* elf_data, size_t elf_size,
                                    void** out_data, size_t* out_size,
                                    const std::string& isa_name) {
  RewriteResult result = {HSA_STATUS_SUCCESS, 0, 0};
  *out_data = const_cast<void*>(elf_data);
  *out_size = elf_size;

  const RulesFile* rules = GetCachedRules();
  if (!rules || rules->rules.empty()) return result;

  // Check target match
  if (!rules->target.empty() && rules->target != isa_name) {
    // Target doesn't match — silently skip
    return result;
  }

  // Parse ELF to find .text
  ElfInfo elf_info;
  uint8_t* elf = static_cast<uint8_t*>(const_cast<void*>(elf_data));
  if (!ParseElfInfo(elf, elf_size, elf_info)) {
    // Not a valid ELF or no .text — skip silently
    return result;
  }

  if (elf_info.text_size == 0) return result;

  // Initialize LLVM MC
  LLVMState llvm_state = InitLLVM(isa_name);
  if (!llvm_state.valid) {
    std::cerr << "hotswap: LLVM MC initialization failed\n";
    result.status = HSA_STATUS_ERROR;
    return result;
  }

  // Get a mutable pointer to .text within the ELF
  uint8_t* text = elf + elf_info.text_offset;

  // Decode all instructions in .text
  std::vector<InternalDecodedInst> decoded;
  if (!DecodeTextSection(text, elf_info.text_size, llvm_state, decoded)) {
    std::cerr << "hotswap: instruction decode failed\n";
    result.status = HSA_STATUS_ERROR;
    return result;
  }

  if (ShouldDump()) {
    DumpInstructions("BEFORE", decoded, text);
  }

  // Collect trampolines needed for size-changing rewrites
  std::vector<Trampoline> trampolines;

  // Apply rules
  for (auto& inst : decoded) {
    for (auto& rule : rules->rules) {
      if (!MatchRule(rule, inst, elf_info)) continue;

      bool applied = false;

      switch (rule.action) {
        case ReplaceAction::MnemonicSwap:
          applied = ApplyMnemonicSwap(rule, inst, text, llvm_state);
          break;

        case ReplaceAction::ByteReplace:
          applied = ApplyByteReplace(rule, inst, text, elf_info.text_size);
          break;

        case ReplaceAction::AsmReplace: {
          // Check if this is same-size (single instruction replacement)
          // or needs a trampoline (multi-instruction or different size)
          uint64_t tramp_offset = elf_info.text_size;
          for (auto& t : trampolines) {
            tramp_offset += t.bytes.size();
          }

          Trampoline tramp = BuildTrampoline(
              rule.replace_asm, inst.offset, inst.size, tramp_offset,
              llvm_state.cpu, llvm_state.STI.get(), llvm_state.MCII.get(),
              llvm_state.MRI.get(), llvm_state.MAI.get(),
              llvm_state.Ctx.get(), llvm_state.CE);

          if (tramp.bytes.empty()) {
            std::cerr << "hotswap: trampoline build failed for rule '"
                      << rule.name << "'\n";
            break;
          }

          // Replace original instruction with s_branch to trampoline
          // Pad remaining bytes with s_nop
          uint8_t branch_bytes[4];
          if (!EncodeSBranch(inst.offset, tramp_offset, branch_bytes)) {
            std::cerr << "hotswap: branch encode failed for rule '"
                      << rule.name << "'\n";
            break;
          }

          std::memcpy(text + inst.offset, branch_bytes, 4);
          // NOP-fill the rest of the original instruction
          for (uint32_t pad = 4; pad < inst.size; pad += 4) {
            uint8_t nop[4];
            EncodeSNop(nop);
            std::memcpy(text + inst.offset + pad, nop, 4);
          }

          trampolines.push_back(std::move(tramp));
          applied = true;
          break;
        }
      }

      if (applied) {
        ++result.rules_matched;

        // Update kernel descriptor if extra registers requested
        if (rule.extra_vgprs > 0 || rule.extra_sgprs > 0) {
          std::string kernel = FindKernelAtOffset(elf_info, inst.offset);
          if (!kernel.empty()) {
            UpdateKernelDescriptor(elf, elf_size, elf_info, kernel,
                                  rule.extra_vgprs, rule.extra_sgprs);
          }
        }

        break; // Only apply first matching rule per instruction
      }
    }
  }

  // If we have trampolines, we need to grow the ELF
  if (!trampolines.empty()) {
    // Calculate total trampoline size
    size_t tramp_total = 0;
    for (auto& t : trampolines) tramp_total += t.bytes.size();

    // Allocate new buffer: original ELF + space for trampolines in .text
    // We append trampoline bytes after the current .text content and
    // update the .text section header size.

    size_t new_elf_size = elf_size + tramp_total;
    uint8_t* new_elf = static_cast<uint8_t*>(std::malloc(new_elf_size));
    if (!new_elf) {
      result.status = HSA_STATUS_ERROR_OUT_OF_RESOURCES;
      return result;
    }

    // Copy everything up to end of .text
    uint64_t text_end = elf_info.text_offset + elf_info.text_size;
    std::memcpy(new_elf, elf, text_end);

    // Append trampolines
    uint64_t tramp_pos = text_end;
    for (auto& t : trampolines) {
      std::memcpy(new_elf + tramp_pos, t.bytes.data(), t.bytes.size());
      tramp_pos += t.bytes.size();
    }

    // Copy remainder of ELF (after .text)
    if (text_end < elf_size) {
      std::memcpy(new_elf + tramp_pos, elf + text_end, elf_size - text_end);
    }

    // Update .text section size in section header
    // Find the .text section header and patch its size field
    uint64_t e_shoff;
    uint16_t e_shentsize;
    std::memcpy(&e_shoff, new_elf + 40, 8);
    std::memcpy(&e_shentsize, new_elf + 58, 2);

    // The section headers may have moved if they were after .text
    if (e_shoff >= text_end) {
      // Section headers were after .text, they shifted by tramp_total
      uint64_t new_shoff = e_shoff + tramp_total;
      std::memcpy(new_elf + 40, &new_shoff, 8);
      e_shoff = new_shoff;
    }

    // Patch .text section header: update sh_size
    uint16_t e_shnum;
    std::memcpy(&e_shnum, new_elf + 60, 2);

    for (uint16_t i = 0; i < e_shnum; ++i) {
      uint8_t* sh = new_elf + e_shoff + i * e_shentsize;
      uint64_t sh_offset;
      std::memcpy(&sh_offset, sh + 24, 8);

      if (sh_offset == elf_info.text_offset) {
        // This is the .text section header — update size
        uint64_t new_text_size = elf_info.text_size + tramp_total;
        std::memcpy(sh + 32, &new_text_size, 8);
        break;
      }

      // Also fix offsets of sections that were after .text
      if (sh_offset > elf_info.text_offset) {
        uint64_t new_offset = sh_offset + tramp_total;
        std::memcpy(sh + 24, &new_offset, 8);
      }
    }

    // Also update program headers if any segment contains .text
    uint64_t e_phoff;
    uint16_t e_phentsize, e_phnum;
    std::memcpy(&e_phoff, new_elf + 32, 8);
    std::memcpy(&e_phentsize, new_elf + 54, 2);
    std::memcpy(&e_phnum, new_elf + 56, 2);

    for (uint16_t i = 0; i < e_phnum; ++i) {
      uint8_t* ph = new_elf + e_phoff + i * e_phentsize;
      // ELF64 Phdr: p_offset at +8, p_filesz at +32, p_memsz at +40
      uint64_t p_offset, p_filesz, p_memsz;
      std::memcpy(&p_offset, ph + 8, 8);
      std::memcpy(&p_filesz, ph + 32, 8);
      std::memcpy(&p_memsz, ph + 40, 8);

      // Check if this segment contains .text
      if (p_offset <= elf_info.text_offset &&
          p_offset + p_filesz >= text_end) {
        // Grow this segment to include trampolines
        p_filesz += tramp_total;
        p_memsz += tramp_total;
        std::memcpy(ph + 32, &p_filesz, 8);
        std::memcpy(ph + 40, &p_memsz, 8);
      } else if (p_offset > elf_info.text_offset) {
        // Shift segments after .text
        p_offset += tramp_total;
        std::memcpy(ph + 8, &p_offset, 8);
      }
    }

    *out_data = new_elf;
    *out_size = new_elf_size;
    result.trampolines_added = static_cast<uint32_t>(trampolines.size());
  }

  // Re-decode and dump after patching
  if (ShouldDump() && result.rules_matched > 0) {
    uint8_t* final_text =
        (*out_data == elf_data) ? text : (static_cast<uint8_t*>(*out_data) +
                                          elf_info.text_offset);
    uint64_t final_text_size = elf_info.text_size;
    if (!trampolines.empty()) {
      size_t tramp_total = 0;
      for (auto& t : trampolines) tramp_total += t.bytes.size();
      final_text_size += tramp_total;
    }

    std::vector<InternalDecodedInst> decoded_after;
    DecodeTextSection(final_text, final_text_size, llvm_state, decoded_after);
    DumpInstructions("AFTER", decoded_after, final_text);
  }

  return result;
}

} // namespace hotswap
} // namespace rocr

// C-linkage wrapper for dlsym access from HIP/CLR
extern "C" __attribute__((visibility("default")))
int rocr_hotswap_retarget(void* elf_data, size_t elf_size,
                          const char* source_isa, const char* target_isa) {
  auto result = rocr::hotswap::RetargetCodeObject(
      elf_data, elf_size, std::string(source_isa), std::string(target_isa));
  return result.rules_matched;
}

// Patch ELF e_flags and NT_AMDGPU_ISA note to match target_isa.
// MSGPACK metadata (NT_AMDGPU_METADATA) is deliberately left untouched so the
// HSA runtime can still detect the original ISA and trigger Salmon.
extern "C" __attribute__((visibility("default")))
int rocr_salmon_patch_elf(void* elf_data, size_t elf_size,
                          const char* target_isa) {
  return rocr::hotswap::PatchElfIsa(elf_data, elf_size,
                                    std::string(target_isa)) ? 0 : -1;
}
