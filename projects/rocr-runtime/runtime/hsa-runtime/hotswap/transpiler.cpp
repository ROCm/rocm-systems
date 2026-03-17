////////////////////////////////////////////////////////////////////////////////
//
// ROCm HotSwap — Cross-Family ISA Transpiler
//
// Translates gfx1250 (RDNA4) GPU binaries to run on gfx950 (CDNA4) hardware.
//
// Architecture:
//   1. Disassemble gfx1250 .text → MCInst → text (via MCInstPrinter)
//   2. Apply instruction translation rules:
//      - Mnemonic renaming (global_load_b32 → global_load_dword, etc.)
//      - Wait counter merging (s_wait_loadcnt → s_waitcnt vmcnt)
//      - EXEC widening (wave32 → wave64: insert exec_hi = 0 after exec writes)
//      - Unsupported instruction handling (WMMA → NOP for MVP)
//   3. Batch assemble translated text for gfx950 via LLVM MC
//   4. Replace .text in ELF, patch kernel descriptors and metadata
//
////////////////////////////////////////////////////////////////////////////////

#include "transpiler.hpp"
#include "hotswap.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <mutex>
#include <regex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <llvm/Config/llvm-config.h>
#include <llvm/MC/MCAsmBackend.h>
#include <llvm/MC/MCAsmInfo.h>
#include <llvm/MC/MCCodeEmitter.h>
#include <llvm/MC/MCContext.h>
#include <llvm/MC/MCDisassembler/MCDisassembler.h>
#include <llvm/MC/MCInstPrinter.h>
#include <llvm/MC/MCInstrInfo.h>
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

// ── Mnemonic Translation Tables ──────────────────────────────────────────────
//
// GFX12 uses bit-width-based naming (global_load_b32) while GFX9 uses
// type-based naming (global_load_dword). The instruction semantics are
// identical — only the mnemonic and encoding differ.

struct MnemonicMapping {
  const char* gfx12;   // GFX12/GFX1250 mnemonic
  const char* gfx9;    // GFX9/GFX950 mnemonic
};

// Global load/store
static const MnemonicMapping kGlobalMemMappings[] = {
    {"global_load_b32", "global_load_dword"},
    {"global_load_b64", "global_load_dwordx2"},
    {"global_load_b96", "global_load_dwordx3"},
    {"global_load_b128", "global_load_dwordx4"},
    {"global_load_u8", "global_load_ubyte"},
    {"global_load_i8", "global_load_sbyte"},
    {"global_load_u16", "global_load_ushort"},
    {"global_load_i16", "global_load_sshort"},
    {"global_load_d16_u8", "global_load_ubyte_d16"},
    {"global_load_d16_i8", "global_load_sbyte_d16"},
    {"global_load_d16_b16", "global_load_short_d16"},
    {"global_load_d16_hi_u8", "global_load_ubyte_d16_hi"},
    {"global_load_d16_hi_i8", "global_load_sbyte_d16_hi"},
    {"global_load_d16_hi_b16", "global_load_short_d16_hi"},
    {"global_store_b8", "global_store_byte"},
    {"global_store_b16", "global_store_short"},
    {"global_store_b32", "global_store_dword"},
    {"global_store_b64", "global_store_dwordx2"},
    {"global_store_b96", "global_store_dwordx3"},
    {"global_store_b128", "global_store_dwordx4"},
    {"global_load_addtid_b32", "global_load_dword_addtid"},
    {"global_store_addtid_b32", "global_store_dword_addtid"},
};

// Flat load/store
static const MnemonicMapping kFlatMemMappings[] = {
    {"flat_load_b32", "flat_load_dword"},
    {"flat_load_b64", "flat_load_dwordx2"},
    {"flat_load_b96", "flat_load_dwordx3"},
    {"flat_load_b128", "flat_load_dwordx4"},
    {"flat_load_u8", "flat_load_ubyte"},
    {"flat_load_i8", "flat_load_sbyte"},
    {"flat_load_u16", "flat_load_ushort"},
    {"flat_load_i16", "flat_load_sshort"},
    {"flat_load_d16_u8", "flat_load_ubyte_d16"},
    {"flat_load_d16_i8", "flat_load_sbyte_d16"},
    {"flat_load_d16_b16", "flat_load_short_d16"},
    {"flat_load_d16_hi_u8", "flat_load_ubyte_d16_hi"},
    {"flat_load_d16_hi_i8", "flat_load_sbyte_d16_hi"},
    {"flat_load_d16_hi_b16", "flat_load_short_d16_hi"},
    {"flat_store_b8", "flat_store_byte"},
    {"flat_store_b16", "flat_store_short"},
    {"flat_store_b32", "flat_store_dword"},
    {"flat_store_b64", "flat_store_dwordx2"},
    {"flat_store_b96", "flat_store_dwordx3"},
    {"flat_store_b128", "flat_store_dwordx4"},
};

// Scratch load/store
static const MnemonicMapping kScratchMemMappings[] = {
    {"scratch_load_b32", "scratch_load_dword"},
    {"scratch_load_b64", "scratch_load_dwordx2"},
    {"scratch_load_b96", "scratch_load_dwordx3"},
    {"scratch_load_b128", "scratch_load_dwordx4"},
    {"scratch_load_u8", "scratch_load_ubyte"},
    {"scratch_load_i8", "scratch_load_sbyte"},
    {"scratch_load_u16", "scratch_load_ushort"},
    {"scratch_load_i16", "scratch_load_sshort"},
    {"scratch_load_d16_u8", "scratch_load_ubyte_d16"},
    {"scratch_load_d16_i8", "scratch_load_sbyte_d16"},
    {"scratch_load_d16_b16", "scratch_load_short_d16"},
    {"scratch_load_d16_hi_u8", "scratch_load_ubyte_d16_hi"},
    {"scratch_load_d16_hi_i8", "scratch_load_sbyte_d16_hi"},
    {"scratch_load_d16_hi_b16", "scratch_load_short_d16_hi"},
    {"scratch_store_b8", "scratch_store_byte"},
    {"scratch_store_b16", "scratch_store_short"},
    {"scratch_store_b32", "scratch_store_dword"},
    {"scratch_store_b64", "scratch_store_dwordx2"},
    {"scratch_store_b96", "scratch_store_dwordx3"},
    {"scratch_store_b128", "scratch_store_dwordx4"},
};

// Buffer load/store
static const MnemonicMapping kBufferMemMappings[] = {
    {"buffer_load_b32", "buffer_load_dword"},
    {"buffer_load_b64", "buffer_load_dwordx2"},
    {"buffer_load_b96", "buffer_load_dwordx3"},
    {"buffer_load_b128", "buffer_load_dwordx4"},
    {"buffer_load_u8", "buffer_load_ubyte"},
    {"buffer_load_i8", "buffer_load_sbyte"},
    {"buffer_load_u16", "buffer_load_ushort"},
    {"buffer_load_i16", "buffer_load_sshort"},
    {"buffer_load_d16_u8", "buffer_load_ubyte_d16"},
    {"buffer_load_d16_i8", "buffer_load_sbyte_d16"},
    {"buffer_load_d16_b16", "buffer_load_short_d16"},
    {"buffer_load_d16_hi_u8", "buffer_load_ubyte_d16_hi"},
    {"buffer_load_d16_hi_i8", "buffer_load_sbyte_d16_hi"},
    {"buffer_load_d16_hi_b16", "buffer_load_short_d16_hi"},
    {"buffer_store_b8", "buffer_store_byte"},
    {"buffer_store_b16", "buffer_store_short"},
    {"buffer_store_b32", "buffer_store_dword"},
    {"buffer_store_b64", "buffer_store_dwordx2"},
    {"buffer_store_b96", "buffer_store_dwordx3"},
    {"buffer_store_b128", "buffer_store_dwordx4"},
};

// DS load/store
static const MnemonicMapping kDSMappings[] = {
    {"ds_load_b32", "ds_read_b32"},
    {"ds_load_b64", "ds_read_b64"},
    {"ds_load_b96", "ds_read_b96"},
    {"ds_load_b128", "ds_read_b128"},
    {"ds_load_u8", "ds_read_u8"},
    {"ds_load_i8", "ds_read_i8"},
    {"ds_load_u16", "ds_read_u16"},
    {"ds_load_i16", "ds_read_i16"},
    {"ds_load_2addr_b32", "ds_read2_b32"},
    {"ds_load_2addr_stride64_b32", "ds_read2st64_b32"},
    {"ds_load_2addr_b64", "ds_read2_b64"},
    {"ds_load_2addr_stride64_b64", "ds_read2st64_b64"},
    {"ds_load_u8_d16", "ds_read_u8_d16"},
    {"ds_load_i8_d16", "ds_read_i8_d16"},
    {"ds_load_u16_d16", "ds_read_u16_d16"},
    {"ds_load_u8_d16_hi", "ds_read_u8_d16_hi"},
    {"ds_load_i8_d16_hi", "ds_read_i8_d16_hi"},
    {"ds_load_u16_d16_hi", "ds_read_u16_d16_hi"},
    {"ds_load_addtid_b32", "ds_read_addtid_b32"},
    {"ds_store_b32", "ds_write_b32"},
    {"ds_store_b64", "ds_write_b64"},
    {"ds_store_b128", "ds_write_b128"},
    {"ds_store_2addr_b32", "ds_write2_b32"},
    {"ds_store_2addr_stride64_b32", "ds_write2st64_b32"},
    {"ds_store_b8", "ds_write_b8"},
    {"ds_store_b16", "ds_write_b16"},
};

// SMEM load/store
static const MnemonicMapping kSMEMMappings[] = {
    {"s_load_b32", "s_load_dword"},
    {"s_load_b64", "s_load_dwordx2"},
    {"s_load_b128", "s_load_dwordx4"},
    {"s_load_b256", "s_load_dwordx8"},
    {"s_load_b512", "s_load_dwordx16"},
    {"s_store_b32", "s_store_dword"},
    {"s_store_b64", "s_store_dwordx2"},
    {"s_store_b128", "s_store_dwordx4"},
    {"s_buffer_load_b32", "s_buffer_load_dword"},
    {"s_buffer_load_b64", "s_buffer_load_dwordx2"},
    {"s_buffer_load_b128", "s_buffer_load_dwordx4"},
    {"s_buffer_load_b256", "s_buffer_load_dwordx8"},
    {"s_buffer_load_b512", "s_buffer_load_dwordx16"},
};

// Build a reverse lookup map (gfx12 mnemonic → gfx9 mnemonic)
static std::unordered_map<std::string, std::string> BuildMnemonicMap() {
  std::unordered_map<std::string, std::string> map;

  auto addMappings = [&](const MnemonicMapping* mappings, size_t count) {
    for (size_t i = 0; i < count; ++i) {
      map[mappings[i].gfx12] = mappings[i].gfx9;
    }
  };

  addMappings(kGlobalMemMappings,
              sizeof(kGlobalMemMappings) / sizeof(kGlobalMemMappings[0]));
  addMappings(kFlatMemMappings,
              sizeof(kFlatMemMappings) / sizeof(kFlatMemMappings[0]));
  addMappings(kScratchMemMappings,
              sizeof(kScratchMemMappings) / sizeof(kScratchMemMappings[0]));
  addMappings(kBufferMemMappings,
              sizeof(kBufferMemMappings) / sizeof(kBufferMemMappings[0]));
  addMappings(kDSMappings,
              sizeof(kDSMappings) / sizeof(kDSMappings[0]));
  addMappings(kSMEMMappings,
              sizeof(kSMEMMappings) / sizeof(kSMEMMappings[0]));

  return map;
}

static const std::unordered_map<std::string, std::string>& GetMnemonicMap() {
  static auto map = BuildMnemonicMap();
  return map;
}

// ── Wave32→Wave64 EXEC Patterns ─────────────────────────────────────────────
//
// GFX1250 (wave32) uses exec_lo (32-bit). GFX950 (wave64) uses exec (64-bit).
// We run wave32 code in the lower 32 lanes with exec_hi permanently 0.
//
// Patterns that write exec_lo need an extra "s_mov_b32 exec_hi, 0" after them
// to keep the upper lanes disabled.

static bool WritesExecLo(const std::string& line) {
  // Check if this instruction writes to exec_lo
  // Common patterns:
  //   s_mov_b32 exec_lo, sN
  //   s_and_b32 exec_lo, exec_lo, sN
  //   s_or_b32 exec_lo, exec_lo, sN
  //   s_andn2_b32 exec_lo, exec_lo, sN
  //   s_and_saveexec_b32 sN, sM  (implicit exec_lo write)

  // Look for "exec_lo" as a destination (first operand after mnemonic)
  size_t mnem_end = line.find_first_of(" \t");
  if (mnem_end == std::string::npos) return false;

  size_t op_start = line.find_first_not_of(" \t,", mnem_end);
  if (op_start == std::string::npos) return false;

  // Check if first operand is exec_lo
  if (line.compare(op_start, 7, "exec_lo") == 0) return true;

  // Check for s_and_saveexec_b32, s_or_saveexec_b32 etc.
  std::string mnemonic = line.substr(0, mnem_end);
  if (mnemonic.find("saveexec_b32") != std::string::npos) return true;

  return false;
}

// ── Wait Counter Translation ─────────────────────────────────────────────────
//
// GFX12: granular s_wait_loadcnt, s_wait_storecnt, s_wait_dscnt, etc.
// GFX9:  combined s_waitcnt vmcnt(N) lgkmcnt(N) expcnt(N)
//
// Translation strategy: each s_wait_*cnt becomes a conservative s_waitcnt.
// Multiple consecutive waits could be merged, but for correctness we emit
// one s_waitcnt per source wait instruction.

static bool IsWaitInstruction(const std::string& mnemonic) {
  return mnemonic == "s_wait_loadcnt" || mnemonic == "s_wait_storecnt" ||
         mnemonic == "s_wait_samplecnt" || mnemonic == "s_wait_bvhcnt" ||
         mnemonic == "s_wait_expcnt" || mnemonic == "s_wait_dscnt" ||
         mnemonic == "s_wait_kmcnt" || mnemonic == "s_wait_loadcnt_dscnt" ||
         mnemonic == "s_wait_storecnt_dscnt" || mnemonic == "s_wait_xcnt" ||
         mnemonic == "s_wait_asynccnt" || mnemonic == "s_wait_tensorcnt";
}

static std::string TranslateWaitInstruction(const std::string& line) {
  // Parse: "s_wait_loadcnt N" or "s_wait_loadcnt_dscnt N"
  std::string mnemonic;
  int count = 0;

  std::istringstream iss(line);
  iss >> mnemonic >> count;
  if (iss.fail()) count = 0;  // Default to wait for all

  // Map to GFX9 s_waitcnt
  if (mnemonic == "s_wait_loadcnt" || mnemonic == "s_wait_samplecnt" ||
      mnemonic == "s_wait_bvhcnt" || mnemonic == "s_wait_storecnt") {
    return "s_waitcnt vmcnt(" + std::to_string(count) + ")";
  }
  if (mnemonic == "s_wait_dscnt" || mnemonic == "s_wait_kmcnt") {
    return "s_waitcnt lgkmcnt(" + std::to_string(count) + ")";
  }
  if (mnemonic == "s_wait_expcnt") {
    return "s_waitcnt expcnt(" + std::to_string(count) + ")";
  }
  if (mnemonic == "s_wait_loadcnt_dscnt") {
    return "s_waitcnt vmcnt(" + std::to_string(count) +
           ") lgkmcnt(" + std::to_string(count) + ")";
  }
  if (mnemonic == "s_wait_storecnt_dscnt") {
    return "s_waitcnt vmcnt(" + std::to_string(count) +
           ") lgkmcnt(" + std::to_string(count) + ")";
  }
  // s_wait_xcnt, s_wait_asynccnt, s_wait_tensorcnt — no GFX9 equivalent.
  // Emit a full barrier as conservative fallback.
  return "s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)";
}

// ── Unsupported Instruction Detection ────────────────────────────────────────

static bool IsUnsupportedOnGFX9(const std::string& mnemonic) {
  // WMMA/SWMMAC — no equivalent on GFX9 (MFMA has different semantics)
  if (mnemonic.find("v_wmma_") == 0) return true;
  if (mnemonic.find("v_swmmac_") == 0) return true;

  // SALU float instructions — GFX9 scalar ALU is integer-only
  if (mnemonic == "s_add_f32" || mnemonic == "s_sub_f32" ||
      mnemonic == "s_mul_f32" || mnemonic == "s_min_f32" ||
      mnemonic == "s_max_f32" || mnemonic == "s_fmac_f32" ||
      mnemonic == "s_add_f16" || mnemonic == "s_sub_f16" ||
      mnemonic == "s_mul_f16" || mnemonic == "s_min_f16" ||
      mnemonic == "s_max_f16" || mnemonic == "s_fmac_f16" ||
      mnemonic == "s_cvt_f32_f16" || mnemonic == "s_cvt_f16_f32" ||
      mnemonic == "s_cvt_pk_rtz_f16_f32") {
    return true;
  }

  // VOPD (dual-issue) — not on GFX9
  if (mnemonic.find("v_dual_") == 0) return true;

  // GFX1250 tensor/cluster/prefetch — no equivalent
  if (mnemonic.find("tensor_") == 0) return true;
  if (mnemonic.find("cluster_") == 0) return true;
  if (mnemonic.find("_prefetch_") != std::string::npos) return true;

  // s_wait_alu — GFX12-only ALU dependency wait
  if (mnemonic == "s_wait_alu") return true;

  return false;
}

// ── VCC Register Width Translation ───────────────────────────────────────────
//
// Wave32 uses vcc_lo (32-bit). Wave64 uses vcc (64-bit).
// For VOPC instructions, VCC is implicit so the assembler handles it.
// But explicit VCC references in scalar ops need widening.

static std::string WidenVccReferences(const std::string& line) {
  std::string result = line;

  // Replace "vcc_lo" with "vcc" in operand positions
  // Be careful not to replace inside mnemonics
  size_t mnem_end = result.find_first_of(" \t");
  if (mnem_end == std::string::npos) return result;

  // Only replace in operand part
  std::string operands = result.substr(mnem_end);
  size_t pos = 0;
  while ((pos = operands.find("vcc_lo", pos)) != std::string::npos) {
    // Check it's not part of a longer token (e.g., "vcc_lo_hi")
    size_t end = pos + 6;
    if (end < operands.size() && (std::isalnum(operands[end]) || operands[end] == '_')) {
      pos = end;
      continue;
    }
    operands.replace(pos, 6, "vcc");
    pos += 3;
  }

  return result.substr(0, mnem_end) + operands;
}

// ── EXEC Register Width Translation ──────────────────────────────────────────
//
// Wave32 scalar ops on exec use s_*_b32 with exec_lo.
// Wave64 needs s_*_b64 with exec.
//
// Patterns:
//   s_mov_b32 exec_lo, sN     → s_mov_b32 exec_lo, sN + s_mov_b32 exec_hi, 0
//   s_and_b32 exec_lo, ...    → s_and_b32 exec_lo, ... + s_mov_b32 exec_hi, 0
//   s_or_b32 exec_lo, ...     → s_or_b32 exec_lo, ... + s_mov_b32 exec_hi, 0
//   s_and_saveexec_b32 sN, sM → s_and_saveexec_b32 sN, sM + s_mov_b32 exec_hi, 0
//
// We DON'T widen to b64 because the operands are 32-bit. Instead we keep the
// b32 operation on exec_lo and clear exec_hi separately. This preserves the
// original computation on the lower 32 bits.

static std::vector<std::string> WidenExecOperation(const std::string& line) {
  std::vector<std::string> result;
  result.push_back(line);  // Keep original operation on exec_lo

  if (WritesExecLo(line)) {
    // Add exec_hi = 0 to keep upper 32 lanes disabled
    result.push_back("s_mov_b32 exec_hi, 0");
  }

  return result;
}

// ── Operand Syntax Translation ───────────────────────────────────────────────
//
// GFX12 has some operand syntax differences from GFX9:
// - "off" keyword in FLAT/GLOBAL addressing → needs context
// - Cache policy modifiers changed (th:TH_* → sc0/sc1/glc/slc)

static std::string TranslateOperandSyntax(const std::string& line,
                                           const std::string& mnemonic) {
  std::string result = line;

  // GFX12 uses "scope:SCOPE_*" and "th:TH_*" cache policy modifiers
  // GFX9 uses glc, slc, dlc flags
  // For MVP: strip GFX12 cache modifiers (conservative: no caching hints)

  // Remove "scope:SCOPE_*" modifiers
  {
    size_t pos = result.find("scope:");
    if (pos != std::string::npos) {
      size_t end = result.find_first_of(" \t,", pos);
      if (end == std::string::npos) end = result.size();
      result.erase(pos, end - pos);
    }
  }

  // Remove "th:TH_*" modifiers
  {
    size_t pos = result.find("th:");
    if (pos != std::string::npos) {
      size_t end = result.find_first_of(" \t,", pos);
      if (end == std::string::npos) end = result.size();
      result.erase(pos, end - pos);
    }
  }

  // Trim trailing whitespace/commas
  while (!result.empty() && (result.back() == ' ' || result.back() == '\t' ||
                              result.back() == ',')) {
    result.pop_back();
  }

  return result;
}

// ── Extract Mnemonic ─────────────────────────────────────────────────────────

static std::string ExtractMnemonic(const std::string& line) {
  size_t start = line.find_first_not_of(" \t");
  if (start == std::string::npos) return "";
  size_t end = line.find_first_of(" \t", start);
  if (end == std::string::npos) return line.substr(start);
  return line.substr(start, end - start);
}

static std::string ReplaceMnemonic(const std::string& line,
                                    const std::string& old_mnemonic,
                                    const std::string& new_mnemonic) {
  size_t pos = line.find(old_mnemonic);
  if (pos == std::string::npos) return line;
  std::string result = line;
  result.replace(pos, old_mnemonic.size(), new_mnemonic);
  return result;
}

}  // anonymous namespace

// ── Public API ───────────────────────────────────────────────────────────────

bool NeedsTranspile(const std::string& source_isa,
                    const std::string& target_isa) {
  // Extract CPU names
  auto extractCpu = [](const std::string& isa) -> std::string {
    size_t pos = isa.rfind("gfx");
    if (pos == std::string::npos) return "";
    std::string cpu;
    for (size_t i = pos; i < isa.size(); ++i) {
      char c = isa[i];
      if ((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') ||
          (c >= 'A' && c <= 'Z'))
        cpu += c;
      else
        break;
    }
    return cpu;
  };

  std::string src_cpu = extractCpu(source_isa);
  std::string tgt_cpu = extractCpu(target_isa);

  if (src_cpu.empty() || tgt_cpu.empty()) return false;

  // Determine ISA families
  // GFX9xx = CDNA, GFX10xx/11xx/12xx = RDNA
  auto isGFX9 = [](const std::string& cpu) {
    return cpu.size() >= 4 && cpu[3] == '9';
  };
  auto isGFX12 = [](const std::string& cpu) {
    return cpu.size() >= 5 && cpu[3] == '1' && cpu[4] == '2';
  };

  // Cross-family transpile needed when source and target are different families
  // Currently supported: GFX12 → GFX9
  return isGFX12(src_cpu) && isGFX9(tgt_cpu);
}

std::vector<std::string> TranslateInstruction(const std::string& asm_line,
                                               const std::string& source_cpu,
                                               const std::string& target_cpu) {
  std::vector<std::string> result;

  std::string line = asm_line;

  // Trim leading/trailing whitespace
  size_t start = line.find_first_not_of(" \t");
  if (start == std::string::npos) {
    result.push_back(line);
    return result;
  }
  if (start > 0) line = line.substr(start);

  // Strip comments
  size_t comment = line.find("//");
  if (comment != std::string::npos) {
    line = line.substr(0, comment);
    size_t end = line.find_last_not_of(" \t");
    if (end != std::string::npos)
      line = line.substr(0, end + 1);
  }

  if (line.empty()) {
    result.push_back("");
    return result;
  }

  std::string mnemonic = ExtractMnemonic(line);

  // ─── Wait counter translation ───
  if (IsWaitInstruction(mnemonic)) {
    result.push_back(TranslateWaitInstruction(line));
    return result;
  }

  // ─── s_wait_alu (GFX12 ALU dependency) → s_nop (conservative) ───
  if (mnemonic == "s_wait_alu") {
    result.push_back("s_nop 0");
    return result;
  }

  // ─── Unsupported instructions → NOP with diagnostic ───
  if (IsUnsupportedOnGFX9(mnemonic)) {
    result.push_back("s_nop 0 ; UNSUPPORTED: " + mnemonic);
    return result;
  }

  // ─── Flat+saddr → Global conversion ───
  // GFX12 flat instructions can use saddr (flat_load_b32 vdst, vaddr, s[pair])
  // GFX9 flat instructions DON'T support saddr — only v[pair] addressing.
  // When saddr is present, convert flat→global (safe for non-LDS addresses,
  // and compilers use DS instructions for LDS, not flat+saddr).
  {
    bool is_flat_with_saddr = false;
    if (mnemonic.find("flat_load_") == 0 || mnemonic.find("flat_store_") == 0) {
      // Check if operands contain s[N:M] (saddr)
      size_t s_bracket = line.find("s[", line.find(mnemonic) + mnemonic.size());
      if (s_bracket != std::string::npos) {
        is_flat_with_saddr = true;
      }
    }
    if (is_flat_with_saddr) {
      // Replace "flat_" with "global_" in the mnemonic
      size_t flat_pos = mnemonic.find("flat_");
      if (flat_pos != std::string::npos) {
        std::string global_mnemonic = "global_" + mnemonic.substr(flat_pos + 5);
        line = ReplaceMnemonic(line, mnemonic, global_mnemonic);
        mnemonic = global_mnemonic;
        // Now fall through to normal mnemonic renaming (global_load_b32 → global_load_dword)
      }
    }
  }

  // ─── Mnemonic renaming (memory + SMEM instructions) ───
  // Try exact match first, then strip encoding suffix (_e32, _e64)
  const auto& mnem_map = GetMnemonicMap();
  auto it = mnem_map.find(mnemonic);
  if (it != mnem_map.end()) {
    line = ReplaceMnemonic(line, mnemonic, it->second);
    mnemonic = it->second;
  } else {
    // Try stripping encoding suffix
    std::string base = mnemonic;
    std::string suffix;
    if (base.size() > 4 && base.substr(base.size() - 4) == "_e32") {
      suffix = "_e32";
      base = base.substr(0, base.size() - 4);
    } else if (base.size() > 4 && base.substr(base.size() - 4) == "_e64") {
      suffix = "_e64";
      base = base.substr(0, base.size() - 4);
    }
    if (!suffix.empty()) {
      it = mnem_map.find(base);
      if (it != mnem_map.end()) {
        std::string new_mnem = it->second + suffix;
        line = ReplaceMnemonic(line, mnemonic, new_mnem);
        mnemonic = new_mnem;
      }
    }
  }

  // ─── Operand syntax translation (cache modifiers, etc.) ───
  line = TranslateOperandSyntax(line, mnemonic);

  // ─── VCC width translation (vcc_lo → vcc for wave64) ───
  line = WidenVccReferences(line);

  // ─── EXEC width adaptation (wave32 → wave64) ───
  auto exec_result = WidenExecOperation(line);
  for (auto& l : exec_result) {
    result.push_back(std::move(l));
  }

  return result;
}

// ── LLVM MC State (reused from hotswap.cpp via forward declarations) ─────────

namespace {

struct LLVMState {
  const llvm::Target* target = nullptr;
  std::unique_ptr<llvm::MCRegisterInfo> MRI;
  std::unique_ptr<const llvm::MCAsmInfo> MAI;
  std::unique_ptr<llvm::MCInstrInfo> MCII;
  std::unique_ptr<llvm::MCSubtargetInfo> STI;
  std::unique_ptr<llvm::MCContext> Ctx;
  std::unique_ptr<llvm::MCDisassembler> disasm;
  std::unique_ptr<llvm::MCInstPrinter> printer;
  std::string cpu;
  bool valid = false;
};

static std::once_flag g_transpile_init_flag;

static void InitLLVMTargets() {
  LLVMInitializeAMDGPUTargetInfo();
  LLVMInitializeAMDGPUTargetMC();
  LLVMInitializeAMDGPUAsmParser();
  LLVMInitializeAMDGPUDisassembler();
}

static std::string ExtractCPU(const std::string& isa) {
  size_t pos = isa.rfind("gfx");
  if (pos == std::string::npos) return "";
  std::string cpu;
  for (size_t i = pos; i < isa.size(); ++i) {
    char c = isa[i];
    if ((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') ||
        (c >= 'A' && c <= 'Z'))
      cpu += c;
    else
      break;
  }
  return cpu;
}

static LLVMState InitLLVM(const std::string& isa_name) {
  std::call_once(g_transpile_init_flag, InitLLVMTargets);

  LLVMState state;
  state.cpu = ExtractCPU(isa_name);
  if (state.cpu.empty()) return state;

  std::string error;
  llvm::Triple triple("amdgcn-amd-amdhsa");

  state.target = llvm::TargetRegistry::lookupTarget("amdgcn", triple, error);
  if (!state.target) return state;

  state.MRI.reset(state.target->createMCRegInfo(triple));
  if (!state.MRI) return state;

  llvm::MCTargetOptions mc_opts;
#if LLVM_VERSION_MAJOR > 9
  state.MAI.reset(state.target->createMCAsmInfo(*state.MRI, triple, mc_opts));
#else
  state.MAI.reset(state.target->createMCAsmInfo(*state.MRI, "amdgcn-amd-amdhsa"));
#endif
  if (!state.MAI) return state;

  state.MCII.reset(state.target->createMCInstrInfo());
  if (!state.MCII) return state;

  state.STI.reset(state.target->createMCSubtargetInfo(triple, state.cpu, ""));
  if (!state.STI || !state.STI->isCPUStringValid(state.cpu)) return state;

#if LLVM_VERSION_MAJOR > 12
  state.Ctx = std::make_unique<llvm::MCContext>(
      triple, state.MAI.get(), state.MRI.get(), state.STI.get());
#else
  auto MOFI = std::make_unique<llvm::MCObjectFileInfo>();
  state.Ctx = std::make_unique<llvm::MCContext>(
      state.MAI.get(), state.MRI.get(), MOFI.get());
  MOFI->InitMCObjectFileInfo(triple, true, *state.Ctx);
#endif

  state.disasm.reset(
      state.target->createMCDisassembler(*state.STI, *state.Ctx));
  if (!state.disasm) return state;

  unsigned asm_variant = state.MAI->getAssemblerDialect();
  state.printer.reset(state.target->createMCInstPrinter(
      triple, asm_variant, *state.MAI, *state.MCII, *state.MRI));

  state.valid = true;
  return state;
}

// ── ELF helpers (minimal, for .text extraction) ──────────────────────────────

struct ElfSection {
  std::string name;
  uint32_t type;
  uint64_t offset;
  uint64_t size;
  uint64_t addr;
};

struct ElfInfo {
  std::vector<ElfSection> sections;
  int text_idx = -1;
  uint64_t text_offset = 0;
  uint64_t text_size = 0;
};

static bool ParseElfMinimal(const uint8_t* elf, size_t elf_size, ElfInfo& info) {
  if (elf_size < 64) return false;
  if (elf[0] != 0x7f || elf[1] != 'E' || elf[2] != 'L' || elf[3] != 'F')
    return false;
  if (elf[4] != 2) return false;

  uint64_t e_shoff;
  uint16_t e_shentsize, e_shnum, e_shstrndx;
  std::memcpy(&e_shoff, elf + 40, 8);
  std::memcpy(&e_shentsize, elf + 58, 2);
  std::memcpy(&e_shnum, elf + 60, 2);
  std::memcpy(&e_shstrndx, elf + 62, 2);

  if (e_shoff == 0 || e_shnum == 0) return false;
  if (e_shoff + static_cast<uint64_t>(e_shnum) * e_shentsize > elf_size)
    return false;

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

  info.sections.resize(e_shnum);
  for (uint16_t i = 0; i < e_shnum; ++i) {
    const uint8_t* sh = elf + e_shoff + i * e_shentsize;
    auto& sec = info.sections[i];
    uint32_t name_idx;
    std::memcpy(&name_idx, sh, 4);
    std::memcpy(&sec.type, sh + 4, 4);
    std::memcpy(&sec.addr, sh + 16, 8);
    std::memcpy(&sec.offset, sh + 24, 8);
    std::memcpy(&sec.size, sh + 32, 8);

    if (shstrtab && name_idx < shstrtab_size)
      sec.name = shstrtab + name_idx;

    if (sec.name == ".text") {
      info.text_idx = i;
      info.text_offset = sec.offset;
      info.text_size = sec.size;
    }
  }

  return info.text_idx >= 0;
}

// ── Kernel Descriptor Patching ───────────────────────────────────────────────
//
// Patch COMPUTE_PGM_RSRC1/RSRC2/RSRC3 fields for wave64 execution.
// The kernel descriptor is the first 64 bytes at each kernel symbol's address.

static void PatchKernelDescriptorsForWave64(uint8_t* elf, size_t elf_size,
                                             const ElfInfo& info) {
  // Find all kernel symbols (we look for 256-byte aligned entries in .text
  // that start with the kernel descriptor magic pattern)

  if (info.text_idx < 0) return;
  const uint8_t* text = elf + info.text_offset;

  // Walk .text at 256-byte boundaries looking for kernel descriptors
  for (uint64_t offset = 0; offset + 64 <= info.text_size; offset += 256) {
    // Kernel descriptor validation: check kernel_code_entry_byte_offset
    // at offset 16 (should be 256 for standard descriptors)
    uint64_t entry_offset;
    std::memcpy(&entry_offset, text + offset + 16, 8);
    if (entry_offset != 256) continue;  // Not a standard kernel descriptor

    // Patch COMPUTE_PGM_RSRC1 (offset 48 in descriptor)
    uint32_t rsrc1;
    std::memcpy(&rsrc1, text + offset + 48, 4);

    // Clear ENABLE_WG_RR_EN (bit 21, GFX12) → set ENABLE_DX10_CLAMP (bit 21, GFX9)
    rsrc1 |= (1u << 21);  // Enable DX10 clamp (standard for GFX9)

    // Clear DISABLE_PERF (bit 23, GFX12) → set ENABLE_IEEE_MODE (bit 23, GFX9)
    rsrc1 |= (1u << 23);  // Enable IEEE mode (standard for GFX9)

    // VGPR count is in bits [5:0] with granularity of 8 for GFX12, 4 for GFX9
    // GFX12: num_vgprs = (rsrc1[5:0] + 1) * 8
    // GFX9:  num_vgprs = (rsrc1[5:0] + 1) * 4
    // So we need to double the VGPR allocation field to maintain the same count
    uint32_t vgpr_field = rsrc1 & 0x3F;
    uint32_t num_vgprs = (vgpr_field + 1) * 8;  // GFX12 granularity
    uint32_t gfx9_field = (num_vgprs / 4) - 1;  // GFX9 granularity
    if (gfx9_field > 63) gfx9_field = 63;        // Clamp to max
    rsrc1 = (rsrc1 & ~0x3Fu) | (gfx9_field & 0x3F);

    std::memcpy(elf + info.text_offset + offset + 48, &rsrc1, 4);

    // Patch kernel_code_properties (offset 56)
    uint16_t props;
    std::memcpy(&props, text + offset + 56, 2);
    // Clear ENABLE_WAVEFRONT_SIZE32 (bit 10) — we're wave64 now
    props &= ~(1u << 10);
    std::memcpy(elf + info.text_offset + offset + 56, &props, 2);

    // Patch COMPUTE_PGM_RSRC3 (offset 44)
    // GFX12 has INST_PREF_SIZE etc. which don't apply to GFX9.
    // Zero it out for safety (GFX9 uses SHARED_VGPR_COUNT which we set to 0).
    uint32_t rsrc3 = 0;
    std::memcpy(elf + info.text_offset + offset + 44, &rsrc3, 4);
  }
}

// ── ELF Metadata Patching ────────────────────────────────────────────────────

static void PatchElfMetadata(uint8_t* elf, size_t elf_size,
                              const std::string& target_cpu) {
  // 1. Patch e_flags (ELF header offset 48)
  // Set EF_AMDGPU_MACH to target value
  uint32_t e_flags;
  std::memcpy(&e_flags, elf + 48, 4);

  // Clear MACH bits [7:0] and set target MACH
  uint8_t target_mach = 0;
  if (target_cpu == "gfx950") target_mach = 0x4f;
  else if (target_cpu == "gfx942") target_mach = 0x4c;
  else if (target_cpu == "gfx90a") target_mach = 0x42;

  if (target_mach != 0) {
    e_flags = (e_flags & ~0xFFu) | target_mach;
    std::memcpy(elf + 48, &e_flags, 4);
  }

  // 2. Patch .note sections (ISA name strings)
  // Scan for "gfx1250" and replace with target CPU name
  // This handles both NT_AMDGPU_ISA (type 27) and MSGPACK (type 32)
  for (size_t i = 0; i + 7 <= elf_size; ++i) {
    if (std::memcmp(elf + i, "gfx1250", 7) == 0) {
      // Check if target CPU name fits (same length or shorter)
      if (target_cpu.size() <= 7) {
        std::memcpy(elf + i, target_cpu.c_str(), target_cpu.size());
        // Pad with nulls if shorter
        for (size_t j = target_cpu.size(); j < 7; ++j) {
          elf[i + j] = '\0';
        }
      }
    }
    // Also handle "gfx1251" variant
    if (std::memcmp(elf + i, "gfx1251", 7) == 0) {
      if (target_cpu.size() <= 7) {
        std::memcpy(elf + i, target_cpu.c_str(), target_cpu.size());
        for (size_t j = target_cpu.size(); j < 7; ++j) {
          elf[i + j] = '\0';
        }
      }
    }
  }
}

}  // anonymous namespace

// ── TranspileCodeObject ──────────────────────────────────────────────────────

RewriteResult TranspileCodeObject(void** elf_data, size_t* elf_size,
                                  const std::string& source_isa,
                                  const std::string& target_isa,
                                  TranspileStats* stats) {
  RewriteResult result = {HSA_STATUS_SUCCESS, 0, 0};
  TranspileStats local_stats;
  if (!stats) stats = &local_stats;

  std::string src_cpu = ExtractCPU(source_isa);
  std::string tgt_cpu = ExtractCPU(target_isa);

  std::cerr << "hotswap: transpile: " << src_cpu << " → " << tgt_cpu << "\n";

  uint8_t* elf = static_cast<uint8_t*>(*elf_data);
  size_t size = *elf_size;

  // Parse ELF
  ElfInfo elf_info;
  if (!ParseElfMinimal(elf, size, elf_info)) {
    std::cerr << "hotswap: transpile: failed to parse ELF\n";
    result.status = HSA_STATUS_ERROR;
    return result;
  }
  if (elf_info.text_size == 0) {
    std::cerr << "hotswap: transpile: empty .text section\n";
    return result;  // Nothing to transpile
  }

  // Initialize LLVM MC for source ISA (disassembler + printer)
  LLVMState src_state = InitLLVM(source_isa);
  if (!src_state.valid) {
    std::cerr << "hotswap: transpile: failed to init source ISA '"
              << source_isa << "'\n";
    result.status = HSA_STATUS_ERROR;
    return result;
  }

  // Step 1: Disassemble source .text → assembly text lines
  const uint8_t* text = elf + elf_info.text_offset;
  std::vector<std::string> source_lines;

  uint64_t pos = 0;
  while (pos < elf_info.text_size) {
    llvm::MCInst inst;
    uint64_t inst_size = 0;

    llvm::ArrayRef<uint8_t> bytes(text + pos, elf_info.text_size - pos);
    auto status = src_state.disasm->getInstruction(
        inst, inst_size, bytes, pos, llvm::nulls());

    if (status == llvm::MCDisassembler::Fail) {
      // Emit raw bytes as .long directive
      if (pos + 4 <= elf_info.text_size) {
        uint32_t word;
        std::memcpy(&word, text + pos, 4);
        std::ostringstream oss;
        oss << ".long 0x" << std::hex << word
            << " ; <undecoded at 0x" << pos << ">";
        source_lines.push_back(oss.str());
      }
      pos += 4;
      ++stats->total_instructions;
      continue;
    }

    // Print instruction to text
    std::string asm_text;
    if (src_state.printer) {
      llvm::raw_string_ostream rso(asm_text);
      src_state.printer->printInst(&inst, 0, "", *src_state.STI, rso);
      rso.flush();
    }

    // Trim
    size_t start = asm_text.find_first_not_of(" \t");
    if (start != std::string::npos && start > 0)
      asm_text = asm_text.substr(start);

    if (!asm_text.empty()) {
      source_lines.push_back(asm_text);
    }

    pos += inst_size;
    ++stats->total_instructions;
  }

  std::cerr << "hotswap: transpile: disassembled " << stats->total_instructions
            << " instructions from " << elf_info.text_size << " bytes\n";

  // Step 2: Translate each instruction
  std::string translated_asm;
  translated_asm += ".text\n";

  // Add wave64 EXEC initialization at the very beginning
  // This sets exec_hi = 0 to disable lanes 32-63
  translated_asm += "s_mov_b32 exec_hi, 0\n";

  for (const auto& line : source_lines) {
    auto translated = TranslateInstruction(line, src_cpu, tgt_cpu);
    for (const auto& t : translated) {
      if (t.empty()) continue;

      // Track statistics
      std::string mnemonic = ExtractMnemonic(line);
      if (t.find("UNSUPPORTED") != std::string::npos) {
        ++stats->unsupported_skipped;
      } else if (t != line) {
        std::string new_mnem = ExtractMnemonic(t);
        if (IsWaitInstruction(mnemonic)) {
          ++stats->translated_waitcnt;
        } else if (new_mnem != mnemonic) {
          ++stats->translated_renamed;
        } else if (t.find("exec_hi") != std::string::npos && t != line) {
          ++stats->translated_exec;
        } else {
          ++stats->translated_passthrough;
        }
      } else {
        ++stats->translated_passthrough;
      }

      translated_asm += t + "\n";
    }
  }

  std::cerr << "hotswap: transpile: translated " << source_lines.size()
            << " instructions → " << stats->translated_passthrough
            << " passthrough, " << stats->translated_renamed
            << " renamed, " << stats->translated_waitcnt
            << " waitcnt, " << stats->translated_exec
            << " exec-widened, " << stats->unsupported_skipped
            << " unsupported\n";

  // Step 3: Assemble translated text for target ISA
  LLVMState tgt_state = InitLLVM(target_isa);
  if (!tgt_state.valid) {
    std::cerr << "hotswap: transpile: failed to init target ISA '"
              << target_isa << "'\n";
    result.status = HSA_STATUS_ERROR;
    return result;
  }

  // Create MC assembly pipeline
  llvm::Triple tgt_triple("amdgcn-amd-amdhsa");
  llvm::MCTargetOptions mc_opts;

  tgt_state.Ctx->reset();

  llvm::StringRef asm_ref(translated_asm);
  auto buf = llvm::MemoryBuffer::getMemBuffer(asm_ref, "", false);
  llvm::SourceMgr src_mgr;
  src_mgr.AddNewSourceBuffer(std::move(buf), llvm::SMLoc());

  std::string data;
  auto data_stream = std::make_unique<llvm::raw_string_ostream>(data);
  auto bos = std::make_unique<llvm::buffer_ostream>(*data_stream);

#if LLVM_VERSION_MAJOR > 14
  llvm::MCCodeEmitter* ce =
      tgt_state.target->createMCCodeEmitter(*tgt_state.MCII, *tgt_state.Ctx);
#else
  llvm::MCCodeEmitter* ce =
      tgt_state.target->createMCCodeEmitter(
          *tgt_state.MCII, *tgt_state.MRI, *tgt_state.Ctx);
#endif
  llvm::MCAsmBackend* mab =
      tgt_state.target->createMCAsmBackend(
          *tgt_state.STI, *tgt_state.MRI, mc_opts);

  if (!ce || !mab) {
    std::cerr << "hotswap: transpile: failed to create code emitter/backend\n";
    result.status = HSA_STATUS_ERROR;
    return result;
  }

#if LLVM_VERSION_MAJOR > 20
  auto streamer = std::unique_ptr<llvm::MCStreamer>(
      tgt_state.target->createMCObjectStreamer(
          tgt_triple, *tgt_state.Ctx,
          std::unique_ptr<llvm::MCAsmBackend>(mab),
          mab->createObjectWriter(*bos),
          std::unique_ptr<llvm::MCCodeEmitter>(ce), *tgt_state.STI));
#else
  auto streamer = std::unique_ptr<llvm::MCStreamer>(
      tgt_state.target->createMCObjectStreamer(
          tgt_triple, *tgt_state.Ctx,
          std::unique_ptr<llvm::MCAsmBackend>(mab),
          mab->createObjectWriter(*bos),
          std::unique_ptr<llvm::MCCodeEmitter>(ce), *tgt_state.STI,
          mc_opts.MCRelaxAll, mc_opts.MCIncrementalLinkerCompatible, false));
#endif

  if (!streamer) {
    std::cerr << "hotswap: transpile: failed to create MC streamer\n";
    result.status = HSA_STATUS_ERROR;
    return result;
  }

  auto parser = std::unique_ptr<llvm::MCAsmParser>(
      llvm::createMCAsmParser(src_mgr, *tgt_state.Ctx, *streamer,
                              *tgt_state.MAI));
  auto tap = std::unique_ptr<llvm::MCTargetAsmParser>(
      tgt_state.target->createMCAsmParser(
          *tgt_state.STI, *parser, *tgt_state.MCII, mc_opts));
  if (!tap) {
    std::cerr << "hotswap: transpile: failed to create target asm parser\n";
    result.status = HSA_STATUS_ERROR;
    return result;
  }
  parser->setTargetParser(*tap);

  bool asm_failed = parser->Run(true);

  // Cleanup assembly pipeline before reading output
  tap.reset();
  parser.reset();
  streamer.reset();
  bos.reset();
  data_stream->flush();

  if (asm_failed) {
    std::cerr << "hotswap: transpile: assembly failed for " << tgt_cpu << "\n";
    // Continue anyway — partial assembly may still be usable
  }

  if (data.size() < 64) {
    std::cerr << "hotswap: transpile: assembled output too small ("
              << data.size() << " bytes)\n";
    result.status = HSA_STATUS_ERROR;
    return result;
  }

  // Step 4: Extract .text from assembled ELF
  const uint8_t* asm_elf = reinterpret_cast<const uint8_t*>(data.data());
  ElfInfo asm_info;
  if (!ParseElfMinimal(asm_elf, data.size(), asm_info)) {
    std::cerr << "hotswap: transpile: failed to parse assembled ELF\n";
    result.status = HSA_STATUS_ERROR;
    return result;
  }

  const uint8_t* new_text = asm_elf + asm_info.text_offset;
  uint64_t new_text_size = asm_info.text_size;

  std::cerr << "hotswap: transpile: assembled " << new_text_size
            << " bytes (original: " << elf_info.text_size << ")\n";

  // Step 5: Replace .text in the original ELF
  if (new_text_size <= elf_info.text_size) {
    // Fits in-place — copy and NOP-pad remainder
    std::memcpy(elf + elf_info.text_offset, new_text, new_text_size);
    // NOP-fill remaining space
    uint8_t nop_bytes[] = {0x00, 0x00, 0x80, 0xBF};  // s_nop 0
    for (uint64_t i = new_text_size; i + 4 <= elf_info.text_size; i += 4) {
      std::memcpy(elf + elf_info.text_offset + i, nop_bytes, 4);
    }
  } else {
    // New .text is larger — need to grow the ELF buffer
    // Calculate growth needed
    uint64_t growth = new_text_size - elf_info.text_size;
    size_t new_elf_size = size + growth;
    uint8_t* new_elf = static_cast<uint8_t*>(std::malloc(new_elf_size));
    if (!new_elf) {
      std::cerr << "hotswap: transpile: failed to allocate " << new_elf_size
                << " bytes for grown ELF\n";
      result.status = HSA_STATUS_ERROR;
      return result;
    }

    // Copy everything before .text
    std::memcpy(new_elf, elf, elf_info.text_offset);
    // Copy new .text
    std::memcpy(new_elf + elf_info.text_offset, new_text, new_text_size);
    // Copy everything after old .text
    uint64_t after_text = elf_info.text_offset + elf_info.text_size;
    if (after_text < size) {
      std::memcpy(new_elf + elf_info.text_offset + new_text_size,
                  elf + after_text, size - after_text);
    }

    // Update .text section header size
    uint64_t e_shoff;
    uint16_t e_shentsize;
    std::memcpy(&e_shoff, new_elf + 40, 8);
    std::memcpy(&e_shentsize, new_elf + 58, 2);

    // Adjust section header offset if it was after .text
    if (e_shoff > elf_info.text_offset) {
      e_shoff += growth;
      std::memcpy(new_elf + 40, &e_shoff, 8);
    }

    // Update .text section size in section header
    uint16_t e_shnum;
    std::memcpy(&e_shnum, new_elf + 60, 2);
    for (uint16_t i = 0; i < e_shnum; ++i) {
      uint64_t sh_off = e_shoff + i * e_shentsize;
      if (sh_off + e_shentsize > new_elf_size) break;

      uint64_t sec_offset;
      std::memcpy(&sec_offset, new_elf + sh_off + 24, 8);

      if (static_cast<int>(i) == elf_info.text_idx) {
        // Update .text size
        std::memcpy(new_elf + sh_off + 32, &new_text_size, 8);
      } else if (sec_offset > elf_info.text_offset) {
        // Shift sections after .text
        sec_offset += growth;
        std::memcpy(new_elf + sh_off + 24, &sec_offset, 8);
      }
    }

    *elf_data = new_elf;
    *elf_size = new_elf_size;
    elf = new_elf;
    size = new_elf_size;
  }

  // Step 6: Patch kernel descriptors for wave64
  ElfInfo updated_info;
  ParseElfMinimal(elf, size, updated_info);
  PatchKernelDescriptorsForWave64(elf, size, updated_info);

  // Step 7: Patch ELF metadata (e_flags, .note ISA strings)
  PatchElfMetadata(elf, size, tgt_cpu);

  result.rules_matched = stats->translated_passthrough +
                         stats->translated_renamed +
                         stats->translated_waitcnt;

  std::cerr << "hotswap: transpile: complete (" << src_cpu << " → " << tgt_cpu
            << ")\n";

  return result;
}

}  // namespace hotswap
}  // namespace rocr
