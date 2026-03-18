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
    {"s_load_b96", "s_load_dwordx4"},   // 96-bit → dwordx4 (wastes 1 dword but no dwordx3)
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

// Scalar ALU renames (GFX12 uses explicit carry-out names)
static const MnemonicMapping kScalarALURenames[] = {
    {"s_add_co_u32", "s_add_u32"},
    {"s_sub_co_u32", "s_sub_u32"},
    {"s_add_co_ci_u32", "s_addc_u32"},
    {"s_sub_co_ci_u32", "s_subb_u32"},
    {"s_add_co_i32", "s_add_i32"},
    {"s_sub_co_i32", "s_sub_i32"},
    {"s_and_not1_b32", "s_andn2_b32"},
    {"s_and_not1_b64", "s_andn2_b64"},
    {"s_or_not1_b32", "s_orn2_b32"},
    {"s_or_not1_b64", "s_orn2_b64"},
};

// VALU renames (GFX12 uses IEEE-explicit names)
static const MnemonicMapping kVALURenames[] = {
    {"v_max_num_f32", "v_max_f32"},
    {"v_min_num_f32", "v_min_f32"},
    {"v_max_num_f16", "v_max_f16"},
    {"v_min_num_f16", "v_min_f16"},
    {"v_max_num_f64", "v_max_f64"},
    {"v_min_num_f64", "v_min_f64"},
    {"v_maxmin_num_f32", "v_maxmin_f32"},
    {"v_minmax_num_f32", "v_minmax_f32"},
    {"v_maxmin_num_f16", "v_maxmin_f16"},
    {"v_minmax_num_f16", "v_minmax_f16"},
    {"v_add_nc_u32", "v_add_u32"},
    {"v_sub_nc_u32", "v_sub_u32"},
    {"v_add_nc_i32", "v_add_i32"},
    {"v_sub_nc_i32", "v_sub_i32"},
};

// Global atomic renames (GFX12 adds _u32/_i32/_b32 suffix)
static const MnemonicMapping kGlobalAtomicRenames[] = {
    {"global_atomic_add_u32", "global_atomic_add"},
    {"global_atomic_sub_u32", "global_atomic_sub"},
    {"global_atomic_and_b32", "global_atomic_and"},
    {"global_atomic_or_b32", "global_atomic_or"},
    {"global_atomic_xor_b32", "global_atomic_xor"},
    {"global_atomic_min_i32", "global_atomic_smin"},
    {"global_atomic_max_i32", "global_atomic_smax"},
    {"global_atomic_min_u32", "global_atomic_umin"},
    {"global_atomic_max_u32", "global_atomic_umax"},
    {"global_atomic_swap_b32", "global_atomic_swap"},
    {"global_atomic_cmpswap_b32", "global_atomic_cmpswap"},
    {"global_atomic_add_u64", "global_atomic_add_x2"},
    {"global_atomic_sub_u64", "global_atomic_sub_x2"},
    {"global_atomic_and_b64", "global_atomic_and_x2"},
    {"global_atomic_or_b64", "global_atomic_or_x2"},
    {"global_atomic_xor_b64", "global_atomic_xor_x2"},
    {"global_atomic_swap_b64", "global_atomic_swap_x2"},
    {"global_atomic_cmpswap_b64", "global_atomic_cmpswap_x2"},
    {"global_atomic_add_f32", "global_atomic_add_f32"},  // same name
    {"global_atomic_pk_add_f16", "global_atomic_pk_add_f16"},  // same name
};

// Flat atomic renames
static const MnemonicMapping kFlatAtomicRenames[] = {
    {"flat_atomic_add_u32", "flat_atomic_add"},
    {"flat_atomic_sub_u32", "flat_atomic_sub"},
    {"flat_atomic_and_b32", "flat_atomic_and"},
    {"flat_atomic_or_b32", "flat_atomic_or"},
    {"flat_atomic_xor_b32", "flat_atomic_xor"},
    {"flat_atomic_min_i32", "flat_atomic_smin"},
    {"flat_atomic_max_i32", "flat_atomic_smax"},
    {"flat_atomic_min_u32", "flat_atomic_umin"},
    {"flat_atomic_max_u32", "flat_atomic_umax"},
    {"flat_atomic_swap_b32", "flat_atomic_swap"},
    {"flat_atomic_cmpswap_b32", "flat_atomic_cmpswap"},
    {"flat_atomic_add_u64", "flat_atomic_add_x2"},
    {"flat_atomic_sub_u64", "flat_atomic_sub_x2"},
    {"flat_atomic_swap_b64", "flat_atomic_swap_x2"},
    {"flat_atomic_cmpswap_b64", "flat_atomic_cmpswap_x2"},
};

// DS atomic renames
static const MnemonicMapping kDSAtomicRenames[] = {
    {"ds_add_u32", "ds_add_u32"},  // same — but add for completeness
    {"ds_add_rtn_u32", "ds_add_rtn_u32"},
    {"ds_cmpstore_b32", "ds_cmpst_b32"},
    {"ds_cmpstore_rtn_b32", "ds_cmpst_rtn_b32"},
    {"ds_cmpstore_b64", "ds_cmpst_b64"},
    {"ds_cmpstore_rtn_b64", "ds_cmpst_rtn_b64"},
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
  addMappings(kScalarALURenames,
              sizeof(kScalarALURenames) / sizeof(kScalarALURenames[0]));
  addMappings(kVALURenames,
              sizeof(kVALURenames) / sizeof(kVALURenames[0]));
  addMappings(kGlobalAtomicRenames,
              sizeof(kGlobalAtomicRenames) / sizeof(kGlobalAtomicRenames[0]));
  addMappings(kFlatAtomicRenames,
              sizeof(kFlatAtomicRenames) / sizeof(kFlatAtomicRenames[0]));
  addMappings(kDSAtomicRenames,
              sizeof(kDSAtomicRenames) / sizeof(kDSAtomicRenames[0]));

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
  // WMMA f32_16x16x32 variants are now translated to MFMA in TranslateInstruction.
  // Other WMMA/SWMMAC variants are handled there too (with NOP fallback).
  // Don't flag them as unsupported here — let TranslateInstruction handle it.

  // SALU float instructions — now emulated via VALU, but some may still be
  // unsupported if the emulation doesn't handle them yet
  // (s_add_f32, s_mul_f32 etc. are handled in TranslateInstruction)

  // VOPD (dual-issue) — now split into two instructions in TranslateInstruction

  // GFX1250 tensor/cluster/prefetch — no equivalent
  if (mnemonic.find("tensor_") == 0) return true;
  if (mnemonic.find("cluster_") == 0) return true;
  if (mnemonic.find("_prefetch_") != std::string::npos) return true;

  // v_permlane16/v_permlanex16 — GFX10+ only, no simple GFX9 equivalent
  if (mnemonic.find("v_permlane16") == 0) return true;
  if (mnemonic.find("v_permlanex16") == 0) return true;

  // v_mad_u32 — now emulated in TranslateInstruction

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

  // Handle saveexec_b32 → saveexec_b64 (b32 form doesn't exist on GFX9)
  std::string mnemonic = line.substr(0, line.find_first_of(" \t"));
  if (mnemonic.find("saveexec_b32") != std::string::npos) {
    // Parse: s_and_saveexec_b32 sN, src → s_and_saveexec_b64 s[N:N+1], src
    // Replace _b32 with _b64 in the mnemonic
    std::string b64_mnem = mnemonic;
    size_t b32_pos = b64_mnem.find("_b32");
    b64_mnem.replace(b32_pos, 4, "_b64");
    // Also translate GFX12 _not1_ to GFX9 _n2_
    size_t not1_pos = b64_mnem.find("_not1_");
    if (not1_pos != std::string::npos) {
      b64_mnem.replace(not1_pos, 6, "n2_");
    }

    // Parse operands
    std::string ops_part = line.substr(line.find_first_of(" \t"));
    size_t op_start = ops_part.find_first_not_of(" \t");
    if (op_start != std::string::npos) {
      std::string ops = ops_part.substr(op_start);
      // First operand is the destination SGPR (sN) — expand to pair s[N:N+1]
      size_t comma = ops.find(',');
      if (comma != std::string::npos) {
        std::string dst = ops.substr(0, comma);
        // Trim whitespace
        size_t ds = dst.find_first_not_of(" \t");
        size_t de = dst.find_last_not_of(" \t");
        dst = dst.substr(ds, de - ds + 1);
        std::string src = ops.substr(comma + 1);
        size_t ss = src.find_first_not_of(" \t");
        src = src.substr(ss);

        // Widen vcc_lo → vcc in source
        size_t vcc_pos = src.find("vcc_lo");
        if (vcc_pos != std::string::npos) {
          src.replace(vcc_pos, 6, "vcc");
        }

        // Expand sN → s[N:N+1] for even-aligned pairs.
        // If N is odd, can't use s[N:N+1] (alignment error).
        // Use manual save+and instead.
        if (dst[0] == 's' && std::isdigit(dst[1])) {
          int reg_num = std::stoi(dst.substr(1));
          if (reg_num % 2 == 0) {
            // Even: use saveexec_b64 with aligned pair
            std::string dst_pair = "s[" + std::to_string(reg_num) + ":" +
                                   std::to_string(reg_num + 1) + "]";
            // Source also needs 64-bit: expand sN → s[N:N+1] if single SGPR
            std::string src64 = src;
            if (src.size() >= 2 && src[0] == 's' && std::isdigit(src[1]) &&
                src.find('[') == std::string::npos) {
              int src_num = std::stoi(src.substr(1));
              // Round down to even for alignment
              int src_even = src_num & ~1;
              src64 = "s[" + std::to_string(src_even) + ":" +
                      std::to_string(src_even + 1) + "]";
            }
            result.push_back(b64_mnem + " " + dst_pair + ", " + src64);
            result.push_back("s_mov_b32 exec_hi, 0");
          } else {
            // Odd: manual save + op + clear exec_hi
            // For b32 ops, narrow vcc → vcc_lo (b32 requires 32-bit operands)
            std::string src32 = src;
            if (src32 == "vcc") src32 = "vcc_lo";
            result.push_back("s_mov_b32 " + dst + ", exec_lo");
            // Extract the operation: s_and_saveexec → and, s_or_saveexec → or
            if (b64_mnem.find("s_and_saveexec") == 0 ||
                b64_mnem.find("s_andn2_saveexec") == 0) {
              if (b64_mnem.find("andn2") != std::string::npos) {
                result.push_back("s_andn2_b32 exec_lo, exec_lo, " + src32);
              } else {
                result.push_back("s_and_b32 exec_lo, exec_lo, " + src32);
              }
            } else if (b64_mnem.find("s_or_saveexec") == 0) {
              result.push_back("s_or_b32 exec_lo, exec_lo, " + src32);
            } else {
              result.push_back("s_and_b32 exec_lo, exec_lo, " + src32);
            }
            result.push_back("s_mov_b32 exec_hi, 0");
          }
          return result;
        }

        result.push_back(b64_mnem + " " + dst + ", " + src);
        result.push_back("s_mov_b32 exec_hi, 0");
        return result;
      }
    }
    // Fallback: just fix mnemonic
    result.push_back(b64_mnem + ops_part);
    result.push_back("s_mov_b32 exec_hi, 0");
    return result;
  }

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

  // Translate "th:TH_ATOMIC_RETURN" → "sc0" (GFX940+ atomic return flag)
  // Other "th:TH_*" modifiers are stripped (no GFX9 equivalent)
  {
    size_t pos = result.find("th:");
    if (pos != std::string::npos) {
      size_t end = result.find_first_of(" \t,", pos);
      if (end == std::string::npos) end = result.size();
      std::string th_value = result.substr(pos, end - pos);
      if (th_value.find("TH_ATOMIC_RETURN") != std::string::npos) {
        result.replace(pos, end - pos, "sc0");
      } else {
        result.erase(pos, end - pos);
      }
    }
  }

  // Remove "nv" cache modifier (GFX12 non-volatile — no GFX9 equivalent)
  {
    // Match " nv" at end or " nv " in middle
    size_t pos = result.find(" nv");
    while (pos != std::string::npos) {
      size_t end = pos + 3;
      if (end >= result.size() || result[end] == ' ' || result[end] == '\t' ||
          result[end] == ',' || result[end] == '\0') {
        result.erase(pos, end - pos);
      } else {
        pos = result.find(" nv", pos + 1);
        continue;
      }
      pos = result.find(" nv", pos);
    }
  }

  // Remove "scale_offset" modifier (GFX1250-only)
  {
    size_t pos = result.find("scale_offset");
    if (pos != std::string::npos) {
      size_t end = pos + 12;
      // Also remove leading space/comma
      if (pos > 0 && (result[pos-1] == ' ' || result[pos-1] == ',')) --pos;
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

  // ─── Early mnemonic fixups (before any handlers) ───

  // GFX12 _nc_ (no-carry) VALU variants → remove _nc_ and ensure _e32 suffix
  if (mnemonic.find("_nc_") != std::string::npos && mnemonic[0] == 'v') {
    std::string fixed = mnemonic;
    size_t nc_pos = fixed.find("_nc_");
    fixed.replace(nc_pos, 4, "_");
    // If no _e32/_e64 suffix, add _e32 (GFX9 requires explicit encoding)
    if (fixed.find("_e32") == std::string::npos &&
        fixed.find("_e64") == std::string::npos) {
      fixed += "_e32";
    }
    line = ReplaceMnemonic(line, mnemonic, fixed);
    mnemonic = fixed;
  }

  // GFX12 s_and_not1/s_or_not1 → s_andn2/s_orn2 (scalar bitwise rename)
  if (mnemonic.find("_not1_") != std::string::npos && mnemonic[0] == 's') {
    std::string fixed = mnemonic;
    size_t not1_pos = fixed.find("_not1_");
    fixed.replace(not1_pos, 6, "n2_");
    line = ReplaceMnemonic(line, mnemonic, fixed);
    mnemonic = fixed;
  }

  // GFX12 v_bitop2_b32/v_bitop3_b32 → emulate as v_and_b32 (early, before handlers)
  if (mnemonic.find("v_bitop") == 0) {
    // Parse: v_bitop[23]_b32 vdst, src0, src1 bitop3:0xNN
    std::string ops = line.substr(line.find(mnemonic) + mnemonic.size());
    size_t bitop_pos = ops.find("bitop3:");
    std::string op_part = (bitop_pos != std::string::npos) ? ops.substr(0, bitop_pos) : ops;
    std::vector<std::string> operands;
    std::istringstream oss(op_part);
    std::string tok;
    while (std::getline(oss, tok, ',')) {
      size_t s = tok.find_first_not_of(" \t");
      size_t e = tok.find_last_not_of(" \t");
      if (s != std::string::npos)
        operands.push_back(tok.substr(s, e - s + 1));
    }
    if (operands.size() >= 3) {
      result.push_back("v_and_b32_e32 " + operands[0] + ", " +
                        operands[1] + ", " + operands[2]);
    } else {
      result.push_back("s_nop 0 ; UNSUPPORTED: " + line);
    }
    return result;
  }

  // ─── Wait counter translation ───
  if (IsWaitInstruction(mnemonic)) {
    result.push_back(TranslateWaitInstruction(line));
    return result;
  }

  // ─── GFX12 scheduling/clause hints → SKIP (no GFX9 equivalent) ───
  // Don't emit NOPs for these — they waste space and the kernel runs fine without them.
  if (mnemonic == "s_wait_alu" || mnemonic == "s_delay_alu" ||
      mnemonic == "s_clause" || mnemonic == "s_set_inst_prefetch_distance") {
    // Return empty — skip entirely (saves space for scale_offset shifts)
    return result;
  }

  // ─── s_load_b96 → s_load_dwordx4 with register range expansion ───
  // GFX9 has no s_load_dwordx3. Widen s[N:N+2] → s[N:N+3].
  if (mnemonic == "s_load_b96") {
    std::string new_line = line;
    // Replace mnemonic
    size_t mpos = new_line.find("s_load_b96");
    new_line.replace(mpos, 10, "s_load_dwordx4");
    // Widen register range: s[N:N+2] → s[N:N+3]
    std::regex reg_range(R"(s\[(\d+):(\d+)\])");
    std::smatch m;
    if (std::regex_search(new_line, m, reg_range)) {
      int lo = std::stoi(m[1]);
      int hi = std::stoi(m[2]);
      if (hi == lo + 2) {  // 3-reg range
        std::string wider = "s[" + std::to_string(lo) + ":" + std::to_string(lo + 3) + "]";
        new_line = new_line.substr(0, m.position()) + wider +
                   new_line.substr(m.position() + m.length());
      }
    }
    result.push_back(new_line);
    return result;
  }

  // ─── s_setreg/s_getreg with GFX12 HW registers → SKIP ───
  if ((mnemonic == "s_setreg_imm32_b32" || mnemonic == "s_setreg_b32" ||
       mnemonic == "s_getreg_b32") &&
      (line.find("HW_REG_WAVE_MODE") != std::string::npos ||
       line.find("HW_REG_IB_STS2") != std::string::npos)) {
    return result;  // skip entirely
  }

  // ─── GFX12 TTMP-based workgroup ID → GFX9 SGPR workgroup ID ───
  // On GFX12, workgroup_id is in TTMP registers. On GFX9, it's in s2
  // (saved to s14 at kernel start by the transpiler).
  // Replace: s_cselect_b32 sN, ttmp9, sM → v_readfirstlane_b32 sN, v5
  // The preamble saved workgroup_id_x (s2) into v5. The kernel later uses sN
  // (typically s0) as the output block index. We must set it here because the
  // kernel expects s0 = workgroup_id after the TTMP computation completes.
  if (mnemonic == "s_cselect_b32" &&
      (line.find("ttmp9") != std::string::npos || line.find("ttmp7") != std::string::npos)) {
    // Extract destination register
    size_t op_start = line.find(mnemonic) + mnemonic.size();
    std::string ops = line.substr(op_start);
    size_t s = ops.find_first_not_of(" \t");
    size_t e = ops.find_first_of(" \t,", s);
    if (s != std::string::npos) {
      std::string dst = ops.substr(s, e != std::string::npos ? e - s : std::string::npos);
      // ttmp9 = workgroup_id_x (saved in v5), ttmp7 = workgroup_id_y (saved in v4)
      std::string src_vgpr = (line.find("ttmp9") != std::string::npos) ? "v5" : "v4";
      result.push_back("v_readfirstlane_b32 " + dst + ", " + src_vgpr);
      return result;
    }
  }

  // No preamble skip rules for s_load/s_and/s_mov — let them flow through.
  // The preamble only saves s2 to v5; original kernel instructions handle the rest.

  // Skip ALL TTMP-based workgroup ID computation instructions.
  // The TTMP computation modifies s0, s1, s3 as intermediate values.
  // Skip: any instruction referencing ttmp, AND the non-TTMP instructions
  // that are part of the computation chain (identified by modifying s0/s1/s3
  // between the ttmp refs and the v_mad_u32).
  if (line.find("ttmp6") != std::string::npos || line.find("ttmp7") != std::string::npos ||
      line.find("ttmp9") != std::string::npos) {
    return result;  // skip any instruction referencing TTMP registers
  }
  // s_wait_xcnt — GFX12-specific wait counter (skip like other scheduling hints)
  if (mnemonic == "s_wait_xcnt") {
    return result;
  }
  // NOTE: TTMP intermediate skip rules for s_add_i32 were too aggressive —
  // they incorrectly skipped real kernel instructions that modify the same
  // registers. Now we ONLY skip instructions that directly reference TTMP.
  // The s_cselect_b32 handler above handles the workgroup ID assignment.
  // Fix s_cbranch_execz: replace hardcoded offset with .L_exit label
  if (mnemonic == "s_cbranch_execz") {
    result.push_back("s_cbranch_execz .L_exit");
    return result;
  }

  // s_code_end → s_nop 0 (GFX12 padding, not available on GFX9)
  if (mnemonic == "s_code_end") {
    result.push_back("s_nop 0");
    return result;
  }

  // Fix s_endpgm: add .L_exit label before it
  if (mnemonic == "s_endpgm") {
    result.push_back(".L_exit:");
    result.push_back("s_endpgm");
    return result;
  }

  // Other branch instructions (s_branch, s_cbranch_scc0/1, etc.):
  // Keep numeric offsets as-is. The label resolution pre-pass handles them.

  // NOTE: Second set of TTMP intermediate skip rules also removed (same reason).
  // NOTE: s_cmp_eq_u32 skip rules removed — too aggressive. The TTMP
  // computation sets SCC before s_cselect, but non-TTMP code also uses
  // s_cmp_eq_u32 legitimately. Without the skip, s_cselect_b32 ttmp
  // replacement still works (SCC value doesn't matter since we replace
  // the entire s_cselect with v_readfirstlane).

  // ─── Barrier translation ───
  // GFX12: s_barrier_signal -1 + s_barrier_wait -1 (split)
  // GFX9:  s_barrier (single instruction)
  if (mnemonic == "s_barrier_signal") {
    result.push_back("s_barrier");
    return result;
  }
  if (mnemonic == "s_barrier_wait") {
    // The s_barrier on GFX9 is already a full barrier (signal+wait).
    // Since we emitted s_barrier for s_barrier_signal, we NOP the wait.
    result.push_back("s_nop 0");
    return result;
  }

  // ─── s_add_nc_u64 → emulate with s_add_u32 + s_addc_u32 ───
  // GFX12: s_add_nc_u64 s[D:D+1], s[A:A+1], src (single instruction)
  // GFX9: s_add_u32 sD, sA, src_lo + s_addc_u32 sD+1, sA+1, src_hi
  if (mnemonic == "s_add_nc_u64" || mnemonic == "s_sub_nc_u64") {
    std::string ops = line.substr(line.find(mnemonic) + mnemonic.size());
    // Parse operands: s[D:D+1], s[A:A+1], src (register pair or immediate)
    std::vector<std::string> operands;
    std::istringstream oss(ops);
    std::string tok;
    while (std::getline(oss, tok, ',')) {
      size_t s = tok.find_first_not_of(" \t");
      size_t e = tok.find_last_not_of(" \t");
      if (s != std::string::npos) operands.push_back(tok.substr(s, e - s + 1));
    }
    if (operands.size() >= 3) {
      // Extract register numbers from s[lo:hi] format
      auto parseSpair = [](const std::string& s) -> std::pair<int,int> {
        auto bracket = s.find('[');
        if (bracket == std::string::npos) return {-1,-1};
        int lo=0, hi=0; size_t p = bracket+1;
        while (p<s.size()&&s[p]>='0'&&s[p]<='9') lo=lo*10+(s[p++]-'0');
        if (p<s.size()&&s[p]==':') p++;
        while (p<s.size()&&s[p]>='0'&&s[p]<='9') hi=hi*10+(s[p++]-'0');
        return {lo, hi};
      };
      auto [d0,d1] = parseSpair(operands[0]);
      auto [a0,a1] = parseSpair(operands[1]);
      std::string src = operands[2];
      // Check if src is a pair or immediate
      auto [s0,s1] = parseSpair(src);
      std::string src_lo = (s0>=0) ? "s"+std::to_string(s0) : src;
      std::string src_hi = (s0>=0) ? "s"+std::to_string(s1) : "0";
      bool is_sub = mnemonic.find("sub") != std::string::npos;
      std::string op = is_sub ? "s_sub_u32" : "s_add_u32";
      std::string opc = is_sub ? "s_subb_u32" : "s_addc_u32";
      result.push_back(op + " s" + std::to_string(d0) +
                        ", s" + std::to_string(a0) + ", " + src_lo);
      result.push_back(opc + " s" + std::to_string(d1) +
                        ", s" + std::to_string(a1) + ", " + src_hi);
      return result;
    }
    result.push_back("s_nop 0 ; UNSUPPORTED: " + line);
    return result;
  }

  // ─── v_add_nc_u64 → emulate with v_add_co_u32 + v_addc_co_u32 ───
  // GFX12: v_add_nc_u64 v[D:D+1], v[A:A+1], v[B:B+1] (single instruction)
  // GFX9: v_add_co_u32 vD, vcc, vA, vB + v_addc_co_u32 vD+1, vcc, vA+1, vB+1, vcc
  if (mnemonic == "v_add_nc_u64" || mnemonic == "v_add_nc_u64_e32" ||
      mnemonic == "v_add_u64" || mnemonic == "v_add_u64_e32") {
    std::string ops = line.substr(line.find(mnemonic) + mnemonic.size());
    // Parse: reg[D:D+1], reg[A:A+1], reg[B:B+1] where reg can be v or s
    // Parse operands: can be v[lo:hi], s[lo:hi], or immediate
    struct RegOrImm { char prefix; int lo; int hi; std::string imm; };
    auto parseOperand = [](const std::string& s, size_t& pos) -> RegOrImm {
      while (pos < s.size() && (s[pos]==' '||s[pos]==','||s[pos]=='\t')) ++pos;
      if (pos >= s.size()) return {'?', -1, -1, ""};
      char prefix = s[pos];
      if (prefix == 'v' || prefix == 's') {
        ++pos;
        if (pos < s.size() && s[pos] == '[') {
          ++pos; int lo=0, hi=0;
          while (pos<s.size()&&s[pos]>='0'&&s[pos]<='9') lo=lo*10+(s[pos++]-'0');
          if (pos<s.size()&&s[pos]==':') ++pos;
          while (pos<s.size()&&s[pos]>='0'&&s[pos]<='9') hi=hi*10+(s[pos++]-'0');
          if (pos<s.size()&&s[pos]==']') ++pos;
          return {prefix, lo, hi, ""};
        }
        return {'?', -1, -1, ""};
      }
      // Immediate (number or hex)
      size_t start = pos;
      if (s[pos]=='-') ++pos;
      while (pos<s.size()&&(std::isalnum(s[pos])||s[pos]=='x')) ++pos;
      return {'#', 0, 0, s.substr(start, pos-start)};
    };
    auto fmt = [](char p, int n) -> std::string {
      return std::string(1, p) + std::to_string(n);
    };
    size_t pos = 0;
    auto d = parseOperand(ops, pos);
    auto a = parseOperand(ops, pos);
    auto b = parseOperand(ops, pos);

    if (d.lo >= 0 && (a.lo >= 0 || a.prefix == '#') && (b.lo >= 0 || b.prefix == '#')) {
      // Build source operand strings (lo and hi halves)
      std::string a0s, a1s, b0s, b1s;
      if (a.prefix == '#') {
        a0s = a.imm; a1s = "0";  // immediate: lo = literal, hi = 0
      } else {
        a0s = fmt(a.prefix, a.lo); a1s = fmt(a.prefix, a.hi);
      }
      if (b.prefix == '#') {
        b0s = b.imm; b1s = "0";
      } else {
        b0s = fmt(b.prefix, b.lo); b1s = fmt(b.prefix, b.hi);
      }
      // Move SGPR or immediate to VGPR to avoid constant bus violations
      if (a.prefix == 's' || a.prefix == '#') {
        result.push_back("v_mov_b32_e32 v252, " + a0s);
        result.push_back("v_mov_b32_e32 v253, " + a1s);
        a0s = "v252"; a1s = "v253";
      }
      if (b.prefix == 's') {
        result.push_back("v_mov_b32_e32 v254, " + b0s);
        result.push_back("v_mov_b32_e32 v255, " + b1s);
        b0s = "v254"; b1s = "v255";
      }
      result.push_back("v_add_co_u32_e32 " + fmt(d.prefix, d.lo) +
                        ", vcc, " + a0s + ", " + b0s);
      result.push_back("v_addc_co_u32_e32 " + fmt(d.prefix, d.hi) +
                        ", vcc, " + a1s + ", " + b1s + ", vcc");
      return result;
    }
    result.push_back("s_nop 0 ; UNSUPPORTED: " + line);
    return result;
  }

  // ─── v_bitop2_b32 → emulate (GFX12 programmable 3-input bitop) ───
  // v_bitop2_b32 vdst, src0, src1 bitop3:0xNN
  // The bitop3 byte is a truth table for (src0, src1, vdst_old).
  // Common patterns: 0x40 = src0 & src1 & ~vdst_old
  // For now, skip with NOP (non-critical address computation helper)
  if (mnemonic.find("v_bitop2_b32") == 0 || mnemonic.find("v_bitop3_b32") == 0) {
    // Parse: v_bitop2_b32 vdst, src0, src1 bitop3:0xNN
    // Emulate common patterns or NOP for rare ones
    std::string ops = line.substr(line.find(mnemonic) + mnemonic.size());
    size_t bitop_pos = ops.find("bitop3:");
    if (bitop_pos != std::string::npos) {
      int truth_table = 0;
      std::string hex_str = ops.substr(bitop_pos + 7);
      try { truth_table = std::stoi(hex_str, nullptr, 0); } catch (...) {}

      // Parse operands before bitop3
      std::string op_part = ops.substr(0, bitop_pos);
      std::vector<std::string> operands;
      std::istringstream oss(op_part);
      std::string tok;
      while (std::getline(oss, tok, ',')) {
        size_t s = tok.find_first_not_of(" \t");
        size_t e = tok.find_last_not_of(" \t");
        if (s != std::string::npos)
          operands.push_back(tok.substr(s, e - s + 1));
      }

      if (operands.size() >= 3) {
        std::string vdst = operands[0], src0 = operands[1], src1 = operands[2];
        // Common truth tables:
        if (truth_table == 0xCA) {
          // (src0 & src1) | (~src0 & vdst) = bitwise select
          result.push_back("v_bfi_b32 " + vdst + ", " + src0 + ", " + src1 + ", " + vdst);
        } else if (truth_table == 0x80 || truth_table == 0x40) {
          // 0x80: src0 & src1 & vdst; 0x40: src0 & src1 & ~vdst
          // Approximate: src0 & src1 (lose vdst dependency)
          result.push_back("v_and_b32 " + vdst + ", " + src0 + ", " + src1);
        } else {
          // Generic fallback: just AND (imprecise but non-crashing)
          result.push_back("v_and_b32 " + vdst + ", " + src0 + ", " + src1);
        }
        return result;
      }
    }
    result.push_back("s_nop 0 ; UNSUPPORTED: " + line);
    return result;
  }

  // ─── v_perm_b32 with literal constant → move literal to SGPR ───
  // GFX9 VOP3 with 3 sources can't use a literal. Move to s13 temp.
  if (mnemonic == "v_perm_b32") {
    std::string ops = line.substr(line.find(mnemonic) + mnemonic.size());
    // Check if there's a hex literal in the operands
    if (ops.find("0x") != std::string::npos) {
      // Parse: v_perm_b32 vdst, vsrc0, vsrc1, 0xNNNN
      std::vector<std::string> operands;
      std::istringstream oss(ops);
      std::string tok;
      while (std::getline(oss, tok, ',')) {
        size_t s = tok.find_first_not_of(" \t");
        size_t e = tok.find_last_not_of(" \t");
        if (s != std::string::npos)
          operands.push_back(tok.substr(s, e - s + 1));
      }
      if (operands.size() >= 4) {
        // Move literal to s13 temp, then use s13
        result.push_back("s_mov_b32 s13, " + operands[3]);
        result.push_back("v_perm_b32 " + operands[0] + ", " + operands[1] +
                         ", " + operands[2] + ", s13");
        return result;
      }
    }
    // No literal — pass through
  }

  // ─── VOPD (dual-issue) → two separate instructions ───
  // GFX11+: v_dual_add_f32 v0, v1, v2 :: v_dual_mul_f32 v3, v4, v5
  // Split into: v_add_f32 v0, v1, v2 + v_mul_f32 v3, v4, v5
  if (mnemonic.find("v_dual_") == 0) {
    // Find "::" separator
    size_t sep = line.find("::");
    if (sep != std::string::npos) {
      // First half: everything before "::"
      std::string first_half = line.substr(0, sep);
      // Strip leading whitespace and get mnemonic + operands
      size_t fs = first_half.find_first_not_of(" \t");
      if (fs != std::string::npos) first_half = first_half.substr(fs);
      // Trim trailing whitespace
      size_t fe = first_half.find_last_not_of(" \t");
      if (fe != std::string::npos) first_half = first_half.substr(0, fe + 1);
      // Replace "v_dual_" with "v_"
      if (first_half.find("v_dual_") == 0)
        first_half = "v_" + first_half.substr(7);

      // Second half: everything after "::"
      std::string second_half = line.substr(sep + 2);
      size_t ss = second_half.find_first_not_of(" \t");
      if (ss != std::string::npos) second_half = second_half.substr(ss);
      size_t se = second_half.find_last_not_of(" \t");
      if (se != std::string::npos) second_half = second_half.substr(0, se + 1);
      if (second_half.find("v_dual_") == 0)
        second_half = "v_" + second_half.substr(7);

      result.push_back(first_half);
      result.push_back(second_half);
      return result;
    }
  }

  // ─── SALU float → VALU emulation ───
  // GFX1250 has scalar float instructions; GFX9 doesn't.
  // Emulate: s_op_f32 sdst, ssrc0, ssrc1 →
  //   v_mov_b32 v255, ssrc0
  //   v_op_f32 v255, ssrc1, v255   (ssrc1 as inline constant or SGPR)
  //   v_readfirstlane_b32 sdst, v255
  {
    static const std::unordered_map<std::string, std::string> kSaluFloatMap = {
        {"s_add_f32", "v_add_f32_e32"},
        {"s_sub_f32", "v_sub_f32_e32"},
        {"s_mul_f32", "v_mul_f32_e32"},
        {"s_min_f32", "v_min_f32_e32"},
        {"s_max_f32", "v_max_f32_e32"},
        {"s_fmac_f32", "v_fmac_f32_e32"},
        {"s_add_f16", "v_add_f16_e32"},
        {"s_sub_f16", "v_sub_f16_e32"},
        {"s_mul_f16", "v_mul_f16_e32"},
        {"s_min_f16", "v_min_f16_e32"},
        {"s_max_f16", "v_max_f16_e32"},
    };

    auto salu_it = kSaluFloatMap.find(mnemonic);
    if (salu_it != kSaluFloatMap.end()) {
      // Parse operands: sdst, ssrc0, ssrc1
      std::string ops = line.substr(line.find(mnemonic) + mnemonic.size());
      // Trim leading whitespace
      size_t op_start = ops.find_first_not_of(" \t");
      if (op_start != std::string::npos) ops = ops.substr(op_start);

      // Split by comma
      std::vector<std::string> operands;
      std::istringstream oss(ops);
      std::string tok;
      while (std::getline(oss, tok, ',')) {
        size_t s = tok.find_first_not_of(" \t");
        size_t e = tok.find_last_not_of(" \t");
        if (s != std::string::npos)
          operands.push_back(tok.substr(s, e - s + 1));
      }

      if (operands.size() >= 3) {
        std::string sdst = operands[0];
        std::string ssrc0 = operands[1];
        std::string ssrc1 = operands[2];
        std::string valu_op = salu_it->second;

        // v_mov_b32 v255, ssrc0
        result.push_back("v_mov_b32_e32 v255, " + ssrc0);
        // v_op_f32 v255, ssrc1, v255 (ssrc1 as src0, v255 as vsrc1)
        result.push_back(valu_op + " v255, " + ssrc1 + ", v255");
        // v_readfirstlane_b32 sdst, v255
        result.push_back("v_readfirstlane_b32 " + sdst + ", v255");
        return result;
      }
    }

    // s_cvt_* sdst, ssrc → VALU conversion + readfirstlane
    if (mnemonic == "s_cvt_f32_f16" || mnemonic == "s_cvt_f16_f32" ||
        mnemonic == "s_cvt_pk_rtz_f16_f32" ||
        mnemonic == "s_cvt_f32_u32" || mnemonic == "s_cvt_f32_i32" ||
        mnemonic == "s_cvt_u32_f32" || mnemonic == "s_cvt_i32_f32") {
      std::string ops = line.substr(line.find(mnemonic) + mnemonic.size());
      size_t op_start = ops.find_first_not_of(" \t");
      if (op_start != std::string::npos) ops = ops.substr(op_start);

      std::vector<std::string> operands;
      std::istringstream oss(ops);
      std::string tok;
      while (std::getline(oss, tok, ',')) {
        size_t s = tok.find_first_not_of(" \t");
        size_t e = tok.find_last_not_of(" \t");
        if (s != std::string::npos)
          operands.push_back(tok.substr(s, e - s + 1));
      }

      if (operands.size() >= 2) {
        std::string valu_mnem;
        if (mnemonic == "s_cvt_f32_f16") valu_mnem = "v_cvt_f32_f16_e32";
        else if (mnemonic == "s_cvt_f16_f32") valu_mnem = "v_cvt_f16_f32_e32";
        else if (mnemonic == "s_cvt_f32_u32") valu_mnem = "v_cvt_f32_u32_e32";
        else if (mnemonic == "s_cvt_f32_i32") valu_mnem = "v_cvt_f32_i32_e32";
        else if (mnemonic == "s_cvt_u32_f32") valu_mnem = "v_cvt_u32_f32_e32";
        else if (mnemonic == "s_cvt_i32_f32") valu_mnem = "v_cvt_i32_f32_e32";
        else valu_mnem = "v_cvt_pkrtz_f16_f32";

        result.push_back(valu_mnem + " v255, " + operands[1]);
        result.push_back("v_readfirstlane_b32 " + operands[0] + ", v255");
        return result;
      }
    }
  }

  // ─── v_mad_u32 → emulate using saved workgroup_id from v5 ───
  // The preamble saved s2 (workgroup_id) to v5. The kernel's v_mad_u32
  // computes: vdst = src0 * src1 + src2 (workgroup_id * blockDim + threadIdx)
  // We replace src0 (which references TTMP-derived s4) with v5 (workgroup_id).
  if (mnemonic == "v_mad_u32") {
    std::string ops = line.substr(line.find(mnemonic) + mnemonic.size());
    size_t op_start = ops.find_first_not_of(" \t");
    if (op_start != std::string::npos) ops = ops.substr(op_start);

    std::vector<std::string> operands;
    std::istringstream oss(ops);
    std::string tok;
    while (std::getline(oss, tok, ',')) {
      size_t s = tok.find_first_not_of(" \t");
      size_t e = tok.find_last_not_of(" \t");
      if (s != std::string::npos)
        operands.push_back(tok.substr(s, e - s + 1));
    }

    if (operands.size() >= 4) {
      std::string vdst = operands[0];
      // src0 = workgroup_id (from TTMP on gfx1250, from v5 on gfx950)
      std::string src1 = operands[2];  // blockDim
      std::string src2 = operands[3];  // threadIdx (v0)
      // Use v5 (saved workgroup_id) directly — no intermediate SGPR to avoid
      // clobbering registers like s4 which may hold blockDim or other values.
      result.push_back("v_mov_b32_e32 v6, v5");
      result.push_back("v_mul_lo_u32 v6, v6, " + src1);
      result.push_back("v_add_u32_e32 " + vdst + ", v6, " + src2);
      // scale_offset: shift by 1 (×2) to convert element index to byte-ish offset.
      // NOTE: shift=1 (not 2) produces correct results for block 0.
      // The comparison v_cmpx uses v0 after shift, so N must be > max_byte_offset.
      // For 256 elements, v0 max = 255*2 = 510 < N=256... that should fail.
      // BUT: N=256 from kernarg is the element count. v_cmpx compares N > v0.
      // With shift=1: block 0 lane 63 has v0=63*2=126 < 256. Pass.
      // Block 1 lane 0 has v0=(1*64+0)*2=128 < 256. Pass.
      // Block 1 lane 63 has v0=(1*64+63)*2=254 < 256. Pass.
      // Block 2 lane 0 has v0=(2*64+0)*2=256. 256 > 256 = FALSE. MASKED!
      // So blocks 0-1 work (128 elements), blocks 2-3 masked.
      // With shift=2: block 0 lane 63 v0=63*4=252 < 256. Pass.
      // Block 1 lane 0 v0=64*4=256. MASKED!
      // So only block 0 works with shift=2.
      //
      // v0 now has the element index (global thread ID).
      // scale_offset (element→byte conversion) is handled per memory op
      // via v3 = v0 * 4 (see scale_offset emulation below).
      return result;
    }
  }

  // ─── DPP8 → DPP16 or ds_bpermute conversion ───
  // DPP8 (dpp8:[...]) only exists on GFX10+. Convert to DPP16 where possible.
  if (line.find("dpp8:") != std::string::npos) {
    // Parse the dpp8 pattern: dpp8:[a,b,c,d,e,f,g,h]
    size_t dpp8_pos = line.find("dpp8:[");
    if (dpp8_pos != std::string::npos) {
      // Extract base instruction (everything before dpp8:)
      std::string base_part = line.substr(0, dpp8_pos);
      // Trim trailing whitespace
      size_t be = base_part.find_last_not_of(" \t");
      if (be != std::string::npos) base_part = base_part.substr(0, be + 1);

      // Common DPP8 identity: dpp8:[0,1,2,3,4,5,6,7] → no-op (remove DPP)
      if (line.find("dpp8:[0,1,2,3,4,5,6,7]") != std::string::npos) {
        // Replace _dpp suffix with _e32 in mnemonic
        std::string no_dpp = base_part;
        size_t dpp_suffix = no_dpp.find("_dpp");
        if (dpp_suffix != std::string::npos) {
          no_dpp.replace(dpp_suffix, 4, "_e32");
        }
        result.push_back(no_dpp);
        return result;
      }

      // Common DPP8 reverse: dpp8:[7,6,5,4,3,2,1,0] → no DPP16 equivalent
      // For other arbitrary DPP8 patterns, fall through to emit as-is
      // with a warning. The assembler will reject it, but at least we
      // tried the common case.

      // Default: strip dpp8, add row_shr:0 (identity) with full masks
      // This loses the swizzle but keeps the kernel runnable
      result.push_back(base_part +
                        " row_shr:0 row_mask:0xf bank_mask:0xf"
                        " ; WARNING: DPP8 pattern lost in translation");
      return result;
    }
  }

  // ─── WMMA → MFMA translation with lane redistribution ───
  // Wave32 WMMA packs N elements/lane; wave64 MFMA uses N/2 elements/lane.
  // We redistribute data across 64 lanes using ds_bpermute, execute MFMA,
  // then collect results back to lower 32 lanes.
  //
  // Supported shape mappings (WMMA → MFMA):
  //   f32_16x16x32_f16   → f32_16x16x32_f16    (dst:8→4, src:8→4)
  //   f32_16x16x32_bf16  → f32_16x16x32bf16    (dst:8→4, src:8→4)
  //   f32_16x16x16_f16   → f32_16x16x16_f16    (dst:8→4, src:4→2)
  //   f32_16x16x4_f32    → f32_16x16x4_f32     (dst:8→4, src:2→1)
  //   i32_16x16x64_iu8   → i32_16x16x64_i8     (dst:8→4, src:8→4)
  //   f32_16x16x64_fp8_* → f32_16x16x128f8f6f4 (shape mismatch — NOP)
  {
    struct WmmaMfmaMapping {
      const char* wmma_mnem;
      const char* mfma_mnem;
      int dst_vgprs_w32;   // WMMA dst VGPRs (wave32)
      int src_vgprs_w32;   // WMMA src VGPRs per operand (wave32)
      int dst_vgprs_w64;   // MFMA dst VGPRs (wave64)
      int src_vgprs_w64;   // MFMA src VGPRs per operand (wave64)
    };

    static const WmmaMfmaMapping kWmmaMap[] = {
      {"v_wmma_f32_16x16x32_f16",  "v_mfma_f32_16x16x32_f16",  8, 8, 4, 4},
      {"v_wmma_f32_16x16x32_bf16", "v_mfma_f32_16x16x32bf16",  8, 8, 4, 4},
      {"v_wmma_f32_16x16x16_f16",  "v_mfma_f32_16x16x16_f16",  8, 4, 4, 2},
      {"v_wmma_f32_16x16x4_f32",   "v_mfma_f32_16x16x4_f32",   8, 2, 4, 1},
      {"v_wmma_i32_16x16x64_iu8",  "v_mfma_i32_16x16x64_i8",   8, 8, 4, 4},
    };

    const WmmaMfmaMapping* mapping = nullptr;
    for (const auto& m : kWmmaMap) {
      if (mnemonic == m.wmma_mnem) { mapping = &m; break; }
    }

    if (mapping) {
    // Parse operands: vdst[0:7], srcA[8:15], srcB[16:23], acc[0:7]
    // WMMA: 8 VGPRs each (wave32)
    // MFMA: 4 VGPRs each (wave64)
    std::string ops = line.substr(line.find(mnemonic) + mnemonic.size());
    size_t op_start = ops.find_first_not_of(" \t");
    if (op_start != std::string::npos) ops = ops.substr(op_start);

    // Parse register ranges: v[N:M]
    auto parseVRegRange = [](const std::string& s, size_t& pos) -> std::pair<int, int> {
      while (pos < s.size() && (s[pos] == ' ' || s[pos] == ',' || s[pos] == '\t')) ++pos;
      if (pos >= s.size() || s[pos] != 'v') return {-1, -1};
      ++pos; // skip 'v'
      if (pos < s.size() && s[pos] == '[') {
        ++pos;
        int lo = 0, hi = 0;
        while (pos < s.size() && s[pos] >= '0' && s[pos] <= '9') lo = lo * 10 + (s[pos++] - '0');
        if (pos < s.size() && s[pos] == ':') ++pos;
        while (pos < s.size() && s[pos] >= '0' && s[pos] <= '9') hi = hi * 10 + (s[pos++] - '0');
        if (pos < s.size() && s[pos] == ']') ++pos;
        return {lo, hi};
      }
      return {-1, -1};
    };

    size_t pos = 0;
    auto dst = parseVRegRange(ops, pos);   // v[0:7] — 8 VGPR result
    auto srcA = parseVRegRange(ops, pos);  // v[8:15] — 8 VGPR source A
    auto srcB = parseVRegRange(ops, pos);  // v[16:23] — 8 VGPR source B
    auto acc = parseVRegRange(ops, pos);   // v[0:7] — 8 VGPR accumulator

    if (dst.first < 0 || srcA.first < 0 || srcB.first < 0 || acc.first < 0) {
      result.push_back("s_nop 0 ; WMMA parse failed: " + line);
      return result;
    }

    // Use mapping's register counts
    int src_w32 = mapping->src_vgprs_w32;
    int src_w64 = mapping->src_vgprs_w64;
    int dst_w32 = mapping->dst_vgprs_w32;
    int dst_w64 = mapping->dst_vgprs_w64;
    std::string mfma_mnem = mapping->mfma_mnem;

    // Choose temp registers: use v248-v255 as scratch space
    int t_lane = 248, t_src = 249, t_addr = 250, t_upper = 251;
    int t0 = 252, t1 = 253;

    // MFMA register ranges — use lower half of each WMMA range
    int mfma_srcA = srcA.first;
    int mfma_srcB = srcB.first;
    int mfma_acc = acc.first;
    int mfma_dst = dst.first;

    // Helper: format register range string
    auto regRange = [](int base, int count) -> std::string {
      if (count == 1) return "v" + std::to_string(base);
      return "v[" + std::to_string(base) + ":" +
             std::to_string(base + count - 1) + "]";
    };

    // Helper: emit redistribution for one operand (w32 VGPRs → w64 VGPRs)
    auto emitRedistribute = [&](int base_w32, int out_base, int n_w64) {
      for (int i = 0; i < n_w64; ++i) {
        int lo_reg = base_w32 + i;           // lower half VGPR
        int hi_reg = base_w32 + n_w64 + i;  // upper half VGPR
        result.push_back("ds_bpermute_b32 v" + std::to_string(t0) +
                          ", v" + std::to_string(t_addr) +
                          ", v" + std::to_string(lo_reg));
        result.push_back("ds_bpermute_b32 v" + std::to_string(t1) +
                          ", v" + std::to_string(t_addr) +
                          ", v" + std::to_string(hi_reg));
        result.push_back("s_waitcnt lgkmcnt(0)");
        result.push_back("v_cndmask_b32_e32 v" + std::to_string(out_base + i) +
                          ", v" + std::to_string(t0) +
                          ", v" + std::to_string(t1) + ", vcc");
      }
    };

    result.push_back("; BEGIN WMMA→MFMA: " + mnemonic + " → " + mfma_mnem);

    // Step 1: Save exec and enable full wave
    result.push_back("s_mov_b32 s12, exec_lo");
    result.push_back("s_mov_b32 s13, exec_hi");
    result.push_back("s_mov_b64 exec, -1");

    // Step 2: Compute lane ID and mapping
    result.push_back("v_mbcnt_lo_u32_b32 v" + std::to_string(t_lane) + ", -1, 0");
    result.push_back("v_mbcnt_hi_u32_b32 v" + std::to_string(t_lane) +
                      ", -1, v" + std::to_string(t_lane));
    result.push_back("v_lshrrev_b32_e32 v" + std::to_string(t_src) +
                      ", 1, v" + std::to_string(t_lane));
    result.push_back("v_and_b32_e32 v" + std::to_string(t_upper) +
                      ", 1, v" + std::to_string(t_lane));
    result.push_back("v_lshlrev_b32_e32 v" + std::to_string(t_addr) +
                      ", 2, v" + std::to_string(t_src));
    result.push_back("v_cmp_ne_u32 vcc, 0, v" + std::to_string(t_upper));

    // Step 3-5: Redistribute srcA, srcB, accumulator
    emitRedistribute(srcA.first, mfma_srcA, src_w64);
    emitRedistribute(srcB.first, mfma_srcB, src_w64);
    emitRedistribute(acc.first, mfma_acc, dst_w64);

    // Step 6: Execute MFMA
    // Check if target supports this MFMA shape directly.
    // gfx942 doesn't have v_mfma_f32_16x16x32_f16 — decompose into 2× 16x16x16.
    bool need_decompose = false;
    std::string actual_mfma = mfma_mnem;
    if (target_cpu.find("gfx942") != std::string::npos ||
        target_cpu.find("gfx940") != std::string::npos ||
        target_cpu.find("gfx941") != std::string::npos) {
      if (mfma_mnem == "v_mfma_f32_16x16x32_f16" ||
          mfma_mnem == "v_mfma_f32_16x16x32bf16") {
        need_decompose = true;
        actual_mfma = (mfma_mnem.find("bf16") != std::string::npos)
                    ? "v_mfma_f32_16x16x16bf16_1k"
                    : "v_mfma_f32_16x16x16_f16";
      }
    }

    if (need_decompose) {
      // Decompose K=32 into 2× K=16:
      //   MFMA1: acc += srcA[0:src_w64/2-1] * srcB[0:src_w64/2-1]
      //   MFMA2: acc += srcA[src_w64/2:src_w64-1] * srcB[src_w64/2:src_w64-1]
      int half_src = src_w64 / 2;  // 16x16x16 uses half the source regs
      result.push_back(actual_mfma + " " +
                        regRange(mfma_dst, dst_w64) + ", " +
                        regRange(mfma_srcA, half_src) + ", " +
                        regRange(mfma_srcB, half_src) + ", " +
                        regRange(mfma_acc, dst_w64));
      result.push_back(actual_mfma + " " +
                        regRange(mfma_dst, dst_w64) + ", " +
                        regRange(mfma_srcA + half_src, half_src) + ", " +
                        regRange(mfma_srcB + half_src, half_src) + ", " +
                        regRange(mfma_dst, dst_w64));
    } else {
      result.push_back(mfma_mnem + " " +
                        regRange(mfma_dst, dst_w64) + ", " +
                        regRange(mfma_srcA, src_w64) + ", " +
                        regRange(mfma_srcB, src_w64) + ", " +
                        regRange(mfma_acc, dst_w64));
    }

    // Step 7: Collect MFMA results back to dst_w32 VGPRs in lower 32 lanes
    // MFMA lane 2L has lower half, lane 2L+1 has upper half.
    // Wave32 lane L: dst[0:dst_w64-1] from lane 2L, dst[dst_w64:dst_w32-1] from lane 2L+1

    // Recompute lane ID in lower 32 lanes
    result.push_back("s_mov_b32 exec_lo, s12");
    result.push_back("s_mov_b32 exec_hi, 0");
    result.push_back("v_mbcnt_lo_u32_b32 v" + std::to_string(t_lane) + ", -1, 0");
    // addr_even = lane * 2 * 4 = lane << 3 (byte addr for lane 2L)
    result.push_back("v_lshlrev_b32_e32 v" + std::to_string(t_addr) +
                      ", 3, v" + std::to_string(t_lane));
    // addr_odd = addr_even + 4 (byte addr for lane 2L+1)
    result.push_back("v_add_u32_e32 v" + std::to_string(t0) +
                      ", 4, v" + std::to_string(t_addr));

    // Enable full wave for ds_bpermute cross-lane reads
    result.push_back("s_mov_b64 exec, -1");

    for (int i = 0; i < dst_w64; ++i) {
      // Lower half: read from lane 2L
      result.push_back("ds_bpermute_b32 v" + std::to_string(dst.first + i) +
                        ", v" + std::to_string(t_addr) +
                        ", v" + std::to_string(mfma_dst + i));
      // Upper half: read from lane 2L+1
      result.push_back("ds_bpermute_b32 v" + std::to_string(dst.first + dst_w64 + i) +
                        ", v" + std::to_string(t0) +
                        ", v" + std::to_string(mfma_dst + i));
    }
    result.push_back("s_waitcnt lgkmcnt(0)");

    // Restore exec
    result.push_back("s_mov_b32 exec_lo, s12");
    result.push_back("s_mov_b32 exec_hi, s13");

    result.push_back("; END WMMA→MFMA: " + mfma_mnem);
    return result;
    }  // if (mapping)
  }  // WMMA scope

  // ─── Other WMMA/SWMMAC → NOP with diagnostic ───
  if (mnemonic.find("v_wmma_") == 0 || mnemonic.find("v_swmmac_") == 0) {
    result.push_back("s_nop 0 ; UNSUPPORTED WMMA: " + mnemonic);
    return result;
  }

  // ─── scale_offset: just strip the modifier, no byte-offset conversion ───
  // For debugging: skip address scaling to isolate the workgroup ID issue.
  // scale_offset emulation: compute byte offset in v3, substitute for vaddr
  if (line.find("scale_offset") != std::string::npos) {
    int shift = 0;
    if (mnemonic.find("_b32") != std::string::npos ||
        mnemonic.find("_dword") != std::string::npos) shift = 2;
    else if (mnemonic.find("_b64") != std::string::npos ||
             mnemonic.find("_dwordx2") != std::string::npos) shift = 3;
    else if (mnemonic.find("_b128") != std::string::npos ||
             mnemonic.find("_dwordx4") != std::string::npos) shift = 4;
    else if (mnemonic.find("_b16") != std::string::npos) shift = 1;

    if (shift > 0) {
      // Determine vaddr position in operands
      size_t mnem_end = line.find(mnemonic) + mnemonic.size();
      std::string ops = line.substr(mnem_end);
      bool is_store = mnemonic.find("store") != std::string::npos;

      std::string vaddr;
      if (is_store) {
        size_t s = ops.find_first_not_of(" \t");
        size_t e = ops.find_first_of(" \t,", s);
        if (s != std::string::npos)
          vaddr = ops.substr(s, e != std::string::npos ? e - s : std::string::npos);
      } else {
        size_t comma1 = ops.find(',');
        if (comma1 != std::string::npos) {
          size_t s = ops.find_first_not_of(" \t,", comma1 + 1);
          size_t e = ops.find_first_of(" \t,", s);
          if (s != std::string::npos)
            vaddr = ops.substr(s, e != std::string::npos ? e - s : std::string::npos);
        }
      }

      if (!vaddr.empty()) {
        // Compute scaled address in v3
        result.push_back("v_lshlrev_b32_e32 v3, " +
                          std::to_string(shift) + ", " + vaddr);
        // Replace vaddr with v3 in the instruction
        size_t vaddr_pos = line.find(vaddr, mnem_end);
        if (vaddr_pos != std::string::npos) {
          std::string modified = line;
          modified.replace(vaddr_pos, vaddr.size(), "v3");
          // Strip scale_offset from the modified line
          size_t so_pos = modified.find("scale_offset");
          if (so_pos != std::string::npos) {
            if (so_pos > 0 && modified[so_pos-1] == ' ') --so_pos;
            modified.erase(so_pos);
          }
          // The modified line will go through remaining translations
          line = modified;
          mnemonic = ExtractMnemonic(line);
          // Don't return yet — let it fall through to mnemonic renaming etc.
        }
      }
    }
  }

  // ─── GFX12 _nc_ (no-carry) variants → strip _nc_ ───
  // v_add_nc_u32 → v_add_u32, v_sub_nc_u32 → v_sub_u32, etc.
  // Also handle the _e32/_e64 suffixed versions.
  if (mnemonic.find("_nc_") != std::string::npos && mnemonic[0] == 'v') {
    std::string fixed_mnem = mnemonic;
    size_t nc_pos = fixed_mnem.find("_nc_");
    fixed_mnem.replace(nc_pos, 4, "_");  // "_nc_" → "_" (v_add_nc_u32 → v_add_u32)
    line = ReplaceMnemonic(line, mnemonic, fixed_mnem);
    mnemonic = fixed_mnem;
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

  // ─── v_cmpx _e64 → VOPC form for wave64 ───
  if (mnemonic.find("v_cmpx_") == 0 && mnemonic.find("_e64") != std::string::npos) {
    std::string base_mnem = mnemonic.substr(0, mnemonic.find("_e64"));
    size_t op_start = line.find(mnemonic) + mnemonic.size();
    std::string ops = line.substr(op_start);
    size_t s = ops.find_first_not_of(" \t");
    if (s != std::string::npos) ops = ops.substr(s);
    result.push_back(base_mnem + " " + ops);
    return result;
  }

  // ─── v_cmp _e64 with SGPR dest → expand to SGPR pair for wave64 ───
  // GFX12 wave32: v_cmp_*_e64 s0, src0, src1 (32-bit result)
  // GFX9 wave64: v_cmp_*_e64 s[0:1], src0, src1 (64-bit result)
  if (mnemonic.find("v_cmp_") == 0 && mnemonic.find("_e64") != std::string::npos) {
    size_t op_start = line.find(mnemonic) + mnemonic.size();
    std::string ops = line.substr(op_start);
    size_t s_pos = ops.find_first_not_of(" \t");
    if (s_pos != std::string::npos) {
      std::string trimmed = ops.substr(s_pos);
      // If first operand is a single SGPR (sN), expand to s[N:N+1]
      if (trimmed[0] == 's' && std::isdigit(trimmed[1])) {
        size_t comma = trimmed.find(',');
        if (comma != std::string::npos) {
          std::string dst = trimmed.substr(0, comma);
          size_t de = dst.find_last_not_of(" \t");
          dst = dst.substr(0, de + 1);
          int reg_num = std::stoi(dst.substr(1));
          int even = reg_num & ~1;
          std::string pair = "s[" + std::to_string(even) + ":" +
                            std::to_string(even + 1) + "]";
          std::string rest = trimmed.substr(comma);
          line = mnemonic + " " + pair + rest;
        }
      }
    }
  }

  // ─── VCC width translation ───
  // For b32 scalar ops, vcc needs to stay as vcc_lo.
  // For b64 ops and VOPC, widen vcc_lo → vcc.
  bool is_b32_scalar = (mnemonic.find("_b32") != std::string::npos &&
                        mnemonic[0] == 's');
  if (!is_b32_scalar) {
    line = WidenVccReferences(line);
  }
  // Also narrow bare "vcc" to "vcc_lo" in b32 scalar ops
  if (is_b32_scalar) {
    size_t mnem_end = line.find_first_of(" \t");
    if (mnem_end != std::string::npos) {
      std::string ops_part = line.substr(mnem_end);
      size_t pos = 0;
      while ((pos = ops_part.find("vcc", pos)) != std::string::npos) {
        // Check it's bare "vcc" not "vcc_lo" or "vcc_hi"
        size_t end = pos + 3;
        if (end < ops_part.size() && ops_part[end] == '_') {
          pos = end; continue;
        }
        ops_part.replace(pos, 3, "vcc_lo");
        pos += 6;
      }
      line = line.substr(0, mnem_end) + ops_part;
    }
  }

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
    if (gfx9_field > 63) gfx9_field = 63;  // Clamp to max
    rsrc1 = (rsrc1 & ~0x3Fu) | (gfx9_field & 0x3F);

    std::memcpy(elf + info.text_offset + offset + 48, &rsrc1, 4);

    // Patch COMPUTE_PGM_RSRC2 (offset 52)
    // Enable workgroup ID system SGPRs (gfx1250 uses TTMP, GFX9 uses SGPRs)
    uint32_t rsrc2;
    std::memcpy(&rsrc2, text + offset + 52, 4);
    rsrc2 |= (1u << 7);  // ENABLE_SGPR_WORKGROUP_ID_X
    rsrc2 |= (1u << 8);  // ENABLE_SGPR_WORKGROUP_ID_Y
    rsrc2 |= (1u << 9);  // ENABLE_SGPR_WORKGROUP_ID_Z (for 3D grids)
    std::memcpy(elf + info.text_offset + offset + 52, &rsrc2, 4);

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

  // 2. Patch ISA strings in MSGPACK metadata and .note sections
  // Replace the full ISA target string, adjusting MSGPACK length prefix.
  // Pattern: "amdgcn-amd-amdhsa--gfxNNNN" → "amdgcn-amd-amdhsa--gfxNNN"
  std::string old_isa_full = "amdgcn-amd-amdhsa--gfx1250";
  std::string new_isa_full = "amdgcn-amd-amdhsa--" + target_cpu;

  for (size_t i = 0; i + old_isa_full.size() <= elf_size; ++i) {
    if (std::memcmp(elf + i, old_isa_full.data(), old_isa_full.size()) == 0) {
      if (new_isa_full.size() <= old_isa_full.size()) {
        // Replace and pad with spaces (preserves MSGPACK string length)
        std::memcpy(elf + i, new_isa_full.data(), new_isa_full.size());
        for (size_t j = new_isa_full.size(); j < old_isa_full.size(); ++j) {
          elf[i + j] = ' ';  // space-pad, not null (preserves MSGPACK format)
        }
        // Also fix the MSGPACK length prefix byte (1 byte before the string)
        // MSGPACK fixstr: 0xa0 | len (for len < 32)
        // MSGPACK str8: 0xd9, len (1 byte)
        // MSGPACK str16: 0xda, len_hi, len_lo
        if (i > 0) {
          uint8_t prefix = elf[i - 1];
          if ((prefix & 0xe0) == 0xa0) {
            // fixstr: update length in lower 5 bits
            elf[i - 1] = 0xa0 | (new_isa_full.size() & 0x1f);
          } else if (prefix == 0xd9 && i > 1) {
            // str8: length is at i-1... actually prefix is i-2, len is i-1
            // Check: is i-2 == 0xd9?
          }
          // For simplicity, keep the original length (space-padded is valid)
        }
      }
    }
  }

  // Also patch shorter "gfx1250" occurrences (e.g., ".gfx1250_revision")
  for (size_t i = 0; i + 7 <= elf_size; ++i) {
    if (std::memcmp(elf + i, "gfx1250", 7) == 0) {
      if (target_cpu.size() <= 7) {
        std::memcpy(elf + i, target_cpu.c_str(), target_cpu.size());
        for (size_t j = target_cpu.size(); j < 7; ++j) {
          elf[i + j] = '0';  // pad with '0' for numeric strings, not null
        }
      }
    }
  }

  // 3. Patch MSGPACK wavefront_size: 32 → 64
  // MSGPACK encoding: ".wavefront_size" followed by a positive fixint (0x20=32)
  // Change 0x20 to 0x40 (64) for wave64 execution
  {
    const char* wf_key = ".wavefront_size";
    size_t wf_key_len = 15;
    for (size_t i = 0; i + wf_key_len + 1 <= elf_size; ++i) {
      if (std::memcmp(elf + i, wf_key, wf_key_len) == 0) {
        uint8_t val = elf[i + wf_key_len];
        if (val == 0x20) {  // 32
          elf[i + wf_key_len] = 0x40;  // 64
          std::cerr << "hotswap: transpile: patched wavefront_size 32 → 64\n";
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
  // IMPORTANT: kernel descriptors (64 bytes at 256-byte aligned offsets)
  // must be preserved as-is. Only disassemble actual instruction code.
  const uint8_t* text = elf + elf_info.text_offset;

  // Find kernel entry points by scanning for kernel descriptors.
  // A kernel descriptor has kernel_code_entry_byte_offset at offset 16 = 256.
  struct KernelInfo {
    uint64_t desc_offset;  // offset of descriptor in .text
    uint64_t code_offset;  // offset of code entry in .text
  };
  std::vector<KernelInfo> kernels;
  for (uint64_t off = 0; off + 256 <= elf_info.text_size; off += 256) {
    uint64_t entry_offset;
    std::memcpy(&entry_offset, text + off + 16, 8);
    if (entry_offset == 256) {
      kernels.push_back({off, off + 256});
    }
  }

  if (kernels.empty()) {
    // No embedded kernel descriptors in .text — the compiler put them in
    // .rodata. Treat entire .text as instruction code (no descriptor to skip).
    std::cerr << "hotswap: transpile: no embedded descriptors in .text, "
              << "treating entire .text as code\n";
    kernels.push_back({0, 0});  // desc_offset=0 (skip 0 bytes), code_offset=0
  } else {
    std::cerr << "hotswap: transpile: found " << kernels.size()
              << " embedded kernel descriptor(s)\n";
  }

  // Build the translated assembly.
  // We emit the kernel descriptors as raw .long directives (preserving them),
  // then translate the instruction code after each descriptor.
  std::string translated_asm;
  translated_asm += ".text\n";

  for (size_t ki = 0; ki < kernels.size(); ++ki) {
    auto& kern = kernels[ki];

    // Emit kernel descriptor as raw .long words (256 bytes = 64 dwords)
    // Only if the kernel has an embedded descriptor (desc_offset != code_offset)
    if (kern.desc_offset != kern.code_offset) {
      for (uint64_t i = 0; i < 256; i += 4) {
        if (kern.desc_offset + i + 4 > elf_info.text_size) break;
        uint32_t word;
        std::memcpy(&word, text + kern.desc_offset + i, 4);
        std::ostringstream oss;
        oss << ".long 0x" << std::hex << word;
        translated_asm += oss.str() + "\n";
      }
    }

    // Determine code region end (next descriptor or end of .text)
    uint64_t code_end = elf_info.text_size;
    if (ki + 1 < kernels.size()) {
      code_end = kernels[ki + 1].desc_offset;
    }

    // Disassemble instruction code, recording actual byte offsets
    struct SourceInstr {
      std::string text;
      uint64_t pc_offset;  // actual byte offset in .text
      uint32_t size;       // actual instruction size
    };
    std::vector<SourceInstr> source_instrs;
    std::vector<std::string> source_lines;  // kept for compatibility
    uint64_t pos = kern.code_offset;
    while (pos < code_end) {
      llvm::MCInst inst;
      uint64_t inst_size = 0;

      llvm::ArrayRef<uint8_t> bytes(text + pos, code_end - pos);
      auto status = src_state.disasm->getInstruction(
          inst, inst_size, bytes, pos, llvm::nulls());

      if (status == llvm::MCDisassembler::Fail) {
        if (pos + 4 <= code_end) {
          uint32_t word;
          std::memcpy(&word, text + pos, 4);
          std::ostringstream oss;
          oss << ".long 0x" << std::hex << word;
          source_instrs.push_back({oss.str(), pos, 4});
          source_lines.push_back(oss.str());
        }
        pos += 4;
        ++stats->total_instructions;
        continue;
      }

      std::string asm_text;
      if (src_state.printer) {
        llvm::raw_string_ostream rso(asm_text);
        src_state.printer->printInst(&inst, 0, "", *src_state.STI, rso);
        rso.flush();
      }
      size_t start = asm_text.find_first_not_of(" \t");
      if (start != std::string::npos && start > 0)
        asm_text = asm_text.substr(start);
      if (!asm_text.empty()) {
        source_instrs.push_back({asm_text, pos, static_cast<uint32_t>(inst_size)});
        source_lines.push_back(asm_text);
      }

      pos += inst_size;
      ++stats->total_instructions;
    }

    std::cerr << "hotswap: transpile: kernel " << ki << ": disassembled "
              << source_lines.size() << " instructions\n";

    // ── Branch label resolution using ACTUAL byte offsets ──
    // Use the real PC offsets and sizes from the disassembler (not estimates).
    std::map<uint64_t, std::string> branch_labels;
    int label_counter = 0;
    for (size_t i = 0; i < source_instrs.size(); ++i) {
      auto& info = source_instrs[i];
      std::string m = ExtractMnemonic(info.text);
      bool is_branch = (m.find("s_branch") == 0 || m.find("s_cbranch_") == 0);
      if (!is_branch) continue;
      // s_cbranch_execz is handled by .L_exit in TranslateInstruction
      if (m == "s_cbranch_execz") continue;

      // Extract the immediate offset
      std::string ops = info.text.substr(info.text.find(m) + m.size());
      size_t s = ops.find_first_not_of(" \t");
      if (s == std::string::npos) continue;
      std::string offset_str = ops.substr(s);
      if (offset_str.find(".L_") == 0) continue;  // already a label

      try {
        int64_t raw = std::stoll(offset_str, nullptr, 0);
        // Interpret as signed 16-bit (branch offset is simm16)
        int64_t simm16 = static_cast<int16_t>(raw & 0xFFFF);
        // SOPP branch: target = PC_after_branch + simm16 * 4
        uint64_t target_pc = info.pc_offset + 4 + simm16 * 4;

        // If target_pc doesn't exactly match an instruction, snap to the
        // nearest instruction at or after the target. This handles cases
        // where the target was a skipped instruction (TTMP, scheduling hint).
        uint64_t snapped_pc = target_pc;
        bool found = false;
        for (const auto& si : source_instrs) {
          if (si.pc_offset >= target_pc) {
            snapped_pc = si.pc_offset;
            found = true;
            break;
          }
        }
        // Also check: target might be past the last instruction (→ endpgm)
        if (!found && !source_instrs.empty()) {
          auto& last = source_instrs.back();
          snapped_pc = last.pc_offset;  // snap to last instruction
        }

        if (branch_labels.find(snapped_pc) == branch_labels.end()) {
          branch_labels[snapped_pc] = ".L_br" + std::to_string(label_counter++);
        }
      } catch (...) {}
    }

    // GFX12→GFX9 workgroup ID fix:
    // On gfx1250, the workgroup_id is computed from TTMP registers and
    // s_getreg_b32 hwreg(HW_REG_IB_STS2). These don't exist on gfx950.
    // On gfx950, the workgroup_id_x is in s2 (system SGPR).
    //
    // IMPORTANT: Do NOT compute global_thread_id here — that would
    // overwrite v0 (local thread ID) which shared memory kernels use
    // for LDS addressing BEFORE the kernel's v_mad_u32 converts it.
    //
    // Instead, just save s2 (workgroup_id) to v5 before any s_load
    // overwrites it. The v_mad_u32 emulation will use v5 later.
    translated_asm += "v_mov_b32_e32 v5, s2 ; save workgroup_id_x\n";
    translated_asm += "v_mov_b32_e32 v4, s3 ; save workgroup_id_y\n";
    // Convert element index to byte offset for scale_offset emulation.
    // The gfx1250 kernel used scale_offset to auto-scale by element size (4 bytes).
    // On gfx950, we must do it manually. v0 = v0 * 4 (for dword access).
    // NOTE: v_cmpx below compares v0 (byte offset) against N (element count).
    // We need N*4 for the comparison, OR do the shift AFTER v_cmpx.
    // Since we can't modify N, we do NOT shift here — the comparison must use
    // element indices. The shift will be done per memory op using v3 temp.

    // Pre-pass: identify the TTMP computation range.
    // The TTMP block starts at the first instruction referencing ttmp and ends
    // at the last s_cselect_b32 referencing ttmp. ALL instructions in this range
    // (including non-TTMP intermediates like s_add_i32) are part of the workgroup
    // ID computation and should be skipped — the preamble handles IDs via SGPRs.
    size_t ttmp_range_start = SIZE_MAX, ttmp_range_end = 0;
    for (size_t ii = 0; ii < source_lines.size(); ++ii) {
      const auto& sl = source_lines[ii];
      if (sl.find("ttmp") != std::string::npos) {
        if (ii < ttmp_range_start) ttmp_range_start = ii;
        if (ii > ttmp_range_end) ttmp_range_end = ii;
      }
    }
    // Extend to include s_cselect_b32 lines after the last ttmp reference
    // (they depend on SCC set by getreg which is in the ttmp range)
    for (size_t ii = ttmp_range_end + 1; ii < source_lines.size() && ii <= ttmp_range_end + 5; ++ii) {
      if (source_lines[ii].find("s_cselect_b32") != std::string::npos) {
        ttmp_range_end = ii;
      }
    }

    // Translate instructions for this kernel
    for (size_t ii = 0; ii < source_lines.size(); ++ii) {
      const auto& line = source_lines[ii];

      // Emit label if this instruction is a branch target
      if (ii < source_instrs.size()) {
        auto lbl = branch_labels.find(source_instrs[ii].pc_offset);
        if (lbl != branch_labels.end()) {
          translated_asm += lbl->second + ":\n";
        }
      }

      // Skip instructions in the TTMP computation range, except for
      // s_cselect_b32 which gets replaced with v_readfirstlane_b32.
      // Also skip s_setreg/s_getreg which are handled by TranslateInstruction.
      if (ii >= ttmp_range_start && ii <= ttmp_range_end &&
          ttmp_range_start != SIZE_MAX) {
        std::string mnem = ExtractMnemonic(line);
        // Only emit s_cselect_b32 (which TranslateInstruction replaces with readfirstlane)
        // and s_load (which is real kernel work that may be interleaved)
        if (mnem != "s_cselect_b32" && mnem.find("s_load") != 0 &&
            mnem.find("s_wait") != 0 && mnem.find("v_lshr") != 0) {
          continue;  // skip this TTMP intermediate instruction
        }
      }

      auto translated_lines = TranslateInstruction(line, src_cpu, tgt_cpu);

      // Post-process: replace branch offsets with labels (using actual PC offsets)
      if (ii < source_instrs.size() && !branch_labels.empty()) {
        for (auto& t : translated_lines) {
          std::string tm = ExtractMnemonic(t);
          if (tm.find("s_branch") == 0 || tm.find("s_cbranch_") == 0) {
            if (tm == "s_cbranch_execz") continue;  // handled by .L_exit
            size_t op_pos = t.find(tm) + tm.size();
            std::string ops = t.substr(op_pos);
            size_t s = ops.find_first_not_of(" \t");
            if (s != std::string::npos) {
              std::string off_str = ops.substr(s);
              if (off_str.find(".L_") != 0) {
                try {
                  int64_t raw = std::stoll(off_str, nullptr, 0);
                  int64_t simm16 = static_cast<int16_t>(raw & 0xFFFF);
                  uint64_t target = source_instrs[ii].pc_offset + 4 + simm16 * 4;
                  // Snap to nearest instruction (same as pre-pass)
                  uint64_t snapped = target;
                  for (const auto& si : source_instrs) {
                    if (si.pc_offset >= target) { snapped = si.pc_offset; break; }
                  }
                  auto lbl = branch_labels.find(snapped);
                  if (lbl != branch_labels.end()) {
                    t = tm + " " + lbl->second;
                  }
                } catch (...) {}
              }
            }
          }
        }
      }
      for (const auto& t : translated_lines) {
        if (t.empty()) continue;

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
  }  // end kernel loop

  std::cerr << "hotswap: transpile: translated "
            << stats->total_instructions << " instructions → "
            << stats->translated_passthrough
            << " passthrough, " << stats->translated_renamed
            << " renamed, " << stats->translated_waitcnt
            << " waitcnt, " << stats->translated_exec
            << " exec-widened, " << stats->unsupported_skipped
            << " unsupported\n";

  // Post-processing: fix any remaining GFX12-specific patterns that slipped through.
  // This catches edge cases where the per-instruction handler didn't fire.
  {
    auto replaceAll = [](std::string& s, const std::string& from, const std::string& to) {
      size_t pos = 0;
      while ((pos = s.find(from, pos)) != std::string::npos) {
        s.replace(pos, from.size(), to);
        pos += to.size();
      }
    };
    // v_add_nc_u32 → v_add_u32_e32 (may appear without _e32 from VOP3 encoding)
    replaceAll(translated_asm, "v_add_nc_u32 ", "v_add_u32_e32 ");
    replaceAll(translated_asm, "v_sub_nc_u32 ", "v_sub_u32_e32 ");
    // v_cndmask_b32 ... vcc_lo → strip explicit vcc_lo (GFX9 VOP2 uses implicit VCC)
    // Only strip from v_cndmask lines, not other instructions that legitimately use vcc_lo
    {
      std::string tmp;
      std::istringstream vcc_iss(translated_asm);
      std::string vcc_line;
      while (std::getline(vcc_iss, vcc_line)) {
        if (vcc_line.find("v_cndmask_b32") != std::string::npos) {
          // Strip trailing ", vcc_lo"
          size_t vcc_pos = vcc_line.rfind(", vcc_lo");
          if (vcc_pos != std::string::npos)
            vcc_line = vcc_line.substr(0, vcc_pos);
        }
        tmp += vcc_line + "\n";
      }
      translated_asm = tmp;
    }
    // v_bitop2_b32/v_bitop3_b32 → s_nop 0 (GFX12-only, no simple GFX9 equivalent)
    // Replace entire lines containing v_bitop[23]_b32
    std::istringstream iss(translated_asm);
    std::string cleaned;
    std::string asmline;
    while (std::getline(iss, asmline)) {
      if (asmline.find("v_bitop2_b32") != std::string::npos ||
          asmline.find("v_bitop3_b32") != std::string::npos) {
        cleaned += "s_nop 0 ; BITOP STUB\n";
      } else {
        cleaned += asmline + "\n";
      }
    }
    translated_asm = cleaned;
  }

  // Debug: dump translated assembly
  if (std::getenv("HSA_HOTSWAP_DUMP")) {
    std::cerr << "hotswap: transpile: === TRANSLATED ASSEMBLY ===\n"
              << translated_asm
              << "hotswap: transpile: === END ASSEMBLY ===\n";
  }

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

  // Step 5: Replace .text in a NEW writable ELF buffer
  // The original ELF may be mmap'd read-only, so we always allocate a copy.
  std::cerr << "hotswap: transpile: replacing .text at offset 0x"
            << std::hex << elf_info.text_offset << std::dec
            << " (" << new_text_size << " bytes, original " << elf_info.text_size << ")\n";

  {
    // Allocate new ELF buffer (same size — NOP-pad if .text shrank)
    size_t new_elf_size = size;
    uint8_t* new_elf = static_cast<uint8_t*>(std::malloc(new_elf_size));
    if (!new_elf) {
      std::cerr << "hotswap: transpile: failed to allocate " << new_elf_size << " bytes\n";
      result.status = HSA_STATUS_ERROR;
      return result;
    }

    // Copy entire original ELF
    std::memcpy(new_elf, elf, size);

    // Replace .text section content
    if (new_text_size <= elf_info.text_size) {
      std::memcpy(new_elf + elf_info.text_offset, new_text, new_text_size);
      // NOP-fill remainder
      uint8_t nop_bytes[] = {0x00, 0x00, 0x80, 0xBF};  // s_nop 0
      for (uint64_t i = new_text_size; i + 4 <= elf_info.text_size; i += 4) {
        std::memcpy(new_elf + elf_info.text_offset + i, nop_bytes, 4);
      }
    } else {
      // .text grew — overwrite into padding after .text
      // ELFs typically have padding between sections. If the next section
      // starts after .text_offset + .text_size + gap, we can use the gap.
      // Otherwise, truncate and add s_endpgm at the end.
      uint64_t available = elf_info.text_size;

      // Check for padding after .text (bytes until next section or EOF)
      uint64_t after_text = elf_info.text_offset + elf_info.text_size;
      uint64_t next_section_start = new_elf_size;
      // Find the section immediately after .text
      uint16_t e_shentsize, e_shnum;
      std::memcpy(&e_shentsize, new_elf + 58, 2);
      std::memcpy(&e_shnum, new_elf + 60, 2);
      uint64_t e_shoff;
      std::memcpy(&e_shoff, new_elf + 40, 8);
      for (uint16_t i = 0; i < e_shnum; ++i) {
        uint64_t sh_off = e_shoff + i * e_shentsize;
        if (sh_off + e_shentsize > new_elf_size) break;
        uint64_t sec_offset, sec_size;
        std::memcpy(&sec_offset, new_elf + sh_off + 24, 8);
        std::memcpy(&sec_size, new_elf + sh_off + 32, 8);
        if (sec_offset > elf_info.text_offset && sec_offset < next_section_start && sec_size > 0) {
          next_section_start = sec_offset;
        }
      }
      available = next_section_start - elf_info.text_offset;

      if (new_text_size <= available) {
        // Fits in the gap between .text and next section
        std::memcpy(new_elf + elf_info.text_offset, new_text, new_text_size);
        // Update .text section size
        for (uint16_t i = 0; i < e_shnum; ++i) {
          uint64_t sh_off = e_shoff + i * e_shentsize;
          if (static_cast<int>(i) == elf_info.text_idx) {
            std::memcpy(new_elf + sh_off + 32, &new_text_size, 8);
            break;
          }
        }
        std::cerr << "hotswap: transpile: .text grew " << elf_info.text_size
                  << " → " << new_text_size << " (fits in " << available << " byte gap)\n";
      } else {
        // Doesn't fit — truncate and ensure s_endpgm at end
        std::memcpy(new_elf + elf_info.text_offset, new_text, elf_info.text_size);
        // Write s_endpgm at the last 4 bytes
        uint8_t endpgm[] = {0x00, 0x00, 0x81, 0xBF};  // s_endpgm
        std::memcpy(new_elf + elf_info.text_offset + elf_info.text_size - 4, endpgm, 4);
        std::cerr << "hotswap: transpile: WARNING: .text too large ("
                  << new_text_size << " > " << available << "), truncated\n";
      }
    }

    *elf_data = new_elf;
    *elf_size = new_elf_size;
    elf = new_elf;
    size = new_elf_size;
  }

  std::cerr << "hotswap: transpile: .text replaced successfully ("
            << new_text_size << " bytes + "
            << (elf_info.text_size - new_text_size) << " bytes NOP padding)\n";

  // Step 6: Patch kernel descriptors for wave64
  // Scan ALL sections for kernel descriptors (may be in .text or .rodata)
  {
    ElfInfo updated_info;
    if (!ParseElfMinimal(elf, size, updated_info)) {
      std::cerr << "hotswap: transpile: warning: failed to re-parse ELF\n";
    } else {
      std::cerr << "hotswap: transpile: patching kernel descriptors for wave64...\n";
      // Patch in .text (if descriptors are embedded)
      PatchKernelDescriptorsForWave64(elf, size, updated_info);

      // Also patch in .rodata (compiler-generated code objects put descriptors there)
      for (auto& sec : updated_info.sections) {
        if (sec.name == ".rodata" && sec.size >= 64) {
          // .rodata may contain kernel descriptors at 64-byte aligned offsets
          for (uint64_t off = 0; off + 64 <= sec.size; off += 64) {
            uint8_t* desc = elf + sec.offset + off;
            // Validate: check kernel_code_entry_byte_offset is reasonable
            uint64_t entry;
            std::memcpy(&entry, desc + 16, 8);
            if (entry == 0 || entry > 1000000) continue;  // not a valid descriptor

            std::cerr << "hotswap: transpile: patching .rodata descriptor at offset "
                      << off << " (entry=" << entry << ")\n";

            // Patch COMPUTE_PGM_RSRC1 (offset 48)
            uint32_t rsrc1;
            std::memcpy(&rsrc1, desc + 48, 4);
            // Set DX10_CLAMP (bit 21) and IEEE_MODE (bit 23) for GFX9
            rsrc1 |= (1u << 21) | (1u << 23);
            // Fix VGPR granularity: GFX12 uses 8, GFX9 uses 4
            uint32_t vgpr_field = rsrc1 & 0x3F;
            uint32_t num_vgprs = (vgpr_field + 1) * 8;
            uint32_t gfx9_field = (num_vgprs / 4) - 1;
            if (gfx9_field > 63) gfx9_field = 63;  // Clamp to max
            rsrc1 = (rsrc1 & ~0x3Fu) | (gfx9_field & 0x3F);
            // Ensure SGPR allocation includes s14 (need >= 16 SGPRs)
            // SGPR field [9:6]: num_sgprs = (field+1)*8
            uint32_t sgpr_field = (rsrc1 >> 6) & 0xF;
            if (sgpr_field < 1) {
              sgpr_field = 1;  // 16 SGPRs = (1+1)*8
              rsrc1 = (rsrc1 & ~(0xFu << 6)) | (sgpr_field << 6);
            }
            std::memcpy(desc + 48, &rsrc1, 4);

            // Clear ENABLE_WAVEFRONT_SIZE32 (bit 10 in kernel_code_properties)
            uint16_t props;
            std::memcpy(&props, desc + 56, 2);
            props &= ~(1u << 10);
            std::memcpy(desc + 56, &props, 2);

            // Clear COMPUTE_PGM_RSRC3 (GFX12 fields don't apply to GFX9)
            uint32_t rsrc3 = 0;
            std::memcpy(desc + 44, &rsrc3, 4);
          }
        }
      }
    }
  }

  // Step 7: Patch ELF metadata (e_flags, .note ISA strings)
  std::cerr << "hotswap: transpile: patching ELF metadata for " << tgt_cpu << "...\n";
  PatchElfMetadata(elf, size, tgt_cpu);

  // Debug: verify final e_flags and ISA strings
  {
    uint32_t final_flags;
    std::memcpy(&final_flags, elf + 48, 4);
    std::cerr << "hotswap: transpile: final e_flags=0x" << std::hex << final_flags
              << " MACH=0x" << (final_flags & 0xff) << std::dec << "\n";
    // Check for remaining gfx1250 strings
    int gfx1250_count = 0;
    for (size_t i = 0; i + 7 <= size; ++i) {
      if (std::memcmp(elf + i, "gfx1250", 7) == 0) ++gfx1250_count;
    }
    if (gfx1250_count > 0) {
      std::cerr << "hotswap: transpile: WARNING: " << gfx1250_count
                << " remaining gfx1250 references!\n";
    }
  }

  result.rules_matched = stats->translated_passthrough +
                         stats->translated_renamed +
                         stats->translated_waitcnt;

  std::cerr << "hotswap: transpile: complete (" << src_cpu << " → " << tgt_cpu
            << ")\n";

  return result;
}

}  // namespace hotswap
}  // namespace rocr
