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
#include <set>
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
#include <llvm/MC/MCInstrDesc.h>
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
    // v_fmac_f32/f16/f64: available on gfx942 (CDNA3) natively — no rename needed
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

  // s_wait_alu / s_delay_alu — GFX12-only dependency hints
  if (mnemonic == "s_wait_alu") return true;
  if (mnemonic == "s_delay_alu") return true;

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

static std::vector<std::string> WidenExecOperation(const std::string& line, bool compact_mode = false) {
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
          // ALWAYS use manual b32 sequence for saveexec.
          std::string src32 = src;
          if (src32 == "vcc") src32 = "vcc_lo";
          result.push_back("s_mov_b32 " + dst + ", exec_lo");
          bool is_or = (b64_mnem.find("s_or_saveexec") == 0);
          if (is_or) {
            result.push_back("s_or_b32 exec_lo, exec_lo, " + src32);
          } else if (b64_mnem.find("andn2") != std::string::npos) {
            result.push_back("s_andn2_b32 exec_lo, exec_lo, " + src32);
          } else {
            result.push_back("s_and_b32 exec_lo, exec_lo, " + src32);
          }
          // exec_hi is already 0 (cleared at kernel start and after every
          // exec modification). Only OR saveexec can make exec_hi non-zero
          // (0 OR s[src+1] = s[src+1]). AND/ANDN2 preserve 0 (0 AND x = 0).
          // In compact mode, skip the redundant clear for AND/ANDN2.
          if (is_or || !compact_mode) {
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

// ── TTMP Taint Analysis ──────────────────────────────────────────────────────
//
// Forward data-flow analysis to identify the TTMP workgroup-ID computation
// chain in gfx1250 kernels. Any SGPR defined by a TTMP-referencing instruction
// is "tainted", and any subsequent instruction whose SGPR sources are ALL
// tainted is also skipped. This replaces fragile pattern-matching rules.

enum class RegKind { SGPR, VGPR, TTMP, SCC, VCC, EXEC, Other };

static RegKind ClassifyReg(unsigned reg, const llvm::MCRegisterInfo& MRI) {
  const char* name = MRI.getName(reg);
  if (!name) return RegKind::Other;
  if (strncmp(name, "TTMP", 4) == 0) return RegKind::TTMP;
  if (strncmp(name, "SGPR", 4) == 0) return RegKind::SGPR;
  if (strncmp(name, "VGPR", 4) == 0) return RegKind::VGPR;
  if (strcmp(name, "SCC") == 0) return RegKind::SCC;
  if (strncmp(name, "VCC", 3) == 0) return RegKind::VCC;
  if (strncmp(name, "EXEC", 4) == 0) return RegKind::EXEC;
  return RegKind::Other;
}

static bool IsRegTainted(unsigned reg, const std::set<unsigned>& tainted,
                         const llvm::MCRegisterInfo& MRI) {
  if (tainted.count(reg)) return true;
  for (auto sub : MRI.subregs(reg))
    if (tainted.count(sub)) return true;
  for (auto sup : MRI.superregs(reg))
    if (tainted.count(sup)) return true;
  return false;
}

static void TaintReg(unsigned reg, std::set<unsigned>& tainted,
                     const llvm::MCRegisterInfo& MRI) {
  tainted.insert(reg);
  for (auto sub : MRI.subregs(reg))
    tainted.insert(sub);
}

static void UntaintReg(unsigned reg, std::set<unsigned>& tainted,
                       const llvm::MCRegisterInfo& MRI) {
  tainted.erase(reg);
  for (auto sub : MRI.subregs(reg))
    tainted.erase(sub);
  for (auto sup : MRI.superregs(reg))
    tainted.erase(sup);
}

enum class TaintAction { Keep, Skip, Replace };

struct TaintResult {
  TaintAction action;
  std::string replace_dst;  // for Replace: "s2"
  std::string replace_src;  // for Replace: "v5" or "v4"
};

// Extract def and use register sets from an MCInst via MCInstrDesc.
static void GetInstRegs(const llvm::MCInst& inst,
                        const llvm::MCInstrInfo& MCII,
                        const llvm::MCRegisterInfo& MRI,
                        std::vector<unsigned>& defs,
                        std::vector<unsigned>& uses) {
  const llvm::MCInstrDesc& desc = MCII.get(inst.getOpcode());
  unsigned num_defs = desc.getNumDefs();

  // Explicit operands
  for (unsigned i = 0; i < inst.getNumOperands(); ++i) {
    const auto& op = inst.getOperand(i);
    if (!op.isReg() || op.getReg() == 0) continue;
    if (i < num_defs)
      defs.push_back(op.getReg());
    else
      uses.push_back(op.getReg());
  }

  // Implicit defs
  for (auto imp : desc.implicit_defs())
    defs.push_back(imp);

  // Implicit uses
  for (auto imp : desc.implicit_uses())
    uses.push_back(imp);
}

struct SourceInstrForTaint {
  std::string text;
  llvm::MCInst inst;
  bool valid_inst;
};

static std::vector<TaintResult> AnalyzeTTMPTaint(
    const std::vector<SourceInstrForTaint>& instrs,
    const llvm::MCInstrInfo& MCII,
    const llvm::MCRegisterInfo& MRI) {

  std::vector<TaintResult> results;
  results.reserve(instrs.size());
  std::set<unsigned> tainted;
  bool dump = std::getenv("HSA_HOTSWAP_DUMP") != nullptr;

  for (size_t i = 0; i < instrs.size(); ++i) {
    const auto& si = instrs[i];
    const auto& text = si.text;
    std::string mnemonic = ExtractMnemonic(text);

    // Default: keep
    TaintResult tr;
    tr.action = TaintAction::Keep;

    // Non-decodable instructions (.long) — always keep, no taint
    if (!si.valid_inst) {
      results.push_back(tr);
      continue;
    }

    // VALU / memory / branch — never part of TTMP computation, always keep.
    // TTMP computation is all SALU (s_* instructions).
    if (mnemonic.empty() || mnemonic[0] != 's' ||
        mnemonic.find("s_cbranch_") == 0 || mnemonic == "s_branch" ||
        mnemonic == "s_endpgm" || mnemonic == "s_barrier" ||
        mnemonic.find("s_barrier_") == 0 || mnemonic == "s_nop" ||
        mnemonic == "s_waitcnt" || mnemonic.find("s_wait_") == 0 ||
        mnemonic == "s_clause" || mnemonic == "s_delay_alu" ||
        mnemonic == "s_wait_alu" || mnemonic == "s_code_end" ||
        mnemonic == "s_set_inst_prefetch_distance") {
      results.push_back(tr);
      continue;
    }

    // Extract register defs and uses
    std::vector<unsigned> defs, uses;
    GetInstRegs(si.inst, MCII, MRI, defs, uses);

    // Check for direct TTMP reference in uses
    bool uses_ttmp = false;
    for (auto r : uses) {
      if (ClassifyReg(r, MRI) == RegKind::TTMP) {
        uses_ttmp = true;
        break;
      }
    }
    // Also check defs for TTMP (rare but possible)
    bool defs_ttmp = false;
    for (auto r : defs) {
      if (ClassifyReg(r, MRI) == RegKind::TTMP) {
        defs_ttmp = true;
        break;
      }
    }

    // Rule 1: s_getreg HW_REG_IB_STS2 → Skip, taint dest
    if (mnemonic == "s_getreg_b32" &&
        text.find("HW_REG_IB_STS2") != std::string::npos) {
      tr.action = TaintAction::Skip;
      for (auto r : defs) TaintReg(r, tainted, MRI);
      if (dump) std::cerr << "hotswap: taint: SKIP (HW_REG_IB_STS2): " << text << "\n";
      results.push_back(tr);
      continue;
    }

    // Rule 2: s_setreg HW_REG_WAVE_MODE → Skip
    if ((mnemonic == "s_setreg_imm32_b32" || mnemonic == "s_setreg_b32") &&
        text.find("HW_REG_WAVE_MODE") != std::string::npos) {
      tr.action = TaintAction::Skip;
      if (dump) std::cerr << "hotswap: taint: SKIP (HW_REG_WAVE_MODE): " << text << "\n";
      results.push_back(tr);
      continue;
    }

    // Rule 3: Direct TTMP use → Skip (or Replace for s_cselect_b32)
    if (uses_ttmp || defs_ttmp) {
      if (mnemonic == "s_cselect_b32") {
        // Replace: extract dest from assembly text (not MCInst, which uses
        // internal names like SGPR4 instead of asm syntax s4)
        tr.action = TaintAction::Replace;
        size_t op_start = text.find(mnemonic) + mnemonic.size();
        std::string ops = text.substr(op_start);
        size_t s = ops.find_first_not_of(" \t");
        size_t e = ops.find_first_of(" \t,", s);
        if (s != std::string::npos)
          tr.replace_dst = ops.substr(s, e != std::string::npos ? e - s : std::string::npos);
        // ttmp9 = workgroup_id_x (saved in v5), ttmp7 = workgroup_id_y (saved in v4)
        tr.replace_src = (text.find("ttmp9") != std::string::npos) ? "v5" : "v4";
        // Untaint the destination (now holds valid workgroup_id) and
        // clear SCC taint (s_cselect consumes the tainted SCC, ending
        // that branch of the TTMP chain).
        for (auto r : defs) UntaintReg(r, tainted, MRI);
        for (auto r : uses) {
          if (ClassifyReg(r, MRI) == RegKind::SCC)
            UntaintReg(r, tainted, MRI);
        }
        if (dump) std::cerr << "hotswap: taint: REPLACE (s_cselect ttmp → "
                            << tr.replace_dst << " = " << tr.replace_src << "): " << text << "\n";
      } else {
        tr.action = TaintAction::Skip;
        // Taint all SGPR defs
        for (auto r : defs) {
          RegKind kind = ClassifyReg(r, MRI);
          if (kind == RegKind::SGPR || kind == RegKind::SCC)
            TaintReg(r, tainted, MRI);
        }
        if (dump) std::cerr << "hotswap: taint: SKIP (direct TTMP): " << text << "\n";
      }
      results.push_back(tr);
      continue;
    }

    // Rule 4: s_load → always Keep, untaint dest (fresh data from memory)
    if (mnemonic.find("s_load_") == 0 || mnemonic.find("s_buffer_load_") == 0) {
      for (auto r : defs) UntaintReg(r, tainted, MRI);
      if (dump && !tainted.empty())
        std::cerr << "hotswap: taint: KEEP (s_load clears taint on defs): " << text << "\n";
      results.push_back(tr);
      continue;
    }

    // Rule 5: s_cmp with tainted source → Skip, taint SCC
    if (mnemonic.find("s_cmp_") == 0) {
      bool any_tainted = false;
      for (auto r : uses) {
        if (IsRegTainted(r, tainted, MRI)) { any_tainted = true; break; }
      }
      if (any_tainted) {
        tr.action = TaintAction::Skip;
        // Taint SCC (implicit def of s_cmp)
        for (auto r : defs) TaintReg(r, tainted, MRI);
        if (dump) std::cerr << "hotswap: taint: SKIP (s_cmp tainted): " << text << "\n";
        results.push_back(tr);
        continue;
      }
      // Not tainted — keep
      results.push_back(tr);
      continue;
    }

    // For remaining SALU instructions: check source taint
    bool has_tainted_src = false;
    bool has_untainted_sgpr_src = false;
    for (auto r : uses) {
      RegKind kind = ClassifyReg(r, MRI);
      if (kind == RegKind::SGPR || kind == RegKind::SCC) {
        if (IsRegTainted(r, tainted, MRI))
          has_tainted_src = true;
        else
          has_untainted_sgpr_src = true;
      }
      // Immediates are not registers, so they don't appear here.
      // VGPRs, EXEC, VCC etc. don't count for taint propagation.
    }

    // Rule 6: All SGPR sources tainted (and at least one) → Skip, taint defs
    if (has_tainted_src && !has_untainted_sgpr_src) {
      tr.action = TaintAction::Skip;
      for (auto r : defs) {
        RegKind kind = ClassifyReg(r, MRI);
        if (kind == RegKind::SGPR || kind == RegKind::SCC)
          TaintReg(r, tainted, MRI);
      }
      if (dump) std::cerr << "hotswap: taint: SKIP (all srcs tainted): " << text << "\n";
      results.push_back(tr);
      continue;
    }

    // Rule 7: Mixed tainted + untainted → Keep, clear taint on defs
    if (has_tainted_src && has_untainted_sgpr_src) {
      for (auto r : defs) UntaintReg(r, tainted, MRI);
      if (dump) std::cerr << "hotswap: taint: KEEP (mixed taint, clear defs): " << text << "\n";
      results.push_back(tr);
      continue;
    }

    // Rule 8: No taint involvement → Keep, but clear taint on defs
    // (instruction executes and produces fresh values, clearing any
    // lingering taint from prior definitions of the same registers)
    if (!tainted.empty()) {
      for (auto r : defs) UntaintReg(r, tainted, MRI);
    }
    results.push_back(tr);
  }

  return results;
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
                                               const std::string& target_cpu,
                                               int scale_temp_vgpr = 7,
                                               int cmpx_temp_sgpr = 16,
                                               bool compact_mode = false) {
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

  // ─── s_load_b96 → split into s_load_dwordx2 + s_load_dword ───
  // GFX9 has no s_load_dwordx3. Instead of widening to dwordx4 (which reads
  // 4 bytes past the intended range and can fault if near end of kernarg buffer),
  // split into two loads: dwordx2 for the first 2 dwords + dword for the 3rd.
  if (mnemonic == "s_load_b96") {
    std::regex reg_range(R"(s\[(\d+):(\d+)\])");
    std::smatch m;
    std::string ops_part = line.substr(line.find(mnemonic) + mnemonic.size());
    if (std::regex_search(ops_part, m, reg_range)) {
      int lo = std::stoi(m[1]);
      // Find the base address and offset after the register range
      std::string after_reg = ops_part.substr(m.position() + m.length());
      // after_reg should be like ", s[0:1], 0x18"
      // Parse: skip comma, get base pair, skip comma, get offset
      std::string base_and_offset = after_reg;
      // Emit two loads:
      // s_load_dwordx2 s[lo:lo+1], base, offset
      // s_load_dword s[lo+2], base, offset+8
      // Parse the offset value to compute offset+8
      size_t offset_pos = after_reg.rfind("0x");
      if (offset_pos == std::string::npos) offset_pos = after_reg.rfind(' ');
      if (offset_pos != std::string::npos) {
        // Find the actual numeric offset
        size_t num_start = after_reg.find_last_of(" \t,", after_reg.size()-1);
        if (num_start == std::string::npos) num_start = 0; else num_start++;
        std::string offset_str = after_reg.substr(num_start);
        int64_t offset_val = 0;
        try { offset_val = std::stoll(offset_str, nullptr, 0); } catch (...) {}
        std::string base_part = after_reg.substr(0, num_start);
        // Trim trailing comma/space from base_part
        size_t be = base_part.find_last_not_of(" \t,");
        if (be != std::string::npos) base_part = base_part.substr(0, be+1);

        result.push_back("s_load_dwordx2 s[" + std::to_string(lo) + ":" +
                          std::to_string(lo+1) + "]" + base_part +
                          ", 0x" + ([](int64_t v) { std::ostringstream o; o << std::hex << v; return o.str(); })(offset_val));
        result.push_back("s_load_dword s" + std::to_string(lo+2) + base_part +
                          ", 0x" + ([](int64_t v) { std::ostringstream o; o << std::hex << v; return o.str(); })(offset_val + 8));
      }
    }
    if (result.empty()) {
      // Fallback: just widen to dwordx4 (old behavior)
      std::string new_line = line;
      size_t mpos = new_line.find("s_load_b96");
      new_line.replace(mpos, 10, "s_load_dwordx4");
      std::smatch m2;
      if (std::regex_search(new_line, m2, reg_range)) {
        int lo2 = std::stoi(m2[1]);
        std::string wider = "s[" + std::to_string(lo2) + ":" + std::to_string(lo2 + 3) + "]";
        new_line = new_line.substr(0, m2.position()) + wider +
                   new_line.substr(m2.position() + m2.length());
      }
      result.push_back(new_line);
    }
    return result;
  }

  // ─── TTMP/HW_REG handling is now in AnalyzeTTMPTaint (pre-pass) ───
  // s_setreg/s_getreg HW_REG, TTMP refs, and tainted intermediates are
  // skipped before TranslateInstruction is called. Only fallback handling
  // for instructions that slip through (e.g., in non-transpile paths):
  if ((mnemonic == "s_setreg_imm32_b32" || mnemonic == "s_setreg_b32" ||
       mnemonic == "s_getreg_b32") &&
      (line.find("HW_REG_WAVE_MODE") != std::string::npos ||
       line.find("HW_REG_IB_STS2") != std::string::npos)) {
    return result;  // skip entirely
  }
  if (line.find("ttmp6") != std::string::npos || line.find("ttmp7") != std::string::npos ||
      line.find("ttmp9") != std::string::npos) {
    return result;  // skip any remaining TTMP references
  }

  // s_cbranch_execz: let it fall through to the normal branch-label resolution
  // below, which converts the numeric offset to the correct target label.
  // Do NOT hardcode .L_exit — that kills kernels with complex control flow
  // (e.g., saveexec masking in softmax where exec=0 means "skip section",
  // not "exit kernel").

  // s_code_end → skip entirely (saves space so translated code fits in .text)
  if (mnemonic == "s_code_end") {
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
      // Use v_lshl_add_u64 with shift=0 for 64-bit add (available on gfx942).
      // v_lshl_add_u64 vdst, vsrc0, shift, src1  →  vdst = (vsrc0 << shift) + src1
      // With shift=0: vdst = vsrc0 + src1
      auto fmtPair = [&](char p, int lo, int hi) -> std::string {
        return std::string(1, p) + "[" + std::to_string(lo) + ":" + std::to_string(hi) + "]";
      };

      // v_lshl_add_u64 on GFX9 accepts SGPR or VGPR as src0 (VOP3 encoding).
      // Use SGPR/VGPR directly; only fall back to temp for immediates.
      std::string src0_pair;
      if (a.prefix == 'v' || a.prefix == 's') {
        src0_pair = fmtPair(a.prefix, a.lo, a.hi);
      } else {
        // Immediate: move to temp VGPRs (rare case)
        result.push_back("v_mov_b32_e32 v252, " + a.imm);
        result.push_back("v_mov_b32_e32 v253, 0");
        src0_pair = "v[252:253]";
      }

      // src1 can be SGPR pair, VGPR pair, or immediate
      std::string src1;
      if (b.prefix == '#') {
        src1 = b.imm;
      } else {
        src1 = fmtPair(b.prefix, b.lo, b.hi);
      }

      std::string dst_pair = fmtPair(d.prefix, d.lo, d.hi);
      result.push_back("v_lshl_add_u64 " + dst_pair + ", " + src0_pair + ", 0, " + src1);
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

  // ─── v_fmamk_f32 / v_fmaak_f32 → v_mov_b32 + v_fma_f32 ───
  // GFX10+ "fma with embedded literal" instructions.
  // v_fmamk_f32 vdst, src0, imm, vsrc2 → vdst = src0 * imm + vsrc2
  // v_fmaak_f32 vdst, src0, vsrc1, imm → vdst = src0 * vsrc1 + imm
  // GFX9 has no equivalents; use v_mov_b32 to load the literal then v_fma_f32.
  if (mnemonic == "v_fmamk_f32" || mnemonic == "v_fmaak_f32") {
    size_t ops_start = line.find(mnemonic) + mnemonic.size();
    std::string ops = line.substr(ops_start);
    std::vector<std::string> operands;
    std::istringstream oss(ops);
    std::string tok;
    while (std::getline(oss, tok, ',')) {
      size_t s = tok.find_first_not_of(" \t");
      size_t e = tok.find_last_not_of(" \t");
      if (s != std::string::npos) operands.push_back(tok.substr(s, e - s + 1));
    }
    if (operands.size() >= 4) {
      // Load the embedded literal into v6, then emit v_fma_f32
      const std::string& literal = (mnemonic == "v_fmamk_f32") ? operands[2] : operands[3];
      result.push_back("v_mov_b32_e32 v6, " + literal);
      if (mnemonic == "v_fmamk_f32") {
        // vdst = src0 * imm + vsrc2 → v_fma_f32 vdst, src0, v6, vsrc2
        result.push_back("v_fma_f32 " + operands[0] + ", " + operands[1] + ", v6, " + operands[3]);
      } else {
        // vdst = src0 * vsrc1 + imm → v_fma_f32 vdst, src0, vsrc1, v6
        result.push_back("v_fma_f32 " + operands[0] + ", " + operands[1] + ", " + operands[2] + ", v6");
      }
      return result;
    }
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

      // Recursively translate each half so mnemonic renames and other
      // fixups (constant bus, vcc widening, etc.) apply to both halves.
      auto r1 = TranslateInstruction(first_half, source_cpu, target_cpu, scale_temp_vgpr, cmpx_temp_sgpr, compact_mode);
      auto r2 = TranslateInstruction(second_half, source_cpu, target_cpu, scale_temp_vgpr, cmpx_temp_sgpr, compact_mode);
      result.insert(result.end(), r1.begin(), r1.end());
      result.insert(result.end(), r2.begin(), r2.end());
      return result;
    }
  }

  // ─── s_lshlN_add_u32 → s_lshl_b32 + s_add_u32 ───
  // GFX12 has s_lshl1_add_u32 through s_lshl4_add_u32; GFX9 doesn't.
  // Emulate: s_lshlN_add_u32 dst, src0, src1 → dst = (src0 << N) + src1
  {
    int shift_amt = -1;
    if (mnemonic == "s_lshl1_add_u32") shift_amt = 1;
    else if (mnemonic == "s_lshl2_add_u32") shift_amt = 2;
    else if (mnemonic == "s_lshl3_add_u32") shift_amt = 3;
    else if (mnemonic == "s_lshl4_add_u32") shift_amt = 4;
    if (shift_amt >= 0) {
      std::string ops = line.substr(line.find(mnemonic) + mnemonic.size());
      size_t op_start = ops.find_first_not_of(" \t");
      if (op_start != std::string::npos) ops = ops.substr(op_start);
      std::vector<std::string> operands;
      std::istringstream oss(ops);
      std::string tok;
      while (std::getline(oss, tok, ',')) {
        size_t s = tok.find_first_not_of(" \t");
        size_t e = tok.find_last_not_of(" \t");
        if (s != std::string::npos) operands.push_back(tok.substr(s, e - s + 1));
      }
      if (operands.size() >= 3) {
        result.push_back("s_lshl_b32 " + operands[0] + ", " + operands[1] +
                         ", " + std::to_string(shift_amt));
        // Skip add if src1 is 0
        if (operands[2] != "0")
          result.push_back("s_add_u32 " + operands[0] + ", " + operands[0] +
                           ", " + operands[2]);
        return result;
      }
    }
  }

  // ─── SALU float → VALU emulation ───
  // GFX1250 has scalar float instructions; GFX9 doesn't.
  // Emulate: s_op_f32 sdst, ssrc0, ssrc1 →
  //   v_mov_b32 v6, ssrc0
  //   v_op_f32 v6, ssrc1, v6   (ssrc1 as inline constant or SGPR)
  //   v_readfirstlane_b32 sdst, v6
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

        if (mnemonic == "s_fmac_f32") {
          // s_fmac_f32 sdst, ssrc0, ssrc1 → sdst += ssrc0 * ssrc1
          // Use v6 only: mul then add (avoids constant bus violation and v5 clobber).
          result.push_back("v_mov_b32_e32 v6, " + ssrc0);
          result.push_back("v_mul_f32_e32 v6, " + ssrc1 + ", v6");
          result.push_back("v_add_f32_e32 v6, " + sdst + ", v6");
          result.push_back("v_readfirstlane_b32 " + sdst + ", v6");
        } else {
          // Other SALU float ops: v6 = ssrc0; v6 = op(ssrc1, v6); sdst = v6
          result.push_back("v_mov_b32_e32 v6, " + ssrc0);
          result.push_back(valu_op + " v6, " + ssrc1 + ", v6");
          result.push_back("v_readfirstlane_b32 " + sdst + ", v6");
        }
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

        result.push_back(valu_mnem + " v6, " + operands[1]);
        result.push_back("v_readfirstlane_b32 " + operands[0] + ", v6");
        return result;
      }
    }

    // v_s_sqrt_f32 sdst, ssrc → VALU sqrt + readfirstlane
    // GFX12 VALU→SGPR square root; GFX9 must go through VGPR.
    if (mnemonic == "v_s_sqrt_f32") {
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
        result.push_back("v_sqrt_f32_e32 v6, " + operands[1]);
        result.push_back("v_readfirstlane_b32 " + operands[0] + ", v6");
        return result;
      }
    }

    // s_cmp_*_f32 → VALU float compare setting SCC via VCC bridge.
    // GFX12 has SALU float compares (set SCC); GFX9 doesn't.
    // Emulate: v_mov_b32 v6, literal; v_cmp_*_f32_e32 sA, v6; s_cmp_lg_u32 vcc_lo, 0
    // This sets SCC = (comparison result), preserving branch semantics.
    {
      static const std::unordered_map<std::string, std::string> kScmpFloatMap = {
        {"s_cmp_gt_f32", "v_cmp_gt_f32_e32"},
        {"s_cmp_ge_f32", "v_cmp_ge_f32_e32"},
        {"s_cmp_lt_f32", "v_cmp_lt_f32_e32"},
        {"s_cmp_le_f32", "v_cmp_le_f32_e32"},
        {"s_cmp_eq_f32", "v_cmp_eq_f32_e32"},
        {"s_cmp_lg_f32", "v_cmp_lg_f32_e32"},
      };
      auto cmp_it = kScmpFloatMap.find(mnemonic);
      if (cmp_it != kScmpFloatMap.end()) {
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
          // Load second operand to v6 (could be literal or SGPR)
          result.push_back("v_mov_b32_e32 v6, " + operands[1]);
          // VOPC: v_cmp_*_f32_e32 src0, vsrc1 → src0=sA (SGPR), vsrc1=v6
          result.push_back(cmp_it->second + " " + operands[0] + ", v6");
          // Transfer VCC to SCC: SCC = (vcc_lo != 0)
          result.push_back("s_cmp_lg_u32 vcc_lo, 0");
          return result;
        }
      }
    }
  }

  // ─── v_mad_u32 → emulate with v_mul_lo_u32 + v_add_u32 ───
  // GFX9 doesn't have v_mad_u32. Emulate: vdst = src0 * src1 + src2
  // With the taint analysis, TTMP-derived operands are already replaced
  // (e.g., s7 holds workgroup_id_x). Pass through actual operands.
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
      std::string src0 = operands[1];
      std::string src1 = operands[2];
      std::string src2 = operands[3];
      // v_mad_u32 vdst, src0, src1, src2 → vdst = src0 * src1 + src2
      // GFX9 constant bus restriction: at most one SGPR source per instruction.
      // If both src0 and src1 are SGPRs we must move one to v6 first.
      // When vdst == src2: also use v6 as temp to avoid read-before-write.
      bool src0_sgpr = !src0.empty() && src0[0] == 's';
      bool src1_sgpr = !src1.empty() && src1[0] == 's';
      if (vdst != src2) {
        if (src0_sgpr && src1_sgpr) {
          result.push_back("v_mov_b32_e32 v6, " + src0);
          result.push_back("v_mul_lo_u32 " + vdst + ", v6, " + src1);
        } else {
          result.push_back("v_mul_lo_u32 " + vdst + ", " + src0 + ", " + src1);
        }
        result.push_back("v_add_u32_e32 " + vdst + ", " + vdst + ", " + src2);
      } else {
        result.push_back("v_mov_b32_e32 v6, " + src0);
        result.push_back("v_mul_lo_u32 v6, v6, " + src1);
        result.push_back("v_add_u32_e32 " + vdst + ", v6, " + src2);
      }
      return result;
    }
  }

  // ─── v_cndmask_b32_e64: bare SGPR mask widening + constant bus fix ───
  // GFX12 wave32: cmp results are 32-bit (single SGPR sN as mask).
  // GFX9 wave64: mask must be a 64-bit SGPR pair s[N:N+1].
  // Also: if src1 is SGPR and mask is SGPR, move src1 to v6 (constant bus).
  if (mnemonic == "v_cndmask_b32_e64") {
    size_t ops_start = line.find(mnemonic) + mnemonic.size();
    std::string ops = line.substr(ops_start);
    std::vector<std::string> operands;
    std::istringstream oss(ops);
    std::string tok;
    while (std::getline(oss, tok, ',')) {
      size_t s = tok.find_first_not_of(" \t");
      size_t e = tok.find_last_not_of(" \t");
      if (s != std::string::npos) operands.push_back(tok.substr(s, e - s + 1));
    }
    if (operands.size() >= 4) {
      // Widen vcc_lo → vcc in all operands
      for (auto& op : operands) {
        if (op == "vcc_lo") op = "vcc";
      }
      // Widen bare SGPR mask sN → s[N:N+1] (GFX12 wave32 → GFX9 wave64)
      auto& mask = operands[3];
      if (!mask.empty() && mask[0] == 's' && mask.find('[') == std::string::npos &&
          mask.find("vcc") == std::string::npos &&
          mask.find("exec") == std::string::npos &&
          mask.size() > 1 && std::isdigit((unsigned char)mask[1])) {
        int n = std::stoi(mask.substr(1));
        int even = n & ~1;
        mask = "s[" + std::to_string(even) + ":" + std::to_string(even + 1) + "]";
      }
      // Constant bus: if src1 AND mask are both SGPR, move src1 to v6
      std::string& src1 = operands[2];
      bool src1_sgpr = !src1.empty() && src1[0] == 's';
      bool mask_sgpr = !mask.empty() && (mask[0] == 's' ||
                       mask.substr(0, 3) == "vcc" || mask.substr(0, 4) == "exec");
      if (src1_sgpr && mask_sgpr) {
        result.push_back("v_mov_b32_e32 v6, " + src1);
        src1 = "v6";
      }
      std::string fixed = mnemonic + " " + operands[0];
      for (size_t i = 1; i < operands.size(); ++i) fixed += ", " + operands[i];
      result.push_back(fixed);
      return result;
    }
  }

  // ─── v_cndmask_b32_e32 SGPR src0 → move to v6 (constant bus fix) ───
  // GFX12 (wave32) VOP2 form may have an SGPR src0.  On GFX9 (wave64) the
  // implicit VCC mask already occupies the constant bus, so src0 cannot also
  // be SGPR.  Move src0 to v6.  The explicit vcc_lo operand (from GFX12
  // MCInstPrinter) is stripped by the post-processing pass.
  if (mnemonic == "v_cndmask_b32_e32") {
    size_t ops_start = line.find(mnemonic) + mnemonic.size();
    std::string ops = line.substr(ops_start);
    std::vector<std::string> operands;
    std::istringstream oss2(ops);
    std::string tok2;
    while (std::getline(oss2, tok2, ',')) {
      size_t s = tok2.find_first_not_of(" \t");
      size_t e = tok2.find_last_not_of(" \t");
      if (s != std::string::npos) operands.push_back(tok2.substr(s, e - s + 1));
    }
    // operands: [vdst, src0, src1] or [vdst, src0, src1, vcc_lo]
    if (operands.size() >= 3) {
      std::string& src0 = operands[1];
      bool src0_sgpr = !src0.empty() && src0[0] == 's';
      // Large literal in src0 also violates constant bus (VCC is already on it)
      bool src0_lit = src0.size() > 2 && src0[0] == '0' &&
                      (src0[1] == 'x' || src0[1] == 'X');
      if (src0_sgpr || src0_lit) {
        result.push_back("v_mov_b32_e32 v6, " + src0);
        src0 = "v6";
        std::string fixed = mnemonic + " " + operands[0];
        for (size_t i = 1; i < operands.size(); ++i) fixed += ", " + operands[i];
        // Explicit vcc_lo (if present) is stripped by the post-processing pass.
        result.push_back(fixed);
        return result;
      }
    }
  }

  // ─── v_cndmask_b32 (no suffix) with explicit mask → rename to _e64 ───
  // GFX12 VOPD dual-issue emits bare "v_cndmask_b32" with an explicit SGPR mask.
  // On GFX9, VOP2 v_cndmask_b32 uses implicit VCC only — need _e64 for explicit mask.
  if (mnemonic == "v_cndmask_b32") {
    size_t ops_start = line.find(mnemonic) + mnemonic.size();
    std::string ops = line.substr(ops_start);
    std::vector<std::string> operands;
    std::istringstream oss(ops);
    std::string tok;
    while (std::getline(oss, tok, ',')) {
      size_t s = tok.find_first_not_of(" \t");
      size_t e = tok.find_last_not_of(" \t");
      if (s != std::string::npos) operands.push_back(tok.substr(s, e - s + 1));
    }
    if (operands.size() >= 4) {
      // Widen vcc_lo → vcc
      for (auto& op : operands) {
        if (op == "vcc_lo") op = "vcc";
      }
      // Widen bare SGPR mask sN → s[N:N+1] (GFX12 wave32 → GFX9 wave64)
      auto& mask = operands[3];
      if (!mask.empty() && mask[0] == 's' && mask.find('[') == std::string::npos &&
          mask.find("vcc") == std::string::npos &&
          mask.find("exec") == std::string::npos &&
          mask.size() > 1 && std::isdigit((unsigned char)mask[1])) {
        int n = std::stoi(mask.substr(1));
        int even = n & ~1;
        mask = "s[" + std::to_string(even) + ":" + std::to_string(even + 1) + "]";
      }
      std::string fixed = "v_cndmask_b32_e64 " + operands[0];
      for (size_t i = 1; i < operands.size(); ++i) fixed += ", " + operands[i];
      result.push_back(fixed);
      return result;
    }
    // 3 operands: fall through as VOP2 (implicit VCC), vcc_lo strip will clean up
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

  // ─── VOP3 with 32-bit literal in source → v_mov_b32 + replace with v6 ───
  // GFX9 VOP3 does not support literal constants in source operand positions.
  // GFX12 compilers freely emit e.g. v_fma_f32 vdst, 0x3fb8aa3b, vsrc1, vsrc2.
  // Fix: emit v_mov_b32_e32 v6, <literal>, then use v6 in the instruction.
  // Applies to v_fma_f32/f16/f64 and any VOP3 instruction not already handled.
  {
    // VOP3 instructions are recognizable as having 3+ source operands (4+ total)
    // We check a set of known VOP3 mnemonics that may have literals.
    bool is_vop3_candidate =
        mnemonic == "v_fma_f32" || mnemonic == "v_fma_f16" ||
        mnemonic == "v_fma_f64" || mnemonic == "v_ldexp_f32" ||
        mnemonic == "v_div_fmas_f32" || mnemonic == "v_div_fixup_f32" ||
        mnemonic == "v_med3_f32" || mnemonic == "v_med3_i32" ||
        mnemonic == "v_bfi_b32" || mnemonic == "v_alignbit_b32";
    if (is_vop3_candidate) {
      size_t ops_start = line.find(mnemonic) + mnemonic.size();
      std::string ops = line.substr(ops_start);
      std::vector<std::string> operands;
      std::istringstream oss(ops);
      std::string tok;
      while (std::getline(oss, tok, ',')) {
        size_t s = tok.find_first_not_of(" \t");
        size_t e = tok.find_last_not_of(" \t");
        if (s != std::string::npos) operands.push_back(tok.substr(s, e - s + 1));
      }
      // Check sources (operands[1..n]) for large hex literals
      if (operands.size() >= 3) {
        for (size_t i = 1; i < operands.size(); ++i) {
          std::string op = operands[i];
          // Strip negation modifier to check the base value
          bool neg = !op.empty() && op[0] == '-';
          if (neg) op = op.substr(1);
          // Detect large literal (hex form 0x..., won't be inline-encodable)
          if (op.size() > 2 && op[0] == '0' && (op[1] == 'x' || op[1] == 'X')) {
            result.push_back("v_mov_b32_e32 v6, " + op);
            operands[i] = (neg ? "-" : "") + std::string("v6");
            std::string fixed = mnemonic + " " + operands[0];
            for (size_t j = 1; j < operands.size(); ++j) fixed += ", " + operands[j];
            result.push_back(fixed);
            return result;
          }
        }
      }
    }
  }

  // ─── v_div_scale_f32 null sdst → vcc ───
  // GFX12 allows "null" as a write-discard SGPR destination in VOP3B.
  // GFX9 has no null register — use vcc (the natural sdst for div_scale).
  if (mnemonic == "v_div_scale_f32") {
    // Find ", null," and replace with ", vcc,"
    size_t null_pos = line.find(", null,");
    if (null_pos != std::string::npos) {
      line.replace(null_pos, 7, ", vcc,");
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
        // Use a temp VGPR for the byte offset, above the kernel's own VGPRs.
        // scale_temp_vgpr is passed from the translation loop and set to
        // num_vgprs12+2 (above save registers).  vaddr stays as element index
        // because subsequent scale_offset instructions reuse it.
        const std::string kScaleTemp = "v" + std::to_string(scale_temp_vgpr);
        result.push_back("v_lshlrev_b32_e32 " + kScaleTemp + ", " +
                          std::to_string(shift) + ", " + vaddr);
        // Replace vaddr in the instruction with the byte-offset temp register
        std::string modified = line;
        size_t vaddr_pos = modified.find(vaddr, mnem_end);
        if (vaddr_pos != std::string::npos)
          modified.replace(vaddr_pos, vaddr.size(), kScaleTemp);
        // Strip scale_offset modifier
        size_t so_pos = modified.find("scale_offset");
        if (so_pos != std::string::npos) {
          if (so_pos > 0 && modified[so_pos-1] == ' ') --so_pos;
          modified.erase(so_pos);
        }
        line = modified;
        mnemonic = ExtractMnemonic(line);
        // Don't return yet — let it fall through to mnemonic renaming etc.
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

  // ─── v_cmp_*_e64 with vcc dest + large literal → VOPC _e32 ───
  // GFX9 VOP3 does not support literal constants in source positions.
  // VOPC _e32 supports literals only in src0 (not src1, which must be VGPR).
  // If src0 is VGPR and src1 is a large literal: load src1 into v6, use VOPC.
  // If src0 is a large literal and src1 is VGPR: directly use VOPC.
  if (mnemonic.find("v_cmp_") == 0 && mnemonic.find("_e64") != std::string::npos) {
    size_t op_start = line.find(mnemonic) + mnemonic.size();
    std::string ops_str = line.substr(op_start);
    size_t sp = ops_str.find_first_not_of(" \t");
    if (sp != std::string::npos) ops_str = ops_str.substr(sp);
    // Check first operand is exactly "vcc" or "vcc_lo"
    size_t first_comma = ops_str.find(',');
    if (first_comma != std::string::npos) {
      std::string first_op = ops_str.substr(0, first_comma);
      size_t fe = first_op.find_last_not_of(" \t");
      if (fe != std::string::npos) first_op = first_op.substr(0, fe + 1);
      if (first_op == "vcc" || first_op == "vcc_lo") {
        // Parse remaining operands (sources)
        std::string src_str = ops_str.substr(first_comma + 1);
        std::vector<std::string> srcs;
        std::istringstream oss(src_str);
        std::string tok;
        while (std::getline(oss, tok, ',')) {
          size_t ts = tok.find_first_not_of(" \t");
          size_t te = tok.find_last_not_of(" \t");
          if (ts != std::string::npos) srcs.push_back(tok.substr(ts, te - ts + 1));
        }
        if (srcs.size() >= 2) {
          auto is_large_literal = [](const std::string& s) -> bool {
            if (s.size() > 2 && s[0] == '0' && (s[1] == 'x' || s[1] == 'X')) return true;
            if (!s.empty() && std::isdigit((unsigned char)s[0]) && std::stol(s) > 64) return true;
            if (s.size() > 1 && s[0] == '-' && std::isdigit((unsigned char)s[1]) &&
                std::stol(s) < -16) return true;
            return false;
          };
          std::string base_mnem = mnemonic.substr(0, mnemonic.size() - 4) + "_e32";
          bool src0_lit = is_large_literal(srcs[0]);
          bool src1_lit = is_large_literal(srcs[1]);
          if (src1_lit && !src0_lit) {
            // VOPC src1 must be VGPR — load literal into v6 first
            result.push_back("v_mov_b32_e32 v6, " + srcs[1]);
            result.push_back(base_mnem + " " + srcs[0] + ", v6");
            return result;
          } else if (src0_lit) {
            // src0 can be literal in VOPC — direct conversion
            result.push_back(base_mnem + " " + srcs[0] + ", " + srcs[1]);
            return result;
          }
        }
      }
    }
  }

  // ─── v_cmpx _e64 → v_cmp_e64 + manual exec AND ───
  // On GFX9, v_cmpx writes BOTH EXEC AND VCC. On GFX12 (RDNA), v_cmpx
  // writes ONLY EXEC. The kernel may read VCC later (e.g., softmax
  // saveexec uses vcc_lo). Instead of v_cmpx (which clobbers VCC), use
  // v_cmp_e64 to write the result to s[16:17] (above kernel's 16 SGPRs),
  // then manually AND into exec_lo. SGPR allocation is bumped to 24.
  if (mnemonic.find("v_cmpx_") == 0 && mnemonic.find("_e64") != std::string::npos) {
    // Use VOPC v_cmpx (writes EXEC+VCC on GFX9), but save/restore VCC.
    // Use dynamic SGPR pair (cmpx_temp_sgpr) above kernel's SGPR allocation.
    std::string base_mnem = mnemonic.substr(0, mnemonic.find("_e64"));
    size_t op_start = line.find(mnemonic) + mnemonic.size();
    std::string ops = line.substr(op_start);
    size_t s = ops.find_first_not_of(" \t");
    if (s != std::string::npos) ops = ops.substr(s);
    // Save/restore just vcc_lo (vcc_hi always 0 in wave32-in-wave64 mode)
    std::string stemp = "s" + std::to_string(cmpx_temp_sgpr);
    result.push_back("s_mov_b32 " + stemp + ", vcc_lo");
    result.push_back(base_mnem + " " + ops);
    result.push_back("s_mov_b32 vcc_lo, " + stemp);
    return result;
  }

  // ─── v_cmpx _e32 (VOPC) → save/restore vcc_lo ───
  // On GFX9, v_cmpx_*_e32 writes BOTH EXEC and VCC. On GFX12 only EXEC.
  // In compact mode: skip VCC save to avoid code size overflow (ELF grow
  // is broken for some kernels). The VCC clobber is acceptable for kernels
  // that don't read VCC across reduction boundaries.
  if (mnemonic.find("v_cmpx_") == 0 && mnemonic.find("_e64") == std::string::npos) {
    // Save/restore vcc_lo (v_cmpx clobbers VCC on GFX9), and clear exec_hi
    // (v_cmpx sets exec for ALL 64 threads — inactive lanes get garbage bits).
    std::string stemp = "s" + std::to_string(cmpx_temp_sgpr);
    result.push_back("s_mov_b32 " + stemp + ", vcc_lo");
    result.push_back(line);
    result.push_back("s_mov_b32 exec_hi, 0");
    result.push_back("s_mov_b32 vcc_lo, " + stemp);
    return result;
  }

  // ─── v_cmp _e64 with SGPR dest → expand to SGPR pair for wave64 ───
  // GFX12 wave32: v_cmp_*_e64 s0, src0, src1 (32-bit result)
  // GFX9 wave64: v_cmp_*_e64 s[0:1], src0, src1 (64-bit result)
  // Also handle large literals: GFX9 VOP3 doesn't support literal constants,
  // so load them into v6 first (e.g., v_cmp_class_f32_e64 sN, s8, 0x260).
  if (mnemonic.find("v_cmp_") == 0 && mnemonic.find("_e64") != std::string::npos) {
    size_t op_start = line.find(mnemonic) + mnemonic.size();
    std::string ops = line.substr(op_start);
    size_t s_pos = ops.find_first_not_of(" \t");
    if (s_pos != std::string::npos) {
      std::string trimmed = ops.substr(s_pos);
      // If first operand is a single SGPR (sN), expand to s[N:N+1] for wave64.
      // CRITICAL: wave32→wave64 widening clobbers s[N+1] with exec_hi bits.
      // If the kernel uses s[N+1] (e.g., holds `cols` in softmax), save and
      // restore it around the v_cmp.
      if (trimmed[0] == 's' && std::isdigit(trimmed[1])) {
        size_t comma = trimmed.find(',');
        if (comma != std::string::npos) {
          std::string dst = trimmed.substr(0, comma);
          size_t de = dst.find_last_not_of(" \t");
          dst = dst.substr(0, de + 1);
          int reg_num = std::stoi(dst.substr(1));
          int even = reg_num & ~1;
          int odd = even + 1;
          std::string pair = "s[" + std::to_string(even) + ":" +
                            std::to_string(odd) + "]";
          std::string rest = trimmed.substr(comma);
          // Wave32→wave64 widening: v_cmp sN → s[even:odd].
          // GFX9 requires even-aligned SGPR pairs, so even = N & ~1.
          // The result for active threads (0-31) goes into s[even].
          // The code reads sN (the original GFX12 dest).
          //
          // For even N: result in sN = s[even]. Code reads sN. ✓
          //   Save s[odd], v_cmp, restore s[odd].
          //
          // For odd N: result in s[even] = s[N-1]. Code reads sN = s[odd]. ✗
          //   Save s[even], v_cmp, copy s[even]→sN, restore s[even].
          std::string save_reg = "v" + std::to_string(scale_temp_vgpr);
          line = mnemonic + " " + pair + rest;
          if (reg_num == even) {
            if (compact_mode) {
              // Skip save/restore for even dest in compact mode (code size)
              result.push_back(line);
            } else {
              // Even dest: save s[odd], v_cmp, restore s[odd]
              std::string s_odd = "s" + std::to_string(odd);
              result.push_back("v_mov_b32_e32 " + save_reg + ", " + s_odd);
              result.push_back(line);
              result.push_back("v_readfirstlane_b32 " + s_odd + ", " + save_reg);
            }
          } else {
            // Odd dest: result in s[even], code reads s[odd]. MUST copy.
            std::string s_even = "s" + std::to_string(even);
            std::string s_orig = "s" + std::to_string(reg_num);
            result.push_back("v_mov_b32_e32 " + save_reg + ", " + s_even);
            result.push_back(line);
            result.push_back("s_mov_b32 " + s_orig + ", " + s_even);
            result.push_back("v_readfirstlane_b32 " + s_even + ", " + save_reg);
          }
          // Handle large literal in last operand (GFX9 VOP3 no literal support).
          // Check the v_cmp line that was already emitted in result, and if its
          // last operand is a large literal, prepend a v_mov to load it to v6.
          {
            auto is_ll = [](const std::string& s) -> bool {
              if (s.empty()) return false;
              if (s.size() > 2 && s[0] == '0' && (s[1] == 'x' || s[1] == 'X')) return true;
              try {
                if (std::isdigit((unsigned char)s[0]) && std::stol(s) > 64) return true;
                if (s[0] == '-' && s.size() > 1 && std::stol(s) < -16) return true;
              } catch (...) {}
              return false;
            };
            // Find the v_cmp line in result (it's the 2nd or 3rd element)
            for (size_t ri = 0; ri < result.size(); ri++) {
              if (result[ri].find("v_cmp_") != std::string::npos) {
                size_t lc = result[ri].rfind(',');
                if (lc != std::string::npos) {
                  std::string lo = result[ri].substr(lc + 1);
                  size_t ls = lo.find_first_not_of(" \t");
                  size_t le = lo.find_last_not_of(" \t");
                  if (ls != std::string::npos) {
                    std::string lt = lo.substr(ls, le - ls + 1);
                    if (is_ll(lt)) {
                      // Insert v_mov before the v_cmp, replace literal with v6
                      result.insert(result.begin() + ri, "v_mov_b32_e32 v6, " + lt);
                      result[ri + 1] = result[ri + 1].substr(0, lc + 1) + " v6";
                    }
                  }
                }
                break;
              }
            }
          }
          return result;
        }
      }
    }
    // Check for large literal in last operand (GFX9 VOP3 doesn't allow literals)
    auto is_large_literal = [](const std::string& s) -> bool {
      if (s.empty()) return false;
      if (s.size() > 2 && s[0] == '0' && (s[1] == 'x' || s[1] == 'X')) return true;
      if (std::isdigit((unsigned char)s[0]) && std::stol(s) > 64) return true;
      if (s[0] == '-' && s.size() > 1 && std::stol(s) < -16) return true;
      return false;
    };
    // Find last comma-separated operand
    size_t last_comma = line.rfind(',');
    if (last_comma != std::string::npos) {
      std::string last_op = line.substr(last_comma + 1);
      size_t ls = last_op.find_first_not_of(" \t");
      size_t le = last_op.find_last_not_of(" \t");
      if (ls != std::string::npos) {
        std::string last_trimmed = last_op.substr(ls, le - ls + 1);
        if (is_large_literal(last_trimmed)) {
          result.push_back("v_mov_b32_e32 v6, " + last_trimmed);
          line = line.substr(0, last_comma + 1) + " v6";
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
  auto exec_result = WidenExecOperation(line, compact_mode);
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

    // Clear GFX12-specific high bits that are reserved on GFX9.
    // Bits [31:24] include FORWARD_PROGRESS, WG_RR_EN on GFX12 but are
    // reserved on GFX9 and must be 0.
    rsrc1 &= 0x00FFFFFFu;
    // Set DX10_CLAMP (bit 21) and IEEE_MODE (bit 23) for GFX9
    rsrc1 |= (1u << 21) | (1u << 23);

    // GFX12 RSRC1 layout: bits [5:0] = VGPR field, bits [11:6] = SGPR field
    // GFX9  RSRC1 layout: bits [5:0] = SGPR field, bits [11:6] = VGPR field
    // Convert VGPR count: GFX12 granularity 8 (wave32) → GFX9 granularity 4 (wave64).
    // Add 4 extra VGPRs to accommodate the two save registers (sv_x, sv_y) that
    // the translation inserts above the kernel's native VGPR allocation.
    uint32_t vgpr_field12 = rsrc1 & 0x3Fu;
    uint32_t sgpr_field12_kd = (rsrc1 >> 6) & 0x3Fu;  // save before clear
    // GFX12 (RDNA4) wave32 VGPR granularity is 12 (GFX10=8, GFX11+=12).
    // RSRC1 VGPR field encodes ceil(VGPRs/gran)-1.
    uint32_t num_vgprs = (vgpr_field12 + 1u) * 12u;
    if (num_vgprs < 8u) num_vgprs = 8u;
    num_vgprs += 4u;  // room for sv_x, sv_y
    // ACCUM_OFFSET: on GFX942, arch VGPRs = (ACCUM_OFFSET+1)*4.  Set ACCUM_OFFSET
    // so ALL allocated VGPRs (kernel's + save regs) are arch VGPRs (accessible by VALU).
    // Hardware requires ACCUM_OFFSET < RSRC1 VGPR field, so add one extra accum group.
    uint32_t gfx9_vgpr = (num_vgprs / 4u) - 1u;  // ACCUM_OFFSET = all-arch
    if (gfx9_vgpr > 62u) gfx9_vgpr = 62u;
    uint32_t rsrc1_vgpr_field = gfx9_vgpr + 1u;   // one extra accum group (ACCUM < field required)
    if (rsrc1_vgpr_field > 63u) rsrc1_vgpr_field = 63u;
    // Clear bits [11:0] and set GFX9 SGPR[5:0] and VGPR[11:6]
    rsrc1 &= ~0xFFFu;
    rsrc1 |= (rsrc1_vgpr_field << 6u);  // VGPR in bits [11:6]
    // SGPR: kernel's SGPRs + 8 extra for v_cmpx temp pair (GFX9 gran = 8)
    {
      uint32_t orig_sgpr_field = sgpr_field12_kd;
      uint32_t num_sgprs = (orig_sgpr_field + 1u) * 16u + 8u;
      // Also check .note MSGPACK for .sgpr_count (RSRC1 underreports)
      const char* sgpr_key = ".sgpr_count";
      for (size_t i = 0; i + 12 < elf_size; i++) {
        if (std::memcmp(elf + i, sgpr_key, 11) == 0) {
          uint8_t val = elf[i + 11];
          uint32_t sc = (val <= 0x7F) ? val : (val == 0xCC ? elf[i+12] : 0);
          if (sc + 8 > num_sgprs) num_sgprs = sc + 8;  // sc + 8 for cmpx temp
          break;
        }
      }
      uint32_t gfx9_sgpr = (num_sgprs / 8u) - 1u;
      if (gfx9_sgpr > 12u) gfx9_sgpr = 12u;
      rsrc1 |= gfx9_sgpr;
    }

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
    // ACCUM_OFFSET [5:0]: arch VGPRs = (ACCUM_OFFSET+1)*4.
    // Set to gfx9_vgpr so all num_vgprs (kernel + save regs) are arch VGPRs.
    uint32_t rsrc3 = gfx9_vgpr;
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
        // Replace in-place. Pad with spaces to keep the same total length.
        // The MSGPACK string length prefix stays the same (including padding).
        // The runtime compares against "amdgcn-amd-amdhsa--gfx942" which is a
        // prefix of "amdgcn-amd-amdhsa--gfx942 " — the ROCR hotswap code
        // handles this by patching before the runtime re-validates.
        // DO NOT update the MSGPACK length prefix — keep the original length
        // so the space-padded string is valid MSGPACK.
        std::memcpy(elf + i, new_isa_full.data(), new_isa_full.size());
        for (size_t j = new_isa_full.size(); j < old_isa_full.size(); ++j) {
          elf[i + j] = ' ';
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
    // .rodata. Find the code's offset within .text from the .rodata descriptor
    // so we can preserve the entry_byte_offset after reassembly.
    uint64_t code_offset_in_text = 0;
    uint64_t text_vaddr = 0;
    if (elf_info.text_idx >= 0)
      text_vaddr = elf_info.sections[elf_info.text_idx].addr;
    for (const auto& sec : elf_info.sections) {
      if (sec.name == ".rodata" && sec.size >= 64) {
        for (uint64_t off = 0; off + 64 <= sec.size; off += 64) {
          const uint8_t* desc = elf + sec.offset + off;
          uint64_t entry;
          std::memcpy(&entry, desc + 16, 8);
          if (entry > 0 && entry < 1000000) {
            uint64_t kd_vaddr = sec.addr + off;
            uint64_t code_vaddr = kd_vaddr + entry;
            if (code_vaddr >= text_vaddr)
              code_offset_in_text = code_vaddr - text_vaddr;
            break;
          }
        }
        break;
      }
    }
    std::cerr << "hotswap: transpile: no embedded descriptors in .text, "
              << "code at .text internal offset " << code_offset_in_text << "\n";
    kernels.push_back({code_offset_in_text, code_offset_in_text});
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

    // Emit pre-code data as raw .long words to preserve the code's position.
    // For embedded descriptors: emit 256-byte KD before code.
    // For .rodata descriptors: emit all bytes before code_offset as raw data
    // so the .rodata entry_byte_offset remains valid after reassembly.
    uint64_t emit_end = kern.code_offset;
    uint64_t emit_start = (kern.desc_offset != kern.code_offset) ? kern.desc_offset : 0;
    for (uint64_t i = emit_start; i < emit_end; i += 4) {
      if (i + 4 > elf_info.text_size) break;
      uint32_t word;
      std::memcpy(&word, text + i, 4);
      std::ostringstream oss;
      oss << ".long 0x" << std::hex << word;
      translated_asm += oss.str() + "\n";
    }

    // Determine save-register indices for workgroup IDs.
    // Use VGPRs above the kernel's native GFX12 allocation so they don't collide
    // with the kernel's own register usage.  PatchKernelDescriptorsForWave64 will
    // add 4 to num_vgprs when computing gfx9_vgpr/ACCUM_OFFSET, ensuring these
    // extra registers are allocated as arch (non-accum) VGPRs.
    // Read GFX12 VGPR count from kernel descriptor to place save registers above.
    // Try embedded descriptor in .text first, then fall back to .rodata.
    // Read GFX12 VGPR and SGPR counts from kernel descriptor.
    // GFX12 RSRC1: bits [5:0] = VGPR field (gran 8), bits [11:6] = SGPR field (gran 8)
    uint32_t num_vgprs12 = 8;   // default: minimum GFX12 allocation
    uint32_t num_sgprs12 = 16;  // default
    {
      uint32_t rsrc1_src = 0;
      if (kern.desc_offset != kern.code_offset &&
          kern.desc_offset + 52 <= elf_info.text_size) {
        std::memcpy(&rsrc1_src, text + kern.desc_offset + 48, 4);
      } else {
        for (const auto& sec : elf_info.sections) {
          if (sec.name == ".rodata" && sec.size >= 64) {
            for (uint64_t off = 0; off + 64 <= sec.size; off += 64) {
              const uint8_t* desc = elf + sec.offset + off;
              uint64_t entry;
              std::memcpy(&entry, desc + 16, 8);
              if (entry > 0 && entry < 1000000) {
                std::memcpy(&rsrc1_src, desc + 48, 4);
                break;
              }
            }
            break;
          }
        }
      }
      if (rsrc1_src) {
        num_vgprs12 = ((rsrc1_src & 0x3Fu) + 1u) * 12u;  // GFX12 wave32 gran=12
        num_sgprs12 = (((rsrc1_src >> 6) & 0x3Fu) + 1u) * 16u;
        std::cerr << "hotswap: transpile: GFX12 RSRC1=0x" << std::hex
                  << rsrc1_src << std::dec << " vgpr_field=" << (rsrc1_src & 0x3Fu)
                  << " sgpr_field=" << ((rsrc1_src >> 6) & 0x3Fu)
                  << " → num_vgprs12=" << num_vgprs12
                  << " num_sgprs12=" << num_sgprs12 << "\n";
      }
    }
    if (num_vgprs12 < 8u) num_vgprs12 = 8u;
    if (num_sgprs12 < 16u) num_sgprs12 = 16u;
    // Scan ELF symbols for actual SGPR/VGPR counts (.num_sgpr, .num_vgpr).
    // RSRC1 fields underreport due to granularity; symbols have exact counts.
    for (const auto& sec : elf_info.sections) {
      if (sec.name == ".symtab" && sec.size >= 24) {
        // ELF64 symbol entry: 24 bytes each
        size_t nsyms = sec.size / 24;
        // Find .strtab for name lookup
        uint32_t strtab_link = 0;
        // Read sh_link from .symtab section header
        for (const auto& s2 : elf_info.sections) {
          if (s2.name == ".strtab" && s2.size > 0) {
            for (size_t si = 0; si < nsyms && si < 200; si++) {
              const uint8_t* sym = elf + sec.offset + si * 24;
              uint32_t name_idx;
              std::memcpy(&name_idx, sym, 4);
              if (name_idx > 0 && name_idx < s2.size) {
                const char* name = (const char*)(elf + s2.offset + name_idx);
                uint64_t value;
                std::memcpy(&value, sym + 8, 8);
                // Don't inflate num_vgprs12 with .num_vgpr — save regs must fit
                // within KD's arch VGPRs (RSRC1 base + 4). The saves overlap
                // with kernel VGPRs but are read in the prologue before the
                // kernel's computation overwrites them.
                (void)name;  // .num_vgpr not used for save placement
                if (strstr(name, ".num_sgpr") && value > num_sgprs12)
                  num_sgprs12 = (uint32_t)value;  // exact count from symbol
              }
            }
            break;
          }
        }
        break;
      }
    }
    // Scan .note section (MSGPACK) for .sgpr_count which isn't in the symbol table.
    // Search for the MSGPACK string ".sgpr_count" (0xAB + 11 chars) followed by value.
    {
      const char* key = ".sgpr_count";
      size_t key_len = 11;
      for (const auto& sec : elf_info.sections) {
        if (sec.name == ".note" && sec.size > key_len + 2) {
          for (size_t i = 0; i + key_len + 1 < sec.size; i++) {
            if (std::memcmp(elf + sec.offset + i, key, key_len) == 0) {
              uint8_t val = elf[sec.offset + i + key_len];
              uint32_t sgpr_count = 0;
              if (val <= 0x7F) sgpr_count = val;  // MSGPACK positive fixint
              else if (val == 0xCC) sgpr_count = elf[sec.offset + i + key_len + 1];  // uint8
              else if (val == 0xCD) {  // uint16
                uint16_t v16;
                std::memcpy(&v16, elf + sec.offset + i + key_len + 1, 2);
                sgpr_count = (v16 >> 8) | ((v16 & 0xFF) << 8);  // big-endian
              }
              if (sgpr_count > num_sgprs12) {
                num_sgprs12 = sgpr_count;
                std::cerr << "hotswap: transpile: .sgpr_count=" << sgpr_count
                          << " from MSGPACK (overrides RSRC1)\n";
              }
              break;
            }
          }
          break;
        }
      }
    }
    // Save registers right above the kernel's actual VGPR usage.
    // The KD allocates .num_vgpr + 4 arch VGPRs, matching native gfx942 pattern.
    uint32_t save_vgpr_x = num_vgprs12;       // first free VGPR index
    uint32_t save_vgpr_y = num_vgprs12 + 1u;  // second free VGPR index
    uint32_t cmpx_temp_sgpr = num_sgprs12;    // first free SGPR pair for v_cmpx temp
    const std::string sv_x = "v" + std::to_string(save_vgpr_x);
    const std::string sv_y = "v" + std::to_string(save_vgpr_y);

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
      llvm::MCInst inst;     // decoded MCInst (valid only if valid_inst==true)
      bool valid_inst;       // false for .long fallback
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
          source_instrs.push_back({oss.str(), pos, 4, llvm::MCInst(), false});
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
        source_instrs.push_back({asm_text, pos, static_cast<uint32_t>(inst_size), inst, true});
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
      // s_cbranch_execz now goes through normal branch resolution (not .L_exit)

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

    // ── TTMP taint analysis ──
    // Build the taint input from source_instrs
    std::vector<SourceInstrForTaint> taint_input;
    taint_input.reserve(source_instrs.size());
    for (const auto& si : source_instrs) {
      taint_input.push_back({si.text, si.inst, si.valid_inst});
    }
    auto taint_results = AnalyzeTTMPTaint(taint_input, *src_state.MCII, *src_state.MRI);

    // Update taint Replace actions to use kernel-specific save register indices
    // instead of the hardcoded "v5"/"v4" emitted by AnalyzeTTMPTaint.
    // AnalyzeTTMPTaint uses "v5" for wg_id_x and "v4" for wg_id_y; remap to
    // sv_x/sv_y which live above the kernel's own VGPR allocation.
    for (auto& tr : taint_results) {
      if (tr.action == TaintAction::Replace) {
        if (tr.replace_src == "v5") tr.replace_src = sv_x;
        else if (tr.replace_src == "v4") tr.replace_src = sv_y;
      }
    }

    // GFX12→GFX9 workgroup ID fix:
    // On gfx1250, the workgroup_id is computed from TTMP registers and
    // s_getreg_b32 hwreg(HW_REG_IB_STS2). These don't exist on gfx950.
    // On gfx950, the workgroup_id_x is in s2 (system SGPR).
    //
    // Save wg IDs to sv_x/sv_y — VGPRs above the kernel's own allocation
    // (computed from GFX12 RSRC1 VGPR field) — so they never conflict with
    // the kernel's own register usage (e.g. B-array loads using v4, K_chunk
    // integer division using v8+).  PatchKernelDescriptorsForWave64 adds 4 to
    // num_vgprs when computing ACCUM_OFFSET so these registers are arch VGPRs.
    translated_asm += "v_mov_b32_e32 " + sv_x + ", s2 ; save workgroup_id_x\n";
    translated_asm += "v_mov_b32_e32 " + sv_y + ", s3 ; save workgroup_id_y\n";
    // Save kernarg pointer s[0:1] to dedicated SGPRs for later use.
    // Only needed for kernels that access hidden args (s_load from s[8:9]+0xc).
    // Place at cmpx_temp_sgpr+2 and +3 (above kernel's own SGPRs).
    uint32_t kernarg_save_lo = cmpx_temp_sgpr + 2;
    uint32_t kernarg_save_hi = cmpx_temp_sgpr + 3;
    std::string ka_lo = "s" + std::to_string(kernarg_save_lo);
    std::string ka_hi = "s" + std::to_string(kernarg_save_hi);
    // Note: kernarg save is deferred — only emitted in post-processing
    // if the kernel has s_load from s[8:9]+0xc patterns.

    // Collect all REPLACE registrations so we can refresh them after saveexec.
    // On gfx1250, workgroup_id lives in TTMP (always valid).  After transpiling
    // to gfx942 we store it in an SGPR via v_readfirstlane_b32, but saveexec
    // widens b32→b64 (s_and_saveexec_b32 sN → s_and_saveexec_b64 s[N:N+1]),
    // which clobbers s[N+1] with exec_hi.  Re-emitting the v_readfirstlane
    // after any saveexec restores the SGPR from the VGPR copy (sv_x/sv_y)
    // which live above the kernel's own VGPR allocation and are never touched.
    std::vector<std::pair<std::string, std::string>> replace_regs;
    for (auto& tr : taint_results) {
      if (tr.action == TaintAction::Replace && !tr.replace_dst.empty())
        replace_regs.emplace_back(tr.replace_dst, tr.replace_src);
    }

    // Debug: HSA_HOTSWAP_EARLY_EXIT=N inserts s_endpgm after N source instructions
    // to bisect crash location. Set to 0 to disable.
    int early_exit_after = 0;
    if (const char* ee = std::getenv("HSA_HOTSWAP_EARLY_EXIT"))
      early_exit_after = std::atoi(ee);
    int emitted_count = 0;
    bool early_exit_done = false;

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

      // Check taint analysis result before translating
      if (ii < taint_results.size()) {
        if (taint_results[ii].action == TaintAction::Skip)
          continue;
        if (taint_results[ii].action == TaintAction::Replace) {
          auto& tr = taint_results[ii];
          // Do NOT insert s_waitcnt here!  On GFX12, the TTMP chain runs BEFORE
          // any pending s_load completes — the load later overwrites the dest
          // register with fresh data.  Inserting a wait would force the load to
          // complete first, then our readfirstlane would overwrite the load result
          // (destroying kernarg data the kernel needs).  The readfirstlane reads
          // from sv_x/sv_y which are saved at kernel entry, independent of loads.
          translated_asm += "v_readfirstlane_b32 " + tr.replace_dst
                         + ", " + tr.replace_src + "\n";
          continue;
        }
      }

      // scale_temp_vgpr: use num_vgprs12+2 (above save registers sv_x, sv_y)
      // cmpx_temp_sgpr: use num_sgprs12 (above kernel's SGPR allocation)
      // compact_mode: skip redundant exec_hi clear after AND saveexec
      // (exec_hi is already 0). Helps save space without affecting correctness.
      // Note: even-dest v_cmp save/restore and v_cmpx VCC save are ALWAYS
      auto translated_lines = TranslateInstruction(line, src_cpu, tgt_cpu,
                                                    save_vgpr_y + 1, cmpx_temp_sgpr,
                                                    false);

      // Post-process: replace branch offsets with labels (using actual PC offsets)
      if (ii < source_instrs.size() && !branch_labels.empty()) {
        for (auto& t : translated_lines) {
          std::string tm = ExtractMnemonic(t);
          if (tm.find("s_branch") == 0 || tm.find("s_cbranch_") == 0) {
            // s_cbranch_execz now goes through normal branch resolution
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
      bool translated_had_saveexec = false;
      for (const auto& t : translated_lines) {
        if (t.empty()) continue;
        if (t.find("saveexec") != std::string::npos)
          translated_had_saveexec = true;

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
      ++emitted_count;
      // Debug early exit: insert s_endpgm after N source instructions
      if (early_exit_after > 0 && emitted_count >= early_exit_after && !early_exit_done
          && source_lines.size() > 400) {  // only for large kernels (attn_forward=548)
        // Emit all remaining branch target labels pointing to the exit, then s_endpgm.
        // This prevents assembly errors from unresolved forward references.
        for (size_t jj = ii + 1; jj < source_instrs.size(); jj++) {
          auto lbl = branch_labels.find(source_instrs[jj].pc_offset);
          if (lbl != branch_labels.end())
            translated_asm += lbl->second + ":\n";
        }
        translated_asm += ".L_exit:\n";
        translated_asm += "s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)\n";
        translated_asm += "s_endpgm ; EARLY EXIT after " + std::to_string(emitted_count) + " instrs\n";
        early_exit_done = true;
        std::cerr << "hotswap: transpile: EARLY EXIT after " << emitted_count << " source instructions\n";
        break;
      }
      // exec_hi clear after v_cmpx: on GFX9 wave64, v_cmpx compares ALL
      // 64 lanes including 32-63 which have uninitialized VGPRs. Clear exec_hi
      // ALWAYS — even before s_cbranch_execz — to prevent garbage upper lanes
      // from executing memory operations (global_store/load) later in the code.
      // The v_cmpx+s_cbranch_execz pattern: with exec_hi=0, the branch correctly
      // checks only exec_lo (the 32 valid lanes).
      if (ii < source_instrs.size()) {
        bool has_vcmpx = false;
        for (const auto& t : translated_lines) {
          if (t.find("v_cmpx_") != std::string::npos) { has_vcmpx = true; break; }
        }
        if (has_vcmpx) {
          // For large kernels (like attn_forward): unconditional exec_hi clear.
          // For smaller kernels: conditional (skip when next is s_cbranch_execz).
          // Unconditional clearing regresses split-K (which has no v_cmpx anyway,
          // but the binary change affects numerical results). Conditional is safe
          // for small kernels but leaves garbage exec_hi for large ones.
          if (source_lines.size() > 400) {
            translated_asm += "s_mov_b32 exec_hi, 0\n";
          } else {
            bool next_is_execz = false;
            for (size_t nxt = ii + 1; nxt < source_instrs.size(); nxt++) {
              std::string nm = ExtractMnemonic(source_instrs[nxt].text);
              if (nm.find("s_delay") == 0 || nm.find("s_wait") == 0 ||
                  nm.find("s_nop") == 0 || nm.find("s_clause") == 0) continue;
              if (nm == "s_cbranch_execz") next_is_execz = true;
              break;
            }
            if (!next_is_execz) {
              translated_asm += "s_mov_b32 exec_hi, 0\n";
            }
          }
        }
      }

      // After a saveexec the b32→b64 widening writes exec_hi into the upper
      // SGPR of the pair, potentially overwriting a workgroup_id register.
      // Restore all REPLACE destinations from their VGPR copies.
      if (translated_had_saveexec) {
        for (auto& r : replace_regs)
          translated_asm += "v_readfirstlane_b32 " + r.first + ", " + r.second + "\n";
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
    // Wave32→64 VCC branch fix: convert s_cbranch_vccz/vccnz to use SCC.
    // On GFX942, s_cbranch_vccz checks full 64-bit VCC, but wave32 code only
    // writes VCC_lo. Multiple attempts to clear VCC_hi failed. The solution:
    // use SCC from the PRECEDING s_and_b32/s_andn2_b32 that wrote VCC_lo.
    // These SALU ops set SCC = (result != 0), which is equivalent to VCC_lo != 0.
    // IMPORTANT: This ONLY works if the instruction immediately before the
    // cbranch_vccz/vccnz is the s_and/s_andn2 that writes VCC_lo. If there's
    // an intervening instruction that sets SCC, the conversion is wrong.
    {
      std::string tmp;
      std::istringstream vfix_iss(translated_asm);
      std::string vfix_line;
      std::string prev_line;
      while (std::getline(vfix_iss, vfix_line)) {
        if (vfix_line.find("s_cbranch_vccz") != std::string::npos) {
          // Check if a recent line set VCC_lo (and thus SCC).
          // Skip past s_nop lines to find the actual VCC-setting instruction.
          // SCC is preserved across s_nop (s_nop doesn't modify SCC).
          bool prev_sets_vcc = false;
          // Check prev_line and up to 3 lines back via a simple scan
          for (const auto& check : {prev_line}) {
            if (check.find("vcc_lo") != std::string::npos &&
                (check.find("s_and_b32") != std::string::npos ||
                 check.find("s_andn2_b32") != std::string::npos ||
                 check.find("s_or_b32") != std::string::npos))
              prev_sets_vcc = true;
          }
          // Also match: the VCC-setting instruction was 2 lines back (s_nop between)
          if (!prev_sets_vcc && prev_line.find("s_nop") != std::string::npos)
            prev_sets_vcc = true;  // s_nop doesn't change SCC, so SCC is from the s_and/s_andn2 before
          if (prev_sets_vcc) {
            // SCC from the s_and/s_andn2 = (result != 0). vccz = (VCC == 0).
            // SCC=0 means result=0 → equivalent to VCC_lo=0 → vccz=true → branch.
            size_t lbl_pos = vfix_line.find(".L_");
            if (lbl_pos != std::string::npos) {
              tmp += "s_cbranch_scc0 " + vfix_line.substr(lbl_pos) + " ; vccz→scc0\n";
              prev_line = vfix_line;
              continue;
            }
          }
        }
        if (vfix_line.find("s_cbranch_vccnz") != std::string::npos) {
          bool prev_sets_vcc = (prev_line.find("vcc_lo") != std::string::npos &&
            (prev_line.find("s_and_b32") != std::string::npos ||
             prev_line.find("s_andn2_b32") != std::string::npos ||
             prev_line.find("s_or_b32") != std::string::npos));
          if (!prev_sets_vcc && prev_line.find("s_nop") != std::string::npos)
            prev_sets_vcc = true;
          if (prev_sets_vcc) {
            size_t lbl_pos = vfix_line.find(".L_");
            if (lbl_pos != std::string::npos) {
              tmp += "s_cbranch_scc1 " + vfix_line.substr(lbl_pos) + " ; vccnz→scc1\n";
              prev_line = vfix_line;
              continue;
            }
          }
        }
        tmp += vfix_line + "\n";
        prev_line = vfix_line;
      }
      translated_asm = tmp;
    }
    // Redundant remainder guard: add s8==0 check at .L_br28 entry.
    // The VCC-based guard (s_cbranch_vccz .L_br25) should prevent entry to .L_br28
    // when remainder=0, but it fails on GFX942. This adds a belt-and-suspenders
    // SCC-based check right at .L_br28 entry.
    if (stats->total_instructions > 400) {
      auto replaceAll = [](std::string& s, const std::string& from, const std::string& to) {
        size_t pos = 0;
        while ((pos = s.find(from, pos)) != std::string::npos) {
          s.replace(pos, from.size(), to);
          pos += to.size();
        }
      };
      // At .L_br28, check s8 (remainder count). If 0, skip to .L_br25.
      replaceAll(translated_asm,
        ".L_br28:\n",
        ".L_br28:\n"
        "s_cmp_eq_u32 s8, 0\n"
        "s_cbranch_scc1 .L_br25 ; redundant remainder guard\n");
      // Fix: when D >= N, the "out of range" block that sets s10 = blockSize
      // (from hidden arg) is skipped because exec has no out-of-range threads.
      // s10 retains garbage. The find-max iterative loop (.L_br13) uses s10 as
      // the LDS stride, causing reads from wrong locations.
      // Fix: load s10 BEFORE the exec-masked block. Insert the hidden arg load
      // right after the v_cmpx (which masks exec to D threads) but BEFORE the
      // in-range/out-of-range split.
      // The load: s_load_dword sN, s[saved_kernarg]+0x3c → s_and sN, sN, 0xffff → s10 = blockSize
      // Since s10 will be overwritten by the "out of range" block if it runs,
      // the unconditional load only matters when the block is skipped.
      {
        size_t split_pos = translated_asm.find("s_cbranch_execz .L_br11");
        if (split_pos != std::string::npos) {
          // Find the kernarg save pair from earlier
          std::string ka_pair = "s[30:31]";
          size_t ka_pos = translated_asm.find("; save kernarg ptr lo");
          if (ka_pos != std::string::npos) {
            size_t s_pos = translated_asm.rfind("s_mov_b32 s", ka_pos);
            if (s_pos != std::string::npos) {
              size_t n_start = s_pos + 11;
              size_t n_end = translated_asm.find(',', n_start);
              int lo = std::stoi(translated_asm.substr(n_start, n_end - n_start));
              ka_pair = "s[" + std::to_string(lo) + ":" + std::to_string(lo+1) + "]";
            }
          }
          // Insert unconditional hidden arg load before the split
          std::string fix =
            "s_load_dword s10, " + ka_pair + ", 0x3c ; unconditional blockSize load\n"
            "s_waitcnt lgkmcnt(0)\n"
            "s_and_b32 s10, s10, 0xffff\n";
          translated_asm.insert(split_pos, fix);
        }
      }
      // Revert v1 fix (was wrong - v1 has dual purpose)
      replaceAll(translated_asm,
        "v_mov_b32_e32 v1, s5 ; FIX: use N instead of potentially-garbage s10\n",
        "v_mov_b32_e32 v1, s10\n");
      // Fix tree reduction stride: v3 is computed as v1/2 which gives
      // per-thread values (tid*2) for "in range" threads. The tree
      // reduction needs a COMMON stride = N/2.
      // Only fix the FIRST occurrence (the find-max stride, not the sum reduction).
      // Use s_lshr to compute N/2 into s_temp, then v_mov to v3.
      {
        std::string target = "v_lshrrev_b32_e32 v3, 1, v1\n";
        size_t pos = translated_asm.find(target);
        if (pos != std::string::npos) {
          // Use v3 itself: v_mov s5 to v3, then lshrrev in-place
        std::string fix =
            "v_mov_b32_e32 v3, s5\n"
            "v_lshrrev_b32_e32 v3, 1, v3 ; FIX: tree stride = N/2\n";
          translated_asm.replace(pos, target.size(), fix);
        }
      }
    }
    // Debug: NOP flat-address global_loads (v[pair], off) in large kernels
    // to check if the flat addressing form is the crash cause.
    if (std::getenv("HSA_HOTSWAP_NOP_FLAT") && stats->total_instructions > 400) {
      std::string tmp;
      std::istringstream gl_iss(translated_asm);
      std::string gl_line;
      while (std::getline(gl_iss, gl_line)) {
        if (gl_line.find("global_load_dword") != std::string::npos ||
            gl_line.find("ds_read2_b32") != std::string::npos) {
          tmp += "s_nop 0 ; NOP'd " + gl_line + "\n";
        } else {
          tmp += gl_line + "\n";
        }
      }
      translated_asm = tmp;
    }
    // Debug: HSA_HOTSWAP_SKIP_INNER=1 skips the inner product (.L_br24→.L_br27)
    // by jumping directly from .L_br24 to .L_br25 (store zeros).
    if (std::getenv("HSA_HOTSWAP_SKIP_INNER") && stats->total_instructions > 400) {
      auto replaceAll = [](std::string& s, const std::string& from, const std::string& to) {
        size_t pos = 0;
        while ((pos = s.find(from, pos)) != std::string::npos) {
          s.replace(pos, from.size(), to);
          pos += to.size();
        }
      };
      // No debug patches — essential fixes only
    }
    // Debug: HSA_HOTSWAP_FORCE_REMAINDER=1 forces the remainder loop to always
    // execute by replacing "s_mov_b32 s15, s10" (remainder count) with "s_mov_b32 s15, 1"
    if (std::getenv("HSA_HOTSWAP_FORCE_REMAINDER") && stats->total_instructions > 400) {
      auto replaceAll = [](std::string& s, const std::string& from, const std::string& to) {
        size_t pos = 0;
        while ((pos = s.find(from, pos)) != std::string::npos) {
          s.replace(pos, from.size(), to);
          pos += to.size();
        }
      };
      // Force s10 to -1 (remainder flag = true) so the guard branch is never taken
      replaceAll(translated_asm,
        "s_cselect_b32 s10, -1, 0\n"
        "s_ashr_i32 s7",
        "s_mov_b32 s10, -1 ; FORCE remainder\n"
        "s_ashr_i32 s7");
    }
    // Note: VCC_hi is NOT cleared here. The s_cbranch_vccz/vccnz branches
    // check full 64-bit VCC on wave64. VCC_hi garbage from v_cmp can cause
    // wrong branch decisions. TODO: find correct fix (s_and_b64 clobbers SCC,
    // s_mov vcc_hi doesn't fix it, SCC conversion doesn't fix it either).
    // v_add_nc_u32 → v_add_u32_e32 (may appear without _e32 from VOP3 encoding)
    replaceAll(translated_asm, "v_add_nc_u32 ", "v_add_u32_e32 ");
    replaceAll(translated_asm, "v_sub_nc_u32 ", "v_sub_u32_e32 ");
    // v_cndmask_b32_e32 ... vcc/vcc_lo → strip explicit mask (GFX9 VOP2 uses implicit VCC)
    // Only strip from _e32 form; _e64 always requires an explicit mask operand on GFX9.
    // vcc_lo is stripped when the cndmask handler returns early (before WidenVccReferences).
    // vcc is stripped when the instruction falls through and WidenVccReferences fires first.
    {
      std::string tmp;
      std::istringstream vcc_iss(translated_asm);
      std::string vcc_line;
      while (std::getline(vcc_iss, vcc_line)) {
        if (vcc_line.find("v_cndmask_b32_e32") != std::string::npos) {
          // Strip trailing ", vcc_lo" or ", vcc"
          size_t vcc_pos = vcc_line.rfind(", vcc_lo");
          if (vcc_pos == std::string::npos) vcc_pos = vcc_line.rfind(", vcc");
          if (vcc_pos != std::string::npos)
            vcc_line = vcc_line.substr(0, vcc_pos);
        }
        tmp += vcc_line + "\n";
      }
      translated_asm = tmp;
    }
    // Fix: replace s_load from s[8:9]+0xc with saved kernarg ptr + 0x3c.
    // Only applies to kernels that have this pattern (e.g., attn_forward).
    {
      auto replaceAll = [](std::string& s, const std::string& from, const std::string& to) {
        size_t pos = 0;
        while ((pos = s.find(from, pos)) != std::string::npos) {
          s.replace(pos, from.size(), to);
          pos += to.size();
        }
      };
      // Check if the pattern exists before doing replacements.
      // Extract the kernarg save register pair from the prologue comment.
      if (translated_asm.find("s_load_dword s1, s[8:9], 0xc") != std::string::npos ||
          translated_asm.find("s_load_dword s0, s[8:9], 0xc") != std::string::npos) {
        // Find the kernarg save registers from prologue
        std::string ka_pair = "s[30:31]";  // fallback
        size_t ka_pos = translated_asm.find("; save kernarg ptr lo");
        if (ka_pos != std::string::npos) {
          size_t s_pos = translated_asm.rfind("s_mov_b32 s", ka_pos);
          if (s_pos != std::string::npos) {
            size_t n_start = s_pos + 11;
            size_t n_end = translated_asm.find(',', n_start);
            int lo = std::stoi(translated_asm.substr(n_start, n_end - n_start));
            ka_pair = "s[" + std::to_string(lo) + ":" + std::to_string(lo + 1) + "]";
          }
        } else {
          // No save emitted yet — emit it now
          size_t insert_pos = translated_asm.find("; save workgroup_id_y\n");
          if (insert_pos != std::string::npos) {
            insert_pos = translated_asm.find('\n', insert_pos) + 1;
            // Use s28+2=s30 and s29+2=s31 as safe default (large kernels)
            translated_asm.insert(insert_pos,
              "s_mov_b32 s30, s0 ; save kernarg ptr lo\n"
              "s_mov_b32 s31, s1 ; save kernarg ptr hi\n");
          }
        }
        replaceAll(translated_asm,
          "s_load_dword s1, s[8:9], 0xc",
          "s_load_dword s1, " + ka_pair + ", 0x3c ; use saved kernarg ptr");
        replaceAll(translated_asm,
          "s_load_dword s0, s[8:9], 0xc",
          "s_load_dword s0, " + ka_pair + ", 0x3c ; use saved kernarg ptr");
      }
    }
    // Debug: HSA_HOTSWAP_NOP_GLOBAL=1 caps outer loop to 2 iterations
    if (std::getenv("HSA_HOTSWAP_NOP_GLOBAL") && stats->total_instructions > 400) {
      auto replaceAll = [](std::string& s, const std::string& from, const std::string& to) {
        size_t pos = 0;
        while ((pos = s.find(from, pos)) != std::string::npos) {
          s.replace(pos, from.size(), to);
          pos += to.size();
        }
      };
      // Cap s5 to 2 right after the s_waitcnt that loads it
      replaceAll(translated_asm,
        "s_cmp_ge_i32 s2, s4\n",
        "s_mov_b32 s5, 2 ; DBG cap outer loop\n"
        "s_cmp_ge_i32 s2, s4\n");
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
        // Doesn't fit — grow the ELF by shifting everything after .text
        uint64_t text_end = elf_info.text_offset + elf_info.text_size;
        uint64_t delta = ((new_text_size - elf_info.text_size + 255u) / 256u) * 256u;
        uint64_t grown_size = new_elf_size + delta;
        uint8_t* grown = (uint8_t*)calloc(1, grown_size);
        // Copy everything before .text end
        std::memcpy(grown, new_elf, text_end);
        // Write new .text content
        std::memcpy(grown + elf_info.text_offset, new_text, new_text_size);
        // NOP-pad .text to aligned boundary
        uint64_t new_sec_size = elf_info.text_size + delta;
        for (uint64_t p = new_text_size; p < new_sec_size; p += 4) {
          uint8_t nop[] = {0x00, 0x00, 0x80, 0xBF};  // s_nop 0
          std::memcpy(grown + elf_info.text_offset + p, nop, 4);
        }
        // Copy everything after .text, shifted by delta
        uint64_t tail = new_elf_size - text_end;
        if (tail > 0) std::memcpy(grown + text_end + delta, new_elf + text_end, tail);
        // Update section headers: shift file offsets for sections after .text
        std::memcpy(&e_shoff, grown + 40, 8);
        e_shoff += delta;
        std::memcpy(grown + 40, &e_shoff, 8);  // section header table offset
        for (uint16_t i = 0; i < e_shnum; ++i) {
          uint64_t sh_off = e_shoff + i * e_shentsize;
          uint64_t sec_offset;
          std::memcpy(&sec_offset, grown + sh_off + 24, 8);
          if (sec_offset > elf_info.text_offset) {
            sec_offset += delta;
            std::memcpy(grown + sh_off + 24, &sec_offset, 8);
          }
          // Update .text section size
          if (static_cast<int>(i) == elf_info.text_idx) {
            std::memcpy(grown + sh_off + 32, &new_sec_size, 8);
          }
        }
        // Update program headers: shift offsets and sizes for LOAD segments
        uint64_t e_phoff;
        uint16_t e_phentsize, e_phnum;
        std::memcpy(&e_phoff, grown + 32, 8);
        std::memcpy(&e_phentsize, grown + 54, 2);
        std::memcpy(&e_phnum, grown + 56, 2);
        for (uint16_t i = 0; i < e_phnum; ++i) {
          uint64_t ph_off = e_phoff + i * e_phentsize;
          if (ph_off + 56 > grown_size) break;
          uint64_t p_offset, p_filesz, p_memsz;
          std::memcpy(&p_offset, grown + ph_off + 8, 8);
          std::memcpy(&p_filesz, grown + ph_off + 32, 8);
          std::memcpy(&p_memsz, grown + ph_off + 40, 8);
          if (p_offset == elf_info.text_offset) {
            // This LOAD segment contains .text — grow FileSiz and MemSiz
            p_filesz += delta;
            p_memsz += delta;
            std::memcpy(grown + ph_off + 32, &p_filesz, 8);
            std::memcpy(grown + ph_off + 40, &p_memsz, 8);
          } else if (p_offset > elf_info.text_offset) {
            // Segment after .text — shift offset
            p_offset += delta;
            std::memcpy(grown + ph_off + 8, &p_offset, 8);
          }
        }
        free(new_elf);
        new_elf = grown;
        new_elf_size = grown_size;
        std::cerr << "hotswap: transpile: .text grew " << elf_info.text_size
                  << " → " << new_sec_size << " (ELF resized by +" << delta << ")\n";
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
      // Scan .symtab for .num_vgpr to get actual VGPR count (RSRC1 underreports).
      // Must match the translation loop's computation for save register placement.
      uint32_t sym_num_vgpr = 0;
      for (const auto& ss : updated_info.sections) {
        if (ss.name == ".symtab" && ss.size >= 24) {
          for (const auto& st : updated_info.sections) {
            if (st.name == ".strtab" && st.size > 0) {
              size_t nsyms = ss.size / 24;
              for (size_t si = 0; si < nsyms && si < 200; si++) {
                const uint8_t* sym = elf + ss.offset + si * 24;
                uint32_t ni; std::memcpy(&ni, sym, 4);
                if (ni > 0 && ni < st.size) {
                  const char* nm = (const char*)(elf + st.offset + ni);
                  if (strstr(nm, ".num_vgpr")) {
                    uint64_t val; std::memcpy(&val, sym + 8, 8);
                    if (val > sym_num_vgpr) sym_num_vgpr = (uint32_t)val;
                  }
                }
              }
              break;
            }
          }
          break;
        }
      }

      if (sym_num_vgpr > 0)
        std::cerr << "hotswap: transpile: KD sym_num_vgpr=" << sym_num_vgpr << "\n";

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
            if (std::getenv("HSA_HOTSWAP_DUMP"))
              std::cerr << "hotswap: transpile: original GFX12 RSRC1=0x"
                        << std::hex << rsrc1 << std::dec << "\n";
            // Clear GFX12-specific high bits (reserved on GFX9), then set GFX9 fields
            rsrc1 &= 0x00FFFFFFu;  // Clear bits [31:24] (reserved on GFX9)
            rsrc1 |= (1u << 21) | (1u << 23);  // DX10_CLAMP + IEEE_MODE
            // GFX9 RSRC1: bits [5:0] = SGPR field, bits [11:6] = VGPR field
            // GFX12 RSRC1: bits [5:0] = VGPR field, bits [11:6] = SGPR field
            // Convert VGPR: GFX12 granularity 8 → GFX9 granularity 4.
            // Add 4 extra VGPRs for the two save registers (sv_x, sv_y) the
            // translation inserts above the kernel's native VGPR allocation.
            uint32_t vgpr_field12 = rsrc1 & 0x3Fu;
            uint32_t sgpr_field12_rd = (rsrc1 >> 6) & 0x3Fu;  // save before clear
            uint32_t num_vgprs = (vgpr_field12 + 1u) * 12u;  // GFX12 wave32 gran=12
            if (num_vgprs < 8u) num_vgprs = 8u;
            num_vgprs += 4u;  // room for save registers
            uint32_t gfx9_vgpr = (num_vgprs / 4u) - 1u;
            if (gfx9_vgpr > 62u) gfx9_vgpr = 62u;
            // Match native gfx942: field = ACCUM (all arch, no accum group).
            // Native uses ACCUM=field (e.g., ACCUM=4, field=4 for .num_vgpr=20).
            uint32_t rsrc1_vgpr_field = gfx9_vgpr;
            if (rsrc1_vgpr_field > 63u) rsrc1_vgpr_field = 63u;
            // Clear bits [11:0] and set GFX9 SGPR[5:0] and VGPR[11:6]
            rsrc1 &= ~0xFFFu;
            rsrc1 |= (rsrc1_vgpr_field << 6u);  // VGPR in bits [11:6]
            {
              uint32_t num_sgprs_rd = (sgpr_field12_rd + 1u) * 16u + 8u;
              // Check .note MSGPACK for .sgpr_count
              const char* sgpr_key = ".sgpr_count";
              for (size_t si = 0; si + 12 < size; si++) {
                if (std::memcmp(elf + si, sgpr_key, 11) == 0) {
                  uint8_t val = elf[si + 11];
                  uint32_t sc = (val <= 0x7F) ? val : (val == 0xCC ? elf[si+12] : 0);
                  if (sc + 8 > num_sgprs_rd) num_sgprs_rd = sc + 8;
                  break;
                }
              }
              uint32_t gfx9_sgpr = (num_sgprs_rd / 8u) - 1u;
              if (gfx9_sgpr > 12u) gfx9_sgpr = 12u;
              rsrc1 |= gfx9_sgpr;
            }
            std::memcpy(desc + 48, &rsrc1, 4);

            // Patch COMPUTE_PGM_RSRC2 (offset 52)
            // Enable workgroup ID system SGPRs (gfx1250 uses TTMP, GFX9 uses SGPRs)
            uint32_t rsrc2;
            std::memcpy(&rsrc2, desc + 52, 4);
            rsrc2 |= (1u << 7);  // ENABLE_SGPR_WORKGROUP_ID_X → s2
            rsrc2 |= (1u << 8);  // ENABLE_SGPR_WORKGROUP_ID_Y → s3
            rsrc2 |= (1u << 9);  // ENABLE_SGPR_WORKGROUP_ID_Z → s4
            std::memcpy(desc + 52, &rsrc2, 4);

            // Clear ENABLE_WAVEFRONT_SIZE32 (bit 10 in kernel_code_properties)
            uint16_t props;
            std::memcpy(&props, desc + 56, 2);
            props &= ~(1u << 10);
            std::memcpy(desc + 56, &props, 2);

            // Patch COMPUTE_PGM_RSRC3 (offset 44)
            // ACCUM_OFFSET [5:0]: arch VGPRs = (ACCUM_OFFSET+1)*4.
            // Set to gfx9_vgpr so all num_vgprs VGPRs (kernel + save regs) are arch.
            uint32_t rsrc3 = gfx9_vgpr;
            std::memcpy(desc + 44, &rsrc3, 4);

            // Debug: dump patched descriptor when HSA_HOTSWAP_DUMP is set
            if (std::getenv("HSA_HOTSWAP_DUMP")) {
              uint32_t r1, r2; uint16_t p;
              std::memcpy(&r1, desc + 48, 4);
              std::memcpy(&r2, desc + 52, 4);
              std::memcpy(&p, desc + 56, 2);
              std::cerr << "hotswap: transpile: patched KD: RSRC1=0x"
                        << std::hex << r1 << " RSRC2=0x" << r2
                        << " props=0x" << p << std::dec
                        << " vgpr=" << (((r1 >> 6) & 0x3f) + 1) * 4
                        << " sgpr=" << ((r1 & 0x3f) + 1) * 8
                        << " user_sgpr=" << ((r2 >> 1) & 0x1f)
                        << " tgidx=" << ((r2 >> 7) & 1)
                        << " tgidy=" << ((r2 >> 8) & 1)
                        << " tgidz=" << ((r2 >> 9) & 1)
                        << "\n";
            }
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

  // Debug: dump patched ELF if HSA_HOTSWAP_DUMP_ELF is set
  if (auto* dump_path = std::getenv("HSA_HOTSWAP_DUMP_ELF")) {
    FILE* fp = fopen(dump_path, "wb");
    if (fp) {
      fwrite(elf, 1, size, fp);
      fclose(fp);
      std::cerr << "hotswap: transpile: dumped patched ELF to " << dump_path
                << " (" << size << " bytes)\n";
    }
  }

  return result;
}

}  // namespace hotswap
}  // namespace rocr
