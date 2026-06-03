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
#include <cassert>
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
    {"s_ctz_i32_b32", "s_ff1_i32_b32"},
    {"s_ctz_i32_b64", "s_ff1_i32_b64"},
    // GFX12 PC-relative call/return mnemonics → GFX9 equivalents
    {"s_get_pc_i64", "s_getpc_b64"},
    {"s_swap_pc_i64", "s_swappc_b64"},
    {"s_set_pc_i64", "s_setpc_b64"},
};

// VALU renames (GFX12 uses IEEE-explicit names)
static const MnemonicMapping kVALURenames[] = {
    {"v_max_num_f32", "v_max_f32"},
    {"v_min_num_f32", "v_min_f32"},
    {"v_max_num_f16", "v_max_f16"},
    {"v_min_num_f16", "v_min_f16"},
    {"v_max_num_f64", "v_max_f64"},
    {"v_min_num_f64", "v_min_f64"},
    // v_maxmin_num_f32/v_minmax_num_f32 removed from rename table —
    // they are decomposed into v_max+v_min pairs by the handler above
    // (neither v_maxmin_f32 nor v_maxmin_num_f32 exist on GFX9)
    {"v_add_nc_u32", "v_add_u32"},
    {"v_sub_nc_u32", "v_sub_u32"},
    {"v_add_nc_i32", "v_add_i32"},
    {"v_sub_nc_i32", "v_sub_i32"},
    // v_fmac_f32/f16/f64: available on gfx942 (CDNA3) natively — no rename needed
    // v_pk_*_num_* → v_pk_* (GFX12 adds _num_ for IEEE compliance, GFX9 omits it)
    {"v_pk_add_num_f16", "v_pk_add_f16"},
    {"v_pk_mul_num_f16", "v_pk_mul_f16"},
    {"v_pk_max_num_f16", "v_pk_max_f16"},
    {"v_pk_min_num_f16", "v_pk_min_f16"},
    {"v_pk_fma_num_f16", "v_pk_fma_f16"},
    {"v_clz_i32_u32", "v_ffbh_u32"},
    {"v_ctz_i32_b32", "v_ffbl_b32"},
    {"v_add_co_ci_u32", "v_addc_co_u32"},
    {"v_sub_co_ci_u32", "v_subb_co_u32"},
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
  // s_wait_xcnt: GFX12 "transaction count" — waits for global store completion.
  // On GFX9, global stores use vmcnt (NOT expcnt).  Map xcnt → vmcnt so that
  // global_store completions are properly waited on.
  if (mnemonic == "s_wait_xcnt") {
    return "s_waitcnt vmcnt(" + std::to_string(count) + ")";
  }
  // s_wait_asynccnt, s_wait_tensorcnt — no GFX9 equivalent.
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

  // v_permlane16/v_permlanex16 — now emulated via ds_bpermute in TranslateInstruction

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

static std::vector<std::string> WidenExecOperation(const std::string& line,
                                                    bool compact_mode = false,
                                                    int cmpx_temp_sgpr = 16) {
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
          (void)reg_num;
          // ALWAYS use manual b32 sequence for saveexec.
          std::string src32 = src;
          if (src32 == "vcc") src32 = "vcc_lo";
          bool is_or = (b64_mnem.find("s_or_saveexec") == 0);
          bool is_andn2 = b64_mnem.find("andn2") != std::string::npos;

          if (is_andn2 && dst == src32) {
            // s_and_not1_saveexec_b32 sN, sN (dst == src, same register).
            //
            // GFX12 semantics of s_and_not1_saveexec_b32 (AMD "NOT1" = NOT exec):
            //   SDST = EXEC   (save old exec to destination)
            //   EXEC = SSRC & ~EXEC  (NOT of operand 1 = NOT EXEC, NOT NOT SSRC!)
            //
            // So exec_new = old_sN & ~old_exec_lo.
            //
            // We cannot do this in-place atomically on GFX9 because sN == exec_lo source.
            // Use cmpx_temp_sgpr as a scratch register:
            //   s_mov_b32 cmpx_temp, sN      ← save old sN (= source mask) before overwriting
            //   s_mov_b32 sN, exec_lo         ← sN = old exec_lo (destination)
            //   s_andn2_b32 exec_lo, cmpx_temp, sN  ← exec = cmpx_temp & ~sN
            //                                         = old_sN & ~old_exec ✓
            std::string stemp = "s" + std::to_string(cmpx_temp_sgpr);
            result.push_back("s_mov_b32 " + stemp + ", " + src32 +
                              " ; save src mask (dst==src conflict for not1_saveexec)");
            result.push_back("s_mov_b32 " + dst + ", exec_lo");
            result.push_back("s_andn2_b32 exec_lo, " + stemp + ", " + dst +
                              " ; exec = old_src & ~old_exec");
          } else {
            result.push_back("s_mov_b32 " + dst + ", exec_lo");
            if (is_or) {
              result.push_back("s_or_b32 exec_lo, exec_lo, " + src32);
            } else if (is_andn2) {
              // s_and_not1_saveexec_b32 sN, sM (dst != src):
              // Semantics: sN = exec; exec = sM & ~exec
              // After s_mov sN=exec, sN holds old exec:
              result.push_back("s_andn2_b32 exec_lo, " + src32 + ", " + dst +
                                " ; exec = src & ~old_exec");
            } else {
              result.push_back("s_and_b32 exec_lo, exec_lo, " + src32);
            }
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

  // Temp VGPR names: scale_temp_vgpr is guaranteed to be above the kernel's
  // own VGPR allocation (save_vgpr_y + 1), so these never collide with live
  // registers used by the kernel.  Previously hardcoded as v5/v6/v7/v8 which
  // clobbered live data in device-library-heavy kernels (gelu, layernorm, f64).
  //
  // IMPORTANT: scale_temp_vgpr must be even-aligned so that f64 VGPR pairs
  // (vt0:vt1, vt2:vt3) start on even boundaries as required by GFX9.
  // The caller ensures this by rounding up if necessary.
  std::string vt0 = "v" + std::to_string(scale_temp_vgpr);      // primary temp (even)
  std::string vt1 = "v" + std::to_string(scale_temp_vgpr + 1);  // secondary temp (odd)
  std::string vt2 = "v" + std::to_string(scale_temp_vgpr + 2);  // tertiary temp (even)
  std::string vt3 = "v" + std::to_string(scale_temp_vgpr + 3);  // quaternary temp (odd)

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
  // v_bitop2/v_bitop3 is handled by the detailed handler below (with truth table)
  // — do NOT catch it here with a simple v_and_b32 approximation.

  // ─── Wait counter translation ───
  if (IsWaitInstruction(mnemonic)) {
    result.push_back(TranslateWaitInstruction(line));
    return result;
  }

  // ─── GFX12 src_flat_scratch_base → flat_scratch (register rename) ───
  // GFX12 uses src_flat_scratch_base_lo as a 64-bit register alias for the
  // flat scratch base (containing both lo and hi halves).
  // GFX9 uses flat_scratch (SGPR hardware register pair).
  {
    size_t fpos;
    while ((fpos = line.find("src_flat_scratch_base_lo")) != std::string::npos)
      line.replace(fpos, 24, "flat_scratch");
    while ((fpos = line.find("src_flat_scratch_base_hi")) != std::string::npos)
      line.replace(fpos, 24, "flat_scratch_hi");
  }

  // ─── GFX12 scheduling/clause hints ───
  // s_delay_alu / s_wait_alu: GFX12 dependency-tracking hints.
  // GFX9 has no equivalent instruction, but the hazards they describe are REAL:
  // - VALU_DEP_N: a VGPR written by a recent VALU instruction is being read.
  //   On GFX9, v_readfirstlane_b32 and SALU ops that read VGPRs need at least
  //   1 wait cycle after the producing VALU instruction.
  // - TRANS32_DEP_N: transcendental instruction (v_rcp, v_rsq, etc.) producing
  //   a VGPR that is immediately consumed. On GFX9 transcendentals have ~4
  //   cycle latency visible to dependent instructions.
  // - SALU_CYCLE_N: SALU result used immediately; SALU→SALU or SALU→VALU
  //   forwarding requires at least 1 cycle on GFX9.
  // Emit s_nop 0 as a conservative 1-cycle stall for all delay variants.
  // s_clause / s_set_inst_prefetch_distance: prefetch grouping hints with no
  // GFX9 equivalent and no associated hazard — safe to skip entirely.
  if (mnemonic == "s_wait_alu" || mnemonic == "s_delay_alu") {
    result.push_back("s_nop 0 ; hazard delay from " + mnemonic);
    return result;
  }
  if (mnemonic == "s_clause" || mnemonic == "s_set_inst_prefetch_distance") {
    return result;  // no GFX9 equivalent, no hazard
  }

  // ─── VOP3-only 64-bit ops: strip _e32 suffix ───
  // GFX12 may emit v_lshlrev_b64_e32, v_mul_f64_e32, etc. but GFX9 only has
  // VOP3 encoding (no VOP2/VOP1/_e32 form) for 64-bit shift/logic ops AND
  // for all f64 VOP instructions. Strip _e32.
  {
    bool is_vop3_only_64bit = false;
    // 64-bit shift/logic
    if (mnemonic == "v_lshlrev_b64_e32" || mnemonic == "v_lshrrev_b64_e32" ||
        mnemonic == "v_ashrrev_i64_e32" || mnemonic == "v_lshlrev_b64" ||
        mnemonic == "v_lshrrev_b64" || mnemonic == "v_ashrrev_i64")
      is_vop3_only_64bit = true;
    // f64 VOP1/VOP2/VOP3 instructions: GFX9 has no _e32 encoding for these
    if (mnemonic.find("_f64") != std::string::npos &&
        mnemonic.find("_e32") != std::string::npos && mnemonic[0] == 'v')
      is_vop3_only_64bit = true;
    if (is_vop3_only_64bit) {
      std::string base_mnem = mnemonic;
      size_t e32_pos = base_mnem.find("_e32");
      if (e32_pos != std::string::npos) base_mnem.erase(e32_pos);
      line = ReplaceMnemonic(line, mnemonic, base_mnem);
      mnemonic = base_mnem;
    }
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
  // Note: s_code_end is NOT emitted here because the gfx942 assembler doesn't
  // support it as a mnemonic. Instead, s_code_end (0xBF9F0000) is written as
  // raw bytes in the .text padding after assembly (see ELF replacement step).
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

  // ─── v_mov_b16 → v_mov_b32 (GFX12-only 16-bit move) ───
  // GFX12 has v_mov_b16 for 16-bit register moves. GFX9 doesn't have it.
  // Widen to v_mov_b32 — the upper 16 bits are undefined anyway for packed ops.
  if (mnemonic == "v_mov_b16" || mnemonic == "v_mov_b16_e32" ||
      mnemonic == "v_mov_b16_e64") {
    std::string new_mnem = "v_mov_b32_e32";
    line = ReplaceMnemonic(line, mnemonic, new_mnem);
    mnemonic = new_mnem;
    result.push_back(line);
    return result;
  }

  // ─── s_bitreplicate_b64_b32 → s_mov_b32 pair ───
  // GFX12: s_bitreplicate_b64_b32 s[D:D+1], sS → each bit of sS is replicated
  // to 2 bits in the 64-bit result. For wave32-in-wave64 translation, we only
  // need the lower 32 bits (exec_lo operates on lanes 0-31).
  // Simplified emulation: s[D] = sS, s[D+1] = 0 (preserves wave32 behavior).
  if (mnemonic == "s_bitreplicate_b64_b32") {
    std::string ops = line.substr(line.find(mnemonic) + mnemonic.size());
    std::vector<std::string> operands;
    std::istringstream oss(ops);
    std::string tok;
    while (std::getline(oss, tok, ',')) {
      size_t s = tok.find_first_not_of(" \t");
      size_t e = tok.find_last_not_of(" \t");
      if (s != std::string::npos) operands.push_back(tok.substr(s, e - s + 1));
    }
    if (operands.size() >= 2) {
      // Parse s[D:D+1] pair
      std::string dst_pair = operands[0];
      std::string src = operands[1];
      std::regex pair_re(R"(s\[(\d+):(\d+)\])");
      std::smatch m;
      if (std::regex_match(dst_pair, m, pair_re)) {
        int d_lo = std::stoi(m[1]);
        int d_hi = std::stoi(m[2]);
        result.push_back("s_mov_b32 s" + std::to_string(d_lo) + ", " + src);
        result.push_back("s_mov_b32 s" + std::to_string(d_hi) + ", 0");
      } else {
        result.push_back("s_mov_b32 " + dst_pair + ", " + src);
      }
      return result;
    }
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
      // GFX12 s_add_nc_u64 does NOT write SCC.  GFX9 s_add_u32+s_addc_u32
      // both write SCC.  We intentionally do NOT save/restore SCC here.
      // The s_cselect_b32/s_cmp_lg_u32 SCC save/restore was causing the LLVM
      // MC assembler on gfx942 to insert additional hazard workaround
      // instructions that non-deterministically interfered with the SALU
      // pipeline.  In practice, all callers of s_add_nc_u64 in translated
      // kernels are followed by s_cmp (which overwrites SCC) or s_branch
      // (which ignores SCC), so SCC preservation is not required.
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
        // Immediate: move to temp VGPRs (scale_temp_vgpr and scale_temp_vgpr+1).
        // Must use the dynamic temp, NOT hardcoded v252/v253 which are out-of-bounds.
        std::string tmp_lo = "v" + std::to_string(scale_temp_vgpr);
        std::string tmp_hi = "v" + std::to_string(scale_temp_vgpr + 1);
        result.push_back("v_mov_b32_e32 " + tmp_lo + ", " + a.imm);
        result.push_back("v_mov_b32_e32 " + tmp_hi + ", 0");
        src0_pair = "v[" + std::to_string(scale_temp_vgpr) + ":" +
                    std::to_string(scale_temp_vgpr + 1) + "]";
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

  // ─── v_bitop2_b32 / v_bitop3_b32 → emulate (GFX12 programmable 3-input bitop) ───
  // v_bitop2_b32 vdst, src0, src1 bitop3:0xNN
  // The bitop3 byte is a truth table indexed by (src0_bit<<2 | src1_bit<<1 | vdst_old_bit):
  //   bit 0: f(0,0,0), bit 1: f(0,0,1), bit 2: f(0,1,0), bit 3: f(0,1,1),
  //   bit 4: f(1,0,0), bit 5: f(1,0,1), bit 6: f(1,1,0), bit 7: f(1,1,1)
  //
  // Decomposition strategy:
  //   1. Try direct mapping for common truth tables (single-instruction).
  //   2. For general case, use Shannon expansion on src0:
  //      result = (src0 & f1(src1, vdst_old)) | (~src0 & f0(src1, vdst_old))
  //      where f1 = tt restricted to src0=1, f0 = tt restricted to src0=0.
  //      Each f is a 2-input function of (src1, vdst_old) — emittable in 1 instruction.
  //      Use v_bfi_b32 to combine: v_bfi_b32 vdst, src0, f1_result, f0_result.
  if (mnemonic.find("v_bitop2_b32") == 0 || mnemonic.find("v_bitop3_b32") == 0) {
    std::string ops = line.substr(line.find(mnemonic) + mnemonic.size());
    size_t bitop_pos = ops.find("bitop3:");
    if (bitop_pos != std::string::npos) {
      int tt = 0;
      std::string hex_str = ops.substr(bitop_pos + 7);
      try { tt = std::stoi(hex_str, nullptr, 0); } catch (...) {}

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
        std::string tmp0 = "v" + std::to_string(scale_temp_vgpr);
        std::string tmp1 = "v" + std::to_string(scale_temp_vgpr + 1);

        // ── Common single-instruction truth tables ──
        bool handled = true;
        if (tt == 0x00) {
          // All zeros
          result.push_back("v_mov_b32_e32 " + vdst + ", 0");
        } else if (tt == 0xFF) {
          // All ones
          result.push_back("v_mov_b32_e32 " + vdst + ", -1");
        } else if (tt == 0xF0) {
          // src0
          if (vdst != src0) result.push_back("v_mov_b32_e32 " + vdst + ", " + src0);
          // else: no-op (vdst already is src0 — but that can't happen since vdst != src0 in the ISA)
        } else if (tt == 0xCC) {
          // src1
          if (vdst != src1) result.push_back("v_mov_b32_e32 " + vdst + ", " + src1);
        } else if (tt == 0xAA) {
          // vdst_old (no-op: output unchanged)
          result.push_back("s_nop 0 ; bitop3 0xAA = identity (vdst unchanged)");
        } else if (tt == 0x0F) {
          // ~src0
          result.push_back("v_not_b32_e32 " + vdst + ", " + src0);
        } else if (tt == 0x33) {
          // ~src1
          result.push_back("v_not_b32_e32 " + vdst + ", " + src1);
        } else if (tt == 0x55) {
          // ~vdst_old
          result.push_back("v_not_b32_e32 " + vdst + ", " + vdst);
        } else if (tt == 0xC0) {
          // src0 & src1
          result.push_back("v_and_b32 " + vdst + ", " + src0 + ", " + src1);
        } else if (tt == 0xFC) {
          // src0 | src1
          result.push_back("v_or_b32 " + vdst + ", " + src0 + ", " + src1);
        } else if (tt == 0x3C) {
          // src0 ^ src1
          result.push_back("v_xor_b32 " + vdst + ", " + src0 + ", " + src1);
        } else if (tt == 0x30) {
          // src0 & ~src1
          result.push_back("v_andn2_b32 " + vdst + ", " + src0 + ", " + src1);
        } else if (tt == 0x0C) {
          // ~src0 & src1
          result.push_back("v_andn2_b32 " + vdst + ", " + src1 + ", " + src0);
        } else if (tt == 0xA0) {
          // src0 & vdst_old
          result.push_back("v_and_b32 " + vdst + ", " + src0 + ", " + vdst);
        } else if (tt == 0x88) {
          // src1 & vdst_old
          result.push_back("v_and_b32 " + vdst + ", " + src1 + ", " + vdst);
        } else if (tt == 0xFA) {
          // src0 | vdst_old
          result.push_back("v_or_b32 " + vdst + ", " + src0 + ", " + vdst);
        } else if (tt == 0xEE) {
          // src1 | vdst_old
          result.push_back("v_or_b32 " + vdst + ", " + src1 + ", " + vdst);
        } else if (tt == 0x5A) {
          // src0 ^ vdst_old
          result.push_back("v_xor_b32 " + vdst + ", " + src0 + ", " + vdst);
        } else if (tt == 0x66) {
          // src1 ^ vdst_old
          result.push_back("v_xor_b32 " + vdst + ", " + src1 + ", " + vdst);
        } else if (tt == 0x3F) {
          // ~(src0 & src1) = NAND
          result.push_back("v_and_b32 " + vdst + ", " + src0 + ", " + src1);
          result.push_back("v_not_b32_e32 " + vdst + ", " + vdst);
        } else if (tt == 0x03) {
          // ~(src0 | src1) = NOR
          result.push_back("v_or_b32 " + vdst + ", " + src0 + ", " + src1);
          result.push_back("v_not_b32_e32 " + vdst + ", " + vdst);
        } else if (tt == 0xCA) {
          // (src0 & src1) | (~src0 & vdst) = bitwise select (BFI)
          result.push_back("v_bfi_b32 " + vdst + ", " + src0 + ", " + src1 + ", " + vdst);
        } else if (tt == 0xAC) {
          // (src1 & vdst) | (~src1 & src0) = BFI with src1 as selector
          result.push_back("v_bfi_b32 " + vdst + ", " + src1 + ", " + vdst + ", " + src0);
        } else if (tt == 0xE2) {
          // (vdst_old & src0) | (~vdst_old & src1) = BFI with vdst as selector
          // vdst_old selects: bit=1 → src0, bit=0 → src1
          // v_bfi_b32 uses first arg as selector: dst = (sel & src2) | (~sel & src3)
          // Need: (vdst_old & src0) | (~vdst_old & src1)
          // = v_bfi_b32 tmp, vdst, src0, src1; mov vdst, tmp
          result.push_back("v_bfi_b32 " + tmp0 + ", " + vdst + ", " + src0 + ", " + src1);
          result.push_back("v_mov_b32_e32 " + vdst + ", " + tmp0);
        } else if (tt == 0x80) {
          // src0 & src1 & vdst_old
          if (src1 == vdst) {
            result.push_back("v_and_b32 " + vdst + ", " + src0 + ", " + vdst);
          } else {
            result.push_back("v_and_b32 " + tmp0 + ", " + src0 + ", " + src1);
            result.push_back("v_and_b32 " + vdst + ", " + tmp0 + ", " + vdst);
          }
        } else if (tt == 0x40) {
          // src0 & src1 & ~vdst_old
          if (src1 == vdst) {
            result.push_back("v_mov_b32_e32 " + vdst + ", 0");
          } else {
            result.push_back("v_and_b32 " + tmp0 + ", " + src0 + ", " + src1);
            result.push_back("v_andn2_b32 " + vdst + ", " + tmp0 + ", " + vdst);
          }
        } else {
          handled = false;
        }

        if (handled) return result;

        // ── General case: Shannon expansion on src0 ──
        // Split tt into f0 (src0=0) and f1 (src0=1):
        //   f0 = {tt[0], tt[1], tt[2], tt[3]} (bits where src0=0)
        //   f1 = {tt[4], tt[5], tt[6], tt[7]} (bits where src0=1)
        // Each is a 2-input truth table indexed by (src1_bit<<1 | vdst_old_bit).
        uint8_t f0 = tt & 0x0F;        // lower nibble
        uint8_t f1 = (tt >> 4) & 0x0F; // upper nibble

        // Emit a 2-input function of (src1, vdst_old) into dst_reg.
        // 2-input truth table indexed by (src1_bit<<1 | vdst_old_bit):
        //   bit 0: f(0,0), bit 1: f(0,1), bit 2: f(1,0), bit 3: f(1,1)
        auto emit_2input = [&](uint8_t f2, const std::string& dst,
                               const std::string& s1, const std::string& vd_old) {
          switch (f2) {
            case 0x0: result.push_back("v_mov_b32_e32 " + dst + ", 0"); break;
            case 0xF: result.push_back("v_mov_b32_e32 " + dst + ", -1"); break;
            case 0xC: // src1
              if (dst != s1) result.push_back("v_mov_b32_e32 " + dst + ", " + s1);
              break;
            case 0xA: // vdst_old
              if (dst != vd_old) result.push_back("v_mov_b32_e32 " + dst + ", " + vd_old);
              break;
            case 0x3: // ~src1
              result.push_back("v_not_b32_e32 " + dst + ", " + s1); break;
            case 0x5: // ~vdst_old
              result.push_back("v_not_b32_e32 " + dst + ", " + vd_old); break;
            case 0x8: // src1 & vdst_old
              result.push_back("v_and_b32 " + dst + ", " + s1 + ", " + vd_old); break;
            case 0xE: // src1 | vdst_old
              result.push_back("v_or_b32 " + dst + ", " + s1 + ", " + vd_old); break;
            case 0x6: // src1 ^ vdst_old
              result.push_back("v_xor_b32 " + dst + ", " + s1 + ", " + vd_old); break;
            case 0x4: // src1 & ~vdst_old
              result.push_back("v_andn2_b32 " + dst + ", " + s1 + ", " + vd_old); break;
            case 0x2: // ~src1 & vdst_old
              result.push_back("v_andn2_b32 " + dst + ", " + vd_old + ", " + s1); break;
            case 0x1: // ~src1 & ~vdst_old = ~(src1 | vdst_old)
              result.push_back("v_or_b32 " + dst + ", " + s1 + ", " + vd_old);
              result.push_back("v_not_b32_e32 " + dst + ", " + dst); break;
            case 0x7: // ~(src1 & vdst_old) = NAND
              result.push_back("v_and_b32 " + dst + ", " + s1 + ", " + vd_old);
              result.push_back("v_not_b32_e32 " + dst + ", " + dst); break;
            case 0x9: // ~(src1 ^ vdst_old) = XNOR
              result.push_back("v_xor_b32 " + dst + ", " + s1 + ", " + vd_old);
              result.push_back("v_not_b32_e32 " + dst + ", " + dst); break;
            case 0xB: // src1 | ~vdst_old = ~(~src1 & vdst_old) = ~(v_andn2 vd_old, s1)
              result.push_back("v_andn2_b32 " + dst + ", " + vd_old + ", " + s1);
              result.push_back("v_not_b32_e32 " + dst + ", " + dst); break;
            case 0xD: // ~src1 | vdst_old = ~(src1 & ~vdst_old) = ~(v_andn2 s1, vd_old)
              result.push_back("v_andn2_b32 " + dst + ", " + s1 + ", " + vd_old);
              result.push_back("v_not_b32_e32 " + dst + ", " + dst); break;
            default:
              // All 16 2-input truth tables (0x0-0xF) are covered above.
              // This path is unreachable; assert to catch logic errors.
              assert(false && "unreachable: all 16 2-input bitop cases covered");
              break;
          }
        };

        // Special cases where one half is trivial:
        if (f0 == f1) {
          // src0 doesn't matter — result depends only on src1 and vdst_old
          emit_2input(f0, vdst, src1, vdst);
        } else if (f0 == 0x0 && f1 != 0x0) {
          // result = src0 & f1(src1, vdst_old)
          emit_2input(f1, tmp0, src1, vdst);
          result.push_back("v_and_b32 " + vdst + ", " + src0 + ", " + tmp0);
        } else if (f1 == 0x0 && f0 != 0x0) {
          // result = ~src0 & f0(src1, vdst_old)
          emit_2input(f0, tmp0, src1, vdst);
          result.push_back("v_andn2_b32 " + vdst + ", " + tmp0 + ", " + src0);
        } else if (f0 == 0xF) {
          // result = ~src0 | f1(src1, vdst_old) = ~(src0 & ~f1) = ~(v_andn2 src0, f1)
          emit_2input(f1, tmp0, src1, vdst);
          result.push_back("v_andn2_b32 " + vdst + ", " + src0 + ", " + tmp0);
          result.push_back("v_not_b32_e32 " + vdst + ", " + vdst);
        } else if (f1 == 0xF) {
          // result = src0 | f0(src1, vdst_old)
          emit_2input(f0, tmp0, src1, vdst);
          result.push_back("v_or_b32 " + vdst + ", " + src0 + ", " + tmp0);
        } else {
          // General case: use v_bfi_b32 to mux between f1 and f0 based on src0
          // v_bfi_b32 dst, selector, val_when_1, val_when_0
          // Save vdst_old to tmp1 before any clobbering (f0/f1 may need it)
          result.push_back("v_mov_b32_e32 " + tmp1 + ", " + vdst);
          emit_2input(f1, tmp0, src1, tmp1);
          emit_2input(f0, vdst, src1, tmp1);
          // Now: tmp0 = f1(src1, vdst_old), vdst = f0(src1, vdst_old)
          // result = (src0 & tmp0) | (~src0 & vdst) = v_bfi_b32 vdst, src0, tmp0, vdst
          result.push_back("v_bfi_b32 " + vdst + ", " + src0 + ", " + tmp0 + ", " + vdst);
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
      // Load the embedded literal into temp VGPR, then emit v_fma_f32
      const std::string& literal = (mnemonic == "v_fmamk_f32") ? operands[2] : operands[3];
      result.push_back("v_mov_b32_e32 " + vt0 + ", " + literal);
      if (mnemonic == "v_fmamk_f32") {
        // vdst = src0 * imm + vsrc2 → v_fma_f32 vdst, src0, vt0, vsrc2
        result.push_back("v_fma_f32 " + operands[0] + ", " + operands[1] + ", " + vt0 + ", " + operands[3]);
      } else {
        // vdst = src0 * vsrc1 + imm → v_fma_f32 vdst, src0, vsrc1, vt0
        result.push_back("v_fma_f32 " + operands[0] + ", " + operands[1] + ", " + operands[2] + ", " + vt0);
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

      // VOPD executes both halves in parallel: all sources are read before
      // any destinations are written. If the first half writes a register that
      // the second half reads, we must save it before the first half executes.
      //
      // Detect: extract dest of first half, check if second half reads it.
      auto extractDest = [](const std::string& instr) -> std::string {
        size_t mend = instr.find_first_of(" \t");
        if (mend == std::string::npos) return "";
        size_t ostart = instr.find_first_not_of(" \t", mend);
        if (ostart == std::string::npos) return "";
        size_t oend = instr.find_first_of(" \t,", ostart);
        return instr.substr(ostart, oend != std::string::npos ? oend - ostart : std::string::npos);
      };
      std::string first_dst = extractDest(first_half);
      // Check if second_half operands (after the dest) reference first_dst
      bool conflict = false;
      if (!first_dst.empty() && first_dst[0] == 'v') {
        // Look for first_dst as a source operand in second_half
        std::string sh_ops = second_half;
        size_t mend2 = sh_ops.find_first_of(" \t");
        if (mend2 != std::string::npos) {
          // Skip past dest (first operand) — look only at source operands
          size_t comma_pos = sh_ops.find(',', mend2);
          if (comma_pos != std::string::npos) {
            std::string src_part = sh_ops.substr(comma_pos);
            // Check if first_dst appears as a word boundary match in sources
            size_t pos = 0;
            while ((pos = src_part.find(first_dst, pos)) != std::string::npos) {
              size_t end = pos + first_dst.size();
              // Check it's a complete register name (not a prefix of a longer name)
              bool start_ok = (pos == 0 || !std::isalnum(src_part[pos-1]));
              bool end_ok = (end >= src_part.size() || !std::isalnum(src_part[end]));
              if (start_ok && end_ok) { conflict = true; break; }
              pos = end;
            }
          }
        }
      }

      if (conflict) {
        // Save first_dst to temp VGPR before first half modifies it.
        std::string tmp = "v" + std::to_string(scale_temp_vgpr);
        result.push_back("v_mov_b32_e32 " + tmp + ", " + first_dst);
        // Translate first half normally
        auto r1 = TranslateInstruction(first_half, source_cpu, target_cpu, scale_temp_vgpr, cmpx_temp_sgpr, compact_mode);
        result.insert(result.end(), r1.begin(), r1.end());
        // In second half, replace references to first_dst with temp
        std::string modified_second = second_half;
        // Replace as source operand only (skip the dest)
        size_t skip_dest = modified_second.find(',');
        if (skip_dest != std::string::npos) {
          std::string before = modified_second.substr(0, skip_dest);
          std::string after = modified_second.substr(skip_dest);
          size_t rpos = 0;
          while ((rpos = after.find(first_dst, rpos)) != std::string::npos) {
            size_t rend = rpos + first_dst.size();
            bool rs = (rpos == 0 || !std::isalnum(after[rpos-1]));
            bool re = (rend >= after.size() || !std::isalnum(after[rend]));
            if (rs && re) {
              after.replace(rpos, first_dst.size(), tmp);
              rpos += tmp.size();
            } else {
              rpos = rend;
            }
          }
          modified_second = before + after;
        }
        auto r2 = TranslateInstruction(modified_second, source_cpu, target_cpu, scale_temp_vgpr, cmpx_temp_sgpr, compact_mode);
        result.insert(result.end(), r2.begin(), r2.end());
      } else {
        // No conflict — safe to run sequentially.
        auto r1 = TranslateInstruction(first_half, source_cpu, target_cpu, scale_temp_vgpr, cmpx_temp_sgpr, compact_mode);
        auto r2 = TranslateInstruction(second_half, source_cpu, target_cpu, scale_temp_vgpr, cmpx_temp_sgpr, compact_mode);
        result.insert(result.end(), r1.begin(), r1.end());
        result.insert(result.end(), r2.begin(), r2.end());
      }
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
  //   v_mov_b32 vt0, ssrc0
  //   v_op_f32 vt0, ssrc1, vt0   (ssrc1 as inline constant or SGPR)
  //   v_readfirstlane_b32 sdst, vt0
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
          // Use v_fma_f32 for precision (single rounding step instead of mul+add).
          // GFX9 VOP3 constant bus: at most one SGPR. ssrc1 and sdst may be
          // different SGPRs. Move sdst (accumulator) to vt1 to avoid bus conflict.
          result.push_back("v_mov_b32_e32 " + vt0 + ", " + ssrc0);
          result.push_back("v_mov_b32_e32 " + vt1 + ", " + sdst);
          result.push_back("v_fma_f32 " + vt0 + ", " + vt0 + ", " + ssrc1 + ", " + vt1);
          result.push_back("s_nop 0 ; VGPR hazard: readfirstlane after VALU");
          result.push_back("v_readfirstlane_b32 " + sdst + ", " + vt0);
        } else if (mnemonic == "s_mul_f32") {
          // s_mul_f32 sdst, ssrc0, ssrc1 → sdst = ssrc0 * ssrc1
          // Use standard v_mul_f32 (same IEEE rounding as SALU float).
          result.push_back("v_mov_b32_e32 " + vt0 + ", " + ssrc0);
          result.push_back("v_mul_f32_e32 " + vt0 + ", " + ssrc1 + ", " + vt0);
          result.push_back("s_nop 0 ; VGPR hazard: readfirstlane after VALU");
          result.push_back("v_readfirstlane_b32 " + sdst + ", " + vt0);
        } else {
          // Other SALU float ops: vt0 = ssrc0; vt0 = op(ssrc1, vt0); sdst = vt0
          result.push_back("v_mov_b32_e32 " + vt0 + ", " + ssrc0);
          result.push_back(valu_op + " " + vt0 + ", " + ssrc1 + ", " + vt0);
          result.push_back("s_nop 0 ; VGPR hazard: readfirstlane after VALU");
          result.push_back("v_readfirstlane_b32 " + sdst + ", " + vt0);
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

        result.push_back(valu_mnem + " " + vt0 + ", " + operands[1]);
        result.push_back("s_nop 0 ; VGPR hazard: readfirstlane after VALU");
        result.push_back("v_readfirstlane_b32 " + operands[0] + ", " + vt0);
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
        result.push_back("v_sqrt_f32_e32 " + vt0 + ", " + operands[1]);
        result.push_back("s_nop 0 ; VGPR hazard: readfirstlane after VALU");
        result.push_back("v_readfirstlane_b32 " + operands[0] + ", " + vt0);
        return result;
      }
    }

    // v_s_rsq_f32 sdst, ssrc → VALU reciprocal sqrt + readfirstlane
    // GFX12 VALU→SGPR reciprocal square root; GFX9 must go through VGPR.
    if (mnemonic == "v_s_rsq_f32") {
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
        result.push_back("v_rsq_f32_e32 " + vt0 + ", " + operands[1]);
        result.push_back("s_nop 0 ; VGPR hazard: readfirstlane after VALU");
        result.push_back("v_readfirstlane_b32 " + operands[0] + ", " + vt0);
        return result;
      }
    }

    // s_cmp_*_f32 → VALU float compare setting SCC via VCC bridge.
    // GFX12 has SALU float compares (set SCC); GFX9 doesn't.
    // Emulate: v_mov_b32 vt0, literal; v_cmp_*_f32_e32 sA, vt0; s_cmp_lg_u32 vcc_lo, 0
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
          // Load second operand to temp VGPR (could be literal or SGPR)
          result.push_back("v_mov_b32_e32 " + vt0 + ", " + operands[1]);
          // VOPC: v_cmp_*_f32_e32 src0, vsrc1 → src0=sA (SGPR), vsrc1=vt0
          result.push_back(cmp_it->second + " " + operands[0] + ", " + vt0);
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
      // If both src0 and src1 are SGPRs we must move one to temp VGPR first.
      // When vdst == src2: also use temp VGPR to avoid read-before-write.
      bool src0_sgpr = !src0.empty() && src0[0] == 's';
      bool src1_sgpr = !src1.empty() && src1[0] == 's';
      if (vdst != src2) {
        if (src0_sgpr && src1_sgpr) {
          result.push_back("v_mov_b32_e32 " + vt0 + ", " + src0);
          result.push_back("v_mul_lo_u32 " + vdst + ", " + vt0 + ", " + src1);
        } else {
          result.push_back("v_mul_lo_u32 " + vdst + ", " + src0 + ", " + src1);
        }
        result.push_back("v_add_u32_e32 " + vdst + ", " + vdst + ", " + src2);
      } else {
        result.push_back("v_mov_b32_e32 " + vt0 + ", " + src0);
        result.push_back("v_mul_lo_u32 " + vt0 + ", " + vt0 + ", " + src1);
        result.push_back("v_add_u32_e32 " + vdst + ", " + vt0 + ", " + src2);
      }
      return result;
    }
  }

  // ─── v_add_nc_u64 → v_add_co_u32 + v_addc_co_u32 ───
  // GFX12 has v_add_nc_u64 for 64-bit integer add. GFX9 doesn't.
  // Emulate: vdst[0:1] = vsrc0[0:1] + vsrc1[0:1] using carry chain.
  if (mnemonic == "v_add_nc_u64" || mnemonic == "v_add_nc_u64_e64" ||
      mnemonic == "v_add_u64" || mnemonic == "v_add_u64_e64") {
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
    if (operands.size() >= 3) {
      // Parse register pairs: v[lo:hi]
      auto parse_pair = [](const std::string& r, std::string& lo, std::string& hi) {
        if (r.find('[') != std::string::npos) {
          size_t lb = r.find('['), colon = r.find(':'), rb = r.find(']');
          if (lb != std::string::npos && colon != std::string::npos && rb != std::string::npos) {
            std::string prefix = r.substr(0, lb);  // "v" or "s"
            lo = prefix + r.substr(lb + 1, colon - lb - 1);
            hi = prefix + r.substr(colon + 1, rb - colon - 1);
            return true;
          }
        }
        return false;
      };
      std::string dst_lo, dst_hi, src0_lo, src0_hi, src1_lo, src1_hi;
      bool dst_ok = parse_pair(operands[0], dst_lo, dst_hi);
      bool src0_ok = parse_pair(operands[1], src0_lo, src0_hi);
      bool src1_ok = parse_pair(operands[2], src1_lo, src1_hi);
      if (dst_ok && src0_ok && src1_ok) {
        // GFX9 constant bus: v_addc_co_u32_e64 has vcc as carry-in (src2),
        // which counts as a constant bus source. If either src0_hi or src1_hi
        // is SGPR, that would be 2 constant bus sources. Move SGPRs to VGPRs.
        bool s0_sgpr = !src0_lo.empty() && (src0_lo[0] == 's' || src0_lo.find("flat_") == 0);
        bool s1_sgpr = !src1_lo.empty() && (src1_lo[0] == 's' || src1_lo.find("flat_") == 0);
        // Always move at least one SGPR source to temp VGPRs for the addc
        if (s0_sgpr || s1_sgpr) {
          // Move src1 to temp VGPRs (always safe — no constant bus conflict)
          result.push_back("v_mov_b32_e32 " + vt0 + ", " + src1_lo);
          result.push_back("v_mov_b32_e32 " + vt1 + ", " + src1_hi);
          if (s0_sgpr && s1_sgpr) {
            // Both SGPRs: also move src0 for the add (constant bus)
            result.push_back("v_mov_b32_e32 " + vt2 + ", " + src0_lo);
            result.push_back("v_mov_b32_e32 " + vt3 + ", " + src0_hi);
            result.push_back("v_add_co_u32_e64 " + dst_lo + ", vcc, " + vt2 + ", " + vt0);
            result.push_back("v_addc_co_u32_e64 " + dst_hi + ", vcc, " + vt3 + ", " + vt1 + ", vcc");
          } else {
            result.push_back("v_add_co_u32_e64 " + dst_lo + ", vcc, " + src0_lo + ", " + vt0);
            result.push_back("v_addc_co_u32_e64 " + dst_hi + ", vcc, " + src0_hi + ", " + vt1 + ", vcc");
          }
        } else {
          result.push_back("v_add_co_u32_e64 " + dst_lo + ", vcc, " + src0_lo + ", " + src1_lo);
          result.push_back("v_addc_co_u32_e64 " + dst_hi + ", vcc, " + src0_hi + ", " + src1_hi + ", vcc");
        }
        return result;
      }
    }
  }

  // ─── v_cndmask_b32_e64: bare SGPR mask widening + constant bus fix ───
  // GFX12 wave32: cmp results are 32-bit (single SGPR sN as mask).
  // GFX9 wave64: mask must be a 64-bit SGPR pair s[N:N+1].
  // Also: if src1 is SGPR and mask is SGPR, move src1 to temp VGPR (constant bus).
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
      // Constant bus fix: GFX9 VOP3 can read at most one SGPR from the constant bus.
      // The mask (operands[3]) uses one SGPR slot. If src0 or src1 is a different SGPR
      // (or a literal not inline-encodable), move it to a VGPR.
      auto is_sgpr = [](const std::string& s) -> bool {
        return !s.empty() && s[0] == 's' && s.size() > 1 &&
               (std::isdigit((unsigned char)s[1]) || s[1] == '[');
      };
      auto is_literal = [](const std::string& s) -> bool {
        if (s.empty()) return false;
        std::string t = s;
        if (t[0] == '-') t = t.substr(1);
        return t.size() > 2 && t[0] == '0' && (t[1] == 'x' || t[1] == 'X');
      };
      bool mask_sgpr = !mask.empty() && (mask[0] == 's' ||
                       mask.substr(0, 3) == "vcc" || mask.substr(0, 4) == "exec");
      std::string& src0 = operands[1];
      std::string& src1 = operands[2];
      // If src0 is SGPR (different from mask's SGPR) or a literal, move to vt0
      if ((is_sgpr(src0) || is_literal(src0)) && mask_sgpr) {
        result.push_back("v_mov_b32_e32 " + vt0 + ", " + src0);
        src0 = vt0;
      }
      // If src1 is also SGPR/literal, move to vt1 (vt0 may already be used for src0)
      if ((is_sgpr(src1) || is_literal(src1)) && mask_sgpr) {
        std::string tmp = (src0 == vt0) ? vt1 : vt0;
        result.push_back("v_mov_b32_e32 " + tmp + ", " + src1);
        src1 = tmp;
      }
      std::string fixed = mnemonic + " " + operands[0];
      for (size_t i = 1; i < operands.size(); ++i) fixed += ", " + operands[i];
      result.push_back(fixed);
      return result;
    }
  }

  // ─── v_cndmask_b32_e32 SGPR src0 → move to temp VGPR (constant bus fix) ───
  // GFX12 (wave32) VOP2 form may have an SGPR src0.  On GFX9 (wave64) the
  // implicit VCC mask already occupies the constant bus, so src0 cannot also
  // be SGPR.  Move src0 to temp VGPR.  The explicit vcc_lo operand (from GFX12
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
        // Use scale_temp_vgpr (above kernel's own VGPRs) to avoid clobbering
        // live VGPRs which may hold kernel-computed values.
        std::string tmp = "v" + std::to_string(scale_temp_vgpr);
        result.push_back("v_mov_b32_e32 " + tmp + ", " + src0);
        src0 = tmp;
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

  // ─── VOP3 with 32-bit literal in source → v_mov_b32 + replace with VGPR ───
  // GFX9 VOP3 does not support literal constants in source operand positions.
  // GFX12 compilers freely emit e.g. v_fma_f32 vdst, 0x3fb8aa3b, vsrc1, vsrc2
  // or even v_mad_u32_u24 vdst, 0x64, vsrc, 0x64 (two literals).
  // Fix: emit v_mov_b32_e32 to load each literal into a temp VGPR (vt0, vt1),
  // then use the VGPR in the instruction.
  // Applies generically to ANY VOP3 instruction (3+ source operands, 4+ total).
  {
    // Any v_ instruction with 4+ operands is a VOP3 candidate.
    // Also catch VOP2 instructions that are VOP3-only on GFX9 (e.g., v_mad_u32_u24).
    // EXCLUDE v_cmp_* and v_cmpx_* — they have dedicated handlers below that also
    // handle SGPR dest widening, VCC clobber save/restore, and f64 literal pairs.
    bool is_vop3_candidate = false;
    if (mnemonic[0] == 'v' && mnemonic[1] == '_' &&
        mnemonic.find("v_cmp") != 0) {
      size_t ops_start = line.find(mnemonic) + mnemonic.size();
      std::string ops = line.substr(ops_start);
      // Quick comma count to determine operand count
      int comma_count = 0;
      for (char c : ops) if (c == ',') comma_count++;
      // 3+ commas means 4+ operands (1 dst + 3 src) — VOP3
      // Also catch any instruction with at least 2 commas that has a hex literal
      // (some VOP2 instructions can appear with literals that aren't inline-encodable)
      if (comma_count >= 3) is_vop3_candidate = true;
      // For 2-comma instructions (3 operands), check if it's a known VOP3-only op
      // or has _e64 suffix
      if (comma_count == 2 && (mnemonic.find("_e64") != std::string::npos ||
                                mnemonic == "v_ldexp_f32" || mnemonic == "v_ldexp_f64" ||
                                mnemonic == "v_ldexp_f16"))
        is_vop3_candidate = true;
    }
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
      // Check sources (operands[1..n]) for large literals (hex or large decimal)
      auto is_non_inline_literal = [](const std::string& raw) -> bool {
        std::string s = raw;
        // Strip negation/abs modifiers
        if (!s.empty() && s[0] == '-') s = s.substr(1);
        if (s.empty()) return false;
        // Hex literal (0x...) — always non-inline unless very small
        if (s.size() > 2 && s[0] == '0' && (s[1] == 'x' || s[1] == 'X')) return true;
        // Large decimal literal (> 64 or < -16)
        if (std::isdigit((unsigned char)s[0])) {
          try {
            long val = std::stol(s);
            if (val > 64) return true;
          } catch (...) {}
        }
        return false;
      };
      if (operands.size() >= 3) {
        // Collect all literal positions and use temp VGPRs (above kernel allocation)
        int temp_vgprs[] = {scale_temp_vgpr, scale_temp_vgpr + 1};
        int temp_idx = 0;
        bool any_fixed = false;
        for (size_t i = 1; i < operands.size() && temp_idx < 2; ++i) {
          std::string op = operands[i];
          bool neg = !op.empty() && op[0] == '-';
          std::string bare = neg ? op.substr(1) : op;
          if (is_non_inline_literal(bare)) {
            int vgpr = temp_vgprs[temp_idx++];
            result.push_back("v_mov_b32_e32 v" + std::to_string(vgpr) + ", " + bare);
            operands[i] = std::string(neg ? "-" : "") + "v" + std::to_string(vgpr);
            any_fixed = true;
          }
        }
        if (any_fixed) {
          std::string fixed = mnemonic + " " + operands[0];
          for (size_t j = 1; j < operands.size(); ++j) fixed += ", " + operands[j];
          result.push_back(fixed);
          return result;
        }
      }
    }
  }

  // ─── v_fma_mix_f32/f16: constant bus fix for multiple SGPRs ───
  // GFX9 VOP3 can read at most one SGPR. v_fma_mix_f32 with two different
  // SGPR source operands violates this. Move one to a temp VGPR.
  if (mnemonic == "v_fma_mix_f32" || mnemonic == "v_fma_mix_f16" ||
      mnemonic == "v_fma_mixlo_f16" || mnemonic == "v_fma_mixhi_f16") {
    size_t ops_start = line.find(mnemonic) + mnemonic.size();
    // Split the operand part (before any modifiers like op_sel_hi)
    std::string ops = line.substr(ops_start);
    // Extract the main operands (up to but not including op_sel/clamp modifiers)
    size_t mod_pos = ops.find("op_sel");
    if (mod_pos == std::string::npos) mod_pos = ops.find("clamp");
    std::string main_ops = (mod_pos != std::string::npos) ? ops.substr(0, mod_pos) : ops;
    std::string modifiers = (mod_pos != std::string::npos) ? ops.substr(mod_pos) : "";
    std::vector<std::string> operands;
    std::istringstream oss(main_ops);
    std::string tok;
    while (std::getline(oss, tok, ',')) {
      size_t s = tok.find_first_not_of(" \t");
      size_t e = tok.find_last_not_of(" \t");
      if (s != std::string::npos) operands.push_back(tok.substr(s, e - s + 1));
    }
    if (operands.size() >= 4) {
      auto is_sgpr = [](const std::string& s) -> bool {
        return !s.empty() && s[0] == 's' && s.size() > 1 &&
               (std::isdigit((unsigned char)s[1]) || s[1] == '[');
      };
      // Count distinct SGPRs used as sources
      int sgpr_count = 0;
      int first_sgpr_idx = -1;
      for (int i = 1; i <= 3 && i < (int)operands.size(); i++) {
        if (is_sgpr(operands[i])) {
          sgpr_count++;
          if (first_sgpr_idx < 0) first_sgpr_idx = i;
        }
      }
      // If more than one SGPR source, move all but one to VGPRs
      if (sgpr_count > 1) {
        int vgpr_tmp = scale_temp_vgpr;
        bool first = true;
        for (int i = 1; i <= 3 && i < (int)operands.size(); i++) {
          if (is_sgpr(operands[i])) {
            if (first) { first = false; continue; }  // keep one SGPR
            std::string tmp = "v" + std::to_string(vgpr_tmp++);
            result.push_back("v_mov_b32_e32 " + tmp + ", " + operands[i]);
            operands[i] = tmp;
          }
        }
      }
      std::string fixed = mnemonic + " " + operands[0];
      for (size_t i = 1; i < operands.size(); ++i) fixed += ", " + operands[i];
      if (!modifiers.empty()) fixed += " " + modifiers;
      result.push_back(fixed);
      return result;
    }
  }

  // ─── v_maxmin_f32/v_minmax_f32 → decompose to v_max + v_min pair ───
  // These 3-input VOP3 instructions don't exist on GFX9. Decompose:
  //   v_maxmin_f32 vd, s0, s1, s2 → vd = min(max(s0, s1), s2)
  //   v_minmax_f32 vd, s0, s1, s2 → vd = max(min(s0, s1), s2)
  // Also handle the GFX12 _num_ variant (before mnemonic rename strips it).
  if (mnemonic == "v_maxmin_f32" || mnemonic == "v_minmax_f32" ||
      mnemonic == "v_maxmin_f16" || mnemonic == "v_minmax_f16" ||
      mnemonic == "v_maxmin_num_f32" || mnemonic == "v_minmax_num_f32" ||
      mnemonic == "v_maxmin_num_f16" || mnemonic == "v_minmax_num_f16") {
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
      bool is_maxmin = (mnemonic.find("v_maxmin") == 0);
      std::string suffix = (mnemonic.find("_f16") != std::string::npos) ? "_f16" : "_f32";
      // v_maxmin_f32 vd, a, b, c → vd = min(max(a, b), c)
      // v_minmax_f32 vd, a, b, c → vd = max(min(a, b), c)
      std::string first_op = is_maxmin ? "v_max" : "v_min";
      std::string second_op = is_maxmin ? "v_min" : "v_max";
      // If vdst == src2, the first instruction clobbers src2 before the
      // second instruction can read it. Save src2 to a temp VGPR first.
      std::string src2 = operands[3];
      if (operands[0] == operands[3]) {
        std::string tmp = "v" + std::to_string(scale_temp_vgpr);
        result.push_back("v_mov_b32_e32 " + tmp + ", " + operands[3]);
        src2 = tmp;
      }
      result.push_back(first_op + suffix + " " + operands[0] + ", " +
                        operands[1] + ", " + operands[2]);
      result.push_back(second_op + suffix + " " + operands[0] + ", " +
                        operands[0] + ", " + src2);
      return result;
    }
  }

  // ─── VOP3B null sdst → vcc ───
  // GFX12 allows "null" as a write-discard SGPR destination in VOP3B.
  // GFX9 has no null register — use vcc (the natural sdst for these ops).
  // Applies to: v_div_scale_f32, v_div_scale_f64, v_add_co_ci_u32, v_sub_co_ci_u32
  if (mnemonic == "v_div_scale_f32" || mnemonic == "v_div_scale_f64" ||
      mnemonic == "v_add_co_ci_u32" || mnemonic == "v_sub_co_ci_u32") {
    // Find ", null," and replace with ", vcc,"
    size_t null_pos = line.find(", null,");
    if (null_pos != std::string::npos) {
      line.replace(null_pos, 7, ", vcc,");
    }
    // Also handle "null" at end of line (no trailing comma)
    null_pos = line.rfind(", null");
    if (null_pos != std::string::npos && null_pos + 6 == line.size()) {
      line.replace(null_pos, 6, ", vcc");
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
  //   f32_16x16x64_fp8_* → 2× f32_16x16x32_fp8 (K=64→2×K=32, dst:8→4, src:8→4)
  //   f32_16x16x64_bf8_* → 2× f32_16x16x32_bf8 (K=64→2×K=32, dst:8→4, src:8→4)
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
      // FP8/BF8 WMMA: K=64 on GFX1250, decomposed to 2× K=32 MFMA on GFX942
      {"v_wmma_f32_16x16x64_fp8_fp8", "v_mfma_f32_16x16x32_fp8_fp8", 8, 8, 4, 4},
      {"v_wmma_f32_16x16x64_fp8_bf8", "v_mfma_f32_16x16x32_fp8_bf8", 8, 8, 4, 4},
      {"v_wmma_f32_16x16x64_bf8_fp8", "v_mfma_f32_16x16x32_bf8_fp8", 8, 8, 4, 4},
      {"v_wmma_f32_16x16x64_bf8_bf8", "v_mfma_f32_16x16x32_bf8_bf8", 8, 8, 4, 4},
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
    // FP8/BF8: gfx942 has v_mfma_f32_16x16x32_fp8_* natively (K=32), but
    // WMMA K=64 needs 2× MFMA K=32. Same decomposition pattern.
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
      // FP8/BF8: WMMA K=64 mapped to MFMA K=32, needs 2× decomposition.
      // GFX942 has these K=32 variants natively — use them directly.
      if (mfma_mnem.find("_fp8") != std::string::npos ||
          mfma_mnem.find("_bf8") != std::string::npos) {
        need_decompose = true;
        actual_mfma = mfma_mnem;  // GFX942 K=32 fp8/bf8 MFMAs exist natively
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

  // ─── Other WMMA/SWMMAC → trap with diagnostic ───
  // v_swmmac (sparse WMMA) has no GFX942 equivalent — the hardware lacks sparse
  // matrix support. Unmatched v_wmma shapes also land here. Use s_trap to make
  // failures visible rather than producing silently wrong results.
  if (mnemonic.find("v_wmma_") == 0 || mnemonic.find("v_swmmac_") == 0) {
    result.push_back("s_trap 2 ; UNSUPPORTED: " + mnemonic + " (no GFX942 equivalent)");
    return result;
  }

  // ─── v_permlane16_b32 / v_permlanex16_b32 → ds_bpermute emulation ───
  // GFX10+ permlane instructions permute lanes within/across 16-lane groups.
  // Emulated on GFX9 using ds_bpermute_b32 to compute arbitrary lane mappings.
  //
  // v_permlane16_b32 vDst, vSrc, sA, sB:
  //   Group 0 (lanes 0-15): lane i reads from lane (i XOR (sA & 0xF))
  //   Group 1 (lanes 16-31): lane i reads from lane 16 + ((i-16) XOR (sB & 0xF))
  //
  // v_permlanex16_b32 vDst, vSrc, sA, sB: (cross-group)
  //   Group 0 (lanes 0-15): lane i reads from lane 16 + (i XOR (sA & 0xF))
  //   Group 1 (lanes 16-31): lane i reads from lane (i-16) XOR (sB & 0xF)
  if (mnemonic == "v_permlane16_b32" || mnemonic == "v_permlanex16_b32") {
    bool cross = (mnemonic == "v_permlanex16_b32");

    // Parse operands: vDst, vSrc, sA, sB [, op_sel:...]
    std::string ops = line.substr(line.find(mnemonic) + mnemonic.size());
    size_t s = ops.find_first_not_of(" \t");
    if (s != std::string::npos) ops = ops.substr(s);

    // Strip trailing modifiers: op_sel, fi, bound_ctrl
    for (const char* mod : {"op_sel:", "fi:", "bound_ctrl:"}) {
      size_t mpos = ops.find(mod);
      if (mpos != std::string::npos) {
        // Find end of modifier (next space or end)
        size_t mend = ops.find(' ', mpos);
        if (mend == std::string::npos) mend = ops.size();
        ops.erase(mpos, mend - mpos);
      }
    }

    // Tokenize comma-separated operands
    std::vector<std::string> operands;
    {
      std::istringstream iss(ops);
      std::string tok;
      while (std::getline(iss, tok, ',')) {
        size_t ts = tok.find_first_not_of(" \t");
        size_t te = tok.find_last_not_of(" \t");
        if (ts != std::string::npos)
          operands.push_back(tok.substr(ts, te - ts + 1));
      }
    }

    if (operands.size() >= 4) {
      std::string vDst = operands[0];
      std::string vSrc = operands[1];
      std::string sA   = operands[2];
      std::string sB   = operands[3];

      // SGPR pair for v_cmp mask (must be even-aligned)
      std::string stemp_lo = "s" + std::to_string(cmpx_temp_sgpr);
      std::string stemp_pair = "s[" + std::to_string(cmpx_temp_sgpr) + ":" +
                                std::to_string(cmpx_temp_sgpr + 1) + "]";

      result.push_back("; BEGIN " + mnemonic + " emulation via ds_bpermute");

      // Load selectors into VGPRs (v_cndmask needs VGPR sources)
      result.push_back("v_mov_b32_e32 " + vt2 + ", " + sA);
      result.push_back("v_mov_b32_e32 " + vt3 + ", " + sB);

      // Compute absolute lane ID (exec-independent: mask=-1 counts all bits)
      result.push_back("v_mbcnt_lo_u32_b32 " + vt0 + ", -1, 0");

      // Group (0 or 1) = lane >> 4
      result.push_back("v_lshrrev_b32_e32 " + vt1 + ", 4, " + vt0);

      // Select sA (group 0) or sB (group 1)
      result.push_back("v_cmp_ne_u32_e64 " + stemp_pair + ", 0, " + vt1);
      result.push_back("v_cndmask_b32_e64 " + vt2 + ", " + vt2 + ", " + vt3 + ", " + stemp_pair);

      // selector & 0xF
      result.push_back("v_and_b32_e32 " + vt2 + ", 0xF, " + vt2);

      // lane_in_group = lane & 0xF
      result.push_back("v_and_b32_e32 " + vt1 + ", 0xF, " + vt0);

      // src_lane_in_group = lane_in_group XOR selector
      result.push_back("v_xor_b32 " + vt1 + ", " + vt1 + ", " + vt2);

      // Reconstruct full source lane
      if (cross) {
        // Cross-group: source is in the OTHER group
        // other_group_base = (lane XOR 0x10) & 0x10
        result.push_back("v_xor_b32 " + vt2 + ", 0x10, " + vt0);
        result.push_back("v_and_b32_e32 " + vt2 + ", 0x10, " + vt2);
      } else {
        // Same group: group_base = lane & 0x10
        result.push_back("v_and_b32_e32 " + vt2 + ", 0x10, " + vt0);
      }
      result.push_back("v_or_b32 " + vt1 + ", " + vt1 + ", " + vt2);

      // ds_bpermute byte address = source_lane * 4
      result.push_back("v_lshlrev_b32_e32 " + vt1 + ", 2, " + vt1);

      // Execute cross-lane permutation
      result.push_back("ds_bpermute_b32 " + vDst + ", " + vt1 + ", " + vSrc);
      result.push_back("s_waitcnt lgkmcnt(0)");

      result.push_back("; END " + mnemonic + " emulation");
      return result;
    }
    // Fallback if operand parsing fails
    result.push_back("s_nop 0 ; PARSE ERROR: " + mnemonic);
    return result;
  }

  // ─── scale_offset: just strip the modifier, no byte-offset conversion ───
  // For debugging: skip address scaling to isolate the workgroup ID issue.
  // scale_offset emulation: compute byte offset in v3, substitute for vaddr
  if (line.find("scale_offset") != std::string::npos) {
    int shift = 0;
    if (mnemonic.find("_b32") != std::string::npos ||
        mnemonic.find("_u32") != std::string::npos ||
        mnemonic.find("_i32") != std::string::npos ||
        mnemonic.find("_f32") != std::string::npos ||
        mnemonic.find("_dword") != std::string::npos) shift = 2;
    else if (mnemonic.find("_b64") != std::string::npos ||
             mnemonic.find("_u64") != std::string::npos ||
             mnemonic.find("_i64") != std::string::npos ||
             mnemonic.find("_f64") != std::string::npos ||
             mnemonic.find("_dwordx2") != std::string::npos) shift = 3;
    else if (mnemonic.find("_b128") != std::string::npos ||
             mnemonic.find("_dwordx4") != std::string::npos) shift = 4;
    else if (mnemonic.find("_b16") != std::string::npos ||
             mnemonic.find("_u16") != std::string::npos ||
             mnemonic.find("_i16") != std::string::npos ||
             mnemonic.find("_f16") != std::string::npos) shift = 1;

    if (shift > 0) {
      // Determine vaddr position in operands.
      // Layout differs by instruction type:
      //   Load:              vdst, vaddr, saddr
      //   Store:             vaddr, vdata, saddr
      //   Atomic (no rtn):   vaddr, vdata, saddr
      //   Atomic (with rtn): vdst, vaddr, vdata, saddr
      // "vaddr is first operand" for stores and atomics-without-return.
      size_t mnem_end = line.find(mnemonic) + mnemonic.size();
      std::string ops = line.substr(mnem_end);
      bool is_store = mnemonic.find("store") != std::string::npos;
      bool is_atomic = mnemonic.find("atomic") != std::string::npos;
      // Atomics without TH_ATOMIC_RETURN have vaddr as first operand (like stores).
      // If TH_ATOMIC_RETURN was present (translated to "sc0"), vaddr is second operand.
      bool atomic_no_return = is_atomic && line.find("sc0") == std::string::npos &&
                              line.find("TH_ATOMIC_RETURN") == std::string::npos;
      bool vaddr_is_first = is_store || atomic_no_return;

      std::string vaddr;
      size_t vaddr_abs_pos = std::string::npos;  // absolute position in `line`
      if (vaddr_is_first) {
        size_t s = ops.find_first_not_of(" \t");
        size_t e = ops.find_first_of(" \t,", s);
        if (s != std::string::npos) {
          vaddr = ops.substr(s, e != std::string::npos ? e - s : std::string::npos);
          vaddr_abs_pos = mnem_end + s;
        }
      } else {
        // vaddr is second operand (after first comma)
        size_t comma1 = ops.find(',');
        if (comma1 != std::string::npos) {
          size_t s = ops.find_first_not_of(" \t", comma1 + 1);
          size_t e = ops.find_first_of(" \t,", s);
          if (s != std::string::npos) {
            vaddr = ops.substr(s, e != std::string::npos ? e - s : std::string::npos);
            vaddr_abs_pos = mnem_end + s;
          }
        }
      }

      if (!vaddr.empty() && vaddr_abs_pos != std::string::npos) {
        // Use a temp VGPR for the byte offset, above the kernel's own VGPRs.
        // scale_temp_vgpr is passed from the translation loop and set to
        // num_vgprs12+2 (above save registers).  vaddr stays as element index
        // because subsequent scale_offset instructions reuse it.
        const std::string kScaleTemp = "v" + std::to_string(scale_temp_vgpr);
        result.push_back("v_lshlrev_b32_e32 " + kScaleTemp + ", " +
                          std::to_string(shift) + ", " + vaddr);
        // Replace vaddr at its EXACT position in the instruction (avoids
        // confusion when dest register has the same name as vaddr).
        std::string modified = line;
        modified.replace(vaddr_abs_pos, vaddr.size(), kScaleTemp);
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

  // ─── v_add_u64 (post _nc_ strip) → v_add_co_u32 + v_addc_co_u32 ───
  // Catches v_add_nc_u64 after _nc_ stripping converts it to v_add_u64.
  if (mnemonic == "v_add_u64" || mnemonic == "v_add_u64_e64") {
    std::string ops = line.substr(line.find(mnemonic) + mnemonic.size());
    size_t op_start = ops.find_first_not_of(" \t");
    if (op_start != std::string::npos) ops = ops.substr(op_start);
    std::vector<std::string> operands;
    std::istringstream u64_oss(ops);
    std::string u64_tok;
    while (std::getline(u64_oss, u64_tok, ',')) {
      size_t s = u64_tok.find_first_not_of(" \t");
      size_t e = u64_tok.find_last_not_of(" \t");
      if (s != std::string::npos)
        operands.push_back(u64_tok.substr(s, e - s + 1));
    }
    if (operands.size() >= 3) {
      auto parse_pair = [](const std::string& r, std::string& lo, std::string& hi) {
        size_t lb = r.find('['), colon = r.find(':'), rb = r.find(']');
        if (lb != std::string::npos && colon != std::string::npos && rb != std::string::npos) {
          std::string prefix = r.substr(0, lb);
          lo = prefix + r.substr(lb + 1, colon - lb - 1);
          hi = prefix + r.substr(colon + 1, rb - colon - 1);
          return true;
        }
        return false;
      };
      std::string dst_lo, dst_hi, src0_lo, src0_hi, src1_lo, src1_hi;
      if (parse_pair(operands[0], dst_lo, dst_hi) &&
          parse_pair(operands[1], src0_lo, src0_hi) &&
          parse_pair(operands[2], src1_lo, src1_hi)) {
        bool s0_sgpr = !src0_lo.empty() && (src0_lo[0] == 's' || src0_lo.find("flat_") == 0);
        bool s1_sgpr = !src1_lo.empty() && (src1_lo[0] == 's' || src1_lo.find("flat_") == 0);
        if (s0_sgpr || s1_sgpr) {
          result.push_back("v_mov_b32_e32 " + vt0 + ", " + src1_lo);
          result.push_back("v_mov_b32_e32 " + vt1 + ", " + src1_hi);
          if (s0_sgpr && s1_sgpr) {
            result.push_back("v_mov_b32_e32 " + vt2 + ", " + src0_lo);
            result.push_back("v_mov_b32_e32 " + vt3 + ", " + src0_hi);
            result.push_back("v_add_co_u32_e64 " + dst_lo + ", vcc, " + vt2 + ", " + vt0);
            result.push_back("v_addc_co_u32_e64 " + dst_hi + ", vcc, " + vt3 + ", " + vt1 + ", vcc");
          } else {
            result.push_back("v_add_co_u32_e64 " + dst_lo + ", vcc, " + src0_lo + ", " + vt0);
            result.push_back("v_addc_co_u32_e64 " + dst_hi + ", vcc, " + src0_hi + ", " + vt1 + ", vcc");
          }
        } else {
          result.push_back("v_add_co_u32_e64 " + dst_lo + ", vcc, " + src0_lo + ", " + src1_lo);
          result.push_back("v_addc_co_u32_e64 " + dst_hi + ", vcc, " + src0_hi + ", " + src1_hi + ", vcc");
        }
        return result;
      }
    }
  }

  // ─── v_mov_b64 → two v_mov_b32 ───
  // GFX12 has v_mov_b64_e32 but GFX9 doesn't.
  if (mnemonic == "v_mov_b64" || mnemonic == "v_mov_b64_e32") {
    std::string ops = line.substr(line.find(mnemonic) + mnemonic.size());
    size_t op_start = ops.find_first_not_of(" \t");
    if (op_start != std::string::npos) ops = ops.substr(op_start);
    std::vector<std::string> operands;
    std::istringstream b64_oss(ops);
    std::string b64_tok;
    while (std::getline(b64_oss, b64_tok, ',')) {
      size_t s = b64_tok.find_first_not_of(" \t");
      size_t e = b64_tok.find_last_not_of(" \t");
      if (s != std::string::npos)
        operands.push_back(b64_tok.substr(s, e - s + 1));
    }
    if (operands.size() >= 2) {
      auto parse_pair = [](const std::string& r, std::string& lo, std::string& hi) {
        size_t lb = r.find('['), colon = r.find(':'), rb = r.find(']');
        if (lb != std::string::npos && colon != std::string::npos && rb != std::string::npos) {
          std::string prefix = r.substr(0, lb);
          lo = prefix + r.substr(lb + 1, colon - lb - 1);
          hi = prefix + r.substr(colon + 1, rb - colon - 1);
          return true;
        }
        return false;
      };
      std::string dst_lo, dst_hi, src_lo, src_hi;
      bool dst_ok = parse_pair(operands[0], dst_lo, dst_hi);
      bool src_ok = parse_pair(operands[1], src_lo, src_hi);
      if (dst_ok && src_ok) {
        result.push_back("v_mov_b32_e32 " + dst_lo + ", " + src_lo);
        result.push_back("v_mov_b32_e32 " + dst_hi + ", " + src_hi);
        return result;
      } else if (dst_ok && !src_ok) {
        // Source is immediate (e.g., 0)
        result.push_back("v_mov_b32_e32 " + dst_lo + ", " + operands[1]);
        result.push_back("v_mov_b32_e32 " + dst_hi + ", 0");
        return result;
      }
    }
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
  // If src0 is VGPR and src1 is a large literal: load src1 into temp VGPR, use VOPC.
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
          bool is_f64_cmp = mnemonic.find("_f64") != std::string::npos;
          bool is_class_cmp = mnemonic.find("_class_") != std::string::npos;
          if (src1_lit && !src0_lit) {
            // VOPC src1 must be VGPR — load literal into VGPR first.
            // For f64 data sources, use VGPR pair. But v_cmp_class src1 is
            // always 32-bit class mask even for _f64 variant.
            bool src1_needs_pair = is_f64_cmp && !is_class_cmp;
            if (src1_needs_pair) {
              result.push_back("v_mov_b32_e32 " + vt0 + ", " + srcs[1]);
              result.push_back("v_mov_b32_e32 " + vt1 + ", 0");
              std::string pair = "v[" + std::to_string(scale_temp_vgpr) + ":" +
                                 std::to_string(scale_temp_vgpr + 1) + "]";
              result.push_back(base_mnem + " " + srcs[0] + ", " + pair);
            } else {
              result.push_back("v_mov_b32_e32 " + vt0 + ", " + srcs[1]);
              result.push_back(base_mnem + " " + srcs[0] + ", " + vt0);
            }
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
    // Clear exec_hi: on GFX9 wave64, v_cmpx compares ALL 64 lanes including
    // 32-63 which have uninitialized VGPRs (wave32 workgroup in wave64 HW).
    // Ghost lanes could pass the comparison and execute stores to garbage addrs.
    result.push_back("s_mov_b32 exec_hi, 0");
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
  // so load them into temp VGPR first (e.g., v_cmp_class_f32_e64 sN, s8, 0x260).
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
          // Handle large literals in ANY source operand (GFX9 VOP3 no literal support).
          // Check the v_cmp line that was already emitted in result, and if any
          // source operand is a large literal, prepend a v_mov to load it to temp VGPR.
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
            // Find the v_cmp line in result
            for (size_t ri = 0; ri < result.size(); ri++) {
              if (result[ri].find("v_cmp_") != std::string::npos) {
                // Parse all comma-separated tokens after the mnemonic
                size_t mstart = result[ri].find("v_cmp_");
                size_t mend2 = result[ri].find(' ', mstart);
                if (mend2 == std::string::npos) break;
                std::string cmp_ops = result[ri].substr(mend2);
                std::vector<std::string> cmp_operands;
                std::istringstream css(cmp_ops);
                std::string ctok;
                while (std::getline(css, ctok, ',')) {
                  size_t cs = ctok.find_first_not_of(" \t");
                  size_t ce = ctok.find_last_not_of(" \t");
                  if (cs != std::string::npos)
                    cmp_operands.push_back(ctok.substr(cs, ce - cs + 1));
                }
                // Check src operands (skip dest at index 0 = s[N:N+1])
                for (size_t ci = 1; ci < cmp_operands.size(); ci++) {
                  if (is_ll(cmp_operands[ci])) {
                    std::string lit_val = cmp_operands[ci];
                    // For f64 instructions, some source operands need 64-bit VGPR pairs.
                    // v_cmp_*_f64 (non-class): both src0 and src1 are f64 → need pair.
                    // v_cmp_class_*_f64: src0 is f64 (ci==1), src1 is i32 class mask (ci==2).
                    bool is_f64_mnem = mnemonic.find("_f64") != std::string::npos;
                    bool is_class = mnemonic.find("_class_") != std::string::npos;
                    bool needs_pair = is_f64_mnem && !(is_class && ci >= 2);
                    if (needs_pair) {
                      // Use vt2:vt3 for f64 pair (vt0 is used by save_reg).
                      // vt2 is even-aligned (scale_temp_vgpr+2) so the pair
                      // satisfies GFX9's 64-bit alignment requirement.
                      std::string f64_pair = "v[" + std::to_string(scale_temp_vgpr + 2) + ":" +
                                             std::to_string(scale_temp_vgpr + 3) + "]";
                      cmp_operands[ci] = f64_pair;
                      result.insert(result.begin() + ri, "v_mov_b32_e32 " + vt3 + ", 0");
                      result.insert(result.begin() + ri, "v_mov_b32_e32 " + vt2 + ", " + lit_val);
                      // Rebuild the v_cmp line (now at ri + 2)
                      std::string rebuilt = result[ri + 2].substr(0, mend2);
                      for (size_t cj = 0; cj < cmp_operands.size(); cj++) {
                        rebuilt += (cj == 0 ? " " : ", ") + cmp_operands[cj];
                      }
                      result[ri + 2] = rebuilt;
                    } else {
                      cmp_operands[ci] = vt1;
                      result.insert(result.begin() + ri, "v_mov_b32_e32 " + vt1 + ", " + lit_val);
                      // Rebuild the v_cmp line (now at ri + 1)
                      std::string rebuilt = result[ri + 1].substr(0, mend2);
                      for (size_t cj = 0; cj < cmp_operands.size(); cj++) {
                        rebuilt += (cj == 0 ? " " : ", ") + cmp_operands[cj];
                      }
                      result[ri + 1] = rebuilt;
                    }
                    break;  // one literal fix per instruction is enough
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
    // Check for large literal in ANY source operand (GFX9 VOP3 doesn't allow literals)
    auto is_large_literal = [](const std::string& s) -> bool {
      if (s.empty()) return false;
      if (s.size() > 2 && s[0] == '0' && (s[1] == 'x' || s[1] == 'X')) return true;
      try {
        if (std::isdigit((unsigned char)s[0]) && std::stol(s) > 64) return true;
        if (s[0] == '-' && s.size() > 1 && std::stol(s) < -16) return true;
      } catch (...) {}
      return false;
    };
    // Parse all comma-separated operands and check sources (skip dest at index 0)
    {
      size_t mnem_end2 = line.find(mnemonic) + mnemonic.size();
      std::string all_ops = line.substr(mnem_end2);
      std::vector<std::string> all_operands;
      std::istringstream aoss(all_ops);
      std::string atok;
      while (std::getline(aoss, atok, ',')) {
        size_t as = atok.find_first_not_of(" \t");
        size_t ae = atok.find_last_not_of(" \t");
        if (as != std::string::npos) all_operands.push_back(atok.substr(as, ae - as + 1));
      }
      // Check sources (indices 1+) for large literals
      bool is_f64_mnem = mnemonic.find("_f64") != std::string::npos;
      bool is_class_mnem = mnemonic.find("_class_") != std::string::npos;
      for (size_t ai = 1; ai < all_operands.size(); ai++) {
        if (is_large_literal(all_operands[ai])) {
          // f64 data sources need 64-bit VGPR pair; class mask (src1) is 32-bit
          bool needs_pair = is_f64_mnem && !(is_class_mnem && ai >= 2);
          if (needs_pair) {
            result.push_back("v_mov_b32_e32 " + vt0 + ", " + all_operands[ai]);
            result.push_back("v_mov_b32_e32 " + vt1 + ", 0");
            std::string f64_pair = "v[" + std::to_string(scale_temp_vgpr) + ":" +
                                   std::to_string(scale_temp_vgpr + 1) + "]";
            all_operands[ai] = f64_pair;
          } else {
            result.push_back("v_mov_b32_e32 " + vt0 + ", " + all_operands[ai]);
            all_operands[ai] = vt0;
          }
          // Rebuild line
          std::string rebuilt = mnemonic;
          for (size_t aj = 0; aj < all_operands.size(); aj++) {
            rebuilt += (aj == 0 ? " " : ", ") + all_operands[aj];
          }
          line = rebuilt;
          break;  // one literal per instruction is enough
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
  auto exec_result = WidenExecOperation(line, compact_mode, cmpx_temp_sgpr);
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
    // GFX9  RSRC1 layout: bits [5:0] = VGPR field (gran 4), bits [9:6] = SGPR field (gran 8)
    //                      bits [11:10] = PRIORITY
    //
    // On GFX942 (CDNA3), RSRC3 ACCUM_OFFSET controls arch VGPR allocation:
    //   arch_vgprs = (ACCUM_OFFSET+1)*4.
    //   total_vgprs = max((RSRC1_VGPR+1)*4, (ACCUM+1)*4).
    //   Setting RSRC1 VGPR = ACCUM means all VGPRs are arch (no accum VGPRs).
    //
    // Convert VGPR count: GFX12 granularity 12 (wave32) → GFX9 granularity 4.
    // Add 8 extra VGPRs: 2 save registers (sv_x, sv_y) + 4 temp VGPRs (vt0-vt3)
    // + padding to align to 4-VGPR granularity.
    uint32_t vgpr_field12 = rsrc1 & 0x3Fu;
    uint32_t sgpr_field12_kd = (rsrc1 >> 6) & 0x3Fu;  // save before clear
    // GFX12 (RDNA4) wave32 VGPR granularity is 12 (GFX10=8, GFX11+=12).
    // RSRC1 VGPR field encodes ceil(VGPRs/gran)-1.
    uint32_t num_vgprs = (vgpr_field12 + 1u) * 12u;
    if (num_vgprs < 8u) num_vgprs = 8u;
    num_vgprs += 8u;  // room for sv_x, sv_y + 4 temp VGPRs (vt0-vt3) + even-align pad
    // ACCUM_OFFSET: on GFX942, arch VGPRs = (ACCUM_OFFSET+1)*4.
    // Set ACCUM_OFFSET so ALL allocated VGPRs (kernel's + save regs + temps) are arch VGPRs.
    // Round up to 4-VGPR granularity.
    uint32_t gfx9_vgpr = ((num_vgprs + 3u) / 4u) - 1u;  // ACCUM_OFFSET = all-arch
    if (gfx9_vgpr > 62u) gfx9_vgpr = 62u;
    // RSRC1 VGPR field = ACCUM_OFFSET so total = arch (no extra accum VGPRs).
    uint32_t rsrc1_vgpr_field = gfx9_vgpr;
    if (rsrc1_vgpr_field > 63u) rsrc1_vgpr_field = 63u;
    // Clear bits [11:0] and set GFX9 VGPR[5:0] and SGPR[9:6]
    rsrc1 &= ~0xFFFu;
    rsrc1 |= (rsrc1_vgpr_field & 0x3Fu);  // VGPR in bits [5:0]
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
      // Ceiling division: round UP to ensure enough SGPRs allocated.
      // Transpiler adds cmpx_temp + kernarg_save above kernel's count.
      uint32_t gfx9_sgpr = ((num_sgprs + 7u) / 8u) - 1u;
      if (gfx9_sgpr > 12u) gfx9_sgpr = 12u;
      rsrc1 |= (gfx9_sgpr << 6u);  // SGPR in bits [9:6]
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

  // 4. Patch MSGPACK .vgpr_count and .sgpr_count to match the transpiled KD.
  // The kernel descriptors were already patched for GFX9 (step 6 in TranspileCodeObject).
  // Re-read RSRC1 from the first valid kernel descriptor to derive actual counts.
  // This ensures the runtime's dispatch setup matches the actual register usage.
  {
    // Parse ELF section headers to find kernel descriptors in .rodata or .text.
    uint64_t e_shoff = 0;
    uint16_t e_shentsize = 0, e_shnum = 0, e_shstrndx = 0;
    std::memcpy(&e_shoff,    elf + 40, 8);
    std::memcpy(&e_shentsize, elf + 58, 2);
    std::memcpy(&e_shnum,    elf + 60, 2);
    std::memcpy(&e_shstrndx, elf + 62, 2);

    // Locate section name string table (.shstrtab).
    const char* shstrtab = nullptr;
    uint64_t shstrtab_size = 0;
    if (e_shstrndx < e_shnum && e_shoff + (uint64_t)(e_shstrndx + 1) * e_shentsize <= elf_size) {
      const uint8_t* sh = elf + e_shoff + e_shstrndx * e_shentsize;
      uint64_t off, sz;
      std::memcpy(&off, sh + 24, 8);
      std::memcpy(&sz,  sh + 32, 8);
      if (off + sz <= elf_size) {
        shstrtab = reinterpret_cast<const char*>(elf + off);
        shstrtab_size = sz;
      }
    }

    // Walk sections to find the first valid kernel descriptor.
    // Prefer .rodata (where compiler-generated KDs live) over .text to avoid
    // false-positives from code bytes when scanning .text at 256-byte steps.
    uint32_t kd_vgpr_count = 0, kd_sgpr_count = 0;
    bool found_kd = false;

    // Build a list of candidate sections ordered: .rodata first, .text second.
    struct SecCandidate { uint64_t off; uint64_t size; bool is_text; };
    std::vector<SecCandidate> candidates;
    for (uint16_t si = 0; si < e_shnum; ++si) {
      if (e_shoff + (uint64_t)(si + 1) * e_shentsize > elf_size) break;
      const uint8_t* sh = elf + e_shoff + si * e_shentsize;
      uint32_t name_idx;
      uint64_t sec_off, sec_size;
      std::memcpy(&name_idx, sh,      4);
      std::memcpy(&sec_off,  sh + 24, 8);
      std::memcpy(&sec_size, sh + 32, 8);
      if (sec_off + sec_size > elf_size) continue;
      const char* sec_name = (shstrtab && name_idx < shstrtab_size)
                             ? shstrtab + name_idx : "";
      if (std::strcmp(sec_name, ".rodata") == 0)
        candidates.insert(candidates.begin(), {sec_off, sec_size, false});
      else if (std::strcmp(sec_name, ".text") == 0)
        candidates.push_back({sec_off, sec_size, true});
    }

    for (const auto& cand : candidates) {
      if (found_kd) break;
      // Kernel descriptors are at 64-byte aligned offsets in .rodata, or 256-byte in .text.
      uint64_t step = cand.is_text ? 256u : 64u;
      for (uint64_t off = 0; off + 64 <= cand.size; off += step) {
        uint64_t entry;
        std::memcpy(&entry, elf + cand.off + off + 16, 8);
        if (entry == 0 || entry > 1000000u) continue;  // not a valid kernel descriptor

        // Read the already-patched GFX9 RSRC1 and RSRC3 (written by PatchKernelDescriptorsForWave64).
        // GFX9 RSRC1: bits [5:0] = VGPR field (gran=4), bits [9:6] = SGPR field (gran=8)
        // On GFX942 (CDNA3): VGPR allocation controlled by RSRC3 ACCUM_OFFSET, not RSRC1.
        // arch_vgprs = (ACCUM_OFFSET + 1) * 4
        uint32_t rsrc1;
        std::memcpy(&rsrc1, elf + cand.off + off + 48, 4);
        uint32_t rsrc3;
        std::memcpy(&rsrc3, elf + cand.off + off + 44, 4);
        uint32_t sgpr_field = (rsrc1 >> 6) & 0xFu;
        uint32_t accum_offset = rsrc3 & 0x3Fu;
        kd_vgpr_count = (accum_offset + 1u) * 4u;  // GFX942: arch VGPRs from ACCUM_OFFSET
        kd_sgpr_count = (sgpr_field + 1u) * 8u;    // GFX9 SGPR granularity = 8
        std::cerr << "hotswap: transpile: KD-derived vgpr_count=" << kd_vgpr_count
                  << " sgpr_count=" << kd_sgpr_count << " (ACCUM_OFFSET="
                  << accum_offset << ", RSRC1=0x"
                  << std::hex << rsrc1 << std::dec << ")\n";
        found_kd = true;
        break;
      }
    }

    if (found_kd) {
      // Patch a MSGPACK integer field in-place.
      // MSGPACK encoding for the value byte(s) immediately following the key string:
      //   0x00-0x7F : positive fixint (single byte, value 0-127)
      //   0xCC, val : uint8  (two bytes, value 0-255)
      //   0xCD, hi, lo : uint16 big-endian (three bytes, value 0-65535)
      // We patch in-place; if the new value needs more bytes than the original encoding,
      // we log a warning and skip (cannot safely expand the binary blob).
      auto patch_msgpack_key = [&](const char* key, size_t key_len, uint32_t new_val,
                                   const char* key_name) {
        for (size_t i = 0; i + key_len + 1 <= elf_size; ++i) {
          if (std::memcmp(elf + i, key, key_len) != 0) continue;
          uint8_t* vp = elf + i + key_len;
          uint8_t enc = *vp;
          if (enc <= 0x7Fu) {
            // Original is positive fixint.
            if (new_val <= 0x7Fu) {
              std::cerr << "hotswap: transpile: patched MSGPACK " << key_name
                        << " " << (uint32_t)enc << " → " << new_val << "\n";
              *vp = static_cast<uint8_t>(new_val);
            } else {
              std::cerr << "hotswap: transpile: WARNING: cannot patch MSGPACK " << key_name
                        << " in-place (new_val=" << new_val << " > 127, orig fixint "
                        << (uint32_t)enc << ")\n";
            }
          } else if (enc == 0xCCu && i + key_len + 2 <= elf_size) {
            // uint8 format: [0xCC, val].
            if (new_val <= 0xFFu) {
              std::cerr << "hotswap: transpile: patched MSGPACK " << key_name
                        << " " << (uint32_t)vp[1] << " → " << new_val << "\n";
              vp[1] = static_cast<uint8_t>(new_val);
            } else {
              std::cerr << "hotswap: transpile: WARNING: cannot patch MSGPACK " << key_name
                        << " in-place (new_val=" << new_val << " > 255, orig uint8)\n";
            }
          } else if (enc == 0xCDu && i + key_len + 3 <= elf_size) {
            // uint16 big-endian format: [0xCD, hi, lo].
            if (new_val <= 0xFFFFu) {
              std::cerr << "hotswap: transpile: patched MSGPACK " << key_name
                        << " " << (uint32_t)((vp[1] << 8) | vp[2]) << " → " << new_val << "\n";
              vp[1] = static_cast<uint8_t>((new_val >> 8) & 0xFF);
              vp[2] = static_cast<uint8_t>(new_val & 0xFF);
            }
          }
          // Patch all occurrences (one per kernel in multi-kernel ELFs).
          // Continue scanning so all kernels' metadata gets updated.
        }
      };

      patch_msgpack_key(".vgpr_count", 11, kd_vgpr_count, ".vgpr_count");
      patch_msgpack_key(".sgpr_count", 11, kd_sgpr_count, ".sgpr_count");
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

  // Saved across kernel loop for post-processing fix blocks.
  uint32_t kernarg_save_lo = 0, kernarg_save_hi = 0;
  std::string ka_lo, ka_hi;

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
    // add 8 to num_vgprs when computing gfx9_vgpr/ACCUM_OFFSET, ensuring these
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
    // The KD allocates .num_vgpr + 8 arch VGPRs (2 save + 4 temp + even-align pad).
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

      // Stop at s_endpgm + s_code_end padding: everything after is dead
      // device library code that may use unsupported GFX12 features
      // (flat_scratch, v_add_nc_u64, etc.). The kernel's actual code ends
      // at s_endpgm.
      if (!asm_text.empty() && asm_text.find("s_endpgm") == 0) {
        // Check if next bytes are s_code_end (0xBF9F0000) or padding
        if (pos + 4 <= code_end) {
          uint32_t next_word;
          std::memcpy(&next_word, text + pos, 4);
          if (next_word == 0xBF9F0000u) {
            // s_code_end follows — this is the end of kernel code.
            // Skip the rest of .text (device library dead code).
            break;
          }
        }
      }
    }

    std::cerr << "hotswap: transpile: kernel " << ki << ": disassembled "
              << source_lines.size() << " instructions\n";

    // Debug: dump original GFX12 disassembly
    if (std::getenv("HSA_HOTSWAP_DUMP")) {
      std::cerr << "hotswap: transpile: === ORIGINAL GFX12 DISASSEMBLY ===\n";
      for (size_t i = 0; i < source_instrs.size(); ++i) {
        std::cerr << "  [" << i << "] pc=0x" << std::hex << source_instrs[i].pc_offset
                  << std::dec << " " << source_instrs[i].text << "\n";
      }
      std::cerr << "hotswap: transpile: === END ORIGINAL ===\n";
    }

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
    // integer division using v8+).  PatchKernelDescriptorsForWave64 adds 8 to
    // num_vgprs when computing ACCUM_OFFSET so these registers are arch VGPRs.
    translated_asm += "v_mov_b32_e32 " + sv_x + ", s2 ; save workgroup_id_x\n";
    translated_asm += "v_mov_b32_e32 " + sv_y + ", s3 ; save workgroup_id_y\n";
    // (cache invalidation at kernel start removed — not the root cause)
    // Save kernarg pointer s[0:1] to dedicated SGPRs for later use.
    // Only needed for kernels that access hidden args (s_load from s[8:9]+0xc).
    // Place at cmpx_temp_sgpr+2 and +3 (above kernel's own SGPRs).
    kernarg_save_lo = cmpx_temp_sgpr + 2;
    kernarg_save_hi = cmpx_temp_sgpr + 3;
    ka_lo = "s" + std::to_string(kernarg_save_lo);
    ka_hi = "s" + std::to_string(kernarg_save_hi);
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

      // scale_temp_vgpr: use num_vgprs12+2 (above save registers sv_x, sv_y),
      // rounded up to even alignment for f64 VGPR pair requirements on GFX9.
      // cmpx_temp_sgpr: use num_sgprs12 (above kernel's SGPR allocation)
      // compact_mode: skip redundant exec_hi clear after AND saveexec
      // (exec_hi is already 0). Helps save space without affecting correctness.
      // Note: even-dest v_cmp save/restore and v_cmpx VCC save are ALWAYS
      uint32_t temp_vgpr_base = save_vgpr_y + 1;
      if (temp_vgpr_base & 1u) temp_vgpr_base++;  // ensure even alignment for f64 pairs
      auto translated_lines = TranslateInstruction(line, src_cpu, tgt_cpu,
                                                    temp_vgpr_base, cmpx_temp_sgpr,
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
      if (early_exit_after > 0 && emitted_count >= early_exit_after && !early_exit_done) {
        // Emit all remaining branch target labels pointing to the exit, then s_endpgm.
        // This prevents assembly errors from unresolved forward references.
        for (size_t jj = ii + 1; jj < source_instrs.size(); jj++) {
          auto lbl = branch_labels.find(source_instrs[jj].pc_offset);
          if (lbl != branch_labels.end())
            translated_asm += lbl->second + ":\n";
        }
        translated_asm += ".L_early_exit:\n";
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
    // Wave32→64 VCC branch fix: s_cbranch_vccz/vccnz checks full 64-bit VCC.
    // In our wave32-in-wave64 model, vcc_hi can be stale from scalar b32 ops
    // that only write vcc_lo. Insert s_mov_b32 vcc_hi, 0 before each VCC branch
    // to ensure VCC = {0, vcc_lo}. This is safe because s_mov_b32 never clobbers SCC.
    {
      std::string tmp;
      std::istringstream vfix_iss(translated_asm);
      std::string vfix_line;
      std::string prev_line;
      while (std::getline(vfix_iss, vfix_line)) {
        if (vfix_line.find("s_cbranch_vccz") != std::string::npos ||
            vfix_line.find("s_cbranch_vccnz") != std::string::npos) {
          // Widen the PRECEDING s_and_b32/s_andn2_b32 vcc_lo to b64 vcc
          // by replacing it in the output. The b64 form writes both VCC_lo and VCC_hi.
          // Look back through prev_line (and s_nop) for the VCC-setting instruction.
          // Instead of replacing prev_line (complex), just insert s_mov_b32 vcc_hi, 0
          // before the branch. This ensures VCC = {0, VCC_lo}.
          tmp += "s_mov_b32 vcc_hi, 0\n";
        }
        tmp += vfix_line + "\n";
        prev_line = vfix_line;
      }
      translated_asm = tmp;
    }
    // Misplaced s_endpgm fix: when the compiler places s_endpgm in the middle
    // of .text (with reachable code after it via forward branches), the GFX9
    // assembler silently includes it but the hardware terminates the wavefront
    // at s_endpgm — making all subsequent instructions dead.  Fix: if s_endpgm
    // is not the last non-empty instruction, replace it with a branch to a new
    // .L_real_exit label and append the real s_endpgm at the end.
    {
      // Find the line containing just "s_endpgm" (no suffix).
      // We look for a line that is exactly "s_endpgm" after trimming.
      std::istringstream ep_iss(translated_asm);
      std::string ep_line;
      std::vector<std::string> ep_lines;
      while (std::getline(ep_iss, ep_line)) ep_lines.push_back(ep_line);

      // Find the index of the s_endpgm line.
      int endpgm_idx = -1;
      for (int i = 0; i < (int)ep_lines.size(); ++i) {
        std::string trimmed = ep_lines[i];
        size_t start = trimmed.find_first_not_of(" \t");
        if (start != std::string::npos) trimmed = trimmed.substr(start);
        if (trimmed == "s_endpgm") {
          endpgm_idx = i;
          break;
        }
      }

      if (endpgm_idx >= 0) {
        // Check if there are non-empty, non-label, non-comment lines after it.
        bool has_code_after = false;
        for (int i = endpgm_idx + 1; i < (int)ep_lines.size(); ++i) {
          std::string t = ep_lines[i];
          size_t s = t.find_first_not_of(" \t");
          if (s == std::string::npos) continue;
          t = t.substr(s);
          if (t.empty() || t[0] == ';') continue;
          if (t[0] == '.' && t.find(':') != std::string::npos) continue; // label
          // Found a real instruction after s_endpgm
          has_code_after = true;
          break;
        }

        if (has_code_after) {
          // Replace s_endpgm with a branch to the real exit.
          ep_lines[endpgm_idx] = "s_branch .L_real_exit ; hoisted s_endpgm to end";
          // Reconstruct translated_asm and append the real exit.
          std::string fixed;
          for (const auto& l : ep_lines) fixed += l + "\n";
          fixed += ".L_real_exit:\n";
          fixed += "s_endpgm\n";
          translated_asm = fixed;
          if (std::getenv("HSA_HOTSWAP_DUMP"))
            std::cerr << "hotswap: transpile: hoisted mid-kernel s_endpgm to end\n";
        }
      }
    }
    // Redundant remainder guard: add s8==0 check at .L_br28 entry.
    // The VCC-based guard (s_cbranch_vccz .L_br25) should prevent entry to .L_br28
    // when remainder=0, but it fails on GFX942. This adds a belt-and-suspenders
    // SCC-based check right at .L_br28 entry.
    // Only apply these fixes to kernels with .L_br28 (attn_forward-specific pattern)
    if (stats->total_instructions > 400 &&
        translated_asm.find(".L_br28:") != std::string::npos) {
      // Ensure kernarg pointer is saved for Fix 2/3 loads.
      // If the s[8:9]+0xc replacement already emitted a save, this is a no-op.
      if (translated_asm.find("; save kernarg ptr lo") == std::string::npos) {
        std::string ka_pair_str = "s[" + std::to_string(kernarg_save_lo) + ":"
                                  + std::to_string(kernarg_save_hi) + "]";
        size_t insert_pos = translated_asm.find("; save workgroup_id_y\n");
        if (insert_pos != std::string::npos) {
          insert_pos = translated_asm.find('\n', insert_pos) + 1;
          translated_asm.insert(insert_pos,
            "s_mov_b32 " + ka_lo + ", s0 ; save kernarg ptr lo\n"
            "s_mov_b32 " + ka_hi + ", s1 ; save kernarg ptr hi\n");
        }
      }
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
      // Fix: force ALL inner product to use the remainder path (.L_br26/.L_br29).
      // The unrolled path (.L_br27) produces zeros on GFX9.
      // Set s9=0 so the .L_br24 branch to .L_br26 is always taken.
      // Set s5=0 and s8=N (original) so the remainder processes all N elements.
      if (stats->total_instructions < 600) {
        auto replaceFirst = [](std::string& s, const std::string& from, const std::string& to) {
          size_t pos = s.find(from);
          if (pos != std::string::npos) s.replace(pos, from.size(), to);
        };
        // Before s_and s5: save N to s8 BEFORE masking, force s5=0 and s9=0
        // attn_forward uses: s_and_b32 s5, s5, 0x7ffffff8
        // attn_multihead uses: s_and_b32 s10, s5, 0x7ffffff8
        replaceFirst(translated_asm,
          "s_and_b32 s5, s5, 0x7ffffff8\n",
          "s_mov_b32 s8, s5 ; FIX: s8 = N (full remainder count)\n"
          "s_mov_b32 s5, 0  ; FIX: force s5=0 (skip unrolled loop)\n"
          "s_mov_b32 s9, 0  ; FIX: force s9=0 (go to remainder path)\n");
        // multihead variant: s10 holds the unroll count instead of s5
        replaceFirst(translated_asm,
          "s_and_b32 s10, s5, 0x7ffffff8\n",
          "s_mov_b32 s8, s5 ; FIX: s8 = N (full remainder count)\n"
          "s_mov_b32 s10, 0 ; FIX: force s10=0 (no unrolled iterations)\n"
          "s_mov_b32 s9, 0  ; FIX: force s9=0 (go to remainder path)\n");
      }
      // Detect multihead by checking for its prologue pattern: s_and_b32 s23, s24, 0xffff.
      // attn_forward uses s10 for blockSize, multihead uses s23.
      bool has_s23_blocksize = translated_asm.find("s_and_b32 s23, s24, 0xffff") != std::string::npos;
      // Fix 1: tree reduction stride.
      // Original: v_lshrrev v3, 1, v1 computes stride = v1/2 where v1 = blockSize.
      // But on GFX9, v1 = 1/sqrt(D) from Fix 4 (not blockSize). Fix: use the
      // blockSize SGPR directly: s10 for attn_forward, s23 for multihead.
      if (stats->total_instructions < 600) {
        size_t br12_pos = translated_asm.find(".L_br12:\n");
        if (br12_pos != std::string::npos) {
          std::string target = "v_lshrrev_b32_e32 v3, 1, v1\n";
          size_t lshr_pos = translated_asm.find(target, br12_pos);
          if (lshr_pos != std::string::npos && lshr_pos - br12_pos < 200) {
            std::string fix;
            if (has_s23_blocksize) {
              // attn_multihead: blockSize is in s23 (prologue), never clobbered.
              // v1 may not hold blockDim here, so use s23 directly.
              fix =
                "v_mov_b32_e32 v3, s5\n"
                "v_min_u32 v3, s23, v3\n"
                "v_lshrrev_b32_e32 v3, 1, v3 ; tree stride = min(N,blockSize)/2\n";
            } else {
              // attn_forward: at .L_br12, v1 = blockDim.x (set by v_mov_b32 v1,s8
              // just before ds_write_b32 that stores per-thread max). s10 has been
              // clobbered by the per-thread max loop (s10 = blockDim*4 = byte
              // stride). Use v1 which correctly holds blockDim at this point.
              fix =
                "v_mov_b32_e32 v3, s5\n"
                "v_min_u32 v3, v1, v3\n"
                "v_lshrrev_b32_e32 v3, 1, v3 ; tree stride = min(N,blockDim)/2\n";
            }
            translated_asm.replace(lshr_pos, target.size(), fix);
          }
        }
      }
      // Fix 2: hoist blockSize before the exec-masked block.
      // On GFX12, the "out of range" block always runs (SALU ignores exec).
      // On GFX9, the block may be skipped. Hoisting ensures blockSize is set.
      //
      // IMPORTANT: attn_forward uses s10 for blockSize (original code does too).
      // attn_multihead uses s23 for blockSize (set in prologue, never clobbered).
      // Writing to s10 in multihead corrupts the head offset needed for O address.
      //
      if (stats->total_instructions < 600 && !has_s23_blocksize) {
        // attn_forward path: hoist s10 = blockSize before the exec-masked block.
        // On GFX12, the "out of range" block always runs (SALU ignores exec).
        // On GFX9, cbranch_execz may skip the block if exec=0. Hoisting ensures
        // s10 (blockSize) is always set before the softmax section begins.
        // Read blockDim.x from hidden_group_size_x = kernarg[explicit_rounded+0xC]
        // which on GFX9/GFX12 both live at the same logical kernarg offset.
        std::string ka_pair = "s[" + std::to_string(kernarg_save_lo) + ":"
                              + std::to_string(kernarg_save_hi) + "]";
        std::string target = "s_xor_b32 s0, exec_lo, s1\n";
        size_t xor_pos = translated_asm.find(target);
        if (xor_pos != std::string::npos) {
          std::string hoist =
            "s_load_dword s10, " + ka_pair + ", 0x3c\n"
            "s_waitcnt lgkmcnt(0)\n"
            "s_and_b32 s10, s10, 0xffff ; FIX2: hoist s10 = blockDim.x\n";
          translated_asm.insert(xor_pos, hoist);
        }
      }
      // Fix 3: v1 = blockSize for exp/normalize loops
      {
        auto replaceFirst = [](std::string& s, const std::string& from, const std::string& to) {
          size_t pos = s.find(from);
          if (pos != std::string::npos) s.replace(pos, from.size(), to);
        };
        if (has_s23_blocksize) {
          // multihead: blockSize is in s23 (prologue), never clobbered
          replaceFirst(translated_asm,
            ".L_br14:\n",
            ".L_br14:\n"
            "v_mov_b32_e32 v1, s23 ; FIX3: v1=blockSize (from prologue s23)\n");
        } else {
          // attn_forward: reload s10 = blockDim.x at .L_br14 (clobbered by find-max tree).
          // Read from hidden_group_size_x at kernarg[explicit_rounded+0xC] (same on GFX9/GFX12).
          std::string ka_pair3 = "s[" + std::to_string(kernarg_save_lo) + ":"
                                 + std::to_string(kernarg_save_hi) + "]";
          replaceFirst(translated_asm,
            ".L_br14:\n",
            ".L_br14:\n"
            "s_load_dword s10, " + ka_pair3 + ", 0x3c\n"
            "s_waitcnt lgkmcnt(0)\n"
            "s_and_b32 s10, s10, 0xffff ; FIX3: reload s10 = blockDim.x\n"
            "v_mov_b32_e32 v1, s10 ; FIX3: v1=blockSize for exp/norm/store loops\n");
        }
      }
      // Fix 4: s3 (thread-0 mask) and v1 (1/sqrt(D) scale factor).
      // s3 is overwritten by v_readfirstlane (float(D)) during sqrt computation,
      // and also by inner product address computation (s_addc_u32 s3, s3, s13).
      // v1 (the sqrt chain result) is 0 on GFX9 (chain produces wrong value),
      // and v1 is reused for other purposes in the loop body.
      // Fix: insert at .L_br4 (outer loop HEAD) so it runs every iteration,
      // not just once before the first iteration.
      if (stats->total_instructions < 600) {
        auto replaceFirst = [](std::string& s, const std::string& from, const std::string& to) {
          size_t pos = s.find(from);
          if (pos != std::string::npos) s.replace(pos, from.size(), to);
        };
        replaceFirst(translated_asm,
          ".L_br4:\n",
          ".L_br4:\n"
          "s_mov_b32 s3, 1 ; FIX4: restore thread-0 mask (clobbered each iter)\n"
          "v_cvt_f32_u32_e32 v1, s6 ; FIX4: v1 = float(D)\n"
          "v_sqrt_f32_e32 v1, v1    ; v1 = sqrt(D)\n"
          "v_rcp_f32_e32 v1, v1     ; v1 = 1/sqrt(D)\n");
      }
      // Fix 5a: save v0 (tid) at .L_br3 before softmax section corrupts it.
      // The exp loop advances v0 by blockSize, making v0 = tid + blockSize.
      // The normalize, sum tree, and inner product entry all need v0 = tid.
      // Only needed for multihead (attn_forward's exp loop doesn't clobber v0).
      if (stats->total_instructions < 600 && has_s23_blocksize) {
        auto replaceFirst = [](std::string& s, const std::string& from, const std::string& to) {
          size_t pos = s.find(from);
          if (pos != std::string::npos) s.replace(pos, from.size(), to);
        };
        replaceFirst(translated_asm,
          ".L_br3:\n",
          ".L_br3:\n"
          "v_mov_b32_e32 v27, v0 ; FIX5a: save tid (exp loop will clobber v0)\n");
      }
      // Fix 5b: restore v0 = tid before v_cmpx_gt_i32 (multihead only).
      // The exp loop in the multihead kernel advances v0 by blockSize, so
      // v0 = tid + blockSize after the loop.  v_cmpx_gt_i32 s6, v0 uses v0
      // as the thread index and must see v0 = tid, not tid + blockSize.
      // attn_forward's exp loop does not clobber v0, so no restore needed.
      // NOTE: Do NOT widen exec after v_cmpx_gt_i32 s6, v0.  That instruction
      // correctly narrows exec to D threads for the output-write section.
      // Widening exec here (exec |= N-thread mask) causes threads D..blockDim-1
      // to write garbage to O[m*D + tid], corrupting adjacent rows' output.
      if (stats->total_instructions < 600 && has_s23_blocksize) {
        size_t cmpx_pos = translated_asm.find("v_cmpx_gt_i32 s6, v0\n");
        if (cmpx_pos != std::string::npos) {
          translated_asm.insert(cmpx_pos,
            "v_mov_b32_e32 v0, v27 ; FIX5b: restore tid (exp loop clobbered v0)\n");
        }
      }
      // Fix 6: force dot-product tree entry in multihead kernel.
      // Root cause: s25 (tree gate) is set by s_cselect_b32 s25, -1, 0 based on SCC.
      // In attn_forward, SCC at that point comes from s_bfe_u32 (blockSize/2 > 0 → SCC=1
      // → s25=-1 → tree always entered). In multihead, six s_add_u32/s_addc_u32
      // address computations follow the s_bfe and clobber SCC with carry-out=0, so
      // s25=0, making s_andn2_b32 vcc_lo,exec_lo,s25 = exec_lo (nonzero) → s_cbranch_vccz
      // falls through to .L_br9 (direct store, SKIPPING the tree reduction entirely).
      // Fix: replace the s_andn2 with s_mov_b32 vcc_lo, 0 to unconditionally enter the
      // tree. The tree self-terminates when stride=0, so this is safe for any N.
      if (has_s23_blocksize) {
        auto replaceFirst = [](std::string& s, const std::string& from, const std::string& to) {
          size_t pos = s.find(from);
          if (pos != std::string::npos) s.replace(pos, from.size(), to);
        };
        replaceFirst(translated_asm,
          "s_andn2_b32 vcc_lo, exec_lo, s25\n",
          "s_mov_b32 vcc_lo, 0 ; FIX6: force tree entry (s25 clobbered by s_addc SCC)\n");
      }
    }
    // Debug: HSA_HOTSWAP_LDS_DUMP=1 injects LDS→global memory dumps at key points.
    // Thread 0 reads LDS values and writes them to the output buffer (s[2:3] = dO).
    // The output buffer will contain debug values instead of actual attention results.
    // Dump layout (multihead): dO[0..N-1] = S[0..N-1] scaled dot products at .L_br3.
    //   The dot products are stored at LDS[0..N*4-1] by the tree's .L_br9 store path
    //   (LDS[i*4] = scaled Q·K_i). Thread 0 reads and writes all N values.
    //   s[2:3] is loaded from kernarg+0x18 (dO pointer) at .L_br3; this is available
    //   after the s_waitcnt. s5 = N (number of keys), used for the loop bound.
    //   Clobbers: v28/v29 (temp VGPRs above kernel range), s34/s35 (scratch SGPRs).
    if (std::getenv("HSA_HOTSWAP_LDS_DUMP") && stats->total_instructions > 400 &&
        stats->total_instructions < 600 &&
        translated_asm.find(".L_br28:") != std::string::npos) {
      // For multihead: Fix 5a inserts "v_mov_b32_e32 v27, v0" at .L_br3 before the
      // s_load_dwordx2 instructions. Match the updated pattern.
      // The multihead kernel loads two pointers: s[14:15] and s[2:3].
      // s[2:3] = dO output buffer (from kernarg+0x18).
      // The dump runs after s_waitcnt so both loads have completed.
      // Only thread 0 performs the dump (exec masked to bit 0).
      // s5 = N (number of keys) — use it to dump N values from LDS[0..N*4-1].
      // Redeclare has_s23_blocksize (defined in prior fix block, now out of scope).
      const bool has_s23_blocksize_dump =
        translated_asm.find("s_and_b32 s23, s24, 0xffff") != std::string::npos;
      if (has_s23_blocksize_dump) {
        // Multihead pattern (after Fix 5a inserted v_mov_b32_e32 v27, v0 at .L_br3)
        // The s_load_dwordx2 s[2:3], s[0:1], 0x18 loads the dO output pointer.
        // After s_waitcnt, s[2:3] = dO base address and s5 = N (keys count).
        // We dump LDS[0..N*4-1] = S[0..N-1] scaled dot products.
        // LDS layout: LDS[i*4] = S[i] (stored by .L_br9 thread-0 store path).
        // Strategy: restrict exec to thread 0, loop N times, read+write, then s_endpgm.
        // Scratch SGPRs: s34 (loop counter i), s35 (byte offset = i*4).
        // Scratch VGPRs: v28 (LDS addr and dO offset), v29 (value read from LDS).
        std::string target =
          "s_load_dwordx2 s[2:3], s[0:1], 0x18\n"
          "s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)\n";
        size_t pos = translated_asm.find(target);
        if (pos != std::string::npos) {
          size_t insert_pos = pos + target.size();
          // Use s34/s35 as scratch SGPRs (above all kernel registers s0..s30 and
          // kernarg save pair s[32:33]). Use v28/v29 as scratch VGPRs (above the
          // transpiler save regs: v24=wg_id_x, v25=wg_id_y used by prologue, and
          // v27 used by Fix 5a for tid save).
          std::string dump =
            "; === LDS DUMP: S[0..N-1] scaled dot products ===\n"
            "s_mov_b32 exec_lo, 1  ; restrict to thread 0 only\n"
            "s_mov_b32 exec_hi, 0\n"
            "s_mov_b32 s34, 0      ; loop counter i = 0 (s34 is above all kernel regs)\n"
            ".L_lds_dump_loop:\n"
            "s_cmp_lt_u32 s34, s5  ; i < N?\n"
            "s_cbranch_scc0 .L_lds_dump_end\n"
            "s_lshl_b32 s35, s34, 2  ; byte_offset = i * 4\n"
            "v_mov_b32_e32 v28, s35  ; LDS read addr (v28 is above all kernel VGPRs)\n"
            "ds_read_b32 v29, v28\n"
            "s_waitcnt lgkmcnt(0)\n"
            "global_store_dword v28, v29, s[2:3]\n"
            "s_add_u32 s34, s34, 1  ; i++\n"
            "s_branch .L_lds_dump_loop\n"
            ".L_lds_dump_end:\n"
            "s_waitcnt vmcnt(0)\n"
            "s_endpgm ; terminate after dump (output is LDS values, not attention)\n"
            "; === END DUMP ===\n";
          translated_asm.insert(insert_pos, dump);
        }
      } else {
        // attn_forward pattern (unchanged from before)
        std::string target = ".L_br3:\ns_load_dwordx2 s[2:3], s[0:1], 0x0\n"
                             "s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)\n";
        size_t pos = translated_asm.find(target);
        if (pos != std::string::npos) {
          size_t insert_pos = pos + target.size();
          std::string dump =
            "; === LDS DUMP: dot products + v1 ===\n"
            "v_mov_b32_e32 v26, 0\n"
            "ds_read_b32 v26, v26\n"
            "v_mov_b32_e32 v27, 4\n"
            "ds_read_b32 v27, v27\n"
            "s_waitcnt lgkmcnt(0)\n"
            "v_mov_b32_e32 v28, 0\n"
            "global_store_dword v28, v26, s[2:3]\n"
            "v_mov_b32_e32 v28, 4\n"
            "global_store_dword v28, v27, s[2:3]\n"
            "v_mov_b32_e32 v28, 8\n"
            "global_store_dword v28, v1, s[2:3]\n"
            "s_waitcnt vmcnt(0)\n"
            "; === END DUMP ===\n";
          translated_asm.insert(insert_pos, dump);
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
    // VCC_hi correctness: In our wave32-in-wave64 model, exec_hi is always 0,
    // so v_cmp produces vcc_hi=0 (inactive lanes → 0). Scalar b32 ops on vcc_lo
    // leave vcc_hi stale, but s_mov_b32 vcc_hi, 0 is inserted before every
    // s_cbranch_vccz/vccnz (see above). s_mov_b32 doesn't clobber SCC. Fixed.
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
    // Fix: force tree reduction entry for kernels with known SCC clobber.
    // s_andn2_b32 vcc_lo, exec_lo, s25 is the tree gate in multihead/attn kernels.
    // s25 should be -1 (from s_cselect) but SCC gets clobbered by s_addc carry-out.
    // Safe: tree self-terminates at stride=0.
    {
      auto replaceFirst = [](std::string& s, const std::string& from, const std::string& to) {
        size_t pos = s.find(from);
        if (pos != std::string::npos) s.replace(pos, from.size(), to);
      };
      // Multihead/attn: s25 + s_cbranch_vccz
      replaceFirst(translated_asm,
        "s_andn2_b32 vcc_lo, exec_lo, s25\n",
        "s_mov_b32 vcc_lo, 0 ; FIX: force tree entry (s25 clobbered by SCC)\n");
    }
    // NOTE: The kernarg[explicit_rounded + 0xC] hidden arg (hidden_group_size_x on GFX9,
    // workgroup_size_x on GFX12) is at the SAME logical kernarg offset on both architectures.
    // GFX1250 kernels read blockDim.x from computed addresses like s[8:9]+0xc or s[0:1]+0x3c,
    // which are derived from the kernarg pointer.  These translated reads work correctly on
    // GFX942 because both GFX9 and GFX12 place blockDim.x at kernarg[explicit_rounded + 0xC].
    // No replacement of these patterns is needed.
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
    // v_bitop2_b32/v_bitop3_b32 are now fully handled in TranslateInstruction
    // (complete 256-entry truth table decomposition). No post-processing needed.
  }

  // (count-down conversion removed — did not fix the split-K issue)

  // ── GFX942 inner loop scheduling fix ──────────────────────────────────────
  // The GFX12→GFX9 translation produces this inner loop pattern:
  //   v_add_u32_e32 vN, sM, vN    ; advance B offset
  //   s_add_i32 sK, sK, 1         ; k++
  //   s_waitcnt expcnt(0)          ; from s_wait_xcnt
  //   s_add_u32 sL, sL, 4         ; advance A ptr lo
  //   s_addc_u32 sH, sH, 0        ; carry
  //   s_cmp_ge_i32 sK, sE         ; k >= end?
  //   s_waitcnt vmcnt(0)           ; redundant
  //   s_waitcnt lgkmcnt(0)         ; redundant
  //   v_fmac_f32_e32 ...           ; accumulate
  //
  // Problems:
  //  1. s_addc_u32 immediately followed by s_cmp_ge_i32 — on GFX942, the SALU
  //     pipeline may forward a stale SCC from s_addc_u32, causing the compare
  //     to read the carry output instead of being an independent SCC write.
  //     Native gfx942 compilers schedule a VALU instruction between them.
  //  2. Three s_waitcnt in the loop where one suffices.
  //
  // Fix: move the v_add_u32 to between s_addc_u32 and s_cmp_ge_i32, and merge
  // the three waitcnts into one s_waitcnt vmcnt(0) lgkmcnt(0).
  {
    std::string fixed;
    std::istringstream fix_iss(translated_asm);
    std::vector<std::string> lines;
    std::string fl;
    while (std::getline(fix_iss, fl)) lines.push_back(fl);

    for (size_t i = 0; i < lines.size(); ++i) {
      // Pattern: v_add_u32_e32 vN, sM, vN  (line i)
      //          s_add_i32 sK, sK, 1        (line i+1)
      //          s_waitcnt expcnt(0)         (line i+2)
      //          s_add_u32 sL, sL, 4        (line i+3)
      //          s_addc_u32 sH, sH, 0       (line i+4)
      //          s_cmp_ge_i32 sK, sE        (line i+5)
      //          s_waitcnt vmcnt(0)          (line i+6)
      //          s_waitcnt lgkmcnt(0)        (line i+7)
      //          v_fmac_f32_e32 ...          (line i+8)
      //          s_cbranch_scc0 ...          (line i+9)
      if (i + 9 < lines.size() &&
          lines[i].find("v_add_u32_e32") != std::string::npos &&
          lines[i+1].find("s_add_i32") != std::string::npos &&
          lines[i+1].find(", 1") != std::string::npos &&
          (lines[i+2].find("s_waitcnt") != std::string::npos) &&
          lines[i+3].find("s_add_u32") != std::string::npos &&
          lines[i+4].find("s_addc_u32") != std::string::npos &&
          lines[i+5].find("s_cmp_ge_i32") != std::string::npos &&
          lines[i+8].find("v_fmac_f32") != std::string::npos &&
          lines[i+9].find("s_cbranch_scc0") != std::string::npos) {
        // Wait for both loads first, then do address updates and loop control.
        // CRITICAL: v_fmac must be BEFORE s_cmp_ge_i32, not after it.
        // On GFX942, a VALU instruction between s_cmp (SCC write) and
        // s_cbranch_scc0 (SCC read) can corrupt SCC forwarding.
        // The branch must IMMEDIATELY follow the compare with no VALU gap.
        fixed += "s_waitcnt vmcnt(0) lgkmcnt(0)\n"; // wait for BOTH loads first
        fixed += lines[i+8] + "\n";               // v_fmac_f32_e32 (BEFORE compare!)
        fixed += lines[i] + "\n";                  // v_add_u32_e32 vN, sM, vN
        fixed += lines[i+1] + "\n";               // s_add_i32 sK, sK, 1
        fixed += lines[i+3] + "\n";               // s_add_u32 sL, sL, 4
        fixed += lines[i+4] + "\n";               // s_addc_u32 sH, sH, 0
        fixed += lines[i+5] + "\n";               // s_cmp_ge_i32 sK, sE
        fixed += lines[i+9] + "\n";               // s_cbranch_scc0 (IMMEDIATELY after cmp)
        i += 9;  // skip all 10 lines
        std::cerr << "hotswap: transpile: applied inner loop scheduling fix\n";
        continue;
      }
      fixed += lines[i] + "\n";
    }
    translated_asm = fixed;
  }

  // (GLC/sc0 additions removed — cache coherency is not the root cause)

  // (xcnt→vmcnt fix is now in TranslateWaitInstruction)

  // ── HSA_HOTSWAP_SPLITK_NOWAIT: isolate s_waitcnt as root cause ───────────
  // Hypothesis: the s_waitcnt vmcnt(0) lgkmcnt(0) inside the split-K inner
  // loop causes a GFX942 SALU pipeline hazard that makes s_cbranch_scc0 read
  // a stale SCC from the implicit serialization fence, not from s_cmp_ge_i32.
  //
  // This pass rewrites the entire inner loop body (from .L_br3: to the back-
  // branch) by:
  //   - Stripping all global_load_dword / s_load_dword inside the loop
  //   - Stripping s_waitcnt lines inside the loop
  //   - Replacing v_fmac_f32_e32 with v_add_f32_e32 v2, 1.0, v2
  //   - Keeping all SALU loop-control instructions intact
  //
  // Expected outcome: v2 == float(K) after the loop if SALU control is correct.
  // If loop runs correctly → waitcnt is the root cause of non-determinism.
  // If loop still fails → root cause is in SALU pipeline independent of waitcnt.
  //
  // Target loop structure (after inner-loop scheduling fix):
  //   .L_br3:
  //   [global_load_dword ...]   ; load A[k] -- STRIPPED
  //   [s_load_dword ...]        ; load B[k] -- STRIPPED
  //   s_waitcnt vmcnt(0) lgkmcnt(0)          -- STRIPPED
  //   v_add_u32_e32 vN, sM, vN              ; advance B ptr
  //   s_add_i32 sK, sK, 1                   ; k++
  //   s_add_u32 sL, sL, 4                   ; advance A ptr lo
  //   s_addc_u32 sH, sH, 0                  ; carry
  //   s_cmp_ge_i32 sK, sE                   ; k >= end_k?
  //   v_fmac_f32_e32 ...        -- REPLACED by v_add_f32_e32 v2, 1.0, v2
  //   s_cbranch_scc0 .L_br3                 ; loop back
  //
  // Gated on env var; only runs on split-K compute kernel
  // (identified by instruction count 200–400, which excludes the attn kernels
  // at 400+ instructions).
  if (std::getenv("HSA_HOTSWAP_SPLITK_NOWAIT") && stats &&
      stats->total_instructions > 200 && stats->total_instructions < 400) {
    std::string nowait_fixed;
    std::istringstream nw_iss(translated_asm);
    std::vector<std::string> nw_lines;
    std::string nw_line;
    while (std::getline(nw_iss, nw_line)) nw_lines.push_back(nw_line);

    // Find .L_br3: label to anchor the loop body.
    size_t loop_start = std::string::npos;
    for (size_t i = 0; i < nw_lines.size(); ++i) {
      if (nw_lines[i] == ".L_br3:") { loop_start = i; break; }
    }

    if (loop_start == std::string::npos) {
      std::cerr << "hotswap: transpile: SPLITK_NOWAIT: .L_br3: label not found\n";
    } else {
      // Find the s_cbranch_scc0 .L_br3 back-branch that closes the loop.
      size_t loop_end = std::string::npos;
      for (size_t i = loop_start + 1; i < nw_lines.size(); ++i) {
        if (nw_lines[i].find("s_cbranch_scc0") != std::string::npos &&
            nw_lines[i].find(".L_br3") != std::string::npos) {
          loop_end = i;
          break;
        }
      }

      if (loop_end == std::string::npos) {
        std::cerr << "hotswap: transpile: SPLITK_NOWAIT: s_cbranch_scc0 .L_br3 not found\n";
      } else {
        std::cerr << "hotswap: transpile: SPLITK_NOWAIT: rewriting loop body ["
                  << loop_start << "–" << loop_end << "]\n";

        // Emit lines before the loop label unchanged.
        for (size_t i = 0; i < loop_start; ++i)
          nowait_fixed += nw_lines[i] + "\n";

        // Emit the loop label itself.
        nowait_fixed += nw_lines[loop_start] + "\n";  // .L_br3:

        // Emit loop body with loads, waitcnt, and v_fmac stripped/replaced.
        for (size_t i = loop_start + 1; i <= loop_end; ++i) {
          const std::string& ln = nw_lines[i];

          // Strip memory loads (no outstanding requests → no need for waitcnt).
          if (ln.find("global_load_dword") != std::string::npos ||
              ln.find("s_load_dword") != std::string::npos) {
            nowait_fixed += "s_nop 0 ; NOWAIT: stripped load: " + ln + "\n";
            continue;
          }
          // Strip waitcnt — this is the key change being tested.
          if (ln.find("s_waitcnt") != std::string::npos) {
            nowait_fixed += "; NOWAIT: stripped: " + ln + "\n";
            continue;
          }
          // Replace accumulate with constant add so v2 == float(K) after loop.
          if (ln.find("v_fmac_f32") != std::string::npos) {
            nowait_fixed += "v_add_f32_e32 v2, 1.0, v2 ; NOWAIT: const accumulate (was: " + ln + ")\n";
            continue;
          }
          // Keep everything else (SALU control, address advances, back-branch).
          nowait_fixed += ln + "\n";
        }

        // Emit lines after the loop unchanged.
        for (size_t i = loop_end + 1; i < nw_lines.size(); ++i)
          nowait_fixed += nw_lines[i] + "\n";

        translated_asm = nowait_fixed;
        std::cerr << "hotswap: transpile: SPLITK_NOWAIT pass complete\n";
      }
    }
  }

  // Post-processing: fix VALU → v_readfirstlane_b32 data hazards.
  // GFX9 requires at least 1 cycle between any VALU writing a VGPR and
  // v_readfirstlane_b32 reading that VGPR. On GFX12, s_delay_alu annotations
  // cover this, but some cases slip through (e.g. the s_delay_alu annotates
  // an instruction pair and only ONE s_nop was inserted). This pass catches
  // all adjacent VALU→readfirstlane pairs in the final assembly.
  {
    // Helper: extract the first destination operand from an instruction line.
    // Returns the register name (e.g. "v14") or empty string on failure.
    auto extractFirstDest = [](const std::string& ln) -> std::string {
      size_t s = ln.find_first_not_of(" \t");
      if (s == std::string::npos || ln[s] == ';' || ln[s] == '.') return "";
      // Skip mnemonic
      size_t e = ln.find_first_of(" \t", s);
      if (e == std::string::npos) return "";
      size_t op_start = ln.find_first_not_of(" \t", e);
      if (op_start == std::string::npos) return "";
      size_t op_end = ln.find_first_of(" \t,;", op_start);
      if (op_end == std::string::npos) op_end = ln.size();
      return ln.substr(op_start, op_end - op_start);
    };
    // Helper: is this line a "blocking" instruction (not s_nop, not label, not comment)?
    auto isEffective = [](const std::string& ln) -> bool {
      size_t s = ln.find_first_not_of(" \t");
      if (s == std::string::npos) return false;
      if (ln[s] == ';') return false;                       // comment
      if (ln[s] == '.') return false;                       // label
      const std::string t = ln.substr(s);
      if (t.find("s_nop") == 0) return false;               // s_nop doesn't count
      return true;
    };
    // Helper: is this line a VALU instruction (writes to a VGPR)?
    auto isVALU = [](const std::string& ln) -> bool {
      size_t s = ln.find_first_not_of(" \t");
      if (s == std::string::npos) return false;
      // Check for v_ prefix (VALU), excluding v_readfirstlane which is SALU
      const std::string t = ln.substr(s);
      if (t.find("v_") != 0) return false;
      if (t.find("v_readfirstlane") == 0) return false;
      return true;
    };

    std::istringstream rfln_iss(translated_asm);
    std::string rfln_line;
    std::vector<std::string> rfln_lines;
    while (std::getline(rfln_iss, rfln_line)) rfln_lines.push_back(rfln_line);

    bool any_inserted = false;
    // Rebuild translated_asm with insertions
    std::string rfln_fixed;
    // Track the last effective instruction and its index in output.
    std::string last_effective;
    std::string last_effective_dest;
    for (size_t i = 0; i < rfln_lines.size(); ++i) {
      const auto& ln = rfln_lines[i];
      size_t ls = ln.find_first_not_of(" \t");
      if (ls != std::string::npos && ln.find("v_readfirstlane_b32", ls) == ls) {
        // Extract the VGPR source of this readfirstlane
        size_t op_s = ln.find_first_of(" \t", ls + 19);
        if (op_s != std::string::npos) {
          op_s = ln.find_first_not_of(" \t", op_s);
          if (op_s != std::string::npos) {
            size_t comma = ln.find(',', op_s);
            if (comma != std::string::npos) {
              size_t src_s = ln.find_first_not_of(" \t", comma + 1);
              if (src_s != std::string::npos) {
                size_t src_e = ln.find_first_of(" \t;", src_s);
                if (src_e == std::string::npos) src_e = ln.size();
                std::string src_reg = ln.substr(src_s, src_e - src_s);
                // If the previous effective instruction wrote this VGPR AND was VALU
                if (!last_effective.empty() && isVALU(last_effective) &&
                    last_effective_dest == src_reg) {
                  rfln_fixed += "s_nop 0 ; GFX9 hazard: VALU→readfirstlane on " + src_reg + "\n";
                  any_inserted = true;
                }
              }
            }
          }
        }
      }
      rfln_fixed += ln + "\n";
      if (isEffective(ln)) {
        last_effective = ln;
        last_effective_dest = extractFirstDest(ln);
      }
    }
    if (any_inserted) {
      translated_asm = rfln_fixed;
    }
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
    // For ELFs with device library code before the kernel, only replace
    // from the kernel's code offset onward, preserving preceding code.
    uint64_t kernel_text_offset = 0;
    if (!kernels.empty())
      kernel_text_offset = kernels[0].code_offset;
    uint64_t kernel_region_size = elf_info.text_size - kernel_text_offset;

    if (new_text_size <= kernel_region_size) {
      // Preserve device library code before kernel (bytes 0..kernel_text_offset-1)
      // Copy transpiled kernel code at the kernel's starting offset
      std::memcpy(new_elf + elf_info.text_offset + kernel_text_offset,
                  new_text, new_text_size);
      // s_nop 0 fill remainder after kernel — GFX9 does not support s_code_end
      // (0xBF9F0000 is an illegal instruction on GFX9 and can trigger a trap if
      // the instruction prefetcher reads past s_endpgm).
      uint8_t nop_bytes[] = {0x00, 0x00, 0x80, 0xBF};  // s_nop 0
      for (uint64_t i = kernel_text_offset + new_text_size;
           i + 4 <= elf_info.text_size; i += 4) {
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
        // s_nop 0 pad .text to aligned boundary (GFX9 has no s_code_end)
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
            << (elf_info.text_size - new_text_size) << " bytes s_nop padding)\n";

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
            // GFX9 RSRC1: bits [5:0] = VGPR field (gran 4), bits [9:6] = SGPR field (gran 8)
            //             bits [11:10] = PRIORITY
            // GFX12 RSRC1: bits [5:0] = VGPR field, bits [11:6] = SGPR field
            // Convert VGPR: GFX12 granularity 12 (wave32) → GFX9 granularity 4.
            // Add 8 extra VGPRs for save registers (sv_x, sv_y) and 4 temp
            // VGPRs (vt0-vt3) used by instruction expansion + even-align pad.
            uint32_t vgpr_field12 = rsrc1 & 0x3Fu;
            uint32_t sgpr_field12_rd = (rsrc1 >> 6) & 0x3Fu;  // save before clear
            uint32_t num_vgprs = (vgpr_field12 + 1u) * 12u;  // GFX12 wave32 gran=12
            if (num_vgprs < 8u) num_vgprs = 8u;
            num_vgprs += 8u;  // room for save registers + temp VGPRs
            uint32_t gfx9_vgpr = ((num_vgprs + 3u) / 4u) - 1u;
            if (gfx9_vgpr > 62u) gfx9_vgpr = 62u;
            // GFX942: RSRC1 VGPR = ACCUM_OFFSET (all arch, no extra accum VGPRs).
            uint32_t rsrc1_vgpr_field = gfx9_vgpr;
            if (rsrc1_vgpr_field > 63u) rsrc1_vgpr_field = 63u;
            rsrc1 &= ~0xFFFu;
            rsrc1 |= (rsrc1_vgpr_field & 0x3Fu);  // VGPR in bits [5:0]
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
              uint32_t gfx9_sgpr = ((num_sgprs_rd + 7u) / 8u) - 1u;  // ceiling division
              if (gfx9_sgpr > 12u) gfx9_sgpr = 12u;
              rsrc1 |= (gfx9_sgpr << 6u);  // SGPR in bits [9:6]
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
              uint32_t r3_log;
              std::memcpy(&r3_log, desc + 44, 4);
              uint32_t accum_off = r3_log & 0x3Fu;
              std::cerr << "hotswap: transpile: patched KD: RSRC1=0x"
                        << std::hex << r1 << " RSRC2=0x" << r2
                        << " RSRC3=0x" << r3_log
                        << " props=0x" << p << std::dec
                        << " arch_vgpr=" << (accum_off + 1) * 4
                        << " sgpr=" << (((r1 >> 6) & 0xf) + 1) * 8
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

  // Debug: dump patched ELF if HSA_HOTSWAP_DUMP_ELF is set.
  // If HSA_HOTSWAP_DUMP_KERNEL is also set, only dump if that kernel name appears in the ELF.
  if (auto* dump_path = std::getenv("HSA_HOTSWAP_DUMP_ELF")) {
    bool do_dump = true;
    if (auto* filter_name = std::getenv("HSA_HOTSWAP_DUMP_KERNEL")) {
      // Check if the kernel name appears anywhere in the ELF binary data.
      std::string_view elf_view(reinterpret_cast<const char*>(elf), size);
      do_dump = elf_view.find(filter_name) != std::string_view::npos;
    }
    if (do_dump) {
      FILE* fp = fopen(dump_path, "wb");
      if (fp) {
        fwrite(elf, 1, size, fp);
        fclose(fp);
        std::cerr << "hotswap: transpile: dumped patched ELF to " << dump_path
                  << " (" << size << " bytes)\n";
      }
    }
  }

  return result;
}

}  // namespace hotswap
}  // namespace rocr
