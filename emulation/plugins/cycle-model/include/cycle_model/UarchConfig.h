// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file UarchConfig.h
/// @brief Runtime machine description loaded from JSON. Replaces the earlier
/// compile-time trait headers — chosen so the cycle model rides rocjitsu's
/// existing JSON config workflow instead of duplicating topology constants in
/// C++. One config object per uarch; selected at startup from the SoC arch.
///
/// Loader/validator live in src/uarch_config.cpp (keeps nlohmann_json out of the
/// hot-path headers). Per-opcode latency is keyed on rocjitsu's string mnemonic
/// directly, so no opcode enum / string->enum translation is needed.

#pragma once

#include "cycle_model/InstrEvent.h"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace cycle_model {

struct CacheSpec {
  uint32_t size_kb = 0, ways = 0, line_bytes = 0, hit_latency = 0, miss_to_next_level = 0;
};

struct DepRule { std::string producer, consumer; uint32_t min_gap_cyc = 0; bool same_dest_only = true; };

struct UarchConfig {
  std::string name;                       // "cdna4"
  uint32_t wave_size = 64, simds_per_cu = 4, wave_slots_per_simd = 8, front_end_issue_per_simd = 1;
  uint32_t vgprs_per_simd = 0, sgprs_per_cu = 0;
  uint32_t lds_bytes_per_cu = 0, lds_banks = 32, lds_bytes_per_bank = 0;

  PipeSpec valu, salu, smem, vmem, lds_pipe, mfma, wmma;
  bool has_mfma = false, has_wmma = false, has_vopd = false;

  CacheSpec l1v, l1s, l2;
  uint32_t mshrs_per_l1v = 0, hbm_channels = 0;
  uint32_t l2_bytes_per_cycle = 0;              // L2 bandwidth-queue drain rate (bytes/cycle)
  uint32_t hbm_access_latency = 0;             // DRAM access latency past the channel queue (cycles)
  uint32_t hbm_bytes_per_channel_per_cycle = 0; // per-channel HBM BW (bytes/cycle); replaces gbs*period calc

  std::unordered_map<std::string, PipeSpec> opcode_latency;   // keyed on Instruction::mnemonic()
  std::vector<DepRule> dep_rules;

  PipeSpec lookup_latency(std::string_view mnemonic, PipeSpec fallback) const {
    auto it = opcode_latency.find(std::string(mnemonic));
    return it != opcode_latency.end() ? it->second : fallback;
  }
};

UarchConfig load_uarch_config(const std::string& json_path);   // src/uarch_config.cpp
void        validate_uarch_config(const UarchConfig&);

}  // namespace cycle_model
