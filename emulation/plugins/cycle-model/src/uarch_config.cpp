// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "cycle_model/UarchConfig.h"

#include <nlohmann/json.hpp>
#include <fstream>
#include <stdexcept>

namespace cycle_model {
namespace {

/// Strict accessors — every consumed key must be present. No silent defaults:
/// a missing/typo'd key fails the load loudly instead of degrading to a
/// plausible-but-wrong machine description.
const nlohmann::json& req(const nlohmann::json& j, const char* key) {
  auto it = j.find(key);
  if (it == j.end())
    throw std::runtime_error(std::string("cycle_model: missing required key: ") + key);
  return *it;
}
template <class T> T reqv(const nlohmann::json& j, const char* key) {
  return req(j, key).get<T>();
}

PipeSpec pipe(const nlohmann::json& j) {
  return {reqv<uint32_t>(j, "issue_rate"), reqv<uint32_t>(j, "base_latency")};
}
CacheSpec cache(const nlohmann::json& j) {
  return {reqv<uint32_t>(j, "size_kb"), reqv<uint32_t>(j, "ways"),
          reqv<uint32_t>(j, "line_bytes"), reqv<uint32_t>(j, "hit_latency"),
          reqv<uint32_t>(j, "miss_to_next_level")};
}
}  // namespace

UarchConfig load_uarch_config(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cycle_model: cannot open uarch config: " + path);
  nlohmann::json j; f >> j;

  UarchConfig c;
  c.name = reqv<std::string>(j, "name");

  c.wave_size             = reqv<uint32_t>(j, "wave_size");
  c.simds_per_cu          = reqv<uint32_t>(j, "simds_per_cu");
  c.wave_slots_per_simd   = reqv<uint32_t>(j, "wave_slots_per_simd");
  c.front_end_issue_per_simd = reqv<uint32_t>(j, "front_end_issue_per_simd");
  c.vgprs_per_simd        = reqv<uint32_t>(j, "vgprs_per_simd");
  c.sgprs_per_cu          = reqv<uint32_t>(j, "sgprs_per_cu");
  c.lds_bytes_per_cu      = reqv<uint32_t>(j, "lds_bytes_per_cu");
  c.lds_banks             = reqv<uint32_t>(j, "lds_banks");
  c.lds_bytes_per_bank    = reqv<uint32_t>(j, "lds_bytes_per_bank");

  c.valu     = pipe(req(j, "valu"));
  c.salu     = pipe(req(j, "salu"));
  c.smem     = pipe(req(j, "smem"));
  c.vmem     = pipe(req(j, "vmem"));
  c.lds_pipe = pipe(req(j, "lds_pipe"));

  // Matrix unit. On CDNA the MFMA pipe IS the XDLOP/Matrix-Core datapath, so
  // there is no separate "xdl" pipe — matrix throughput is encoded per-opcode
  // in opcode_latency. RDNA has no MFMA; its matrix path is WMMA on VALU lanes.
  c.has_mfma = reqv<bool>(j, "has_mfma");
  if (c.has_mfma) c.mfma = pipe(req(j, "mfma"));
  c.has_wmma = reqv<bool>(j, "has_wmma");
  if (c.has_wmma) c.wmma = pipe(req(j, "wmma"));
  c.has_vopd = reqv<bool>(j, "has_vopd");

  c.l1v = cache(req(j, "l1v"));
  c.l1s = cache(req(j, "l1s"));
  c.l2  = cache(req(j, "l2"));
  c.mshrs_per_l1v     = reqv<uint32_t>(j, "mshrs_per_l1v");
  c.hbm_channels      = reqv<uint32_t>(j, "hbm_channels");
  c.l2_bytes_per_cycle = reqv<uint32_t>(j, "l2_bytes_per_cycle");
  c.hbm_access_latency = reqv<uint32_t>(j, "hbm_access_latency");
  c.hbm_bytes_per_channel_per_cycle = reqv<uint32_t>(j, "hbm_bytes_per_channel_per_cycle");

  for (auto& [k, v] : req(j, "opcode_latency").items())
    c.opcode_latency[k] = pipe(v);

  for (const auto& dr : req(j, "dep_rules")) {
    DepRule r;
    r.producer       = reqv<std::string>(dr, "producer");
    r.consumer       = reqv<std::string>(dr, "consumer");
    r.min_gap_cyc    = reqv<uint32_t>(dr, "min_gap_cyc");
    r.same_dest_only = reqv<bool>(dr, "same_dest_only");
    c.dep_rules.push_back(std::move(r));
  }

  validate_uarch_config(c);
  return c;
}

void validate_uarch_config(const UarchConfig& c) {
  if (c.simds_per_cu != 2 && c.simds_per_cu != 4) throw std::runtime_error("simds_per_cu must be 2 or 4");
  if (c.wave_size != 32 && c.wave_size != 64) throw std::runtime_error("wave_size must be 32 or 64");
  if (c.lds_bytes_per_bank && c.lds_banks * c.lds_bytes_per_bank != c.lds_bytes_per_cu)
    throw std::runtime_error("lds banks*bytes != lds_bytes_per_cu");

  // Cache line size drives address coalescing (line_base = addr & ~(line_bytes-1)) and
  // set indexing, both of which require a non-zero power of two. A present cache
  // (size_kb>0) must have a pow2 line_bytes that evenly partitions its capacity.
  auto check_cache = [](const char* name, const CacheSpec& s) {
    if (!s.size_kb) return;                                   // absent cache: skip
    const bool pow2 = s.line_bytes && (s.line_bytes & (s.line_bytes - 1)) == 0;
    if (!s.ways || !pow2)
      throw std::runtime_error(std::string("cache ") + name +
                               ": line_bytes must be a non-zero power of two and ways>0");
    if ((s.size_kb * 1024u) % (s.ways * s.line_bytes) != 0)
      throw std::runtime_error(std::string("cache ") + name +
                               ": size_kb*1024 not divisible by ways*line_bytes");
  };
  check_cache("l1v", c.l1v);
  check_cache("l1s", c.l1s);
  check_cache("l2", c.l2);
}

}  // namespace cycle_model
