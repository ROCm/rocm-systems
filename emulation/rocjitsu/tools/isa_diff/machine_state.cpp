// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "machine_state.h"

#include <algorithm>
#include <cstdlib>

namespace rocjitsu::isa_diff {

Json MachineState::to_json() const {
  Json j = Json::object();
  Json locs = Json::object();
  for (unsigned l = 0; l < kNumLocs; ++l)
    locs.set(loc_name(l), hex32(loc[l]));
  j.set("locations", std::move(locs));
  Json f = Json::object();
  f.set("status_except_scc", hex32(frame.status_except_scc));
  f.set("mode", hex32(frame.mode));
  f.set("trapsts", hex32(frame.trapsts));
  Json ttmp = Json::array();
  for (uint32_t t : frame.ttmp)
    ttmp.push(hex32(t));
  f.set("ttmp", std::move(ttmp));
  f.set("scratch_base", hex64(frame.scratch_base));
  f.set("s102", hex32(frame.s102));
  f.set("s103", hex32(frame.s103));
  Json wc = Json::array();
  for (uint8_t c : frame.wait_counters)
    wc.push(static_cast<uint32_t>(c));
  f.set("wait_counters", std::move(wc));
  f.set("in_trap_handler", frame.in_trap_handler);
  f.set("wave_state", static_cast<uint32_t>(frame.wave_state));
  f.set("sleep_cycles", frame.sleep_cycles);
  j.set("frame", std::move(f));
  if (vgpr_count) {
    Json v = Json::object();
    for (unsigned r = 0; r < vgpr_count; ++r) {
      Json lanes = Json::array();
      for (unsigned lane = 0; lane < kWaveLanes; ++lane)
        lanes.push(hex32(vget(r, lane)));
      v.set("v" + std::to_string(r), std::move(lanes));
    }
    j.set("vgprs", std::move(v));
  }
  if (mem_seed != 0 || !mem_written.empty()) {
    Json m = Json::object();
    m.set("seed", hex64(mem_seed));
    Json init = Json::object();
    for (const auto &[w, v] : mem_init)
      init.set(hex64(w << 2), hex32(v));
    m.set("initial", std::move(init));
    Json w = Json::object();
    for (const auto &[addr, byte] : mem_written)
      w.set(hex64(addr), hex32(byte));
    m.set("written", std::move(w));
    if (!mem_dead.empty()) {
      Json dead = Json::array();
      for (uint64_t addr : mem_dead)
        dead.push(hex64(addr));
      m.set("dead", std::move(dead));
    }
    j.set("memory", std::move(m));
  }
  if (exit_pc)
    j.set("exit_pc", hex64(exit_pc));
  if (asm_seed != 0 || !asm_uses.empty()) {
    Json a = Json::object();
    a.set("seed", hex64(asm_seed));
    Json uses = Json::array();
    for (const AsmUse &u : asm_uses) {
      Json uj = Json::object();
      uj.set("block", static_cast<uint64_t>(u.block));
      Json vals = Json::array();
      for (uint32_t v : u.values)
        vals.push(hex32(v));
      uj.set("values", std::move(vals));
      if (std::ranges::any_of(u.fp, [](uint8_t t) { return t != 0; })) {
        Json tags = Json::array();
        for (uint8_t t : u.fp)
          tags.push(static_cast<uint64_t>(t));
        uj.set("fp", std::move(tags));
      }
      uses.push(std::move(uj));
    }
    a.set("uses", std::move(uses));
    j.set("inline_asm", std::move(a));
  }
  // Floating-point provenance: per tagged VGPR, two hex digits (one FpTag) per lane.
  const bool stagged = std::ranges::any_of(sfp, [](uint8_t t) { return t != 0; });
  if (std::ranges::any_of(vfp, [](uint8_t t) { return t != 0; }) || !mem_fp.empty() || stagged) {
    Json f = Json::object();
    if (stagged) {
      Json sc = Json::object();
      for (unsigned l = 0; l < kNumLocs; ++l)
        if (sfp_tag(l))
          sc.set(loc_name(l), static_cast<uint64_t>(sfp_tag(l)));
      f.set("scalars", std::move(sc));
    }
    Json regs = Json::object();
    for (unsigned r = 0; r < vgpr_count; ++r) {
      std::string lanes;
      bool any = false;
      for (unsigned lane = 0; lane < kWaveLanes; ++lane) {
        const uint8_t t = fp_tag(r, lane);
        any = any || t;
        static constexpr char kHex[] = "0123456789abcdef";
        lanes += kHex[t >> 4];
        lanes += kHex[t & 15];
      }
      if (any)
        regs.set("v" + std::to_string(r), lanes);
    }
    f.set("vgprs", std::move(regs));
    Json units = Json::array();
    for (const FpUnit &u : mem_fp) {
      Json uj = Json::array();
      uj.push(hex64(u.addr));
      uj.push(static_cast<uint64_t>(u.size));
      uj.push(static_cast<uint64_t>(u.format));
      units.push(std::move(uj));
    }
    f.set("memory", std::move(units));
    j.set("fp", std::move(f));
  }
  return j;
}

std::optional<MachineState> MachineState::from_json(const Json &j, std::string &error) {
  MachineState s;
  const Json &locs = j["locations"];
  for (unsigned l = 0; l < kNumLocs; ++l) {
    auto v = json_u64(locs[loc_name(l)]);
    if (!v && l >= 102 && l < kNumSgprs)
      v = 0; // s102-s105: absent from states written for GFX9 before they were tracked
    if (!v) {
      error = "state is missing location " + loc_name(l);
      return std::nullopt;
    }
    s.set(l, static_cast<uint32_t>(*v));
  }
  s.exit_pc = json_u64(j["exit_pc"]).value_or(0);
  const Json &f = j["frame"];
  auto u32 = [&](const char *key) { return static_cast<uint32_t>(json_u64(f[key]).value_or(0)); };
  s.frame.status_except_scc = u32("status_except_scc");
  s.frame.mode = u32("mode");
  s.frame.trapsts = u32("trapsts");
  const auto &ttmp = f["ttmp"].items();
  for (size_t i = 0; i < s.frame.ttmp.size() && i < ttmp.size(); ++i)
    s.frame.ttmp[i] = static_cast<uint32_t>(json_u64(ttmp[i]).value_or(0));
  s.frame.scratch_base = json_u64(f["scratch_base"]).value_or(0);
  s.frame.s102 = u32("s102");
  s.frame.s103 = u32("s103");
  const auto &wc = f["wait_counters"].items();
  for (size_t i = 0; i < s.frame.wait_counters.size() && i < wc.size(); ++i)
    s.frame.wait_counters[i] = static_cast<uint8_t>(wc[i].as_uint());
  s.frame.in_trap_handler = f["in_trap_handler"].as_bool();
  s.frame.wave_state = static_cast<uint8_t>(f["wave_state"].as_uint());
  s.frame.sleep_cycles = static_cast<uint32_t>(f["sleep_cycles"].as_uint());
  const Json &mem = j["memory"];
  if (mem.is_object()) {
    s.mem_seed = json_u64(mem["seed"]).value_or(0);
    for (const auto &[addr, byte] : mem["written"].members()) {
      auto a = json_u64(Json(addr));
      auto b = json_u64(byte);
      if (!a || !b) {
        error = "bad memory byte in state";
        return std::nullopt;
      }
      s.mem_written.emplace_back(*a, static_cast<uint8_t>(*b));
    }
    std::sort(s.mem_written.begin(), s.mem_written.end());
    for (const Json &d : mem["dead"].items()) {
      auto a = json_u64(d);
      if (!a) {
        error = "bad dead memory address in state";
        return std::nullopt;
      }
      s.mem_dead.push_back(*a);
    }
    std::sort(s.mem_dead.begin(), s.mem_dead.end());
    for (const auto &[addr, value] : mem["initial"].members()) {
      auto a = json_u64(Json(addr));
      auto v = json_u64(value);
      if (!a || !v) {
        error = "bad initial memory word in state";
        return std::nullopt;
      }
      s.mem_init.emplace_back(*a >> 2, static_cast<uint32_t>(*v));
    }
    std::sort(s.mem_init.begin(), s.mem_init.end());
  }
  const Json &asm_state = j["inline_asm"];
  if (asm_state.is_object()) {
    s.asm_seed = json_u64(asm_state["seed"]).value_or(0);
    for (const Json &uj : asm_state["uses"].items()) {
      AsmUse u;
      u.block = static_cast<uint32_t>(uj["block"].as_uint());
      for (const Json &v : uj["values"].items()) {
        auto x = json_u64(v);
        if (!x) {
          error = "bad inline asm value in state";
          return std::nullopt;
        }
        u.values.push_back(static_cast<uint32_t>(*x));
      }
      for (const Json &t : uj["fp"].items())
        u.fp.push_back(static_cast<uint8_t>(t.as_uint()));
      if (u.fp.empty())
        u.fp.assign(u.values.size(), 0);
      s.asm_uses.push_back(std::move(u));
    }
  }
  const Json &v = j["vgprs"];
  if (v.is_object()) {
    s.resize_vgprs(static_cast<unsigned>(v.members().size()));
    for (unsigned r = 0; r < s.vgpr_count; ++r) {
      const auto &lanes = v["v" + std::to_string(r)].items();
      if (lanes.size() != kWaveLanes) {
        error = "state VGPR v" + std::to_string(r) + " does not have 64 lanes";
        return std::nullopt;
      }
      for (unsigned lane = 0; lane < kWaveLanes; ++lane) {
        auto x = json_u64(lanes[lane]);
        if (!x) {
          error = "bad VGPR lane value";
          return std::nullopt;
        }
        s.vset(r, lane, static_cast<uint32_t>(*x));
      }
    }
  }
  const Json &fp = j["fp"];
  if (fp.is_object()) {
    if (fp["scalars"].is_object()) {
      s.sfp.assign(kNumLocs, 0);
      for (const auto &[name, tag] : fp["scalars"].members()) {
        const auto l = parse_loc(name);
        if (!l) {
          error = "bad floating-point scalar tag for " + name;
          return std::nullopt;
        }
        s.sfp[*l] = static_cast<uint8_t>(tag.as_uint());
      }
    }
    s.vfp.assign(size_t{s.vgpr_count} * kWaveLanes, 0);
    for (const auto &[name, lanes] : fp["vgprs"].members()) {
      const unsigned r = static_cast<unsigned>(std::strtoul(name.c_str() + 1, nullptr, 10));
      const std::string &hex = lanes.as_string();
      if (name.empty() || name[0] != 'v' || r >= s.vgpr_count || hex.size() != 2 * kWaveLanes) {
        error = "bad floating-point tags for " + name;
        return std::nullopt;
      }
      for (unsigned lane = 0; lane < kWaveLanes; ++lane)
        s.vfp[size_t{r} * kWaveLanes + lane] =
            static_cast<uint8_t>(std::stoul(hex.substr(2 * lane, 2), nullptr, 16));
    }
    for (const Json &u : fp["memory"].items()) {
      const auto &it = u.items();
      auto a = it.size() == 3 ? json_u64(it[0]) : std::nullopt;
      if (!a) {
        error = "bad floating-point memory unit";
        return std::nullopt;
      }
      s.mem_fp.push_back(
          {*a, static_cast<uint8_t>(it[1].as_uint()), static_cast<uint8_t>(it[2].as_uint())});
    }
  }
  return s;
}

VSet differing_vgprs(const MachineState &a, const MachineState &b) {
  VSet out;
  const unsigned n = std::max(a.vgpr_count, b.vgpr_count);
  for (unsigned r = 0; r < n; ++r) {
    if (r >= a.vgpr_count || r >= b.vgpr_count) {
      out.set(r);
      continue;
    }
    const uint32_t *pa = a.vgpr.data() + size_t{r} * kWaveLanes;
    const uint32_t *pb = b.vgpr.data() + size_t{r} * kWaveLanes;
    if (!std::equal(pa, pa + kWaveLanes, pb))
      out.set(r);
  }
  return out;
}

uint64_t differing_lanes(const MachineState &a, const MachineState &b, unsigned r) {
  if (r >= a.vgpr_count || r >= b.vgpr_count)
    return ~uint64_t{0};
  uint64_t mask = 0;
  for (unsigned lane = 0; lane < kWaveLanes; ++lane)
    if (a.vget(r, lane) != b.vget(r, lane))
      mask |= uint64_t{1} << lane;
  return mask;
}

LocSet differing_locs(const MachineState &a, const MachineState &b) {
  LocSet out;
  for (unsigned l = 0; l < kNumLocs; ++l)
    if (a.loc[l] != b.loc[l])
      out.set(l);
  return out;
}

std::vector<std::string> differing_frame_fields(const Frame &a, const Frame &b) {
  std::vector<std::string> out;
  if (a.status_except_scc != b.status_except_scc)
    out.push_back("status_except_scc");
  if (a.mode != b.mode)
    out.push_back("mode");
  // TRAPSTS.EXCP[6:0] are sticky floating-point (and integer divide) exception flags that VALU
  // arithmetic accumulates. LLVM's default floating-point environment gives them no meaning
  // (only strictfp does), so they are neither frame nor output; trap enables (MODE) stay fixed.
  if ((a.trapsts ^ b.trapsts) & ~kTrapstsArithmeticExceptions)
    out.push_back("trapsts");
  if (a.ttmp != b.ttmp)
    out.push_back("ttmp[0:15]");
  if (a.scratch_base != b.scratch_base)
    out.push_back("scratch_base");
  if (a.s102 != b.s102)
    out.push_back("s102");
  if (a.s103 != b.s103)
    out.push_back("s103");
  if (a.wait_counters != b.wait_counters)
    out.push_back("wait_counters");
  if (a.in_trap_handler != b.in_trap_handler)
    out.push_back("in_trap_handler");
  if (a.wave_state != b.wave_state)
    out.push_back("wave_state");
  if (a.sleep_cycles != b.sleep_cycles)
    out.push_back("sleep_cycles");
  return out;
}

} // namespace rocjitsu::isa_diff
