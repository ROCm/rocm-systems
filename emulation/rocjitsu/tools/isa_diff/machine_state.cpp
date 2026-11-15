// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "machine_state.h"

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
  return j;
}

std::optional<MachineState> MachineState::from_json(const Json &j, std::string &error) {
  MachineState s;
  const Json &locs = j["locations"];
  for (unsigned l = 0; l < kNumLocs; ++l) {
    auto v = json_u64(locs[loc_name(l)]);
    if (!v) {
      error = "state is missing location " + loc_name(l);
      return std::nullopt;
    }
    s.set(l, static_cast<uint32_t>(*v));
  }
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
  return s;
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
  if (a.trapsts != b.trapsts)
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
