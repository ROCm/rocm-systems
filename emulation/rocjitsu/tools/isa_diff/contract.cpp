// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "contract.h"

#include <algorithm>

namespace rocjitsu::isa_diff {

const char *contract_status_name(ContractStatus s) {
  switch (s) {
  case ContractStatus::Established:
    return "ESTABLISHED";
  case ContractStatus::Assumed:
    return "ASSUMED";
  case ContractStatus::Unresolved:
    return "UNRESOLVED";
  }
  return "?";
}

std::optional<ContractStatus> parse_contract_status(std::string_view s) {
  for (auto v : {ContractStatus::Established, ContractStatus::Assumed, ContractStatus::Unresolved})
    if (s == contract_status_name(v))
      return v;
  return std::nullopt;
}

std::array<unsigned, kNumLocs>
complete_register_map(const std::vector<std::pair<unsigned, unsigned>> &register_map) {
  std::array<unsigned, kNumLocs> f{};
  for (unsigned l = 0; l < kNumLocs; ++l)
    f[l] = l;
  std::array<bool, kNumLocs> in_domain{}, in_image{};
  std::array<unsigned, kNumLocs> pre{};
  for (const auto &[a, b] : register_map) {
    f[a] = b;
    in_domain[a] = true;
    in_image[b] = true;
    pre[b] = a;
  }
  // The map is a partial injection. Complete it to a permutation: a target that is not
  // itself mapped takes the start of its chain (s4=s8,s8=s9 gives s9 -> s4). The backward
  // walk ends because the chain cannot return to a location outside the domain.
  for (const auto &[a, b] : register_map) {
    if (in_domain[b])
      continue;
    unsigned x = a;
    while (in_image[x])
      x = pre[x];
    f[b] = x;
  }
  return f;
}

std::array<unsigned, kNumLocs> invert_permutation(const std::array<unsigned, kNumLocs> &f) {
  std::array<unsigned, kNumLocs> inv{};
  for (unsigned l = 0; l < kNumLocs; ++l)
    inv[f[l]] = l;
  return inv;
}

std::array<unsigned, kNumLocs> RegionContract::entry_forward() const {
  return complete_register_map(entry_map);
}
std::array<unsigned, kNumLocs> RegionContract::entry_inverse() const {
  return invert_permutation(entry_forward());
}
std::array<unsigned, kNumLocs> RegionContract::exit_forward() const {
  return complete_register_map(exit_map);
}
std::array<unsigned, kNumLocs> RegionContract::exit_inverse() const {
  return invert_permutation(exit_forward());
}

std::optional<std::vector<std::pair<unsigned, unsigned>>> parse_register_map(std::string_view text,
                                                                             std::string &error) {
  std::vector<std::pair<unsigned, unsigned>> out;
  while (!text.empty()) {
    const size_t comma = text.find(',');
    const std::string_view item = text.substr(0, comma);
    const size_t eq = item.find('=');
    if (eq == std::string_view::npos) {
      error = "register map entry '" + std::string(item) + "' is not old=new";
      return std::nullopt;
    }
    auto a = parse_loc(item.substr(0, eq));
    auto b = parse_loc(item.substr(eq + 1));
    if (!a || !b) {
      error = "register map entry '" + std::string(item) + "' names an unknown location";
      return std::nullopt;
    }
    // EXEC/VCC/M0/SCC have aliases and derived views that must stay consistent (handoff 6),
    // so only plain SGPRs can be renamed.
    if (*a >= kNumSgprs || *b >= kNumSgprs) {
      error = "register map entry '" + std::string(item) + "': only s0..s101 can be mapped";
      return std::nullopt;
    }
    for (const auto &[x, y] : out)
      if (x == *a || y == *b) {
        error = "register map maps a location twice: '" + std::string(item) + "'";
        return std::nullopt;
      }
    if (*a != *b)
      out.emplace_back(*a, *b);
    if (comma == std::string_view::npos)
      break;
    text.remove_prefix(comma + 1);
  }
  return out;
}

std::string format_register_map(const std::vector<std::pair<unsigned, unsigned>> &m) {
  std::string s;
  for (const auto &[a, b] : m) {
    if (!s.empty())
      s += ',';
    s += loc_name(a) + "=" + loc_name(b);
  }
  return s;
}

const std::vector<std::string> &frame_field_names() {
  static const std::vector<std::string> names = {
      "status_except_scc", "mode",       "trapsts",     "ttmp[0:15]",
      "scratch_base",      "s102",       "s103",        "wait_counters",
      "in_trap_handler",   "wave_state", "sleep_cycles"};
  return names;
}

Json RegionContract::to_json() const {
  Json j = Json::object();
  j.set("target", target);
  j.set("wave", wave);
  j.set("arithmetic", arithmetic);
  j.set("mapping", mapping);
  j.set("entry_equal", format_locs(entry_equal));
  j.set("canary", format_locs(canary));
  j.set("observable", format_locs(observable));
  j.set("declared_writes_old", format_locs(declared_writes_old));
  j.set("declared_writes_new", format_locs(declared_writes_new));
  j.set("wide_reads", format_locs(wide_reads));
  Json cs = Json::array();
  for (const auto &c : constraints) {
    Json cj = Json::object();
    cj.set("loc", loc_name(c.loc));
    if (c.known_zero)
      cj.set("known_zero", hex32(c.known_zero));
    if (c.known_one)
      cj.set("known_one", hex32(c.known_one));
    if (c.umin)
      cj.set("umin", hex32(*c.umin));
    if (c.umax)
      cj.set("umax", hex32(*c.umax));
    if (c.equal_to)
      cj.set("equal_to", loc_name(*c.equal_to));
    cj.set("source", c.source);
    cs.push(std::move(cj));
  }
  j.set("constraints", std::move(cs));
  j.set("exit_mapping", exit_mapping);
  if (!entry_map.empty())
    j.set("entry_map", format_register_map(entry_map));
  if (!exit_map.empty())
    j.set("exit_map", format_register_map(exit_map));
  j.set("pending_effects", pending_effects);
  j.set("composition", composition);
  Json frame = Json::array();
  for (const auto &f : frame_field_names())
    frame.push(f);
  j.set("frame_preserved", std::move(frame));
  j.set("status", contract_status_name(status));
  Json deps = Json::array();
  for (const auto &d : depends_on)
    deps.push(d);
  j.set("depends_on", std::move(deps));
  Json src = Json::array();
  for (const auto &s : sources) {
    Json sj = Json::object();
    sj.set("what", s.what);
    sj.set("source", s.source);
    src.push(std::move(sj));
  }
  j.set("sources", std::move(src));
  return j;
}

std::optional<RegionContract> RegionContract::from_json(const Json &j, std::string &error) {
  if (!j.is_object()) {
    error = "contract is not an object";
    return std::nullopt;
  }
  RegionContract c;
  c.target = j["target"].as_string();
  c.wave = j["wave"].as_string();
  c.arithmetic = j["arithmetic"].as_string();
  c.mapping = j["mapping"].as_string();
  if (c.target != "gfx950" || c.wave != "wave64") {
    error = "contract target/wave is not gfx950/wave64";
    return std::nullopt;
  }
  auto locs = [&](const char *key, LocSet &out) {
    auto parsed = parse_locs(j[key].as_string());
    if (!parsed) {
      error = std::string("bad location list in '") + key + "'";
      return false;
    }
    out = *parsed;
    return true;
  };
  if (!locs("entry_equal", c.entry_equal) || !locs("canary", c.canary) ||
      !locs("observable", c.observable) || !locs("declared_writes_old", c.declared_writes_old) ||
      !locs("declared_writes_new", c.declared_writes_new) || !locs("wide_reads", c.wide_reads))
    return std::nullopt;
  if ((c.entry_equal & c.canary).any()) {
    error = "a location is both entry_equal and canary";
    return std::nullopt;
  }
  if ((c.entry_equal | c.canary) != all_locs()) {
    error = "every location must be initialised: entry_equal | canary must cover all locations";
    return std::nullopt;
  }
  for (const Json &cj : j["constraints"].items()) {
    Constraint k;
    auto loc = parse_loc(cj["loc"].as_string());
    if (!loc) {
      error = "constraint on unknown location '" + cj["loc"].as_string() + "'";
      return std::nullopt;
    }
    k.loc = *loc;
    auto u32 = [&](const char *key) -> std::optional<uint32_t> {
      if (!cj.has(key))
        return std::nullopt;
      auto v = json_u64(cj[key]);
      return v ? std::optional<uint32_t>(static_cast<uint32_t>(*v)) : std::nullopt;
    };
    k.known_zero = u32("known_zero").value_or(0);
    k.known_one = u32("known_one").value_or(0);
    k.umin = u32("umin");
    k.umax = u32("umax");
    if (cj.has("equal_to")) {
      auto other = parse_loc(cj["equal_to"].as_string());
      if (!other) {
        error = "constraint equal_to names an unknown location";
        return std::nullopt;
      }
      k.equal_to = *other;
    }
    k.source = cj.has("source") ? cj["source"].as_string() : "user_assumption";
    if (!c.entry_equal.test(k.loc) || (k.equal_to && !c.entry_equal.test(*k.equal_to))) {
      error = "constraint on " + loc_name(k.loc) +
              " requires the location to be entry_equal (a canary is never constrained)";
      return std::nullopt;
    }
    c.constraints.push_back(k);
  }
  c.exit_mapping = j["exit_mapping"].as_string();
  for (const char *key : {"entry_map", "exit_map"}) {
    if (!j.has(key))
      continue;
    auto m = parse_register_map(j[key].as_string(), error);
    if (!m)
      return std::nullopt;
    (std::string_view(key) == "entry_map" ? c.entry_map : c.exit_map) = std::move(*m);
  }
  c.pending_effects = j.has("pending_effects") ? j["pending_effects"].as_string() : "NONE";
  c.composition = j.has("composition") ? j["composition"].as_string() : "NOT_CHECKED";
  auto status = parse_contract_status(j["status"].as_string());
  if (!status) {
    error = "bad contract status";
    return std::nullopt;
  }
  c.status = *status;
  for (const Json &d : j["depends_on"].items())
    c.depends_on.push_back(d.as_string());
  for (const Json &s : j["sources"].items())
    c.sources.push_back({s["what"].as_string(), s["source"].as_string()});
  return c;
}

} // namespace rocjitsu::isa_diff
