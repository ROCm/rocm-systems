// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "input_policy.h"

#include "boundary.h"

namespace rocjitsu::isa_diff {
namespace {

/// SplitMix64 stream: deterministic, portable, and free to seed once per case
/// (std::mt19937_64 seeding dominated the per-case cost of small regions).
class CaseRng {
public:
  explicit CaseRng(uint64_t seed) : state_(seed) {}
  uint64_t operator()() {
    state_ += 0x9e3779b97f4a7c15ull;
    uint64_t z = state_;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
  }

private:
  uint64_t state_;
};

} // namespace

uint64_t splitmix64(uint64_t x) {
  x += 0x9e3779b97f4a7c15ull;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
  return x ^ (x >> 31);
}

uint32_t canary_value(uint64_t seed, uint64_t case_index, unsigned loc, bool new_side) {
  const uint64_t h =
      splitmix64(seed ^ splitmix64(case_index * 0x100000001b3ull + loc) ^ (new_side ? 0x5a5a : 0));
  uint32_t v = static_cast<uint32_t>(h >> 17);
  if (loc == kLocScc)
    return new_side ? 1u : 0u; // dead SCC differs by side on every case
  // Keep the two sides' canaries distinct so any use of one is visible.
  if (new_side)
    v ^= 0xc3a5c3a5u;
  return v;
}

RegionFeatures make_region_features(const std::vector<RegionInst> &old_insts,
                                    const std::vector<RegionInst> &new_insts,
                                    const std::array<unsigned, kNumLocs> *new_to_old,
                                    const Target &target) {
  RegionFeatures f;
  for (const auto &i : old_insts)
    f.old_text.push_back(i.text);
  for (const auto &i : new_insts)
    f.new_text.push_back(i.text);
  f.roles = infer_roles(old_insts, new_insts, new_to_old, target);
  for (unsigned l = 0; l < kNumLocs; ++l) {
    for (const auto &v : role_catalog(f.roles.roles[l]))
      f.role_values[l].push_back(v.value);
    for (const auto &rel : f.roles.const_relations)
      if (rel.loc == l)
        for (const auto &v : const_relative_values(rel.constant))
          f.role_values[l].push_back(v.value);
  }
  return f;
}

void History::record(const Feedback &f) {
  ++cases;
  accepted += f.accepted ? 1 : 0;
  valid += f.category == "ok" || f.mismatch ? 1 : 0;
  mismatches += f.mismatch ? 1 : 0;
  novel += f.novel ? 1 : 0;
  ++categories[f.category];
  if (recent.size() == kWindow)
    recent.erase(recent.begin());
  recent.push_back(f);
}

std::optional<std::string> ConstraintChecker::validate(const Sample &sample,
                                                       const RegionContract &contract) {
  return validate_sample(sample, contract);
}

Sample BaselinePolicy::propose(const RegionFeatures &features, const RegionContract &contract,
                               const History & /*history: fixed policy*/, uint64_t case_index) {
  CaseRng rng(splitmix64(seed_ ^ splitmix64(case_index)));
  auto pick = [&](uint64_t n) { return static_cast<size_t>(rng() % n); };
  const auto &c32 = catalog32();
  const auto &c64 = catalog64();
  Sample s;
  std::vector<unsigned> assigned;
  for (unsigned l = 0; l < kNumLocs; ++l) {
    if (!contract.entry_equal.test(l))
      continue;
    if (l == kLocScc) {
      const bool flip = pick(4) == 0;
      s.value[l] = static_cast<uint32_t>((case_index & 1u) ^ (flip ? 1u : 0u));
      continue;
    }
    // A pair operand whose low half is here draws a 64-bit catalog value sometimes.
    const unsigned hi = l + 1;
    if (contract.wide_reads.test(l) && hi < kNumLocs && contract.entry_equal.test(hi) &&
        pick(3) == 0) {
      const NamedValue64 &v = c64[pick(c64.size())];
      s.value[l] = static_cast<uint32_t>(v.value);
      s.value[hi] = static_cast<uint32_t>(v.value >> 32);
      s.events.push_back(loc_name(l) + "=" + v.name);
      assigned.push_back(l);
      assigned.push_back(hi);
      ++l; // high half done
      continue;
    }
    const auto &roles = features.role_values[l];
    const size_t choice = pick(20);
    if (choice < 4 && !roles.empty()) {
      s.value[l] = roles[pick(roles.size())];
      s.events.push_back(loc_name(l) + ":role");
    } else if (choice < 9) {
      const NamedValue &v = c32[pick(c32.size())];
      s.value[l] = v.value;
      s.events.push_back(loc_name(l) + "=" + v.name);
    } else if (choice < 11) {
      const uint32_t bit = 1u << pick(32);
      s.value[l] = pick(2) ? bit : ~bit;
      s.events.push_back(loc_name(l) + ":single_bit");
    } else if (choice < 14 && !assigned.empty()) {
      const unsigned other = assigned[pick(assigned.size())];
      const bool complement = pick(2) != 0;
      s.value[l] = complement ? ~s.value[other] : s.value[other];
      s.events.push_back(loc_name(l) + (complement ? ":complement_of:" : ":equal_to:") +
                         loc_name(other));
    } else {
      s.value[l] = static_cast<uint32_t>(rng());
    }
    assigned.push_back(l);
  }
  // VALU regions: lanes of every VGPR, and a full EXEC on half the cases, so most
  // cases compare all 64 results (inactive lanes are covered by the other half).
  fill_vgpr_lanes(s, contract.vgpr_count, contract.vwide_reads, seed_, case_index);
  if (contract.vgpr_count && contract.entry_equal.test(kLocExecLo) &&
      (contract.wave == "wave32" || contract.entry_equal.test(kLocExecHi)) && pick(2) == 0) {
    s.value[kLocExecLo] = s.value[kLocExecHi] = 0xffffffffu;
    s.events.push_back("exec:full");
  }
  return s;
}

void fill_vgpr_lanes(Sample &s, unsigned vgpr_count, const VSet &wide, uint64_t seed,
                     uint64_t index) {
  s.vvalue.assign(size_t{vgpr_count} * kWaveLanes, 0);
  if (!vgpr_count)
    return;
  const auto &c32 = catalog32();
  const auto &c64 = catalog64();
  for (unsigned r = 0; r < vgpr_count; ++r) {
    CaseRng reg(splitmix64(seed ^ splitmix64(index * 0x9e3779b97f4a7c15ull + r) ^ 0x7667707232ull));
    const bool pair = wide.test(r) && r + 1 < vgpr_count;
    const bool uniform = reg() % 10 == 0;
    const uint64_t uniform_draw = reg();
    for (unsigned lane = 0; lane < kWaveLanes; ++lane) {
      CaseRng rng(uniform ? uniform_draw : splitmix64(reg() ^ lane));
      uint32_t &lo = s.vvalue[size_t{r} * kWaveLanes + lane];
      if (pair && rng() % 3 == 0) {
        const uint64_t v = c64[rng() % c64.size()].value;
        lo = static_cast<uint32_t>(v);
        s.vvalue[size_t{r + 1} * kWaveLanes + lane] = static_cast<uint32_t>(v >> 32);
        continue;
      }
      const uint64_t choice = rng() % 20;
      if (choice < 9) {
        lo = c32[rng() % c32.size()].value;
      } else if (choice < 11) {
        const uint32_t bit = 1u << (rng() % 32);
        lo = rng() % 2 ? bit : ~bit;
      } else if (choice < 13 && r > 0) {
        const uint32_t other = s.vvalue[size_t{rng() % r} * kWaveLanes + lane];
        lo = rng() % 2 ? other : ~other;
      } else {
        lo = static_cast<uint32_t>(rng());
      }
    }
    if (pair)
      ++r; // the high half was drawn with the low half when a 64-bit value was chosen
  }
}

uint32_t initial_memory_word(uint64_t seed, uint64_t word) {
  const uint64_t h = splitmix64(seed ^ splitmix64(word ^ 0x6d656d6f7279ull));
  if (h & 1u) {
    const auto &c32 = catalog32();
    return c32[(h >> 1) % c32.size()].value;
  }
  return static_cast<uint32_t>(h >> 32);
}

void project_onto_constraints(Sample &sample, const RegionContract &contract, uint64_t entropy) {
  CaseRng rng(splitmix64(entropy));
  for (const Constraint &c : contract.constraints) {
    uint32_t &v = sample.value[c.loc];
    if (c.equal_to)
      v = sample.value[*c.equal_to];
    v = (v & ~c.known_zero) | c.known_one;
    if (c.umin && c.umax && *c.umin <= *c.umax && (v < *c.umin || v > *c.umax) && (rng() & 1u)) {
      const uint64_t span = uint64_t{*c.umax} - *c.umin + 1;
      v = static_cast<uint32_t>(*c.umin + rng() % span);
      v = (v & ~c.known_zero) | c.known_one;
    }
  }
}

std::optional<std::string> validate_sample(const Sample &sample, const RegionContract &contract) {
  if (contract.exec_nonzero &&
      (sample.value[kLocExecLo] | (contract.wave == "wave32" ? 0u : sample.value[kLocExecHi])) == 0)
    return "EXEC is zero (exec_nonzero)";
  for (const Constraint &c : contract.constraints) {
    const uint32_t v = sample.value[c.loc];
    const std::string name = loc_name(c.loc);
    if (c.known_zero & c.known_one)
      return "contradictory known bits on " + name;
    if (v & c.known_zero)
      return "known_zero violated on " + name;
    if ((v & c.known_one) != c.known_one)
      return "known_one violated on " + name;
    if (c.umin && v < *c.umin)
      return "below umin on " + name;
    if (c.umax && v > *c.umax)
      return "above umax on " + name;
    if (c.equal_to && v != sample.value[*c.equal_to])
      return "equality violated on " + name;
  }
  return std::nullopt;
}

} // namespace rocjitsu::isa_diff
