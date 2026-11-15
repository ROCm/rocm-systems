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
                                    const std::vector<RegionInst> &new_insts) {
  RegionFeatures f;
  for (const auto &i : old_insts)
    f.old_text.push_back(i.text);
  for (const auto &i : new_insts)
    f.new_text.push_back(i.text);
  f.roles = infer_roles(old_insts, new_insts);
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

Sample BaselinePolicy::propose(const RegionFeatures &features, const RegionContract &contract,
                               uint64_t case_index) {
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
  return s;
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
