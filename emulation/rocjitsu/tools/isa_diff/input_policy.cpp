// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "input_policy.h"

namespace rocjitsu::isa_diff {
namespace {

constexpr uint32_t kBoundary32[] = {
    0x00000000u, 0x00000001u, 0x00000002u, 0xffffffffu, 0xfffffffeu, 0x7fffffffu,
    0x80000000u, 0x80000001u, 0xaaaaaaaau, 0x55555555u, 0x0000001fu, 0x00000020u,
    0x00000021u, 0x0000003fu, 0x00000040u, 0x0000ffffu, 0x00010000u, 0x00008000u,
    0x000000ffu, 0x00000100u, 0xffff0000u, 0x0000007fu, 0x00000080u, 0xffffff80u,
};

constexpr uint64_t kBoundary64[] = {
    0x0000000000000000ull, 0x0000000000000001ull, 0xffffffffffffffffull, 0x7fffffffffffffffull,
    0x8000000000000000ull, 0x00000000ffffffffull, 0x0000000100000000ull, 0xffffffff00000000ull,
    0x8000000000000001ull, 0x000000000000003full, 0x0000000000000040ull, 0xaaaaaaaaaaaaaaaaull,
};

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

Sample BaselinePolicy::propose(const RegionFeatures &, const RegionContract &contract,
                               uint64_t case_index) {
  CaseRng rng(splitmix64(seed_ ^ splitmix64(case_index)));
  auto pick = [&](uint64_t n) { return static_cast<size_t>(rng() % n); };
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
    // A pair operand whose low half is here draws a 64-bit boundary sometimes.
    const unsigned hi = l + 1;
    if (contract.wide_reads.test(l) && hi < kNumLocs && contract.entry_equal.test(hi) &&
        pick(3) == 0) {
      const uint64_t v = pick(2) ? kBoundary64[pick(std::size(kBoundary64))]
                                 : (uint64_t{1} << pick(64)) ^ (pick(2) ? ~uint64_t{0} : 0);
      s.value[l] = static_cast<uint32_t>(v);
      s.value[hi] = static_cast<uint32_t>(v >> 32);
      s.events.push_back(loc_name(l) + ":pair64");
      assigned.push_back(l);
      assigned.push_back(hi);
      ++l; // high half done
      continue;
    }
    const size_t choice = pick(20);
    if (choice < 7) {
      s.value[l] = kBoundary32[pick(std::size(kBoundary32))];
      s.events.push_back(loc_name(l) + ":boundary");
    } else if (choice < 9) {
      const uint32_t bit = 1u << pick(32);
      s.value[l] = pick(2) ? bit : ~bit;
      s.events.push_back(loc_name(l) + ":single_bit");
    } else if (choice < 12 && !assigned.empty()) {
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
