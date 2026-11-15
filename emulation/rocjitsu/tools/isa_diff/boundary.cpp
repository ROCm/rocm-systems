// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "boundary.h"

#include "json.h"

#include <algorithm>
#include <unordered_set>

namespace rocjitsu::isa_diff {
namespace {

template <typename T, typename V> void add_unique(std::vector<T> &out, V value, std::string name) {
  for (const auto &e : out)
    if (e.value == value)
      return;
  out.push_back({value, std::move(name)});
}

std::vector<NamedValue> build_catalog32() {
  std::vector<NamedValue> c;
  // Integer limits and neighbours.
  add_unique(c, 0x00000000u, "int.zero");
  add_unique(c, 0x00000001u, "int.one");
  add_unique(c, 0x00000002u, "int.two");
  add_unique(c, 0x00000003u, "int.three");
  add_unique(c, 0xffffffffu, "int.minus_one");
  add_unique(c, 0xfffffffeu, "int.minus_two");
  add_unique(c, 0x7fffffffu, "int.smax");
  add_unique(c, 0x80000000u, "int.smin");
  add_unique(c, 0x80000001u, "int.smin_plus_one");
  add_unique(c, 0x7ffffffeu, "int.smax_minus_one");
  // Byte and half-word edges (sign extension, packing, truncation).
  for (auto [v, n] :
       std::initializer_list<std::pair<uint32_t, const char *>>{{0x7f, "edge.s8_max"},
                                                                {0x80, "edge.s8_min"},
                                                                {0xff, "edge.u8_max"},
                                                                {0x100, "edge.u8_wrap"},
                                                                {0x7fff, "edge.s16_max"},
                                                                {0x8000, "edge.s16_min"},
                                                                {0xffff, "edge.u16_max"},
                                                                {0x10000, "edge.u16_wrap"},
                                                                {0xffffff80, "edge.s8_min_sext"},
                                                                {0xffff8000, "edge.s16_min_sext"},
                                                                {0xffff0000, "edge.hi16_ones"},
                                                                {0x00ff00ff, "edge.bytes_lo"},
                                                                {0xff00ff00, "edge.bytes_hi"}})
    add_unique(c, v, n);
  // Small counts that are error-prone as shift amounts or bit indexes.
  for (uint32_t v : {31u, 32u, 33u, 63u, 64u})
    add_unique(c, v, "small." + std::to_string(v));
  // Bit patterns.
  for (auto [v, n] :
       std::initializer_list<std::pair<uint32_t, const char *>>{{0x55555555, "bits.alt01"},
                                                                {0xaaaaaaaa, "bits.alt10"},
                                                                {0x33333333, "bits.pair01"},
                                                                {0xcccccccc, "bits.pair10"},
                                                                {0x0f0f0f0f, "bits.nib01"},
                                                                {0xf0f0f0f0, "bits.nib10"},
                                                                {0x01010101, "bits.byte_lsb"},
                                                                {0x80808080, "bits.byte_msb"}})
    add_unique(c, v, n);
  for (unsigned k : {0u, 1u, 7u, 8u, 15u, 16u, 23u, 24u, 30u, 31u}) {
    add_unique(c, 1u << k, "bit.only" + std::to_string(k));
    add_unique(c, ~(1u << k), "bit.clear" + std::to_string(k));
  }
  // f32 special encodings.
  for (auto [v, n] : std::initializer_list<std::pair<uint32_t, const char *>>{
           {0x80000000, "f32.neg_zero"},
           {0x7f800000, "f32.pos_inf"},
           {0xff800000, "f32.neg_inf"},
           {0x7fc00000, "f32.qnan"},
           {0xffc00000, "f32.neg_qnan"},
           {0x7fa00000, "f32.snan"},
           {0x7f800001, "f32.snan_min_payload"},
           {0x7fffffff, "f32.qnan_max_payload"},
           {0x00000001, "f32.min_denorm"},
           {0x007fffff, "f32.max_denorm"},
           {0x80000001, "f32.neg_min_denorm"},
           {0x00800000, "f32.min_normal"},
           {0x7f7fffff, "f32.max_normal"},
           {0xff7fffff, "f32.neg_max_normal"},
           {0x3f800000, "f32.one"},
           {0xbf800000, "f32.neg_one"},
           {0x3f000000, "f32.half"},
           {0x4f000000, "f32.two_pow_31"},
           {0x4f800000, "f32.two_pow_32"},
           {0xcf000000, "f32.neg_two_pow_31"}})
    add_unique(c, v, n);
  // f16 and bf16 encodings, in the low half and packed.
  for (auto [v, n] :
       std::initializer_list<std::pair<uint32_t, const char *>>{{0x00007e00, "f16.qnan"},
                                                                {0x00007d00, "f16.snan"},
                                                                {0x00007c00, "f16.pos_inf"},
                                                                {0x0000fc00, "f16.neg_inf"},
                                                                {0x00007bff, "f16.max_normal"},
                                                                {0x000003ff, "f16.max_denorm"},
                                                                {0x00000400, "f16.min_normal"},
                                                                {0x00003c00, "f16.one"},
                                                                {0x7e007e00, "f16x2.qnan"},
                                                                {0x7c00fc00, "f16x2.pos_neg_inf"},
                                                                {0x80008000, "f16x2.neg_zero"},
                                                                {0x03ff03ff, "f16x2.max_denorm"},
                                                                {0x7e003c00, "f16x2.qnan_one"},
                                                                {0x00007fc0, "bf16.qnan"},
                                                                {0x00007f80, "bf16.pos_inf"},
                                                                {0x0000ff80, "bf16.neg_inf"},
                                                                {0x0000007f, "bf16.max_denorm"},
                                                                {0x7fc07fc0, "bf16x2.qnan"},
                                                                {0x7f80ff80, "bf16x2.pos_neg_inf"}})
    add_unique(c, v, n);
  return c;
}

std::vector<NamedValue64> build_catalog64() {
  std::vector<NamedValue64> c;
  for (auto [v, n] : std::initializer_list<std::pair<uint64_t, const char *>>{
           {0x0000000000000000ull, "int64.zero"},
           {0x0000000000000001ull, "int64.one"},
           {0xffffffffffffffffull, "int64.minus_one"},
           {0x7fffffffffffffffull, "int64.smax"},
           {0x8000000000000000ull, "int64.smin"},
           {0x8000000000000001ull, "int64.smin_plus_one"},
           {0x00000000ffffffffull, "int64.u32_max"},
           {0x0000000100000000ull, "int64.two_pow_32"},
           {0x0000000100000001ull, "int64.two_pow_32_plus_one"},
           {0x00000001ffffffffull, "int64.carry_into_hi"},
           {0xffffffff00000000ull, "int64.hi_ones"},
           {0x0000000080000000ull, "int64.lo_sign"},
           {0xffffffff7fffffffull, "int64.lo_smax_hi_ones"},
           {0x7ff8000000000000ull, "f64.qnan"},
           {0xfff8000000000000ull, "f64.neg_qnan"},
           {0x7ff4000000000000ull, "f64.snan"},
           {0x7ff0000000000001ull, "f64.snan_min_payload"},
           {0x7ff0000000000000ull, "f64.pos_inf"},
           {0xfff0000000000000ull, "f64.neg_inf"},
           {0x000fffffffffffffull, "f64.max_denorm"},
           {0x0010000000000000ull, "f64.min_normal"},
           {0x7fefffffffffffffull, "f64.max_normal"},
           {0x3ff0000000000000ull, "f64.one"},
           {0xbff0000000000000ull, "f64.neg_one"},
           {0x5555555555555555ull, "bits64.alt01"},
           {0xaaaaaaaaaaaaaaaaull, "bits64.alt10"},
           {0x0000000080000000ull, "exec.lane31"},
           {0x8000000000000000ull, "exec.lane63"},
           {0x0000000100000000ull, "exec.lane32"},
           {0x00000000ffffffffull, "exec.low32"},
           {0xffffffff00000000ull, "exec.high32"}})
    add_unique(c, v, n);
  for (unsigned k : {0u, 31u, 32u, 63u}) {
    add_unique(c, uint64_t{1} << k, "bit64.only" + std::to_string(k));
    add_unique(c, ~(uint64_t{1} << k), "bit64.clear" + std::to_string(k));
  }
  return c;
}

std::vector<NamedValue> shift_values() {
  std::vector<NamedValue> c;
  for (uint32_t v : {0u,  1u,  2u,  7u,  8u,  15u, 16u, 17u, 30u,  31u,  32u,  33u,
                     47u, 48u, 62u, 63u, 64u, 65u, 95u, 96u, 127u, 128u, 255u, 256u})
    add_unique(c, v, "shift." + std::to_string(v));
  // High bits set: the ISA reads only the low 5 or 6 bits of the amount.
  for (auto [v, n] : std::initializer_list<std::pair<uint32_t, const char *>>{
           {0xffffffe0, "shift.hi_garbage_low5_0"},
           {0xffffffc0, "shift.hi_garbage_low6_0"},
           {0xffffffdf, "shift.hi_garbage_low5_31"},
           {0xffffffbf, "shift.hi_garbage_low6_63"},
           {0x80000020, "shift.sign_plus_32"},
           {0x7fffffff, "shift.smax"},
           {0x80000000, "shift.smin"},
           {0xffffffff, "shift.all_ones"}})
    add_unique(c, v, n);
  return c;
}

std::vector<NamedValue> bfe_values() {
  // S_BFE control: offset in [5:0] (u32 uses [4:0]), width in [22:16].
  std::vector<NamedValue> c;
  for (uint32_t off : {0u, 1u, 8u, 15u, 16u, 31u, 32u, 33u, 63u, 64u})
    for (uint32_t w : {0u, 1u, 8u, 15u, 16u, 31u, 32u, 33u, 63u, 64u, 127u})
      add_unique(c, (w << 16) | off, "bfe.off" + std::to_string(off) + "_w" + std::to_string(w));
  for (uint32_t off : {0u, 16u, 31u})
    for (uint32_t w : {1u, 16u, 32u})
      add_unique(c, 0xff800000u | (w << 16) | off,
                 "bfe.garbage_off" + std::to_string(off) + "_w" + std::to_string(w));
  return c;
}

std::vector<NamedValue> sext_values() {
  std::vector<NamedValue> c;
  for (auto [v, n] : std::initializer_list<std::pair<uint32_t, const char *>>{
           {0x7f, "sext.s8_max"},
           {0x80, "sext.s8_min"},
           {0xff, "sext.s8_minus_one"},
           {0x100, "sext.bit8_only"},
           {0x17f, "sext.garbage_s8_max"},
           {0x180, "sext.garbage_s8_min"},
           {0x7fff, "sext.s16_max"},
           {0x8000, "sext.s16_min"},
           {0xffff, "sext.s16_minus_one"},
           {0x10000, "sext.bit16_only"},
           {0x17fff, "sext.garbage_s16_max"},
           {0x18000, "sext.garbage_s16_min"},
           {0xffffff7f, "sext.hi_ones_s8_max"},
           {0xffff7fff, "sext.hi_ones_s16_max"},
           {0x12345680, "sext.garbage_s8_neg"},
           {0x12348000, "sext.garbage_s16_neg"}})
    add_unique(c, v, n);
  return c;
}

std::vector<PairValue> build_pairs() {
  std::vector<PairValue> p;
  auto add = [&](uint32_t a, uint32_t b, std::string n) {
    for (const auto &e : p)
      if (e.a == a && e.b == b)
        return;
    p.push_back({a, b, std::move(n)});
  };
  const uint32_t x = 0x12345678u;
  for (auto [a, b, n] : std::initializer_list<std::tuple<uint32_t, uint32_t, const char *>>{
           {0xffffffff, 1, "pair.carry_umax_plus_one"},
           {1, 0xffffffff, "pair.carry_one_plus_umax"},
           {0xffffffff, 0xffffffff, "pair.carry_umax_umax"},
           {0x80000000, 0x80000000, "pair.carry_smin_smin"},
           {0x7fffffff, 1, "pair.ovf_smax_plus_one"},
           {1, 0x7fffffff, "pair.ovf_one_plus_smax"},
           {0x7fffffff, 0x7fffffff, "pair.ovf_smax_smax"},
           {0x80000000, 0xffffffff, "pair.ovf_smin_minus_one"},
           {0xffffffff, 0x80000000, "pair.ovf_minus_one_smin"},
           {0x80000000, 1, "pair.ovf_smin_one"},
           {0x7fffffff, 0x80000000, "pair.sign_smax_smin"},
           {0x80000000, 0x7fffffff, "pair.sign_smin_smax"},
           {0, 1, "pair.borrow_zero_one"},
           {1, 0, "pair.borrow_one_zero"},
           {0, 0xffffffff, "pair.borrow_zero_umax"},
           {0xffffffff, 0, "pair.borrow_umax_zero"},
           {0, 0x80000000, "pair.borrow_zero_smin"},
           {0x80000000, 0, "pair.borrow_smin_zero"},
           {x, x + 1, "pair.adjacent_lt"},
           {x + 1, x, "pair.adjacent_gt"},
           {0x10000, 0x10000, "pair.mul_wrap_2pow32"},
           {0xffff, 0x10001, "pair.mul_umax"},
           {0x10001, 0x10001, "pair.mul_hi_one"},
           {0, 0, "pair.equal_zero"},
           {x, x, "pair.equal_pattern"},
           {x, ~x, "pair.complement_pattern"},
           {x, 0u - x, "pair.negation_pattern"},
           {0x7fc00000, 0x7fc00000, "pair.nan_qnan_qnan"},
           {0x7fc00000, 0x7f800000, "pair.nan_qnan_inf"},
           {0x80000000, 0, "pair.signed_zeros"},
           {0x7f800000, 0xff800000, "pair.inf_neg_inf"}})
    add(a, b, n);
  static constexpr uint32_t kCore[] = {0,          1,          2,          0x7fffffff,
                                       0x80000000, 0x80000001, 0xfffffffe, 0xffffffff,
                                       0xffff,     0x10000,    0x7fc00000, 0x55555555};
  for (uint32_t a : kCore)
    for (uint32_t b : kCore)
      add(a, b, "cross." + hex32(a) + "_" + hex32(b));
  return p;
}

std::vector<PairValue64> build_pairs64() {
  std::vector<PairValue64> p;
  auto add = [&](uint64_t a, uint64_t b, std::string n) {
    for (const auto &e : p)
      if (e.a == a && e.b == b)
        return;
    p.push_back({a, b, std::move(n)});
  };
  add(0x0000000100000000ull, 0x0000000100000001ull, "pair64.equal_hi_differ_lo");
  add(0x0000000100000001ull, 0x0000000200000001ull, "pair64.differ_hi_equal_lo");
  add(0x00000000ffffffffull, 0x0000000100000000ull, "pair64.half_crossing");
  add(0xffffffff00000000ull, 0x00000000ffffffffull, "pair64.complementary_halves");
  add(0x8000000000000000ull, 0x7fffffffffffffffull, "pair64.sign_smin_smax");
  static constexpr uint64_t kCore[] = {0,
                                       1,
                                       ~uint64_t{0},
                                       0x7fffffffffffffffull,
                                       0x8000000000000000ull,
                                       0x00000000ffffffffull,
                                       0x0000000100000000ull,
                                       0x7ff8000000000000ull};
  for (uint64_t a : kCore)
    for (uint64_t b : kCore)
      add(a, b, "cross64." + hex64(a) + "_" + hex64(b));
  return p;
}

struct Hash128 {
  uint64_t a, b;
  bool operator==(const Hash128 &o) const { return a == o.a && b == o.b; }
};
struct Hash128Hasher {
  size_t operator()(const Hash128 &h) const { return static_cast<size_t>(h.a ^ (h.b * 31)); }
};

} // namespace

const std::vector<NamedValue> &catalog32() {
  static const std::vector<NamedValue> c = build_catalog32();
  return c;
}

const std::vector<NamedValue64> &catalog64() {
  static const std::vector<NamedValue64> c = build_catalog64();
  return c;
}

std::vector<NamedValue> role_catalog(uint32_t roles) {
  std::vector<NamedValue> out;
  auto merge = [&](const std::vector<NamedValue> &values) {
    for (const auto &v : values)
      add_unique(out, v.value, v.name);
  };
  if (roles & (kRoleShiftAmount | kRoleBitIndex))
    merge(shift_values());
  if (roles & kRoleBfeControl) {
    merge(bfe_values());
    merge(shift_values());
  }
  if (roles & kRoleSextInput)
    merge(sext_values());
  return out;
}

std::vector<NamedValue> const_relative_values(uint32_t c) {
  std::vector<NamedValue> out;
  const std::string tag = "constrel." + hex32(c) + ".";
  for (auto [v, n] : std::initializer_list<std::pair<uint32_t, const char *>>{
           {c, "c"},
           {c - 1, "c_minus_1"},
           {c + 1, "c_plus_1"},
           {c - 2, "c_minus_2"},
           {c + 2, "c_plus_2"},
           {0u - c, "neg_c"},
           {0u - c - 1, "neg_c_minus_1"},
           {0u - c + 1, "neg_c_plus_1"},
           {~c, "not_c"},
           {c ^ 0x80000000u, "c_sign_flipped"},
           {0x7fffffffu - c, "smax_minus_c"},
           {0x7fffffffu - c + 1, "smax_minus_c_plus_1"},
           {0x80000000u - c, "smin_minus_c"},
           {0x80000000u - c - 1, "smin_minus_c_minus_1"}})
    add_unique(out, v, tag + n);
  return out;
}

const std::vector<PairValue> &pair_catalog() {
  static const std::vector<PairValue> p = build_pairs();
  return p;
}

const std::vector<PairValue64> &pair_catalog64() {
  static const std::vector<PairValue64> p = build_pairs64();
  return p;
}

std::string value_family(const std::string &name) {
  const size_t dot = name.find('.');
  return dot == std::string::npos ? name : name.substr(0, dot);
}

void clamp_to_constraints(std::array<uint32_t, kNumLocs> &values, const RegionContract &contract) {
  for (const Constraint &c : contract.constraints) {
    uint32_t v = values[c.loc];
    if (c.equal_to)
      v = values[*c.equal_to];
    v = (v & ~c.known_zero) | c.known_one;
    if (c.umin && v < *c.umin)
      v = *c.umin;
    if (c.umax && v > *c.umax)
      v = *c.umax;
    values[c.loc] = (v & ~c.known_zero) | c.known_one;
  }
}

Sample BoundarySweep::materialize(const SweepCase &c) const {
  Sample s;
  s.value = bases[c.base];
  for (const auto &[loc, v] : c.overrides)
    s.value[loc] = v & loc_value_mask(loc);
  if (c.scc >= 0)
    s.value[kLocScc] = static_cast<uint32_t>(c.scc);
  if (exec_entry && c.exec >= 0)
    s.value[kLocExecLo] = s.value[kLocExecHi] = c.exec == 0 ? 0xffffffffu : 0x55555555u;
  fill_vgpr_lanes(s, vgpr_count, vwide, seed ^ 0x5357454550ull, c.fill);
  const auto &c64 = catalog64();
  for (const VOverride &o : c.voverrides) {
    for (unsigned lane = 0; lane < kWaveLanes; ++lane) {
      const uint64_t idx = uint64_t{o.chunk} * kWaveLanes + lane;
      auto put = [&](unsigned r, uint32_t v) {
        if (r < vgpr_count)
          s.vvalue[size_t{r} * kWaveLanes + lane] = v;
      };
      switch (o.kind) {
      case VOverride::Chunk32:
        put(o.a, vcat[idx % vcat.size()]);
        break;
      case VOverride::Chunk64: {
        const uint64_t v = c64[idx % c64.size()].value;
        put(o.a, static_cast<uint32_t>(v));
        put(o.a + 1u, static_cast<uint32_t>(v >> 32));
        break;
      }
      case VOverride::Cross:
        if (o.wide) {
          const uint64_t x = c64[idx % c64.size()].value,
                         y = c64[(idx / c64.size()) % c64.size()].value;
          put(o.a, static_cast<uint32_t>(x));
          put(o.a + 1u, static_cast<uint32_t>(x >> 32));
          put(o.b, static_cast<uint32_t>(y));
          put(o.b + 1u, static_cast<uint32_t>(y >> 32));
        } else {
          put(o.a, vcat[idx % vcat.size()]);
          put(o.b, vcat[(idx / vcat.size()) % vcat.size()]);
        }
        break;
      case VOverride::Values: {
        const auto &list = value_lists[o.list];
        put(o.a, list[idx % list.size()]);
        break;
      }
      }
    }
  }
  s.events.push_back(c.label);
  return s;
}

BoundarySweep build_boundary_sweep(const RegionRoles &roles, const RegionContract &contract,
                                   const SweepOptions &options) {
  BoundarySweep sweep;
  const LocSet entry = contract.entry_equal;
  sweep.vgpr_count = contract.vgpr_count;
  sweep.vwide = contract.vwide_reads;
  sweep.seed = options.seed;
  sweep.exec_entry = entry.test(kLocExecLo) && entry.test(kLocExecHi);
  for (const NamedValue &v : catalog32())
    if (std::ranges::find(sweep.vcat, v.value) == sweep.vcat.end())
      sweep.vcat.push_back(v.value);
  // Bases: all zeros (SCC 0), all ones (SCC 1), seeded random.
  for (unsigned l = 0; l < kNumLocs; ++l) {
    sweep.bases[0][l] = 0;
    sweep.bases[1][l] = loc_value_mask(l);
    sweep.bases[2][l] = static_cast<uint32_t>(splitmix64(options.seed ^ 0x6a09e667f3bcc908ull ^
                                                         (uint64_t{l} << 32))) &
                        loc_value_mask(l);
  }
  for (auto &base : sweep.bases)
    clamp_to_constraints(base, contract);
  static constexpr const char *kBaseName[] = {"zeros", "ones", "random"};

  const bool scc_input = entry.test(kLocScc);
  auto scc_variants = [&](uint8_t base) -> std::vector<int8_t> {
    if (!scc_input)
      return {-1};
    if (base == 2)
      return {0, 1};
    return {static_cast<int8_t>(base)};
  };

  std::unordered_set<Hash128, Hash128Hasher> seen;
  std::vector<VOverride> pending_v; // vector part of the next emitted case
  int8_t pending_exec = -1;
  auto emit = [&](uint8_t base, int8_t scc, std::vector<std::pair<unsigned, uint32_t>> overrides,
                  std::string label, std::string family) {
    ++sweep.generated;
    if (sweep.cases.size() >= options.max_cases) {
      sweep.truncated = true;
      return;
    }
    SweepCase c{base, scc, std::move(overrides), {}, -1, 0, std::move(label), std::move(family)};
    c.voverrides = std::move(pending_v);
    c.exec = pending_exec;
    c.fill = static_cast<uint32_t>(sweep.generated);
    pending_v.clear();
    pending_exec = -1;
    const Sample s = sweep.materialize(c);
    Hash128 h{0xcbf29ce484222325ull, 0x84222325cbf29ce4ull};
    for (unsigned l = 0; l < kNumLocs; ++l) {
      if (!entry.test(l))
        continue;
      h.a = (h.a ^ s.value[l]) * 0x100000001b3ull;
      h.b = splitmix64(h.b ^ (uint64_t{s.value[l]} << 8) ^ l);
    }
    for (unsigned r = 0; r < sweep.vgpr_count; ++r) {
      if (!contract.ventry_equal.test(r))
        continue;
      for (unsigned lane = 0; lane < kWaveLanes; ++lane) {
        const uint32_t v = s.vvalue[size_t{r} * kWaveLanes + lane];
        h.a = (h.a ^ v) * 0x100000001b3ull;
        h.b = splitmix64(h.b ^ (uint64_t{v} << 8) ^ (uint64_t{r} << 40) ^ lane);
      }
    }
    if (!seen.insert(h).second) {
      ++sweep.duplicates;
      return;
    }
    if (scc_input)
      c.label += " scc=" + std::to_string(s.value[kLocScc]);
    sweep.cases.push_back(std::move(c));
  };
  auto for_bases = [&](std::initializer_list<uint8_t> bases,
                       const std::vector<std::pair<unsigned, uint32_t>> &overrides,
                       const std::string &what, const std::string &family) {
    for (uint8_t b : bases)
      for (int8_t scc : scc_variants(b))
        emit(b, scc, overrides, what + " base=" + kBaseName[b], family);
  };

  // Uniform cases cover every entry register. The per-register phases target the
  // registers the region reads; a register it only carries through is still
  // initialised from the bases and compared, but its value cannot steer the result.
  std::vector<unsigned> regs, inputs;
  const LocSet read = roles.decoded ? roles.read : all_locs();
  for (unsigned l = 0; l < kNumLocs; ++l)
    if (entry.test(l) && l != kLocScc) {
      regs.push_back(l);
      if (read.test(l))
        inputs.push_back(l);
    }

  // 1. uniform
  for (const NamedValue &v : catalog32()) {
    std::vector<std::pair<unsigned, uint32_t>> all;
    for (unsigned l : regs)
      all.emplace_back(l, v.value);
    if (!all.empty())
      for (int8_t scc : scc_variants(2))
        emit(0, scc, all, "uniform all=" + v.name, "uniform");
  }
  // 2. single
  for (unsigned l : inputs) {
    std::vector<NamedValue> values = catalog32();
    for (const auto &v : role_catalog(roles.roles[l]))
      add_unique(values, v.value, v.name);
    for (const auto &rel : roles.const_relations)
      if (rel.loc == l)
        for (const auto &v : const_relative_values(rel.constant))
          add_unique(values, v.value, v.name);
    for (const NamedValue &v : values)
      for_bases({0, 1, 2}, {{l, v.value}}, "single " + loc_name(l) + "=" + v.name,
                value_family(v.name));
  }
  // 3. pair64: 64-bit operands through the 64-bit catalog.
  for (unsigned l : inputs) {
    if (!contract.wide_reads.test(l) || l + 1 >= kNumLocs || !entry.test(l + 1))
      continue;
    for (const NamedValue64 &v : catalog64())
      for_bases(
          {0, 1, 2},
          {{l, static_cast<uint32_t>(v.value)}, {l + 1, static_cast<uint32_t>(v.value >> 32)}},
          "pair64 " + loc_name(l) + ":" + loc_name(l + 1) + "=" + v.name, value_family(v.name));
  }
  // 4. two-register source pairs.
  for (const OperandPair &p : roles.pairs) {
    if (p.a == p.b)
      continue;
    if (!p.wide) {
      if (!entry.test(p.a) || !entry.test(p.b))
        continue;
      for (const PairValue &v : pair_catalog())
        for_bases({0, 2}, {{p.a, v.a}, {p.b, v.b}},
                  "pair " + p.mnemonic + " " + loc_name(p.a) + "," + loc_name(p.b) + "=" + v.name,
                  value_family(v.name) == "cross" ? "pair.cross" : "pair.targeted");
    } else {
      if (!entry.test(p.a) || !entry.test(p.a + 1) || !entry.test(p.b) || !entry.test(p.b + 1))
        continue;
      for (const PairValue64 &v : pair_catalog64())
        for_bases({0, 2},
                  {{p.a, static_cast<uint32_t>(v.a)},
                   {p.a + 1, static_cast<uint32_t>(v.a >> 32)},
                   {p.b, static_cast<uint32_t>(v.b)},
                   {p.b + 1, static_cast<uint32_t>(v.b >> 32)}},
                  "pair64 " + p.mnemonic + " " + loc_name(p.a) + "," + loc_name(p.b) + "=" + v.name,
                  "pair64");
    }
  }
  // 6. VALU regions: lane-parallel vector phases. Each case puts 64 catalog values into a
  // VGPR at once, with every lane active; scalar bases stay as in the other phases.
  if (sweep.vgpr_count) {
    const VSet ventry = contract.ventry_equal;
    const VSet vread = roles.decoded ? roles.vread : ventry;
    const uint32_t n32 = static_cast<uint32_t>(sweep.vcat.size());
    const uint32_t n64 = static_cast<uint32_t>(catalog64().size());
    auto chunks = [](uint64_t n, uint64_t cap) {
      return static_cast<uint32_t>(std::min<uint64_t>((n + kWaveLanes - 1) / kWaveLanes, cap));
    };
    auto vemit = [&](std::vector<VOverride> v, std::vector<std::pair<unsigned, uint32_t>> scalar,
                     int8_t exec, uint8_t base, const std::string &label,
                     const std::string &family) {
      for (int8_t scc : scc_variants(base)) {
        pending_v = v;
        pending_exec = exec;
        emit(base, scc, scalar, label + " base=" + kBaseName[base], family);
      }
    };
    for (unsigned r = 0; r < sweep.vgpr_count; ++r) {
      if (!ventry.test(r) || !vread.test(r))
        continue;
      const std::string name = "v" + std::to_string(r);
      for (uint32_t k = 0; k < chunks(n32, 64); ++k)
        for (uint8_t base : {uint8_t{0}, uint8_t{2}})
          vemit({{VOverride::Chunk32, static_cast<uint16_t>(r), 0, false, k, 0}}, {}, 0, base,
                "vlanes " + name + " chunk " + std::to_string(k), "vector.lanes");
      if (sweep.vwide.test(r) && r + 1 < sweep.vgpr_count && ventry.test(r + 1))
        for (uint32_t k = 0; k < chunks(n64, 64); ++k)
          vemit({{VOverride::Chunk64, static_cast<uint16_t>(r), 0, true, k, 0}}, {}, 0, 2,
                "vlanes64 " + name + ":v" + std::to_string(r + 1) + " chunk " + std::to_string(k),
                "vector.lanes64");
    }
    for (const VectorPair &p : roles.vpairs) {
      if (p.a_vector && p.b_vector) {
        if (p.a == p.b || !ventry.test(p.a) || !ventry.test(p.b))
          continue;
        const uint64_t n = p.wide ? n64 : n32;
        for (uint32_t k = 0; k < chunks(n * n, 1024); ++k)
          vemit({{VOverride::Cross, static_cast<uint16_t>(p.a), static_cast<uint16_t>(p.b), p.wide,
                  k, 0}},
                {}, 0, 2,
                "vcross " + p.mnemonic + " v" + std::to_string(p.a) + ",v" + std::to_string(p.b) +
                    " chunk " + std::to_string(k),
                "vector.cross");
      } else if (!p.wide) {
        // One scalar and one vector source: every scalar catalog value against every
        // lane chunk of the vector source.
        const unsigned s = p.a_vector ? p.b : p.a;
        const unsigned v = p.a_vector ? p.a : p.b;
        if (!entry.test(s) || !ventry.test(v))
          continue;
        for (const NamedValue &x : catalog32())
          for (uint32_t k = 0; k < chunks(n32, 64); ++k)
            vemit({{VOverride::Chunk32, static_cast<uint16_t>(v), 0, false, k, 0}}, {{s, x.value}},
                  0, 2,
                  "svcross " + p.mnemonic + " " + loc_name(s) + "=" + x.name + " v" +
                      std::to_string(v) + " chunk " + std::to_string(k),
                  "vector.scalar_cross");
      }
    }
    for (const auto &[r, constant] : roles.vconst_relations) {
      if (!ventry.test(r))
        continue;
      std::vector<uint32_t> list;
      for (const NamedValue &v : const_relative_values(constant))
        list.push_back(v.value);
      if (list.empty())
        continue;
      sweep.value_lists.push_back(std::move(list));
      const uint32_t id = static_cast<uint32_t>(sweep.value_lists.size() - 1);
      for (uint32_t k = 0; k < chunks(sweep.value_lists[id].size(), 16); ++k)
        vemit({{VOverride::Values, static_cast<uint16_t>(r), 0, false, k, id}}, {}, 0, 2,
              "vconstrel v" + std::to_string(r) + " vs " + hex32(constant), "vector.constrel");
    }
    // EXEC: alternate lanes (inactive-lane preservation) and the 64-bit EXEC patterns.
    if (sweep.exec_entry) {
      for (uint8_t base : {uint8_t{0}, uint8_t{1}, uint8_t{2}})
        vemit({}, {}, 1, base, "vexec alternate lanes", "vector.exec");
      for (const NamedValue64 &v : catalog64())
        if (value_family(v.name) == "exec")
          vemit({},
                {{kLocExecLo, static_cast<uint32_t>(v.value)},
                 {kLocExecHi, static_cast<uint32_t>(v.value >> 32)}},
                -1, 2, "vexec " + v.name, "vector.exec");
    }
  }
  // 5. constraint edges.
  for (const Constraint &c : contract.constraints) {
    std::vector<NamedValue> edges;
    if (c.umin) {
      add_unique(edges, *c.umin, "constraint.umin");
      add_unique(edges, *c.umin + 1, "constraint.umin_plus_1");
      add_unique(edges, *c.umin - 1, "constraint.umin_minus_1");
    }
    if (c.umax) {
      add_unique(edges, *c.umax, "constraint.umax");
      add_unique(edges, *c.umax - 1, "constraint.umax_minus_1");
      add_unique(edges, *c.umax + 1, "constraint.umax_plus_1");
    }
    add_unique(edges, c.known_one, "constraint.known_min");
    add_unique(edges, ~c.known_zero, "constraint.known_max");
    add_unique(edges, c.known_one | (~c.known_zero & 0x55555555u), "constraint.known_alt");
    for (const NamedValue &v : edges)
      for_bases({0, 2}, {{c.loc, v.value}}, "edge " + loc_name(c.loc) + "=" + v.name,
                "constraint_edge");
  }
  return sweep;
}

} // namespace rocjitsu::isa_diff
