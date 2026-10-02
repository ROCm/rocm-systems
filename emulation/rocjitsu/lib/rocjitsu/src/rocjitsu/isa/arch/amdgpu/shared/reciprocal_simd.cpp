// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "util/amdgpu_rcp.h"
#include "util/amdgpu_rsq.h"

#include <array>
#include <cstddef>
#include <cstdint>

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>

namespace rocjitsu::amdgpu::transcendental::detail {
namespace {
// The scalar polynomial coefficients and sparse residuals remain authoritative.
// Packing columns only changes their layout for two bounded vector gathers.
template <typename Coefficient, size_t N>
constexpr auto coefficient_columns(const Coefficient (&rows)[N], bool curved) {
  std::array<long long, N> result{};
  for (size_t i = 0; i < N; ++i)
    result[i] = static_cast<long long>(
        curved ? uint64_t(rows[i].quadratic) | (uint64_t(rows[i].cubic) << 16)
               : uint64_t(rows[i].constant) | (uint64_t(rows[i].linear) << 32));
  return result;
}

// Regenerate the plans with scripts/generate-reciprocal-simd.py.
// A bucket plan encodes an odd multiplier and log2(slot count). Multiplication
// modulo 2^15 followed by a shift gives a collision-free slot for its residuals.
// The complete stored key is checked at runtime, so other inputs remain misses.
// Payload construction below rejects collisions or use of the empty sentinel.
template <size_t N> struct CorrectionLayout {
  std::array<long long, N> buckets{};
  size_t entries = 0;
};
template <size_t N> constexpr auto correction_layout(const uint32_t (&plan)[N]) {
  CorrectionLayout<N> result;
  for (size_t bucket = 0; bucket < N; ++bucket) {
    const uint32_t seed = plan[bucket] & 0xffff, log_size = plan[bucket] >> 16;
    if (!(seed & 1) || log_size > 15)
      throw "invalid correction hash bucket";
    result.buckets[bucket] = static_cast<long long>(result.entries | (uint64_t(seed) << 32) |
                                                    (uint64_t(15 - log_size) << 48));
    result.entries += size_t{1} << log_size;
  }
  return result;
}
template <size_t Entries, size_t N, size_t O, size_t C>
constexpr auto correction_words(const CorrectionLayout<N> &layout, const uint16_t (&offsets)[O],
                                const uint16_t (&corrections)[C]) {
  static_assert(O == N + 1);
  std::array<uint32_t, (Entries + 1) / 2> words{};
  for (auto &word : words)
    word = 0xffffffff;
  for (size_t bucket = 0; bucket < N; ++bucket) {
    const uint64_t metadata = static_cast<uint64_t>(layout.buckets[bucket]);
    const uint32_t offset = uint32_t(metadata), seed = (metadata >> 32) & 0xffff;
    const unsigned shift = metadata >> 48;
    for (size_t i = offsets[bucket]; i < offsets[bucket + 1]; ++i) {
      const uint16_t entry = corrections[i];
      const size_t slot = offset + (((uint32_t(entry & 0x7fff) * seed) & 0x7fff) >> shift);
      const unsigned half_shift = unsigned(slot & 1) * 16;
      if (entry == 0xffff || slot >= Entries ||
          ((words[slot / 2] >> half_shift) & 0xffff) != 0xffff)
        throw "correction hash plan does not match source entries";
      words[slot / 2] =
          (words[slot / 2] & ~(uint32_t{0xffff} << half_shift)) | (uint32_t(entry) << half_shift);
    }
  }
  return words;
}
struct Tables {
  const long long *constant_linear, *quadratic_cubic, *buckets;
  const uint32_t *corrections;
};
constexpr auto rcp_constant_linear = coefficient_columns(util::detail::rcp::coefficients, false);
constexpr auto rcp_quadratic_cubic = coefficient_columns(util::detail::rcp::coefficients, true);
constexpr uint32_t rcp_hash_plan[] = {
    0x502a9u, 0x6035bu, 0x60163u, 0x6013fu, 0x600d7u, 0x60309u, 0x6020bu, 0x60283u, 0x50111u,
    0x50003u, 0x6001bu, 0x5008du, 0x60003u, 0x5001bu, 0x6002du, 0x601ffu, 0x6003fu, 0x5005fu,
    0x50051u, 0x5000fu, 0x60015u, 0x50021u, 0x602e1u, 0x70127u, 0x60009u, 0x500d5u, 0x50013u,
    0x5052bu, 0x60049u, 0x5001bu, 0x60083u, 0x60017u, 0x5000du, 0x50045u, 0x40009u, 0x5012fu,
    0x5000bu, 0x600c7u, 0x6005bu, 0x50057u, 0x5017fu, 0x40007u, 0x60001u, 0x5001bu, 0x50001u,
    0x5000fu, 0x60035u, 0x60029u, 0x40007u, 0x5002du, 0x50001u, 0x50001u, 0x60003u, 0x6000du,
    0x50007u, 0x50041u, 0x50001u, 0x5003bu, 0x40003u, 0x5000du, 0x40001u, 0x5001fu, 0x40001u,
    0x60017u, 0x500f3u, 0x40001u, 0x50009u, 0x50029u, 0x40001u, 0x50031u, 0x50005u, 0x6007du,
    0x5000bu, 0x50101u, 0x40005u, 0x50027u, 0x50005u, 0x40001u, 0x50007u, 0x50005u, 0x4002fu,
    0x30001u, 0x5000fu, 0x50003u, 0x40001u, 0x50003u, 0x5001du, 0x60019u, 0x4000bu, 0x50009u,
    0x50001u, 0x50041u, 0x4001fu, 0x5003fu, 0x40001u, 0x40007u, 0x50013u, 0x40003u, 0x30003u,
    0x5014du, 0x5005du, 0x501afu, 0x50003u, 0x5000bu, 0x40011u, 0x40013u, 0x40001u, 0x50007u,
    0x40003u, 0x50071u, 0x5006du, 0x50005u, 0x50003u, 0x40005u, 0x50005u, 0x50017u, 0x40015u,
    0x40003u, 0x40001u, 0x50005u, 0x5001fu, 0x4001bu, 0x40017u, 0x4001bu, 0x40001u, 0x5000bu,
    0x50003u, 0x50019u, 0x500a5u, 0x50021u, 0x40009u, 0x5000bu, 0x50009u, 0x50033u, 0x40001u,
    0x5000bu, 0x40003u, 0x50003u, 0x50033u, 0x30001u, 0x30005u, 0x5000bu, 0x50017u, 0x50001u,
    0x40001u, 0x4004du, 0x40001u, 0x4000du, 0x5000du, 0x40013u, 0x5000du, 0x50005u, 0x50025u,
    0x40017u, 0x40001u, 0x40003u, 0x4000bu, 0x40007u, 0x40003u, 0x40005u, 0x50091u, 0x40007u,
    0x20001u, 0x40003u, 0x40007u, 0x40003u, 0x5000fu, 0x4005bu, 0x4000bu, 0x50011u, 0x40001u,
    0x50001u, 0x50001u, 0x50007u, 0x4000bu, 0x40001u, 0x40005u, 0x40001u, 0x40001u, 0x5000bu,
    0x40001u, 0x40001u, 0x50017u, 0x50035u, 0x40035u, 0x4000bu, 0x50009u, 0x30001u, 0x40003u,
    0x50067u, 0x4000du, 0x40001u, 0x50021u, 0x30001u, 0x50013u, 0x40001u, 0x30005u, 0x50003u,
    0x40005u, 0x50021u, 0x40007u, 0x50007u, 0x4000bu, 0x5001fu, 0x50005u, 0x5001bu, 0x50015u,
    0x50055u, 0x50009u, 0x40011u, 0x40001u, 0x40009u, 0x5000bu, 0x40007u, 0x50009u, 0x40005u,
    0x50049u, 0x50005u, 0x50001u, 0x5003bu, 0x30003u, 0x40005u, 0x30001u, 0x50009u, 0x4012bu,
    0x5000du, 0x50001u, 0x50011u, 0x30001u, 0x50003u, 0x50017u, 0x50013u, 0x40001u, 0x50001u,
    0x5000fu, 0x50021u, 0x30001u, 0x30001u, 0x40003u, 0x40001u, 0x5000bu, 0x30001u, 0x40007u,
    0x50001u, 0x50005u, 0x30001u, 0x40007u, 0x30007u, 0x50025u, 0x40003u, 0x40015u, 0x4000fu,
    0x5000bu, 0x40003u, 0x50003u, 0x5006du,
};
constexpr auto rcp_layout = correction_layout(rcp_hash_plan);
constexpr auto rcp_corrections = correction_words<rcp_layout.entries>(
    rcp_layout, util::detail::kAmdgpuRcpCorrectionOffsets, util::detail::kAmdgpuRcpCorrections);
constexpr Tables rcp_tables{rcp_constant_linear.data(), rcp_quadratic_cubic.data(),
                            rcp_layout.buckets.data(), rcp_corrections.data()};
constexpr auto rsq_constant_linear = coefficient_columns(util::detail::rsq::coefficients, false);
constexpr auto rsq_quadratic_cubic = coefficient_columns(util::detail::rsq::coefficients, true);
constexpr uint32_t rsq_hash_plan[] = {
    0x30001u, 0x40019u, 0x40071u, 0x40077u, 0x30001u, 0x4008bu, 0x5001du, 0x50081u, 0x4004du,
    0x50001u, 0x40007u, 0x40005u, 0x4001fu, 0x40003u, 0x40003u, 0x40007u, 0x30001u, 0x40001u,
    0x5002du, 0x4000du, 0x30001u, 0x40001u, 0x40021u, 0x50005u, 0x40049u, 0x40011u, 0x40005u,
    0x40001u, 0x40019u, 0x4000bu, 0x4000bu, 0x50011u, 0x4002bu, 0x30005u, 0x40011u, 0x4000bu,
    0x40003u, 0x50005u, 0x50007u, 0x3000fu, 0x40011u, 0x4000bu, 0x10001u, 0x40005u, 0x40001u,
    0x10001u, 0x40007u, 0x5001bu, 0x30001u, 0x30003u, 0x30001u, 0x40015u, 0x40011u, 0x30001u,
    0x50015u, 0x50005u, 0x40001u, 0x30003u, 0x40017u, 0x40015u, 0x30001u, 0x4000bu, 0x30001u,
    0x30005u, 0x40027u, 0x40005u, 0x30001u, 0x40001u, 0x30003u, 0x40005u, 0x40003u, 0x5000bu,
    0x40047u, 0x4000bu, 0x30005u, 0x3000bu, 0x30001u, 0x20003u, 0x20001u, 0x50005u, 0x50159u,
    0x40003u, 0x1u,     0x40001u, 0x20001u, 0x40001u, 0x40001u, 0x30003u, 0x3000bu, 0x40007u,
    0x4000bu, 0x30001u, 0x3000bu, 0x30005u, 0x3000bu, 0x40001u, 0x40049u, 0x30001u, 0x40009u,
    0x30003u, 0x20001u, 0x20005u, 0x10001u, 0x40003u, 0x30003u, 0x4000bu, 0x30001u, 0x20003u,
    0x20001u, 0x40005u, 0x20003u, 0x30001u, 0x30005u, 0x10001u, 0x30007u, 0x30001u, 0x4001du,
    0x30001u, 0x40001u, 0x40001u, 0x40001u, 0x1u,     0x40001u, 0x20001u, 0x30005u, 0x30001u,
    0x40005u, 0x40003u, 0x50007u, 0x30001u, 0x40003u, 0x20003u, 0x10001u, 0x20001u, 0x20001u,
    0x30001u, 0x30001u, 0x40005u, 0x30001u, 0x40003u, 0x40005u, 0x40001u, 0x40001u, 0x40005u,
    0x40001u, 0x40005u, 0x20001u, 0x20001u, 0x10001u, 0x40001u, 0x30003u, 0x30001u, 0x3002fu,
    0x10001u, 0x30001u, 0x20007u, 0x30001u, 0x20003u, 0x40003u, 0x40001u, 0x5002fu, 0x40009u,
    0x30001u, 0x30005u, 0x40007u, 0x40001u, 0x20005u, 0x40007u, 0x40009u, 0x30001u, 0x3000du,
    0x50001u, 0x30001u, 0x40009u, 0x30001u, 0x1u,     0x40001u, 0x3000bu, 0x20001u, 0x10001u,
    0x40005u, 0x40007u, 0x40001u, 0x40005u, 0x3002fu, 0x20005u, 0x20001u, 0x30001u, 0x40003u,
    0x40001u, 0x30007u, 0x40005u, 0x40001u, 0x20001u, 0x40003u, 0x40003u, 0x30005u, 0x30001u,
    0x50001u, 0x40003u, 0x50027u, 0x30001u, 0x30009u, 0x30005u, 0x30001u, 0x30001u, 0x30001u,
    0x4000du, 0x40027u, 0x40001u, 0x10001u, 0x40001u, 0x10001u, 0x20005u, 0x30003u, 0x40035u,
    0x30031u, 0x30003u, 0x30003u, 0x30001u, 0x2000bu, 0x40001u, 0x40009u, 0x40001u, 0x4001bu,
    0x4000bu, 0x20001u, 0x30001u, 0x40005u, 0x10001u, 0x30003u, 0x40027u, 0x40031u, 0x30017u,
    0x30007u, 0x30001u, 0x40007u, 0x40001u, 0x40011u, 0x40001u, 0x30001u, 0x40011u, 0x50001u,
    0x30001u, 0x20001u, 0x20001u, 0x4000bu, 0x30001u, 0x3003du, 0x20001u, 0x30001u, 0x4000bu,
    0x30005u, 0x30005u, 0x30001u, 0x30009u, 0x50049u, 0x50007u, 0x40001u, 0x30009u, 0x40001u,
    0x40001u, 0x40003u, 0x40003u, 0x30001u, 0x20001u, 0x40001u, 0x4000du, 0x40003u, 0x50001u,
    0x40007u, 0x40003u, 0x40001u, 0x30001u, 0x30001u, 0x5000bu, 0x40001u, 0x40003u, 0x20001u,
    0x5001du, 0x40001u, 0x30001u, 0x10001u, 0x40009u, 0x40001u, 0x30001u, 0x5000fu, 0x5000bu,
    0x5002bu, 0x1u,     0x30001u, 0x40001u, 0x40001u, 0x5000bu, 0x40007u, 0x30005u, 0x40031u,
    0x30001u, 0x40001u, 0x30001u, 0x40001u, 0x4000bu, 0x40001u, 0x40003u, 0x4009fu, 0x10001u,
    0x30017u, 0x20001u, 0x40001u, 0x40001u, 0x40005u, 0x30001u, 0x30001u, 0x3000bu, 0x40001u,
    0x30005u, 0x30001u, 0x30003u, 0x40001u, 0x4000bu, 0x20005u, 0x30001u, 0x4002fu, 0x1u,
    0x30001u, 0x40039u, 0x40003u, 0x30001u, 0x40009u, 0x30001u, 0x40001u, 0x30001u, 0x20001u,
    0x20001u, 0x30005u, 0x30009u, 0x4007fu, 0x10001u, 0x40001u, 0x30001u, 0x40005u, 0x4000bu,
    0x40001u, 0x50007u, 0x40023u, 0x30011u, 0x30001u, 0x40001u, 0x30003u, 0x30003u, 0x20001u,
    0x30005u, 0x40001u, 0x30003u, 0x40009u, 0x30001u, 0x30001u, 0x30003u, 0x30001u, 0x30001u,
    0x50021u, 0x30001u, 0x30001u, 0x30003u, 0x30001u, 0x4000fu, 0x40007u, 0x30001u, 0x3003fu,
    0x30003u, 0x20001u, 0x30007u, 0x40005u, 0x20003u, 0x4000du, 0x40001u, 0x40005u, 0x20003u,
    0x40001u, 0x30003u, 0x40001u, 0x30001u, 0x30005u, 0x4000fu, 0x30033u, 0x30001u, 0x40003u,
    0x40001u, 0x20001u, 0x30003u, 0x30001u, 0x40003u, 0x50013u, 0x30007u, 0x20001u, 0x40001u,
    0x4001fu, 0x40003u, 0x30003u, 0x40001u, 0x30175u, 0x4000bu, 0x40001u, 0x30001u, 0x30001u,
    0x30001u, 0x40049u, 0x40003u, 0x50043u, 0x30001u, 0x4000du, 0x30011u, 0x30001u, 0x40007u,
    0x30003u, 0x40017u, 0x40029u, 0x40005u, 0x10001u, 0x40005u, 0x4000bu, 0x30001u, 0x50053u,
    0x5000fu, 0x30001u, 0x40005u, 0x10001u, 0x30009u, 0x30001u, 0x20001u, 0x30007u, 0x30001u,
    0x10001u, 0x40001u, 0x30001u, 0x10001u, 0x30001u, 0x50011u, 0x40015u, 0x50001u, 0x40015u,
    0x30001u, 0x40003u, 0x30001u, 0x30001u, 0x10001u, 0x10001u, 0x40001u, 0x30003u, 0x30001u,
    0x10001u, 0x40001u, 0x30001u, 0x30005u, 0x40005u, 0x50007u, 0x4001du, 0x10001u, 0x30001u,
    0x20001u, 0x20001u, 0x20001u, 0x40003u, 0x30001u, 0x40015u, 0x20001u, 0x40005u, 0x30001u,
    0x30003u, 0x30001u, 0x30001u, 0x30001u, 0x50035u, 0x40005u, 0x4001du, 0x20001u, 0x20003u,
    0x40009u, 0x20001u, 0x30001u, 0x30001u, 0x20001u, 0x20001u, 0x20007u, 0x30001u, 0x30001u,
    0x40007u, 0x40001u, 0x40027u, 0x30001u, 0x40001u, 0x30001u, 0x40003u, 0x10001u, 0x50001u,
    0x10001u, 0x50009u, 0x40005u, 0x40009u, 0x1u,     0x30001u, 0x40003u, 0x30007u, 0x40005u,
    0x40001u, 0x20001u, 0x50005u, 0x40003u, 0x40001u, 0x40015u, 0x4000du, 0x40019u,
};
constexpr auto rsq_layout = correction_layout(rsq_hash_plan);
constexpr auto rsq_corrections = correction_words<rsq_layout.entries>(
    rsq_layout, util::detail::kAmdgpuRsqCorrectionOffsets, util::detail::kAmdgpuRsqCorrections);
constexpr Tables rsq_tables{rsq_constant_linear.data(), rsq_quadratic_cubic.data(),
                            rsq_layout.buckets.data(), rsq_corrections.data()};
// The two layouts together use 33,964 bytes of metadata and correction words.

// Powers of two need no polynomial or correction lookup. These constants are
// the same rounded coefficient at zero fraction, with no residual at that key.
constexpr uint32_t rcp_one =
    0x3e800000u + (util::detail::rcp::coefficients[0].constant * 16u + 32u) / 64u;
constexpr uint32_t rsq_one =
    0x3e800000u + (util::detail::rsq::coefficients[0].constant * 32u + 64u) / 128u;
constexpr uint32_t rsq_two =
    0x3e800000u + (util::detail::rsq::coefficients[32].constant * 32u + 64u) / 128u;
static_assert((util::detail::kAmdgpuRcpCorrections[0] & 0x7fff) != 0);
static_assert((util::detail::kAmdgpuRsqCorrections[0] & 0x7fff) != 0);
static_assert((util::detail::kAmdgpuRsqCorrections[util::detail::kAmdgpuRsqCorrectionOffsets[256]] &
               0x7fff) != 0);

using V = __m512i;
using M = __mmask8;
[[gnu::target("avx512f"), gnu::always_inline]] inline V set(uint64_t a) {
  return _mm512_set1_epi64(static_cast<long long>(a));
}
[[gnu::target("avx512f"), gnu::always_inline]] inline V add(V a, V b) {
  return _mm512_add_epi64(a, b);
}
[[gnu::target("avx512f"), gnu::always_inline]] inline V sub(V a, V b) {
  return _mm512_sub_epi64(a, b);
}
[[gnu::target("avx512f"), gnu::always_inline]] inline V band(V a, V b) {
  return _mm512_and_si512(a, b);
}
[[gnu::target("avx512f"), gnu::always_inline]] inline V bor(V a, V b) {
  return _mm512_or_si512(a, b);
}
[[gnu::target("avx512f"), gnu::always_inline]] inline V bxor(V a, V b) {
  return _mm512_xor_si512(a, b);
}
[[gnu::target("avx512f"), gnu::always_inline]] inline V multiply_low32(V a, V b) {
  return _mm512_mul_epu32(a, b);
}
[[gnu::target("avx512f"), gnu::always_inline]] inline V shr_each(V a, V b) {
  return _mm512_srlv_epi64(a, b);
}
[[gnu::target("avx512f"), gnu::always_inline]] inline V shl_each(V a, V b) {
  return _mm512_sllv_epi64(a, b);
}
[[gnu::target("avx512f"), gnu::always_inline]] inline V shr(V a, unsigned n) {
  return shr_each(a, set(n));
}
[[gnu::target("avx512f"), gnu::always_inline]] inline V shl(V a, unsigned n) {
  return shl_each(a, set(n));
}
[[gnu::target("avx512f"), gnu::always_inline]] inline M gt(V a, V b) {
  return _mm512_cmp_epi64_mask(a, b, _MM_CMPINT_GT);
}
[[gnu::target("avx512f"), gnu::always_inline]] inline M eq(V a, V b) {
  return _mm512_cmp_epi64_mask(a, b, _MM_CMPINT_EQ);
}
[[gnu::target("avx512f"), gnu::always_inline]] inline M mand(M a, M b) { return a & b; }
[[gnu::target("avx512f"), gnu::always_inline]] inline M mor(M a, M b) { return a | b; }
[[gnu::target("avx512f"), gnu::always_inline]] inline M mnot(M a) { return static_cast<M>(~a); }
[[gnu::target("avx512f"), gnu::always_inline]] inline bool any(M a) { return a != 0; }
[[gnu::target("avx512f"), gnu::always_inline]] inline V select(M mask, V yes, V no) {
  return _mm512_mask_blend_epi64(mask, no, yes);
}
[[gnu::target("avx512f"), gnu::always_inline]] inline V gather_masked(const long long *p, V index,
                                                                      M mask) {
  return _mm512_mask_i64gather_epi64(set(0), mask, index, p, 8);
}
[[gnu::target("avx512f"), gnu::always_inline]] inline V gather32_masked(const uint32_t *p, V index,
                                                                        M mask) {
  return _mm512_cvtepu32_epi64(
      _mm512_mask_i64gather_epi32(_mm256_setzero_si256(), mask, index, p, 4));
}
[[gnu::target("avx512f"), gnu::always_inline]] inline V load(const uint32_t *p) {
  return _mm512_cvtepu32_epi64(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(p)));
}

[[gnu::target("avx512f"), gnu::always_inline]] inline V normalized(V index, bool rsq,
                                                                   const Tables &table, M useful) {
  const V zero = set(0), one = set(1);
  const V segment = shr(index, 18);
  const V packed = gather_masked(table.constant_linear, segment, useful);
  const V curved = gather_masked(table.quadratic_cubic, segment, useful);
  const V constant = band(packed, set(0xffffffff));
  const V linear_coefficient = shr(packed, 32);
  const V quadratic_coefficient = band(curved, set(0xffff));
  const V cubic = shr(curved, 16);
  const V fraction = band(index, set(0x3ffff));
  const V product = multiply_low32(linear_coefficient, fraction);
  const unsigned small_shift = rsq ? 13 : 14;
  const V shift = select(gt(product, set((uint64_t{1} << (rsq ? 37 : 38)) - 1)),
                         set(small_shift + 1), set(small_shift));
  const V quotient = shr_each(product, shift);
  const V remainder = band(product, sub(shl_each(one, shift), one));
  const V midpoint = shl_each(one, sub(shift, one));
  const M increment =
      mor(gt(remainder, midpoint), mand(eq(remainder, midpoint), eq(band(quotient, one), one)));
  const V rounded = add(quotient, select(increment, one, zero));
  const V linear = shl_each(rounded, sub(shift, set(small_shift)));
  const V inner = sub(shl(quadratic_coefficient, 18), multiply_low32(cubic, fraction));
  const V f = band(fraction, set(0x3fffe));
  // P=I*f fits 52 bits. Split P*f before the original exact >>51.
  const V p = add(multiply_low32(inner, f), shl(multiply_low32(shr(inner, 32), f), 32));
  const V lo = multiply_low32(p, f), hi = multiply_low32(shr(p, 32), f);
  const V quadratic = add(shr(hi, 19), shr(add(lo, shl(band(hi, set(0x7ffff)), 32)), 51));
  V value;
  if (rsq)
    value = shr(add(add(sub(shl(constant, 5), linear), shl(quadratic, 2)), set(64)), 7);
  else
    value = shr(add(add(sub(shl(constant, 4), linear), shl(quadratic, 1)), set(32)), 6);
  value = add(value, set(0x3e800000));
  const V bucket = shr(index, 15), key = band(index, set(0x7fff));
  const V metadata = gather_masked(table.buckets, bucket, useful);
  const V offset = band(metadata, set(0xffffffff));
  const V seed = band(shr(metadata, 32), set(0xffff));
  const V slot =
      add(offset, shr_each(band(multiply_low32(key, seed), set(0x7fff)), shr(metadata, 48)));
  const V packed_correction = gather32_masked(table.corrections, shr(slot, 1), useful);
  const V correction = band(shr_each(packed_correction, shl(band(slot, one), 4)), set(0xffff));
  const M match =
      mand(useful, mand(mnot(eq(correction, set(0xffff))), eq(band(correction, set(0x7fff)), key)));
  const V corrected =
      select(eq(band(correction, set(0x8000)), zero), sub(value, one), add(value, one));
  return select(match, corrected, value);
}

} // namespace

// Called only after feature admission and for complete eight-lane chunks.
// Integer arithmetic leaves host FP control, sticky flags and errno untouched.
[[gnu::target("avx512f")]] void evaluate_reciprocal_f32_simd(bool rsq, const uint32_t *input,
                                                             uint32_t *output, size_t count,
                                                             bool quiet_snan) {
  const Tables &table = rsq ? rsq_tables : rcp_tables;
  const V zero = set(0), sign_mask = set(0x80000000), magnitude_mask = set(0x7fffffff);
  for (size_t start = 0; start < count; start += 8) {
    const V bits = load(input + start);
    const V sign = band(bits, sign_mask), magnitude = band(bits, magnitude_mask);
    const V exponent = shr(magnitude, 23);
    const V parity = band(bxor(exponent, set(1)), set(1));
    const V mantissa = band(magnitude, set(0x7fffff));
    const V index = rsq ? bor(shl(parity, 23), mantissa) : mantissa;
    M normal = mand(gt(magnitude, set(0x7fffff)), gt(set(0x7f800000), magnitude));
    if (rsq)
      normal = mand(normal, eq(sign, zero));
    // Mantissa zero is an exact captured constant for either exponent parity.
    const M polynomial = mand(normal, gt(mantissa, zero));
    V approximation = rsq ? select(eq(parity, zero), set(rsq_one), set(rsq_two)) : set(rcp_one);
    if (any(polynomial))
      approximation = select(polynomial, normalized(index, rsq, table, polynomial), approximation);
    V result;
    if (rsq) {
      const V e = sub(add(add(shr(approximation, 23), set(63)), parity), shr(exponent, 1));
      result = bor(shl(e, 23), band(approximation, set(0x7fffff)));
      result = select(eq(magnitude, set(0x7f800000)), zero, result);
      result = select(eq(sign, sign_mask), set(0xffc00000), result);
    } else {
      const V e = sub(add(shr(approximation, 23), set(127)), exponent);
      result = bor(sign, bor(shl(e, 23), band(approximation, set(0x7fffff))));
      result = select(gt(e, zero), result, sign);
      result = select(gt(e, set(254)), bor(sign, set(0x7f800000)), result);
      result = select(eq(magnitude, set(0x7f800000)), sign, result);
    }
    result = select(gt(set(0x800000), magnitude), bor(sign, set(0x7f800000)), result);
    result =
        select(gt(magnitude, set(0x7f800000)), bor(bits, set(quiet_snan ? 0x400000 : 0)), result);
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(output + start), _mm512_cvtepi64_epi32(result));
  }
}
} // namespace rocjitsu::amdgpu::transcendental::detail
#endif
