// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/vgpr_writeback.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <cstdint>
#include <cstring>
#include <optional>
#include <random>
#include <type_traits>
#include <vector>

namespace {

namespace wb = rocjitsu::amdgpu::vgpr_writeback;

uint64_t random_exec(std::mt19937_64 &rng, uint32_t wave) {
  uint64_t exec = rng();
  switch (rng() % 7) {
  case 0:
    exec = ~uint64_t{0};
    break;
  case 1:
    exec = 0;
    break;
  case 2:
    exec = uint64_t{1} << (rng() % 64);
    break;
  case 3:
    exec = 0xFFFF'FFFF'FFFFull;
    break;
  case 4:
    exec &= rng() & rng();
    break;
  case 5:
    // Two to four lanes: the range write_back() handles one lane at a time.
    exec = 0;
    for (uint32_t n = 2 + rng() % 3; n; --n)
      exec |= uint64_t{1} << (rng() % wave);
    break;
  default:
    break;
  }
  return exec & wb::prefix(wave);
}

/// The per-lane loop the typed writebacks replace: for each active lane and
/// destination register, the dword it held before the typed writeback.
template <wb::DstFormat F>
void reference_write_back(uint32_t *regs, bool sram_ecc, uint64_t exec,
                          const uint8_t *const *groups) {
  const uint32_t copy_size = std::min<uint32_t>(F.elem_size, 4);
  for (uint64_t active = exec; active; active &= active - 1) {
    const uint32_t lane = static_cast<uint32_t>(std::countr_zero(active));
    const uint8_t *source = groups[lane / 16] + (lane % 16) * wb::kStride<F>;
    for (uint32_t r = 0; r < wb::kRegs<F>; ++r) {
      uint32_t value = 0;
      std::memcpy(&value, source + 4 * r, copy_size);
      if (F.sign_extend && F.elem_size == 1)
        value = static_cast<uint32_t>(static_cast<int32_t>(static_cast<int8_t>(value)));
      if (F.sign_extend && F.elem_size == 2)
        value = static_cast<uint32_t>(static_cast<int32_t>(static_cast<int16_t>(value)));
      uint32_t &dword = regs[r * 64 + lane];
      if (F.d16_hi)
        value = (sram_ecc ? 0 : dword & 0xFFFFu) | value << 16;
      else if (F.d16_lo)
        value = (sram_ecc ? 0 : dword & 0xFFFF0000u) | (value & 0xFFFFu);
      dword = value;
    }
  }
}

/// Each ISA against the reference for one format: whole
/// destination registers plus a register on either side, wave32 and wave64,
/// SRAM ECC on and off, groups apart from each other as when read in place.
template <wb::DstFormat F> void check_write_back_format(std::mt19937_64 &rng) {
  constexpr uint32_t kRegs = wb::kRegs<F>;
  for (int iteration = 0; iteration < 3000; ++iteration) {
    const uint32_t wave = rng() % 2 ? 64 : 32;
    const bool sram_ecc = rng() % 2;
    const uint64_t exec = random_exec(rng, wave);
    std::vector<std::vector<uint8_t>> group_data(4);
    const uint8_t *groups[4];
    for (uint32_t g = 0; g < 4; ++g) {
      group_data[g].resize(16 * wb::kStride<F> + rng() % 64);
      for (auto &b : group_data[g])
        b = static_cast<uint8_t>(rng());
      groups[g] = group_data[g].data() + (group_data[g].size() - 16 * wb::kStride<F>);
    }
    // Registers 1..kRegs are the destination; 0 and kRegs + 1 must not change.
    std::vector<uint32_t> before((kRegs + 2) * 64);
    for (auto &dword : before)
      dword = static_cast<uint32_t>(rng());
    std::vector<uint32_t> expected = before, scalar = before, avx2 = before;
    reference_write_back<F>(expected.data() + 64, sram_ecc, exec, groups);
    const auto run = [&](std::vector<uint32_t> &file, auto isa) {
      uint8_t *dst[kRegs];
      for (uint32_t r = 0; r < kRegs; ++r)
        dst[r] = reinterpret_cast<uint8_t *>(file.data() + (r + 1) * 64);
      wb::write_back_as<F, decltype(isa)::value>(dst, sram_ecc, wave, exec, groups);
    };
    run(scalar, std::integral_constant<wb::Isa, wb::Isa::scalar>{});
    ASSERT_EQ(scalar, expected) << "scalar elem_size " << unsigned(F.elem_size) << " x"
                                << unsigned(F.num_elems) << " iteration " << iteration;
#if RJ_HOST_HAS_AVX2
    run(avx2, std::integral_constant<wb::Isa, wb::Isa::avx2>{});
    ASSERT_EQ(avx2, expected) << "avx2 elem_size " << unsigned(F.elem_size) << " x"
                              << unsigned(F.num_elems) << " iteration " << iteration;
#endif
  }
}

// Every format format_of() returns, and the DS spelling of b64.
TEST(VgprWritebackTest, WritebacksMatchThePerLaneReference) {
  std::mt19937_64 rng(0x3b17e8a5);
  check_write_back_format<wb::DstFormat{.elem_size = 4, .num_elems = 1}>(rng);
  check_write_back_format<wb::DstFormat{.elem_size = 4, .num_elems = 2}>(rng);
  check_write_back_format<wb::DstFormat{.elem_size = 8, .num_elems = 1}>(rng);
  check_write_back_format<wb::DstFormat{.elem_size = 4, .num_elems = 3}>(rng);
  check_write_back_format<wb::DstFormat{.elem_size = 4, .num_elems = 4}>(rng);
  check_write_back_format<wb::DstFormat{.elem_size = 1, .num_elems = 1}>(rng);
  check_write_back_format<wb::DstFormat{.elem_size = 1, .num_elems = 1, .sign_extend = true}>(rng);
  check_write_back_format<wb::DstFormat{.elem_size = 2, .num_elems = 1}>(rng);
  check_write_back_format<wb::DstFormat{.elem_size = 2, .num_elems = 1, .sign_extend = true}>(rng);
  check_write_back_format<wb::DstFormat{.elem_size = 1, .num_elems = 1, .d16_lo = true}>(rng);
  check_write_back_format<wb::DstFormat{.elem_size = 1, .num_elems = 1, .d16_hi = true}>(rng);
  check_write_back_format<wb::DstFormat{
      .elem_size = 1, .num_elems = 1, .sign_extend = true, .d16_lo = true}>(rng);
  check_write_back_format<wb::DstFormat{
      .elem_size = 1, .num_elems = 1, .sign_extend = true, .d16_hi = true}>(rng);
  check_write_back_format<wb::DstFormat{.elem_size = 2, .num_elems = 1, .d16_lo = true}>(rng);
  check_write_back_format<wb::DstFormat{.elem_size = 2, .num_elems = 1, .d16_hi = true}>(rng);
  check_write_back_format<wb::DstFormat{
      .elem_size = 2, .num_elems = 1, .sign_extend = true, .d16_lo = true}>(rng);
  check_write_back_format<wb::DstFormat{
      .elem_size = 2, .num_elems = 1, .sign_extend = true, .d16_hi = true}>(rng);
}

/// A copy of vector_complete()'s per-lane loop, for the lanes it writes: what
/// the routed path wrote for every load and returning atomic before the shared
/// writeback took them.
void reference_vector_complete(std::vector<uint32_t> &file, const std::vector<uint8_t> &response,
                               uint32_t wf_size, uint64_t lane_mask, uint32_t elem_size,
                               uint32_t num_elems, bool is_atomic, bool sign_extend, bool d16_hi,
                               bool d16_lo, bool sram_ecc) {
  const uint32_t stride = is_atomic ? elem_size : num_elems * elem_size;
  const uint32_t result_bytes = is_atomic ? elem_size : num_elems * elem_size;
  const uint32_t vgpr_count = std::max(1u, (result_bytes + 3u) / 4u);
  for (uint32_t lane = 0; lane < wf_size; ++lane) {
    if (!(lane_mask & (1ULL << lane)))
      continue;
    for (uint32_t i = 0; i < vgpr_count; ++i) {
      uint32_t val = 0;
      const uint32_t data_offset = lane * stride + i * 4;
      const uint32_t copy_size =
          is_atomic ? std::min(elem_size - i * 4, 4u) : std::min(elem_size, 4u);
      std::memcpy(&val, &response[data_offset], copy_size);
      if (sign_extend && i == 0 && elem_size < 4) {
        if (elem_size == 1)
          val = static_cast<uint32_t>(static_cast<int32_t>(static_cast<int8_t>(val)));
        else if (elem_size == 2)
          val = static_cast<uint32_t>(static_cast<int32_t>(static_cast<int16_t>(val)));
      }
      uint32_t &dword = file[(i + 1) * 64 + lane];
      if (copy_size <= 2 && (d16_hi || d16_lo)) {
        if (sram_ecc)
          val = d16_hi ? val << 16 : val & 0xFFFF;
        else
          val = d16_hi ? (dword & 0xFFFF) | (val << 16) : (dword & 0xFFFF0000) | (val & 0xFFFF);
      }
      dword = val;
    }
  }
}

// The routed path's run-time entry: for every element shape, extension and
// d16 flag a VectorMemState can carry, format_of() either declines it or
// write_back() leaves the register file the per-lane loop would.
TEST(VgprWritebackTest, RuntimeFormatsMatchTheRoutedPerLaneLoop) {
  std::mt19937_64 rng(0x51d0c0de);
  uint32_t taken = 0;
  for (const uint32_t elem_size : {1u, 2u, 3u, 4u, 8u, 12u, 16u})
    for (uint32_t num_elems = 1; num_elems <= 5; ++num_elems)
      for (const bool is_atomic : {false, true})
        for (uint32_t flags = 0; flags < 8; ++flags) {
          const bool sign_extend = flags & 1, d16_hi = flags & 2, d16_lo = flags & 4;
          const uint32_t result_elems = is_atomic ? 1 : num_elems;
          const std::optional<wb::DstFormat> format =
              wb::format_of(elem_size, result_elems, sign_extend, d16_hi, d16_lo);
          const uint32_t stride = elem_size * result_elems;
          const uint32_t vgpr_count = std::max(1u, (stride + 3u) / 4u);
          if (!format) {
            const bool narrow = elem_size < 4;
            EXPECT_TRUE(stride > wb::kMaxStride || elem_size == 3 ||
                        (narrow && (result_elems > 1 || (d16_hi && d16_lo))))
                << "declined elem_size " << elem_size << " x" << result_elems;
            continue;
          }
          ++taken;
          ASSERT_EQ(wb::regs(*format), vgpr_count);
          for (int iteration = 0; iteration < 200; ++iteration) {
            const uint32_t wave = rng() % 2 ? 64 : 32;
            const bool sram_ecc = rng() % 2;
            const uint64_t lanes = random_exec(rng, wave);
            std::vector<uint8_t> response(size_t{wave} * stride);
            for (auto &b : response)
              b = static_cast<uint8_t>(rng());
            std::vector<uint32_t> before((vgpr_count + 2) * 64);
            for (auto &dword : before)
              dword = static_cast<uint32_t>(rng());
            std::vector<uint32_t> expected = before, actual = before;
            reference_vector_complete(expected, response, wave, lanes, elem_size, num_elems,
                                      is_atomic, sign_extend, d16_hi, d16_lo, sram_ecc);
            uint8_t *dst[4];
            for (uint32_t r = 0; r < vgpr_count; ++r)
              dst[r] = reinterpret_cast<uint8_t *>(actual.data() + (r + 1) * 64);
            wb::write_back(*format, dst, sram_ecc, wave, lanes, response.data());
            ASSERT_EQ(actual, expected)
                << "elem_size " << elem_size << " x" << num_elems << (is_atomic ? " atomic" : "")
                << " flags " << flags << " wave " << wave << " iteration " << iteration;
          }
        }
  // Whole dwords up to b128, loads and atomics, and every narrow form.
  EXPECT_GT(taken, 100u);
}

} // namespace
