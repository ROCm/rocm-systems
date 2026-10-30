// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for explicit guest register reads/writes and MODE-field updates.

#include "scalar/integer_helpers.h"

#include <array>
#include <cstring>

namespace {

// Encode a hardware-register ID, field offset, and field width.
uint16_t descriptor(int id, int offset, int width) {
  return uint16_t(id | (offset << 6) | ((width - 1) << 11));
}

// Replace writable bits of a selected register field, preserving all other bits.
uint32_t insert(uint32_t old, uint32_t source, int offset, int width, uint32_t writable) {
  for (int bit = 0; bit < width && bit + offset < 32; ++bit) {
    uint32_t target = 1U << (bit + offset);
    if (writable & target) {
      old = (old & ~target) | (((source >> bit) & 1) ? target : 0);
    }
  }
  return old;
}

// Assemble two adjacent scalar-register words into a 64-bit value.
uint64_t pair(const std::array<uint32_t, 64> &v, int low) {
  return v[low] | (uint64_t(v[low + 1]) << 32);
}

// Call the gfx11/gfx12 GETREG API with the supplied register bank and optional output snapshot.
int read(bool rdna4, uint64_t flags, uint64_t instruction_flags, uint16_t hwreg, uint32_t *d,
         const std::array<uint32_t, 64> &v, std::array<uint32_t, 5> *snapshot = nullptr,
         unsigned report = 31) {
  if (rdna4) {
    return rj_arith_s_getreg_b32_gfx12(flags, instruction_flags, d, hwreg, v[1], v[2], v[4], v[5],
                                       v[6], v[10], v[11], v[12], v[15], v[16], v[17], v[18], v[19],
                                       pair(v, 20), v[23], v[24], v[28], pair(v, 29),
                                       snapshot && (report & 1) ? &(*snapshot)[0] : nullptr,
                                       snapshot && (report & 2) ? &(*snapshot)[1] : nullptr,
                                       snapshot && (report & 4) ? &(*snapshot)[2] : nullptr,
                                       snapshot && (report & 8) ? &(*snapshot)[3] : nullptr,
                                       snapshot && (report & 16) ? &(*snapshot)[4] : nullptr);
  }
  return rj_arith_s_getreg_b32(flags, instruction_flags, d, hwreg, v[1], v[2], v[3], v[5], v[6],
                               v[7], pair(v, 8), v[13], v[15], pair(v, 16), pair(v, 18),
                               pair(v, 20), v[23], v[24], v[25], v[26], v[27], v[28], pair(v, 29));
}

// Check architectural IDs and writable fields, sampling field widths and
// privilege together rather than all 65536 selectors for each CPU and semantics.
template <bool Gfx12, typename Call> void check_setreg(Call call) {
  for (auto cpu : rocjitsu::arith::test::supported_cpu_levels()) {
    for (auto semantics : rocjitsu::arith::test::semantics_modes()) {
      std::minstd_rand random;
      for (int id = 0; id < 64; ++id) {
        for (int scenario = 0; scenario < 8; ++scenario) {
          std::array<uint32_t, 5> state;
          for (auto &v : state) {
            v = rocjitsu::arith::test::random_word(random);
          }
          auto expected = state;
          uint64_t scratch = 0x123456789abcdef0ULL;
          int offset = (scenario * 7) % 32, width = 1 + (scenario * 11) % 32;
          uint32_t source = rocjitsu::arith::test::random_word(random);
          bool privileged = scenario & 1;
          int target = -1;
          uint32_t writable = 0;
          bool known = false, unsupported = false;
          if constexpr (Gfx12) {
            for (int value :
                 {1, 2, 4, 5, 6, 10, 11, 12, 15, 16, 17, 18, 19, 20, 21, 23, 24, 28, 29, 30}) {
              known |= id == value;
            }
            if (id == 1) {
              target = 0;
              writable = 0x098000ffU;
            }
            if (id == 4) {
              target = 1;
              writable = privileged ? 0x47e07U : 0;
            }
            if (id == 17) {
              target = 2;
              writable = privileged ? 0xc0000fffU : 0;
            }
            if (id == 18) {
              target = 3;
              writable = 0xc000007fU;
            }
            if (id == 19) {
              target = 4;
              writable = privileged ? 0x3ffU : 0;
            }
            unsupported = privileged && (id == 20 || id == 21);
          } else {
            for (int value : {1,  2,  3,  5,  6,  7,  8,  9,  13, 14, 15, 16,
                              17, 18, 19, 20, 21, 23, 24, 25, 26, 27, 28, 29}) {
              known |= id == value;
            }
            if (id == 1) {
              target = 0;
              writable = ((1U << 20) - 1) | (1U << 21) | (1U << 23) | (1U << 27);
            }
            if (id == 2) {
              target = 1;
              writable = privileged ? 31U | (1U << 8) | (1U << 13) | (1U << 18) | (1U << 20) |
                                          (1U << 21) | (1U << 25) | (1U << 27) | (1U << 29)
                                    : 0;
            }
            if (id == 3) {
              target = 2;
              writable = 511U | (511U << 10) | (1U << 20);
            }
            unsupported = id == 14 || (privileged && (id == 20 || id == 21));
          }
          if (target >= 0) {
            expected[target] = insert(state[target], source, offset, width, writable);
          }
          int wanted = unsupported && semantics != RJ_ARITH_SEMANTICS_LOOSE
                           ? RJ_ARITH_ERROR_UNSUPPORTED_REGISTER_STATE
                       : known ? RJ_ARITH_SUCCESS
                               : RJ_ARITH_ERROR_INVALID_FLAGS;
          ASSERT_EQ(call(cpu | semantics, 0, descriptor(id, offset, width), source,
                         privileged ? 32U : 0U, state, &scratch),
                    wanted)
              << id << '/' << semantics;
          EXPECT_EQ(state, expected);
          EXPECT_EQ(scratch, 0x123456789abcdef0ULL);
        }
      }
    }
  }
  std::array<uint32_t, 5> state = {123, 234, 345, 456, 567};
  auto before = state;
  uint64_t scratch = 789;
  rocjitsu::arith::test::expect_invalid_flags(
      [&](auto f, auto m) {
        int error = call(f, m, descriptor(1, 0, 32), 0, 0, state, &scratch);
        EXPECT_EQ(state, before);
        EXPECT_EQ(scratch, 789U);
        return error;
      },
      0);
}

// Check guest register-field reads and unsupported-state errors for the supplied entry point.
template <bool Gfx12> void check_getreg() {
  std::array<bool, 64> known{};
  if constexpr (Gfx12) {
    for (int id : {1, 2, 4, 5, 6, 10, 11, 12, 15, 16, 17, 18, 19, 20, 21, 23, 24, 28, 29, 30}) {
      known[id] = true;
    }
  } else {
    for (int id :
         {1, 2, 3, 5, 6, 7, 8, 9, 13, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 26, 27, 28, 29}) {
      known[id] = true;
    }
  }
  for (auto cpu : rocjitsu::arith::test::supported_cpu_levels()) {
    for (auto semantics : rocjitsu::arith::test::semantics_modes()) {
      std::minstd_rand random;
      for (int id = 0; id < 64; ++id) {
        for (int scenario = 0; scenario < 8; ++scenario) {
          std::array<uint32_t, 64> values;
          for (auto &v : values) {
            v = rocjitsu::arith::test::random_word(random);
          }
          int offset = scenario * 7 % 32, width = 1 + scenario * 11 % 32;
          uint32_t expected = 0;
          for (int bit = 0; bit < width && offset + bit < 32; ++bit) {
            expected |= ((values[id] >> (offset + bit)) & 1) << bit;
          }
          uint32_t d = 123;
          ASSERT_EQ(read(Gfx12, cpu | semantics, 0, descriptor(id, offset, width), &d, values),
                    known[id] ? RJ_ARITH_SUCCESS : RJ_ARITH_ERROR_INVALID_FLAGS);
          EXPECT_EQ(d, known[id] ? expected : 123U);
          if constexpr (Gfx12) {
            if (id == 12) {
              std::array<uint32_t, 5> snapshot = {123, 234, 345, 456, 567}, before = snapshot;
              d = 123;
              ASSERT_EQ(read(true, cpu | semantics, 0, descriptor(id, offset, width), &d, values,
                             &snapshot),
                        semantics == RJ_ARITH_SEMANTICS_LOOSE
                            ? RJ_ARITH_SUCCESS
                            : RJ_ARITH_ERROR_UNSUPPORTED_REGISTER_STATE);
              EXPECT_EQ(snapshot, before);
              EXPECT_EQ(d, semantics == RJ_ARITH_SEMANTICS_LOOSE ? expected : 123U);
            }
          }
        }
      }
    }
  }
  std::array<uint32_t, 64> values{};
  uint32_t d = 123;
  rocjitsu::arith::test::expect_invalid_flags(
      [&](auto f, auto m) {
        int error = read(Gfx12, f, m, descriptor(1, 0, 32), &d, values);
        EXPECT_EQ(d, 123U);
        return error;
      },
      0);
}

// Check updates to one guest MODE field with preservation of the other bits for the supplied
// entry point.
template <typename Call> void check_mode(Call call, int shift) {
  for (auto cpu : rocjitsu::arith::test::supported_cpu_levels()) {
    for (auto semantics : rocjitsu::arith::test::semantics_modes()) {
      std::minstd_rand random;
      for (int sample = 0; sample < 128; ++sample) {
        uint32_t mode = rocjitsu::arith::test::random_word(random), before = mode;
        uint16_t immediate = uint16_t(rocjitsu::arith::test::random_word(random));
        ASSERT_EQ(call(cpu | semantics, 0, immediate, &mode), RJ_ARITH_SUCCESS);
        EXPECT_EQ(mode, (before & ~(15U << shift)) | ((immediate & 15U) << shift));
        ASSERT_EQ(call(cpu | semantics, 0, immediate, nullptr), RJ_ARITH_SUCCESS);
      }
    }
  }
  uint32_t mode = 123;
  rocjitsu::arith::test::expect_invalid_flags(
      [&](auto f, auto m) {
        int error = call(f, m, 0, &mode);
        EXPECT_EQ(mode, 123U);
        return error;
      },
      0);
}

} // namespace

// Check guest register-field reads and unsupported-state errors.
TEST(ScalarGetregB32Gfx11, Arithmetic) { check_getreg<false>(); }

// Check guest register-field reads and unsupported-state errors.
TEST(ScalarGetregB32Gfx12, Arithmetic) { check_getreg<true>(); }

// Check updates to one guest MODE field with preservation of the other bits.
TEST(s_round_mode, Arithmetic) { check_mode(rj_arith_s_round_mode, 0); }

// Check updates to one guest MODE field with preservation of the other bits.
TEST(s_denorm_mode, Arithmetic) { check_mode(rj_arith_s_denorm_mode, 4); }

// Check explicit guest register reads/writes and MODE-field updates.
TEST(ScalarSetregB32Gfx11, Arithmetic) {
  check_setreg<false>(
      [](auto f, auto m, auto selector, auto a, auto privilege, auto &state, auto scratch) {
        return rj_arith_s_setreg_b32(f, m, selector, a, privilege, &state[0], &state[1], &state[2],
                                     &state[3], scratch);
      });
}

// Check explicit guest register reads/writes and MODE-field updates.
TEST(ScalarSetregImm32B32Gfx11, Arithmetic) {
  check_setreg<false>(
      [](auto f, auto m, auto selector, auto a, auto privilege, auto &state, auto scratch) {
        return rj_arith_s_setreg_imm32_b32(f, m, selector, a, privilege, &state[0], &state[1],
                                           &state[2], &state[3], scratch);
      });
}

// Check explicit guest register reads/writes and MODE-field updates.
TEST(ScalarSetregB32Gfx12, Arithmetic) {
  check_setreg<true>(
      [](auto f, auto m, auto selector, auto a, auto privilege, auto &state, auto scratch) {
        return rj_arith_s_setreg_b32_gfx12(f, m, selector, a, privilege, &state[0], &state[1],
                                           &state[2], &state[3], &state[4], scratch);
      });
}

// Check explicit guest register reads/writes and MODE-field updates.
TEST(ScalarSetregImm32B32Gfx12, Arithmetic) {
  check_setreg<true>(
      [](auto f, auto m, auto selector, auto a, auto privilege, auto &state, auto scratch) {
        return rj_arith_s_setreg_imm32_b32_gfx12(f, m, selector, a, privilege, &state[0], &state[1],
                                                 &state[2], &state[3], &state[4], scratch);
      });
}
