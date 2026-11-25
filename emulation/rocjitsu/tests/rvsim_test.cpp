// Copyright (c) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/risc_v/hart.h"
#include "simdojo/sim/clock_domain.h"
#include "simdojo/sim/component.h"
#include "simdojo/sim/simulation.h"

#include <gtest/gtest.h>

#include <bit>
#include <cstdint>
#include <iterator>
#include <limits>
#include <memory>

namespace {

/// Helper: build a single-hart topology, load a program, and run it.
class RvSimTest : public ::testing::Test {
protected:
  void SetUp() override {
    util::StringDiagnostic admission_diagnostic;
    simdojo::SimulationEngine::Config config{};
    config.max_ticks = 1000;
    config.num_threads = 1;
    engine_ = std::make_unique<simdojo::SimulationEngine>(config);

    auto maybe_clk = engine_->topology().add_clock_domain(
        "core_clk",
        /*frequency_hz=*/simdojo::TICKS_PER_SECOND, 0, admission_diagnostic.emitter());
    ASSERT_TRUE(maybe_clk.succeeded()) << admission_diagnostic.message();
    auto *clk = maybe_clk.value();

    auto root_ptr = std::make_unique<simdojo::CompositeComponent>("soc");
    auto hart_ptr = std::make_unique<rocjitsu::risc_v::Hart>("hart0", *clk);
    hart_ = static_cast<rocjitsu::risc_v::Hart *>(root_ptr->add_child(std::move(hart_ptr)));

    engine_->topology().set_root(std::move(root_ptr));
    ASSERT_TRUE(engine_->create(admission_diagnostic.emitter()).succeeded())
        << admission_diagnostic.message();
  }

  void load_and_run(const uint32_t *program, size_t num_words, uint64_t base_addr = 0) {
    hart_->memory().load_image(reinterpret_cast<const uint8_t *>(program),
                               num_words * sizeof(uint32_t), base_addr);
    hart_->state().pc = base_addr;

    engine_->run();
  }

  rocjitsu::risc_v::Hart *hart_ = nullptr;
  std::unique_ptr<simdojo::SimulationEngine> engine_;
};

TEST_F(RvSimTest, ArithmeticAndMemory) {
  //   addi x1, x0, 42      # x1 = 42
  //   addi x2, x0, 8       # x2 = 8
  //   add  x3, x1, x2      # x3 = 50
  //   addi x4, x0, 100     # x4 = 100
  //   sub  x5, x4, x3      # x5 = 50
  //   lui  x6, 0x10        # x6 = 0x10000
  //   sw   x3, 0(x6)       # mem[0x10000] = 50
  //   lw   x7, 0(x6)       # x7 = 50
  //   ebreak                # halt
  const uint32_t program[] = {
      0x02a00093, // addi x1, x0, 42
      0x00800113, // addi x2, x0, 8
      0x002081b3, // add  x3, x1, x2
      0x06400213, // addi x4, x0, 100
      0x403202b3, // sub  x5, x4, x3
      0x00010337, // lui  x6, 0x10
      0x00332023, // sw   x3, 0(x6)
      0x00032383, // lw   x7, 0(x6)
      0x00100073, // ebreak
  };
  load_and_run(program, std::size(program));

  const auto &s = hart_->state();
  EXPECT_TRUE(s.halted);
  EXPECT_EQ(s.pc, 0x24u); // ebreak at 0x20, PC advances to 0x24
  EXPECT_EQ(s.read_xreg(1), 42);
  EXPECT_EQ(s.read_xreg(2), 8);
  EXPECT_EQ(s.read_xreg(3), 50);
  EXPECT_EQ(s.read_xreg(4), 100);
  EXPECT_EQ(s.read_xreg(5), 50);
  EXPECT_EQ(s.read_xreg(6), 0x10000);
  EXPECT_EQ(s.read_xreg(7), 50);
  EXPECT_EQ(hart_->memory().read32(0x10000), 50u);
}

TEST_F(RvSimTest, FloatToIntegerUsesStaticAndDynamicRounding) {
  auto &state = hart_->state();
  // f1 and f2 hold the same tie in the two source formats.
  state.write_freg(1, rocjitsu::risc_v::nan_box(std::bit_cast<uint32_t>(2.5f)));
  state.write_freg(2, std::bit_cast<uint64_t>(2.5));
  struct Case {
    unsigned rm;
    unsigned frm;
    int expected;
  };
  constexpr Case cases[] = {
      {0, 3, 2}, // RNE
      {1, 3, 2}, // RTZ
      {2, 3, 2}, // RDN
      {3, 3, 3}, // RUP
      {4, 3, 3}, // RMM
      {7, 0, 2}, // Dynamic RNE rounds the tie to even.
      {7, 3, 3}, // Dynamic RUP
      {7, 4, 3}, // Dynamic RMM uses all three frm bits and rounds the tie away from zero.
  };
  for (unsigned format : {0u, 1u}) {
    for (unsigned dest = 0; dest < 4; ++dest) {
      for (const Case &c : cases) {
        SCOPED_TRACE(::testing::Message() << "format=" << format << " dest=" << dest
                                          << " rm=" << c.rm << " frm=" << c.frm);
        const uint32_t word =
            0xc00002d3 | (format << 25) | (dest << 20) | ((format + 1) << 15) | (c.rm << 12);
        hart_->memory().load_image(reinterpret_cast<const uint8_t *>(&word), sizeof(word), 0);
        state.pc = 0;
        state.write_csr(0x003, (c.frm << 5) | 8u); // DZ already accrued.
        ASSERT_TRUE(hart_->advance(0));
        EXPECT_EQ(state.read_xreg(5), c.expected);
        EXPECT_EQ(state.read_csr(0x003), (c.frm << 5) | 9u);
      }
    }
  }
}

TEST_F(RvSimTest, FloatToIntegerNearestModes) {
  auto &state = hart_->state();
  struct Case {
    double input;
    int64_t rne;
    int64_t rmm;
    unsigned flags;
  };
  constexpr Case cases[] = {
      {0x1.fffffffffffffp-2, 0, 0, 1}, // Just below the first tie.
      {0.5, 0, 1, 1},
      {0x1.0000000000001p-1, 1, 1, 1}, // Just above the first tie.
      {1.5, 2, 2, 1},
      {-2.5, -2, -3, 1},
      {0x1.0000000000001p52, 4503599627370497LL, 4503599627370497LL, 0},
  };
  for (unsigned rm : {0u, 4u}) {
    const uint32_t word = 0xc22082d3 | (rm << 12); // fcvt.l.d x5, f1
    hart_->memory().load_image(reinterpret_cast<const uint8_t *>(&word), sizeof(word), 0);
    for (const Case &c : cases) {
      SCOPED_TRACE(::testing::Message() << c.input << " rm=" << rm);
      state.pc = 0;
      state.write_freg(1, std::bit_cast<uint64_t>(c.input));
      state.write_csr(0x001, 0);
      ASSERT_TRUE(hart_->advance(0));
      EXPECT_EQ(state.read_xreg(5), rm == 0 ? c.rne : c.rmm);
      EXPECT_EQ(state.read_csr(0x001), c.flags);
    }
  }
}

TEST_F(RvSimTest, FloatToIntegerClipsAfterRoundingAndAccruesExceptions) {
  auto &state = hart_->state();
  struct Case {
    double input;
    unsigned dest; // W, WU, L, LU
    unsigned mode;
    int64_t expected;
    unsigned flags;
  };
  constexpr unsigned nx = 1, nv = 16;
  constexpr Case cases[] = {
      {-2.5, 0, 0, -2, nx},
      {-2.5, 0, 4, -3, nx},
      {-2.1, 2, 2, -3, nx},
      {-0.5, 1, 1, 0, nx},
      {-0.5, 3, 2, 0, nv},
      {-0.0, 3, 0, 0, 0},
      {2147483647.25, 0, 0, INT32_MAX, nx},
      {2147483647.5, 0, 0, INT32_MAX, nv},
      {-2147483648.5, 0, 0, INT32_MIN, nx},
      {-2147483648.5, 0, 4, INT32_MIN, nv},
      {4294967295.0, 1, 0, -1, 0}, // WU sign-extends to XLEN.
      {-0x1p63, 2, 0, INT64_MIN, 0},
      {0x1p63, 2, 0, INT64_MAX, nv},
      {0x1.fffffffffffffp63, 3, 0, -2048, 0},
      {0x1p64, 3, 0, -1, nv},
      {std::numeric_limits<double>::quiet_NaN(), 0, 0, INT32_MAX, nv},
      {-std::numeric_limits<double>::infinity(), 2, 0, INT64_MIN, nv},
      {std::numeric_limits<double>::infinity(), 3, 0, -1, nv},
  };
  for (const Case &c : cases) {
    SCOPED_TRACE(::testing::Message() << c.input << " dest=" << c.dest << " rm=" << c.mode);
    const uint32_t word = 0xc20082d3 | (c.dest << 20) | (c.mode << 12); // fcvt.*.d x5, f1
    hart_->memory().load_image(reinterpret_cast<const uint8_t *>(&word), sizeof(word), 0);
    state.pc = 0;
    state.write_freg(1, std::bit_cast<uint64_t>(c.input));
    state.write_csr(0x001, 8); // An earlier DZ survives either exception.
    ASSERT_TRUE(hart_->advance(0));
    EXPECT_EQ(state.read_xreg(5), c.expected);
    EXPECT_EQ(state.read_csr(0x001), 8u | c.flags);
  }
}

TEST_F(RvSimTest, FloatingPointCsrViewsPreserveSiblingFields) {
  auto &state = hart_->state();
  state.write_xreg(5, 0x161);
  const uint32_t program[] = {
      0x00329073, // csrw fcsr, x5
      0x00202373, // csrr x6, frm
      0x001023f3, // csrr x7, fflags
      0x00215073, // csrwi frm, 2
      0x00126073, // csrsi fflags, 4
      0x00302473, // csrr x8, fcsr
      0x00100073, // ebreak
  };
  load_and_run(program, std::size(program));
  EXPECT_EQ(state.read_xreg(6), 3);
  EXPECT_EQ(state.read_xreg(7), 1);
  EXPECT_EQ(state.read_xreg(8), 0x45);
}

TEST_F(RvSimTest, BranchAndJump) {
  //   addi x1, x0, 5       # x1 = 5
  //   addi x2, x0, 5       # x2 = 5
  //   beq  x1, x2, 8       # branch forward 8 bytes (skip next instr)
  //   addi x3, x0, 99      # x3 = 99 (should be skipped)
  //   addi x4, x0, 1       # x4 = 1 (branch target)
  //   ebreak                # halt
  const uint32_t program[] = {
      0x00500093, // addi x1, x0, 5
      0x00500113, // addi x2, x0, 5
      0x00208463, // beq  x1, x2, +8
      0x06300193, // addi x3, x0, 99
      0x00100213, // addi x4, x0, 1
      0x00100073, // ebreak
  };
  load_and_run(program, std::size(program));

  const auto &s = hart_->state();
  EXPECT_TRUE(s.halted);
  EXPECT_EQ(s.read_xreg(1), 5);
  EXPECT_EQ(s.read_xreg(3), 0); // skipped
  EXPECT_EQ(s.read_xreg(4), 1); // branch target reached
}

TEST_F(RvSimTest, JalAndJalr) {
  //   jal  x1, 8           # x1 = PC+4 = 4, jump to 0x8
  //   ebreak                # should be skipped
  //   addi x2, x0, 42      # x2 = 42  (jal target at 0x8)
  //   jalr x3, x1, 0       # x3 = PC+4 = 0xC+4 = 0x10, jump to x1 = 4
  //   ebreak                # halt (at 0x4, reached via jalr)
  const uint32_t program[] = {
      0x008000ef, // jal  x1, 8
      0x00100073, // ebreak  (addr 0x4 - jalr target)
      0x02a00113, // addi x2, x0, 42  (addr 0x8 - jal target)
      0x000081e7, // jalr x3, x1, 0  (addr 0xC)
  };
  load_and_run(program, std::size(program));

  const auto &s = hart_->state();
  EXPECT_TRUE(s.halted);
  EXPECT_EQ(s.read_xreg(1), 4); // return address from jal
  EXPECT_EQ(s.read_xreg(2), 42);
  EXPECT_EQ(s.read_xreg(3), 0x10); // return address from jalr
}

TEST_F(RvSimTest, ShiftsAndLogic) {
  //   addi  x1, x0, 1      # x1 = 1
  //   slli  x2, x1, 10     # x2 = 1024
  //   ori   x3, x0, 0xFF   # x3 = 255
  //   andi  x4, x3, 0x0F   # x4 = 15
  //   xori  x5, x3, 0xFF   # x5 = 0
  //   srli  x6, x2, 5      # x6 = 32
  //   ebreak
  const uint32_t program[] = {
      0x00100093, // addi  x1, x0, 1
      0x00a09113, // slli  x2, x1, 10
      0x0ff06193, // ori   x3, x0, 0xFF
      0x00f1f213, // andi  x4, x3, 0x0F
      0x0ff1c293, // xori  x5, x3, 0xFF
      0x00515313, // srli  x6, x2, 5
      0x00100073, // ebreak
  };
  load_and_run(program, std::size(program));

  const auto &s = hart_->state();
  EXPECT_EQ(s.read_xreg(1), 1);
  EXPECT_EQ(s.read_xreg(2), 1024);
  EXPECT_EQ(s.read_xreg(3), 255);
  EXPECT_EQ(s.read_xreg(4), 15);
  EXPECT_EQ(s.read_xreg(5), 0);
  EXPECT_EQ(s.read_xreg(6), 32);
}

} // namespace
