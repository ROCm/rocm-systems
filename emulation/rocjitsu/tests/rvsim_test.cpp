// Copyright (c) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/risc_v/hart.h"
#include "simdojo/sim/clock_domain.h"
#include "simdojo/sim/component.h"
#include "simdojo/sim/simulation.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>

namespace {

/// Helper: build a single-hart topology, load a program, and run it.
class RvSimTest : public ::testing::Test {
protected:
  void SetUp() override {
    simdojo::SimulationEngine::Config config{};
    config.max_ticks = 1000;
    config.num_threads = 1;
    engine_ = std::make_unique<simdojo::SimulationEngine>(config);

    auto *clk = engine_->topology().add_clock_domain("core_clk",
                                                     /*frequency_hz=*/simdojo::TICKS_PER_SECOND);

    auto root_ptr = std::make_unique<simdojo::CompositeComponent>("soc");
    auto hart_ptr = std::make_unique<rocjitsu::risc_v::Hart>("hart0", *clk);
    hart_ = static_cast<rocjitsu::risc_v::Hart *>(root_ptr->add_child(std::move(hart_ptr)));

    engine_->topology().set_root(std::move(root_ptr));
    engine_->build();
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
