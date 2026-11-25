// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Decoded gfx1150 dual slots: arithmetic, paired reads, masks and inherited FP status.
#include "compute_unit_test_util.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3_5/execution_backend.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/execution_backend.h"
#include "rocjitsu/isa/isa_family.h"
#include "rocjitsu/vm/amdgpu/arithmetic_policy.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include <array>
#include <bit>
#include <gtest/gtest.h>

namespace rocjitsu::amdgpu {
namespace {

// Own a decoded Wave32/Wave64 machine with explicitly selected arithmetic semantics.
struct Machine {
  GpuMemory memory{"dual_memory"};
  L2Cache cache{"dual_cache"};
  std::unique_ptr<ComputeUnitCore> cu;
  std::unique_ptr<Decoder> decoder;
  Wavefront *wave = nullptr;
  void initialize(ArithmeticSemantics semantics, int lanes) {
    ComputeUnitCore::Config config{};
    config.arch = rocjitsu::IsaFamily::Rdna3_5;
    config.arithmetic = {semantics};
    config.num_wf_slots = 1;
    config.sgprs_per_wf = 104;
    config.vgprs_per_wf = 8;
    config.lds_size_kb = 64;
    ASSERT_NO_FATAL_FAILURE(test::initialize_compute_unit(cu, "dual", config, &memory, &cache));
    decoder = Decoder::create(config.arch);
    ASSERT_NE(decoder, nullptr);
    wave = cu->dispatch_wf(0, 0, 104, 8, lanes);
    ASSERT_NE(wave, nullptr);
  }
  uint32_t get(int reg, int lane) { return cu->read_vgpr(wave->vgpr_alloc().base + reg, lane); }
  void put(int reg, int lane, uint32_t bits) {
    cu->write_vgpr(wave->vgpr_alloc().base + reg, lane, bits);
  }
  void execute(const std::array<uint32_t, 3> &words) {
    auto instruction = decoder->decode_window(words);
    ASSERT_TRUE(instruction.succeeded());
    ASSERT_TRUE(cu->execute_instruction(instruction.value().get(), *wave).succeeded());
  }
};

// Encode X(v2,v1) and Y(v0,v3), whose corresponding sources use different banks.
// X's even destination may overwrite Y's v0 source; Y always writes odd v5.
std::array<uint32_t, 3> pair(int x, int y, int destination) {
  return {0xc8000000U | (uint32_t(x) << 22) | (uint32_t(y) << 17) | (1U << 9) | 258U,
          (uint32_t(destination) << 24) | (2U << 17) | (3U << 9) | 256U, 0x3f000000U};
}

// Every legal slot executes with the old paired inputs; Wave64 retains its no-op contract.
TEST(Gfx1150Dual, ArithmeticAndPairReads) {
  ScopedIsaExecutionBackend backend(&rdna3_5::execution_backend());
  const int operations[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16, 17, 18};
  for (int lanes : {32, 64}) {
    for (auto semantics : {ArithmeticSemantics::Compatibility, ArithmeticSemantics::Loose}) {
      Machine m;
      ASSERT_NO_FATAL_FAILURE(m.initialize(semantics, lanes));
      for (int op : operations) {
        for (int aliasing_offset : {0, 4}) {
          for (uint32_t exec_mask : {0U, 0xa5a5f0f0U, UINT32_MAX}) {
            SCOPED_TRACE(::testing::Message() << lanes << ':' << int(semantics) << ':' << op << ':'
                                              << aliasing_offset << ':' << exec_mask);
            for (int reg = 0; reg < 6; ++reg) {
              for (int lane = 0; lane < lanes; ++lane) {
                m.put(reg, lane, std::bit_cast<uint32_t>(float(reg + 1)));
              }
            }
            m.wave->set_exec(exec_mask);
            m.wave->set_vcc_mask(0xaaaaaaaaU);
            m.wave->set_mode_raw(0xf0);
            m.wave->set_trapsts(0x100);
            ASSERT_NO_FATAL_FAILURE(
                m.execute(pair(op < 16 ? op : 8, op < 16 ? 8 : op, aliasing_offset)));
            for (int lane = 0; lane < lanes; ++lane) {
              const uint32_t old_x = std::bit_cast<uint32_t>(float(aliasing_offset + 1));
              uint32_t x = std::bit_cast<uint32_t>(3.0f), y = 0x3f800000U;
              switch (op) {
              case 0:
                x = std::bit_cast<uint32_t>(6.0f + float(aliasing_offset + 1));
                break;
              case 1:
                x = std::bit_cast<uint32_t>(6.5f);
                break;
              case 2:
                x = std::bit_cast<uint32_t>(3.5f);
                break;
              case 3:
              case 7:
                x = std::bit_cast<uint32_t>(6.0f);
                break;
              case 4:
                x = std::bit_cast<uint32_t>(5.0f);
                break;
              case 5:
                x = std::bit_cast<uint32_t>(1.0f);
                break;
              case 6:
                x = std::bit_cast<uint32_t>(-1.0f);
                break;
              case 9:
                x = std::bit_cast<uint32_t>(lane & 1 ? 2.0f : 3.0f);
                break;
              case 11:
                x = std::bit_cast<uint32_t>(2.0f);
                break;
              case 12:
                x = std::bit_cast<uint32_t>(4.25f + float(aliasing_offset + 1));
                break;
              case 13:
                x = std::bit_cast<uint32_t>(6.0f + float(aliasing_offset + 1));
                break;
              case 16:
                y = 0x3f800000U + 0x40800000U;
                break;
              case 17:
                y = 0x40800000U;
                break;
              case 18:
                y = 0x3f800000U & 0x40800000U;
                break;
              default:
                break;
              }
              const bool active = lanes == 32 && (exec_mask & (1U << (lane % 32)));
              EXPECT_EQ(m.get(aliasing_offset, lane), active ? x : old_x);
              EXPECT_EQ(m.get(5, lane), active ? y : std::bit_cast<uint32_t>(6.0f));
            }
            EXPECT_EQ(m.wave->trapsts(), 0x100U);
            EXPECT_EQ(m.wave->vcc(), 0xaaaaaaaaU);
          }
        }
      }
    }
  }
}

// Status follows EXEC even when destination stores are suppressed; DX10 disables causes.
TEST(Gfx1150Dual, StatusWithSuppressedWrites) {
  ScopedIsaExecutionBackend backend(&rdna3_5::execution_backend());
  for (auto semantics : {ArithmeticSemantics::Compatibility, ArithmeticSemantics::Loose}) {
    Machine m;
    ASSERT_NO_FATAL_FAILURE(m.initialize(semantics, 32));
    for (int opcode : {0, 4}) { // FMAC and ADD share this INEXACT witness.
      for (bool suppress : {false, true}) {
        for (bool dx10 : {false, true}) {
          for (uint32_t rounding = 0; rounding < 4; ++rounding) {
            for (uint32_t exec_mask : {0U, 1U << 31, UINT32_MAX}) {
              for (int reg = 0; reg < 6; ++reg) {
                for (int lane = 0; lane < 32; ++lane) {
                  m.put(reg, lane, 0x3f800000U);
                }
              }
              m.put(1, 31, 0x33800001U); // More than half an ULP at one: INEXACT in every rounding.
              m.wave->set_exec(exec_mask);
              m.wave->set_vgpr_write_mask(suppress ? 0 : ~0ULL);
              m.wave->set_mode_raw(0xf0 | rounding | (dx10 ? 0x100 : 0));
              m.wave->set_trapsts(0x100);
              m.wave->clear_pending_alu_causes();
              ASSERT_NO_FATAL_FAILURE(m.execute(pair(opcode, 8, 4)));
              const uint32_t causes = semantics == ArithmeticSemantics::Compatibility && !dx10 &&
                                              (exec_mask & (1U << 31))
                                          ? 32
                                          : 0;
              EXPECT_EQ(m.wave->trapsts(), 0x100U | causes);
              EXPECT_EQ(m.wave->pending_alu_causes(), causes);
              const uint32_t expected =
                  (rounding == 0 || rounding == 1) ? 0x3f800001U : 0x3f800000U;
              if (semantics == ArithmeticSemantics::Loose && !suppress &&
                  (exec_mask & (1U << 31))) {
                EXPECT_NEAR(std::bit_cast<float>(m.get(4, 31)), 1.0f, 1e-6f);
              } else {
                EXPECT_EQ(m.get(4, 31),
                          !suppress && (exec_mask & (1U << 31)) ? expected : 0x3f800000U);
              }
            }
          }
        }
      }
    }
  }
}
} // namespace
} // namespace rocjitsu::amdgpu
