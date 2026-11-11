// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file wave_debug_test.cpp
/// @brief Wave-level KFD debugger groundwork, exercised through the compute unit.

#include "compute_unit_test_util.h"
#include "rocjitsu/isa/arch/amdgpu/cdna3/isa.h"
#include "rocjitsu/isa/arch/amdgpu/cdna4/isa.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/builders.h"
#include "rocjitsu/isa/arch/amdgpu/rdna4/isa.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/isa/isa_traits.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/pm4/pm4.h"
#include "rocjitsu/vm/amdgpu/spi.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "rocjitsu/vm/plugins/execution_plugin_group.h"

#include "legacy_gpu_memory_fixture.h"
#include "rocjitsu/kmd/linux/cwsr.h"
#include "rocjitsu/kmd/linux/kfd_process.h"

#include "rocjitsu/vm/amdgpu/matrix_coexecution.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <span>
#include <vector>

namespace {

using namespace rocjitsu;

constexpr uint32_t SGPRS_PER_WF = 106;
constexpr uint32_t VGPRS_PER_WF = 64;
constexpr uint32_t kKernelAddr = 0x1000;
constexpr uint32_t kTrapHandlerAddr = 0x2000;
constexpr uint32_t kSTrapBreakpoint = 0xBF920001u; // s_trap 1 (rocm-dbgapi breakpoint)
constexpr uint32_t kSTrapSeven = 0xBF92AB07u;      // s_trap 0xab07 (trap id is low 8 bits)
constexpr uint32_t kGfx12STrap = 0xBF900003u;      // s_trap 3 on gfx12+
constexpr uint32_t kSEndpgm = 0xBF810000u;         // s_endpgm

struct WaveDebugFixture {
  test::LegacyGpuMemoryFixture gpu_mem;
  amdgpu::L2Cache l2;
  std::unique_ptr<amdgpu::ComputeUnitCore> cu;
  amdgpu::Wavefront *wf = nullptr;

  explicit WaveDebugFixture(rj_code_arch_t arch = ROCJITSU_CODE_ARCH_CDNA4,
                            uint32_t num_wf_slots = 1,
                            uint32_t quantum = amdgpu::ComputeUnitCore::Config{}.functional_quantum,
                            uint32_t async_helpers = 0)
      : gpu_mem("wave_debug_mem"), l2("wave_debug_l2") {
    amdgpu::ComputeUnitCore::Config cfg{};
    cfg.arch = arch;
    cfg.functional_quantum = quantum;
    if (async_helpers)
      cfg.async_resources =
          std::make_shared<amdgpu::matrix_coexecution::ExecutionResources>(async_helpers);
    cfg.num_wf_slots = num_wf_slots;
    cfg.sgprs_per_wf = SGPRS_PER_WF;
    cfg.vgprs_per_wf = VGPRS_PER_WF;
    cfg.lds_size_kb = 64;
    rocjitsu::test::initialize_compute_unit(cu, "cu_wave_debug", cfg, &gpu_mem, &l2);
    if (!cu)
      return;
    l2.set_gpu_vm(&gpu_mem.gpu_vm());
    cu->set_gpu_vm(&gpu_mem.gpu_vm());
  }

  amdgpu::Wavefront *dispatch(uint64_t pc) {
    wf = cu->dispatch_wf(0, pc, SGPRS_PER_WF, VGPRS_PER_WF);
    return wf;
  }
};

TEST(WaveDebugTest, TrapDrainsBeforeEnteringPrivilegedHandler) {
  for (auto arch : {ROCJITSU_CODE_ARCH_CDNA1, ROCJITSU_CODE_ARCH_CDNA2, ROCJITSU_CODE_ARCH_CDNA3,
                    ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_RDNA1,
                    ROCJITSU_CODE_ARCH_RDNA2, ROCJITSU_CODE_ARCH_RDNA3, ROCJITSU_CODE_ARCH_RDNA3_5,
                    ROCJITSU_CODE_ARCH_RDNA4}) {
    SCOPED_TRACE(arch);
    WaveDebugFixture f(arch);
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
    const bool opcode16 = arch == ROCJITSU_CODE_ARCH_CDNA5 || arch == ROCJITSU_CODE_ARCH_RDNA3 ||
                          arch == ROCJITSU_CODE_ARCH_RDNA3_5 || arch == ROCJITSU_CODE_ARCH_RDNA4;
    f.gpu_mem.write32(kKernelAddr, opcode16 ? kGfx12STrap : kSTrapSeven);
    f.cu->set_trap_handler_resolver([](const amdgpu::Wavefront &) {
      return amdgpu::ComputeUnitCore::TrapHandlerConfig{kTrapHandlerAddr, 0x3000, true};
    });
    auto *wave = f.dispatch(kKernelAddr);
    ASSERT_NE(wave, nullptr);
    wave->wait_counters().increment(amdgpu::WaitCounterType::VMCNT);
    f.cu->step();
    EXPECT_EQ(wave->pc, kKernelAddr);
    EXPECT_FALSE(wave->in_trap_handler());
    wave->release_wait_counter(amdgpu::WaitCounterType::VMCNT);
    f.cu->step();
    ASSERT_EQ(wave->pc, kTrapHandlerAddr);
  }
}

TEST(WaveDebugTest, Gfx10TrapDrainsBeforeEnteringPrivilegedHandler) {
  for (auto arch : {ROCJITSU_CODE_ARCH_RDNA1, ROCJITSU_CODE_ARCH_RDNA2}) {
    WaveDebugFixture f(arch);
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
    f.gpu_mem.write32(kKernelAddr, kSTrapSeven);
    f.gpu_mem.write32(kTrapHandlerAddr, 0xb90a0142u);     // s10 = STATUS.PRIV
    f.gpu_mem.write32(kTrapHandlerAddr + 4, 0xbef003aau); // ttmp4 = 42
    f.gpu_mem.write32(kTrapHandlerAddr + 8, 0xbe80226cu); // s_rfe_b64 ttmp[0:1]
    f.cu->set_trap_handler_resolver([](const amdgpu::Wavefront &) {
      return amdgpu::ComputeUnitCore::TrapHandlerConfig{kTrapHandlerAddr, 0x3000, true};
    });
    auto *wave = f.dispatch(kKernelAddr);
    ASSERT_NE(wave, nullptr);
    wave->wait_counters().increment(amdgpu::WaitCounterType::VMCNT);
    f.cu->step();
    EXPECT_EQ(wave->pc, kKernelAddr);
    EXPECT_FALSE(wave->in_trap_handler());
    wave->release_wait_counter(amdgpu::WaitCounterType::VMCNT);
    f.cu->step();
    ASSERT_EQ(wave->pc, kTrapHandlerAddr);
    EXPECT_EQ((wave->ttmp(1) >> 16) & 0xff, 7u);
    f.cu->step();
    EXPECT_EQ(f.cu->read_sgpr(wave->sgpr_alloc().base + 10), 1u);
    f.cu->step();
    EXPECT_EQ(wave->ttmp(4), 42u);
    f.cu->step();
    EXPECT_EQ(wave->status_raw() & (1u << 5), 0u);
    EXPECT_FALSE(wave->in_trap_handler());
  }
}

TEST(WaveDebugTest, Gfx10CodeEndRaisesIllegalInstruction) {
  WaveDebugFixture f(ROCJITSU_CODE_ARCH_RDNA1);
  ASSERT_FALSE(::testing::Test::HasFatalFailure());
  f.gpu_mem.write32(kKernelAddr, 0xbf9f0000u);
  uint32_t interrupts = 0;
  f.cu->set_illegal_inst_handler([&](amdgpu::Wavefront &wave) {
    ++interrupts;
    wave.set_debug_halted(true);
    return true;
  });
  auto *wave = f.dispatch(kKernelAddr);
  ASSERT_NE(wave, nullptr);
  f.cu->step();
  EXPECT_EQ(interrupts, 1u);
  EXPECT_EQ(wave->pc, kKernelAddr);
  EXPECT_TRUE(wave->debug_halted());
}

TEST(WaveDebugTest, Rdna2PrivateMemoryProbeUsesTransferredSpans) {
  constexpr uint64_t page = amdgpu::GpuMemory::PAGE_SIZE, base = 0x40000;
  constexpr uint32_t pid = 9;
  for (bool clipped : {true, false}) {
    SCOPED_TRACE(clipped);
    WaveDebugFixture fx(ROCJITSU_CODE_ARCH_RDNA2);
    std::vector<uint8_t> kernel(page), data(page);
    KfdProcess::PageTable pt;
    util::DistributedSharedMutex mutex;
    pt[kKernelAddr >> amdgpu::GpuMemory::PAGE_SHIFT] = {kernel.data(), amdgpu::Mtype::RW};
    pt[base >> amdgpu::GpuMemory::PAGE_SHIFT] = {data.data(), amdgpu::Mtype::RW};
    fx.gpu_mem.register_process(pid, &pt, &mutex);
    fx.gpu_mem.write32(kKernelAddr, 0xdc344000, pid); // scratch_load_dwordx2 v[8:9], v0, off
    fx.gpu_mem.write32(kKernelAddr + 4, 0x087d0000, pid);
    auto *wave = fx.dispatch(kKernelAddr);
    ASSERT_NE(wave, nullptr);
    const auto width = wave->wf_size(), lane = clipped ? width - 1 : 0u;
    wave->set_process_id(pid);
    wave->set_exec(uint64_t{1} << lane);
    wave->set_scratch_base(base);
    wave->set_scratch_lane_size(page / width + (clipped ? 0 : 4));
    fx.cu->write_vgpr(wave->vgpr_alloc().base, lane, page / width - 4);
    fx.gpu_mem.write32(base + page - 4, 0x76543210, pid);
    ASSERT_FALSE(fx.gpu_mem.is_mapped(base + page, pid));
    fx.cu->set_debug_active(true);
    fx.l2.set_backing_memory(&fx.gpu_mem);
    std::vector<uint64_t> faults;
    fx.cu->set_memory_violation_handler(
        [&](amdgpu::Wavefront &, uint64_t address, bool write, uint64_t pc) {
          EXPECT_FALSE(write);
          EXPECT_EQ(pc, kKernelAddr);
          faults.push_back(address);
          return true;
        });
    fx.cu->step();
    if (clipped) {
      EXPECT_TRUE(faults.empty());
      EXPECT_EQ(wave->debug_read_vgpr(8, lane), 0x76543210u);
      EXPECT_EQ(wave->debug_read_vgpr(9, lane), 0u);
      EXPECT_EQ(wave->trapsts(), 1u << 8);
    } else {
      // Lane zero's second DWORD starts on the missing page even though eight
      // contiguous bytes at its first address would fit within the mapped page.
      EXPECT_EQ(faults, std::vector<uint64_t>{base + page});
      EXPECT_EQ(wave->trapsts(), 0u);
    }
    fx.gpu_mem.unregister_process(pid);
  }
}

TEST(WaveDebugTest, Gfx10DeclinedGraphicsAndComputeControlMessagesReportMissingBackend) {
  for (auto arch : {ROCJITSU_CODE_ARCH_RDNA1, ROCJITSU_CODE_ARCH_RDNA2}) {
    WaveDebugFixture fx(arch);
    fx.cu->set_sendmsg_handler(
        [](amdgpu::Wavefront &, uint32_t id) { return id == 1 || id == 10; });
    auto *wave = fx.dispatch(kKernelAddr);
    auto decoder = Decoder::create(arch);
    for (uint32_t id : {1, 2, 3, 4, 5, 6, 7, 9, 10}) {
      SCOPED_TRACE(testing::Message() << arch << " message " << id);
      const std::array words{0xbf900000u | id};
      auto inst = decoder->decode_window(words);
      ASSERT_TRUE(inst.succeeded());
      EXPECT_EQ(inst.value()->mnemonic(), "s_sendmsg");
      std::string detail;
      const bool failed = fx.cu->execute_instruction(inst.value().get(), *wave, &detail).failed();
      EXPECT_EQ(failed, id != 1 && id != 10);
      if (failed) {
        EXPECT_EQ(wave->instruction_execution_error(),
                  amdgpu::InstructionExecutionError::MissingExecutionBackend);
        EXPECT_NE(detail.find(id >= 4 && id <= 6 ? "compute backend" : "graphics backend"),
                  std::string::npos);
      }
    }
  }
}

TEST(WaveDebugTest, Gfx11ShaderCyclesWrapsAtTwentyBitsOnAFullRegisterRead) {
  for (auto arch : {ROCJITSU_CODE_ARCH_RDNA3, ROCJITSU_CODE_ARCH_RDNA3_5}) {
    WaveDebugFixture fx(arch);
    auto *wave = fx.dispatch(kKernelAddr);
    auto decoder = Decoder::create(arch);
    const auto words = rdna3::build_sopk(17, {.simm16 = 0xf81d, .sdst = 4});
    auto inst = decoder->decode_window(words);
    ASSERT_TRUE(inst.succeeded());
    for (uint64_t cycle : {0xfffffull, 0x100000ull, 0x123456ull}) {
      fx.cu->restore_cycle_count(cycle);
      ASSERT_TRUE(fx.cu->execute_instruction(inst.value().get(), *wave).succeeded());
      EXPECT_EQ(wave->debug_read_sgpr(4), cycle & 0xfffffu);
    }
  }
}

TEST(WaveDebugTest, SaveWaveRequiresAContextSaveGrantInTheTrapHandler) {
  for (auto arch : {ROCJITSU_CODE_ARCH_RDNA3, ROCJITSU_CODE_ARCH_RDNA3_5, ROCJITSU_CODE_ARCH_RDNA4,
                    ROCJITSU_CODE_ARCH_CDNA5}) {
    SCOPED_TRACE(arch);
    WaveDebugFixture fx(arch);
    ASSERT_TRUE(fx.cu);
    auto *wf = fx.dispatch(kKernelAddr);
    ASSERT_NE(wf, nullptr);
    auto decoder = Decoder::create(arch);
    ASSERT_TRUE(decoder);
    const uint32_t sgpr = wf->sgpr_alloc().base + 4;
    for (const uint32_t op : {0xbe844c00u, 0xbe844d00u}) { // s_sendmsg_rtn_b{32,64} s4/ s[4:5]
      for (const bool handler : {false, true}) {
        wf->set_in_trap_handler(handler);
        for (const uint32_t message : {0xffu, 0x84u}) {
          const uint32_t word = op | message;
          auto inst = decoder->decode_window(std::span(&word, 1), 0);
          ASSERT_TRUE(inst.succeeded());
          fx.cu->write_sgpr(sgpr, 0x12345678);
          fx.cu->write_sgpr(sgpr + 1, 0x9abcdef0);
          std::string detail;
          const bool failed = fx.cu->execute_instruction(inst.value().get(), *wf, &detail).failed();
          const bool needs_record = handler && message == 0x84;
          EXPECT_EQ(failed, needs_record);
          EXPECT_EQ(fx.cu->read_sgpr(sgpr), needs_record ? 0x12345678u : 0u);
          EXPECT_EQ(fx.cu->read_sgpr(sgpr + 1),
                    needs_record || op == 0xbe844c00u ? 0x9abcdef0u : 0u);
          if (needs_record) {
            EXPECT_EQ(wf->instruction_execution_error(),
                      amdgpu::InstructionExecutionError::MissingExecutionBackend);
            EXPECT_NE(detail.find("CWSR record"), std::string::npos);
          }
        }
      }
    }
  }
}

TEST(WaveDebugTest, Gfx1151PrivilegedDestinations) {
  WaveDebugFixture fx(ROCJITSU_CODE_ARCH_RDNA3_5, 1, 1);
  ASSERT_TRUE(fx.cu);
  const uint32_t words[] = {
      0xbeec0081, // s_mov_b32 ttmp0, 1
      0xbe82006c, // s_mov_b32 s2, ttmp0
      0x806c0100, // s_add_u32 ttmp0, s0, s1
  };
  for (size_t i = 0; i < std::size(words); ++i)
    fx.gpu_mem.write32(kKernelAddr + 4 * i, words[i]);
  auto *wf = fx.dispatch(kKernelAddr);
  ASSERT_NE(wf, nullptr);
  wf->set_ttmp(0, 0x12345678);
  fx.cu->write_sgpr(wf->sgpr_alloc().base, 0xffffffff);
  fx.cu->write_sgpr(wf->sgpr_alloc().base + 1, 1);
  for (unsigned i = 0; i < 3; ++i)
    fx.cu->step();
  EXPECT_EQ(fx.cu->read_sgpr(wf->sgpr_alloc().base + 2), 0u);
  EXPECT_EQ(wf->ttmp(0), 0x12345678u);
  EXPECT_FALSE(wf->read_scc());
}

TEST(WaveDebugTest, Gfx1151IndependentWaits) {
  WaveDebugFixture fx(ROCJITSU_CODE_ARCH_RDNA3_5, 1, 1);
  ASSERT_TRUE(fx.cu);
  fx.gpu_mem.write32(kKernelAddr, 0xbcfc0040);      // s_waitcnt_vmcnt null, 0x40
  fx.gpu_mem.write32(kKernelAddr + 4, 0xbf800000);  // s_nop 0
  fx.gpu_mem.write32(kKernelAddr + 8, 0xbd7c0000);  // s_waitcnt_expcnt null, 0
  fx.gpu_mem.write32(kKernelAddr + 12, 0xbf800000); // s_nop 0
  auto *wf = fx.dispatch(kKernelAddr);
  ASSERT_NE(wf, nullptr);
  wf->wait_counters().vmcnt = 1;
  fx.cu->step();
  EXPECT_EQ(wf->state(), amdgpu::WfState::WAITCNT);
  EXPECT_EQ(wf->wait_target().vmcnt, 0u);
  wf->wait_counters().vmcnt = 0;
  fx.cu->step();
  wf->wait_counters().vmcnt = 1;
  fx.cu->step();
  ASSERT_EQ(wf->state(), amdgpu::WfState::RUNNING);
  fx.cu->step();
  EXPECT_EQ(wf->pc, kKernelAddr + 16);
}

TEST(WaveDebugTest, Gfx1151IllegalPaddingEntersPrivilegedHandler) {
  WaveDebugFixture fx(ROCJITSU_CODE_ARCH_RDNA3_5, 1, 1);
  ASSERT_TRUE(fx.cu);
  fx.gpu_mem.write32(kKernelAddr, 0xbf9f0000);          // s_code_end
  fx.gpu_mem.write32(kTrapHandlerAddr, 0xb8820142);     // s_getreg_b32 s2, STATUS.PRIV
  fx.gpu_mem.write32(kTrapHandlerAddr + 4, 0xbe844d82); // s_sendmsg_rtn_b64 s[4:5], GET_TMA
  fx.cu->set_trap_handler_resolver([](const amdgpu::Wavefront &) {
    return amdgpu::ComputeUnitCore::TrapHandlerConfig{kTrapHandlerAddr, 0x123456789abc, false};
  });
  auto *wf = fx.dispatch(kKernelAddr);
  ASSERT_NE(wf, nullptr);
  fx.cu->step();
  ASSERT_TRUE(wf->in_trap_handler());
  EXPECT_EQ(wf->pc, kTrapHandlerAddr);
  EXPECT_EQ(wf->ttmp(0), kKernelAddr);
  EXPECT_NE(wf->trapsts() & (1u << 11), 0u);
  fx.cu->step();
  fx.cu->step();
  EXPECT_EQ(fx.cu->read_sgpr(wf->sgpr_alloc().base + 2), 1u);
  EXPECT_EQ(fx.cu->read_sgpr(wf->sgpr_alloc().base + 4), 0x3456789au);
  EXPECT_EQ(fx.cu->read_sgpr(wf->sgpr_alloc().base + 5), 0x12u);
}

TEST(WaveDebugTest, Rdna4ProtectsTrapWrites) {
  WaveDebugFixture fx(ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_NE(fx.cu, nullptr);
  constexpr uint32_t program[] = {
      0x806c81c1u, // s_add_co_u32 ttmp0, -1, 1
      0xbe84006cu, // s_mov_b32 s4, ttmp0
  };
  for (size_t i = 0; i < std::size(program); ++i)
    fx.gpu_mem.write32(kKernelAddr + 4 * i, program[i]);
  auto *wf = fx.dispatch(kKernelAddr);
  ASSERT_NE(wf, nullptr);
  wf->set_ttmp(0, 0x1234u);
  wf->write_scc(false);
  for (size_t i = 0; i < std::size(program); ++i)
    fx.cu->step();
  EXPECT_FALSE(wf->read_scc());
  EXPECT_EQ(wf->ttmp(0), 0x1234u);
  EXPECT_EQ(fx.cu->read_sgpr(wf->sgpr_alloc().base + 4), 0x1234u);
  EXPECT_EQ(wf->pc, kKernelAddr + sizeof(program));
}

TEST(WaveDebugTest, Gfx12SleepingWaveRestoresDelayWhenDebugStopIsRolledBack) {
  for (const auto arch : {ROCJITSU_CODE_ARCH_RDNA4, ROCJITSU_CODE_ARCH_CDNA5}) {
    SCOPED_TRACE(arch);
    WaveDebugFixture fx(arch);
    ASSERT_NE(fx.cu, nullptr);
    auto *wf = fx.dispatch(kKernelAddr);
    ASSERT_NE(wf, nullptr);
    for (const uint32_t delay : {17u, amdgpu::Wavefront::kSleepForever}) {
      wf->set_sleep_cycles(delay);
      const auto previous = wf->debug_stop_state();
      wf->debug_trap(0);
      EXPECT_EQ(wf->sleep_cycles(), 0u);
      EXPECT_TRUE(wf->debug_halted());
      wf->restore_debug_stop_state(previous);
      EXPECT_EQ(wf->sleep_cycles(), delay);
      EXPECT_FALSE(wf->debug_halted());
    }
  }
}

TEST(WaveDebugTest, Rdna4TrapDrainDependsOnAnInstalledHandler) {
  WaveDebugFixture fx(ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_NE(fx.cu, nullptr);
  fx.gpu_mem.write32(kKernelAddr - 4, 0xb8820182u); // s_getreg_b32 s2,hwreg(STATUS,6,1)
  fx.gpu_mem.write32(kTrapHandlerAddr, 0xb8820182u);
  fx.gpu_mem.write32(kKernelAddr, 0xbf900021u); // Trap ID 1; upper immediate bits are ignored.
  fx.gpu_mem.write32(kKernelAddr + 4, 0xbf900021u);
  auto *wf = fx.dispatch(kKernelAddr - 4);
  ASSERT_NE(wf, nullptr);
  fx.cu->step();
  EXPECT_EQ(fx.cu->read_sgpr(wf->sgpr_alloc().base + 2), 0u);
  wf->wait_counters().increment(amdgpu::WaitCounterType::LOADCNT);
  fx.cu->step();
  EXPECT_EQ(wf->pc, kKernelAddr + 4);
  EXPECT_EQ(wf->state(), amdgpu::WfState::RUNNING);
  fx.cu->set_trap_handler_resolver([](const amdgpu::Wavefront &) {
    return amdgpu::ComputeUnitCore::TrapHandlerConfig{kTrapHandlerAddr, 0, true};
  });
  fx.cu->step();
  EXPECT_EQ(wf->pc, kKernelAddr + 4);
  EXPECT_EQ(wf->state(), amdgpu::WfState::WAITCNT);
  EXPECT_FALSE(wf->in_trap_handler());
  wf->wait_counters().decrement(amdgpu::WaitCounterType::LOADCNT);
  fx.cu->step();
  EXPECT_EQ(wf->pc, kTrapHandlerAddr);
  EXPECT_TRUE(wf->in_trap_handler());
  EXPECT_EQ(wf->trap_id(), 1u);
  EXPECT_EQ(wf->ttmp(0), kKernelAddr + 4);
  EXPECT_EQ(wf->ttmp(1) >> 28, 1u);
  fx.cu->step();
  EXPECT_EQ(fx.cu->read_sgpr(wf->sgpr_alloc().base + 2), 1u);
}

TEST(WaveDebugTest, Gfx10OrderedPsDoneIssuesOnceAfterOlderMemory) {
  class Observer : public ExecutionPlugin {
  public:
    Observer() : ExecutionPlugin("ordered_ps_done") {}
    using ExecutionPlugin::onAmdgpuBeforeExecuteInstruction;
    void onAmdgpuBeforeExecuteInstruction(uint64_t, const Instruction &, amdgpu::Wavefront &,
                                          std::span<const uint32_t>) override {
      ++before;
    }
    void onAmdgpuAfterExecuteInstruction(uint64_t, const Instruction &,
                                         amdgpu::Wavefront &) override {
      ++after;
    }
    void onAmdgpuWavefrontHalted(amdgpu::Wavefront &) override { ++halted; }
    unsigned before = 0, after = 0, halted = 0;
  };
  for (auto arch : {ROCJITSU_CODE_ARCH_RDNA1, ROCJITSU_CODE_ARCH_RDNA2}) {
    SCOPED_TRACE(arch);
    WaveDebugFixture fx(arch);
    ASSERT_NE(fx.cu, nullptr);
    auto group = std::make_shared<ExecutionPluginGroup>(PluginSinkConfig{});
    auto observer = std::make_unique<Observer>();
    auto *counts = observer.get();
    ASSERT_TRUE(group->add(std::move(observer)));
    fx.cu->set_plugin_group(group);
    fx.gpu_mem.write32(kKernelAddr, 0xbf9e0000u);
    unsigned messages = 0;
    fx.cu->set_sendmsg_handler([&](amdgpu::Wavefront &, uint32_t message) {
      EXPECT_EQ(message, 7u);
      ++messages;
      return true;
    });
    auto *wave = fx.dispatch(kKernelAddr);
    ASSERT_NE(wave, nullptr);
    wave->wait_counters().increment(amdgpu::WaitCounterType::VMCNT);
    fx.cu->step();
    EXPECT_EQ(wave->state(), amdgpu::WfState::WAITCNT);
    EXPECT_EQ(wave->pc, kKernelAddr);
    EXPECT_EQ(messages, 0u);
    EXPECT_EQ(counts->before, 0u);
    EXPECT_EQ(counts->after, 0u);
    wave->release_wait_counter(amdgpu::WaitCounterType::VMCNT);
    fx.cu->step();
    EXPECT_EQ(messages, 1u);
    EXPECT_EQ(counts->before, 1u);
    EXPECT_EQ(counts->after, 0u);
    EXPECT_EQ(counts->halted, 1u);
  }
}

TEST(WaveDebugTest, RdnaCodeEndRaisesIllegalInstructionAtThePaddingWord) {
  for (auto arch : {ROCJITSU_CODE_ARCH_RDNA1, ROCJITSU_CODE_ARCH_RDNA2, ROCJITSU_CODE_ARCH_RDNA4}) {
    SCOPED_TRACE(arch);
    WaveDebugFixture fx(arch);
    ASSERT_NE(fx.cu, nullptr);
    fx.gpu_mem.write32(kKernelAddr, 0xbf9f0000u); // s_code_end
    fx.cu->set_trap_handler_resolver([](const amdgpu::Wavefront &) {
      return amdgpu::ComputeUnitCore::TrapHandlerConfig{kTrapHandlerAddr, 0, false};
    });
    auto *wf = fx.dispatch(kKernelAddr);
    ASSERT_NE(wf, nullptr);
    fx.cu->step();
    ASSERT_TRUE(wf->in_trap_handler());
    EXPECT_EQ(wf->ttmp(0), kKernelAddr);
    EXPECT_EQ(wf->ttmp(1) >> 28, 0u);
    EXPECT_EQ(wf->pc, kTrapHandlerAddr);
    EXPECT_NE(wf->trapsts() & (1u << 11), 0u);
  }
}

TEST(WaveDebugTest, ShaderRealtimeUsesTheDeviceTickPeriodInTheModelEpoch) {
  for (auto arch :
       {ROCJITSU_CODE_ARCH_RDNA1, ROCJITSU_CODE_ARCH_RDNA2, ROCJITSU_CODE_ARCH_RDNA3,
        ROCJITSU_CODE_ARCH_RDNA3_5, ROCJITSU_CODE_ARCH_RDNA4, ROCJITSU_CODE_ARCH_CDNA5}) {
    SCOPED_TRACE(arch);
    WaveDebugFixture f(arch);
    ASSERT_NE(f.cu, nullptr);
    auto *wave = f.dispatch(kKernelAddr);
    ASSERT_NE(wave, nullptr);
    auto decoder = Decoder::create(arch);
    const bool gfx10 = arch == ROCJITSU_CODE_ARCH_RDNA1 || arch == ROCJITSU_CODE_ARCH_RDNA2;
    // s_memrealtime s[0:1] or s_sendmsg_rtn_b64 s[0:1], GET_REALTIME.
    auto inst =
        decoder->decode_window(gfx10 ? std::array{0xf4940000u, 0u} : std::array{0xbe804d83u, 0u});
    ASSERT_TRUE(inst.succeeded());
    for (const uint64_t ticks : {999ull, 1000ull, 48776ull, 0x123456789ull * 1000 + 999}) {
      f.cu->restore_model_realtime(ticks);
      ASSERT_TRUE(f.cu->execute_instruction(inst.value().get(), *wave).succeeded());
      const uint64_t value = wave->debug_read_sgpr(0) | (uint64_t{wave->debug_read_sgpr(1)} << 32);
      EXPECT_EQ(value, ticks / 1000);
    }
    wave->halt();
  }
}

TEST(WaveDebugTest, Gfx1250TrapMessagesAgreeWithTrapEnable) {
  constexpr uint64_t tma = 0x123456789abcu;
  for (const uint64_t address : {uint64_t{0}, uint64_t{kTrapHandlerAddr}}) {
    WaveDebugFixture fixture(ROCJITSU_CODE_ARCH_CDNA5, 1, 1);
    fixture.cu->set_trap_handler_resolver([&](const amdgpu::Wavefront &) {
      return amdgpu::ComputeUnitCore::TrapHandlerConfig{address, tma, false};
    });
    fixture.gpu_mem.write32(kKernelAddr, 0xb8820182u);     // s_getreg_b32 s2, STATUS.TRAP_EN
    fixture.gpu_mem.write32(kKernelAddr + 4, 0xbe844d85u); // s_sendmsg_rtn_b64 s[4:5], GET_TBA
    fixture.gpu_mem.write32(kKernelAddr + 8, 0xbe864d82u); // s_sendmsg_rtn_b64 s[6:7], GET_TMA
    auto *wave = fixture.dispatch(kKernelAddr);
    ASSERT_NE(wave, nullptr);
    for (unsigned i = 0; i < 3; ++i)
      fixture.cu->step();
    EXPECT_EQ(wave->debug_read_sgpr(2), address != 0);
    EXPECT_EQ(wave->debug_read_sgpr(4), address);
    EXPECT_EQ(wave->debug_read_sgpr(5), uint32_t(address != 0) << 31);
    EXPECT_EQ(wave->debug_read_sgpr(6), uint32_t(tma >> 8));
    EXPECT_EQ(wave->debug_read_sgpr(7), tma >> 40);
  }
}

TEST(WaveDebugTest, TrapReturnClearsPrivilegeInsideAndOutsideTheHandler) {
  for (auto arch : {ROCJITSU_CODE_ARCH_CDNA1, ROCJITSU_CODE_ARCH_CDNA2, ROCJITSU_CODE_ARCH_CDNA3,
                    ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_RDNA1,
                    ROCJITSU_CODE_ARCH_RDNA2, ROCJITSU_CODE_ARCH_RDNA3, ROCJITSU_CODE_ARCH_RDNA3_5,
                    ROCJITSU_CODE_ARCH_RDNA4}) {
    SCOPED_TRACE(arch);
    for (bool handler : {false, true}) {
      WaveDebugFixture fixture(arch, 1, 1);
      auto *wave = fixture.dispatch(kKernelAddr);
      ASSERT_NE(wave, nullptr);
      wave->set_status_raw(wave->status_raw() | amdgpu::Wavefront::kStatusPrivMask);
      wave->set_trap_saved_status(wave->status_raw());
      wave->set_in_trap_handler(handler);
      const bool gfx10 = arch == ROCJITSU_CODE_ARCH_RDNA1 || arch == ROCJITSU_CODE_ARCH_RDNA2;
      const uint32_t word = arch_is_cdna_4_or_lower(arch) ? 0xbe801f00u
                            : gfx10                       ? 0xbe802200u
                                                          : 0xbe804a00u;
      auto decoder = Decoder::create(arch);
      auto inst = decoder->decode_window(std::span(&word, 1));
      ASSERT_TRUE(inst.succeeded());
      ASSERT_EQ(inst.value()->mnemonic(),
                arch == ROCJITSU_CODE_ARCH_CDNA5 ? "s_rfe_i64" : "s_rfe_b64");
      ASSERT_TRUE(fixture.cu->execute_instruction(inst.value().get(), *wave).succeeded());
      EXPECT_FALSE(wave->in_trap_handler());
      EXPECT_EQ(wave->status_raw() & amdgpu::Wavefront::kStatusPrivMask, 0u);
    }
  }
}

TEST(WaveDebugTest, ReturnMessagesUseTheIssuingQueueAndDispatch) {
  for (auto arch : {ROCJITSU_CODE_ARCH_RDNA3, ROCJITSU_CODE_ARCH_RDNA3_5, ROCJITSU_CODE_ARCH_RDNA4,
                    ROCJITSU_CODE_ARCH_CDNA5}) {
    SCOPED_TRACE(arch);
    WaveDebugFixture fixture(arch, 1, 1);
    auto *wave = fixture.dispatch(kKernelAddr);
    ASSERT_NE(wave, nullptr);
    wave->set_dispatch_id(0x12345678);
    fixture.cu->set_doorbell_resolver([&](const amdgpu::Wavefront &issuing) {
      EXPECT_EQ(&issuing, wave);
      return 0x345u;
    });
    constexpr uint64_t tma = 0x123456789a00ull;
    fixture.cu->set_trap_handler_resolver([](const amdgpu::Wavefront &) {
      return amdgpu::ComputeUnitCore::TrapHandlerConfig{kTrapHandlerAddr, tma, false};
    });
    fixture.gpu_mem.write32(kKernelAddr, 0xbe844d80u);     // s_sendmsg_rtn_b64 s[4:5], GET_DOORBELL
    fixture.gpu_mem.write32(kKernelAddr + 4, 0xbe864d81u); // s_sendmsg_rtn_b64 s[6:7], GET_DDID
    fixture.gpu_mem.write32(kKernelAddr + 8, 0xbe884d82u); // s_sendmsg_rtn_b64 s[8:9], GET_TMA
    fixture.gpu_mem.write32(kKernelAddr + 12, 0xbe8a4d85u); // s_sendmsg_rtn_b64 s[10:11], GET_TBA
    fixture.gpu_mem.write32(kKernelAddr + 16, 0xbe8c4c80u); // s_sendmsg_rtn_b32 s12, GET_DOORBELL
    for (unsigned i = 0; i < 4; ++i)
      fixture.cu->step();
    EXPECT_EQ(wave->debug_read_sgpr(4), 0x345u);
    EXPECT_EQ(wave->debug_read_sgpr(5), 0u);
    EXPECT_EQ(wave->debug_read_sgpr(6), 0x12345678u);
    EXPECT_EQ(wave->debug_read_sgpr(7), 0u);
    EXPECT_EQ(wave->debug_read_sgpr(8), uint32_t(tma >> 8));
    EXPECT_EQ(wave->debug_read_sgpr(9), tma >> 40);
    EXPECT_EQ(wave->debug_read_sgpr(10), kTrapHandlerAddr);
    EXPECT_EQ(wave->debug_read_sgpr(11), 1u << 31);
    fixture.cu->set_doorbell_resolver({});
    fixture.cu->step();
    EXPECT_EQ(wave->debug_read_sgpr(12), 0xbadu);
  }
}

TEST(WaveDebugTest, TrapReturnPreservesTheHandlersExec) {
  for (auto arch : {ROCJITSU_CODE_ARCH_CDNA1, ROCJITSU_CODE_ARCH_CDNA2, ROCJITSU_CODE_ARCH_CDNA3,
                    ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_RDNA1,
                    ROCJITSU_CODE_ARCH_RDNA2, ROCJITSU_CODE_ARCH_RDNA3, ROCJITSU_CODE_ARCH_RDNA3_5,
                    ROCJITSU_CODE_ARCH_RDNA4}) {
    const bool gfx10 = arch == ROCJITSU_CODE_ARCH_RDNA1 || arch == ROCJITSU_CODE_ARCH_RDNA2;
    const bool legacy = arch_is_cdna_4_or_lower(arch);
    for (bool change : {false, true}) {
      SCOPED_TRACE(testing::Message() << arch << " change=" << change);
      WaveDebugFixture fixture(arch, 1, 1);
      fixture.cu->set_trap_handler_resolver([](const amdgpu::Wavefront &) {
        return amdgpu::ComputeUnitCore::TrapHandlerConfig{kTrapHandlerAddr, 0, false};
      });
      fixture.gpu_mem.write32(kKernelAddr, legacy || gfx10 ? 0xbf920007u : 0xbf900007u); // s_trap 7
      fixture.gpu_mem.write32(kKernelAddr + 4,
                              gfx10 ? 0xbe84047eu : 0xbe84017eu); // s_mov_b64 s[4:5], exec
      fixture.gpu_mem.write32(kTrapHandlerAddr, 0x806c846cu);     // s_add_u32 ttmp0, ttmp0, 4
      fixture.gpu_mem.write32(kTrapHandlerAddr + 4,
                              change ? (gfx10 ? 0xbefe0481u : 0xbefe0181u) : 0xbf800000u);
      fixture.gpu_mem.write32(kTrapHandlerAddr + 8,
                              gfx10    ? 0xbe80226cu
                              : legacy ? 0xbe801f6cu
                                       : 0xbe804a6cu); // s_rfe ttmp[0:1]
      auto *wave = fixture.dispatch(kKernelAddr);
      ASSERT_NE(wave, nullptr);
      wave->set_exec(0x15);
      fixture.cu->step();
      ASSERT_TRUE(wave->in_trap_handler());
      for (unsigned i = 0; i < 3; ++i)
        fixture.cu->step();
      EXPECT_FALSE(wave->in_trap_handler());
      ASSERT_EQ(wave->pc, kKernelAddr + 4);
      fixture.cu->step();
      EXPECT_EQ(wave->debug_read_sgpr(4), change ? 1u : 0x15u);
    }
  }
}

TEST(WaveDebugTest, Gfx11StatusReflectsMultiwaveMembershipAndTrapBasePresence) {
  for (auto arch : {ROCJITSU_CODE_ARCH_RDNA3, ROCJITSU_CODE_ARCH_RDNA3_5}) {
    for (unsigned members : {1u, 2u}) {
      SCOPED_TRACE(arch);
      SCOPED_TRACE(members);
      WaveDebugFixture fx(arch, 2, 1);
      ASSERT_NE(fx.cu, nullptr);
      fx.gpu_mem.write32(kKernelAddr, 0xb88202c2u);     // s_getreg_b32 s2, hwreg(STATUS, 11, 1)
      fx.gpu_mem.write32(kKernelAddr + 4, 0xbe844d85u); // s_sendmsg_rtn_b64 s[4:5], GET_TBA
      fx.gpu_mem.write32(kKernelAddr + 8, 0xbfa0ffffu); // s_branch -1
      const uint64_t tba = members == 1 ? 0 : kTrapHandlerAddr;
      fx.cu->set_trap_handler_resolver([&](const amdgpu::Wavefront &) {
        return amdgpu::ComputeUnitCore::TrapHandlerConfig{tba, 0, false};
      });
      fx.cu->begin_workgroup(0, 0, members);
      std::vector<amdgpu::Wavefront *> waves;
      for (unsigned i = 0; i < members; ++i) {
        auto *wave = fx.cu->dispatch_wf(0, kKernelAddr, SGPRS_PER_WF, VGPRS_PER_WF);
        ASSERT_NE(wave, nullptr);
        waves.push_back(wave);
      }
      for (unsigned step = 0; step < 12; ++step)
        fx.cu->step();
      for (const auto *wave : waves) {
        const auto base = wave->sgpr_alloc().base;
        EXPECT_EQ(fx.cu->read_sgpr(base + 2), members == 2);
        EXPECT_EQ(fx.cu->read_sgpr(base + 4), tba);
        EXPECT_EQ(fx.cu->read_sgpr(base + 5), tba ? 1u << 31 : 0);
      }
    }
  }
}

TEST(WaveDebugTest, Rdna4DrainsPendingMemoryBeforeEnteringTheConfiguredTrapHandler) {
  WaveDebugFixture fx(ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_NE(fx.cu, nullptr);
  constexpr uint64_t tma = 0x123456789abcu;
  fx.gpu_mem.write32(kKernelAddr, kGfx12STrap);
  fx.gpu_mem.write32(kTrapHandlerAddr, 0xbe864d85u);     // s_sendmsg_rtn_b64 s[6:7], GET_TBA
  fx.gpu_mem.write32(kTrapHandlerAddr + 4, 0xbe884d82u); // s_sendmsg_rtn_b64 s[8:9], GET_TMA
  fx.cu->set_trap_handler_resolver([](const amdgpu::Wavefront &) {
    return amdgpu::ComputeUnitCore::TrapHandlerConfig{kTrapHandlerAddr, tma, false};
  });
  auto *wf = fx.dispatch(kKernelAddr);
  ASSERT_NE(wf, nullptr);
  wf->wait_counters().vmcnt = 1;
  fx.cu->step();
  EXPECT_EQ(wf->pc, kKernelAddr);
  EXPECT_FALSE(wf->in_trap_handler());
  wf->wait_counters().vmcnt = 0;
  fx.cu->step();
  ASSERT_TRUE(wf->in_trap_handler());
  EXPECT_EQ(wf->pc, kTrapHandlerAddr);
  fx.cu->step();
  fx.cu->step();
  const uint32_t base = wf->sgpr_alloc().base;
  EXPECT_EQ(fx.cu->read_sgpr(base + 6), kTrapHandlerAddr);
  EXPECT_EQ(fx.cu->read_sgpr(base + 7), 1u << 31);
  EXPECT_EQ(fx.cu->read_sgpr(base + 8), static_cast<uint32_t>(tma >> 8));
  EXPECT_EQ(fx.cu->read_sgpr(base + 9), tma >> 40);
}

TEST(WaveDebugTest, FlatB64AtomicPreservesPrivateDwordsAndNeighboringLane) {
  constexpr uint64_t private_base = uint64_t{4} << 32;
  constexpr uint64_t scratch = 0x20000;
  constexpr unsigned pid = 9;
  constexpr unsigned shift = amdgpu::GpuMemory::PAGE_SHIFT;
  constexpr unsigned size = amdgpu::GpuMemory::PAGE_SIZE;
  // flat_atomic_add_x2 (flat_atomic_add_u64 on GFX11/GFX12) returning v[8:9].
  constexpr std::array cdna{0xdd890000u, 0x08000402u, 0u};
  constexpr std::array gfx10{0xdd490000u, 0x087d0402u, 0u};
  constexpr std::array gfx11{0xdd0c4000u, 0x087c0402u, 0u};
  constexpr std::array gfx12{0xec10c07cu, 0x02100000u, 2u};
  for (const auto &[arch, words] :
       {std::pair{ROCJITSU_CODE_ARCH_CDNA2, cdna}, std::pair{ROCJITSU_CODE_ARCH_RDNA2, gfx10},
        std::pair{ROCJITSU_CODE_ARCH_RDNA3, gfx11}, std::pair{ROCJITSU_CODE_ARCH_RDNA3_5, gfx11},
        std::pair{ROCJITSU_CODE_ARCH_RDNA4, gfx12}, std::pair{ROCJITSU_CODE_ARCH_CDNA5, gfx12}}) {
    const bool private_memviol =
        arch == ROCJITSU_CODE_ARCH_RDNA3 || arch == ROCJITSU_CODE_ARCH_RDNA3_5 ||
        arch == ROCJITSU_CODE_ARCH_RDNA4 || arch == ROCJITSU_CODE_ARCH_CDNA5;
    const bool encoded_scratch = arch == ROCJITSU_CODE_ARCH_CDNA5;
    const uint64_t global = encoded_scratch ? 0x200050000ull : 0x50000;
    for (bool private_lane : {false, true}) {
      for (const auto &[global_mapped, shader] : {std::pair{false, false}, std::pair{true, false},
                                                  std::pair{false, true}, std::pair{true, true}}) {
        if (shader && (!private_memviol || !private_lane))
          continue;
        SCOPED_TRACE(testing::Message()
                     << "arch=" << arch << " private_lane=" << private_lane
                     << " global_mapped=" << global_mapped << " shader=" << shader);
        WaveDebugFixture fx(arch);
        ASSERT_NE(fx.cu, nullptr);
        fx.l2.set_backing_memory(&fx.gpu_mem);
        fx.cu->set_apertures(0, 0, private_base, private_base + UINT32_MAX);
        KfdProcess::PageTable pages;
        util::DistributedSharedMutex mutex;
        std::vector<uint8_t> code(size), scratch_page(size), global_page(size);
        pages[kKernelAddr >> shift] = {code.data(), amdgpu::Mtype::RW};
        pages[scratch >> shift] = {scratch_page.data(), amdgpu::Mtype::RW};
        if (global_mapped)
          pages[global >> shift] = {global_page.data(), amdgpu::Mtype::RW};
        fx.gpu_mem.register_process(pid, &pages, &mutex);
        for (unsigned i = 0; i < words.size(); ++i)
          fx.gpu_mem.write32(kKernelAddr + 4 * i, words[i], pid);
        fx.gpu_mem.write32(scratch, 41, pid);
        if (global_mapped)
          fx.gpu_mem.write32(global, 37, pid);
        auto *wf = fx.dispatch(kKernelAddr);
        ASSERT_NE(wf, nullptr);
        const uint64_t second_dword = scratch + wf->wf_size() * sizeof(uint32_t);
        fx.gpu_mem.write32(second_dword, 73, pid);
        fx.gpu_mem.write32(scratch + 4, 53, pid);
        wf->set_process_id(pid);
        wf->set_apertures(0, 0, private_base, private_base + UINT32_MAX,
                          /*override_launch=*/true);
        wf->set_scratch_base(scratch);
        wf->set_scratch_lane_size(256);
        wf->set_exec(private_lane ? 3 : 2);
        for (unsigned lane = 0; lane < 2; ++lane) {
          const uint64_t address = lane ? global : (encoded_scratch ? scratch : private_base);
          const auto base = wf->vgpr_alloc().base;
          fx.cu->write_vgpr(base + 2, lane, uint32_t(address));
          fx.cu->write_vgpr(base + 3, lane, uint32_t(address >> 32));
          fx.cu->write_vgpr(base + 4, lane, 1 + lane);
          fx.cu->write_vgpr(base + 5, lane, lane == 0 ? 1 : 0);
          fx.cu->write_vgpr(base + 8, lane, 0xdeadbeef);
          fx.cu->write_vgpr(base + 9, lane, 0xabcdef01);
        }
        using Fault = std::pair<uint64_t, bool>;
        std::vector<Fault> faults;
        fx.cu->set_debug_active(true);
        std::vector<uint64_t> watched;
        fx.cu->set_watchpoint_handler([&](auto &, uint64_t address, uint32_t, bool, bool) {
          watched.push_back(address);
          return false;
        });
        fx.cu->set_memory_violation_handler(
            [&](auto &wave, uint64_t address, bool write, uint64_t) {
              faults.emplace_back(address, write);
              wave.debug_trap(0);
              return true;
            });
        unsigned step_stops = 0;
        wf->set_debug_single_step(global_mapped && (!private_lane || private_memviol));
        fx.cu->set_single_step_handler([&](auto &wave) {
          ++step_stops;
          wave.debug_trap(0);
          return true;
        });
        unsigned aperture_stops = 0;
        fx.cu->set_aperture_violation_handler([&](auto &wave) {
          ++aperture_stops;
          wave.debug_trap(0);
          return true;
        });
        if (shader)
          fx.cu->set_trap_handler_resolver([](const auto &) {
            return amdgpu::ComputeUnitCore::TrapHandlerConfig{.tba = kTrapHandlerAddr};
          });
        fx.cu->step();
        EXPECT_EQ(step_stops, unsigned(global_mapped && !private_lane));
        if (private_lane && private_memviol) {
          ASSERT_TRUE(wf->memory_trap_pending());
          fx.cu->step();
          EXPECT_FALSE(wf->memory_trap_pending());
          EXPECT_EQ(wf->in_trap_handler(), shader);
          EXPECT_EQ(wf->debug_halted(), !shader);
          EXPECT_EQ(aperture_stops, unsigned(!shader));
          EXPECT_EQ(step_stops, 0u);
          if (shader) {
            EXPECT_EQ(wf->pc, kTrapHandlerAddr);
            EXPECT_EQ(wf->ttmp(0), kKernelAddr + (words == gfx12 ? 12u : 8u));
          }
        }
        EXPECT_EQ(fx.gpu_mem.read32(scratch, pid), 41u);
        EXPECT_EQ(fx.gpu_mem.read32(scratch + 4, pid), 53u);
        EXPECT_EQ(fx.gpu_mem.read32(second_dword, pid), 73u);
        // The older targets have no qualified trap or global-lane result
        // when the same instruction also addresses private storage.
        if (!private_lane || private_memviol) {
          const auto expected = !private_lane && !global_mapped ? std::vector<Fault>{{global, true}}
                                                                : std::vector<Fault>{};
          EXPECT_EQ(faults, expected);
          EXPECT_EQ(bool(wf->trapsts() & (1u << 8)), private_lane);
          EXPECT_EQ(watched, (!private_lane && global_mapped ? std::vector<uint64_t>{global}
                                                             : std::vector<uint64_t>{}));
          EXPECT_EQ(fx.cu->read_vgpr(wf->vgpr_alloc().base + 8, 0), 0xdeadbeefu);
          EXPECT_EQ(fx.cu->read_vgpr(wf->vgpr_alloc().base + 9, 0), 0xabcdef01u);
          if (global_mapped) {
            EXPECT_EQ(fx.gpu_mem.read32(global, pid), private_lane ? 37u : 39u);
          }
        }
        wf->halt();
        fx.gpu_mem.unregister_process(pid);
      }
    }
  }
}

TEST(WaveDebugTest, MisalignedFlatLdsAtomicPrecedesGlobalFaultAndWatchpoint) {
  constexpr uint64_t shared = uint64_t{3} << 32, global = 0x50000;
  constexpr uint32_t pid = 9, shift = amdgpu::GpuMemory::PAGE_SHIFT;
  constexpr uint32_t size = amdgpu::GpuMemory::PAGE_SIZE;
  for (auto arch : {ROCJITSU_CODE_ARCH_RDNA3, ROCJITSU_CODE_ARCH_RDNA3_5, ROCJITSU_CODE_ARCH_RDNA4,
                    ROCJITSU_CODE_ARCH_CDNA5}) {
    const bool gfx11 = arch == ROCJITSU_CODE_ARCH_RDNA3 || arch == ROCJITSU_CODE_ARCH_RDNA3_5;
    const std::array code =
        gfx11 ? std::array{0xdcd44000u, 0x007c0402u, 0u} : std::array{0xec0d407cu, 0x02100000u, 2u};
    for (unsigned offset : {4u, 5u, 0x80000004u}) {
      if (offset >= 0x80000000u && arch != ROCJITSU_CODE_ARCH_CDNA5)
        continue;
      for (bool mapped : {false, true}) {
        SCOPED_TRACE(testing::Message()
                     << arch << " LDS offset=" << offset << " mapped=" << mapped);
        WaveDebugFixture fx(arch);
        ASSERT_NE(fx.cu, nullptr);
        fx.l2.set_backing_memory(&fx.gpu_mem);
        fx.cu->set_apertures(shared, shared + UINT32_MAX, 0, 0);
        std::vector<uint8_t> kernel(size), data(size);
        KfdProcess::PageTable pages;
        util::DistributedSharedMutex mutex;
        pages[kKernelAddr >> shift] = {kernel.data(), amdgpu::Mtype::RW};
        if (mapped)
          pages[global >> shift] = {data.data(), amdgpu::Mtype::RW};
        fx.gpu_mem.register_process(pid, &pages, &mutex);
        for (unsigned i = 0; i < code.size(); ++i)
          fx.gpu_mem.write32(kKernelAddr + 4 * i, code[i], pid);
        if (mapped)
          fx.gpu_mem.write32(global, 37, pid);
        auto *wf = fx.dispatch(kKernelAddr);
        ASSERT_NE(wf, nullptr);
        wf->set_process_id(pid);
        wf->set_exec(3);
        wf->set_lds_size(1024);
        wf->lds().write32(4, 41);
        for (unsigned lane = 0; lane < 2; ++lane) {
          const uint64_t address = lane ? global : shared + offset;
          const auto base = wf->vgpr_alloc().base;
          fx.cu->write_vgpr(base + 2, lane, uint32_t(address));
          fx.cu->write_vgpr(base + 3, lane, uint32_t(address >> 32));
          fx.cu->write_vgpr(base + 4, lane, 1 + lane);
        }
        std::vector<uint64_t> watched;
        std::vector<uint64_t> faults;
        fx.cu->set_debug_active(true);
        fx.cu->set_watchpoint_handler(
            [&](auto &, uint64_t address, uint32_t bytes, bool write, bool atomic) {
              EXPECT_EQ(bytes, 4u);
              EXPECT_TRUE(write && atomic);
              watched.push_back(address);
              return false;
            });
        fx.cu->set_memory_violation_handler(
            [&](auto &wave, uint64_t address, bool write, uint64_t) {
              // The synchronous memory violation has no translated address.
              EXPECT_EQ(write, address == global);
              faults.push_back(address);
              wave.debug_trap(0);
              return true;
            });
        fx.cu->step();
        if (offset != 4) {
          EXPECT_EQ(faults, (std::vector<uint64_t>{0}));
        } else if (!mapped) {
          EXPECT_EQ(faults, (std::vector<uint64_t>{global}));
        } else {
          EXPECT_TRUE(faults.empty());
        }
        if (mapped && offset == 4) {
          EXPECT_EQ(watched, (std::vector<uint64_t>{shared + 4, global}));
          EXPECT_EQ(wf->lds().read32(4), 42u);
        } else {
          EXPECT_TRUE(watched.empty());
          EXPECT_EQ(wf->lds().read32(4), 41u);
        }
        wf->halt();
        fx.gpu_mem.unregister_process(pid);
      }
    }
  }
}

TEST(WaveDebugTest, ScalarBufferProbesOnlyInBoundsDwords) {
  constexpr uint64_t page = amdgpu::GpuMemory::PAGE_SIZE, data = 0x10000, addr = data + page - 4;
  constexpr uint32_t pid = 9, value = 0x12345678u, sentinel = 0xdeadbeefu;
  for (auto arch : {ROCJITSU_CODE_ARCH_RDNA3, ROCJITSU_CODE_ARCH_RDNA4}) {
    SCOPED_TRACE(static_cast<unsigned>(arch));
    for (unsigned bound : {0u, 4u, 8u}) {
      SCOPED_TRACE(bound);
      WaveDebugFixture fx(arch);
      fx.l2.set_backing_memory(&fx.gpu_mem);
      std::vector<uint8_t> code(page), payload(page);
      KfdProcess::PageTable pages;
      util::DistributedSharedMutex mutex;
      pages[kKernelAddr >> amdgpu::GpuMemory::PAGE_SHIFT] = {code.data(), amdgpu::Mtype::RW};
      pages[data >> amdgpu::GpuMemory::PAGE_SHIFT] = {payload.data(), amdgpu::Mtype::RW};
      fx.gpu_mem.register_process(pid, &pages, &mutex);
      // s_buffer_load_b64 s[4:5], s[0:3], null
      fx.gpu_mem.write32(kKernelAddr, arch == ROCJITSU_CODE_ARCH_RDNA4 ? 0xf4022100u : 0xf4240100u,
                         pid);
      fx.gpu_mem.write32(kKernelAddr + 4, 0xf8000000u, pid);
      fx.gpu_mem.write32(addr, value, pid);
      auto *wf = fx.dispatch(kKernelAddr);
      ASSERT_NE(wf, nullptr);
      wf->set_process_id(pid);
      const uint32_t sb = wf->sgpr_alloc().base;
      const std::array<uint32_t, 6> registers{uint32_t{addr}, 0, bound, 0, sentinel, sentinel};
      for (size_t i = 0; i < registers.size(); ++i)
        fx.cu->write_sgpr(sb + i, registers[i]);
      unsigned faults = 0;
      fx.cu->set_debug_active(true);
      fx.cu->set_memory_violation_handler([&](amdgpu::Wavefront &w, uint64_t, bool, uint64_t) {
        ++faults;
        w.debug_trap(0);
        return true;
      });
      fx.cu->step();
      EXPECT_EQ(faults, bound == 8);
      EXPECT_EQ(wf->debug_read_sgpr(4), bound == 8 ? sentinel : bound ? value : 0u);
      EXPECT_EQ(wf->debug_read_sgpr(5), bound == 8 ? sentinel : 0u);
      wf->halt();
      fx.gpu_mem.unregister_process(pid);
    }
  }
}

TEST(WaveDebugTest, DebuggerReadsPreservePhysicalStorageBoundsAtNonzeroOffsets) {
  const auto check = []<GpuIsa Isa>(rj_code_arch_t arch) {
    SCOPED_TRACE(arch);
    WaveDebugFixture fx(arch, 2);
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
    using ComputeUnit = amdgpu::IsaExecComputeUnit<simdojo::ExecMode::FUNCTIONAL, Isa>;
    const auto *cu = dynamic_cast<const ComputeUnit *>(fx.cu.get());
    ASSERT_NE(cu, nullptr);
    auto *first_wave = fx.dispatch(kKernelAddr);
    ASSERT_NE(first_wave, nullptr);
    auto *wave = fx.dispatch(kKernelAddr);
    ASSERT_NE(wave, nullptr);
    ASSERT_GT(wave->sgpr_alloc().base, first_wave->sgpr_alloc().base);
    ASSERT_GT(wave->vgpr_alloc().base, first_wave->vgpr_alloc().base);
    const auto last_sgpr = wave->num_sgprs() - 1;
    const auto last_vgpr = wave->num_vgprs() - 1;
    const auto last_lane = wave->wf_size() - 1;
    fx.cu->write_sgpr(wave->sgpr_alloc().base + last_sgpr, 42);
    fx.cu->write_vgpr(wave->vgpr_alloc().base + last_vgpr, last_lane, 43);
    fx.cu->write_vgpr(wave->vgpr_alloc().base + 1, 0, 44);

    EXPECT_EQ(wave->debug_read_sgpr(last_sgpr), 42u);
    EXPECT_EQ(wave->debug_read_vgpr(last_vgpr, last_lane), 43u);

    const auto second_wave_sgpr =
        wave->sgpr_alloc().base - first_wave->sgpr_alloc().base + last_sgpr;
    const auto second_wave_vgpr =
        wave->vgpr_alloc().base - first_wave->vgpr_alloc().base + last_vgpr;
    EXPECT_EQ(first_wave->debug_read_sgpr(second_wave_sgpr), 42u);
    EXPECT_EQ(first_wave->debug_sgpr_capacity(), SGPRS_PER_WF);
    ASSERT_GE(second_wave_sgpr, first_wave->debug_sgpr_capacity());
    first_wave->debug_write_sgpr(second_wave_sgpr, 17);
    EXPECT_EQ(wave->debug_read_sgpr(last_sgpr), 17u);
    // An out-of-file offset must not wrap around into the first wave.
    fx.cu->write_sgpr(0, 55);
    const uint32_t wrapping_offset = UINT32_MAX - wave->sgpr_alloc().base + 1u;
    EXPECT_EQ(wave->debug_read_sgpr(wrapping_offset), 0u);
    wave->debug_write_sgpr(wrapping_offset, 56);
    EXPECT_EQ(fx.cu->read_sgpr(0), 55u);
    EXPECT_EQ(first_wave->debug_read_vgpr(second_wave_vgpr, last_lane), 43u);

    const auto sgpr_end = fx.cu->sgpr_file().total_regs() - wave->sgpr_alloc().base;
    const auto vgpr_end = cu->vgpr_file().total_regs() - wave->vgpr_alloc().base;
    EXPECT_EQ(wave->debug_read_sgpr(sgpr_end), 0u);
    EXPECT_EQ(wave->debug_read_sgpr(sgpr_end + 1), 0u);
    EXPECT_EQ(wave->debug_read_vgpr(vgpr_end, 0), 0u);
    EXPECT_EQ(wave->debug_read_vgpr(vgpr_end + 1, 0), 0u);
    EXPECT_EQ(wave->debug_read_vgpr(1, 0), 44u);
    EXPECT_EQ(wave->debug_read_vgpr(0, fx.cu->vgpr_storage_lane_count()), 0u);
  };
  check.operator()<cdna3::Isa>(ROCJITSU_CODE_ARCH_CDNA3);
  check.operator()<cdna4::Isa>(ROCJITSU_CODE_ARCH_CDNA4);
  check.operator()<rdna4::Isa>(ROCJITSU_CODE_ARCH_RDNA4);
}

TEST(WaveDebugTest, DebuggerReadsPreserveSmallAllocationStorage) {
  for (auto arch : {ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_RDNA4}) {
    SCOPED_TRACE(arch);
    WaveDebugFixture fx(arch, 2);
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
    auto *first_wave = fx.dispatch(kKernelAddr);
    ASSERT_NE(first_wave, nullptr);
    auto *wave = fx.cu->dispatch_wf(0, kKernelAddr, 16, 4);
    ASSERT_NE(wave, nullptr);
    ASSERT_EQ(wave->num_sgprs(), 16u);
    ASSERT_EQ(wave->num_vgprs(), 4u);
    ASSERT_GT(wave->sgpr_alloc().base, first_wave->sgpr_alloc().base);
    ASSERT_GT(wave->vgpr_alloc().base, first_wave->vgpr_alloc().base);
    const auto sbase = wave->sgpr_alloc().base;
    const auto vbase = wave->vgpr_alloc().base;
    const auto last_lane = fx.cu->vgpr_storage_lane_count() - 1;
    fx.cu->write_sgpr(sbase + 32, 42);
    fx.cu->write_sgpr(sbase + 33, 43);
    fx.cu->write_vgpr(vbase + 8, last_lane, 44);

    EXPECT_EQ(fx.cu->read_sgpr(sbase + 32), 42u);
    EXPECT_EQ(fx.cu->read_sgpr(sbase + 33), 43u);
    EXPECT_EQ(fx.cu->read_vgpr(vbase + 8, last_lane), 44u);
    EXPECT_EQ(wave->debug_read_sgpr(32), 42u);
    EXPECT_EQ(wave->debug_read_sgpr(33), 43u);
    EXPECT_EQ(wave->debug_read_vgpr(8, last_lane), 44u);
  }
}

TEST(WaveDebugTest, DebuggerReadsPreserveStoredLanesForWave32) {
  WaveDebugFixture fx(ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_FALSE(::testing::Test::HasFatalFailure());
  auto *wave = fx.cu->dispatch_wf(0, kKernelAddr, 16, 4, 32);
  ASSERT_NE(wave, nullptr);
  ASSERT_EQ(wave->wf_size(), 32u);
  ASSERT_EQ(fx.cu->vgpr_storage_lane_count(), 64u);
  const auto vbase = wave->vgpr_alloc().base;
  fx.cu->write_vgpr(vbase, 32, 42);
  fx.cu->write_vgpr(vbase, 63, 43);
  fx.cu->write_vgpr(vbase + 1, 0, 44);

  EXPECT_EQ(fx.cu->read_vgpr(vbase, 32), 42u);
  EXPECT_EQ(fx.cu->read_vgpr(vbase, 63), 43u);
  EXPECT_EQ(wave->debug_read_vgpr(0, 32), 42u);
  EXPECT_EQ(wave->debug_read_vgpr(0, 63), 43u);
  EXPECT_EQ(wave->debug_read_vgpr(1, 0), 44u);
  EXPECT_EQ(wave->debug_read_vgpr(0, fx.cu->vgpr_storage_lane_count()), 0u);
}

TEST(WaveDebugTest, DebuggerReadsPreservePhysicalLaneBoundsForGfx1250) {
  WaveDebugFixture fx(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_FALSE(::testing::Test::HasFatalFailure());
  auto *wave = fx.cu->dispatch_wf(0, kKernelAddr, 16, 4, 32);
  ASSERT_NE(wave, nullptr);
  ASSERT_EQ(wave->wf_size(), 32u);
  ASSERT_EQ(fx.cu->vgpr_storage_lane_count(), 32u);
  const auto vbase = wave->vgpr_alloc().base;
  fx.cu->write_vgpr(vbase, 31, 43);
  fx.cu->write_vgpr(vbase + 1, 0, 44);

  EXPECT_EQ(wave->debug_read_vgpr(0, 31), 43u);
  EXPECT_EQ(wave->debug_read_vgpr(1, 0), 44u);
  EXPECT_EQ(wave->debug_read_vgpr(0, fx.cu->vgpr_storage_lane_count()), 0u);
}

TEST(WaveDebugTest, Cdna2TrapDrainsMemoryAndExposesPrivilegeAndHandlerAddresses) {
  WaveDebugFixture fx(ROCJITSU_CODE_ARCH_CDNA2);
  fx.gpu_mem.write32(kKernelAddr, kSTrapSeven);
  fx.gpu_mem.write32(kKernelAddr + 4, kSEndpgm);
  const std::array handler{
      0xb8820142u, // s_getreg_b32 s2, hwreg(STATUS, 5, 1)
      0xb883f810u, // s_getreg_b32 s3, hwreg(TBA_LO)
      0xb884f812u, // s_getreg_b32 s4, hwreg(TMA_LO)
      0x806c846cu, // s_add_u32 ttmp0, ttmp0, 4
      0xbe801f6cu, // s_rfe_b64 ttmp[0:1]
  };
  for (uint32_t i = 0; i < handler.size(); ++i)
    fx.gpu_mem.write32(kTrapHandlerAddr + i * 4, handler[i]);
  fx.cu->set_trap_handler_resolver([](const amdgpu::Wavefront &) {
    return amdgpu::ComputeUnitCore::TrapHandlerConfig{kTrapHandlerAddr, 0x3000, false};
  });
  auto *wf = fx.dispatch(kKernelAddr);
  ASSERT_NE(wf, nullptr);
  wf->wait_counters().lgkmcnt = 1;
  fx.cu->step();
  EXPECT_EQ(wf->pc, kKernelAddr);
  EXPECT_FALSE(wf->in_trap_handler());
  wf->wait_counters().lgkmcnt = 0;
  fx.cu->step();
  ASSERT_TRUE(wf->in_trap_handler());
  EXPECT_EQ(wf->pc, kTrapHandlerAddr);
  for (size_t i = 0; i < handler.size(); ++i)
    fx.cu->step();
  EXPECT_FALSE(wf->in_trap_handler());
  EXPECT_EQ(wf->status_raw() & amdgpu::Wavefront::kStatusPrivMask, 0u);
  EXPECT_EQ(wf->pc, kKernelAddr + 4);
  const auto base = wf->sgpr_alloc().base;
  EXPECT_EQ(fx.cu->read_sgpr(base + 2), 1u);
  EXPECT_EQ(fx.cu->read_sgpr(base + 3) << 8, kTrapHandlerAddr);
  EXPECT_EQ(fx.cu->read_sgpr(base + 4) << 8, 0x3000u);
  wf->halt();
}

TEST(WaveDebugTest, Cdna2FlatBoundsReportOnCompletionAndEnterEnabledHandlerAtNextPc) {
  constexpr uint64_t shared = uint64_t{1} << 48;
  constexpr uint64_t priv = uint64_t{2} << 48;
  constexpr uint32_t backing = 0x4000;
  constexpr uint32_t mem_viol = 1u << 8;
  for (bool local : {false, true}) {
    SCOPED_TRACE(local);
    WaveDebugFixture fx(ROCJITSU_CODE_ARCH_CDNA2);
    fx.l2.set_backing_memory(&fx.gpu_mem);
    fx.cu->set_apertures(shared, shared + UINT32_MAX, priv, priv + UINT32_MAX);
    const std::array code = {0xdc500000u, 0x08000002u, // flat_load_dword v8, v[2:3]
                             0xdc700000u, 0x00000402u, // flat_store_dword v[2:3], v4
                             kSEndpgm};
    for (size_t i = 0; i < code.size(); ++i)
      fx.gpu_mem.write32(kKernelAddr + 4 * i, code[i]);
    fx.gpu_mem.write32(kTrapHandlerAddr, 0xbe801f6cu); // s_rfe_b64 ttmp[0:1]
    fx.cu->set_trap_handler_resolver([](const amdgpu::Wavefront &) {
      return amdgpu::ComputeUnitCore::TrapHandlerConfig{kTrapHandlerAddr, 0x3000, false};
    });
    auto *wave = fx.dispatch(kKernelAddr);
    ASSERT_NE(wave, nullptr);
    wave->set_exec(3);
    wave->set_scratch_base(backing);
    wave->set_scratch_lane_size(4);
    wave->set_lds_base(128);
    wave->set_lds_size(4);
    wave->set_trapsts(1u << 3);
    const uint32_t vb = wave->vgpr_alloc().base;
    fx.cu->write_vgpr(vb + 2, 0, 0);
    fx.cu->write_vgpr(vb + 2, 1, 4);
    for (uint32_t lane = 0; lane < 2; ++lane) {
      fx.cu->write_vgpr(vb + 3, lane, (local ? shared : priv) >> 32);
      fx.cu->write_vgpr(vb + 4, lane, 0x11223344);
      fx.cu->write_vgpr(vb + 8, lane, 0xaaaaaaaa);
    }
    const uint32_t guard = backing + wave->wf_size() * 4 + 4;
    fx.gpu_mem.write32(backing, 0x76543210);
    fx.gpu_mem.write32(guard, 0xcccccccc);
    wave->lds().write32(128, 0x76543210);
    wave->lds().write32(132, 0xcccccccc);
    fx.cu->step();
    EXPECT_EQ(fx.cu->read_vgpr(vb + 8, 0), 0x76543210u);
    EXPECT_EQ(fx.cu->read_vgpr(vb + 8, 1), 0u);
    EXPECT_EQ(wave->trapsts(), mem_viol | (1u << 3));
    EXPECT_FALSE(wave->in_trap_handler());
    EXPECT_EQ(wave->pc, kKernelAddr + 8);
    EXPECT_TRUE(wave->wait_counters().empty());

    // Enabling a subsequent violation traps even when its sticky bit was set.
    wave->set_mode_raw(wave->mode_raw() | (1u << 20));
    fx.cu->step();
    fx.cu->step();
    ASSERT_TRUE(wave->in_trap_handler());
    EXPECT_EQ(wave->pc, kTrapHandlerAddr);
    EXPECT_EQ(wave->ttmp(0), kKernelAddr + 16);
    EXPECT_EQ(wave->ttmp(1), 0u);
    EXPECT_EQ(wave->trapsts(), mem_viol | (1u << 3));
    EXPECT_TRUE(wave->wait_counters().empty());
    if (local) {
      EXPECT_EQ(wave->lds().read32(128), 0x11223344u);
      EXPECT_EQ(wave->lds().read32(132), 0xccccccccu);
    } else {
      fx.l2.flush_all(wave->process_id());
      EXPECT_EQ(fx.gpu_mem.read32(backing), 0x11223344u);
      EXPECT_EQ(fx.gpu_mem.read32(guard), 0xccccccccu);
    }
    fx.cu->step();
    EXPECT_FALSE(wave->in_trap_handler());
    EXPECT_EQ(wave->pc, kKernelAddr + 16);
    wave->halt();
  }
}

TEST(WaveDebugTest, Cdna2FlatBoundsReportPartialElementsOnlyForActiveLanes) {
  constexpr uint64_t shared = uint64_t{1} << 48;
  constexpr uint64_t priv = uint64_t{2} << 48;
  for (const uint64_t aperture : {priv, shared}) {
    SCOPED_TRACE(aperture);
    WaveDebugFixture fx(ROCJITSU_CODE_ARCH_CDNA2);
    fx.l2.set_backing_memory(&fx.gpu_mem);
    fx.cu->set_apertures(shared, shared + UINT32_MAX, priv, priv + UINT32_MAX);
    for (uint32_t i = 0; i != 2; ++i) {
      fx.gpu_mem.write32(kKernelAddr + i * 8, 0xdc540000); // flat_load_dwordx2 v[8:9], v[2:3]
      fx.gpu_mem.write32(kKernelAddr + i * 8 + 4, 0x08000002);
    }
    auto *wave = fx.dispatch(kKernelAddr);
    ASSERT_NE(wave, nullptr);
    wave->set_scratch_base(0x4000);
    wave->set_scratch_lane_size(4);
    wave->set_lds_base(128);
    wave->set_lds_size(4);
    wave->lds().write32(128, 7);
    fx.gpu_mem.write32(0x4000, 7);
    const auto vb = wave->vgpr_alloc().base;
    fx.cu->write_vgpr(vb + 2, 0, 0);
    fx.cu->write_vgpr(vb + 3, 0, aperture >> 32);
    fx.cu->write_vgpr(vb + 8, 0, 11);
    fx.cu->write_vgpr(vb + 9, 0, 12);
    wave->set_exec(0);
    fx.cu->step();
    EXPECT_EQ(wave->trapsts() & (1u << 8), 0u);
    EXPECT_EQ(fx.cu->read_vgpr(vb + 8, 0), 11u);
    EXPECT_EQ(fx.cu->read_vgpr(vb + 9, 0), 12u);
    wave->set_exec(1);
    fx.cu->step();
    EXPECT_EQ(wave->trapsts() & (1u << 8), 1u << 8);
    EXPECT_EQ(fx.cu->read_vgpr(vb + 8, 0), 7u);
    EXPECT_EQ(fx.cu->read_vgpr(vb + 9, 0), 0u);
    wave->halt();
  }
}

TEST(WaveDebugTest, Cdna2TakenTrapRetiresDependenciesBeforeHandlerReads) {
  for (const bool handler_installed : {false, true}) {
    SCOPED_TRACE(handler_installed);
    WaveDebugFixture fx(ROCJITSU_CODE_ARCH_CDNA2);
    fx.l2.set_backing_memory(&fx.gpu_mem);
    constexpr uint32_t kData = 0x4000;
    constexpr uint32_t kValue = 0x11223344;
    constexpr uint32_t kReadLoadedRegister = 0xbe830002; // s_mov_b32 s3, s2
    fx.gpu_mem.write32(kData, kValue);
    fx.gpu_mem.write32(kKernelAddr, 0xc0020080); // s_load_dword s2, s[0:1], 0
    fx.gpu_mem.write32(kKernelAddr + 4, 0);
    fx.gpu_mem.write32(kKernelAddr + 8, kSTrapSeven);
    fx.gpu_mem.write32(kKernelAddr + 12, kReadLoadedRegister);
    fx.gpu_mem.write32(kTrapHandlerAddr, kReadLoadedRegister);
    if (handler_installed)
      fx.cu->set_trap_handler_resolver([](const amdgpu::Wavefront &) {
        return amdgpu::ComputeUnitCore::TrapHandlerConfig{kTrapHandlerAddr, 0, false};
      });
    auto *wave = fx.dispatch(kKernelAddr);
    ASSERT_NE(wave, nullptr);
    const uint32_t base = wave->sgpr_alloc().base;
    fx.cu->write_sgpr(base, kData);
    fx.cu->write_sgpr(base + 1, 0);
    fx.cu->step();
    ASSERT_EQ(wave->pc, kKernelAddr + 8);
    ASSERT_TRUE(wave->wait_counters().empty());
    ASSERT_NE(wave->memory_wait_scoreboard(), nullptr);
    ASSERT_FALSE(wave->memory_wait_scoreboard()->empty());
    fx.cu->step();
    ASSERT_EQ(wave->pc, handler_installed ? kTrapHandlerAddr : kKernelAddr + 12);
    fx.cu->step();
    EXPECT_EQ(fx.cu->read_sgpr(base + 3), kValue);
    EXPECT_EQ(fx.cu->memory_wait_diagnostic_count(), handler_installed ? 0u : 1u);
    wave->halt();
  }
}

TEST(WaveDebugTest, Cdna2ScalarSourcesUseLiveWaveState) {
  WaveDebugFixture fx(ROCJITSU_CODE_ARCH_CDNA2);
  const std::array program{
      0xbf820001u, // s_branch +1
      0xbeec0081u, // s_mov_b32 ttmp0, 1 (skipped)
      0xbeed0081u, // s_mov_b32 ttmp1, 1 (unprivileged)
      0xbe82014cu, // s_mov_b64 s[2:3], s[76:77] (out of allocation)
      0xbe8401fbu, // s_mov_b64 s[4:5], src_vccz
      0xbe8601fcu, // s_mov_b64 s[6:7], src_execz
      kSEndpgm,
  };
  for (uint32_t i = 0; i < program.size(); ++i)
    fx.gpu_mem.write32(kKernelAddr + i * 4, program[i]);
  auto *wf = fx.cu->dispatch_wf(0, kKernelAddr, 16, 16);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(0);
  wf->set_vcc(0);
  const auto base = wf->sgpr_alloc().base;
  fx.cu->write_sgpr(base, 0x11223344);
  fx.cu->write_sgpr(base + 1, 0x55667788);
  for (unsigned i = 0; i < 5; ++i)
    fx.cu->step();
  EXPECT_EQ(wf->ttmp(0), 0u);
  EXPECT_EQ(wf->ttmp(1), 0u);
  EXPECT_EQ(fx.cu->read_sgpr(base + 2), 0x11223344u);
  EXPECT_EQ(fx.cu->read_sgpr(base + 3), 0x55667788u);
  EXPECT_EQ(fx.cu->read_sgpr(base + 4), 1u);
  EXPECT_EQ(fx.cu->read_sgpr(base + 6), 1u);
  fx.cu->step();
  EXPECT_TRUE(wf->is_halted());
}

TEST(WaveDebugTest, Cdna2DeferredScalarLoadsHonorTtmpWritePermission) {
  WaveDebugFixture fx(ROCJITSU_CODE_ARCH_CDNA2);
  fx.l2.set_backing_memory(&fx.gpu_mem);
  fx.gpu_mem.write32(kKernelAddr, 0xc0021c00); // s_load_dword ttmp4, s[0:1], 0
  fx.gpu_mem.write32(kKernelAddr + 4, 0);
  fx.gpu_mem.write32(kKernelAddr + 8, 0xbf8c0000);  // s_waitcnt 0
  fx.gpu_mem.write32(kKernelAddr + 12, 0xbf800000); // s_nop 0
  fx.gpu_mem.write32(0x4000, 0x11223344);
  for (const bool privileged : {false, true}) {
    auto *wave = fx.dispatch(kKernelAddr);
    ASSERT_NE(wave, nullptr);
    wave->set_in_trap_handler(privileged);
    wave->set_ttmp(4, 0x55667788);
    fx.cu->write_sgpr(wave->sgpr_alloc().base, 0x4000);
    fx.cu->write_sgpr(wave->sgpr_alloc().base + 1, 0);
    for (unsigned i = 0; i < 32 && wave->pc != kKernelAddr + 16; ++i)
      fx.cu->step();
    EXPECT_EQ(wave->pc, kKernelAddr + 16);
    EXPECT_EQ(wave->ttmp(4), privileged ? 0x11223344u : 0x55667788u);
    wave->halt();
  }
}

TEST(WaveDebugTest, Cdna4GetregReadsConfiguredTrapAddressWords) {
  WaveDebugFixture fx;
  ASSERT_FALSE(::testing::Test::HasFatalFailure());
  fx.cu->set_trap_handler_resolver([](const amdgpu::Wavefront &) {
    return amdgpu::ComputeUnitCore::TrapHandlerConfig{0xabcd12345678ULL, 0x9876fedcba00ULL, true};
  });
  // s_getreg_b32 sN, hwreg(16+N) for the four address words.
  for (unsigned i = 0; i < 4; ++i)
    fx.gpu_mem.write32(kKernelAddr + i * 4, 0xb880f810u + (i << 16) + i);
  auto *wave = fx.dispatch(kKernelAddr);
  ASSERT_NE(wave, nullptr);
  for (unsigned i = 0; i < 4; ++i)
    fx.cu->step();
  const auto byte_address = [&](unsigned lo, unsigned hi) {
    return (uint64_t{wave->debug_read_sgpr(hi)} << 40) | (uint64_t{wave->debug_read_sgpr(lo)} << 8);
  };
  EXPECT_EQ(byte_address(0, 1), 0xabcd12345600ULL);
  EXPECT_EQ(byte_address(2, 3), 0x9876fedcba00ULL);
}

TEST(WaveDebugTest, Cdna4UserShaderReadsLiveState) {
  WaveDebugFixture fx;
  ASSERT_FALSE(::testing::Test::HasFatalFailure());
  const uint32_t program[] = {
      0xb8800a42, // s_getreg_b32 s0, hwreg(STATUS,9,2)
      0xbe8201fc, // s_mov_b64 s[2:3], src_execz
      0xbeee0081, // s_mov_b32 ttmp2, 1
      0xbe84006e, // s_mov_b32 s4, ttmp2
  };
  for (size_t i = 0; i < std::size(program); ++i)
    fx.gpu_mem.write32(kKernelAddr + 4 * i, program[i]);
  auto *wave = fx.dispatch(kKernelAddr);
  ASSERT_NE(wave, nullptr);
  wave->set_exec(0);
  wave->set_vcc(0);
  wave->set_status_raw(3u << 20);
  wave->set_ttmp(2, 0xa5);
  for (size_t i = 0; i < std::size(program); ++i)
    fx.cu->step();
  EXPECT_EQ(wave->debug_read_sgpr(0), 3u);
  EXPECT_EQ(wave->debug_read_sgpr(2), 1u);
  EXPECT_EQ(wave->debug_read_sgpr(3), 0u);
  EXPECT_EQ(wave->ttmp(2), 0xa5u);
  EXPECT_EQ(wave->debug_read_sgpr(4), 0u);
}

TEST(WaveDebugTest, Cdna4TrapWaitsAndReturns) {
  for (const uint32_t helpers : {0u, 1u}) {
    SCOPED_TRACE(helpers);
    WaveDebugFixture fx(ROCJITSU_CODE_ARCH_CDNA4, 1,
                        amdgpu::ComputeUnitCore::Config{}.functional_quantum, helpers);
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
    fx.gpu_mem.write32(kKernelAddr, kSTrapSeven);
    fx.gpu_mem.write32(kKernelAddr + 4, 0xbf82ffff); // s_branch -1
    const uint32_t handler[] = {
        0xb8800142, // s_getreg_b32 s0, hwreg(STATUS,5,1)
        0x806c846c, // s_add_u32 ttmp0, ttmp0, 4
        0x826d806d, // s_addc_u32 ttmp1, ttmp1, 0
        0xbe801f6c, // s_rfe_b64 ttmp[0:1]
    };
    for (size_t i = 0; i < std::size(handler); ++i)
      fx.gpu_mem.write32(kTrapHandlerAddr + 4 * i, handler[i]);
    fx.cu->set_trap_handler_resolver([](const amdgpu::Wavefront &) {
      return amdgpu::ComputeUnitCore::TrapHandlerConfig{kTrapHandlerAddr, 0x3000, true};
    });
    auto *wave = fx.dispatch(kKernelAddr);
    ASSERT_NE(wave, nullptr);
    uint32_t step_count = 0;
    fx.cu->set_single_step_handler([&](amdgpu::Wavefront &stepped) {
      ++step_count;
      EXPECT_EQ(stepped.pc, kKernelAddr + 4);
      stepped.set_debug_halted(true);
      return true;
    });
    wave->set_debug_single_step(true);
    auto &scoreboard = wave->ensure_memory_wait_scoreboard();
    const auto ds = amdgpu::WaitCounterKind::Ds;
    const uint64_t seq = scoreboard.issue(ds);
    scoreboard.add({seq, kKernelAddr - 4, 1, {RegClass::VGPR, 1, 1}, ds, 0xf});
    wave->wait_counters().increment(amdgpu::WaitCounterType::VMCNT);
    fx.cu->step();
    EXPECT_EQ(step_count, 0u);
    EXPECT_FALSE(wave->debug_halted());
    EXPECT_EQ(wave->pc, kKernelAddr);
    EXPECT_FALSE(wave->in_trap_handler());
    EXPECT_FALSE(scoreboard.empty());
    wave->release_wait_counter(amdgpu::WaitCounterType::VMCNT);
    fx.cu->step();
    ASSERT_TRUE(wave->in_trap_handler());
    EXPECT_TRUE(scoreboard.empty());
    for (unsigned step = 0; step < 8 && wave->pc != kKernelAddr + 4; ++step)
      fx.cu->step();
    EXPECT_EQ(step_count, 1u);
    EXPECT_TRUE(wave->debug_halted());
    EXPECT_EQ(wave->debug_read_sgpr(0), 1u);
    EXPECT_EQ(wave->pc, kKernelAddr + 4);
    EXPECT_FALSE(wave->in_trap_handler());
    EXPECT_EQ(wave->status_raw() & (1u << 5), 0u);
  }
}

TEST(WaveDebugTest, STrapExecutesConfiguredTrapHandlerInstructions) {
  WaveDebugFixture fx;
  ASSERT_FALSE(::testing::Test::HasFatalFailure());
  fx.gpu_mem.write32(kKernelAddr, kSTrapBreakpoint);
  fx.gpu_mem.write32(kKernelAddr + 4, kSEndpgm);

  // Assembled for gfx950. The handler writes a proof value to TTMP4,
  // advances the saved PC, restores STATUS.HALT, emits MSG_INTERRUPT, and
  // returns through s_rfe_b64.
  const uint32_t handler[] = {
      0xBEF000FFu, 0x12345678u, // s_mov_b32 ttmp4, 0x12345678
      0x806C846Cu,              // s_add_u32 ttmp0, ttmp0, 4
      0x826D806Du,              // s_addc_u32 ttmp1, ttmp1, 0
      0xBEF800FFu, 0x00002000u, // s_mov_b32 ttmp12, STATUS.HALT
      0xBF900001u,              // s_sendmsg sendmsg(MSG_INTERRUPT)
      0xB978F802u,              // s_setreg_b32 hwreg(HW_REG_STATUS), ttmp12
      0xBE801F6Cu,              // s_rfe_b64 ttmp[0:1]
  };
  for (uint32_t i = 0; i < std::size(handler); ++i)
    fx.gpu_mem.write32(kTrapHandlerAddr + i * 4, handler[i]);

  fx.cu->set_trap_handler_resolver([](const amdgpu::Wavefront &) {
    return amdgpu::ComputeUnitCore::TrapHandlerConfig{kTrapHandlerAddr, 0x3000, true};
  });
  uint32_t interrupt_count = 0;
  fx.cu->set_sendmsg_handler([&](amdgpu::Wavefront &, uint32_t message) {
    if (message == 1)
      ++interrupt_count;
    return message == 1;
  });
  uint32_t completion_count = 0;
  fx.cu->set_trap_completion_handler([&](amdgpu::Wavefront &) { ++completion_count; });

  auto *wf = fx.dispatch(kKernelAddr);
  ASSERT_NE(wf, nullptr);

  fx.cu->step();
  ASSERT_TRUE(wf->in_trap_handler());
  EXPECT_EQ(wf->pc, kTrapHandlerAddr);
  EXPECT_EQ(wf->ttmp(0), kKernelAddr);
  EXPECT_EQ(wf->ttmp(1), 1u << 16);
  EXPECT_EQ(wf->ttmp(14), 0x3000u);

  for (int i = 0; i < 7; ++i)
    fx.cu->step();

  EXPECT_FALSE(wf->in_trap_handler());
  EXPECT_TRUE(wf->debug_halted());
  EXPECT_EQ(wf->pc, kKernelAddr + 4);
  EXPECT_EQ(wf->ttmp(4), 0x12345678u);
  EXPECT_EQ(interrupt_count, 1u);
  EXPECT_EQ(completion_count, 1u);
}

// The ROCr handler temporarily reuses EXEC_LO for MSG_GET_DOORBELL. The wave
// must be reported with the interrupted lane mask, not the handler's doorbell
// response, or resuming a breakpoint changes which lanes execute user code.
TEST(WaveDebugTest, TrapHandlerRestoresInterruptedExecBeforeReporting) {
  WaveDebugFixture fx;
  ASSERT_FALSE(::testing::Test::HasFatalFailure());
  fx.gpu_mem.write32(kKernelAddr, kSTrapBreakpoint);
  fx.gpu_mem.write32(kKernelAddr + 4, kSEndpgm);

  const uint32_t handler[] = {
      0xBEEE017Eu,              // s_mov_b64 ttmp[2:3], exec
      0x806C846Cu,              // s_add_u32 ttmp0, ttmp0, 4
      0x826D806Du,              // s_addc_u32 ttmp1, ttmp1, 0
      0xBEFE00FFu, 0x80000000u, // s_mov_b32 exec_lo, 0x80000000
      0xBF90000Au,              // s_sendmsg sendmsg(MSG_GET_DOORBELL)
      0xBEF800FFu, 0x00002000u, // s_mov_b32 ttmp12, STATUS.HALT
      0xBF900001u,              // s_sendmsg sendmsg(MSG_INTERRUPT)
      0xB978F802u,              // s_setreg_b32 hwreg(HW_REG_STATUS), ttmp12
      0xBEFE016Eu,              // s_mov_b64 exec, ttmp[2:3]
      0xBE801F6Cu,              // s_rfe_b64 ttmp[0:1]
  };
  for (uint32_t i = 0; i < std::size(handler); ++i)
    fx.gpu_mem.write32(kTrapHandlerAddr + i * 4, handler[i]);

  fx.cu->set_trap_handler_resolver([](const amdgpu::Wavefront &) {
    return amdgpu::ComputeUnitCore::TrapHandlerConfig{kTrapHandlerAddr, 0, true};
  });
  fx.cu->set_sendmsg_handler([](amdgpu::Wavefront &wave, uint32_t message) {
    if (message == 10)
      wave.set_exec((wave.exec() & 0xFFFFFFFF00000000ULL) | 7u);
    return message == 1 || message == 10;
  });

  constexpr uint64_t kInterruptedExec = 0x00FF00FF00FF00FFULL;
  auto *wf = fx.dispatch(kKernelAddr);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(kInterruptedExec);

  for (int i = 0; i < 11 && !wf->debug_halted(); ++i)
    fx.cu->step();

  ASSERT_TRUE(wf->debug_halted());
  EXPECT_EQ(wf->exec(), kInterruptedExec);
}

// Restoring the interrupted EXEC belongs to returning from the handler, not to
// stopping for a debugger. A handler that returns without stopping the wave
// used to leave its own mask installed, and nothing later put the
// application's back: the kernel ran on with every lane active, silently
// un-diverging a branch. gdb.rocm/lane-info.exp catches it as lanes that had
// converged out of a branch being reported active again.
TEST(WaveDebugTest, TrapHandlerRestoresInterruptedExecWhenReturningWithoutStopping) {
  WaveDebugFixture fx;
  ASSERT_FALSE(::testing::Test::HasFatalFailure());
  fx.gpu_mem.write32(kKernelAddr, kSTrapBreakpoint);
  fx.gpu_mem.write32(kKernelAddr + 4, kSEndpgm);

  // Same shape as the handler above with the halt removed, so the wave runs on
  // after s_rfe instead of stopping.
  const uint32_t handler[] = {
      0xBEEE017Eu,              // s_mov_b64 ttmp[2:3], exec
      0x806C846Cu,              // s_add_u32 ttmp0, ttmp0, 4
      0x826D806Du,              // s_addc_u32 ttmp1, ttmp1, 0
      0xBEFE00FFu, 0x80000000u, // s_mov_b32 exec_lo, 0x80000000
      0xBF90000Au,              // s_sendmsg sendmsg(MSG_GET_DOORBELL)
      0xBEFE016Eu,              // s_mov_b64 exec, ttmp[2:3]
      0xBE801F6Cu,              // s_rfe_b64 ttmp[0:1]
  };
  for (uint32_t i = 0; i < std::size(handler); ++i)
    fx.gpu_mem.write32(kTrapHandlerAddr + i * 4, handler[i]);

  fx.cu->set_trap_handler_resolver([](const amdgpu::Wavefront &) {
    return amdgpu::ComputeUnitCore::TrapHandlerConfig{kTrapHandlerAddr, 0, true};
  });
  fx.cu->set_sendmsg_handler([](amdgpu::Wavefront &wave, uint32_t message) {
    if (message == 10)
      wave.set_exec((wave.exec() & 0xFFFFFFFF00000000ULL) | 7u);
    return message == 1 || message == 10;
  });

  // Lanes 0, 2 and 4 of a five-lane wave, the shape lane-info.exp produces.
  constexpr uint64_t kInterruptedExec = 0x15ULL;
  auto *wf = fx.dispatch(kKernelAddr);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(kInterruptedExec);

  for (int i = 0; i < 6 && wf->pc == kKernelAddr; ++i)
    fx.cu->step();
  // Every assertion below is satisfied by a wave that never entered the handler
  // at all -- EXEC is unmodified, nothing halts it for the debugger, and
  // s_endpgm ends it. Anchor on the handler actually running, or a regression
  // that stops dispatching into it passes this test.
  ASSERT_TRUE(wf->in_trap_handler());
  for (int i = 0; i < 8 && wf->in_trap_handler(); ++i)
    fx.cu->step();

  ASSERT_FALSE(wf->in_trap_handler());
  EXPECT_FALSE(wf->debug_halted());
  EXPECT_EQ(wf->exec(), kInterruptedExec);

  // Run the wave off the end. Leaving one resident past the fixture is not
  // inert: the CU pool-allocates decoded instructions and frees the pool with
  // the decoder.
  for (int i = 0; i < 4 && !wf->is_halted(); ++i)
    fx.cu->step();
  EXPECT_TRUE(wf->is_halted());
}

// s_rfe_restore_b64 is the GFX9 legacy trap return: the manuals define it as
// `PRIV = 0; PC = S0.u64` with the second operand ignored, the same return as
// s_rfe_b64. Returning through it must leave the wave exactly as s_rfe_b64
// would, whatever its second operand holds and however long its encoding is.
TEST(WaveDebugTest, RfeRestoreReturnsFromTheTrapHandlerLikeRfe) {
  struct ReturnForm {
    const char *name;
    std::vector<uint32_t> words;
  };
  const ReturnForm forms[] = {
      {"sgpr", {0x9580016Cu}},                 // s_rfe_restore_b64 ttmp[0:1], s1
      {"literal", {0x9580FF6Cu, 0xDEADBEEFu}}, // s_rfe_restore_b64 ttmp[0:1], 0xdeadbeef
  };
  for (auto arch : {ROCJITSU_CODE_ARCH_CDNA1, ROCJITSU_CODE_ARCH_CDNA2, ROCJITSU_CODE_ARCH_CDNA3,
                    ROCJITSU_CODE_ARCH_CDNA4}) {
    for (const auto &form : forms) {
      SCOPED_TRACE(form.name);
      SCOPED_TRACE(static_cast<int>(arch));
      WaveDebugFixture fx(arch);
      fx.gpu_mem.write32(kKernelAddr, kSTrapBreakpoint);
      fx.gpu_mem.write32(kKernelAddr + 4, kSEndpgm);

      std::vector<uint32_t> handler = {
          0x806C846Cu,              // s_add_u32 ttmp0, ttmp0, 4
          0x826D806Du,              // s_addc_u32 ttmp1, ttmp1, 0
          0xBEFE00FFu, 0x80000000u, // s_mov_b32 exec_lo, 0x80000000
      };
      handler.insert(handler.end(), form.words.begin(), form.words.end());
      for (uint32_t i = 0; i < handler.size(); ++i)
        fx.gpu_mem.write32(kTrapHandlerAddr + i * 4, handler[i]);

      fx.cu->set_trap_handler_resolver([](const amdgpu::Wavefront &) {
        return amdgpu::ComputeUnitCore::TrapHandlerConfig{kTrapHandlerAddr, 0, true};
      });

      constexpr uint64_t kInterruptedExec = 0x15ULL;
      auto *wf = fx.dispatch(kKernelAddr);
      ASSERT_NE(wf, nullptr);
      wf->set_exec(kInterruptedExec);
      fx.cu->write_sgpr(wf->sgpr_alloc().base + 1, 0xFFFFFFFFu);

      fx.cu->step();
      ASSERT_TRUE(wf->in_trap_handler());
      ASSERT_EQ(wf->pc, kTrapHandlerAddr);
      for (int i = 0; i < 4 && wf->in_trap_handler(); ++i)
        fx.cu->step();

      ASSERT_FALSE(wf->in_trap_handler());
      EXPECT_FALSE(wf->debug_halted());
      EXPECT_EQ(wf->pc, kKernelAddr + 4);
      EXPECT_EQ(wf->exec(), 0x80000000u);
      EXPECT_FALSE(wf->instruction_execution_failed());

      for (int i = 0; i < 4 && !wf->is_halted(); ++i)
        fx.cu->step();
      EXPECT_TRUE(wf->is_halted());
    }
  }
}

// The ignored second operand is not a read: a load still pending into it must
// not be charged to the return. Charging it there also used up the hazard, so
// a resumed read that skipped the wait went unreported.
TEST(WaveDebugTest, RfeRestoreDoesNotWaitOnItsIgnoredOperand) {
  for (auto arch : {ROCJITSU_CODE_ARCH_CDNA1, ROCJITSU_CODE_ARCH_CDNA2, ROCJITSU_CODE_ARCH_CDNA3,
                    ROCJITSU_CODE_ARCH_CDNA4}) {
    for (bool resumed_wait : {false, true}) {
      SCOPED_TRACE(static_cast<int>(arch));
      SCOPED_TRACE(resumed_wait);
      WaveDebugFixture fx(arch);
      std::vector<uint32_t> kernel = {kSTrapBreakpoint};
      if (resumed_wait)
        kernel.push_back(0xBF8C0000u); // s_waitcnt 0
      kernel.insert(kernel.end(), {
                                      0xBE850001u, // s_mov_b32 s5, s1
                                      kSEndpgm,
                                  });
      for (uint32_t i = 0; i < kernel.size(); ++i)
        fx.gpu_mem.write32(kKernelAddr + i * 4, kernel[i]);

      const uint32_t handler[] = {
          0x806C846Cu,              // s_add_u32 ttmp0, ttmp0, 4
          0x826D806Du,              // s_addc_u32 ttmp1, ttmp1, 0
          0xC0020041u, 0x00000000u, // s_load_dword s1, s[2:3], 0
          0x9580016Cu,              // s_rfe_restore_b64 ttmp[0:1], s1
      };
      for (uint32_t i = 0; i < std::size(handler); ++i)
        fx.gpu_mem.write32(kTrapHandlerAddr + i * 4, handler[i]);

      fx.cu->set_trap_handler_resolver([](const amdgpu::Wavefront &) {
        return amdgpu::ComputeUnitCore::TrapHandlerConfig{kTrapHandlerAddr, 0, true};
      });

      // The handler's scalar load needs the cache hierarchy's functional path.
      fx.l2.set_backing_memory(&fx.gpu_mem);

      auto *wf = fx.dispatch(kKernelAddr);
      ASSERT_NE(wf, nullptr);
      const uint32_t sbase = wf->sgpr_alloc().base;
      fx.cu->write_sgpr(sbase + 2, kKernelAddr);
      fx.cu->write_sgpr(sbase + 3, 0);

      fx.cu->step();
      ASSERT_TRUE(wf->in_trap_handler());
      for (int i = 0; i < 4 && wf->in_trap_handler(); ++i)
        fx.cu->step();
      ASSERT_FALSE(wf->in_trap_handler());
      EXPECT_EQ(fx.cu->memory_wait_diagnostic_count(), 0u);

      const uint64_t endpgm = kKernelAddr + (kernel.size() - 1) * 4;
      for (int i = 0; i < 4 && wf->pc != endpgm; ++i)
        fx.cu->step();
      ASSERT_EQ(wf->pc, endpgm);
      EXPECT_FALSE(wf->instruction_execution_failed());
      EXPECT_EQ(fx.cu->read_sgpr(sbase + 5), kSTrapBreakpoint);
      EXPECT_EQ(fx.cu->memory_wait_diagnostic_count(), resumed_wait ? 0u : 1u);
    }
  }
}

// Only the second operand is ignored: the return address is a real read, so
// loading it straight into the return without a wait is reported there.
TEST(WaveDebugTest, RfeRestoreWaitsOnItsReturnAddress) {
  constexpr uint32_t kSavedPcAddr = kTrapHandlerAddr + 0x100;
  for (auto arch : {ROCJITSU_CODE_ARCH_CDNA1, ROCJITSU_CODE_ARCH_CDNA2, ROCJITSU_CODE_ARCH_CDNA3,
                    ROCJITSU_CODE_ARCH_CDNA4}) {
    SCOPED_TRACE(static_cast<int>(arch));
    WaveDebugFixture fx(arch);
    fx.gpu_mem.write32(kKernelAddr, kSTrapBreakpoint);
    fx.gpu_mem.write32(kKernelAddr + 4, kSEndpgm);
    fx.gpu_mem.write32(kSavedPcAddr, kKernelAddr + 4);
    fx.gpu_mem.write32(kSavedPcAddr + 4, 0);

    const uint32_t handler[] = {
        0xC0060001u, 0x00000000u, // s_load_dwordx2 s[0:1], s[2:3], 0
        0x95800500u,              // s_rfe_restore_b64 s[0:1], s5
    };
    for (uint32_t i = 0; i < std::size(handler); ++i)
      fx.gpu_mem.write32(kTrapHandlerAddr + i * 4, handler[i]);

    fx.cu->set_trap_handler_resolver([](const amdgpu::Wavefront &) {
      return amdgpu::ComputeUnitCore::TrapHandlerConfig{kTrapHandlerAddr, 0, true};
    });
    // The handler's scalar load needs the cache hierarchy's functional path.
    fx.l2.set_backing_memory(&fx.gpu_mem);

    auto *wf = fx.dispatch(kKernelAddr);
    ASSERT_NE(wf, nullptr);
    const uint32_t sbase = wf->sgpr_alloc().base;
    fx.cu->write_sgpr(sbase + 2, kSavedPcAddr);
    fx.cu->write_sgpr(sbase + 3, 0);

    fx.cu->step();
    ASSERT_TRUE(wf->in_trap_handler());
    for (int i = 0; i < 2 && wf->in_trap_handler(); ++i)
      fx.cu->step();
    ASSERT_FALSE(wf->in_trap_handler());
    EXPECT_EQ(wf->pc, kKernelAddr + 4);
    EXPECT_FALSE(wf->instruction_execution_failed());
    EXPECT_EQ(fx.cu->memory_wait_diagnostic_count(), 1u);
  }
}

TEST(WaveDebugTest, UnmappedInstructionFetchReportsMemoryViolationAtBranchTarget) {
  WaveDebugFixture fx;
  ASSERT_FALSE(::testing::Test::HasFatalFailure());
  constexpr uint32_t kProcessId = 7;
  constexpr uint64_t kBranchTarget = 0x100;
  std::vector<uint8_t> kernel_page(amdgpu::GpuMemory::PAGE_SIZE);
  KfdProcess::PageTable page_table;
  util::DistributedSharedMutex page_table_mutex;
  page_table[kKernelAddr >> amdgpu::GpuMemory::PAGE_SHIFT] = {kernel_page.data(),
                                                              amdgpu::Mtype::RW};
  fx.gpu_mem.register_process(kProcessId, &page_table, &page_table_mutex);

  fx.gpu_mem.write32(kKernelAddr, 0xBE801D00u, kProcessId); // s_setpc_b64 s[0:1]
  auto *wf = fx.dispatch(kKernelAddr);
  ASSERT_NE(wf, nullptr);
  wf->set_process_id(kProcessId);
  fx.cu->write_sgpr(wf->sgpr_alloc().base, static_cast<uint32_t>(kBranchTarget));
  fx.cu->write_sgpr(wf->sgpr_alloc().base + 1, 0);

  uint64_t fault_address = 0;
  fx.cu->set_memory_violation_handler(
      [&](amdgpu::Wavefront &faulting_wave, uint64_t address, bool is_write, uint64_t) {
        EXPECT_EQ(&faulting_wave, wf);
        EXPECT_FALSE(is_write);
        fault_address = address;
        faulting_wave.debug_trap(0);
        return true;
      });

  fx.cu->step();
  EXPECT_EQ(wf->pc, kBranchTarget);
  EXPECT_FALSE(wf->debug_halted());

  fx.cu->step();
  EXPECT_EQ(fault_address, kBranchTarget);
  EXPECT_TRUE(wf->debug_halted());
  EXPECT_EQ(wf->pc, kBranchTarget);

  fx.gpu_mem.unregister_process(kProcessId);
}

TEST(WaveDebugTest, Rdna4SuppressedReturningAtomicSkipsUnmappedDebuggerProbe) {
  WaveDebugFixture fx(ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_NE(fx.cu, nullptr);
  constexpr uint32_t process = 7;
  constexpr uint32_t unmapped = 0x4000;
  std::vector<uint8_t> kernel_page(amdgpu::GpuMemory::PAGE_SIZE);
  KfdProcess::PageTable table;
  util::DistributedSharedMutex mutex;
  table[kKernelAddr >> amdgpu::GpuMemory::PAGE_SHIFT] = {kernel_page.data(), amdgpu::Mtype::RW};
  fx.gpu_mem.register_process(process, &table, &mutex);

  // global_atomic_add_u32 v64/v0, v[2:3], v4, off th:TH_ATOMIC_RETURN.
  constexpr std::array<uint32_t, 6> program{0xee0d407c, 0x02100040, 2, 0xee0d407c, 0x02100000, 2};
  for (size_t i = 0; i < program.size(); ++i)
    fx.gpu_mem.write32(kKernelAddr + 4 * i, program[i], process);
  auto *wave = fx.dispatch(kKernelAddr);
  ASSERT_NE(wave, nullptr);
  ASSERT_EQ(wave->vgpr_alloc().count, 64u);
  wave->set_process_id(process);
  wave->set_exec(1);
  const auto vb = wave->vgpr_alloc().base;
  fx.cu->write_vgpr(vb + 2, 0, unmapped);
  fx.cu->write_vgpr(vb + 3, 0, 0);
  fx.cu->write_vgpr(vb + 4, 0, 1);
  fx.cu->set_debug_active(true);
  unsigned faults = 0;
  fx.cu->set_memory_violation_handler(
      [&](amdgpu::Wavefront &faulting_wave, uint64_t address, bool write, uint64_t issue_pc) {
        ++faults;
        EXPECT_EQ(address, unmapped);
        EXPECT_TRUE(write);
        EXPECT_EQ(issue_pc, kKernelAddr + 12);
        faulting_wave.debug_trap(0);
        return true;
      });
  fx.cu->step();
  EXPECT_FALSE(wave->debug_halted());
  EXPECT_EQ(wave->pc, kKernelAddr + 12);
  EXPECT_EQ(faults, 0u);
  EXPECT_EQ(wave->exec(), 1u);
  fx.cu->step();
  EXPECT_TRUE(wave->debug_halted());
  EXPECT_EQ(faults, 1u);
  fx.gpu_mem.unregister_process(process);
}

TEST(WaveDebugTest, UnmappedScalarLoadReportsMemoryViolationAfterInstruction) {
  WaveDebugFixture fx;
  ASSERT_FALSE(::testing::Test::HasFatalFailure());
  constexpr uint32_t kProcessId = 7;
  std::vector<uint8_t> kernel_page(amdgpu::GpuMemory::PAGE_SIZE);
  KfdProcess::PageTable page_table;
  util::DistributedSharedMutex page_table_mutex;
  page_table[kKernelAddr >> amdgpu::GpuMemory::PAGE_SHIFT] = {kernel_page.data(),
                                                              amdgpu::Mtype::RW};
  fx.gpu_mem.register_process(kProcessId, &page_table, &page_table_mutex);

  fx.gpu_mem.write32(kKernelAddr, 0xC0020000u, kProcessId);     // s_load_dword s0, s[0:1]
  fx.gpu_mem.write32(kKernelAddr + 4, 0, kProcessId);           // immediate offset 0
  fx.gpu_mem.write32(kKernelAddr + 8, 0xBF800000u, kProcessId); // s_nop 0
  auto *wave = fx.dispatch(kKernelAddr);
  ASSERT_NE(wave, nullptr);
  wave->set_process_id(kProcessId);
  fx.cu->set_debug_active(true);

  uint64_t fault_address = ~uint64_t{0};
  fx.cu->set_memory_violation_handler(
      [&](amdgpu::Wavefront &faulting_wave, uint64_t address, bool is_write, uint64_t) {
        EXPECT_EQ(&faulting_wave, wave);
        EXPECT_FALSE(is_write);
        fault_address = address;
        faulting_wave.debug_trap(0);
        return true;
      });

  fx.cu->step();
  EXPECT_EQ(fault_address, 0u);
  EXPECT_TRUE(wave->debug_halted());
  EXPECT_EQ(wave->pc, kKernelAddr + 8);

  fx.gpu_mem.unregister_process(kProcessId);
}

// Runtime suspension (queue_percentage 0) and debugger suspension overlap in
// time and are lifted by different actors. Representing both with one wave bit
// let a runtime resume clear a debugger pause, and a debugger or CWSR resume
// clear an active runtime pause -- in either case the wave started running
// while something still meant it to be stopped.
TEST(WaveDebugTest, RuntimeAndDebuggerSuspensionDoNotClobberEachOther) {
  WaveDebugFixture fx;
  ASSERT_FALSE(::testing::Test::HasFatalFailure());
  fx.gpu_mem.write32(kKernelAddr, kSEndpgm);
  auto *wave = fx.dispatch(kKernelAddr);
  ASSERT_NE(wave, nullptr);
  ASSERT_FALSE(wave->debug_paused());

  // Both reasons applied; releasing either one alone must not resume the wave.
  wave->set_runtime_suspended(true);
  wave->set_debug_suspended(true);
  EXPECT_TRUE(wave->debug_paused());

  wave->set_debug_suspended(false);
  EXPECT_TRUE(wave->runtime_suspended());
  EXPECT_TRUE(wave->debug_paused()) << "a debugger resume lifted the runtime pause";

  wave->set_runtime_suspended(false);
  EXPECT_FALSE(wave->debug_paused());

  // And the other way round.
  wave->set_runtime_suspended(true);
  wave->set_debug_suspended(true);
  wave->set_runtime_suspended(false);
  EXPECT_TRUE(wave->debug_suspended());
  EXPECT_TRUE(wave->debug_paused()) << "a runtime resume lifted the debugger pause";

  // A halt is a third, independent reason.
  wave->set_debug_suspended(false);
  wave->set_debug_halted(true);
  EXPECT_TRUE(wave->debug_paused());
}

// Trap entry and CWSR must agree on where a wave's dispatch identity lives, or
// rocm-dbgapi correlates the stopped wave to the wrong workgroup. On the gfx9
// layout that is TTMP8/9/10 = workgroup id x/y/z, which is exactly what
// cwsr.cpp serializes; trap entry used to write the flat wg_id() into TTMP8 and
// zero the other two.
TEST(WaveDebugTest, TrapEntryPublishesTheWorkgroupCoordinateCwsrSerializes) {
  WaveDebugFixture fx;
  ASSERT_FALSE(::testing::Test::HasFatalFailure());
  fx.gpu_mem.write32(kKernelAddr, kSTrapBreakpoint);
  fx.gpu_mem.write32(kKernelAddr + 4, kSEndpgm);
  fx.gpu_mem.write32(kTrapHandlerAddr, kSEndpgm);
  fx.cu->set_trap_handler_resolver([](const amdgpu::Wavefront &) {
    return amdgpu::ComputeUnitCore::TrapHandlerConfig{kTrapHandlerAddr, 0, true};
  });

  auto *wf = fx.dispatch(kKernelAddr);
  ASSERT_NE(wf, nullptr);
  wf->set_wg_coord(3, 5, 7);

  fx.cu->step();
  ASSERT_TRUE(wf->in_trap_handler());

  const auto &wg = wf->wg_coord();
  EXPECT_EQ(wf->ttmp(8), wg[0]);
  EXPECT_EQ(wf->ttmp(9), wg[1]);
  EXPECT_EQ(wf->ttmp(10), wg[2]);
  EXPECT_EQ(wf->ttmp(9), 5u) << "the y coordinate was zeroed on trap entry";
  EXPECT_EQ(wf->ttmp(10), 7u) << "the z coordinate was zeroed on trap entry";
}

// S_SENDMSGHALT halts the wave -- that is the instruction, not a debugger
// artefact. It has to publish the architectural STATUS.HALT and not only the
// scheduler's private flag, because s_rfe reads STATUS.HALT on the way out of
// the handler to decide whether the wave stays stopped. With only the private
// flag set, s_rfe saw 0 for a wave s_sendmsghalt had already halted and left
// single-step armed, so resuming took an extra step.
TEST(WaveDebugTest, SendmsghaltPublishesArchitecturalStatusHalt) {
  WaveDebugFixture fx;
  ASSERT_FALSE(::testing::Test::HasFatalFailure());
  constexpr uint32_t kStatusHalt = 1u << 13;
  constexpr uint32_t kSSendmsghaltInterrupt = 0xBF910001u; // s_sendmsghalt sendmsg(MSG_INTERRUPT)

  fx.gpu_mem.write32(kKernelAddr, kSTrapBreakpoint);
  fx.gpu_mem.write32(kKernelAddr + 4, kSEndpgm);

  const uint32_t handler[] = {
      0x806C846Cu,            // s_add_u32  ttmp0, ttmp0, 4
      0x826D806Du,            // s_addc_u32 ttmp1, ttmp1, 0
      kSSendmsghaltInterrupt, // s_sendmsghalt sendmsg(MSG_INTERRUPT)
      0xBE801F6Cu,            // s_rfe_b64 ttmp[0:1]
  };
  for (uint32_t i = 0; i < std::size(handler); ++i)
    fx.gpu_mem.write32(kTrapHandlerAddr + i * 4, handler[i]);

  fx.cu->set_trap_handler_resolver([](const amdgpu::Wavefront &) {
    return amdgpu::ComputeUnitCore::TrapHandlerConfig{kTrapHandlerAddr, 0, true};
  });
  fx.cu->set_sendmsg_handler([](amdgpu::Wavefront &, uint32_t message) { return message == 1; });

  auto *wf = fx.dispatch(kKernelAddr);
  ASSERT_NE(wf, nullptr);
  wf->set_debug_single_step(true);

  fx.cu->step(); // s_trap -> handler
  ASSERT_TRUE(wf->in_trap_handler());
  ASSERT_EQ(wf->status_raw() & kStatusHalt, 0u) << "STATUS.HALT set before s_sendmsghalt";

  for (int i = 0; i < 3 && !wf->debug_halted(); ++i)
    fx.cu->step();

  EXPECT_TRUE(wf->debug_halted());
  EXPECT_NE(wf->status_raw() & kStatusHalt, 0u)
      << "s_sendmsghalt halted the wave without publishing STATUS.HALT";
  // Which instruction raised the bit is the only thing that separates this stop
  // from a handler that raised HALT with s_setreg and then returned. The two
  // want opposite treatment on resume and are identical in the CWSR record.
  EXPECT_TRUE(wf->self_halted())
      << "s_sendmsghalt raised STATUS.HALT without recording that it did";
}

// The other origin of STATUS.HALT. The ROCr handler raises it through s_setreg
// on the way out, and there the bit means "keep this wave stopped" -- so a
// resume must not clear it. Nothing in the CWSR record distinguishes that from
// a wave halted at s_sendmsghalt, which wants exactly the opposite, so the
// provenance marker has to come out clear on this path.
TEST(WaveDebugTest, HandlerSetregHaltIsNotAttributedToSendmsghalt) {
  WaveDebugFixture fx;
  ASSERT_FALSE(::testing::Test::HasFatalFailure());
  constexpr uint32_t kStatusHalt = 1u << 13;

  fx.gpu_mem.write32(kKernelAddr, kSTrapBreakpoint);
  fx.gpu_mem.write32(kKernelAddr + 4, kSEndpgm);

  // SOPK s_setreg_imm32_b32 (op 20), hwreg id 2 = STATUS, offset 13, size 1 --
  // simm16 = 2 | (13 << 6) = 0x342. This is the ROCr handler's halt sequence.
  const uint32_t handler[] = {
      0x806C846Cu,              // s_add_u32  ttmp0, ttmp0, 4
      0x826D806Du,              // s_addc_u32 ttmp1, ttmp1, 0
      0xBA000342u, 0x00000001u, // s_setreg_imm32_b32 hwreg(STATUS, 13, 1), 1
      0xBE801F6Cu,              // s_rfe_b64 ttmp[0:1]
  };
  for (uint32_t i = 0; i < std::size(handler); ++i)
    fx.gpu_mem.write32(kTrapHandlerAddr + i * 4, handler[i]);

  fx.cu->set_trap_handler_resolver([](const amdgpu::Wavefront &) {
    return amdgpu::ComputeUnitCore::TrapHandlerConfig{kTrapHandlerAddr, 0, true};
  });

  auto *wf = fx.dispatch(kKernelAddr);
  ASSERT_NE(wf, nullptr);

  fx.cu->step(); // s_trap -> handler
  ASSERT_TRUE(wf->in_trap_handler());

  for (int i = 0; i < 4 && !wf->debug_halted(); ++i)
    fx.cu->step();

  ASSERT_NE(wf->status_raw() & kStatusHalt, 0u) << "the handler's s_setreg did not raise HALT";
  EXPECT_FALSE(wf->self_halted())
      << "a handler-raised HALT was attributed to s_sendmsghalt, so a resume would clear it "
         "and lose the breakpoint";
}

// Provenance is per stop, not per wave. A marker left behind by a resumed
// s_sendmsghalt would make the *next* stop -- one the handler halted with
// s_setreg -- look self-halted, and the resume after that would clear a HALT it
// must leave alone.
TEST(WaveDebugTest, SelfHaltedMarkerDoesNotSurviveTheStopThatSetIt) {
  WaveDebugFixture fx;
  ASSERT_FALSE(::testing::Test::HasFatalFailure());
  constexpr uint32_t kSSendmsghaltInterrupt = 0xBF910001u;

  fx.gpu_mem.write32(kKernelAddr, kSTrapBreakpoint);
  fx.gpu_mem.write32(kKernelAddr + 4, kSEndpgm);

  const uint32_t handler[] = {
      0x806C846Cu,            // s_add_u32  ttmp0, ttmp0, 4
      0x826D806Du,            // s_addc_u32 ttmp1, ttmp1, 0
      kSSendmsghaltInterrupt, // s_sendmsghalt sendmsg(MSG_INTERRUPT)
      0xBE801F6Cu,            // s_rfe_b64 ttmp[0:1]
  };
  for (uint32_t i = 0; i < std::size(handler); ++i)
    fx.gpu_mem.write32(kTrapHandlerAddr + i * 4, handler[i]);

  fx.cu->set_trap_handler_resolver([](const amdgpu::Wavefront &) {
    return amdgpu::ComputeUnitCore::TrapHandlerConfig{kTrapHandlerAddr, 0, true};
  });
  fx.cu->set_sendmsg_handler([](amdgpu::Wavefront &, uint32_t message) { return message == 1; });

  auto *wf = fx.dispatch(kKernelAddr);
  ASSERT_NE(wf, nullptr);

  fx.cu->step(); // s_trap -> handler
  for (int i = 0; i < 3 && !wf->self_halted(); ++i)
    fx.cu->step();
  ASSERT_TRUE(wf->self_halted());

  // What a debugger resume does: drop both halves of the stop, then let the
  // handler return.
  wf->set_status_halt(false);
  wf->set_self_halted(false);
  wf->set_debug_halted(false);
  for (int i = 0; i < 3 && wf->in_trap_handler(); ++i)
    fx.cu->step();
  ASSERT_FALSE(wf->in_trap_handler()) << "the handler never returned";
  EXPECT_FALSE(wf->self_halted());

  for (int i = 0; i < 4 && !wf->is_halted(); ++i)
    fx.cu->step();
  EXPECT_TRUE(wf->is_halted());
}

// S_SENDMSGHALT with nobody attached. The instruction halts the wave whether or
// not a debugger is there to resume it, so the flags it raises must be exactly
// the same ones a debugger stop raises -- and no more. What this pins is that
// an undebugged run does not diverge: the halt is a stop, not a corruption. The
// wave keeps the interrupted EXEC rather than running on under the handler's
// mask, its slot is not silently retired behind the halt, and clearing the two
// halves of the stop -- which is all a session release does -- puts it back on
// the scheduler and lets it retire.
TEST(WaveDebugTest, SendmsghaltWithoutADebugSessionHaltsAndStaysReleasable) {
  WaveDebugFixture fx;
  ASSERT_FALSE(::testing::Test::HasFatalFailure());
  constexpr uint32_t kStatusHalt = 1u << 13;
  constexpr uint32_t kSSendmsghaltInterrupt = 0xBF910001u;
  constexpr uint64_t kApplicationExec = 0x5u;

  fx.gpu_mem.write32(kKernelAddr, kSTrapBreakpoint);
  fx.gpu_mem.write32(kKernelAddr + 4, kSEndpgm);

  // The handler runs under its own mask, the way the real one does on its way
  // to MSG_INTERRUPT, so a lost EXEC restore is visible here.
  const uint32_t handler[] = {
      0xBEEE017Eu, // s_mov_b64 ttmp[2:3], exec
      0x806C846Cu, // s_add_u32  ttmp0, ttmp0, 4
      0x826D806Du, // s_addc_u32 ttmp1, ttmp1, 0
      0xBEFE00FFu,
      0x80000000u,            // s_mov_b32 exec_lo, 0x80000000
      kSSendmsghaltInterrupt, // s_sendmsghalt sendmsg(MSG_INTERRUPT)
      0xBEFE016Eu,            // s_mov_b64 exec, ttmp[2:3]
      0xBE801F6Cu,            // s_rfe_b64 ttmp[0:1]
  };
  for (uint32_t i = 0; i < std::size(handler); ++i)
    fx.gpu_mem.write32(kTrapHandlerAddr + i * 4, handler[i]);

  // debug_enabled clear: no debugger, so TTMP13 bit 23 stays 0 and nothing in
  // the driver is watching this wave.
  fx.cu->set_trap_handler_resolver([](const amdgpu::Wavefront &) {
    return amdgpu::ComputeUnitCore::TrapHandlerConfig{kTrapHandlerAddr, 0, false};
  });
  fx.cu->set_sendmsg_handler([](amdgpu::Wavefront &, uint32_t message) { return message == 1; });

  auto *wf = fx.dispatch(kKernelAddr);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(kApplicationExec);

  fx.cu->step(); // s_trap -> handler
  ASSERT_TRUE(wf->in_trap_handler());

  for (int i = 0; i < 6 && !wf->debug_halted(); ++i)
    fx.cu->step();

  EXPECT_TRUE(wf->debug_halted()) << "s_sendmsghalt did not halt an undebugged wave";
  EXPECT_NE(wf->status_raw() & kStatusHalt, 0u);
  EXPECT_TRUE(wf->self_halted())
      << "an undebugged s_sendmsghalt must record provenance too, or the release below "
         "leaves STATUS.HALT set and the wave re-halts at s_rfe";
  EXPECT_FALSE(wf->is_halted()) << "the slot was retired behind a wave that is only stopped";

  // Stepping a stopped wave must not advance it, with or without a debugger.
  const uint64_t stopped_pc = wf->pc;
  fx.cu->step();
  EXPECT_EQ(wf->pc, stopped_pc);

  // What a session release does: drop both halves of the stop. Nothing else is
  // needed to make the wave runnable again.
  wf->set_status_halt(false);
  wf->set_self_halted(false);
  wf->set_debug_halted(false);
  for (int i = 0; i < 4 && wf->in_trap_handler(); ++i)
    fx.cu->step();
  ASSERT_FALSE(wf->in_trap_handler()) << "the handler never returned";
  EXPECT_EQ(wf->exec(), kApplicationExec)
      << "the application resumed under the handler's EXEC mask";

  for (int i = 0; i < 4 && !wf->is_halted(); ++i)
    fx.cu->step();
  EXPECT_TRUE(wf->is_halted()) << "the wave never retired, so its dispatch cannot complete";
}

// The CWSR codec reproduces the gfx9.4 and gfx12.5 record layouts, and the
// differences elsewhere are not confined to one field. Serializing a wave from an
// unmodelled architecture would hand rocm-dbgapi an image it decodes against a
// different layout, so the codec has to say plainly which ones it covers.
TEST(WaveDebugTest, CwsrLayoutIsModelledForGfx94AndGfx1250) {
  EXPECT_EQ(kmd::cwsr_layout_kind(ROCJITSU_CODE_ARCH_CDNA3), kmd::CwsrLayoutKind::Gfx9_4);
  EXPECT_EQ(kmd::cwsr_layout_kind(ROCJITSU_CODE_ARCH_CDNA4), kmd::CwsrLayoutKind::Gfx9_4);
  EXPECT_EQ(kmd::cwsr_layout_kind(ROCJITSU_CODE_ARCH_CDNA5), kmd::CwsrLayoutKind::Gfx12_5);
  EXPECT_TRUE(kmd::cwsr_layout_modelled(ROCJITSU_CODE_ARCH_CDNA3)); // gfx942
  EXPECT_TRUE(kmd::cwsr_layout_modelled(ROCJITSU_CODE_ARCH_CDNA4)); // gfx950
  EXPECT_TRUE(kmd::cwsr_layout_modelled(ROCJITSU_CODE_ARCH_CDNA5)); // gfx1250

  // gfx908 saves an ACC-VGPR block and gfx908/gfx90a keep the packet id in
  // TTMP6, so being gfx9 is not enough.
  EXPECT_FALSE(kmd::cwsr_layout_modelled(ROCJITSU_CODE_ARCH_CDNA1));
  EXPECT_FALSE(kmd::cwsr_layout_modelled(ROCJITSU_CODE_ARCH_CDNA2));

  // RDNA layouts remain outside this codec.
  for (rj_code_arch_t arch :
       {ROCJITSU_CODE_ARCH_RDNA1, ROCJITSU_CODE_ARCH_RDNA2, ROCJITSU_CODE_ARCH_RDNA3,
        ROCJITSU_CODE_ARCH_RDNA3_5, ROCJITSU_CODE_ARCH_RDNA4})
    EXPECT_EQ(kmd::cwsr_layout_kind(arch), kmd::CwsrLayoutKind::Unsupported)
        << "arch " << static_cast<int>(arch);
}

// TRAPSTS.EXCP is architecturally sticky, so "the bit changed" is not the same
// question as "this instruction raised the cause". Delivering on the rising
// edge went permanently quiet for any cause already latched -- including every
// cause raised before a debugger attached, since the handler declines then.
TEST(WaveDebugTest, EnabledExceptionIsDeliveredAfterAnEarlierDisabledOccurrence) {
  WaveDebugFixture fx;
  ASSERT_FALSE(::testing::Test::HasFatalFailure());
  constexpr uint32_t kInvalidCause = 1u << 0;
  constexpr uint32_t kOverflowCause = 1u << 3;
  constexpr uint32_t kModeExcpEnShift = 12;
  // v_mul_f32 v0, v0, v0  (VOP2 op 5, src0 = v0 = 256, vsrc1 = v0, vdst = v0)
  constexpr uint32_t kVMulF32V0 = 0x0A000000u | 256u;
  constexpr float kHuge = 1e38f;

  fx.gpu_mem.write32(kKernelAddr, kVMulF32V0);
  fx.gpu_mem.write32(kKernelAddr + 4, kVMulF32V0);
  fx.gpu_mem.write32(kKernelAddr + 8, kSEndpgm);

  auto *wave = fx.dispatch(kKernelAddr);
  ASSERT_NE(wave, nullptr);
  wave->set_exec(1ULL);
  const uint32_t vbase = wave->vgpr_alloc().base;

  uint32_t handler_calls = 0;
  fx.cu->set_alu_exception_handler([&](amdgpu::Wavefront &) {
    ++handler_calls;
    return false; // observe delivery without stopping the wave
  });

  // First occurrence with overflow *not* enabled. Some other cause is enabled
  // so the instruction takes the exception-aware path and latches TRAPSTS --
  // with no enable at all the generated body takes a SIMD fast path that skips
  // the sticky update, which would not exercise the sticky-bit problem.
  wave->set_mode_raw(kInvalidCause << kModeExcpEnShift);
  fx.cu->write_vgpr(vbase + 0, 0, std::bit_cast<uint32_t>(kHuge));
  fx.cu->step();
  EXPECT_EQ(handler_calls, 0u) << "overflow is not enabled yet";
  ASSERT_NE(wave->trapsts() & kOverflowCause, 0u) << "overflow was not latched";

  // Software now enables overflow and repeats the operation. Hardware traps on
  // this occurrence; a rising-edge test cannot, because the bit is already set.
  wave->set_mode_raw(kOverflowCause << kModeExcpEnShift);
  fx.cu->write_vgpr(vbase + 0, 0, std::bit_cast<uint32_t>(kHuge));
  fx.cu->step();
  EXPECT_EQ(handler_calls, 1u) << "second, enabled occurrence was not delivered";
}

TEST(WaveDebugTest, Gfx10HalfAddReportsBothArithmeticAndOutputModifierCauses) {
  constexpr uint32_t add = 0xd5320004u; // v_add_f16_e64 v4, v1, v2
  constexpr uint32_t operands = 0x02020501u;
  constexpr uint32_t mul2 = 1u << 27, div2 = 3u << 27, clamp = 1u << 15;
  constexpr uint32_t ovfl = 1u << 3, unfl = 1u << 4, inexact = 1u << 5;
  constexpr uint32_t keep_input = 1u << 6, keep_output = 2u << 6;
  constexpr uint32_t saturate = amdgpu::Wavefront::FP16_OVFL_BIT;
  constexpr uint32_t enable_overflow = ovfl << 12;
  constexpr uint32_t sdwa = 0x640804f9u;
  struct Case {
    const char *name;
    uint32_t first, second, mode, lhs, rhs, destination, causes;
  };
  const Case cases[] = {
      {"unmodified overflow", add, operands, 0, 0x7bff, 0x7bff, 0xdead7c00, ovfl | inexact},
      {"overflow before div2", add, operands | div2, enable_overflow, 0x7bff, 0x7bff, 0xdead7c00,
       ovfl | inexact},
      {"saturated intermediate", add, operands | div2, saturate, 0x7bff, 0x7bff, 0xdead77ff,
       ovfl | inexact},
      {"round toward zero", add, operands | div2, 3u << 2, 0x7bff, 0x7bff, 0xdead77ff,
       ovfl | inexact},
      {"modifier overflow", add, operands | mul2, enable_overflow, 0x7bff, 0, 0xdead7c00,
       ovfl | inexact},
      {"clamp after modifier overflow", add | clamp, operands | mul2, saturate, 0x7bff, 0,
       0xdead3c00, ovfl | inexact},
      {"modifier underflow", add, operands | div2, 0, 0x8400, 0, 0xdead8000, unfl | inexact},
      {"arithmetic flush before mul2", add, operands | mul2, 0, 0x0600, 0x8400, 0xdead0000,
       unfl | inexact},
      {"denormal kept and OMOD disabled", add, operands | div2, keep_input | keep_output, 1, 0,
       0xdead0001, 2},
      {"IEEE disables OMOD", add, operands | mul2, amdgpu::Wavefront::IEEE_BIT, 0x7bff, 0,
       0xdead7bff, 0},
      {"exact scaling", add, operands | mul2, enable_overflow, 0x3c00, 0x3c00, 0xdead4400, 0},
      // SDWA div:2 selects WORD_0 inputs and preserves the low destination half.
      {"SDWA rounded overflow", sdwa, 0x0404d501, 0, 0x7bff, 0x7bff, 0x7c00beef, ovfl | inexact},
      {"SDWA negative modifier underflow", sdwa, 0x0404d501, 0, 0x8400, 0, 0x8000beef,
       unfl | inexact},
  };
  for (auto arch : {ROCJITSU_CODE_ARCH_RDNA1, ROCJITSU_CODE_ARCH_RDNA2}) {
    WaveDebugFixture fx(arch);
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
    uint32_t pc = kKernelAddr;
    for (const auto &c : cases) {
      fx.gpu_mem.write32(pc, c.first);
      fx.gpu_mem.write32(pc + 4, c.second);
      fx.gpu_mem.write32(pc + 8, 0xb900f803u); // s_getreg_b32 s0, hwreg(HW_REG_TRAPSTS)
      pc += 12;
    }
    auto *wf = fx.dispatch(kKernelAddr);
    ASSERT_NE(wf, nullptr);
    wf->set_exec(1);
    const uint32_t base = wf->vgpr_alloc().base;
    unsigned traps = 0;
    fx.cu->set_alu_exception_handler([&](amdgpu::Wavefront &) {
      ++traps;
      return false;
    });
    for (const auto &c : cases) {
      SCOPED_TRACE(::testing::Message() << c.name << " arch=" << arch);
      wf->set_mode_raw(c.mode);
      // Keep an older, unrelated cause sticky, and repeat a pre-latched overflow
      // with its trap enabled to exercise the per-instruction delivery path.
      const uint32_t prior = (c.mode & enable_overflow) ? 0x100u | ovfl : 0x100u;
      wf->set_trapsts(prior);
      fx.cu->write_vgpr(base + 1, 0, c.lhs);
      fx.cu->write_vgpr(base + 2, 0, c.rhs);
      fx.cu->write_vgpr(base + 4, 0, 0xdeadbeef);
      fx.cu->write_vgpr(base + 4, 1, 0x12345678);
      const auto previous_traps = traps;
      fx.cu->step();
      EXPECT_EQ(fx.cu->read_vgpr(base + 4, 0), c.destination);
      EXPECT_EQ(fx.cu->read_vgpr(base + 4, 1), 0x12345678u);
      EXPECT_EQ(traps - previous_traps, ((c.mode >> 12) & c.causes & 0x7fu) != 0);
      fx.cu->step();
      EXPECT_EQ(fx.cu->read_sgpr(wf->sgpr_alloc().base), prior | c.causes);
    }
  }
}

TEST(WaveDebugTest, Gfx1250AluExceptionDeliveryUsesTrapCtrlNotModeVgprMsb) {
  WaveDebugFixture fx(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_FALSE(::testing::Test::HasFatalFailure());
  constexpr uint32_t kInvalidCause = 1u << 0;
  constexpr uint32_t kOverflowCause = 1u << 3;
  // Keep SRC2 in a nonzero bank. VOP2 does not use that role, but a regression
  // that reads gfx9's MODE[18:12] exception enables sees a nonzero mask here.
  constexpr uint32_t kSrc2VgprMsb = 1u << 18;
  // v_mul_f32 v0, v0, v0 assembled for gfx1250.
  constexpr uint32_t kGfx1250VMulF32V0 = 0x10000100u;
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr float kHuge = 1e38f;

  fx.gpu_mem.write32(kKernelAddr, kGfx1250VMulF32V0);
  fx.gpu_mem.write32(kKernelAddr + 4, kGfx1250VMulF32V0);
  fx.gpu_mem.write32(kKernelAddr + 8, kGfx1250SEndpgm);

  auto *wave = fx.dispatch(kKernelAddr);
  ASSERT_NE(wave, nullptr);
  wave->set_exec(1ULL);
  wave->set_mode_raw(kSrc2VgprMsb);
  const uint32_t vbase = wave->vgpr_alloc().base;

  uint32_t handler_calls = 0;
  fx.cu->set_alu_exception_handler([&](amdgpu::Wavefront &) {
    ++handler_calls;
    return false;
  });

  // Keep one dedicated enable set so the first instruction takes the
  // exception-aware path and latches overflow without delivering it.
  wave->set_gfx12_trap_ctrl_raw(kInvalidCause);
  fx.cu->write_vgpr(vbase, 0, std::bit_cast<uint32_t>(kHuge));
  fx.cu->step();
  EXPECT_EQ(handler_calls, 0u) << "TRAP_CTRL does not enable overflow yet";
  ASSERT_NE(wave->trapsts() & kOverflowCause, 0u) << "overflow was not latched";
  EXPECT_EQ(wave->mode_raw(), kSrc2VgprMsb) << "exception delivery rewrote MODE.VGPR_MSB";

  wave->set_gfx12_trap_ctrl_raw(kOverflowCause);
  fx.cu->write_vgpr(vbase, 0, std::bit_cast<uint32_t>(kHuge));
  fx.cu->step();
  EXPECT_EQ(handler_calls, 1u) << "enabled TRAP_CTRL overflow was not delivered";
  EXPECT_EQ(wave->mode_raw(), kSrc2VgprMsb) << "TRAP_CTRL was folded into MODE.VGPR_MSB";
}

// debug_active_ is CU-wide, so the memory probe also runs for waves belonging to
// a process nobody is debugging. When the handler declines the stop for such a
// wave the access must still be issued -- discarding it silently lost stores and
// left loads' destination registers stale in an undebugged process.
TEST(WaveDebugTest, DeclinedMemoryViolationStillIssuesTheAccess) {
  WaveDebugFixture fx;
  ASSERT_FALSE(::testing::Test::HasFatalFailure());
  constexpr uint32_t kProcessId = 11;
  constexpr uint64_t kPageSize = amdgpu::GpuMemory::PAGE_SIZE;
  constexpr uint64_t kUnmappedAddr = 0x900000;

  std::vector<uint8_t> kernel_page(kPageSize);
  std::vector<uint8_t> data_page(kPageSize);
  KfdProcess::PageTable page_table;
  util::DistributedSharedMutex page_table_mutex;
  page_table[kKernelAddr >> amdgpu::GpuMemory::PAGE_SHIFT] = {kernel_page.data(),
                                                              amdgpu::Mtype::RW};
  fx.gpu_mem.register_process(kProcessId, &page_table, &page_table_mutex);

  fx.gpu_mem.write32(kKernelAddr, 0xC0020000u, kProcessId);     // s_load_dword s0, s[0:1]
  fx.gpu_mem.write32(kKernelAddr + 4, 0, kProcessId);           // immediate offset 0
  fx.gpu_mem.write32(kKernelAddr + 8, 0xBF800000u, kProcessId); // s_nop 0
  auto *wave = fx.dispatch(kKernelAddr);
  ASSERT_NE(wave, nullptr);
  wave->set_process_id(kProcessId);
  // Another process turned debugging on for this CU; this wave is not debugged.
  fx.cu->set_debug_active(true);

  // The declined access is expected to reach the pipeline, so the cache
  // hierarchy needs its functional writeback path wired up.
  fx.l2.set_backing_memory(&fx.gpu_mem);

  ASSERT_FALSE(fx.gpu_mem.is_mapped(kUnmappedAddr, kProcessId));
  const uint32_t sbase = wave->sgpr_alloc().base;
  fx.cu->write_sgpr(sbase + 0, static_cast<uint32_t>(kUnmappedAddr));
  fx.cu->write_sgpr(sbase + 1, static_cast<uint32_t>(kUnmappedAddr >> 32));
  constexpr uint32_t kExpectedValue = 0xFEEDFACEu;
  std::memcpy(data_page.data(), &kExpectedValue, sizeof(kExpectedValue));

  bool handler_called = false;
  fx.cu->set_memory_violation_handler([&](amdgpu::Wavefront &, uint64_t, bool, uint64_t issue_pc) {
    handler_called = true;
    EXPECT_EQ(issue_pc, kKernelAddr);
    return false; // no debug session for this wave's process
  });

  fx.cu->step();

  EXPECT_TRUE(handler_called);
  EXPECT_FALSE(wave->debug_halted());
  EXPECT_EQ(wave->pc, kKernelAddr + 8);
  EXPECT_EQ(fx.cu->read_sgpr(sbase + 0), static_cast<uint32_t>(kUnmappedAddr));

  // A declined debugger notification must leave the operation retryable. Once
  // the missing translation is published, the retained load completes instead
  // of being discarded or fabricated from sparse backing.
  {
    std::unique_lock lock(page_table_mutex);
    page_table[kUnmappedAddr >> amdgpu::GpuMemory::PAGE_SHIFT] = {data_page.data(),
                                                                  amdgpu::Mtype::RW};
  }
  for (int step = 0; step < 8 && fx.cu->read_sgpr(sbase + 0) != kExpectedValue; ++step)
    fx.cu->step();

  EXPECT_EQ(fx.cu->read_sgpr(sbase + 0), kExpectedValue);

  fx.gpu_mem.unregister_process(kProcessId);
}

// A multi-byte access that starts on a mapped page can still run off the end of
// it. Checking only the first byte let such an access through with no
// EC_QUEUE_WAVE_MEMORY_VIOLATION, so the whole touched range is validated.
TEST(WaveDebugTest, FlatFaultUsesTheOriginalSegmentAtAnApertureBoundary) {
  constexpr uint64_t shared = uint64_t{3} << 32;
  for (bool starts_in_lds : {false, true}) {
    WaveDebugFixture fx(ROCJITSU_CODE_ARCH_RDNA4);
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
    constexpr unsigned process = 9;
    constexpr uint64_t page_size = amdgpu::GpuMemory::PAGE_SIZE;
    std::vector<uint8_t> code(page_size);
    KfdProcess::PageTable pages;
    util::DistributedSharedMutex mutex;
    pages[kKernelAddr >> amdgpu::GpuMemory::PAGE_SHIFT] = {code.data(), amdgpu::Mtype::RW};
    fx.gpu_mem.register_process(process, &pages, &mutex);
    const std::array words{0xec05007cu, 0u, 0x402u}; // flat_load_b32 v0, v[2:3] offset:4
    for (unsigned i = 0; i < words.size(); ++i)
      fx.gpu_mem.write32(kKernelAddr + 4 * i, words[i], process);
    fx.cu->set_apertures(shared, shared + UINT32_MAX, 0, 0);
    auto *wave = fx.dispatch(kKernelAddr);
    ASSERT_NE(wave, nullptr);
    wave->set_process_id(process);
    wave->set_exec(1);
    wave->set_lds_size(512);
    const uint64_t pointer = starts_in_lds ? shared + 0xfffffffc : shared - 4;
    const auto base = wave->vgpr_alloc().base;
    fx.cu->write_vgpr(base + 2, 0, static_cast<uint32_t>(pointer));
    fx.cu->write_vgpr(base + 3, 0, static_cast<uint32_t>(pointer >> 32));
    fx.cu->set_debug_active(true);
    std::optional<uint64_t> fault;
    fx.cu->set_memory_violation_handler(
        [&](amdgpu::Wavefront &stopped, uint64_t addr, bool, uint64_t) {
          fault = addr;
          stopped.debug_trap(0);
          return true;
        });
    fx.cu->step();
    EXPECT_EQ(fault, starts_in_lds ? std::nullopt : std::optional(shared));
    fx.gpu_mem.unregister_process(process);
  }
}

TEST(WaveDebugTest, ScalarLoadStraddlingIntoUnmappedPageReportsMemoryViolation) {
  WaveDebugFixture fx;
  ASSERT_FALSE(::testing::Test::HasFatalFailure());
  constexpr uint32_t kProcessId = 9;
  constexpr uint64_t kPageSize = amdgpu::GpuMemory::PAGE_SIZE;
  constexpr uint64_t kDataAddr = 0x40000;

  std::vector<uint8_t> kernel_page(kPageSize);
  std::vector<uint8_t> data_page(kPageSize);
  KfdProcess::PageTable page_table;
  util::DistributedSharedMutex page_table_mutex;
  page_table[kKernelAddr >> amdgpu::GpuMemory::PAGE_SHIFT] = {kernel_page.data(),
                                                              amdgpu::Mtype::RW};
  // Exactly one data page is mapped; the page above it deliberately is not.
  page_table[kDataAddr >> amdgpu::GpuMemory::PAGE_SHIFT] = {data_page.data(), amdgpu::Mtype::RW};
  fx.gpu_mem.register_process(kProcessId, &page_table, &page_table_mutex);

  fx.gpu_mem.write32(kKernelAddr, 0xC0060000u, kProcessId);     // s_load_dwordx2 s[0:1], s[0:1]
  fx.gpu_mem.write32(kKernelAddr + 4, 0, kProcessId);           // immediate offset 0
  fx.gpu_mem.write32(kKernelAddr + 8, 0xBF800000u, kProcessId); // s_nop 0
  auto *wave = fx.dispatch(kKernelAddr);
  ASSERT_NE(wave, nullptr);
  wave->set_process_id(kProcessId);
  fx.cu->set_debug_active(true);

  // Wired up so that failing to detect the straddle lets the access proceed
  // through the pipeline, and the test then fails on the missing violation
  // rather than on unbacked cache plumbing.
  fx.l2.set_backing_memory(&fx.gpu_mem);

  // The last four bytes of the mapped page: the first dword is mapped, the
  // second runs into the unmapped page above.
  const uint64_t straddle = kDataAddr + kPageSize - 4;
  ASSERT_TRUE(fx.gpu_mem.is_mapped(straddle, kProcessId));
  ASSERT_FALSE(fx.gpu_mem.is_mapped(straddle + 4, kProcessId));
  const uint32_t sbase = wave->sgpr_alloc().base;
  fx.cu->write_sgpr(sbase + 0, static_cast<uint32_t>(straddle));
  fx.cu->write_sgpr(sbase + 1, static_cast<uint32_t>(straddle >> 32));

  uint64_t fault_address = ~uint64_t{0};
  fx.cu->set_memory_violation_handler(
      [&](amdgpu::Wavefront &faulting_wave, uint64_t address, bool is_write, uint64_t) {
        EXPECT_EQ(&faulting_wave, wave);
        EXPECT_FALSE(is_write);
        fault_address = address;
        faulting_wave.debug_trap(0);
        return true;
      });

  fx.cu->step();
  EXPECT_EQ(fault_address, straddle);
  EXPECT_TRUE(wave->debug_halted());

  fx.gpu_mem.unregister_process(kProcessId);
}

TEST(WaveDebugTest, ScalarAtomicX2AtPageEndChecksOnlyItsAccessWidth) {
  WaveDebugFixture fx;
  ASSERT_FALSE(::testing::Test::HasFatalFailure());
  constexpr uint32_t kProcessId = 9;
  constexpr uint64_t kPageSize = amdgpu::GpuMemory::PAGE_SIZE;
  constexpr uint64_t kDataAddr = 0x40000;

  std::vector<uint8_t> kernel_page(kPageSize);
  std::vector<uint8_t> data_page(kPageSize);
  KfdProcess::PageTable page_table;
  util::DistributedSharedMutex page_table_mutex;
  page_table[kKernelAddr >> amdgpu::GpuMemory::PAGE_SHIFT] = {kernel_page.data(),
                                                              amdgpu::Mtype::RW};
  // Exactly one data page is mapped; the page above it deliberately is not.
  page_table[kDataAddr >> amdgpu::GpuMemory::PAGE_SHIFT] = {data_page.data(), amdgpu::Mtype::RW};
  fx.gpu_mem.register_process(kProcessId, &page_table, &page_table_mutex);

  const auto atomic =
      cdna4::build_smem(cdna4::kSAtomicDecX2Smem, {.sbase = 0, .sdata = 2, .glc = 1, .imm = 1});
  fx.gpu_mem.write32(kKernelAddr, atomic[0], kProcessId);
  fx.gpu_mem.write32(kKernelAddr + 4, atomic[1], kProcessId);
  fx.gpu_mem.write32(kKernelAddr + 8, 0xBF800000u, kProcessId); // s_nop 0
  auto *wave = fx.dispatch(kKernelAddr);
  ASSERT_NE(wave, nullptr);
  wave->set_process_id(kProcessId);
  fx.cu->set_debug_active(true);
  fx.l2.set_backing_memory(&fx.gpu_mem);

  // The last eight bytes of the mapped page. The atomic is one 8-byte element
  // returned as two dwords; checking elem_size * num_dwords bytes would run into
  // the unmapped page above.
  const uint64_t last_qword = kDataAddr + kPageSize - 8;
  ASSERT_TRUE(fx.gpu_mem.is_mapped(last_qword, kProcessId));
  ASSERT_FALSE(fx.gpu_mem.is_mapped(last_qword + 8, kProcessId));
  const uint64_t initial = 5;
  std::memcpy(data_page.data() + kPageSize - 8, &initial, sizeof(initial));
  const uint32_t sbase = wave->sgpr_alloc().base;
  fx.cu->write_sgpr(sbase + 0, static_cast<uint32_t>(last_qword));
  fx.cu->write_sgpr(sbase + 1, static_cast<uint32_t>(last_qword >> 32));
  fx.cu->write_sgpr(sbase + 2, 9);
  fx.cu->write_sgpr(sbase + 3, 0);

  unsigned violations = 0;
  fx.cu->set_memory_violation_handler([&](amdgpu::Wavefront &, uint64_t, bool, uint64_t) {
    ++violations;
    return true;
  });

  fx.cu->step();
  EXPECT_EQ(violations, 0u);
  EXPECT_FALSE(wave->debug_halted());
  EXPECT_FALSE(wave->is_halted());
  EXPECT_EQ(wave->pc, kKernelAddr + 8);

  // The atomic completes: memory decrements and GLC returns the old qword.
  for (int step = 0; step < 8 && fx.cu->read_sgpr(sbase + 2) != initial; ++step)
    fx.cu->step();
  uint64_t updated = 0;
  std::memcpy(&updated, data_page.data() + kPageSize - 8, sizeof(updated));
  EXPECT_EQ(updated, initial - 1);
  EXPECT_EQ(fx.cu->read_sgpr(sbase + 2), static_cast<uint32_t>(initial));
  EXPECT_EQ(fx.cu->read_sgpr(sbase + 3), 0u);

  fx.gpu_mem.unregister_process(kProcessId);
}

// Linked gfx950 code forms helper addresses as GETPC plus a signed rel32
// relocation. The 64-bit sum can have a 0x1ffff high dword even though it
// denotes an address relative to the loaded code object. SWAPPC must relocate
// that value to the dispatch load bias and preserve the architectural return
// address for SETPC.
TEST(WaveDebugTest, SwappcResolvesLinkedRel32AndSetpcReturns) {
  WaveDebugFixture fx;
  ASSERT_FALSE(::testing::Test::HasFatalFailure());
  constexpr uint64_t kLoadBias = 0x7FFF00000000ULL;
  constexpr uint64_t kCallerPc = kLoadBias + 0xA3A8;
  constexpr uint64_t kCalleePc = kLoadBias + 0x8D60;

  fx.gpu_mem.write32(kCallerPc, 0xBE9E1E00u); // s_swappc_b64 s[30:31], s[0:1]
  fx.gpu_mem.write32(kCallerPc + 4, kSEndpgm);
  fx.gpu_mem.write32(kCalleePc, 0xBE801D1Eu); // s_setpc_b64 s[30:31]

  auto *wf = fx.dispatch(kCallerPc);
  ASSERT_NE(wf, nullptr);
  wf->set_code_load_bias(kLoadBias);
  // Linked gfx950 calls place GETPC, ADD_LO, ADDC_HI, then SWAPPC. At SWAPPC
  // the relocation base is therefore 16 bytes behind the current PC.
  const int32_t rel32 = static_cast<int32_t>(kCalleePc - (kCallerPc - 16));
  const uint64_t encoded_target = 0x0001FFFF00000000ULL | static_cast<uint32_t>(rel32);
  fx.cu->write_sgpr(wf->sgpr_alloc().base + 0, static_cast<uint32_t>(encoded_target));
  fx.cu->write_sgpr(wf->sgpr_alloc().base + 1, static_cast<uint32_t>(encoded_target >> 32));

  fx.cu->step();
  EXPECT_EQ(wf->pc, kCalleePc);
  EXPECT_EQ(wf->debug_read_sgpr(30), static_cast<uint32_t>(kCallerPc + 4));
  EXPECT_EQ(wf->debug_read_sgpr(31), static_cast<uint32_t>((kCallerPc + 4) >> 32));

  fx.cu->step();
  EXPECT_EQ(wf->pc, kCallerPc + 4);
}

// A single-stepped wave (rocm-dbgapi MODE.debug_en=1) executes exactly one
// instruction and is then handed to the single-step completion handler, which
// re-stops it. The engine must not run past that one instruction.
TEST(WaveDebugTest, SingleStepExecutesOneInstructionThenReports) {
  WaveDebugFixture fx;
  ASSERT_FALSE(::testing::Test::HasFatalFailure());
  fx.gpu_mem.write32(kKernelAddr, 0xBF800000u);     // s_nop 0
  fx.gpu_mem.write32(kKernelAddr + 4, 0xBF800000u); // s_nop 0
  fx.gpu_mem.write32(kKernelAddr + 8, kSEndpgm);

  uint32_t step_count = 0;
  fx.cu->set_single_step_handler([&](amdgpu::Wavefront &w) {
    ++step_count;
    w.set_debug_halted(true); // re-stop after the step, as the driver does
    return true;
  });

  auto *wf = fx.dispatch(kKernelAddr);
  ASSERT_NE(wf, nullptr);
  wf->set_debug_single_step(true);

  // One engine step runs exactly one instruction, then the handler re-stops it.
  fx.cu->step();
  EXPECT_EQ(step_count, 1u);
  EXPECT_EQ(wf->pc, kKernelAddr + 4);
  EXPECT_TRUE(wf->debug_halted());

  // While halted, stepping the CU makes no further progress.
  fx.cu->step();
  EXPECT_EQ(step_count, 1u);
  EXPECT_EQ(wf->pc, kKernelAddr + 4);

  // Resuming single-step runs the next instruction and re-stops again.
  wf->set_debug_halted(false);
  wf->set_debug_single_step(true);
  fx.cu->step();
  EXPECT_EQ(step_count, 2u);
  EXPECT_EQ(wf->pc, kKernelAddr + 8);
  EXPECT_TRUE(wf->debug_halted());
}

// Without a debugger the s_trap is a no-op: the wave advances past it and runs
// to completion. This keeps kernels that embed traps in unreached paths from
// aborting the emulator when no debugger is attached.
TEST(WaveDebugTest, STrapWithoutDebuggerIsNoOp) {
  WaveDebugFixture fx;
  ASSERT_FALSE(::testing::Test::HasFatalFailure());
  fx.gpu_mem.write32(kKernelAddr, kSTrapBreakpoint);
  fx.gpu_mem.write32(kKernelAddr + 4, kSEndpgm);

  auto *wf = fx.dispatch(kKernelAddr);
  ASSERT_NE(wf, nullptr);

  fx.cu->step(); // s_trap -> no handler -> no-op, PC advances
  EXPECT_FALSE(wf->debug_halted());
  EXPECT_EQ(wf->pc, kKernelAddr + 4);

  fx.cu->step(); // s_endpgm -> wave halts
  EXPECT_TRUE(wf->is_halted());
}

TEST(WaveDebugTest, IllegalInstructionStopsWaveUnderDebugger) {
  // RDNA2 S_CODE_END is a valid encoding that raises the same illegal-instruction interrupt.
  for (const auto &[arch, word] : {std::pair{ROCJITSU_CODE_ARCH_CDNA4, 0xffffffffu},
                                   std::pair{ROCJITSU_CODE_ARCH_RDNA2, 0xbf9f0000u}}) {
    SCOPED_TRACE(arch);
    WaveDebugFixture fx(arch);
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
    fx.gpu_mem.write32(kKernelAddr, word);
    uint32_t illegal_count = 0;
    fx.cu->set_illegal_inst_handler([&](amdgpu::Wavefront &wave) {
      ++illegal_count;
      wave.set_fatal_exception_pending(true);
      wave.set_debug_halted(true);
      return true;
    });
    auto *wave = fx.dispatch(kKernelAddr);
    ASSERT_NE(wave, nullptr);
    fx.cu->step();
    EXPECT_EQ(illegal_count, 1u);
    EXPECT_TRUE(wave->debug_halted());
    EXPECT_TRUE(wave->fatal_exception_pending());
    EXPECT_EQ(wave->pc, kKernelAddr);
    wave->reset();
    EXPECT_FALSE(wave->fatal_exception_pending());
  }
}

TEST(WaveDebugTest, IllegalInstructionWithoutHandlerHalts) {
  WaveDebugFixture fx;
  ASSERT_FALSE(::testing::Test::HasFatalFailure());
  fx.gpu_mem.write32(kKernelAddr, 0xFFFFFFFFu);
  auto *wave = fx.dispatch(kKernelAddr);
  ASSERT_NE(wave, nullptr);
  fx.cu->step();
  EXPECT_TRUE(wave->is_halted());
}

TEST(WaveDebugTest, Pm4DecodeFailureDoesNotNotifyAqlOwner) {
  WaveDebugFixture fx;
  ASSERT_FALSE(::testing::Test::HasFatalFailure());
  fx.gpu_mem.write32(kKernelAddr, 0xFFFFFFFFu);
  uint32_t notifications = 0;
  fx.cu->set_illegal_inst_handler([&](amdgpu::Wavefront &) {
    ++notifications;
    return true;
  });
  auto failure = std::make_shared<amdgpu::Pm4FailureState>();
  auto *wave = fx.dispatch(kKernelAddr);
  ASSERT_NE(wave, nullptr);
  wave->set_pm4_failure(failure);
  fx.cu->step();
  EXPECT_TRUE(failure->failed.load());
  EXPECT_TRUE(wave->is_halted());
  EXPECT_EQ(notifications, 0u);
}

TEST(WaveDebugTest, STrapWithoutConfiguredHandlerAdvancesPcWithoutChangingWaveState) {
  WaveDebugFixture fx;
  ASSERT_FALSE(::testing::Test::HasFatalFailure());
  fx.gpu_mem.write32(kKernelAddr, kSTrapSeven);

  auto *wf = fx.dispatch(kKernelAddr);
  ASSERT_NE(wf, nullptr);
  wf->set_status_raw(0x12345678u);
  wf->set_trapsts(0x87654321u);

  fx.cu->step();

  EXPECT_EQ(wf->pc, kKernelAddr + 4);
  EXPECT_EQ(wf->status_raw(), 0x12345678u);
  EXPECT_EQ(wf->trapsts(), 0x87654321u);
  EXPECT_EQ(wf->trap_id(), 0u);
  EXPECT_FALSE(wf->debug_halted());
}

TEST(WaveDebugTest, Gfx1250TrapWithoutConfiguredHandlerAdvancesPc) {
  WaveDebugFixture fx(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_FALSE(::testing::Test::HasFatalFailure());
  fx.gpu_mem.write32(kKernelAddr, kGfx12STrap);

  auto *wf = fx.dispatch(kKernelAddr);
  ASSERT_NE(wf, nullptr);
  fx.cu->step();

  EXPECT_EQ(wf->pc, kKernelAddr + 4);
  EXPECT_FALSE(wf->debug_halted());
}

TEST(WaveDebugTest, Rdna4TrapEntryAndReturnUseTheGfx12WaveStateLayout) {
  constexpr uint64_t kHighKernelAddr = 0x0000123400001000ULL;
  constexpr uint32_t kApplicationStatus = 1u | (0xAu << 1);
  constexpr uint32_t kSchedMode = 2u;
  WaveDebugFixture fx(ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_FALSE(::testing::Test::HasFatalFailure());
  fx.gpu_mem.write32(kHighKernelAddr, kGfx12STrap);
  fx.gpu_mem.write32(kHighKernelAddr + 4, kSEndpgm);

  // GFX12.0 uses the common split STATE_PRIV/TRAP_CTRL ABI with a 48-bit PC,
  // preserves SCHED_MODE in TTMP1[27:26], and returns through s_rfe_b64.
  const uint32_t handler[] = {
      0x806C846Cu, 0x826D806Du, 0xB9800384u, 0x00000001u, 0xBE804A6Cu,
  };
  for (uint32_t i = 0; i < std::size(handler); ++i)
    fx.gpu_mem.write32(kTrapHandlerAddr + i * 4, handler[i]);

  fx.cu->set_trap_handler_resolver([](const amdgpu::Wavefront &) {
    return amdgpu::ComputeUnitCore::TrapHandlerConfig{kTrapHandlerAddr, 0, true};
  });

  auto *wf = fx.dispatch(kHighKernelAddr);
  ASSERT_NE(wf, nullptr);
  wf->set_status_raw(kApplicationStatus);
  wf->set_sleep_wakeup(true);
  wf->set_workgroup_barrier_complete(true);
  wf->set_wave_sched_mode_raw(kSchedMode);
  for (uint32_t i = 6; i <= 9; ++i)
    wf->set_ttmp(i, 0xB6000000u + i);

  fx.cu->step();
  ASSERT_TRUE(wf->in_trap_handler());
  EXPECT_TRUE(wf->uses_separate_trap_ctrl());
  EXPECT_EQ(wf->pc, kTrapHandlerAddr);
  EXPECT_EQ(wf->ttmp(0), static_cast<uint32_t>(kHighKernelAddr));
  EXPECT_EQ(wf->ttmp(1), 0x38001234u);
  for (uint32_t i = 6; i <= 9; ++i)
    EXPECT_EQ(wf->ttmp(i), 0xB6000000u + i) << "TTMP" << i << " was not preserved";
  EXPECT_EQ(wf->ttmp(11), 1u << 23);
  EXPECT_EQ(wf->ttmp(12), (1u << 1) | (1u << 2) | (1u << 9) | (0xAu << 10));
  EXPECT_NE(wf->status_raw() & (1u << 5), 0u) << "trap entry did not enter privileged mode";

  for (int i = 0; i < 4; ++i)
    fx.cu->step();

  EXPECT_FALSE(wf->in_trap_handler());
  EXPECT_TRUE(wf->sleep_wakeup());
  EXPECT_TRUE(wf->workgroup_barrier_complete());
  EXPECT_TRUE(wf->debug_halted());
  EXPECT_EQ(wf->pc, kHighKernelAddr + 4);
  EXPECT_EQ(wf->status_raw(), kApplicationStatus | amdgpu::Wavefront::kStatusHaltMask);
}

// gfx12.5 changed both halves of the first-level trap ABI: PC is 57 bits with
// TrapID in TTMP1[31:28], and DebugEnabled/STATE_PRIV moved to TTMP11/12.
// Exercise a real gfx1250 s_rfe_i64 sequence as well, so a layout that merely
// looks right at entry cannot pass while truncating the return address.
TEST(WaveDebugTest, Gfx1250TrapEntryAndReturnFollowTheGfx12Abi) {
  constexpr uint64_t kHighKernelAddr = 0x0100000000001000ULL;
  constexpr uint32_t kApplicationStatus = 1u | (0xAu << 1);
  constexpr uint64_t kScratchBase = 0x0000001240008000ULL;
  WaveDebugFixture fx(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_FALSE(::testing::Test::HasFatalFailure());
  fx.gpu_mem.write32(kHighKernelAddr, kGfx12STrap);
  fx.gpu_mem.write32(kHighKernelAddr + 4, kSEndpgm);

  // Assembled for gfx1250:
  //   s_add_co_u32 ttmp0, ttmp0, 4
  //   s_add_co_ci_u32 ttmp1, ttmp1, 0
  //   s_setreg_imm32_b32 hwreg(WAVE_STATE_PRIV, 14, 1), 1
  //   s_rfe_i64 ttmp[0:1]
  const uint32_t handler[] = {
      0x806C846Cu, 0x826D806Du, 0xB9800384u, 0x00000001u, 0xBE804A6Cu,
  };
  for (uint32_t i = 0; i < std::size(handler); ++i)
    fx.gpu_mem.write32(kTrapHandlerAddr + i * 4, handler[i]);

  fx.cu->set_trap_handler_resolver([](const amdgpu::Wavefront &) {
    return amdgpu::ComputeUnitCore::TrapHandlerConfig{kTrapHandlerAddr, 0x123456789ABCuLL, true};
  });

  auto *wf = fx.dispatch(kHighKernelAddr);
  ASSERT_NE(wf, nullptr);
  wf->set_status_raw(kApplicationStatus);
  wf->set_sleep_wakeup(true);
  wf->set_workgroup_barrier_complete(true);
  wf->set_scratch_base(kScratchBase);
  for (uint32_t i = 6; i <= 9; ++i)
    wf->set_ttmp(i, 0xA5000000u + i);

  fx.cu->step();
  ASSERT_TRUE(wf->in_trap_handler());
  EXPECT_EQ(wf->pc, kTrapHandlerAddr);
  EXPECT_EQ(wf->ttmp(0), static_cast<uint32_t>(kHighKernelAddr));
  EXPECT_EQ(wf->ttmp(1), 0x31000000u);
  for (uint32_t i = 6; i <= 9; ++i)
    EXPECT_EQ(wf->ttmp(i), 0xA5000000u + i) << "TTMP" << i << " was not preserved";
  EXPECT_EQ(wf->ttmp(11), 1u << 23);
  EXPECT_EQ(wf->ttmp(12), (1u << 1) | (1u << 2) | (1u << 9) | (0xAu << 10) | (1u << 18));
  EXPECT_EQ(wf->ttmp(14), 0x56789ABCu);
  EXPECT_EQ(wf->ttmp(15), 0x00001234u);
  EXPECT_NE(wf->status_raw() & (1u << 5), 0u) << "trap entry did not enter privileged mode";

  for (int i = 0; i < 4; ++i)
    fx.cu->step();

  EXPECT_FALSE(wf->in_trap_handler());
  EXPECT_TRUE(wf->sleep_wakeup());
  EXPECT_TRUE(wf->workgroup_barrier_complete());
  EXPECT_TRUE(wf->debug_halted());
  EXPECT_EQ(wf->pc, kHighKernelAddr + 4);
  EXPECT_EQ(wf->status_raw(), kApplicationStatus | amdgpu::Wavefront::kStatusHaltMask);
  EXPECT_FALSE(wf->self_halted());
}

// The trap temporary registers and trap status register round-trip, and reset()
// (slot reuse) clears all debugger state.
TEST(WaveDebugTest, TrapRegistersRoundTripAndReset) {
  WaveDebugFixture fx;
  ASSERT_FALSE(::testing::Test::HasFatalFailure());
  auto *wf = fx.dispatch(kKernelAddr);
  ASSERT_NE(wf, nullptr);

  for (uint32_t i = 0; i < 16; ++i)
    wf->set_ttmp(i, 0x1000 + i);
  for (uint32_t i = 0; i < 16; ++i)
    EXPECT_EQ(wf->ttmp(i), 0x1000 + i);
  EXPECT_EQ(wf->ttmp(16), 0u); // out of range reads zero

  wf->set_trapsts(0xDEAD);
  EXPECT_EQ(wf->trapsts(), 0xDEADu);
  wf->set_queue_id(42);
  EXPECT_EQ(wf->queue_id(), 42u);

  wf->debug_trap(7);
  EXPECT_TRUE(wf->debug_halted());
  EXPECT_EQ(wf->trap_id(), 7u);

  wf->reset();
  EXPECT_FALSE(wf->debug_halted());
  EXPECT_FALSE(wf->debug_single_step());
  EXPECT_EQ(wf->trap_id(), 0u);
  EXPECT_EQ(wf->trapsts(), 0u);
  EXPECT_EQ(wf->queue_id(), 0u);
  for (uint32_t i = 0; i < 16; ++i)
    EXPECT_EQ(wf->ttmp(i), 0u);
}

// -----------------------------------------------------------------------------
// CWSR serialization round-trip: parse the serialized area back with the exact
// formulas rocm-dbgapi uses (projects/rocdbgapi/src/architecture.cpp gfx9/mi
// cwsr_record_t) and confirm the header invariants and every register offset.
// -----------------------------------------------------------------------------

struct ParsedWave {
  uint64_t sgprs_addr = 0;
  uint64_t pc = 0, exec = 0, vcc = 0, wave_id = 0;
  uint32_t status = 0, trapsts = 0, mode = 0, m0 = 0, ttmp6 = 0, ttmp11 = 0;
  bool first = false, last = false;
  uint32_t group[3] = {};
  std::vector<uint32_t> sgprs;
  std::vector<uint32_t> vgprs;
  std::vector<uint8_t> lds;
};

// Walk the CWSR area exactly as rocm-dbgapi's control_stack_iterate +
// register_address do, asserting the contiguity invariants dbgapi enforces.
std::vector<ParsedWave> parse_cwsr(const std::map<uint64_t, uint32_t> &mem, uint64_t base,
                                   uint32_t lds_granule) {
  auto rd = [&](uint64_t va) -> uint32_t {
    auto it = mem.find(va);
    return it == mem.end() ? 0u : it->second;
  };
  auto rd64 = [&](uint64_t va) -> uint64_t {
    return rd(va) | (static_cast<uint64_t>(rd(va + 4)) << 32);
  };

  const uint32_t cs_off = rd(base + 0);
  const uint32_t cs_size = rd(base + 4);
  const uint32_t ws_off = rd(base + 8);
  const uint32_t ws_size = rd(base + 12);

  // dbgapi: control_stack_end must equal wave_area_begin.
  EXPECT_EQ(base + cs_off + cs_size, base + ws_off - ws_size);

  std::vector<ParsedWave> out;
  uint64_t last_wave_area = base + ws_off;
  uint32_t state = 0;
  const uint32_t words = cs_size / 4;
  for (uint32_t i = 2; i < words; ++i) {
    uint32_t rl = rd(base + cs_off + i * 4);
    if (rl & (1u << 30))
      continue; // event
    if (rl & (1u << 31)) {
      state = rl;
      continue;
    }
    const uint32_t vgprs_field = state & 0x3F;
    const uint32_t sgprs_field = (state >> 6) & 0x7;
    const uint32_t accum = (state >> 24) & 0x3F;
    const uint32_t vgpr_count = (accum + 1) * 4;
    const uint32_t acc = (vgprs_field + 1) * 8 - vgpr_count;
    const uint32_t sgpr_count = (sgprs_field + 1) * 16 - 16;
    const uint32_t lds_size = ((state >> 9) & 0xFF) * lds_granule;

    const uint64_t save = last_wave_area - 64;
    const uint64_t register_area_end = save - ((rl & (1u << 17)) ? lds_size : 0);
    const uint64_t hwregs = register_area_end - 32 * 4;
    const uint64_t ttmps = register_area_end - 16 * 4;
    const uint64_t sgprs_addr = hwregs - sgpr_count * 4;
    const uint64_t accv = sgprs_addr - acc * 256;
    const uint64_t vgprs_addr = accv - static_cast<uint64_t>(vgpr_count) * 256;

    ParsedWave pw;
    pw.sgprs_addr = sgprs_addr;
    pw.last = rl & (1u << 16);
    pw.first = rl & (1u << 17);
    pw.m0 = rd(hwregs + 0 * 4);
    pw.pc = rd64(hwregs + 1 * 4);
    pw.exec = rd64(hwregs + 3 * 4);
    pw.status = rd(hwregs + 5 * 4);
    pw.trapsts = rd(hwregs + 6 * 4);
    pw.mode = rd(hwregs + 9 * 4);
    pw.wave_id = rd64(ttmps + 4 * 4);
    pw.ttmp6 = rd(ttmps + 6 * 4);
    pw.group[0] = rd(ttmps + 8 * 4);
    pw.group[1] = rd(ttmps + 9 * 4);
    pw.group[2] = rd(ttmps + 10 * 4);
    pw.ttmp11 = rd(ttmps + 11 * 4);
    const uint32_t vcc_lo_slot = std::min<uint32_t>(108, sgpr_count) - 2;
    pw.vcc = rd64(sgprs_addr + vcc_lo_slot * 4);
    pw.sgprs.resize(sgpr_count);
    for (uint32_t s = 0; s < sgpr_count; ++s)
      pw.sgprs[s] = rd(sgprs_addr + s * 4);
    pw.vgprs.resize(static_cast<size_t>(vgpr_count) * 64);
    for (uint32_t r = 0; r < vgpr_count; ++r)
      for (uint32_t l = 0; l < 64; ++l)
        pw.vgprs[r * 64 + l] = rd(vgprs_addr + r * 256 + l * 4);
    if (pw.first) {
      pw.lds.resize(lds_size);
      for (uint32_t byte = 0; byte < lds_size; ++byte)
        pw.lds[byte] =
            static_cast<uint8_t>(rd(register_area_end + (byte & ~3u)) >> ((byte & 3u) * 8));
    }
    out.push_back(std::move(pw));
    last_wave_area = vgprs_addr;
  }
  // dbgapi: the walk must bottom out exactly at wave_area_begin.
  EXPECT_EQ(last_wave_area, base + ws_off - ws_size);
  return out;
}

kmd::CwsrWaveState make_wave(uint64_t id, uint64_t pc) {
  kmd::CwsrWaveState w;
  w.pc = pc;
  w.exec = 0xF0F0F0F0ULL;
  w.vcc = 0xABCD1234ULL;
  w.status = (1u << 13); // HALT
  w.trapsts = 0;
  w.mode = 0;
  w.m0 = 0x55;
  w.wave_id = id;
  w.group_ids = {1, 2, 3};
  w.wave_in_group = 0;
  w.queue_packet_id = 7;
  w.trap_id = 1;
  w.wave_stopped = true;
  w.num_sgprs = 16;
  w.num_vgprs = 4;
  w.sgprs.resize(w.num_sgprs);
  for (uint32_t s = 0; s < w.num_sgprs; ++s)
    w.sgprs[s] = 0x1000 + s;
  w.vgprs.resize(static_cast<size_t>(w.num_vgprs) * 64);
  for (uint32_t r = 0; r < w.num_vgprs; ++r)
    for (uint32_t l = 0; l < 64; ++l)
      w.vgprs[r * 64 + l] = (r << 16) | l;
  return w;
}

TEST(WaveDebugTest, CwsrRejectsUnmodelledTargetsWithoutAccessingMemory) {
  constexpr uint64_t kCtxBase = 0x400000000ULL;
  constexpr uint32_t kAreaSize = 0x40000;
  for (const auto arch :
       {ROCJITSU_CODE_ARCH_CDNA1, ROCJITSU_CODE_ARCH_CDNA2, ROCJITSU_CODE_ARCH_RDNA1,
        ROCJITSU_CODE_ARCH_RDNA2, ROCJITSU_CODE_ARCH_RDNA3, ROCJITSU_CODE_ARCH_RDNA3_5,
        ROCJITSU_CODE_ARCH_RDNA4}) {
    SCOPED_TRACE(static_cast<int>(arch));
    std::vector<kmd::CwsrWaveState> waves = {make_wave(0xAA01, 0x4000)};
    waves[0].is_first_in_group = waves[0].is_last_in_group = true;
    waves[0].lds.resize(512, 0xA5);
    bool accessed_memory = false;
    EXPECT_FALSE(
        kmd::serialize_queue_cwsr(
            kCtxBase, kAreaSize, waves, [&](uint64_t, uint32_t) { accessed_memory = true; }, arch)
            .ok);
    EXPECT_FALSE(kmd::serialize_queue_cwsr_bulk(
                     kCtxBase, kAreaSize, waves,
                     [&](uint64_t, std::span<const uint8_t>) { accessed_memory = true; }, arch)
                     .ok);
    EXPECT_FALSE(kmd::deserialize_queue_cwsr(
        kCtxBase, kAreaSize, waves,
        [&](uint64_t) {
          accessed_memory = true;
          return 0u;
        },
        arch));
    EXPECT_FALSE(kmd::deserialize_queue_cwsr_bulk(
        kCtxBase, kAreaSize, waves, [&](uint64_t, std::span<uint8_t>) { accessed_memory = true; },
        arch));
    EXPECT_FALSE(accessed_memory);
  }
}

TEST(WaveDebugTest, Gfx1250CwsrMatchesDbgapiWave32LayoutAndRoundTrips) {
  constexpr uint64_t kCtxBase = 0x400000000ULL;
  constexpr uint32_t kAreaSize = 0x40000;
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_CDNA5;

  auto wave = make_wave(0xAABBCCDD11223344ULL, 0x123456789ABCuLL);
  wave.exec = 0xFFFFFFFFu;
  wave.flat_scratch = 0x0000001240008000ULL;
  wave.state_priv = (1u << 9) | (1u << 14) | (1u << 18);
  wave.status = (1u << 6) | (1u << 16);
  wave.excp_flag_priv = (1u << 6) | (1u << 12);
  wave.excp_flag_user = 1u << 3;
  wave.trap_ctrl = (1u << 3) | (1u << 9);
  wave.xnack_state_priv = 0x00057F7Fu;
  wave.xnack_mask = 0x76543210u;
  wave.scratch_scoreboard_id = 1023;
  wave.shader_engine_id = 1;
  wave.group_ids = {9, 10, 11};
  wave.wave_in_group = 3;
  wave.queue_packet_id = 0x12345;
  wave.trap_id = 7;
  wave.saved_status_halt = true;
  wave.spi_ttmps_setup = true;
  wave.num_sgprs = 112;
  wave.sgprs.resize(wave.num_sgprs);
  for (uint32_t s = 0; s < wave.num_sgprs; ++s)
    wave.sgprs[s] = 0x51000000u + s;
  wave.lds = {0x11, 0x22, 0x33, 0x44};

  std::map<uint64_t, uint32_t> mem;
  auto write32 = [&](uint64_t address, uint32_t value) { mem[address] = value; };
  auto read32 = [&](uint64_t address) -> uint32_t {
    auto it = mem.find(address);
    return it == mem.end() ? 0u : it->second;
  };

  const auto layout = kmd::serialize_queue_cwsr(kCtxBase, kAreaSize, {wave}, write32, kArch);
  ASSERT_TRUE(layout.ok);
  EXPECT_EQ(layout.control_stack_size, 5u * sizeof(uint32_t));
  const uint64_t control = kCtxBase + layout.control_stack_offset;
  EXPECT_EQ(read32(control + 8), (1u << 31) | (1u << 24) | (1u << 10));
  EXPECT_EQ(read32(control + 12), 0u);
  EXPECT_EQ(read32(control + 16), 1023u | (1u << 11) | (1u << 12) | (1u << 13) | (1u << 24));

  // gfx12.5 walks directly down from wave_state_offset: one 1-KiB LDS block,
  // 128 HWREG dwords, 128 SGPR dwords and sixteen 32-lane VGPRs.
  const uint64_t wave_end = kCtxBase + layout.wave_state_offset;
  const uint64_t hwregs = wave_end - 1024 - 128 * sizeof(uint32_t);
  const uint64_t sgprs = hwregs - 128 * sizeof(uint32_t);
  const uint64_t vgprs = sgprs - 16 * 32 * sizeof(uint32_t);
  const uint64_t ttmps = sgprs + 112 * sizeof(uint32_t);
  EXPECT_EQ(vgprs, kCtxBase + layout.wave_state_offset - layout.wave_state_size);
  EXPECT_EQ(read32(hwregs + 5 * 4), wave.state_priv);
  EXPECT_EQ(read32(hwregs + 6 * 4), wave.excp_flag_priv);
  EXPECT_EQ(read32(hwregs + 7 * 4), wave.xnack_mask);
  EXPECT_EQ(read32(hwregs + 9 * 4), static_cast<uint32_t>(wave.flat_scratch));
  EXPECT_EQ(read32(hwregs + 10 * 4), static_cast<uint32_t>(wave.flat_scratch >> 32));
  EXPECT_EQ(read32(hwregs + 11 * 4), wave.excp_flag_user);
  EXPECT_EQ(read32(hwregs + 12 * 4), wave.trap_ctrl);
  EXPECT_EQ(read32(hwregs + 13 * 4), wave.status);
  EXPECT_EQ(read32(ttmps + 4 * 4), static_cast<uint32_t>(wave.wave_id));
  EXPECT_EQ(read32(ttmps + 5 * 4), static_cast<uint32_t>(wave.wave_id >> 32));
  EXPECT_EQ(read32(ttmps + 6 * 4), (1u << 30) | (1u << 29));
  EXPECT_EQ(read32(ttmps + 7 * 4), 10u | (11u << 16));
  EXPECT_EQ(read32(ttmps + 8 * 4), wave.queue_packet_id | (3u << 25) | (1u << 30) | (1u << 31));
  EXPECT_EQ(read32(ttmps + 9 * 4), 9u);
  EXPECT_EQ(read32(ttmps + 11 * 4), wave.xnack_state_priv | (7u << 28));
  EXPECT_EQ(read32(vgprs + 3 * 32 * 4 + 31 * 4), wave.vgprs[3 * 64 + 31]);

  std::vector<kmd::CwsrWaveState> output(1);
  output[0].num_sgprs = wave.num_sgprs;
  output[0].num_vgprs = wave.num_vgprs;
  output[0].lds.resize(wave.lds.size());
  ASSERT_TRUE(kmd::deserialize_queue_cwsr(kCtxBase, kAreaSize, output, read32, kArch));
  const auto &restored = output[0];
  EXPECT_EQ(restored.pc, wave.pc);
  EXPECT_EQ(restored.exec, wave.exec);
  EXPECT_EQ(restored.vcc, wave.vcc);
  EXPECT_EQ(restored.flat_scratch, wave.flat_scratch);
  EXPECT_EQ(restored.state_priv, wave.state_priv);
  EXPECT_EQ(restored.status, wave.status);
  EXPECT_EQ(restored.excp_flag_priv, wave.excp_flag_priv);
  EXPECT_EQ(restored.excp_flag_user, wave.excp_flag_user);
  EXPECT_EQ(restored.trap_ctrl, wave.trap_ctrl);
  EXPECT_EQ(restored.xnack_state_priv, wave.xnack_state_priv);
  EXPECT_EQ(restored.xnack_mask, wave.xnack_mask);
  EXPECT_EQ(restored.wave_id, wave.wave_id);
  EXPECT_EQ(restored.group_ids, wave.group_ids);
  EXPECT_EQ(restored.wave_in_group, wave.wave_in_group);
  EXPECT_EQ(restored.queue_packet_id, wave.queue_packet_id);
  EXPECT_EQ(restored.trap_id, wave.trap_id);
  EXPECT_EQ(restored.lds, wave.lds);
  for (uint32_t r = 0; r < wave.num_vgprs; ++r)
    for (uint32_t lane = 0; lane < 32; ++lane)
      EXPECT_EQ(restored.vgprs[r * 64 + lane], wave.vgprs[r * 64 + lane]);
}

TEST(WaveDebugTest, Gfx1250CwsrAcceptsTheFullAddressableVgprFile) {
  constexpr uint64_t kCtxBase = 0x400000000ULL;
  constexpr uint32_t kAreaSize = 0x40000;
  constexpr uint32_t kMaxVgprs = 1024;
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_CDNA5;

  auto wave = make_wave(0xAA55, 0x4000);
  wave.num_vgprs = kMaxVgprs;
  wave.vgprs.resize(static_cast<size_t>(wave.num_vgprs) * 64);
  wave.vgprs[(kMaxVgprs - 1) * 64 + 31] = 0xA5A55A5Au;

  std::map<uint64_t, uint32_t> mem;
  auto write32 = [&](uint64_t address, uint32_t value) { mem[address] = value; };
  auto read32 = [&](uint64_t address) -> uint32_t {
    auto it = mem.find(address);
    return it == mem.end() ? 0u : it->second;
  };

  const auto layout = kmd::serialize_queue_cwsr(kCtxBase, kAreaSize, {wave}, write32, kArch);
  ASSERT_TRUE(layout.ok);
  EXPECT_EQ(read32(kCtxBase + layout.control_stack_offset + 8) & 0x3Fu, 0x3Fu);

  std::vector<kmd::CwsrWaveState> output(1);
  output[0].num_sgprs = wave.num_sgprs;
  output[0].num_vgprs = wave.num_vgprs;
  ASSERT_TRUE(kmd::deserialize_queue_cwsr(kCtxBase, kAreaSize, output, read32, kArch));
  EXPECT_EQ(output[0].vgprs[(kMaxVgprs - 1) * 64 + 31], 0xA5A55A5Au);

  mem.clear();
  wave.num_vgprs = kMaxVgprs + 1;
  EXPECT_FALSE(kmd::serialize_queue_cwsr(kCtxBase, kAreaSize, {wave}, write32, kArch).ok);
  EXPECT_TRUE(mem.empty());
}

TEST(WaveDebugTest, CwsrHonorsTheRuntimeControlStackBoundary) {
  constexpr uint64_t kCtxBase = 0x400000000ULL;
  constexpr uint32_t kAreaSize = 0x40000, kControlCapacity = 4096;
  for (const auto arch :
       {ROCJITSU_CODE_ARCH_CDNA3, ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA5}) {
    auto wave = make_wave(0x1234, 0x98765432);
    std::map<uint64_t, uint32_t> memory;
    auto write32 = [&](uint64_t address, uint32_t value) { memory[address] = value; };
    const auto layout =
        kmd::serialize_queue_cwsr(kCtxBase, kAreaSize, {wave}, write32, arch, kControlCapacity);
    ASSERT_TRUE(layout.ok);
    EXPECT_EQ(layout.control_stack_offset + layout.control_stack_size, kControlCapacity);
    EXPECT_EQ(layout.wave_state_offset - layout.wave_state_size, kControlCapacity);
    std::vector<kmd::CwsrWaveState> restored = {wave};
    restored[0].pc = 0;
    EXPECT_TRUE(kmd::deserialize_queue_cwsr(
        kCtxBase, kAreaSize, restored, [&](uint64_t address) { return memory[address]; }, arch,
        kControlCapacity));
    EXPECT_EQ(restored[0].pc, wave.pc);
    EXPECT_FALSE(
        kmd::serialize_queue_cwsr(kCtxBase, kAreaSize, {wave}, write32, arch, sizeof(uint32_t)).ok);
  }
}

TEST(WaveDebugTest, CwsrRetainsTheShaderEngineAndScratchSlot) {
  constexpr uint64_t kCtxBase = 0x400000000ULL;
  constexpr uint32_t kAreaSize = 0x40000;
  for (const auto arch : {ROCJITSU_CODE_ARCH_CDNA3, ROCJITSU_CODE_ARCH_CDNA4}) {
    for (uint32_t engine = 0; engine < 4; ++engine) {
      auto wave = make_wave(0xAA01, 0x2000);
      wave.shader_engine_id = engine;
      wave.scratch_scoreboard_id = 173;
      std::map<uint64_t, uint32_t> memory;
      auto layout = kmd::serialize_queue_cwsr(
          kCtxBase, kAreaSize, {wave},
          [&](uint64_t address, uint32_t value) { memory[address] = value; }, arch);
      ASSERT_TRUE(layout.ok);
      uint32_t word = memory[kCtxBase + layout.control_stack_offset + 3 * sizeof(uint32_t)];
      EXPECT_EQ((word >> 9) & 3u, engine);
      EXPECT_EQ(word & 0x1ffu, 173u);
    }
  }
}

TEST(WaveDebugTest, CwsrLdsUsesTheTargetAbiGranularity) {
  constexpr uint64_t kCtxBase = 0x400000000ULL;
  constexpr uint32_t kAreaSize = 0x40000;
  for (const auto arch : {ROCJITSU_CODE_ARCH_CDNA3, ROCJITSU_CODE_ARCH_CDNA4}) {
    const uint32_t granule = arch == ROCJITSU_CODE_ARCH_CDNA3 ? 512 : 1280;
    for (const uint32_t bytes : {1u, 512u, 513u, 1280u, 1291u}) {
      SCOPED_TRACE(static_cast<int>(arch));
      SCOPED_TRACE(bytes);
      auto wave = make_wave(0x1234, 0x98765432);
      wave.is_first_in_group = wave.is_last_in_group = true;
      wave.lds.resize(bytes, 0xA5);
      std::map<uint64_t, uint32_t> memory;
      auto layout = kmd::serialize_queue_cwsr(
          kCtxBase, kAreaSize, {wave},
          [&](uint64_t address, uint32_t value) { memory[address] = value; }, arch);
      ASSERT_TRUE(layout.ok);
      auto parsed = parse_cwsr(memory, kCtxBase, granule);
      ASSERT_EQ(parsed.size(), 1u);
      EXPECT_EQ(parsed[0].pc, wave.pc);
      EXPECT_EQ(parsed[0].wave_id, wave.wave_id);
      EXPECT_EQ(parsed[0].lds.size(), ((bytes + granule - 1) / granule) * granule);
      EXPECT_TRUE(std::equal(wave.lds.begin(), wave.lds.end(), parsed[0].lds.begin()));
    }
  }
}

TEST(WaveDebugTest, CwsrSerializationRoundTripsThroughDbgapiLayout) {
  constexpr uint64_t kCtxBase = 0x400000000ULL;
  constexpr uint32_t kAreaSize = 0x40000; // 256 KiB

  std::map<uint64_t, uint32_t> mem;
  auto write32 = [&](uint64_t va, uint32_t val) { mem[va] = val; };

  std::vector<kmd::CwsrWaveState> waves = {make_wave(0xAA01, 0x2000), make_wave(0xAA02, 0x2040),
                                           make_wave(0xAA03, 0x2080)};
  waves[0].wave_in_group = 0;
  waves[0].is_first_in_group = true;
  waves[0].is_last_in_group = false;
  waves[1].wave_in_group = 1;
  waves[1].is_first_in_group = false;
  waves[1].is_last_in_group = true;
  waves[2].group_ids[0] = 2;
  waves[2].is_first_in_group = true;
  waves[2].is_last_in_group = true;
  waves[0].lds = {0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88};
  waves[2].lds = {0xA1, 0xB2, 0xC3, 0xD4, 0xE5, 0xF6, 0x07, 0x18};

  kmd::CwsrLayout layout =
      kmd::serialize_queue_cwsr(kCtxBase, kAreaSize, waves, write32, ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_TRUE(layout.ok);

  std::vector<ParsedWave> parsed = parse_cwsr(mem, kCtxBase, /*lds_granule=*/1280);
  ASSERT_EQ(parsed.size(), waves.size());

  for (size_t i = 0; i < waves.size(); ++i) {
    const auto &in = waves[i];
    const auto &out = parsed[i];
    EXPECT_EQ(out.pc, in.pc) << "wave " << i;
    EXPECT_EQ(out.exec, in.exec);
    EXPECT_EQ(out.status, in.status);
    EXPECT_EQ(out.mode, in.mode);
    EXPECT_EQ(out.m0, in.m0);
    EXPECT_EQ(out.wave_id, in.wave_id);
    EXPECT_EQ(out.vcc, in.vcc);
    EXPECT_EQ(out.group[0], in.group_ids[0]);
    EXPECT_EQ(out.group[1], in.group_ids[1]);
    EXPECT_EQ(out.group[2], in.group_ids[2]);
    EXPECT_EQ(out.first, i == 0 || i == 2);
    EXPECT_EQ(out.last, i == 1 || i == 2);
    // TTMP6: wave_stopped (bit30) and trap id (bits 25:28).
    EXPECT_TRUE(out.ttmp6 & (1u << 30));
    EXPECT_EQ((out.ttmp6 >> 25) & 0xF, in.trap_id);
    // TTMP11: trap-handler-setup (bit31) and packet id (bits 6:30).
    EXPECT_TRUE(out.ttmp11 & (1u << 31));
    EXPECT_EQ((out.ttmp11 >> 6) & 0x1FFFFFF, in.queue_packet_id);
    // Meaningful scalars and vectors round-trip.
    for (uint32_t s = 0; s < in.num_sgprs; ++s)
      EXPECT_EQ(out.sgprs[s], in.sgprs[s]) << "sgpr " << s;
    for (uint32_t r = 0; r < in.num_vgprs; ++r)
      for (uint32_t l = 0; l < 64; ++l)
        EXPECT_EQ(out.vgprs[r * 64 + l], in.vgprs[r * 64 + l]) << "vgpr " << r << " lane " << l;
    if (in.is_first_in_group) {
      ASSERT_GE(out.lds.size(), in.lds.size());
      EXPECT_TRUE(std::ranges::equal(in.lds, std::span(out.lds).first(in.lds.size())));
    } else {
      EXPECT_TRUE(out.lds.empty());
    }
  }

  const auto original_mem = mem;
  auto invalid = make_wave(1, 0x3000);
  invalid.num_vgprs = 257;
  EXPECT_FALSE(
      kmd::serialize_queue_cwsr(kCtxBase, kAreaSize, {invalid}, write32, ROCJITSU_CODE_ARCH_CDNA4)
          .ok);
  EXPECT_EQ(mem, original_mem);

  EXPECT_FALSE(
      kmd::serialize_queue_cwsr(kCtxBase + 1, kAreaSize, waves, write32, ROCJITSU_CODE_ARCH_CDNA4)
          .ok);
  EXPECT_FALSE(
      kmd::serialize_queue_cwsr(UINT64_MAX - 3, kAreaSize, waves, write32, ROCJITSU_CODE_ARCH_CDNA4)
          .ok);
  EXPECT_FALSE(kmd::serialize_queue_cwsr(kCtxBase, layout.wave_state_offset - 1, waves, write32,
                                         ROCJITSU_CODE_ARCH_CDNA4)
                   .ok);
  EXPECT_EQ(mem, original_mem);
}

// rocm-dbgapi applies each COMPUTE_RELAUNCH state to following wave words.
// A large LDS group must not make unrelated groups consume the same stride.
TEST(WaveDebugTest, Cdna3CwsrUsesEachWavesPhysicalSgprAllocation) {
  constexpr uint64_t base = 0x400000000ULL;
  constexpr uint32_t bytes = 0x40000;
  const std::array<uint32_t, 3> encoded{16, 24, 106}, physical{16, 32, 112}, vcc_slot{14, 30, 106};
  std::vector<kmd::CwsrWaveState> waves;
  for (uint32_t i = 0; i < encoded.size(); ++i) {
    auto wave = make_wave(i + 1, 0x2000 + 4 * i);
    wave.num_sgprs = encoded[i];
    wave.sgprs.resize(encoded[i], 0x01020304);
    wave.vcc = 0x123456789abcdef0ull + i;
    wave.flat_scratch = 0x23456789abcdef00ull + i;
    waves.push_back(wave);
  }
  std::map<uint64_t, uint32_t> image;
  auto write32 = [&](uint64_t address, uint32_t value) { image[address] = value; };
  auto read32 = [&](uint64_t address) { return image.at(address); };
  auto layout = kmd::serialize_queue_cwsr(base, bytes, waves, write32, ROCJITSU_CODE_ARCH_CDNA3);
  ASSERT_TRUE(layout.ok);
  const auto parsed = parse_cwsr(image, base, 512);
  ASSERT_EQ(parsed.size(), waves.size());
  for (size_t i = 0; i < waves.size(); ++i) {
    ASSERT_EQ(parsed[i].sgprs.size(), physical[i]);
    EXPECT_EQ(parsed[i].vcc, waves[i].vcc);
    const uint64_t vcc = parsed[i].sgprs_addr + vcc_slot[i] * 4;
    const uint64_t flat = vcc - 4 * sizeof(uint32_t);
    EXPECT_EQ(read32(flat), static_cast<uint32_t>(waves[i].flat_scratch));
    EXPECT_EQ(read32(flat + 4), static_cast<uint32_t>(waves[i].flat_scratch >> 32));
    write32(flat, 0x87654321u);
    write32(vcc + 4, 0xfedcba98u);
  }
  ASSERT_TRUE(kmd::deserialize_queue_cwsr(base, bytes, waves, read32, ROCJITSU_CODE_ARCH_CDNA3));
  for (uint32_t i = 0; i < waves.size(); ++i) {
    EXPECT_EQ(waves[i].flat_scratch, 0x2345678987654321ull);
    EXPECT_EQ(waves[i].vcc, 0xfedcba989abcdef0ull + i);
  }
}

TEST(WaveDebugTest, CwsrUsesGroupLdsStateChanges) {
  constexpr uint64_t kCtxBase = 0x400000000ULL;
  constexpr uint32_t kAreaSize = 0x4000;
  for (const auto arch :
       {ROCJITSU_CODE_ARCH_CDNA3, ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA5}) {
    SCOPED_TRACE(static_cast<int>(arch));
    std::vector<kmd::CwsrWaveState> waves = {make_wave(1, 0x1000), make_wave(2, 0x2000),
                                             make_wave(3, 0x3000)};
    for (size_t i = 0; i < waves.size(); ++i)
      waves[i].group_ids[0] = static_cast<uint32_t>(i + 1);
    waves[0].lds.resize(4096, 0x5a);

    std::map<uint64_t, uint32_t> mem;
    auto write32 = [&](uint64_t address, uint32_t value) { mem[address] = value; };
    auto read32 = [&](uint64_t address) -> uint32_t {
      auto it = mem.find(address);
      return it == mem.end() ? 0u : it->second;
    };
    const auto layout = kmd::serialize_queue_cwsr(kCtxBase, kAreaSize, waves, write32, arch);
    ASSERT_TRUE(layout.ok);
    const uint32_t state_words = arch == ROCJITSU_CODE_ARCH_CDNA5 ? 2 : 1;
    EXPECT_EQ(layout.control_stack_size, (2 + 2 * state_words + waves.size()) * sizeof(uint32_t));
    if (arch != ROCJITSU_CODE_ARCH_CDNA5) {
      const auto parsed =
          parse_cwsr(mem, kCtxBase, arch == ROCJITSU_CODE_ARCH_CDNA3 ? 512u : 1280u);
      ASSERT_EQ(parsed.size(), waves.size());
      EXPECT_EQ(parsed[0].lds.size(), arch == ROCJITSU_CODE_ARCH_CDNA3 ? 4096u : 5120u);
      EXPECT_TRUE(parsed[1].lds.empty());
      EXPECT_TRUE(parsed[2].lds.empty());
    }
    ASSERT_TRUE(kmd::deserialize_queue_cwsr(kCtxBase, kAreaSize, waves, read32, arch));
    EXPECT_EQ(waves[0].lds.size(), 4096u);
    EXPECT_TRUE(waves[1].lds.empty());
    EXPECT_TRUE(waves[2].lds.empty());
    EXPECT_EQ(waves[0].lds[0], 0x5a);
  }
}

TEST(WaveDebugTest, GroupLdsChangesPreserveRuntimeContextBoundaries) {
  constexpr uint64_t kCtxBase = 0x400000000ULL;
  constexpr uint32_t kAreaSize = 64 * 1024;
  constexpr uint32_t kControlStackCapacity = 4096;
  const std::array<uint32_t, 6> runtime_fields{128 * 1024, 4096, 0x1234, 0x5678, 42, 0xA5A5};
  for (const auto arch :
       {ROCJITSU_CODE_ARCH_CDNA3, ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA5}) {
    SCOPED_TRACE(static_cast<int>(arch));
    std::vector<kmd::CwsrWaveState> waves = {make_wave(1, 0x1000), make_wave(2, 0x2000),
                                             make_wave(3, 0x3000)};
    for (size_t i = 0; i < waves.size(); ++i)
      waves[i].group_ids[0] = static_cast<uint32_t>(i + 1);
    waves[0].lds.assign(5121, 0x5A);
    waves[1].lds.clear();
    waves[2].lds.assign(129, 0x7C);
    std::map<uint64_t, uint32_t> mem;
    for (size_t i = 0; i < runtime_fields.size(); ++i)
      mem[kCtxBase + 16 + i * sizeof(uint32_t)] = runtime_fields[i];
    auto write32 = [&](uint64_t address, uint32_t value) { mem[address] = value; };
    auto read32 = [&](uint64_t address) { return mem.at(address); };
    const auto layout =
        kmd::serialize_queue_cwsr(kCtxBase, kAreaSize, waves, write32, arch, kControlStackCapacity,
                                  kmd::CwsrDebugRegion{runtime_fields[0], runtime_fields[1]});
    ASSERT_TRUE(layout.ok);
    const uint32_t state_words = arch == ROCJITSU_CODE_ARCH_CDNA5 ? 2 : 1;
    EXPECT_EQ(layout.control_stack_size, (2 + waves.size() * (state_words + 1)) * sizeof(uint32_t));
    EXPECT_EQ(layout.control_stack_offset + layout.control_stack_size, kControlStackCapacity);
    EXPECT_EQ(layout.wave_state_offset - layout.wave_state_size, kControlStackCapacity);
    for (size_t i = 0; i < runtime_fields.size(); ++i)
      EXPECT_EQ(read32(kCtxBase + 16 + i * sizeof(uint32_t)), runtime_fields[i]);
    if (arch != ROCJITSU_CODE_ARCH_CDNA5) {
      const uint32_t granule = arch == ROCJITSU_CODE_ARCH_CDNA3 ? 512 : 1280;
      const auto parsed = parse_cwsr(mem, kCtxBase, granule);
      ASSERT_EQ(parsed.size(), waves.size());
      for (size_t i = 0; i < waves.size(); ++i)
        EXPECT_EQ(parsed[i].lds.size(), (waves[i].lds.size() + granule - 1) / granule * granule);
    }
    auto restored = waves;
    for (auto &wave : restored) {
      wave.pc = 0;
      std::ranges::fill(wave.lds, 0);
    }
    ASSERT_TRUE(kmd::deserialize_queue_cwsr(kCtxBase, kAreaSize, restored, read32, arch,
                                            kControlStackCapacity));
    for (size_t i = 0; i < waves.size(); ++i) {
      EXPECT_EQ(restored[i].pc, waves[i].pc);
      EXPECT_EQ(restored[i].lds, waves[i].lds);
    }
  }
}

// rocm-dbgapi carves 32-byte instruction buffers out of a per-queue "debugger
// memory" region declared in the context-save header (debug_offset/size at byte
// 16/20). It aborts displaced stepping if that region is absent, so the
// serializer must reserve a non-zero, in-bounds, non-overlapping region.
TEST(WaveDebugTest, CwsrReservesDebuggerMemoryForDisplacedStepping) {
  constexpr uint64_t kCtxBase = 0x400000000ULL;
  constexpr uint32_t kAreaSize = 0x40000; // 256 KiB

  std::map<uint64_t, uint32_t> mem;
  auto write32 = [&](uint64_t va, uint32_t val) { mem[va] = val; };
  auto rd = [&](uint64_t va) -> uint32_t {
    auto it = mem.find(va);
    return it == mem.end() ? 0u : it->second;
  };

  std::vector<kmd::CwsrWaveState> waves = {make_wave(0xAA01, 0x2000)};
  kmd::CwsrLayout layout =
      kmd::serialize_queue_cwsr(kCtxBase, kAreaSize, waves, write32, ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_TRUE(layout.ok);

  // The header advertises the region dbgapi reads (kfd_context_save_area_header
  // debug_offset/size).
  EXPECT_EQ(rd(kCtxBase + 16), layout.debug_offset);
  EXPECT_EQ(rd(kCtxBase + 20), layout.debug_size);

  // Non-zero, or dbgapi aborts with "reserved memory is missing".
  EXPECT_NE(layout.debug_offset, 0u);
  EXPECT_NE(layout.debug_size, 0u);
  // 64-byte aligned (DEBUGGER_BYTES_ALIGN) so each 32-byte chunk is aligned.
  EXPECT_EQ(layout.debug_offset % 64u, 0u);
  EXPECT_EQ(layout.debug_size % 64u, 0u);
  // Sits above the wave area (no overlap) and inside the save area.
  EXPECT_GE(layout.debug_offset, layout.wave_state_offset);
  EXPECT_LE(layout.debug_offset + layout.debug_size, kAreaSize);
  // Holds enough 32-byte chunks for dbgapi's park + terminating buffers plus a
  // per-wave displaced-step buffer.
  EXPECT_GE(layout.debug_size / 32u, waves.size() + 2);
}

// FLAT_SCRATCH and VCC are written into aliased slots at the top of the saved
// SGPR block. num_sgprs is accepted up to 106, so those slots overlap real
// SGPR indices: serialization overwrote s102/s103 with FLAT_SCRATCH and
// deserialization read the alias bytes back as if they were registers, while
// flat_scratch itself was never restored at all. A round trip therefore
// corrupted live register state and lost the scratch base.
TEST(WaveDebugTest, CwsrRoundTripPreservesAliasedSgprsAndFlatScratch) {
  constexpr uint64_t kCtxBase = 0x400000000ULL;
  constexpr uint32_t kAreaSize = 0x40000;
  // The full saved block. CDNA3/CDNA4 default to 112 SGPRs per wave
  // (config_loader default_sgprs_per_wf), so a wave that uses the whole
  // allocation is the ordinary case, not an extreme one.
  constexpr uint32_t kMaxAcceptedSgprs = kmd::kCwsrSavedSgprSlots;
  constexpr uint64_t kFlatScratch = 0x0000DEAD0000BEEFULL;

  std::map<uint64_t, uint32_t> mem;
  auto write32 = [&](uint64_t va, uint32_t val) { mem[va] = val; };
  auto read32 = [&](uint64_t va) -> uint32_t {
    auto it = mem.find(va);
    return it == mem.end() ? 0u : it->second;
  };

  std::vector<kmd::CwsrWaveState> in = {make_wave(0xCC01, 0x3000)};
  in[0].num_sgprs = kMaxAcceptedSgprs;
  in[0].sgprs.resize(kMaxAcceptedSgprs);
  for (uint32_t s = 0; s < kMaxAcceptedSgprs; ++s)
    in[0].sgprs[s] = 0xA5A50000u + s;
  in[0].flat_scratch = kFlatScratch;
  in[0].vcc = 0x0123456789ABCDEFULL;

  ASSERT_TRUE(
      kmd::serialize_queue_cwsr(kCtxBase, kAreaSize, in, write32, ROCJITSU_CODE_ARCH_CDNA4).ok);

  std::vector<kmd::CwsrWaveState> out(1);
  out[0].num_sgprs = in[0].num_sgprs;
  out[0].num_vgprs = in[0].num_vgprs;
  ASSERT_TRUE(
      kmd::deserialize_queue_cwsr(kCtxBase, kAreaSize, out, read32, ROCJITSU_CODE_ARCH_CDNA4));

  EXPECT_EQ(out[0].flat_scratch, kFlatScratch) << "flat_scratch was not restored";
  EXPECT_EQ(out[0].vcc, in[0].vcc);

  // FLAT_SCRATCH occupies s102/s103 and VCC s106/s107. Only the slots an alias
  // actually covers may be lost: clamping at the gfx9.4 architected count of
  // 102 instead would also drop s104 and s105, which sit *between* the two
  // pairs and are ordinary registers. That is why the predicate is an explicit
  // slot test and not a bound.
  for (uint32_t s = 0; s < kMaxAcceptedSgprs; ++s) {
    if (kmd::cwsr_sgpr_slot_is_aliased(s, ROCJITSU_CODE_ARCH_CDNA4))
      EXPECT_EQ(out[0].sgprs[s], 0u) << "aliased slot " << s << " decoded as an SGPR";
    else
      EXPECT_EQ(out[0].sgprs[s], in[0].sgprs[s]) << "sgpr " << s << " did not survive";
  }
  // Pin the two slots the "everything at or above 102 is an alias" shortcut
  // would silently drop, so a future simplification back to a bound fails here.
  EXPECT_EQ(out[0].sgprs[104], in[0].sgprs[104]);
  EXPECT_EQ(out[0].sgprs[105], in[0].sgprs[105]);
}

// The saved block is 112 slots, and CDNA3/CDNA4 default to exactly that many
// SGPRs per wave. When the codec accepted only 106 the geometry for a
// default-configured wave came back not-ok, so the whole queue's CWSR publish
// failed and the debugger saw no waves at all -- reachable from any CDNA4
// config that simply omits sgprs_per_wf.
TEST(WaveDebugTest, CwsrAcceptsTheFullDefaultSgprAllocation) {
  constexpr uint64_t kCtxBase = 0x400000000ULL;
  constexpr uint32_t kAreaSize = 0x40000;

  std::map<uint64_t, uint32_t> mem;
  auto write32 = [&](uint64_t va, uint32_t val) { mem[va] = val; };

  std::vector<kmd::CwsrWaveState> in = {make_wave(0xCC02, 0x3000)};
  in[0].num_sgprs = kmd::kCwsrSavedSgprSlots;
  in[0].sgprs.assign(kmd::kCwsrSavedSgprSlots, 0u);
  EXPECT_TRUE(
      kmd::serialize_queue_cwsr(kCtxBase, kAreaSize, in, write32, ROCJITSU_CODE_ARCH_CDNA4).ok)
      << "a wave using the default " << kmd::kCwsrSavedSgprSlots << "-SGPR allocation was rejected";

  // One past the block is genuinely out of range and must still be refused
  // rather than written past the end of the SGPR area.
  mem.clear();
  in[0].num_sgprs = kmd::kCwsrSavedSgprSlots + 1;
  in[0].sgprs.assign(in[0].num_sgprs, 0u);
  EXPECT_FALSE(
      kmd::serialize_queue_cwsr(kCtxBase, kAreaSize, in, write32, ROCJITSU_CODE_ARCH_CDNA4).ok);
  EXPECT_TRUE(mem.empty()) << "a rejected image still wrote to the ctx-save area";
}

TEST(WaveDebugTest, CwsrDeserializeRecoversSerializedWaveState) {
  // serialize_queue_cwsr followed by deserialize_queue_cwsr must round-trip the
  // register state the resume path reads back (the debugger writes its edits
  // straight into this area, and resume reloads them onto the live wave).
  constexpr uint64_t kCtxBase = 0x400000000ULL;
  constexpr uint32_t kAreaSize = 0x40000; // 256 KiB

  std::map<uint64_t, uint32_t> mem;
  auto write32 = [&](uint64_t va, uint32_t val) { mem[va] = val; };
  auto read32 = [&](uint64_t va) -> uint32_t {
    auto it = mem.find(va);
    return it == mem.end() ? 0u : it->second;
  };

  std::vector<kmd::CwsrWaveState> in = {make_wave(0xBB01, 0x3000), make_wave(0xBB02, 0x3080)};
  // Give the second wave MODE.debug_en set so the round-trip preserves it (the
  // resume path uses this bit to select single-step).
  in[1].mode = (1u << 11);

  ASSERT_TRUE(
      kmd::serialize_queue_cwsr(kCtxBase, kAreaSize, in, write32, ROCJITSU_CODE_ARCH_CDNA4).ok);

  // Reload: supply only the per-wave geometry (num_sgprs/num_vgprs) so the
  // decoder reproduces the exact layout, then read the values back.
  std::vector<kmd::CwsrWaveState> out;
  for (const auto &w : in) {
    kmd::CwsrWaveState g;
    g.num_sgprs = w.num_sgprs;
    g.num_vgprs = w.num_vgprs;
    out.push_back(g);
  }
  ASSERT_TRUE(
      kmd::deserialize_queue_cwsr(kCtxBase, kAreaSize, out, read32, ROCJITSU_CODE_ARCH_CDNA4));
  ASSERT_EQ(out.size(), in.size());

  for (size_t i = 0; i < in.size(); ++i) {
    EXPECT_EQ(out[i].pc, in[i].pc) << "wave " << i;
    EXPECT_EQ(out[i].exec, in[i].exec);
    EXPECT_EQ(out[i].vcc, in[i].vcc);
    EXPECT_EQ(out[i].status, in[i].status);
    EXPECT_EQ(out[i].mode, in[i].mode);
    EXPECT_EQ(out[i].m0, in[i].m0);
    EXPECT_EQ(out[i].wave_id, in[i].wave_id);
    EXPECT_EQ(out[i].wave_stopped, in[i].wave_stopped);
    EXPECT_EQ(out[i].spi_ttmps_setup, in[i].spi_ttmps_setup);
    EXPECT_EQ(out[i].group_ids, in[i].group_ids);
    EXPECT_EQ(out[i].wave_in_group, in[i].wave_in_group);
    EXPECT_EQ(out[i].queue_packet_id, in[i].queue_packet_id);
    for (uint32_t s = 0; s < in[i].num_sgprs; ++s)
      EXPECT_EQ(out[i].sgprs[s], in[i].sgprs[s]) << "wave " << i << " sgpr " << s;
    for (uint32_t r = 0; r < in[i].num_vgprs; ++r)
      for (uint32_t l = 0; l < 64; ++l)
        EXPECT_EQ(out[i].vgprs[r * 64 + l], in[i].vgprs[r * 64 + l])
            << "wave " << i << " vgpr " << r << " lane " << l;
  }
}

TEST(WaveDebugTest, CwsrBulkCodecMatchesDwordCodecAndPublishesHeaderLast) {
  constexpr uint64_t kCtxBase = 0x400000000ULL;
  constexpr uint32_t kAreaSize = 0x40000;
  std::vector<kmd::CwsrWaveState> waves = {make_wave(0xCC01, 0x4000), make_wave(0xCC02, 0x4080)};

  std::map<uint64_t, uint32_t> dwords;
  const auto reference = kmd::serialize_queue_cwsr(
      kCtxBase, kAreaSize, waves,
      [&](uint64_t address, uint32_t value) { dwords[address] = value; }, ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_TRUE(reference.ok);

  std::vector<uint8_t> bulk_image(reference.wave_state_offset);
  std::vector<std::pair<uint64_t, size_t>> writes;
  const auto bulk = kmd::serialize_queue_cwsr_bulk(
      kCtxBase, kAreaSize, waves,
      [&](uint64_t address, std::span<const uint8_t> bytes) {
        writes.emplace_back(address, bytes.size());
        const size_t offset = static_cast<size_t>(address - kCtxBase);
        ASSERT_LE(offset + bytes.size(), bulk_image.size());
        std::memcpy(bulk_image.data() + offset, bytes.data(), bytes.size());
      },
      ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_TRUE(bulk.ok);
  EXPECT_EQ(bulk.control_stack_offset, reference.control_stack_offset);
  EXPECT_EQ(bulk.control_stack_size, reference.control_stack_size);
  EXPECT_EQ(bulk.wave_state_offset, reference.wave_state_offset);
  EXPECT_EQ(bulk.wave_state_size, reference.wave_state_size);
  EXPECT_EQ(bulk.debug_offset, reference.debug_offset);
  EXPECT_EQ(bulk.debug_size, reference.debug_size);

  ASSERT_EQ(writes.size(), 2u);
  EXPECT_EQ(writes[0].first, kCtxBase + reference.control_stack_offset);
  EXPECT_EQ(writes[0].second, reference.wave_state_offset - reference.control_stack_offset);
  EXPECT_EQ(writes[1], (std::pair<uint64_t, size_t>{kCtxBase, 10 * sizeof(uint32_t)}));

  for (uint32_t offset = 0; offset < reference.wave_state_offset; offset += sizeof(uint32_t)) {
    uint32_t actual = 0;
    std::memcpy(&actual, bulk_image.data() + offset, sizeof(actual));
    auto expected = dwords.find(kCtxBase + offset);
    EXPECT_EQ(actual, expected == dwords.end() ? 0u : expected->second) << "offset " << offset;
  }
}

TEST(WaveDebugTest, CwsrPublicationPreservesRuntimeFieldsAndDebuggerMemory) {
  constexpr uint64_t kBase = 0x400000000ULL;
  constexpr uint32_t kAreaSize = 0x40000;
  const kmd::CwsrDebugRegion debug{kAreaSize - 256, 256};
  const std::array<uint32_t, 6> owned = {debug.offset, debug.size, 0x76543210u,
                                         0x1234u,      83u,        0xA5A5u};
  for (const auto arch :
       {ROCJITSU_CODE_ARCH_CDNA3, ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA5}) {
    SCOPED_TRACE(static_cast<int>(arch));
    for (bool bulk : {false, true}) {
      SCOPED_TRACE(bulk);
      auto wave = make_wave(0xAA01, 0x6000);
      wave.is_first_in_group = wave.is_last_in_group = true;
      wave.lds.resize(513, 0x5A);
      std::vector<uint8_t> image(kAreaSize, 0xA5);
      std::memcpy(image.data() + 16, owned.data(), sizeof(owned));
      auto write_block = [&](uint64_t address, std::span<const uint8_t> bytes) {
        ASSERT_GE(address, kBase);
        ASSERT_LE(address - kBase + bytes.size(), image.size());
        if (address == kBase) {
          EXPECT_EQ(bytes.size(), 16u);
        }
        std::memcpy(image.data() + (address - kBase), bytes.data(), bytes.size());
      };
      const auto layout =
          bulk ? kmd::serialize_queue_cwsr_bulk(kBase, kAreaSize, {wave}, write_block, arch, 0,
                                                debug)
               : kmd::serialize_queue_cwsr(
                     kBase, kAreaSize, {wave},
                     [&](uint64_t address, uint32_t value) {
                       EXPECT_FALSE(address >= kBase + 16 && address < kBase + 40);
                       std::memcpy(image.data() + (address - kBase), &value, sizeof(value));
                     },
                     arch, 0, debug);
      ASSERT_TRUE(layout.ok);
      EXPECT_EQ(layout.debug_offset, debug.offset);
      EXPECT_EQ(layout.debug_size, debug.size);
      EXPECT_EQ(std::memcmp(image.data() + 16, owned.data(), sizeof(owned)), 0);
      EXPECT_TRUE(std::ranges::all_of(std::span(image).subspan(debug.offset, debug.size),
                                      [](uint8_t value) { return value == 0xA5; }));
      std::vector<kmd::CwsrWaveState> restored{wave};
      ASSERT_TRUE(kmd::deserialize_queue_cwsr_bulk(
          kBase, kAreaSize, restored,
          [&](uint64_t address, std::span<uint8_t> bytes) {
            ASSERT_GE(address, kBase);
            ASSERT_LE(address - kBase + bytes.size(), image.size());
            std::memcpy(bytes.data(), image.data() + (address - kBase), bytes.size());
          },
          arch));
      EXPECT_EQ(restored[0].pc, wave.pc);
      EXPECT_EQ(restored[0].vgprs, wave.vgprs);
      bool wrote = false;
      EXPECT_FALSE(kmd::serialize_queue_cwsr_bulk(
                       kBase, kAreaSize, {wave},
                       [&](uint64_t, std::span<const uint8_t>) { wrote = true; }, arch, 0,
                       kmd::CwsrDebugRegion{512, 64})
                       .ok);
      EXPECT_FALSE(wrote);
    }
  }
}

TEST(WaveDebugTest, CwsrBulkDeserializeRoundTripsAndRejectsCorruptHeader) {
  constexpr uint64_t kCtxBase = 0x400000000ULL;
  constexpr uint32_t kAreaSize = 0x40000;
  std::vector<kmd::CwsrWaveState> input = {make_wave(0xDD01, 0x5000), make_wave(0xDD02, 0x5080)};
  input[1].mode = 1u << 11;

  std::vector<uint8_t> image(kAreaSize);
  ASSERT_TRUE(kmd::serialize_queue_cwsr_bulk(
                  kCtxBase, kAreaSize, input,
                  [&](uint64_t address, std::span<const uint8_t> bytes) {
                    std::memcpy(image.data() + (address - kCtxBase), bytes.data(), bytes.size());
                  },
                  ROCJITSU_CODE_ARCH_CDNA4)
                  .ok);

  auto geometry_only = [&] {
    std::vector<kmd::CwsrWaveState> states(input.size());
    for (size_t index = 0; index < input.size(); ++index) {
      states[index].num_sgprs = input[index].num_sgprs;
      states[index].num_vgprs = input[index].num_vgprs;
    }
    return states;
  };
  auto read_block = [&](uint64_t address, std::span<uint8_t> bytes) {
    std::memcpy(bytes.data(), image.data() + (address - kCtxBase), bytes.size());
  };

  auto output = geometry_only();
  ASSERT_TRUE(kmd::deserialize_queue_cwsr_bulk(kCtxBase, kAreaSize, output, read_block,
                                               ROCJITSU_CODE_ARCH_CDNA4));
  ASSERT_EQ(output.size(), input.size());
  for (size_t index = 0; index < input.size(); ++index) {
    EXPECT_EQ(output[index].pc, input[index].pc);
    EXPECT_EQ(output[index].exec, input[index].exec);
    EXPECT_EQ(output[index].vcc, input[index].vcc);
    EXPECT_EQ(output[index].mode, input[index].mode);
    EXPECT_EQ(output[index].wave_id, input[index].wave_id);
    EXPECT_EQ(output[index].group_ids, input[index].group_ids);
    EXPECT_EQ(output[index].sgprs, input[index].sgprs);
    EXPECT_EQ(output[index].vgprs, input[index].vgprs);
  }

  image[8] ^= 1;
  output = geometry_only();
  EXPECT_FALSE(kmd::deserialize_queue_cwsr_bulk(kCtxBase, kAreaSize, output, read_block,
                                                ROCJITSU_CODE_ARCH_CDNA4));
}

TEST(WaveDebugTest, CwsrBulkCodecUsesConstantBlockOperationsAtFullScale) {
  constexpr uint64_t kCtxBase = 0x400000000ULL;
  constexpr uint32_t kWaveCount = 1024;
  constexpr uint32_t kAreaSize = 80 * 1024 * 1024;
  std::vector<kmd::CwsrWaveState> waves(kWaveCount);
  for (uint32_t index = 0; index < kWaveCount; ++index) {
    auto &wave = waves[index];
    wave.pc = 0x1000 + index * 4;
    wave.wave_id = index + 1;
    wave.group_ids[0] = index / 4;
    wave.wave_in_group = index % 4;
    wave.queue_packet_id = 1;
    wave.num_sgprs = 106;
    wave.num_vgprs = 256;
    wave.sgprs.resize(wave.num_sgprs, index);
    wave.vgprs.resize(static_cast<size_t>(wave.num_vgprs) * 64, index);
  }

  size_t write_calls = 0;
  size_t bytes_written = 0;
  const auto layout = kmd::serialize_queue_cwsr_bulk(
      kCtxBase, kAreaSize, waves,
      [&](uint64_t, std::span<const uint8_t> bytes) {
        ++write_calls;
        bytes_written += bytes.size();
      },
      ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_TRUE(layout.ok);
  EXPECT_EQ(write_calls, 2u);
  EXPECT_EQ(bytes_written,
            layout.wave_state_offset - layout.control_stack_offset + 10 * sizeof(uint32_t));
  EXPECT_GT(layout.wave_state_size, 64u * 1024u * 1024u);
}

} // namespace
