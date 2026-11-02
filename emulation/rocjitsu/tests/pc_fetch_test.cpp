// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/gpu_memory_access.h"
#include "rocjitsu/vm/amdgpu/gpu_vm.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <iomanip>
#include <memory>
#include <string>

namespace {

using namespace rocjitsu;

struct PcFetchArchitecture {
  rj_code_arch_t arch;
  bool swap;
  const char *name;
};

class PcFetchTest : public ::testing::TestWithParam<PcFetchArchitecture> {
protected:
  void SetUp() override {
    const auto profile = GetParam().arch == ROCJITSU_CODE_ARCH_RDNA4
                             ? amdgpu::Gfx12VmConfig::gfx12_0()
                             : amdgpu::Gfx12VmConfig::gfx12_1();
    first_page_table_shift = 12 + 9 * (profile.page_table_levels() - 1);
    vm = std::make_unique<amdgpu::GpuVm>(profile);
    physical = std::make_shared<amdgpu::GpuMemoryPhysicalAccess>(memory);
    address_space = vm->register_gfx12_address_space(77, kPageTableRoot, physical);
    ASSERT_TRUE(address_space);

    amdgpu::ComputeUnitCore::Config config{};
    config.arch = GetParam().arch;
    config.num_wf_slots = 1;
    config.sgprs_per_wf = 106;
    config.vgprs_per_wf = 16;
    config.lds_size_kb = 64;
    cu = amdgpu::ComputeUnitCore::create("pc_fetch", config, &memory, &cache);
    ASSERT_NE(cu, nullptr);
    cache.set_gpu_vm(vm.get());
    cu->set_gpu_vm(vm.get());
    wave = cu->dispatch_wf(0, kBranchPc, config.sgprs_per_wf, config.vgprs_per_wf);
    ASSERT_NE(wave, nullptr);
    wave->set_process_id(77);
    wave->set_address_space(address_space);

    map_page(kBranchPc, kBranchPhysicalPage);
    // s_setpc_b64 / s_set_pc_i64 s[4:5], or
    // s_swappc_b64 / s_swap_pc_i64 s[6:7], s[4:5].
    memory.write32(kBranchPhysicalPage, GetParam().swap ? build_s_swappc_b64(6, 4, GetParam().arch)
                                                        : build_s_setpc_b64(4, GetParam().arch));
    // s_mov_b32 s8, 42: execution proves the high target was actually fetched.
    memory.write32(kTargetPhysicalPage,
                   build_s_mov_b32(8, scalar_positive_inline_u32(42), GetParam().arch));
    memory.write32(kTargetPhysicalPage + 4, build_s_endpgm(GetParam().arch));
  }

  void TearDown() override {
    if (wave && !wave->is_halted())
      wave->halt();
  }

  void map_page(uint64_t virtual_address, uint64_t physical_address) {
    constexpr uint64_t kValid = 1;
    constexpr uint64_t kExecutable = uint64_t{1} << 4;
    constexpr uint64_t kReadable = uint64_t{1} << 5;
    constexpr uint64_t kPdePte = uint64_t{1} << 63;
    uint64_t table = kPageTableRoot;
    for (uint32_t shift = first_page_table_shift; shift > 12; shift -= 9) {
      const uint64_t entry_address = table + ((virtual_address >> shift) & 0x1FF) * 8;
      uint64_t entry = memory.read64(entry_address);
      if (!(entry & kValid)) {
        entry = next_page_table | kValid;
        memory.write64(entry_address, entry);
        next_page_table += 4096;
      }
      table = entry & ~uint64_t{4095};
    }
    memory.write64(table + ((virtual_address >> 12) & 0x1FF) * 8,
                   physical_address | kValid | kPdePte | kExecutable | kReadable);
  }

  void run_branch_and_fetch(uint64_t encoded_target, uint64_t target, uint64_t load_bias = 0) {
    SCOPED_TRACE(::testing::Message() << "encoded=" << std::hex << encoded_target
                                      << " target=" << target << " load_bias=" << load_bias);
    map_page(target, kTargetPhysicalPage);
    ASSERT_EQ(vm->translate(address_space, target, 4, amdgpu::VmAccessKind::Execute).outcome,
              amdgpu::VmAccessOutcome::Complete);
    wave->pc = kBranchPc;
    wave->set_code_load_bias(load_bias);
    const uint32_t base = wave->sgpr_alloc().base;
    cu->write_sgpr(base + 4, static_cast<uint32_t>(encoded_target));
    cu->write_sgpr(base + 5, static_cast<uint32_t>(encoded_target >> 32));
    cu->write_sgpr(base + 8, 0xDEADBEEFu);

    cu->step();
    ASSERT_FALSE(wave->is_halted());
    EXPECT_EQ(wave->pc, target);
    EXPECT_EQ(vm->translate(address_space, wave->pc, 4, amdgpu::VmAccessKind::Execute).outcome,
              amdgpu::VmAccessOutcome::Complete);
    if (GetParam().swap) {
      EXPECT_EQ(cu->read_sgpr(base + 6), kBranchPc + 4);
      EXPECT_EQ(cu->read_sgpr(base + 7), 0u);
    }

    cu->step();
    ASSERT_FALSE(wave->is_halted());
    EXPECT_EQ(cu->read_sgpr(base + 8), 42u);
    EXPECT_EQ(wave->pc, target + 4);
  }

  static constexpr uint64_t kPageTableRoot = 0x1000;
  static constexpr uint64_t kBranchPc = 0x10000;
  static constexpr uint64_t kBranchPhysicalPage = 0x80000;
  static constexpr uint64_t kTargetPhysicalPage = 0x81000;
  uint64_t next_page_table = 0x2000;
  uint32_t first_page_table_shift = 0;
  amdgpu::GpuMemory memory{"pc_fetch_memory"};
  amdgpu::L2Cache cache{"pc_fetch_cache"};
  std::shared_ptr<amdgpu::GpuMemoryPhysicalAccess> physical;
  std::unique_ptr<amdgpu::GpuVm> vm;
  std::unique_ptr<amdgpu::ComputeUnitCore> cu;
  amdgpu::AddressSpaceHandle address_space;
  amdgpu::Wavefront *wave = nullptr;
};

INSTANTIATE_TEST_SUITE_P(
    Gfx12Architectures, PcFetchTest,
    ::testing::Values(PcFetchArchitecture{ROCJITSU_CODE_ARCH_RDNA4, false, "Gfx1201Setpc"},
                      PcFetchArchitecture{ROCJITSU_CODE_ARCH_RDNA4, true, "Gfx1201Swappc"},
                      PcFetchArchitecture{ROCJITSU_CODE_ARCH_CDNA5, false, "Gfx1250Setpc"},
                      PcFetchArchitecture{ROCJITSU_CODE_ARCH_CDNA5, true, "Gfx1250Swappc"}),
    [](const ::testing::TestParamInfo<PcFetchArchitecture> &info) {
      return std::string(info.param.name);
    });

TEST_P(PcFetchTest, FetchesMappedHighAddressAfterIndirectBranch) {
  if (GetParam().arch == ROCJITSU_CODE_ARCH_RDNA4) {
    ASSERT_NO_FATAL_FAILURE(run_branch_and_fetch(0x0000800012345000ULL, 0x0000800012345000ULL));
    ASSERT_NO_FATAL_FAILURE(run_branch_and_fetch(0xFFFF800012345000ULL, 0x0000800012345000ULL));
    ASSERT_NO_FATAL_FAILURE(
        run_branch_and_fetch(0x0000FFFF12345000ULL, 0x0000FFFF12345000ULL, 0x100000));
    ASSERT_NO_FATAL_FAILURE(
        run_branch_and_fetch(0xFFFFFFFF12345000ULL, 0x0000FFFF12345000ULL, 0x100000));
  } else {
    ASSERT_NO_FATAL_FAILURE(run_branch_and_fetch(0x0100000012345000ULL, 0x0100000012345000ULL));
    ASSERT_NO_FATAL_FAILURE(run_branch_and_fetch(0xFF00000012345000ULL, 0x0100000012345000ULL));
    // This is a genuine positive 57-bit address, even with a nonzero load bias.
    ASSERT_NO_FATAL_FAILURE(
        run_branch_and_fetch(0x0001FFFF12345000ULL, 0x0001FFFF12345000ULL, 0x100000));
    // The top 4 GiB of the 57-bit aperture also has a valid signed encoding.
    ASSERT_NO_FATAL_FAILURE(
        run_branch_and_fetch(0x01FFFFFF12345000ULL, 0x01FFFFFF12345000ULL, 0x100000));
    ASSERT_NO_FATAL_FAILURE(
        run_branch_and_fetch(0xFFFFFFFF12345000ULL, 0x01FFFFFF12345000ULL, 0x100000));
  }
}

} // namespace
