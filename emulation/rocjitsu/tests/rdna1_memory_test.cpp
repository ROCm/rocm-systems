// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "compute_unit_test_util.h"
#include "counting_gpu_memory.h"
#include "decode_test_util.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna1/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna1/opcodes.h"
#include "rocjitsu/kmd/linux/kfd_process.h"
#include "rocjitsu/kmd/linux/legacy_gpu_vm.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/hwreg.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/memory_pipeline.h"
#include "rocjitsu/vm/amdgpu/memory_wait_scoreboard.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <memory>
#include <optional>
#include <tuple>
#include <vector>

namespace rocjitsu {
namespace {
class Rdna1Memory : public testing::Test {
protected:
  virtual rj_code_arch_t arch() const { return ROCJITSU_CODE_ARCH_RDNA1; }

  void SetUp() override {
    config_.arch = arch();
    config_.num_wf_slots = 2;
    config_.sgprs_per_wf = 106;
    config_.vgprs_per_wf = 32;
    config_.lds_size_kb = 64;
    cache_.set_backing_memory(&memory_);
    ASSERT_NO_FATAL_FAILURE(
        test::initialize_compute_unit(cu_, "rdna1_cu", config_, &memory_, &cache_));
    decoder_ = Decoder::create(config_.arch);
    wf_ = cu_->dispatch_wf(0, 0, 106, 16, 64);
    ASSERT_TRUE(wf_);
    wf_->set_exec(1);
    wf_->set_lds_base(512);
    wf_->set_lds_size(512);
  }
  void TearDown() override {
    if (wf_)
      wf_->halt();
  }
  void execute(std::array<uint32_t, 2> words) {
    inst_.reset(decode_valid(*decoder_, words.data()));
    ASSERT_NE(inst_, nullptr);
    ASSERT_TRUE(cu_->execute_instruction(inst_.get(), *wf_).succeeded());
  }
  void issue(std::array<uint32_t, 2> words) {
    ASSERT_NO_FATAL_FAILURE(execute(words));
    ASSERT_NE(inst_->data(), nullptr);
    if (inst_->data()->tag() == amdgpu::LOCAL_MEM)
      ASSERT_EQ(local_.issue(inst_.release(), *wf_), amdgpu::VmAccessOutcome::Complete);
    else {
      amdgpu::GlobalMemPipeline pipeline(&cu_->l1_vector(), &cache_);
      ASSERT_EQ(pipeline.issue(inst_.release(), *wf_), amdgpu::VmAccessOutcome::Complete);
    }
  }
  void scalar(unsigned reg, uint32_t value) {
    cu_->write_sgpr(wf_->sgpr_alloc().base + reg, value);
  }
  void vector(unsigned reg, unsigned lane, uint32_t value) {
    cu_->write_vgpr(wf_->vgpr_alloc().base + reg, lane, value);
  }
  uint32_t vector(unsigned reg, unsigned lane) const {
    return cu_->read_vgpr(wf_->vgpr_alloc().base + reg, lane);
  }
  amdgpu::ComputeUnitCore::Config config_{};
  amdgpu::GpuMemory memory_{"rdna1_memory"};
  amdgpu::L2Cache cache_{"rdna1_cache"};
  std::unique_ptr<amdgpu::ComputeUnitCore> cu_;
  amdgpu::Wavefront *wf_ = nullptr;
  std::unique_ptr<Decoder> decoder_;
  std::unique_ptr<Instruction> inst_;
  amdgpu::LocalMemPipeline local_;
};

TEST_F(Rdna1Memory, ScalarScratchScalesOnlyAlignedRegisterOffset) {
  scalar(0, 0x4003);
  scalar(1, 0);
  scalar(2, 7);
  for (unsigned op : {rdna1::kSLoadDword, rdna1::kSScratchLoadDword, rdna1::kSScratchLoadDwordx2,
                      rdna1::kSScratchLoadDwordx4}) {
    ASSERT_NO_FATAL_FAILURE(
        execute(rdna1::build_smem(op, {.sdata = 8, .offset = 7, .soffset = 2})));
    EXPECT_EQ(inst_->data_as<amdgpu::ScalarMemState>()->addr,
              op == rdna1::kSLoadDword ? 0x4008u : 0x4104u);
  }
}

TEST_F(Rdna1Memory, RawBuffersRespectUnboundResourcesAlignmentAndLdsExtent) {
  scalar(0, 0x4000);
  scalar(1, 0);
  scalar(2, 256);
  memory_.write32(0x4000, 0x44332211);
  memory_.write32(0x4004, 0x88776655);
  const auto load =
      rdna1::build_mubuf(rdna1::kBufferLoadDword, {.offset = 1, .vdata = 8, .soffset = 128});
  scalar(3, 3u << 28);
  vector(8, 0, ~0u);
  ASSERT_NO_FATAL_FAILURE(issue(load));
  EXPECT_EQ(vector(8, 0), 0u);
  scalar(3, (3u << 28) | (20u << 12));
  ASSERT_NO_FATAL_FAILURE(issue(load));
  EXPECT_EQ(vector(8, 0), 0x44332211u);
  // M0[15:0] points at the last DWORD. Lane 1 would reach the next group's LDS.
  wf_->set_exec(3);
  wf_->set_m0(0x10000 + 508);
  const auto load_lds =
      rdna1::build_mubuf(rdna1::kBufferLoadUbyte, {.lds = 1, .vdata = 8, .soffset = 128});
  wf_->lds().write32(1020, ~0u);
  wf_->lds().write32(1024, 0xcccccccc);
  ASSERT_NO_FATAL_FAILURE(issue(load_lds));
  EXPECT_EQ(wf_->lds().read32(1020), 0x11u);
  EXPECT_EQ(wf_->lds().read32(1024), 0xccccccccu);
}

class SparseTranslator final : public amdgpu::AddressSpaceTranslator {
public:
  amdgpu::VmTranslationResult translate(uint64_t address, size_t size,
                                        amdgpu::VmAccessKind access) const override {
    return identity.translate(address, size, access);
  }
  std::shared_ptr<const amdgpu::PrtRanges> nonresident_prt_ranges() const override {
    auto snapshot = holes;
    if (next_holes)
      holes = *next_holes;
    return snapshot;
  }

  amdgpu::IdentityAddressSpaceTranslator identity;
  std::optional<std::shared_ptr<const amdgpu::PrtRanges>> next_holes;
  mutable std::shared_ptr<const amdgpu::PrtRanges> holes =
      std::make_shared<const amdgpu::PrtRanges>(amdgpu::PrtRanges{{0x5000, 0x6000}});
};

class Gfx10MemoryAllocation : public Rdna1Memory,
                              public testing::WithParamInterface<rj_code_arch_t> {
protected:
  rj_code_arch_t arch() const override { return GetParam(); }

  void bind_single_texel_image(uint64_t address) {
    const bool gfx1010 = arch() == ROCJITSU_CODE_ARCH_RDNA1;
    scalar(0, address >> 8);
    scalar(1, (20u << 20) | (gfx1010 ? 1u << 29 : 0u));
    scalar(3, (9u << 28) | 4u);
    // GFX10.3 encodes linear pitch in texels minus one. A DWORD row spans 256 bytes.
    scalar(4, gfx1010 ? 0u : 63u);
  }
};

INSTANTIATE_TEST_SUITE_P(Gfx10, Gfx10MemoryAllocation,
                         testing::Values(ROCJITSU_CODE_ARCH_RDNA1, ROCJITSU_CODE_ARCH_RDNA2));

TEST_P(Gfx10MemoryAllocation, DebuggerAndTexelPipelineShareTheIssuedResidencySnapshot) {
  constexpr uint32_t code = 0x8000, texel = 0x5000, value = 0x13579000;
  amdgpu::GpuVm vm;
  auto translator = std::make_shared<SparseTranslator>();
  const auto hole = translator->holes;
  auto handle =
      vm.register_address_space(5, translator, std::make_shared<test::CountingGpuMemory>(memory_));
  cu_->set_gpu_vm(&vm);
  wf_->set_address_space(handle);
  wf_->set_process_id(5);
  cu_->set_debug_active(true);
  std::vector<uint64_t> watches;
  cu_->set_watchpoint_handler(
      [&](amdgpu::Wavefront &, uint64_t address, uint32_t bytes, bool, bool) {
        EXPECT_EQ(bytes, sizeof(value));
        watches.push_back(address);
        return false;
      });
  memory_.write32(texel, value);
  const auto buffer =
      rdna1::build_mubuf(rdna1::kBufferLoadDword, {.vdata = 8, .tfe = 1, .soffset = 128});
  const auto sample = rdna1::build_mimg(rdna1::kImageSampleLzMimg,
                                        {.dim = 1, .dmask = 1, .tfe = 1, .vdata = 8, .ssamp = 2});
  for (bool sampled : {false, true}) {
    for (unsigned reg = 0; reg < 12; ++reg)
      scalar(reg, 0);
    if (sampled) {
      bind_single_texel_image(texel);
    } else {
      scalar(0, texel);
      scalar(2, 4);
      scalar(3, (3u << 28) | (20u << 12));
    }
    const auto &words = sampled ? sample : buffer;
    memory_.load_image(reinterpret_cast<const uint8_t *>(words.data()), sizeof(words), code);
    for (bool initially_resident : {true, false}) {
      SCOPED_TRACE(testing::Message()
                   << "sampled=" << sampled << " initially_resident=" << initially_resident);
      // Publish the opposite PRT map just after the instruction takes its first snapshot.
      translator->holes = initially_resident ? nullptr : hole;
      translator->next_holes = initially_resident ? hole : nullptr;
      wf_->pc = code;
      vector(8, 0, 0xdeadbeef);
      vector(9, 0, 0);
      watches.clear();
      ASSERT_TRUE(cu_->step());
      EXPECT_EQ(vector(8, 0), initially_resident ? value : 0u);
      EXPECT_EQ(vector(9, 0), initially_resident ? 0u : 1u);
      EXPECT_EQ(watches,
                initially_resident ? std::vector<uint64_t>{texel} : std::vector<uint64_t>{});
    }
  }
  cu_->set_debug_active(false);
  cu_->set_watchpoint_handler({});
  wf_->set_address_space({});
  cu_->set_gpu_vm(nullptr);
}

TEST_F(Rdna1Memory, UnallocatedDestinationSuppressesReturningGdsAtomic) {
  ASSERT_EQ(wf_->num_vgprs(), 16u);
  ASSERT_EQ(config_.vgprs_per_wf, 32u);
  wf_->set_m0(8);
  vector(0, 0, 4);
  vector(1, 0, 3);
  vector(16, 0, 0x1616);
  EXPECT_FALSE(cu_->gws_device().has_gds_data());
  cu_->gws_device().access_gds(cu_->arch(), 0, [](amdgpu::Lds &gds) { gds.write32(4, 9); });
  EXPECT_TRUE(cu_->gws_device().has_gds_data());
  const auto add = [](uint8_t dst) {
    return rdna1::build_ds(rdna1::kDsAddRtnU32, {.gds = 1, .addr = 0, .data0 = 1, .vdst = dst});
  };
  ASSERT_NO_FATAL_FAILURE(execute(add(16)));
  const auto *state = inst_->data_as<amdgpu::VectorMemState>();
  ASSERT_NE(state, nullptr);
  EXPECT_EQ(state->exec_mask, 0u);
  EXPECT_EQ(state->wait_counter_type, amdgpu::WaitCounterType::LGKMCNT);
  EXPECT_EQ(inst_->amdgpu_memory_issue_info()->counter_obligations().size(), 2u);
  EXPECT_EQ(amdgpu::MemoryWaitScoreboard::result_lanes(*inst_, *wf_), 0u);
  EXPECT_EQ(local_.issue(inst_.release(), *wf_), amdgpu::VmAccessOutcome::Complete);
  cu_->gws_device().access_gds(cu_->arch(), 0,
                               [](amdgpu::Lds &gds) { EXPECT_EQ(gds.read32(4), 9u); });
  EXPECT_EQ(vector(16, 0), 0x1616u);
  EXPECT_EQ(wf_->exec_raw(), 1u);
  ASSERT_NO_FATAL_FAILURE(issue(add(15)));
  cu_->gws_device().access_gds(cu_->arch(), 0,
                               [](amdgpu::Lds &gds) { EXPECT_EQ(gds.read32(4), 12u); });
  EXPECT_EQ(vector(15, 0), 9u);
}

TEST_P(Gfx10MemoryAllocation, BufferLdsLoadsExtendNarrowValuesAtM0PlusOffset) {
  constexpr uint32_t address = 0x4000;
  constexpr uint32_t offset = 4;
  constexpr uint32_t m0_offset = 32;
  constexpr uint32_t first = 0x123480ab;
  constexpr uint32_t second = 0x987681f5;
  constexpr uint32_t sentinel = 0xcccccccc;
  scalar(0, address);
  scalar(1, 0);
  scalar(2, 256);
  scalar(3, (3u << 28) | (1u << 24) | (20u << 12) | 4u); // 32_UINT, select X.
  wf_->set_exec(5);
  wf_->set_m0(0x10000 + m0_offset);
  vector(2, 0, 0);
  vector(2, 2, 8);
  memory_.write32(address + offset, first);
  memory_.write32(address + offset + 8, second);
  const auto destination = wf_->lds_base() + m0_offset + offset;
  for (const auto &[op, lane0, lane2] :
       {std::tuple{rdna1::kBufferLoadSbyte, 0xffffffabu, 0xfffffff5u},
        std::tuple{rdna1::kBufferLoadSshort, 0xffff80abu, 0xffff81f5u},
        std::tuple{rdna1::kBufferLoadFormatX, first, second}}) {
    SCOPED_TRACE(op);
    for (unsigned i = 0; i < 5; ++i)
      wf_->lds().write32(destination - 4 + 4 * i, sentinel);
    ASSERT_NO_FATAL_FAILURE(issue(rdna1::build_mubuf(
        op, {.offset = offset, .offen = 1, .lds = 1, .vaddr = 2, .vdata = 8, .soffset = 128})));
    EXPECT_EQ(wf_->lds().read32(destination - 4), sentinel);
    EXPECT_EQ(wf_->lds().read32(destination), lane0);
    EXPECT_EQ(wf_->lds().read32(destination + 4), sentinel);
    EXPECT_EQ(wf_->lds().read32(destination + 8), lane2);
    EXPECT_EQ(wf_->lds().read32(destination + 12), sentinel);
  }
}

TEST_P(Gfx10MemoryAllocation, GenericFlatIgnoresImmediateBitEleven) {
  constexpr uint32_t pointer = 0x100000;
  vector(2, 0, pointer);
  vector(3, 0, 0);
  for (const auto &[word, address] : {std::pair{0xdc300800u, pointer},
                                      {0xdc3007fcu, pointer + 2044},
                                      {0xdc308800u, pointer - 2048}}) {
    const std::array words{word, 0x017d0002u};
    ASSERT_NO_FATAL_FAILURE(execute(words));
    const auto *data = inst_->data_as<amdgpu::VectorMemState>();
    ASSERT_NE(data, nullptr);
    EXPECT_EQ(data->per_lane_addr[0], address);
    if (word != 0xdc308800u) {
      const auto lanes = amdgpu::MemoryWaitScoreboard::flat_lanes(*inst_, *wf_, address, address);
      EXPECT_EQ(lanes.shared, 1u);
    }
  }
}

TEST_P(Gfx10MemoryAllocation, FlatWaitUsesTheSharedApertureAtIssue) {
  constexpr uint16_t sh_mem_bases = 15 | (31 << 11);
  const auto setreg = rdna1::build_sopk(rdna1::kSSetregB32, {.simm16 = sh_mem_bases, .sdst = 2});
  const auto load = rdna1::build_flat(rdna1::kFlatLoadDword, {.saddr = 125, .vdst = 8});
  const auto vm_wait = rdna1::build_sopp(rdna1::kSWaitcnt, {.simm16 = 0x3f70});
  const auto read = rdna1::build_vop1(rdna1::kVMovB32Vop1, {.src0 = 256 + 8, .vdst = 9});
  const auto full_wait = rdna1::build_sopp(rdna1::kSWaitcnt, {.simm16 = 0});
  constexpr uint64_t pc = 0x1000;
  const std::array code{setreg[0], load[0], load[1], vm_wait[0], read[0], full_wait[0], read[0]};
  for (size_t i = 0; i < code.size(); ++i)
    memory_.write32(pc + 4 * i, code[i]);

  for (bool ptr32 : {false, true}) {
    if (ptr32 && arch() != ROCJITSU_CODE_ARCH_RDNA1)
      continue;
    const uint64_t launch = uint64_t{1} << (ptr32 ? 16 : 48);
    for (bool relocate : {false, true}) {
      SCOPED_TRACE(ptr32);
      SCOPED_TRACE(relocate);
      wf_->halt();
      cu_->set_apertures(launch, launch + 0xffff, 0, 0, ptr32);
      wf_ = cu_->dispatch_wf(0, relocate ? pc : pc + 4, 106, 16, 64);
      ASSERT_NE(wf_, nullptr);
      wf_->set_exec(1);
      wf_->set_lds_size(512);
      wf_->lds().write32(wf_->lds_base() + 4, 0x12345678);
      scalar(2, 2u << 16);
      const uint64_t address = (relocate ? 2 * launch : launch) + 4;
      vector(0, 0, static_cast<uint32_t>(address));
      vector(1, 0, address >> 32);
      vector(9, 0, 0);

      const auto warnings = cu_->memory_wait_diagnostic_count();
      if (relocate) {
        ASSERT_TRUE(cu_->step()); // s_setreg_b32 SH_MEM_BASES, s2
      }
      ASSERT_TRUE(cu_->step()); // flat_load_dword v8, v[0:1]
      ASSERT_TRUE(cu_->step()); // s_waitcnt vmcnt(0)
      ASSERT_TRUE(cu_->step()); // v_mov_b32 v9, v8
      EXPECT_EQ(vector(9, 0), 0x12345678u);
      EXPECT_EQ(cu_->memory_wait_diagnostic_count(), warnings + 1);
      ASSERT_TRUE(cu_->step()); // s_waitcnt 0
      ASSERT_TRUE(cu_->step()); // v_mov_b32 v9, v8
      EXPECT_EQ(cu_->memory_wait_diagnostic_count(), warnings + 1);
    }
  }
}

TEST_P(Gfx10MemoryAllocation, UnallocatedLastWordSuppressesWholeLdsTuple) {
  wf_->set_m0(8);
  vector(0, 0, 4);
  vector(15, 0, 0x1515);
  vector(16, 0, 0x1616);
  wf_->lds().write32(wf_->lds_base() + 4, 41);
  ASSERT_NO_FATAL_FAILURE(issue(rdna1::build_ds(rdna1::kDsRead2B32, {.addr = 0, .vdst = 15})));
  EXPECT_EQ(vector(15, 0), 0x1515u);
  EXPECT_EQ(vector(16, 0), 0x1616u);
  ASSERT_NO_FATAL_FAILURE(issue(rdna1::build_ds(rdna1::kDsRead2B32, {.addr = 0, .vdst = 14})));
  EXPECT_EQ(vector(14, 0), 41u);
  EXPECT_EQ(vector(15, 0), 41u);
}

TEST_P(Gfx10MemoryAllocation, UnallocatedBufferTupleSuppressesFaultingReadsIncludingLastWord) {
  amdgpu::GpuVm vm;
  auto backing = std::make_shared<test::CountingGpuMemory>(memory_);
  auto handle = vm.register_address_space(5, std::make_shared<SparseTranslator>(), backing);
  cu_->set_gpu_vm(&vm);
  wf_->set_address_space(handle);
  wf_->set_process_id(5);
  scalar(0, 0x4000);
  scalar(1, 0);
  scalar(2, 256);
  scalar(3, (3u << 28) | (20u << 12));
  vector(15, 0, 0x1515);
  vector(16, 0, 0x1616);
  backing->fault_reads = true;
  const auto load = [](uint8_t dst) {
    return rdna1::build_mubuf(rdna1::kBufferLoadDwordx2, {.vdata = dst, .soffset = 128});
  };
  ASSERT_NO_FATAL_FAILURE(issue(load(15)));
  ASSERT_NO_FATAL_FAILURE(issue(rdna1::build_mtbuf(rdna1::kTbufferLoadFormatXy,
                                                   {.format = 20, .vdata = 15, .soffset = 128})));
  EXPECT_EQ(backing->reads, 0u);
  EXPECT_EQ(vector(15, 0), 0x1515u);
  EXPECT_EQ(vector(16, 0), 0x1616u);
  ASSERT_NO_FATAL_FAILURE(execute(load(14)));
  amdgpu::GlobalMemPipeline pipeline(&cu_->l1_vector(), &cache_);
  EXPECT_EQ(pipeline.issue(inst_.release(), *wf_), amdgpu::VmAccessOutcome::Faulted);
  EXPECT_EQ(backing->reads, 1u);
  wf_->set_address_space({});
  cu_->set_gpu_vm(nullptr);
}

TEST_P(Gfx10MemoryAllocation, ConditionalAtomicResultsAndLdsRedirectUseActualDestination) {
  constexpr uint32_t address = 0x4000;
  memory_.write32(address, 9);
  vector(0, 0, address);
  vector(1, 0, 0);
  vector(2, 0, 3);
  vector(16, 0, 0x1616);
  const auto add = [](uint8_t glc) {
    return rdna1::build_flat(rdna1::kFlatAtomicAdd,
                             {.seg = 2, .glc = glc, .data = 2, .saddr = 125, .vdst = 16});
  };
  ASSERT_NO_FATAL_FAILURE(issue(add(1)));
  EXPECT_EQ(memory_.read32(address), 9u);
  EXPECT_EQ(vector(16, 0), 0x1616u);
  ASSERT_NO_FATAL_FAILURE(issue(add(0)));
  EXPECT_EQ(memory_.read32(address), 12u);
  EXPECT_EQ(vector(16, 0), 0x1616u);

  scalar(0, address);
  scalar(1, 0);
  scalar(2, 256);
  scalar(3, (3u << 28) | (20u << 12));
  wf_->set_m0(0);
  ASSERT_NO_FATAL_FAILURE(
      issue(rdna1::build_mubuf(rdna1::kBufferLoadDword, {.lds = 1, .vdata = 16, .soffset = 128})));
  EXPECT_EQ(wf_->lds().read32(wf_->lds_base()), 12u);
  EXPECT_EQ(vector(16, 0), 0x1616u);

  // CMPSWAP consumes two VGPRs but returns only the old dword. V15 is a
  // legal destination even though the second source is outside the allocation.
  const auto cmp =
      rdna1::build_mubuf(rdna1::kBufferAtomicCmpswap, {.glc = 1, .vdata = 15, .soffset = 128});
  ASSERT_NO_FATAL_FAILURE(issue(cmp));
  EXPECT_EQ(vector(15, 0), 12u);
  EXPECT_EQ(vector(16, 0), 0x1616u);

  // In a non-returning MUBUF atomic, VDATA is only an input. A source with
  // an undefined value must still issue the transaction with the active lanes.
  ASSERT_NO_FATAL_FAILURE(
      execute(rdna1::build_mubuf(rdna1::kBufferAtomicAdd, {.vdata = 16, .soffset = 128})));
  const auto *state = inst_->data_as<amdgpu::VectorMemState>();
  ASSERT_NE(state, nullptr);
  EXPECT_EQ(state->lane_mask, 1u);
  EXPECT_FALSE(state->is_load);
  amdgpu::GlobalMemPipeline pipeline(&cu_->l1_vector(), &cache_);
  EXPECT_EQ(pipeline.issue(inst_.release(), *wf_), amdgpu::VmAccessOutcome::Complete);
  EXPECT_EQ(vector(16, 0), 0x1616u);
}

TEST_P(Gfx10MemoryAllocation, InlineImageResultHonorsMaskWidth) {
  scalar(0, 0x4000 >> 8);
  scalar(1, (20u << 20) | (1u << 29));
  scalar(2, 0);
  scalar(3, 9u << 28);
  for (unsigned i = 4; i < 8; ++i)
    scalar(i, 0);
  vector(0, 0, 0);
  vector(15, 0, 0x1515);
  vector(16, 0, 0x1616);
  // With two components, the whole tuple v[15:16] must be suppressed.
  const auto query = [](uint8_t dmask) {
    return rdna1::build_mimg(rdna1::kImageGetResinfo, {.dim = 1, .dmask = dmask, .vdata = 15});
  };
  ASSERT_NO_FATAL_FAILURE(execute(query(3)));
  EXPECT_EQ(vector(15, 0), 0x1515u);
  EXPECT_EQ(vector(16, 0), 0x1616u);
  ASSERT_NO_FATAL_FAILURE(execute(query(1)));
  EXPECT_EQ(vector(15, 0), 1u);
  EXPECT_EQ(vector(16, 0), 0x1616u);
}

TEST_F(Rdna1Memory, BufferTfeReportsOnlyExplicitSparseHoles) {
  amdgpu::GpuVm vm;
  auto backing = std::make_shared<test::CountingGpuMemory>(memory_);
  uint32_t faults = 0;
  auto handle = vm.register_address_space(5, std::make_shared<SparseTranslator>(), backing,
                                          [&](uint64_t, amdgpu::VmAccessKind) { ++faults; });
  cu_->set_gpu_vm(&vm);
  wf_->set_address_space(handle);
  wf_->set_process_id(5);
  wf_->set_exec(3);
  scalar(0, 0x4000);
  scalar(1, 0);
  scalar(2, 0x3000);
  scalar(3, (3u << 28) | (20u << 12));
  vector(0, 0, 0);
  vector(0, 1, 0x1000);
  vector(9, 0, 0xdeadbeef);
  vector(9, 1, 0);
  memory_.write32(0x4000, 0xaabbccdd);
  const auto load = rdna1::build_mubuf(rdna1::kBufferLoadDword,
                                       {.offen = 1, .vdata = 8, .tfe = 1, .soffset = 128});
  ASSERT_NO_FATAL_FAILURE(issue(load));
  EXPECT_EQ(vector(8, 0), 0xaabbccddu);
  EXPECT_EQ(vector(8, 1), 0u);
  EXPECT_EQ(vector(9, 0), 0xdeadbeefu);
  EXPECT_EQ(vector(9, 1), 1u);
  EXPECT_EQ(faults, 0u);
  EXPECT_EQ(backing->reads, 1u);

  // V15 is allocated but the required V16 status is outside this wave.
  const auto out_of_range = rdna1::build_mubuf(rdna1::kBufferLoadDword,
                                               {.offen = 1, .vdata = 15, .tfe = 1, .soffset = 128});
  vector(15, 0, 1234);
  vector(16, 1, 0x76543210); // The CU slot has storage beyond the wave allocation.
  ASSERT_NO_FATAL_FAILURE(issue(out_of_range));
  EXPECT_EQ(vector(15, 0), 1234u);
  EXPECT_EQ(vector(16, 1), 0x76543210u);
  EXPECT_EQ(backing->reads, 1u);

  backing->fault_reads = true;
  ASSERT_NO_FATAL_FAILURE(execute(load));
  amdgpu::GlobalMemPipeline pipeline(&cu_->l1_vector(), &cache_);
  EXPECT_EQ(pipeline.issue(inst_.release(), *wf_), amdgpu::VmAccessOutcome::Faulted);
  EXPECT_EQ(vector(9, 0), 0xdeadbeefu);
  wf_->set_address_space({});
  cu_->set_gpu_vm(nullptr);
}

TEST_F(Rdna1Memory, ImageTfeTakesPrecedenceOverLodWarningAndPreservesResidentStatus) {
  amdgpu::GpuVm vm;
  auto backing = std::make_shared<test::CountingGpuMemory>(memory_);
  auto handle = vm.register_address_space(5, std::make_shared<SparseTranslator>(), backing);
  cu_->set_gpu_vm(&vm);
  wf_->set_address_space(handle);
  wf_->set_process_id(5);
  for (unsigned i = 0; i < 12; ++i)
    scalar(i, 0);
  scalar(0, 0x4000 >> 8);
  scalar(1, (20u << 20) | (1u << 29)); // 1x1 linear R32_UINT resource.
  scalar(3, (9u << 28) | 4);
  scalar(5, 256u << 8); // T#.MIN_LOD_WARN = 1.0.
  vector(0, 0, 0);
  vector(1, 0, 0);
  vector(9, 0, 0xdeadbeef);
  memory_.write32(0x4000, 12345);
  const auto load =
      rdna1::build_mimg(rdna1::kImageLoadMimg, {.dim = 1, .dmask = 1, .tfe = 1, .vdata = 8});
  ASSERT_NO_FATAL_FAILURE(issue(load));
  EXPECT_EQ(vector(8, 0), 12345u);
  EXPECT_EQ(vector(9, 0), 0xdeadbeefu);
  const auto sample =
      rdna1::build_mimg(rdna1::kImageSampleLzMimg,
                        {.dim = 1, .dmask = 1, .tfe = 1, .lwe = 1, .vdata = 8, .ssamp = 2});
  const auto expect_hwreg_bit = [&](uint16_t reg, uint32_t expected) {
    const auto get = rdna1::build_sopk(
        rdna1::kSGetregB32, {.simm16 = static_cast<uint16_t>(reg | (10u << 6)), .sdst = 12});
    ASSERT_NO_FATAL_FAILURE(execute({get[0], 0}));
    EXPECT_EQ(cu_->read_sgpr(wf_->sgpr_alloc().base + 12), expected);
  };
  ASSERT_NO_FATAL_FAILURE(expect_hwreg_bit(1, 0));
  ASSERT_NO_FATAL_FAILURE(issue(sample));
  EXPECT_EQ(vector(8, 0), 12345u);
  EXPECT_EQ(vector(9, 0), 2u);
  ASSERT_NO_FATAL_FAILURE(expect_hwreg_bit(1, 1)); // MODE.LOD_CLAMPED
  scalar(0, 0x5000 >> 8);
  const auto prior_reads = backing->reads;
  ASSERT_NO_FATAL_FAILURE(issue(sample));
  EXPECT_EQ(vector(8, 0), 0u);
  EXPECT_EQ(vector(9, 0), 1u);
  EXPECT_EQ(backing->reads, prior_reads);
  ASSERT_NO_FATAL_FAILURE(expect_hwreg_bit(1, 1)); // Sticky across the NACK.
  wf_->set_address_space({});
  cu_->set_gpu_vm(nullptr);
}

TEST_P(Gfx10MemoryAllocation, DebuggerProbesOnlyResidentTexelRanges) {
  constexpr uint64_t code_va = 0x40000000, resident = 0x50000000;
  constexpr uint64_t hole = resident + amdgpu::kLegacyPageSize;
  constexpr uint64_t unmapped = hole + amdgpu::kLegacyPageSize;
  constexpr uint32_t vmid = 7, texel = 0xdecafbad, sentinel = 0xa5a5;
  amdgpu::GpuVm vm;
  amdgpu::LegacyGpuVmAdapter adapter(vm, &memory_);
  KfdProcess process(vmid);
  std::array<uint8_t, amdgpu::kLegacyPageSize> code{}, page{};
  std::memcpy(page.data(), &texel, sizeof(texel));
  const auto buffer = rdna1::build_mubuf(rdna1::kBufferLoadDword,
                                         {.offen = 1, .vdata = 8, .tfe = 1, .soffset = 128});
  const auto image =
      rdna1::build_mimg(rdna1::kImageLoadMimg, {.dim = 1, .dmask = 1, .tfe = 1, .vdata = 8});
  const auto sample = rdna1::build_mimg(rdna1::kImageSampleLzMimg,
                                        {.dim = 1, .dmask = 1, .tfe = 1, .vdata = 8, .ssamp = 2});
  std::memcpy(code.data(), buffer.data(), sizeof(buffer));
  std::memcpy(code.data() + 16, image.data(), sizeof(image));
  std::memcpy(code.data() + 32, sample.data(), sizeof(sample));
  process.map_pages(code_va, code.data(), code.size(), amdgpu::Mtype::RW);
  process.map_pages(resident, page.data(), page.size(), amdgpu::Mtype::RW);
  process.prt_residency->nonresident.store(
      std::make_shared<const amdgpu::PrtRanges>(amdgpu::PrtRanges{{hole, unmapped}}));
  auto handle = adapter.register_address_space(
      vmid, {.page_table = &process.page_table_,
             .page_table_mutex = &process.page_table_mutex_,
             .request_mutex = process.page_table_request_mutex(),
             .page_table_cache_state = process.page_table_cache_state(),
             .prt_residency = process.prt_residency});
  ASSERT_TRUE(handle);
  cu_->set_gpu_vm(&vm);
  cache_.set_gpu_vm(&vm);
  wf_->set_address_space(handle);
  wf_->set_process_id(vmid);
  std::vector<uint64_t> faults, watches;
  cu_->set_memory_violation_handler([&](amdgpu::Wavefront &, uint64_t address, bool, uint64_t) {
    faults.push_back(address);
    return true;
  });
  cu_->set_watchpoint_handler(
      [&](amdgpu::Wavefront &, uint64_t address, uint32_t bytes, bool, bool) {
        EXPECT_EQ(bytes, sizeof(texel));
        watches.push_back(address);
        return false;
      });

  const auto step = [&](unsigned pc_offset, uint64_t address, bool debug, bool mixed = false) {
    SCOPED_TRACE(testing::Message() << pc_offset << "/" << address << "/" << debug << "/" << mixed);
    for (unsigned sgpr = 0; sgpr < 12; ++sgpr)
      scalar(sgpr, 0);
    if (pc_offset == 0) {
      scalar(0, address);
      scalar(2, 2 * amdgpu::kLegacyPageSize + 4);
      scalar(3, (3u << 28) | (20u << 12));
    } else {
      bind_single_texel_image(address);
    }
    wf_->pc = code_va + pc_offset;
    wf_->set_exec(mixed ? 3 : 1);
    for (unsigned lane = 0; lane < 2; ++lane) {
      vector(0, lane, mixed && lane == 1 ? amdgpu::kLegacyPageSize : 0);
      vector(1, lane, 0);
      vector(8, lane, sentinel);
      vector(9, lane, 0);
    }
    faults.clear();
    watches.clear();
    cu_->set_debug_active(debug);
    ASSERT_TRUE(cu_->step());
    if (address == unmapped) {
      EXPECT_EQ(faults, std::vector<uint64_t>{unmapped});
      EXPECT_TRUE(watches.empty());
      EXPECT_EQ(vector(8, 0), sentinel);
      EXPECT_EQ(vector(9, 0), 0u);
    } else {
      EXPECT_TRUE(faults.empty());
      EXPECT_EQ(vector(8, 0), address == hole ? 0u : texel);
      EXPECT_EQ(vector(9, 0), address == hole ? 1u : 0u);
      if (debug && address == resident) {
        EXPECT_FALSE(watches.empty());
        EXPECT_TRUE(std::ranges::all_of(watches, [&](auto addr) { return addr == resident; }));
      } else {
        EXPECT_TRUE(watches.empty());
      }
      if (mixed) {
        EXPECT_EQ(vector(8, 1), 0u);
        EXPECT_EQ(vector(9, 1), 1u);
      }
    }
  };
  for (auto pc_offset : {0u, 16u, 32u}) {
    ASSERT_NO_FATAL_FAILURE(step(pc_offset, hole, false));
    ASSERT_NO_FATAL_FAILURE(step(pc_offset, resident, true));
    ASSERT_NO_FATAL_FAILURE(step(pc_offset, hole, true));
    ASSERT_NO_FATAL_FAILURE(step(pc_offset, unmapped, true));
  }
  ASSERT_NO_FATAL_FAILURE(step(0, resident, true, true));
  cu_->set_debug_active(false);
  cu_->set_watchpoint_handler({});
  cu_->set_memory_violation_handler({});
  wf_->set_address_space({});
  cu_->set_gpu_vm(nullptr);
  cache_.set_gpu_vm(nullptr);
}

TEST_F(Rdna1Memory, DualDsHonorsIndependentBoundsAndAliasing) {
  vector(0, 0, 508);
  vector(1, 0, 0x12345678);
  vector(2, 0, 0xabcdef01);
  wf_->lds().write32(1024, 0xcccccccc);
  ASSERT_NO_FATAL_FAILURE(issue(
      rdna1::build_ds(rdna1::kDsWrite2B32, {.offset1 = 1, .addr = 0, .data0 = 1, .data1 = 2})));
  EXPECT_EQ(wf_->lds().read32(1020), 0x12345678u);
  EXPECT_EQ(wf_->lds().read32(1024), 0xccccccccu);
  ASSERT_NO_FATAL_FAILURE(issue(
      rdna1::build_ds(rdna1::kDsWrxchg2RtnB32, {.addr = 0, .data0 = 2, .data1 = 1, .vdst = 4})));
  EXPECT_EQ(wf_->lds().read32(1020), 0xabcdef01u);
  EXPECT_EQ(vector(4, 0), 0x12345678u);
  EXPECT_EQ(vector(5, 0), 0x12345678u);
}

TEST_F(Rdna1Memory, ScratchBoundsApplyPerLaneAndFlatOffsetIsUnsigned) {
  wf_->set_exec(3);
  wf_->set_scratch_lane_size(16);
  wf_->set_scratch_base(0x10000);
  vector(0, 0, 16);
  vector(0, 1, 12);
  memory_.write32(0x10400, 0xfeedface);
  memory_.write32(0x10304, 0x12345678);
  ASSERT_NO_FATAL_FAILURE(
      issue(rdna1::build_flat(rdna1::kFlatLoadDword, {.seg = 1, .saddr = 125, .vdst = 8})));
  EXPECT_EQ(vector(8, 0), 0u);
  EXPECT_EQ(vector(8, 1), 0x12345678u);
  wf_->set_exec(1);
  vector(0, 0, 0x4000);
  vector(1, 0, 0);
  ASSERT_NO_FATAL_FAILURE(execute(
      rdna1::build_flat(rdna1::kFlatLoadDword, {.offset = 0x800, .saddr = 125, .vdst = 8})));
  EXPECT_EQ(inst_->data_as<amdgpu::VectorMemState>()->per_lane_addr[0], 0x4000u);
}

TEST_F(Rdna1Memory, Ptr32IsCapturedAtLaunchAndReadsOneAddressRegister) {
  wf_->halt();
  cu_->set_apertures(0x10000, 0x1ffff, 0x20000, 0x2ffff, /*ptr32=*/true);
  wf_ = cu_->dispatch_wf(0, 0, 106, 16, 64);
  ASSERT_TRUE(wf_);
  wf_->set_exec(7);
  wf_->set_lds_size(512);
  wf_->set_scratch_base(0x40000);
  wf_->set_scratch_lane_size(16);
  constexpr uint16_t sh_mem_bases = 15 | (31 << 11);
  uint32_t bases = 0;
  ASSERT_EQ(amdgpu::read_hwreg_field(*wf_, sh_mem_bases, bases),
            amdgpu::HwregAccessResult::Success);
  EXPECT_EQ(bases, 0x10002u);
  ASSERT_EQ(amdgpu::write_hwreg_field(*wf_, sh_mem_bases, 0x60005),
            amdgpu::HwregAccessResult::Success);
  vector(15, 0, 0x60004);
  vector(15, 1, 0x50008);
  vector(15, 2, 0x30008);
  wf_->lds().write32(4, 17);
  memory_.write32(0x40000 + 2 * 64 * 4 + 4, 23);
  memory_.write32(0x30008, 31);
  const auto load = rdna1::build_flat(rdna1::kFlatLoadDword, {.addr = 15, .saddr = 125, .vdst = 8});
  ASSERT_NO_FATAL_FAILURE(execute(load));
  const auto &state = *inst_->data_as<amdgpu::VectorMemState>();
  EXPECT_EQ(state.scratch_lane_mask, 2u);
  EXPECT_EQ(state.per_lane_addr[0], 0x60004u);
  EXPECT_EQ(state.per_lane_addr[1], 0x40204u);
  EXPECT_EQ(state.per_lane_addr[2], 0x30008u);
  memory_.write32(wf_->pc, load[0]);
  memory_.write32(wf_->pc + sizeof(uint32_t), load[1]);
  ASSERT_TRUE(cu_->step());
  EXPECT_EQ(vector(8, 0), 17u);
  EXPECT_EQ(vector(8, 1), 23u);
  EXPECT_EQ(vector(8, 2), 31u);
}

TEST_P(Gfx10MemoryAllocation, StatusVcczFollowsVccAcrossBufferLoads) {
  constexpr uint32_t address = 0x4000, value = 0x12345678;
  scalar(0, address);
  scalar(1, 0);
  scalar(2, sizeof(value));
  scalar(3, (3u << 28) | (20u << 12));
  memory_.write32(address, value);
  const auto load = rdna1::build_mubuf(rdna1::kBufferLoadDword, {.vdata = 8, .soffset = 128});
  const auto status =
      rdna1::build_sopk(rdna1::kSGetregB32, {.simm16 = 2u | (10u << 6), .sdst = 12});
  const auto expect_vccz = [&](bool zero) {
    ASSERT_NO_FATAL_FAILURE(execute({status[0], 0}));
    EXPECT_EQ(cu_->read_sgpr(wf_->sgpr_alloc().base + 12), unsigned(zero));
  };
  for (const uint64_t vcc : {1u, 0u}) {
    wf_->set_vcc_raw(vcc);
    ASSERT_NO_FATAL_FAILURE(expect_vccz(vcc == 0));
    vector(8, 0, 0);
    ASSERT_NO_FATAL_FAILURE(issue(load));
    EXPECT_EQ(vector(8, 0), value);
    ASSERT_NO_FATAL_FAILURE(expect_vccz(vcc == 0));
  }
}

TEST_F(Rdna1Memory, Wave64PermutesWithinEachPhysicalHalf) {
  wf_->set_exec(~uint64_t{0});
  for (unsigned lane = 0; lane < 64; ++lane) {
    vector(0, lane, 0);
    vector(1, lane, 100 + lane);
  }
  ASSERT_NO_FATAL_FAILURE(
      execute(rdna1::build_ds(rdna1::kDsBpermuteB32, {.addr = 0, .data0 = 1, .vdst = 2})));
  for (unsigned lane = 0; lane < 64; ++lane)
    EXPECT_EQ(vector(2, lane), lane < 32 ? 100u : 132u);
}

TEST_P(Gfx10MemoryAllocation, ConditionalExchangeAlignsToTheGdsWindowWithNonzeroLdsBase) {
  ASSERT_EQ(wf_->lds_base(), 512u);
  wf_->set_m0((32u << 16) | 16u);
  cu_->gws_device().access_gds(arch(), 0, [](amdgpu::Lds &gds) { gds.write32(32, 0x11); });
  vector(0, 0, 0);
  vector(1, 0, 0x80000005);
  vector(2, 0, 0);
  ASSERT_NO_FATAL_FAILURE(issue(
      rdna1::build_ds(rdna1::kDsCondxchg32RtnB64, {.gds = 1, .addr = 0, .data0 = 1, .vdst = 4})));
  EXPECT_EQ(vector(4, 0), 0x11u);
  cu_->gws_device().access_gds(arch(), 0, [](amdgpu::Lds &gds) { EXPECT_EQ(gds.read32(32), 5u); });
}

TEST_P(Gfx10MemoryAllocation, OrderedCountWaitChecksOnlyItsFirstActiveSourceLane) {
  const auto load = rdna1::build_ds(rdna1::kDsReadB32, {.addr = 0, .vdst = 1});
  auto &scoreboard = wf_->ensure_memory_wait_scoreboard();
  for (const auto &[load_exec, count_exec, dst, expected] :
       std::array<std::tuple<uint64_t, uint64_t, uint8_t, unsigned>, 5>{
           {{1, 2, 2, 0}, {2, 6, 2, 1}, {4, 6, 2, 0}, {1, 0, 2, 0}, {2, 2, 16, 0}}}) {
    SCOPED_TRACE(testing::Message() << load_exec << "/" << count_exec << "/" << unsigned(dst));
    scoreboard.clear();
    wf_->set_m0(0);
    wf_->set_exec(load_exec);
    ASSERT_NO_FATAL_FAILURE(execute(load));
    cu_->track_memory_wait(*inst_, *wf_);
    ASSERT_EQ(local_.issue(inst_.release(), *wf_), amdgpu::VmAccessOutcome::Complete);
    wf_->set_exec(count_exec);
    const auto count = rdna1::build_ds(rdna1::kDsOrderedCount, {.gds = 1, .addr = 1, .vdst = dst});
    inst_.reset(decode_valid(*decoder_, count.data()));
    ASSERT_TRUE(inst_);
    unsigned reports = 0;
    scoreboard.bind(0x40, &reports, [](void *context, const auto &hazard) {
      EXPECT_EQ(hazard.reg.cls, RegClass::VGPR);
      EXPECT_EQ(hazard.reg.index, 1u);
      EXPECT_FALSE(hazard.write);
      ++*static_cast<unsigned *>(context);
    });
    scoreboard.check_instruction(*inst_, *wf_);
    EXPECT_EQ(reports, expected);
  }
}

TEST_F(Rdna1Memory, OrderedCountBroadcastsAcrossTheWaveIndependentlyOfExec) {
  const auto counter = [](uint8_t dst, bool release = false) {
    return rdna1::build_ds(
        rdna1::kDsOrderedCount,
        {.offset0 = 4, .offset1 = uint8_t(release), .gds = 1, .addr = 0, .vdst = dst});
  };
  for (uint32_t width : {32u, 64u}) {
    SCOPED_TRACE(width);
    wf_->halt();
    wf_ = cu_->dispatch_wf(0, 0, 106, 16, width);
    ASSERT_TRUE(wf_);
    wf_->set_process_id(width);
    const auto issue_ordered = [&](uint8_t dst, bool release = false) {
      ASSERT_NO_FATAL_FAILURE(execute(counter(dst, release)));
      auto &scoreboard = wf_->ensure_memory_wait_scoreboard();
      scoreboard.clear();
      cu_->track_memory_wait(*inst_, *wf_);
      uint64_t tracked = 0;
      for (const auto &event : scoreboard.events())
        if (event.reg.cls == RegClass::VGPR && event.reg.index == dst)
          tracked |= event.lanes;
      const bool valid = dst < wf_->num_vgprs();
      const uint64_t wave_lanes = width == 64 ? ~uint64_t{0} : uint64_t{0xffffffff};
      EXPECT_EQ(tracked, valid ? wave_lanes : 0);
      // Routed memory observers see the incoming lanes independently of the result.
      const auto *state = inst_->data_as<amdgpu::VectorMemState>();
      EXPECT_EQ(state->exec_mask, valid ? wf_->exec() : 0);
      EXPECT_EQ(local_.issue(inst_.release(), *wf_), amdgpu::VmAccessOutcome::Complete);
    };
    wf_->set_m0(0);
    wf_->set_exec(2);
    for (uint32_t lane = 0; lane < 64; ++lane) {
      vector(2, lane, 0xdeadbeef);
      vector(16, lane, 0xbaadf00d);
    }
    cu_->gws_device().access_gds(cu_->arch(), width, [](amdgpu::Lds &gds) { gds.write32(4, 9); });
    vector(0, 0, 87);
    vector(0, 1, 3);
    ASSERT_NO_FATAL_FAILURE(issue_ordered(2));
    for (uint32_t lane = 0; lane < 64; ++lane)
      EXPECT_EQ(vector(2, lane), lane < width ? 9u : 0xdeadbeefu) << lane;

    // Empty EXEC contributes zero and still broadcasts the pre-operation value.
    wf_->set_exec(0);
    ASSERT_NO_FATAL_FAILURE(issue_ordered(2));
    for (uint32_t lane = 0; lane < width; ++lane)
      EXPECT_EQ(vector(2, lane), 12u) << lane;

    // V16 has physical backing but falls outside this wave's logical allocation.
    // The release must advance the walker without accumulating or writing a result.
    wf_->set_exec(2);
    ASSERT_NO_FATAL_FAILURE(issue_ordered(16, true));
    for (uint32_t lane = 0; lane < 64; ++lane)
      EXPECT_EQ(vector(16, lane), 0xbaadf00du) << lane;
    wf_->set_m0(1);
    ASSERT_NO_FATAL_FAILURE(issue_ordered(2, true));
    EXPECT_EQ(vector(2, width - 1), 12u);
    cu_->gws_device().access_gds(cu_->arch(), width,
                                 [](amdgpu::Lds &gds) { EXPECT_EQ(gds.read32(4), 15u); });
  }
}

TEST_F(Rdna1Memory, OrderedCountRetainsOwnershipUntilReleaseAndRetriesYoungerWaves) {
  auto *older = wf_;
  auto *younger = cu_->dispatch_wf(0, 8, 106, 16, 64);
  ASSERT_TRUE(younger);
  // Supply the hardware launch ordinal in M0[12:0]. The younger request arrives first.
  auto words = rdna1::build_ds(rdna1::kDsOrderedCount,
                               {.offset0 = 4, .offset1 = 3, .gds = 1, .addr = 0, .vdst = 2});
  wf_ = younger;
  younger->set_m0(1);
  younger->set_exec(6);
  vector(0, 1, 7);
  vector(0, 2, 111);
  ASSERT_NO_FATAL_FAILURE(execute(words));
  EXPECT_EQ(local_.issue_deferred(inst_.release(), *younger), amdgpu::VmAccessOutcome::Complete);
  EXPECT_EQ(younger->state(), amdgpu::WfState::VM_RETRY);
  wf_ = older;
  older->set_m0(0);
  vector(0, 0, 3);
  ASSERT_NO_FATAL_FAILURE(issue(
      rdna1::build_ds(rdna1::kDsOrderedCount, {.offset0 = 4, .gds = 1, .addr = 0, .vdst = 2})));
  EXPECT_EQ(vector(2, 0), 0u);
  local_.tick();
  EXPECT_EQ(younger->state(), amdgpu::WfState::VM_RETRY);
  older->set_exec(0); // Release still advances the walker; increment is zero.
  ASSERT_NO_FATAL_FAILURE(issue(words));
  local_.tick();
  EXPECT_EQ(younger->state(), amdgpu::WfState::RUNNING);
  EXPECT_EQ(cu_->read_vgpr(younger->vgpr_alloc().base + 2, 1), 3u);
  EXPECT_EQ(cu_->read_vgpr(younger->vgpr_alloc().base + 2, 2), 3u);
  cu_->gws_device().access_gds(cu_->arch(), 0,
                               [&](amdgpu::Lds &gds) { EXPECT_EQ(gds.read32(4), 10u); });
  younger->halt();
}

TEST_F(Rdna1Memory, AtcProbesWarmOnlyResidentInBoundsScalarLines) {
  constexpr uint32_t address = 0x8000;
  scalar(0, address);
  scalar(1, 0);
  scalar(2, 4);
  amdgpu::ScalarMemPipeline pipeline(&cu_->l1_scalar());
  auto probe = [&](auto words) {
    ASSERT_NO_FATAL_FAILURE(execute(words));
    ASSERT_NE(inst_->data(), nullptr);
    ASSERT_EQ(pipeline.issue(inst_.release(), *wf_), amdgpu::VmAccessOutcome::Complete);
  };
  memory_.write32(address, 42);
  const auto plain = rdna1::build_smem(rdna1::kSAtcProbe, {.soffset = 125});
  ASSERT_NO_FATAL_FAILURE(probe(plain));
  EXPECT_EQ(cache_.backing_read_transactions(), 0u);
  std::array<uint8_t, 64> line{};
  ASSERT_EQ(cache_.read(address, line.data(), line.size(), amdgpu::Mtype::RW),
            amdgpu::VmAccessOutcome::Complete);
  const auto buffered = rdna1::build_smem(rdna1::kSAtcProbeBuffer, {.soffset = 125});
  uint32_t cached = 42, value = 0;
  for (const auto &instruction : {plain, buffered}) {
    cu_->l1_scalar().invalidate_all();
    ASSERT_NO_FATAL_FAILURE(probe(instruction));
    value = cached + 1;
    ASSERT_EQ(cache_.write(address, reinterpret_cast<const uint8_t *>(&value), sizeof(value),
                           amdgpu::Mtype::RW),
              amdgpu::VmAccessOutcome::Complete);
    ASSERT_EQ(cu_->l1_scalar().load(address, 1, &value), amdgpu::VmAccessOutcome::Complete);
    EXPECT_EQ(value, cached);
    ++cached;
  }
  cu_->l1_scalar().invalidate_all();
  constexpr uint32_t offset = amdgpu::L2Cache::LINE_SIZE;
  ASSERT_EQ(cache_.read(address + offset, line.data(), line.size(), amdgpu::Mtype::RW),
            amdgpu::VmAccessOutcome::Complete);
  ASSERT_NO_FATAL_FAILURE(
      probe(rdna1::build_smem(rdna1::kSAtcProbeBuffer, {.offset = offset, .soffset = 125})));
  value = 99;
  ASSERT_EQ(cache_.write(address + offset, reinterpret_cast<const uint8_t *>(&value), sizeof(value),
                         amdgpu::Mtype::RW),
            amdgpu::VmAccessOutcome::Complete);
  ASSERT_EQ(cu_->l1_scalar().load(address + offset, 1, &value), amdgpu::VmAccessOutcome::Complete);
  EXPECT_EQ(value, 99u);
}

TEST_F(Rdna1Memory, GdsIsSharedAcrossComputeUnitsAndIsolatedFromLdsAndOtherProcesses) {
  wf_->halt();
  auto share = std::make_shared<amdgpu::GwsDevice>();
  ASSERT_TRUE(cu_->set_gws_device(share));
  wf_ = cu_->dispatch_wf(0, 0, 106, 16, 64);
  ASSERT_TRUE(wf_);
  wf_->set_exec(1);
  wf_->set_m0((32u << 16) | 8);
  wf_->set_process_id(5);
  vector(0, 0, 4);
  vector(1, 0, 9);
  ASSERT_NO_FATAL_FAILURE(
      issue(rdna1::build_ds(rdna1::kDsWriteB32, {.gds = 1, .addr = 0, .data0 = 1})));
  EXPECT_EQ(wf_->lds().read32(36), 0u);
  wf_->halt();
  wf_ = nullptr;
  ASSERT_NO_FATAL_FAILURE(
      test::initialize_compute_unit(cu_, "rdna1_peer", config_, &memory_, &cache_));
  ASSERT_TRUE(cu_->set_gws_device(share));
  wf_ = cu_->dispatch_wf(0, 0, 106, 16, 64);
  ASSERT_TRUE(wf_);
  wf_->set_exec(1);
  wf_->set_process_id(5);
  wf_->set_m0((32u << 16) | 8);
  vector(0, 0, 4);
  vector(1, 0, 3);
  ASSERT_NO_FATAL_FAILURE(
      issue(rdna1::build_ds(rdna1::kDsAddRtnU32, {.gds = 1, .addr = 0, .data0 = 1, .vdst = 2})));
  EXPECT_EQ(vector(2, 0), 9u);
  const auto read = rdna1::build_ds(rdna1::kDsReadB32, {.gds = 1, .addr = 0, .vdst = 3});
  ASSERT_NO_FATAL_FAILURE(issue(read));
  EXPECT_EQ(vector(3, 0), 12u);
  wf_->set_m0((32u << 16) | 4); // Same physical address, outside the smaller bound.
  ASSERT_NO_FATAL_FAILURE(issue(read));
  EXPECT_EQ(vector(3, 0), 0u);
  wf_->set_m0((32u << 16) | 8);
  wf_->set_process_id(6);
  ASSERT_NO_FATAL_FAILURE(issue(read));
  EXPECT_EQ(vector(3, 0), 0u);
}
} // namespace
} // namespace rocjitsu
