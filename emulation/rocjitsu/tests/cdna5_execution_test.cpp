// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "cdna5_sim_test_common.h"
#include "decode_test_util.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/vbuffer.h"
#include "rocjitsu/isa/arch/amdgpu/shared/gfx12_cache_flags.h"
#include "rocjitsu/vm/plugins/execution_plugin_group.h"

namespace {

using namespace rocjitsu;
using namespace rocjitsu::test::cdna5;

class ForceScalarGuard {
public:
  ForceScalarGuard() : original_(util::force_scalar()) {}
  ~ForceScalarGuard() { util::set_force_scalar_for_testing(original_); }

private:
  bool original_;
};

class VgprReadRecorder final : public ExecutionPlugin {
public:
  VgprReadRecorder() : ExecutionPlugin("vgpr_read_recorder") {}

  void onAmdgpuReadVgprLanes(const amdgpu::Wavefront *, uint32_t, uint64_t, uint8_t) override {
    ++read_count;
  }

  uint32_t read_count = 0;
};

class Gfx1250MemoryTestCu
    : public amdgpu::IsaExecComputeUnit<simdojo::ExecMode::FUNCTIONAL, cdna5::Isa> {
public:
  using Base = amdgpu::IsaExecComputeUnit<simdojo::ExecMode::FUNCTIONAL, cdna5::Isa>;

  Gfx1250MemoryTestCu(std::string name, const amdgpu::ComputeUnitCore::Config &config,
                      amdgpu::GpuMemory *memory, amdgpu::L2Cache *l2)
      : Base(std::move(name), config, memory, l2) {
    l2->set_backing_memory(memory);
    set_memory(memory);
    set_l2(l2);
  }

  void execute_and_route(std::unique_ptr<Instruction> instruction, amdgpu::Wavefront &wave) {
    execute_instruction(instruction.get(), wave);
    if (instruction->is_memory_op())
      route_memory_inst(instruction.release(), wave);
  }
};

TEST(Gfx1250ExecutionTest, TargetProvidesImmutableExecutionBackend) {
  const IsaTargetDescriptor *target = default_isa_target_registry().find("cdna5");
  ASSERT_NE(target, nullptr);
  EXPECT_TRUE(target->supports_execution);
  EXPECT_TRUE(cdna5::Operand::full_execution_backend_complete());
}

TEST(Gfx1250ExecutionTest, SramEccD16LoadsZeroUnselectedHalf) {
  amdgpu::GpuMemory memory("gfx1250_d16_memory");
  amdgpu::L2Cache l2("gfx1250_d16_l2");
  amdgpu::ComputeUnitCore::Config config{};
  config.arch = ROCJITSU_CODE_ARCH_CDNA5;
  config.num_wf_slots = 1;
  config.sgprs_per_wf = kGfx1250ScalarSlots;
  config.vgprs_per_wf = 64;
  config.lds_size_kb = kGfx1250LdsSizeKb;
  auto compute_unit = std::make_unique<Gfx1250MemoryTestCu>("gfx1250_d16_cu", config, &memory, &l2);

  auto *wave = compute_unit->dispatch_wf(0, 0, config.sgprs_per_wf, config.vgprs_per_wf);
  ASSERT_NE(wave, nullptr);
  ASSERT_TRUE(compute_unit->sram_ecc());
  wave->set_exec(1u);

  constexpr uint64_t kAddress = 0x1000;
  memory.write8(kAddress + 1, 0x7eu);
  memory.write8(kAddress + 2, 0x7fu);
  write_wave_sgpr(*compute_unit, *wave, 24, static_cast<uint32_t>(kAddress));
  write_wave_sgpr(*compute_unit, *wave, 25, static_cast<uint32_t>(kAddress >> 32));
  write_wave_sgpr(*compute_unit, *wave, 26, 0x100u);
  write_wave_sgpr(*compute_unit, *wave, 27, 0u);

  const uint32_t vgpr_base = wave->vgpr_alloc().base;
  compute_unit->write_vgpr(vgpr_base + 33, 0, 0u);
  struct LoadCase {
    std::string_view mnemonic;
    std::array<uint32_t, 3> words;
    uint32_t destination;
    uint32_t expected;
  };

  constexpr std::array<LoadCase, 2> kLoads = {{
      {
          "buffer_load_d16_u8",
          {0xc407807cu, 0x40803038u, 0x00000121u},
          56,
          0x0000007eu,
      },
      {
          "buffer_load_d16_hi_u8",
          {0xc408407cu, 0x40803039u, 0x00000221u},
          57,
          0x007f0000u,
      },
  }};

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  for (const auto &test : kLoads) {
    SCOPED_TRACE(test.mnemonic);
    compute_unit->write_vgpr(vgpr_base + test.destination, 0, 0xdeadbeefu);
    std::unique_ptr<Instruction> load(decode_valid(*decoder, test.words.data()));
    ASSERT_NE(load, nullptr);
    ASSERT_EQ(std::string_view(load->mnemonic()), test.mnemonic);

    compute_unit->execute_and_route(std::move(load), *wave);

    EXPECT_EQ(compute_unit->read_vgpr(vgpr_base + test.destination, 0), test.expected);
  }
}

TEST(Gfx1250ExecutionTest, ScalarMovesTreatS102AndS103AsOrdinarySgprs) {
  Gfx1250Sim sim;
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);

  constexpr uint64_t kScratchBase = 0xabcde00012345000ull;
  wf->set_scratch_base(kScratchBase);
  write_wave_sgpr(*cu, *wf, 90, 0x11223344u);
  write_wave_sgpr(*cu, *wf, 91, 0x55667788u);
  write_wave_sgpr(*cu, *wf, 92, 0xaabbccddu);
  write_wave_sgpr(*cu, *wf, 93, 0xeeff0011u);
  write_wave_sgpr(*cu, *wf, 102, 0);
  write_wave_sgpr(*cu, *wf, 103, 0);
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);

  constexpr std::array<uint32_t, 3> kMove64 = {
      0xbee6015au, // s_mov_b64 s[102:103], s[90:91]
      0,
      0,
  };
  std::unique_ptr<Instruction> move64(decode_valid(*decoder, kMove64.data()));
  ASSERT_NE(move64, nullptr);
  EXPECT_EQ(move64->mnemonic(), "s_mov_b64");
  EXPECT_EQ(move64->size(), 4);
  move64->execute(*move64, wf);
  EXPECT_EQ(read_wave_sgpr(*cu, *wf, 102), 0x11223344u);
  EXPECT_EQ(read_wave_sgpr(*cu, *wf, 103), 0x55667788u);
  EXPECT_EQ(wf->scratch_base(), kScratchBase);

  constexpr std::array<uint32_t, 3> kWriteS102 = {
      0xbee6005cu, // s_mov_b32 s102, s92
      0,
      0,
  };
  constexpr std::array<uint32_t, 3> kWriteS103 = {
      0xbee7005du, // s_mov_b32 s103, s93
      0,
      0,
  };
  for (const auto *words : {kWriteS102.data(), kWriteS103.data()}) {
    std::unique_ptr<Instruction> move32(decode_valid(*decoder, words));
    ASSERT_NE(move32, nullptr);
    EXPECT_EQ(move32->mnemonic(), "s_mov_b32");
    EXPECT_EQ(move32->size(), 4);
    move32->execute(*move32, wf);
  }
  EXPECT_EQ(read_wave_sgpr(*cu, *wf, 102), 0xaabbccddu);
  EXPECT_EQ(read_wave_sgpr(*cu, *wf, 103), 0xeeff0011u);

  constexpr std::array<uint32_t, 3> kReadS102 = {
      0xbed20066u, // s_mov_b32 s82, s102
      0,
      0,
  };
  constexpr std::array<uint32_t, 3> kReadS103 = {
      0xbed30067u, // s_mov_b32 s83, s103
      0,
      0,
  };
  for (const auto *words : {kReadS102.data(), kReadS103.data()}) {
    std::unique_ptr<Instruction> move32(decode_valid(*decoder, words));
    ASSERT_NE(move32, nullptr);
    EXPECT_EQ(move32->mnemonic(), "s_mov_b32");
    EXPECT_EQ(move32->size(), 4);
    move32->execute(*move32, wf);
  }
  EXPECT_EQ(read_wave_sgpr(*cu, *wf, 82), 0xaabbccddu);
  EXPECT_EQ(read_wave_sgpr(*cu, *wf, 83), 0xeeff0011u);

  constexpr std::array<uint32_t, 3> kReadS102S103 = {
      0xbed00166u, // s_mov_b64 s[80:81], s[102:103]
      0,
      0,
  };
  std::unique_ptr<Instruction> read64(decode_valid(*decoder, kReadS102S103.data()));
  ASSERT_NE(read64, nullptr);
  EXPECT_EQ(read64->mnemonic(), "s_mov_b64");
  EXPECT_EQ(read64->size(), 4);
  read64->execute(*read64, wf);
  EXPECT_EQ(read_wave_sgpr(*cu, *wf, 80), 0xaabbccddu);
  EXPECT_EQ(read_wave_sgpr(*cu, *wf, 81), 0xeeff0011u);
  EXPECT_EQ(wf->scratch_base(), kScratchBase);
}

TEST(Gfx1250ExecutionTest, Wave32VectorComparePreservesVccHiScratch) {
  ForceScalarGuard force_scalar_guard;
  for (const bool force_scalar : {false, true}) {
    SCOPED_TRACE(force_scalar ? "scalar" : "simd");
    util::set_force_scalar_for_testing(force_scalar);

    Gfx1250Sim sim;
    auto *cu = sim.cu();
    auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
    ASSERT_NE(wf, nullptr);
    wf->set_exec(0x3u);
    wf->set_vcc(0x000001c0ffffffffull);
    write_wave_sgpr(*cu, *wf, 28, 7u);
    const uint32_t vgpr_base = wf->vgpr_alloc().base;
    cu->write_vgpr(vgpr_base + 18, 0, 6u);
    cu->write_vgpr(vgpr_base + 18, 1, 8u);

    auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
    ASSERT_NE(decoder, nullptr);
    constexpr auto kCompare = cdna5::build_vopc(cdna5::kVCmpGtI32Vopc, {.src0 = 28, .vsrc1 = 18});
    std::unique_ptr<Instruction> decoded(decode_valid(*decoder, kCompare.data()));
    ASSERT_NE(decoded, nullptr);
    EXPECT_EQ(decoded->mnemonic(), "v_cmp_gt_i32_e32");
    decoded->execute(*decoded, wf);
    EXPECT_EQ(wf->vcc(), 0x000001c000000001ull);
  }
}

TEST(Gfx1250ExecutionTest, VbufferB128LoadsZeroAndStoresDropPartialOobDwords) {
  Gfx1250Sim sim;
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(1u);

  constexpr uint64_t kAddr = 0xC000;
  constexpr uint32_t kResourceSgpr = 4;
  constexpr uint32_t kLoadVgpr = 8;
  constexpr uint32_t kStoreVgpr = 12;
  constexpr std::array<uint32_t, 4> kInitial = {0x101u, 0x102u, 0x103u, 0x104u};
  constexpr std::array<uint32_t, 4> kStored = {0x201u, 0x202u, 0x203u, 0x204u};
  for (uint32_t i = 0; i < kInitial.size(); ++i)
    sim.memory->write32(kAddr + i * sizeof(uint32_t), kInitial[i]);

  // gfx1250 NUM_RECORDS is a 45-bit byte count split across SRD words 1-3.
  constexpr uint64_t kNumRecords = 10;
  const std::array<uint32_t, 4> resource = {
      static_cast<uint32_t>(kAddr),
      static_cast<uint32_t>((kAddr >> 32) & 0x01FF'FFFFu) |
          static_cast<uint32_t>((kNumRecords & 0x7Fu) << 25),
      static_cast<uint32_t>(kNumRecords >> 7),
      static_cast<uint32_t>((kNumRecords >> 39) & 0x3Fu),
  };
  for (uint32_t i = 0; i < resource.size(); ++i)
    cu->write_sgpr(wf->sgpr_alloc().base + kResourceSgpr + i, resource[i]);

  cdna5::VbufferMachineInst machine{};
  machine.vdata = kLoadVgpr;
  machine.rsrc = kResourceSgpr;
  machine.soffset = cdna5::OPR_SREG_NULL;
  machine.scope = 3; // System scope selects uncached accesses for direct memory checks.

  amdgpu::GlobalMemPipeline pipeline(&cu->l1_vector(), cu->l2());
  auto *load =
      new cdna5::BufferLoadB128Vbuffer(reinterpret_cast<const cdna5::MachineInst *>(&machine));
  load->execute_impl(*wf);
  pipeline.issue(load, *wf);

  EXPECT_EQ(cu->read_vgpr(wf->vgpr_alloc().base + kLoadVgpr + 0, 0), kInitial[0]);
  EXPECT_EQ(cu->read_vgpr(wf->vgpr_alloc().base + kLoadVgpr + 1, 0), kInitial[1]);
  EXPECT_EQ(cu->read_vgpr(wf->vgpr_alloc().base + kLoadVgpr + 2, 0), 0u);
  EXPECT_EQ(cu->read_vgpr(wf->vgpr_alloc().base + kLoadVgpr + 3, 0), 0u);

  for (uint32_t i = 0; i < kStored.size(); ++i)
    cu->write_vgpr(wf->vgpr_alloc().base + kStoreVgpr + i, 0, kStored[i]);
  machine.vdata = kStoreVgpr;
  auto *store =
      new cdna5::BufferStoreB128Vbuffer(reinterpret_cast<const cdna5::MachineInst *>(&machine));
  store->execute_impl(*wf);
  pipeline.issue(store, *wf);

  EXPECT_EQ(sim.memory->read32(kAddr), kStored[0]);
  EXPECT_EQ(sim.memory->read32(kAddr + 4), kStored[1]);
  EXPECT_EQ(sim.memory->read32(kAddr + 8), kInitial[2]);
  EXPECT_EQ(sim.memory->read32(kAddr + 12), kInitial[3]);
}

TEST(Gfx1250ExecutionTest, VbufferB64B96MixedLanesHonorStructuredPartialOob) {
  Gfx1250Sim sim;
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(0x3u);

  constexpr uint64_t kAddr = 0xC080;
  constexpr uint32_t kResourceSgpr = 4;
  constexpr uint32_t kSoffsetSgpr = 12;
  constexpr uint32_t kAddressVgpr = 4;
  constexpr uint32_t kB96LoadVgpr = 8;
  constexpr uint32_t kB64LoadVgpr = 12;
  constexpr uint32_t kStoreVgpr = 16;
  constexpr uint64_t kNumRecords = 32;
  constexpr uint32_t kRawStride = 4;
  constexpr uint32_t kStrideScaleEncoding = 1;
  constexpr uint32_t kStrideMultiplier = 4;
  constexpr uint32_t kStride = kRawStride * kStrideMultiplier;
  const std::array<uint32_t, 4> resource = {
      static_cast<uint32_t>(kAddr),
      static_cast<uint32_t>((kAddr >> 32) & 0x01FF'FFFFu) |
          static_cast<uint32_t>((kNumRecords & 0x7Fu) << 25),
      static_cast<uint32_t>(kNumRecords >> 7),
      static_cast<uint32_t>((kNumRecords >> 39) & 0x3Fu) | (kRawStride << 12) |
          (kStrideScaleEncoding << 26) | (1u << 29),
  };
  for (uint32_t i = 0; i < resource.size(); ++i)
    cu->write_sgpr(wf->sgpr_alloc().base + kResourceSgpr + i, resource[i]);
  cu->write_sgpr(wf->sgpr_alloc().base + kSoffsetSgpr, 4);

  const uint32_t vbase = wf->vgpr_alloc().base;
  cu->write_vgpr(vbase + kAddressVgpr, 0, 0);
  cu->write_vgpr(vbase + kAddressVgpr + 1, 0, 0);
  cu->write_vgpr(vbase + kAddressVgpr, 1, 1);
  cu->write_vgpr(vbase + kAddressVgpr + 1, 1, 8);

  constexpr std::array<uint32_t, 3> kLane0Initial = {0x101u, 0x102u, 0x103u};
  constexpr std::array<uint32_t, 3> kLane1Initial = {0x201u, 0x202u, 0x203u};
  for (uint32_t i = 0; i < kLane0Initial.size(); ++i)
    sim.memory->write32(kAddr + 4 + i * sizeof(uint32_t), kLane0Initial[i]);
  for (uint32_t i = 0; i < kLane1Initial.size(); ++i)
    sim.memory->write32(kAddr + 28 + i * sizeof(uint32_t), kLane1Initial[i]);

  cdna5::VbufferMachineInst machine{};
  machine.rsrc = kResourceSgpr;
  machine.soffset = kSoffsetSgpr;
  machine.vaddr = kAddressVgpr;
  machine.idxen = 1;
  machine.offen = 1;
  machine.scope = 3;
  amdgpu::GlobalMemPipeline pipeline(&cu->l1_vector(), cu->l2());

  machine.vdata = kB64LoadVgpr;
  auto *b64_load =
      new cdna5::BufferLoadB64Vbuffer(reinterpret_cast<const cdna5::MachineInst *>(&machine));
  b64_load->execute_impl(*wf);
  pipeline.issue(b64_load, *wf);
  EXPECT_EQ(cu->read_vgpr(vbase + kB64LoadVgpr, 0), kLane0Initial[0]);
  EXPECT_EQ(cu->read_vgpr(vbase + kB64LoadVgpr + 1, 0), kLane0Initial[1]);
  EXPECT_EQ(cu->read_vgpr(vbase + kB64LoadVgpr, 1), kLane1Initial[0]);
  EXPECT_EQ(cu->read_vgpr(vbase + kB64LoadVgpr + 1, 1), 0u);

  machine.vdata = kB96LoadVgpr;
  auto *b96_load =
      new cdna5::BufferLoadB96Vbuffer(reinterpret_cast<const cdna5::MachineInst *>(&machine));
  b96_load->execute_impl(*wf);
  pipeline.issue(b96_load, *wf);
  for (uint32_t i = 0; i < kLane0Initial.size(); ++i)
    EXPECT_EQ(cu->read_vgpr(vbase + kB96LoadVgpr + i, 0), kLane0Initial[i]);
  EXPECT_EQ(cu->read_vgpr(vbase + kB96LoadVgpr, 1), kLane1Initial[0]);
  EXPECT_EQ(cu->read_vgpr(vbase + kB96LoadVgpr + 1, 1), 0u);
  EXPECT_EQ(cu->read_vgpr(vbase + kB96LoadVgpr + 2, 1), 0u);

  constexpr std::array<uint32_t, 3> kLane0Stored = {0x301u, 0x302u, 0x303u};
  constexpr std::array<uint32_t, 3> kLane1Stored = {0x401u, 0x402u, 0x403u};
  for (uint32_t i = 0; i < kLane0Stored.size(); ++i) {
    cu->write_vgpr(vbase + kStoreVgpr + i, 0, kLane0Stored[i]);
    cu->write_vgpr(vbase + kStoreVgpr + i, 1, kLane1Stored[i]);
  }
  machine.vdata = kStoreVgpr;
  auto *b96_store =
      new cdna5::BufferStoreB96Vbuffer(reinterpret_cast<const cdna5::MachineInst *>(&machine));
  b96_store->execute_impl(*wf);
  pipeline.issue(b96_store, *wf);

  for (uint32_t i = 0; i < kLane0Stored.size(); ++i)
    EXPECT_EQ(sim.memory->read32(kAddr + 4 + i * sizeof(uint32_t)), kLane0Stored[i]);
  EXPECT_EQ(sim.memory->read32(kAddr + kStride + 12), kLane1Stored[0]);
  EXPECT_EQ(sim.memory->read32(kAddr + kStride + 16), kLane1Initial[1]);
  EXPECT_EQ(sim.memory->read32(kAddr + kStride + 20), kLane1Initial[2]);
}

TEST(Gfx1250ExecutionTest, NonReturningVbufferB64AtomicHonorsWholePayloadBounds) {
  Gfx1250Sim sim;
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(1u);

  constexpr uint64_t kAddr = 0xC100;
  constexpr uint32_t kResourceSgpr = 4;
  constexpr uint32_t kDataVgpr = 8;
  constexpr uint64_t kInitial = 0x0102'0304'0506'0708ULL;
  constexpr uint64_t kAddend = 0x1011'1213'1415'1617ULL;
  auto write_resource = [&](uint64_t num_records) {
    const std::array<uint32_t, 4> resource = {
        static_cast<uint32_t>(kAddr),
        static_cast<uint32_t>((kAddr >> 32) & 0x01FF'FFFFu) |
            static_cast<uint32_t>((num_records & 0x7Fu) << 25),
        static_cast<uint32_t>(num_records >> 7),
        static_cast<uint32_t>((num_records >> 39) & 0x3Fu),
    };
    for (uint32_t i = 0; i < resource.size(); ++i)
      cu->write_sgpr(wf->sgpr_alloc().base + kResourceSgpr + i, resource[i]);
  };

  cu->write_vgpr(wf->vgpr_alloc().base + kDataVgpr, 0, static_cast<uint32_t>(kAddend));
  cu->write_vgpr(wf->vgpr_alloc().base + kDataVgpr + 1, 0, static_cast<uint32_t>(kAddend >> 32));
  cdna5::VbufferMachineInst machine{};
  machine.vdata = kDataVgpr;
  machine.rsrc = kResourceSgpr;
  machine.soffset = cdna5::OPR_SREG_NULL;
  machine.scope = 3;
  machine.th = 0;
  amdgpu::GlobalMemPipeline pipeline(&cu->l1_vector(), cu->l2());

  sim.memory->write64(kAddr, kInitial);
  write_resource(/*num_records=*/8);
  auto *exact_end =
      new cdna5::BufferAtomicAddU64Vbuffer(reinterpret_cast<const cdna5::MachineInst *>(&machine));
  exact_end->execute_impl(*wf);
  pipeline.issue(exact_end, *wf);
  EXPECT_EQ(sim.memory->read64(kAddr), kInitial + kAddend);

  sim.memory->write64(kAddr, kInitial);
  write_resource(/*num_records=*/6);
  auto *partial_oob =
      new cdna5::BufferAtomicAddU64Vbuffer(reinterpret_cast<const cdna5::MachineInst *>(&machine));
  partial_oob->execute_impl(*wf);
  pipeline.issue(partial_oob, *wf);
  EXPECT_EQ(sim.memory->read64(kAddr), kInitial);
}

TEST(Gfx1250ExecutionTest, ReturningVbufferAtomicIgnoresNonBufferResourceType) {
  Gfx1250Sim sim;
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(1u);

  constexpr uint64_t kAddr = 0xC180;
  constexpr uint32_t kResourceSgpr = 4;
  constexpr uint32_t kDataVgpr = 8;
  constexpr uint32_t kNonBufferType = 2;
  constexpr uint32_t kInitialMemory = 0x0102'0304u;
  constexpr uint32_t kDataSentinel = 0xA1A2'A3A4u;
  constexpr uint64_t kNumRecords = sizeof(uint32_t);
  const std::array<uint32_t, 4> resource = {
      static_cast<uint32_t>(kAddr),
      static_cast<uint32_t>((kAddr >> 32) & 0x01FF'FFFFu) |
          static_cast<uint32_t>((kNumRecords & 0x7Fu) << 25),
      static_cast<uint32_t>(kNumRecords >> 7),
      static_cast<uint32_t>((kNumRecords >> 39) & 0x3Fu) | (kNonBufferType << 30),
  };
  for (uint32_t i = 0; i < resource.size(); ++i)
    cu->write_sgpr(wf->sgpr_alloc().base + kResourceSgpr + i, resource[i]);
  sim.memory->write32(kAddr, kInitialMemory);
  cu->write_vgpr(wf->vgpr_alloc().base + kDataVgpr, 0, kDataSentinel);

  auto plugin_group = std::make_shared<ExecutionPluginGroup>(PluginSinkConfig{});
  auto recorder = std::make_unique<VgprReadRecorder>();
  auto *recorder_ptr = recorder.get();
  ASSERT_TRUE(plugin_group->add(std::move(recorder)));
  cu->set_plugin_group(plugin_group);
  plugin_group->onInit();

  cdna5::VbufferMachineInst machine{};
  machine.vdata = kDataVgpr;
  machine.rsrc = kResourceSgpr;
  machine.soffset = cdna5::OPR_SREG_NULL;
  machine.scope = 3;
  machine.th = amdgpu::GFX12_TH_ATOMIC_RETURN;

  amdgpu::GlobalMemPipeline pipeline(&cu->l1_vector(), cu->l2());
  auto *atomic =
      new cdna5::BufferAtomicSwapB32Vbuffer(reinterpret_cast<const cdna5::MachineInst *>(&machine));
  atomic->execute_impl(*wf);
  pipeline.issue(atomic, *wf);

  EXPECT_EQ(recorder_ptr->read_count, 0u);
  EXPECT_EQ(sim.memory->read32(kAddr), kInitialMemory);
  EXPECT_EQ(cu->read_vgpr(wf->vgpr_alloc().base + kDataVgpr, 0), kDataSentinel);
  EXPECT_TRUE(wf->wait_counters().empty());
}

TEST(Gfx1250ExecutionTest, DivScaleWritesExplicitSdstMask) {
  Gfx1250Sim sim;
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(1u);

  constexpr uint32_t kLane = 0;
  constexpr uint32_t kOne = 0x3f800000u;
  constexpr uint32_t kTwoTo8 = 0x43800000u;
  constexpr uint32_t kTwoTo100 = 0x71800000u;
  const uint32_t vgpr_base = wf->vgpr_alloc().base;
  auto write_vgpr = [&](uint32_t reg, uint32_t value) {
    cu->write_vgpr(vgpr_base + reg, kLane, value);
  };
  auto read_vgpr = [&](uint32_t reg) { return cu->read_vgpr(vgpr_base + reg, kLane); };
  auto write_sgpr = [&](uint32_t reg, uint32_t value) {
    cu->write_sgpr(wf->sgpr_alloc().base + reg, value);
  };

  write_vgpr(1, kOne);
  write_vgpr(2, kTwoTo100);
  wf->set_vcc(0x5a5a5a5au);
  const std::array<uint32_t, 2> null_sdst_words = {
      0xd6fc7c00u, 0x040a0301u}; // v_div_scale_f32 v0, null, v1, v1, v2
  cdna5::VDivScaleF32Vop3SdstEnc null_sdst(null_sdst_words.data());
  null_sdst.execute_impl(*wf);
  EXPECT_EQ(wf->vcc(), 0x5a5a5a5au);
  EXPECT_EQ(read_vgpr(0), 0x5f800000u); // 2^64

  write_sgpr(7, kTwoTo8);
  write_vgpr(3, kOne);
  wf->set_vcc(0xa5a5a5a5u);
  const std::array<uint32_t, 2> normal_null_sdst_words = {
      0xd6fc7c09u, 0x040c0e07u}; // v_div_scale_f32 v9, null, s7, s7, v3
  cdna5::VDivScaleF32Vop3SdstEnc normal_null_sdst(normal_null_sdst_words.data());
  normal_null_sdst.execute_impl(*wf);
  EXPECT_EQ(wf->vcc(), 0xa5a5a5a5u);
  EXPECT_EQ(read_vgpr(9), kTwoTo8);

  write_vgpr(4, kOne);
  write_vgpr(5, kTwoTo100);
  wf->set_vcc(0);
  const std::array<uint32_t, 2> vcc_sdst_words = {
      0xd6fc6a03u, 0x04160904u}; // v_div_scale_f32 v3, vcc_lo, v4, v4, v5
  cdna5::VDivScaleF32Vop3SdstEnc vcc_sdst(vcc_sdst_words.data());
  vcc_sdst.execute_impl(*wf);
  EXPECT_EQ(wf->vcc(), 1u);
  EXPECT_EQ(read_vgpr(3), 0x5f800000u);

  write_vgpr(7, kOne);
  write_vgpr(8, kTwoTo100);
  write_sgpr(3, 0xfefefefeu);
  wf->set_vcc(0x12345678u);
  const std::array<uint32_t, 2> sgpr_sdst_words = {
      0xd6fc0206u, 0x04220f07u}; // v_div_scale_f32 v6, s2, v7, v7, v8
  cdna5::VDivScaleF32Vop3SdstEnc sgpr_sdst(sgpr_sdst_words.data());
  sgpr_sdst.execute_impl(*wf);
  EXPECT_EQ(wf->vcc(), 0x12345678u);
  EXPECT_EQ(read_wave_sgpr(*cu, *wf, 2), 0x12345679u);
  EXPECT_EQ(read_wave_sgpr(*cu, *wf, 3), 0xfefefefeu);
  EXPECT_EQ(read_vgpr(6), 0x5f800000u);
}

TEST(Gfx1250ExecutionTest, VMovB16HighVdstMergesIntoLowPhysicalVgpr) {
  Gfx1250Sim sim;
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, kGfx1250Wave32VgprAllocation);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(1u);

  constexpr uint32_t kLane = 0;
  const uint32_t vgpr_base = wf->vgpr_alloc().base;
  cu->write_vgpr(vgpr_base + 1, kLane, 0xAAAA5555u);
  cu->write_vgpr(vgpr_base + 129, kLane, 0xDEADBEEFu);

  const std::array<uint32_t, 1> words = {0x7F023880u}; // v_mov_b16_e32 v1.h, 0
  cdna5::VMovB16Vop1 high_half_mov(words.data());
  high_half_mov.execute_impl(*wf);

  EXPECT_EQ(cu->read_vgpr(vgpr_base + 1, kLane), 0x00005555u);
  EXPECT_EQ(cu->read_vgpr(vgpr_base + 129, kLane), 0xDEADBEEFu);
}

TEST(Gfx1250ExecutionTest, VNotB16HighVdstMergesIntoLowPhysicalVgpr) {
  Gfx1250Sim sim;
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, kGfx1250Wave32VgprAllocation);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(1u);

  constexpr uint32_t kLane = 0;
  const uint32_t vgpr_base = wf->vgpr_alloc().base;
  cu->write_vgpr(vgpr_base + 0, kLane, 0x000000FFu);
  cu->write_vgpr(vgpr_base + 1, kLane, 0xAAAA5555u);
  cu->write_vgpr(vgpr_base + 129, kLane, 0xDEADBEEFu);

  const std::array<uint32_t, 1> words = {0x7F02D300u}; // v_not_b16_e32 v1.h, v0.l
  cdna5::VNotB16Vop1 high_half_not(words.data());
  high_half_not.execute_impl(*wf);

  EXPECT_EQ(cu->read_vgpr(vgpr_base + 1, kLane), 0xFF005555u);
  EXPECT_EQ(cu->read_vgpr(vgpr_base + 129, kLane), 0xDEADBEEFu);
}

TEST(Gfx1250ExecutionTest, VAddF16HighVdstMergesIntoLowPhysicalVgpr) {
  Gfx1250Sim sim;
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, kGfx1250Wave32VgprAllocation);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(1u);

  constexpr uint32_t kLane = 0;
  const uint32_t vgpr_base = wf->vgpr_alloc().base;
  cu->write_vgpr(vgpr_base + 0, kLane, 0x00003C00u);
  cu->write_vgpr(vgpr_base + 1, kLane, 0xAAAA5555u);
  cu->write_vgpr(vgpr_base + 2, kLane, 0x00003C00u);
  cu->write_vgpr(vgpr_base + 129, kLane, 0xDEADBEEFu);

  const std::array<uint32_t, 1> words = {0x65020500u}; // v_add_f16_e32 v1.h, v0.l, v2.l
  cdna5::VAddF16Vop2 high_half_add(words.data());
  high_half_add.execute_impl(*wf);

  EXPECT_EQ(cu->read_vgpr(vgpr_base + 1, kLane), 0x40005555u);
  EXPECT_EQ(cu->read_vgpr(vgpr_base + 129, kLane), 0xDEADBEEFu);
}

TEST(Gfx1250ExecutionTest, IreeF16ReductionTailKeepsLane31Sum) {
  Gfx1250Sim sim;
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, kGfx1250Wave32VgprAllocation);
  ASSERT_NE(wf, nullptr);
  ASSERT_EQ(wf->wf_size(), 32u);
  wf->set_exec(0xffffffffu);

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);

  const uint32_t vgpr_base = wf->vgpr_alloc().base;
  const uint32_t packed_1_2 = 0x40003c00u;
  const uint32_t packed_3_4 = 0x44004200u;
  const uint32_t packed_5_6 = 0x46004500u;
  const uint32_t packed_7_8 = 0x48004700u;
  for (uint32_t lane = 0; lane < wf->wf_size(); ++lane) {
    cu->write_vgpr(vgpr_base + 10, lane, packed_7_8);
    cu->write_vgpr(vgpr_base + 11, lane, packed_1_2);
    cu->write_vgpr(vgpr_base + 16, lane, packed_5_6);
    cu->write_vgpr(vgpr_base + 17, lane, packed_3_4);
  }

  const std::array<std::array<uint32_t, 3>, 20> words = {{
      {0x64021680u, 0, 0},                     // v_add_f16_e32 v1, 0, v11
      {0x32041690u, 0, 0},                     // v_lshrrev_b32_e32 v2, 16, v11
      {0x64020501u, 0, 0},                     // v_add_f16_e32 v1, v1, v2
      {0x32042290u, 0, 0},                     // v_lshrrev_b32_e32 v2, 16, v17
      {0x64022301u, 0, 0},                     // v_add_f16_e32 v1, v1, v17
      {0x64020501u, 0, 0},                     // v_add_f16_e32 v1, v1, v2
      {0x32042090u, 0, 0},                     // v_lshrrev_b32_e32 v2, 16, v16
      {0x64022101u, 0, 0},                     // v_add_f16_e32 v1, v1, v16
      {0x64020501u, 0, 0},                     // v_add_f16_e32 v1, v1, v2
      {0x32041490u, 0, 0},                     // v_lshrrev_b32_e32 v2, 16, v10
      {0x64021501u, 0, 0},                     // v_add_f16_e32 v1, v1, v10
      {0x64020501u, 0, 0},                     // v_add_f16_e32 v1, v1, v2
      {0xd5320001u, 0x000202fau, 0xff08b101u}, // quad_perm:[1,0,3,2]
      {0xd5320001u, 0x000202fau, 0xff084e01u}, // quad_perm:[2,3,0,1]
      {0xd5320001u, 0x000202fau, 0xff094101u}, // row_half_mirror
      {0xd5320001u, 0x000202fau, 0xff094001u}, // row_mirror
      {0xd65c0802u, 0x03058301u, 0},           // v_permlanex16_b32 v2, v1, -1, -1
      {0x64020302u, 0, 0},                     // v_add_f16_e32 v1, v2, v1
      {0xd7600000u, 0x02013f01u, 0},           // v_readlane_b32 s0, v1, 31
      {0xa4808000u, 0, 0},                     // s_add_f16 s0, s0, 0
  }};

  for (const auto &inst_words : words) {
    std::unique_ptr<Instruction> inst(decode_valid(*decoder, inst_words.data()));
    ASSERT_NE(inst, nullptr);
    cu->execute_instruction(inst.get(), *wf);
  }

  EXPECT_EQ(read_wave_sgpr(*cu, *wf, 0) & 0xffffu, 0x6480u);
}

TEST(Gfx1250ExecutionTest, VFmacF16Vop3HighVdstUsesHighHalfAddend) {
  Gfx1250Sim sim;
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, kGfx1250Wave32VgprAllocation);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(1u);

  constexpr uint32_t kLane = 0;
  const uint32_t vgpr_base = wf->vgpr_alloc().base;
  cu->write_vgpr(vgpr_base + 0, kLane, 0x00003C00u);
  cu->write_vgpr(vgpr_base + 1, kLane, 0x40003C00u);
  cu->write_vgpr(vgpr_base + 2, kLane, 0x00003C00u);

  const std::array<uint32_t, 2> words = {
      0xD5364001u, // v_fmac_f16 v1.h, v0.l, v2.l
      0x02020500u,
  };
  cdna5::VFmacF16Vop3 high_half_fmac(words.data());
  high_half_fmac.execute_impl(*wf);

  EXPECT_EQ(cu->read_vgpr(vgpr_base + 1, kLane), 0x42003C00u);
}

TEST(Gfx1250ExecutionTest, VFmacF16Vop2HighVdstUsesHighHalfAddend) {
  Gfx1250Sim sim;
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, kGfx1250Wave32VgprAllocation);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(1u);

  constexpr uint32_t kLane = 0;
  const uint32_t vgpr_base = wf->vgpr_alloc().base;
  cu->write_vgpr(vgpr_base + 0, kLane, 0x00003C00u);
  cu->write_vgpr(vgpr_base + 1, kLane, 0x40003C00u);
  cu->write_vgpr(vgpr_base + 2, kLane, 0x00003C00u);
  cu->write_vgpr(vgpr_base + 129, kLane, 0x3C003C00u);

  const std::array<uint32_t, 1> words = {0x6D020500u}; // v_fmac_f16_e32 v1.h, v0.l, v2.l
  cdna5::VFmacF16Vop2 high_half_fmac(words.data());
  high_half_fmac.execute_impl(*wf);

  EXPECT_EQ(cu->read_vgpr(vgpr_base + 1, kLane), 0x42003C00u);
  EXPECT_EQ(cu->read_vgpr(vgpr_base + 129, kLane), 0x3C003C00u);
}

TEST(Gfx1250ExecutionTest, VMadU32LiteralTimesScalarAddsVector) {
  Gfx1250Sim sim;
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(1u);

  constexpr uint32_t kLane = 0;
  const uint32_t vgpr_base = wf->vgpr_alloc().base;
  write_wave_sgpr(*cu, *wf, 3, 1);
  cu->write_vgpr(vgpr_base + 4, kLane, 0x24u);

  const std::array<uint32_t, 3> words = {
      0xD6350004u, // v_mad_u32 v4, 0x48, s3, v4
      0x041006FFu,
      0x00000048u,
  };
  cdna5::VMadU32Vop3 mad(words.data());
  mad.execute_impl(*wf);

  EXPECT_EQ(cu->read_vgpr(vgpr_base + 4, kLane), 0x6Cu);
}

TEST(Gfx1250LiteralOperandTest, SplitBackendPreservesSignedAndEncodingSemantics) {
  Gfx1250Sim sim;
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(1u);

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  ASSERT_TRUE(cdna5::Operand::full_execution_backend_complete());
  amdgpu::RegisterAccess regs(*wf);

  struct LiteralCase {
    uint32_t encoded;
    uint64_t signed_value;
  };
  constexpr std::array cases{
      LiteralCase{0x7fffffffu, 0x000000007fffffffULL},
      LiteralCase{0x80000000u, 0xffffffff80000000ULL},
      LiteralCase{0xffffffffu, 0xffffffffffffffffULL},
  };

  for (const auto &[literal, signed_value] : cases) {
    SCOPED_TRACE(::testing::Message() << "literal=" << literal);

    const auto signed_mad_base = cdna5::build_vop3(
        cdna5::kVMadNcI64I32Vop3, {.vdst = 4, .src0 = 129, .src1 = 129, .src2 = 255});
    const std::array signed_mad_words{signed_mad_base[0], signed_mad_base[1], literal};
    std::unique_ptr<Instruction> signed_mad_decoded(
        decode_valid(*decoder, signed_mad_words.data()));
    ASSERT_NE(signed_mad_decoded, nullptr);
    EXPECT_EQ(signed_mad_decoded->mnemonic(), "v_mad_nc_i64_i32");
    EXPECT_EQ(signed_mad_decoded->size(), 12);
    auto *signed_mad = dynamic_cast<cdna5::VMadNcI64I32Vop3 *>(signed_mad_decoded.get());
    ASSERT_NE(signed_mad, nullptr);

    const Operand *signed_addend = signed_mad->src_operand(2);
    ASSERT_NE(signed_addend, nullptr);
    EXPECT_EQ(signed_addend->name(), std::format("0x{:x}", literal));
    EXPECT_EQ(static_cast<uint32_t>(signed_addend->encoding_value()), literal);
    EXPECT_FALSE(signed_addend->literal64_value().has_value());
    EXPECT_EQ(regs.read_lane64(*signed_addend, 0), signed_value);
    signed_mad->execute_impl(*wf);
    EXPECT_EQ(regs.read_lane64(*signed_mad->dst_operand(0), 0), signed_value + 1u);

    const auto unsigned_mad_base = cdna5::build_vop3(
        cdna5::kVMadNcU64U32Vop3, {.vdst = 6, .src0 = 129, .src1 = 129, .src2 = 255});
    const std::array unsigned_mad_words{unsigned_mad_base[0], unsigned_mad_base[1], literal};
    std::unique_ptr<Instruction> unsigned_mad_decoded(
        decode_valid(*decoder, unsigned_mad_words.data()));
    ASSERT_NE(unsigned_mad_decoded, nullptr);
    auto *unsigned_mad = dynamic_cast<cdna5::VMadNcU64U32Vop3 *>(unsigned_mad_decoded.get());
    ASSERT_NE(unsigned_mad, nullptr);

    const Operand *unsigned_addend = unsigned_mad->src_operand(2);
    ASSERT_NE(unsigned_addend, nullptr);
    EXPECT_FALSE(unsigned_addend->literal64_value().has_value());
    EXPECT_EQ(regs.read_lane64(*unsigned_addend, 0), static_cast<uint64_t>(literal));
    unsigned_mad->execute_impl(*wf);
    EXPECT_EQ(regs.read_lane64(*unsigned_mad->dst_operand(0), 0),
              static_cast<uint64_t>(literal) + 1u);

    const auto scalar_base =
        cdna5::build_sop2(cdna5::kSAshrI64Sop2, {.ssrc0 = 255, .ssrc1 = 128, .sdst = 0});
    const std::array scalar_words{scalar_base[0], literal};
    std::unique_ptr<Instruction> scalar_decoded(decode_valid(*decoder, scalar_words.data()));
    ASSERT_NE(scalar_decoded, nullptr);
    EXPECT_EQ(scalar_decoded->mnemonic(), "s_ashr_i64");
    EXPECT_EQ(scalar_decoded->size(), 8);
    auto *scalar = dynamic_cast<cdna5::SAshrI64Sop2 *>(scalar_decoded.get());
    ASSERT_NE(scalar, nullptr);

    const Operand *scalar_value = scalar->src_operand(0);
    ASSERT_NE(scalar_value, nullptr);
    EXPECT_FALSE(scalar_value->literal64_value().has_value());
    EXPECT_EQ(regs.read_scalar64(*scalar_value), signed_value);
    scalar->execute_impl(*wf);
    EXPECT_EQ(regs.read_scalar64(*scalar->dst_operand(0)), signed_value);

    const auto b64_base =
        cdna5::build_sop2(cdna5::kSAndB64Sop2, {.ssrc0 = 255, .ssrc1 = 193, .sdst = 2});
    const std::array b64_words{b64_base[0], literal};
    std::unique_ptr<Instruction> b64_decoded(decode_valid(*decoder, b64_words.data()));
    ASSERT_NE(b64_decoded, nullptr);
    auto *b64 = dynamic_cast<cdna5::SAndB64Sop2 *>(b64_decoded.get());
    ASSERT_NE(b64, nullptr);

    const Operand *b64_value = b64->src_operand(0);
    ASSERT_NE(b64_value, nullptr);
    EXPECT_FALSE(b64_value->literal64_value().has_value());
    EXPECT_EQ(regs.read_scalar64(*b64_value), static_cast<uint64_t>(literal));
    b64->execute_impl(*wf);
    EXPECT_EQ(regs.read_scalar64(*b64->dst_operand(0)), static_cast<uint64_t>(literal));
  }
}

TEST(Gfx1250LiteralOperandTest, NegativeI64CompareCoversScalarAndAvailableSimdPath) {
  ForceScalarGuard force_scalar_guard;
  const auto run_case = [](bool force_scalar) {
    SCOPED_TRACE(force_scalar ? "scalar" : "simd");
    util::set_force_scalar_for_testing(force_scalar);

    Gfx1250Sim sim;
    auto *cu = sim.cu();
    auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
    ASSERT_NE(wf, nullptr);
    wf->set_exec(0x3u);

    const uint32_t vgpr_base = wf->vgpr_alloc().base;
    for (uint32_t lane = 0; lane < 2; ++lane) {
      cu->write_vgpr(vgpr_base, lane, 0u);
      cu->write_vgpr(vgpr_base + 1, lane, 0u);
    }

    auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
    ASSERT_NE(decoder, nullptr);
    const auto compare_base =
        cdna5::build_vop3(cdna5::kVCmpLtI64Vop3, {.vdst = 0, .src0 = 255, .src1 = 256});
    const std::array compare_words{compare_base[0], compare_base[1], 0xffffffffu};
    std::unique_ptr<Instruction> compare(decode_valid(*decoder, compare_words.data()));
    ASSERT_NE(compare, nullptr);
    EXPECT_EQ(compare->mnemonic(), "v_cmp_lt_i64");
    EXPECT_EQ(compare->size(), 12);

    auto *typed_compare = dynamic_cast<cdna5::VCmpLtI64Vop3 *>(compare.get());
    ASSERT_NE(typed_compare, nullptr);
    if (!force_scalar) {
      EXPECT_TRUE(amdgpu::try_execute_vopc64_vop3_int_simd<int64_t>(
          *typed_compare, *wf, [](auto a, auto b) { return a < b; }));
      EXPECT_EQ(read_wave_sgpr(*cu, *wf, 0), 0x3u);
      write_wave_sgpr(*cu, *wf, 0, 0u);
      write_wave_sgpr(*cu, *wf, 1, 0u);
    }

    cu->execute_instruction(compare.get(), *wf);
    EXPECT_EQ(read_wave_sgpr(*cu, *wf, 0), 0x3u);
  };

  run_case(true);
  if constexpr (util::has_stdx_simd && !UTIL_SIMD_BROKEN_NATIVE_64BIT_MASKS)
    run_case(false);
}

TEST(Gfx1250LiteralOperandTest, ScalarMaskOperandsRejectLiteralMarkers) {
  struct TestCase {
    const char *name;
    std::array<uint32_t, 2> encoding;
  };
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  for (const uint16_t marker : {uint16_t{254}, uint16_t{255}}) {
    const std::array test_cases{
        TestCase{"v_cndmask_b32", cdna5::build_vop3(cdna5::kVCndmaskB32Vop3,
                                                    {.src0 = 128, .src1 = 128, .src2 = marker})},
        TestCase{"v_cndmask_b16", cdna5::build_vop3(cdna5::kVCndmaskB16Vop3,
                                                    {.src0 = 128, .src1 = 128, .src2 = marker})},
        TestCase{"v_add_co_ci_u32",
                 cdna5::build_vop3_sdst_enc(cdna5::kVAddCoCiU32Vop3SdstEnc,
                                            {.src0 = 128, .src1 = 128, .src2 = marker})},
        TestCase{"v_sub_co_ci_u32",
                 cdna5::build_vop3_sdst_enc(cdna5::kVSubCoCiU32Vop3SdstEnc,
                                            {.src0 = 128, .src1 = 128, .src2 = marker})},
        TestCase{"v_subrev_co_ci_u32",
                 cdna5::build_vop3_sdst_enc(cdna5::kVSubrevCoCiU32Vop3SdstEnc,
                                            {.src0 = 128, .src1 = 128, .src2 = marker})},
    };
    for (const TestCase &test_case : test_cases) {
      SCOPED_TRACE(test_case.name);
      SCOPED_TRACE(marker);
      const std::array words{test_case.encoding[0], test_case.encoding[1], 0xffffffffu, 0u};
      EXPECT_TRUE(decode_fails(*decoder, words.data()));
    }
  }
}

TEST(Gfx1250LiteralOperandTest, PkF32LiteralReplicatesAndUsesAvailableSimdPath) {
  ForceScalarGuard force_scalar_guard;
  const auto run_case = [](bool force_scalar) {
    SCOPED_TRACE(force_scalar ? "scalar" : "simd");
    util::set_force_scalar_for_testing(force_scalar);

    constexpr uint32_t literal = 0x3f800000u;
    constexpr uint64_t replicated =
        (static_cast<uint64_t>(literal) << 32) | static_cast<uint64_t>(literal);
    const auto add_base = cdna5::build_vop3p(cdna5::kVPkAddF32Vop3p,
                                             {.vdst = 0, .src0 = 255, .src1 = 128, .opsel_hi = 3});
    const std::array add_words{add_base[0], add_base[1], literal};

    auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
    ASSERT_NE(decoder, nullptr);
    std::unique_ptr<Instruction> add(decode_valid(*decoder, add_words.data()));
    ASSERT_NE(add, nullptr);
    auto *typed_add = dynamic_cast<cdna5::VPkAddF32Vop3p *>(add.get());
    ASSERT_NE(typed_add, nullptr);

    Gfx1250Sim sim;
    auto *cu = sim.cu();
    auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
    ASSERT_NE(wf, nullptr);
    wf->set_exec(0x3u);

    const Operand *literal_operand = typed_add->src_operand(0);
    ASSERT_NE(literal_operand, nullptr);
    EXPECT_EQ(static_cast<uint32_t>(literal_operand->encoding_value()), literal);
    EXPECT_FALSE(literal_operand->literal64_value().has_value());
    EXPECT_EQ(amdgpu::RegisterAccess(*wf).read_lane64(*literal_operand, 0), replicated);

    const uint32_t vgpr_base = wf->vgpr_alloc().base;
    if (!force_scalar) {
      EXPECT_TRUE(amdgpu::try_execute_vop3p_pk_binary_f32_simd(
          *typed_add, *wf, 0u, 3u, [](auto a, auto b) { return a + b; }));
      for (uint32_t lane = 0; lane < 2; ++lane) {
        EXPECT_EQ(cu->read_vgpr(vgpr_base, lane), literal);
        EXPECT_EQ(cu->read_vgpr(vgpr_base + 1, lane), literal);
        cu->write_vgpr(vgpr_base, lane, 0u);
        cu->write_vgpr(vgpr_base + 1, lane, 0u);
      }
    }

    cu->execute_instruction(add.get(), *wf);
    for (uint32_t lane = 0; lane < 2; ++lane) {
      EXPECT_EQ(cu->read_vgpr(vgpr_base, lane), literal);
      EXPECT_EQ(cu->read_vgpr(vgpr_base + 1, lane), literal);
    }
  };

  run_case(true);
  if constexpr (util::has_stdx_simd)
    run_case(false);
}

TEST(Gfx1250LiteralOperandTest, PkF32MixedLiteralVgprSourcesUseAvailableSimdPath) {
  if constexpr (!util::has_stdx_simd)
    GTEST_SKIP() << "requires stdx SIMD";

  ForceScalarGuard force_scalar_guard;
  util::set_force_scalar_for_testing(false);
  enum class Operation { Add, Mul, Fma };
  struct TestCase {
    Operation operation;
    uint16_t opcode;
    uint32_t literal_source;
    float expected_lo;
    float expected_hi;
  };
  constexpr std::array test_cases{
      TestCase{Operation::Add, cdna5::kVPkAddF32Vop3p, 0, 7.0f, 8.0f},
      TestCase{Operation::Add, cdna5::kVPkAddF32Vop3p, 1, 5.0f, 6.0f},
      TestCase{Operation::Mul, cdna5::kVPkMulF32Vop3p, 0, 10.0f, 12.0f},
      TestCase{Operation::Mul, cdna5::kVPkMulF32Vop3p, 1, 6.0f, 8.0f},
      TestCase{Operation::Fma, cdna5::kVPkFmaF32Vop3p, 0, 17.0f, 20.0f},
      TestCase{Operation::Fma, cdna5::kVPkFmaF32Vop3p, 1, 13.0f, 16.0f},
      TestCase{Operation::Fma, cdna5::kVPkFmaF32Vop3p, 2, 17.0f, 26.0f},
  };
  constexpr uint32_t kLiteral = 0x40000000u; // 2.0f
  constexpr uint64_t kReplicatedLiteral = (static_cast<uint64_t>(kLiteral) << 32) | kLiteral;

  for (const TestCase &test_case : test_cases) {
    SCOPED_TRACE(static_cast<uint32_t>(test_case.operation));
    SCOPED_TRACE(test_case.literal_source);
    cdna5::Vop3pBuilderFields fields;
    fields.vdst = 6;
    fields.src0 = 256;
    fields.src1 = 258;
    fields.src2 = 260;
    fields.opsel_hi = 3;
    if (test_case.literal_source == 0)
      fields.src0 = 255;
    else if (test_case.literal_source == 1)
      fields.src1 = 255;
    else
      fields.src2 = 255;

    auto base = cdna5::build_vop3p(test_case.opcode, fields);
    if (test_case.operation == Operation::Fma)
      base[0] |= uint32_t{1} << 14; // pad_14 is the src2 high-half selector.
    const std::array words{base[0], base[1], kLiteral};
    auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
    ASSERT_NE(decoder, nullptr);
    std::unique_ptr<Instruction> instruction(decode_valid(*decoder, words.data()));
    ASSERT_NE(instruction, nullptr);

    const Operand *literal_operand = instruction->src_operand(test_case.literal_source);
    ASSERT_NE(literal_operand, nullptr);

    Gfx1250Sim sim;
    auto *cu = sim.cu();
    auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
    ASSERT_NE(wf, nullptr);
    wf->set_exec(1u);
    EXPECT_EQ(amdgpu::RegisterAccess(*wf).read_lane64(*literal_operand, 0), kReplicatedLiteral);
    const uint32_t vgpr_base = wf->vgpr_alloc().base;
    cu->write_vgpr(vgpr_base, 0, std::bit_cast<uint32_t>(3.0f));
    cu->write_vgpr(vgpr_base + 1, 0, std::bit_cast<uint32_t>(4.0f));
    cu->write_vgpr(vgpr_base + 2, 0, std::bit_cast<uint32_t>(5.0f));
    cu->write_vgpr(vgpr_base + 3, 0, std::bit_cast<uint32_t>(6.0f));
    cu->write_vgpr(vgpr_base + 4, 0, std::bit_cast<uint32_t>(7.0f));
    cu->write_vgpr(vgpr_base + 5, 0, std::bit_cast<uint32_t>(8.0f));

    bool accepted = false;
    if (test_case.operation == Operation::Add) {
      auto *typed = dynamic_cast<cdna5::VPkAddF32Vop3p *>(instruction.get());
      ASSERT_NE(typed, nullptr);
      accepted = amdgpu::try_execute_vop3p_pk_binary_f32_simd(*typed, *wf, 0u, 3u,
                                                              [](auto a, auto b) { return a + b; });
    } else if (test_case.operation == Operation::Mul) {
      auto *typed = dynamic_cast<cdna5::VPkMulF32Vop3p *>(instruction.get());
      ASSERT_NE(typed, nullptr);
      accepted = amdgpu::try_execute_vop3p_pk_binary_f32_simd(*typed, *wf, 0u, 3u,
                                                              [](auto a, auto b) { return a * b; });
    } else {
      auto *typed = dynamic_cast<cdna5::VPkFmaF32Vop3p *>(instruction.get());
      ASSERT_NE(typed, nullptr);
      accepted = amdgpu::try_execute_vop3p_pk_ternary_f32_simd(
          *typed, *wf, 0u, 3u, 1u, [](auto a, auto b, auto c) { return util::stdx::fma(a, b, c); });
    }
    EXPECT_TRUE(accepted);
    EXPECT_EQ(cu->read_vgpr(vgpr_base + 6, 0), std::bit_cast<uint32_t>(test_case.expected_lo));
    EXPECT_EQ(cu->read_vgpr(vgpr_base + 7, 0), std::bit_cast<uint32_t>(test_case.expected_hi));

    cu->write_vgpr(vgpr_base + 6, 0, 0u);
    cu->write_vgpr(vgpr_base + 7, 0, 0u);
    cu->execute_instruction(instruction.get(), *wf);
    EXPECT_EQ(cu->read_vgpr(vgpr_base + 6, 0), std::bit_cast<uint32_t>(test_case.expected_lo));
    EXPECT_EQ(cu->read_vgpr(vgpr_base + 7, 0), std::bit_cast<uint32_t>(test_case.expected_hi));
  }
}

TEST(Gfx1250LiteralOperandTest, PkF32MixedLiteralSourceSpecificSelectorFallsBackToScalar) {
  ForceScalarGuard force_scalar_guard;
  util::set_force_scalar_for_testing(false);
  constexpr uint32_t kLiteral = 0x40000000u; // 2.0f
  const auto base = cdna5::build_vop3p(cdna5::kVPkAddF32Vop3p,
                                       {.vdst = 4, .src0 = 255, .src1 = 258, .opsel_hi = 2});
  const std::array words{base[0], base[1], kLiteral};
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> instruction(decode_valid(*decoder, words.data()));
  ASSERT_NE(instruction, nullptr);
  auto *typed = dynamic_cast<cdna5::VPkAddF32Vop3p *>(instruction.get());
  ASSERT_NE(typed, nullptr);

  Gfx1250Sim sim;
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(1u);
  const uint32_t vgpr_base = wf->vgpr_alloc().base;
  cu->write_vgpr(vgpr_base + 2, 0, std::bit_cast<uint32_t>(5.0f));
  cu->write_vgpr(vgpr_base + 3, 0, std::bit_cast<uint32_t>(6.0f));

  EXPECT_FALSE(amdgpu::try_execute_vop3p_pk_binary_f32_simd(*typed, *wf, 0u, 2u,
                                                            [](auto a, auto b) { return a + b; }));
  cu->execute_instruction(instruction.get(), *wf);
  EXPECT_EQ(cu->read_vgpr(vgpr_base + 4, 0), std::bit_cast<uint32_t>(7.0f));
  EXPECT_EQ(cu->read_vgpr(vgpr_base + 5, 0), std::bit_cast<uint32_t>(8.0f));
}

TEST(Gfx1250ExecutionTest, PkF32AddMulSimdMatchesScalarWithPartialExec) {
  ForceScalarGuard force_scalar_guard;
  struct TestCase {
    uint16_t opcode;
    const char *name;
  };
  constexpr std::array test_cases{
      TestCase{cdna5::kVPkAddF32Vop3p, "add"},
      TestCase{cdna5::kVPkMulF32Vop3p, "mul"},
  };
  constexpr uint32_t kExec = 0xa5a5a5a5u;
  constexpr uint32_t kDstLoSeed = 0xdeadbeefu;
  constexpr uint32_t kDstHiSeed = 0xbaadf00du;

  for (const TestCase &test_case : test_cases) {
    SCOPED_TRACE(test_case.name);
    std::array<uint32_t, 64> scalar_result{};
    std::array<uint32_t, 64> simd_result{};
    const auto run_case = [&](bool force_scalar, std::array<uint32_t, 64> &result) {
      SCOPED_TRACE(force_scalar ? "scalar" : "simd");
      util::set_force_scalar_for_testing(force_scalar);

      const auto words = cdna5::build_vop3p(
          test_case.opcode,
          {.vdst = 4, .neg_hi = 2, .src0 = 256, .src1 = 258, .opsel_hi = 3, .neg = 1});
      auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
      ASSERT_NE(decoder, nullptr);
      std::unique_ptr<Instruction> instruction(decode_valid(*decoder, words.data()));
      ASSERT_NE(instruction, nullptr);

      Gfx1250Sim sim;
      auto *cu = sim.cu();
      auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
      ASSERT_NE(wf, nullptr);
      wf->set_exec(kExec);
      const uint32_t vgpr_base = wf->vgpr_alloc().base;
      for (uint32_t lane = 0; lane < 32; ++lane) {
        const float lane_value = static_cast<float>(lane + 1);
        cu->write_vgpr(vgpr_base, lane, std::bit_cast<uint32_t>(lane_value * 0.25f));
        cu->write_vgpr(vgpr_base + 1, lane, std::bit_cast<uint32_t>(lane_value * -0.5f));
        cu->write_vgpr(vgpr_base + 2, lane, std::bit_cast<uint32_t>(lane_value + 0.75f));
        cu->write_vgpr(vgpr_base + 3, lane, std::bit_cast<uint32_t>(lane_value * 1.5f));
        cu->write_vgpr(vgpr_base + 4, lane, kDstLoSeed);
        cu->write_vgpr(vgpr_base + 5, lane, kDstHiSeed);
      }

      if (!force_scalar) {
        bool accepted = false;
        if (test_case.opcode == cdna5::kVPkAddF32Vop3p) {
          auto *typed = dynamic_cast<cdna5::VPkAddF32Vop3p *>(instruction.get());
          ASSERT_NE(typed, nullptr);
          accepted = amdgpu::try_execute_vop3p_pk_binary_f32_simd(
              *typed, *wf, 0u, 3u, [](auto a, auto b) { return a + b; });
        } else {
          auto *typed = dynamic_cast<cdna5::VPkMulF32Vop3p *>(instruction.get());
          ASSERT_NE(typed, nullptr);
          accepted = amdgpu::try_execute_vop3p_pk_binary_f32_simd(
              *typed, *wf, 0u, 3u, [](auto a, auto b) { return a * b; });
        }
        EXPECT_TRUE(accepted);
        for (uint32_t lane = 0; lane < 32; ++lane) {
          cu->write_vgpr(vgpr_base + 4, lane, kDstLoSeed);
          cu->write_vgpr(vgpr_base + 5, lane, kDstHiSeed);
        }
      }

      cu->execute_instruction(instruction.get(), *wf);
      for (uint32_t lane = 0; lane < 32; ++lane) {
        result[lane * 2] = cu->read_vgpr(vgpr_base + 4, lane);
        result[lane * 2 + 1] = cu->read_vgpr(vgpr_base + 5, lane);
        if ((kExec & (1u << lane)) == 0u) {
          EXPECT_EQ(result[lane * 2], kDstLoSeed);
          EXPECT_EQ(result[lane * 2 + 1], kDstHiSeed);
        } else {
          const float lane_value = static_cast<float>(lane + 1);
          const float a_lo = lane_value * 0.25f;
          const float a_hi = lane_value * -0.5f;
          const float b_lo = lane_value + 0.75f;
          const float b_hi = lane_value * 1.5f;
          const float expected_lo =
              test_case.opcode == cdna5::kVPkAddF32Vop3p ? -a_lo + b_lo : -a_lo * b_lo;
          const float expected_hi =
              test_case.opcode == cdna5::kVPkAddF32Vop3p ? a_hi - b_hi : a_hi * -b_hi;
          EXPECT_EQ(result[lane * 2], std::bit_cast<uint32_t>(expected_lo));
          EXPECT_EQ(result[lane * 2 + 1], std::bit_cast<uint32_t>(expected_hi));
        }
      }
    };

    run_case(true, scalar_result);
    if constexpr (util::has_stdx_simd) {
      run_case(false, simd_result);
      EXPECT_EQ(simd_result, scalar_result);
    }
  }
}

TEST(Gfx1250ExecutionTest, PkF32EveryNondefaultSelectorGateFallsBackToScalar) {
  ForceScalarGuard force_scalar_guard;
  struct TestCase {
    const char *name;
    bool ternary;
    uint32_t op_sel;
    uint32_t op_sel_hi;
    uint32_t op_sel_hi_2;
  };
  constexpr std::array test_cases{
      TestCase{"binary-opsel", false, 1u, 3u, 0u},
      TestCase{"binary-opsel-hi", false, 0u, 2u, 0u},
      TestCase{"ternary-opsel", true, 1u, 3u, 1u},
      TestCase{"ternary-opsel-hi", true, 0u, 2u, 1u},
      TestCase{"ternary-opsel-hi-2", true, 0u, 3u, 0u},
  };
  constexpr uint32_t kExec = 0x5a5a5a5au;
  constexpr uint32_t kDstLoSeed = 0xdeadbeefu;
  constexpr uint32_t kDstHiSeed = 0xbaadf00du;

  for (const TestCase &test_case : test_cases) {
    SCOPED_TRACE(test_case.name);
    std::array<uint32_t, 64> scalar_result{};
    std::array<uint32_t, 64> fallback_result{};
    const auto run_case = [&](bool force_scalar, std::array<uint32_t, 64> &result) {
      util::set_force_scalar_for_testing(force_scalar);
      cdna5::Vop3pBuilderFields fields;
      fields.vdst = 6;
      fields.opsel = static_cast<uint8_t>(test_case.op_sel);
      fields.src0 = 256;
      fields.src1 = 258;
      fields.src2 = 260;
      fields.opsel_hi = static_cast<uint8_t>(test_case.op_sel_hi);
      auto words = cdna5::build_vop3p(
          test_case.ternary ? cdna5::kVPkFmaF32Vop3p : cdna5::kVPkAddF32Vop3p, fields);
      if (test_case.op_sel_hi_2 != 0u)
        words[0] |= uint32_t{1} << 14;
      auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
      ASSERT_NE(decoder, nullptr);
      std::unique_ptr<Instruction> instruction(decode_valid(*decoder, words.data()));
      ASSERT_NE(instruction, nullptr);

      Gfx1250Sim sim;
      auto *cu = sim.cu();
      auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
      ASSERT_NE(wf, nullptr);
      wf->set_exec(kExec);
      const uint32_t vgpr_base = wf->vgpr_alloc().base;
      for (uint32_t lane = 0; lane < 32; ++lane) {
        const float value = static_cast<float>(lane + 1);
        cu->write_vgpr(vgpr_base, lane, std::bit_cast<uint32_t>(value));
        cu->write_vgpr(vgpr_base + 1, lane, std::bit_cast<uint32_t>(value + 0.5f));
        cu->write_vgpr(vgpr_base + 2, lane, std::bit_cast<uint32_t>(value * 2.0f));
        cu->write_vgpr(vgpr_base + 3, lane, std::bit_cast<uint32_t>(value * -3.0f));
        cu->write_vgpr(vgpr_base + 4, lane, std::bit_cast<uint32_t>(value + 4.0f));
        cu->write_vgpr(vgpr_base + 5, lane, std::bit_cast<uint32_t>(value * 0.25f));
        cu->write_vgpr(vgpr_base + 6, lane, kDstLoSeed);
        cu->write_vgpr(vgpr_base + 7, lane, kDstHiSeed);
      }

      if (!force_scalar) {
        if (test_case.ternary) {
          auto *typed = dynamic_cast<cdna5::VPkFmaF32Vop3p *>(instruction.get());
          ASSERT_NE(typed, nullptr);
          EXPECT_FALSE(amdgpu::try_execute_vop3p_pk_ternary_f32_simd(
              *typed, *wf, test_case.op_sel, test_case.op_sel_hi, test_case.op_sel_hi_2,
              [](auto a, auto b, auto c) { return util::stdx::fma(a, b, c); }));
        } else {
          auto *typed = dynamic_cast<cdna5::VPkAddF32Vop3p *>(instruction.get());
          ASSERT_NE(typed, nullptr);
          EXPECT_FALSE(amdgpu::try_execute_vop3p_pk_binary_f32_simd(
              *typed, *wf, test_case.op_sel, test_case.op_sel_hi,
              [](auto a, auto b) { return a + b; }));
        }
      }
      cu->execute_instruction(instruction.get(), *wf);
      for (uint32_t lane = 0; lane < 32; ++lane) {
        result[lane * 2] = cu->read_vgpr(vgpr_base + 6, lane);
        result[lane * 2 + 1] = cu->read_vgpr(vgpr_base + 7, lane);
        if ((kExec & (1u << lane)) == 0u) {
          EXPECT_EQ(result[lane * 2], kDstLoSeed);
          EXPECT_EQ(result[lane * 2 + 1], kDstHiSeed);
        }
      }
    };

    run_case(true, scalar_result);
    if constexpr (util::has_stdx_simd) {
      run_case(false, fallback_result);
      EXPECT_EQ(fallback_result, scalar_result);
    }
  }
}

TEST(Gfx1250ExecutionTest, PkFmaF32SimdMatchesScalarWithPartialExec) {
  ForceScalarGuard force_scalar_guard;
  constexpr uint32_t kExec = 0xc3c3c3c3u;
  constexpr uint32_t kDstLoSeed = 0xdeadbeefu;
  constexpr uint32_t kDstHiSeed = 0xbaadf00du;
  std::array<uint32_t, 64> scalar_result{};
  std::array<uint32_t, 64> simd_result{};

  const auto run_case = [&](bool force_scalar, std::array<uint32_t, 64> &result) {
    util::set_force_scalar_for_testing(force_scalar);
    auto words = cdna5::build_vop3p(
        cdna5::kVPkFmaF32Vop3p,
        {.vdst = 6, .neg_hi = 4, .src0 = 256, .src1 = 258, .src2 = 260, .opsel_hi = 3, .neg = 2});
    words[0] |= uint32_t{1} << 14; // pad_14 is the src2 high-half selector.
    auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
    ASSERT_NE(decoder, nullptr);
    std::unique_ptr<Instruction> instruction(decode_valid(*decoder, words.data()));
    ASSERT_NE(instruction, nullptr);
    auto *typed = dynamic_cast<cdna5::VPkFmaF32Vop3p *>(instruction.get());
    ASSERT_NE(typed, nullptr);

    Gfx1250Sim sim;
    auto *cu = sim.cu();
    auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
    ASSERT_NE(wf, nullptr);
    wf->set_exec(kExec);
    const uint32_t vgpr_base = wf->vgpr_alloc().base;
    for (uint32_t lane = 0; lane < 32; ++lane) {
      const float value = static_cast<float>(lane + 1);
      cu->write_vgpr(vgpr_base, lane, std::bit_cast<uint32_t>(value * 0.25f));
      cu->write_vgpr(vgpr_base + 1, lane, std::bit_cast<uint32_t>(value * -0.5f));
      cu->write_vgpr(vgpr_base + 2, lane, std::bit_cast<uint32_t>(value + 0.75f));
      cu->write_vgpr(vgpr_base + 3, lane, std::bit_cast<uint32_t>(value * 1.5f));
      cu->write_vgpr(vgpr_base + 4, lane, std::bit_cast<uint32_t>(value * -2.0f));
      cu->write_vgpr(vgpr_base + 5, lane, std::bit_cast<uint32_t>(value + 3.0f));
      cu->write_vgpr(vgpr_base + 6, lane, kDstLoSeed);
      cu->write_vgpr(vgpr_base + 7, lane, kDstHiSeed);
    }

    if (!force_scalar) {
      EXPECT_TRUE(amdgpu::try_execute_vop3p_pk_ternary_f32_simd(
          *typed, *wf, 0u, 3u, 1u,
          [](auto a, auto b, auto c) { return util::stdx::fma(a, b, c); }));
      for (uint32_t lane = 0; lane < 32; ++lane) {
        cu->write_vgpr(vgpr_base + 6, lane, kDstLoSeed);
        cu->write_vgpr(vgpr_base + 7, lane, kDstHiSeed);
      }
    }

    cu->execute_instruction(instruction.get(), *wf);
    for (uint32_t lane = 0; lane < 32; ++lane) {
      result[lane * 2] = cu->read_vgpr(vgpr_base + 6, lane);
      result[lane * 2 + 1] = cu->read_vgpr(vgpr_base + 7, lane);
      if ((kExec & (1u << lane)) == 0u) {
        EXPECT_EQ(result[lane * 2], kDstLoSeed);
        EXPECT_EQ(result[lane * 2 + 1], kDstHiSeed);
      } else {
        const float value = static_cast<float>(lane + 1);
        const float expected_lo = std::fma(value * 0.25f, -(value + 0.75f), value * -2.0f);
        const float expected_hi = std::fma(value * -0.5f, value * 1.5f, -(value + 3.0f));
        EXPECT_EQ(result[lane * 2], std::bit_cast<uint32_t>(expected_lo));
        EXPECT_EQ(result[lane * 2 + 1], std::bit_cast<uint32_t>(expected_hi));
      }
    }
  };

  run_case(true, scalar_result);
  if constexpr (util::has_stdx_simd) {
    run_case(false, simd_result);
    EXPECT_EQ(simd_result, scalar_result);
  }
}

TEST(Gfx1250DecodeTest, Vop3pRejectsLiteral64SelectorInEverySourcePosition) {
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);

  for (const cdna5::Vop3pBuilderFields fields : {
           cdna5::Vop3pBuilderFields{.src0 = 254},
           cdna5::Vop3pBuilderFields{.src1 = 254},
           cdna5::Vop3pBuilderFields{.src2 = 254},
       }) {
    const auto words = cdna5::build_vop3p(cdna5::kVPkFmaF32Vop3p, fields);
    EXPECT_TRUE(decode_fails(*decoder, words.data()));
  }
}

TEST(Gfx1250DecodeTest, BinaryVop3pIgnoresLiteral64SelectorInUnusedSrc2) {
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);

  for (const uint32_t opcode : {cdna5::kVPkAddF32Vop3p, cdna5::kVPkMulF32Vop3p}) {
    const auto words = cdna5::build_vop3p(opcode, {.src0 = 128, .src1 = 129, .src2 = 254});
    std::unique_ptr<Instruction> instruction(decode_valid(*decoder, words.data()));
    ASSERT_NE(instruction, nullptr);
    EXPECT_EQ(instruction->size(), 8);
    EXPECT_EQ(instruction->num_src_operands(), 2);
  }
}

TEST(Gfx1250ExecutionTest, VCmpGtU32Wave32ExplicitSdstPreservesHighSgpr) {
  Gfx1250Sim sim;
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(0x3u);

  const uint32_t vgpr_base = wf->vgpr_alloc().base;
  cu->write_vgpr(vgpr_base + 4, 0, 3u);
  cu->write_vgpr(vgpr_base + 4, 1, 5u);
  write_wave_sgpr(*cu, *wf, 2, 0xaaaaaaaau);
  write_wave_sgpr(*cu, *wf, 3, 0xfefefefeu);
  wf->set_vcc(0x12345678u);

  const std::array<uint32_t, 2> words = {
      0xD44C0002u, // v_cmp_gt_u32_e64 s2, 4, v4
      0x02020884u,
  };
  cdna5::VCmpGtU32Vop3 cmp(words.data());
  cmp.execute_impl(*wf);

  EXPECT_EQ(read_wave_sgpr(*cu, *wf, 2), 0x1u);
  EXPECT_EQ(read_wave_sgpr(*cu, *wf, 3), 0xfefefefeu);
  EXPECT_EQ(wf->vcc(), 0x12345678u);
}

TEST(Gfx1250ExecutionTest, Wave32ScalarVccHiWritePreservesUpperHalf) {
  Gfx1250Sim sim;
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  ASSERT_EQ(wf->wf_size(), 32u);
  wf->set_exec(0xffff0000u);
  wf->set_vcc(0);

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);

  const uint32_t words[] = {0x8c6b7e6bu, 0}; // s_or_b32 vcc_hi, vcc_hi, exec_lo
  std::unique_ptr<Instruction> inst(decode_valid(*decoder, words));
  ASSERT_NE(inst, nullptr);
  ASSERT_EQ(std::string_view(inst->mnemonic()), "s_or_b32");
  cu->execute_instruction(inst.get(), *wf);

  EXPECT_EQ(wf->vcc(), 0xffff0000'00000000ULL);
}

} // namespace
