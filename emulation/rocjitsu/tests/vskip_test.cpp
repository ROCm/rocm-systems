// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "mma_test_util.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna1/test_encodings.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna2/test_encodings.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna3/test_encodings.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/test_encodings.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/matrix_coexecution.h"
#include "rocjitsu/vm/plugins/execution_plugin_group.h"

#include <gtest/gtest.h>

#include <array>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace {
using namespace rocjitsu;
using namespace rocjitsu::amdgpu;

constexpr uint64_t kProgram = 0x100000;
constexpr uint32_t kEnable = 0xBF108081u;     // s_setvskip 1, 0
constexpr uint32_t kDisable = 0xBF108080u;    // s_setvskip 0, 0
constexpr uint32_t kMoveVector = 0x7E000281u; // v_mov_b32 v0, 1

class IssueRecorder final : public ExecutionPlugin {
public:
  IssueRecorder() : ExecutionPlugin("vskip_recorder") {}
  void onAmdgpuBeforeExecuteInstruction(uint64_t, const Instruction &inst, Wavefront &) override {
    issued.emplace_back(inst.mnemonic());
  }
  std::vector<std::string> issued;
};

class VskipTest : public ::testing::TestWithParam<rj_code_arch_t> {
protected:
  void SetUp() override {
    ComputeUnitCore::Config config{};
    config.arch = GetParam();
    config.num_wf_slots = 1;
    config.sgprs_per_wf = 104;
    config.vgprs_per_wf = 256;
    config.lds_size_kb = 64;
    config.functional_quantum = 1;
    config.memory_wait_diagnostics = MemoryWaitDiagnostics::Warn;
    cu = ComputeUnitCore::create("vskip_cu", config, &memory, &l2);
    decoder = Decoder::create(GetParam());
    ASSERT_NE(cu, nullptr);
    ASSERT_NE(decoder, nullptr);
    wf = cu->dispatch_wf(0, kProgram, 104, 256);
    ASSERT_NE(wf, nullptr);
    wf->set_exec(~uint64_t{0});
  }

  void TearDown() override {
    if (wf && !wf->is_halted())
      wf->halt();
  }

  void program(std::span<const uint32_t> words) {
    for (size_t i = 0; i < words.size(); ++i)
      memory.write32(kProgram + i * 4, words[i]);
    // Supply the complete instruction fetch window at the end of the program.
    for (size_t i = 0; i < 4; ++i)
      memory.write32(kProgram + (words.size() + i) * 4, 0xBF800000u);
  }

  void step(unsigned bytes = 4) {
    const uint64_t previous = wf->pc;
    (void)cu->step();
    EXPECT_FALSE(wf->is_halted());
    EXPECT_EQ(wf->pc, previous + bytes);
  }

  GpuMemory memory{"vskip_memory"};
  L2Cache l2{"vskip_l2"};
  std::unique_ptr<ComputeUnitCore> cu;
  std::unique_ptr<Decoder> decoder;
  Wavefront *wf = nullptr;
};

TEST_P(VskipTest, SelectedSourceBitUpdatesOnlyVskip) {
  // Index 63 aliases bit 31. Literal operands and EXEC=0 still update MODE.
  const std::array words{0xBF100100u, kDisable, 0xBF109FFFu, 0x80000000u, 0xBF108081u, kDisable};
  program(words);
  constexpr uint32_t mode = 0x000003C5u;
  wf->set_mode_raw(mode);
  wf->write_scc(1);
  wf->set_exec(0);
  wf->set_vcc(0x123456789ABCDEF0ull);
  cu->write_sgpr(wf->sgpr_alloc().base, 0x80000000u);
  cu->write_sgpr(wf->sgpr_alloc().base + 1, 63);
  for (unsigned bytes : {4, 4, 8, 4, 4}) {
    const bool enabled = wf->pc != kProgram + 4 && wf->pc != kProgram + 20;
    step(bytes);
    EXPECT_EQ(wf->mode_raw(), mode | (enabled ? Wavefront::VSKIP_BIT : 0));
    EXPECT_EQ(wf->read_scc(), 1u);
    EXPECT_EQ(wf->exec(), 0u);
    EXPECT_EQ(wf->vcc(), 0x123456789ABCDEF0ull);
  }
}

TEST_P(VskipTest, SkipsVectorEffectsAndResumesAfterScalarDisable) {
  const std::array words{kEnable,     kMoveVector,
                         0x7E080500u, // v_readfirstlane_b32 s4, v0 also skips with EXEC=0.
                         0xBE850087u, // s_mov_b32 s5, 7 still executes.
                         kDisable,    kMoveVector};
  program(words);
  cu->write_vgpr(wf->vgpr_alloc().base, 0, 99);
  cu->write_sgpr(wf->sgpr_alloc().base + 4, 42);
  auto plugins = std::make_shared<ExecutionPluginGroup>(PluginSinkConfig{});
  auto recorder = std::make_unique<IssueRecorder>();
  auto *record = recorder.get();
  ASSERT_TRUE(plugins->add(std::move(recorder)));
  cu->set_plugin_group(plugins);
  step();
  step();
  wf->set_exec(0);
  step();
  step();
  EXPECT_EQ(cu->read_vgpr(wf->vgpr_alloc().base, 0), 99u);
  EXPECT_EQ(cu->read_sgpr(wf->sgpr_alloc().base + 4), 42u);
  EXPECT_EQ(cu->read_sgpr(wf->sgpr_alloc().base + 5), 7u);
  wf->set_exec(~uint64_t{0});
  step();
  step();
  EXPECT_EQ(cu->read_vgpr(wf->vgpr_alloc().base, 0), 1u);
  EXPECT_EQ(record->issued,
            (std::vector<std::string>{"s_setvskip", "s_mov_b32", "s_setvskip", "v_mov_b32_e32"}));
}

TEST_P(VskipTest, SetregAndGetregShareModeStateAndRedispatchClearsIt) {
  // MODE writes require two wait states before a vector op or S_GETREG.
  constexpr uint32_t kModeWait = 0xBF800001u; // s_nop 1
  const std::array words{0xBA000701u, 1u,     // s_setreg_imm32_b32 hwreg(MODE,28,1), 1
                         kModeWait,   kMoveVector,
                         0xB8860701u, // s_getreg_b32 s6, hwreg(MODE,28,1)
                         0xB9070701u, // s_setreg_b32 hwreg(MODE,28,1), s7
                         kModeWait,   kMoveVector, kEnable};
  program(words);
  step(8);
  step();
  step();
  EXPECT_EQ(cu->read_vgpr(wf->vgpr_alloc().base, 0), 0u);
  step();
  EXPECT_EQ(cu->read_sgpr(wf->sgpr_alloc().base + 6), 1u);
  step();
  step();
  step();
  EXPECT_EQ(cu->read_vgpr(wf->vgpr_alloc().base, 0), 1u);
  step();
  ASSERT_NE(wf->mode_raw() & Wavefront::VSKIP_BIT, 0u);
  wf->halt();
  const uint64_t resumed_vector = kProgram + (words.size() - 2) * sizeof(uint32_t);
  wf = cu->dispatch_wf(1, resumed_vector, 104, 256);
  ASSERT_NE(wf, nullptr);
  EXPECT_EQ(wf->mode_raw() & Wavefront::VSKIP_BIT, 0u);
}

TEST_P(VskipTest, VectorEncodingFamiliesSkipWithoutIssueOrWaitCounterEffects) {
  // Sample the available vector encoding families. Dedicated tests below cover
  // scalar-destination VOP3 and matrix forms absent from this table. Unsupported
  // vector operations must also skip before reaching their execute callback.
  auto check = [&](const auto &encodings) {
    std::vector<uint32_t> words{kEnable};
    unsigned skipped = 0;
    for (const auto &entry : encodings) {
      std::string_view name = entry.mnemonic;
      if (!(name.starts_with("v_") || name.starts_with("buffer_") || name.starts_with("tbuffer_") ||
            name.starts_with("image_") || name.starts_with("ds_") || name.starts_with("flat_") ||
            name.starts_with("global_") || name.starts_with("scratch_") || name == "exp"))
        continue;
      // The generated sample table has at most two words; wider MFMA extension
      // encodings have separate coverage in the async test below.
      const std::array<uint32_t, 4> padded{entry.words[0], entry.words[1], 0, 0};
      auto inst = decoder->decode(padded.data());
      // Generated samples can violate operand constraints (for example, the
      // v_readfirstlane sample selects an SGPR source). Dedicated cases use
      // valid encodings and require successful decoding.
      if (inst.failed() || inst.value()->size() > 8)
        continue;
      ASSERT_TRUE(inst.value()->is_vskip_affected()) << name;
      words.insert(words.end(), padded.begin(), padded.begin() + inst.value()->size() / 4);
      ++skipped;
    }
    words.push_back(kDisable);
    program(words);
    auto plugins = std::make_shared<ExecutionPluginGroup>(PluginSinkConfig{});
    auto recorder = std::make_unique<IssueRecorder>();
    auto *record = recorder.get();
    ASSERT_TRUE(plugins->add(std::move(recorder)));
    cu->set_plugin_group(plugins);
    wf->wait_counters().vmcnt = 3;
    wf->wait_counters().lgkmcnt = 2;
    wf->wait_counters().expcnt = 1;
    for (unsigned i = 0; i < skipped + 2; ++i)
      (void)cu->step();
    EXPECT_FALSE(wf->is_halted());
    EXPECT_EQ(wf->pc, kProgram + words.size() * 4);
    EXPECT_EQ(record->issued, (std::vector<std::string>{"s_setvskip", "s_setvskip"}));
    EXPECT_EQ(wf->wait_counters().vmcnt, 3u);
    EXPECT_EQ(wf->wait_counters().lgkmcnt, 2u);
    EXPECT_EQ(wf->wait_counters().expcnt, 1u);
    wf->wait_counters() = {};
  };
  switch (GetParam()) {
  case ROCJITSU_CODE_ARCH_CDNA1:
    check(cdna1::test_data::ENCODINGS);
    break;
  case ROCJITSU_CODE_ARCH_CDNA2:
    check(cdna2::test_data::ENCODINGS);
    break;
  case ROCJITSU_CODE_ARCH_CDNA3:
    check(cdna3::test_data::ENCODINGS);
    break;
  case ROCJITSU_CODE_ARCH_CDNA4:
    check(cdna4::test_data::ENCODINGS);
    break;
  default:
    FAIL() << "Unexpected target";
  }
}

TEST_P(VskipTest, Vop3SubencodingsSkipWithoutIssue) {
  // These XML encoding names omit ENC_: VOP3_SDST_ENC and VOP3P_MFMA.
  // Neither form is sampled by the generated mnemonic-only encoding table.
  const uint32_t matrix = GetParam() == ROCJITSU_CODE_ARCH_CDNA1 ? 0xD3C50000u : 0xD3C58000u;
  const std::array<std::array<uint32_t, 4>, 2> encodings{{
      {0xD1E00806u, 0x040A0300u, 0, 0}, // v_div_scale_f32 v6, s[8:9], v0, v1, v2
      {matrix, 0x04020300u, 0, 0},      // v_mfma_f32_16x16x4_f32 a[0:3], v0, v1, a[0:3]
  }};
  std::vector<uint32_t> words{kEnable};
  for (const auto &encoding : encodings) {
    auto inst = decoder->decode(encoding.data());
    ASSERT_TRUE(inst.succeeded());
    ASSERT_EQ(inst.value()->size(), 8);
    EXPECT_TRUE(inst.value()->is_vskip_affected()) << inst.value()->mnemonic();
    words.insert(words.end(), encoding.begin(), encoding.begin() + 2);
  }
  words.insert(words.end(), {kDisable, kMoveVector});
  program(words);
  cu->write_vgpr(wf->vgpr_alloc().base + 6, 0, 0x12345678u);
  cu->write_sgpr(wf->sgpr_alloc().base + 8, 0xA5A5A5A5u);
  auto plugins = std::make_shared<ExecutionPluginGroup>(PluginSinkConfig{});
  auto recorder = std::make_unique<IssueRecorder>();
  auto *record = recorder.get();
  ASSERT_TRUE(plugins->add(std::move(recorder)));
  cu->set_plugin_group(plugins);
  for (unsigned bytes : {4, 8, 8, 4, 4})
    step(bytes);
  EXPECT_EQ(cu->read_vgpr(wf->vgpr_alloc().base + 6, 0), 0x12345678u);
  EXPECT_EQ(cu->read_sgpr(wf->sgpr_alloc().base + 8), 0xA5A5A5A5u);
  EXPECT_EQ(cu->read_vgpr(wf->vgpr_alloc().base, 0), 1u);
  EXPECT_EQ(record->issued,
            (std::vector<std::string>{"s_setvskip", "s_setvskip", "v_mov_b32_e32"}));
}

INSTANTIATE_TEST_SUITE_P(Cdna, VskipTest,
                         ::testing::Values(ROCJITSU_CODE_ARCH_CDNA1, ROCJITSU_CODE_ARCH_CDNA2,
                                           ROCJITSU_CODE_ARCH_CDNA3, ROCJITSU_CODE_ARCH_CDNA4));

TEST(VskipAsyncTest, ExtendedMfmaSkipsBeforeAsyncSubmission) {
  GpuMemory memory{"vskip_async_memory"};
  L2Cache l2{"vskip_async_l2"};
  ComputeUnitCore::Config config{};
  config.arch = ROCJITSU_CODE_ARCH_CDNA4;
  config.num_wf_slots = 1;
  config.sgprs_per_wf = 104;
  config.vgprs_per_wf = 512;
  config.async_resources = std::make_shared<matrix_coexecution::ExecutionResources>(1);
  auto cu = ComputeUnitCore::create("vskip_async_cu", config, &memory, &l2);
  auto *wf = cu->dispatch_wf(0, kProgram, 104, 256);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(~uint64_t{0});
  const auto matrix = mma_test::make_cdna4_mfma_scale_words(45, 1, 448, 449);
  memory.write32(kProgram, kEnable);
  for (size_t i = 0; i < matrix.size(); ++i)
    memory.write32(kProgram + 4 + i * 4, matrix[i]);
  memory.write32(kProgram + 20, kDisable);
  memory.write32(kProgram + 24, kMoveVector);
  for (unsigned i = 28; i < 48; i += 4)
    memory.write32(kProgram + i, 0xBF800000u);
  cu->write_vgpr(wf->vgpr_alloc().base + 64, 0, 0x3F800000);
  (void)cu->step();
  (void)cu->step();
  EXPECT_FALSE(wf->is_halted());
  EXPECT_EQ(wf->pc, kProgram + 20);
  EXPECT_EQ(cu->read_vgpr(wf->vgpr_alloc().base + 64, 0), 0x3F800000u);
  EXPECT_TRUE(wf->wait_counters().empty());
  (void)cu->step();
  (void)cu->step();
  EXPECT_EQ(cu->read_vgpr(wf->vgpr_alloc().base, 0), 1u);
  wf->halt();
}

TEST(VskipTargetTest, ModeBitDoesNotSuppressRdnaOrCdna5) {
  for (auto arch : {ROCJITSU_CODE_ARCH_RDNA1, ROCJITSU_CODE_ARCH_RDNA2, ROCJITSU_CODE_ARCH_RDNA3,
                    ROCJITSU_CODE_ARCH_RDNA4, ROCJITSU_CODE_ARCH_CDNA5}) {
    SCOPED_TRACE(arch);
    GpuMemory memory{"vskip_other_memory"};
    L2Cache l2{"vskip_other_l2"};
    ComputeUnitCore::Config config{};
    config.arch = arch;
    config.num_wf_slots = 1;
    config.sgprs_per_wf = 106;
    config.vgprs_per_wf = 256;
    auto cu = ComputeUnitCore::create("vskip_other_cu", config, &memory, &l2);
    auto *wf = cu->dispatch_wf(0, kProgram, 106, 256);
    ASSERT_NE(wf, nullptr);
    wf->set_exec(~uint64_t{0});
    wf->set_mode_raw(Wavefront::VSKIP_BIT);
    memory.write32(kProgram, kMoveVector);
    for (unsigned i = 4; i < 16; i += 4)
      memory.write32(kProgram + i, 0xBF800000u);
    (void)cu->step();
    EXPECT_EQ(wf->pc, kProgram + 4);
    EXPECT_EQ(cu->read_vgpr(wf->vgpr_alloc().base, 0), 1u);
    wf->halt();
  }
}
} // namespace
