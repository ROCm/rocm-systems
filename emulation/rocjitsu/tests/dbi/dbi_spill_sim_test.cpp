// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file dbi_spill_sim_test.cpp
/// @brief Simulator end-to-end for DBI VGPR and SGPR register spilling
/// (CDNA4/gfx950).
///
/// The static InstrumentorProbeSpill.* tests (tests/patch/instrumentor_test.cpp)
/// prove the patched ELF *contains* the scratch store/load/wait bracket and the
/// bumped descriptor. This test proves the spill is *correct at runtime*: it
/// patches a kernel so a probe clobbers a register that is live across the anchor,
/// executes the patched kernel in the rocjitsu simulator, and confirms the live
/// value survives the probe call (i.e. it was saved to scratch and restored).
///
/// This test runs entirely in the simulator, so it exercises the CDNA4-only spill
/// path on any host. A wavefront frees its register file at s_endpgm, so the
/// final register state is read from a HaltSnapshotPlugin captured at halt.
///
/// VGPR kernel (entry at .text offset 0):
///   v_mov_b32 v2, K   ; offset 0: sentinel into v2
///   v_mov_b32 v3, v2  ; offset 4: ANCHOR -- reads v2 into v3 (v2 live here)
///   s_endpgm          ; offset 8
/// Probe: { v_mov_b32 v2, 0 ; s_setpc_b64 s[30:31] } clobbers v2.
/// With a correct spill: v2 is saved before the probe and restored after, so the
/// relocated `v_mov v3, v2` copies K into v3. Read v3 back: it must equal K.
///
/// The SGPR case mirrors this with a live s8 (clobbered by the probe) copied into
/// v3 after the probe returns; the spill bridges the scalar through a VGPR via
/// v_writelane/v_readlane. Read v3 back: it must equal K.

#include "../aql_queue.h"
#include "../halt_snapshot_plugin.h"
#include "../patch/gfx950_test_fixtures.h"
#include "embedded_schema.h"

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/patch/instruction_builder.h"
#include "rocjitsu/code/patch/instrumentor.h"
#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/config/config_loader.h"
#include "rocjitsu/vm/amdgpu/command_processor.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/shader_engine.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "rocjitsu/vm/amdgpu/xcd.h"
#include "rocjitsu/vm/soc.h"

#include "simdojo/sim/simulation.h"

#include "rocjitsu/base/rj_compiler.h"
RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include "hsa/AMDHSAKernelDescriptor.h"
RJ_DIAGNOSTIC_POP

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace rocjitsu {
namespace {

using test::kMovV2Zero;
using test::kMovV3V2;
using test::kProbeSetpcS30S31;

constexpr uint32_t kWaveSize = 64;
constexpr uint32_t kSentinel = 7; // Inline-const value placed into v2 (1..64).

// Minimal single-CU CDNA4 simulator that lays out a kernel descriptor + code in
// GPU memory (AMDHSA ABI), dispatches one workgroup, runs to completion, and
// reads back a VGPR. Self-contained so this slice does not disturb the
// file-local VmFixture in amdgpu_vm_test.cpp.
class Cdna4Sim {
public:
  Cdna4Sim() {
    const std::string json = R"({"max_ticks":100000,"num_threads":1,"vm":{"arch":"cdna4"},)"
                             R"("topology":{"root":{"name":"soc","type":"soc","children":[)"
                             R"({"name":"vram","type":"gpu_memory"},)"
                             R"({"name":"xcd0","type":"xcd","children":[)"
                             R"({"name":"l2","type":"l2_cache"},)"
                             R"({"name":"cp","type":"command_processor"},)"
                             R"({"name":"se0","type":"shader_engine","children":[)"
                             R"({"name":"cu[0:1]","type":"compute_unit","config":[)"
                             R"({"key":"num_wf_slots","value":"10"},)"
                             R"({"key":"sgprs_per_wf","value":"800"},)"
                             R"({"key":"vgprs_per_wf","value":"256"},)"
                             R"({"key":"lds_size_kb","value":"64"})"
                             R"(]}]}]}]},"links":[)"
                             R"({"src":"xcd0.cp.req_0","dst":"xcd0.se0.cu0.cpl","latency":1,"weight":2},)"
                             R"({"src":"xcd0.se0.cu0.req","dst":"xcd0.l2.cpl_0","latency":1,"weight":10})"
                             R"(]}})";
    loaded_ = config::load_config_from_string(json, kEmbeddedSchema);
    soc_ = loaded_.soc();
    mem_ = loaded_.memory();
    engine_ = std::make_unique<simdojo::SimulationEngine>(loaded_.engine_config);
    engine_->topology().set_root(loaded_.take_root());
    loaded_.wire_links(engine_->topology());
    engine_->create();
    plugin_group_ = test::make_halt_snapshot_group(&snapshot_plugin_);
    soc_->set_plugin_group(plugin_group_);
  }

  amdgpu::GpuMemory *mem() { return mem_; }
  amdgpu::CommandProcessor *cp() { return soc_->xcd(0)->command_processor(); }
  amdgpu::ComputeUnitCore *cu() { return soc_->xcd(0)->shader_engine(0)->compute_unit(0); }

  // Write a kernel_descriptor_t (entry at code start) followed by `code`, with
  // the given per-lane scratch size, and return the kernel_object address.
  uint64_t write_kernel(uint64_t addr, const std::vector<uint32_t> &code, uint32_t private_bytes) {
    using namespace rocr::llvm::amdhsa;
    kernel_descriptor_t kd{};
    kd.kernel_code_entry_byte_offset = sizeof(kernel_descriptor_t);
    AMDHSA_BITS_SET(kd.compute_pgm_rsrc1, COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT,
                    ((256 / 8) - 1));
    // 104 SGPRs is ample for the probe link pair s[30:31] and envelope temps.
    AMDHSA_BITS_SET(kd.compute_pgm_rsrc1, COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT,
                    ((104 / 8) - 1));
    AMDHSA_BITS_SET(kd.compute_pgm_rsrc2, COMPUTE_PGM_RSRC2_USER_SGPR_COUNT, 2);
    kd.private_segment_fixed_size = private_bytes;

    mem_->load_image(reinterpret_cast<const uint8_t *>(&kd), sizeof(kd), addr);
    mem_->load_image(reinterpret_cast<const uint8_t *>(code.data()),
                     code.size() * sizeof(uint32_t), addr + sizeof(kernel_descriptor_t));
    return addr;
  }

  // Dispatch `code` (scratch = private_bytes) over one wave and return v[reg]
  // for every lane after the kernel halts.
  std::vector<uint32_t> run_and_read_vgpr(const std::vector<uint32_t> &code, uint32_t private_bytes,
                                          uint32_t reg) {
    const uint64_t ko = write_kernel(0x1000, code, private_bytes);
    test::AqlQueue queue(mem_, cp());
    queue.dispatch(ko, /*grid_size_x=*/kWaveSize, /*workgroup_size_x=*/kWaveSize);
    engine_->run();

    if (snapshot_plugin_->snapshots().empty())
      return {};
    const test::WavefrontSnapshot &wf = snapshot_plugin_->snapshots().front();

    std::vector<uint32_t> out(kWaveSize);
    for (uint32_t lane = 0; lane < kWaveSize; ++lane)
      out[lane] = wf.vgpr(reg, lane);
    return out;
  }

private:
  config::LoadedConfig loaded_;
  SoC *soc_ = nullptr;
  amdgpu::GpuMemory *mem_ = nullptr;
  std::shared_ptr<ExecutionPluginGroup> plugin_group_;
  test::HaltSnapshotPlugin *snapshot_plugin_ = nullptr;
  std::unique_ptr<simdojo::SimulationEngine> engine_;
};

// Patch a gfx950 kernel with a probe that clobbers the live VGPR v2, forcing a
// spill. Exposes the patched .text words and the bumped scratch size.
class DbiSpillSimFixture : public ::testing::Test {
protected:
  void SetUp() override {
    const uint32_t endpgm = build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4);
    // v_mov_b32 v2, K ; v_mov_b32 v3, v2 ; s_endpgm. Anchor at offset 4 so v2 is
    // live across the anchor and its restored value flows into the observable v3.
    auto target = test::make_gfx950_kernel_elf(
        {test::make_mov_v2_inline(kSentinel), kMovV3V2, endpgm}, /*private_bytes=*/64);
    auto probe = test::make_gfx950_probe_elf("rj_test_probe", {kMovV2Zero, kProbeSetpcS30S31});

    AmdGpuCodeObject obj(target.data(), target.size());
    AmdGpuCodeObject probe_obj(probe.data(), probe.size());
    ASSERT_TRUE(obj.is_valid());
    ASSERT_TRUE(probe_obj.is_valid());

    Instrumentor instr(obj, ROCJITSU_CODE_ARCH_CDNA4);
    InstrumentationPoint pt;
    pt.anchor_offset = 4; // v_mov_b32 v3, v2 -> reads v2 (v2 live at the anchor).
    pt.probe_obj = &probe_obj;
    pt.probe_symbol = "rj_test_probe";
    instr.add_point(pt);

    auto result = instr.patch_with_debug_summaries();
    ASSERT_TRUE(result.errors.empty())
        << (result.errors.empty() ? std::string{} : result.errors.front());
    ASSERT_EQ(result.patches.size(), 1u);
    ASSERT_TRUE(result.patches[0].is_probe_call);

    AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
    ASSERT_TRUE(patched.is_valid());
    patched_text_ = test::section_words(patched, ".text");
    ASSERT_FALSE(patched_text_.empty());
    patched_scratch_ = test::patched_private_segment_size(patched);
  }

  std::vector<uint32_t> patched_text_;
  uint32_t patched_scratch_ = 0;
};

// The spilled VGPR survives the clobbering probe when executed: v3 (a copy of
// the restored v2, made after the probe returns) equals the sentinel on every
// lane. Also confirms the descriptor scratch grew (64 -> 68) so the sim
// allocated a per-lane spill slot.
TEST_F(DbiSpillSimFixture, SpilledVgprSurvivesClobberingProbe) {
  EXPECT_EQ(patched_scratch_, 68u) << "descriptor scratch must grow to cover the spill slot";

  Cdna4Sim sim;
  const std::vector<uint32_t> v3 = sim.run_and_read_vgpr(patched_text_, patched_scratch_, /*reg=*/3);
  ASSERT_EQ(v3.size(), kWaveSize) << "kernel did not run to completion (no dispatched wavefront)";
  for (uint32_t lane = 0; lane < kWaveSize; ++lane)
    EXPECT_EQ(v3[lane], kSentinel)
        << "lane " << lane << ": v2 was not restored after the probe clobbered it";
}

// Negative control: prove the *restore* is what preserves the value. Overwrite
// the epilogue scratch_load (the restore) with s_nop, so v2 stays clobbered (0)
// and the copy into v3 reads 0. Then run the intact patched text to confirm the
// zero result is caused by the missing restore, not unrelated corruption.
TEST_F(DbiSpillSimFixture, MissingRestoreLeavesVgprClobbered) {
  const std::vector<uint32_t> load = build_scratch_load_dword(2, 64, ROCJITSU_CODE_ARCH_CDNA4);
  const uint32_t nop = build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4);

  std::vector<uint32_t> sabotaged = patched_text_;
  auto it = std::search(sabotaged.begin(), sabotaged.end(), load.begin(), load.end());
  ASSERT_NE(it, sabotaged.end()) << "epilogue scratch_load (the restore) not found in patched text";
  for (size_t i = 0; i < load.size(); ++i)
    *(it + static_cast<std::ptrdiff_t>(i)) = nop;

  Cdna4Sim broken;
  const std::vector<uint32_t> v3_broken =
      broken.run_and_read_vgpr(sabotaged, patched_scratch_, /*reg=*/3);
  ASSERT_EQ(v3_broken.size(), kWaveSize);
  for (uint32_t lane = 0; lane < kWaveSize; ++lane)
    EXPECT_EQ(v3_broken[lane], 0u)
        << "lane " << lane << ": without the restore, v3 should read the clobbered 0";

  // Revert: the intact patched text restores v2, so v3 reads the sentinel again.
  Cdna4Sim intact;
  const std::vector<uint32_t> v3_intact =
      intact.run_and_read_vgpr(patched_text_, patched_scratch_, /*reg=*/3);
  ASSERT_EQ(v3_intact.size(), kWaveSize);
  for (uint32_t lane = 0; lane < kWaveSize; ++lane)
    EXPECT_EQ(v3_intact[lane], kSentinel) << "lane " << lane << ": intact restore should recover v2";
}

//==============================================================================
// SGPR spill (bridged through a VGPR)
//==============================================================================

constexpr uint32_t kMovV3S8 = 0x7E060208u; // v_mov_b32 v3, s8 -> reads s8 into v3.
constexpr uint16_t kSpilledSgpr = 8;

// Patch a gfx950 kernel with a probe that clobbers the live SGPR s8, forcing an
// SGPR spill (v_writelane -> scratch -> v_readlane through a bridge VGPR).
class DbiSgprSpillSimFixture : public ::testing::Test {
protected:
  void SetUp() override {
    const uint32_t endpgm = build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4);
    const uint32_t mov_s8_k =
        build_s_mov_b32(kSpilledSgpr, static_cast<uint16_t>(128 + kSentinel), ROCJITSU_CODE_ARCH_CDNA4);
    const uint32_t mov_s8_0 = build_s_mov_b32(kSpilledSgpr, 128, ROCJITSU_CODE_ARCH_CDNA4);
    // s_mov_b32 s8, K ; v_mov_b32 v3, s8 ; s_endpgm. Anchor at offset 4 so s8 is
    // live across the anchor and its restored value flows into the observable v3.
    auto target = test::make_gfx950_kernel_elf({mov_s8_k, kMovV3S8, endpgm}, /*private_bytes=*/64);
    auto probe = test::make_gfx950_probe_elf("rj_test_probe", {mov_s8_0, kProbeSetpcS30S31});

    AmdGpuCodeObject obj(target.data(), target.size());
    AmdGpuCodeObject probe_obj(probe.data(), probe.size());
    ASSERT_TRUE(obj.is_valid());
    ASSERT_TRUE(probe_obj.is_valid());

    Instrumentor instr(obj, ROCJITSU_CODE_ARCH_CDNA4);
    InstrumentationPoint pt;
    pt.anchor_offset = 4; // v_mov_b32 v3, s8 -> reads s8 (s8 live at the anchor).
    pt.probe_obj = &probe_obj;
    pt.probe_symbol = "rj_test_probe";
    instr.add_point(pt);

    auto result = instr.patch_with_debug_summaries();
    ASSERT_TRUE(result.errors.empty())
        << (result.errors.empty() ? std::string{} : result.errors.front());
    ASSERT_EQ(result.patches.size(), 1u);
    ASSERT_TRUE(result.patches[0].is_probe_call);

    AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
    ASSERT_TRUE(patched.is_valid());
    patched_text_ = test::section_words(patched, ".text");
    ASSERT_FALSE(patched_text_.empty());
    patched_scratch_ = test::patched_private_segment_size(patched);
  }

  std::vector<uint32_t> patched_text_;
  uint32_t patched_scratch_ = 0;
};

// The spilled SGPR survives the clobbering probe: v3 (a copy of the restored s8,
// made after the probe returns) equals the sentinel on every lane.
TEST_F(DbiSgprSpillSimFixture, SpilledSgprSurvivesClobberingProbe) {
  EXPECT_EQ(patched_scratch_, 68u) << "descriptor scratch must grow to cover the spill slot";

  Cdna4Sim sim;
  const std::vector<uint32_t> v3 = sim.run_and_read_vgpr(patched_text_, patched_scratch_, /*reg=*/3);
  ASSERT_EQ(v3.size(), kWaveSize) << "kernel did not run to completion (no dispatched wavefront)";
  for (uint32_t lane = 0; lane < kWaveSize; ++lane)
    EXPECT_EQ(v3[lane], kSentinel)
        << "lane " << lane << ": s8 was not restored after the probe clobbered it";
}

// Negative control: overwrite the epilogue v_readlane (the scalar restore) with
// s_nop, so s8 stays clobbered (0) and the copy into v3 reads 0. Then run the
// intact patched text to confirm the zero is caused by the missing readlane.
TEST_F(DbiSgprSpillSimFixture, MissingReadlaneLeavesSgprClobbered) {
  const std::array<uint32_t, 2> readlane =
      build_v_readlane_b32(kSpilledSgpr, /*bridge=*/0, /*lane=*/0, ROCJITSU_CODE_ARCH_CDNA4);
  const uint32_t nop = build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4);

  std::vector<uint32_t> sabotaged = patched_text_;
  auto it = std::search(sabotaged.begin(), sabotaged.end(), readlane.begin(), readlane.end());
  ASSERT_NE(it, sabotaged.end()) << "epilogue v_readlane (the scalar restore) not found";
  for (size_t i = 0; i < readlane.size(); ++i)
    *(it + static_cast<std::ptrdiff_t>(i)) = nop;

  Cdna4Sim broken;
  const std::vector<uint32_t> v3_broken =
      broken.run_and_read_vgpr(sabotaged, patched_scratch_, /*reg=*/3);
  ASSERT_EQ(v3_broken.size(), kWaveSize);
  for (uint32_t lane = 0; lane < kWaveSize; ++lane)
    EXPECT_EQ(v3_broken[lane], 0u)
        << "lane " << lane << ": without the readlane, v3 should read the clobbered 0";

  Cdna4Sim intact;
  const std::vector<uint32_t> v3_intact =
      intact.run_and_read_vgpr(patched_text_, patched_scratch_, /*reg=*/3);
  ASSERT_EQ(v3_intact.size(), kWaveSize);
  for (uint32_t lane = 0; lane < kWaveSize; ++lane)
    EXPECT_EQ(v3_intact[lane], kSentinel) << "lane " << lane << ": intact restore should recover s8";
}

} // namespace
} // namespace rocjitsu
