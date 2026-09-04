// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file dbi_spill_sim_test.cpp
/// @brief Simulator end-to-end for DBI VGPR and SGPR register spilling on CDNA3
/// (gfx942, wave64), CDNA4 (gfx950, wave64), and RDNA4 (gfx1200, wave32).
///
/// The static InstrumentorProbeSpill.* tests (tests/patch/instrumentor_test.cpp)
/// prove the patched ELF *contains* the scratch store/load/wait bracket and the
/// bumped descriptor. This test proves the spill is *correct at runtime*: it
/// patches a kernel so a probe clobbers a register that is live across the anchor,
/// executes the patched kernel in the rocjitsu simulator, and confirms the live
/// value survives the probe call (i.e. it was saved to scratch and restored).
///
/// This test runs entirely in the simulator, so it exercises the spill path on any
/// host. A wavefront frees its register file at s_endpgm, so the final register
/// state is read from a HaltSnapshotPlugin captured at halt.
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
#include "../dbi_test_util.h"
#include "../halt_snapshot_plugin.h"
#include "embedded_schema.h"

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/code/builders/spill_builders.h"
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
#include <array>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace rocjitsu {
namespace {

using test::kAccReadV3A0Hi;
using test::kAccReadV3A0Lo;
using test::kAccReadV7A0Hi;
using test::kAccReadV7A0Lo;
using test::kAccWriteA0ZeroHi;
using test::kAccWriteA0ZeroLo;
using test::kMovS8Zero;
using test::kMovS9Zero;
using test::kMovV2Zero;
using test::kMovV3S8;
using test::kMovV3V2;
using test::kMovV4S9;
using test::kMovV5V2;
using test::kMovV6S8;

constexpr uint32_t kWaveSize = 64;
constexpr uint32_t kSentinel = 7;  // Inline-const value placed into the spilled reg (1..64).
constexpr uint32_t kSentinel2 = 9; // Distinct value for a second reg, to catch swapped restores.

// v_add_u32 (gfx9) / v_add_nc_u32 (gfx11) vdst, src0, v{vsrc1}: a VOP2 that reads two
// operands and writes one VGPR. src0 is the 9-bit VOP2 src field (VGPR = 256 + index, SGPR
// = index); vsrc1 and vdst are VGPR indices. Reading a register as a source keeps it live
// across the anchor without clobbering it -- used to pin many registers live at once.
[[nodiscard]] uint32_t build_v_add_u32(uint16_t vdst, uint16_t src0, uint16_t vsrc1,
                                       rj_code_arch_t arch) {
  const uint8_t d = static_cast<uint8_t>(vdst);
  const uint8_t s1 = static_cast<uint8_t>(vsrc1);
  switch (arch) {
  case ROCJITSU_CODE_ARCH_CDNA3:
    return cdna3::build_vop2(cdna3::kVAddU32Vop2, {.src0 = src0, .vsrc1 = s1, .vdst = d})[0];
  case ROCJITSU_CODE_ARCH_CDNA4:
    return cdna4::build_vop2(cdna4::kVAddU32Vop2, {.src0 = src0, .vsrc1 = s1, .vdst = d})[0];
  case ROCJITSU_CODE_ARCH_RDNA4:
    return rdna4::build_vop2(rdna4::kVAddNcU32Vop2, {.src0 = src0, .vsrc1 = s1, .vdst = d})[0];
  default:
    throw util::UnimplementedInst("v_add_u32 for target architecture");
  }
}

// Minimal single-CU simulator (CDNA3, CDNA4, or RDNA4, selected by the constructor
// arch) that lays out a kernel descriptor + code in GPU memory (AMDHSA ABI), dispatches
// one workgroup, runs to completion, and reads back a VGPR. Self-contained so this
// slice does not disturb the file-local VmFixture in amdgpu_vm_test.cpp.
class DbiSim {
public:
  // Defaults to CDNA4/wave64; pass ("cdna3", 64) or ("rdna4", 32) for other arches.
  explicit DbiSim(std::string_view arch = "cdna4", uint32_t wave_size = kWaveSize)
      : wave_size_(wave_size) {
    const std::string json =
        std::string(R"({"max_ticks":100000,"num_threads":1,"vm":{"arch":")") + std::string(arch) +
        R"("},)"
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

  uint32_t queue_seq_ = 0;
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
    // Split the unified CDNA2-4 allocation evenly between ordinary and
    // accumulator registers. These simulations exercise both namespaces.
    AMDHSA_BITS_SET(kd.compute_pgm_rsrc3, COMPUTE_PGM_RSRC3_GFX90A_ACCUM_OFFSET, ((128 / 4) - 1));
    // 104 SGPRs is ample for the probe link pair s[30:31] and envelope temps.
    AMDHSA_BITS_SET(kd.compute_pgm_rsrc1, COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT,
                    ((104 / 8) - 1));
    AMDHSA_BITS_SET(kd.compute_pgm_rsrc2, COMPUTE_PGM_RSRC2_USER_SGPR_COUNT, 2);
    kd.private_segment_fixed_size = private_bytes;

    mem_->load_image(reinterpret_cast<const uint8_t *>(&kd), sizeof(kd), addr);
    mem_->load_image(reinterpret_cast<const uint8_t *>(code.data()), code.size() * sizeof(uint32_t),
                     addr + sizeof(kernel_descriptor_t));
    return addr;
  }

  // Dispatch `code` (scratch = private_bytes) over one wave and return v[reg]
  // for every lane after the kernel halts.
  std::vector<uint32_t> run_and_read_vgpr(const std::vector<uint32_t> &code, uint32_t private_bytes,
                                          uint32_t reg) {
    const uint64_t ko = write_kernel(0x1000, code, private_bytes);
    // Callers reuse one DbiSim for several dispatches, and each AqlQueue leaves
    // its registration behind on the CP, so every run needs its own queue --
    // its own id, since two live queues sharing one on a CP are rejected (fan-out
    // routes shards back by (queue_id, process_id)), and its own ring and pointer
    // page, since queues sharing a ring would let one doorbell be fetched and
    // dispatched once per registration.
    const uint32_t queue_id = ++queue_seq_;
    const uint64_t ring = test::AqlQueue::DEFAULT_RING_ADDR + uint64_t{queue_id} * 0x100000ULL;
    test::AqlQueue queue(mem_, cp(), ring, test::AqlQueue::DEFAULT_RING_SIZE, ring + 0x10000,
                         ring + 0x10008, ring + 0x10010, /*xcd_fanout=*/false,
                         /*queue_id=*/queue_id);
    queue.dispatch(ko, /*grid_size_x=*/wave_size_, /*workgroup_size_x=*/wave_size_);
    engine_->run();

    if (snapshot_plugin_->snapshots().empty())
      return {};
    const test::WavefrontSnapshot &wf = snapshot_plugin_->snapshots().front();

    std::vector<uint32_t> out(wave_size_);
    for (uint32_t lane = 0; lane < wave_size_; ++lane)
      out[lane] = wf.vgpr(reg, lane);
    return out;
  }

private:
  uint32_t wave_size_ = kWaveSize;
  config::LoadedConfig loaded_;
  SoC *soc_ = nullptr;
  amdgpu::GpuMemory *mem_ = nullptr;
  std::shared_ptr<ExecutionPluginGroup> plugin_group_;
  test::HaltSnapshotPlugin *snapshot_plugin_ = nullptr;
  std::unique_ptr<simdojo::SimulationEngine> engine_;
};

// VGPR / SGPR spill, executed on CDNA3 (gfx942, wave64), CDNA4 (gfx950, wave64),
// and RDNA4 (gfx1200, wave32). The arches share one base fixture each; the
// DbiCdna3*/DbiCdna4*/DbiRdna4* fixtures are thin wrappers selecting the arch config.

// Per-arch knobs for a spill sim fixture: the sim/config arch string, the code
// arch (for builders), the ELF machine flag, and the wavefront size.
struct SpillSimArch {
  const char *sim_arch;
  rj_code_arch_t arch;
  uint32_t e_flags;
  uint32_t wave_size;
};

inline constexpr SpillSimArch kCdna3SpillArch{"cdna3", ROCJITSU_CODE_ARCH_CDNA3,
                                              EF_AMDGPU_MACH_AMDGCN_GFX942, /*wave_size=*/64};
inline constexpr SpillSimArch kCdna4SpillArch{"cdna4", ROCJITSU_CODE_ARCH_CDNA4,
                                              EF_AMDGPU_MACH_AMDGCN_GFX950, /*wave_size=*/64};
inline constexpr SpillSimArch kRdna4SpillArch{"rdna4", ROCJITSU_CODE_ARCH_RDNA4,
                                              EF_AMDGPU_MACH_AMDGCN_GFX1200, /*wave_size=*/32};

// Patches a kernel with a probe that clobbers the live VGPR v2, forcing a spill,
// then runs it. Arch-parameterized; SpilledVgprSurvives/MissingRestore bodies are
// shared helpers so the per-arch fixtures below stay thin.
class DbiVgprSpillSimBase : public ::testing::Test {
protected:
  explicit DbiVgprSpillSimBase(const SpillSimArch &a) : a_(a) {}

  void SetUp() override {
    const uint32_t endpgm = build_s_endpgm(a_.arch);
    const uint32_t setpc = build_s_setpc_b64(/*s[30:31]=*/30, a_.arch);
    // v_mov v2, K ; v_mov v3, v2 (ANCHOR at offset 4, v2 live) ; s_endpgm.
    auto target = test::make_amdgpu_kernel_elf(
        {test::make_mov_vgpr_inline(2, kSentinel), kMovV3V2, endpgm}, /*private_bytes=*/64,
        /*granulated_sgpr_count=*/3, a_.e_flags);
    auto probe = test::make_amdgpu_probe_elf("rj_test_probe", {kMovV2Zero, setpc}, a_.e_flags);

    AmdGpuCodeObject obj(target.data(), target.size());
    AmdGpuCodeObject probe_obj(probe.data(), probe.size());
    ASSERT_TRUE(obj.is_valid());
    ASSERT_TRUE(probe_obj.is_valid());

    Instrumentor instr(obj, a_.arch);
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

  // The spilled VGPR survives the clobbering probe: v3 (a copy of the restored v2,
  // made after the probe returns) equals the sentinel on every active lane, and
  // the descriptor scratch grew (64 -> 68) to hold the per-lane slot.
  void expect_spilled_vgpr_survives() {
    EXPECT_EQ(patched_scratch_, 68u) << "descriptor scratch must grow to cover the spill slot";
    DbiSim sim(a_.sim_arch, a_.wave_size);
    const std::vector<uint32_t> v3 =
        sim.run_and_read_vgpr(patched_text_, patched_scratch_, /*reg=*/3);
    ASSERT_EQ(v3.size(), a_.wave_size) << "kernel did not run to completion (no dispatched wave)";
    for (uint32_t lane = 0; lane < a_.wave_size; ++lane)
      EXPECT_EQ(v3[lane], kSentinel)
          << "lane " << lane << ": v2 was not restored after the probe clobbered it";
  }

  // Negative control: prove the *restore* preserves the value. Nop the epilogue
  // scratch/VSCRATCH load so v2 stays clobbered (0) and v3 reads 0; then confirm
  // the intact text restores it.
  void expect_missing_restore_clobbers_vgpr() {
    const std::vector<uint32_t> load = build_scratch_load_dword(2, 64, a_.arch);
    const uint32_t nop = build_s_nop(0, a_.arch);

    std::vector<uint32_t> sabotaged = patched_text_;
    auto it = std::search(sabotaged.begin(), sabotaged.end(), load.begin(), load.end());
    ASSERT_NE(it, sabotaged.end()) << "epilogue scratch load (the restore) not found";
    for (size_t i = 0; i < load.size(); ++i)
      *(it + static_cast<std::ptrdiff_t>(i)) = nop;

    DbiSim broken(a_.sim_arch, a_.wave_size);
    const std::vector<uint32_t> v3_broken =
        broken.run_and_read_vgpr(sabotaged, patched_scratch_, /*reg=*/3);
    ASSERT_EQ(v3_broken.size(), a_.wave_size);
    for (uint32_t lane = 0; lane < a_.wave_size; ++lane)
      EXPECT_EQ(v3_broken[lane], 0u)
          << "lane " << lane << ": without the restore, v3 should read the clobbered 0";

    DbiSim intact(a_.sim_arch, a_.wave_size);
    const std::vector<uint32_t> v3_intact =
        intact.run_and_read_vgpr(patched_text_, patched_scratch_, /*reg=*/3);
    ASSERT_EQ(v3_intact.size(), a_.wave_size);
    for (uint32_t lane = 0; lane < a_.wave_size; ++lane)
      EXPECT_EQ(v3_intact[lane], kSentinel)
          << "lane " << lane << ": intact restore should recover v2";
  }

  SpillSimArch a_;
  std::vector<uint32_t> patched_text_;
  uint32_t patched_scratch_ = 0;
};

class DbiCdna3VgprSpillSimFixture : public DbiVgprSpillSimBase {
protected:
  DbiCdna3VgprSpillSimFixture() : DbiVgprSpillSimBase(kCdna3SpillArch) {}
};
class DbiCdna4VgprSpillSimFixture : public DbiVgprSpillSimBase {
protected:
  DbiCdna4VgprSpillSimFixture() : DbiVgprSpillSimBase(kCdna4SpillArch) {}
};
class DbiRdna4VgprSpillSimFixture : public DbiVgprSpillSimBase {
protected:
  DbiRdna4VgprSpillSimFixture() : DbiVgprSpillSimBase(kRdna4SpillArch) {}
};

TEST_F(DbiCdna3VgprSpillSimFixture, SpilledVgprSurvivesClobberingProbe) {
  expect_spilled_vgpr_survives();
}
TEST_F(DbiCdna3VgprSpillSimFixture, MissingRestoreLeavesVgprClobbered) {
  expect_missing_restore_clobbers_vgpr();
}
TEST_F(DbiCdna4VgprSpillSimFixture, SpilledVgprSurvivesClobberingProbe) {
  expect_spilled_vgpr_survives();
}
TEST_F(DbiCdna4VgprSpillSimFixture, MissingRestoreLeavesVgprClobbered) {
  expect_missing_restore_clobbers_vgpr();
}
TEST_F(DbiRdna4VgprSpillSimFixture, SpilledVgprSurvivesClobberingProbe) {
  expect_spilled_vgpr_survives();
}
TEST_F(DbiRdna4VgprSpillSimFixture, MissingRestoreLeavesVgprClobbered) {
  expect_missing_restore_clobbers_vgpr();
}

// SGPR spill (bridged through a VGPR), on CDNA3, CDNA4, and RDNA4.

constexpr uint16_t kSpilledSgpr = 8;

// Patches a kernel with a probe that clobbers the live SGPR s8, forcing an SGPR
// spill (v_writelane -> scratch -> v_readlane through a bridge VGPR). Arch config
// as in DbiVgprSpillSimBase; the DbiCdna4*/DbiRdna4* wrappers select it.
class DbiSgprSpillSimBase : public ::testing::Test {
protected:
  explicit DbiSgprSpillSimBase(const SpillSimArch &a) : a_(a) {}

  void SetUp() override {
    const uint32_t endpgm = build_s_endpgm(a_.arch);
    const uint32_t setpc = build_s_setpc_b64(/*s[30:31]=*/30, a_.arch);
    const uint32_t mov_s8_k =
        build_s_mov_b32(kSpilledSgpr, static_cast<uint16_t>(128 + kSentinel), a_.arch);
    const uint32_t mov_s8_0 = build_s_mov_b32(kSpilledSgpr, 128, a_.arch);
    // s_mov s8, K ; v_mov v3, s8 (ANCHOR at offset 4, s8 live) ; s_endpgm.
    auto target = test::make_amdgpu_kernel_elf({mov_s8_k, kMovV3S8, endpgm}, /*private_bytes=*/64,
                                               /*granulated_sgpr_count=*/3, a_.e_flags);
    auto probe = test::make_amdgpu_probe_elf("rj_test_probe", {mov_s8_0, setpc}, a_.e_flags);

    AmdGpuCodeObject obj(target.data(), target.size());
    AmdGpuCodeObject probe_obj(probe.data(), probe.size());
    ASSERT_TRUE(obj.is_valid());
    ASSERT_TRUE(probe_obj.is_valid());

    Instrumentor instr(obj, a_.arch);
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

  // The spilled SGPR survives the clobbering probe: v3 (a copy of the restored s8,
  // made after the probe returns) equals the sentinel on every active lane.
  void expect_spilled_sgpr_survives() {
    EXPECT_EQ(patched_scratch_, 68u) << "descriptor scratch must grow to cover the spill slot";
    DbiSim sim(a_.sim_arch, a_.wave_size);
    const std::vector<uint32_t> v3 =
        sim.run_and_read_vgpr(patched_text_, patched_scratch_, /*reg=*/3);
    ASSERT_EQ(v3.size(), a_.wave_size) << "kernel did not run to completion (no dispatched wave)";
    for (uint32_t lane = 0; lane < a_.wave_size; ++lane)
      EXPECT_EQ(v3[lane], kSentinel)
          << "lane " << lane << ": s8 was not restored after the probe clobbered it";
  }

  // Negative control: nop the epilogue v_readlane (the scalar restore) so s8 stays
  // clobbered (0) and v3 reads 0; then confirm the intact text restores it.
  void expect_missing_readlane_clobbers_sgpr() {
    const std::array<uint32_t, 2> readlane =
        build_v_readlane_b32(kSpilledSgpr, /*bridge=*/0, /*lane=*/0, a_.arch);
    const uint32_t nop = build_s_nop(0, a_.arch);

    std::vector<uint32_t> sabotaged = patched_text_;
    auto it = std::search(sabotaged.begin(), sabotaged.end(), readlane.begin(), readlane.end());
    ASSERT_NE(it, sabotaged.end()) << "epilogue v_readlane (the scalar restore) not found";
    for (size_t i = 0; i < readlane.size(); ++i)
      *(it + static_cast<std::ptrdiff_t>(i)) = nop;

    DbiSim broken(a_.sim_arch, a_.wave_size);
    const std::vector<uint32_t> v3_broken =
        broken.run_and_read_vgpr(sabotaged, patched_scratch_, /*reg=*/3);
    ASSERT_EQ(v3_broken.size(), a_.wave_size);
    for (uint32_t lane = 0; lane < a_.wave_size; ++lane)
      EXPECT_EQ(v3_broken[lane], 0u)
          << "lane " << lane << ": without the readlane, v3 should read the clobbered 0";

    DbiSim intact(a_.sim_arch, a_.wave_size);
    const std::vector<uint32_t> v3_intact =
        intact.run_and_read_vgpr(patched_text_, patched_scratch_, /*reg=*/3);
    ASSERT_EQ(v3_intact.size(), a_.wave_size);
    for (uint32_t lane = 0; lane < a_.wave_size; ++lane)
      EXPECT_EQ(v3_intact[lane], kSentinel)
          << "lane " << lane << ": intact restore should recover s8";
  }

  SpillSimArch a_;
  std::vector<uint32_t> patched_text_;
  uint32_t patched_scratch_ = 0;
};

class DbiCdna3SgprSpillSimFixture : public DbiSgprSpillSimBase {
protected:
  DbiCdna3SgprSpillSimFixture() : DbiSgprSpillSimBase(kCdna3SpillArch) {}
};
class DbiCdna4SgprSpillSimFixture : public DbiSgprSpillSimBase {
protected:
  DbiCdna4SgprSpillSimFixture() : DbiSgprSpillSimBase(kCdna4SpillArch) {}
};
class DbiRdna4SgprSpillSimFixture : public DbiSgprSpillSimBase {
protected:
  DbiRdna4SgprSpillSimFixture() : DbiSgprSpillSimBase(kRdna4SpillArch) {}
};

TEST_F(DbiCdna3SgprSpillSimFixture, SpilledSgprSurvivesClobberingProbe) {
  expect_spilled_sgpr_survives();
}
TEST_F(DbiCdna3SgprSpillSimFixture, MissingReadlaneLeavesSgprClobbered) {
  expect_missing_readlane_clobbers_sgpr();
}
TEST_F(DbiCdna4SgprSpillSimFixture, SpilledSgprSurvivesClobberingProbe) {
  expect_spilled_sgpr_survives();
}
TEST_F(DbiCdna4SgprSpillSimFixture, MissingReadlaneLeavesSgprClobbered) {
  expect_missing_readlane_clobbers_sgpr();
}
TEST_F(DbiRdna4SgprSpillSimFixture, SpilledSgprSurvivesClobberingProbe) {
  expect_spilled_sgpr_survives();
}
TEST_F(DbiRdna4SgprSpillSimFixture, MissingReadlaneLeavesSgprClobbered) {
  expect_missing_readlane_clobbers_sgpr();
}

// Two live+clobbered SGPRs spilled through one shared bridge VGPR, on CDNA3/CDNA4/
// RDNA4. Runtime counterpart of the static SpillsMultipleLiveClobberedSgprs: it
// asserts both restored scalar values survive, not just that the words were emitted.

constexpr uint16_t kSpilledSgprHi = 9;

class DbiTwoSgprSpillSimBase : public ::testing::Test {
protected:
  explicit DbiTwoSgprSpillSimBase(const SpillSimArch &a) : a_(a) {}

  void SetUp() override {
    const uint32_t endpgm = build_s_endpgm(a_.arch);
    const uint32_t setpc = build_s_setpc_b64(/*s[30:31]=*/30, a_.arch);
    const uint32_t mov_s8_k =
        build_s_mov_b32(kSpilledSgpr, static_cast<uint16_t>(128 + kSentinel), a_.arch);
    const uint32_t mov_s9_k =
        build_s_mov_b32(kSpilledSgprHi, static_cast<uint16_t>(128 + kSentinel2), a_.arch);
    // s_mov s8,K1 ; s_mov s9,K2 ; v_mov v3,s8 (ANCHOR at offset 8: reads s8, s9 read
    // next -> both live) ; v_mov v4,s9 ; s_endpgm.
    auto target =
        test::make_amdgpu_kernel_elf({mov_s8_k, mov_s9_k, kMovV3S8, kMovV4S9, endpgm},
                                     /*private_bytes=*/64, /*granulated_sgpr_count=*/3, a_.e_flags);
    auto probe =
        test::make_amdgpu_probe_elf("rj_test_probe", {kMovS8Zero, kMovS9Zero, setpc}, a_.e_flags);

    AmdGpuCodeObject obj(target.data(), target.size());
    AmdGpuCodeObject probe_obj(probe.data(), probe.size());
    ASSERT_TRUE(obj.is_valid());
    ASSERT_TRUE(probe_obj.is_valid());

    Instrumentor instr(obj, a_.arch);
    InstrumentationPoint pt;
    pt.anchor_offset = 8; // v_mov v3, s8 -> reads s8; v_mov v4, s9 next -> s9 also live.
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

  // Both SGPRs survive the clobbering probe through the one shared bridge: v3
  // (restored s8) reads kSentinel and v4 (restored s9) reads kSentinel2 on every
  // lane; the distinct sentinels also catch a swapped restore. v3 and v4 come from
  // two deterministic runs (identical code -> identical register file), mirroring the
  // two-instance pattern the controls above use.
  void expect_both_sgprs_survive() {
    EXPECT_EQ(patched_scratch_, 72u) << "descriptor scratch must grow to cover both spill slots";

    DbiSim sim_s8(a_.sim_arch, a_.wave_size);
    const std::vector<uint32_t> v3 =
        sim_s8.run_and_read_vgpr(patched_text_, patched_scratch_, /*reg=*/3);
    DbiSim sim_s9(a_.sim_arch, a_.wave_size);
    const std::vector<uint32_t> v4 =
        sim_s9.run_and_read_vgpr(patched_text_, patched_scratch_, /*reg=*/4);
    ASSERT_EQ(v3.size(), a_.wave_size) << "kernel did not run to completion (no dispatched wave)";
    ASSERT_EQ(v4.size(), a_.wave_size);
    for (uint32_t lane = 0; lane < a_.wave_size; ++lane) {
      EXPECT_EQ(v3[lane], kSentinel) << "lane " << lane << ": s8 was not restored";
      EXPECT_EQ(v4[lane], kSentinel2) << "lane " << lane << ": s9 was not restored";
    }
  }

  SpillSimArch a_;
  std::vector<uint32_t> patched_text_;
  uint32_t patched_scratch_ = 0;
};

class DbiCdna3TwoSgprSpillSimFixture : public DbiTwoSgprSpillSimBase {
protected:
  DbiCdna3TwoSgprSpillSimFixture() : DbiTwoSgprSpillSimBase(kCdna3SpillArch) {}
};
class DbiCdna4TwoSgprSpillSimFixture : public DbiTwoSgprSpillSimBase {
protected:
  DbiCdna4TwoSgprSpillSimFixture() : DbiTwoSgprSpillSimBase(kCdna4SpillArch) {}
};
class DbiRdna4TwoSgprSpillSimFixture : public DbiTwoSgprSpillSimBase {
protected:
  DbiRdna4TwoSgprSpillSimFixture() : DbiTwoSgprSpillSimBase(kRdna4SpillArch) {}
};

TEST_F(DbiCdna3TwoSgprSpillSimFixture, BothSpilledSgprsSurviveClobberingProbe) {
  expect_both_sgprs_survive();
}
TEST_F(DbiCdna4TwoSgprSpillSimFixture, BothSpilledSgprsSurviveClobberingProbe) {
  expect_both_sgprs_survive();
}
TEST_F(DbiRdna4TwoSgprSpillSimFixture, BothSpilledSgprsSurviveClobberingProbe) {
  expect_both_sgprs_survive();
}

//==============================================================================
// AccVGPR (AGPR) spill, on CDNA3 and CDNA4. AGPRs are CDNA-only, so there is no
// RDNA4 variant. The spill is direct (scratch store/load with the FLAT acc bit,
// no bridge VGPR), unlike the SGPR path.
//==============================================================================

// Patches a kernel with a probe that clobbers the live AGPR acc0, forcing an
// AccVGPR spill (acc=1 scratch store -> acc=1 scratch load). Arch config as in
// DbiVgprSpillSimBase; the DbiCdna3*/DbiCdna4* wrappers select it.
class DbiAccVgprSpillSimBase : public ::testing::Test {
protected:
  explicit DbiAccVgprSpillSimBase(const SpillSimArch &a) : a_(a) {}

  void SetUp() override {
    const uint32_t endpgm = build_s_endpgm(a_.arch);
    const uint32_t setpc = build_s_setpc_b64(/*s[30:31]=*/30, a_.arch);
    // v_accvgpr_write a0, K ; v_accvgpr_read v3, a0 (ANCHOR at offset 8, acc0
    // live) ; s_endpgm. The read copies restored acc0 into v3 for observability.
    auto target = test::make_amdgpu_kernel_elf(
        {kAccWriteA0ZeroLo, test::make_accvgpr_write_a0_inline_hi(kSentinel), kAccReadV3A0Lo,
         kAccReadV3A0Hi, endpgm},
        /*private_bytes=*/64, /*granulated_sgpr_count=*/3, a_.e_flags);
    auto probe = test::make_amdgpu_probe_elf(
        "rj_test_probe", {kAccWriteA0ZeroLo, kAccWriteA0ZeroHi, setpc}, a_.e_flags);

    AmdGpuCodeObject obj(target.data(), target.size());
    AmdGpuCodeObject probe_obj(probe.data(), probe.size());
    ASSERT_TRUE(obj.is_valid());
    ASSERT_TRUE(probe_obj.is_valid());

    Instrumentor instr(obj, a_.arch);
    InstrumentationPoint pt;
    pt.anchor_offset = 8; // v_accvgpr_read v3, a0 -> reads acc0 (acc0 live at the anchor).
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

  // The spilled AGPR survives the clobbering probe: v3 (a copy of the restored
  // acc0, read after the probe returns) equals the sentinel on every active lane,
  // and the descriptor scratch grew (64 -> 68) to hold the per-lane slot.
  void expect_spilled_accvgpr_survives() {
    EXPECT_EQ(patched_scratch_, 68u) << "descriptor scratch must grow to cover the spill slot";
    DbiSim sim(a_.sim_arch, a_.wave_size);
    const std::vector<uint32_t> v3 =
        sim.run_and_read_vgpr(patched_text_, patched_scratch_, /*reg=*/3);
    ASSERT_EQ(v3.size(), a_.wave_size) << "kernel did not run to completion (no dispatched wave)";
    for (uint32_t lane = 0; lane < a_.wave_size; ++lane)
      EXPECT_EQ(v3[lane], kSentinel)
          << "lane " << lane << ": acc0 was not restored after the probe clobbered it";
  }

  // Negative control: nop the epilogue acc=1 scratch load (the AGPR restore) so
  // acc0 stays clobbered (0) and v3 reads 0; then confirm the intact text
  // restores it.
  void expect_missing_restore_clobbers_accvgpr() {
    const std::vector<uint32_t> load =
        build_scratch_load_dword(/*acc0=*/0, 64, a_.arch, /*acc=*/true);
    const uint32_t nop = build_s_nop(0, a_.arch);

    std::vector<uint32_t> sabotaged = patched_text_;
    auto it = std::search(sabotaged.begin(), sabotaged.end(), load.begin(), load.end());
    ASSERT_NE(it, sabotaged.end()) << "epilogue acc scratch load (the restore) not found";
    for (size_t i = 0; i < load.size(); ++i)
      *(it + static_cast<std::ptrdiff_t>(i)) = nop;

    DbiSim broken(a_.sim_arch, a_.wave_size);
    const std::vector<uint32_t> v3_broken =
        broken.run_and_read_vgpr(sabotaged, patched_scratch_, /*reg=*/3);
    ASSERT_EQ(v3_broken.size(), a_.wave_size);
    for (uint32_t lane = 0; lane < a_.wave_size; ++lane)
      EXPECT_EQ(v3_broken[lane], 0u)
          << "lane " << lane << ": without the restore, v3 should read the clobbered 0";

    DbiSim intact(a_.sim_arch, a_.wave_size);
    const std::vector<uint32_t> v3_intact =
        intact.run_and_read_vgpr(patched_text_, patched_scratch_, /*reg=*/3);
    ASSERT_EQ(v3_intact.size(), a_.wave_size);
    for (uint32_t lane = 0; lane < a_.wave_size; ++lane)
      EXPECT_EQ(v3_intact[lane], kSentinel)
          << "lane " << lane << ": intact restore should recover acc0";
  }

  SpillSimArch a_;
  std::vector<uint32_t> patched_text_;
  uint32_t patched_scratch_ = 0;
};

class DbiCdna3AccVgprSpillSimFixture : public DbiAccVgprSpillSimBase {
protected:
  DbiCdna3AccVgprSpillSimFixture() : DbiAccVgprSpillSimBase(kCdna3SpillArch) {}
};
class DbiCdna4AccVgprSpillSimFixture : public DbiAccVgprSpillSimBase {
protected:
  DbiCdna4AccVgprSpillSimFixture() : DbiAccVgprSpillSimBase(kCdna4SpillArch) {}
};

TEST_F(DbiCdna3AccVgprSpillSimFixture, SpilledAccVgprSurvivesClobberingProbe) {
  expect_spilled_accvgpr_survives();
}
TEST_F(DbiCdna3AccVgprSpillSimFixture, MissingRestoreLeavesAccVgprClobbered) {
  expect_missing_restore_clobbers_accvgpr();
}
TEST_F(DbiCdna4AccVgprSpillSimFixture, SpilledAccVgprSurvivesClobberingProbe) {
  expect_spilled_accvgpr_survives();
}
TEST_F(DbiCdna4AccVgprSpillSimFixture, MissingRestoreLeavesAccVgprClobbered) {
  expect_missing_restore_clobbers_accvgpr();
}

//==============================================================================
// Combined VGPR + SGPR + AccVGPR spill in a single bracket, on CDNA3 and CDNA4.
// Proves the three spill paths (direct VGPR scratch, SGPR bridge, direct acc-bit
// scratch) coexist in one trampoline and all round-trip. AGPR is CDNA-only, so no
// RDNA4 variant. Shares the kernel/probe shape with the static
// InstrumentorProbeSpill.Cdna4SpillsLiveClobberedVgprSgprAndAccVgpr.
//==============================================================================

// Kernel inits v2, s8, acc0 to the sentinel; the anchor reads v2 and the two
// following instructions read s8 then acc0 into distinct dests (v5/v6/v7), so all
// three are live at the anchor. The probe clobbers all three, forcing a combined
// spill; after it returns the reads copy the restored values out for observation.
class DbiCombinedSpillSimBase : public ::testing::Test {
protected:
  explicit DbiCombinedSpillSimBase(const SpillSimArch &a) : a_(a) {}

  void SetUp() override {
    const uint32_t endpgm = build_s_endpgm(a_.arch);
    const uint32_t setpc = build_s_setpc_b64(/*s[30:31]=*/30, a_.arch);
    const uint32_t mov_s8_k =
        build_s_mov_b32(kSpilledSgpr, static_cast<uint16_t>(128 + kSentinel), a_.arch);
    const uint32_t mov_s8_0 = build_s_mov_b32(kSpilledSgpr, 128, a_.arch);
    // v_mov v2,K ; s_mov s8,K ; v_accvgpr_write a0,K ; (ANCHOR at offset 16)
    // v_mov v5,v2 ; v_mov v6,s8 ; v_accvgpr_read v7,a0 ; s_endpgm.
    // 16 VGPRs with ACCUM_OFFSET=1 puts the accumulator window at v8-v15, so v2 and
    // the v5/v6/v7 dests are genuine ordinary VGPRs and acc0 sits in a nonempty
    // window -- the same layout as the static twin
    // (InstrumentorProbeSpill.Cdna4SpillsLiveClobberedVgprSgprAndAccVgpr).
    auto target = test::make_amdgpu_kernel_elf(
        {test::make_mov_vgpr_inline(2, kSentinel), mov_s8_k, kAccWriteA0ZeroLo,
         test::make_accvgpr_write_a0_inline_hi(kSentinel), kMovV5V2, kMovV6S8, kAccReadV7A0Lo,
         kAccReadV7A0Hi, endpgm},
        /*private_bytes=*/64, /*granulated_sgpr_count=*/3, a_.e_flags,
        /*granulated_vgpr_count=*/1, /*accum_offset=*/1);
    auto probe = test::make_amdgpu_probe_elf(
        "rj_test_probe", {kMovV2Zero, mov_s8_0, kAccWriteA0ZeroLo, kAccWriteA0ZeroHi, setpc},
        a_.e_flags);

    AmdGpuCodeObject obj(target.data(), target.size());
    AmdGpuCodeObject probe_obj(probe.data(), probe.size());
    ASSERT_TRUE(obj.is_valid());
    ASSERT_TRUE(probe_obj.is_valid());

    Instrumentor instr(obj, a_.arch);
    InstrumentationPoint pt;
    pt.anchor_offset = 16; // v_mov v5, v2 -> reads v2 (v2, s8, acc0 all live here).
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

  // All three restored values (v5<-v2, v6<-s8, v7<-acc0) equal the sentinel on
  // every active lane, and the descriptor grew to cover three slots (64 -> 76).
  void expect_all_survive() {
    EXPECT_EQ(patched_scratch_, 76u) << "descriptor scratch must cover three spill slots";
    DbiSim sim(a_.sim_arch, a_.wave_size);
    // One dispatch per reg: the harness snapshots a fresh run each call, which is
    // fine here since the kernel is deterministic.
    for (const auto &[reg, src] :
         {std::pair{uint32_t{5}, "v2"}, {uint32_t{6}, "s8"}, {uint32_t{7}, "acc0"}}) {
      const std::vector<uint32_t> v = sim.run_and_read_vgpr(patched_text_, patched_scratch_, reg);
      ASSERT_EQ(v.size(), a_.wave_size) << "kernel did not run to completion";
      for (uint32_t lane = 0; lane < a_.wave_size; ++lane)
        EXPECT_EQ(v[lane], kSentinel)
            << "lane " << lane << ": " << src << " was not restored after the probe clobbered it";
    }
  }

  // Negative control: nop all three restores (VGPR scratch load, SGPR readlane,
  // acc scratch load) so every value stays clobbered (0); confirm the intact text
  // restores all three.
  void expect_missing_restores_clobber_all() {
    const std::vector<uint32_t> vload = build_scratch_load_dword(2, 64, a_.arch);
    const std::array<uint32_t, 2> readlane =
        build_v_readlane_b32(kSpilledSgpr, /*bridge=*/0, /*lane=*/0, a_.arch);
    const std::vector<uint32_t> aload =
        build_scratch_load_dword(/*acc0=*/0, 72, a_.arch, /*acc=*/true);
    const uint32_t nop = build_s_nop(0, a_.arch);

    std::vector<uint32_t> sabotaged = patched_text_;
    const auto nop_seq = [&](const auto &seq, const char *what) {
      auto it = std::search(sabotaged.begin(), sabotaged.end(), seq.begin(), seq.end());
      ASSERT_NE(it, sabotaged.end()) << "restore not found: " << what;
      for (size_t i = 0; i < seq.size(); ++i)
        *(it + static_cast<std::ptrdiff_t>(i)) = nop;
    };
    nop_seq(vload, "VGPR scratch load");
    nop_seq(readlane, "SGPR readlane");
    nop_seq(aload, "acc scratch load");

    DbiSim broken(a_.sim_arch, a_.wave_size);
    for (uint32_t reg : {5u, 6u, 7u}) {
      const std::vector<uint32_t> v = broken.run_and_read_vgpr(sabotaged, patched_scratch_, reg);
      ASSERT_EQ(v.size(), a_.wave_size);
      for (uint32_t lane = 0; lane < a_.wave_size; ++lane)
        EXPECT_EQ(v[lane], 0u) << "lane " << lane << ": v" << reg << " should read the clobbered 0";
    }

    DbiSim intact(a_.sim_arch, a_.wave_size);
    for (uint32_t reg : {5u, 6u, 7u}) {
      const std::vector<uint32_t> v =
          intact.run_and_read_vgpr(patched_text_, patched_scratch_, reg);
      ASSERT_EQ(v.size(), a_.wave_size);
      for (uint32_t lane = 0; lane < a_.wave_size; ++lane)
        EXPECT_EQ(v[lane], kSentinel)
            << "lane " << lane << ": intact restore should recover v" << reg;
    }
  }

  SpillSimArch a_;
  std::vector<uint32_t> patched_text_;
  uint32_t patched_scratch_ = 0;
};

class DbiCdna3CombinedSpillSimFixture : public DbiCombinedSpillSimBase {
protected:
  DbiCdna3CombinedSpillSimFixture() : DbiCombinedSpillSimBase(kCdna3SpillArch) {}
};
class DbiCdna4CombinedSpillSimFixture : public DbiCombinedSpillSimBase {
protected:
  DbiCdna4CombinedSpillSimFixture() : DbiCombinedSpillSimBase(kCdna4SpillArch) {}
};

TEST_F(DbiCdna3CombinedSpillSimFixture, AllClassesSurviveClobberingProbe) { expect_all_survive(); }
TEST_F(DbiCdna3CombinedSpillSimFixture, MissingRestoresLeaveAllClobbered) {
  expect_missing_restores_clobber_all();
}
TEST_F(DbiCdna4CombinedSpillSimFixture, AllClassesSurviveClobberingProbe) { expect_all_survive(); }
TEST_F(DbiCdna4CombinedSpillSimFixture, MissingRestoresLeaveAllClobbered) {
  expect_missing_restores_clobber_all();
}

//==============================================================================
// A live+clobbered SGPR bridged through a *spilled* VGPR (reused bridge), on CDNA3/CDNA4/
// RDNA4. When no VGPR in the kernel's allocation is dead, the SGPR bridge reuses a clobbered
// VGPR that is itself being spilled; the epilogue must restore the SGPR (readlane out of the
// bridge) before it reloads the bridge VGPR's own value. A swapped reload would leave the
// bridge holding the writelaned scalar in lane 0. Runtime counterpart of the static
// PlanSgprSpillsReusesSpilledBridgeWhenNoneDead, which only checks bridge *selection*.
//
// The kernel allocates the minimum 8 VGPRs (encoding granule 8) and keeps all of v0-v7 live
// at the anchor via an add chain that reads each without writing v0, so the bridge search
// finds no dead VGPR in [0,8) and must reuse spilled v0. The probe clobbers v0 and s8 with
// distinct sentinels (v0 = kSentinel, s8 = kSentinel2): a correct restore leaves v0 = kSentinel
// on every lane; a swapped reload leaves v0[lane 0] = kSentinel2 (s8's writelaned value).
class DbiReusedBridgeSpillSimBase : public ::testing::Test {
protected:
  explicit DbiReusedBridgeSpillSimBase(const SpillSimArch &a) : a_(a) {}

  void SetUp() override {
    const uint32_t endpgm = build_s_endpgm(a_.arch);
    const uint32_t setpc = build_s_setpc_b64(/*s[30:31]=*/30, a_.arch);
    const uint32_t mov_s8_k =
        build_s_mov_b32(kSpilledSgpr, static_cast<uint16_t>(128 + kSentinel2), a_.arch);
    constexpr uint16_t kVgprSrc = 256; // VOP2 src field: VGPR = 256 + index.
    auto vadd = [&](uint16_t d, uint16_t s0, uint16_t s1) {
      return build_v_add_u32(d, s0, s1, a_.arch);
    };
    // v_mov v0,K ; s_mov s8,K2 ; then an add chain (ANCHOR at offset 8) that reads v0..v7
    // and s8 -- keeping all eight VGPRs and s8 live -- while writing only v1/v3/v5/v7, so v0
    // and s8 stay intact for readback. s_endpgm.
    auto target = test::make_amdgpu_kernel_elf(
        {test::make_mov_vgpr_inline(0, kSentinel), mov_s8_k, vadd(1, kVgprSrc + 0, 1),
         vadd(3, kVgprSrc + 2, 3), vadd(5, kVgprSrc + 4, 5), vadd(7, kVgprSrc + 6, 7),
         vadd(1, kSpilledSgpr, 1), endpgm},
        /*private_bytes=*/64, /*granulated_sgpr_count=*/3, a_.e_flags);
    auto probe = test::make_amdgpu_probe_elf(
        "rj_test_probe", {test::make_mov_vgpr_inline(0, 0), kMovS8Zero, setpc}, a_.e_flags);

    AmdGpuCodeObject obj(target.data(), target.size());
    AmdGpuCodeObject probe_obj(probe.data(), probe.size());
    ASSERT_TRUE(obj.is_valid());
    ASSERT_TRUE(probe_obj.is_valid());

    Instrumentor instr(obj, a_.arch);
    InstrumentationPoint pt;
    pt.anchor_offset = 8; // first add: reads v0; the chain keeps v0-v7 and s8 live.
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

  // The reused bridge is restored in the right order: v0 (the spilled bridge) reads its own
  // sentinel on every lane. A swapped reload -- bridge VGPR reloaded before the SGPR readlane
  // -- would leave lane 0 holding s8's sentinel. The target ELF's descriptor declares 8 VGPRs
  // (make_amdgpu_kernel_elf leaves the VGPR-count field 0 -> (0+1)*8), so the patch-time bridge
  // window is [0,8); with v0-v7 all live there is no dead bridge and the spiller must reuse
  // spilled v0. Scratch grows by v0's and s8's slots (64 -> 72).
  void expect_reused_bridge_restored_in_order() {
    EXPECT_EQ(patched_scratch_, 72u) << "descriptor scratch must grow for the v0 and s8 slots";
    // Guard non-triviality: the bridge must be the reused spilled v0, not a dead VGPR. A live
    // VGPR can only become the bridge via the reuse fallback, so an epilogue readlane pulling
    // s8 out of v0 proves the reused-bridge path was taken.
    const auto readlane = build_v_readlane_b32(kSpilledSgpr, /*bridge=*/0, /*lane=*/0, a_.arch);
    EXPECT_NE(
        std::search(patched_text_.begin(), patched_text_.end(), readlane.begin(), readlane.end()),
        patched_text_.end())
        << "SGPR bridge is not the reused spilled v0";
    DbiSim sim(a_.sim_arch, a_.wave_size);
    const std::vector<uint32_t> v0 =
        sim.run_and_read_vgpr(patched_text_, patched_scratch_, /*reg=*/0);
    ASSERT_EQ(v0.size(), a_.wave_size) << "kernel did not run to completion (no dispatched wave)";
    for (uint32_t lane = 0; lane < a_.wave_size; ++lane)
      EXPECT_EQ(v0[lane], kSentinel)
          << "lane " << lane << ": bridge v0 not restored; a swapped reload corrupts lane 0";
  }

  SpillSimArch a_;
  std::vector<uint32_t> patched_text_;
  uint32_t patched_scratch_ = 0;
};

class DbiCdna3ReusedBridgeSpillSimFixture : public DbiReusedBridgeSpillSimBase {
protected:
  DbiCdna3ReusedBridgeSpillSimFixture() : DbiReusedBridgeSpillSimBase(kCdna3SpillArch) {}
};
class DbiCdna4ReusedBridgeSpillSimFixture : public DbiReusedBridgeSpillSimBase {
protected:
  DbiCdna4ReusedBridgeSpillSimFixture() : DbiReusedBridgeSpillSimBase(kCdna4SpillArch) {}
};
class DbiRdna4ReusedBridgeSpillSimFixture : public DbiReusedBridgeSpillSimBase {
protected:
  DbiRdna4ReusedBridgeSpillSimFixture() : DbiReusedBridgeSpillSimBase(kRdna4SpillArch) {}
};

TEST_F(DbiCdna3ReusedBridgeSpillSimFixture, ReusedSpilledBridgeRestoredBeforeOwnReload) {
  expect_reused_bridge_restored_in_order();
}
TEST_F(DbiCdna4ReusedBridgeSpillSimFixture, ReusedSpilledBridgeRestoredBeforeOwnReload) {
  expect_reused_bridge_restored_in_order();
}
TEST_F(DbiRdna4ReusedBridgeSpillSimFixture, ReusedSpilledBridgeRestoredBeforeOwnReload) {
  expect_reused_bridge_restored_in_order();
}

//==============================================================================
// EXEC preservation + full-mask spilling on CDNA3 (wave64), CDNA4 (wave64), and
// RDNA4 (wave32). The partial-EXEC-mask scenario is wave-size-specific, so it rides
// a per-arch config (ExecSimArch) with thin DbiCdna3*/DbiCdna4*/DbiRdna4* wrappers.
// Only EXEC is runtime-observable; VCC/M0 are static-only
// (InstrumentorProbeSpill.*Preserves*).
//
// wave32 caveat: EXEC save/restore + full-mask toggle stay wave64-shaped
// (s_mov_b64); on wave32 the extra EXEC_HI write is harmless (EXEC_LO round-trips).

// s_mov_b32 <sdst>, <32-bit literal>: two words (src0 = 255 selects a trailing
// literal). Used for the RDNA4 wave32 partial EXEC mask (exec_lo=0x0000FFFF).
[[nodiscard]] std::vector<uint32_t> s_mov_b32_literal(uint16_t sdst, uint32_t literal,
                                                      rj_code_arch_t arch) {
  return {build_s_mov_b32(sdst, /*literal marker=*/255, arch), literal};
}

// Per-arch EXEC-sim config. set_partial_mask narrows EXEC to [0, active_lanes);
// restore_full_mask widens it back (both variable length).
struct ExecSimArch {
  SpillSimArch base;
  uint32_t active_lanes;
  std::vector<uint32_t> set_partial_mask;
  std::vector<uint32_t> restore_full_mask;
};

// CDNA3 wave64: clear exec_hi -> lanes 0..31; restore with exec_hi = -1.
[[nodiscard]] inline ExecSimArch cdna3_exec_arch() {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_CDNA3;
  return {kCdna3SpillArch,
          /*active_lanes=*/32,
          {build_s_mov_b32(/*exec_hi=*/127, /*inline 0=*/128, kArch)},
          {build_s_mov_b32(/*exec_hi=*/127, scalar_inline_neg_one(kArch), kArch)}};
}

// CDNA4 wave64: clear exec_hi -> lanes 0..31; restore with exec_hi = -1.
[[nodiscard]] inline ExecSimArch cdna4_exec_arch() {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_CDNA4;
  return {kCdna4SpillArch,
          /*active_lanes=*/32,
          {build_s_mov_b32(/*exec_hi=*/127, /*inline 0=*/128, kArch)},
          {build_s_mov_b32(/*exec_hi=*/127, scalar_inline_neg_one(kArch), kArch)}};
}

// RDNA4 wave32: narrow exec_lo to 0x0000FFFF (lanes 0..15); restore with exec_lo = -1.
[[nodiscard]] inline ExecSimArch rdna4_exec_arch() {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_RDNA4;
  return {kRdna4SpillArch,
          /*active_lanes=*/16,
          s_mov_b32_literal(scalar_operand_exec_lo(kArch), /*lanes 0..15=*/0x0000FFFFu, kArch),
          {build_s_mov_b32(scalar_operand_exec_lo(kArch), scalar_inline_neg_one(kArch), kArch)}};
}

// Index of the trampoline's EXEC restore: an s_mov_b64 writing EXEC_LO from a
// dead SGPR temp (< VCC_LO), not the probe's own `s_mov_b64 exec, 0` (reads
// inline 0 = 128). text.size() if absent.
[[nodiscard]] size_t find_exec_restore(const std::vector<uint32_t> &text, rj_code_arch_t arch) {
  const uint32_t mov64 = sop1_op_mov_b64(arch);
  for (size_t i = 0; i < text.size(); ++i) {
    const uint32_t w = text[i];
    if ((w >> 23) == kSop1EncodingPrefix && ((w >> 8) & 0xFFu) == mov64 &&
        ((w >> 16) & 0x7Fu) == scalar_operand_exec_lo(arch) &&
        (w & 0xFFu) < scalar_operand_vcc_lo(arch))
      return i;
  }
  return text.size();
}

// Patch a kernel whose anchor (v_mov_b32 v3, K) runs under a partial EXEC mask,
// with a probe that clobbers EXEC (s_mov_b64 exec, 0). If EXEC is preserved, the
// relocated v_mov writes K only to the originally-active [0, active_lanes) lanes.
class DbiExecPreserveSimBase : public ::testing::Test {
protected:
  explicit DbiExecPreserveSimBase(ExecSimArch a) : a_(std::move(a)) {}

  void SetUp() override {
    const uint32_t endpgm = build_s_endpgm(a_.base.arch);
    const uint32_t setpc = build_s_setpc_b64(/*s[30:31]=*/30, a_.base.arch);
    const uint32_t mov_v3_0 = 0x7E060280u;                      // v_mov_b32 v3, 0
    const uint32_t mov_v3_k = 0x7E060200u | (128u + kSentinel); // v_mov_b32 v3, K
    // Probe clobbers the whole EXEC pair: s_mov_b64 exec, 0.
    const uint32_t probe_clobber_exec =
        build_s_mov_b64(scalar_operand_exec_lo(a_.base.arch), /*inline 0=*/128, a_.base.arch);

    // v_mov v3,0 (all lanes) ; <narrow EXEC> ; ANCHOR v_mov v3,K ; s_endpgm.
    std::vector<uint32_t> code = {mov_v3_0};
    code.insert(code.end(), a_.set_partial_mask.begin(), a_.set_partial_mask.end());
    const uint32_t anchor_off = static_cast<uint32_t>(code.size() * sizeof(uint32_t));
    code.push_back(mov_v3_k);
    code.push_back(endpgm);

    auto target = test::make_amdgpu_kernel_elf(code, /*private_bytes=*/0,
                                               /*granulated_sgpr_count=*/3, a_.base.e_flags);
    auto probe =
        test::make_amdgpu_probe_elf("rj_test_probe", {probe_clobber_exec, setpc}, a_.base.e_flags);

    AmdGpuCodeObject obj(target.data(), target.size());
    AmdGpuCodeObject probe_obj(probe.data(), probe.size());
    ASSERT_TRUE(obj.is_valid());
    ASSERT_TRUE(probe_obj.is_valid());

    Instrumentor instr(obj, a_.base.arch);
    InstrumentationPoint pt;
    pt.anchor_offset = anchor_off; // v_mov_b32 v3, K, run under the partial exec mask.
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
    // EXEC preservation needs no scratch, so the descriptor is untouched.
    ASSERT_EQ(test::patched_private_segment_size(patched), 0u);
  }

  // EXEC survives: the relocated v_mov writes K to the active low lanes, leaving
  // the inactive high lanes at their prior 0.
  void expect_preserved_exec_survives() {
    DbiSim sim(a_.base.sim_arch, a_.base.wave_size);
    const std::vector<uint32_t> v3 = sim.run_and_read_vgpr(patched_text_, /*private_bytes=*/0,
                                                           /*reg=*/3);
    ASSERT_EQ(v3.size(), a_.base.wave_size) << "kernel did not run to completion";
    for (uint32_t lane = 0; lane < a_.base.wave_size; ++lane) {
      const uint32_t expected = lane < a_.active_lanes ? kSentinel : 0u;
      EXPECT_EQ(v3[lane], expected)
          << "lane " << lane << ": EXEC not restored to its anchor value after the probe";
    }
  }

  // Negative control: nop the EXEC restore, so exec stays 0 and the relocated
  // v_mov writes no lane -- v3 keeps its prior 0 everywhere.
  void expect_missing_exec_restore_clobbers_all() {
    const size_t restore = find_exec_restore(patched_text_, a_.base.arch);
    ASSERT_LT(restore, patched_text_.size()) << "EXEC restore (s_mov_b64 exec, tmp) not found";

    std::vector<uint32_t> sabotaged = patched_text_;
    sabotaged[restore] = build_s_nop(0, a_.base.arch);

    DbiSim broken(a_.base.sim_arch, a_.base.wave_size);
    const std::vector<uint32_t> v3_broken =
        broken.run_and_read_vgpr(sabotaged, /*private_bytes=*/0, /*reg=*/3);
    ASSERT_EQ(v3_broken.size(), a_.base.wave_size);
    for (uint32_t lane = 0; lane < a_.base.wave_size; ++lane)
      EXPECT_EQ(v3_broken[lane], 0u)
          << "lane " << lane << ": without the EXEC restore, no lane should be written";

    // Revert: the intact restore recovers the low-lane writes.
    DbiSim intact(a_.base.sim_arch, a_.base.wave_size);
    const std::vector<uint32_t> v3_intact =
        intact.run_and_read_vgpr(patched_text_, /*private_bytes=*/0, /*reg=*/3);
    ASSERT_EQ(v3_intact.size(), a_.base.wave_size);
    for (uint32_t lane = 0; lane < a_.base.wave_size; ++lane)
      EXPECT_EQ(v3_intact[lane], lane < a_.active_lanes ? kSentinel : 0u)
          << "lane " << lane << ": intact EXEC restore should recover the low-lane writes";
  }

  ExecSimArch a_;
  std::vector<uint32_t> patched_text_;
};

class DbiCdna3ExecPreserveSimFixture : public DbiExecPreserveSimBase {
protected:
  DbiCdna3ExecPreserveSimFixture() : DbiExecPreserveSimBase(cdna3_exec_arch()) {}
};
class DbiCdna4ExecPreserveSimFixture : public DbiExecPreserveSimBase {
protected:
  DbiCdna4ExecPreserveSimFixture() : DbiExecPreserveSimBase(cdna4_exec_arch()) {}
};
class DbiRdna4ExecPreserveSimFixture : public DbiExecPreserveSimBase {
protected:
  DbiRdna4ExecPreserveSimFixture() : DbiExecPreserveSimBase(rdna4_exec_arch()) {}
};

TEST_F(DbiCdna3ExecPreserveSimFixture, PreservedExecSurvivesClobberingProbe) {
  expect_preserved_exec_survives();
}
TEST_F(DbiCdna3ExecPreserveSimFixture, MissingExecRestoreLeavesAllLanesClobbered) {
  expect_missing_exec_restore_clobbers_all();
}
TEST_F(DbiCdna4ExecPreserveSimFixture, PreservedExecSurvivesClobberingProbe) {
  expect_preserved_exec_survives();
}
TEST_F(DbiCdna4ExecPreserveSimFixture, MissingExecRestoreLeavesAllLanesClobbered) {
  expect_missing_exec_restore_clobbers_all();
}
TEST_F(DbiRdna4ExecPreserveSimFixture, PreservedExecSurvivesClobberingProbe) {
  expect_preserved_exec_survives();
}
TEST_F(DbiRdna4ExecPreserveSimFixture, MissingExecRestoreLeavesAllLanesClobbered) {
  expect_missing_exec_restore_clobbers_all();
}

// Full-mask spilling: an EXEC-widening probe must not corrupt inactive-lane
// spilled registers. Executed on CDNA3 (wave64), CDNA4 (wave64), and RDNA4 (wave32).

// Spill a live VGPR (v2) at an anchor under a partial EXEC mask, with a probe that
// *widens* EXEC then clobbers v2. Full-mask spilling runs the store/load under
// EXEC=-1, saving/restoring v2 on all lanes; v2 is then copied to the observable v3.
class DbiExecWidenSpillSimBase : public ::testing::Test {
protected:
  explicit DbiExecWidenSpillSimBase(ExecSimArch a) : a_(std::move(a)) {}

  void SetUp() override {
    const uint32_t endpgm = build_s_endpgm(a_.base.arch);
    const uint32_t setpc = build_s_setpc_b64(/*s[30:31]=*/30, a_.base.arch);
    const uint32_t mov_v2_k = test::make_mov_vgpr_inline(2, kSentinel); // v_mov v2, K (all lanes)
    const uint32_t mov_v3_0 = 0x7E060280u;                              // v_mov v3, 0
    const uint32_t mov_v4_v2 = 0x7E080302u;                             // v_mov v4, v2 (v2 live)
    const uint32_t mov_v3_v2 = kMovV3V2;                                // v_mov v3, v2
    // Probe widens EXEC to all lanes, then clobbers v2 on all of them.
    const uint32_t probe_widen = build_s_mov_b64(scalar_operand_exec_lo(a_.base.arch),
                                                 scalar_inline_neg_one(a_.base.arch), a_.base.arch);

    // v2=K ; v3=0 ; <narrow EXEC> ; ANCHOR v_mov v4,v2 ; <widen EXEC> ; v3=v2 ; endpgm.
    std::vector<uint32_t> code = {mov_v2_k, mov_v3_0};
    code.insert(code.end(), a_.set_partial_mask.begin(), a_.set_partial_mask.end());
    const uint32_t anchor_off = static_cast<uint32_t>(code.size() * sizeof(uint32_t));
    code.push_back(mov_v4_v2);
    code.insert(code.end(), a_.restore_full_mask.begin(), a_.restore_full_mask.end());
    code.push_back(mov_v3_v2);
    code.push_back(endpgm);

    auto target = test::make_amdgpu_kernel_elf(code, /*private_bytes=*/64,
                                               /*granulated_sgpr_count=*/3, a_.base.e_flags);
    auto probe = test::make_amdgpu_probe_elf("rj_test_probe", {probe_widen, kMovV2Zero, setpc},
                                             a_.base.e_flags);

    AmdGpuCodeObject obj(target.data(), target.size());
    AmdGpuCodeObject probe_obj(probe.data(), probe.size());
    ASSERT_TRUE(obj.is_valid());
    ASSERT_TRUE(probe_obj.is_valid());

    Instrumentor instr(obj, a_.base.arch);
    InstrumentationPoint pt;
    pt.anchor_offset = anchor_off; // v_mov v4, v2 -> v2 live at the anchor, under the partial mask.
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
    ASSERT_EQ(patched_scratch_, 68u) << "descriptor scratch must grow to cover the spill slot";
  }

  // Full-mask spilling saves/restores v2 on all lanes, so v3 (a full-mask copy made
  // after the probe) reads the sentinel everywhere despite the widening probe.
  void expect_full_mask_spill_survives() {
    DbiSim sim(a_.base.sim_arch, a_.base.wave_size);
    const std::vector<uint32_t> v3 =
        sim.run_and_read_vgpr(patched_text_, patched_scratch_, /*reg=*/3);
    ASSERT_EQ(v3.size(), a_.base.wave_size) << "kernel did not run to completion";
    for (uint32_t lane = 0; lane < a_.base.wave_size; ++lane)
      EXPECT_EQ(v3[lane], kSentinel)
          << "lane " << lane
          << ": v2 was not saved/restored under full mask across the widening probe";
  }

  // Negative control: nop the store-side EXEC=-1 toggle, so the store runs under the
  // anchor mask and v2's inactive lanes are never saved -- they lose the sentinel.
  void expect_missing_store_full_mask_loses_high_lanes() {
    const std::vector<uint32_t> store = build_scratch_store_dword(2, 64, a_.base.arch);
    const uint32_t toggle = build_s_mov_b64(scalar_operand_exec_lo(a_.base.arch),
                                            scalar_inline_neg_one(a_.base.arch), a_.base.arch);
    // The in-flight-load drain now sits at the envelope top, not in the spill
    // prologue, so the store's immediately preceding word is the EXEC=-1 toggle.
    auto it = std::search(patched_text_.begin(), patched_text_.end(), store.begin(), store.end());
    ASSERT_NE(it, patched_text_.end()) << "spill scratch_store not found";
    const size_t store_idx = static_cast<size_t>(it - patched_text_.begin());
    ASSERT_GE(store_idx, 1u);
    const size_t toggle_idx = store_idx - 1;
    ASSERT_EQ(patched_text_[toggle_idx], toggle)
        << "word before the store should be the prologue EXEC=-1 toggle";

    std::vector<uint32_t> sabotaged = patched_text_;
    sabotaged[toggle_idx] = build_s_nop(0, a_.base.arch);

    DbiSim broken(a_.base.sim_arch, a_.base.wave_size);
    const std::vector<uint32_t> v3 =
        broken.run_and_read_vgpr(sabotaged, patched_scratch_, /*reg=*/3);
    ASSERT_EQ(v3.size(), a_.base.wave_size);
    for (uint32_t lane = 0; lane < a_.active_lanes; ++lane)
      EXPECT_EQ(v3[lane], kSentinel)
          << "lane " << lane << ": active-lane spill should still round-trip";
    for (uint32_t lane = a_.active_lanes; lane < a_.base.wave_size; ++lane)
      EXPECT_NE(v3[lane], kSentinel)
          << "lane " << lane << ": without the store full-mask, this lane must not be recovered";
  }

  ExecSimArch a_;
  std::vector<uint32_t> patched_text_;
  uint32_t patched_scratch_ = 0;
};

class DbiCdna3ExecWidenSpillSimFixture : public DbiExecWidenSpillSimBase {
protected:
  DbiCdna3ExecWidenSpillSimFixture() : DbiExecWidenSpillSimBase(cdna3_exec_arch()) {}
};
class DbiCdna4ExecWidenSpillSimFixture : public DbiExecWidenSpillSimBase {
protected:
  DbiCdna4ExecWidenSpillSimFixture() : DbiExecWidenSpillSimBase(cdna4_exec_arch()) {}
};
class DbiRdna4ExecWidenSpillSimFixture : public DbiExecWidenSpillSimBase {
protected:
  DbiRdna4ExecWidenSpillSimFixture() : DbiExecWidenSpillSimBase(rdna4_exec_arch()) {}
};

TEST_F(DbiCdna3ExecWidenSpillSimFixture, FullMaskSpillSurvivesExecWideningProbe) {
  expect_full_mask_spill_survives();
}
TEST_F(DbiCdna3ExecWidenSpillSimFixture, MissingStoreFullMaskLosesHighLanes) {
  expect_missing_store_full_mask_loses_high_lanes();
}
TEST_F(DbiCdna4ExecWidenSpillSimFixture, FullMaskSpillSurvivesExecWideningProbe) {
  expect_full_mask_spill_survives();
}
TEST_F(DbiCdna4ExecWidenSpillSimFixture, MissingStoreFullMaskLosesHighLanes) {
  expect_missing_store_full_mask_loses_high_lanes();
}
TEST_F(DbiRdna4ExecWidenSpillSimFixture, FullMaskSpillSurvivesExecWideningProbe) {
  expect_full_mask_spill_survives();
}
TEST_F(DbiRdna4ExecWidenSpillSimFixture, MissingStoreFullMaskLosesHighLanes) {
  expect_missing_store_full_mask_loses_high_lanes();
}

// Spilling site: the probe body runs under the ANCHOR mask, not EXEC=-1.
//
// The store bracket forces EXEC=-1 to spill all lanes; the trampoline must restore
// the anchor mask before the call so a probe doing per-lane work only touches
// anchor-active lanes. Observed via v5, which the probe writes but the kernel never
// reads -- so it is dead at the anchor and NOT spilled, letting the probe's write
// (and thus the mask it ran under) survive to the end. v2 is the spilled register
// (live + clobbered) that makes this a spilling site.
class DbiExecMaskAtSpillSimBase : public ::testing::Test {
protected:
  explicit DbiExecMaskAtSpillSimBase(ExecSimArch a) : a_(std::move(a)) {}

  void SetUp() override {
    const uint32_t endpgm = build_s_endpgm(a_.base.arch);
    const uint32_t setpc = build_s_setpc_b64(/*s[30:31]=*/30, a_.base.arch);
    const uint32_t mov_v5_0 = 0x7E0A0280u;                      // v_mov v5, 0 (all lanes)
    const uint32_t mov_v5_k = 0x7E0A0200u | (128u + kSentinel); // v_mov v5, K (probe marker)
    const uint32_t mov_v2_k = test::make_mov_vgpr_inline(2, kSentinel); // v_mov v2, K (spilled reg)
    const uint32_t mov_v4_v2 = 0x7E080302u;                             // v_mov v4, v2 (v2 live)
    // Probe marks v5 per lane, then clobbers v2 (forcing the spill + EXEC toggles).
    auto probe = test::make_amdgpu_probe_elf("rj_test_probe", {mov_v5_k, kMovV2Zero, setpc},
                                             a_.base.e_flags);

    // v5=0 ; v2=K ; <narrow EXEC> ; ANCHOR v_mov v4,v2 ; <widen EXEC> ; endpgm.
    std::vector<uint32_t> code = {mov_v5_0, mov_v2_k};
    code.insert(code.end(), a_.set_partial_mask.begin(), a_.set_partial_mask.end());
    const uint32_t anchor_off = static_cast<uint32_t>(code.size() * sizeof(uint32_t));
    code.push_back(mov_v4_v2);
    code.insert(code.end(), a_.restore_full_mask.begin(), a_.restore_full_mask.end());
    code.push_back(endpgm);

    auto target = test::make_amdgpu_kernel_elf(code, /*private_bytes=*/64,
                                               /*granulated_sgpr_count=*/3, a_.base.e_flags);
    AmdGpuCodeObject obj(target.data(), target.size());
    AmdGpuCodeObject probe_obj(probe.data(), probe.size());
    ASSERT_TRUE(obj.is_valid());
    ASSERT_TRUE(probe_obj.is_valid());

    Instrumentor instr(obj, a_.base.arch);
    InstrumentationPoint pt;
    pt.anchor_offset = anchor_off;
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
    ASSERT_EQ(patched_scratch_, 68u) << "descriptor scratch must grow to cover the spill slot";
  }

  // The probe ran under the anchor mask: v5 holds the marker only on the lanes that
  // were active at the anchor; the rest keep their prior 0.
  void expect_probe_runs_under_anchor_mask() {
    DbiSim sim(a_.base.sim_arch, a_.base.wave_size);
    const std::vector<uint32_t> v5 =
        sim.run_and_read_vgpr(patched_text_, patched_scratch_, /*reg=*/5);
    ASSERT_EQ(v5.size(), a_.base.wave_size) << "kernel did not run to completion";
    for (uint32_t lane = 0; lane < a_.base.wave_size; ++lane)
      EXPECT_EQ(v5[lane], lane < a_.active_lanes ? kSentinel : 0u)
          << "lane " << lane << ": probe write should honor the anchor mask";
  }

  // Negative control: nop the pre-call anchor-mask restore, so the probe runs under
  // the store-side EXEC=-1 and marks every lane, including anchor-inactive ones.
  void expect_missing_anchor_restore_runs_probe_full_mask() {
    const size_t restore = find_exec_restore(patched_text_, a_.base.arch);
    ASSERT_LT(restore, patched_text_.size()) << "pre-call EXEC anchor-restore not found";

    std::vector<uint32_t> sabotaged = patched_text_;
    sabotaged[restore] = build_s_nop(0, a_.base.arch);

    DbiSim broken(a_.base.sim_arch, a_.base.wave_size);
    const std::vector<uint32_t> v5 =
        broken.run_and_read_vgpr(sabotaged, patched_scratch_, /*reg=*/5);
    ASSERT_EQ(v5.size(), a_.base.wave_size);
    for (uint32_t lane = a_.active_lanes; lane < a_.base.wave_size; ++lane)
      EXPECT_EQ(v5[lane], kSentinel)
          << "lane " << lane
          << ": without the restore the probe runs full-mask and marks this lane";
  }

  ExecSimArch a_;
  std::vector<uint32_t> patched_text_;
  uint32_t patched_scratch_ = 0;
};

class DbiCdna3ExecMaskAtSpillSimFixture : public DbiExecMaskAtSpillSimBase {
protected:
  DbiCdna3ExecMaskAtSpillSimFixture() : DbiExecMaskAtSpillSimBase(cdna3_exec_arch()) {}
};
class DbiCdna4ExecMaskAtSpillSimFixture : public DbiExecMaskAtSpillSimBase {
protected:
  DbiCdna4ExecMaskAtSpillSimFixture() : DbiExecMaskAtSpillSimBase(cdna4_exec_arch()) {}
};
class DbiRdna4ExecMaskAtSpillSimFixture : public DbiExecMaskAtSpillSimBase {
protected:
  DbiRdna4ExecMaskAtSpillSimFixture() : DbiExecMaskAtSpillSimBase(rdna4_exec_arch()) {}
};

TEST_F(DbiCdna3ExecMaskAtSpillSimFixture, ProbeRunsUnderAnchorMask) {
  expect_probe_runs_under_anchor_mask();
}
TEST_F(DbiCdna3ExecMaskAtSpillSimFixture, MissingAnchorRestoreRunsProbeFullMask) {
  expect_missing_anchor_restore_runs_probe_full_mask();
}
TEST_F(DbiCdna4ExecMaskAtSpillSimFixture, ProbeRunsUnderAnchorMask) {
  expect_probe_runs_under_anchor_mask();
}
TEST_F(DbiCdna4ExecMaskAtSpillSimFixture, MissingAnchorRestoreRunsProbeFullMask) {
  expect_missing_anchor_restore_runs_probe_full_mask();
}
TEST_F(DbiRdna4ExecMaskAtSpillSimFixture, ProbeRunsUnderAnchorMask) {
  expect_probe_runs_under_anchor_mask();
}
TEST_F(DbiRdna4ExecMaskAtSpillSimFixture, MissingAnchorRestoreRunsProbeFullMask) {
  expect_missing_anchor_restore_runs_probe_full_mask();
}

} // namespace
} // namespace rocjitsu
