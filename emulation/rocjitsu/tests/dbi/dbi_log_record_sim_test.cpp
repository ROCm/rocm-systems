// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file dbi_log_record_sim_test.cpp
/// @brief Runs the record-writing test probe (log_record_probe.h) on the
///        simulator: the framework hands it a log buffer, it writes records,
///        and the host drains them.
///
/// Each case patches a small guest kernel, seeds a LogBuffer image in guest
/// memory, passes the image's address as the DBI kernarg payload, runs one wave,
/// copies the image back and drains it through LogBuffer::validate() and
/// drain(). The records are therefore read by the same code a host would use,
/// not by a parser written for the test.
///
/// The zero-mask pair is the mechanism test for forcing the full mask: with it
/// the probe records a guest region running under EXEC == 0, and without it the
/// same probe writes nothing at all, since every load and store it issues is
/// EXEC-masked.
///
/// Not established here: store ordering and the probe's waits (the simulator
/// completes every memory operation inside the instruction that issues it), and
/// Wave32 execution. DbiSim dispatches RDNA4 as Wave64; the RDNA4 cases are
/// Wave32 at patch time only, which is what makes the probe's high mask dword
/// the immediate zero.

#include "../dbi_test_util.h"
#include "../log_buffer_test_access.h"
#include "dbi_sim.h"
#include "log_record_probe.h"

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/code/patch/entry_prologue.h"
#include "rocjitsu/code/patch/instrumentor.h"
#include "rocjitsu/code/patch/kernarg_extension.h"
#include "rocjitsu/code/patch/log_abi.h"
#include "rocjitsu/code/patch/log_buffer.h"
#include "rocjitsu/code/rj_code.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace rocjitsu {
namespace {

// Clear of the kernel, kernargs, queue rings and scratch DbiSim places.
constexpr uint64_t kBufferAddr = 0x80000;
// Where the negative control puts its buffer instead; see
// expect_record_lost_under_the_guests_zero_mask(). Below the kernel at 0x1000.
constexpr uint64_t kControlBufferAddr = 0;
constexpr uint32_t kSlotCount = 4;
constexpr uint32_t kRecordType = 0x5A;
constexpr char kProbeSymbol[] = "rj_log_record_probe";

// The guest's own kernargs, which the wrapper copies and nothing here reads.
constexpr uint32_t kGuestKernargSize = 20;
constexpr uint64_t kGuestKernargPtr = 0x0000BEEF00001000ull;

// Eight lanes. On Wave64 they are all in the high dword, so the first of them
// (lane 36) is found only by a search over both dwords; on Wave32 the first is
// lane 4.
constexpr uint64_t kWave64PartialMask = 0x0000F0F000000000ull;
constexpr uint32_t kWave64PartialMaskFirstLane = 36;
constexpr uint64_t kWave32PartialMask = 0x0000F0F0;
constexpr uint32_t kWave32PartialMaskFirstLane = 4;
constexpr uint32_t kPartialMaskLanes = 8;

constexpr uint32_t kNoWriterLane = 0xFFFFFFFF;

// s_cbranch_scc1, which the shared builders do not cover.
uint32_t build_s_cbranch_scc1(int16_t offset_dwords, rj_code_arch_t arch) {
  uint16_t op = 0;
  switch (arch) {
  case ROCJITSU_CODE_ARCH_CDNA3:
    op = cdna3::kSCbranchScc1Sopp;
    break;
  case ROCJITSU_CODE_ARCH_CDNA4:
    op = cdna4::kSCbranchScc1Sopp;
    break;
  case ROCJITSU_CODE_ARCH_RDNA4:
    op = rdna4::kSCbranchScc1Sopp;
    break;
  default:
    ADD_FAILURE() << "no s_cbranch_scc1 opcode for this target";
    break;
  }
  return build_sopp_encoding(arch, op, static_cast<uint16_t>(offset_dwords));
}

struct SimTarget {
  const char *sim_arch;
  rj_code_arch_t arch;
  uint32_t e_flags;
  uint32_t wave_size;
  bool wave32;
};

constexpr SimTarget kCdna3{"cdna3", ROCJITSU_CODE_ARCH_CDNA3, EF_AMDGPU_MACH_AMDGCN_GFX942, 64,
                           false};
constexpr SimTarget kCdna4{"cdna4", ROCJITSU_CODE_ARCH_CDNA4, EF_AMDGPU_MACH_AMDGCN_GFX950, 64,
                           false};
constexpr SimTarget kRdna4{"rdna4", ROCJITSU_CODE_ARCH_RDNA4, EF_AMDGPU_MACH_AMDGCN_GFX1200, 32,
                           true};

// The kernarg image the CP delivers, with @p buffer_addr as the DBI payload.
// Built through the production helper, so the offset the prologue loads from is
// the one a runtime would write to.
std::vector<uint8_t> make_wrapper_image(uint64_t buffer_addr) {
  const std::array<KernargExtensionPayloadLayout, 1> payloads{kDbiEntryPayloadLayout};
  const auto layout = make_kernarg_extension_layout(kGuestKernargSize, payloads);
  if (!layout)
    return {};
  const std::vector<uint8_t> guest_kernargs(kGuestKernargSize, 0x5A);
  const KernargExtensionPayloadWrite payload{&buffer_addr, sizeof(buffer_addr)};
  std::vector<uint8_t> wrapper(layout->wrapper_size, 0);
  if (!write_kernarg_extension_wrapper(wrapper, *layout, guest_kernargs.data(), kGuestKernargPtr,
                                       std::span{&payload, 1}))
    return {};
  return wrapper;
}

// What one dispatch leaves in the buffer.
struct Drained {
  RjLogBufferHeader header{};
  std::vector<RjLogRecord> records;
  DrainStats stats;
  // The whole image before and after the dispatch.
  std::vector<uint8_t> before;
  std::vector<uint8_t> after;
};

class DbiLogRecordSimBase : public ::testing::Test {
protected:
  explicit DbiLogRecordSimBase(const SimTarget &t) : t_(t) {}

  struct PatchedKernel {
    std::vector<uint32_t> text;
    uint32_t scratch = 0;
    uint64_t entry = 0;
  };

  // Every case's kernels allocate 32 VGPRs (all ordinary on CDNA) and 48 SGPRs,
  // enough for the probe's declared footprint, and leave all of it dead at
  // their anchors.
  void patch(const std::vector<uint32_t> &text, const std::vector<uint64_t> &anchors,
             bool force_full_exec, PatchedKernel &out) {
    const auto target = test::make_kernarg_kernel_elf(
        text, /*private_bytes=*/0, t_.e_flags, kGuestKernargSize, t_.wave32,
        /*granulated_sgpr_count=*/5, /*entry_text_offset=*/0, /*granulated_vgpr_count=*/3,
        /*accum_offset=*/t_.wave32 ? 0 : 7);
    const auto probe = test::make_amdgpu_probe_elf(
        kProbeSymbol, test::build_log_record_probe_body(t_.arch, kRecordType, kSlotCount),
        t_.e_flags);
    AmdGpuCodeObject obj(target.data(), target.size());
    AmdGpuCodeObject probe_obj(probe.data(), probe.size());
    ASSERT_TRUE(obj.is_valid());
    ASSERT_TRUE(probe_obj.is_valid());

    Instrumentor instr(obj, t_.arch);
    for (const uint64_t anchor : anchors) {
      InstrumentationPoint pt;
      pt.anchor_offset = anchor;
      pt.probe_obj = &probe_obj;
      pt.probe_symbol = kProbeSymbol;
      pt.probe_args = test::log_record_probe_args(anchor, t_.wave32);
      pt.force_full_exec = force_full_exec;
      instr.add_point(pt);
    }

    const auto result = instr.patch_with_debug_summaries();
    ASSERT_TRUE(result.errors.empty()) << result.errors.front();
    ASSERT_EQ(result.patches.size(), anchors.size());

    AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
    ASSERT_TRUE(patched.is_valid());
    out.text = test::section_words(patched, ".text");
    ASSERT_FALSE(out.text.empty());
    out.scratch = test::patched_private_segment_size(patched);
    const auto entry = test::patched_entry_text_offset(patched);
    ASSERT_TRUE(entry.has_value());
    out.entry = *entry;
  }

  void run(const PatchedKernel &kernel, Drained &out, uint64_t buffer_addr = kBufferAddr) {
    const std::vector<uint8_t> wrapper = make_wrapper_image(buffer_addr);
    ASSERT_FALSE(wrapper.empty()) << "could not build the kernarg wrapper";
    auto buffer = LogBufferTestAccess::create(kSlotCount);
    ASSERT_NE(buffer, nullptr);
    // A new buffer is all zeros, which would let a field the probe never writes
    // pass as one it wrote as zero. Fill the slots with a pattern, then let
    // reinitialize() clear what a buffer handed to a dispatch starts with: the
    // counters and every valid flag.
    std::memset(LogBufferTestAccess::records(*buffer), 0xA5, kSlotCount * sizeof(RjLogRecord));
    buffer->reinitialize();
    const std::span<uint8_t> image = LogBufferTestAccess::bytes(*buffer);
    out.before.assign(image.begin(), image.end());

    test::DbiSim sim(t_.sim_arch, t_.wave_size);
    sim.set_kernarg(wrapper);
    sim.set_entry_offset(kernel.entry);
    sim.write_memory(buffer_addr, image);
    const std::optional<std::vector<uint8_t>> bytes =
        sim.run_and_read_memory(kernel.text, kernel.scratch, buffer_addr, image.size());
    ASSERT_TRUE(bytes.has_value()) << "kernel did not run to completion";
    ASSERT_EQ(bytes->size(), image.size());
    out.after = *bytes;

    std::ranges::copy(*bytes, image.begin());
    out.header = *LogBufferTestAccess::header(*buffer);
    std::string err;
    ASSERT_TRUE(buffer->validate(&err)) << err;
    out.stats = buffer->drain([&](const RjLogRecord &record) { out.records.push_back(record); });
  }

  // A kernel that sets EXEC to @p mask and reaches its anchor, the s_nop at
  // kMaskedAnchor, under it.
  std::vector<uint32_t> masked_kernel(uint64_t mask) const {
    const uint16_t exec_lo = scalar_operand_exec_lo(t_.arch);
    return {build_s_mov_b32(exec_lo, test::kScalarSrcLiteral, t_.arch),
            static_cast<uint32_t>(mask),
            build_s_mov_b32(exec_lo + 1, test::kScalarSrcLiteral, t_.arch),
            static_cast<uint32_t>(mask >> 32),
            build_s_nop(0, t_.arch),
            build_s_endpgm(t_.arch)};
  }
  static constexpr uint64_t kMaskedAnchor = 16;

  // The anchor mask the probe records for one full wave at the patch-time wave
  // size. RDNA4 is Wave32 there, so only exec_lo reaches the record even though
  // DbiSim runs it as Wave64.
  uint64_t launch_mask() const { return t_.wave_size == 64 ? ~uint64_t{0} : 0xFFFFFFFFull; }

  uint64_t partial_mask() const { return t_.wave32 ? kWave32PartialMask : kWave64PartialMask; }

  static void expect_record(const RjLogRecord &record, uint64_t exec_mask, uint32_t site,
                            uint32_t active_lanes, uint32_t writer_lane) {
    EXPECT_EQ(record.valid, 1u);
    EXPECT_EQ(record.abi_version, kRjLogAbiVersion);
    EXPECT_EQ(record.reserved0, 0u);
    EXPECT_EQ(record.record_size, kRjLogRecordSize);
    EXPECT_EQ(record.record_type, kRecordType);
    EXPECT_EQ(record.exec_mask, exec_mask);
    EXPECT_EQ(record.site, site);
    EXPECT_EQ(record.active_lane_count, active_lanes);
    EXPECT_EQ(record.writer_lane, writer_lane);
    // Identity is not filled in yet.
    EXPECT_EQ(record.wave_id, 0u);
    EXPECT_EQ(record.workgroup_x, 0u);
    EXPECT_EQ(record.workgroup_y, 0u);
    EXPECT_EQ(record.workgroup_z, 0u);
    EXPECT_EQ(record.reserved1, 0u);
    EXPECT_EQ(record.payload, 0u);
  }

  void expect_one_record_under_the_launch_mask() {
    PatchedKernel kernel;
    ASSERT_NO_FATAL_FAILURE(
        patch({build_s_nop(0, t_.arch), build_s_nop(0, t_.arch), build_s_endpgm(t_.arch)},
              {/*anchor_offset=*/4}, /*force_full_exec=*/true, kernel));
    Drained drained;
    ASSERT_NO_FATAL_FAILURE(run(kernel, drained));
    EXPECT_EQ(drained.header.write_ptr, 1u);
    EXPECT_EQ(drained.stats.records_drained, 1u);
    EXPECT_EQ(drained.stats.invalid_slots, 0u);
    ASSERT_EQ(drained.records.size(), 1u);
    expect_record(drained.records[0], launch_mask(), /*site=*/4, t_.wave_size, /*writer_lane=*/0);
  }

  void expect_partial_guest_mask_recorded() {
    PatchedKernel kernel;
    ASSERT_NO_FATAL_FAILURE(
        patch(masked_kernel(partial_mask()), {kMaskedAnchor}, /*force_full_exec=*/true, kernel));
    Drained drained;
    ASSERT_NO_FATAL_FAILURE(run(kernel, drained));
    EXPECT_EQ(drained.header.write_ptr, 1u);
    ASSERT_EQ(drained.records.size(), 1u);
    expect_record(drained.records[0], partial_mask(), kMaskedAnchor, kPartialMaskLanes,
                  t_.wave32 ? kWave32PartialMaskFirstLane : kWave64PartialMaskFirstLane);
  }

  void expect_zero_mask_recorded() {
    PatchedKernel kernel;
    ASSERT_NO_FATAL_FAILURE(
        patch(masked_kernel(0), {kMaskedAnchor}, /*force_full_exec=*/true, kernel));
    Drained drained;
    ASSERT_NO_FATAL_FAILURE(run(kernel, drained));
    EXPECT_EQ(drained.header.write_ptr, 1u);
    ASSERT_EQ(drained.records.size(), 1u);
    expect_record(drained.records[0], /*exec_mask=*/0, kMaskedAnchor, /*active_lanes=*/0,
                  kNoWriterLane);
  }

  // Negative control. Under the guest's zero mask every load and store the
  // probe issues is switched off, so not one byte of the buffer changes. The
  // buffer sits at address 0 here because the simulator's v_readfirstlane
  // returns 0 under EXEC == 0 (hardware reads lane 0): either way the probe's
  // base is the buffer, so a store that escaped the mask would land in the image.
  void expect_record_lost_under_the_guests_zero_mask() {
    PatchedKernel kernel;
    ASSERT_NO_FATAL_FAILURE(
        patch(masked_kernel(0), {kMaskedAnchor}, /*force_full_exec=*/false, kernel));
    Drained drained;
    ASSERT_NO_FATAL_FAILURE(run(kernel, drained, kControlBufferAddr));
    EXPECT_EQ(drained.header.write_ptr, 0u);
    EXPECT_TRUE(drained.records.empty());
    EXPECT_EQ(drained.after, drained.before);
  }

  // One site inside a loop records every time it runs, each record claiming
  // the next slot. One trip per slot, so the last record fills the ring.
  void expect_every_execution_of_a_site_recorded() {
    constexpr uint16_t kCounter = 20; // clear of the probe's registers; live at the anchor
    constexpr uint32_t kTrips = kSlotCount;
    // s20 counts down from kTrips; s_cbranch_scc1 returns to the anchor, the
    // s_nop at 4, while it is nonzero.
    const std::vector<uint32_t> text{
        build_s_mov_b32(kCounter, scalar_positive_inline_u32(kTrips), t_.arch),
        build_s_nop(0, t_.arch),
        build_s_add_u32(kCounter, kCounter, scalar_inline_neg_one(t_.arch), t_.arch),
        build_s_cmp_lg_u32(kCounter, scalar_positive_inline_u32(0), t_.arch),
        build_s_cbranch_scc1(/*back to the anchor=*/-4, t_.arch),
        build_s_endpgm(t_.arch)};
    PatchedKernel kernel;
    ASSERT_NO_FATAL_FAILURE(patch(text, {/*anchor_offset=*/4}, /*force_full_exec=*/true, kernel));
    Drained drained;
    ASSERT_NO_FATAL_FAILURE(run(kernel, drained));
    EXPECT_EQ(drained.header.write_ptr, kTrips);
    ASSERT_EQ(drained.records.size(), kTrips);
    for (const RjLogRecord &record : drained.records)
      expect_record(record, launch_mask(), /*site=*/4, t_.wave_size, /*writer_lane=*/0);
  }

  // Two sites share one copied body; each passes its own offset, and the
  // records land in execution order.
  void expect_each_site_records_its_own_offset() {
    PatchedKernel kernel;
    ASSERT_NO_FATAL_FAILURE(patch({build_s_nop(0, t_.arch), build_s_nop(0, t_.arch),
                                   build_s_nop(0, t_.arch), build_s_endpgm(t_.arch)},
                                  {/*first=*/4, /*second=*/8}, /*force_full_exec=*/true, kernel));
    Drained drained;
    ASSERT_NO_FATAL_FAILURE(run(kernel, drained));
    EXPECT_EQ(drained.header.write_ptr, 2u);
    ASSERT_EQ(drained.records.size(), 2u);
    expect_record(drained.records[0], launch_mask(), /*site=*/4, t_.wave_size, /*writer_lane=*/0);
    expect_record(drained.records[1], launch_mask(), /*site=*/8, t_.wave_size, /*writer_lane=*/0);
  }

private:
  const SimTarget &t_;
};

class DbiCdna3LogRecordSim : public DbiLogRecordSimBase {
protected:
  DbiCdna3LogRecordSim() : DbiLogRecordSimBase(kCdna3) {}
};
class DbiCdna4LogRecordSim : public DbiLogRecordSimBase {
protected:
  DbiCdna4LogRecordSim() : DbiLogRecordSimBase(kCdna4) {}
};
class DbiRdna4LogRecordSim : public DbiLogRecordSimBase {
protected:
  DbiRdna4LogRecordSim() : DbiLogRecordSimBase(kRdna4) {}
};

TEST_F(DbiCdna3LogRecordSim, WritesOneRecordUnderTheLaunchMask) {
  expect_one_record_under_the_launch_mask();
}
TEST_F(DbiCdna3LogRecordSim, RecordsAPartialGuestMask) { expect_partial_guest_mask_recorded(); }
TEST_F(DbiCdna3LogRecordSim, WritesARecordWhenTheGuestMaskIsZero) { expect_zero_mask_recorded(); }
TEST_F(DbiCdna3LogRecordSim, UnderTheGuestsZeroMaskTheRecordIsLost) {
  expect_record_lost_under_the_guests_zero_mask();
}
TEST_F(DbiCdna3LogRecordSim, EachSiteRecordsItsOwnOffset) {
  expect_each_site_records_its_own_offset();
}
TEST_F(DbiCdna3LogRecordSim, RecordsEveryExecutionOfASite) {
  expect_every_execution_of_a_site_recorded();
}

TEST_F(DbiCdna4LogRecordSim, WritesOneRecordUnderTheLaunchMask) {
  expect_one_record_under_the_launch_mask();
}
TEST_F(DbiCdna4LogRecordSim, RecordsAPartialGuestMask) { expect_partial_guest_mask_recorded(); }
TEST_F(DbiCdna4LogRecordSim, WritesARecordWhenTheGuestMaskIsZero) { expect_zero_mask_recorded(); }
TEST_F(DbiCdna4LogRecordSim, UnderTheGuestsZeroMaskTheRecordIsLost) {
  expect_record_lost_under_the_guests_zero_mask();
}
TEST_F(DbiCdna4LogRecordSim, EachSiteRecordsItsOwnOffset) {
  expect_each_site_records_its_own_offset();
}
TEST_F(DbiCdna4LogRecordSim, RecordsEveryExecutionOfASite) {
  expect_every_execution_of_a_site_recorded();
}

TEST_F(DbiRdna4LogRecordSim, WritesOneRecordUnderTheLaunchMask) {
  expect_one_record_under_the_launch_mask();
}
TEST_F(DbiRdna4LogRecordSim, RecordsAPartialGuestMask) { expect_partial_guest_mask_recorded(); }
TEST_F(DbiRdna4LogRecordSim, WritesARecordWhenTheGuestMaskIsZero) { expect_zero_mask_recorded(); }
TEST_F(DbiRdna4LogRecordSim, UnderTheGuestsZeroMaskTheRecordIsLost) {
  expect_record_lost_under_the_guests_zero_mask();
}
TEST_F(DbiRdna4LogRecordSim, EachSiteRecordsItsOwnOffset) {
  expect_each_site_records_its_own_offset();
}
TEST_F(DbiRdna4LogRecordSim, RecordsEveryExecutionOfASite) {
  expect_every_execution_of_a_site_recorded();
}

} // namespace
} // namespace rocjitsu
