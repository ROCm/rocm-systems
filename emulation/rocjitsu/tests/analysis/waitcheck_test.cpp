// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "../amdhsa_metadata_test_util.h"
#include "../tools/waitcheck_fixture.h"
#include "decode_test_util.h"
#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/analysis/def_use_chain.h"
#include "rocjitsu/code/analysis/indirect_branch_discovery.h"
#include "rocjitsu/code/analysis/waitcheck.h"
#include "rocjitsu/code/analysis/waitcheck/byte_interval.h"
#include "rocjitsu/code/basic_block.h"
#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/code/builders/spill_builders.h"
#include "rocjitsu/code/code_object.h"

#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/shared/wait_counter_policy.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstring>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace {

using namespace rocjitsu;

class TestTextSection : public Section {
public:
  TestTextSection(std::unique_ptr<char[]> data, std::size_t size)
      : Section(".text", std::move(data)), size_(size) {}

  std::size_t size() const override { return size_; }
  uint32_t sectionHeaderNameIdx() const override { return 0; }
  uint64_t sectionOffset() const override { return 0; }

private:
  std::size_t size_;
};

class TestCodeObject : public CodeObject {
public:
  explicit TestCodeObject(const std::vector<uint32_t> &words) {
    const auto byte_size = words.size() * sizeof(uint32_t);
    image_.resize(byte_size);
    std::memcpy(image_.data(), words.data(), byte_size);

    auto data = std::make_unique<char[]>(byte_size);
    std::memcpy(data.get(), words.data(), byte_size);
    sections_.push_back(std::make_unique<TestTextSection>(std::move(data), byte_size));
    text_sections_.push_back(sections_.back().get());
  }
};

template <typename T> void append_inst(std::vector<uint32_t> &words, const T &inst) {
  static_assert(sizeof(T) % sizeof(uint32_t) == 0);
  std::array<uint32_t, sizeof(T) / sizeof(uint32_t)> encoded =
      std::bit_cast<std::array<uint32_t, sizeof(T) / sizeof(uint32_t)>>(inst);
  words.reserve(words.size() + encoded.size());
  for (uint32_t word : encoded)
    words.push_back(word);
}

[[nodiscard]] rdna4::SoppMachineInst sopp(uint32_t op, uint32_t simm16 = 0) {
  rdna4::SoppMachineInst inst{};
  inst.encoding = 0x17F;
  inst.op = op;
  inst.simm16 = simm16;
  return inst;
}

[[nodiscard]] rdna4::SoppMachineInst s_endpgm() { return sopp(48, 0); }

[[nodiscard]] rdna4::SoppMachineInst s_wait_idle() { return sopp(10, 0); }

[[nodiscard]] rdna4::SoppMachineInst s_set_vgpr_msb(uint32_t mode) { return sopp(6, mode); }

[[nodiscard]] constexpr uint32_t hwreg(uint32_t id, uint32_t offset, uint32_t size) {
  return (id & 0x3fu) | ((offset & 0x1fu) << 6u) | (((size - 1u) & 0x1fu) << 11u);
}

[[nodiscard]] rdna4::SopkInstLiteralMachineInst s_setreg_imm32_b32(uint32_t reg, uint32_t value) {
  rdna4::SopkInstLiteralMachineInst inst{};
  inst.encoding = 0xb;
  inst.op = 19;
  inst.simm16 = reg;
  inst.simm32 = value;
  return inst;
}

[[nodiscard]] constexpr uint32_t vgpr_msb_mode(uint32_t src0, uint32_t src1, uint32_t src2,
                                               uint32_t dst) {
  return (src0 & 0x3u) | ((src1 & 0x3u) << 2u) | ((src2 & 0x3u) << 4u) | ((dst & 0x3u) << 6u);
}

[[nodiscard]] std::string access_name(WaitcheckAccessKind access) {
  switch (access) {
  case WaitcheckAccessKind::Use:
    return "use";
  case WaitcheckAccessKind::Def:
    return "def";
  case WaitcheckAccessKind::MemoryOrder:
    return "memory-order";
  case WaitcheckAccessKind::ProgramEnd:
    return "program-end";
  case WaitcheckAccessKind::ControlTransfer:
    return "control-transfer";
  }
  return "unknown";
}

[[nodiscard]] std::string diagnostic_summary(const WaitcheckReport &report) {
  std::string result;
  for (const auto &diag : report.diagnostics) {
    result += '\n';
    result += std::string(wait_counter_name(diag.counter));
    result += '/';
    result += access_name(diag.access);
    result += " required=";
    result += std::to_string(diag.required_count);
    result += " at ";
    result += diag.instruction;
    result += " from ";
    result += diag.producer_instruction;
    if (!diag.message.empty()) {
      result += ": ";
      result += diag.message;
    }
  }
  return result;
}

[[nodiscard]] WaitcheckReport analyze_gfx1250_normal(const std::vector<uint32_t> &program) {
  TestCodeObject code_object(program);
  return analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_CDNA5);
}

constexpr std::array<uint32_t, 4> kGfx1250ScaledWmma{
    0xcc3a0200u,
    0x42020d04u,
    0xcc336008u,
    0x04223110u,
};

TEST(WaitcheckTest, ByteIntervalsRequireACommonBoundedBase) {
  using namespace waitcheck_detail;
  constexpr auto a = ByteInterval::within(4, 8, 16);
  EXPECT_EQ(compare_intervals(a, ByteInterval::within(4, 8, 16)), IntervalRelation::Identical);
  EXPECT_EQ(compare_intervals(a, ByteInterval::within(8, 4, 16)), IntervalRelation::Overlap);
  EXPECT_EQ(compare_intervals(a, ByteInterval::within(0, 8, 16)), IntervalRelation::Overlap);
  EXPECT_EQ(compare_intervals(a, ByteInterval::within(12, 4, 16)), IntervalRelation::Disjoint);
  EXPECT_EQ(compare_intervals(a, ByteInterval::within(0, 4, 16)), IntervalRelation::Disjoint);
  EXPECT_EQ(compare_intervals(a, std::nullopt), IntervalRelation::Unknown);
  EXPECT_EQ(compare_intervals(a, ByteInterval::within(UINT32_MAX - 3, 4, UINT32_MAX)),
            IntervalRelation::Unknown);
}

WaitcheckReport check_gfx942_kernarg(std::vector<uint32_t> words, uint32_t size = 16,
                                     uint16_t properties = 8,
                                     std::initializer_list<test::MetadataKernel> metadata = {}) {
  words.push_back(0xbf8cc07f);  // s_waitcnt lgkmcnt(0)
  words.push_back(0xBF810000u); // s_endpgm
  auto image = waitcheck_test::make_gfx_code_object(words, EF_AMDGPU_MACH_AMDGCN_GFX942);
  Elf64_Ehdr header{};
  std::memcpy(&header, image.data(), sizeof(header));
  Elf64_Shdr rodata{};
  std::memcpy(&rodata, image.data() + header.e_shoff + 2 * sizeof(rodata), sizeof(rodata));
  const size_t kd = rodata.sh_offset;
  std::memcpy(image.data() + kd + offsetof(amdhsa::kernel_descriptor_t, kernarg_size), &size,
              sizeof(size));
  std::memcpy(image.data() + kd + offsetof(amdhsa::kernel_descriptor_t, kernel_code_properties),
              &properties, sizeof(properties));
  if (metadata.size())
    image = test::with_resource_metadata(image, metadata);
  const AmdGpuCodeObject object(image.data(), image.size());
  return analyze_waitcnts(object, ROCJITSU_CODE_ARCH_CDNA3);
}

TEST(WaitcheckTest, Gfx942KernargMetadataUsesTheUniqueDescriptorSymbol) {
  constexpr uint32_t load = 0xc0020100; // s_load_dword s4, s[0:1], imm
  const test::MetadataKernel first{"source", 16, 16, 4, 0, "waitcheck.kd"};
  const test::MetadataKernel other{"alias", 64, 16, 4, 0, "waitcheck.kd"};
  const test::MetadataKernel unrelated{"waitcheck", 64, 16, 4, 0, "other.kd"};
  const auto check = [&](std::initializer_list<test::MetadataKernel> entries, size_t expected) {
    const auto report = check_gfx942_kernarg({load, 4, load, 4}, 0, 8, entries);
    ASSERT_TRUE(report.complete && report.supported) << report.analysis_error;
    EXPECT_EQ(report.diagnostics.size(), expected);
  };
  check({first}, 0);
  check({first, unrelated}, 0);
  check({first, other}, 1);
  check({unrelated}, 1);
  const auto bounded = check_gfx942_kernarg({load, 12, load, 12}, 8, 8, {first});
  ASSERT_TRUE(bounded.complete && bounded.supported) << bounded.analysis_error;
  EXPECT_EQ(bounded.diagnostics.size(), 1u);
}

TEST(WaitcheckTest, Gfx942IdenticalKernargLoadsOnlyExemptTheirResultOverwrite) {
  constexpr uint32_t load_s4 = 0xc0020100; // s_load_dword s4, s[0:1], imm
  auto report = check_gfx942_kernarg({0x8601ff01, 0xffff, load_s4, 4, load_s4, 4});
  ASSERT_TRUE(report.supported && report.complete) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
  report = check_gfx942_kernarg({load_s4, 4, load_s4, 4, 0xbe880004}); // read s4 before wait
  ASSERT_TRUE(report.supported && report.complete) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 2u) << diagnostic_summary(report);
  for (const auto &d : report.diagnostics) {
    EXPECT_EQ(d.access, WaitcheckAccessKind::Use);
    EXPECT_EQ(d.section_offset, 16u);
  }
}

TEST(WaitcheckTest, Gfx942IdenticalKernargBytesMustMapToTheSameSgprs) {
  // The two x8 loads map different argument dwords into the shared s[8:11] lanes.
  const auto report = check_gfx942_kernarg({0xc00e0100, 0, 0xc00e0200, 0}, 32);
  ASSERT_TRUE(report.complete && report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u);
  const auto &diagnostic = report.diagnostics.front();
  EXPECT_EQ(diagnostic.access, WaitcheckAccessKind::Def);
  EXPECT_EQ(diagnostic.reg.index, 8u);
  EXPECT_EQ(diagnostic.section_offset, 8u);
}

TEST(WaitcheckTest, Gfx942KernargProofRejectsABackedgeThatCanClobberTheBase) {
  constexpr uint32_t load = 0xc0020100;
  const auto report = check_gfx942_kernarg({load, 0, load, 0, 0xbf8cc07f, 0xbe800002, 0xbf84fff9});
  ASSERT_TRUE(report.complete && report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u);
  EXPECT_EQ(report.diagnostics[0].producer_section_offset, 0u);
  EXPECT_EQ(report.diagnostics[0].section_offset, 8u);
}

TEST(WaitcheckTest, CdnaMemoryResultsRespectBothDescriptorBanksAndAliasLimits) {
  constexpr uint32_t load = 0xe0501000, load_x4 = 0xe05c1000;
  constexpr uint32_t dst_v224 = 0x2304e008, read_v224 = 0x7e0003e0;
  const auto check = [&](std::vector<uint32_t> words, unsigned first, unsigned second) {
    words.push_back(0xbf8c0f70); // s_waitcnt vmcnt(0)
    words.push_back(0xbf810000); // s_endpgm
    auto image = waitcheck_test::make_gfx_multi_kernel_code_object(
        {{"first", words}, {"alias", words}}, EF_AMDGPU_MACH_AMDGCN_GFX942);
    Elf64_Ehdr header{};
    std::memcpy(&header, image.data(), sizeof(header));
    Elf64_Shdr rodata{};
    std::memcpy(&rodata, image.data() + header.e_shoff + 2 * sizeof(rodata), sizeof(rodata));
    for (size_t i = 0; i < 2; ++i) {
      const auto offset = rodata.sh_offset + i * sizeof(amdhsa::kernel_descriptor_t);
      amdhsa::kernel_descriptor_t kd{};
      std::memcpy(&kd, image.data() + offset, sizeof(kd));
      kd.compute_pgm_rsrc1 = 31; // 256 total VGPRs
      kd.compute_pgm_rsrc3 = (i ? second : first) / 4 - 1;
      if (i)
        kd.kernel_code_entry_byte_offset -= words.size() * sizeof(uint32_t);
      std::memcpy(image.data() + offset, &kd, sizeof(kd));
    }
    const AmdGpuCodeObject object(image.data(), image.size());
    const auto report = analyze_waitcnts(object, ROCJITSU_CODE_ARCH_CDNA3);
    EXPECT_TRUE(report.supported && report.complete) << report.analysis_error;
    if (first == 160 && second == 256) {
      const auto &text = *object.text_sections().front();
      const auto descriptors = scan_kernel_descriptors(image, text.sectionOffset(), text.size(),
                                                        text.sectionHeaderIndex());
      EXPECT_EQ(descriptors.size(), 2u);
      std::vector<size_t> exact_counts;
      for (const auto &descriptor : descriptors) {
        const auto exact = waitcheck_kernel_from_descriptor(
            descriptor.kernel_name, descriptor.descriptor_vaddr, descriptor.entry_text_offset,
            0, ROCJITSU_CODE_ARCH_CDNA3, descriptor.descriptor);
        const auto exact_report =
            analyze_waitcnts_for_kernel(object, ROCJITSU_CODE_ARCH_CDNA3, exact);
        EXPECT_TRUE(exact_report.supported && exact_report.complete) << exact_report.analysis_error;
        exact_counts.push_back(exact_report.diagnostics.size());
      }
      EXPECT_EQ(exact_counts, (std::vector<size_t>{0, 1}));
    }
    return report.diagnostics.size();
  };
  // A November AITER load into v224 needs 225 ordinary VGPRs. Either
  // descriptor can select the same code, so the larger bank remains possible.
  EXPECT_EQ(check({load, dst_v224, read_v224}, 160, 160), 0u);
  EXPECT_EQ(check({load, dst_v224, read_v224}, 160, 256), 1u);
  EXPECT_EQ(check({load_x4, 0x23049c08, 0x7e00039c}, 160, 160), 1u); // v[156:159]
  EXPECT_EQ(check({load_x4, 0x23049e08, 0x7e00039e}, 160, 160), 0u); // v[158:161]
  EXPECT_EQ(check({load, 0x23845f08, 0xd3d84000, 0x1800015f}, 160, 160), 1u); // acc95
  EXPECT_EQ(check({load, 0x23846008, 0xd3d84000, 0x18000160}, 160, 160), 0u); // acc96
  // A nullified load still occupies an ordered VM counter slot for the
  // preceding load into v156. A partial wait retires that older result.
  EXPECT_EQ(check({load, 0x23049c08, load, dst_v224, 0xbf8c0f71, 0x7e00039c}, 160, 160), 0u);
  const std::array<uint32_t, 5> raw{load, dst_v224, read_v224, 0xbf8c0f70, 0xbf810000};
  EXPECT_EQ(analyze_waitcnts(raw, ROCJITSU_CODE_ARCH_CDNA3).diagnostics.size(), 1u);
}

TEST(WaitcheckTest, Gfx942KernargProofRequiresAnUnambiguousEntryAbi) {
  const std::vector<uint32_t> words{0xc0020100, 0x38, 0xc0020100, 0x38, 0xbf8cc07f, 0xbf810000};
  auto image = waitcheck_test::make_gfx_multi_kernel_code_object(
      {{"first", words}, {"alias", words}}, EF_AMDGPU_MACH_AMDGCN_GFX942);
  Elf64_Ehdr header{};
  std::memcpy(&header, image.data(), sizeof(header));
  Elf64_Shdr rodata{};
  std::memcpy(&rodata, image.data() + header.e_shoff + 2 * sizeof(rodata), sizeof(rodata));
  for (size_t i = 0; i < 2; ++i) {
    const auto offset = rodata.sh_offset + i * sizeof(amdhsa::kernel_descriptor_t);
    amdhsa::kernel_descriptor_t descriptor{};
    std::memcpy(&descriptor, image.data() + offset, sizeof(descriptor));
    descriptor.kernarg_size = 64;
    descriptor.kernel_code_properties =
        amdhsa::KERNEL_CODE_PROPERTY_ENABLE_SGPR_KERNARG_SEGMENT_PTR;
    if (i == 1) {
      // This descriptor puts the queue pointer in s[0:1] and moves kernarg to s[2:3].
      descriptor.kernel_code_properties |= amdhsa::KERNEL_CODE_PROPERTY_ENABLE_SGPR_QUEUE_PTR;
      descriptor.kernel_code_entry_byte_offset -= words.size() * sizeof(uint32_t);
    }
    std::memcpy(image.data() + offset, &descriptor, sizeof(descriptor));
  }
  const AmdGpuCodeObject object(image.data(), image.size());
  const auto kernels = waitcheck_kernels(object);
  ASSERT_EQ(kernels.size(), 1u);
  ASSERT_EQ(kernels.front().alias_descriptor_vaddrs.size(), 1u);
  for (const auto &report :
       {analyze_waitcnts(object, ROCJITSU_CODE_ARCH_CDNA3),
        analyze_waitcnts_for_kernel(object, ROCJITSU_CODE_ARCH_CDNA3, kernels.front())}) {
    ASSERT_TRUE(report.complete && report.supported) << report.analysis_error;
    ASSERT_EQ(report.diagnostics.size(), 1u);
    EXPECT_EQ(report.diagnostics[0].section_offset, 8u);
    EXPECT_EQ(report.diagnostics[0].reg.index, 4u);
  }
}

TEST(WaitcheckTest, Gfx942KernargProofStopsAtTheBackwardSearchLimit) {
  constexpr uint32_t load = 0xc0020100;
  for (const unsigned padding : {31u, 32u}) {
    std::vector<uint32_t> words(padding, 0xbf800000u); // s_nop 0
    words.insert(words.end(), {load, 4, load, 4});
    const auto report = check_gfx942_kernarg(words);
    ASSERT_TRUE(report.complete && report.supported) << report.analysis_error;
    EXPECT_EQ(report.diagnostics.size(), padding == 31 ? 0u : 1u);
  }
}

TEST(WaitcheckTest, Gfx942DistinctOrUnknownKernargLoadsKeepOverwriteWaits) {
  constexpr uint32_t load = 0xc0020100; // s_load_dword s4, s[0:1], imm
  const std::vector<std::vector<uint32_t>> programs{
      {load, 4, load, 8},                               // disjoint byte interval, same SGPR
      {0xc0060100, 4, load, 8},                         // partially overlapping input/destination
      {load, 0, load + 1, 0},                           // second SMEM base is s[2:3]
      {load, 0, 0xbe800002, load, 0},                   // overwrite the low kernarg SGPR
      {0x8601ff01, 0xfffe, load, 0, load, 0},           // mask changes a translated address bit
      {0x8601ff02, 0xffff, load, 0, load, 0},           // mask source is a different SGPR
      {load, 16, load, 16},                             // first byte beyond the argument segment
      {0xc0060100, 12, 0xc0060100, 12},                 // last dword falls outside
      {load, 0x1ffffc, load, 0x1ffffc},                 // signed offset -4
      {0xc0024100, 0x04000000, 0xc0024100, 0x04000000}, // s2 + immediate offset
      {0xbf840001, 0xbe800002, load, 0, load, 0},       // changed and unchanged bases join
      {load, 0, load, 0, 0xbe800002, 0xbf84fffa},       // backedge with a changed base
  };
  for (const auto &words : programs) {
    const auto report = check_gfx942_kernarg(words);
    ASSERT_TRUE(report.supported && report.complete) << report.analysis_error;
    EXPECT_FALSE(report.diagnostics.empty()) << std::hex << words.front();
  }
  for (const auto &[size, properties] :
       std::array<std::pair<uint32_t, uint16_t>, 2>{{{0, 8}, {16, 0}}}) {
    const auto report = check_gfx942_kernarg({load, 4, load, 4}, size, properties);
    ASSERT_TRUE(report.supported && report.complete) << report.analysis_error;
    EXPECT_EQ(report.diagnostics.size(), 1u);
  }
}

TEST(WaitcheckTest, ReachableGfx1250DecodePreservesFourWordInstruction) {
  std::vector<uint32_t> program(kGfx1250ScaledWmma.begin(), kGfx1250ScaledWmma.end());
  program.push_back(0xbfb00000u); // s_endpgm.
  TestCodeObject code_object(program);
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  const std::array<uint64_t, 1> entries{0};
  auto blocks =
      BasicBlock::build_reachable(code_object, *decoder, ROCJITSU_CODE_ARCH_CDNA5, entries);
  ASSERT_TRUE(blocks.succeeded());

  ASSERT_FALSE(blocks.value().empty());
  const Instruction &instruction = *blocks.value().front()->instructions().begin();
  ASSERT_EQ(instruction.size(), 4 * sizeof(uint32_t));
  ASSERT_EQ(instruction.num_dst_operands(), 1);
  ASSERT_EQ(instruction.num_src_operands(), 5);
  EXPECT_EQ(instruction.dst_operand(0)->to_register_ref(), (RegisterRef{RegClass::VGPR, 8, 8}));
  EXPECT_EQ(instruction.src_operand(0)->to_register_ref(), (RegisterRef{RegClass::VGPR, 16, 8}));
  EXPECT_EQ(instruction.src_operand(1)->to_register_ref(), (RegisterRef{RegClass::VGPR, 24, 8}));
  EXPECT_EQ(instruction.src_operand(2)->to_register_ref(), (RegisterRef{RegClass::VGPR, 8, 8}));
  ASSERT_NE(instruction.raw_encoding(), nullptr);
  EXPECT_EQ(instruction.raw_encoding()[3], program[3]);
}

[[nodiscard]] rdna4::Vop1MachineInst v_mov_b32(uint32_t vdst, uint32_t src_vgpr) {
  rdna4::Vop1MachineInst inst{};
  inst.encoding = 0x3F;
  inst.op = 1;
  inst.vdst = vdst;
  inst.src0 = 256 + src_vgpr;
  return inst;
}

[[nodiscard]] rdna4::Vop1MachineInst v_sqrt_f32(uint32_t vdst, uint32_t src_vgpr) {
  rdna4::Vop1MachineInst inst{};
  inst.encoding = 0x3F;
  inst.op = 51;
  inst.vdst = vdst;
  inst.src0 = 256 + src_vgpr;
  return inst;
}

[[nodiscard]] rdna4::Vop2MachineInst v_add_f32_e32(uint32_t vdst, uint32_t src0, uint32_t vsrc1) {
  auto inst = std::bit_cast<rdna4::Vop2MachineInst>(0x06000000U);
  inst.vdst = vdst;
  inst.src0 = src0;
  inst.vsrc1 = vsrc1;
  return inst;
}

[[nodiscard]] rdna4::Vop2MachineInst v_cndmask_b32_e32(uint32_t vdst, uint32_t src0,
                                                       uint32_t vsrc1) {
  auto inst = std::bit_cast<rdna4::Vop2MachineInst>(0x02000000U);
  inst.vdst = vdst;
  inst.src0 = src0;
  inst.vsrc1 = vsrc1;
  return inst;
}

[[nodiscard]] rdna4::VopcMachineInst v_cmp_gt_u32_e32(uint32_t src0, uint32_t vsrc1) {
  auto inst = std::bit_cast<rdna4::VopcMachineInst>(0x7C980000U);
  inst.src0 = src0;
  inst.vsrc1 = vsrc1;
  return inst;
}

[[nodiscard]] rdna4::Sop1MachineInst s_mov_b32(uint32_t sdst, uint32_t ssrc0) {
  auto inst = std::bit_cast<rdna4::Sop1MachineInst>(0xBE800000U);
  inst.sdst = sdst;
  inst.ssrc0 = ssrc0;
  return inst;
}

[[nodiscard]] rdna4::Sop1MachineInst s_or_saveexec_b32(uint32_t sdst, uint32_t ssrc0) {
  auto inst = std::bit_cast<rdna4::Sop1MachineInst>(0xBE802200U);
  inst.sdst = sdst;
  inst.ssrc0 = ssrc0;
  return inst;
}

[[nodiscard]] rdna4::Sop1MachineInst s_mov_b64_exec_from_s0() {
  auto inst = std::bit_cast<rdna4::Sop1MachineInst>(0xBE800100U);
  inst.sdst = 126; // exec_lo
  inst.ssrc0 = 0;
  return inst;
}

[[nodiscard]] cdna4::Sop2MachineInst s_or_b32_exec_lo(uint32_t ssrc1) {
  cdna4::Sop2MachineInst inst{};
  inst.encoding = 2;
  inst.op = 14;
  inst.sdst = 126;  // EXEC_LO
  inst.ssrc0 = 126; // EXEC_LO
  inst.ssrc1 = ssrc1;
  return inst;
}

[[nodiscard]] rdna4::Sop1MachineInst s_barrier_signal_isfirst(uint32_t barrier_id) {
  auto inst = std::bit_cast<rdna4::Sop1MachineInst>(0xBE804F00U);
  inst.ssrc0 = barrier_id + 128u;
  return inst;
}

[[nodiscard]] rdna4::Sop1MachineInst s_sendmsg_rtn_b32(uint32_t sdst) {
  auto inst = std::bit_cast<rdna4::Sop1MachineInst>(0xBE804C00U);
  inst.sdst = sdst;
  inst.ssrc0 = 1;
  return inst;
}

[[nodiscard]] rdna4::Sop1MachineInst s_sendmsg_rtn_b64(uint32_t sdst) {
  auto inst = std::bit_cast<rdna4::Sop1MachineInst>(0xBE804D00U);
  inst.sdst = sdst;
  inst.ssrc0 = 1;
  return inst;
}

[[nodiscard]] rdna4::Sop2MachineInst s_cselect_b32(uint32_t sdst) {
  auto inst = std::bit_cast<rdna4::Sop2MachineInst>(0x98000000U);
  inst.sdst = sdst;
  inst.ssrc0 = 129; // literal 1
  inst.ssrc1 = 128; // literal 0
  return inst;
}

[[nodiscard]] rdna4::SopcMachineInst s_cmp_eq_u32(uint32_t ssrc0, uint32_t ssrc1) {
  auto inst = std::bit_cast<rdna4::SopcMachineInst>(0xBF060000U);
  inst.ssrc0 = ssrc0;
  inst.ssrc1 = ssrc1;
  return inst;
}

[[nodiscard]] rdna4::SopcMachineInst s_cmp_ge_u32(uint32_t ssrc0, uint32_t ssrc1) {
  auto inst = std::bit_cast<rdna4::SopcMachineInst>(0xBF090000U);
  inst.ssrc0 = ssrc0;
  inst.ssrc1 = ssrc1;
  return inst;
}

[[nodiscard]] rdna4::SmemMachineInst s_load_b32(uint32_t sdata, uint32_t sbase) {
  auto inst =
      std::bit_cast<rdna4::SmemMachineInst>(std::array<uint32_t, 2>{0xF4000000U, 0xF8000000U});
  inst.sdata = sdata;
  inst.sbase = sbase;
  return inst;
}

void append_s_buffer_load_b128_s8_s12_imm48(std::vector<uint32_t> &program) {
  program.push_back(0xF4124206u);
  program.push_back(0xF8000030u);
}

void append_s_buffer_load_b64_s20_s12_imm64(std::vector<uint32_t> &program) {
  program.push_back(0xF4122506u);
  program.push_back(0xF8000040u);
}

void append_s_buffer_load_b128_s16_s8_imm32(std::vector<uint32_t> &program) {
  program.push_back(0xF4024404u);
  program.push_back(0xF8000020u);
}

void append_s_buffer_load_b128_s0_s8_imm0(std::vector<uint32_t> &program) {
  program.push_back(0xF4024004u);
  program.push_back(0xF8000000u);
}

void append_s_load_b64_s2_s16_imm196(std::vector<uint32_t> &program) {
  program.push_back(0xF4002088u);
  program.push_back(0xF80000C4u);
}

void append_v_cndmask_b32_e64_v6_0_1_s(std::vector<uint32_t> &program, uint32_t selector) {
  program.push_back(0xD5010006u);
  program.push_back(0x00010280u | (selector << 18u));
}

void append_v_sub_co_u32_v0_s0_v0_v8(std::vector<uint32_t> &program) {
  program.push_back(0xD7010000u);
  program.push_back(0x02021100u);
}

void append_v_sub_co_ci_u32_v1_null_v1_v12_s0(std::vector<uint32_t> &program) {
  program.push_back(0xD5217C01u);
  program.push_back(0x00021901u);
}

void append_v_div_scale_f32_v5_s2_s8_s9_s8(std::vector<uint32_t> &program) {
  program.push_back(0xD6FC0205u);
  program.push_back(0x00201208u);
}

void append_v_cmp_gt_u32_sdst_s5_v12(std::vector<uint32_t> &program, uint32_t sdst) {
  program.push_back(0xD44C0000u | sdst);
  program.push_back(0x02021805u);
}

void append_v_cmp_gt_u32_s2_s5_v12(std::vector<uint32_t> &program) {
  append_v_cmp_gt_u32_sdst_s5_v12(program, 2);
}

void append_v_mad_co_s2(std::vector<uint32_t> &program, uint16_t opcode) {
  const auto inst = rdna4::build_vop3_sdst_enc(
      opcode, {.vdst = 0, .sdst = 2, .src0 = 256, .src1 = 257, .src2 = 258});
  program.insert(program.end(), inst.begin(), inst.end());
}

void append_v_dual_cndmask_b32_v2_v1_v2_dual_mov_b32_v1_0(std::vector<uint32_t> &program) {
  program.push_back(0xCA500501u);
  program.push_back(0x02000080u);
}

void append_v_dual_add_nc_u32_v11_1_v10_cndmask_v8_v8_v12_s0(std::vector<uint32_t> &program) {
  program.push_back(0xCF409081u);
  program.push_back(0x000A0108u);
  program.push_back(0x08000C0Bu);
}

void append_v_cndmask_b32_v8_v8_v12_s0(std::vector<uint32_t> &program) {
  program.push_back(0xD5010008u);
  program.push_back(0x00021908u);
}

void append_v_cndmask_b16_v1_v1_v14_s2(std::vector<uint32_t> &program) {
  program.push_back(0xD65D0001u);
  program.push_back(0x000A1D01u);
}

void append_v_and_b16_v134_ff_v134(std::vector<uint32_t> &program) {
  program.push_back(0xD7620086u);
  program.push_back(0x02030CFFu);
  program.push_back(0x000000FFu);
}

void append_v_dual_lshlrev_b32_v17_2_v9_dual_mov_b32_v9_s11(std::vector<uint32_t> &program) {
  program.push_back(0xCF448082u);
  program.push_back(0x0009000Bu);
  program.push_back(0x09000011u);
}

void append_buffer_load_b128_v32_v7_s4_offen(std::vector<uint32_t> &program) {
  program.push_back(0xC405C07Cu);
  program.push_back(0x40800820u);
  program.push_back(0x00000007u);
}

[[nodiscard]] rdna4::VglobalMachineInst global_load_b32(uint32_t vdst, uint32_t vaddr = 8,
                                                        uint32_t saddr = 0) {
  rdna4::VglobalMachineInst inst{};
  inst.encoding = 0xEE;
  inst.op = 20;
  inst.vdst = vdst;
  inst.vaddr = vaddr;
  inst.saddr = saddr;
  return inst;
}

void append_global_load_b64_v8_v7_s0_offset16(std::vector<uint32_t> &program) {
  program.push_back(0xEE054080u);
  program.push_back(0x00000008u);
  program.push_back(0x00001007u);
}

void append_global_load_b128_v2_v7_s0(std::vector<uint32_t> &program) {
  program.push_back(0xEE05C080u);
  program.push_back(0x00000002u);
  program.push_back(0x00000007u);
}

void append_global_load_b128_v6_v137_s2(std::vector<uint32_t> &program) {
  program.push_back(0xEE05C002u);
  program.push_back(0x00000006u);
  program.push_back(0x00000089u);
}

void append_global_loads(std::vector<uint32_t> &program, uint32_t count, uint32_t first_vdst) {
  for (uint32_t i = 0; i < count; ++i)
    append_inst(program, global_load_b32(first_vdst + i));
}

[[nodiscard]] rdna4::VscratchMachineInst scratch_load_b32(uint32_t vdst) {
  auto inst = std::bit_cast<rdna4::VscratchMachineInst>(std::array<uint32_t, 3>{0xED050000U, 0, 0});
  inst.vdst = vdst;
  inst.vaddr = 8;
  inst.saddr = 0;
  return inst;
}

[[nodiscard]] rdna4::VscratchMachineInst scratch_load_b128_off(uint32_t vdst) {
  auto inst = std::bit_cast<rdna4::VscratchMachineInst>(std::array<uint32_t, 3>{0xED05C000U, 0, 0});
  inst.vdst = vdst;
  inst.vaddr = 0;
  inst.saddr = 64;
  return inst;
}

[[nodiscard]] rdna4::VglobalMachineInst global_store_b32(uint32_t vsrc, uint32_t vaddr = 8,
                                                         uint32_t saddr = 0) {
  auto inst = std::bit_cast<rdna4::VglobalMachineInst>(std::array<uint32_t, 3>{0xEE068000U, 0, 0});
  inst.vsrc = vsrc;
  inst.vaddr = vaddr;
  inst.saddr = saddr;
  return inst;
}

[[nodiscard]] rdna4::VglobalMachineInst global_atomic_add_u32(uint32_t vsrc, uint32_t vaddr = 8,
                                                              uint32_t saddr = 124,
                                                              bool returns_value = false) {
  auto inst = std::bit_cast<rdna4::VglobalMachineInst>(std::array<uint32_t, 3>{0xEE0D4000U, 0, 0});
  inst.vdst = vsrc;
  inst.vsrc = vsrc;
  inst.vaddr = vaddr;
  inst.saddr = saddr;
  inst.th = returns_value ? 1 : 0;
  return inst;
}

[[nodiscard]] cdna5::VbufferMachineInst buffer_atomic_add_u32(uint32_t vdata,
                                                               bool returns_value = true) {
  auto inst =
      std::bit_cast<cdna5::VbufferMachineInst>(std::array<uint32_t, 3>{0xC40D4000U, 0, 0});
  inst.vdata = vdata;
  inst.vaddr = 8;
  inst.rsrc = 40;
  inst.soffset = 124;
  inst.offen = 1;
  inst.th = returns_value ? 1 : 0;
  return inst;
}

[[nodiscard]] WaitcheckReport
analyze_gfx12_with_vgpr_allocation(const std::vector<uint32_t> &program, uint32_t machine,
                                   rj_code_arch_t arch, bool wave32, uint32_t vgprs,
                                   WaitcheckOptions options = {}) {
  auto image = waitcheck_test::make_gfx_code_object(program, machine, wave32,
                                                    /*allocation_alias=*/false);
  Elf64_Ehdr header{};
  std::memcpy(&header, image.data(), sizeof(header));
  Elf64_Shdr rodata{};
  std::memcpy(&rodata, image.data() + header.e_shoff + 2 * sizeof(rodata), sizeof(rodata));
  amdhsa::kernel_descriptor_t descriptor{};
  std::memcpy(&descriptor, image.data() + rodata.sh_offset, sizeof(descriptor));
  const uint32_t granule =
      descriptor_vgpr_count_granule_for_wavefront(arch, wave32 ? 32 : 64).value();
  descriptor.compute_pgm_rsrc1 =
      (descriptor.compute_pgm_rsrc1 & ~uint32_t{63}) | (vgprs / granule - 1);
  std::memcpy(image.data() + rodata.sh_offset, &descriptor, sizeof(descriptor));
  image = test::with_resource_metadata(image, {{"waitcheck", 0, 128, vgprs}});
  const AmdGpuCodeObject object(image.data(), image.size());
  return analyze_waitcnts_for_kernel(object, arch, 0, options);
}

[[nodiscard]] std::array<uint32_t, 2> s_prefetch_data() { return {0xF404C000U, 0}; }

void append_global_stores(std::vector<uint32_t> &program, uint32_t count, uint32_t first_vsrc) {
  for (uint32_t i = 0; i < count; ++i)
    append_inst(program, global_store_b32(first_vsrc + i));
}

[[nodiscard]] std::array<uint32_t, 3> global_inv() { return {0xEE0AC07CU, 0, 0}; }

[[nodiscard]] std::array<uint32_t, 3> global_wb() { return {0xEE0B007CU, 0, 0}; }

[[nodiscard]] std::array<uint32_t, 3> global_wbinv() { return {0xEE13C07CU, 0, 0}; }

[[nodiscard]] rdna4::VflatMachineInst flat_load_b32(uint32_t vdst) {
  auto inst = std::bit_cast<rdna4::VflatMachineInst>(std::array<uint32_t, 3>{0xEC050000U, 0, 0});
  inst.vdst = vdst;
  inst.vaddr = 8;
  return inst;
}

[[nodiscard]] rdna4::VflatMachineInst flat_store_b32(uint32_t vsrc) {
  auto inst = std::bit_cast<rdna4::VflatMachineInst>(std::array<uint32_t, 3>{0xEC068000U, 0, 0});
  inst.vsrc = vsrc;
  inst.vaddr = 8;
  return inst;
}

[[nodiscard]] rdna4::VdsMachineInst ds_load_b32(uint32_t vdst, uint32_t addr) {
  auto inst = std::bit_cast<rdna4::VdsMachineInst>(std::array<uint32_t, 2>{0xD8D80000U, 0});
  inst.vdst = vdst;
  inst.addr = addr;
  return inst;
}

void append_gfx1250_ds_bpermute_b32(std::vector<uint32_t> &program, uint32_t vdst, uint32_t addr,
                                    uint32_t data) {
  program.push_back(0xDACC0000U);
  program.push_back((vdst << 24U) | (data << 8U) | addr);
}

void append_ds_loads(std::vector<uint32_t> &program, uint32_t count, uint32_t addr) {
  for (uint32_t i = 0; i < count; ++i)
    append_inst(program, ds_load_b32(i, addr));
}

[[nodiscard]] rdna4::VdsMachineInst ds_store_b32(uint32_t addr, uint32_t data0) {
  auto inst = std::bit_cast<rdna4::VdsMachineInst>(std::array<uint32_t, 2>{0xD8340000U, 0});
  inst.addr = addr;
  inst.data0 = data0;
  return inst;
}

[[nodiscard]] rdna4::VdsMachineInst ds_swizzle_b32(uint32_t vdst, uint32_t addr) {
  auto inst = std::bit_cast<rdna4::VdsMachineInst>(std::array<uint32_t, 2>{0xD8D40000U, 0});
  inst.vdst = vdst;
  inst.addr = addr;
  return inst;
}

[[nodiscard]] rdna4::VdsMachineInst ds_nop() {
  return std::bit_cast<rdna4::VdsMachineInst>(std::array<uint32_t, 2>{0xD8500000U, 0});
}

void append_global_load_async_to_lds_b32(std::vector<uint32_t> &program) {
  program.insert(program.end(), {0xEE180000u, 0x00000000u, 0x00000800u});
}

void append_global_store_async_from_lds_b32(std::vector<uint32_t> &program) {
  program.insert(program.end(), {0xEE190000u, 0x00000000u, 0x00000800u});
}

void append_ds_atomic_async_barrier_arrive_b64(std::vector<uint32_t> &program) {
  program.insert(program.end(), {0xD9580000u, 0x00000000u});
}

void append_tensor_load_to_lds(std::vector<uint32_t> &program) {
  program.insert(program.end(), {0xD0710001u, 0x7C000000u, 0x7C7C0C00u});
}

void append_tensor_store_from_lds(std::vector<uint32_t> &program) {
  program.insert(program.end(), {0xD0714001u, 0x7C000000u, 0x7C7C0C08u});
}

[[nodiscard]] rdna4::SoppMachineInst s_wait_asynccnt(uint32_t count) { return sopp(74, count); }

[[nodiscard]] rdna4::SoppMachineInst s_wait_tensorcnt(uint32_t count) { return sopp(75, count); }

[[nodiscard]] rdna4::VdsdirMachineInst ds_param_load(uint32_t vdst) {
  auto inst = std::bit_cast<rdna4::VdsdirMachineInst>(0xCE000000U);
  inst.vdst = vdst;
  inst.wait_va_vdst = 15;
  inst.wait_vm_vsrc = 1;
  return inst;
}

[[nodiscard]] rdna4::VdsdirMachineInst
ds_param_load_with_waits(uint32_t vdst, uint32_t wait_va_vdst, uint32_t wait_vm_vsrc) {
  auto inst = ds_param_load(vdst);
  inst.wait_va_vdst = wait_va_vdst;
  inst.wait_vm_vsrc = wait_vm_vsrc;
  return inst;
}

[[nodiscard]] rdna4::VdsdirMachineInst ds_direct_load(uint32_t vdst) {
  auto inst = std::bit_cast<rdna4::VdsdirMachineInst>(0xCE100000U);
  inst.vdst = vdst;
  inst.wait_va_vdst = 15;
  inst.wait_vm_vsrc = 1;
  return inst;
}

void append_ds_direct_loads(std::vector<uint32_t> &program, uint32_t count, uint32_t first_vdst) {
  for (uint32_t i = 0; i < count; ++i)
    append_inst(program, ds_direct_load(first_vdst + i));
}

[[nodiscard]] rdna4::VinterpMachineInst v_interp_p10_f32(uint32_t vdst, uint32_t src0,
                                                         uint32_t wait_exp) {
  auto inst = std::bit_cast<rdna4::VinterpMachineInst>(std::array<uint32_t, 2>{0xCD000000U, 0});
  inst.vdst = vdst;
  inst.src0 = 256 + src0;
  inst.src1 = 256;
  inst.src2 = 256;
  inst.wait_exp = wait_exp;
  return inst;
}

[[nodiscard]] rdna4::VsampleMachineInst image_sample(uint32_t vdata) {
  auto inst = std::bit_cast<rdna4::VsampleMachineInst>(std::array<uint32_t, 3>{0xE406C000U, 0, 0});
  inst.vdata = vdata;
  inst.vaddr0 = 200;
  return inst;
}

[[nodiscard]] rdna4::VsampleMachineInst image_msaa_load(uint32_t vdata) {
  auto inst = std::bit_cast<rdna4::VsampleMachineInst>(std::array<uint32_t, 3>{0xE4060000U, 0, 0});
  inst.vdata = vdata;
  inst.rsrc = 0;
  inst.vaddr0 = 200;
  return inst;
}

void append_image_samples(std::vector<uint32_t> &program, uint32_t count, uint32_t first_vdata) {
  for (uint32_t i = 0; i < count; ++i)
    append_inst(program, image_sample(first_vdata + i));
}

[[nodiscard]] rdna4::VimageMachineInst image_load(uint32_t vdata) {
  auto inst = std::bit_cast<rdna4::VimageMachineInst>(std::array<uint32_t, 3>{0xD0000000U, 0, 0});
  inst.dmask = 0xf;
  inst.vdata = vdata;
  inst.rsrc = 0;
  return inst;
}

[[nodiscard]] rdna4::VimageMachineInst image_atomic_add_uint(uint32_t vdata) {
  auto inst = std::bit_cast<rdna4::VimageMachineInst>(std::array<uint32_t, 3>{0xD0030000U, 0, 0});
  inst.dmask = 1;
  inst.vdata = vdata;
  inst.rsrc = 0;
  inst.vaddr0 = 200;
  inst.th = 1; // Return the pre-operation value.
  return inst;
}

[[nodiscard]] rdna4::VimageMachineInst image_bvh_intersect_ray(uint32_t vdata) {
  auto inst = std::bit_cast<rdna4::VimageMachineInst>(std::array<uint32_t, 3>{0xD0064000U, 0, 0});
  inst.vdata = vdata;
  inst.vaddr0 = 200;
  inst.vaddr1 = 204;
  inst.vaddr2 = 208;
  inst.vaddr3 = 212;
  inst.vaddr4 = 216;
  return inst;
}

void append_image_bvhs(std::vector<uint32_t> &program, uint32_t count, uint32_t first_vdata) {
  for (uint32_t i = 0; i < count; ++i)
    append_inst(program, image_bvh_intersect_ray(first_vdata + 4 * i));
}

[[nodiscard]] rdna4::VimageMachineInst image_store_b32(uint32_t vdata) {
  auto inst = std::bit_cast<rdna4::VimageMachineInst>(std::array<uint32_t, 3>{0xD0018000U, 0, 0});
  inst.dmask = 1;
  inst.vdata = vdata;
  inst.rsrc = 0;
  return inst;
}

[[nodiscard]] rdna4::VexportMachineInst export_mrt0_v0() {
  rdna4::VexportMachineInst inst{};
  inst.encoding = 0x3e;
  inst.en = 1;
  inst.tgt = 0;
  inst.done = 1;
  inst.vsrc0 = 0;
  return inst;
}

[[nodiscard]] std::vector<uint32_t> words(std::initializer_list<rdna4::SoppMachineInst> insts) {
  std::vector<uint32_t> result;
  for (const auto &inst : insts)
    append_inst(result, inst);
  return result;
}

[[nodiscard]] rdna4::SoppMachineInst s_wait_alu_sa_sdst_0() {
  return sopp(8, 0xff9e); // depctr_sa_sdst(0)
}

[[nodiscard]] rdna4::SoppMachineInst s_wait_alu_va_vcc_0() {
  return sopp(8, 0xff9d); // depctr_va_vcc(0)
}

[[nodiscard]] rdna4::SoppMachineInst s_wait_alu_va_sdst_0() {
  return sopp(8, 0xf19f); // depctr_va_sdst(0)
}

[[nodiscard]] rdna4::SoppMachineInst s_wait_alu_vm_vsrc(uint32_t count) {
  return sopp(8, 0xff83u | (count << 2u));
}

[[nodiscard]] rdna4::SoppMachineInst s_wait_alu_vm_vsrc_0() { return s_wait_alu_vm_vsrc(0); }

[[nodiscard]] rdna4::SoppMachineInst s_delay_alu(uint32_t simm16) { return sopp(7, simm16); }

[[nodiscard]] rdna4::SoppMachineInst s_wait_xcnt(uint32_t count) { return sopp(69, count); }

[[nodiscard]] rdna4::SoppMachineInst s_wait_xcnt_0() { return s_wait_xcnt(0); }

[[nodiscard]] std::vector<uint32_t> gfx1201_swappc_valu_sgpr_boundary_program(bool include_wait) {
  constexpr uint16_t kTargetSreg = 8;
  constexpr uint16_t kReturnSreg = 30;
  constexpr uint16_t kLiteralOperand = 255;
  constexpr uint16_t kInlineInt0 = 128;

  std::vector<uint32_t> program;
  program.push_back(build_s_getpc_b64(kTargetSreg, ROCJITSU_CODE_ARCH_RDNA4).value());
  program.push_back(
      build_s_add_u32(kTargetSreg, kTargetSreg, kLiteralOperand, ROCJITSU_CODE_ARCH_RDNA4).value());
  const size_t delta_word = program.size();
  program.push_back(0);
  program.push_back(
      build_s_addc_u32(kTargetSreg + 1, kTargetSreg + 1, kInlineInt0, ROCJITSU_CODE_ARCH_RDNA4)
          .value());
  append_inst(program, v_add_f32_e32(/*vdst=*/0, /*src0=*/2, /*vsrc1=*/0));
  const auto readlane =
      build_v_readlane_b32(/*sdst=*/2, /*vsrc=*/40, /*lane=*/0, ROCJITSU_CODE_ARCH_RDNA4);
  if (readlane.failed())
    return {};
  program.insert(program.end(), readlane.value().begin(), readlane.value().end());
  if (include_wait)
    append_inst(program, s_wait_alu_va_sdst_0());
  program.push_back(build_s_swappc_b64(kReturnSreg, kTargetSreg, ROCJITSU_CODE_ARCH_RDNA4).value());
  append_inst(program, s_endpgm());
  const uint64_t helper_offset = program.size() * sizeof(uint32_t);
  constexpr uint64_t kGetpcResultOffset = sizeof(uint32_t);
  program[delta_word] = static_cast<uint32_t>(helper_offset - kGetpcResultOffset);
  program.push_back(build_s_setpc_b64(kReturnSreg, ROCJITSU_CODE_ARCH_RDNA4).value());
  return program;
}

[[nodiscard]] std::vector<uint32_t> gfx1201_call_valu_sgpr_boundary_program(bool include_wait) {
  constexpr uint16_t kReturnSreg = 30;
  std::vector<uint32_t> program;
  append_inst(program, v_add_f32_e32(/*vdst=*/0, /*src0=*/2, /*vsrc1=*/0));
  const auto readlane =
      build_v_readlane_b32(/*sdst=*/2, /*vsrc=*/40, /*lane=*/0, ROCJITSU_CODE_ARCH_RDNA4);
  if (readlane.failed())
    return {};
  program.insert(program.end(), readlane.value().begin(), readlane.value().end());
  if (include_wait)
    append_inst(program, s_wait_alu_va_sdst_0());
  program.push_back(
      build_s_call_b64(kReturnSreg, /*offset_dwords=*/1, ROCJITSU_CODE_ARCH_RDNA4).value());
  append_inst(program, s_endpgm());
  program.push_back(build_s_setpc_b64(kReturnSreg, ROCJITSU_CODE_ARCH_RDNA4).value());
  return program;
}

void append_gfx950_s_waitcnt_vmcnt_0(std::vector<uint32_t> &program) {
  program.push_back(0xBF8C0F70u);
}

void append_gfx950_s_waitcnt_vmcnt_1(std::vector<uint32_t> &program) {
  program.push_back(0xBF8C0F71u);
}

void append_gfx950_s_waitcnt_lgkmcnt_0(std::vector<uint32_t> &program) {
  program.push_back(0xBF8CC07Fu);
}

void append_gfx950_s_waitcnt_lgkmcnt_1(std::vector<uint32_t> &program) {
  program.push_back(0xBF8CC17Fu);
}

void append_gfx950_s_waitcnt_lgkmcnt_2(std::vector<uint32_t> &program) {
  program.push_back(0xBF8CC27Fu);
}

void append_gfx950_s_waitcnt_expcnt_0(std::vector<uint32_t> &program) {
  program.push_back(0xBF8CCF0Fu);
}

void append_gfx950_buffer_load_dword_v0_v8_s0_offen(std::vector<uint32_t> &program) {
  program.push_back(0xE0501000u);
  program.push_back(0x80000008u);
}

void append_gfx950_global_atomic_add_x2(std::vector<uint32_t> &program, uint32_t vdst,
                                        uint32_t vaddr, uint32_t data, bool return_old) {
  cdna4::FlatMachineInst inst{};
  inst.encoding = 0x37;
  inst.seg = 2;
  inst.op = 98;
  inst.sc0 = return_old;
  inst.addr = vaddr;
  inst.data = data;
  inst.saddr = 0x7f;
  inst.vdst = vdst;
  append_inst(program, inst);
}

void append_gfx950_buffer_atomic_add(std::vector<uint32_t> &program, uint32_t vdata,
                                     bool return_old) {
  cdna4::MubufMachineInst inst{};
  inst.encoding = 0x38;
  inst.op = 66;
  inst.offen = 1;
  inst.sc0 = return_old;
  inst.vaddr = 8;
  inst.vdata = vdata;
  inst.srsrc = 0;
  inst.soffset = 0;
  append_inst(program, inst);
}

void append_gfx950_buffer_load_dword_v1_v8_s0_offen(std::vector<uint32_t> &program) {
  program.push_back(0xE0501000u);
  program.push_back(0x80000108u);
}

void append_gfx950_buffer_load_dword_v1_off_s0(std::vector<uint32_t> &program) {
  program.push_back(0xE0500000u);
  program.push_back(0x80000100u);
}

void append_gfx950_buffer_load_dword_v2_v0_s0_offen(std::vector<uint32_t> &program) {
  program.push_back(0xE0501000u);
  program.push_back(0x80000200u);
}

void append_gfx950_buffer_load_dwordx4_v0_s64_offen_lds(std::vector<uint32_t> &program) {
  program.push_back(0xE05D1000u);
  program.push_back(0x80100008u);
}

void append_gfx950_ds_read_b128_v18_v12_offset64(std::vector<uint32_t> &program) {
  program.push_back(0xD9FE0040u);
  program.push_back(0x1200000Cu);
}

void append_gfx950_ds_read_b128_v82_v13_offset64(std::vector<uint32_t> &program) {
  program.push_back(0xD9FE0040u);
  program.push_back(0x5200000Du);
}

void append_gfx950_ds_read_b128_v148_v13_offset1056(std::vector<uint32_t> &program) {
  program.push_back(0xD9FE0420u);
  program.push_back(0x9400000Du);
}

void append_gfx950_ds_read_b128_v90_v13_offset1120(std::vector<uint32_t> &program) {
  program.push_back(0xD9FE0460u);
  program.push_back(0x5A00000Du);
}

void append_gfx950_v_cvt_pk_bf16_f32_v86_v148_v149(std::vector<uint32_t> &program) {
  program.push_back(0xD2680056u);
  program.push_back(0x00032B94u);
}

void append_gfx950_v_cvt_pk_bf16_f32_v88_v90_v91(std::vector<uint32_t> &program) {
  program.push_back(0xD2680058u);
  program.push_back(0x0002B75Au);
}

void append_gfx950_v_mfma_f32_16x16x32_bf16_v16_acc_cd_0(std::vector<uint32_t> &program) {
  program.push_back(0xD3B50010u);
  program.push_back(0x04425D66u);
}

void append_gfx950_v_mfma_f32_16x16x32_bf16_acc16_acc_cd_1(std::vector<uint32_t> &program) {
  program.push_back(0xD3B58010u);
  program.push_back(0x04425D66u);
}

void append_gfx950_buffer_load_dword_v1_v1_s24_offen(std::vector<uint32_t> &program) {
  program.push_back(0xE0501000u);
  program.push_back(0x80060101u);
}

void append_gfx950_s_buffer_load_dwordx8_s0_s28_imm16(std::vector<uint32_t> &program) {
  program.push_back(0xC02E000Eu);
  program.push_back(0x00000010u);
}

void append_gfx950_s_buffer_load_dwordx8_s24_s28_imm16(std::vector<uint32_t> &program) {
  program.push_back(0xC02E060Eu);
  program.push_back(0x00000010u);
}

void append_gfx950_ds_read_b32_v0_v4(std::vector<uint32_t> &program) {
  program.push_back(0xD86C0000u);
  program.push_back(0x00000004u);
}

void append_gfx950_s_load_dword_s4_s0(std::vector<uint32_t> &program) {
  program.push_back(0xC0020100u);
  program.push_back(0x00000000u);
}

void append_gfx950_s_mov_b32_s8_s4(std::vector<uint32_t> &program) {
  program.push_back(0xBE880004u);
}

void append_gfx950_v_mov_b32_v1_v0(std::vector<uint32_t> &program) {
  program.push_back(0x7E020300u);
}

void append_gfx950_v_mov_b32_v0_v2(std::vector<uint32_t> &program) {
  program.push_back(0x7E000302u);
}

void append_gfx950_v_mov_b32_v8_v10(std::vector<uint32_t> &program) {
  program.push_back(0x7E10030Au);
}

void append_gfx942_s_waitcnt_vmcnt_0(std::vector<uint32_t> &program) {
  program.push_back(0xBF8C0F70u);
}

void append_gfx942_s_waitcnt_vmcnt_1(std::vector<uint32_t> &program) {
  program.push_back(0xBF8C0F71u);
}

void append_gfx942_s_waitcnt_lgkmcnt_0(std::vector<uint32_t> &program) {
  program.push_back(0xBF8CC07Fu);
}

void append_gfx942_buffer_load_dword_v0_v8_s0_offen(std::vector<uint32_t> &program) {
  program.push_back(0xE0501000u);
  program.push_back(0x80000008u);
}

void append_gfx942_buffer_load_dword(std::vector<uint32_t> &program, uint8_t vdst, uint8_t vaddr) {
  program.push_back(0xE0501000u);
  program.push_back(0x80010000u | (static_cast<uint32_t>(vdst) << 8u) | vaddr);
}

void append_gfx942_s_load_dword_s4_s0(std::vector<uint32_t> &program) {
  program.push_back(0xC0020100u);
  program.push_back(0x00000000u);
}

void append_gfx942_s_mov_b32_s8_s4(std::vector<uint32_t> &program) {
  program.push_back(0xBE880004u);
}

void append_gfx942_v_mov_b32_v1_v0(std::vector<uint32_t> &program) {
  program.push_back(0x7E020300u);
}

void append_gfx942_v_mov_b32_v4_v2(std::vector<uint32_t> &program) {
  program.push_back(0x7E080302u);
}

void append_gfx942_global_load_dword_v4_v2(std::vector<uint32_t> &program) {
  program.push_back(0xDC508000u);
  program.push_back(0x047F0002u);
}

void append_gfx942_ds_read_b32_v4_v0(std::vector<uint32_t> &program) {
  program.push_back(0xD86C0000u);
  program.push_back(0x04000000u);
}

void append_gfx942_v_mov_b32_v5_v4(std::vector<uint32_t> &program) {
  program.push_back(0x7E0A0304u);
}

void append_gfx1100_s_waitcnt_vmcnt_0(std::vector<uint32_t> &program) {
  program.push_back(0xBF8903F7u);
}

void append_gfx1100_s_waitcnt_vscnt_0(std::vector<uint32_t> &program) {
  program.push_back(0xBC7C0000u);
}

void append_gfx1100_s_waitcnt_vmcnt_sopk_0(std::vector<uint32_t> &program) {
  program.push_back(0xBCFC0000u);
}

void append_gfx1100_s_waitcnt_vmcnt_sopk_1(std::vector<uint32_t> &program) {
  program.push_back(0xBCFC0001u);
}

void append_gfx1100_s_waitcnt_expcnt_sopk_0(std::vector<uint32_t> &program) {
  program.push_back(0xBD7C0000u);
}

void append_gfx1100_s_waitcnt_lgkmcnt_sopk_0(std::vector<uint32_t> &program) {
  program.push_back(0xBDFC0000u);
}

void append_gfx1100_global_load_b32_v0_v8_s0(std::vector<uint32_t> &program) {
  program.push_back(0xDC520000u);
  program.push_back(0x00000008u);
}

void append_gfx1100_global_load_b32_v2_v8_s0(std::vector<uint32_t> &program) {
  program.push_back(0xDC520000u);
  program.push_back(0x02000008u);
}

void append_gfx1150_global_load_b32(std::vector<uint32_t> &program, uint8_t vdst) {
  program.push_back(0xDC520000u);
  program.push_back((static_cast<uint32_t>(vdst) << 24u) | 0x00000008u);
}

void append_gfx1100_s_load_b32_s4_s0(std::vector<uint32_t> &program) {
  program.push_back(0xF4000100u);
  program.push_back(0xF8000000u);
}

void append_gfx1100_s_mov_b32_s8_s4(std::vector<uint32_t> &program) {
  program.push_back(0xBE880004u);
}

void append_gfx1100_v_add_co_u32_v0_s1_s4_v0(std::vector<uint32_t> &program) {
  program.push_back(0xD7000100u);
  program.push_back(0x02020004u);
}

void append_gfx1100_v_add_co_ci_u32_v2_null_s25_0_s1(std::vector<uint32_t> &program) {
  program.push_back(0xD5207C02u);
  program.push_back(0x00050019u);
}

void append_gfx1100_v_mov_b32_v1_v0(std::vector<uint32_t> &program) {
  program.push_back(0x7E020300u);
}

void append_gfx1100_v_mov_b32_v3_v2(std::vector<uint32_t> &program) {
  program.push_back(0x7E060302u);
}

void append_gfx1150_ds_load_u8_d16_hi_v1_v1(std::vector<uint32_t> &program) {
  program.push_back(0xDA8C0000u);
  program.push_back(0x01000001u);
}

void append_gfx1150_ds_load_u8_d16_hi_v1_v6(std::vector<uint32_t> &program) {
  program.push_back(0xDA8C0000u);
  program.push_back(0x01000006u);
}

void append_gfx1150_ds_load_u8_d16_v1_v5_offset512(std::vector<uint32_t> &program) {
  program.push_back(0xDA880200u);
  program.push_back(0x01000005u);
}

void append_gfx1150_s_waitcnt_lgkmcnt_1(std::vector<uint32_t> &program) {
  program.push_back(0xBF89FC17u);
}

void append_gfx1150_global_store_d16_hi_b8_v2_v1(std::vector<uint32_t> &program) {
  program.push_back(0xDC920000u);
  program.push_back(0x007C0102u);
}

constexpr uint32_t kOverflowQueueSize = 40;
constexpr uint32_t kOverflowRequiredCount = kOverflowQueueSize - 1;
constexpr uint32_t kOverflowBaseVgpr = 32;
constexpr uint32_t kOverflowConsumerVgpr = 96;
constexpr uint32_t kOverflowKmcntBaseSgpr = 0;
constexpr uint32_t kOverflowKmcntSbase = 48;
constexpr uint32_t kOverflowKmcntConsumerSgpr = 80;

void append_s_loads(std::vector<uint32_t> &program, uint32_t count, uint32_t first_sdata,
                    uint32_t sbase) {
  for (uint32_t i = 0; i < count; ++i)
    append_inst(program, s_load_b32(first_sdata + i, sbase));
}

void expect_single_overflow_diagnostic(const WaitcheckReport &report, WaitCounterKind counter,
                                       WaitcheckAccessKind access, RegClass reg_class,
                                       uint32_t reg_index,
                                       uint32_t required_count = kOverflowRequiredCount) {
  ASSERT_TRUE(report.supported);
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, counter);
  EXPECT_EQ(report.diagnostics[0].access, access);
  EXPECT_EQ(report.diagnostics[0].reg.cls, reg_class);
  EXPECT_EQ(report.diagnostics[0].reg.index, reg_index);
  EXPECT_EQ(report.diagnostics[0].required_count, required_count);
}

TEST(WaitcheckTest, ReportsUnsupportedArchitectures) {
  auto program = words({sopp(0)});

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RV32I);

  EXPECT_FALSE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty());
}

TEST(WaitcheckTest, MapsGfx942TargetToCdna3) {
  EXPECT_EQ(waitcheck_arch_for_target(ROCJITSU_CODE_TARGET_GFX942), ROCJITSU_CODE_ARCH_CDNA3);
}

TEST(WaitcheckTest, MapsGfx950TargetToCdna4) {
  EXPECT_EQ(waitcheck_arch_for_target(ROCJITSU_CODE_TARGET_GFX950), ROCJITSU_CODE_ARCH_CDNA4);
}

TEST(WaitcheckTest, MapsGfx1250AndGfx1251TargetsToCdna5) {
  EXPECT_EQ(waitcheck_arch_for_target(ROCJITSU_CODE_TARGET_GFX1250), ROCJITSU_CODE_ARCH_CDNA5);
  EXPECT_EQ(waitcheck_arch_for_target(ROCJITSU_CODE_TARGET_GFX1251), ROCJITSU_CODE_ARCH_CDNA5);

  const auto image = waitcheck_test::make_gfx_code_object(
      {build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA5).value()}, EF_AMDGPU_MACH_AMDGCN_GFX1251);
  AmdGpuCodeObject object(image.data(), image.size());
  ASSERT_EQ(object.target_id(), ROCJITSU_CODE_TARGET_GFX1251);
  EXPECT_EQ(waitcheck_kernels(object).size(), 1u);
}

TEST(WaitcheckTest, MapsGfx1100TargetToRdna3) {
  EXPECT_EQ(waitcheck_arch_for_target(ROCJITSU_CODE_TARGET_GFX1100), ROCJITSU_CODE_ARCH_RDNA3);
}

TEST(WaitcheckTest, MapsGfx1150AndGfx1151TargetsToRdna35) {
  EXPECT_EQ(waitcheck_arch_for_target(ROCJITSU_CODE_TARGET_GFX1150), ROCJITSU_CODE_ARCH_RDNA3_5);
  EXPECT_EQ(waitcheck_arch_for_target(ROCJITSU_CODE_TARGET_GFX1151), ROCJITSU_CODE_ARCH_RDNA3_5);
}

TEST(WaitcheckTest, DecodesRdna4CombinedStoreDsWait) {
  std::vector<uint32_t> program;
  append_inst(program, sopp(73, 0));

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> inst(decode_valid(*decoder, program.data()));
  ASSERT_NE(inst, nullptr);
  EXPECT_EQ(inst->mnemonic(), "s_wait_storecnt_dscnt");
  EXPECT_TRUE(inst->is_waitcnt());
}

TEST(WaitcheckTest, DecodesCombinedStoreDsWaitSequenceAtExpectedOffset) {
  std::vector<uint32_t> program;
  append_inst(program, global_store_b32(10));
  append_inst(program, ds_load_b32(0, 4));
  append_inst(program, sopp(73, 0));
  append_inst(program, v_mov_b32(2, 0));
  append_inst(program, s_endpgm());

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_NE(decoder, nullptr);

  struct ExpectedInst {
    const char *mnemonic;
    int size;
  };
  const ExpectedInst expected[] = {
      {"global_store_b32", 12}, {"ds_load_b32", 8}, {"s_wait_storecnt_dscnt", 4},
      {"v_mov_b32_e32", 4},     {"s_endpgm", 4},
  };

  size_t word_index = 0;
  for (const auto &tc : expected) {
    std::unique_ptr<Instruction> inst(decode_valid(*decoder, &program[word_index]));
    ASSERT_NE(inst, nullptr);
    EXPECT_EQ(std::string(inst->mnemonic()), tc.mnemonic);
    EXPECT_EQ(inst->size(), tc.size) << inst->mnemonic();
    word_index += static_cast<size_t>(inst->size() / sizeof(uint32_t));
  }
  EXPECT_EQ(word_index, program.size());
}

TEST(WaitcheckTest, ReportsMissingLoadcntBeforeUse) {
  std::vector<uint32_t> program;
  append_inst(program, global_load_b32(0));
  append_inst(program, v_mov_b32(1, 0));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported);
  ASSERT_EQ(report.diagnostics.size(), 1u);
  EXPECT_EQ(report.diagnostics[0].kind, WaitcheckDiagnosticKind::WaitCounter);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Load);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Use);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::VGPR);
  EXPECT_EQ(report.diagnostics[0].reg.index, 0u);
  EXPECT_EQ(report.diagnostics[0].required_count, 0u);
}

TEST(WaitcheckTest, Gfx1250StreamDecodePreservesFourWordInstructionAtEnd) {
  std::vector<uint32_t> program;
  append_inst(program, global_load_b32(16));
  program.insert(program.end(), kGfx1250ScaledWmma.begin(), kGfx1250ScaledWmma.end());

  const auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA5);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  const auto diagnostic =
      std::ranges::find_if(report.diagnostics, [](const WaitcheckDiagnostic &candidate) {
        return candidate.instruction.starts_with("v_wmma_scale16_f32_16x16x128_f8f6f4");
      });
  ASSERT_NE(diagnostic, report.diagnostics.end()) << diagnostic_summary(report);
  EXPECT_EQ(diagnostic->counter, WaitCounterKind::Load);
  EXPECT_EQ(diagnostic->access, WaitcheckAccessKind::Use);
  EXPECT_EQ(diagnostic->reg, (RegisterRef{RegClass::VGPR, 16, 1}));
}

TEST(WaitcheckTest, AcceptsLoadcntZeroBeforeUse) {
  std::vector<uint32_t> program;
  append_inst(program, global_load_b32(0));
  append_inst(program, sopp(64, 0)); // s_wait_loadcnt 0
  append_inst(program, v_mov_b32(1, 0));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx942ReportsMissingVmcntBeforeBufferLoadUse) {
  std::vector<uint32_t> program;
  append_gfx942_buffer_load_dword_v0_v8_s0_offen(program);
  append_gfx942_v_mov_b32_v1_v0(program);

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA3);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Load);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Use);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::VGPR);
  EXPECT_EQ(report.diagnostics[0].reg.index, 0u);
  EXPECT_EQ(report.diagnostics[0].required_count, 0u);
  EXPECT_NE(report.diagnostics[0].message.find("s_waitcnt vmcnt(0)"), std::string::npos);
}

TEST(WaitcheckTest, Gfx942AcceptsSWaitcntVmcntZeroBeforeBufferLoadUse) {
  std::vector<uint32_t> program;
  append_gfx942_buffer_load_dword_v0_v8_s0_offen(program);
  append_gfx942_s_waitcnt_vmcnt_0(program);
  append_gfx942_v_mov_b32_v1_v0(program);

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA3);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx942OffenBufferAddressDoesNotUseAdjacentVgpr) {
  // Reduced from PyTorch's CK kernel at code object 202, .text+0x43500.
  // An offen-only MUBUF address is one VGPR wide: the final load uses v67,
  // not v[67:68], so the earlier asynchronous definition of v68 is unrelated.
  std::vector<uint32_t> program;
  append_gfx942_buffer_load_dword(program, 68, 63);
  append_gfx942_buffer_load_dword(program, 69, 64);
  append_gfx942_buffer_load_dword(program, 70, 65);
  append_gfx942_buffer_load_dword(program, 71, 67);

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA3);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx942BufferDescriptorUsesEncodedSgprQuad) {
  // SRSRC=1 names s[4:7], not the unscaled register range s[1:4].
  std::vector<uint32_t> program;
  append_gfx942_s_load_dword_s4_s0(program);
  append_gfx942_buffer_load_dword(program, 0, 8);

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA3);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Ds);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Use);
  EXPECT_EQ(report.diagnostics[0].reg, (RegisterRef{RegClass::SGPR, 4, 1}));
  EXPECT_NE(report.diagnostics[0].message.find("s_waitcnt lgkmcnt(0)"), std::string::npos);
}

TEST(WaitcheckTest, Gfx942ReportsMissingVmcntBeforeGlobalLoadUse) {
  std::vector<uint32_t> program;
  append_gfx942_global_load_dword_v4_v2(program);
  append_gfx942_v_mov_b32_v5_v4(program);

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA3);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Load);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Use);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::VGPR);
  EXPECT_EQ(report.diagnostics[0].reg.index, 4u);
}

TEST(WaitcheckTest, Gfx942ReportsMissingLgkmcntBeforeDsReadUse) {
  std::vector<uint32_t> program;
  append_gfx942_ds_read_b32_v4_v0(program);
  append_gfx942_v_mov_b32_v5_v4(program);

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA3);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Ds);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Use);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::VGPR);
  EXPECT_EQ(report.diagnostics[0].reg.index, 4u);
}

TEST(WaitcheckTest, Gfx942IndexedDsReadsWriteTheirDestinationInOrder) {
  std::vector<uint32_t> program;
  append_gfx942_ds_read_b32_v4_v0(program);
  append_gfx942_ds_read_b32_v4_v0(program);

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA3);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, RelativeMovesUseM0ForEachPhysicalOperand) {
  for (const bool scalar : {false, true}) {
    const auto arch = scalar ? ROCJITSU_CODE_ARCH_CDNA3 : ROCJITSU_CODE_ARCH_RDNA3;
    for (const bool write : {false, true}) {
      for (const auto [m0_selector, producer, expected] :
           {std::tuple{129u, 0u, false}, // A valid relative register never falls back to R0.
            std::tuple{129u, 4u, false}, // M0 = 1, encoded base is independent.
            std::tuple{129u, 5u, true},  // M0 = 1, accessed register is pending.
            std::tuple{8u, 5u, true}}) { // Unknown M0 can select register 5.
        SCOPED_TRACE(::testing::Message()
                     << scalar << '/' << write << '/' << m0_selector << '/' << producer);
        const uint32_t set_m0 = (scalar ? 0xbefc0000u : 0xbefd0000u) | m0_selector;
        // S_LOAD_DWORD sN, s[0:1], 0 or DS_LOAD_B32 vN, v1.
        const uint32_t load = scalar ? 0xc0020000u | (producer << 6) : 0xd8d80000u;
        const uint32_t operands = scalar ? 0u : (producer << 24) | 1u;
        // MOVRELD writes R[4+M0] from R7. MOVRELS reads R[4+M0] into R7.
        const uint32_t move =
            scalar ? (write ? 0xbe842c07u : 0xbe872a04u) : (write ? 0x7e088507u : 0x7e0e8704u);
        std::vector<uint32_t> program{set_m0, load, operands, move,
                                      scalar ? 0xbf810000u : 0xbfb00000u};
        WaitcheckOptions options;
        options.fallback_wavefront_size = scalar ? 64 : 32;
        const auto report = analyze_waitcnts(TestCodeObject(program), arch, options);
        ASSERT_TRUE(report.supported) << report.analysis_error;
        ASSERT_EQ(report.diagnostics.size(), expected ? 1u : 0u) << diagnostic_summary(report);
        if (expected) {
          EXPECT_EQ(report.diagnostics[0].reg,
                    (RegisterRef{scalar ? RegClass::SGPR : RegClass::VGPR, 5, 1}));
          EXPECT_EQ(report.diagnostics[0].access,
                    write ? WaitcheckAccessKind::Def : WaitcheckAccessKind::Use);
        }
      }
    }
  }
}

TEST(WaitcheckTest, RelativeMoveUsesEveryScalarCopyValueAcrossTheLoopBackedge) {
  for (uint32_t producer : {4u, 10u}) {
    for (bool write : {false, true}) {
      for (bool unknown_backedge : {false, true}) {
        SCOPED_TRACE(::testing::Message() << producer << '/' << write << '/' << unknown_backedge);
        // s3 starts at 0. On the next iteration it is 4 (or unknown). Copy
        // through s4 into M0 before accessing v[0+M0] on each iteration.
        const std::vector<uint32_t> words{0xbe830080,
                                          0xd8d80000,
                                          (producer << 24) | 1u,
                                          0xbe840003,
                                          0xbefd0004,
                                          write ? 0x7e00852du : 0x7e5a8700u,
                                          unknown_backedge ? 0xbe830009u : 0xbe830084u,
                                          0xbfa2fffb,
                                          0xbfb00000};
        const auto report = analyze_waitcnts(TestCodeObject(words), ROCJITSU_CODE_ARCH_RDNA3);
        ASSERT_TRUE(report.supported && report.complete) << report.analysis_error;
        EXPECT_EQ(report.diagnostics.size(), producer == 4 || unknown_backedge)
            << diagnostic_summary(report);
        for (const auto &diag : report.diagnostics) {
          EXPECT_EQ(diag.reg, (RegisterRef{RegClass::VGPR, static_cast<uint16_t>(producer), 1}));
          EXPECT_EQ(diag.access, write ? WaitcheckAccessKind::Def : WaitcheckAccessKind::Use);
        }
      }
    }
  }
}

TEST(WaitcheckTest, RelativeMoveUsesMaskedAndScaledScalarIndices) {
  for (const auto [mask, operation, producer, hazard] : {
           // s3 = unknown & 7. The scaled offset is even and at most 14.
           std::tuple{0x8b038710u, 0x84038103u, 14u, true},
           std::tuple{0x8b038710u, 0x84038103u, 13u, false},
           std::tuple{0x8b038f10u, 0x84038103u, 30u, true}, // Sixteen peers.
           std::tuple{0x8b038f10u, 0x84038103u, 31u, false},
           std::tuple{0x8b038710u, 0x8403a103u, 16u, false}, // Shift amount modulo 32.
           std::tuple{0x8b031087u, 0x8d030384u, 7u, true},   // Commuted AND/XOR.
           std::tuple{0x8b038a10u, 0x8d038403u, 6u, true},   // (unknown & 10) XOR 4.
           std::tuple{0x8b038a10u, 0x8d038403u, 3u, false},
           std::tuple{0x8b038a10u, 0x8c038403u, 14u, true}, // (unknown & 10) OR 4.
           std::tuple{0x8b038a10u, 0x8c030384u, 10u, false},
           std::tuple{0x8b039f10u, 0x84038103u, 31u, true}, // Mask exceeds finite set.
           std::tuple{0x8b030510u, 0x84038103u, 31u, true}, // Runtime mask.
       }) {
    SCOPED_TRACE(::testing::Message() << mask << '/' << operation << '/' << producer);
    const std::vector<uint32_t> words{
        mask, operation, 0xbefd0003u, 0xd8d80000u, (producer << 24) | 1u, 0x7e00852du, 0xbfb00000u};
    const auto result = analyze_waitcnts(TestCodeObject(words), ROCJITSU_CODE_ARCH_RDNA3);
    ASSERT_TRUE(result.supported && result.complete) << result.analysis_error;
    EXPECT_EQ(result.diagnostics.size(), hazard ? 1u : 0u) << diagnostic_summary(result);
  }
}

TEST(WaitcheckTest, RelativeMoveWidensWhenTooManyScalarValuesReachTheJoin) {
  for (uint32_t paths : {4u, 5u}) {
    std::vector<uint32_t> words{0xd8d80000, 0x0a000001}; // DS_LOAD_B32 v10, v1.
    for (uint32_t path = 0; path < paths; ++path) {
      words.push_back(0xbe830080u + path); // S_MOV_B32 s3, path.
      if (path + 1 < paths) {
        words.push_back(0xbf068010u + path);                   // S_CMP_EQ_U32 s[16+path], 0.
        words.push_back(0xbfa20000u + 3 * (paths - path) - 5); // S_CBRANCH_SCC1 join.
      }
    }
    words.insert(words.end(), {0xbefd0003, 0x7e5a8700, 0xbfb00000}); // M0=s3; read v[0+M0].
    const auto report = analyze_waitcnts(TestCodeObject(words), ROCJITSU_CODE_ARCH_RDNA3);
    ASSERT_TRUE(report.supported && report.complete) << report.analysis_error;
    EXPECT_EQ(report.diagnostics.size(), paths == 5u) << diagnostic_summary(report);
  }
}

TEST(WaitcheckTest, PendingScalarLoadsInvalidateRelativeOffsetCopies) {
  for (const bool overwrite : {false, true}) {
    // s_load_b64 clobbers s[2:3]. Even a later synchronous write to s3 cannot
    // establish its value while the asynchronous replacement remains pending.
    std::vector<uint32_t> words{0xbe830080, 0xf4040080, 0xf8000000};
    if (overwrite)
      words.push_back(0xbe830080);
    words.insert(words.end(),
                 {0xbe840003, 0xbefd0004, 0xd8d80000, 0x0a000001, 0x7e5a8700, 0xbfb00000});
    const auto report = analyze_waitcnts(TestCodeObject(words), ROCJITSU_CODE_ARCH_RDNA3);
    ASSERT_TRUE(report.supported && report.complete) << report.analysis_error;
    EXPECT_TRUE(std::ranges::any_of(report.diagnostics, [](const WaitcheckDiagnostic &diag) {
      return diag.reg == RegisterRef{RegClass::SGPR, 3, 1} &&
             diag.access == WaitcheckAccessKind::Use;
    })) << diagnostic_summary(report);
    EXPECT_TRUE(std::ranges::any_of(report.diagnostics, [](const WaitcheckDiagnostic &diag) {
      return diag.reg == RegisterRef{RegClass::VGPR, 10, 1} &&
             diag.access == WaitcheckAccessKind::Use;
    })) << diagnostic_summary(report);
  }
}

TEST(WaitcheckTest, RelativeVectorSourcesFallBackOnlyOutsideTheDescriptorAllocation) {
  for (uint32_t vgpr_field : {0u, 1u}) {
    for (uint32_t producer : {0u, 9u}) {
      SCOPED_TRACE(::testing::Message() << vgpr_field << '/' << producer);
      // Wave32 allocates 8 or 16 VGPRs. With M0 = 5, MOVRELS reads v[4+5].
      std::vector<uint32_t> words{0xbefd0085, 0xd8d80000, (producer << 24) | 1, 0x7e0e8704,
                                  0xbfb00000};
      auto image = waitcheck_test::make_gfx_code_object(words, EF_AMDGPU_MACH_AMDGCN_GFX1100, true);
      // The fixture places its descriptor directly after .text.
      const size_t rsrc1_offset = 0x100 + words.size() * sizeof(uint32_t) + 48;
      std::memcpy(image.data() + rsrc1_offset, &vgpr_field, sizeof(vgpr_field));
      AmdGpuCodeObject object(image.data(), image.size());
      const auto report = analyze_waitcnts(object, ROCJITSU_CODE_ARCH_RDNA3);
      ASSERT_TRUE(report.supported && report.complete) << report.analysis_error;
      EXPECT_EQ(report.diagnostics.size(), producer == (vgpr_field ? 9u : 0u))
          << diagnostic_summary(report);
    }
  }
}

TEST(WaitcheckTest, Gfx942ReportsMissingLgkmcntBeforeScalarLoadUse) {
  std::vector<uint32_t> program;
  append_gfx942_s_load_dword_s4_s0(program);
  append_gfx942_s_mov_b32_s8_s4(program);

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA3);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Ds);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Use);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::SGPR);
  EXPECT_EQ(report.diagnostics[0].reg.index, 4u);
  EXPECT_EQ(report.diagnostics[0].required_count, 0u);
  EXPECT_NE(report.diagnostics[0].message.find("s_waitcnt lgkmcnt(0)"), std::string::npos);
}

TEST(WaitcheckTest, Gfx942AcceptsSWaitcntLgkmcntZeroBeforeScalarLoadUse) {
  std::vector<uint32_t> program;
  append_gfx942_s_load_dword_s4_s0(program);
  append_gfx942_s_waitcnt_lgkmcnt_0(program);
  append_gfx942_s_mov_b32_s8_s4(program);

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA3);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx942ReportsReadAndWriteWhileDsReplacementIsPending) {
  std::vector<uint32_t> program;
  append_gfx942_v_mov_b32_v4_v2(program);   // Initialize v4.
  append_gfx942_ds_read_b32_v4_v0(program); // Start an asynchronous replacement.
  append_gfx942_v_mov_b32_v5_v4(program);   // Cannot assume the old v4 value.
  append_gfx942_v_mov_b32_v4_v2(program);   // Cannot overwrite a pending physical destination.
  append_gfx942_s_waitcnt_lgkmcnt_0(program);
  append_gfx942_v_mov_b32_v5_v4(program); // The load has completed.

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA3);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 2u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Ds);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Use);
  EXPECT_EQ(report.diagnostics[0].reg, (RegisterRef{RegClass::VGPR, 4, 1}));
  EXPECT_EQ(report.diagnostics[0].producer_section_offset, 4u);
  EXPECT_EQ(report.diagnostics[0].section_offset, 12u);
  EXPECT_EQ(report.diagnostics[0].required_count, 0u);
  EXPECT_EQ(report.diagnostics[1].counter, WaitCounterKind::Ds);
  EXPECT_EQ(report.diagnostics[1].access, WaitcheckAccessKind::Def);
  EXPECT_EQ(report.diagnostics[1].reg, (RegisterRef{RegClass::VGPR, 4, 1}));
  EXPECT_EQ(report.diagnostics[1].producer_section_offset, 4u);
  EXPECT_EQ(report.diagnostics[1].section_offset, 16u);
  EXPECT_EQ(report.diagnostics[1].required_count, 0u);
}

TEST(WaitcheckTest, Gfx942ReportsLiveInVgprUseWhileReplacementIsPending) {
  std::vector<uint32_t> program;
  append_gfx942_v_mov_b32_v5_v4(program);   // Read the ABI/live-in v4 value.
  append_gfx942_ds_read_b32_v4_v0(program); // Start an asynchronous replacement.
  append_gfx942_v_mov_b32_v5_v4(program);   // Cannot assume the old v4 value.
  append_gfx942_s_waitcnt_lgkmcnt_0(program);
  append_gfx942_v_mov_b32_v5_v4(program); // The load has completed.

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA3);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Ds);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Use);
  EXPECT_EQ(report.diagnostics[0].reg, (RegisterRef{RegClass::VGPR, 4, 1}));
  EXPECT_EQ(report.diagnostics[0].producer_section_offset, 4u);
  EXPECT_EQ(report.diagnostics[0].section_offset, 12u);
  EXPECT_EQ(report.diagnostics[0].required_count, 0u);
}

TEST(WaitcheckTest, Gfx942ReportsSynchronousOverwriteAndReadAfterDsRead) {
  std::vector<uint32_t> program;
  append_gfx942_ds_read_b32_v4_v0(program); // Issue an asynchronous load.
  append_gfx942_v_mov_b32_v4_v2(program);   // Overwrite its pending destination.
  append_gfx942_v_mov_b32_v5_v4(program);   // The overwrite does not retire the load.
  append_gfx942_s_waitcnt_lgkmcnt_0(program);
  append_gfx942_v_mov_b32_v5_v4(program); // The load has completed.

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA3);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 2u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Ds);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Def);
  EXPECT_EQ(report.diagnostics[0].reg, (RegisterRef{RegClass::VGPR, 4, 1}));
  EXPECT_EQ(report.diagnostics[0].producer_section_offset, 0u);
  EXPECT_EQ(report.diagnostics[0].section_offset, 8u);
  EXPECT_EQ(report.diagnostics[0].required_count, 0u);
  EXPECT_EQ(report.diagnostics[1].counter, WaitCounterKind::Ds);
  EXPECT_EQ(report.diagnostics[1].access, WaitcheckAccessKind::Use);
  EXPECT_EQ(report.diagnostics[1].reg, (RegisterRef{RegClass::VGPR, 4, 1}));
  EXPECT_EQ(report.diagnostics[1].producer_section_offset, 0u);
  EXPECT_EQ(report.diagnostics[1].section_offset, 12u);
  EXPECT_EQ(report.diagnostics[1].required_count, 0u);
}

TEST(WaitcheckTest, CdnaPartialWaitDoesNotMakeNewerPhysicalDestinationReady) {
  for (rj_code_arch_t arch : {ROCJITSU_CODE_ARCH_CDNA3, ROCJITSU_CODE_ARCH_CDNA4}) {
    SCOPED_TRACE(arch);
    for (bool wait_for_newer : {false, true}) {
      SCOPED_TRACE(wait_for_newer);
      std::vector<uint32_t> program;
      // Same-class VMEM writes are ordered, but waiting for the older one
      // does not make the destination readable while the newer one is pending.
      append_gfx942_buffer_load_dword(program, 0, 8);
      append_gfx942_buffer_load_dword(program, 0, 8);
      append_gfx942_s_waitcnt_vmcnt_1(program);
      if (wait_for_newer)
        append_gfx942_s_waitcnt_vmcnt_0(program);
      append_gfx942_v_mov_b32_v1_v0(program);

      auto report = analyze_waitcnts(program, arch);
      ASSERT_TRUE(report.supported) << report.analysis_error;
      if (wait_for_newer) {
        EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
      } else {
        ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
        EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Load);
        EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Use);
        EXPECT_EQ(report.diagnostics[0].reg, (RegisterRef{RegClass::VGPR, 0, 1}));
        EXPECT_EQ(report.diagnostics[0].producer_section_offset, 8u);
        EXPECT_EQ(report.diagnostics[0].section_offset, 20u);
        EXPECT_EQ(report.diagnostics[0].required_count, 0u);
      }
    }
  }
}

TEST(WaitcheckTest, CdnaMaskedOverwriteDoesNotHidePendingResultAfterExecRestore) {
  for (rj_code_arch_t arch : {ROCJITSU_CODE_ARCH_CDNA3, ROCJITSU_CODE_ARCH_CDNA4}) {
    SCOPED_TRACE(arch);
    for (bool wait_before_overwrite : {false, true}) {
      SCOPED_TRACE(wait_before_overwrite);
      std::vector<uint32_t> program;
      append_gfx942_ds_read_b32_v4_v0(program);
      if (wait_before_overwrite)
        append_gfx942_s_waitcnt_lgkmcnt_0(program);
      program.push_back(0xbe842000u); // s_and_saveexec_b64 s[4:5], s[0:1]
      append_gfx942_v_mov_b32_v4_v2(program);
      program.push_back(0xbefe0104u); // s_mov_b64 exec, s[4:5]
      append_gfx942_v_mov_b32_v5_v4(program);
      append_gfx942_s_waitcnt_lgkmcnt_0(program);
      append_gfx942_v_mov_b32_v5_v4(program);

      auto report = analyze_waitcnts(program, arch);
      ASSERT_TRUE(report.supported) << report.analysis_error;
      if (wait_before_overwrite) {
        EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
      } else {
        ASSERT_EQ(report.diagnostics.size(), 2u) << diagnostic_summary(report);
        EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Def);
        EXPECT_EQ(report.diagnostics[0].section_offset, 12u);
        EXPECT_EQ(report.diagnostics[1].access, WaitcheckAccessKind::Use);
        EXPECT_EQ(report.diagnostics[1].section_offset, 20u);
        for (const auto &diagnostic : report.diagnostics) {
          EXPECT_EQ(diagnostic.counter, WaitCounterKind::Ds);
          EXPECT_EQ(diagnostic.reg, (RegisterRef{RegClass::VGPR, 4, 1}));
          EXPECT_EQ(diagnostic.producer_section_offset, 0u);
          EXPECT_EQ(diagnostic.required_count, 0u);
        }
      }
    }
  }
}

TEST(WaitcheckTest, Gfx942ReportsPendingVgprUseWithOtherInitializedRegisters) {
  std::vector<uint32_t> program;
  append_gfx942_v_mov_b32_v4_v2(program); // An unrelated initialized register.
  append_gfx942_buffer_load_dword_v0_v8_s0_offen(program);
  append_gfx942_v_mov_b32_v4_v2(program); // Another unrelated definition.
  append_gfx942_v_mov_b32_v1_v0(program); // Consume the pending load result.

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA3);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Load);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Use);
  EXPECT_EQ(report.diagnostics[0].reg, (RegisterRef{RegClass::VGPR, 0, 1}));
}

TEST(WaitcheckTest, Gfx942ReportsPendingCrossCounterAsyncReplacement) {
  std::vector<uint32_t> program;
  append_gfx942_ds_read_b32_v4_v0(program);
  append_gfx942_global_load_dword_v4_v2(program);

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA3);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Ds);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Def);
  EXPECT_EQ(report.diagnostics[0].reg, (RegisterRef{RegClass::VGPR, 4, 1}));
}

TEST(WaitcheckTest, Gfx1100DecodesLegacyAndSopkWaitcnts) {
  std::vector<uint32_t> program;
  append_gfx1100_s_waitcnt_vmcnt_0(program);
  append_gfx1100_s_waitcnt_vscnt_0(program);
  append_gfx1100_s_waitcnt_vmcnt_sopk_0(program);
  append_gfx1100_s_waitcnt_expcnt_sopk_0(program);
  append_gfx1100_s_waitcnt_lgkmcnt_sopk_0(program);

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_RDNA3);
  ASSERT_NE(decoder, nullptr);

  const std::array<std::string_view, 5> expected = {
      "s_waitcnt", "s_waitcnt_vscnt", "s_waitcnt_vmcnt", "s_waitcnt_expcnt", "s_waitcnt_lgkmcnt"};
  size_t word_index = 0;
  for (const std::string_view mnemonic : expected) {
    std::unique_ptr<Instruction> inst(decode_valid(*decoder, &program[word_index]));
    ASSERT_NE(inst, nullptr);
    EXPECT_EQ(inst->mnemonic(), mnemonic);
    EXPECT_TRUE(inst->is_waitcnt());
    word_index += static_cast<size_t>(inst->size() / sizeof(uint32_t));
  }
  EXPECT_EQ(word_index, program.size());
}

TEST(WaitcheckTest, Gfx1100ReportsMissingVmcntBeforeGlobalLoadUse) {
  std::vector<uint32_t> program;
  append_gfx1100_global_load_b32_v0_v8_s0(program);
  append_gfx1100_v_mov_b32_v1_v0(program);

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA3);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Load);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Use);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::VGPR);
  EXPECT_EQ(report.diagnostics[0].reg.index, 0u);
  EXPECT_EQ(report.diagnostics[0].required_count, 0u);
  EXPECT_NE(report.diagnostics[0].message.find("s_waitcnt vmcnt(0)"), std::string::npos);
}

TEST(WaitcheckTest, Rdna35ReportsMissingVmcntBeforeGlobalLoadUse) {
  std::vector<uint32_t> program;
  append_gfx1100_global_load_b32_v0_v8_s0(program);
  append_gfx1100_v_mov_b32_v1_v0(program);

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA3_5);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Load);
  EXPECT_EQ(report.diagnostics[0].required_count, 0u);
  EXPECT_NE(report.diagnostics[0].message.find("s_waitcnt vmcnt(0)"), std::string::npos);
}

TEST(WaitcheckTest, Rdna4ComplementaryD16LoadsPreserveIndependentHalves) {
  const std::vector<uint32_t> program{
      0xee08007c, 0x00000004, 0x00000000, // global_load_d16_b16 v4, v[0:1], off
      0xee08c07c, 0x00000004, 0x00000000, // global_load_d16_hi_b16 v4, v[0:1], off
      0xda980000, 0x04000000,             // ds_load_u16_d16 v4, v0
      0xda9c0000, 0x04000000,             // ds_load_u16_d16_hi v4, v0
  };
  for (size_t prefix : {6u, 10u}) {
    const size_t first = prefix == 6 ? 0 : 6;
    auto pair = std::vector<uint32_t>(program.begin() + first, program.begin() + prefix);
    auto report = analyze_waitcnts(pair, ROCJITSU_CODE_ARCH_RDNA4);
    EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);

    pair.push_back(0x7e100304); // v_mov_b32_e32 v8, v4
    report = analyze_waitcnts(pair, ROCJITSU_CODE_ARCH_RDNA4);
    ASSERT_EQ(report.diagnostics.size(), 2u) << diagnostic_summary(report);
    EXPECT_EQ(report.diagnostics[0].reg.index, 4u);
    EXPECT_EQ(report.diagnostics[1].reg.index, 4u);
  }
}

TEST(WaitcheckTest, Rdna4D16AddressAndPackedFormatUseFullVgpr) {
  const std::vector<uint32_t> address{
      0xee08007c, 0x00000004, 0x00000000, // global_load_d16_b16 v4, v[0:1], off
      0xee08c07c, 0x00000004, 0x00000004, // global_load_d16_hi_b16 v4, v[4:5], off
  };
  auto report = analyze_waitcnts(address, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics.front().reg.index, 4u);

  const std::vector<uint32_t> packed{
      0xc4024003, 0x00801004, 0x00000000, // buffer_load_d16_format_xy v4, off, s[8:11], s3
      0xc4094003, 0x00801004, 0x00000000, // buffer_store_d16_hi_b16 v4, off, s[8:11], s3
  };
  report = analyze_waitcnts(packed, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics.front().reg.index, 4u);
}

TEST(WaitcheckTest, Rdna4D16StoreAddressReadsBothHalvesWhenAliasedWithData) {
  for (bool high_load : {false, true}) {
    for (uint32_t addr : {0u, 3u, 4u}) {
      SCOPED_TRACE(high_load);
      SCOPED_TRACE(addr);
      // Opposite halves of v4, with v4 outside, second or first in the address pair.
      const std::vector<uint32_t> program{
          high_load ? 0xee08c07cu : 0xee08007cu, 0x00000004, 0,    // global_load_d16[_hi]_b16
          high_load ? 0xee06407cu : 0xee09407cu, 0x02000000, addr, // global_store[_d16_hi]_b16
      };
      const auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);
      ASSERT_EQ(report.diagnostics.size(), addr ? 1u : 0u) << diagnostic_summary(report);
      if (addr) {
        EXPECT_EQ(report.diagnostics.front().access, WaitcheckAccessKind::Use);
        EXPECT_EQ(report.diagnostics.front().reg.index, 4u);
      }
    }
  }
}

TEST(WaitcheckTest, ScratchStoreReadsAllDataWordsAndOnlyTheSelectedHalf) {
  for (auto arch : {ROCJITSU_CODE_ARCH_RDNA3, ROCJITSU_CODE_ARCH_RDNA3_5}) {
    for (uint32_t address : {0x00020000u, 0x007c0000u, 0x00820000u, 0x00fc0000u}) {
      SCOPED_TRACE(arch);
      SCOPED_TRACE(address);
      // Toggling SVE and SADDR must not change the width of the b64 payload v[4:5].
      const std::vector<uint32_t> wide{
          0xdc520000,
          0x057c0000, // global_load_b32 v5, v[0:1], off
          0xdc6d0000,
          address | 0x00000400u, // scratch_store_b64 off/v0, v[4:5], off/s2
      };
      const auto full = analyze_waitcnts(wide, arch);
      ASSERT_EQ(full.diagnostics.size(), 1u) << diagnostic_summary(full);
      EXPECT_EQ(full.diagnostics.front().reg.index, 5u);
      const std::vector<uint32_t> half{
          0xda980000,
          0x04000000, // ds_load_u16_d16 v4, v0
          0xdc950000,
          address | 0x00000400u, // scratch_store_d16_hi_b16 off/v0, v4, off/s2
      };
      const auto partial = analyze_waitcnts(half, arch);
      EXPECT_TRUE(partial.diagnostics.empty()) << diagnostic_summary(partial);
    }
  }
  const std::vector<uint32_t> cdna3{
      0xdc508000,
      0x057f0000, // global_load_dword v5, v[0:1], off
      0xdc744000,
      0x00020400, // scratch_store_dwordx2 off, v[4:5], s2
  };
  const auto full = analyze_waitcnts(cdna3, ROCJITSU_CODE_ARCH_CDNA3);
  ASSERT_EQ(full.diagnostics.size(), 1u) << diagnostic_summary(full);
  EXPECT_EQ(full.diagnostics.front().reg.index, 5u);
}

TEST(WaitcheckTest, Rdna35AcceptsPendingD16LoadToOppositeVgprHalf) {
  std::vector<uint32_t> program;
  append_gfx1150_ds_load_u8_d16_hi_v1_v1(program);
  append_gfx1150_ds_load_u8_d16_v1_v5_offset512(program);
  append_gfx1150_s_waitcnt_lgkmcnt_1(program);
  append_gfx1150_global_store_d16_hi_b8_v2_v1(program);

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA3_5);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Rdna35ReportsPendingD16LoadToSameVgprHalf) {
  std::vector<uint32_t> program;
  append_gfx1150_ds_load_u8_d16_v1_v5_offset512(program);
  append_gfx1150_ds_load_u8_d16_hi_v1_v6(program);
  append_gfx1150_s_waitcnt_lgkmcnt_1(program);
  append_gfx1150_global_store_d16_hi_b8_v2_v1(program);

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA3_5);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Ds);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Use);
  EXPECT_EQ(report.diagnostics[0].reg, (RegisterRef{RegClass::VGPR, 1, 1}));
  EXPECT_EQ(report.diagnostics[0].required_count, 0u);
}

TEST(WaitcheckTest, Gfx1100AcceptsLegacySWaitcntVmcntZeroBeforeGlobalLoadUse) {
  std::vector<uint32_t> program;
  append_gfx1100_global_load_b32_v0_v8_s0(program);
  append_gfx1100_s_waitcnt_vmcnt_0(program);
  append_gfx1100_v_mov_b32_v1_v0(program);

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA3);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx1100AcceptsSopkSWaitcntVmcntZeroBeforeGlobalLoadUse) {
  std::vector<uint32_t> program;
  append_gfx1100_global_load_b32_v0_v8_s0(program);
  append_gfx1100_s_waitcnt_vmcnt_sopk_0(program);
  append_gfx1100_v_mov_b32_v1_v0(program);

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA3);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx1100SopkSWaitcntVmcntOneLeavesYoungerLoadPending) {
  std::vector<uint32_t> program;
  append_gfx1100_global_load_b32_v0_v8_s0(program);
  append_gfx1100_global_load_b32_v2_v8_s0(program);
  append_gfx1100_s_waitcnt_vmcnt_sopk_1(program);
  append_gfx1100_v_mov_b32_v1_v0(program);
  append_gfx1100_v_mov_b32_v3_v2(program);

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA3);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Load);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Use);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::VGPR);
  EXPECT_EQ(report.diagnostics[0].reg.index, 2u);
  EXPECT_EQ(report.diagnostics[0].required_count, 0u);
}

TEST(WaitcheckTest, Gfx1100ReportsMissingLgkmcntBeforeScalarLoadUse) {
  std::vector<uint32_t> program;
  append_gfx1100_s_load_b32_s4_s0(program);
  append_gfx1100_s_mov_b32_s8_s4(program);

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA3);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Ds);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Use);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::SGPR);
  EXPECT_EQ(report.diagnostics[0].reg.index, 4u);
  EXPECT_EQ(report.diagnostics[0].required_count, 0u);
  EXPECT_NE(report.diagnostics[0].message.find("s_waitcnt lgkmcnt(0)"), std::string::npos);
}

TEST(WaitcheckTest, Gfx1100AcceptsSopkSWaitcntLgkmcntZeroBeforeScalarLoadUse) {
  std::vector<uint32_t> program;
  append_gfx1100_s_load_b32_s4_s0(program);
  append_gfx1100_s_waitcnt_lgkmcnt_sopk_0(program);
  append_gfx1100_s_mov_b32_s8_s4(program);

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA3);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx1100Wave32Vop3CarryOutDoesNotDefineAdjacentSgpr) {
  std::vector<uint32_t> program;
  append_inst(program, s_load_b32(2, 0));
  append_gfx1100_v_add_co_u32_v0_s1_s4_v0(program);

  const auto image =
      rocjitsu::waitcheck_test::make_gfx_code_object(program, EF_AMDGPU_MACH_AMDGCN_GFX1100, true);
  AmdGpuCodeObject code_object(image.data(), image.size());
  auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_RDNA3);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx1100Wave64Vop3CarryOutDefinesBothMaskSgprs) {
  std::vector<uint32_t> program;
  append_inst(program, s_load_b32(2, 0));
  append_gfx1100_v_add_co_u32_v0_s1_s4_v0(program);

  const auto image =
      rocjitsu::waitcheck_test::make_gfx_code_object(program, EF_AMDGPU_MACH_AMDGCN_GFX1100);
  AmdGpuCodeObject code_object(image.data(), image.size());
  auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_RDNA3);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Ds);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Def);
  EXPECT_EQ(report.diagnostics[0].reg, (RegisterRef{RegClass::SGPR, 2, 1}));
}

TEST(WaitcheckTest, Gfx1100Wave32Vop3CarryInDoesNotUseAdjacentSgpr) {
  std::vector<uint32_t> program;
  append_inst(program, s_load_b32(2, 0));
  append_gfx1100_v_add_co_ci_u32_v2_null_s25_0_s1(program);

  const auto image =
      rocjitsu::waitcheck_test::make_gfx_code_object(program, EF_AMDGPU_MACH_AMDGCN_GFX1100, true);
  AmdGpuCodeObject code_object(image.data(), image.size());
  auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_RDNA3);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx1100Wave64Vop3CarryInUsesBothMaskSgprs) {
  std::vector<uint32_t> program;
  append_inst(program, s_load_b32(2, 0));
  append_gfx1100_v_add_co_ci_u32_v2_null_s25_0_s1(program);

  const auto image =
      rocjitsu::waitcheck_test::make_gfx_code_object(program, EF_AMDGPU_MACH_AMDGCN_GFX1100);
  AmdGpuCodeObject code_object(image.data(), image.size());
  auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_RDNA3);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Ds);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Use);
  EXPECT_EQ(report.diagnostics[0].reg, (RegisterRef{RegClass::SGPR, 2, 1}));
}

TEST(WaitcheckTest, Gfx950ReportsMissingVmcntBeforeBufferLoadUse) {
  std::vector<uint32_t> program;
  append_gfx950_buffer_load_dword_v0_v8_s0_offen(program);
  append_gfx950_v_mov_b32_v1_v0(program);

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA4);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Load);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Use);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::VGPR);
  EXPECT_EQ(report.diagnostics[0].reg.index, 0u);
  EXPECT_EQ(report.diagnostics[0].required_count, 0u);
  EXPECT_NE(report.diagnostics[0].message.find("s_waitcnt vmcnt(0)"), std::string::npos);
}

TEST(WaitcheckTest, Gfx950AcceptsSWaitcntVmcntZeroBeforeBufferLoadUse) {
  std::vector<uint32_t> program;
  append_gfx950_buffer_load_dword_v0_v8_s0_offen(program);
  append_gfx950_s_waitcnt_vmcnt_0(program);
  append_gfx950_v_mov_b32_v1_v0(program);

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA4);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx950ScalarExecTestGuardsGlobalLoadDrain) {
  enum class GuardShape { MoveExec, Split, SplitWait, ScalarExec, ScalarExecWithUnrelatedLoad };
  const auto check = [](bool branch_on_nonempty, uint32_t mask, bool guarded_wait, bool later_wait,
                        bool expect_missing, GuardShape shape = GuardShape::MoveExec) {
    const bool scalar_exec =
        shape == GuardShape::ScalarExec || shape == GuardShape::ScalarExecWithUnrelatedLoad;
    std::vector<uint32_t> words{
        0xbf068104,             // s_cmp_eq_u32 s4, 1
        0xbf850001,             // s_cbranch_scc1 .Lload
        0xbefe0180,             // s_mov_b64 exec, 0
        0xdc548000, 0x007f0004, // .Lload: global_load_dwordx2 v[0:1], v[4:5], off
        mask,                   // s_mov_b64 s[6:7], -1 or 0
        0x86827e06              // s_and_b64 s[2:3], s[6:7], exec
    };
    size_t producer = 12;
    if (scalar_exec)
      words[2] = 0x86fe807e; // s_and_b64 exec, exec, 0
    if (shape == GuardShape::Split) {
      // A second incoming edge targets S_AND without executing the producer.
      words.insert(words.begin(), {0xbf068205, 0xbf850006});
      producer += 8;
    }
    if (shape == GuardShape::SplitWait) {
      // An independent edge enters the guarded wait after the scalar prefix.
      words.insert(words.begin(), {0xbf068205, 0xbf850000});
      producer += 8;
    }
    if (branch_on_nonempty)
      words.insert(words.end(), {0xbf850001, shape == GuardShape::SplitWait ? 0xbf820002 : 0xbf820001});
    else
      words.push_back(shape == GuardShape::SplitWait ? 0xbf840002 : 0xbf840001);
    if (shape == GuardShape::SplitWait) {
      words.push_back(0xbe8a0080); // s_mov_b32 s10, 0
      words[1] |= words.size() - 2;
    }
    words.push_back(guarded_wait ? 0xbf8c0f70 : 0xbf8ccf7f);
    if (scalar_exec)
      words.push_back(0x87fe817e); // s_or_b64 exec, exec, 1
    else
      words.push_back(0xbefe0181); // s_mov_b64 exec, 1
    if (shape == GuardShape::ScalarExecWithUnrelatedLoad) {
      producer = words.size() * sizeof(uint32_t);
      words.insert(words.end(), {0xdc548000, 0x007f0006}); // global_load_dwordx2 v[0:1], v[6:7], off
    }
    words.push_back(later_wait ? 0xbf8c0070 : 0xbf8c0071);
    const size_t consumer = 4 * words.size();
    words.insert(words.end(), {0x7e120300, 0xbf810000}); // v_mov_b32 v9, v0, end
    const auto image = waitcheck_test::make_gfx_code_object(words, EF_AMDGPU_MACH_AMDGCN_GFX950);
    AmdGpuCodeObject obj(image.data(), image.size());
    const auto report = analyze_waitcnts_for_kernel(obj, ROCJITSU_CODE_ARCH_CDNA4, 0);
    ASSERT_TRUE(report.supported && report.complete) << report.analysis_error;
    const bool missing = std::ranges::any_of(report.diagnostics, [&](const auto &d) {
      return d.counter == WaitCounterKind::Load && d.producer_section_offset == producer &&
             d.section_offset == consumer && d.access == WaitcheckAccessKind::Use;
    });
    EXPECT_EQ(missing, expect_missing) << diagnostic_summary(report);
  };

  constexpr uint32_t all_lanes = 0xbe8601c1;
  check(false, all_lanes, true, true, false);
  check(false, all_lanes, true, false, false);
  check(true, all_lanes, true, false, false);
  check(false, all_lanes, true, false, false, GuardShape::Split);
  check(false, all_lanes, true, false, false, GuardShape::SplitWait);
  check(true, all_lanes, true, false, false, GuardShape::SplitWait);
  check(false, all_lanes, false, false, true, GuardShape::SplitWait);
  check(false, all_lanes, false, false, true);
  check(false, 0xbe860180, true, false, true);
  check(false, all_lanes, true, false, false, GuardShape::ScalarExec);
  check(false, all_lanes, false, false, true, GuardShape::ScalarExec);
  check(false, all_lanes, true, false, true, GuardShape::ScalarExecWithUnrelatedLoad);
}

TEST(WaitcheckTest, Gfx950ScratchStoreOnlyReadsLanesActiveAtTheStore) {
  const auto check = [](std::vector<uint32_t> after_save, std::vector<uint32_t> after_load,
                        uint32_t select, bool expect_missing) {
    std::vector<uint32_t> words{0xbe88206a}; // s_and_saveexec_b64 s[8:9], vcc
    words.insert(words.end(), after_save.begin(), after_save.end());
    const auto producer = 4 * words.size();
    words.insert(words.end(), {0xdc500000, 0x04000002, // flat_load_dword v4, v[2:3]
                               0xbf8c0f70,             // s_waitcnt vmcnt(0)
                               0x7e0e0304});           // v_mov_b32_e32 v7, v4
    words.insert(words.end(), after_load.begin(), after_load.end());
    words.insert(words.end(), {0x87fe087e,   // s_or_b64 exec, exec, s[8:9]
                               0xbe88017e,   // s_mov_b64 s[8:9], exec
                               0x888ac108,   // s_xor_b64 s[10:11], s[8:9], -1
                               0x87fe0c7e,   // s_or_b64 exec, exec, s[12:13]
                               select,       // s_orn2_b64 s[8:9], s[10:11], exec
                               0x87fe0e7e,   // s_or_b64 exec, exec, s[14:15]
                               0xbe8c2008}); // s_and_saveexec_b64 s[12:13], s[8:9]
    const auto consumer = 4 * words.size();
    words.insert(words.end(), {0xdc784024, 0x00210200,   // scratch_store_dwordx3 v[2:4]
                               0xbf8cc07f, 0xbf810000}); // LGKM0, end
    const auto elf = waitcheck_test::make_gfx_code_object(words, EF_AMDGPU_MACH_AMDGCN_GFX950);
    AmdGpuCodeObject obj(elf.data(), elf.size());
    const auto report = analyze_waitcnts_for_kernel(obj, ROCJITSU_CODE_ARCH_CDNA4, 0);
    ASSERT_TRUE(report.supported && report.complete) << report.analysis_error;
    const auto missing = [&](size_t pc) {
      return std::ranges::any_of(report.diagnostics, [&](const auto &d) {
        return d.counter == WaitCounterKind::Ds && d.producer_section_offset == producer &&
               d.section_offset == pc;
      });
    };
    EXPECT_TRUE(missing(producer + 12)) << diagnostic_summary(report);
    EXPECT_EQ(missing(consumer), expect_missing) << diagnostic_summary(report);
  };

  constexpr uint32_t complement = 0x8a887e0a;
  check({}, {}, complement, false);
  check({0xbe880000}, {}, complement, true);  // s_mov_b32 s8, s0 can clobber lanes 0–31
  check({0xbe890000}, {}, complement, true);  // s_mov_b32 s9, s0 can clobber lanes 32–63
  check({}, {0xbe880180}, complement, false); // Subsequent MOV EXEC captures the issuing lanes
  check({0x87fe0c7e}, {}, complement, true);  // Widen EXEC before the load
  check({}, {}, 0xbe8801c1, true);            // Select every lane instead of the complement
  check({0xbf850001, 0xbe890000}, {}, complement, true); // One incoming path clobbers the high half
  check({}, {0xbf85fffb}, complement, true);             // The producer can be reissued
  check({}, {0xbf850001, 0xbf800000}, complement, false); // Both paths retain the same facts
}

TEST(WaitcheckTest, Gfx1100ScratchStoreFollowsTheSavedLowExecWord) {
  const std::vector<uint32_t> code{
      0xbe88007e,            // s_mov_b32 s8, exec_lo
      0x7d945080,            // v_cmpx_eq_u32_e32 0, v40
      0xdc500000, 0x47c0002, // flat_load_b32 v4, v[2:3]
      0xbf8903f7,            // s_waitcnt vmcnt(0)
      0x7e0e0304,            // v_mov_b32_e32 v7, v4
      0x8c7e087e,            // s_or_b32 exec_lo, exec_lo, s8
      0xbe88007e,            // s_mov_b32 s8, exec_lo
      0x8d0ac108,            // s_xor_b32 s10, s8, -1
      0x8c7e0c7e,            // s_or_b32 exec_lo, exec_lo, s12
      0x92087e0a,            // s_or_not1_b32 s8, s10, exec_lo
      0x8c7e0e7e,            // s_or_b32 exec_lo, exec_lo, s14
      0xbe8c2008,            // s_and_saveexec_b32 s12, s8
      0xdc750024, 0x210200,  // scratch_store_b128 off, v[2:5], s33 offset:36
      0xbf89fc07,            // s_waitcnt lgkmcnt(0)
      0xbfb00000,            // s_endpgm
  };
  for (bool replace_mask : {false, true}) {
    auto words = code;
    if (replace_mask)
      words[10] = 0xbe8800c1; // s_mov_b32 s8, -1
    const auto elf =
        waitcheck_test::make_gfx_code_object(words, EF_AMDGPU_MACH_AMDGCN_GFX1100, true);
    AmdGpuCodeObject obj(elf.data(), elf.size());
    const auto report = analyze_waitcnts_for_kernel(obj, ROCJITSU_CODE_ARCH_RDNA3, 0);
    ASSERT_TRUE(report.supported && report.complete) << report.analysis_error;
    EXPECT_EQ(std::ranges::any_of(report.diagnostics,
                                  [&](const auto &d) {
                                    return d.counter == WaitCounterKind::Ds &&
                                           d.producer_section_offset == 8 && d.section_offset == 52;
                                  }),
              replace_mask)
        << diagnostic_summary(report);
  }
}

TEST(WaitcheckTest, Gfx1100RestoredIssuingLaneCreditsAYoungerFlatBeforeScratchOverwrite) {
  for (const auto &[restore, wait, missing] :
       std::initializer_list<std::tuple<uint32_t, uint32_t, bool>>{
           {0x8c7e007e, 0xbf890017, false}, // restore EXEC from s0; LGKM1
           {0xbe800080, 0xbf890017, true},  // clobber s0 instead of restoring EXEC
           {0x8c7e007e, 0xbf890027, true},  // LGKM2 leaves the older result outstanding
           {0xbe800080, 0xbf890007, false}, // LGKM0 also clears the result
       }) {
    const std::vector<uint32_t> code{
        0xdc540000, 0x017c0004,             // flat_load_b64 v[1:2], v[4:5]
        0xbf8903f7,                         // VM0 makes zero-EXEC FLAT issue depend on active lanes
        0xbe803000,                         // s_and_not1_saveexec_b32 s0, s0
        0xbfa50001, 0xbf89fc07,             // Skip LGKM0 if the new EXEC is zero
        restore,    0xdc500000, 0x067c0004, // flat_load_b32 v6, v[4:5]
        wait,                               // VM0 plus tested LGKM threshold
        0xbefe00c1,                         // s_mov_b32 exec_lo, -1
        0xdc5d000c, 0x00210000,             // scratch_load_b128 v[0:3], off, s33 offset:12
        0xbf89fc07, 0xbfb00000,
    };
    const auto elf =
        waitcheck_test::make_gfx_code_object(code, EF_AMDGPU_MACH_AMDGCN_GFX1100, true);
    AmdGpuCodeObject obj(elf.data(), elf.size());
    const auto report = analyze_waitcnts_for_kernel(obj, ROCJITSU_CODE_ARCH_RDNA3, 0);
    ASSERT_TRUE(report.supported && report.complete) << report.analysis_error;
    EXPECT_EQ(std::ranges::any_of(report.diagnostics,
                                  [](const auto &d) {
                                    return d.counter == WaitCounterKind::Ds &&
                                           d.producer_section_offset == 0 && d.section_offset == 44;
                                  }),
              missing)
        << std::hex << restore << " wait=" << wait << diagnostic_summary(report);
  }
}

TEST(WaitcheckTest, Gfx11MaskedConsumerProvesTheYoungerFlatIssued) {
  enum class Case { Covered, Looser, MaskChange, Bypass, NewProducer };
  for (const auto machine : {EF_AMDGPU_MACH_AMDGCN_GFX1100, EF_AMDGPU_MACH_AMDGCN_GFX1150}) {
    const auto arch = machine == EF_AMDGPU_MACH_AMDGCN_GFX1100 ? ROCJITSU_CODE_ARCH_RDNA3
                                                              : ROCJITSU_CODE_ARCH_RDNA3_5;
    for (const auto test :
         {Case::Covered, Case::Looser, Case::MaskChange, Case::Bypass, Case::NewProducer}) {
      std::vector<uint32_t> words{
          0xdc540080, 0x067c0000, // flat_load_b64 v[6:7], v[0:1] offset:128
          0xbf8903f7,             // VM0 permits a zero-EXEC younger FLAT to be skipped
          0xbe803000,             // s_and_not1_saveexec_b32 s0, s0
      };
      if (test == Case::Bypass)
        words.push_back(0xbfa10002); // s_cbranch_scc0 skips the younger FLAT
      words.insert(words.end(), {0xdc500014, 0x037c0004}); // flat_load_b32 v3, v[4:5]
      if (test == Case::NewProducer)
        words.insert(words.end(), {0xdc540080, 0x067c0000});
      words.push_back(test == Case::Looser ? 0xbf890427 : 0xbf890417);
      if (test == Case::MaskChange)
        words.push_back(0xbefe00c1); // s_mov_b32 exec_lo, -1
      const auto consumer = 4 * words.size();
      words.insert(words.end(), {0xdccc4088, 0x067c0500, 0xbfb00000});
      const auto elf = waitcheck_test::make_gfx_code_object(words, machine, true);
      AmdGpuCodeObject obj(elf.data(), elf.size());
      const auto report = analyze_waitcnts_for_kernel(obj, arch, 0);
      ASSERT_TRUE(report.supported && report.complete) << report.analysis_error;
      EXPECT_EQ(std::ranges::any_of(report.diagnostics,
                                    [&](const auto &d) {
                                      return d.counter == WaitCounterKind::Ds &&
                                             d.section_offset == consumer && d.reg.index == 6;
                                    }),
                test != Case::Covered)
          << static_cast<int>(test) << diagnostic_summary(report);
    }
  }
}

TEST(WaitcheckTest, Gfx1100ParityCannotUseTheWaitBeingMeasuredToProveScratchSafe) {
  const std::vector<uint32_t> words{
      0xdc540000, 0x017c0004, // flat_load_b64 v[1:2], v[4:5]
      0xbf890007,             // s_waitcnt vmcnt(0) lgkmcnt(0)
      0xdc5d000c, 0x00210000, // scratch_load_b128 v[0:3], off, s33 offset:12
      0xbfb00000,             // s_endpgm
  };
  const auto elf = waitcheck_test::make_gfx_code_object(words, EF_AMDGPU_MACH_AMDGCN_GFX1100, true);
  AmdGpuCodeObject obj(elf.data(), elf.size());
  WaitcheckOptions options;
  options.check_counter_parity = true;
  const auto report = analyze_waitcnts_for_kernel(obj, ROCJITSU_CODE_ARCH_RDNA3, 0, options);
  ASSERT_TRUE(report.supported && report.complete) << report.analysis_error;
  EXPECT_TRUE(report.passed()) << diagnostic_summary(report);
  // Both writes use the ordered VM load pipe. The LDS half of an unknown
  // FLAT address still needs the LGKM wait for the overwrite.
  EXPECT_EQ(report.counter_parity_exact, 1u);
  ASSERT_EQ(report.counter_underaccounting_diagnostics.size(), 1u);
  EXPECT_EQ(report.counter_underaccounting_diagnostics.front().counter, WaitCounterKind::Load);
}

TEST(WaitcheckTest, Gfx950NonReturningFlatAtomicDoesNotDefineVdst) {
  std::vector<uint32_t> program;
  append_gfx950_buffer_load_dword_v0_v8_s0_offen(program);
  append_gfx950_global_atomic_add_x2(program, 0, 40, 36, false);

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA4);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx950ReturningFlatAtomicDefinesVdst) {
  std::vector<uint32_t> program;
  append_gfx950_buffer_load_dword_v0_v8_s0_offen(program);
  append_gfx950_global_atomic_add_x2(program, 0, 40, 36, true);

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA4);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Def);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::VGPR);
  EXPECT_EQ(report.diagnostics[0].reg.index, 0u);
}

TEST(WaitcheckTest, Gfx950BufferAtomicAlwaysReadsPayload) {
  std::vector<uint32_t> program;
  append_gfx950_buffer_load_dword_v0_v8_s0_offen(program);
  append_gfx950_buffer_atomic_add(program, 0, false);

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA4);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Use);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::VGPR);
  EXPECT_EQ(report.diagnostics[0].reg.index, 0u);
}

TEST(WaitcheckTest, Gfx950DoesNotTreatDirectToLdsBufferLoadAsVgprProducer) {
  std::vector<uint32_t> program;
  append_gfx950_buffer_load_dwordx4_v0_s64_offen_lds(program);
  append_gfx950_v_mov_b32_v1_v0(program);

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA4);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx950DirectToLdsBufferLoadAgesVmcnt) {
  std::vector<uint32_t> program;
  append_gfx950_buffer_load_dword_v0_v8_s0_offen(program);
  append_gfx950_buffer_load_dwordx4_v0_s64_offen_lds(program);
  append_gfx950_s_waitcnt_vmcnt_1(program);
  append_gfx950_v_mov_b32_v1_v0(program);

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA4);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

// Counter-parity fixtures retain final-ISA shapes from LLVM's
// waitcnt-overflow.mir, waitcnt-flat.ll, waitcnt-sample-{out-order,waw}.mir,
// waitcnt-bvh.mir, waitcnt-gfx1250.mir, lds-direct-hazards-gfx12.mir,
// memory-legalizer-barriers.ll, and waitcnt-loop-*.mir tests, plus the
// checked-in LLVM-produced Triton fixtures and extracted ROCm PyTorch objects.
// The encoded waits are stable inputs: normal tests neither invoke LLVM nor
// weaken a kernel before analysis.
TEST(WaitcheckTest, Gfx950CounterParityMatchesRequiredVmcnt) {
  std::vector<uint32_t> program;
  append_gfx950_buffer_load_dword_v0_v8_s0_offen(program);
  append_gfx950_buffer_load_dwordx4_v0_s64_offen_lds(program);
  append_gfx950_s_waitcnt_vmcnt_1(program);
  append_gfx950_v_mov_b32_v1_v0(program);

  WaitcheckOptions options;
  options.check_counter_parity = true;
  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA4, options);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.passed()) << diagnostic_summary(report);
  EXPECT_EQ(report.counter_parity_wait_groups, 1u);
  EXPECT_EQ(report.counter_parity_fields_checked, 1u);
  EXPECT_EQ(report.counter_parity_exact, 1u);
  EXPECT_EQ(report.counter_underaccounting_observed, 0u);
  EXPECT_TRUE(report.counter_underaccounting_diagnostics.empty());
}

TEST(WaitcheckTest, Gfx950CounterParityMatchesPendingReplacementDefinitionsLikeLlvm) {
  std::vector<uint32_t> program;
  append_gfx950_v_mov_b32_v0_v2(program); // Initialize v0.
  append_gfx950_v_mov_b32_v1_v0(program); // Initialize v1.
  append_gfx950_buffer_load_dword_v0_v8_s0_offen(program);
  append_gfx950_buffer_load_dword_v1_v8_s0_offen(program);
  append_gfx950_s_waitcnt_vmcnt_0(program);
  program.push_back(0x68040300u); // v_add_u32_e32 v2, v0, v1.

  WaitcheckOptions options;
  options.check_counter_parity = true;
  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA4, options);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.passed()) << diagnostic_summary(report);
  EXPECT_EQ(report.counter_parity_wait_groups, 1u);
  EXPECT_EQ(report.counter_parity_fields_checked, 1u);
  EXPECT_EQ(report.counter_parity_exact, 1u);
  EXPECT_EQ(report.counter_underaccounting_observed, 0u);
  EXPECT_TRUE(report.counter_underaccounting_diagnostics.empty());
}

TEST(WaitcheckTest, Gfx950CounterParityForcesZeroForDsWithPendingScalarMemory) {
  std::vector<uint32_t> program;
  append_gfx950_ds_read_b32_v0_v4(program);
  append_gfx950_s_load_dword_s4_s0(program);
  append_gfx950_s_waitcnt_lgkmcnt_0(program);
  append_gfx950_v_mov_b32_v1_v0(program);

  WaitcheckOptions options;
  options.check_counter_parity = true;
  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA4, options);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.passed()) << diagnostic_summary(report);
  EXPECT_EQ(report.counter_parity_wait_groups, 1u);
  EXPECT_EQ(report.counter_parity_fields_checked, 1u);
  EXPECT_EQ(report.counter_parity_exact, 1u);
  EXPECT_EQ(report.counter_underaccounting_observed, 0u);
}

TEST(WaitcheckTest, Gfx950RejectsPartialLgkmcntForDsWithPendingScalarMemory) {
  std::vector<uint32_t> program;
  append_gfx950_ds_read_b32_v0_v4(program);
  append_gfx950_s_load_dword_s4_s0(program);
  append_gfx950_s_waitcnt_lgkmcnt_1(program);
  append_gfx950_v_mov_b32_v1_v0(program);

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA4);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Ds);
  EXPECT_EQ(report.diagnostics[0].required_count, 0u);
  EXPECT_EQ(report.diagnostics[0].producer_instruction, "ds_read_b32 v0, v4");
}

TEST(WaitcheckTest, Gfx950CounterParityReportsStrongerEmittedVmcnt) {
  std::vector<uint32_t> program;
  append_gfx950_buffer_load_dword_v0_v8_s0_offen(program);
  append_gfx950_buffer_load_dwordx4_v0_s64_offen_lds(program);
  append_gfx950_s_waitcnt_vmcnt_0(program);
  append_gfx950_v_mov_b32_v1_v0(program);

  WaitcheckOptions options;
  options.check_counter_parity = true;
  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA4, options);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.passed()) << diagnostic_summary(report);
  ASSERT_EQ(report.counter_underaccounting_diagnostics.size(), 1u);
  const auto &diag = report.counter_underaccounting_diagnostics.front();
  EXPECT_EQ(diag.counter, WaitCounterKind::Load);
  EXPECT_EQ(diag.emitted_count, 0u);
  EXPECT_EQ(diag.required_count, 1u);
  EXPECT_TRUE(diag.has_required_dependency);
  EXPECT_EQ(diag.access, WaitcheckAccessKind::Use);
  EXPECT_EQ(diag.reg, (RegisterRef{RegClass::VGPR, 0, 1}));
  EXPECT_EQ(diag.wait_section_offset, 16u);
  EXPECT_EQ(diag.consumer_section_offset, 20u);
  EXPECT_EQ(diag.producer_section_offset, 0u);
}

TEST(WaitcheckTest, CdnaLegacyBufferWbl2CompletionWaitNeedsVmcntZero) {
  for (const auto arch : {ROCJITSU_CODE_ARCH_CDNA3, ROCJITSU_CODE_ARCH_CDNA4}) {
    SCOPED_TRACE(arch);
    // From the gfx942/gfx950 release sequence in PyTorch's c10d barrier:
    // global_load_dwordx2, BUFFER_WBL2, vmcnt(0), global_atomic_add_x2.
    std::vector<uint32_t> program{0xDC548010u, 0x000C0000u, 0xE0A0C000u, 0u,
                                  0xBF8C0F70u, 0xDF888008u, 0x007F0200u, 0xBF810000u};
    WaitcheckOptions options;
    options.check_counter_parity = true;
    const auto exact = analyze_waitcnts(program, arch, options);
    ASSERT_TRUE(exact.supported && exact.complete) << exact.analysis_error;
    EXPECT_TRUE(exact.diagnostics.empty()) << diagnostic_summary(exact);
    EXPECT_EQ(exact.counter_parity_fields_checked, 1u);
    EXPECT_EQ(exact.counter_parity_exact, 1u);
    EXPECT_TRUE(exact.counter_underaccounting_diagnostics.empty()) << diagnostic_summary(exact);

    // With only the writeback outstanding, vmcnt(1) is too weak to retire
    // its ACK and must not receive the zero-wait completion attribution.
    const std::vector<uint32_t> weak_program{0xE0A0C000u, 0u,          0xBF8C0F71u,
                                             0xDF888008u, 0x007F0200u, 0xBF810000u};
    const auto weak = analyze_waitcnts(weak_program, arch, options);
    ASSERT_TRUE(weak.supported && weak.complete) << weak.analysis_error;
    EXPECT_EQ(weak.counter_parity_exact, 0u) << diagnostic_summary(weak);
    ASSERT_EQ(weak.counter_underaccounting_diagnostics.size(), 1u);
    EXPECT_EQ(weak.counter_underaccounting_diagnostics.front().emitted_count, 1u);
  }
}

TEST(WaitcheckTest, Gfx950CounterParityUsesNoWaitSentinelWhenNoDependencyIsModeled) {
  std::vector<uint32_t> program;
  append_gfx950_buffer_load_dword_v0_v8_s0_offen(program);
  append_gfx950_s_waitcnt_vmcnt_0(program);
  append_gfx950_v_mov_b32_v8_v10(program);

  WaitcheckOptions options;
  options.check_counter_parity = true;
  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA4, options);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.passed()) << diagnostic_summary(report);
  ASSERT_EQ(report.counter_underaccounting_diagnostics.size(), 1u);
  const auto &diag = report.counter_underaccounting_diagnostics.front();
  EXPECT_NE(diag.message.find("no attributable modeled dependency"), std::string::npos)
      << diag.message;
  EXPECT_EQ(diag.counter, WaitCounterKind::Load);
  EXPECT_EQ(diag.emitted_count, 0u);
  EXPECT_EQ(diag.required_count, 63u);
  EXPECT_FALSE(diag.has_required_dependency);
}

TEST(WaitcheckTest, Gfx950CounterParityCombinesConsecutiveWaits) {
  std::vector<uint32_t> program;
  append_gfx950_buffer_load_dword_v0_v8_s0_offen(program);
  append_gfx950_buffer_load_dwordx4_v0_s64_offen_lds(program);
  append_gfx950_s_waitcnt_vmcnt_1(program);
  append_gfx950_s_waitcnt_vmcnt_0(program);
  append_gfx950_v_mov_b32_v1_v0(program);

  WaitcheckOptions options;
  options.check_counter_parity = true;
  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA4, options);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.passed()) << diagnostic_summary(report);
  ASSERT_EQ(report.counter_underaccounting_diagnostics.size(), 1u);
  const auto &diag = report.counter_underaccounting_diagnostics.front();
  EXPECT_EQ(diag.emitted_count, 0u);
  EXPECT_EQ(diag.required_count, 1u);
  EXPECT_NE(diag.wait_instruction.find(';'), std::string::npos);
}

TEST(WaitcheckTest, Gfx950CounterParityPreservesKernelIdentityInCodeObject) {
  std::vector<uint32_t> program;
  append_gfx950_buffer_load_dword_v0_v8_s0_offen(program);
  append_gfx950_buffer_load_dwordx4_v0_s64_offen_lds(program);
  append_gfx950_s_waitcnt_vmcnt_0(program);
  append_gfx950_v_mov_b32_v1_v0(program);

  const auto image =
      rocjitsu::waitcheck_test::make_gfx_code_object(program, EF_AMDGPU_MACH_AMDGCN_GFX950);
  AmdGpuCodeObject code_object(image.data(), image.size());
  WaitcheckOptions options;
  options.check_counter_parity = true;
  auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_CDNA4, options);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.passed()) << diagnostic_summary(report);
  ASSERT_EQ(report.counter_underaccounting_diagnostics.size(), 1u);
  const auto &diag = report.counter_underaccounting_diagnostics.front();
  EXPECT_TRUE(diag.has_kernel);
  EXPECT_EQ(diag.kernel_name, "waitcheck");
  EXPECT_EQ(diag.kernel_entry_offset, 0u);
  EXPECT_EQ(diag.emitted_count, 0u);
  EXPECT_EQ(diag.required_count, 1u);
}

TEST(WaitcheckTest, Gfx950CounterParityReportsStrongerEmittedLgkmcnt) {
  std::vector<uint32_t> program;
  append_gfx950_ds_read_b32_v0_v4(program);
  append_gfx950_ds_read_b128_v82_v13_offset64(program);
  append_gfx950_s_waitcnt_lgkmcnt_0(program);
  append_gfx950_v_mov_b32_v1_v0(program);

  WaitcheckOptions options;
  options.check_counter_parity = true;
  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA4, options);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.passed()) << diagnostic_summary(report);
  ASSERT_EQ(report.counter_underaccounting_diagnostics.size(), 1u);
  const auto &diag = report.counter_underaccounting_diagnostics.front();
  EXPECT_EQ(diag.counter, WaitCounterKind::Ds);
  EXPECT_EQ(diag.emitted_count, 0u);
  EXPECT_EQ(diag.required_count, 1u);
  EXPECT_TRUE(diag.has_required_dependency);
}

TEST(WaitcheckTest, Gfx950CounterParityDecodesExpcntField) {
  std::vector<uint32_t> program;
  append_gfx950_s_waitcnt_expcnt_0(program);
  append_gfx950_v_mov_b32_v8_v10(program);

  WaitcheckOptions options;
  options.check_counter_parity = true;
  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA4, options);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.passed()) << diagnostic_summary(report);
  ASSERT_EQ(report.counter_underaccounting_diagnostics.size(), 1u);
  const auto &diag = report.counter_underaccounting_diagnostics.front();
  EXPECT_EQ(diag.counter, WaitCounterKind::Exp);
  EXPECT_EQ(diag.emitted_count, 0u);
  EXPECT_EQ(diag.required_count, 7u);
  EXPECT_FALSE(diag.has_required_dependency);
}

TEST(WaitcheckTest, Gfx1100FlatPrivateApertureSeparatesResultAndIssueCounter) {
  const auto check = [](std::vector<uint32_t> setup, bool wait_vm, bool expect_ds) {
    // Every path ends in FLAT v2 from v[6:7], optionally VM zero, then reads v2.
    setup.insert(setup.end(), {0xdc500000u, 0x027c0006u});
    if (wait_vm)
      setup.push_back(0xbf8903f7u);
    setup.insert(setup.end(), {0x7e080302u, 0xbfb00000u});
    const auto image =
        waitcheck_test::make_gfx_code_object(setup, EF_AMDGPU_MACH_AMDGCN_GFX1100, /*wave32=*/true);
    AmdGpuCodeObject object(image.data(), image.size());
    const auto report = analyze_waitcnts_for_kernel(object, ROCJITSU_CODE_ARCH_RDNA3, 0);
    ASSERT_TRUE(report.supported && report.complete) << report.analysis_error;
    EXPECT_EQ(report.diagnostics_observed, unsigned(!wait_vm) + unsigned(expect_ds))
        << diagnostic_summary(report);
    if (!report.diagnostics.empty())
      EXPECT_EQ(report.diagnostics.front().counter,
                expect_ds ? WaitCounterKind::Ds : WaitCounterKind::Load);
  };
  constexpr uint32_t private_base = 0xbe8c01edu;    // s_mov_b64 s[12:13], src_private_base
  constexpr uint32_t shared_base = 0xbe8c01ebu;     // s_mov_b64 s[12:13], src_shared_base
  constexpr uint32_t move_high = 0x7e0e020du;       // v_mov_b32_e32 v7, s13
  constexpr uint32_t save_high = 0xbe8f000du;       // s_mov_b32 s15, s13
  constexpr uint32_t save_pair = 0xbe8e010cu;       // s_mov_b64 s[14:15], s[12:13]
  constexpr uint32_t copy_saved_high = 0x7e0e020fu; // v_mov_b32 v7, s15
  constexpr uint32_t clobber = 0xbe8f0008u;         // s_mov_b32 s15, s8
  constexpr uint32_t nop = 0xbf800000u;
  constexpr uint32_t branch_one = 0xbfa00001u;
  constexpr uint32_t branch_if_scc0_two = 0xbfa10002u;
  // The reaching scalar value can span branches and survive EXEC changes.
  for (auto save : {save_high, save_pair}) {
    check({private_base, save, copy_saved_high}, true, false);
    check({shared_base, save, copy_saved_high}, true, true);
    check({private_base, save, clobber, copy_saved_high}, true, true);
    std::vector<uint32_t> long_lived{private_base, save, 0xbe88007eu, 0xbefe0081u, 0xbefe0008u};
    long_lived.insert(long_lived.end(), 80, nop);
    long_lived.push_back(copy_saved_high);
    check(long_lived, true, false);
    check(long_lived, false, false);
  }
  check({private_base, save_high, branch_if_scc0_two, save_pair, branch_one, save_high,
         copy_saved_high},
        true, false);
  check({private_base, save_high, branch_if_scc0_two, shared_base, save_high, copy_saved_high},
        true, true);
  check({private_base, save_high, branch_if_scc0_two, clobber, nop, copy_saved_high}, true, true);
  check({private_base, save_high, branch_one, nop, 0xbe88007eu, 0xbefe0081u, copy_saved_high,
         0xbefe0008u},
        true, true); // Restoring EXEC can expose lanes that missed the vector write.
  check({private_base, move_high}, true, false);
  check({private_base, move_high}, false, false);
  check({shared_base, move_high}, true, true);
  check({private_base, shared_base, move_high}, true, true);
  check({private_base, 0xca100080u, 0x0806000du}, true, false); // MOV in VOPD Y
  check({private_base, 0xca10000du, 0x07080080u}, true, false); // MOV in VOPD X

  // The mask skips the private write for a lane that subsequently executes FLAT.
  check({shared_base, move_high, 0xbefe0080u, private_base, move_high, 0xbefe0081u}, true, true);
  // One incoming branch goes directly to FLAT with the original LDS tag.
  check({shared_base, move_high, 0xbf068009u, 0xbfa20002u, private_base, move_high}, true, true);
}

TEST(WaitcheckTest, Gfx11FlatResultIsWrittenOnlyByItsProvenAperture) {
  constexpr uint32_t private_base = 0xbe8c01ed, shared_base = 0xbe8c01eb;
  constexpr uint32_t vm0 = 0xbf8903f7, ds0 = 0xbf89fc07, both0 = 0xbf890007;
  for (const bool gfx115 : {false, true}) {
    const auto arch = gfx115 ? ROCJITSU_CODE_ARCH_RDNA3_5 : ROCJITSU_CODE_ARCH_RDNA3;
    for (const auto selector : {private_base, shared_base}) {
      const auto counter = selector == private_base ? WaitCounterKind::Load : WaitCounterKind::Ds;
      for (const auto wait : {0u, vm0, ds0, both0}) {
        std::vector<uint32_t> words{selector, 0x7e0e020du, 0xdc500000u, 0x027c0006u};
        if (wait)
          words.push_back(wait);
        words.insert(words.end(), {0x7e080302u, 0xbfb00000u});
        const auto mach = gfx115 ? EF_AMDGPU_MACH_AMDGCN_GFX1150 : EF_AMDGPU_MACH_AMDGCN_GFX1100;
        const auto image = waitcheck_test::make_gfx_code_object(words, mach, true);
        AmdGpuCodeObject object(image.data(), image.size());
        const auto report = analyze_waitcnts_for_kernel(object, arch, 0);
        ASSERT_TRUE(report.supported && report.complete) << report.analysis_error;
        const bool unsafe = wait != both0 && wait != (selector == private_base ? vm0 : ds0);
        ASSERT_EQ(report.diagnostics_observed, unsigned(unsafe)) << diagnostic_summary(report);
        if (unsafe)
          EXPECT_EQ(report.diagnostics.front().counter, counter);
      }
    }
  }
}

TEST(WaitcheckTest, Gfx950CounterParityFollowsUnconditionalBranchToGuardedUse) {
  std::vector<uint32_t> program;
  append_gfx950_s_load_dword_s4_s0(program);
  append_gfx950_s_waitcnt_lgkmcnt_0(program);
  program.push_back(0xBF820001u); // s_branch over the dead instruction.
  program.push_back(0xBF800000u); // s_nop 0.
  append_gfx950_s_mov_b32_s8_s4(program);
  program.push_back(0xBF810000u); // s_endpgm.

  const auto image =
      rocjitsu::waitcheck_test::make_gfx_code_object(program, EF_AMDGPU_MACH_AMDGCN_GFX950);
  AmdGpuCodeObject code_object(image.data(), image.size());
  WaitcheckOptions options;
  options.check_counter_parity = true;
  const auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_CDNA4, options);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.passed()) << diagnostic_summary(report);
  EXPECT_EQ(report.counter_parity_fields_checked, 1u);
  EXPECT_EQ(report.counter_parity_exact, 1u);
  EXPECT_EQ(report.counter_underaccounting_observed, 0u);
}

TEST(WaitcheckTest, Gfx950CounterParityContinuesAcrossFallthroughBlockLabel) {
  std::vector<uint32_t> program;
  program.push_back(0xBF840003u); // s_cbranch_scc0 to the guarded use.
  append_gfx950_s_load_dword_s4_s0(program);
  append_gfx950_s_waitcnt_lgkmcnt_0(program);
  // This instruction starts a basic block because it is also the branch
  // target, while the wait reaches it through the fallthrough predecessor.
  append_gfx950_s_mov_b32_s8_s4(program);
  program.push_back(0xBF810000u); // s_endpgm.

  const auto image =
      rocjitsu::waitcheck_test::make_gfx_code_object(program, EF_AMDGPU_MACH_AMDGCN_GFX950);
  AmdGpuCodeObject code_object(image.data(), image.size());
  WaitcheckOptions options;
  options.check_counter_parity = true;
  const auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_CDNA4, options);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.passed()) << diagnostic_summary(report);
  EXPECT_EQ(report.counter_parity_wait_groups, 1u);
  EXPECT_EQ(report.counter_parity_fields_checked, 1u);
  EXPECT_EQ(report.counter_parity_exact, 1u);
  EXPECT_EQ(report.counter_underaccounting_observed, 0u);
  EXPECT_EQ(report.counter_parity_indeterminate_groups, 0u);
}

TEST(WaitcheckTest, Gfx950CounterParityFindsGuardedUseAfterIndependentInstruction) {
  std::vector<uint32_t> program;
  append_gfx950_ds_read_b32_v0_v4(program);
  program.push_back(0xD86C0000u); // ds_read_b32 v2, v4.
  program.push_back(0x02000004u);
  append_gfx950_s_waitcnt_lgkmcnt_0(program);
  append_gfx950_v_mov_b32_v8_v10(program);
  append_gfx950_v_mov_b32_v0_v2(program);
  program.push_back(0xBF810000u); // s_endpgm.

  const auto image =
      rocjitsu::waitcheck_test::make_gfx_code_object(program, EF_AMDGPU_MACH_AMDGCN_GFX950);
  AmdGpuCodeObject code_object(image.data(), image.size());
  WaitcheckOptions options;
  options.check_counter_parity = true;
  const auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_CDNA4, options);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.passed()) << diagnostic_summary(report);
  EXPECT_EQ(report.counter_parity_fields_checked, 1u);
  EXPECT_EQ(report.counter_parity_exact, 1u);
  EXPECT_EQ(report.counter_underaccounting_observed, 0u);
}

TEST(WaitcheckTest, Gfx950CounterParityCollectsAllUsesBeforeNextWait) {
  std::vector<uint32_t> program;
  append_gfx950_ds_read_b32_v0_v4(program);
  program.push_back(0xD86C0000u); // ds_read_b32 v2, v4.
  program.push_back(0x02000004u);
  append_gfx950_s_waitcnt_lgkmcnt_0(program);
  append_gfx950_v_mov_b32_v1_v0(program);
  program.push_back(0xBF8A0000u); // s_barrier consumes the younger DS event.
  program.push_back(0xBF810000u); // s_endpgm.

  const auto image =
      rocjitsu::waitcheck_test::make_gfx_code_object(program, EF_AMDGPU_MACH_AMDGCN_GFX950);
  AmdGpuCodeObject code_object(image.data(), image.size());
  WaitcheckOptions options;
  options.check_counter_parity = true;
  const auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_CDNA4, options);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.passed()) << diagnostic_summary(report);
  EXPECT_EQ(report.counter_parity_fields_checked, 1u);
  EXPECT_EQ(report.counter_parity_exact, 1u);
  EXPECT_EQ(report.counter_underaccounting_observed, 0u);
}

TEST(WaitcheckTest, Gfx950CounterParityAttributesLegacyLgkmDrainToBarrier) {
  std::vector<uint32_t> program;
  append_gfx950_s_load_dword_s4_s0(program);
  append_gfx950_s_waitcnt_lgkmcnt_0(program);
  program.push_back(0xBF8A0000u); // s_barrier.
  program.push_back(0xBF810000u); // s_endpgm.

  const auto image =
      rocjitsu::waitcheck_test::make_gfx_code_object(program, EF_AMDGPU_MACH_AMDGCN_GFX950);
  AmdGpuCodeObject code_object(image.data(), image.size());
  WaitcheckOptions options;
  options.check_counter_parity = true;
  const auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_CDNA4, options);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.passed()) << diagnostic_summary(report);
  EXPECT_EQ(report.counter_parity_fields_checked, 1u);
  EXPECT_EQ(report.counter_parity_exact, 1u);
  EXPECT_EQ(report.counter_underaccounting_observed, 0u);
}

TEST(WaitcheckTest, Gfx950DoesNotRequireBarrierWaitForRegisterOnlyDsLoad) {
  std::vector<uint32_t> program;
  append_gfx950_ds_read_b32_v0_v4(program);
  program.push_back(0xBF8A0000u); // s_barrier does not consume the VGPR result.
  program.push_back(0xBF810000u); // s_endpgm.

  const auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA4);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx950CounterParityTracksPartialDirectToLdsWaitBeforeBarrier) {
  std::vector<uint32_t> program;
  append_gfx950_buffer_load_dwordx4_v0_s64_offen_lds(program);
  append_gfx950_buffer_load_dword_v0_v8_s0_offen(program);
  append_gfx950_s_waitcnt_vmcnt_1(program);
  program.push_back(0xBF8A0000u); // s_barrier.
  program.push_back(0xBF810000u); // s_endpgm.

  const auto image =
      rocjitsu::waitcheck_test::make_gfx_code_object(program, EF_AMDGPU_MACH_AMDGCN_GFX950);
  AmdGpuCodeObject code_object(image.data(), image.size());
  WaitcheckOptions options;
  options.check_counter_parity = true;
  const auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_CDNA4, options);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.passed()) << diagnostic_summary(report);
  EXPECT_EQ(report.counter_parity_fields_checked, 1u);
  EXPECT_EQ(report.counter_parity_exact, 1u);
  EXPECT_EQ(report.counter_underaccounting_observed, 0u);
}

TEST(WaitcheckTest, Gfx950CounterParityForcesZeroWithPendingFlatEvent) {
  std::vector<uint32_t> program;
  append_gfx950_ds_read_b32_v0_v4(program);
  program.push_back(0xDC500000u); // flat_load_dword v109, v[12:13].
  program.push_back(0x6D00000Cu);
  append_gfx950_s_waitcnt_lgkmcnt_0(program);
  append_gfx950_v_mov_b32_v1_v0(program);
  program.push_back(0xBF810000u); // s_endpgm.

  const auto image =
      rocjitsu::waitcheck_test::make_gfx_code_object(program, EF_AMDGPU_MACH_AMDGCN_GFX950);
  AmdGpuCodeObject code_object(image.data(), image.size());
  WaitcheckOptions options;
  options.check_counter_parity = true;
  const auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_CDNA4, options);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.passed()) << diagnostic_summary(report);
  EXPECT_EQ(report.counter_parity_fields_checked, 1u);
  EXPECT_EQ(report.counter_parity_exact, 1u);
  EXPECT_EQ(report.counter_underaccounting_observed, 0u);
}

TEST(WaitcheckTest, Gfx942CounterParityMatchesLegacyVmcnt) {
  std::vector<uint32_t> program;
  append_gfx942_buffer_load_dword(program, 0, 8);
  append_gfx942_buffer_load_dword(program, 2, 8);
  program.push_back(0xBF8C0F71u); // s_waitcnt vmcnt(1).
  append_gfx942_v_mov_b32_v1_v0(program);

  WaitcheckOptions options;
  options.check_counter_parity = true;
  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA3, options);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.passed()) << diagnostic_summary(report);
  EXPECT_EQ(report.counter_parity_wait_groups, 1u);
  EXPECT_EQ(report.counter_parity_fields_checked, 1u);
  EXPECT_EQ(report.counter_parity_exact, 1u);
  EXPECT_EQ(report.counter_underaccounting_observed, 0u);
}

TEST(WaitcheckTest, Rdna3TransUseNeedsZeroVaVdstWithinBothIssueWindows) {
  constexpr uint32_t log_v3 = 0x7e064f04;      // v_log_f32 v3, v4
  constexpr uint32_t read_v3 = 0x7e140303;     // v_mov_b32 v10, v3
  constexpr uint32_t other_valu = 0x7e14030b;  // v_mov_b32 v10, v11
  constexpr uint32_t other_trans = 0x7e0a5506; // v_rcp_f32 v5, v6
  constexpr uint32_t wait0 = 0xbf880f9f;       // s_waitcnt_depctr depctr_va_vdst(0)
  constexpr uint32_t wait1 = 0xbf881f9f;       // s_waitcnt_depctr depctr_va_vdst(1)
  struct Probe {
    std::vector<uint32_t> words;
    bool missing;
    unsigned exact;
  };
  const Probe probes[] = {
      {{log_v3, read_v3}, true, 0},
      {{log_v3, wait0, read_v3}, false, 1},
      {{log_v3, wait0, other_valu, read_v3}, false, 1},
      {{log_v3, wait0, other_valu, wait1, read_v3}, false, 1},
      {{log_v3, wait0, other_valu, other_valu, other_valu, other_valu, other_valu, other_valu,
        read_v3},
       false,
       0},
      {{log_v3, wait0, 0xbfa00001, other_valu, read_v3}, false, 1}, // direct branch to use
      {{0xbfa10001, log_v3, read_v3}, true, 0}, // one incoming path issues TRANS
      {{log_v3, wait1, read_v3}, true, 0},
      {{log_v3, other_valu, other_valu, other_valu, other_valu, other_valu, read_v3}, true, 0},
      {{log_v3, other_valu, other_valu, other_valu, other_valu, other_valu, other_valu, read_v3},
       false,
       0},
      {{log_v3, other_trans, read_v3}, true, 0},
      {{log_v3, other_trans, other_trans, read_v3}, false, 0},
      {{log_v3, other_valu}, false, 0},
      {{log_v3, 0xd8500000, 0, read_v3}, false, 0}, // ds_nop drains VA_VDST
      {{0x7e085f04, 0x7e020305}, true, 0},          // v_rcp_f64 v[4:5]; use its second VGPR
      {{log_v3, 0xbf800000, read_v3}, true, 0},     // scalar instructions do not age the hazard
  };
  for (const auto &probe : probes) {
    for (uint32_t wave : {32u, 64u}) {
      std::vector<uint32_t> words = probe.words;
      words.push_back(0xbfb00000); // s_endpgm
      WaitcheckKernelInfo kernel;
      kernel.code_size = words.size() * sizeof(uint32_t);
      kernel.wavefront_size = wave;
      WaitcheckOptions options;
      options.check_counter_parity = true;
      const auto report = analyze_waitcnts_for_kernel(TestCodeObject(words),
                                                      ROCJITSU_CODE_ARCH_RDNA3, kernel, options);
      ASSERT_TRUE(report.supported) << report.analysis_error;
      EXPECT_EQ(report.diagnostics.size(), probe.missing ? 1u : 0u) << diagnostic_summary(report);
      EXPECT_EQ(report.counter_parity_exact, probe.exact);
      if (!report.diagnostics.empty()) {
        EXPECT_EQ(report.diagnostics.front().counter, WaitCounterKind::VaVdst);
        EXPECT_EQ(report.diagnostics.front().required_count, 0u);
        EXPECT_EQ(report.diagnostics.front().access, WaitcheckAccessKind::Use);
      }
    }
  }
  for (auto arch : {ROCJITSU_CODE_ARCH_RDNA3_5, ROCJITSU_CODE_ARCH_RDNA4}) {
    const std::vector<uint32_t> words{log_v3, read_v3, 0xbfb00000};
    const auto report = analyze_waitcnts(TestCodeObject(words), arch);
    ASSERT_TRUE(report.supported) << report.analysis_error;
    EXPECT_TRUE(report.passed()) << diagnostic_summary(report);
  }
}

TEST(WaitcheckTest, Wave64PartialForwardingNeedsTwoSourcesAcrossScalarExecOnOnePath) {
  constexpr uint32_t write_a = 0x7e060280; // v_mov_b32 v3, 0
  constexpr uint32_t write_b = 0x7e0a0280; // v_mov_b32 v5, 0
  constexpr uint32_t other = 0x7e140280;   // v_mov_b32 v10, 0
  constexpr uint32_t mask = 0xbe802302;    // s_or_saveexec_b64 s[0:1], s[2:3]
  constexpr uint32_t use = 0x06180b03;     // v_add_f32 v12, v3, v5
  constexpr uint32_t wait0 = 0xbf880f9f;   // s_waitcnt_depctr depctr_va_vdst(0)
  constexpr uint32_t wait1 = 0xbf881f9f;   // s_waitcnt_depctr depctr_va_vdst(1)
  struct Probe {
    std::vector<uint32_t> words;
    bool hazard;
    bool exact = false;
  };
  const Probe probes[] = {
      {{0x7e063880, mask, 0x7f063880, 0x64190703}, true}, // distinct halves of v3
      {{0x7e063880, mask, 0x7f063880, wait0, 0x64190703}, false, true},
      {{0x7e063880, mask, 0x7f0a3880, 0x64190b83}, false}, // older writer misses v3.hi
      {{0x7f063880, mask, 0x7f0a3880, 0x64190b83}, true},
      {{0x7e063880, mask, 0x7e063880, 0x64190703}, false}, // only v3.lo was written
      {{0x7e063880, mask, 0xd59c4003, 0x02010080, 0xd532100c, 0x02020703}, true},
      {{write_a, mask, write_b, wait0, other, use}, false, true},
      {{write_a, mask, write_b, wait0, other, other, other, other, use}, false, true},
      {{write_a, mask, write_b, wait0, other, other, other, other, other, use}, false},
      {{write_a, mask, write_b, wait0, write_a, use}, false},
      {{write_a, mask, write_b, wait0, other, 0xd8500000, 0, use}, false},
      {{write_a, mask, write_b, wait0, other, 0xce100008, use}, false},
      {{write_a, mask, write_b, wait0, other, 0xbfa10001, write_a, use}, false, true},
      {{0xbfa10002, write_a, 0xbfa00002, mask, write_b, wait0, other, use}, false},
      {{write_a, mask, write_b, use}, true},
      {{write_a, mask, write_b, wait0, use}, false, true},
      {{write_a, mask, write_b, wait1, use}, true},
      {{write_a, other, mask, other, write_b, other, other, other, other, use}, true},
      {{write_a, other, mask, other, other, write_b, use}, false},
      {{write_a, mask, write_b, other, other, other, other, other, use}, false},
      {{write_a, write_b, use}, false},
      {{write_a, mask, write_b, write_a, use}, false},          // both latest writers follow EXEC
      {{write_a, mask, write_b, 0x06180703}, false},            // v3 appears twice
      {{write_a, 0x7d940903, write_b, use}, false},             // VALU CMPX modifies EXEC
      {{write_a, mask, write_b, 0xd8500000, 0, use}, false},    // ds_nop
      {{write_a, mask, write_b, 0xbfa00001, other, use}, true}, // branch goes directly to use
      // Either arm reaches the use; only one arm contains both the mask
      // change and B. A is on the other, mutually exclusive arm.
      {{0xbfa10002, write_a, 0xbfa00002, mask, write_b, use}, false},
  };
  for (const auto &probe : probes) {
    for (auto arch : {ROCJITSU_CODE_ARCH_RDNA3, ROCJITSU_CODE_ARCH_RDNA3_5}) {
      for (uint32_t wave : {32u, 64u}) {
        std::vector<uint32_t> words = probe.words;
        words.push_back(0xbfb00000); // s_endpgm
        WaitcheckKernelInfo kernel;
        kernel.code_size = words.size() * sizeof(uint32_t);
        kernel.wavefront_size = wave;
        WaitcheckOptions options;
        options.check_counter_parity = true;
        const auto report =
            analyze_waitcnts_for_kernel(TestCodeObject(words), arch, kernel, options);
        ASSERT_TRUE(report.supported) << report.analysis_error;
        EXPECT_EQ(report.diagnostics.size(), wave == 64 && probe.hazard ? 1u : 0u)
            << diagnostic_summary(report);
        EXPECT_EQ(report.counter_parity_exact, wave == 64 && probe.exact ? 1u : 0u);
      }
    }
  }
}

TEST(WaitcheckTest, Rdna3TransWaitUsesPhysicalM0AndTrue16Sources) {
  constexpr uint32_t trans = 0x7e064f04;    // v_log_f32 v3, v4
  constexpr uint32_t wait = 0xbf880f9f;     // s_waitcnt_depctr depctr_va_vdst(0)
  constexpr uint32_t other = 0x7e14030b;    // v_mov_b32 v10, v11
  constexpr uint32_t use = 0x7e140303;      // v_mov_b32 v10, v3
  constexpr uint32_t m0_one = 0xbefd0081;   // s_mov_b32 m0, 1
  constexpr uint32_t lds_zero = 0xce100008; // lds_direct_load v8 wait_vdst:0
  struct Probe {
    std::vector<uint32_t> words;
    unsigned missing;
    unsigned exact;
    std::vector<uint64_t> unmodeled;
  };
  const Probe probes[] = {
      {{trans, wait, other, lds_zero, use}, 0, 0, {4, 12}},
      {{trans, other, lds_zero, use}, 0, 1, {}},
      {{trans, other, use}, 1, 0, {}},
      {{m0_one, trans, wait, other, 0x7e148703}, 0, 0, {8}}, // MOVRELS base v3 reads v4
      {{m0_one, trans, wait, other, 0x7e148702}, 0, 1, {}},  // MOVRELS base v2 reads v3
      {{m0_one, trans, other, 0x7e148703}, 0, 0, {}},
      {{m0_one, trans, other, 0x7e148702}, 1, 0, {}},
      {{0x7e06a904, 0x7f06a905}, 0, 0, {}},             // rcp v3.lo; rcp v3.hi reads only v5.lo
      {{0x7e06a904, 0x7e06a905}, 0, 0, {}},             // rcp v3.lo; rcp v3.lo reads only v5.lo
      {{0x7e065504, 0x7e0638ff, 0x7fc0}, 0, 0, {}},     // mov literal into previous RCP's VDST
      {{0x7e065504, 0xd7390003, 0x02020a82}, 0, 0, {}}, // shift reads v5.lo, writes v3.lo
      {{0x7e06a904, 0x6c060b04}, 1, 0, {}},             // fmac reads its tied v3.lo accumulator
      {{0x7e06a904, 0x6d060b04}, 0, 0, {}},             // fmac reads the opposite v3.hi accumulator
      {{0x7e065504, 0x7e0638fa, 0x3f00e404}, 1, 0, {}}, // masked DPP may read original v3
      {{0x7f06a904, 0xd5364003, 0x02020b04}, 1, 0, {}}, // e64 high FMAC reads high VDST
      {{0x7e06a904, 0xd5364003, 0x02020b04}, 0, 0, {}}, // e64 high FMAC ignores low VDST
      {{0x7f06a904, wait, other, 0xd5364003, 0x02020b04}, 0, 1, {}},
      {{0x7e06a904, wait, other, 0xd5364003, 0x02020b04}, 0, 0, {4}},
      {{0x7e065504, wait, other, 0x7e0638ff, 0x7fc0}, 0, 0, {4}},
      {{0x7e06a904, wait, other, 0x6c060b04}, 0, 1, {}},
      {{0x7e00b100, 0x6b010180}, 0, 0, {}}, // exp v0.lo; mul v0.hi, v0.hi, v0.hi
      {{0x7e06af04, 0x6b150703}, 1, 0, {}}, // log v3.lo; mul reads both v3 halves
      {{0x7f06af04, 0x7f143983}, 1, 0, {}}, // log v3.hi; mov reads v3.hi
      {{0x7f06af04, 0x7e140303}, 1, 0, {}}, // a full dword read includes both halves
      {{0xd5d74003, 0x02010104, wait, other, 0x7e143903}, 0, 0, {8}}, // e64 log v3.hi; use v3.lo
      {{0xd5d74003, 0x02010104, wait, other, 0x7f143983}, 0, 1, {}},  // e64 log v3.hi; use v3.hi
  };
  for (const auto &probe : probes) {
    auto words = probe.words;
    words.push_back(0xbfb00000); // s_endpgm
    WaitcheckOptions options;
    options.check_counter_parity = true;
    const auto report = analyze_waitcnts(TestCodeObject(words), ROCJITSU_CODE_ARCH_RDNA3, options);
    ASSERT_TRUE(report.supported) << report.analysis_error;
    EXPECT_EQ(report.diagnostics.size(), probe.missing) << diagnostic_summary(report);
    EXPECT_EQ(report.counter_parity_exact, probe.exact);
    std::vector<uint64_t> unmodeled;
    for (const auto &row : report.counter_underaccounting_diagnostics)
      if (row.counter == WaitCounterKind::VaVdst && !row.has_required_dependency)
        unmodeled.push_back(row.wait_section_offset);
    EXPECT_EQ(unmodeled, probe.unmodeled);
  }
}

TEST(WaitcheckTest, Gfx1100CounterParityMatchesSopkVmcnt) {
  std::vector<uint32_t> program;
  append_gfx1100_global_load_b32_v0_v8_s0(program);
  append_gfx1100_global_load_b32_v2_v8_s0(program);
  append_gfx1100_s_waitcnt_vmcnt_sopk_1(program);
  append_gfx1100_v_mov_b32_v1_v0(program);

  WaitcheckOptions options;
  options.check_counter_parity = true;
  TestCodeObject code_object(program);
  auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_RDNA3, options);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.passed()) << diagnostic_summary(report);
  EXPECT_EQ(report.counter_parity_wait_groups, 1u);
  EXPECT_EQ(report.counter_parity_fields_checked, 1u);
  EXPECT_EQ(report.counter_parity_exact, 1u);
  EXPECT_EQ(report.counter_underaccounting_observed, 0u);
}

TEST(WaitcheckTest, Gfx1100CounterParityLooksPastAdjacentDepctrWait) {
  std::vector<uint32_t> program;
  append_gfx1100_global_load_b32_v0_v8_s0(program);
  append_gfx1100_s_waitcnt_vmcnt_0(program);
  program.push_back(0xBF880000u); // s_waitcnt_depctr 0.
  append_gfx1100_v_mov_b32_v1_v0(program);

  WaitcheckOptions options;
  options.check_counter_parity = true;
  TestCodeObject code_object(program);
  auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_RDNA3, options);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.passed()) << diagnostic_summary(report);
  EXPECT_EQ(report.counter_parity_fields_checked, 3u);
  EXPECT_EQ(report.counter_parity_exact, 1u);
  EXPECT_EQ(report.counter_underaccounting_observed, 2u);
  EXPECT_EQ(report.counter_parity_indeterminate_groups, 0u);
}

TEST(WaitcheckTest, Gfx1100CounterParityNormalizesWaitIdle) {
  std::vector<uint32_t> program;
  append_gfx1100_global_load_b32_v0_v8_s0(program);
  program.push_back(0xBF8A0000u); // s_wait_idle.
  append_gfx1100_v_mov_b32_v1_v0(program);

  WaitcheckOptions options;
  options.check_counter_parity = true;
  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA3, options);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.passed()) << diagnostic_summary(report);
  EXPECT_EQ(report.counter_parity_fields_checked, 6u);
  EXPECT_EQ(report.counter_parity_exact, 1u);
  EXPECT_EQ(report.counter_underaccounting_observed, 5u);
  EXPECT_EQ(report.counter_unmodeled_wait_observed, 5u);
  EXPECT_EQ(report.counter_parity_indeterminate_groups, 0u);
}

TEST(WaitcheckTest, Gfx1100CounterParityDecodesVscnt) {
  std::vector<uint32_t> program;
  append_gfx1100_s_waitcnt_vscnt_0(program);
  append_gfx1100_v_mov_b32_v1_v0(program);

  WaitcheckOptions options;
  options.check_counter_parity = true;
  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA3, options);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.counter_underaccounting_diagnostics.size(), 1u);
  const auto &diag = report.counter_underaccounting_diagnostics.front();
  EXPECT_EQ(diag.counter, WaitCounterKind::Store);
  EXPECT_EQ(diag.emitted_count, 0u);
  EXPECT_EQ(diag.required_count, 63u);
  EXPECT_FALSE(diag.has_required_dependency);
}

TEST(WaitcheckTest, Gfx1100CounterParityCatalogsConservativePairedD16Wait) {
  // Final ISA from PyTorch 2.9.1 code object 139, entry 0, .text+0x4dc.
  // The packaged compiler emits lgkmcnt(0) before the paired high-half load;
  // current LLVM emits lgkmcnt(1) for the later v0 use, matching waitcheck.
  std::vector<uint32_t> program{
      0xDA880000u, 0x0000000Eu, // ds_load_u8_d16 v0, v14.
      0xBF89FC07u,              // s_waitcnt lgkmcnt(0).
      0xDA8C0002u, 0x0000000Eu, // ds_load_u8_d16_hi v0, v14 offset:2.
      0xDA880000u, 0x0100000Fu, // ds_load_u8_d16 v1, v15.
      0xBF89FC07u,              // s_waitcnt lgkmcnt(0).
      0xDA8C0002u, 0x0100000Fu, // ds_load_u8_d16_hi v1, v15 offset:2.
      0xD7620012u, 0x02020081u, // v_and_b16 v18.l, 1, v0.l.
      0xBF810000u,              // s_endpgm.
  };

  WaitcheckOptions options;
  options.check_counter_parity = true;
  TestCodeObject code_object(program);
  const auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_RDNA3, options);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.passed()) << diagnostic_summary(report);
  EXPECT_EQ(report.counter_parity_fields_checked, 2u);
  EXPECT_EQ(report.counter_parity_exact, 0u);
  EXPECT_EQ(report.counter_unmodeled_wait_observed, 1u);
  ASSERT_EQ(report.counter_underaccounting_diagnostics.size(), 2u);
  const auto diag_it =
      std::ranges::find_if(report.counter_underaccounting_diagnostics,
                           [](const WaitcheckCounterUnderaccountingDiagnostic &diag) {
                             return diag.has_required_dependency;
                           });
  ASSERT_NE(diag_it, report.counter_underaccounting_diagnostics.end());
  const auto &diag = *diag_it;
  EXPECT_EQ(diag.counter, WaitCounterKind::Ds);
  EXPECT_EQ(diag.emitted_count, 0u);
  EXPECT_EQ(diag.required_count, 1u);
  EXPECT_EQ(diag.producer_instruction, "ds_load_u8_d16_hi v0, v14 offset:2");
  EXPECT_EQ(diag.consumer_instruction, "v_and_b16 v18.l, 1, v0.l");
}

TEST(WaitcheckTest, Gfx1250CounterParityMatchesSplitLoadcnt) {
  std::vector<uint32_t> program;
  append_inst(program, global_load_b32(0));
  append_inst(program, global_load_b32(2));
  append_inst(program, sopp(64, 1)); // s_wait_loadcnt 1.
  append_inst(program, v_mov_b32(1, 0));

  WaitcheckOptions options;
  options.check_counter_parity = true;
  TestCodeObject code_object(program);
  auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_CDNA5, options);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.passed()) << diagnostic_summary(report);
  EXPECT_EQ(report.counter_parity_wait_groups, 1u);
  EXPECT_EQ(report.counter_parity_fields_checked, 1u);
  EXPECT_EQ(report.counter_parity_exact, 1u);
  EXPECT_EQ(report.counter_underaccounting_observed, 0u);
}

TEST(WaitcheckTest, Gfx1250CounterParityNormalizesCombinedLoadAndDsWait) {
  std::vector<uint32_t> program;
  append_inst(program, flat_load_b32(0));
  append_inst(program, global_load_b32(2));
  append_inst(program, ds_load_b32(4, 12));
  append_inst(program, sopp(72, 0x0101)); // s_wait_loadcnt_dscnt 1, 1.
  append_inst(program, v_mov_b32(1, 0));

  WaitcheckOptions options;
  options.check_counter_parity = true;
  TestCodeObject code_object(program);
  auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_CDNA5, options);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.passed()) << diagnostic_summary(report);
  EXPECT_EQ(report.counter_parity_wait_groups, 1u);
  EXPECT_EQ(report.counter_parity_fields_checked, 2u);
  EXPECT_EQ(report.counter_parity_exact, 2u);
  EXPECT_EQ(report.counter_underaccounting_observed, 0u);
}

TEST(WaitcheckTest, Gfx11CalleeLgkmWaitUsesTheGuardedFlatIssuePath) {
  for (const uint32_t mach : {EF_AMDGPU_MACH_AMDGCN_GFX1100, EF_AMDGPU_MACH_AMDGCN_GFX1150}) {
    const auto arch = mach == EF_AMDGPU_MACH_AMDGCN_GFX1100 ? ROCJITSU_CODE_ARCH_RDNA3
                                                            : ROCJITSU_CODE_ARCH_RDNA3_5;
    for (const bool drained : {false, true}) {
      for (const unsigned count : {0u, 1u, 2u}) {
        SCOPED_TRACE(testing::Message() << "arch " << arch << " alternate drain " << drained
                                        << " helper LGKM " << count);
        // The v8 result can reach the helper after either a younger FLAT or
        // an alternate branch. A drain on the alternate branch means that
        // every live result has a younger request despite the joined age.
        const std::vector<uint32_t> words{
            0xbefe0000, // s_mov_b32 exec_lo, s0
            0xbfa50009, // s_cbranch_execz call_helper
            0xdc500000,
            0x087c0000,                          // flat_load_b32 v8, v[0:1]
            0xbf068003,                          // s_cmp_eq_u32 s3, 0
            0xbfa20002,                          // s_cbranch_scc1 extra
            drained ? 0xbf890007u : 0xbf800000u, // s_waitcnt vmcnt(0) lgkmcnt(0), or s_nop 0
            0xbfa00003,                          // s_branch call_helper
            0xbfa50002,                          // extra: s_cbranch_execz call_helper
            0xdc500000,
            0x097c0000, // flat_load_b32 v9, v[0:1]
            0xbe904700, // call_helper: s_getpc_b64 s[16:17]
            0x8010ff10,
            0x00000014,                 // s_add_u32 s16, s16, helper - (call_helper + 4)
            0x82118011,                 // s_addc_u32 s17, s17, 0
            0xbe9e4910,                 // s_swappc_b64 s[30:31], s[16:17]
            0xbfb00000,                 // s_endpgm
            0xbf890000u | (count << 4), // helper: s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(count)
            0xbefe00c1,                 // s_mov_b32 exec_lo, -1
            0x7e060308,                 // v_mov_b32 v3, v8
            0xbe80481e};                // s_setpc_b64 s[30:31]
        const auto image = waitcheck_test::make_gfx_code_object(words, mach, true);
        AmdGpuCodeObject object(image.data(), image.size());
        WaitcheckOptions options;
        options.check_counter_parity = true;
        const auto report = analyze_waitcnts(object, arch, options);
        ASSERT_TRUE(report.supported && report.complete) << report.analysis_error;
        const bool unsafe = count > (drained ? 1u : 0u);
        EXPECT_EQ(std::ranges::any_of(report.diagnostics,
                                      [](const auto &d) {
                                        return d.section_offset == 0x4c &&
                                               d.producer_section_offset == 8 &&
                                               d.counter == WaitCounterKind::Ds;
                                      }),
                  unsafe)
            << diagnostic_summary(report);
        if (count != 0)
          continue;
        const auto &fields = report.counter_underaccounting_diagnostics;
        const auto found = std::ranges::find_if(fields, [](const auto &d) {
          return d.wait_section_offset == 0x44 && d.counter == WaitCounterKind::Ds;
        });
        if (drained) {
          ASSERT_NE(found, fields.end());
          EXPECT_FALSE(found->has_required_dependency);
          EXPECT_NE(found->message.find("function-boundary policy"), std::string::npos);
        } else {
          EXPECT_EQ(found, fields.end());
        }
      }
    }
  }
}

TEST(WaitcheckTest, Gfx11CalleeLgkmFindsOtherGuardedFlatResults) {
  // With a nonempty EXEC both loads issue. Only v8 has a younger LGKM event.
  // The zero wait is necessary if v9 is used before another full drain.
  enum class Uses {
    Older,
    Younger,
    Both,
    BothAfterExec,
    PartialBetween,
    FullBetween,
    MaskedAfterPartial,
    OnlyEmptyPath,
    DisjointLanes,
    OverlappingLanes,
    Complement,
    ClobberedComplement,
    RestoredAfterPartial,
    PartialFirst,
    OlderAfterPartial
  };
  for (const uint32_t mach : {EF_AMDGPU_MACH_AMDGCN_GFX1100, EF_AMDGPU_MACH_AMDGCN_GFX1150}) {
    const auto arch = mach == EF_AMDGPU_MACH_AMDGCN_GFX1100 ? ROCJITSU_CODE_ARCH_RDNA3
                                                            : ROCJITSU_CODE_ARCH_RDNA3_5;
    for (auto uses :
         {Uses::Older, Uses::Younger, Uses::Both, Uses::BothAfterExec, Uses::PartialBetween,
          Uses::FullBetween, Uses::MaskedAfterPartial, Uses::OnlyEmptyPath, Uses::DisjointLanes,
          Uses::OverlappingLanes, Uses::Complement, Uses::ClobberedComplement,
          Uses::RestoredAfterPartial, Uses::PartialFirst, Uses::OlderAfterPartial}) {
      for (const unsigned count : {0u, 1u}) {
        SCOPED_TRACE(testing::Message() << "arch " << arch << " uses " << static_cast<int>(uses)
                                        << " count " << count);
        std::vector<uint32_t> words{
            0xbefe0000,
            0xbfa50005, // s_mov_b32 exec_lo, s0; s_cbranch_execz call_helper
            0xdc500000,
            0x087c0000, // flat_load_b32 v8, v[0:1]
            0xbfa50002, // s_cbranch_execz call_helper
            0xdc500000,
            0x097c0000, // flat_load_b32 v9, v[0:1]
            0xbe904700, // call_helper: s_getpc_b64 s[16:17]
            0x8010ff10,
            0x14, // s_add_u32 s16, s16, helper - (call_helper + 4)
            0x82118011,
            0xbe9e4910,                  // s_addc_u32 s17, s17, 0; s_swappc_b64 s[30:31], s[16:17]
            0xbfb00000,                  // s_endpgm
            0xbf890000u | (count << 4)}; // helper: s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(count)
        const bool literal_mask = uses == Uses::DisjointLanes || uses == Uses::OverlappingLanes;
        if (literal_mask)
          words[0] = 0xbefe0081; // s_mov_b32 exec_lo, 1
        if (uses == Uses::BothAfterExec)
          words.push_back(0xbefe00c1); // s_mov_b32 exec_lo, -1
        if (uses == Uses::Both || uses == Uses::BothAfterExec) {
          words.push_back(0x4a061308); // v_add_nc_u32_e32 v3, v8, v9
        } else {
          if (uses != Uses::Younger) {
            const bool independent = uses == Uses::PartialFirst || uses == Uses::OlderAfterPartial;
            words.push_back(independent ? 0x7e060301 : 0x7e060308); // v_mov v3,v1/v8
          }
          if (uses == Uses::MaskedAfterPartial || uses == Uses::RestoredAfterPartial)
            words.push_back(0xbefe0080); // s_mov_b32 exec_lo, 0
          if (literal_mask)
            words.push_back(uses == Uses::DisjointLanes ? 0xbefe0082 : 0xbefe0083);
          if (uses == Uses::ClobberedComplement)
            words.push_back(0xbe800001); // s_mov_b32 s0, s1
          if (uses == Uses::Complement || uses == Uses::ClobberedComplement)
            words.push_back(0xbefe1e00); // s_not_b32 exec_lo, s0
          if (uses == Uses::OnlyEmptyPath)
            words.push_back(0xbfa60002); // s_cbranch_execnz return
          if (uses == Uses::PartialBetween || uses == Uses::PartialFirst ||
              uses == Uses::MaskedAfterPartial || uses == Uses::OnlyEmptyPath ||
              uses == Uses::RestoredAfterPartial || literal_mask || uses == Uses::Complement ||
              uses == Uses::ClobberedComplement)
            words.push_back(0xbf89fc17); // s_waitcnt lgkmcnt(1)
          if (uses == Uses::RestoredAfterPartial)
            words.push_back(0xbefe00c1); // s_mov_b32 exec_lo, -1
          if (uses == Uses::OlderAfterPartial)
            words.push_back(0xbf89fc27); // s_waitcnt lgkmcnt(2)
          if (uses == Uses::FullBetween)
            words.push_back(0xbf89fc07); // s_waitcnt lgkmcnt(0)
          if (uses != Uses::Older)
            words.push_back(uses == Uses::OlderAfterPartial ? 0x7e060308 : 0x7e060309);
        }
        words.push_back(0xbe80481e); // s_setpc_b64 s[30:31]
        const auto image = waitcheck_test::make_gfx_code_object(words, mach, true);
        AmdGpuCodeObject object(image.data(), image.size());
        WaitcheckOptions options;
        options.check_counter_parity = true;
        const auto report = analyze_waitcnts(object, arch, options);
        ASSERT_TRUE(report.supported && report.complete) << report.analysis_error;
        const bool needs_zero = uses != Uses::Older && uses != Uses::FullBetween &&
                                uses != Uses::OlderAfterPartial &&
                                uses != Uses::MaskedAfterPartial && uses != Uses::OnlyEmptyPath &&
                                uses != Uses::DisjointLanes && uses != Uses::Complement;
        EXPECT_EQ(std::ranges::any_of(report.diagnostics,
                                      [](const auto &d) {
                                        return d.counter == WaitCounterKind::Ds &&
                                               d.producer_section_offset == 0x14;
                                      }),
                  count == 1 && needs_zero)
            << diagnostic_summary(report);
        if (count == 0) {
          const auto field =
              std::ranges::find_if(report.counter_underaccounting_diagnostics, [](const auto &d) {
                return d.counter == WaitCounterKind::Ds && d.wait_section_offset == 0x34;
              });
          EXPECT_EQ(field == report.counter_underaccounting_diagnostics.end(), needs_zero);
          if (uses == Uses::OlderAfterPartial) {
            ASSERT_NE(field, report.counter_underaccounting_diagnostics.end());
            EXPECT_NE(field->message.find("a dependency was found"), std::string::npos);
          }
        }
      }
    }
  }
}

TEST(WaitcheckTest, Gfx11GuardedFlatUsesPhysicalSourceAndDestinationLanes) {
  enum Witness { Disjoint, Overlap, Unknown };
  struct Case {
    const char *name;
    std::vector<uint32_t> instruction;
    uint32_t active, issuing;
    Witness witness;
  };
  const Case cases[] = {
      {"DPP16 FI", {0x7e0a02fa, 0xff04b109}, 2, 1, Overlap},
      {"DPP16 active sources", {0x7e0a02fa, 0xff00b109}, 2, 1, Disjoint},
      {"DPP16 identity", {0x7e0a02fa, 0xff04e409}, 2, 1, Disjoint},
      {"DPP8 FI", {0x7e0a02ea, 0xde54c109}, 2, 1, Overlap},
      {"DPP8 active sources", {0x7e0a02e9, 0xde54c109}, 2, 1, Disjoint},
      {"DPP ordinary operand", {0x4a0a12fa, 0xff04b104}, 2, 1, Disjoint},
      {"DPP both sources", {0x4a0a12fa, 0xff04b109}, 2, 1, Overlap},
      {"DPP destination", {0x7e1202fa, 0xff04b104}, 2, 1, Disjoint},
      {"PERMLANE16 FI", {0xd65b0805, 0x02010109}, 2, 1, Overlap},
      {"PERMLANE16 active sources", {0xd65b0005, 0x02010109}, 2, 1, Disjoint},
      {"PERMLANEX16 other row", {0xd65c0805, 0x02010109}, 2, 1, Disjoint},
      {"PERMLANE16 unknown selector", {0xd65b0805, 0x00100709}, 2, 1, Unknown},
      {"PERMLANE64 wave32", {0x7e0acf09}, 2, 1, Disjoint},
      {"DS_SWIZZLE inactive", {0xd8d480b1, 0x05000009}, 2, 1, Disjoint},
      {"DS_SWIZZLE active", {0xd8d480b1, 0x05000009}, 3, 1, Overlap},
      {"DS_PERMUTE active writers", {0xdac80000, 0x05000904}, 2, 1, Disjoint},
      {"DS_BPERMUTE inactive", {0xdacc0000, 0x05000904}, 2, 1, Disjoint},
      {"DS_BPERMUTE unknown address", {0xdacc0000, 0x05000904}, 3, 1, Unknown},
      {"INTERP P10 source2 quad lane 0", {0xcd000005, 0x04260704}, 2, 1, Overlap},
      {"INTERP P2 source0 quad lane 2", {0xcd010005, 0x04120709}, 2, 1, Disjoint},
      {"INTERP P2 source0 quad lane 2 active", {0xcd010005, 0x04120709}, 2, 4, Overlap},
      {"READLANE selected 0", {0xd7600005, 0x02010109}, 2, 1, Overlap},
      {"READLANE selected 1", {0xd7600005, 0x02010309}, 2, 1, Disjoint},
      {"READFIRSTLANE selected 1", {0x7e0a0509}, 2, 1, Disjoint},
      {"READFIRSTLANE empty selects 0", {0x7e0a0509}, 0, 1, Overlap},
      {"WRITELANE selected 0", {0xd7610009, 0x02010005}, 2, 1, Overlap},
      {"LDS_DIRECT_LOAD valid M0 can write quad", {0xce100009}, 2, 1, Unknown},
      {"LDS_DIRECT_LOAD other quad", {0xce100009}, 16, 1, Disjoint},
      {"LDS_PARAM_LOAD complete quad", {0xce000009}, 2, 1, Overlap},
      {"LDS_PARAM_LOAD other quad", {0xce000009}, 16, 1, Disjoint},
  };
  for (const auto mach : {EF_AMDGPU_MACH_AMDGCN_GFX1100, EF_AMDGPU_MACH_AMDGCN_GFX1150}) {
    const auto arch = mach == EF_AMDGPU_MACH_AMDGCN_GFX1100 ? ROCJITSU_CODE_ARCH_RDNA3
                                                            : ROCJITSU_CODE_ARCH_RDNA3_5;
    for (const auto &tc : cases) {
      for (bool vm : {false, true}) {
        const auto counter = vm ? WaitCounterKind::Load : WaitCounterKind::Ds;
        for (const uint32_t count : {0u, 1u}) {
          SCOPED_TRACE(testing::Message()
                       << tc.name << ", machine " << mach << ", VM " << vm << ", count " << count);
          std::vector<uint32_t> words{
              0xbefe0080u | tc.issuing,
              0xbfa50005, // EXEC_LO = issuing; branch to call if empty
              0xdc500000,
              0x087c0000, // flat_load_b32 v8, v[0:1]
              0xbfa50002,
              0xdc500000,
              0x097c0000, // branch if empty; flat_load_b32 v9, v[0:1]
              0xba1e0001,
              0xbfb00000,                             // s_call_b64 s[30:31], helper; s_endpgm
              0xbf890000u | (count << (vm ? 10 : 4)), // helper: wait all except the selected field
              0x7e060308,                             // read older v8
              0xbefe0080u | tc.active,
              vm ? 0xbf8907f7u
                 : 0xbf89fc17u}; // EXEC_LO = active; partial wait on the selected field
          words.insert(words.end(), tc.instruction.begin(), tc.instruction.end());
          words.push_back(0xbe80481e); // return
          const auto image = waitcheck_test::make_gfx_code_object(words, mach, true);
          AmdGpuCodeObject object(image.data(), image.size());
          WaitcheckOptions options;
          options.check_counter_parity = true;
          const auto report = analyze_waitcnts(object, arch, options);
          ASSERT_TRUE(report.supported && report.complete) << report.analysis_error;
          EXPECT_EQ(std::ranges::any_of(report.diagnostics,
                                        [&](const auto &d) {
                                          return d.counter == counter &&
                                                 d.producer_section_offset == 0x14 &&
                                                 d.section_offset == 0x34;
                                        }),
                    count == 1 && tc.witness != Disjoint)
              << diagnostic_summary(report);
          if (!count)
            EXPECT_EQ(std::ranges::none_of(report.counter_underaccounting_diagnostics,
                                           [&](const auto &d) {
                                             return d.counter == counter &&
                                                    d.wait_section_offset == 0x24;
                                           }),
                      tc.witness == Overlap);
        }
      }
    }
  }
}

TEST(WaitcheckTest, Gfx11UsesScalarValuesForPhysicalLaneSelectors) {
  struct Case {
    const char *name;
    std::array<uint32_t, 2> op;
    uint32_t active, disjoint_selector;
  };
  const Case cases[] = {
      {"PERMLANE16 low selector", {0xd65b0805, 0x02000709}, 2, 0x11111111},
      {"PERMLANE16 high selector", {0xd65b0805, 0x000d0109}, 256, 0x11111111},
      {"READLANE", {0xd7600005, 0x02000709}, 2, 1},
      {"WRITELANE", {0xd7610009, 0x02000605}, 2, 1},
  };
  for (uint32_t mach : {EF_AMDGPU_MACH_AMDGCN_GFX1100, EF_AMDGPU_MACH_AMDGCN_GFX1150}) {
    const auto arch = mach == EF_AMDGPU_MACH_AMDGCN_GFX1100 ? ROCJITSU_CODE_ARCH_RDNA3
                                                            : ROCJITSU_CODE_ARCH_RDNA3_5;
    for (const auto &tc : cases) {
      for (uint32_t selector : {0u, tc.disjoint_selector}) {
        for (uint32_t count : {0u, 1u}) {
          SCOPED_TRACE(testing::Message()
                       << mach << " " << tc.name << " selector " << selector << " count " << count);
          std::vector<uint32_t> words{0xbefe0081,
                                      0xbfa50005,
                                      0xdc500000,
                                      0x087c0000, // lane 0 loads v8
                                      0xbfa50002,
                                      0xdc500000,
                                      0x097c0000, // then v9
                                      0xba1e0001,
                                      0xbfb00000, // call helper; endpgm
                                      0xbf890000u | (count << 4),
                                      0x7e060308, // helper: wait, read v8
                                      0xbefe00ff,
                                      tc.active,
                                      0xbe8300ff,
                                      selector, // set EXEC_LO and s3
                                      0xbf89fc17,
                                      tc.op[0],
                                      tc.op[1],
                                      0xbe80481e}; // LGKM(1), access v9, return
          const auto image = waitcheck_test::make_gfx_code_object(words, mach, true);
          AmdGpuCodeObject object(image.data(), image.size());
          WaitcheckOptions options;
          options.check_counter_parity = true;
          const auto report = analyze_waitcnts(object, arch, options);
          ASSERT_TRUE(report.supported && report.complete) << report.analysis_error;
          EXPECT_EQ(std::ranges::any_of(report.diagnostics,
                                        [](const auto &d) {
                                          return d.counter == WaitCounterKind::Ds &&
                                                 d.producer_section_offset == 0x14 &&
                                                 d.section_offset == 0x40;
                                        }),
                    count == 1 && selector == 0)
              << diagnostic_summary(report);
          if (count == 0) {
            EXPECT_EQ(std::ranges::any_of(report.counter_underaccounting_diagnostics,
                                          [](const auto &d) {
                                            return d.counter == WaitCounterKind::Ds &&
                                                   d.wait_section_offset == 0x24;
                                          }),
                      selector != 0);
          }
        }
      }
    }
  }
}

TEST(WaitcheckTest, Gfx11DefiniteTupleWriteJustifiesWaitWithUnresolvedSourceSelector) {
  for (uint32_t mach : {EF_AMDGPU_MACH_AMDGCN_GFX1100, EF_AMDGPU_MACH_AMDGCN_GFX1150}) {
    const auto arch = mach == EF_AMDGPU_MACH_AMDGCN_GFX1100 ? ROCJITSU_CODE_ARCH_RDNA3
                                                            : ROCJITSU_CODE_ARCH_RDNA3_5;
    for (bool vm : {false, true}) {
      const auto counter = vm ? WaitCounterKind::Load : WaitCounterKind::Ds;
      for (bool immediate_consumer : {false, true}) {
        for (bool writes_tuple : {false, true}) {
          for (uint32_t count : {0u, 1u}) {
            SCOPED_TRACE(testing::Message()
                         << mach << " VM " << vm << " immediate " << immediate_consumer
                         << " writes tuple " << writes_tuple << " count " << count);
            std::vector<uint32_t> words{
                0xbefe0081, 0xbfa50005, 0xdc500000, 0x087c0000,
                0xbfa50002, 0xdc540000, 0x097c0000,              // lane 0 loads v8 and v[9:10]
                0xba1e0001, 0xbfb00000, 0xbf800000, 0x7e3c031f}; // call, end, helper
            if (immediate_consumer)
              words.push_back(0xbefe0083); // lanes 0 and 1
            words.push_back(0xbf890000u | (count << (vm ? 10 : 4)));
            if (!immediate_consumer)
              words.push_back(0xbefe0083);
            // The unknown selector may read v9. With FI set, both active
            // lanes write the destination regardless of the selected lane.
            words.insert(words.end(),
                         {0xd65b0800u | (writes_tuple ? 10u : 11u), 0x02000709, 0xbe80481e});
            const auto image = waitcheck_test::make_gfx_code_object(words, mach, true);
            AmdGpuCodeObject object(image.data(), image.size());
            WaitcheckOptions options;
            options.check_counter_parity = true;
            const auto report = analyze_waitcnts(object, arch, options);
            ASSERT_TRUE(report.supported && report.complete) << report.analysis_error;
            for (uint16_t v : {9, 10}) {
              EXPECT_EQ(std::ranges::any_of(report.diagnostics,
                                            [&](const auto &d) {
                                              return d.counter == counter &&
                                                     d.reg == (RegisterRef{RegClass::VGPR, v, 1}) &&
                                                     d.producer_section_offset == 0x14 &&
                                                     d.section_offset == 0x34;
                                            }),
                        count == 1 && (v == 9 || writes_tuple))
                  << diagnostic_summary(report);
            }
            EXPECT_EQ(report.counter_parity_indeterminate_groups, writes_tuple ? 0u : 1u);
            if (!count)
              EXPECT_EQ(report.counter_parity_exact, writes_tuple ? 2u : 0u);
          }
        }
      }
    }
  }
}

TEST(WaitcheckTest, Gfx11PerLaneMemoryResultsIntersectTheConsumersPhysicalLanes) {
  for (uint32_t mach : {EF_AMDGPU_MACH_AMDGCN_GFX1100, EF_AMDGPU_MACH_AMDGCN_GFX1150}) {
    const auto arch = mach == EF_AMDGPU_MACH_AMDGCN_GFX1100 ? ROCJITSU_CODE_ARCH_RDNA3
                                                            : ROCJITSU_CODE_ARCH_RDNA3_5;
    for (const auto &producer : std::vector<std::vector<uint32_t>>{
             {0xdc500000, 0x027c0000}, // flat_load_b32 v2, v[0:1]
             {0xdc520000, 0x027c0000}, // global_load_b32 v2, v[0:1], off
             {0xdc510000, 0x02fc0000}, // scratch_load_b32 v2, v0, off
             {0xd8d80000, 0x02000000}, // ds_load_b32 v2, v0
             {0xce000002}}) {          // lds_param_load v2, attr0.x writes the active quad
      for (unsigned active : {1u, 2u, 3u}) {
        for (bool drained : {false, true}) {
          SCOPED_TRACE(testing::Message() << mach << " opcode " << std::hex << producer.front()
                                          << " EXEC_LO " << active << " drained " << drained);
          std::vector<uint32_t> words{0xbefe0081}; // issue in lane zero
          words.insert(words.end(), producer.begin(), producer.end());
          words.insert(words.end(), {drained ? 0xbf890000u : 0xbf800000u, 0xbefe0080u | active,
                                     0x7e080302, 0xbfb00000}); // read v2 in the selected lanes
          const auto image = waitcheck_test::make_gfx_code_object(words, mach, true);
          AmdGpuCodeObject object(image.data(), image.size());
          const auto report = analyze_waitcnts(object, arch);
          ASSERT_TRUE(report.supported && report.complete) << report.analysis_error;
          for (const auto counter :
               {WaitCounterKind::Load, WaitCounterKind::Ds, WaitCounterKind::Exp}) {
            const auto opcode = producer.front();
            const bool routes = counter == WaitCounterKind::Load ? (opcode >> 24) == 0xdc
                                : counter == WaitCounterKind::Ds
                                    ? opcode == 0xdc500000 || opcode == 0xd8d80000
                                    : opcode == 0xce000002;
            EXPECT_EQ(std::ranges::any_of(report.diagnostics,
                                          [&](const auto &d) {
                                            return d.producer_section_offset == 4 &&
                                                   d.section_offset == (words.size() - 2) * 4 &&
                                                   d.counter == counter &&
                                                   d.reg == (RegisterRef{RegClass::VGPR, 2, 1});
                                          }),
                      routes && !drained && ((active & 1) || counter == WaitCounterKind::Exp))
                << diagnostic_summary(report);
          }
        }
      }
    }
  }
}

TEST(WaitcheckTest, Gfx11GuardedFlatUsesScalarFactsAcrossResolvedCalls) {
  enum class Guard { SameScc, UnknownScc, ClobberedScc, SavedMask, ChangedMask, YoungerIssue };
  for (const uint32_t mach : {EF_AMDGPU_MACH_AMDGCN_GFX1100, EF_AMDGPU_MACH_AMDGCN_GFX1150}) {
    const auto arch = mach == EF_AMDGPU_MACH_AMDGCN_GFX1100 ? ROCJITSU_CODE_ARCH_RDNA3
                                                            : ROCJITSU_CODE_ARCH_RDNA3_5;
    for (auto guard : {Guard::SameScc, Guard::UnknownScc, Guard::ClobberedScc, Guard::SavedMask,
                       Guard::ChangedMask, Guard::YoungerIssue}) {
      SCOPED_TRACE(testing::Message() << "arch " << arch << " guard " << static_cast<int>(guard));
      std::vector<uint32_t> words{
          0xbe870000,             // s_mov_b32 s7, s0
          0xbefe0007, 0xbfa50005, // s_mov_b32 exec_lo, s7; s_cbranch_execz call_helper
          0xdc500000, 0x087c0000, // flat_load_b32 v8, v[0:1]
          0xbfa50002,             // s_cbranch_execz call_helper
          0xdc500000, 0x097c0000, // flat_load_b32 v9, v[0:1]
          0xbe904700,             // call_helper: s_getpc_b64 s[16:17]
          0x8010ff10, 0x14,       // s_add_u32 s16, s16, helper - (call_helper + 4)
          0x82118011, 0xbe9e4910, // s_addc_u32 s17, s17, 0; s_swappc_b64 s[30:31], s[16:17]
          0xbfb00000,             // s_endpgm
          0xbf890010,             // helper: s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(1)
          0x7e060308};            // v_mov_b32 v3, v8
      if (guard == Guard::YoungerIssue) {
        append_inst(words, std::array<uint32_t, 2>{0xdc500000, 0x0a7c0000}); // flat load v10
        words.push_back(0xbf89fc17);                                         // s_waitcnt lgkmcnt(1)
      } else if (guard == Guard::SavedMask || guard == Guard::ChangedMask) {
        if (guard == Guard::ChangedMask)
          words.push_back(0xbe870008); // s_mov_b32 s7, s8: retains a reachable v9 use
        words.push_back(0xbefe0007);   // s_mov_b32 exec_lo, s7
        words.push_back(0xbfa60002);   // s_cbranch_execnz ret
        words.push_back(0xbefe00c1);   // s_mov_b32 exec_lo, -1
      } else {
        words.push_back(guard == Guard::UnknownScc ? 0xbf060100 : 0xbf060000);
        if (guard == Guard::ClobberedScc)
          words.push_back(0xbf060100); // s_cmp_eq_u32 s0, s1 overwrites the identity comparison
        words.push_back(0xbfa20001);   // s_cbranch_scc1 ret
      }
      words.push_back(0x7e060309); // v_mov_b32 v3, v9
      words.push_back(0xbe80481e); // ret: s_setpc_b64 s[30:31]
      const auto image = waitcheck_test::make_gfx_code_object(words, mach, true);
      AmdGpuCodeObject object(image.data(), image.size());
      const auto report = analyze_waitcnts(object, arch);
      ASSERT_TRUE(report.supported && report.complete) << report.analysis_error;
      const bool reachable =
          guard == Guard::UnknownScc || guard == Guard::ClobberedScc || guard == Guard::ChangedMask;
      EXPECT_EQ(std::ranges::any_of(report.diagnostics,
                                    [](const auto &finding) {
                                      return finding.counter == WaitCounterKind::Ds &&
                                             finding.reg == RegisterRef{RegClass::VGPR, 9, 1};
                                    }),
                reachable)
          << diagnostic_summary(report);
    }
  }
}

TEST(WaitcheckTest, Gfx11IncompleteGuardedSearchKeepsHelperWaitPolicy) {
  // Independent mask diamonds consume the feasibility budget before the
  // helper consumer. Its local zero cannot be attributed to that search.
  std::vector<uint32_t> words;
  for (uint32_t sgpr = 4; sgpr < 20; ++sgpr) {
    words.insert(words.end(), {0xbefe0000u | sgpr, 0xbfa50001, 0xbe830004}); // s_mov_b32 s3, s4
  }
  const size_t body_offset = words.size() * sizeof(uint32_t);
  words.insert(words.end(),
               {0xbefe0000, 0xbfa50005, 0xdc500000, 0x087c0000, 0xbfa50002, 0xdc500000, 0x097c0000,
                0xbe904700, 0x8010ff10, 0x14, 0x82118011, 0xbe9e4910, 0xbfb00000, 0xbf890000,
                0x7e060308, 0xbefe0080, 0xbf89fc17, 0x7e060309, 0xbe80481e});
  const auto image =
      waitcheck_test::make_gfx_code_object(words, EF_AMDGPU_MACH_AMDGCN_GFX1100, true);
  AmdGpuCodeObject object(image.data(), image.size());
  WaitcheckOptions options;
  options.check_counter_parity = true;
  const auto report = analyze_waitcnts(object, ROCJITSU_CODE_ARCH_RDNA3, options);
  ASSERT_TRUE(report.supported && report.complete) << report.analysis_error;
  const auto found =
      std::ranges::find_if(report.counter_underaccounting_diagnostics, [&](const auto &d) {
        return d.wait_section_offset == body_offset + 0x34 && d.counter == WaitCounterKind::Ds;
      });
  ASSERT_NE(found, report.counter_underaccounting_diagnostics.end());
  EXPECT_FALSE(found->has_required_dependency);
}

TEST(WaitcheckTest, Gfx11GuardedFlatJoinsNopOnlyExecBranches) {
  for (uint32_t mach : {EF_AMDGPU_MACH_AMDGCN_GFX1100, EF_AMDGPU_MACH_AMDGCN_GFX1150}) {
    const auto arch = mach == EF_AMDGPU_MACH_AMDGCN_GFX1100 ? ROCJITSU_CODE_ARCH_RDNA3
                                                            : ROCJITSU_CODE_ARCH_RDNA3_5;
    for (uint32_t branch : {0xbfa50002u, 0xbfa60002u}) {
      for (uint32_t count : {0u, 1u, 2u}) {
        SCOPED_TRACE(testing::Message()
                     << arch << " conditional branch " << branch << " LGKM " << count);
        std::vector<uint32_t> words;
        for (uint32_t sgpr = 4; sgpr < 20; ++sgpr)
          words.insert(words.end(), {0xbefe0000u | sgpr, branch, 0xbf800000, 0xbf800001});
        const uint64_t body = words.size() * sizeof(uint32_t);
        words.insert(words.end(), {0xbefe0000, 0xbfa50005, 0xdc500000, 0x087c0000, 0xbfa50002,
                                   0xdc500000, 0x097c0000, 0xbe904700, 0x8010ff10, 0x14, 0x82118011,
                                   0xbe9e4910, 0xbfb00000, 0xbf890000u | (count << 4), 0x7e060308,
                                   0xbefe0080, 0xbf89fc17, 0x7e060309, 0xbe80481e});
        const auto image = waitcheck_test::make_gfx_code_object(words, mach, true);
        AmdGpuCodeObject object(image.data(), image.size());
        WaitcheckOptions options;
        options.check_counter_parity = true;
        const auto report = analyze_waitcnts(object, arch, options);
        ASSERT_TRUE(report.supported && report.complete) << report.analysis_error;
        EXPECT_EQ(std::ranges::any_of(report.diagnostics,
                                      [&](const auto &d) {
                                        return d.counter == WaitCounterKind::Ds &&
                                               d.section_offset == body + 0x38 &&
                                               d.producer_section_offset == body + 8;
                                      }),
                  count == 2);
        EXPECT_FALSE(std::ranges::any_of(report.diagnostics, [&](const auto &d) {
          return d.counter == WaitCounterKind::Ds && d.section_offset == body + 0x44;
        }));
        if (count == 0) {
          const auto found =
              std::ranges::find_if(report.counter_underaccounting_diagnostics, [&](const auto &d) {
                return d.counter == WaitCounterKind::Ds && d.wait_section_offset == body + 0x34;
              });
          ASSERT_NE(found, report.counter_underaccounting_diagnostics.end());
          EXPECT_FALSE(found->has_required_dependency);
        }
      }
    }
  }
}

TEST(WaitcheckTest, Gfx11ExecBranchPreservesAConditionalScalarMaskWrite) {
  for (uint32_t branch : {0xbfa50001u, 0xbfa60001u}) {
    // Depending on s0, the fallthrough changes s7 and enables lane zero at the read.
    const std::vector<uint32_t> words{
        0xbefe0081, 0xdc500000, 0x097c0000,  // EXEC_LO=1; flat_load_b32 v9, v[0:1]
        0xbe870080, 0xbefe0000, branch,      // s7=0; EXEC_LO=s0; conditional skip
        0xbe870081, 0xbefe0007,              // s7=1; join: EXEC_LO=s7
        0xbf890010, 0x7e060309, 0xbfb00000}; // s_waitcnt lgkmcnt(1) vmcnt(0); consume v9
    const auto image =
        waitcheck_test::make_gfx_code_object(words, EF_AMDGPU_MACH_AMDGCN_GFX1100, true);
    AmdGpuCodeObject object(image.data(), image.size());
    const auto report = analyze_waitcnts(object, ROCJITSU_CODE_ARCH_RDNA3);
    ASSERT_TRUE(report.supported && report.complete) << report.analysis_error;
    EXPECT_TRUE(std::ranges::any_of(report.diagnostics, [](const auto &finding) {
      return finding.counter == WaitCounterKind::Ds && finding.section_offset == 0x24 &&
             finding.producer_section_offset == 4;
    })) << diagnostic_summary(report);
  }
}

TEST(WaitcheckTest, Gfx11GuardedFlatUsesTheProvedYoungerLgkmCount) {
  for (const uint32_t mach : {EF_AMDGPU_MACH_AMDGCN_GFX1100, EF_AMDGPU_MACH_AMDGCN_GFX1150}) {
    const auto arch = mach == EF_AMDGPU_MACH_AMDGCN_GFX1100 ? ROCJITSU_CODE_ARCH_RDNA3
                                                            : ROCJITSU_CODE_ARCH_RDNA3_5;
    for (bool drained : {false, true}) {
      for (unsigned count : {0u, 1u, 2u, 3u}) {
        SCOPED_TRACE(testing::Message()
                     << arch << " alternate drain " << drained << " LGKM " << count);
        const std::vector<uint32_t> words{
            0xbefe0000, // s_mov_b32 exec_lo, s0
            0xbfa50009, // s_cbranch_execz join
            0xdc500000,
            0x027c0000,                          // flat_load_b32 v2, v[0:1]
            0xbf068003,                          // s_cmp_eq_u32 s3, 0
            0xbfa20002,                          // s_cbranch_scc1 extra
            drained ? 0xbf890007u : 0xbf800000u, // s_waitcnt vmcnt(0) lgkmcnt(0), or s_nop 0
            0xbfa00003,                          // s_branch join
            0xbfa50002,                          // extra: s_cbranch_execz join
            0xdc680000,
            0x007c0400, // flat_store_b32 v[0:1], v4
            0xbefe00c1, // join: s_mov_b32 exec_lo, -1
            0xdc680000,
            0x007c0400,                 // flat_store_b32 v[0:1], v4
            0xbf89fc07u | (count << 4), // s_waitcnt lgkmcnt(count)
            0x7e060302,                 // v_mov_b32 v3, v2
            0xbfb00000};                // s_endpgm
        const auto image = waitcheck_test::make_gfx_code_object(words, mach, true);
        AmdGpuCodeObject object(image.data(), image.size());
        WaitcheckOptions options;
        options.check_counter_parity = true;
        const auto report = analyze_waitcnts(object, arch, options);
        ASSERT_TRUE(report.supported && report.complete) << report.analysis_error;
        const uint32_t safe = drained ? 2u : 1u;
        EXPECT_EQ(std::ranges::any_of(report.diagnostics,
                                      [](const auto &d) {
                                        return d.section_offset == 0x3c &&
                                               d.producer_section_offset == 8 &&
                                               d.counter == WaitCounterKind::Ds;
                                      }),
                  count > safe);
        if (count)
          continue;
        const auto &fields = report.counter_underaccounting_diagnostics;
        const auto found = std::ranges::find_if(fields, [](const auto &d) {
          return d.wait_section_offset == 0x38 && d.counter == WaitCounterKind::Ds;
        });
        ASSERT_NE(found, fields.end());
        EXPECT_EQ(found->required_count, safe);
      }
    }
  }
}

TEST(WaitcheckTest, Gfx11ImmediateGuardedFlatKeepsTheStrongestProvedFiniteCount) {
  for (const uint32_t mach : {EF_AMDGPU_MACH_AMDGCN_GFX1100, EF_AMDGPU_MACH_AMDGCN_GFX1150}) {
    const auto arch = mach == EF_AMDGPU_MACH_AMDGCN_GFX1100 ? ROCJITSU_CODE_ARCH_RDNA3
                                                            : ROCJITSU_CODE_ARCH_RDNA3_5;
    for (unsigned guarded_stores : {2u, 6u, 61u}) {
      for (bool drained : {false, true}) {
        const uint32_t safe = drained ? guarded_stores + 1 : 1;
        for (uint32_t count : {0u, safe, safe + 1}) {
          SCOPED_TRACE(testing::Message() << arch << " guarded stores " << guarded_stores
                                          << " alternate drain " << drained << " LGKM " << count);
          const uint32_t skip_stores = 2 * guarded_stores;
          std::vector<uint32_t> words{0xbefe0000,                      // s_mov_b32 exec_lo, s0
                                      0xbfa50000u | (7 + skip_stores), // s_cbranch_execz join
                                      0xdc500000,
                                      0x027c0000, // flat_load_b32 v2, v[0:1]
                                      0xbf068003,
                                      0xbfa20002, // s_cmp_eq_u32 s3, 0; s_cbranch_scc1 extra
                                      drained ? 0xbf890007u
                                              : 0xbf800000u, // drain the alternate path, or s_nop
                                      0xbfa00000u | (1 + skip_stores), // s_branch join
                                      0xbfa50000u | skip_stores}; // extra: s_cbranch_execz join
          for (unsigned i = 0; i < guarded_stores; ++i)
            append_inst(words, std::array<uint32_t, 2>{0xdc680000, 0x007c0400}); // flat_store_b32
          words.push_back(0xbefe00c1); // join: s_mov_b32 exec_lo, -1
          append_inst(words, std::array<uint32_t, 2>{0xdc680000, 0x007c0400}); // flat_store_b32
          const uint64_t wait = words.size() * sizeof(uint32_t);
          words.push_back(0xbf89fc07u | (count << 4)); // s_waitcnt lgkmcnt(count)
          words.push_back(0x7e060302);                 // v_mov_b32 v3, v2
          words.push_back(0xbfb00000);                 // s_endpgm
          const auto image = waitcheck_test::make_gfx_code_object(words, mach, true);
          AmdGpuCodeObject object(image.data(), image.size());
          WaitcheckOptions options;
          options.check_counter_parity = true;
          const auto report = analyze_waitcnts(object, arch, options);
          ASSERT_TRUE(report.supported && report.complete) << report.analysis_error;
          EXPECT_EQ(std::ranges::any_of(report.diagnostics,
                                        [&](const auto &finding) {
                                          return finding.section_offset == wait + 4 &&
                                                 finding.producer_section_offset == 8 &&
                                                 finding.counter == WaitCounterKind::Ds;
                                        }),
                    count > safe)
              << diagnostic_summary(report);
          if (count == 0) {
            const auto &fields = report.counter_underaccounting_diagnostics;
            const auto found = std::ranges::find_if(fields, [&](const auto &finding) {
              return finding.wait_section_offset == wait && finding.counter == WaitCounterKind::Ds;
            });
            ASSERT_NE(found, fields.end());
            EXPECT_EQ(found->required_count, safe);
          }
        }
      }
    }
  }
}

TEST(WaitcheckTest, OrderedQueueAdmissionAppliesToAdjacentOperandsAndExactWaits) {
  for (bool cdna5 : {false, true}) {
    const unsigned capacity = cdna5 ? 63 : 15;
    const auto arch = cdna5 ? ROCJITSU_CODE_ARCH_CDNA5 : ROCJITSU_CODE_ARCH_CDNA4;
    const auto mach = cdna5 ? EF_AMDGPU_MACH_AMDGCN_GFX1250 : EF_AMDGPU_MACH_AMDGCN_GFX950;
    const auto counter = cdna5 ? WaitCounterKind::Load : WaitCounterKind::Ds;
    for (bool adjacent : {false, true}) {
      for (bool issuing_consumer : {false, true}) {
        for (unsigned younger : {capacity - 2, capacity - 1, capacity}) {
          SCOPED_TRACE(testing::Message() << cdna5 << " adjacent " << adjacent << " issuing "
                                          << issuing_consumer << " younger " << younger);
          std::vector<uint32_t> words;
          auto append_read = [&](unsigned dst, unsigned address) {
            if (cdna5)
              words.insert(words.end(), {0xee05007c, dst, address});
            else
              words.insert(words.end(), {0xd86c0000, (dst << 24) | address});
          };
          auto younger_reads = [&] {
            for (unsigned i = 0; i < younger; ++i)
              append_read(20 + i, cdna5 ? 0 : 1);
          };
          append_read(9, cdna5 ? 0 : 1);
          if (adjacent)
            younger_reads();
          const size_t wait_word = words.size();
          words.push_back(0);
          if (!adjacent) {
            words.push_back(0xbf800000); // s_nop
            younger_reads();
          }
          const auto consumer_pc = words.size() * sizeof(uint32_t);
          if (issuing_consumer)
            append_read(7, cdna5 ? 8 : 9); // physical global address v[8:9] or LDS address v9
          else
            words.push_back(0x7e060309); // v_mov_b32 v3,v9
          words.push_back(cdna5 ? 0xbfb00000 : 0xbf810000);
          const bool pending = younger + unsigned(issuing_consumer) < capacity;
          const unsigned threshold = adjacent ? younger : 0u;
          for (const unsigned encoded : {0u, 1u, capacity}) {
            words[wait_word] = cdna5 ? 0xbfc00000 | encoded : 0xbf8cc07f | (encoded << 8);
            const auto image = waitcheck_test::make_gfx_code_object(words, mach, cdna5);
            AmdGpuCodeObject object(image.data(), image.size());
            WaitcheckOptions options;
            options.check_counter_parity = true;
            const auto report = analyze_waitcnts(object, arch, options);
            ASSERT_TRUE(report.supported && report.complete) << report.analysis_error;
            EXPECT_EQ(std::ranges::any_of(report.diagnostics,
                                          [&](const auto &d) {
                                            return d.counter == counter &&
                                                   d.producer_section_offset == 0 &&
                                                   d.section_offset == consumer_pc;
                                          }),
                      pending && encoded > threshold);
            if (encoded)
              continue;
            const auto diag = std::ranges::find_if(
                report.counter_underaccounting_diagnostics, [&](const auto &d) {
                  return d.counter == counter &&
                         d.wait_section_offset == wait_word * sizeof(uint32_t);
                });
            if (pending && threshold == 0) {
              EXPECT_EQ(diag, report.counter_underaccounting_diagnostics.end());
              EXPECT_EQ(report.counter_parity_exact, 1u);
            } else {
              ASSERT_NE(diag, report.counter_underaccounting_diagnostics.end());
              EXPECT_EQ(diag->has_required_dependency, pending);
              if (pending)
                EXPECT_EQ(diag->required_count, threshold);
            }
          }
        }
      }
    }
  }
}

TEST(WaitcheckTest, Gfx1250OrdinaryLoadParityTracksLaterQueueAdmission) {
  constexpr uint32_t scratch = 0xed05007c;
  constexpr uint32_t global = 0xee05007c;
  for (uint32_t producer_op : {scratch, global}) {
    for (bool independent_queue : {false, true}) {
      for (unsigned younger : {61u, 62u, 63u}) {
        SCOPED_TRACE(testing::Message() << producer_op << " independent " << independent_queue
                                        << " younger " << younger);
        std::vector<uint32_t> words{producer_op, 9, 0, scratch, 8, 0, 0xbfc00000, 0xbf800000};
        for (unsigned i = 0; i < younger; ++i) {
          if (independent_queue)
            words.insert(words.end(), {0xd8d80000, (10u + i) << 24});
          else
            words.insert(words.end(), {global, 10u + i, 0});
        }
        const auto consumer_pc = words.size() * sizeof(uint32_t);
        words.insert(words.end(), {0x7e060309, 0xbfb00000}); // v_mov_b32 v3, v9; s_endpgm
        const bool pending = independent_queue || 1u + younger < 63u;
        for (uint32_t count : {0u, 1u, 2u, 63u}) {
          words[6] = 0xbfc00000 | count;
          const auto image =
              waitcheck_test::make_gfx_code_object(words, EF_AMDGPU_MACH_AMDGCN_GFX1250, true);
          AmdGpuCodeObject object(image.data(), image.size());
          WaitcheckOptions options;
          options.check_counter_parity = true;
          const auto report = analyze_waitcnts(object, ROCJITSU_CODE_ARCH_CDNA5, options);
          ASSERT_TRUE(report.supported && report.complete) << report.analysis_error;
          EXPECT_EQ(std::ranges::any_of(report.diagnostics,
                                        [&](const auto &d) {
                                          return d.counter == WaitCounterKind::Load &&
                                                 d.producer_section_offset == 0 &&
                                                 d.section_offset == consumer_pc;
                                        }),
                    count > 1 && pending);
          if (count)
            continue;
          const auto field =
              std::ranges::find_if(report.counter_underaccounting_diagnostics, [](const auto &d) {
                return d.counter == WaitCounterKind::Load && d.wait_section_offset == 24;
              });
          ASSERT_NE(field, report.counter_underaccounting_diagnostics.end());
          EXPECT_EQ(field->has_required_dependency, pending);
          if (pending) {
            EXPECT_EQ(field->required_count, 1u);
            EXPECT_EQ(field->consumer_section_offset, consumer_pc);
          }
        }
      }
    }
  }
}

TEST(WaitcheckTest, Gfx950IndexedLdsParityFollowsQueueAdmissionToTheActualConsumer) {
  for (unsigned younger_after_wait : {12u, 13u, 14u, 24u}) {
    for (bool ds_consumer : {false, true}) {
      SCOPED_TRACE(testing::Message() << younger_after_wait << " DS consumer " << ds_consumer);
      std::vector<uint32_t> words{0xd86c0000, 0x09000001,   // ds_read_b32 v9, v1
                                  0xd86c0000, 0x08000001,   // ds_read_b32 v8, v1
                                  0xbf8cc07f, 0xbf800000}; // s_waitcnt lgkmcnt(0); s_nop
      for (unsigned i = 0; i < younger_after_wait; ++i)
        words.insert(words.end(), {0xd86c0000, 0x08000001});
      const auto consumer_pc = words.size()*sizeof(uint32_t);
      if (ds_consumer)
        words.insert(words.end(), {0xd86c0000, 0x07000009}); // issue before reading v9
      else
        words.push_back(0x7e060309); // v_mov_b32 v3,v9
      words.push_back(0xbf810000);
      const auto analyze = [&](uint32_t count) {
        words[4] = 0xbf8cc07fu | (count << 8);
        const auto image =
            waitcheck_test::make_gfx_code_object(words, EF_AMDGPU_MACH_AMDGCN_GFX950, false);
        AmdGpuCodeObject object(image.data(), image.size());
        WaitcheckOptions options;
        options.check_counter_parity = true;
        return analyze_waitcnts(object, ROCJITSU_CODE_ARCH_CDNA4, options);
      };
      const bool still_pending = 1u + younger_after_wait + unsigned(ds_consumer) < 15u;
      for (uint32_t count : {0u, 1u, 2u, 15u}) {
        const auto report = analyze(count);
        ASSERT_TRUE(report.supported && report.complete) << report.analysis_error;
        EXPECT_EQ(std::ranges::any_of(report.diagnostics,
                                      [&](const auto &d) {
                                        return d.counter == WaitCounterKind::Ds &&
                                               d.producer_section_offset == 0 &&
                                               d.section_offset == consumer_pc;
                                      }),
                  count > 1 && still_pending);
        if (count)
          continue;
        const auto field =
            std::ranges::find_if(report.counter_underaccounting_diagnostics, [](const auto &d) {
              return d.counter == WaitCounterKind::Ds && d.wait_section_offset == 0x10;
            });
        ASSERT_NE(field, report.counter_underaccounting_diagnostics.end());
        EXPECT_EQ(field->has_required_dependency, still_pending);
        if (still_pending) {
          EXPECT_EQ(field->required_count, 1u);
          EXPECT_EQ(field->consumer_section_offset, consumer_pc);
        }
      }
    }
  }
}

TEST(WaitcheckTest, Gfx11GuardedFlatWaitChecksEveryLaterConsumerBeforeRelaxing) {
  for (uint32_t mach : {EF_AMDGPU_MACH_AMDGCN_GFX1100, EF_AMDGPU_MACH_AMDGCN_GFX1150}) {
    const auto arch = mach == EF_AMDGPU_MACH_AMDGCN_GFX1100 ? ROCJITSU_CODE_ARCH_RDNA3
                                                            : ROCJITSU_CODE_ARCH_RDNA3_5;
    for (bool competing_use : {false, true}) {
      for (bool partial_wait : {false, true}) {
        for (uint32_t count : {0u, 1u, 2u, 3u, 63u}) {
          SCOPED_TRACE(testing::Message() << arch << " later use " << competing_use << " partial "
                                          << partial_wait << " LGKM " << count);
          const std::vector<uint32_t> words{
              0xbefe0081,
              0xbfa50009, // EXEC_LO=1; s_cbranch_execz join
              0xdc500000,
              0x027c0000, // flat_load_b32 v2, v[0:1]
              0xbf068003,
              0xbfa20002, // s_cmp_eq_u32 s3,0; s_cbranch_scc1 extra
              0xbf890007,
              0xbfa00003, // drain alternate path; s_branch join
              0xbfa50002,
              0xdc680000,
              0x007c0400, // extra: skip masked flat_store_b32
              0xbefe00c1,
              0xdc680000,
              0x007c0400,                 // join: EXEC_LO=-1; flat_store_b32
              0xbf89fc07u | (count << 4), // target s_waitcnt lgkmcnt(count)
              0xbefe0082,
              0x7e060302,                               // lane 1 uses v2, issued in lane 0
              partial_wait ? 0xbf89fc37u : 0xbf800000u, // optional later lgkmcnt(3)
              competing_use ? 0xbefe0081u : 0xbefe0082u,
              0x7e080302,
              0xbfb00000}; // second use of v2; s_endpgm
          const auto image = waitcheck_test::make_gfx_code_object(words, mach, true);
          AmdGpuCodeObject object(image.data(), image.size());
          WaitcheckOptions options;
          options.check_counter_parity = true;
          const auto report = analyze_waitcnts(object, arch, options);
          ASSERT_TRUE(report.supported && report.complete) << report.analysis_error;
          EXPECT_FALSE(std::ranges::any_of(report.diagnostics, [](const auto &d) {
            return d.counter == WaitCounterKind::Ds && d.section_offset == 0x40;
          }));
          EXPECT_EQ(std::ranges::any_of(report.diagnostics,
                                        [](const auto &d) {
                                          return d.counter == WaitCounterKind::Ds &&
                                                 d.section_offset == 0x4c;
                                        }),
                    competing_use && count > 2);
          if (count != 0)
            continue;
          const auto found =
              std::ranges::find_if(report.counter_underaccounting_diagnostics, [](const auto &d) {
                return d.counter == WaitCounterKind::Ds && d.wait_section_offset == 0x38;
              });
          ASSERT_NE(found, report.counter_underaccounting_diagnostics.end());
          EXPECT_EQ(found->has_required_dependency, competing_use);
          if (competing_use) {
            EXPECT_EQ(found->required_count, 2u);
            EXPECT_EQ(found->consumer_section_offset, 0x4cu);
          }
        }
      }
    }
  }
}

TEST(WaitcheckTest, Gfx11EmptyExecFlatCannotAgeAnIsolatedLdsResult) {
  for (auto arch : {ROCJITSU_CODE_ARCH_RDNA3, ROCJITSU_CODE_ARCH_RDNA3_5})
    for (uint32_t exec : {0xbefe0080u, 0xbefe0008u, 0xbefe00c1u}) {
      SCOPED_TRACE(testing::Message() << arch << " EXEC_LO write " << exec);
      const std::vector<uint32_t> words{0xd8d80000, 0x02000001,  // ds_load_b32 v2,v1
                                        exec,                    // s_mov_b32 exec_lo, 0 / s8 / -1
                                        0xdc680000, 0x007c0503,  // flat_store_b32 v[3:4],v5
                                        0xbefe00c1,              // s_mov_b32 exec_lo, -1
                                        0xbf89fc17,              // s_waitcnt lgkmcnt(1)
                                        0x7e0c0302, 0xbfb00000}; // v_mov_b32 v6,v2; s_endpgm
      const auto report = analyze_waitcnts(TestCodeObject(words), arch);
      ASSERT_TRUE(report.supported && report.complete) << report.analysis_error;
      ASSERT_EQ(report.diagnostics.size(), exec != 0xbefe00c1) << diagnostic_summary(report);
      if (!report.diagnostics.empty()) {
        EXPECT_EQ(report.diagnostics[0].reg, (RegisterRef{RegClass::VGPR, 2, 1}));
        EXPECT_EQ(report.diagnostics[0].required_count, 0u);
      }
    }
}

TEST(WaitcheckTest, Gfx11SkippedFlatDoesNotForceLdsAdmission) {
  for (auto arch : {ROCJITSU_CODE_ARCH_RDNA3, ROCJITSU_CODE_ARCH_RDNA3_5})
    for (bool active : {false, true}) {
      std::vector<uint32_t> words{0xbefe00c1u, 0xd8d80000, 0x02000001}; // EXEC_LO=-1; load v2.
      for (unsigned younger = 0; younger < 62; ++younger)
        words.insert(words.end(), {0xd8d80000, 0x07000001}); // ds_load_b32 v7,v1
      words.insert(words.end(), {active ? 0xbefe00c1u : 0xbefe0080u, 0xdc680000, 0x007c0503,
                                 0xbefe00c1, 0x7e0c0302, 0xbfb00000});
      const auto report = analyze_waitcnts(TestCodeObject(words), arch);
      ASSERT_TRUE(report.supported && report.complete) << report.analysis_error;
      ASSERT_EQ(report.diagnostics.size(), !active) << diagnostic_summary(report);
      if (!active) {
        EXPECT_EQ(report.diagnostics[0].reg, (RegisterRef{RegClass::VGPR, 2, 1}));
        EXPECT_EQ(report.diagnostics[0].required_count, 62u);
      }
    }
}

TEST(WaitcheckTest, Gfx11FlatStoreAgesTheDsSideOfAnEarlierFlatLoad) {
  for (auto arch : {ROCJITSU_CODE_ARCH_RDNA3, ROCJITSU_CODE_ARCH_RDNA3_5})
    for (uint32_t ds : {1u, 2u}) {
      SCOPED_TRACE(testing::Message() << arch << " lgkm=" << ds);
      std::vector<uint32_t> words{0xdc500000,
                                  0x027c0000, // flat_load_b32 v2,v[0:1]
                                  0xdc680000,
                                  0x007c0300,             // flat_store_b32 v[0:1],v3
                                  0xbfbd0000,             // s_barrier
                                  0xbf890007 | (ds << 4), // s_waitcnt vmcnt(0) lgkmcnt(ds)
                                  0xdc680000,
                                  0x007c0200,  // flat_store_b32 v[0:1],v2
                                  0xbfb00000}; // s_endpgm
      TestCodeObject object(words);
      const auto report = analyze_waitcnts(object, arch);
      ASSERT_TRUE(report.supported) << report.analysis_error;
      ASSERT_EQ(report.diagnostics.size(), ds > 1) << diagnostic_summary(report);
      if (ds > 1) {
        EXPECT_EQ(report.diagnostics.front().counter, WaitCounterKind::Ds);
        EXPECT_EQ(report.diagnostics.front().required_count, 1u);
      }
    }
}

TEST(WaitcheckTest, SplitCounterFlatLoadsAgeBothRoutesIndependently) {
  for (auto arch : {ROCJITSU_CODE_ARCH_RDNA4, ROCJITSU_CODE_ARCH_CDNA5})
    for (unsigned load : {3u, 4u})
      for (unsigned ds : {3u, 4u}) {
        SCOPED_TRACE(testing::Message() << arch << " load=" << load << " ds=" << ds);
        std::vector<uint32_t> program;
        for (uint32_t dst = 0; dst < 4; ++dst)
          append_inst(program, flat_load_b32(dst));
        append_inst(program, sopp(72, (load << 8) | ds));
        append_inst(program, v_mov_b32(5, 0));
        TestCodeObject object(program);
        const auto report = analyze_waitcnts(object, arch);
        ASSERT_TRUE(report.supported) << report.analysis_error;
        ASSERT_EQ(report.diagnostics.size(), (load > 3) + (ds > 3)) << diagnostic_summary(report);
        for (const auto &diagnostic : report.diagnostics) {
          EXPECT_EQ(diagnostic.required_count, 3u);
          EXPECT_TRUE((diagnostic.counter == WaitCounterKind::Load && load > 3) ||
                      (diagnostic.counter == WaitCounterKind::Ds && ds > 3));
        }
      }
}

TEST(WaitcheckTest, Gfx1250CombinedWaitCompletesFlatReplayOnEitherRoute) {
  constexpr auto arch = ROCJITSU_CODE_ARCH_CDNA5;
  for (bool same_family : {false, true}) {
    for (uint32_t load : {1u, 2u}) {
      for (uint32_t ds : {1u, 2u}) {
        SCOPED_TRACE(testing::Message() << same_family << " load=" << load << " ds=" << ds);
        // Independent destination and address pairs keep the observer specific
        // to the older address. A GLOBAL_LOAD adds no guaranteed DS credit.
        std::vector<uint32_t> program{0xb9800641u, 1u, // MODE.REPLAY_MODE
                                     0xec05407cu, 8u, 4u}; // flat_load_b64 v[8:9], v[4:5]
        if (same_family)
          program.insert(program.end(), {0xec05407cu, 10u, 6u});
        else
          append_inst(program, global_load_b32(10, 6));
        append_inst(program, sopp(72, (load << 8) | ds));
        append_inst(program, v_mov_b32(4, 0));
        append_inst(program, s_endpgm());
        const auto image =
            waitcheck_test::make_gfx_code_object(program, EF_AMDGPU_MACH_AMDGCN_GFX1250, true);
        AmdGpuCodeObject object(image.data(), image.size());
        const auto report = analyze_waitcnts_for_kernel(object, arch, 0);
        ASSERT_TRUE(report.complete) << report.analysis_error;
        const bool warned = std::ranges::any_of(report.diagnostics, [](const auto &diagnostic) {
          return diagnostic.counter == WaitCounterKind::X && diagnostic.reg.index == 4;
        });
        EXPECT_EQ(warned, !(same_family && load == 1 && ds == 1)) << diagnostic_summary(report);
      }
    }
  }
}

TEST(WaitcheckTest, Gfx1250XcntParityRetainsJointFlatCompletion) {
  WaitcheckOptions options;
  options.check_counter_parity = true;
  for (bool stores : {false, true}) {
    for (uint32_t split : {1u, 2u}) {
      for (uint32_t x : {1u, 2u, 63u}) {
        SCOPED_TRACE(testing::Message() << stores << " split=" << split << " x=" << x);
        // The older global store protects v40. Each FLAT contributes a younger
        // X credit; clearing the first FLAT also retires the ordered store.
        std::vector<uint32_t> words{0xb9800641u, 1u, 0xee06807cu, 0x14000000u, 100u};
        if (stores) {
          words.insert(words.end(), {0xec06807cu, 0x16000000u, 4u, 0xec06807cu, 0x17000000u, 6u});
        } else {
          words.insert(words.end(), {0xec05007cu, 44u, 4u, 0xec05007cu, 46u, 6u});
        }
        append_inst(words, sopp(stores ? 73 : 72, (split << 8) | split));
        append_inst(words, s_wait_xcnt(x));
        append_inst(words, v_mov_b32(40, 0));
        append_inst(words, s_endpgm());
        const auto bytes =
            waitcheck_test::make_gfx_code_object(words, EF_AMDGPU_MACH_AMDGCN_GFX1250, true);
        AmdGpuCodeObject obj(bytes.data(), bytes.size());
        const auto report = analyze_waitcnts_for_kernel(obj, ROCJITSU_CODE_ARCH_CDNA5, 0, options);
        ASSERT_TRUE(report.complete) << report.analysis_error;
        const auto &rows = report.counter_underaccounting_diagnostics;
        const auto found = std::ranges::find(rows, WaitCounterKind::X,
                                             &WaitcheckCounterUnderaccountingDiagnostic::counter);
        // For stores, STORECNT(2) also completes the older same-family store directly.
        const bool needs_x = !stores && split == 2;
        if (x < 63 && (!needs_x || x == 1)) {
          ASSERT_NE(found, rows.end());
          EXPECT_EQ(found->has_required_dependency, needs_x);
          EXPECT_EQ(found->required_count, needs_x ? 2u : 63u);
        } else {
          EXPECT_EQ(found, rows.end());
        }
        const bool missing = std::ranges::any_of(report.diagnostics, [](const auto &diag) {
          return diag.counter == WaitCounterKind::X && diag.reg.index == 40;
        });
        EXPECT_EQ(missing, needs_x && x == 63);
      }
    }
  }
}

TEST(WaitcheckTest, Gfx1250CompletedFlatRetiresOnlyGuaranteedOlderTranslations) {
  for (bool store_first : {false, true}) {
    for (uint32_t count : {0u, 1u, 2u}) {
      SCOPED_TRACE(testing::Message() << store_first << " count=" << count);
      std::vector<uint32_t> program{0xb9800641u, 1u}; // MODE.REPLAY_MODE
      // global_store_b64 v4, v[2:3], s[4:5]
      const std::initializer_list<uint32_t> store{0xee06c004u, 0x01000000u, 4u};
      if (store_first)
        program.insert(program.end(), store);
      // flat_load_b64 v[0:1], v4, s[10:11]
      program.insert(program.end(), {0xec05400au, 0u, 4u});
      if (count)
        program.insert(program.end(), {0xec05400au, 8u, 4u});
      if (!store_first)
        program.insert(program.end(), store);
      append_inst(program, sopp(72, (count << 8) | count));
      append_inst(program, v_mov_b32(2, 0));
      append_inst(program, s_endpgm());
      const auto bytes =
          waitcheck_test::make_gfx_code_object(program, EF_AMDGPU_MACH_AMDGCN_GFX1250, true);
      AmdGpuCodeObject object(bytes.data(), bytes.size());
      const auto report = analyze_waitcnts_for_kernel(object, ROCJITSU_CODE_ARCH_CDNA5, 0);
      ASSERT_TRUE(report.complete) << report.analysis_error;
      const bool warns = std::ranges::any_of(report.diagnostics, [](const auto &diagnostic) {
        return diagnostic.counter == WaitCounterKind::X && diagnostic.reg.index == 2;
      });
      EXPECT_EQ(warns, !store_first || count == 2) << diagnostic_summary(report);
    }
  }
}

TEST(WaitcheckTest, Gfx1250CounterParityRecognizesOrderedDsOverwriteNeedsNoWait) {
  std::vector<uint32_t> program;
  append_inst(program, ds_load_b32(0, 12));
  append_inst(program, ds_load_b32(2, 12));
  append_inst(program, ds_load_b32(4, 12));
  append_inst(program, sopp(70, 2)); // s_wait_dscnt 2 retires the oldest result.
  append_inst(program, ds_load_b32(0, 12));

  WaitcheckOptions options;
  options.check_counter_parity = true;
  TestCodeObject code_object(program);
  auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_CDNA5, options);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.passed()) << diagnostic_summary(report);
  EXPECT_EQ(report.counter_parity_wait_groups, 1u);
  EXPECT_EQ(report.counter_parity_fields_checked, 1u);
  EXPECT_EQ(report.counter_parity_exact, 0u);
  EXPECT_EQ(report.counter_underaccounting_observed, 1u);
}

TEST(WaitcheckTest, Gfx1250CounterParityMatchesXcnt) {
  std::vector<uint32_t> program;
  append_inst(program, global_load_b32(0));
  append_inst(program, s_wait_xcnt_0());
  append_inst(program, v_mov_b32(8, 10));

  WaitcheckOptions options;
  options.check_counter_parity = true;
  TestCodeObject code_object(program);
  auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_CDNA5, options);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.passed()) << diagnostic_summary(report);
  EXPECT_EQ(report.counter_parity_wait_groups, 1u);
  EXPECT_EQ(report.counter_parity_fields_checked, 1u);
  EXPECT_EQ(report.counter_parity_exact, 1u);
  EXPECT_EQ(report.counter_underaccounting_observed, 0u);
}

TEST(WaitcheckTest, Gfx1250CounterParityTracksTheOriginalTranslationThroughLaterMemory) {
  for (const bool switch_class : {false, true}) {
    std::vector<uint32_t> words;
    append_inst(words, global_load_b32(0, 8));
    append_inst(words, global_load_b32(1, 9));
    append_inst(words, s_wait_xcnt(0));
    if (switch_class)
      append_inst(words, s_load_b32(20, 0));
    else
      append_inst(words, global_load_b32(2, 10));
    append_inst(words, v_mov_b32(8, 11));
    append_inst(words, s_endpgm());
    TestCodeObject object(words);
    WaitcheckOptions options;
    options.check_counter_parity = true;
    const auto report = analyze_waitcnts(object, ROCJITSU_CODE_ARCH_CDNA5, options);
    ASSERT_TRUE(report.supported && report.complete) << report.analysis_error;
    const auto field =
        std::ranges::find_if(report.counter_underaccounting_diagnostics, [](const auto &d) {
          return d.counter == WaitCounterKind::X && d.wait_section_offset == 0x18;
        });
    ASSERT_NE(field, report.counter_underaccounting_diagnostics.end())
        << diagnostic_summary(report);
    EXPECT_EQ(field->has_required_dependency, !switch_class);
    if (!switch_class)
      EXPECT_EQ(field->required_count, 1u);
  }
}

TEST(WaitcheckTest, Gfx1250GroupedCompletionCanDischargeAnOrdinaryXcntDependency) {
  for (const uint32_t completion : {2u, 3u}) {
    std::vector<uint32_t> words;
    append_inst(words, global_load_b32(0, 8));
    append_inst(words, global_load_b32(1, 9));
    append_inst(words, global_load_b32(2, 10));
    append_inst(words, s_wait_xcnt(1));
    append_inst(words, sopp(64, completion)); // s_wait_loadcnt
    append_inst(words, v_mov_b32(8, 11));
    append_inst(words, s_endpgm());
    TestCodeObject object(words);
    WaitcheckOptions options;
    options.check_counter_parity = true;
    const auto report = analyze_waitcnts(object, ROCJITSU_CODE_ARCH_CDNA5, options);
    ASSERT_TRUE(report.supported && report.complete) << report.analysis_error;
    const auto field =
        std::ranges::find_if(report.counter_underaccounting_diagnostics, [](const auto &d) {
          return d.counter == WaitCounterKind::X && d.wait_section_offset == 0x24;
        });
    ASSERT_NE(field, report.counter_underaccounting_diagnostics.end())
        << diagnostic_summary(report);
    EXPECT_EQ(field->has_required_dependency, completion == 3);
    if (completion == 3)
      EXPECT_EQ(field->required_count, 2u);
  }
}

TEST(WaitcheckTest, Gfx1250CounterParityStopsReplayAtImplicitXcntDrains) {
  for (const uint32_t intervening : {0xbf86c003u, 0xbfa00000u, 0xbf800000u}) {
    for (const bool wait_for_load : {false, true}) {
      SCOPED_TRACE(::testing::Message()
                   << std::hex << intervening << " wait_for_load=" << wait_for_load);
      std::vector<uint32_t> program = {0xee05007cu, 0, 10}; // global_load_b32 v0, v[10:11], off
      append_inst(program, wait_for_load ? 0xbfc00000u : 0xbfc50000u);
      // s_set_vgpr_msb 0xc003, s_branch to the next instruction, or s_nop.
      append_inst(program, intervening);
      append_inst(program, 0x8c7e007eu); // s_or_b32 exec_lo, exec_lo, s0
      append_inst(program, 0xbfb00000u); // s_endpgm
      WaitcheckOptions options;
      options.check_counter_parity = true;
      TestCodeObject obj(program);
      const auto report = analyze_waitcnts(obj, ROCJITSU_CODE_ARCH_CDNA5, options);
      ASSERT_TRUE(report.passed()) << diagnostic_summary(report);
      const bool drains = intervening != 0xbf800000u;
      EXPECT_EQ(report.counter_parity_exact, drains ? 0u : 1u);
      EXPECT_EQ(report.counter_unmodeled_wait_observed, drains ? 1u : 0u);
      if (wait_for_load && drains) {
        program.pop_back();
        // The bank-changing case selected bank 3 for src0. Restore bank 0
        // so the result use really reads the load's physical destination.
        if (intervening == 0xbf86c003u)
          append_inst(program, s_set_vgpr_msb(0));
        append_inst(program, v_mov_b32(1, 0));
        append_inst(program, 0xbfb00000u);
        TestCodeObject use_result(program);
        const auto actual = analyze_waitcnts(use_result, ROCJITSU_CODE_ARCH_CDNA5, options);
        EXPECT_EQ(actual.counter_parity_exact, 1u) << diagnostic_summary(actual);
      }
    }
  }
}

TEST(WaitcheckTest, Gfx1250CounterParityMatchesXcntBeforeImplicitCmpxExecDef) {
  std::vector<uint32_t> program;
  append_inst(program, global_load_b32(0));
  append_inst(program, s_wait_xcnt_0());
  // v_cmpx_ne_u16_e32 0, v1.h implicitly defines EXEC, but the final encoding
  // has no explicit destination operand.
  append_inst(program, 0x7D7B0280u);

  WaitcheckOptions options;
  options.check_counter_parity = true;
  TestCodeObject code_object(program);
  auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_CDNA5, options);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.passed()) << diagnostic_summary(report);
  EXPECT_EQ(report.counter_parity_wait_groups, 1u);
  EXPECT_EQ(report.counter_parity_fields_checked, 1u);
  EXPECT_EQ(report.counter_parity_exact, 1u);
  EXPECT_EQ(report.counter_underaccounting_observed, 0u);
}

TEST(WaitcheckTest, Gfx1250CounterParityCatalogsConservativeXcntClauseDrain) {
  std::vector<uint32_t> program;
  append_inst(program, global_load_b32(16, 2));
  append_inst(program, global_load_b32(17, 7));
  append_inst(program, global_load_b32(18, 10));
  append_inst(program, global_load_b32(19, 12));
  // LLVM's global-load clause policy emits one all-zero XCNT drain before a
  // staged sequence of partial loadcnt waits and address-register reuses.
  append_inst(program, s_wait_xcnt_0());
  append_inst(program, v_mov_b32(2, 20));

  WaitcheckOptions options;
  options.check_counter_parity = true;
  TestCodeObject code_object(program);
  auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_CDNA5, options);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.passed()) << diagnostic_summary(report);
  EXPECT_EQ(report.counter_parity_fields_checked, 1u);
  EXPECT_EQ(report.counter_parity_exact, 0u);
  EXPECT_EQ(report.counter_underaccounting_observed, 1u);
  ASSERT_EQ(report.counter_underaccounting_diagnostics.size(), 1u);
  EXPECT_TRUE(report.counter_underaccounting_diagnostics[0].has_required_dependency);
  EXPECT_EQ(report.counter_underaccounting_diagnostics[0].emitted_count, 0u);
  EXPECT_EQ(report.counter_underaccounting_diagnostics[0].required_count, 3u);
  EXPECT_EQ(report.counter_underaccounting_diagnostics[0].reg, (RegisterRef{RegClass::VGPR, 2, 1}));
}

TEST(WaitcheckTest, Gfx1250CounterParityCatalogsConservativeScratchStoreXcnt) {
  std::vector<uint32_t> program = {
      // Four scratch_store_b128 instructions from the final gfx1250 corpus.
      0xED0740FCu, 0x20000000u, 0x00002000u, 0xED0740FCu, 0x22000000u, 0x00003000u,
      0xED0740FCu, 0x24000000u, 0x00004000u, 0xED0740FCu, 0x26000000u, 0x00005000u,
  };
  append_inst(program, s_wait_xcnt_0());
  append_inst(program, v_mov_b32(68, 100));

  WaitcheckOptions options;
  options.check_counter_parity = true;
  TestCodeObject code_object(program);
  auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_CDNA5, options);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.passed()) << diagnostic_summary(report);
  EXPECT_EQ(report.counter_parity_fields_checked, 1u);
  EXPECT_EQ(report.counter_parity_exact, 0u);
  EXPECT_EQ(report.counter_underaccounting_observed, 1u);
  ASSERT_EQ(report.counter_underaccounting_diagnostics.size(), 1u);
  EXPECT_TRUE(report.counter_underaccounting_diagnostics[0].has_required_dependency);
  EXPECT_EQ(report.counter_underaccounting_diagnostics[0].emitted_count, 0u);
  EXPECT_EQ(report.counter_underaccounting_diagnostics[0].required_count, 2u);
  EXPECT_EQ(report.counter_underaccounting_diagnostics[0].reg,
            (RegisterRef{RegClass::VGPR, 68, 1}));
}

TEST(WaitcheckTest, Gfx1250CounterParityCatalogsConservativeScratchLoadReplacement) {
  std::vector<uint32_t> program;
  for (uint32_t vdst = 0; vdst != 64; vdst += 4)
    append_inst(program, scratch_load_b128_off(vdst));
  append_inst(program, sopp(64, 8)); // s_wait_loadcnt 8.
  append_inst(program, v_mov_b32(0, 100));

  WaitcheckOptions options;
  options.check_counter_parity = true;
  TestCodeObject code_object(program);
  auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_CDNA5, options);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.passed()) << diagnostic_summary(report);
  EXPECT_EQ(report.counter_parity_fields_checked, 1u);
  EXPECT_EQ(report.counter_parity_exact, 0u);
  EXPECT_EQ(report.counter_underaccounting_observed, 1u);
  ASSERT_EQ(report.counter_underaccounting_diagnostics.size(), 1u);
  EXPECT_TRUE(report.counter_underaccounting_diagnostics[0].has_required_dependency);
  EXPECT_EQ(report.counter_underaccounting_diagnostics[0].emitted_count, 8u);
  EXPECT_EQ(report.counter_underaccounting_diagnostics[0].required_count, 15u);
  EXPECT_EQ(report.counter_underaccounting_diagnostics[0].reg, (RegisterRef{RegClass::VGPR, 0, 1}));
}

TEST(WaitcheckTest, Gfx1250CounterParityMatchesVmVsrc) {
  std::vector<uint32_t> program;
  append_inst(program, global_load_b32(0));
  append_inst(program, s_wait_alu_vm_vsrc_0());
  append_inst(program, v_mov_b32(8, 10));

  WaitcheckOptions options;
  options.check_counter_parity = true;
  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4, options);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.passed()) << diagnostic_summary(report);
  EXPECT_EQ(report.counter_parity_wait_groups, 1u);
  EXPECT_EQ(report.counter_parity_fields_checked, 1u);
  EXPECT_EQ(report.counter_parity_exact, 1u);
  EXPECT_EQ(report.counter_underaccounting_observed, 0u);
}

TEST(WaitcheckTest, Gfx1250CounterParityKeepsPrimaryWaitsWhenAuditingVmVsrc) {
  // Each request consumes a VGPR that is subsequently overwritten. The
  // primary zero wait already releases the source. A partial wait that
  // leaves the only request live still needs VM_VSRC.
  for (uint32_t opcode : {64u, 65u, 70u}) {
    for (bool source_first : {false, true}) {
      for (uint32_t count : {0u, 1u}) {
        SCOPED_TRACE(opcode);
        SCOPED_TRACE(source_first);
        SCOPED_TRACE(count);
        std::vector<uint32_t> program;
        append_inst(program, s_setreg_imm32_b32(hwreg(26, 0, 2), 2));
        constexpr uint32_t source = 100;
        switch (opcode) {
        case 64:
          append_inst(program, global_load_b32(40, source));
          break;
        case 65:
          append_inst(program, global_store_b32(40, source));
          break;
        case 70:
          append_gfx1250_ds_bpermute_b32(program, 40, 2, source);
          break;
        }
        const auto wait = program.size() * sizeof(uint32_t);
        if (source_first)
          append_inst(program, s_wait_alu_vm_vsrc_0());
        append_inst(program, sopp(opcode, count));
        if (!source_first)
          append_inst(program, s_wait_alu_vm_vsrc_0());
        append_inst(program, s_wait_xcnt(0));
        append_inst(program, v_mov_b32(source, 0));

        WaitcheckOptions options;
        options.check_counter_parity = true;
        TestCodeObject object(program);
        const auto report = analyze_waitcnts(object, ROCJITSU_CODE_ARCH_CDNA5, options);
        ASSERT_TRUE(report.supported) << report.analysis_error;
        ASSERT_TRUE(report.complete) << report.analysis_error;
        EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
        const auto vm_source = std::ranges::find_if(
            report.counter_underaccounting_diagnostics, [&](const auto &diagnostic) {
              return diagnostic.wait_section_offset == wait &&
                     diagnostic.counter == WaitCounterKind::VmVsrc;
            });
        if (count == 0) {
          ASSERT_NE(vm_source, report.counter_underaccounting_diagnostics.end());
          EXPECT_FALSE(vm_source->has_required_dependency);
          EXPECT_EQ(vm_source->required_count, 7u);
        } else {
          EXPECT_EQ(vm_source, report.counter_underaccounting_diagnostics.end());
        }
      }
    }
  }
}

TEST(WaitcheckTest, Gfx1250CounterParityCreditsOnlyDefinitelyCompletedVmVsrc) {
  for (bool use_younger : {false, true}) {
    SCOPED_TRACE(use_younger);
    std::vector<uint32_t> program;
    append_inst(program, s_setreg_imm32_b32(hwreg(26, 0, 2), 2));
    append_inst(program, global_load_b32(40, 100));
    append_inst(program, global_load_b32(41, 102));
    const auto wait = program.size() * sizeof(uint32_t);
    append_inst(program, sopp(64, 1)); // Retires the older primary request.
    append_inst(program, s_wait_alu_vm_vsrc_0());
    append_inst(program, s_wait_xcnt(0));
    append_inst(program, v_mov_b32(use_younger ? 102 : 100, 0));

    WaitcheckOptions options;
    options.check_counter_parity = true;
    TestCodeObject object(program);
    const auto report = analyze_waitcnts(object, ROCJITSU_CODE_ARCH_CDNA5, options);
    ASSERT_TRUE(report.supported) << report.analysis_error;
    EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
    const auto vm_source = std::ranges::find_if(
        report.counter_underaccounting_diagnostics, [&](const auto &diagnostic) {
          return diagnostic.wait_section_offset == wait &&
                 diagnostic.counter == WaitCounterKind::VmVsrc;
        });
    if (use_younger) {
      EXPECT_EQ(vm_source, report.counter_underaccounting_diagnostics.end());
    } else {
      ASSERT_NE(vm_source, report.counter_underaccounting_diagnostics.end());
      EXPECT_FALSE(vm_source->has_required_dependency);
    }
  }
}

TEST(WaitcheckTest, Gfx1250CounterParityReplaysLaterPrimarySourceWait) {
  for (const auto [opcode, count] : {std::pair{64u, 0u}, std::pair{65u, 0u}, std::pair{65u, 1u}}) {
    SCOPED_TRACE(opcode);
    SCOPED_TRACE(count);
    std::vector<uint32_t> program;
    append_inst(program, s_setreg_imm32_b32(hwreg(26, 0, 2), 2));
    append_inst(program, s_mov_b32(126, 129)); // exec_lo = 1
    append_inst(program, global_store_b32(11, 0, 124));
    const auto field = program.size() * sizeof(uint32_t);
    append_inst(program, s_wait_xcnt(0));
    append_inst(program, s_wait_alu_vm_vsrc_0());
    program.push_back(build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA5).value());
    append_inst(program, sopp(opcode, count));
    program.push_back(build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA5).value());
    append_inst(program, v_mov_b32(11, 13));
    append_inst(program, s_endpgm());
    WaitcheckOptions options;
    options.check_counter_parity = true;
    const auto report =
        analyze_waitcnts(TestCodeObject(program), ROCJITSU_CODE_ARCH_CDNA5, options);
    ASSERT_TRUE(report.complete) << report.analysis_error;
    EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
    const auto row = std::ranges::find_if(report.counter_underaccounting_diagnostics,
                                          [&](const auto &diagnostic) {
                                            return diagnostic.wait_section_offset == field &&
                                                   diagnostic.counter == WaitCounterKind::VmVsrc;
                                          });
    if (opcode == 65 && count == 0) {
      ASSERT_NE(row, report.counter_underaccounting_diagnostics.end());
      EXPECT_FALSE(row->has_required_dependency);
    } else {
      EXPECT_EQ(row, report.counter_underaccounting_diagnostics.end());
    }
  }
}

TEST(WaitcheckTest, Gfx1250CounterParityRetainsVmVsrcForUnknownFlatRoute) {
  for (bool both_routes : {false, true}) {
    SCOPED_TRACE(both_routes);
    std::vector<uint32_t> program;
    append_inst(program, s_setreg_imm32_b32(hwreg(26, 0, 2), 2));
    append_inst(program, flat_load_b32(40));
    const auto wait = program.size() * sizeof(uint32_t);
    append_inst(program, sopp(both_routes ? 72 : 64, 0));
    append_inst(program, s_wait_alu_vm_vsrc_0());
    append_inst(program, s_wait_xcnt(0));
    append_inst(program, v_mov_b32(8, 0));

    WaitcheckOptions options;
    options.check_counter_parity = true;
    TestCodeObject object(program);
    const auto report = analyze_waitcnts(object, ROCJITSU_CODE_ARCH_CDNA5, options);
    ASSERT_TRUE(report.supported) << report.analysis_error;
    EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
    const auto vm_source = std::ranges::find_if(
        report.counter_underaccounting_diagnostics, [&](const auto &diagnostic) {
          return diagnostic.wait_section_offset == wait &&
                 diagnostic.counter == WaitCounterKind::VmVsrc;
        });
    if (both_routes) {
      ASSERT_NE(vm_source, report.counter_underaccounting_diagnostics.end());
      EXPECT_FALSE(vm_source->has_required_dependency);
    } else {
      EXPECT_EQ(vm_source, report.counter_underaccounting_diagnostics.end());
    }
  }
}

TEST(WaitcheckTest, Gfx1250PartialVmVsrcWaitTracksEachSourceFifo) {
  enum class Later { Mixed, FullDrain, SameKind, Independent, Youngest };
  for (const auto field : {WaitCounterKind::VmVsrc, WaitCounterKind::Load}) {
    SCOPED_TRACE(static_cast<int>(field));
    for (const auto later :
         {Later::Mixed, Later::FullDrain, Later::SameKind, Later::Independent, Later::Youngest}) {
      for (const uint32_t count : {0u, 1u}) {
        SCOPED_TRACE(static_cast<int>(later));
        SCOPED_TRACE(count);
        std::vector<uint32_t> program;
        append_inst(program, s_setreg_imm32_b32(hwreg(26, 0, 2), 2));
        append_inst(program, global_load_b32(40, 100));
        append_inst(program, global_load_b32(41, 102));
        const auto wait = program.size() * sizeof(uint32_t);
        if (field == WaitCounterKind::VmVsrc)
          append_inst(program, s_wait_alu_vm_vsrc(count));
        else
          append_inst(program, sopp(64, count)); // s_wait_loadcnt.
        append_inst(program, s_wait_xcnt(0));
        append_inst(program, v_mov_b32(100, 0));
        if (later == Later::SameKind) {
          append_inst(program, global_load_b32(42, 104));
          append_inst(program, global_load_b32(43, 106));
        } else {
          append_gfx1250_ds_bpermute_b32(program, 42, 2, 104);
          append_gfx1250_ds_bpermute_b32(program, 43, 2, 106);
        }
        append_inst(program, s_wait_alu_vm_vsrc(later == Later::FullDrain ? 0 : 1));
        append_inst(program, s_wait_xcnt(0));
        const uint32_t src = later == Later::Youngest ? 106 : 104;
        append_inst(program, v_mov_b32(later == Later::Independent ? 120 : src, 0));
        WaitcheckOptions options;
        options.check_counter_parity = true;
        TestCodeObject object(program);
        auto report = analyze_waitcnts(object, ROCJITSU_CODE_ARCH_CDNA5, options);
        ASSERT_TRUE(report.supported) << report.analysis_error;
        ASSERT_TRUE(report.complete) << report.analysis_error;
        // The older of two LDS source reads is released at one even while a VMEM
        // source is outstanding. The younger read still requires zero.
        EXPECT_EQ(report.diagnostics.empty(), later != Later::Youngest)
            << diagnostic_summary(report);
        if (count == 0) {
          const bool suggests_one = std::ranges::any_of(
              report.counter_underaccounting_diagnostics, [&](const auto &diagnostic) {
                return diagnostic.wait_section_offset == wait && diagnostic.counter == field &&
                       diagnostic.has_required_dependency && diagnostic.required_count == 1;
              });
          EXPECT_TRUE(suggests_one);
        }
      }
    }
  }
}

TEST(WaitcheckTest, Gfx1250SourceWaitRetainsPerFifoCreditAtAJoin) {
  for (auto field : {WaitCounterKind::Load, WaitCounterKind::VmVsrc}) {
    for (uint32_t count : {0u, 1u}) {
      for (uint32_t later : {0u, 1u}) {
        SCOPED_TRACE(static_cast<int>(field));
        SCOPED_TRACE(count);
        SCOPED_TRACE(later);
        std::vector<uint32_t> code;
        append_inst(code, s_setreg_imm32_b32(hwreg(26, 0, 2), 2));
        append_inst(code, sopp(33, 10)); // s_cbranch_scc0 to the DS arm.
        append_inst(code, global_load_b32(40, 100));
        append_inst(code, global_load_b32(41, 102));
        const auto wait = code.size() * sizeof(uint32_t);
        if (field == WaitCounterKind::Load)
          append_inst(code, sopp(64, count));
        else
          append_inst(code, s_wait_alu_vm_vsrc(count));
        append_inst(code, s_wait_xcnt(0));
        append_inst(code, v_mov_b32(100, 0));
        append_inst(code, sopp(32, 4)); // s_branch to the shared wait.
        append_gfx1250_ds_bpermute_b32(code, 42, 2, 104);
        append_gfx1250_ds_bpermute_b32(code, 43, 2, 106);
        append_inst(code, s_wait_alu_vm_vsrc(later));
        append_inst(code, s_wait_xcnt(0));
        append_inst(code, v_mov_b32(104, 0));
        append_inst(code, s_endpgm());
        WaitcheckOptions options;
        options.check_counter_parity = true;
        TestCodeObject object(code);
        const auto report = analyze_waitcnts(object, ROCJITSU_CODE_ARCH_CDNA5, options);
        ASSERT_TRUE(report.complete) << report.analysis_error;
        EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
        if (count == 0) {
          const bool suggests_one = std::ranges::any_of(
              report.counter_underaccounting_diagnostics, [&](const auto &d) {
                return d.wait_section_offset == wait && d.counter == field &&
                       d.has_required_dependency && d.required_count == 1;
              });
          EXPECT_TRUE(suggests_one);
        }
      }
    }
  }
}

TEST(WaitcheckTest, Gfx1250SourceWaitChecksLowerCountsAcrossAPlateau) {
  for (const auto field : {WaitCounterKind::VmVsrc, WaitCounterKind::Load}) {
    for (const uint32_t later : {0u, 1u}) {
      for (const uint32_t count : {0u, 1u, 2u}) {
        SCOPED_TRACE(static_cast<int>(field));
        SCOPED_TRACE(later);
        SCOPED_TRACE(count);
        std::vector<uint32_t> program;
        append_inst(program, s_setreg_imm32_b32(hwreg(26, 0, 2), 2));
        append_inst(program, global_load_b32(40, 100));
        append_inst(program, global_load_b32(41, 102));
        append_inst(program, global_load_b32(42, 108));
        const auto pc = program.size() * sizeof(uint32_t);
        if (field == WaitCounterKind::Load)
          append_inst(program, sopp(64, count));
        else
          append_inst(program, s_wait_alu_vm_vsrc(count));
        append_inst(program, s_wait_xcnt(0));
        append_inst(program, v_mov_b32(100, 0));
        append_gfx1250_ds_bpermute_b32(program, 52, 2, 104);
        append_gfx1250_ds_bpermute_b32(program, 53, 2, 106);
        append_inst(program, s_wait_alu_vm_vsrc(later));
        append_inst(program, s_wait_xcnt(0));
        append_inst(program, v_mov_b32(104, 0));
        append_inst(program, s_endpgm());
        WaitcheckOptions options;
        options.check_counter_parity = true;
        TestCodeObject object(program);
        const auto report = analyze_waitcnts(object, ROCJITSU_CODE_ARCH_CDNA5, options);
        ASSERT_TRUE(report.complete) << report.analysis_error;
        EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
        if (count == 1) {
          const bool suggests_two =
              std::ranges::any_of(report.counter_underaccounting_diagnostics, [&](const auto &d) {
                return d.wait_section_offset == pc && d.counter == field &&
                       d.has_required_dependency && d.required_count == 2;
              });
          EXPECT_TRUE(suggests_two);
        }
      }
    }
  }
}

TEST(WaitcheckTest, Gfx1250PositiveSourceWaitAccountsForYoungerLdsUses) {
  for (const uint32_t later : {0u, 1u}) {
    SCOPED_TRACE(later);
    std::vector<uint32_t> program;
    append_inst(program, s_setreg_imm32_b32(hwreg(26, 0, 2), 2));
    append_inst(program, global_store_b32(40, 100));
    append_inst(program, global_store_b32(41, 102));
    append_inst(program, global_store_b32(42, 104));
    append_inst(program, global_load_b32(52, 106));
    append_inst(program, s_wait_alu_vm_vsrc(2));
    append_inst(program, s_wait_xcnt(0));
    append_inst(program, v_mov_b32(102, 0));
    append_inst(program, sopp(64, 0)); // s_wait_loadcnt.
    append_gfx1250_ds_bpermute_b32(program, 52, 2, 116);
    append_gfx1250_ds_bpermute_b32(program, 53, 2, 118);
    append_inst(program, s_wait_alu_vm_vsrc(later));
    append_inst(program, s_wait_xcnt(0));
    append_inst(program, v_mov_b32(116, 0));
    append_inst(program, sopp(65, 0)); // s_wait_storecnt.
    append_inst(program, s_endpgm());
    WaitcheckOptions options;
    options.check_counter_parity = true;
    auto image = waitcheck_test::make_gfx_code_object(program, EF_AMDGPU_MACH_AMDGCN_GFX1250, true);
    // The younger returning requests need their full VGPR tuple allocated.
    const uint32_t rsrc1 = 7; // 128 VGPRs, covering v118.
    std::memcpy(image.data() + 0x100 + program.size() * sizeof(uint32_t) + 48, &rsrc1,
                sizeof(rsrc1));
    AmdGpuCodeObject object(image.data(), image.size());
    const auto report = analyze_waitcnts(object, ROCJITSU_CODE_ARCH_CDNA5, options);
    ASSERT_TRUE(report.complete) << report.analysis_error;
    EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
    // Both thresholds release the older LDS source. Only one is exact.
    EXPECT_EQ(report.counter_parity_exact, later == 1 ? 3u : 2u);
  }
}

TEST(WaitcheckTest, Gfx1201CounterParityMatchesKmcnt) {
  std::vector<uint32_t> program;
  append_inst(program, s_load_b32(4, 0));
  append_inst(program, sopp(71, 0)); // s_wait_kmcnt 0.
  append_inst(program, s_mov_b32(8, 4));

  WaitcheckOptions options;
  options.check_counter_parity = true;
  const auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4, options);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.passed()) << diagnostic_summary(report);
  EXPECT_EQ(report.counter_parity_fields_checked, 1u);
  EXPECT_EQ(report.counter_parity_exact, 1u);
  EXPECT_EQ(report.counter_underaccounting_observed, 0u);
}

TEST(WaitcheckTest, Gfx1201CounterParityMatchesSamplecnt) {
  std::vector<uint32_t> program;
  append_inst(program, image_sample(4));
  append_inst(program, sopp(66, 0)); // s_wait_samplecnt 0.
  append_inst(program, v_mov_b32(8, 4));

  WaitcheckOptions options;
  options.check_counter_parity = true;
  const auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4, options);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.passed()) << diagnostic_summary(report);
  EXPECT_EQ(report.counter_parity_fields_checked, 1u);
  EXPECT_EQ(report.counter_parity_exact, 1u);
  EXPECT_EQ(report.counter_underaccounting_observed, 0u);
}

TEST(WaitcheckTest, Gfx1201CounterParityMatchesBvhcnt) {
  std::vector<uint32_t> program;
  append_inst(program, image_bvh_intersect_ray(4));
  append_inst(program, sopp(67, 0)); // s_wait_bvhcnt 0.
  append_inst(program, v_mov_b32(8, 4));

  WaitcheckOptions options;
  options.check_counter_parity = true;
  const auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4, options);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.passed()) << diagnostic_summary(report);
  EXPECT_EQ(report.counter_parity_fields_checked, 1u);
  EXPECT_EQ(report.counter_parity_exact, 1u);
  EXPECT_EQ(report.counter_underaccounting_observed, 0u);
}

TEST(WaitcheckTest, Gfx1201CounterParityMatchesExpcnt) {
  std::vector<uint32_t> program;
  append_inst(program, export_mrt0_v0());
  append_inst(program, sopp(68, 0)); // s_wait_expcnt 0.
  append_inst(program, s_mov_b64_exec_from_s0());

  WaitcheckOptions options;
  options.check_counter_parity = true;
  const auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4, options);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.passed()) << diagnostic_summary(report);
  EXPECT_EQ(report.counter_parity_fields_checked, 1u);
  EXPECT_EQ(report.counter_parity_exact, 1u);
  EXPECT_EQ(report.counter_underaccounting_observed, 0u);
}

TEST(WaitcheckTest, Gfx1250CounterParityMatchesAsynccnt) {
  std::vector<uint32_t> program;
  append_global_load_async_to_lds_b32(program);
  append_inst(program, s_wait_asynccnt(0));
  append_inst(program, ds_load_b32(4, 12));

  WaitcheckOptions options;
  options.check_counter_parity = true;
  TestCodeObject code_object(program);
  const auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_CDNA5, options);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.passed()) << diagnostic_summary(report);
  EXPECT_EQ(report.counter_parity_fields_checked, 1u);
  EXPECT_EQ(report.counter_parity_exact, 1u);
  EXPECT_EQ(report.counter_underaccounting_observed, 0u);
}

TEST(WaitcheckTest, Gfx1250CounterParityMatchesTensorcnt) {
  std::vector<uint32_t> program;
  append_tensor_load_to_lds(program);
  append_inst(program, s_wait_tensorcnt(0));
  append_inst(program, ds_load_b32(4, 12));

  WaitcheckOptions options;
  options.check_counter_parity = true;
  TestCodeObject code_object(program);
  const auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_CDNA5, options);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.passed()) << diagnostic_summary(report);
  EXPECT_EQ(report.counter_parity_fields_checked, 1u);
  EXPECT_EQ(report.counter_parity_exact, 1u);
  EXPECT_EQ(report.counter_underaccounting_observed, 0u);
}

TEST(WaitcheckTest, Gfx1201CounterParityMatchesEmbeddedVmVsrc) {
  std::vector<uint32_t> program;
  append_inst(program, global_load_b32(0)); // Uses v8 as the vector offset.
  append_inst(program, ds_param_load_with_waits(8, 15, 0));

  WaitcheckOptions options;
  options.check_counter_parity = true;
  const auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4, options);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.passed()) << diagnostic_summary(report);
  EXPECT_EQ(report.counter_parity_fields_checked, 1u);
  EXPECT_EQ(report.counter_parity_exact, 1u);
  EXPECT_EQ(report.counter_underaccounting_observed, 0u);
}

TEST(WaitcheckTest, Gfx1201CounterParityMatchesEmbeddedVaVdst) {
  std::vector<uint32_t> program;
  append_inst(program, v_add_f32_e32(0, 257, 1)); // VALU reads v1.
  append_inst(program, ds_param_load_with_waits(1, 0, 1));

  WaitcheckOptions options;
  options.check_counter_parity = true;
  const auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4, options);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.passed()) << diagnostic_summary(report);
  EXPECT_EQ(report.counter_parity_fields_checked, 1u);
  EXPECT_EQ(report.counter_parity_exact, 1u);
  EXPECT_EQ(report.counter_underaccounting_observed, 0u);
}

TEST(WaitcheckTest, Gfx1201CounterParityMatchesEmbeddedVinterpExpcnt) {
  std::vector<uint32_t> program;
  append_inst(program, ds_direct_load(1));
  append_inst(program, v_interp_p10_f32(2, 1, 0));

  WaitcheckOptions options;
  options.check_counter_parity = true;
  const auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4, options);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.passed()) << diagnostic_summary(report);
  EXPECT_EQ(report.counter_parity_fields_checked, 1u);
  EXPECT_EQ(report.counter_parity_exact, 1u);
  EXPECT_EQ(report.counter_underaccounting_observed, 0u);
}

TEST(WaitcheckTest, Gfx1201CounterParityAttributesImpliedVmVsrcToLoadcnt) {
  std::vector<uint32_t> program;
  append_inst(program, global_load_b32(0)); // Uses v8 as the vector offset.
  append_inst(program, sopp(64, 0));        // s_wait_loadcnt 0 also waits VM_VSRC.
  append_inst(program, v_mov_b32(8, 10));

  WaitcheckOptions options;
  options.check_counter_parity = true;
  const auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4, options);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.passed()) << diagnostic_summary(report);
  EXPECT_EQ(report.counter_parity_fields_checked, 1u);
  EXPECT_EQ(report.counter_parity_exact, 1u);
  EXPECT_EQ(report.counter_underaccounting_observed, 0u);
}

TEST(WaitcheckTest, Gfx1250CounterParityAttributesImpliedXcntToLoadcnt) {
  std::vector<uint32_t> program;
  append_inst(program, global_load_b32(0)); // Uses s[0:1] as the scalar address.
  append_inst(program, sopp(64, 0));        // s_wait_loadcnt 0 also waits VMEM X_CNT.
  append_inst(program, s_mov_b32(0, 128));

  WaitcheckOptions options;
  options.check_counter_parity = true;
  TestCodeObject code_object(program);
  const auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_CDNA5, options);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.passed()) << diagnostic_summary(report);
  EXPECT_EQ(report.counter_parity_fields_checked, 1u);
  EXPECT_EQ(report.counter_parity_exact, 1u);
  EXPECT_EQ(report.counter_underaccounting_observed, 0u);
}

TEST(WaitcheckTest, Gfx1250CounterParityAttributesImpliedXcntToKmcnt) {
  std::vector<uint32_t> program;
  append_inst(program, s_load_b32(4, 0)); // Uses s[0:1] as the scalar address.
  append_inst(program, sopp(71, 0));      // s_wait_kmcnt 0 also waits SMEM X_CNT.
  append_inst(program, s_mov_b32(0, 128));

  WaitcheckOptions options;
  options.check_counter_parity = true;
  TestCodeObject code_object(program);
  const auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_CDNA5, options);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.passed()) << diagnostic_summary(report);
  EXPECT_EQ(report.counter_parity_fields_checked, 1u);
  EXPECT_EQ(report.counter_parity_exact, 1u);
  EXPECT_EQ(report.counter_underaccounting_observed, 0u);
}

TEST(WaitcheckTest, Gfx1201CounterParityMatchesStoreDataWar) {
  constexpr uint32_t kHwRegWaveSchedMode = 26;
  std::vector<uint32_t> program;
  append_inst(program, s_setreg_imm32_b32(hwreg(kHwRegWaveSchedMode, 0, 2), 2));
  append_inst(program, global_store_b32(10));
  append_inst(program, s_wait_alu_vm_vsrc_0());
  append_inst(program, v_mov_b32(10, 0));

  WaitcheckOptions options;
  options.check_counter_parity = true;
  TestCodeObject code_object(program);
  const auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_RDNA4, options);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.passed()) << diagnostic_summary(report);
  EXPECT_EQ(report.counter_parity_fields_checked, 1u);
  EXPECT_EQ(report.counter_parity_exact, 1u);
  EXPECT_EQ(report.counter_underaccounting_observed, 0u);
}

TEST(WaitcheckTest, Gfx1201CounterParityMatchesUncertainLoadOrderAtCfgMerge) {
  std::vector<uint32_t> program;
  append_inst(program, sopp(33, 7)); // s_cbranch_scc0 to the else path.
  append_inst(program, global_load_b32(0));
  append_inst(program, global_load_b32(1));
  append_inst(program, sopp(32, 6)); // s_branch to the join.
  append_inst(program, global_load_b32(1));
  append_inst(program, global_load_b32(0));
  append_inst(program, sopp(64, 0)); // s_wait_loadcnt 0.
  append_inst(program, v_mov_b32(2, 0));

  WaitcheckOptions options;
  options.check_counter_parity = true;
  TestCodeObject code_object(program);
  const auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_RDNA4, options);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.passed()) << diagnostic_summary(report);
  EXPECT_EQ(report.counter_parity_fields_checked, 1u);
  EXPECT_EQ(report.counter_parity_exact, 1u);
  EXPECT_EQ(report.counter_underaccounting_observed, 0u);
}

TEST(WaitcheckTest, Gfx1201CounterParityMatchesLoopCarriedDsDependency) {
  std::vector<uint32_t> program;
  append_inst(program, sopp(70, 0)); // Loop header: s_wait_dscnt 0.
  append_inst(program, v_mov_b32(1, 0));
  append_inst(program, ds_load_b32(0, 4));
  append_inst(program, sopp(34, static_cast<uint16_t>(-5))); // Back to the loop header.
  append_inst(program, sopp(48, 0));                         // s_endpgm on the fallthrough path.

  WaitcheckOptions options;
  options.check_counter_parity = true;
  TestCodeObject code_object(program);
  const auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_RDNA4, options);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.passed()) << diagnostic_summary(report);
  EXPECT_EQ(report.counter_parity_fields_checked, 1u);
  EXPECT_EQ(report.counter_parity_exact, 1u);
  EXPECT_EQ(report.counter_underaccounting_observed, 0u);
}

TEST(WaitcheckTest, Gfx1201CounterParityFollowsOriginalLoadsPastPartialWait) {
  for (uint32_t first_source : {4u, 6u}) {
    SCOPED_TRACE(first_source);
    for (const uint32_t second_count : {0u, 1u}) {
      SCOPED_TRACE(second_count);
      std::vector<uint32_t> program;
      append_inst(program, global_load_b32(4));
      append_inst(program, global_load_b32(5));
      append_inst(program, sopp(64, 0));                // s_wait_loadcnt 0.
      append_inst(program, v_mov_b32(7, first_source)); // v6 is independent of both loads.
      // A different wait group follows. Its partial value does not protect v5.
      append_inst(program, sopp(64, second_count));
      append_inst(program, v_mov_b32(8, 5));

      WaitcheckOptions options;
      options.check_counter_parity = true;
      TestCodeObject obj(program);
      const auto report = analyze_waitcnts(obj, ROCJITSU_CODE_ARCH_RDNA4, options);
      ASSERT_TRUE(report.passed()) << diagnostic_summary(report);
      const auto &fields = report.counter_underaccounting_diagnostics;
      const auto first = std::ranges::find_if(fields, [](const auto &field) {
        return field.wait_section_offset == 2 * sizeof(global_load_b32(4));
      });
      if (second_count == 0) {
        ASSERT_NE(first, fields.end());
        EXPECT_EQ(first->has_required_dependency, first_source == 4u);
        if (first_source == 4u)
          EXPECT_EQ(first->required_count, 1u);
      } else {
        EXPECT_EQ(first, fields.end());
        EXPECT_EQ(report.counter_parity_exact, 1u);
        EXPECT_EQ(report.counter_unmodeled_wait_observed, 1u);
      }
    }
  }
}

TEST(WaitcheckTest, Gfx1250CounterParityUsesBankAtGuardedConsumer) {
  for (uint32_t consumer_bank : {0u, 1u, 2u}) {
    SCOPED_TRACE(consumer_bank);
    std::vector<uint32_t> program;
    append_inst(program, global_load_b32(4)); // v4 has one younger load.
    append_inst(program, s_set_vgpr_msb(vgpr_msb_mode(0, 0, 0, 1)));
    append_inst(program, global_load_b32(4)); // v260 is the youngest.
    append_inst(program, s_set_vgpr_msb(0));
    append_inst(program, sopp(64, 0)); // s_wait_loadcnt 0.
    append_inst(program, s_set_vgpr_msb(vgpr_msb_mode(consumer_bank, 0, 0, 0)));
    append_inst(program, v_mov_b32(8, 4));
    append_inst(program, s_endpgm());

    WaitcheckOptions options;
    options.check_counter_parity = true;
    TestCodeObject code_object(program);
    const auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_CDNA5, options);
    ASSERT_TRUE(report.passed()) << diagnostic_summary(report);
    EXPECT_EQ(report.counter_parity_fields_checked, 1u);
    if (consumer_bank == 1) {
      EXPECT_EQ(report.counter_parity_exact, 1u);
      EXPECT_EQ(report.counter_underaccounting_observed, 0u);
    } else {
      ASSERT_EQ(report.counter_underaccounting_diagnostics.size(), 1u);
      const auto &diag = report.counter_underaccounting_diagnostics.front();
      EXPECT_EQ(diag.has_required_dependency, consumer_bank == 0);
      if (consumer_bank == 0) {
        EXPECT_EQ(diag.required_count, 1u);
        EXPECT_EQ(diag.reg, (RegisterRef{RegClass::VGPR, 4, 1}));
      }
    }
  }
}

TEST(WaitcheckTest, Gfx1201CounterParityAttributesDsDrainToSplitBarrierSignal) {
  std::vector<uint32_t> program;
  append_inst(program, ds_load_b32(0, 12));
  append_inst(program, sopp(70, 0));     // s_wait_dscnt 0.
  append_inst(program, v_mov_b32(8, 9)); // Independent work scheduled before the signal.
  program.push_back(0xBE804EC1u);        // s_barrier_signal -1.
  program.push_back(0xBF810000u);        // s_endpgm.

  WaitcheckOptions options;
  options.check_counter_parity = true;
  TestCodeObject code_object(program);
  const auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_RDNA4, options);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.passed()) << diagnostic_summary(report);
  EXPECT_EQ(report.counter_parity_fields_checked, 1u);
  EXPECT_EQ(report.counter_parity_exact, 1u);
  EXPECT_EQ(report.counter_underaccounting_observed, 0u);
}

TEST(WaitcheckTest, Gfx1201CounterParityAttributesAcquireFenceWaitBeforeGlobalInv) {
  std::vector<uint32_t> program;
  append_inst(program, global_load_b32(1));
  append_inst(program, ds_load_b32(0, 12));
  append_inst(program, ds_store_b32(13, 2));
  append_inst(program, sopp(72, 0));     // s_wait_loadcnt_dscnt 0.
  append_inst(program, v_mov_b32(8, 9)); // Independent work scheduled before the invalidate.
  append_inst(program, global_inv());
  append_inst(program, v_mov_b32(3, 1));
  append_inst(program, ds_load_b32(0, 12));
  program.push_back(0xBF810000u); // s_endpgm.

  WaitcheckOptions options;
  options.check_counter_parity = true;
  TestCodeObject code_object(program);
  const auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_RDNA4, options);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.passed()) << diagnostic_summary(report);
  EXPECT_EQ(report.counter_parity_fields_checked, 2u);
  EXPECT_EQ(report.counter_parity_exact, 2u);
  EXPECT_EQ(report.counter_underaccounting_observed, 0u);
}

TEST(WaitcheckTest, Gfx1250AcquireFenceKeepsZeroLoadCountAcrossIndependentWork) {
  for (bool separated : {false, true}) {
    std::vector<uint32_t> program;
    append_inst(program, global_load_b32(9, 0));
    append_inst(program, global_load_b32(10, 0));
    append_inst(program, sopp(64, 0)); // s_wait_loadcnt 0
    if (separated)
      append_inst(program, v_mov_b32(16, 17));
    auto invalidate = global_inv();
    invalidate[1] = 2u << 18; // SCOPE_DEV
    append_inst(program, invalidate);
    append_inst(program, v_mov_b32(12, 9));
    program.push_back(0xBFB00000u);

    WaitcheckOptions options;
    options.check_counter_parity = true;
    const auto report = analyze_waitcnts(TestCodeObject(program), ROCJITSU_CODE_ARCH_CDNA5, options);
    ASSERT_TRUE(report.passed()) << diagnostic_summary(report);
    EXPECT_EQ(report.counter_parity_fields_checked, 1u);
    EXPECT_EQ(report.counter_parity_exact, 1u);
    EXPECT_EQ(report.counter_underaccounting_observed, 0u);
  }
}

TEST(WaitcheckTest, Gfx1250ReleaseWritebackDistinguishesWritesFromReads) {
  const auto run = [](rj_code_arch_t arch, unsigned scope, std::vector<uint32_t> program,
                      unsigned wanted) {
    append_inst(program, sopp(64, 0)); // s_wait_loadcnt 0
    append_inst(program, sopp(65, 0)); // s_wait_storecnt 0
    append_inst(program, sopp(70, 0)); // s_wait_dscnt 0
    append_inst(program, v_mov_b32(20, 21));
    auto wb = global_wb();
    wb[1] |= scope << 18;
    append_inst(program, wb);
    append_inst(program, s_endpgm());
    WaitcheckOptions options;
    options.check_counter_parity = true;
    TestCodeObject object(program);
    const auto report = analyze_waitcnts(object, arch, options);
    EXPECT_TRUE(report.supported && report.complete) << report.analysis_error;
    EXPECT_EQ(report.counter_parity_fields_checked, 3u);
    EXPECT_EQ(report.counter_parity_exact, wanted) << diagnostic_summary(report);
    EXPECT_EQ(report.counter_unmodeled_wait_observed, 3u - wanted);
  };
  std::vector<uint32_t> store, returning_atomic, nonreturning_atomic, load, lds;
  append_inst(store, global_store_b32(16));
  append_inst(returning_atomic, global_atomic_add_u32(4, 8, 124, true));
  append_inst(nonreturning_atomic, global_atomic_add_u32(4));
  append_inst(load, global_load_b32(4));
  append_inst(lds, ds_store_b32(24, 28));
  for (unsigned scope : {2u, 3u}) {
    run(ROCJITSU_CODE_ARCH_CDNA5, scope, store, 1);
    run(ROCJITSU_CODE_ARCH_CDNA5, scope, returning_atomic, 1);
    run(ROCJITSU_CODE_ARCH_CDNA5, scope, nonreturning_atomic, 1);
    run(ROCJITSU_CODE_ARCH_CDNA5, scope, load, 0);
    run(ROCJITSU_CODE_ARCH_CDNA5, scope, lds, 0);
    run(ROCJITSU_CODE_ARCH_CDNA5, scope, {0xed0680fcu, 0u, 0x0002a800u}, 0); // scratch store
    run(ROCJITSU_CODE_ARCH_CDNA5, scope, {}, 0);
  }
  run(ROCJITSU_CODE_ARCH_CDNA5, 1, store, 0);
  run(ROCJITSU_CODE_ARCH_RDNA4, 3, store, 0);
}

TEST(WaitcheckTest, Gfx1250AcquireInvalidateWaitOrdersFollowingGlobalRead) {
  const auto run = [](rj_code_arch_t arch, unsigned scope, bool store, bool earlier_drain,
                      bool omit_invalidate, unsigned expected_unmodeled) {
    std::vector<uint32_t> program;
    if (!omit_invalidate) {
      auto inv = global_inv();
      inv[1] |= scope << 18;
      append_inst(program, inv);
    }
    if (earlier_drain) {
      append_inst(program, sopp(64, 0));
      append_inst(program, v_mov_b32(0, 1));
    }
    append_inst(program, sopp(64, 0));
    append_inst(program, v_mov_b32(1, 0));
    if (store)
      append_inst(program, global_store_b32(16));
    else
      append_inst(program, global_load_b32(16));
    append_inst(program, s_endpgm());
    WaitcheckOptions options;
    options.check_counter_parity = true;
    TestCodeObject object(program);
    const auto report = analyze_waitcnts(object, arch, options);
    EXPECT_TRUE(report.supported && report.complete) << report.analysis_error;
    EXPECT_EQ(report.counter_unmodeled_wait_observed, expected_unmodeled)
        << diagnostic_summary(report);
  };
  run(ROCJITSU_CODE_ARCH_CDNA5, 2, false, false, false, 0);
  run(ROCJITSU_CODE_ARCH_CDNA5, 3, false, false, false, 0);
  run(ROCJITSU_CODE_ARCH_CDNA5, 3, true, false, false, 1);
  run(ROCJITSU_CODE_ARCH_CDNA5, 3, false, false, true, 1);
  run(ROCJITSU_CODE_ARCH_CDNA5, 1, false, false, false, 1);
  run(ROCJITSU_CODE_ARCH_CDNA5, 3, false, true, false, 2);
  run(ROCJITSU_CODE_ARCH_RDNA4, 3, false, false, false, 1);
}

TEST(WaitcheckTest, Gfx1201CounterParityHandlesConsecutiveWaitsAcrossCfgLabel) {
  std::vector<uint32_t> program;
  append_inst(program, sopp(33, 6)); // Branch target is the second wait below.
  append_inst(program, global_load_b32(0));
  append_inst(program, ds_load_b32(1, 12));
  append_inst(program, sopp(70, 0)); // s_wait_dscnt 0.
  append_inst(program, sopp(64, 0)); // s_wait_loadcnt 0 at a CFG label.
  append_inst(program, v_mov_b32(2, 0));
  append_inst(program, v_mov_b32(3, 1));

  WaitcheckOptions options;
  options.check_counter_parity = true;
  TestCodeObject code_object(program);
  const auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_RDNA4, options);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.passed()) << diagnostic_summary(report);
  EXPECT_EQ(report.counter_parity_fields_checked, 2u);
  EXPECT_EQ(report.counter_parity_exact, 2u);
  EXPECT_EQ(report.counter_underaccounting_observed, 0u);
  EXPECT_EQ(report.counter_parity_indeterminate_groups, 0u);
}

TEST(WaitcheckTest, Gfx1150CounterParityMatchesPartialLoopCarriedDsDependency) {
  std::vector<uint32_t> program;
  append_gfx1150_s_waitcnt_lgkmcnt_1(program); // 0x00 loop header.
  program.push_back(0x7E040300u);              // 0x04: v_mov_b32_e32 v2, v0.
  program.push_back(0xD8D80000u);              // 0x08: ds_load_b32 v0, v4.
  program.push_back(0x00000004u);
  program.push_back(0xD8340000u); // 0x10: ds_store_b32 v4, v2.
  program.push_back(0x00000204u);
  program.push_back(0xBFA0FFF9u); // 0x18: s_branch to loop header.

  const auto image =
      rocjitsu::waitcheck_test::make_gfx_code_object(program, EF_AMDGPU_MACH_AMDGCN_GFX1150);
  AmdGpuCodeObject code_object(image.data(), image.size());
  WaitcheckOptions options;
  options.check_counter_parity = true;
  const auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_RDNA3_5, options);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.passed()) << diagnostic_summary(report);
  EXPECT_EQ(report.counter_parity_fields_checked, 1u);
  EXPECT_EQ(report.counter_parity_exact, 1u);
  EXPECT_EQ(report.counter_underaccounting_observed, 0u);
  if (!report.counter_underaccounting_diagnostics.empty()) {
    ADD_FAILURE() << report.counter_underaccounting_diagnostics.front().message << " from "
                  << report.counter_underaccounting_diagnostics.front().producer_instruction;
  }
}

// Reduced final ISA from a gfx1150 PyTorch CK kernel.  The compiler-emitted
// vmcnt(2) is one step stronger than the vmcnt(3) required before the VOPD
// reads v[24:25].  Keep the encoded instruction stream here so parity remains
// anchored to the artifact rather than to an assembler or a synthetic opcode.
TEST(WaitcheckTest, Gfx1150CounterParityCapturesConservativeScratchClauseWait) {
  const std::vector<uint32_t> program = {
      0xBF850003u,              // s_clause 0x3.
      0xDC5D0640u, 0x187C0000u, // scratch_load_b128 v[24:27].
      0xDC5D0650u, 0x1C7C0000u, // scratch_load_b128 v[28:31].
      0xDC5D0040u, 0x007C0000u, // scratch_load_b128 v[0:3].
      0xDC5D0050u, 0x047C0000u, // scratch_load_b128 v[4:7].
      0xBF890BF7u,              // s_waitcnt vmcnt(2).
      0xCA100118u, 0x10100119u, // v_dual_mov_b32 v16, v24 :: v17, v25.
      0xBF810000u,              // s_endpgm.
  };

  const auto image =
      rocjitsu::waitcheck_test::make_gfx_code_object(program, EF_AMDGPU_MACH_AMDGCN_GFX1150);
  AmdGpuCodeObject code_object(image.data(), image.size());
  WaitcheckOptions options;
  options.check_counter_parity = true;
  const auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_RDNA3_5, options);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.passed()) << diagnostic_summary(report);
  EXPECT_EQ(report.counter_parity_fields_checked, 1u);
  EXPECT_EQ(report.counter_parity_exact, 0u);
  EXPECT_EQ(report.counter_underaccounting_observed, 1u);
  EXPECT_EQ(report.counter_unmodeled_wait_observed, 0u);
  ASSERT_EQ(report.counter_underaccounting_diagnostics.size(), 1u);
  const auto &diag = report.counter_underaccounting_diagnostics.front();
  EXPECT_EQ(diag.counter, WaitCounterKind::Load);
  EXPECT_EQ(diag.emitted_count, 2u);
  EXPECT_EQ(diag.required_count, 3u);
  EXPECT_TRUE(diag.has_required_dependency);
  EXPECT_EQ(diag.access, WaitcheckAccessKind::Use);
  EXPECT_EQ(diag.reg, (RegisterRef{RegClass::VGPR, 24, 1}));
  EXPECT_EQ(diag.wait_section_offset, 36u);
  EXPECT_EQ(diag.consumer_section_offset, 40u);
  EXPECT_EQ(diag.producer_section_offset, 4u);
}

TEST(WaitcheckTest, Gfx1150CounterParityChecksTheGuardOfAConditionalScratchResult) {
  for (const bool bypass_producer_consumer : {false, true}) {
    SCOPED_TRACE(bypass_producer_consumer);
    const std::vector<uint32_t> words{
        0xbe8300c1,              // s_mov_b32 s3, -1
        0xbf068004,              // s_cmp_eq_u32 s4, 0
        0xbfa20007,              // s_cbranch_scc1 join
        0xdc5d004c, 0x297c0000, // scratch_load_b128 v[41:44], off, off offset:76
        0xdc51008c, 0x527c0000, // scratch_load_b32 v82, off, off offset:140
        bypass_producer_consumer ? 0xbe830080u : 0xbe8300c1u, // s_mov_b32 s3, 0 or -1
        0xbfa00001,              // s_branch join
        0xbf800000,              // unreachable s_nop 0
        0x8b6a037e,              // join: s_and_b32 vcc_lo, exec_lo, s3
        0xbfa30004,              // s_cbranch_vccz done
        0xbf8903f7,              // s_waitcnt vmcnt(0)
        0xbf870193,              // s_delay_alu
        0xca100080, 0x2c580080, // v_dual_mov_b32 v44, 0 :: v89, 0
        0xbfb00000,              // done: s_endpgm
    };
    const auto elf =
        waitcheck_test::make_gfx_code_object(words, EF_AMDGPU_MACH_AMDGCN_GFX1150, true);
    AmdGpuCodeObject object(elf.data(), elf.size());
    WaitcheckOptions options;
    options.check_counter_parity = true;
    const auto report = analyze_waitcnts_for_kernel(object, ROCJITSU_CODE_ARCH_RDNA3_5, 0, options);
    ASSERT_TRUE(report.supported && report.complete) << report.analysis_error;
    EXPECT_TRUE(report.passed()) << diagnostic_summary(report);
    const bool has_scratch_dependency =
        std::ranges::any_of(report.counter_underaccounting_diagnostics, [](const auto &diagnostic) {
          return diagnostic.counter == WaitCounterKind::Load && diagnostic.has_required_dependency &&
                 diagnostic.producer_section_offset == 12 && diagnostic.reg.index == 44 &&
                 diagnostic.required_count == 1;
        });
    EXPECT_EQ(has_scratch_dependency, !bypass_producer_consumer) << diagnostic_summary(report);
  }
}

TEST(WaitcheckTest, Gfx11ChecksGuardedScratchInOrdinaryAndParity) {
  for (const auto &[arch, machine] :
       {std::pair{ROCJITSU_CODE_ARCH_RDNA3, EF_AMDGPU_MACH_AMDGCN_GFX1100},
        std::pair{ROCJITSU_CODE_ARCH_RDNA3_5, EF_AMDGPU_MACH_AMDGCN_GFX1150}}) {
    for (const bool split_wait_block : {false, true}) {
      for (const bool open_guard : {false, true}) {
        for (const uint32_t threshold : {0u, 63u}) {
          SCOPED_TRACE(testing::Message() << machine << '/' << split_wait_block << '/' << open_guard
                                         << '/' << threshold);
          std::vector<uint32_t> words{
              0xbf068004,              // s_cmp_eq_u32 s4, 0
              0xbfa20006,              // s_cbranch_scc1 consumer
              0xdc51004c, 0x2a7c0000, // scratch_load_b32 v42, off, off offset:76
              open_guard ? 0xbe8300c1u : 0xbe830080u, // s_mov_b32 s3, -1 or 0
          };
          if (split_wait_block) {
            ++words[1];
            words.push_back(0xbfa00000); // s_branch wait
          }
          const uint64_t wait = words.size() * 4;
          words.insert(words.end(), {0xbf8903f7u | (threshold << 10), // s_waitcnt vmcnt(N)
                                     0x8b6a037e,                    // s_and_b32 vcc_lo, exec_lo, s3
                                     0xbfa30001,                    // s_cbranch_vccz end
                                     0x7e02032a,                    // v_mov_b32_e32 v1, v42
                                     0xbfb00000});                  // end: s_endpgm
          const auto elf = waitcheck_test::make_gfx_code_object(words, machine, true);
          AmdGpuCodeObject object(elf.data(), elf.size());
          WaitcheckOptions options;
          options.check_counter_parity = true;
          const auto report = analyze_waitcnts_for_kernel(object, arch, 0, options);
          ASSERT_TRUE(report.supported && report.complete) << report.analysis_error;
          const bool missing = std::ranges::any_of(report.diagnostics, [](const auto &d) {
            return d.counter == WaitCounterKind::Load && d.reg.index == 42 &&
                   d.producer_section_offset == 8 && d.required_count == 0;
          });
          EXPECT_EQ(missing, open_guard && threshold == 63) << diagnostic_summary(report);
          if (threshold == 0) {
            EXPECT_EQ(report.counter_parity_exact, open_guard ? 1u : 0u);
            const bool unattributed = std::ranges::any_of(
                report.counter_underaccounting_diagnostics, [&](const auto &d) {
                  return d.wait_section_offset == wait && d.counter == WaitCounterKind::Load &&
                         d.required_count == 63 && !d.has_required_dependency;
                });
            EXPECT_EQ(unattributed, !open_guard);
          }
        }
      }
    }
  }
}

TEST(WaitcheckTest, Gfx11GuardedScratchResolvesAdjacentRelativeScalarWrite) {
  enum class M0 { Disjoint, Aliases, Unknown, Mixed, BranchBypassesLiteral };
  for (const auto [arch, machine] :
       {std::pair{ROCJITSU_CODE_ARCH_RDNA3, EF_AMDGPU_MACH_AMDGCN_GFX1100},
        std::pair{ROCJITSU_CODE_ARCH_RDNA3_5, EF_AMDGPU_MACH_AMDGCN_GFX1150}}) {
    for (const auto offset : {M0::Disjoint, M0::Aliases, M0::Unknown, M0::Mixed,
                              M0::BranchBypassesLiteral}) {
      SCOPED_TRACE(arch);
      SCOPED_TRACE(static_cast<unsigned>(offset));
      for (const uint32_t count : {0u, 63u}) {
        SCOPED_TRACE(count);
        std::vector<uint32_t> code{0xbefe0081, 0xbf068004, 0xbfa20000};
        const auto producer = code.size() * sizeof(uint32_t);
        code.insert(code.end(), {0xdc51004c, 0x2a7c0000, 0xbe830080}); // scratch v42, s3=0.
        if (offset == M0::Mixed) {
          code.insert(code.end(), {
                                      0xbf068005, // s_cmp_eq_u32 s5, 0
                                      0xbfa20002, // s_cbranch_scc1 second
                                      0xbefd0080, // s_mov_b32 m0, 0
                                      0xbfa00001, // s_branch join
                                      0xbefd0081, // second: s_mov_b32 m0, 1
                                  });
        } else if (offset == M0::BranchBypassesLiteral) {
          code.insert(code.end(), {
                                      0xbefd0081, // s_mov_b32 m0, 1
                                      0xbf068005, // s_cmp_eq_u32 s5, 0
                                      0xbfa20001, // s_cbranch_scc1 join
                                      0xbefd0080, // s_mov_b32 m0, 0
                                  });
        } else {
          const uint32_t source = offset == M0::Unknown ? 5 : offset == M0::Disjoint ? 128 : 129;
          code.push_back(0xbefd0000u | source); // s_mov_b32 m0, s5/0/1
        }
        code.push_back(0xbe8242c1); // join: s_movreld_b32 s2, -1
        const auto wait = code.size() * sizeof(uint32_t);
        code.insert(code.end(), {0xbf8903f7u | (count << 10), 0x8b6a037e, 0xbfa30002,
                                 0xd760000a, 0x0201012a, 0xbfb00000});
        const auto consumer = wait + 3 * sizeof(uint32_t);
        code[2] |= static_cast<uint32_t>(consumer / sizeof(uint32_t) - 3);
        const auto bytes = waitcheck_test::make_gfx_code_object(code, machine, true);
        const AmdGpuCodeObject object(bytes.data(), bytes.size());
        WaitcheckOptions options;
        options.check_counter_parity = true;
        const auto report = analyze_waitcnts_for_kernel(object, arch, 0, options);
        ASSERT_TRUE(report.supported && report.complete) << report.analysis_error;
        const bool can_warn = offset != M0::Disjoint;
        ASSERT_EQ(report.diagnostics_observed, can_warn && count != 0 ? 1u : 0u)
            << diagnostic_summary(report);
        if (report.diagnostics_observed) {
          const auto &diagnostic = report.diagnostics.front();
          EXPECT_EQ(diagnostic.producer_section_offset, producer);
          EXPECT_EQ(diagnostic.section_offset, consumer);
          EXPECT_EQ(diagnostic.reg, (RegisterRef{RegClass::VGPR, 42, 1}));
          EXPECT_EQ(diagnostic.counter, WaitCounterKind::Load);
          EXPECT_EQ(diagnostic.required_count, 0u);
        }
        if (count == 0 && !can_warn) {
          const auto &rows = report.counter_underaccounting_diagnostics;
          const auto field = std::ranges::find_if(rows, [&](const auto &row) {
            return row.counter == WaitCounterKind::Load && row.wait_section_offset == wait;
          });
          ASSERT_NE(field, rows.end());
          EXPECT_FALSE(field->has_required_dependency);
        }
      }
    }
  }
}

TEST(WaitcheckTest, Gfx11GuardedScratchKeepsScalarFullDrainProof) {
  constexpr uint32_t kPrefixInstructions = 9600;
  const std::vector<uint32_t> prefix = [] {
    std::vector<uint32_t> words{0xbe8300c1, 0xbefe0081}; // s3 = -1, exec_lo = 1.
    std::vector<size_t> to_prefix;
    // The scalar search merges these routes; the lane search keeps each mask.
    for (uint32_t lane = 1; lane < 32; ++lane) {
      words.insert(words.end(), {0xbf068000u | (lane + 128) << 8, // s_cmp_eq_u32 s0, lane.
                                 0xbfa10003,                   // s_cbranch_scc0 next.
                                 0xbefe00ff, 1u << lane});      // s_mov_b32 exec_lo, mask.
      to_prefix.push_back(words.size());
      words.push_back(0xbfa00000); // s_branch prefix.
    }
    for (const size_t branch : to_prefix)
      words[branch] |= static_cast<uint32_t>(words.size() - branch - 1);
    words.insert(words.end(), kPrefixInstructions, 0xbf800000); // s_nop 0.
    return words;
  }();

  for (const auto &[arch, machine] :
       {std::pair{ROCJITSU_CODE_ARCH_RDNA3, EF_AMDGPU_MACH_AMDGCN_GFX1100},
        std::pair{ROCJITSU_CODE_ARCH_RDNA3_5, EF_AMDGPU_MACH_AMDGCN_GFX1150}}) {
    for (const uint32_t threshold : {0u, 63u}) {
      SCOPED_TRACE(testing::Message() << machine << '/' << threshold);
      auto words = prefix;
      words.insert(words.end(), {0xbf068004, 0xbfa10002}); // If s4 == 0, issue the scratch load.
      const uint64_t producer = words.size() * 4;
      words.insert(words.end(), {0xdc510004, 0x087c0000, // scratch_load_b32 v8, off, off offset:4.
                                 0xbf068004, 0xbfa10001, // The same arm executes the full drain.
                                 0xbf8903f7u | (threshold << 10),
                                 0x8b6a037e, 0xbfa30001}); // VCC mask and branch to s_endpgm.
      const uint64_t consumer = words.size() * 4;
      words.insert(words.end(), {0x7e120308, 0xbfb00000}); // v_mov_b32 v9, v8; s_endpgm.
      const auto elf = waitcheck_test::make_gfx_code_object(words, machine, true);
      AmdGpuCodeObject object(elf.data(), elf.size());
      const auto report = analyze_waitcnts_for_kernel(object, arch, 0);
      ASSERT_TRUE(report.supported && report.complete) << report.analysis_error;
      EXPECT_EQ(report.diagnostics_observed, threshold == 63 ? 1u : 0u)
          << diagnostic_summary(report);
      if (threshold == 63 && !report.diagnostics.empty()) {
        const auto &diagnostic = report.diagnostics.front();
        EXPECT_EQ(diagnostic.counter, WaitCounterKind::Load);
        EXPECT_EQ(diagnostic.producer_section_offset, producer);
        EXPECT_EQ(diagnostic.section_offset, consumer);
        EXPECT_EQ(diagnostic.reg, (RegisterRef{RegClass::VGPR, 8, 1}));
      }
    }
  }
}

TEST(WaitcheckTest, Gfx11GuardedScratchKeepsVccAcrossRelativeSgprWrite) {
  for (const auto &[arch, machine] :
       {std::pair{ROCJITSU_CODE_ARCH_RDNA3, EF_AMDGPU_MACH_AMDGCN_GFX1100},
        std::pair{ROCJITSU_CODE_ARCH_RDNA3_5, EF_AMDGPU_MACH_AMDGCN_GFX1150}}) {
    for (const uint32_t offset : {0u, 104u}) {
      for (const uint32_t count : {0u, 63u}) {
        SCOPED_TRACE(testing::Message() << machine << '/' << offset << '/' << count);
        const std::vector<uint32_t> words{
            0xbefe0081, 0xbe8300c1, 0x8b6a037e, 0xbfa3000d, // Guarded scratch CFG.
            0xbf068004, 0xbfa20009,                        // Bypass the scratch load.
            0xdc51004c, 0x2a7c0000,                        // scratch_load_b32 v42.
            0xbe830080, 0xbf8903f7 | (count << 10), 0x8b6a037e, // Capture VCC=0.
            0xbefd00ff, offset, 0xbe8242c1,                // Index s2 or VCC_LO.
            0xbfa30002, 0xd760000a, 0x0201012a, 0xbfb00000 // Branch, read v42, exit.
        };
        const auto elf = waitcheck_test::make_gfx_code_object(words, machine, true);
        AmdGpuCodeObject object(elf.data(), elf.size());
        WaitcheckOptions options;
        options.check_counter_parity = true;
        const auto report = analyze_waitcnts_for_kernel(object, arch, 0, options);
        ASSERT_TRUE(report.supported && report.complete) << report.analysis_error;
        ASSERT_EQ(report.diagnostics_observed, offset == 104 && count == 63 ? 1u : 0u)
            << diagnostic_summary(report);
        if (report.diagnostics_observed) {
          const auto &diagnostic = report.diagnostics.front();
          EXPECT_EQ(diagnostic.producer_section_offset, 0x18u);
          EXPECT_EQ(diagnostic.section_offset, 0x3cu);
          EXPECT_EQ(diagnostic.required_count, 0u);
        }
        if (offset == 0 && count == 0) {
          const auto &rows = report.counter_underaccounting_diagnostics;
          const auto field = std::ranges::find_if(rows, [](const auto &row) {
            return row.counter == WaitCounterKind::Load && row.wait_section_offset == 0x24u;
          });
          ASSERT_NE(field, rows.end());
          EXPECT_FALSE(field->has_required_dependency);
        }
      }
    }
  }
}

TEST(WaitcheckTest, Gfx1150CounterParityRetainsStrongerCyclicWaitDependency) {
  std::vector<uint32_t> program;
  program.push_back(0xBF89FC07u); // 0x00 loop header: s_waitcnt lgkmcnt(0).
  program.push_back(0x7E040300u); // 0x04: v_mov_b32_e32 v2, v0.
  program.push_back(0xD8D80000u); // 0x08: ds_load_b32 v0, v4.
  program.push_back(0x00000004u);
  program.push_back(0xD8340000u); // 0x10: ds_store_b32 v4, v2.
  program.push_back(0x00000204u);
  program.push_back(0xBFA0FFF9u); // 0x18: s_branch to loop header.

  const auto image =
      rocjitsu::waitcheck_test::make_gfx_code_object(program, EF_AMDGPU_MACH_AMDGCN_GFX1150);
  AmdGpuCodeObject code_object(image.data(), image.size());
  WaitcheckOptions options;
  options.check_counter_parity = true;
  const auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_RDNA3_5, options);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.passed()) << diagnostic_summary(report);
  EXPECT_EQ(report.counter_parity_fields_checked, 1u);
  EXPECT_EQ(report.counter_parity_exact, 0u);
  EXPECT_EQ(report.counter_underaccounting_observed, 1u);
  EXPECT_EQ(report.counter_unmodeled_wait_observed, 0u);
  ASSERT_EQ(report.counter_underaccounting_diagnostics.size(), 1u);
  const auto &diagnostic = report.counter_underaccounting_diagnostics.front();
  EXPECT_TRUE(diagnostic.has_required_dependency);
  EXPECT_EQ(diagnostic.producer_section_offset, 8u);
  EXPECT_EQ(diagnostic.consumer_section_offset, 4u);
  EXPECT_EQ(diagnostic.required_count, 1u);
  EXPECT_NE(diagnostic.message.find("loop-carried"), std::string::npos);
}

TEST(WaitcheckTest, Gfx1201CounterParityMatchesSaturatedCounterAge) {
  std::vector<uint32_t> program;
  append_global_loads(program, kOverflowQueueSize, kOverflowBaseVgpr);
  append_inst(program, sopp(64, kOverflowRequiredCount)); // s_wait_loadcnt 39.
  append_inst(program, v_mov_b32(kOverflowConsumerVgpr, kOverflowBaseVgpr));

  WaitcheckOptions options;
  options.check_counter_parity = true;
  TestCodeObject code_object(program);
  const auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_RDNA4, options);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.passed()) << diagnostic_summary(report);
  EXPECT_EQ(report.counter_parity_fields_checked, 1u);
  EXPECT_EQ(report.counter_parity_exact, 1u);
  EXPECT_EQ(report.counter_underaccounting_observed, 0u);
}

TEST(WaitcheckTest, Gfx950CounterParityFollowsResolvedCallReturn) {
  constexpr uint16_t kTargetSreg = 8;
  constexpr uint16_t kReturnSreg = 30;
  constexpr uint16_t kLiteralOperand = 255;
  constexpr uint16_t kInlineInt0 = 128;

  std::vector<uint32_t> program;
  append_gfx950_buffer_load_dword_v0_v8_s0_offen(program);                             // 0x00.
  program.push_back(build_s_getpc_b64(kTargetSreg, ROCJITSU_CODE_ARCH_CDNA4).value()); // 0x08.
  program.push_back(build_s_add_u32(kTargetSreg, kTargetSreg, kLiteralOperand,
                                    ROCJITSU_CODE_ARCH_CDNA4)
                        .value()); // 0x0c.
  program.push_back(24);           // 0x10 -> helper 0x24.
  program.push_back(build_s_addc_u32(kTargetSreg + 1, kTargetSreg + 1, kInlineInt0,
                                     ROCJITSU_CODE_ARCH_CDNA4)
                        .value()); // 0x14.
  program.push_back(
      build_s_swappc_b64(kReturnSreg, kTargetSreg, ROCJITSU_CODE_ARCH_CDNA4).value()); // 0x18.
  append_gfx950_v_mov_b32_v1_v0(program);                                              // 0x1c.
  program.push_back(build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4).value());                 // 0x20.
  append_gfx950_s_waitcnt_vmcnt_0(program); // 0x24 helper.
  program.push_back(build_sop1_encoding(ROCJITSU_CODE_ARCH_CDNA4, 0x1d, 0,
                                        kReturnSreg)
                        .value()); // 0x28 return.

  const auto image =
      rocjitsu::waitcheck_test::make_gfx_code_object(program, EF_AMDGPU_MACH_AMDGCN_GFX950);
  AmdGpuCodeObject code_object(image.data(), image.size());
  WaitcheckOptions options;
  options.check_counter_parity = true;
  const auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_CDNA4, options);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.passed()) << diagnostic_summary(report);
  EXPECT_EQ(report.counter_parity_fields_checked, 1u);
  EXPECT_EQ(report.counter_parity_exact, 1u);
  EXPECT_EQ(report.counter_underaccounting_observed, 0u);
}

TEST(WaitcheckTest, Gfx950SharedReturnWaitRetainsTheDependentCaller) {
  constexpr auto arch = ROCJITSU_CODE_ARCH_CDNA4;
  std::vector<uint32_t> program;
  auto call_helper = [&](uint32_t target_distance) {
    program.push_back(build_s_getpc_b64(8, arch).value());
    program.push_back(build_s_add_u32(8, 8, 255, arch).value());
    program.push_back(target_distance);
    program.push_back(build_s_addc_u32(9, 9, 128, arch).value());
    program.push_back(build_s_swappc_b64(30, 8, arch).value());
  };
  program.push_back(0xBF840007u); // s_cbranch_scc0 to the second call at 0x20.
  call_helper(60);                // Return to the v0 consumer at 0x18.
  append_gfx950_v_mov_b32_v1_v0(program);
  program.push_back(build_s_endpgm(arch).value());
  append_gfx950_buffer_load_dword_v0_v8_s0_offen(program);
  call_helper(24); // The second caller's continuation drains the load independently.
  append_gfx950_s_waitcnt_vmcnt_0(program);
  program.push_back(build_s_endpgm(arch).value());
  append_gfx950_buffer_load_dword_v0_v8_s0_offen(program); // Shared helper, 0x44.
  append_gfx950_s_waitcnt_vmcnt_0(program);                // Shared return wait, 0x4c.
  program.push_back(build_sop1_encoding(arch, 0x1d, 0, 30).value());

  const auto image =
      rocjitsu::waitcheck_test::make_gfx_code_object(program, EF_AMDGPU_MACH_AMDGCN_GFX950);
  AmdGpuCodeObject object(image.data(), image.size());
  WaitcheckOptions options;
  options.check_counter_parity = true;
  const auto report = analyze_waitcnts(object, arch, options);
  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_GE(report.counter_parity_exact, 1u);
  EXPECT_FALSE(std::ranges::any_of(report.counter_underaccounting_diagnostics,
                                   [](const auto &d) { return d.wait_section_offset == 0x4c; }));
}

TEST(WaitcheckTest, Gfx950SharedWaitUsesTheStrongestCallerThresholdOnce) {
  constexpr auto arch = ROCJITSU_CODE_ARCH_CDNA4;
  for (bool younger_load : {false, true}) {
    for (bool partial : {false, true}) {
      SCOPED_TRACE(testing::Message() << younger_load << ", " << partial);
      std::vector<uint32_t> program;
      append_gfx950_buffer_load_dword_v0_v8_s0_offen(program);
      if (younger_load)
        append_gfx950_buffer_load_dword_v1_v8_s0_offen(program);
      program.push_back(build_s_call_b64(30, 5, arch).value());
      append_gfx950_s_waitcnt_vmcnt_0(program);
      append_gfx950_buffer_load_dword_v0_v8_s0_offen(program);
      program.push_back(build_s_call_b64(30, 1, arch).value());
      program.push_back(build_s_endpgm(arch).value());
      program.push_back(0xbf800000u); // s_nop 0 inside the shared helper.
      if (partial)
        append_gfx950_s_waitcnt_vmcnt_1(program);
      else
        append_gfx950_s_waitcnt_vmcnt_0(program);
      program.push_back(0x7e060300u); // v_mov_b32 v3, v0
      program.push_back(build_sop1_encoding(arch, 0x1d, 0, 30).value());
      const auto image =
          waitcheck_test::make_gfx_code_object(program, EF_AMDGPU_MACH_AMDGCN_GFX950);
      AmdGpuCodeObject object(image.data(), image.size());
      WaitcheckOptions options;
      options.check_counter_parity = true;
      const auto report = analyze_waitcnts(object, arch, options);
      ASSERT_TRUE(report.supported && report.complete) << report.analysis_error;
      EXPECT_EQ(report.diagnostics_observed, partial ? (younger_load ? 1u : 2u) : 0u);
      EXPECT_EQ(report.counter_parity_fields_checked, 2u);
      EXPECT_EQ(report.counter_parity_exact, partial ? 0u : 1u);
    }
  }
}

TEST(WaitcheckTest, Gfx950PartialWaitChecksYoungerLoadsAfterItsFirstExactUse) {
  constexpr auto arch = ROCJITSU_CODE_ARCH_CDNA4;
  WaitcheckOptions options;
  options.check_counter_parity = true;
  for (unsigned callers : {0u, 1u, 2u, 3u}) {
    for (bool use_younger : {false, true}) {
      for (bool drain : {false, true}) {
        SCOPED_TRACE(testing::Message() << callers << ", " << use_younger << ", " << drain);
        std::vector<uint32_t> program;
        auto issue = [&] {
          append_gfx950_buffer_load_dword_v0_v8_s0_offen(program);
          append_gfx950_buffer_load_dword_v1_v8_s0_offen(program);
        };
        auto helper = [&] {
          program.push_back(0xbf800000u); // s_nop 0.
          append_gfx950_s_waitcnt_vmcnt_1(program);
          program.push_back(0x7e060300u); // v_mov_b32 v3, v0 needs vmcnt(1).
          if (drain)
            append_gfx950_s_waitcnt_vmcnt_0(program);
          if (use_younger)
            program.push_back(0x7e080301u); // v_mov_b32 v4, v1 needs vmcnt(0).
        };
        if (!callers) {
          issue();
          helper();
        } else {
          program.push_back(0xbf820000u); // s_branch main.
          helper();
          program.push_back(build_sop1_encoding(arch, 0x1d, 0, 30).value());
          program[0] |= static_cast<uint32_t>(program.size() - 1);
          for (unsigned call = 0; call < callers; ++call) {
            if (call)
              append_gfx950_s_waitcnt_vmcnt_0(program);
            issue();
            program.push_back(
                build_s_call_b64(30, -static_cast<int>(program.size()), arch).value());
          }
        }
        program.push_back(build_s_endpgm(arch).value());
        const auto image =
            waitcheck_test::make_gfx_code_object(program, EF_AMDGPU_MACH_AMDGCN_GFX950);
        AmdGpuCodeObject object(image.data(), image.size());
        const auto report = analyze_waitcnts(object, arch, options);
        ASSERT_TRUE(report.supported && report.complete) << report.analysis_error;
        EXPECT_EQ(report.diagnostics_observed, use_younger && !drain ? std::max(callers, 1u) : 0u);
        EXPECT_EQ(report.counter_parity_exact, use_younger ? (drain ? 2u : 0u) : 1u);
        if (!callers) {
          const auto stream = analyze_waitcnts(program, arch, options);
          EXPECT_EQ(stream.diagnostics_observed, report.diagnostics_observed);
          EXPECT_EQ(stream.counter_parity_exact, report.counter_parity_exact);
        }
      }
    }
  }
}

TEST(WaitcheckTest, Gfx950PartialWaitChecksSiblingBranchAndIgnoresPostWaitProducers) {
  constexpr auto arch = ROCJITSU_CODE_ARCH_CDNA4;
  WaitcheckOptions options;
  options.check_counter_parity = true;
  for (bool reissue : {false, true}) {
    std::vector<uint32_t> program;
    append_gfx950_buffer_load_dword_v0_v8_s0_offen(program);
    append_gfx950_buffer_load_dword_v1_v8_s0_offen(program);
    append_gfx950_s_waitcnt_vmcnt_1(program);
    program.push_back(0x7e060300u); // v_mov_b32 v3, v0.
    program.push_back(0xbf840001u); // s_cbranch_scc0 to the younger read path.
    program.push_back(build_s_endpgm(arch).value());
    if (reissue) {
      append_gfx950_s_waitcnt_vmcnt_0(program);
      append_gfx950_buffer_load_dword_v1_v8_s0_offen(program);
    }
    program.push_back(0x7e080301u); // v_mov_b32 v4, v1.
    program.push_back(build_s_endpgm(arch).value());
    const auto image = waitcheck_test::make_gfx_code_object(program, EF_AMDGPU_MACH_AMDGCN_GFX950);
    AmdGpuCodeObject object(image.data(), image.size());
    const auto report = analyze_waitcnts(object, arch, options);
    ASSERT_TRUE(report.supported && report.complete) << report.analysis_error;
    EXPECT_EQ(report.diagnostics_observed, 1u);
    // The second physical wait independently guards the v1 overwrite.
    EXPECT_EQ(report.counter_parity_exact, reissue ? 2u : 0u);
  }
}

TEST(WaitcheckTest, Gfx950PartialLgkmcntGuardsOlderReadBeforeBarrier) {
  constexpr auto arch = ROCJITSU_CODE_ARCH_CDNA4;
  std::vector<uint32_t> program;
  append_gfx950_ds_read_b32_v0_v4(program);
  append_gfx950_ds_read_b32_v0_v4(program);
  program.back() |= 1u << 24; // ds_read_b32 v1, v4.
  append_gfx950_s_waitcnt_lgkmcnt_1(program);
  program.push_back(0x7e060300u); // v_mov_b32 v3, v0.
  program.push_back(0xbf8a0000u); // s_barrier does not consume v1.
  program.push_back(build_s_endpgm(arch).value());
  const auto image = waitcheck_test::make_gfx_code_object(program, EF_AMDGPU_MACH_AMDGCN_GFX950);
  AmdGpuCodeObject object(image.data(), image.size());
  WaitcheckOptions options;
  options.check_counter_parity = true;
  const auto report = analyze_waitcnts(object, arch, options);
  EXPECT_EQ(report.diagnostics_observed, 0u);
  EXPECT_EQ(report.counter_parity_exact, 1u);
}

TEST(WaitcheckTest, PartialEmbeddedWaitChecksOtherOriginalValuResults) {
  WaitcheckOptions options;
  options.check_counter_parity = true;
  for (bool use_younger : {false, true}) {
    std::vector<uint32_t> program;
    append_inst(program, s_setreg_imm32_b32(hwreg(26, 0, 2), 2)); // Enable expert scheduling.
    append_inst(program, v_add_f32_e32(0, 257, 1));               // Reads v1, then reaches age 1.
    append_inst(program, v_add_f32_e32(4, 261, 5));               // Reads v5, still age 0.
    append_inst(program, ds_param_load_with_waits(1, 1, 1));
    append_inst(program, ds_param_load_with_waits(use_younger ? 5 : 12, 15, 1));
    append_inst(program, s_endpgm());
    TestCodeObject object(program);
    const auto report = analyze_waitcnts(object, ROCJITSU_CODE_ARCH_RDNA4, options);
    EXPECT_EQ(report.diagnostics_observed, use_younger ? 1u : 0u);
    EXPECT_EQ(report.counter_parity_exact, use_younger ? 0u : 1u);
  }
}

TEST(WaitcheckTest, Gfx950PhysicalWaitCombinesOrdinaryAndReturnPaths) {
  constexpr auto arch = ROCJITSU_CODE_ARCH_CDNA4;
  for (bool issue_load : {false, true}) {
    for (bool partial : {false, true}) {
      SCOPED_TRACE(testing::Message() << issue_load << ", " << partial);
      std::vector<uint32_t> program{0xbf840003u}; // Branch to the call at 0x10.
      if (issue_load) {
        append_gfx950_buffer_load_dword_v0_v8_s0_offen(program);
      } else {
        program.insert(program.end(), 2, 0xbf800000u); // Preserve the layout.
      }
      program.push_back(0xbf820001u); // Branch to the same wait without calling.
      program.push_back(build_s_call_b64(30, 3, arch).value());
      if (partial)
        append_gfx950_s_waitcnt_vmcnt_1(program);
      else
        append_gfx950_s_waitcnt_vmcnt_0(program);
      append_gfx950_v_mov_b32_v1_v0(program);
      program.push_back(build_s_endpgm(arch).value());
      program.push_back(build_sop1_encoding(arch, 0x1d, 0, 30).value());

      const auto image =
          waitcheck_test::make_gfx_code_object(program, EF_AMDGPU_MACH_AMDGCN_GFX950);
      AmdGpuCodeObject object(image.data(), image.size());
      WaitcheckOptions options;
      options.check_counter_parity = true;
      const auto report = analyze_waitcnts(object, arch, options);
      ASSERT_TRUE(report.supported && report.complete) << report.analysis_error;
      EXPECT_EQ(report.diagnostics.empty(), !(issue_load && partial));
      EXPECT_EQ(std::ranges::count_if(report.counter_underaccounting_diagnostics,
                                      [](const auto &d) { return d.wait_section_offset == 0x14; }),
                issue_load ? 0 : 1);
    }
  }
}

TEST(WaitcheckTest, Gfx950CounterParityPreservesFunctionPrologWaitDependency) {
  constexpr uint8_t kTargetSreg = 26;
  constexpr uint8_t kReturnSreg = 30;
  constexpr uint8_t kLiteralOperand = 0xff;
  constexpr uint8_t kInlineInt0 = 128;

  std::vector<uint32_t> program;
  append_gfx950_buffer_load_dword_v0_v8_s0_offen(program);                             // 0x00.
  append_gfx950_buffer_load_dword_v1_v8_s0_offen(program);                             // 0x08.
  program.push_back(build_s_getpc_b64(kTargetSreg, ROCJITSU_CODE_ARCH_CDNA4).value()); // 0x10.
  program.push_back(build_s_add_u32(kTargetSreg, kTargetSreg, kLiteralOperand,
                                    ROCJITSU_CODE_ARCH_CDNA4)
                        .value()); // 0x14.
  program.push_back(20);           // 0x18 -> helper 0x28.
  program.push_back(build_s_addc_u32(kTargetSreg + 1, kTargetSreg + 1, kInlineInt0,
                                     ROCJITSU_CODE_ARCH_CDNA4)
                        .value()); // 0x1c.
  program.push_back(
      build_s_swappc_b64(kReturnSreg, kTargetSreg, ROCJITSU_CODE_ARCH_CDNA4).value()); // 0x20.
  program.push_back(build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4).value());                 // 0x24.
  const size_t helper_entry = program.size();
  append_gfx950_s_waitcnt_vmcnt_0(program); // 0x28 helper.
  program.push_back(0x7E040300u);           // 0x2c: v_mov_b32_e32 v2, v0.
  program.push_back(build_sop1_encoding(ROCJITSU_CODE_ARCH_CDNA4, 0x1d, 0,
                                        kReturnSreg)
                        .value()); // 0x30 return.

  const auto image =
      rocjitsu::waitcheck_test::make_gfx_code_object(program, EF_AMDGPU_MACH_AMDGCN_GFX950);
  AmdGpuCodeObject code_object(image.data(), image.size());
  WaitcheckOptions options;
  options.check_counter_parity = true;
  const auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_CDNA4, options);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.passed()) << diagnostic_summary(report);
  EXPECT_EQ(report.counter_parity_fields_checked, 1u);
  EXPECT_EQ(report.counter_parity_exact, 0u);
  EXPECT_EQ(report.counter_underaccounting_observed, 1u);
  EXPECT_EQ(report.counter_unmodeled_wait_observed, 0u);
  ASSERT_EQ(report.counter_underaccounting_diagnostics.size(), 1u);
  EXPECT_TRUE(report.counter_underaccounting_diagnostics[0].has_required_dependency);
  EXPECT_EQ(report.counter_underaccounting_diagnostics[0].required_count, 1u);
  EXPECT_NE(report.counter_underaccounting_diagnostics[0].message.find("function ABI"),
            std::string::npos);
  // The same data dependency applies at an interior block with no ABI policy.
  program.insert(program.begin() + helper_entry, 0xbf820000u); // s_branch to next instruction
  const auto interior_image =
      rocjitsu::waitcheck_test::make_gfx_code_object(program, EF_AMDGPU_MACH_AMDGCN_GFX950);
  AmdGpuCodeObject interior(interior_image.data(), interior_image.size());
  const auto inner = analyze_waitcnts(interior, ROCJITSU_CODE_ARCH_CDNA4, options);
  ASSERT_TRUE(inner.passed()) << diagnostic_summary(inner);
  ASSERT_EQ(inner.counter_underaccounting_diagnostics.size(), 1u);
  EXPECT_TRUE(inner.counter_underaccounting_diagnostics[0].has_required_dependency);
  EXPECT_EQ(inner.counter_underaccounting_diagnostics[0].required_count, 1u);
}

TEST(WaitcheckTest, Gfx950CounterParityPreservesFunctionReturnWaitDependency) {
  constexpr uint8_t kTargetSreg = 26;
  constexpr uint8_t kReturnSreg = 30;
  constexpr uint8_t kLiteralOperand = 0xff;
  constexpr uint8_t kInlineInt0 = 128;

  std::vector<uint32_t> program;
  program.push_back(build_s_getpc_b64(kTargetSreg, ROCJITSU_CODE_ARCH_CDNA4).value()); // 0x00.
  program.push_back(build_s_add_u32(kTargetSreg, kTargetSreg, kLiteralOperand,
                                    ROCJITSU_CODE_ARCH_CDNA4)
                        .value()); // 0x04.
  program.push_back(24);           // PC 0x04 -> helper 0x1c.
  program.push_back(build_s_addc_u32(kTargetSreg + 1, kTargetSreg + 1, kInlineInt0,
                                     ROCJITSU_CODE_ARCH_CDNA4)
                        .value()); // 0x0c.
  program.push_back(
      build_s_swappc_b64(kReturnSreg, kTargetSreg, ROCJITSU_CODE_ARCH_CDNA4).value()); // 0x10.
  program.push_back(0x7E040300u); // 0x14: v_mov_b32_e32 v2, v0.
  program.push_back(build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4).value()); // 0x18.
  append_gfx950_buffer_load_dword_v0_v8_s0_offen(program);             // 0x1c helper.
  append_gfx950_buffer_load_dword_v1_v8_s0_offen(program);             // 0x24.
  append_gfx950_s_waitcnt_vmcnt_0(program);                            // 0x2c return wait.
  program.push_back(build_sop1_encoding(ROCJITSU_CODE_ARCH_CDNA4, 0x1d, 0,
                                        kReturnSreg)
                        .value()); // 0x30 return.

  const auto image =
      rocjitsu::waitcheck_test::make_gfx_code_object(program, EF_AMDGPU_MACH_AMDGCN_GFX950);
  AmdGpuCodeObject code_object(image.data(), image.size());
  WaitcheckOptions options;
  options.check_counter_parity = true;
  const auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_CDNA4, options);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.passed()) << diagnostic_summary(report);
  EXPECT_EQ(report.counter_parity_fields_checked, 1u);
  EXPECT_EQ(report.counter_parity_exact, 0u);
  EXPECT_EQ(report.counter_underaccounting_observed, 1u);
  EXPECT_EQ(report.counter_unmodeled_wait_observed, 0u);
  ASSERT_EQ(report.counter_underaccounting_diagnostics.size(), 1u);
  EXPECT_TRUE(report.counter_underaccounting_diagnostics[0].has_required_dependency);
  EXPECT_EQ(report.counter_underaccounting_diagnostics[0].required_count, 1u);
  EXPECT_NE(report.counter_underaccounting_diagnostics[0].message.find("function ABI"),
            std::string::npos);
}

TEST(WaitcheckTest, Gfx950SharedReturnWaitAccountsForPartialCallerDrains) {
  constexpr auto arch = ROCJITSU_CODE_ARCH_CDNA4;
  for (unsigned callers : {1u, 2u, 3u}) {
    const bool separate_call = callers == 2;
    const bool exact_caller = callers == 3;
    for (bool partial : {false, true}) {
      for (bool relax_return : {false, true}) {
        std::vector<uint32_t> program;
        if (separate_call || exact_caller)
          program.push_back(0xbf840000u); // s_cbranch_scc0 to the second caller
        std::vector<size_t> call_literals;
        auto call = [&] {
          program.push_back(build_s_getpc_b64(8, arch).value());
          program.push_back(build_s_add_u32(8, 8, 255, arch).value());
          call_literals.push_back(program.size());
          program.push_back(0);
          program.push_back(build_s_addc_u32(9, 9, 128, arch).value());
          program.push_back(build_s_swappc_b64(30, 8, arch).value());
        };
        call();
        program.push_back(0x7e040300u); // v_mov_b32 v2, v0
        if (separate_call) {
          program.push_back(build_s_endpgm(arch).value());
          program[0] |= static_cast<uint32_t>(program.size() - 1);
          call();
        }
        if (partial)
          append_gfx950_s_waitcnt_vmcnt_1(program);
        else
          append_gfx950_s_waitcnt_vmcnt_0(program);
        const auto younger_consumer = program.size() * 4;
        program.push_back(0x7e060301u); // v_mov_b32 v3, v1
        program.push_back(build_s_endpgm(arch).value());
        if (exact_caller) {
          program[0] |= static_cast<uint32_t>(program.size() - 1);
          call();
          program.push_back(0x7e060301u); // This caller already requires the younger load.
          program.push_back(build_s_endpgm(arch).value());
        }
        for (size_t literal : call_literals)
          program[literal] = (program.size() - literal + 1) * 4;
        append_gfx950_buffer_load_dword_v0_v8_s0_offen(program);
        const auto younger_producer = program.size() * 4;
        append_gfx950_buffer_load_dword_v1_v8_s0_offen(program);
        const auto return_wait = program.size() * 4;
        if (relax_return)
          append_gfx950_s_waitcnt_vmcnt_1(program);
        else
          append_gfx950_s_waitcnt_vmcnt_0(program);
        program.push_back(build_sop1_encoding(arch, 0x1d, 0, 30).value());

        const auto image =
            waitcheck_test::make_gfx_code_object(program, EF_AMDGPU_MACH_AMDGCN_GFX950);
        AmdGpuCodeObject object(image.data(), image.size());
        WaitcheckOptions options;
        options.check_counter_parity = true;
        const auto report = analyze_waitcnts(object, arch, options);
        ASSERT_TRUE(report.supported && report.complete) << report.analysis_error;
        if (relax_return && (partial || exact_caller)) {
          ASSERT_EQ(report.diagnostics.size(), unsigned(partial) + unsigned(exact_caller))
              << diagnostic_summary(report);
          if (partial)
            EXPECT_TRUE(std::ranges::any_of(report.diagnostics, [&](const auto &d) {
              return d.section_offset == younger_consumer &&
                     d.producer_section_offset == younger_producer && d.required_count == 0;
            }));
        } else {
          EXPECT_TRUE(report.passed()) << diagnostic_summary(report);
        }
        if (relax_return)
          continue;
        const auto &diagnostics = report.counter_underaccounting_diagnostics;
        auto found =
            std::ranges::find(diagnostics, return_wait,
                              &WaitcheckCounterUnderaccountingDiagnostic::wait_section_offset);
        if (exact_caller || (partial && !separate_call)) {
          EXPECT_EQ(found, diagnostics.end());
          // The original caller already requiring zero and the staged witness
          // refer to one shared helper wait.
          EXPECT_GE(report.counter_parity_exact, 1u);
          continue;
        }
        ASSERT_NE(found, diagnostics.end());
        EXPECT_EQ(found->has_required_dependency, !partial);
        if (partial)
          EXPECT_NE(found->message.find("unresolved"), std::string::npos);
        else
          EXPECT_EQ(found->required_count, 1u);
      }
    }
  }
}

TEST(WaitcheckTest, SkippedVectorIssuesCannotCreditAnOlderLoadAtAPartialWait) {
  constexpr uint32_t enabled = 0xbf108081, disabled = 0xbf108080;
  constexpr uint32_t dynamic = 0xbf108002; // s_setvskip s2, 0
  // Independently assembled gfx942 schedules. In the first, a repeated s12
  // comparison controls the younger load and the MFMA. The second uses two
  // comparisons of the masked split count before its accumulator write.
  const std::vector<uint32_t> matrix{
      0xe05c1000, 0x800e000a, 0xbf06810c, 0xbf850004, enabled, 0xe05c1000, 0x800e040b,
      disabled,   0xbf800000, 0xbf06810c, 0xbf850003, 0xbf8c0f71, 0xd3e18000, 0x0402010c,
      0xbf810000};
  const std::vector<uint32_t> accumulator{
      0x860cff37, 0x00003fff, 0xbf06810c, 0xbf850002, 0xe0501000, 0x800e000a, 0x8608ff37,
      0x00003fff, 0x81888108, 0xbf008008, 0xbf850004, enabled,    0xe0501000, 0x800e040b,
      disabled,   0xbf8c0f71, 0xd3d94000, 0x18000100, 0xbf810000};
  auto check = [=](std::vector<uint32_t> words, size_t mode, size_t wait, size_t producer) {
    for (uint32_t setting : {enabled, disabled, dynamic}) {
      for (uint32_t threshold : {0u, 1u}) {
        words[mode] = setting;
        words[wait] = 0xbf8c0f70u + threshold;
        const auto image = waitcheck_test::make_gfx_code_object(words, EF_AMDGPU_MACH_AMDGCN_GFX942);
        const AmdGpuCodeObject object(image.data(), image.size());
        const auto report = analyze_waitcnts_for_kernel(object, ROCJITSU_CODE_ARCH_CDNA3, 0);
        ASSERT_TRUE(report.complete) << report.analysis_error;
        const bool hazard = setting != disabled && threshold != 0;
        ASSERT_EQ(report.diagnostics.size(), hazard ? 1u : 0u) << diagnostic_summary(report);
        if (hazard) {
          EXPECT_EQ(report.diagnostics.front().producer_section_offset, producer);
          EXPECT_EQ(report.diagnostics.front().section_offset, (wait + 1) * sizeof(uint32_t));
          EXPECT_EQ(report.diagnostics.front().required_count, 0u);
        }
      }
    }
  };
  check(matrix, 4, 11, 0);
  check(accumulator, 11, 15, 16);
}

TEST(WaitcheckTest, VectorSkipModeControlsWholeInstructionsAndCounterCredits) {
  const auto check = [](std::vector<uint32_t> mode, unsigned expected) {
    std::vector<uint32_t> words{0xe05c1000, 0x800e000a}; // buffer_load_dwordx4 v[0:3]
    words.insert(words.end(), mode.begin(), mode.end());
    words.insert(words.end(), {0xe05c1000, 0x800e040b, // buffer_load_dwordx4 v[4:7]
                               0xbf108080,             // s_setvskip 0, 0
                               0xbf8c0f71,             // s_waitcnt vmcnt(1)
                               0x7e140300,             // v_mov_b32 v10, v0
                               0xbf810000});
    const auto image = waitcheck_test::make_gfx_code_object(words, EF_AMDGPU_MACH_AMDGCN_GFX942);
    const AmdGpuCodeObject object(image.data(), image.size());
    const auto report = analyze_waitcnts_for_kernel(object, ROCJITSU_CODE_ARCH_CDNA3, 0);
    ASSERT_TRUE(report.complete) << report.analysis_error;
    ASSERT_EQ(report.diagnostics.size(), expected) << diagnostic_summary(report);
  };
  check({0xbf108081}, 1);    // Explicit enable.
  check({0xbf108080}, 0);    // Explicit disable.
  check({0xbf108002}, 1);    // Run-time selector can enable the skip.
  check({0xba000701, 1}, 1); // Immediate write to MODE[28].
  check({0xba000701, 0}, 0);
  check({0xba0006c1, 1}, 0); // Writing an adjacent bit preserves VSKIP.
  check({0xb9070701}, 1);    // Register-sourced MODE[28] write.
}

TEST(WaitcheckTest, MatrixUseRespectsRepeatedScalarGuardsOnYoungerLoads) {
  const auto check = [](uint32_t second_compare, uint32_t intervening, uint32_t wait,
                        size_t expected) {
    std::vector<uint32_t> words{
        0xe05c1000,
        0x800e000a, // buffer_load_dwordx4 v[0:3], v10, s[56:59], 0 offen
        0xbf06810c, // s_cmp_eq_u32 s12, 1
        0xbf850002, // s_cbranch_scc1 +2 (skip the younger load)
        0xe05c1000,
        0x800e040b, // buffer_load_dwordx4 v[4:7], v11, s[56:59], 0 offen
        intervening,
        second_compare,
        0xbf850003,         // s_cbranch_scc1 +3 (skip the consumer)
        0xbf8c0f70u | wait, // s_waitcnt vmcnt(wait)
        0xd3e18000,
        0x0402010c, // v_mfma_f32_16x16x16_bf16 acc[0:3], v[12:13], v[0:1], acc[0:3]
        0xbf810000, // s_endpgm
    };
    const auto image = waitcheck_test::make_gfx_code_object(words, EF_AMDGPU_MACH_AMDGCN_GFX942);
    const AmdGpuCodeObject object(image.data(), image.size());
    const auto report = analyze_waitcnts(object, ROCJITSU_CODE_ARCH_CDNA3);
    ASSERT_TRUE(report.supported && report.complete) << report.analysis_error;
    ASSERT_EQ(report.diagnostics.size(), expected) << diagnostic_summary(report);
    if (expected != 0) {
      EXPECT_EQ(report.diagnostics[0].producer_section_offset, 0u);
      EXPECT_EQ(report.diagnostics[0].reg.index, 0u);
      EXPECT_EQ(report.diagnostics[0].required_count, 0u);
    }
  };
  constexpr uint32_t same = 0xbf06810c, independent = 0xbf06810d;
  constexpr uint32_t nop = 0xbf800000, redefine = 0xbe8c0090;
  check(same, nop, 1, 0);
  check(independent, nop, 1, 1);
  check(same, redefine, 1, 1);
  check(independent, nop, 0, 0);
  check(same, nop, 2, 1);
}

TEST(WaitcheckTest, MatrixWaitCountsAccountForYoungerLoadsAndCompetingUses) {
  const auto check = [](uint32_t second_compare, uint32_t wait, bool use_younger, size_t ordinary,
                        size_t exact, bool expects_one) {
    const std::vector<uint32_t> words{
        0xe05c1000,
        0x800e000a, // buffer_load_dwordx4 v[0:3], v10, s[56:59], 0 offen
        0xbf06810c, // s_cmp_eq_u32 s12, 1
        0xbf850002, // s_cbranch_scc1 skip_load
        0xe05c1000,
        0x800e040b, // buffer_load_dwordx4 v[4:7], v11, s[56:59], 0 offen
        second_compare,
        0xbf850004,         // s_cbranch_scc1 exit
        0xbf8c0f70u | wait, // s_waitcnt vmcnt(wait)
        0xd3e18000,
        0x0402010c,                              // MFMA uses v[0:1]
        use_younger ? 0x7e1e0304u : 0xbf800000u, // v_mov_b32 v15, v4 or s_nop
        0xbf810000,                              // s_endpgm
    };
    const auto image = waitcheck_test::make_gfx_code_object(words, EF_AMDGPU_MACH_AMDGCN_GFX942);
    const AmdGpuCodeObject object(image.data(), image.size());
    WaitcheckOptions options;
    options.check_counter_parity = true;
    const auto report = analyze_waitcnts_for_kernel(object, ROCJITSU_CODE_ARCH_CDNA3, 0, options);
    ASSERT_TRUE(report.supported && report.complete) << report.analysis_error;
    EXPECT_EQ(report.diagnostics.size(), ordinary) << diagnostic_summary(report);
    EXPECT_EQ(report.counter_parity_fields_checked, 1u);
    EXPECT_EQ(report.counter_parity_exact, exact);
    EXPECT_EQ(report.counter_parity_indeterminate_groups, 0u);
    ASSERT_EQ(report.counter_underaccounting_diagnostics.size(), expects_one ? 1u : 0u);
    if (expects_one) {
      const auto &field = report.counter_underaccounting_diagnostics.front();
      EXPECT_TRUE(field.has_required_dependency);
      EXPECT_EQ(field.reg.index, 0u);
      EXPECT_EQ(field.required_count, 1u);
    }
  };
  constexpr uint32_t same = 0xbf06810c, other = 0xbf06810d;
  check(same, 0, false, 0, 0, true);
  check(same, 1, false, 0, 1, false);
  check(same, 2, false, 1, 0, false);
  check(other, 0, false, 0, 1, false);
  check(other, 1, false, 1, 0, false);
  check(same, 0, true, 0, 1, false);
  check(same, 1, true, 1, 0, false);
}

TEST(WaitcheckTest, GuardedCdna3MatrixWaitCountsAnEmptyVmemIssue) {
  constexpr uint32_t nop = 0xbf800000;
  constexpr uint32_t exec_zero = 0xbefe0180;    // s_mov_b64 exec, 0
  constexpr uint32_t exec_unknown = 0xbefe0102; // s_mov_b64 exec, s[2:3]
  constexpr uint32_t exec_full = 0xbefe01c1;    // s_mov_b64 exec, -1
  constexpr uint32_t vskip_on = 0xbf108081, vskip_off = 0xbf108080;
  constexpr uint32_t load_buffer = 0xe05c1000, load_lds = 0xd8d80000;
  constexpr uint32_t buffer_operands = 0x800e040b, lds_operands = 0x0400000b;
  struct Issue {
    uint32_t mask;
    uint32_t vskip;
    uint32_t memory;
    uint32_t operands;
    unsigned required;
  };
  const Issue cases[] = {
      {exec_zero, nop, load_buffer, buffer_operands, 1},
      {exec_unknown, nop, load_buffer, buffer_operands, 1},
      {exec_full, nop, load_buffer, buffer_operands, 1},
      {exec_zero, vskip_on, load_buffer, buffer_operands, 0},
      {exec_zero, nop, load_lds, lds_operands, 0},
  };
  for (const auto &issue : cases) {
    for (uint32_t threshold : {0u, 1u, 2u}) {
      SCOPED_TRACE(testing::Message() << "mask " << issue.mask << " vskip " << issue.vskip
                                      << " memory " << issue.memory << " threshold " << threshold);
      const std::vector<uint32_t> words{
          0xe05c1000,
          0x800e000a, // buffer_load_dwordx4 v[0:3], v10, s[56:59], 0 offen
          0xbf06810c, // s_cmp_eq_u32 s12, 1
          0xbf850006, // s_cbranch_scc1 skip_load
          issue.mask,
          issue.vskip,
          issue.memory,
          issue.operands,
          vskip_off,
          exec_full,
          0xbf06810c,              // s_cmp_eq_u32 s12, 1
          0xbf850003,              // s_cbranch_scc1 exit
          0xbf8c0f70u | threshold, // s_waitcnt vmcnt(threshold)
          0xd3e18000,
          0x0402010c, // MFMA reads v[0:1].
          0xbf810000, // s_endpgm
      };
      const auto image = waitcheck_test::make_gfx_code_object(words, EF_AMDGPU_MACH_AMDGCN_GFX942);
      const AmdGpuCodeObject object(image.data(), image.size());
      WaitcheckOptions options;
      options.check_counter_parity = true;
      const auto report = analyze_waitcnts_for_kernel(object, ROCJITSU_CODE_ARCH_CDNA3, 0, options);
      ASSERT_TRUE(report.supported && report.complete) << report.analysis_error;
      EXPECT_EQ(report.diagnostics.size(), threshold > issue.required);
      EXPECT_EQ(report.counter_parity_fields_checked, 1u);
      EXPECT_EQ(report.counter_parity_exact, threshold == issue.required);
      EXPECT_EQ(report.counter_parity_indeterminate_groups, 0u);
      ASSERT_EQ(report.counter_underaccounting_diagnostics.size(), threshold < issue.required);
      if (threshold < issue.required) {
        const auto &finding = report.counter_underaccounting_diagnostics.front();
        EXPECT_TRUE(finding.has_required_dependency);
        EXPECT_EQ(finding.required_count, issue.required);
        EXPECT_EQ(finding.reg.index, 0u);
      }
    }
  }
}

TEST(WaitcheckTest, Cdna3MatrixWaitCountsAnEmptyReissueAtTheSameInstruction) {
  for (const uint32_t threshold : {0u, 1u, 2u}) {
    SCOPED_TRACE(threshold);
    const std::vector<uint32_t> words{
        0xbe8c0080, // s_mov_b32 s12, 0
        0xe05c1000,
        0x800e000a, // loop: buffer_load_dwordx4 v[0:3], v10, s[56:59], 0 offen
        0xbf06810c, // s_cmp_eq_u32 s12, 1
        0xbf850003, // s_cbranch_scc1 consume
        0xbe8c0081, // s_mov_b32 s12, 1
        0xbefe0180, // s_mov_b64 exec, 0
        0xbf82fff9, // s_branch loop
        0xbefe01c1, // consume: s_mov_b64 exec, -1
        0xbf8c0f70u | threshold,
        0xd3e18000,
        0x0402010c, // MFMA reads v[0:1] from the first trip.
        0xbf810000,
    };
    const auto image = waitcheck_test::make_gfx_code_object(words, EF_AMDGPU_MACH_AMDGCN_GFX942);
    const AmdGpuCodeObject object(image.data(), image.size());
    WaitcheckOptions options;
    for (const bool parity : {false, true}) {
      SCOPED_TRACE(parity);
      options.check_counter_parity = parity;
      const auto report = analyze_waitcnts_for_kernel(object, ROCJITSU_CODE_ARCH_CDNA3, 0, options);
      ASSERT_TRUE(report.supported && report.complete) << report.analysis_error;
      ASSERT_EQ(report.diagnostics.size(), threshold == 2);
      if (threshold == 2)
        EXPECT_EQ(report.diagnostics.front().required_count, 1u);
      if (!parity)
        continue;
      EXPECT_EQ(report.counter_parity_fields_checked, 1u);
      EXPECT_EQ(report.counter_parity_exact, threshold == 1);
      EXPECT_EQ(report.counter_parity_indeterminate_groups, 0u);
      ASSERT_EQ(report.counter_underaccounting_diagnostics.size(), threshold == 0);
      if (threshold == 0) {
        const auto &finding = report.counter_underaccounting_diagnostics.front();
        EXPECT_TRUE(finding.has_required_dependency);
        EXPECT_EQ(finding.required_count, 1u);
        EXPECT_EQ(finding.producer_section_offset, 4u);
      }
    }
  }
}

TEST(WaitcheckTest, AccumulatorWaitNeedsAFeasibleMaskedUse) {
  const auto check = [](uint32_t mask, uint32_t clobber, uint32_t wait, size_t ordinary,
                        size_t exact, bool unattributed, size_t indeterminate) {
    const std::vector<uint32_t> words{
        0x860cff37,
        0x3fff,     // s_and_b32 s12, s55, 0x3fff
        0xbf06810c, // s_cmp_eq_u32 s12, 1
        0xbf850002, // s_cbranch_scc1 wait
        0xe050d000,
        0x0d0f0017,         // buffer_load_dword v0, v23, s[60:63], s13
        0xbf8c0f70u | wait, // s_waitcnt vmcnt(wait)
        clobber,
        0x8608ff37,
        mask,       // s_and_b32 s8, s55, mask
        0x81888108, // s_sub_i32 s8, s8, 1
        0xbf008008, // s_cmp_eq_i32 s8, 0
        0xbf840002, // s_cbranch_scc0 exit
        0xd3d94000,
        0x18000100, // v_accvgpr_write acc0, v0
        0xbf810000, // s_endpgm
    };
    const auto image = waitcheck_test::make_gfx_code_object(words, EF_AMDGPU_MACH_AMDGCN_GFX942);
    const AmdGpuCodeObject object(image.data(), image.size());
    WaitcheckOptions options;
    options.check_counter_parity = true;
    const auto report = analyze_waitcnts_for_kernel(object, ROCJITSU_CODE_ARCH_CDNA3, 0, options);
    ASSERT_TRUE(report.supported && report.complete) << report.analysis_error;
    EXPECT_EQ(report.diagnostics.size(), ordinary) << diagnostic_summary(report);
    EXPECT_EQ(report.counter_parity_exact, exact);
    EXPECT_EQ(report.counter_parity_indeterminate_groups, indeterminate);
    ASSERT_EQ(report.counter_underaccounting_diagnostics.size(), unattributed ? 1u : 0u);
    if (unattributed)
      EXPECT_FALSE(report.counter_underaccounting_diagnostics.front().has_required_dependency);
  };
  constexpr uint32_t mask = 0x3fff, nop = 0xbf800000, redefine_root = 0xbeb70081;
  check(mask, nop, 0, 0, 0, true, 0);
  check(mask, nop, 1, 0, 0, true, 0);
  check(0xffff, nop, 0, 0, 1, false, 0);
  check(0xffff, nop, 1, 1, 0, false, 0);
  check(mask, redefine_root, 0, 0, 1, false, 0);
  check(mask, 0xbe872a64, 0, 0, 0, false, 1); // the relative SGPR path query is inconclusive
}

TEST(WaitcheckTest, AccumulatorCopyCorrelatesMaskedCountsWithAnOptionalSubtraction) {
  constexpr uint32_t nop = 0xbf800000, split_count = 0x3fffu;
  const auto check = [&](uint32_t first_branch, uint32_t first_mask, uint32_t between,
                         uint32_t late_mask, uint32_t after_mask, size_t expected) {
    const std::vector<uint32_t> words{
        0x860cff37,   first_mask,             // s_and_b32 s12, s55, mask
        0xbf06810c,                           // s_cmp_eq_u32 s12, 1
        first_branch, 0xe050d000, 0x0d0f0017, // buffer_load_dword v0, v23, s[60:63], s13
        between,      0x8608ff37, late_mask,  // s_and_b32 s8, s55, mask
        0x81888108,                           // s_sub_i32 s8, s8, 1
        after_mask,
        0xbf008008,               // s_cmp_eq_i32 s8, 0
        0xbf850001,               // s_cbranch_scc1 +1 (skip the drain)
        0xbf8c0f70,               // s_waitcnt vmcnt(0)
        0xd3d94000,   0x18000100, // v_accvgpr_write acc0, v0
        0xbf810000,               // s_endpgm
    };
    const auto image = waitcheck_test::make_gfx_code_object(words, EF_AMDGPU_MACH_AMDGCN_GFX942);
    const AmdGpuCodeObject object(image.data(), image.size());
    const auto report = analyze_waitcnts_for_kernel(object, ROCJITSU_CODE_ARCH_CDNA3, 0);
    ASSERT_TRUE(report.supported && report.complete) << report.analysis_error;
    ASSERT_EQ(report.diagnostics.size(), expected) << diagnostic_summary(report);
    if (expected) {
      EXPECT_EQ(report.diagnostics[0].producer_section_offset, 16u);
      EXPECT_EQ(report.diagnostics[0].reg.index, 0u);
      EXPECT_EQ(report.diagnostics[0].required_count, 0u);
    }
  };
  constexpr uint32_t bypass_equal = 0xbf85000c, bypass_unequal = 0xbf84000c;
  check(bypass_equal, split_count, nop, split_count, nop, 0);
  check(bypass_unequal, split_count, nop, split_count, nop, 1);
  check(bypass_equal, split_count, 0xbeb70081, split_count, nop, 1); // redefine s55
  check(bypass_equal, 0xffff, nop, split_count, nop, 1);
  check(bypass_equal, split_count, nop, split_count, 0xbe880080, 1); // redefine s8
  check(bypass_equal, split_count, nop, split_count, 0xbeb70080, 1); // redefine the AND root
}

TEST(WaitcheckTest, MatrixEpilogueRetainsSavedAndDecrementedTripCountParity) {
  constexpr uint32_t nop = 0xbf800000;
  const auto check = [&](uint32_t copy, uint32_t clobber, uint32_t branch, size_t expected) {
    const std::vector<uint32_t> words{
        copy,                   // save trip count
        0xe05c1000, 0x800e000a, // buffer_load_dwordx4 v[0:3], v10, s[56:59], 0 offen
        0x80858105,             // s_sub_u32 s5, s5, 1
        0xbf008205,             // s_cmp_eq_i32 s5, 2
        0xbf840003,             // s_cbranch_scc0 +3 (drain)
        clobber,
        0x861f8106, // s_and_b32 s31, s6, 1
        branch,
        0xbf8c0f70,             // s_waitcnt vmcnt(0)
        0xd3e18000, 0x0402010c, // v_mfma_f32_16x16x16_bf16 acc[0:3], v[12:13], v[0:1], acc[0:3]
        0xbf810000,             // s_endpgm
    };
    const auto image = waitcheck_test::make_gfx_code_object(words, EF_AMDGPU_MACH_AMDGCN_GFX942);
    const AmdGpuCodeObject object(image.data(), image.size());
    const auto report = analyze_waitcnts_for_kernel(object, ROCJITSU_CODE_ARCH_CDNA3, 0);
    ASSERT_TRUE(report.supported && report.complete) << report.analysis_error;
    ASSERT_EQ(report.diagnostics.size(), expected) << diagnostic_summary(report);
    if (expected) {
      EXPECT_EQ(report.diagnostics[0].reg.index, 0u);
      EXPECT_EQ(report.diagnostics[0].required_count, 0u);
    }
  };
  constexpr uint32_t saved = 0xbe860005, other = 0xbe860008;
  constexpr uint32_t skips_if_even = 0xbf840001, skips_if_odd = 0xbf850001;
  check(saved, nop, skips_if_even, 0);
  check(saved, nop, skips_if_odd, 1);
  check(other, nop, skips_if_even, 1);
  check(saved, 0xbe860080, skips_if_even, 1); // overwrite s6
}

TEST(WaitcheckTest, MatrixWaitUsesTheSavedAndDecrementedTripCountAtTheConsumer) {
  const auto check = [](uint32_t branch, uint32_t count, bool hazard, bool exact, bool unused) {
    const std::vector<uint32_t> words{
        0xbe860005, // s_mov_b32 s6, s5
        0xe05c1000,
        0x800e000a,          // buffer_load_dwordx4 v[0:3], v10, s[56:59], 0 offen
        0x80858105,          // s_sub_u32 s5, s5, 1
        0xbf008205,          // s_cmp_eq_i32 s5, 2
        0xbf840005,          // s_cbranch_scc0 exit
        0xbf8c0f70u | count, // s_waitcnt vmcnt(count)
        0x861f8106,          // s_and_b32 s31, s6, 1
        branch,              // test saved count before MFMA
        0xd3e18000,
        0x0402010c, // MFMA uses v[0:1]
        0xbf810000, // s_endpgm
    };
    const auto image = waitcheck_test::make_gfx_code_object(words, EF_AMDGPU_MACH_AMDGCN_GFX942);
    const AmdGpuCodeObject object(image.data(), image.size());
    WaitcheckOptions options;
    options.check_counter_parity = true;
    const auto report = analyze_waitcnts_for_kernel(object, ROCJITSU_CODE_ARCH_CDNA3, 0, options);
    ASSERT_TRUE(report.supported && report.complete) << report.analysis_error;
    EXPECT_EQ(!report.diagnostics.empty(), hazard);
    EXPECT_EQ(report.counter_parity_exact, exact ? 1u : 0u);
    EXPECT_EQ(report.counter_parity_indeterminate_groups, 0u);
    ASSERT_EQ(report.counter_underaccounting_diagnostics.size(), unused ? 1u : 0u);
    if (unused)
      EXPECT_FALSE(report.counter_underaccounting_diagnostics.front().has_required_dependency);
  };
  constexpr uint32_t skip_odd = 0xbf850002, skip_even = 0xbf840002;
  check(skip_odd, 0, false, false, true);
  check(skip_odd, 1, false, false, true);
  check(skip_even, 0, false, true, false);
  check(skip_even, 1, true, false, false);
}

TEST(WaitcheckTest, MatrixPipelineComparesYoungerIssueEqualityWithAnUnsignedBound) {
  auto check = [](uint32_t bound, uint32_t redefine, uint32_t exit, uint32_t wait, size_t ordinary,
                  size_t exact, bool modeled, bool byte_permute) {
    const std::vector<uint32_t> words{
        0xe05c1000,
        0x800e000a, // buffer_load_dwordx4 v[0:3], v10, s[56:59], 0 offen
        0xbf068105, // s_cmp_eq_u32 s5, 1
        0xbf850002, // s_cbranch_scc1 skip_load
        0xe05c1000,
        0x800e040b, // buffer_load_dwordx4 v[4:7], v11, s[56:59], 0 offen
        redefine,
        bound,              // s_cmp_le_u32 s5, 2, or with a different root/bound
        exit,               // branch to end
        0xbf8c0f70u | wait, // s_waitcnt vmcnt(wait)
        byte_permute ? 0xd1ed0020u : 0xd3e18000u,
        byte_permute ? 0x007e010cu : 0x0402010cu, // V_PERM_B32 or MFMA reads v0.
        0xbf810000,                               // s_endpgm
    };
    const auto image = waitcheck_test::make_gfx_code_object(words, EF_AMDGPU_MACH_AMDGCN_GFX942);
    const AmdGpuCodeObject object(image.data(), image.size());
    WaitcheckOptions options;
    options.check_counter_parity = true;
    const auto report = analyze_waitcnts_for_kernel(object, ROCJITSU_CODE_ARCH_CDNA3, 0, options);
    ASSERT_TRUE(report.supported && report.complete) << report.analysis_error;
    EXPECT_EQ(report.diagnostics.size(), ordinary);
    EXPECT_EQ(report.counter_parity_exact, exact);
    EXPECT_EQ(report.counter_parity_indeterminate_groups, 0u);
    ASSERT_EQ(report.counter_underaccounting_diagnostics.size(), modeled ? 1u : 0u);
    if (modeled)
      EXPECT_EQ(report.counter_underaccounting_diagnostics.front().required_count, 1u);
  };
  constexpr uint32_t le = 0xbf0b8205, nop = 0xbf800000, exit_small = 0xbf850003;
  for (const bool byte_permute : {false, true}) {
    SCOPED_TRACE(byte_permute);
    check(le, nop, exit_small, 0, 0, 0, true, byte_permute);
    check(le, nop, exit_small, 1, 0, 1, false, byte_permute);
    check(le, nop, exit_small, 2, 1, 0, false, byte_permute);
    // The consumer runs on the small trip count.
    check(le, nop, 0xbf840003, 1, 1, 0, false, byte_permute);
    check(0xbf0b8206, nop, exit_small, 1, 1, 0, false, byte_permute); // Unguarded s6.
    check(le, 0xbe850083, exit_small, 1, 1, 0, false, byte_permute);  // s_mov_b32 s5, 3.
    check(0xbf0b8005, nop, exit_small, 1, 1, 0, false,
          byte_permute); // A one-trip path exceeds zero.
  }
}

TEST(WaitcheckTest, Gfx942GuardedBytePermuteAccountsForVmcntCapacity) {
  for (uint32_t younger_count : {62u, 63u}) {
    SCOPED_TRACE(younger_count);
    std::vector<uint32_t> words{
        0xe05c1000,
        0x800e000a,                       // buffer_load_dwordx4 v[0:3], v10, s[56:59], 0 offen
        0xbf068105,                       // s_cmp_eq_u32 s5, 1
        0xbf850000 | (2 * younger_count), // s_cbranch_scc1 skip_loads
    };
    for (uint32_t i = 0; i < younger_count; ++i) {
      words.push_back(0xe05c1000);
      words.push_back(0x800e040b); // buffer_load_dwordx4 v[4:7], v11, s[56:59], 0 offen
    }
    const std::array tail{
        0xbf0b8205u,              // s_cmp_le_u32 s5, 2
        0xbf850002u,              // s_cbranch_scc1 end
        0xd1ed0020u, 0x007e010cu, // v_perm_b32 v32, v12, v0, exec_lo
        0xbf810000u,              // s_endpgm
    };
    words.insert(words.end(), tail.begin(), tail.end());
    const auto image = waitcheck_test::make_gfx_code_object(words, EF_AMDGPU_MACH_AMDGCN_GFX942);
    const AmdGpuCodeObject object(image.data(), image.size());
    const auto report = analyze_waitcnts_for_kernel(object, ROCJITSU_CODE_ARCH_CDNA3, 0);
    ASSERT_TRUE(report.supported && report.complete) << report.analysis_error;
    EXPECT_EQ(report.diagnostics.size(), younger_count == 62 ? 1u : 0u)
        << diagnostic_summary(report);
  }
}

TEST(WaitcheckTest, MatrixEpilogueChecksExcludedValuesBelowSmallUnrollLimit) {
  const auto check = [](uint32_t excluded, size_t expected) {
    const std::vector<uint32_t> words{
        0xbe860005,             // s_mov_b32 s6, s5
        0xbf068005,             // s_cmp_eq_u32 s5, 0
        0xbf85000c,             // s_cbranch_scc1 end
        excluded,               // s_cmp_eq_u32 s5, 1 or 4
        0xbf85000a,             // s_cbranch_scc1 end
        0xe05c1000, 0x800e000a, // buffer_load_dwordx4 v[0:3], v10, s[56:59], 0 offen
        0xbf0b8205,             // s_cmp_le_u32 s5, 2
        0xbf840002,             // s_cbranch_scc0 drain
        0x861f8106,             // s_and_b32 s31, s6, 1
        0xbf850001,             // s_cbranch_scc1 consume
        0xbf8c0f70,             // s_waitcnt vmcnt(0)
        0xd3e18000, 0x0402010c, // v_mfma_f32_16x16x16_bf16 acc[0:3], v[12:13], v[0:1], acc[0:3]
        0x80858105,             // s_sub_u32 s5, s5, 1
        0xbf810000,             // s_endpgm
    };
    const auto image = waitcheck_test::make_gfx_code_object(words, EF_AMDGPU_MACH_AMDGCN_GFX942);
    const AmdGpuCodeObject object(image.data(), image.size());
    const auto report = analyze_waitcnts_for_kernel(object, ROCJITSU_CODE_ARCH_CDNA3, 0);
    ASSERT_TRUE(report.supported && report.complete) << report.analysis_error;
    ASSERT_EQ(report.diagnostics.size(), expected) << diagnostic_summary(report);
  };
  check(0xbf068105, 0);
  check(0xbf068405, 1); // The odd incoming count of one reaches the unwaited consumer.
}

TEST(WaitcheckTest, Gfx950DoesNotTreatMfmaAccCdAccumulatorAsVgprUse) {
  std::vector<uint32_t> program;
  append_gfx950_ds_read_b128_v18_v12_offset64(program);
  append_gfx950_v_mfma_f32_16x16x32_bf16_acc16_acc_cd_1(program);

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA4);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx950StillTracksMfmaVgprAccumulatorUse) {
  std::vector<uint32_t> program;
  append_gfx950_ds_read_b128_v18_v12_offset64(program);
  append_gfx950_v_mfma_f32_16x16x32_bf16_v16_acc_cd_0(program);

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA4);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Ds);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Use);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::VGPR);
  EXPECT_EQ(report.diagnostics[0].reg.index, 18);
  EXPECT_EQ(report.diagnostics[0].required_count, 0u);
}

TEST(WaitcheckTest, Gfx950ReportsKnownInc0001LgkmcntTwoHazardInIsolation) {
  std::vector<uint32_t> program;
  append_gfx950_ds_read_b128_v82_v13_offset64(program);
  append_gfx950_ds_read_b128_v148_v13_offset1056(program);
  append_gfx950_ds_read_b128_v90_v13_offset1120(program);
  append_gfx950_s_waitcnt_lgkmcnt_2(program);
  append_gfx950_v_cvt_pk_bf16_f32_v86_v148_v149(program);
  append_gfx950_v_cvt_pk_bf16_f32_v88_v90_v91(program);

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA4);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 2u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Ds);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Use);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::VGPR);
  EXPECT_EQ(report.diagnostics[0].reg.index, 148u);
  EXPECT_EQ(report.diagnostics[0].required_count, 1u);
  EXPECT_EQ(report.diagnostics[1].counter, WaitCounterKind::Ds);
  EXPECT_EQ(report.diagnostics[1].access, WaitcheckAccessKind::Use);
  EXPECT_EQ(report.diagnostics[1].reg.cls, RegClass::VGPR);
  EXPECT_EQ(report.diagnostics[1].reg.index, 90u);
  EXPECT_EQ(report.diagnostics[1].required_count, 0u);
}

TEST(WaitcheckTest, Gfx950AcceptsKnownInc0001LgkmcntZeroFixInIsolation) {
  std::vector<uint32_t> program;
  append_gfx950_ds_read_b128_v82_v13_offset64(program);
  append_gfx950_ds_read_b128_v148_v13_offset1056(program);
  append_gfx950_ds_read_b128_v90_v13_offset1120(program);
  append_gfx950_s_waitcnt_lgkmcnt_0(program);
  append_gfx950_v_cvt_pk_bf16_f32_v86_v148_v149(program);
  append_gfx950_v_cvt_pk_bf16_f32_v88_v90_v91(program);

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA4);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx950DoesNotTrackGfx12VmVsrcSourceOverwrite) {
  std::vector<uint32_t> program;
  append_gfx950_buffer_load_dword_v0_v8_s0_offen(program);
  append_gfx950_v_mov_b32_v8_v10(program);

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA4);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx950BufferOffenUsesSingleVaddrRegister) {
  std::vector<uint32_t> program;
  append_gfx950_buffer_load_dword_v1_v8_s0_offen(program);
  append_gfx950_buffer_load_dword_v2_v0_s0_offen(program);

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA4);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx950BufferOffsetModeDoesNotReadEncodedVaddr) {
  std::vector<uint32_t> program;
  append_gfx950_buffer_load_dword_v0_v8_s0_offen(program);
  append_gfx950_buffer_load_dword_v1_off_s0(program);

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA4);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx950BufferResourceUsesPhysicalDescriptorRegister) {
  std::vector<uint32_t> program;
  append_gfx950_s_buffer_load_dwordx8_s0_s28_imm16(program);
  append_gfx950_buffer_load_dword_v1_v1_s24_offen(program);

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA4);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx950ReportsMissingLgkmcntBeforePhysicalBufferResourceUse) {
  std::vector<uint32_t> program;
  append_gfx950_s_buffer_load_dwordx8_s24_s28_imm16(program);
  append_gfx950_buffer_load_dword_v1_v1_s24_offen(program);

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA4);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Ds);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Use);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::SGPR);
  EXPECT_EQ(report.diagnostics[0].reg.index, 24u);
  EXPECT_EQ(report.diagnostics[0].required_count, 0u);
  EXPECT_NE(report.diagnostics[0].message.find("s_waitcnt lgkmcnt(0)"), std::string::npos);
}

TEST(WaitcheckTest, ReportsMissingWaitAluVmVsrcBeforeVmemSourceOverwrite) {
  std::vector<uint32_t> program;
  append_inst(program, global_load_b32(0)); // uses v8 as the vector offset.
  append_inst(program, v_mov_b32(8, 10));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].kind, WaitcheckDiagnosticKind::WaitCounter);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::VmVsrc);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Def);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::VGPR);
  EXPECT_EQ(report.diagnostics[0].reg.index, 8u);
  EXPECT_NE(report.diagnostics[0].message.find("depctr_vm_vsrc(0)"), std::string::npos);
}

TEST(WaitcheckTest, Gfx1250GlobalLoadWithSaddrUsesSingleVaddrRegister) {
  std::vector<uint32_t> program;
  append_global_load_b64_v8_v7_s0_offset16(program);
  append_global_load_b128_v2_v7_s0(program);

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA5);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, AcceptsWaitAluVmVsrcBeforeVmemSourceOverwrite) {
  std::vector<uint32_t> program;
  append_inst(program, global_load_b32(0));
  append_inst(program, s_wait_alu_vm_vsrc_0());
  append_inst(program, v_mov_b32(8, 10));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, AcceptsWaitXcntZeroBeforeVmemSourceOverwriteOnGfx1250) {
  std::vector<uint32_t> program;
  append_inst(program, global_load_b32(0));
  append_inst(program, s_wait_xcnt_0());
  append_inst(program, v_mov_b32(8, 10));

  auto report = analyze_gfx1250_normal(program);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx1250CodeObjectTracksXcntInNormalMode) {
  std::vector<uint32_t> program;
  append_inst(program, global_load_b32(0));
  append_inst(program, v_mov_b32(8, 10));

  auto report = analyze_gfx1250_normal(program);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::X);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Def);
  EXPECT_EQ(report.diagnostics[0].reg, (RegisterRef{RegClass::VGPR, 8, 1}));
  EXPECT_EQ(report.diagnostics[0].required_count, 0u);
}

TEST(WaitcheckTest, Gfx1250ReportsXcntBeforeExecOverwriteAfterVmemLoad) {
  std::vector<uint32_t> program;
  append_inst(program, global_load_b32(0));
  append_inst(program, s_or_b32_exec_lo(18));

  auto report = analyze_gfx1250_normal(program);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::X);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Def);
  EXPECT_EQ(report.diagnostics[0].reg, (RegisterRef{RegClass::EXEC, 0, 1}));
  EXPECT_EQ(report.diagnostics[0].required_count, 0u);
}

TEST(WaitcheckTest, Gfx1250ReportsXcntBeforeImplicitCmpxExecDef) {
  std::vector<uint32_t> program;
  append_inst(program, global_load_b32(0));
  // v_cmpx_ne_u16_e32 0, v1.h implicitly defines EXEC.
  append_inst(program, 0x7D7B0280u);

  auto report = analyze_gfx1250_normal(program);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::X);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Def);
  EXPECT_EQ(report.diagnostics[0].reg, (RegisterRef{RegClass::EXEC, 0, 1}));
  EXPECT_EQ(report.diagnostics[0].required_count, 0u);
}

TEST(WaitcheckTest, Gfx1250ReportsXcntBeforeExecOverwriteAfterVmemStore) {
  std::vector<uint32_t> program;
  append_inst(program, global_store_b32(4));
  append_inst(program, s_or_b32_exec_lo(29));

  auto report = analyze_gfx1250_normal(program);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::X);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Def);
  EXPECT_EQ(report.diagnostics[0].reg, (RegisterRef{RegClass::EXEC, 0, 1}));
  EXPECT_EQ(report.diagnostics[0].required_count, 0u);
}

TEST(WaitcheckTest, Gfx1250ReportsXcntBeforeSmemBaseOverwrite) {
  std::vector<uint32_t> program;
  append_inst(program, s_load_b32(4, 0));
  append_inst(program, s_mov_b32(0, 128));

  auto report = analyze_gfx1250_normal(program);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::X);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Def);
  EXPECT_EQ(report.diagnostics[0].reg, (RegisterRef{RegClass::SGPR, 0, 1}));
  EXPECT_EQ(report.diagnostics[0].required_count, 0u);
}

TEST(WaitcheckTest, Gfx1250AcceptsXcntZeroBeforeSmemBaseOverwrite) {
  std::vector<uint32_t> program;
  append_inst(program, s_load_b32(4, 0));
  append_inst(program, s_wait_xcnt(0));
  append_inst(program, s_mov_b32(0, 128));

  auto report = analyze_gfx1250_normal(program);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx1250TranslationAdmissionPreservesOnlyTheYoungerSources) {
  for (const bool store_first : {false, true}) {
    for (const unsigned intervening : {60u, 61u, 62u}) {
      for (const bool wait_load : {false, true}) {
        SCOPED_TRACE(testing::Message() << store_first << '/' << intervening << '/' << wait_load);
        std::vector<uint32_t> words;
        append_inst(words, s_setreg_imm32_b32(hwreg(1, 25, 1), 1));
        words.push_back(0xbefe0081u); // s_mov_b32 exec_lo, 1
        if (store_first)
          append_inst(words, global_store_b32(6, 8));
        append_inst(words, global_load_b32(3, 12, 2));
        if (!store_first)
          append_inst(words, global_store_b32(6, 8));
        for (unsigned i = 0; i < intervening; ++i) {
          if (i % 2 == 0) {
            append_inst(words, global_store_b32(16, 10, 4));
          } else {
            auto load = flat_load_b32(32 + i / 2);
            load.vaddr = 24;
            append_inst(words, load);
          }
        }
        append_inst(words, sopp(64, wait_load ? 0 : 63));
        append_inst(words, global_load_b32(12, 20, 6));
        append_inst(words, v_mov_b32(8, 28));
        const auto report = analyze_gfx1250_normal(words);
        ASSERT_TRUE(report.supported) << report.analysis_error;
        const bool outstanding = !store_first && intervening < 62;
        ASSERT_EQ(report.diagnostics.size(), outstanding) << diagnostic_summary(report);
        if (outstanding) {
          EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::X);
          EXPECT_EQ(report.diagnostics[0].reg, (RegisterRef{RegClass::VGPR, 8, 1}));
        }
      }
    }
  }
}

TEST(WaitcheckTest, Gfx1250PartialXcntRetiresOldestVmemTranslation) {
  std::vector<uint32_t> program;
  append_inst(program, global_load_b32(0, 8));
  append_inst(program, global_load_b32(1, 10));
  append_inst(program, s_wait_xcnt(1));
  append_inst(program, v_mov_b32(8, 12));

  auto report = analyze_gfx1250_normal(program);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx1250PartialXcntKeepsYoungestVmemTranslation) {
  std::vector<uint32_t> program;
  append_inst(program, global_load_b32(0, 8));
  append_inst(program, global_load_b32(1, 10));
  append_inst(program, s_wait_xcnt(1));
  append_inst(program, v_mov_b32(10, 12));

  auto report = analyze_gfx1250_normal(program);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::X);
  EXPECT_EQ(report.diagnostics[0].reg, (RegisterRef{RegClass::VGPR, 10, 1}));
  EXPECT_EQ(report.diagnostics[0].required_count, 0u);
}

TEST(WaitcheckTest, Gfx1250NonzeroXcntCannotRetireSmemTranslation) {
  std::vector<uint32_t> program;
  append_inst(program, s_load_b32(4, 0));
  append_inst(program, s_load_b32(5, 4));
  append_inst(program, s_wait_xcnt(1));
  append_inst(program, s_mov_b32(0, 128));

  auto report = analyze_gfx1250_normal(program);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::X);
  EXPECT_EQ(report.diagnostics[0].required_count, 0u);
}

TEST(WaitcheckTest, Gfx1250SmemAfterVmemImplicitlyDrainsVmemXcnt) {
  std::vector<uint32_t> program;
  append_inst(program, global_load_b32(0, 8));
  append_inst(program, s_load_b32(4, 0));
  append_inst(program, v_mov_b32(8, 12));

  auto report = analyze_gfx1250_normal(program);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx1250VmemAfterSmemImplicitlyDrainsSmemXcnt) {
  std::vector<uint32_t> program;
  append_inst(program, s_load_b32(12, 0));
  append_inst(program, global_load_b32(0, 8, 4));
  append_inst(program, s_mov_b32(0, 128));

  auto report = analyze_gfx1250_normal(program);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx1250LoadcntWaitAlsoRetiresVmemLoadXcnt) {
  std::vector<uint32_t> program;
  append_inst(program, global_load_b32(0, 8));
  append_inst(program, sopp(64, 0)); // s_wait_loadcnt 0
  append_inst(program, v_mov_b32(8, 12));

  auto report = analyze_gfx1250_normal(program);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx1250LoadcntWaitDoesNotRetireVmemStoreXcnt) {
  std::vector<uint32_t> program;
  append_inst(program, global_store_b32(4, 8));
  append_inst(program, sopp(64, 0)); // s_wait_loadcnt 0
  append_inst(program, v_mov_b32(8, 12));

  auto report = analyze_gfx1250_normal(program);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::X);
}

TEST(WaitcheckTest, Gfx1250FlatScratchTranslationFollowsItsPrimaryCounter) {
  const std::vector<uint32_t> scratch{
      0xd71f0001u, 0x020100c1u, // v_mbcnt_lo_u32_b32 v1, -1, 0
      0x30060294u,              // v_lshlrev_b32_e32 v3, 20, v1
      0x7e040284u,              // v_mov_b32_e32 v2, 4
      0x500804e6u,              // v_add_nc_u64_e32 v[4:5], src_flat_scratch_base_lo, v[2:3]
  };
  const auto check = [&](const std::vector<uint32_t> &prefix, const std::vector<uint32_t> &address,
                         const std::vector<uint32_t> &suffix, bool store, uint32_t wait,
                         bool expect_x, unsigned first_address_vgpr = 4) {
    std::vector<uint32_t> code;
    append_inst(code, s_setreg_imm32_b32(hwreg(1, 25, 1), 1));
    code.insert(code.end(), prefix.begin(), prefix.end());
    code.insert(code.end(), address.begin(), address.end());
    code.insert(code.end(), suffix.begin(), suffix.end());
    const size_t producer = code.size() * sizeof(uint32_t);
    if (store)
      code.insert(code.end(), {0xec06807cu, 0x04000000u, 0x00000004u}); // flat_store_b32 v[4:5], v8
    else
      code.insert(code.end(),
                  {0xec05007cu, 0x00000008u, first_address_vgpr}); // flat_load_b32 v8, v[4:5]
    if (wait)
      code.push_back(wait);
    const size_t consumer = code.size() * sizeof(uint32_t);
    code.push_back(0x7e080280u); // v_mov_b32_e32 v4, 0
    append_inst(code, s_endpgm());
    const auto image =
        waitcheck_test::make_gfx_code_object(code, EF_AMDGPU_MACH_AMDGCN_GFX1250, /*wave32=*/true);
    AmdGpuCodeObject object(image.data(), image.size());
    const auto report = analyze_waitcnts_for_kernel(object, ROCJITSU_CODE_ARCH_CDNA5, 0);
    ASSERT_TRUE(report.supported && report.complete) << report.analysis_error;
    EXPECT_EQ(std::ranges::any_of(report.diagnostics,
                                  [&](const auto &diagnostic) {
                                    return diagnostic.counter == WaitCounterKind::X &&
                                           diagnostic.producer_section_offset == producer &&
                                           diagnostic.section_offset == consumer;
                                  }),
              expect_x)
        << diagnostic_summary(report);
  };
  constexpr uint32_t load_zero = 0xbfc00000u, store_zero = 0xbfc10000u;
  check({}, scratch, {}, false, load_zero, false);
  check({}, scratch, {}, true, store_zero, false);
  check({}, scratch, {}, false, 0, true);
  check({}, scratch, {}, false, store_zero, true);
  check({}, scratch, {}, true, load_zero, true);
  check({}, scratch, {}, false, load_zero, true, 3); // v4 is the low result, used as a high tag
  for (bool shift_in_y : {false, true}) {
    // Extended VOPD3 shifts in native gfx1250 PyTorch complex math kernels.
    auto address = scratch;
    address.erase(address.begin() + 2);
    const std::array<uint32_t, 3> shift =
        shift_in_y ? std::array<uint32_t, 3>{0xcf551090, 0x00020094, 0x03000100}
                   : std::array<uint32_t, 3>{0xcf455094, 0x00010090, 0x00000203};
    address.insert(address.begin() + 2, shift.begin(), shift.end());
    check({}, address, {}, false, load_zero, false);
    check({}, address, {}, true, store_zero, false);
    check({}, address, {}, false, store_zero, true);
    auto modified = address;
    modified[3] |= 1u << (shift_in_y ? 12 : 9);
    check({}, modified, {}, false, load_zero, true);
    auto wrong_count = address;
    ++wrong_count[shift_in_y ? 3 : 2];
    check({}, wrong_count, {}, false, load_zero, true);
    auto wrong_word = address;
    wrong_word[4] ^= 2u << (shift_in_y ? 24 : 0);
    check({}, wrong_word, {}, false, load_zero, true);
    auto uninitialized_lane = address;
    uninitialized_lane.insert(uninitialized_lane.begin() + 5,
                              0xbefe00c1u); // widen EXEC after shift
    check({}, uninitialized_lane, {}, false, load_zero, true);
  }
  const auto offset_with_masks = [&](std::vector<uint32_t> masks) {
    auto address = scratch;
    address.insert(address.end() - 1, masks.begin(), masks.end());
    return address;
  };
  // Save the incoming EXEC, narrow it, then restore lanes using a mask derived
  // from the same incoming EXEC. The prologue initialized every returning lane.
  check({}, offset_with_masks({0xbe8a007eu, 0xbe8b2009u, 0xbe8c300bu, 0x8c7e0b7eu}),
        {}, false, load_zero, false);
  check({}, offset_with_masks({0xbe8a007eu, 0xbe8b2009u, 0xbefe000au}),
        {}, false, load_zero, false);
  // A caller-supplied mask can introduce previously inactive lanes.
  check({}, offset_with_masks({0xbe8a007eu, 0xbe8b2009u, 0x8c0b090bu, 0x8c7e0b7eu}),
        {}, false, load_zero, true);
  check({}, offset_with_masks({0xbe8a007eu, 0xbe8b2009u, 0xbe8a00c1u, 0xbefe000au}),
        {}, false, load_zero, true);
  check({}, offset_with_masks({0xbe8b2009u, 0x7e060282u, 0x8c7e0b7eu}),
        {}, false, load_zero, true);
  // The incoming mask may also cover the completed address pair itself.
  check({}, scratch, {0xbe8a007eu, 0xbe8b2009u, 0xbefe000au}, false, load_zero, false);
  check({}, scratch, {0xbe8a007eu, 0xbe8b2009u, 0xbefe0009u}, false, load_zero, true);
  check({}, scratch, {0xbe8b2009u, 0x7e0a0282u, 0x8c7e0b7eu}, false, load_zero, true);
  // The shift itself may execute after divergence. A saved mask from its
  // current region can restore its own lanes, a mask saved before it can add others.
  const std::vector<uint32_t> narrowed{0xbe8a007eu, 0xbe8b2009u};
  check(narrowed, offset_with_masks({0xbe8c007eu, 0xbe8d2009u, 0xbefe000cu}),
        {}, false, load_zero, false);
  check(narrowed, offset_with_masks({0xbe8c007eu, 0xbe8d2009u, 0xbefe000au}),
        {}, false, load_zero, true);
  // Preserve a mask saved before this writer when no intervening instruction
  // changed EXEC. A pre-narrowing mask and a clobbered SGPR remain unknown.
  auto saved_before_shift = scratch;
  saved_before_shift.insert(saved_before_shift.begin() + 2, 0xbe8c007eu);
  saved_before_shift.insert(saved_before_shift.end() - 1,
                            {0x7d940480u, 0x8d0b0c7eu, 0x8c7e0b7eu});
  check(narrowed, saved_before_shift, {}, false, load_zero, false);
  auto clobbered_save = saved_before_shift;
  clobbered_save.insert(clobbered_save.begin() + 3, 0xbe8c00c1u);
  check(narrowed, clobbered_save, {}, false, load_zero, true);
  // The call returns past a branch over its body. Certify v3 only when
  // every returning path preserves that dword and the incoming lane set.
  const auto through_call = [&](std::uint32_t body, bool hazard) {
    auto address = scratch;
    const std::array<std::uint32_t, 4> helper{
        build_s_call_b64(30, 1, ROCJITSU_CODE_ARCH_CDNA5).value(),
        0xbfa00002u, body, 0xbe80481eu};
    address.insert(address.end() - 1, helper.begin(), helper.end());
    check({}, address, {}, false, load_zero, hazard);
  };
  through_call(0x7e0c0280u, false); // v_mov_b32 v6, 0
  through_call(0x7e060280u, true);  // v_mov_b32 v3, 0
  through_call(0xbefe00c1u, true);  // s_mov_b32 exec_lo, -1
  // In wave32 a write to EXEC_HI does not modify the mask that issues VALU.
  check({}, offset_with_masks({0xbe8a007eu, 0xbefe0082u, 0xbeff000au}),
        {}, false, load_zero, true);
  check({}, offset_with_masks({0xbe8a007eu, 0xbeff0082u, 0xbefe000au}),
        {}, false, load_zero, false);
  check({}, offset_with_masks({0xbeff0082u, 0xbe8b007fu, 0xbefe000bu}),
        {}, false, load_zero, true);
  check({}, offset_with_masks({0xbe8a007eu, 0xbe8a0182u, 0xbefe000au}),
        {}, false, load_zero, true);
  // An AND sourced from EXEC_HI can enable low lanes that missed the shift.
  check({0xbeff0082u}, offset_with_masks({0x8b7ec17fu}), {}, false, load_zero, true);
  check({0xbeff0082u}, offset_with_masks({0x8b7ec17eu}), {}, false, load_zero, false);
  // Both arms construct private addresses. The restored lanes may keep
  // either definition; an opaque address word blocks that conclusion.
  auto arm = scratch;
  arm.insert(arm.end() - 1, 0xbe890082u); // s_mov_b32 s9, 2
  arm.insert(arm.end(), {0xbe8a2009u, 0xbfa50001u, 0x500804e6u, 0x8c7e0a7eu});
  check({}, arm, {}, false, load_zero, false);
  check({}, arm, {}, false, 0xbfc00001u, true);
  auto opaque = arm;
  opaque[opaque.size() - 2] = 0x7e0a020fu; // v_mov_b32 v5, s15
  check({}, opaque, {}, false, load_zero, true);
  // The b64 result replaces its address pair in active lanes. Only the
  // complement of those same lanes may keep the older address.
  arm.push_back(0x8b7e817eu); // s_and_b32 exec_lo, exec_lo, 1
  arm.insert(arm.end(), {0xec05407cu, 0x00000004u, 0x00000004u, load_zero, 0xbfc50000u});
  check({}, arm, {0x8c7e807eu, 0xbe8b300au}, false, load_zero, false);
  check({}, arm, {0x8b7e817eu}, false, load_zero, true);
  auto unknown_high = scratch;
  unknown_high[2] = 0x7e06020fu; // v_mov_b32_e32 v3, s15
  check({}, unknown_high, {}, false, load_zero, true);
  check({}, scratch, {0x8b7e817eu}, false, load_zero, false); // narrow EXEC after address
  check({}, scratch, {0x7e080280u}, false, load_zero, false); // low word cannot change the aperture
  check({}, scratch, {0x7e0a0282u}, false, load_zero, true);  // overwrite high word with LDS tag
  check({0x7e0a0282u, 0xbf068009u, 0xbfa20005u}, scratch, {}, false, load_zero, true);
  check({0x7e0a0282u, 0xbefe0080u}, scratch, {0xbefe0081u}, false, load_zero, true);
}

TEST(WaitcheckTest, Gfx1250PrivateToFlatCompilerLoweringsExcludeTheLdsAperture) {
  // LLVM private-to-flat addrspacecast, using SelectionDAG or GlobalISel.
  const std::vector<uint32_t> sdag{
      0xd71f0000, 0x020100c1, // v_mbcnt_lo_u32_b32 v0, -1, 0
      0xca220002, 0x00000094, // v_dual_mov_b32 v0, s2 :: v_dual_lshlrev_b32 v1, 20, v0
      0x500000e6,             // v_add_nc_u64_e32 v[0:1], src_flat_scratch_base_lo, v[0:1]
  };
  const std::vector<uint32_t> gisel{
      0x7e003ae6,             // v_mov_b64_e32 v[0:1], src_flat_scratch_base_lo
      0xd71f0002, 0x020100c1, // v_mbcnt_lo_u32_b32 v2, -1, 0
      0x30040494,             // v_lshlrev_b32_e32 v2, 20, v2
      0xd7006a00, 0x02020002, // v_add_co_u32 v0, vcc_lo, s2, v0
      0xd5207c01, 0x01aa0302, // v_add_co_ci_u32_e64 v1, null, v2, v1, vcc_lo
  };
  const auto check = [&](const std::vector<uint32_t> &address, std::initializer_list<uint32_t> mask,
                         unsigned load_wait, bool expect_x) {
    std::vector<uint32_t> program;
    append_inst(program, s_setreg_imm32_b32(hwreg(1, 25, 1), 1));
    program.insert(program.end(), address.begin(), address.end());
    program.insert(program.end(), mask.begin(), mask.end());
    const size_t producer = program.size() * sizeof(uint32_t);
    program.insert(program.end(), {0xec05007c, 0x000c0000, 0}); // flat_load_b32 v0, v[0:1]
    append_inst(program, sopp(64, load_wait));
    const size_t consumer = program.size() * sizeof(uint32_t);
    program.push_back(0x7e020280); // v_mov_b32_e32 v1, 0
    append_inst(program, s_endpgm());
    const auto image =
        waitcheck_test::make_gfx_code_object(program, EF_AMDGPU_MACH_AMDGCN_GFX1250, true);
    AmdGpuCodeObject object(image.data(), image.size());
    const auto report = analyze_waitcnts_for_kernel(object, ROCJITSU_CODE_ARCH_CDNA5, 0);
    ASSERT_TRUE(report.supported && report.complete) << report.analysis_error;
    EXPECT_EQ(std::ranges::any_of(report.diagnostics,
                                  [&](const auto &d) {
                                    return d.counter == WaitCounterKind::X &&
                                           d.producer_section_offset == producer &&
                                           d.section_offset == consumer;
                                  }),
              expect_x)
        << diagnostic_summary(report);
  };
  check(sdag, {}, 0, false);
  check(gisel, {}, 0, false);
  check(sdag, {0xca520280, 0x01000080}, 0, false);  // CNDMASK of zero vs scratch high in X
  check(gisel, {0xca520080, 0x00000280}, 0, false); // CNDMASK of zero vs scratch high in Y
  check(sdag, {0xca520282, 0x01000080}, 0, true);   // X may select high word 2
  check(gisel, {0xca520080, 0x00000282}, 0, true);  // Y may select high word 2
  check(gisel, {0x7e020282}, 0, true);              // Clobbered high word
  check(gisel, {}, 1, true);                        // The primary counter still needs zero
  auto wrong_shift = sdag;
  wrong_shift[3] = 0x00000093; // Shift by 19 is outside the matched compiler form
  check(wrong_shift, {}, 0, true);
  auto unknown_base = gisel;
  unknown_base[0] = 0x7e003a14; // v_mov_b64_e32 v[0:1], s[20:21]
  check(unknown_base, {}, 0, true);
  auto unknown_offset = gisel;
  unknown_offset[7] = 0x01aa0303; // Carry-add reads v3 instead of the proven v2
  check(unknown_offset, {}, 0, true);
}

TEST(WaitcheckTest, Gfx1250StoresRetireTranslationsByTheirOwnFifoPosition) {
  for (unsigned younger_stores : {0u, 1u}) {
    SCOPED_TRACE(younger_stores);
    std::vector<uint32_t> program;
    append_inst(program, s_setreg_imm32_b32(hwreg(1, 25, 1), 1));
    append_inst(program, global_atomic_add_u32(4));
    // An interleaved ordinary load ages X but cannot age the atomic in STORE.
    append_inst(program, global_load_b32(20, 16));
    if (younger_stores)
      append_inst(program, global_store_b32(10, 18));
    append_inst(program, sopp(65, 1)); // s_wait_storecnt 1
    append_inst(program, v_mov_b32(4, 12));

    const auto report = analyze_gfx1250_normal(program);
    ASSERT_TRUE(report.supported) << report.analysis_error;
    ASSERT_EQ(report.diagnostics.size(), younger_stores ? 0u : 1u) << diagnostic_summary(report);
    if (!younger_stores)
      EXPECT_EQ(report.diagnostics.front().counter, WaitCounterKind::X);
  }
}

TEST(WaitcheckTest, Gfx1250FullStoreCompletionReleasesAtomicReplaySources) {
  std::vector<uint32_t> program;
  append_inst(program, s_setreg_imm32_b32(hwreg(1, 25, 1), 1));
  append_inst(program, global_atomic_add_u32(4));
  append_inst(program, sopp(65, 0)); // s_wait_storecnt 0
  append_inst(program, v_mov_b32(4, 12));
  const auto report = analyze_gfx1250_normal(program);
  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx1250KmcntZeroAlsoRetiresSmemXcnt) {
  std::vector<uint32_t> program;
  append_inst(program, s_load_b32(4, 0));
  append_inst(program, sopp(71, 0)); // s_wait_kmcnt 0
  append_inst(program, s_mov_b32(0, 128));

  auto report = analyze_gfx1250_normal(program);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx1250SetVgprMsbImplicitlyDrainsXcnt) {
  std::vector<uint32_t> program;
  append_inst(program, global_load_b32(0, 8));
  append_inst(program, s_set_vgpr_msb(0));
  append_inst(program, v_mov_b32(8, 12));

  auto report = analyze_gfx1250_normal(program);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx1250ModeSetregUpdatesVgprMsbAndDropsImmediateSetVgprMsb) {
  constexpr uint32_t kHwRegMode = 1;
  // MODE stores dst/src0/src1/src2, so 0x04 corresponds to the
  // S_SET_VGPR_MSB layout src0=1, src1=src2=dst=0.
  constexpr uint32_t kModeSrc0V256 = 0x04u << 12u;
  std::vector<uint32_t> program;
  append_inst(program, s_setreg_imm32_b32(hwreg(kHwRegMode, 0, 16), kModeSrc0V256));
  append_inst(program, s_set_vgpr_msb(0)); // Dropped by the gfx1250 hazard.
  append_inst(program, global_load_b32(0, 8));
  append_inst(program, v_mov_b32(8, 12));

  auto report = analyze_gfx1250_normal(program);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, ModeSetregBankWritesUseConcreteGpuSemantics) {
  // An AOTriton spill saves v0 before a writelane whose encoded v0 may
  // resolve to v256 after an immediate write to MODE[25].
  for (const auto machine : {EF_AMDGPU_MACH_AMDGCN_GFX1250, EF_AMDGPU_MACH_AMDGCN_GFX1251}) {
    const auto check = [&](std::vector<uint32_t> program, bool hazard) {
      for (uint32_t word : {0xed0680fcu, 0u, 0x0002a800u, // scratch_store_b32 off,v0,off offset:680
                            0xd7610000u, 0x02010205u,     // v_writelane_b32 v0,s5,1
                            0xbfb00000u})                 // s_endpgm
        program.push_back(word);
      const auto image = waitcheck_test::make_gfx_code_object(program, machine, true);
      AmdGpuCodeObject object(image.data(), image.size());
      const auto report = analyze_waitcnts(object, ROCJITSU_CODE_ARCH_CDNA5);
      ASSERT_TRUE(report.supported && report.complete) << report.analysis_error;
      ASSERT_EQ(report.diagnostics.size(), hazard ? 1u : 0u) << diagnostic_summary(report);
      if (hazard) {
        EXPECT_EQ(report.diagnostics.front().counter, WaitCounterKind::X);
        EXPECT_EQ(report.diagnostics.front().reg, (RegisterRef{RegClass::VGPR, 0, 1}));
      }
    };
    for (unsigned bit : {24u, 25u}) {
      SCOPED_TRACE(testing::Message() << "machine=" << machine << " mode bit=" << bit);
      std::vector<uint32_t> program;
      append_inst(program, s_setreg_imm32_b32(hwreg(1, bit, 1), 0x1001));
      check(program, machine == EF_AMDGPU_MACH_AMDGCN_GFX1251);
      append_inst(program, s_set_vgpr_msb(0));
      check(program, machine == EF_AMDGPU_MACH_AMDGCN_GFX1251); // Dropped on gfx1250.
      program.clear();
      append_inst(program, s_setreg_imm32_b32(hwreg(1, bit, 1), 1));
      check(program, true);
      program.clear();
      append_inst(program, s_set_vgpr_msb(vgpr_msb_mode(0, 0, 0, 1)));
      program.push_back(0xb9070000u | hwreg(1, bit, 1));        // s_setreg_b32 MODE field,s7
      check(program, machine == EF_AMDGPU_MACH_AMDGCN_GFX1250); // Unknown source may select v0.
    }
  }
}

TEST(WaitcheckTest, CodeObjectSelectsConcreteDecoderFromCallerRegistry) {
  constexpr auto arch = ROCJITSU_CODE_ARCH_CDNA5;
  // v_pk_fma_f64 v[4:7], v[8:11], v[12:15], v[16:19]; s_endpgm.
  const std::vector<uint32_t> program{0xcc3b4004u, 0x1c421908u, 0xbfb00000u};
  const auto image = waitcheck_test::make_gfx_code_object(program, EF_AMDGPU_MACH_AMDGCN_GFX1251);
  AmdGpuCodeObject object(image.data(), image.size());
  ASSERT_EQ(object.target_id(), ROCJITSU_CODE_TARGET_GFX1251);
  const auto report = analyze_waitcnts(object, arch);
  ASSERT_TRUE(report.passed()) << report.analysis_error;
  EXPECT_EQ(report.instructions_analyzed, 2u);

  // Raw input uses the registry's architecture default. Only gfx1251 decodes the FMA.
  EXPECT_FALSE(analyze_waitcnts(program, arch).supported);
  const auto *cdna5 = default_isa_target_registry().find(arch);
  ASSERT_NE(cdna5, nullptr);
  auto descriptor = *cdna5;
  descriptor.default_gpu_target = ROCJITSU_CODE_TARGET_GFX1251;
  const std::array<IsaTargetDescriptor, 1> targets{descriptor};
  const IsaTargetRegistry registry(targets);
  WaitcheckOptions options;
  options.registry = registry;
  EXPECT_TRUE(analyze_waitcnts(program, arch, options).passed());

  const std::array<IsaTargetDescriptor, 0> no_targets{};
  const IsaTargetRegistry empty_registry(no_targets);
  options.registry = empty_registry;
  EXPECT_FALSE(analyze_waitcnts(object, arch, options).supported);
}

TEST(WaitcheckTest, Gfx1250NopMakesSetVgprMsbAfterModeSetregEffective) {
  constexpr uint32_t kHwRegMode = 1;
  constexpr uint32_t kModeSrc0V256 = 0x04u << 12u;
  std::vector<uint32_t> program;
  append_inst(program, s_setreg_imm32_b32(hwreg(kHwRegMode, 0, 16), kModeSrc0V256));
  append_inst(program, sopp(0, 0)); // s_nop 0
  append_inst(program, s_set_vgpr_msb(0));
  append_inst(program, global_load_b32(0, 8));
  append_inst(program, v_mov_b32(8, 12));

  auto report = analyze_gfx1250_normal(program);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::X);
  EXPECT_EQ(report.diagnostics[0].reg, (RegisterRef{RegClass::VGPR, 8, 1}));
}

TEST(WaitcheckTest, Gfx1250SetregImplicitlyDrainsXcnt) {
  constexpr uint32_t kHwRegStatus = 2;
  std::vector<uint32_t> program;
  append_inst(program, global_load_b32(0, 8));
  append_inst(program, s_setreg_imm32_b32(hwreg(kHwRegStatus, 0, 1), 0));
  append_inst(program, v_mov_b32(8, 12));

  auto report = analyze_gfx1250_normal(program);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx1250ConditionalBranchImplicitlyDrainsXcnt) {
  std::vector<uint32_t> program;
  append_inst(program, global_load_b32(0, 8));
  append_inst(program, sopp(33, 0)); // s_cbranch_scc0 to the fallthrough.
  append_inst(program, v_mov_b32(8, 12));

  auto report = analyze_gfx1250_normal(program);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx1250ReportsXcntBeforeVmemStoreDataOverwrite) {
  std::vector<uint32_t> program;
  append_inst(program, global_store_b32(4, 8));
  append_inst(program, v_mov_b32(4, 12));

  auto report = analyze_gfx1250_normal(program);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::X);
  EXPECT_EQ(report.diagnostics[0].reg, (RegisterRef{RegClass::VGPR, 4, 1}));
}

TEST(WaitcheckTest, Gfx1250ReportsXcntBeforeVmemAtomicDataOverwrite) {
  std::vector<uint32_t> program;
  append_inst(program, global_atomic_add_u32(4));
  append_inst(program, v_mov_b32(4, 12));

  auto report = analyze_gfx1250_normal(program);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::X);
  EXPECT_EQ(report.diagnostics[0].reg, (RegisterRef{RegClass::VGPR, 4, 1}));
}

TEST(WaitcheckTest, Gfx1250LoadcntRetiresReturningVmemAtomicXcnt) {
  std::vector<uint32_t> program;
  append_inst(program, global_atomic_add_u32(4, 8, 124, true));
  append_inst(program, sopp(64, 0)); // s_wait_loadcnt 0
  append_inst(program, v_mov_b32(8, 12));

  auto report = analyze_gfx1250_normal(program);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx1250SingleAccessAtomicsUseOutstandingTranslation) {
  // The returning atomic's destination can implicitly order an older address,
  // and two RMWs following a drain do not require a wait between them.
  for (bool returning : {false, true}) {
    for (bool overlaps_address : {false, true}) {
      for (int wait : {-1, 0, 1, 2}) {
        SCOPED_TRACE(testing::Message() << returning << overlaps_address << wait);
        std::vector<uint32_t> program;
        append_inst(program, global_load_b32(14, 8));
        if (wait == 2)
          append_inst(program, sopp(64, 0)); // LOADcnt(0)
        else if (wait >= 0)
          append_inst(program, s_wait_xcnt(wait));
        append_inst(program, global_atomic_add_u32(overlaps_address ? 8 : 6, 16, 124, returning));
        append_inst(program, global_atomic_add_u32(12, 20));
        const auto report = analyze_gfx1250_normal(program);
        ASSERT_TRUE(report.supported) << report.analysis_error;
        // On an unsafe path the older load stays live at both atomics.
        const bool unsafe = wait != 0 && wait != 2 && !(returning && overlaps_address);
        EXPECT_EQ(report.diagnostics_observed, unsafe ? 2u : 0u) << diagnostic_summary(report);
        for (const auto &diag : report.diagnostics) {
          EXPECT_EQ(diag.counter, WaitCounterKind::X);
          EXPECT_EQ(diag.access, WaitcheckAccessKind::MemoryOrder);
          EXPECT_EQ(diag.producer_section_offset, 0u);
        }
      }
    }
  }
}

TEST(WaitcheckTest, RelativeWaitSelectorIsAReadBeforeItsCounterIsApplied) {
  using amdgpu::WaitCounterPolicy;
  for (auto arch : {ROCJITSU_CODE_ARCH_RDNA1, ROCJITSU_CODE_ARCH_RDNA2}) {
    auto decoder = Decoder::create(arch);
    ASSERT_NE(decoder, nullptr);
    constexpr uint32_t load_words[] = {0xf4000100u, 0xfa000004u}; // s_load_dword s4, s[0:1], 4
    std::unique_ptr<Instruction> load(decode_valid(*decoder, load_words));
    ASSERT_NE(load, nullptr);
    const InstDefUse producer(*load, 32);
    for (uint32_t selector : {4u, 5u, 125u}) {
      const uint32_t word = 0xbd000000u | (selector << 16); // s_waitcnt_lgkmcnt sN, 0
      std::unique_ptr<Instruction> wait(decode_valid(*decoder, &word));
      ASSERT_NE(wait, nullptr);
      const Operand *source = WaitCounterPolicy::relative_wait_operand(*wait, arch);
      if (selector == 125) {
        EXPECT_EQ(source, nullptr); // NULL is an immediate zero base.
        continue;
      }
      ASSERT_NE(source, nullptr);
      const auto reg = source->to_register_ref();
      ASSERT_TRUE(reg);
      EXPECT_TRUE(InstDefUse(*wait, 32).uses.contains(*reg));
      EXPECT_EQ(producer.defs.contains(*reg), selector == 4);
    }
  }
}

TEST(WaitcheckTest, Gfx1250FlatResultNeedsAVmemRouteToOrderTranslation) {
  // The returning dword overlaps the older load's address. Unknown FLAT may
  // return through LDS; a lane-private scratch dword uses VMEM instead.
  const std::vector<uint32_t> scratch{
      0xd71f0001u, 0x020100c1u, // v_mbcnt_lo_u32_b32 v1, -1, 0
      0x30060294u,              // v_lshlrev_b32_e32 v3, 20, v1
      0x7e040284u,              // v_mov_b32_e32 v2, 4
      0x500804e6u,              // v_add_nc_u64_e32 v[4:5], src_flat_scratch_base_lo, v[2:3]
  };
  for (bool private_address : {false, true}) {
    for (bool atomic : {false, true}) {
      for (int wait : {-1, 0, 1}) {
        SCOPED_TRACE(testing::Message() << private_address << ' ' << atomic << ' ' << wait);
        std::vector<uint32_t> code;
        append_inst(code, s_setreg_imm32_b32(hwreg(1, 25, 1), 1));
        if (private_address)
          code.insert(code.end(), scratch.begin(), scratch.end());
        const auto producer = code.size() * sizeof(uint32_t);
        append_inst(code, global_load_b32(14, 8));
        if (wait >= 0)
          append_inst(code, s_wait_xcnt(wait));
        const auto consumer = code.size() * sizeof(uint32_t);
        // flat_atomic_add_u32 v8, v[4:5], v3 th:TH_ATOMIC_RETURN
        if (atomic)
          code.insert(code.end(), {0xec0d407cu, 0x01900008u, 0x00000004u});
        else // flat_load_b32 v8, v[4:5]
          code.insert(code.end(), {0xec05007cu, 8u, 4u});
        append_inst(code, s_endpgm());
        const auto image =
            waitcheck_test::make_gfx_code_object(code, EF_AMDGPU_MACH_AMDGCN_GFX1250, true);
        AmdGpuCodeObject object(image.data(), image.size());
        WaitcheckOptions options;
        options.check_counter_parity = true;
        const auto report =
            analyze_waitcnts_for_kernel(object, ROCJITSU_CODE_ARCH_CDNA5, 0, options);
        ASSERT_TRUE(report.supported && report.complete) << report.analysis_error;
        const bool found = std::ranges::any_of(report.diagnostics, [&](const auto &diagnostic) {
          return diagnostic.counter == WaitCounterKind::X &&
                 diagnostic.producer_section_offset == producer &&
                 diagnostic.section_offset == consumer;
        });
        EXPECT_EQ(found, !private_address && wait != 0) << diagnostic_summary(report);
        if (wait == 0)
          EXPECT_EQ(report.counter_parity_exact, private_address ? 0u : 1u);
      }
    }
  }
}

TEST(WaitcheckTest, Gfx1250JoinedAtomicKeepsUnorderedIncomingSource) {
  for (bool scalar_first_path : {false, true}) {
    std::vector<uint32_t> program;
    // Only the fallthrough can issue the first VMEM. The branch itself drains
    // translation; a second candidate after the join is always visible.
    append_inst(program, sopp(34, 3)); // s_cbranch_scc1 over the load
    append_inst(program, global_load_b32(14, 8));
    if (scalar_first_path)
      append_inst(program, s_load_b32(36, 2));
    append_inst(program, global_load_b32(20, 16));
    append_inst(program, global_atomic_add_u32(8, 24, 124, true));
    const auto report = analyze_gfx1250_normal(program);
    ASSERT_TRUE(report.supported) << report.analysis_error;
    ASSERT_EQ(report.diagnostics_observed, 1u) << diagnostic_summary(report);
    EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::X);
    EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::MemoryOrder);
  }
}

TEST(WaitcheckTest, Gfx12ReturningAtomicsRetireOlderLoadsAtCapacity) {
  for (const auto [arch, machine] :
       {std::pair{ROCJITSU_CODE_ARCH_CDNA5, EF_AMDGPU_MACH_AMDGCN_GFX1250},
        std::pair{ROCJITSU_CODE_ARCH_RDNA4, EF_AMDGPU_MACH_AMDGCN_GFX1201}}) {
    for (bool buffer : {false, true}) {
      for (bool out_of_range : {false, true}) {
        SCOPED_TRACE(testing::Message() << machine << '/' << buffer << '/' << out_of_range);
        std::vector<uint32_t> program;
        append_inst(program, global_load_b32(9, 8));
        for (unsigned i = 0; i < 62; ++i)
          append_inst(program, global_load_b32(0, 8));
        const uint32_t dest = out_of_range ? 16 : 15;
        if (buffer)
          append_inst(program, buffer_atomic_add_u32(dest));
        else
          append_inst(program, global_atomic_add_u32(dest, 8, 124, true));
        const uint64_t consumer = program.size() * sizeof(uint32_t);
        append_inst(program, v_mov_b32(3, 9));
        append_inst(program, s_endpgm());
        const auto report = analyze_gfx12_with_vgpr_allocation(program, machine, arch, true, 16);
        ASSERT_TRUE(report.supported && report.complete) << report.analysis_error;
        const bool older_load_survives =
            std::ranges::any_of(report.diagnostics, [&](const auto &d) {
              return d.counter == WaitCounterKind::Load && d.producer_section_offset == 0 &&
                     d.section_offset == consumer && d.reg.index == 9;
            });
        EXPECT_FALSE(older_load_survives) << diagnostic_summary(report);
      }
    }
  }
}

TEST(WaitcheckTest, GenericGfx1250FixtureKeepsAllocationBoundsSeparate) {
  const auto image = waitcheck_test::make_gfx1250_code_object({0xbfb00000u});
  const AmdGpuCodeObject object(image.data(), image.size());
  const auto found = scan_kernel_descriptors(image, object.text_sections().front()->sectionOffset(),
                                             object.text_sections().front()->size(),
                                             object.text_sections().front()->sectionHeaderIndex());
  EXPECT_EQ(found.size(), 2u);
  const auto kernels = waitcheck_kernels(object);
  ASSERT_EQ(kernels.size(), 1u);
  EXPECT_EQ(kernels[0].min_vgprs, 16u);
  EXPECT_EQ(kernels[0].max_vgprs, 1024u);
  EXPECT_EQ(kernels[0].max_memory_vgprs, 1024u);
  const auto exact = waitcheck_kernels_exact(object);
  ASSERT_EQ(exact.size(), 2u);
  EXPECT_NE(exact[0].descriptor_vaddr, exact[1].descriptor_vaddr);
  EXPECT_EQ(exact[0].entry_offset, exact[1].entry_offset);
  EXPECT_EQ(exact[0].min_vgprs, exact[0].max_vgprs);
  EXPECT_EQ(exact[1].min_vgprs, exact[1].max_vgprs);
  EXPECT_NE(exact[0].max_vgprs, exact[1].max_vgprs);
}

TEST(WaitcheckTest, ExactDescriptorDiscoveryKeepsDispatchAllocation) {
  std::vector<uint32_t> program;
  append_inst(program, global_load_b32(16));
  append_inst(program, v_mov_b32(1, 16));
  append_inst(program, s_endpgm());
  const auto image = waitcheck_test::make_gfx_code_object(
      program, EF_AMDGPU_MACH_AMDGCN_GFX1201, /*wave32=*/true);
  const AmdGpuCodeObject object(image.data(), image.size());
  const auto exact = waitcheck_kernels_exact(object);
  const auto grouped = waitcheck_kernels(object);
  ASSERT_EQ(exact.size(), 2u);
  ASSERT_EQ(grouped.size(), 1u);
  ASSERT_EQ(exact[0].max_vgprs, 8u);
  ASSERT_EQ(exact[1].max_vgprs, 256u);
  EXPECT_EQ(exact[0].entry_offset, exact[1].entry_offset);
  EXPECT_NE(exact[0].descriptor_vaddr, exact[1].descriptor_vaddr);
  const auto small = analyze_waitcnts_for_kernel(object, ROCJITSU_CODE_ARCH_RDNA4, exact[0]);
  const auto large = analyze_waitcnts_for_kernel(object, ROCJITSU_CODE_ARCH_RDNA4, exact[1]);
  const auto offline = analyze_waitcnts_for_kernel(object, ROCJITSU_CODE_ARCH_RDNA4, grouped[0]);
  ASSERT_TRUE(small.complete && large.complete && offline.complete);
  EXPECT_EQ(small.diagnostics_observed, 0u) << diagnostic_summary(small);
  EXPECT_EQ(large.diagnostics_observed, 1u) << diagnostic_summary(large);
  EXPECT_EQ(offline.diagnostics_observed, 1u) << diagnostic_summary(offline);
}

TEST(WaitcheckTest, Gfx1250ReturningAtomicRespectsVgprMsbDestinationBank) {
  for (bool buffer : {false, true}) {
    for (uint32_t allocation : {16u, 272u}) {
      SCOPED_TRACE(testing::Message() << buffer << '/' << allocation);
      std::vector<uint32_t> program;
      append_inst(program, s_set_vgpr_msb(vgpr_msb_mode(1, 0, 0, 1)));
      const uint64_t atomic = program.size() * sizeof(uint32_t);
      if (buffer)
        append_inst(program, buffer_atomic_add_u32(0));
      else
        append_inst(program, global_atomic_add_u32(0, 8, 124, true));
      const uint64_t consumer = program.size() * sizeof(uint32_t);
      append_inst(program, v_mov_b32(1, 0));
      append_inst(program, s_endpgm());
      const auto report = analyze_gfx12_with_vgpr_allocation(
          program, EF_AMDGPU_MACH_AMDGCN_GFX1250, ROCJITSU_CODE_ARCH_CDNA5, true, allocation);
      ASSERT_TRUE(report.supported && report.complete) << report.analysis_error;
      const bool pending_return = std::ranges::any_of(report.diagnostics, [&](const auto &d) {
        return d.counter == WaitCounterKind::Load && d.producer_section_offset == atomic &&
               d.section_offset == consumer && d.reg.index == 256;
      });
      EXPECT_EQ(pending_return, allocation == 272) << diagnostic_summary(report);
    }
  }
}

TEST(WaitcheckTest, Gfx12NullifiedGlobalAtomicSkipsPayloadAndAddressReads) {
  for (const auto [arch, machine] :
       {std::pair{ROCJITSU_CODE_ARCH_CDNA5, EF_AMDGPU_MACH_AMDGCN_GFX1250},
        std::pair{ROCJITSU_CODE_ARCH_RDNA4, EF_AMDGPU_MACH_AMDGCN_GFX1201}}) {
    for (bool out_of_range : {false, true}) {
      SCOPED_TRACE(testing::Message() << machine << '/' << out_of_range);
      std::vector<uint32_t> program;
      append_inst(program, global_load_b32(9, 4));
      const uint64_t atomic_pc = program.size() * sizeof(uint32_t);
      auto atomic = global_atomic_add_u32(out_of_range ? 16 : 15, 9, 124, true);
      atomic.vsrc = 9;
      append_inst(program, atomic);
      const uint64_t consumer_pc = program.size() * sizeof(uint32_t);
      append_inst(program, v_mov_b32(3, 9));
      append_inst(program, s_endpgm());
      const auto report = analyze_gfx12_with_vgpr_allocation(program, machine, arch, true, 16);
      ASSERT_TRUE(report.supported && report.complete) << report.analysis_error;
      EXPECT_EQ(std::ranges::any_of(report.diagnostics,
                                    [&](const auto &d) {
                                      return d.counter == WaitCounterKind::Load &&
                                             d.producer_section_offset == 0 &&
                                             d.section_offset == atomic_pc && d.reg.index == 9;
                                    }),
                !out_of_range)
          << diagnostic_summary(report);
      EXPECT_TRUE(std::ranges::any_of(report.diagnostics, [&](const auto &d) {
        return d.counter == WaitCounterKind::Load && d.producer_section_offset == 0 &&
               d.section_offset == consumer_pc && d.reg.index == 9;
      })) << diagnostic_summary(report);
    }
  }
}

TEST(WaitcheckTest, Gfx1250DescriptorAloneProvesNullifiedMemoryResult) {
  std::vector<uint32_t> program;
  append_inst(program, global_load_b32(16, 8));
  const uint64_t consumer = program.size() * sizeof(uint32_t);
  append_inst(program, v_mov_b32(3, 16));
  append_inst(program, s_endpgm());
  const auto image = waitcheck_test::make_gfx_code_object(program, EF_AMDGPU_MACH_AMDGCN_GFX1250,
                                                          true, /*allocation_alias=*/false);
  const AmdGpuCodeObject object(image.data(), image.size());
  const auto report = analyze_waitcnts_for_kernel(object, ROCJITSU_CODE_ARCH_CDNA5, 0);
  ASSERT_TRUE(report.supported && report.complete) << report.analysis_error;
  EXPECT_FALSE(std::ranges::any_of(report.diagnostics, [&](const auto &d) {
    return d.counter == WaitCounterKind::Load && d.producer_section_offset == 0 &&
           d.section_offset == consumer;
  })) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx12NullifiedAtomicDoesNotJustifyAnEmittedLoadWait) {
  for (bool out_of_range : {false, true}) {
    SCOPED_TRACE(out_of_range);
    std::vector<uint32_t> program;
    const uint32_t dest = out_of_range ? 16 : 15;
    append_inst(program, global_atomic_add_u32(dest, 8, 124, true));
    append_inst(program, sopp(64, 0)); // s_wait_loadcnt 0
    append_inst(program, v_mov_b32(3, dest));
    append_inst(program, s_endpgm());
    WaitcheckOptions options;
    options.check_counter_parity = true;
    const auto report = analyze_gfx12_with_vgpr_allocation(
        program, EF_AMDGPU_MACH_AMDGCN_GFX1250, ROCJITSU_CODE_ARCH_CDNA5, true, 16, options);
    ASSERT_TRUE(report.supported && report.complete) << report.analysis_error;
    EXPECT_EQ(report.counter_parity_exact, out_of_range ? 0u : 1u);
    EXPECT_EQ(report.counter_unmodeled_wait_observed, out_of_range ? 1u : 0u);
  }
}

TEST(WaitcheckTest, Gfx12OutOfRangeOrdinaryLoadsDoNotReadTheirAddresses) {
  for (const auto [arch, machine] :
       {std::pair{ROCJITSU_CODE_ARCH_CDNA5, EF_AMDGPU_MACH_AMDGCN_GFX1250},
        std::pair{ROCJITSU_CODE_ARCH_RDNA4, EF_AMDGPU_MACH_AMDGCN_GFX1201}}) {
    for (unsigned kind = 0; kind < 3; ++kind) {
      for (bool out_of_range : {false, true}) {
        SCOPED_TRACE(testing::Message() << machine << '/' << kind << '/' << out_of_range);
        std::vector<uint32_t> program;
        append_inst(program, global_load_b32(8, 4));
        const uint64_t second = program.size() * sizeof(uint32_t);
        const uint32_t dest = out_of_range ? 16 : 15;
        if (kind == 0) {
          append_inst(program, global_load_b32(dest, 8));
        } else if (kind == 1) {
          auto load = std::bit_cast<rdna4::VbufferMachineInst>(
              std::array<uint32_t, 3>{0xC4050000U, 0, 0}); // buffer_load_b32
          load.vdata = dest;
          load.vaddr = 8;
          load.rsrc = 40;
          load.soffset = 124;
          load.offen = 1;
          append_inst(program, load);
        } else {
          program.insert(program.end(), {0xEC05007CU, dest, 8u}); // flat_load_b32
        }
        const uint64_t consumer = program.size() * sizeof(uint32_t);
        append_inst(program, v_mov_b32(3, 8));
        append_inst(program, s_endpgm());
        const auto report = analyze_gfx12_with_vgpr_allocation(program, machine, arch, true, 16);
        ASSERT_TRUE(report.supported && report.complete) << report.analysis_error;
        const bool address_hazard = std::ranges::any_of(report.diagnostics, [&](const auto &d) {
          return d.producer_section_offset == 0 && d.section_offset == second && d.reg.index == 8;
        });
        EXPECT_EQ(address_hazard, !out_of_range) << diagnostic_summary(report);
        EXPECT_TRUE(std::ranges::any_of(report.diagnostics, [&](const auto &d) {
          return d.producer_section_offset == 0 && d.section_offset == consumer && d.reg.index == 8;
        })) << diagnostic_summary(report);
      }
    }
  }
}

TEST(WaitcheckTest, Gfx12NullifiedFlatStillOccupiesAnOlderLoadsCounterPosition) {
  for (const auto [arch, machine] :
       {std::pair{ROCJITSU_CODE_ARCH_CDNA5, EF_AMDGPU_MACH_AMDGCN_GFX1250},
        std::pair{ROCJITSU_CODE_ARCH_RDNA4, EF_AMDGPU_MACH_AMDGCN_GFX1201}}) {
    for (bool wait : {false, true}) {
      SCOPED_TRACE(testing::Message() << machine << '/' << wait);
      std::vector<uint32_t> program;
      append_inst(program, global_load_b32(2, 8));
      append_inst(program, flat_load_b32(16)); // Beyond the 16-VGPR allocation.
      if (wait)
        append_inst(program, sopp(64, 1)); // Older load is one position behind FLAT.
      const uint64_t consumer = program.size() * sizeof(uint32_t);
      append_inst(program, v_mov_b32(3, 2));
      append_inst(program, s_endpgm());
      const auto report = analyze_gfx12_with_vgpr_allocation(program, machine, arch, true, 16);
      ASSERT_TRUE(report.supported && report.complete) << report.analysis_error;
      EXPECT_EQ(std::ranges::any_of(report.diagnostics,
                                    [&](const auto &d) {
                                      return d.counter == WaitCounterKind::Load &&
                                             d.producer_section_offset == 0 &&
                                             d.section_offset == consumer && d.reg.index == 2;
                                    }),
                !wait)
          << diagnostic_summary(report);
    }
  }
}

TEST(WaitcheckTest, Gfx12NullifiedVmemStillOccupiesAnOlderLoadsCounterPosition) {
  for (const auto [arch, machine] :
       {std::pair{ROCJITSU_CODE_ARCH_CDNA5, EF_AMDGPU_MACH_AMDGCN_GFX1250},
        std::pair{ROCJITSU_CODE_ARCH_RDNA4, EF_AMDGPU_MACH_AMDGCN_GFX1201}}) {
    for (unsigned kind = 0; kind < 4; ++kind) {
      for (bool wait : {false, true}) {
        SCOPED_TRACE(testing::Message() << machine << '/' << kind << '/' << wait);
        std::vector<uint32_t> program;
        append_inst(program, global_load_b32(2, 8));
        if (kind == 1) {
          auto load = std::bit_cast<rdna4::VbufferMachineInst>(
              std::array<uint32_t, 3>{0xC4050000U, 0, 0}); // buffer_load_b32
          load.vdata = 16; // Beyond the 16-VGPR allocation.
          load.vaddr = 8;
          load.rsrc = 40;
          load.soffset = 124;
          load.offen = 1;
          append_inst(program, load);
        } else if (kind == 0) {
          append_inst(program, global_load_b32(16, 8));
        } else if (kind == 2) {
          append_inst(program, global_atomic_add_u32(16, 8, 124, true));
        } else {
          append_inst(program, buffer_atomic_add_u32(16));
        }
        if (wait)
          append_inst(program, sopp(64, 1));
        const uint64_t consumer = program.size() * sizeof(uint32_t);
        append_inst(program, v_mov_b32(3, 2));
        append_inst(program, s_endpgm());
        const auto report = analyze_gfx12_with_vgpr_allocation(program, machine, arch, true, 16);
        ASSERT_TRUE(report.supported && report.complete) << report.analysis_error;
        EXPECT_EQ(std::ranges::any_of(report.diagnostics,
                                      [&](const auto &d) {
                                        return d.counter == WaitCounterKind::Load &&
                                               d.producer_section_offset == 0 &&
                                               d.section_offset == consumer && d.reg.index == 2;
                                      }),
                  !wait)
            << diagnostic_summary(report);
      }
    }
  }
}

TEST(WaitcheckTest, Gfx12NullifiedVmemWithoutOlderRequestDoesNotJustifyWait) {
  for (const auto [arch, machine] :
       {std::pair{ROCJITSU_CODE_ARCH_CDNA5, EF_AMDGPU_MACH_AMDGCN_GFX1250},
        std::pair{ROCJITSU_CODE_ARCH_RDNA4, EF_AMDGPU_MACH_AMDGCN_GFX1201}}) {
    for (unsigned kind = 0; kind < 4; ++kind) {
      SCOPED_TRACE(testing::Message() << machine << '/' << kind);
      std::vector<uint32_t> program;
      if (kind == 1) {
        auto load = std::bit_cast<rdna4::VbufferMachineInst>(
            std::array<uint32_t, 3>{0xC4050000U, 0, 0}); // buffer_load_b32
        load.vdata = 16;
        load.vaddr = 8;
        load.rsrc = 40;
        load.soffset = 124;
        load.offen = 1;
        append_inst(program, load);
      } else if (kind == 0) {
        append_inst(program, global_load_b32(16, 8));
      } else if (kind == 2) {
        append_inst(program, global_atomic_add_u32(16, 8, 124, true));
      } else {
        append_inst(program, buffer_atomic_add_u32(16));
      }
      append_inst(program, sopp(64, 0));
      append_inst(program, s_endpgm());
      WaitcheckOptions options;
      options.check_counter_parity = true;
      const auto report =
          analyze_gfx12_with_vgpr_allocation(program, machine, arch, true, 16, options);
      ASSERT_TRUE(report.supported && report.complete) << report.analysis_error;
      EXPECT_EQ(report.counter_parity_exact, 0u) << diagnostic_summary(report);
      EXPECT_EQ(report.counter_unmodeled_wait_observed, 1u) << diagnostic_summary(report);
    }
  }
}

TEST(WaitcheckTest, Gfx12NullifiedFlatWithoutOlderRequestDoesNotJustifyWait) {
  for (const auto [arch, machine] :
       {std::pair{ROCJITSU_CODE_ARCH_CDNA5, EF_AMDGPU_MACH_AMDGCN_GFX1250},
        std::pair{ROCJITSU_CODE_ARCH_RDNA4, EF_AMDGPU_MACH_AMDGCN_GFX1201}}) {
    std::vector<uint32_t> program;
    append_inst(program, flat_load_b32(16));
    append_inst(program, sopp(64, 0));
    append_inst(program, s_endpgm());
    WaitcheckOptions options;
    options.check_counter_parity = true;
    const auto report =
        analyze_gfx12_with_vgpr_allocation(program, machine, arch, true, 16, options);
    ASSERT_TRUE(report.supported && report.complete) << report.analysis_error;
    EXPECT_EQ(report.counter_parity_exact, 0u) << diagnostic_summary(report);
    EXPECT_EQ(report.counter_unmodeled_wait_observed, 1u) << diagnostic_summary(report);
  }
}

TEST(WaitcheckTest, Gfx1201FlatChecksAddressBeforeQueueCapacityAdmission) {
  for (unsigned younger : {61u, 62u}) {
    for (bool wait : {false, true}) {
      SCOPED_TRACE(testing::Message() << younger << '/' << wait);
      std::vector<uint32_t> program;
      append_global_load_b64_v8_v7_s0_offset16(program);
      for (unsigned i = 0; i < younger; ++i)
        append_inst(program, global_load_b32(0, 4));
      if (wait)
        append_inst(program, sopp(64, 0));
      const uint64_t consumer = program.size() * sizeof(uint32_t);
      append_inst(program, flat_load_b32(1)); // Address v[8:9] reads the oldest load.
      append_inst(program, s_endpgm());
      const auto image = waitcheck_test::make_gfx1201_code_object(program);
      const AmdGpuCodeObject object(image.data(), image.size());
      const auto report = analyze_waitcnts_for_kernel(object, ROCJITSU_CODE_ARCH_RDNA4, 0);
      ASSERT_TRUE(report.supported && report.complete) << report.analysis_error;
      EXPECT_EQ(std::ranges::any_of(report.diagnostics,
                                    [&](const auto &d) {
                                      return d.counter == WaitCounterKind::Load &&
                                             d.producer_section_offset == 0 &&
                                             d.section_offset == consumer && d.reg.index == 8;
                                    }),
                !wait)
          << diagnostic_summary(report);
    }
  }
}

TEST(WaitcheckTest, Gfx12BufferTfeStatusWordParticipatesInResultBounds) {
  for (const auto [arch, machine] :
       {std::pair{ROCJITSU_CODE_ARCH_CDNA5, EF_AMDGPU_MACH_AMDGCN_GFX1250},
        std::pair{ROCJITSU_CODE_ARCH_RDNA4, EF_AMDGPU_MACH_AMDGCN_GFX1201}}) {
    for (bool tfe : {false, true}) {
      SCOPED_TRACE(testing::Message() << machine << '/' << tfe);
      std::vector<uint32_t> program;
      append_inst(program, global_load_b32(8, 4));
      const uint64_t load_pc = program.size() * sizeof(uint32_t);
      auto load = std::bit_cast<rdna4::VbufferMachineInst>(
          std::array<uint32_t, 3>{0xC4050000U, 0, 0}); // buffer_load_b32
      load.vdata = 15;
      load.vaddr = 8;
      load.rsrc = 40;
      load.soffset = 124;
      load.offen = 1;
      load.tfe = tfe;
      append_inst(program, load);
      append_inst(program, s_endpgm());
      const auto report = analyze_gfx12_with_vgpr_allocation(program, machine, arch, true, 16);
      ASSERT_TRUE(report.supported && report.complete) << report.analysis_error;
      EXPECT_EQ(std::ranges::any_of(report.diagnostics,
                                    [&](const auto &d) {
                                      return d.counter == WaitCounterKind::Load &&
                                             d.producer_section_offset == 0 &&
                                             d.section_offset == load_pc && d.reg.index == 8;
                                    }),
                !tfe)
          << diagnostic_summary(report);
    }
  }
}

TEST(WaitcheckTest, Gfx1201FormattedLoadUsesOpcodeResultWidthBeforeAddressRead) {
  for (uint32_t dest : {14u, 15u}) {
    SCOPED_TRACE(dest);
    std::vector<uint32_t> program;
    append_inst(program, global_load_b32(8, 4));
    const uint64_t load_pc = program.size() * sizeof(uint32_t);
    auto load = std::bit_cast<rdna4::VbufferMachineInst>(
        std::array<uint32_t, 3>{0xC4204000U, 0, 0}); // tbuffer_load_format_xy
    load.vdata = dest;
    load.vaddr = 8;
    load.rsrc = 40;
    load.soffset = 124;
    load.offen = 1;
    append_inst(program, load);
    append_inst(program, s_endpgm());
    const auto report = analyze_gfx12_with_vgpr_allocation(program, EF_AMDGPU_MACH_AMDGCN_GFX1201,
                                                           ROCJITSU_CODE_ARCH_RDNA4, true, 16);
    ASSERT_TRUE(report.supported && report.complete) << report.analysis_error;
    EXPECT_EQ(std::ranges::any_of(report.diagnostics,
                                  [&](const auto &d) {
                                    return d.counter == WaitCounterKind::Load &&
                                           d.producer_section_offset == 0 &&
                                           d.section_offset == load_pc && d.reg.index == 8;
                                  }),
              dest == 14)
        << diagnostic_summary(report);
  }
}

TEST(WaitcheckTest, Gfx1201CompactWave64MemoryResultUsesPhysicalExtent) {
  for (uint32_t dest : {14u, 15u}) {
    SCOPED_TRACE(dest);
    std::vector<uint32_t> program;
    auto load = std::bit_cast<rdna4::VglobalMachineInst>(
        std::array<uint32_t, 3>{0xEE15C000U, 0, 0}); // global_load_tr_b128
    load.vdst = dest;
    load.vaddr = 8;
    load.saddr = 124;
    append_inst(program, load);
    const uint64_t consumer = program.size() * sizeof(uint32_t);
    append_inst(program, v_mov_b32(3, dest));
    append_inst(program, s_endpgm());
    const auto report = analyze_gfx12_with_vgpr_allocation(program, EF_AMDGPU_MACH_AMDGCN_GFX1201,
                                                           ROCJITSU_CODE_ARCH_RDNA4, false, 16);
    ASSERT_TRUE(report.supported && report.complete) << report.analysis_error;
    const bool pending_return = std::ranges::any_of(report.diagnostics, [&](const auto &d) {
      return d.counter == WaitCounterKind::Load && d.producer_section_offset == 0 &&
             d.section_offset == consumer && d.reg.index == dest;
    });
    EXPECT_EQ(pending_return, dest == 14) << diagnostic_summary(report);
  }
}

TEST(WaitcheckTest, Gfx1250AtomicParityKeepsCompanionCompletionAndOlderCredits) {
  WaitcheckOptions options;
  options.check_counter_parity = true;
  for (bool companion_load_zero : {false, true}) {
    std::vector<uint32_t> program;
    append_inst(program, global_load_b32(14, 8));
    append_inst(program, s_wait_xcnt_0());
    if (companion_load_zero)
      append_inst(program, sopp(64, 0));
    append_inst(program, global_atomic_add_u32(6, 16));
    const auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA5, options);
    EXPECT_EQ(report.diagnostics_observed, 0u) << diagnostic_summary(report);
    EXPECT_EQ(report.counter_parity_exact, companion_load_zero ? 0u : 1u);
  }

  // The first atomic does not create another ordinary translation to flush.
  // X(1) retires the older load and may leave the atomic outstanding.
  std::vector<uint32_t> program;
  append_inst(program, global_load_b32(14, 8));
  append_inst(program, global_atomic_add_u32(6, 16));
  append_inst(program, s_wait_xcnt(1));
  append_inst(program, global_atomic_add_u32(12, 20));
  const auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA5, options);
  EXPECT_EQ(report.diagnostics_observed, 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.counter_parity_exact, 1u);
}

TEST(WaitcheckTest, Gfx1250ScalarPrefetchRequiresPriorTranslationDrain) {
  for (bool drain : {false, true}) {
    std::vector<uint32_t> program;
    append_inst(program, global_load_b32(1, 8));
    if (drain)
      append_inst(program, s_wait_xcnt_0());
    append_inst(program, s_prefetch_data());
    const auto report = analyze_gfx1250_normal(program);
    ASSERT_TRUE(report.supported);
    EXPECT_EQ(report.diagnostics_observed, drain ? 0u : 1u);
    if (!drain) {
      ASSERT_FALSE(report.diagnostics.empty());
      EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::X);
      EXPECT_EQ(report.diagnostics[0].required_count, 0u);
    }
  }
}

TEST(WaitcheckTest, Gfx1250ParityAttributesOnlyEnabledScalarPrefetch) {
  for (const bool enabled : {false, true}) {
    for (const bool initial_scalar_load : {false, true}) {
      std::vector<uint32_t> program;
      append_inst(program, s_setreg_imm32_b32(hwreg(1, 24, 1), unsigned(enabled)));
      if (initial_scalar_load)
        append_inst(program, s_load_b32(8, 0));
      else
        append_inst(program, global_load_b32(1, 8));
      append_inst(program, s_wait_xcnt_0());
      append_inst(program, v_mov_b32(20, 21));
      append_inst(program, s_prefetch_data());
      TestCodeObject object(program);
      WaitcheckOptions options;
      options.check_counter_parity = true;
      const auto report = analyze_waitcnts(object, ROCJITSU_CODE_ARCH_CDNA5, options);
      ASSERT_TRUE(report.supported) << report.analysis_error;
      EXPECT_EQ(report.counter_parity_exact, enabled ? 1u : 0u);
      EXPECT_EQ(report.counter_unmodeled_wait_observed, enabled ? 0u : 1u);
    }
  }
}

TEST(WaitcheckTest, Gfx1250PrefetchUsesTheOutstandingTranslationAfterCompanionWaits) {
  for (bool enable : {false, true}) {
    for (bool reverse : {false, true}) {
      for (bool separated : {false, true}) {
        for (bool x_wait : {false, true}) {
          for (unsigned load_count : {0u, 1u}) {
            SCOPED_TRACE(testing::Message() << enable << ", " << reverse << ", " << separated
                                            << ", " << x_wait << ", " << load_count);
            std::vector<uint32_t> program;
            append_inst(program, s_setreg_imm32_b32(hwreg(1, 24, 1), unsigned(enable)));
            append_inst(program, global_load_b32(1, 8));
            if (reverse)
              append_inst(program, sopp(64, load_count));
            else if (x_wait)
              append_inst(program, s_wait_xcnt_0());
            if (separated)
              append_inst(program, v_mov_b32(20, 21));
            if (!reverse)
              append_inst(program, sopp(64, load_count));
            else if (x_wait)
              append_inst(program, s_wait_xcnt_0());
            append_inst(program, s_prefetch_data());
            WaitcheckOptions options;
            options.check_counter_parity = true;
            TestCodeObject object(program);
            const auto report = analyze_waitcnts(object, ROCJITSU_CODE_ARCH_CDNA5, options);
            ASSERT_TRUE(report.supported) << report.analysis_error;
            EXPECT_EQ(report.diagnostics_observed, unsigned(enable && !x_wait && load_count));
            if (!enable || (!x_wait && load_count)) {
              EXPECT_EQ(report.counter_parity_exact, 0u);
              continue;
            }
            const bool sequential_drains = x_wait && !load_count && separated;
            EXPECT_EQ(report.counter_parity_exact, sequential_drains ? 0u : 1u);
            if (x_wait && !sequential_drains) {
              const auto &rows = report.counter_underaccounting_diagnostics;
              ASSERT_EQ(rows.size(), 1u);
              EXPECT_EQ(rows.front().counter,
                        load_count ? WaitCounterKind::Load : WaitCounterKind::X);
            }
          }
        }
      }
    }
  }
}

TEST(WaitcheckTest, Gfx1250PrefetchPartialCompanionLeavesTheYoungerTranslationForX) {
  WaitcheckOptions options;
  options.check_counter_parity = true;
  for (bool x_wait : {false, true}) {
    for (unsigned load_count : {0u, 1u, 2u}) {
      SCOPED_TRACE(testing::Message() << x_wait << ", " << load_count);
      std::vector<uint32_t> program;
      append_inst(program, s_setreg_imm32_b32(hwreg(1, 24, 1), 1));
      append_inst(program, global_load_b32(1, 8));
      append_inst(program, global_load_b32(2, 10));
      if (x_wait)
        append_inst(program, s_wait_xcnt_0());
      append_inst(program, sopp(64, load_count));
      append_inst(program, s_prefetch_data());
      TestCodeObject object(program);
      const auto report = analyze_waitcnts(object, ROCJITSU_CODE_ARCH_CDNA5, options);
      ASSERT_TRUE(report.supported && report.complete) << report.analysis_error;
      EXPECT_EQ(report.diagnostics_observed, x_wait ? 0u : load_count);
      EXPECT_EQ(report.counter_parity_exact, unsigned(x_wait || load_count == 0));
      if (x_wait) {
        ASSERT_EQ(report.counter_underaccounting_diagnostics.size(), 1u);
        EXPECT_EQ(report.counter_underaccounting_diagnostics.front().counter,
                  load_count ? WaitCounterKind::Load : WaitCounterKind::X);
      }
    }
  }
}

TEST(WaitcheckTest, Gfx1250ScalarPrefetchHonorsExplicitEnable) {
  for (bool enabled : {false, true}) {
    std::vector<uint32_t> program;
    append_inst(program, s_setreg_imm32_b32(hwreg(1, 24, 1), unsigned(enabled)));
    append_inst(program, global_load_b32(1, 8));
    append_inst(program, s_prefetch_data());
    const auto report = analyze_gfx1250_normal(program);
    ASSERT_TRUE(report.supported) << report.analysis_error;
    EXPECT_EQ(report.diagnostics_observed, enabled ? 1u : 0u) << diagnostic_summary(report);
  }
}

TEST(WaitcheckTest, Gfx1250ReportsXcntBeforeScalarPrefetchBaseOverwrite) {
  std::vector<uint32_t> program;
  append_inst(program, s_prefetch_data());
  append_inst(program, s_mov_b32(0, 128));

  auto report = analyze_gfx1250_normal(program);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::X);
  EXPECT_EQ(report.diagnostics[0].reg, (RegisterRef{RegClass::SGPR, 0, 1}));
}

TEST(WaitcheckTest, Gfx1250VmemTranslationOrderingHandlesVmemDestinationOverwrite) {
  std::vector<uint32_t> program;
  append_inst(program, global_store_b32(4, 8));
  append_inst(program, global_load_b32(4, 12));

  auto report = analyze_gfx1250_normal(program);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx1250NullSaddrGlobalTracksBothVaddrRegisters) {
  std::vector<uint32_t> program;
  append_inst(program, global_load_b32(4, 2, 124));
  append_inst(program, v_mov_b32(3, 12));

  auto report = analyze_gfx1250_normal(program);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::X);
  EXPECT_EQ(report.diagnostics[0].reg, (RegisterRef{RegClass::VGPR, 3, 1}));
}

TEST(WaitcheckTest, Gfx1250VmemOrderingHandlesNullSaddrVaddrHighOverwrite) {
  std::vector<uint32_t> program;
  append_inst(program, global_load_b32(4, 2, 124));
  append_inst(program, global_load_b32(3, 10, 124));
  append_inst(program, v_mov_b32(2, 12));

  auto report = analyze_gfx1250_normal(program);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx1250PartialXcntAcrossFallthroughJoinRetiresOldestTranslation) {
  std::vector<uint32_t> program;
  append_inst(program, sopp(33, 6)); // s_cbranch_scc0 over both VMEM operations.
  append_inst(program, global_load_b32(0, 8));
  append_inst(program, global_load_b32(1, 10));
  append_inst(program, s_wait_xcnt(1));
  append_inst(program, v_mov_b32(8, 12));

  auto report = analyze_gfx1250_normal(program);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx1250PartialXcntAcrossFallthroughJoinKeepsYoungestTranslation) {
  std::vector<uint32_t> program;
  append_inst(program, sopp(33, 6)); // s_cbranch_scc0 over both VMEM operations.
  append_inst(program, global_load_b32(0, 8));
  append_inst(program, global_load_b32(1, 10));
  append_inst(program, s_wait_xcnt(1));
  append_inst(program, v_mov_b32(10, 12));

  auto report = analyze_gfx1250_normal(program);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::X);
  EXPECT_EQ(report.diagnostics[0].reg, (RegisterRef{RegClass::VGPR, 10, 1}));
}

TEST(WaitcheckTest, Gfx1250ReportsAsynccntBeforeLdsUse) {
  std::vector<uint32_t> program;
  append_global_load_async_to_lds_b32(program);
  append_inst(program, ds_load_b32(4, 12));

  auto report = analyze_gfx1250_normal(program);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Async);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::MemoryOrder);
  EXPECT_EQ(report.diagnostics[0].required_count, 0u);
}

TEST(WaitcheckTest, Gfx1250AcceptsAsynccntBeforeLdsUse) {
  std::vector<uint32_t> program;
  append_global_load_async_to_lds_b32(program);
  append_inst(program, s_wait_asynccnt(0));
  append_inst(program, ds_load_b32(4, 12));

  auto report = analyze_gfx1250_normal(program);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx1250PartialAsynccntKeepsBothLoadAndStore) {
  std::vector<uint32_t> program;
  append_global_load_async_to_lds_b32(program);
  append_global_store_async_from_lds_b32(program);
  append_inst(program, s_wait_asynccnt(1));
  append_inst(program, ds_load_b32(4, 12));

  auto report = analyze_gfx1250_normal(program);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 2u) << diagnostic_summary(report);
  for (const auto &diagnostic : report.diagnostics) {
    EXPECT_EQ(diagnostic.counter, WaitCounterKind::Async);
    EXPECT_EQ(diagnostic.required_count, 0u);
  }
}

TEST(WaitcheckTest, Gfx1250PartialAsynccntRetiresOlderAsyncLoad) {
  std::vector<uint32_t> program;
  append_global_load_async_to_lds_b32(program);
  append_global_load_async_to_lds_b32(program);
  append_inst(program, s_wait_asynccnt(1));
  append_inst(program, ds_load_b32(4, 12));
  auto report = analyze_gfx1250_normal(program);
  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics.front().counter, WaitCounterKind::Async);
  EXPECT_EQ(report.diagnostics.front().required_count, 0u);
}

TEST(WaitcheckTest, Gfx1250AsyncLoadDoesNotAdvanceLoadcntAge) {
  std::vector<uint32_t> program;
  append_inst(program, global_load_b32(20, 16));
  append_global_load_async_to_lds_b32(program);
  append_inst(program, sopp(64, 1)); // s_wait_loadcnt 1
  append_inst(program, v_mov_b32(4, 20));

  auto report = analyze_gfx1250_normal(program);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics.front().counter, WaitCounterKind::Load);
  EXPECT_EQ(report.diagnostics.front().required_count, 0u);
}

TEST(WaitcheckTest, Gfx1250AsyncReplaySourcesAndLdsTransferHaveIndependentWaits) {
  for (bool store : {false, true}) {
    for (auto wait : {WaitCounterKind::Load, WaitCounterKind::X, WaitCounterKind::Async}) {
      SCOPED_TRACE(testing::Message() << store << '/' << static_cast<unsigned>(wait));
      std::vector<uint32_t> words;
      (store ? append_global_store_async_from_lds_b32 : append_global_load_async_to_lds_b32)(words);
      if (wait == WaitCounterKind::Load)
        append_inst(words, sopp(64, 0));
      else if (wait == WaitCounterKind::X)
        append_inst(words, s_wait_xcnt(0));
      else
        append_inst(words, s_wait_asynccnt(0));
      append_inst(words, s_mov_b32(0, 128)); // Overwrite the scalar address.
      append_inst(words, ds_store_b32(4, 12));
      const auto report = analyze_gfx1250_normal(words);
      ASSERT_TRUE(report.supported) << report.analysis_error;
      const auto count = [&](WaitCounterKind counter) {
        return std::ranges::count(report.diagnostics, counter, &WaitcheckDiagnostic::counter);
      };
      EXPECT_EQ(count(WaitCounterKind::X), wait == WaitCounterKind::Load)
          << diagnostic_summary(report);
      EXPECT_EQ(count(WaitCounterKind::Async), wait != WaitCounterKind::Async)
          << diagnostic_summary(report);
    }
  }
}

TEST(WaitcheckTest, Gfx1250PossibleAsyncTranslationCannotReleaseAnotherSource) {
  for (bool scalar : {false, true}) {
    for (bool drain : {false, true}) {
      SCOPED_TRACE(testing::Message() << scalar << '/' << drain);
      std::vector<uint32_t> words;
      if (scalar)
        append_inst(words, s_load_b32(20, 4)); // Scalar base s[8:9].
      else
        append_inst(words, global_load_b32(20, 8));
      append_global_load_async_to_lds_b32(words);
      append_inst(words, s_wait_xcnt(drain ? 0 : 1));
      if (scalar)
        append_inst(words, s_mov_b32(8, 128));
      else
        append_inst(words, v_mov_b32(8, 12));
      const auto report = analyze_gfx1250_normal(words);
      ASSERT_TRUE(report.supported) << report.analysis_error;
      ASSERT_EQ(report.diagnostics.size(), !drain) << diagnostic_summary(report);
      if (!drain) {
        EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::X);
        EXPECT_EQ(report.diagnostics[0].producer_section_offset, 0u);
        EXPECT_EQ(report.diagnostics[0].required_count, 0u);
      }
    }
  }
}

TEST(WaitcheckTest, Gfx1250AsyncReplayFollowsItsOwnCompletionClass) {
  for (bool younger_store : {false, true}) {
    std::vector<uint32_t> words;
    append_global_load_async_to_lds_b32(words);
    // The younger operation uses a different scalar address pair.
    (younger_store ? append_global_store_async_from_lds_b32
                   : append_global_load_async_to_lds_b32)(words);
    words[3] |= 8; // SADDR = s[8:9].
    append_inst(words, s_wait_asynccnt(1));
    append_inst(words, s_mov_b32(0, 128));
    const auto report = analyze_gfx1250_normal(words);
    ASSERT_TRUE(report.supported) << report.analysis_error;
    ASSERT_EQ(report.diagnostics.size(), younger_store) << diagnostic_summary(report);
    if (younger_store) {
      EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::X);
      EXPECT_EQ(report.diagnostics[0].producer_section_offset, 0u);
    }
  }
}

TEST(WaitcheckTest, Gfx1250DoesNotInventAsynccntBeforeProgramEnd) {
  std::vector<uint32_t> program;
  append_global_store_async_from_lds_b32(program);
  append_inst(program, s_endpgm());

  auto report = analyze_gfx1250_normal(program);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx1250AsyncArriveNeedsGuaranteedIssueToAgeAnOlderLoad) {
  for (uint32_t exec : {0u, 1u}) {
    for (uint32_t wait : {0u, 1u}) {
      SCOPED_TRACE(testing::Message() << exec << '/' << wait);
      std::vector<uint32_t> words;
      append_global_load_async_to_lds_b32(words);
      append_inst(words, s_wait_xcnt(0));  // Release EXEC without completing the LDS transfer.
      words.push_back(0xbefe0080u | exec); // s_mov_b32 exec_lo, 0 or 1
      append_inst(words, s_wait_alu_vm_vsrc_0());
      append_ds_atomic_async_barrier_arrive_b64(words);
      append_inst(words, s_wait_alu_vm_vsrc_0());
      words.push_back(0xbefe00c1); // s_mov_b32 exec_lo, -1
      append_inst(words, s_wait_asynccnt(wait));
      append_inst(words, ds_load_b32(4, 12));
      const auto report = analyze_gfx1250_normal(words);
      ASSERT_TRUE(report.supported) << report.analysis_error;
      ASSERT_EQ(report.diagnostics.size(), !exec && wait) << diagnostic_summary(report);
      if (!report.diagnostics.empty())
        EXPECT_EQ(report.diagnostics.front().producer_section_offset, 0u);
    }
  }
}

TEST(WaitcheckTest, Gfx1250AsyncBarrierUsesOnlyTheGuardedNonemptyExecEdge) {
  // EXECZ can fall through to the arrive or EXECNZ can jump to it. Reversing
  // either guard makes the arrive reachable with no active lanes.
  enum Guard { None, Execz, Execnz };
  for (Guard guard : {None, Execz, Execnz})
    for (bool taken : {false, true})
      for (unsigned wait : {0u, 1u, 2u}) {
        if (guard == None && taken)
          continue;
        SCOPED_TRACE(testing::Message() << guard << '/' << taken << '/' << wait);
        std::vector<uint32_t> words;
        append_global_load_async_to_lds_b32(words);
        append_inst(words, s_wait_xcnt(0));
        words.push_back(0xbe842108); // s_and_saveexec_b64 s[4:5],s[8:9]
        const size_t branch = words.size();
        if (guard != None) {
          words.push_back(guard == Execz ? 0xbfa50000u : 0xbfa60000u);
          if (taken) {
            append_inst(words, s_wait_asynccnt(0));
            append_inst(words, s_endpgm());
            words[branch] |= words.size() - branch - 1;
          }
        }
        append_inst(words, s_wait_alu_vm_vsrc_0());
        append_ds_atomic_async_barrier_arrive_b64(words);
        append_inst(words, s_wait_alu_vm_vsrc_0());
        words.push_back(0xbefe01c1); // Restore EXEC before the LDS read.
        append_inst(words, s_wait_asynccnt(wait));
        append_inst(words, ds_load_b32(4, 12));
        append_inst(words, s_endpgm());
        if (guard != None && !taken) {
          words[branch] |= words.size() - branch - 1;
          append_inst(words, s_wait_asynccnt(0));
          append_inst(words, s_endpgm());
        }
        auto image =
            waitcheck_test::make_gfx_code_object(words, EF_AMDGPU_MACH_AMDGCN_GFX1250, true);
        AmdGpuCodeObject object(image.data(), image.size());
        const auto report = analyze_waitcnts(object, ROCJITSU_CODE_ARCH_CDNA5);
        const bool nonempty = (guard == Execz && !taken) || (guard == Execnz && taken);
        const bool hazard = wait > static_cast<unsigned>(nonempty);
        ASSERT_TRUE(report.supported && report.complete) << report.analysis_error;
        ASSERT_EQ(report.diagnostics.size(), hazard) << diagnostic_summary(report);
        if (hazard) {
          EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Async);
          EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::MemoryOrder);
          EXPECT_EQ(report.diagnostics[0].producer_section_offset, 0u);
        }
      }
}

TEST(WaitcheckTest, Gfx1250AsyncBarrierArriveRequiresWaitBeforeBarrierWait) {
  std::vector<uint32_t> program;
  append_inst(program, s_wait_alu_vm_vsrc_0());
  append_ds_atomic_async_barrier_arrive_b64(program);
  append_inst(program, s_wait_alu_vm_vsrc_0());
  append_inst(program, sopp(20, 0)); // s_barrier_wait 0

  auto report = analyze_gfx1250_normal(program);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Async);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::MemoryOrder);
}

TEST(WaitcheckTest, Gfx1250ReportsMissingVmVsrcWaitBeforeAsyncBarrierArrive) {
  std::vector<uint32_t> program;
  append_ds_atomic_async_barrier_arrive_b64(program);
  append_inst(program, s_wait_alu_vm_vsrc_0());
  append_inst(program, s_wait_asynccnt(0));
  append_inst(program, s_endpgm());

  auto report = analyze_gfx1250_normal(program);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].kind, WaitcheckDiagnosticKind::AsyncBarrierPreWait);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::VmVsrc);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::MemoryOrder);
  EXPECT_NE(report.diagnostics[0].message.find("immediately before"), std::string::npos);
}

TEST(WaitcheckTest, Gfx1250ReportsMissingVmVsrcWaitAfterAsyncBarrierArrive) {
  std::vector<uint32_t> program;
  append_inst(program, s_wait_alu_vm_vsrc_0());
  append_ds_atomic_async_barrier_arrive_b64(program);
  append_inst(program, s_wait_asynccnt(0));
  append_inst(program, s_endpgm());

  auto report = analyze_gfx1250_normal(program);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].kind, WaitcheckDiagnosticKind::AsyncBarrierPostWait);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::VmVsrc);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::MemoryOrder);
  EXPECT_NE(report.diagnostics[0].message.find("immediately after"), std::string::npos);
}

TEST(WaitcheckTest, Gfx1250ReportsTensorcntBeforeLdsUse) {
  std::vector<uint32_t> program;
  append_tensor_load_to_lds(program);
  append_inst(program, ds_load_b32(4, 12));

  auto report = analyze_gfx1250_normal(program);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Tensor);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::MemoryOrder);
}

TEST(WaitcheckTest, Gfx1250AcceptsTensorcntBeforeLdsUse) {
  std::vector<uint32_t> program;
  append_tensor_load_to_lds(program);
  append_inst(program, s_wait_tensorcnt(0));
  append_inst(program, ds_load_b32(4, 12));

  auto report = analyze_gfx1250_normal(program);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx1250DoesNotInventTensorcntBeforeProgramEnd) {
  std::vector<uint32_t> program;
  append_tensor_store_from_lds(program);
  append_inst(program, s_endpgm());

  auto report = analyze_gfx1250_normal(program);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx1201CodeObjectTracksSgprHazardsInNormalMode) {
  std::vector<uint32_t> program;
  append_inst(program, v_add_f32_e32(0, 102, 0));
  append_inst(program, s_mov_b32(102, 128));
  append_inst(program, v_add_f32_e32(1, 102, 1));

  TestCodeObject code_object(program);
  auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].kind, WaitcheckDiagnosticKind::SgprDepctr);
  EXPECT_NE(report.diagnostics[0].message.find("depctr_sa_sdst(0)"), std::string::npos);
}

TEST(WaitcheckTest, Gfx1201DisablingExpertModeDoesNotClearSgprHazards) {
  constexpr uint32_t kHwRegWaveSchedMode = 26;
  std::vector<uint32_t> program;
  append_inst(program, s_setreg_imm32_b32(hwreg(kHwRegWaveSchedMode, 0, 2), 2));
  append_inst(program, v_add_f32_e32(0, 102, 0));
  append_inst(program, s_mov_b32(102, 128));
  append_inst(program, s_setreg_imm32_b32(hwreg(kHwRegWaveSchedMode, 0, 2), 0));
  append_inst(program, v_add_f32_e32(1, 102, 1));

  TestCodeObject code_object(program);
  auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].kind, WaitcheckDiagnosticKind::SgprDepctr);
  EXPECT_NE(report.diagnostics[0].message.find("depctr_sa_sdst(0)"), std::string::npos);
}

TEST(WaitcheckTest, Gfx1201CodeObjectTracksAluDependenciesInExpertMode) {
  constexpr uint32_t kHwRegWaveSchedMode = 26;
  std::vector<uint32_t> program;
  append_inst(program, s_setreg_imm32_b32(hwreg(kHwRegWaveSchedMode, 0, 2), 2));
  append_inst(program, v_add_f32_e32(0, 102, 0));
  append_inst(program, s_mov_b32(102, 128));
  append_inst(program, v_add_f32_e32(1, 102, 1));

  TestCodeObject code_object(program);
  auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_NE(report.diagnostics[0].message.find("depctr_sa_sdst(0)"), std::string::npos);
}

TEST(WaitcheckTest, ExpertValuRegisterHazardsRequireOrderedIssueCredits) {
  for (auto arch : {ROCJITSU_CODE_ARCH_RDNA4, ROCJITSU_CODE_ARCH_CDNA5}) {
    for (bool read : {false, true}) {
      for (bool mixed : {false, true}) {
        for (unsigned count : {0u, 1u, 2u, 15u}) {
          SCOPED_TRACE(testing::Message() << arch << "/" << read << "/" << mixed << "/" << count);
          std::vector<uint32_t> code;
          append_inst(code, s_setreg_imm32_b32(hwreg(26, 0, 2), 2));
          append_inst(code, v_mov_b32(read ? 8 : 12, read ? 3 : 8));
          if (mixed)
            append_inst(code, v_sqrt_f32(14, 11));
          else
            append_inst(code, v_mov_b32(14, 11));
          append_inst(code, sopp(8, 0x0f9f | (count << 12)));
          append_inst(code, global_load_b32(read ? 4 : 8, read ? 8 : 2));
          append_inst(code, s_endpgm());
          const auto report = analyze_waitcnts(TestCodeObject(code), arch);
          ASSERT_TRUE(report.supported) << report.analysis_error;
          const unsigned required = mixed ? 0 : 1;
          ASSERT_EQ(report.diagnostics.size(), count > required ? 1u : 0u)
              << diagnostic_summary(report);
          if (count > required) {
            const auto &d = report.diagnostics.front();
            EXPECT_EQ(d.counter, WaitCounterKind::VaVdst);
            EXPECT_EQ(d.required_count, required);
            EXPECT_EQ(d.reg.index, 8u);
            EXPECT_EQ(d.access, read ? WaitcheckAccessKind::Use : WaitcheckAccessKind::Def);
          }
        }
      }
    }
    for (unsigned mode : {0u, 2u}) {
      std::vector<uint32_t> code;
      append_inst(code, s_setreg_imm32_b32(hwreg(26, 0, 2), mode));
      append_inst(code, v_mov_b32(8, 3));
      append_inst(code, v_mov_b32(8, 8));       // Interlocked VALU WAW and RAW.
      append_inst(code, global_load_b32(8, 2)); // External WAW on v8.
      append_inst(code, s_endpgm());
      const auto report = analyze_waitcnts(TestCodeObject(code), arch);
      ASSERT_TRUE(report.supported) << report.analysis_error;
      EXPECT_EQ(std::ranges::any_of(report.diagnostics,
                                    [](const auto &d) {
                                      return d.counter == WaitCounterKind::VaVdst &&
                                             d.access == WaitcheckAccessKind::Def;
                                    }),
                mode == 2)
          << diagnostic_summary(report);
      for (const auto &d : report.diagnostics)
        EXPECT_EQ(d.section_offset, 16u) << diagnostic_summary(report);
    }
  }
}

TEST(WaitcheckTest, ExpertDelaySelectsTheCorrectValuIssueHistory) {
  for (auto arch : {ROCJITSU_CODE_ARCH_RDNA4, ROCJITSU_CODE_ARCH_CDNA5}) {
    for (bool trans : {false, true}) {
      for (unsigned dep : {0u, 1u, 2u, 5u, 6u}) {
        SCOPED_TRACE(testing::Message() << arch << "/" << trans << "/" << dep);
        std::vector<uint32_t> code;
        append_inst(code, s_setreg_imm32_b32(hwreg(26, 0, 2), 2));
        if (trans)
          append_inst(code, v_sqrt_f32(8, 3));
        else
          append_inst(code, v_mov_b32(8, 3));
        append_inst(code, sopp(7, dep)); // s_delay_alu applies to the following VALU.
        append_inst(code, v_mov_b32(12, 11));
        append_inst(code, global_load_b32(4, 8));
        append_inst(code, s_endpgm());
        const auto report = analyze_waitcnts(TestCodeObject(code), arch);
        ASSERT_TRUE(report.supported) << report.analysis_error;
        EXPECT_EQ(std::ranges::any_of(report.diagnostics,
                                      [](const auto &d) {
                                        return d.counter == WaitCounterKind::VaVdst &&
                                               d.reg.index == 8;
                                      }),
                  dep != (trans ? 5u : 1u))
            << diagnostic_summary(report);
      }
    }
  }
}

TEST(WaitcheckTest, ExpertCounterParityReplaysLaterAluDelay) {
  for (const auto [arch, machine] :
       {std::pair{ROCJITSU_CODE_ARCH_RDNA4, EF_AMDGPU_MACH_AMDGCN_GFX1201},
        std::pair{ROCJITSU_CODE_ARCH_CDNA5, EF_AMDGPU_MACH_AMDGCN_GFX1250}}) {
    for (const bool receives_when_empty : {false, true}) {
      SCOPED_TRACE(arch);
      SCOPED_TRACE(receives_when_empty);
      std::vector<uint32_t> code;
      append_inst(code, s_setreg_imm32_b32(hwreg(26, 0, 2), 2));
      append_inst(code, s_mov_b32(126, 129));
      append_inst(code, v_mov_b32(8, 3));
      const auto field = code.size() * sizeof(uint32_t);
      append_inst(code, sopp(8, 0x0f9f)); // s_wait_alu depctr_va_vdst(0)
      append_inst(code, s_mov_b32(126, 128));
      append_inst(code, s_delay_alu(1));
      if (receives_when_empty) {
        const auto readlane = build_v_readlane_b32(9, 20, 0, arch);
        ASSERT_TRUE(readlane.succeeded());
        append_inst(code, readlane.value());
      } else {
        append_inst(code, v_mov_b32(20, 3));
      }
      append_inst(code, s_mov_b32(126, 129));
      append_inst(code, global_load_b32(9, 8));
      append_inst(code, s_endpgm());
      const auto bytes = waitcheck_test::make_gfx_code_object(code, machine, true);
      const AmdGpuCodeObject object(bytes.data(), bytes.size());
      WaitcheckOptions options;
      options.check_counter_parity = true;
      const auto report = analyze_waitcnts_for_kernel(object, arch, 0, options);
      ASSERT_TRUE(report.complete) << report.analysis_error;
      EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
      const auto row = std::ranges::find_if(report.counter_underaccounting_diagnostics,
                                            [&](const auto &diagnostic) {
                                              return diagnostic.wait_section_offset == field &&
                                                     diagnostic.counter == WaitCounterKind::VaVdst;
                                            });
      if (receives_when_empty) {
        ASSERT_NE(row, report.counter_underaccounting_diagnostics.end());
        EXPECT_FALSE(row->has_required_dependency);
      } else {
        EXPECT_EQ(row, report.counter_underaccounting_diagnostics.end());
      }
    }
  }
}

TEST(WaitcheckTest, ExpertWave32VccHiPreservesTheSavedBranchCondition) {
  for (const auto [arch, machine] :
       {std::pair{ROCJITSU_CODE_ARCH_RDNA4, EF_AMDGPU_MACH_AMDGCN_GFX1201},
        std::pair{ROCJITSU_CODE_ARCH_CDNA5, EF_AMDGPU_MACH_AMDGCN_GFX1250}}) {
    for (const uint32_t dst : {106u, 107u}) {
      SCOPED_TRACE(arch);
      SCOPED_TRACE(dst);
      std::vector<uint32_t> code;
      append_inst(code, s_setreg_imm32_b32(hwreg(26, 0, 2), 2));
      append_inst(code, s_mov_b32(126, 131)); // EXEC_LO = 3.
      code.push_back(0xbf068004);            // s_cmp_eq_u32 s4, 0
      append_inst(code, sopp(34, 3));        // s_cbranch_scc1 alternate
      const auto producer = code.size() * sizeof(uint32_t);
      append_inst(code, v_mov_b32(16, 2));
      append_inst(code, s_mov_b32(9, 128)); // s9 = 0 on the producer arm.
      append_inst(code, sopp(32, 1));      // s_branch join
      append_inst(code, s_mov_b32(9, 193)); // alternate: s9 = -1
      code.push_back(0x8b6a097e);           // join: s_and_b32 vcc_lo, exec_lo, s9
      append_inst(code, s_mov_b32(dst, 129));
      append_inst(code, v_mov_b32(32, 2));
      const size_t branch = code.size();
      append_inst(code, sopp(35)); // s_cbranch_vccz done
      append_inst(code, sopp(8, 0x2f9f)); // VA_VDST(2).
      const auto consumer = code.size() * sizeof(uint32_t);
      append_inst(code, global_load_b32(12, 16));
      if (arch == ROCJITSU_CODE_ARCH_CDNA5)
        append_inst(code, sopp(69, 0)); // XCNT(0).
      append_inst(code, sopp(64, 0));   // LOADCNT(0).
      code[branch] |= static_cast<uint32_t>(code.size() - branch - 1);
      append_inst(code, s_endpgm());

      const auto bytes = waitcheck_test::make_gfx_code_object(code, machine, true);
      const AmdGpuCodeObject object(bytes.data(), bytes.size());
      const auto report = analyze_waitcnts_for_kernel(object, arch, 0);
      ASSERT_TRUE(report.supported && report.complete) << report.analysis_error;
      ASSERT_EQ(report.diagnostics_observed, dst == 106 ? 1u : 0u) << diagnostic_summary(report);
      if (dst == 106) {
        const auto &diagnostic = report.diagnostics.front();
        EXPECT_EQ(diagnostic.counter, WaitCounterKind::VaVdst);
        EXPECT_EQ(diagnostic.producer_section_offset, producer);
        EXPECT_EQ(diagnostic.section_offset, consumer);
        EXPECT_EQ(diagnostic.reg, (RegisterRef{RegClass::VGPR, 16, 1}));
        EXPECT_EQ(diagnostic.required_count, 1u);
      }
    }
  }
}

TEST(WaitcheckTest, ExpertParityKeepsOnlyProducersThatCanReachTheVccGuardedWait) {
  enum class Control {
    Zero,
    Nonzero,
    ClobberMask,
    ClobberVcc,
    ClobberVccHi,
    ClobberVccPair,
    ImplicitVcc,
    Relative,
    Deferred,
    Drained,
    Cutoff
  };
  for (const auto [arch, machine] :
       {std::pair{ROCJITSU_CODE_ARCH_RDNA4, EF_AMDGPU_MACH_AMDGCN_GFX1201},
        std::pair{ROCJITSU_CODE_ARCH_CDNA5, EF_AMDGPU_MACH_AMDGCN_GFX1250}}) {
    for (const auto control : {Control::Zero, Control::Nonzero, Control::ClobberMask,
                               Control::ClobberVcc, Control::ClobberVccHi, Control::ClobberVccPair,
                               Control::ImplicitVcc, Control::Relative, Control::Deferred,
                               Control::Drained, Control::Cutoff}) {
      SCOPED_TRACE(arch);
      SCOPED_TRACE(static_cast<unsigned>(control));
      std::vector<uint32_t> code;
      append_inst(code, s_setreg_imm32_b32(hwreg(26, 0, 2), 2));
      append_inst(code, s_mov_b32(126, 129)); // EXEC_LO = 1.
      const auto older = code.size() * sizeof(uint32_t);
      append_inst(code, v_mov_b32(24, 3));
      for (unsigned i = 0; i < 16; ++i)
        append_inst(code, v_mov_b32(6, 5));
      code.push_back(0xbf068004); // s_cmp_eq_u32 s4, 0
      const size_t skip = code.size();
      append_inst(code, sopp(34)); // s_cbranch_scc1 alternate.
      const auto younger = code.size() * sizeof(uint32_t);
      append_inst(code, v_mov_b32(16, 2));
      if (control == Control::Deferred || control == Control::Drained)
        append_inst(code, s_load_b32(9, 6));
      if (control == Control::Drained)
        append_inst(code, sopp(71, 0)); // s_wait_kmcnt 0.
      append_inst(code, s_mov_b32(9, control == Control::Nonzero ? 193 : 128));
      if (control == Control::Relative)
        code.push_back(0xbe844205); // s_movreld_b32 s4, s5
      if (control == Control::Cutoff)
        for (unsigned i = 0; i < 65; ++i)
          append_inst(code, sopp(0));
      const size_t join = code.size();
      append_inst(code, sopp(32)); // s_branch join.
      code[skip] |= static_cast<uint32_t>(code.size() - skip - 1);
      append_inst(code, s_mov_b32(9, 193)); // Alternate arm sets s9 = -1.
      code[join] |= static_cast<uint32_t>(code.size() - join - 1);
      if (control == Control::ClobberMask)
        append_inst(code, s_mov_b32(9, 193));
      code.push_back(0x8b6a097e);           // s_and_b32 vcc_lo, exec_lo, s9
      append_inst(code, s_mov_b32(9, 193)); // Capture survives this overwrite.
      append_inst(code, v_mov_b32(32, 2));  // One younger issue before the field.
      if (control == Control::ClobberVcc)
        append_inst(code, s_mov_b32(106, 129));
      if (control == Control::ClobberVccHi)
        append_inst(code, s_mov_b32(107, 129));
      if (control == Control::ClobberVccPair)
        code.push_back(0xbeea0181); // s_mov_b64 vcc, 1
      if (control == Control::ImplicitVcc)
        code.push_back(0x7c940702); // v_cmp_eq_u32_e32 vcc_lo, v2, v3
      const size_t bypass = code.size();
      append_inst(code, sopp(35)); // s_cbranch_vccz exit.
      const auto wait = code.size() * sizeof(uint32_t);
      append_inst(code, sopp(64, 0));     // Grouped LOADCNT.
      append_inst(code, sopp(8, 0x0f9f)); // VA_VDST(0).
      append_inst(code, sopp(0));
      append_inst(code, global_load_b32(12, 16));
      append_inst(code, global_load_b32(14, 24));
      code[bypass] |= static_cast<uint32_t>(code.size() - bypass - 1);
      append_inst(code, s_endpgm());
      const auto bytes = waitcheck_test::make_gfx_code_object(code, machine, true);
      const AmdGpuCodeObject object(bytes.data(), bytes.size());
      WaitcheckOptions options;
      options.check_counter_parity = true;
      const auto report = analyze_waitcnts_for_kernel(object, arch, 0, options);
      ASSERT_TRUE(report.complete) << report.analysis_error;
      const auto &rows = report.counter_underaccounting_diagnostics;
      const auto field = std::ranges::find_if(rows, [&](const auto &row) {
        return row.counter == WaitCounterKind::VaVdst && row.wait_section_offset == wait;
      });
      const auto before = analyze_waitcnts_for_kernel(object, arch, 0, WaitcheckOptions{});
      EXPECT_EQ(before.diagnostics.size(), report.diagnostics.size());
      const bool bypassed = control == Control::Zero || control == Control::Drained ||
                            control == Control::ClobberVccHi;
      if (bypassed) {
        ASSERT_NE(field, rows.end());
        EXPECT_TRUE(field->has_required_dependency);
        EXPECT_EQ(field->producer_section_offset, older);
        EXPECT_EQ(field->required_count, 14u);
      } else if (control == Control::Nonzero || control == Control::ClobberMask ||
                 control == Control::ClobberVcc || control == Control::ClobberVccPair ||
                 control == Control::ImplicitVcc) {
        ASSERT_NE(field, rows.end());
        EXPECT_TRUE(field->has_required_dependency);
        EXPECT_EQ(field->producer_section_offset, younger);
      }
    }
  }
}

TEST(WaitcheckTest, ExpertCounterParityUsesDelayAfterWaitGroup) {
  enum class Primary { None, Before, After };
  for (const auto [arch, machine] :
       {std::pair{ROCJITSU_CODE_ARCH_RDNA4, EF_AMDGPU_MACH_AMDGCN_GFX1201},
        std::pair{ROCJITSU_CODE_ARCH_CDNA5, EF_AMDGPU_MACH_AMDGCN_GFX1250}}) {
    for (const auto primary : {Primary::None, Primary::Before, Primary::After}) {
      for (const uint32_t count : {0u, 2u}) {
        SCOPED_TRACE(arch);
        SCOPED_TRACE(static_cast<int>(primary));
        SCOPED_TRACE(count);
        std::vector<uint32_t> code;
        append_inst(code, s_setreg_imm32_b32(hwreg(26, 0, 2), 2));
        append_inst(code, s_mov_b32(126, 129));
        const auto producer = code.size() * sizeof(uint32_t);
        append_inst(code, v_mov_b32(8, 3));
        append_inst(code, s_delay_alu(1));
        const auto field = code.size() * sizeof(uint32_t);
        if (primary == Primary::Before)
          append_inst(code, sopp(64, 0)); // LOADCNT(0).
        append_inst(code, sopp(8, 0x0f9f | (count << 12)));
        if (primary == Primary::After)
          append_inst(code, sopp(64, 0));
        append_inst(code, v_mov_b32(12, 11));
        const auto consumer = code.size() * sizeof(uint32_t);
        append_inst(code, ds_load_b32(8, 4));
        append_inst(code, s_endpgm());
        const auto bytes = waitcheck_test::make_gfx_code_object(code, machine, true);
        const AmdGpuCodeObject object(bytes.data(), bytes.size());
        WaitcheckOptions options;
        options.check_counter_parity = true;
        const auto report = analyze_waitcnts_for_kernel(object, arch, 0, options);
        ASSERT_TRUE(report.complete) << report.analysis_error;
        const bool delay_survives = arch == ROCJITSU_CODE_ARCH_CDNA5 && primary == Primary::None;
        const auto missing = std::ranges::find_if(report.diagnostics, [&](const auto &row) {
          return row.counter == WaitCounterKind::VaVdst &&
                 row.producer_section_offset == producer && row.section_offset == consumer &&
                 row.reg.index == 8;
        });
        EXPECT_EQ(missing != report.diagnostics.end(), !delay_survives && count == 2);
        if (missing != report.diagnostics.end())
          EXPECT_EQ(missing->required_count, 1u);
        const auto parity =
            std::ranges::find_if(report.counter_underaccounting_diagnostics, [&](const auto &row) {
              return row.counter == WaitCounterKind::VaVdst && row.wait_section_offset == field;
            });
        EXPECT_EQ(parity != report.counter_underaccounting_diagnostics.end(), delay_survives);
        if (parity != report.counter_underaccounting_diagnostics.end())
          EXPECT_FALSE(parity->has_required_dependency);
      }
    }
  }
}

TEST(WaitcheckTest, ExpertDelayCountsEmptyExecValuPositionsAcrossAJoin) {
  for (const auto [arch, machine] :
       {std::pair{ROCJITSU_CODE_ARCH_RDNA4, EF_AMDGPU_MACH_AMDGCN_GFX1201},
        std::pair{ROCJITSU_CODE_ARCH_CDNA5, EF_AMDGPU_MACH_AMDGCN_GFX1250}}) {
    for (unsigned mask : {128u, 129u, 7u}) { // Literal zero, one, or unknown s7.
      for (unsigned tag : {0u, 1u, 2u, 3u, 4u, (3u << 7) | (1u << 4)}) {
        std::vector<uint32_t> code;
        append_inst(code, s_setreg_imm32_b32(hwreg(26, 0, 2), 2));
        append_inst(code, s_mov_b32(126, 129));
        const auto producer = code.size() * sizeof(uint32_t);
        append_inst(code, v_mov_b32(8, 3));
        append_inst(code, s_cmp_eq_u32(1, 128));
        append_inst(code, sopp(34, 2));
        append_inst(code, v_mov_b32(14, 17));
        append_inst(code, sopp(32, 1));
        code.push_back(0x041c2510); // v_add_f64_e32 v[14:15], v[16:17], v[18:19].
        append_inst(code, s_mov_b32(126, mask));
        append_inst(code, v_mov_b32(22, 23));
        append_inst(code, s_mov_b32(126, 129));
        append_inst(code, sopp(7, tag));
        if (tag > 15)
          append_inst(code, sopp(0));
        append_inst(code, v_mov_b32(16, 18));
        const auto consumer = code.size() * sizeof(uint32_t);
        append_inst(code, global_load_b32(9, 8));
        append_inst(code, s_endpgm());
        const auto image = waitcheck_test::make_gfx_code_object(code, machine, true);
        AmdGpuCodeObject object(image.data(), image.size());
        const auto report = analyze_waitcnts(object, arch);
        ASSERT_TRUE(report.supported && report.complete) << report.analysis_error;
        const bool completed = (mask != 7 && (tag == 3 || tag > 15)) || (mask == 129 && tag == 1);
        EXPECT_EQ(std::ranges::any_of(report.diagnostics,
                                      [&](const auto &d) {
                                        return d.counter == WaitCounterKind::VaVdst &&
                                               d.reg.index == 8 &&
                                               d.producer_section_offset == producer &&
                                               d.section_offset == consumer;
                                      }),
                  !completed)
            << mask << '/' << tag << ' ' << diagnostic_summary(report);
      }
    }
  }
}

TEST(WaitcheckTest, ExpertDelayedValuUsesItsOwnHistoryOnEachCfgPath) {
  for (const auto [arch, machine] :
       {std::pair{ROCJITSU_CODE_ARCH_RDNA4, EF_AMDGPU_MACH_AMDGCN_GFX1201},
        std::pair{ROCJITSU_CODE_ARCH_CDNA5, EF_AMDGPU_MACH_AMDGCN_GFX1250}}) {
    for (const unsigned tag : {0u, 2u, 3u, 5u, (2u << 7) | (1u << 4)}) {
      for (const bool bypass_delay : {false, true}) {
        for (const bool unknown_issue : {false, true}) {
          SCOPED_TRACE(testing::Message()
                       << arch << '/' << tag << '/' << bypass_delay << '/' << unknown_issue);
          std::vector<uint32_t> code;
          append_inst(code, s_setreg_imm32_b32(hwreg(26, 0, 2), 2));
          append_inst(code, s_mov_b32(126, 129));
          const auto producer = code.size() * sizeof(uint32_t);
          append_inst(code, v_mov_b32(8, 3));
          append_inst(code, s_cmp_eq_u32(1, 128));
          append_inst(code, sopp(34, 2)); // s_cbranch_scc1 to DP.
          append_inst(code, v_mov_b32(14, 17));
          const auto branch = code.size();
          append_inst(code, sopp(32, 1)); // s_branch to the join.
          code.push_back(0x041c2510);     // v_add_f64_e32 v[14:15], v[16:17], v[18:19].
          if (unknown_issue) {
            append_inst(code, s_mov_b32(126, 5));
            append_inst(code, v_sqrt_f32(22, 23)); // May skip; different delay class.
            append_inst(code, s_mov_b32(126, 129));
          }
          if (!bypass_delay)
            code[branch] = std::bit_cast<uint32_t>(sopp(32, code.size() - branch - 1));
          append_inst(code, sopp(7, tag)); // S_DELAY_ALU.
          if (tag > 15)
            append_inst(code, sopp(0)); // S_NOP consumes the SKIP position.
          if (bypass_delay)
            code[branch] = std::bit_cast<uint32_t>(sopp(32, code.size() - branch - 1));
          append_inst(code, v_mov_b32(16, 18));
          const auto consumer = code.size() * sizeof(uint32_t);
          append_inst(code, global_load_b32(9, 8));
          append_inst(code, s_endpgm());
          const auto image = waitcheck_test::make_gfx_code_object(code, machine, true);
          AmdGpuCodeObject object(image.data(), image.size());
          const auto report = analyze_waitcnts(object, arch);
          ASSERT_TRUE(report.supported && report.complete) << report.analysis_error;
          // An ambiguous issue conservatively invalidates the local delay
          // history. A bypass, wrong tag or older empty slot cannot help.
          const bool completed = !unknown_issue && !bypass_delay && (tag == 2 || tag > 15);
          EXPECT_EQ(std::ranges::any_of(report.diagnostics,
                                        [&](const auto &d) {
                                          return d.counter == WaitCounterKind::VaVdst &&
                                                 d.reg.index == 8 &&
                                                 d.producer_section_offset == producer &&
                                                 d.section_offset == consumer;
                                        }),
                    !completed)
              << diagnostic_summary(report);
        }
      }
    }
  }
}

TEST(WaitcheckTest, ExpertSecondDelayTagSkipsTheSelectedInstructions) {
  for (auto arch : {ROCJITSU_CODE_ARCH_RDNA4, ROCJITSU_CODE_ARCH_CDNA5}) {
    for (bool wait : {false, true}) {
      for (unsigned skip : {0u, 1u, 2u, 5u, 6u}) {
        SCOPED_TRACE(testing::Message() << arch << "/" << wait << "/" << skip);
        std::vector<uint32_t> code;
        append_inst(code, s_setreg_imm32_b32(hwreg(26, 0, 2), 2));
        append_inst(code, v_mov_b32(8, 3));
        append_inst(code, sopp(7, (1u << 7) | (skip << 4)));
        append_inst(code, wait ? sopp(8, 0xff9f) : sopp(0, 0));
        append_inst(code, v_mov_b32(14, 11));
        append_inst(code, global_load_b32(4, 8));
        append_inst(code, s_endpgm());
        const auto report = analyze_waitcnts(TestCodeObject(code), arch);
        ASSERT_TRUE(report.supported) << report.analysis_error;
        const unsigned selected = wait && arch == ROCJITSU_CODE_ARCH_CDNA5 ? 0 : 1;
        EXPECT_EQ(std::ranges::any_of(report.diagnostics,
                                      [](const auto &d) {
                                        return d.counter == WaitCounterKind::VaVdst &&
                                               d.reg.index == 8;
                                      }),
                  skip != selected)
            << diagnostic_summary(report);
      }
    }
  }
}

TEST(WaitcheckTest, Gfx1250MatrixOperandWaitThresholdIncludesCorePositions) {
  for (const unsigned count : {0u, 2u, 3u, 15u}) {
    SCOPED_TRACE(count);
    std::vector<uint32_t> code;
    append_inst(code, s_setreg_imm32_b32(hwreg(26, 0, 2), 2));
    // v_wmma_f32_16x16x32_bf16 v[0:7], v[8:15], v[16:23], v[0:7].
    code.insert(code.end(), {0xcc620000, 0x1c022108});
    for (uint8_t reg = 28; reg <= 34; ++reg)
      append_inst(code, v_mov_b32(reg, 27));
    append_inst(code, sopp(8, 0x0f9f | (count << 12)));
    append_inst(code, global_load_b32(8, 24));
    append_inst(code, s_endpgm());
    WaitcheckOptions options;
    options.check_counter_parity = true;
    const auto image =
        waitcheck_test::make_gfx_code_object(code, EF_AMDGPU_MACH_AMDGCN_GFX1250, true);
    const AmdGpuCodeObject object(image.data(), image.size());
    const auto report = analyze_waitcnts(object, ROCJITSU_CODE_ARCH_CDNA5, options);
    ASSERT_TRUE(report.complete && report.supported) << report.analysis_error;
    auto hazard = std::ranges::find_if(report.diagnostics, [](const auto &diagnostic) {
      return diagnostic.counter == WaitCounterKind::VaVdst && diagnostic.reg.index == 8;
    });
    ASSERT_EQ(hazard != report.diagnostics.end(), count > 2) << diagnostic_summary(report);
    if (hazard != report.diagnostics.end())
      EXPECT_EQ(hazard->required_count, 2u);
    EXPECT_EQ(report.counter_parity_exact, unsigned(count == 2));
  }
}

TEST(WaitcheckTest, Gfx1250CompletedCoreValuOrdersEarlierMatrixOperand) {
  // v_wmma_f32_16x16x32_f16 v[0:7], v[40:47], v[114:121], v[0:7].
  for (unsigned unrelated : {4u, 5u, 6u}) {
    std::vector<uint32_t> code;
    append_inst(code, s_setreg_imm32_b32(hwreg(26, 0, 2), 2));
    code.insert(code.end(), {0xcc600000, 0x1c02e528});
    for (unsigned i = 0; i < unrelated; ++i)
      append_inst(code, v_mov_b32(12, 11));
    append_inst(code, sopp(8, 0x1f9f)); // VA_VDST(1) leaves one core instruction.
    append_inst(code, global_load_b32(42, 8));
    append_inst(code, s_endpgm());
    const auto report = analyze_waitcnts(TestCodeObject(code), ROCJITSU_CODE_ARCH_CDNA5);
    ASSERT_TRUE(report.supported) << report.analysis_error;
    EXPECT_EQ(std::ranges::any_of(report.diagnostics,
                                  [](const auto &d) {
                                    return d.counter == WaitCounterKind::VaVdst &&
                                           d.reg.index == 42;
                                  }),
              unrelated < 6)
        << diagnostic_summary(report);
  }
}

TEST(WaitcheckTest, ExpertNonemptyExecEdgeSuppliesOnlyItsOwnValuCredits) {
  for (const auto [arch, machine] :
       {std::pair{ROCJITSU_CODE_ARCH_RDNA4, EF_AMDGPU_MACH_AMDGCN_GFX1201},
        std::pair{ROCJITSU_CODE_ARCH_CDNA5, EF_AMDGPU_MACH_AMDGCN_GFX1250}}) {
    for (bool alternate : {false, true}) {
      std::vector<uint32_t> code;
      append_inst(code, s_setreg_imm32_b32(hwreg(26, 0, 2), 2));
      append_inst(code, s_mov_b32(126, 3)); // Unknown s3.
      if (alternate) {
        append_inst(code, s_cmp_eq_u32(2, 128));
        append_inst(code, sopp(34, 1)); // A second edge bypasses the EXEC test.
      }
      const size_t branch = code.size();
      append_inst(code, sopp(37)); // s_cbranch_execz to the end.
      // READLANE uses v8 independently of EXEC. Only the following ordinary
      // VALU is conditional on the tested mask.
      code.push_back(0xd7600001); // v_readlane_b32 s1, v8, 0.
      code.push_back(0x02010108);
      append_inst(code, v_mov_b32(12, 3));
      append_inst(code, s_mov_b32(126, 129));
      append_inst(code, sopp(8, 0x1f9f));
      append_inst(code, ds_load_b32(8, 5));
      code[branch] = std::bit_cast<uint32_t>(sopp(37, code.size() - branch - 1));
      append_inst(code, s_endpgm());
      const auto image = waitcheck_test::make_gfx_code_object(code, machine, true);
      AmdGpuCodeObject object(image.data(), image.size());
      const auto report = analyze_waitcnts(object, arch);
      ASSERT_TRUE(report.complete) << report.analysis_error;
      EXPECT_EQ(std::ranges::any_of(report.diagnostics,
                                    [](const auto &diag) {
                                      return diag.counter == WaitCounterKind::VaVdst &&
                                             diag.reg.index == 8;
                                    }),
                alternate)
          << diagnostic_summary(report);
    }
  }
}

TEST(WaitcheckTest, ExpertValuIssueUsesAllNonemptyLoopEntryEdges) {
  for (const auto [arch, machine] :
       {std::pair{ROCJITSU_CODE_ARCH_RDNA4, EF_AMDGPU_MACH_AMDGCN_GFX1201},
        std::pair{ROCJITSU_CODE_ARCH_CDNA5, EF_AMDGPU_MACH_AMDGCN_GFX1250}}) {
    for (bool nonempty_backedge : {false, true}) {
      for (unsigned wait : {0u, 1u, 2u}) {
        std::vector<uint32_t> code;
        append_inst(code, s_setreg_imm32_b32(hwreg(26, 0, 2), 2));
        append_inst(code, s_mov_b32(126, 3));
        const size_t forward = code.size();
        append_inst(code, sopp(37));
        const auto producer = code.size() * sizeof(uint32_t);
        code.insert(code.end(), {0xd7600001, 0x02010108}); // READLANE reads v8 for any EXEC.
        append_inst(code, v_mov_b32(12, 9));
        append_inst(code, s_mov_b32(126, 129)); // The consumer can always write v8.
        const auto emitted = code.size() * sizeof(uint32_t);
        append_inst(code, sopp(8, 0x0f9f | (wait << 12)));
        const auto consumer = code.size() * sizeof(uint32_t);
        append_inst(code, ds_load_b32(8, 4));
        append_inst(code, s_mov_b32(126, 5));
        // Alternate the backedge polarity. A zero-taking edge cannot guarantee
        // that the MOV after READLANE occupies a VA FIFO position.
        append_inst(code, sopp(nonempty_backedge ? 38 : 37,
                               static_cast<int>(forward) - static_cast<int>(code.size())));
        code[forward] = std::bit_cast<uint32_t>(sopp(37, code.size() - forward - 1));
        append_inst(code, s_endpgm());
        const auto image = waitcheck_test::make_gfx_code_object(code, machine, true);
        AmdGpuCodeObject object(image.data(), image.size());
        WaitcheckOptions options;
        options.check_counter_parity = true;
        const auto report = analyze_waitcnts(object, arch, options);
        ASSERT_TRUE(report.complete);
        const auto missing = std::ranges::find_if(report.diagnostics, [&](const auto &d) {
          return d.counter == WaitCounterKind::VaVdst && d.producer_section_offset == producer &&
                 d.section_offset == consumer && d.reg.index == 8;
        });
        const unsigned bound = nonempty_backedge ? 1u : 0u;
        EXPECT_EQ(missing != report.diagnostics.end(), wait > bound);
        if (missing != report.diagnostics.end())
          EXPECT_EQ(missing->required_count, bound);
        const bool stronger =
            std::ranges::any_of(report.counter_underaccounting_diagnostics, [&](const auto &d) {
              return d.counter == WaitCounterKind::VaVdst && d.wait_section_offset == emitted &&
                     d.has_required_dependency && d.required_count == bound;
            });
        EXPECT_EQ(stronger, wait < bound);
      }
    }
  }
}

TEST(WaitcheckTest, ExpertValuIssueRestoresWave32ScalarVccSaves) {
  enum class Write {
    None,
    Scalar,
    Pair,
    Compare,
    ScalarCompare,
    Relative,
    PendingLoad,
    OtherLoad,
    DifferentHalf,
    Cmpx
  };
  for (const auto [arch, machine] :
       {std::pair{ROCJITSU_CODE_ARCH_RDNA4, EF_AMDGPU_MACH_AMDGCN_GFX1201},
        std::pair{ROCJITSU_CODE_ARCH_CDNA5, EF_AMDGPU_MACH_AMDGCN_GFX1250}}) {
    for (unsigned saved : {106u, 107u}) {
      for (auto write : {Write::None, Write::Scalar, Write::Pair, Write::Compare,
                         Write::ScalarCompare, Write::Relative, Write::PendingLoad,
                         Write::OtherLoad, Write::DifferentHalf, Write::Cmpx}) {
        for (unsigned count : {0u, 1u, 2u}) {
          std::vector<uint32_t> code;
          append_inst(code, s_setreg_imm32_b32(hwreg(26, 0, 2), 2));
          append_inst(code, s_mov_b32(126, 129)); // One issuing lane.
          if (write == Write::PendingLoad || write == Write::OtherLoad)
            append_inst(code, s_load_b32(write == Write::PendingLoad ? saved : saved ^ 1u, 12));
          if (write == Write::Cmpx) {
            append_inst(code, s_mov_b32(saved, 126));
            code.push_back(0x7d894623); // GFX12 CMPX writes EXEC.
          } else {
            code.push_back(0xbe802080 | (saved << 16)); // s_and_saveexec_b32 saved, 0.
          }
          switch (write) {
          case Write::Scalar:
            append_inst(code, s_mov_b32(saved, 128));
            break;
          case Write::Pair:
            code.push_back(0xbeea0180); // s_mov_b64 vcc, 0.
            break;
          case Write::Compare:
            append_inst(code, v_cmp_gt_u32_e32(5, 12)); // Implicit VCC mask destination.
            break;
          case Write::ScalarCompare:
            append_v_cmp_gt_u32_s2_s5_v12(code); // Compare writes s2, leaving VCC stable.
            break;
          case Write::Relative:
            code.push_back(0xbe804280); // s_movreld_b32 s0, 0 with unknown M0.
            break;
          default:
            break;
          }
          const unsigned mask = write == Write::DifferentHalf ? (saved ^ 1u) : saved;
          code.push_back(0x8c7e007e | (mask << 8)); // s_or_b32 exec_lo, exec_lo, mask.
          const auto producer = code.size() * sizeof(uint32_t);
          code.insert(code.end(), {0xd7600001, 0x02010108}); // READLANE reads v8 for any EXEC.
          append_inst(code, v_mov_b32(12, 9));
          append_inst(code, s_mov_b32(126, 129));
          append_inst(code, sopp(8, 0x0f9f | (count << 12)));
          const auto consumer = code.size() * sizeof(uint32_t);
          append_inst(code, ds_load_b32(8, 4));
          append_inst(code, s_endpgm());
          const auto image = waitcheck_test::make_gfx_code_object(code, machine, true);
          AmdGpuCodeObject object(image.data(), image.size());
          const auto report = analyze_waitcnts(object, arch);
          ASSERT_TRUE(report.complete);
          const auto missing = std::ranges::find_if(report.diagnostics, [&](const auto &d) {
            return d.counter == WaitCounterKind::VaVdst && d.producer_section_offset == producer &&
                   d.section_offset == consumer && d.reg.index == 8;
          });
          const unsigned limit =
              write == Write::None || write == Write::Cmpx || write == Write::ScalarCompare ? 1 : 0;
          EXPECT_EQ(missing != report.diagnostics.end(), count > limit)
              << saved << " " << static_cast<int>(write) << " " << diagnostic_summary(report);
          if (missing != report.diagnostics.end())
            EXPECT_EQ(missing->required_count, limit);
        }
      }
    }
  }
}

TEST(WaitcheckTest, ExpertValuIssuePreservesNonemptyExecAcrossSelfOr) {
  for (const auto [arch, machine] :
       {std::pair{ROCJITSU_CODE_ARCH_RDNA4, EF_AMDGPU_MACH_AMDGCN_GFX1201},
        std::pair{ROCJITSU_CODE_ARCH_CDNA5, EF_AMDGPU_MACH_AMDGCN_GFX1250}}) {
    for (bool active : {false, true}) {
      for (bool reads_exec : {false, true}) {
        std::vector<uint32_t> code;
        append_inst(code, s_setreg_imm32_b32(hwreg(26, 0, 2), 2));
        append_inst(code, s_mov_b32(126, active ? 129 : 128));
        // s_or_b32 exec_lo, 0, exec_lo/s8. Only the former retains the old lanes.
        code.push_back(reads_exec ? 0x8c7e7e80 : 0x8c7e0880);
        code.insert(code.end(), {0xd7600001, 0x02010108});
        append_inst(code, v_mov_b32(12, 9));
        append_inst(code, s_mov_b32(126, 129));
        append_inst(code, sopp(8, 0x1f9f));
        append_inst(code, ds_load_b32(8, 4));
        append_inst(code, s_endpgm());
        const auto image = waitcheck_test::make_gfx_code_object(code, machine, true);
        AmdGpuCodeObject object(image.data(), image.size());
        const auto report = analyze_waitcnts(object, arch);
        ASSERT_TRUE(report.complete);
        EXPECT_EQ(std::ranges::any_of(report.diagnostics,
                                      [](const auto &d) {
                                        return d.counter == WaitCounterKind::VaVdst &&
                                               d.reg.index == 8;
                                      }),
                  !(active && reads_exec))
            << diagnostic_summary(report);
      }
    }
  }
}

TEST(WaitcheckTest, Gfx1250SavedNonemptyExecRestoresCoreIssueCredit) {
  for (bool copy : {false, true}) {
    for (bool clobber : {false, true}) {
      for (bool other : {false, true}) {
        std::vector<uint32_t> code;
        append_inst(code, s_setreg_imm32_b32(hwreg(26, 0, 2), 2));
        append_inst(code, s_mov_b32(126, 129));
        if (copy) {
          append_inst(code, s_mov_b32(8, 126));
          append_inst(code, s_mov_b32(126, 6));
        } else {
          code.push_back(0xbe882006); // s_and_saveexec_b32 s8, s6.
        }
        // One predecessor may clobber the saved mask.
        append_inst(code, s_cmp_eq_u32(3, 128));
        append_inst(code, sopp(34, 1));
        if (clobber)
          append_inst(code, s_mov_b32(8, 128));
        else
          append_inst(code, sopp(0));
        code.push_back(other ? 0x8c7e097e : 0x8c7e087e);   // s_or_b32 exec_lo, exec_lo, s9/s8.
        code.insert(code.end(), {0xcc600000, 0x1c02e528}); // WMMA reads v42 even with empty EXEC.
        for (unsigned i = 0; i < 6; ++i)
          append_inst(code, v_mov_b32(12, 11));
        // A later full mask cannot supply issue credit retroactively.
        append_inst(code, s_mov_b32(126, 129));
        append_inst(code, sopp(8, 0x1f9f));
        append_inst(code, ds_load_b32(42, 8));
        append_inst(code, s_endpgm());
        const auto image =
            waitcheck_test::make_gfx_code_object(code, EF_AMDGPU_MACH_AMDGCN_GFX1250, true);
        AmdGpuCodeObject object(image.data(), image.size());
        const auto report = analyze_waitcnts(object, ROCJITSU_CODE_ARCH_CDNA5);
        ASSERT_TRUE(report.complete);
        EXPECT_EQ(std::ranges::any_of(report.diagnostics,
                                      [](const auto &d) {
                                        return d.counter == WaitCounterKind::VaVdst &&
                                               d.reg.index == 42;
                                      }),
                  clobber || other)
            << diagnostic_summary(report);
      }
    }
  }
}

TEST(WaitcheckTest, Gfx1250SavedExecRejectsUnstableScalarRegisters) {
  enum class Change { NestedCopy, PendingLoad, RelativeWrite, ScalarValuWrite, PairWrite };
  for (auto change : {Change::NestedCopy, Change::PendingLoad, Change::RelativeWrite,
                      Change::ScalarValuWrite, Change::PairWrite}) {
    std::vector<uint32_t> code;
    append_inst(code, s_setreg_imm32_b32(hwreg(26, 0, 2), 2));
    append_inst(code, s_mov_b32(126, 129));
    if (change == Change::PendingLoad)
      append_inst(code, s_load_b32(8, 12));
    append_inst(code, s_mov_b32(8, 126)); // Save the known nonempty outer mask.
    switch (change) {
    case Change::NestedCopy:
      append_inst(code, s_mov_b32(9, 126));
      break;
    case Change::RelativeWrite:
      code.push_back(0xbe804280); // s_movreld_b32 s0, 0
      break;
    case Change::ScalarValuWrite:
      code.push_back(0x7e100500); // v_readfirstlane_b32 s8, v0
      break;
    case Change::PairWrite:
      code.push_back(0xbe880180); // s_mov_b64 s[8:9], 0
      break;
    case Change::PendingLoad:
      break;
    }
    append_inst(code, s_mov_b32(126, 6));
    append_inst(code, s_mov_b32(126, 8));              // Restore s8, which must still be stable.
    code.insert(code.end(), {0xcc600000, 0x1c02e528}); // WMMA reads v42.
    for (unsigned i = 0; i < 6; ++i)
      append_inst(code, v_mov_b32(12, 11));
    append_inst(code, s_mov_b32(126, 129));
    append_inst(code, sopp(8, 0x1f9f));
    append_inst(code, ds_load_b32(42, 8));
    append_inst(code, s_endpgm());
    const auto image =
        waitcheck_test::make_gfx_code_object(code, EF_AMDGPU_MACH_AMDGCN_GFX1250, true);
    AmdGpuCodeObject object(image.data(), image.size());
    const auto report = analyze_waitcnts(object, ROCJITSU_CODE_ARCH_CDNA5);
    ASSERT_TRUE(report.complete);
    EXPECT_EQ(std::ranges::any_of(report.diagnostics,
                                  [](const auto &d) {
                                    return d.counter == WaitCounterKind::VaVdst &&
                                           d.reg.index == 42;
                                  }),
              change != Change::NestedCopy)
        << diagnostic_summary(report);
  }
}

TEST(WaitcheckTest, Gfx1250EmptyExecValuDoesNotFillWmmaCoexecutionWindow) {
  for (bool nop : {false, true}) {
    for (unsigned wait : {0u, 1u}) {
      std::vector<uint32_t> code;
      append_inst(code, s_setreg_imm32_b32(hwreg(26, 0, 2), 2));
      append_inst(code, s_mov_b32(126, 129));
      code.insert(code.end(), {0xcc600000, 0x1c02e528}); // WMMA reads v42.
      for (unsigned i = 0; i < 5; ++i)
        append_inst(code, v_mov_b32(12, 11));
      append_inst(code, s_mov_b32(126, 128));
      if (nop)
        code.push_back(0x7e000000); // V_NOP issues even with empty EXEC.
      else
        append_inst(code, v_mov_b32(20, 21));
      append_inst(code, s_mov_b32(126, 129));
      append_inst(code, sopp(8, 0x0f9f | (wait << 12)));
      append_inst(code, ds_load_b32(42, 8));
      append_inst(code, s_endpgm());
      const auto report = analyze_waitcnts(TestCodeObject(code), ROCJITSU_CODE_ARCH_CDNA5);
      ASSERT_TRUE(report.supported && report.complete) << report.analysis_error;
      EXPECT_EQ(std::ranges::any_of(report.diagnostics,
                                    [](const auto &d) {
                                      return d.counter == WaitCounterKind::VaVdst &&
                                             d.reg.index == 42;
                                    }),
                wait == 1 && !nop)
          << diagnostic_summary(report);
    }
  }
}

TEST(WaitcheckTest, Gfx1250ExpertDelayWaitsForCoreValuBeyondWmmaCoexecution) {
  for (unsigned younger : {4u, 5u, 6u}) {
    for (unsigned tag : {0u, 1u, 2u, 5u}) {
      SCOPED_TRACE(testing::Message() << younger << '/' << tag);
      std::vector<uint32_t> code;
      append_inst(code, s_setreg_imm32_b32(hwreg(26, 0, 2), 2));
      const auto producer = code.size() * sizeof(uint32_t);
      code.insert(code.end(), {0xcc600000, 0x1c02e528}); // WMMA reads v[40:47].
      for (unsigned i = 0; i < younger; ++i)
        append_inst(code, v_mov_b32(12, 11));
      // TRANS32_DEP_1 selects this independent trans instruction. Its
      // completion gives no information about the younger core VALUs.
      append_inst(code, v_sqrt_f32(15, 16));
      append_inst(code, sopp(7, tag));
      append_inst(code, v_mov_b32(14, 13));
      const auto consumer = code.size() * sizeof(uint32_t);
      append_inst(code, ds_load_b32(42, 8));
      append_inst(code, s_endpgm());
      const auto image =
          waitcheck_test::make_gfx_code_object(code, EF_AMDGPU_MACH_AMDGCN_GFX1250, true);
      AmdGpuCodeObject object(image.data(), image.size());
      const auto report = analyze_waitcnts(object, ROCJITSU_CODE_ARCH_CDNA5);
      ASSERT_TRUE(report.supported && report.complete) << report.analysis_error;
      const bool completed = (tag == 1 || tag == 2) && younger >= 4 + tag;
      EXPECT_EQ(std::ranges::any_of(report.diagnostics,
                                    [&](const auto &d) {
                                      return d.counter == WaitCounterKind::VaVdst &&
                                             d.producer_section_offset == producer &&
                                             d.section_offset == consumer && d.reg.index == 42;
                                    }),
                !completed)
          << diagnostic_summary(report);
    }
  }
}

TEST(WaitcheckTest, Gfx1250ReappliedExecFilterCountsCoreIssuesOnTheLiveConsumerPath) {
  enum class Guard { Same, Different, ChangeMask, ChangeSaved, WidenExec, EnterAtRestore };
  for (Guard guard : {Guard::Same, Guard::Different, Guard::ChangeMask, Guard::ChangeSaved,
                      Guard::WidenExec, Guard::EnterAtRestore}) {
    for (unsigned count : {1u, 2u, 3u}) {
      SCOPED_TRACE(testing::Message() << static_cast<unsigned>(guard) << '/' << count);
      std::vector<uint32_t> code;
      append_inst(code, s_setreg_imm32_b32(hwreg(26, 0, 2), 2));
      const auto producer = code.size() * sizeof(uint32_t);
      append_inst(code, v_mov_b32(8, 3));
      append_inst(code, v_mov_b32(12, 11));
      const auto alternative = code.size();
      if (guard == Guard::EnterAtRestore)
        code.push_back(0xbfa10000); // s_cbranch_scc0 restore
      code.push_back(0xbe8a2009);   // s_and_saveexec_b32 s10, s9
      const auto first_branch = code.size();
      code.push_back(0xbfa50000); // s_cbranch_execz restore
      append_inst(code, v_mov_b32(14, 13));
      const auto restore = code.size();
      code[first_branch] |= restore - first_branch - 1;
      if (guard == Guard::EnterAtRestore)
        code[alternative] |= restore - alternative - 1;
      if (guard == Guard::ChangeSaved)
        code.push_back(0xbe8a00c1); // s_mov_b32 s10, -1
      code.push_back(0x8c7e0a7e);   // s_or_b32 exec_lo, exec_lo, s10
      if (guard == Guard::ChangeMask)
        code.push_back(0xbe8900c1); // s_mov_b32 s9, -1
      if (guard == Guard::WidenExec)
        code.push_back(0xbefe00c1); // s_mov_b32 exec_lo, -1
      code.push_back(guard == Guard::Different ? 0xbe8a200b : 0xbe8a2009);
      const auto second_branch = code.size();
      code.push_back(0xbfa50000); // s_cbranch_execz end
      append_inst(code, sopp(8, 0x0f9f | (count << 12)));
      const auto consumer = code.size() * sizeof(uint32_t);
      append_inst(code, global_load_b32(8, 2));
      code[second_branch] |= code.size() - second_branch - 1;
      append_inst(code, s_endpgm());
      const auto image =
          waitcheck_test::make_gfx_code_object(code, EF_AMDGPU_MACH_AMDGCN_GFX1250, true);
      AmdGpuCodeObject object(image.data(), image.size());
      WaitcheckOptions options;
      options.check_counter_parity = true;
      const auto report = analyze_waitcnts(object, ROCJITSU_CODE_ARCH_CDNA5, options);
      ASSERT_TRUE(report.supported && report.complete) << report.analysis_error;
      const unsigned required = guard == Guard::Same ? 2 : 1;
      const auto hazard = std::ranges::find_if(report.diagnostics, [&](const auto &d) {
        return d.counter == WaitCounterKind::VaVdst && d.section_offset == consumer &&
               d.producer_section_offset == producer && d.reg.index == 8;
      });
      EXPECT_EQ(hazard != report.diagnostics.end(), count > required) << diagnostic_summary(report);
      if (hazard != report.diagnostics.end())
        EXPECT_EQ(hazard->required_count, required);
      const auto &fields = report.counter_underaccounting_diagnostics;
      const auto stronger = std::ranges::find_if(fields, [&](const auto &d) {
        return d.counter == WaitCounterKind::VaVdst && d.wait_section_offset + 4 == consumer;
      });
      EXPECT_EQ(stronger != fields.end(), count < required);
      if (stronger != fields.end())
        EXPECT_EQ(stronger->required_count, required);
    }
  }
}

TEST(WaitcheckTest, ExpertWaitReusesEarlierWaitUnderRepeatedExecFilter) {
  enum class Prefix { None, Scalar, LoadWait, Branch };
  const auto check = [](uint32_t machine, rj_code_arch_t arch, Prefix prefix, bool same_mask,
                        unsigned earlier, unsigned later, bool joined = false) {
    SCOPED_TRACE(testing::Message()
                 << machine << '/' << static_cast<int>(prefix) << '/' << same_mask << '/' << earlier
                 << '/' << later << '/' << joined);
    std::vector<uint32_t> code;
    append_inst(code, s_setreg_imm32_b32(hwreg(26, 0, 2), 2));
    const auto producer = code.size() * sizeof(uint32_t);
    append_inst(code, v_mov_b32(8, 3));
    append_inst(code, v_mov_b32(12, 11));
    code.push_back(0xbe8a2009); // s_and_saveexec_b32 s10, s9
    code.push_back(0xbfa50001); // s_cbranch_execz restore
    const auto first_wait = code.size() * sizeof(uint32_t);
    append_inst(code, sopp(8, 0x0f9f | (earlier << 12)));
    code.push_back(0x8c7e0a7e); // s_or_b32 exec_lo, exec_lo, s10
    code.push_back(same_mask ? 0xbe8a2009 : 0xbe8a200b);
    const auto branch = code.size();
    code.push_back(0xbfa50000); // s_cbranch_execz end
    if (prefix == Prefix::Scalar)
      code.push_back(0xbe940081); // s_mov_b32 s20, 1
    if (prefix == Prefix::Branch)
      code.push_back(0xbfa00000); // s_branch to the next block
    const auto wait = code.size() * sizeof(uint32_t);
    if (prefix == Prefix::LoadWait)
      append_inst(code, sopp(64, 0)); // s_wait_loadcnt 0
    append_inst(code, sopp(8, 0x0f9f | (later << 12)));
    if (joined) {
      append_inst(code, sopp(0, 0)); // s_nop 0
      code[branch] |= code.size() - branch - 1;
      append_inst(code, s_mov_b32(126, 129)); // s_mov_b32 exec_lo, 1
    }
    const auto consumer = code.size() * sizeof(uint32_t);
    append_inst(code, global_load_b32(8, 2));
    if (!joined)
      code[branch] |= code.size() - branch - 1;
    append_inst(code, s_endpgm());
    const auto image = waitcheck_test::make_gfx_code_object(code, machine, true);
    AmdGpuCodeObject object(image.data(), image.size());
    WaitcheckOptions options;
    options.check_counter_parity = true;
    const auto report = analyze_waitcnts(object, arch, options);
    ASSERT_TRUE(report.supported && report.complete) << report.analysis_error;
    const bool needs_wait = !same_mask || earlier == 15;
    const auto &fields = report.counter_underaccounting_diagnostics;
    const auto attributed = std::ranges::find_if(fields, [&](const auto &field) {
      return field.counter == WaitCounterKind::VaVdst && field.wait_section_offset == wait &&
             field.has_required_dependency;
    });
    EXPECT_EQ(attributed != fields.end(), later == 0 && needs_wait);
    if (attributed != fields.end()) {
      EXPECT_EQ(attributed->producer_section_offset, producer);
      EXPECT_EQ(attributed->consumer_section_offset, consumer);
      EXPECT_EQ(attributed->required_count, 1u);
    }
    if (joined) {
      EXPECT_FALSE(std::ranges::any_of(fields, [&](const auto &field) {
        return field.counter == WaitCounterKind::VaVdst &&
               field.wait_section_offset == first_wait && field.has_required_dependency;
      }));
    }
    const auto &diagnostics = report.diagnostics;
    const auto hazard = std::ranges::find_if(diagnostics, [&](const auto &diagnostic) {
      return diagnostic.counter == WaitCounterKind::VaVdst && diagnostic.section_offset == consumer;
    });
    EXPECT_EQ(hazard != diagnostics.end(), joined || (later == 15 && needs_wait));
    if (hazard != diagnostics.end()) {
      EXPECT_EQ(hazard->producer_section_offset, producer);
      EXPECT_EQ(hazard->required_count, 1u);
    }
  };
  for (auto prefix : {Prefix::None, Prefix::Scalar, Prefix::LoadWait, Prefix::Branch}) {
    for (bool same_mask : {false, true}) {
      for (unsigned earlier : {1u, 15u}) {
        for (unsigned later : {0u, 15u}) {
          check(EF_AMDGPU_MACH_AMDGCN_GFX1201, ROCJITSU_CODE_ARCH_RDNA4, prefix, same_mask, earlier,
                later);
          check(EF_AMDGPU_MACH_AMDGCN_GFX1250, ROCJITSU_CODE_ARCH_CDNA5, prefix, same_mask, earlier,
                later);
        }
      }
    }
  }
  check(EF_AMDGPU_MACH_AMDGCN_GFX1201, ROCJITSU_CODE_ARCH_RDNA4, Prefix::None, true, 1, 0, true);
  check(EF_AMDGPU_MACH_AMDGCN_GFX1250, ROCJITSU_CODE_ARCH_CDNA5, Prefix::None, true, 1, 0, true);
}

TEST(WaitcheckTest, Gfx1250KernelPathUsesWmmaProgressAtEachPhysicalWait) {
  // Reduced from the AOTriton dQ bf16 schedule. VA(1) is adjacent to the
  // WMMA. It cannot use any of the eleven independent core VALUs that follow.
  for (unsigned prior : {0u, 1u}) {
    for (unsigned final : {0u, 6u, 7u, 9u, 15u}) {
      for (unsigned consumer_mask : {0u, 1u}) {
        SCOPED_TRACE(testing::Message() << prior << '/' << final << '/' << consumer_mask);
        std::vector<uint32_t> code;
        append_inst(code, s_setreg_imm32_b32(hwreg(26, 0, 2), 2));
        const auto producer = code.size() * sizeof(uint32_t);
        code.insert(code.end(), {0xcc600000, 0x1c02e528}); // WMMA reads v[40:47].
        append_inst(code, sopp(8, 0x0f9f | (prior << 12)));
        for (unsigned i = 0; i < 11; ++i)
          append_inst(code, v_mov_b32(12, 11));
        append_inst(code, sopp(8, 0x0f9f | (final << 12)));
        append_inst(code, s_mov_b32(126, 128 + consumer_mask));
        const auto consumer = code.size() * sizeof(uint32_t);
        append_inst(code, ds_load_b32(42, 8));
        append_inst(code, s_endpgm());
        const auto image =
            waitcheck_test::make_gfx_code_object(code, EF_AMDGPU_MACH_AMDGCN_GFX1250, true);
        AmdGpuCodeObject object(image.data(), image.size());
        const auto report = analyze_waitcnts(object, ROCJITSU_CODE_ARCH_CDNA5);
        ASSERT_TRUE(report.supported && report.complete) << report.analysis_error;
        EXPECT_EQ(std::ranges::any_of(report.diagnostics,
                                      [&](const auto &d) {
                                        return d.counter == WaitCounterKind::VaVdst &&
                                               d.producer_section_offset == producer &&
                                               d.section_offset == consumer && d.reg.index == 42;
                                      }),
                  prior != 0 && final > 6 && consumer_mask != 0)
            << diagnostic_summary(report);
      }
    }
  }
}

TEST(WaitcheckTest, ExpertKernelPathRetainsOnlyOrderedSourcesInTheActiveMode) {
  for (const auto [arch, machine] :
       {std::pair{ROCJITSU_CODE_ARCH_RDNA4, EF_AMDGPU_MACH_AMDGCN_GFX1201},
        std::pair{ROCJITSU_CODE_ARCH_CDNA5, EF_AMDGPU_MACH_AMDGCN_GFX1250}}) {
    for (const bool same_fifo : {false, true}) {
      for (const bool leave_expert : {false, true}) {
        for (unsigned count : {0u, 1u, 2u}) {
          SCOPED_TRACE(testing::Message()
                       << arch << '/' << same_fifo << '/' << leave_expert << '/' << count);
          std::vector<uint32_t> code;
          append_inst(code, s_setreg_imm32_b32(hwreg(26, 0, 2), 2));
          const auto producer = code.size() * sizeof(uint32_t);
          append_inst(code, ds_load_b32(2, 6));
          append_inst(code, s_mov_b32(126, 5)); // Unknown younger EXEC.
          if (same_fifo)
            append_inst(code, ds_load_b32(3, 7));
          else
            append_inst(code, global_load_b32(3, 20));
          if (leave_expert) {
            append_inst(code, s_setreg_imm32_b32(hwreg(26, 0, 2), 0));
            append_inst(code, s_setreg_imm32_b32(hwreg(26, 0, 2), 2));
          }
          append_inst(code, s_wait_alu_vm_vsrc(count));
          append_inst(code, s_mov_b32(126, 129)); // A real active destination lane.
          const auto consumer = code.size() * sizeof(uint32_t);
          append_inst(code, ds_load_b32(6, 9));
          append_inst(code, s_endpgm());
          const auto image = waitcheck_test::make_gfx_code_object(code, machine, true);
          AmdGpuCodeObject object(image.data(), image.size());
          const auto report = analyze_waitcnts(object, arch);
          ASSERT_TRUE(report.supported && report.complete) << report.analysis_error;
          EXPECT_EQ(std::ranges::any_of(report.diagnostics,
                                        [&](const auto &d) {
                                          return d.counter == WaitCounterKind::VmVsrc &&
                                                 d.producer_section_offset == producer &&
                                                 d.section_offset == consumer && d.reg.index == 6;
                                        }),
                    !leave_expert && count > static_cast<unsigned>(same_fifo))
              << diagnostic_summary(report);
        }
      }
    }
  }
}

TEST(WaitcheckTest, Gfx1250EffectiveExecQualifiesExpertVmSourceCredit) {
  for (const bool scratch : {false, true}) {
    for (const unsigned bank : {0u, 64u}) {
      for (const unsigned allocated : {48u, 320u}) {
        for (const unsigned wait : {0u, 1u, 2u}) {
          std::vector<uint32_t> code;
          append_inst(code, s_setreg_imm32_b32(hwreg(26, 0, 2), 2));
          append_inst(code, s_mov_b32(126, 129)); // EXEC_LO=1 on every path.
          const auto producer = code.size() * sizeof(uint32_t);
          append_inst(code, global_store_b32(11, 0, 124));
          append_inst(code, s_wait_xcnt(0));
          append_inst(code, s_set_vgpr_msb(bank));
          if (scratch)
            append_inst(code, scratch_load_b128_off(24));
          else
            append_inst(code, global_load_b32(24, 20, 124));
          append_inst(code, s_set_vgpr_msb(0));
          const auto field = code.size() * sizeof(uint32_t);
          append_inst(code, s_wait_xcnt(0));
          append_inst(code, s_wait_alu_vm_vsrc(wait));
          const auto consumer = code.size() * sizeof(uint32_t);
          append_inst(code, v_mov_b32(11, 13));
          append_inst(code, s_endpgm());
          auto image =
              waitcheck_test::make_gfx_code_object(code, EF_AMDGPU_MACH_AMDGCN_GFX1250, true);
          const uint32_t rsrc1 = allocated / 16 - 1;
          std::memcpy(image.data() + 0x100 + code.size() * sizeof(uint32_t) + 48, &rsrc1,
                      sizeof(rsrc1));
          AmdGpuCodeObject object(image.data(), image.size());
          for (bool parity : {false, true}) {
            SCOPED_TRACE(testing::Message() << scratch << '/' << bank << '/' << allocated << '/'
                                            << wait << '/' << parity);
            WaitcheckOptions options;
            options.check_counter_parity = parity;
            const auto report = analyze_waitcnts(object, ROCJITSU_CODE_ARCH_CDNA5, options);
            ASSERT_TRUE(report.complete);
            const unsigned bound = bank == 0 || allocated == 320;
            const auto it = std::ranges::find_if(report.diagnostics, [&](const auto &d) {
              return d.counter == WaitCounterKind::VmVsrc && d.section_offset == consumer &&
                     d.producer_section_offset == producer && d.reg.index == 11;
            });
            EXPECT_EQ(it != report.diagnostics.end(), wait > bound) << diagnostic_summary(report);
            if (it != report.diagnostics.end())
              EXPECT_EQ(it->required_count, bound);
            const bool recommendation =
                std::ranges::any_of(report.counter_underaccounting_diagnostics, [&](const auto &d) {
                  return d.counter == WaitCounterKind::VmVsrc && d.wait_section_offset == field &&
                         d.has_required_dependency && d.required_count == bound;
                });
            EXPECT_EQ(recommendation, parity && wait < bound);
          }
        }
      }
    }
  }
}

TEST(WaitcheckTest, ExpertVmSourceNestedSavesRetainEachIntactMask) {
  for (const auto [arch, machine] :
       {std::pair{ROCJITSU_CODE_ARCH_RDNA4, EF_AMDGPU_MACH_AMDGCN_GFX1201},
        std::pair{ROCJITSU_CODE_ARCH_CDNA5, EF_AMDGPU_MACH_AMDGCN_GFX1250}}) {
    for (bool restore_outer : {false, true}) {
      for (unsigned invalidated : {0u, 4u, 5u}) {
        for (unsigned wait : {0u, 1u, 2u}) {
          std::vector<uint32_t> code;
          append_inst(code, s_setreg_imm32_b32(hwreg(26, 0, 2), 2));
          append_inst(code, s_mov_b32(126, 129));
          const auto producer = code.size() * sizeof(uint32_t);
          append_inst(code, global_store_b32(11, 0, 124));
          code.push_back(0xbe842006); // s_and_saveexec_b32 s4,s6
          code.push_back(0xbfa50001); // s_cbranch_execz over one NOP
          append_inst(code, sopp(0));
          code.push_back(0x8c7e047e); // s_or_b32 exec_lo,exec_lo,s4
          code.push_back(0xbe852006); // s_and_saveexec_b32 s5,s6
          if (invalidated)
            append_inst(code, s_mov_b32(invalidated, 128));
          append_inst(code, s_mov_b32(126, 128));
          code.push_back(restore_outer ? 0x8c7e047e : 0x8c7e057e);
          append_inst(code, global_load_b32(4, 8, 124));
          append_inst(code, s_wait_alu_vm_vsrc(wait));
          append_inst(code, s_mov_b32(126, 129));
          const auto consumer = code.size() * sizeof(uint32_t);
          append_inst(code, v_mov_b32(11, 13));
          append_inst(code, s_endpgm());
          const auto image = waitcheck_test::make_gfx_code_object(code, machine, true);
          AmdGpuCodeObject object(image.data(), image.size());
          const auto report = analyze_waitcnts(object, arch);
          ASSERT_TRUE(report.complete);
          const auto bound = unsigned(invalidated != (restore_outer ? 4u : 5u));
          const auto it = std::ranges::find_if(report.diagnostics, [&](const auto &d) {
            return d.counter == WaitCounterKind::VmVsrc && d.section_offset == consumer &&
                   d.producer_section_offset == producer && d.reg.index == 11;
          });
          EXPECT_EQ(it != report.diagnostics.end(), wait > bound) << diagnostic_summary(report);
          if (it != report.diagnostics.end())
            EXPECT_EQ(it->required_count, bound);
        }
      }
    }
  }
}

TEST(WaitcheckTest, PartialStoreWaitRetiresOnlyOlderDedicatedStoreSources) {
  enum class Younger { Store, MaskedStore, OptionalStore, Load, Writeback, Invalidate };
  for (auto arch : {ROCJITSU_CODE_ARCH_RDNA4, ROCJITSU_CODE_ARCH_CDNA5}) {
    for (const auto younger : {Younger::Store, Younger::MaskedStore, Younger::OptionalStore,
                               Younger::Load, Younger::Writeback, Younger::Invalidate}) {
      for (const unsigned reg : {5u, 11u}) {
        std::vector<uint32_t> code;
        append_inst(code, s_setreg_imm32_b32(hwreg(26, 0, 2), 2));
        code.push_back(0xbefe0081u); // s_mov_b32 exec_lo, 1
        append_inst(code, global_store_b32(5, 16, 124));
        if (arch == ROCJITSU_CODE_ARCH_CDNA5)
          append_inst(code, s_wait_xcnt_0());
        if (younger == Younger::OptionalStore)
          code.push_back(0xbfa10003u); // s_cbranch_scc0 over the store.
        if (younger == Younger::MaskedStore)
          code.push_back(0xbefe0080u); // Empty EXEC, older STORECNT remains live.
        if (younger == Younger::Writeback)
          append_inst(code, global_wb());
        else if (younger == Younger::Invalidate)
          append_inst(code, global_inv());
        else if (younger == Younger::Load)
          append_inst(code, global_load_b32(6, 11, 124));
        else
          append_inst(code, global_store_b32(11, 18, 124));
        if (younger == Younger::MaskedStore)
          code.push_back(0xbefe0081u); // Reactivate the consumer.
        if (arch == ROCJITSU_CODE_ARCH_CDNA5) {
          append_inst(code, s_wait_xcnt_0());
          append_inst(code, sopp(0, 0)); // Keep XCNT and VM_VSRC in separate groups.
        }
        const auto pcs = code.size() * sizeof(uint32_t);
        append_inst(code, s_wait_alu_vm_vsrc(7));
        append_inst(code, sopp(0, 0));
        append_inst(code, sopp(65, 1)); // STORECNT(1)
        append_inst(code, sopp(0, 0));
        const auto consumer = code.size() * sizeof(uint32_t);
        append_inst(code, v_mov_b32(reg, 25));
        append_inst(code, s_endpgm());
        // The existing primary policy also orders global writeback on STORECNT.
        const bool store_pair = younger == Younger::Store || younger == Younger::MaskedStore;
        const bool has_source = reg == 5 || younger == Younger::Store ||
                                younger == Younger::OptionalStore || younger == Younger::Load;
        const bool required =
            has_source && !((store_pair || younger == Younger::Writeback) && reg == 5);
        for (const unsigned bound : {0u, 7u}) {
          code[pcs / sizeof(uint32_t)] = std::bit_cast<uint32_t>(s_wait_alu_vm_vsrc(bound));
          const auto machine = arch == ROCJITSU_CODE_ARCH_CDNA5 ? EF_AMDGPU_MACH_AMDGCN_GFX1250
                                                                : EF_AMDGPU_MACH_AMDGCN_GFX1201;
          const auto image = waitcheck_test::make_gfx_code_object(code, machine, true);
          AmdGpuCodeObject object(image.data(), image.size());
          WaitcheckOptions options;
          options.check_counter_parity = true;
          const auto report = analyze_waitcnts(object, arch, options);
          ASSERT_TRUE(report.complete);
          const bool warned = std::ranges::any_of(report.diagnostics, [&](const auto &diagnostic) {
            return diagnostic.counter == WaitCounterKind::VmVsrc &&
                   diagnostic.section_offset == consumer && diagnostic.reg.index == reg;
          });
          EXPECT_EQ(warned, bound == 7 && required) << "younger " << static_cast<int>(younger)
                                                    << " reg " << reg << diagnostic_summary(report);
          if (bound == 0 && store_pair && reg == 5) {
            const auto it = std::ranges::find_if(
                report.counter_underaccounting_diagnostics, [&](const auto &row) {
                  return row.wait_section_offset == pcs && row.counter == WaitCounterKind::VmVsrc;
                });
            EXPECT_NE(it, report.counter_underaccounting_diagnostics.end())
                << "arch " << arch << " younger " << static_cast<int>(younger);
            if (it == report.counter_underaccounting_diagnostics.end())
              continue;
            EXPECT_FALSE(it->has_required_dependency);
            EXPECT_EQ(it->required_count, 7u);
          }
        }
      }
    }
  }
}

TEST(WaitcheckTest, Gfx1201VmSourceAuditKeepsIndependentPrimaryWaits) {
  for (const unsigned reg : {5u, 8u}) {
    std::vector<uint32_t> code;
    append_inst(code, s_setreg_imm32_b32(hwreg(26, 0, 2), 2));
    append_inst(code, global_store_b32(5, 0, 124));
    append_inst(code, global_load_b32(4, 8, 124));
    const auto field = code.size() * sizeof(uint32_t);
    append_inst(code, sopp(65, 0)); // STORECNT(0) releases only the store.
    append_inst(code, s_wait_alu_vm_vsrc(0));
    append_inst(code, v_mov_b32(reg, 13));
    append_inst(code, s_endpgm());
    const auto image =
        waitcheck_test::make_gfx_code_object(code, EF_AMDGPU_MACH_AMDGCN_GFX1201, true);
    AmdGpuCodeObject object(image.data(), image.size());
    WaitcheckOptions options;
    options.check_counter_parity = true;
    const auto report = analyze_waitcnts(object, ROCJITSU_CODE_ARCH_RDNA4, options);
    ASSERT_TRUE(report.complete);
    const auto it =
        std::ranges::find_if(report.counter_underaccounting_diagnostics, [&](const auto &d) {
          return d.counter == WaitCounterKind::VmVsrc && d.wait_section_offset == field;
        });
    ASSERT_EQ(it != report.counter_underaccounting_diagnostics.end(), reg == 5);
    if (it != report.counter_underaccounting_diagnostics.end()) {
      EXPECT_FALSE(it->has_required_dependency);
      EXPECT_EQ(it->required_count, 7u);
    }
  }
}

TEST(WaitcheckTest, Gfx1250ExpertSourceChecksAllocatedBankInBothAnalysisModes) {
  enum class Bank { Low, High, Either };
  for (const Bank bank : {Bank::Low, Bank::High, Bank::Either}) {
    std::vector<uint32_t> code;
    append_inst(code, s_setreg_imm32_b32(hwreg(26, 0, 2), 2));
    const auto producer = 4 * code.size();
    append_inst(code, global_store_b32(11, 0, 124));
    code.push_back(0x8c7e0f7eu); // s_or_b32 exec_lo,exec_lo,s15
    code.push_back(0xbe842006u); // s_and_saveexec_b32 s4,s6
    code.push_back(0xbfa50001u); // s_cbranch_execz over the NOP.
    append_inst(code, sopp(0, 0));
    code.push_back(0x8c7e047eu); // s_or_b32 exec_lo,exec_lo,s4
    if (bank == Bank::Either) {
      // rsrc1 allocates only the low bank. The scalar predicate admits both choices.
      code.push_back(0xbf060709u); // s_cmp_eq_u32 s9,s7
      code.push_back(0xbfa10001u); // s_cbranch_scc0 skips the high-bank switch.
    }
    append_inst(code, s_set_vgpr_msb(bank == Bank::Low ? 0 : 64));
    append_inst(code, scratch_load_b128_off(4));
    append_inst(code, s_set_vgpr_msb(0));
    append_inst(code, s_wait_alu_vm_vsrc(1));
    append_inst(code, s_mov_b32(126, 129));
    const auto consumer = 4 * code.size();
    append_inst(code, scratch_load_b128_off(8));
    append_inst(code, s_endpgm());
    const auto image =
        waitcheck_test::make_gfx_code_object(code, EF_AMDGPU_MACH_AMDGCN_GFX1250, true);
    AmdGpuCodeObject object(image.data(), image.size());
    for (const bool parity : {false, true}) {
      WaitcheckOptions options;
      options.check_counter_parity = parity;
      const auto report = analyze_waitcnts(object, ROCJITSU_CODE_ARCH_CDNA5, options);
      ASSERT_TRUE(report.supported && report.complete) << report.analysis_error;
      EXPECT_EQ(std::ranges::any_of(report.diagnostics,
                                    [&](const auto &d) {
                                      return d.counter == WaitCounterKind::VmVsrc &&
                                             d.reg.index == 11 &&
                                             d.producer_section_offset == producer &&
                                             d.section_offset == consumer && d.required_count == 0;
                                    }),
                bank != Bank::Low)
          << diagnostic_summary(report);
    }
  }
}

TEST(WaitcheckTest, ExpertVmSourceMaskRecoveredOnlyFromItsOwnIntactSavedExec) {
  enum class Save {
    Intact,
    Overwrite,
    Bypass,
    OtherRegister,
    SpecialDestination,
    PendingSmem,
    DrainedSmem,
    PartialSmem,
    MayDrainSmem
  };
  for (const auto [arch, machine] :
       {std::pair{ROCJITSU_CODE_ARCH_RDNA4, EF_AMDGPU_MACH_AMDGCN_GFX1201},
        std::pair{ROCJITSU_CODE_ARCH_CDNA5, EF_AMDGPU_MACH_AMDGCN_GFX1250}}) {
    for (Save test : {Save::Intact, Save::Overwrite, Save::Bypass, Save::OtherRegister,
                      Save::SpecialDestination, Save::PendingSmem, Save::DrainedSmem,
                      Save::PartialSmem, Save::MayDrainSmem}) {
      for (uint32_t wait : {0u, 1u, 2u}) {
        SCOPED_TRACE(testing::Message() << machine << '/' << static_cast<int>(test) << '/' << wait);
        const unsigned saved = test == Save::SpecialDestination ? 124 : 4;
        std::vector<uint32_t> code;
        append_inst(code, s_setreg_imm32_b32(hwreg(26, 0, 2), 2));
        if (test == Save::PendingSmem || test == Save::DrainedSmem || test == Save::PartialSmem ||
            test == Save::MayDrainSmem) {
          append_inst(code, s_load_b32(saved, 8));
          if (test == Save::MayDrainSmem)
            code.push_back(0xbfa10001u); // One predecessor bypasses KMCNT(0).
          if (test != Save::PendingSmem)
            append_inst(code, sopp(71, test == Save::PartialSmem ? 1 : 0));
        }
        const auto producer = 4 * code.size();
        append_inst(code, global_store_b32(11, 0, 124));
        if (test == Save::Bypass) {
          code.push_back(0xbfa10002u); // SCC0 skips the real save on one incoming path.
          code.push_back(0xbe802000u | (saved << 16)); // s_and_saveexec_b32 saved,s0
          code.push_back(0xbfa00001u); // s_branch over the alternative EXEC definition.
          append_inst(code, s_mov_b32(126, 6));
        } else {
          code.push_back(0xbe802000u | (saved << 16));
          code.push_back(0xbfa50001u); // s_cbranch_execz skips one instruction.
          if (test == Save::Overwrite)
            append_inst(code, s_mov_b32(saved, 128));
          else
            append_inst(code, sopp(0, 0));
        }
        code.push_back(0x8c7e007eu | ((test == Save::OtherRegister ? 5 : saved) << 8));
        append_inst(code, scratch_load_b128_off(0));
        append_inst(code, s_wait_alu_vm_vsrc(wait));
        append_inst(code, s_mov_b32(126, 129));
        const auto consumer = 4 * code.size();
        append_inst(code, scratch_load_b128_off(8));
        append_inst(code, s_endpgm());
        const auto image = waitcheck_test::make_gfx_code_object(code, machine, true);
        AmdGpuCodeObject object(image.data(), image.size());
        const auto report = analyze_waitcnts(object, arch);
        ASSERT_TRUE(report.supported && report.complete) << report.analysis_error;
        const bool hazard = std::ranges::any_of(report.diagnostics, [&](const auto &d) {
          return d.counter == WaitCounterKind::VmVsrc && d.producer_section_offset == producer &&
                 d.section_offset == consumer && d.reg.index == 11;
        });
        EXPECT_EQ(hazard,
                  wait > static_cast<unsigned>(test == Save::Intact || test == Save::DrainedSmem))
            << diagnostic_summary(report);
      }
    }
  }
}

TEST(WaitcheckTest, ExpertVmSourcesSurviveMonotoneExecMaskUpdates) {
  for (const auto [arch, machine] :
       {std::pair{ROCJITSU_CODE_ARCH_RDNA4, EF_AMDGPU_MACH_AMDGCN_GFX1201},
        std::pair{ROCJITSU_CODE_ARCH_CDNA5, EF_AMDGPU_MACH_AMDGCN_GFX1250}}) {
    for (const bool wave32 : {false, true}) {
      if (arch == ROCJITSU_CODE_ARCH_CDNA5 && !wave32)
        continue;
      for (const auto [mask, preserves] : {
               std::pair{0x8c7e067eu, true},  // s_or_b32 exec_lo, exec_lo, s6
               std::pair{0x8c7e7e06u, true},  // s_or_b32 exec_lo, s6, exec_lo
               std::pair{0x8cfe047eu, true},  // s_or_b64 exec, exec, s[4:5]
               std::pair{0x8c7f7f06u, true},  // s_or_b32 exec_hi, s6, exec_hi
               std::pair{0x8c7e067fu, false}, // s_or_b32 exec_lo, exec_hi, s6
               std::pair{0x8c7e0706u, false}, // s_or_b32 exec_lo, s6, s7
               std::pair{0xbefe0006u, false}, // s_mov_b32 exec_lo, s6
           }) {
        for (const auto [dst, valid] : {std::pair{0u, true}, std::pair{96u, false}}) {
          for (unsigned wait : {0u, 1u, 2u}) {
            SCOPED_TRACE(testing::Message()
                         << machine << '/' << wave32 << '/' << mask << '/' << dst << '/' << wait);
            std::vector<uint32_t> code;
            append_inst(code, s_setreg_imm32_b32(hwreg(26, 0, 2), 2));
            const auto producer = code.size() * sizeof(uint32_t);
            append_inst(code, global_store_b32(11, 0, 124));
            append_inst(code, mask);
            append_inst(code, scratch_load_b128_off(dst));
            append_inst(code, s_wait_alu_vm_vsrc(wait));
            append_inst(code, s_mov_b32(126, 129));
            const auto consumer = code.size() * sizeof(uint32_t);
            append_inst(code, scratch_load_b128_off(8));
            append_inst(code, s_endpgm());
            const auto image = waitcheck_test::make_gfx_code_object(code, machine, wave32);
            AmdGpuCodeObject object(image.data(), image.size());
            const auto report = analyze_waitcnts(object, arch);
            ASSERT_TRUE(report.supported && report.complete) << report.analysis_error;
            const auto hazard = std::ranges::find_if(report.diagnostics, [&](const auto &d) {
              return d.counter == WaitCounterKind::VmVsrc && d.section_offset == consumer &&
                     d.producer_section_offset == producer && d.reg.index == 11;
            });
            const unsigned required = preserves && valid ? 1 : 0;
            EXPECT_EQ(hazard != report.diagnostics.end(), wait > required)
                << diagnostic_summary(report);
            if (hazard != report.diagnostics.end())
              EXPECT_EQ(hazard->required_count, required);
          }
        }
      }
    }
  }
}

TEST(WaitcheckTest, ExpertEmptyExecIssueCanFollowAYoungerActiveSourceOnItsPrimary) {
  enum class Store { Absent, Older, Younger, Drained, Bypassed, Empty };
  for (const auto [arch, machine] :
       {std::pair{ROCJITSU_CODE_ARCH_RDNA4, EF_AMDGPU_MACH_AMDGCN_GFX1201},
        std::pair{ROCJITSU_CODE_ARCH_CDNA5, EF_AMDGPU_MACH_AMDGCN_GFX1250}}) {
    for (const auto store : {Store::Absent, Store::Older, Store::Younger, Store::Drained,
                             Store::Bypassed, Store::Empty}) {
      for (unsigned count : {0u, 1u, 2u, 3u}) {
        SCOPED_TRACE(testing::Message() << machine << '/' << int(store) << '/' << count);
        std::vector<uint32_t> code;
        append_inst(code, s_setreg_imm32_b32(hwreg(26, 0, 2), 2));
        append_inst(code, s_mov_b32(126, 129));
        if (store == Store::Older)
          append_inst(code, global_store_b32(5, 16, 124));
        const auto producer = code.size() * sizeof(uint32_t);
        append_inst(code, global_load_b32(2, 8));
        if (store == Store::Bypassed)
          code.push_back(0xbfa10003u); // s_cbranch_scc0 skips store.
        if (store == Store::Empty)
          append_inst(code, s_mov_b32(126, 128));
        if (store != Store::Older && store != Store::Absent)
          append_inst(code, global_store_b32(5, 16, 124));
        if (store == Store::Drained)
          code.push_back(0xbfc10000u); // s_wait_storecnt 0.
        append_inst(code, s_mov_b32(126, 128));
        append_inst(code, global_store_b32(7, 20, 124));
        if (arch == ROCJITSU_CODE_ARCH_CDNA5)
          append_inst(code, s_wait_xcnt(0));
        append_inst(code, s_wait_alu_vm_vsrc(count));
        append_inst(code, s_mov_b32(126, 129));
        const auto consumer = code.size() * sizeof(uint32_t);
        append_inst(code, v_mov_b32(8, 13));
        append_inst(code, s_endpgm());
        const auto image = waitcheck_test::make_gfx_code_object(code, machine, true);
        AmdGpuCodeObject object(image.data(), image.size());
        const auto report = analyze_waitcnts(object, arch);
        ASSERT_TRUE(report.supported && report.complete) << report.analysis_error;
        const auto hazard = std::ranges::find_if(report.diagnostics, [&](const auto &d) {
          return d.counter == WaitCounterKind::VmVsrc && d.section_offset == consumer &&
                 d.producer_section_offset == producer && d.reg.index == 8;
        });
        const unsigned required = store == Store::Younger ? 2 : store == Store::Drained ? 1 : 0;
        EXPECT_EQ(hazard != report.diagnostics.end(), count > required)
            << diagnostic_summary(report);
        if (hazard != report.diagnostics.end())
          EXPECT_EQ(hazard->required_count, required);
      }
    }
  }
}

TEST(WaitcheckTest, ExpertEmptyExecMemoryIssuesFollowTheLivePrimaryCounter) {
  enum class Younger { Load, Store, Ds, FlatStore };
  for (const auto [arch, machine] :
       {std::pair{ROCJITSU_CODE_ARCH_RDNA4, EF_AMDGPU_MACH_AMDGCN_GFX1201},
        std::pair{ROCJITSU_CODE_ARCH_CDNA5, EF_AMDGPU_MACH_AMDGCN_GFX1250}}) {
    for (const auto younger : {Younger::Load, Younger::Store, Younger::Ds, Younger::FlatStore}) {
      for (const bool drained : {false, true}) {
        for (unsigned count : {0u, 1u, 2u}) {
          SCOPED_TRACE(testing::Message()
                       << arch << '/' << int(younger) << '/' << drained << '/' << count);
          std::vector<uint32_t> code;
          append_inst(code, s_setreg_imm32_b32(hwreg(26, 0, 2), 2));
          append_inst(code, s_mov_b32(126, 129)); // EXEC_LO = 1.
          const auto producer = code.size() * sizeof(uint32_t);
          if (younger == Younger::FlatStore)
            append_inst(code, flat_load_b32(2));
          else
            append_inst(code, global_load_b32(2, 8));
          if (drained)
            append_inst(code, sopp(72, 0));       // LOADCNT and DSCNT.
          append_inst(code, s_mov_b32(126, 128)); // EXEC_LO = 0 in wave32.
          switch (younger) {
          case Younger::Load:
            append_inst(code, global_load_b32(3, 20));
            break;
          case Younger::Store:
            append_inst(code, global_store_b32(3, 20));
            break;
          case Younger::Ds:
            append_inst(code, ds_load_b32(3, 20));
            break;
          case Younger::FlatStore: {
            auto store = flat_store_b32(3);
            store.vaddr = 20;
            append_inst(code, store);
            break;
          }
          }
          append_inst(code, s_wait_alu_vm_vsrc(count));
          append_inst(code, s_mov_b32(126, 129));
          const auto consumer = code.size() * sizeof(uint32_t);
          append_inst(code, v_mov_b32(8, 13));
          append_inst(code, v_mov_b32(20, 13)); // Empty issue owns no vector sources.
          append_inst(code, s_endpgm());
          const auto image = waitcheck_test::make_gfx_code_object(code, machine, true);
          AmdGpuCodeObject object(image.data(), image.size());
          WaitcheckOptions options;
          options.check_counter_parity = true;
          const auto report = analyze_waitcnts(object, arch, options);
          ASSERT_TRUE(report.supported && report.complete) << report.analysis_error;
          const bool same = younger == Younger::Load || younger == Younger::FlatStore;
          const auto source = std::ranges::find_if(report.diagnostics, [&](const auto &d) {
            return d.counter == WaitCounterKind::VmVsrc && d.producer_section_offset == producer &&
                   d.section_offset == consumer && d.reg.index == 8;
          });
          ASSERT_EQ(source != report.diagnostics.end(), !drained && count > unsigned(same))
              << diagnostic_summary(report);
          if (source != report.diagnostics.end())
            EXPECT_EQ(source->required_count, unsigned(same));
          EXPECT_FALSE(std::ranges::any_of(report.diagnostics, [](const auto &d) {
            return d.counter == WaitCounterKind::VmVsrc && d.reg.index == 20;
          })) << diagnostic_summary(report);
        }
      }
    }
  }
}

TEST(WaitcheckTest, ExpertVmSourceWaitCountsYoungerRequestsInTheSameFifo) {
  for (auto arch : {ROCJITSU_CODE_ARCH_RDNA4, ROCJITSU_CODE_ARCH_CDNA5}) {
    for (bool lds : {false, true}) {
      for (bool change_mask : {false, true}) {
        for (unsigned count : {0u, 1u, 2u, 7u}) {
          SCOPED_TRACE(testing::Message()
                       << arch << "/" << lds << "/" << change_mask << "/" << count);
          std::vector<uint32_t> code;
          append_inst(code, s_setreg_imm32_b32(hwreg(26, 0, 2), 2));
          append_inst(code, global_load_b32(22, 16)); // Older, unrelated source class.
          append_inst(code, ds_load_b32(2, 6));
          if (change_mask)
            append_inst(code, s_mov_b32(126, 5)); // Unknown new EXEC.
          if (lds)
            append_inst(code, ds_load_b32(3, 7));
          else
            append_inst(code, global_load_b32(3, 20));
          append_inst(code, sopp(8, 0xff83 | (count << 2)));
          append_inst(code, ds_load_b32(6, 7)); // Writes the first LDS address.
          append_inst(code, s_endpgm());
          const auto report = analyze_waitcnts(TestCodeObject(code), arch);
          ASSERT_TRUE(report.supported) << report.analysis_error;
          const unsigned required = lds ? 1 : 0;
          auto it = std::ranges::find_if(report.diagnostics, [](const auto &d) {
            return d.counter == WaitCounterKind::VmVsrc && d.reg.index == 6;
          });
          ASSERT_EQ(it != report.diagnostics.end(), count > required) << diagnostic_summary(report);
          if (it != report.diagnostics.end())
            EXPECT_EQ(it->required_count, required);
        }
      }
    }
  }
}

TEST(WaitcheckTest, Gfx1250CodeObjectTracksVmVsrcInExpertMode) {
  constexpr uint32_t kHwRegWaveSchedMode = 26;
  std::vector<uint32_t> program;
  append_inst(program, s_setreg_imm32_b32(hwreg(kHwRegWaveSchedMode, 0, 2), 2));
  append_inst(program, global_load_b32(0));
  append_inst(program, v_mov_b32(8, 10));

  TestCodeObject code_object(program);
  auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_CDNA5);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 2u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::X);
  EXPECT_EQ(report.diagnostics[1].counter, WaitCounterKind::VmVsrc);
  EXPECT_EQ(report.diagnostics[1].access, WaitcheckAccessKind::Def);
  EXPECT_EQ(report.diagnostics[1].reg.cls, RegClass::VGPR);
  EXPECT_EQ(report.diagnostics[1].reg.index, 8u);
}

TEST(WaitcheckTest, Gfx1250WaitIdlePreservesExpertSchedulingMode) {
  constexpr uint32_t kHwRegWaveSchedMode = 26;
  std::vector<uint32_t> program;
  append_inst(program, s_setreg_imm32_b32(hwreg(kHwRegWaveSchedMode, 0, 2), 2));
  append_inst(program, s_wait_idle());
  append_inst(program, global_load_b32(0));
  append_inst(program, v_mov_b32(8, 10));

  TestCodeObject code_object(program);
  auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_CDNA5);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 2u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::X);
  EXPECT_EQ(report.diagnostics[1].counter, WaitCounterKind::VmVsrc);
  EXPECT_EQ(report.diagnostics[1].access, WaitcheckAccessKind::Def);
  EXPECT_EQ(report.diagnostics[1].reg.cls, RegClass::VGPR);
  EXPECT_EQ(report.diagnostics[1].reg.index, 8u);
}

TEST(WaitcheckTest, ScratchOffDoesNotReadVaddrZero) {
  std::vector<uint32_t> program;
  append_inst(program, global_load_b32(0));
  append_inst(program, scratch_load_b128_off(4));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA5);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, LoadcntZeroAlsoClearsVmemSourceRead) {
  std::vector<uint32_t> program;
  append_inst(program, global_load_b32(0));
  append_inst(program, sopp(64, 0)); // s_wait_loadcnt 0
  append_inst(program, v_mov_b32(8, 10));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, ReportsWrongLoadcntForNewestEvent) {
  std::vector<uint32_t> program;
  append_inst(program, global_load_b32(0));
  append_inst(program, global_load_b32(1));
  append_inst(program, sopp(64, 1)); // leaves the newest load pending
  append_inst(program, v_mov_b32(2, 1));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::VGPR);
  EXPECT_EQ(report.diagnostics[0].reg.index, 1u);
  EXPECT_EQ(report.diagnostics[0].required_count, 0u);
}

TEST(WaitcheckTest, AcceptsLoadcntOneForOlderEvent) {
  std::vector<uint32_t> program;
  append_inst(program, global_load_b32(0));
  append_inst(program, global_load_b32(1));
  append_inst(program, sopp(64, 1)); // completes the older load, leaves v1 pending
  append_inst(program, v_mov_b32(2, 0));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, LoadcntOneRetiresOnlyOlderVmemSourceRead) {
  std::vector<uint32_t> program;
  auto older = global_load_b32(0);
  older.vaddr = 8;
  auto newer = global_load_b32(1);
  newer.vaddr = 9;
  append_inst(program, older);
  append_inst(program, newer);
  append_inst(program, sopp(64, 1)); // leaves only the newer load and its source read pending
  append_inst(program, v_mov_b32(8, 10));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, ReportsUnorderedVmemLoadOverwrite) {
  std::vector<uint32_t> program;
  append_inst(program, global_load_b32(0));
  append_inst(program, global_load_b32(0));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported);
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Load);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Def);
}

TEST(WaitcheckTest, ReportsUnorderedImageSampleOverwrite) {
  std::vector<uint32_t> program;
  append_inst(program, image_sample(4));
  append_inst(program, image_sample(4));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported);
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Sample);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Def);
}

TEST(WaitcheckTest, ReportsFlatToVmemLoadOverwriteOnUnresolvedDsCounter) {
  std::vector<uint32_t> program;
  append_inst(program, flat_load_b32(0));
  append_inst(program, global_load_b32(0));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported);
  ASSERT_EQ(report.diagnostics.size(), 2u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Load);
  EXPECT_EQ(report.diagnostics[1].counter, WaitCounterKind::Ds);
  for (const auto &diagnostic : report.diagnostics) {
    EXPECT_EQ(diagnostic.access, WaitcheckAccessKind::Def);
    EXPECT_EQ(diagnostic.reg, (RegisterRef{RegClass::VGPR, 0, 1}));
  }
}

TEST(WaitcheckTest, AcceptsGfx942FlatToScratchOverwriteAfterLgkmcntWaitLikeLlvm) {
  // LLVM SIInsertWaitcnts relies on same-class VMEM loads writing VGPR
  // results in order.  The lgkmcnt wait retires the older generic FLAT load's
  // possible LDS completion; no vmcnt wait is needed before the scratch load.
  std::vector<uint32_t> program{
      0xDC500000u, 0x6D00000Cu, // flat_load_dword v109, v[12:13]
      0xBF8CC07Fu,              // s_waitcnt lgkmcnt(0)
      0xDC504000u, 0x6D210000u, // scratch_load_dword v109, off, s33
  };

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA3);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Cdna3And4ActiveFlatLoadsCreditBothPartialWaitCounters) {
  // Both generic FLAT loads issue VM and LGKM credits, even though each
  // lane's result completes on only one route. LLVM MC encodes the same
  // instructions on gfx942 and gfx950.
  const std::vector<uint32_t> prefix{
      0xDC500000u, 0x0000000Cu, // flat_load_dword v0, v[12:13]
      0xDC500000u, 0x0100000Eu, // flat_load_dword v1, v[14:15]
  };
  const auto check = [&](rj_code_arch_t arch, uint32_t wait) {
    auto program = prefix;
    program.insert(program.end(), {wait, 0x7E040300u, 0xBF810000u}); // v_mov v2,v0; end
    return analyze_waitcnts(program, arch);
  };
  for (auto arch : {ROCJITSU_CODE_ARCH_CDNA3, ROCJITSU_CODE_ARCH_CDNA4}) {
    SCOPED_TRACE(arch);
    const auto both = check(arch, 0xBF8C0070u); // vmcnt(0) lgkmcnt(0)
    ASSERT_TRUE(both.supported && both.complete) << both.analysis_error;
    EXPECT_TRUE(both.diagnostics.empty()) << diagnostic_summary(both);

    const auto weak_vm = check(arch, 0xBF8C0071u); // vmcnt(1) lgkmcnt(0)
    ASSERT_TRUE(weak_vm.supported && weak_vm.complete) << weak_vm.analysis_error;
    ASSERT_EQ(weak_vm.diagnostics.size(), 1u) << diagnostic_summary(weak_vm);
    EXPECT_EQ(weak_vm.diagnostics.front().counter, WaitCounterKind::Load);

    const auto weak_ds = check(arch, 0xBF8C0170u); // vmcnt(0) lgkmcnt(1)
    ASSERT_TRUE(weak_ds.supported && weak_ds.complete) << weak_ds.analysis_error;
    ASSERT_EQ(weak_ds.diagnostics.size(), 1u) << diagnostic_summary(weak_ds);
    EXPECT_EQ(weak_ds.diagnostics.front().counter, WaitCounterKind::Ds);
  }
}

TEST(WaitcheckTest, LegacyUnroutedFlatCannotMakeOlderNativeLoadReadyAtCountOne) {
  for (auto arch : {ROCJITSU_CODE_ARCH_CDNA3, ROCJITSU_CODE_ARCH_CDNA4}) {
    SCOPED_TRACE(arch);
    for (bool ds : {false, true})
      for (bool full_wait : {false, true}) {
        SCOPED_TRACE(ds);
        std::vector<uint32_t> program =
            ds ? std::vector<uint32_t>{0xD86C0000u, 0x00000001u}   // ds_read_b32 v0,v1
               : std::vector<uint32_t>{0xDC508000u, 0x007F0002u};  // global_load_dword v0
        program.insert(program.end(), {0xDC500000u, 0x0100000Eu}); // flat_load_dword v1
        const uint32_t weak_wait = ds ? 0xBF8C0170u : 0xBF8C0071u;
        program.insert(program.end(), {full_wait ? 0xBF8C0070u : weak_wait, 0x7E040300u,
                                       0xBF810000u}); // v_mov v2,v0; end
        const auto report = analyze_waitcnts(program, arch);
        ASSERT_TRUE(report.supported && report.complete) << report.analysis_error;
        const bool missing = std::ranges::any_of(report.diagnostics, [&](const auto &d) {
          return d.counter == (ds ? WaitCounterKind::Ds : WaitCounterKind::Load) &&
                 d.producer_section_offset == 0 && d.section_offset == 20 &&
                 d.access == WaitcheckAccessKind::Use;
        });
        EXPECT_EQ(missing, !full_wait) << diagnostic_summary(report);
      }
  }
}

TEST(WaitcheckTest, PreGfx12FlatReadsItsAddressBeforeCounterCapacityRetirement) {
  // A full VMcnt queue cannot retire the oldest load before the incoming
  // generic FLAT reads that load's address result.
  std::vector<uint32_t> program{
      0xBEFE01C1u,              // s_mov_b64 exec, -1
      0xDC548000u, 0x0C7F0002u, // global_load_dwordx2 v[12:13], v[2:3], off
  };
  for (uint32_t dest = 16; dest < 78; ++dest) {
    program.push_back(0xDC508000u);                // global_load_dword
    program.push_back((dest << 24) | 0x007F0002u); // v{dest}, v[2:3], off
  }
  const auto flat_offset = program.size() * sizeof(uint32_t);
  program.insert(program.end(), {0xDC500000u, 0x0000000Cu}); // flat_load_dword v0, v[12:13]
  for (auto arch : {ROCJITSU_CODE_ARCH_CDNA3, ROCJITSU_CODE_ARCH_CDNA4}) {
    SCOPED_TRACE(arch);
    const auto report = analyze_waitcnts(program, arch);
    ASSERT_TRUE(report.supported && report.complete) << report.analysis_error;
    ASSERT_FALSE(report.diagnostics.empty()) << diagnostic_summary(report);
    EXPECT_EQ(report.diagnostics.front().counter, WaitCounterKind::Load);
    EXPECT_EQ(report.diagnostics.front().access, WaitcheckAccessKind::Use);
    EXPECT_EQ(report.diagnostics.front().section_offset, flat_offset);
    auto waited = program;
    waited.insert(waited.end() - 2, 0xBF8C0070u); // vmcnt(0) lgkmcnt(0)
    const auto safe = analyze_waitcnts(waited, arch);
    ASSERT_TRUE(safe.supported && safe.complete) << safe.analysis_error;
    EXPECT_TRUE(safe.diagnostics.empty()) << diagnostic_summary(safe);
  }
}

TEST(WaitcheckTest, ReportsFlatToDsLoadOverwriteOnBothPossibleFlatCounters) {
  std::vector<uint32_t> program;
  append_inst(program, flat_load_b32(0));
  append_inst(program, ds_load_b32(0, 4));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported);
  ASSERT_EQ(report.diagnostics.size(), 2u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Load);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Def);
  EXPECT_EQ(report.diagnostics[0].reg, (RegisterRef{RegClass::VGPR, 0, 1}));
  EXPECT_EQ(report.diagnostics[1].counter, WaitCounterKind::Ds);
  EXPECT_EQ(report.diagnostics[1].access, WaitcheckAccessKind::Def);
  EXPECT_EQ(report.diagnostics[1].reg, (RegisterRef{RegClass::VGPR, 0, 1}));
}

TEST(WaitcheckTest, ReportsDsToFlatLoadOverwrite) {
  std::vector<uint32_t> program;
  append_inst(program, ds_load_b32(0, 4));
  append_inst(program, flat_load_b32(0));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported);
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Ds);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Def);
  EXPECT_EQ(report.diagnostics[0].reg, (RegisterRef{RegClass::VGPR, 0, 1}));
}

TEST(WaitcheckTest, ReportsVmemToFlatLoadOverwrite) {
  std::vector<uint32_t> program;
  append_inst(program, global_load_b32(0));
  append_inst(program, flat_load_b32(0));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported);
  ASSERT_EQ(report.diagnostics.size(), 1u);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Load);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Def);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::VGPR);
  EXPECT_EQ(report.diagnostics[0].reg.index, 0u);
}

TEST(WaitcheckTest, ReportsLoadcntBeforeImageLoadSampleOverwrite) {
  std::vector<uint32_t> program;
  append_inst(program, image_load(4));
  append_inst(program, image_sample(4));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported);
  ASSERT_EQ(report.diagnostics.size(), 1u);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Load);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Def);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::VGPR);
  EXPECT_EQ(report.diagnostics[0].reg.index, 4u);
  EXPECT_EQ(report.diagnostics[0].required_count, 0u);
}

TEST(WaitcheckTest, AcceptsLoadcntBeforeImageLoadSampleOverwrite) {
  std::vector<uint32_t> program;
  append_inst(program, image_load(4));
  append_inst(program, sopp(64, 0)); // s_wait_loadcnt 0
  append_inst(program, image_sample(4));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty());
}

TEST(WaitcheckTest, ReportsSamplecntBeforeImageSampleLoadOverwrite) {
  std::vector<uint32_t> program;
  append_inst(program, image_sample(4));
  append_inst(program, image_load(4));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported);
  ASSERT_EQ(report.diagnostics.size(), 1u);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Sample);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Def);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::VGPR);
  EXPECT_EQ(report.diagnostics[0].reg.index, 4u);
  EXPECT_EQ(report.diagnostics[0].required_count, 0u);
}

TEST(WaitcheckTest, AcceptsSamplecntBeforeImageSampleLoadOverwrite) {
  std::vector<uint32_t> program;
  append_inst(program, image_sample(4));
  append_inst(program, sopp(66, 0)); // s_wait_samplecnt 0
  append_inst(program, image_load(4));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty());
}

TEST(WaitcheckTest, ReportsLoadcntAndDscntBeforeFlatLoadUse) {
  std::vector<uint32_t> program;
  append_inst(program, flat_load_b32(0));
  append_inst(program, v_mov_b32(1, 0));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported);
  ASSERT_EQ(report.diagnostics.size(), 2u);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Load);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Use);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::VGPR);
  EXPECT_EQ(report.diagnostics[0].reg.index, 0u);
  EXPECT_EQ(report.diagnostics[1].counter, WaitCounterKind::Ds);
  EXPECT_EQ(report.diagnostics[1].access, WaitcheckAccessKind::Use);
  EXPECT_EQ(report.diagnostics[1].reg.cls, RegClass::VGPR);
  EXPECT_EQ(report.diagnostics[1].reg.index, 0u);
}

TEST(WaitcheckTest, ReportsDscntWhenOnlyLoadcntWaitsFlatLoad) {
  std::vector<uint32_t> program;
  append_inst(program, flat_load_b32(0));
  append_inst(program, sopp(64, 0)); // s_wait_loadcnt 0
  append_inst(program, v_mov_b32(1, 0));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported);
  ASSERT_EQ(report.diagnostics.size(), 1u);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Ds);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Use);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::VGPR);
  EXPECT_EQ(report.diagnostics[0].reg.index, 0u);
}

TEST(WaitcheckTest, ReportsLoadcntWhenOnlyDscntWaitsFlatLoad) {
  std::vector<uint32_t> program;
  append_inst(program, flat_load_b32(0));
  append_inst(program, sopp(70, 0)); // s_wait_dscnt 0
  append_inst(program, v_mov_b32(1, 0));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported);
  ASSERT_EQ(report.diagnostics.size(), 1u);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Load);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Use);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::VGPR);
  EXPECT_EQ(report.diagnostics[0].reg.index, 0u);
}

TEST(WaitcheckTest, AcceptsCombinedLoadcntDscntBeforeFlatLoadUse) {
  std::vector<uint32_t> program;
  append_inst(program, flat_load_b32(0));
  append_inst(program, sopp(72, 0)); // s_wait_loadcnt_dscnt loadcnt(0), dscnt(0)
  append_inst(program, v_mov_b32(1, 0));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty());
}

// Encodings from the standalone gfx1250 reproducer in issue #11843.
TEST(WaitcheckTest, Gfx1250ImplicitUsesRespectRegisterWindows) {
  struct Case {
    const char *name;
    std::vector<uint32_t> consumer;
    bool hazard;
    WaitcheckAccessKind access = WaitcheckAccessKind::Use;
  };
  const Case cases[] = {
      {"windowed writelane", {0xbf860140, 0xd7610001, 0x02010800}, false},
      {"same-register writelane", {0xd7610001, 0x02010800}, true, WaitcheckAccessKind::Def},
      {"disjoint writelane", {0xd7610005, 0x02010800}, false},
      {"real explicit read", {0x06040301}, true},                  // v_add_f32 v2, v1, v1
      {"windowed explicit read", {0xbf860101, 0x7e040301}, false}, // v_mov_b32 v2, v1
      {"windowed full write", {0xbf860140, 0x06020702}, false},
      {"windowed partial write", {0xbf860140, 0x64020702}, false},
      // An explicit low-bank source must survive remapping an implicit use of
      // the same encoded VGPR through the destination's different bank.
      {"low source aliases encoded destination", {0xbf860140, 0x64020701}, true},
  };
  for (const auto &test : cases) {
    SCOPED_TRACE(test.name);
    // buffer_load_b32 v1, v28, s[0:3], null offen
    std::vector<uint32_t> program{0xc405007c, 0x40800001, 0x0000001c};
    program.insert(program.end(), test.consumer.begin(), test.consumer.end());
    auto report = analyze_gfx1250_normal(program);
    ASSERT_TRUE(report.supported) << report.analysis_error;
    ASSERT_EQ(report.diagnostics.size(), test.hazard ? 1u : 0u) << diagnostic_summary(report);
    if (test.hazard) {
      EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Load);
      EXPECT_EQ(report.diagnostics[0].reg, (RegisterRef{RegClass::VGPR, 1, 1}));
      EXPECT_EQ(report.diagnostics[0].access, test.access);
      EXPECT_EQ(report.diagnostics[0].required_count, 0u);
    }
  }
}

TEST(WaitcheckTest, Gfx1250ImplicitUsesRetainHighBankDependencies) {
  for (uint32_t bank = 1; bank < 4; ++bank) {
    for (bool wait : {false, true}) {
      SCOPED_TRACE(bank);
      SCOPED_TRACE(wait);
      std::vector<uint32_t> program;
      append_inst(program, s_set_vgpr_msb(vgpr_msb_mode(0, 0, 0, bank)));
      // Load and preserve-read the same high-bank v1, with low-bank sources.
      program.insert(program.end(), {0xc405007c, 0x40800001, 0x0000001c});
      if (wait)
        append_inst(program, sopp(64, 0)); // s_wait_loadcnt 0
      program.push_back(0x64020702);       // v_add_f16 v1, v2, v3
      auto report = analyze_gfx1250_normal(program);
      ASSERT_TRUE(report.supported) << report.analysis_error;
      ASSERT_EQ(report.diagnostics.size(), wait ? 0u : 1u) << diagnostic_summary(report);
      if (!wait) {
        EXPECT_EQ(report.diagnostics[0].reg,
                  (RegisterRef{RegClass::VGPR, static_cast<uint16_t>(1 + 256 * bank), 1}));
        EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Load);
        EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Use);
      }
    }
  }
}

TEST(WaitcheckTest, Gfx1250UnknownWindowKeepsImplicitHighBankDependency) {
  // Load v257, then make the window unknown. The partial write must retain
  // its possible dependency on v257 rather than assuming bank zero.
  const std::vector<uint32_t> program{
      0xbf860140,                         // s_set_vgpr_msb 0x140
      0xc405007c, 0x40800001, 0x0000001c, // buffer_load_b32 v1, v28, s[0:3], null offen
      0xb908f801,                         // s_setreg_b32 hwreg(HW_REG_MODE), s8
      0x64020702,                         // v_add_f16 v1, v2, v3
  };
  auto report = analyze_gfx1250_normal(program);
  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].reg, (RegisterRef{RegClass::VGPR, 257, 1}));
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Load);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Use);
}

TEST(WaitcheckTest, Gfx1250HighVgprModeSeparatesLowLoadFromHighStoreData) {
  std::vector<uint32_t> program;
  append_inst(program, s_set_vgpr_msb(vgpr_msb_mode(0, 0, 0, 0)));
  append_inst(program, flat_load_b32(0));
  append_inst(program, s_set_vgpr_msb(vgpr_msb_mode(0, 1, 0, 0)));
  append_inst(program, flat_store_b32(0));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA5);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx1250HighVgprModeSeparatesLowLoadFromV512StoreData) {
  std::vector<uint32_t> program;
  append_inst(program, s_set_vgpr_msb(vgpr_msb_mode(0, 0, 0, 0)));
  append_inst(program, flat_load_b32(0));
  append_inst(program, s_set_vgpr_msb(vgpr_msb_mode(0, 2, 0, 0)));
  append_inst(program, flat_store_b32(0));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA5);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx1250HighVgprModeReportsSameHighStoreDataUse) {
  std::vector<uint32_t> program;
  append_inst(program, s_set_vgpr_msb(vgpr_msb_mode(0, 0, 0, 1)));
  append_inst(program, flat_load_b32(255));
  append_inst(program, s_set_vgpr_msb(vgpr_msb_mode(0, 1, 0, 0)));
  append_inst(program, flat_store_b32(255));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA5);

  ASSERT_TRUE(report.supported);
  ASSERT_EQ(report.diagnostics.size(), 2u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Load);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Use);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::VGPR);
  EXPECT_EQ(report.diagnostics[0].reg.index, 511u);
  EXPECT_EQ(report.diagnostics[1].counter, WaitCounterKind::Ds);
  EXPECT_EQ(report.diagnostics[1].access, WaitcheckAccessKind::Use);
  EXPECT_EQ(report.diagnostics[1].reg.cls, RegClass::VGPR);
  EXPECT_EQ(report.diagnostics[1].reg.index, 511u);
}

TEST(WaitcheckTest, Gfx1250HighVgprModeAcceptsCombinedWaitBeforeSameHighStoreDataUse) {
  std::vector<uint32_t> program;
  append_inst(program, s_set_vgpr_msb(vgpr_msb_mode(0, 0, 0, 2)));
  append_inst(program, flat_load_b32(0));
  append_inst(program, sopp(72, 0)); // s_wait_loadcnt_dscnt loadcnt(0), dscnt(0)
  append_inst(program, s_set_vgpr_msb(vgpr_msb_mode(0, 2, 0, 0)));
  append_inst(program, flat_store_b32(0));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA5);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx1250HighVgprModeReportsSameV768StoreDataUse) {
  std::vector<uint32_t> program;
  append_inst(program, s_set_vgpr_msb(vgpr_msb_mode(0, 0, 0, 3)));
  append_inst(program, flat_load_b32(0));
  append_inst(program, s_set_vgpr_msb(vgpr_msb_mode(0, 3, 0, 0)));
  append_inst(program, flat_store_b32(0));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA5);

  ASSERT_TRUE(report.supported);
  ASSERT_EQ(report.diagnostics.size(), 2u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Load);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Use);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::VGPR);
  EXPECT_EQ(report.diagnostics[0].reg.index, 768u);
  EXPECT_EQ(report.diagnostics[1].counter, WaitCounterKind::Ds);
  EXPECT_EQ(report.diagnostics[1].access, WaitcheckAccessKind::Use);
  EXPECT_EQ(report.diagnostics[1].reg.cls, RegClass::VGPR);
  EXPECT_EQ(report.diagnostics[1].reg.index, 768u);
}

TEST(WaitcheckTest, Gfx1250HighVgprModeSeparatesDifferentHighBanks) {
  std::vector<uint32_t> program;
  append_inst(program, s_set_vgpr_msb(vgpr_msb_mode(0, 0, 0, 3)));
  append_inst(program, flat_load_b32(0));
  append_inst(program, s_set_vgpr_msb(vgpr_msb_mode(0, 2, 0, 0)));
  append_inst(program, flat_store_b32(0));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA5);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx1250WaitIdlePreservesHighVgprMode) {
  std::vector<uint32_t> program;
  append_inst(program, s_set_vgpr_msb(vgpr_msb_mode(0, 0, 0, 2)));
  append_inst(program, s_wait_idle());
  append_inst(program, flat_load_b32(0));
  append_inst(program, s_set_vgpr_msb(vgpr_msb_mode(0, 2, 0, 0)));
  append_inst(program, flat_store_b32(0));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA5);

  ASSERT_TRUE(report.supported);
  ASSERT_EQ(report.diagnostics.size(), 2u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::VGPR);
  EXPECT_EQ(report.diagnostics[0].reg.index, 512u);
  EXPECT_EQ(report.diagnostics[1].reg.cls, RegClass::VGPR);
  EXPECT_EQ(report.diagnostics[1].reg.index, 512u);
}

TEST(WaitcheckTest, ReportsMissingWaitAluSaSdstBeforeValuReadsTrackedSgpr) {
  std::vector<uint32_t> program;
  append_inst(program, v_add_f32_e32(0, 102, 0)); // VALU reads s102, tracking its SGPR pair.
  append_inst(program, s_mov_b32(102, 128));      // SALU writes the tracked SGPR.
  append_inst(program, v_add_f32_e32(1, 102, 1));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported);
  ASSERT_EQ(report.diagnostics.size(), 1u);
  EXPECT_EQ(report.diagnostics[0].kind, WaitcheckDiagnosticKind::SgprDepctr);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Depctr);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Use);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::SGPR);
  EXPECT_EQ(report.diagnostics[0].reg.index, 102u);
  EXPECT_NE(report.diagnostics[0].message.find("depctr_sa_sdst(0)"), std::string::npos);
}

TEST(WaitcheckTest, SaluConsumerDoesNotRequireWaitAluSaSdst) {
  std::vector<uint32_t> program;
  append_inst(program, v_add_f32_e32(0, 102, 0));
  append_inst(program, s_mov_b32(102, 128));
  append_inst(program, s_mov_b32(0, 102));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, AcceptsWaitAluSaSdstBeforeValuReadsTrackedSgpr) {
  std::vector<uint32_t> program;
  append_inst(program, v_add_f32_e32(0, 102, 0));
  append_inst(program, s_mov_b32(102, 128));
  append_inst(program, s_wait_alu_sa_sdst_0());
  append_inst(program, v_add_f32_e32(1, 102, 1));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty());
}

TEST(WaitcheckTest, AcceptsDelayAluSaluCycleBeforeValuReadsTrackedSgpr) {
  std::vector<uint32_t> program;
  append_inst(program, v_add_f32_e32(0, 102, 0));
  append_inst(program, s_mov_b32(102, 128));
  append_inst(program, s_delay_alu(9)); // SALU_CYCLE_1 for the next instruction.
  append_inst(program, v_add_f32_e32(1, 102, 1));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, AcceptsDelayAluSkippedSaluCycleBeforeValuReadsTrackedSgpr) {
  std::vector<uint32_t> program;
  append_inst(program, v_add_f32_e32(0, 102, 0));
  append_inst(program, s_mov_b32(102, 128));
  append_inst(program, s_delay_alu((3u << 4u) | (9u << 7u)));
  append_inst(program, s_mov_b32(0, 128));
  append_inst(program, s_mov_b32(1, 128));
  append_inst(program, s_mov_b32(2, 128));
  append_inst(program, v_add_f32_e32(1, 102, 1));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, ReportsWaitAluSaSdstAfterOnlyThreeDsNops) {
  std::vector<uint32_t> program;
  append_inst(program, v_add_f32_e32(0, 102, 0));
  append_inst(program, ds_nop());
  append_inst(program, ds_nop());
  append_inst(program, ds_nop());
  append_inst(program, s_mov_b32(102, 128));
  append_inst(program, v_add_f32_e32(1, 102, 1));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported);
  ASSERT_EQ(report.diagnostics.size(), 1u);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Depctr);
  EXPECT_NE(report.diagnostics[0].message.find("depctr_sa_sdst(0)"), std::string::npos);
}

TEST(WaitcheckTest, FourDsNopsCullTrackedSgprHazardState) {
  std::vector<uint32_t> program;
  append_inst(program, v_add_f32_e32(0, 102, 0));
  append_inst(program, ds_nop());
  append_inst(program, ds_nop());
  append_inst(program, ds_nop());
  append_inst(program, ds_nop());
  append_inst(program, s_mov_b32(102, 128));
  append_inst(program, v_add_f32_e32(1, 102, 1));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty());
}

TEST(WaitcheckTest, NonFlatVmemCullClearsTrackedSgprHazardState) {
  std::vector<uint32_t> program;
  append_inst(program, v_add_f32_e32(0, 102, 0));
  append_inst(program, s_mov_b32(102, 128));
  append_inst(program, global_load_b32(4));
  append_inst(program, v_add_f32_e32(1, 102, 1));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, ScalarMemoryCullClearsTrackedSgprHazardState) {
  std::vector<uint32_t> program;
  append_inst(program, v_add_f32_e32(0, 102, 0));
  append_inst(program, s_mov_b32(102, 128));
  append_inst(program, s_load_b32(4, 0));
  append_inst(program, v_add_f32_e32(1, 102, 1));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, ScratchLoadDoesNotCullTrackedSgprHazardState) {
  std::vector<uint32_t> program;
  append_inst(program, v_add_f32_e32(0, 102, 0));
  append_inst(program, s_mov_b32(102, 128));
  append_inst(program, scratch_load_b32(4));
  append_inst(program, v_add_f32_e32(1, 102, 1));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported);
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Depctr);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Use);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::SGPR);
  EXPECT_EQ(report.diagnostics[0].reg.index, 102u);
  EXPECT_NE(report.diagnostics[0].message.find("depctr_sa_sdst(0)"), std::string::npos);
}

TEST(WaitcheckTest, ReportsMissingWaitAluVaSdstBeforeValuReadsTrackedSgpr) {
  std::vector<uint32_t> program;
  append_inst(program, v_add_f32_e32(0, 2, 0)); // VALU reads s2, tracking its SGPR pair.
  append_v_cmp_gt_u32_s2_s5_v12(program);       // VALU writes s2 as a scalar mask.
  append_inst(program, v_add_f32_e32(1, 2, 1));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported);
  ASSERT_EQ(report.diagnostics.size(), 1u);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Depctr);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Use);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::SGPR);
  EXPECT_EQ(report.diagnostics[0].reg.index, 2u);
  EXPECT_NE(report.diagnostics[0].message.find("depctr_va_sdst(0)"), std::string::npos);
}

TEST(WaitcheckTest, UnrelatedSaluRegisterOperandWaitsForValuSgprWrites) {
  std::vector<uint32_t> program;
  append_inst(program, v_add_f32_e32(0, 2, 0)); // Track the s[2:3] pair.
  const auto readlane =
      build_v_readlane_b32(/*sdst=*/2, /*vsrc=*/40, /*lane=*/0, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(readlane.succeeded());
  program.insert(program.end(), readlane.value().begin(), readlane.value().end());
  append_inst(program, s_or_saveexec_b32(/*sdst=*/68, /*ssrc0=*/193)); // 193 is inline constant -1.
  append_inst(program, v_add_f32_e32(1, 2, 1));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, SpecialSaluRegisterOperandsWaitForValuSgprWrites) {
  std::vector<uint32_t> program;
  append_inst(program, v_add_f32_e32(0, 2, 0));
  const auto readlane =
      build_v_readlane_b32(/*sdst=*/2, /*vsrc=*/40, /*lane=*/0, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(readlane.succeeded());
  program.insert(program.end(), readlane.value().begin(), readlane.value().end());
  program.push_back(0xBEFC01EBu); // s_mov_b64 null, src_shared_base
  append_inst(program, v_add_f32_e32(1, 2, 1));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, ImplicitSaluRegisterOperandWaitsForValuSgprWrites) {
  std::vector<uint32_t> program;
  append_inst(program, v_add_f32_e32(0, 2, 0));
  const auto readlane =
      build_v_readlane_b32(/*sdst=*/2, /*vsrc=*/40, /*lane=*/0, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(readlane.succeeded());
  program.insert(program.end(), readlane.value().begin(), readlane.value().end());
  append_inst(program, sopp(/*op=*/33, /*simm16=*/0)); // s_cbranch_scc0
  append_inst(program, v_add_f32_e32(1, 2, 1));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, HiddenModeRegisterOperandWaitsForValuSgprWrites) {
  std::vector<uint32_t> program;
  append_inst(program, v_add_f32_e32(0, 2, 0));
  const auto readlane =
      build_v_readlane_b32(/*sdst=*/2, /*vsrc=*/40, /*lane=*/0, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(readlane.succeeded());
  program.insert(program.end(), readlane.value().begin(), readlane.value().end());
  append_inst(program, s_setreg_imm32_b32(hwreg(/*id=*/1, /*offset=*/0, /*size=*/32), 0));
  append_inst(program, v_add_f32_e32(1, 2, 1));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, SaluWithoutRegisterOperandDoesNotWaitForValuSgprWrites) {
  std::vector<uint32_t> program;
  append_inst(program, v_add_f32_e32(0, 2, 0));
  append_v_cmp_gt_u32_s2_s5_v12(program);
  append_inst(program, sopp(/*op=*/0, /*simm16=*/0)); // s_nop 0
  append_inst(program, v_add_f32_e32(1, 2, 1));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_NE(report.diagnostics[0].message.find("depctr_va_sdst(0)"), std::string::npos);
}

TEST(WaitcheckTest, Gfx1201Wave32MadCarryDoesNotDefineEncodedHighSgpr) {
  for (const uint16_t opcode : {rdna4::kVMadCoU64U32Vop3SdstEnc, rdna4::kVMadCoI64I32Vop3SdstEnc}) {
    SCOPED_TRACE(opcode);
    std::vector<uint32_t> program;
    append_inst(program, v_add_f32_e32(4, 2, 4)); // Track the s[2:3] pair.
    append_v_mad_co_s2(program, opcode);
    append_inst(program, v_add_f32_e32(5, 3, 5));

    WaitcheckOptions options;
    options.fallback_wavefront_size = 32;
    auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4, options);

    EXPECT_TRUE(report.supported) << report.analysis_error;
    EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
  }
}

TEST(WaitcheckTest, Gfx1201Wave32MadCarryStillRequiresWaitForLowSgpr) {
  for (const uint16_t opcode : {rdna4::kVMadCoU64U32Vop3SdstEnc, rdna4::kVMadCoI64I32Vop3SdstEnc}) {
    SCOPED_TRACE(opcode);
    std::vector<uint32_t> program;
    append_inst(program, v_add_f32_e32(4, 2, 4));
    append_v_mad_co_s2(program, opcode);
    append_inst(program, v_add_f32_e32(5, 2, 5));

    auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

    ASSERT_TRUE(report.supported) << report.analysis_error;
    ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
    EXPECT_EQ(report.diagnostics[0].reg.index, 2u);
    EXPECT_NE(report.diagnostics[0].message.find("depctr_va_sdst(0)"), std::string::npos);
  }
}

TEST(WaitcheckTest, Gfx1201Wave64MadCarryDefinesEncodedHighSgpr) {
  for (const uint16_t opcode : {rdna4::kVMadCoU64U32Vop3SdstEnc, rdna4::kVMadCoI64I32Vop3SdstEnc}) {
    SCOPED_TRACE(opcode);
    std::vector<uint32_t> program;
    append_inst(program, v_add_f32_e32(4, 2, 4));
    append_v_mad_co_s2(program, opcode);
    append_inst(program, v_add_f32_e32(5, 3, 5));

    const auto image =
        rocjitsu::waitcheck_test::make_gfx_code_object(program, EF_AMDGPU_MACH_AMDGCN_GFX1201);
    AmdGpuCodeObject code_object(image.data(), image.size());
    auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_RDNA4);

    ASSERT_TRUE(report.supported) << report.analysis_error;
    ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
    EXPECT_EQ(report.diagnostics[0].reg.index, 3u);
    EXPECT_NE(report.diagnostics[0].message.find("depctr_va_sdst(0)"), std::string::npos);
  }
}

TEST(WaitcheckTest, AcceptsWaitAluVaSdstBeforeValuReadsTrackedSgpr) {
  std::vector<uint32_t> program;
  append_inst(program, v_add_f32_e32(0, 2, 0));
  append_v_cmp_gt_u32_s2_s5_v12(program);
  append_inst(program, s_wait_alu_va_sdst_0());
  append_inst(program, v_add_f32_e32(1, 2, 1));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty());
}

TEST(WaitcheckTest, AcceptsDelayAluValuDepBeforeValuReadsTrackedSgpr) {
  std::vector<uint32_t> program;
  append_inst(program, v_add_f32_e32(0, 2, 0));
  append_v_cmp_gt_u32_s2_s5_v12(program);
  append_inst(program, s_delay_alu(1)); // VALU_DEP_1 for the next instruction.
  append_inst(program, v_add_f32_e32(1, 2, 1));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, ReportsMissingWaitAluVaVccBeforeValuReadsTrackedVcc) {
  std::vector<uint32_t> program;
  append_inst(program, v_cndmask_b32_e32(0, 128, 1)); // VALU reads VCC, tracking it.
  append_inst(program, v_cmp_gt_u32_e32(5, 12));      // VALU writes VCC.
  append_inst(program, v_cndmask_b32_e32(2, 128, 2));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported);
  ASSERT_EQ(report.diagnostics.size(), 1u);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Depctr);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Use);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::VCC);
  EXPECT_NE(report.diagnostics[0].message.find("depctr_va_vcc(0)"), std::string::npos);
}

TEST(WaitcheckTest, Gfx1250VopdCndmaskDoesNotRequireWaitAluVaVcc) {
  std::vector<uint32_t> program;
  append_inst(program, v_cndmask_b32_e32(0, 128, 1)); // VALU reads VCC, tracking it.
  append_inst(program, v_cmp_gt_u32_e32(5, 12));      // VALU writes VCC.
  append_v_dual_cndmask_b32_v2_v1_v2_dual_mov_b32_v1_0(program);

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA5);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, AcceptsWaitAluVaVccBeforeValuReadsTrackedVcc) {
  std::vector<uint32_t> program;
  append_inst(program, v_cndmask_b32_e32(0, 128, 1));
  append_inst(program, v_cmp_gt_u32_e32(5, 12));
  append_inst(program, s_wait_alu_va_vcc_0());
  append_inst(program, v_cndmask_b32_e32(2, 128, 2));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty());
}

TEST(WaitcheckTest, AcceptsDelayAluValuDepBeforeValuReadsTrackedVcc) {
  std::vector<uint32_t> program;
  append_inst(program, v_cndmask_b32_e32(0, 128, 1));
  append_inst(program, v_cmp_gt_u32_e32(5, 12));
  append_inst(program, s_delay_alu(1)); // VALU_DEP_1 for the next instruction.
  append_inst(program, v_cndmask_b32_e32(2, 128, 2));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, ReportsMissingWaitAluSaSdstBeforeValuReadsSaluWrittenVcc) {
  std::vector<uint32_t> program;
  append_inst(program, v_cndmask_b32_e32(0, 128, 1)); // VALU reads VCC, tracking it.
  append_inst(program, s_cselect_b32(106));           // SALU writes vcc_lo.
  append_inst(program, v_cndmask_b32_e32(2, 128, 2));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported);
  ASSERT_EQ(report.diagnostics.size(), 1u);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Depctr);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Use);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::VCC);
  EXPECT_NE(report.diagnostics[0].message.find("depctr_sa_sdst(0)"), std::string::npos);
}

TEST(WaitcheckTest, AcceptsWaitAluSaSdstBeforeValuReadsSaluWrittenVcc) {
  std::vector<uint32_t> program;
  append_inst(program, v_cndmask_b32_e32(0, 128, 1));
  append_inst(program, s_cselect_b32(106));
  append_inst(program, s_wait_alu_sa_sdst_0());
  append_inst(program, v_cndmask_b32_e32(2, 128, 2));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty());
}

TEST(WaitcheckTest, SaluVccReadClearsValuVccHazard) {
  std::vector<uint32_t> program;
  append_inst(program, v_cndmask_b32_e32(0, 128, 1));
  append_inst(program, v_cmp_gt_u32_e32(5, 12));
  append_inst(program, s_mov_b32(0, 106)); // SALU reads vcc_lo, clearing the VALU VCC hazard.
  append_inst(program, v_cndmask_b32_e32(2, 128, 2));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty());
}

TEST(WaitcheckTest, Gfx1201ScalarVccBranchesClearValuVccBeforeIndirectTransfer) {
  for (const uint32_t branch : {33u, 34u, 35u, 36u, 37u, 38u}) {
    SCOPED_TRACE(branch);
    std::vector<uint32_t> program;
    append_inst(program, v_cndmask_b32_e32(0, 128, 1));
    append_inst(program, v_cmp_gt_u32_e32(5, 12));
    append_inst(program, sopp(branch, 0));
    program.push_back(build_s_setpc_b64(8, ROCJITSU_CODE_ARCH_RDNA4).value());
    const auto image = waitcheck_test::make_gfx_code_object(
        program, EF_AMDGPU_MACH_AMDGCN_GFX1201, /*wave32=*/true);
    const AmdGpuCodeObject object(image.data(), image.size());
    const auto report = analyze_waitcnts_for_kernel(object, ROCJITSU_CODE_ARCH_RDNA4, 0);
    ASSERT_TRUE(report.supported) << report.analysis_error;
    if (branch == 35u || branch == 36u) {
      EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
    } else {
      ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
      EXPECT_EQ(report.diagnostics.front().access, WaitcheckAccessKind::ControlTransfer);
      EXPECT_NE(report.diagnostics.front().message.find("depctr_va_vcc(0)"), std::string::npos);
    }
  }
}

TEST(WaitcheckTest, ScalarVccBranchRetiresConcurrentValuSgprAndVccWrites) {
  for (uint32_t branch : {35u, 36u}) {
    std::vector<uint32_t> program;
    append_inst(program, v_add_f32_e32(0, 2, 0));       // Track s2.
    append_inst(program, v_cndmask_b32_e32(0, 128, 1)); // Track VCC.
    const auto readlane = build_v_readlane_b32(2, 40, 0, ROCJITSU_CODE_ARCH_RDNA4);
    ASSERT_TRUE(readlane.succeeded());
    program.insert(program.end(), readlane.value().begin(), readlane.value().end());
    append_inst(program, v_cmp_gt_u32_e32(5, 12));
    const auto jump = build_s_setpc_b64(8, ROCJITSU_CODE_ARCH_RDNA4).value();
    auto control = program;
    control.push_back(jump);
    const auto outstanding = analyze_waitcnts(control, ROCJITSU_CODE_ARCH_RDNA4);
    ASSERT_EQ(outstanding.diagnostics.size(), 2u) << diagnostic_summary(outstanding);
    bool sdst = false, vcc = false;
    for (const auto &diagnostic : outstanding.diagnostics) {
      sdst |= diagnostic.message.find("depctr_va_sdst(0)") != std::string::npos;
      vcc |= diagnostic.message.find("depctr_va_vcc(0)") != std::string::npos;
    }
    EXPECT_TRUE(sdst && vcc) << diagnostic_summary(outstanding);
    append_inst(program, sopp(branch, 0));
    program.push_back(jump);
    const auto drained = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);
    EXPECT_TRUE(drained.diagnostics.empty()) << diagnostic_summary(drained);
  }
}

// Typed operand classification must retain both selector-encoded halves and
// the fieldless VCC operands of VALU instructions, without confusing SGPRs.
TEST(WaitcheckTest, VccSelectorHalvesPreserveSaluReadAndWriteHazards) {
  for (uint32_t selector : {0u, 106u, 107u}) {
    SCOPED_TRACE(selector);
    std::vector<uint32_t> program;
    append_inst(program, v_cndmask_b32_e32(0, 128, 1));
    append_inst(program, s_cselect_b32(selector));
    append_inst(program, v_cndmask_b32_e32(2, 128, 2));
    auto written = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);
    ASSERT_TRUE(written.supported);
    if (selector == 0) {
      EXPECT_TRUE(written.diagnostics.empty()) << diagnostic_summary(written);
    } else {
      ASSERT_EQ(written.diagnostics.size(), 1u) << diagnostic_summary(written);
      EXPECT_EQ(written.diagnostics[0].reg.cls, RegClass::VCC);
    }

    program.clear();
    append_inst(program, v_cndmask_b32_e32(0, 128, 1));
    append_inst(program, v_cmp_gt_u32_e32(5, 12));
    append_inst(program, s_mov_b32(4, selector));
    append_inst(program, v_cndmask_b32_e32(2, 128, 2));
    auto read = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);
    ASSERT_TRUE(read.supported);
    if (selector == 0) {
      ASSERT_EQ(read.diagnostics.size(), 1u) << diagnostic_summary(read);
      EXPECT_EQ(read.diagnostics[0].reg.cls, RegClass::VCC);
    } else {
      EXPECT_TRUE(read.diagnostics.empty()) << diagnostic_summary(read);
    }
  }
}

TEST(WaitcheckTest, ExecSelectorHalvesPreserveExportOverwriteHazards) {
  for (uint32_t selector : {4u, 126u, 127u}) {
    SCOPED_TRACE(selector);
    std::vector<uint32_t> program;
    append_inst(program, export_mrt0_v0());
    append_inst(program, s_mov_b32(selector, 128));
    auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);
    ASSERT_TRUE(report.supported);
    if (selector == 4) {
      EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
    } else {
      ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
      EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Exp);
      EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::EXEC);
    }
  }
}

TEST(WaitcheckTest, ReportsMissingDscntBeforeDsLoadUse) {
  std::vector<uint32_t> program;
  append_inst(program, ds_load_b32(0, 4));
  append_inst(program, v_mov_b32(1, 0));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported);
  ASSERT_EQ(report.diagnostics.size(), 1u);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Ds);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Use);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::VGPR);
  EXPECT_EQ(report.diagnostics[0].reg.index, 0u);
}

TEST(WaitcheckTest, AcceptsDscntZeroBeforeDsLoadUse) {
  std::vector<uint32_t> program;
  append_inst(program, ds_load_b32(0, 4));
  append_inst(program, sopp(70, 0)); // s_wait_dscnt 0
  append_inst(program, v_mov_b32(1, 0));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty());
}

TEST(WaitcheckTest, DscntPartialWaitCountsYoungerDsSwizzles) {
  std::vector<uint32_t> program;
  append_inst(program, ds_load_b32(29, 4));
  append_inst(program, ds_swizzle_b32(28, 4));
  append_inst(program, ds_swizzle_b32(5, 19));
  append_inst(program, sopp(70, 2)); // s_wait_dscnt 2 retires the oldest result.
  append_inst(program, v_mov_b32(29, 4));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA5);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, ReportsDscntMaxCountForOverflowSizedQueue) {
  std::vector<uint32_t> program;
  append_ds_loads(program, 40, 99);
  append_inst(program, v_mov_b32(40, 0));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  expect_single_overflow_diagnostic(report, WaitCounterKind::Ds, WaitcheckAccessKind::Use,
                                    RegClass::VGPR, 0);
}

TEST(WaitcheckTest, AcceptsDscntMaxCountForOldestOverflowSizedQueue) {
  std::vector<uint32_t> program;
  append_ds_loads(program, 40, 99);
  append_inst(program, sopp(70, kOverflowRequiredCount)); // s_wait_dscnt 39
  append_inst(program, v_mov_b32(40, 0));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, DscntMaxCountLeavesSecondOldestOverflowSizedQueuePending) {
  std::vector<uint32_t> program;
  append_ds_loads(program, 40, 99);
  append_inst(program, sopp(70, kOverflowRequiredCount)); // s_wait_dscnt 39
  append_inst(program, v_mov_b32(40, 1));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported);
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Ds);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Use);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::VGPR);
  EXPECT_EQ(report.diagnostics[0].reg.index, 1u);
  EXPECT_EQ(report.diagnostics[0].required_count, kOverflowRequiredCount - 1);
}

TEST(WaitcheckTest, ReportsLoadcntMaxCountForOverflowSizedQueue) {
  std::vector<uint32_t> program;
  append_global_loads(program, kOverflowQueueSize, kOverflowBaseVgpr);
  append_inst(program, v_mov_b32(kOverflowConsumerVgpr, kOverflowBaseVgpr));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  expect_single_overflow_diagnostic(report, WaitCounterKind::Load, WaitcheckAccessKind::Use,
                                    RegClass::VGPR, kOverflowBaseVgpr);
}

TEST(WaitcheckTest, AcceptsLoadcntMaxCountForOldestOverflowSizedQueue) {
  std::vector<uint32_t> program;
  append_global_loads(program, kOverflowQueueSize, kOverflowBaseVgpr);
  append_inst(program, sopp(64, kOverflowRequiredCount)); // s_wait_loadcnt 39
  append_inst(program, v_mov_b32(kOverflowConsumerVgpr, kOverflowBaseVgpr));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

// Minimized from a gfx1150 CK kernel in the PyTorch nightly corpus.  The
// producer has 61 younger VMEM events at vmcnt(62), so that wait may leave it
// pending. Eight further ordered loads force its retirement at admission.
TEST(WaitcheckTest, Gfx1150NewLoadsRetireOldResultAfterMaximumPartialWait) {
  std::vector<uint32_t> program;
  append_gfx1150_global_load_b32(program, 32);
  for (uint32_t i = 0; i < 61; ++i)
    append_gfx1150_global_load_b32(program, static_cast<uint8_t>(64 + i));
  program.push_back(0xBF89FBF7u); // s_waitcnt vmcnt(62)
  for (uint32_t i = 0; i < 8; ++i)
    append_gfx1150_global_load_b32(program, static_cast<uint8_t>(160 + i));
  append_inst(program, v_mov_b32(200, 32));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA3_5);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, AcceptsStorecntOverflowSizedQueueBeforeProgramEnd) {
  std::vector<uint32_t> program;
  append_global_stores(program, kOverflowQueueSize, kOverflowBaseVgpr);
  append_inst(program, s_endpgm());
  WaitcheckOptions options;
  options.max_diagnostics = 1;

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4, options);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics_observed, 0u);
  EXPECT_FALSE(report.diagnostics_truncated);
}

TEST(WaitcheckTest, ReportsKmcntZeroForOverflowSizedSmemQueue) {
  std::vector<uint32_t> program;
  append_s_loads(program, kOverflowQueueSize, kOverflowKmcntBaseSgpr, kOverflowKmcntSbase);
  append_inst(program, s_mov_b32(kOverflowKmcntConsumerSgpr, kOverflowKmcntBaseSgpr));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported);
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Km);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Use);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::SGPR);
  EXPECT_EQ(report.diagnostics[0].reg.index, kOverflowKmcntBaseSgpr);
  EXPECT_EQ(report.diagnostics[0].required_count, 0u);
  EXPECT_NE(report.diagnostics[0].message.find("s_wait_kmcnt <= 0"), std::string::npos);
}

TEST(WaitcheckTest, NonzeroKmcntDoesNotRetireOldestSmemResult) {
  std::vector<uint32_t> program;
  append_s_loads(program, kOverflowQueueSize, kOverflowKmcntBaseSgpr, kOverflowKmcntSbase);
  append_inst(program, sopp(71, kOverflowRequiredCount)); // s_wait_kmcnt 39
  append_inst(program, s_mov_b32(kOverflowKmcntConsumerSgpr, kOverflowKmcntBaseSgpr));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported);
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Km);
  EXPECT_EQ(report.diagnostics[0].required_count, 0u);
}

TEST(WaitcheckTest, KmcntZeroRetiresOverflowSizedSmemQueue) {
  std::vector<uint32_t> program;
  append_s_loads(program, kOverflowQueueSize, kOverflowKmcntBaseSgpr, kOverflowKmcntSbase);
  append_inst(program, sopp(71, 0)); // s_wait_kmcnt 0
  append_inst(program, s_mov_b32(kOverflowKmcntConsumerSgpr, kOverflowKmcntBaseSgpr));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, ReportsSamplecntMaxCountForOverflowSizedQueue) {
  std::vector<uint32_t> program;
  append_image_samples(program, kOverflowQueueSize, kOverflowBaseVgpr);
  append_inst(program, v_mov_b32(kOverflowConsumerVgpr, kOverflowBaseVgpr));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  expect_single_overflow_diagnostic(report, WaitCounterKind::Sample, WaitcheckAccessKind::Use,
                                    RegClass::VGPR, kOverflowBaseVgpr);
}

TEST(WaitcheckTest, AcceptsSamplecntMaxCountForOldestOverflowSizedQueue) {
  std::vector<uint32_t> program;
  append_image_samples(program, kOverflowQueueSize, kOverflowBaseVgpr);
  append_inst(program, sopp(66, kOverflowRequiredCount)); // s_wait_samplecnt 39
  append_inst(program, v_mov_b32(kOverflowConsumerVgpr, kOverflowBaseVgpr));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, BvhcntAdmissionReleasesOldestResult) {
  std::vector<uint32_t> program;
  append_image_bvhs(program, kOverflowQueueSize, kOverflowBaseVgpr);
  append_inst(program, v_mov_b32(196, kOverflowBaseVgpr));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, AcceptsBvhcntMaxCountForOldestOverflowSizedQueue) {
  std::vector<uint32_t> program;
  append_image_bvhs(program, kOverflowQueueSize, kOverflowBaseVgpr);
  append_inst(program, sopp(67, 6)); // Largest non-sentinel 3-bit wait value.
  append_inst(program, v_mov_b32(196, kOverflowBaseVgpr));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, ExpcntAdmissionReleasesOldestResult) {
  std::vector<uint32_t> program;
  append_ds_direct_loads(program, kOverflowQueueSize, kOverflowBaseVgpr);
  append_inst(program, v_mov_b32(kOverflowConsumerVgpr, kOverflowBaseVgpr));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, AcceptsExpcntMaxCountForOldestOverflowSizedQueue) {
  std::vector<uint32_t> program;
  append_ds_direct_loads(program, kOverflowQueueSize, kOverflowBaseVgpr);
  append_inst(program, sopp(68, 6)); // Largest non-sentinel 3-bit wait value.
  append_inst(program, v_mov_b32(kOverflowConsumerVgpr, kOverflowBaseVgpr));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, ReportsMissingWaitAluVmVsrcBeforeDsSourceOverwrite) {
  std::vector<uint32_t> program;
  append_inst(program, ds_load_b32(0, 4));
  append_inst(program, v_mov_b32(4, 10));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported);
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::VmVsrc);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Def);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::VGPR);
  EXPECT_EQ(report.diagnostics[0].reg.index, 4u);
}

TEST(WaitcheckTest, AcceptsWaitAluVmVsrcBeforeDsSourceOverwrite) {
  std::vector<uint32_t> program;
  append_inst(program, ds_load_b32(0, 4));
  append_inst(program, s_wait_alu_vm_vsrc_0());
  append_inst(program, v_mov_b32(4, 10));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, DscntZeroAlsoClearsDsSourceRead) {
  std::vector<uint32_t> program;
  append_inst(program, ds_load_b32(0, 4));
  append_inst(program, sopp(70, 0)); // s_wait_dscnt 0
  append_inst(program, v_mov_b32(4, 10));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

// Internal ordering ages may exceed VM_VSRC's 3-bit wait field, but a reported
// dependency must remain encodable.
TEST(WaitcheckTest, Gfx1250ClampsVmVsrcDiagnosticPastDependencyWaitLimit) {
  constexpr uint32_t kHwRegWaveSchedMode = 26;
  std::vector<uint32_t> program;
  append_inst(program, s_setreg_imm32_b32(hwreg(kHwRegWaveSchedMode, 0, 2), 2));
  for (uint32_t i = 0; i < 8; ++i)
    append_gfx1250_ds_bpermute_b32(program, 10 + i, 1, 100 + i);
  append_inst(program, v_mov_b32(100, 0));

  TestCodeObject code_object(program);
  auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_CDNA5);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::VmVsrc);
  EXPECT_EQ(report.diagnostics[0].required_count, 6u);
}

// depctr_vm_vsrc(7) is the no-wait sentinel. Other s_wait_alu fields must not
// make it retire an old VM_VSRC hazard whose internal ordering age exceeds 6.
TEST(WaitcheckTest, Gfx1250WaitAluNoWaitPreservesVmVsrcPastDependencyWaitLimit) {
  constexpr uint32_t kHwRegWaveSchedMode = 26;
  std::vector<uint32_t> program;
  append_inst(program, s_setreg_imm32_b32(hwreg(kHwRegWaveSchedMode, 0, 2), 2));
  for (uint32_t i = 0; i < 8; ++i)
    append_gfx1250_ds_bpermute_b32(program, 10 + i, 1, 100 + i);
  append_inst(program, s_wait_alu_sa_sdst_0());
  append_inst(program, v_mov_b32(100, 0));

  TestCodeObject code_object(program);
  auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_CDNA5);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::VmVsrc);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Def);
  EXPECT_EQ(report.diagnostics[0].reg, (RegisterRef{RegClass::VGPR, 100, 1}));
  EXPECT_EQ(report.diagnostics[0].required_count, 6u);
}

TEST(WaitcheckTest, Gfx1250MaximumWaitAluRetiresOldVmVsrcPastDependencyWaitLimit) {
  constexpr uint32_t kHwRegWaveSchedMode = 26;
  std::vector<uint32_t> program;
  append_inst(program, s_setreg_imm32_b32(hwreg(kHwRegWaveSchedMode, 0, 2), 2));
  for (uint32_t i = 0; i < 8; ++i)
    append_gfx1250_ds_bpermute_b32(program, 10 + i, 1, 100 + i);
  append_inst(program, s_wait_alu_vm_vsrc(6));
  append_inst(program, v_mov_b32(100, 0));
  append_inst(program, v_mov_b32(102, 0));

  TestCodeObject code_object(program);
  auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_CDNA5);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::VmVsrc);
  EXPECT_EQ(report.diagnostics[0].reg, (RegisterRef{RegClass::VGPR, 102, 1}));
  EXPECT_EQ(report.diagnostics[0].required_count, 5u);
}

// Reduced from the gfx1250 top-k/top-p kernel attached to #11864. VM_VSRC has
// only a 3-bit wait field, but DS_CNT can distinguish more than seven younger
// source reads. Saturating internal VM_VSRC ages at depctr_vm_vsrc(6) makes the
// first wait spuriously lose the ordering needed by the second wait.
TEST(WaitcheckTest, Gfx1250DscntPartialWaitsRetireVmVsrcPastDependencyWaitLimit) {
  constexpr uint32_t kHwRegWaveSchedMode = 26;
  std::vector<uint32_t> program;
  append_inst(program, s_setreg_imm32_b32(hwreg(kHwRegWaveSchedMode, 0, 2), 2));
  for (uint32_t i = 0; i < 8; ++i)
    append_gfx1250_ds_bpermute_b32(program, 10 + i, 1, 100 + i);
  append_inst(program, sopp(70, 7)); // s_wait_dscnt 7 retires the oldest operation.
  append_inst(program, sopp(70, 1)); // The source read of v106 is now complete.
  append_inst(program, v_mov_b32(106, 0));
  append_inst(program, v_mov_b32(107, 0));

  TestCodeObject code_object(program);
  auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_CDNA5);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::VmVsrc);
  EXPECT_EQ(report.diagnostics[0].reg, (RegisterRef{RegClass::VGPR, 107, 1}));
  EXPECT_EQ(report.diagnostics[0].required_count, 0u);
}

TEST(WaitcheckTest, Gfx1250DscntMaximumPartialWaitRetiresSaturatedVmVsrcAges) {
  constexpr uint32_t kHwRegWaveSchedMode = 26;
  std::vector<uint32_t> program;
  append_inst(program, s_setreg_imm32_b32(hwreg(kHwRegWaveSchedMode, 0, 2), 2));
  for (uint32_t i = 0; i < 64; ++i)
    append_gfx1250_ds_bpermute_b32(program, 10 + i, 1, 100 + i);
  append_inst(program, sopp(70, 62)); // Leaves the 62 youngest DS operations pending.
  append_inst(program, v_mov_b32(100, 0));
  append_inst(program, v_mov_b32(102, 0));

  TestCodeObject code_object(program);
  auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_CDNA5);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::VmVsrc);
  EXPECT_EQ(report.diagnostics[0].reg, (RegisterRef{RegClass::VGPR, 102, 1}));
  EXPECT_EQ(report.diagnostics[0].required_count, 6u);
}

TEST(WaitcheckTest, Gfx1250DscntNoWaitPreservesTheOldestStillAdmittedVmVsrc) {
  constexpr uint32_t kHwRegWaveSchedMode = 26;
  std::vector<uint32_t> program;
  append_inst(program, s_setreg_imm32_b32(hwreg(kHwRegWaveSchedMode, 0, 2), 2));
  for (uint32_t i = 0; i < 64; ++i)
    append_gfx1250_ds_bpermute_b32(program, 10 + i, 1, 100 + i);
  append_inst(program, sopp(70, 63)); // DSCNT's all-ones no-wait sentinel.
  append_inst(program, v_mov_b32(101, 0));

  TestCodeObject code_object(program);
  auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_CDNA5);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::VmVsrc);
  EXPECT_EQ(report.diagnostics[0].reg, (RegisterRef{RegClass::VGPR, 101, 1}));
  EXPECT_EQ(report.diagnostics[0].required_count, 6u);
}

// Interleaved VMEM requests do not count as younger LDS source reads.
// The DSCNT wait releases the first DS, and one younger DS still orders the second.
TEST(WaitcheckTest, Gfx1250DscntUsesPrimaryOrderPastInterleavedVmVsrcSaturation) {
  constexpr uint32_t kHwRegWaveSchedMode = 26;
  std::vector<uint32_t> program;
  append_inst(program, s_setreg_imm32_b32(hwreg(kHwRegWaveSchedMode, 0, 2), 2));
  append_gfx1250_ds_bpermute_b32(program, 10, 1, 100);
  append_gfx1250_ds_bpermute_b32(program, 11, 1, 101);
  append_global_loads(program, 63, 20);
  append_gfx1250_ds_bpermute_b32(program, 12, 1, 102);
  append_inst(program, sopp(70, 2)); // Leaves the two youngest DS operations pending.
  append_inst(program, v_mov_b32(100, 0));
  append_inst(program, v_mov_b32(101, 0));

  TestCodeObject code_object(program);
  auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_CDNA5);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::VmVsrc);
  EXPECT_EQ(report.diagnostics[0].reg, (RegisterRef{RegClass::VGPR, 101, 1}));
  EXPECT_EQ(report.diagnostics[0].required_count, 1u);
}

TEST(WaitcheckTest, VopdMovDoesNotReadUnusedVsrc1Encoding) {
  std::vector<uint32_t> program;
  append_inst(program, ds_load_b32(0, 10));
  append_v_dual_lshlrev_b32_v17_2_v9_dual_mov_b32_v9_s11(program);

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA5);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, BufferOffenUsesSingleVaddrRegister) {
  std::vector<uint32_t> program;
  append_inst(program, ds_load_b32(8, 10));
  append_buffer_load_b128_v32_v7_s4_offen(program);

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA5);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, ReportsMissingWaitAluVmVsrcBeforeDsStoreDataOverwrite) {
  std::vector<uint32_t> program;
  append_inst(program, ds_store_b32(4, 0));
  append_inst(program, v_mov_b32(0, 1));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported);
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::VmVsrc);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Def);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::VGPR);
  EXPECT_EQ(report.diagnostics[0].reg.index, 0u);
}

TEST(WaitcheckTest, ReportsMissingWaitAluVmVsrcBeforeDsStoreAddressOverwrite) {
  std::vector<uint32_t> program;
  append_inst(program, ds_store_b32(4, 0));
  append_inst(program, v_mov_b32(4, 1));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported);
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::VmVsrc);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Def);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::VGPR);
  EXPECT_EQ(report.diagnostics[0].reg.index, 4u);
}

TEST(WaitcheckTest, AcceptsWaitAluVmVsrcBeforeDsStoreSourceOverwrite) {
  std::vector<uint32_t> program;
  append_inst(program, ds_store_b32(4, 0));
  append_inst(program, s_wait_alu_vm_vsrc_0());
  append_inst(program, v_mov_b32(0, 1));
  append_inst(program, v_mov_b32(4, 1));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, ReportsMissingKmcntBeforeScalarLoadUse) {
  std::vector<uint32_t> program;
  append_inst(program, s_load_b32(4, 0));
  append_inst(program, s_mov_b32(8, 4));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported);
  ASSERT_EQ(report.diagnostics.size(), 1u);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Km);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Use);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::SGPR);
  EXPECT_EQ(report.diagnostics[0].reg.index, 4u);
}

TEST(WaitcheckTest, AcceptsKmcntZeroBeforeScalarLoadUse) {
  std::vector<uint32_t> program;
  append_inst(program, s_load_b32(4, 0));
  append_inst(program, sopp(71, 0)); // s_wait_kmcnt 0
  append_inst(program, s_mov_b32(8, 4));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty());
}

TEST(WaitcheckTest, Gfx1201CndmaskDoesNotReadAdjacentMaskSgpr) {
  std::vector<uint32_t> program;
  append_s_load_b64_s2_s16_imm196(program);
  append_v_cndmask_b32_e64_v6_0_1_s(program, 1);

  WaitcheckOptions options;
  options.fallback_wavefront_size = 32;
  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4, options);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx1201CndmaskReportsPendingMaskSgpr) {
  std::vector<uint32_t> program;
  append_s_load_b64_s2_s16_imm196(program);
  append_v_cndmask_b32_e64_v6_0_1_s(program, 2);

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Km);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Use);
  EXPECT_EQ(report.diagnostics[0].reg, (RegisterRef{RegClass::SGPR, 2, 1}));
}

TEST(WaitcheckTest, Gfx1201Vop3CarryOutDoesNotDefineAdjacentSgpr) {
  std::vector<uint32_t> program;
  append_inst(program, s_load_b32(1, 16));
  append_v_sub_co_u32_v0_s0_v0_v8(program);

  WaitcheckOptions options;
  options.fallback_wavefront_size = 32;
  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4, options);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx1201ReportsPendingVop3CarryOutSgpr) {
  std::vector<uint32_t> program;
  append_inst(program, s_load_b32(0, 16));
  append_v_sub_co_u32_v0_s0_v0_v8(program);

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Km);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Def);
  EXPECT_EQ(report.diagnostics[0].reg, (RegisterRef{RegClass::SGPR, 0, 1}));
}

TEST(WaitcheckTest, Gfx1201DivScaleDoesNotDefineAdjacentMaskSgpr) {
  std::vector<uint32_t> program;
  append_inst(program, s_load_b32(3, 16));
  append_v_div_scale_f32_v5_s2_s8_s9_s8(program);

  WaitcheckOptions options;
  options.fallback_wavefront_size = 32;
  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4, options);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx1201ReportsPendingDivScaleMaskSgpr) {
  std::vector<uint32_t> program;
  append_inst(program, s_load_b32(2, 16));
  append_v_div_scale_f32_v5_s2_s8_s9_s8(program);

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Km);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Def);
  EXPECT_EQ(report.diagnostics[0].reg, (RegisterRef{RegClass::SGPR, 2, 1}));
}

TEST(WaitcheckTest, Gfx1201Wave32CodeObjectDivScaleDoesNotDefineAdjacentMaskSgpr) {
  std::vector<uint32_t> program;
  append_inst(program, s_load_b32(3, 16));
  append_v_div_scale_f32_v5_s2_s8_s9_s8(program);

  const auto image =
      rocjitsu::waitcheck_test::make_gfx_code_object(program, EF_AMDGPU_MACH_AMDGCN_GFX1201, true);
  AmdGpuCodeObject code_object(image.data(), image.size());
  const auto kernels = waitcheck_kernels(code_object);
  auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_EQ(kernels.size(), 1u);
  EXPECT_EQ(kernels[0].wavefront_size, 32u);
  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx1201Wave64DivScaleDefinesBothMaskSgprs) {
  std::vector<uint32_t> program;
  append_inst(program, s_load_b32(3, 16));
  append_v_div_scale_f32_v5_s2_s8_s9_s8(program);

  const auto image =
      rocjitsu::waitcheck_test::make_gfx_code_object(program, EF_AMDGPU_MACH_AMDGCN_GFX1201);
  AmdGpuCodeObject code_object(image.data(), image.size());
  auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Km);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Def);
  EXPECT_EQ(report.diagnostics[0].reg, (RegisterRef{RegClass::SGPR, 3, 1}));
}

TEST(WaitcheckTest, RawStreamWaveSizeIsExplicitAndDefaultsToWave64) {
  EXPECT_EQ(waitcheck_default_wavefront_size(ROCJITSU_CODE_ARCH_RDNA4), 32u);
  EXPECT_EQ(waitcheck_default_wavefront_size(ROCJITSU_CODE_ARCH_CDNA5), 32u);
  EXPECT_EQ(waitcheck_default_wavefront_size(ROCJITSU_CODE_ARCH_CDNA4), 64u);
  std::vector<uint32_t> program;
  append_inst(program, s_load_b32(3, 16));
  append_v_div_scale_f32_v5_s2_s8_s9_s8(program);
  const auto wave64 = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);
  WaitcheckOptions wave32_options;
  wave32_options.fallback_wavefront_size = 32;
  const auto wave32 = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4, wave32_options);
  WaitcheckOptions invalid_options;
  invalid_options.fallback_wavefront_size = 16;
  const auto invalid = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4, invalid_options);
  ASSERT_TRUE(wave64.supported && wave32.supported);
  EXPECT_EQ(wave64.diagnostics_observed, 1u) << diagnostic_summary(wave64);
  EXPECT_EQ(wave32.diagnostics_observed, 0u) << diagnostic_summary(wave32);
  EXPECT_FALSE(invalid.supported);
  EXPECT_NE(invalid.analysis_error.find("wavefront size"), std::string::npos);
}

TEST(WaitcheckTest, SymbolLessWave32FallbackGuidesIndirectCfgRecovery) {
  constexpr auto arch = ROCJITSU_CODE_ARCH_RDNA4;
  std::vector<uint32_t> program;
  program.push_back(build_s_getpc_b64(4, arch).value());
  program.push_back(build_s_add_u32(4, 4, 255, arch).value());
  const size_t delta_word = program.size();
  program.push_back(0);
  program.push_back(build_s_addc_u32(5, 5, 128, arch).value());
  append_inst(program, s_wait_alu_sa_sdst_0());
  append_v_cmp_gt_u32_sdst_s5_v12(program, 3); // Wave64 also clobbers target s4.
  append_inst(program, s_wait_alu_va_sdst_0());
  program.push_back(build_s_setpc_b64(4, arch).value());
  program.push_back(build_s_endpgm(arch).value());
  const uint64_t target = program.size() * sizeof(uint32_t);
  program[delta_word] = static_cast<uint32_t>(target - sizeof(uint32_t));
  program.push_back(build_s_endpgm(arch).value());
  const auto image = waitcheck_test::make_gfx_symbol_less_code_object(
      program, EF_AMDGPU_MACH_AMDGCN_GFX1201, /*wave32=*/true);
  const AmdGpuCodeObject object(image.data(), image.size());
  ASSERT_TRUE(object.is_valid());
  ASSERT_TRUE(waitcheck_kernels(object).empty());
  const auto automatic = analyze_waitcnts(object, arch);
  WaitcheckOptions wave32;
  wave32.fallback_wavefront_size = 32;
  const auto explicit32 = analyze_waitcnts(object, arch, wave32);
  ASSERT_TRUE(automatic.supported && explicit32.supported);
  EXPECT_TRUE(automatic.complete) << automatic.analysis_error;
  EXPECT_TRUE(explicit32.complete) << explicit32.analysis_error;
  EXPECT_TRUE(automatic.diagnostics.empty()) << diagnostic_summary(automatic);
  EXPECT_TRUE(explicit32.diagnostics.empty()) << diagnostic_summary(explicit32);
}

TEST(WaitcheckTest, SymbolLessRdna4KeepsTargetDerivedWave32Default) {
  std::vector<uint32_t> program;
  append_inst(program, s_load_b32(3, 16));
  append_v_div_scale_f32_v5_s2_s8_s9_s8(program);
  append_inst(program, sopp(71, 0)); // s_wait_kmcnt 0
  append_inst(program, s_endpgm());
  const auto image = waitcheck_test::make_gfx_symbol_less_code_object(
      program, EF_AMDGPU_MACH_AMDGCN_GFX1201, /*wave32=*/true);
  const AmdGpuCodeObject object(image.data(), image.size());
  ASSERT_TRUE(object.is_valid());
  ASSERT_TRUE(waitcheck_kernels(object).empty());
  const auto automatic = analyze_waitcnts(object, ROCJITSU_CODE_ARCH_RDNA4);
  WaitcheckOptions wave64;
  wave64.fallback_wavefront_size = 64;
  const auto explicit64 = analyze_waitcnts(object, ROCJITSU_CODE_ARCH_RDNA4, wave64);
  ASSERT_TRUE(automatic.supported && explicit64.supported);
  EXPECT_TRUE(automatic.diagnostics.empty()) << diagnostic_summary(automatic);
  EXPECT_EQ(explicit64.diagnostics_observed, 1u) << diagnostic_summary(explicit64);
}

TEST(WaitcheckTest, Gfx1250Wave32VopCompareDoesNotDefineAdjacentMaskSgpr) {
  std::vector<uint32_t> program;
  append_inst(program, s_load_b32(2, 16));
  append_v_cmp_gt_u32_sdst_s5_v12(program, 1);

  const auto image =
      rocjitsu::waitcheck_test::make_gfx_code_object(program, EF_AMDGPU_MACH_AMDGCN_GFX1250, true);
  AmdGpuCodeObject code_object(image.data(), image.size());
  auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_CDNA5);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx1250Wave64DescriptorPreservesFixedCompareWidth) {
  std::vector<uint32_t> program;
  append_inst(program, s_load_b32(2, 16));
  append_v_cmp_gt_u32_sdst_s5_v12(program, 1);

  const auto image =
      rocjitsu::waitcheck_test::make_gfx_code_object(program, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject code_object(image.data(), image.size());
  auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_CDNA5);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx1250Wave32VopdCndmaskDoesNotUseAdjacentMaskSgpr) {
  std::vector<uint32_t> program;
  append_inst(program, s_load_b32(1, 16));
  append_v_dual_add_nc_u32_v11_1_v10_cndmask_v8_v8_v12_s0(program);

  const auto image =
      rocjitsu::waitcheck_test::make_gfx_code_object(program, EF_AMDGPU_MACH_AMDGCN_GFX1250, true);
  AmdGpuCodeObject code_object(image.data(), image.size());
  auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_CDNA5);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx1250LegacyDescriptorVopdCndmaskUsesWave32Mask) {
  std::vector<uint32_t> program;
  append_inst(program, s_load_b32(1, 16));
  append_v_dual_add_nc_u32_v11_1_v10_cndmask_v8_v8_v12_s0(program);

  const auto image =
      rocjitsu::waitcheck_test::make_gfx_code_object(program, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject code_object(image.data(), image.size());
  auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_CDNA5);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx1250Wave32Vop3CndmaskDoesNotUseAdjacentMaskSgpr) {
  std::vector<uint32_t> program;
  append_inst(program, s_load_b32(1, 16));
  append_v_cndmask_b32_v8_v8_v12_s0(program);

  const auto image =
      rocjitsu::waitcheck_test::make_gfx_code_object(program, EF_AMDGPU_MACH_AMDGCN_GFX1250, true);
  AmdGpuCodeObject code_object(image.data(), image.size());
  auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_CDNA5);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx1250LegacyDescriptorVop3CndmaskUsesWave32Mask) {
  std::vector<uint32_t> program;
  append_inst(program, s_load_b32(1, 16));
  append_v_cndmask_b32_v8_v8_v12_s0(program);

  const auto image =
      rocjitsu::waitcheck_test::make_gfx_code_object(program, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject code_object(image.data(), image.size());
  auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_CDNA5);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx1250Wave32Vop3CndmaskB16DoesNotUseAdjacentMaskSgpr) {
  std::vector<uint32_t> program;
  append_inst(program, s_load_b32(3, 16));
  append_v_cndmask_b16_v1_v1_v14_s2(program);

  const auto image =
      rocjitsu::waitcheck_test::make_gfx_code_object(program, EF_AMDGPU_MACH_AMDGCN_GFX1250, true);
  AmdGpuCodeObject code_object(image.data(), image.size());
  auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_CDNA5);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx1250LegacyDescriptorVop3CndmaskB16UsesWave32Mask) {
  std::vector<uint32_t> program;
  append_inst(program, s_load_b32(3, 16));
  append_v_cndmask_b16_v1_v1_v14_s2(program);

  const auto image =
      rocjitsu::waitcheck_test::make_gfx_code_object(program, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject code_object(image.data(), image.size());
  auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_CDNA5);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx1250Vop3True16DestinationUsesFullEightBitVgprIndex) {
  std::vector<uint32_t> program;
  append_global_load_b128_v6_v137_s2(program);
  append_v_and_b16_v134_ff_v134(program);

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA5);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx1201Vop3CarryInDoesNotReadAdjacentSgpr) {
  std::vector<uint32_t> program;
  append_inst(program, s_load_b32(1, 16));
  append_v_sub_co_ci_u32_v1_null_v1_v12_s0(program);

  WaitcheckOptions options;
  options.fallback_wavefront_size = 32;
  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4, options);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx1201ReportsPendingVop3CarryInSgpr) {
  std::vector<uint32_t> program;
  append_inst(program, s_load_b32(0, 16));
  append_v_sub_co_ci_u32_v1_null_v1_v12_s0(program);

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Km);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Use);
  EXPECT_EQ(report.diagnostics[0].reg, (RegisterRef{RegClass::SGPR, 0, 1}));
}

TEST(WaitcheckTest, Gfx950ReportsMissingLgkmcntBeforeScalarLoadUse) {
  std::vector<uint32_t> program;
  append_gfx950_s_load_dword_s4_s0(program);
  append_gfx950_s_mov_b32_s8_s4(program);

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA4);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Ds);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Use);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::SGPR);
  EXPECT_EQ(report.diagnostics[0].reg.index, 4u);
  EXPECT_EQ(report.diagnostics[0].required_count, 0u);
  EXPECT_NE(report.diagnostics[0].message.find("s_waitcnt lgkmcnt(0)"), std::string::npos);
}

TEST(WaitcheckTest, Gfx950AcceptsSWaitcntLgkmcntZeroBeforeScalarLoadUse) {
  std::vector<uint32_t> program;
  append_gfx950_s_load_dword_s4_s0(program);
  append_gfx950_s_waitcnt_lgkmcnt_0(program);
  append_gfx950_s_mov_b32_s8_s4(program);

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA4);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx950ReportsMissingLgkmcntBeforeDsReadUse) {
  std::vector<uint32_t> program;
  append_gfx950_ds_read_b32_v0_v4(program);
  append_gfx950_v_mov_b32_v1_v0(program);

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA4);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Ds);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Use);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::VGPR);
  EXPECT_EQ(report.diagnostics[0].reg.index, 0u);
  EXPECT_EQ(report.diagnostics[0].required_count, 0u);
  EXPECT_NE(report.diagnostics[0].message.find("s_waitcnt lgkmcnt(0)"), std::string::npos);
}

TEST(WaitcheckTest, Gfx950AcceptsSWaitcntLgkmcntZeroBeforeDsReadUse) {
  std::vector<uint32_t> program;
  append_gfx950_ds_read_b32_v0_v4(program);
  append_gfx950_s_waitcnt_lgkmcnt_0(program);
  append_gfx950_v_mov_b32_v1_v0(program);

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA4);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx950FmoeDsOverwritesRequireOrderedLgkmWaits) {
  for (bool insert_waits : {false, true}) {
    SCOPED_TRACE(insert_waits);
    // Encodings from fMoE, with LLVM SIInsertWaitcnts' lgkmcnt(1)/lgkmcnt(0)
    // repair as the control. These are physical writes, even without a read
    // of either DS result before its destination is overwritten.
    std::vector<uint32_t> program{
        0xd9fe1140u,
        0x60000006u, // ds_read_b128 v[96:99], v6 offset:4416
        0xd9fe1180u,
        0x64000006u, // ds_read_b128 v[100:103], v6 offset:4480
    };
    if (insert_waits)
      append_gfx950_s_waitcnt_lgkmcnt_1(program);
    program.insert(program.end(), {0xd1340060u, 0x00007902u}); // v_add_u32 v96, v2, s60
    if (insert_waits)
      append_gfx950_s_waitcnt_lgkmcnt_0(program);
    program.insert(program.end(), {0xd1340064u, 0x00007902u}); // v_add_u32 v100, v2, s60

    auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA4);
    ASSERT_TRUE(report.supported) << report.analysis_error;
    if (insert_waits) {
      EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
    } else {
      ASSERT_EQ(report.diagnostics.size(), 2u) << diagnostic_summary(report);
      for (size_t i = 0; i < report.diagnostics.size(); ++i) {
        const auto &diagnostic = report.diagnostics[i];
        EXPECT_EQ(diagnostic.counter, WaitCounterKind::Ds);
        EXPECT_EQ(diagnostic.access, WaitcheckAccessKind::Def);
        EXPECT_EQ(diagnostic.reg,
                  (RegisterRef{RegClass::VGPR, static_cast<uint16_t>(96 + 4 * i), 1}));
        EXPECT_EQ(diagnostic.producer_section_offset, 8 * i);
        EXPECT_EQ(diagnostic.section_offset, 16 + 8 * i);
        EXPECT_EQ(diagnostic.required_count, 1 - i);
      }
    }
  }
}

TEST(WaitcheckTest, Gfx950ReportsReadAndWriteWhileDsReplacementIsPending) {
  std::vector<uint32_t> program;
  append_gfx950_v_mov_b32_v0_v2(program);   // Initialize v0.
  append_gfx950_ds_read_b32_v0_v4(program); // Start an asynchronous replacement.
  append_gfx950_v_mov_b32_v1_v0(program);   // Cannot assume the old v0 value.
  append_gfx950_v_mov_b32_v0_v2(program);   // Cannot overwrite a pending physical destination.
  append_gfx950_s_waitcnt_lgkmcnt_0(program);
  append_gfx950_v_mov_b32_v1_v0(program); // The load has completed.

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA4);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 2u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Ds);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Use);
  EXPECT_EQ(report.diagnostics[0].reg, (RegisterRef{RegClass::VGPR, 0, 1}));
  EXPECT_EQ(report.diagnostics[0].producer_section_offset, 4u);
  EXPECT_EQ(report.diagnostics[0].section_offset, 12u);
  EXPECT_EQ(report.diagnostics[0].required_count, 0u);
  EXPECT_EQ(report.diagnostics[1].counter, WaitCounterKind::Ds);
  EXPECT_EQ(report.diagnostics[1].access, WaitcheckAccessKind::Def);
  EXPECT_EQ(report.diagnostics[1].reg, (RegisterRef{RegClass::VGPR, 0, 1}));
  EXPECT_EQ(report.diagnostics[1].producer_section_offset, 4u);
  EXPECT_EQ(report.diagnostics[1].section_offset, 16u);
  EXPECT_EQ(report.diagnostics[1].required_count, 0u);
}

TEST(WaitcheckTest, Gfx950ReportsLiveInVgprUseWhileReplacementIsPending) {
  std::vector<uint32_t> program;
  append_gfx950_v_mov_b32_v1_v0(program);   // Read the ABI/live-in v0 value.
  append_gfx950_ds_read_b32_v0_v4(program); // Start an asynchronous replacement.
  append_gfx950_v_mov_b32_v1_v0(program);   // Cannot assume the old v0 value.
  append_gfx950_s_waitcnt_lgkmcnt_0(program);
  append_gfx950_v_mov_b32_v1_v0(program); // The load has completed.

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA4);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Ds);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Use);
  EXPECT_EQ(report.diagnostics[0].reg, (RegisterRef{RegClass::VGPR, 0, 1}));
  EXPECT_EQ(report.diagnostics[0].producer_section_offset, 4u);
  EXPECT_EQ(report.diagnostics[0].section_offset, 12u);
  EXPECT_EQ(report.diagnostics[0].required_count, 0u);
}

TEST(WaitcheckTest, Gfx950ReportsSynchronousOverwriteAndReadAfterDsRead) {
  std::vector<uint32_t> program;
  append_gfx950_ds_read_b32_v0_v4(program); // Issue an asynchronous load.
  append_gfx950_v_mov_b32_v0_v2(program);   // Overwrite its pending destination.
  append_gfx950_v_mov_b32_v1_v0(program);   // The overwrite does not retire the load.
  append_gfx950_s_waitcnt_lgkmcnt_0(program);
  append_gfx950_v_mov_b32_v1_v0(program); // The load has completed.

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA4);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 2u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Ds);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Def);
  EXPECT_EQ(report.diagnostics[0].reg, (RegisterRef{RegClass::VGPR, 0, 1}));
  EXPECT_EQ(report.diagnostics[0].producer_section_offset, 0u);
  EXPECT_EQ(report.diagnostics[0].section_offset, 8u);
  EXPECT_EQ(report.diagnostics[0].required_count, 0u);
  EXPECT_EQ(report.diagnostics[1].counter, WaitCounterKind::Ds);
  EXPECT_EQ(report.diagnostics[1].access, WaitcheckAccessKind::Use);
  EXPECT_EQ(report.diagnostics[1].reg, (RegisterRef{RegClass::VGPR, 0, 1}));
  EXPECT_EQ(report.diagnostics[1].producer_section_offset, 0u);
  EXPECT_EQ(report.diagnostics[1].section_offset, 12u);
  EXPECT_EQ(report.diagnostics[1].required_count, 0u);
}

TEST(WaitcheckTest, Gfx950ReportsPendingVgprUseWithOtherInitializedRegisters) {
  std::vector<uint32_t> program;
  append_gfx950_v_mov_b32_v0_v2(program); // An unrelated initialized register.
  append_gfx950_buffer_load_dword_v1_v8_s0_offen(program);
  append_gfx950_v_mov_b32_v8_v10(program); // Another unrelated definition.
  program.push_back(0x7E040301u);          // v_mov_b32_e32 v2, v1.

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA4);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Load);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Use);
  EXPECT_EQ(report.diagnostics[0].reg, (RegisterRef{RegClass::VGPR, 1, 1}));
}

TEST(WaitcheckTest, ReportsKmcntBeforeScalarLoadOverwrite) {
  std::vector<uint32_t> program;
  append_inst(program, s_load_b32(4, 0));
  append_inst(program, s_load_b32(4, 0));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported);
  ASSERT_EQ(report.diagnostics.size(), 1u);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Km);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Def);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::SGPR);
  EXPECT_EQ(report.diagnostics[0].reg.index, 4u);
  EXPECT_EQ(report.diagnostics[0].required_count, 0u);
}

TEST(WaitcheckTest, Gfx1250SBufferLoadSbaseUsesPhysicalDescriptorRegister) {
  std::vector<uint32_t> program;
  append_s_buffer_load_b128_s8_s12_imm48(program);
  append_s_buffer_load_b64_s20_s12_imm64(program);

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA5);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx1201SBufferLoadSbaseUsesPhysicalDescriptorRegister) {
  std::vector<uint32_t> program;
  append_s_buffer_load_b128_s16_s8_imm32(program);
  append_s_buffer_load_b128_s0_s8_imm0(program);

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx1201ReportsKmcntBeforeSBufferDescriptorUse) {
  std::vector<uint32_t> program;
  append_s_buffer_load_b128_s8_s12_imm48(program);
  append_s_buffer_load_b128_s0_s8_imm0(program);

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Km);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Use);
  EXPECT_EQ(report.diagnostics[0].reg, (RegisterRef{RegClass::SGPR, 8, 1}));
}

TEST(WaitcheckTest, AcceptsKmcntBeforeScalarLoadOverwrite) {
  std::vector<uint32_t> program;
  append_inst(program, s_load_b32(4, 0));
  append_inst(program, sopp(71, 0)); // s_wait_kmcnt 0
  append_inst(program, s_load_b32(4, 0));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty());
}

TEST(WaitcheckTest, ReportsKmcntBeforeSendmsgRtnB32Use) {
  std::vector<uint32_t> program;
  append_inst(program, s_sendmsg_rtn_b32(4));
  append_inst(program, s_mov_b32(8, 4));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported);
  ASSERT_EQ(report.diagnostics.size(), 1u);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Km);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Use);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::SGPR);
  EXPECT_EQ(report.diagnostics[0].reg.index, 4u);
  EXPECT_EQ(report.diagnostics[0].required_count, 0u);
}

TEST(WaitcheckTest, AcceptsKmcntBeforeSendmsgRtnB32Use) {
  std::vector<uint32_t> program;
  append_inst(program, s_sendmsg_rtn_b32(4));
  append_inst(program, sopp(71, 0)); // s_wait_kmcnt 0
  append_inst(program, s_mov_b32(8, 4));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty());
}

TEST(WaitcheckTest, Gfx1201CounterParityForcesZeroForMixedKmcntEventKinds) {
  std::vector<uint32_t> program;
  append_inst(program, s_sendmsg_rtn_b32(4));
  append_inst(program, s_barrier_signal_isfirst(0));
  append_inst(program, sopp(71, 0)); // s_wait_kmcnt 0.
  append_inst(program, s_mov_b32(8, 4));

  WaitcheckOptions options;
  options.check_counter_parity = true;
  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4, options);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.passed()) << diagnostic_summary(report);
  EXPECT_EQ(report.counter_parity_fields_checked, 1u);
  EXPECT_EQ(report.counter_parity_exact, 1u);
  EXPECT_EQ(report.counter_underaccounting_observed, 0u);
}

TEST(WaitcheckTest, Gfx1201RejectsPartialWaitForMixedKmcntEventKinds) {
  std::vector<uint32_t> program;
  append_inst(program, s_sendmsg_rtn_b32(4));
  append_inst(program, s_barrier_signal_isfirst(0));
  append_inst(program, sopp(71, 1)); // s_wait_kmcnt 1 cannot advance a mixed counter.
  append_inst(program, s_mov_b32(8, 4));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Km);
  EXPECT_EQ(report.diagnostics[0].required_count, 0u);
  EXPECT_EQ(report.diagnostics[0].producer_instruction, "s_sendmsg_rtn_b32 s4, 1");
}

TEST(WaitcheckTest, ReportsKmcntBeforeSendmsgRtnB64Use) {
  std::vector<uint32_t> program;
  append_inst(program, s_sendmsg_rtn_b64(4));
  append_inst(program, s_mov_b32(8, 5));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported);
  ASSERT_EQ(report.diagnostics.size(), 1u);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Km);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Use);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::SGPR);
  EXPECT_EQ(report.diagnostics[0].reg.index, 5u);
  EXPECT_EQ(report.diagnostics[0].required_count, 0u);
}

TEST(WaitcheckTest, Vop3CompareSdstDoesNotOverlapAdjacentScalarLoadResult) {
  std::vector<uint32_t> program;
  append_inst(program, s_load_b32(3, 0));
  append_v_cmp_gt_u32_s2_s5_v12(program);

  WaitcheckOptions options;
  options.fallback_wavefront_size = 32;
  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4, options);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty());
}

TEST(WaitcheckTest, ReportsKmcntWhenVop3CompareOverwritesLoadedScalarMask) {
  std::vector<uint32_t> program;
  append_inst(program, s_load_b32(2, 0));
  append_v_cmp_gt_u32_s2_s5_v12(program);

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported);
  ASSERT_EQ(report.diagnostics.size(), 1u);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Km);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Def);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::SGPR);
  EXPECT_EQ(report.diagnostics[0].reg.index, 2u);
}

TEST(WaitcheckTest, ReportsKmcntBeforeSccUseAfterBarrierSignalIsfirst) {
  std::vector<uint32_t> program;
  append_inst(program, s_barrier_signal_isfirst(0));
  append_inst(program, s_cselect_b32(1));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported);
  ASSERT_EQ(report.diagnostics.size(), 1u);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Km);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Use);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::SCC);
  EXPECT_EQ(report.diagnostics[0].required_count, 0u);
}

TEST(WaitcheckTest, AcceptsKmcntBeforeSccUseAfterBarrierSignalIsfirst) {
  std::vector<uint32_t> program;
  append_inst(program, s_barrier_signal_isfirst(0));
  append_inst(program, sopp(71, 0)); // s_wait_kmcnt 0
  append_inst(program, s_cselect_b32(1));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty());
}

TEST(WaitcheckTest, MatchingBarrierWaitClearsSccWrite) {
  std::vector<uint32_t> program;
  append_inst(program, s_barrier_signal_isfirst(0));
  append_inst(program, sopp(20, 0)); // s_barrier_wait 0
  append_inst(program, s_cselect_b32(1));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty());
}

TEST(WaitcheckTest, DifferentBarrierWaitDoesNotClearSccWrite) {
  std::vector<uint32_t> program;
  append_inst(program, s_barrier_signal_isfirst(0));
  append_inst(program, sopp(20, 1)); // s_barrier_wait 1
  append_inst(program, s_cselect_b32(1));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported);
  ASSERT_EQ(report.diagnostics.size(), 1u);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Km);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::SCC);
}

TEST(WaitcheckTest, ReportsSccWriteFromOtherBlockBeforeSccUse) {
  std::vector<uint32_t> program;
  append_inst(program, s_cmp_eq_u32(0, 128)); // s_cmp_eq_u32 s0, 0
  append_inst(program, sopp(33, 1));          // s_cbranch_scc0 over signal
  append_inst(program, s_barrier_signal_isfirst(0));
  append_inst(program, s_cselect_b32(1));

  TestCodeObject code_object(program);
  auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported);
  ASSERT_EQ(report.diagnostics.size(), 1u);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Km);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Use);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::SCC);
}

TEST(WaitcheckTest, BarrierWaitClearsSccWriteFromOtherBlock) {
  std::vector<uint32_t> program;
  append_inst(program, s_cmp_eq_u32(0, 128)); // s_cmp_eq_u32 s0, 0
  append_inst(program, sopp(33, 1));          // s_cbranch_scc0 over signal
  append_inst(program, s_barrier_signal_isfirst(0));
  append_inst(program, sopp(20, 0)); // s_barrier_wait 0
  append_inst(program, s_cselect_b32(1));

  TestCodeObject code_object(program);
  auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty());
}

TEST(WaitcheckTest, ReportsMissingSamplecntBeforeImageSampleUse) {
  std::vector<uint32_t> program;
  append_inst(program, image_sample(0));
  append_inst(program, v_mov_b32(4, 0));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported);
  ASSERT_EQ(report.diagnostics.size(), 1u);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Sample);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Use);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::VGPR);
  EXPECT_EQ(report.diagnostics[0].reg.index, 0u);
}

TEST(WaitcheckTest, AcceptsSamplecntZeroBeforeImageSampleUse) {
  std::vector<uint32_t> program;
  append_inst(program, image_sample(0));
  append_inst(program, sopp(66, 0)); // s_wait_samplecnt 0
  append_inst(program, v_mov_b32(4, 0));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty());
}

TEST(WaitcheckTest, ReportsMissingSamplecntBeforeImageMsaaLoadUse) {
  std::vector<uint32_t> program;
  append_inst(program, image_msaa_load(0));
  append_inst(program, v_mov_b32(4, 0));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported);
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Sample);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Use);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::VGPR);
  EXPECT_EQ(report.diagnostics[0].reg.index, 0u);
}

TEST(WaitcheckTest, AcceptsSamplecntZeroBeforeImageMsaaLoadUse) {
  std::vector<uint32_t> program;
  append_inst(program, image_msaa_load(0));
  append_inst(program, sopp(66, 0)); // s_wait_samplecnt 0
  append_inst(program, v_mov_b32(4, 0));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, ReportsMissingLoadcntBeforeImageAtomicResultUse) {
  std::vector<uint32_t> program;
  append_inst(program, image_atomic_add_uint(0));
  append_inst(program, v_mov_b32(4, 0));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported);
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Load);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Use);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::VGPR);
  EXPECT_EQ(report.diagnostics[0].reg.index, 0u);
}

TEST(WaitcheckTest, AcceptsLoadcntZeroBeforeImageAtomicResultUse) {
  std::vector<uint32_t> program;
  append_inst(program, image_atomic_add_uint(0));
  append_inst(program, sopp(64, 0)); // s_wait_loadcnt 0
  append_inst(program, v_mov_b32(4, 0));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, ReportsImageAtomicWaitsBeforeImageMsaaOverwrite) {
  std::vector<uint32_t> program;
  append_inst(program, image_atomic_add_uint(0));
  append_inst(program, image_msaa_load(0));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported);
  ASSERT_EQ(report.diagnostics.size(), 2u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Load);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Def);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::VGPR);
  EXPECT_EQ(report.diagnostics[0].reg.index, 0u);
  EXPECT_EQ(report.diagnostics[1].counter, WaitCounterKind::VmVsrc);
  EXPECT_EQ(report.diagnostics[1].access, WaitcheckAccessKind::Def);
  EXPECT_EQ(report.diagnostics[1].reg.cls, RegClass::VGPR);
  EXPECT_EQ(report.diagnostics[1].reg.index, 0u);
}

TEST(WaitcheckTest, AcceptsImageAtomicWaitsBeforeImageMsaaOverwrite) {
  std::vector<uint32_t> program;
  append_inst(program, image_atomic_add_uint(0));
  append_inst(program, sopp(64, 0)); // s_wait_loadcnt 0
  append_inst(program, s_wait_alu_vm_vsrc_0());
  append_inst(program, image_msaa_load(0));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, ReportsMissingBvhcntBeforeImageBvhUse) {
  std::vector<uint32_t> program;
  append_inst(program, image_bvh_intersect_ray(0));
  append_inst(program, v_mov_b32(4, 0));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported);
  ASSERT_EQ(report.diagnostics.size(), 1u);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Bvh);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Use);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::VGPR);
  EXPECT_EQ(report.diagnostics[0].reg.index, 0u);
}

TEST(WaitcheckTest, AcceptsBvhcntZeroBeforeImageBvhUse) {
  std::vector<uint32_t> program;
  append_inst(program, image_bvh_intersect_ray(0));
  append_inst(program, sopp(67, 0)); // s_wait_bvhcnt 0
  append_inst(program, v_mov_b32(4, 0));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty());
}

TEST(WaitcheckTest, ReportsBvhcntBeforeVmemOverwrite) {
  std::vector<uint32_t> program;
  append_inst(program, image_bvh_intersect_ray(0));
  append_inst(program, global_load_b32(0));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported);
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Bvh);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Def);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::VGPR);
  EXPECT_EQ(report.diagnostics[0].reg.index, 0u);
}

TEST(WaitcheckTest, AcceptsBvhcntBeforeVmemOverwrite) {
  std::vector<uint32_t> program;
  append_inst(program, image_bvh_intersect_ray(0));
  append_inst(program, sopp(67, 0)); // s_wait_bvhcnt 0
  append_inst(program, global_load_b32(0));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, ReportsBvhcntBeforeImageSampleOverwrite) {
  std::vector<uint32_t> program;
  append_inst(program, image_bvh_intersect_ray(0));
  append_inst(program, image_sample(0));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported);
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Bvh);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Def);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::VGPR);
  EXPECT_EQ(report.diagnostics[0].reg.index, 0u);
}

TEST(WaitcheckTest, ReportsLoadcntBeforeBvhOverwrite) {
  std::vector<uint32_t> program;
  append_inst(program, global_load_b32(0));
  append_inst(program, image_bvh_intersect_ray(0));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported);
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Load);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Def);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::VGPR);
  EXPECT_EQ(report.diagnostics[0].reg.index, 0u);
}

TEST(WaitcheckTest, AcceptsLoadcntBeforeBvhOverwrite) {
  std::vector<uint32_t> program;
  append_inst(program, global_load_b32(0));
  append_inst(program, sopp(64, 0)); // s_wait_loadcnt 0
  append_inst(program, image_bvh_intersect_ray(0));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, ReportsMissingWaitAluVmVsrcBeforeVmemStoreSourceOverwrite) {
  std::vector<uint32_t> program;
  append_inst(program, global_store_b32(0));
  append_inst(program, v_mov_b32(0, 4));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported);
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::VmVsrc);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Def);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::VGPR);
  EXPECT_EQ(report.diagnostics[0].reg.index, 0u);
}

TEST(WaitcheckTest, StoreSourceReadDoesNotNeedExpcnt) {
  std::vector<uint32_t> program;
  append_inst(program, global_store_b32(0));
  append_inst(program, v_mov_b32(1, 0));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty());
}

TEST(WaitcheckTest, GlobalWbDoesNotInventMemoryDependency) {
  std::vector<uint32_t> program;
  append_inst(program, global_wb());
  append_inst(program, global_store_b32(0));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty());
}

TEST(WaitcheckTest, GlobalWbinvDoesNotInventMemoryDependency) {
  std::vector<uint32_t> program;
  append_inst(program, global_wbinv());
  append_inst(program, global_store_b32(0));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty());
}

TEST(WaitcheckTest, GlobalInvDoesNotInventMemoryDependency) {
  std::vector<uint32_t> program;
  append_inst(program, global_inv());
  append_inst(program, global_store_b32(0));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty());
}

TEST(WaitcheckTest, GlobalInvalidationOccupiesOrderedLoadCompletionStream) {
  for (auto arch : {ROCJITSU_CODE_ARCH_RDNA4, ROCJITSU_CODE_ARCH_CDNA5}) {
    for (unsigned invalidations : {0u, 1u, 2u, 62u, 63u, 80u}) {
      for (unsigned threshold : {0u, 1u, 2u, 61u, 62u, 63u}) {
        SCOPED_TRACE(testing::Message()
                     << arch << " invalidations=" << invalidations << " threshold=" << threshold);
        std::vector<uint32_t> program;
        append_inst(program, global_load_b32(0));
        for (unsigned i = 0; i < invalidations; ++i)
          append_inst(program, global_inv());
        append_inst(program, sopp(64, threshold));
        append_inst(program, v_mov_b32(1, 0));
        const auto report = analyze_waitcnts(program, arch);
        ASSERT_TRUE(report.supported) << report.analysis_error;
        const unsigned required = std::min(invalidations, 63u);
        const bool missing = threshold > required;
        ASSERT_EQ(report.diagnostics.size(), missing) << diagnostic_summary(report);
        if (missing) {
          EXPECT_EQ(report.diagnostics.front().counter, WaitCounterKind::Load);
          EXPECT_EQ(report.diagnostics.front().required_count, required);
        }
      }
    }
  }
}

TEST(WaitcheckTest, GlobalWritebackDoesNotOccupyLoadCompletionStream) {
  for (auto arch : {ROCJITSU_CODE_ARCH_RDNA4, ROCJITSU_CODE_ARCH_CDNA5}) {
    std::vector<uint32_t> program;
    append_inst(program, global_load_b32(0));
    append_inst(program, global_wb());
    append_inst(program, sopp(64, 1));
    append_inst(program, v_mov_b32(1, 0));
    const auto report = analyze_waitcnts(program, arch);
    ASSERT_TRUE(report.supported) << report.analysis_error;
    ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
    EXPECT_EQ(report.diagnostics.front().counter, WaitCounterKind::Load);
    EXPECT_EQ(report.diagnostics.front().required_count, 0u);
  }
}

TEST(WaitcheckTest, IndexedLdsWritesStayOrderedAndNopsConsumeQueuePositions) {
  for (unsigned younger : {0u, 1u, 61u, 62u}) {
    std::vector<uint32_t> program;
    append_inst(program, ds_swizzle_b32(2, 1));
    append_inst(program, ds_swizzle_b32(2, 1));
    for (unsigned i = 0; i < younger; ++i)
      append_inst(program, ds_nop());
    if (younger == 1)
      append_inst(program, sopp(70, 1));
    append_inst(program, ds_swizzle_b32(3, 2));
    const auto report = analyze_gfx1250_normal(program);
    ASSERT_TRUE(report.supported) << report.analysis_error;
    EXPECT_FALSE(std::ranges::any_of(report.diagnostics, [](const auto &d) {
      return d.access == WaitcheckAccessKind::Def;
    })) << diagnostic_summary(report);
    EXPECT_EQ(!report.diagnostics.empty(), younger == 0 || younger == 61)
        << diagnostic_summary(report);
  }
}

TEST(WaitcheckTest, Gfx950AdmissionReservesTheActualScalarTransferWidth) {
  for (bool doubleword : {false, true}) {
    std::vector<uint32_t> program;
    // Fourteen ordered LDS reads fit in the 15-entry legacy LGKM queue.
    for (uint32_t dst = 5; dst < 19; ++dst)
      program.insert(program.end(), {0xd86c0000, (dst << 24) | 1}); // ds_read_b32 vdst,v1
    program.insert(program.end(), {doubleword ? 0xc0060100u : 0xc0020100u, 0});
    append_inst(program, v_mov_b32(20, 5));
    const auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA4);
    ASSERT_TRUE(report.supported) << report.analysis_error;
    ASSERT_EQ(report.diagnostics.size(), doubleword ? 0u : 1u) << diagnostic_summary(report);
    if (!doubleword)
      EXPECT_EQ(report.diagnostics.front().required_count, 13u);
  }
}

TEST(WaitcheckTest, MixedLgkmRetiresOnlyTheBoundedLdsFifo) {
  for (unsigned younger : {0u, 1u}) {
    std::vector<uint32_t> program{0xd86c0000, 0x05000001}; // ds_read_b32 v5,v1
    append_gfx942_s_load_dword_s4_s0(program);
    if (younger)
      program.insert(program.end(), {0xd86c0000, 0x06000001}); // ds_read_b32 v6,v1
    program.push_back(0xbf8cc17f);                             // lgkmcnt(1)
    append_inst(program, v_mov_b32(7, 5));
    const auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA3);
    ASSERT_TRUE(report.supported) << report.analysis_error;
    EXPECT_EQ(report.diagnostics.size(), younger == 0) << diagnostic_summary(report);
  }
}

TEST(WaitcheckTest, AcceptsEndpgmAfterPendingVmemStore) {
  std::vector<uint32_t> program;
  append_inst(program, global_store_b32(0));
  append_inst(program, s_endpgm());

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, EndpgmClearsPendingStateBeforeNextEntry) {
  std::vector<uint32_t> program;
  append_inst(program, global_load_b32(0));
  append_inst(program, s_endpgm());
  append_inst(program, v_mov_b32(1, 0));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, AcceptsStorecntBeforeEndpgmAfterVmemStore) {
  std::vector<uint32_t> program;
  append_inst(program, global_store_b32(0));
  append_inst(program, sopp(65, 0)); // s_wait_storecnt 0
  append_inst(program, s_endpgm());

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty());
}

TEST(WaitcheckTest, AcceptsWaitAluVmVsrcBeforeVmemStoreSourceOverwrite) {
  std::vector<uint32_t> program;
  append_inst(program, global_store_b32(0));
  append_inst(program, s_wait_alu_vm_vsrc_0());
  append_inst(program, v_mov_b32(0, 4));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, ReportsMissingWaitAluVmVsrcBeforeImageStoreSourceOverwrite) {
  std::vector<uint32_t> program;
  append_inst(program, image_store_b32(0));
  append_inst(program, v_mov_b32(0, 4));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported);
  EXPECT_EQ(report.memory_events_tracked, 2u);
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::VmVsrc);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Def);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::VGPR);
  EXPECT_EQ(report.diagnostics[0].reg.index, 0u);
}

TEST(WaitcheckTest, AcceptsWaitAluVmVsrcBeforeImageStoreSourceOverwrite) {
  std::vector<uint32_t> program;
  append_inst(program, image_store_b32(0));
  append_inst(program, s_wait_alu_vm_vsrc_0());
  append_inst(program, v_mov_b32(0, 4));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, ReportsMissingExpcntBeforeDsParamLoadUse) {
  std::vector<uint32_t> program;
  append_inst(program, ds_param_load(1));
  append_inst(program, v_mov_b32(2, 1));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported);
  ASSERT_EQ(report.diagnostics.size(), 1u);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Exp);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Use);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::VGPR);
  EXPECT_EQ(report.diagnostics[0].reg.index, 1u);
}

TEST(WaitcheckTest, ReportsMissingWaitVmVsrcBeforeDsParamLoadSourceOverwrite) {
  std::vector<uint32_t> program;
  append_inst(program, global_load_b32(0)); // uses v8 as the vector offset.
  append_inst(program, ds_param_load_with_waits(8, 15, 1));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported);
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::VmVsrc);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Def);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::VGPR);
  EXPECT_EQ(report.diagnostics[0].reg.index, 8u);
}

TEST(WaitcheckTest, AcceptsDsParamLoadWaitVmVsrcBeforeSourceOverwrite) {
  std::vector<uint32_t> program;
  append_inst(program, global_load_b32(0)); // uses v8 as the vector offset.
  append_inst(program, ds_param_load_with_waits(8, 15, 0));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, ReportsMissingWaitVaVdstBeforeDsParamLoadAfterImmediateValuRead) {
  std::vector<uint32_t> program;
  append_inst(program, v_add_f32_e32(0, 257, 1)); // VALU reads v1.
  append_inst(program, ds_param_load_with_waits(1, 15, 1));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported);
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].kind, WaitcheckDiagnosticKind::VaVdst);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::VaVdst);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Def);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::VGPR);
  EXPECT_EQ(report.diagnostics[0].reg.index, 1u);
  EXPECT_EQ(report.diagnostics[0].required_count, 0u);
}

TEST(WaitcheckTest, AcceptsWaitVaVdstZeroBeforeDsParamLoadAfterImmediateValuRead) {
  std::vector<uint32_t> program;
  append_inst(program, v_add_f32_e32(0, 257, 1)); // VALU reads v1.
  append_inst(program, ds_param_load_with_waits(1, 0, 1));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, ReportsMissingWaitVaVdstWithOneInterveningValu) {
  std::vector<uint32_t> program;
  append_inst(program, v_add_f32_e32(0, 257, 1)); // VALU reads v1.
  append_inst(program, v_add_f32_e32(2, 258, 2));
  append_inst(program, ds_param_load_with_waits(1, 15, 1));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported);
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::VaVdst);
  EXPECT_EQ(report.diagnostics[0].required_count, 1u);
}

TEST(WaitcheckTest, AcceptsWaitVaVdstOneWithOneInterveningValu) {
  std::vector<uint32_t> program;
  append_inst(program, v_add_f32_e32(0, 257, 1)); // VALU reads v1.
  append_inst(program, v_add_f32_e32(2, 258, 2));
  append_inst(program, ds_param_load_with_waits(1, 1, 1));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, ReportsWaitVaVdstZeroForTransMixedDsParamLoadHazard) {
  std::vector<uint32_t> program;
  append_inst(program, v_add_f32_e32(0, 257, 1)); // VALU reads v1.
  append_inst(program, v_sqrt_f32(2, 2));
  append_inst(program, v_add_f32_e32(3, 258, 2));
  append_inst(program, ds_param_load_with_waits(1, 1, 1));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported);
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::VaVdst);
  EXPECT_EQ(report.diagnostics[0].required_count, 0u);
}

TEST(WaitcheckTest, AcceptsWaitVaVdstZeroForTransMixedDsParamLoadHazard) {
  std::vector<uint32_t> program;
  append_inst(program, v_add_f32_e32(0, 257, 1)); // VALU reads v1.
  append_inst(program, v_sqrt_f32(2, 2));
  append_inst(program, v_add_f32_e32(3, 258, 2));
  append_inst(program, ds_param_load_with_waits(1, 0, 1));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, VaVdstProducerWitnessConvergesAcrossInterlockingBackedges) {
  for (auto [arch, mach] : {std::pair{ROCJITSU_CODE_ARCH_CDNA5, EF_AMDGPU_MACH_AMDGCN_GFX1250},
                            std::pair{ROCJITSU_CODE_ARCH_RDNA4, EF_AMDGPU_MACH_AMDGCN_GFX1201}}) {
    for (uint32_t mode : {0u, 2u}) {
      std::vector<uint32_t> code;
      append_inst(code, s_setreg_imm32_b32(hwreg(26, 0, 2), mode));
      code.insert(code.end(), {
          0xbfa10001, //           s_cbranch_scc0 block2
          0xbfa10002, // block1:   s_cbranch_scc0 writer_a
          0xbfa00000, // block2:   s_branch block3
          0xbfa1fffd, // block3:   s_cbranch_scc0 block1
          0x7e000280, // writer_a: v_mov_b32 v0, 0
          0xbfa1fffd, //           s_cbranch_scc0 block3
          0x7e000281, // writer_b: v_mov_b32 v0, 1
          0xbfa1fff9, //           s_cbranch_scc0 block1
          0xbfb00000, //           s_endpgm
      });
      const auto image = waitcheck_test::make_gfx_code_object(code, mach, true);
      const AmdGpuCodeObject object(image.data(), image.size());
      const auto report = analyze_waitcnts(object, arch);
      EXPECT_TRUE(report.complete && report.supported) << report.analysis_error;
      EXPECT_TRUE(report.diagnostics.empty());
    }
  }
}

TEST(WaitcheckTest, VaVdstJoinKeepsTransExpiryAndTheYoungOrdinaryProducer) {
  for (const unsigned trans_age : {13u, 14u}) {
    for (const unsigned count : {0u, 1u, 2u, 15u}) {
      SCOPED_TRACE(testing::Message() << "TRANS age " << trans_age << ", embedded wait " << count);
      std::vector<uint32_t> code;
      append_inst(code, s_setreg_imm32_b32(hwreg(26, 0, 2), 2));
      // The fallthrough producer requires a zero wait until it ages out.
      append_inst(code, sopp(33, trans_age + 2)); // s_cbranch_scc0 to ordinary
      const size_t trans_pc = code.size() * sizeof(uint32_t);
      append_inst(code, v_sqrt_f32(1, 1));
      for (unsigned i = 0; i < trans_age; ++i)
        append_inst(code, v_add_f32_e32(8, 264, 8));
      append_inst(code, sopp(32, 1)); // s_branch to join
      const size_t ordinary_pc = code.size() * sizeof(uint32_t);
      append_inst(code, v_add_f32_e32(1, 257, 1));
      // After this issue only the age-13 TRANS can still need zero.
      append_inst(code, v_add_f32_e32(8, 264, 8));
      const size_t consumer_pc = code.size() * sizeof(uint32_t);
      append_inst(code, ds_param_load_with_waits(1, count, 1));
      append_inst(code, s_endpgm());
      const TestCodeObject object(code);
      WaitcheckOptions options;
      options.check_counter_parity = true;
      const auto report = analyze_waitcnts(object, ROCJITSU_CODE_ARCH_RDNA4, options);
      ASSERT_TRUE(report.complete && report.supported) << report.analysis_error;
      const bool trans_live = trans_age == 13;
      const unsigned required = trans_live ? 0 : 1;
      const auto missing = std::ranges::find_if(report.diagnostics, [&](const auto &d) {
        return d.counter == WaitCounterKind::VaVdst && d.section_offset == consumer_pc;
      });
      if (count <= required) {
        EXPECT_EQ(missing, report.diagnostics.end()) << diagnostic_summary(report);
      } else {
        ASSERT_NE(missing, report.diagnostics.end()) << diagnostic_summary(report);
        EXPECT_EQ(missing->required_count, required);
        EXPECT_EQ(missing->producer_section_offset, trans_live ? trans_pc : ordinary_pc);
      }
      if (count < 15)
        EXPECT_EQ(report.counter_parity_exact, count == required ? 1u : 0u);
    }
  }
}

TEST(WaitcheckTest, AcceptsWaitVaVdstFifteenAfterVmemExpiry) {
  std::vector<uint32_t> program;
  append_inst(program, v_add_f32_e32(0, 257, 1)); // VALU reads v1.
  append_inst(program, global_load_b32(4));
  append_inst(program, ds_param_load_with_waits(1, 15, 1));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, ReportsMissingExpcntBeforeDsDirectLoadOverwrite) {
  std::vector<uint32_t> program;
  append_inst(program, ds_direct_load(1));
  append_inst(program, v_mov_b32(1, 4));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported);
  ASSERT_EQ(report.diagnostics.size(), 1u);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Exp);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Def);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::VGPR);
  EXPECT_EQ(report.diagnostics[0].reg.index, 1u);
}

TEST(WaitcheckTest, AcceptsExpcntBeforeDsDirectLoadUse) {
  std::vector<uint32_t> program;
  append_inst(program, ds_direct_load(1));
  append_inst(program, sopp(68, 0)); // s_wait_expcnt 0
  append_inst(program, v_mov_b32(2, 1));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty());
}

TEST(WaitcheckTest, AcceptsVinterpWaitExpBeforeDsDirectLoadUse) {
  std::vector<uint32_t> program;
  append_inst(program, ds_direct_load(1));
  append_inst(program, v_interp_p10_f32(2, 1, 0));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty());
}

TEST(WaitcheckTest, ReportsVinterpWaitExpOneLeavesNewestDsDirectLoadPending) {
  std::vector<uint32_t> program;
  append_inst(program, ds_direct_load(1));
  append_inst(program, ds_direct_load(2));
  append_inst(program, v_interp_p10_f32(3, 2, 1));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported);
  ASSERT_EQ(report.diagnostics.size(), 1u);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Exp);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Use);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::VGPR);
  EXPECT_EQ(report.diagnostics[0].reg.index, 2u);
  EXPECT_EQ(report.diagnostics[0].required_count, 0u);
}

TEST(WaitcheckTest, AcceptsVinterpWaitExpOneForOlderDsDirectLoad) {
  std::vector<uint32_t> program;
  append_inst(program, ds_direct_load(1));
  append_inst(program, ds_direct_load(2));
  append_inst(program, v_interp_p10_f32(3, 1, 1));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty());
}

TEST(WaitcheckTest, ReportsMissingExpcntBeforeExecOverwriteAfterExport) {
  std::vector<uint32_t> program;
  append_inst(program, export_mrt0_v0());
  append_inst(program, s_mov_b64_exec_from_s0());

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported);
  ASSERT_EQ(report.diagnostics.size(), 1u);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Exp);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Def);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::EXEC);
}

TEST(WaitcheckTest, AcceptsExpcntZeroBeforeExecOverwriteAfterExport) {
  std::vector<uint32_t> program;
  append_inst(program, export_mrt0_v0());
  append_inst(program, sopp(68, 0)); // s_wait_expcnt 0
  append_inst(program, s_mov_b64_exec_from_s0());

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty());
}

TEST(WaitcheckTest, CombinedLoadcntDscntWaitsBothCounters) {
  std::vector<uint32_t> program;
  append_inst(program, global_load_b32(0));
  append_inst(program, ds_load_b32(1, 4));
  append_inst(program, sopp(72, 0)); // s_wait_loadcnt_dscnt loadcnt(0), dscnt(0)
  append_inst(program, v_mov_b32(2, 0));
  append_inst(program, v_mov_b32(3, 1));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty());
}

TEST(WaitcheckTest, CombinedLoadcntDscntLeavesNewestLoadPending) {
  std::vector<uint32_t> program;
  append_inst(program, global_load_b32(0));
  append_inst(program, global_load_b32(1));
  append_inst(program, ds_load_b32(2, 4));
  append_inst(program, sopp(72, 0x100)); // loadcnt(1), dscnt(0)
  append_inst(program, v_mov_b32(3, 1));
  append_inst(program, v_mov_b32(4, 2));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported);
  ASSERT_EQ(report.diagnostics.size(), 1u);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Load);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::VGPR);
  EXPECT_EQ(report.diagnostics[0].reg.index, 1u);
}

TEST(WaitcheckTest, CombinedStorecntDscntWaitsBothCounters) {
  std::vector<uint32_t> program;
  append_inst(program, global_store_b32(10));
  append_inst(program, ds_load_b32(0, 4));
  append_inst(program, sopp(73, 0)); // s_wait_storecnt_dscnt storecnt(0), dscnt(0)
  append_inst(program, v_mov_b32(2, 0));
  append_inst(program, s_endpgm());

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, CombinedStorecntDscntLeavesStorePending) {
  std::vector<uint32_t> program;
  append_inst(program, global_store_b32(10));
  append_inst(program, ds_load_b32(0, 4));
  append_inst(program, sopp(73, 0x100)); // storecnt(1), dscnt(0)
  append_inst(program, v_mov_b32(2, 0));
  append_inst(program, s_endpgm());

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, CombinedStorecntDscntLeavesDsPending) {
  std::vector<uint32_t> program;
  append_inst(program, global_store_b32(10));
  append_inst(program, ds_load_b32(0, 4));
  append_inst(program, sopp(73, 1)); // storecnt(0), dscnt(1)
  append_inst(program, v_mov_b32(2, 0));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported);
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Ds);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Use);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::VGPR);
  EXPECT_EQ(report.diagnostics[0].reg.index, 0u);
}

TEST(WaitcheckTest, CombinedLoadcntDscntDecodesSixBitFields) {
  std::vector<uint32_t> program;
  append_inst(program, global_load_b32(0));
  append_inst(program, ds_load_b32(1, 4));
  append_inst(program, sopp(72, 0x3F01)); // loadcnt(63), dscnt(1)
  append_inst(program, v_mov_b32(2, 0));
  append_inst(program, v_mov_b32(3, 1));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported);
  ASSERT_EQ(report.diagnostics.size(), 2u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Load);
  EXPECT_EQ(report.diagnostics[0].reg.index, 0u);
  EXPECT_EQ(report.diagnostics[1].counter, WaitCounterKind::Ds);
  EXPECT_EQ(report.diagnostics[1].reg.index, 1u);
}

TEST(WaitcheckTest, ReportsOverwriteBeforeLoadCompletes) {
  std::vector<uint32_t> program;
  append_inst(program, global_load_b32(0));
  append_inst(program, v_mov_b32(0, 2));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported);
  ASSERT_EQ(report.diagnostics.size(), 1u);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Def);
  EXPECT_EQ(report.diagnostics[0].reg.index, 0u);
}

TEST(WaitcheckTest, ObjectAnalysisIgnoresUnreachableSkippedUse) {
  std::vector<uint32_t> program;
  append_inst(program, global_load_b32(0));
  append_inst(program, sopp(32, 1)); // s_branch over the next instruction
  append_inst(program, v_mov_b32(1, 0));
  append_inst(program, sopp(64, 0)); // s_wait_loadcnt 0
  append_inst(program, v_mov_b32(2, 0));

  TestCodeObject code_object(program);
  auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty());
}

TEST(WaitcheckTest, Gfx950KernelAnalysisIgnoresHazardInUnreachableBlock) {
  const std::vector<uint32_t> program{
      0xBF820004u,              // 0x00: s_branch to s_endpgm.
      0xC00E0A00u, 0x00000050u, // 0x04: dead s_load_dwordx8 s[40:47], s[0:1], 0x50.
      0xC0020B40u, 0x0000008Cu, // 0x0c: dead s_load_dword s45, s[0:1], 0x8c.
      0xBF810000u,              // 0x14: s_endpgm.
  };
  const auto image = rocjitsu::waitcheck_test::make_gfx_multi_kernel_code_object(
      {{"dead_block", program}}, EF_AMDGPU_MACH_AMDGCN_GFX950);
  AmdGpuCodeObject code_object(image.data(), image.size());

  const auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_CDNA4);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
  EXPECT_EQ(report.instructions_analyzed, 2u);
  EXPECT_EQ(report.memory_events_tracked, 0u);
}

TEST(WaitcheckTest, Gfx950KernelAnalysisAttributesRank1HazardToReachableProducer) {
  const std::vector<uint32_t> program{
      0xBF068205u,              // 0x00: s_cmp_eq_u32 s5, 2.
      0xBF850003u,              // 0x04: s_cbranch_scc1 to the alternate path.
      0xC00E0A00u, 0x00000050u, // 0x08: fallthrough s_load_dwordx8 s[40:47].
      0xBF820004u,              // 0x10: s_branch to the join.
      0xC00E0A00u, 0x00000050u, // 0x14: reachable s_load_dwordx8 s[40:47].
      0xC0020B40u, 0x0000008Cu, // 0x1c: s_load_dword s45 overlaps the reachable load.
      0xBF8CC07Fu,              // 0x24: s_waitcnt lgkmcnt(0).
      0xBF810000u,              // 0x28: s_endpgm.
  };
  const auto image = rocjitsu::waitcheck_test::make_gfx_multi_kernel_code_object(
      {{"rank1_branch", program}}, EF_AMDGPU_MACH_AMDGCN_GFX950);
  AmdGpuCodeObject code_object(image.data(), image.size());

  const auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_CDNA4);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Def);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::SGPR);
  EXPECT_EQ(report.diagnostics[0].reg.index, 45u);
  EXPECT_EQ(report.diagnostics[0].producer_section_offset, 0x14u);
  EXPECT_EQ(report.diagnostics[0].section_offset, 0x1cu);
}

TEST(WaitcheckTest, ObjectAnalysisReportsPathThatSkipsWaitAtJoin) {
  std::vector<uint32_t> program;
  append_inst(program, global_load_b32(0));
  append_inst(program, sopp(33, 1)); // s_cbranch_scc0 over the wait
  append_inst(program, sopp(64, 0)); // s_wait_loadcnt 0 on fallthrough only
  append_inst(program, v_mov_b32(1, 0));

  TestCodeObject code_object(program);
  auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported);
  ASSERT_EQ(report.diagnostics.size(), 1u);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Load);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Use);
  EXPECT_EQ(report.diagnostics[0].reg.index, 0u);
}

TEST(WaitcheckTest, ScratchVectorAddressZeroDependsOnEnableBit) {
  for (auto arch : {ROCJITSU_CODE_ARCH_RDNA3, ROCJITSU_CODE_ARCH_RDNA3_5, ROCJITSU_CODE_ARCH_RDNA4,
                    ROCJITSU_CODE_ARCH_CDNA5}) {
    const bool legacy = arch == ROCJITSU_CODE_ARCH_RDNA3 || arch == ROCJITSU_CODE_ARCH_RDNA3_5;
    for (bool enabled : {false, true}) {
      SCOPED_TRACE(arch);
      SCOPED_TRACE(enabled);
      std::vector<uint32_t> words = legacy
                                        ? std::vector<uint32_t>{0xdc510000, 0x007c0000, 0xdc550010,
                                                                enabled ? 0x10fc0000u : 0x107c0000u}
                                        : std::vector<uint32_t>{0xed05007c,
                                                                0x00000000,
                                                                0x00000000,
                                                                0xed05007c,
                                                                enabled ? 0x00020010u : 0x00000010u,
                                                                0x00001000};
      const auto result = analyze_waitcnts(words, arch);
      ASSERT_TRUE(result.supported) << result.analysis_error;
      EXPECT_EQ(std::ranges::any_of(result.diagnostics,
                                    [](const WaitcheckDiagnostic &diagnostic) {
                                      return diagnostic.access == WaitcheckAccessKind::Use &&
                                             diagnostic.reg == (RegisterRef{RegClass::VGPR, 0, 1});
                                    }),
                enabled)
          << diagnostic_summary(result);
    }
  }
}

TEST(WaitcheckTest, RelativeVgprUseRespectsPendingLoadOnLoopEntry) {
  for (const bool drained : {false, true}) {
    for (const bool known_offset : {false, true}) {
      SCOPED_TRACE(drained);
      SCOPED_TRACE(known_offset);
      std::vector<uint32_t> program{
          known_offset ? 0xbe810083u : 0xbe810002u, // s_mov_b32 s1, 3 or s2
          0xdc520000u,
          0x047c0000u,                         // global_load_b32 v4, v[0:1], off
          0xbfa00002u,                         // s_branch to the loop header
          0x81018401u,                         // s_add_i32 s1, s1, 4
          0xbfa00000u,                         // s_branch to the loop header
          0x817dc301u,                         // s_add_i32 m0, s1, -3
          0x7e0e8700u,                         // v_movrels_b32_e32 v7, v0
          drained ? 0xbf8903f7u : 0xbf800000u, // s_waitcnt vmcnt(0) or s_nop 0
          0xbfa2fffau,                         // s_cbranch_scc1 to the backedge
          0xbfb00000u,                         // s_endpgm
      };
      const auto image = rocjitsu::waitcheck_test::make_gfx_code_object(
          program, EF_AMDGPU_MACH_AMDGCN_GFX1100, true);
      AmdGpuCodeObject object(image.data(), image.size());
      const auto result = analyze_waitcnts(object, ROCJITSU_CODE_ARCH_RDNA3);
      ASSERT_TRUE(result.supported) << result.analysis_error;
      EXPECT_EQ(std::ranges::any_of(result.diagnostics,
                                    [](const WaitcheckDiagnostic &diagnostic) {
                                      return diagnostic.instruction.starts_with("v_movrels");
                                    }),
                !drained || !known_offset)
          << diagnostic_summary(result);
    }
  }
}

TEST(WaitcheckTest, ScalarMemoryProtectsEachVccDwordAndWholeWaveMaskUses) {
  struct Consumer {
    uint32_t word;
    uint8_t halves;
    WaitcheckAccessKind access;
  };
  constexpr Consumer consumers[] = {
      {0xbe87006a, 1, WaitcheckAccessKind::Use}, // s_mov_b32 s7, vcc_lo
      {0xbe87006b, 2, WaitcheckAccessKind::Use}, // s_mov_b32 s7, vcc_hi
      {0xbeeb0080, 2, WaitcheckAccessKind::Def}, // s_mov_b32 vcc_hi, 0
      {0xbe8700fb, 3, WaitcheckAccessKind::Use}, // s_mov_b32 s7, src_vccz
      {0x00040903, 3, WaitcheckAccessKind::Use}, // v_cndmask_b32_e32 v2, v3, v4, vcc
      {0xbe872a64, 1, WaitcheckAccessKind::Use}, // s_movrels_b32 s7, s100 (s100+6 = VCC_LO)
      {0xbe872a65, 2, WaitcheckAccessKind::Use}, // s_movrels_b32 s7, s101 (s101+6 = VCC_HI)
  };
  for (unsigned produced : {1u, 2u, 3u}) {
    for (const auto &consumer : consumers) {
      for (bool waited : {false, true}) {
        SCOPED_TRACE(testing::Message() << "produced=" << produced << " consumer=" << std::hex
                                        << consumer.word << " waited=" << waited);
        // s_load_dword vcc_lo/hi or s_load_dwordx2 vcc from the same address.
        const uint32_t load = produced == 1 ? 0xc0021a80 : produced == 2 ? 0xc0021ac0 : 0xc0061a80;
        std::vector<uint32_t> program = {0xbefc0086, load, 0}; // s_mov_b32 m0, 6
        if (waited)
          program.push_back(0xbf8cc07f); // s_waitcnt lgkmcnt(0)
        program.push_back(consumer.word);
        program.push_back(0xbf810000); // s_endpgm
        auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA3);
        ASSERT_TRUE(report.supported && report.complete) << report.analysis_error;
        const unsigned overlap = produced & consumer.halves;
        ASSERT_EQ(report.diagnostics.size(), !waited && overlap ? 1u : 0u)
            << diagnostic_summary(report);
        if (!report.diagnostics.empty()) {
          EXPECT_EQ(report.diagnostics.front().reg.cls, RegClass::VCC);
          EXPECT_EQ(report.diagnostics.front().reg.index, overlap & 1 ? 0u : 1u);
          EXPECT_EQ(report.diagnostics.front().access, consumer.access);
        }
      }
    }
  }
}

TEST(WaitcheckTest, Wave64ZeroExecPathsPreserveScalarAndUnmaskedLaneAccesses) {
  const auto check = [](std::vector<uint32_t> program, size_t expected) {
    program.push_back(0xbf810000); // s_endpgm
    const auto image = waitcheck_test::make_gfx_code_object(program, EF_AMDGPU_MACH_AMDGCN_GFX942);
    AmdGpuCodeObject object(image.data(), image.size());
    auto report = analyze_waitcnts(object, ROCJITSU_CODE_ARCH_CDNA3);
    EXPECT_TRUE(report.supported && report.complete) << report.analysis_error;
    EXPECT_EQ(report.diagnostics.size(), expected) << diagnostic_summary(report);
  };
  constexpr uint32_t ds_read = 0xd86c0000, ds_args = 0x05000001; // ds_read_b32 v5,v1
  constexpr uint32_t exec_zero = 0xbefe0180, exec_unknown = 0xbefe010a, exec_all = 0xbefe01c1;
  constexpr uint32_t branch_z = 0xbf880001, branch_nz = 0xbf890001, wait = 0xbf8cc07f;
  constexpr uint32_t consume = 0x7e0e0305; // v_mov_b32_e32 v7,v5
  check({ds_read, ds_args, exec_unknown, branch_z, wait, consume}, 0);
  check({ds_read, ds_args, exec_unknown, branch_nz, wait, consume}, 1);
  check({ds_read, ds_args, exec_zero, consume}, 0);
  check({ds_read, ds_args, exec_zero, exec_all, consume}, 1);
  check({exec_zero, 0xdc500000, 0x05000002, exec_all, consume}, 0); // flat_load_dword
  check({ds_read, ds_args, exec_zero, 0x7e0e0505}, 1);              // v_readfirstlane_b32 s7,v5
  check({ds_read, ds_args, exec_zero, 0xd2890007, 0x00010105}, 1);  // v_readlane_b32 s7,v5,0
  check({ds_read, ds_args, exec_zero, 0xd28a0005, 0x00010007}, 1);  // v_writelane_b32 v5,s7,0
  check({exec_zero, 0xc00201c0, 0, 0xbe880007}, 1); // s_load_dword s7; s_mov_b32 s8,s7
}

TEST(WaitcheckTest, RelativeScalarDestinationCanInvalidateVccBranchFacts) {
  for (bool clobber : {false, true}) {
    for (bool waited : {false, true}) {
      SCOPED_TRACE(testing::Message() << clobber << '/' << waited);
      std::vector<uint32_t> words{
          0xf4080100,
          0xf8000000, // s_load_b128 s[4:7], s[0:1], 0
          0xbe8200c1,
          0x8b6a027e, // s2 = -1; vcc_lo = exec_lo & s2
      };
      if (clobber)
        words.insert(words.end(), {0xbefd0086, 0xbee44280}); // m0 = 6; s[100 + m0] = 0
      if (waited)
        words.push_back(0xbf89fc07); // lgkmcnt(0)
      words.insert(words.end(), {0xbfa40001, 0xbe850080, 0xbf89fc07, 0xbfb00000});
      const auto image =
          waitcheck_test::make_gfx_code_object(words, EF_AMDGPU_MACH_AMDGCN_GFX1100, true);
      AmdGpuCodeObject object(image.data(), image.size());
      const auto report = analyze_waitcnts(object, ROCJITSU_CODE_ARCH_RDNA3);
      ASSERT_TRUE(report.supported && report.complete) << report.analysis_error;
      ASSERT_EQ(report.diagnostics.size(), clobber && !waited) << diagnostic_summary(report);
      if (!report.diagnostics.empty())
        EXPECT_EQ(report.diagnostics.front().reg, (RegisterRef{RegClass::SGPR, 5, 1}));
    }
  }
}

TEST(WaitcheckTest, Wave32KernelMasksDistinguishReachableScalarLoadOverwrites) {
  // A real kernel descriptor gives the entry EXEC invariant. Both branch
  // forms skip the overwriting instruction when their mask is nonzero.
  for (bool use_vcc : {false, true}) {
    for (bool clobber : {false, true}) {
      SCOPED_TRACE(testing::Message() << "VCC=" << use_vcc << " clobber=" << clobber);
      std::vector<uint32_t> program = {0xf4004100, 0xf8000100}; // s_load_b128 s[4:7], s[0:1], 256
      if (use_vcc) {
        program.push_back(0xbe8200c1); // s_mov_b32 s2, -1
        if (clobber)
          program.push_back(0xbe820003); // s_mov_b32 s2, s3
        program.push_back(0x8b6a027e);   // s_and_b32 vcc_lo, exec_lo, s2
      } else if (clobber) {
        program.push_back(0xbefe0003); // s_mov_b32 exec_lo, s3
      }
      program.push_back(use_vcc ? 0xbfa40001u : 0xbfa60001u); // s_cbranch_{vcc,exec}nz +1
      program.push_back(0xbe850080);                          // s_mov_b32 s5, 0
      program.push_back(0xbfc70000);                          // s_wait_kmcnt 0
      program.push_back(0xbfb00000);                          // s_endpgm
      const auto image =
          waitcheck_test::make_gfx_code_object(program, EF_AMDGPU_MACH_AMDGCN_GFX1250, true);
      AmdGpuCodeObject object(image.data(), image.size());
      auto report = analyze_waitcnts(object, ROCJITSU_CODE_ARCH_CDNA5);
      ASSERT_TRUE(report.supported && report.complete) << report.analysis_error;
      EXPECT_EQ(report.diagnostics.size(), clobber ? 1u : 0u) << diagnostic_summary(report);
    }
  }
}

TEST(WaitcheckTest, GuardedLoadKeepsScalarFullDrainProofWhenLaneSearchIsIncomplete) {
  for (const bool rewrite_predicate : {false, true}) {
    for (const uint32_t prior_wait : {0u, 1u}) {
      for (const uint32_t final_wait : {0u, 1u}) {
        SCOPED_TRACE(testing::Message()
                     << rewrite_predicate << '/' << prior_wait << '/' << final_wait);
        std::vector<uint32_t> words;
        // Independent VCC diamonds create many lane states with the same
        // scalar decision about whether the producer bypasses the full drain.
        for (uint32_t reg = 4; reg < 20; ++reg) {
          words.push_back(0xbeea0028);               // s_mov_b32 vcc_lo, s40
          words.push_back(0xbfa30001);               // s_cbranch_vccz +1
          words.push_back(0xbe800080 | (reg << 16)); // s_mov_b32 sN, 0
          words.push_back(0x8b6a007e | (reg << 8));  // s_and_b32 vcc_lo, exec_lo, sN
        }
        words.push_back(0xbefe0029);               // s_mov_b32 exec_lo, s41
        append_inst(words, s_cmp_eq_u32(22, 129)); // s22 == 1
        append_inst(words, sopp(34, 3));           // skip load on equality
        append_inst(words, global_load_b32(4));
        if (rewrite_predicate)
          append_inst(words, s_mov_b32(22, 129));
        append_inst(words, s_cmp_eq_u32(22, 129));
        append_inst(words, sopp(34, 1)); // skip prior wait on equality
        append_inst(words, sopp(64, prior_wait));
        append_inst(words, sopp(64, final_wait));
        append_inst(words, v_mov_b32(2, 4));
        append_inst(words, s_endpgm());
        const auto image =
            waitcheck_test::make_gfx_code_object(words, EF_AMDGPU_MACH_AMDGCN_GFX1250, true);
        AmdGpuCodeObject object(image.data(), image.size());
        const auto report = analyze_waitcnts(object, ROCJITSU_CODE_ARCH_CDNA5);
        ASSERT_TRUE(report.supported && report.complete) << report.analysis_error;
        const bool hazard = final_wait != 0 && (prior_wait != 0 || rewrite_predicate);
        EXPECT_EQ(report.diagnostics_observed, hazard ? 1u : 0u) << diagnostic_summary(report);
      }
    }
  }
}

TEST(WaitcheckTest, Wave32ExecDerivedSccGuardProvesOnlyFeasibleWaitPaths) {
  constexpr uint32_t load = 0xdc520000, load_args = 0x017c0004; // global_load_b32 v1
  constexpr uint32_t minus_one = 0xbe8200c1, zero = 0xbe820080;
  constexpr uint32_t unknown = 0xbe820003; // s_mov_b32 s2,s3
  constexpr uint32_t and_scc = 0x8b007e02; // s_and_b32 s0,s2,exec_lo
  constexpr uint32_t and_vcc = 0x8b6a087e; // s_and_b32 vcc_lo,exec_lo,s8
  constexpr uint32_t branch_scc_zero = 0xbfa10001, branch_vcc_zero = 0xbfa30001;
  constexpr uint32_t full_wait = 0xbf8903f7, weak_wait = 0xbf8907f7;
  constexpr uint32_t overwrite = 0x7e020280, end = 0xbfb00000; // v_mov_b32 v1,0
  const auto check = [&](std::vector<uint32_t> program, bool expected_waw) {
    const auto image =
        waitcheck_test::make_gfx_code_object(program, EF_AMDGPU_MACH_AMDGCN_GFX1100, true);
    AmdGpuCodeObject object(image.data(), image.size());
    const auto report = analyze_waitcnts(object, ROCJITSU_CODE_ARCH_RDNA3);
    ASSERT_TRUE(report.supported && report.complete) << report.analysis_error;
    const bool waw = std::ranges::any_of(report.diagnostics, [](const auto &diag) {
      return diag.reg == RegisterRef{RegClass::VGPR, 1, 1} &&
             diag.access == WaitcheckAccessKind::Def;
    });
    EXPECT_EQ(waw, expected_waw) << diagnostic_summary(report);
  };

  const std::vector<uint32_t> guarded{load,      load_args, minus_one, and_scc, branch_scc_zero,
                                      full_wait, overwrite, end};
  check(guarded, false);
  auto zero_mask = guarded;
  zero_mask[2] = zero;
  check(zero_mask, true);
  auto unknown_mask = guarded;
  unknown_mask[2] = unknown;
  check(unknown_mask, true);
  auto weakened_wait = guarded;
  weakened_wait[5] = weak_wait;
  check(weakened_wait, true);

  auto changed_exec = guarded;
  changed_exec.insert(changed_exec.begin() + 3, 0xbefe0003); // s_mov_b32 exec_lo,s3
  changed_exec.insert(changed_exec.end() - 2, 0xbefe00c1);   // restore active EXEC
  check(changed_exec, true);

  auto deferred_scalar = guarded;
  deferred_scalar.insert(deferred_scalar.begin(), {0xf4000082, 0xf8000008}); // s_load_b32 s2
  check(deferred_scalar, true);
  auto unrelated_scalar = guarded;
  unrelated_scalar.insert(unrelated_scalar.begin(), {0xf4000100, 0xf8000008}); // s_load_b32 s4
  check(unrelated_scalar, false);

  auto pending_message = guarded;
  pending_message.insert(pending_message.begin() + 2, 0xbe824c01); // s_sendmsg_rtn_b32 s2
  check(pending_message, true);
  pending_message.insert(pending_message.begin() + 3, 0xbf89fc07); // lgkmcnt(0)
  check(pending_message, false);

  check({0xbf06c102, load, load_args, 0xf4000080, 0xf8000008, 0xbf06c102, 0xbfa20001, end,
         0xbf89fc07, and_scc, branch_scc_zero, full_wait, overwrite, end},
        true); // A compare of a pending scalar result cannot constrain its value after a wait.

  auto overwritten_scc = guarded;
  overwritten_scc.insert(overwritten_scc.begin() + 4, 0x8c068080); // s_or_b32 s6,0,0
  check(overwritten_scc, true);

  check({load, load_args, 0xbfa10002, full_wait, 0xbfa00001, 0xbf900002, overwrite, end},
        false); // Both branch arms drain or terminate.

  auto vcc_guard = guarded;
  vcc_guard[2] = 0xbe880080; // s_mov_b32 s8,0
  vcc_guard[3] = and_vcc;
  vcc_guard[4] = branch_vcc_zero;
  check(vcc_guard, true);
  vcc_guard[2] = 0xbe8800c1; // s_mov_b32 s8,-1
  check(vcc_guard, false);

  check({0xbefe00c1, load, load_args, 0xbeea00c1, 0xf4001a82, 0xf8000008, 0xbfa30004, 0xbf89fc07,
         0xbfa30001, full_wait, overwrite, end},
        true); // A pending VCC result may change after an earlier VCC branch.
}

TEST(WaitcheckTest, Wave32PendingSccResultKeepsUnsafeWaitPath) {
  const auto check = [&](std::vector<uint32_t> program, bool expected_waw) {
    const auto image =
        waitcheck_test::make_gfx_code_object(program, EF_AMDGPU_MACH_AMDGCN_GFX1201, true);
    AmdGpuCodeObject object(image.data(), image.size());
    const auto report = analyze_waitcnts(object, ROCJITSU_CODE_ARCH_RDNA4);
    ASSERT_TRUE(report.supported && report.complete) << report.analysis_error;
    const bool waw = std::ranges::any_of(report.diagnostics, [](const auto &diag) {
      return diag.reg == RegisterRef{RegClass::VGPR, 1, 1} &&
             diag.access == WaitcheckAccessKind::Def;
    });
    EXPECT_EQ(waw, expected_waw) << diagnostic_summary(report);
  };
  std::vector<uint32_t> program{0xee050000, 0x00000001, 0x00000008,  // global_load_b32 v1
                                0xbe804f80,                          // s_barrier_signal_isfirst 0
                                0xbe8200c1, 0x8b007e02, 0xbfa10001,  // s2=-1; SCC=EXEC&s2; branch
                                0xbfc00000, 0x7e020280, 0xbfb00000}; // loadcnt(0); overwrite; end
  check(program, true);
  program.insert(program.begin() + 4, 0xbfc70000); // kmcnt(0) drains the pending SCC write
  check(program, false);

  check({0xbe820003, 0xbf06c102, 0xee050000, 0x00000001, 0x00000008, 0xbe804f80, 0xbf06c102,
         0xbfa10005, 0xbfc70000, 0x8b007e02, 0xbfa10001, 0xbfc00000, 0x7e020280, 0xbfb00000},
        true); // A pending SCC return can invalidate a repeated comparison's branch fact.
}

TEST(WaitcheckTest, Wave32PathStateLimitKeepsMissingWait) {
  std::vector<uint32_t> program{0xbefe00c1, 0xdc520000, 0x017c0004}; // EXEC=-1; global_load v1
  program.insert(program.end(), 200001, 0xbf800000);                 // s_nop 0
  program.insert(program.end(), {0x7e020280, 0xbfb00000});           // overwrite v1; end
  const auto image =
      waitcheck_test::make_gfx_code_object(program, EF_AMDGPU_MACH_AMDGCN_GFX1100, true);
  AmdGpuCodeObject object(image.data(), image.size());
  const auto report = analyze_waitcnts(object, ROCJITSU_CODE_ARCH_RDNA3);
  ASSERT_TRUE(report.supported && report.complete) << report.analysis_error;
  EXPECT_TRUE(std::ranges::any_of(report.diagnostics, [](const auto &diag) {
    return diag.reg == RegisterRef{RegClass::VGPR, 1, 1} && diag.access == WaitcheckAccessKind::Def;
  })) << diagnostic_summary(report);
}

TEST(WaitcheckTest, UnrelatedScalarExecAndKeepsGuardedWaitProof) {
  std::vector<uint32_t> program{0xbe872a64}; // s_movrels_b32 s7,s100; dynamic scalar read
  for (uint32_t reg = 4; reg < 90; ++reg) {
    program.push_back(0xbeea0003);                 // s_mov_b32 vcc_lo,s3: unknown condition
    program.push_back(0xbfa30001);                 // s_cbranch_vccz +1
    program.push_back(0xbe800080 | (reg << 16));   // s_mov_b32 sN,0 on one branch
    program.push_back(0x8b00007e | (reg << 8));    // s_and_b32 s0,exec_lo,sN; SCC is unused
  }
  append_inst(program, s_cmp_eq_u32(2, 128)); // An unknown backedge revisits the diamonds.
  const int16_t backedge = 1 - static_cast<int>(program.size() + 1);
  program.push_back(0xbfa20000u | static_cast<uint16_t>(backedge));
  append_inst(program, s_cmp_eq_u32(22, 129)); // The repeated comparison gates the load.
  append_inst(program, sopp(34, 3));
  program.insert(program.end(), {0xdc520000, 0x047c0004}); // global_load_b32 v4,v[4:5]
  append_inst(program, s_cmp_eq_u32(22, 129));
  append_inst(program, sopp(34, 1));
  program.push_back(0xbf8903f7); // s_waitcnt vmcnt(0) drains every issued load.
  program.push_back(0x7e040304); // v_mov_b32 v2,v4
  append_inst(program, s_endpgm());
  const auto image =
      waitcheck_test::make_gfx_code_object(program, EF_AMDGPU_MACH_AMDGCN_GFX1100, true);
  AmdGpuCodeObject object(image.data(), image.size());
  const auto report = analyze_waitcnts(object, ROCJITSU_CODE_ARCH_RDNA3);
  ASSERT_TRUE(report.supported && report.complete) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, ObjectAnalysisIgnoresBranchInfeasibleSkippedWaitPath) {
  std::vector<uint32_t> program;
  append_inst(program, s_cmp_eq_u32(0, 129)); // s0 == 1
  append_inst(program, sopp(34, 2));          // s_cbranch_scc1 over the load
  append_inst(program, global_load_b32(0));   // load executes only when s0 != 1
  append_inst(program, s_cmp_eq_u32(0, 129)); // s0 == 1
  append_inst(program, sopp(34, 1));          // s_cbranch_scc1 over the wait
  append_inst(program, sopp(64, 0));          // s_wait_loadcnt 0
  append_inst(program, v_mov_b32(1, 0));

  TestCodeObject code_object(program);
  auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, CounterParityRequiresPendingLoadAtActiveConsumer) {
  for (const uint32_t machine : {EF_AMDGPU_MACH_AMDGCN_GFX1150,
                                 EF_AMDGPU_MACH_AMDGCN_GFX1151}) {
    for (const bool active_consumer : {false, true}) {
      SCOPED_TRACE(machine);
      SCOPED_TRACE(active_consumer);
      std::vector<uint32_t> code;
      append_inst(code, s_cmp_eq_u32(0, 129));
      append_inst(code, sopp(34, 2)); // Skip the two-word load on one scalar path.
      code.insert(code.end(), {0xdc520000u, 0x007c0004u}); // global_load_b32 v0,v[4:5]
      append_inst(code, s_mov_b32(126, active_consumer ? 129 : 128));
      const auto wait = code.size() * sizeof(uint32_t);
      code.push_back(0xbf8903f7u); // VMcnt(0).
      code.push_back(0x7e000280u); // v_mov_b32 v0,0: overwrite, with no source use.
      append_inst(code, s_endpgm());

      const auto bytes = waitcheck_test::make_gfx_code_object(code, machine, true);
      const AmdGpuCodeObject object(bytes.data(), bytes.size());
      WaitcheckOptions options;
      options.check_counter_parity = true;
      const auto report = analyze_waitcnts_for_kernel(object, ROCJITSU_CODE_ARCH_RDNA3_5, 0,
                                                      options);
      ASSERT_TRUE(report.supported && report.complete) << report.analysis_error;
      EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
      EXPECT_GE(report.counter_parity_fields_checked, 1u);
      EXPECT_EQ(report.counter_parity_exact, active_consumer ? 1u : 0u);
      EXPECT_EQ(report.counter_parity_indeterminate_groups, active_consumer ? 0u : 1u);
      EXPECT_FALSE(std::ranges::any_of(report.counter_underaccounting_diagnostics,
                                       [&](const auto &row) {
                                         return row.wait_section_offset == wait;
                                       }));
    }
  }
}

TEST(WaitcheckTest, ObjectAnalysisPreservesInfeasiblePathWithReachabilityCacheDisabled) {
  std::vector<uint32_t> program;
  append_inst(program, s_cmp_eq_u32(0, 129)); // s0 == 1
  append_inst(program, sopp(34, 2));          // s_cbranch_scc1 over the load
  append_inst(program, global_load_b32(0));   // load executes only when s0 != 1
  append_inst(program, s_cmp_eq_u32(0, 129)); // s0 == 1
  append_inst(program, sopp(34, 1));          // s_cbranch_scc1 over the wait
  append_inst(program, sopp(64, 0));          // s_wait_loadcnt 0
  append_inst(program, v_mov_b32(1, 0));

  TestCodeObject code_object(program);
  WaitcheckOptions options;
  options.max_reachability_cache_bytes = 0;
  auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_RDNA4, options);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, ObjectAnalysisPreservesFeasiblePathWithReachabilityCacheDisabled) {
  std::vector<uint32_t> program;
  append_inst(program, global_load_b32(0));
  append_inst(program, sopp(33, 1)); // s_cbranch_scc0 over the wait
  append_inst(program, sopp(64, 0)); // s_wait_loadcnt 0 on fallthrough only
  append_inst(program, v_mov_b32(1, 0));

  TestCodeObject code_object(program);
  WaitcheckOptions options;
  options.max_reachability_cache_bytes = 0;
  auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_RDNA4, options);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Load);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Use);
  EXPECT_EQ(report.diagnostics[0].reg.index, 0u);
}

TEST(WaitcheckTest, ObjectAnalysisCorrelatesUnsignedRangeChecksAroundWait) {
  std::vector<uint32_t> program;
  append_inst(program, s_cmp_ge_u32(0, 129)); // s0 >= 1
  append_inst(program, sopp(33, 2));          // s_cbranch_scc0 over the load
  append_inst(program, global_load_b32(0));   // load executes only when s0 >= 1
  append_inst(program, s_cmp_ge_u32(0, 129)); // s0 >= 1
  append_inst(program, sopp(33, 1));          // s_cbranch_scc0 over the wait
  append_inst(program, sopp(64, 0));          // s_wait_loadcnt 0
  append_inst(program, v_mov_b32(1, 0));

  TestCodeObject code_object(program);
  auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, ObjectAnalysisPreservesPredicatesAndConsumersAcrossNeutralRuns) {
  for (bool rewrite : {false, true}) {
    for (unsigned padding : {0u, 64u}) {
      SCOPED_TRACE(::testing::Message() << "rewrite=" << rewrite << " padding=" << padding);
      std::vector<uint32_t> program;
      auto neutral_run = [&] {
        for (unsigned i = 0; i < padding; ++i)
          append_inst(program, v_mov_b32(7, 6));
      };
      append_inst(program, s_cmp_eq_u32(0, 129)); // s0 == 1
      append_inst(program, sopp(34, 2));          // skip load when s0 == 1
      append_inst(program, global_load_b32(0));
      neutral_run();
      if (rewrite)
        append_inst(program, s_mov_b32(0, 129)); // invalidate the load's s0 != 1 fact
      neutral_run();
      append_inst(program, s_cmp_eq_u32(0, 129));
      append_inst(program, sopp(34, 1)); // skip wait when s0 == 1
      append_inst(program, sopp(64, 0)); // s_wait_loadcnt 0
      neutral_run();
      std::vector<uint64_t> consumer_offsets;
      for (unsigned dst : {1u, 2u}) {
        consumer_offsets.push_back(program.size() * sizeof(uint32_t));
        append_inst(program, v_mov_b32(dst, 0));
        neutral_run();
      }

      TestCodeObject code_object(program);
      const auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_RDNA4);
      ASSERT_TRUE(report.supported) << report.analysis_error;
      ASSERT_EQ(report.diagnostics.size(), rewrite ? consumer_offsets.size() : 0u)
          << diagnostic_summary(report);
      if (rewrite) {
        for (size_t i = 0; i < consumer_offsets.size(); ++i) {
          EXPECT_EQ(report.diagnostics[i].section_offset, consumer_offsets[i]);
          EXPECT_EQ(report.diagnostics[i].reg.index, 0u);
          EXPECT_EQ(report.diagnostics[i].access, WaitcheckAccessKind::Use);
        }
      }
    }
  }
}

TEST(WaitcheckTest, CorrelatedBranchesPreserveOutOfOrderScalarLoadsAtPartialWait) {
  for (uint32_t count : {0u, 1u}) {
    std::vector<uint32_t> program;
    append_inst(program, s_cmp_eq_u32(10, 129));
    append_inst(program, sopp(34, 4));
    append_inst(program, s_load_b32(4, 16));
    append_inst(program, s_load_b32(5, 16));
    append_inst(program, s_cmp_eq_u32(10, 129));
    append_inst(program, sopp(34, 1));
    append_inst(program, sopp(71, count));
    append_inst(program, s_mov_b32(6, 4));
    TestCodeObject object(program);
    const auto report = analyze_waitcnts(object, ROCJITSU_CODE_ARCH_RDNA4);
    ASSERT_TRUE(report.supported);
    EXPECT_EQ(report.diagnostics_observed, count);
  }
}

TEST(WaitcheckTest, CorrelatedBranchesRetireOrderedLoadsAtPartialWait) {
  for (uint32_t count : {0u, 1u, 2u}) {
    std::vector<uint32_t> program;
    append_inst(program, s_cmp_eq_u32(10, 129));
    append_inst(program, sopp(34, 2 * sizeof(rdna4::VglobalMachineInst) / sizeof(uint32_t)));
    append_inst(program, global_load_b32(0));
    append_inst(program, global_load_b32(1));
    append_inst(program, s_cmp_eq_u32(10, 129));
    append_inst(program, sopp(34, 1));
    append_inst(program, sopp(64, count));
    append_inst(program, v_mov_b32(2, 0));
    TestCodeObject object(program);
    const auto report = analyze_waitcnts(object, ROCJITSU_CODE_ARCH_RDNA4);
    ASSERT_TRUE(report.passed() || count == 2) << report.analysis_error;
    EXPECT_EQ(report.diagnostics_observed, count == 2 ? 1u : 0u);
  }
}

TEST(WaitcheckTest, CorrelatedBranchesRequireBothFlatSourceRoutes) {
  for (const bool wait_for_ds : {false, true}) {
    std::vector<uint32_t> program;
    append_inst(program, s_setreg_imm32_b32(hwreg(26, 0, 2), 2)); // Enable expert scheduling.
    append_inst(program, s_cmp_eq_u32(10, 129));
    append_inst(program, sopp(34, sizeof(rdna4::VflatMachineInst) / sizeof(uint32_t)));
    append_inst(program, flat_load_b32(0));
    append_inst(program, s_cmp_eq_u32(10, 129));
    append_inst(program, sopp(34, 1));
    append_inst(program, sopp(wait_for_ds ? 72 : 64, 0));
    append_inst(program, v_mov_b32(8, 7));
    append_inst(program, s_endpgm());
    const auto image = rocjitsu::waitcheck_test::make_gfx_multi_kernel_code_object(
        {{"flat_source", program}}, EF_AMDGPU_MACH_AMDGCN_GFX1201);
    AmdGpuCodeObject object(image.data(), image.size());
    const auto report = analyze_waitcnts(object, ROCJITSU_CODE_ARCH_RDNA4);
    ASSERT_TRUE(report.complete) << report.analysis_error;
    EXPECT_EQ(report.diagnostics_observed, wait_for_ds ? 0u : 1u);
  }
}

TEST(WaitcheckTest, PredicateChangesBetweenCompareAndBranchKeepHazardReachable) {
  for (bool overwrite_scc : {false, true}) {
    std::vector<uint32_t> program;
    append_inst(program, s_cmp_eq_u32(0, 129));
    if (!overwrite_scc)
      append_inst(program, s_mov_b32(0, 129));
    append_inst(program, sopp(34, 2));
    append_inst(program, global_load_b32(0));
    append_inst(program, s_cmp_eq_u32(0, 129));
    if (overwrite_scc)
      program.push_back(0x8c068180); // s_or_b32 s6, 0, 1
    append_inst(program, sopp(34, 1));
    append_inst(program, sopp(64, 0));
    append_inst(program, v_mov_b32(1, 0));
    TestCodeObject object(program);
    const auto report = analyze_waitcnts(object, ROCJITSU_CODE_ARCH_RDNA4);
    ASSERT_TRUE(report.supported) << report.analysis_error;
    EXPECT_EQ(report.diagnostics_observed, 1u) << overwrite_scc;
  }
}

TEST(WaitcheckTest, ObjectAnalysisKeepsPathFeasibleAfterScalarRewrite) {
  std::vector<uint32_t> program;
  append_inst(program, global_load_b32(0));
  append_inst(program, s_cmp_eq_u32(0, 129)); // s0 == 1
  append_inst(program, sopp(34, 3));          // s_cbranch_scc1 to the wait
  append_inst(program, s_mov_b32(0, 129));    // redefine s0, invalidating prior path fact
  append_inst(program, s_cmp_eq_u32(0, 129)); // s0 == 1
  append_inst(program, sopp(34, 1));          // s_cbranch_scc1 over the wait
  append_inst(program, sopp(64, 0));          // s_wait_loadcnt 0
  append_inst(program, v_mov_b32(1, 0));

  TestCodeObject code_object(program);
  auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Load);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Use);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::VGPR);
  EXPECT_EQ(report.diagnostics[0].reg.index, 0u);
}

TEST(WaitcheckTest, ObjectAnalysisRequiresZeroWaitForMixedLoadOrderAtJoin) {
  std::vector<uint32_t> program;
  append_inst(program, sopp(33, 7)); // s_cbranch_scc0 to the else path
  append_inst(program, global_load_b32(0));
  append_inst(program, global_load_b32(1));
  append_inst(program, sopp(32, 6)); // s_branch to join
  append_inst(program, global_load_b32(1));
  append_inst(program, global_load_b32(0));
  append_inst(program, sopp(64, 1)); // one predecessor still has v0 as newest load
  append_inst(program, v_mov_b32(2, 0));

  TestCodeObject code_object(program);
  auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported);
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  for (const auto &diag : report.diagnostics) {
    EXPECT_EQ(diag.counter, WaitCounterKind::Load);
    EXPECT_EQ(diag.access, WaitcheckAccessKind::Use);
    EXPECT_EQ(diag.reg.cls, RegClass::VGPR);
    EXPECT_EQ(diag.reg.index, 0u);
    EXPECT_EQ(diag.required_count, 0u);
  }
}

TEST(WaitcheckTest, ObjectAnalysisAcceptsConsistentOlderLoadAtJoinAfterPartialWait) {
  std::vector<uint32_t> program;
  append_inst(program, sopp(33, 7)); // s_cbranch_scc0 to the else path
  append_inst(program, global_load_b32(0));
  append_inst(program, global_load_b32(1));
  append_inst(program, sopp(32, 6)); // s_branch to join
  append_inst(program, global_load_b32(0));
  append_inst(program, global_load_b32(1));
  append_inst(program, sopp(64, 1)); // both predecessors have v0 older than v1
  append_inst(program, v_mov_b32(2, 0));

  TestCodeObject code_object(program);
  auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, ObjectAnalysisAcceptsZeroWaitForMixedLoadOrderAtJoin) {
  std::vector<uint32_t> program;
  append_inst(program, sopp(33, 7)); // s_cbranch_scc0 to the else path
  append_inst(program, global_load_b32(0));
  append_inst(program, global_load_b32(1));
  append_inst(program, sopp(32, 6)); // s_branch to join
  append_inst(program, global_load_b32(1));
  append_inst(program, global_load_b32(0));
  append_inst(program, sopp(64, 0)); // s_wait_loadcnt 0
  append_inst(program, v_mov_b32(2, 0));

  TestCodeObject code_object(program);
  auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty());
}

TEST(WaitcheckTest, ObjectAnalysisKeepsCdnaPendingDestinationAcrossJoin) {
  for (rj_code_arch_t arch : {ROCJITSU_CODE_ARCH_CDNA3, ROCJITSU_CODE_ARCH_CDNA4}) {
    SCOPED_TRACE(arch);
    for (bool wait_at_join : {false, true}) {
      SCOPED_TRACE(wait_at_join);
      std::vector<uint32_t> program;
      append_gfx942_v_mov_b32_v4_v2(program);
      append_gfx942_ds_read_b32_v4_v0(program);
      program.push_back(0xbf850001u); // s_cbranch_scc1 skips the overwrite
      append_gfx942_v_mov_b32_v4_v2(program);
      if (wait_at_join)
        append_gfx942_s_waitcnt_lgkmcnt_0(program);
      append_gfx942_v_mov_b32_v5_v4(program);
      program.push_back(0xbf810000u); // s_endpgm
      TestCodeObject code_object(program);
      auto report = analyze_waitcnts(code_object, arch);

      ASSERT_TRUE(report.supported) << report.analysis_error;
      ASSERT_EQ(report.diagnostics.size(), wait_at_join ? 1u : 2u) << diagnostic_summary(report);
      EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Def);
      EXPECT_EQ(report.diagnostics[0].section_offset, 16u);
      if (!wait_at_join) {
        EXPECT_EQ(report.diagnostics[1].access, WaitcheckAccessKind::Use);
        EXPECT_EQ(report.diagnostics[1].section_offset, 20u);
      }
      for (const auto &diagnostic : report.diagnostics) {
        EXPECT_EQ(diagnostic.counter, WaitCounterKind::Ds);
        EXPECT_EQ(diagnostic.reg, (RegisterRef{RegClass::VGPR, 4, 1}));
        EXPECT_EQ(diagnostic.producer_section_offset, 4u);
        EXPECT_EQ(diagnostic.required_count, 0u);
      }
    }
  }
}

TEST(WaitcheckTest, ObjectAnalysisConvergesWithRepeatedRegisterLoadsAroundLoop) {
  for (rj_code_arch_t arch : {ROCJITSU_CODE_ARCH_CDNA3, ROCJITSU_CODE_ARCH_CDNA4}) {
    SCOPED_TRACE(arch);
    for (bool wait_before_use : {false, true}) {
      SCOPED_TRACE(wait_before_use);
      // Reduced from the Tensile GEMM in ROCm/aorta#453. Repeated v0 loads
      // reach the loop through two pre-loop joins. Analysis must converge
      // without losing pending physical destinations as the joins are revisited.
      std::vector<uint32_t> program{
          0xbf850002u,              // s_cbranch_scc1 to the wait
          0xe0501000u, 0x80000008u, // buffer_load_dword v0, v8, s[0:3], 0 offen
          0xbf8c0f70u,              // s_waitcnt vmcnt(0)
          0xbf060100u,              // s_cmp_eq_u32 s0, s1: independent load condition
          0xbf850002u,              // s_cbranch_scc1 to loop header
          0xe0501000u, 0x80000008u, // buffer_load_dword v0, v8, s[0:3], 0 offen
          0xbf060302u,              // loop header: s_cmp_eq_u32 s2, s3
          0xbf850002u,              // s_cbranch_scc1 to exit
          0xbf850001u,              // s_cbranch_scc1 to exit
          0xbf84fffcu,              // s_cbranch_scc0 to loop header
      };
      // A fresh result at the exit must still be diagnosed unless waited for.
      // Convergence must not hide pending events or stop diagnostic emission.
      append_gfx942_buffer_load_dword(program, /*vdst=*/3, /*vaddr=*/8);
      if (wait_before_use)
        append_gfx942_s_waitcnt_vmcnt_0(program);
      program.push_back(0x7e020300u); // v_mov_b32 v1, v0: observe the loop-carried fact
      program.push_back(0x7e020303u); // v_mov_b32 v1, v3
      program.push_back(0xbf810000u); // s_endpgm
      TestCodeObject code_object(program);
      WaitcheckKernelInfo kernel;
      kernel.entry_offset = 0;
      kernel.code_size = program.size() * sizeof(uint32_t);
      kernel.wavefront_size = 64;
      auto report = analyze_waitcnts_for_kernel(code_object, arch, kernel);

      ASSERT_TRUE(report.supported) << report.analysis_error;
      if (wait_before_use) {
        EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
      } else {
        ASSERT_EQ(report.diagnostics.size(), 2u) << diagnostic_summary(report);
        for (const auto &diagnostic : report.diagnostics) {
          EXPECT_EQ(diagnostic.counter, WaitCounterKind::Load);
          EXPECT_EQ(diagnostic.access, WaitcheckAccessKind::Use);
        }
        EXPECT_EQ(report.diagnostics[0].reg, (RegisterRef{RegClass::VGPR, 0, 1}));
        EXPECT_EQ(report.diagnostics[1].reg, (RegisterRef{RegClass::VGPR, 3, 1}));
      }
    }
  }
}

TEST(WaitcheckTest, ObjectAnalysisReportsLoopCarriedDsLoadUse) {
  std::vector<uint32_t> program;
  append_inst(program, v_mov_b32(1, 0));
  append_inst(program, ds_load_b32(0, 4));
  append_inst(program, sopp(34, static_cast<uint16_t>(-4))); // s_cbranch_scc1 to loop header

  TestCodeObject code_object(program);
  auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported);
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics.front().counter, WaitCounterKind::Ds);
  EXPECT_EQ(report.diagnostics.front().reg, (RegisterRef{RegClass::VGPR, 0, 1}));
  EXPECT_EQ(report.diagnostics.front().access, WaitcheckAccessKind::Use);
}

TEST(WaitcheckTest, ObjectAnalysisRetainsEachLoopCarriedUseAcrossNeutralRuns) {
  for (bool wait : {false, true}) {
    SCOPED_TRACE(wait);
    std::vector<uint32_t> program;
    if (wait)
      append_inst(program, sopp(70, 0)); // s_wait_dscnt 0 at loop header
    auto neutral_run = [&] {
      for (unsigned i = 0; i < 32; ++i)
        append_inst(program, v_mov_b32(7, 6));
    };
    neutral_run();
    for (unsigned dst : {1u, 2u}) {
      append_inst(program, v_mov_b32(dst, 0));
      neutral_run();
    }
    append_inst(program, ds_load_b32(0, 4));
    const int32_t backedge = -static_cast<int32_t>(program.size() + 1);
    append_inst(program, sopp(34, static_cast<uint16_t>(backedge)));

    TestCodeObject code_object(program);
    const auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_RDNA4);
    ASSERT_TRUE(report.supported) << report.analysis_error;
    ASSERT_EQ(report.diagnostics.size(), wait ? 0u : 2u) << diagnostic_summary(report);
    if (!wait) {
      EXPECT_EQ(std::ranges::count_if(report.diagnostics,
                                      [](const auto &diag) {
                                        return diag.access == WaitcheckAccessKind::Use &&
                                               diag.reg.index == 0;
                                      }),
                2);
    }
  }
}

TEST(WaitcheckTest, ObjectAnalysisAcceptsLoopCarriedDsLoadUseAfterWait) {
  std::vector<uint32_t> program;
  append_inst(program, sopp(70, 0)); // s_wait_dscnt 0 at loop header
  append_inst(program, v_mov_b32(1, 0));
  append_inst(program, ds_load_b32(0, 4));
  append_inst(program, sopp(34, static_cast<uint16_t>(-5))); // s_cbranch_scc1 to loop header

  TestCodeObject code_object(program);
  auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty());
}

TEST(WaitcheckTest, DescriptorEntryAnalysisIgnoresPaddingAfterEndpgm) {
  const auto image = rocjitsu::waitcheck_test::make_gfx1200_padded_code_object();
  AmdGpuCodeObject code_object(image.data(), image.size());
  ASSERT_TRUE(code_object.is_valid());

  auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty());
  EXPECT_EQ(report.instructions_analyzed, 2u);
}

TEST(WaitcheckTest, DescriptorEntryAnalysisStopsAtUnterminatedFunctionEnd) {
  const auto image = rocjitsu::waitcheck_test::make_gfx1250_unterminated_stub_code_object();
  AmdGpuCodeObject code_object(image.data(), image.size());
  const auto kernels = waitcheck_kernels(code_object);

  ASSERT_EQ(kernels.size(), 1u);
  EXPECT_EQ(kernels[0].code_size, 2 * sizeof(uint32_t));

  auto report = analyze_waitcnts_for_kernel(code_object, ROCJITSU_CODE_ARCH_CDNA5, kernels[0]);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
  EXPECT_EQ(report.instructions_analyzed, 1u);
}

TEST(WaitcheckTest, FunctionEntryAnalysisIgnoresInterFunctionPadding) {
  const auto image = rocjitsu::waitcheck_test::make_gfx1200_function_only_padded_code_object();
  AmdGpuCodeObject code_object(image.data(), image.size());
  ASSERT_TRUE(code_object.is_valid());

  auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Load);
  EXPECT_EQ(report.instructions_analyzed, 5u);
}

TEST(WaitcheckTest, SharedKernelEntryDiagnosticsAreReportedOnce) {
  std::vector<uint32_t> words;
  rocjitsu::waitcheck_test::append_inst(words, rocjitsu::waitcheck_test::global_load_b32(0));
  rocjitsu::waitcheck_test::append_inst(words, rocjitsu::waitcheck_test::v_mov_b32(1, 0));
  words.push_back(0xBFB00000U); // s_endpgm

  auto image = rocjitsu::waitcheck_test::make_gfx_multi_kernel_code_object(
      {{"first", words}, {"alias", words}}, EF_AMDGPU_MACH_AMDGCN_GFX1200);
  constexpr uint64_t kTextOffset = 0x100;
  constexpr uint64_t kTextVaddr = 0x1100;
  constexpr uint64_t kDescriptorSize = 64;
  constexpr uint64_t kEntryOffsetField = 16;
  const uint64_t text_size = 2 * words.size() * sizeof(uint32_t);
  const uint64_t rodata_offset = kTextOffset + text_size;
  const uint64_t rodata_vaddr = kTextVaddr + text_size + 0x1000;
  const uint64_t alias_descriptor_vaddr = rodata_vaddr + kDescriptorSize;
  const int64_t alias_entry_offset =
      static_cast<int64_t>(kTextVaddr) - static_cast<int64_t>(alias_descriptor_vaddr);
  std::memcpy(image.data() + rodata_offset + kDescriptorSize + kEntryOffsetField,
              &alias_entry_offset, sizeof(alias_entry_offset));

  AmdGpuCodeObject code_object(image.data(), image.size());
  ASSERT_TRUE(code_object.is_valid());
  const auto kernels = waitcheck_kernels(code_object);
  auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_EQ(kernels.size(), 1u);
  EXPECT_EQ(kernels[0].entry_offset, 0u);
  EXPECT_EQ(kernels[0].alias_descriptor_vaddrs, (std::vector<uint64_t>{alias_descriptor_vaddr}));
  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_EQ(report.diagnostics_observed, 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
}

TEST(WaitcheckTest, SharedCalleeDiagnosticIsAttributedToEachKernelEntry) {
  constexpr auto arch = ROCJITSU_CODE_ARCH_CDNA4;
  std::vector<uint32_t> first{
      build_s_call_b64(30, 3, arch).value(), build_s_endpgm(arch).value()};
  std::vector<uint32_t> second{
      build_s_call_b64(30, 1, arch).value(), build_s_endpgm(arch).value()};
  std::vector<uint32_t> helper;
  append_gfx950_buffer_load_dword_v0_v8_s0_offen(helper);
  helper.push_back(0x7e060300u); // v_mov_b32 v3, v0
  helper.push_back(build_sop1_encoding(arch, 0x1d, 0, 30).value()); // s_setpc_b64 s[30:31]
  auto image = waitcheck_test::make_gfx_multi_kernel_code_object(
      {{"first", first}, {"second", second}, {"helper", helper}},
      EF_AMDGPU_MACH_AMDGCN_GFX950);
  auto *ehdr = reinterpret_cast<Elf64_Ehdr *>(image.data());
  auto *shdrs = reinterpret_cast<Elf64_Shdr *>(image.data() + ehdr->e_shoff);
  auto *symbols = reinterpret_cast<Elf64_Sym *>(image.data() + shdrs[3].sh_offset);
  symbols[5] = {}; // Keep the helper STT_FUNC, but remove its dispatch descriptor.
  AmdGpuCodeObject object(image.data(), image.size());
  ASSERT_TRUE(object.is_valid());
  const auto kernels = waitcheck_kernels(object);
  ASSERT_EQ(kernels.size(), 2u);
  const auto first_report = analyze_waitcnts_for_kernel(object, arch, kernels[0]);
  const auto second_report = analyze_waitcnts_for_kernel(object, arch, kernels[1]);
  const auto batch = analyze_waitcnts_for_kernels(object, arch, kernels);
  ASSERT_TRUE(first_report.complete && second_report.complete && batch.complete)
      << batch.analysis_error;
  EXPECT_EQ(first_report.diagnostics_observed, 1u) << diagnostic_summary(first_report);
  EXPECT_EQ(second_report.diagnostics_observed, 1u) << diagnostic_summary(second_report);
  ASSERT_EQ(batch.diagnostics_observed, 2u) << diagnostic_summary(batch);
  ASSERT_EQ(batch.diagnostics.size(), 2u);
  EXPECT_NE(batch.diagnostics[0].kernel_entry_offset,
            batch.diagnostics[1].kernel_entry_offset);
  EXPECT_EQ(batch.diagnostics[0].producer_section_offset,
            batch.diagnostics[1].producer_section_offset);
}

TEST(WaitcheckTest, SharedKernelEntryPreservesDistinctWaveModes) {
  const std::vector<uint32_t> words{0xBFB00000U}; // s_endpgm.
  auto image = rocjitsu::waitcheck_test::make_gfx_multi_kernel_code_object(
      {{"wave64", words}, {"wave32", words}}, EF_AMDGPU_MACH_AMDGCN_GFX1200);
  constexpr uint64_t kTextOffset = 0x100;
  constexpr uint64_t kTextVaddr = 0x1100;
  constexpr uint64_t kDescriptorSize = 64;
  constexpr uint64_t kEntryOffsetField = 16;
  constexpr uint64_t kPropertiesField = 56;
  constexpr uint16_t kWave32 = 1u << 10u;
  const uint64_t text_size = 2 * words.size() * sizeof(uint32_t);
  const uint64_t rodata_offset = kTextOffset + text_size;
  const uint64_t rodata_vaddr = kTextVaddr + text_size + 0x1000;
  const uint64_t wave32_descriptor_vaddr = rodata_vaddr + kDescriptorSize;
  const int64_t shared_entry_offset =
      static_cast<int64_t>(kTextVaddr) - static_cast<int64_t>(wave32_descriptor_vaddr);
  std::memcpy(image.data() + rodata_offset + kDescriptorSize + kEntryOffsetField,
              &shared_entry_offset, sizeof(shared_entry_offset));
  std::memcpy(image.data() + rodata_offset + kDescriptorSize + kPropertiesField, &kWave32,
              sizeof(kWave32));

  AmdGpuCodeObject code_object(image.data(), image.size());
  ASSERT_TRUE(code_object.is_valid());
  const auto kernels = waitcheck_kernels(code_object);

  ASSERT_EQ(kernels.size(), 2u);
  EXPECT_EQ(kernels[0].entry_offset, 0u);
  EXPECT_EQ(kernels[1].entry_offset, 0u);
  EXPECT_NE(kernels[0].wavefront_size, kernels[1].wavefront_size);
  const auto report = analyze_waitcnts_for_kernel(code_object, ROCJITSU_CODE_ARCH_RDNA4, 0);
  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_EQ(report.kernels_analyzed, 2u);
}

TEST(WaitcheckTest, KnownKernelBatchReportsPerKernelCompletion) {
  std::vector<uint32_t> hazardous;
  rocjitsu::waitcheck_test::append_inst(hazardous, rocjitsu::waitcheck_test::global_load_b32(0));
  rocjitsu::waitcheck_test::append_inst(hazardous, rocjitsu::waitcheck_test::v_mov_b32(1, 0));
  hazardous.push_back(0xBFB00000U); // s_endpgm
  const std::vector<uint32_t> clean{0xBFB00000U};
  const auto image = rocjitsu::waitcheck_test::make_gfx_multi_kernel_code_object(
      {{"hazardous", hazardous}, {"clean", clean}}, EF_AMDGPU_MACH_AMDGCN_GFX1200);
  AmdGpuCodeObject code_object(image.data(), image.size());
  ASSERT_TRUE(code_object.is_valid());
  const auto kernels = waitcheck_kernels(code_object);
  ASSERT_EQ(kernels.size(), 2u);

  std::vector<std::string> completed;
  WaitcheckOptions options;
  size_t completion_count = 0;
  options.kernel_analyzed_callback = [&] { ++completion_count; };
  options.kernel_timing_callback = [&](const WaitcheckKernelInfo &kernel,
                                       std::chrono::nanoseconds elapsed) {
    EXPECT_GE(elapsed.count(), 0);
    completed.push_back(kernel.name);
  };
  const auto report =
      analyze_waitcnts_for_kernels(code_object, ROCJITSU_CODE_ARCH_RDNA4, kernels, options);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_EQ(report.kernels_discovered, 2u);
  EXPECT_EQ(report.kernels_analyzed, 2u);
  EXPECT_EQ(completion_count, 2u);
  EXPECT_EQ(completed.size(), 2u);
  EXPECT_EQ(completed[0], kernels[0].name);
  EXPECT_EQ(completed[1], kernels[1].name);
  EXPECT_EQ(report.diagnostics_observed, 1u) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx950ResolvedSwappcHelperWaitAppliesBeforeContinuation) {
  constexpr uint16_t kTargetSreg = 8;
  constexpr uint16_t kReturnSreg = 30;
  constexpr uint16_t kLiteralOperand = 255;
  constexpr uint16_t kInlineInt0 = 128;

  // The kernel leaves a VMEM result pending, builds the helper address relative
  // to s_getpc, and uses s_swappc to call it. The helper's entry wait is the only
  // wait before the continuation consumes v0. Runtime waitcheck must decode the
  // proven helper target without walking unrelated .text bytes and propagate
  // the helper state back through its matching s[30:31] return.
  std::vector<uint32_t> program;
  append_gfx950_buffer_load_dword_v0_v8_s0_offen(program);                             // 0x00.
  program.push_back(build_s_getpc_b64(kTargetSreg, ROCJITSU_CODE_ARCH_CDNA4).value()); // 0x08.
  program.push_back(build_s_add_u32(kTargetSreg, kTargetSreg, kLiteralOperand,
                                    ROCJITSU_CODE_ARCH_CDNA4)
                        .value()); // 0x0c.
  program.push_back(24);           // 0x10 -> 0x24.
  program.push_back(build_s_addc_u32(kTargetSreg + 1, kTargetSreg + 1, kInlineInt0,
                                     ROCJITSU_CODE_ARCH_CDNA4)
                        .value()); // 0x14.
  program.push_back(
      build_s_swappc_b64(kReturnSreg, kTargetSreg, ROCJITSU_CODE_ARCH_CDNA4).value()); // 0x18.
  append_gfx950_v_mov_b32_v1_v0(program);                                              // 0x1c.
  program.push_back(build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4).value());                 // 0x20.
  append_gfx950_s_waitcnt_vmcnt_0(program); // 0x24 helper.
  program.push_back(build_sop1_encoding(ROCJITSU_CODE_ARCH_CDNA4, 0x1d, 0,
                                        kReturnSreg)
                        .value()); // 0x28 return.

  const auto image =
      rocjitsu::waitcheck_test::make_gfx_code_object(program, EF_AMDGPU_MACH_AMDGCN_GFX950);
  AmdGpuCodeObject code_object(image.data(), image.size());
  auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_CDNA4);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx1201ResolvedSwappcPreservesCanonicalizedPcHighHalf) {
  constexpr uint16_t kTargetSreg = 8;
  constexpr uint16_t kReturnSreg = 30;
  constexpr uint16_t kLiteralOperand = 255;
  constexpr uint16_t kInlineInt0 = 128;

  // PyTorch's gfx1201 address builder canonicalizes the upper 16 bits of the
  // getpc result before applying its PC-relative delta. The helper entry wait
  // must still flow back to the continuation that consumes v0.
  std::vector<uint32_t> program;
  append_inst(program, global_load_b32(0));                                            // 0x00.
  program.push_back(build_s_getpc_b64(kTargetSreg, ROCJITSU_CODE_ARCH_RDNA4).value()); // 0x0c.
  program.push_back(0xBE890F09u); // 0x10: s_sext_i32_i16 s9, s9.
  program.push_back(build_s_add_u32(kTargetSreg, kTargetSreg, kLiteralOperand,
                                    ROCJITSU_CODE_ARCH_RDNA4)
                        .value()); // 0x14.
  program.push_back(28);           // Base 0x10 -> helper 0x2c.
  program.push_back(build_s_addc_u32(kTargetSreg + 1, kTargetSreg + 1, kInlineInt0,
                                     ROCJITSU_CODE_ARCH_RDNA4)
                        .value()); // 0x1c.
  program.push_back(
      build_s_swappc_b64(kReturnSreg, kTargetSreg, ROCJITSU_CODE_ARCH_RDNA4).value()); // 0x20.
  append_inst(program, v_mov_b32(1, 0));                                               // 0x24.
  append_inst(program, s_endpgm());                                                    // 0x28.
  append_inst(program, sopp(64, 0)); // 0x2c helper.
  program.push_back(build_sop1_encoding(ROCJITSU_CODE_ARCH_RDNA4, 0x48, 0,
                                        kReturnSreg)
                        .value()); // 0x30 return.

  const auto image =
      rocjitsu::waitcheck_test::make_gfx_code_object(program, EF_AMDGPU_MACH_AMDGCN_GFX1201);
  AmdGpuCodeObject code_object(image.data(), image.size());
  auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx1201ReportsValuSgprHazardBeforeResolvedSwappcCall) {
  const std::vector<uint32_t> program =
      gfx1201_swappc_valu_sgpr_boundary_program(/*include_wait=*/false);
  ASSERT_FALSE(program.empty());
  const auto image =
      rocjitsu::waitcheck_test::make_gfx_code_object(program, EF_AMDGPU_MACH_AMDGCN_GFX1201);
  AmdGpuCodeObject code_object(image.data(), image.size());

  const auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  const WaitcheckDiagnostic &diagnostic = report.diagnostics.front();
  EXPECT_EQ(diagnostic.kind, WaitcheckDiagnosticKind::SgprDepctr);
  EXPECT_EQ(diagnostic.access, WaitcheckAccessKind::ControlTransfer);
  EXPECT_EQ(diagnostic.reg, (RegisterRef{RegClass::SGPR, 2, 1}));
  EXPECT_NE(diagnostic.instruction.find("s_swappc_b64"), std::string::npos);
  EXPECT_NE(diagnostic.message.find("depctr_va_sdst(0)"), std::string::npos);
  EXPECT_NE(diagnostic.message.find("control transfer"), std::string::npos);
}

TEST(WaitcheckTest, Gfx1201AcceptsValuSgprWaitBeforeResolvedSwappcCall) {
  const std::vector<uint32_t> program =
      gfx1201_swappc_valu_sgpr_boundary_program(/*include_wait=*/true);
  ASSERT_FALSE(program.empty());
  const auto image =
      rocjitsu::waitcheck_test::make_gfx_code_object(program, EF_AMDGPU_MACH_AMDGCN_GFX1201);
  AmdGpuCodeObject code_object(image.data(), image.size());

  const auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx1201ReportsValuSgprHazardBeforeResolvedDirectCall) {
  const std::vector<uint32_t> program =
      gfx1201_call_valu_sgpr_boundary_program(/*include_wait=*/false);
  ASSERT_FALSE(program.empty());
  const auto image = rocjitsu::waitcheck_test::make_gfx1201_code_object(program);
  AmdGpuCodeObject code_object(image.data(), image.size());

  const auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  const WaitcheckDiagnostic &diagnostic = report.diagnostics.front();
  EXPECT_EQ(diagnostic.access, WaitcheckAccessKind::ControlTransfer);
  EXPECT_EQ(diagnostic.reg, (RegisterRef{RegClass::SGPR, 2, 1}));
  EXPECT_NE(diagnostic.instruction.find("s_call_b64"), std::string::npos);
  EXPECT_NE(diagnostic.message.find("depctr_va_sdst(0)"), std::string::npos);
}

TEST(WaitcheckTest, Gfx1201AcceptsValuSgprWaitBeforeResolvedDirectCall) {
  const std::vector<uint32_t> program =
      gfx1201_call_valu_sgpr_boundary_program(/*include_wait=*/true);
  ASSERT_FALSE(program.empty());
  const auto image = rocjitsu::waitcheck_test::make_gfx1201_code_object(program);
  AmdGpuCodeObject code_object(image.data(), image.size());

  const auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx1201ReportsSaluSgprHazardBeforeResolvedFunctionReturn) {
  constexpr uint16_t kReturnSreg = 30;
  std::vector<uint32_t> program;
  program.push_back(build_s_call_b64(kReturnSreg, 1, ROCJITSU_CODE_ARCH_RDNA4).value());
  append_inst(program, s_endpgm());
  append_inst(program, s_mov_b32(/*sdst=*/2, /*ssrc0=*/128));
  program.push_back(build_s_setpc_b64(kReturnSreg, ROCJITSU_CODE_ARCH_RDNA4).value());
  const auto image =
      rocjitsu::waitcheck_test::make_gfx_code_object(program, EF_AMDGPU_MACH_AMDGCN_GFX1201);
  AmdGpuCodeObject code_object(image.data(), image.size());

  const auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  const WaitcheckDiagnostic &diagnostic = report.diagnostics.front();
  EXPECT_EQ(diagnostic.access, WaitcheckAccessKind::ControlTransfer);
  EXPECT_EQ(diagnostic.reg, (RegisterRef{RegClass::SGPR, 2, 1}));
  EXPECT_NE(diagnostic.instruction.find("s_setpc_b64"), std::string::npos);
  EXPECT_NE(diagnostic.message.find("depctr_sa_sdst(0)"), std::string::npos);
}

TEST(WaitcheckTest, Gfx1201AcceptsSaluSgprWaitBeforeResolvedFunctionReturn) {
  constexpr uint16_t kReturnSreg = 30;
  std::vector<uint32_t> program;
  program.push_back(build_s_call_b64(kReturnSreg, 1, ROCJITSU_CODE_ARCH_RDNA4).value());
  append_inst(program, s_endpgm());
  append_inst(program, s_mov_b32(/*sdst=*/2, /*ssrc0=*/128));
  append_inst(program, s_wait_alu_sa_sdst_0());
  program.push_back(build_s_setpc_b64(kReturnSreg, ROCJITSU_CODE_ARCH_RDNA4).value());
  const auto image =
      rocjitsu::waitcheck_test::make_gfx_code_object(program, EF_AMDGPU_MACH_AMDGCN_GFX1201);
  AmdGpuCodeObject code_object(image.data(), image.size());

  const auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx1201ReportsValuVccHazardBeforeResolvedSwappcCall) {
  constexpr uint16_t kTargetSreg = 8;
  constexpr uint16_t kReturnSreg = 30;
  constexpr uint16_t kLiteralOperand = 255;
  constexpr uint16_t kInlineInt0 = 128;
  std::vector<uint32_t> program;
  program.push_back(build_s_getpc_b64(kTargetSreg, ROCJITSU_CODE_ARCH_RDNA4).value());
  program.push_back(
      build_s_add_u32(kTargetSreg, kTargetSreg, kLiteralOperand, ROCJITSU_CODE_ARCH_RDNA4).value());
  const size_t delta_word = program.size();
  program.push_back(0);
  program.push_back(
      build_s_addc_u32(kTargetSreg + 1, kTargetSreg + 1, kInlineInt0, ROCJITSU_CODE_ARCH_RDNA4)
          .value());
  append_inst(program, v_cndmask_b32_e32(/*vdst=*/0, /*src0=*/0, /*vsrc1=*/0));
  append_inst(program, v_cmp_gt_u32_e32(/*src0=*/0, /*vsrc1=*/0));
  program.push_back(build_s_swappc_b64(kReturnSreg, kTargetSreg, ROCJITSU_CODE_ARCH_RDNA4).value());
  append_inst(program, s_endpgm());
  const uint64_t helper_offset = program.size() * sizeof(uint32_t);
  constexpr uint64_t kGetpcResultOffset = sizeof(uint32_t);
  program[delta_word] = static_cast<uint32_t>(helper_offset - kGetpcResultOffset);
  program.push_back(build_s_setpc_b64(kReturnSreg, ROCJITSU_CODE_ARCH_RDNA4).value());
  const auto image =
      rocjitsu::waitcheck_test::make_gfx_code_object(program, EF_AMDGPU_MACH_AMDGCN_GFX1201);
  AmdGpuCodeObject code_object(image.data(), image.size());

  const auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  const WaitcheckDiagnostic &diagnostic = report.diagnostics.front();
  EXPECT_EQ(diagnostic.access, WaitcheckAccessKind::ControlTransfer);
  EXPECT_EQ(diagnostic.reg, (RegisterRef{RegClass::VCC, 0, 1}));
  EXPECT_NE(diagnostic.message.find("depctr_va_vcc(0)"), std::string::npos);
}

TEST(WaitcheckTest, Gfx1201UnresolvedSetpcStillChecksSgprBoundaryHazards) {
  std::vector<uint32_t> program;
  append_inst(program, v_add_f32_e32(/*vdst=*/0, /*src0=*/2, /*vsrc1=*/0));
  append_inst(program, s_mov_b32(/*sdst=*/2, /*ssrc0=*/128));
  program.push_back(build_s_setpc_b64(/*ssrc0=*/8, ROCJITSU_CODE_ARCH_RDNA4).value());
  const auto image =
      rocjitsu::waitcheck_test::make_gfx_code_object(program, EF_AMDGPU_MACH_AMDGCN_GFX1201);
  AmdGpuCodeObject code_object(image.data(), image.size());

  const auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_FALSE(report.complete);
  EXPECT_FALSE(report.passed());
  EXPECT_NE(report.analysis_error.find("incomplete control flow"), std::string::npos);
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics.front().access, WaitcheckAccessKind::ControlTransfer);
  EXPECT_NE(report.diagnostics.front().message.find("depctr_sa_sdst(0)"), std::string::npos);
}

TEST(WaitcheckTest, Gfx1201UnresolvedSwappcConservativelySeedsItsContinuation) {
  constexpr uint16_t kReturnSreg = 30;
  std::vector<uint32_t> program;
  program.push_back(build_s_swappc_b64(kReturnSreg, /*ssrc0=*/8, ROCJITSU_CODE_ARCH_RDNA4).value());
  append_inst(program, s_mov_b32(/*sdst=*/2, /*ssrc0=*/128));
  append_inst(program, v_add_f32_e32(/*vdst=*/0, /*src0=*/2, /*vsrc1=*/0));
  append_inst(program, s_endpgm());
  const auto image =
      rocjitsu::waitcheck_test::make_gfx_code_object(program, EF_AMDGPU_MACH_AMDGCN_GFX1201);
  AmdGpuCodeObject code_object(image.data(), image.size());

  const auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_FALSE(report.complete);
  EXPECT_FALSE(report.passed());
  EXPECT_NE(report.analysis_error.find("incomplete control flow"), std::string::npos);
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  const WaitcheckDiagnostic &diagnostic = report.diagnostics.front();
  EXPECT_EQ(diagnostic.access, WaitcheckAccessKind::Use);
  EXPECT_EQ(diagnostic.reg, (RegisterRef{RegClass::SGPR, 2, 1}));
  EXPECT_NE(diagnostic.message.find("depctr_sa_sdst(0)"), std::string::npos);
}

TEST(WaitcheckTest, Gfx1201UnresolvedSwappcLinkRegisterIsPendingAtContinuation) {
  constexpr uint16_t kReturnSreg = 30;
  std::vector<uint32_t> program;
  program.push_back(build_s_swappc_b64(kReturnSreg, /*ssrc0=*/8, ROCJITSU_CODE_ARCH_RDNA4).value());
  append_inst(program, v_add_f32_e32(/*vdst=*/0, /*src0=*/kReturnSreg, /*vsrc1=*/0));
  append_inst(program, s_endpgm());
  const auto image = rocjitsu::waitcheck_test::make_gfx1201_code_object(program);
  AmdGpuCodeObject code_object(image.data(), image.size());

  const auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_FALSE(report.complete);
  EXPECT_FALSE(report.passed());
  EXPECT_NE(report.analysis_error.find("incomplete control flow"), std::string::npos);
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  const WaitcheckDiagnostic &diagnostic = report.diagnostics.front();
  EXPECT_EQ(diagnostic.access, WaitcheckAccessKind::Use);
  EXPECT_EQ(diagnostic.reg, (RegisterRef{RegClass::SGPR, kReturnSreg, 1}));
  EXPECT_NE(diagnostic.producer_instruction.find("s_swappc_b64"), std::string::npos);
  EXPECT_NE(diagnostic.message.find("depctr_sa_sdst(0)"), std::string::npos);
}

TEST(WaitcheckTest, Gfx1201ResolvedCallLinkRegisterIsPendingAtContinuation) {
  constexpr uint16_t kReturnSreg = 30;
  std::vector<uint32_t> program;
  program.push_back(
      build_s_call_b64(kReturnSreg, /*offset_dwords=*/2, ROCJITSU_CODE_ARCH_RDNA4).value());
  append_inst(program, v_add_f32_e32(/*vdst=*/0, /*src0=*/kReturnSreg, /*vsrc1=*/0));
  append_inst(program, s_endpgm());
  program.push_back(build_s_setpc_b64(kReturnSreg, ROCJITSU_CODE_ARCH_RDNA4).value());
  const auto image = rocjitsu::waitcheck_test::make_gfx1201_code_object(program);
  AmdGpuCodeObject code_object(image.data(), image.size());

  const auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  const WaitcheckDiagnostic &diagnostic = report.diagnostics.front();
  EXPECT_EQ(diagnostic.access, WaitcheckAccessKind::Use);
  EXPECT_EQ(diagnostic.reg, (RegisterRef{RegClass::SGPR, kReturnSreg, 1}));
  EXPECT_NE(diagnostic.producer_instruction.find("s_call_b64"), std::string::npos);
  EXPECT_NE(diagnostic.message.find("depctr_sa_sdst(0)"), std::string::npos);
}

TEST(WaitcheckTest, Gfx1201WaitClearsResolvedCallLinkRegisterAtContinuation) {
  constexpr uint16_t kReturnSreg = 30;
  std::vector<uint32_t> program;
  program.push_back(
      build_s_call_b64(kReturnSreg, /*offset_dwords=*/3, ROCJITSU_CODE_ARCH_RDNA4).value());
  append_inst(program, s_wait_alu_sa_sdst_0());
  append_inst(program, v_add_f32_e32(/*vdst=*/0, /*src0=*/kReturnSreg, /*vsrc1=*/0));
  append_inst(program, s_endpgm());
  program.push_back(build_s_setpc_b64(kReturnSreg, ROCJITSU_CODE_ARCH_RDNA4).value());
  const auto image = rocjitsu::waitcheck_test::make_gfx1201_code_object(program);
  AmdGpuCodeObject code_object(image.data(), image.size());

  const auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx1250ClosesBankPreservingNestedDeviceCalls) {
  constexpr auto arch = ROCJITSU_CODE_ARCH_CDNA5;
  for (bool unknown_callee : {false, true}) {
    for (bool changes_bank : {false, true}) {
      SCOPED_TRACE(::testing::Message()
                   << "unknown callee " << unknown_callee << ", changes bank " << changes_bank);
      const uint32_t end = build_s_endpgm(arch).value();
      const uint32_t ret = build_s_setpc_b64(30, arch).value();
      const std::vector<uint32_t> entry = {
          0xbe804700u, // 0x00: getpc s[0:1]
          0xa980fe00u,
          80u,
          0u, // 0x04: s[0:1] = .text+0x54
          0xd761002cu,
          0x02010000u, // 0x10: stash both halves in v44
          0xd761002cu,
          0x02010201u,                           // 0x18: v44 lane 1
          build_s_call_b64(30, 6, arch).value(), // 0x20: nested device helper at 0x3c
          0xd7600000u,
          0x0201012cu, // 0x24: readlane s0, v44, 0
          0xd7600001u,
          0x0201032cu,                             // 0x2c: readlane s1, v44, 1
          build_s_swappc_b64(30, 0, arch).value(), // 0x34: use stashed target
          end,                                     // 0x38
      };
      const std::vector<uint32_t> helper = {
          0xbec0011eu,                           // 0x3c: save caller return address in s[64:65]
          build_s_call_b64(30, 2, arch).value(), // 0x40: nested leaf at 0x4c
          0xbe9e0140u,                           // 0x44: restore s[30:31]
          ret,                                   // 0x48: return to entry
          unknown_callee ? build_s_swappc_b64(4, 8, arch).value()
          : changes_bank ? 0xbf860041u
                         : build_s_nop(0, arch).value(),
          ret, // 0x50
      };
      auto program = entry;
      // Independent callee at 0x54 is reached through the reconstructed pair.
      program.insert(program.end(), helper.begin(), helper.end());
      program.push_back(ret);
      const auto image =
          waitcheck_test::make_gfx_code_object(program, EF_AMDGPU_MACH_AMDGCN_GFX1250, true);
      AmdGpuCodeObject object(image.data(), image.size());
      const auto report = analyze_waitcnts_for_kernel(object, arch, 0);
      EXPECT_EQ(report.complete, !unknown_callee && !changes_bank) << report.analysis_error;
    }
  }
}

TEST(WaitcheckTest, Gfx1250DecodesReferencedHelperToVerifyItsLoopCarriedCallTarget) {
  constexpr auto arch = ROCJITSU_CODE_ARCH_CDNA5;
  for (bool clobber_target : {false, true}) {
    SCOPED_TRACE(clobber_target);
    const uint32_t end = build_s_endpgm(arch).value();
    const uint32_t ret = build_s_setpc_b64(30, arch).value();
    // The backward branch reaches the copy from the temporary's previous
    // call. Only a leaf that leaves s72:s73 intact permits the next iteration.
    const std::vector<uint32_t> caller{
        0xbec84700u, // 0x00: s_get_pc_i64 s[72:73]
        0xa9c8fe48u, 28u, 0u, // 0x04: s[72:73] = .text+0x20
        0xbe800148u, // 0x10: s_mov_b64 s[0:1], s[72:73]
        build_s_swappc_b64(30, 0, arch).value(),
        0xbfa1fffdu, // 0x18: s_cbranch_scc0 back to the copy
        end};
    const std::vector<uint32_t> helper{
        clobber_target ? 0xbec80080u : build_s_nop(0, arch).value(), ret};
    const auto image = waitcheck_test::make_gfx_multi_kernel_code_object(
        {{"caller", caller}, {"helper", helper}}, EF_AMDGPU_MACH_AMDGCN_GFX1250, true);
    AmdGpuCodeObject object(image.data(), image.size());
    const auto report = analyze_waitcnts_for_kernel(object, arch, 0);
    EXPECT_EQ(report.complete, !clobber_target) << report.analysis_error;
  }
}

TEST(WaitcheckTest, Gfx1201RelaxedProgramExitPreservesOnlyTheObservableTransferHazards) {
  constexpr auto arch = ROCJITSU_CODE_ARCH_RDNA4;
  for (bool program_exit : {false, true}) {
    for (bool report_sgpr : {false, true}) {
      SCOPED_TRACE(::testing::Message() << program_exit << " " << report_sgpr);
      std::vector<uint32_t> program;
      append_inst(program, v_cndmask_b32_e32(0, 0, 0)); // Track VCC through a VALU read.
      append_inst(program, v_add_f32_e32(1, 2, 1));     // Track s2 independently.
      append_inst(program, v_cmp_gt_u32_e32(0, 0));
      program.push_back(build_s_getpc_b64(8, arch).value());
      const size_t getpc_next = program.size();
      append_inst(program, s_wait_alu_sa_sdst_0());
      program.push_back(build_s_add_u32(8, 8, 255, arch).value());
      const size_t delta_word = program.size();
      program.push_back(0);
      program.push_back(build_s_addc_u32(9, 9, 128, arch).value());
      append_inst(program, s_wait_alu_sa_sdst_0());
      if (report_sgpr)
        append_inst(program, s_mov_b32(2, 128));
      const size_t branch = program.size() * sizeof(uint32_t);
      program.push_back(build_s_setpc_b64(8, arch).value());
      append_inst(program, s_endpgm());
      program[delta_word] = static_cast<uint32_t>((program.size() - getpc_next) * sizeof(uint32_t));
      if (!program_exit)
        append_inst(program, v_cndmask_b32_e32(3, 0, 3));
      append_inst(program, s_endpgm());
      const auto image = waitcheck_test::make_gfx1201_code_object(program);
      AmdGpuCodeObject object(image.data(), image.size());
      const auto report = analyze_waitcnts_for_kernel(object, arch, 0);
      ASSERT_TRUE(report.complete) << report.analysis_error;
      EXPECT_EQ(report.diagnostics.size(), unsigned(!program_exit) + unsigned(report_sgpr))
          << diagnostic_summary(report);
      for (const auto &diagnostic : report.diagnostics) {
        EXPECT_EQ(diagnostic.section_offset, branch);
        EXPECT_EQ(diagnostic.access, WaitcheckAccessKind::ControlTransfer);
        EXPECT_TRUE((!program_exit && diagnostic.reg.cls == RegClass::VCC) ||
                    (report_sgpr && diagnostic.reg == (RegisterRef{RegClass::SGPR, 2, 1})));
      }
    }
  }
}

TEST(WaitcheckTest, Gfx1201ResolvedIndirectBranchPreservesOrdinarySgprState) {
  constexpr uint16_t kTargetSreg = 8;
  constexpr uint16_t kLiteralOperand = 255;
  constexpr uint16_t kInlineInt0 = 128;
  std::vector<uint32_t> program;
  program.push_back(build_s_getpc_b64(kTargetSreg, ROCJITSU_CODE_ARCH_RDNA4).value());
  program.push_back(
      build_s_add_u32(kTargetSreg, kTargetSreg, kLiteralOperand, ROCJITSU_CODE_ARCH_RDNA4).value());
  const size_t delta_word = program.size();
  program.push_back(0);
  program.push_back(
      build_s_addc_u32(kTargetSreg + 1, kTargetSreg + 1, kInlineInt0, ROCJITSU_CODE_ARCH_RDNA4)
          .value());
  program.push_back(build_s_setpc_b64(kTargetSreg, ROCJITSU_CODE_ARCH_RDNA4).value());
  append_inst(program, s_endpgm());
  const uint64_t target_offset = program.size() * sizeof(uint32_t);
  constexpr uint64_t kGetpcResultOffset = sizeof(uint32_t);
  program[delta_word] = static_cast<uint32_t>(target_offset - kGetpcResultOffset);
  append_inst(program, s_mov_b32(/*sdst=*/2, /*ssrc0=*/128));
  append_inst(program, v_add_f32_e32(/*vdst=*/0, /*src0=*/2, /*vsrc1=*/0));
  append_inst(program, s_endpgm());
  const auto image = rocjitsu::waitcheck_test::make_gfx1201_code_object(program);
  AmdGpuCodeObject code_object(image.data(), image.size());

  const auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_GE(report.instructions_analyzed, 7u);
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx1201DirectBranchPreservesPendingSgprHazard) {
  std::vector<uint32_t> program;
  append_inst(program, v_add_f32_e32(/*vdst=*/0, /*src0=*/2, /*vsrc1=*/0));
  append_inst(program, s_mov_b32(/*sdst=*/2, /*ssrc0=*/128));
  program.push_back(build_s_branch(/*offset_dwords=*/1, ROCJITSU_CODE_ARCH_RDNA4).value());
  append_inst(program, s_endpgm());
  append_inst(program, v_add_f32_e32(/*vdst=*/1, /*src0=*/2, /*vsrc1=*/1));
  append_inst(program, s_endpgm());
  const auto image = rocjitsu::waitcheck_test::make_gfx1201_code_object(program);
  AmdGpuCodeObject code_object(image.data(), image.size());

  const auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  const WaitcheckDiagnostic &diagnostic = report.diagnostics.front();
  EXPECT_EQ(diagnostic.access, WaitcheckAccessKind::Use);
  EXPECT_EQ(diagnostic.reg, (RegisterRef{RegClass::SGPR, 2, 1}));
  EXPECT_NE(diagnostic.instruction.find("v_add_f32"), std::string::npos);
}

TEST(WaitcheckTest, Gfx1201SymbolLessSectionConservativelySeedsEveryCfgRoot) {
  std::vector<uint32_t> program;
  append_inst(program, s_endpgm());
  append_inst(program, s_mov_b32(/*sdst=*/2, /*ssrc0=*/128));
  append_inst(program, v_add_f32_e32(/*vdst=*/0, /*src0=*/2, /*vsrc1=*/0));
  append_inst(program, s_endpgm());
  const auto image = rocjitsu::waitcheck_test::make_gfx_symbol_less_code_object(
      program, EF_AMDGPU_MACH_AMDGCN_GFX1201);
  AmdGpuCodeObject code_object(image.data(), image.size());

  const auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics.front().access, WaitcheckAccessKind::Use);
  EXPECT_EQ(report.diagnostics.front().reg, (RegisterRef{RegClass::SGPR, 2, 1}));
}

TEST(WaitcheckTest, Gfx1201ReportsSaluVccHazardAtControlTransfer) {
  std::vector<uint32_t> program;
  append_inst(program, v_cndmask_b32_e32(/*vdst=*/0, /*src0=*/0, /*vsrc1=*/0));
  append_inst(program, s_mov_b32(/*sdst=*/106, /*ssrc0=*/128)); // vcc_lo.
  program.push_back(build_s_setpc_b64(/*ssrc0=*/8, ROCJITSU_CODE_ARCH_RDNA4).value());
  const auto image = rocjitsu::waitcheck_test::make_gfx1201_code_object(program);
  AmdGpuCodeObject code_object(image.data(), image.size());

  const auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  const WaitcheckDiagnostic &diagnostic = report.diagnostics.front();
  EXPECT_EQ(diagnostic.access, WaitcheckAccessKind::ControlTransfer);
  EXPECT_EQ(diagnostic.reg, (RegisterRef{RegClass::VCC, 0, 1}));
  EXPECT_NE(diagnostic.message.find("depctr_sa_sdst(0)"), std::string::npos);
}

TEST(WaitcheckTest, Gfx1201ControlTransferReportsOneRepresentativePerDepctrField) {
  std::vector<uint32_t> program;
  append_inst(program, v_add_f32_e32(/*vdst=*/0, /*src0=*/2, /*vsrc1=*/0));
  append_inst(program, v_cndmask_b32_e32(/*vdst=*/1, /*src0=*/0, /*vsrc1=*/1));
  append_inst(program, s_mov_b32(/*sdst=*/2, /*ssrc0=*/128));
  append_inst(program, s_mov_b32(/*sdst=*/106, /*ssrc0=*/128)); // vcc_lo.
  program.push_back(build_s_setpc_b64(/*ssrc0=*/8, ROCJITSU_CODE_ARCH_RDNA4).value());
  const auto image = rocjitsu::waitcheck_test::make_gfx1201_code_object(program);
  AmdGpuCodeObject code_object(image.data(), image.size());

  const auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  const WaitcheckDiagnostic &diagnostic = report.diagnostics.front();
  EXPECT_EQ(diagnostic.access, WaitcheckAccessKind::ControlTransfer);
  EXPECT_EQ(diagnostic.reg, (RegisterRef{RegClass::SGPR, 2, 1}));
  EXPECT_NE(diagnostic.message.find("depctr_sa_sdst(0)"), std::string::npos);
}

TEST(WaitcheckTest, Gfx1201ScalarIssueStallClearsValuSgprHazardBeforeTransfer) {
  std::vector<uint32_t> program;
  append_inst(program, v_add_f32_e32(/*vdst=*/0, /*src0=*/2, /*vsrc1=*/0));
  const auto readlane =
      build_v_readlane_b32(/*sdst=*/2, /*vsrc=*/40, /*lane=*/0, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(readlane.succeeded());
  program.insert(program.end(), readlane.value().begin(), readlane.value().end());
  // Any register-bearing SALU instruction supplies the hardware issue stall
  // modeled by AMDGPUInsertDelayAlu, even when its operands are unrelated.
  append_inst(program, s_mov_b32(/*sdst=*/4, /*ssrc0=*/128));
  program.push_back(build_s_setpc_b64(/*ssrc0=*/8, ROCJITSU_CODE_ARCH_RDNA4).value());
  const auto image = rocjitsu::waitcheck_test::make_gfx1201_code_object(program);
  AmdGpuCodeObject code_object(image.data(), image.size());

  const auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx1250DoesNotApplyRdna4SgprControlTransferRules) {
  std::vector<uint32_t> program;
  append_inst(program, v_add_f32_e32(/*vdst=*/0, /*src0=*/2, /*vsrc1=*/0));
  append_inst(program, s_mov_b32(/*sdst=*/2, /*ssrc0=*/128));
  program.push_back(build_s_setpc_b64(/*ssrc0=*/8, ROCJITSU_CODE_ARCH_CDNA5).value());
  const auto image = rocjitsu::waitcheck_test::make_gfx1250_code_object(program);
  AmdGpuCodeObject code_object(image.data(), image.size());

  const auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_CDNA5);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx1201ControlTransferReportsEveryOutstandingSgprWaitField) {
  std::vector<uint32_t> program;
  append_inst(program, v_add_f32_e32(/*vdst=*/0, /*src0=*/2, /*vsrc1=*/0));
  append_inst(program, v_add_f32_e32(/*vdst=*/1, /*src0=*/4, /*vsrc1=*/1));
  append_inst(program, v_cndmask_b32_e32(/*vdst=*/2, /*src0=*/0, /*vsrc1=*/2));
  append_inst(program, s_mov_b32(/*sdst=*/2, /*ssrc0=*/128));
  const auto readlane =
      build_v_readlane_b32(/*sdst=*/4, /*vsrc=*/40, /*lane=*/0, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(readlane.succeeded());
  program.insert(program.end(), readlane.value().begin(), readlane.value().end());
  append_inst(program, v_cmp_gt_u32_e32(/*src0=*/0, /*vsrc1=*/0));
  program.push_back(build_s_setpc_b64(/*ssrc0=*/8, ROCJITSU_CODE_ARCH_RDNA4).value());
  const auto image =
      rocjitsu::waitcheck_test::make_gfx_code_object(program, EF_AMDGPU_MACH_AMDGCN_GFX1201);
  AmdGpuCodeObject code_object(image.data(), image.size());

  const auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 3u) << diagnostic_summary(report);
  // A single all-zero depctr wait repairs every register covered by a field.
  // Report one representative register per required field instead of emitting
  // a duplicate diagnostic for every outstanding lane.
  std::set<std::string> fields;
  for (const WaitcheckDiagnostic &diagnostic : report.diagnostics) {
    EXPECT_EQ(diagnostic.access, WaitcheckAccessKind::ControlTransfer);
    if (diagnostic.message.find("depctr_sa_sdst(0)") != std::string::npos)
      fields.insert("sa_sdst");
    if (diagnostic.message.find("depctr_va_sdst(0)") != std::string::npos)
      fields.insert("va_sdst");
    if (diagnostic.message.find("depctr_va_vcc(0)") != std::string::npos)
      fields.insert("va_vcc");
  }
  EXPECT_EQ(fields, (std::set<std::string>{"sa_sdst", "va_sdst", "va_vcc"}));
}

TEST(WaitcheckTest, Gfx1201NonEntryFunctionConservativelyTracksEverySgprPair) {
  constexpr uint16_t kReturnSreg = 30;
  std::vector<uint32_t> function;
  append_inst(function, s_mov_b32(/*sdst=*/2, /*ssrc0=*/128));
  append_inst(function, v_add_f32_e32(/*vdst=*/0, /*src0=*/2, /*vsrc1=*/0));
  append_inst(function, s_wait_alu_sa_sdst_0());
  function.push_back(build_s_setpc_b64(kReturnSreg, ROCJITSU_CODE_ARCH_RDNA4).value());
  const auto image =
      rocjitsu::waitcheck_test::make_gfx1201_function_only_code_object({{"helper", function}});
  AmdGpuCodeObject code_object(image.data(), image.size());

  const auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  const WaitcheckDiagnostic &diagnostic = report.diagnostics.front();
  EXPECT_EQ(diagnostic.access, WaitcheckAccessKind::Use);
  EXPECT_EQ(diagnostic.reg, (RegisterRef{RegClass::SGPR, 2, 1}));
  EXPECT_NE(diagnostic.message.find("depctr_sa_sdst(0)"), std::string::npos);
}

TEST(WaitcheckTest, Gfx1201NonEntryFunctionConservativelyTracksVcc) {
  constexpr uint16_t kReturnSreg = 30;
  std::vector<uint32_t> function;
  append_inst(function, v_cmp_gt_u32_e32(/*src0=*/0, /*vsrc1=*/0));
  append_inst(function, v_cndmask_b32_e32(/*vdst=*/0, /*src0=*/0, /*vsrc1=*/0));
  append_inst(function, s_wait_alu_va_vcc_0());
  function.push_back(build_s_setpc_b64(kReturnSreg, ROCJITSU_CODE_ARCH_RDNA4).value());
  const auto image =
      rocjitsu::waitcheck_test::make_gfx1201_function_only_code_object({{"helper", function}});
  AmdGpuCodeObject code_object(image.data(), image.size());

  const auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  const WaitcheckDiagnostic &diagnostic = report.diagnostics.front();
  EXPECT_EQ(diagnostic.access, WaitcheckAccessKind::Use);
  EXPECT_EQ(diagnostic.reg, (RegisterRef{RegClass::VCC, 0, 1}));
  EXPECT_NE(diagnostic.message.find("depctr_va_vcc(0)"), std::string::npos);
}

TEST(WaitcheckTest, Gfx1250ResolvedSwappcLiteral64HelperWaitAppliesBeforeContinuation) {
  constexpr uint16_t kTargetSreg = 8;
  constexpr uint16_t kReturnSreg = 30;

  std::vector<uint32_t> program;
  append_inst(program, global_load_b32(0));                                            // 0x00.
  program.push_back(build_s_getpc_b64(kTargetSreg, ROCJITSU_CODE_ARCH_CDNA5).value()); // 0x0c.
  program.push_back(0xA988FE08u); // 0x10: s_add_nc_u64 s[8:9], s[8:9], literal64.
  program.push_back(24);          // getpc next 0x10 + 24 -> helper 0x28.
  program.push_back(0);
  program.push_back(
      build_s_swappc_b64(kReturnSreg, kTargetSreg, ROCJITSU_CODE_ARCH_CDNA5).value()); // 0x1c.
  append_inst(program, v_mov_b32(1, 0));                                               // 0x20.
  append_inst(program, s_endpgm());                                                    // 0x24.
  append_inst(program, sopp(64, 0)); // 0x28 helper: s_wait_loadcnt 0.
  program.push_back(build_sop1_encoding(ROCJITSU_CODE_ARCH_CDNA5, 0x48, 0,
                                        kReturnSreg)
                        .value()); // 0x2c return.

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  std::unique_ptr<Instruction> add_inst(decode_valid(*decoder, &program[4], 0x10));
  ASSERT_NE(add_inst, nullptr);
  EXPECT_EQ(add_inst->mnemonic(), "s_add_nc_u64");
  ASSERT_EQ(add_inst->num_dst_operands(), 1);
  ASSERT_EQ(add_inst->num_src_operands(), 2);
  EXPECT_EQ(add_inst->dst_operand(0)->to_register_ref(),
            (RegisterRef{RegClass::SGPR, kTargetSreg, 2}));
  EXPECT_EQ(add_inst->src_operand(0)->to_register_ref(),
            (RegisterRef{RegClass::SGPR, kTargetSreg, 2}));
  EXPECT_EQ(add_inst->src_operand(1)->literal64_value(), 24u);

  std::vector<std::unique_ptr<Instruction>> decoded;
  std::vector<const Instruction *> decoded_ptrs;
  for (uint64_t offset = 0; offset < program.size() * sizeof(uint32_t);) {
    std::unique_ptr<Instruction> inst(
        decode_valid(*decoder, &program[offset / sizeof(uint32_t)], offset));
    ASSERT_NE(inst, nullptr);
    offset += static_cast<uint64_t>(inst->size());
    decoded_ptrs.push_back(inst.get());
    decoded.push_back(std::move(inst));
  }
  const auto text = std::span<const uint8_t>(reinterpret_cast<const uint8_t *>(program.data()),
                                             program.size() * sizeof(uint32_t));
  const auto fixups = discover_indirect_branch_edges(decoded_ptrs, text, ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_EQ(fixups.size(), 1u);
  EXPECT_EQ(fixups[0].source_target_offset, 0x28u);
  EXPECT_TRUE(fixups[0].source_is_call);

  const auto image =
      rocjitsu::waitcheck_test::make_gfx_code_object(program, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject code_object(image.data(), image.size());
  auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_CDNA5);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
  EXPECT_GE(report.instructions_analyzed, 8u);
}

TEST(WaitcheckTest, Gfx1250Wave32VopcDoesNotClobberAdjacentSwappcTarget) {
  constexpr uint16_t kTargetSreg = 4;
  constexpr uint16_t kReturnSreg = 30;

  std::vector<uint32_t> program;
  append_inst(program, global_load_b32(0));                                            // 0x00.
  program.push_back(build_s_getpc_b64(kTargetSreg, ROCJITSU_CODE_ARCH_CDNA5).value()); // 0x0c.
  program.push_back(0xA984FE04u); // 0x10: s_add_nc_u64 s[4:5], s[4:5], literal64.
  program.push_back(32);          // getpc next 0x10 + 32 -> helper 0x30.
  program.push_back(0);
  program.push_back(0xD4410003u); // 0x1c: v_cmp_lt_i32_e64 s3, v20, v101.
  program.push_back(0x0202CB14u);
  program.push_back(
      build_s_swappc_b64(kReturnSreg, kTargetSreg, ROCJITSU_CODE_ARCH_CDNA5).value()); // 0x24.
  append_inst(program, v_mov_b32(1, 0));                                               // 0x28.
  append_inst(program, s_endpgm());                                                    // 0x2c.
  append_inst(program, sopp(72, 0)); // 0x30 helper: s_wait_loadcnt_dscnt 0.
  program.push_back(build_sop1_encoding(ROCJITSU_CODE_ARCH_CDNA5, 0x48, 0,
                                        kReturnSreg)
                        .value()); // 0x34 return.

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  std::unique_ptr<Instruction> cmp_inst(decode_valid(*decoder, &program[7], 0x1c));
  ASSERT_NE(cmp_inst, nullptr);
  EXPECT_EQ(cmp_inst->mnemonic(), "v_cmp_lt_i32");
  ASSERT_EQ(cmp_inst->num_dst_operands(), 1);
  EXPECT_EQ(cmp_inst->dst_operand(0)->to_register_ref(), (RegisterRef{RegClass::SGPR, 3, 1}));

  std::vector<std::unique_ptr<Instruction>> decoded;
  std::vector<const Instruction *> decoded_ptrs;
  for (uint64_t offset = 0; offset < program.size() * sizeof(uint32_t);) {
    std::unique_ptr<Instruction> inst(
        decode_valid(*decoder, &program[offset / sizeof(uint32_t)], offset));
    ASSERT_NE(inst, nullptr);
    offset += static_cast<uint64_t>(inst->size());
    decoded_ptrs.push_back(inst.get());
    decoded.push_back(std::move(inst));
  }
  const auto text = std::span<const uint8_t>(reinterpret_cast<const uint8_t *>(program.data()),
                                             program.size() * sizeof(uint32_t));
  const auto fixups = discover_indirect_branch_edges(
      decoded_ptrs, text, ROCJITSU_CODE_ARCH_CDNA5, {}, ExternalEntryPolicy::InferPredecessorless,
      nullptr, {}, {}, ROCJITSU_CODE_TARGET_GFX1250, 32);
  ASSERT_EQ(fixups.size(), 1u);
  EXPECT_EQ(fixups[0].source_target_offset, 0x30u);
  EXPECT_TRUE(fixups[0].source_is_call);
  EXPECT_EQ(discover_indirect_branch_edges(decoded_ptrs, text, ROCJITSU_CODE_ARCH_CDNA5, {},
                                           ExternalEntryPolicy::InferPredecessorless, nullptr, {},
                                           {}, ROCJITSU_CODE_TARGET_GFX1250, 64)
                .size(),
            1u);

  const auto image =
      rocjitsu::waitcheck_test::make_gfx_code_object(program, EF_AMDGPU_MACH_AMDGCN_GFX1250, true);
  AmdGpuCodeObject code_object(image.data(), image.size());
  auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_CDNA5);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx950ResolvedSwappcPropagatesHelperLoadToContinuation) {
  constexpr uint16_t kTargetSreg = 8;
  constexpr uint16_t kReturnSreg = 30;
  constexpr uint16_t kLiteralOperand = 255;
  constexpr uint16_t kInlineInt0 = 128;

  std::vector<uint32_t> program;
  program.push_back(build_s_getpc_b64(kTargetSreg, ROCJITSU_CODE_ARCH_CDNA4).value()); // 0x00.
  program.push_back(build_s_add_u32(kTargetSreg, kTargetSreg, kLiteralOperand,
                                    ROCJITSU_CODE_ARCH_CDNA4)
                        .value()); // 0x04.
  program.push_back(24);           // 0x08 -> 0x1c.
  program.push_back(build_s_addc_u32(kTargetSreg + 1, kTargetSreg + 1, kInlineInt0,
                                     ROCJITSU_CODE_ARCH_CDNA4)
                        .value()); // 0x0c.
  program.push_back(
      build_s_swappc_b64(kReturnSreg, kTargetSreg, ROCJITSU_CODE_ARCH_CDNA4).value()); // 0x10.
  append_gfx950_v_mov_b32_v1_v0(program);                                              // 0x14.
  program.push_back(build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4).value());                 // 0x18.
  append_gfx950_buffer_load_dword_v0_v8_s0_offen(program); // 0x1c helper.
  program.push_back(build_sop1_encoding(ROCJITSU_CODE_ARCH_CDNA4, 0x1d, 0,
                                        kReturnSreg)
                        .value()); // 0x24 return.

  const auto image =
      rocjitsu::waitcheck_test::make_gfx_code_object(program, EF_AMDGPU_MACH_AMDGCN_GFX950);
  AmdGpuCodeObject code_object(image.data(), image.size());
  auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_CDNA4);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Load);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Use);
  EXPECT_EQ(report.diagnostics[0].reg.index, 0u);
}

TEST(WaitcheckTest, Gfx950RecoveredSetpcTargetOverridesMatchingReturnRegister) {
  constexpr auto arch = ROCJITSU_CODE_ARCH_CDNA4;
  for (uint16_t target_sreg : {uint16_t{28}, uint16_t{30}}) {
    SCOPED_TRACE(target_sreg);
    std::vector<uint32_t> program;
    program.push_back(build_s_call_b64(30, 3, arch).value()); // Helper at 0x10.
    append_gfx950_s_waitcnt_vmcnt_0(program);
    append_gfx950_v_mov_b32_v1_v0(program);
    program.push_back(build_s_endpgm(arch).value());
    ASSERT_EQ(program.size() * sizeof(uint32_t), 0x10u);
    append_gfx950_buffer_load_dword_v0_v8_s0_offen(program);
    program.push_back(0xbea0011eu); // Save caller link s[30:31] in s[32:33].
    program.push_back(build_s_getpc_b64(target_sreg, arch).value());
    const uint64_t getpc_result = program.size() * sizeof(uint32_t);
    program.push_back(build_s_add_u32(target_sreg, target_sreg, 255, arch).value());
    const size_t delta_word = program.size();
    program.push_back(0);
    program.push_back(build_s_addc_u32(target_sreg + 1, target_sreg + 1, 128, arch).value());
    program.push_back(build_s_setpc_b64(target_sreg, arch).value());
    const uint64_t target = program.size() * sizeof(uint32_t);
    program[delta_word] = static_cast<uint32_t>(target - getpc_result);
    append_gfx950_v_mov_b32_v1_v0(program); // Recovered local target reads pending v0.
    program.push_back(0xbe9e0120u);         // Restore the caller link.
    program.push_back(build_s_setpc_b64(30, arch).value());

    const auto image = waitcheck_test::make_gfx_code_object(program, EF_AMDGPU_MACH_AMDGCN_GFX950);
    const AmdGpuCodeObject object(image.data(), image.size());
    const auto report = analyze_waitcnts_for_kernel(object, arch, 0);
    ASSERT_TRUE(report.supported && report.complete) << report.analysis_error;
    EXPECT_TRUE(std::ranges::any_of(report.diagnostics, [&](const auto &d) {
      return d.counter == WaitCounterKind::Load && d.section_offset == target && d.reg.index == 0;
    })) << diagnostic_summary(report);
  }
}

TEST(WaitcheckTest, Gfx950OverwrittenReturnPairLeavesIndirectJumpIncomplete) {
  constexpr auto arch = ROCJITSU_CODE_ARCH_CDNA4;
  std::vector<uint32_t> program;
  program.push_back(build_s_call_b64(30, 3, arch).value()); // Helper at 0x10.
  append_gfx950_s_waitcnt_vmcnt_0(program);
  append_gfx950_v_mov_b32_v1_v0(program);
  program.push_back(build_s_endpgm(arch).value());
  ASSERT_EQ(program.size() * sizeof(uint32_t), 0x10u);
  append_gfx950_buffer_load_dword_v0_v8_s0_offen(program);
  program.push_back(build_sop1_encoding(arch, 0x01, 30, 8).value()); // s_mov_b64 s[30:31],s[8:9]
  program.push_back(build_s_setpc_b64(30, arch).value());

  const auto image = waitcheck_test::make_gfx_code_object(program, EF_AMDGPU_MACH_AMDGCN_GFX950);
  const AmdGpuCodeObject object(image.data(), image.size());
  const auto report = analyze_waitcnts_for_kernel(object, arch, 0);
  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_FALSE(report.complete) << diagnostic_summary(report);
  EXPECT_NE(report.analysis_error.find("incomplete control flow"), std::string::npos)
      << report.analysis_error;
}

TEST(WaitcheckTest, Gfx950StandaloneHelperRequiresProvenExternalReturnLink) {
  constexpr auto arch = ROCJITSU_CODE_ARCH_CDNA4;
  std::vector<uint32_t> program;
  append_gfx950_buffer_load_dword_v0_v8_s0_offen(program);
  program.push_back(build_sop1_encoding(arch, 0x01, 30, 8).value()); // Unknown target in s[8:9].
  program.push_back(build_s_setpc_b64(30, arch).value());
  auto image = waitcheck_test::make_gfx_function_only_code_object(
      {{"helper", program}}, EF_AMDGPU_MACH_AMDGCN_GFX950);
  AmdGpuCodeObject object(image.data(), image.size());
  ASSERT_TRUE(object.is_valid());
  const auto unknown = analyze_waitcnts(object, arch);
  EXPECT_TRUE(unknown.supported) << unknown.analysis_error;
  EXPECT_FALSE(unknown.complete) << diagnostic_summary(unknown);
  EXPECT_NE(unknown.analysis_error.find("incomplete control flow"), std::string::npos);

  // An untouched incoming link is a valid return to the external caller.
  program.erase(program.begin() + 2);
  image = waitcheck_test::make_gfx_function_only_code_object(
      {{"helper", program}}, EF_AMDGPU_MACH_AMDGCN_GFX950);
  AmdGpuCodeObject returned(image.data(), image.size());
  ASSERT_TRUE(returned.is_valid());
  const auto direct_return = analyze_waitcnts(returned, arch);
  EXPECT_TRUE(direct_return.supported && direct_return.complete)
      << direct_return.analysis_error;

  program.insert(program.begin() + 2,
                 build_sop1_encoding(arch, 0x01, 32, 30).value()); // Save incoming link.
  program.insert(program.begin() + 3,
                 build_sop1_encoding(arch, 0x01, 30, 8).value()); // Reuse link registers.
  program.insert(program.begin() + 4,
                 build_sop1_encoding(arch, 0x01, 30, 32).value()); // Restore incoming link.
  image = waitcheck_test::make_gfx_function_only_code_object(
      {{"helper", program}}, EF_AMDGPU_MACH_AMDGCN_GFX950);
  AmdGpuCodeObject restored(image.data(), image.size());
  ASSERT_TRUE(restored.is_valid());
  const auto restored_return = analyze_waitcnts(restored, arch);
  EXPECT_TRUE(restored_return.supported && restored_return.complete)
      << restored_return.analysis_error;
}

TEST(WaitcheckTest, RawIndirectTransfersRequireCfgBeforeFollowingWords) {
  constexpr auto arch = ROCJITSU_CODE_ARCH_CDNA4;
  for (uint32_t transfer : {build_s_setpc_b64(8, arch).value(),
                            build_s_swappc_b64(30, 8, arch).value()}) {
    std::vector<uint32_t> program;
    append_gfx950_buffer_load_dword_v0_v8_s0_offen(program);
    program.push_back(transfer);
    append_gfx950_s_waitcnt_vmcnt_0(program);
    append_gfx950_v_mov_b32_v1_v0(program);
    const auto report = analyze_waitcnts(std::span<const uint32_t>(program), arch);
    EXPECT_TRUE(report.supported) << report.analysis_error;
    EXPECT_FALSE(report.complete) << diagnostic_summary(report);
    EXPECT_FALSE(report.passed());
    EXPECT_NE(report.analysis_error.find("control flow"), std::string::npos);
  }
}

TEST(WaitcheckTest, Gfx1250RelativeDestinationWriteInvalidatesReturnLink) {
  constexpr auto arch = ROCJITSU_CODE_ARCH_CDNA5;
  std::vector<uint32_t> program;
  program.push_back(build_s_call_b64(30, 1, arch).value()); // Helper at 0x08.
  program.push_back(build_s_endpgm(arch).value());
  program.push_back(0xbefd00ffu); // s_mov_b32 m0, 2 << 16.
  program.push_back(0x00020000u);
  program.push_back(build_sop1_encoding(arch, 0x44, 28, 8).value());
  program.push_back(build_s_setpc_b64(30, arch).value());

  auto decoder = Decoder::create(arch);
  std::unique_ptr<Instruction> relative_write(decode_valid(*decoder, &program[4], 0x10));
  ASSERT_NE(relative_write, nullptr);
  ASSERT_EQ(relative_write->mnemonic(), "s_movrelsd_2_b32");

  const auto image = waitcheck_test::make_gfx_code_object(
      program, EF_AMDGPU_MACH_AMDGCN_GFX1250, /*wave32=*/true);
  const AmdGpuCodeObject object(image.data(), image.size());
  const auto report = analyze_waitcnts_for_kernel(object, arch, 0);
  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_FALSE(report.complete) << diagnostic_summary(report);
  EXPECT_NE(report.analysis_error.find("incomplete control flow"), std::string::npos)
      << report.analysis_error;
}

TEST(WaitcheckTest, Gfx11And12NestedCallRestoresLinkFromProvenVgprLanes) {
  for (const auto [arch, machine] :
       {std::pair{ROCJITSU_CODE_ARCH_RDNA3, EF_AMDGPU_MACH_AMDGCN_GFX1100},
        std::pair{ROCJITSU_CODE_ARCH_RDNA3_5, EF_AMDGPU_MACH_AMDGCN_GFX1150},
        std::pair{ROCJITSU_CODE_ARCH_CDNA5, EF_AMDGPU_MACH_AMDGCN_GFX1250},
        std::pair{ROCJITSU_CODE_ARCH_RDNA4, EF_AMDGPU_MACH_AMDGCN_GFX1201}}) {
    for (unsigned clobber_spill : {0u, 1u, 2u}) {
      SCOPED_TRACE(testing::Message() << machine << '/' << clobber_spill);
      std::vector<uint32_t> program;
      program.push_back(build_s_call_b64(30, 1, arch).value()); // Helper at 0x08.
      program.push_back(build_s_endpgm(arch).value());
      program.insert(program.end(), {0xd7610002u, 0x0201001eu, // v_writelane v2,s30,0.
                                     0xd7610002u, 0x0201021fu}); // v_writelane v2,s31,1.
      const size_t inner_call = program.size();
      program.push_back(0);
      if (clobber_spill == 1)
        append_inst(program, v_mov_b32(2, 0)); // Destroy both saved lanes.
      if (clobber_spill == 2) {
        program.push_back(0xbefd0081u); // s_mov_b32 m0, 1.
        program.push_back(0x7e028480u); // v_movreld_b32 v1, 0 writes v2.
      }
      program.insert(program.end(), {0xd760001eu, 0x02010102u, // v_readlane s30,v2,0.
                                     0xd760001fu, 0x02010302u}); // v_readlane s31,v2,1.
      program.push_back(build_s_setpc_b64(30, arch).value());
      const size_t leaf = program.size();
      program[inner_call] = build_s_call_b64(30, static_cast<int>(leaf - inner_call - 1), arch).value();
      program.push_back(build_s_setpc_b64(30, arch).value());

      const auto image = waitcheck_test::make_gfx_code_object(program, machine, /*wave32=*/true);
      const AmdGpuCodeObject object(image.data(), image.size());
      const auto report = analyze_waitcnts_for_kernel(object, arch, 0);
      ASSERT_TRUE(report.supported) << report.analysis_error;
      EXPECT_EQ(report.complete, clobber_spill == 0) << diagnostic_summary(report);
      if (clobber_spill)
        EXPECT_NE(report.analysis_error.find("incomplete control flow"), std::string::npos)
            << report.analysis_error;
    }
  }
}

TEST(WaitcheckTest, Gfx1250NestedCallRequiresMatchingPhysicalLinkLaneBank) {
  constexpr auto arch = ROCJITSU_CODE_ARCH_CDNA5;
  for (uint32_t mode : {0x41u, 0x40u}) {
    SCOPED_TRACE(mode);
    std::vector<uint32_t> program;
    program.push_back(build_s_call_b64(30, 1, arch).value());
    program.push_back(build_s_endpgm(arch).value());
    append_inst(program, s_set_vgpr_msb(mode));
    program.insert(program.end(), {0xd7610002u, 0x0201001eu, // v_writelane v2,s30,0.
                                   0xd7610002u, 0x0201021fu}); // v_writelane v2,s31,1.
    const size_t inner_call = program.size();
    program.push_back(0);
    program.insert(program.end(), {0xd760001eu, 0x02010102u, // v_readlane s30,v2,0.
                                   0xd760001fu, 0x02010302u}); // v_readlane s31,v2,1.
    program.push_back(build_s_setpc_b64(30, arch).value());
    const size_t leaf = program.size();
    program[inner_call] = build_s_call_b64(30, static_cast<int>(leaf - inner_call - 1), arch).value();
    program.push_back(build_s_setpc_b64(30, arch).value());
    const auto image = waitcheck_test::make_gfx_code_object(
        program, EF_AMDGPU_MACH_AMDGCN_GFX1250, /*wave32=*/true);
    const AmdGpuCodeObject object(image.data(), image.size());
    const auto report = analyze_waitcnts_for_kernel(object, arch, 0);
    ASSERT_TRUE(report.supported) << report.analysis_error;
    EXPECT_EQ(report.complete, mode == 0x41) << diagnostic_summary(report);
    if (mode == 0x40)
      EXPECT_NE(report.analysis_error.find("incomplete control flow"), std::string::npos);
  }
}

TEST(WaitcheckTest, Gfx950NestedCallRestoresLinkFromProvenVgprLanes) {
  constexpr auto arch = ROCJITSU_CODE_ARCH_CDNA4;
  for (const auto [clobber_spill, enable_vskip] :
       {std::pair{false, false}, std::pair{true, false}, std::pair{false, true}}) {
    SCOPED_TRACE(testing::Message() << clobber_spill << '/' << enable_vskip);
    std::vector<uint32_t> program;
    program.push_back(build_s_call_b64(30, 1, arch).value()); // Helper at 0x08.
    program.push_back(build_s_endpgm(arch).value());
    if (enable_vskip)
      program.push_back(0xbf108081u); // VSKIP can suppress both lane writes.
    const size_t first_write = program.size();
    program.insert(program.end(), {0xd28a0002u, 0x0001001eu, // v_writelane v2,s30,0.
                                   0xd28a0002u, 0x0001021fu}); // v_writelane v2,s31,1.
    const size_t inner_call = program.size();
    program.push_back(0);
    if (clobber_spill)
      program.push_back(0x7e040300u); // v_mov_b32 v2,v0 destroys both saved lanes.
    if (enable_vskip)
      program.push_back(0xbf108080u); // Disable VSKIP before the lane reads.
    program.insert(program.end(), {0xd289001eu, 0x00010102u, // v_readlane s30,v2,0.
                                   0xd289001fu, 0x00010302u}); // v_readlane s31,v2,1.
    program.push_back(build_s_setpc_b64(30, arch).value());
    const size_t leaf = program.size();
    program[inner_call] = build_s_call_b64(30, static_cast<int>(leaf - inner_call - 1), arch).value();
    program.push_back(build_s_setpc_b64(30, arch).value());

    auto decoder = Decoder::create(arch);
    std::unique_ptr<Instruction> write(decode_valid(
        *decoder, &program[first_write], first_write * sizeof(uint32_t)));
    ASSERT_NE(write, nullptr);
    EXPECT_EQ(write->mnemonic(), "v_writelane_b32");
    EXPECT_EQ(write->num_src_operands(), 3);
    EXPECT_EQ(write->src_operand(1)->to_register_ref(),
              (RegisterRef{RegClass::SGPR, 30, 1}));
    EXPECT_EQ(write->src_operand(2)->const_value(), 0);
    EXPECT_TRUE(write->is_vskip_affected());
    const size_t read_index = inner_call + 1 + (clobber_spill ? 1 : 0) +
                              (enable_vskip ? 1 : 0);
    std::unique_ptr<Instruction> read(decode_valid(*decoder, &program[read_index],
                                                    read_index * sizeof(uint32_t)));
    ASSERT_NE(read, nullptr);
    EXPECT_EQ(read->mnemonic(), "v_readlane_b32");
    EXPECT_EQ(read->dst_operand(0)->to_register_ref(),
              (RegisterRef{RegClass::SGPR, 30, 1}));
    EXPECT_EQ(read->src_operand(0)->to_register_ref(),
              (RegisterRef{RegClass::VGPR, 2, 1}));
    EXPECT_EQ(read->src_operand(1)->const_value(), 0);
    EXPECT_TRUE(read->is_vskip_affected());

    const auto image = waitcheck_test::make_gfx_code_object(program, EF_AMDGPU_MACH_AMDGCN_GFX950);
    const AmdGpuCodeObject object(image.data(), image.size());
    const auto report = analyze_waitcnts_for_kernel(object, arch, 0);
    ASSERT_TRUE(report.supported) << report.analysis_error;
    EXPECT_EQ(report.complete, !clobber_spill && !enable_vskip)
        << report.analysis_error << '\n' << diagnostic_summary(report);
  }
}

TEST(WaitcheckTest, Gfx950SymbolledCalleePreservesAbiSavedLinkLane) {
  constexpr auto arch = ROCJITSU_CODE_ARCH_CDNA4;
  for (const auto [spill_vgpr, symbolled_leaf] :
       {std::pair{40u, true}, std::pair{40u, false}, std::pair{2u, true}}) {
    SCOPED_TRACE(testing::Message() << spill_vgpr << '/' << symbolled_leaf);
    std::vector<uint32_t> caller{build_s_call_b64(30, 1, arch).value(),
                                 build_s_endpgm(arch).value()};
    std::vector<uint32_t> outer{0xd28a0000u | spill_vgpr,
                                0x0001001eu, // Save s30 in lane 0.
                                0xd28a0000u | spill_vgpr,
                                0x0001021fu, // Save s31 in lane 1.
                                build_s_call_b64(30, 5, arch).value(),
                                0xd289001eu,
                                0x00010100u | spill_vgpr, // Restore s30.
                                0xd289001fu,
                                0x00010300u | spill_vgpr, // Restore s31.
                                build_s_setpc_b64(30, arch).value()};
    std::vector<uint32_t> leaf{0xdc70401cu,
                               0x00200000u | (spill_vgpr << 8),  // Save the VGPR in scratch.
                               0x7e000300u | (spill_vgpr << 17), // Clobber it temporarily.
                               0xdc50401cu,
                               (spill_vgpr << 24) | 0x00200000u, // Restore the VGPR.
                               build_s_setpc_b64(30, arch).value()};
    auto image = waitcheck_test::make_gfx_multi_kernel_code_object(
        {{"caller", caller}, {"outer", outer}, {"leaf", leaf}}, EF_AMDGPU_MACH_AMDGCN_GFX950);
    if (!symbolled_leaf) {
      const auto *ehdr = reinterpret_cast<const Elf64_Ehdr *>(image.data());
      const auto *sections = reinterpret_cast<const Elf64_Shdr *>(image.data() + ehdr->e_shoff);
      auto *symbols = reinterpret_cast<Elf64_Sym *>(image.data() + sections[3].sh_offset);
      symbols[6].st_info = 0; // Keep the leaf code but remove its ABI function symbol.
    }
    const AmdGpuCodeObject object(image.data(), image.size());
    const auto report = analyze_waitcnts_for_kernel(object, arch, 0);
    ASSERT_TRUE(report.supported) << report.analysis_error;
    EXPECT_EQ(report.complete, spill_vgpr == 40 && symbolled_leaf) << diagnostic_summary(report);
  }
}

TEST(WaitcheckTest, Gfx950SharedSwappcReturnsToMatchingContinuation) {
  constexpr uint16_t kTargetSreg = 8;
  constexpr uint16_t kReturnSreg = 30;
  constexpr uint16_t kLiteralOperand = 255;
  constexpr uint16_t kInlineInt0 = 128;

  // The branch selects exactly one call to the shared helper. The B1 path has
  // a pending VMEM load but waits in its own continuation. Pairing the helper's
  // return with the B0 continuation would create an impossible path to the B0
  // v0 consumer and a false missing-wait diagnostic.
  std::vector<uint32_t> program;
  program.push_back(0xBF840007u); // 0x00: s_cbranch_scc0 to B1 at 0x20.

  program.push_back(build_s_getpc_b64(kTargetSreg, ROCJITSU_CODE_ARCH_CDNA4).value()); // 0x04.
  program.push_back(build_s_add_u32(kTargetSreg, kTargetSreg, kLiteralOperand,
                                    ROCJITSU_CODE_ARCH_CDNA4)
                        .value()); // 0x08.
  program.push_back(60);           // Base 0x08 -> helper 0x44.
  program.push_back(build_s_addc_u32(kTargetSreg + 1, kTargetSreg + 1, kInlineInt0,
                                     ROCJITSU_CODE_ARCH_CDNA4)
                        .value()); // 0x10.
  program.push_back(
      build_s_swappc_b64(kReturnSreg, kTargetSreg, ROCJITSU_CODE_ARCH_CDNA4).value()); // 0x14.
  append_gfx950_v_mov_b32_v1_v0(program);                                              // 0x18 B0.
  program.push_back(build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4).value());                 // 0x1c.

  append_gfx950_buffer_load_dword_v0_v8_s0_offen(program);                             // 0x20 B1.
  program.push_back(build_s_getpc_b64(kTargetSreg, ROCJITSU_CODE_ARCH_CDNA4).value()); // 0x28.
  program.push_back(build_s_add_u32(kTargetSreg, kTargetSreg, kLiteralOperand,
                                    ROCJITSU_CODE_ARCH_CDNA4)
                        .value()); // 0x2c.
  program.push_back(24);           // Base 0x2c -> helper 0x44.
  program.push_back(build_s_addc_u32(kTargetSreg + 1, kTargetSreg + 1, kInlineInt0,
                                     ROCJITSU_CODE_ARCH_CDNA4)
                        .value()); // 0x34.
  program.push_back(
      build_s_swappc_b64(kReturnSreg, kTargetSreg, ROCJITSU_CODE_ARCH_CDNA4).value()); // 0x38.
  append_gfx950_s_waitcnt_vmcnt_0(program);                                            // 0x3c B1.
  program.push_back(build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4).value());                 // 0x40.
  program.push_back(build_sop1_encoding(ROCJITSU_CODE_ARCH_CDNA4, 0x1d, 0,
                                        kReturnSreg)
                        .value()); // 0x44 shared return.

  const auto image =
      rocjitsu::waitcheck_test::make_gfx_code_object(program, EF_AMDGPU_MACH_AMDGCN_GFX950);
  AmdGpuCodeObject code_object(image.data(), image.size());
  auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_CDNA4);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx950RecursiveHelperReportsIncomplete) {
  constexpr uint16_t kReturnSreg = 30;

  // The helper recursively calls its own entry before waiting for a VMEM load.
  // A finite context graph cannot prove its eventual return state.
  std::vector<uint32_t> program;
  program.push_back(
      build_s_call_b64(kReturnSreg, 2, ROCJITSU_CODE_ARCH_CDNA4).value()); // 0x00 -> 0x0c.
  append_gfx950_v_mov_b32_v1_v0(program);                                  // 0x04.
  program.push_back(build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4).value());     // 0x08.
  append_gfx950_buffer_load_dword_v0_v8_s0_offen(program);                 // 0x0c helper.
  program.push_back(
      build_s_call_b64(kReturnSreg, -3, ROCJITSU_CODE_ARCH_CDNA4).value()); // 0x14 -> 0x0c.
  append_gfx950_s_waitcnt_vmcnt_0(program);                                 // 0x18.
  program.push_back(
      build_sop1_encoding(ROCJITSU_CODE_ARCH_CDNA4, 0x1d, 0, kReturnSreg).value()); // 0x1c return.

  const auto image =
      rocjitsu::waitcheck_test::make_gfx_code_object(program, EF_AMDGPU_MACH_AMDGCN_GFX950);
  AmdGpuCodeObject code_object(image.data(), image.size());
  auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_CDNA4);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_FALSE(report.complete);
  EXPECT_FALSE(report.passed());
  EXPECT_FALSE(report.analysis_error.empty());
}

TEST(WaitcheckTest, Gfx950ReturningRecursiveCallCannotSkipCalleeLoad) {
  // A conditional helper recurses once and returns a load to the outer
  // continuation. A direct leaf call exposes the missing VMEM wait; a
  // recursive call must not be reported clean by skipping the inner callee.
  const std::vector<uint32_t> recursive{
      0xbe940081u, 0xba9e0001u, 0xbf810000u, 0xbf068014u, 0xbf850006u,
      0xbea2011eu, 0xbe940080u, 0xba9efffbu, 0x7e020300u, 0xbe9e0122u,
      0xbe801d1eu, 0xe0501000u, 0x80000008u, 0xbe801d1eu};
  const auto check = [](const std::vector<uint32_t> &program) {
    const auto image =
        waitcheck_test::make_gfx_code_object(program, EF_AMDGPU_MACH_AMDGCN_GFX950);
    AmdGpuCodeObject code_object(image.data(), image.size());
    EXPECT_TRUE(code_object.is_valid());
    return analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_CDNA4);
  };

  const auto recursive_report = check(recursive);
  EXPECT_TRUE(recursive_report.supported) << recursive_report.analysis_error;
  EXPECT_FALSE(recursive_report.complete);
  EXPECT_FALSE(recursive_report.passed());
  EXPECT_NE(recursive_report.analysis_error.find("recursive call"), std::string::npos);

  auto direct_leaf = recursive;
  direct_leaf[7] = 0xba9e0003u; // Retarget inner call to the leaf load.
  const auto leaf_report = check(direct_leaf);
  ASSERT_TRUE(leaf_report.supported && leaf_report.complete) << leaf_report.analysis_error;
  ASSERT_EQ(leaf_report.diagnostics.size(), 1u) << diagnostic_summary(leaf_report);
  EXPECT_EQ(leaf_report.diagnostics.front().counter, WaitCounterKind::Load);
}

TEST(WaitcheckTest, ExecutableSectionOutsideTextReportsIncomplete) {
  // The branch can skip the wait between the load and its use. The same
  // instructions in a non-text executable section lack CFG coverage.
  const std::vector<uint32_t> program{0xee05007cu, 0u, 0x8u, 0xbfa20001u,
                                      0xbfc00000u, 0x7e020300u, 0xbfb00000u};
  auto image = waitcheck_test::make_gfx_code_object(program, EF_AMDGPU_MACH_AMDGCN_GFX1200);
  AmdGpuCodeObject text_object(image.data(), image.size());
  ASSERT_TRUE(text_object.is_valid());
  const auto text_report = analyze_waitcnts(text_object, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(text_report.supported && text_report.complete) << text_report.analysis_error;
  ASSERT_EQ(text_report.diagnostics.size(), 1u) << diagnostic_summary(text_report);

  const auto raw_report = analyze_waitcnts(std::span<const uint32_t>(program),
                                          ROCJITSU_CODE_ARCH_RDNA4);
  EXPECT_TRUE(raw_report.supported) << raw_report.analysis_error;
  EXPECT_FALSE(raw_report.complete) << diagnostic_summary(raw_report);
  EXPECT_FALSE(raw_report.passed());
  EXPECT_NE(raw_report.analysis_error.find("control flow"), std::string::npos);

  auto extra_executable = image;
  auto *extra_ehdr = reinterpret_cast<Elf64_Ehdr *>(extra_executable.data());
  auto *extra_sections = reinterpret_cast<Elf64_Shdr *>(extra_executable.data() +
                                                         extra_ehdr->e_shoff);
  extra_sections[2].sh_flags |= SHF_EXECINSTR; // Unreferenced executable .rodata.
  AmdGpuCodeObject extra_object(extra_executable.data(), extra_executable.size());
  ASSERT_TRUE(extra_object.is_valid());
  const auto whole_extra = analyze_waitcnts(extra_object, ROCJITSU_CODE_ARCH_RDNA4);
  EXPECT_FALSE(whole_extra.complete);
  const auto selected_offset =
      analyze_waitcnts_for_kernel(extra_object, ROCJITSU_CODE_ARCH_RDNA4, 0);
  ASSERT_TRUE(selected_offset.supported && selected_offset.complete)
      << selected_offset.analysis_error;
  EXPECT_EQ(selected_offset.diagnostics.size(), 1u) << diagnostic_summary(selected_offset);
  const auto kernels = waitcheck_kernels(extra_object);
  ASSERT_EQ(kernels.size(), 1u);
  const auto batch =
      analyze_waitcnts_for_kernels(extra_object, ROCJITSU_CODE_ARCH_RDNA4, kernels);
  EXPECT_FALSE(batch.complete);
  EXPECT_FALSE(batch.passed());
  const auto selected_descriptor =
      analyze_waitcnts_for_kernel(extra_object, ROCJITSU_CODE_ARCH_RDNA4, kernels[0]);
  ASSERT_TRUE(selected_descriptor.supported && selected_descriptor.complete)
      << selected_descriptor.analysis_error;
  EXPECT_EQ(selected_descriptor.diagnostics.size(), 1u)
      << diagnostic_summary(selected_descriptor);

  Elf64_Ehdr ehdr{};
  std::memcpy(&ehdr, image.data(), sizeof(ehdr));
  Elf64_Shdr text_section{}, section_names{};
  std::memcpy(&text_section, image.data() + ehdr.e_shoff + ehdr.e_shentsize,
              sizeof(text_section));
  std::memcpy(&section_names,
              image.data() + ehdr.e_shoff + ehdr.e_shentsize * ehdr.e_shstrndx,
              sizeof(section_names));
  std::memcpy(image.data() + section_names.sh_offset + text_section.sh_name, ".code", 5);
  AmdGpuCodeObject code_object(image.data(), image.size());
  ASSERT_TRUE(code_object.is_valid());
  const auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_RDNA4);
  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_FALSE(report.complete);
  EXPECT_FALSE(report.passed());
  EXPECT_NE(report.analysis_error.find(".code"), std::string::npos);
}

} // namespace
