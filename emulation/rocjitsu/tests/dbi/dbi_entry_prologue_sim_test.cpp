// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file dbi_entry_prologue_sim_test.cpp
/// @brief Simulator end-to-end for the DBI kernel-entry prologue on CDNA3
/// (gfx942), CDNA4 (gfx950), and RDNA4 (gfx1200).
///
/// The RDNA4 code object is patched as Wave32, but DbiSim dispatches Wave64
/// regardless (see dbi_sim.h). The prologue is pure scalar, so nothing here
/// depends on the executing wave size; the lanes read back are lanes 0-31.
///
/// The static tests (tests/patch/instrumentor_test.cpp,
/// tests/patch/entry_prologue_test.cpp) prove the patched ELF *contains* the
/// prologue and declares the wrapper it reads. This is the only thing in the
/// series that runs it.
///
/// What execution adds is the reason the prologue exists: a probe at an
/// arbitrary site cannot be handed a launch value, because no register holding
/// one survives that far. The delivery kernel below **destroys the kernarg SGPR
/// pair before the anchor**, so a probe that still receives the pointer received
/// it from the framework's reserved storage and not from a register that
/// happened to survive. Without that clobber the test would pass on a design
/// that never needed a prologue at all.
///
/// Two kernels, because the two halves of the prologue are observed differently.
///
/// Delivery (entry at .text offset 0):
///   s_mov_b32 s0, 0   ; offset 0:  ENTRY, so the prologue anchors here and this
///                     ;            runs after it, destroying the kernarg pair
///   s_mov_b32 s1, 0   ; offset 4
///   v_mov_b32 v1, v0  ; offset 8:  ANCHOR for the probe site
///   s_endpgm          ; offset 12
/// Probe: { v_mov_b32 v2, v0 ; v_mov_b32 v3, v1 ; s_setpc_b64 s[30:31] },
/// publishing its two argument dwords, so v[2:3] must be the payload pointer.
///
/// Restore (entry at .text offset 0):
///   s_nop             ; offset 0:  ENTRY
///   v_mov_b32 v1, v0  ; offset 4:  ANCHOR for the probe site
///   v_mov_b32 v2, s0  ; offset 8:  the guest reads the restored kernarg pair
///   v_mov_b32 v3, s1  ; offset 12
///   s_endpgm          ; offset 16
/// v[2:3] must be the guest's own pointer rather than the wrapper the CP
/// delivered. The guest reading the pair is also what keeps it live across the
/// anchor: dead registers are what the call envelope takes for its own temps,
/// so a kernel that never used s[0:1] again would see them overwritten by the
/// trampoline rather than by anything the prologue did.
///
/// @note No nop-the-wait control. The simulator retires scalar loads
/// synchronously, so removing the prologue's wait changes nothing observable and
/// the control would fail to fail. Separately, build_s_load_dwordx2 always sets
/// RDNA4's SMEM soffset to NULL, so although the model does read that field,
/// nothing here exercises a non-NULL value. Both need hardware.

#include "../dbi_test_util.h"
#include "dbi_sim.h"

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/code/builders/smem_builders.h"
#include "rocjitsu/code/builders/vector_builders.h"
#include "rocjitsu/code/patch/entry_prologue.h"
#include "rocjitsu/code/patch/instrumentor.h"
#include "rocjitsu/code/patch/kernarg_extension.h"
#include "rocjitsu/code/rj_code.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <span>
#include <vector>

namespace rocjitsu {
namespace {

// The value the harness puts in the wrapper's DBI payload. The probe must
// publish exactly this.
constexpr uint64_t kLogBufferSentinel = 0xCAFEF00DDEADBEEFull;
// The guest's own kernarg pointer, recorded in the wrapper for the prologue to
// restore. Distinct from DbiSim::KERNARG_ADDR so the restore is distinguishable
// from the prologue having left the CP's pointer in place.
constexpr uint64_t kGuestKernargSentinel = 0x0000BEEF00001000ull;
// Not payload-aligned, so the payload offset is a value only this layout gives.
constexpr uint32_t kGuestKernargSize = 20;

// v0 and v1 are the first two argument VGPRs, so the probe publishes into v2 and
// v3. That keeps the fixture inside the four ordinary VGPRs the default
// descriptor grants (see dbi_arg_sim_test.cpp on the AGPR window at v4).
constexpr uint16_t kVgprSrcBase = 256; // VGPR n is scalar-source code 256 + n.

struct PrologueSimArch {
  const char *sim_arch;
  rj_code_arch_t arch;
  uint32_t e_flags;
  uint32_t wave_size;
  bool wave32;
};

constexpr PrologueSimArch kCdna3{"cdna3", ROCJITSU_CODE_ARCH_CDNA3, EF_AMDGPU_MACH_AMDGCN_GFX942,
                                 64, false};
constexpr PrologueSimArch kCdna4{"cdna4", ROCJITSU_CODE_ARCH_CDNA4, EF_AMDGPU_MACH_AMDGCN_GFX950,
                                 64, false};
constexpr PrologueSimArch kRdna4{"rdna4", ROCJITSU_CODE_ARCH_RDNA4, EF_AMDGPU_MACH_AMDGCN_GFX1200,
                                 32, true};

// The kernarg image the CP delivers: the guest's own kernargs, then its
// pointer, then the DBI payload. Built through the production helper, so the
// offsets the prologue loads from are the ones a runtime would write to.
std::vector<uint8_t> make_wrapper_image() {
  const std::array<KernargExtensionPayloadLayout, 1> payloads{kDbiEntryPayloadLayout};
  const auto layout = make_kernarg_extension_layout(kGuestKernargSize, payloads);
  if (!layout)
    return {};

  const std::vector<uint8_t> guest_kernargs(kGuestKernargSize, 0x5A);
  const KernargExtensionPayloadWrite payload{&kLogBufferSentinel, sizeof(kLogBufferSentinel)};
  std::vector<uint8_t> wrapper(layout->wrapper_size, 0);
  if (!write_kernarg_extension_wrapper(wrapper, *layout, guest_kernargs.data(),
                                       kGuestKernargSentinel, std::span{&payload, 1}))
    return {};
  return wrapper;
}

class DbiEntryPrologueSimBase : public ::testing::Test {
protected:
  explicit DbiEntryPrologueSimBase(const PrologueSimArch &a) : a_(a) {}

  void SetUp() override {
    wrapper_ = make_wrapper_image();
    ASSERT_FALSE(wrapper_.empty()) << "could not build the kernarg wrapper";

    const uint32_t endpgm = build_s_endpgm(a_.arch);
    const uint32_t setpc = build_s_setpc_b64(/*s[30:31]=*/30, a_.arch);
    // Inline constant 0 is scalar-source code 128.
    const uint32_t clobber_s0 = build_s_mov_b32(0, 128, a_.arch);
    const uint32_t clobber_s1 = build_s_mov_b32(1, 128, a_.arch);
    const uint32_t guest_anchor = build_v_mov_b32_src(1, kVgprSrcBase + 0, a_.arch);

    const std::vector<uint32_t> publish{build_v_mov_b32_src(2, kVgprSrcBase + 0, a_.arch),
                                        build_v_mov_b32_src(3, kVgprSrcBase + 1, a_.arch), setpc};
    auto probe = test::make_amdgpu_probe_elf("rj_test_log_probe", publish, a_.e_flags);
    AmdGpuCodeObject probe_obj(probe.data(), probe.size());
    ASSERT_TRUE(probe_obj.is_valid());

    ASSERT_NO_FATAL_FAILURE(patch({clobber_s0, clobber_s1, guest_anchor, endpgm},
                                  /*anchor_offset=*/8, probe_obj, delivery_text_,
                                  delivery_scratch_));
    ASSERT_NO_FATAL_FAILURE(
        patch({build_s_nop(0, a_.arch), guest_anchor, build_v_mov_b32_src(2, 0, a_.arch),
               build_v_mov_b32_src(3, 1, a_.arch), endpgm},
              /*anchor_offset=*/4, probe_obj, restore_text_, restore_scratch_));
  }

  // The prologue anchors the kernel entry, so a site there would collide with
  // it. Both fixtures put theirs on a later word, which is why this takes the
  // offset rather than fixing one.
  void patch(const std::vector<uint32_t> &text, uint64_t anchor_offset,
             const AmdGpuCodeObject &probe_obj, std::vector<uint32_t> &text_out,
             uint32_t &scratch_out) {
    auto target = test::make_kernarg_kernel_elf(text, /*private_bytes=*/64, a_.e_flags,
                                                kGuestKernargSize, a_.wave32);
    AmdGpuCodeObject obj(target.data(), target.size());
    ASSERT_TRUE(obj.is_valid());

    Instrumentor instr(obj, a_.arch);
    InstrumentationPoint pt;
    pt.anchor_offset = anchor_offset;
    pt.probe_obj = &probe_obj;
    pt.probe_symbol = "rj_test_log_probe";
    pt.probe_args = {{ProbeArgSource::LogBufferPtrLo, 0}, {ProbeArgSource::LogBufferPtrHi, 0}};
    instr.add_point(pt);

    auto result = instr.patch_with_debug_summaries();
    ASSERT_TRUE(result.errors.empty())
        << (result.errors.empty() ? std::string{} : result.errors.front());
    ASSERT_EQ(result.patches.size(), 1u);

    AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
    ASSERT_TRUE(patched.is_valid());
    text_out = test::section_words(patched, ".text");
    ASSERT_FALSE(text_out.empty());
    scratch_out = test::patched_private_segment_size(patched);
  }

  test::DbiSim make_sim() {
    test::DbiSim sim(a_.sim_arch, a_.wave_size);
    sim.set_kernarg(wrapper_);
    return sim;
  }

  // The kernarg pair is destroyed between the prologue and the site, and the
  // probe still receives the pointer.
  void expect_probe_receives_the_payload_pointer() {
    test::DbiSim sim = make_sim();
    const std::vector<std::vector<uint32_t>> regs =
        sim.run_and_read_vgprs(delivery_text_, delivery_scratch_, {/*v2=*/2, /*v3=*/3});
    ASSERT_EQ(regs[0].size(), a_.wave_size) << "kernel did not run to completion";
    ASSERT_EQ(regs[1].size(), a_.wave_size);

    for (uint32_t lane = 0; lane < a_.wave_size; ++lane) {
      const uint64_t seen =
          (static_cast<uint64_t>(regs[1][lane]) << 32) | static_cast<uint64_t>(regs[0][lane]);
      EXPECT_EQ(seen, kLogBufferSentinel)
          << "lane " << lane << ": probe did not receive the payload pointer";
    }
  }

  // Negative control: the pointer comes from the prologue's load, not from
  // anything the wave happened to hold. Nop that load and it must not arrive.
  void expect_without_the_payload_load_the_pointer_does_not_arrive() {
    std::vector<uint32_t> sabotaged = delivery_text_;
    ASSERT_NO_FATAL_FAILURE(nop_payload_load(sabotaged));

    test::DbiSim sim = make_sim();
    const std::vector<std::vector<uint32_t>> regs =
        sim.run_and_read_vgprs(sabotaged, delivery_scratch_, {/*v2=*/2, /*v3=*/3});
    ASSERT_EQ(regs[0].size(), a_.wave_size) << "kernel did not run to completion";

    for (uint32_t lane = 0; lane < a_.wave_size; ++lane) {
      const uint64_t seen =
          (static_cast<uint64_t>(regs[1][lane]) << 32) | static_cast<uint64_t>(regs[0][lane]);
      EXPECT_NE(seen, kLogBufferSentinel)
          << "lane " << lane << ": the pointer arrived without the prologue loading it";
    }
  }

  // The transparency half. The CP hands the kernel a pointer to the wrapper, so
  // without the restore every kernarg-relative guest access reads through the
  // wrapper prefix at the wrong base.
  void expect_guest_kernarg_pointer_is_restored() {
    test::DbiSim sim = make_sim();
    const uint64_t seen = read_guest_kernarg_pointer(sim, restore_text_);
    EXPECT_EQ(seen, kGuestKernargSentinel)
        << "the guest's kernarg pointer was not restored at entry";
    EXPECT_NE(seen, test::DbiSim::KERNARG_ADDR)
        << "the guest still sees the wrapper pointer the CP delivered";
  }

  // The pointer the guest itself reads out of s[0:1] after the anchor, as the
  // two halves it copied into v2 and v3.
  uint64_t read_guest_kernarg_pointer(test::DbiSim &sim, const std::vector<uint32_t> &text) {
    const std::vector<std::vector<uint32_t>> regs =
        sim.run_and_read_vgprs(text, restore_scratch_, {/*v2=*/2, /*v3=*/3});
    if (regs[0].size() != a_.wave_size)
      return 0;
    return (static_cast<uint64_t>(regs[1][0]) << 32) | static_cast<uint64_t>(regs[0][0]);
  }

  // Negative control for the restore: with the two moves gone, the kernel keeps
  // the wrapper pointer. Confirms the restore is what put the guest's back.
  void expect_without_the_restore_the_wrapper_pointer_remains() {
    std::vector<uint32_t> sabotaged = restore_text_;
    const uint32_t nop = build_s_nop(0, a_.arch);
    size_t replaced = 0;
    for (uint32_t &word : sabotaged) {
      // The restores are the only s_mov_b32 into s0/s1 in this kernel.
      for (uint16_t dst = 0; dst < 2; ++dst) {
        for (uint16_t src = 0; src < REGISTER_SET_ALLOCATABLE_SGPRS; ++src) {
          if (word == build_s_mov_b32(dst, src, a_.arch)) {
            word = nop;
            ++replaced;
          }
        }
      }
    }
    ASSERT_EQ(replaced, 2u) << "did not find both kernarg-pointer restores";

    test::DbiSim sim = make_sim();
    EXPECT_EQ(read_guest_kernarg_pointer(sim, sabotaged), test::DbiSim::KERNARG_ADDR)
        << "without the restore the guest must still see the CP's wrapper pointer";
  }

  // Replace the prologue's payload load with nops. Both halves of the 64-bit
  // encoding go, since a bare s_nop over the low word would leave the high word
  // to decode as something else.
  //
  // The exact encoding is reconstructed rather than matched by mask. The storage
  // pair is not known here, so every even destination is tried against the one
  // address pair (s[0:1], the descriptor's only user SGPR) and the payload offset
  // the layout gives. Masking the opcode field instead would be wrong in a way
  // that happens to work: the SMEM opcode sits at bits 18-25 on CDNA and 13-18 on
  // RDNA4, so any mask wide enough to be arch-neutral matches unrelated scalar
  // memory ops, and this would then nop whichever came first.
  void nop_payload_load(std::vector<uint32_t> &words) {
    const std::array<KernargExtensionPayloadLayout, 1> payloads{kDbiEntryPayloadLayout};
    const auto layout = make_kernarg_extension_layout(kGuestKernargSize, payloads);
    ASSERT_TRUE(layout.has_value());
    const uint32_t nop = build_s_nop(0, a_.arch);

    for (uint16_t sdst = 0; sdst < REGISTER_SET_ALLOCATABLE_SGPRS - 1; sdst += 2) {
      const auto load =
          build_s_load_dwordx2(sdst, /*sbase=*/0, layout->payload_offsets.front(), a_.arch);
      for (size_t i = 0; i + 1 < words.size(); ++i) {
        if (words[i] != load[0] || words[i + 1] != load[1])
          continue;
        words[i] = nop;
        words[i + 1] = nop;
        return;
      }
    }
    FAIL() << "the prologue's payload load was not found in the patched text";
  }

  PrologueSimArch a_;
  std::vector<uint8_t> wrapper_;
  std::vector<uint32_t> delivery_text_;
  uint32_t delivery_scratch_ = 0;
  std::vector<uint32_t> restore_text_;
  uint32_t restore_scratch_ = 0;
};

class DbiCdna3EntryPrologueSim : public DbiEntryPrologueSimBase {
protected:
  DbiCdna3EntryPrologueSim() : DbiEntryPrologueSimBase(kCdna3) {}
};
class DbiCdna4EntryPrologueSim : public DbiEntryPrologueSimBase {
protected:
  DbiCdna4EntryPrologueSim() : DbiEntryPrologueSimBase(kCdna4) {}
};
class DbiRdna4EntryPrologueSim : public DbiEntryPrologueSimBase {
protected:
  DbiRdna4EntryPrologueSim() : DbiEntryPrologueSimBase(kRdna4) {}
};

TEST_F(DbiCdna3EntryPrologueSim, ProbeReceivesThePayloadPointer) {
  expect_probe_receives_the_payload_pointer();
}
TEST_F(DbiCdna3EntryPrologueSim, WithoutThePayloadLoadThePointerDoesNotArrive) {
  expect_without_the_payload_load_the_pointer_does_not_arrive();
}
TEST_F(DbiCdna3EntryPrologueSim, GuestKernargPointerIsRestored) {
  expect_guest_kernarg_pointer_is_restored();
}
TEST_F(DbiCdna3EntryPrologueSim, WithoutTheRestoreTheWrapperPointerRemains) {
  expect_without_the_restore_the_wrapper_pointer_remains();
}

TEST_F(DbiCdna4EntryPrologueSim, ProbeReceivesThePayloadPointer) {
  expect_probe_receives_the_payload_pointer();
}
TEST_F(DbiCdna4EntryPrologueSim, WithoutThePayloadLoadThePointerDoesNotArrive) {
  expect_without_the_payload_load_the_pointer_does_not_arrive();
}
TEST_F(DbiCdna4EntryPrologueSim, GuestKernargPointerIsRestored) {
  expect_guest_kernarg_pointer_is_restored();
}
TEST_F(DbiCdna4EntryPrologueSim, WithoutTheRestoreTheWrapperPointerRemains) {
  expect_without_the_restore_the_wrapper_pointer_remains();
}

TEST_F(DbiRdna4EntryPrologueSim, ProbeReceivesThePayloadPointer) {
  expect_probe_receives_the_payload_pointer();
}
TEST_F(DbiRdna4EntryPrologueSim, WithoutThePayloadLoadThePointerDoesNotArrive) {
  expect_without_the_payload_load_the_pointer_does_not_arrive();
}
TEST_F(DbiRdna4EntryPrologueSim, GuestKernargPointerIsRestored) {
  expect_guest_kernarg_pointer_is_restored();
}
TEST_F(DbiRdna4EntryPrologueSim, WithoutTheRestoreTheWrapperPointerRemains) {
  expect_without_the_restore_the_wrapper_pointer_remains();
}

} // namespace
} // namespace rocjitsu
