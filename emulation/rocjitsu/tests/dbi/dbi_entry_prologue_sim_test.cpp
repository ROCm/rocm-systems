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
///   s_mov_b32 s0, 0   ; offset 0:  ENTRY. Dispatch enters the stub, which runs
///                     ;            the prologue and branches here, so this
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
/// Call-and-return variants of both kernels put an s_call_b64 to a helper
/// returning through s_setpc_b64 s[4:5] between the entry and the anchor, which
/// then sits on the call's continuation. The return goes to the continuation,
/// not the entry, so these show a call and an indirect return leave the
/// prologue's storage and the restored kernarg pointer intact.
///
/// Call delivery:                          Call restore:
///   s_mov_b32 s0, 0       ; 0:  ENTRY       s_nop                 ; 0:  ENTRY
///   s_mov_b32 s1, 0       ; 4               s_call_b64 s[4:5], 24 ; 4
///   s_call_b64 s[4:5], 20 ; 8               v_mov_b32 v1, v0      ; 8:  ANCHOR
///   v_mov_b32 v1, v0      ; 12: ANCHOR      v_mov_b32 v2, s0      ; 12
///   s_endpgm              ; 16              v_mov_b32 v3, s1      ; 16
///   s_setpc_b64 s[4:5]    ; 20: helper      s_endpgm              ; 20
///                                           s_setpc_b64 s[4:5]    ; 24: helper
///
/// Entry-point variants put the site on the entry itself ({v_mov_b32 v1, v0;
/// s_endpgm}, plus the two reads for restore), so the stub branches into the
/// entry's spliced trampoline.
///
/// Re-entry variants return to the original entry once more after the first
/// pass, counting passes in s6 (the simulator starts every SGPR at zero):
///
/// Loop restore:                           Rewritten-return restore:
///   s_add_u32 s6, s6, 1     ; 0:  ENTRY     s_add_u32 s6, s6, 1     ; 0:  ENTRY
///   v_mov_b32 v1, v0        ; 4:  ANCHOR    v_mov_b32 v1, v0        ; 4:  ANCHOR
///   s_cmp_lg_u32 s6, 2      ; 8             s_cmp_lg_u32 s6, 2      ; 8
///   s_cbranch_scc1 0        ; 12            s_cbranch_scc0 20       ; 12
///   v_mov_b32 v2, s0        ; 16            s_call_b64 s[4:5], 32   ; 16
///   v_mov_b32 v3, s1        ; 20            v_mov_b32 v2, s0        ; 20
///   s_endpgm                ; 24            v_mov_b32 v3, s1        ; 24
///                                           s_endpgm                ; 28
///                                           s_add_u32 s4, s4, -20   ; 32: helper
///                                           s_addc_u32 s5, s5, -1   ; 40
///                                           s_setpc_b64 s[4:5]      ; 44
///
/// Indirect-jump restore builds the entry's address from s_getpc_b64 and jumps
/// to it, with no call involved:
///   s_add_u32 s6, s6, 1     ; 0:  ENTRY
///   v_mov_b32 v1, v0        ; 4:  ANCHOR
///   s_cmp_lg_u32 s6, 2      ; 8
///   s_cbranch_scc0 36       ; 12
///   s_getpc_b64 s[4:5]      ; 16: s[4:5] = address of offset 20
///   s_add_u32 s4, s4, -20   ; 20
///   s_addc_u32 s5, s5, -1   ; 28
///   s_setpc_b64 s[4:5]      ; 32
///   v_mov_b32 v2, s0        ; 36
///   v_mov_b32 v3, s1        ; 40
///   s_endpgm                ; 44
/// The delivery variants end at s_endpgm where the restore variants read s[0:1].
/// The descriptor enters at the prologue's stub, so the second pass through the
/// original entry runs guest code only; a second prologue run would reload both
/// the storage and the kernarg pair through the guest's own pointer.
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

// s_cbranch_scc0 / s_cbranch_scc1, which the shared builders do not cover.
uint32_t build_s_cbranch_scc(bool scc1, int16_t offset_dwords, rj_code_arch_t arch) {
  uint16_t op = 0;
  switch (arch) {
  case ROCJITSU_CODE_ARCH_CDNA3:
    op = scc1 ? cdna3::kSCbranchScc1Sopp : cdna3::kSCbranchScc0Sopp;
    break;
  case ROCJITSU_CODE_ARCH_CDNA4:
    op = scc1 ? cdna4::kSCbranchScc1Sopp : cdna4::kSCbranchScc0Sopp;
    break;
  case ROCJITSU_CODE_ARCH_RDNA4:
    op = scc1 ? rdna4::kSCbranchScc1Sopp : rdna4::kSCbranchScc0Sopp;
    break;
  default:
    ADD_FAILURE() << "no s_cbranch_scc opcode for this target";
    break;
  }
  return build_sopp_encoding(arch, op, static_cast<uint16_t>(offset_dwords));
}

// Scalar-source codes for inline integer constants.
constexpr uint16_t kInline1 = 129;
constexpr uint16_t kInline2 = 130;
constexpr uint16_t kInlineMinus1 = 193;
constexpr uint16_t kLiteral = 255;

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
                                  /*anchor_offset=*/8, probe_obj, delivery_));
    ASSERT_NO_FATAL_FAILURE(
        patch({build_s_nop(0, a_.arch), guest_anchor, build_v_mov_b32_src(2, 0, a_.arch),
               build_v_mov_b32_src(3, 1, a_.arch), endpgm},
              /*anchor_offset=*/4, probe_obj, restore_));

    // s_call_b64's immediate is a signed dword offset from the next instruction.
    const uint32_t ret = build_s_setpc_b64(/*s[4:5]=*/4, a_.arch);
    ASSERT_NO_FATAL_FAILURE(
        patch({clobber_s0, clobber_s1, build_s_call_b64(4, 2, a_.arch), guest_anchor, endpgm, ret},
              /*anchor_offset=*/12, probe_obj, call_delivery_));
    ASSERT_NO_FATAL_FAILURE(
        patch({build_s_nop(0, a_.arch), build_s_call_b64(4, 4, a_.arch), guest_anchor,
               build_v_mov_b32_src(2, 0, a_.arch), build_v_mov_b32_src(3, 1, a_.arch), endpgm, ret},
              /*anchor_offset=*/8, probe_obj, call_restore_));

    // A point on the entry itself: the stub runs the prologue, then branches to
    // the entry, which is spliced to the site's trampoline.
    ASSERT_NO_FATAL_FAILURE(
        patch({guest_anchor, endpgm}, /*anchor_offset=*/0, probe_obj, entry_point_delivery_));
    ASSERT_NO_FATAL_FAILURE(patch({guest_anchor, build_v_mov_b32_src(2, 0, a_.arch),
                                   build_v_mov_b32_src(3, 1, a_.arch), endpgm},
                                  /*anchor_offset=*/0, probe_obj, entry_point_restore_));

    // Re-entry kernels. s6 counts passes through the entry; the second pass
    // leaves the loop or skips the call or jump.
    const uint32_t count_pass = build_s_add_u32(6, 6, kInline1, a_.arch);
    const uint32_t second_pass = build_s_cmp_lg_u32(6, kInline2, a_.arch);
    const uint32_t read_s0 = build_v_mov_b32_src(2, 0, a_.arch);
    const uint32_t read_s1 = build_v_mov_b32_src(3, 1, a_.arch);
    const uint32_t loop_to_entry = build_s_cbranch_scc(/*scc1=*/true, -4, a_.arch);
    ASSERT_NO_FATAL_FAILURE(patch({count_pass, guest_anchor, second_pass, loop_to_entry, endpgm},
                                  /*anchor_offset=*/4, probe_obj, loop_delivery_));
    ASSERT_NO_FATAL_FAILURE(
        patch({count_pass, guest_anchor, second_pass, loop_to_entry, read_s0, read_s1, endpgm},
              /*anchor_offset=*/4, probe_obj, loop_restore_));

    // The helper subtracts the call's return offset from the saved address, so
    // it returns to the entry instead of the continuation.
    const std::vector<uint32_t> rewrite_return{build_s_add_u32(4, 4, kLiteral, a_.arch),
                                               static_cast<uint32_t>(-20),
                                               build_s_addc_u32(5, 5, kInlineMinus1, a_.arch), ret};
    std::vector<uint32_t> words{count_pass,
                                guest_anchor,
                                second_pass,
                                build_s_cbranch_scc(/*scc1=*/false, 1, a_.arch),
                                build_s_call_b64(4, 1, a_.arch),
                                endpgm};
    words.insert(words.end(), rewrite_return.begin(), rewrite_return.end());
    ASSERT_NO_FATAL_FAILURE(patch(words, /*anchor_offset=*/4, probe_obj, return_delivery_));
    words = {count_pass,
             guest_anchor,
             second_pass,
             build_s_cbranch_scc(/*scc1=*/false, 1, a_.arch),
             build_s_call_b64(4, 3, a_.arch),
             read_s0,
             read_s1,
             endpgm};
    words.insert(words.end(), rewrite_return.begin(), rewrite_return.end());
    ASSERT_NO_FATAL_FAILURE(patch(words, /*anchor_offset=*/4, probe_obj, return_restore_));

    // A plain indirect jump to the entry: the address comes from s_getpc_b64,
    // not from a call's saved return address.
    const std::vector<uint32_t> jump_to_entry{build_s_cmp_lg_u32(6, kInline2, a_.arch),
                                              build_s_cbranch_scc(/*scc1=*/false, 5, a_.arch),
                                              build_s_getpc_b64(4, a_.arch),
                                              build_s_add_u32(4, 4, kLiteral, a_.arch),
                                              static_cast<uint32_t>(-20),
                                              build_s_addc_u32(5, 5, kInlineMinus1, a_.arch),
                                              ret};
    words = {count_pass, guest_anchor};
    words.insert(words.end(), jump_to_entry.begin(), jump_to_entry.end());
    words.push_back(endpgm);
    ASSERT_NO_FATAL_FAILURE(patch(words, /*anchor_offset=*/4, probe_obj, jump_delivery_));
    words = {count_pass, guest_anchor};
    words.insert(words.end(), jump_to_entry.begin(), jump_to_entry.end());
    words.insert(words.end(), {read_s0, read_s1, endpgm});
    ASSERT_NO_FATAL_FAILURE(patch(words, /*anchor_offset=*/4, probe_obj, jump_restore_));
  }

  // What dispatching one patched kernel needs: its words, its scratch, and the
  // stub the patched descriptor enters at.
  struct PatchedKernel {
    std::vector<uint32_t> text;
    uint32_t scratch = 0;
    uint64_t entry = 0;
  };

  // The fixtures put their site on different words, which is why this takes the
  // offset rather than fixing one.
  void patch(const std::vector<uint32_t> &text, uint64_t anchor_offset,
             const AmdGpuCodeObject &probe_obj, PatchedKernel &out) {
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
    out.text = test::section_words(patched, ".text");
    ASSERT_FALSE(out.text.empty());
    out.scratch = test::patched_private_segment_size(patched);
    const auto entry = test::patched_entry_text_offset(patched);
    ASSERT_TRUE(entry.has_value());
    out.entry = *entry;
  }

  test::DbiSim make_sim(const PatchedKernel &kernel) {
    test::DbiSim sim(a_.sim_arch, a_.wave_size);
    sim.set_kernarg(wrapper_);
    sim.set_entry_offset(kernel.entry);
    return sim;
  }

  // The kernarg pair is destroyed between the prologue and the site, and the
  // probe still receives the pointer.
  void expect_probe_receives_the_payload_pointer() {
    expect_probe_receives_the_payload_pointer(delivery_);
  }

  void expect_probe_receives_the_payload_pointer(const PatchedKernel &kernel) {
    test::DbiSim sim = make_sim(kernel);
    const std::vector<std::vector<uint32_t>> regs =
        sim.run_and_read_vgprs(kernel.text, kernel.scratch, {/*v2=*/2, /*v3=*/3});
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
    std::vector<uint32_t> sabotaged = delivery_.text;
    ASSERT_NO_FATAL_FAILURE(nop_payload_load(sabotaged));

    test::DbiSim sim = make_sim(delivery_);
    const std::vector<std::vector<uint32_t>> regs =
        sim.run_and_read_vgprs(sabotaged, delivery_.scratch, {/*v2=*/2, /*v3=*/3});
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
    expect_guest_kernarg_pointer_is_restored(restore_);
  }

  void expect_guest_kernarg_pointer_is_restored(const PatchedKernel &kernel) {
    test::DbiSim sim = make_sim(kernel);
    const uint64_t seen = read_guest_kernarg_pointer(sim, kernel.text, kernel.scratch);
    EXPECT_EQ(seen, kGuestKernargSentinel)
        << "the guest's kernarg pointer was not restored at entry";
    EXPECT_NE(seen, test::DbiSim::KERNARG_ADDR)
        << "the guest still sees the wrapper pointer the CP delivered";
  }

  // The pointer the guest itself reads out of s[0:1] after the anchor, as the
  // two halves it copied into v2 and v3.
  uint64_t read_guest_kernarg_pointer(test::DbiSim &sim, const std::vector<uint32_t> &text,
                                      uint32_t scratch) {
    const std::vector<std::vector<uint32_t>> regs =
        sim.run_and_read_vgprs(text, scratch, {/*v2=*/2, /*v3=*/3});
    if (regs[0].size() != a_.wave_size)
      return 0;
    return (static_cast<uint64_t>(regs[1][0]) << 32) | static_cast<uint64_t>(regs[0][0]);
  }

  // Negative control for the restore: with the two moves gone, the kernel keeps
  // the wrapper pointer. Confirms the restore is what put the guest's back.
  void expect_without_the_restore_the_wrapper_pointer_remains() {
    std::vector<uint32_t> sabotaged = restore_.text;
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

    test::DbiSim sim = make_sim(restore_);
    EXPECT_EQ(read_guest_kernarg_pointer(sim, sabotaged, restore_.scratch),
              test::DbiSim::KERNARG_ADDR)
        << "without the restore the guest must still see the CP's wrapper pointer";
  }

  // The kernel really comes back to the entry, so the re-entry tests are not
  // passing on one that never did.
  void expect_two_passes_through_the_entry(const PatchedKernel &kernel) {
    test::DbiSim sim = make_sim(kernel);
    const auto s6 = sim.run_and_read_sgpr64(kernel.text, kernel.scratch, 6);
    ASSERT_TRUE(s6.has_value()) << "kernel did not run to completion";
    EXPECT_EQ(static_cast<uint32_t>(*s6), 2u);
  }

  // Negative control for the re-entry tests: sending the backedge to the stub
  // instead of the original entry re-runs the prologue, and the guest must then
  // stop seeing its own kernarg pointer. Without this, the re-entry tests could
  // pass on a harness that cannot observe a second run.
  void expect_reentering_the_stub_reruns_the_prologue() {
    std::vector<uint32_t> sabotaged = loop_restore_.text;
    // The backedge is the fourth word; its branch base is offset 16.
    const auto to_stub = compute_sopp_branch_simm16(12, loop_restore_.entry);
    ASSERT_TRUE(to_stub.has_value());
    sabotaged[3] = build_s_cbranch_scc(/*scc1=*/true, *to_stub, a_.arch);

    // The kernel has to finish and take both passes, or a fault or a hang
    // would pass this control as readily as a second prologue run.
    test::DbiSim sim = make_sim(loop_restore_);
    const std::vector<std::vector<uint32_t>> regs =
        sim.run_and_read_vgprs(sabotaged, loop_restore_.scratch, {/*v2=*/2, /*v3=*/3});
    ASSERT_EQ(regs[0].size(), a_.wave_size) << "kernel did not run to completion";
    test::DbiSim count_sim = make_sim(loop_restore_);
    const auto s6 = count_sim.run_and_read_sgpr64(sabotaged, loop_restore_.scratch, 6);
    ASSERT_TRUE(s6.has_value()) << "kernel did not run to completion";
    ASSERT_EQ(static_cast<uint32_t>(*s6), 2u);

    const uint64_t seen =
        (static_cast<uint64_t>(regs[1][0]) << 32) | static_cast<uint64_t>(regs[0][0]);
    EXPECT_NE(seen, kGuestKernargSentinel)
        << "a second prologue run left the guest's kernarg pointer intact";
  }

  // Negative control: the original entry no longer runs the prologue, so
  // dispatching there skips the stub and the pointer must not arrive.
  void expect_dispatch_at_the_original_entry_skips_the_prologue() {
    PatchedKernel original = delivery_;
    original.entry = 0;
    test::DbiSim sim = make_sim(original);
    const std::vector<std::vector<uint32_t>> regs =
        sim.run_and_read_vgprs(original.text, original.scratch, {/*v2=*/2, /*v3=*/3});
    ASSERT_EQ(regs[0].size(), a_.wave_size) << "kernel did not run to completion";
    const uint64_t seen =
        (static_cast<uint64_t>(regs[1][0]) << 32) | static_cast<uint64_t>(regs[0][0]);
    EXPECT_NE(seen, kLogBufferSentinel) << "the pointer arrived without the stub running";
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
  PatchedKernel delivery_;
  PatchedKernel restore_;
  PatchedKernel call_delivery_;
  PatchedKernel call_restore_;
  PatchedKernel entry_point_delivery_;
  PatchedKernel entry_point_restore_;
  PatchedKernel loop_delivery_;
  PatchedKernel loop_restore_;
  PatchedKernel return_delivery_;
  PatchedKernel return_restore_;
  PatchedKernel jump_delivery_;
  PatchedKernel jump_restore_;
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
TEST_F(DbiCdna3EntryPrologueSim, ProbeReceivesThePayloadPointerAcrossACallReturn) {
  expect_probe_receives_the_payload_pointer(call_delivery_);
}
TEST_F(DbiCdna3EntryPrologueSim, GuestKernargPointerIsRestoredAcrossACallReturn) {
  expect_guest_kernarg_pointer_is_restored(call_restore_);
}
TEST_F(DbiCdna3EntryPrologueSim, ProbeReceivesThePayloadPointerAtTheEntry) {
  expect_probe_receives_the_payload_pointer(entry_point_delivery_);
}
TEST_F(DbiCdna3EntryPrologueSim, GuestKernargPointerIsRestoredAtTheEntry) {
  expect_guest_kernarg_pointer_is_restored(entry_point_restore_);
}
TEST_F(DbiCdna3EntryPrologueSim, TheLoopPassesThroughTheEntryTwice) {
  expect_two_passes_through_the_entry(loop_restore_);
}
TEST_F(DbiCdna3EntryPrologueSim, TheRewrittenReturnPassesThroughTheEntryTwice) {
  expect_two_passes_through_the_entry(return_restore_);
}
TEST_F(DbiCdna3EntryPrologueSim, TheIndirectJumpPassesThroughTheEntryTwice) {
  expect_two_passes_through_the_entry(jump_restore_);
}
TEST_F(DbiCdna3EntryPrologueSim, ProbeReceivesThePayloadPointerAfterAnIndirectJumpToTheEntry) {
  expect_probe_receives_the_payload_pointer(jump_delivery_);
}
TEST_F(DbiCdna3EntryPrologueSim, GuestKernargPointerIsRestoredAfterAnIndirectJumpToTheEntry) {
  expect_guest_kernarg_pointer_is_restored(jump_restore_);
}
TEST_F(DbiCdna3EntryPrologueSim, ProbeReceivesThePayloadPointerAfterALoopToTheEntry) {
  expect_probe_receives_the_payload_pointer(loop_delivery_);
}
TEST_F(DbiCdna3EntryPrologueSim, GuestKernargPointerIsRestoredAfterALoopToTheEntry) {
  expect_guest_kernarg_pointer_is_restored(loop_restore_);
}
TEST_F(DbiCdna3EntryPrologueSim, ProbeReceivesThePayloadPointerAfterAReturnToTheEntry) {
  expect_probe_receives_the_payload_pointer(return_delivery_);
}
TEST_F(DbiCdna3EntryPrologueSim, GuestKernargPointerIsRestoredAfterAReturnToTheEntry) {
  expect_guest_kernarg_pointer_is_restored(return_restore_);
}
TEST_F(DbiCdna3EntryPrologueSim, ReenteringTheStubRerunsThePrologue) {
  expect_reentering_the_stub_reruns_the_prologue();
}
TEST_F(DbiCdna3EntryPrologueSim, DispatchAtTheOriginalEntrySkipsThePrologue) {
  expect_dispatch_at_the_original_entry_skips_the_prologue();
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
TEST_F(DbiCdna4EntryPrologueSim, ProbeReceivesThePayloadPointerAcrossACallReturn) {
  expect_probe_receives_the_payload_pointer(call_delivery_);
}
TEST_F(DbiCdna4EntryPrologueSim, GuestKernargPointerIsRestoredAcrossACallReturn) {
  expect_guest_kernarg_pointer_is_restored(call_restore_);
}
TEST_F(DbiCdna4EntryPrologueSim, ProbeReceivesThePayloadPointerAtTheEntry) {
  expect_probe_receives_the_payload_pointer(entry_point_delivery_);
}
TEST_F(DbiCdna4EntryPrologueSim, GuestKernargPointerIsRestoredAtTheEntry) {
  expect_guest_kernarg_pointer_is_restored(entry_point_restore_);
}
TEST_F(DbiCdna4EntryPrologueSim, TheLoopPassesThroughTheEntryTwice) {
  expect_two_passes_through_the_entry(loop_restore_);
}
TEST_F(DbiCdna4EntryPrologueSim, TheRewrittenReturnPassesThroughTheEntryTwice) {
  expect_two_passes_through_the_entry(return_restore_);
}
TEST_F(DbiCdna4EntryPrologueSim, TheIndirectJumpPassesThroughTheEntryTwice) {
  expect_two_passes_through_the_entry(jump_restore_);
}
TEST_F(DbiCdna4EntryPrologueSim, ProbeReceivesThePayloadPointerAfterAnIndirectJumpToTheEntry) {
  expect_probe_receives_the_payload_pointer(jump_delivery_);
}
TEST_F(DbiCdna4EntryPrologueSim, GuestKernargPointerIsRestoredAfterAnIndirectJumpToTheEntry) {
  expect_guest_kernarg_pointer_is_restored(jump_restore_);
}
TEST_F(DbiCdna4EntryPrologueSim, ProbeReceivesThePayloadPointerAfterALoopToTheEntry) {
  expect_probe_receives_the_payload_pointer(loop_delivery_);
}
TEST_F(DbiCdna4EntryPrologueSim, GuestKernargPointerIsRestoredAfterALoopToTheEntry) {
  expect_guest_kernarg_pointer_is_restored(loop_restore_);
}
TEST_F(DbiCdna4EntryPrologueSim, ProbeReceivesThePayloadPointerAfterAReturnToTheEntry) {
  expect_probe_receives_the_payload_pointer(return_delivery_);
}
TEST_F(DbiCdna4EntryPrologueSim, GuestKernargPointerIsRestoredAfterAReturnToTheEntry) {
  expect_guest_kernarg_pointer_is_restored(return_restore_);
}
TEST_F(DbiCdna4EntryPrologueSim, ReenteringTheStubRerunsThePrologue) {
  expect_reentering_the_stub_reruns_the_prologue();
}
TEST_F(DbiCdna4EntryPrologueSim, DispatchAtTheOriginalEntrySkipsThePrologue) {
  expect_dispatch_at_the_original_entry_skips_the_prologue();
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
TEST_F(DbiRdna4EntryPrologueSim, ProbeReceivesThePayloadPointerAcrossACallReturn) {
  expect_probe_receives_the_payload_pointer(call_delivery_);
}
TEST_F(DbiRdna4EntryPrologueSim, GuestKernargPointerIsRestoredAcrossACallReturn) {
  expect_guest_kernarg_pointer_is_restored(call_restore_);
}
TEST_F(DbiRdna4EntryPrologueSim, ProbeReceivesThePayloadPointerAtTheEntry) {
  expect_probe_receives_the_payload_pointer(entry_point_delivery_);
}
TEST_F(DbiRdna4EntryPrologueSim, GuestKernargPointerIsRestoredAtTheEntry) {
  expect_guest_kernarg_pointer_is_restored(entry_point_restore_);
}
TEST_F(DbiRdna4EntryPrologueSim, TheLoopPassesThroughTheEntryTwice) {
  expect_two_passes_through_the_entry(loop_restore_);
}
TEST_F(DbiRdna4EntryPrologueSim, TheRewrittenReturnPassesThroughTheEntryTwice) {
  expect_two_passes_through_the_entry(return_restore_);
}
TEST_F(DbiRdna4EntryPrologueSim, TheIndirectJumpPassesThroughTheEntryTwice) {
  expect_two_passes_through_the_entry(jump_restore_);
}
TEST_F(DbiRdna4EntryPrologueSim, ProbeReceivesThePayloadPointerAfterAnIndirectJumpToTheEntry) {
  expect_probe_receives_the_payload_pointer(jump_delivery_);
}
TEST_F(DbiRdna4EntryPrologueSim, GuestKernargPointerIsRestoredAfterAnIndirectJumpToTheEntry) {
  expect_guest_kernarg_pointer_is_restored(jump_restore_);
}
TEST_F(DbiRdna4EntryPrologueSim, ProbeReceivesThePayloadPointerAfterALoopToTheEntry) {
  expect_probe_receives_the_payload_pointer(loop_delivery_);
}
TEST_F(DbiRdna4EntryPrologueSim, GuestKernargPointerIsRestoredAfterALoopToTheEntry) {
  expect_guest_kernarg_pointer_is_restored(loop_restore_);
}
TEST_F(DbiRdna4EntryPrologueSim, ProbeReceivesThePayloadPointerAfterAReturnToTheEntry) {
  expect_probe_receives_the_payload_pointer(return_delivery_);
}
TEST_F(DbiRdna4EntryPrologueSim, GuestKernargPointerIsRestoredAfterAReturnToTheEntry) {
  expect_guest_kernarg_pointer_is_restored(return_restore_);
}
TEST_F(DbiRdna4EntryPrologueSim, ReenteringTheStubRerunsThePrologue) {
  expect_reentering_the_stub_reruns_the_prologue();
}
TEST_F(DbiRdna4EntryPrologueSim, DispatchAtTheOriginalEntrySkipsThePrologue) {
  expect_dispatch_at_the_original_entry_skips_the_prologue();
}

} // namespace
} // namespace rocjitsu
