// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/lds_race.h"
#include "rocjitsu/code/patch/consan/prepare.h"

#include "rocjitsu/code/basic_block.h"
#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/code/builders/smem_builders.h"
#include "rocjitsu/code/builders/vector_builders.h"
#include "rocjitsu/code/patch/kernel_text_layout.h"
#include "rocjitsu/isa/instruction.h"

#include <algorithm>
#include <bitset>
#include <cassert>
#include <cstring>
#include <set>

namespace rocjitsu::consan {
namespace {
Result reject_instruction(InputError &error, InputErrorCode code, std::string message,
                          std::optional<uint64_t> offset = std::nullopt) {
  error = {.code = code, .text_offset = offset, .message = std::move(message)};
  return Result::failure();
}

bool admitted_instruction(std::string_view name) {
  constexpr std::array names = {
      "s_load_dword",       "s_load_dwordx2",     "s_mov_b64",         "s_and_saveexec_b64",
      "s_and_b64",          "s_or_b64",           "s_delay_alu",       "v_cmpx_eq_u32_e32",
      "v_add_u32_e32",      "global_store_dword", "ds_read_b32",       "ds_write_b32",
      "s_wait_kmcnt",       "s_wait_storecnt",    "s_wait_dscnt",      "s_wait_loadcnt_dscnt",
      "s_wait_loadcnt",     "s_wait_xcnt",        "v_nop_e32",         "s_load_b32",
      "s_load_b64",         "s_waitcnt",          "s_waitcnt_vscnt",   "s_mov_b32",
      "s_and_saveexec_b32", "s_and_b32",          "s_or_b32",          "s_cmp_eq_u32",
      "s_branch",           "s_cbranch_scc0",     "s_cbranch_scc1",    "s_cbranch_vccz",
      "s_cbranch_vccnz",    "s_cbranch_execz",    "s_cbranch_execnz",  "s_nop",
      "s_endpgm",           "v_mov_b32_e32",      "v_lshrrev_b32_e32", "v_lshlrev_b32_e32",
      "v_add_nc_u32_e32",   "v_and_b32_e32",      "v_cmp_eq_u32_e32",  "v_cmp_gt_u32_e32",
      "global_store_b32",   "ds_store_b32",       "ds_load_b32",
  };
  return std::find(names.begin(), names.end(), name) != names.end();
}

// These are scalar operand selectors, not integer values: selector 128 means
// inline zero, 255 introduces a literal dword, and 253 reads SCC. All five
// supported generated ISA tables share these encodings; RDNA3 supplies the
// names here, not a target-specific lowering choice. Keep using the generated
// definitions and check the shared assumption when those tables change.
constexpr uint8_t kZero = rdna3::OPR_SSRC_POS_INT_MIN;
constexpr uint8_t kOne = kZero + 1;
constexpr uint8_t kLiteral = rdna3::OPR_SSRC_SRC_LITERAL;
constexpr uint8_t kScc = rdna3::OPR_SSRC_SRC_SCC;
static_assert(cdna3::OPR_SSRC_POS_INT_MIN == kZero && cdna4::OPR_SSRC_POS_INT_MIN == kZero &&
              rdna4::OPR_SSRC_POS_INT_MIN == kZero && cdna5::OPR_SSRC_POS_INT_MIN == kZero);
static_assert(cdna3::OPR_SSRC_SRC_LITERAL == kLiteral && cdna4::OPR_SSRC_SRC_LITERAL == kLiteral &&
              rdna4::OPR_SSRC_SRC_LITERAL == kLiteral && cdna5::OPR_SSRC_SRC_LITERAL == kLiteral);
static_assert(cdna3::OPR_SSRC_SRC_SCC == kScc && cdna4::OPR_SSRC_SRC_SCC == kScc &&
              rdna4::OPR_SSRC_SRC_SCC == kScc && cdna5::OPR_SSRC_SRC_SCC == kScc);
enum class ScalarOp { Add, AddCarry, Multiply, ShiftRight, Select, AndNot };

Result reject(InputError &error, std::string message) {
  error = {.code = InputErrorCode::ResourceLimit, .text_offset = {}, .message = std::move(message)};
  return Result::failure();
}

/// @brief Emit per-wave LDS observations for the host's offline race analysis.
/// @details The coordinator has already placed the shared context bootstrap.
/// Entry converts its context pointer into this wave's report-row pointer,
/// publishes serial/entry/wave identity, and initializes sequence/loss counters.
/// Access probes precede the unchanged guest operation; exit publishes final
/// counters and row status. These observations feed diagnosis, not a GPU verdict.
///
/// The qualified plan owns 22 SGPRs and two VGPRs beyond both declared guest
/// registers and initialized ABI inputs. The row pointer and sequence/loss
/// counters persist across probes; the remaining slots hold saved EXEC/VCC/SCC,
/// address-group masks and publication scratch. save()/restore() bracket each
/// probe, including lane-zero publication under a temporary EXEC. SCC is restored
/// from its saved boolean with a scalar comparison; guest operands/results never
/// occupy this scratch, including when an LDS load aliases its address register.
///
/// Each access consumes the saved active-lane mask: readfirstlane selects one
/// address, comparison forms its exact group mask, and that group is removed
/// before the next iteration. The instruction's common immediate is added with
/// carry recorded. Empty EXEC publishes no access. Up to four groups have fixed
/// per-site reservations; any remaining lanes increment lost occurrences once.
/// Neither the publication EXEC nor unused reservations invent guest accesses.
///
/// Payload stores drain before each commit/status store, and that store drains
/// before continuing, using the selected target's store-completion wait. Exit
/// similarly publishes final counters before status. These waits protect report
/// publication; they are not evidence of guest synchronization. Qualification
/// and resource planning must succeed before constructing this emitter.
class ProbeEmitter {
public:
  ProbeEmitter(std::vector<uint8_t> &text, uint8_t scalar_base, uint8_t vector_base,
               rj_code_arch_t arch, uint32_t wave_size)
      : text_(text), scalar_(scalar_base), vector_(vector_base), arch_(arch), wave_size_(wave_size),
        exec_(scalar_operand_exec_lo(arch)), vcc_(scalar_operand_vcc_lo(arch)) {}

  template <size_t Count> void emit(const std::array<uint32_t, Count> &words) {
    append_words(text_, words);
  }
  void word(uint32_t value) { emit(std::array<uint32_t, 1>{value}); }
  void wait_rdna4_scalar(uint16_t control) {
    if (arch_ != ROCJITSU_CODE_ARCH_RDNA4)
      return;
    word(build_sopp_encoding(arch_, rdna4::kSWaitAluSopp, control));
    if ((control & 1) == 0)
      pending_scalar_writes_.reset();
  }
  void scalar_read(uint8_t source, uint8_t width = 1) {
    if (arch_ != ROCJITSU_CODE_ARCH_RDNA4 || source >= 128)
      return;
    for (uint16_t lane = source; lane < source + width; ++lane)
      if (pending_scalar_writes_[lane]) {
        wait_rdna4_scalar(0xfffe); // sa_sdst(0), including SGPR-pair reuse.
        return;
      }
  }
  void scalar_write(uint8_t destination, uint8_t width = 1) {
    if (arch_ == ROCJITSU_CODE_ARCH_RDNA4 && destination < 128)
      for (uint16_t lane = destination; lane < destination + width; ++lane)
        pending_scalar_writes_.set(lane);
  }
  void move(uint8_t destination, uint8_t source) {
    scalar_read(source);
    word(build_s_mov_b32(destination, source, arch_));
    scalar_write(destination);
  }
  void pair(uint8_t destination, uint8_t source) {
    scalar_read(source, 2);
    word(build_s_mov_b64(destination, source, arch_));
    scalar_write(destination, 2);
  }
  void immediate(uint8_t destination, uint32_t value) {
    move(destination, kLiteral);
    word(value);
  }
  void mask(uint8_t destination, uint8_t source) {
    pair(destination, source);
    if (wave_size_ == 32 && destination != exec_)
      move(destination + 1, kZero);
  }
  void publish_lane() { pair(exec_, kOne); }
  void binary(ScalarOp operation, uint8_t destination, uint8_t left, uint8_t right) {
    const uint8_t width = operation == ScalarOp::AndNot && wave_size_ == 64 ? 2 : 1;
    scalar_read(left, width);
    scalar_read(right, width);
    scalar_write(destination, width);
    switch (operation) {
    case ScalarOp::Add:
      word(build_s_add_u32(destination, left, right, arch_));
      return;
    case ScalarOp::AddCarry:
      word(build_s_addc_u32(destination, left, right, arch_));
      return;
    case ScalarOp::ShiftRight:
      word(build_s_lshr_b32(destination, left, right, arch_));
      return;
    case ScalarOp::Select:
      word(build_s_cselect_b32(destination, left, right, arch_));
      return;
    default:
      break;
    }
    uint16_t opcode = 0;
    switch (arch_) {
    case ROCJITSU_CODE_ARCH_CDNA3:
      opcode = operation == ScalarOp::Multiply
                   ? cdna3::kSMulI32Sop2
                   : (wave_size_ == 64 ? cdna3::kSAndn2B64Sop2 : cdna3::kSAndn2B32Sop2);
      break;
    case ROCJITSU_CODE_ARCH_CDNA4:
      opcode = operation == ScalarOp::Multiply
                   ? cdna4::kSMulI32Sop2
                   : (wave_size_ == 64 ? cdna4::kSAndn2B64Sop2 : cdna4::kSAndn2B32Sop2);
      break;
    case ROCJITSU_CODE_ARCH_RDNA3:
      opcode = operation == ScalarOp::Multiply
                   ? rdna3::kSMulI32Sop2
                   : (wave_size_ == 64 ? rdna3::kSAndNot1B64Sop2 : rdna3::kSAndNot1B32Sop2);
      break;
    case ROCJITSU_CODE_ARCH_RDNA4:
      opcode = operation == ScalarOp::Multiply
                   ? rdna4::kSMulI32Sop2
                   : (wave_size_ == 64 ? rdna4::kSAndNot1B64Sop2 : rdna4::kSAndNot1B32Sop2);
      break;
    case ROCJITSU_CODE_ARCH_CDNA5:
      opcode = operation == ScalarOp::Multiply
                   ? cdna5::kSMulI32Sop2
                   : (wave_size_ == 64 ? cdna5::kSAndNot1B64Sop2 : cdna5::kSAndNot1B32Sop2);
      break;
    default:
      valid_ = false;
      return;
    }
    word(build_sop2_encoding(arch_, opcode, destination, left, right));
  }
  void binary_immediate(ScalarOp operation, uint8_t destination, uint8_t source, uint32_t value) {
    binary(operation, destination, source, kLiteral);
    word(value);
  }
  void wait_stores() {
    switch (arch_) {
    case ROCJITSU_CODE_ARCH_CDNA3:
      word(build_sopp_encoding(arch_, cdna3::kSWaitcntSopp, 0));
      break;
    case ROCJITSU_CODE_ARCH_CDNA4:
      word(build_sopp_encoding(arch_, cdna4::kSWaitcntSopp, 0));
      break;
    case ROCJITSU_CODE_ARCH_RDNA3:
      emit(rdna3::build_sopk(rdna3::kSWaitcntVscntSopk,
                             {.simm16 = 0, .sdst = rdna3::OPR_SSRC_NULL}));
      break;
    case ROCJITSU_CODE_ARCH_RDNA4:
      word(build_sopp_encoding(arch_, rdna4::kSWaitStorecntSopp, 0));
      break;
    case ROCJITSU_CODE_ARCH_CDNA5:
      word(build_sopp_encoding(arch_, cdna5::kSWaitStorecntSopp, 0));
      break;
    default:
      valid_ = false;
      return;
    }
  }
  void save() {
    // Probe boundaries may follow guest VALU activity or a different emitter.
    // Drain scalar/VCC hazards before saving architectural state.
    wait_rdna4_scalar(0xf1fc);
    pair(scalar_ + 4, exec_);
    move(scalar_ + 6, kScc);
    pair(scalar_ + 8, vcc_);
  }
  void restore() {
    // Every forward early-exit path joins here. A wait on the join itself
    // covers paths that skipped waits in the address-group publication body.
    wait_rdna4_scalar(0xf1fc);
    pair(vcc_, scalar_ + 8);
    pair(exec_, scalar_ + 4);
    scalar_read(scalar_ + 6);
    word(build_s_cmp_lg_u32(scalar_ + 6, kZero, arch_));
    wait_rdna4_scalar(0xf1fc);
  }
  void read_first(uint8_t destination, uint8_t source) {
    switch (arch_) {
    case ROCJITSU_CODE_ARCH_CDNA3:
      emit(cdna3::build_vop1(cdna3::kVReadfirstlaneB32Vop1,
                             {.src0 = uint16_t(256 + source), .vdst = destination}));
      break;
    case ROCJITSU_CODE_ARCH_CDNA4:
      emit(cdna4::build_vop1(cdna4::kVReadfirstlaneB32Vop1,
                             {.src0 = uint16_t(256 + source), .vdst = destination}));
      break;
    case ROCJITSU_CODE_ARCH_RDNA3:
      emit(rdna3::build_vop1(rdna3::kVReadfirstlaneB32Vop1,
                             {.src0 = uint16_t(256 + source), .vdst = destination}));
      break;
    case ROCJITSU_CODE_ARCH_RDNA4:
      emit(rdna4::build_vop1(rdna4::kVReadfirstlaneB32Vop1,
                             {.src0 = uint16_t(256 + source), .vdst = destination}));
      break;
    case ROCJITSU_CODE_ARCH_CDNA5:
      emit(cdna5::build_vop1(cdna5::kVReadfirstlaneB32Vop1,
                             {.src0 = uint16_t(256 + source), .vdst = destination}));
      break;
    default:
      valid_ = false;
      return;
    }
    // GFX9 VALU-to-SGPR consumers require two wait states. RDNA4 instead
    // drains va_sdst before a comparison can consume a reused SGPR pair.
    if (arch_ == ROCJITSU_CODE_ARCH_CDNA3 || arch_ == ROCJITSU_CODE_ARCH_CDNA4)
      word(build_s_nop(1, arch_));
    wait_rdna4_scalar(0xf1ff);
  }
  void compare_address(uint8_t scalar, uint8_t vector) {
    scalar_read(scalar);
    switch (arch_) {
    case ROCJITSU_CODE_ARCH_CDNA3:
      emit(cdna3::build_vopc(cdna3::kVCmpEqU32Vopc, {.src0 = scalar, .vsrc1 = vector}));
      break;
    case ROCJITSU_CODE_ARCH_CDNA4:
      emit(cdna4::build_vopc(cdna4::kVCmpEqU32Vopc, {.src0 = scalar, .vsrc1 = vector}));
      break;
    case ROCJITSU_CODE_ARCH_RDNA3:
      emit(rdna3::build_vopc(rdna3::kVCmpEqU32Vopc, {.src0 = scalar, .vsrc1 = vector}));
      break;
    case ROCJITSU_CODE_ARCH_RDNA4:
      emit(rdna4::build_vopc(rdna4::kVCmpEqU32Vopc, {.src0 = scalar, .vsrc1 = vector}));
      break;
    case ROCJITSU_CODE_ARCH_CDNA5:
      emit(cdna5::build_vopc(cdna5::kVCmpEqU32Vopc, {.src0 = scalar, .vsrc1 = vector}));
      break;
    default:
      valid_ = false;
      return;
    }
    wait_rdna4_scalar(0xfffd); // va_vcc(0) before copying the comparison mask.
  }
  void store(uint32_t offset, uint8_t source) {
    scalar_read(source);
    scalar_read(scalar_, 2);
    emit(build_v_mov_b32_imm(vector_, offset, arch_));
    word(build_v_mov_b32_src(vector_ + 1, source, arch_));
    switch (arch_) {
    case ROCJITSU_CODE_ARCH_CDNA3:
      emit(cdna3::build_flat(
          cdna3::kFlatStoreDwordFlat,
          {.seg = 2, .addr = vector_, .data = uint8_t(vector_ + 1), .saddr = scalar_}));
      break;
    case ROCJITSU_CODE_ARCH_CDNA4:
      emit(cdna4::build_flat(
          cdna4::kFlatStoreDwordFlat,
          {.seg = 2, .addr = vector_, .data = uint8_t(vector_ + 1), .saddr = scalar_}));
      break;
    case ROCJITSU_CODE_ARCH_RDNA3:
      emit(rdna3::build_flat(
          rdna3::kFlatStoreB32Flat,
          {.seg = 2, .addr = vector_, .data = uint8_t(vector_ + 1), .saddr = scalar_}));
      break;
    case ROCJITSU_CODE_ARCH_RDNA4:
      emit(
          rdna4::build_vglobal(rdna4::kGlobalStoreB32Vglobal,
                               {.saddr = scalar_, .vsrc = uint8_t(vector_ + 1), .vaddr = vector_}));
      break;
    case ROCJITSU_CODE_ARCH_CDNA5:
      emit(
          cdna5::build_vglobal(cdna5::kGlobalStoreB32Vglobal,
                               {.saddr = scalar_, .vsrc = uint8_t(vector_ + 1), .vaddr = vector_}));
      break;
    default:
      valid_ = false;
      return;
    }
  }
  void store_immediate(uint32_t offset, uint32_t value) {
    immediate(scalar_ + 2, value);
    store(offset, scalar_ + 2);
  }
  size_t branch_execz() {
    // Both successors must inherit a drained state; emission-time bookkeeping
    // alone cannot describe which forward branch the wave will take.
    wait_rdna4_scalar(0xfffe);
    const size_t offset = text_.size();
    const uint16_t opcode = arch_ == ROCJITSU_CODE_ARCH_CDNA3   ? cdna3::kSCbranchExeczSopp
                            : arch_ == ROCJITSU_CODE_ARCH_CDNA4 ? cdna4::kSCbranchExeczSopp
                            : arch_ == ROCJITSU_CODE_ARCH_RDNA3 ? rdna3::kSCbranchExeczSopp
                            : arch_ == ROCJITSU_CODE_ARCH_RDNA4 ? rdna4::kSCbranchExeczSopp
                                                                : cdna5::kSCbranchExeczSopp;
    word(build_sopp_encoding(arch_, opcode, 0));
    return offset;
  }
  bool bind(size_t branch_offset, size_t destination) {
    const std::optional<int16_t> delta = compute_sopp_branch_simm16(branch_offset, destination);
    if (!delta)
      return false;
    uint32_t original = 0;
    std::memcpy(&original, text_.data() + branch_offset, 4);
    original = (original & 0xffff0000u) | static_cast<uint16_t>(*delta);
    std::memcpy(text_.data() + branch_offset, &original, 4);
    return true;
  }
  void entry(uint32_t row_stride) {
    save();
    emit(build_s_load_dwordx2(scalar_ + 2, scalar_, wire::kContextSerial, arch_));
    emit(build_s_load_dwordx2(scalar_ + 18, scalar_, wire::kContextReportPointer, arch_));
    word(build_wait_scalar_loads_complete(arch_));
    pair(scalar_, scalar_ + 18);
    read_first(scalar_ + 20, 0);
    binary(ScalarOp::ShiftRight, scalar_ + 20, scalar_ + 20, kZero + (wave_size_ == 64 ? 6 : 5));
    binary_immediate(ScalarOp::Multiply, scalar_ + 19, scalar_ + 20, row_stride);
    binary(ScalarOp::Add, scalar_ + 19, scalar_ + 19, kZero + kReportHeaderBytes);
    binary(ScalarOp::Add, scalar_, scalar_, scalar_ + 19);
    binary(ScalarOp::AddCarry, scalar_ + 1, scalar_ + 1, kZero);
    move(scalar_ + 15, kZero);
    move(scalar_ + 16, kZero);
    publish_lane();
    store(wire::kRowSerial, scalar_ + 2);
    store(wire::kRowSerial + 4, scalar_ + 3);
    store_immediate(wire::kRowEntry, 0);
    store(wire::kRowWave, scalar_ + 20);
    wait_stores();
    store_immediate(wire::kRowStatus, wire::kRowEntered);
    wait_stores();
    restore();
  }
  bool access(const NativeLdsAccess &access) {
    save();
    mask(scalar_ + 10, exec_);
    const size_t empty = branch_execz();
    binary(ScalarOp::Add, scalar_ + 15, scalar_ + 15, kOne);
    std::vector<size_t> done;
    for (uint32_t address_group = 0; address_group < kAddressGroupsPerSite; ++address_group) {
      mask(exec_, scalar_ + 10);
      done.push_back(branch_execz());
      read_first(scalar_ + 14, static_cast<uint8_t>(access.address_vgpr));
      compare_address(scalar_ + 14, uint8_t(access.address_vgpr));
      mask(scalar_ + 12, vcc_);
      binary(ScalarOp::AndNot, scalar_ + 10, scalar_ + 10, scalar_ + 12);
      binary_immediate(ScalarOp::Add, scalar_ + 14, scalar_ + 14, access.immediate_bytes);
      binary(ScalarOp::Select, scalar_ + 17, kZero + 3, kOne);
      publish_lane();
      const uint32_t record =
          kRowHeaderBytes +
          (access.site * kAddressGroupsPerSite + address_group) * kAccessRecordBytes;
      store_immediate(record + wire::kRecordAnalysis, uint32_t(AnalysisKind::LdsRaceDetection));
      store_immediate(record + wire::kRecordKind, uint32_t(RecordKind::Access));
      store_immediate(record + wire::kRecordSite, access.site);
      store_immediate(record + wire::kRecordRange, 0);
      store(record + wire::kRecordSequence, scalar_ + 15);
      store(record + wire::kRecordLdsOffset, scalar_ + 14);
      store_immediate(record + wire::kRecordWidth, 4);
      store(record + wire::kRecordGuestMask, scalar_ + 12);
      store(record + wire::kRecordGuestMask + 4, scalar_ + 13);
      store(record + wire::kRecordFlags, scalar_ + 17);
      wait_stores();
      store_immediate(record + wire::kRecordCommit, wire::kAccessCommitted);
      wait_stores();
    }
    mask(exec_, scalar_ + 10);
    move(scalar_ + 2, kZero);
    const size_t no_loss = branch_execz();
    move(scalar_ + 2, kOne);
    if (!bind(no_loss, text_.size()))
      return false;
    binary(ScalarOp::Add, scalar_ + 16, scalar_ + 16, scalar_ + 2);
    const size_t end = text_.size();
    done.push_back(empty);
    for (size_t offset : done)
      if (!bind(offset, end))
        return false;
    restore();
    return true;
  }
  bool valid() const { return valid_; }

  void exit() {
    save();
    publish_lane();
    store(wire::kRowFinalSequence, scalar_ + 15);
    store(wire::kRowLoss, scalar_ + 16);
    wait_stores();
    scalar_read(scalar_ + 16);
    word(build_s_cmp_lg_u32(scalar_ + 16, kZero, arch_));
    binary(ScalarOp::Select, scalar_ + 2,
           kZero + wire::kRowEntered + wire::kRowExited + wire::kRowExhausted,
           kZero + wire::kRowEntered + wire::kRowExited);
    store(wire::kRowStatus, scalar_ + 2);
    wait_stores();
    restore();
  }

private:
  std::vector<uint8_t> &text_;
  uint8_t scalar_, vector_;
  rj_code_arch_t arch_;
  uint32_t wave_size_;
  uint8_t exec_, vcc_;
  bool valid_ = true;
  // Treat every scalar destination as potentially tracked by an earlier VALU
  // read. The probe ABI is small; this conservative set avoids a second ISA
  // decoder/scheduler while coalescing writes until an actual scalar consumer.
  std::bitset<128> pending_scalar_writes_;
};

ProbeEmitter make_probe_emitter(std::vector<uint8_t> &text, const LdsRaceProbePlan &plan) {
  // These internal preconditions precede both narrowing and constructor operand
  // helpers. Expected input/resource failures are handled by qualification/planning.
  assert(
      std::ranges::any_of(kNativeLdsTargets,
                          [&](const TargetProfile &target) { return target.arch == plan.arch; }) &&
      "native LDS probes require a supported architecture");
  assert(std::ranges::any_of(kNativeLdsTargets,
                             [&](const TargetProfile &target) {
                               return target.arch == plan.arch &&
                                      supports_wave(target, plan.wave_size);
                             }) &&
         "native LDS probes require a supported wave mode");
  assert(plan.sgprs >= 22 && plan.sgprs <= 102 && plan.scalar_base <= plan.sgprs - 22 &&
         plan.scalar_base % 2 == 0 && "native LDS probes require bounded SGPR scratch");
  assert(plan.vgprs >= 2 && plan.vgprs <= 256 && plan.vector_base <= plan.vgprs - 2 &&
         "native LDS probes require bounded VGPR scratch");
  assert(plan.row_stride >= kRowHeaderBytes + kAddressGroupsPerSite * kAccessRecordBytes &&
         plan.row_stride <= ContractLimits{}.report_bytes - kReportHeaderBytes &&
         "native LDS probes require a bounded report row");
  return ProbeEmitter(text, static_cast<uint8_t>(plan.scalar_base),
                      static_cast<uint8_t>(plan.vector_base), plan.arch, plan.wave_size);
}

} // namespace

Result qualify_lds_race_entry(const InputProgram &program, InputError &error) {
  const uint32_t group_bytes = program.kernel.descriptor.group_segment_fixed_size;
  if (group_bytes < 4 || group_bytes > 65536)
    return reject_instruction(error, InputErrorCode::UnsupportedEntry,
                              "native LDS b32 requires an LDS extent in [4, 65536] bytes");
  return Result::success();
}

Result qualify_lds_race_block(const BasicBlock &block, InputError &error) {
  // Static per-site reservations and occurrence ordering require acyclic flow.
  for (const BasicBlock *successor : block.successors())
    if (successor->start_offset() <= block.start_offset())
      return reject_instruction(error, InputErrorCode::UnsupportedFlow,
                                "MVP does not admit CFG backedges", block.start_offset());
  return Result::success();
}

Result classify_lds_race_instruction(const Instruction &instruction, rj_code_arch_t arch,
                                     uint32_t allocated_vgprs,
                                     std::vector<NativeLdsAccess> &accesses, InputError &error) {
  const std::string_view name = instruction.mnemonic();
  const uint64_t offset = instruction.src_loc();
  if ((instruction.flags() & BARRIER) != 0 || name.find("atomic") != std::string_view::npos ||
      name.starts_with("s_barrier") || name.starts_with("s_sendmsg"))
    return reject_instruction(error, InputErrorCode::UnsupportedSynchronization,
                              "unmodeled synchronization: " + std::string(name), offset);
  if (name.starts_with("ds_") && name != "ds_load_b32" && name != "ds_store_b32" &&
      name != "ds_read_b32" && name != "ds_write_b32")
    return reject_instruction(error, InputErrorCode::UnsupportedAccess,
                              "unsupported LDS form: " + std::string(name), offset);
  if (!admitted_instruction(name))
    return reject_instruction(error, InputErrorCode::UnsupportedInstruction,
                              "instruction outside MVP: " + std::string(name), offset);
  if (name == "ds_load_b32" || name == "ds_store_b32" || name == "ds_read_b32" ||
      name == "ds_write_b32") {
    // CDNA3/4 place GDS at bit 16; RDNA and CDNA5 use bit 17.
    const uint32_t gds_bit =
        (arch == ROCJITSU_CODE_ARCH_CDNA3 || arch == ROCJITSU_CODE_ARCH_CDNA4) ? 16 : 17;
    if (instruction.size() != 8 || (instruction.raw_encoding()[0] & (1u << gds_bit)) != 0)
      return reject_instruction(error, InputErrorCode::UnsupportedAccess, "non-native LDS form",
                                offset);
    const bool read = name == "ds_load_b32" || name == "ds_read_b32";
    const Operand *address = instruction.src_operand(0);
    const Operand *value = read ? instruction.dst_operand(0) : instruction.src_operand(1);
    const std::optional<RegisterRef> address_reg = address->decoded_vgpr();
    const std::optional<RegisterRef> value_reg = value->decoded_vgpr();
    if (!address_reg || !value_reg || address_reg->width != 1 || value_reg->width != 1 ||
        address_reg->index >= allocated_vgprs || value_reg->index >= allocated_vgprs)
      return reject_instruction(error, InputErrorCode::UnsupportedAccess,
                                "LDS operands exceed allocation", offset);
    accesses.push_back(
        {.site = static_cast<uint32_t>(accesses.size()),
         .original_offset = offset,
         .kind = read ? AccessKind::Read : AccessKind::Write,
         .address_vgpr = address_reg->index,
         .immediate_bytes = static_cast<uint16_t>(instruction.raw_encoding()[0] & 0xffff)});
  }
  return Result::success();
}

FailureOr<LdsRaceProbePlan> plan_lds_race_probes(const InputProgram &program, InputError &error) {
  assert(program.target != nullptr && "qualified input requires a target");
  assert(
      std::ranges::any_of(kNativeLdsTargets,
                          [&](const TargetProfile &target) { return &target == program.target; }) &&
      "qualified input requires a canonical target");
  assert(supports_wave(*program.target, program.wave_size) &&
         "qualified input requires a supported wave mode");
  const rj_code_arch_t arch = program.target->arch;
  const uint32_t granule = descriptor_vgpr_granularity_for_wavefront(arch, program.wave_size);
  const uint32_t scalar_base = program.entry_plan.storage.persistent_base;
  const uint32_t vector_base = program.original_vgprs;
  assert(scalar_base >= program.original_sgprs && "LDS probe scratch must not overlap guest SGPRs");
  // Twenty-two SGPRs own report/state/grouping counters; two VGPRs publish
  // uniform payloads. Preserve the original declared registers and ABI inputs.
  const uint32_t sgprs = std::max(program.original_sgprs, scalar_base + 22);
  const uint32_t vgprs = (vector_base + 2 + granule - 1) / granule * granule;
  const uint64_t stride = kRowHeaderBytes + uint64_t{program.accesses.size()} *
                                                kAddressGroupsPerSite * kAccessRecordBytes;
  if (sgprs > 102 || vgprs > 256 || stride > ContractLimits{}.report_bytes - kReportHeaderBytes)
    return reject(error, "no bounded, disjoint probe register/report plan");
  return LdsRaceProbePlan{.arch = arch,
                          .wave_size = program.wave_size,
                          .scalar_base = scalar_base,
                          .vector_base = vector_base,
                          .sgprs = sgprs,
                          .vgprs = vgprs,
                          .row_stride = static_cast<uint32_t>(stride)};
}

Result emit_lds_race_entry(std::vector<uint8_t> &text, const LdsRaceProbePlan &plan,
                           InputError &error) {
  ProbeEmitter emitter = make_probe_emitter(text, plan);
  emitter.entry(plan.row_stride);
  if (!emitter.valid())
    return reject(error, "selected architecture has no native LDS probe lowering");
  return Result::success();
}

FailureOr<SiteFacts> emit_lds_race_access(std::vector<uint8_t> &text, const LdsRaceProbePlan &plan,
                                          const NativeLdsAccess &access, InputError &error) {
  const uint64_t probe = text.size();
  ProbeEmitter emitter = make_probe_emitter(text, plan);
  if (!emitter.access(access))
    return reject(error, "probe branch exceeds encoding range");
  if (!emitter.valid())
    return reject(error, "selected architecture has no native LDS probe lowering");
  return SiteFacts{.id = access.site,
                   .analysis = AnalysisKind::LdsRaceDetection,
                   .access = access.kind,
                   .original_offset = access.original_offset,
                   .original_size = 8,
                   .prepared_guest_offset = text.size(),
                   .prepared_guest_size = 8,
                   .probe_offset = probe,
                   .probe_size = static_cast<uint32_t>(text.size() - probe),
                   .first_slot = access.site * kAddressGroupsPerSite};
}

Result emit_lds_race_exit(std::vector<uint8_t> &text, const LdsRaceProbePlan &plan,
                          InputError &error) {
  ProbeEmitter emitter = make_probe_emitter(text, plan);
  emitter.exit();
  if (!emitter.valid())
    return reject(error, "selected architecture has no native LDS probe lowering");
  return Result::success();
}

Result validate_lds_race_accesses(const InputProgram &program, const PreparedArtifact &artifact,
                                  std::span<BasicBlock *const> blocks,
                                  std::span<const uint8_t> prepared_text, InputError &error) {
  std::set<uint64_t> access_offsets;
  for (BasicBlock *block : blocks)
    for (const Instruction &instruction : block->instructions()) {
      const std::string_view name = instruction.mnemonic();
      if (name == "ds_load_b32" || name == "ds_store_b32" || name == "ds_read_b32" ||
          name == "ds_write_b32")
        access_offsets.insert(instruction.src_loc());
    }
  if (access_offsets.size() != artifact.manifest.sites.size())
    return reject(error, "final reachable guest access multiplicity differs from original");
  for (const SiteFacts &site : artifact.manifest.sites)
    if (!access_offsets.contains(site.prepared_guest_offset) ||
        std::memcmp(program.image.data() + program.text_file_offset + site.original_offset,
                    prepared_text.data() + site.prepared_guest_offset, 8) != 0)
      return reject(error, "final guest LDS operation/map differs from original bytes");
  return Result::success();
}

} // namespace rocjitsu::consan
