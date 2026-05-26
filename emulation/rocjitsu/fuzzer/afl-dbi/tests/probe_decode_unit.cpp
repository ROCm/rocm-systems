#include "rocjitsu_fuzzer/afl_dbi_plan.h"

#include "rocjitsu/analysis/liveness.h"
#include "rocjitsu/code/basic_block.h"
#include "rocjitsu/code/code_object.h"
#include "instrumentation_planner.h"
#include "rocjitsu/code/patch/code_object_patcher.h"
#include "rocjitsu/code/patch/instruction_builder.h"
#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/isa/operand.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <exception>
#include <initializer_list>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace {

void check(bool condition, const char *message) {
  if (!condition) {
    fprintf(stderr, "probe_decode_unit: %s\n", message);
    std::exit(1);
  }
}

class TestOperand final : public rocjitsu::Operand {
public:
  TestOperand() = default;
  explicit TestOperand(rocjitsu::RegisterRef ref)
      : rocjitsu::Operand(ref.width * 32, ref.index), ref_(ref) {}

  std::optional<rocjitsu::RegisterRef> to_register_ref() const override { return ref_; }

private:
  std::optional<rocjitsu::RegisterRef> ref_;
};

class TestInstruction final : public rocjitsu::Instruction {
public:
  TestInstruction(std::string_view mnemonic,
                  std::initializer_list<rocjitsu::RegisterRef> uses,
                  uint64_t flags = 0)
      : rocjitsu::Instruction(mnemonic, nullptr) {
    size_ = sizeof(uint32_t);
    flags_ = flags;
    for (rocjitsu::RegisterRef ref : uses) {
      src_storage_[num_src_] = TestOperand(ref);
      src_operands_[num_src_] = &src_storage_[num_src_];
      ++num_src_;
    }
  }

  TestInstruction(std::string_view mnemonic, std::span<const uint32_t> raw_words)
      : rocjitsu::Instruction(mnemonic, nullptr) {
    const size_t word_count = std::min(raw_words.size(), raw_storage_.size());
    for (size_t i = 0; i < word_count; ++i)
      raw_storage_[i] = raw_words[i];
    size_ = static_cast<int>(word_count * sizeof(uint32_t));
    raw_encoding_ = raw_storage_.data();
  }

private:
  std::array<TestOperand, 4> src_storage_{};
  std::array<uint32_t, 3> raw_storage_{};
};

class TestTextSection final : public rocjitsu::Section {
public:
  TestTextSection(std::unique_ptr<char[]> data, size_t size)
      : rocjitsu::Section(".text", std::move(data)), size_(size) {}

  size_t size() const override { return size_; }
  uint32_t sectionHeaderNameIdx() const override { return 0; }
  uint64_t sectionOffset() const override { return 0; }

private:
  size_t size_ = 0;
};

class TestCodeObject final : public rocjitsu::CodeObject {
public:
  explicit TestCodeObject(std::span<const uint32_t> words) {
    const size_t byte_size = words.size() * sizeof(uint32_t);
    image_.resize(byte_size);
    memcpy(image_.data(), words.data(), byte_size);

    auto data = std::make_unique<char[]>(byte_size);
    memcpy(data.get(), words.data(), byte_size);
    sections_.push_back(std::make_unique<TestTextSection>(std::move(data), byte_size));
    text_sections_.push_back(sections_.back().get());
  }
};

enum class TestOpcode : uint32_t {
  UseVgpr0To3 = 1,
  End = 2,
  WaitKmcnt = 3,
  SaveExec = 4,
  Relocatable = 5,
  Opaque = 6,
  DelayAlu = 7,
  ModeledVopd = 8,
  UnmodeledVopd = 9,
  UseSgpr0To1 = 10,
};

class TestDecoder final : public rocjitsu::Decoder {
public:
  rocjitsu::Instruction *decode(const rj_code_binary_inst_t *inst) override {
    switch (static_cast<TestOpcode>(*inst)) {
    case TestOpcode::UseVgpr0To3:
      return new TestInstruction("test_use_v0_v3",
                                 {{rocjitsu::RegClass::VGPR, 0, 1},
                                  {rocjitsu::RegClass::VGPR, 1, 1},
                                  {rocjitsu::RegClass::VGPR, 2, 1},
                                  {rocjitsu::RegClass::VGPR, 3, 1}});
    case TestOpcode::UseSgpr0To1:
      return new TestInstruction("test_use_s0_s1",
                                 {{rocjitsu::RegClass::SGPR, 0, 1},
                                  {rocjitsu::RegClass::SGPR, 1, 1}});
    case TestOpcode::End:
      return new TestInstruction("test_end", {}, rocjitsu::PROGRAM_TERMINATOR);
    case TestOpcode::WaitKmcnt:
      return new TestInstruction("s_wait_kmcnt", {}, rocjitsu::WAITCNT);
    case TestOpcode::SaveExec:
      return new TestInstruction("s_and_saveexec_b64", {});
    case TestOpcode::Relocatable:
      return new TestInstruction("v_add_u32", {});
    case TestOpcode::Opaque:
      return new TestInstruction("unknown_opaque", {});
    case TestOpcode::DelayAlu:
      return new TestInstruction("s_delay_alu", {});
    case TestOpcode::ModeledVopd: {
      const std::array<uint32_t, 2> words = {3391357056u, 36569223u};
      return new TestInstruction("vopd_opaque", words);
    }
    case TestOpcode::UnmodeledVopd: {
      constexpr uint32_t word0 =
          (0x32u << 26) | (0xeu << 22) | (0x8u << 17) | (7u << 9) | 260u;
      constexpr uint32_t word1 =
          (10u << 24) | (6u << 17) | (8u << 9) | 261u;
      const std::array<uint32_t, 2> words = {word0, word1};
      return new TestInstruction("vopd_opaque", words);
    }
    }
    return new TestInstruction("test_end", {}, rocjitsu::PROGRAM_TERMINATOR);
  }
};

const char *decode_arch_name(rj_code_arch_t arch) {
  switch (arch) {
  case ROCJITSU_CODE_ARCH_CDNA2:
    return "CDNA2";
  case ROCJITSU_CODE_ARCH_CDNA3:
    return "CDNA3";
  case ROCJITSU_CODE_ARCH_CDNA4:
    return "CDNA4";
  case ROCJITSU_CODE_ARCH_RDNA1:
    return "RDNA1";
  case ROCJITSU_CODE_ARCH_RDNA2:
    return "RDNA2";
  case ROCJITSU_CODE_ARCH_RDNA3:
    return "RDNA3";
  case ROCJITSU_CODE_ARCH_RDNA3_5:
    return "RDNA3.5";
  case ROCJITSU_CODE_ARCH_RDNA4:
    return "RDNA4";
  default:
    return "unexpected";
  }
}

void check_probe_decodes(rj_code_arch_t arch, const std::vector<uint32_t> &words,
                         const char *probe_name) {
  std::unique_ptr<rocjitsu::Decoder> decoder = rocjitsu::Decoder::create(arch);
  check(decoder != nullptr, "decoder should be available");

  size_t word = 0;
  while (word < words.size()) {
    std::unique_ptr<rocjitsu::Instruction> inst;
    try {
      inst.reset(decoder->decode(&words[word]));
    } catch (const std::exception &e) {
      fprintf(stderr,
              "probe_decode_unit: %s threw while decoding %s word %zu "
              "(0x%08x): %s\n",
              decode_arch_name(arch), probe_name, word, words[word], e.what());
      std::exit(1);
    }
    if (inst == nullptr) {
      fprintf(stderr, "probe_decode_unit: %s failed to decode %s word %zu\n",
              decode_arch_name(arch), probe_name, word);
      std::exit(1);
    }
    check(inst->size() >= sizeof(uint32_t), "decoded instruction has invalid size");
    check(inst->size() % sizeof(uint32_t) == 0, "decoded instruction size should be word-aligned");
    word += inst->size() / sizeof(uint32_t);
  }
  check(word == words.size(), "probe decoder walked past probe end");
}

std::vector<uint8_t> bytes(std::span<const uint32_t> words) {
  std::vector<uint8_t> out(words.size() * sizeof(uint32_t));
  memcpy(out.data(), words.data(), out.size());
  return out;
}

void check_decodes_one(rj_code_arch_t arch, const uint32_t *words, size_t word_count,
                       std::string_view expected_mnemonic, int expected_size,
                       uint32_t expected_literal) {
  std::unique_ptr<rocjitsu::Decoder> decoder = rocjitsu::Decoder::create(arch);
  check(decoder != nullptr, "decoder should be available");

  std::unique_ptr<rocjitsu::Instruction> inst;
  try {
    inst.reset(decoder->decode(words));
  } catch (const std::exception &e) {
    fprintf(stderr, "probe_decode_unit: %s threw while decoding 0x%08x: %s\n",
            decode_arch_name(arch), words[0], e.what());
    std::exit(1);
  }

  check(inst != nullptr, "single instruction should decode");
  check(inst->mnemonic() == expected_mnemonic, "decoded instruction mnemonic mismatch");
  check(inst->size() == expected_size, "decoded instruction size mismatch");
  check(word_count >= static_cast<size_t>(expected_size / sizeof(uint32_t)),
        "expected size should fit provided words");
  check(inst->raw_encoding() != nullptr, "decoded instruction should expose raw encoding");
  check(inst->raw_encoding()[expected_size / sizeof(uint32_t) - 1] == expected_literal,
        "decoded instruction should retain inline literal in raw encoding");
}

void check_block_entry_direct_branch_trampoline(rj_code_arch_t arch) {
  using namespace rocjitsu::fuzzer::afl_dbi;

  const std::array<uint32_t, 5> text_words = {
      rocjitsu::build_s_branch(/*offset_dwords=*/2, arch),
      rocjitsu::build_s_nop(0, arch),
      rocjitsu::build_s_nop(0, arch),
      rocjitsu::build_s_nop(0, arch),
      rocjitsu::build_s_nop(0, arch),
  };
  const std::vector<uint8_t> text = bytes(text_words);
  constexpr uint64_t kOriginalTarget = 3 * sizeof(uint32_t);

  EdgeSite site;
  site.kind = EdgePatchKind::BlockEntry;
  site.kernel_name = "branch_entry_kernel";
  site.patch_text_offset = 0;
  site.return_text_offset = sizeof(uint32_t);
  site.first_inst_size = sizeof(uint32_t);
  site.bb_id = 0x345678u;
  site.slot_policy = EdgeSlotPolicyKind::FixedCounter;
  site.fixed_slot = kFirstEdgeCounterSlot;
  site.self_contained_probe = true;

  LocalTextCaveAllocator local_caves(text);
  const char *failure = nullptr;
  std::optional<PlannedEdgeTrampoline> planned =
      plan_edge_trampoline(site, text, /*appended_cave_body_size=*/0,
                           /*cave_start=*/0x100, local_caves, arch,
                           /*state_pointer=*/0x1234567887654321ull, &failure);
  check(planned.has_value(), "direct-branch block entry trampoline should plan");
  check(failure == nullptr, "successful direct-branch trampoline should not report failure");
  check(planned->trampoline.cave_words.size() > 1,
        "direct-branch block entry trampoline should contain probe and branch");

  const uint64_t relocated_branch_pc =
      planned->result.cave_text_offset +
      (planned->trampoline.cave_words.size() - 1) * sizeof(uint32_t);
  const std::optional<int16_t> expected_target_offset =
      s_branch_offset_dwords(relocated_branch_pc, kOriginalTarget);
  check(expected_target_offset.has_value(),
        "direct-branch trampoline fixture target should be encodable");
  check(planned->trampoline.cave_words.back() ==
            rocjitsu::build_s_branch(*expected_target_offset, arch),
        "direct-branch block entry trampoline should branch to original target");

  const std::optional<int16_t> fallthrough_offset =
      s_branch_offset_dwords(relocated_branch_pc, site.return_text_offset);
  check(fallthrough_offset.has_value(), "fallthrough return should be encodable in fixture");
  check(planned->trampoline.cave_words.back() !=
            rocjitsu::build_s_branch(*fallthrough_offset, arch),
        "direct-branch block entry trampoline should not return to fallthrough");
}

void check_conditional_block_entry_trampoline(rj_code_arch_t arch) {
  using namespace rocjitsu::fuzzer::afl_dbi;

  constexpr uint8_t kSCbranchExeczOp = 0x25;
  const std::array<uint32_t, 4> text_words = {
      rocjitsu::pack_sopp(kSCbranchExeczOp, /*offset_dwords=*/2),
      rocjitsu::build_s_nop(0, arch),
      rocjitsu::build_s_nop(0, arch),
      rocjitsu::build_s_nop(0, arch),
  };
  const std::vector<uint8_t> text = bytes(text_words);
  constexpr uint64_t kFallthroughTarget = sizeof(uint32_t);
  constexpr uint64_t kTakenTarget = 3 * sizeof(uint32_t);
  constexpr uint32_t kFixedSlot = kFirstEdgeCounterSlot + 7;

  EdgeSite site;
  site.kind = EdgePatchKind::ConditionalBlockEntry;
  site.kernel_name = "conditional_branch_entry_kernel";
  site.patch_text_offset = 0;
  site.return_text_offset = kTakenTarget;
  site.first_inst_size = sizeof(uint32_t);
  site.bb_id = 0x456789u;
  site.fallthrough_bb_id = site.bb_id;
  site.slot_policy = EdgeSlotPolicyKind::FixedCounter;
  site.fixed_slot = kFixedSlot;
  site.fallthrough_slot = kFixedSlot;
  site.branch_opcode = kSCbranchExeczOp;

  LocalTextCaveAllocator local_caves(text);
  const char *failure = nullptr;
  std::optional<PlannedEdgeTrampoline> planned =
      plan_edge_trampoline(site, text, /*appended_cave_body_size=*/0,
                           /*cave_start=*/0x100, local_caves, arch,
                           /*state_pointer=*/0x1234567887654321ull, &failure);
  check(planned.has_value(), "conditional block-entry trampoline should plan");
  check(failure == nullptr, "successful conditional block-entry trampoline should not fail");

  std::optional<std::vector<uint32_t>> expected_probe =
      rdna4_counter_probe(kFixedSlot, arch);
  check(expected_probe.has_value(), "conditional block-entry fixed probe should build");
  const size_t probe_words = expected_probe->size();
  check(planned->trampoline.cave_words.size() == 1 + probe_words + 1 + probe_words + 1,
        "conditional block-entry trampoline shape changed");

  const uint64_t cond_branch_pc = planned->result.cave_text_offset;
  const uint64_t fallthrough_probe_pc = cond_branch_pc + sizeof(uint32_t);
  const uint64_t fallthrough_branch_pc =
      fallthrough_probe_pc + probe_words * sizeof(uint32_t);
  const uint64_t taken_probe_pc = fallthrough_branch_pc + sizeof(uint32_t);
  const uint64_t taken_branch_pc = taken_probe_pc + probe_words * sizeof(uint32_t);

  const std::optional<int16_t> cond_offset =
      s_branch_offset_dwords(cond_branch_pc, taken_probe_pc);
  check(cond_offset.has_value(), "conditional block-entry dispatcher should be encodable");
  check(planned->trampoline.cave_words[0] ==
            rocjitsu::pack_sopp(kSCbranchExeczOp, static_cast<uint16_t>(*cond_offset)),
        "conditional block-entry trampoline should dispatch before probes");
  check(std::equal(expected_probe->begin(), expected_probe->end(),
                   planned->trampoline.cave_words.begin() + 1),
        "conditional block-entry fallthrough probe should record the block");
  check(std::equal(expected_probe->begin(), expected_probe->end(),
                   planned->trampoline.cave_words.begin() + 1 + probe_words + 1),
        "conditional block-entry taken probe should record the same block");

  const std::optional<int16_t> fallthrough_ret =
      s_branch_offset_dwords(fallthrough_branch_pc, kFallthroughTarget);
  const std::optional<int16_t> taken_ret =
      s_branch_offset_dwords(taken_branch_pc, kTakenTarget);
  check(fallthrough_ret.has_value() && taken_ret.has_value(),
        "conditional block-entry return branches should be encodable");
  check(planned->trampoline.cave_words[1 + probe_words] ==
            rocjitsu::build_s_branch(*fallthrough_ret, arch),
        "conditional block-entry fallthrough path should return to fallthrough");
  check(planned->trampoline.cave_words.back() ==
            rocjitsu::build_s_branch(*taken_ret, arch),
        "conditional block-entry taken path should return to original target");
}

void check_conditional_previous_bb_branch_trampoline(rj_code_arch_t arch) {
  using namespace rocjitsu::fuzzer::afl_dbi;

  // This captures the required trampoline shape if EXEC-conditioned previous-BB
  // branches become safe to enable in the planner.
  constexpr uint8_t kSCbranchExeczOp = 0x25;
  const std::array<uint32_t, 4> text_words = {
      rocjitsu::pack_sopp(kSCbranchExeczOp, /*offset_dwords=*/2),
      rocjitsu::build_s_nop(0, arch),
      rocjitsu::build_s_nop(0, arch),
      rocjitsu::build_s_nop(0, arch),
  };
  const std::vector<uint8_t> text = bytes(text_words);
  constexpr uint64_t kFallthroughTarget = sizeof(uint32_t);
  constexpr uint64_t kTakenTarget = 3 * sizeof(uint32_t);
  constexpr uint64_t kStatePointer = 0x1234567887654321ull;

  EdgeSite site;
  site.kind = EdgePatchKind::ConditionalBranchTerminator;
  site.kernel_name = "conditional_previous_bb_branch_kernel";
  site.patch_text_offset = 0;
  site.return_text_offset = kTakenTarget;
  site.first_inst_size = sizeof(uint32_t);
  site.bb_id = 0x456789u;
  site.fallthrough_bb_id = 0x654321u;
  site.slot_policy = EdgeSlotPolicyKind::PreviousBbHash;
  site.branch_opcode = kSCbranchExeczOp;
  site.self_contained_probe = true;

  LocalTextCaveAllocator local_caves(text);
  const char *failure = nullptr;
  std::optional<PlannedEdgeTrampoline> planned =
      plan_edge_trampoline(site, text, /*appended_cave_body_size=*/0,
                           /*cave_start=*/0x100, local_caves, arch,
                           kStatePointer, &failure);
  check(planned.has_value(),
        "conditional previous-BB branch trampoline should plan");
  check(failure == nullptr,
        "successful conditional previous-BB branch trampoline should not fail");

  std::vector<uint32_t> expected_fallthrough_probe =
      rdna4_previous_bb_edge_probe_with_state_pointer(site.fallthrough_bb_id,
                                                      kStatePointer, arch,
                                                      site.probe_registers);
  std::vector<uint32_t> expected_taken_probe =
      rdna4_previous_bb_edge_probe_with_state_pointer(site.bb_id, kStatePointer,
                                                      arch, site.probe_registers);
  const size_t fallthrough_probe_words = expected_fallthrough_probe.size();
  const size_t taken_probe_words = expected_taken_probe.size();
  check(planned->trampoline.cave_words.size() ==
            1 + fallthrough_probe_words + 1 + taken_probe_words + 1,
        "conditional previous-BB branch trampoline shape changed");

  const uint64_t cond_branch_pc = planned->result.cave_text_offset;
  const uint64_t fallthrough_probe_pc = cond_branch_pc + sizeof(uint32_t);
  const uint64_t fallthrough_branch_pc =
      fallthrough_probe_pc + fallthrough_probe_words * sizeof(uint32_t);
  const uint64_t taken_probe_pc = fallthrough_branch_pc + sizeof(uint32_t);
  const uint64_t taken_branch_pc =
      taken_probe_pc + taken_probe_words * sizeof(uint32_t);

  const std::optional<int16_t> cond_offset =
      s_branch_offset_dwords(cond_branch_pc, taken_probe_pc);
  check(cond_offset.has_value(),
        "conditional previous-BB branch dispatcher should be encodable");
  check(planned->trampoline.cave_words[0] ==
            rocjitsu::pack_sopp(kSCbranchExeczOp,
                                static_cast<uint16_t>(*cond_offset)),
        "conditional previous-BB branch should dispatch on the original EXEC predicate before probing");
  check(std::equal(expected_fallthrough_probe.begin(),
                   expected_fallthrough_probe.end(),
                   planned->trampoline.cave_words.begin() + 1),
        "conditional previous-BB branch fallthrough probe should record the fallthrough BB");
  check(std::equal(expected_taken_probe.begin(), expected_taken_probe.end(),
                   planned->trampoline.cave_words.begin() + 1 +
                       fallthrough_probe_words + 1),
        "conditional previous-BB branch taken probe should record the taken BB");

  const std::optional<int16_t> fallthrough_ret =
      s_branch_offset_dwords(fallthrough_branch_pc, kFallthroughTarget);
  const std::optional<int16_t> taken_ret =
      s_branch_offset_dwords(taken_branch_pc, kTakenTarget);
  check(fallthrough_ret.has_value() && taken_ret.has_value(),
        "conditional previous-BB branch return branches should be encodable");
  check(planned->trampoline.cave_words[1 + fallthrough_probe_words] ==
            rocjitsu::build_s_branch(*fallthrough_ret, arch),
        "conditional previous-BB branch fallthrough path should return to fallthrough");
  check(planned->trampoline.cave_words.back() ==
            rocjitsu::build_s_branch(*taken_ret, arch),
        "conditional previous-BB branch taken path should return to original target");
}

void check_exec_conditioned_fixed_branch_trampoline(rj_code_arch_t arch,
                                                    uint8_t branch_opcode) {
  using namespace rocjitsu::fuzzer::afl_dbi;

  const std::array<uint32_t, 4> text_words = {
      rocjitsu::pack_sopp(branch_opcode, /*offset_dwords=*/2),
      rocjitsu::build_s_nop(0, arch),
      rocjitsu::build_s_nop(0, arch),
      rocjitsu::build_s_nop(0, arch),
  };
  const std::vector<uint8_t> text = bytes(text_words);
  constexpr uint64_t kFallthroughTarget = sizeof(uint32_t);
  constexpr uint64_t kTakenTarget = 3 * sizeof(uint32_t);
  constexpr uint64_t kStatePointer = 0x1234567887654321ull;
  constexpr uint32_t kFallthroughSlot = kFirstEdgeCounterSlot + 11;
  constexpr uint32_t kTakenSlot = kFirstEdgeCounterSlot + 12;

  EdgeSite site;
  site.kind = EdgePatchKind::ConditionalBranchTerminator;
  site.kernel_name = "exec_conditioned_fixed_branch_kernel";
  site.patch_text_offset = 0;
  site.return_text_offset = kTakenTarget;
  site.first_inst_size = sizeof(uint32_t);
  site.bb_id = 0x456789u;
  site.fallthrough_bb_id = 0x654321u;
  site.slot_policy = EdgeSlotPolicyKind::FixedCounter;
  site.fixed_slot = kTakenSlot;
  site.fallthrough_slot = kFallthroughSlot;
  site.branch_opcode = branch_opcode;
  site.self_contained_probe = true;
  site.force_lane0_exec_for_fixed_counter = true;

  LocalTextCaveAllocator local_caves(text);
  const char *failure = nullptr;
  std::optional<PlannedEdgeTrampoline> planned =
      plan_edge_trampoline(site, text, /*appended_cave_body_size=*/0,
                           /*cave_start=*/0x100, local_caves, arch, kStatePointer,
                           &failure);
  check(planned.has_value(), "EXEC-conditioned fixed branch trampoline should plan");
  check(failure == nullptr,
        "successful EXEC-conditioned fixed branch trampoline should not fail");

  const bool taken_is_empty_exec = branch_opcode == 0x25;
  const bool fallthrough_is_empty_exec = branch_opcode == 0x26;
  std::optional<std::vector<uint32_t>> expected_fallthrough_probe =
      fallthrough_is_empty_exec
          ? rdna4_flagless_counter_probe_force_lane0_with_state_pointer(
                kFallthroughSlot, kStatePointer, arch, site.probe_registers)
          : rdna4_flagless_counter_probe_with_state_pointer(
                kFallthroughSlot, kStatePointer, arch, site.probe_registers);
  std::optional<std::vector<uint32_t>> expected_taken_probe =
      taken_is_empty_exec
          ? rdna4_flagless_counter_probe_force_lane0_with_state_pointer(
                kTakenSlot, kStatePointer, arch, site.probe_registers)
          : rdna4_flagless_counter_probe_with_state_pointer(
                kTakenSlot, kStatePointer, arch, site.probe_registers);
  check(expected_fallthrough_probe.has_value() && expected_taken_probe.has_value(),
        "EXEC-conditioned fixed probes should build");
  const size_t fallthrough_probe_words = expected_fallthrough_probe->size();
  const size_t taken_probe_words = expected_taken_probe->size();
  check(planned->trampoline.cave_words.size() ==
            1 + fallthrough_probe_words + 1 + taken_probe_words + 1,
        "EXEC-conditioned fixed branch trampoline shape changed");

  const uint64_t cond_branch_pc = planned->result.cave_text_offset;
  const uint64_t fallthrough_probe_pc = cond_branch_pc + sizeof(uint32_t);
  const uint64_t fallthrough_branch_pc =
      fallthrough_probe_pc + fallthrough_probe_words * sizeof(uint32_t);
  const uint64_t taken_probe_pc = fallthrough_branch_pc + sizeof(uint32_t);
  const uint64_t taken_branch_pc =
      taken_probe_pc + taken_probe_words * sizeof(uint32_t);

  const std::optional<int16_t> cond_offset =
      s_branch_offset_dwords(cond_branch_pc, taken_probe_pc);
  check(cond_offset.has_value(),
        "EXEC-conditioned fixed branch dispatcher should be encodable");
  check(planned->trampoline.cave_words[0] ==
            rocjitsu::pack_sopp(branch_opcode, static_cast<uint16_t>(*cond_offset)),
        "EXEC-conditioned fixed branch should dispatch on the original EXEC predicate");
  check(std::equal(expected_fallthrough_probe->begin(),
                   expected_fallthrough_probe->end(),
                   planned->trampoline.cave_words.begin() + 1),
        "EXEC-conditioned fixed branch fallthrough probe should record the fallthrough edge");
  check(std::equal(expected_taken_probe->begin(), expected_taken_probe->end(),
                   planned->trampoline.cave_words.begin() + 1 +
                       fallthrough_probe_words + 1),
        "EXEC-conditioned fixed branch taken probe should record the taken edge");

  const uint32_t force_exec_lo =
      build_s_mov_b32_word(static_cast<uint8_t>(kScalarExecLo),
                           amdgpu_positive_inline_const(1), arch);
  const auto &forced_probe =
      taken_is_empty_exec ? *expected_taken_probe : *expected_fallthrough_probe;
  check(std::find(forced_probe.begin(), forced_probe.end(), force_exec_lo) !=
            forced_probe.end(),
        "EXEC-empty fixed fallback should force lane 0 before the counter update");

  const std::optional<int16_t> fallthrough_ret =
      s_branch_offset_dwords(fallthrough_branch_pc, kFallthroughTarget);
  const std::optional<int16_t> taken_ret =
      s_branch_offset_dwords(taken_branch_pc, kTakenTarget);
  check(fallthrough_ret.has_value() && taken_ret.has_value(),
        "EXEC-conditioned fixed branch return branches should be encodable");
  check(planned->trampoline.cave_words[1 + fallthrough_probe_words] ==
            rocjitsu::build_s_branch(*fallthrough_ret, arch),
        "EXEC-conditioned fixed branch fallthrough path should return to fallthrough");
  check(planned->trampoline.cave_words.back() ==
            rocjitsu::build_s_branch(*taken_ret, arch),
        "EXEC-conditioned fixed branch taken path should return to original target");
}

void check_scratch_spill_builders(rj_code_arch_t arch, std::span<const uint32_t> expected_words) {
  using namespace rocjitsu::fuzzer::afl_dbi;

  Rdna4ProbeBuilder builder({}, arch);
  check(builder.scratch_store_b32(/*vaddr=*/0, /*vdata=*/1, /*saddr=*/2,
                                  /*byte_offset=*/16),
        "scratch store builder should support target");
  check(builder.scratch_load_b32(/*vdst=*/3, /*vaddr=*/0, /*saddr=*/2,
                                 /*byte_offset=*/16),
        "scratch load builder should support target");
  std::vector<uint32_t> words = builder.take();
  check(words.size() == expected_words.size(), "scratch builder word count mismatch");
  check(std::equal(words.begin(), words.end(), expected_words.begin()),
        "scratch builder encoding mismatch");
  check_probe_decodes(arch, words, "scratch spill load/store");
}

void check_vgpr_scratch_spill_wrapper(rj_code_arch_t arch) {
  using namespace rocjitsu::fuzzer::afl_dbi;

  const std::array<uint8_t, 2> spilled_vgprs = {5, 6};
  std::optional<ProbeScratchSpillPlan> spill_plan =
      plan_vgpr_scratch_spills(arch, /*address_vgpr=*/4, spilled_vgprs,
                               /*original_private_segment_bytes=*/0);
  check(spill_plan.has_value(), "VGPR scratch spill planning should support target");
  check(spill_plan->private_segment_bytes == 8,
        "VGPR scratch spill planning should report private segment bytes");
  check(spill_plan->vgpr_spills.size() == spilled_vgprs.size(),
        "VGPR scratch spill planning should allocate every requested spill");
  check(spill_plan->vgpr_spills[0].byte_offset == 0 &&
            spill_plan->vgpr_spills[1].byte_offset == 4,
        "VGPR scratch spill planning should use stable packed slots");
  const std::array<uint8_t, 3> three_spilled_vgprs = {5, 6, 7};
  std::optional<ProbeScratchSpillPlan> rounded_spill_plan =
      plan_vgpr_scratch_spills(arch, /*address_vgpr=*/4, three_spilled_vgprs,
                               /*original_private_segment_bytes=*/0);
  check(rounded_spill_plan.has_value(),
        "VGPR scratch spill planning should support rounded private strides");
  check(rounded_spill_plan->private_segment_bytes == 16,
        "VGPR scratch spill planning should round non-power-of-two strides");
  const std::array<uint8_t, 2> spilled_sgprs = {8, 9};
  std::optional<ProbeScratchSpillPlan> sgpr_spill_plan =
      plan_probe_scratch_spills(arch, /*address_vgpr=*/4, spilled_vgprs,
                                spilled_sgprs,
                                /*original_private_segment_bytes=*/0);
  check(sgpr_spill_plan.has_value(),
        "scratch spill planning should support SGPR slots");
  check(sgpr_spill_plan->vgpr_spills.size() == spilled_vgprs.size(),
        "combined scratch plan should preserve VGPR spills");
  check(sgpr_spill_plan->sgpr_spills.size() == spilled_sgprs.size(),
        "combined scratch plan should allocate SGPR spill slots");
  check(sgpr_spill_plan->sgpr_spills[0].byte_offset == 8 &&
            sgpr_spill_plan->sgpr_spills[1].byte_offset == 12,
        "SGPR spills should follow VGPR spill slots");
  check(sgpr_spill_plan->private_segment_bytes == 16,
        "SGPR spills should contribute to rounded private bytes");
  check(build_rdna4_v_readfirstlane_b32(/*sdst=*/0, /*vsrc=*/0) == 0x7e000500u,
        "v_readfirstlane_b32 s0, v0 encoding should match llvm-mc");
  check(build_rdna4_v_readfirstlane_b32(/*sdst=*/4, /*vsrc=*/0) == 0x7e080500u,
        "v_readfirstlane_b32 s4, v0 encoding should encode the scalar destination");
  check(amdgpu_negative_inline_const(1) == 0xc1u,
        "AMDGPU inline constant -1 encoding should match llvm-mc");
  check(build_rdna4_v_mbcnt_lo_u32_b32_word0(/*vdst=*/120) == 0xd71f0078u &&
            build_rdna4_v_mbcnt_lo_u32_b32_word1(amdgpu_negative_inline_const(1),
                                                 amdgpu_positive_inline_const(0)) ==
                0x000100c1u &&
            build_rdna4_v_mbcnt_hi_u32_b32_word0(/*vdst=*/120) == 0xd7200078u &&
            build_rdna4_v_mbcnt_hi_u32_b32_word1(amdgpu_negative_inline_const(1),
                                                 amdgpu_vgpr_src(120)) ==
                0x0002f0c1u,
        "previous-BB lane id helper should not read clobbered v0 or ttmp9");

  Rdna4ProbeBuilder lane_builder({}, ROCJITSU_CODE_ARCH_RDNA4);
  lane_builder.v_mov_workitem_id(/*vdst=*/120);
  const std::vector<uint32_t> lane_words = lane_builder.take();
  check(lane_words.size() == 6 && lane_words[0] == 0xd71f0078u &&
            lane_words[1] == 0x000100c1u && lane_words[3] == 0xd7200078u &&
            lane_words[4] == 0x0002f0c1u,
        "previous-BB lane id helper should derive a wave-local lane index from EXEC");

  Rdna4ProbeRegisters regs;
  regs.state_sgpr = 100;
  regs.workitem_vgpr = 5;
  regs.tmp0_vgpr = 6;
  std::optional<std::vector<uint32_t>> probe =
      rdna4_flagless_counter_probe(kFirstEdgeCounterSlot, arch, regs);
  check(probe.has_value(), "base counter probe should be encodable");
  std::optional<std::vector<uint32_t>> wrapped =
      wrap_probe_with_vgpr_scratch_spills(*probe, *spill_plan, regs, arch);
  check(wrapped.has_value(), "VGPR scratch spill wrapper should be encodable");
  check(wrapped->size() > probe->size(), "VGPR scratch spill wrapper should add code");
  const uint32_t mbcnt_lo_word = build_rdna4_v_mbcnt_lo_u32_b32_word0(
      spill_plan->address_vgpr);
  const uint32_t mbcnt_lo_operands =
      build_rdna4_v_mbcnt_lo_u32_b32_word1(regs.saved_exec_sgpr);
  auto mbcnt_lo_it = std::find(wrapped->begin(), wrapped->end(), mbcnt_lo_word);
  check(mbcnt_lo_it != wrapped->end() && mbcnt_lo_it + 1 != wrapped->end() &&
            *(mbcnt_lo_it + 1) == mbcnt_lo_operands,
        "VGPR scratch spill wrapper should compute active-lane rank from saved EXEC");
  std::optional<std::vector<uint32_t>> direct_exec_wrapped =
      wrap_probe_with_vgpr_scratch_spills(*probe, *spill_plan, regs, arch,
                                          /*use_saved_exec_for_address=*/false);
  check(direct_exec_wrapped.has_value(),
        "fixed-counter scratch spill wrapper should support direct EXEC addressing");
  auto direct_mbcnt_lo_it =
      std::find(direct_exec_wrapped->begin(), direct_exec_wrapped->end(), mbcnt_lo_word);
  check(direct_mbcnt_lo_it != direct_exec_wrapped->end() &&
            direct_mbcnt_lo_it + 1 != direct_exec_wrapped->end() &&
            *(direct_mbcnt_lo_it + 1) ==
                build_rdna4_v_mbcnt_lo_u32_b32_word1(kScalarExecLo),
        "direct EXEC scratch wrapper should not reserve saved EXEC for address formation");
  check_probe_decodes(arch, *direct_exec_wrapped,
                      "direct EXEC VGPR scratch spill wrapper");
  std::optional<std::vector<uint32_t>> forced_probe =
      rdna4_flagless_counter_probe_force_lane0(kFirstEdgeCounterSlot, arch, regs);
  check(forced_probe.has_value(), "forced-lane fixed counter probe should be encodable");
  std::optional<std::vector<uint32_t>> forced_wrapped =
      wrap_forced_lane0_probe_with_vgpr_scratch_spills(*forced_probe, *spill_plan,
                                                       regs, arch);
  check(forced_wrapped.has_value(),
        "forced-lane fixed-counter scratch wrapper should be encodable");
  check(forced_wrapped->size() > forced_probe->size(),
        "forced-lane fixed-counter scratch wrapper should add code");
  const uint32_t force_exec_lo =
      build_s_mov_b32_word(static_cast<uint8_t>(kScalarExecLo),
                           amdgpu_positive_inline_const(1), arch);
  check(static_cast<size_t>(std::count(forced_wrapped->begin(), forced_wrapped->end(),
                                       force_exec_lo)) >= 3,
        "forced-lane scratch wrapper should force lane 0 for save, probe, and fill");
  check_probe_decodes(arch, *forced_wrapped,
                      "forced-lane fixed-counter VGPR scratch spill wrapper");
  ProbeScratchSpillPlan wave32_spill_plan = *spill_plan;
  wave32_spill_plan.wave32 = true;
  std::optional<std::vector<uint32_t>> direct_exec_wave32_wrapped =
      wrap_probe_with_vgpr_scratch_spills(*probe, wave32_spill_plan, regs, arch,
                                          /*use_saved_exec_for_address=*/false);
  check(direct_exec_wave32_wrapped.has_value(),
        "wave32 direct EXEC scratch spill wrapper should be encodable");
  check(std::find(direct_exec_wave32_wrapped->begin(),
                  direct_exec_wave32_wrapped->end(),
                  build_rdna4_v_mbcnt_hi_u32_b32_word0(spill_plan->address_vgpr)) ==
            direct_exec_wave32_wrapped->end(),
        "wave32 direct EXEC scratch wrapper should omit high-half mbcnt");
  check_probe_decodes(arch, *direct_exec_wave32_wrapped,
                      "wave32 direct EXEC VGPR scratch spill wrapper");
  std::optional<std::vector<uint32_t>> sgpr_wrapped =
      wrap_probe_with_vgpr_scratch_spills(*probe, *sgpr_spill_plan, regs, arch,
                                          /*use_saved_exec_for_address=*/false);
  check(sgpr_wrapped.has_value(),
        "SGPR scratch spill wrapper should be encodable");
  if (arch == ROCJITSU_CODE_ARCH_RDNA4) {
    check(std::find(sgpr_wrapped->begin(), sgpr_wrapped->end(),
                    build_rdna4_sopp(/*s_wait_loadcnt=*/0x40, /*imm=*/0)) !=
              sgpr_wrapped->end(),
          "RDNA4 SGPR scratch spill wrapper should wait for scratch loads");
    check(std::find(sgpr_wrapped->begin(), sgpr_wrapped->end(),
                    build_rdna4_sopp(/*s_wait_storecnt=*/0x41, /*imm=*/0)) !=
              sgpr_wrapped->end(),
          "RDNA4 SGPR scratch spill wrapper should wait for scratch stores");
  }
  check(std::find(sgpr_wrapped->begin(), sgpr_wrapped->end(),
                  build_rdna4_v_readfirstlane_b32(spilled_sgprs[0],
                                                  spilled_vgprs[0])) !=
            sgpr_wrapped->end(),
        "SGPR scratch spill wrapper should restore scalars through readfirstlane");
  check_probe_decodes(arch, *sgpr_wrapped, "SGPR scratch spill wrapper");
  const uint32_t stride_shift =
      log2_power_of_two_u32(spill_plan->private_segment_bytes);
  const uint32_t shift_word = build_rdna4_v_lshlrev_b32(
      spill_plan->address_vgpr, amdgpu_positive_inline_const(stride_shift),
      spill_plan->address_vgpr);
  check(std::find(wrapped->begin(), wrapped->end(), shift_word) != wrapped->end(),
        "VGPR scratch spill wrapper should scale by private segment stride");
  check_probe_decodes(arch, *wrapped, "VGPR scratch spill wrapper");
  ProbeScratchSpillPlan invalid_stride = *spill_plan;
  invalid_stride.private_segment_bytes = 12;
  check(!wrap_probe_with_vgpr_scratch_spills(*probe, invalid_stride, regs, arch).has_value(),
        "VGPR scratch spill wrapper should reject unencodable private strides");

  EdgeSite site;
  site.kind = EdgePatchKind::BranchTerminator;
  site.kernel_name = "spill_kernel";
  site.patch_text_offset = 0;
  site.return_text_offset = sizeof(uint32_t);
  site.first_inst_size = sizeof(uint32_t);
  site.bb_id = 0x5678u;
  site.slot_policy = EdgeSlotPolicyKind::FixedCounter;
  site.fixed_slot = kFirstEdgeCounterSlot;
  site.self_contained_probe = true;
  site.probe_registers = regs;
  site.scratch_spill_plan = *spill_plan;
  ProbeRegisterRequirements requirements = edge_site_probe_register_requirements(site);
  check(requirements.private_segment_bytes == spill_plan->private_segment_bytes,
        "edge-site requirements should include scratch private bytes");
  check(requirements.vgprs >= 7,
        "edge-site requirements should include probe and scratch address VGPRs");

  std::vector<uint8_t> text_without_caves(64, 0xff);
  LocalTextCaveAllocator no_local_caves(text_without_caves);
  const char *failure = nullptr;
  std::optional<PlannedEdgeTrampoline> planned =
      plan_edge_trampoline(site, text_without_caves,
                           /*appended_cave_body_size=*/0, /*cave_start=*/0x100,
                           no_local_caves, arch,
                           /*state_pointer=*/0x1234567887654321ull, &failure);
  check(planned.has_value(), "spill-backed edge trampoline should plan");
  check(failure == nullptr, "spill-backed edge trampoline should not report failure");
  check_probe_decodes(arch, planned->trampoline.cave_words,
                      "spill-backed edge trampoline");
  auto planned_mbcnt_lo = std::find(planned->trampoline.cave_words.begin(),
                                    planned->trampoline.cave_words.end(),
                                    mbcnt_lo_word);
  check(planned_mbcnt_lo != planned->trampoline.cave_words.end() &&
            planned_mbcnt_lo + 1 != planned->trampoline.cave_words.end() &&
            *(planned_mbcnt_lo + 1) ==
                build_rdna4_v_mbcnt_lo_u32_b32_word1(regs.saved_exec_sgpr),
        "fixed-counter scratch trampoline should honor reserved saved EXEC");

  EdgeSite direct_exec_site = site;
  direct_exec_site.probe_registers.saved_exec_sgpr =
      direct_exec_site.probe_registers.state_sgpr;
  LocalTextCaveAllocator direct_no_local_caves(text_without_caves);
  failure = nullptr;
  std::optional<PlannedEdgeTrampoline> direct_planned =
      plan_edge_trampoline(direct_exec_site, text_without_caves,
                           /*appended_cave_body_size=*/0, /*cave_start=*/0x100,
                           direct_no_local_caves, arch,
                           /*state_pointer=*/0x1234567887654321ull, &failure);
  check(direct_planned.has_value(), "direct-EXEC spill-backed edge trampoline should plan");
  check(failure == nullptr,
        "direct-EXEC spill-backed edge trampoline should not report failure");
  auto direct_mbcnt_lo =
      std::find(direct_planned->trampoline.cave_words.begin(),
                direct_planned->trampoline.cave_words.end(), mbcnt_lo_word);
  check(direct_mbcnt_lo != direct_planned->trampoline.cave_words.end() &&
            direct_mbcnt_lo + 1 != direct_planned->trampoline.cave_words.end() &&
            *(direct_mbcnt_lo + 1) ==
                build_rdna4_v_mbcnt_lo_u32_b32_word1(kScalarExecLo),
        "saved-EXEC-free fixed-counter scratch trampoline should use direct EXEC");

  std::optional<ProbeScratchSpillPlan> overlapping_plan =
      plan_vgpr_scratch_spills(arch, /*address_vgpr=*/5, spilled_vgprs,
                               /*original_private_segment_bytes=*/0);
  check(!overlapping_plan.has_value(),
        "VGPR scratch spill planning should reject address/spill overlap");
}

void check_vgpr_scratch_spill_private_growth_invariant(rj_code_arch_t arch) {
  using namespace rocjitsu::fuzzer::afl_dbi;

  const uint32_t original_private_segment_bytes = 64;
  const std::array<uint8_t, 1> spilled_vgprs = {5};
  std::optional<ProbeScratchSpillPlan> spill_plan =
      plan_vgpr_scratch_spills(arch, /*address_vgpr=*/4, spilled_vgprs,
                               original_private_segment_bytes);
  check(spill_plan.has_value(),
        "VGPR scratch spill planning should support existing private segments");
  check(spill_plan->vgpr_spills[0].byte_offset >= original_private_segment_bytes,
        "DBI spill slots must live after the kernel's original private segment");
  check(spill_plan->private_segment_bytes > original_private_segment_bytes,
        "non-empty DBI spill zones require loader-visible private segment growth");
}

void check_vgpr_scratch_spill_liveness_selection(rj_code_arch_t arch) {
  using namespace rocjitsu::fuzzer::afl_dbi;

  const std::array<uint32_t, 3> words = {
      static_cast<uint32_t>(TestOpcode::UseVgpr0To3),
      static_cast<uint32_t>(TestOpcode::UseSgpr0To1),
      static_cast<uint32_t>(TestOpcode::End),
  };
  TestCodeObject co(words);
  TestDecoder decoder;
  std::vector<std::unique_ptr<rocjitsu::BasicBlock>> blocks =
      rocjitsu::BasicBlock::build(co, decoder);
  check(!blocks.empty(), "scratch spill liveness fixture should build blocks");

  std::vector<rocjitsu::BasicBlock *> scope;
  scope.reserve(blocks.size());
  for (const std::unique_ptr<rocjitsu::BasicBlock> &block : blocks)
    scope.push_back(block.get());
  rocjitsu::LivenessAnalysis liveness{rocjitsu::KernelBlockScope(scope)};

  const rocjitsu::Instruction *probe_point = &*blocks[0]->instructions().begin();
  check(liveness.is_live_before(*probe_point, {rocjitsu::RegClass::VGPR, 0, 1}),
        "scratch spill fixture should make probe VGPRs live");
  check(liveness.is_live_before(*probe_point, {rocjitsu::RegClass::SGPR, 0, 1}),
        "scratch spill fixture should make low SGPRs live");

  KernelSite kernel;
  kernel.name = "spill_selection_kernel";
  kernel.allocated_sgpr_count = 8;
  kernel.allocated_vgpr_count = 5;
  kernel.private_segment_fixed_size = 16;
  kernel.wave32 = true;
  std::array<const rocjitsu::Instruction *, 1> probe_points = {probe_point};

  KernelSite force_fresh_kernel = kernel;
  force_fresh_kernel.allocated_sgpr_count = 16;
  force_fresh_kernel.allocated_vgpr_count = 8;

  const char *force_failure = nullptr;
  std::optional<EdgeProbeRegisterSelection> allocated_selection =
      select_edge_probe_registers_from_liveness(
          force_fresh_kernel, arch, liveness, probe_points,
          /*previous_bb_probe_registers=*/false, /*stable_state_sgpr=*/false,
          Rdna4ProbeRegisters{}, /*allow_fresh_registers=*/true,
          /*allow_vgpr_scratch_spills=*/false, &force_failure);
  check(allocated_selection.has_value(),
        "force-fresh fixture should allow allocated register selection");
  check(allocated_selection->probe_registers.state_sgpr <
            force_fresh_kernel.allocated_sgpr_count &&
            allocated_selection->probe_registers.workitem_vgpr <
                force_fresh_kernel.allocated_vgpr_count,
        "default liveness selection should prefer allocated dead registers");

  force_failure = nullptr;
  std::optional<EdgeProbeRegisterSelection> force_fresh_sgprs_selection =
      select_edge_probe_registers_from_liveness(
          force_fresh_kernel, arch, liveness, probe_points,
          /*previous_bb_probe_registers=*/false, /*stable_state_sgpr=*/false,
          Rdna4ProbeRegisters{}, /*allow_fresh_registers=*/true,
          /*allow_vgpr_scratch_spills=*/false, &force_failure,
          /*allow_direct_exec_fixed_counter_scratch_spills=*/false,
          /*allow_sgpr_scratch_spills=*/false,
          /*force_fresh_sgprs=*/true);
  check(force_fresh_sgprs_selection.has_value(),
        "fresh SGPR diagnostic should keep allocated VGPR selection available");
  check(force_fresh_sgprs_selection->probe_registers.state_sgpr >=
            force_fresh_kernel.allocated_sgpr_count &&
            force_fresh_sgprs_selection->probe_registers.workitem_vgpr <
                force_fresh_kernel.allocated_vgpr_count,
        "fresh SGPR diagnostic should not force fresh VGPRs");

  force_failure = nullptr;
  std::optional<EdgeProbeRegisterSelection> force_fresh_vgprs_selection =
      select_edge_probe_registers_from_liveness(
          force_fresh_kernel, arch, liveness, probe_points,
          /*previous_bb_probe_registers=*/false, /*stable_state_sgpr=*/false,
          Rdna4ProbeRegisters{}, /*allow_fresh_registers=*/true,
          /*allow_vgpr_scratch_spills=*/false, &force_failure,
          /*allow_direct_exec_fixed_counter_scratch_spills=*/false,
          /*allow_sgpr_scratch_spills=*/false,
          /*force_fresh_sgprs=*/false,
          /*force_saved_exec_sgpr_pair=*/false,
          /*force_fresh_vgprs=*/true);
  check(force_fresh_vgprs_selection.has_value(),
        "fresh VGPR diagnostic should keep allocated SGPR selection available");
  check(force_fresh_vgprs_selection->probe_registers.state_sgpr <
            force_fresh_kernel.allocated_sgpr_count &&
            force_fresh_vgprs_selection->probe_registers.workitem_vgpr >=
                force_fresh_kernel.allocated_vgpr_count,
        "fresh VGPR diagnostic should not force fresh SGPRs");

  const char *failure = nullptr;
  std::optional<EdgeProbeRegisterSelection> no_spill =
      select_edge_probe_registers_from_liveness(
          kernel, arch, liveness, probe_points,
          /*previous_bb_probe_registers=*/true, /*stable_state_sgpr=*/true,
          Rdna4ProbeRegisters{}, /*allow_fresh_registers=*/false,
          /*allow_vgpr_scratch_spills=*/false, &failure);
  check(!no_spill.has_value(), "fixture should reject normal allocated VGPR selection");
  check(failure != nullptr, "normal VGPR selection failure should report a reason");

  failure = nullptr;
  std::optional<EdgeProbeRegisterSelection> selection =
      select_edge_probe_registers_from_liveness(
          kernel, arch, liveness, probe_points,
          /*previous_bb_probe_registers=*/true, /*stable_state_sgpr=*/true,
          Rdna4ProbeRegisters{}, /*allow_fresh_registers=*/false,
          /*allow_vgpr_scratch_spills=*/true, &failure);
  check(selection.has_value(),
        "runtime scratch-backed selection should allow loader-visible scratch growth");
  check(selection->scratch_spill_plan.has_value(),
        "runtime scratch-backed selection should produce a spill plan");
  check(selection->scratch_spill_plan->private_segment_bytes >
            kernel.private_segment_fixed_size,
        "runtime scratch-backed selection should report required private growth");
  check(failure == nullptr,
        "runtime scratch-backed selection should not report failure after metadata support");

  kernel.private_segment_fixed_size = 64;
  failure = nullptr;
  selection = select_edge_probe_registers_from_liveness(
      kernel, arch, liveness, probe_points,
      /*previous_bb_probe_registers=*/true, /*stable_state_sgpr=*/true,
      Rdna4ProbeRegisters{}, /*allow_fresh_registers=*/false,
      /*allow_vgpr_scratch_spills=*/true, &failure);
  check(selection.has_value(),
        "existing private segment bytes should still allow a larger spill plan");
  check(selection->scratch_spill_plan.has_value() &&
            selection->scratch_spill_plan->private_segment_bytes >
                kernel.private_segment_fixed_size,
        "existing private segment bytes are not reused for DBI spill slots");
  check(failure == nullptr,
        "scratch-backed selection with existing private segment should not fail");

  failure = nullptr;
  std::optional<EdgeProbeRegisterSelection> fixed_counter_selection =
      select_edge_probe_registers_from_liveness(
          kernel, arch, liveness, probe_points,
          /*previous_bb_probe_registers=*/false, /*stable_state_sgpr=*/true,
          Rdna4ProbeRegisters{}, /*allow_fresh_registers=*/false,
          /*allow_vgpr_scratch_spills=*/true, &failure);
  check(fixed_counter_selection.has_value(),
        "scratch-backed fixed-counter selection should reserve wrapper EXEC temporaries");
  check(fixed_counter_selection->scratch_spill_plan.has_value(),
        "scratch-backed fixed-counter selection should produce a spill plan");
  check(fixed_counter_selection->scratch_spill_plan->vgpr_spills.size() == 2,
        "fixed-counter scratch selection should spill only the two counter-probe VGPRs");
  check(fixed_counter_selection->scratch_spill_plan->sgpr_spills.empty(),
        "fixed-counter scratch selection with stable state should not spill SGPRs");
  check(fixed_counter_selection->probe_registers.saved_exec_sgpr !=
            fixed_counter_selection->probe_registers.state_sgpr,
        "conservative fixed-counter scratch selection should reserve saved EXEC SGPRs");
  check(failure == nullptr,
        "scratch-backed fixed-counter selection should not report failure");

  failure = nullptr;
  std::optional<EdgeProbeRegisterSelection> direct_exec_fixed_counter_selection =
      select_edge_probe_registers_from_liveness(
          kernel, arch, liveness, probe_points,
          /*previous_bb_probe_registers=*/false, /*stable_state_sgpr=*/true,
          Rdna4ProbeRegisters{}, /*allow_fresh_registers=*/false,
          /*allow_vgpr_scratch_spills=*/true, &failure,
          /*allow_direct_exec_fixed_counter_scratch_spills=*/true);
  check(direct_exec_fixed_counter_selection.has_value(),
        "direct-EXEC fixed-counter scratch selection should succeed");
  check(direct_exec_fixed_counter_selection->scratch_spill_plan.has_value(),
        "direct-EXEC fixed-counter selection should produce a spill plan");
  check(direct_exec_fixed_counter_selection->scratch_spill_plan->wave32,
        "direct-EXEC fixed-counter selection should preserve kernel wave32");
  check(direct_exec_fixed_counter_selection->probe_registers.saved_exec_sgpr ==
            direct_exec_fixed_counter_selection->probe_registers.state_sgpr,
        "direct-EXEC fixed-counter scratch wrapper should not reserve saved EXEC SGPRs");
  check(failure == nullptr,
        "direct-EXEC fixed-counter scratch selection should not report failure");

  KernelSite forced_lane_scratch_kernel = kernel;
  forced_lane_scratch_kernel.allocated_vgpr_count =
      static_cast<uint32_t>(rocjitsu::REGISTER_SET_MAX_VGPRS - 1);

  failure = nullptr;
  std::optional<EdgeProbeRegisterSelection> forced_lane_scratch_selection =
      select_edge_probe_registers_from_liveness(
          forced_lane_scratch_kernel, arch, liveness, probe_points,
          /*previous_bb_probe_registers=*/false, /*stable_state_sgpr=*/true,
          Rdna4ProbeRegisters{}, /*allow_fresh_registers=*/true,
          /*allow_vgpr_scratch_spills=*/true, &failure,
          /*allow_direct_exec_fixed_counter_scratch_spills=*/true,
          /*allow_sgpr_scratch_spills=*/false,
          /*force_fresh_sgprs=*/false,
          /*force_saved_exec_sgpr_pair=*/true,
          /*force_fresh_vgprs=*/true,
          /*force_fresh_scratch_address_vgpr=*/true);
  check(forced_lane_scratch_selection.has_value(),
        "forced-lane fixed counter should use scratch when only one fresh VGPR remains");
  check(forced_lane_scratch_selection->scratch_spill_plan.has_value(),
        "forced-lane fixed counter should produce a scratch plan");
  check(forced_lane_scratch_selection->scratch_spill_plan->address_vgpr >=
            forced_lane_scratch_kernel.allocated_vgpr_count,
        "forced-lane scratch selection should use a fresh address VGPR");
  check(forced_lane_scratch_selection->probe_registers.saved_exec_sgpr !=
            forced_lane_scratch_selection->probe_registers.state_sgpr,
        "forced-lane scratch selection should reserve a distinct saved EXEC pair");
  check(failure == nullptr,
        "forced-lane fixed-counter scratch selection should not report failure");

  KernelSite no_fresh_address_kernel = forced_lane_scratch_kernel;
  no_fresh_address_kernel.allocated_vgpr_count =
      static_cast<uint32_t>(rocjitsu::REGISTER_SET_MAX_VGPRS);

  failure = nullptr;
  std::optional<EdgeProbeRegisterSelection> saturated_forced_lane_scratch_selection =
      select_edge_probe_registers_from_liveness(
          no_fresh_address_kernel, arch, liveness, probe_points,
          /*previous_bb_probe_registers=*/false, /*stable_state_sgpr=*/true,
          Rdna4ProbeRegisters{}, /*allow_fresh_registers=*/false,
          /*allow_vgpr_scratch_spills=*/true, &failure,
          /*allow_direct_exec_fixed_counter_scratch_spills=*/true,
          /*allow_sgpr_scratch_spills=*/false,
          /*force_fresh_sgprs=*/false,
          /*force_saved_exec_sgpr_pair=*/true,
          /*force_fresh_vgprs=*/true,
          /*force_fresh_scratch_address_vgpr=*/false);
  check(saturated_forced_lane_scratch_selection.has_value(),
        "forced-lane scratch selection should support fully allocated VGPR files");
  check(!saturated_forced_lane_scratch_selection->uses_fresh_registers,
        "saturated forced-lane scratch selection should not require fresh VGPRs");
  check(saturated_forced_lane_scratch_selection->scratch_spill_plan.has_value(),
        "saturated forced-lane fixed counter should produce a scratch plan");
  check(saturated_forced_lane_scratch_selection->scratch_spill_plan->address_vgpr <
            no_fresh_address_kernel.allocated_vgpr_count,
        "saturated forced-lane scratch selection should use an allocated address VGPR");
  check(std::none_of(
            saturated_forced_lane_scratch_selection->scratch_spill_plan->vgpr_spills.begin(),
            saturated_forced_lane_scratch_selection->scratch_spill_plan->vgpr_spills.end(),
            [&](const ProbeScratchSpillSlot &slot) {
              return slot.vgpr ==
                     saturated_forced_lane_scratch_selection->scratch_spill_plan
                         ->address_vgpr;
            }),
        "saturated forced-lane scratch selection must not spill the address VGPR");
  check(saturated_forced_lane_scratch_selection->probe_registers.saved_exec_sgpr !=
            saturated_forced_lane_scratch_selection->probe_registers.state_sgpr,
        "saturated forced-lane scratch selection should still preserve EXEC");
  check(failure == nullptr,
        "saturated forced-lane fixed-counter scratch selection should not report failure");

  failure = nullptr;
  std::optional<EdgeProbeRegisterSelection> no_fresh_address_selection =
      select_edge_probe_registers_from_liveness(
          no_fresh_address_kernel, arch, liveness, probe_points,
          /*previous_bb_probe_registers=*/false, /*stable_state_sgpr=*/true,
          Rdna4ProbeRegisters{}, /*allow_fresh_registers=*/true,
          /*allow_vgpr_scratch_spills=*/true, &failure,
          /*allow_direct_exec_fixed_counter_scratch_spills=*/true,
          /*allow_sgpr_scratch_spills=*/false,
          /*force_fresh_sgprs=*/false,
          /*force_saved_exec_sgpr_pair=*/true,
          /*force_fresh_vgprs=*/true,
          /*force_fresh_scratch_address_vgpr=*/true);
  check(!no_fresh_address_selection.has_value(),
        "diagnostic fresh-address scratch selection should fail without a fresh VGPR");
  check(failure != nullptr,
        "diagnostic fresh-address scratch selection failure should report pressure");

  KernelSite sgpr_saturated_kernel = kernel;
  sgpr_saturated_kernel.allocated_sgpr_count = 0;

  failure = nullptr;
  std::optional<EdgeProbeRegisterSelection> previous_bb_sgpr_saturated =
      select_edge_probe_registers_from_liveness(
          sgpr_saturated_kernel, arch, liveness, probe_points,
          /*previous_bb_probe_registers=*/true, /*stable_state_sgpr=*/true,
          Rdna4ProbeRegisters{}, /*allow_fresh_registers=*/false,
          /*allow_vgpr_scratch_spills=*/true, &failure,
          /*allow_direct_exec_fixed_counter_scratch_spills=*/true);
  check(!previous_bb_sgpr_saturated.has_value(),
        "previous-BB scratch selection should still require saved EXEC SGPRs");
  check(failure != nullptr,
        "previous-BB scratch rejection should report the saved EXEC pressure");

  failure = nullptr;
  std::optional<EdgeProbeRegisterSelection> fixed_counter_sgpr_saturated =
      select_edge_probe_registers_from_liveness(
          sgpr_saturated_kernel, arch, liveness, probe_points,
          /*previous_bb_probe_registers=*/false, /*stable_state_sgpr=*/true,
          Rdna4ProbeRegisters{}, /*allow_fresh_registers=*/false,
          /*allow_vgpr_scratch_spills=*/true, &failure,
          /*allow_direct_exec_fixed_counter_scratch_spills=*/true);
  check(fixed_counter_sgpr_saturated.has_value(),
        "fixed-counter scratch selection should not require saved EXEC SGPRs");
  check(fixed_counter_sgpr_saturated->scratch_spill_plan.has_value(),
        "SGPR-saturated fixed-counter selection should use scratch backing");
  check(fixed_counter_sgpr_saturated->probe_registers.saved_exec_sgpr ==
            fixed_counter_sgpr_saturated->probe_registers.state_sgpr,
        "SGPR-saturated fixed-counter selection should share the stable state SGPR");
  check(failure == nullptr,
        "SGPR-saturated fixed-counter scratch selection should not report failure");

  KernelSite saved_exec_spill_kernel = kernel;
  saved_exec_spill_kernel.allocated_sgpr_count = 5;

  failure = nullptr;
  std::optional<EdgeProbeRegisterSelection> no_sgpr_spill =
      select_edge_probe_registers_from_liveness(
          saved_exec_spill_kernel, arch, liveness, probe_points,
          /*previous_bb_probe_registers=*/true, /*stable_state_sgpr=*/true,
          Rdna4ProbeRegisters{}, /*allow_fresh_registers=*/false,
          /*allow_vgpr_scratch_spills=*/true, &failure,
          /*allow_direct_exec_fixed_counter_scratch_spills=*/false,
          /*allow_sgpr_scratch_spills=*/false);
  check(!no_sgpr_spill.has_value(),
        "previous-BB scratch selection should fail without a saved EXEC spill pair");
  check(failure != nullptr,
        "previous-BB scratch selection failure should report saved EXEC pressure");

  failure = nullptr;
  std::optional<EdgeProbeRegisterSelection> sgpr_spill_selection =
      select_edge_probe_registers_from_liveness(
          saved_exec_spill_kernel, arch, liveness, probe_points,
          /*previous_bb_probe_registers=*/true, /*stable_state_sgpr=*/true,
          Rdna4ProbeRegisters{}, /*allow_fresh_registers=*/false,
          /*allow_vgpr_scratch_spills=*/true, &failure,
          /*allow_direct_exec_fixed_counter_scratch_spills=*/false,
          /*allow_sgpr_scratch_spills=*/true);
  check(sgpr_spill_selection.has_value(),
        "SGPR scratch spills should recover previous-BB scalar state selection");
  check(sgpr_spill_selection->probe_registers.saved_exec_sgpr == 0,
        "SGPR scratch spill selection should reuse the allocated low SGPR pair");
  check(sgpr_spill_selection->probe_registers.tmp0_sgpr == 2 &&
            sgpr_spill_selection->probe_registers.tmp1_sgpr == 2 &&
            sgpr_spill_selection->probe_registers.scc_sgpr == 4,
        "SGPR scratch spill selection should reserve previous-BB temp and SCC SGPRs");
  check(sgpr_spill_selection->scratch_spill_plan.has_value() &&
            sgpr_spill_selection->scratch_spill_plan->sgpr_spills.size() == 5,
        "SGPR scratch spill selection should spill saved EXEC, temp, and SCC SGPRs");
  check(sgpr_spill_selection->scratch_spill_plan->vgpr_spills.size() == 4,
        "SGPR scratch spill selection should still spill previous-BB VGPR temporaries");
  check(failure == nullptr,
        "SGPR scratch spill selection should not report failure");
}

std::vector<std::unique_ptr<rocjitsu::BasicBlock>>
build_test_blocks(std::initializer_list<TestOpcode> opcodes) {
  std::vector<uint32_t> words;
  words.reserve(opcodes.size());
  for (TestOpcode opcode : opcodes)
    words.push_back(static_cast<uint32_t>(opcode));

  TestCodeObject co(words);
  TestDecoder decoder;
  std::vector<std::unique_ptr<rocjitsu::BasicBlock>> blocks =
      rocjitsu::BasicBlock::build(co, decoder);
  check(blocks.size() == 1, "block-entry patch fixture should build one block");
  check(blocks[0]->num_instructions() == words.size(),
        "block-entry patch fixture should preserve instruction count");
  return blocks;
}

std::vector<std::unique_ptr<rocjitsu::BasicBlock>>
build_saveexec_vopd_test_blocks(TestOpcode vopd_opcode) {
  const std::array<uint32_t, 3> words = {
      static_cast<uint32_t>(TestOpcode::SaveExec),
      static_cast<uint32_t>(vopd_opcode),
      0,
  };
  TestCodeObject co(words);
  TestDecoder decoder;
  std::vector<std::unique_ptr<rocjitsu::BasicBlock>> blocks =
      rocjitsu::BasicBlock::build(co, decoder);
  check(blocks.size() == 1, "VOPD block-entry fixture should build one block");
  check(blocks[0]->num_instructions() == 2,
        "VOPD block-entry fixture should model VOPD as a two-word instruction");
  return blocks;
}

void check_block_entry_patch_selection() {
  using namespace rocjitsu::fuzzer::afl_dbi;

  std::vector<std::unique_ptr<rocjitsu::BasicBlock>> wait_saveexec_blocks =
      build_test_blocks({TestOpcode::WaitKmcnt, TestOpcode::SaveExec,
                         TestOpcode::DelayAlu, TestOpcode::Relocatable,
                         TestOpcode::End});
  const rocjitsu::Instruction &wait_first = *wait_saveexec_blocks[0]->instructions().begin();
  std::string_view skip_reason;
  std::optional<BlockEntryPatchPoint> wait_saveexec_patch =
      select_block_entry_patch_point(*wait_saveexec_blocks[0], wait_first, &skip_reason);
  check(wait_saveexec_patch.has_value(),
        "wait/saveexec prefix should select a following relocatable patch point");
  check(wait_saveexec_patch->text_offset == 3 * sizeof(uint32_t),
        "wait/saveexec prefix patch point should skip hazard prefix instructions");
  check(wait_saveexec_patch->instruction->mnemonic() == "v_add_u32",
        "wait/saveexec prefix patch point should select the first relocatable instruction");

  std::vector<std::unique_ptr<rocjitsu::BasicBlock>> saveexec_blocks =
      build_test_blocks({TestOpcode::SaveExec, TestOpcode::Relocatable, TestOpcode::End});
  const rocjitsu::Instruction &saveexec_first = *saveexec_blocks[0]->instructions().begin();
  std::optional<BlockEntryPatchPoint> saveexec_patch =
      select_block_entry_patch_point(*saveexec_blocks[0], saveexec_first, &skip_reason);
  check(saveexec_patch.has_value(),
        "saveexec-fronted block should select a following relocatable patch point");
  check(saveexec_patch->text_offset == sizeof(uint32_t),
        "saveexec-fronted block patch point should leave the EXEC transition in place");

  std::vector<std::unique_ptr<rocjitsu::BasicBlock>> prefix_then_control_blocks =
      build_test_blocks({TestOpcode::WaitKmcnt, TestOpcode::SaveExec, TestOpcode::End});
  const rocjitsu::Instruction &control_first =
      *prefix_then_control_blocks[0]->instructions().begin();
  skip_reason = {};
  BlockEntryPatchSkip control_skip;
  std::optional<BlockEntryPatchPoint> prefix_then_control_patch =
      select_block_entry_patch_point(*prefix_then_control_blocks[0], control_first, &skip_reason,
                                     &control_skip);
  check(!prefix_then_control_patch.has_value(),
        "wait/saveexec prefix before control flow should stay skipped");
  check(skip_reason == "entry instruction is control flow and is not PC-relocatable yet",
        "wait/saveexec prefix before control flow should report the precise blocker");
  check(control_skip.text_offset == 2 * sizeof(uint32_t),
        "wait/saveexec prefix skip should report the delayed blocker offset");
  check(control_skip.instruction != nullptr &&
            control_skip.instruction->mnemonic() == "test_end",
        "wait/saveexec prefix skip should report the delayed blocker instruction");

  std::vector<std::unique_ptr<rocjitsu::BasicBlock>> prefix_only_blocks =
      build_test_blocks({TestOpcode::WaitKmcnt, TestOpcode::SaveExec});
  const rocjitsu::Instruction &prefix_only_first = *prefix_only_blocks[0]->instructions().begin();
  skip_reason = {};
  BlockEntryPatchSkip prefix_only_skip;
  std::optional<BlockEntryPatchPoint> prefix_only_patch =
      select_block_entry_patch_point(*prefix_only_blocks[0], prefix_only_first, &skip_reason,
                                     &prefix_only_skip);
  check(!prefix_only_patch.has_value(), "prefix-only block should stay skipped");
  check(skip_reason == "entry EXEC transition has no following relocatable instruction",
        "prefix-only EXEC transition should report missing relocatable follower");
  check(prefix_only_skip.instruction == nullptr &&
            prefix_only_skip.text_offset == 2 * sizeof(uint32_t),
        "prefix-only EXEC transition should report the end of the skipped prefix");

  std::vector<std::unique_ptr<rocjitsu::BasicBlock>> prefix_then_opaque_blocks =
      build_test_blocks({TestOpcode::SaveExec, TestOpcode::Opaque});
  const rocjitsu::Instruction &opaque_first =
      *prefix_then_opaque_blocks[0]->instructions().begin();
  skip_reason = {};
  BlockEntryPatchSkip opaque_skip;
  std::optional<BlockEntryPatchPoint> prefix_then_opaque_patch =
      select_block_entry_patch_point(*prefix_then_opaque_blocks[0], opaque_first, &skip_reason,
                                     &opaque_skip);
  check(!prefix_then_opaque_patch.has_value(),
        "saveexec prefix before opaque instruction should stay skipped");
  check(skip_reason == "entry instruction is an unknown encoding and is not relocatable yet",
        "saveexec prefix before opaque instruction should report the opaque blocker");
  check(opaque_skip.text_offset == sizeof(uint32_t) &&
            opaque_skip.instruction != nullptr &&
            opaque_skip.instruction->mnemonic() == "unknown_opaque",
        "saveexec prefix before opaque instruction should report the opaque blocker instruction");

  std::vector<std::unique_ptr<rocjitsu::BasicBlock>> modeled_vopd_blocks =
      build_saveexec_vopd_test_blocks(TestOpcode::ModeledVopd);
  const rocjitsu::Instruction &modeled_vopd_first =
      *modeled_vopd_blocks[0]->instructions().begin();
  skip_reason = {};
  std::optional<BlockEntryPatchPoint> modeled_vopd_patch = select_block_entry_patch_point(
      *modeled_vopd_blocks[0], modeled_vopd_first, &skip_reason);
  check(modeled_vopd_patch.has_value(),
        "saveexec prefix before modeled VOPD should select a replayable patch point");
  check(modeled_vopd_patch->text_offset == sizeof(uint32_t) &&
            modeled_vopd_patch->instruction != nullptr &&
            modeled_vopd_patch->instruction->mnemonic() == "vopd_opaque",
        "modeled VOPD patch should report the concrete replayed instruction");

  std::vector<std::unique_ptr<rocjitsu::BasicBlock>> unmodeled_vopd_blocks =
      build_saveexec_vopd_test_blocks(TestOpcode::UnmodeledVopd);
  const rocjitsu::Instruction &unmodeled_vopd_first =
      *unmodeled_vopd_blocks[0]->instructions().begin();
  skip_reason = {};
  BlockEntryPatchSkip unmodeled_vopd_skip;
  std::optional<BlockEntryPatchPoint> unmodeled_vopd_patch = select_block_entry_patch_point(
      *unmodeled_vopd_blocks[0], unmodeled_vopd_first, &skip_reason, &unmodeled_vopd_skip);
  check(!unmodeled_vopd_patch.has_value(),
        "saveexec prefix before unmodeled VOPD should stay skipped");
  check(skip_reason ==
            "entry VOPD liveness is unmodeled and replay relocation is not enabled yet",
        "unmodeled VOPD block-entry skip reason should keep the liveness gap explicit");
  check(unmodeled_vopd_skip.text_offset == sizeof(uint32_t) &&
            unmodeled_vopd_skip.instruction != nullptr &&
            unmodeled_vopd_skip.instruction->mnemonic() == "vopd_opaque",
        "unmodeled VOPD skip should report the concrete blocker instruction");
}

} // namespace

int main() {
  using namespace rocjitsu::fuzzer::afl_dbi;

  check(probe_target_for_elf_mach(0xdeadbeefu) == nullptr,
        "unknown ELF machine should not map to a probe target");
  check(std::string_view(arch_name(ROCJITSU_CODE_ARCH_RDNA3)) == "RDNA3/gfx110x",
        "planner arch name should be available outside the preload");
  check(stable_bb_id("kernel", 0x10) == stable_bb_id("kernel", 0x10),
        "stable BB ID should be deterministic");
  check(stable_bb_id("kernel", 0x10) != stable_bb_id("kernel", 0x14),
        "stable BB ID should include block offset");
  check(stable_edge_id("kernel", 0x10, 0x20) == stable_edge_id("kernel", 0x10, 0x20),
        "stable edge ID should be deterministic");

  std::optional<Vopd64LivenessModel> vopd_mov_and =
      decode_vopd64_liveness_model(3391357056u, 36569223u);
  check(vopd_mov_and.has_value(), "sampled VOPD64 should decode liveness model");
  check(vopd_mov_and->defs.contains({rocjitsu::RegClass::VGPR, 2, 1}), "VOPD X dst");
  check(vopd_mov_and->defs.contains({rocjitsu::RegClass::VGPR, 47, 1}), "VOPD Y dst");
  check(vopd_mov_and->uses.contains({rocjitsu::RegClass::VGPR, 0, 1}), "VOPD Y vsrc");
  check(!vopd_mov_and->uses.contains({rocjitsu::RegClass::VGPR, 2, 1}),
        "non-accumulate VOPD X dst should not be modeled as a source");
  check(!vopd_mov_and->uses.contains({rocjitsu::RegClass::VGPR, 47, 1}),
        "non-accumulate VOPD Y dst should not be modeled as a source");
  check(!vopd_mov_and->uses.contains({rocjitsu::RegClass::SGPR, 3, 1}),
        "inline src0 should not be modeled as an SGPR");
  std::optional<Vopd64LivenessModel> vopd_mov_s3_and =
      decode_vopd64_liveness_model(3391356931u, 70123655u);
  check(vopd_mov_s3_and.has_value(), "sampled VOPD64 SGPR source should decode");
  check(vopd_mov_s3_and->uses.contains({rocjitsu::RegClass::SGPR, 3, 1}),
        "VOPD src0 SGPR should be modeled");
  check(!decode_vopd64_liveness_model(127u, 0).has_value(),
        "invalid word should not decode as VOPD64");
  check(decode_vopd64_word_count(3391357056u, 36569223u) == 2,
        "VOPD without literal should consume two words");
  check(decode_vopd64_word_count(3391094912u, 238447871u) == 3,
        "VOPD with literal src0 should consume three words");
  check(decode_vopd64_word_count(127u, 0) == 0,
        "invalid VOPD should not report an encoded size");

  constexpr uint32_t vopd_rmw_word0 =
      (0x32u << 26) | (0xcu << 17) | (3u << 9) | 256u;
  constexpr uint32_t vopd_rmw_word1 =
      (6u << 24) | (4u << 17) | (4u << 9) | 257u;
  std::optional<Vopd64LivenessModel> vopd_rmw =
      decode_vopd64_liveness_model(vopd_rmw_word0, vopd_rmw_word1);
  check(vopd_rmw.has_value(), "VOPD accumulate fixture should decode");
  check(vopd_rmw->defs.contains({rocjitsu::RegClass::VGPR, 6, 1}),
        "VOPD accumulate X dst should be modeled as a def");
  check(vopd_rmw->uses.contains({rocjitsu::RegClass::VGPR, 6, 1}),
        "v_fmac_f32 VOPD X dst should be modeled as a source");
  check(vopd_rmw->defs.contains({rocjitsu::RegClass::VGPR, 9, 1}),
        "VOPD accumulate Y dst should be modeled as a def");
  check(vopd_rmw->uses.contains({rocjitsu::RegClass::VGPR, 9, 1}),
        "v_dot2acc VOPD Y dst should be modeled as a source");

  constexpr uint32_t vopd_mov_mov_word0 =
      (0x32u << 26) | (0x8u << 22) | (0x8u << 17) | (7u << 9) | 260u;
  constexpr uint32_t vopd_mov_mov_word1 =
      (10u << 24) | (6u << 17) | (8u << 9) | 261u;
  std::optional<Vopd64LivenessModel> vopd_mov_mov =
      decode_vopd64_liveness_model(vopd_mov_mov_word0, vopd_mov_mov_word1);
  check(vopd_mov_mov.has_value(), "VOPD move/move fixture should decode");
  check(vopd_mov_mov->uses.contains({rocjitsu::RegClass::VGPR, 4, 1}),
        "VOPD move X src0 VGPR should be modeled as a source");
  check(vopd_mov_mov->uses.contains({rocjitsu::RegClass::VGPR, 5, 1}),
        "VOPD move Y src0 VGPR should be modeled as a source");
  check(!vopd_mov_mov->uses.contains({rocjitsu::RegClass::VGPR, 7, 1}),
        "VOPD move X vsrc1 placeholder should not be modeled as a source");
  check(!vopd_mov_mov->uses.contains({rocjitsu::RegClass::VGPR, 8, 1}),
        "VOPD move Y vsrc1 placeholder should not be modeled as a source");

  constexpr uint32_t vopd_y_only_word0 =
      (0x32u << 26) | (0x8u << 22) | (0x10u << 17) | (7u << 9) | 260u;
  constexpr uint32_t vopd_y_only_word1 =
      (12u << 24) | (6u << 17) | (8u << 9) | 261u;
  std::optional<Vopd64LivenessModel> vopd_y_only =
      decode_vopd64_liveness_model(vopd_y_only_word0, vopd_y_only_word1);
  check(vopd_y_only.has_value(), "VOPD Y-only opcode fixture should decode");
  check(vopd_y_only->defs.contains({rocjitsu::RegClass::VGPR, 12, 1}),
        "VOPD Y-only dst should be modeled as a def");
  check(vopd_y_only->uses.contains({rocjitsu::RegClass::VGPR, 8, 1}),
        "VOPD Y-only vsrc1 should be modeled as a source");
  check(!vopd_y_only->uses.contains({rocjitsu::RegClass::VGPR, 12, 1}),
        "VOPD Y-only dst should not be modeled as a source");

  constexpr uint32_t vopd_reserved_x_word0 =
      (0x32u << 26) | (0xeu << 22) | (0x8u << 17) | (7u << 9) | 260u;
  constexpr uint32_t vopd_reserved_word1 =
      (10u << 24) | (6u << 17) | (8u << 9) | 261u;
  check(!decode_vopd64_liveness_model(vopd_reserved_x_word0, vopd_reserved_word1)
             .has_value(),
        "reserved VOPD X opcode should stay conservatively opaque");
  check(decode_vopd64_word_count(vopd_reserved_x_word0, vopd_reserved_word1) == 2,
        "reserved VOPD opcode should still preserve CFG instruction size");

  constexpr uint32_t vopd_y_bf16_word0 =
      (0x32u << 26) | (0x8u << 22) | (0xdu << 17) | (7u << 9) | 260u;
  std::optional<Vopd64LivenessModel> vopd_y_bf16 =
      decode_vopd64_liveness_model(vopd_y_bf16_word0, vopd_reserved_word1);
  check(vopd_y_bf16.has_value(),
        "VOPD BF16 dot accumulate should decode in the Y slot");

  constexpr uint32_t vopd_x_bf16_word0 =
      (0x32u << 26) | (0xdu << 22) | (0x8u << 17) | (7u << 9) | 260u;
  check(!decode_vopd64_liveness_model(vopd_x_bf16_word0, vopd_reserved_word1)
             .has_value(),
        "Y-only VOPD BF16 dot opcode should stay opaque in the X slot");

  constexpr uint32_t vopd_reserved_y_word0 =
      (0x32u << 26) | (0x8u << 22) | (0x13u << 17) | (7u << 9) | 260u;
  check(!decode_vopd64_liveness_model(vopd_reserved_y_word0, vopd_reserved_word1)
             .has_value(),
        "reserved VOPD Y opcode should stay conservatively opaque");

  const std::array<uint32_t, 3> rdna4_vop3_lit = {
      0xd657000bu, 0x007200ffu, 0x0000007fu};
  check_decodes_one(ROCJITSU_CODE_ARCH_RDNA4, rdna4_vop3_lit.data(),
                    rdna4_vop3_lit.size(), "v_and_or_b32", 12, 0x0000007fu);
  const std::array<uint32_t, 3> rdna3_vop3_lit = {
      0xd5030000u, 0x000200ffu, 0x0000007fu};
  check_decodes_one(ROCJITSU_CODE_ARCH_RDNA3, rdna3_vop3_lit.data(),
                    rdna3_vop3_lit.size(), "v_add_f32", 12, 0x0000007fu);
  check_decodes_one(ROCJITSU_CODE_ARCH_RDNA3_5, rdna3_vop3_lit.data(),
                    rdna3_vop3_lit.size(), "v_add_f32", 12, 0x0000007fu);
  check_block_entry_direct_branch_trampoline(ROCJITSU_CODE_ARCH_RDNA3);
  check_block_entry_direct_branch_trampoline(ROCJITSU_CODE_ARCH_RDNA4);
  check_conditional_block_entry_trampoline(ROCJITSU_CODE_ARCH_RDNA3);
  check_conditional_block_entry_trampoline(ROCJITSU_CODE_ARCH_RDNA4);
  check_conditional_previous_bb_branch_trampoline(ROCJITSU_CODE_ARCH_RDNA3);
  check_conditional_previous_bb_branch_trampoline(ROCJITSU_CODE_ARCH_RDNA3_5);
  check_conditional_previous_bb_branch_trampoline(ROCJITSU_CODE_ARCH_RDNA4);
  check_exec_conditioned_fixed_branch_trampoline(ROCJITSU_CODE_ARCH_RDNA3, 0x25);
  check_exec_conditioned_fixed_branch_trampoline(ROCJITSU_CODE_ARCH_RDNA3, 0x26);
  check_exec_conditioned_fixed_branch_trampoline(ROCJITSU_CODE_ARCH_RDNA3_5, 0x25);
  check_exec_conditioned_fixed_branch_trampoline(ROCJITSU_CODE_ARCH_RDNA3_5, 0x26);
  check_exec_conditioned_fixed_branch_trampoline(ROCJITSU_CODE_ARCH_RDNA4, 0x25);
  check_exec_conditioned_fixed_branch_trampoline(ROCJITSU_CODE_ARCH_RDNA4, 0x26);

  const std::array<uint32_t, 4> gfx11_scratch_words = {
      0xDC690010u, 0x00820100u, 0xDC510010u, 0x03820000u};
  check_scratch_spill_builders(ROCJITSU_CODE_ARCH_RDNA3, gfx11_scratch_words);
  check_scratch_spill_builders(ROCJITSU_CODE_ARCH_RDNA3_5, gfx11_scratch_words);
  const std::array<uint32_t, 6> gfx12_scratch_words = {
      0xED068002u, 0x00820000u, 0x00001000u,
      0xED050002u, 0x00020003u, 0x00001000u};
  check_scratch_spill_builders(ROCJITSU_CODE_ARCH_RDNA4, gfx12_scratch_words);
  check_vgpr_scratch_spill_wrapper(ROCJITSU_CODE_ARCH_RDNA3);
  check_vgpr_scratch_spill_wrapper(ROCJITSU_CODE_ARCH_RDNA3_5);
  check_vgpr_scratch_spill_wrapper(ROCJITSU_CODE_ARCH_RDNA4);
  check_vgpr_scratch_spill_private_growth_invariant(ROCJITSU_CODE_ARCH_RDNA3);
  check_vgpr_scratch_spill_private_growth_invariant(ROCJITSU_CODE_ARCH_RDNA3_5);
  check_vgpr_scratch_spill_private_growth_invariant(ROCJITSU_CODE_ARCH_RDNA4);
  check_vgpr_scratch_spill_liveness_selection(ROCJITSU_CODE_ARCH_RDNA3);
  check_vgpr_scratch_spill_liveness_selection(ROCJITSU_CODE_ARCH_RDNA3_5);
  check_vgpr_scratch_spill_liveness_selection(ROCJITSU_CODE_ARCH_RDNA4);
  check_block_entry_patch_selection();
  const std::array<uint8_t, 1> one_spill = {5};
  check(!plan_vgpr_scratch_spills(ROCJITSU_CODE_ARCH_CDNA3, /*address_vgpr=*/4,
                                  one_spill, /*original_private_segment_bytes=*/0)
             .has_value(),
        "VGPR scratch spill planning should reject unsupported targets");
  Rdna4ProbeBuilder unsupported_scratch_builder({}, ROCJITSU_CODE_ARCH_CDNA3);
  check(!unsupported_scratch_builder.scratch_store_b32(/*vaddr=*/0, /*vdata=*/1,
                                                       /*saddr=*/2,
                                                       /*byte_offset=*/16),
        "scratch store builder should reject unsupported targets");
  check(!unsupported_scratch_builder.scratch_load_b32(/*vdst=*/3, /*vaddr=*/0,
                                                      /*saddr=*/2,
                                                      /*byte_offset=*/16),
        "scratch load builder should reject unsupported targets");
  check(unsupported_scratch_builder.take().empty(),
        "unsupported scratch builders should not append words");

  constexpr ProbeTarget unsupported_target{0, "unsupported", ROCJITSU_CODE_ARCH_INVALID,
                                           ProbeMemoryModel::Unsupported,
                                           ProbeInstrumentationTier::Unsupported};
  check(std::string_view(probe_target_edge_support_reason(unsupported_target)) ==
            "unsupported-probe-target",
        "unsupported probe targets should report unsupported edge coverage");

  DeviceElfPatchPlan patch_plan;
  patch_plan.entry_candidate_count = 3;
  patch_plan.entry_backed_edge_sites.resize(1);
  patch_plan.self_contained_edge_sites.resize(2);
  patch_plan.edge_sites.resize(3);
  patch_plan.edge_patch_failures = 1;
  patch_plan.branch_range_failures = 1;
  patch_plan.hybrid_edge_probes = true;
  patch_plan.local_text_cave_summary.range_count = 2;
  patch_plan.local_text_cave_summary.total_bytes = 96;
  patch_plan.local_text_cave_summary.largest_range_bytes = 64;
  patch_plan.sampled_edge_failures.push_back(
      {"kernel", "branch", /*patch_text_offset=*/16, /*return_text_offset=*/32,
       "branch range exceeds s_branch simm16"});
  EdgeSite sampled_edge;
  sampled_edge.kernel_name = "kernel";
  sampled_edge.kind = EdgePatchKind::ConditionalBranchTerminator;
  sampled_edge.pred_text_offset = 96;
  sampled_edge.block_text_offset = 160;
  sampled_edge.patch_text_offset = 128;
  sampled_edge.return_text_offset = 160;
  sampled_edge.bb_id = 0x1234u;
  sampled_edge.fallthrough_bb_id = 0x5678u;
  sampled_edge.slot_policy = EdgeSlotPolicyKind::FixedCounter;
  sampled_edge.fixed_slot = 17;
  sampled_edge.fallthrough_slot = 18;
  sampled_edge.self_contained_probe = true;
  ProbeScratchSpillPlan sampled_spill_plan;
  sampled_spill_plan.address_vgpr = 4;
  sampled_spill_plan.private_segment_bytes = 16;
  sampled_spill_plan.vgpr_spills.push_back({5, 0});
  sampled_edge.scratch_spill_plan = sampled_spill_plan;
  patch_plan.edge_trampolines.push_back(
      PlannedEdgeTrampoline{sampled_edge, EdgeTrampoline{},
                            EdgePatchResult{EdgeTrampolinePlacement::AppendedCave, 2048}});
  EdgeSite exec_empty_edge = sampled_edge;
  exec_empty_edge.fixed_slot = 19;
  exec_empty_edge.fallthrough_slot = 20;
  exec_empty_edge.branch_opcode = 0x25;
  exec_empty_edge.force_lane0_exec_for_fixed_counter = true;
  exec_empty_edge.scratch_spill_plan.reset();
  EdgeTrampoline exec_empty_trampoline;
  exec_empty_trampoline.cave_words.resize(5);
  patch_plan.edge_trampolines.push_back(PlannedEdgeTrampoline{
      exec_empty_edge, exec_empty_trampoline,
      EdgePatchResult{EdgeTrampolinePlacement::AppendedCave, 3072}});
  EdgeSite previous_bb_edge = sampled_edge;
  previous_bb_edge.slot_policy = EdgeSlotPolicyKind::PreviousBbHash;
  previous_bb_edge.fixed_slot = 0;
  previous_bb_edge.fallthrough_slot = 0;
  previous_bb_edge.scratch_spill_plan.reset();
  EdgeTrampoline previous_bb_trampoline;
  previous_bb_trampoline.cave_words.resize(7);
  patch_plan.edge_trampolines.push_back(PlannedEdgeTrampoline{
      previous_bb_edge, previous_bb_trampoline,
      EdgePatchResult{EdgeTrampolinePlacement::LocalTextCave, 4096}});
  patch_plan.edge_selection.failure_reason = "cfg decoded with skipped sites";
  patch_plan.edge_selection.slot_policy_summary.hashed_edge_sites = 5;
  patch_plan.edge_selection.slot_policy_summary.fixed_edge_sites = 7;
  patch_plan.edge_selection.slot_policy_summary.fixed_slot_requests = 13;
  patch_plan.edge_selection.slot_policy_summary.fixed_slots_reserved = 12;
  patch_plan.edge_selection.slot_policy_summary.fixed_slot_exhaustions = 3;
  patch_plan.edge_selection.slot_policy_summary.fixed_slot_collisions = 4;
  patch_plan.edge_selection.slot_policy_summary.inline_slot_requests = 11;
  patch_plan.edge_selection.slot_policy_summary.inline_slot_exhaustions = 2;
  KernelEdgeSelectionSummary kernel_summary;
  kernel_summary.kernel_name = "kernel";
  kernel_summary.previous_bb_branch_edges_selected = 6;
  kernel_summary.branch_edges_degraded_to_fixed = 4;
  kernel_summary.fixed_counter_branch_edge_aggregate_fallback_used = 1;
  kernel_summary.fixed_counter_branch_edge_safety_fallback_used = 2;
  kernel_summary.fixed_counter_branch_edge_liveness_fallback_used = 1;
  kernel_summary.opaque_instruction_count = 9;
  kernel_summary.unmodeled_opaque_instruction_count = 2;
  kernel_summary.fresh_register_growth_disabled_by_opaque_probe_points = 6;
  kernel_summary.opaque_fresh_register_candidate_probe_points = 7;
  kernel_summary.opaque_fresh_register_candidate_sgpr_growth_probe_points = 5;
  kernel_summary.opaque_fresh_register_candidate_vgpr_growth_probe_points = 3;
  kernel_summary.opaque_fresh_register_candidate_required_sgprs = 16;
  kernel_summary.opaque_fresh_register_candidate_required_vgprs = 104;
  kernel_summary.sgpr_scratch_spill_disabled_by_opaque_probe_points = 4;
  kernel_summary.sgpr_scratch_spill_disabled_by_exec_condition_probe_points = 5;
  kernel_summary.direct_exec_fixed_scratch_disabled_by_opaque_probe_points = 3;
  kernel_summary.sampled_opaque_instructions.push_back(
      {"vopd_opaque", /*text_offset=*/32, {0xc8000000u, 0xdeadbeefu}, true});
  OpaqueFreshRegisterCandidateSample fresh_candidate_sample;
  fresh_candidate_sample.kind = "branch";
  fresh_candidate_sample.patch_text_offset = 64;
  fresh_candidate_sample.mnemonic = "s_cbranch_scc1";
  fresh_candidate_sample.words.push_back(0xbf850002u);
  fresh_candidate_sample.required_sgprs = 16;
  fresh_candidate_sample.required_vgprs = 80;
  fresh_candidate_sample.allocated_sgprs = 8;
  fresh_candidate_sample.allocated_vgprs = 80;
  fresh_candidate_sample.sgpr_growth = true;
  fresh_candidate_sample.previous_bb_probe_registers = true;
  fresh_candidate_sample.slot_policy = "previous-bb-hash";
  fresh_candidate_sample.state_sgpr = 8;
  fresh_candidate_sample.saved_exec_sgpr = 10;
  fresh_candidate_sample.tmp0_sgpr = 12;
  fresh_candidate_sample.tmp1_sgpr = 14;
  fresh_candidate_sample.workitem_vgpr = 73;
  fresh_candidate_sample.tmp0_vgpr = 74;
  fresh_candidate_sample.tmp1_vgpr = 75;
  fresh_candidate_sample.tmp2_vgpr = 76;
  kernel_summary.sampled_opaque_fresh_register_candidates.push_back(
      fresh_candidate_sample);
  EdgeSiteSkipSample skip_sample;
  skip_sample.kind = "branch";
  skip_sample.text_offset = 64;
  skip_sample.reason = "terminator is not a direct patchable branch";
  skip_sample.mnemonic = "s_cbranch_scc1";
  skip_sample.instruction_size = 4;
  skip_sample.instruction_flags = rocjitsu::COND_BRANCH;
  skip_sample.words.push_back(0xbf850002u);
  kernel_summary.sampled_skips.push_back(skip_sample);
  patch_plan.edge_selection.kernel_summaries.push_back(kernel_summary);

  PatchDeviceElfReport patch_report;
  record_patch_plan_summary(patch_report, patch_plan);
  check(patch_report.entry_candidate_count == 3,
        "patch plan summary should record entry candidates");
  check(patch_report.entry_backed_edge_kernels == 1,
        "patch plan summary should record entry-backed kernels");
  check(patch_report.self_contained_edge_kernels == 2,
        "patch plan summary should record self-contained kernels");
  check(patch_report.hybrid_edge_probes, "patch plan summary should preserve hybrid flag");
  check(patch_report.cfg_failure_reason == "cfg decoded with skipped sites",
        "patch plan summary should record CFG failure reason");
  check(patch_report.kernel_summaries.size() == 1,
        "patch plan summary should copy kernel summaries");
  const KernelEdgeSelectionSummary &copied_summary = patch_report.kernel_summaries[0];
  check(copied_summary.opaque_instruction_count == 9 &&
            copied_summary.unmodeled_opaque_instruction_count == 2 &&
            copied_summary.fresh_register_growth_disabled_by_opaque_probe_points == 6 &&
            copied_summary.opaque_fresh_register_candidate_probe_points == 7 &&
            copied_summary.opaque_fresh_register_candidate_sgpr_growth_probe_points == 5 &&
            copied_summary.opaque_fresh_register_candidate_vgpr_growth_probe_points == 3 &&
            copied_summary.opaque_fresh_register_candidate_required_sgprs == 16 &&
            copied_summary.opaque_fresh_register_candidate_required_vgprs == 104 &&
            copied_summary.sgpr_scratch_spill_disabled_by_opaque_probe_points == 4 &&
            copied_summary.sgpr_scratch_spill_disabled_by_exec_condition_probe_points == 5 &&
            copied_summary.direct_exec_fixed_scratch_disabled_by_opaque_probe_points == 3,
        "patch plan summary should preserve opaque liveness diagnostics");
  check(copied_summary.sampled_opaque_instructions.size() == 1 &&
            copied_summary.sampled_opaque_instructions[0].words.size() == 2 &&
            copied_summary.sampled_opaque_instructions[0].liveness_modeled,
        "patch plan summary should preserve opaque instruction samples");
  check(copied_summary.sampled_opaque_fresh_register_candidates.size() == 1 &&
            copied_summary.sampled_opaque_fresh_register_candidates[0].patch_text_offset ==
                64 &&
            copied_summary.sampled_opaque_fresh_register_candidates[0].sgpr_growth &&
            copied_summary.sampled_opaque_fresh_register_candidates[0].saved_exec_sgpr == 10,
        "patch plan summary should preserve opaque fresh-register candidate samples");
  check(copied_summary.sampled_skips.size() == 1 &&
            copied_summary.sampled_skips[0].mnemonic == "s_cbranch_scc1" &&
            copied_summary.sampled_skips[0].words.size() == 1,
        "patch plan summary should preserve skipped instruction samples");
  check(patch_report.hashed_edge_sites == 5,
        "patch plan summary should record hashed edge sites");
  check(patch_report.fixed_edge_sites == 7,
        "patch plan summary should record fixed edge sites");
  check(patch_report.fixed_slot_requests == 13,
        "patch plan summary should record fixed slot requests");
  check(patch_report.fixed_slots_reserved == 12,
        "patch plan summary should record fixed slots reserved");
  check(patch_report.fixed_slot_exhaustions == 3,
        "patch plan summary should record fixed slot exhaustions");
  check(patch_report.fixed_slot_collisions == 4,
        "patch plan summary should record fixed slot collisions");
  check(patch_report.inline_slot_requests == 11,
        "patch plan summary should record inline slot requests");
  check(patch_report.inline_slot_exhaustions == 2,
        "patch plan summary should record inline slot exhaustions");
  check(patch_report.branch_edges_degraded_to_fixed == 4,
        "patch plan summary should aggregate branch fallback edges");
  check(patch_report.fixed_counter_branch_edge_aggregate_fallback_used == 1 &&
            patch_report.fixed_counter_branch_edge_safety_fallback_used == 2 &&
            patch_report.fixed_counter_branch_edge_liveness_fallback_used == 1 &&
            patch_report.fixed_counter_branch_edge_placement_fallback_used == 0,
        "patch plan summary should aggregate fixed-counter fallback causes");
  check(patch_report.exec_empty_fixed_counter_edges == 1 &&
            copied_summary.exec_empty_fixed_counter_edges == 1,
        "patch plan summary should report EXEC-empty fixed fallback probes");
  check(patch_report.previous_bb_branch_edges_selected == 6,
        "patch plan summary should aggregate previous-BB branch edges");
  check(patch_report.previous_bb_branch_edge_trampolines_planned == 1 &&
            patch_report.previous_bb_branch_edge_trampoline_bytes == 28 &&
            patch_report.largest_previous_bb_branch_edge_trampoline_bytes == 28 &&
            patch_report.previous_bb_branch_planned_appended_edge_trampolines == 0 &&
            patch_report.previous_bb_branch_planned_local_edge_trampolines == 1 &&
            patch_report.previous_bb_branch_planned_appended_edge_trampoline_bytes == 0 &&
            patch_report.previous_bb_branch_planned_local_edge_trampoline_bytes == 28 &&
            patch_report.previous_bb_branch_afl_map_pressure_ppm == 183 &&
            patch_report.previous_bb_branch_trampoline_avg_bytes_x100 == 2800 &&
            patch_report.previous_bb_branch_appended_trampoline_ratio_ppm == 0 &&
            patch_report.previous_bb_branch_local_trampoline_ratio_ppm == 1000000 &&
            patch_report.previous_bb_branch_overhead_status ==
                "partially-degraded-to-fixed" &&
            copied_summary.previous_bb_branch_edge_trampolines_planned == 1 &&
            copied_summary.previous_bb_branch_edge_trampoline_bytes == 28 &&
            copied_summary.largest_previous_bb_branch_edge_trampoline_bytes == 28 &&
            copied_summary.previous_bb_branch_planned_appended_edge_trampolines == 0 &&
            copied_summary.previous_bb_branch_planned_local_edge_trampolines == 1 &&
            copied_summary.previous_bb_branch_planned_appended_edge_trampoline_bytes == 0 &&
            copied_summary.previous_bb_branch_planned_local_edge_trampoline_bytes == 28 &&
            copied_summary.previous_bb_branch_afl_map_pressure_ppm == 183 &&
            copied_summary.previous_bb_branch_trampoline_avg_bytes_x100 == 2800 &&
            copied_summary.previous_bb_branch_appended_trampoline_ratio_ppm == 0 &&
            copied_summary.previous_bb_branch_local_trampoline_ratio_ppm == 1000000 &&
            copied_summary.previous_bb_branch_overhead_status ==
                "partially-degraded-to-fixed",
        "patch plan summary should aggregate previous-BB branch trampoline footprint");
  check(patch_report.edge_sites_selected == 3,
        "patch plan summary should record selected edge sites");
  check(patch_report.local_text_cave_ranges == 2 &&
            patch_report.local_text_cave_bytes == 96 &&
            patch_report.largest_local_text_cave_bytes == 64,
        "patch plan summary should record local cave budget");
  check(patch_report.edge_patch_failures == 1 &&
            patch_report.branch_range_failures == 1 &&
            patch_report.sampled_failures.size() == 1,
        "patch plan summary should record edge patch failures");
  check(patch_report.sampled_selected_edges.size() == 3,
        "patch plan summary should sample planned edge sites");
  const EdgeSiteSelectionSample &selected = patch_report.sampled_selected_edges[0];
  check(selected.kernel_name == "kernel" && selected.kind == "cond-branch" &&
            selected.slot_policy == "fixed-counter" && selected.fixed_slot == 17 &&
            selected.scratch_spill && selected.vgpr_scratch_spill &&
            !selected.sgpr_scratch_spill &&
            selected.scratch_address_exec_source == "saved-exec" &&
            selected.placement == "appended-cave",
        "patch plan summary should preserve selected edge diagnostics");
  check(selected.state_sgpr == 100 && selected.saved_exec_sgpr == 102 &&
            selected.tmp0_sgpr == 104 && selected.tmp1_sgpr == 104 &&
            selected.workitem_vgpr == 120 && selected.tmp0_vgpr == 121 &&
            selected.tmp1_vgpr == 122 && selected.tmp2_vgpr == 123 &&
            selected.scratch_address_vgpr == 4 &&
            selected.scratch_spilled_vgprs.size() == 1 &&
            selected.scratch_spilled_vgprs[0] == 5 &&
            selected.scratch_spilled_sgprs.empty(),
        "patch plan summary should preserve selected edge register allocation");
  check(!selected.force_lane0_exec_for_fixed_counter &&
            patch_report.sampled_selected_edges[1].force_lane0_exec_for_fixed_counter,
        "patch plan summary should preserve selected EXEC-empty fixed fallback markers");

  InstrumentationPlan placement_fallback_selection;
  placement_fallback_selection.slot_policy_summary.hashed_edge_sites = 1;
  KernelEdgeSelectionSummary placement_fallback_summary;
  placement_fallback_summary.kernel_name = "kernel";
  placement_fallback_summary.previous_bb_branch_sites_selected = 1;
  placement_fallback_summary.fixed_counter_branch_edge_fallback_budget = 4;
  placement_fallback_summary.fixed_counter_branch_edge_fallback_used = 1;
  placement_fallback_summary.slot_policy_summary.hashed_edge_sites = 1;
  placement_fallback_selection.kernel_summaries.push_back(
      placement_fallback_summary);

  EdgeSite previous_bb_branch;
  previous_bb_branch.kernel_name = "kernel";
  previous_bb_branch.kind = EdgePatchKind::ConditionalBranchTerminator;
  previous_bb_branch.slot_policy = EdgeSlotPolicyKind::PreviousBbHash;
  previous_bb_branch.bb_id = 0x1234u;
  previous_bb_branch.fallthrough_bb_id =
      previous_bb_branch.bb_id + FixedEdgeSlotAllocator::fixed_slot_budget();
  check(previous_bb_branch_site(previous_bb_branch),
        "placement fallback should only target previous-BB branch sites");
  check(edge_count_for_site(previous_bb_branch) == 2,
        "conditional branch fallback should account for two logical edges");
  check(placement_failure_can_degrade_to_fixed("no branch-reachable local text cave"),
        "local cave placement failures should be fixed-counter fallback candidates");
  check(placement_fixed_fallback_has_budget(placement_fallback_selection,
                                            previous_bb_branch),
        "placement fallback should honor the existing fixed-counter budget");
  EdgeSite fixed_fallback =
      make_stable_fixed_counter_fallback_site(previous_bb_branch);
  check(fixed_fallback.slot_policy == EdgeSlotPolicyKind::FixedCounter &&
            fixed_fallback.fixed_slot != 0 &&
            fixed_fallback.fallthrough_slot != fixed_fallback.fixed_slot,
        "placement fallback should build a stable fixed-counter branch site");

  FixedEdgeSlotTracker placement_tracker;
  std::vector<EdgeSite> existing_fixed_sites = {fixed_fallback};
  prime_fixed_counter_placement_tracker(placement_tracker, existing_fixed_sites);
  const uint32_t fallback_collisions =
      record_fixed_counter_placement_slots(placement_tracker, fixed_fallback);
  check(fallback_collisions == 2,
        "placement fallback should report stable fixed-slot collisions");
  check(record_previous_bb_branch_placement_fallback(
            placement_fallback_selection, fixed_fallback, fallback_collisions,
            "no branch-reachable local text cave"),
        "placement fallback accounting should update the selected kernel");
  const KernelEdgeSelectionSummary &placement_fallback_result =
      placement_fallback_selection.kernel_summaries[0];
  check(placement_fallback_result.previous_bb_branch_sites_selected == 0 &&
            placement_fallback_result.previous_bb_branch_sites_degraded_to_fixed == 1 &&
            placement_fallback_result.branch_edges_degraded_to_fixed == 2 &&
            placement_fallback_result.fixed_counter_branch_edge_fallback_used == 3 &&
            placement_fallback_result.fixed_counter_branch_edge_placement_fallback_used == 2,
        "placement fallback should update fallback counters");
  check(placement_fallback_result.slot_policy_summary.hashed_edge_sites == 0 &&
            placement_fallback_result.slot_policy_summary.fixed_edge_sites == 1 &&
            placement_fallback_result.slot_policy_summary.fixed_slot_requests == 2 &&
            placement_fallback_result.slot_policy_summary.fixed_slots_reserved == 2 &&
            placement_fallback_result.slot_policy_summary.fixed_slot_collisions == 2,
        "placement fallback should update per-kernel slot summaries");
  check(placement_fallback_selection.slot_policy_summary.hashed_edge_sites == 0 &&
            placement_fallback_selection.slot_policy_summary.fixed_edge_sites == 1 &&
            placement_fallback_selection.slot_policy_summary.fixed_slot_collisions == 2,
        "placement fallback should update aggregate slot summaries");
  check(placement_fallback_result.degradation_reason_counts.size() == 1 &&
            placement_fallback_result.degradation_reason_counts[0].count == 2,
        "placement fallback should report a degradation reason");
  check(!record_previous_bb_branch_placement_fallback(
            placement_fallback_selection, fixed_fallback, fallback_collisions,
            "no branch-reachable local text cave"),
        "placement fallback should not exceed the fixed-counter fallback budget");

  DeviceElfPatchPlan many_edge_plan;
  many_edge_plan.edge_sites.resize(40);
  for (uint32_t i = 0; i < 40; ++i) {
    EdgeSite edge = sampled_edge;
    edge.patch_text_offset = 128 + i * 4;
    many_edge_plan.edge_trampolines.push_back(
        PlannedEdgeTrampoline{edge, EdgeTrampoline{},
                              EdgePatchResult{EdgeTrampolinePlacement::AppendedCave,
                                              2048 + i * 64}});
  }
  PatchDeviceElfReport capped_edge_report;
  record_patch_plan_summary(capped_edge_report, many_edge_plan);
  check(capped_edge_report.sampled_selected_edges.size() == 32,
        "normal patch reports should keep selected-edge samples bounded");
  PatchDeviceElfReport opaque_fresh_edge_report;
  opaque_fresh_edge_report.allow_opaque_fresh_registers = true;
  record_patch_plan_summary(opaque_fresh_edge_report, many_edge_plan);
  check(opaque_fresh_edge_report.sampled_selected_edges.size() == 40,
        "opaque fresh-growth minimizer reports should retain every selected edge");
  KernelEdgeSelectionSummary aggregate_cap_summary;
  aggregate_cap_summary.kernel_name = "kernel";
  aggregate_cap_summary.previous_bb_branch_site_over_budget = 1;
  many_edge_plan.edge_selection.kernel_summaries.push_back(aggregate_cap_summary);
  PatchDeviceElfReport aggregate_cap_edge_report;
  record_patch_plan_summary(aggregate_cap_edge_report, many_edge_plan);
  check(aggregate_cap_edge_report.sampled_selected_edges.size() == 40,
        "previous-BB aggregate cap reports should retain every selected edge");

  const std::array<uint32_t, 1> entry_prologue = {
      rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4)};
  std::optional<rocjitsu::KernelEntryProloguePlan> entry_plan =
      rocjitsu::plan_kernel_entry_prologue(/*cave_start=*/0, /*cave_body_size=*/0,
                                           /*entry_text_offset=*/256, entry_prologue,
                                           ROCJITSU_CODE_ARCH_RDNA4);
  check(entry_plan.has_value(), "entry prologue should be plannable");
  check(entry_plan->new_entry_text_offset % 256 == 0,
        "planned entry prologue should preserve launch alignment residue");
  check(entry_plan->cave_words.size() == entry_prologue.size() + 1,
        "planned entry prologue should include body and return branch");
  check(!rocjitsu::plan_kernel_entry_prologue(/*cave_start=*/200000,
                                             /*cave_body_size=*/0,
                                             /*entry_text_offset=*/0,
                                             entry_prologue,
                                             ROCJITSU_CODE_ARCH_RDNA4)
             .has_value(),
        "out-of-range entry prologue branch should fail during planning");

  std::vector<uint8_t> text_without_caves(64, 0xff);
  LocalTextCaveAllocator no_local_caves(text_without_caves);
  const LocalTextCaveSummary no_local_caves_before = no_local_caves.summary();
  EdgeSite out_of_range_edge;
  out_of_range_edge.kernel_name = "kernel";
  out_of_range_edge.kind = EdgePatchKind::BranchTerminator;
  out_of_range_edge.slot_policy = EdgeSlotPolicyKind::FixedCounter;
  out_of_range_edge.self_contained_probe = true;
  out_of_range_edge.patch_text_offset = 0;
  out_of_range_edge.return_text_offset = sizeof(uint32_t);
  out_of_range_edge.first_inst_size = sizeof(uint32_t);
  out_of_range_edge.fixed_slot = kFirstEdgeCounterSlot;
  const char *edge_failure = nullptr;
  std::optional<PlannedEdgeTrampoline> edge_plan =
      plan_edge_trampoline(out_of_range_edge, text_without_caves,
                           /*appended_cave_body_size=*/0,
                           /*cave_start=*/1ull << 40, no_local_caves,
                           ROCJITSU_CODE_ARCH_RDNA4,
                           /*state_pointer=*/0x1234567887654321ull, &edge_failure);
  check(!edge_plan.has_value(), "out-of-range cave plan should fail");
  check(std::string_view(edge_failure) == "no branch-reachable local text cave",
        "out-of-range cave plan should report stable local-cave failure");
  const LocalTextCaveSummary no_local_caves_after = no_local_caves.summary();
  check(no_local_caves_after.range_count == no_local_caves_before.range_count &&
            no_local_caves_after.total_bytes == no_local_caves_before.total_bytes,
        "failed cave planning should not consume local cave budget");

  for (const ProbeTarget &target : kProbeTargets) {
    const rj_code_arch_t arch = target.arch;
    const ProbeTarget *roundtrip = probe_target_for_elf_mach(target.elf_mach);
    check(roundtrip == &target, "probe target table lookup should be stable");
    check(probe_target_edge_support_reason(target)[0] != '\0',
          "probe target edge support reason should be reportable");
    if (probe_target_supports_previous_bb_edges(target)) {
      check(std::string_view(probe_target_edge_support_reason(target)) ==
                "previous-bb-edge-probes-supported",
            "previous-BB targets should report supported edge probes");
    } else {
      check(std::string_view(probe_target_edge_support_reason(target)) ==
                "missing-previous-bb-probe-helpers",
            "entry-only targets should report missing previous-BB helpers");
    }

    std::vector<std::vector<uint32_t>> probes;
    auto flagless_counter = rdna4_flagless_counter_probe_with_state_pointer(
        /*slot=*/0x123u, /*state_pointer=*/0x1234567887654321ull, arch);
    check(flagless_counter.has_value(), "flagless counter probe should be encodable");
    probes.push_back(*flagless_counter);

    Rdna4ProbeBuilder memory_ops(/*regs=*/{}, arch);
    const auto &regs = memory_ops.regs();
    memory_ops.global_load_b32(regs.tmp1_vgpr, regs.workitem_vgpr, regs.state_sgpr);
    memory_ops.global_store_b32(regs.workitem_vgpr, regs.tmp1_vgpr, regs.state_sgpr);
    memory_ops.global_atomic_add_u32(regs.workitem_vgpr, regs.tmp0_vgpr, regs.state_sgpr);
    memory_ops.s_wait_kmcnt();
    probes.push_back(memory_ops.take());

    if (probe_target_supports_previous_bb_edges(target)) {
      auto entry_counter = rdna4_counter_probe(kEntryCounterSlot, arch);
      check(entry_counter.has_value(), "entry counter probe should be encodable");
      probes.push_back(*entry_counter);
      probes.push_back(rdna4_edge_entry_probe(arch));
      probes.push_back(rdna4_previous_bb_edge_probe(/*bb_id=*/0x101u,
                                                    /*load_state_base=*/true,
                                                    /*state_pointer_kernarg_offset=*/0, arch));
      probes.push_back(rdna4_previous_bb_edge_probe_with_state_pointer(
          /*bb_id=*/0x101u, /*state_pointer=*/0x1234567887654321ull, arch));
    }

    for (size_t i = 0; i < probes.size(); ++i) {
      char name[32];
      snprintf(name, sizeof(name), "probe%zu", i);
      check_probe_decodes(arch, probes[i], name);
    }
  }

  return 0;
}
