// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/lds_race.h"
#include "rocjitsu/code/patch/consan/prepare.h"

#include "rocjitsu/code/basic_block.h"
#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/isa/instruction.h"

#include <algorithm>
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

Result reject(InputError &error, std::string message) {
  error = {.code = InputErrorCode::ResourceLimit, .text_offset = {}, .message = std::move(message)};
  return Result::failure();
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
  if (arch == ROCJITSU_CODE_ARCH_CDNA5 && (name == "s_barrier_signal" || name == "s_barrier_wait"))
    return Result::success();
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

Result qualify_lds_race_barrier(InputProgram &program, std::span<BasicBlock *const> blocks,
                                InputError &error) {
  std::vector<const BasicBlock *> barrier_blocks;
  uint32_t members = 0;
  for (BasicBlock *block : blocks) {
    std::vector<const Instruction *> instructions;
    for (const Instruction &instruction : block->instructions()) {
      instructions.push_back(&instruction);
      if (instruction.mnemonic() == "s_barrier_signal" ||
          instruction.mnemonic() == "s_barrier_wait")
        ++members;
    }
    for (size_t index = 0; index < instructions.size(); ++index) {
      if (instructions[index]->mnemonic() != "s_barrier_signal")
        continue;
      const uint32_t signal =
          cdna5::build_sop1(cdna5::kSBarrierSignalSop1, {.ssrc0 = cdna5::OPR_SSRC_NEG_INT_MIN})[0];
      const uint32_t wait =
          build_sopp_encoding(ROCJITSU_CODE_ARCH_CDNA5, cdna5::kSBarrierWaitSopp, 0xffff).value();
      const uint32_t drain =
          build_sopp_encoding(ROCJITSU_CODE_ARCH_CDNA5, cdna5::kSWaitDscntSopp, 0).value();
      size_t wait_index = index + 1;
      bool intervening_memory = false;
      while (wait_index < instructions.size() &&
             instructions[wait_index]->mnemonic() != "s_barrier_wait") {
        const std::string_view name = instructions[wait_index]->mnemonic();
        if (name.starts_with("ds_") || name.starts_with("global_") || name.starts_with("s_load") ||
            name.starts_with("s_barrier"))
          intervening_memory = true;
        ++wait_index;
      }
      // Require the original guest drain: DBI envelopes add their own waits,
      // which must not make a store/signal/wait group without s_wait_dscnt 0
      // qualify as guest synchronization.
      if (program.target->id != 1250 || program.wave_size != 32 || index == 0 ||
          wait_index == instructions.size() || intervening_memory ||
          instructions[index - 1]->size() != 4 || instructions[index]->size() != 4 ||
          instructions[wait_index]->size() != 4 ||
          instructions[index - 1]->raw_encoding()[0] != drain ||
          instructions[index]->raw_encoding()[0] != signal ||
          instructions[wait_index]->raw_encoding()[0] != wait)
        return reject_instruction(
            error, InputErrorCode::UnsupportedSynchronization,
            "requires original LDS drain/signal -1/wait -1 groups without intervening memory",
            instructions[index]->src_loc());
      barrier_blocks.push_back(block);
      program.barriers.push_back(
          BarrierFacts{.original_drain = instructions[index - 1]->src_loc(),
                       .original_signal = instructions[index]->src_loc(),
                       .original_wait = instructions[wait_index]->src_loc()});
    }
  }
  if (members == 0)
    return Result::success();
  if (members != 2 * program.barriers.size() || program.barriers.empty() ||
      program.barriers.size() > kSyncCapacity)
    return reject_instruction(error, InputErrorCode::UnsupportedSynchronization,
                              "incomplete or repeated workgroup barrier group");
  // In this closed forward CFG, deleting the group block must leave no path
  // from the hardware entry to an exit. This also rejects divergent bypasses;
  // scalar barriers themselves execute independently of the current EXEC.
  const auto entry =
      std::ranges::find(blocks, program.kernel.entry_text_offset, &BasicBlock::start_offset);
  if (entry == blocks.end())
    return reject_instruction(error, InputErrorCode::UnsupportedSynchronization,
                              "missing barrier entry");
  for (const BasicBlock *barrier_block : barrier_blocks) {
    std::vector<const BasicBlock *> pending{*entry};
    std::set<const BasicBlock *> visited;
    while (!pending.empty()) {
      const BasicBlock *block = pending.back();
      pending.pop_back();
      if (block == barrier_block || !visited.insert(block).second)
        continue;
      if (block->successors().empty())
        return reject_instruction(error, InputErrorCode::UnsupportedSynchronization,
                                  "workgroup barrier does not converge on every exit",
                                  block->start_offset());
      for (const BasicBlock *successor : block->successors())
        pending.push_back(successor);
    }
  }
  std::ranges::sort(program.barriers, {}, &BarrierFacts::original_signal);
  return Result::success();
}

FailureOr<uint32_t> lds_race_row_stride(size_t sites, InputError &error, bool barriers) {
  if (sites > ContractLimits{}.sites)
    return reject(error, "no bounded LDS report-row layout");
  const uint64_t stride = report_row_bytes(sites, barriers);
  if (stride > ContractLimits{}.report_bytes - kReportHeaderBytes)
    return reject(error, "no bounded LDS report-row layout");
  return static_cast<uint32_t>(stride);
}

Result validate_lds_race_accesses(const InputProgram &program, const PreparedArtifact &artifact,
                                  std::span<BasicBlock *const> blocks,
                                  std::span<const uint8_t> prepared_text, InputError &error) {
  std::set<uint64_t> barrier_offsets;
  std::set<uint64_t> access_offsets;
  for (BasicBlock *block : blocks)
    for (const Instruction &instruction : block->instructions()) {
      const std::string_view name = instruction.mnemonic();
      if (name == "s_barrier_signal" || name == "s_barrier_wait")
        barrier_offsets.insert(instruction.src_loc());
      if (name == "ds_load_b32" || name == "ds_store_b32" || name == "ds_read_b32" ||
          name == "ds_write_b32")
        access_offsets.insert(instruction.src_loc());
    }
  std::set<uint64_t> expected_barriers;
  for (const BarrierFacts &barrier : artifact.manifest.barriers) {
    expected_barriers.insert(barrier.prepared_signal);
    expected_barriers.insert(barrier.prepared_wait);
    if (std::memcmp(program.image.data() + program.text_file_offset + barrier.original_signal,
                    prepared_text.data() + barrier.prepared_signal, 4) != 0 ||
        std::memcmp(program.image.data() + program.text_file_offset + barrier.original_wait,
                    prepared_text.data() + barrier.prepared_wait, 4) != 0)
      return reject(error, "final reachable barrier operations differ from original group");
  }
  if (barrier_offsets != expected_barriers)
    return reject(error, "final reachable barrier multiplicity differs from original");
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
