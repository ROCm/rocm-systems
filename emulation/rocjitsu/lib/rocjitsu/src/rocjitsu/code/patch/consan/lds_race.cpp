// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/lds_race.h"
#include "rocjitsu/code/patch/consan/prepare.h"

#include "rocjitsu/code/basic_block.h"
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

bool admitted_instruction(const Instruction &instruction) {
  switch (instruction.amdgpu_operation()) {
  case amdgpu::Operation::S_LOAD_DWORD:
  case amdgpu::Operation::S_LOAD_DWORDX2:
  case amdgpu::Operation::S_MOV_B64:
  case amdgpu::Operation::S_AND_SAVEEXEC_B64:
  case amdgpu::Operation::S_AND_B64:
  case amdgpu::Operation::S_OR_B64:
  case amdgpu::Operation::S_DELAY_ALU:
  case amdgpu::Operation::GLOBAL_STORE_DWORD:
  case amdgpu::Operation::DS_READ_B32:
  case amdgpu::Operation::DS_WRITE_B32:
  case amdgpu::Operation::S_WAIT_KMCNT:
  case amdgpu::Operation::S_WAIT_STORECNT:
  case amdgpu::Operation::S_WAIT_DSCNT:
  case amdgpu::Operation::S_WAIT_LOADCNT_DSCNT:
  case amdgpu::Operation::S_WAIT_LOADCNT:
  case amdgpu::Operation::S_WAIT_XCNT:
  case amdgpu::Operation::S_LOAD_B32:
  case amdgpu::Operation::S_LOAD_B64:
  case amdgpu::Operation::S_WAITCNT:
  case amdgpu::Operation::S_WAITCNT_VSCNT:
  case amdgpu::Operation::S_MOV_B32:
  case amdgpu::Operation::S_AND_SAVEEXEC_B32:
  case amdgpu::Operation::S_AND_B32:
  case amdgpu::Operation::S_OR_B32:
  case amdgpu::Operation::S_CMP_EQ_U32:
  case amdgpu::Operation::S_BRANCH:
  case amdgpu::Operation::S_CBRANCH_SCC0:
  case amdgpu::Operation::S_CBRANCH_SCC1:
  case amdgpu::Operation::S_CBRANCH_VCCZ:
  case amdgpu::Operation::S_CBRANCH_VCCNZ:
  case amdgpu::Operation::S_CBRANCH_EXECZ:
  case amdgpu::Operation::S_CBRANCH_EXECNZ:
  case amdgpu::Operation::S_NOP:
  case amdgpu::Operation::S_ENDPGM:
  case amdgpu::Operation::GLOBAL_STORE_B32:
  case amdgpu::Operation::DS_STORE_B32:
  case amdgpu::Operation::DS_LOAD_B32:
    return true;
  case amdgpu::Operation::V_CMPX_EQ_U32:
  case amdgpu::Operation::V_ADD_U32:
  case amdgpu::Operation::V_NOP:
  case amdgpu::Operation::V_MOV_B32:
  case amdgpu::Operation::V_LSHRREV_B32:
  case amdgpu::Operation::V_LSHLREV_B32:
  case amdgpu::Operation::V_ADD_NC_U32:
  case amdgpu::Operation::V_AND_B32:
  case amdgpu::Operation::V_CMP_EQ_U32:
  case amdgpu::Operation::V_CMP_GT_U32:
    break;
  default:
    return false;
  }

  // The qualified vector forms use VOP1/VOP2/VOPC rather than VOP3. Their
  // first word has bit 31 clear; literals and DPP extensions do not change it.
  // SDWA has the same base format but was never part of this admission set.
  if (instruction.raw_encoding() == nullptr || instruction.size() < 4 ||
      (instruction.raw_encoding()[0] & (uint32_t{1} << 31)) != 0)
    return false;
  amdgpu::RegisterModifiers modifiers;
  instruction.amdgpu_register_modifiers(modifiers);
  return !modifiers.sdwa;
}

bool is_qualified_lds_word_access(const Instruction &instruction) {
  switch (instruction.amdgpu_operation()) {
  case amdgpu::Operation::DS_LOAD_B32:
  case amdgpu::Operation::DS_STORE_B32:
  case amdgpu::Operation::DS_READ_B32:
  case amdgpu::Operation::DS_WRITE_B32:
    return true;
  default:
    return false;
  }
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
  const amdgpu::Operation operation = instruction.amdgpu_operation();
  const amdgpu::MemoryAddressing addressing = instruction.amdgpu_memory_addressing();
  const uint64_t offset = instruction.src_loc();
  if ((instruction.flags() & BARRIER) != 0 ||
      instruction.amdgpu_memory_kind() == amdgpu::MemoryKind::Atomic ||
      operation == amdgpu::Operation::S_SENDMSG || operation == amdgpu::Operation::S_SENDMSGHALT ||
      operation == amdgpu::Operation::S_SENDMSG_RTN_B32 ||
      operation == amdgpu::Operation::S_SENDMSG_RTN_B64)
    return reject_instruction(error, InputErrorCode::UnsupportedSynchronization,
                              "unmodeled synchronization: " + std::string(name), offset);
  if ((addressing == amdgpu::MemoryAddressing::Lds || addressing == amdgpu::MemoryAddressing::Gds ||
       addressing == amdgpu::MemoryAddressing::DirectLds) &&
      !is_qualified_lds_word_access(instruction))
    return reject_instruction(error, InputErrorCode::UnsupportedAccess,
                              "unsupported LDS form: " + std::string(name), offset);
  if (!admitted_instruction(instruction))
    return reject_instruction(error, InputErrorCode::UnsupportedInstruction,
                              "instruction outside MVP: " + std::string(name), offset);
  if (is_qualified_lds_word_access(instruction)) {
    // CDNA3/4 place GDS at bit 16; RDNA and CDNA5 use bit 17.
    const uint32_t gds_bit =
        (arch == ROCJITSU_CODE_ARCH_CDNA3 || arch == ROCJITSU_CODE_ARCH_CDNA4) ? 16 : 17;
    if (instruction.size() != 8 || (instruction.raw_encoding()[0] & (1u << gds_bit)) != 0)
      return reject_instruction(error, InputErrorCode::UnsupportedAccess, "non-native LDS form",
                                offset);
    const bool read =
        operation == amdgpu::Operation::DS_LOAD_B32 || operation == amdgpu::Operation::DS_READ_B32;
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

FailureOr<uint32_t> lds_race_row_stride(size_t sites, InputError &error) {
  if (sites > ContractLimits{}.sites)
    return reject(error, "no bounded LDS report-row layout");
  const uint64_t stride =
      kRowHeaderBytes + uint64_t{sites} * kAddressGroupsPerSite * kAccessRecordBytes;
  if (stride > ContractLimits{}.report_bytes - kReportHeaderBytes)
    return reject(error, "no bounded LDS report-row layout");
  return static_cast<uint32_t>(stride);
}

Result validate_lds_race_accesses(const InputProgram &program, const PreparedArtifact &artifact,
                                  std::span<BasicBlock *const> blocks,
                                  std::span<const uint8_t> prepared_text, InputError &error) {
  std::set<uint64_t> access_offsets;
  for (BasicBlock *block : blocks)
    for (const Instruction &instruction : block->instructions()) {
      if (is_qualified_lds_word_access(instruction))
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
