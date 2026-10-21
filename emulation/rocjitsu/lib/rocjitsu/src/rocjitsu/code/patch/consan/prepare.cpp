// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/prepare.h"

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/basic_block.h"
#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/code/consan/image/artifact.h"
#include "rocjitsu/code/consan/image/metadata.h"
#include "rocjitsu/code/patch/code_object_patcher.h"
#include "rocjitsu/code/patch/kernel_text_layout.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/isa/target_registry.h"
#include "util/utf8.h"

#include <algorithm>
#include <cstring>
#include <map>

namespace rocjitsu::consan {
namespace {

namespace kd = rocjitsu::amdhsa;

Result reject(InputError &error, std::string message) {
  error = {.code = InputErrorCode::ResourceLimit, .text_offset = {}, .message = std::move(message)};
  return Result::failure();
}

bool bind_guest_branch(std::vector<uint8_t> &text, size_t branch_offset, uint64_t destination) {
  const std::optional<int16_t> displacement =
      compute_sopp_branch_simm16(branch_offset, destination);
  if (!displacement)
    return false;
  uint32_t word = 0;
  std::memcpy(&word, text.data() + branch_offset, sizeof(word));
  word = (word & 0xffff0000u) | static_cast<uint16_t>(*displacement);
  std::memcpy(text.data() + branch_offset, &word, sizeof(word));
  return true;
}

FailureOr<PreparedArtifact> lower(const InputProgram &program, InputError &error) {
  const FailureOr<LdsRaceProbePlan> planned = plan_lds_race_probes(program, error);
  if (planned.failed())
    return Result::failure();
  const LdsRaceProbePlan &plan = planned.value();
  const rj_code_arch_t arch = plan.arch;
  const uint32_t granule = descriptor_vgpr_granularity_for_wavefront(arch, program.wave_size);

  // Keep the original text intact and append a descriptor-visible new body.
  AmdGpuCodeObject original(program.image.data(), program.image.size());
  CodeObjectPatcher patcher(original);
  std::vector<uint8_t> text(patcher.text_bytes().begin(), patcher.text_bytes().end());
  append_nop_padding(
      text,
      padding_for_residue(text.size(), program.kernel.entry_text_offset, kKernelEntryAlignment),
      arch);
  const uint64_t entry = text.size();
  // Context delivery belongs to the shared entry protocol, once per entry.
  append_words(text, program.entry_plan.prologue.words);
  if (emit_lds_race_entry(text, plan, error).failed())
    return Result::failure();
  std::map<uint64_t, uint64_t> placements;
  std::vector<std::pair<size_t, uint64_t>> branches;
  std::vector<TextOffsetRelocation> relocations;
  PreparedArtifact artifact;
  Manifest &manifest = artifact.manifest;
  std::vector<NativeLdsAccess>::const_iterator access = program.accesses.begin();
  // Internal branches land before a target instruction's observation probe.
  // External entry relocation additionally includes the common bootstrap.
  for (const OriginalInstruction &instruction : program.instructions) {
    placements[instruction.offset] = text.size();
    relocations.push_back(
        {instruction.offset,
         instruction.offset == program.kernel.entry_text_offset ? entry : text.size()});
    if (access != program.accesses.end() && access->original_offset == instruction.offset) {
      const FailureOr<SiteFacts> site = emit_lds_race_access(text, plan, *access, error);
      if (site.failed())
        return Result::failure();
      manifest.sites.push_back(site.value());
      ++access;
    }
    if (instruction.terminates && emit_lds_race_exit(text, plan, error).failed())
      return Result::failure();
    if (instruction.branch_target)
      branches.emplace_back(text.size(), *instruction.branch_target);
    append_words(text, instruction.words);
  }
  // Resolve guest transfers only after every destination has its final offset.
  relocations.push_back({program.text_bytes, text.size()});
  for (const auto &[offset, target] : branches) {
    const std::map<uint64_t, uint64_t>::const_iterator found = placements.find(target);
    if (found == placements.end() || !bind_guest_branch(text, offset, found->second))
      return reject(error, "relocated guest branch is unavailable or out of range");
  }
  // Mutate ELF layout, then reparse offsets before updating resources/entry.
  if (!patcher.replace_text(text, relocations, {}, {}, true))
    return reject(error, "ELF text layout/relocation failed");
  artifact.image = std::move(patcher).emit();
  AmdGpuCodeObject moved(artifact.image.data(), artifact.image.size());
  if (!moved.is_valid() || moved.text_sections().size() != 1)
    return reject(error, "final ELF cannot be reparsed");
  const Section &section = *moved.text_sections().front();
  const std::vector<KernelDescriptorInfo> kernels = scan_kernel_descriptors(
      artifact.image, section.sectionOffset(), section.size(), section.sectionHeaderIndex());
  if (kernels.size() != 1)
    return reject(error, "final descriptor is unavailable");
  KernelDescriptorInfo kernel = kernels.front();
  const std::optional<KernargExtensionLayout> kernarg = make_kernarg_extension_layout(
      program.kernel.descriptor.kernarg_size, kContextPointerPayloads);
  if (!kernarg)
    return reject(error, "wrapper size overflow");
  if (validate_entry_kernarg_layout(program.entry_plan.prologue, *kernarg, error).failed())
    return Result::failure();
  std::string metadata_error;
  const MetadataResources resources{kernarg->wrapper_size, 8, plan.sgprs, plan.vgprs,
                                    program.maximum_workgroup_size};
  if (update_metadata(artifact.image, kernel, resources, &metadata_error).failed())
    return reject(error, metadata_error);
  kernel.descriptor.kernarg_size = resources.kernarg_bytes;
  RJ_AMDHSA_BITS_SET(kernel.descriptor.compute_pgm_rsrc1,
                     kd::COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT,
                     (plan.vgprs / granule - 1));
  if (arch == ROCJITSU_CODE_ARCH_CDNA3 || arch == ROCJITSU_CODE_ARCH_CDNA4) {
    // No accumulator instructions are admitted. Move the accumulator window
    // beyond the enlarged ordinary VGPR prefix, including the probe scratch.
    RJ_AMDHSA_BITS_SET(kernel.descriptor.compute_pgm_rsrc3,
                       kd::COMPUTE_PGM_RSRC3_GFX90A_ACCUM_OFFSET, (plan.vgprs / 4 - 1));
    RJ_AMDHSA_BITS_SET(kernel.descriptor.compute_pgm_rsrc1,
                       kd::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT,
                       ((plan.sgprs + kCdnaSpecialSgprTailReserve + 7) / 8 - 1));
  }
  // replace_text has already kept the descriptor coherent with the moved
  // symbol. Redirect from the reparsed descriptor, never from a stale offset.
  CodeObjectPatcher final_patcher(AmdGpuCodeObject(artifact.image.data(), artifact.image.size()));
  if (!final_patcher.patch_kernel_descriptor(
          kernel.descriptor_file_offset,
          {reinterpret_cast<const uint8_t *>(&kernel.descriptor), sizeof(kernel.descriptor)}) ||
      !final_patcher.redirect_kernel_entry(kernel.descriptor_file_offset, kernel.entry_text_offset,
                                           entry))
    return reject(error, "final descriptor/resource patch failed");
  artifact.image = std::move(final_patcher).emit();
  manifest.original = program.identity;
  manifest.prepared = {digest(artifact.image), artifact.image.size()};
  manifest.options = mvp_options_digest();
  manifest.target = program.target->id;
  manifest.wave_size = program.wave_size;
  manifest.code_object_abi = 5;
  manifest.original_text_bytes = program.text_bytes;
  manifest.prepared_text_bytes = text.size();
  manifest.entry = {.name = program.kernel.kernel_name,
                    .original_descriptor_offset = program.kernel.descriptor_file_offset,
                    .prepared_descriptor_offset = kernel.descriptor_file_offset,
                    .original_entry_offset = program.kernel.entry_text_offset,
                    .prepared_entry_offset = entry,
                    .original_kernarg_bytes = program.kernel.descriptor.kernarg_size,
                    .kernarg_alignment = 8,
                    .original_group_bytes = program.kernel.descriptor.group_segment_fixed_size,
                    .prepared_group_bytes = program.kernel.descriptor.group_segment_fixed_size,
                    .prepared_sgprs = plan.sgprs,
                    .prepared_vgprs = plan.vgprs,
                    .maximum_workgroup_size = program.maximum_workgroup_size};
  ContractError contract_error;
  if (validate_manifest(manifest, &contract_error).failed())
    return reject(error, "invalid lowered facts: " + contract_error.message);
  return artifact;
}

} // namespace

Result validate_entry_kernarg_layout(const DbiEntryPrologue &prologue,
                                     const KernargExtensionLayout &kernarg, InputError &error) {
  if (kernarg.payload_offsets.size() != 1 ||
      prologue.payload_byte_offset != kernarg.payload_offsets.front() ||
      prologue.original_kernarg_pointer_offset != kernarg.original_kernarg_pointer_offset) {
    error = {.code = InputErrorCode::UnsupportedEntry,
             .text_offset = {},
             .message = "entry prologue and ConSan kernarg wrapper offsets disagree"};
    return Result::failure();
  }
  return Result::success();
}

Result validate_prepared(std::span<const uint8_t> original, const PreparedArtifact &artifact,
                         InputError &error) {
  // Shared framing, executable mapping and resource checks precede decoding.
  ContractError contract_error;
  if (verify_artifact_images(artifact.manifest, original, artifact.image, &contract_error).failed())
    return reject(error, contract_error.message);
  const TargetProfile *target = target_profile(artifact.manifest.target);
  const rj_code_arch_t arch = target->arch;
  const FailureOr<InputProgram> program =
      build_input_program(original, *target, artifact.manifest.entry.name, error);
  if (program.failed())
    return Result::failure();
  // Re-lowering detects inconsistencies/tampering; it cannot establish that
  // the emitter's algorithm preserves state. Simulator/physical regressions
  // provide independent execution evidence for that property.
  const FailureOr<PreparedArtifact> canonical = lower(program.value(), error);
  if (canonical.failed())
    return Result::failure();
  if (canonical.value().image != artifact.image || canonical.value().manifest != artifact.manifest)
    return reject(error, "prepared bytes/facts do not match the canonical preservation plan");
  AmdGpuCodeObject object(artifact.image.data(), artifact.image.size());
  const Section &text = *object.text_sections().front();
  const std::vector<KernelDescriptorInfo> kernels = scan_kernel_descriptors(
      artifact.image, text.sectionOffset(), text.size(), text.sectionHeaderIndex());
  if (kernels.size() != 1 ||
      kernels.front().entry_text_offset != artifact.manifest.entry.prepared_entry_offset)
    return reject(error, "final entry/descriptor placement mismatch");
  std::string metadata_error;
  const FailureOr<MetadataResources> metadata =
      inspect_metadata(artifact.image, kernels.front(), &metadata_error);
  if (metadata.failed() || metadata.value().sgprs != artifact.manifest.entry.prepared_sgprs ||
      metadata.value().vgprs != artifact.manifest.entry.prepared_vgprs)
    return reject(error, "final AMDHSA resource mismatch: " + metadata_error);
  std::unique_ptr<Decoder> decoder =
      target->public_id == ROCJITSU_CODE_TARGET_INVALID
          ? Decoder::create(default_isa_target_registry(), arch)
          : Decoder::create(default_isa_target_registry(), target->public_id);
  if (decoder == nullptr)
    return reject(error, "final target decoder is unavailable");
  const std::array<uint64_t, 1> entries{kernels.front().entry_text_offset};
  const FailureOr<std::vector<std::unique_ptr<BasicBlock>>> blocks =
      BasicBlock::build_reachable(object, *decoder, arch, entries);
  if (blocks.failed())
    return reject(error, "final executable CFG cannot be decoded");
  std::vector<BasicBlock *> scope;
  for (const std::unique_ptr<BasicBlock> &block : blocks.value()) {
    if (block->has_implicit_terminator() ||
        block->successor_issue() != BasicBlock::SuccessorIssue::None)
      return reject(error, "final CFG has an invalid successor or implicit terminator");
    scope.push_back(block.get());
  }
  return validate_lds_race_accesses(
      program.value(), artifact, scope,
      std::span(artifact.image).subspan(text.sectionOffset(), text.size()), error);
}

FailureOr<PreparedArtifact> prepare(std::span<const uint8_t> original, const TargetProfile &target,
                                    std::string_view entry, InputError &error) {
  const FailureOr<InputProgram> program = build_input_program(original, target, entry, error);
  if (program.failed())
    return Result::failure();
  if (!util::is_valid_utf8(program.value().kernel.kernel_name)) {
    error = {.code = InputErrorCode::UnsupportedEntry,
             .text_offset = {},
             .message = "entry name is not valid UTF-8 for JSON output"};
    return Result::failure();
  }
  FailureOr<PreparedArtifact> artifact = lower(program.value(), error);
  if (artifact.failed() || validate_prepared(original, artifact.value(), error).failed())
    return Result::failure();
  return artifact;
}

} // namespace rocjitsu::consan
