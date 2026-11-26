// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/prepare.h"
#include "rocjitsu/isa/arch/amdgpu/generated/shared/isa_properties.h"
#include "rocjitsu/isa/isa_family.h"

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/basic_block.h"
#include "rocjitsu/code/consan/image/artifact.h"
#include "rocjitsu/code/consan/image/metadata.h"
#include "rocjitsu/code/patch/consan/probes.h"
#include "rocjitsu/code/patch/instrumentor.h"
#include "rocjitsu/code/scoped_cfg_edges.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/isa/target_registry.h"
#include "util/diagnostic.h"
#include "util/utf8.h"

#include <algorithm>
#include <unordered_set>

namespace rocjitsu::consan {
namespace {

uint32_t workgroup_id_scalar(const KernelDescriptorInfo &kernel, const TargetProfile &target) {
  const bool enabled = RJ_AMDHSA_BITS_GET(kernel.descriptor.compute_pgm_rsrc2,
                                          amdhsa::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_X);
  if (!enabled) {
    return UINT32_MAX;
  }
  if (isa_properties(target.arch).uses_ttmp_workgroup_ids) {
    return kWorkgroupIdTtmp9Selector;
  }
  return kernel_descriptor_user_sgpr_count(target.arch, kernel.descriptor);
}

uint32_t workgroup_id_scalar(const InputProgram &program) {
  return workgroup_id_scalar(program.kernel, *program.target);
}

ScopeKind program_scope(const InputProgram &program) {
  if (program.scope == ScopeKind::ExecutionTrace)
    return ScopeKind::ExecutionTrace;
  return program.barriers.empty() ? ScopeKind::NativeLdsNoSynchronization
                                  : (program.repeated ? ScopeKind::NativeLdsWorkgroupBarrierLoop
                                                      : ScopeKind::NativeLdsWorkgroupBarrier);
}

Result reject(InputError &error, std::string message) {
  error = {.code = InputErrorCode::ResourceLimit, .text_offset = {}, .message = std::move(message)};
  return Result::failure();
}

Result finish_manifest(const InputProgram &program, std::span<const uint8_t> image,
                       Manifest &manifest, uint32_t sgprs, uint32_t vgprs, InputError &error) {
  AmdGpuCodeObject object(image.data(), image.size());
  if (!object.is_valid() || object.text_sections().size() != 1) {
    return reject(error, "final instrumented text cannot be reparsed");
  }
  const Section &section = *object.text_sections().front();
  const std::vector<KernelDescriptorInfo> kernels =
      scan_kernel_descriptors(image, section.sectionOffset(), section.size());
  const std::vector<KernelDescriptorInfo>::const_iterator found =
      std::ranges::find(kernels, program.kernel.kernel_name, &KernelDescriptorInfo::kernel_name);
  if (found == kernels.end()) {
    return reject(error, "final instrumented descriptor is unavailable");
  }
  const KernelDescriptorInfo &kernel = *found;
  manifest.original = program.identity;
  manifest.prepared = {digest(image), image.size()};
  manifest.scope = program_scope(program);
  manifest.options = profile_options_digest(manifest.scope);
  manifest.target = program.target->id;
  manifest.wave_size = program.wave_size;
  manifest.code_object_abi = 5;
  manifest.original_text_bytes = program.text_bytes;
  manifest.prepared_text_bytes = section.size();
  manifest.entry = {.name = program.kernel.kernel_name,
                    .original_descriptor_offset = program.kernel.descriptor_file_offset,
                    .prepared_descriptor_offset = kernel.descriptor_file_offset,
                    .original_entry_offset = program.kernel.entry_text_offset,
                    .prepared_entry_offset = kernel.entry_text_offset,
                    .original_kernarg_bytes = program.kernel.descriptor.kernarg_size,
                    .kernarg_alignment = program.kernarg_alignment,
                    .original_group_bytes = program.kernel.descriptor.group_segment_fixed_size,
                    .prepared_group_bytes = program.kernel.descriptor.group_segment_fixed_size,
                    .prepared_sgprs = sgprs,
                    .prepared_vgprs = vgprs,
                    .maximum_workgroup_size = program.maximum_workgroup_size,
                    .workgroup_id_scalar = workgroup_id_scalar(program),
                    .required_workgroup_size = program.required_workgroup_size};
  util::StringDiagnostic diagnostic;
  if (validate_manifest(manifest, diagnostic.emitter()).failed()) {
    return reject(error, "invalid lowered facts: " + diagnostic.message());
  }
  return Result::success();
}

void add_program_points(Instrumentor &instrumentor, const InputProgram &program,
                        const AmdGpuCodeObject &probes, uint32_t stride) {
  const uint32_t workgroup_scalar = workgroup_id_scalar(program);
  ProbeArgValue workgroup_arg = probe_arg_imm(0);
  if (workgroup_scalar == kWorkgroupIdTtmp9Selector) {
    workgroup_arg = {.source = ProbeArgSource::EntryTtmp, .register_index = 9};
  } else if (workgroup_scalar != UINT32_MAX) {
    workgroup_arg = {.source = ProbeArgSource::EntrySgpr,
                     .register_index = static_cast<uint16_t>(workgroup_scalar)};
  }
  const std::vector<ProbeArgValue> context_args{
      {ProbeArgSource::LogBufferPtrLo},
      {ProbeArgSource::LogBufferPtrHi},
      {.source = ProbeArgSource::EntryVgprLane0, .register_index = 0},
      probe_arg_imm(stride),
      workgroup_arg};
  const bool trace = program.scope == ScopeKind::ExecutionTrace;
  auto add = [&](uint64_t offset, InstrumentationKind kind, const char *symbol,
                 std::vector<ProbeArgValue> args) {
    InstrumentationPoint point;
    point.anchor_offset = offset;
    point.kind = kind;
    point.probe_obj = &probes;
    point.probe_symbol = symbol;
    point.probe_args = std::move(args);
    // Lane zero publishes even at empty guest EXEC. A full-wave callback also
    // preserves every address needed by CDNA5 transpose reads. Each access
    // callback receives the original EXEC separately from its own execution mask.
    point.force_full_exec = true;
    instrumentor.add_point(point);
  };
  add(program.kernel.entry_text_offset, InstrumentationKind::KernelEntry,
      trace ? "rj_consan_trace_entry" : "rj_consan_entry", context_args);
  for (const NativeLdsAccess &access : program.accesses) {
    std::vector<ProbeArgValue> args = context_args;
    args.push_back(probe_arg_imm(access.site));
    args.push_back(probe_arg_vgpr(access.address_vgpr));
    args.push_back(probe_arg_imm(access.immediate_bytes));
    args.push_back({ProbeArgSource::AnchorExecLo});
    args.push_back(program.wave_size == 64 ? ProbeArgValue{ProbeArgSource::AnchorExecHi}
                                           : probe_arg_imm(0));
    args.push_back(probe_arg_imm(access.width));
    args.push_back(probe_arg_imm(program.repeated));
    const char *symbol =
        is_transpose(access.lds_ranges.shape) ? "rj_consan_transpose" : "rj_consan_access";
    add(access.original_offset, InstrumentationKind::BeforeInst, symbol, std::move(args));
  }
  // Advance phase only after the unchanged guest wait completes. Recording at
  // the signal or before the wait could certify ordering before synchronization.
  for (uint32_t group = 0; group < program.barriers.size(); ++group) {
    std::vector<ProbeArgValue> args = context_args;
    args.push_back(probe_arg_imm(group));
    args.push_back(probe_arg_imm(static_cast<uint32_t>(sync_offset(program.accesses.size()))));
    args.push_back(probe_arg_imm(static_cast<uint32_t>(program.barriers.size())));
    add(program.barriers[group].original_wait, InstrumentationKind::AfterInst, "rj_consan_barrier",
        std::move(args));
  }
  for (const uint64_t offset : program.exit_offsets) {
    add(offset, InstrumentationKind::KernelExit, trace ? "rj_consan_trace_exit" : "rj_consan_exit",
        context_args);
  }
}

Result append_program_facts(const InputProgram &program, const std::vector<InstrumentedSite> &sites,
                            Manifest &manifest, InputError &error) {
  if (program.scope == ScopeKind::ExecutionTrace) {
    const auto add_boundary = [&](InstrumentationKind kind, uint64_t offset) -> bool {
      const std::vector<InstrumentedSite>::const_iterator site =
          std::ranges::find_if(sites, [&](const InstrumentedSite &candidate) {
            return candidate.kind == kind && candidate.original_offset == offset;
          });
      if (site == sites.end())
        return false;
      manifest.boundaries.push_back({.kind = kind == InstrumentationKind::KernelEntry
                                                 ? BoundaryKind::Entry
                                                 : BoundaryKind::Exit,
                                     .original_offset = offset,
                                     .prepared_guest_offset = site->guest_offset,
                                     .prepared_guest_size = site->guest_size,
                                     .probe_offset = site->instrumentation_offset,
                                     .probe_size = site->instrumentation_size});
      return true;
    };
    if (!add_boundary(InstrumentationKind::KernelEntry, program.kernel.entry_text_offset))
      return reject(error, "SBI omitted the entry-owned trace boundary");
    for (const uint64_t offset : program.exit_offsets)
      if (!add_boundary(InstrumentationKind::KernelExit, offset))
        return reject(error, "SBI omitted an entry-owned trace exit");
    return Result::success();
  }
  for (const NativeLdsAccess &access : program.accesses) {
    const std::vector<InstrumentedSite>::const_iterator mapping =
        std::ranges::find_if(sites, [&](const auto &site) {
          return site.kind == InstrumentationKind::BeforeInst &&
                 site.original_offset == access.original_offset;
        });
    if (mapping == sites.end() || mapping->guest_size != 8) {
      return reject(error, "callable DBI lowering omitted an LDS site mapping");
    }
    manifest.sites.push_back({.id = access.site,
                              .analysis = AnalysisKind::LdsRaceDetection,
                              .access = access.kind,
                              .width = access.width,
                              .lds_ranges = access.lds_ranges,
                              .original_offset = access.original_offset,
                              .original_size = 8,
                              .prepared_guest_offset = mapping->guest_offset,
                              .prepared_guest_size = mapping->guest_size,
                              .probe_offset = mapping->instrumentation_offset,
                              .probe_size = mapping->instrumentation_size,
                              .first_slot = access.site * kAddressGroupsPerSite});
  }
  for (const BarrierFacts &barrier : program.barriers) {
    const std::vector<InstrumentedSite>::const_iterator mapping =
        std::ranges::find_if(sites, [&](const auto &site) {
          return site.kind == InstrumentationKind::AfterInst &&
                 site.original_offset == barrier.original_wait;
        });
    if (mapping == sites.end() || mapping->guest_size != 4) {
      return reject(error, "callable DBI lowering omitted a barrier completion mapping");
    }
    BarrierFacts facts = barrier;
    // Split signals stay in original text. A legacy barrier both signals and
    // waits, so its two facts refer to the same relocated guest instruction.
    facts.prepared_signal = barrier.original_signal == barrier.original_wait
                                ? mapping->guest_offset
                                : barrier.original_signal;
    facts.prepared_wait = mapping->guest_offset;
    facts.completion_probe = mapping->instrumentation_offset;
    facts.probe_size = mapping->instrumentation_size;
    manifest.barriers.push_back(facts);
  }
  return Result::success();
}

FailureOr<PreparedArtifact> lower(const InputProgram &program, InputError &error) {
  const FailureOr<uint32_t> stride =
      lds_race_row_stride(program.accesses.size(), error, !program.barriers.empty());
  if (stride.failed()) {
    return Result::failure();
  }
  const std::span<const uint8_t> probe_image =
      lds_probe_image(program.target->id, program.wave_size);
  if (probe_image.empty()) {
    return reject(error, "no compiled callable probe for the selected target/wave mode");
  }
  AmdGpuCodeObject object(program.image.data(), program.image.size());
  AmdGpuCodeObject probes(probe_image.data(), probe_image.size());
  // Preserve overwritten guest registers even when liveness marks them dead.
  // This conservative client policy reduces reliance on liveness, at the cost
  // of extra backups, register growth, and trampoline bytes.
  const InstrumentationOptions options{
      .kernarg_binding = DbiKernargBinding::CallerSuppliedWrapper,
      .preservation = GuestStatePreservation::DeclaredRegisters,
      .register_growth = OriginalRegisterCounts{program.original_sgprs, program.original_vgprs}};
  Instrumentor instrumentor(object, program.target->arch, options);
  add_program_points(instrumentor, program, probes, stride.value());
  InstrumentedCodeObject output = instrumentor.patch();
  if (!output.errors.empty()) {
    return reject(error, "callable DBI lowering: " + output.errors.front());
  }
  PreparedArtifact artifact;
  artifact.image = std::move(output.elf_bytes);
  if (append_program_facts(program, output.sites, artifact.manifest, error).failed()) {
    return Result::failure();
  }
  if (finish_manifest(program, artifact.image, artifact.manifest, output.resources.sgprs,
                      output.resources.vgprs, error)
          .failed()) {
    return Result::failure();
  }
  return artifact;
}

Result validate_image_entry_cfg(const InputProgram &program, const Manifest &manifest,
                                std::span<const uint8_t> image, InputError &error) {
  AmdGpuCodeObject object(image.data(), image.size());
  const Section &text = *object.text_sections().front();
  std::unique_ptr<Decoder> decoder =
      Decoder::create(default_isa_target_registry(), program.target->public_id);
  if (decoder == nullptr) {
    return reject(error, "final target decoder is unavailable");
  }
  const std::array<uint64_t, 1> entries = {manifest.entry.prepared_entry_offset};
  const FailureOr<std::vector<std::unique_ptr<BasicBlock>>> blocks =
      BasicBlock::build_reachable(object, *decoder, program.target->arch, entries);
  if (blocks.failed()) {
    return reject(error, "final executable CFG cannot be decoded");
  }
  std::vector<BasicBlock *> scope;
  for (const std::unique_ptr<BasicBlock> &block : blocks.value()) {
    scope.push_back(block.get());
  }
  const std::span<const uint8_t> prepared_text = image.subspan(text.sectionOffset(), text.size());
  const std::unordered_set<uint64_t> returns = scoped_call_return_offsets(scope, prepared_text);
  const std::unordered_set<BasicBlock *> allowed(scope.begin(), scope.end());
  for (const std::unique_ptr<BasicBlock> &block : blocks.value()) {
    const Instruction *terminator = block->terminator();
    const bool call_return =
        terminator && returns.contains(terminator->src_loc()) &&
        block->successor_issue() == BasicBlock::SuccessorIssue::IndirectControlFlow;
    const std::vector<BasicBlock::CallEdge> &calls = block->call_edges();
    const bool resolved_call =
        terminator && (terminator->flags() & INDIRECT_CALL) && calls.size() == 1 &&
        calls.front().source_call_offset == terminator->src_loc() &&
        allowed.contains(calls.front().callee) && allowed.contains(calls.front().continuation) &&
        block->successor_issue() == BasicBlock::SuccessorIssue::IndirectControlFlow;
    if (block->has_implicit_terminator() ||
        (block->successor_issue() != BasicBlock::SuccessorIssue::None && !call_return &&
         !resolved_call)) {
      return reject(error, "final CFG has an invalid successor or implicit terminator");
    }
  }
  if (program.scope == ScopeKind::ExecutionTrace) {
    if (manifest.boundaries.size() != program.exit_offsets.size() + 1)
      return reject(error, "trace boundary attribution differs from original exits");
    for (size_t index = 1; index < manifest.boundaries.size(); ++index) {
      const BoundaryFacts &boundary = manifest.boundaries[index];
      if (boundary.original_offset != program.exit_offsets[index - 1] ||
          boundary.prepared_guest_size != sizeof(uint32_t) ||
          boundary.prepared_guest_offset >= prepared_text.size() ||
          prepared_text.size() - boundary.prepared_guest_offset < sizeof(uint32_t))
        return reject(error, "trace exit mapping differs from original program");
      uint32_t terminal = 0;
      std::memcpy(&terminal, prepared_text.data() + boundary.prepared_guest_offset,
                  sizeof(terminal));
      const DecodeResult decoded = decoder->decode_window(std::span(&terminal, 1));
      if (!decoded.succeeded() || decoded.value()->mnemonic() != "s_endpgm")
        return reject(error, "trace mapped exit is not the original kernel terminator");
    }
    return Result::success();
  }
  return validate_lds_race_accesses(program, manifest, scope, prepared_text, error);
}

FailureOr<PreparedImageArtifact> lower_image(std::span<const uint8_t> original,
                                             const TargetProfile &target,
                                             std::span<const std::string_view> entries,
                                             InputError &error, ScopeKind requested_scope) {
  const InputLimits limits;
  // Each selected program owns a copy of the image. Divide to bound their total
  // bytes without overflowing a size * count product; the empty check must stay
  // before the division so short-circuit evaluation prevents division by zero.
  if (entries.empty() || entries.size() > kImageEntryLimit ||
      original.size() > limits.image_bytes / entries.size()) {
    return reject(error, "selected input copies exceed bounded image analysis budget");
  }
  std::unordered_set<std::string_view> names;
  std::vector<InputProgram> programs;
  for (const std::string_view name : entries) {
    if (!util::is_valid_utf8(name)) {
      error = {.code = InputErrorCode::UnsupportedEntry,
               .text_offset = {},
               .message = "entry name is not valid UTF-8 for JSON output"};
      return Result::failure();
    }
    if (!names.insert(name).second) {
      error = {.code = InputErrorCode::UnsupportedEntry,
               .text_offset = {},
               .message = "duplicate selected entry name"};
      return Result::failure();
    }
    FailureOr<InputProgram> program =
        build_image_entry_program(original, target, name, error, {}, requested_scope);
    if (program.failed()) {
      return Result::failure();
    }
    programs.push_back(std::move(program.value()));
  }
  std::ranges::sort(programs, {}, [](const InputProgram &program) {
    return program.kernel.descriptor_file_offset;
  });
  const InputProgram &first = programs.front();
  const ScopeKind scope = program_scope(first);
  for (const InputProgram &program : programs) {
    if (program.wave_size != first.wave_size) {
      return reject(error, "selected entries have inconsistent wave modes");
    }
    if (program_scope(program) != scope) {
      error = {.code = InputErrorCode::UnsupportedEntry,
               .text_offset = {},
               .message = "selected entries have inconsistent analysis scopes"};
      return Result::failure();
    }
  }
  const std::span<const uint8_t> probe_image = lds_probe_image(first.target->id, first.wave_size);
  if (probe_image.empty()) {
    return reject(error, "no compiled callable probe for selected target/wave mode");
  }
  AmdGpuCodeObject object(original.data(), original.size());
  AmdGpuCodeObject probes(probe_image.data(), probe_image.size());
  std::vector<EntryInstrumentationOptions> requests;
  for (const InputProgram &program : programs) {
    requests.push_back({.descriptor_file_offset = program.kernel.descriptor_file_offset,
                        .kernarg_binding = DbiKernargBinding::CallerSuppliedWrapper,
                        .preservation = GuestStatePreservation::DeclaredRegisters,
                        .register_growth = OriginalRegisterCounts{
                            .sgprs = program.original_sgprs, .vgprs = program.original_vgprs}});
  }
  const InstrumentationOptions options = {.kernarg_binding = DbiKernargBinding::None,
                                          .preservation = GuestStatePreservation::LiveRegisters,
                                          .register_growth = std::nullopt,
                                          .entries = std::move(requests)};
  Instrumentor instrumentor(object, first.target->arch, options);
  for (const InputProgram &program : programs) {
    const FailureOr<uint32_t> stride =
        program.scope == ScopeKind::ExecutionTrace
            ? FailureOr<uint32_t>(kRowHeaderBytes)
            : lds_race_row_stride(program.accesses.size(), error, !program.barriers.empty());
    if (stride.failed()) {
      return Result::failure();
    }
    add_program_points(instrumentor, program, probes, stride.value());
  }
  InstrumentedCodeObject output = instrumentor.patch();
  if (!output.errors.empty() || output.entries.size() != programs.size()) {
    return reject(error, "whole-image callable DBI lowering: " +
                             (output.errors.empty() ? std::string("missing entry attribution")
                                                    : output.errors.front()));
  }
  PreparedImageArtifact artifact;
  artifact.image = std::move(output.elf_bytes);
  AmdGpuCodeObject prepared(artifact.image.data(), artifact.image.size());
  if (!prepared.is_valid() || prepared.text_sections().size() != 1) {
    return reject(error, "final whole image cannot be reparsed");
  }
  const Section &original_text = *object.text_sections().front();
  const Section &prepared_text = *prepared.text_sections().front();
  const std::vector<KernelDescriptorInfo> original_kernels =
      scan_kernel_descriptors(original, original_text.sectionOffset(), original_text.size(),
                              original_text.sectionHeaderIndex());
  const std::vector<KernelDescriptorInfo> prepared_kernels =
      scan_kernel_descriptors(artifact.image, prepared_text.sectionOffset(), prepared_text.size(),
                              prepared_text.sectionHeaderIndex());
  if (original_kernels.size() != prepared_kernels.size()) {
    return reject(error, "whole-image entry inventory changed during lowering");
  }
  ImageFacts &facts = artifact.facts;
  facts.original = first.identity;
  facts.prepared = {.sha256 = digest(artifact.image), .bytes = artifact.image.size()};
  facts.target = first.target->id;
  facts.wave_size = first.wave_size;
  facts.code_object_abi = 5;
  facts.original_text_bytes = original_text.size();
  facts.prepared_text_bytes = prepared_text.size();
  std::vector<EntryKey> selected_keys;
  for (const KernelDescriptorInfo &kernel : original_kernels) {
    // Native names also appear in the image-level JSON disposition table.
    if (!util::is_valid_utf8(kernel.kernel_name)) {
      error = {.code = InputErrorCode::UnsupportedEntry,
               .text_offset = {},
               .message = "entry name is not valid UTF-8 for JSON output"};
      return Result::failure();
    }
    const std::vector<InputProgram>::const_iterator selected =
        std::ranges::find_if(programs, [&](const InputProgram &program) {
          return program.kernel.descriptor_file_offset == kernel.descriptor_file_offset;
        });
    std::string metadata_error;
    const FailureOr<MetadataResources> metadata = inspect_metadata(
        original, kernel, &metadata_error,
        selected == programs.end() ? MetadataPolicy::Native : MetadataPolicy::Selected);
    if (metadata.failed()) {
      return reject(error, metadata_error);
    }
    const std::vector<KernelDescriptorInfo>::const_iterator final_kernel =
        std::ranges::find(prepared_kernels, kernel.kernel_name, &KernelDescriptorInfo::kernel_name);
    if (final_kernel == prepared_kernels.end()) {
      return reject(error, "prepared image lost an original kernel");
    }
    ImageEntryFacts row;
    row.key = kernel.descriptor_file_offset;
    row.original_sgprs = metadata.value().sgprs;
    row.original_vgprs = metadata.value().vgprs;
    if (selected != programs.end()) {
      row.disposition = EntryDisposition::Selected;
      const std::vector<InstrumentedEntry>::const_iterator mapping = std::ranges::find(
          output.entries, row.key, &InstrumentedEntry::original_descriptor_offset);
      if (mapping == output.entries.end() ||
          mapping->prepared_descriptor_offset != final_kernel->descriptor_file_offset ||
          mapping->original_entry_offset != kernel.entry_text_offset ||
          mapping->prepared_entry_offset != final_kernel->entry_text_offset) {
        return reject(error, "SBI entry attribution differs from final image");
      }
      Manifest manifest;
      if (append_program_facts(*selected, mapping->sites, manifest, error).failed()) {
        return Result::failure();
      }
      if (finish_manifest(*selected, artifact.image, manifest, mapping->resources.sgprs,
                          mapping->resources.vgprs, error)
              .failed()) {
        return Result::failure();
      }
      row.binding = manifest.entry;
      row.selected = std::move(manifest);
      selected_keys.push_back(row.key);
    } else {
      row.binding = {.name = kernel.kernel_name,
                     .original_descriptor_offset = row.key,
                     .prepared_descriptor_offset = final_kernel->descriptor_file_offset,
                     .original_entry_offset = kernel.entry_text_offset,
                     .prepared_entry_offset = final_kernel->entry_text_offset,
                     .original_kernarg_bytes = kernel.descriptor.kernarg_size,
                     .kernarg_alignment = metadata.value().kernarg_alignment,
                     .original_group_bytes = kernel.descriptor.group_segment_fixed_size,
                     .prepared_group_bytes = final_kernel->descriptor.group_segment_fixed_size,
                     .original_private_bytes = kernel.descriptor.private_segment_fixed_size,
                     .prepared_private_bytes = final_kernel->descriptor.private_segment_fixed_size,
                     .prepared_sgprs = metadata.value().sgprs,
                     .prepared_vgprs = metadata.value().vgprs,
                     .maximum_workgroup_size = metadata.value().maximum_workgroup_size,
                     .workgroup_id_scalar = workgroup_id_scalar(kernel, *first.target),
                     .required_workgroup_size = metadata.value().required_workgroup_size};
      std::unique_ptr<Decoder> decoder =
          Decoder::create(default_isa_target_registry(), first.target->public_id);
      if (decoder == nullptr) {
        return reject(error, "native ownership decoder is unavailable");
      }
      const std::array<uint64_t, 1> entry_offsets = {kernel.entry_text_offset};
      const FailureOr<std::vector<std::unique_ptr<BasicBlock>>> blocks =
          BasicBlock::build_reachable(object, *decoder, target.arch, entry_offsets);
      if (blocks.failed() || blocks.value().empty()) {
        return reject(error, "native reachable code cannot be enumerated");
      }
      for (const std::unique_ptr<BasicBlock> &block : blocks.value()) {
        // Text growth can move data sections while these native bytes stay fixed.
        // Reject PC readers until their data references can be relocated.
        for (const Instruction &instruction : block->instructions()) {
          if (instruction.reads_pc()) {
            error = {.code = InputErrorCode::UnsupportedInstruction,
                     .text_offset = instruction.src_loc(),
                     .message = "native entry '" + kernel.kernel_name +
                                "' reads PC; PC-relative data references are not preserved"};
            return Result::failure();
          }
        }
        row.native_code.push_back({.original_offset = block->start_offset(),
                                   .prepared_offset = block->start_offset(),
                                   .bytes = block->end_offset() - block->start_offset()});
      }
      std::ranges::sort(row.native_code, {}, &NativeCodeRange::original_offset);
    }
    facts.entries.push_back(std::move(row));
  }
  std::ranges::sort(facts.entries, {}, &ImageEntryFacts::key);
  std::ranges::sort(selected_keys);
  facts.selection_profile = selection_profile_digest(selected_keys, scope);
  util::StringDiagnostic contract_error;
  if (verify_artifact_images(facts, original, artifact.image, contract_error.emitter()).failed()) {
    return reject(error, contract_error.message());
  }
  for (const InputProgram &program : programs) {
    const Manifest *manifest = selected_manifest(facts, program.kernel.descriptor_file_offset);
    if (manifest == nullptr ||
        validate_image_entry_cfg(program, *manifest, artifact.image, error).failed()) {
      return Result::failure();
    }
  }
  return artifact;
}

} // namespace

Result validate_prepared(std::span<const uint8_t> original, const PreparedArtifact &artifact,
                         InputError &error) {
  // Shared framing, executable mapping and resource checks precede decoding.
  util::StringDiagnostic diagnostic;
  if (verify_artifact_images(artifact.manifest, original, artifact.image, diagnostic.emitter())
          .failed()) {
    return reject(error, diagnostic.message());
  }
  const TargetProfile *target = target_profile(artifact.manifest.target);
  const rocjitsu::IsaFamily arch = target->arch;
  const FailureOr<InputProgram> program =
      build_input_program(original, *target, artifact.manifest.entry.name, error);
  if (program.failed()) {
    return Result::failure();
  }
  // Re-lowering detects inconsistencies/tampering; it cannot establish that
  // the emitter's algorithm preserves state. Simulator/physical regressions
  // provide independent execution evidence for that property.
  const FailureOr<PreparedArtifact> canonical = lower(program.value(), error);
  if (canonical.failed()) {
    return Result::failure();
  }
  if (canonical.value().image != artifact.image ||
      canonical.value().manifest != artifact.manifest) {
    return reject(error, "prepared bytes/facts do not match the canonical preservation plan");
  }
  AmdGpuCodeObject object(artifact.image.data(), artifact.image.size());
  const Section &text = *object.text_sections().front();
  const std::vector<KernelDescriptorInfo> kernels = scan_kernel_descriptors(
      artifact.image, text.sectionOffset(), text.size(), text.sectionHeaderIndex());
  if (kernels.size() != 1 ||
      kernels.front().entry_text_offset != artifact.manifest.entry.prepared_entry_offset) {
    return reject(error, "final entry/descriptor placement mismatch");
  }
  std::string metadata_error;
  const FailureOr<MetadataResources> metadata =
      inspect_metadata(artifact.image, kernels.front(), &metadata_error);
  if (metadata.failed() || metadata.value().sgprs != artifact.manifest.entry.prepared_sgprs ||
      metadata.value().vgprs != artifact.manifest.entry.prepared_vgprs) {
    return reject(error, "final AMDHSA resource mismatch: " + metadata_error);
  }
  std::unique_ptr<Decoder> decoder =
      Decoder::create(default_isa_target_registry(), target->public_id);
  if (decoder == nullptr) {
    return reject(error, "final target decoder is unavailable");
  }
  const std::array<uint64_t, 1> entries{kernels.front().entry_text_offset};
  const FailureOr<std::vector<std::unique_ptr<BasicBlock>>> blocks =
      BasicBlock::build_reachable(object, *decoder, arch, entries);
  if (blocks.failed()) {
    return reject(error, "final executable CFG cannot be decoded");
  }
  std::vector<BasicBlock *> scope;
  for (const std::unique_ptr<BasicBlock> &block : blocks.value()) {
    scope.push_back(block.get());
  }
  const std::span<const uint8_t> prepared_text =
      std::span(artifact.image).subspan(text.sectionOffset(), text.size());
  const std::unordered_set<uint64_t> returns = scoped_call_return_offsets(scope, prepared_text);
  const std::unordered_set<BasicBlock *> allowed(scope.begin(), scope.end());
  for (const std::unique_ptr<BasicBlock> &block : blocks.value()) {
    const Instruction *terminator = block->terminator();
    const bool call_return =
        terminator && returns.contains(terminator->src_loc()) &&
        block->successor_issue() == BasicBlock::SuccessorIssue::IndirectControlFlow;
    const std::vector<BasicBlock::CallEdge> &calls = block->call_edges();
    const bool resolved_call =
        terminator && (terminator->flags() & INDIRECT_CALL) && calls.size() == 1 &&
        calls.front().source_call_offset == terminator->src_loc() &&
        allowed.contains(calls.front().callee) && allowed.contains(calls.front().continuation) &&
        block->successor_issue() == BasicBlock::SuccessorIssue::IndirectControlFlow;
    if (block->has_implicit_terminator() ||
        (block->successor_issue() != BasicBlock::SuccessorIssue::None && !call_return &&
         !resolved_call)) {
      return reject(error, "final CFG has an invalid successor or implicit terminator");
    }
  }
  return validate_lds_race_accesses(
      program.value(), artifact.manifest, scope,
      std::span(artifact.image).subspan(text.sectionOffset(), text.size()), error);
}

FailureOr<PreparedArtifact> prepare(std::span<const uint8_t> original, const TargetProfile &target,
                                    std::string_view entry, InputError &error) {
  const FailureOr<InputProgram> program = build_input_program(original, target, entry, error);
  if (program.failed()) {
    return Result::failure();
  }
  if (!util::is_valid_utf8(program.value().kernel.kernel_name)) {
    error = {.code = InputErrorCode::UnsupportedEntry,
             .text_offset = {},
             .message = "entry name is not valid UTF-8 for JSON output"};
    return Result::failure();
  }
  FailureOr<PreparedArtifact> artifact = lower(program.value(), error);
  if (artifact.failed() || validate_prepared(original, artifact.value(), error).failed()) {
    return Result::failure();
  }
  return artifact;
}

FailureOr<PreparedImageArtifact> prepare_image(std::span<const uint8_t> original,
                                               const TargetProfile &target,
                                               std::span<const std::string_view> entries,
                                               InputError &error, ScopeKind scope) {
  FailureOr<PreparedImageArtifact> artifact = lower_image(original, target, entries, error, scope);
  if (artifact.failed()) {
    return Result::failure();
  }
  const FailureOr<PreparedImageArtifact> canonical =
      lower_image(original, target, entries, error, scope);
  if (canonical.failed()) {
    return Result::failure();
  }
  if (artifact.value().image != canonical.value().image ||
      artifact.value().facts != canonical.value().facts) {
    return reject(error, "whole prepared image/facts differ from canonical plan");
  }
  return artifact;
}

} // namespace rocjitsu::consan
