// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/fault_injection/ordinary_wrong_address.h"

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/patch/fault_injection/fault_mutation_internal.h"
#include "rocjitsu/code/patch/fault_injection/ordinary_acquire_internal.h"
#include "rocjitsu/code/patch/instrumentor.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/shared/memory_displacement_encoded_fields.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/isa/target_registry.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace rocjitsu {

namespace {

bool admitted_profile(rj_code_target_id_t target, const OrdinaryAcquireSite &site) {
  switch (site.profile) {
  case OrdinaryAcquireProfile::Cdna5GlobalB32:
    return (target == ROCJITSU_CODE_TARGET_GFX1250 || target == ROCJITSU_CODE_TARGET_GFX1251) &&
           site.instruction_size == 12 && site.form == "global_load_b32";
  case OrdinaryAcquireProfile::Rdna4GlobalB32:
    return false;
  }
  return false;
}

std::optional<int32_t> read_load_offset(rj_code_target_id_t target, const OrdinaryAcquireSite &site,
                                        std::span<const uint32_t> words) {
  switch (site.profile) {
  case OrdinaryAcquireProfile::Cdna5GlobalB32:
    if (words.size() != 3)
      return std::nullopt;
    return amdgpu::read_vmem_signed24_offset(target, words);
  case OrdinaryAcquireProfile::Rdna4GlobalB32:
    return std::nullopt;
  }
  return std::nullopt;
}

std::optional<std::vector<uint32_t>> replace_load_offset(rj_code_target_id_t target,
                                                         const OrdinaryAcquireSite &site,
                                                         std::span<const uint32_t> words,
                                                         int64_t offset) {
  switch (site.profile) {
  case OrdinaryAcquireProfile::Cdna5GlobalB32: {
    const std::optional<std::array<uint32_t, 3>> replacement =
        amdgpu::replace_vmem_signed24_offset(target, words, offset);
    if (!replacement)
      return std::nullopt;
    return std::vector<uint32_t>(replacement->begin(), replacement->end());
  }
  case OrdinaryAcquireProfile::Rdna4GlobalB32:
    return std::nullopt;
  }
  return std::nullopt;
}

bool load_scope_matches(const OrdinaryAcquireSite &site, std::span<const uint32_t> words) {
  switch (site.profile) {
  case OrdinaryAcquireProfile::Cdna5GlobalB32: {
    if (words.size_bytes() != sizeof(cdna5::VglobalMachineInst))
      return false;
    cdna5::VglobalMachineInst raw = {};
    std::memcpy(&raw, words.data(), sizeof(raw));
    return site.raw_scope && (*site.raw_scope == 2 || *site.raw_scope == 3) &&
           raw.scope == *site.raw_scope;
  }
  case OrdinaryAcquireProfile::Rdna4GlobalB32:
    return false;
  }
  return false;
}

bool valid_extent(const FaultMutationSection &section, std::span<const uint8_t> image,
                  uint64_t offset, uint64_t size) {
  return size > 0 && offset <= section.size && size <= section.size - offset &&
         section.file_offset <= image.size() && offset <= image.size() - section.file_offset &&
         size <= image.size() - section.file_offset - offset;
}

} // namespace

std::vector<OrdinaryWrongAddressCapability> ordinary_wrong_address_capabilities() {
  std::vector<OrdinaryWrongAddressCapability> capabilities;
  const IsaTargetRegistry &registry = default_isa_target_registry();
  for (rj_code_target_id_t target : {ROCJITSU_CODE_TARGET_GFX1250, ROCJITSU_CODE_TARGET_GFX1251}) {
    if (registry.find_target(target) && Decoder::create(registry, target))
      capabilities.push_back(
          {.target = target, .wave_size = 32, .profile = OrdinaryAcquireProfile::Cdna5GlobalB32});
  }
  return capabilities;
}

FailureOr<OrdinaryWrongAddressResult>
apply_ordinary_wrong_address(const FaultMutationContext &context,
                             const OrdinaryAcquireSelection &selection,
                             std::optional<uint32_t> byte_delta, FaultMutationError &error) {
  error = {};
  if (!selection.record_ || selection.record_->source.lock() != context.source_ ||
      std::ranges::find(selection.record_->sites, selection.site_) ==
          selection.record_->sites.end() ||
      std::ranges::find(selection.site_.affected_owners, selection.record_->owner) ==
          selection.site_.affected_owners.end())
    return reject_fault_mutation(error, FaultMutationErrorCode::BadIdentity,
                                 "selection belongs to another or expired ordinary acquire source");
  const OrdinaryAcquireSite &site = selection.site_;
  const uint64_t offset = site.offset;
  if (!byte_delta || *byte_delta == 0 || *byte_delta % sizeof(uint32_t) != 0 ||
      *byte_delta > 0x7fffffu)
    return reject_fault_mutation(
        error, FaultMutationErrorCode::InvalidParameters,
        "ordinary byte delta must be positive, four-byte aligned, and at most 0x7fffff", offset);
  const rj_code_target_id_t target = context.source().target_id();
  if (!admitted_profile(target, site))
    return reject_fault_mutation(error, FaultMutationErrorCode::UnsupportedForm,
                                 "selected acquire profile has no ordinary address recipe", offset);

  const std::span<const uint8_t> image = context.image();
  for (const std::pair<const uint64_t, std::vector<uint8_t>> &proof :
       selection.record_->source_bytes) {
    const uint64_t proof_offset = proof.first;
    const std::vector<uint8_t> &bytes = proof.second;
    if (!valid_extent(selection.record_->section, image, proof_offset, bytes.size()))
      return reject_fault_mutation(error, FaultMutationErrorCode::ValidationFailure,
                                   "invalid ordinary acquire proof extent", proof_offset);
    if (std::memcmp(image.data() + selection.record_->section.file_offset + proof_offset,
                    bytes.data(), bytes.size()) != 0)
      return reject_fault_mutation(error, FaultMutationErrorCode::BadIdentity,
                                   "ordinary acquire proof bytes changed", proof_offset);
  }
  const std::map<uint64_t, std::vector<uint8_t>>::const_iterator expected =
      selection.record_->source_bytes.find(offset);
  const uint32_t size = site.instruction_size;
  if (expected == selection.record_->source_bytes.end() || expected->second.size() != size ||
      size % sizeof(uint32_t) != 0 ||
      !valid_extent(selection.record_->section, image, offset, size))
    return reject_fault_mutation(error, FaultMutationErrorCode::ValidationFailure,
                                 "invalid ordinary load extent", offset);
  std::vector<uint32_t> original(size / sizeof(uint32_t));
  std::memcpy(original.data(), expected->second.data(), size);
  std::unique_ptr<Decoder> decoder = Decoder::create(default_isa_target_registry(), target);
  if (!decoder)
    return reject_fault_mutation(error, FaultMutationErrorCode::ValidationFailure,
                                 "missing ordinary load decoder", offset);
  const DecodeResult decoded = decoder->decode_window(original, offset);
  const std::optional<int32_t> original_offset = read_load_offset(target, site, original);
  if (decoded.failed() || static_cast<uint32_t>(decoded.value()->size()) != size ||
      decoded.value()->mnemonic() != site.form || !original_offset ||
      *original_offset != site.byte_offset || !load_scope_matches(site, original))
    return reject_fault_mutation(error, FaultMutationErrorCode::BadIdentity,
                                 "stale ordinary load form, offset, or scope", offset);

  const int64_t replacement_offset = static_cast<int64_t>(*original_offset) + *byte_delta;
  if (replacement_offset % static_cast<int64_t>(sizeof(uint32_t)) != 0)
    return reject_fault_mutation(error, FaultMutationErrorCode::InvalidParameters,
                                 "ordinary delta produces an unaligned byte offset", offset);
  const std::optional<std::vector<uint32_t>> replacement =
      replace_load_offset(target, site, original, replacement_offset);
  if (!replacement)
    return reject_fault_mutation(error, FaultMutationErrorCode::InvalidParameters,
                                 "ordinary delta overflows the profile's encoded offset", offset);

  FailureOr<std::unique_ptr<Instrumentor>> fork = context.fork_instrumentor(error);
  if (fork.failed())
    return Result::failure();
  InstructionReplacement edit = {.expected_bytes = expected->second,
                                 .replacement_bytes = std::vector<uint8_t>(size)};
  std::memcpy(edit.replacement_bytes.data(), replacement->data(), size);
  std::unique_ptr<Instrumentor> instrumentor = std::move(fork).value();
  instrumentor->add_point({.anchor_offset = offset,
                           .kind = InstrumentationKind::ReplaceWithInst,
                           .probe_symbol = {},
                           .probe_args = {},
                           .replacement = std::move(edit)});
  InstrumentedCodeObject applied = instrumentor->patch();
  if (applied.edit_error) {
    error = translate_fault_mutation_edit_error(*applied.edit_error);
    return Result::failure();
  }
  if (!applied.errors.empty())
    return reject_fault_mutation(error, FaultMutationErrorCode::ValidationFailure,
                                 applied.errors.front(), offset);
  return OrdinaryWrongAddressResult{.selection = selection, .image = std::move(applied.elf_bytes)};
}

} // namespace rocjitsu
