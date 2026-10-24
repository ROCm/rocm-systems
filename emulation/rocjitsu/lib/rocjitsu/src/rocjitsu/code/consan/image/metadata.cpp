// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/consan/image/metadata.h"

#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/code/amdhsa_metadata.h"
#include "rocjitsu/code/consan/target.h"

#include <cstring>
#include <optional>

namespace rocjitsu::consan {
namespace {

bool matches_target(const amdhsa::MetadataValue &root, std::string_view processor) {
  const amdhsa::MetadataValue *target = root.field("amdhsa.target");
  if (target == nullptr || target->kind != amdhsa::MetadataValue::Kind::String)
    return false;
  // An omitted triple environment and "unknown" are equivalent. Keep the
  // complete processor match: feature suffixes need their own ELF ABI checks.
  for (const std::string_view prefix : {"amdgcn-amd-amdhsa--", "amdgcn-amd-amdhsa-unknown-"})
    if (target->string.starts_with(prefix) && target->string.substr(prefix.size()) == processor)
      return true;
  return false;
}

bool parse(std::span<const uint8_t> image, const KernelDescriptorInfo &kernel,
           MetadataResources &resources) {
  Elf64_Ehdr header{};
  if (image.size() < sizeof(header))
    return false;
  std::memcpy(&header, image.data(), sizeof(header));
  const TargetProfile *target = target_profile(elf_mach_name(header.e_flags));
  if (target == nullptr)
    return false;
  const uint32_t wave_size = kernel_wavefront_size(target->arch, kernel.descriptor);
  const uint32_t granule = descriptor_vgpr_granularity_for_wavefront(target->arch, wave_size);
  const FailureOr<amdhsa::MetadataDocument> document = amdhsa::MetadataDocument::parse(image);
  if (document.failed())
    return false;
  const amdhsa::MetadataValue &root = document.value().root();
  if (!matches_target(root, target->name))
    return false;
  const amdhsa::MetadataValue *version = root.field("amdhsa.version");
  if (version == nullptr || version->array.size() != 2 ||
      version->array[0].kind != amdhsa::MetadataValue::Kind::Integer ||
      version->array[0].integer != 1 ||
      version->array[1].kind != amdhsa::MetadataValue::Kind::Integer ||
      version->array[1].integer != 2)
    return false;
  const amdhsa::MetadataValue *kernels = root.field("amdhsa.kernels");
  if (kernels == nullptr || kernels->kind != amdhsa::MetadataValue::Kind::Array ||
      kernels->array.size() != 1)
    return false;
  const amdhsa::MetadataValue &entry = kernels->array.front();
  const std::optional<uint32_t> kernarg = entry.integer_field(".kernarg_segment_size");
  const std::optional<uint32_t> alignment = entry.integer_field(".kernarg_segment_align");
  const std::optional<uint32_t> sgprs = entry.integer_field(".sgpr_count");
  const std::optional<uint32_t> vgprs = entry.integer_field(".vgpr_count");
  if (target->arch == ROCJITSU_CODE_ARCH_CDNA3 || target->arch == ROCJITSU_CODE_ARCH_CDNA4) {
    const uint32_t accumulator_base =
        (RJ_AMDHSA_BITS_GET(kernel.descriptor.compute_pgm_rsrc3,
                            rocjitsu::amdhsa::COMPUTE_PGM_RSRC3_GFX90A_ACCUM_OFFSET) +
         1) *
        4;
    if (!vgprs || *vgprs > accumulator_base)
      return false;
  }
  if (!entry.text_field_is(".name", kernel.kernel_name) ||
      !entry.text_field_is(".symbol", kernel.kernel_name + ".kd") || !kernarg ||
      *kernarg != kernel.descriptor.kernarg_size || !alignment || *alignment != 8 || !sgprs ||
      *sgprs == 0 || *sgprs > 106 || !vgprs || *vgprs == 0 ||
      ((*vgprs + granule - 1) / granule) != 1 + (kernel.descriptor.compute_pgm_rsrc1 & 63) ||
      entry.integer_field(".wavefront_size") != wave_size || !supports_wave(*target, wave_size) ||
      entry.integer_field(".group_segment_fixed_size") !=
          kernel.descriptor.group_segment_fixed_size ||
      entry.integer_field(".private_segment_fixed_size") != 0)
    return false;
  const std::optional<uint32_t> maximum = entry.integer_field(".max_flat_workgroup_size");
  if (!maximum || *maximum == 0 || *maximum > 1024)
    return false;
  const amdhsa::MetadataValue *dynamic_stack = entry.field(".uses_dynamic_stack");
  if ((dynamic_stack != nullptr && (dynamic_stack->kind != amdhsa::MetadataValue::Kind::Boolean ||
                                    dynamic_stack->integer != 0)) ||
      entry.field(".reqd_workgroup_size") != nullptr)
    return false;
  const amdhsa::MetadataValue *arguments = entry.field(".args");
  if (arguments == nullptr || arguments->kind != amdhsa::MetadataValue::Kind::Array)
    return false;
  uint64_t previous_end = 0;
  for (const amdhsa::MetadataValue &argument : arguments->array) {
    const std::optional<uint32_t> size = argument.integer_field(".size");
    const std::optional<uint32_t> offset = argument.integer_field(".offset");
    if (!size || *size == 0 || !offset || *offset < previous_end ||
        uint64_t{*offset} + *size > *kernarg ||
        (!argument.text_field_is(".value_kind", "by_value") &&
         !argument.text_field_is(".value_kind", "global_buffer")))
      return false;
    if (argument.text_field_is(".value_kind", "global_buffer") &&
        (*size != 8 || (*offset % 8) != 0 || !argument.text_field_is(".address_space", "global")))
      return false;
    previous_end = uint64_t{*offset} + *size;
  }
  resources = {*kernarg, *alignment, *sgprs, *vgprs, *maximum};
  return true;
}

} // namespace

FailureOr<MetadataResources> inspect_metadata(std::span<const uint8_t> image,
                                              const KernelDescriptorInfo &kernel,
                                              std::string *error) {
  MetadataResources resources;
  if (!parse(image, kernel, resources)) {
    if (error != nullptr)
      *error = "unsupported or inconsistent AMDHSA metadata/argument ABI";
    return Result::failure();
  }
  return resources;
}

} // namespace rocjitsu::consan
