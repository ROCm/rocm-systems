// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/consan/image/metadata.h"
#include "rocjitsu/isa/isa_family.h"

#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/code/amdhsa_metadata.h"
#include "rocjitsu/code/consan/target.h"

#include <bit>
#include <cstring>
#include <optional>
#include <set>

namespace rocjitsu::consan {
namespace {

bool matches_target(const amdhsa::MetadataValue &root, std::string_view processor) {
  const amdhsa::MetadataValue *target = root.field("amdhsa.target");
  if (target == nullptr || target->kind != amdhsa::MetadataValue::Kind::String) {
    return false;
  }
  // An omitted triple environment and "unknown" are equivalent. Keep the
  // complete processor match: feature suffixes need their own ELF ABI checks.
  for (const std::string_view prefix : {"amdgcn-amd-amdhsa--", "amdgcn-amd-amdhsa-unknown-"}) {
    if (target->string.starts_with(prefix) && target->string.substr(prefix.size()) == processor) {
      return true;
    }
  }
  // This producer spelling names the same concrete ISA, not a target suffix.
  if (processor == "gfx1250" && target->string == "amdgpu12.50-amd-amdhsa-unknown-gfx1250") {
    return true;
  }
  return false;
}

uint32_t hidden_argument_width(std::string_view kind) {
  constexpr std::array<std::pair<std::string_view, uint32_t>, 4> prefixes = {
      {{"hidden_block_count_", 4},
       {"hidden_group_size_", 2},
       {"hidden_remainder_", 2},
       {"hidden_global_offset_", 8}}};
  for (const std::pair<std::string_view, uint32_t> &prefix : prefixes) {
    if (kind.starts_with(prefix.first) && kind.size() == prefix.first.size() + 1 &&
        (kind.back() == 'x' || kind.back() == 'y' || kind.back() == 'z')) {
      return prefix.second;
    }
  }
  return kind == "hidden_grid_dims" ? 2 : 0;
}

bool parse(std::span<const uint8_t> image, const KernelDescriptorInfo &kernel,
           MetadataResources &resources, MetadataPolicy policy) {
  Elf64_Ehdr header{};
  if (image.size() < sizeof(header)) {
    return false;
  }
  std::memcpy(&header, image.data(), sizeof(header));
  const TargetProfile *target = target_profile(elf_mach_name(header.e_flags));
  if (target == nullptr) {
    return false;
  }
  const uint32_t wave_size = kernel_wavefront_size(target->arch, kernel.descriptor);
  const uint32_t granule = descriptor_vgpr_granularity_for_wavefront(target->arch, wave_size);
  const FailureOr<amdhsa::MetadataDocument> document = amdhsa::MetadataDocument::parse(image);
  if (document.failed()) {
    return false;
  }
  const amdhsa::MetadataValue &root = document.value().root();
  if (!matches_target(root, target->name)) {
    return false;
  }
  const amdhsa::MetadataValue *version = root.field("amdhsa.version");
  if (version == nullptr || version->array.size() != 2 ||
      version->array[0].kind != amdhsa::MetadataValue::Kind::Integer ||
      version->array[0].integer != 1 ||
      version->array[1].kind != amdhsa::MetadataValue::Kind::Integer ||
      version->array[1].integer != 2) {
    return false;
  }
  const amdhsa::MetadataValue *kernels = root.field("amdhsa.kernels");
  if (kernels == nullptr || kernels->kind != amdhsa::MetadataValue::Kind::Array ||
      kernels->array.empty() || kernels->array.size() > 1024) {
    return false;
  }
  const amdhsa::MetadataValue *selected = nullptr;
  std::set<std::string_view> names;
  for (const amdhsa::MetadataValue &candidate : kernels->array) {
    const amdhsa::MetadataValue *name = candidate.field(".name");
    if (name == nullptr || name->kind != amdhsa::MetadataValue::Kind::String ||
        !names.insert(name->string).second) {
      return false;
    }
    if (name->string == kernel.kernel_name) {
      selected = &candidate;
    }
  }
  if (selected == nullptr) {
    return false;
  }
  const amdhsa::MetadataValue &entry = *selected;
  const std::optional<uint32_t> kernarg = entry.integer_field(".kernarg_segment_size");
  const std::optional<uint32_t> alignment = entry.integer_field(".kernarg_segment_align");
  const std::optional<uint32_t> sgprs = entry.integer_field(".sgpr_count");
  const std::optional<uint32_t> vgprs = entry.integer_field(".vgpr_count");
  const uint32_t allocated_vgpr_units = 1 + (kernel.descriptor.compute_pgm_rsrc1 & 63);
  const bool valid_vgprs =
      vgprs && *vgprs != 0 &&
      (policy == MetadataPolicy::Native ? *vgprs <= allocated_vgpr_units * granule
                                        : (*vgprs + granule - 1) / granule == allocated_vgpr_units);
  const bool is_cdna =
      target->arch == rocjitsu::IsaFamily::Cdna3 || target->arch == rocjitsu::IsaFamily::Cdna4;
  if (is_cdna && policy == MetadataPolicy::Native) {
    // Compiler metadata includes special SGPRs in the total allocation.
    const uint32_t allocated_sgprs =
        (RJ_AMDHSA_BITS_GET(kernel.descriptor.compute_pgm_rsrc1,
                            rocjitsu::amdhsa::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT) +
         1) *
        8;
    if (!sgprs || *sgprs > allocated_sgprs)
      return false;
  }
  if (is_cdna && policy == MetadataPolicy::Selected) {
    const uint32_t accumulator_base =
        (RJ_AMDHSA_BITS_GET(kernel.descriptor.compute_pgm_rsrc3,
                            rocjitsu::amdhsa::COMPUTE_PGM_RSRC3_GFX90A_ACCUM_OFFSET) +
         1) *
        4;
    if (!vgprs || *vgprs > accumulator_base) {
      return false;
    }
  }
  if (!entry.text_field_is(".name", kernel.kernel_name) ||
      !entry.text_field_is(".symbol", kernel.kernel_name + ".kd") || !kernarg ||
      *kernarg != kernel.descriptor.kernarg_size || !alignment ||
      !std::has_single_bit(*alignment) || (policy == MetadataPolicy::Selected && *alignment < 8) ||
      !sgprs || *sgprs == 0 || *sgprs > 106 || !valid_vgprs ||
      entry.integer_field(".wavefront_size") != wave_size || !supports_wave(*target, wave_size) ||
      entry.integer_field(".group_segment_fixed_size") !=
          kernel.descriptor.group_segment_fixed_size ||
      entry.integer_field(".private_segment_fixed_size") !=
          kernel.descriptor.private_segment_fixed_size) {
    return false;
  }
  const std::optional<uint32_t> maximum = entry.integer_field(".max_flat_workgroup_size");
  if (!maximum || *maximum == 0 || *maximum > 1024) {
    return false;
  }
  const amdhsa::MetadataValue *dynamic_stack = entry.field(".uses_dynamic_stack");
  if (dynamic_stack != nullptr && dynamic_stack->kind != amdhsa::MetadataValue::Kind::Boolean) {
    return false;
  }
  std::array<uint32_t, 3> required{};
  const amdhsa::MetadataValue *shape = entry.field(".reqd_workgroup_size");
  if (shape != nullptr) {
    if (shape->kind != amdhsa::MetadataValue::Kind::Array ||
        shape->array.size() != required.size()) {
      return false;
    }
    uint32_t product = 1;
    for (size_t index = 0; index < required.size(); ++index) {
      const amdhsa::MetadataValue &dimension = shape->array[index];
      if (dimension.kind != amdhsa::MetadataValue::Kind::Integer || dimension.integer == 0 ||
          dimension.integer > *maximum / product) {
        return false;
      }
      required[index] = static_cast<uint32_t>(dimension.integer);
      product *= required[index];
    }
  }
  const amdhsa::MetadataValue *arguments = entry.field(".args");
  if (arguments == nullptr || arguments->kind != amdhsa::MetadataValue::Kind::Array) {
    return false;
  }
  uint64_t previous_end = 0;
  for (const amdhsa::MetadataValue &argument : arguments->array) {
    const std::optional<uint32_t> size = argument.integer_field(".size");
    const std::optional<uint32_t> offset = argument.integer_field(".offset");
    if (!size || *size == 0 || !offset || *offset < previous_end ||
        uint64_t{*offset} + *size > *kernarg) {
      return false;
    }
    const amdhsa::MetadataValue *kind = argument.field(".value_kind");
    if (kind == nullptr || kind->kind != amdhsa::MetadataValue::Kind::String) {
      return false;
    }
    if (kind->string != "by_value" && kind->string != "global_buffer") {
      const uint32_t width = hidden_argument_width(kind->string);
      if (width == 0 || *size != width || *offset % width != 0) {
        return false;
      }
    }
    if (argument.text_field_is(".value_kind", "global_buffer") &&
        (*size != 8 || (*offset % 8) != 0 ||
         (!argument.text_field_is(".address_space", "global") &&
          !argument.text_field_is(".address_space", "constant")))) {
      return false;
    }
    previous_end = uint64_t{*offset} + *size;
  }
  resources = {.sgprs = *sgprs,
               .vgprs = *vgprs,
               .maximum_workgroup_size = *maximum,
               .kernarg_alignment = *alignment,
               .required_workgroup_size = required,
               .uses_dynamic_stack = dynamic_stack != nullptr && dynamic_stack->integer != 0};
  return true;
}

} // namespace

FailureOr<MetadataResources> inspect_metadata(std::span<const uint8_t> image,
                                              const KernelDescriptorInfo &kernel,
                                              std::string *error, MetadataPolicy policy) {
  MetadataResources resources;
  if (!parse(image, kernel, resources, policy)) {
    if (error != nullptr) {
      *error = "unsupported or inconsistent AMDHSA metadata/argument ABI for " + kernel.kernel_name;
    }
    return Result::failure();
  }
  return resources;
}

} // namespace rocjitsu::consan
