// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/fault_injection/lds_wrong_address.h"

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/code/analysis/gfx1250_vgpr_msb.h"
#include "rocjitsu/code/analysis/instruction_boundaries.h"
#include "rocjitsu/code/basic_block.h"
#include "rocjitsu/code/patch/fault_injection/fault_mutation_internal.h"
#include "rocjitsu/code/patch/instrumentor.h"
#include "rocjitsu/isa/arch/amdgpu/cdna5/encoded_fields.h"
#include "rocjitsu/isa/arch/amdgpu/rdna4/encoded_fields.h"
#include "rocjitsu/isa/arch/amdgpu/shared/cdna_ds_encoded_fields.h"
#include "rocjitsu/isa/arch/amdgpu/shared/ds_address_encoded_fields.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/isa/isa_family.h"
#include "rocjitsu/isa/operand.h"
#include "rocjitsu/isa/target_registry.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

namespace rocjitsu {

struct LdsRecord {
  std::weak_ptr<const AmdGpuCodeObject> source;
  FaultMutationSection section;
  FaultMutationOwner owner;
  uint32_t allocated_vgprs = 0;
  std::vector<LdsSite> sites;
};

namespace {
using LdsAddressRewriter = std::optional<std::array<uint32_t, 2>> (*)(rj_code_target_id_t,
                                                                      std::span<const uint32_t>,
                                                                      uint32_t);

struct LdsProfile {
  rj_code_target_id_t target;
  IsaFamily arch;
  std::string_view form;
  LdsAddressRewriter rewrite;
};

std::optional<std::array<uint32_t, 2>> replace_rdna4_b32_address(rj_code_target_id_t target,
                                                                 std::span<const uint32_t> words,
                                                                 uint32_t reg) {
  std::optional<std::array<uint32_t, 2>> replacement =
      rdna4::replace_ds_store_b32_address(target, words, reg);
  return replacement ? replacement : rdna4::replace_ds_load_b32_address(target, words, reg);
}

template <typename... Forms>
constexpr std::array<LdsProfile, sizeof...(Forms)>
make_lds_profiles(rj_code_target_id_t target, IsaFamily arch, LdsAddressRewriter rewrite,
                  Forms... forms) {
  return {{LdsProfile{.target = target, .arch = arch, .form = forms, .rewrite = rewrite}...}};
}

template <size_t... Sizes>
constexpr std::array<LdsProfile, (Sizes + ...)>
join_lds_profiles(const std::array<LdsProfile, Sizes> &...groups) {
  std::array<LdsProfile, (Sizes + ...)> profiles = {};
  LdsProfile *next = profiles.data();
  ((next = std::copy(groups.begin(), groups.end(), next)), ...);
  return profiles;
}

#define RDNA4_LDS_PROFILES(target_name, ...)                                                       \
  make_lds_profiles(ROCJITSU_CODE_TARGET_##target_name, IsaFamily::Rdna4, __VA_ARGS__)
#define CDNA3_LDS_PROFILES(...)                                                                    \
  make_lds_profiles(ROCJITSU_CODE_TARGET_GFX942, IsaFamily::Cdna3, __VA_ARGS__)
#define CDNA4_LDS_PROFILES(...)                                                                    \
  make_lds_profiles(ROCJITSU_CODE_TARGET_GFX950, IsaFamily::Cdna4, __VA_ARGS__)
#define CDNA5_LDS_PROFILES(...)                                                                    \
  make_lds_profiles(ROCJITSU_CODE_TARGET_GFX1250, IsaFamily::Cdna5, __VA_ARGS__)

constexpr std::array<LdsProfile, 145> kLdsProfiles = join_lds_profiles(
    RDNA4_LDS_PROFILES(GFX1200, replace_rdna4_b32_address, "ds_store_b32"),
    RDNA4_LDS_PROFILES(GFX1201, replace_rdna4_b32_address, "ds_store_b32"),
    RDNA4_LDS_PROFILES(GFX1200, replace_rdna4_b32_address, "ds_load_b32"),
    RDNA4_LDS_PROFILES(GFX1201, replace_rdna4_b32_address, "ds_load_b32"),
    RDNA4_LDS_PROFILES(GFX1200, amdgpu::replace_ds_address_field, "ds_load_2addr_b64",
                       "ds_store_2addr_b32"),
    RDNA4_LDS_PROFILES(GFX1201, amdgpu::replace_ds_address_field, "ds_load_2addr_b64",
                       "ds_store_2addr_b32"),
    CDNA3_LDS_PROFILES(amdgpu::replace_cdna_ds_write_address, "ds_write_b32", "ds_write_b128"),
    CDNA4_LDS_PROFILES(amdgpu::replace_cdna_ds_write_address, "ds_write_b32", "ds_write_b128"),
    CDNA3_LDS_PROFILES(amdgpu::replace_ds_address_field, "ds_read2_b64", "ds_write2st64_b64"),
    CDNA4_LDS_PROFILES(amdgpu::replace_ds_address_field, "ds_read2_b64", "ds_write2st64_b64"),
    CDNA5_LDS_PROFILES(cdna5::replace_ds_b32_address, "ds_store_b32", "ds_load_b32"),
    RDNA4_LDS_PROFILES(GFX1200, amdgpu::replace_ds_address_field, "ds_load_b96", "ds_store_b96"),
    RDNA4_LDS_PROFILES(GFX1201, amdgpu::replace_ds_address_field, "ds_load_b96", "ds_store_b96"),
    CDNA3_LDS_PROFILES(amdgpu::replace_ds_address_field, "ds_read_b96", "ds_write_b96"),
    CDNA4_LDS_PROFILES(amdgpu::replace_ds_address_field, "ds_read_b96", "ds_write_b96"),
    CDNA5_LDS_PROFILES(amdgpu::replace_ds_address_field, "ds_load_b96", "ds_store_b96"),
    RDNA4_LDS_PROFILES(GFX1200, amdgpu::replace_ds_address_field, "ds_load_b64", "ds_store_b64",
                       "ds_load_b128", "ds_store_b128", "ds_load_2addr_b32",
                       "ds_load_2addr_stride64_b32", "ds_store_2addr_stride64_b32",
                       "ds_store_2addr_b64", "ds_load_2addr_stride64_b64",
                       "ds_store_2addr_stride64_b64"),
    RDNA4_LDS_PROFILES(GFX1201, amdgpu::replace_ds_address_field, "ds_load_b64", "ds_store_b64",
                       "ds_load_b128", "ds_store_b128", "ds_load_2addr_b32",
                       "ds_load_2addr_stride64_b32", "ds_store_2addr_stride64_b32",
                       "ds_store_2addr_b64", "ds_load_2addr_stride64_b64",
                       "ds_store_2addr_stride64_b64"),
    CDNA5_LDS_PROFILES(amdgpu::replace_ds_address_field, "ds_load_b64", "ds_store_b64",
                       "ds_load_b128", "ds_store_b128"),
    CDNA3_LDS_PROFILES(amdgpu::replace_ds_address_field, "ds_read_b32", "ds_read_b64",
                       "ds_write_b64", "ds_read_b128", "ds_read2_b32", "ds_write2_b32",
                       "ds_read2st64_b32", "ds_write2st64_b32", "ds_write2_b64",
                       "ds_read2st64_b64"),
    CDNA4_LDS_PROFILES(amdgpu::replace_ds_address_field, "ds_read_b32", "ds_read_b64",
                       "ds_write_b64", "ds_read_b128", "ds_read2_b32", "ds_write2_b32",
                       "ds_read2st64_b32", "ds_write2st64_b32", "ds_write2_b64",
                       "ds_read2st64_b64"),
    CDNA5_LDS_PROFILES(amdgpu::replace_ds_address_field, "ds_load_u16", "ds_load_i16", "ds_load_u8",
                       "ds_load_i8", "ds_store_b8", "ds_store_b8_d16_hi", "ds_store_b16",
                       "ds_store_b16_d16_hi"),
    CDNA4_LDS_PROFILES(amdgpu::replace_ds_address_field, "ds_read_u16", "ds_read_b64_tr_b16",
                       "ds_write_b16", "ds_read_i8", "ds_read_u8", "ds_write_b8",
                       "ds_write_b8_d16_hi", "ds_write_b16_d16_hi"),
    RDNA4_LDS_PROFILES(GFX1200, amdgpu::replace_ds_address_field, "ds_load_i8", "ds_load_u8",
                       "ds_load_i16", "ds_load_u16", "ds_load_u8_d16", "ds_load_u8_d16_hi",
                       "ds_load_i8_d16", "ds_load_i8_d16_hi", "ds_load_u16_d16",
                       "ds_load_u16_d16_hi", "ds_store_b8", "ds_store_b16", "ds_store_b8_d16_hi",
                       "ds_store_b16_d16_hi"),
    RDNA4_LDS_PROFILES(GFX1201, amdgpu::replace_ds_address_field, "ds_load_i8", "ds_load_u8",
                       "ds_load_i16", "ds_load_u16", "ds_load_u8_d16", "ds_load_u8_d16_hi",
                       "ds_load_i8_d16", "ds_load_i8_d16_hi", "ds_load_u16_d16",
                       "ds_load_u16_d16_hi", "ds_store_b8", "ds_store_b16", "ds_store_b8_d16_hi",
                       "ds_store_b16_d16_hi"),
    CDNA5_LDS_PROFILES(amdgpu::replace_ds_address_field, "ds_load_u8_d16", "ds_load_u8_d16_hi",
                       "ds_load_i8_d16", "ds_load_i8_d16_hi", "ds_load_u16_d16",
                       "ds_load_u16_d16_hi", "ds_load_tr8_b64", "ds_load_tr16_b128"),
    CDNA3_LDS_PROFILES(amdgpu::replace_ds_address_field, "ds_read_i8", "ds_read_u8", "ds_read_i16",
                       "ds_read_u16", "ds_read_u8_d16", "ds_read_u8_d16_hi", "ds_read_i8_d16",
                       "ds_read_i8_d16_hi", "ds_read_u16_d16", "ds_read_u16_d16_hi", "ds_write_b8",
                       "ds_write_b16", "ds_write_b8_d16_hi", "ds_write_b16_d16_hi"),
    CDNA4_LDS_PROFILES(amdgpu::replace_ds_address_field, "ds_read_i16", "ds_read_u8_d16",
                       "ds_read_u8_d16_hi", "ds_read_i8_d16", "ds_read_i8_d16_hi",
                       "ds_read_u16_d16", "ds_read_u16_d16_hi"));

#undef RDNA4_LDS_PROFILES
#undef CDNA3_LDS_PROFILES
#undef CDNA4_LDS_PROFILES
#undef CDNA5_LDS_PROFILES

const LdsProfile *find_lds_profile(rj_code_target_id_t target, std::string_view form) {
  const decltype(kLdsProfiles)::const_iterator found =
      std::ranges::find_if(kLdsProfiles, [&](const LdsProfile &profile) {
        return profile.target == target && profile.form == form;
      });
  return found == kLdsProfiles.end() ? nullptr : &*found;
}

std::optional<std::array<uint32_t, 2>>
replace_lds_address(const LdsProfile &profile, std::span<const uint32_t> words, uint32_t reg) {
  if (!profile.rewrite || words.size() != 2) {
    return std::nullopt;
  }
  // GDS is outside the LDS profile. ACC refers to data and is preserved.
  if ((profile.arch == IsaFamily::Cdna3 || profile.arch == IsaFamily::Cdna4) &&
      (words[0] & 0x00010000u) != 0) {
    return std::nullopt;
  }
  return profile.rewrite(profile.target, words, reg);
}
} // namespace

std::vector<LdsCapability> lds_wrong_address_capabilities() {
  const IsaTargetRegistry &registry = default_isa_target_registry();
  std::vector<LdsCapability> capabilities;
  for (const LdsProfile &profile : kLdsProfiles) {
    if (!registry.find_target(profile.target) || !Decoder::create(registry, profile.target)) {
      continue;
    }
    capabilities.push_back({.target = profile.target,
                            .wave_size = profile.target == ROCJITSU_CODE_TARGET_GFX1250 ? 32u : 64u,
                            .form = profile.form});
    if (profile.arch == IsaFamily::Rdna4) {
      capabilities.push_back({.target = profile.target, .wave_size = 32u, .form = profile.form});
    }
  }
  return capabilities;
}

std::span<const LdsSite> LdsInventory::sites() const {
  return record_ ? std::span<const LdsSite>(record_->sites) : std::span<const LdsSite>();
}

FailureOr<LdsInventory> discover_lds_wrong_address(const FaultMutationContext &context,
                                                   std::string_view entry,
                                                   FaultMutationError &error) {
  error = {};
  const rj_code_target_id_t target = context.source().target_id();
  const decltype(kLdsProfiles)::const_iterator target_profile =
      std::ranges::find(kLdsProfiles, target, &LdsProfile::target);
  if (target_profile == kLdsProfiles.end()) {
    return reject_fault_mutation(error, FaultMutationErrorCode::UnsupportedTarget,
                                 "target has no LDS profile");
  }
  FailureOr<const FaultSourceFacts *> found = context.source_facts(entry, error);
  if (found.failed()) {
    return Result::failure();
  }
  const FaultSourceFacts &facts = *found.value();
  const bool cdna5 = target == ROCJITSU_CODE_TARGET_GFX1250;
  const uint32_t wave = kernel_wavefront_size(target_profile->arch, facts.descriptor.descriptor);
  const bool rdna4 = target_profile->arch == IsaFamily::Rdna4;
  if ((!rdna4 && wave != (cdna5 ? 32u : 64u)) ||
      (!rdna4 && !cdna5 &&
       (facts.descriptor.descriptor.kernel_code_properties &
        amdhsa::KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32) != 0)) {
    return reject_fault_mutation(error, FaultMutationErrorCode::UnsupportedForm,
                                 "unsupported LDS wave size");
  }
  std::shared_ptr<LdsRecord> record = std::make_shared<LdsRecord>();
  record->source = context.source_;
  record->section = facts.section;
  record->owner = facts.owner;
  const uint32_t granulated =
      RJ_AMDHSA_BITS_GET(facts.descriptor.descriptor.compute_pgm_rsrc1,
                         amdhsa::COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT);
  record->allocated_vgprs =
      (granulated + 1) * descriptor_vgpr_granularity_for_wavefront(target_profile->arch, wave);
  if (target_profile->arch == IsaFamily::Cdna3 || target_profile->arch == IsaFamily::Cdna4) {
    // The unified allocation's ordinary prefix ends where the accumulator window begins.
    const uint32_t accum_base = (RJ_AMDHSA_BITS_GET(facts.descriptor.descriptor.compute_pgm_rsrc3,
                                                    amdhsa::COMPUTE_PGM_RSRC3_GFX90A_ACCUM_OFFSET) +
                                 1) *
                                4;
    record->allocated_vgprs = std::min(record->allocated_vgprs, accum_base);
  }
  std::unique_ptr<Gfx1250VgprMsbAnalysis> banks;
  if (cdna5) {
    const IsaTargetBinding *binding = default_isa_target_registry().find_target(target);
    const std::vector<BasicBlock *>::const_iterator entry_block =
        std::ranges::find_if(facts.blocks, [&](const BasicBlock *block) {
          return block->start_offset() == facts.owner.entry_offset;
        });
    if (!binding || entry_block == facts.blocks.end()) {
      return reject_fault_mutation(error, FaultMutationErrorCode::UnsupportedForm,
                                   "missing LDS bank analysis entry");
    }
    const std::span<const uint8_t> text =
        context.image().subspan(facts.section.file_offset, facts.section.size);
    banks = std::make_unique<Gfx1250VgprMsbAnalysis>(
        facts.blocks, *entry_block, std::span<const ScopedCfgEdge>{}, text,
        std::span<BasicBlock *const>{}, binding->capabilities.setreg_vgpr_msb_fixup);
  }
  for (const BasicBlock *block : facts.blocks) {
    for (const Instruction &inst : block->instructions()) {
      const LdsProfile *profile = find_lds_profile(target, inst.mnemonic());
      if (!profile || inst.size() != 8 || !inst.src_operand(0)) {
        continue;
      }
      const std::optional<RegisterRef> address = inst.src_operand(0)->decoded_vgpr();
      if (!address) {
        continue;
      }
      uint32_t bank = 0;
      if (cdna5) {
        const amdgpu::VgprMsbRole role = inst.src_operand(0)->vgpr_msb_role();
        if (role != amdgpu::VgprMsbRole::Src0) {
          continue;
        }
        const std::optional<uint8_t> selected_bank = banks->bank_before(inst, role);
        if (!selected_bank) {
          continue;
        }
        bank = *selected_bank;
      }
      if (cdna5 && bank * 256u + address->index >= record->allocated_vgprs) {
        continue;
      }
      const InstructionBoundaries::Entry *boundary = facts.boundaries->find(inst.src_loc());
      if (!boundary || boundary->size != 8 || boundary->clause || boundary->clause_member) {
        continue;
      }
      const std::span<const uint8_t> image = context.image();
      const uint64_t offset = inst.src_loc();
      if (facts.section.file_offset > image.size() ||
          offset > image.size() - facts.section.file_offset ||
          8 > image.size() - facts.section.file_offset - offset || offset > facts.section.size ||
          8 > facts.section.size - offset) {
        continue;
      }
      std::array<uint32_t, 2> words = {};
      std::memcpy(words.data(), image.data() + facts.section.file_offset + offset, 8);
      if ((words[1] & 255u) != address->index ||
          !replace_lds_address(*profile, words, address->index)) {
        continue;
      }
      record->sites.push_back({.offset = inst.src_loc(),
                               .address_vgpr = address->index,
                               .address_bank = bank,
                               .form = profile->form});
    }
  }
  std::ranges::sort(record->sites, {}, &LdsSite::offset);
  LdsInventory inventory;
  inventory.record_ = std::move(record);
  return inventory;
}

FailureOr<LdsSelection> select_lds_wrong_address(const LdsInventory &inventory,
                                                 const LdsSelector &selector,
                                                 FaultMutationError &error) {
  error = {};
  if (!inventory.record_ || inventory.record_->source.expired()) {
    return reject_fault_mutation(error, FaultMutationErrorCode::BadIdentity,
                                 "source context expired");
  }
  if (selector.exact) {
    const LdsSelection &exact = *selector.exact;
    if (!exact.record_ || exact.record_->source.lock() != inventory.record_->source.lock() ||
        exact.record_->owner != inventory.record_->owner ||
        std::ranges::none_of(inventory.record_->sites, [&](const LdsSite &site) {
          return site.offset == exact.site_.offset &&
                 site.address_vgpr == exact.site_.address_vgpr &&
                 site.address_bank == exact.site_.address_bank && site.form == exact.site_.form;
        })) {
      return reject_fault_mutation(error, FaultMutationErrorCode::BadIdentity,
                                   "exact LDS selection does not match source and owner");
    }
    return exact;
  }
  if (inventory.record_->sites.empty()) {
    return reject_fault_mutation(error, FaultMutationErrorCode::NoCandidates,
                                 "no supported LDS site");
  }
  if (selector.ordinal >= inventory.record_->sites.size()) {
    return reject_fault_mutation(error, FaultMutationErrorCode::OrdinalOutOfRange,
                                 "LDS ordinal out of range");
  }
  LdsSelection selection;
  selection.record_ = inventory.record_;
  selection.site_ = inventory.record_->sites[selector.ordinal];
  return selection;
}

FailureOr<LdsResult> apply_lds_wrong_address(const FaultMutationContext &context,
                                             const LdsSelection &selection,
                                             std::optional<uint32_t> replacement_selector,
                                             FaultMutationError &error) {
  error = {};
  if (!selection.record_ || selection.record_->source.lock() != context.source_) {
    return reject_fault_mutation(error, FaultMutationErrorCode::BadIdentity,
                                 "selection belongs to another or expired source");
  }
  if (!replacement_selector || *replacement_selector > 255 ||
      *replacement_selector == selection.site_.address_vgpr ||
      selection.site_.address_bank * 256u + *replacement_selector >=
          selection.record_->allocated_vgprs) {
    return reject_fault_mutation(
        error, FaultMutationErrorCode::InvalidParameters,
        "replacement must be distinct, encoded, and allocated as an ordinary VGPR");
  }
  const uint64_t offset = selection.site_.offset;
  const std::span<const uint8_t> image = context.image();
  if (offset > selection.record_->section.size || 8 > selection.record_->section.size - offset ||
      selection.record_->section.file_offset > image.size() ||
      offset > image.size() - selection.record_->section.file_offset ||
      8 > image.size() - selection.record_->section.file_offset - offset) {
    return reject_fault_mutation(error, FaultMutationErrorCode::ValidationFailure,
                                 "invalid source extent", offset);
  }
  const uint64_t file_offset = selection.record_->section.file_offset + offset;
  std::array<uint32_t, 2> original = {};
  std::memcpy(original.data(), image.data() + file_offset, 8);
  std::unique_ptr<Decoder> decoder =
      Decoder::create(default_isa_target_registry(), context.source().target_id());
  if (!decoder) {
    return reject_fault_mutation(error, FaultMutationErrorCode::ValidationFailure,
                                 "missing LDS decoder", offset);
  }
  const DecodeResult decoded = decoder->decode_window(original, offset);
  const LdsProfile *profile = find_lds_profile(context.source().target_id(), selection.site_.form);
  if (!profile || decoded.failed() || decoded.value()->mnemonic() != selection.site_.form ||
      decoded.value()->size() != 8) {
    return reject_fault_mutation(error, FaultMutationErrorCode::ValidationFailure, "stale LDS form",
                                 offset);
  }
  const std::optional<std::array<uint32_t, 2>> replacement =
      replace_lds_address(*profile, original, *replacement_selector);
  if (!replacement || (original[1] & 255u) != selection.site_.address_vgpr) {
    return reject_fault_mutation(error, FaultMutationErrorCode::ValidationFailure,
                                 "stale LDS source", offset);
  }
  FailureOr<std::unique_ptr<Instrumentor>> fork = context.fork_instrumentor(error);
  if (fork.failed()) {
    return Result::failure();
  }
  InstructionReplacement edit = {.expected_bytes = std::vector<uint8_t>(8),
                                 .replacement_bytes = std::vector<uint8_t>(8)};
  std::memcpy(edit.expected_bytes.data(), original.data(), 8);
  std::memcpy(edit.replacement_bytes.data(), replacement->data(), 8);
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
  if (!applied.errors.empty()) {
    return reject_fault_mutation(error, FaultMutationErrorCode::ValidationFailure,
                                 applied.errors.front(), offset);
  }
  return LdsResult{.selection = selection, .image = std::move(applied.elf_bytes)};
}
} // namespace rocjitsu
