// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/fault_injection/atomic_wrong_address.h"

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/code/analysis/instruction_boundaries.h"
#include "rocjitsu/code/basic_block.h"
#include "rocjitsu/code/kernel_scope.h"
#include "rocjitsu/code/patch/fault_injection/atomic_legacy_fields.h"
#include "rocjitsu/code/patch/fault_injection/fault_mutation_internal.h"
#include "rocjitsu/code/patch/instrumentor.h"
#include "rocjitsu/isa/arch/amdgpu/shared/ds_offset_encoded_fields.h"
#include "rocjitsu/isa/arch/amdgpu/shared/memory_displacement_encoded_fields.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/isa/target_registry.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace rocjitsu {

struct AtomicRecord {
  std::weak_ptr<const AmdGpuCodeObject> source;
  FaultMutationSection section;
  FaultMutationOwner owner;
  std::vector<AtomicSite> sites;
  std::unordered_map<uint64_t, std::vector<uint8_t>> source_bytes;
};

const FaultMutationOwner *AtomicSelection::owner() const {
  return record_ ? &record_->owner : nullptr;
}

namespace {

constexpr std::array<rj_code_target_id_t, 7> kAtomicTargets = {
    ROCJITSU_CODE_TARGET_GFX1100, ROCJITSU_CODE_TARGET_GFX1200, ROCJITSU_CODE_TARGET_GFX1201,
    ROCJITSU_CODE_TARGET_GFX942,  ROCJITSU_CODE_TARGET_GFX950,  ROCJITSU_CODE_TARGET_GFX1250,
    ROCJITSU_CODE_TARGET_GFX1251};

bool modern_atomic_target(rj_code_target_id_t target) {
  return target == ROCJITSU_CODE_TARGET_GFX1200 || target == ROCJITSU_CODE_TARGET_GFX1201 ||
         target == ROCJITSU_CODE_TARGET_GFX1250 || target == ROCJITSU_CODE_TARGET_GFX1251;
}

bool flat_like_atomic(std::string_view form) {
  return form.starts_with("flat_atomic") || form.starts_with("global_atomic");
}

uint32_t modern_atomic_alignment(amdgpu::Operation operation) {
  // Alignment follows the memory element, not CAS's two-element input or the
  // scalar type of a packed operation. Unknown operations have no recipe.
  switch (operation) {
  case amdgpu::Operation::FLAT_ATOMIC_ADD_F32:
  case amdgpu::Operation::FLAT_ATOMIC_ADD_U32:
  case amdgpu::Operation::FLAT_ATOMIC_AND_B32:
  case amdgpu::Operation::FLAT_ATOMIC_CMPSWAP_B32:
  case amdgpu::Operation::FLAT_ATOMIC_COND_SUB_U32:
  case amdgpu::Operation::FLAT_ATOMIC_DEC_U32:
  case amdgpu::Operation::FLAT_ATOMIC_INC_U32:
  case amdgpu::Operation::FLAT_ATOMIC_MAX_I32:
  case amdgpu::Operation::FLAT_ATOMIC_MAX_NUM_F32:
  case amdgpu::Operation::FLAT_ATOMIC_MAX_U32:
  case amdgpu::Operation::FLAT_ATOMIC_MIN_I32:
  case amdgpu::Operation::FLAT_ATOMIC_MIN_NUM_F32:
  case amdgpu::Operation::FLAT_ATOMIC_MIN_U32:
  case amdgpu::Operation::FLAT_ATOMIC_OR_B32:
  case amdgpu::Operation::FLAT_ATOMIC_PK_ADD_BF16:
  case amdgpu::Operation::FLAT_ATOMIC_PK_ADD_F16:
  case amdgpu::Operation::FLAT_ATOMIC_SUB_CLAMP_U32:
  case amdgpu::Operation::FLAT_ATOMIC_SUB_U32:
  case amdgpu::Operation::FLAT_ATOMIC_SWAP_B32:
  case amdgpu::Operation::FLAT_ATOMIC_XOR_B32:
  case amdgpu::Operation::GLOBAL_ATOMIC_ADD_F32:
  case amdgpu::Operation::GLOBAL_ATOMIC_ADD_U32:
  case amdgpu::Operation::GLOBAL_ATOMIC_AND_B32:
  case amdgpu::Operation::GLOBAL_ATOMIC_CMPSWAP_B32:
  case amdgpu::Operation::GLOBAL_ATOMIC_COND_SUB_U32:
  case amdgpu::Operation::GLOBAL_ATOMIC_DEC_U32:
  case amdgpu::Operation::GLOBAL_ATOMIC_INC_U32:
  case amdgpu::Operation::GLOBAL_ATOMIC_MAX_I32:
  case amdgpu::Operation::GLOBAL_ATOMIC_MAX_NUM_F32:
  case amdgpu::Operation::GLOBAL_ATOMIC_MAX_U32:
  case amdgpu::Operation::GLOBAL_ATOMIC_MIN_I32:
  case amdgpu::Operation::GLOBAL_ATOMIC_MIN_NUM_F32:
  case amdgpu::Operation::GLOBAL_ATOMIC_MIN_U32:
  case amdgpu::Operation::GLOBAL_ATOMIC_OR_B32:
  case amdgpu::Operation::GLOBAL_ATOMIC_PK_ADD_BF16:
  case amdgpu::Operation::GLOBAL_ATOMIC_PK_ADD_F16:
  case amdgpu::Operation::GLOBAL_ATOMIC_SUB_CLAMP_U32:
  case amdgpu::Operation::GLOBAL_ATOMIC_SUB_U32:
  case amdgpu::Operation::GLOBAL_ATOMIC_SWAP_B32:
  case amdgpu::Operation::GLOBAL_ATOMIC_XOR_B32:
  case amdgpu::Operation::BUFFER_ATOMIC_ADD_F32:
  case amdgpu::Operation::BUFFER_ATOMIC_ADD_U32:
  case amdgpu::Operation::BUFFER_ATOMIC_AND_B32:
  case amdgpu::Operation::BUFFER_ATOMIC_CMPSWAP_B32:
  case amdgpu::Operation::BUFFER_ATOMIC_COND_SUB_U32:
  case amdgpu::Operation::BUFFER_ATOMIC_DEC_U32:
  case amdgpu::Operation::BUFFER_ATOMIC_INC_U32:
  case amdgpu::Operation::BUFFER_ATOMIC_MAX_I32:
  case amdgpu::Operation::BUFFER_ATOMIC_MAX_NUM_F32:
  case amdgpu::Operation::BUFFER_ATOMIC_MAX_U32:
  case amdgpu::Operation::BUFFER_ATOMIC_MIN_I32:
  case amdgpu::Operation::BUFFER_ATOMIC_MIN_NUM_F32:
  case amdgpu::Operation::BUFFER_ATOMIC_MIN_U32:
  case amdgpu::Operation::BUFFER_ATOMIC_OR_B32:
  case amdgpu::Operation::BUFFER_ATOMIC_PK_ADD_BF16:
  case amdgpu::Operation::BUFFER_ATOMIC_PK_ADD_F16:
  case amdgpu::Operation::BUFFER_ATOMIC_SUB_CLAMP_U32:
  case amdgpu::Operation::BUFFER_ATOMIC_SUB_U32:
  case amdgpu::Operation::BUFFER_ATOMIC_SWAP_B32:
  case amdgpu::Operation::BUFFER_ATOMIC_XOR_B32:
    return 4;
  case amdgpu::Operation::FLAT_ATOMIC_ADD_F64:
  case amdgpu::Operation::FLAT_ATOMIC_ADD_U64:
  case amdgpu::Operation::FLAT_ATOMIC_AND_B64:
  case amdgpu::Operation::FLAT_ATOMIC_CMPSWAP_B64:
  case amdgpu::Operation::FLAT_ATOMIC_DEC_U64:
  case amdgpu::Operation::FLAT_ATOMIC_INC_U64:
  case amdgpu::Operation::FLAT_ATOMIC_MAX_I64:
  case amdgpu::Operation::FLAT_ATOMIC_MAX_NUM_F64:
  case amdgpu::Operation::FLAT_ATOMIC_MAX_U64:
  case amdgpu::Operation::FLAT_ATOMIC_MIN_I64:
  case amdgpu::Operation::FLAT_ATOMIC_MIN_NUM_F64:
  case amdgpu::Operation::FLAT_ATOMIC_MIN_U64:
  case amdgpu::Operation::FLAT_ATOMIC_OR_B64:
  case amdgpu::Operation::FLAT_ATOMIC_SUB_U64:
  case amdgpu::Operation::FLAT_ATOMIC_SWAP_B64:
  case amdgpu::Operation::FLAT_ATOMIC_XOR_B64:
  case amdgpu::Operation::GLOBAL_ATOMIC_ADD_F64:
  case amdgpu::Operation::GLOBAL_ATOMIC_ADD_U64:
  case amdgpu::Operation::GLOBAL_ATOMIC_AND_B64:
  case amdgpu::Operation::GLOBAL_ATOMIC_CMPSWAP_B64:
  case amdgpu::Operation::GLOBAL_ATOMIC_DEC_U64:
  case amdgpu::Operation::GLOBAL_ATOMIC_INC_U64:
  case amdgpu::Operation::GLOBAL_ATOMIC_MAX_I64:
  case amdgpu::Operation::GLOBAL_ATOMIC_MAX_NUM_F64:
  case amdgpu::Operation::GLOBAL_ATOMIC_MAX_U64:
  case amdgpu::Operation::GLOBAL_ATOMIC_MIN_I64:
  case amdgpu::Operation::GLOBAL_ATOMIC_MIN_NUM_F64:
  case amdgpu::Operation::GLOBAL_ATOMIC_MIN_U64:
  case amdgpu::Operation::GLOBAL_ATOMIC_ORDERED_ADD_B64:
  case amdgpu::Operation::GLOBAL_ATOMIC_OR_B64:
  case amdgpu::Operation::GLOBAL_ATOMIC_SUB_U64:
  case amdgpu::Operation::GLOBAL_ATOMIC_SWAP_B64:
  case amdgpu::Operation::GLOBAL_ATOMIC_XOR_B64:
  case amdgpu::Operation::BUFFER_ATOMIC_ADD_F64:
  case amdgpu::Operation::BUFFER_ATOMIC_ADD_U64:
  case amdgpu::Operation::BUFFER_ATOMIC_AND_B64:
  case amdgpu::Operation::BUFFER_ATOMIC_CMPSWAP_B64:
  case amdgpu::Operation::BUFFER_ATOMIC_DEC_U64:
  case amdgpu::Operation::BUFFER_ATOMIC_INC_U64:
  case amdgpu::Operation::BUFFER_ATOMIC_MAX_I64:
  case amdgpu::Operation::BUFFER_ATOMIC_MAX_NUM_F64:
  case amdgpu::Operation::BUFFER_ATOMIC_MAX_U64:
  case amdgpu::Operation::BUFFER_ATOMIC_MIN_I64:
  case amdgpu::Operation::BUFFER_ATOMIC_MIN_NUM_F64:
  case amdgpu::Operation::BUFFER_ATOMIC_MIN_U64:
  case amdgpu::Operation::BUFFER_ATOMIC_OR_B64:
  case amdgpu::Operation::BUFFER_ATOMIC_SUB_U64:
  case amdgpu::Operation::BUFFER_ATOMIC_SWAP_B64:
  case amdgpu::Operation::BUFFER_ATOMIC_XOR_B64:
    return 8;
  default:
    return 0;
  }
}

bool admitted_modern_atomic_form(std::string_view form, std::optional<int32_t> byte_offset) {
  if (!byte_offset)
    return false;
  if (flat_like_atomic(form))
    return true;
  if (!form.starts_with("buffer_atomic"))
    return false;
  // Both modern families require nonnegative VBUFFER immediates despite the signed raw field.
  return *byte_offset >= 0;
}

bool atomic_instruction(std::string_view form) {
  if (flat_like_atomic(form) || form.starts_with("scratch_atomic") ||
      form.starts_with("buffer_atomic") || form.starts_with("s_atomic"))
    return true;
  constexpr std::array<std::string_view, 19> ds_prefixes = {
      "ds_add",      "ds_sub",       "ds_rsub",   "ds_inc",       "ds_dec",
      "ds_min",      "ds_max",       "ds_and",    "ds_or",        "ds_xor",
      "ds_mskor",    "ds_cmpstore",  "ds_cmpst",  "ds_storexchg", "ds_condxchg",
      "ds_cond_sub", "ds_sub_clamp", "ds_pk_add", "ds_wrxchg"};
  return std::ranges::any_of(ds_prefixes,
                             [&](std::string_view prefix) { return form.starts_with(prefix); });
}

bool ds_atomic32(std::string_view form) {
  if (!form.starts_with("ds_") || !atomic_instruction(form))
    return false;
  // Preserve the prototype's width admission, including its stride64 exchange exclusion.
  for (std::string_view wider : {"128", "96", "64", "x4", "x3", "x2"}) {
    if (form.find(wider) != std::string_view::npos)
      return false;
  }
  return form.find("32") != std::string_view::npos || form.find("dword") != std::string_view::npos;
}

std::optional<int32_t> read_atomic_offset(rj_code_target_id_t target, std::string_view form,
                                          std::span<const uint32_t> words) {
  if (words.size() == 2 && form.starts_with("ds_")) {
    const std::optional<uint32_t> offset = amdgpu::read_ds_offset0_field(target, words);
    return offset ? std::optional<int32_t>{static_cast<int32_t>(*offset)} : std::nullopt;
  }
  if (words.size() == 2)
    return fault_mutation_detail::read_legacy_atomic_byte_offset(target, form, words);
  return amdgpu::read_vmem_signed24_offset(target, words);
}

bool admitted_atomic_form(rj_code_target_id_t target, std::string_view form, uint32_t size,
                          std::optional<int32_t> offset) {
  if (!offset)
    return false;
  if (size == 8 && ds_atomic32(form))
    return true;
  if (size == 8 && fault_mutation_detail::legacy_atomic_target(target) &&
      (flat_like_atomic(form) || form.starts_with("buffer_atomic")))
    return true;
  return size == 12 && modern_atomic_target(target) && admitted_modern_atomic_form(form, offset);
}

std::optional<std::vector<uint32_t>> replace_atomic_offset(rj_code_target_id_t target,
                                                           std::string_view form,
                                                           std::span<const uint32_t> words,
                                                           int64_t offset) {
  if (words.size() == 2 && ds_atomic32(form)) {
    if (offset < 0 || offset > 255 || offset % 4 != 0)
      return std::nullopt;
    const std::optional<std::array<uint32_t, 2>> replacement =
        amdgpu::replace_ds_offset0_field(target, words, static_cast<uint32_t>(offset));
    if (!replacement)
      return std::nullopt;
    return std::vector<uint32_t>(replacement->begin(), replacement->end());
  }
  if (words.size() == 2) {
    const std::optional<std::array<uint32_t, 2>> replacement =
        fault_mutation_detail::replace_legacy_atomic_byte_offset(target, form, words, offset);
    if (!replacement)
      return std::nullopt;
    return std::vector<uint32_t>(replacement->begin(), replacement->end());
  }
  const std::optional<std::array<uint32_t, 3>> replacement =
      amdgpu::replace_vmem_signed24_offset(target, words, offset);
  if (!replacement)
    return std::nullopt;
  return std::vector<uint32_t>(replacement->begin(), replacement->end());
}

bool supported_wave(IsaFamily arch, const KernelDescriptorInfo &kernel) {
  const uint32_t wave = kernel_wavefront_size(arch, kernel.descriptor);
  if (arch == IsaFamily::Rdna3 || arch == IsaFamily::Rdna4)
    return wave == 32 || wave == 64;
  if (arch == IsaFamily::Cdna5)
    return wave == 32;
  return wave == 64 && (kernel.descriptor.kernel_code_properties &
                        amdhsa::KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32) == 0;
}

} // namespace

std::vector<AtomicCapability> atomic_wrong_address_capabilities() {
  const IsaTargetRegistry &registry = default_isa_target_registry();
  std::vector<AtomicCapability> capabilities;
  for (rj_code_target_id_t target : kAtomicTargets) {
    if (!registry.find_target(target) || !Decoder::create(registry, target))
      continue;
    const bool cdna5 =
        target == ROCJITSU_CODE_TARGET_GFX1250 || target == ROCJITSU_CODE_TARGET_GFX1251;
    const bool rdna = target == ROCJITSU_CODE_TARGET_GFX1100 ||
                      target == ROCJITSU_CODE_TARGET_GFX1200 ||
                      target == ROCJITSU_CODE_TARGET_GFX1201;
    for (std::string_view form : {"flat_atomic", "global_atomic", "buffer_atomic", "ds_atomic"}) {
      if (!modern_atomic_target(target) && !fault_mutation_detail::legacy_atomic_target(target) &&
          form != "ds_atomic")
        continue;
      capabilities.push_back({.target = target, .wave_size = cdna5 ? 32u : 64u, .form = form});
      if (rdna)
        capabilities.push_back({.target = target, .wave_size = 32u, .form = form});
    }
  }
  return capabilities;
}

std::span<const AtomicSite> AtomicInventory::sites() const {
  return record_ ? std::span<const AtomicSite>(record_->sites) : std::span<const AtomicSite>();
}

FailureOr<AtomicInventory> discover_atomic_wrong_address(const FaultMutationContext &context,
                                                         std::string_view entry,
                                                         FaultMutationError &error) {
  error = {};
  const rj_code_target_id_t target = context.source().target_id();
  const IsaTargetDescriptor *target_facts = default_isa_target_registry().find(target);
  if (std::ranges::find(kAtomicTargets, target) == kAtomicTargets.end() || !target_facts)
    return reject_fault_mutation(error, FaultMutationErrorCode::UnsupportedTarget,
                                 "target has no atomic inventory profile");
  FailureOr<const FaultSourceFacts *> found = context.source_facts(entry, error);
  const KernelOwnership *ownership = nullptr;
  std::optional<std::span<const KernelDescriptorInfo>> owner_kernels;
  std::optional<KernelOwnerFlow> selected_flow;
  size_t selected_owner = 0;
  FaultSourceFacts shared_facts;
  if (found.failed()) {
    if (!context.state_->complete_ownership_needed)
      return Result::failure();
    FailureOr<const KernelOwnership *> owner_result = context.execution_ownership(error);
    if (owner_result.failed())
      return Result::failure();
    ownership = owner_result.value();
    owner_kernels = ownership->kernels();
    if (!owner_kernels)
      return reject_fault_mutation(error, FaultMutationErrorCode::BadIdentity,
                                   "expired atomic owner facts");
    const std::ptrdiff_t matches =
        std::ranges::count_if(*owner_kernels, [&](const KernelDescriptorInfo &kernel) {
          return kernel.kernel_name == entry;
        });
    if (matches == 0) {
      std::string message = "entry must match one exact kernel name; descriptor entries:";
      for (const KernelDescriptorInfo &kernel : *owner_kernels)
        message += " " + kernel.kernel_name;
      return reject_fault_mutation(error, FaultMutationErrorCode::BadIdentity, std::move(message));
    }
    if (matches > 1)
      return reject_fault_mutation(error, FaultMutationErrorCode::AmbiguousSelection,
                                   "exact entry name identifies multiple descriptors");
    const std::span<const KernelDescriptorInfo>::iterator kernel =
        std::ranges::find_if(*owner_kernels, [&](const KernelDescriptorInfo &item) {
          return item.kernel_name == entry;
        });
    selected_owner = static_cast<size_t>(kernel - owner_kernels->begin());
    selected_flow = ownership->owner_flow(selected_owner);
    if (!selected_flow)
      return reject_fault_mutation(error, FaultMutationErrorCode::BadIdentity,
                                   "missing atomic owner flow");
    const Section &section = *context.source().text_sections().front();
    TextEditError edit_error;
    const InstructionBoundaries *boundaries =
        context.state_->instrumentor->instruction_boundaries(edit_error);
    if (!boundaries) {
      error = translate_fault_mutation_edit_error(edit_error);
      return Result::failure();
    }
    shared_facts = {.section = {.index = *section.sectionHeaderIndex(),
                                .file_offset = section.sectionOffset(),
                                .vaddr = section.vaddr(),
                                .size = section.size()},
                    .owner = {.entry = kernel->kernel_name,
                              .entry_offset = kernel->entry_text_offset,
                              .descriptor_file_offset = kernel->descriptor_file_offset},
                    .descriptor = *kernel,
                    .blocks = {selected_flow->blocks.begin(), selected_flow->blocks.end()},
                    .boundaries = boundaries};
  }
  error = {};
  const FaultSourceFacts &facts = ownership ? shared_facts : *found.value();
  if (!supported_wave(target_facts->family, facts.descriptor))
    return reject_fault_mutation(error, FaultMutationErrorCode::UnsupportedForm,
                                 "unsupported atomic wave size");
  std::shared_ptr<AtomicRecord> record = std::make_shared<AtomicRecord>();
  record->source = context.source_;
  record->section = facts.section;
  record->owner = facts.owner;
  const std::span<const uint8_t> image = context.image();
  for (const BasicBlock *block : facts.blocks) {
    for (const Instruction &instruction : block->instructions()) {
      if (instruction.amdgpu_memory_kind() != amdgpu::MemoryKind::Atomic &&
          !atomic_instruction(instruction.mnemonic()))
        continue;
      const uint64_t offset = instruction.src_loc();
      const uint32_t size = instruction.size();
      if (size == 0 || size % sizeof(uint32_t) != 0 || offset > facts.section.size ||
          size > facts.section.size - offset || facts.section.file_offset > image.size() ||
          offset > image.size() - facts.section.file_offset ||
          size > image.size() - facts.section.file_offset - offset)
        return reject_fault_mutation(error, FaultMutationErrorCode::ValidationFailure,
                                     "invalid atomic source extent", offset);
      std::vector<FaultMutationOwner> affected;
      if (ownership) {
        const std::optional<std::span<const KernelExecutionOwner>> owners =
            ownership->owners_at(offset);
        if (!owners || std::ranges::none_of(*owners, [&](const KernelExecutionOwner &owner) {
              return owner.kernel_index == selected_owner;
            }))
          return reject_fault_mutation(error, FaultMutationErrorCode::BadIdentity,
                                       "atomic site has incomplete execution owners", offset);
        for (const KernelExecutionOwner &owner : *owners) {
          if (owner.kernel_index >= owner_kernels->size())
            return reject_fault_mutation(error, FaultMutationErrorCode::BadIdentity,
                                         "atomic owner index is out of range", offset);
          const KernelDescriptorInfo &kernel = (*owner_kernels)[owner.kernel_index];
          if (!supported_wave(target_facts->family, kernel))
            return reject_fault_mutation(error, FaultMutationErrorCode::UnsupportedForm,
                                         "unsupported affected atomic owner wave", offset);
          affected.push_back({.entry = kernel.kernel_name,
                              .entry_offset = kernel.entry_text_offset,
                              .descriptor_file_offset = kernel.descriptor_file_offset});
        }
      } else {
        affected.push_back(facts.owner);
      }
      std::vector<uint8_t> original(size);
      std::memcpy(original.data(), image.data() + facts.section.file_offset + offset, size);
      std::vector<uint32_t> words(size / sizeof(uint32_t));
      std::memcpy(words.data(), original.data(), size);
      const std::optional<int32_t> byte_offset =
          read_atomic_offset(target, instruction.mnemonic(), words);
      const InstructionBoundaries::Entry *boundary = facts.boundaries->find(offset);
      const bool supported =
          admitted_atomic_form(target, instruction.mnemonic(), size, byte_offset) &&
          (ds_atomic32(instruction.mnemonic()) ||
           modern_atomic_alignment(instruction.amdgpu_operation()) != 0) &&
          boundary && boundary->size == size && !boundary->clause && !boundary->clause_member;
      record->source_bytes.emplace(offset, std::move(original));
      record->sites.push_back({.offset = offset,
                               .instruction_size = size,
                               .form = std::string(instruction.mnemonic()),
                               .byte_offset = byte_offset,
                               .address_supported = supported,
                               .affected_owners = std::move(affected)});
    }
  }
  std::ranges::sort(record->sites, {}, &AtomicSite::offset);
  if (std::ranges::adjacent_find(record->sites, {}, &AtomicSite::offset) != record->sites.end())
    return reject_fault_mutation(error, FaultMutationErrorCode::AmbiguousSelection,
                                 "atomic inventory contains duplicate source sites");
  AtomicInventory inventory;
  inventory.record_ = std::move(record);
  return inventory;
}

FailureOr<AtomicSelection> select_atomic_wrong_address(const AtomicInventory &inventory,
                                                       const AtomicSelector &selector,
                                                       FaultMutationError &error) {
  error = {};
  if (!inventory.record_ || inventory.record_->source.expired())
    return reject_fault_mutation(error, FaultMutationErrorCode::BadIdentity,
                                 "source context expired");
  if (selector.exact) {
    const AtomicSelection &exact = *selector.exact;
    if (!exact.record_ || exact.record_->source.lock() != inventory.record_->source.lock() ||
        std::ranges::none_of(inventory.record_->sites, [&](const AtomicSite &site) {
          return site == exact.site_ &&
                 std::ranges::find(site.affected_owners, inventory.record_->owner) !=
                     site.affected_owners.end();
        }))
      return reject_fault_mutation(error, FaultMutationErrorCode::BadIdentity,
                                   "exact atomic selection does not match source and owner");
    AtomicSelection selection;
    selection.record_ = inventory.record_;
    selection.site_ = exact.site_;
    return selection;
  }
  if (inventory.record_->sites.empty())
    return reject_fault_mutation(error, FaultMutationErrorCode::NoCandidates,
                                 "no owned atomic site");
  if (selector.ordinal >= inventory.record_->sites.size())
    return reject_fault_mutation(error, FaultMutationErrorCode::OrdinalOutOfRange,
                                 "atomic ordinal out of range");
  AtomicSelection selection;
  selection.record_ = inventory.record_;
  selection.site_ = inventory.record_->sites[selector.ordinal];
  return selection;
}

FailureOr<AtomicResult> apply_atomic_wrong_address(const FaultMutationContext &context,
                                                   const AtomicSelection &selection,
                                                   std::optional<uint32_t> byte_delta,
                                                   FaultMutationError &error) {
  error = {};
  if (!selection.record_ || selection.record_->source.lock() != context.source_ ||
      std::ranges::find(selection.record_->sites, selection.site_) ==
          selection.record_->sites.end())
    return reject_fault_mutation(error, FaultMutationErrorCode::BadIdentity,
                                 "selection belongs to another or expired atomic source");
  const uint64_t offset = selection.site_.offset;
  if (!byte_delta || *byte_delta == 0 || *byte_delta % sizeof(uint32_t) != 0 ||
      *byte_delta > 0x7fffffu)
    return reject_fault_mutation(
        error, FaultMutationErrorCode::InvalidParameters,
        "atomic byte delta must be positive, four-byte aligned, and at most 0x7fffff", offset);
  const uint32_t size = selection.site_.instruction_size;
  if (!selection.site_.address_supported ||
      !admitted_atomic_form(context.source().target_id(), selection.site_.form, size,
                            selection.site_.byte_offset))
    return reject_fault_mutation(error, FaultMutationErrorCode::UnsupportedForm,
                                 "selected atomic form has no address recipe", offset);
  const std::unordered_map<uint64_t, std::vector<uint8_t>>::const_iterator expected =
      selection.record_->source_bytes.find(offset);
  const std::span<const uint8_t> image = context.image();
  if (expected == selection.record_->source_bytes.end() || expected->second.size() != size ||
      offset > selection.record_->section.size || size > selection.record_->section.size - offset ||
      selection.record_->section.file_offset > image.size() ||
      offset > image.size() - selection.record_->section.file_offset ||
      size > image.size() - selection.record_->section.file_offset - offset)
    return reject_fault_mutation(error, FaultMutationErrorCode::ValidationFailure,
                                 "invalid atomic extent", offset);
  const uint8_t *source = image.data() + selection.record_->section.file_offset + offset;
  if (std::memcmp(source, expected->second.data(), size) != 0)
    return reject_fault_mutation(error, FaultMutationErrorCode::BadIdentity,
                                 "atomic source bytes changed", offset);
  std::vector<uint32_t> original(size / sizeof(uint32_t));
  std::memcpy(original.data(), source, size);
  std::unique_ptr<Decoder> decoder =
      Decoder::create(default_isa_target_registry(), context.source().target_id());
  if (!decoder)
    return reject_fault_mutation(error, FaultMutationErrorCode::ValidationFailure,
                                 "missing atomic decoder", offset);
  const DecodeResult decoded = decoder->decode_window(original, offset);
  const std::optional<int32_t> original_offset =
      read_atomic_offset(context.source().target_id(), selection.site_.form, original);
  if (decoded.failed() || static_cast<uint32_t>(decoded.value()->size()) != size ||
      decoded.value()->mnemonic() != selection.site_.form || !original_offset ||
      original_offset != selection.site_.byte_offset)
    return reject_fault_mutation(error, FaultMutationErrorCode::BadIdentity,
                                 "stale atomic form or offset", offset);
  const uint32_t alignment = ds_atomic32(selection.site_.form)
                                 ? 4u
                                 : modern_atomic_alignment(decoded.value()->amdgpu_operation());
  if (alignment == 0)
    return reject_fault_mutation(error, FaultMutationErrorCode::UnsupportedForm,
                                 "selected atomic operation has no alignment recipe", offset);
  if (*byte_delta % alignment != 0)
    return reject_fault_mutation(error, FaultMutationErrorCode::InvalidParameters,
                                 "atomic byte delta must preserve the selected operation's " +
                                     std::to_string(alignment) + "-byte natural alignment",
                                 offset);
  const int64_t replacement_offset = static_cast<int64_t>(*original_offset) + *byte_delta;
  const std::optional<std::vector<uint32_t>> replacement = replace_atomic_offset(
      context.source().target_id(), selection.site_.form, original, replacement_offset);
  if (!replacement)
    return reject_fault_mutation(error, FaultMutationErrorCode::InvalidParameters,
                                 "atomic delta exceeds the form's offset range or alignment",
                                 offset);
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
  return AtomicResult{.selection = selection, .image = std::move(applied.elf_bytes)};
}

} // namespace rocjitsu
