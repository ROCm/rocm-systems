// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/fault_injection/drop_barrier.h"

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/code/patch/fault_injection/fault_mutation_internal.h"
#include "rocjitsu/code/patch/instrumentor.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/target_registry.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <utility>

namespace rocjitsu {

// Cached per-kernel qualification, owned by the context and any returned inventories.
// Its weak source link lets a retained inventory detect context expiry.
struct FaultBarrierRecord {
  FaultMutationSection section;
  FaultMutationOwner owner;
  std::vector<StaticBarrier> barriers;
  std::weak_ptr<const AmdGpuCodeObject> source;
  FaultMutationError error; // Kernel-local qualification failure; siblings may still be usable.
};

bool DropBarrierSelection::belongs_to(const FaultMutationContext &context) const {
  return source_.lock().get() == &context.source();
}

const FaultMutationSection &FaultBarrierInventory::section() const { return record_->section; }
const FaultMutationOwner &FaultBarrierInventory::owner() const { return record_->owner; }
std::span<const StaticBarrier> FaultBarrierInventory::barriers() const { return record_->barriers; }

FailureOr<DropBarrierSelection> select_drop_barrier(const FaultBarrierInventory &inventory,
                                                    const DropBarrierSelector &selector,
                                                    FaultMutationError &error) {
  error = {};
  const std::shared_ptr<const FaultBarrierRecord> &record = inventory.record_;
  if (!record || record->source.expired())
    return reject_fault_mutation(error, FaultMutationErrorCode::BadIdentity,
                                 "expired or unbound inventory");
  // TODO(hanchung): Add a better barrier-selection heuristic while preserving explicit selection
  // and reproducibility.
  if (selector.exact) {
    if (selector.exact->record_.lock() != record)
      return reject_fault_mutation(error, FaultMutationErrorCode::BadIdentity,
                                   "exact selector belongs to another source or kernel");
    return *selector.exact;
  }
  if (record->barriers.empty())
    return reject_fault_mutation(error, FaultMutationErrorCode::NoCandidates,
                                 "no complete qualified barriers");
  if (selector.ordinal >= record->barriers.size())
    return reject_fault_mutation(error, FaultMutationErrorCode::OrdinalOutOfRange,
                                 "logical barrier ordinal out of range");
  DropBarrierSelection selection;
  selection.owner_ = record->owner;
  selection.barrier_ = record->barriers[selector.ordinal];
  selection.source_ = record->source;
  selection.record_ = record;
  selection.ordinal_ = selector.ordinal;
  return selection;
}

std::vector<FaultMutationCapability> fault_mutation_capabilities() {
  const IsaTargetRegistry &registry = default_isa_target_registry();
  std::vector<FaultMutationCapability> capabilities;
  for (const IsaTargetDescriptor &family : registry.targets()) {
    for (const IsaTargetBinding &binding : family.bindings) {
      const std::optional<StaticBarrierPolicy> policy =
          static_barrier_policy(binding.public_id, registry);
      if (!policy || !Decoder::create(registry, binding.public_id))
        continue;
      for (const uint32_t wave : policy->wave_sizes)
        capabilities.push_back(
            {.kind = FaultMutationKind::DropBarrier,
             .form = policy->form,
             .architecture = family.family,
             .target = binding.public_id,
             .wave_size = wave,
             .admission = "shared DBI ELF admission; file-backed text/descriptors; exact kernel; "
                          "independent siblings; closed direct CFG; complete builtin barrier"});
    }
  }
  return capabilities;
}

void FaultMutationContext::qualify_drop() const {
  const rj_code_target_id_t target = source().target_id();
  const IsaTargetRegistry &registry = default_isa_target_registry();
  const IsaTargetDescriptor *target_facts = registry.find(target);
  const std::optional<StaticBarrierPolicy> policy = static_barrier_policy(target, registry);
  assert(target_facts && policy);
  for (const FaultSourceFacts &facts : state_->sources) {
    auto record = std::make_shared<FaultBarrierRecord>();
    record->section = facts.section;
    record->owner = facts.owner;
    record->source = source_;
    const uint32_t wave = kernel_wavefront_size(target_facts->family, facts.descriptor.descriptor);
    // CDNA3/4 normalize to Wave64 even when the raw Wave32 bit is set.
    // Check that bit against the recipe's allowed wave sizes independently.
    const bool descriptor_requests_wave32 =
        (facts.descriptor.descriptor.kernel_code_properties &
         amdhsa::KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32) != 0;
    if (std::ranges::find(policy->wave_sizes, wave) == policy->wave_sizes.end() ||
        (descriptor_requests_wave32 &&
         std::ranges::find(policy->wave_sizes, 32u) == policy->wave_sizes.end())) {
      record->error = {.code = FaultMutationErrorCode::UnsupportedForm,
                       .offset = 0,
                       .message = "descriptor wave mode is not admitted by this barrier family"};
    } else {
      StaticBarrierError barrier_error;
      FailureOr<std::vector<StaticBarrier>> barriers =
          qualify_static_barriers(facts.blocks, target, *facts.boundaries, barrier_error);
      if (barriers.failed()) {
        record->error = {.code = barrier_error.code == StaticBarrierErrorCode::IncompleteBarrier
                                     ? FaultMutationErrorCode::IncompleteSelection
                                     : FaultMutationErrorCode::UnsupportedForm,
                         .offset = barrier_error.offset,
                         .message = barrier_error.message};
      } else {
        record->barriers = std::move(barriers).value();
        std::ranges::sort(record->barriers, {}, [](const StaticBarrier &barrier) {
          return barrier.members.front().offset;
        });
      }
    }
    state_->kernels.push_back(std::move(record));
  }
}

FailureOr<FaultBarrierInventory> discover_drop_barriers(const FaultMutationContext &context,
                                                        std::string_view entry,
                                                        FaultMutationError &error) {
  error = {};
  // Preserve header and recipe-policy error precedence before structural analysis.
  const std::span<const uint8_t> image = context.image();
  Elf64_Ehdr header = {};
  if (validate_fault_mutation_source_header(image, header, error).failed())
    return Result::failure();
  const rj_code_target_id_t target = context.source().target_id();
  const IsaTargetRegistry &registry = default_isa_target_registry();
  const IsaTargetDescriptor *target_facts = registry.find(target);
  const IsaTargetBinding *binding = registry.find_target(target);
  if (!target_facts || !binding || !static_barrier_policy(target, registry) ||
      (header.e_flags & EF_AMDGPU_MACH) != binding->elf_machine ||
      (header.e_flags & ~uint32_t{0xfff}) != 0)
    return reject_fault_mutation(
        error, FaultMutationErrorCode::UnsupportedTarget,
        "requires a registered complete-barrier target with matching ELF flags");
  const FailureOr<const FaultSourceFacts *> facts = context.source_facts(entry, error);
  if (facts.failed())
    return Result::failure();
  std::call_once(context.state_->drop_once, [&] { context.qualify_drop(); });
  // First choose one exact kernel record; ambiguous names never select the first.
  // Logical-barrier ordering and exact-handle precedence apply within that record.
  const std::vector<std::shared_ptr<const FaultBarrierRecord>> &kernels = context.state_->kernels;
  const std::ptrdiff_t matches =
      std::ranges::count_if(kernels, [&](const std::shared_ptr<const FaultBarrierRecord> &record) {
        return record->owner.entry == entry;
      });
  if (matches > 1)
    return reject_fault_mutation(error, FaultMutationErrorCode::AmbiguousSelection,
                                 "exact entry name identifies multiple descriptors");
  if (entry.empty() || matches == 0) {
    std::string message = "entry must match one exact kernel name; descriptor entries:";
    for (const std::shared_ptr<const FaultBarrierRecord> &record : kernels)
      message += " " + record->owner.entry;
    return reject_fault_mutation(error, FaultMutationErrorCode::BadIdentity, std::move(message));
  }
  const std::shared_ptr<const FaultBarrierRecord> record = *std::ranges::find_if(
      kernels, [&](const std::shared_ptr<const FaultBarrierRecord> &candidate) {
        return candidate->owner.entry == entry;
      });
  if (record->error.code != FaultMutationErrorCode::None) {
    error = record->error;
    return Result::failure();
  }
  FaultBarrierInventory inventory;
  inventory.record_ = record;
  return inventory;
}

FailureOr<DropBarrierResult> apply_drop_barrier(const FaultMutationContext &context,
                                                const DropBarrierSelection &selection,
                                                FaultMutationError &error) {
  error = {};
  if (!selection.belongs_to(context))
    return reject_fault_mutation(error, FaultMutationErrorCode::BadIdentity,
                                 "selection belongs to another or expired source context");
  const std::shared_ptr<const FaultBarrierRecord> record = selection.record_.lock();
  // Only selection can bind the private record and validated ordinal. The matching
  // live context retains that immutable record throughout apply.
  assert(record && selection.ordinal_ < record->barriers.size());
  const StaticBarrier &barrier = record->barriers[selection.ordinal_];
  const IsaTargetDescriptor *target =
      default_isa_target_registry().find(context.source().target_id());
  // Only successful analyze/discover can produce a bound selection. That path
  // admits the immutable source target and initializes the shared source facts.
  assert(target && context.state_->instrumentor);
  FailureOr<std::unique_ptr<Instrumentor>> fork = context.fork_instrumentor(error);
  if (fork.failed())
    return Result::failure();
  std::unique_ptr<Instrumentor> instrumentor = std::move(fork).value();
  const FailureOr<uint32_t> encoded_nop = build_s_nop(0, target->family);
  if (encoded_nop.failed())
    return reject_fault_mutation(error, FaultMutationErrorCode::UnsupportedTarget,
                                 "target cannot encode the Drop replacement NOP");
  const uint32_t nop = encoded_nop.value();
  for (const StaticBarrierMember &member : barrier.members) {
    InstructionReplacement replacement = {.expected_bytes = member.bytes,
                                          .replacement_bytes =
                                              std::vector<uint8_t>(member.bytes.size())};
    std::memcpy(replacement.replacement_bytes.data(), &nop, sizeof(nop));
    InstrumentationPoint point = {.anchor_offset = member.offset,
                                  .kind = InstrumentationKind::ReplaceWithInst,
                                  .probe_symbol = {},
                                  .probe_args = {},
                                  .replacement = std::move(replacement)};
    instrumentor->add_point(std::move(point));
  }
  InstrumentedCodeObject applied = instrumentor->patch();
  if (applied.edit_error) {
    error = translate_fault_mutation_edit_error(*applied.edit_error);
    return Result::failure();
  }
  if (!applied.errors.empty()) {
    std::string message;
    for (const std::string &diagnostic : applied.errors) {
      if (!message.empty())
        message += "; ";
      message += diagnostic;
    }
    return reject_fault_mutation(error, FaultMutationErrorCode::ValidationFailure,
                                 std::move(message));
  }
  DropBarrierResult result = {
      .selection = selection,
      .image = std::move(applied.elf_bytes),
      .assumptions = {
          "Static mutation affects every executed occurrence of the selected logical barrier.",
          "Same-block membership does not prove barrier convergence.",
          "Execution and detector qualification have not been performed."}};
  if (barrier.scope == StaticBarrierScope::Cluster)
    result.assumptions.emplace_back(
        "Cluster barrier -3 assumes a clustered dispatch with multiple workgroups per cluster; "
        "it is inert otherwise. A valid pristine barrier requires exactly one signal per workgroup "
        "after workgroup synchronization, with every wave waiting. Static qualification does not "
        "verify workgroup size or this protocol.");
  return result;
}

} // namespace rocjitsu
