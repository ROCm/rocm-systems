// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/analysis/static_barriers.h"
#include "rocjitsu/code/patch/fault_injection/fault_mutation_context.h"
#include "rocjitsu/isa/isa_family.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace rocjitsu {

struct FaultBarrierRecord;

/// @brief Context-bound exact handle with inspectable selection facts.
/// @details Read-only fact snapshots belong to this handle; inspecting them does not
/// extend the source lifetime. References remain valid while this selection is alive.
/// Selection binds these facts to an immutable private record; apply uses that record
/// after checking source identity, without rediscovery.
class DropBarrierSelection {
public:
  FaultMutationKind kind() const { return kind_; }
  const FaultMutationOwner &owner() const { return owner_; }
  const StaticBarrier &barrier() const { return barrier_; }
  bool belongs_to(const FaultMutationContext &context) const;

private:
  FaultMutationKind kind_ = FaultMutationKind::DropBarrier;
  FaultMutationOwner owner_;
  StaticBarrier barrier_;
  std::weak_ptr<const AmdGpuCodeObject> source_;
  std::weak_ptr<const FaultBarrierRecord> record_;
  size_t ordinal_ = 0;
  friend FailureOr<DropBarrierSelection> select_drop_barrier(const FaultBarrierInventory &,
                                                             const struct DropBarrierSelector &,
                                                             FaultMutationError &);
  friend FailureOr<struct DropBarrierResult> apply_drop_barrier(const FaultMutationContext &,
                                                                const DropBarrierSelection &,
                                                                FaultMutationError &);
};

/// @brief Read-only qualified records; retaining an inventory does not extend source lifetime.
/// @details Accessor views borrow metadata retained by this inventory. Source expiry still
/// prevents selection and apply, even while those metadata views remain readable.
class FaultBarrierInventory {
public:
  const FaultMutationSection &section() const;
  const FaultMutationOwner &owner() const;
  std::span<const StaticBarrier> barriers() const;

private:
  FaultBarrierInventory() = default;
  std::shared_ptr<const FaultBarrierRecord> record_;
  friend FailureOr<FaultBarrierInventory>
  discover_drop_barriers(const FaultMutationContext &, std::string_view, FaultMutationError &);
  friend FailureOr<DropBarrierSelection> select_drop_barrier(const FaultBarrierInventory &,
                                                             const struct DropBarrierSelector &,
                                                             FaultMutationError &);
};

/// @brief Exact wins over ordinal; ordinal counts complete logical barriers, not physical sites.
struct DropBarrierSelector {
  std::optional<DropBarrierSelection> exact;
  size_t ordinal = 0;
};

/// @brief Select by exact handle or by zero-based source first-member order (default zero).
/// @details Invalid exact selectors fail without fallback, including wrong owner/context.
/// Default-first is deterministic, not a race heuristic or legacy physical-index adapter.
[[nodiscard]] FailureOr<DropBarrierSelection>
select_drop_barrier(const FaultBarrierInventory &inventory, const DropBarrierSelector &selector,
                    FaultMutationError &error);

/// @brief Result evidence state, not an admission policy or target authority.
enum class FaultQualification { Unverified, Validated };
/// @brief Listing of registry-derived transformation admission, not a second policy registry.
struct FaultMutationCapability {
  FaultMutationKind kind;
  StaticBarrierForm form;
  IsaFamily architecture;
  rj_code_target_id_t target;
  uint32_t wave_size;
  std::string_view admission;
};
/// @brief One logical Drop and its output evidence; emission alone leaves qualification unverified.
struct DropBarrierResult {
  DropBarrierSelection selection;
  std::vector<uint8_t> image; // Owned mutant ELF; ordered source members remain in selection.
  uint32_t logical_mutation_count = 1; // One split pair or singleton, independent of edit count.
  FaultQualification execution_validation = FaultQualification::Unverified;
  FaultQualification detector_qualification = FaultQualification::Unverified;
  std::vector<std::string> assumptions; // Mutation limits and unchecked runtime preconditions.
};

/// @brief Enumerate transformation admission from the registry used by discovery and apply.
/// @details Uses the default registry; no execution or detector claim.
[[nodiscard]] std::vector<FaultMutationCapability> fault_mutation_capabilities();

/// @brief Inventory one exact kernel, allowing independent siblings in one .text.
/// @details ABI admission follows the shared ELF reader and DBI patcher. Requires an admitted
/// family/target/wave, complete file-backed executable .text at its declared address, and loaded
/// descriptors resolving to the analyzed entries. Reject ambiguous entry names, malformed
/// descriptors, overlapping virtual load ranges, and additional full or partial file aliases
/// even at disjoint virtual addresses. Every owner's complete 64-byte descriptor must remain
/// immutable under supported dynamic loader relocation writes; static metadata and unrelated
/// writes retain shared admission. This loaded-ownership policy is specific to FI.
/// Reject unresolved/shared ownership. Multi-text,
/// helpers and indirect flow remain outside this slice. Admission is not runtime or
/// detector qualification.
/// Non-aliasing unselected executable sections are preserved after shared source checks;
/// additional allocated executable sections must decode completely. Declared executable
/// function/kernel owners outside .text require broader ownership analysis.
[[nodiscard]] FailureOr<FaultBarrierInventory>
discover_drop_barriers(const FaultMutationContext &context, std::string_view entry,
                       FaultMutationError &error);

/// @brief Resolve an immutable same-context record and emit its complete members.
/// @details Each member is replaced by a branch to a NOP trampoline and continuation.
/// ELF layout and descriptor entry displacements may change; selection offsets still
/// describe the pristine source. Branch-range or other emission failures return no
/// mutant and leave the source unchanged. Runtime checks input preconditions; recipe
/// output semantics are tested, not reanalyzed by a production candidate validator.
[[nodiscard]] FailureOr<DropBarrierResult> apply_drop_barrier(const FaultMutationContext &context,
                                                              const DropBarrierSelection &selection,
                                                              FaultMutationError &error);

} // namespace rocjitsu
