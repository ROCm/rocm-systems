// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/patch/fault_injection/fault_mutation_context.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace rocjitsu {

/// @brief One owned atomic instruction in pristine .text-relative coordinates.
/// @details The inventory includes forms without an address recipe, so adding
/// mutation support does not renumber the source-order atomic ordinal.
struct AtomicSite {
  uint64_t offset = 0;
  uint32_t instruction_size = 0;
  std::string form;
  std::optional<int32_t> byte_offset;
  bool address_supported = false;
  std::vector<FaultMutationOwner> affected_owners;
  bool operator==(const AtomicSite &) const = default;
};
struct AtomicCapability {
  rj_code_target_id_t target;
  uint32_t wave_size;
  std::string_view form;
};
/// @brief Enumerate linked atomic-address transformation families, without execution claims.
[[nodiscard]] std::vector<AtomicCapability> atomic_wrong_address_capabilities();
struct AtomicRecord;
/// @brief An atomic inventory that does not extend its source context's lifetime.
class AtomicInventory {
public:
  std::span<const AtomicSite> sites() const;

private:
  std::shared_ptr<const AtomicRecord> record_;
  friend FailureOr<AtomicInventory> discover_atomic_wrong_address(const FaultMutationContext &,
                                                                  std::string_view,
                                                                  FaultMutationError &);
  friend FailureOr<AtomicSelection> select_atomic_wrong_address(const AtomicInventory &,
                                                                const struct AtomicSelector &,
                                                                FaultMutationError &);
};
/// @brief An exact atomic handle bound to one immutable source and selected owner.
class AtomicSelection {
public:
  FaultMutationKind kind() const { return FaultMutationKind::AtomicWrongAddress; }
  const AtomicSite &site() const { return site_; }
  const FaultMutationOwner *owner() const;

private:
  AtomicSite site_;
  std::shared_ptr<const AtomicRecord> record_;
  friend FailureOr<AtomicSelection> select_atomic_wrong_address(const AtomicInventory &,
                                                                const struct AtomicSelector &,
                                                                FaultMutationError &);
  friend FailureOr<struct AtomicResult> apply_atomic_wrong_address(const FaultMutationContext &,
                                                                   const AtomicSelection &,
                                                                   std::optional<uint32_t>,
                                                                   FaultMutationError &);
};
struct AtomicSelector {
  // Exact takes precedence; invalid exact never falls back to ordinal.
  std::optional<AtomicSelection> exact;
  size_t ordinal = 0;
};
struct AtomicResult {
  AtomicSelection selection;
  std::vector<uint8_t> image;
  uint32_t logical_mutation_count = 1;
};
/// @brief Discover all owned atomic sites, before address-form admission.
/// @details Entry names match exactly. Shared instructions retain all affected
/// owners. Unsupported forms remain selectable and fail explicitly on apply.
/// Successful discovery does not guarantee emission through the shared writer.
[[nodiscard]] FailureOr<AtomicInventory>
discover_atomic_wrong_address(const FaultMutationContext &, std::string_view, FaultMutationError &);
/// @brief Select an exact handle or a zero-based source-order atomic ordinal.
[[nodiscard]] FailureOr<AtomicSelection>
select_atomic_wrong_address(const AtomicInventory &, const AtomicSelector &, FaultMutationError &);
/// @brief Apply a positive, four-byte-aligned byte delta to one supported atomic.
/// @details The delta must not exceed 0x7fffff; the resulting offset must fit the
/// form's signed field. Modern Buffer atomics require a nonnegative source offset.
/// Each call produces an independent pristine-source mutant
/// and changes only the immediate; runtime address validity remains the caller's duty.
[[nodiscard]] FailureOr<AtomicResult> apply_atomic_wrong_address(const FaultMutationContext &,
                                                                 const AtomicSelection &,
                                                                 std::optional<uint32_t> byte_delta,
                                                                 FaultMutationError &);

} // namespace rocjitsu
