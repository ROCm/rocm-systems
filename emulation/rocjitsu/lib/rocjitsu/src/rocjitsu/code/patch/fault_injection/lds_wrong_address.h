// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/patch/fault_injection/fault_mutation_context.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace rocjitsu {

/// @brief One eligible source instruction in .text-relative coordinates.
struct LdsSite {
  uint64_t offset = 0;
  uint32_t address_vgpr = 0; // Encoded low selector.
  uint32_t address_bank = 0; // Proven address-source bank.
  std::string_view form;     // Static admitted instruction form.
  std::vector<FaultMutationOwner> affected_owners;
};
struct LdsCapability {
  rj_code_target_id_t target;
  uint32_t wave_size;
  std::string_view form;
};
/// @brief Enumerate linked transformation profiles, without execution claims.
[[nodiscard]] std::vector<LdsCapability> lds_wrong_address_capabilities();
struct LdsRecord;
/// @brief An inventory that does not extend its source context's lifetime.
class LdsInventory {
public:
  std::span<const LdsSite> sites() const;

private:
  std::shared_ptr<const LdsRecord> record_;
  friend FailureOr<LdsInventory> discover_lds_wrong_address(const FaultMutationContext &,
                                                            std::string_view, FaultMutationError &);
  friend FailureOr<LdsSelection>
  select_lds_wrong_address(const LdsInventory &, const struct LdsSelector &, FaultMutationError &);
};
/// @brief An exact source-bound selection.
class LdsSelection {
public:
  FaultMutationKind kind() const { return FaultMutationKind::LdsWrongAddress; }
  const LdsSite &site() const { return site_; }
  const FaultMutationOwner *owner() const;

private:
  LdsSite site_;
  std::shared_ptr<const LdsRecord> record_;
  friend FailureOr<LdsSelection>
  select_lds_wrong_address(const LdsInventory &, const struct LdsSelector &, FaultMutationError &);
  friend FailureOr<struct LdsResult> apply_lds_wrong_address(const FaultMutationContext &,
                                                             const LdsSelection &,
                                                             std::optional<uint32_t>,
                                                             FaultMutationError &);
};
struct LdsSelector {
  // Exact takes precedence; invalid exact never falls back to ordinal.
  std::optional<LdsSelection> exact;
  size_t ordinal = 0;
};
struct LdsResult {
  LdsSelection selection;
  std::vector<uint8_t> image;
  uint32_t logical_mutation_count = 1;
};
/// @brief Discover supported LDS forms with complete affected-owner facts.
/// @details Clause members and incomplete instructions are not advertised. Successful
/// discovery does not guarantee apply: the shared replacement pipeline may reject
/// text growth that cannot preserve retained PC-relative references.
[[nodiscard]] FailureOr<LdsInventory>
discover_lds_wrong_address(const FaultMutationContext &, std::string_view, FaultMutationError &);
/// @brief Select an exact handle or a zero-based source-order ordinal.
[[nodiscard]] FailureOr<LdsSelection>
select_lds_wrong_address(const LdsInventory &, const LdsSelector &, FaultMutationError &);
/// @brief Emit one replacement from the pristine context using a distinct allocated VGPR.
/// @details The caller must initialize that register with an appropriate runtime address.
/// Static checks establish neither runtime execution nor detector qualification.
[[nodiscard]] FailureOr<LdsResult> apply_lds_wrong_address(const FaultMutationContext &,
                                                           const LdsSelection &,
                                                           std::optional<uint32_t>,
                                                           FaultMutationError &);

} // namespace rocjitsu
