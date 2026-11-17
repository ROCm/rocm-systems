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

enum class OrdinaryAcquireProfile { Cdna5GlobalB32, Rdna4GlobalB32 };
enum class OrdinaryAcquirePath { SameBlock, ExactFallthroughJoin, ExactSelfLoopExit };

/// @brief One exact pristine instruction used by a bounded acquire proof.
struct OrdinaryAcquireMember {
  uint64_t offset = 0;
  uint32_t instruction_size = 0;
  std::string form;
  bool operator==(const OrdinaryAcquireMember &) const = default;
};

/// @brief One acquire-qualified load and its exact cache/wait association.
/// @details All offsets refer to the context's pristine .text. Cache members
/// are an ordered list, not a fixed load/cache pair. Raw scope is present only when
/// the load encodes a scope field; cache instructions need not encode the same
/// raw scope. Legacy profiles define an implicit Agent scope and leave this
/// field absent; their cache modifiers are not scope-field encodings.
/// This query does not advertise any particular mutation or execution support.
struct OrdinaryAcquireSite {
  uint64_t offset = 0;
  uint32_t instruction_size = 0;
  std::string form;
  OrdinaryAcquireProfile profile = OrdinaryAcquireProfile::Cdna5GlobalB32;
  int32_t byte_offset = 0;
  std::optional<uint32_t> raw_scope;
  OrdinaryAcquirePath path = OrdinaryAcquirePath::SameBlock;
  uint64_t container_entry = 0;
  uint64_t container_size = 0;
  std::string container_name;
  std::optional<uint64_t> polling_loop_header;
  std::vector<OrdinaryAcquireMember> cache_members;
  std::vector<OrdinaryAcquireMember> wait_members;
  std::vector<FaultMutationOwner> affected_owners;
  bool operator==(const OrdinaryAcquireSite &) const = default;
};

struct OrdinaryAcquireRecord;
class OrdinaryAcquireSelection;

/// @brief Value records that do not prolong the pristine context's lifetime.
class OrdinaryAcquireInventory {
public:
  std::span<const OrdinaryAcquireSite> sites() const;

private:
  std::shared_ptr<const OrdinaryAcquireRecord> record_;
  friend FailureOr<OrdinaryAcquireInventory>
  discover_ordinary_acquires(const FaultMutationContext &, std::string_view, FaultMutationError &);
  friend FailureOr<OrdinaryAcquireSelection>
  select_ordinary_acquire(const OrdinaryAcquireInventory &, const struct OrdinaryAcquireSelector &,
                          FaultMutationError &);
};

/// @brief An exact source-bound acquire operation shared by FI recipes.
class OrdinaryAcquireSelection {
public:
  const OrdinaryAcquireSite &site() const { return site_; }
  const FaultMutationOwner *owner() const;

private:
  OrdinaryAcquireSite site_;
  std::shared_ptr<const OrdinaryAcquireRecord> record_;
  friend FailureOr<OrdinaryAcquireSelection>
  select_ordinary_acquire(const OrdinaryAcquireInventory &, const struct OrdinaryAcquireSelector &,
                          FaultMutationError &);
};

struct OrdinaryAcquireSelector {
  // Exact takes precedence; invalid exact never falls back to ordinal.
  std::optional<OrdinaryAcquireSelection> exact;
  size_t ordinal = 0;
};

/// @brief Discover bounded CDNA5/RDNA4 acquire-qualified global_load_b32 sites.
/// @details Inventories contain qualified sites only, in pristine source order.
/// Endpoint owners must be complete and equal; the kernel name only selects a
/// real affected owner. Shared helper container identity is preserved separately.
/// Facts and source handles expire with the context. Successful discovery proves
/// neither that a future rewrite can be emitted nor runtime/detector behavior.
[[nodiscard]] FailureOr<OrdinaryAcquireInventory>
discover_ordinary_acquires(const FaultMutationContext &, std::string_view, FaultMutationError &);

/// @brief Select an exact handle or a zero-based qualified-load ordinal.
[[nodiscard]] FailureOr<OrdinaryAcquireSelection>
select_ordinary_acquire(const OrdinaryAcquireInventory &, const OrdinaryAcquireSelector &,
                        FaultMutationError &);

} // namespace rocjitsu
