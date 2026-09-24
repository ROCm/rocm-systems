// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#pragma once

#include <cstdint>
#include <span>

namespace rocjitsu::consan::hook {

// Unpacked host representation. Device records must be identity-checked and
// decoded before entering this proof; physical report-slot order is never time.
struct PublicationDomain {
  uint64_t generation = 0;
  uint64_t dispatch = 0;
  uint32_t workgroup_x = 0;
  uint32_t workgroup_y = 0;
  uint32_t workgroup_z = 0;
  uint32_t cluster_workgroup = 0;
  bool operator==(const PublicationDomain &) const = default;
};
struct PublicationPoint {
  PublicationDomain domain;
  uint32_t owner = 0;
  uint64_t sequence = 0;
};
enum class PublicationOperation : uint8_t { Read, Rmw };
struct PublicationEvent {
  PublicationPoint point;
  uint64_t address = 0;
  uint32_t bytes = 0;
  uint64_t observed = 0;
  uint64_t written = 0;
  PublicationOperation operation = PublicationOperation::Rmw;
  bool release = false;
  bool acquire = false;
  bool covers_workgroup = false;
  bool observation_valid = false;
};
enum class PublicationOrdering { Unordered, Ordered, Incomplete };

// Requires a complete trace of modifications of the relevant atomic objects,
// including relaxed RMWs. Non-returning RMWs need an actual observation witness;
// an address/role match is insufficient. Read includes failed CAS (no release).
// Repeated values, ABA, non-changing RMWs, inconsistent per-owner order, missing
// observations and overlapping object identities fail closed. A complete result
// with no ordering path is Unordered, not a claim that either access raced.
[[nodiscard]] PublicationOrdering publication_orders(const PublicationPoint &before,
                                                     const PublicationPoint &after,
                                                     std::span<const PublicationEvent> events,
                                                     bool complete);

} // namespace rocjitsu::consan::hook
