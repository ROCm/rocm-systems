// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// Content-stable identity of the pristine code object from which semantic
/// inventory was decoded.
///
/// `fingerprint` preserves ConSan's externally reported FNV-1a spelling.
/// Equality additionally includes the byte size and an independently seeded,
/// reverse-order verification hash. That second digest makes a collision in
/// the externally visible fingerprint detectable inside semantic IDs rather
/// than silently treating two different inputs as one program. This is a
/// content identity only: it deliberately contains no HSA reader, executable,
/// load, or dispatch handle and therefore says nothing about runtime lifetime.
struct ConSanCodeObjectId {
  /// Stable diagnostic fingerprint retained for compatibility with existing
  /// report and validation formats.
  std::string fingerprint;
  /// Size of the complete pristine input image in bytes.
  uint64_t byte_size = 0;
  /// Independent content digest used to disambiguate a primary-fingerprint
  /// collision before semantic records are compared or joined.
  uint64_t collision_verifier = 0;

  /// Return whether construction supplied a content fingerprint. The empty
  /// default is the sole invalid identity; an empty byte sequence can still
  /// have a valid content identity even though it is not a valid code object.
  [[nodiscard]] bool valid() const { return !fingerprint.empty(); }

  bool operator==(const ConSanCodeObjectId &) const = default;
};

/// Build the content identity used by `ProgramInventory` and stable site IDs.
/// The function is deterministic across hosts and retains the pre-production
/// `fnv1a64:<hex>` diagnostic representation.
[[nodiscard]] inline ConSanCodeObjectId make_consan_code_object_id(std::span<const uint8_t> bytes) {
  const auto fnv1a = [](auto begin, auto end, uint64_t seed) {
    uint64_t hash = seed;
    for (auto cursor = begin; cursor != end; ++cursor) {
      hash ^= static_cast<uint8_t>(*cursor);
      hash *= 1099511628211ull;
    }
    return hash;
  };
  const auto hex = [](uint64_t value) {
    constexpr char kHex[] = "0123456789abcdef";
    std::string result(16, '0');
    for (unsigned i = 0; i < 16; ++i) {
      result[15 - i] = kHex[value & 0xfu];
      value >>= 4u;
    }
    return result;
  };
  const uint64_t primary = fnv1a(bytes.begin(), bytes.end(), 14695981039346656037ull);
  const uint64_t verifier = fnv1a(bytes.rbegin(), bytes.rend(), 7809847782465536322ull);
  return {
      .fingerprint = "fnv1a64:" + hex(primary),
      .byte_size = bytes.size(),
      .collision_verifier = verifier,
  };
}

/// Stable identity of one physical instruction in the pristine code object.
///
/// The original text offset is independent of ELF file layout changes made by
/// later transforms. All semantic interpretations and all aliases of the same
/// instruction share this value, ensuring that placement can eventually emit
/// at most one physical insertion point for them.
struct PhysicalSiteId {
  /// Content identity of the pristine image containing the instruction.
  ConSanCodeObjectId code_object;
  /// Instruction offset in the original executable text address space.
  uint64_t original_text_offset = 0;

  /// Return whether the containing code-object identity was initialized.
  [[nodiscard]] bool valid() const { return code_object.valid(); }

  bool operator==(const PhysicalSiteId &) const = default;
};

/// Semantic domain that gives meaning to the member ordinals in a
/// `SemanticSiteId`.
///
/// The domain prevents an access range and a synchronization event at the same
/// instruction from colliding merely because both use member zero. New domains
/// belong here only when they describe an original-program fact; lowerer patch
/// kinds and engine-specific plan entries are not semantic site domains.
enum class ConSanSemanticSiteDomain : uint8_t {
  /// A logical byte range read, written, or atomically accessed by an
  /// instruction.
  Access,
  /// A synchronization event decoded from an instruction or qualified
  /// instruction sequence.
  SynchronizationEvent,
  /// A stable member of a logical synchronization sequence.
  SynchronizationSequenceMember,
  /// One past the last domain. This is an iteration bound and is never a valid
  /// semantic identity.
  Count,
};

/// Stable identity of one logical fact contributed by a physical instruction.
///
/// `member_ordinal` distinguishes multiple semantic members in one domain;
/// `range_ordinal` distinguishes multiple byte ranges of one access member.
/// For example, a two-address LDS instruction has one access member and two
/// range ordinals while still retaining one `PhysicalSiteId`. Ordinals describe
/// decoded program meaning, never traversal or patch-emission order.
struct SemanticSiteId {
  /// Physical instruction that contributes this semantic fact.
  PhysicalSiteId physical;
  /// Vocabulary in which the member and range ordinals are interpreted.
  ConSanSemanticSiteDomain domain = ConSanSemanticSiteDomain::Count;
  /// Stable logical-member ordinal within the selected domain.
  uint32_t member_ordinal = 0;
  /// Stable range/member sub-ordinal within the logical member.
  uint32_t range_ordinal = 0;

  /// Return whether both the physical identity and semantic domain are valid.
  [[nodiscard]] bool valid() const {
    return physical.valid() && domain != ConSanSemanticSiteDomain::Count;
  }

  bool operator==(const SemanticSiteId &) const = default;
};
