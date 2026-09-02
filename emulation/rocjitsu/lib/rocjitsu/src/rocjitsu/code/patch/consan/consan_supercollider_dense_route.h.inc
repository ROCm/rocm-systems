// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// How a SuperCollider dense dispatcher identifies one instrumentation body.
enum class ConSanSuperColliderDenseKeyKind : uint8_t {
  CallReturnPair,
  ExplicitScalar,
};

/// Body-only continuation state carried beside the shared dense route.
struct ConSanSuperColliderDenseBodyRoute {
  uint16_t vcc_save_sgpr;
  uint64_t return_offset;
};

/// Complete mode-local routing effect shared by one dense dispatcher, its
/// relocated entry host, and every instrumentation body that it serves.
/// Coordinator patches leave `body` absent. Body patches retain their exact
/// VCC-preservation register and return location as one additional facet.
struct ConSanSuperColliderDenseRouteEffect {
  uint64_t dispatcher_offset;
  ConSanIndirectJumpSgprs jump;
  uint16_t key_sgpr;
  ConSanSuperColliderDenseKeyKind key_kind;
  std::optional<ConSanSuperColliderDenseBodyRoute> body;

  [[nodiscard]] uint16_t key_width() const {
    switch (key_kind) {
    case ConSanSuperColliderDenseKeyKind::CallReturnPair:
      return 2u;
    case ConSanSuperColliderDenseKeyKind::ExplicitScalar:
      return 1u;
    }
    return 0u;
  }

  [[nodiscard]] uint32_t minimum_required_sgpr_count() const {
    const uint16_t width = key_width();
    if (width == 0u)
      return 0u;
    return std::max(jump.minimum_required_sgpr_count(), static_cast<uint32_t>(key_sgpr) + width);
  }

  [[nodiscard]] bool is_well_formed() const {
    const uint16_t width = key_width();
    return width != 0u && jump.is_well_formed() && minimum_required_sgpr_count() <= 65535u &&
           !jump.overlaps(key_sgpr, width) &&
           (!body || key_kind != ConSanSuperColliderDenseKeyKind::ExplicitScalar ||
            body->vcc_save_sgpr == key_sgpr);
  }

  [[nodiscard]] bool same_route(const ConSanSuperColliderDenseRouteEffect &other) const {
    return dispatcher_offset == other.dispatcher_offset && jump == other.jump &&
           key_sgpr == other.key_sgpr && key_kind == other.key_kind;
  }
};
