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
  uint16_t jump_pc_sgpr;
  uint16_t saved_scc_sgpr;
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

  [[nodiscard]] uint16_t minimum_required_sgpr_count() const {
    const uint16_t width = key_width();
    if (width == 0u)
      return 0u;
    return static_cast<uint16_t>(std::max({static_cast<uint32_t>(jump_pc_sgpr) + 2u,
                                           static_cast<uint32_t>(saved_scc_sgpr) + 1u,
                                           static_cast<uint32_t>(key_sgpr) + width}));
  }

  [[nodiscard]] bool is_well_formed() const {
    const uint16_t width = key_width();
    if (width == 0u || static_cast<uint32_t>(jump_pc_sgpr) + 2u > 65535u ||
        static_cast<uint32_t>(saved_scc_sgpr) + 1u > 65535u ||
        static_cast<uint32_t>(key_sgpr) + width > 65535u)
      return false;
    const auto overlaps = [](uint16_t lhs_base, uint16_t lhs_width, uint16_t rhs_base,
                             uint16_t rhs_width) {
      return static_cast<uint32_t>(lhs_base) < static_cast<uint32_t>(rhs_base) + rhs_width &&
             static_cast<uint32_t>(rhs_base) < static_cast<uint32_t>(lhs_base) + lhs_width;
    };
    if (overlaps(jump_pc_sgpr, 2u, saved_scc_sgpr, 1u) ||
        overlaps(jump_pc_sgpr, 2u, key_sgpr, width) ||
        overlaps(saved_scc_sgpr, 1u, key_sgpr, width))
      return false;
    return !body || key_kind != ConSanSuperColliderDenseKeyKind::ExplicitScalar ||
           body->vcc_save_sgpr == key_sgpr;
  }

  [[nodiscard]] bool same_route(const ConSanSuperColliderDenseRouteEffect &other) const {
    return dispatcher_offset == other.dispatcher_offset && jump_pc_sgpr == other.jump_pc_sgpr &&
           saved_scc_sgpr == other.saved_scc_sgpr && key_sgpr == other.key_sgpr &&
           key_kind == other.key_kind;
  }
};
