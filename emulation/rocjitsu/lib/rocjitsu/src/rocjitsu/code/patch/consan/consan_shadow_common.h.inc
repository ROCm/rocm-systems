// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

struct LdsCellRange {
  uint32_t start_cell = 0;
  uint32_t cell_count = 0;
};

namespace shadow_cell {

inline constexpr uint32_t granule_shift = 2;
inline constexpr uint32_t granule_bytes = 1u << granule_shift;

} // namespace shadow_cell

[[nodiscard]] constexpr uint64_t low_bit_mask(uint32_t bits) {
  return bits >= 64 ? ~uint64_t{0} : ((uint64_t{1} << bits) - 1u);
}

[[nodiscard]] constexpr uint64_t bit_field_mask(uint32_t shift, uint32_t bits) {
  return low_bit_mask(bits) << shift;
}

[[nodiscard]] constexpr ShadowAccessKind shadow_kind_from_access_kind(LdsAccessKind kind) {
  switch (kind) {
  case LdsAccessKind::Read:
    return ShadowAccessKind::Read;
  case LdsAccessKind::Write:
    return ShadowAccessKind::Write;
  case LdsAccessKind::Atomic:
    return ShadowAccessKind::Atomic;
  case LdsAccessKind::Other:
    return ShadowAccessKind::Empty;
  }
  return ShadowAccessKind::Empty;
}

[[nodiscard]] constexpr bool shadow_kind_is_empty(ShadowAccessKind kind) {
  return kind == ShadowAccessKind::Empty;
}

[[nodiscard]] constexpr bool shadow_kind_conflicts(ShadowAccessKind current,
                                                   ShadowAccessKind prior) {
  if (shadow_kind_is_empty(current) || shadow_kind_is_empty(prior))
    return false;
  if (current == ShadowAccessKind::Read)
    return prior != ShadowAccessKind::Read;
  if (current == ShadowAccessKind::Atomic)
    return prior != ShadowAccessKind::Atomic;
  return true;
}

[[nodiscard]] constexpr uint32_t range_end(uint32_t start, uint32_t count) { return start + count; }

[[nodiscard]] constexpr bool ranges_overlap(uint32_t first_start, uint32_t first_count,
                                            uint32_t second_start, uint32_t second_count) {
  return first_start < range_end(second_start, second_count) &&
         second_start < range_end(first_start, first_count);
}

[[nodiscard]] constexpr bool cell_ranges_overlap(uint32_t first_start, uint32_t first_count,
                                                 uint32_t second_start, uint32_t second_count) {
  return ranges_overlap(first_start, first_count, second_start, second_count);
}

[[nodiscard]] constexpr bool cell_ranges_overlap(LdsCellRange first, LdsCellRange second) {
  return cell_ranges_overlap(first.start_cell, first.cell_count, second.start_cell,
                             second.cell_count);
}

[[nodiscard]] constexpr LdsCellRange lds_cell_range_for_bytes(uint32_t byte_offset,
                                                              uint32_t byte_count) {
  const uint32_t start_cell = byte_offset >> shadow_cell::granule_shift;
  const uint32_t end_cell =
      (byte_offset + byte_count + shadow_cell::granule_bytes - 1u) >> shadow_cell::granule_shift;
  return LdsCellRange{start_cell, end_cell - start_cell};
}
