// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

struct ConSanMoiLdsCellRange {
  uint32_t start_cell = 0;
  uint32_t cell_count = 0;
};

namespace consan_moi_shadow_cell {

inline constexpr uint32_t granule_shift = 2;
inline constexpr uint32_t granule_bytes = 1u << granule_shift;

} // namespace consan_moi_shadow_cell

[[nodiscard]] constexpr uint64_t consan_moi_low_bit_mask(uint32_t bits) {
  return bits >= 64 ? ~uint64_t{0} : ((uint64_t{1} << bits) - 1u);
}

[[nodiscard]] constexpr uint64_t consan_moi_bit_field_mask(uint32_t shift, uint32_t bits) {
  return consan_moi_low_bit_mask(bits) << shift;
}

[[nodiscard]] constexpr ConSanMoiShadowAccessKind
consan_moi_shadow_kind_from_access_kind(ConSanLdsAccessKind kind) {
  switch (kind) {
  case ConSanLdsAccessKind::Read:
    return ConSanMoiShadowAccessKind::Read;
  case ConSanLdsAccessKind::Write:
    return ConSanMoiShadowAccessKind::Write;
  case ConSanLdsAccessKind::Atomic:
    return ConSanMoiShadowAccessKind::Atomic;
  case ConSanLdsAccessKind::Other:
    return ConSanMoiShadowAccessKind::Empty;
  }
  return ConSanMoiShadowAccessKind::Empty;
}

[[nodiscard]] constexpr bool consan_moi_shadow_kind_is_empty(ConSanMoiShadowAccessKind kind) {
  return kind == ConSanMoiShadowAccessKind::Empty;
}

[[nodiscard]] constexpr bool consan_moi_shadow_kind_conflicts(ConSanMoiShadowAccessKind current,
                                                              ConSanMoiShadowAccessKind prior) {
  if (consan_moi_shadow_kind_is_empty(current) || consan_moi_shadow_kind_is_empty(prior))
    return false;
  if (current == ConSanMoiShadowAccessKind::Read)
    return prior != ConSanMoiShadowAccessKind::Read;
  if (current == ConSanMoiShadowAccessKind::Atomic)
    return prior != ConSanMoiShadowAccessKind::Atomic;
  return true;
}

[[nodiscard]] constexpr uint32_t consan_moi_range_end(uint32_t start, uint32_t count) {
  return start + count;
}

[[nodiscard]] constexpr bool consan_moi_ranges_overlap(uint32_t first_start, uint32_t first_count,
                                                       uint32_t second_start,
                                                       uint32_t second_count) {
  return first_start < consan_moi_range_end(second_start, second_count) &&
         second_start < consan_moi_range_end(first_start, first_count);
}

[[nodiscard]] constexpr bool consan_moi_cell_ranges_overlap(uint32_t first_start,
                                                            uint32_t first_count,
                                                            uint32_t second_start,
                                                            uint32_t second_count) {
  return consan_moi_ranges_overlap(first_start, first_count, second_start, second_count);
}

[[nodiscard]] constexpr bool consan_moi_cell_ranges_overlap(ConSanMoiLdsCellRange first,
                                                            ConSanMoiLdsCellRange second) {
  return consan_moi_cell_ranges_overlap(first.start_cell, first.cell_count, second.start_cell,
                                        second.cell_count);
}

[[nodiscard]] constexpr ConSanMoiLdsCellRange
consan_moi_lds_cell_range_for_bytes(uint32_t byte_offset, uint32_t byte_count) {
  const uint32_t start_cell = byte_offset >> consan_moi_shadow_cell::granule_shift;
  const uint32_t end_cell =
      (byte_offset + byte_count + consan_moi_shadow_cell::granule_bytes - 1u) >>
      consan_moi_shadow_cell::granule_shift;
  return ConSanMoiLdsCellRange{start_cell, end_cell - start_cell};
}
