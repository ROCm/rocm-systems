/// Creates a production header from the exact heterogeneous layout. Focused
/// tests that deliberately construct partial headers own their convenience
/// builder outside the production contract.
[[nodiscard]] constexpr ReportHeader
make_report_header_for_layout(uint64_t generation, uint64_t dispatch_id,
                              const ReportBufferLayout &layout) {
  ReportHeader header;
  header.header_size = sizeof(ReportHeader);
  header.generation = generation;
  header.dispatch_id = dispatch_id;
  header.watchpoint_capacity = layout.watchpoint_capacity;
  header.causal_window_capacity = layout.causal_window_capacity;
  header.sync_metadata_capacity = layout.sync_metadata_capacity;
  header.pending_acquire_capacity = layout.pending_acquire_capacity;
  header.publication_event_capacity = layout.publication_event_capacity;
  return header;
}

[[nodiscard]] constexpr uint32_t clamp_u32_capacity(uint64_t capacity) {
  return capacity > std::numeric_limits<uint32_t>::max() ? std::numeric_limits<uint32_t>::max()
                                                         : static_cast<uint32_t>(capacity);
}

[[nodiscard]] constexpr bool report_layout_matches_header(const ReportHeader &header,
                                                          const ReportBufferLayout &layout,
                                                          uint64_t report_buffer_size) {
  return report_header_is_current(header) && layout.valid &&
         layout.required_bytes <= report_buffer_size &&
         header.watchpoint_capacity == layout.watchpoint_capacity &&
         header.causal_window_capacity == layout.causal_window_capacity &&
         header.sync_metadata_capacity == layout.sync_metadata_capacity &&
         header.pending_acquire_capacity == layout.pending_acquire_capacity &&
         header.publication_event_capacity == layout.publication_event_capacity;
}

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
