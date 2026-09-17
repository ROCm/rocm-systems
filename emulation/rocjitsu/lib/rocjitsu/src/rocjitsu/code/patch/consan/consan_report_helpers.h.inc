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
         header.pending_acquire_capacity == layout.pending_acquire_capacity;
}
