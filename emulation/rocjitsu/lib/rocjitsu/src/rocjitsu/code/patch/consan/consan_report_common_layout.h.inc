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
  return header;
}
