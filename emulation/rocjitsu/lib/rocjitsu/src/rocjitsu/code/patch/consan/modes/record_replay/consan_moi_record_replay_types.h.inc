/// Translate the common causal scope into Record/Replay's stable report ABI.
/// Wavefront-local atomics are not recordable causal communication.
[[nodiscard]] constexpr std::optional<uint32_t>
consan_moi_record_replay_scope(ConSanMemoryScope scope) {
  switch (scope) {
  case ConSanMemoryScope::Wavefront:
    return std::nullopt;
  case ConSanMemoryScope::Workgroup:
    return 1u;
  case ConSanMemoryScope::Agent:
    return 2u;
  case ConSanMemoryScope::System:
    return 3u;
  }
  return std::nullopt;
}
