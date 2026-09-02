/// Address-free private-memory layout for owner-local persistent MOI identity.
///
/// The persistent range contains only state that survives between probes. The
/// ephemeral base is the target-normalized first byte from which a site-local
/// spill manager may allocate temporary preservation state. Keeping both
/// boundaries prevents consumers from silently substituting one for the
/// other when target allocation granularity leaves padding between them.
struct ConSanMoiPrivateStateLayout {
  uint32_t epoch_offset = 0;
  std::optional<uint32_t> owner_offset;
  std::optional<uint32_t> workgroup_key_offset;
  std::optional<uint32_t> dispatch_id_offset;
  ConSanMoiPersistentWorkgroupPrivateOffsets record_replay_workgroup_offsets;
  uint32_t persistent_state_end = 0;
  uint32_t ephemeral_base = 0;

  /// Prove the target-neutral structural invariant shared by planning,
  /// emission, patch proof, and independent validation.
  [[nodiscard]] bool is_well_formed() const {
    struct Range {
      uint32_t begin = 0;
      uint32_t width = 0;
    };
    std::array<std::optional<Range>, 8> ranges;
    size_t range_count = 0;
    const auto append = [&](std::optional<uint32_t> begin, uint32_t width) {
      if (!begin)
        return true;
      const uint64_t end = static_cast<uint64_t>(*begin) + width;
      if (*begin % sizeof(uint32_t) != 0u || end > persistent_state_end)
        return false;
      ranges[range_count++] = Range{*begin, width};
      return true;
    };
    if (!append(epoch_offset, sizeof(uint32_t)) || !append(owner_offset, sizeof(uint32_t)) ||
        !append(workgroup_key_offset, sizeof(uint32_t)) ||
        !append(dispatch_id_offset, 2u * sizeof(uint32_t))) {
      return false;
    }
    for (std::optional<uint32_t> offset : record_replay_workgroup_offsets.values()) {
      if (!append(offset, sizeof(uint32_t)))
        return false;
    }
    uint32_t exact_end = 0;
    for (size_t i = 0; i < range_count; ++i) {
      const uint32_t end = ranges[i]->begin + ranges[i]->width;
      exact_end = std::max(exact_end, end);
      for (size_t j = 0; j < i; ++j) {
        const uint32_t other_end = ranges[j]->begin + ranges[j]->width;
        if (ranges[i]->begin < other_end && ranges[j]->begin < end)
          return false;
      }
    }
    return persistent_state_end == exact_end && ephemeral_base >= persistent_state_end &&
           ephemeral_base % sizeof(uint32_t) == 0u;
  }

  auto operator<=>(const ConSanMoiPrivateStateLayout &) const = default;
};
