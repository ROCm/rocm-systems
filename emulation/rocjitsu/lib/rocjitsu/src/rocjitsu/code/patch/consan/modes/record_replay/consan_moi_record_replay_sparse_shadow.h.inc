/// Bounded sparse byte-range provenance for host-side Record/Replay detection.
/// Cross-owner state is partitioned by generation, epoch, access kind, and
/// owner. Same-owner exact-group state additionally retains the instruction
/// site and lane group. These keys contain every non-spatial field read by
/// consan_moi_exact_byte_accesses_conflict; the cross-owner key also contains
/// every prior-access field read by consan_moi_acquired_epoch_orders. An access
/// replaces byte intervals only within a class that is equivalent for every
/// future conflict and acquired-order decision. Storage stays linear in
/// admitted accesses and interval boundaries rather than the LDS extent.
/// Lookup visits the conflicting access-kind classes and the owners or lane
/// groups in the current workgroup that share the relevant class; it does not
/// scale with the LDS extent or the total retained site inventory.
/// maximum_access_count is a structural input bound, not a second runtime
/// policy budget.
class ConSanMoiSparseExactByteShadow {
public:
  explicit ConSanMoiSparseExactByteShadow(uint64_t byte_capacity, uint32_t maximum_access_count);

  [[nodiscard]] ConSanMoiExactByteAccessResult
  access(const ConSanMoiExactByteAccess &current,
         std::span<const ConSanMoiAcquiredEpochToken> acquired_epoch_tokens = {});

private:
  friend void retire_consan_moi_sparse_exact_byte_shadow_before_epoch(
      ConSanMoiSparseExactByteShadow &, uint64_t, uint32_t);

  struct Interval {
    uint64_t end = 0;
    uint32_t provenance_index = 0;
  };

  struct EpochKindClass {
    uint64_t generation = 0;
    uint32_t epoch = 0;
    ConSanMoiShadowAccessKind kind = ConSanMoiShadowAccessKind::Empty;

    auto operator<=>(const EpochKindClass &) const = default;
  };

  struct SameSiteClass {
    EpochKindClass epoch_kind;
    uint32_t owner_id = 0;
    uint32_t instruction_offset = 0;

    auto operator<=>(const SameSiteClass &) const = default;
  };

  using IntervalMap = std::map<uint64_t, Interval>;
  using OwnerIntervalMaps = std::map<uint32_t, IntervalMap>;
  using LaneIntervalMaps = std::map<uint64_t, IntervalMap>;

  uint64_t byte_capacity_ = 0;
  uint32_t maximum_access_count_ = 0;
  std::vector<ConSanMoiExactByteAccess> provenance_;
  std::map<EpochKindClass, OwnerIntervalMaps> cross_owner_intervals_;
  std::map<SameSiteClass, LaneIntervalMaps> same_site_intervals_;
};
