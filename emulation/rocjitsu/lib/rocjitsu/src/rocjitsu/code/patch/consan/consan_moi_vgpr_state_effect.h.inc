/// Lifetime of the owner/epoch destinations initialized by an MOI entry
/// prologue or consumed by a probe. Scalar- and private-persistent paths use
/// transient vector scratch; vector-persistent kernels retain the pair either
/// as one code-object-wide ABI or as an owner-component-local ABI.
enum class ConSanMoiOwnerEpochVgprLifetime : uint8_t {
  Transient,
  CodeObjectPersistent,
  OwnerLocalPersistent,
};

[[nodiscard]] constexpr ConSanMoiOwnerEpochVgprLifetime
consan_moi_owner_epoch_vgpr_lifetime(bool transient, bool owner_local) {
  return transient     ? ConSanMoiOwnerEpochVgprLifetime::Transient
         : owner_local ? ConSanMoiOwnerEpochVgprLifetime::OwnerLocalPersistent
                       : ConSanMoiOwnerEpochVgprLifetime::CodeObjectPersistent;
}

/// Complete vector-register state effect initialized or consumed by one MOI
/// patch. The owner/epoch pair is indivisible, while the lifetime distinguishes
/// true persistent state from entry-only prologue scratch.
struct ConSanMoiVgprStateEffect {
  ConSanMoiOwnerEpochRegisters owner_epoch;
  std::optional<uint16_t> workgroup_key;
  ConSanMoiPersistentWorkgroupRegisters record_replay_workgroup;
  ConSanMoiOwnerEpochVgprLifetime owner_epoch_lifetime = ConSanMoiOwnerEpochVgprLifetime::Transient;

  template <typename Visitor> void for_each_range(Visitor &&visit) const {
    visit(owner_epoch.owner, 1u);
    visit(owner_epoch.epoch, 1u);
    if (workgroup_key)
      visit(*workgroup_key, 1u);
    for (std::optional<uint16_t> reg : record_replay_workgroup.values()) {
      if (reg)
        visit(*reg, 1u);
    }
  }

  [[nodiscard]] bool is_well_formed() const {
    switch (owner_epoch_lifetime) {
    case ConSanMoiOwnerEpochVgprLifetime::Transient:
    case ConSanMoiOwnerEpochVgprLifetime::CodeObjectPersistent:
    case ConSanMoiOwnerEpochVgprLifetime::OwnerLocalPersistent:
      break;
    default:
      return false;
    }
    std::array<bool, 256> occupied{};
    bool valid = true;
    for_each_range([&](uint16_t base, uint16_t width) {
      const uint32_t end = static_cast<uint32_t>(base) + width;
      if (end > occupied.size()) {
        valid = false;
        return;
      }
      for (uint32_t reg = base; reg < end; ++reg) {
        if (occupied[reg])
          valid = false;
        occupied[reg] = true;
      }
    });
    return valid;
  }

  [[nodiscard]] uint32_t required_vgpr_count() const {
    uint32_t required = 0;
    for_each_range([&](uint16_t base, uint16_t width) {
      required = std::max<uint32_t>(required, static_cast<uint32_t>(base) + width);
    });
    return required;
  }
};
