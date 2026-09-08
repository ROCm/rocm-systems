struct ConSanMoiExactByteAccess {
  uint64_t generation = 0;
  uint32_t owner_id = 0;
  uint32_t epoch = 0;
  ConSanMoiShadowAccessKind kind = ConSanMoiShadowAccessKind::Empty;
  uint32_t lds_byte_offset = 0;
  uint32_t lds_byte_count = 0;
  uint32_t instruction_offset = 0;
  uint64_t lane_mask = 0;
  bool exact_address_group = false;
};

struct ConSanMoiExactByteAccessResult {
  bool capacity_exhausted = false;
  bool conflict = false;
  std::optional<ConSanMoiExactByteAccess> prior;
};
