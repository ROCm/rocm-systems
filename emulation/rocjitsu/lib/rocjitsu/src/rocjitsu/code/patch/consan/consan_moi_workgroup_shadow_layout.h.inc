/// Exact workgroup-local InlineShadow storage and initialization ABI.
///
/// This layout crosses planning, emitted patch proof, prologue composition,
/// descriptor growth, and independent validation. Keep it narrow enough for
/// the shared patch contract without importing the broader MOI report model.
struct ConSanMoiWorkgroupShadowLayout {
  uint32_t base = 0;
  uint32_t size = 0;
  uint32_t validity_base = 0;
  uint32_t validity_size = 0;
  uint32_t required_group_segment_size = 0;
  uint8_t workitem_id_dimensions = 1;
  uint16_t initialization_lanes = 32;
  bool lazy_initialization = false;
  std::optional<uint16_t> visible_evidence_sgpr;

  bool operator==(const ConSanMoiWorkgroupShadowLayout &) const = default;
};
