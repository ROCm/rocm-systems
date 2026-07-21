namespace {

using KD = rocr::llvm::amdhsa::kernel_descriptor_t;
namespace kd = rocr::llvm::amdhsa;

[[nodiscard]] const char *target_name(rj_code_target_id_t target) {
  switch (target) {
  case ROCJITSU_CODE_TARGET_GFX90A:
    return "gfx90a";
  case ROCJITSU_CODE_TARGET_GFX942:
    return "gfx942";
  case ROCJITSU_CODE_TARGET_GFX950:
    return "gfx950";
  case ROCJITSU_CODE_TARGET_GFX1200:
    return "gfx1200";
  case ROCJITSU_CODE_TARGET_GFX1201:
    return "gfx1201";
  case ROCJITSU_CODE_TARGET_GFX1250:
    return "gfx1250";
  default:
    return "invalid";
  }
}

[[nodiscard]] const char *arch_name(rj_code_target_id_t target) {
  switch (target) {
  case ROCJITSU_CODE_TARGET_GFX90A:
    return "cdna2";
  case ROCJITSU_CODE_TARGET_GFX942:
    return "cdna3";
  case ROCJITSU_CODE_TARGET_GFX950:
    return "cdna4";
  case ROCJITSU_CODE_TARGET_GFX1200:
  case ROCJITSU_CODE_TARGET_GFX1201:
    return "rdna4";
  case ROCJITSU_CODE_TARGET_GFX1250:
    return "gfx1250";
  default:
    return "invalid";
  }
}

[[nodiscard]] bool is_cdna_arch(rj_code_arch_t arch) {
  return arch == ROCJITSU_CODE_ARCH_CDNA1 || arch == ROCJITSU_CODE_ARCH_CDNA2 ||
         arch == ROCJITSU_CODE_ARCH_CDNA3 || arch == ROCJITSU_CODE_ARCH_CDNA4;
}

[[nodiscard]] bool is_rdna_arch(rj_code_arch_t arch) {
  return arch == ROCJITSU_CODE_ARCH_RDNA1 || arch == ROCJITSU_CODE_ARCH_RDNA2 ||
         arch == ROCJITSU_CODE_ARCH_RDNA3 || arch == ROCJITSU_CODE_ARCH_RDNA3_5 ||
         arch == ROCJITSU_CODE_ARCH_RDNA4 || arch == ROCJITSU_CODE_ARCH_GFX1250;
}

[[nodiscard]] uint32_t descriptor_vgpr_granularity_for_wavefront(rj_code_arch_t arch,
                                                                 uint32_t wavefront_size) {
  if (arch == ROCJITSU_CODE_ARCH_CDNA1)
    return 4;
  if (is_cdna_arch(arch))
    return 8;
  if (arch == ROCJITSU_CODE_ARCH_GFX1250)
    return wavefront_size == 32 ? 16 : 8;
  if (is_rdna_arch(arch))
    return wavefront_size == 32 ? 8 : 4;
  return 1;
}

[[nodiscard]] uint32_t descriptor_wavefront_size(rj_code_arch_t arch, const KD &desc) {
  if (is_rdna_arch(arch)) {
    const bool wave32 = AMDHSA_BITS_GET(desc.kernel_code_properties,
                                        kd::KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32);
    return wave32 ? 32 : 64;
  }
  return 64;
}

[[nodiscard]] uint16_t descriptor_vgpr_allocation_count(const KD &desc, rj_code_arch_t arch) {
  const uint32_t wavefront_size = descriptor_wavefront_size(arch, desc);
  const uint32_t granularity = descriptor_vgpr_granularity_for_wavefront(arch, wavefront_size);
  const uint32_t granulated =
      AMDHSA_BITS_GET(desc.compute_pgm_rsrc1, kd::COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT);
  const uint32_t allocated = (granulated + 1u) * std::max(granularity, 1u);
  return static_cast<uint16_t>(std::min<uint32_t>(allocated, REGISTER_SET_MAX_VGPRS));
}

[[nodiscard]] std::optional<uint16_t>
descriptor_vgpr_allocation_count(std::span<const uint8_t> image, uint64_t descriptor_file_offset,
                                 rj_code_arch_t arch) {
  if (descriptor_file_offset > image.size() || sizeof(KD) > image.size() - descriptor_file_offset)
    return std::nullopt;
  KD desc{};
  std::memcpy(&desc, image.data() + descriptor_file_offset, sizeof(desc));
  return descriptor_vgpr_allocation_count(desc, arch);
}

[[nodiscard]] uint16_t descriptor_sgpr_allocation_count(const KD &desc) {
  constexpr uint32_t kMaxOrdinarySgprs = 106u;
  const uint32_t granulated = AMDHSA_BITS_GET(
      desc.compute_pgm_rsrc1, kd::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT);
  return static_cast<uint16_t>(std::min<uint32_t>((granulated + 1u) * 8u, kMaxOrdinarySgprs));
}

[[nodiscard]] bool grow_descriptor_sgpr_allocation(KD &desc, uint16_t required_count) {
  constexpr uint32_t kSgprGranularity = 8u;
  constexpr uint32_t kMaxRdna4OrdinarySgprs = 106u;
  if (required_count == 0 || required_count > kMaxRdna4OrdinarySgprs)
    return false;
  if (required_count <= descriptor_sgpr_allocation_count(desc))
    return true;
  const uint32_t rounded = (static_cast<uint32_t>(required_count) + kSgprGranularity - 1u) /
                           kSgprGranularity * kSgprGranularity;
  if (rounded > kMaxRdna4OrdinarySgprs)
    return false;
  AMDHSA_BITS_SET(desc.compute_pgm_rsrc1, kd::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT,
                  (rounded / kSgprGranularity - 1u));
  return true;
}

[[nodiscard]] bool grow_descriptor_vgpr_allocation(KD &desc, uint16_t required_count,
                                                   rj_code_arch_t arch) {
  if (required_count > REGISTER_SET_MAX_VGPRS)
    return false;
  if (required_count <= descriptor_vgpr_allocation_count(desc, arch))
    return true;

  const uint32_t wavefront_size = descriptor_wavefront_size(arch, desc);
  const uint32_t granularity =
      std::max<uint32_t>(descriptor_vgpr_granularity_for_wavefront(arch, wavefront_size), 1u);
  const uint32_t rounded =
      ((std::max<uint32_t>(required_count, 1u) + granularity - 1u) / granularity) * granularity;
  if (rounded > REGISTER_SET_MAX_VGPRS)
    return false;
  const uint32_t granulated = rounded / granularity - 1u;
  AMDHSA_BITS_SET(desc.compute_pgm_rsrc1, kd::COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT,
                  granulated);
  return true;
}

struct DescriptorVgprGrowth {
  uint64_t descriptor_file_offset = 0;
  uint16_t required_count = 0;
};

struct DescriptorSgprGrowth {
  uint64_t descriptor_file_offset = 0;
  uint16_t required_count = 0;
};

[[nodiscard]] std::unordered_map<uint64_t, uint16_t>
merge_descriptor_sgpr_growths(std::span<const DescriptorSgprGrowth> growths) {
  std::unordered_map<uint64_t, uint16_t> merged;
  for (const DescriptorSgprGrowth &growth : growths) {
    auto [it, inserted] = merged.emplace(growth.descriptor_file_offset, growth.required_count);
    if (!inserted)
      it->second = std::max(it->second, growth.required_count);
  }
  return merged;
}

[[nodiscard]] bool
apply_descriptor_sgpr_growths_to_patcher(CodeObjectPatcher &patcher,
                                         std::span<const DescriptorSgprGrowth> growths,
                                         std::vector<std::string> &errors) {
  const std::span<const uint8_t> current_image = patcher.image_bytes();
  for (const auto &[descriptor_file_offset, required_count] :
       merge_descriptor_sgpr_growths(growths)) {
    if (descriptor_file_offset > current_image.size() ||
        sizeof(KD) > current_image.size() - descriptor_file_offset) {
      errors.emplace_back("ConSan relay-reservoir descriptor SGPR growth exceeds ELF bytes");
      return false;
    }
    KD desc{};
    std::memcpy(&desc, current_image.data() + descriptor_file_offset, sizeof(desc));
    if (!grow_descriptor_sgpr_allocation(desc, required_count) ||
        !patcher.patch_kernel_descriptor(
            descriptor_file_offset, {reinterpret_cast<const uint8_t *>(&desc), sizeof(desc)})) {
      errors.emplace_back("ConSan relay reservoir could not grow descriptor SGPR allocation");
      return false;
    }
  }
  return true;
}

[[nodiscard]] std::unordered_map<uint64_t, uint16_t>
merge_descriptor_vgpr_growths(std::span<const DescriptorVgprGrowth> growths) {
  std::unordered_map<uint64_t, uint16_t> merged;
  for (const DescriptorVgprGrowth &growth : growths) {
    auto [it, inserted] = merged.emplace(growth.descriptor_file_offset, growth.required_count);
    if (!inserted)
      it->second = std::max(it->second, growth.required_count);
  }
  return merged;
}

[[nodiscard]] bool
apply_descriptor_vgpr_growths_to_bytes(std::span<uint8_t> image,
                                       std::span<const DescriptorVgprGrowth> growths,
                                       rj_code_arch_t arch, std::vector<std::string> &errors) {
  for (const auto &[descriptor_file_offset, required_count] :
       merge_descriptor_vgpr_growths(growths)) {
    if (descriptor_file_offset > image.size() ||
        sizeof(KD) > image.size() - descriptor_file_offset) {
      errors.emplace_back("ConSan LDS check/trap proof descriptor VGPR growth exceeds ELF bytes");
      return false;
    }
    KD desc{};
    std::memcpy(&desc, image.data() + descriptor_file_offset, sizeof(desc));
    if (!grow_descriptor_vgpr_allocation(desc, required_count, arch)) {
      errors.emplace_back("ConSan LDS check/trap proof could not grow descriptor VGPR allocation");
      return false;
    }
    std::memcpy(image.data() + descriptor_file_offset, &desc, sizeof(desc));
  }
  return true;
}

[[nodiscard]] bool
apply_descriptor_vgpr_growths_to_patcher(CodeObjectPatcher &patcher,
                                         std::span<const uint8_t> original_image,
                                         std::span<const DescriptorVgprGrowth> growths,
                                         rj_code_arch_t arch, std::vector<std::string> &errors) {
  for (const auto &[descriptor_file_offset, required_count] :
       merge_descriptor_vgpr_growths(growths)) {
    if (descriptor_file_offset > original_image.size() ||
        sizeof(KD) > original_image.size() - descriptor_file_offset) {
      errors.emplace_back("ConSan LDS check/trap proof descriptor VGPR growth exceeds ELF bytes");
      return false;
    }
    KD desc{};
    std::memcpy(&desc, original_image.data() + descriptor_file_offset, sizeof(desc));
    if (!grow_descriptor_vgpr_allocation(desc, required_count, arch)) {
      errors.emplace_back("ConSan LDS check/trap proof could not grow descriptor VGPR allocation");
      return false;
    }
    if (!patcher.patch_kernel_descriptor(
            descriptor_file_offset, {reinterpret_cast<const uint8_t *>(&desc), sizeof(desc)})) {
      errors.emplace_back("ConSan LDS check/trap proof could not patch descriptor VGPR allocation");
      return false;
    }
  }
  return true;
}

[[nodiscard]] rj_code_arch_t arch_for_target(rj_code_target_id_t target) {
  switch (target) {
  case ROCJITSU_CODE_TARGET_GFX950:
    return ROCJITSU_CODE_ARCH_CDNA4;
  case ROCJITSU_CODE_TARGET_GFX1201:
    return ROCJITSU_CODE_ARCH_RDNA4;
  case ROCJITSU_CODE_TARGET_GFX1250:
    return ROCJITSU_CODE_ARCH_GFX1250;
  default:
    return ROCJITSU_CODE_ARCH_INVALID;
  }
}

[[nodiscard]] bool starts_with_any(std::string_view value,
                                   std::initializer_list<std::string_view> prefixes) {
  for (std::string_view prefix : prefixes) {
    if (value.starts_with(prefix))
      return true;
  }
  return false;
}

[[nodiscard]] bool is_ds_atomic(std::string_view mnemonic) {
  return starts_with_any(mnemonic,
                         {"ds_add", "ds_sub", "ds_rsub", "ds_inc", "ds_dec", "ds_min", "ds_max",
                          "ds_and", "ds_or", "ds_xor", "ds_mskor", "ds_cmpstore", "ds_storexchg",
                          "ds_condxchg", "ds_cond_sub", "ds_sub_clamp", "ds_pk_add"});
}

[[nodiscard]] bool is_ds_read(std::string_view mnemonic) {
  return starts_with_any(mnemonic, {"ds_read", "ds_load", "ds_param_load", "ds_direct_load"});
}

[[nodiscard]] bool is_ds_write(std::string_view mnemonic) {
  return starts_with_any(mnemonic, {"ds_write", "ds_store"});
}

[[nodiscard]] bool is_ds_register_lane_operation(std::string_view mnemonic) {
  return starts_with_any(mnemonic, {"ds_bpermute", "ds_permute", "ds_swizzle"});
}

[[nodiscard]] ConSanLdsAccessKind lds_access_kind(std::string_view mnemonic) {
  if (!mnemonic.starts_with("ds_"))
    return ConSanLdsAccessKind::Other;
  if (is_ds_atomic(mnemonic))
    return ConSanLdsAccessKind::Atomic;
  if (is_ds_read(mnemonic))
    return ConSanLdsAccessKind::Read;
  if (is_ds_write(mnemonic))
    return ConSanLdsAccessKind::Write;
  return ConSanLdsAccessKind::Other;
}

[[nodiscard]] ConSanLdsAccessKind flat_access_kind(std::string_view mnemonic) {
  if (mnemonic.starts_with("flat_atomic"))
    return ConSanLdsAccessKind::Atomic;
  if (mnemonic.starts_with("flat_load"))
    return ConSanLdsAccessKind::Read;
  if (mnemonic.starts_with("flat_store"))
    return ConSanLdsAccessKind::Write;
  return ConSanLdsAccessKind::Other;
}

[[nodiscard]] uint32_t lds_width_bits(std::string_view mnemonic) {
  if (mnemonic.find("128") != std::string_view::npos ||
      mnemonic.find("x4") != std::string_view::npos)
    return 128;
  if (mnemonic.find("96") != std::string_view::npos ||
      mnemonic.find("x3") != std::string_view::npos)
    return 96;
  if (mnemonic.find("64") != std::string_view::npos ||
      mnemonic.find("x2") != std::string_view::npos)
    return 64;
  if (mnemonic.find("32") != std::string_view::npos ||
      mnemonic.find("dword") != std::string_view::npos)
    return 32;
  if (mnemonic.find("u8") != std::string_view::npos ||
      mnemonic.find("i8") != std::string_view::npos ||
      mnemonic.find("b8") != std::string_view::npos)
    return 8;
  if (mnemonic.find("16") != std::string_view::npos ||
      mnemonic.find("b16") != std::string_view::npos ||
      mnemonic.find("short") != std::string_view::npos)
    return 16;
  if (mnemonic.find("8") != std::string_view::npos || mnemonic.find("b8") != std::string_view::npos)
    return 8;
  return 0;
}

[[nodiscard]] bool is_supported_mvp_lds_site(ConSanLdsAccessKind kind, uint32_t width_bits,
                                             std::string_view mnemonic) {
  if (kind != ConSanLdsAccessKind::Read && kind != ConSanLdsAccessKind::Write)
    return false;
  if (width_bits == 8)
    return true;
  if (kind == ConSanLdsAccessKind::Read && width_bits == 16)
    return true;
  if (kind == ConSanLdsAccessKind::Write && width_bits == 16)
    return mnemonic == "ds_store_b16" || mnemonic == "ds_store_b16_d16_hi";
  if (kind == ConSanLdsAccessKind::Read && width_bits == 96)
    return mnemonic == "ds_load_b96";
  if (kind == ConSanLdsAccessKind::Write && width_bits == 96)
    return mnemonic == "ds_store_b96";
  return width_bits == 32 || width_bits == 64 || width_bits == 128;
}

[[nodiscard]] std::optional<uint16_t> vgpr_index(const Operand *operand) {
  if (operand == nullptr || !operand->is_vgpr())
    return std::nullopt;
  const uint16_t index = operand->unified_vgpr_index();
  if (index > 255)
    return std::nullopt;
  return index;
}

enum class PointerHalfHint : uint8_t {
  Unknown,
  Low,
  High,
};

struct PointerComponentHint {
  ConSanFlatAddressSpaceHint space = ConSanFlatAddressSpaceHint::Unknown;
  PointerHalfHint half = PointerHalfHint::Unknown;
  bool maybe = false;
};

struct FlatPointerPairHint {
  ConSanFlatAddressSpaceHint space = ConSanFlatAddressSpaceHint::Unknown;
  bool maybe = false;
};

struct ScalarRegisterOrigin {
  uint16_t index = 0;
  uint32_t generation = 0;
  bool valid = false;
};

struct PrivateComponentKey {
  uint16_t low_sgpr = 0;
  uint32_t low_generation = 0;
  uint32_t high_generation = 0;
  uint32_t byte_offset = 0;
  bool operator==(const PrivateComponentKey &) const = default;
};

struct PrivateComponentKeyHash {
  [[nodiscard]] size_t operator()(const PrivateComponentKey &key) const {
    size_t value = key.low_sgpr;
    value = value * 16777619u ^ key.low_generation;
    value = value * 16777619u ^ key.high_generation;
    value = value * 16777619u ^ key.byte_offset;
    return value;
  }
};

struct FlatPointerTracker {
  std::array<PointerComponentHint, REGISTER_SET_MAX_SGPRS> sgprs{};
  std::array<PointerComponentHint, REGISTER_SET_MAX_VGPRS> vgprs{};
  std::array<uint32_t, REGISTER_SET_MAX_SGPRS> sgpr_generations{};
  std::array<ScalarRegisterOrigin, REGISTER_SET_MAX_VGPRS> vgpr_scalar_origins{};
  std::unordered_map<uint32_t, PointerComponentHint> vgpr_lane_components;
  // Keyed by (scratch scalar-base selector, signed immediate byte offset).
  // This deliberately models only address-static scratch traffic. Dynamic
  // vaddr scratch accesses cannot provide an exact stack-slot identity.
  std::unordered_map<uint64_t, PointerComponentHint> scratch_components;
  std::unordered_map<PrivateComponentKey, PointerComponentHint, PrivateComponentKeyHash>
      private_components;
};

[[nodiscard]] bool is_known_flat_space(ConSanFlatAddressSpaceHint space) {
  return space == ConSanFlatAddressSpaceHint::Group ||
         space == ConSanFlatAddressSpaceHint::Private ||
         space == ConSanFlatAddressSpaceHint::Global;
}

[[nodiscard]] bool is_known_component(const PointerComponentHint &component) {
  return is_known_flat_space(component.space) && component.half != PointerHalfHint::Unknown;
}

[[nodiscard]] ConSanFlatAddressSpaceHint maybe_flat_space(ConSanFlatAddressSpaceHint space) {
  switch (space) {
  case ConSanFlatAddressSpaceHint::Group:
    return ConSanFlatAddressSpaceHint::MaybeGroup;
  case ConSanFlatAddressSpaceHint::Private:
    return ConSanFlatAddressSpaceHint::MaybePrivate;
  case ConSanFlatAddressSpaceHint::Global:
    return ConSanFlatAddressSpaceHint::Global;
  case ConSanFlatAddressSpaceHint::Unknown:
  case ConSanFlatAddressSpaceHint::MaybeGroup:
  case ConSanFlatAddressSpaceHint::MaybePrivate:
    return ConSanFlatAddressSpaceHint::Unknown;
  }
  return ConSanFlatAddressSpaceHint::Unknown;
}

[[nodiscard]] ConSanFlatAddressSpaceHint flat_pair_hint_to_public(FlatPointerPairHint hint) {
  if (!is_known_flat_space(hint.space))
    return ConSanFlatAddressSpaceHint::Unknown;
  return hint.maybe ? maybe_flat_space(hint.space) : hint.space;
}

[[nodiscard]] std::optional<RegisterRef> register_ref(const Operand *operand) {
  if (operand == nullptr)
    return std::nullopt;
  return operand->to_register_ref();
}

[[nodiscard]] std::optional<RegisterRef> register_ref(const Operand *operand, RegClass cls) {
  auto ref = register_ref(operand);
  if (!ref || ref->cls != cls)
    return std::nullopt;
  return ref;
}

[[nodiscard]] std::optional<uint16_t> register_index(const Operand *operand, RegClass cls) {
  auto ref = register_ref(operand, cls);
  if (!ref)
    return std::nullopt;
  return ref->index;
}

[[nodiscard]] std::optional<ConSanFlatAddressSpaceHint>
special_pointer_base_space(const Operand *operand) {
  if (operand == nullptr)
    return std::nullopt;
  const std::string name = operand->name();
  const auto ascii_case_equal = [&](std::string_view expected) {
    return name.size() == expected.size() &&
           std::ranges::equal(name, expected, [](char lhs, char rhs) {
             return std::tolower(static_cast<unsigned char>(lhs)) ==
                    std::tolower(static_cast<unsigned char>(rhs));
           });
  };
  if (ascii_case_equal("src_shared_base"))
    return ConSanFlatAddressSpaceHint::Group;
  if (ascii_case_equal("src_private_base"))
    return ConSanFlatAddressSpaceHint::Private;
  return std::nullopt;
}

template <size_t N>
void clear_components(std::array<PointerComponentHint, N> &components, RegisterRef ref) {
  if (ref.index >= N)
    return;
  const size_t end = std::min(N, static_cast<size_t>(ref.index) + ref.width);
  for (size_t index = ref.index; index < end; ++index)
    components[index] = {};
}

void clear_components(FlatPointerTracker &tracker, RegisterRef ref) {
  if (ref.cls == RegClass::SGPR) {
    const size_t end = std::min(tracker.sgprs.size(), static_cast<size_t>(ref.index) + ref.width);
    for (size_t index = ref.index; index < end; ++index)
      ++tracker.sgpr_generations[index];
    clear_components(tracker.sgprs, ref);
  } else if (ref.cls == RegClass::VGPR) {
    clear_components(tracker.vgprs, ref);
    const size_t end =
        std::min(tracker.vgpr_scalar_origins.size(), static_cast<size_t>(ref.index) + ref.width);
    for (size_t index = ref.index; index < end; ++index)
      tracker.vgpr_scalar_origins[index] = {};
    std::erase_if(tracker.vgpr_lane_components, [&](const auto &entry) {
      const uint32_t vgpr = entry.first >> 6u;
      return vgpr >= ref.index && vgpr < end;
    });
  }
}

void clear_destination_components(const Instruction &inst, FlatPointerTracker &tracker) {
  for (int i = 0; i < inst.num_dst_operands(); ++i) {
    if (auto ref = register_ref(inst.dst_operand(i)))
      clear_components(tracker, *ref);
  }
}

template <size_t N>
[[nodiscard]] std::optional<PointerComponentHint>
component_hint_at(const std::array<PointerComponentHint, N> &components, uint16_t index) {
  if (index >= N)
    return std::nullopt;
  const PointerComponentHint &component = components[index];
  if (!is_known_component(component))
    return std::nullopt;
  return component;
}

[[nodiscard]] std::optional<PointerComponentHint>
component_hint_for_ref(const FlatPointerTracker &tracker, RegisterRef ref) {
  if (ref.cls == RegClass::SGPR)
    return component_hint_at(tracker.sgprs, ref.index);
  if (ref.cls == RegClass::VGPR)
    return component_hint_at(tracker.vgprs, ref.index);
  return std::nullopt;
}

[[nodiscard]] std::optional<PointerComponentHint>
source_component_hint(const Operand *operand, const FlatPointerTracker &tracker) {
  auto ref = register_ref(operand);
  if (!ref)
    return std::nullopt;
  return component_hint_for_ref(tracker, *ref);
}

template <size_t N>
void set_pair_components(std::array<PointerComponentHint, N> &components, uint16_t index,
                         ConSanFlatAddressSpaceHint space, bool maybe) {
  if (index >= N || static_cast<size_t>(index) + 1 >= N)
    return;
  components[index] = PointerComponentHint{space, PointerHalfHint::Low, maybe};
  components[index + 1] = PointerComponentHint{space, PointerHalfHint::High, maybe};
}

void set_sgpr_pair_components(FlatPointerTracker &tracker, uint16_t index,
                              ConSanFlatAddressSpaceHint space, bool maybe) {
  set_pair_components(tracker.sgprs, index, space, maybe);
}

void set_vgpr_component(FlatPointerTracker &tracker, uint16_t index,
                        PointerComponentHint component) {
  if (index >= tracker.vgprs.size())
    return;
  tracker.vgprs[index] = component;
}

void set_vgpr_pair_components(FlatPointerTracker &tracker, uint16_t index,
                              ConSanFlatAddressSpaceHint space, bool maybe) {
  set_pair_components(tracker.vgprs, index, space, maybe);
}

void set_sgpr_component(FlatPointerTracker &tracker, uint16_t index,
                        PointerComponentHint component) {
  if (index >= tracker.sgprs.size())
    return;
  tracker.sgprs[index] = component;
}

template <size_t N>
[[nodiscard]] std::optional<FlatPointerPairHint>
pair_hint_at(const std::array<PointerComponentHint, N> &components, uint16_t index) {
  if (index >= N)
    return std::nullopt;

  const std::optional<PointerComponentHint> low = component_hint_at(components, index);
  const std::optional<PointerComponentHint> high =
      static_cast<size_t>(index) + 1 < N
          ? component_hint_at(components, static_cast<uint16_t>(index + 1))
          : std::nullopt;

  if (low && high && low->space == high->space && low->half == PointerHalfHint::Low &&
      high->half == PointerHalfHint::High)
    return FlatPointerPairHint{low->space, low->maybe || high->maybe};

  if (high && high->half == PointerHalfHint::High)
    return FlatPointerPairHint{high->space, true};

  if (low && low->half == PointerHalfHint::Low)
    return FlatPointerPairHint{low->space, true};

  return std::nullopt;
}

[[nodiscard]] std::optional<FlatPointerPairHint>
source_pair_hint(const Operand *operand, const FlatPointerTracker &tracker) {
  auto ref = register_ref(operand);
  if (!ref)
    return std::nullopt;
  if (ref->cls == RegClass::SGPR)
    return pair_hint_at(tracker.sgprs, ref->index);
  if (ref->cls == RegClass::VGPR)
    return pair_hint_at(tracker.vgprs, ref->index);
  return std::nullopt;
}

[[nodiscard]] std::optional<PointerComponentHint>
merge_select_components(std::optional<PointerComponentHint> lhs,
                        std::optional<PointerComponentHint> rhs) {
  if (lhs && rhs && lhs->space == rhs->space && lhs->half == rhs->half)
    return PointerComponentHint{lhs->space, lhs->half, true};
  if (lhs && !rhs)
    return PointerComponentHint{lhs->space, lhs->half, true};
  if (!lhs && rhs)
    return PointerComponentHint{rhs->space, rhs->half, true};
  return std::nullopt;
}

[[nodiscard]] std::optional<PointerComponentHint>
merge_arithmetic_components(std::optional<PointerComponentHint> lhs,
                            std::optional<PointerComponentHint> rhs) {
  if (lhs && rhs && lhs->space == rhs->space && lhs->half == rhs->half)
    return PointerComponentHint{lhs->space, lhs->half, true};
  if (lhs && !rhs)
    return PointerComponentHint{lhs->space, lhs->half, true};
  if (!lhs && rhs)
    return PointerComponentHint{rhs->space, rhs->half, true};
  return std::nullopt;
}

[[nodiscard]] std::optional<PointerComponentHint>
merge_source_component_hints(const Instruction &inst, const FlatPointerTracker &tracker) {
  std::optional<ConSanFlatAddressSpaceHint> space;
  bool saw_low = false;
  bool saw_high = false;
  bool maybe = false;
  for (int i = 0; i < inst.num_src_operands(); ++i) {
    const auto hint = source_component_hint(inst.src_operand(i), tracker);
    if (!hint)
      continue;
    if (space && *space != hint->space)
      return std::nullopt;
    space = hint->space;
    saw_low |= hint->half == PointerHalfHint::Low;
    saw_high |= hint->half == PointerHalfHint::High;
    maybe |= hint->maybe;
  }
  if (!space)
    return std::nullopt;
  // Carry-chain encodings expose the low-half carry producer as a source of
  // the high-half add. Prefer an observed high component rather than treating
  // that mixed-half source list as contradictory provenance.
  return PointerComponentHint{*space, saw_high ? PointerHalfHint::High : PointerHalfHint::Low,
                              maybe || (saw_low && saw_high)};
}

[[nodiscard]] std::optional<FlatPointerPairHint>
merge_pair_arithmetic(std::optional<FlatPointerPairHint> lhs,
                      std::optional<FlatPointerPairHint> rhs) {
  if (lhs && rhs && lhs->space == rhs->space)
    return FlatPointerPairHint{lhs->space, true};
  if (lhs && !rhs)
    return FlatPointerPairHint{lhs->space, true};
  if (!lhs && rhs)
    return FlatPointerPairHint{rhs->space, true};
  return std::nullopt;
}

void set_sgpr_pair_from_hint(FlatPointerTracker &tracker, uint16_t index,
                             FlatPointerPairHint hint) {
  if (!is_known_flat_space(hint.space))
    return;
  set_sgpr_pair_components(tracker, index, hint.space, hint.maybe);
}

[[nodiscard]] std::optional<uint16_t> first_vgpr_dst_index(const Instruction &inst) {
  for (int i = 0; i < inst.num_dst_operands(); ++i) {
    if (const auto dst = register_index(inst.dst_operand(i), RegClass::VGPR))
      return dst;
  }
  return std::nullopt;
}

[[nodiscard]] bool is_vector_pointer_component_arithmetic(std::string_view mnemonic) {
  return starts_with_any(mnemonic, {"v_add_co_u32", "v_add_co_ci_u32", "v_add_nc_u32",
                                    "v_sub_co_u32", "v_sub_co_ci_u32", "v_sub_nc_u32"});
}

[[nodiscard]] uint64_t scratch_component_key(uint32_t saddr, uint32_t ioffset) {
  return (static_cast<uint64_t>(saddr) << 32u) | ioffset;
}

void invalidate_scratch_components_for_scalar_writes(const Instruction &inst,
                                                     FlatPointerTracker &tracker) {
  for (int i = 0; i < inst.num_dst_operands(); ++i) {
    const auto dst = register_ref(inst.dst_operand(i), RegClass::SGPR);
    if (!dst)
      continue;
    const uint32_t end = static_cast<uint32_t>(dst->index) + dst->width;
    std::erase_if(tracker.scratch_components, [&](const auto &entry) {
      const uint32_t saddr = static_cast<uint32_t>(entry.first >> 32u);
      return saddr >= dst->index && saddr < end;
    });
  }
}

[[nodiscard]] bool update_scratch_pointer_components(const Instruction &inst,
                                                     FlatPointerTracker &tracker,
                                                     rj_code_arch_t arch,
                                                     std::span<const uint8_t> instruction_bytes) {
  const std::string_view mnemonic = inst.mnemonic();
  const bool load = mnemonic == "scratch_load_b32" || mnemonic == "scratch_load_b64";
  const bool store = mnemonic == "scratch_store_b32" || mnemonic == "scratch_store_b64";
  if (!load && !store)
    return false;

  uint32_t encoding = 0;
  uint32_t vaddr = 0;
  uint32_t vdst = 0;
  uint32_t vsrc = 0;
  uint32_t saddr = 0;
  uint32_t ioffset = 0;
  if (arch == ROCJITSU_CODE_ARCH_RDNA4 &&
      instruction_bytes.size() == sizeof(rdna4::VscratchMachineInst)) {
    rdna4::VscratchMachineInst raw{};
    std::memcpy(&raw, instruction_bytes.data(), sizeof(raw));
    encoding = raw.encoding;
    vaddr = raw.vaddr;
    vdst = raw.vdst;
    vsrc = raw.vsrc;
    saddr = raw.saddr;
    ioffset = raw.ioffset;
  } else if (arch == ROCJITSU_CODE_ARCH_GFX1250 &&
             instruction_bytes.size() == sizeof(gfx1250::VscratchMachineInst)) {
    gfx1250::VscratchMachineInst raw{};
    std::memcpy(&raw, instruction_bytes.data(), sizeof(raw));
    encoding = raw.encoding;
    vaddr = raw.vaddr;
    vdst = raw.vdst;
    vsrc = raw.vsrc;
    saddr = raw.saddr;
    ioffset = raw.ioffset;
  } else {
    return false;
  }
  // A vector address makes the effective slot lane-dependent and prevents an
  // exact store/load match. Fail closed instead of aliasing such traffic.
  if (encoding != 0xedu || vaddr != 0u)
    return false;

  const uint32_t width = mnemonic.ends_with("b64") ? 2u : 1u;
  const uint16_t vgpr = static_cast<uint16_t>(load ? vdst : vsrc);
  if (static_cast<uint32_t>(vgpr) + width > tracker.vgprs.size())
    return false;

  if (load)
    clear_components(tracker, RegisterRef{RegClass::VGPR, vgpr, static_cast<uint8_t>(width)});

  for (uint32_t half = 0; half < width; ++half) {
    const uint32_t offset = ioffset + half * sizeof(uint32_t);
    const uint64_t key = scratch_component_key(saddr, offset);
    if (store) {
      const auto component = component_hint_at(tracker.vgprs, static_cast<uint16_t>(vgpr + half));
      if (component)
        tracker.scratch_components[key] = *component;
      else
        tracker.scratch_components.erase(key);
    } else if (const auto it = tracker.scratch_components.find(key);
               it != tracker.scratch_components.end()) {
      set_vgpr_component(tracker, static_cast<uint16_t>(vgpr + half), it->second);
    }
  }
  return true;
}

[[nodiscard]] std::optional<PrivateComponentKey>
private_component_key(const FlatPointerTracker &tracker, uint16_t address_vgpr,
                      uint32_t byte_offset) {
  if (address_vgpr + 1u >= tracker.vgpr_scalar_origins.size())
    return std::nullopt;
  const ScalarRegisterOrigin low = tracker.vgpr_scalar_origins[address_vgpr];
  const ScalarRegisterOrigin high = tracker.vgpr_scalar_origins[address_vgpr + 1u];
  if (!low.valid || !high.valid || high.index != low.index + 1u)
    return std::nullopt;
  return PrivateComponentKey{.low_sgpr = low.index,
                             .low_generation = low.generation,
                             .high_generation = high.generation,
                             .byte_offset = byte_offset};
}

[[nodiscard]] bool update_private_pointer_components(const Instruction &inst,
                                                     FlatPointerTracker &tracker,
                                                     rj_code_arch_t arch,
                                                     std::span<const uint8_t> instruction_bytes) {
  const std::string_view mnemonic = inst.mnemonic();
  const bool load = mnemonic == "flat_load_b32" || mnemonic == "flat_load_b64";
  const bool store = mnemonic == "flat_store_b32" || mnemonic == "flat_store_b64";
  if ((!load && !store) || arch != ROCJITSU_CODE_ARCH_RDNA4 ||
      instruction_bytes.size() != sizeof(rdna4::VflatMachineInst))
    return false;

  rdna4::VflatMachineInst raw{};
  std::memcpy(&raw, instruction_bytes.data(), sizeof(raw));
  if (raw.encoding != 0xecu)
    return false;
  const auto address_hint = pair_hint_at(tracker.vgprs, static_cast<uint16_t>(raw.vaddr));
  if (!address_hint || address_hint->space != ConSanFlatAddressSpaceHint::Private)
    return false;

  const uint32_t width = mnemonic.ends_with("b64") ? 2u : 1u;
  const uint16_t data_vgpr = static_cast<uint16_t>(load ? raw.vdst : raw.vsrc);
  if (static_cast<uint32_t>(data_vgpr) + width > tracker.vgprs.size())
    return false;
  if (load)
    clear_components(tracker, RegisterRef{RegClass::VGPR, data_vgpr, static_cast<uint8_t>(width)});

  for (uint32_t half = 0; half < width; ++half) {
    const auto key =
        private_component_key(tracker, static_cast<uint16_t>(raw.vaddr),
                              static_cast<uint32_t>(raw.ioffset) + half * sizeof(uint32_t));
    if (!key)
      return false;
    if (store) {
      const auto component =
          component_hint_at(tracker.vgprs, static_cast<uint16_t>(data_vgpr + half));
      if (component)
        tracker.private_components[*key] = *component;
      else
        tracker.private_components.erase(*key);
    } else if (const auto it = tracker.private_components.find(*key);
               it != tracker.private_components.end()) {
      set_vgpr_component(tracker, static_cast<uint16_t>(data_vgpr + half), it->second);
    }
  }
  return true;
}

void update_flat_pointer_tracker(const Instruction &inst, FlatPointerTracker &tracker,
                                 rj_code_arch_t arch, std::span<const uint8_t> instruction_bytes) {
  const std::string_view mnemonic = inst.mnemonic();

  invalidate_scratch_components_for_scalar_writes(inst, tracker);
  if (update_scratch_pointer_components(inst, tracker, arch, instruction_bytes))
    return;
  if (update_private_pointer_components(inst, tracker, arch, instruction_bytes))
    return;

  if (mnemonic == "v_writelane_b32" || mnemonic == "v_readlane_b32") {
    std::optional<uint32_t> lane_selector;
    if (arch == ROCJITSU_CODE_ARCH_RDNA4 &&
        instruction_bytes.size() == sizeof(rdna4::Vop3MachineInst)) {
      rdna4::Vop3MachineInst raw{};
      std::memcpy(&raw, instruction_bytes.data(), sizeof(raw));
      lane_selector = static_cast<uint32_t>(raw.src1);
    } else if (arch == ROCJITSU_CODE_ARCH_GFX1250 &&
               instruction_bytes.size() == sizeof(gfx1250::Vop3MachineInst)) {
      gfx1250::Vop3MachineInst raw{};
      std::memcpy(&raw, instruction_bytes.data(), sizeof(raw));
      lane_selector = static_cast<uint32_t>(raw.src1);
    }
    if (!lane_selector)
      return;
    const std::optional<uint32_t> lane = *lane_selector < 64u ? lane_selector
                                         : *lane_selector >= 128u && *lane_selector < 192u
                                             ? std::optional<uint32_t>(*lane_selector - 128u)
                                             : std::nullopt;
    if (lane && *lane < 64u) {
      if (mnemonic == "v_writelane_b32") {
        const auto vgpr = register_index(inst.dst_operand(0), RegClass::VGPR);
        if (!vgpr)
          return;
        const uint32_t key = (static_cast<uint32_t>(*vgpr) << 6u) | *lane;
        // RDNA4 exposes the tied destination as source operand 0. gfx1250's
        // decoder exposes only the scalar value and lane selector.
        const int value_operand = arch == ROCJITSU_CODE_ARCH_GFX1250 ? 0 : 1;
        const auto component = source_component_hint(inst.src_operand(value_operand), tracker);
        tracker.vgprs[*vgpr] = {};
        tracker.vgpr_scalar_origins[*vgpr] = {};
        if (component)
          tracker.vgpr_lane_components[key] = *component;
        else
          tracker.vgpr_lane_components.erase(key);
      } else {
        const auto vgpr = register_index(inst.src_operand(0), RegClass::VGPR);
        if (!vgpr)
          return;
        const uint32_t key = (static_cast<uint32_t>(*vgpr) << 6u) | *lane;
        const auto component = tracker.vgpr_lane_components.find(key);
        const auto sgpr = register_index(inst.dst_operand(0), RegClass::SGPR);
        clear_destination_components(inst, tracker);
        if (component != tracker.vgpr_lane_components.end() && sgpr)
          set_sgpr_component(tracker, *sgpr, component->second);
      }
      return;
    }
  }

  if (mnemonic.starts_with("v_lshrrev_b64") || mnemonic.starts_with("v_lshlrev_b64")) {
    const auto dst = register_ref(inst.dst_operand(0), RegClass::VGPR);
    std::optional<FlatPointerPairHint> source;
    for (int i = 0; i < inst.num_src_operands() && !source; ++i)
      source = source_pair_hint(inst.src_operand(i), tracker);
    clear_destination_components(inst, tracker);
    if (dst && dst->width >= 2u && source)
      set_vgpr_pair_components(tracker, dst->index, source->space, true);
    return;
  }

  if (mnemonic.starts_with("v_add_nc_u64") || mnemonic.starts_with("v_sub_nc_u64")) {
    const auto dst = register_ref(inst.dst_operand(0), RegClass::VGPR);
    std::optional<FlatPointerPairHint> merged;
    for (int i = 0; i < inst.num_src_operands(); ++i) {
      const auto source = source_pair_hint(inst.src_operand(i), tracker);
      if (source)
        merged = merge_pair_arithmetic(merged, source);
    }
    clear_destination_components(inst, tracker);
    if (dst && dst->width >= 2u && merged)
      set_vgpr_pair_components(tracker, dst->index, merged->space, true);
    return;
  }

  if (mnemonic == "s_mov_b64") {
    const auto dst = register_ref(inst.dst_operand(0), RegClass::SGPR);
    const auto base_space = special_pointer_base_space(inst.src_operand(0));
    const auto src_pair = source_pair_hint(inst.src_operand(0), tracker);
    clear_destination_components(inst, tracker);
    if (!dst || dst->width < 2)
      return;
    if (base_space) {
      set_sgpr_pair_components(tracker, dst->index, *base_space, false);
    } else if (src_pair) {
      set_sgpr_pair_from_hint(tracker, dst->index, *src_pair);
    }
    return;
  }

  if (mnemonic == "s_add_nc_u64" || mnemonic == "s_sub_nc_u64") {
    const auto dst = register_ref(inst.dst_operand(0), RegClass::SGPR);
    const auto src0 = source_pair_hint(inst.src_operand(0), tracker);
    const auto src1 = source_pair_hint(inst.src_operand(1), tracker);
    const auto merged = merge_pair_arithmetic(src0, src1);
    clear_destination_components(inst, tracker);
    if (dst && dst->width >= 2 && merged)
      set_sgpr_pair_from_hint(tracker, dst->index, *merged);
    return;
  }

  if (mnemonic == "s_mov_b32") {
    const auto dst = register_index(inst.dst_operand(0), RegClass::SGPR);
    const auto src = source_component_hint(inst.src_operand(0), tracker);
    clear_destination_components(inst, tracker);
    if (dst && src)
      set_sgpr_component(tracker, *dst, *src);
    return;
  }

  if (mnemonic == "s_cselect_b32") {
    const auto dst = register_index(inst.dst_operand(0), RegClass::SGPR);
    const auto selected =
        merge_select_components(source_component_hint(inst.src_operand(0), tracker),
                                source_component_hint(inst.src_operand(1), tracker));
    clear_destination_components(inst, tracker);
    if (dst && selected)
      set_sgpr_component(tracker, *dst, *selected);
    return;
  }

  if (starts_with_any(mnemonic, {"s_add", "s_sub", "s_and"})) {
    const auto dst = register_index(inst.dst_operand(0), RegClass::SGPR);
    const auto selected =
        merge_arithmetic_components(source_component_hint(inst.src_operand(0), tracker),
                                    source_component_hint(inst.src_operand(1), tracker));
    clear_destination_components(inst, tracker);
    if (dst && selected)
      set_sgpr_component(tracker, *dst, *selected);
    return;
  }

  if (mnemonic.starts_with("v_mov_b32")) {
    const auto dst = register_index(inst.dst_operand(0), RegClass::VGPR);
    const auto src = source_component_hint(inst.src_operand(0), tracker);
    ScalarRegisterOrigin scalar_origin;
    if (const auto src_ref = register_ref(inst.src_operand(0))) {
      if (src_ref->cls == RegClass::SGPR && src_ref->index < tracker.sgpr_generations.size()) {
        scalar_origin = {.index = src_ref->index,
                         .generation = tracker.sgpr_generations[src_ref->index],
                         .valid = true};
      } else if (src_ref->cls == RegClass::VGPR &&
                 src_ref->index < tracker.vgpr_scalar_origins.size()) {
        scalar_origin = tracker.vgpr_scalar_origins[src_ref->index];
      }
    }
    clear_destination_components(inst, tracker);
    if (dst) {
      if (src)
        set_vgpr_component(tracker, *dst, *src);
      if (scalar_origin.valid)
        tracker.vgpr_scalar_origins[*dst] = scalar_origin;
    }
    return;
  }

  if (mnemonic.starts_with("v_cndmask_b32")) {
    const auto dst = register_index(inst.dst_operand(0), RegClass::VGPR);
    const auto selected =
        merge_select_components(source_component_hint(inst.src_operand(0), tracker),
                                source_component_hint(inst.src_operand(1), tracker));
    clear_destination_components(inst, tracker);
    if (dst && selected)
      set_vgpr_component(tracker, *dst, *selected);
    return;
  }

  if (is_vector_pointer_component_arithmetic(mnemonic)) {
    const auto dst = first_vgpr_dst_index(inst);
    const auto selected = merge_source_component_hints(inst, tracker);
    clear_destination_components(inst, tracker);
    if (dst && selected)
      set_vgpr_component(tracker, *dst, *selected);
    return;
  }

  clear_destination_components(inst, tracker);
}

[[nodiscard]] ConSanFlatAddressSpaceHint
flat_address_space_hint(const Instruction &inst, const FlatPointerTracker &tracker) {
  const std::optional<uint16_t> addr = vgpr_index(inst.src_operand(0));
  if (!addr)
    return ConSanFlatAddressSpaceHint::Unknown;
  const auto hint = pair_hint_at(tracker.vgprs, *addr);
  if (!hint)
    return ConSanFlatAddressSpaceHint::Unknown;
  return flat_pair_hint_to_public(*hint);
}

void count_flat_address_space_hint(ConSanKernelStats &stats, ConSanFlatAddressSpaceHint hint) {
  switch (hint) {
  case ConSanFlatAddressSpaceHint::Group:
    ++stats.flat_group_hint_count;
    break;
  case ConSanFlatAddressSpaceHint::Private:
    ++stats.flat_private_hint_count;
    break;
  case ConSanFlatAddressSpaceHint::MaybeGroup:
    ++stats.flat_maybe_group_hint_count;
    break;
  case ConSanFlatAddressSpaceHint::MaybePrivate:
    ++stats.flat_maybe_private_hint_count;
    break;
  case ConSanFlatAddressSpaceHint::Global:
    ++stats.flat_global_hint_count;
    break;
  case ConSanFlatAddressSpaceHint::Unknown:
    ++stats.flat_unknown_hint_count;
    break;
  }
}

[[nodiscard]] bool is_fence_like(std::string_view mnemonic) {
  return starts_with_any(mnemonic, {"s_dcache", "s_icache", "global_wb", "global_inv",
                                    "global_wbinv", "buffer_wb", "buffer_inv", "buffer_wbinv"});
}

[[nodiscard]] bool is_atomic_instruction(const Instruction &inst) {
  const std::string_view mnemonic = inst.mnemonic();
  if (mnemonic.starts_with("ds_"))
    return lds_access_kind(mnemonic) == ConSanLdsAccessKind::Atomic;
  return starts_with_any(
      mnemonic, {"flat_atomic", "global_atomic", "scratch_atomic", "buffer_atomic", "s_atomic"});
}

[[nodiscard]] ConSanAtomicAddressSpaceHint
flat_hint_to_atomic_hint(ConSanFlatAddressSpaceHint hint) {
  switch (hint) {
  case ConSanFlatAddressSpaceHint::Group:
    return ConSanAtomicAddressSpaceHint::FlatGroup;
  case ConSanFlatAddressSpaceHint::Private:
    return ConSanAtomicAddressSpaceHint::FlatPrivate;
  case ConSanFlatAddressSpaceHint::MaybeGroup:
    return ConSanAtomicAddressSpaceHint::FlatMaybeGroup;
  case ConSanFlatAddressSpaceHint::MaybePrivate:
    return ConSanAtomicAddressSpaceHint::FlatMaybePrivate;
  case ConSanFlatAddressSpaceHint::Global:
    return ConSanAtomicAddressSpaceHint::FlatGlobal;
  case ConSanFlatAddressSpaceHint::Unknown:
    return ConSanAtomicAddressSpaceHint::FlatUnknown;
  }
  return ConSanAtomicAddressSpaceHint::Unknown;
}

[[nodiscard]] ConSanAtomicAddressSpaceHint
atomic_address_space_hint(const Instruction &inst, const FlatPointerTracker &tracker) {
  const std::string_view mnemonic = inst.mnemonic();
  if (mnemonic.starts_with("ds_") && lds_access_kind(mnemonic) == ConSanLdsAccessKind::Atomic)
    return ConSanAtomicAddressSpaceHint::Lds;
  if (mnemonic.starts_with("flat_atomic"))
    return flat_hint_to_atomic_hint(flat_address_space_hint(inst, tracker));
  if (mnemonic.starts_with("global_atomic"))
    return ConSanAtomicAddressSpaceHint::Global;
  if (mnemonic.starts_with("scratch_atomic"))
    return ConSanAtomicAddressSpaceHint::Scratch;
  if (mnemonic.starts_with("buffer_atomic"))
    return ConSanAtomicAddressSpaceHint::Buffer;
  if (mnemonic.starts_with("s_atomic"))
    return ConSanAtomicAddressSpaceHint::Scalar;
  return ConSanAtomicAddressSpaceHint::Unknown;
}

[[nodiscard]] std::optional<bool> atomic_returns_from_mnemonic(std::string_view mnemonic) {
  if (mnemonic.find("_rtn") != std::string_view::npos ||
      mnemonic.find("_return") != std::string_view::npos)
    return true;
  if (mnemonic.starts_with("ds_") && is_ds_atomic(mnemonic))
    return false;
  return std::nullopt;
}

[[nodiscard]] int32_t sign_extend_24(uint32_t value) {
  return static_cast<int32_t>(value << 8) >> 8;
}

template <typename Raw> void fill_flat_like_atomic_raw(ConSanAtomicSite &site, const Raw &raw) {
  site.raw_op = static_cast<uint32_t>(raw.op);
  site.raw_saddr = static_cast<uint32_t>(raw.saddr);
  site.raw_vaddr = static_cast<uint32_t>(raw.vaddr);
  site.raw_vsrc = static_cast<uint32_t>(raw.vsrc);
  site.raw_vdst = static_cast<uint32_t>(raw.vdst);
  site.raw_ioffset = sign_extend_24(static_cast<uint32_t>(raw.ioffset));
  site.raw_scope = static_cast<uint32_t>(raw.scope);
  site.raw_th = static_cast<uint32_t>(raw.th);
  site.returns_old_value = amdgpu::gfx12_atomic_returns(static_cast<uint8_t>(raw.th));
}

void fill_cdna4_flat_atomic_raw(ConSanAtomicSite &site, const cdna4::FlatMachineInst &raw) {
  site.raw_op = static_cast<uint32_t>(raw.op);
  site.raw_saddr = static_cast<uint32_t>(raw.saddr);
  site.raw_vaddr = static_cast<uint32_t>(raw.addr);
  site.raw_vsrc = static_cast<uint32_t>(raw.data);
  site.raw_vdst = static_cast<uint32_t>(raw.vdst);
  site.raw_ioffset = static_cast<int32_t>(raw.offset);
  // CDNA4 FLAT atomics do not carry gfx12's explicit scope field. ConSan's
  // admitted forms are device-scoped, matching the builder contract.
  site.raw_scope = 2u;
  site.returns_old_value = raw.sc0 != 0u;
}

template <typename Raw> void fill_buffer_atomic_raw(ConSanAtomicSite &site, const Raw &raw) {
  site.raw_op = static_cast<uint32_t>(raw.op);
  site.raw_vdata = static_cast<uint32_t>(raw.vdata);
  site.raw_rsrc = static_cast<uint32_t>(raw.rsrc);
  site.raw_soffset = static_cast<uint32_t>(raw.soffset);
  site.raw_vaddr = static_cast<uint32_t>(raw.vaddr);
  site.raw_ioffset = sign_extend_24(static_cast<uint32_t>(raw.ioffset));
  site.raw_scope = static_cast<uint32_t>(raw.scope);
  site.raw_th = static_cast<uint32_t>(raw.th);
  site.returns_old_value = amdgpu::gfx12_atomic_returns(static_cast<uint8_t>(raw.th));
}

template <typename Raw> void fill_ds_atomic_raw(ConSanAtomicSite &site, const Raw &raw) {
  site.raw_op = static_cast<uint32_t>(raw.op);
  site.raw_addr = static_cast<uint32_t>(raw.addr);
  site.raw_data0 = static_cast<uint32_t>(raw.data0);
  site.raw_data1 = static_cast<uint32_t>(raw.data1);
  site.raw_vdst = static_cast<uint32_t>(raw.vdst);
  // Single-address DS atomics consume offset0 as an unsigned byte offset.
  // Preserve it in the common effective-address field so synchronization
  // engines can key the exact LDS byte addressed by the guest instruction.
  site.raw_ioffset = static_cast<int32_t>(raw.offset0);
}

[[nodiscard]] bool is_barrier_instruction(const Instruction &inst) {
  const std::string_view mnemonic = inst.mnemonic();
  return inst.is_barrier() || mnemonic.starts_with("s_barrier") ||
         mnemonic.find("barrier") != std::string_view::npos;
}

void classify_instruction(const Instruction &inst, ConSanKernelStats &stats) {
  const std::string_view mnemonic = inst.mnemonic();

  if (mnemonic.starts_with("ds_") && !is_ds_register_lane_operation(mnemonic)) {
    switch (lds_access_kind(mnemonic)) {
    case ConSanLdsAccessKind::Atomic:
      ++stats.lds_atomic_count;
      break;
    case ConSanLdsAccessKind::Read:
      ++stats.lds_read_count;
      break;
    case ConSanLdsAccessKind::Write:
      ++stats.lds_write_count;
      break;
    case ConSanLdsAccessKind::Other:
      ++stats.ds_other_count;
      break;
    }
  }

  if (mnemonic.starts_with("flat_load")) {
    ++stats.flat_read_count;
  } else if (mnemonic.starts_with("flat_store")) {
    ++stats.flat_write_count;
  } else if (mnemonic.starts_with("flat_atomic")) {
    ++stats.flat_atomic_count;
  }

  if (mnemonic.starts_with("global_"))
    ++stats.global_memory_count;
  if (mnemonic.starts_with("scratch_"))
    ++stats.scratch_memory_count;

  if (is_barrier_instruction(inst))
    ++stats.barrier_count;
  if (inst.is_waitcnt() || mnemonic.starts_with("s_wait"))
    ++stats.wait_count;
  if (is_fence_like(mnemonic))
    ++stats.fence_like_count;
}

void record_lds_site(const Instruction &inst, ConSanKernelInfo &kernel,
                     uint64_t instruction_text_offset) {
  const std::string_view mnemonic = inst.mnemonic();
  if (!mnemonic.starts_with("ds_") || is_ds_register_lane_operation(mnemonic))
    return;

  ConSanLdsSite site;
  site.kind = lds_access_kind(mnemonic);
  site.text_offset = instruction_text_offset;
  site.file_offset = kernel.text_file_offset + instruction_text_offset;
  site.size = static_cast<uint32_t>(inst.size());
  site.width_bits = lds_width_bits(mnemonic);
  site.supported_mvp = is_supported_mvp_lds_site(site.kind, site.width_bits, mnemonic);
  if (site.kind == ConSanLdsAccessKind::Read) {
    site.dst_vgpr = vgpr_index(inst.dst_operand(0));
    site.addr_vgpr = vgpr_index(inst.src_operand(0));
  } else if (site.kind == ConSanLdsAccessKind::Write) {
    site.addr_vgpr = vgpr_index(inst.src_operand(0));
    site.data_vgpr = vgpr_index(inst.src_operand(1));
    site.second_data_vgpr = vgpr_index(inst.src_operand(2));
  } else if (site.kind == ConSanLdsAccessKind::Atomic) {
    site.dst_vgpr = vgpr_index(inst.dst_operand(0));
    site.addr_vgpr = vgpr_index(inst.src_operand(0));
    site.data_vgpr = vgpr_index(inst.src_operand(1));
    site.second_data_vgpr = vgpr_index(inst.src_operand(2));
  }
  site.mnemonic = std::string(mnemonic);
  kernel.lds_sites.push_back(std::move(site));
}

[[nodiscard]] ConSanBarrierSite::Scope barrier_scope_for_id(int32_t barrier_id) {
  switch (barrier_id) {
  case -4: // Cluster trap barrier.
  case -3: // Cluster barrier.
    return ConSanBarrierSite::Scope::Cluster;
  case -2: // Trap barrier.
  case -1: // Workgroup barrier.
    return ConSanBarrierSite::Scope::Workgroup;
  default:
    // RDNA4 named barriers 1-16 are workgroup scoped.
    return barrier_id >= 1 && barrier_id <= 16 ? ConSanBarrierSite::Scope::Workgroup
                                               : ConSanBarrierSite::Scope::Unknown;
  }
}

void decode_barrier_operand(const Instruction &inst, std::span<const uint8_t> instruction_bytes,
                            ConSanBarrierSite &site) {
  if (inst.mnemonic() == "s_barrier") {
    site.barrier_id = -1;
    site.operand_source = ConSanBarrierSite::OperandSource::Immediate;
    site.scope = ConSanBarrierSite::Scope::Workgroup;
    return;
  }
  if (inst.mnemonic() == "s_barrier_wait") {
    if (instruction_bytes.size() < sizeof(uint32_t))
      return;
    uint32_t word = 0;
    std::memcpy(&word, instruction_bytes.data(), sizeof(word));
    site.raw_simm16 = static_cast<uint16_t>(word);
    site.barrier_id = static_cast<int16_t>(word & 0xffffu);
    site.operand_source = ConSanBarrierSite::OperandSource::Immediate;
    site.scope = barrier_scope_for_id(*site.barrier_id);
    return;
  }
  if (inst.mnemonic() == "s_barrier_leave") {
    if (instruction_bytes.size() < sizeof(uint32_t))
      return;
    uint32_t word = 0;
    std::memcpy(&word, instruction_bytes.data(), sizeof(word));
    site.raw_simm16 = static_cast<uint16_t>(word);
    return;
  }
  const std::string_view mnemonic = inst.mnemonic();
  if (mnemonic != "s_barrier_signal" && mnemonic != "s_barrier_signal_isfirst" &&
      mnemonic != "s_barrier_init" && mnemonic != "s_barrier_join" &&
      mnemonic != "s_wakeup_barrier" && mnemonic != "s_get_barrier_state") {
    return;
  }
  if (instruction_bytes.size() < sizeof(uint32_t))
    return;
  uint32_t word = 0;
  std::memcpy(&word, instruction_bytes.data(), sizeof(word));
  const uint32_t source = word & 0xffu;
  site.raw_operand_selector = source;
  if (source == 125u) {
    site.operand_source = ConSanBarrierSite::OperandSource::DynamicM0;
    return;
  }
  if (source >= 128u && source <= 192u) {
    site.barrier_id = static_cast<int32_t>(source - 128u);
  } else if (source >= 193u && source <= 208u) {
    site.barrier_id = -static_cast<int32_t>(source - 192u);
  } else if (source == 255u && instruction_bytes.size() >= 2u * sizeof(uint32_t)) {
    int32_t literal = 0;
    std::memcpy(&literal, instruction_bytes.data() + sizeof(uint32_t), sizeof(literal));
    site.barrier_id = literal;
    site.operand_source = ConSanBarrierSite::OperandSource::Literal32;
    site.literal_width_bits = 32u;
    site.literal_value = static_cast<uint32_t>(literal);
    site.scope = barrier_scope_for_id(*site.barrier_id);
    return;
  } else if (source == 254u && instruction_bytes.size() >= 3u * sizeof(uint32_t)) {
    uint64_t literal = 0;
    std::memcpy(&literal, instruction_bytes.data() + sizeof(uint32_t), sizeof(literal));
    site.operand_source = ConSanBarrierSite::OperandSource::Literal64;
    site.literal_width_bits = 64u;
    site.literal_value = literal;
    return;
  } else {
    return;
  }
  site.operand_source = ConSanBarrierSite::OperandSource::Immediate;
  site.scope = barrier_scope_for_id(*site.barrier_id);
}

void record_barrier_site(const Instruction &inst, ConSanKernelInfo &kernel,
                         uint64_t instruction_text_offset,
                         std::span<const uint8_t> instruction_bytes, uint64_t previous_text_offset,
                         std::span<const uint8_t> previous_instruction_bytes, rj_code_arch_t arch) {
  if (!is_barrier_instruction(inst))
    return;

  ConSanBarrierSite site;
  site.text_offset = instruction_text_offset;
  site.file_offset = kernel.text_file_offset + instruction_text_offset;
  site.size = static_cast<uint32_t>(inst.size());
  site.mnemonic = std::string(inst.mnemonic());
  if (site.mnemonic == "s_barrier_signal" || site.mnemonic == "s_barrier_signal_isfirst")
    site.operation = ConSanBarrierSite::Operation::Signal;
  else if (site.mnemonic == "s_barrier_wait")
    site.operation = ConSanBarrierSite::Operation::Wait;
  else if (site.mnemonic == "s_barrier")
    site.operation = ConSanBarrierSite::Operation::Full;
  else if (site.mnemonic == "s_barrier_init")
    site.operation = ConSanBarrierSite::Operation::Init;
  else if (site.mnemonic == "s_barrier_join")
    site.operation = ConSanBarrierSite::Operation::Join;
  else if (site.mnemonic == "s_barrier_leave")
    site.operation = ConSanBarrierSite::Operation::Leave;
  else if (site.mnemonic == "s_wakeup_barrier")
    site.operation = ConSanBarrierSite::Operation::Wakeup;
  else if (site.mnemonic == "s_get_barrier_state")
    site.operation = ConSanBarrierSite::Operation::StateQuery;
  decode_barrier_operand(inst, instruction_bytes, site);
  // LLVM lowers a uniform named-barrier init count into M0[21:16] and the
  // barrier ID into M0[5:0]. Only accept the immediately preceding literal
  // s_mov_b32 m0 materialization: arbitrary M0 dataflow is not proof.
  if (arch == ROCJITSU_CODE_ARCH_GFX1250 && site.operation == ConSanBarrierSite::Operation::Init &&
      site.operand_source == ConSanBarrierSite::OperandSource::DynamicM0 &&
      previous_instruction_bytes.size() == 2u * sizeof(uint32_t)) {
    uint32_t setup_word = 0;
    uint32_t setup_literal = 0;
    std::memcpy(&setup_word, previous_instruction_bytes.data(), sizeof(setup_word));
    std::memcpy(&setup_literal, previous_instruction_bytes.data() + sizeof(uint32_t),
                sizeof(setup_literal));
    constexpr uint16_t kM0 = 125;
    constexpr uint16_t kLiteral32 = 255;
    constexpr uint32_t kBarrierIdMask = 0x3fu;
    constexpr uint32_t kParticipantCountMask = 0x3fu << 16u;
    const uint32_t barrier_id = setup_literal & kBarrierIdMask;
    const uint32_t participant_count = (setup_literal >> 16u) & 0x3fu;
    if (setup_word == build_s_mov_b32(kM0, kLiteral32, arch) && barrier_id >= 1u &&
        barrier_id <= 16u && participant_count >= 1u &&
        (setup_literal & ~(kBarrierIdMask | kParticipantCountMask)) == 0u) {
      site.barrier_id = static_cast<int32_t>(barrier_id);
      site.operand_source = ConSanBarrierSite::OperandSource::StaticM0Literal32;
      site.scope = ConSanBarrierSite::Scope::Workgroup;
      site.participant_encoding = ConSanBarrierSite::ParticipantEncoding::M0Literal32;
      site.participant_count = participant_count;
      site.participant_setup_text_offset = previous_text_offset;
      site.participant_setup_file_offset = kernel.text_file_offset + previous_text_offset;
      site.participant_setup_literal = setup_literal;
    }
  }
  kernel.barrier_sites.push_back(std::move(site));
}

void record_fence_site(const Instruction &inst, ConSanKernelInfo &kernel,
                       uint64_t instruction_text_offset) {
  if (!is_fence_like(inst.mnemonic()))
    return;

  ConSanFenceSite site;
  site.text_offset = instruction_text_offset;
  site.file_offset = kernel.text_file_offset + instruction_text_offset;
  site.size = static_cast<uint32_t>(inst.size());
  site.mnemonic = std::string(inst.mnemonic());
  kernel.fence_sites.push_back(std::move(site));
}

void record_flat_site(const Instruction &inst, const FlatPointerTracker &tracker,
                      ConSanKernelInfo &kernel, uint64_t instruction_text_offset,
                      rj_code_arch_t arch, std::span<const uint8_t> instruction_bytes) {
  const std::string_view mnemonic = inst.mnemonic();
  if (!mnemonic.starts_with("flat_"))
    return;

  ConSanFlatSite site;
  site.kind = flat_access_kind(mnemonic);
  site.text_offset = instruction_text_offset;
  site.file_offset = kernel.text_file_offset + instruction_text_offset;
  site.size = static_cast<uint32_t>(inst.size());
  site.width_bits = lds_width_bits(mnemonic);
  if (arch == ROCJITSU_CODE_ARCH_CDNA4 && mnemonic.starts_with("flat_atomic") &&
      site.width_bits == 0u) {
    site.width_bits = 32u;
  }
  site.address_space_hint = flat_address_space_hint(inst, tracker);
  if (site.kind == ConSanLdsAccessKind::Read) {
    site.dst_vgpr = vgpr_index(inst.dst_operand(0));
    site.addr_vgpr = vgpr_index(inst.src_operand(0));
  } else if (site.kind == ConSanLdsAccessKind::Write) {
    site.addr_vgpr = vgpr_index(inst.src_operand(0));
    site.data_vgpr = vgpr_index(inst.src_operand(1));
  } else if (site.kind == ConSanLdsAccessKind::Atomic) {
    site.addr_vgpr = vgpr_index(inst.src_operand(0));
    site.data_vgpr = vgpr_index(inst.src_operand(1));
  }
  if (arch == ROCJITSU_CODE_ARCH_CDNA4 &&
      instruction_bytes.size() >= sizeof(cdna4::FlatMachineInst)) {
    cdna4::FlatMachineInst raw{};
    std::memcpy(&raw, instruction_bytes.data(), sizeof(raw));
    site.raw_op = static_cast<uint32_t>(raw.op);
    site.raw_saddr = static_cast<uint32_t>(raw.saddr);
    site.raw_vaddr = static_cast<uint32_t>(raw.addr);
    site.raw_vsrc = static_cast<uint32_t>(raw.data);
    site.raw_vdst = static_cast<uint32_t>(raw.vdst);
    site.raw_ioffset = static_cast<int32_t>(raw.offset);
    site.raw_segment = static_cast<uint32_t>(raw.seg);
    site.raw_scope = 0u;
    site.raw_th = static_cast<uint32_t>(raw.sc0) | (static_cast<uint32_t>(raw.sc1) << 1u);
    if (raw.seg == 1u)
      site.address_space_hint = ConSanFlatAddressSpaceHint::Private;
    else if (raw.seg == 2u)
      site.address_space_hint = ConSanFlatAddressSpaceHint::Global;
  } else if (arch == ROCJITSU_CODE_ARCH_GFX1250 &&
             instruction_bytes.size() >= sizeof(gfx1250::VflatMachineInst)) {
    gfx1250::VflatMachineInst raw{};
    std::memcpy(&raw, instruction_bytes.data(), sizeof(raw));
    site.raw_op = static_cast<uint32_t>(raw.op);
    site.raw_saddr = static_cast<uint32_t>(raw.saddr);
    site.raw_vaddr = static_cast<uint32_t>(raw.vaddr);
    site.raw_vsrc = static_cast<uint32_t>(raw.vsrc);
    site.raw_vdst = static_cast<uint32_t>(raw.vdst);
    site.raw_ioffset = sign_extend_24(static_cast<uint32_t>(raw.ioffset));
    site.raw_scope = static_cast<uint32_t>(raw.scope);
    site.raw_th = static_cast<uint32_t>(raw.th);
  } else if (arch == ROCJITSU_CODE_ARCH_RDNA4 &&
             instruction_bytes.size() >= sizeof(rdna4::VflatMachineInst)) {
    rdna4::VflatMachineInst raw{};
    std::memcpy(&raw, instruction_bytes.data(), sizeof(raw));
    site.raw_op = static_cast<uint32_t>(raw.op);
    site.raw_saddr = static_cast<uint32_t>(raw.saddr);
    site.raw_vaddr = static_cast<uint32_t>(raw.vaddr);
    site.raw_vsrc = static_cast<uint32_t>(raw.vsrc);
    site.raw_vdst = static_cast<uint32_t>(raw.vdst);
    site.raw_ioffset = sign_extend_24(static_cast<uint32_t>(raw.ioffset));
    site.raw_scope = static_cast<uint32_t>(raw.scope);
    site.raw_th = static_cast<uint32_t>(raw.th);
  }
  count_flat_address_space_hint(kernel.stats, site.address_space_hint);
  site.mnemonic = std::string(mnemonic);
  kernel.flat_sites.push_back(std::move(site));
}

[[nodiscard]] bool is_bounded_ordinary_memory_instruction(std::string_view mnemonic) {
  return mnemonic == "flat_load_b32" || mnemonic == "flat_store_b32" ||
         mnemonic == "global_load_b32" || mnemonic == "global_store_b32" ||
         starts_with_any(mnemonic, {"buffer_load_b32", "buffer_load_b64", "buffer_load_b96",
                                    "buffer_load_b128", "buffer_store_b32", "buffer_store_b64",
                                    "buffer_store_b96", "buffer_store_b128"});
}

[[nodiscard]] uint32_t bounded_ordinary_memory_width(std::string_view mnemonic) {
  if (mnemonic.ends_with("b128"))
    return 128u;
  if (mnemonic.ends_with("b96"))
    return 96u;
  if (mnemonic.ends_with("b64"))
    return 64u;
  return 32u;
}

template <typename Raw>
void fill_ordinary_memory_raw(ConSanOrdinaryMemorySite &site, const Raw &raw) {
  site.raw_saddr = static_cast<uint32_t>(raw.saddr);
  site.raw_nv = static_cast<uint32_t>(raw.nv);
  site.raw_sve = static_cast<uint32_t>(raw.sve);
  site.raw_vaddr = static_cast<uint32_t>(raw.vaddr);
  site.raw_vsrc = static_cast<uint32_t>(raw.vsrc);
  site.raw_vdst = static_cast<uint32_t>(raw.vdst);
  site.raw_ioffset = sign_extend_24(static_cast<uint32_t>(raw.ioffset));
  site.raw_scope = static_cast<uint32_t>(raw.scope);
  site.raw_th = static_cast<uint32_t>(raw.th);
  site.address_vgpr = static_cast<uint16_t>(raw.vaddr);
  if (raw.saddr != rdna4::OPR_SREG_NULL)
    site.address_sgpr = static_cast<uint16_t>(raw.saddr);
  if (site.operation == ConSanOrdinaryMemoryOperation::Load)
    site.destination_vgpr = static_cast<uint16_t>(raw.vdst);
  else
    site.value_vgpr = static_cast<uint16_t>(raw.vsrc);
}

template <typename Raw>
[[nodiscard]] bool ordinary_memory_encoding_is_well_formed(const Raw &raw,
                                                           uint32_t expected_encoding) {
  return raw.encoding == expected_encoding && raw.pad_8_13 == 0u && raw.pad_22_23 == 0u &&
         raw.pad_40_48 == 0u && raw.pad_63 == 0u;
}

void record_ordinary_memory_site(const Instruction &inst, ConSanKernelInfo &kernel,
                                 uint64_t instruction_text_offset, rj_code_arch_t arch,
                                 std::span<const uint8_t> instruction_bytes) {
  const std::string_view mnemonic = inst.mnemonic();
  if (!is_bounded_ordinary_memory_instruction(mnemonic))
    return;

  ConSanOrdinaryMemorySite site;
  site.operation = mnemonic.find("load") != std::string_view::npos
                       ? ConSanOrdinaryMemoryOperation::Load
                       : ConSanOrdinaryMemoryOperation::Store;
  site.text_offset = instruction_text_offset;
  site.file_offset = kernel.text_file_offset + instruction_text_offset;
  site.size = static_cast<uint32_t>(inst.size());
  site.width_bits = bounded_ordinary_memory_width(mnemonic);
  site.mnemonic = std::string(mnemonic);

  // Operand objects remain useful on unsupported architectures, while the
  // raw RDNA4 fields below are the authoritative bounded decode.
  if (mnemonic.starts_with("buffer_")) {
    if (arch != ROCJITSU_CODE_ARCH_GFX1250) {
      site.support_reason = ConSanOrdinaryMemorySupportReason::UnsupportedArchitecture;
    } else if (instruction_bytes.size() != sizeof(gfx1250::VbufferMachineInst)) {
      site.support_reason = ConSanOrdinaryMemorySupportReason::UnsupportedEncodingSize;
    } else {
      gfx1250::VbufferMachineInst raw{};
      std::memcpy(&raw, instruction_bytes.data(), sizeof(raw));
      site.raw_ioffset = sign_extend_24(static_cast<uint32_t>(raw.ioffset));
      site.raw_scope = static_cast<uint32_t>(raw.scope);
      site.raw_th = static_cast<uint32_t>(raw.th);
      site.raw_rsrc = static_cast<uint32_t>(raw.rsrc);
      site.raw_soffset = static_cast<uint32_t>(raw.soffset);
      site.raw_offen = raw.offen != 0u;
      site.raw_idxen = raw.idxen != 0u;
      site.raw_vaddr = static_cast<uint32_t>(raw.vaddr);
      site.address_sgpr = static_cast<uint16_t>(raw.rsrc);
      if (raw.offen || raw.idxen)
        site.address_vgpr = static_cast<uint16_t>(raw.vaddr);
      if (site.operation == ConSanOrdinaryMemoryOperation::Load)
        site.destination_vgpr = static_cast<uint16_t>(raw.vdata);
      else
        site.value_vgpr = static_cast<uint16_t>(raw.vdata);
      const bool well_formed =
          raw.encoding == 0x31u && raw.pad_8_13 == 0u && raw.pad_23_25 == 0u && raw.pad_40 == 0u;
      site.support_reason = well_formed
                                ? ConSanOrdinaryMemorySupportReason::SupportedSynchronizationOnly
                                : ConSanOrdinaryMemorySupportReason::MalformedEncoding;
    }
    kernel.ordinary_memory_sites.push_back(std::move(site));
    return;
  }

  if (site.operation == ConSanOrdinaryMemoryOperation::Load) {
    site.destination_vgpr = vgpr_index(inst.dst_operand(0));
    site.address_vgpr = vgpr_index(inst.src_operand(0));
    if (mnemonic.starts_with("global_") && inst.num_src_operands() > 1)
      site.address_sgpr = register_index(inst.src_operand(1), RegClass::SGPR);
  } else {
    site.address_vgpr = vgpr_index(inst.src_operand(0));
    site.value_vgpr = vgpr_index(inst.src_operand(1));
    if (mnemonic.starts_with("global_") && inst.num_src_operands() > 2)
      site.address_sgpr = register_index(inst.src_operand(2), RegClass::SGPR);
  }

  if (arch != ROCJITSU_CODE_ARCH_RDNA4) {
    site.support_reason = ConSanOrdinaryMemorySupportReason::UnsupportedArchitecture;
  } else if (instruction_bytes.size() != (mnemonic.starts_with("flat_")
                                              ? sizeof(rdna4::VflatMachineInst)
                                              : sizeof(rdna4::VglobalMachineInst))) {
    site.support_reason = ConSanOrdinaryMemorySupportReason::UnsupportedEncodingSize;
  } else {
    bool well_formed = false;
    if (mnemonic.starts_with("flat_")) {
      rdna4::VflatMachineInst raw{};
      std::memcpy(&raw, instruction_bytes.data(), sizeof(raw));
      fill_ordinary_memory_raw(site, raw);
      well_formed = ordinary_memory_encoding_is_well_formed(raw, 0xecu);
    } else {
      rdna4::VglobalMachineInst raw{};
      std::memcpy(&raw, instruction_bytes.data(), sizeof(raw));
      fill_ordinary_memory_raw(site, raw);
      well_formed = ordinary_memory_encoding_is_well_formed(raw, 0xeeu);
    }
    if (!well_formed)
      site.support_reason = ConSanOrdinaryMemorySupportReason::MalformedEncoding;
    else if (!site.address_vgpr)
      site.support_reason = ConSanOrdinaryMemorySupportReason::MissingAddressVgpr;
    else if (site.operation == ConSanOrdinaryMemoryOperation::Load && !site.destination_vgpr)
      site.support_reason = ConSanOrdinaryMemorySupportReason::MissingDestinationVgpr;
    else if (site.operation == ConSanOrdinaryMemoryOperation::Store && !site.value_vgpr)
      site.support_reason = ConSanOrdinaryMemorySupportReason::MissingValueVgpr;
    else
      site.support_reason = ConSanOrdinaryMemorySupportReason::Supported;
  }
  kernel.ordinary_memory_sites.push_back(std::move(site));
}

void record_atomic_site(const Instruction &inst, const FlatPointerTracker &tracker,
                        ConSanKernelInfo &kernel, uint64_t instruction_text_offset,
                        rj_code_arch_t arch, std::span<const uint8_t> instruction_bytes) {
  if (!is_atomic_instruction(inst))
    return;

  const std::string_view mnemonic = inst.mnemonic();
  ConSanAtomicSite site;
  site.address_space_hint = atomic_address_space_hint(inst, tracker);
  site.text_offset = instruction_text_offset;
  site.file_offset = kernel.text_file_offset + instruction_text_offset;
  site.size = static_cast<uint32_t>(inst.size());
  site.width_bits = lds_width_bits(mnemonic);
  if (arch == ROCJITSU_CODE_ARCH_CDNA4 && mnemonic.starts_with("flat_atomic") &&
      site.width_bits == 0u) {
    site.width_bits = 32u;
  }
  site.dst_vgpr = vgpr_index(inst.dst_operand(0));
  site.addr_vgpr = vgpr_index(inst.src_operand(0));
  site.data_vgpr = vgpr_index(inst.src_operand(1));
  for (int i = 0; i < inst.num_src_operands(); ++i) {
    if (const auto sgpr = register_index(inst.src_operand(i), RegClass::SGPR)) {
      site.saddr_sgpr = *sgpr;
      break;
    }
  }
  site.returns_old_value = atomic_returns_from_mnemonic(mnemonic);

  if (arch == ROCJITSU_CODE_ARCH_CDNA4 && mnemonic.starts_with("flat_atomic") &&
      instruction_bytes.size() >= sizeof(cdna4::FlatMachineInst)) {
    cdna4::FlatMachineInst raw{};
    std::memcpy(&raw, instruction_bytes.data(), sizeof(raw));
    fill_cdna4_flat_atomic_raw(site, raw);
  } else if (arch == ROCJITSU_CODE_ARCH_RDNA4) {
    if (mnemonic.starts_with("ds_") && instruction_bytes.size() >= sizeof(rdna4::VdsMachineInst)) {
      rdna4::VdsMachineInst raw{};
      std::memcpy(&raw, instruction_bytes.data(), sizeof(raw));
      fill_ds_atomic_raw(site, raw);
    } else if (mnemonic.starts_with("flat_atomic") &&
               instruction_bytes.size() >= sizeof(rdna4::VflatMachineInst)) {
      rdna4::VflatMachineInst raw{};
      std::memcpy(&raw, instruction_bytes.data(), sizeof(raw));
      fill_flat_like_atomic_raw(site, raw);
    } else if (mnemonic.starts_with("global_atomic") &&
               instruction_bytes.size() >= sizeof(rdna4::VglobalMachineInst)) {
      rdna4::VglobalMachineInst raw{};
      std::memcpy(&raw, instruction_bytes.data(), sizeof(raw));
      fill_flat_like_atomic_raw(site, raw);
    } else if (mnemonic.starts_with("scratch_atomic") &&
               instruction_bytes.size() >= sizeof(rdna4::VscratchMachineInst)) {
      rdna4::VscratchMachineInst raw{};
      std::memcpy(&raw, instruction_bytes.data(), sizeof(raw));
      fill_flat_like_atomic_raw(site, raw);
    } else if (mnemonic.starts_with("buffer_atomic") &&
               instruction_bytes.size() >= sizeof(rdna4::VbufferMachineInst)) {
      rdna4::VbufferMachineInst raw{};
      std::memcpy(&raw, instruction_bytes.data(), sizeof(raw));
      fill_buffer_atomic_raw(site, raw);
    }
  } else if (arch == ROCJITSU_CODE_ARCH_GFX1250) {
    if (mnemonic.starts_with("ds_") &&
        instruction_bytes.size() >= sizeof(gfx1250::VdsMachineInst)) {
      gfx1250::VdsMachineInst raw{};
      std::memcpy(&raw, instruction_bytes.data(), sizeof(raw));
      fill_ds_atomic_raw(site, raw);
    } else if (mnemonic.starts_with("flat_atomic") &&
               instruction_bytes.size() >= sizeof(gfx1250::VflatMachineInst)) {
      gfx1250::VflatMachineInst raw{};
      std::memcpy(&raw, instruction_bytes.data(), sizeof(raw));
      fill_flat_like_atomic_raw(site, raw);
    } else if (mnemonic.starts_with("global_atomic") &&
               instruction_bytes.size() >= sizeof(gfx1250::VglobalMachineInst)) {
      gfx1250::VglobalMachineInst raw{};
      std::memcpy(&raw, instruction_bytes.data(), sizeof(raw));
      fill_flat_like_atomic_raw(site, raw);
    } else if (mnemonic.starts_with("scratch_atomic") &&
               instruction_bytes.size() >= sizeof(gfx1250::VscratchMachineInst)) {
      gfx1250::VscratchMachineInst raw{};
      std::memcpy(&raw, instruction_bytes.data(), sizeof(raw));
      fill_flat_like_atomic_raw(site, raw);
    } else if (mnemonic.starts_with("buffer_atomic") &&
               instruction_bytes.size() >= sizeof(gfx1250::VbufferMachineInst)) {
      gfx1250::VbufferMachineInst raw{};
      std::memcpy(&raw, instruction_bytes.data(), sizeof(raw));
      fill_buffer_atomic_raw(site, raw);
    }
  }

  // Some decoder revisions expose VGLOBAL's scalar address only through the
  // raw instruction field. Normalize that equivalent register selector so
  // downstream address consumers do not depend on operand-list presentation.
  if (mnemonic.starts_with("global_atomic") && !site.saddr_sgpr && site.raw_saddr &&
      *site.raw_saddr <= std::numeric_limits<uint16_t>::max()) {
    site.saddr_sgpr = static_cast<uint16_t>(*site.raw_saddr);
  }

  site.mnemonic = std::string(mnemonic);
  kernel.atomic_sites.push_back(std::move(site));
}

void decode_kernel_stats(std::span<const uint8_t> code_object_bytes, Decoder &decoder,
                         rj_code_arch_t arch, ConSanKernelInfo &kernel,
                         std::vector<std::string> &warnings,
                         const FlatPointerTracker *entry_tracker = nullptr,
                         const std::unordered_map<uint64_t, uint64_t> *call_targets = nullptr,
                         std::unordered_map<uint64_t, FlatPointerTracker> *call_states = nullptr) {
  if (!kernel.has_text_range) {
    warnings.emplace_back("ConSan cannot decode kernel '" + kernel.name +
                          "': function text range is unavailable");
    return;
  }
  if (kernel.code_size == 0) {
    warnings.emplace_back("ConSan cannot decode kernel '" + kernel.name +
                          "': function text range is empty");
    return;
  }
  if (kernel.entry_text_offset > code_object_bytes.size() ||
      kernel.text_file_offset > code_object_bytes.size() - kernel.entry_text_offset) {
    ++kernel.stats.decode_error_count;
    warnings.emplace_back("ConSan cannot decode kernel '" + kernel.name +
                          "': function text range starts outside the code object");
    return;
  }

  const uint64_t start = kernel.text_file_offset + kernel.entry_text_offset;
  if (start > code_object_bytes.size() || kernel.code_size > code_object_bytes.size() - start) {
    ++kernel.stats.decode_error_count;
    warnings.emplace_back("ConSan cannot decode kernel '" + kernel.name +
                          "': function text range extends outside the code object");
    return;
  }

  const auto text =
      code_object_bytes.subspan(static_cast<size_t>(start), static_cast<size_t>(kernel.code_size));
  FlatPointerTracker flat_pointer_tracker;
  if (entry_tracker != nullptr) {
    flat_pointer_tracker.sgprs = entry_tracker->sgprs;
    flat_pointer_tracker.vgprs = entry_tracker->vgprs;
  }
  size_t offset = 0;
  size_t previous_offset = 0;
  size_t previous_size = 0;
  while (offset < text.size()) {
    const size_t remaining = text.size() - offset;
    if (remaining < sizeof(uint32_t)) {
      ++kernel.stats.decode_error_count;
      warnings.emplace_back("ConSan stopped decoding kernel '" + kernel.name +
                            "': trailing bytes are smaller than one instruction word");
      break;
    }

    // Some generated kernel symbols include zero alignment padding after the
    // terminal s_endpgm. It is not executable code and opcode zero is not an
    // instruction. Retain internal/branch-target endpgm instructions, but
    // stop cleanly when the complete remaining symbol tail is padding.
    if (previous_size != 0u &&
        std::ranges::all_of(text.subspan(offset), [](uint8_t byte) { return byte == 0u; })) {
      uint32_t previous_word = 0;
      std::memcpy(&previous_word, text.data() + previous_offset, sizeof(previous_word));
      if (previous_word == build_s_endpgm(arch))
        break;
    }

    std::array<uint32_t, 4> words{};
    std::memcpy(words.data(), text.data() + offset,
                std::min(words.size() * sizeof(uint32_t), remaining));

    std::unique_ptr<Instruction> inst;
    try {
      inst.reset(
          decoder.decode(words.data(), kernel.entry_text_offset + static_cast<uint64_t>(offset)));
    } catch (const std::exception &error) {
      ++kernel.stats.decode_error_count;
      warnings.emplace_back(
          "ConSan stopped decoding kernel '" + kernel.name + "' at text offset " +
          std::to_string(kernel.entry_text_offset + static_cast<uint64_t>(offset)) + ": " +
          error.what());
      break;
    }
    if (!inst) {
      ++kernel.stats.decode_error_count;
      warnings.emplace_back("ConSan stopped decoding kernel '" + kernel.name +
                            "': decoder returned null");
      break;
    }

    const int inst_size = inst->size();
    if (inst_size <= 0 || inst_size % static_cast<int>(sizeof(uint32_t)) != 0) {
      ++kernel.stats.decode_error_count;
      warnings.emplace_back("ConSan stopped decoding kernel '" + kernel.name +
                            "': decoder returned an invalid instruction size");
      break;
    }
    if (static_cast<size_t>(inst_size) > remaining) {
      ++kernel.stats.decode_error_count;
      warnings.emplace_back("ConSan stopped decoding kernel '" + kernel.name +
                            "': decoded instruction exceeds the function text range");
      break;
    }

    ++kernel.stats.instruction_count;
    if (arch == ROCJITSU_CODE_ARCH_GFX1250) {
      for (int operand_index = 0; operand_index < inst->num_src_operands(); ++operand_index) {
        const Operand *operand = inst->src_operand(operand_index);
        // TTMPs are deliberately outside liveness's allocatable register
        // classes, so identify this architectural launch input by its decoded
        // operand name rather than treating it as an SGPR.
        if (operand && operand->name() == "TTMP6") {
          kernel.uses_gfx1250_cluster_workgroup_id = true;
          break;
        }
      }
    }
    const uint64_t instruction_text_offset =
        kernel.entry_text_offset + static_cast<uint64_t>(offset);
    if (call_targets != nullptr && call_states != nullptr &&
        call_targets->contains(instruction_text_offset))
      (*call_states)[instruction_text_offset] = flat_pointer_tracker;
    classify_instruction(*inst, kernel.stats);
    record_lds_site(*inst, kernel, kernel.entry_text_offset + static_cast<uint64_t>(offset));
    const std::span<const uint8_t> previous_instruction_bytes =
        previous_size == 0 ? std::span<const uint8_t>{}
                           : text.subspan(previous_offset, previous_size);
    record_barrier_site(*inst, kernel, kernel.entry_text_offset + static_cast<uint64_t>(offset),
                        text.subspan(offset, static_cast<size_t>(inst_size)),
                        kernel.entry_text_offset + static_cast<uint64_t>(previous_offset),
                        previous_instruction_bytes, arch);
    record_fence_site(*inst, kernel, kernel.entry_text_offset + static_cast<uint64_t>(offset));
    record_flat_site(*inst, flat_pointer_tracker, kernel,
                     kernel.entry_text_offset + static_cast<uint64_t>(offset), arch,
                     text.subspan(offset, static_cast<size_t>(inst_size)));
    record_ordinary_memory_site(*inst, kernel,
                                kernel.entry_text_offset + static_cast<uint64_t>(offset), arch,
                                text.subspan(offset, static_cast<size_t>(inst_size)));
    record_atomic_site(*inst, flat_pointer_tracker, kernel,
                       kernel.entry_text_offset + static_cast<uint64_t>(offset), arch,
                       text.subspan(offset, static_cast<size_t>(inst_size)));
    update_flat_pointer_tracker(*inst, flat_pointer_tracker, arch,
                                text.subspan(offset, static_cast<size_t>(inst_size)));
    previous_offset = offset;
    previous_size = static_cast<size_t>(inst_size);
    offset += previous_size;
  }

  kernel.decoded = true;
}

void decode_function_stats(
    std::span<const uint8_t> code_object_bytes, Decoder &decoder, rj_code_arch_t arch,
    ConSanFunctionInfo &function, std::vector<std::string> &warnings,
    const FlatPointerTracker *entry_tracker = nullptr,
    const std::unordered_map<uint64_t, uint64_t> *call_targets = nullptr,
    std::unordered_map<uint64_t, FlatPointerTracker> *call_states = nullptr) {
  ConSanKernelInfo range;
  range.name = function.name;
  range.entry_text_offset = function.entry_text_offset;
  range.text_file_offset = function.text_file_offset;
  range.code_size = function.code_size;
  range.has_text_range = true;

  decode_kernel_stats(code_object_bytes, decoder, arch, range, warnings, entry_tracker,
                      call_targets, call_states);

  function.decoded = range.decoded;
  function.uses_gfx1250_cluster_workgroup_id = range.uses_gfx1250_cluster_workgroup_id;
  function.stats = range.stats;
  function.lds_sites = std::move(range.lds_sites);
  function.flat_sites = std::move(range.flat_sites);
  function.ordinary_memory_sites = std::move(range.ordinary_memory_sites);
  function.barrier_sites = std::move(range.barrier_sites);
  function.fence_sites = std::move(range.fence_sites);
  function.atomic_sites = std::move(range.atomic_sites);

  const bool lds_load_api = function.name.find("lds_load_at") != std::string::npos;
  const bool lds_store_api = function.name.find("lds_store_at") != std::string::npos;
  if (lds_load_api || lds_store_api) {
    const ConSanLdsAccessKind expected_kind =
        lds_load_api ? ConSanLdsAccessKind::Read : ConSanLdsAccessKind::Write;
    const auto terminal = std::ranges::find_if(
        function.flat_sites.rbegin(), function.flat_sites.rend(),
        [&](const ConSanFlatSite &site) { return site.kind == expected_kind; });
    if (terminal != function.flat_sites.rend() &&
        (terminal->address_space_hint == ConSanFlatAddressSpaceHint::Unknown ||
         terminal->address_space_hint == ConSanFlatAddressSpaceHint::MaybePrivate)) {
      if (terminal->address_space_hint == ConSanFlatAddressSpaceHint::Unknown)
        --function.stats.flat_unknown_hint_count;
      else
        --function.stats.flat_maybe_private_hint_count;
      // Symbolized hip-moi helpers carry an explicit API contract: their
      // terminal load/store is the source-level `*ptr` LDS access. Treat only
      // that dereference as likely-group; strict provenance still excludes it.
      terminal->address_space_hint = ConSanFlatAddressSpaceHint::MaybeGroup;
      ++function.stats.flat_maybe_group_hint_count;
    }
  }
}

template <typename Container>
void prune_inferred_range_sites(Container &container, const BlockOffsetIndex &block_index,
                                const std::unordered_set<const BasicBlock *> &owned_blocks) {
  if (!container.code_size_inferred_from_zero)
    return;
  const auto is_unowned = [&](const auto &site) {
    const BasicBlock *block = block_for_offset(block_index, site.text_offset);
    return block == nullptr || !owned_blocks.contains(block);
  };
  std::erase_if(container.lds_sites, is_unowned);
  std::erase_if(container.flat_sites, is_unowned);
  std::erase_if(container.ordinary_memory_sites, is_unowned);
  std::erase_if(container.barrier_sites, is_unowned);
  std::erase_if(container.fence_sites, is_unowned);
  std::erase_if(container.atomic_sites, is_unowned);
}

// Assembly kernels commonly leave STT_FUNC sizes at zero. The code-object
// parser conservatively extends each such symbol to the next distinct function
// entry (or .text end for the final function), but generated objects can place
// additional, deliberately unreachable variants after a terminal path. Keep
// the parser's architecture-neutral symbol range intact while restricting
// ConSan's inventory in only these inferred ranges to instructions with a
// validated kernel CFG owner.
void prune_unreachable_inferred_ranges(const AmdGpuCodeObject &code_object, Decoder &decoder,
                                       rj_code_arch_t arch, const ConSanOptions &options,
                                       ConSanResult &result) {
  const bool needs_pruning =
      std::ranges::any_of(result.kernels, &ConSanKernelInfo::code_size_inferred_from_zero) ||
      std::ranges::any_of(result.functions, &ConSanFunctionInfo::code_size_inferred_from_zero);
  if (!needs_pruning)
    return;

  std::vector<uint64_t> leaders;
  std::vector<uint64_t> kernel_entries;
  for (const ConSanKernelInfo &kernel : result.kernels) {
    if (!kernel.has_text_range)
      continue;
    leaders.push_back(kernel.entry_text_offset);
    kernel_entries.push_back(kernel.entry_text_offset);
  }
  for (const ConSanFunctionInfo &function : result.functions)
    leaders.push_back(function.entry_text_offset);
  for (const ConSanPreappliedCodeRange &range : options.preapplied_code_ranges) {
    leaders.push_back(range.text_offset);
    if (range.continuation_text_offset)
      leaders.push_back(*range.continuation_text_offset);
  }
  std::ranges::sort(leaders);
  leaders.erase(std::ranges::unique(leaders).begin(), leaders.end());
  std::ranges::sort(kernel_entries);
  kernel_entries.erase(std::ranges::unique(kernel_entries).begin(), kernel_entries.end());

  std::vector<BasicBlock::CodeRange> code_ranges;
  for (const AmdGpuFunctionInfo &function : code_object.functions()) {
    if (function.code_size != 0)
      code_ranges.push_back({function.entry_text_offset, function.code_size});
  }
  for (const ConSanPreappliedCodeRange &range : options.preapplied_code_ranges) {
    if (range.size != 0)
      code_ranges.push_back({range.text_offset, range.size});
  }

  auto blocks = BasicBlock::build(code_object, decoder, arch, leaders, code_ranges);
  const BlockOffsetIndex block_index = build_block_offset_index(blocks);
  CodeObjectPatcher patcher(code_object);
  const std::span<const uint8_t> text = patcher.text_bytes();
  std::unordered_set<const BasicBlock *> owned_blocks;
  for (const ConSanKernelInfo &kernel : result.kernels) {
    if (!kernel.has_text_range)
      continue;
    std::vector<uint64_t> additional_entries;
    for (const ConSanPreappliedCodeRange &range : options.preapplied_code_ranges) {
      if (range.kernel_name != kernel.name)
        continue;
      if (block_for_offset(block_index, range.text_offset) != nullptr)
        additional_entries.push_back(range.text_offset);
      if (range.continuation_text_offset &&
          block_for_offset(block_index, *range.continuation_text_offset) != nullptr)
        additional_entries.push_back(*range.continuation_text_offset);
    }
    auto scope = build_kernel_cfg_scope(
        blocks, block_index,
        KernelScopeRequest{kernel.entry_text_offset, std::move(additional_entries)}, kernel_entries,
        text);
    // One kernel's valid scope cannot prove that another kernel's sites are
    // dead. If any owned entry remains unresolved, preserve the complete
    // conservative inventory instead of pruning against a partial owner set.
    if (!scope)
      return;
    owned_blocks.insert(scope->blocks.begin(), scope->blocks.end());
  }
  // A missing CFG proof is an analysis failure, not evidence that every
  // decoded site is dead. Preserve the previous conservative inventory.
  if (owned_blocks.empty())
    return;
  for (ConSanKernelInfo &kernel : result.kernels)
    prune_inferred_range_sites(kernel, block_index, owned_blocks);
  for (ConSanFunctionInfo &function : result.functions)
    prune_inferred_range_sites(function, block_index, owned_blocks);
}

[[nodiscard]] PointerComponentHint merge_call_entry_component(PointerComponentHint lhs,
                                                              PointerComponentHint rhs) {
  const bool lhs_known = is_known_component(lhs);
  const bool rhs_known = is_known_component(rhs);
  if (lhs_known && rhs_known && lhs.space == rhs.space && lhs.half == rhs.half)
    return PointerComponentHint{lhs.space, lhs.half, true};
  if (lhs_known && !rhs_known)
    return PointerComponentHint{lhs.space, lhs.half, true};
  if (!lhs_known && rhs_known)
    return PointerComponentHint{rhs.space, rhs.half, true};
  return {};
}

void relay_flat_pointer_provenance_across_calls(std::span<const uint8_t> code_object_bytes,
                                                const AmdGpuCodeObject &code_object,
                                                rj_code_arch_t arch,
                                                std::vector<ConSanKernelInfo> &kernels,
                                                std::vector<ConSanFunctionInfo> &functions,
                                                std::vector<std::string> &warnings) {
  std::unique_ptr<Decoder> relay_decoder = Decoder::create(arch);
  if (!relay_decoder)
    return;
  std::vector<uint64_t> leaders;
  std::vector<BasicBlock::CodeRange> code_ranges;
  for (const AmdGpuFunctionInfo &function : code_object.functions()) {
    leaders.push_back(function.entry_text_offset);
    if (function.code_size != 0u)
      code_ranges.push_back(
          {.start_offset = function.entry_text_offset, .size = function.code_size});
  }
  std::ranges::sort(leaders);
  leaders.erase(std::ranges::unique(leaders).begin(), leaders.end());
  std::vector<std::unique_ptr<BasicBlock>> blocks;
  try {
    blocks = BasicBlock::build(code_object, *relay_decoder, arch, leaders, code_ranges);
  } catch (...) {
    // Provenance relay is an optional refinement. The primary bounded decoder
    // remains authoritative for code objects whose CFG cannot be recovered.
    return;
  }

  std::unordered_map<uint64_t, uint64_t> call_targets;
  for (const auto &block : blocks) {
    if (block == nullptr)
      continue;
    for (const BasicBlock::CallEdge &call : block->call_edges()) {
      if (call.callee != nullptr)
        call_targets.emplace(call.source_call_offset, call.callee->start_offset());
    }
  }
  if (call_targets.empty())
    return;

  const std::vector<ConSanFunctionInfo> original_functions = functions;
  std::unordered_map<uint64_t, FlatPointerTracker> entry_trackers;
  constexpr uint32_t kMaxRelayPasses = 6u;
  for (uint32_t pass = 0; pass < kMaxRelayPasses; ++pass) {
    std::unordered_map<uint64_t, FlatPointerTracker> call_states;
    std::vector<std::string> pass_warnings;
    for (const ConSanKernelInfo &kernel : kernels) {
      ConSanKernelInfo decoded;
      decoded.name = kernel.name;
      decoded.entry_text_offset = kernel.entry_text_offset;
      decoded.text_file_offset = kernel.text_file_offset;
      decoded.code_size = kernel.code_size;
      decoded.has_text_range = kernel.has_text_range;
      try {
        decode_kernel_stats(code_object_bytes, *relay_decoder, arch, decoded, pass_warnings,
                            nullptr, &call_targets, &call_states);
      } catch (...) {
        functions = original_functions;
        return;
      }
    }
    for (ConSanFunctionInfo &function : functions) {
      ConSanFunctionInfo decoded;
      decoded.name = function.name;
      decoded.entry_text_offset = function.entry_text_offset;
      decoded.text_file_offset = function.text_file_offset;
      decoded.code_size = function.code_size;
      const auto entry = entry_trackers.find(function.entry_text_offset);
      try {
        decode_function_stats(code_object_bytes, *relay_decoder, arch, decoded, pass_warnings,
                              entry == entry_trackers.end() ? nullptr : &entry->second,
                              &call_targets, &call_states);
      } catch (...) {
        functions = original_functions;
        return;
      }
      function = std::move(decoded);
    }

    std::unordered_map<uint64_t, FlatPointerTracker> next_entry_trackers;
    std::unordered_set<uint64_t> initialized;
    for (const auto &[call_offset, callee_offset] : call_targets) {
      const auto state = call_states.find(call_offset);
      if (state == call_states.end())
        continue;
      FlatPointerTracker &entry = next_entry_trackers[callee_offset];
      if (initialized.insert(callee_offset).second) {
        entry.sgprs = state->second.sgprs;
        entry.vgprs = state->second.vgprs;
        continue;
      }
      for (size_t index = 0; index < entry.sgprs.size(); ++index)
        entry.sgprs[index] =
            merge_call_entry_component(entry.sgprs[index], state->second.sgprs[index]);
      for (size_t index = 0; index < entry.vgprs.size(); ++index)
        entry.vgprs[index] =
            merge_call_entry_component(entry.vgprs[index], state->second.vgprs[index]);
    }
    entry_trackers = std::move(next_entry_trackers);
    if (pass + 1u == kMaxRelayPasses) {
      warnings.insert(warnings.end(), std::make_move_iterator(pass_warnings.begin()),
                      std::make_move_iterator(pass_warnings.end()));
    }
  }
}

[[nodiscard]] std::string count_reason(std::string_view label, uint64_t count) {
  return std::string(label) + ": " + std::to_string(count);
}

[[nodiscard]] std::string preflight_summary(const ConSanKernelInfo &kernel) {
  std::string summary;
  for (const std::string &reason : kernel.preflight_reasons) {
    if (!summary.empty())
      summary += "; ";
    summary += reason;
  }
  return summary;
}

[[nodiscard]] bool starts_with(std::string_view value, std::string_view prefix) {
  return value.size() >= prefix.size() && value.substr(0, prefix.size()) == prefix;
}

void preflight_kernel(ConSanKernelInfo &kernel, const ConSanOptions &options,
                      std::vector<std::string> &errors, std::vector<std::string> &warnings) {
  std::vector<std::string> blockers;
  if (!kernel.has_text_range)
    blockers.emplace_back("function text range unavailable");
  if (!kernel.decoded)
    blockers.emplace_back("kernel was not decoded");
  if (kernel.stats.decode_error_count > 0)
    blockers.push_back(count_reason("decode errors", kernel.stats.decode_error_count));
  // SuperCollider does not reconstruct synchronization and leaves atomics
  // unchanged. Their presence therefore does not invalidate redundant checks
  // at independently supported non-atomic LDS sites.
  if (kernel.stats.ds_other_count > 0)
    blockers.push_back(
        count_reason("unsupported or unclassified ds_* instructions", kernel.stats.ds_other_count));
  const uint64_t supported_lds_sites = kernel.stats.lds_read_count + kernel.stats.lds_write_count;
  if (!blockers.empty()) {
    kernel.preflight_action =
        options.fail_closed ? ConSanPreflightAction::Reject : ConSanPreflightAction::Skip;
    kernel.preflight_reasons = std::move(blockers);
    const std::string message = "ConSan preflight " +
                                std::string(options.fail_closed ? "rejected" : "skipped") +
                                " kernel '" + kernel.name + "': " + preflight_summary(kernel);
    if (options.fail_closed)
      errors.push_back(message);
    else
      warnings.push_back(message);
    return;
  }

  if (supported_lds_sites == 0) {
    kernel.preflight_action = ConSanPreflightAction::Skip;
    kernel.preflight_reasons.emplace_back("no supported non-atomic LDS reads or writes");
    if (kernel.stats.lds_atomic_count > 0)
      kernel.preflight_reasons.push_back(
          count_reason("non-instrumented DS atomics observed", kernel.stats.lds_atomic_count));
    const uint64_t flat_memory_sites = kernel.stats.flat_read_count +
                                       kernel.stats.flat_write_count +
                                       kernel.stats.flat_atomic_count;
    if (flat_memory_sites > 0)
      kernel.preflight_reasons.push_back(
          count_reason("flat/generic memory instructions observed", flat_memory_sites));
    if (kernel.stats.global_memory_count > 0)
      kernel.preflight_reasons.push_back(
          count_reason("global memory instructions observed", kernel.stats.global_memory_count));
    if (kernel.stats.scratch_memory_count > 0)
      kernel.preflight_reasons.push_back(
          count_reason("scratch memory instructions observed", kernel.stats.scratch_memory_count));
    warnings.emplace_back("ConSan preflight skipped kernel '" + kernel.name +
                          "': " + preflight_summary(kernel));
    return;
  }

  kernel.preflight_action = ConSanPreflightAction::Candidate;
  kernel.preflight_reasons.push_back(
      count_reason("supported LDS reads", kernel.stats.lds_read_count));
  kernel.preflight_reasons.push_back(
      count_reason("supported LDS writes", kernel.stats.lds_write_count));
  // SuperCollider checks non-atomic LDS accesses for conflicting writes by
  // duplicating the access.  Replaying an atomic as an ordinary load/store
  // would change program semantics, but the atomic is itself indivisible and
  // is not a race candidate.  Keep it as a typed exclusion without rejecting
  // co-resident ordinary LDS accesses under fail-closed policy.
  if (kernel.stats.lds_atomic_count > 0)
    kernel.preflight_reasons.push_back(
        count_reason("atomic LDS accesses excluded", kernel.stats.lds_atomic_count));
  if (kernel.stats.barrier_count > 0)
    kernel.preflight_reasons.push_back(
        count_reason("barriers observed", kernel.stats.barrier_count));
}

[[nodiscard]] bool has_unsafe_proof_trampoline_flags(const Instruction &inst) {
  const uint64_t unsafe_flags = MEMORY_OP | WAITCNT | BARRIER | MFMA | ACCVGPR | PREDICATED_DEF;
  return (inst.flags() & unsafe_flags) != 0;
}

[[nodiscard]] bool is_preferred_proof_anchor(const Instruction &inst) {
  if (inst.size() != static_cast<int>(sizeof(uint32_t)))
    return false;
  if (has_unsafe_proof_trampoline_flags(inst))
    return false;

  const std::string_view mnemonic = inst.mnemonic();
  return starts_with(mnemonic, "v_add_f") || starts_with(mnemonic, "v_mul_f") ||
         starts_with(mnemonic, "v_fmac");
}

[[nodiscard]] bool is_s_clause(const Instruction &inst) { return inst.mnemonic() == "s_clause"; }

[[nodiscard]] uint32_t s_clause_following_instruction_count(const Instruction &inst) {
  if (!is_s_clause(inst) || inst.raw_encoding() == nullptr || inst.size() != sizeof(uint32_t))
    return 0;

  uint32_t word = 0;
  std::memcpy(&word, inst.raw_encoding(), sizeof(word));
  return (word & 0xffffu) + 1u;
}
