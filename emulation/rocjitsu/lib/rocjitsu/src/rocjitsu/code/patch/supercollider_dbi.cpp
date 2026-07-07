// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/supercollider_dbi.h"

#include "rocjitsu/analysis/def_use_chain.h"
#include "rocjitsu/analysis/liveness.h"
#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/basic_block.h"
#include "rocjitsu/code/patch/code_object_patcher.h"
#include "rocjitsu/code/patch/instruction_builder.h"
#include "rocjitsu/code/patch/instrumentor.h"
#include "rocjitsu/isa/arch/amdgpu/rdna4/machine_insts.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "util/bit.h"

#include "rocjitsu/base/rj_compiler.h"
RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include "hsa/AMDHSAKernelDescriptor.h"
RJ_DIAGNOSTIC_POP

#include <algorithm>
#include <array>
#include <cstring>
#include <initializer_list>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>

namespace rocjitsu {

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

[[nodiscard]] std::optional<uint16_t>
descriptor_vgpr_allocation_count(std::span<const uint8_t> image, uint64_t descriptor_file_offset,
                                 rj_code_arch_t arch) {
  if (descriptor_file_offset > image.size() || sizeof(KD) > image.size() - descriptor_file_offset)
    return std::nullopt;
  const auto *desc = reinterpret_cast<const KD *>(image.data() + descriptor_file_offset);
  const uint32_t wavefront_size = descriptor_wavefront_size(arch, *desc);
  const uint32_t granularity = descriptor_vgpr_granularity_for_wavefront(arch, wavefront_size);
  const uint32_t granulated = AMDHSA_BITS_GET(desc->compute_pgm_rsrc1,
                                              kd::COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT);
  const uint32_t allocated = (granulated + 1u) * std::max(granularity, 1u);
  return static_cast<uint16_t>(std::min<uint32_t>(allocated, REGISTER_SET_MAX_VGPRS));
}

[[nodiscard]] rj_code_arch_t arch_for_target(rj_code_target_id_t target) {
  switch (target) {
  case ROCJITSU_CODE_TARGET_GFX90A:
    return ROCJITSU_CODE_ARCH_CDNA2;
  case ROCJITSU_CODE_TARGET_GFX942:
    return ROCJITSU_CODE_ARCH_CDNA3;
  case ROCJITSU_CODE_TARGET_GFX950:
    return ROCJITSU_CODE_ARCH_CDNA4;
  case ROCJITSU_CODE_TARGET_GFX1200:
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

[[nodiscard]] SuperColliderDbiLdsAccessKind lds_access_kind(std::string_view mnemonic) {
  if (!mnemonic.starts_with("ds_"))
    return SuperColliderDbiLdsAccessKind::Other;
  if (is_ds_atomic(mnemonic))
    return SuperColliderDbiLdsAccessKind::Atomic;
  if (is_ds_read(mnemonic))
    return SuperColliderDbiLdsAccessKind::Read;
  if (is_ds_write(mnemonic))
    return SuperColliderDbiLdsAccessKind::Write;
  return SuperColliderDbiLdsAccessKind::Other;
}

[[nodiscard]] SuperColliderDbiLdsAccessKind flat_access_kind(std::string_view mnemonic) {
  if (mnemonic.starts_with("flat_atomic"))
    return SuperColliderDbiLdsAccessKind::Atomic;
  if (mnemonic.starts_with("flat_load"))
    return SuperColliderDbiLdsAccessKind::Read;
  if (mnemonic.starts_with("flat_store"))
    return SuperColliderDbiLdsAccessKind::Write;
  return SuperColliderDbiLdsAccessKind::Other;
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
  if (mnemonic.find("16") != std::string_view::npos ||
      mnemonic.find("b16") != std::string_view::npos)
    return 16;
  if (mnemonic.find("8") != std::string_view::npos || mnemonic.find("b8") != std::string_view::npos)
    return 8;
  return 0;
}

[[nodiscard]] bool is_supported_mvp_lds_site(SuperColliderDbiLdsAccessKind kind,
                                             uint32_t width_bits) {
  if (kind != SuperColliderDbiLdsAccessKind::Read && kind != SuperColliderDbiLdsAccessKind::Write)
    return false;
  if (kind == SuperColliderDbiLdsAccessKind::Read && width_bits == 16)
    return true;
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
  SuperColliderDbiFlatAddressSpaceHint space = SuperColliderDbiFlatAddressSpaceHint::Unknown;
  PointerHalfHint half = PointerHalfHint::Unknown;
  bool maybe = false;
};

struct FlatPointerPairHint {
  SuperColliderDbiFlatAddressSpaceHint space = SuperColliderDbiFlatAddressSpaceHint::Unknown;
  bool maybe = false;
};

struct FlatPointerTracker {
  std::array<PointerComponentHint, REGISTER_SET_MAX_SGPRS> sgprs{};
  std::array<PointerComponentHint, REGISTER_SET_MAX_VGPRS> vgprs{};
};

[[nodiscard]] bool is_known_flat_space(SuperColliderDbiFlatAddressSpaceHint space) {
  return space == SuperColliderDbiFlatAddressSpaceHint::Group ||
         space == SuperColliderDbiFlatAddressSpaceHint::Private ||
         space == SuperColliderDbiFlatAddressSpaceHint::Global;
}

[[nodiscard]] bool is_known_component(const PointerComponentHint &component) {
  return is_known_flat_space(component.space) && component.half != PointerHalfHint::Unknown;
}

[[nodiscard]] SuperColliderDbiFlatAddressSpaceHint
maybe_flat_space(SuperColliderDbiFlatAddressSpaceHint space) {
  switch (space) {
  case SuperColliderDbiFlatAddressSpaceHint::Group:
    return SuperColliderDbiFlatAddressSpaceHint::MaybeGroup;
  case SuperColliderDbiFlatAddressSpaceHint::Private:
    return SuperColliderDbiFlatAddressSpaceHint::MaybePrivate;
  case SuperColliderDbiFlatAddressSpaceHint::Global:
    return SuperColliderDbiFlatAddressSpaceHint::Global;
  case SuperColliderDbiFlatAddressSpaceHint::Unknown:
  case SuperColliderDbiFlatAddressSpaceHint::MaybeGroup:
  case SuperColliderDbiFlatAddressSpaceHint::MaybePrivate:
    return SuperColliderDbiFlatAddressSpaceHint::Unknown;
  }
  return SuperColliderDbiFlatAddressSpaceHint::Unknown;
}

[[nodiscard]] SuperColliderDbiFlatAddressSpaceHint
flat_pair_hint_to_public(FlatPointerPairHint hint) {
  if (!is_known_flat_space(hint.space))
    return SuperColliderDbiFlatAddressSpaceHint::Unknown;
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

[[nodiscard]] std::optional<SuperColliderDbiFlatAddressSpaceHint>
special_pointer_base_space(const Operand *operand) {
  if (operand == nullptr)
    return std::nullopt;
  const std::string name = operand->name();
  if (name == "src_shared_base")
    return SuperColliderDbiFlatAddressSpaceHint::Group;
  if (name == "src_private_base")
    return SuperColliderDbiFlatAddressSpaceHint::Private;
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
  if (ref.cls == RegClass::SGPR)
    clear_components(tracker.sgprs, ref);
  else if (ref.cls == RegClass::VGPR)
    clear_components(tracker.vgprs, ref);
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
                         SuperColliderDbiFlatAddressSpaceHint space, bool maybe) {
  if (index >= N || index + 1 >= N)
    return;
  components[index] = PointerComponentHint{space, PointerHalfHint::Low, maybe};
  components[index + 1] = PointerComponentHint{space, PointerHalfHint::High, maybe};
}

void set_sgpr_pair_components(FlatPointerTracker &tracker, uint16_t index,
                              SuperColliderDbiFlatAddressSpaceHint space, bool maybe) {
  set_pair_components(tracker.sgprs, index, space, maybe);
}

void set_vgpr_component(FlatPointerTracker &tracker, uint16_t index,
                        PointerComponentHint component) {
  if (index >= tracker.vgprs.size())
    return;
  tracker.vgprs[index] = component;
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
      index + 1 < N ? component_hint_at(components, static_cast<uint16_t>(index + 1))
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
  std::optional<PointerComponentHint> merged;
  for (int i = 0; i < inst.num_src_operands(); ++i) {
    const auto hint = source_component_hint(inst.src_operand(i), tracker);
    if (!hint)
      continue;
    merged = merge_arithmetic_components(merged, hint);
    if (!merged)
      return std::nullopt;
  }
  return merged;
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

void update_flat_pointer_tracker(const Instruction &inst, FlatPointerTracker &tracker) {
  const std::string_view mnemonic = inst.mnemonic();

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
    clear_destination_components(inst, tracker);
    if (dst && src)
      set_vgpr_component(tracker, *dst, *src);
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

[[nodiscard]] SuperColliderDbiFlatAddressSpaceHint
flat_address_space_hint(const Instruction &inst, const FlatPointerTracker &tracker) {
  const std::optional<uint16_t> addr = vgpr_index(inst.src_operand(0));
  if (!addr)
    return SuperColliderDbiFlatAddressSpaceHint::Unknown;
  const auto hint = pair_hint_at(tracker.vgprs, *addr);
  if (!hint)
    return SuperColliderDbiFlatAddressSpaceHint::Unknown;
  return flat_pair_hint_to_public(*hint);
}

void count_flat_address_space_hint(SuperColliderDbiKernelStats &stats,
                                   SuperColliderDbiFlatAddressSpaceHint hint) {
  switch (hint) {
  case SuperColliderDbiFlatAddressSpaceHint::Group:
    ++stats.flat_group_hint_count;
    break;
  case SuperColliderDbiFlatAddressSpaceHint::Private:
    ++stats.flat_private_hint_count;
    break;
  case SuperColliderDbiFlatAddressSpaceHint::MaybeGroup:
    ++stats.flat_maybe_group_hint_count;
    break;
  case SuperColliderDbiFlatAddressSpaceHint::MaybePrivate:
    ++stats.flat_maybe_private_hint_count;
    break;
  case SuperColliderDbiFlatAddressSpaceHint::Global:
    ++stats.flat_global_hint_count;
    break;
  case SuperColliderDbiFlatAddressSpaceHint::Unknown:
    ++stats.flat_unknown_hint_count;
    break;
  }
}

[[nodiscard]] bool is_fence_like(std::string_view mnemonic) {
  return starts_with_any(mnemonic, {"s_dcache", "s_icache", "global_wb", "global_inv",
                                    "global_wbinv", "buffer_wb", "buffer_inv", "buffer_wbinv"});
}

[[nodiscard]] bool is_barrier_instruction(const Instruction &inst) {
  const std::string_view mnemonic = inst.mnemonic();
  return inst.is_barrier() || mnemonic.starts_with("s_barrier") ||
         mnemonic.find("barrier") != std::string_view::npos;
}

void classify_instruction(const Instruction &inst, SuperColliderDbiKernelStats &stats) {
  const std::string_view mnemonic = inst.mnemonic();

  if (mnemonic.starts_with("ds_")) {
    switch (lds_access_kind(mnemonic)) {
    case SuperColliderDbiLdsAccessKind::Atomic:
      ++stats.lds_atomic_count;
      break;
    case SuperColliderDbiLdsAccessKind::Read:
      ++stats.lds_read_count;
      break;
    case SuperColliderDbiLdsAccessKind::Write:
      ++stats.lds_write_count;
      break;
    case SuperColliderDbiLdsAccessKind::Other:
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

void record_lds_site(const Instruction &inst, SuperColliderDbiKernelInfo &kernel,
                     uint64_t instruction_text_offset) {
  const std::string_view mnemonic = inst.mnemonic();
  if (!mnemonic.starts_with("ds_"))
    return;

  SuperColliderDbiLdsSite site;
  site.kind = lds_access_kind(mnemonic);
  site.text_offset = instruction_text_offset;
  site.file_offset = kernel.text_file_offset + instruction_text_offset;
  site.size = static_cast<uint32_t>(inst.size());
  site.width_bits = lds_width_bits(mnemonic);
  site.supported_mvp = is_supported_mvp_lds_site(site.kind, site.width_bits);
  if (site.kind == SuperColliderDbiLdsAccessKind::Read) {
    site.dst_vgpr = vgpr_index(inst.dst_operand(0));
    site.addr_vgpr = vgpr_index(inst.src_operand(0));
  } else if (site.kind == SuperColliderDbiLdsAccessKind::Write) {
    site.addr_vgpr = vgpr_index(inst.src_operand(0));
    site.data_vgpr = vgpr_index(inst.src_operand(1));
  }
  site.mnemonic = std::string(mnemonic);
  kernel.lds_sites.push_back(std::move(site));
}

void record_flat_site(const Instruction &inst, const FlatPointerTracker &tracker,
                      SuperColliderDbiKernelInfo &kernel, uint64_t instruction_text_offset,
                      rj_code_arch_t arch, std::span<const uint8_t> instruction_bytes) {
  const std::string_view mnemonic = inst.mnemonic();
  if (!mnemonic.starts_with("flat_"))
    return;

  SuperColliderDbiFlatSite site;
  site.kind = flat_access_kind(mnemonic);
  site.text_offset = instruction_text_offset;
  site.file_offset = kernel.text_file_offset + instruction_text_offset;
  site.size = static_cast<uint32_t>(inst.size());
  site.width_bits = lds_width_bits(mnemonic);
  site.address_space_hint = flat_address_space_hint(inst, tracker);
  count_flat_address_space_hint(kernel.stats, site.address_space_hint);
  if (site.kind == SuperColliderDbiLdsAccessKind::Read) {
    site.dst_vgpr = vgpr_index(inst.dst_operand(0));
    site.addr_vgpr = vgpr_index(inst.src_operand(0));
  } else if (site.kind == SuperColliderDbiLdsAccessKind::Write) {
    site.addr_vgpr = vgpr_index(inst.src_operand(0));
    site.data_vgpr = vgpr_index(inst.src_operand(1));
  } else if (site.kind == SuperColliderDbiLdsAccessKind::Atomic) {
    site.addr_vgpr = vgpr_index(inst.src_operand(0));
    site.data_vgpr = vgpr_index(inst.src_operand(1));
  }
  if (arch == ROCJITSU_CODE_ARCH_RDNA4 &&
      instruction_bytes.size() >= sizeof(rdna4::VflatMachineInst)) {
    rdna4::VflatMachineInst raw{};
    std::memcpy(&raw, instruction_bytes.data(), sizeof(raw));
    site.raw_saddr = static_cast<uint32_t>(raw.saddr);
    site.raw_vaddr = static_cast<uint32_t>(raw.vaddr);
    site.raw_vsrc = static_cast<uint32_t>(raw.vsrc);
    site.raw_vdst = static_cast<uint32_t>(raw.vdst);
    site.raw_ioffset = static_cast<int32_t>(raw.ioffset << 8) >> 8;
    site.raw_scope = static_cast<uint32_t>(raw.scope);
    site.raw_th = static_cast<uint32_t>(raw.th);
  }
  site.mnemonic = std::string(mnemonic);
  kernel.flat_sites.push_back(std::move(site));
}

void decode_kernel_stats(std::span<const uint8_t> code_object_bytes, Decoder &decoder,
                         rj_code_arch_t arch, SuperColliderDbiKernelInfo &kernel,
                         std::vector<std::string> &warnings) {
  if (!kernel.has_text_range) {
    warnings.emplace_back("DBI SuperCollider cannot decode kernel '" + kernel.name +
                          "': function text range is unavailable");
    return;
  }
  if (kernel.code_size == 0) {
    warnings.emplace_back("DBI SuperCollider cannot decode kernel '" + kernel.name +
                          "': function text range is empty");
    return;
  }
  if (kernel.entry_text_offset > code_object_bytes.size() ||
      kernel.text_file_offset > code_object_bytes.size() - kernel.entry_text_offset) {
    ++kernel.stats.decode_error_count;
    warnings.emplace_back("DBI SuperCollider cannot decode kernel '" + kernel.name +
                          "': function text range starts outside the code object");
    return;
  }

  const uint64_t start = kernel.text_file_offset + kernel.entry_text_offset;
  if (start > code_object_bytes.size() || kernel.code_size > code_object_bytes.size() - start) {
    ++kernel.stats.decode_error_count;
    warnings.emplace_back("DBI SuperCollider cannot decode kernel '" + kernel.name +
                          "': function text range extends outside the code object");
    return;
  }

  const auto text =
      code_object_bytes.subspan(static_cast<size_t>(start), static_cast<size_t>(kernel.code_size));
  FlatPointerTracker flat_pointer_tracker;
  size_t offset = 0;
  while (offset < text.size()) {
    const size_t remaining = text.size() - offset;
    if (remaining < sizeof(uint32_t)) {
      ++kernel.stats.decode_error_count;
      warnings.emplace_back("DBI SuperCollider stopped decoding kernel '" + kernel.name +
                            "': trailing bytes are smaller than one instruction word");
      break;
    }

    std::array<uint32_t, 4> words{};
    std::memcpy(words.data(), text.data() + offset,
                std::min(words.size() * sizeof(uint32_t), remaining));

    std::unique_ptr<Instruction> inst(
        decoder.decode(words.data(), kernel.entry_text_offset + static_cast<uint64_t>(offset)));
    if (!inst) {
      ++kernel.stats.decode_error_count;
      warnings.emplace_back("DBI SuperCollider stopped decoding kernel '" + kernel.name +
                            "': decoder returned null");
      break;
    }

    const int inst_size = inst->size();
    if (inst_size <= 0 || inst_size % static_cast<int>(sizeof(uint32_t)) != 0) {
      ++kernel.stats.decode_error_count;
      warnings.emplace_back("DBI SuperCollider stopped decoding kernel '" + kernel.name +
                            "': decoder returned an invalid instruction size");
      break;
    }
    if (static_cast<size_t>(inst_size) > remaining) {
      ++kernel.stats.decode_error_count;
      warnings.emplace_back("DBI SuperCollider stopped decoding kernel '" + kernel.name +
                            "': decoded instruction exceeds the function text range");
      break;
    }

    ++kernel.stats.instruction_count;
    classify_instruction(*inst, kernel.stats);
    record_lds_site(*inst, kernel, kernel.entry_text_offset + static_cast<uint64_t>(offset));
    record_flat_site(*inst, flat_pointer_tracker, kernel,
                     kernel.entry_text_offset + static_cast<uint64_t>(offset), arch,
                     text.subspan(offset, static_cast<size_t>(inst_size)));
    update_flat_pointer_tracker(*inst, flat_pointer_tracker);
    offset += static_cast<size_t>(inst_size);
  }

  kernel.decoded = true;
}

void decode_function_stats(std::span<const uint8_t> code_object_bytes, Decoder &decoder,
                           rj_code_arch_t arch, SuperColliderDbiFunctionInfo &function,
                           std::vector<std::string> &warnings) {
  SuperColliderDbiKernelInfo range;
  range.name = function.name;
  range.entry_text_offset = function.entry_text_offset;
  range.text_file_offset = function.text_file_offset;
  range.code_size = function.code_size;
  range.has_text_range = true;

  decode_kernel_stats(code_object_bytes, decoder, arch, range, warnings);

  function.decoded = range.decoded;
  function.stats = range.stats;
  function.lds_sites = std::move(range.lds_sites);
  function.flat_sites = std::move(range.flat_sites);
}

[[nodiscard]] std::string count_reason(std::string_view label, uint64_t count) {
  return std::string(label) + ": " + std::to_string(count);
}

[[nodiscard]] std::string preflight_summary(const SuperColliderDbiKernelInfo &kernel) {
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

void preflight_kernel(SuperColliderDbiKernelInfo &kernel, const SuperColliderDbiOptions &options,
                      std::vector<std::string> &errors, std::vector<std::string> &warnings) {
  std::vector<std::string> blockers;
  if (!kernel.has_text_range)
    blockers.emplace_back("function text range unavailable");
  if (!kernel.decoded)
    blockers.emplace_back("kernel was not decoded");
  if (kernel.stats.decode_error_count > 0)
    blockers.push_back(count_reason("decode errors", kernel.stats.decode_error_count));
  if (kernel.stats.lds_atomic_count > 0)
    blockers.push_back(count_reason("unsupported DS atomics", kernel.stats.lds_atomic_count));
  if (kernel.stats.ds_other_count > 0)
    blockers.push_back(
        count_reason("unsupported or unclassified ds_* instructions", kernel.stats.ds_other_count));
  if (kernel.stats.fence_like_count > 0)
    blockers.push_back(
        count_reason("unsupported fence/cache-like instructions", kernel.stats.fence_like_count));

  const uint64_t supported_lds_sites = kernel.stats.lds_read_count + kernel.stats.lds_write_count;
  if (!blockers.empty()) {
    kernel.preflight_action = options.fail_closed ? SuperColliderDbiPreflightAction::Reject
                                                  : SuperColliderDbiPreflightAction::Skip;
    kernel.preflight_reasons = std::move(blockers);
    const std::string message = "DBI SuperCollider preflight " +
                                std::string(options.fail_closed ? "rejected" : "skipped") +
                                " kernel '" + kernel.name + "': " + preflight_summary(kernel);
    if (options.fail_closed)
      errors.push_back(message);
    else
      warnings.push_back(message);
    return;
  }

  if (supported_lds_sites == 0) {
    kernel.preflight_action = SuperColliderDbiPreflightAction::Skip;
    kernel.preflight_reasons.emplace_back("no supported non-atomic LDS reads or writes");
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
    warnings.emplace_back("DBI SuperCollider preflight skipped kernel '" + kernel.name +
                          "': " + preflight_summary(kernel));
    return;
  }

  kernel.preflight_action = SuperColliderDbiPreflightAction::Candidate;
  kernel.preflight_reasons.push_back(
      count_reason("supported LDS reads", kernel.stats.lds_read_count));
  kernel.preflight_reasons.push_back(
      count_reason("supported LDS writes", kernel.stats.lds_write_count));
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

struct InPlaceNopSite {
  uint64_t text_offset = 0;
  uint64_t file_offset = 0;
};

struct InPlaceInstructionSite {
  uint64_t text_offset = 0;
  uint64_t file_offset = 0;
  uint32_t size = 0;
};

struct BarrierSite {
  uint64_t text_offset = 0;
  uint64_t file_offset = 0;
  uint32_t size = 0;
  std::string range_name;
  std::string mnemonic;
};

struct TextRange {
  uint64_t begin = 0;
  uint64_t end = 0;
};

struct LocalNopCave {
  uint64_t text_offset = 0;
  uint64_t file_offset = 0;
  uint32_t word_count = 0;
};

void add_covered_text_range(std::vector<TextRange> &ranges, uint64_t begin, uint64_t size,
                            uint64_t text_size) {
  if (size == 0 || begin >= text_size)
    return;
  const uint64_t end = std::min<uint64_t>(text_size, begin + std::min(size, text_size - begin));
  if (begin < end)
    ranges.push_back({begin, end});
}

[[nodiscard]] std::vector<TextRange> covered_text_ranges(const SuperColliderDbiResult &result,
                                                         uint64_t text_size) {
  std::vector<TextRange> ranges;
  ranges.reserve(result.kernels.size() + result.functions.size());
  for (const SuperColliderDbiKernelInfo &kernel : result.kernels) {
    if (kernel.has_text_range)
      add_covered_text_range(ranges, kernel.entry_text_offset, kernel.code_size, text_size);
  }
  for (const SuperColliderDbiFunctionInfo &function : result.functions)
    add_covered_text_range(ranges, function.entry_text_offset, function.code_size, text_size);

  std::ranges::sort(ranges, [](const TextRange &lhs, const TextRange &rhs) {
    if (lhs.begin != rhs.begin)
      return lhs.begin < rhs.begin;
    return lhs.end < rhs.end;
  });

  std::vector<TextRange> merged;
  for (const TextRange &range : ranges) {
    if (merged.empty() || range.begin > merged.back().end) {
      merged.push_back(range);
    } else {
      merged.back().end = std::max(merged.back().end, range.end);
    }
  }
  return merged;
}

[[nodiscard]] bool text_offset_is_covered(uint64_t offset, std::span<const TextRange> ranges) {
  for (const TextRange &range : ranges) {
    if (offset < range.begin)
      return false;
    if (offset < range.end)
      return true;
  }
  return false;
}

[[nodiscard]] std::vector<LocalNopCave>
find_uncovered_nop_caves(const AmdGpuCodeObject &code_object, const SuperColliderDbiResult &result,
                         rj_code_arch_t arch) {
  if (code_object.text_sections().size() != 1)
    return {};

  const Section *text = code_object.text_sections().front();
  const auto text_bytes =
      std::span<const uint8_t>(reinterpret_cast<const uint8_t *>(text->data()), text->size());
  const std::vector<TextRange> ranges = covered_text_ranges(result, text->size());
  const uint32_t nop = build_s_nop(0, arch);

  std::vector<LocalNopCave> caves;
  for (uint64_t text_offset = 0; text_offset + sizeof(uint32_t) <= text_bytes.size();) {
    if (text_offset_is_covered(text_offset, ranges)) {
      text_offset += sizeof(uint32_t);
      continue;
    }

    uint32_t word = 0;
    std::memcpy(&word, text_bytes.data() + text_offset, sizeof(word));
    if (word != nop) {
      text_offset += sizeof(uint32_t);
      continue;
    }

    const uint64_t run_begin = text_offset;
    uint32_t run_words = 0;
    while (text_offset + sizeof(uint32_t) <= text_bytes.size() &&
           !text_offset_is_covered(text_offset, ranges)) {
      std::memcpy(&word, text_bytes.data() + text_offset, sizeof(word));
      if (word != nop)
        break;
      ++run_words;
      text_offset += sizeof(uint32_t);
    }
    caves.push_back({run_begin, text->sectionOffset() + run_begin, run_words});
  }

  return caves;
}

[[nodiscard]] std::optional<InPlaceNopSite> find_existing_nop_site(const AmdGpuCodeObject &obj,
                                                                   rj_code_arch_t arch) {
  if (obj.text_sections().size() != 1)
    return std::nullopt;

  std::unique_ptr<Decoder> decoder = Decoder::create(arch);
  if (!decoder)
    return std::nullopt;

  const Section *text = obj.text_sections().front();
  auto blocks = BasicBlock::build(obj, *decoder, arch);
  for (const auto &block : blocks) {
    uint64_t offset = block->start_offset();
    for (const Instruction &inst : block->instructions()) {
      if (inst.size() == static_cast<int>(sizeof(uint32_t)) && inst.mnemonic() == "s_nop") {
        InPlaceNopSite site;
        site.text_offset = offset;
        site.file_offset = text->sectionOffset() + offset;
        return site;
      }
      offset += static_cast<uint64_t>(inst.size());
    }
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<InPlaceInstructionSite>
find_preferred_in_place_instruction_site(const AmdGpuCodeObject &obj, rj_code_arch_t arch) {
  if (obj.text_sections().size() != 1)
    return std::nullopt;

  std::unique_ptr<Decoder> decoder = Decoder::create(arch);
  if (!decoder)
    return std::nullopt;

  const Section *text = obj.text_sections().front();
  auto blocks = BasicBlock::build(obj, *decoder, arch);
  for (const auto &block : blocks) {
    uint64_t offset = block->start_offset();
    for (const Instruction &inst : block->instructions()) {
      if (inst.size() == static_cast<int>(sizeof(uint32_t)) && is_preferred_proof_anchor(inst)) {
        InPlaceInstructionSite site;
        site.text_offset = offset;
        site.file_offset = text->sectionOffset() + offset;
        site.size = sizeof(uint32_t);
        return site;
      }
      offset += static_cast<uint64_t>(inst.size());
    }
  }
  return std::nullopt;
}

[[nodiscard]] bool rewrite_word_in_place(const AmdGpuCodeObject &code_object, uint64_t file_offset,
                                         uint32_t replacement, SuperColliderDbiResult &result) {
  result.elf_bytes.assign(reinterpret_cast<const uint8_t *>(code_object.image_data()),
                          reinterpret_cast<const uint8_t *>(code_object.image_data()) +
                              code_object.image_size());
  if (file_offset > result.elf_bytes.size() ||
      sizeof(replacement) > result.elf_bytes.size() - file_offset) {
    result.errors.emplace_back("DBI SuperCollider proof rewrite offset is outside ELF bytes");
    result.elf_bytes.clear();
    return false;
  }
  std::memcpy(result.elf_bytes.data() + file_offset, &replacement, sizeof(replacement));
  result.modified = true;
  return true;
}

[[nodiscard]] std::optional<uint64_t> find_first_relocatable_anchor(const AmdGpuCodeObject &obj,
                                                                    rj_code_arch_t arch,
                                                                    std::string *error_out) {
  if (obj.text_sections().empty()) {
    *error_out = "code object has no .text section";
    return std::nullopt;
  }
  if (obj.text_sections().size() > 1) {
    *error_out = "code object has multiple .text sections";
    return std::nullopt;
  }

  std::unique_ptr<Decoder> decoder = Decoder::create(arch);
  if (!decoder) {
    *error_out = "no decoder available for proof NOP patch";
    return std::nullopt;
  }

  const Section *text = obj.text_sections().front();
  const std::span<const uint8_t> text_bytes(reinterpret_cast<const uint8_t *>(text->data()),
                                            text->size());
  auto blocks = BasicBlock::build(obj, *decoder, arch);
  std::optional<uint64_t> fallback_anchor;
  for (const auto &block : blocks) {
    uint64_t offset = block->start_offset();
    uint32_t clause_remaining = 0;
    for (const Instruction &inst : block->instructions()) {
      const bool blocked_by_clause = clause_remaining > 0;
      if (clause_remaining > 0)
        --clause_remaining;
      if (is_s_clause(inst))
        clause_remaining = s_clause_following_instruction_count(inst);
      if (!blocked_by_clause && inst.size() == static_cast<int>(sizeof(uint32_t)) &&
          !has_unsafe_proof_trampoline_flags(inst) &&
          is_relocatable_anchor(inst, offset, text_bytes, arch)) {
        if (!fallback_anchor)
          fallback_anchor = offset;
        if (is_preferred_proof_anchor(inst))
          return offset;
      }
      offset += static_cast<uint64_t>(inst.size());
    }
  }

  if (fallback_anchor)
    return fallback_anchor;

  *error_out = "no relocatable instruction anchor found for proof NOP patch";
  return std::nullopt;
}

[[nodiscard]] uint32_t build_sopp_word(uint32_t op, uint16_t simm16) {
  return pack_sopp(op, simm16);
}

[[nodiscard]] std::optional<uint32_t> build_s_trap_word(uint16_t simm16, rj_code_arch_t arch) {
  if (arch != ROCJITSU_CODE_ARCH_RDNA4)
    return std::nullopt;
  constexpr uint32_t kRdna4STrapOp = 16;
  return build_sopp_word(kRdna4STrapOp, simm16);
}

[[nodiscard]] std::optional<uint32_t> build_s_cbranch_vccz_word(int16_t offset_dwords,
                                                                rj_code_arch_t arch) {
  if (arch != ROCJITSU_CODE_ARCH_RDNA4)
    return std::nullopt;
  constexpr uint32_t kRdna4SCbranchVcczOp = 35;
  return build_sopp_word(kRdna4SCbranchVcczOp, static_cast<uint16_t>(offset_dwords));
}

[[nodiscard]] std::optional<uint32_t> build_s_wait_dscnt_word(uint16_t count, rj_code_arch_t arch) {
  if (arch != ROCJITSU_CODE_ARCH_RDNA4)
    return std::nullopt;
  constexpr uint32_t kRdna4SWaitDscntOp = 70;
  return build_sopp_word(kRdna4SWaitDscntOp, count);
}

[[nodiscard]] const char *delay_mode_name(SuperColliderDbiDelayMode mode) {
  switch (mode) {
  case SuperColliderDbiDelayMode::Nop:
    return "nop";
  case SuperColliderDbiDelayMode::Sleep:
    return "sleep";
  case SuperColliderDbiDelayMode::SleepVar:
    return "sleep_var";
  }
  return "unknown";
}

[[nodiscard]] std::optional<uint32_t>
delay_instruction_word_count(const SuperColliderDbiOptions &options,
                             std::vector<std::string> &errors, std::string_view context) {
  if (options.delay_nops == 0)
    return 0u;

  switch (options.delay_mode) {
  case SuperColliderDbiDelayMode::Nop:
    return options.delay_nops;
  case SuperColliderDbiDelayMode::Sleep:
    if (options.delay_nops > UINT16_MAX) {
      errors.emplace_back(std::string(context) +
                          " sleep delay immediate exceeds the 16-bit s_sleep field");
      return std::nullopt;
    }
    return 1u;
  case SuperColliderDbiDelayMode::SleepVar:
    if (options.delay_var_ssrc > 255) {
      errors.emplace_back(std::string(context) +
                          " sleep_var source exceeds the 8-bit scalar source field");
      return std::nullopt;
    }
    return 1u;
  }

  errors.emplace_back(std::string(context) + " has unknown delay mode '" +
                      delay_mode_name(options.delay_mode) + "'");
  return std::nullopt;
}

[[nodiscard]] bool append_delay_words(std::vector<uint32_t> &words, rj_code_arch_t arch,
                                      const SuperColliderDbiOptions &options,
                                      std::vector<std::string> &errors, std::string_view context) {
  if (options.delay_nops == 0)
    return true;

  switch (options.delay_mode) {
  case SuperColliderDbiDelayMode::Nop:
    for (uint32_t i = 0; i < options.delay_nops; ++i)
      words.push_back(build_s_nop(0, arch));
    return true;
  case SuperColliderDbiDelayMode::Sleep:
    if (options.delay_nops > UINT16_MAX) {
      errors.emplace_back(std::string(context) +
                          " sleep delay immediate exceeds the 16-bit s_sleep field");
      return false;
    }
    words.push_back(build_s_sleep(static_cast<uint16_t>(options.delay_nops), arch));
    return true;
  case SuperColliderDbiDelayMode::SleepVar:
    if (options.delay_var_ssrc > 255) {
      errors.emplace_back(std::string(context) +
                          " sleep_var source exceeds the 8-bit scalar source field");
      return false;
    }
    words.push_back(build_s_sleep_var(options.delay_var_ssrc, arch));
    return true;
  }

  errors.emplace_back(std::string(context) + " has unknown delay mode '" +
                      delay_mode_name(options.delay_mode) + "'");
  return false;
}

[[nodiscard]] std::optional<uint32_t> build_v_cmp_ne_u32_e32_word(uint16_t src0, uint16_t vsrc1,
                                                                  rj_code_arch_t arch) {
  if (arch != ROCJITSU_CODE_ARCH_RDNA4 || src0 > 255 || vsrc1 > 255)
    return std::nullopt;
  constexpr uint32_t kRdna4VCmpNeU32E32Base = 0x7C9A0000u;
  const uint32_t encoded_vsrc1 = (static_cast<uint32_t>(vsrc1) << 1u) | 1u;
  return kRdna4VCmpNeU32E32Base | (encoded_vsrc1 << 8u) | src0;
}

[[nodiscard]] std::optional<uint32_t>
build_ds_load_word0_from_vds_word0(uint32_t word0, uint32_t width_bits, rj_code_arch_t arch) {
  if (arch != ROCJITSU_CODE_ARCH_RDNA4)
    return std::nullopt;
  uint32_t base = 0;
  switch (width_bits) {
  case 32:
    base = 0xD8D80000u;
    break;
  case 64:
    base = 0xD9D80000u;
    break;
  case 128:
    base = 0xDBFC0000u;
    break;
  default:
    return std::nullopt;
  }
  constexpr uint32_t kRdna4VdsOffsetMask = 0x0000FFFFu;
  return base | (word0 & kRdna4VdsOffsetMask);
}

[[nodiscard]] uint32_t build_ds_load_word1(uint16_t addr_vgpr, uint16_t dst_vgpr) {
  return static_cast<uint32_t>(addr_vgpr) | (static_cast<uint32_t>(dst_vgpr) << 24u);
}

[[nodiscard]] std::optional<uint16_t> lds_dword_count(const SuperColliderDbiLdsSite &site) {
  const bool is_two_addr_load = site.mnemonic == "ds_load_2addr_b32" ||
                                site.mnemonic == "ds_load_2addr_b64" ||
                                site.mnemonic == "ds_load_2addr_stride64_b32" ||
                                site.mnemonic == "ds_load_2addr_stride64_b64";
  if (site.mnemonic == "ds_load_u16_d16" || site.mnemonic == "ds_load_u16_d16_hi")
    return 1;
  if (site.width_bits == 32 || site.width_bits == 64 || site.width_bits == 128)
    return static_cast<uint16_t>((site.width_bits / 32u) * (is_two_addr_load ? 2u : 1u));
  return std::nullopt;
}

[[nodiscard]] bool vgpr_ranges_overlap(uint16_t lhs_base, uint16_t lhs_count, uint16_t rhs_base,
                                       uint16_t rhs_count) {
  const uint32_t lhs_end = static_cast<uint32_t>(lhs_base) + lhs_count;
  const uint32_t rhs_end = static_cast<uint32_t>(rhs_base) + rhs_count;
  return static_cast<uint32_t>(lhs_base) < rhs_end && static_cast<uint32_t>(rhs_base) < lhs_end;
}

[[nodiscard]] bool is_forbidden_scratch_vgpr_run(const SuperColliderDbiLdsSite &site,
                                                 uint16_t candidate, uint16_t required_vgprs) {
  if (site.addr_vgpr && vgpr_ranges_overlap(candidate, required_vgprs, *site.addr_vgpr, 1))
    return true;

  const uint16_t data_vgprs = lds_dword_count(site).value_or(1);
  if (site.dst_vgpr && vgpr_ranges_overlap(candidate, required_vgprs, *site.dst_vgpr, data_vgprs))
    return true;
  if (site.data_vgpr && vgpr_ranges_overlap(candidate, required_vgprs, *site.data_vgpr, data_vgprs))
    return true;
  return false;
}

[[nodiscard]] uint16_t legacy_scratch_search_start(const SuperColliderDbiLdsSite &site) {
  std::optional<uint32_t> first_after_operands;
  auto note_range = [&first_after_operands](std::optional<uint16_t> base, uint16_t count) {
    if (!base)
      return;
    const uint32_t end = static_cast<uint32_t>(*base) + count;
    if (!first_after_operands || end > *first_after_operands)
      first_after_operands = end;
  };

  const uint16_t data_vgprs = lds_dword_count(site).value_or(1);
  note_range(site.addr_vgpr, 1);
  note_range(site.dst_vgpr, data_vgprs);
  note_range(site.data_vgpr, data_vgprs);
  if (!first_after_operands)
    return 0;
  return *first_after_operands <= 255 ? static_cast<uint16_t>(*first_after_operands) : 256;
}

[[nodiscard]] uint16_t scratch_search_start(const SuperColliderDbiLdsSite &site,
                                            std::optional<uint16_t> min_auto_scratch_vgpr) {
  return std::max(legacy_scratch_search_start(site), min_auto_scratch_vgpr.value_or(0));
}

[[nodiscard]] bool needs_scratch_headroom_at_descriptor_edge(const SuperColliderDbiLdsSite &site) {
  return site.mnemonic == "ds_load_b64" || site.mnemonic == "ds_load_2addr_b64" ||
         site.mnemonic == "ds_load_2addr_stride64_b64";
}

[[nodiscard]] std::optional<uint16_t>
find_liveness_scratch_vgpr(const SuperColliderDbiLdsSite &site, const Instruction *inst,
                           const LivenessAnalysis *liveness,
                           std::optional<uint16_t> min_auto_scratch_vgpr,
                           std::optional<uint16_t> max_auto_scratch_vgpr, uint16_t required_vgprs) {
  if (inst == nullptr || liveness == nullptr)
    return std::nullopt;

  uint16_t search_start = scratch_search_start(site, min_auto_scratch_vgpr);
  if (max_auto_scratch_vgpr && search_start >= *max_auto_scratch_vgpr)
    return std::nullopt;
  while (search_start < REGISTER_SET_MAX_VGPRS) {
    auto candidate = liveness->find_free_run(inst, required_vgprs, search_start);
    if (!candidate)
      return std::nullopt;
    const uint32_t candidate_end = static_cast<uint32_t>(*candidate) + required_vgprs;
    if (max_auto_scratch_vgpr && candidate_end > *max_auto_scratch_vgpr)
      return std::nullopt;
    if (max_auto_scratch_vgpr && candidate_end == *max_auto_scratch_vgpr &&
        needs_scratch_headroom_at_descriptor_edge(site))
      return std::nullopt;
    if (candidate_end <= 256 && !is_forbidden_scratch_vgpr_run(site, *candidate, required_vgprs))
      return candidate;
    if (*candidate == UINT16_MAX)
      return std::nullopt;
    search_start = static_cast<uint16_t>(*candidate + 1u);
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<uint16_t>
choose_scratch_vgpr(const SuperColliderDbiLdsSite &site, const SuperColliderDbiOptions &options,
                    const Instruction *inst, const LivenessAnalysis *liveness,
                    std::optional<uint16_t> min_auto_scratch_vgpr,
                    std::optional<uint16_t> max_auto_scratch_vgpr, uint16_t required_vgprs) {
  if (options.scratch_vgpr) {
    if (static_cast<uint32_t>(*options.scratch_vgpr) + required_vgprs <= 256 &&
        !is_forbidden_scratch_vgpr_run(site, *options.scratch_vgpr, required_vgprs))
      return *options.scratch_vgpr;
    return std::nullopt;
  }

  if (auto scratch = find_liveness_scratch_vgpr(site, inst, liveness, min_auto_scratch_vgpr,
                                                max_auto_scratch_vgpr, required_vgprs))
    return scratch;

  return std::nullopt;
}

[[nodiscard]] std::optional<uint16_t>
choose_vcc_save_sgpr(const Instruction *inst, const LivenessAnalysis *liveness,
                     std::optional<uint16_t> min_preferred_sgpr) {
  if (inst == nullptr || liveness == nullptr)
    return std::nullopt;
  if (min_preferred_sgpr) {
    if (auto high_sgpr = liveness->find_free_sgpr(inst, *min_preferred_sgpr))
      return high_sgpr;
  }
  return liveness->find_free_sgpr(inst);
}

[[nodiscard]] std::vector<uint64_t> kernel_entry_offsets(const SuperColliderDbiResult &result) {
  std::vector<uint64_t> offsets;
  offsets.reserve(result.kernels.size());
  for (const SuperColliderDbiKernelInfo &kernel : result.kernels) {
    if (kernel.has_text_range)
      offsets.push_back(kernel.entry_text_offset);
  }
  std::ranges::sort(offsets);
  offsets.erase(std::ranges::unique(offsets).begin(), offsets.end());
  return offsets;
}

[[nodiscard]] std::vector<BasicBlock *>
block_ptrs_for(std::vector<std::unique_ptr<BasicBlock>> &blocks) {
  std::vector<BasicBlock *> ptrs;
  ptrs.reserve(blocks.size());
  for (const auto &block : blocks) {
    if (block != nullptr)
      ptrs.push_back(block.get());
  }
  return ptrs;
}

[[nodiscard]] const Instruction *
find_instruction_at_text_offset(std::span<BasicBlock *const> blocks, uint64_t text_offset) {
  for (BasicBlock *block : blocks) {
    if (block == nullptr || text_offset < block->start_offset() ||
        text_offset >= block->end_offset())
      continue;
    for (const Instruction &inst : block->instructions()) {
      if (inst.src_loc() == text_offset)
        return &inst;
    }
  }
  return nullptr;
}

[[nodiscard]] bool text_offset_is_inside_s_clause(std::span<BasicBlock *const> blocks,
                                                  uint64_t text_offset) {
  for (BasicBlock *block : blocks) {
    if (block == nullptr || text_offset < block->start_offset() ||
        text_offset >= block->end_offset())
      continue;
    uint32_t clause_remaining = 0;
    for (const Instruction &inst : block->instructions()) {
      const bool blocked_by_clause = clause_remaining > 0;
      if (clause_remaining > 0)
        --clause_remaining;
      if (is_s_clause(inst))
        clause_remaining = s_clause_following_instruction_count(inst);
      if (inst.src_loc() == text_offset)
        return blocked_by_clause;
    }
  }
  return false;
}

void update_max_vgpr_ref(const RegisterSet &set, std::optional<uint16_t> &max_vgpr) {
  set.for_each([&](RegisterRef ref) {
    if (ref.cls != RegClass::VGPR)
      return;
    if (!max_vgpr || ref.index > *max_vgpr)
      max_vgpr = ref.index;
  });
}

void update_max_sgpr_ref(const RegisterSet &set, std::optional<uint16_t> &max_sgpr) {
  set.for_each([&](RegisterRef ref) {
    if (ref.cls != RegClass::SGPR)
      return;
    if (!max_sgpr || ref.index > *max_sgpr)
      max_sgpr = ref.index;
  });
}

struct KernelMaxRegisterRefs {
  std::optional<uint16_t> sgpr;
  std::optional<uint16_t> vgpr;
};

[[nodiscard]] KernelMaxRegisterRefs
max_register_refs_in_kernel(const SuperColliderDbiKernelInfo &kernel,
                            std::span<BasicBlock *const> blocks) {
  KernelMaxRegisterRefs max_refs;
  if (!kernel.has_text_range || kernel.code_size == 0)
    return max_refs;

  const uint64_t begin = kernel.entry_text_offset;
  const uint64_t end = begin + kernel.code_size;
  for (BasicBlock *block : blocks) {
    if (block == nullptr || block->end_offset() <= begin || block->start_offset() >= end)
      continue;
    for (const Instruction &inst : block->instructions()) {
      if (inst.src_loc() < begin || inst.src_loc() >= end)
        continue;
      InstDefUse du(inst);
      update_max_sgpr_ref(du.uses, max_refs.sgpr);
      update_max_sgpr_ref(du.defs, max_refs.sgpr);
      update_max_vgpr_ref(du.uses, max_refs.vgpr);
      update_max_vgpr_ref(du.defs, max_refs.vgpr);
    }
  }
  return max_refs;
}

[[nodiscard]] bool read_words_at(std::span<const uint8_t> bytes, uint64_t offset,
                                 std::span<uint32_t> words) {
  const uint64_t size = static_cast<uint64_t>(words.size() * sizeof(uint32_t));
  if (offset > bytes.size() || size > bytes.size() - offset)
    return false;
  std::memcpy(words.data(), bytes.data() + offset, static_cast<size_t>(size));
  return true;
}

void append_words(std::vector<uint8_t> &bytes, std::span<const uint32_t> words) {
  const size_t old_size = bytes.size();
  const size_t added_size = words.size() * sizeof(uint32_t);
  bytes.resize(old_size + added_size);
  std::memcpy(bytes.data() + old_size, words.data(), added_size);
}

[[nodiscard]] bool find_indexed_barrier_in_range(std::span<const uint8_t> code_object_bytes,
                                                 Decoder &decoder, std::string_view range_name,
                                                 uint64_t text_file_offset,
                                                 uint64_t entry_text_offset, uint64_t code_size,
                                                 uint32_t barrier_index, uint32_t &decoded_barriers,
                                                 BarrierSite &out) {
  if (code_size == 0)
    return false;
  if (entry_text_offset > code_object_bytes.size() ||
      text_file_offset > code_object_bytes.size() - entry_text_offset)
    return false;
  const uint64_t start = text_file_offset + entry_text_offset;
  if (start > code_object_bytes.size() || code_size > code_object_bytes.size() - start)
    return false;

  const auto text =
      code_object_bytes.subspan(static_cast<size_t>(start), static_cast<size_t>(code_size));
  size_t offset = 0;
  while (offset < text.size()) {
    const size_t remaining = text.size() - offset;
    if (remaining < sizeof(uint32_t))
      return false;

    std::array<uint32_t, 4> words{};
    std::memcpy(words.data(), text.data() + offset,
                std::min(words.size() * sizeof(uint32_t), remaining));

    std::unique_ptr<Instruction> inst(
        decoder.decode(words.data(), entry_text_offset + static_cast<uint64_t>(offset)));
    if (!inst)
      return false;

    const int inst_size = inst->size();
    if (inst_size <= 0 || inst_size % static_cast<int>(sizeof(uint32_t)) != 0 ||
        static_cast<size_t>(inst_size) > remaining)
      return false;

    if (is_barrier_instruction(*inst)) {
      if (decoded_barriers == barrier_index && inst_size == static_cast<int>(sizeof(uint32_t))) {
        out.text_offset = entry_text_offset + static_cast<uint64_t>(offset);
        out.file_offset = text_file_offset + out.text_offset;
        out.size = static_cast<uint32_t>(inst_size);
        out.range_name = std::string(range_name);
        out.mnemonic = std::string(inst->mnemonic());
        return true;
      }
      ++decoded_barriers;
    }

    offset += static_cast<size_t>(inst_size);
  }
  return false;
}

[[nodiscard]] std::optional<BarrierSite>
find_indexed_barrier_site(std::span<const uint8_t> code_object_bytes, rj_code_arch_t arch,
                          const SuperColliderDbiResult &result, uint32_t barrier_index,
                          uint32_t &decoded_barriers) {
  std::unique_ptr<Decoder> decoder = Decoder::create(arch);
  if (!decoder)
    return std::nullopt;

  BarrierSite site;
  for (const SuperColliderDbiKernelInfo &kernel : result.kernels) {
    if (!kernel.has_text_range || !kernel.decoded)
      continue;
    const std::string range_name = "kernel:" + kernel.name;
    if (find_indexed_barrier_in_range(code_object_bytes, *decoder, range_name,
                                      kernel.text_file_offset, kernel.entry_text_offset,
                                      kernel.code_size, barrier_index, decoded_barriers, site))
      return site;
  }
  for (const SuperColliderDbiFunctionInfo &function : result.functions) {
    if (!function.decoded)
      continue;
    const std::string range_name = "function:" + function.name;
    if (find_indexed_barrier_in_range(code_object_bytes, *decoder, range_name,
                                      function.text_file_offset, function.entry_text_offset,
                                      function.code_size, barrier_index, decoded_barriers, site))
      return site;
  }
  return std::nullopt;
}

[[nodiscard]] uint32_t count_nop_padding(std::span<const uint8_t> bytes, uint64_t offset,
                                         uint32_t max_word_count, rj_code_arch_t arch) {
  const uint32_t nop = build_s_nop(0, arch);
  uint32_t count = 0;
  for (; count < max_word_count; ++count) {
    uint32_t word = 0;
    if (!read_words_at(bytes, offset + count * sizeof(uint32_t), std::span<uint32_t>(&word, 1)) ||
        word != nop)
      break;
  }
  return count;
}

[[nodiscard]] bool is_rocclr_runtime_kernel_name(std::string_view name) {
  return name.starts_with("__amd_rocclr_");
}

[[nodiscard]] bool has_only_rocclr_runtime_kernels(const SuperColliderDbiResult &result) {
  if (result.kernels.empty())
    return false;
  for (const SuperColliderDbiKernelInfo &kernel : result.kernels) {
    if (!is_rocclr_runtime_kernel_name(kernel.name))
      return false;
  }
  return true;
}

[[nodiscard]] bool has_proof_nop_candidate(const SuperColliderDbiResult &result) {
  for (const SuperColliderDbiKernelInfo &kernel : result.kernels) {
    if (kernel.preflight_action == SuperColliderDbiPreflightAction::Candidate)
      return true;
  }
  return false;
}

void try_apply_proof_nop_patch(const AmdGpuCodeObject &code_object, rj_code_arch_t arch,
                               bool force_trampoline, SuperColliderDbiResult &result) {
  if (has_only_rocclr_runtime_kernels(result)) {
    result.warnings.emplace_back(
        "DBI SuperCollider proof NOP patch skipped ROCclr runtime helper code object");
    return;
  }
  if (!has_proof_nop_candidate(result)) {
    result.warnings.emplace_back(
        "DBI SuperCollider proof NOP patch skipped code object without supported DBI candidate "
        "sites");
    return;
  }

  if (!force_trampoline) {
    if (auto nop_site = find_existing_nop_site(code_object, arch)) {
      const uint32_t replacement_nop = build_s_nop(1, arch);
      if (!rewrite_word_in_place(code_object, nop_site->file_offset, replacement_nop, result))
        return;

      SuperColliderDbiPatchInfo info;
      info.kind = SuperColliderDbiPatchKind::InlineNopRewrite;
      info.anchor_offset = nop_site->text_offset;
      info.trampoline_offset = nop_site->text_offset;
      info.original_size = sizeof(replacement_nop);
      result.patches.push_back(info);
      result.modified = true;
      return;
    }
  }

  std::string error;
  auto anchor = find_first_relocatable_anchor(code_object, arch, &error);
  if (!anchor) {
    result.errors.push_back("DBI SuperCollider proof NOP patch skipped: " + error);
    return;
  }

  Instrumentor instrumentor(code_object, arch);
  instrumentor.add_point_by_offset(*anchor);
  auto patched = instrumentor.patch_with_debug_summaries();
  if (!patched.errors.empty()) {
    result.errors.push_back("DBI SuperCollider proof NOP patch failed: " + patched.errors.front());
    return;
  }
  if (patched.elf_bytes.empty()) {
    result.errors.emplace_back("DBI SuperCollider proof NOP patch produced empty ELF bytes");
    return;
  }

  result.elf_bytes = std::move(patched.elf_bytes);
  result.modified = true;
  result.patches.reserve(patched.patches.size());
  for (const InstrumentationPatch &patch : patched.patches) {
    SuperColliderDbiPatchInfo info;
    info.kind = SuperColliderDbiPatchKind::TrampolineNop;
    info.anchor_offset = patch.anchor_offset;
    info.trampoline_offset = patch.trampoline_offset;
    info.original_size = patch.original_size;
    result.patches.push_back(info);
  }
}

void try_apply_barrier_drop_fault_patch(const AmdGpuCodeObject &code_object, rj_code_arch_t arch,
                                        const SuperColliderDbiOptions &options,
                                        SuperColliderDbiResult &result) {
  if (has_only_rocclr_runtime_kernels(result)) {
    result.warnings.emplace_back(
        "DBI SuperCollider barrier fault skipped ROCclr runtime helper code object");
    return;
  }

  const auto original_bytes = std::span<const uint8_t>(
      reinterpret_cast<const uint8_t *>(code_object.image_data()), code_object.image_size());

  uint32_t decoded_barriers = 0;
  const std::optional<BarrierSite> site = find_indexed_barrier_site(
      original_bytes, arch, result, options.fault_barrier_index, decoded_barriers);
  if (!site) {
    result.warnings.emplace_back(
        "DBI SuperCollider barrier fault found no patchable barrier at index " +
        std::to_string(options.fault_barrier_index) +
        "; decoded_barriers=" + std::to_string(decoded_barriers));
    return;
  }

  if (result.elf_bytes.empty())
    result.elf_bytes.assign(original_bytes.begin(), original_bytes.end());

  const uint32_t replacement_nop = build_s_nop(0, arch);
  if (site->file_offset > result.elf_bytes.size() ||
      sizeof(replacement_nop) > result.elf_bytes.size() - site->file_offset) {
    result.errors.emplace_back("DBI SuperCollider barrier fault offset is outside ELF bytes");
    if (!result.modified)
      result.elf_bytes.clear();
    return;
  }

  std::memcpy(result.elf_bytes.data() + site->file_offset, &replacement_nop,
              sizeof(replacement_nop));

  SuperColliderDbiPatchInfo info;
  info.kind = SuperColliderDbiPatchKind::InlineBarrierNopRewrite;
  info.anchor_offset = site->text_offset;
  info.trampoline_offset = site->text_offset;
  info.original_size = site->size;
  result.patches.push_back(info);
  result.modified = true;
  result.warnings.emplace_back("DBI SuperCollider barrier fault rewrote " + site->mnemonic +
                               " in " + site->range_name);
}

void try_apply_proof_endpgm_patch(const AmdGpuCodeObject &code_object, rj_code_arch_t arch,
                                  SuperColliderDbiResult &result) {
  auto site = find_preferred_in_place_instruction_site(code_object, arch);
  if (!site) {
    result.errors.emplace_back(
        "DBI SuperCollider proof endpgm patch found no preferred 4-byte vector ALU site");
    return;
  }

  const uint32_t replacement_endpgm = build_s_endpgm(arch);
  if (!rewrite_word_in_place(code_object, site->file_offset, replacement_endpgm, result))
    return;

  SuperColliderDbiPatchInfo info;
  info.kind = SuperColliderDbiPatchKind::InlineEndpgmRewrite;
  info.anchor_offset = site->text_offset;
  info.trampoline_offset = site->text_offset;
  info.original_size = site->size;
  result.patches.push_back(info);
}

[[nodiscard]] const SuperColliderDbiLdsSite *
find_first_supported_lds_read_b32_site(const SuperColliderDbiResult &result) {
  const SuperColliderDbiLdsSite *best = nullptr;
  for (const SuperColliderDbiKernelInfo &kernel : result.kernels) {
    if (kernel.preflight_action == SuperColliderDbiPreflightAction::Reject)
      continue;
    for (const SuperColliderDbiLdsSite &site : kernel.lds_sites) {
      if (site.kind == SuperColliderDbiLdsAccessKind::Read && site.supported_mvp &&
          site.width_bits == 32 && (best == nullptr || site.file_offset < best->file_offset))
        best = &site;
    }
  }
  return best;
}

void try_apply_lds_endpgm_patch(const AmdGpuCodeObject &code_object, rj_code_arch_t arch,
                                SuperColliderDbiResult &result) {
  const SuperColliderDbiLdsSite *site = find_first_supported_lds_read_b32_site(result);
  if (site == nullptr) {
    result.warnings.emplace_back(
        "DBI SuperCollider LDS endpgm proof found no supported ds_read_b32 site");
    return;
  }

  const uint32_t replacement_endpgm = build_s_endpgm(arch);
  if (!rewrite_word_in_place(code_object, site->file_offset, replacement_endpgm, result))
    return;

  SuperColliderDbiPatchInfo info;
  info.kind = SuperColliderDbiPatchKind::InlineLdsEndpgmRewrite;
  info.anchor_offset = site->text_offset;
  info.trampoline_offset = site->text_offset;
  info.original_size = site->size;
  result.patches.push_back(info);
}

[[nodiscard]] bool is_supported_check_trap_site(const SuperColliderDbiLdsSite &site) {
  if (site.size != 2u * sizeof(uint32_t) || !site.addr_vgpr)
    return false;
  auto dword_count = lds_dword_count(site);
  if (!dword_count)
    return false;
  if (site.kind == SuperColliderDbiLdsAccessKind::Read) {
    if (site.mnemonic != "ds_load_b32" && site.mnemonic != "ds_load_b64" &&
        site.mnemonic != "ds_load_b128" && site.mnemonic != "ds_load_2addr_b32" &&
        site.mnemonic != "ds_load_2addr_b64" && site.mnemonic != "ds_load_2addr_stride64_b32" &&
        site.mnemonic != "ds_load_2addr_stride64_b64" && site.mnemonic != "ds_load_u16_d16" &&
        site.mnemonic != "ds_load_u16_d16_hi")
      return false;
    return site.dst_vgpr.has_value() && static_cast<uint32_t>(*site.dst_vgpr) + *dword_count <= 256;
  }
  if (site.kind == SuperColliderDbiLdsAccessKind::Write) {
    if (site.mnemonic != "ds_store_b32" && site.mnemonic != "ds_store_b64" &&
        site.mnemonic != "ds_store_b128")
      return false;
    return site.data_vgpr.has_value() &&
           static_cast<uint32_t>(*site.data_vgpr) + *dword_count <= 256;
  }
  return false;
}

[[nodiscard]] bool is_likely_group_flat_hint(SuperColliderDbiFlatAddressSpaceHint hint) {
  return hint == SuperColliderDbiFlatAddressSpaceHint::Group ||
         hint == SuperColliderDbiFlatAddressSpaceHint::MaybeGroup;
}

[[nodiscard]] bool is_supported_flat_trap_site(const SuperColliderDbiFlatSite &site) {
  if (site.kind != SuperColliderDbiLdsAccessKind::Read &&
      site.kind != SuperColliderDbiLdsAccessKind::Write)
    return false;
  if (site.size != 3u * sizeof(uint32_t))
    return false;
  return is_likely_group_flat_hint(site.address_space_hint);
}

[[nodiscard]] std::optional<uint16_t> flat_dword_count(const SuperColliderDbiFlatSite &site) {
  if (site.width_bits == 32 || site.width_bits == 64 || site.width_bits == 128)
    return static_cast<uint16_t>(site.width_bits / 32u);
  return std::nullopt;
}

[[nodiscard]] bool is_supported_flat_check_trap_site(const SuperColliderDbiFlatSite &site) {
  if (site.kind != SuperColliderDbiLdsAccessKind::Read &&
      site.kind != SuperColliderDbiLdsAccessKind::Write)
    return false;
  if (site.size != 3u * sizeof(uint32_t) || !site.addr_vgpr)
    return false;
  auto dword_count = flat_dword_count(site);
  if (!dword_count)
    return false;
  if (site.kind == SuperColliderDbiLdsAccessKind::Read) {
    if (site.mnemonic != "flat_load_b32" && site.mnemonic != "flat_load_b64" &&
        site.mnemonic != "flat_load_b128")
      return false;
    if (!site.dst_vgpr || static_cast<uint32_t>(*site.dst_vgpr) + *dword_count > 256)
      return false;
  } else {
    if (site.mnemonic != "flat_store_b32" && site.mnemonic != "flat_store_b64" &&
        site.mnemonic != "flat_store_b128")
      return false;
    if (!site.data_vgpr || static_cast<uint32_t>(*site.data_vgpr) + *dword_count > 256)
      return false;
  }
  return is_likely_group_flat_hint(site.address_space_hint);
}

[[nodiscard]] bool is_forbidden_flat_scratch_vgpr_run(const SuperColliderDbiFlatSite &site,
                                                      uint16_t candidate, uint16_t required_vgprs) {
  if (site.addr_vgpr && vgpr_ranges_overlap(candidate, required_vgprs, *site.addr_vgpr, 2))
    return true;

  const uint16_t data_vgprs = flat_dword_count(site).value_or(1);
  if (site.dst_vgpr && vgpr_ranges_overlap(candidate, required_vgprs, *site.dst_vgpr, data_vgprs))
    return true;
  if (site.data_vgpr && vgpr_ranges_overlap(candidate, required_vgprs, *site.data_vgpr, data_vgprs))
    return true;
  return false;
}

[[nodiscard]] uint16_t flat_scratch_search_start(const SuperColliderDbiFlatSite &site) {
  std::optional<uint32_t> first_after_operands;
  auto note_range = [&first_after_operands](std::optional<uint16_t> base, uint16_t count) {
    if (!base)
      return;
    const uint32_t end = static_cast<uint32_t>(*base) + count;
    if (!first_after_operands || end > *first_after_operands)
      first_after_operands = end;
  };

  const uint16_t data_vgprs = flat_dword_count(site).value_or(1);
  note_range(site.addr_vgpr, 2);
  note_range(site.dst_vgpr, data_vgprs);
  note_range(site.data_vgpr, data_vgprs);
  if (!first_after_operands)
    return 0;
  return *first_after_operands <= 255 ? static_cast<uint16_t>(*first_after_operands) : 256;
}

[[nodiscard]] std::optional<uint16_t>
choose_flat_scratch_vgpr(const SuperColliderDbiFlatSite &site,
                         const SuperColliderDbiOptions &options, uint16_t required_vgprs) {
  if (options.scratch_vgpr) {
    if (static_cast<uint32_t>(*options.scratch_vgpr) + required_vgprs <= 256 &&
        !is_forbidden_flat_scratch_vgpr_run(site, *options.scratch_vgpr, required_vgprs))
      return *options.scratch_vgpr;
    return std::nullopt;
  }

  const uint16_t first = flat_scratch_search_start(site);
  return static_cast<uint32_t>(first) + required_vgprs <= 256 ? std::optional<uint16_t>(first)
                                                              : std::nullopt;
}

[[nodiscard]] std::array<uint32_t, 3> retarget_flat_load_vdst(std::array<uint32_t, 3> words,
                                                              uint16_t vdst) {
  rdna4::VflatMachineInst inst{};
  std::memcpy(&inst, words.data(), sizeof(inst));
  inst.vdst = vdst;
  std::memcpy(words.data(), &inst, sizeof(inst));
  return words;
}

[[nodiscard]] std::optional<uint8_t> flat_load_op_for_width(uint32_t width_bits) {
  switch (width_bits) {
  case 32:
    return 0x14;
  case 64:
    return 0x15;
  case 128:
    return 0x17;
  default:
    return std::nullopt;
  }
}

[[nodiscard]] std::optional<std::array<uint32_t, 3>>
build_flat_load_from_flat_store(std::array<uint32_t, 3> words, uint32_t width_bits, uint16_t vdst) {
  auto load_op = flat_load_op_for_width(width_bits);
  if (!load_op)
    return std::nullopt;
  rdna4::VflatMachineInst inst{};
  std::memcpy(&inst, words.data(), sizeof(inst));
  inst.op = *load_op;
  inst.vdst = vdst;
  inst.vsrc = 0;
  std::memcpy(words.data(), &inst, sizeof(inst));
  return words;
}

[[nodiscard]] std::optional<uint16_t>
flat_check_trap_compare_vgpr(const SuperColliderDbiFlatSite &site) {
  if (site.kind == SuperColliderDbiLdsAccessKind::Read)
    return site.dst_vgpr;
  if (site.kind == SuperColliderDbiLdsAccessKind::Write)
    return site.data_vgpr;
  return std::nullopt;
}

[[nodiscard]] bool has_supported_check_trap_site(const SuperColliderDbiResult &result) {
  for (const SuperColliderDbiKernelInfo &kernel : result.kernels) {
    if (kernel.preflight_action == SuperColliderDbiPreflightAction::Reject)
      continue;
    for (const SuperColliderDbiLdsSite &site : kernel.lds_sites) {
      if (is_supported_check_trap_site(site))
        return true;
    }
  }
  return false;
}

[[nodiscard]] std::optional<uint16_t> check_trap_compare_vgpr(const SuperColliderDbiLdsSite &site) {
  if (site.kind == SuperColliderDbiLdsAccessKind::Read)
    return site.dst_vgpr;
  if (site.kind == SuperColliderDbiLdsAccessKind::Write)
    return site.data_vgpr;
  return std::nullopt;
}

struct ByteRange {
  uint64_t begin = 0;
  uint64_t end = 0;
};

[[nodiscard]] bool ranges_overlap(ByteRange lhs, ByteRange rhs) {
  return lhs.begin < rhs.end && rhs.begin < lhs.end;
}

[[nodiscard]] bool overlaps_reserved_range(std::span<const ByteRange> ranges, ByteRange range) {
  for (ByteRange reserved : ranges) {
    if (ranges_overlap(reserved, range))
      return true;
  }
  return false;
}

[[nodiscard]] bool reserve_byte_range(std::vector<ByteRange> &ranges, ByteRange range) {
  if (range.begin >= range.end || overlaps_reserved_range(ranges, range))
    return false;
  ranges.push_back(range);
  return true;
}

[[nodiscard]] SuperColliderDbiPatchKind
lds_check_trap_patch_kind(const SuperColliderDbiLdsSite &site, bool use_local_cave) {
  if (site.kind == SuperColliderDbiLdsAccessKind::Write)
    return use_local_cave ? SuperColliderDbiPatchKind::LocalCaveLdsStoreCheckTrap
                          : SuperColliderDbiPatchKind::InlineLdsStoreCheckTrap;
  return use_local_cave ? SuperColliderDbiPatchKind::LocalCaveLdsLoadCheckTrap
                        : SuperColliderDbiPatchKind::InlineLdsLoadCheckTrap;
}

[[nodiscard]] bool is_d16_lds_load(const SuperColliderDbiLdsSite &site) {
  return site.mnemonic == "ds_load_u16_d16" || site.mnemonic == "ds_load_u16_d16_hi";
}

[[nodiscard]] uint32_t lds_check_trap_predelay_setup_words(const SuperColliderDbiLdsSite &site) {
  return is_d16_lds_load(site) ? 2u : 0u;
}

[[nodiscard]] std::optional<uint32_t>
build_v_mov_b32_e32_vgpr_word(uint16_t vdst, uint16_t src_vgpr, rj_code_arch_t arch) {
  if (arch != ROCJITSU_CODE_ARCH_RDNA4 || vdst > 255 || src_vgpr > 255)
    return std::nullopt;
  return build_v_mov_b32_e32(vdst, vector_source_vgpr(src_vgpr), arch);
}

[[nodiscard]] std::optional<std::vector<uint32_t>> build_lds_check_trap_words(
    std::span<const uint8_t> original_bytes, const SuperColliderDbiLdsSite &site,
    rj_code_arch_t arch, const SuperColliderDbiOptions &options, uint32_t delay_words,
    uint16_t scratch_vgpr, uint16_t vcc_save_sgpr, std::vector<std::string> &errors) {
  auto required_vgprs = lds_dword_count(site);
  if (!required_vgprs) {
    errors.emplace_back("DBI SuperCollider LDS check/trap proof selected unsupported width");
    return std::nullopt;
  }

  std::array<uint32_t, 2> original_access{};
  if (!read_words_at(original_bytes, site.file_offset, std::span<uint32_t>(original_access))) {
    errors.emplace_back("DBI SuperCollider LDS check/trap proof could not read LDS access");
    return std::nullopt;
  }

  std::array<uint32_t, 2> duplicate_load = original_access;
  if (site.kind == SuperColliderDbiLdsAccessKind::Read) {
    duplicate_load[1] =
        (duplicate_load[1] & 0x00FFFFFFu) | (static_cast<uint32_t>(scratch_vgpr) << 24u);
  } else if (site.kind == SuperColliderDbiLdsAccessKind::Write) {
    auto word0 = build_ds_load_word0_from_vds_word0(original_access[0], site.width_bits, arch);
    if (!word0) {
      errors.emplace_back("DBI SuperCollider LDS check/trap proof could not encode store readback");
      return std::nullopt;
    }
    duplicate_load[0] = *word0;
    duplicate_load[1] = build_ds_load_word1(*site.addr_vgpr, scratch_vgpr);
  } else {
    errors.emplace_back("DBI SuperCollider LDS check/trap proof selected unsupported site");
    return std::nullopt;
  }

  auto wait_dscnt = build_s_wait_dscnt_word(0, arch);
  auto compare_vgpr = check_trap_compare_vgpr(site);
  if (!compare_vgpr) {
    errors.emplace_back(
        "DBI SuperCollider LDS check/trap proof selected site without compare VGPR");
    return std::nullopt;
  }
  auto skip_trap = build_s_cbranch_vccz_word(1, arch);
  auto trap = build_s_trap_word(0, arch);
  constexpr uint16_t kRdna4VccLo = 106;
  const uint32_t save_vcc = build_s_mov_b32(vcc_save_sgpr, kRdna4VccLo, arch);
  const uint32_t restore_vcc = build_s_mov_b32(kRdna4VccLo, vcc_save_sgpr, arch);
  if (!wait_dscnt || !skip_trap || !trap) {
    errors.emplace_back("DBI SuperCollider LDS check/trap proof could not encode sequence");
    return std::nullopt;
  }

  std::optional<uint32_t> d16_seed_scratch;
  if (is_d16_lds_load(site)) {
    d16_seed_scratch = build_v_mov_b32_e32_vgpr_word(scratch_vgpr, *compare_vgpr, arch);
    if (!d16_seed_scratch) {
      errors.emplace_back("DBI SuperCollider LDS check/trap proof could not encode d16 scratch "
                          "seed move");
      return std::nullopt;
    }
  }

  const uint32_t total_words =
      static_cast<uint32_t>(2u + lds_check_trap_predelay_setup_words(site) + delay_words + 2u + 1u +
                            1u + 3u * *required_vgprs + 1u);
  std::vector<uint32_t> words;
  words.reserve(total_words);
  words.push_back(original_access[0]);
  words.push_back(original_access[1]);
  if (d16_seed_scratch) {
    words.push_back(*wait_dscnt);
    words.push_back(*d16_seed_scratch);
  }
  if (!append_delay_words(words, arch, options, errors, "DBI SuperCollider LDS check/trap proof"))
    return std::nullopt;
  words.push_back(duplicate_load[0]);
  words.push_back(duplicate_load[1]);
  words.push_back(*wait_dscnt);
  words.push_back(save_vcc);
  for (uint16_t i = 0; i < *required_vgprs; ++i) {
    auto chunk_cmp_ne = build_v_cmp_ne_u32_e32_word(static_cast<uint16_t>(*compare_vgpr + i),
                                                    static_cast<uint16_t>(scratch_vgpr + i), arch);
    if (!chunk_cmp_ne) {
      errors.emplace_back("DBI SuperCollider LDS check/trap proof could not encode chunk compare");
      return std::nullopt;
    }
    words.push_back(*chunk_cmp_ne);
    words.push_back(*skip_trap);
    words.push_back(*trap);
  }
  words.push_back(restore_vcc);
  return words;
}

void try_apply_lds_load_check_trap_patch(const AmdGpuCodeObject &code_object, rj_code_arch_t arch,
                                         const SuperColliderDbiOptions &options,
                                         SuperColliderDbiResult &result) {
  if (arch != ROCJITSU_CODE_ARCH_RDNA4) {
    result.warnings.emplace_back(
        "DBI SuperCollider LDS check/trap proof currently supports only RDNA4");
    return;
  }

  const auto delay_words = delay_instruction_word_count(options, result.errors,
                                                        "DBI SuperCollider LDS check/trap proof");
  if (!delay_words)
    return;

  if (!has_supported_check_trap_site(result)) {
    result.warnings.emplace_back("DBI SuperCollider LDS check/trap proof found no supported "
                                 "ds_load_b{32,64,128}, ds_load_2addr_b{32,64}, "
                                 "ds_load_u16_d16(_hi), or ds_store_b{32,64,128} site");
    return;
  }

  const auto original_bytes = std::span<const uint8_t>(
      reinterpret_cast<const uint8_t *>(code_object.image_data()), code_object.image_size());
  uint64_t appended_cave_offset = 0;
  if (code_object.text_sections().size() == 1)
    appended_cave_offset = code_object.text_sections().front()->size();
  const std::vector<LocalNopCave> local_nop_caves =
      find_uncovered_nop_caves(code_object, result, arch);
  uint32_t max_uncovered_nop_cave_words = 0;
  for (const LocalNopCave &cave : local_nop_caves)
    max_uncovered_nop_cave_words = std::max(max_uncovered_nop_cave_words, cave.word_count);

  std::vector<std::unique_ptr<BasicBlock>> blocks;
  std::vector<BasicBlock *> block_ptrs;
  std::unique_ptr<LivenessAnalysis> liveness;
  std::unique_ptr<Decoder> decoder = Decoder::create(arch);
  if (decoder) {
    const std::vector<uint64_t> extra_leaders = kernel_entry_offsets(result);
    blocks = BasicBlock::build(code_object, *decoder, arch, extra_leaders);
    block_ptrs = block_ptrs_for(blocks);
    if (!block_ptrs.empty())
      liveness = std::make_unique<LivenessAnalysis>(KernelBlockScope(block_ptrs));
  }

  struct LdsCheckTrapCandidate {
    const SuperColliderDbiLdsSite *site = nullptr;
    const LocalNopCave *local_cave = nullptr;
    uint64_t kernel_entry_text_offset = 0;
    uint64_t appended_cave_text_offset = 0;
    uint16_t scratch_vgpr = 0;
    uint16_t vcc_save_sgpr = 0;
    uint64_t requested_words = 0;
    bool use_local_cave = false;
    bool use_appended_cave = false;
  };

  std::vector<LdsCheckTrapCandidate> inline_candidates;
  std::vector<LdsCheckTrapCandidate> local_cave_candidates;
  std::vector<LdsCheckTrapCandidate> appended_cave_candidates;
  bool skipped_supported_site_for_excessive_delay = false;
  size_t supported_candidate_count = 0;
  size_t scratchable_candidate_count = 0;
  size_t append_cave_reachable_candidate_count = 0;
  size_t local_cave_reachable_candidate_count = 0;
  size_t s_clause_blocked_candidate_count = 0;
  uint32_t max_observed_padding_words = 0;
  auto reachable_local_caves = [&](const SuperColliderDbiLdsSite &site, uint64_t requested_words) {
    std::vector<const LocalNopCave *> caves;
    const uint64_t cave_words = requested_words + 1u;
    for (const LocalNopCave &cave : local_nop_caves) {
      if (cave.word_count < cave_words)
        continue;
      const uint64_t return_branch_pc = cave.text_offset + requested_words * sizeof(uint32_t);
      if (compute_sopp_branch_simm16(site.text_offset, cave.text_offset) &&
          compute_sopp_branch_simm16(return_branch_pc, site.text_offset + site.size))
        caves.push_back(&cave);
    }
    return caves;
  };
  for (const SuperColliderDbiKernelInfo &kernel : result.kernels) {
    if (kernel.preflight_action == SuperColliderDbiPreflightAction::Reject)
      continue;
    const auto max_auto_scratch_vgpr =
        descriptor_vgpr_allocation_count(original_bytes, kernel.descriptor_file_offset, arch);
    std::optional<uint16_t> min_auto_scratch_vgpr;
    std::optional<uint16_t> min_preferred_vcc_save_sgpr;
    const KernelMaxRegisterRefs max_refs =
        max_register_refs_in_kernel(kernel, KernelBlockScope(block_ptrs));
    if (!options.scratch_vgpr) {
      if (max_refs.vgpr && *max_refs.vgpr < 255)
        min_auto_scratch_vgpr = static_cast<uint16_t>(*max_refs.vgpr + 1u);
    }
    if (max_refs.sgpr && *max_refs.sgpr + 1u < REGISTER_SET_ALLOCATABLE_SGPRS)
      min_preferred_vcc_save_sgpr = static_cast<uint16_t>(*max_refs.sgpr + 1u);
    for (const SuperColliderDbiLdsSite &site : kernel.lds_sites) {
      if (!is_supported_check_trap_site(site))
        continue;
      ++supported_candidate_count;
      auto required_vgprs = lds_dword_count(site);
      if (!required_vgprs)
        continue;
      const uint64_t requested_words = 2u + lds_check_trap_predelay_setup_words(site) +
                                       static_cast<uint64_t>(*delay_words) + 2u + 1u + 1u +
                                       3u * *required_vgprs + 1u;
      if (requested_words > 256u) {
        skipped_supported_site_for_excessive_delay = true;
        continue;
      }
      const uint32_t padding_words = static_cast<uint32_t>(requested_words - 2u);
      const Instruction *site_inst =
          find_instruction_at_text_offset(KernelBlockScope(block_ptrs), site.text_offset);
      auto scratch =
          choose_scratch_vgpr(site, options, site_inst, liveness.get(), min_auto_scratch_vgpr,
                              max_auto_scratch_vgpr, *required_vgprs);
      if (!scratch)
        continue;
      auto vcc_save = choose_vcc_save_sgpr(site_inst, liveness.get(), min_preferred_vcc_save_sgpr);
      if (!vcc_save)
        continue;
      ++scratchable_candidate_count;
      const uint32_t observed_padding =
          count_nop_padding(original_bytes, site.file_offset + site.size, padding_words, arch);
      max_observed_padding_words = std::max(max_observed_padding_words, observed_padding);
      if (text_offset_is_inside_s_clause(KernelBlockScope(block_ptrs), site.text_offset)) {
        ++s_clause_blocked_candidate_count;
      } else {
        if (appended_cave_offset != 0) {
          const uint64_t return_branch_pc =
              appended_cave_offset + requested_words * sizeof(uint32_t);
          if (compute_sopp_branch_simm16(site.text_offset, appended_cave_offset) &&
              compute_sopp_branch_simm16(return_branch_pc, site.text_offset + site.size)) {
            ++append_cave_reachable_candidate_count;
            appended_cave_candidates.push_back({&site, nullptr, kernel.entry_text_offset,
                                                appended_cave_offset, *scratch, *vcc_save,
                                                requested_words, false, true});
          }
        }
        const std::vector<const LocalNopCave *> caves =
            reachable_local_caves(site, requested_words);
        if (!caves.empty())
          ++local_cave_reachable_candidate_count;
        for (const LocalNopCave *cave : caves) {
          local_cave_candidates.push_back({&site, cave, kernel.entry_text_offset, 0, *scratch,
                                           *vcc_save, requested_words, true, false});
        }
      }
      if (observed_padding < padding_words)
        continue;
      inline_candidates.push_back({&site, nullptr, kernel.entry_text_offset, 0, *scratch, *vcc_save,
                                   requested_words, false, false});
    }
  }

  auto candidate_order = [](const LdsCheckTrapCandidate &lhs, const LdsCheckTrapCandidate &rhs) {
    if (lhs.site->file_offset != rhs.site->file_offset)
      return lhs.site->file_offset < rhs.site->file_offset;
    if (lhs.use_local_cave != rhs.use_local_cave)
      return !lhs.use_local_cave;
    const uint64_t lhs_cave = lhs.local_cave ? lhs.local_cave->file_offset : 0;
    const uint64_t rhs_cave = rhs.local_cave ? rhs.local_cave->file_offset : 0;
    return lhs_cave < rhs_cave;
  };
  std::ranges::sort(inline_candidates, candidate_order);
  std::ranges::sort(local_cave_candidates, candidate_order);
  std::ranges::sort(appended_cave_candidates, candidate_order);

  const uint32_t max_patches = std::max<uint32_t>(options.max_patches, 1u);
  std::vector<LdsCheckTrapCandidate> selected_candidates;
  selected_candidates.reserve(max_patches);
  std::vector<ByteRange> reserved_ranges;
  std::unordered_set<uint64_t> selected_local_cave_kernels;
  auto try_select_candidate = [&](const LdsCheckTrapCandidate &candidate) {
    if (selected_candidates.size() >= max_patches)
      return;
    const SuperColliderDbiLdsSite &site = *candidate.site;
    if (candidate.use_local_cave) {
      if (candidate.local_cave == nullptr)
        return;
      if (selected_local_cave_kernels.contains(candidate.kernel_entry_text_offset))
        return;
      const ByteRange anchor_range{site.file_offset, site.file_offset + site.size};
      const uint64_t cave_bytes = (candidate.requested_words + 1u) * sizeof(uint32_t);
      const ByteRange cave_range{candidate.local_cave->file_offset,
                                 candidate.local_cave->file_offset + cave_bytes};
      if (overlaps_reserved_range(reserved_ranges, anchor_range) ||
          overlaps_reserved_range(reserved_ranges, cave_range))
        return;
      reserved_ranges.push_back(anchor_range);
      reserved_ranges.push_back(cave_range);
      selected_local_cave_kernels.insert(candidate.kernel_entry_text_offset);
      selected_candidates.push_back(candidate);
      return;
    }

    const uint64_t patch_bytes = candidate.requested_words * sizeof(uint32_t);
    const ByteRange patch_range{site.file_offset, site.file_offset + patch_bytes};
    if (!reserve_byte_range(reserved_ranges, patch_range))
      return;
    selected_candidates.push_back(candidate);
  };

  for (const LdsCheckTrapCandidate &candidate : inline_candidates)
    try_select_candidate(candidate);
  for (const LdsCheckTrapCandidate &candidate : local_cave_candidates)
    try_select_candidate(candidate);
  if (selected_candidates.empty()) {
    for (const LdsCheckTrapCandidate &candidate : appended_cave_candidates) {
      const SuperColliderDbiLdsSite &site = *candidate.site;
      const ByteRange anchor_range{site.file_offset, site.file_offset + site.size};
      if (!reserve_byte_range(reserved_ranges, anchor_range))
        continue;
      selected_candidates.push_back(candidate);
      break;
    }
  }

  if (selected_candidates.empty()) {
    if (skipped_supported_site_for_excessive_delay) {
      result.warnings.emplace_back(
          "DBI SuperCollider LDS check/trap proof skipped: requested delay needs too much "
          "padding for the supported LDS site");
      return;
    }
    result.warnings.emplace_back(
        "DBI SuperCollider LDS check/trap proof found no padded or local-cave-reachable "
        "ds_load_b{32,64,128}, ds_load_2addr_b{32,64}, ds_load_u16_d16(_hi), or "
        "ds_store_b{32,64,128} site; supported_candidates=" +
        std::to_string(supported_candidate_count) +
        " scratchable_candidates=" + std::to_string(scratchable_candidate_count) +
        " max_observed_padding_words=" + std::to_string(max_observed_padding_words) +
        " append_cave_reachable_candidates=" +
        std::to_string(append_cave_reachable_candidate_count) +
        " uncovered_nop_caves=" + std::to_string(local_nop_caves.size()) +
        " max_uncovered_nop_cave_words=" + std::to_string(max_uncovered_nop_cave_words) +
        " local_cave_reachable_candidates=" + std::to_string(local_cave_reachable_candidate_count) +
        " s_clause_blocked_candidates=" + std::to_string(s_clause_blocked_candidate_count));
    return;
  }

  struct PlannedLdsCheckTrapPatch {
    const SuperColliderDbiLdsSite *site = nullptr;
    const LocalNopCave *local_cave = nullptr;
    std::vector<uint32_t> words;
    std::vector<uint32_t> cave_words;
    std::array<uint32_t, 2> anchor_words{};
    SuperColliderDbiPatchInfo info;
    bool use_local_cave = false;
    bool use_appended_cave = false;
  };

  std::vector<PlannedLdsCheckTrapPatch> planned_patches;
  planned_patches.reserve(selected_candidates.size());
  for (const LdsCheckTrapCandidate &candidate : selected_candidates) {
    const SuperColliderDbiLdsSite &site = *candidate.site;
    auto words =
        build_lds_check_trap_words(original_bytes, site, arch, options, *delay_words,
                                   candidate.scratch_vgpr, candidate.vcc_save_sgpr, result.errors);
    if (!words)
      return;

    PlannedLdsCheckTrapPatch planned;
    planned.site = &site;
    planned.local_cave = candidate.local_cave;
    planned.words = std::move(*words);
    planned.use_local_cave = candidate.use_local_cave;
    planned.use_appended_cave = candidate.use_appended_cave;
    planned.info.kind =
        lds_check_trap_patch_kind(site, candidate.use_local_cave || candidate.use_appended_cave);
    planned.info.anchor_offset = site.text_offset;
    planned.info.scratch_vgpr = candidate.scratch_vgpr;

    if (candidate.use_local_cave || candidate.use_appended_cave) {
      if (candidate.use_local_cave && candidate.local_cave == nullptr) {
        result.errors.emplace_back(
            "DBI SuperCollider LDS check/trap proof selected missing local cave");
        return;
      }
      if (site.size != 2u * sizeof(uint32_t)) {
        result.errors.emplace_back(
            "DBI SuperCollider LDS check/trap proof local cave expects an 8-byte LDS site");
        return;
      }

      const uint64_t trampoline_text_offset = candidate.use_appended_cave
                                                  ? candidate.appended_cave_text_offset
                                                  : candidate.local_cave->text_offset;
      const auto fwd = compute_sopp_branch_simm16(site.text_offset, trampoline_text_offset);
      const uint64_t return_branch_pc =
          trampoline_text_offset + planned.words.size() * sizeof(uint32_t);
      const auto ret = compute_sopp_branch_simm16(return_branch_pc, site.text_offset + site.size);
      if (!fwd || !ret) {
        result.errors.emplace_back(
            "DBI SuperCollider LDS check/trap proof local cave branch exceeds s_branch simm16");
        return;
      }

      planned.cave_words = planned.words;
      planned.cave_words.push_back(build_s_branch(*ret, arch));
      if (candidate.use_local_cave &&
          candidate.local_cave->word_count < planned.cave_words.size()) {
        result.errors.emplace_back(
            "DBI SuperCollider LDS check/trap proof local cave is too small");
        return;
      }

      planned.anchor_words = {build_s_branch(*fwd, arch), build_s_nop(0, arch)};
      const uint64_t anchor_bytes = planned.anchor_words.size() * sizeof(uint32_t);
      const uint64_t cave_bytes = planned.cave_words.size() * sizeof(uint32_t);
      if (site.file_offset > original_bytes.size() ||
          anchor_bytes > original_bytes.size() - site.file_offset ||
          (candidate.use_local_cave &&
           (candidate.local_cave->file_offset > original_bytes.size() ||
            cave_bytes > original_bytes.size() - candidate.local_cave->file_offset))) {
        result.errors.emplace_back("DBI SuperCollider LDS check/trap proof exceeds ELF bytes");
        return;
      }

      planned.info.trampoline_offset = trampoline_text_offset;
      planned.info.original_size = site.size;
    } else {
      const uint64_t patch_bytes = static_cast<uint64_t>(planned.words.size() * sizeof(uint32_t));
      if (site.file_offset > original_bytes.size() ||
          patch_bytes > original_bytes.size() - site.file_offset) {
        result.errors.emplace_back("DBI SuperCollider LDS check/trap proof exceeds ELF bytes");
        return;
      }
      planned.info.trampoline_offset = site.text_offset + site.size;
      planned.info.original_size = static_cast<uint32_t>(patch_bytes);
    }

    planned_patches.push_back(std::move(planned));
  }

  const bool uses_appended_cave =
      std::ranges::any_of(planned_patches, [](const PlannedLdsCheckTrapPatch &patch) {
        return patch.use_appended_cave;
      });
  if (uses_appended_cave) {
    CodeObjectPatcher patcher(code_object);
    std::span<const uint8_t> text_bytes = patcher.text_bytes();
    std::vector<uint8_t> new_text(text_bytes.begin(), text_bytes.end());
    for (const PlannedLdsCheckTrapPatch &planned : planned_patches) {
      const SuperColliderDbiLdsSite &site = *planned.site;
      if (planned.use_local_cave) {
        std::memcpy(new_text.data() + site.text_offset, planned.anchor_words.data(),
                    planned.anchor_words.size() * sizeof(uint32_t));
        std::memcpy(new_text.data() + planned.local_cave->text_offset, planned.cave_words.data(),
                    planned.cave_words.size() * sizeof(uint32_t));
      } else if (planned.use_appended_cave) {
        std::memcpy(new_text.data() + site.text_offset, planned.anchor_words.data(),
                    planned.anchor_words.size() * sizeof(uint32_t));
        append_words(new_text, planned.cave_words);
      } else {
        std::memcpy(new_text.data() + site.text_offset, planned.words.data(),
                    planned.words.size() * sizeof(uint32_t));
      }
      result.patches.push_back(planned.info);
    }
    if (!patcher.replace_text(new_text)) {
      result.errors.emplace_back(
          "DBI SuperCollider LDS check/trap proof could not append trampoline text");
      result.patches.clear();
      return;
    }
    result.elf_bytes = patcher.emit();
    result.modified = true;
    return;
  }

  if (result.elf_bytes.empty())
    result.elf_bytes.assign(original_bytes.begin(), original_bytes.end());
  for (const PlannedLdsCheckTrapPatch &planned : planned_patches) {
    const SuperColliderDbiLdsSite &site = *planned.site;
    if (planned.use_local_cave) {
      const uint64_t anchor_bytes = planned.anchor_words.size() * sizeof(uint32_t);
      const uint64_t cave_bytes = planned.cave_words.size() * sizeof(uint32_t);
      std::memcpy(result.elf_bytes.data() + site.file_offset, planned.anchor_words.data(),
                  static_cast<size_t>(anchor_bytes));
      std::memcpy(result.elf_bytes.data() + planned.local_cave->file_offset,
                  planned.cave_words.data(), static_cast<size_t>(cave_bytes));
    } else {
      const uint64_t patch_bytes = static_cast<uint64_t>(planned.words.size() * sizeof(uint32_t));
      std::memcpy(result.elf_bytes.data() + site.file_offset, planned.words.data(),
                  static_cast<size_t>(patch_bytes));
    }
    result.patches.push_back(planned.info);
  }
  result.modified = true;
}

void try_apply_flat_check_trap_patch(const AmdGpuCodeObject &code_object, rj_code_arch_t arch,
                                     const SuperColliderDbiOptions &options,
                                     SuperColliderDbiResult &result) {
  if (arch != ROCJITSU_CODE_ARCH_RDNA4) {
    result.warnings.emplace_back(
        "DBI SuperCollider flat check/trap proof currently supports only RDNA4");
    return;
  }

  const auto delay_words = delay_instruction_word_count(options, result.errors,
                                                        "DBI SuperCollider flat check/trap proof");
  if (!delay_words)
    return;

  const auto original_bytes = std::span<const uint8_t>(
      reinterpret_cast<const uint8_t *>(code_object.image_data()), code_object.image_size());
  uint64_t appended_cave_offset = 0;
  if (code_object.text_sections().size() == 1)
    appended_cave_offset = code_object.text_sections().front()->size();
  const std::vector<LocalNopCave> local_nop_caves =
      find_uncovered_nop_caves(code_object, result, arch);
  uint32_t max_uncovered_nop_cave_words = 0;
  for (const LocalNopCave &cave : local_nop_caves)
    max_uncovered_nop_cave_words = std::max(max_uncovered_nop_cave_words, cave.word_count);

  const SuperColliderDbiFlatSite *selected = nullptr;
  uint16_t scratch_vgpr = 0;
  const SuperColliderDbiFlatSite *local_cave_selected = nullptr;
  const LocalNopCave *selected_local_cave = nullptr;
  uint16_t local_cave_scratch_vgpr = 0;
  bool skipped_supported_site_for_excessive_delay = false;
  size_t supported_candidate_count = 0;
  size_t scratchable_candidate_count = 0;
  size_t append_cave_reachable_candidate_count = 0;
  size_t local_cave_reachable_candidate_count = 0;
  uint32_t max_observed_padding_words = 0;
  auto reachable_local_cave = [&](const SuperColliderDbiFlatSite &site,
                                  uint64_t requested_words) -> const LocalNopCave * {
    const uint64_t cave_words = requested_words + 1u;
    for (const LocalNopCave &cave : local_nop_caves) {
      if (cave.word_count < cave_words)
        continue;
      const uint64_t return_branch_pc = cave.text_offset + requested_words * sizeof(uint32_t);
      if (compute_sopp_branch_simm16(site.text_offset, cave.text_offset) &&
          compute_sopp_branch_simm16(return_branch_pc, site.text_offset + site.size))
        return &cave;
    }
    return nullptr;
  };
  auto visit_site = [&](const SuperColliderDbiFlatSite &site) {
    if (!is_supported_flat_check_trap_site(site))
      return;
    ++supported_candidate_count;
    auto required_vgprs = flat_dword_count(site);
    if (!required_vgprs)
      return;
    const uint64_t requested_words =
        3u + static_cast<uint64_t>(*delay_words) + 3u + 1u + 3u * *required_vgprs;
    if (requested_words > 256u) {
      skipped_supported_site_for_excessive_delay = true;
      return;
    }
    const uint32_t padding_words = static_cast<uint32_t>(requested_words - 3u);
    auto scratch = choose_flat_scratch_vgpr(site, options, *required_vgprs);
    if (!scratch)
      return;
    ++scratchable_candidate_count;
    const uint32_t observed_padding =
        count_nop_padding(original_bytes, site.file_offset + site.size, padding_words, arch);
    max_observed_padding_words = std::max(max_observed_padding_words, observed_padding);
    if (appended_cave_offset != 0) {
      const uint64_t return_branch_pc = appended_cave_offset + requested_words * sizeof(uint32_t);
      if (compute_sopp_branch_simm16(site.text_offset, appended_cave_offset) &&
          compute_sopp_branch_simm16(return_branch_pc, site.text_offset + site.size))
        ++append_cave_reachable_candidate_count;
    }
    if (const LocalNopCave *cave = reachable_local_cave(site, requested_words)) {
      ++local_cave_reachable_candidate_count;
      if (local_cave_selected == nullptr || site.file_offset < local_cave_selected->file_offset) {
        local_cave_selected = &site;
        selected_local_cave = cave;
        local_cave_scratch_vgpr = *scratch;
      }
    }
    if (observed_padding < padding_words)
      return;
    if (selected == nullptr || site.file_offset < selected->file_offset) {
      selected = &site;
      scratch_vgpr = *scratch;
    }
  };

  for (const SuperColliderDbiKernelInfo &kernel : result.kernels) {
    if (kernel.preflight_action == SuperColliderDbiPreflightAction::Reject)
      continue;
    for (const SuperColliderDbiFlatSite &site : kernel.flat_sites)
      visit_site(site);
  }
  for (const SuperColliderDbiFunctionInfo &function : result.functions) {
    for (const SuperColliderDbiFlatSite &site : function.flat_sites)
      visit_site(site);
  }

  bool use_local_cave = false;
  if (selected == nullptr && local_cave_selected != nullptr) {
    selected = local_cave_selected;
    scratch_vgpr = local_cave_scratch_vgpr;
    use_local_cave = true;
  }

  if (selected == nullptr) {
    if (skipped_supported_site_for_excessive_delay) {
      result.warnings.emplace_back(
          "DBI SuperCollider flat check/trap proof skipped: requested delay needs too much "
          "padding for the supported flat site");
      return;
    }
    result.warnings.emplace_back(
        "DBI SuperCollider flat check/trap proof found no padded likely "
        "group flat_load/store_b{32,64,128} site; supported_candidates=" +
        std::to_string(supported_candidate_count) +
        " scratchable_candidates=" + std::to_string(scratchable_candidate_count) +
        " max_observed_padding_words=" + std::to_string(max_observed_padding_words) +
        " append_cave_reachable_candidates=" +
        std::to_string(append_cave_reachable_candidate_count) +
        " uncovered_nop_caves=" + std::to_string(local_nop_caves.size()) +
        " max_uncovered_nop_cave_words=" + std::to_string(max_uncovered_nop_cave_words) +
        " local_cave_reachable_candidates=" + std::to_string(local_cave_reachable_candidate_count));
    return;
  }

  auto required_vgprs = flat_dword_count(*selected);
  auto compare_vgpr = flat_check_trap_compare_vgpr(*selected);
  if (!required_vgprs || !compare_vgpr) {
    result.errors.emplace_back("DBI SuperCollider flat check/trap proof selected unsupported site");
    return;
  }
  const uint32_t total_words =
      static_cast<uint32_t>(3u + *delay_words + 3u + 1u + 3u * *required_vgprs);

  std::array<uint32_t, 3> original_access{};
  if (!read_words_at(original_bytes, selected->file_offset, std::span<uint32_t>(original_access))) {
    result.errors.emplace_back("DBI SuperCollider flat check/trap proof could not read access");
    return;
  }

  std::optional<std::array<uint32_t, 3>> duplicate_load;
  SuperColliderDbiPatchKind patch_kind = use_local_cave
                                             ? SuperColliderDbiPatchKind::LocalCaveFlatLoadCheckTrap
                                             : SuperColliderDbiPatchKind::InlineFlatLoadCheckTrap;
  if (selected->kind == SuperColliderDbiLdsAccessKind::Read) {
    duplicate_load = retarget_flat_load_vdst(original_access, scratch_vgpr);
  } else if (selected->kind == SuperColliderDbiLdsAccessKind::Write) {
    duplicate_load =
        build_flat_load_from_flat_store(original_access, selected->width_bits, scratch_vgpr);
    patch_kind = use_local_cave ? SuperColliderDbiPatchKind::LocalCaveFlatStoreCheckTrap
                                : SuperColliderDbiPatchKind::InlineFlatStoreCheckTrap;
  }
  if (!duplicate_load) {
    result.errors.emplace_back("DBI SuperCollider flat check/trap proof could not encode readback");
    return;
  }

  auto wait_dscnt = build_s_wait_dscnt_word(0, arch);
  auto skip_trap = build_s_cbranch_vccz_word(1, arch);
  auto trap = build_s_trap_word(0, arch);
  if (!wait_dscnt || !skip_trap || !trap) {
    result.errors.emplace_back("DBI SuperCollider flat check/trap proof could not encode sequence");
    return;
  }

  std::vector<uint32_t> words;
  words.reserve(total_words);
  words.insert(words.end(), original_access.begin(), original_access.end());
  if (!append_delay_words(words, arch, options, result.errors,
                          "DBI SuperCollider flat check/trap proof"))
    return;
  words.insert(words.end(), duplicate_load->begin(), duplicate_load->end());
  words.push_back(*wait_dscnt);
  for (uint16_t i = 0; i < *required_vgprs; ++i) {
    auto chunk_cmp_ne = build_v_cmp_ne_u32_e32_word(static_cast<uint16_t>(*compare_vgpr + i),
                                                    static_cast<uint16_t>(scratch_vgpr + i), arch);
    if (!chunk_cmp_ne) {
      result.errors.emplace_back(
          "DBI SuperCollider flat check/trap proof could not encode chunk compare");
      return;
    }
    words.push_back(*chunk_cmp_ne);
    words.push_back(*skip_trap);
    words.push_back(*trap);
  }

  result.elf_bytes.assign(original_bytes.begin(), original_bytes.end());
  SuperColliderDbiPatchInfo info;
  info.kind = patch_kind;
  info.anchor_offset = selected->text_offset;
  info.scratch_vgpr = scratch_vgpr;

  if (use_local_cave) {
    if (selected_local_cave == nullptr) {
      result.errors.emplace_back(
          "DBI SuperCollider flat check/trap proof selected missing local cave");
      result.elf_bytes.clear();
      return;
    }
    if (selected->size != 3u * sizeof(uint32_t)) {
      result.errors.emplace_back(
          "DBI SuperCollider flat check/trap proof local cave expects a 12-byte flat site");
      result.elf_bytes.clear();
      return;
    }

    const auto fwd =
        compute_sopp_branch_simm16(selected->text_offset, selected_local_cave->text_offset);
    const uint64_t return_branch_pc =
        selected_local_cave->text_offset + words.size() * sizeof(uint32_t);
    const auto ret =
        compute_sopp_branch_simm16(return_branch_pc, selected->text_offset + selected->size);
    if (!fwd || !ret) {
      result.errors.emplace_back(
          "DBI SuperCollider flat check/trap proof local cave branch exceeds s_branch simm16");
      result.elf_bytes.clear();
      return;
    }

    std::vector<uint32_t> cave_words = words;
    cave_words.push_back(build_s_branch(*ret, arch));
    if (selected_local_cave->word_count < cave_words.size()) {
      result.errors.emplace_back("DBI SuperCollider flat check/trap proof local cave is too small");
      result.elf_bytes.clear();
      return;
    }

    const std::array<uint32_t, 3> anchor_words = {build_s_branch(*fwd, arch), build_s_nop(0, arch),
                                                  build_s_nop(0, arch)};
    const uint64_t anchor_bytes = anchor_words.size() * sizeof(uint32_t);
    const uint64_t cave_bytes = cave_words.size() * sizeof(uint32_t);
    if (selected->file_offset > result.elf_bytes.size() ||
        anchor_bytes > result.elf_bytes.size() - selected->file_offset ||
        selected_local_cave->file_offset > result.elf_bytes.size() ||
        cave_bytes > result.elf_bytes.size() - selected_local_cave->file_offset) {
      result.errors.emplace_back("DBI SuperCollider flat check/trap proof exceeds ELF bytes");
      result.elf_bytes.clear();
      return;
    }

    std::memcpy(result.elf_bytes.data() + selected->file_offset, anchor_words.data(),
                static_cast<size_t>(anchor_bytes));
    std::memcpy(result.elf_bytes.data() + selected_local_cave->file_offset, cave_words.data(),
                static_cast<size_t>(cave_bytes));
    info.trampoline_offset = selected_local_cave->text_offset;
    info.original_size = selected->size;
  } else {
    const uint64_t patch_bytes = static_cast<uint64_t>(words.size() * sizeof(uint32_t));
    if (selected->file_offset > result.elf_bytes.size() ||
        patch_bytes > result.elf_bytes.size() - selected->file_offset) {
      result.errors.emplace_back("DBI SuperCollider flat check/trap proof exceeds ELF bytes");
      result.elf_bytes.clear();
      return;
    }
    std::memcpy(result.elf_bytes.data() + selected->file_offset, words.data(),
                static_cast<size_t>(patch_bytes));
    info.trampoline_offset = selected->text_offset + selected->size;
    info.original_size = static_cast<uint32_t>(patch_bytes);
  }

  result.patches.push_back(info);
  result.modified = true;
}

void try_apply_flat_trap_patch(const AmdGpuCodeObject &code_object, rj_code_arch_t arch,
                               SuperColliderDbiResult &result) {
  if (arch != ROCJITSU_CODE_ARCH_RDNA4) {
    result.warnings.emplace_back("DBI SuperCollider flat trap proof currently supports only RDNA4");
    return;
  }

  auto trap = build_s_trap_word(0, arch);
  if (!trap) {
    result.errors.emplace_back("DBI SuperCollider flat trap proof could not encode s_trap");
    return;
  }

  std::vector<const SuperColliderDbiFlatSite *> candidates;
  for (const SuperColliderDbiKernelInfo &kernel : result.kernels) {
    if (kernel.preflight_action == SuperColliderDbiPreflightAction::Reject)
      continue;
    for (const SuperColliderDbiFlatSite &site : kernel.flat_sites) {
      if (is_supported_flat_trap_site(site))
        candidates.push_back(&site);
    }
  }
  for (const SuperColliderDbiFunctionInfo &function : result.functions) {
    for (const SuperColliderDbiFlatSite &site : function.flat_sites) {
      if (is_supported_flat_trap_site(site))
        candidates.push_back(&site);
    }
  }

  if (candidates.empty()) {
    result.warnings.emplace_back(
        "DBI SuperCollider flat trap proof found no likely group flat_load/store site");
    return;
  }

  std::ranges::sort(candidates,
                    [](const SuperColliderDbiFlatSite *lhs, const SuperColliderDbiFlatSite *rhs) {
                      if (lhs->file_offset != rhs->file_offset)
                        return lhs->file_offset < rhs->file_offset;
                      return lhs->text_offset < rhs->text_offset;
                    });

  result.elf_bytes.assign(reinterpret_cast<const uint8_t *>(code_object.image_data()),
                          reinterpret_cast<const uint8_t *>(code_object.image_data()) +
                              code_object.image_size());
  const std::array<uint32_t, 3> replacement = {*trap, build_s_nop(0, arch), build_s_nop(0, arch)};
  for (const SuperColliderDbiFlatSite *site : candidates) {
    if (site->file_offset > result.elf_bytes.size() ||
        site->size > result.elf_bytes.size() - site->file_offset) {
      result.errors.emplace_back("DBI SuperCollider flat trap proof exceeds ELF bytes");
      result.elf_bytes.clear();
      result.patches.clear();
      result.modified = false;
      return;
    }
    std::memcpy(result.elf_bytes.data() + site->file_offset, replacement.data(),
                replacement.size() * sizeof(uint32_t));

    SuperColliderDbiPatchInfo info;
    info.kind = SuperColliderDbiPatchKind::InlineFlatTrapRewrite;
    info.anchor_offset = site->text_offset;
    info.trampoline_offset = site->text_offset + site->size;
    info.original_size = site->size;
    result.patches.push_back(info);
  }
  result.modified = true;
}

} // namespace

SuperColliderDbiResult try_patch_supercollider_dbi(std::span<const uint8_t> code_object_bytes,
                                                   const SuperColliderDbiOptions &options) {
  SuperColliderDbiResult result;
  result.visited_code_object = true;
  result.input_size = code_object_bytes.size();

  if (!options.enabled)
    return result;

  if (code_object_bytes.empty()) {
    result.errors.emplace_back("DBI SuperCollider received an empty code object");
    return result;
  }

  AmdGpuCodeObject code_object(code_object_bytes.data(), code_object_bytes.size());
  if (!code_object.is_valid()) {
    result.errors.emplace_back("DBI SuperCollider could not parse AMDGPU code object");
    return result;
  }

  result.target_name = target_name(code_object.target_id());
  result.arch_name = arch_name(code_object.target_id());

  for (const Section *section : code_object.text_sections()) {
    SuperColliderDbiTextSection info;
    info.name = section->name();
    info.file_offset = section->sectionOffset();
    info.virtual_address = section->vaddr();
    info.size = section->size();
    result.text_sections.push_back(std::move(info));
  }

  for (const AmdGpuKernelInfo &kernel : code_object.kernels()) {
    SuperColliderDbiKernelInfo info;
    info.name = kernel.name;
    info.descriptor_file_offset = kernel.descriptor_file_offset;
    info.entry_text_offset = kernel.entry_text_offset;
    info.text_file_offset = kernel.text_file_offset;
    info.code_size = kernel.code_size;
    info.has_text_range = kernel.has_text_range;
    result.kernels.push_back(std::move(info));
  }

  std::unordered_set<std::string> kernel_names;
  kernel_names.reserve(code_object.kernels().size());
  for (const AmdGpuKernelInfo &kernel : code_object.kernels())
    kernel_names.insert(kernel.name);

  for (const AmdGpuFunctionInfo &function : code_object.functions()) {
    if (kernel_names.contains(function.name))
      continue;
    SuperColliderDbiFunctionInfo info;
    info.name = function.name;
    info.entry_text_offset = function.entry_text_offset;
    info.text_file_offset = function.text_file_offset;
    info.code_size = function.code_size;
    result.functions.push_back(std::move(info));
  }

  if (result.text_sections.empty())
    result.warnings.emplace_back("DBI SuperCollider found no .text sections");
  if (result.kernels.empty())
    result.warnings.emplace_back("DBI SuperCollider found no kernel descriptor symbols");

  const rj_code_arch_t arch = arch_for_target(code_object.target_id());
  if (arch == ROCJITSU_CODE_ARCH_INVALID) {
    result.warnings.emplace_back("DBI SuperCollider cannot decode unknown target '" +
                                 result.target_name + "'");
    return result;
  }

  std::unique_ptr<Decoder> decoder = Decoder::create(arch);
  if (!decoder) {
    result.warnings.emplace_back("DBI SuperCollider could not create decoder for arch '" +
                                 result.arch_name + "'");
    return result;
  }

  for (SuperColliderDbiKernelInfo &kernel : result.kernels) {
    decode_kernel_stats(code_object_bytes, *decoder, arch, kernel, result.warnings);
    preflight_kernel(kernel, options, result.errors, result.warnings);
  }
  for (SuperColliderDbiFunctionInfo &function : result.functions)
    decode_function_stats(code_object_bytes, *decoder, arch, function, result.warnings);

  if (options.probe_lds_check_trap)
    try_apply_lds_load_check_trap_patch(code_object, arch, options, result);
  else if (options.probe_flat_check_trap)
    try_apply_flat_check_trap_patch(code_object, arch, options, result);
  else if (options.probe_flat_trap)
    try_apply_flat_trap_patch(code_object, arch, result);
  else if (options.probe_lds_endpgm)
    try_apply_lds_endpgm_patch(code_object, arch, result);
  else if (options.probe_endpgm)
    try_apply_proof_endpgm_patch(code_object, arch, result);
  else if (options.probe_nop || options.probe_trampoline_nop)
    try_apply_proof_nop_patch(code_object, arch, options.probe_trampoline_nop, result);

  if (options.fault_drop_barrier)
    try_apply_barrier_drop_fault_patch(code_object, arch, options, result);

  return result;
}

} // namespace rocjitsu
