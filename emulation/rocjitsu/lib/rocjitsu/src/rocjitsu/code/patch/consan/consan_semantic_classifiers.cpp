// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan_semantic_classifiers.h"

#include "rocjitsu/code/patch/consan/consan_access_shape.h"
#include "rocjitsu/isa/instruction.h"

namespace rocjitsu::consan {

bool starts_with_any(std::string_view value, std::initializer_list<std::string_view> prefixes) {
  for (std::string_view prefix : prefixes) {
    if (value.starts_with(prefix))
      return true;
  }
  return false;
}

bool is_ds_atomic(std::string_view mnemonic) {
  return starts_with_any(mnemonic, {"ds_add", "ds_sub", "ds_rsub", "ds_inc", "ds_dec", "ds_min",
                                    "ds_max", "ds_and", "ds_or", "ds_xor", "ds_mskor",
                                    "ds_cmpstore", "ds_cmpst", "ds_storexchg", "ds_condxchg",
                                    "ds_cond_sub", "ds_sub_clamp", "ds_pk_add"});
}

bool is_ds_read(std::string_view mnemonic) {
  return starts_with_any(mnemonic, {"ds_read", "ds_load", "ds_param_load", "ds_direct_load"});
}

bool is_ds_write(std::string_view mnemonic) {
  return starts_with_any(mnemonic, {"ds_write", "ds_store"});
}

bool is_ds_register_lane_operation(std::string_view mnemonic) {
  return starts_with_any(mnemonic, {"ds_bpermute", "ds_permute", "ds_swizzle"});
}

LdsAccessKind lds_access_kind(std::string_view mnemonic) {
  if (!mnemonic.starts_with("ds_"))
    return LdsAccessKind::Other;
  if (is_ds_atomic(mnemonic))
    return LdsAccessKind::Atomic;
  if (is_ds_read(mnemonic))
    return LdsAccessKind::Read;
  if (is_ds_write(mnemonic))
    return LdsAccessKind::Write;
  return LdsAccessKind::Other;
}

LdsAccessKind flat_access_kind(std::string_view mnemonic) {
  if (mnemonic.starts_with("flat_atomic"))
    return LdsAccessKind::Atomic;
  if (mnemonic.starts_with("flat_load"))
    return LdsAccessKind::Read;
  if (mnemonic.starts_with("flat_store"))
    return LdsAccessKind::Write;
  return LdsAccessKind::Other;
}

uint32_t lds_width_bits(std::string_view mnemonic) {
  if (const auto subword = flat_load_subword_semantics(mnemonic))
    return subword->memory_width_bits;
  if (const auto subword = flat_store_subword_semantics(mnemonic))
    return subword->memory_width_bits;
  // In two-address stride-64 forms, the "64" token describes address
  // spacing, not transfer width. Consume the authoritative form rather than
  // inferring the first numeric token in the spelling.
  if (const auto form = detail::decode_native_lds_two_range_shape(mnemonic))
    return form->element_width_bits;
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

bool is_fence_like(std::string_view mnemonic) {
  return starts_with_any(mnemonic, {"s_dcache", "s_icache", "global_wb", "global_inv",
                                    "global_wbinv", "buffer_wb", "buffer_inv", "buffer_wbinv",
                                    "buffer_gl0_inv", "buffer_gl1_inv"});
}

bool is_atomic_instruction(const Instruction &instruction) {
  const std::string_view mnemonic = instruction.mnemonic();
  if (mnemonic.starts_with("ds_"))
    return lds_access_kind(mnemonic) == LdsAccessKind::Atomic;
  return starts_with_any(
      mnemonic, {"flat_atomic", "global_atomic", "scratch_atomic", "buffer_atomic", "s_atomic"});
}

} // namespace rocjitsu::consan
