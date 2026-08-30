// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_program_analysis_gfx12_target_ops.h
/// @brief Shared gfx12 raw atomic normalization recipes.

#pragma once

#include "rocjitsu/code/patch/consan/consan.h"
#include "rocjitsu/code/patch/consan/consan_instruction_semantics.h"
#include "rocjitsu/isa/arch/amdgpu/shared/gfx12_cache_flags.h"

namespace rocjitsu::consan_program_analysis_target_detail {

template <typename Raw>
ConSanFlatMemoryEncoding make_gfx12_flat_memory_encoding(const Raw &raw, uint32_t null_saddr,
                                                         bool exact_size, bool ordinary_well_formed,
                                                         bool ordinary_mutation_supported) {
  return ConSanFlatMemoryEncoding{
      .raw_op = static_cast<uint32_t>(raw.op),
      .raw_saddr = static_cast<uint32_t>(raw.saddr),
      .raw_nv = static_cast<uint32_t>(raw.nv),
      .raw_scale_offset =
          [&] {
            if constexpr (requires { raw.scale_offset; })
              return raw.scale_offset != 0u;
            return false;
          }(),
      .raw_sve = static_cast<uint32_t>(raw.sve),
      .raw_vaddr = static_cast<uint32_t>(raw.vaddr),
      .raw_vsrc = static_cast<uint32_t>(raw.vsrc),
      .raw_vdst = static_cast<uint32_t>(raw.vdst),
      .raw_ioffset = sign_extend_24(static_cast<uint32_t>(raw.ioffset)),
      .raw_segment = 0u,
      .raw_scope = static_cast<uint32_t>(raw.scope),
      .raw_th = static_cast<uint32_t>(raw.th),
      .encoded_segment = ConSanEncodedFlatSegment::Unspecified,
      .scalar_provenance_sgpr = raw.saddr == null_saddr
                                    ? std::nullopt
                                    : std::optional<uint16_t>(static_cast<uint16_t>(raw.saddr)),
      .scope_follows_address_space = false,
      .exact_size = exact_size,
      .ordinary_well_formed = ordinary_well_formed,
      .ordinary_requires_complete_registers = true,
      .ordinary_mutation_supported = ordinary_mutation_supported,
  };
}

template <typename Raw> void fill_gfx12_flat_atomic_site(ConSanAtomicSite &site, const Raw &raw) {
  site.raw_op = static_cast<uint32_t>(raw.op);
  site.raw_saddr = static_cast<uint32_t>(raw.saddr);
  if constexpr (requires { raw.scale_offset; })
    site.raw_scale_offset = raw.scale_offset != 0u;
  site.raw_vaddr = static_cast<uint32_t>(raw.vaddr);
  site.raw_vsrc = static_cast<uint32_t>(raw.vsrc);
  site.raw_vdst = static_cast<uint32_t>(raw.vdst);
  site.raw_ioffset = sign_extend_24(static_cast<uint32_t>(raw.ioffset));
  site.raw_scope = static_cast<uint32_t>(raw.scope);
  site.raw_th = static_cast<uint32_t>(raw.th);
  site.returns_old_value = amdgpu::gfx12_atomic_returns(static_cast<uint8_t>(raw.th));
}

template <typename Raw> void fill_gfx12_buffer_atomic_site(ConSanAtomicSite &site, const Raw &raw) {
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

template <typename Raw> void fill_gfx12_ds_atomic_site(ConSanAtomicSite &site, const Raw &raw) {
  site.raw_op = static_cast<uint32_t>(raw.op);
  site.raw_addr = static_cast<uint32_t>(raw.addr);
  site.raw_data0 = static_cast<uint32_t>(raw.data0);
  site.raw_data1 = static_cast<uint32_t>(raw.data1);
  site.raw_vdst = static_cast<uint32_t>(raw.vdst);
  site.raw_ioffset = static_cast<int32_t>(raw.offset0);
}

} // namespace rocjitsu::consan_program_analysis_target_detail
