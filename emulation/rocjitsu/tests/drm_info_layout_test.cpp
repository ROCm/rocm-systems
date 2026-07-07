// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file drm_info_layout_test.cpp
/// @brief Verifies that the local drm_amdgpu_info_device struct in the
///        interposer matches the layout from the system libdrm header.

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>

#include <libdrm/amdgpu_drm.h>

namespace {

struct LocalDevInfo {
  uint32_t device_id;
  uint32_t chip_rev;
  uint32_t external_rev;
  uint32_t pci_rev;
  uint32_t family;
  uint32_t num_shader_engines;
  uint32_t num_shader_arrays_per_engine;
  uint32_t gpu_counter_freq;
  uint64_t max_engine_clock;
  uint64_t max_memory_clock;
  uint32_t cu_active_number;
  uint32_t cu_ao_mask;
  uint32_t cu_bitmap[4][4];
  uint32_t enabled_rb_pipes_mask;
  uint32_t num_rb_pipes;
  uint32_t num_hw_gfx_contexts;
  uint32_t pcie_gen;
  uint64_t ids_flags;
  uint64_t virtual_address_offset;
  uint64_t virtual_address_max;
  uint32_t virtual_address_alignment;
  uint32_t pte_fragment_size;
  uint32_t gart_page_size;
  uint32_t ce_ram_size;
  uint32_t vram_type;
  uint32_t vram_bit_width;
  uint32_t vce_harvest_config;
  uint32_t gc_double_offchip_lds_buf;
  uint64_t prim_buf_gpu_addr;
  uint64_t pos_buf_gpu_addr;
  uint64_t cntl_sb_buf_gpu_addr;
  uint64_t param_buf_gpu_addr;
  uint32_t prim_buf_size;
  uint32_t pos_buf_size;
  uint32_t cntl_sb_buf_size;
  uint32_t param_buf_size;
  uint32_t wave_front_size;
  uint32_t num_shader_visible_vgprs;
  uint32_t num_cu_per_sh;
  uint32_t num_tcc_blocks;
  uint32_t gs_vgt_table_depth;
  uint32_t gs_prim_buffer_depth;
  uint32_t max_gs_waves_per_vgt;
  uint32_t pcie_num_lanes;
  uint32_t cu_ao_bitmap[4][4];
  uint64_t high_va_offset;
  uint64_t high_va_max;
  uint32_t pa_sc_tile_steering_override;
  uint64_t tcc_disabled_mask;
};

#define CHECK_FIELD(field)                                                                         \
  static_assert(offsetof(LocalDevInfo, field) == offsetof(drm_amdgpu_info_device, field),          \
                "offset mismatch for " #field)

CHECK_FIELD(device_id);
CHECK_FIELD(family);
CHECK_FIELD(num_shader_engines);
CHECK_FIELD(num_shader_arrays_per_engine);
CHECK_FIELD(vram_type);
CHECK_FIELD(vram_bit_width);
CHECK_FIELD(wave_front_size);
CHECK_FIELD(num_cu_per_sh);
CHECK_FIELD(cu_active_number);
CHECK_FIELD(cu_ao_bitmap);
CHECK_FIELD(high_va_offset);
CHECK_FIELD(high_va_max);
CHECK_FIELD(pa_sc_tile_steering_override);
CHECK_FIELD(tcc_disabled_mask);

#undef CHECK_FIELD

TEST(DrmInfoLayoutTest, VramTypeHbmMatchesKernelDefine) { EXPECT_EQ(6u, AMDGPU_VRAM_TYPE_HBM); }

TEST(DrmInfoLayoutTest, LocalStructSizeDoesNotExceedKernel) {
  EXPECT_LE(sizeof(LocalDevInfo), sizeof(drm_amdgpu_info_device));
}

} // namespace
