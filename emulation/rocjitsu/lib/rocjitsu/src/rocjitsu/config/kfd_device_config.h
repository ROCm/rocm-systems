// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file kfd_device_config.h
/// @brief KFD device identity shared by VM simulation and DBT guest discovery.

#ifndef ROCJITSU_CONFIG_KFD_DEVICE_CONFIG_H_
#define ROCJITSU_CONFIG_KFD_DEVICE_CONFIG_H_

#include <cstdint>
#include <string>

namespace rocjitsu::config {

/// @brief KFD device identity extracted from vm.gpu.device or dbt_guest.guest_device.
///
/// @details The KMD interposer copies these values into generated KFD topology
/// files and synthetic DRM/AMDGPU metadata. VM simulation uses them for the
/// simulated GPU. DBT guest mode uses them for the guest GPU that applications
/// see, not the host GPU that executes translated code.
struct KfdDeviceConfig {
  uint32_t gpu_id = 0;              ///< KFD gpu_id advertised in topology and ioctls.
  uint32_t gfx_target_version = 0;  ///< KFD gfx target version, for example 90500 for gfx950.
  uint32_t vendor_id = 0x1002;      ///< PCI vendor id used by synthetic DRM metadata.
  uint32_t device_id = 0;           ///< PCI device id used by synthetic DRM metadata.
  uint32_t family_id = 0;           ///< AMDGPU family id reported for this device.
  uint64_t unique_id = 0;           ///< Stable synthetic unique id for this node.
  std::string marketing_name;       ///< User-visible GPU name, for example MI350X.
  uint32_t drm_render_minor = 128;  ///< Preferred synthetic /dev/dri/renderD minor.
  uint32_t revision_id = 0;         ///< AMDGPU ASIC revision id.
  uint32_t pci_revision_id = 0;     ///< PCI config-space revision id.
  uint32_t simd_count = 0;          ///< Total SIMD units exposed in KFD properties.
  uint32_t max_waves_per_simd = 10; ///< Maximum waves per SIMD.
  uint32_t num_shader_engines = 0;  ///< KFD array_count: total shader arrays.
  uint32_t num_shader_arrays_per_engine = 1; ///< Shader arrays per shader engine.
  uint32_t num_cu_per_sh = 0;                ///< Compute units per shader array.
  uint32_t simd_per_cu = 4;                  ///< SIMD units per compute unit.
  uint32_t wave_front_size = 64;             ///< Wavefront width reported to runtimes.
  uint32_t max_slots_scratch_cu = 32;        ///< Scratch slots per compute unit.
  uint64_t local_mem_size = 0;               ///< Advertised local memory size.
  uint32_t vram_type = 6;                    ///< AMDGPU_VRAM_TYPE_* (6 = HBM).
  uint32_t lds_size_kb = 64;                 ///< LDS size per CU in KiB.
  uint32_t mem_width = 4096;                 ///< Memory interface width in bits.
  uint32_t mem_clk_max = 1200;               ///< Maximum memory clock in MHz.
  uint32_t l1_size_kb = 32;                  ///< L1 cache size in KiB.
  uint32_t l1_line_size = 128;               ///< L1 cache line size in bytes.
  uint32_t l1_assoc = 4;                     ///< L1 cache associativity.
  uint32_t l2_size_kb = 4096;                ///< L2 cache size in KiB.
  uint32_t l2_line_size = 128;               ///< L2 cache line size in bytes.
  uint32_t l2_assoc = 16;                    ///< L2 cache associativity.
  uint32_t num_sdma_engines = 2;             ///< SDMA engine count.
  uint32_t num_sdma_xgmi_engines = 0;        ///< XGMI SDMA engine count.
  uint32_t num_cp_queues = 128;              ///< Hardware queue count.
  uint32_t max_engine_clk_fcompute = 2100;   ///< Maximum compute clock in MHz.
  uint32_t location_id = 0x0300;             ///< PCI BDF location id.
  uint64_t hive_id = 0;                      ///< XGMI hive id.
  uint32_t domain = 0;                       ///< PCI domain.
  uint32_t capability = 0;                   ///< KFD debug capability bits, or 0 to derive.
  uint32_t capability2 = 0;                  ///< KFD debug capability2 bits, or 0 to derive.
  uint64_t debug_prop = 0;                   ///< KFD debug_prop bits, or 0 to derive.
  bool present = false;                      ///< True if device section existed in config.
};

} // namespace rocjitsu::config

#endif // ROCJITSU_CONFIG_KFD_DEVICE_CONFIG_H_
