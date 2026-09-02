// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file sysfs.h
/// @brief Generates a sysfs-compatible KFD topology directory for ROCR discovery.

#ifndef ROCJITSU_KMD_LINUX_SYSFS_H_
#define ROCJITSU_KMD_LINUX_SYSFS_H_

#include "rocjitsu/config/kfd_device_config.h"

#include <cstdint>
#include <string>
#include <vector>

namespace rocjitsu {

/// @brief Generates a sysfs-compatible KFD topology directory for ROCR discovery.
///
/// @details ROCR's libhsakmt reads GPU topology from
/// /sys/devices/virtual/kfd/kfd/topology/. This class generates a compatible
/// directory structure with properties matching the simulated GPU configuration.
/// The LD_PRELOAD interposer redirects sysfs reads to the generated directory
/// without setting HSA_MODEL_TOPOLOGY (which would trigger model mode and
/// require HSA_MODEL_LIB).
class Sysfs {
public:
  /// @brief GPU configuration for sysfs topology generation.
  struct GpuInfo {
    // Identification
    uint32_t gpu_id = 0;
    uint32_t gfx_target_version = 0;
    uint32_t vendor_id = 0x1002; // AMD
    uint32_t device_id = 0;
    uint32_t family_id = 0;
    uint64_t unique_id = 0;
    uint32_t location_id = 0x0300; // PCI BDF: bus 3, dev 0, func 0
    uint32_t domain = 0;
    uint64_t hive_id = 0;
    uint32_t drm_render_minor = 128;
    uint32_t revision_id = 0;
    uint32_t pci_revision_id = 0;
    std::string marketing_name;

    // Compute unit organization
    uint32_t simd_count = 0;
    uint32_t max_waves_per_simd = 10;
    uint32_t num_shader_engines = 0; ///< Shader engines per XCC, matching the
                                     ///< simulated SoC's se[] count. KFD's
                                     ///< array_count is derived, not this.
    uint32_t num_shader_arrays_per_engine = 1;
    uint32_t num_cu_per_sh = 0;
    uint32_t simd_per_cu = 4;
    uint32_t wave_front_size = 64;
    uint32_t num_xcc = 1;
    uint32_t max_slots_scratch_cu = 32;

    // Memory
    uint64_t local_mem_size = 0;
    uint32_t vram_type = 6;
    uint32_t lds_size_kb = 64;
    uint32_t mem_width = 4096;   // HBM interface width in bits
    uint32_t mem_clk_max = 1200; // MHz

    // Caches
    uint32_t l1_size_kb = 32;
    uint32_t l1_line_size = 128;
    uint32_t l1_assoc = 4;
    uint32_t l2_size_kb = 4096;
    uint32_t l2_line_size = 128;
    uint32_t l2_assoc = 16;

    // Engines and queues
    uint32_t num_sdma_engines = 0;
    uint32_t num_sdma_xgmi_engines = 0;
    uint32_t num_sdma_queues_per_engine = 0;
    uint32_t num_cp_queues = 128;
    uint32_t max_engine_clk_fcompute = 2100; // MHz

    // Capability flags
    uint32_t capability = 0; // 0 = auto-compute from defaults
    uint32_t capability2 = 0;
    uint64_t debug_prop = 0;

    // Firmware
    uint32_t fw_version = 0;
    uint32_t sdma_fw_version = 0;

    /// @brief XCC count with a floor of one, as every KFD consumer needs it.
    /// @details A node reporting "num_xcc 0" alongside a scaled array_count
    /// makes rocdbgapi's array_count * num_xcc / simd_arrays_per_engine come out
    /// zero, so both the sysfs generator and the DBG_TRAP device snapshot
    /// normalize through here rather than each applying its own floor.
    uint32_t effective_num_xcc() const { return num_xcc == 0 ? 1u : num_xcc; }

    /// @brief Shader arrays per engine with a floor of one.
    /// @details This is the divisor libhsakmt and rocdbgapi apply to
    /// array_count to recover the shader-engine count, so a node that publishes
    /// a non-zero array_count next to "simd_arrays_per_engine 0" makes them
    /// divide by zero. array_count_per_xcc() already floors it on the dividend
    /// side; every publisher of the divisor goes through here so the two halves
    /// of the quotient cannot be normalized differently.
    uint32_t effective_arrays_per_engine() const {
      return num_shader_arrays_per_engine == 0 ? 1u : num_shader_arrays_per_engine;
    }

    /// @brief Shader arrays per XCC -- KFD's node_props.array_count.
    ///
    /// @details The driver reports shader *arrays*, not engines: libhsakmt
    /// recovers NumShaderBanks as array_count / simd_arrays_per_engine and
    /// rocdbgapi recovers the engine count as
    /// array_count * num_xcc / simd_arrays_per_engine, so both invert this
    /// product to get num_shader_engines back. Deriving the array count from
    /// the configured geometry keeps these representations consistent.
    uint32_t array_count_per_xcc() const {
      return num_shader_engines * effective_arrays_per_engine();
    }
  };

  Sysfs() = default;
  ~Sysfs();

  Sysfs(const Sysfs &) = delete;
  Sysfs &operator=(const Sysfs &) = delete;
  Sysfs(Sysfs &&other) noexcept;
  Sysfs &operator=(Sysfs &&other) noexcept;

  /// @brief Generate the sysfs topology directory for one or more GPUs.
  /// @param gpu GPU configuration to represent (single GPU).
  /// @returns Path to the generated directory.
  std::string generate(const GpuInfo &gpu);

  /// @brief Generate the sysfs topology directory for multiple GPUs.
  /// @param gpus Per-GPU configurations. Each gets its own topology node.
  std::string generate(const std::vector<GpuInfo> &gpus);

  /// @brief Get the generated KFD topology path (empty if not yet generated).
  const std::string &path() const { return topology_dir_; }

  /// @brief Get the generated DRM sysfs path (empty if not yet generated).
  const std::string &drm_path() const { return drm_dir_; }

  /// @brief Get the GPU info used to generate the topology.
  const GpuInfo &gpu_info() const { return gpu_info_; }

  /// @brief Reserved for future environment setup (currently a no-op).
  void setup_environment();

  /// @brief Remove the generated directories.
  void cleanup();

private:
  std::string topology_dir_;
  std::string drm_dir_;
  GpuInfo gpu_info_{};

  void write_file(const std::string &path, const std::string &content);
  void make_dir(const std::string &path);
  void write_generation_id();
  void write_system_properties(uint32_t num_devices);
  void write_cpu_node(const std::string &nodes_dir, uint32_t num_gpu_links);
  void write_gpu_node(const std::string &nodes_dir, uint32_t node_idx, const GpuInfo &gpu,
                      uint32_t total_gpus);
  void write_drm_tree(const std::vector<GpuInfo> &gpus);
};

/// @brief Convert a parsed KFD device config into generated sysfs GPU metadata.
Sysfs::GpuInfo gpu_info_from_config(const config::KfdDeviceConfig &dev, uint32_t num_xcc);

} // namespace rocjitsu

#endif // ROCJITSU_KMD_LINUX_SYSFS_H_
