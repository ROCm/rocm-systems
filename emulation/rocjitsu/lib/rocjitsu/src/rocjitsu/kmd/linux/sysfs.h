// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file sysfs.h
/// @brief Generates a sysfs-compatible KFD topology directory for ROCR discovery.

#ifndef ROCJITSU_KMD_LINUX_SYSFS_H_
#define ROCJITSU_KMD_LINUX_SYSFS_H_

#include <cstdint>
#include <string>

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
    const char *marketing_name = "";

    // Compute unit organization
    uint32_t simd_count = 0;
    uint32_t max_waves_per_simd = 10;
    uint32_t num_shader_engines = 0;
    uint32_t num_shader_arrays_per_engine = 1;
    uint32_t num_cu_per_sh = 0;
    uint32_t simd_per_cu = 4;
    uint32_t wave_front_size = 64;
    uint32_t num_xcc = 1;
    uint32_t max_slots_scratch_cu = 32;

    // Memory
    uint64_t local_mem_size = 0;
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
    uint32_t num_sdma_engines = 2;
    uint32_t num_sdma_xgmi_engines = 0;
    uint32_t num_cp_queues = 128;
    uint32_t max_engine_clk_fcompute = 2100; // MHz

    // Capability flags
    uint32_t capability = 0; // 0 = auto-compute from defaults
    uint32_t capability2 = 0;
    uint64_t debug_prop = 0;

    // Firmware
    uint32_t fw_version = 0;
    uint32_t sdma_fw_version = 0;
  };

  Sysfs() = default;
  ~Sysfs();

  Sysfs(const Sysfs &) = delete;
  Sysfs &operator=(const Sysfs &) = delete;
  Sysfs(Sysfs &&other) noexcept;
  Sysfs &operator=(Sysfs &&other) noexcept;

  /// @brief Generate the sysfs topology directory.
  /// @param gpu GPU configuration to represent.
  /// @returns Path to the generated directory.
  std::string generate(const GpuInfo &gpu);

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
  void write_system_properties();
  void write_cpu_node(const std::string &nodes_dir);
  void write_gpu_node(const std::string &nodes_dir, const GpuInfo &gpu);
  void write_drm_tree(const GpuInfo &gpu);
};

} // namespace rocjitsu

#endif // ROCJITSU_KMD_LINUX_SYSFS_H_
