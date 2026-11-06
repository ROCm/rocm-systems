// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_VM_GPU_INFO_H_
#define ROCJITSU_VM_GPU_INFO_H_

#include "rocjitsu/kmd/linux/sysfs.h"
#include "rocjitsu/vm/rj_vm.h"

#include <cstring>

namespace rocjitsu {

inline rj_vm_gpu_info_t gpu_info_to_c(const Sysfs::GpuInfo &gpu) {
  rj_vm_gpu_info_t info{};
  info.present = 1;
  info.gpu_id = gpu.gpu_id;
  info.gfx_target_version = gpu.gfx_target_version;
  info.vendor_id = gpu.vendor_id;
  info.device_id = gpu.device_id;
  info.family_id = gpu.family_id;
  info.unique_id = gpu.unique_id;
  info.location_id = gpu.location_id;
  info.domain = gpu.domain;
  info.hive_id = gpu.hive_id;
  info.drm_render_minor = gpu.drm_render_minor;
  info.revision_id = gpu.revision_id;
  info.pci_revision_id = gpu.pci_revision_id;
  info.simd_count = gpu.simd_count;
  info.max_waves_per_simd = gpu.max_waves_per_simd;
  info.num_shader_engines = gpu.num_shader_engines;
  info.num_shader_arrays_per_engine = gpu.num_shader_arrays_per_engine;
  info.num_cu_per_sh = gpu.num_cu_per_sh;
  info.simd_per_cu = gpu.simd_per_cu;
  info.wave_front_size = gpu.wave_front_size;
  info.num_xcc = gpu.num_xcc;
  info.max_slots_scratch_cu = gpu.max_slots_scratch_cu;
  info.local_mem_size = gpu.local_mem_size;
  info.visible_vram_size = gpu.visible_vram_size;
  info.control_stack_size = gpu.control_stack_size;
  info.context_save_size = gpu.context_save_size;
  info.vram_type = gpu.vram_type;
  info.lds_size_kb = gpu.lds_size_kb;
  info.mem_width = gpu.mem_width;
  info.mem_clk_max = gpu.mem_clk_max;
  info.l1_size_kb = gpu.l1_size_kb;
  info.l1_line_size = gpu.l1_line_size;
  info.l1_assoc = gpu.l1_assoc;
  info.l2_size_kb = gpu.l2_size_kb;
  info.l2_line_size = gpu.l2_line_size;
  info.l2_assoc = gpu.l2_assoc;
  info.num_sdma_engines = gpu.num_sdma_engines;
  info.num_sdma_xgmi_engines = gpu.num_sdma_xgmi_engines;
  info.num_cp_queues = gpu.num_cp_queues;
  info.max_engine_clk_fcompute = gpu.max_engine_clk_fcompute;
  info.capability = gpu.capability;
  info.capability2 = gpu.capability2;
  info.debug_prop = gpu.debug_prop;
  info.fw_version = gpu.fw_version;
  info.sdma_fw_version = gpu.sdma_fw_version;

  size_t name_len = gpu.marketing_name.size();
  if (name_len >= sizeof(info.marketing_name))
    name_len = sizeof(info.marketing_name) - 1;
  std::memcpy(info.marketing_name, gpu.marketing_name.data(), name_len);
  info.marketing_name[name_len] = '\0';
  return info;
}

} // namespace rocjitsu

#endif // ROCJITSU_VM_GPU_INFO_H_
