// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file config_common.h
/// @brief Shared FlatBuffers config parsing helpers.

#ifndef ROCJITSU_CONFIG_CONFIG_COMMON_H_
#define ROCJITSU_CONFIG_CONFIG_COMMON_H_

#include "rocjitsu/config/kfd_device_config.h"

#include "flatbuffers/idl.h"
#include "simulation_config_generated.h"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace rocjitsu {
namespace config {

inline std::string read_config_file(const std::string &path) {
  std::ifstream file(path);
  if (!file.is_open())
    throw std::runtime_error("Cannot open file: " + path);

  std::ostringstream contents;
  contents << file.rdbuf();
  return contents.str();
}

/// @brief Parse simulation-config JSON and consume the FlatBuffer while parser storage is alive.
///
/// @details FlatBuffers returns pointers into `Parser::builder_`. Keeping the
/// parser local to this helper makes the lifetime explicit: callers must copy
/// any values they need inside @p callback rather than retaining raw pointers.
template <typename Callback>
decltype(auto) with_parsed_simulation_config_json(const std::string &json,
                                                  const std::string &schema_text,
                                                  Callback &&callback) {
  flatbuffers::Parser parser;
  parser.opts.skip_unexpected_fields_in_json = true;
  if (!parser.Parse(schema_text.c_str()))
    throw std::runtime_error("Failed to parse schema: " + std::string(parser.error_));
  if (!parser.Parse(json.c_str()))
    throw std::runtime_error("Failed to parse JSON config: " + std::string(parser.error_));
  const fb::SimulationConfig *config =
      flatbuffers::GetRoot<fb::SimulationConfig>(parser.builder_.GetBufferPointer());
  return std::forward<Callback>(callback)(config);
}

/// @brief Convert a FlatBuffers KFD identity table into the runtime config form.
///
/// @details This copies only scalar/string topology values. The resulting
/// config owns its marketing-name string so it is safe after the FlatBuffers
/// parser storage goes out of scope.
inline KfdDeviceConfig kfd_device_from_fb(const fb::KfdDeviceInfo *device) {
  KfdDeviceConfig config;
  if (device == nullptr)
    return config;

  config.present = true;
  config.gpu_id = device->gpu_id();
  config.gfx_target_version = device->gfx_target_version();
  config.vendor_id = device->vendor_id();
  config.device_id = device->device_id();
  config.family_id = device->family_id();
  config.unique_id = device->unique_id();
  if (device->marketing_name())
    config.marketing_name = device->marketing_name()->str();
  config.drm_render_minor = device->drm_render_minor();
  config.revision_id = device->revision_id();
  config.pci_revision_id = device->pci_revision_id();
  config.simd_count = device->simd_count();
  config.max_waves_per_simd = device->max_waves_per_simd();
  config.num_shader_engines = device->num_shader_engines();
  config.num_shader_arrays_per_engine = device->num_shader_arrays_per_engine();
  config.num_cu_per_sh = device->num_cu_per_sh();
  config.simd_per_cu = device->simd_per_cu();
  config.wave_front_size = device->wave_front_size();
  config.max_slots_scratch_cu = device->max_slots_scratch_cu();
  config.local_mem_size = device->local_mem_size();
  config.vram_type = device->vram_type();
  config.lds_size_kb = device->lds_size_kb();
  config.mem_width = device->mem_width();
  config.mem_clk_max = device->mem_clk_max();
  config.l1_size_kb = device->l1_size_kb();
  config.l1_line_size = device->l1_line_size();
  config.l1_assoc = device->l1_assoc();
  config.l2_size_kb = device->l2_size_kb();
  config.l2_line_size = device->l2_line_size();
  config.l2_assoc = device->l2_assoc();
  config.num_sdma_engines = device->num_sdma_engines();
  config.num_sdma_xgmi_engines = device->num_sdma_xgmi_engines();
  config.num_cp_queues = device->num_cp_queues();
  config.max_engine_clk_fcompute = device->max_engine_clk_fcompute();
  config.location_id = device->location_id();
  config.hive_id = device->hive_id();
  config.domain = device->domain();
  config.capability = device->capability();
  config.capability2 = device->capability2();
  config.debug_prop = device->debug_prop();
  return config;
}

} // namespace config
} // namespace rocjitsu

#endif // ROCJITSU_CONFIG_CONFIG_COMMON_H_
