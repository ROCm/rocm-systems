// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/rj_vm_config.h"

#include "embedded_schema.h"
#include "rocjitsu/config/config_loader.h"
#include "rocjitsu/vm/gpu_info.h"
#include "util/diagnostic.h"
#include "util/log.h"

#include <algorithm>
#include <memory>
#include <new>
#include <string>
#include <utility>
#include <vector>

struct rj_vm_config_t {
  std::string json;                      ///< Owned input snapshot used for every VM construction.
  std::string local_node_id;             ///< Resolved node selected when the snapshot was parsed.
  std::vector<rj_vm_gpu_info_t> devices; ///< Immutable metadata in node-local GPU ordinal order.
};

rj_status_t rj_vm_config_create_from_string(const char *json, rj_vm_config_t **config) {
  return rj_vm_config_create_from_string_for_node(json, nullptr, config);
}

rj_status_t rj_vm_config_create_from_string_for_node(const char *json, const char *node_id,
                                                     rj_vm_config_t **config) {
  if (!json || !config)
    return ROCJITSU_STATUS_INVALID_ARGUMENT;
  try {
    auto owned = std::make_unique<rj_vm_config_t>();
    owned->json = json;
    util::StringDiagnostic diagnostic;
    auto identity = rocjitsu::config::load_node_device_identity_from_string(
        owned->json, rocjitsu::kEmbeddedSchema, node_id ? node_id : "", diagnostic.emitter());
    if (identity.failed()) {
      util::Logger::warn("rj_vm_config_create_from_string_for_node failed: ", diagnostic.message());
      return ROCJITSU_STATUS_INVALID_ARGUMENT;
    }
    auto &node = identity.value();
    if (node.devices.empty() || node.num_xcc == 0 ||
        std::ranges::any_of(node.devices, [](const auto &device) { return !device.present; }))
      return ROCJITSU_STATUS_UNSUPPORTED;
    owned->local_node_id = std::move(node.local_node_id);
    owned->devices.reserve(node.devices.size());
    for (const auto &device : node.devices)
      owned->devices.push_back(rocjitsu::gpu_info_to_c(
          rocjitsu::gpu_info_from_config(device, node.num_xcc, node.pci, node.publish_peer_links)));
    *config = owned.release();
    return ROCJITSU_STATUS_SUCCESS;
  } catch (const std::bad_alloc &) {
    return ROCJITSU_STATUS_OUT_OF_RESOURCES;
  } catch (const std::exception &error) {
    util::Logger::warn("rj_vm_config_create_from_string_for_node failed: ", error.what());
    return ROCJITSU_STATUS_INVALID_ARGUMENT;
  }
}

void rj_vm_config_destroy(rj_vm_config_t *config) { delete config; }

rj_status_t rj_vm_config_gpu_count(const rj_vm_config_t *config, uint32_t *count) {
  if (!config || !count)
    return ROCJITSU_STATUS_INVALID_ARGUMENT;
  *count = static_cast<uint32_t>(config->devices.size());
  return ROCJITSU_STATUS_SUCCESS;
}

rj_status_t rj_vm_config_gpu_info_at(const rj_vm_config_t *config, uint32_t ordinal,
                                     rj_vm_gpu_info_t *info) {
  if (!config || !info || ordinal >= config->devices.size())
    return ROCJITSU_STATUS_INVALID_ARGUMENT;
  *info = config->devices[ordinal];
  return ROCJITSU_STATUS_SUCCESS;
}

rj_status_t rj_vm_create_from_config(const rj_vm_config_t *config, rj_vm_mode_t mode,
                                     rj_vm_t **vm) {
  if (!config || !vm)
    return ROCJITSU_STATUS_INVALID_ARGUMENT;
  return rj_vm_create_from_string_for_node(config->json.c_str(), config->local_node_id.c_str(),
                                           mode, vm);
}
