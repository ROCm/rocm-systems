/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef ABCE_TOPOLOGY_H_
#define ABCE_TOPOLOGY_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "abce_config.h"

// Whether the KFD sysfs topology reader below is available. It walks
// /sys/class/kfd/kfd/topology/nodes with opendir/readdir, which only exists on
// Linux; abce_topology_data_t itself is platform-neutral, so a client on
// another OS populates it from its own source (see abce_sdma_policy_set_topology).
#ifndef ABCE_HAS_KFD_TOPOLOGY
#if defined(__linux__)
#define ABCE_HAS_KFD_TOPOLOGY 1
#else
#define ABCE_HAS_KFD_TOPOLOGY 0
#endif
#endif  // ABCE_HAS_KFD_TOPOLOGY

#if ABCE_HAS_KFD_TOPOLOGY
#include <dirent.h>
#include <stdio.h>
#include <string.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// abce_topology_data_t
//===----------------------------------------------------------------------===//

// Source/destination topology data consumed by engine policies.
typedef struct abce_topology_data_t {
  int xgmi_physical_id[ABCE_MAX_TOPOLOGY_DEVICES];
  uint64_t hive_id[ABCE_MAX_TOPOLOGY_DEVICES];
  uint64_t recommended_mask[ABCE_MAX_TOPOLOGY_DEVICES][ABCE_MAX_TOPOLOGY_DEVICES];
} abce_topology_data_t;

// Zero-initializing this struct is NOT the default state: an unknown
// xgmi_physical_id is -1. Always initialize through here, which also serves as
// the reset used before reloading topology.
static inline void abce_topology_data_initialize(abce_topology_data_t* out_topology) {
  for (uint32_t device_id = 0; device_id < ABCE_MAX_TOPOLOGY_DEVICES; ++device_id) {
    out_topology->xgmi_physical_id[device_id] = -1;
    out_topology->hive_id[device_id] = 0;
    for (uint32_t peer_id = 0; peer_id < ABCE_MAX_TOPOLOGY_DEVICES; ++peer_id)
      out_topology->recommended_mask[device_id][peer_id] = 0;
  }
}

static inline void abce_topology_data_set_device(abce_topology_data_t* topology,
                                                 uint32_t device_id, int xgmi_physical_id,
                                                 uint64_t hive_id) {
  if (device_id >= ABCE_MAX_TOPOLOGY_DEVICES) return;
  topology->xgmi_physical_id[device_id] = xgmi_physical_id;
  topology->hive_id[device_id] = hive_id;
}

static inline void abce_topology_data_set_recommended_mask(abce_topology_data_t* topology,
                                                           uint32_t source_device_id,
                                                           uint32_t destination_device_id,
                                                           uint64_t mask) {
  if (source_device_id >= ABCE_MAX_TOPOLOGY_DEVICES ||
      destination_device_id >= ABCE_MAX_TOPOLOGY_DEVICES)
    return;
  topology->recommended_mask[source_device_id][destination_device_id] = mask;
}

static inline int abce_topology_data_xgmi_physical_id(const abce_topology_data_t* topology,
                                                      uint32_t device_id) {
  return device_id < ABCE_MAX_TOPOLOGY_DEVICES ? topology->xgmi_physical_id[device_id] : -1;
}

static inline uint64_t abce_topology_data_hive_id(const abce_topology_data_t* topology,
                                                  uint32_t device_id) {
  return device_id < ABCE_MAX_TOPOLOGY_DEVICES ? topology->hive_id[device_id] : 0;
}

static inline uint64_t abce_topology_data_recommended_mask(const abce_topology_data_t* topology,
                                                           uint32_t source_device_id,
                                                           uint32_t destination_device_id) {
  return source_device_id < ABCE_MAX_TOPOLOGY_DEVICES &&
                 destination_device_id < ABCE_MAX_TOPOLOGY_DEVICES
             ? topology->recommended_mask[source_device_id][destination_device_id]
             : 0;
}

//===----------------------------------------------------------------------===//
// abce_topology_load_result_t
//===----------------------------------------------------------------------===//

typedef struct abce_topology_load_result_t {
  bool found_gpu;
  uint8_t gfx_minor;
  uint8_t total_sdma;

  // KFD's num_sdma_engines: how many of the SDMA engines are *not* xGMI. The
  // driver numbers the non-xGMI engines first, so hardware ids at or above this
  // are the xGMI engines (MI300X reports 2 and 14, and its CPU io_link
  // recommends engine mask 0x3, confirming engines 0-1 are the non-xGMI pair).
  // This is the xGMI band, derived rather than declared by the client.
  uint8_t num_non_xgmi_sdma;
} abce_topology_load_result_t;

#if ABCE_HAS_KFD_TOPOLOGY

//===----------------------------------------------------------------------===//
// Linux/KFD sysfs reader
//===----------------------------------------------------------------------===//

#define ABCE_KFD_TOPOLOGY_DEFAULT_PATH "/sys/class/kfd/kfd/topology/nodes"

// Every sysfs path this reader builds is well under this; snprintf truncation
// is treated as "skip this entry".
#define ABCE_TOPOLOGY_PATH_MAX 512

// Longest sysfs property key this reader cares about, plus room to spare. The
// fscanf width below MUST stay one less than this.
#define ABCE_TOPOLOGY_KEY_MAX 64

static inline uint32_t abce_topology_parse_node_id(const char* name) {
  if (!name || name[0] < '0' || name[0] > '9') return UINT32_MAX;
  uint32_t value = 0;
  for (const char* cursor = name; *cursor; ++cursor) {
    if (*cursor < '0' || *cursor > '9') return UINT32_MAX;
    value = value * 10u + (uint32_t)(*cursor - '0');
  }
  return value;
}

static inline int abce_topology_read_xgmi_physical_id(const char* pci_bdf) {
  char path[ABCE_TOPOLOGY_PATH_MAX];
  const int written =
      snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/xgmi_physical_id", pci_bdf);
  if (written < 0 || (size_t)written >= sizeof(path)) return -1;
  FILE* file = fopen(path, "r");
  if (!file) return -1;
  int physical_id = -1;
  const bool parsed = fscanf(file, "%d", &physical_id) == 1;
  fclose(file);
  return parsed ? physical_id : -1;
}

static inline void abce_topology_load_links(uint32_t source_node, const char* links_directory,
                                            abce_topology_data_t* topology) {
  DIR* directory = opendir(links_directory);
  if (!directory) return;
  for (struct dirent* entry = readdir(directory); entry; entry = readdir(directory)) {
    if (abce_topology_parse_node_id(entry->d_name) == UINT32_MAX) continue;
    char path[ABCE_TOPOLOGY_PATH_MAX];
    const int written =
        snprintf(path, sizeof(path), "%s/%s/properties", links_directory, entry->d_name);
    if (written < 0 || (size_t)written >= sizeof(path)) continue;
    FILE* properties = fopen(path, "r");
    if (!properties) continue;
    uint64_t destination_node = UINT32_MAX;
    uint64_t recommended_mask = 0;
    char key[ABCE_TOPOLOGY_KEY_MAX];
    unsigned long long value = 0;
    while (fscanf(properties, "%63s %llu", key, &value) == 2) {
      if (strcmp(key, "node_to") == 0)
        destination_node = value;
      else if (strcmp(key, "recommended_sdma_engine_id_mask") == 0)
        recommended_mask = value;
    }
    fclose(properties);
    if (destination_node < ABCE_MAX_TOPOLOGY_DEVICES && recommended_mask)
      abce_topology_data_set_recommended_mask(topology, source_node, (uint32_t)destination_node,
                                              recommended_mask);
  }
  closedir(directory);
}

static inline void abce_topology_load_node(uint32_t node_id, const char* node_directory,
                                           abce_topology_data_t* topology,
                                           abce_topology_load_result_t* result) {
  char path[ABCE_TOPOLOGY_PATH_MAX];
  int written = snprintf(path, sizeof(path), "%s/properties", node_directory);
  if (written < 0 || (size_t)written >= sizeof(path)) return;
  FILE* properties = fopen(path, "r");
  if (!properties) return;
  uint64_t simd_count = 0;
  uint64_t hive_id = 0;
  uint64_t gfx_target_version = 0;
  uint64_t location_id = UINT64_MAX;
  uint64_t domain = 0;
  uint64_t num_sdma_engines = 0;
  uint64_t num_sdma_xgmi_engines = 0;
  char key[ABCE_TOPOLOGY_KEY_MAX];
  unsigned long long value = 0;
  while (fscanf(properties, "%63s %llu", key, &value) == 2) {
    if (strcmp(key, "simd_count") == 0)
      simd_count = value;
    else if (strcmp(key, "hive_id") == 0)
      hive_id = value;
    else if (strcmp(key, "gfx_target_version") == 0)
      gfx_target_version = value;
    else if (strcmp(key, "num_sdma_engines") == 0)
      num_sdma_engines = value;
    else if (strcmp(key, "num_sdma_xgmi_engines") == 0)
      num_sdma_xgmi_engines = value;
    else if (strcmp(key, "location_id") == 0)
      location_id = value;
    else if (strcmp(key, "domain") == 0)
      domain = value;
  }
  fclose(properties);
  if (simd_count == 0) return;

  int physical_id = -1;
  if (location_id != UINT64_MAX) {
    char pci_bdf[32];
    snprintf(pci_bdf, sizeof(pci_bdf), "%04llx:%02llx:%02llx.%01llx",
             (unsigned long long)domain, (unsigned long long)((location_id >> 8) & 0xff),
             (unsigned long long)((location_id >> 3) & 0x1f),
             (unsigned long long)(location_id & 0x7));
    physical_id = abce_topology_read_xgmi_physical_id(pci_bdf);
  }
  abce_topology_data_set_device(topology, node_id, physical_id, hive_id);
  written = snprintf(path, sizeof(path), "%s/io_links", node_directory);
  if (written >= 0 && (size_t)written < sizeof(path))
    abce_topology_load_links(node_id, path, topology);
  written = snprintf(path, sizeof(path), "%s/p2p_links", node_directory);
  if (written >= 0 && (size_t)written < sizeof(path))
    abce_topology_load_links(node_id, path, topology);

  result->found_gpu = true;
  if (result->total_sdma == 0 && gfx_target_version != 0) {
    result->gfx_minor = (uint8_t)((gfx_target_version / 100u) % 100u);
    result->total_sdma = (uint8_t)(num_sdma_engines + num_sdma_xgmi_engines);
    result->num_non_xgmi_sdma = (uint8_t)num_sdma_engines;
  }
}

// Header-only Linux/KFD topology reader. A null |base_path| reads
// ABCE_KFD_TOPOLOGY_DEFAULT_PATH. Parsing into a temporary
// abce_topology_data_t makes policy reconfiguration reset-safe: |out_topology|
// is overwritten only once a GPU has been found.
//
// Returns whether a GPU was found, which is also reported in
// out_result->found_gpu alongside the engine counts.
static inline bool abce_topology_load_from_kfd(const char* base_path,
                                               abce_topology_data_t* out_topology,
                                               abce_topology_load_result_t* out_result) {
  if (!out_topology || !out_result) return false;
  out_result->found_gpu = false;
  out_result->gfx_minor = 0;
  out_result->total_sdma = 0;
  out_result->num_non_xgmi_sdma = 0;

  abce_topology_data_t loaded;
  abce_topology_data_initialize(&loaded);
  const char* root = base_path ? base_path : ABCE_KFD_TOPOLOGY_DEFAULT_PATH;
  DIR* directory = opendir(root);
  if (!directory) return false;
  for (struct dirent* entry = readdir(directory); entry; entry = readdir(directory)) {
    const uint32_t node_id = abce_topology_parse_node_id(entry->d_name);
    if (node_id == UINT32_MAX || node_id >= ABCE_MAX_TOPOLOGY_DEVICES) continue;
    char node_directory[ABCE_TOPOLOGY_PATH_MAX];
    const int written =
        snprintf(node_directory, sizeof(node_directory), "%s/%s", root, entry->d_name);
    if (written < 0 || (size_t)written >= sizeof(node_directory)) continue;
    abce_topology_load_node(node_id, node_directory, &loaded, out_result);
  }
  closedir(directory);
  if (out_result->found_gpu) *out_topology = loaded;
  return out_result->found_gpu;
}

#endif  // ABCE_HAS_KFD_TOPOLOGY

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // ABCE_TOPOLOGY_H_
