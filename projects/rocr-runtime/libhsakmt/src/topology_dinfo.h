/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/*
 * Interface between topology.c and the per-OS device-info ("dinfo")
 * backends that read the KFD topology published by the kernel.
 *
 * topology.c builds the HSA topology (node enumeration, generation-id
 * snapshotting, gfx-version overrides, marketing names, CPU caches, ...)
 * without knowing where the kernel data comes from.  Each backend fetches
 * and parses the raw records:
 *
 *   src/linux/topology_dinfo.c   - /sys/devices/virtual/kfd/kfd/topology
 *
 * Model mode (hsakmt_use_model) is a plain directory of files on every OS,
 * so each backend must honour it as well.
 */

#ifndef TOPOLOGY_DINFO_H_INCLUDED
#define TOPOLOGY_DINFO_H_INCLUDED

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#include "libhsakmt.h"

/* Shared with the backends, which fill in and look up the node-id map. */
typedef struct {
	HsaNodeProperties node;
	HsaMemoryProperties *mem;     /* node->NumBanks elements */
	HsaCacheProperties *cache;
	HsaIoLinkProperties *link;
} node_props_t;

struct hsa_kfd_topology_context
{
	HsaSystemProperties *system_props;
	node_props_t *node_props;

	/* This array caches dinfo-based node IDs of CPU nodes + all supported
	 * GPU nodes. It will be used to map user-node IDs to dinfo-node IDs.
	 */
	uint32_t *map_user_to_sysfs_node_id;
	uint32_t map_user_to_sysfs_node_id_size;

	uint32_t num_sysfs_nodes;
};

/* Helpers defined in topology.c for use by the backends. */
int num_subdirs(char *dirpath, char *prefix);
HSAKMT_STATUS topology_map_sysfs_to_user_node_id(struct hsa_kfd_topology_context *topology_ctx,
						  uint32_t sys_node_id, uint32_t *user_node_id);

/* ---- Implemented by each backend ---- */

/* Root of the live topology (the sysfs directory on Linux), or
 * hsakmt_model_topology when hsakmt_use_model is set. */
const char *get_dinfo_dir(void);

HSAKMT_STATUS topology_dinfo_get_generation(uint32_t *gen);

/* Number of processors to size the CPU info table for, or 0 on failure. */
uint32_t topology_dinfo_get_num_procs(void);

HSAKMT_STATUS topology_dinfo_map_node_id(struct hsa_kfd_topology_context *topology_ctx,
					  uint32_t node_id, uint32_t *sys_node_id);

HSAKMT_STATUS topology_dinfo_get_gpu_id(uint32_t sysfs_node_id, uint32_t *gpu_id);

HSAKMT_STATUS topology_dinfo_check_node_supported(HsaKFDContext *ctx,
						   uint32_t sysfs_node_id,
						   bool *is_node_supported);

HSAKMT_STATUS hsakmt_topology_dinfo_get_system_props(HsaKFDContext *ctx,
						      HsaSystemProperties *props);

/* Fetches the raw node "properties" record for @sys_node_id into @buf as a
 * NUL-terminated "key value\n" text blob.  topology.c parses it, because
 * the gfx-version override and marketing-name logic depend on it. */
HSAKMT_STATUS topology_dinfo_get_node_props_blob(uint32_t sys_node_id,
						  char *buf, size_t buflen);

HSAKMT_STATUS topology_dinfo_get_mem_props(struct hsa_kfd_topology_context *topology_ctx,
					    uint32_t node_id, uint32_t mem_id,
					    HsaMemoryProperties *props);

HSAKMT_STATUS topology_dinfo_get_cache_props(struct hsa_kfd_topology_context *topology_ctx,
					      uint32_t node_id, uint32_t cache_id,
					      HsaCacheProperties *props);

HSAKMT_STATUS topology_dinfo_get_iolink_props(HsaKFDContext *ctx,
					       uint32_t node_id, uint32_t iolink_id,
					       HsaIoLinkProperties *props, bool p2pLink);

/* Whether the GPU behind DRM render node @drm_render_minor supports UALink
 * fabric handles. */
bool topology_dinfo_check_fabric_handle_supported(uint32_t drm_render_minor);

#endif /* TOPOLOGY_DINFO_H_INCLUDED */
