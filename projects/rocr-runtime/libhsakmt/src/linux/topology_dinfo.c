/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/*
 * Linux device-info ("dinfo") backend for the KFD topology: reads the
 * sysfs tree under /sys/devices/virtual/kfd/kfd/topology.  See
 * topology_dinfo.h for the interface.
 */

#include <assert.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/sysinfo.h>

#include "fmm.h"
#include "hsakmt/hsakmtmodel.h"
#include "topology_dinfo.h"

#define KFD_DINFO_PATH "/sys/devices/virtual/kfd/kfd/topology"
#define KFD_DINFO_GENERATION_ID "%s/generation_id"
#define KFD_DINFO_SYSTEM_PROPERTIES "%s/system_properties"
#define KFD_DINFO_NODES "%s/nodes"

const char *get_dinfo_dir(void)
{
	if (hsakmt_use_model)
		return hsakmt_model_topology;
	return KFD_DINFO_PATH;
}

uint32_t topology_dinfo_get_num_procs(void)
{
	return get_nprocs();
}

HSAKMT_STATUS topology_dinfo_get_generation(uint32_t *gen)
{
	FILE *fd;
	HSAKMT_STATUS ret = HSAKMT_STATUS_SUCCESS;
	char path[256];

	snprintf(path, sizeof(path), KFD_DINFO_GENERATION_ID, get_dinfo_dir());

	assert(gen);
	fd = fopen(path, "r");
	if (!fd)
		return HSAKMT_STATUS_ERROR;
	if (fscanf(fd, "%ul", gen) != 1)
		ret = HSAKMT_STATUS_ERROR;

	fclose(fd);
	return ret;
}

HSAKMT_STATUS topology_dinfo_map_node_id(
					struct hsa_kfd_topology_context *topology_ctx,
					uint32_t node_id, uint32_t *sys_node_id)
{
	if ((!topology_ctx->map_user_to_sysfs_node_id) ||
		(node_id >= topology_ctx->map_user_to_sysfs_node_id_size))
		return HSAKMT_STATUS_NOT_SUPPORTED;

	*sys_node_id = topology_ctx->map_user_to_sysfs_node_id[node_id];
	return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS topology_dinfo_get_gpu_id(uint32_t sysfs_node_id, uint32_t *gpu_id)
{
	FILE *fd;
	char path[256];
	HSAKMT_STATUS ret = HSAKMT_STATUS_SUCCESS;

	assert(gpu_id);
	snprintf(path, sizeof(path), KFD_DINFO_NODES "/%d/gpu_id", get_dinfo_dir(), sysfs_node_id);
	fd = fopen(path, "r");
	if (!fd)
		return HSAKMT_STATUS_ERROR;
	if (fscanf(fd, "%ul", gpu_id) != 1)
		ret = (errno == EPERM) ? HSAKMT_STATUS_NOT_SUPPORTED :
					 HSAKMT_STATUS_ERROR;
	fclose(fd);

	return ret;
}

/* Check if the @sysfs_node_id is supported. This function will be passed with
 * a dinfo node id. A dinfo node is not supported
 *	- if corresponding drm render node is not available.
 *	- if node information is not accessible (EPERM)
 */
HSAKMT_STATUS topology_dinfo_check_node_supported(HsaKFDContext *ctx,
					uint32_t sysfs_node_id, bool *is_node_supported)
{
	uint32_t gpu_id;
	FILE *fd;
	char *read_buf, *p;
	int read_size;
	char prop_name[256];
	char path[256];
	unsigned long long prop_val;
	uint32_t prog;
	uint32_t drm_render_minor = 0;
	int ret_value;
	HSAKMT_STATUS ret = HSAKMT_STATUS_SUCCESS;

	*is_node_supported = false;

	/* Retrieve the GPU ID */
	ret = topology_dinfo_get_gpu_id(sysfs_node_id, &gpu_id);
	if (ret == HSAKMT_STATUS_NOT_SUPPORTED)
		return HSAKMT_STATUS_SUCCESS;
	if (ret != HSAKMT_STATUS_SUCCESS)
		return ret;

	if (gpu_id == 0) {
		*is_node_supported = true;
		return HSAKMT_STATUS_SUCCESS;
	}

	read_buf = malloc(PAGE_SIZE);
	if (!read_buf)
		return HSAKMT_STATUS_NO_MEMORY;

	/* Retrieve the node properties */
	snprintf(path, 256, KFD_DINFO_NODES "/%d/properties", get_dinfo_dir(), sysfs_node_id);
	fd = fopen(path, "r");
	if (!fd) {
		free(read_buf);
		return HSAKMT_STATUS_ERROR;
	}

	read_size = fread(read_buf, 1, PAGE_SIZE, fd);
	if (read_size <= 0) {
		ret = HSAKMT_STATUS_ERROR;
		goto err;
	}

	/* Since we're using the buffer as a string, we make sure the string terminates */
	if (read_size >= PAGE_SIZE)
		read_size = PAGE_SIZE - 1;
	read_buf[read_size] = 0;

	/* Read the node properties */
	prog = 0;
	p = read_buf;
	while (sscanf(p += prog, "%255s %llu\n%n", prop_name, &prop_val, &prog) == 2) {
		if (strcmp(prop_name, "drm_render_minor") == 0) {
			drm_render_minor = (int32_t)prop_val;
			break;
		}
	}
	if (!drm_render_minor) {
		ret = HSAKMT_STATUS_ERROR;
		goto err;
	}

	/* Open DRM Render device */
	ret_value = hsakmt_open_drm_render_device(ctx, drm_render_minor);
	if (ret_value > 0)
		*is_node_supported = true;
	else if (ret_value != -ENOENT && ret_value != -EPERM)
		ret = HSAKMT_STATUS_ERROR;

err:
	free(read_buf);
	fclose(fd);
	return ret;
}

HSAKMT_STATUS hsakmt_topology_dinfo_get_system_props(HsaKFDContext *ctx,
					HsaSystemProperties *props)
{
	FILE *fd;
	char *read_buf, *p;
	char path[256];
	char prop_name[256];
	unsigned long long prop_val;
	uint32_t prog;
	int read_size;
	HSAKMT_STATUS ret = HSAKMT_STATUS_SUCCESS;
	bool is_node_supported = true;
	uint32_t num_supported_nodes = 0;
	struct hsa_kfd_topology_context *topology_ctx = ctx->topology_context;

	assert(props);
	snprintf(path, sizeof(path), KFD_DINFO_SYSTEM_PROPERTIES, get_dinfo_dir());
	fd = fopen(path, "r");
	if (!fd)
		return HSAKMT_STATUS_ERROR;

	read_buf = malloc(PAGE_SIZE);
	if (!read_buf) {
		ret = HSAKMT_STATUS_NO_MEMORY;
		goto err1;
	}

	read_size = fread(read_buf, 1, PAGE_SIZE, fd);
	if (read_size <= 0) {
		ret = HSAKMT_STATUS_ERROR;
		goto err2;
	}

	/* Since we're using the buffer as a string, we make sure the string terminates */
	if (read_size >= PAGE_SIZE)
		read_size = PAGE_SIZE - 1;
	read_buf[read_size] = 0;

	/* Read the system properties */
	prog = 0;
	p = read_buf;
	while (sscanf(p += prog, "%255s %llu\n%n", prop_name, &prop_val, &prog) == 2) {
		if (strcmp(prop_name, "platform_oem") == 0)
			props->PlatformOem = (uint32_t)prop_val;
		else if (strcmp(prop_name, "platform_id") == 0)
			props->PlatformId = (uint32_t)prop_val;
		else if (strcmp(prop_name, "platform_rev") == 0)
			props->PlatformRev = (uint32_t)prop_val;
	}

	/*
	 * Discover the number of dinfo nodes:
	 * Assuming that inside nodes folder there are only folders
	 * which represent the node numbers
	 */
	snprintf(path, sizeof(path), KFD_DINFO_NODES, get_dinfo_dir());
	topology_ctx->num_sysfs_nodes = num_subdirs(path, "");

	if (topology_ctx->map_user_to_sysfs_node_id == NULL) {
		/* Trade off - num_sysfs_nodes includes all CPU and GPU nodes.
		 * Slightly more memory is allocated than necessary.
		 */
		topology_ctx->map_user_to_sysfs_node_id = calloc(topology_ctx->num_sysfs_nodes, sizeof(uint32_t));
		if (topology_ctx->map_user_to_sysfs_node_id == NULL) {
			ret = HSAKMT_STATUS_NO_MEMORY;
			goto err2;
		}
		topology_ctx->map_user_to_sysfs_node_id_size = topology_ctx->num_sysfs_nodes;
	} else if (topology_ctx->num_sysfs_nodes > topology_ctx->map_user_to_sysfs_node_id_size) {
		free(topology_ctx->map_user_to_sysfs_node_id);
		topology_ctx->map_user_to_sysfs_node_id = calloc(topology_ctx->num_sysfs_nodes, sizeof(uint32_t));
		if (topology_ctx->map_user_to_sysfs_node_id == NULL) {
			ret = HSAKMT_STATUS_NO_MEMORY;
			goto err2;
		}
		topology_ctx->map_user_to_sysfs_node_id_size = topology_ctx->num_sysfs_nodes;
	}

	for (uint32_t i = 0; i < topology_ctx->num_sysfs_nodes; i++) {
		ret = topology_dinfo_check_node_supported(ctx, i, &is_node_supported);
		if (ret != HSAKMT_STATUS_SUCCESS)
			goto sysfs_parse_failed;
		if (is_node_supported)
			topology_ctx->map_user_to_sysfs_node_id[num_supported_nodes++] = i;
	}
	props->NumNodes = num_supported_nodes;

	free(read_buf);
	fclose(fd);
	return ret;

sysfs_parse_failed:
	free(topology_ctx->map_user_to_sysfs_node_id);
	topology_ctx->map_user_to_sysfs_node_id = NULL;
err2:
	free(read_buf);
err1:
	fclose(fd);
	return ret;
}

HSAKMT_STATUS topology_dinfo_get_node_props_blob(uint32_t sys_node_id,
					char *buf, size_t buflen)
{
	FILE *fd;
	char path[256];
	int read_size;

	snprintf(path, sizeof(path), KFD_DINFO_NODES "/%d/properties", get_dinfo_dir(), sys_node_id);
	fd = fopen(path, "r");
	if (!fd)
		return HSAKMT_STATUS_ERROR;

	read_size = fread(buf, 1, buflen, fd);
	fclose(fd);
	if (read_size <= 0)
		return HSAKMT_STATUS_ERROR;

	/* Since we're using the buffer as a string, we make sure the string terminates */
	if ((size_t)read_size >= buflen)
		read_size = (int)buflen - 1;
	buf[read_size] = 0;

	return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS topology_dinfo_get_mem_props(
					  struct hsa_kfd_topology_context *topology_ctx,
					  uint32_t node_id,
					  uint32_t mem_id,
					  HsaMemoryProperties *props)
{
	FILE *fd;
	char *read_buf, *p;
	char prop_name[256];
	char path[256];
	unsigned long long prop_val;
	uint32_t prog;
	int read_size;
	HSAKMT_STATUS ret = HSAKMT_STATUS_SUCCESS;
	uint32_t sys_node_id;

	assert(props);
	ret = topology_dinfo_map_node_id(topology_ctx, node_id, &sys_node_id);
	if (ret != HSAKMT_STATUS_SUCCESS)
		return ret;

	snprintf(path, 256, KFD_DINFO_NODES "/%d/mem_banks/%d/properties", get_dinfo_dir(), sys_node_id, mem_id);
	fd = fopen(path, "r");
	if (!fd)
		return HSAKMT_STATUS_ERROR;
	read_buf = malloc(PAGE_SIZE);
	if (!read_buf) {
		ret = HSAKMT_STATUS_NO_MEMORY;
		goto err1;
	}

	read_size = fread(read_buf, 1, PAGE_SIZE, fd);
	if (read_size <= 0) {
		ret = HSAKMT_STATUS_ERROR;
		goto err2;
	}

	if (read_size >= PAGE_SIZE)
		read_size = PAGE_SIZE - 1;
	read_buf[read_size] = 0;

	prog = 0;
	p = read_buf;
	while (sscanf(p += prog, "%255s %llu\n%n", prop_name, &prop_val, &prog) == 2) {
		if (strcmp(prop_name, "heap_type") == 0)
			props->HeapType = (uint32_t)prop_val;
		else if (strcmp(prop_name, "size_in_bytes") == 0)
			props->SizeInBytes = (uint64_t)prop_val;
		else if (strcmp(prop_name, "flags") == 0)
			props->Flags.MemoryProperty = (uint32_t)prop_val;
		else if (strcmp(prop_name, "width") == 0)
			props->Width = (uint32_t)prop_val;
		else if (strcmp(prop_name, "mem_clk_max") == 0)
			props->MemoryClockMax = (uint32_t)prop_val;
	}

err2:
	free(read_buf);
err1:
	fclose(fd);
	return ret;
}

HSAKMT_STATUS topology_dinfo_get_cache_props(
					    struct hsa_kfd_topology_context *topology_ctx,
					    uint32_t node_id,
					    uint32_t cache_id,
					    HsaCacheProperties *props)
{
	FILE *fd;
	char *read_buf, *p;
	char prop_name[256];
	char path[256];
	unsigned long long prop_val;
	uint32_t i, prog;
	int read_size;
	HSAKMT_STATUS ret = HSAKMT_STATUS_SUCCESS;
	uint32_t sys_node_id;

	assert(props);
	ret = topology_dinfo_map_node_id(topology_ctx, node_id, &sys_node_id);
	if (ret != HSAKMT_STATUS_SUCCESS)
		return ret;

	snprintf(path, 256, KFD_DINFO_NODES "/%d/caches/%d/properties", get_dinfo_dir(), sys_node_id, cache_id);
	fd = fopen(path, "r");
	if (!fd)
		return HSAKMT_STATUS_ERROR;
	read_buf = malloc(PAGE_SIZE);
	if (!read_buf) {
		ret = HSAKMT_STATUS_NO_MEMORY;
		goto err1;
	}

	read_size = fread(read_buf, 1, PAGE_SIZE, fd);
	if (read_size <= 0) {
		ret = HSAKMT_STATUS_ERROR;
		goto err2;
	}

	if (read_size >= PAGE_SIZE)
		read_size = PAGE_SIZE - 1;
	read_buf[read_size] = 0;

	prog = 0;
	p = read_buf;
	while (sscanf(p += prog, "%255s %llu\n%n", prop_name, &prop_val, &prog) == 2) {
		if (strcmp(prop_name, "processor_id_low") == 0)
			props->ProcessorIdLow = (uint32_t)prop_val;
		else if (strcmp(prop_name, "level") == 0)
			props->CacheLevel = (uint32_t)prop_val;
		else if (strcmp(prop_name, "size") == 0)
			props->CacheSize = (uint32_t)prop_val;
		else if (strcmp(prop_name, "cache_line_size") == 0)
			props->CacheLineSize = (uint32_t)prop_val;
		else if (strcmp(prop_name, "cache_lines_per_tag") == 0)
			props->CacheLinesPerTag = (uint32_t)prop_val;
		else if (strcmp(prop_name, "association") == 0)
			props->CacheAssociativity = (uint32_t)prop_val;
		else if (strcmp(prop_name, "latency") == 0)
			props->CacheLatency = (uint32_t)prop_val;
		else if (strcmp(prop_name, "type") == 0)
			props->CacheType.Value = (uint32_t)prop_val;
		else if (strcmp(prop_name, "persisting_cache_size_max") == 0)
			props->PersistingCacheSizeMax = (uint32_t)prop_val;
		else if (strcmp(prop_name, "sibling_map") == 0)
			break;
	}

	prog = 0;
	if ((sscanf(p, "sibling_map %n", &prog)) == 0 && prog) {
		i = 0;
		while ((i < HSA_CPU_SIBLINGS) &&
			(sscanf(p += prog, "%u%*[,\n]%n", &props->SiblingMap[i++], &prog) == 1))
			continue;
	}

err2:
	free(read_buf);
err1:
	fclose(fd);
	return ret;
}

/* For a given node @node_id the function gets @iolink_id information, i.e.
 * parses the ./nodes/@node_id/io_links/@iolink_id/properties dinfo record.
 * @node_id has to be a valid, accessible node.
 *
 * If node_to specified by the @iolink_id is not accessible the function
 * returns HSAKMT_STATUS_NOT_SUPPORTED. If node_to is accessible, then
 * node_to is mapped from the dinfo node to the user node and returns
 * HSAKMT_STATUS_SUCCESS.
 */
HSAKMT_STATUS topology_dinfo_get_iolink_props(HsaKFDContext *ctx,
					     uint32_t node_id,
					     uint32_t iolink_id,
					     HsaIoLinkProperties *props, bool p2pLink)
{
	FILE *fd;
	char *read_buf, *p;
	char prop_name[256];
	char path[256];
	unsigned long long prop_val;
	uint32_t prog;
	int read_size;
	HSAKMT_STATUS ret = HSAKMT_STATUS_SUCCESS;
	uint32_t sys_node_id;
	struct hsa_kfd_topology_context *topology_ctx = ctx->topology_context;

	assert(props);
	ret = topology_dinfo_map_node_id(topology_ctx, node_id, &sys_node_id);
	if (ret != HSAKMT_STATUS_SUCCESS)
		return ret;

	snprintf(path, 256, KFD_DINFO_NODES "/%d/%s/%d/properties", get_dinfo_dir(), sys_node_id, p2pLink ? "p2p_links" : "io_links", iolink_id);

	fd = fopen(path, "r");
	if (!fd)
		return HSAKMT_STATUS_ERROR;
	read_buf = malloc(PAGE_SIZE);
	if (!read_buf) {
		ret = HSAKMT_STATUS_NO_MEMORY;
		goto err1;
	}

	read_size = fread(read_buf, 1, PAGE_SIZE, fd);
	if (read_size <= 0) {
		ret = (errno == EPERM) ? HSAKMT_STATUS_NOT_SUPPORTED :
					 HSAKMT_STATUS_ERROR;
		goto err2;
	}

	if (read_size >= PAGE_SIZE)
		read_size = PAGE_SIZE - 1;
	read_buf[read_size] = 0;

	prog = 0;
	p = read_buf;
	while (sscanf(p += prog, "%255s %llu\n%n", prop_name, &prop_val, &prog) == 2) {
		if (strcmp(prop_name, "type") == 0)
			props->IoLinkType = (uint32_t)prop_val;
		else if (strcmp(prop_name, "version_major") == 0)
			props->VersionMajor = (uint32_t)prop_val;
		else if (strcmp(prop_name, "version_minor") == 0)
			props->VersionMinor = (uint32_t)prop_val;
		else if (strcmp(prop_name, "node_from") == 0) {
			if (sys_node_id != (uint32_t)prop_val) {
				ret = HSAKMT_STATUS_INVALID_NODE_UNIT;
				goto err2;
			}
			props->NodeFrom = node_id;
		} else if (strcmp(prop_name, "node_to") == 0) {
			bool is_node_supported;
			uint32_t sysfs_node_id;

			sysfs_node_id = (uint32_t)prop_val;
			ret = topology_dinfo_check_node_supported(ctx, sysfs_node_id, &is_node_supported);
			if (!is_node_supported) {
				ret = HSAKMT_STATUS_NOT_SUPPORTED;
				memset(props, 0, sizeof(*props));
				goto err2;
			}
			ret = topology_map_sysfs_to_user_node_id(topology_ctx, sysfs_node_id, &props->NodeTo);
			if (ret != HSAKMT_STATUS_SUCCESS)
				goto err2;
		} else if (strcmp(prop_name, "weight") == 0)
			props->Weight = (uint32_t)prop_val;
		else if (strcmp(prop_name, "min_latency") == 0)
			props->MinimumLatency = (uint32_t)prop_val;
		else if (strcmp(prop_name, "max_latency") == 0)
			props->MaximumLatency = (uint32_t)prop_val;
		else if (strcmp(prop_name, "min_bandwidth") == 0)
			props->MinimumBandwidth = (uint32_t)prop_val;
		else if (strcmp(prop_name, "max_bandwidth") == 0)
			props->MaximumBandwidth = (uint32_t)prop_val;
		else if (strcmp(prop_name, "recommended_transfer_size") == 0)
			props->RecTransferSize = (uint32_t)prop_val;
		else if (strcmp(prop_name, "recommended_sdma_engine_id_mask") == 0)
			props->RecSdmaEngIdMask = (uint32_t)prop_val;
		else if (strcmp(prop_name, "flags") == 0)
			props->Flags.LinkProperty = (uint32_t)prop_val;
	}

err2:
	free(read_buf);
err1:
	fclose(fd);
	return ret;
}

bool topology_dinfo_check_fabric_handle_supported(uint32_t drm_render_minor)
{
	char path[256];

	/* FabricHandleSupported means UALink sysfs is present (hardware
	 * capability). Accelerator readiness (accel_state ready/active) is
	 * checked at fabric handle export time in ROCr, which may return
	 * RESOURCE_NOT_READY.
	 */
	snprintf(path, sizeof(path), "/sys/class/drm/renderD%d/device/ualink", drm_render_minor);
	FILE *file = fopen(path, "r");
	if (!file)
		return false;
	fclose(file);
	return true;
}
