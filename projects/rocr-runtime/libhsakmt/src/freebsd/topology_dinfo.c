/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/*
 * FreeBSD device-info ("dinfo") backend for the KFD topology.
 *
 * FreeBSD has no sysfs, so the amdkfd kernel module exposes the topology as
 * a sysctl(3) MIB tree that mirrors the Linux sysfs layout leaf-for-leaf,
 * rooted at sys.class.kfd.kfd.topology (a literal "/" -> "." translation of
 * /sys/class/kfd/kfd/topology):
 *
 *   sys.class.kfd.kfd.topology.generation_id
 *   sys.class.kfd.kfd.topology.system_properties
 *   sys.class.kfd.kfd.topology.nodes.<N>.{gpu_id,name,properties}
 *   sys.class.kfd.kfd.topology.nodes.<N>.mem_banks.<M>.properties
 *   sys.class.kfd.kfd.topology.nodes.<N>.caches.<C>.properties
 *   sys.class.kfd.kfd.topology.nodes.<N>.{io_links,p2p_links}.<L>.properties
 *
 * Every leaf is a CTLTYPE_STRING containing the exact same text as the
 * corresponding Linux sysfs file (including scalar leaves like gpu_id and
 * generation_id, which are decimal strings, not integers). So the parsing
 * below matches the Linux backend; only the byte-fetch differs.
 *
 * generation_id is load-bearing: topology_take_snapshot() in topology.c reads
 * it before and after the whole node walk and retries on mismatch, so the
 * kernel side MUST bump it on any topology change.
 *
 * Model mode (hsakmt_use_model) is a plain directory of files
 * (hsakmt_model_topology) on every OS, so dinfo_read() falls back to file
 * reads in that case.
 */

#include <assert.h>
#include <errno.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/sysctl.h>
#include <unistd.h>

#include "fmm.h"
#include "hsakmt/hsakmtmodel.h"
#include "topology_dinfo.h"

#define KFD_DINFO_SYSCTL_PREFIX "sys.class.kfd.kfd.topology"

const char *get_dinfo_dir(void)
{
	if (hsakmt_use_model)
		return hsakmt_model_topology;
	return KFD_DINFO_SYSCTL_PREFIX;
}

/* Turn a sysfs-style relative path ("nodes/1/properties") into a full
 * sysctl name ("sys.class.kfd.kfd.topology.nodes.1.properties"). */
static bool dinfo_sysctl_name(char *key, size_t keylen, const char *rel)
{
	int n = snprintf(key, keylen, KFD_DINFO_SYSCTL_PREFIX ".%s", rel);

	if (n < 0 || (size_t)n >= keylen)
		return false;
	for (char *c = key; *c; c++)
		if (*c == '/')
			*c = '.';
	return true;
}

/* Fetch one dinfo record into @buf as a NUL-terminated string.
 * @fmt/... produce a sysfs-style path relative to the topology root, e.g.
 * "nodes/%u/properties". In model mode that's read as a file under
 * hsakmt_model_topology; otherwise it's fetched with sysctlbyname().
 *
 * Mirrors the Linux fopen()/fread() semantics: oversized records are
 * truncated to @buflen - 1 bytes, an empty record is an error, and EPERM
 * maps to HSAKMT_STATUS_NOT_SUPPORTED (node hidden from this process).
 */
static HSAKMT_STATUS dinfo_read(char *buf, size_t buflen, const char *fmt, ...)
	__attribute__((format(printf, 3, 4)));

/* NUL-terminate the @len-byte record in @buf and make sure it ends in '\n'.
 * sysfs records always do, and the parsers rely on it: the sibling_map loop
 * would otherwise read past the end of the buffer.  A record that fills the
 * buffer loses its last byte to the newline, as it was truncated anyway.
 */
static void dinfo_terminate(char *buf, size_t buflen, size_t len)
{
	if (len > buflen - 1)
		len = buflen - 1;
	if (len > 0 && buf[len - 1] != '\n') {
		if (len == buflen - 1)
			len--;
		buf[len++] = '\n';
	}
	buf[len] = '\0';
}

static HSAKMT_STATUS dinfo_read(char *buf, size_t buflen, const char *fmt, ...)
{
	char rel[192];
	char key[256];
	size_t len;
	va_list ap;
	int n;

	assert(buf && buflen > 1);

	va_start(ap, fmt);
	n = vsnprintf(rel, sizeof(rel), fmt, ap);
	va_end(ap);
	if (n < 0 || (size_t)n >= sizeof(rel))
		return HSAKMT_STATUS_ERROR;

	if (hsakmt_use_model) {
		FILE *fd;

		snprintf(key, sizeof(key), "%s/%s", hsakmt_model_topology, rel);
		fd = fopen(key, "r");
		if (!fd)
			return HSAKMT_STATUS_ERROR;
		len = fread(buf, 1, buflen - 1, fd);
		fclose(fd);
		if (len == 0)
			return HSAKMT_STATUS_ERROR;
		dinfo_terminate(buf, buflen, len);
		return HSAKMT_STATUS_SUCCESS;
	}

	if (!dinfo_sysctl_name(key, sizeof(key), rel))
		return HSAKMT_STATUS_ERROR;

	len = buflen - 1;
	if (sysctlbyname(key, buf, &len, NULL, 0)) {
		if (errno == ENOMEM) {
			/* Record larger than buf: sysctl copied out what fit.
			 * Truncate like fread() into a fixed buffer would. */
			len = buflen - 1;
		} else {
			return (errno == EPERM) ? HSAKMT_STATUS_NOT_SUPPORTED :
						  HSAKMT_STATUS_ERROR;
		}
	}
	/* sysctl string handlers count the trailing NUL in len. */
	if (len > buflen - 1)
		len = buflen - 1;
	buf[len] = '\0';
	if (buf[0] == '\0')
		return HSAKMT_STATUS_ERROR;
	dinfo_terminate(buf, buflen, strlen(buf));
	return HSAKMT_STATUS_SUCCESS;
}

/* Does a sysctl node exist at this relative path? Used for node enumeration
 * since there's no directory to num_subdirs() over. */
static bool dinfo_exists(const char *rel)
{
	char key[256];
	int mib[CTL_MAXNAME];
	size_t miblen = CTL_MAXNAME;

	if (!dinfo_sysctl_name(key, sizeof(key), rel))
		return false;
	return sysctlnametomib(key, mib, &miblen) == 0;
}

/* Scalar leaf (gpu_id, generation_id): decimal string -> uint32_t. */
static HSAKMT_STATUS dinfo_read_u32(uint32_t *val, const char *rel)
{
	char buf[32];
	char *end;
	unsigned long v;
	HSAKMT_STATUS ret;

	ret = dinfo_read(buf, sizeof(buf), "%s", rel);
	if (ret != HSAKMT_STATUS_SUCCESS)
		return ret;

	errno = 0;
	v = strtoul(buf, &end, 10);
	if (errno || end == buf)
		return HSAKMT_STATUS_ERROR;
	*val = (uint32_t)v;
	return HSAKMT_STATUS_SUCCESS;
}

uint32_t topology_dinfo_get_num_procs(void)
{
	long n = sysconf(_SC_NPROCESSORS_ONLN);

	if (n <= 0 || n > UINT32_MAX)
		return 0;
	return (uint32_t)n;
}

HSAKMT_STATUS topology_dinfo_get_generation(uint32_t *gen)
{
	assert(gen);
	return dinfo_read_u32(gen, "generation_id");
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
	char rel[64];

	assert(gpu_id);
	snprintf(rel, sizeof(rel), "nodes/%u/gpu_id", sysfs_node_id);
	return dinfo_read_u32(gpu_id, rel);
}

/* Check if the @sysfs_node_id is supported. A dinfo node is not supported
 *	- if corresponding drm render node is not available.
 *	- if node information is not accessible (EPERM)
 */
HSAKMT_STATUS topology_dinfo_check_node_supported(HsaKFDContext *ctx,
					uint32_t sysfs_node_id, bool *is_node_supported)
{
	uint32_t gpu_id;
	char *read_buf, *p;
	char prop_name[256];
	unsigned long long prop_val;
	int prog;
	uint32_t drm_render_minor = 0;
	int ret_value;
	HSAKMT_STATUS ret;

	*is_node_supported = false;

	/* Retrieve the GPU ID */
	ret = topology_dinfo_get_gpu_id(sysfs_node_id, &gpu_id);
	if (ret == HSAKMT_STATUS_NOT_SUPPORTED)
		return HSAKMT_STATUS_SUCCESS;
	if (ret != HSAKMT_STATUS_SUCCESS)
		return ret;

	if (gpu_id == 0) {	/* CPU node */
		*is_node_supported = true;
		return HSAKMT_STATUS_SUCCESS;
	}

	read_buf = malloc(PAGE_SIZE);
	if (!read_buf)
		return HSAKMT_STATUS_NO_MEMORY;

	ret = dinfo_read(read_buf, PAGE_SIZE, "nodes/%u/properties", sysfs_node_id);
	if (ret != HSAKMT_STATUS_SUCCESS) {
		ret = HSAKMT_STATUS_ERROR;
		goto err;
	}

	/* drm_render_minor is one line of the node properties record */
	prog = 0;
	p = read_buf;
	while (sscanf(p += prog, "%255s %llu\n%n", prop_name, &prop_val, &prog) == 2) {
		if (strcmp(prop_name, "drm_render_minor") == 0) {
			drm_render_minor = (uint32_t)prop_val;
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
	return ret;
}

/* Number of nodes.<N> entries. Like the Linux num_subdirs() approach this
 * assumes node ids are contiguous from 0, so probe until the first gap. */
static uint32_t dinfo_count_nodes(void)
{
	char rel[64];
	uint32_t n;

	if (hsakmt_use_model) {
		char path[256];

		snprintf(path, sizeof(path), "%s/nodes", hsakmt_model_topology);
		return (uint32_t)num_subdirs(path, "");
	}

	for (n = 0;; n++) {
		snprintf(rel, sizeof(rel), "nodes/%u", n);
		if (!dinfo_exists(rel))
			break;
	}
	return n;
}

HSAKMT_STATUS hsakmt_topology_dinfo_get_system_props(HsaKFDContext *ctx,
					HsaSystemProperties *props)
{
	char *read_buf, *p;
	char prop_name[256];
	unsigned long long prop_val;
	int prog;
	HSAKMT_STATUS ret;
	bool is_node_supported = true;
	uint32_t num_supported_nodes = 0;
	struct hsa_kfd_topology_context *topology_ctx = ctx->topology_context;

	assert(props);

	read_buf = malloc(PAGE_SIZE);
	if (!read_buf)
		return HSAKMT_STATUS_NO_MEMORY;

	ret = dinfo_read(read_buf, PAGE_SIZE, "system_properties");
	if (ret != HSAKMT_STATUS_SUCCESS) {
		ret = HSAKMT_STATUS_ERROR;
		goto err;
	}

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

	topology_ctx->num_sysfs_nodes = dinfo_count_nodes();

	if (topology_ctx->map_user_to_sysfs_node_id == NULL ||
	    topology_ctx->num_sysfs_nodes > topology_ctx->map_user_to_sysfs_node_id_size) {
		/* Trade off - num_sysfs_nodes includes all CPU and GPU nodes.
		 * Slightly more memory is allocated than necessary.
		 */
		free(topology_ctx->map_user_to_sysfs_node_id);
		topology_ctx->map_user_to_sysfs_node_id =
			(uint32_t *)calloc(topology_ctx->num_sysfs_nodes, sizeof(uint32_t));
		if (topology_ctx->map_user_to_sysfs_node_id == NULL) {
			topology_ctx->map_user_to_sysfs_node_id_size = 0;
			ret = HSAKMT_STATUS_NO_MEMORY;
			goto err;
		}
		topology_ctx->map_user_to_sysfs_node_id_size = topology_ctx->num_sysfs_nodes;
	}

	for (uint32_t i = 0; i < topology_ctx->num_sysfs_nodes; i++) {
		ret = topology_dinfo_check_node_supported(ctx, i, &is_node_supported);
		if (ret != HSAKMT_STATUS_SUCCESS) {
			free(topology_ctx->map_user_to_sysfs_node_id);
			topology_ctx->map_user_to_sysfs_node_id = NULL;
			topology_ctx->map_user_to_sysfs_node_id_size = 0;
			goto err;
		}
		if (is_node_supported)
			topology_ctx->map_user_to_sysfs_node_id[num_supported_nodes++] = i;
	}
	props->NumNodes = num_supported_nodes;

err:
	free(read_buf);
	return ret;
}

HSAKMT_STATUS topology_dinfo_get_node_props_blob(uint32_t sys_node_id,
					char *buf, size_t buflen)
{
	HSAKMT_STATUS ret = dinfo_read(buf, buflen, "nodes/%u/properties", sys_node_id);

	return (ret == HSAKMT_STATUS_SUCCESS) ? ret : HSAKMT_STATUS_ERROR;
}

HSAKMT_STATUS topology_dinfo_get_mem_props(
					  struct hsa_kfd_topology_context *topology_ctx,
					  uint32_t node_id,
					  uint32_t mem_id,
					  HsaMemoryProperties *props)
{
	char *read_buf, *p;
	char prop_name[256];
	unsigned long long prop_val;
	int prog;
	HSAKMT_STATUS ret;
	uint32_t sys_node_id;

	assert(props);
	ret = topology_dinfo_map_node_id(topology_ctx, node_id, &sys_node_id);
	if (ret != HSAKMT_STATUS_SUCCESS)
		return ret;

	read_buf = malloc(PAGE_SIZE);
	if (!read_buf)
		return HSAKMT_STATUS_NO_MEMORY;

	ret = dinfo_read(read_buf, PAGE_SIZE, "nodes/%u/mem_banks/%u/properties",
			 sys_node_id, mem_id);
	if (ret != HSAKMT_STATUS_SUCCESS) {
		ret = HSAKMT_STATUS_ERROR;
		goto err;
	}

	prog = 0;
	p = read_buf;
	while (sscanf(p += prog, "%255s %llu\n%n", prop_name, &prop_val, &prog) == 2) {
		if (strcmp(prop_name, "heap_type") == 0)
			props->HeapType = (HSA_HEAPTYPE)prop_val;
		else if (strcmp(prop_name, "size_in_bytes") == 0)
			props->SizeInBytes = (uint64_t)prop_val;
		else if (strcmp(prop_name, "flags") == 0)
			props->Flags.MemoryProperty = (uint32_t)prop_val;
		else if (strcmp(prop_name, "width") == 0)
			props->Width = (uint32_t)prop_val;
		else if (strcmp(prop_name, "mem_clk_max") == 0)
			props->MemoryClockMax = (uint32_t)prop_val;
	}

err:
	free(read_buf);
	return ret;
}

HSAKMT_STATUS topology_dinfo_get_cache_props(
					    struct hsa_kfd_topology_context *topology_ctx,
					    uint32_t node_id,
					    uint32_t cache_id,
					    HsaCacheProperties *props)
{
	char *read_buf, *p;
	char prop_name[256];
	unsigned long long prop_val;
	uint32_t i;
	int prog;
	HSAKMT_STATUS ret;
	uint32_t sys_node_id;

	assert(props);
	ret = topology_dinfo_map_node_id(topology_ctx, node_id, &sys_node_id);
	if (ret != HSAKMT_STATUS_SUCCESS)
		return ret;

	read_buf = malloc(PAGE_SIZE);
	if (!read_buf)
		return HSAKMT_STATUS_NO_MEMORY;

	ret = dinfo_read(read_buf, PAGE_SIZE, "nodes/%u/caches/%u/properties",
			 sys_node_id, cache_id);
	if (ret != HSAKMT_STATUS_SUCCESS) {
		ret = HSAKMT_STATUS_ERROR;
		goto err;
	}

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

	/* "sibling_map 1,0,1,...\n" doesn't match "%s %llu\n" in full, so the
	 * loop above stops with p pointing at it (same as the Linux backend). */
	prog = 0;
	if ((sscanf(p, "sibling_map %n", &prog)) == 0 && prog) {
		i = 0;
		while ((i < HSA_CPU_SIBLINGS) &&
			(sscanf(p += prog, "%u%*[,\n]%n", &props->SiblingMap[i++], &prog) == 1))
			continue;
	}

err:
	free(read_buf);
	return ret;
}

/* For a given node @node_id the function gets @iolink_id information, i.e.
 * parses the nodes.@node_id.{io,p2p}_links.@iolink_id.properties record.
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
	char *read_buf, *p;
	char prop_name[256];
	unsigned long long prop_val;
	int prog;
	HSAKMT_STATUS ret;
	uint32_t sys_node_id;
	struct hsa_kfd_topology_context *topology_ctx = ctx->topology_context;

	assert(props);
	ret = topology_dinfo_map_node_id(topology_ctx, node_id, &sys_node_id);
	if (ret != HSAKMT_STATUS_SUCCESS)
		return ret;

	read_buf = malloc(PAGE_SIZE);
	if (!read_buf)
		return HSAKMT_STATUS_NO_MEMORY;

	/* EPERM -> NOT_SUPPORTED passes through, as on Linux */
	ret = dinfo_read(read_buf, PAGE_SIZE, "nodes/%u/%s/%u/properties", sys_node_id,
			 p2pLink ? "p2p_links" : "io_links", iolink_id);
	if (ret != HSAKMT_STATUS_SUCCESS)
		goto err;

	prog = 0;
	p = read_buf;
	while (sscanf(p += prog, "%255s %llu\n%n", prop_name, &prop_val, &prog) == 2) {
		if (strcmp(prop_name, "type") == 0)
			props->IoLinkType = (HSA_IOLINKTYPE)prop_val;
		else if (strcmp(prop_name, "version_major") == 0)
			props->VersionMajor = (uint32_t)prop_val;
		else if (strcmp(prop_name, "version_minor") == 0)
			props->VersionMinor = (uint32_t)prop_val;
		else if (strcmp(prop_name, "node_from") == 0) {
			if (sys_node_id != (uint32_t)prop_val) {
				ret = HSAKMT_STATUS_INVALID_NODE_UNIT;
				goto err;
			}
			props->NodeFrom = node_id;
		} else if (strcmp(prop_name, "node_to") == 0) {
			bool is_node_supported = false;
			uint32_t sysfs_node_id;

			sysfs_node_id = (uint32_t)prop_val;
			ret = topology_dinfo_check_node_supported(ctx, sysfs_node_id, &is_node_supported);
			if (!is_node_supported) {
				ret = HSAKMT_STATUS_NOT_SUPPORTED;
				memset(props, 0, sizeof(*props));
				goto err;
			}
			ret = topology_map_sysfs_to_user_node_id(topology_ctx, sysfs_node_id, &props->NodeTo);
			if (ret != HSAKMT_STATUS_SUCCESS)
				goto err;
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

err:
	free(read_buf);
	return ret;
}

bool topology_dinfo_check_fabric_handle_supported(uint32_t drm_render_minor)
{
	/* UALink fabric-handle capability probe. Linux checks for
	 * /sys/class/drm/renderD<N>/device/ualink; FreeBSD has no equivalent
	 * yet, so report unsupported until one exists. */
	(void)drm_render_minor;
	return false;
}
