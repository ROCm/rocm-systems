/*
 * Copyright © 2020 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use, copy,
 * modify, merge, publish, distribute, sublicense, and/or sell copies
 * of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including
 * the next paragraph) shall be included in all copies or substantial
 * portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT.  IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */
#include "libhsakmt.h"
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <inttypes.h>
#include <sys/mman.h>
#include <sys/time.h>

#ifdef USE_DRM_AMDGPU_SVM
#include "hsakmt/drm/amdgpu_svm.h"
#include "fmm.h"



/**
 * HSA_SVM_FLAGS bitmask to new DRM boolean attribute type mapping.
 * Each old flag bit maps to an individual DRM attr type with value 0 or 1.
 */
struct flag_to_attr_map {
	HSAuint32 hsa_flag;
	HSAuint32 drm_attr_type;
};

static const struct flag_to_attr_map flag_map[] = {
	{ HSA_SVM_FLAG_HOST_ACCESS,	AMDGPU_SVM_ATTR_HOST_ACCESS },
	{ HSA_SVM_FLAG_COHERENT,	AMDGPU_SVM_ATTR_COHERENT },
	{ HSA_SVM_FLAG_HIVE_LOCAL,	AMDGPU_SVM_ATTR_HIVE_LOCAL },
	{ HSA_SVM_FLAG_GPU_RO,		AMDGPU_SVM_ATTR_GPU_RO },
	{ HSA_SVM_FLAG_GPU_EXEC,	AMDGPU_SVM_ATTR_GPU_EXEC },
	{ HSA_SVM_FLAG_GPU_READ_MOSTLY, AMDGPU_SVM_ATTR_GPU_READ_MOSTLY },
	{ HSA_SVM_FLAG_EXT_COHERENT,    AMDGPU_SVM_ATTR_EXT_COHERENT },
};
#define FLAG_MAP_COUNT (sizeof(flag_map) / sizeof(flag_map[0]))

/**
 * Translate HSA-level SVM attributes to new DRM UAPI attributes.
 *
 * HSA API uses:
 *   - ACCESS/ACCESS_IN_PLACE/NO_ACCESS as separate types with node ID as value
 *   - SET_FLAGS/CLR_FLAGS with bitmask values
 *   - GRANULARITY as type=7
 *
 * New DRM API uses:
 *   - ACCESS type=2 with value enum (INACCESSIBLE/IN_PLACE/ALLOW_MIGRATE)
 *   - Individual boolean attrs (type=4..11, value=0/1)
 *   - GRANULARITY as type=3
 *
 * Returns number of DRM attrs written, or negative on error.
 * out_attrs must have space for at least nattr + FLAG_MAP_COUNT entries
 * (SET_FLAGS can expand to multiple individual attrs).
 */
static int hsa_to_drm_attrs(HsaKFDContext *ctx,
			    unsigned int nattr,
			    const HSA_SVM_ATTRIBUTE *attrs,
			    struct drm_amdgpu_svm_attribute *out_attrs,
			    HsaAMDGPUDeviceHandle *out_device)
{
	int out_count = 0;
	HSAuint32 i, j;
	HSAKMT_STATUS r;
	HsaAMDGPUDeviceHandle deviceHandle = 0;

	for (i = 0; i < nattr; i++) {
		switch (attrs[i].type) {
		case HSA_SVM_ATTR_PREFERRED_LOC:
		case HSA_SVM_ATTR_PREFETCH_LOC: {
			HSAuint32 gpu_id = 0;
			HsaAMDGPUDeviceHandle target_dev = 0;

			out_attrs[out_count].type =
				(attrs[i].type == HSA_SVM_ATTR_PREFERRED_LOC) ?
				AMDGPU_SVM_ATTR_PREFERRED_LOC : AMDGPU_SVM_ATTR_PREFETCH_LOC;

			if (attrs[i].value == INVALID_NODEID) {
				out_attrs[out_count].value = AMDGPU_SVM_LOCATION_UNDEFINED;
				out_count++;
				break;
			}

			r = hsakmt_validate_nodeid(ctx, attrs[i].value, &gpu_id);
			if (r != HSAKMT_STATUS_SUCCESS)
				return -1;

			if (!gpu_id) {
				/* CPU node maps to SYSMEM. */
				out_attrs[out_count].value = AMDGPU_SVM_LOCATION_SYSMEM;
				out_count++;
				break;
			}

			r = hsaKmtGetAMDGPUDeviceHandleCtx(ctx, attrs[i].value, &target_dev);
			if (r != HSAKMT_STATUS_SUCCESS || !target_dev)
				return -1;

			if (!deviceHandle) {
				deviceHandle = target_dev;
			} else if (deviceHandle != target_dev) {
				pr_debug("conflicting per-fd GPU scope for location attrs\n");
				return -1;
			}

			/*
			 * New UAPI encodes device target via ioctl fd context.
			 * A concrete GPU node maps to LOCAL on that GPU's fd.
			 */
			out_attrs[out_count].value = AMDGPU_SVM_LOCATION_LOCAL;
			out_count++;
			break;
		}

		case HSA_SVM_ATTR_ACCESS:
			/* Old: type=ACCESS, value=nodeId → New: type=ACCESS, value=ALLOW_MIGRATE */
			out_attrs[out_count].type = AMDGPU_SVM_ATTR_ACCESS;
			out_attrs[out_count].value = AMDGPU_SVM_ACCESS_ALLOW_MIGRATE;
			out_count++;
			/* Get device handle from node */
			if (!deviceHandle) {
				r = hsaKmtGetAMDGPUDeviceHandleCtx(ctx, attrs[i].value, &deviceHandle);
				if (r != HSAKMT_STATUS_SUCCESS)
					return -1;
			} else {
				HsaAMDGPUDeviceHandle access_dev = 0;
				r = hsaKmtGetAMDGPUDeviceHandleCtx(ctx, attrs[i].value, &access_dev);
				if (r != HSAKMT_STATUS_SUCCESS || access_dev != deviceHandle) {
					pr_debug("conflicting per-fd GPU scope for access attrs\n");
					return -1;
				}
			}
			break;

		case HSA_SVM_ATTR_ACCESS_IN_PLACE:
			out_attrs[out_count].type = AMDGPU_SVM_ATTR_ACCESS;
			out_attrs[out_count].value = AMDGPU_SVM_ACCESS_IN_PLACE;
			out_count++;
			if (!deviceHandle) {
				r = hsaKmtGetAMDGPUDeviceHandleCtx(ctx, attrs[i].value, &deviceHandle);
				if (r != HSAKMT_STATUS_SUCCESS)
					return -1;
			} else {
				HsaAMDGPUDeviceHandle access_dev = 0;
				r = hsaKmtGetAMDGPUDeviceHandleCtx(ctx, attrs[i].value, &access_dev);
				if (r != HSAKMT_STATUS_SUCCESS || access_dev != deviceHandle) {
					pr_debug("conflicting per-fd GPU scope for access attrs\n");
					return -1;
				}
			}
			break;

		case HSA_SVM_ATTR_NO_ACCESS:
			out_attrs[out_count].type = AMDGPU_SVM_ATTR_ACCESS;
			out_attrs[out_count].value = AMDGPU_SVM_ACCESS_INACCESSIBLE;
			out_count++;
			if (!deviceHandle) {
				r = hsaKmtGetAMDGPUDeviceHandleCtx(ctx, attrs[i].value, &deviceHandle);
				if (r != HSAKMT_STATUS_SUCCESS)
					return -1;
			} else {
				HsaAMDGPUDeviceHandle access_dev = 0;
				r = hsaKmtGetAMDGPUDeviceHandleCtx(ctx, attrs[i].value, &access_dev);
				if (r != HSAKMT_STATUS_SUCCESS || access_dev != deviceHandle) {
					pr_debug("conflicting per-fd GPU scope for access attrs\n");
					return -1;
				}
			}
			break;

		case HSA_SVM_ATTR_SET_FLAGS:
			/* Expand bitmask to individual boolean attrs (set to 1) */
			for (j = 0; j < FLAG_MAP_COUNT; j++) {
				if (attrs[i].value & flag_map[j].hsa_flag) {
					out_attrs[out_count].type = flag_map[j].drm_attr_type;
					out_attrs[out_count].value = 1;
					out_count++;
				}
			}
			break;

		case HSA_SVM_ATTR_CLR_FLAGS:
			/* Expand bitmask to individual boolean attrs (set to 0) */
			for (j = 0; j < FLAG_MAP_COUNT; j++) {
				if (attrs[i].value & flag_map[j].hsa_flag) {
					out_attrs[out_count].type = flag_map[j].drm_attr_type;
					out_attrs[out_count].value = 0;
					out_count++;
				}
			}
			break;

		case HSA_SVM_ATTR_GRANULARITY:
			out_attrs[out_count].type = AMDGPU_SVM_ATTR_GRANULARITY;
			out_attrs[out_count].value = attrs[i].value;
			out_count++;
			break;

		default:
			pr_debug("unknown HSA SVM attr type %d\n", attrs[i].type);
			return -1;
		}
	}

	*out_device = deviceHandle;
	return out_count;
}

static HSAKMT_STATUS resolve_node_from_device_handle(HsaKFDContext *ctx,
					      HsaAMDGPUDeviceHandle deviceHandle,
					      HSAuint32 *node_id)
{
	HsaSystemProperties sys_props = {0};
	HSAKMT_STATUS r;
	HSAuint32 i;
	int fd;

	if (!deviceHandle || !node_id)
		return HSAKMT_STATUS_INVALID_PARAMETER;

	fd = hsakmt_fn_amdgpu_device_get_fd(deviceHandle);
	if (fd < 0)
		return HSAKMT_STATUS_ERROR;

	r = hsakmt_topology_sysfs_get_system_props(ctx, &sys_props);
	if (r != HSAKMT_STATUS_SUCCESS)
		return r;

	for (i = 0; i < sys_props.NumNodes; i++) {
		HsaNodeProperties props;
		HsaAMDGPUDeviceHandle node_dev = 0;

		r = hsakmt_topology_get_node_props(ctx, i, &props);
		if (r != HSAKMT_STATUS_SUCCESS || !props.KFDGpuID)
			continue;

		r = hsaKmtGetAMDGPUDeviceHandleCtx(ctx, i, &node_dev);
		if (r != HSAKMT_STATUS_SUCCESS || !node_dev)
			continue;

		if (hsakmt_fn_amdgpu_device_get_fd(node_dev) == fd) {
			*node_id = i;
			return HSAKMT_STATUS_SUCCESS;
		}
	}

	return HSAKMT_STATUS_INVALID_NODE_UNIT;
}

static HSAKMT_STATUS resolve_hive_target_node(HsaKFDContext *ctx,
					      HSAuint32 current_node,
					      HSAuint32 *target_node)
{
	HsaNodeProperties cur_props;
	HsaIoLinkProperties *links = NULL;
	HSAKMT_STATUS r;
	HSAuint32 i;

	if (!target_node)
		return HSAKMT_STATUS_INVALID_PARAMETER;

	r = hsakmt_topology_get_node_props(ctx, current_node, &cur_props);
	if (r != HSAKMT_STATUS_SUCCESS)
		return r;

	if (!cur_props.KFDGpuID)
		return HSAKMT_STATUS_INVALID_NODE_UNIT;

	if (!cur_props.NumIOLinks) {
		*target_node = current_node;
		return HSAKMT_STATUS_SUCCESS;
	}

	links = calloc(cur_props.NumIOLinks, sizeof(*links));
	if (!links)
		return HSAKMT_STATUS_NO_MEMORY;

	r = hsakmt_topology_get_iolink_props(ctx, current_node,
					     cur_props.NumIOLinks, links);
	if (r != HSAKMT_STATUS_SUCCESS)
		goto out;

	/* Prefer same HiveID peers when available. */
	for (i = 0; i < cur_props.NumIOLinks; i++) {
		HsaNodeProperties peer_props;

		if (links[i].NodeTo == current_node)
			continue;

		r = hsakmt_topology_get_node_props(ctx, links[i].NodeTo, &peer_props);
		if (r != HSAKMT_STATUS_SUCCESS)
			continue;

		if (peer_props.KFDGpuID && cur_props.HiveID &&
		    peer_props.HiveID == cur_props.HiveID) {
			*target_node = links[i].NodeTo;
			r = HSAKMT_STATUS_SUCCESS;
			goto out;
		}
	}

	/* Fallback to any directly reachable GPU node. */
	for (i = 0; i < cur_props.NumIOLinks; i++) {
		HsaNodeProperties peer_props;

		if (links[i].NodeTo == current_node)
			continue;

		r = hsakmt_topology_get_node_props(ctx, links[i].NodeTo, &peer_props);
		if (r != HSAKMT_STATUS_SUCCESS)
			continue;

		if (peer_props.KFDGpuID) {
			*target_node = links[i].NodeTo;
			r = HSAKMT_STATUS_SUCCESS;
			goto out;
		}
	}

	/* No peer GPU link found: fallback to current GPU node. */
	*target_node = current_node;
	r = HSAKMT_STATUS_SUCCESS;

out:
	free(links);
	return r;
}

static HSAKMT_STATUS get_drm_svm_attrs_aggregated(HsaAMDGPUDeviceHandle deviceHandle,
						  uint64_t start_addr,
						  uint64_t size,
						  uint32_t nattr,
						  struct drm_amdgpu_svm_attribute *attrs)
{
	struct drm_amdgpu_svm_attribute segment_attrs[nattr];
	uint64_t offset = 0;
	uint64_t remaining = size;
	uint64_t seg_size = 0;
	uint32_t i;
	int r;

	if (!attrs || !size || !nattr)
		return HSAKMT_STATUS_INVALID_PARAMETER;

	for (i = 0; i < nattr; i++)
		segment_attrs[i].type = attrs[i].type;

	while (remaining) {
		for (i = 0; i < nattr; i++)
			segment_attrs[i].value = 0;

		seg_size = 0;
		r = amdgpu_svm_get_attr_ex(deviceHandle, start_addr + offset,
					  remaining, nattr, segment_attrs, &seg_size);
		if (r)
			return HSAKMT_STATUS_ERROR;

		if (!seg_size || seg_size > remaining)
			return HSAKMT_STATUS_ERROR;

		if (offset == 0) {
			for (i = 0; i < nattr; i++)
				attrs[i].value = segment_attrs[i].value;
		} else {
			for (i = 0; i < nattr; i++) {
				switch (attrs[i].type) {
				case AMDGPU_SVM_ATTR_PREFERRED_LOC:
				case AMDGPU_SVM_ATTR_PREFETCH_LOC:
					if (attrs[i].value != segment_attrs[i].value)
						attrs[i].value = AMDGPU_SVM_LOCATION_UNDEFINED;
					break;
				case AMDGPU_SVM_ATTR_ACCESS:
					if (attrs[i].value != segment_attrs[i].value)
						attrs[i].value = AMDGPU_SVM_ACCESS_INACCESSIBLE;
					break;
				case AMDGPU_SVM_ATTR_GRANULARITY:
					if (segment_attrs[i].value < attrs[i].value)
						attrs[i].value = segment_attrs[i].value;
					break;
				case AMDGPU_SVM_ATTR_HOST_ACCESS:
				case AMDGPU_SVM_ATTR_COHERENT:
				case AMDGPU_SVM_ATTR_HIVE_LOCAL:
				case AMDGPU_SVM_ATTR_GPU_RO:
				case AMDGPU_SVM_ATTR_GPU_EXEC:
				case AMDGPU_SVM_ATTR_GPU_READ_MOSTLY:
				case AMDGPU_SVM_ATTR_EXT_COHERENT:
					attrs[i].value = attrs[i].value && segment_attrs[i].value;
					break;
				default:
					return HSAKMT_STATUS_INVALID_PARAMETER;
				}
			}
		}

		offset += seg_size;
		remaining -= seg_size;
	}

	return HSAKMT_STATUS_SUCCESS;
}

static HSAKMT_STATUS get_drm_svm_location_attr(HsaAMDGPUDeviceHandle deviceHandle,
						 uint64_t start_addr,
						 uint64_t size,
						 uint32_t drm_attr_type,
						 uint32_t *value)
{
	struct drm_amdgpu_svm_attribute attr = {
		.type = drm_attr_type,
		.value = 0,
	};
	HSAKMT_STATUS r;

	if (!value)
		return HSAKMT_STATUS_INVALID_PARAMETER;

	r = get_drm_svm_attrs_aggregated(deviceHandle, start_addr, size, 1, &attr);
	if (r != HSAKMT_STATUS_SUCCESS)
		return r;

	*value = attr.value;

	return HSAKMT_STATUS_SUCCESS;
}

static HSAKMT_STATUS validate_svm_range_query_args(void *start_addr,
						   HSAuint64 size,
						   unsigned int nattr,
						   HSA_SVM_ATTRIBUTE *attrs)
{
	if (!start_addr || !size)
		return HSAKMT_STATUS_INVALID_PARAMETER;
	if ((uint64_t)start_addr & (PAGE_SIZE - 1))
		return HSAKMT_STATUS_INVALID_PARAMETER;
	if (size & (PAGE_SIZE - 1))
		return HSAKMT_STATUS_INVALID_PARAMETER;
	if (nattr && !attrs)
		return HSAKMT_STATUS_INVALID_PARAMETER;

	return HSAKMT_STATUS_SUCCESS;
}

static HSAKMT_STATUS hsa_get_query_to_drm_attrs(HsaKFDContext *ctx,
						 unsigned int nattr,
						 const HSA_SVM_ATTRIBUTE *attrs,
						 struct drm_amdgpu_svm_attribute *drm_attrs,
						 HsaAMDGPUDeviceHandle *out_device,
						 int *need_flags)
{
	HsaAMDGPUDeviceHandle deviceHandle = 0;
	HSAKMT_STATUS r;
	HSAuint32 i;

	if (!ctx || (nattr && !attrs) || (nattr && !drm_attrs) || !out_device || !need_flags)
		return HSAKMT_STATUS_INVALID_PARAMETER;

	*need_flags = 0;

	for (i = 0; i < nattr; i++) {
		drm_attrs[i].value = 0;
		switch (attrs[i].type) {
		case HSA_SVM_ATTR_PREFERRED_LOC:
			drm_attrs[i].type = AMDGPU_SVM_ATTR_PREFERRED_LOC;
			break;
		case HSA_SVM_ATTR_PREFETCH_LOC:
			drm_attrs[i].type = AMDGPU_SVM_ATTR_PREFETCH_LOC;
			break;
		case HSA_SVM_ATTR_ACCESS:
		case HSA_SVM_ATTR_ACCESS_IN_PLACE:
		case HSA_SVM_ATTR_NO_ACCESS:
			drm_attrs[i].type = AMDGPU_SVM_ATTR_ACCESS;
			if (!deviceHandle) {
				r = hsaKmtGetAMDGPUDeviceHandleCtx(ctx, attrs[i].value, &deviceHandle);
				if (r != HSAKMT_STATUS_SUCCESS)
					return r;
			} else {
				HsaAMDGPUDeviceHandle access_dev = 0;

				r = hsaKmtGetAMDGPUDeviceHandleCtx(ctx, attrs[i].value, &access_dev);
				if (r != HSAKMT_STATUS_SUCCESS || access_dev != deviceHandle)
					return HSAKMT_STATUS_INVALID_PARAMETER;
			}
			break;
		case HSA_SVM_ATTR_GRANULARITY:
			drm_attrs[i].type = AMDGPU_SVM_ATTR_GRANULARITY;
			break;
		case HSA_SVM_ATTR_SET_FLAGS:
		case HSA_SVM_ATTR_CLR_FLAGS:
			/* Placeholder; queried separately as full flag vector. */
			drm_attrs[i].type = AMDGPU_SVM_ATTR_COHERENT;
			*need_flags = 1;
			break;
		default:
			return HSAKMT_STATUS_INVALID_PARAMETER;
		}
	}

	*out_device = deviceHandle;
	return HSAKMT_STATUS_SUCCESS;
}

static HSAKMT_STATUS translate_drm_get_result_to_hsa_attr(
					      HsaKFDContext *ctx,
					      HsaAMDGPUDeviceHandle deviceHandle,
					      uint64_t start_addr,
					      uint64_t size,
					      HSA_SVM_ATTRIBUTE *attr,
					      uint32_t drm_attr_type,
					      const struct drm_amdgpu_svm_attribute *query,
					      int *qi,
					      HSAuint32 flags_mask)
{
	HSAKMT_STATUS r;

	if (!ctx || !attr || !qi)
		return HSAKMT_STATUS_INVALID_PARAMETER;

	switch (attr->type) {
	case HSA_SVM_ATTR_PREFERRED_LOC:
	case HSA_SVM_ATTR_PREFETCH_LOC: {
		uint32_t location_value;

		r = get_drm_svm_location_attr(deviceHandle, start_addr, size,
					      drm_attr_type, &location_value);
		if (r != HSAKMT_STATUS_SUCCESS)
			return r;

		switch (location_value) {
		case AMDGPU_SVM_LOCATION_SYSMEM:
			attr->value = 0;
			break;
		case AMDGPU_SVM_LOCATION_UNDEFINED:
			attr->value = INVALID_NODEID;
			break;
		case AMDGPU_SVM_LOCATION_LOCAL:
			/* LOCAL means current query GPU context. */
			r = resolve_node_from_device_handle(ctx, deviceHandle,
							   &attr->value);
			if (r != HSAKMT_STATUS_SUCCESS)
				return r;
			break;
		case AMDGPU_SVM_LOCATION_HIVE: {
			HSAuint32 current_node;

			/* HIVE means a GPU directly reachable from current GPU. */
			r = resolve_node_from_device_handle(ctx, deviceHandle,
							   &current_node);
			if (r != HSAKMT_STATUS_SUCCESS)
				return r;

			r = resolve_hive_target_node(ctx, current_node, &attr->value);
			if (r != HSAKMT_STATUS_SUCCESS)
				return r;
			break;
		}
		default:
			r = hsakmt_gpuid_to_nodeid(ctx, location_value, &attr->value);
			if (r != HSAKMT_STATUS_SUCCESS)
				return r;
		}
		break;
	}
	case HSA_SVM_ATTR_ACCESS:
	case HSA_SVM_ATTR_ACCESS_IN_PLACE:
	case HSA_SVM_ATTR_NO_ACCESS:
		if (!query)
			return HSAKMT_STATUS_INVALID_PARAMETER;
		switch (query[*qi].value) {
		case AMDGPU_SVM_ACCESS_ALLOW_MIGRATE:
			attr->type = HSA_SVM_ATTR_ACCESS;
			break;
		case AMDGPU_SVM_ACCESS_IN_PLACE:
			attr->type = HSA_SVM_ATTR_ACCESS_IN_PLACE;
			break;
		default:
			attr->type = HSA_SVM_ATTR_NO_ACCESS;
			break;
		}
		(*qi)++;
		break;
	case HSA_SVM_ATTR_SET_FLAGS:
		attr->value = flags_mask;
		break;
	case HSA_SVM_ATTR_CLR_FLAGS:
		attr->value = ~flags_mask;
		break;
	case HSA_SVM_ATTR_GRANULARITY:
		if (!query)
			return HSAKMT_STATUS_INVALID_PARAMETER;
		attr->value = query[*qi].value;
		(*qi)++;
		break;
	default:
		break;
	}

	return HSAKMT_STATUS_SUCCESS;
}

static HSAKMT_STATUS
hsaKmtSVMSetAttrCtx_drm(HsaKFDContext *ctx,
		 void *start_addr, HSAuint64 size,
		 unsigned int nattr,
		 HSA_SVM_ATTRIBUTE *attrs)
{
	/* Worst case: each SET_FLAGS/CLR_FLAGS can expand to FLAG_MAP_COUNT attrs */
	struct drm_amdgpu_svm_attribute drm_attrs[nattr + nattr * FLAG_MAP_COUNT];
	HSAKMT_STATUS r;
	HsaAMDGPUDeviceHandle deviceHandle = 0;
	int drm_nattr;

	CHECK_KFD_OPEN();
	CHECK_KFD_MINOR_VERSION(5);

	pr_debug("%s: address 0x%p size 0x%lx\n", __func__, start_addr, size);

	r = validate_svm_range_query_args(start_addr, size, nattr, attrs);
	if (r != HSAKMT_STATUS_SUCCESS)
		return r;

	drm_nattr = hsa_to_drm_attrs(ctx, nattr, attrs, drm_attrs, &deviceHandle);
	if (drm_nattr < 0)
		return HSAKMT_STATUS_INVALID_PARAMETER;

	/* No access attr provided; fall back to the first GPU's device handle */
	if (!deviceHandle) {
		r = hsakmt_fmm_get_default_amdgpu_device_handle(ctx, &deviceHandle);
		if (r != HSAKMT_STATUS_SUCCESS) {
			pr_debug("failed to get default AMDGPU device handle\n");
			return r;
		}
	}

	if (amdgpu_svm_set_attr(deviceHandle, (uint64_t)start_addr, size,
				drm_nattr, drm_attrs)) {
		pr_debug("op set range attrs failed %s\n", strerror(errno));
		return HSAKMT_STATUS_ERROR;
	}

	return HSAKMT_STATUS_SUCCESS;
}

static HSAKMT_STATUS
hsaKmtSVMGetAttrCtx_drm(HsaKFDContext *ctx,
		 void *start_addr, HSAuint64 size,
		 unsigned int nattr,
		 HSA_SVM_ATTRIBUTE *attrs)
{
	struct drm_amdgpu_svm_attribute drm_attrs[nattr];
	HSAKMT_STATUS r;
	HSAuint32 i, j;
	HsaAMDGPUDeviceHandle deviceHandle = 0;
	int need_flags = 0;

	CHECK_KFD_OPEN();
	CHECK_KFD_MINOR_VERSION(5);

	pr_debug("%s: address 0x%p size 0x%lx\n", __func__, start_addr, size);

	r = validate_svm_range_query_args(start_addr, size, nattr, attrs);
	if (r != HSAKMT_STATUS_SUCCESS)
		return r;

	r = hsa_get_query_to_drm_attrs(ctx, nattr, attrs, drm_attrs,
					       &deviceHandle, &need_flags);
	if (r != HSAKMT_STATUS_SUCCESS)
		return r;

	if (!deviceHandle) {
		r = hsakmt_fmm_get_default_amdgpu_device_handle(ctx, &deviceHandle);
		if (r != HSAKMT_STATUS_SUCCESS)
			return r;
	}

	/* Query all boolean flag attrs to reconstruct bitmask if needed */
	HSAuint32 flags_mask = 0;
	if (need_flags) {
		struct drm_amdgpu_svm_attribute fq[FLAG_MAP_COUNT];
		for (j = 0; j < FLAG_MAP_COUNT; j++) {
			fq[j].type = flag_map[j].drm_attr_type;
			fq[j].value = 0;
		}
		if (get_drm_svm_attrs_aggregated(deviceHandle, (uint64_t)start_addr,
						 size, FLAG_MAP_COUNT, fq)) {
			pr_debug("op get flag attrs failed %s\n", strerror(errno));
			return HSAKMT_STATUS_ERROR;
		}
		for (j = 0; j < FLAG_MAP_COUNT; j++) {
			if (fq[j].value)
				flags_mask |= flag_map[j].hsa_flag;
		}
	}

	/* Query non-flag attrs */
	struct drm_amdgpu_svm_attribute query[nattr];
	int qcount = 0;
	for (i = 0; i < nattr; i++) {
		if (attrs[i].type == HSA_SVM_ATTR_SET_FLAGS ||
		    attrs[i].type == HSA_SVM_ATTR_CLR_FLAGS)
			continue;
		if (attrs[i].type == HSA_SVM_ATTR_PREFERRED_LOC ||
		    attrs[i].type == HSA_SVM_ATTR_PREFETCH_LOC)
			continue;
		query[qcount] = drm_attrs[i];
		qcount++;
	}
	if (qcount > 0) {
		if (get_drm_svm_attrs_aggregated(deviceHandle, (uint64_t)start_addr,
						 size, qcount, query)) {
			pr_debug("op get range attrs failed %s\n", strerror(errno));
			return HSAKMT_STATUS_ERROR;
		}
	}

	/* Translate results back to HSA format */
	int qi = 0;
	for (i = 0; i < nattr; i++) {
		r = translate_drm_get_result_to_hsa_attr(ctx, deviceHandle,
							 (uint64_t)start_addr, size,
							 &attrs[i], drm_attrs[i].type,
							 query, &qi, flags_mask);
		if (r != HSAKMT_STATUS_SUCCESS)
			return r;
	}

	return HSAKMT_STATUS_SUCCESS;
}
#endif

/* Helper functions for calling KFD SVM ioctl */

HSAKMT_STATUS HSAKMTAPI
hsaKmtSVMSetAttrCtx(HsaKFDContext *ctx,
		 void *start_addr, HSAuint64 size, unsigned int nattr,
		 HSA_SVM_ATTRIBUTE *attrs)
{
#ifdef USE_DRM_AMDGPU_SVM
	return hsaKmtSVMSetAttrCtx_drm(ctx, start_addr, size, nattr, attrs);
#else
	struct kfd_ioctl_svm_args *args;
	HSAuint64 s_attr;
	HSAKMT_STATUS r;
	HSAuint32 i;

	CHECK_KFD_OPEN();
	CHECK_KFD_MINOR_VERSION(5);

	pr_debug("%s: address 0x%p size 0x%lx\n", __func__, start_addr, size);

	if (!start_addr || !size)
		return HSAKMT_STATUS_INVALID_PARAMETER;
	if ((uint64_t)start_addr & (PAGE_SIZE - 1))
		return HSAKMT_STATUS_INVALID_PARAMETER;
	if (size & (PAGE_SIZE - 1))
		return HSAKMT_STATUS_INVALID_PARAMETER;

	/* Check ioctl size-field limit (14 bits = 16383 bytes max, ~2044 attrs) */
	if (sizeof(*args) + nattr * sizeof(*attrs) > ((1UL << _IOC_SIZEBITS) - 1))
		return HSAKMT_STATUS_INVALID_PARAMETER;

	s_attr = sizeof(*attrs) * nattr;
	args = malloc(sizeof(*args) + s_attr);
	if (!args)
		return HSAKMT_STATUS_NO_MEMORY;

	args->start_addr = (uint64_t)start_addr;
	args->size = size;
	args->op = KFD_IOCTL_SVM_OP_SET_ATTR;
	args->nattr = nattr;
	memcpy(args->attrs, attrs, s_attr);

	for (i = 0; i < nattr; i++) {
		if (attrs[i].type != KFD_IOCTL_SVM_ATTR_PREFERRED_LOC &&
		    attrs[i].type != KFD_IOCTL_SVM_ATTR_PREFETCH_LOC &&
		    attrs[i].type != KFD_IOCTL_SVM_ATTR_ACCESS &&
		    attrs[i].type != KFD_IOCTL_SVM_ATTR_ACCESS_IN_PLACE &&
		    attrs[i].type != KFD_IOCTL_SVM_ATTR_NO_ACCESS)
		    continue;

		if (attrs[i].type == KFD_IOCTL_SVM_ATTR_PREFERRED_LOC &&
		    attrs[i].value == INVALID_NODEID) {
			args->attrs[i].value = KFD_IOCTL_SVM_LOCATION_UNDEFINED;
			continue;
		}
		//attrs[i].value is a node ID, the svm ioctl needs gpu ID
		r = hsakmt_validate_nodeid(ctx, attrs[i].value, &args->attrs[i].value);
		if (r != HSAKMT_STATUS_SUCCESS) {
			pr_debug("invalid node ID: %d\n", attrs[i].value);
			goto out;
		} else if (!args->attrs[i].value &&
			   (attrs[i].type == KFD_IOCTL_SVM_ATTR_ACCESS ||
			    attrs[i].type == KFD_IOCTL_SVM_ATTR_ACCESS_IN_PLACE ||
			    attrs[i].type == KFD_IOCTL_SVM_ATTR_NO_ACCESS)) {
			pr_debug("CPU node invalid for access attribute\n");
			r = HSAKMT_STATUS_INVALID_NODE_UNIT;
			goto out;
		}
	}

	/* Driver does one copy_from_user, with extra attrs size */
	r = hsakmt_ioctl(ctx->fd, AMDKFD_IOC_SVM + (s_attr << _IOC_SIZESHIFT), args);
	if (r) {
		pr_debug("op set range attrs failed %s\n", strerror(errno));
		r = HSAKMT_STATUS_ERROR;
		goto out;
	}

	r = HSAKMT_STATUS_SUCCESS;

out:
	free(args);
	return r;
#endif
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtSVMGetAttrCtx(HsaKFDContext *ctx,
		 void *start_addr, HSAuint64 size, unsigned int nattr,
		 HSA_SVM_ATTRIBUTE *attrs)
{
#ifdef USE_DRM_AMDGPU_SVM
	return hsaKmtSVMGetAttrCtx_drm(ctx, start_addr, size, nattr, attrs);
#else
	struct kfd_ioctl_svm_args *args;
	HSAuint64 s_attr;
	HSAKMT_STATUS r;
	HSAuint32 i;

	CHECK_KFD_OPEN();
	CHECK_KFD_MINOR_VERSION(5);

	pr_debug("%s: address 0x%p size 0x%lx\n", __func__, start_addr, size);

	if (!start_addr || !size)
		return HSAKMT_STATUS_INVALID_PARAMETER;
	if ((uint64_t)start_addr & (PAGE_SIZE - 1))
		return HSAKMT_STATUS_INVALID_PARAMETER;
	if (size & (PAGE_SIZE - 1))
		return HSAKMT_STATUS_INVALID_PARAMETER;

	/* Check ioctl size-field limit (14 bits = 16383 bytes max, ~2044 attrs) */
	if (sizeof(*args) + nattr * sizeof(*attrs) > ((1UL << _IOC_SIZEBITS) - 1))
		return HSAKMT_STATUS_INVALID_PARAMETER;

	s_attr = sizeof(*attrs) * nattr;
	args = malloc(sizeof(*args) + s_attr);
	if (!args)
		return HSAKMT_STATUS_NO_MEMORY;

	args->start_addr = (uint64_t)start_addr;
	args->size = size;
	args->op = KFD_IOCTL_SVM_OP_GET_ATTR;
	args->nattr = nattr;
	memcpy(args->attrs, attrs, s_attr);

	for (i = 0; i < nattr; i++) {
		if (attrs[i].type != KFD_IOCTL_SVM_ATTR_ACCESS &&
		    attrs[i].type != KFD_IOCTL_SVM_ATTR_ACCESS_IN_PLACE &&
		    attrs[i].type != KFD_IOCTL_SVM_ATTR_NO_ACCESS)
		    continue;

		r = hsakmt_validate_nodeid(ctx, attrs[i].value, &args->attrs[i].value);
		if (r != HSAKMT_STATUS_SUCCESS) {
			pr_debug("invalid node ID: %d\n", attrs[i].value);
			goto out;
		} else if (!args->attrs[i].value) {
			pr_debug("CPU node invalid for access attribute\n");
			r = HSAKMT_STATUS_INVALID_NODE_UNIT;
			goto out;
		}
	}

	/* Driver does one copy_from_user, with extra attrs size */
	r = hsakmt_ioctl(ctx->fd, AMDKFD_IOC_SVM + (s_attr << _IOC_SIZESHIFT), args);
	if (r) {
		pr_debug("op get range attrs failed %s\n", strerror(errno));
		r = HSAKMT_STATUS_ERROR;
		goto out;
	}

	memcpy(attrs, args->attrs, s_attr);

	for (i = 0; i < nattr; i++) {
		if (attrs[i].type != KFD_IOCTL_SVM_ATTR_PREFERRED_LOC &&
		    attrs[i].type != KFD_IOCTL_SVM_ATTR_PREFETCH_LOC &&
		    attrs[i].type != KFD_IOCTL_SVM_ATTR_ACCESS &&
		    attrs[i].type != KFD_IOCTL_SVM_ATTR_ACCESS_IN_PLACE &&
		    attrs[i].type != KFD_IOCTL_SVM_ATTR_NO_ACCESS)
			continue;

		switch (attrs[i].value) {
		case KFD_IOCTL_SVM_LOCATION_SYSMEM:
			attrs[i].value = 0;
			break;
		case KFD_IOCTL_SVM_LOCATION_UNDEFINED:
			attrs[i].value = INVALID_NODEID;
			break;
		default:
			r = hsakmt_gpuid_to_nodeid(ctx, attrs[i].value, &attrs[i].value);
			if (r != HSAKMT_STATUS_SUCCESS) {
				pr_debug("invalid GPU ID: %d\n",
					 attrs[i].value);
				goto out;
			}
		}
	}

	r = HSAKMT_STATUS_SUCCESS;

out:
	free(args);
	return r;
#endif
}

static HSAKMT_STATUS
hsaKmtSetGetXNACKModeCtx(HsaKFDContext *ctx, HSAint32 * enable)
{
	struct kfd_ioctl_set_xnack_mode_args args;

	CHECK_KFD_OPEN();
	CHECK_KFD_MINOR_VERSION(5);

	args.xnack_enabled = *enable;

	if (hsakmt_ioctl(ctx->fd, AMDKFD_IOC_SET_XNACK_MODE, &args)) {
		if (errno == EPERM) {
			pr_debug("set mode not supported %s\n",
				 strerror(errno));
			return HSAKMT_STATUS_NOT_SUPPORTED;
		} else if (errno == EBUSY) {
			pr_debug("hsakmt_ioctl queues not empty %s\n",
				 strerror(errno));
		}
		return HSAKMT_STATUS_ERROR;
	}

	*enable = args.xnack_enabled;

	return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtSetXNACKModeCtx(HsaKFDContext *ctx, HSAint32 enable)
{
	return hsaKmtSetGetXNACKModeCtx(ctx, &enable);
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtGetXNACKModeCtx(HsaKFDContext *ctx, HSAint32 * enable)
{
	*enable = -1;
	return hsaKmtSetGetXNACKModeCtx(ctx, enable);
}


HSAKMT_STATUS HSAKMTAPI
hsaKmtSVMSetAttr(void *start_addr, HSAuint64 size, unsigned int nattr,
		 HSA_SVM_ATTRIBUTE *attrs)
{
	return hsaKmtSVMSetAttrCtx(&hsakmt_primary_kfd_ctx, start_addr, size, nattr, attrs);
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtSVMGetAttr(void *start_addr, HSAuint64 size, unsigned int nattr,
		 HSA_SVM_ATTRIBUTE *attrs)
{
	return hsaKmtSVMGetAttrCtx(&hsakmt_primary_kfd_ctx, start_addr, size, nattr, attrs);
}

static HSAKMT_STATUS
hsaKmtSetGetXNACKMode(HSAint32 * enable)
{
	return hsaKmtSetGetXNACKModeCtx(&hsakmt_primary_kfd_ctx, enable);
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtSetXNACKMode(HSAint32 enable)
{
	return hsaKmtSetGetXNACKMode(&enable);
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtGetXNACKMode(HSAint32 * enable)
{
	*enable = -1;
	return hsaKmtSetGetXNACKMode(enable);
}
