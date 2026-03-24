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
#include <string.h>
#include <errno.h>
#include <alloca.h>
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
			out_attrs[out_count].type = AMDGPU_SVM_ATTR_PREFERRED_LOC;
			if (attrs[i].value == INVALID_NODEID) {
				out_attrs[out_count].value = AMDGPU_SVM_LOCATION_UNDEFINED;
			} else {
				r = hsakmt_validate_nodeid(ctx, attrs[i].value,
							  &out_attrs[out_count].value);
				if (r != HSAKMT_STATUS_SUCCESS)
					return -1;
			}
			out_count++;
			break;

		case HSA_SVM_ATTR_PREFETCH_LOC:
			out_attrs[out_count].type = AMDGPU_SVM_ATTR_PREFETCH_LOC;
			if (attrs[i].value == INVALID_NODEID) {
				out_attrs[out_count].value = AMDGPU_SVM_LOCATION_UNDEFINED;
			} else {
				r = hsakmt_validate_nodeid(ctx, attrs[i].value,
							  &out_attrs[out_count].value);
				if (r != HSAKMT_STATUS_SUCCESS)
					return -1;
			}
			out_count++;
			break;

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

	if (!start_addr || !size)
		return HSAKMT_STATUS_INVALID_PARAMETER;
	if ((uint64_t)start_addr & (PAGE_SIZE - 1))
		return HSAKMT_STATUS_INVALID_PARAMETER;
	if (size & (PAGE_SIZE - 1))
		return HSAKMT_STATUS_INVALID_PARAMETER;
	if (nattr && !attrs)
		return HSAKMT_STATUS_INVALID_PARAMETER;

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

	CHECK_KFD_OPEN();
	CHECK_KFD_MINOR_VERSION(5);

	pr_debug("%s: address 0x%p size 0x%lx\n", __func__, start_addr, size);

	if (!start_addr || !size)
		return HSAKMT_STATUS_INVALID_PARAMETER;
	if ((uint64_t)start_addr & (PAGE_SIZE - 1))
		return HSAKMT_STATUS_INVALID_PARAMETER;
	if (size & (PAGE_SIZE - 1))
		return HSAKMT_STATUS_INVALID_PARAMETER;
	if (nattr && !attrs)
		return HSAKMT_STATUS_INVALID_PARAMETER;

	/* Translate HSA query types to DRM types */
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
			}
			break;
		case HSA_SVM_ATTR_GRANULARITY:
			drm_attrs[i].type = AMDGPU_SVM_ATTR_GRANULARITY;
			break;
		case HSA_SVM_ATTR_SET_FLAGS:
		case HSA_SVM_ATTR_CLR_FLAGS:
			/* Placeholder; will handle separately below */
			drm_attrs[i].type = AMDGPU_SVM_ATTR_COHERENT;
			break;
		default:
			return HSAKMT_STATUS_INVALID_PARAMETER;
		}
	}

	if (!deviceHandle) {
		r = hsakmt_fmm_get_default_amdgpu_device_handle(ctx, &deviceHandle);
		if (r != HSAKMT_STATUS_SUCCESS)
			return r;
	}

	/* Query all boolean flag attrs to reconstruct bitmask if needed */
	HSAuint32 flags_mask = 0;
	int need_flags = 0;
	for (i = 0; i < nattr; i++) {
		if (attrs[i].type == HSA_SVM_ATTR_SET_FLAGS ||
		    attrs[i].type == HSA_SVM_ATTR_CLR_FLAGS) {
			need_flags = 1;
			break;
		}
	}
	if (need_flags) {
		struct drm_amdgpu_svm_attribute fq[FLAG_MAP_COUNT];
		for (j = 0; j < FLAG_MAP_COUNT; j++) {
			fq[j].type = flag_map[j].drm_attr_type;
			fq[j].value = 0;
		}
		if (amdgpu_svm_get_attr(deviceHandle, (uint64_t)start_addr,
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
		query[qcount] = drm_attrs[i];
		qcount++;
	}
	if (qcount > 0) {
		if (amdgpu_svm_get_attr(deviceHandle, (uint64_t)start_addr,
					size, qcount, query)) {
			pr_debug("op get range attrs failed %s\n", strerror(errno));
			return HSAKMT_STATUS_ERROR;
		}
	}

	/* Translate results back to HSA format */
	int qi = 0;
	for (i = 0; i < nattr; i++) {
		switch (attrs[i].type) {
		case HSA_SVM_ATTR_PREFERRED_LOC:
		case HSA_SVM_ATTR_PREFETCH_LOC:
			switch (query[qi].value) {
			case AMDGPU_SVM_LOCATION_SYSMEM:
				attrs[i].value = 0;
				break;
			case AMDGPU_SVM_LOCATION_UNDEFINED:
				attrs[i].value = INVALID_NODEID;
				break;
			default:
				r = hsakmt_gpuid_to_nodeid(ctx, query[qi].value,
							   &attrs[i].value);
				if (r != HSAKMT_STATUS_SUCCESS)
					return r;
			}
			qi++;
			break;
		case HSA_SVM_ATTR_ACCESS:
		case HSA_SVM_ATTR_ACCESS_IN_PLACE:
		case HSA_SVM_ATTR_NO_ACCESS:
			switch (query[qi].value) {
			case AMDGPU_SVM_ACCESS_ALLOW_MIGRATE:
				attrs[i].type = HSA_SVM_ATTR_ACCESS;
				break;
			case AMDGPU_SVM_ACCESS_IN_PLACE:
				attrs[i].type = HSA_SVM_ATTR_ACCESS_IN_PLACE;
				break;
			default:
				attrs[i].type = HSA_SVM_ATTR_NO_ACCESS;
				break;
			}
			qi++;
			break;
		case HSA_SVM_ATTR_SET_FLAGS:
			attrs[i].value = flags_mask;
			break;
		case HSA_SVM_ATTR_CLR_FLAGS:
			attrs[i].value = ~flags_mask;
			break;
		case HSA_SVM_ATTR_GRANULARITY:
			attrs[i].value = query[qi].value;
			qi++;
			break;
		default:
			break;
		}
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

	s_attr = sizeof(*attrs) * nattr;
	args = alloca(sizeof(*args) + s_attr);

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
			return r;
		} else if (!args->attrs[i].value &&
			   (attrs[i].type == KFD_IOCTL_SVM_ATTR_ACCESS ||
			    attrs[i].type == KFD_IOCTL_SVM_ATTR_ACCESS_IN_PLACE ||
			    attrs[i].type == KFD_IOCTL_SVM_ATTR_NO_ACCESS)) {
			pr_debug("CPU node invalid for access attribute\n");
			return HSAKMT_STATUS_INVALID_NODE_UNIT;
		}
	}

	/* Driver does one copy_from_user, with extra attrs size */
	r = hsakmt_ioctl(ctx->fd, AMDKFD_IOC_SVM + (s_attr << _IOC_SIZESHIFT), args);
	if (r) {
		pr_debug("op set range attrs failed %s\n", strerror(errno));
		return HSAKMT_STATUS_ERROR;
	}

	return HSAKMT_STATUS_SUCCESS;
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

	s_attr = sizeof(*attrs) * nattr;
	args = alloca(sizeof(*args) + s_attr);

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
			return r;
		} else if (!args->attrs[i].value) {
			pr_debug("CPU node invalid for access attribute\n");
			return HSAKMT_STATUS_INVALID_NODE_UNIT;
		}
	}

	/* Driver does one copy_from_user, with extra attrs size */
	r = hsakmt_ioctl(ctx->fd, AMDKFD_IOC_SVM + (s_attr << _IOC_SIZESHIFT), args);
	if (r) {
		pr_debug("op get range attrs failed %s\n", strerror(errno));
		return HSAKMT_STATUS_ERROR;
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
				return r;
			}
		}
	}

	return HSAKMT_STATUS_SUCCESS;
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
