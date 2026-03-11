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
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <inttypes.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <sys/time.h>
#include "drm/amdgpu_drm.h"


static int svm_is_access_attr(HSAuint32 type)
{
	return type == HSA_SVM_ATTR_ACCESS ||
		type == HSA_SVM_ATTR_ACCESS_IN_PLACE ||
		type == HSA_SVM_ATTR_NO_ACCESS;
}

static int svm_is_location_attr(HSAuint32 type)
{
	return type == HSA_SVM_ATTR_PREFERRED_LOC ||
		type == HSA_SVM_ATTR_PREFETCH_LOC;
}

static HSAKMT_STATUS svm_resolve_gpu_id(HSAuint32 node_or_gpu_id, HSAuint32 *gpu_id)
{
	HSAKMT_STATUS r;
	HSAuint32 node_id;

	r = hsakmt_validate_nodeid(node_or_gpu_id, gpu_id);
	if (r == HSAKMT_STATUS_SUCCESS)
		return HSAKMT_STATUS_SUCCESS;

	r = hsakmt_gpuid_to_nodeid(node_or_gpu_id, &node_id);
	if (r == HSAKMT_STATUS_SUCCESS) {
		*gpu_id = node_or_gpu_id;
		return HSAKMT_STATUS_SUCCESS;
	}

	return r;
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtSVMSetAttr(void *start_addr, HSAuint64 size, unsigned int nattr,
		 HSA_SVM_ATTRIBUTE *attrs)
{
	struct drm_amdgpu_svm_attribute *drm_attrs;
	struct drm_amdgpu_gem_svm args = {0};
	HSAuint64 s_attr;
	HSAKMT_STATUS r;
	HSAuint32 i, gpu_id = 0;
	int drm_fd;

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

	s_attr = sizeof(*drm_attrs) * nattr;
	drm_attrs = alloca(s_attr);

	args.start_addr = (uint64_t)start_addr;
	args.size = size;
	args.operation = AMDGPU_SVM_OP_SET_ATTR;
	args.nattr = nattr;
	args.attrs_ptr = (__u64)(uintptr_t)drm_attrs;

	for (i = 0; i < nattr; i++) {
		drm_attrs[i].type = attrs[i].type;
		drm_attrs[i].value = attrs[i].value;

		if (!svm_is_location_attr(attrs[i].type) &&
		    !svm_is_access_attr(attrs[i].type))
			continue;

		if (attrs[i].type == HSA_SVM_ATTR_PREFERRED_LOC &&
		    attrs[i].value == INVALID_NODEID) {
			drm_attrs[i].value = AMDGPU_SVM_LOCATION_UNDEFINED;
			continue;
		}

		r = svm_resolve_gpu_id(attrs[i].value, &drm_attrs[i].value);
		if (r != HSAKMT_STATUS_SUCCESS) {
			pr_debug("invalid node/GPU ID: %d\n", attrs[i].value);
			return r;
		}

		if (!gpu_id && drm_attrs[i].value)
			gpu_id = drm_attrs[i].value;

		if (!drm_attrs[i].value && svm_is_access_attr(attrs[i].type)) {
			pr_debug("CPU node invalid for access attribute\n");
			return HSAKMT_STATUS_INVALID_NODE_UNIT;
		}
	}

	drm_fd = get_drm_render_fd_by_gpu_id(gpu_id);
	if (drm_fd < 0) {
		pr_debug("failed to get drm render fd for gpu_id 0x%x\n", gpu_id);
		return HSAKMT_STATUS_ERROR;
	}

	if (ioctl(drm_fd, DRM_IOCTL_AMDGPU_GEM_SVM, &args)) {
		pr_debug("op set range attrs failed %s\n", strerror(errno));
		return HSAKMT_STATUS_ERROR;
	}

	return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtSVMGetAttr(void *start_addr, HSAuint64 size, unsigned int nattr,
		 HSA_SVM_ATTRIBUTE *attrs)
{
	struct drm_amdgpu_svm_attribute *drm_attrs;
	struct drm_amdgpu_gem_svm args = {0};
	HSAuint64 s_attr;
	HSAKMT_STATUS r;
	HSAuint32 i, gpu_id = 0;
	int drm_fd;

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

	s_attr = sizeof(*drm_attrs) * nattr;
	drm_attrs = alloca(s_attr);

	args.start_addr = (uint64_t)start_addr;
	args.size = size;
	args.operation = AMDGPU_SVM_OP_GET_ATTR;
	args.nattr = nattr;
	args.attrs_ptr = (__u64)(uintptr_t)drm_attrs;
	if (nattr)
		memcpy(drm_attrs, attrs, s_attr);

	for (i = 0; i < nattr; i++) {
		if (!svm_is_access_attr(attrs[i].type))
		    continue;

		r = svm_resolve_gpu_id(attrs[i].value, &drm_attrs[i].value);
		if (r != HSAKMT_STATUS_SUCCESS) {
			pr_debug("invalid node/GPU ID: %d\n", attrs[i].value);
			return r;
		} else if (!drm_attrs[i].value) {
			pr_debug("CPU node invalid for access attribute\n");
			return HSAKMT_STATUS_INVALID_NODE_UNIT;
		}

		if (!gpu_id)
			gpu_id = drm_attrs[i].value;
	}

	drm_fd = get_drm_render_fd_by_gpu_id(gpu_id);
	if (drm_fd < 0) {
		pr_debug("failed to get drm render fd for gpu_id 0x%x\n", gpu_id);
		return HSAKMT_STATUS_ERROR;
	}

	if (ioctl(drm_fd, DRM_IOCTL_AMDGPU_GEM_SVM, &args)) {
		pr_debug("op get range attrs failed %s\n", strerror(errno));
		return HSAKMT_STATUS_ERROR;
	}

	memcpy(attrs, drm_attrs, s_attr);

	for (i = 0; i < nattr; i++) {
		if (!svm_is_location_attr(attrs[i].type) &&
		    !svm_is_access_attr(attrs[i].type))
			continue;

		switch (attrs[i].value) {
		case AMDGPU_SVM_LOCATION_SYSMEM:
			attrs[i].value = 0;
			break;
		case AMDGPU_SVM_LOCATION_UNDEFINED:
			attrs[i].value = INVALID_NODEID;
			break;
		default:
			r = hsakmt_gpuid_to_nodeid(attrs[i].value, &attrs[i].value);
			if (r != HSAKMT_STATUS_SUCCESS) {
				pr_debug("invalid GPU ID: %d\n",
					 attrs[i].value);
				return r;
			}
		}
	}

	return HSAKMT_STATUS_SUCCESS;
}

static HSAKMT_STATUS
hsaKmtSetGetXNACKMode(HSAint32 * enable)
{
	struct kfd_ioctl_set_xnack_mode_args args;

	CHECK_KFD_OPEN();
	CHECK_KFD_MINOR_VERSION(5);

	args.xnack_enabled = *enable;

	if (hsakmt_ioctl(hsakmt_kfd_fd, AMDKFD_IOC_SET_XNACK_MODE, &args)) {
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
