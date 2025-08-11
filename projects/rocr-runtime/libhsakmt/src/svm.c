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
#include <sys/time.h>
#include <errno.h>

/* Helper functions for calling KFD SVM ioctl */

HSAKMT_STATUS HSAKMTAPI
hsaKmtSVMSetAttr(void *start_addr, HSAuint64 size, unsigned int nattr,
		 HSA_SVM_ATTRIBUTE *attrs)
{
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

		r = hsakmt_validate_nodeid(attrs[i].value, &args->attrs[i].value);
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
	r = hsakmt_ioctl(hsakmt_kfd_fd, AMDKFD_IOC_SVM + (s_attr << _IOC_SIZESHIFT), args);
	if (r) {
		pr_debug("op set range attrs failed %s\n", strerror(errno));
		return HSAKMT_STATUS_ERROR;
	}

	return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtSVMGetAttr(void *start_addr, HSAuint64 size, unsigned int nattr,
		 HSA_SVM_ATTRIBUTE *attrs)
{
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

		r = hsakmt_validate_nodeid(attrs[i].value, &args->attrs[i].value);
		if (r != HSAKMT_STATUS_SUCCESS) {
			pr_debug("invalid node ID: %d\n", attrs[i].value);
			return r;
		} else if (!args->attrs[i].value) {
			pr_debug("CPU node invalid for access attribute\n");
			return HSAKMT_STATUS_INVALID_NODE_UNIT;
		}
	}

	/* Driver does one copy_from_user, with extra attrs size */
	r = hsakmt_ioctl(hsakmt_kfd_fd, AMDKFD_IOC_SVM + (s_attr << _IOC_SIZESHIFT), args);
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
