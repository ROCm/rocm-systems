/*
 * Copyright © 2026 Advanced Micro Devices, Inc.
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

#include "hsakmt/drm/xf86drm.h"
#include "hsakmt/drm/amdgpu_svm.h"
#include "libhsakmt.h"

/**
 * amdgpu_svm_set_attr - Set SVM attributes
 *
 * @dev: handle to the AMDGPU device
 * @start_addr: start address of the memory range
 * @size: size of the memory range
 * @nattr: number of attributes
 * @attrs: array of attributes
 *
 * This function sets the specified SVM attributes for the given memory
 * range. It can be used to map SVM to specific GPU nodes or to set
 * access permissions.
 */
int amdgpu_svm_set_attr(amdgpu_device_handle dev,
		 __u64 start_addr,
		 __u64 size, __u32 nattr,
		 struct drm_amdgpu_svm_attribute *attrs)
{
	struct drm_amdgpu_gem_svm args = {0};
	__u32 r;

	if (!start_addr || !size)
		return -1;
	if (start_addr & (PAGE_SIZE - 1))
		return -1;
	if (size & (PAGE_SIZE - 1))
		return -1;

	args.start_addr = start_addr;
	args.size = size;
	args.operation = AMDGPU_SVM_OP_SET_ATTR;
	args.nattr = nattr;
	args.attrs_ptr = (__u64)(uintptr_t)attrs;

	r = drmCommandWriteRead(amdgpu_device_get_fd(dev), DRM_AMDGPU_GEM_SVM,
				&args, sizeof(args));
	if (r)
		return r;

	return 0;
}

/**
 * amdgpu_svm_get_attr - Get SVM attributes
 *
 * @start_addr: start address of the memory range
 * @size: size of the memory range
 * @nattr: number of attributes
 * @attrs: array of attributes
 *
 * This function retrieves the specified SVM attributes for the given memory
 * range. It can be used to query the current mapping of SVM to GPU nodes or
 * to check access permissions.
 */
int amdgpu_svm_get_attr(amdgpu_device_handle dev,
		 __u64 start_addr,
		 __u64 size, __u32 nattr,
		 struct drm_amdgpu_svm_attribute *attrs)
{
	struct drm_amdgpu_gem_svm args = {0};
	__u32 r;
	__u32 i;

	if (!start_addr || !size)
		return -1;
	if (start_addr & (PAGE_SIZE - 1))
		return -1;
	if (size & (PAGE_SIZE - 1))
		return -1;

	args.start_addr = start_addr;
	args.size = size;
	args.operation = AMDGPU_SVM_OP_GET_ATTR;
	args.nattr = nattr;
	args.attrs_ptr = (__u64)(uintptr_t)attrs;

	/*
	 * The kernel reads attrs via copy_from_user(attrs_ptr) and
	 * writes results back via copy_to_user(attrs_ptr), so the
	 * caller's attrs array is updated in place.
	 */
	r = drmCommandWriteRead(amdgpu_device_get_fd(dev), DRM_AMDGPU_GEM_SVM,
				&args, sizeof(args));
	if (r)
		return r;

	/*
	 * Post-process location values:
	 *   SYSMEM    (0)          → 0
	 *   UNDEFINED (0xffffffff) → 0xffffffff
	 *   GPU fd    (other)      → pass through as-is
	 */
	for (i = 0; i < nattr; i++) {
		if (attrs[i].type != AMDGPU_SVM_ATTR_PREFERRED_LOC &&
		    attrs[i].type != AMDGPU_SVM_ATTR_PREFETCH_LOC)
			continue;

		switch (attrs[i].value) {
		case AMDGPU_SVM_LOCATION_SYSMEM:
			attrs[i].value = 0;
			break;
		case AMDGPU_SVM_LOCATION_UNDEFINED:
			attrs[i].value = AMDGPU_SVM_LOCATION_UNDEFINED;
			break;
		default:
			/* GPU id – leave as-is */
			break;
		}
	}

	return 0;
}

/**
 * amdgpu_svm_reset_attr - Reset SVM attributes to defaults
 *
 * @dev: handle to the AMDGPU device
 * @start_addr: start address of the memory range
 * @size: size of the memory range
 *
 * This function resets all SVM attributes for the given memory range
 * back to their default values. No attribute array is needed.
 */
int amdgpu_svm_reset_attr(amdgpu_device_handle dev,
		 __u64 start_addr,
		 __u64 size)
{
	struct drm_amdgpu_gem_svm args = {0};

	if (!start_addr || !size)
		return -1;
	if (start_addr & (PAGE_SIZE - 1))
		return -1;
	if (size & (PAGE_SIZE - 1))
		return -1;

	args.start_addr = start_addr;
	args.size = size;
	args.operation = AMDGPU_SVM_OP_RESET_ATTR;
	args.nattr = 0;
	args.attrs_ptr = 0;

	return drmCommandWriteRead(amdgpu_device_get_fd(dev),
				   DRM_AMDGPU_GEM_SVM,
				   &args, sizeof(args));
}
