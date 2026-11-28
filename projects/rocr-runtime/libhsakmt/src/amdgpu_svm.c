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

static int validate_page_aligned_range(uint64_t start_addr, uint64_t size)
{
	static long page_size;

	if (!page_size)
		page_size = PAGE_SIZE;

	if (!start_addr || !size)
		return -EINVAL;
	if (start_addr & (page_size - 1))
		return -EINVAL;
	if (size & (page_size - 1))
		return -EINVAL;

	return 0;
}

/**
 * amdgpu_svm_set_attr - Set SVM attributes
 *
 * @dev: handle to the AMDGPU device
 * @start_addr: start address, non-zero and page aligned
 * @size: size of the range in bytes, non-zero and page aligned
 * @nattr: number of entries in @attrs
 * @attrs: array of (type, value) pairs to set
 *
 * Return: 0 on success, negative POSIX error code on failure.
 */
int amdgpu_svm_set_attr(amdgpu_device_handle dev,
		 uint64_t start_addr,
		 uint64_t size, uint32_t nattr,
		 struct drm_amdgpu_svm_attribute *attrs)
{
	struct drm_amdgpu_gem_svm args = {0};
	int r;

	r = validate_page_aligned_range(start_addr, size);
	if (r)
		return r;

	args.start_addr = start_addr;
	args.size = size;
	args.operation = AMDGPU_SVM_OP_SET_ATTR;
	args.nattr = nattr;
	args.attrs_ptr = (uint64_t)(uintptr_t)attrs;

	return drmCommandWriteRead(amdgpu_device_get_fd(dev),
				  DRM_AMDGPU_GEM_SVM,
				  &args, sizeof(args));
}

/**
 * amdgpu_svm_get_attr - Get SVM attributes
 *
 * @dev: handle to the AMDGPU device
 * @start_addr: start address, non-zero and page aligned
 * @size: upper bound of the range in bytes, non-zero and page aligned
 * @nattr: number of entries in @attrs
 * @attrs: .type selects the attribute to query (in); .value receives its
 *         value (out)
 * @out_size: optional, may be NULL. On success, the values in @attrs hold for
 *            [@start_addr, @start_addr + *@out_size), with *@out_size <= @size.
 *            Untouched on failure.
 *
 * Return: 0 on success, negative POSIX error code on failure.
 */
int amdgpu_svm_get_attr(amdgpu_device_handle dev,
		 uint64_t start_addr,
		 uint64_t size, uint32_t nattr,
		 struct drm_amdgpu_svm_attribute *attrs,
		 uint64_t *out_size)
{
	struct drm_amdgpu_gem_svm args = {0};
	int r;

	r = validate_page_aligned_range(start_addr, size);
	if (r)
		return r;

	args.start_addr = start_addr;
	args.size = size;
	args.operation = AMDGPU_SVM_OP_GET_ATTR;
	args.nattr = nattr;
	args.attrs_ptr = (uint64_t)(uintptr_t)attrs;

	r = drmCommandWriteRead(amdgpu_device_get_fd(dev),
			       DRM_AMDGPU_GEM_SVM,
			       &args, sizeof(args));
	if (!r && out_size)
		*out_size = args.size;

	return r;
}

/**
 * amdgpu_svm_reset_attr - Reset SVM attributes to defaults
 *
 * @dev: handle to the AMDGPU device
 * @start_addr: start address, non-zero and page aligned
 * @size: size of the range in bytes, non-zero and page aligned
 *
 * Return: 0 on success, negative POSIX error code on failure.
 */
int amdgpu_svm_reset_attr(amdgpu_device_handle dev,
		 uint64_t start_addr,
		 uint64_t size)
{
	struct drm_amdgpu_gem_svm args = {0};
	int r;

	r = validate_page_aligned_range(start_addr, size);
	if (r)
		return r;

	args.start_addr = start_addr;
	args.size = size;
	args.operation = AMDGPU_SVM_OP_RESET_ATTR;
	args.nattr = 0;
	args.attrs_ptr = 0;

	return drmCommandWriteRead(amdgpu_device_get_fd(dev),
				   DRM_AMDGPU_GEM_SVM,
				   &args, sizeof(args));
}
