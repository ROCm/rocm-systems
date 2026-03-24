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


/* per-fd drm svm ioctl, should be moved to libdrm.
 * This implementation is only used to validate the DRM SVM functionality.
 */

#ifndef __AMDGPU_DRM_SVM_H__
#define __AMDGPU_DRM_SVM_H__

#include "drm.h"

#if defined(__cplusplus)
extern "C" {
#endif

#define DRM_AMDGPU_GEM_SVM              0x1B
#define DRM_IOCTL_AMDGPU_GEM_SVM        DRM_IOWR(DRM_COMMAND_BASE + DRM_AMDGPU_GEM_SVM, struct drm_amdgpu_gem_svm)

/*
 * SVM (Shared Virtual Memory) IOCTL definitions
 * Matches kernel UAPI: include/uapi/drm/amdgpu_drm.h
 */

/* SVM IOCTL operations */
enum amdgpu_ioctl_svm_op {
	AMDGPU_SVM_OP_SET_ATTR = 0,
	AMDGPU_SVM_OP_GET_ATTR = 1,
	AMDGPU_SVM_OP_RESET_ATTR = 2,
};

/* GPU access policy for AMDGPU_SVM_ATTR_ACCESS */
enum amdgpu_ioctl_svm_access {
	AMDGPU_SVM_ACCESS_INACCESSIBLE         = 0,
	AMDGPU_SVM_ACCESS_IN_PLACE             = 1,
	AMDGPU_SVM_ACCESS_ALLOW_MIGRATE        = 2,
};

/* Special values for preferred/prefetch location */
enum amdgpu_ioctl_svm_location {
	AMDGPU_SVM_LOCATION_SYSMEM     = 0,
	AMDGPU_SVM_LOCATION_UNDEFINED  = 0xffffffffU,
};

/* SVM attribute types */
enum amdgpu_ioctl_svm_attr_type {
	AMDGPU_SVM_ATTR_PREFERRED_LOC           = 0,
	AMDGPU_SVM_ATTR_PREFETCH_LOC            = 1,
	AMDGPU_SVM_ATTR_ACCESS                  = 2,
	AMDGPU_SVM_ATTR_GRANULARITY             = 3,
	/* Boolean attributes below: value must be 0 or 1 */
	AMDGPU_SVM_ATTR_HOST_ACCESS             = 4,
	AMDGPU_SVM_ATTR_COHERENT                = 5,
	AMDGPU_SVM_ATTR_EXT_COHERENT            = 6,
	AMDGPU_SVM_ATTR_HIVE_LOCAL              = 7,
	AMDGPU_SVM_ATTR_GPU_RO                  = 8,
	AMDGPU_SVM_ATTR_GPU_EXEC                = 9,
	AMDGPU_SVM_ATTR_GPU_READ_MOSTLY         = 10,
};

/**
 * struct drm_amdgpu_svm_attribute - SVM range attribute
 *
 * @type: Attribute type (AMDGPU_SVM_ATTR_*)
 * @value: Attribute value (interpretation depends on type)
 */
struct drm_amdgpu_svm_attribute {
	__u32 type;
	__u32 value;
};

/**
 * struct drm_amdgpu_gem_svm - AMDGPU SVM IOCTL arguments
 *
 * @start_addr: Start address of the SVM range
 * @size: Size of the SVM range in bytes
 * @operation: AMDGPU_SVM_OP_*
 * @nattr: Number of attributes in the attrs_ptr array
 * @attrs_ptr: User pointer to array of struct drm_amdgpu_svm_attribute
 */
struct drm_amdgpu_gem_svm {
	__u64 start_addr;
	__u64 size;
	__u32 operation;
	__u32 nattr;
	__u64 attrs_ptr;
};

#if defined(__cplusplus)
}
#endif

#endif /* __AMDGPU_DRM_SVM_H__ */
