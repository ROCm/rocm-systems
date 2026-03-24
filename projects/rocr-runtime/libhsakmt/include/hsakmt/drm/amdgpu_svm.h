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

#ifndef __AMDGPU_SVM_H__
#define __AMDGPU_SVM_H__
#include "amdgpu_drm_svm.h"
#include "amdgpu.h"

#ifdef __cplusplus
extern "C" {
#endif

/* per-fd drm svm ioctl, should be moved to libdrm.
 * This is only used to validate the DRM SVM functionality.
 */
int amdgpu_svm_set_attr(amdgpu_device_handle dev,
		 __u64 start_addr,
		 __u64 size, __u32 nattr,
		 struct drm_amdgpu_svm_attribute *attrs);

int amdgpu_svm_get_attr(amdgpu_device_handle dev,
		 __u64 start_addr,
		 __u64 size, __u32 nattr,
		 struct drm_amdgpu_svm_attribute *attrs);

int amdgpu_svm_reset_attr(amdgpu_device_handle dev,
		 __u64 start_addr,
		 __u64 size);

#ifdef __cplusplus
}
#endif

#endif /* __AMDGPU_SVM_H__ */
