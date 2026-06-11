/*
 * Copyright © 2025 Advanced Micro Devices, Inc.
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

#ifndef _HSAKMTMODEL_H_
#define _HSAKMTMODEL_H_
#include <stdbool.h>
#include <stdint.h>
#include <amdgpu.h>

extern bool hsakmt_use_model;
extern char *hsakmt_model_topology;
void model_init_env_vars(void);
void model_init(void);
int model_kfd_ioctl(unsigned long request, void *arg);

/* DRM/amdgpu wrappers — route to model in model mode, real libdrm otherwise. */

/* BO operations */
int hsakmt_amdgpu_bo_import(amdgpu_device_handle dev, enum amdgpu_bo_handle_type type,
			     uint32_t shared_handle, struct amdgpu_bo_import_result *output);
int hsakmt_amdgpu_bo_query_info(amdgpu_bo_handle buf_handle, struct amdgpu_bo_info *info);
int hsakmt_amdgpu_bo_set_metadata(amdgpu_bo_handle buf_handle, struct amdgpu_bo_metadata *info);
int hsakmt_amdgpu_bo_va_op(amdgpu_bo_handle bo, uint64_t offset, uint64_t size,
			    uint64_t addr, uint64_t flags, uint32_t ops);
int hsakmt_amdgpu_bo_free(amdgpu_bo_handle buf_handle);
int hsakmt_amdgpu_bo_cpu_map(amdgpu_bo_handle buf_handle, void **cpu);
int hsakmt_amdgpu_bo_export(amdgpu_bo_handle buf_handle,
			     enum amdgpu_bo_handle_type type, uint32_t *shared_handle);
int hsakmt_drm_command_write_read(int fd, unsigned long drmCommandIndex,
				   void *data, unsigned size);

/* Device-level operations */
int hsakmt_drm_open_render(int minor);
void hsakmt_drm_close(int fd);
int hsakmt_amdgpu_device_initialize(int fd, uint32_t *major_version, uint32_t *minor_version,
				     amdgpu_device_handle *device_handle);
int hsakmt_amdgpu_device_deinitialize(amdgpu_device_handle device_handle);
int hsakmt_amdgpu_device_get_fd(amdgpu_device_handle dev);
const char *hsakmt_amdgpu_get_marketing_name(amdgpu_device_handle dev);
int hsakmt_amdgpu_query_gpu_info(amdgpu_device_handle dev, struct amdgpu_gpu_info *info);

#endif /* _HSAKMTMODEL_H_ */