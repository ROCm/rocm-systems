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

#ifndef _HSAKMTMODELIFACE_H_
#define _HSAKMTMODELIFACE_H_

#include <inttypes.h>
#include <stdbool.h>

// Changelog:
//  1.0: Breaking ABI cleanup: remove unused entry points (create/destroy/
//       set_global_aperture/init_from_topology). Model interface is now
//       create_memfd + handle_ioctl.
//  1.1: Add handle_drm_call for DRM/amdgpu simulation. Old clients (1.0)
//       that lack this field will have no-op success for all DRM calls.
#define HSAKMT_MODEL_INTERFACE_VERSION_MAJOR 1
#define HSAKMT_MODEL_INTERFACE_VERSION_MINOR 1

typedef struct hsakmt_model hsakmt_model_t;
typedef struct hsakmt_model_queue hsakmt_model_queue_t;

// Pointer to a "set event" function.
//
// data is a user-provided opaque pointer.
// event_id is the ID of the event to set (as in amd_signal_s::event_id).
typedef void (*hsakmt_model_set_event_fn)(void *data, unsigned event_id);

// Interface provided by the software model implementation.
//
// Queried from a shared library by calling an export called
// `get_hsakmt_model_functions`
//
// Interface versioning follows the semantic versioning model: clients that
// know about interface version X.Y can use any implementation that provides
// version X.Z with Z >= Y.
//
// The model is designed to support only one VMID space.
struct hsakmt_model_functions {
	uint32_t version_major; // HSAKMT_MODEL_INTERFACE_VERSION_MAJOR
	uint32_t version_minor; // HSAKMT_MODEL_INTERFACE_VERSION_MINOR

	// Create memfd for model use (v0.7+)
	// FFM owns memfd creation and sizing logic
	// Returns: File descriptor on success, -1 on error (errno set)
	int (*create_memfd)(void);

	// Unified IOCTL handler - FFM owns all IOCTL dispatch logic
	// Returns 0 on success, -1 on error (with errno set)
	int (*handle_ioctl)(unsigned long request, void *arg);

	// DRM/amdgpu call simulation (v1.1+)
	// Routes libdrm/amdgpu API calls through the model.
	// NULL in v1.0 models — callers must degrade gracefully.
	// Returns 0 on success, -1 on error (with errno set)
	int (*handle_drm_call)(unsigned cmd, void *arg);
};

// Commands for handle_drm_call (v1.1+)
enum hsakmt_drm_cmd {
	// BO operations
	HSAKMT_DRM_BO_VA_OP,
	HSAKMT_DRM_BO_FREE,
	HSAKMT_DRM_BO_IMPORT,
	HSAKMT_DRM_BO_EXPORT,
	HSAKMT_DRM_BO_CPU_MAP,
	HSAKMT_DRM_BO_QUERY_INFO,
	HSAKMT_DRM_BO_SET_METADATA,
	HSAKMT_DRM_COMMAND_WRITE_READ,
	// Device-level operations
	HSAKMT_DRM_OPEN_RENDER,
	HSAKMT_DRM_CLOSE,
	HSAKMT_DRM_DEVICE_INITIALIZE,
	HSAKMT_DRM_DEVICE_DEINITIALIZE,
	HSAKMT_DRM_DEVICE_GET_FD,
	HSAKMT_DRM_GET_MARKETING_NAME,
	HSAKMT_DRM_QUERY_GPU_INFO,
};

// Arg structs for each hsakmt_drm_cmd. void *bo / void *dev are opaque
// handles whose meaning is private to the model implementation.
struct hsakmt_drm_bo_va_op_args {
	void    *bo;
	uint64_t offset;
	uint64_t size;
	uint64_t addr;
	uint64_t flags;
	uint32_t ops;
};
struct hsakmt_drm_bo_free_args         { void *bo; };
struct hsakmt_drm_bo_import_args       { int fd; uint32_t type; void *result_out; }; // amdgpu_bo_import_result*
struct hsakmt_drm_bo_export_args       { void *bo; uint32_t type; uint32_t *handle_out; };
struct hsakmt_drm_bo_cpu_map_args      { void *bo; void **cpu_ptr_out; };
struct hsakmt_drm_bo_query_info_args   { void *bo; void *info_out; };   // amdgpu_bo_info*
struct hsakmt_drm_bo_set_metadata_args { void *bo; void *metadata; };   // amdgpu_bo_metadata*
struct hsakmt_drm_cmd_write_read_args  { int fd; unsigned long cmd; void *data; unsigned size; };
struct hsakmt_drm_open_render_args     { int minor; int *fd_out; };
struct hsakmt_drm_close_args           { int fd; };
struct hsakmt_drm_device_initialize_args {
	int       fd;
	uint32_t *major_out;
	uint32_t *minor_out;
	void    **dev_out;   // amdgpu_device_handle*
};
struct hsakmt_drm_device_deinitialize_args { void *dev; };
struct hsakmt_drm_device_get_fd_args       { void *dev; int *fd_out; };
struct hsakmt_drm_get_marketing_name_args  { void *dev; const char **name_out; };
struct hsakmt_drm_query_gpu_info_args      { void *dev; void *info_out; }; // amdgpu_gpu_info*

// Type of a shared library export called `get_hsakmt_model_functions`.
typedef const struct hsakmt_model_functions *(*get_hsakmt_model_functions_t)(void);

#endif // _HSAKMTMODELIFACE_H_