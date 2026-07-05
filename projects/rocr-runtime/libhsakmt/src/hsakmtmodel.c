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

#include "hsakmt/hsakmtmodel.h"
#include "hsakmt/hsakmt_drm.h"
#include "libhsakmt.h"
#include "hsakmt/hsakmttypes.h"
#include "hsakmt/hsakmtmodeliface.h"
#define _GNU_SOURCE
#define __USE_GNU
#include <assert.h>
#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>
#include <dlfcn.h>
#include <sys/mman.h>
#include <fcntl.h>

bool hsakmt_use_model;
char *hsakmt_model_topology;

static pthread_mutex_t model_call_mutex = PTHREAD_MUTEX_INITIALIZER;
static void *model_library;
static const struct hsakmt_model_functions *model_functions;
static uint32_t model_reported_minor;

HSAKMT_STATUS HSAKMTAPI hsaKmtModelEnabled(bool* enable)
{
	*enable = hsakmt_use_model;
	return HSAKMT_STATUS_SUCCESS;
}

void model_init_env_vars(void)
{
	/* Check whether to use a model instead of real hardware */
	hsakmt_model_topology = getenv("HSA_MODEL_TOPOLOGY");
	if (hsakmt_model_topology)
		hsakmt_use_model = true;
	if (hsakmt_use_model)
	{
		/* Load model library first to get interface functions */
		const char *libname = getenv("HSA_MODEL_LIB");
		if (!libname)
		{
			fprintf(stderr, "model: HSA_MODEL_LIB environment variable must be set to FFM .so\n");
			abort();
		}
		// model_library = dlmopen(LM_ID_NEWLM, libname, RTLD_NOW);
		model_library = dlopen(libname, RTLD_NOW | RTLD_LOCAL);
		if (!model_library)
		{
			fprintf(stderr, "model: failed to load %s: %s\n", libname, dlerror());
			abort();
		}
		get_hsakmt_model_functions_t getter = dlsym(model_library, "get_hsakmt_model_functions");
		if (!getter)
		{
			fprintf(stderr, "model: Failed to get hsakmt_model_functions\n");
			abort();
		}
		model_functions = getter();
		const uint32_t expected_version_major = (uint32_t)HSAKMT_MODEL_INTERFACE_VERSION_MAJOR;
		const uint32_t expected_version_minor = (uint32_t)HSAKMT_MODEL_INTERFACE_VERSION_MINOR;

		if (model_functions->version_major != expected_version_major) {
			fprintf(stderr, "[MODEL] FATAL: Major version mismatch (breaking API change)!\n");
			fprintf(stderr, "[MODEL]   Model file: %s\n", libname);
			fprintf(stderr, "[MODEL]   Model version: %u.%u, expected major %u\n",
				model_functions->version_major, model_functions->version_minor,
				expected_version_major);
			abort();
		}
		model_reported_minor = model_functions->version_minor;
		if (model_reported_minor < expected_version_minor) {
			fprintf(stderr, "[MODEL] WARN: Model version %u.%u < expected %u.%u — "
				"DRM simulation unavailable, using v1.0 compatibility fallbacks\n",
				model_functions->version_major, model_reported_minor,
				expected_version_major, expected_version_minor);
		}

		/* Let FFM create the memfd - it owns sizing and lifecycle.
		 *
		 * As of interface v1.0 this is mandatory for correct model operation.
		 */
		if (!model_functions->create_memfd) {
			fprintf(stderr, "[MODEL] FATAL: Model library does not provide create_memfd (required for v%u.%u)\n",
					HSAKMT_MODEL_INTERFACE_VERSION_MAJOR,
					HSAKMT_MODEL_INTERFACE_VERSION_MINOR);
			abort();
		}

		int fd = model_functions->create_memfd();
		if (fd < 0) {
			fprintf(stderr, "model: FFM failed to create memfd\n");
			abort();
		}

		assert(hsakmt_primary_kfd_ctx.fd < 0);
		hsakmt_kfdcontext_init_context(fd, &hsakmt_primary_kfd_ctx);
	}
}

void model_init(void)
{
	// Don't need to do anything here. This can probably be removed.
}

/* Model implementation of KFD ioctl. */

static int model_kfd_ioctl_dispatch(unsigned long request, void *arg)
{
	assert(_IOC_TYPE(request) == AMDKFD_IOCTL_BASE);

	/* Delegate all KFD IOCTL handling to the model backend, including
	 * AMDKFD_IOC_SVM requests with variable encoded sizes. */
	return model_functions->handle_ioctl(request, arg);
}

static bool is_wait_events_ioctl(unsigned long request)
{
	return (_IOC_TYPE(request) == AMDKFD_IOCTL_BASE) &&
	       (_IOC_NR(request) == _IOC_NR(AMDKFD_IOC_WAIT_EVENTS));
}

/* ── DRM wrapper helpers ─────────────────────────────────────────────────── */

/* Invoke handle_drm_call if the model supports it (v1.1+)
 *
 * Uses model_call_mutex for the same reason as model_kfd_ioctl:
 * DRM calls and KFD ioctls share hsakmt-sim state, so they must be serialized
 * against each other. DRM calls are similarly rare and uncontended. */
static int model_drm_call(unsigned cmd, void *arg)
{
	if (model_reported_minor >= 1 && model_functions->handle_drm_call) {
		pthread_mutex_lock(&model_call_mutex);
		int ret = model_functions->handle_drm_call(cmd, arg);
		pthread_mutex_unlock(&model_call_mutex);
		return ret;
	}
	return 0; /* old FFM (< v1.1): degrade gracefully */
}

/* ── BO operation wrappers ───────────────────────────────────────────────── */

int hsakmt_amdgpu_bo_import(amdgpu_device_handle dev, enum amdgpu_bo_handle_type type,
			     uint32_t shared_handle, struct amdgpu_bo_import_result *output)
{
	if (hsakmt_use_model) {
		if (model_reported_minor >= 1) {
			struct hsakmt_drm_bo_import_args a = {(int)shared_handle, (uint32_t)type, output};
			return model_drm_call(HSAKMT_DRM_BO_IMPORT, &a);
		}
		/* v1.0 fallback: return fd cast to bo_handle as synthetic handle,
		 * replicating the old hsakmt_model_mode_enabled() early-return. */
		output->buf_handle = (amdgpu_bo_handle)(uintptr_t)shared_handle;
		output->alloc_size = 0;
		return 0;
	}
	return amdgpu_bo_import(dev, type, shared_handle, output);
}

int hsakmt_amdgpu_bo_query_info(amdgpu_bo_handle buf_handle, struct amdgpu_bo_info *info)
{
	if (hsakmt_use_model) {
		if (model_reported_minor >= 1) {
			struct hsakmt_drm_bo_query_info_args a = {buf_handle, info};
			return model_drm_call(HSAKMT_DRM_BO_QUERY_INFO, &a);
		}
		/* v1.0 fallback: was inside guarded block, return zeroed info. */
		memset(info, 0, sizeof(*info));
		return 0;
	}
	return amdgpu_bo_query_info(buf_handle, info);
}

int hsakmt_amdgpu_bo_set_metadata(amdgpu_bo_handle buf_handle, struct amdgpu_bo_metadata *info)
{
	if (hsakmt_use_model) {
		if (model_reported_minor >= 1) {
			struct hsakmt_drm_bo_set_metadata_args a = {buf_handle, info};
			return model_drm_call(HSAKMT_DRM_BO_SET_METADATA, &a);
		}
		return 0; /* v1.0 fallback: was guarded no-op */
	}
	return amdgpu_bo_set_metadata(buf_handle, info);
}

int hsakmt_amdgpu_bo_va_op(amdgpu_bo_handle bo, uint64_t offset, uint64_t size,
			    uint64_t addr, uint64_t flags, uint32_t ops)
{
	if (hsakmt_use_model) {
		if (model_reported_minor >= 1) {
			struct hsakmt_drm_bo_va_op_args a = {bo, offset, size, addr, flags, ops};
			return model_drm_call(HSAKMT_DRM_BO_VA_OP, &a);
		}
		return 0; /* v1.0 fallback: was previously unguarded (crash); no-op is better */
	}
	return amdgpu_bo_va_op(bo, offset, size, addr, flags, ops);
}

int hsakmt_amdgpu_bo_free(amdgpu_bo_handle buf_handle)
{
	if (hsakmt_use_model) {
		if (model_reported_minor >= 1) {
			struct hsakmt_drm_bo_free_args a = {buf_handle};
			return model_drm_call(HSAKMT_DRM_BO_FREE, &a);
		}
		return 0; /* v1.0 fallback: was previously unguarded (crash); no-op is better */
	}
	return amdgpu_bo_free(buf_handle);
}

int hsakmt_amdgpu_bo_cpu_map(amdgpu_bo_handle buf_handle, void **cpu)
{
	if (hsakmt_use_model) {
		if (model_reported_minor >= 1) {
			struct hsakmt_drm_bo_cpu_map_args a = {buf_handle, cpu};
			return model_drm_call(HSAKMT_DRM_BO_CPU_MAP, &a);
		}
		return -1; /* v1.0 fallback: was previously unguarded (crash); clean error */
	}
	return amdgpu_bo_cpu_map(buf_handle, cpu);
}

int hsakmt_amdgpu_bo_export(amdgpu_bo_handle buf_handle,
			     enum amdgpu_bo_handle_type type, uint32_t *shared_handle)
{
	int ret;
	if (hsakmt_use_model) {
		if (model_reported_minor >= 1) {
			struct hsakmt_drm_bo_export_args a = {buf_handle, (uint32_t)type, shared_handle};
			ret = model_drm_call(HSAKMT_DRM_BO_EXPORT, &a);
			if (ret == 0 && type == amdgpu_bo_handle_type_dma_buf_fd && *shared_handle > INT32_MAX)
				return -1;
			return ret;
		}
		return -1; /* v1.0 fallback: was previously unguarded (crash); clean error */
	}
	ret = amdgpu_bo_export(buf_handle, type, shared_handle);
	if (ret == 0 && type == amdgpu_bo_handle_type_dma_buf_fd && *shared_handle > INT32_MAX)
		return -1;
	return ret;
}

int hsakmt_drm_command_write_read(int fd, unsigned long drmCommandIndex,
				   void *data, unsigned size)
{
	if (hsakmt_use_model) {
		if (model_reported_minor >= 1) {
			struct hsakmt_drm_cmd_write_read_args a = {fd, drmCommandIndex, data, size};
			return model_drm_call(HSAKMT_DRM_COMMAND_WRITE_READ, &a);
		}
		return -1; /* v1.0 fallback: was previously unguarded (crash); clean error */
	}
	return drmCommandWriteRead(fd, drmCommandIndex, data, size);
}

/* ── Device-level operation wrappers ────────────────────────────────────── */

int hsakmt_drm_open_render(int minor)
{
	if (hsakmt_use_model) {
		if (model_reported_minor >= 1) {
			int fd = -1;
			struct hsakmt_drm_open_render_args a = {minor, &fd};
			return model_drm_call(HSAKMT_DRM_OPEN_RENDER, &a) == 0 ? fd : -1;
		}
		/* v1.0 fallback: old code returned ctx->fd directly; dup so the
		 * eventual hsakmt_drm_close() doesn't destroy the real memfd. */
		return dup(hsakmt_primary_kfd_ctx.fd);
	}
	char path[128];
	sprintf(path, "/dev/dri/renderD%d", minor);
	return open(path, O_RDWR | O_CLOEXEC);
}

void hsakmt_drm_close(int fd)
{
	if (hsakmt_use_model) {
		if (model_reported_minor >= 1) {
			struct hsakmt_drm_close_args a = {fd};
			model_drm_call(HSAKMT_DRM_CLOSE, &a);
			return;
		}
		close(fd); /* v1.0 fallback: close the dup'd fd from hsakmt_drm_open_render */
		return;
	}
	close(fd);
}

int hsakmt_amdgpu_device_initialize(int fd, uint32_t *major_version, uint32_t *minor_version,
				     amdgpu_device_handle *device_handle)
{
	if (hsakmt_use_model) {
		if (model_reported_minor >= 1) {
			struct hsakmt_drm_device_initialize_args a = {fd, major_version, minor_version,
								      (void **)device_handle};
			return model_drm_call(HSAKMT_DRM_DEVICE_INITIALIZE, &a);
		}
		/* v1.0 fallback: old code short-circuited before this call.
		 * Return failure so the caller's if(!initialize) success block
		 * is skipped and device_handle stays NULL. */
		return -1;
	}
	return amdgpu_device_initialize(fd, major_version, minor_version, device_handle);
}

int hsakmt_amdgpu_device_deinitialize(amdgpu_device_handle device_handle)
{
	if (hsakmt_use_model) {
		if (model_reported_minor >= 1) {
			struct hsakmt_drm_device_deinitialize_args a = {device_handle};
			return model_drm_call(HSAKMT_DRM_DEVICE_DEINITIALIZE, &a);
		}
		return 0; /* v1.0 fallback: old code never reached this; no-op */
	}
	return amdgpu_device_deinitialize(device_handle);
}

int hsakmt_amdgpu_device_get_fd(amdgpu_device_handle dev)
{
	if (hsakmt_use_model) {
		if (model_reported_minor >= 1) {
			int fd = -1;
			struct hsakmt_drm_device_get_fd_args a = {dev, &fd};
			model_drm_call(HSAKMT_DRM_DEVICE_GET_FD, &a);
			return fd;
		}
		return -1; /* v1.0 fallback: device_handle is NULL (initialize failed) */
	}
	return amdgpu_device_get_fd(dev);
}

const char *hsakmt_amdgpu_get_marketing_name(amdgpu_device_handle dev)
{
	if (hsakmt_use_model) {
		if (model_reported_minor >= 1) {
			const char *name = NULL;
			struct hsakmt_drm_get_marketing_name_args a = {dev, &name};
			model_drm_call(HSAKMT_DRM_GET_MARKETING_NAME, &a);
			return name;
		}
		return NULL; /* v1.0 fallback: topology caller guard prevented reaching this */
	}
	return amdgpu_get_marketing_name(dev);
}

int hsakmt_amdgpu_query_gpu_info(amdgpu_device_handle dev, struct amdgpu_gpu_info *info)
{
	if (hsakmt_use_model) {
		if (model_reported_minor >= 1) {
			struct hsakmt_drm_query_gpu_info_args a = {dev, info};
			return model_drm_call(HSAKMT_DRM_QUERY_GPU_INFO, &a);
		}
		return -1; /* v1.0 fallback: topology caller guard prevented reaching this */
	}
	return amdgpu_query_gpu_info(dev, info);
}

/* ── KFD IOCTL path ──────────────────────────────────────────────────────── */

int model_kfd_ioctl(unsigned long request, void *arg)
{
	/* WAIT_EVENTS can block for long periods. Holding the global model IOCTL
	 * mutex across a blocking wait prevents other threads from issuing IOCTLs
	 * like SET_EVENT that are required to wake the wait, which can deadlock
	 * user-space event tests under the model.
	 *
	 * Keep the conservative serialization for all other IOCTLs.
	 */
	if (is_wait_events_ioctl(request))
		return model_kfd_ioctl_dispatch(request, arg);

	/* Use a very simple locking strategy for correctness. IOCTLs should
	 * be rare anyway and not contended considering the cost of running
	 * the model itself.
	 *
	 * The bulk of model execution happens in a separate thread *without*
	 * holding the IOCTL mutex. */
	pthread_mutex_lock(&model_call_mutex);
	int ret = model_kfd_ioctl_dispatch(request, arg);
	pthread_mutex_unlock(&model_call_mutex);
	return ret;
}
