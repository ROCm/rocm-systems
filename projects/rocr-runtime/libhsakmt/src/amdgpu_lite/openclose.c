/*
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
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
#include "amdgpu_lite_device.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>

static const char amdgpu_lite_device_name[] = "/dev/amdgpu_lite0";

/* pthread_atfork handlers for fork detection.
 * CHECK_KFD_OPEN() checks hsakmt_forked directly, so we must
 * set it in the child fork handler.
 */
static void prepare_fork_handler(void)
{
	pthread_mutex_lock(&hsakmt_mutex);
}
static void parent_fork_handler(void)
{
	pthread_mutex_unlock(&hsakmt_mutex);
}
static void child_fork_handler(void)
{
	pthread_mutex_init(&hsakmt_mutex, NULL);
	hsakmt_forked = true;
}

static bool atfork_installed;

/*
 * Map a BAR by index using the MAP_BAR ioctl + mmap.
 * Returns the mapped pointer, or MAP_FAILED on error.
 */
static void *map_bar(struct amdgpu_lite_device *dev, uint32_t bar_idx,
		     size_t *out_size)
{
	struct amdgpu_lite_bar_info *bar;
	struct amdgpu_lite_map_bar map_args;
	void *ptr;

	if (bar_idx >= dev->info.num_bars)
		return MAP_FAILED;

	bar = &dev->info.bars[bar_idx];
	if (bar->size == 0)
		return MAP_FAILED;

	memset(&map_args, 0, sizeof(map_args));
	map_args.bar_index = bar_idx;
	map_args.offset = 0;
	map_args.size = 0; /* 0 = entire BAR */

	if (amdgpu_lite_ioctl_map_bar(dev->fd, &map_args) == -1)
		return MAP_FAILED;

	ptr = mmap(NULL, bar->size, PROT_READ | PROT_WRITE, MAP_SHARED,
		   dev->fd, map_args.mmap_offset);

	if (ptr != MAP_FAILED && out_size)
		*out_size = bar->size;

	return ptr;
}

/*
 * Derive the GFX version from the device ID.
 * For now we only know the RX 9070 XT (0x7551) = GFX12.0.1.
 * Returns gfxv in packed format: (major << 16) | (minor << 8) | stepping.
 */
static uint32_t device_id_to_gfx_version(uint16_t device_id)
{
	switch (device_id) {
	case 0x7551: /* RX 9070 XT */
	case 0x7550: /* RX 9070 */
		return 0x0C0001; /* GFX12.0.1 */
	default:
		return 0;
	}
}

int amdgpu_lite_open(struct amdgpu_lite_device *dev)
{
	int fd;

	if (dev->is_open)
		return 0;

	memset(dev, 0, sizeof(*dev));
	dev->fd = -1;

	fd = open(amdgpu_lite_device_name, O_RDWR | O_CLOEXEC);
	if (fd < 0) {
		pr_err("amdgpu_lite: failed to open %s: %s\n",
		       amdgpu_lite_device_name, strerror(errno));
		return -1;
	}
	dev->fd = fd;

	/* Query device info */
	memset(&dev->info, 0, sizeof(dev->info));
	if (amdgpu_lite_ioctl_get_info(fd, &dev->info) == -1) {
		pr_err("amdgpu_lite: GET_INFO ioctl failed: %s\n",
		       strerror(errno));
		goto fail;
	}

	dev->device_id = dev->info.device_id;
	dev->vram_size = dev->info.vram_size;
	dev->gfx_version = device_id_to_gfx_version(dev->device_id);

	pr_info("amdgpu_lite: device %04x:%04x, VRAM %llu MB, gfxv 0x%06x\n",
		dev->info.vendor_id, dev->device_id,
		(unsigned long long)dev->vram_size / (1024 * 1024),
		dev->gfx_version);

	/* Map MMIO BAR */
	dev->mmio_bar = map_bar(dev, dev->info.mmio_bar_index,
				&dev->mmio_bar_size);
	if (dev->mmio_bar == MAP_FAILED) {
		pr_err("amdgpu_lite: failed to map MMIO BAR %u: %s\n",
		       dev->info.mmio_bar_index, strerror(errno));
		dev->mmio_bar = NULL;
		goto fail;
	}
	pr_info("amdgpu_lite: MMIO BAR mapped at %p, size %zu\n",
		dev->mmio_bar, dev->mmio_bar_size);

	/* Map doorbell BAR */
	dev->doorbell_bar = map_bar(dev, dev->info.doorbell_bar_index,
				    &dev->doorbell_bar_size);
	if (dev->doorbell_bar == MAP_FAILED) {
		pr_warn("amdgpu_lite: failed to map doorbell BAR %u: %s\n",
			dev->info.doorbell_bar_index, strerror(errno));
		dev->doorbell_bar = NULL;
		/* Not fatal - doorbells needed later for queues */
	}

	/* Map VRAM BAR (optional, only if resizable BAR is enabled) */
	dev->vram_bar = map_bar(dev, dev->info.vram_bar_index,
				&dev->vram_bar_size);
	if (dev->vram_bar == MAP_FAILED) {
		pr_info("amdgpu_lite: VRAM BAR not mappable (no rBAR?)\n");
		dev->vram_bar = NULL;
		dev->vram_bar_size = 0;
	} else {
		pr_info("amdgpu_lite: VRAM BAR mapped at %p, size %zu MB\n",
			dev->vram_bar,
			dev->vram_bar_size / (1024 * 1024));
	}

	dev->is_open = true;
	return 0;

fail:
	if (dev->mmio_bar && dev->mmio_bar != MAP_FAILED) {
		munmap(dev->mmio_bar, dev->mmio_bar_size);
		dev->mmio_bar = NULL;
	}
	close(fd);
	dev->fd = -1;
	return -1;
}

void amdgpu_lite_close(struct amdgpu_lite_device *dev)
{
	if (!dev->is_open)
		return;

	if (dev->vram_bar) {
		munmap(dev->vram_bar, dev->vram_bar_size);
		dev->vram_bar = NULL;
	}
	if (dev->doorbell_bar) {
		munmap(dev->doorbell_bar, dev->doorbell_bar_size);
		dev->doorbell_bar = NULL;
	}
	if (dev->mmio_bar) {
		munmap(dev->mmio_bar, dev->mmio_bar_size);
		dev->mmio_bar = NULL;
	}
	if (dev->fd >= 0) {
		close(dev->fd);
		dev->fd = -1;
	}

	dev->is_open = false;
}

/*
 * Initialize debug level from environment.
 */
static void init_debug_level(void)
{
	char *envvar;
	int level;

	hsakmt_debug_level = HSAKMT_DEBUG_LEVEL_DEFAULT;
	envvar = getenv("HSAKMT_DEBUG_LEVEL");
	if (envvar) {
		level = atoi(envvar);
		if (level >= HSAKMT_DEBUG_LEVEL_ERR &&
		    level <= HSAKMT_DEBUG_LEVEL_DEBUG)
			hsakmt_debug_level = level;
	}
}

static inline void init_page_size(void)
{
	hsakmt_page_size = sysconf(_SC_PAGESIZE);
	hsakmt_page_shift = ffs(hsakmt_page_size) - 1;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtOpenKFDCtx(HsaKFDContext **pCtx)
{
	HSAKMT_STATUS result;

	pthread_mutex_lock(&hsakmt_mutex);

	if (hsakmt_kfd_open_count == 0) {
		init_debug_level();
		init_page_size();

		if (!atfork_installed) {
			pthread_atfork(prepare_fork_handler,
				       parent_fork_handler,
				       child_fork_handler);
			atfork_installed = true;
		}

		if (amdgpu_lite_open(&g_amdgpu_lite_dev) != 0) {
			result = HSAKMT_STATUS_KERNEL_IO_CHANNEL_NOT_OPENED;
			goto open_failed;
		}

		/* Store the fd in the primary context for CHECK_KFD_OPEN */
		hsakmt_primary_kfd_ctx.fd = g_amdgpu_lite_dev.fd;

		result = hsakmt_init_kfd_version();
		if (result != HSAKMT_STATUS_SUCCESS)
			goto version_failed;

		hsakmt_is_dgpu = true;
		hsakmt_kfd_open_count = 1;
		*pCtx = &hsakmt_primary_kfd_ctx;
	} else {
		hsakmt_kfd_open_count++;
		*pCtx = &hsakmt_primary_kfd_ctx;
		result = HSAKMT_STATUS_KERNEL_ALREADY_OPENED;
	}

	pthread_mutex_unlock(&hsakmt_mutex);
	return result;

version_failed:
	amdgpu_lite_close(&g_amdgpu_lite_dev);
	hsakmt_primary_kfd_ctx.fd = -1;
open_failed:
	pthread_mutex_unlock(&hsakmt_mutex);
	return result;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtCloseKFDCtx(void)
{
	HSAKMT_STATUS result;

	pthread_mutex_lock(&hsakmt_mutex);

	if (hsakmt_kfd_open_count > 0) {
		if (--hsakmt_kfd_open_count == 0) {
			amdgpu_lite_close(&g_amdgpu_lite_dev);
			hsakmt_primary_kfd_ctx.fd = -1;
		}
		result = HSAKMT_STATUS_SUCCESS;
	} else {
		result = HSAKMT_STATUS_KERNEL_IO_CHANNEL_NOT_OPENED;
	}

	pthread_mutex_unlock(&hsakmt_mutex);
	return result;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtOpenKFD(void)
{
	HsaKFDContext *pCtx = NULL;
	return hsaKmtOpenKFDCtx(&pCtx);
}

HSAKMT_STATUS HSAKMTAPI hsaKmtCloseKFD(void)
{
	return hsaKmtCloseKFDCtx();
}
