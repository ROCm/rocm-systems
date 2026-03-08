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

#ifndef AMDGPU_LITE_IOCTL_H_INCLUDED
#define AMDGPU_LITE_IOCTL_H_INCLUDED

#include <stdint.h>
#include <sys/ioctl.h>
#include <errno.h>
#include <unistd.h>

/*
 * Userspace type aliases matching the kernel header convention.
 * Guard against redefinition if linux/types.h is already included
 * (e.g., transitively through kfd_ioctl.h -> drm.h -> linux/types.h).
 */
#include <linux/types.h>

#define AMDGPU_LITE_MAX_BARS    6

/* ======================================================================
 * BAR info (returned in GET_INFO)
 * ====================================================================== */

struct amdgpu_lite_bar_info {
	__u64 phys_addr;
	__u64 size;
	__u32 is_memory;
	__u32 is_64bit;
	__u32 is_prefetchable;
	__u32 bar_index;
};

/* ======================================================================
 * GET_INFO - Return device identification and BAR layout
 * ====================================================================== */

struct amdgpu_lite_get_info {
	__u16 vendor_id;
	__u16 device_id;
	__u16 subsystem_vendor_id;
	__u16 subsystem_id;
	__u8  revision_id;
	__u8  reserved1[3];
	__u32 num_bars;
	struct amdgpu_lite_bar_info bars[AMDGPU_LITE_MAX_BARS];
	__u64 vram_size;
	__u64 visible_vram_size;
	__u32 mmio_bar_index;
	__u32 vram_bar_index;
	__u32 doorbell_bar_index;
	__u32 reserved2;
	__u64 gart_table_bus_addr;
	__u64 gart_table_size;
	__u64 gart_gpu_va_start;
};

/* ======================================================================
 * MAP_BAR - Set up mmap offset for a specific BAR
 * ====================================================================== */

struct amdgpu_lite_map_bar {
	__u32 bar_index;
	__u32 reserved1;
	__u64 offset;
	__u64 size;
	__u64 mmap_offset;
	__u32 reserved2[4];
};

/* ======================================================================
 * ALLOC_GTT - Allocate DMA-coherent system memory
 * ====================================================================== */

struct amdgpu_lite_alloc_gtt {
	__u64 size;
	__u32 reserved1[2];
	__u64 handle;
	__u64 bus_addr;
	__u64 mmap_offset;
	__u32 reserved2[4];
};

/* ======================================================================
 * FREE_GTT - Free DMA-coherent system memory
 * ====================================================================== */

struct amdgpu_lite_free_gtt {
	__u64 handle;
	__u32 reserved[4];
};

/* ======================================================================
 * ALLOC_VRAM / FREE_VRAM
 * ====================================================================== */

struct amdgpu_lite_alloc_vram {
	__u64 size;
	__u32 flags;
	__u32 reserved1;
	__u64 handle;
	__u64 gpu_addr;
	__u64 mmap_offset;
	__u32 reserved2[4];
};

struct amdgpu_lite_free_vram {
	__u64 handle;
	__u32 reserved[4];
};

/* ======================================================================
 * MAP_GPU / UNMAP_GPU - GPU page table programming
 * ====================================================================== */

struct amdgpu_lite_map_gpu {
	__u64 handle;
	__u64 gpu_va;
	__u64 size;
	__u32 flags;
	__u32 reserved[3];
	__u64 mapped_gpu_va;
};

struct amdgpu_lite_unmap_gpu {
	__u64 gpu_va;
	__u64 size;
	__u32 reserved[4];
};

/* ======================================================================
 * SETUP_IRQ - Register eventfd for interrupt forwarding
 * ====================================================================== */

struct amdgpu_lite_setup_irq {
	__s32 eventfd;
	__u32 irq_source;
	__u32 registration_id;
	__u32 reserved[3];
	__u32 out_registration_id;
	__u32 reserved2;
};

/* ======================================================================
 * Ioctl numbers - type 'L' for Lite
 * ====================================================================== */

#define AMDGPU_LITE_IOC_MAGIC   'L'

#define AMDGPU_LITE_IOC_GET_INFO    _IOR(AMDGPU_LITE_IOC_MAGIC, 0x01, \
					 struct amdgpu_lite_get_info)
#define AMDGPU_LITE_IOC_MAP_BAR     _IOWR(AMDGPU_LITE_IOC_MAGIC, 0x02, \
					   struct amdgpu_lite_map_bar)
#define AMDGPU_LITE_IOC_ALLOC_GTT   _IOWR(AMDGPU_LITE_IOC_MAGIC, 0x10, \
					   struct amdgpu_lite_alloc_gtt)
#define AMDGPU_LITE_IOC_FREE_GTT    _IOW(AMDGPU_LITE_IOC_MAGIC, 0x11, \
					  struct amdgpu_lite_free_gtt)
#define AMDGPU_LITE_IOC_ALLOC_VRAM  _IOWR(AMDGPU_LITE_IOC_MAGIC, 0x20, \
					   struct amdgpu_lite_alloc_vram)
#define AMDGPU_LITE_IOC_FREE_VRAM   _IOW(AMDGPU_LITE_IOC_MAGIC, 0x21, \
					  struct amdgpu_lite_free_vram)
#define AMDGPU_LITE_IOC_MAP_GPU     _IOWR(AMDGPU_LITE_IOC_MAGIC, 0x30, \
					   struct amdgpu_lite_map_gpu)
#define AMDGPU_LITE_IOC_UNMAP_GPU   _IOW(AMDGPU_LITE_IOC_MAGIC, 0x31, \
					  struct amdgpu_lite_unmap_gpu)
#define AMDGPU_LITE_IOC_SETUP_IRQ   _IOWR(AMDGPU_LITE_IOC_MAGIC, 0x40, \
					   struct amdgpu_lite_setup_irq)

/* ======================================================================
 * Helper functions
 * ====================================================================== */

static inline int amdgpu_lite_ioctl_get_info(int fd,
					     struct amdgpu_lite_get_info *info)
{
	int ret;

	do {
		ret = ioctl(fd, AMDGPU_LITE_IOC_GET_INFO, info);
	} while (ret == -1 && (errno == EINTR || errno == EAGAIN));

	return ret;
}

static inline int amdgpu_lite_ioctl_map_bar(int fd,
					    struct amdgpu_lite_map_bar *args)
{
	int ret;

	do {
		ret = ioctl(fd, AMDGPU_LITE_IOC_MAP_BAR, args);
	} while (ret == -1 && (errno == EINTR || errno == EAGAIN));

	return ret;
}

static inline int amdgpu_lite_ioctl_alloc_gtt(int fd,
					      struct amdgpu_lite_alloc_gtt *args)
{
	int ret;

	do {
		ret = ioctl(fd, AMDGPU_LITE_IOC_ALLOC_GTT, args);
	} while (ret == -1 && (errno == EINTR || errno == EAGAIN));

	return ret;
}

static inline int amdgpu_lite_ioctl_free_gtt(int fd,
					     struct amdgpu_lite_free_gtt *args)
{
	int ret;

	do {
		ret = ioctl(fd, AMDGPU_LITE_IOC_FREE_GTT, args);
	} while (ret == -1 && (errno == EINTR || errno == EAGAIN));

	return ret;
}

static inline int amdgpu_lite_ioctl_alloc_vram(int fd,
					       struct amdgpu_lite_alloc_vram *args)
{
	int ret;

	do {
		ret = ioctl(fd, AMDGPU_LITE_IOC_ALLOC_VRAM, args);
	} while (ret == -1 && (errno == EINTR || errno == EAGAIN));

	return ret;
}

static inline int amdgpu_lite_ioctl_free_vram(int fd,
					      struct amdgpu_lite_free_vram *args)
{
	int ret;

	do {
		ret = ioctl(fd, AMDGPU_LITE_IOC_FREE_VRAM, args);
	} while (ret == -1 && (errno == EINTR || errno == EAGAIN));

	return ret;
}

static inline int amdgpu_lite_ioctl_map_gpu(int fd,
					    struct amdgpu_lite_map_gpu *args)
{
	int ret;

	do {
		ret = ioctl(fd, AMDGPU_LITE_IOC_MAP_GPU, args);
	} while (ret == -1 && (errno == EINTR || errno == EAGAIN));

	return ret;
}

static inline int amdgpu_lite_ioctl_unmap_gpu(int fd,
					      struct amdgpu_lite_unmap_gpu *args)
{
	int ret;

	do {
		ret = ioctl(fd, AMDGPU_LITE_IOC_UNMAP_GPU, args);
	} while (ret == -1 && (errno == EINTR || errno == EAGAIN));

	return ret;
}

static inline int amdgpu_lite_ioctl_setup_irq(int fd,
					      struct amdgpu_lite_setup_irq *args)
{
	int ret;

	do {
		ret = ioctl(fd, AMDGPU_LITE_IOC_SETUP_IRQ, args);
	} while (ret == -1 && (errno == EINTR || errno == EAGAIN));

	return ret;
}

#endif /* AMDGPU_LITE_IOCTL_H_INCLUDED */
