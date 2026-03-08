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

#ifndef AMDGPU_LITE_DEVICE_H_INCLUDED
#define AMDGPU_LITE_DEVICE_H_INCLUDED

#include <stdint.h>
#include <stdbool.h>
#include "amdgpu_lite_ioctl.h"

/*
 * Userspace state for a single amdgpu_lite device.
 * There is one global instance since we target a single GPU.
 */
struct amdgpu_lite_device {
	int fd;                             /* /dev/amdgpu_lite0 fd */
	struct amdgpu_lite_get_info info;   /* GET_INFO result */

	void *mmio_bar;                     /* Mapped MMIO BAR */
	size_t mmio_bar_size;

	void *doorbell_bar;                 /* Mapped doorbell BAR */
	size_t doorbell_bar_size;

	void *vram_bar;                     /* Mapped VRAM BAR (if resizable BAR) */
	size_t vram_bar_size;

	uint16_t device_id;
	uint64_t vram_size;
	uint32_t gfx_version;               /* e.g., 0x0C0001 for GFX12.0.1 */

	bool is_open;
};

/* Global device instance */
extern struct amdgpu_lite_device g_amdgpu_lite_dev;

/*
 * Open /dev/amdgpu_lite0, call GET_INFO, and map the BARs.
 * Returns 0 on success, -1 on failure (errno set).
 */
int amdgpu_lite_open(struct amdgpu_lite_device *dev);

/*
 * Unmap BARs and close the device fd.
 */
void amdgpu_lite_close(struct amdgpu_lite_device *dev);

/*
 * Read a 32-bit MMIO register via the mapped MMIO BAR.
 */
static inline uint32_t amdgpu_lite_read_reg32(struct amdgpu_lite_device *dev,
					      uint32_t offset)
{
	volatile uint32_t *reg = (volatile uint32_t *)((uint8_t *)dev->mmio_bar + offset);
	return *reg;
}

/*
 * Write a 32-bit MMIO register via the mapped MMIO BAR.
 */
static inline void amdgpu_lite_write_reg32(struct amdgpu_lite_device *dev,
					   uint32_t offset, uint32_t value)
{
	volatile uint32_t *reg = (volatile uint32_t *)((uint8_t *)dev->mmio_bar + offset);
	*reg = value;
}

#endif /* AMDGPU_LITE_DEVICE_H_INCLUDED */
