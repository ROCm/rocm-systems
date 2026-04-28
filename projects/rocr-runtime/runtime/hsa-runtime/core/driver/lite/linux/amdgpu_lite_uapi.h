/* SPDX-License-Identifier: GPL-2.0 WITH Linux-syscall-note */
/*
 * Userspace ABI for the amdgpu_lite kernel module.
 */

#ifndef HSA_RUNTIME_CORE_DRIVER_LITE_LINUX_AMDGPU_LITE_UAPI_H_
#define HSA_RUNTIME_CORE_DRIVER_LITE_LINUX_AMDGPU_LITE_UAPI_H_

#include <linux/types.h>
#include <sys/ioctl.h>

#define AMDGPU_LITE_MAX_BARS 6

struct amdgpu_lite_bar_info {
  __u64 phys_addr;
  __u64 size;
  __u32 is_memory;
  __u32 is_64bit;
  __u32 is_prefetchable;
  __u32 bar_index;
};

struct amdgpu_lite_get_info {
  __u16 vendor_id;
  __u16 device_id;
  __u16 subsystem_vendor_id;
  __u16 subsystem_id;
  __u8 revision_id;
  __u8 reserved1[3];
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

struct amdgpu_lite_map_bar {
  __u32 bar_index;
  __u32 reserved1;
  __u64 offset;
  __u64 size;
  __u64 mmap_offset;
  __u32 reserved2[4];
};

struct amdgpu_lite_alloc_gtt {
  __u64 size;
  __u32 reserved1[2];
  __u64 handle;
  __u64 bus_addr;
  __u64 mmap_offset;
  __u32 reserved2[4];
};

struct amdgpu_lite_free_gtt {
  __u64 handle;
  __u32 reserved[4];
};

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

struct amdgpu_lite_setup_irq {
  __s32 eventfd;
  __u32 irq_source;
  __u32 registration_id;
  __u32 reserved[3];
  __u32 out_registration_id;
  __u32 reserved2;
};

#define AMDGPU_LITE_IOC_MAGIC 'L'

#define AMDGPU_LITE_IOC_GET_INFO \
  _IOR(AMDGPU_LITE_IOC_MAGIC, 0x01, struct amdgpu_lite_get_info)
#define AMDGPU_LITE_IOC_MAP_BAR \
  _IOWR(AMDGPU_LITE_IOC_MAGIC, 0x02, struct amdgpu_lite_map_bar)
#define AMDGPU_LITE_IOC_ALLOC_GTT \
  _IOWR(AMDGPU_LITE_IOC_MAGIC, 0x10, struct amdgpu_lite_alloc_gtt)
#define AMDGPU_LITE_IOC_FREE_GTT \
  _IOW(AMDGPU_LITE_IOC_MAGIC, 0x11, struct amdgpu_lite_free_gtt)
#define AMDGPU_LITE_IOC_ALLOC_VRAM \
  _IOWR(AMDGPU_LITE_IOC_MAGIC, 0x20, struct amdgpu_lite_alloc_vram)
#define AMDGPU_LITE_IOC_FREE_VRAM \
  _IOW(AMDGPU_LITE_IOC_MAGIC, 0x21, struct amdgpu_lite_free_vram)
#define AMDGPU_LITE_IOC_MAP_GPU \
  _IOWR(AMDGPU_LITE_IOC_MAGIC, 0x30, struct amdgpu_lite_map_gpu)
#define AMDGPU_LITE_IOC_UNMAP_GPU \
  _IOW(AMDGPU_LITE_IOC_MAGIC, 0x31, struct amdgpu_lite_unmap_gpu)
#define AMDGPU_LITE_IOC_SETUP_IRQ \
  _IOWR(AMDGPU_LITE_IOC_MAGIC, 0x40, struct amdgpu_lite_setup_irq)

#endif  // HSA_RUNTIME_CORE_DRIVER_LITE_LINUX_AMDGPU_LITE_UAPI_H_
