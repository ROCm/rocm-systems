/*
 * Copyright © 2014 Advanced Micro Devices, Inc.
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

#ifndef LIBHSAKMT_H_INCLUDED
#define LIBHSAKMT_H_INCLUDED

#include "hsakmt/linux/kfd_ioctl.h"
#include "hsakmt/hsakmt.h"
#include <pthread.h>
#include <stdint.h>
#include <limits.h>

extern int hsakmt_kfd_fd;
extern unsigned long hsakmt_kfd_open_count;
extern bool hsakmt_forked;
extern pthread_mutex_t hsakmt_mutex;
extern bool hsakmt_is_dgpu;
extern bool hsakmt_is_svm_api_supported;
extern int hsakmt_zfb_support;

extern HsaVersionInfo hsakmt_kfd_version_info;

#undef HSAKMTAPI
#define HSAKMTAPI __attribute__((visibility ("default")))

#if defined(__clang__)
#if __has_feature(address_sanitizer)
#define SANITIZER_AMDGPU 1
#endif
#endif

/*Avoid pointer-to-int-cast warning*/
#define PORT_VPTR_TO_UINT64(vptr) ((uint64_t)(unsigned long)(vptr))

/*Avoid int-to-pointer-cast warning*/
#define PORT_UINT64_TO_VPTR(v) ((void*)(unsigned long)(v))

#define CHECK_KFD_OPEN() \
	do { if (hsakmt_kfd_open_count == 0 || hsakmt_forked) return HSAKMT_STATUS_KERNEL_IO_CHANNEL_NOT_OPENED; } while (0)

#define CHECK_KFD_MINOR_VERSION(minor)					\
	do { if ((minor) > hsakmt_kfd_version_info.KernelInterfaceMinorVersion)\
		return HSAKMT_STATUS_NOT_SUPPORTED; } while (0)

extern int hsakmt_page_size;
extern int hsakmt_page_shift;

/* Might be defined in limits.h on platforms where it is constant (used by musl) */
/* See also: https://pubs.opengroup.org/onlinepubs/7908799/xsh/limits.h.html */
#ifndef PAGE_SIZE
#define PAGE_SIZE hsakmt_page_size
#endif
#ifndef PAGE_SHIFT
#define PAGE_SHIFT hsakmt_page_shift
#endif

/* VI HW bug requires this virtual address alignment */
#define TONGA_PAGE_SIZE 0x8000

/* 64KB BigK fragment size for TLB efficiency */
#define GPU_BIGK_PAGE_SIZE (1 << 16)

/* 2MB huge page size for 4-level page tables on Vega10 and later GPUs */
#define GPU_HUGE_PAGE_SIZE (2 << 20)

#define CHECK_PAGE_MULTIPLE(x) \
	do { if ((uint64_t)PORT_VPTR_TO_UINT64(x) % PAGE_SIZE) return HSAKMT_STATUS_INVALID_PARAMETER; } while(0)

#define ALIGN_UP(x,align) (((uint64_t)(x) + (align) - 1) & ~(uint64_t)((align)-1))
#define ALIGN_UP_32(x,align) (((uint32_t)(x) + (align) - 1) & ~(uint32_t)((align)-1))
#define PAGE_ALIGN_UP(x) ALIGN_UP(x,PAGE_SIZE)
#define BITMASK(n) ((n) ? (UINT64_MAX >> (sizeof(UINT64_MAX) * CHAR_BIT - (n))) : 0)
#define ARRAY_LEN(array) (sizeof(array) / sizeof(array[0]))

/* HSA Thunk logging usage */
extern int hsakmt_debug_level;
#define hsakmt_print(level, fmt, ...) \
	do { if (level <= hsakmt_debug_level) fprintf(stderr, fmt, ##__VA_ARGS__); } while (0)
#define HSAKMT_DEBUG_LEVEL_DEFAULT	-1
#define HSAKMT_DEBUG_LEVEL_ERR		3
#define HSAKMT_DEBUG_LEVEL_WARNING	4
#define HSAKMT_DEBUG_LEVEL_INFO		6
#define HSAKMT_DEBUG_LEVEL_DEBUG	7
#define pr_err(fmt, ...) \
	hsakmt_print(HSAKMT_DEBUG_LEVEL_ERR, fmt, ##__VA_ARGS__)
#define pr_warn(fmt, ...) \
	hsakmt_print(HSAKMT_DEBUG_LEVEL_WARNING, fmt, ##__VA_ARGS__)
#define pr_info(fmt, ...) \
	hsakmt_print(HSAKMT_DEBUG_LEVEL_INFO, fmt, ##__VA_ARGS__)
#define pr_debug(fmt, ...) \
	hsakmt_print(HSAKMT_DEBUG_LEVEL_DEBUG, fmt, ##__VA_ARGS__)
#define pr_err_once(fmt, ...)                   \
({                                              \
        static bool __print_once;               \
        if (!__print_once) {                    \
                __print_once = true;            \
                pr_err(fmt, ##__VA_ARGS__);     \
        }                                       \
})
#define pr_warn_once(fmt, ...)                  \
({                                              \
        static bool __print_once;               \
        if (!__print_once) {                    \
                __print_once = true;            \
                pr_warn(fmt, ##__VA_ARGS__);    \
        }                                       \
})

/* Expects gfxv (full) in decimal */
#define HSA_GET_GFX_VERSION_MAJOR(gfxv)   (((gfxv) / 10000) % 100)
#define HSA_GET_GFX_VERSION_MINOR(gfxv)   (((gfxv) / 100) % 100)
#define HSA_GET_GFX_VERSION_STEP(gfxv)    ((gfxv) % 100)

/* Expects HSA_ENGINE_ID.ui32, returns gfxv (full) in hex */
#define HSA_GET_GFX_VERSION_FULL(ui32) \
	(((ui32.Major) << 16) | ((ui32.Minor) << 8) | (ui32.Stepping))

enum full_gfx_versions {
	GFX_VERSION_KAVERI		= 0x070000,
	GFX_VERSION_HAWAII		= 0x070001,
	GFX_VERSION_CARRIZO		= 0x080001,
	GFX_VERSION_TONGA		= 0x080002,
	GFX_VERSION_FIJI		= 0x080003,
	GFX_VERSION_POLARIS10		= 0x080003,
	GFX_VERSION_POLARIS11		= 0x080003,
	GFX_VERSION_POLARIS12		= 0x080003,
	GFX_VERSION_VEGAM		= 0x080003,
	GFX_VERSION_VEGA10		= 0x090000,
	GFX_VERSION_RAVEN		= 0x090002,
	GFX_VERSION_VEGA12		= 0x090004,
	GFX_VERSION_VEGA20		= 0x090006,
	GFX_VERSION_ARCTURUS		= 0x090008,
	GFX_VERSION_ALDEBARAN		= 0x09000A,
	GFX_VERSION_AQUA_VANJARAM	= 0x090400,
	GFX_VERSION_GFX950		= 0x090500,
	GFX_VERSION_RENOIR		= 0x09000C,
	GFX_VERSION_NAVI10		= 0x0A0100,
	GFX_VERSION_NAVI12		= 0x0A0101,
	GFX_VERSION_NAVI14		= 0x0A0102,
	GFX_VERSION_CYAN_SKILLFISH	= 0x0A0103,
	GFX_VERSION_SIENNA_CICHLID	= 0x0A0300,
	GFX_VERSION_NAVY_FLOUNDER	= 0x0A0301,
	GFX_VERSION_DIMGREY_CAVEFISH	= 0x0A0302,
	GFX_VERSION_VANGOGH	 	= 0x0A0303,
	GFX_VERSION_BEIGE_GOBY	 	= 0x0A0304,
	GFX_VERSION_YELLOW_CARP	 	= 0x0A0305,
	GFX_VERSION_PLUM_BONITO		= 0x0B0000,
	GFX_VERSION_WHEAT_NAS		= 0x0B0001,
	GFX_VERSION_GFX1200		= 0x0C0000,
	GFX_VERSION_GFX1201		= 0x0C0001,
};

struct hsa_gfxip_table {
	uint16_t device_id;		// Device ID
	unsigned char major;		// GFXIP Major engine version
	unsigned char minor;		// GFXIP Minor engine version
	unsigned char stepping;		// GFXIP Stepping info
	const char *amd_name;		// CALName of the device
};

HSAKMT_STATUS hsakmt_init_kfd_version(void);

#define IS_SOC15(gfxv) ((gfxv) >= GFX_VERSION_VEGA10)

HSAKMT_STATUS hsakmt_validate_nodeid(uint32_t nodeid, uint32_t *gpu_id);
HSAKMT_STATUS hsakmt_gpuid_to_nodeid(uint32_t gpu_id, uint32_t* node_id);
uint32_t hsakmt_get_gfxv_by_node_id(HSAuint32 node_id);
bool hsakmt_prefer_ats(HSAuint32 node_id);
uint16_t hsakmt_get_device_id_by_node_id(HSAuint32 node_id);
uint16_t hsakmt_get_device_id_by_gpu_id(HSAuint32 gpu_id);
uint32_t hsakmt_get_direct_link_cpu(uint32_t gpu_node);
int get_drm_render_fd_by_gpu_id(HSAuint32 gpu_id);
HSAKMT_STATUS hsakmt_validate_nodeid_array(uint32_t **gpu_id_array,
		uint32_t NumberOfNodes, uint32_t *NodeArray);

HSAKMT_STATUS hsakmt_topology_sysfs_get_system_props(HsaSystemProperties *props);
HSAKMT_STATUS hsakmt_topology_get_node_props(HSAuint32 NodeId,
				      HsaNodeProperties *NodeProperties);
HSAKMT_STATUS hsakmt_topology_get_iolink_props(HSAuint32 NodeId,
					HSAuint32 NumIoLinks,
					HsaIoLinkProperties *IoLinkProperties);
void hsakmt_topology_setup_is_dgpu_param(HsaNodeProperties *props);
bool hsakmt_topology_is_svm_needed(HSA_ENGINE_ID EngineId);

HSAuint32 hsakmt_PageSizeFromFlags(unsigned int pageSizeFlags);

void* hsakmt_allocate_exec_aligned_memory_gpu(uint32_t size, uint32_t align,
				       uint32_t gpu_id,
				       uint32_t NodeId, bool NonPaged,
				       bool DeviceLocal, bool Uncached);
void hsakmt_free_exec_aligned_memory_gpu(void *addr, uint32_t size, uint32_t align);
HSAKMT_STATUS hsakmt_init_process_doorbells(unsigned int NumNodes);
void hsakmt_destroy_process_doorbells(void);
HSAKMT_STATUS hsakmt_init_device_debugging_memory(unsigned int NumNodes);
void hsakmt_destroy_device_debugging_memory(void);
bool hsakmt_debug_get_reg_status(uint32_t node_id);
HSAKMT_STATUS hsakmt_init_counter_props(unsigned int NumNodes);
void hsakmt_destroy_counter_props(void);
uint32_t *hsakmt_convert_queue_ids(HSAuint32 NumQueues, HSA_QUEUEID *Queues);

extern int hsakmt_ioctl(int fd, unsigned long request, void *arg);

/* Void pointer arithmetic (or remove -Wpointer-arith to allow void pointers arithmetic) */
#define VOID_PTR_ADD32(ptr,n) (void*)((uint32_t*)(ptr) + n)/*ptr + offset*/
#define VOID_PTR_ADD(ptr,n) (void*)((uint8_t*)(ptr) + n)/*ptr + offset*/
#define VOID_PTR_SUB(ptr,n) (void*)((uint8_t*)(ptr) - n)/*ptr - offset*/
#define VOID_PTRS_SUB(ptr1,ptr2) (uint64_t)((uint8_t*)(ptr1) - (uint8_t*)(ptr2)) /*ptr1 - ptr2*/

#define MIN(a, b) ({				\
	typeof(a) tmp1 = (a), tmp2 = (b);	\
	tmp1 < tmp2 ? tmp1 : tmp2; })

#define MAX(a, b) ({				\
	typeof(a) tmp1 = (a), tmp2 = (b);	\
	tmp1 > tmp2 ? tmp1 : tmp2; })

#define POWER_OF_2(x) ((x && (!(x & (x - 1)))) ? 1 : 0)

void hsakmt_clear_events_page(void);
void hsakmt_fmm_clear_all_mem(void);
void hsakmt_clear_process_doorbells(void);
uint32_t hsakmt_get_num_sysfs_nodes(void);

bool hsakmt_is_forked_child(void);

/* Calculate VGPR and SGPR register file size per CU */
uint32_t hsakmt_get_vgpr_size_per_cu(uint32_t gfxv);
#define SGPR_SIZE_PER_CU 0x4000
#endif
