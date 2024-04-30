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

#include <pthread.h>
#include <stdint.h>
#include <limits.h>
#include "hsakmt/hsakmt.h"

#include "inc/wddm/types.h"
#include "inc/wddm/device.h"

rocr::core::WDDMDevice* get_wddmdev(uint32_t node_id);

extern unsigned long dxg_open_count;
extern bool hsakmt_forked;
extern pthread_mutex_t hsakmt_mutex;
extern bool is_dgpu;
extern bool is_svm_api_supported;
extern int zfb_support;
extern int vendor_packet_support;

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

#define CHECK_DXG_OPEN() \
	do { if (dxg_open_count == 0 || hsakmt_forked) return HSAKMT_STATUS_KERNEL_IO_CHANNEL_NOT_OPENED; } while (0)

/* Might be defined in limits.h on platforms where it is constant (used by musl) */
/* See also: https://pubs.opengroup.org/onlinepubs/7908799/xsh/limits.h.html */
#ifndef PAGE_SIZE
extern int PAGE_SIZE;
#endif
extern int PAGE_SHIFT;

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

/* Expects HSA_ENGINE_ID.ui32, returns gfxv (full) in hex */
#define HSA_GET_GFX_VERSION_FULL(ui32) \
	(((ui32.Major) << 16) | ((ui32.Minor) << 8) | (ui32.Stepping))

HSAKMT_STATUS validate_nodeid(uint32_t nodeid, uint32_t *gpu_id);
HSAKMT_STATUS gpuid_to_nodeid(uint32_t gpu_id, uint32_t* node_id);
bool prefer_ats(HSAuint32 node_id);
uint16_t get_device_id_by_node_id(HSAuint32 node_id);
uint16_t get_device_id_by_gpu_id(HSAuint32 gpu_id);
uint32_t get_direct_link_cpu(uint32_t gpu_node);

HSAKMT_STATUS topology_sysfs_get_system_props(HsaSystemProperties *props);
HSAKMT_STATUS topology_get_node_props(HSAuint32 NodeId,
				      HsaNodeProperties *NodeProperties);
HSAKMT_STATUS topology_get_iolink_props(HSAuint32 NodeId,
					HSAuint32 NumIoLinks,
					HsaIoLinkProperties *IoLinkProperties);
void topology_setup_is_dgpu_param(HsaNodeProperties *props);

HSAuint32 PageSizeFromFlags(unsigned int pageSizeFlags);

#define MIN(a, b) ({				\
	typeof(a) tmp1 = (a), tmp2 = (b);	\
	tmp1 < tmp2 ? tmp1 : tmp2; })

#define MAX(a, b) ({				\
	typeof(a) tmp1 = (a), tmp2 = (b);	\
	tmp1 > tmp2 ? tmp1 : tmp2; })

uint32_t get_num_sysfs_nodes(void);

bool is_forked_child(void);

/* Calculate VGPR and SGPR register file size per CU */
uint32_t get_vgpr_size_per_cu(HSA_ENGINE_ID id);
#define SGPR_SIZE_PER_CU 0x4000

#endif
