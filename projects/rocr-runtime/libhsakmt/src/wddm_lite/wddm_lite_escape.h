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

/*
 * Userspace escape command definitions for our custom WDDM driver.
 * Mirrors the non-kernel-mode portion of amdgpu_wddm.h.
 *
 * These structures are passed through D3DKMTEscape to the custom
 * WDDM miniport driver. They are NOT compatible with the AMD
 * Adrenaline driver — use the DXG backend for Adrenaline.
 */

#ifndef WDDM_LITE_ESCAPE_H_INCLUDED
#define WDDM_LITE_ESCAPE_H_INCLUDED

#include <windows.h>

/* Use NTSTATUS from windows headers */
#ifndef STATUS_SUCCESS
#define STATUS_SUCCESS ((NTSTATUS)0x00000000L)
#endif

typedef enum _AMDGPU_ESCAPE_CODE {
    AMDGPU_ESCAPE_GET_INFO          = 0x0001,
    AMDGPU_ESCAPE_READ_REG32        = 0x0010,
    AMDGPU_ESCAPE_WRITE_REG32       = 0x0011,
    AMDGPU_ESCAPE_MAP_BAR           = 0x0020,
    AMDGPU_ESCAPE_UNMAP_BAR         = 0x0021,
    AMDGPU_ESCAPE_ALLOC_DMA         = 0x0030,
    AMDGPU_ESCAPE_FREE_DMA          = 0x0031,
    AMDGPU_ESCAPE_MAP_VRAM          = 0x0040,
    AMDGPU_ESCAPE_REGISTER_EVENT    = 0x0050,
    AMDGPU_ESCAPE_ENABLE_MSI        = 0x0051,
    AMDGPU_ESCAPE_GET_IOMMU_INFO    = 0x0060,

    /* KFD-equivalent compute operations (Phase 2) */
    AMDGPU_ESCAPE_ALLOC_MEMORY      = 0x0100,
    AMDGPU_ESCAPE_FREE_MEMORY       = 0x0101,
    AMDGPU_ESCAPE_MAP_MEMORY        = 0x0102,
    AMDGPU_ESCAPE_UNMAP_MEMORY      = 0x0103,
    AMDGPU_ESCAPE_CREATE_QUEUE      = 0x0110,
    AMDGPU_ESCAPE_DESTROY_QUEUE     = 0x0111,
    AMDGPU_ESCAPE_UPDATE_QUEUE      = 0x0112,
    AMDGPU_ESCAPE_CREATE_EVENT      = 0x0120,
    AMDGPU_ESCAPE_DESTROY_EVENT     = 0x0121,
    AMDGPU_ESCAPE_SET_EVENT         = 0x0122,
    AMDGPU_ESCAPE_RESET_EVENT       = 0x0123,
    AMDGPU_ESCAPE_WAIT_EVENTS       = 0x0124,
    AMDGPU_ESCAPE_GET_PROCESS_APERTURES = 0x0130,
    AMDGPU_ESCAPE_SET_MEMORY_POLICY = 0x0131,
    AMDGPU_ESCAPE_SET_SCRATCH_BACKING = 0x0132,
    AMDGPU_ESCAPE_SET_TRAP_HANDLER  = 0x0133,
    AMDGPU_ESCAPE_GET_CLOCK_COUNTERS = 0x0140,
    AMDGPU_ESCAPE_GET_VERSION       = 0x0150,
    AMDGPU_ESCAPE_GET_PHYS_PAGES    = 0x0160,
} AMDGPU_ESCAPE_CODE;

typedef struct _AMDGPU_ESCAPE_HEADER {
    AMDGPU_ESCAPE_CODE  Command;
    NTSTATUS            Status;
    ULONG               Size;
} AMDGPU_ESCAPE_HEADER;

typedef struct _AMDGPU_ESCAPE_GET_INFO_DATA {
    AMDGPU_ESCAPE_HEADER Header;
    USHORT  VendorId;
    USHORT  DeviceId;
    USHORT  SubsystemVendorId;
    USHORT  SubsystemId;
    UCHAR   RevisionId;
    UCHAR   Reserved[3];
    ULONG   NumBars;
    struct {
        LARGE_INTEGER   PhysicalAddress;
        ULONGLONG       Length;
        BOOLEAN         IsMemory;
        BOOLEAN         Is64Bit;
        BOOLEAN         IsPrefetchable;
        UCHAR           Reserved;
    } Bars[6];
    ULONGLONG   VramSizeBytes;
    ULONGLONG   VisibleVramSizeBytes;
    ULONG       MmioBarIndex;
    ULONG       VramBarIndex;
    BOOLEAN     Headless;
    UCHAR       Reserved2[3];
} AMDGPU_ESCAPE_GET_INFO_DATA;

typedef struct _AMDGPU_ESCAPE_REG32_DATA {
    AMDGPU_ESCAPE_HEADER Header;
    ULONG   BarIndex;
    ULONG   Offset;
    ULONG   Value;
} AMDGPU_ESCAPE_REG32_DATA;

typedef struct _AMDGPU_ESCAPE_MAP_BAR_DATA {
    AMDGPU_ESCAPE_HEADER Header;
    ULONG       BarIndex;
    ULONGLONG   Offset;
    ULONGLONG   Length;
    PVOID       MappedAddress;
    PVOID       MappingHandle;
} AMDGPU_ESCAPE_MAP_BAR_DATA;

typedef struct _AMDGPU_ESCAPE_ALLOC_DMA_DATA {
    AMDGPU_ESCAPE_HEADER Header;
    ULONGLONG   Size;
    PVOID       CpuAddress;
    ULONGLONG   BusAddress;
    PVOID       AllocationHandle;
} AMDGPU_ESCAPE_ALLOC_DMA_DATA;

typedef struct _AMDGPU_ESCAPE_MAP_VRAM_DATA {
    AMDGPU_ESCAPE_HEADER Header;
    ULONGLONG   Offset;
    ULONGLONG   Length;
    PVOID       MappedAddress;
    PVOID       MappingHandle;
} AMDGPU_ESCAPE_MAP_VRAM_DATA;

/* ======================================================================
 * KFD-equivalent compute escape structures (Phase 2)
 * ====================================================================== */

/* Memory type flags for ALLOC_MEMORY */
#define AMDGPU_MEM_TYPE_VRAM        0x0001
#define AMDGPU_MEM_TYPE_GTT         0x0002
#define AMDGPU_MEM_TYPE_SYSTEM      0x0004
#define AMDGPU_MEM_FLAG_USERPTR     0x0010
#define AMDGPU_MEM_FLAG_HOST_ACCESS 0x0020
#define AMDGPU_MEM_FLAG_NONPAGED    0x0040
#define AMDGPU_MEM_FLAG_READONLY    0x0080
#define AMDGPU_MEM_FLAG_EXECUTABLE  0x0100
#define AMDGPU_MEM_FLAG_AQL_QUEUE   0x0200
#define AMDGPU_MEM_FLAG_UNCACHED    0x0400
#define AMDGPU_MEM_FLAG_CONTIGUOUS  0x0800
#define AMDGPU_MEM_FLAG_NO_SUBSTITUTE 0x1000
#define AMDGPU_MEM_FLAG_SCRATCH     0x2000
#define AMDGPU_MEM_FLAG_GDS         0x4000
#define AMDGPU_MEM_FLAG_COHERENT    0x8000

typedef struct _AMDGPU_ESCAPE_ALLOC_MEMORY_DATA {
    AMDGPU_ESCAPE_HEADER Header;
    ULONG       GpuId;             /* KFD GPU node ID */
    ULONGLONG   SizeInBytes;
    ULONGLONG   Alignment;         /* 0 = page-aligned */
    ULONG       Flags;             /* AMDGPU_MEM_FLAG_* | AMDGPU_MEM_TYPE_* */
    ULONGLONG   VaAddress;         /* Preferred VA (0 = driver chooses) */
    /* Output */
    PVOID       CpuAddress;        /* CPU-visible mapping (if HOST_ACCESS) */
    ULONGLONG   GpuAddress;        /* GPU virtual address */
    ULONGLONG   Handle;            /* Opaque handle for free/map/unmap */
} AMDGPU_ESCAPE_ALLOC_MEMORY_DATA;

typedef struct _AMDGPU_ESCAPE_FREE_MEMORY_DATA {
    AMDGPU_ESCAPE_HEADER Header;
    ULONGLONG   Handle;
} AMDGPU_ESCAPE_FREE_MEMORY_DATA;

typedef struct _AMDGPU_ESCAPE_MAP_MEMORY_DATA {
    AMDGPU_ESCAPE_HEADER Header;
    ULONGLONG   Handle;
    ULONG       GpuId;
    /* Output */
    ULONGLONG   GpuAddress;
} AMDGPU_ESCAPE_MAP_MEMORY_DATA;

typedef struct _AMDGPU_ESCAPE_UNMAP_MEMORY_DATA {
    AMDGPU_ESCAPE_HEADER Header;
    ULONGLONG   Handle;
    ULONG       GpuId;
} AMDGPU_ESCAPE_UNMAP_MEMORY_DATA;

/* Queue types matching KFD */
#define AMDGPU_QUEUE_TYPE_COMPUTE     1
#define AMDGPU_QUEUE_TYPE_SDMA        2
#define AMDGPU_QUEUE_TYPE_COMPUTE_AQL 21

typedef struct _AMDGPU_ESCAPE_CREATE_QUEUE_DATA {
    AMDGPU_ESCAPE_HEADER Header;
    ULONG       GpuId;
    ULONG       QueueType;         /* AMDGPU_QUEUE_TYPE_* */
    ULONG       QueuePercentage;
    LONG        Priority;          /* -3 to +3 */
    ULONGLONG   QueueAddress;      /* Userspace ring buffer VA */
    ULONGLONG   QueueSizeInBytes;
    ULONGLONG   WritePointerAddress;  /* Userspace write ptr VA */
    ULONGLONG   ReadPointerAddress;   /* Userspace read ptr VA */
    ULONGLONG   EopBufferAddress;  /* End-of-pipe buffer VA */
    ULONG       EopBufferSize;
    ULONGLONG   ContextSaveAddress;
    ULONG       ContextSaveSize;
    ULONG       SdmaEngineId;      /* For SDMA queues */
    /* Output */
    ULONGLONG   QueueId;
    ULONGLONG   DoorbellOffset;    /* Offset into doorbell BAR */
} AMDGPU_ESCAPE_CREATE_QUEUE_DATA;

typedef struct _AMDGPU_ESCAPE_DESTROY_QUEUE_DATA {
    AMDGPU_ESCAPE_HEADER Header;
    ULONGLONG   QueueId;
} AMDGPU_ESCAPE_DESTROY_QUEUE_DATA;

typedef struct _AMDGPU_ESCAPE_UPDATE_QUEUE_DATA {
    AMDGPU_ESCAPE_HEADER Header;
    ULONGLONG   QueueId;
    ULONG       QueuePercentage;
    LONG        Priority;
    ULONGLONG   QueueAddress;
    ULONGLONG   QueueSizeInBytes;
} AMDGPU_ESCAPE_UPDATE_QUEUE_DATA;

/* Event types matching KFD HSA_EVENTTYPE */
#define AMDGPU_EVENT_TYPE_SIGNAL        0
#define AMDGPU_EVENT_TYPE_QUEUE         7
#define AMDGPU_EVENT_TYPE_MEMORY        8

typedef struct _AMDGPU_ESCAPE_CREATE_EVENT_DATA {
    AMDGPU_ESCAPE_HEADER Header;
    ULONG       EventType;         /* AMDGPU_EVENT_TYPE_* */
    ULONG       GpuId;
    BOOLEAN     AutoReset;         /* ManualReset = !AutoReset */
    UCHAR       Reserved[3];
    /* Output */
    ULONG       EventId;
    ULONGLONG   EventPageAddress;  /* Shared page for signal value */
    ULONG       EventSlotIndex;    /* Slot in event page */
} AMDGPU_ESCAPE_CREATE_EVENT_DATA;

typedef struct _AMDGPU_ESCAPE_DESTROY_EVENT_DATA {
    AMDGPU_ESCAPE_HEADER Header;
    ULONG       EventId;
} AMDGPU_ESCAPE_DESTROY_EVENT_DATA;

typedef struct _AMDGPU_ESCAPE_SET_EVENT_DATA {
    AMDGPU_ESCAPE_HEADER Header;
    ULONG       EventId;
} AMDGPU_ESCAPE_SET_EVENT_DATA;

typedef struct _AMDGPU_ESCAPE_RESET_EVENT_DATA {
    AMDGPU_ESCAPE_HEADER Header;
    ULONG       EventId;
} AMDGPU_ESCAPE_RESET_EVENT_DATA;

#define AMDGPU_MAX_WAIT_EVENTS 16

typedef struct _AMDGPU_ESCAPE_WAIT_EVENTS_DATA {
    AMDGPU_ESCAPE_HEADER Header;
    ULONG       NumEvents;
    ULONG       EventIds[AMDGPU_MAX_WAIT_EVENTS];
    BOOLEAN     WaitAll;
    UCHAR       Reserved[3];
    ULONG       TimeoutMs;         /* INFINITE = 0xFFFFFFFF */
    /* Output */
    ULONG       SignaledIndex;     /* Which event fired (if !WaitAll) */
} AMDGPU_ESCAPE_WAIT_EVENTS_DATA;

typedef struct _AMDGPU_ESCAPE_GET_PROCESS_APERTURES_DATA {
    AMDGPU_ESCAPE_HEADER Header;
    ULONG       GpuId;
    /* Output */
    ULONGLONG   LdsBase;
    ULONGLONG   LdsLimit;
    ULONGLONG   ScratchBase;
    ULONGLONG   ScratchLimit;
    ULONGLONG   GpuVmBase;
    ULONGLONG   GpuVmLimit;
} AMDGPU_ESCAPE_GET_PROCESS_APERTURES_DATA;

typedef struct _AMDGPU_ESCAPE_SET_MEMORY_POLICY_DATA {
    AMDGPU_ESCAPE_HEADER Header;
    ULONG       GpuId;
    ULONG       DefaultPolicy;     /* HSA caching type */
    ULONG       AlternatePolicy;
    ULONGLONG   AlternateApertureBase;
    ULONGLONG   AlternateApertureSize;
} AMDGPU_ESCAPE_SET_MEMORY_POLICY_DATA;

typedef struct _AMDGPU_ESCAPE_SET_SCRATCH_BACKING_DATA {
    AMDGPU_ESCAPE_HEADER Header;
    ULONG       GpuId;
    ULONGLONG   ScratchBackingVa;
    ULONGLONG   ScratchBackingSize;
} AMDGPU_ESCAPE_SET_SCRATCH_BACKING_DATA;

typedef struct _AMDGPU_ESCAPE_SET_TRAP_HANDLER_DATA {
    AMDGPU_ESCAPE_HEADER Header;
    ULONG       GpuId;
    ULONGLONG   TbaAddress;        /* Trap handler base address */
    ULONGLONG   TbaSize;
    ULONGLONG   TmaAddress;        /* Trap handler memory area */
    ULONGLONG   TmaSize;
} AMDGPU_ESCAPE_SET_TRAP_HANDLER_DATA;

typedef struct _AMDGPU_ESCAPE_GET_CLOCK_COUNTERS_DATA {
    AMDGPU_ESCAPE_HEADER Header;
    ULONG       GpuId;
    /* Output */
    ULONGLONG   GpuClockCounter;
    ULONGLONG   CpuClockCounter;
    ULONGLONG   SystemClockCounter;
    ULONGLONG   SystemClockFrequencyHz;
    ULONGLONG   GpuClockFrequencyHz;
} AMDGPU_ESCAPE_GET_CLOCK_COUNTERS_DATA;

typedef struct _AMDGPU_ESCAPE_GET_VERSION_DATA {
    AMDGPU_ESCAPE_HEADER Header;
    /* Output */
    ULONG       KfdMajorVersion;
    ULONG       KfdMinorVersion;
} AMDGPU_ESCAPE_GET_VERSION_DATA;

/*
 * GET_PHYS_PAGES: Return per-page physical/bus addresses for an allocation.
 * Used by userspace to populate GART page table entries.
 * Handle must be from a prior ALLOC_MEMORY call.
 * Returns physical addresses for each 4KB page of the allocation.
 */
#define AMDGPU_MAX_PHYS_PAGES 256  /* Up to 1MB per call */
typedef struct _AMDGPU_ESCAPE_GET_PHYS_PAGES_DATA {
    AMDGPU_ESCAPE_HEADER Header;
    ULONGLONG   Handle;                     /* Input: allocation handle */
    ULONG       PageOffset;                 /* Input: start page index */
    /* Output */
    ULONG       NumPages;                   /* Actual pages returned */
    ULONG       TotalPages;                 /* Total pages in allocation */
    ULONGLONG   PhysAddrs[AMDGPU_MAX_PHYS_PAGES];
} AMDGPU_ESCAPE_GET_PHYS_PAGES_DATA;

#endif /* WDDM_LITE_ESCAPE_H_INCLUDED */
