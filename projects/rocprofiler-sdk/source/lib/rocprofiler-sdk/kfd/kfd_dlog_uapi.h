/* SPDX-License-Identifier: MIT */
/* KFD dispatch-log stream UAPI (profiler ABI v6, stream ABI v3). Vendored subset
 * of include/uapi/linux/kfd_ioctl.h needed by the SDK's dispatch-log reader; keep
 * byte-accurate with the kernel header. The backing is always KFD-owned GTT
 * consumed zero-copy via RAW_MMAP, so this carries only the OPEN_STREAM request
 * and the stream-fd ioctl -- no ALLOC/MAP/REGISTER/UNREGISTER variants.
 *
 * INCOMPATIBLE WITH lib/rocprofiler-sdk/details/kfd_ioctl.h. This header carries
 * the active dispatch-log UAPI: AMDKFD_IOC_PROFILER at NR 0x28, profiler version
 * 6, with the single DLOG/OPEN_STREAM op. It deliberately conflicts with the
 * older details/kfd_ioctl.h (profiler version 1, AMDKFD_IOC_PROFILER at NR 0x86,
 * no dlog ops) -- both define the same macros and structs with different values.
 * The two must never be included in the same translation unit. Any TU that needs
 * to see both constants at once (e.g. to static_assert their agreement) belongs
 * in a file that includes ONLY this header; see kfd/dlog_abi.cpp. */
#ifndef KFD_DLOG_UAPI_H
#define KFD_DLOG_UAPI_H

#include <linux/ioctl.h>
#include <linux/types.h>

#define AMDKFD_IOCTL_BASE     'K'
#define AMDKFD_IOWR(nr, type) _IOWR(AMDKFD_IOCTL_BASE, nr, type)

#define KFD_DISPATCH_LOG_FW_RECORD_BYTES 20U
#define KFD_IOC_PROFILER_VERSION_NUM     6

enum kfd_profiler_ops
{
    KFD_IOC_PROFILER_PMC         = 0,
    KFD_IOC_PROFILER_PC_SAMPLE   = 1,
    KFD_IOC_PROFILER_VERSION     = 2,
    KFD_IOC_PROFILER_PTL_CONTROL = 3,
    KFD_IOC_PROFILER_DLOG        = 4,
};

/* Sub-operation for KFD_IOC_PROFILER_DLOG. */
enum kfd_profiler_dlog_op
{
    KFD_IOC_PROFILER_DLOG_OPEN_STREAM = 0,
};

/* RAW_MMAP zero-copy consumption is the only supported mode; every other flag
 * bit is reserved and must be zero. */
#define KFD_DLOG_OPEN_F_RAW_MMAP (1u << 0)

/*
 * Args for the KFD_IOC_PROFILER_DLOG op. @dlog_op selects the action (only
 * OPEN_STREAM) and @gpu_id is the KFD user gpu_id; both stay at fixed offsets
 * (0 and 4) so the kernel reads them before dispatching. @buffer_size is the
 * requested records-region size in bytes; the kernel allocates the backing and
 * never silently alters the requested geometry.
 */
struct kfd_ioctl_dlog_args
{
    __u32 dlog_op;     /* IN: enum kfd_profiler_dlog_op */
    __u32 gpu_id;      /* IN: KFD user gpu_id */
    __u32 target_pid;  /* IN: target tgid */
    __u32 flags;       /* IN: KFD_DLOG_OPEN_F_* (reserved bits 0) */
    __u32 buffer_size; /* IN: requested records-region bytes */
    __s32 stream_fd;   /* OUT: anon_inode stream fd */
};

/*
 * Mirror of the kernel's AMDKFD_IOC_PROFILER (0x28) args. @op + @pad precede a
 * union capped at exactly 32 bytes (reserved[8]); the _IOWR-encoded size baked
 * into AMDKFD_IOC_PROFILER matches the kernel. Do not grow the union past 32
 * bytes.
 */
struct kfd_ioctl_profiler_args
{
    __u32 op;  /* enum kfd_profiler_ops */
    __u32 pad; /* IN: must be 0 (reserved) */
    union
    {
        __u32                      version; /* KFD_IOC_PROFILER_VERSION_NUM */
        struct kfd_ioctl_dlog_args dlog;
        __u32                      reserved[8];
        /* The kernel union carries a __u64-aligned member, so it is 8-aligned.
         * Every member above is only 4-aligned; this mirror keeps the struct
         * 8-aligned to match the kernel exactly. Size stays 32 bytes (union is
         * already 32), only alignment changes. */
        __u64 _align;
    };
};

/* Dispatch-log profiler stream (anon_inode fd; KFD-owned GTT BO, RAW mmap) */
#define KFD_DLOG_STREAM_ABI_VERSION 3

#define KFD_DLOG_STATUS_TARGET_EXITED (1ULL << 2)
#define KFD_DLOG_STATUS_FATAL         (1ULL << 5)

/*
 * RAW_MMAP layout in the mmap'd BO: an array of 20-byte firmware records,
 * followed by u64 wptr[num_regions] (firmware producer) then u64
 * rptr[num_regions] (consumer); byte offsets come from kfd_dlog_stream_info.
 * Firmware publishes payload before wptr[]; consumers read wptr[], issue a read
 * barrier, then read payload. The consumer barrier cannot repair missing
 * firmware ordering. Treat record_type==0 or doorbell_off==0 as
 * padding/not-yet-written and skip it.
 */
struct kfd_dlog_stream_info
{
    __u32 abi_version;
    /* raw per-record stride from firmware */
    __u32 fw_record_size;
    __u32 num_regions;
    __u32 region_record_count;
    __u64 buffer_size;
    __u64 mmap_size;
    __u64 records_offset;
    __u64 wptr_offset;
    __u64 rptr_offset;
    __u32 gpu_id;
    __u32 target_pid;
    __u32 pasid;
    __u32 flags;
};

struct kfd_dlog_stream_status
{
    __u64 status;
    __u64 target_exit_count;
};

/* Sub-operation selector for the dispatch-log stream fd ioctl. */
enum kfd_dlog_stream_op
{
    KFD_DLOG_STREAM_OP_INFO = 0,
    KFD_DLOG_STREAM_OP_STATUS,
};

struct kfd_dlog_stream_args
{
    __u32 op;  /* IN: enum kfd_dlog_stream_op */
    __u32 pad; /* IN: must be 0 */
    union
    {
        struct kfd_dlog_stream_info   info;   /* OUT for OP_INFO */
        struct kfd_dlog_stream_status status; /* OUT for OP_STATUS */
    };
};

/*
 * Stream-fd ioctl. Dispatched only on the anon_inode dispatch-log stream fd,
 * never through /dev/kfd. Reuses amdkfd's 'K' magic with NR 0x88.
 */
#define KFD_DLOG_STREAM_IOC _IOWR('K', 0x88, struct kfd_dlog_stream_args)

#define AMDKFD_IOC_PROFILER AMDKFD_IOWR(0x28, struct kfd_ioctl_profiler_args)

#endif /* KFD_DLOG_UAPI_H */
