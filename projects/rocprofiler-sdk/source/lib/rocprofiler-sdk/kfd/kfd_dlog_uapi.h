/* SPDX-License-Identifier: MIT */
/* Minimal KFD dispatch-log stream UAPI for standalone test builds (v2). */
#ifndef KFD_DLOG_UAPI_H
#define KFD_DLOG_UAPI_H

#include <linux/ioctl.h>
#include <linux/types.h>

#define AMDKFD_IOCTL_BASE     'K'
#define AMDKFD_IOWR(nr, type) _IOWR(AMDKFD_IOCTL_BASE, nr, type)
#define AMDKFD_IOW(nr, type)  _IOW(AMDKFD_IOCTL_BASE, nr, type)

#define KFD_IOC_ALLOC_MEM_FLAGS_GTT           (1 << 1)
#define KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE      (1u << 31)
#define KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE    (1u << 30)
#define KFD_IOC_ALLOC_MEM_FLAGS_NO_SUBSTITUTE (1u << 28)

#define KFD_DISPATCH_LOG_FW_RECORD_BYTES 20U
#define KFD_IOC_PROFILER_VERSION_NUM     5

enum kfd_profiler_op
{
    KFD_IOC_PROFILER_PMC                    = 0,
    KFD_IOC_PROFILER_VERSION                = 2,
    KFD_IOC_PROFILER_PTL_CONTROL            = 3,
    KFD_IOC_PROFILER_DLOG_REGISTER_BUFFER   = 4,
    KFD_IOC_PROFILER_DLOG_UNREGISTER_BUFFER = 5,
    KFD_IOC_PROFILER_DLOG_OPEN_STREAM       = 6,
    KFD_IOC_PROFILER_DISPATCH_LOG_TARGET    = 9,
};

enum kfd_dispatch_log_op
{
    KFD_DISPATCH_LOG_OP_DISABLE = 0,
    KFD_DISPATCH_LOG_OP_ENABLE  = 1,
};

struct kfd_ioctl_alloc_memory_of_gpu_args
{
    __u64 va_addr;
    __u64 size;
    __u64 handle;
    __u64 mmap_offset;
    __u32 gpu_id;
    __u32 flags;
};

struct kfd_ioctl_free_memory_of_gpu_args
{
    __u64 handle;
};

struct kfd_ioctl_map_memory_to_gpu_args
{
    __u64 handle;
    __u64 device_ids_array_ptr;
    __u32 n_devices;
    __u32 n_success;
};

struct kfd_ioctl_unmap_memory_from_gpu_args
{
    __u64 handle;
    __u64 device_ids_array_ptr;
    __u32 n_devices;
    __u32 n_success;
};

struct kfd_ioctl_export_dmabuf_args
{
    __u64 handle;
    __u32 flags;
    __u32 dmabuf_fd;
};

struct kfd_process_device_apertures
{
    __u64 lds_base;
    __u64 lds_limit;
    __u64 scratch_base;
    __u64 scratch_limit;
    __u64 gpuvm_base;
    __u64 gpuvm_limit;
    __u32 gpu_id;
    __u32 pad;
};

struct kfd_ioctl_get_process_apertures_new_args
{
    __u64 kfd_process_device_apertures_ptr;
    __u32 num_of_nodes;
    __u32 pad;
};

struct kfd_ioctl_dlog_register_buffer_args
{
    __u32 gpu_id;
    __u32 buffer_size;
    __u64 buffer_addr;
};

struct kfd_ioctl_dlog_unregister_buffer_args
{
    __u32 gpu_id;
    __u32 pad;
};

#define KFD_DLOG_OPEN_F_RAW_MMAP     (1u << 0)
#define KFD_DLOG_OPEN_F_READ_RECORDS (1u << 1)

struct kfd_ioctl_dlog_open_stream_args
{
    __u32 gpu_id;
    __u32 target_pid;
    __u32 pad;
    __u32 flags;
    __s32 stream_fd;
};

struct kfd_ioctl_dispatch_log_target_args
{
    __u32 target_pid;
    __u32 gpu_id;
    __u32 op;
    __u32 region_records;
    __s32 dmabuf_fd_out;
    __u32 num_regions_out;
    __u32 region_records_out;
    __u32 record_bytes_out;
    __u64 wptr_array_off_out;
    __u64 rptr_array_off_out;
    __u64 signal_array_off_out;
};

struct kfd_ioctl_profiler_args
{
    __u32 op;
    union
    {
        __u32                                        version;
        struct kfd_ioctl_dlog_register_buffer_args   dlog_register;
        struct kfd_ioctl_dlog_unregister_buffer_args dlog_unregister;
        struct kfd_ioctl_dlog_open_stream_args       dlog_open;
        __u64                                        reserved[7];
    };
};

struct legacy_kfd_ioctl_profiler_args
{
    __u32 op;
    union
    {
        struct kfd_ioctl_dispatch_log_target_args dispatch_log_target;
    };
};

#define KFD_DLOG_STREAM_ABI_VERSION       2
#define KFD_DLOG_STREAM_RECORD_BYTES      21U
#define KFD_DLOG_STREAM_MODE_RAW_MMAP     1U
#define KFD_DLOG_STREAM_MODE_READ_RECORDS 2U

#define KFD_DLOG_STATUS_SOURCE_OVERRUN    (1ULL << 0)
#define KFD_DLOG_STATUS_PADDING_SKIPPED   (1ULL << 1)
#define KFD_DLOG_STATUS_TARGET_EXITED     (1ULL << 2)
#define KFD_DLOG_STATUS_MQD_UNBIND_FAILED (1ULL << 3)
#define KFD_DLOG_STATUS_BACKING_RETAINED  (1ULL << 4)
#define KFD_DLOG_STATUS_FATAL             (1ULL << 5)

struct kfd_dlog_stream_record
{
    __u8 pipe;
    __u8 payload[KFD_DISPATCH_LOG_FW_RECORD_BYTES];
};

struct kfd_dlog_stream_info
{
    __u32 abi_version;
    __u32 mode;
    __u32 fw_record_size;
    __u32 stream_record_size;
    __u32 num_regions;
    __u32 region_record_count;
    __u64 buffer_size;
    __u64 mmap_size;
    __u64 records_offset;
    __u64 wptr_offset;
    __u64 rptr_offset;
    __u64 signal_offset;
    __u32 gpu_id;
    __u32 target_pid;
    __u32 pasid;
    __u32 flags;
};

struct kfd_dlog_stream_status
{
    __u64 status;
    __u64 records_read;
    __u64 source_overruns;
    __u64 copy_faults;
    __u64 target_exit_count;
    __u64 mqd_unbind_failures;
    __u64 backing_retained_count;
};

struct kfd_dlog_mem_span
{
    __u64 offset;
    __u64 size;
};

struct kfd_dlog_layout
{
    __u32                    num_pipes;
    __u32                    slot_bytes;
    __u32                    slots_per_pipe;
    __u32                    pad;
    struct kfd_dlog_mem_span records;
    struct kfd_dlog_mem_span wptr;
    struct kfd_dlog_mem_span rptr;
    struct kfd_dlog_mem_span signal;
};

#define KFD_DLOG_STREAM_IOC_INFO   _IOR('D', 1, struct kfd_dlog_stream_info)
#define KFD_DLOG_STREAM_IOC_STATUS _IOR('D', 2, struct kfd_dlog_stream_status)

#define AMDKFD_IOC_PROFILER            AMDKFD_IOWR(0x28, struct kfd_ioctl_profiler_args)
#define AMDKFD_IOC_ALLOC_MEMORY_OF_GPU AMDKFD_IOWR(0x16, struct kfd_ioctl_alloc_memory_of_gpu_args)
#define AMDKFD_IOC_FREE_MEMORY_OF_GPU  AMDKFD_IOW(0x17, struct kfd_ioctl_free_memory_of_gpu_args)
#define AMDKFD_IOC_MAP_MEMORY_TO_GPU   AMDKFD_IOWR(0x18, struct kfd_ioctl_map_memory_to_gpu_args)
#define AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU                                                           \
    AMDKFD_IOWR(0x19, struct kfd_ioctl_unmap_memory_from_gpu_args)
#define AMDKFD_IOC_EXPORT_DMABUF AMDKFD_IOWR(0x24, struct kfd_ioctl_export_dmabuf_args)
#define AMDKFD_IOC_GET_PROCESS_APERTURES_NEW                                                       \
    AMDKFD_IOWR(0x14, struct kfd_ioctl_get_process_apertures_new_args)

#endif /* KFD_DLOG_UAPI_H */
