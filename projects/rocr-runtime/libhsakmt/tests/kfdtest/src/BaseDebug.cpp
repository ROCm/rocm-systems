/*
 * Copyright (C) 2023 Advanced Micro Devices, Inc. All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 *
 */

#include "BaseDebug.hpp"
#include <string.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <hsakmt/linux/kfd_ioctl.h>
#include <fcntl.h>
#include "unistd.h"

BaseDebug::BaseDebug(void) {
}

BaseDebug::~BaseDebug(void) {
    /*
     * If the process is still attached, close and destroy the polling file
     * descriptor.  Note that on process termination, the KFD automatically
     * disables processes that are still runtime enabled and debug enabled
     * so we don't do it here.
     */
    if (m_Pid) {
        close(m_Fd.fd);
        unlink(m_Fd_Name);
    }
}

// Creates temp file descriptor and debug attaches.
HSAKMT_STATUS BaseDebug::Attach(struct kfd_runtime_info *rInfo,
                                int rInfoSize,
                                unsigned int pid,
                                uint64_t exceptionEnable) {
    struct kfd_ioctl_dbg_trap_args args = {0};
    char fd_name[32];

    memset(&args, 0x00, sizeof(args));

    mkfifo(m_Fd_Name, 0666);
    m_Fd.fd = open(m_Fd_Name, O_CLOEXEC | O_NONBLOCK | O_RDWR);
    m_Fd.events = POLLIN | POLLRDNORM;

    args.pid = pid;
    args.op = KFD_IOC_DBG_TRAP_ENABLE;
    args.enable.rinfo_ptr = (uint64_t)rInfo;
    args.enable.rinfo_size = rInfoSize;
    args.enable.dbg_fd = m_Fd.fd;
    args.enable.exception_mask = exceptionEnable;

    if (hsaKmtDebugTrapIoctl(&args, NULL, NULL)) {
        close(m_Fd.fd);
        unlink(m_Fd_Name);
        return HSAKMT_STATUS_ERROR;
    }

    m_Pid = pid;

    return HSAKMT_STATUS_SUCCESS;
}


void BaseDebug::Detach(void) {
    struct kfd_ioctl_dbg_trap_args args = {0};

    memset(&args, 0x00, sizeof(args));

    args.pid = m_Pid;
    args.op = KFD_IOC_DBG_TRAP_DISABLE;

    hsaKmtDebugTrapIoctl(&args, NULL, NULL);

    close(m_Fd.fd);
    unlink(m_Fd_Name);

    m_Pid = 0;
    m_Fd.fd = 0;
    m_Fd.events = 0;
}

HSAKMT_STATUS BaseDebug::SendRuntimeEvent(uint64_t exceptions, int gpuId, int queueId)
{
    struct kfd_ioctl_dbg_trap_args args = {0};

    memset(&args, 0x00, sizeof(args));

    args.pid = m_Pid;
    args.op = KFD_IOC_DBG_TRAP_SEND_RUNTIME_EVENT;
    args.send_runtime_event.exception_mask = exceptions;
    args.send_runtime_event.gpu_id = gpuId;
    args.send_runtime_event.queue_id = queueId;

    return hsaKmtDebugTrapIoctl(&args, NULL, NULL);
}

HSAKMT_STATUS BaseDebug::QueryDebugEvent(uint64_t *exceptions,
                                         uint32_t *gpuId, uint32_t *queueId,
                                         int timeoutMsec)
{
    struct kfd_ioctl_dbg_trap_args args = {0};
    HSAKMT_STATUS result;
    int r = poll(&m_Fd, 1, timeoutMsec);

    if (r > 0) {
        char tmp[r];

        read(m_Fd.fd, tmp, sizeof(tmp));
    } else {
        return HSAKMT_STATUS_ERROR;
    }

    memset(&args, 0x00, sizeof(args));

    args.pid = m_Pid;
    args.op = KFD_IOC_DBG_TRAP_QUERY_DEBUG_EVENT;
    args.query_debug_event.exception_mask = *exceptions;

    result = hsaKmtDebugTrapIoctl(&args, NULL, NULL);

    *exceptions = args.query_debug_event.exception_mask;

    if (gpuId)
        *gpuId = args.query_debug_event.gpu_id;

    if (queueId)
        *queueId = args.query_debug_event.queue_id;

    return result;
}

void BaseDebug::SetExceptionsEnabled(uint64_t exceptions)
{
    struct kfd_ioctl_dbg_trap_args args = {0};

    memset(&args, 0x00, sizeof(args));

    args.pid = m_Pid;
    args.op = KFD_IOC_DBG_TRAP_SET_EXCEPTIONS_ENABLED;
    args.set_exceptions_enabled.exception_mask = exceptions;

    hsaKmtDebugTrapIoctl(&args, NULL, NULL);
}

HSAKMT_STATUS BaseDebug::SuspendQueues(unsigned int *numQueues,
                                       HSA_QUEUEID *queues,
                                       uint32_t *queueIds,
                                       uint64_t exceptionsToClear)
{
    struct kfd_ioctl_dbg_trap_args args = {0};

    memset(&args, 0x00, sizeof(args));

    args.pid = m_Pid;
    args.op = KFD_IOC_DBG_TRAP_SUSPEND_QUEUES;
    args.suspend_queues.num_queues = *numQueues;
    args.suspend_queues.queue_array_ptr = (uint64_t)queueIds;
    args.suspend_queues.exception_mask = exceptionsToClear;

    return hsaKmtDebugTrapIoctl(&args, queues, (HSAuint64 *)numQueues);
}

HSAKMT_STATUS BaseDebug::ResumeQueues(unsigned int *numQueues,
                                       HSA_QUEUEID *queues,
                                       uint32_t *queueIds)
{
    struct kfd_ioctl_dbg_trap_args args = {0};

    memset(&args, 0x00, sizeof(args));

    args.pid = m_Pid;
    args.op = KFD_IOC_DBG_TRAP_RESUME_QUEUES;
    args.resume_queues.num_queues = *numQueues;
    args.resume_queues.queue_array_ptr = (uint64_t)queueIds;

    return hsaKmtDebugTrapIoctl(&args, queues, (HSAuint64 *)numQueues);
}

HSAKMT_STATUS BaseDebug::QueueSnapshot(uint64_t exceptionsToClear,
                                  uint64_t snapshotBufAddr,
                                  uint32_t *numSnapshots)
{
    struct kfd_ioctl_dbg_trap_args args = {0};
    HSAKMT_STATUS result;

    memset(&args, 0x00, sizeof(args));

    args.pid = m_Pid;
    args.op = KFD_IOC_DBG_TRAP_GET_QUEUE_SNAPSHOT;
    args.queue_snapshot.exception_mask = exceptionsToClear;
    args.queue_snapshot.snapshot_buf_ptr = snapshotBufAddr;
    args.queue_snapshot.num_queues = *numSnapshots;
    args.queue_snapshot.entry_size = sizeof(struct kfd_queue_snapshot_entry);

    result = hsaKmtDebugTrapIoctl(&args, NULL, NULL);

    *numSnapshots = args.queue_snapshot.num_queues;

    return result;
}

HSAKMT_STATUS BaseDebug::DeviceSnapshot(uint64_t exceptionsToClear,
                                  uint64_t snapshotBufAddr,
                                  uint32_t *numSnapshots)
{
    struct kfd_ioctl_dbg_trap_args args = {0};
    HSAKMT_STATUS result;

    memset(&args, 0x00, sizeof(args));

    args.pid = m_Pid;
    args.op = KFD_IOC_DBG_TRAP_GET_DEVICE_SNAPSHOT;
    args.device_snapshot.exception_mask = exceptionsToClear;
    args.device_snapshot.snapshot_buf_ptr = snapshotBufAddr;
    args.device_snapshot.num_devices = *numSnapshots;
    args.device_snapshot.entry_size = sizeof(struct kfd_dbg_device_info_entry);

    result = hsaKmtDebugTrapIoctl(&args, NULL, NULL);

    *numSnapshots = args.device_snapshot.num_devices;

    return result;
}

HSAKMT_STATUS BaseDebug::SetWaveLaunchOverride(int mode,
                                               uint32_t *enableMask,
                                               uint32_t *supportMask)
{
    struct kfd_ioctl_dbg_trap_args args = {0};
    HSAKMT_STATUS Result;

    memset(&args, 0x00, sizeof(args));

    args.pid = m_Pid;
    args.op = KFD_IOC_DBG_TRAP_SET_WAVE_LAUNCH_OVERRIDE;
    args.launch_override.override_mode = mode;
    args.launch_override.enable_mask = *enableMask;
    args.launch_override.support_request_mask = *supportMask;

    Result = hsaKmtDebugTrapIoctl(&args, NULL, NULL);

    *enableMask = args.launch_override.enable_mask;
    *supportMask = args.launch_override.support_request_mask;

    return Result;
}

HSAKMT_STATUS BaseDebug::SetAddressWatch(uint64_t address,
                                         int mode,
                                         uint64_t mask,
                                         uint32_t gpuId,
                                         uint32_t *id)
{
    struct kfd_ioctl_dbg_trap_args args = {};
    args.pid = m_Pid;
    args.op = KFD_IOC_DBG_TRAP_SET_NODE_ADDRESS_WATCH;
    args.set_node_address_watch.address = address;
    args.set_node_address_watch.mode = mode;
    args.set_node_address_watch.mask = mask;
    args.set_node_address_watch.gpu_id = gpuId;

    HSAKMT_STATUS result = hsaKmtDebugTrapIoctl(&args, NULL, NULL);

    *id = args.set_node_address_watch.id;

    return result;
}

HSAKMT_STATUS BaseDebug::ClearAddressWatch(uint32_t gpuId,
                                           uint32_t id)
{
    struct kfd_ioctl_dbg_trap_args args = {};
    args.pid = m_Pid;
    args.op = KFD_IOC_DBG_TRAP_CLEAR_NODE_ADDRESS_WATCH;
    args.clear_node_address_watch.gpu_id = gpuId;
    args.clear_node_address_watch.id = id;

    return hsaKmtDebugTrapIoctl(&args, NULL, NULL);
}

HSAKMT_STATUS BaseDebug::SetFlags(uint32_t *flags)
{
    struct kfd_ioctl_dbg_trap_args args = {};
    args.pid = m_Pid;
    args.op = KFD_IOC_DBG_TRAP_SET_FLAGS;
    args.set_flags.flags = *flags;

    HSAKMT_STATUS result = hsaKmtDebugTrapIoctl(&args, NULL, NULL);

    *flags = args.set_flags.flags;

    return result;
}
