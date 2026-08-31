// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

// Owning translation unit for the dispatch-log UAPI ABI. It includes ONLY
// kfd_dlog_uapi.h (never details/kfd_ioctl.h, which is TU-exclusive with it), so
// the request layout the kernel reads and the firmware record stride are pinned
// here at compile time. Adding a field to kfd_ioctl_dlog_args, growing the union
// past its reserved 32 bytes, or reordering op/pad silently changes the _IOWR
// size baked into AMDKFD_IOC_PROFILER and breaks the handshake; these asserts
// turn that into a build failure instead. This is also the only place the drain's
// kFwRecBytes and the UAPI's KFD_DISPATCH_LOG_FW_RECORD_BYTES are simultaneously
// visible, so their agreement is checked here.

#include "lib/rocprofiler-sdk/kfd/dlog_drain.hpp"  // kFwRecBytes
#include "lib/rocprofiler-sdk/kfd/kfd_dlog_uapi.h"

#include <cstddef>

namespace rocprofiler
{
namespace kfd
{
// op (offset 0) and pad (offset 4) precede the union the kernel reads before
// dispatching; the union is 32 bytes (reserved[8]) and 8-aligned (_align), so the
// whole request is 8 + 32 = 40 bytes. 40 is the _IOWR size in AMDKFD_IOC_PROFILER.
static_assert(sizeof(kfd_ioctl_profiler_args) == 40, "profiler request ABI size changed");
static_assert(alignof(kfd_ioctl_profiler_args) == 8, "profiler request alignment changed");
static_assert(offsetof(kfd_ioctl_profiler_args, op) == 0, "profiler request op moved");
static_assert(offsetof(kfd_ioctl_profiler_args, pad) == 4, "profiler request pad moved");
static_assert(offsetof(kfd_ioctl_profiler_args, version) == 8, "profiler request union moved");

// The whole dlog request must fit the union's reserved 32 bytes; every field is
// pinned by offset so reordering two equal-sized (u32) fields fails the build.
static_assert(sizeof(kfd_ioctl_dlog_args) == 24, "dlog request ABI size changed");
static_assert(sizeof(kfd_ioctl_dlog_args) <= 32, "dlog request no longer fits the union");
static_assert(offsetof(kfd_ioctl_dlog_args, dlog_op) == 0, "dlog request dlog_op moved");
static_assert(offsetof(kfd_ioctl_dlog_args, gpu_id) == 4, "dlog request gpu_id moved");
static_assert(offsetof(kfd_ioctl_dlog_args, target_pid) == 8, "dlog request target_pid moved");
static_assert(offsetof(kfd_ioctl_dlog_args, flags) == 12, "dlog request flags moved");
static_assert(offsetof(kfd_ioctl_dlog_args, buffer_size) == 16, "dlog request buffer_size moved");
static_assert(offsetof(kfd_ioctl_dlog_args, stream_fd) == 20, "dlog request stream_fd moved");

// The stream-info OUT struct read via KFD_DLOG_STREAM_OP_INFO: the reader indexes
// wptr[]/rptr[] and derives geometry from these offsets, so their layout is ABI.
static_assert(sizeof(kfd_dlog_stream_info) == 72, "stream_info ABI size changed");
static_assert(alignof(kfd_dlog_stream_info) == 8, "stream_info alignment changed");
static_assert(offsetof(kfd_dlog_stream_info, abi_version) == 0, "stream_info abi_version moved");
static_assert(offsetof(kfd_dlog_stream_info, fw_record_size) == 4,
              "stream_info fw_record_size moved");
static_assert(offsetof(kfd_dlog_stream_info, num_regions) == 8, "stream_info num_regions moved");
static_assert(offsetof(kfd_dlog_stream_info, region_record_count) == 12,
              "stream_info region_record_count moved");
static_assert(offsetof(kfd_dlog_stream_info, buffer_size) == 16, "stream_info buffer_size moved");
static_assert(offsetof(kfd_dlog_stream_info, mmap_size) == 24, "stream_info mmap_size moved");
static_assert(offsetof(kfd_dlog_stream_info, records_offset) == 32,
              "stream_info records_offset moved");
static_assert(offsetof(kfd_dlog_stream_info, wptr_offset) == 40, "stream_info wptr_offset moved");
static_assert(offsetof(kfd_dlog_stream_info, rptr_offset) == 48, "stream_info rptr_offset moved");
static_assert(offsetof(kfd_dlog_stream_info, gpu_id) == 56, "stream_info gpu_id moved");
static_assert(offsetof(kfd_dlog_stream_info, target_pid) == 60, "stream_info target_pid moved");
static_assert(offsetof(kfd_dlog_stream_info, pasid) == 64, "stream_info pasid moved");
static_assert(offsetof(kfd_dlog_stream_info, flags) == 68, "stream_info flags moved");

// The stream-status OUT struct read via KFD_DLOG_STREAM_OP_STATUS.
static_assert(sizeof(kfd_dlog_stream_status) == 16, "stream_status ABI size changed");
static_assert(offsetof(kfd_dlog_stream_status, status) == 0, "stream_status status moved");
static_assert(offsetof(kfd_dlog_stream_status, target_exit_count) == 8,
              "stream_status target_exit_count moved");

// The firmware record the drain parses out of the ring: 20-byte stride, and each
// field at the byte offset copy_pipes()/pair_records() read. kFwRecBytes and the
// UAPI constant are only simultaneously visible in this TU.
static_assert(sizeof(fw_record) == kFwRecBytes, "fw_record is not the 20-byte stride");
static_assert(kFwRecBytes == KFD_DISPATCH_LOG_FW_RECORD_BYTES,
              "firmware record stride disagrees with the UAPI");
static_assert(offsetof(fw_record, ts_lo) == 0, "fw_record ts_lo moved");
static_assert(offsetof(fw_record, ts_hi) == 4, "fw_record ts_hi moved");
static_assert(offsetof(fw_record, record_type) == 8, "fw_record record_type moved");
static_assert(offsetof(fw_record, dispatch_id) == 12, "fw_record dispatch_id moved");
static_assert(offsetof(fw_record, doorbell_off) == 16, "fw_record doorbell_off moved");
}  // namespace kfd
}  // namespace rocprofiler
