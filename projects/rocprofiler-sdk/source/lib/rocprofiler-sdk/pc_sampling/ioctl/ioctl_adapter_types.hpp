// MIT License
//
// Copyright (c) 2023-2025 ROCm Developer Tools
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <rocprofiler-sdk/fwd.h>

#include "lib/rocprofiler-sdk/context/context.hpp"
#include "lib/rocprofiler-sdk/pc_sampling/types.hpp"

#include <vector>

namespace rocprofiler
{
namespace pc_sampling
{
namespace ioctl
{
#define INVALID_TRACE_ID 0x0

// The data structure copied from the HsaKmt
// Currently, we are using the following status codes:
// 1. ROCPROFILER_IOCTL_STATUS_SUCCESS
// 2. ROCPROFILER_IOCTL_STATUS_ERROR
// 3. ROCPROFILER_IOCTL_STATUS_BUFFER_TOO_SMALL
// 4. ROCPROFILER_IOCTL_STATUS_UNAVAILABLE
// We might replace 1, 2, and 4 with rocprofiler_status_t, but still lacking a counterpart
// for the ROCPROFILER_IOCTL_STATUS_BUFFER_TOO_SMALL
typedef enum rocprofiler_ioctl_status_t
{
    ROCPROFILER_IOCTL_STATUS_SUCCESS = 0,  /// Operation successful // USED
    ROCPROFILER_IOCTL_STATUS_ERROR = 1,  /// General error return if not otherwise specified // USED
    ROCPROFILER_IOCTL_STATUS_DRIVER_MISMATCH =
        2,  /// User mode component is not compatible with kernel HSA driver
    ROCPROFILER_IOCTL_STATUS_INVALID_NODE_UNIT =
        5,  /// KFD identifies node or unit parameter invalid
    ROCPROFILER_IOCTL_STATUS_NO_MEMORY =
        6,  /// No memory available (when allocating queues or memory)
    ROCPROFILER_IOCTL_STATUS_BUFFER_TOO_SMALL =
        7,  /// A buffer needed to handle a request is too small                     //USED
    ROCPROFILER_IOCTL_STATUS_NOT_IMPLEMENTED =
        10,  /// KFD function is not implemented for this set of paramters
    ROCPROFILER_IOCTL_STATUS_UNAVAILABLE = 12,  /// KFD function is not available currently on this
                                                /// // USED node (but may be at a later time)
    ROCPROFILER_IOCTL_STATUS_OUT_OF_RESOURCES =
        13,  /// KFD function request exceeds the resources currently available.
    ROCPROFILER_IOCTL_STATUS_KERNEL_COMMUNICATION_ERROR =
        21,                                               /// user-kernel mode communication failure
    ROCPROFILER_IOCTL_STATUS_KERNEL_ALREADY_OPENED = 22,  /// KFD driver path already opened
    ROCPROFILER_IOCTL_STATUS_HSAMMU_UNAVAILABLE =
        23,  /// ATS/PRI 1.1 (Address Translation Services) not available
             /// (IOMMU driver not installed or not-available)
    ROCPROFILER_IOCTL_STATUS_WAIT_FAILURE              = 30,  /// The wait operation failed
    ROCPROFILER_IOCTL_STATUS_WAIT_TIMEOUT              = 31,  /// The wait operation timed out
    ROCPROFILER_IOCTL_STATUS_MEMORY_ALREADY_REGISTERED = 35,  /// Memory buffer already registered
    ROCPROFILER_IOCTL_STATUS_MEMORY_NOT_REGISTERED     = 36,  /// Memory buffer not registered
    ROCPROFILER_IOCTL_STATUS_MEMORY_ALIGNMENT          = 37,  /// Memory parameter not aligned
} rocprofiler_ioctl_status_t;

typedef struct rocprofiler_ioctl_version_info_s
{
    uint32_t major_version;  /// supported IOCTL interface major version
    uint32_t minor_version;  /// supported IOCTL interface minor version
} rocprofiler_ioctl_version_info_t;

typedef enum rocprofiler_ioctl_pc_sampling_method_kind_t
{
    ROCPROFILER_IOCTL_PC_SAMPLING_METHOD_KIND_NONE        = 0,
    ROCPROFILER_IOCTL_PC_SAMPLING_METHOD_KIND_HOSTTRAP_V1 = 1,
    ROCPROFILER_IOCTL_PC_SAMPLING_METHOD_KIND_STOCHASTIC_V1,
    ROCPROFILER_IOCTL_PC_SAMPLING_METHOD_KIND_LAST,
} rocprofiler_ioctl_pc_sampling_method_kind_t;

typedef enum rocprofiler_ioctl_pc_sampling_unit_interval_t
{
    ROCPROFILER_IOCTL_PC_SAMPLING_UNIT_INTERVAL_MICROSECONDS,
    ROCPROFILER_IOCTL_PC_SAMPLING_UNIT_INTERVAL_CYCLES,
    ROCPROFILER_IOCTL_PC_SAMPLING_UNIT_INTERVAL_INSTRUCTIONS,
} rocprofiler_ioctl_pc_sampling_unit_interval_t;

typedef struct rocprofiler_ioctl_pc_sampling_info_s
{
    uint64_t                                      interval;
    uint64_t                                      interval_min;
    uint64_t                                      interval_max;
    uint64_t                                      flags;
    rocprofiler_ioctl_pc_sampling_method_kind_t   method;
    rocprofiler_ioctl_pc_sampling_unit_interval_t units;
} rocprofiler_ioctl_pc_sampling_info_t;

}  // namespace ioctl
}  // namespace pc_sampling
}  // namespace rocprofiler
