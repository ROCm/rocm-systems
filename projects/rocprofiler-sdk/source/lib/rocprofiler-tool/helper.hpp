// MIT License
//
// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include <cxxabi.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>
#include <fstream>
#include <iostream>
#include <map>
#include <ostream>
#include <regex>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "lib/common/filesystem.hpp"

#include <amd_comgr/amd_comgr.h>
#include <hsa/amd_hsa_kernel_code.h>
#include <hsa/hsa.h>
#include <hsa/hsa_api_trace.h>
#include <hsa/hsa_ext_amd.h>
#include <hsa/hsa_ven_amd_aqlprofile.h>
#include <hsa/hsa_ven_amd_loader.h>
#include <rocprofiler/registration.h>
#include <rocprofiler/rocprofiler.h>

#define ROCPROFILER_CALL(result, msg)                                                              \
    {                                                                                              \
        rocprofiler_status_t CHECKSTATUS = result;                                                 \
        if(CHECKSTATUS != ROCPROFILER_STATUS_SUCCESS)                                              \
        {                                                                                          \
            std::string status_msg = rocprofiler_get_status_string(CHECKSTATUS);                   \
            std::cerr << "[" #result "][" << __FILE__ << ":" << __LINE__ << "] " << msg            \
                      << " failed with error code " << CHECKSTATUS << ": " << status_msg           \
                      << std::endl;                                                                \
            std::stringstream errmsg{};                                                            \
            errmsg << "[" #result "][" << __FILE__ << ":" << __LINE__ << "] " << msg " failure ("  \
                   << status_msg << ")";                                                           \
            throw std::runtime_error(errmsg.str());                                                \
        }                                                                                          \
    }

constexpr size_t BUFFER_SIZE_BYTES = 4096;
constexpr size_t WATERMARK         = (BUFFER_SIZE_BYTES / 2);

// This can be different for different architecture
// Lets follow the v1 rocprof
// I will have a kernel id from the rocprofiler
// address the kernel descriptor and access the information
// This works for gfx9 but may not for Navi arch
// Interecept the kernel symbol load build a table for kernel id
// when kenel dispatch callback. Here is the kernel id
// Use the kernel id
typedef struct
{
    uint64_t               grid_size;
    uint64_t               workgroup_size;
    uint64_t               lds_size;
    uint64_t               scratch_size;
    uint64_t               arch_vgpr_count;
    uint64_t               accum_vgpr_count;
    uint64_t               sgpr_count;
    uint64_t               wave_size;
    uint64_t               signal_handle;
    uint64_t               kernel_object;
    rocprofiler_queue_id_t queue_id;
    rocprofiler_agent_t    gpu_agent;

} rocprofiler_tool_kernel_properties_t;

typedef struct
{
    std::vector<rocprofiler_agent_t> gpu_agents_lists;

} rocprofiler_tool_agent_callback_t;

struct kernel_descriptor_t
{
    uint8_t  reserved0[16];
    int64_t  kernel_code_entry_byte_offset;
    uint8_t  reserved1[20];
    uint32_t compute_pgm_rsrc3;
    uint32_t compute_pgm_rsrc1;
    uint32_t compute_pgm_rsrc2;
    uint16_t kernel_code_properties;
    uint8_t  reserved2[6];
};

using rocprofiler_tool_callback_kind_names_t =
    std::map<rocprofiler_callback_tracing_kind_t, const char*>;
using rocprofiler_tool_callback_kind_operation_names_t =
    std::map<rocprofiler_callback_tracing_kind_t, std::map<uint32_t, const char*>>;

struct rocprofiler_tool_callback_name_info_t
{
    rocprofiler_tool_callback_kind_names_t           kind_names      = {};
    rocprofiler_tool_callback_kind_operation_names_t operation_names = {};
};

std::vector<std::string>
GetCounterNames();

void
SetKernelDescriptorName(rocprofiler_address_t kernel_descriptor, const char* name);

void
SetKernelProperties(uint64_t                             correlation_id,
                    rocprofiler_tool_kernel_properties_t kernel_properties);
void
SetKernelProperties(uint64_t                             correlation_id,
                    rocprofiler_tool_kernel_properties_t kernel_properties);

rocprofiler_tool_kernel_properties_t
GetKernelProperties(uint64_t correlation_id);

const char*
GetKernelDescriptorName(rocprofiler_address_t kernel_descriptor);

void
populate_kernel_properties_data(rocprofiler_tool_kernel_properties_t* kernel_properties,
                                const hsa_kernel_dispatch_packet_t    dispatch_packet);

void
TracerFlushRecord(void* data, rocprofiler_callback_tracing_kind_t kind);

std::string
cxa_demangle(std::string_view _mangled_name, int* _status);
