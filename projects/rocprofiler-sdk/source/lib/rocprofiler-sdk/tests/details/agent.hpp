// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#pragma once

#include <hsa/hsa.h>
#include <hsa/hsa_api_trace.h>
#include <hsa/hsa_ext_amd.h>

#include "fmt/core.h"
#include "fmt/ranges.h"

#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "lib/common/utility.hpp"

namespace rocprofiler
{
namespace test
{
// This structure holds system information acquired through hsa info related
// calls, and is later used for reference when displaying the information.
struct system_info_t
{
    uint16_t            major               = 0;
    uint16_t            minor               = 0;
    uint64_t            timestamp_frequency = 0;
    uint64_t            max_wait            = 0;
    hsa_endianness_t    endianness          = {};
    hsa_machine_model_t machine_model       = {};
};

// This structure holds agent information acquired through hsa info related
// calls, and is later used for reference when displaying the information.
struct agent_info_t
{
    hsa_agent_t                       hsa_agent               = {.handle = 0};
    char                              name[64]                = {'\0'};
    char                              vendor_name[64]         = {'\0'};
    char                              device_mkt_name[64]     = {'\0'};
    hsa_agent_feature_t               agent_feature           = {};
    hsa_profile_t                     device_counting_service = {};
    hsa_default_float_rounding_mode_t float_rounding_mode     = {};
    uint32_t                          max_queue               = 0;
    uint32_t                          queue_min_size          = 0;
    uint32_t                          queue_max_size          = 0;
    hsa_queue_type_t                  queue_type              = {};
    uint32_t                          node                    = 0;
    hsa_device_type_t                 device_type             = {};
    uint32_t                          cache_size[4]           = {0, 0, 0, 0};
    uint32_t                          chip_id                 = 0;
    uint32_t                          cacheline_size          = 0;
    uint32_t                          max_clock_freq          = 0;
    uint32_t                          internal_node_id        = 0;
    uint32_t                          max_addr_watch_pts      = 0;
    uint32_t                          family_id               = 0;
    uint32_t                          ucode_version           = 0;
    uint32_t                          sdma_ucode_version      = 0;
    // HSA_AMD_AGENT_INFO_MEMORY_WIDTH is deprecated, so exclude
    // uint32_t mem_max_freq; Not supported by get_info
    uint32_t   compute_unit           = 0;
    uint32_t   wavefront_size         = 0;
    uint32_t   workgroup_max_size     = 0;
    uint32_t   grid_max_size          = 0;
    uint32_t   fbarrier_max_size      = 0;
    uint32_t   max_waves_per_cu       = 0;
    uint32_t   simds_per_cu           = 0;
    uint32_t   shader_engs            = 0;
    uint32_t   shader_arrs_per_sh_eng = 0;
    hsa_isa_t  agent_isa              = {};
    hsa_dim3_t grid_max_dim           = {0, 0, 0};
    uint16_t   workgroup_max_dim[3]   = {0, 0, 0};
    uint16_t   bdf_id                 = 0;
    bool       fast_f16               = false;
};

// This structure holds memory pool information acquired through hsa info
// related calls, and is later used for reference when displaying the
// information.
struct pool_info_t
{
    uint32_t segment              = 0;
    size_t   pool_size            = 0;
    bool     alloc_allowed        = false;
    size_t   alloc_granule        = 0;
    size_t   pool_alloc_alignment = 0;
    bool     pl_access            = false;
    uint32_t global_flag          = 0;
};

// This structure holds ISA information acquired through hsa info
// related calls, and is later used for reference when displaying the
// information.
struct isa_info_t
{
    char*      name_str               = nullptr;
    uint32_t   workgroup_max_size     = 0;
    hsa_dim3_t grid_max_dim           = {0, 0, 0};
    uint64_t   grid_max_size          = 0;
    uint32_t   fbarrier_max_size      = 0;
    uint16_t   workgroup_max_dim[3]   = {0, 0, 0};
    bool       def_rounding_modes[3]  = {false, false, false};
    bool       base_rounding_modes[3] = {false, false, false};
    bool       mach_models[2]         = {false, false};
    bool       profiles[2]            = {false, false};
    bool       fast_f16               = false;
};

// This structure holds cache information acquired through hsa info
// related calls, and is later used for reference when displaying the
// information.
struct cache_info_t
{
    char*    name_str = nullptr;
    uint8_t  level    = 0;
    uint32_t size     = 0;
};

struct rocm_info
{
    system_info_t             system = {};
    std::vector<agent_info_t> agents = {};
    std::vector<pool_info_t>  pools  = {};
    std::vector<isa_info_t>   isas   = {};
};

int
get_info(rocm_info& info);
}  // namespace test
}  // namespace rocprofiler
