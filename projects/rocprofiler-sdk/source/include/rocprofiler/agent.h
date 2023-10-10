// MIT License
//
// Copyright (c) 2023 ROCm Developer Tools
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

#pragma once

#include <rocprofiler/defines.h>
#include <rocprofiler/fwd.h>

#include <hsakmt/hsakmttypes.h>

#include <stdint.h>

ROCPROFILER_EXTERN_C_INIT

/**
 * @defgroup AGENTS Agent Information
 * @brief needs brief description
 *
 * @{
 */

/**
 * @brief Cache information for an agent.
 */
typedef struct rocprofiler_agent_cache_t
{
    uint64_t     processor_id_low;     ///< Identifies the processor number
    uint64_t     size;                 ///< Size of the cache
    uint32_t     level;                ///< Integer representing level: 1, 2, 3, 4, etc.
    uint32_t     cache_line_size;      ///< Cache line size in bytes
    uint32_t     cache_lines_per_tag;  ///< Cache lines per Cache Tag
    uint32_t     association;          ///< Cache Associativity
    uint32_t     latency;              ///< Cache latency in ns
    HsaCacheType type;
} rocprofiler_agent_cache_t;

/**
 * @brief IO link information for an agent.
 */
typedef struct rocprofiler_agent_io_link_t
{
    HSA_IOLINKTYPE type;                 ///< Discoverable IoLink Properties (optional)
    uint32_t       version_major;        ///< Bus interface version (optional)
    uint32_t       version_minor;        ///< Bus interface version (optional)
    uint32_t       node_from;            ///< See @ref rocprofiler_agent_id_t
    uint32_t       node_to;              ///< See @ref rocprofiler_agent_id_t
    uint32_t       weight;               ///< weight factor (derived from CDIT)
    uint32_t       min_latency;          ///< minimum cost of time to transfer (rounded to ns)
    uint32_t       max_latency;          ///< maximum cost of time to transfer (rounded to ns)
    uint32_t       min_bandwidth;        ///< minimum interface Bandwidth in MB/s
    uint32_t       max_bandwidth;        ///< maximum interface Bandwidth in MB/s
    uint32_t recommended_transfer_size;  ///< recommended transfer size to reach maximum bandwidth
                                         ///< in bytes
    HSA_LINKPROPERTY flags;              ///< override flags (may be active for specific platforms)
} rocprofiler_agent_io_link_t;

/**
 * @brief Memory bank information for an agent.
 */
typedef struct rocprofiler_agent_mem_bank_t
{
    HSA_HEAPTYPE       heap_type;
    HSA_MEMORYPROPERTY flags;
    uint32_t           width;        ///< the number of parallel bits of the memoryinterface
    uint32_t           mem_clk_max;  ///< clock for the memory, this allows computing the available
                                     ///< bandwidth to the memory when needed
    uint64_t size_in_bytes;          ///< physical memory size of the memory range in bytes
} rocprofiler_agent_mem_bank_t;

/**
 * @brief Multi-dimensional struct of data
 */
typedef struct rocprofiler_dim3_t
{
    uint32_t x;
    uint32_t y;
    uint32_t z;
} rocprofiler_dim3_t;

/**
 * @brief Agent.
 */
typedef struct rocprofiler_agent_t
{
    uint64_t size;  ///< set to sizeof(rocprofiler_agent_t) by rocprofiler. This can be used for
                    ///< versioning and compatibility handling
    rocprofiler_agent_id_t   id;    ///< Internal opaque identifier
    rocprofiler_agent_type_t type;  ///< Enumeration for identifying the agent type (CPU, GPU, etc.)
    uint32_t cpu_cores_count;  ///< # of latency (= CPU) cores present on this HSA node. This value
                               ///< is 0 for a HSA node with no such cores, e.g a "discrete HSA GPU"
    uint32_t simd_count;  ///< # of HSA throughtput (= GPU) FCompute cores ("SIMD") present in a
                          ///< node. This value is 0 if no FCompute cores are present (e.g. pure
                          ///< "CPU node").
    uint32_t mem_banks_count;  ///< # of discoverable memory bank affinity properties on this
                               ///< "H-NUMA" node.
    uint32_t caches_count;  ///< # of discoverable cache affinity properties on this "H-NUMA"  node.
    uint32_t io_links_count;    ///< # of discoverable IO link affinity properties of this node
                                ///< connecting to other nodes.
    uint32_t cpu_core_id_base;  ///< low value of the logical processor ID of the latency (= CPU)
                                ///< cores available on this node
    uint32_t simd_id_base;      ///< low value of the logical processor ID of the throughput (= GPU)
                                ///< units available on this node
    uint32_t max_waves_per_simd;  ///< This identifies the max. number of launched waves per SIMD.
                                  ///< If NumFComputeCores is 0, this value is ignored.
    uint32_t lds_size_in_kb;      ///< Size of Local Data Store in Kilobytes per SIMD Wavefront
    uint32_t gds_size_in_kb;      ///< Size of Global Data Store in Kilobytes shared across SIMD
                                  ///< Wavefronts
    uint32_t num_gws;             ///< Number of GWS barriers
    uint32_t wave_front_size;   ///< Number of SIMD cores per wavefront executed, typically 64, may
                                ///< be 32 or a different value for some HSA based architectures
    uint32_t num_xcc;           ///< Number of XCC
    uint32_t cu_count;          ///< Number of compute units
    uint32_t array_count;       ///< Number of SIMD arrays
    uint32_t num_shader_banks;  ///< Number of Shader Banks or Shader Engines, typical values are 1
                                ///< or 2
    uint32_t simd_arrays_per_engine;  ///< Number of SIMD arrays per engine
    uint32_t cu_per_simd_array;       ///< Number of Compute Units (CU) per SIMD array
    uint32_t simd_per_cu;             ///< Number of SIMD representing a Compute Unit (CU)
    uint32_t max_slots_scratch_cu;  ///< Number of temp. memory ("scratch") wave slots available to
                                    ///< access, may be 0 if HW has no restrictions
    uint32_t gfx_target_version;    ///< major_version=((value / 10000) % 100)
                                    ///< minor_version=((value / 100) % 100)
                                    ///< patch_version=(value % 100)
    uint16_t vendor_id;             ///< GPU vendor id; 0 on latency (= CPU)-only nodes
    uint16_t device_id;             ///< GPU device id; 0 on latency (= CPU)-only nodes
    uint32_t location_id;       ///< GPU BDF (Bus/Device/function number) - identifies the device
                                ///< location in the overall system
    uint32_t domain;            ///< PCI domain of the GPU
    uint32_t drm_render_minor;  ///< DRM render device minor device number
    uint32_t num_sdma_engines;  ///< number of PCIe optimized SDMA engines
    uint32_t num_sdma_xgmi_engines;       ///< number of XGMI optimized SDMA engines
    uint32_t num_sdma_queues_per_engine;  ///< number of SDMA queue per one engine
    uint32_t num_cp_queues;               ///< number of Compute queues
    uint32_t max_engine_clk_ccompute;     ///< maximum engine clocks for CPU, including any boost
                                          ///< capabilities
    uint32_t max_engine_clk_fcompute;    ///< GPU only. Maximum engine clocks for GPU, including any
                                         ///< boost capabilities
    HSA_ENGINE_VERSION sdma_fw_version;  ///< GPU only
    HSA_ENGINE_ID
    fw_version;  ///< GPU only. Identifier (rev) of the GPU uEngine or Firmware, may be 0
    HSA_CAPABILITY capability;        ///< GPU only
    uint32_t       cu_per_engine;     ///< computed
    uint32_t       max_waves_per_cu;  ///< computed
    uint32_t       family_id;         ///< Family code
    uint32_t workgroup_max_size;  ///< GPU only. Maximum total number of work-items in a work-group.
    uint32_t grid_max_size;   ///< GPU only. Maximum number of fbarriers per work-group. Must be at
                              ///< least 32.
    uint64_t local_mem_size;  ///< GPU only. Local memory size
    uint64_t hive_id;  ///< XGMI Hive the GPU node belongs to in the system. It is an opaque and
                       ///< static number hash created by the PSP
    uint64_t           gpu_id;             ///< GPU only. KFD identifier
    rocprofiler_dim3_t workgroup_max_dim;  ///< GPU only.  Maximum number of work-items of each
                                           ///< dimension of a work-group.
    rocprofiler_dim3_t grid_max_dim;  ///< GPU only. Maximum number of work-items of each dimension
                                      ///< of a grid.
    rocprofiler_agent_mem_bank_t* mem_banks;
    rocprofiler_agent_cache_t*    caches;
    rocprofiler_agent_io_link_t*  io_links;
    const char* name;          ///< Name of the agent. Will be identical to product name for CPU
    const char* vendor_name;   ///< Vendor of agent (will be AMD)
    const char* product_name;  ///< Marketing name
    const char* model_name;    ///< GPU only. Will be something like vega20, mi200, etc.
    rocprofiler_pc_sampling_config_array_t pc_sampling_configs;
} rocprofiler_agent_t;

/**
 * @brief Callback function type for querying the available agents
 *
 * @param [in] agents Array of pointers to agents
 * @param [in] num_agents Number of agents in array
 * @param [in] user_data Data pointer passback
 * @return ::rocprofiler_status_t
 */
typedef rocprofiler_status_t (*rocprofiler_available_agents_cb_t)(
    const rocprofiler_agent_t** agents,
    size_t                      num_agents,
    void*                       user_data);

/**
 * @brief Receive synchronous callback with an array of available agents at moment of invocation
 *
 * @param [in] callback Callback function accepting list of agents
 * @param [in] agent_size Should be set to sizeof(rocprofiler_agent_t)
 * @param [in] user_data Data pointer provided to callback
 * @return ::rocprofiler_status_t
 */
rocprofiler_status_t ROCPROFILER_API
rocprofiler_query_available_agents(rocprofiler_available_agents_cb_t callback,
                                   size_t                            agent_size,
                                   void* user_data) ROCPROFILER_NONNULL(1);

/** @} */

ROCPROFILER_EXTERN_C_FINI
