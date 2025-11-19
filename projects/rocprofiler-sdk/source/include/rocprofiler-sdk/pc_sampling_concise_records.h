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

#include <rocprofiler-sdk/agent.h>
#include <rocprofiler-sdk/defines.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/pc_sampling_common.h>

ROCPROFILER_EXTERN_C_INIT

/**
 * @defgroup PC_SAMPLING_SERVICE PC Sampling
 * @brief Enabling PC (Program Counter) Sampling for GPU Activity
 * @{
 */

/**
 * @brief Information provided by snapshot block (relevant for stochastic PC sampling only)
 * 
 * 8B in total.
 */
typedef struct ROCPROFILER_SDK_EXPERIMENTAL rocprofiler_pc_sampling_snapshot_information_t
{
    uint8_t  wave_issued; ///< 1 - wave issued an instruction at the moment of sampling;
                         ///< 0 - wave didn't issue an instruction at the moment of sampling.
    uint8_t  instruction_type; ///< if wave_issued=1, type of issued instruction (see ::rocprofiler_pc_sampling_instruction_type_t);
                              ///< otherwise might be irrelevant
    uint8_t  no_issue_reason; ///< if wave_issue=0, reason for not issuing the instruction (see ::rocprofiler_pc_sampling_no_issue_reason_t);
                             ///< otherwise irrelevant
    uint8_t  wave_count; ///< number of concurrently running waves on CU (on GFX9) or SIMD (GFX12+) at the moment of sampling
    uint32_t arbiter_state;  ///< upper 4 bits tells about the arbiter version (see ::rocprofiler_pc_sampling_arbiter_version_t)
                             ///< and indicates how to decode subfields 
                             ///< (see ::rocprofiler_pc_sampling_arbiter_state_offset_* and ::rocprofiler_pc_sampling_arbiter_state_width_*)
} rocprofiler_pc_sampling_snapshot_information_t;

/**
 * @brief (experimental) Hardware version enumeration for PC sampling records.
 *
 * Indicates which GPU architecture generated the PC sampling record.
 * This value determines which member of ::rocprofiler_pc_sampling_memory_counterss_t
 * union should be accessed:
 * - GFX12 -> use .gfx12 member
 * - FUTURE -> use .future member
 */
typedef enum ROCPROFILER_SDK_EXPERIMENTAL rocprofiler_pc_sampling_hardware_version_t
{
    ROCPROFILER_PC_SAMPLING_HARDWARE_VERSION_NONE = 0,
    ROCPROFILER_PC_SAMPLING_HARDWARE_VERSION_GFX9,     ///< GFX9 architecture (e.g., MI200, MI300)
    ROCPROFILER_PC_SAMPLING_HARDWARE_VERSION_GFX12,    ///< GFX12 architecture (e.g., Navi4x) - use .gfx12
    ROCPROFILER_PC_SAMPLING_HARDWARE_VERSION_FUTURE,   ///< Future architectures - use .future
    ROCPROFILER_PC_SAMPLING_HARDWARE_VERSION_LAST
} rocprofiler_pc_sampling_hardware_version_t;

/**
 * @brief (experimental) Memory counters for GFX12 architectures.
 *
 */
typedef struct ROCPROFILER_SDK_EXPERIMENTAL rocprofiler_pc_sampling_memory_counters_gfx12_t
{
    uint8_t load_cnt;   ///< Number of VMEM load instructions issued but not yet completed
    uint8_t store_cnt;  ///< Number of VMEM store instructions issued but not yet completed
    uint8_t ds_cnt;     ///< Number of LDS instructions issued but not yet completed
    uint8_t km_cnt;     ///< Number of scalar memory reads/instructions issued but not yet completed
    uint8_t bvh_cnt;    ///< Number of VMEM BVH instructions issued but not yet completed
    uint8_t sample_cnt; ///< Number of VMEM sample instructions issued but not yet completed
} rocprofiler_pc_sampling_memory_counters_gfx12_t;

/**
 * @brief (experimental) Memory counters for future architectures.
 *
 * Total size must not exceed 16 bytes.
 */
typedef struct ROCPROFILER_SDK_EXPERIMENTAL rocprofiler_pc_sampling_memory_counters_future_t
{
    uint8_t load_cnt;       ///< Number of VMEM load instructions issued but not yet completed
    uint8_t store_cnt;      ///< Number of VMEM store instructions issued but not yet completed
    uint8_t ds_cnt;         ///< Number of LDS instructions issued but not yet completed
    uint8_t km_cnt;         ///< Number of scalar memory reads/instructions issued but not yet completed
    uint8_t bvh_cnt;        ///< Number of VMEM BVH instructions issued but not yet completed
    uint8_t sample_cnt;     ///< Number of VMEM sample instructions issued but not yet completed
    uint8_t reserved1;      ///< Reserved for future use
    uint8_t reserved2;      ///< Reserved for future use
    uint8_t reserved3;      ///< Reserved for future use
    uint8_t reserved4;      ///< Reserved for future use
    uint8_t reserved5;      ///< Reserved for future use
    uint8_t reserved6;      ///< Reserved for future use
    uint8_t reserved7;      ///< Reserved for future use
    uint8_t reserved8;      ///< Reserved for future use
    uint8_t reserved_padding[2];  ///< Room for additional counters
} rocprofiler_pc_sampling_memory_counters_future_t;

/**
 * @brief (experimental) Memory counters union.
 *
 * Union providing access to memory counters in architecture-specific layouts.
 * The correct union member to access depends on the pcs_hw_version field in the
 * PC sampling record:
 * - If pcs_hw_version == ROCPROFILER_PC_SAMPLING_HARDWARE_VERSION_GFX12, use .gfx12
 * - If pcs_hw_version == ROCPROFILER_PC_SAMPLING_HARDWARE_VERSION_FUTURE, use .future
 *
 * Total size must not exceed 16 bytes.
 */
typedef union ROCPROFILER_SDK_EXPERIMENTAL rocprofiler_pc_sampling_memory_counterss_t
{
    rocprofiler_pc_sampling_memory_counters_gfx12_t gfx12;   ///< GFX12-specific counters (6 bytes)
    rocprofiler_pc_sampling_memory_counters_future_t future;  ///< Future architecture counters (16 bytes)
    uint64_t raw[2];  ///< Raw access ensuring 16-byte size
} rocprofiler_pc_sampling_memory_counterss_t; 

/* To save space, we use bit fields
typedef struct rocprofiler_pc_sampling_hw_id_record_t
{
    uint8_t chiplet;            ///< chiplet index (3 bits allocated by the ROCr runtime)
    uint8_t wave_id;            ///< wave slot index
    uint8_t simd_id;            ///< SIMD index
    uint8_t pipe_id;            ///< pipe index
    uint8_t cu_or_wgp_id;     
    uint8_t shader_array_id;     ///< Shared array index
    uint8_t shader_engine_id;    ///< shared engine index
    uint8_t workgroup_id;        ///< thread_group index on GFX9, and workgroup index on GFX10+
    uint8_t vm_id;               ///< virtual memory ID
    uint8_t queue_id;            ///< queue id
    uint8_t microengine_id;      ///< ACE (microengine) index
    uint8_t reserved_padding[5]; 

    /// @var cu_or_wgp_id
    /// @brief Compute unit index on GFX9 or workgroup processor index on GFX10+.
} rocprofiler_pc_sampling_hw_id_record_t;
*/

/**
 * @brief Information about where was running when sampled.
 */
typedef struct rocprofiler_pc_sampling_hw_id_record_t
{
    uint64_t value; ///< Upper 4 bits tells the hw_id version. The lower 60 bits are encoded as specifiec in the
                    ///<  that's encoded inside the rocprofiler_pc_sampling_hw_id_v*_offset_t
                    ///< and the rocprofiler_pc_sampling_hw_id_v*_width_t.
} rocprofiler_pc_sampling_hw_id_record_t;


/**
 * @brief 88B in total (experimental) Host-Trap sampling record for GFX9 starting from MI200 and GFX120* (Navi4x)
 */
typedef struct ROCPROFILER_SDK_EXPERIMENTAL rocprofiler_pc_sampling_record_v0_t
{
    // 64B
    uint64_t                           size;         ///< Size of this struct
    rocprofiler_pc_t                   pc;           ///< information about sampled program counter
    uint64_t                           exec_mask;    ///< active SIMD lanes when sampled
    uint64_t                           timestamp;    ///< timestamp when sample is generated
    uint64_t                           dispatch_id;  ///< originating kernel dispatch ID
    rocprofiler_async_correlation_id_t correlation_id;

    // 8B
    rocprofiler_pc_sampling_hw_id_record_t hw_id;  ///< see ::rocprofiler_pc_sampling_hw_id_record_t
    // 12B
    rocprofiler_dim3_t workgroup_position;  ///< work group position in 3D grid
    // 4B
    uint8_t             wave_in_group; ///< wave position in the workgroup
    uint8_t            reserved_padding[3]; ///< reserved for the future use.

    /// @var correlation_id
    /// @brief API launch call id that matches dispatch ID
} rocprofiler_pc_sampling_record_v0_t;

/**
 * @brief 96B in total (experimental) Stochastic Sampling Record for GFX9 starting from MI300.
 */
typedef struct ROCPROFILER_SDK_EXPERIMENTAL rocprofiler_pc_sampling_record_v1_t
{
    // 64B
    uint64_t                           size;         ///< Size of this struct
    rocprofiler_pc_t                   pc;           ///< information about sampled program counter
    uint64_t                           exec_mask;    ///< active SIMD lanes when sampled
    uint64_t                           timestamp;    ///< timestamp when sample is generated
    uint64_t                           dispatch_id;  ///< originating kernel dispatch ID
    rocprofiler_async_correlation_id_t correlation_id;

    // 8B if we introduce bitfields
    rocprofiler_pc_sampling_hw_id_record_t hw_id;
    // 8B
    rocprofiler_pc_sampling_snapshot_information_t snapshot_information;

    // 12B
    rocprofiler_dim3_t workgroup_position;  ///< work group position in 3D grid
    // 4B
    uint8_t            wave_in_group; ///< wave position in the workgroup
    uint8_t            pcs_hw_version; ///< Hardware version (see ::rocprofiler_pc_sampling_hardware_version_t)
    uint8_t            reserved_padding[2]; ///< reserved for the future use.

    /// @var correlation_id
    /// @brief API launch call id that matches dispatch ID
} rocprofiler_pc_sampling_record_v1_t;

/**
 * @brief 104B in total (experimental) Host-Trap Sampling Record on future gen architectures
 */
typedef struct ROCPROFILER_SDK_EXPERIMENTAL rocprofiler_pc_sampling_record_v2_t
{
    // 64B
    uint64_t                           size;         ///< Size of this struct
    rocprofiler_pc_t                   pc;           ///< information about sampled program counter
    uint64_t                           exec_mask;    ///< active SIMD lanes when sampled
    uint64_t                           timestamp;    ///< timestamp when sample is generated
    uint64_t                           dispatch_id;  ///< originating kernel dispatch ID
    rocprofiler_async_correlation_id_t correlation_id;

    // 8B if we introduce bitfields
    rocprofiler_pc_sampling_hw_id_record_t hw_id;
    // 12B
    rocprofiler_dim3_t workgroup_position;  ///< work group position in 3D grid
    // 12B
    rocprofiler_dim3_t reserved0;
    // 8B
    uint8_t            wave_in_group; ///< wave position in the workgroup
    uint8_t            reserved1;
    uint8_t            reserved2;
    uint8_t            reserved3;
    uint8_t            reserved4;
    uint8_t            reserved5;
    uint8_t            reserved6;
    uint8_t            reserved_padding; ///< reserved for the future use.

    /// @var correlation_id
    /// @brief API launch call id that matches dispatch ID
} rocprofiler_pc_sampling_record_v2_t;


/**
 * @brief 128B in total (experimental) Stochastic Sampling Record for future gen architectures.
 */
typedef struct ROCPROFILER_SDK_EXPERIMENTAL rocprofiler_pc_sampling_record_v3_t
{
    // 64B
    uint64_t                           size;         ///< Size of this struct
    rocprofiler_pc_t                   pc;           ///< information about sampled program counter
    uint64_t                           exec_mask;    ///< active SIMD lanes when sampled
    uint64_t                           timestamp;    ///< timestamp when sample is generated
    uint64_t                           dispatch_id;  ///< originating kernel dispatch ID
    rocprofiler_async_correlation_id_t correlation_id;

    // 8B if we introduce bitfields
    rocprofiler_pc_sampling_hw_id_record_t hw_id;
    // 12B
    rocprofiler_dim3_t workgroup_position;  ///< work group position in 3D grid
    // 12B
    rocprofiler_dim3_t reserved0;
    // 8B
    rocprofiler_pc_sampling_snapshot_information_t snapshot_information;
    // 16B (8B if we introduce bitfields)
    rocprofiler_pc_sampling_memory_counterss_t memory_counters;

    // 8B
    uint8_t            wave_in_group; ///< wave position in the workgroup
    uint8_t            pcs_hw_version; ///< Hardware version (see ::rocprofiler_pc_sampling_hardware_version_t)
    uint8_t            reserved1;
    uint8_t            reserved2;
    uint8_t            reserved3;
    uint8_t            reserved4;
    uint8_t            reserved5;
    uint8_t            reserved_padding; ///< reserved for the future use.

    /// @var correlation_id
    /// @brief API launch call id that matches dispatch ID
} rocprofiler_pc_sampling_record_v3_t;

/**
 * @brief HW ID versions.
 *
 * As we're limited with space in the hw_id field, 0 is used as the first version,
 * and ROCPROFILER_PC_SAMPLING_HW_ID_VERSION_NONE doesn't exist.
 */
typedef enum
{
    ROCPROFILER_PC_SAMPLING_HW_ID_VERSION_V0 = 0,  ///< GFX9 hw_id
    ROCPROFILER_PC_SAMPLING_HW_ID_VERSION_LAST
} rocprofiler_pc_sampling_hw_id_version_t;

/**
 * @brief (experimental) IDs of hw_id field.
 *
 * This enumeration contains a union of all fields available in different hw_id versions
 * (see ::rocprofiler_pc_sampling_hw_id_version_t).
 * The offsets and widths for a specific version is encoded inside:
 * - ::rocprofiler_pc_sampling_hw_id_v0_offset_t and
 * ::rocprofiler_pc_sampling_hw_id_v0_width_t for
 * ROCPROFILER_PC_SAMPLING_HW_ID_VERSION_V0
 */
typedef enum
{
    ROCPROFILER_PC_SAMPLING_HW_ID_FIELD_ID_NONE = 0,
    ROCPROFILER_PC_SAMPLING_HW_ID_FIELD_ID_CHIPLET,
    ROCPROFILER_PC_SAMPLING_HW_ID_FIELD_ID_WAVE_ID,
    ROCPROFILER_PC_SAMPLING_HW_ID_FIELD_ID_SIMD_ID,
    ROCPROFILER_PC_SAMPLING_HW_ID_FIELD_ID_PIPE_ID,
    ROCPROFILER_PC_SAMPLING_HW_ID_FIELD_ID_CU_OR_WGP_ID,
    ROCPROFILER_PC_SAMPLING_HW_ID_FIELD_ID_SHADER_ARRAY_ID,
    ROCPROFILER_PC_SAMPLING_HW_ID_FIELD_ID_SHADER_ENGINE_ID,
    ROCPROFILER_PC_SAMPLING_HW_ID_FIELD_ID_WORKGROUP_ID,
    ROCPROFILER_PC_SAMPLING_HW_ID_FIELD_ID_VM_ID,
    ROCPROFILER_PC_SAMPLING_HW_ID_FIELD_ID_QUEUE_ID,
    ROCPROFILER_PC_SAMPLING_HW_ID_FIELD_ID_MICROENGINE_ID,
    ROCPROFILER_PC_SAMPLING_HW_ID_FIELD_ID_PERF_SNAPSHOT_HW_VERSION,
    ROCPROFILER_PC_SAMPLING_HW_ID_FIELD_ID_LAST
} rocprofiler_pc_sampling_hw_id_field_id_t;

/**
 * @brief (experimental) Enumeration for bitfield offsets in the ABI-compatible version of a hw_id field
 * available in host-trap and stochastic PC sampling records.
 *
 * The hw_id field encodes information about the GPU part where wave was executing
 * at the moment of sampling. The hw_id tells a user about the:
 * - chiplet index
 * - wave slot index
 * - SIMD index
 * - pipe index
 * - compute unit index on GFX9 or workgroup processor index on GFX10+
 * - shader array index
 * - shader engine index
 * - thread_group index on GFX9, and workgroup index on GFX10+
 * - virtual memory ID
 * - queue id
 * - ACE (microengine) index
 * - version of the perf snapshot block (please see ::rocprofiler_pc_sampling_perf_snapshot_hw_version_t)
 */
typedef enum ROCPROFILER_SDK_EXPERIMENTAL rocprofiler_pc_sampling_hw_id_v0_offset_t {
    ROCPROFILER_PC_SAMPLING_HW_ID_V0_CHIPLET = 0,
    ROCPROFILER_PC_SAMPLING_HW_ID_V0_WAVE_ID = 6,
    ROCPROFILER_PC_SAMPLING_HW_ID_V0_SIMD_ID = 13,
    ROCPROFILER_PC_SAMPLING_HW_ID_V0_PIPE_ID = 15,
    ROCPROFILER_PC_SAMPLING_HW_ID_V0_CU_OR_WGP_ID = 19,
    ROCPROFILER_PC_SAMPLING_HW_ID_V0_SHADER_ARRAY_ID = 23,
    ROCPROFILER_PC_SAMPLING_HW_ID_V0_SHADER_ENGINE_ID = 24,
    ROCPROFILER_PC_SAMPLING_HW_ID_V0_WORKGROUP_ID = 29,
    ROCPROFILER_PC_SAMPLING_HW_ID_V0_VM_ID = 36,
    ROCPROFILER_PC_SAMPLING_HW_ID_V0_QUEUE_ID = 42,
    ROCPROFILER_PC_SAMPLING_HW_ID_V0_MICROENGINE_ID = 46,
    ROCPROFILER_PC_SAMPLING_HW_ID_V0_PERF_SNAPSHOT_HW_VERSION = 48,
    ROCPROFILER_PC_SAMPLING_HW_ID_V0_RESERVED0 = 56
} rocprofiler_pc_sampling_hw_id_v0_offset_t;

/**
 * @brief (experimental) Enumeration for bitfield widths in the ABI-compatible version of hw_id field.
 * 
 * @see rocprofiler_pc_sampling_hw_id_v0_offset_t for more information about each subfield.
 */
typedef enum ROCPROFILER_SDK_EXPERIMENTAL rocprofiler_pc_sampling_hw_id_v0_width_t {
    ROCPROFILER_PC_SAMPLING_HW_ID_V0_WIDTH_NONE = 0,
    ROCPROFILER_PC_SAMPLING_HW_ID_V0_WIDTH_CHIPLET = 6,
    ROCPROFILER_PC_SAMPLING_HW_ID_V0_WIDTH_WAVE_ID = 7,
    ROCPROFILER_PC_SAMPLING_HW_ID_V0_WIDTH_SIMD_ID = 2,
    ROCPROFILER_PC_SAMPLING_HW_ID_V0_WIDTH_PIPE_ID = 4,
    ROCPROFILER_PC_SAMPLING_HW_ID_V0_WIDTH_CU_OR_WGP_ID = 4,
    ROCPROFILER_PC_SAMPLING_HW_ID_V0_WIDTH_SHADER_ARRAY_ID = 1,
    ROCPROFILER_PC_SAMPLING_HW_ID_V0_WIDTH_SHADER_ENGINE_ID = 5,
    ROCPROFILER_PC_SAMPLING_HW_ID_V0_WIDTH_WORKGROUP_ID = 7,
    ROCPROFILER_PC_SAMPLING_HW_ID_V0_WIDTH_VM_ID = 6,
    ROCPROFILER_PC_SAMPLING_HW_ID_V0_WIDTH_QUEUE_ID = 4,
    ROCPROFILER_PC_SAMPLING_HW_ID_V0_WIDTH_MICROENGINE_ID = 2,
    ROCPROFILER_PC_SAMPLING_HW_ID_V0_WIDTH_PERF_SNAPSHOT_HW_VERSION = 8,  // NOTE: we could reduce this to e.g., 6 bits (64 should be sufficient)
    ROCPROFILER_PC_SAMPLING_HW_ID_V0_WIDTH_RESERVED0 = 8
} rocprofiler_pc_sampling_hw_id_v0_width_t;

/**
 * @brief (experimental) IDs of arbiter_state field.
 *
 * This enumeration contains a union of all fields available in different hardware versions
 * (see ::rocprofiler_pc_sampling_hardware_version_t).
 * The offsets and widths for a specific version are encoded inside:
 * - ::rocprofiler_pc_sampling_arbiter_state_gfx9_field_offset_t and ::rocprofiler_pc_sampling_arbiter_state_gfx9_field_width_t
 *   for ROCPROFILER_PC_SAMPLING_HARDWARE_VERSION_GFX9
 * - ::rocprofiler_pc_sampling_arbiter_state_gfx12_field_offset_t and ::rocprofiler_pc_sampling_arbiter_state_gfx12_field_width_t
 *   for ROCPROFILER_PC_SAMPLING_HARDWARE_VERSION_GFX12
 */
typedef enum {
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_FIELD_ID_NONE=0,
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_FIELD_ID_ISSUE_VALU,
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_FIELD_ID_ISSUE_MATRIX,
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_FIELD_ID_ISSUE_LDS,
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_FIELD_ID_ISSUE_LDS_DIRECT,
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_FIELD_ID_ISSUE_SCALAR,
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_FIELD_ID_ISSUE_TEX,
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_FIELD_ID_ISSUE_VMEM,
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_FIELD_ID_ISSUE_FLAT,
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_FIELD_ID_ISSUE_EXP,   
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_FIELD_ID_ISSUE_MISC,
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_FIELD_ID_ISSUE_BRMSG,
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_FIELD_ID_STALL_VALU,
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_FIELD_ID_STALL_MATRIX,
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_FIELD_ID_STALL_LDS,
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_FIELD_ID_STALL_LDS_DIRECT,
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_FIELD_ID_STALL_SCALAR,
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_FIELD_ID_STALL_TEX,
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_FIELD_ID_STALL_VMEM,
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_FIELD_ID_STALL_FLAT,
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_FIELD_ID_STALL_EXP,
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_FIELD_ID_STALL_MISC,
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_FIELD_ID_STALL_BRMSG,
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_FIELD_ID_DUAL_ISSUE_VALU,
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_FIELD_ID_LAST
} rocprofiler_pc_sampling_arbiter_state_field_id_t;

/**
 * @brief (experimental) Enumeration for bitfield offsets in arbiter_state for GFX9 architectures
 */
typedef enum ROCPROFILER_SDK_EXPERIMENTAL rocprofiler_pc_sampling_arbiter_state_gfx9_field_offset_t {
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX9_FIELD_OFFSET_ISSUE_VALU = 0,
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX9_FIELD_OFFSET_ISSUE_MATRIX = 1,
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX9_FIELD_OFFSET_ISSUE_LDS = 2,
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX9_FIELD_OFFSET_ISSUE_SCALAR = 3,
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX9_FIELD_OFFSET_ISSUE_TEX = 4,
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX9_FIELD_OFFSET_ISSUE_FLAT = 5,
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX9_FIELD_OFFSET_ISSUE_EXP = 6,
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX9_FIELD_OFFSET_ISSUE_MISC = 7,
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX9_FIELD_OFFSET_STALL_VALU = 8,
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX9_FIELD_OFFSET_STALL_MATRIX = 9,
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX9_FIELD_OFFSET_STALL_LDS = 10,
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX9_FIELD_OFFSET_STALL_SCALAR = 11,
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX9_FIELD_OFFSET_STALL_TEX = 12,
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX9_FIELD_OFFSET_STALL_FLAT = 13,
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX9_FIELD_OFFSET_STALL_EXP = 14,
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX9_FIELD_OFFSET_STALL_MISC = 15,
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX9_FIELD_OFFSET_DUAL_ISSUE_VALU = 16,
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX9_FIELD_OFFSET_RESERVED = 17
} rocprofiler_pc_sampling_arbiter_state_gfx9_field_offset_t;

/**
 * @brief (experimental) Enumeration for bitfield widths in arbiter_state for GFX9 architectures
 */
typedef enum ROCPROFILER_SDK_EXPERIMENTAL rocprofiler_pc_sampling_arbiter_state_gfx9_field_width_t {
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX9_FIELD_WIDTH_ISSUE_VALU = 1,
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX9_FIELD_WIDTH_ISSUE_MATRIX = 1,
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX9_FIELD_WIDTH_ISSUE_LDS = 1,
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX9_FIELD_WIDTH_ISSUE_SCALAR = 1,
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX9_FIELD_WIDTH_ISSUE_TEX = 1,
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX9_FIELD_WIDTH_ISSUE_FLAT = 1,
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX9_FIELD_WIDTH_ISSUE_EXP = 1,
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX9_FIELD_WIDTH_ISSUE_MISC = 1,
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX9_FIELD_WIDTH_STALL_VALU = 1,
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX9_FIELD_WIDTH_STALL_MATRIX = 1,
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX9_FIELD_WIDTH_STALL_LDS = 1,
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX9_FIELD_WIDTH_STALL_SCALAR = 1,
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX9_FIELD_WIDTH_STALL_TEX = 1,
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX9_FIELD_WIDTH_STALL_FLAT = 1,
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX9_FIELD_WIDTH_STALL_EXP = 1,
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX9_FIELD_WIDTH_STALL_MISC = 1,
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX9_FIELD_WIDTH_DUAL_ISSUE_VALU = 1,
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX9_FIELD_WIDTH_RESERVED = 11
} rocprofiler_pc_sampling_arbiter_state_gfx9_field_width_t;

/**
 * @brief (experimental) Enumeration for bitfield offsets in arbiter_state for GFX12 architectures
 */
typedef enum ROCPROFILER_SDK_EXPERIMENTAL rocprofiler_pc_sampling_arbiter_state_gfx12_field_offset_t {
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX12_FIELD_OFFSET_ISSUE_VALU = 0,
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX12_FIELD_OFFSET_ISSUE_LDS = 1,
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX12_FIELD_OFFSET_ISSUE_LDS_DIRECT = 2,
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX12_FIELD_OFFSET_ISSUE_SCALAR = 3,
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX12_FIELD_OFFSET_ISSUE_VMEM = 4,
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX12_FIELD_OFFSET_ISSUE_EXP = 5,
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX12_FIELD_OFFSET_ISSUE_BRMSG = 6,
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX12_FIELD_OFFSET_STALL_VALU = 8,
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX12_FIELD_OFFSET_STALL_LDS = 9,
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX12_FIELD_OFFSET_STALL_LDS_DIRECT = 10,
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX12_FIELD_OFFSET_STALL_SCALAR = 11,
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX12_FIELD_OFFSET_STALL_VMEM = 12,
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX12_FIELD_OFFSET_STALL_EXP = 13,
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX12_FIELD_OFFSET_STALL_BRMSG = 14,
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX12_FIELD_OFFSET_RESERVED = 17
} rocprofiler_pc_sampling_arbiter_state_gfx12_field_offset_t;

/**
 * @brief (experimental) Enumeration for bitfield widths in arbiter_state for GFX12 architectures
 */
typedef enum ROCPROFILER_SDK_EXPERIMENTAL rocprofiler_pc_sampling_arbiter_state_gfx12_field_width_t {
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX12_FIELD_WIDTH_ISSUE_VALU = 1,
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX12_FIELD_WIDTH_ISSUE_LDS = 1,
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX12_FIELD_WIDTH_ISSUE_LDS_DIRECT = 1,
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX12_FIELD_WIDTH_ISSUE_SCALAR = 1,
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX12_FIELD_WIDTH_ISSUE_VMEM = 1,
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX12_FIELD_WIDTH_ISSUE_EXP = 1,
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX12_FIELD_WIDTH_ISSUE_BRMSG = 1,
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX12_FIELD_WIDTH_STALL_VALU = 1,
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX12_FIELD_WIDTH_STALL_LDS = 1,
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX12_FIELD_WIDTH_STALL_LDS_DIRECT = 1,
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX12_FIELD_WIDTH_STALL_SCALAR = 1,
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX12_FIELD_WIDTH_STALL_VMEM = 1,
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX12_FIELD_WIDTH_STALL_EXP = 1,
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX12_FIELD_WIDTH_STALL_BRMSG = 1,
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX12_FIELD_WIDTH_RESERVED = 11
} rocprofiler_pc_sampling_arbiter_state_gfx12_field_width_t;

// I guess our users could either do bit extraction on their own, or use the following
// getter that will do that for them in the architecture agnostic way.

/**
 * @brief Get field value from arbiter state
 * @param hw_version The hardware version from pcs_hw_version field
 * @param arbiter_state The 32-bit arbiter state value
 * @param field_id The field ID to extract
 * @return The field value if successful, -1 if field is not supported for the given hardware version
 *
 * NOTE: Internal LLM generated this code, so it could probably be done better.
 */
static inline int
rocprofiler_pc_sampling_get_arbiter_state_field(rocprofiler_pc_sampling_hardware_version_t hw_version,
                                                uint32_t arbiter_state,
                                                rocprofiler_pc_sampling_arbiter_state_field_id_t field_id) {
    // arbiter_state contains all 32 bits of field data (no version bits)
    uint32_t field_data = arbiter_state;

    int offset = -1;
    int width = -1;

    if (hw_version == ROCPROFILER_PC_SAMPLING_HARDWARE_VERSION_GFX9) {
        // Handle GFX9 fields
        switch (field_id) {
            case ROCPROFILER_PC_SAMPLING_ARBITER_STATE_FIELD_ID_ISSUE_VALU:
                offset = ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX9_FIELD_OFFSET_ISSUE_VALU;
                width = ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX9_FIELD_WIDTH_ISSUE_VALU;
                break;
            case ROCPROFILER_PC_SAMPLING_ARBITER_STATE_FIELD_ID_ISSUE_MATRIX:
                offset = ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX9_FIELD_OFFSET_ISSUE_MATRIX;
                width = ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX9_FIELD_WIDTH_ISSUE_MATRIX;
                break;
            case ROCPROFILER_PC_SAMPLING_ARBITER_STATE_FIELD_ID_ISSUE_LDS:
                offset = ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX9_FIELD_OFFSET_ISSUE_LDS;
                width = ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX9_FIELD_WIDTH_ISSUE_LDS;
                break;
            case ROCPROFILER_PC_SAMPLING_ARBITER_STATE_FIELD_ID_ISSUE_SCALAR:
                offset = ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX9_FIELD_OFFSET_ISSUE_SCALAR;
                width = ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX9_FIELD_WIDTH_ISSUE_SCALAR;
                break;
            case ROCPROFILER_PC_SAMPLING_ARBITER_STATE_FIELD_ID_ISSUE_TEX:
                offset = ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX9_FIELD_OFFSET_ISSUE_TEX;
                width = ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX9_FIELD_WIDTH_ISSUE_TEX;
                break;
            case ROCPROFILER_PC_SAMPLING_ARBITER_STATE_FIELD_ID_ISSUE_FLAT:
                offset = ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX9_FIELD_OFFSET_ISSUE_FLAT;
                width = ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX9_FIELD_WIDTH_ISSUE_FLAT;
                break;
            case ROCPROFILER_PC_SAMPLING_ARBITER_STATE_FIELD_ID_ISSUE_EXP:
                offset = ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX9_FIELD_OFFSET_ISSUE_EXP;
                width = ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX9_FIELD_WIDTH_ISSUE_EXP;
                break;
            case ROCPROFILER_PC_SAMPLING_ARBITER_STATE_FIELD_ID_ISSUE_MISC:
                offset = ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX9_FIELD_OFFSET_ISSUE_MISC;
                width = ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX9_FIELD_WIDTH_ISSUE_MISC;
                break;
            case ROCPROFILER_PC_SAMPLING_ARBITER_STATE_FIELD_ID_STALL_VALU:
                offset = ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX9_FIELD_OFFSET_STALL_VALU;
                width = ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX9_FIELD_WIDTH_STALL_VALU;
                break;
            case ROCPROFILER_PC_SAMPLING_ARBITER_STATE_FIELD_ID_STALL_MATRIX:
                offset = ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX9_FIELD_OFFSET_STALL_MATRIX;
                width = ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX9_FIELD_WIDTH_STALL_MATRIX;
                break;
            case ROCPROFILER_PC_SAMPLING_ARBITER_STATE_FIELD_ID_STALL_LDS:
                offset = ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX9_FIELD_OFFSET_STALL_LDS;
                width = ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX9_FIELD_WIDTH_STALL_LDS;
                break;
            case ROCPROFILER_PC_SAMPLING_ARBITER_STATE_FIELD_ID_STALL_SCALAR:
                offset = ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX9_FIELD_OFFSET_STALL_SCALAR;
                width = ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX9_FIELD_WIDTH_STALL_SCALAR;
                break;
            case ROCPROFILER_PC_SAMPLING_ARBITER_STATE_FIELD_ID_STALL_TEX:
                offset = ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX9_FIELD_OFFSET_STALL_TEX;
                width = ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX9_FIELD_WIDTH_STALL_TEX;
                break;
            case ROCPROFILER_PC_SAMPLING_ARBITER_STATE_FIELD_ID_STALL_FLAT:
                offset = ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX9_FIELD_OFFSET_STALL_FLAT;
                width = ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX9_FIELD_WIDTH_STALL_FLAT;
                break;
            case ROCPROFILER_PC_SAMPLING_ARBITER_STATE_FIELD_ID_STALL_EXP:
                offset = ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX9_FIELD_OFFSET_STALL_EXP;
                width = ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX9_FIELD_WIDTH_STALL_EXP;
                break;
            case ROCPROFILER_PC_SAMPLING_ARBITER_STATE_FIELD_ID_STALL_MISC:
                offset = ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX9_FIELD_OFFSET_STALL_MISC;
                width = ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX9_FIELD_WIDTH_STALL_MISC;
                break;
            case ROCPROFILER_PC_SAMPLING_ARBITER_STATE_FIELD_ID_DUAL_ISSUE_VALU:
                offset = ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX9_FIELD_OFFSET_DUAL_ISSUE_VALU;
                width = ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX9_FIELD_WIDTH_DUAL_ISSUE_VALU;
                break;
            default:
                return -1;
        }
    }
    else if (hw_version == ROCPROFILER_PC_SAMPLING_HARDWARE_VERSION_GFX12)
    {
        // Handle GFX12 fields
        switch (field_id) {
            case ROCPROFILER_PC_SAMPLING_ARBITER_STATE_FIELD_ID_ISSUE_VALU:
                offset = ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX12_FIELD_OFFSET_ISSUE_VALU;
                width = ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX12_FIELD_WIDTH_ISSUE_VALU;
                break;
            case ROCPROFILER_PC_SAMPLING_ARBITER_STATE_FIELD_ID_ISSUE_LDS:
                offset = ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX12_FIELD_OFFSET_ISSUE_LDS;
                width = ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX12_FIELD_WIDTH_ISSUE_LDS;
                break;
            case ROCPROFILER_PC_SAMPLING_ARBITER_STATE_FIELD_ID_ISSUE_LDS_DIRECT:
                offset = ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX12_FIELD_OFFSET_ISSUE_LDS_DIRECT;
                width = ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX12_FIELD_WIDTH_ISSUE_LDS_DIRECT;
                break;
            case ROCPROFILER_PC_SAMPLING_ARBITER_STATE_FIELD_ID_ISSUE_SCALAR:
                offset = ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX12_FIELD_OFFSET_ISSUE_SCALAR;
                width = ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX12_FIELD_WIDTH_ISSUE_SCALAR;
                break;
            case ROCPROFILER_PC_SAMPLING_ARBITER_STATE_FIELD_ID_ISSUE_VMEM:
                offset = ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX12_FIELD_OFFSET_ISSUE_VMEM;
                width = ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX12_FIELD_WIDTH_ISSUE_VMEM;
                break;
            case ROCPROFILER_PC_SAMPLING_ARBITER_STATE_FIELD_ID_ISSUE_EXP:
                offset = ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX12_FIELD_OFFSET_ISSUE_EXP;
                width = ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX12_FIELD_WIDTH_ISSUE_EXP;
                break;
            case ROCPROFILER_PC_SAMPLING_ARBITER_STATE_FIELD_ID_ISSUE_BRMSG:
                offset = ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX12_FIELD_OFFSET_ISSUE_BRMSG;
                width = ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX12_FIELD_WIDTH_ISSUE_BRMSG;
                break;
            case ROCPROFILER_PC_SAMPLING_ARBITER_STATE_FIELD_ID_STALL_VALU:
                offset = ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX12_FIELD_OFFSET_STALL_VALU;
                width = ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX12_FIELD_WIDTH_STALL_VALU;
                break;
            case ROCPROFILER_PC_SAMPLING_ARBITER_STATE_FIELD_ID_STALL_LDS:
                offset = ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX12_FIELD_OFFSET_STALL_LDS;
                width = ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX12_FIELD_WIDTH_STALL_LDS;
                break;
            case ROCPROFILER_PC_SAMPLING_ARBITER_STATE_FIELD_ID_STALL_LDS_DIRECT:
                offset = ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX12_FIELD_OFFSET_STALL_LDS_DIRECT;
                width = ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX12_FIELD_WIDTH_STALL_LDS_DIRECT;
                break;
            case ROCPROFILER_PC_SAMPLING_ARBITER_STATE_FIELD_ID_STALL_SCALAR:
                offset = ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX12_FIELD_OFFSET_STALL_SCALAR;
                width = ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX12_FIELD_WIDTH_STALL_SCALAR;
                break;
            case ROCPROFILER_PC_SAMPLING_ARBITER_STATE_FIELD_ID_STALL_VMEM:
                offset = ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX12_FIELD_OFFSET_STALL_VMEM;
                width = ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX12_FIELD_WIDTH_STALL_VMEM;
                break;
            case ROCPROFILER_PC_SAMPLING_ARBITER_STATE_FIELD_ID_STALL_EXP:
                offset = ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX12_FIELD_OFFSET_STALL_EXP;
                width = ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX12_FIELD_WIDTH_STALL_EXP;
                break;
            case ROCPROFILER_PC_SAMPLING_ARBITER_STATE_FIELD_ID_STALL_BRMSG:
                offset = ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX12_FIELD_OFFSET_STALL_BRMSG;
                width = ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX12_FIELD_WIDTH_STALL_BRMSG;
                break;
            default:
                return -1;
        }
    }


    // If offset or width is -1, the field is not supported
    if (offset == -1 || width == -1) {
        return -1;
    }

    // Extract and return the field value
    uint32_t mask = (1u << width) - 1;
    return (field_data >> offset) & mask;
}

/**
 * @brief Get field value from hw_id
 * @param hw_id The 64-bit hw_id value
 * @param field_id The field ID to extract
 * @return The field value if successful, -1 if field is not supported in hw_id version
 * encoded inside hw_id.
 */
static inline int
rocprofiler_pc_sampling_get_hw_id_field(
    uint64_t                                 hw_id,
    rocprofiler_pc_sampling_hw_id_field_id_t field_id)
{
    // Extract version from upper 4 bits (bits 60-63)
    uint64_t version = (hw_id >> 60) & 0xF;

    // Check if version is valid
    if(version >= ROCPROFILER_PC_SAMPLING_HW_ID_VERSION_LAST)
    {
        return -1;
    }

    // Get the lower 60 bits containing the field data
    uint64_t field_data = hw_id & 0x0FFFFFFFFFFFFFFFULL;

    int offset = -1;
    int width  = -1;

    if(version == ROCPROFILER_PC_SAMPLING_HW_ID_VERSION_V0)
    {
        // Handle V0 fields
        switch(field_id)
        {
            case ROCPROFILER_PC_SAMPLING_HW_ID_FIELD_ID_CHIPLET:
                offset = ROCPROFILER_PC_SAMPLING_HW_ID_V0_CHIPLET;
                width  = ROCPROFILER_PC_SAMPLING_HW_ID_V0_WIDTH_CHIPLET;
                break;
            case ROCPROFILER_PC_SAMPLING_HW_ID_FIELD_ID_WAVE_ID:
                offset = ROCPROFILER_PC_SAMPLING_HW_ID_V0_WAVE_ID;
                width  = ROCPROFILER_PC_SAMPLING_HW_ID_V0_WIDTH_WAVE_ID;
                break;
            case ROCPROFILER_PC_SAMPLING_HW_ID_FIELD_ID_SIMD_ID:
                offset = ROCPROFILER_PC_SAMPLING_HW_ID_V0_SIMD_ID;
                width  = ROCPROFILER_PC_SAMPLING_HW_ID_V0_WIDTH_SIMD_ID;
                break;
            case ROCPROFILER_PC_SAMPLING_HW_ID_FIELD_ID_PIPE_ID:
                offset = ROCPROFILER_PC_SAMPLING_HW_ID_V0_PIPE_ID;
                width  = ROCPROFILER_PC_SAMPLING_HW_ID_V0_WIDTH_PIPE_ID;
                break;
            case ROCPROFILER_PC_SAMPLING_HW_ID_FIELD_ID_CU_OR_WGP_ID:
                offset = ROCPROFILER_PC_SAMPLING_HW_ID_V0_CU_OR_WGP_ID;
                width  = ROCPROFILER_PC_SAMPLING_HW_ID_V0_WIDTH_CU_OR_WGP_ID;
                break;
            case ROCPROFILER_PC_SAMPLING_HW_ID_FIELD_ID_SHADER_ARRAY_ID:
                offset = ROCPROFILER_PC_SAMPLING_HW_ID_V0_SHADER_ARRAY_ID;
                width  = ROCPROFILER_PC_SAMPLING_HW_ID_V0_WIDTH_SHADER_ARRAY_ID;
                break;
            case ROCPROFILER_PC_SAMPLING_HW_ID_FIELD_ID_SHADER_ENGINE_ID:
                offset = ROCPROFILER_PC_SAMPLING_HW_ID_V0_SHADER_ENGINE_ID;
                width  = ROCPROFILER_PC_SAMPLING_HW_ID_V0_WIDTH_SHADER_ENGINE_ID;
                break;
            case ROCPROFILER_PC_SAMPLING_HW_ID_FIELD_ID_WORKGROUP_ID:
                offset = ROCPROFILER_PC_SAMPLING_HW_ID_V0_WORKGROUP_ID;
                width  = ROCPROFILER_PC_SAMPLING_HW_ID_V0_WIDTH_WORKGROUP_ID;
                break;
            case ROCPROFILER_PC_SAMPLING_HW_ID_FIELD_ID_VM_ID:
                offset = ROCPROFILER_PC_SAMPLING_HW_ID_V0_VM_ID;
                width  = ROCPROFILER_PC_SAMPLING_HW_ID_V0_WIDTH_VM_ID;
                break;
            case ROCPROFILER_PC_SAMPLING_HW_ID_FIELD_ID_QUEUE_ID:
                offset = ROCPROFILER_PC_SAMPLING_HW_ID_V0_QUEUE_ID;
                width  = ROCPROFILER_PC_SAMPLING_HW_ID_V0_WIDTH_QUEUE_ID;
                break;
            case ROCPROFILER_PC_SAMPLING_HW_ID_FIELD_ID_MICROENGINE_ID:
                offset = ROCPROFILER_PC_SAMPLING_HW_ID_V0_MICROENGINE_ID;
                width  = ROCPROFILER_PC_SAMPLING_HW_ID_V0_WIDTH_MICROENGINE_ID;
                break;
            case ROCPROFILER_PC_SAMPLING_HW_ID_FIELD_ID_PERF_SNAPSHOT_HW_VERSION:
                offset = ROCPROFILER_PC_SAMPLING_HW_ID_V0_PERF_SNAPSHOT_HW_VERSION;
                width  = ROCPROFILER_PC_SAMPLING_HW_ID_V0_WIDTH_PERF_SNAPSHOT_HW_VERSION;
                break;
            default: return -1;
        }
    }

    // If offset or width is -1, the field is not supported
    if(offset == -1 || width == -1)
    {
        return -1;
    }

    // Extract and return the field value
    uint64_t mask = (1ull << width) - 1;
    return (field_data >> offset) & mask;
}

/** @} */

ROCPROFILER_EXTERN_C_FINI
