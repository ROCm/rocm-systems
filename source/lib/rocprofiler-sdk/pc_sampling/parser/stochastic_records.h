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

#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/pc_sampling.h>

#include <stdint.h>

/**
 * @brief The header of the @ref rocprofiler_pc_sampling_record_stochastic_v0_t, indicating
 * what fields of the @ref rocprofiler_pc_sampling_record_stochastic_v0_t instance are meaningful
 * for the sample.
 */
typedef struct rocprofiler_pc_sampling_record_stochastic_header_t
{
    uint8_t valid              : 1;  ///< pc sample is valid
    uint8_t has_memory_counter : 1;  ///< pc sample provides memory counters information
                                     ///< via ::rocprofiler_pc_sampling_memory_counters_t
    uint8_t reserved_type : 6;
} rocprofiler_pc_sampling_record_stochastic_header_t;

/**
 * @brief Enumaration describing sampled instruction type.
 */
typedef enum rocprofiler_pc_sampling_instruction_type_t
{
    // Do we need ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_NONE=0? (we defined *_NONE in some other
    // enums ) If so, then parser needs to add offset +1 after determining the type
    ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_VALU = 0,
    ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_MATRIX,
    ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_SCALAR,
    ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_TEX,
    ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_LDS,
    ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_LDS_DIRECT,
    ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_FLAT,
    ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_EXPORT,
    ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_MESSAGE,
    ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_BARRIER,
    ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_BRANCH_NOT_TAKEN,
    ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_BRANCH_TAKEN,
    ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_JUMP,
    ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_OTHER,
    ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_NO_INST,
    ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_DUAL_VALU,
    ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_LAST
} rocprofiler_pc_sampling_instruction_type_t;

/**
 * @brief Enumaration describing reason for not issuing an instruction.
 */
typedef enum pcsample_reason_not_issued
{
    // Do we need ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_NONE=0?  (we defined *_NONE in some
    // other enums ) If so, then parser needs to add offset +1 after determining the reason.
    ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_NOT_AVAILABLE = 0,
    ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_ALU,
    ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_WAITCNT,
    ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_INTERNAL,
    ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_BARRIER,
    ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_ARBITER,
    ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_EX_STALL,
    ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_OTHER_WAIT,
    ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_SLEEP,
    ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_LAST
} rocprofiler_pc_sampling_instruction_not_issued_reason_t;

/**
 * @brief Data provided by stochastic sampling hardware.
 *
 */
typedef struct rocprofiler_pc_sampling_snapshot_v0_t
{
    uint32_t
        reason_not_issued : 4;  ///< The reason for not issuing an instruction.
                                ///< (9 different issue reason fits in 4 bits)
                                ///< The field takes one of the value defined in
                                ///< @ref ::rocprofiler_pc_sampling_instruction_not_issued_reason_t
    uint32_t reserved0                  : 1;  ///< reserved for future use
    uint32_t arb_state_issue_valu       : 1;  ///< arbiter issued a VALU instruction
    uint32_t arb_state_issue_matrix     : 1;  ///< arbiter issued a matrix instruction
    uint32_t arb_state_issue_lds        : 1;  ///< arbiter issued a LDS instruction
    uint32_t arb_state_issue_lds_direct : 1;  ///< arbiter issued a LDS direct instruction
    uint32_t arb_state_issue_scalar     : 1;  ///< arbiter issued a scalar (SALU/SMEM) instruction
    uint32_t arb_state_issue_vmem_tex   : 1;  ///< arbiter issued a texture instruction
    uint32_t arb_state_issue_flat       : 1;  ///< arbiter issued a FLAT instruction
    uint32_t arb_state_issue_exp        : 1;  ///< arbiter issued a export instruction
    uint32_t arb_state_issue_misc       : 1;  ///< arbiter issued a miscellaneous instruction
    uint32_t arb_state_issue_brmsg      : 1;  ///< arbiter issued a branch/message instruction
    uint32_t arb_state_issue_reserved   : 1;  ///< reserved for the future use
    // Replacing `uint32_t arb_state_stall   : 10;`
    uint32_t arb_state_stall_valu : 1;  ///< VALU instruction was stalled when sampled is generated
    uint32_t
        arb_state_stall_matrix   : 1;  ///< matrix instruction was stalled when sampled is generated
    uint32_t arb_state_stall_lds : 1;  ///< LDS instruction was stalled when sampled is generated
    uint32_t arb_state_stall_lds_direct : 1;  ///< LDS direct instruction was stalled when sampled
                                              ///< is generated
    uint32_t arb_state_stall_scalar : 1;      ///< Scalar (SALU/SMEM) instruction was stalled when
                                              ///< sampled is generated
    uint32_t arb_state_stall_vmem_tex : 1;    ///< texture instruction was stalled when sampled is
                                              ///< generated
    uint32_t arb_state_stall_flat : 1;  ///< flat instruction was stalled when sampled is generated
    uint32_t arb_state_stall_exp : 1;  ///< export instruction was stalled when sampled is generated
    uint32_t arb_state_stall_misc : 1;   ///< miscellaneous instruction was stalled when sampled is
                                         ///< generated
    uint32_t arb_state_stall_brmsg : 1;  ///< branch/message instruction was stalled when sampled is
                                         ///< generated
    uint32_t arb_state_state_reserved : 1;  ///< reserved for the future use
    // We have two reserved bits
    uint32_t
        dual_issue_valu : 1;  ///< two VALU instructions issued for coexecution (MI3xx specific)
    uint32_t reserved1  : 1;  ///< reserved for the future use
    uint32_t reserved2  : 3;  ///< reserved for the future use
} rocprofiler_pc_sampling_snapshot_v0_t;

/**
 * @brief Counters of issued instructions.
 */
typedef struct rocprofiler_pc_sampling_memory_counters_t
{
    uint32_t load_cnt : 6;   ///< Counts the number of VMEM load instructions issued but not yet
                             ///< completed.
    uint32_t store_cnt : 6;  ///< Counts the number of VMEM store instructions issued but not yet
                             ///< completed.
    uint32_t
        bvh_cnt : 3;  ///< Counts the number of VMEM BVH instructions issued but not yet completed.
    uint32_t sample_cnt : 6;  ///< Counts the number of VMEM sample instructions issued but not yet
                              ///< completed.
    uint32_t ds_cnt : 6;  ///< Counts the number of LDS instructions issued but not yet completed.
    uint32_t km_cnt : 5;  ///< Counts the number of scalar memory reads and memory instructions
                          ///< issued but not yet completed.
} rocprofiler_pc_sampling_memory_counters_t;

/**
 * @brief ROCProfiler Stochastic PC Sampling Record.
 */
typedef struct rocprofiler_pc_sampling_record_stochastic_v0_t
{
    // TODO: use size to know whether memory counters exist or not
    uint64_t size;  ///< Size of this struct
    rocprofiler_pc_sampling_record_stochastic_header_t
            flags;            ///< defines what fields are relevant for the sample
    uint8_t wave_in_group;    ///< wave position within the workgroup (0-15)
    uint8_t wave_issued : 1;  ///< wave issued the instruction represented with the PC
    uint8_t inst_type   : 5;  ///< instruction type, takes a value defined in @ref
                              ///< ::rocprofiler_pc_sampling_instruction_type_t
    uint8_t                            reserved : 2;  ///< reserved 2 bits must be zero
    rocprofiler_pc_sampling_hw_id_v0_t hw_id;         ///< @see ::rocprofiler_pc_sampling_hw_id_v0_t
    rocprofiler_pc_t                   pc;            ///< information about sampled program counter
    uint64_t                           exec_mask;     ///< active SIMD lanes at the moment sampling
    rocprofiler_dim3_t                 workgroup_id;  ///< wave coordinates within the workgroup
    uint32_t wave_count;   /// active waves on the CU at the moment of sampling
    uint64_t timestamp;    ///< timestamp when sample is generated
    uint64_t dispatch_id;  ///< originating kernel dispatch ID
    rocprofiler_async_correlation_id_t correlation_id;
    rocprofiler_pc_sampling_snapshot_v0_t
        snapshot;  ///< @see ::rocprofiler_pc_sampling_snapshot_v0_t
    rocprofiler_pc_sampling_memory_counters_t
        memory_counters;  ///< @see ::rocprofiler_pc_sampling_memory_counters_t
} rocprofiler_pc_sampling_record_stochastic_v0_t;
