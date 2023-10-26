/*
   Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
   THE SOFTWARE.
*/

#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <optional>

/**
 * ######## Parser Definitions ########
 */
namespace PCSAMPLE
{
enum pcsample_inst_type_issued
{
    TYPE_VALU = 0,
    TYPE_MATRIX,
    TYPE_SCALAR,
    TYPE_TEX,
    TYPE_LDS,
    TYPE_LDS_DIRECT,
    TYPE_FLAT,
    TYPE_EXP,
    TYPE_MESSAGE,
    TYPE_BARRIER,
    TYPE_BRANCH_NOT_TAKEN,
    TYPE_BRANCH_TAKEN,
    TYPE_JUMP,
    TYPE_OTHER,
    TYPE_NO_INST,
    TYPE_DUAL_VALU,
    TYPE_LAST
};

enum pcsample_reason_not_issued
{
    REASON_NOT_AVAILABLE = 0,
    REASON_ALU,
    REASON_WAITCNT,
    REASON_INTERNAL,
    REASON_BARRIER,
    REASON_ARBITER,
    REASON_EX_STALL,
    REASON_OTHER_WAIT,
    REASON_SLEEP,
    REASON_LAST
};

enum pcsample_arb_issue_state
{
    ISSUE_VALU = 0,
    ISSUE_MATRIX,
    ISSUE_LDS,
    ISSUE_LDS_DIRECT,
    ISSUE_SCALAR,
    ISSUE_VMEM_TEX,
    ISSUE_FLAT,
    ISSUE_EXP,
    ISSUE_MISC,
    ISSUE_BRMSG,
    ISSUE_LAST
};
};  // namespace PCSAMPLE

typedef struct
{
    uint8_t valid : 1;
    uint8_t type  : 4;  // 0=reserved, 1=hosttrap, 2=stochastic, 3=perfcounter, >=4 possible v2?
    uint8_t has_stall_reason   : 1;
    uint8_t has_wave_cnt       : 1;
    uint8_t has_memory_counter : 1;
} pcsample_header_v1_t;

typedef struct
{
    uint32_t dual_issue_valu : 1;
    uint32_t inst_type       : 4;

    uint32_t reason_not_issued : 7;
    uint32_t arb_state_issue   : 10;
    uint32_t arb_state_stall   : 10;
} pcsample_snapshot_v1_t;

typedef union
{
    struct
    {
        uint32_t load_cnt   : 6;
        uint32_t store_cnt  : 6;
        uint32_t bvh_cnt    : 3;
        uint32_t sample_cnt : 6;
        uint32_t ds_cnt     : 6;
        uint32_t km_cnt     : 5;
    };
    uint32_t raw;
} pcsample_memorycounters_v1_t;

typedef struct
{
    pcsample_header_v1_t flags;
    uint8_t              chiplet;
    uint8_t              wave_id;
    uint8_t              wave_issued : 1;
    uint8_t              reserved    : 7;
    uint32_t             hw_id;

    uint64_t pc;
    uint64_t exec_mask;
    uint32_t workgroud_id_x;
    uint32_t workgroud_id_y;
    uint32_t workgroud_id_z;

    uint32_t wave_count;
    uint64_t timestamp;
    uint64_t correlation_id;

    pcsample_snapshot_v1_t snapshot;

    pcsample_memorycounters_v1_t memory_counters;
} pcsample_v1_t;

typedef uint64_t (*user_callback_t)(pcsample_v1_t**, uint64_t, void*);

/**
 * The types of errors to be returned by parse_buffer.
 */
enum PCSAMPLE_STATUS
{
    /**
     * No error
     */
    PCSAMPLE_STATUS_SUCCESS = 0,
    /**
     * Input is valid, but the parser detected it was unable to unwrap some correlation_id(s).
     * The returned data is valid except for possible incorrect correlation_ids.
     * Error is nonfatal and parsing will continue.
     */
    PCSAMPLE_STATUS_PARSER_ERROR,
    /**
     * Unknown/generic error
     */
    PCSAMPLE_STATUS_GENERIC_ERROR,
    /**
     * The parser has seen a invalid sample type
     */
    PCSAMPLE_STATUS_INVALID_SAMPLE,
    /**
     * The user callback has returned 0 or a memory size larger than requested
     */
    PCSAMPLE_STATUS_CALLBACK_ERROR,
    /**
     * Upcoming_samples_t has suggested there are more incoming samples than
     * the parser can read without going out of bounds (buffer_size).
     */
    PCSAMPLE_STATUS_OUT_OF_BOUNDS_ERROR,
    /**
     * Invalid GFXIP string was passed to the parser.
     */
    PCSAMPLE_STATUS_INVALID_GFXIP,
    /**
     * Last error type
     */
    PCSAMPLE_STATUS_LAST
};

typedef int pcsample_status_t;
