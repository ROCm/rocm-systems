// MIT License
//
// Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include <stddef.h>
#include <stdint.h>

/**
 * This file describes the instrumentation format for rocprof trace decoder 0.1.5.
 * Instrumentation is optional for decoding, with the exception of rocprof_trace_decoder_gfx9_header_t.
 * Unless specified, all instrumentation packets are written to the USERDATA2 register.
 * This is an experimental feature, and as such the instrumentation may be changed without notice.
 *
 * It is recommended to use code object instrumentation for long traces, to avoid overlapping address spaces
 * if code objects are loaded/unloaded during the trace.
 * When an ID is found, callbacks involving _trace_decoder_pc_t will return {id, vaddr} instead of {0, memory_address}.
 */

/**
 * @brief For gfx9, must be first 8 bytes of the trace binary buffer. Not added in gfx10+.
 */
typedef union rocprof_trace_decoder_gfx9_header_t
{
    struct {
        uint64_t legacy_version : 13; ///< Must be 0x0 or 0x11
        uint64_t gfx9_version2 : 3;   ///< 4: MI200 or earlier - 5: MI300 - 6: MI350
        uint64_t DSIMDM : 4;          ///< Bitmask of SIMDs active
        uint64_t DCU : 5;             ///< Target CU
        uint64_t DSA : 1;             ///< Must be zero
        uint64_t SEID : 6;            ///< Optional: Shader engine ID
        uint64_t double_buffer : 1;   ///< Double buffering mode enabled
        uint64_t reserved2 : 31;
    };
    uint64_t raw;
} rocprof_trace_decoder_gfx9_header_t;

/**
 * @brief Must be first packet on userdata2. Activates instrumentation.
 * The 4 characters must be defined as ASCII '\0ROC'. Instrumentation ignored otherwise.
 * Optionally, a subsequent write can be sent with version number 524801
 */
typedef union rocprof_trace_decoder_instrument_enable_t
{
    struct {
        unsigned int char1 : 8;  ///< '\0'
        unsigned int char2 : 8;  ///< 'R'
        unsigned int char3 : 8;  ///< 'O'
        unsigned int char4 : 8;  ///< 'C'
    };
    unsigned int u32All;
} rocprof_trace_decoder_instrument_enable_t;

/**
 * @brief Header packet for instrumentation.
 * Opcode defines the kind of instrumentation, and type (sometimes) defines a subtype.
 * Header packets are expected to be followed by a 32-bit payload on the same register, except where especified.
 */
typedef union rocprof_trace_decoder_packet_header_t
{
    struct
    {
        unsigned int opcode : 8;  ///< one of rocprof_trace_decoder_packet_opcode_t
        unsigned int type : 4;    ///< one of rocprof_trace_decoder_agent_info_type_t or rocprof_trace_decoder_codeobj_marker_type_t
        unsigned int data20 : 20; ///< Agent data, if rocprof_trace_decoder_agent_info_type_t.
    };
    unsigned int u32All;
} rocprof_trace_decoder_packet_header_t;

typedef enum rocprof_trace_decoder_packet_opcode_t
{
    ROCPROF_TRACE_DECODER_PACKET_OPCODE_CODEOBJ = 4, 
    ROCPROF_TRACE_DECODER_PACKET_OPCODE_RT_TIMESTAMP,
    ROCPROF_TRACE_DECODER_PACKET_OPCODE_AGENT_INFO     ///< Agent info, passed in data20. No payload.

    /// @var ROCPROF_TRACE_DECODER_PACKET_OPCODE_CODEOBJ
    /// @brief Followed by several rocprof_trace_decoder_codeobj_marker_t
    /// Once relevant data is sent, finalize with rocprof_trace_decoder_codeobj_marker_tail_t

    /// @var ROCPROF_TRACE_DECODER_PACKET_OPCODE_RT_TIMESTAMP
    /// @brief Realtime timestamp to correlate the trace with outside information.
    /// Notes: userdata--3--. Gfx9 only. Not necessary for gfx10+.
    /// Instead of a single payload, must be followed by 3x USERDATA3 writes, in order:
    /// 1) Timestamp low 64bits
    /// 2) Timestamp high 64bits
    /// 3) Instant sync timestamp, low 32 bits.
} rocprof_trace_decoder_packet_opcode_t;

typedef enum rocprof_trace_decoder_agent_info_type_t
{
    ROCPROF_TRACE_DECODER_AGENT_INFO_TYPE_RT_FREQUENCY_KHZ = 0,  ///< Realtime TS frequency in Khz
    ROCPROF_TRACE_DECODER_AGENT_INFO_TYPE_COUNTER_INTERVAL,      ///< (gfx9) SQTT counter interval in cycles
    ROCPROF_TRACE_DECODER_AGENT_INFO_TYPE_LAST
} rocprof_trace_decoder_agent_info_type_t;

/**
 * @brief Applies code object instrumentation. Sent as the last code object instrumentation packet.
 * Instead of _ID_LO and _ID_HI, the legacy_id field can be used for setting the ID.
 * IDs can be any (nonzero) user defined number, and will be used in callbacks involving _trace_decoder_pc_t.
 * The combination {id, offset} instead of {0, memory_address} is used to avoid overlapping
 * addresses when code objects are loaded/unloaded during the trace.
 */
typedef union rocprof_trace_decoder_codeobj_marker_tail_t
{
    struct {
        uint32_t isUnload   : 1;  // 0 if code object is being loaded, 1 for unload
        uint32_t bFromStart : 1;  // Has this code object been loaded before thread trace started?
        uint32_t legacy_id  : 30; // Nonzero: Code object ID, if it fits in 30 bits.
    };
    uint32_t raw;
} rocprof_trace_decoder_codeobj_marker_tail_t;

/**
 * @brief Defines the type of code object marker. Followed by 32-bit payload.
 * Send ADDR/SIZE with _LO/_HI combinations, followed by _TAIL to apply the instrumentation.
 */
typedef enum rocprof_trace_decoder_codeobj_marker_type_t
{
    ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_TAIL = 0, ///< Payload is a rocprof_trace_decoder_codeobj_marker_tail_t
    ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_SIZE_LO,
    ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_ADDR_LO,
    ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_ADDR_HI,
    ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_SIZE_HI,
    ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_ID_LO,
    ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_ID_HI,
    ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_LAST
} rocprof_trace_decoder_codeobj_marker_type_t;
