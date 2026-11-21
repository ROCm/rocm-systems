/* Copyright (c) 2026 Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: MIT */

/* ROCm trace contract values: provider names and GUIDs, keywords and levels.
 *
 * Shared by every ROCm library that writes trace events (TraceLogging: LTTng on
 * Linux, ETW on Windows) and every tool that enables or decodes them. C and
 * C++, no dependencies. Runtimes include <rocm/trace/trace.h>, which adds the
 * enable check; tools need only this header.
 *
 * These values are part of the contract between runtimes and tools. Once a
 * ROCm release ships a value, it never changes; later releases may add
 * values, and each addition increments ROCM_TRACE_VALUES_VERSION. */

#ifndef ROCM_TRACE_VALUES_H
#define ROCM_TRACE_VALUES_H

#define ROCM_TRACE_VALUES_VERSION 1

/* ---------------------------------------------------------------- providers
 *
 * One provider per library, named rocm_<library>. ETW identifies a provider
 * by GUID; each GUID is ETW's standard hash of the name (the same as .NET
 * EventSource), so ETW tools also accept "*<name>". LTTng uses the name.
 *
 * _GUID is the canonical string form, for tools. _GUID_INIT is the
 * parenthesized initializer TRACELOGGING_DEFINE_PROVIDER expects. */

/* HIP runtime (CLR, HRX) */
#define ROCM_TRACE_PROVIDER_HIP_NAME "rocm_hip"
#define ROCM_TRACE_PROVIDER_HIP_GUID "07416523-fe27-5716-e852-d18774a03743"
#define ROCM_TRACE_PROVIDER_HIP_GUID_INIT                                      \
  (0x07416523, 0xfe27, 0x5716, 0xe8, 0x52, 0xd1, 0x87, 0x74, 0xa0, 0x37, 0x43)

/* HSA runtime (ROCr) */
#define ROCM_TRACE_PROVIDER_HSA_NAME "rocm_hsa"
#define ROCM_TRACE_PROVIDER_HSA_GUID "2c9aef54-065d-562d-5381-a58b99143ed7"
#define ROCM_TRACE_PROVIDER_HSA_GUID_INIT                                      \
  (0x2c9aef54, 0x065d, 0x562d, 0x53, 0x81, 0xa5, 0x8b, 0x99, 0x14, 0x3e, 0xd7)

/* RCCL */
#define ROCM_TRACE_PROVIDER_RCCL_NAME "rocm_rccl"
#define ROCM_TRACE_PROVIDER_RCCL_GUID "56b90528-a98a-521e-31bc-d4c59f48a44d"
#define ROCM_TRACE_PROVIDER_RCCL_GUID_INIT                                     \
  (0x56b90528, 0xa98a, 0x521e, 0x31, 0xbc, 0xd4, 0xc5, 0x9f, 0x48, 0xa4, 0x4d)

/* rocDecode */
#define ROCM_TRACE_PROVIDER_ROCDECODE_NAME "rocm_rocdecode"
#define ROCM_TRACE_PROVIDER_ROCDECODE_GUID                                     \
  "f9f75514-2de0-5888-e7d3-19849e6dba61"
#define ROCM_TRACE_PROVIDER_ROCDECODE_GUID_INIT                                \
  (0xf9f75514, 0x2de0, 0x5888, 0xe7, 0xd3, 0x19, 0x84, 0x9e, 0x6d, 0xba, 0x61)

/* rocJPEG */
#define ROCM_TRACE_PROVIDER_ROCJPEG_NAME "rocm_rocjpeg"
#define ROCM_TRACE_PROVIDER_ROCJPEG_GUID "ba839d66-8bc5-5e40-0bd4-11d9be55f62d"
#define ROCM_TRACE_PROVIDER_ROCJPEG_GUID_INIT                                  \
  (0xba839d66, 0x8bc5, 0x5e40, 0x0b, 0xd4, 0x11, 0xd9, 0xbe, 0x55, 0xf6, 0x2d)

/* roctx */
#define ROCM_TRACE_PROVIDER_ROCTX_NAME "rocm_roctx"
#define ROCM_TRACE_PROVIDER_ROCTX_GUID "e855a1ea-de6c-5550-4ba2-75eb27a92038"
#define ROCM_TRACE_PROVIDER_ROCTX_GUID_INIT                                    \
  (0xe855a1ea, 0xde6c, 0x5550, 0x4b, 0xa2, 0x75, 0xeb, 0x27, 0xa9, 0x20, 0x38)

/* rocSHMEM */
#define ROCM_TRACE_PROVIDER_ROCSHMEM_NAME "rocm_rocshmem"
#define ROCM_TRACE_PROVIDER_ROCSHMEM_GUID "0fe41f2d-6a0f-5e5e-0c70-6c1f513229a3"
#define ROCM_TRACE_PROVIDER_ROCSHMEM_GUID_INIT                                 \
  (0x0fe41f2d, 0x6a0f, 0x5e5e, 0x0c, 0x70, 0x6c, 0x1f, 0x51, 0x32, 0x29, 0xa3)

/* hipFile */
#define ROCM_TRACE_PROVIDER_HIPFILE_NAME "rocm_hipfile"
#define ROCM_TRACE_PROVIDER_HIPFILE_GUID "40b50d4a-3fda-51ab-9174-215916e80270"
#define ROCM_TRACE_PROVIDER_HIPFILE_GUID_INIT                                  \
  (0x40b50d4a, 0x3fda, 0x51ab, 0x91, 0x74, 0x21, 0x59, 0x16, 0xe8, 0x02, 0x70)

/* ---------------------------------------------------------------- keywords
 *
 * Every event carries the keywords of its group. On ETW a session selects
 * events by keyword mask; on LTTng the keyword becomes a suffix of the event
 * name (bit 0 is ";k0;"), which LTTng sessions match with "*".
 *
 * Control events (marked below) carry no data: the runtime checks whether a
 * session enabled them and changes its behavior. ETW cannot enable a single
 * event, so each control event has a keyword of its own. */

/* API call events */
#define ROCM_TRACE_KW_API 0x01ULL
/* kernel_dispatch, graph_dispatch, kernel_symbol, cmd_handoff, cmd_submit_* */
#define ROCM_TRACE_KW_DISPATCH 0x02ULL
/* state_request (control): re-emit runtime state for a late-attaching tool */
#define ROCM_TRACE_KW_STATE 0x04ULL
/* serialize_request (control): run the process's kernels one at a time */
#define ROCM_TRACE_KW_SERIALIZE 0x08ULL
/* copy_complete (control): time SDMA copies */
#define ROCM_TRACE_KW_COPY_TIME 0x10ULL
/* code_object_bytes (control): emit code-object bytes loaded from memory */
#define ROCM_TRACE_KW_CODE_BYTES 0x20ULL
/* Allocation and free API events, in addition to ROCM_TRACE_KW_API, so a
 * session can trace memory without every API call */
#define ROCM_TRACE_KW_ALLOC 0x40ULL

/* ---------------------------------------------------------------- levels
 *
 * Runtimes write TRACE_LEVEL_INFORMATION and TRACE_LEVEL_VERBOSE through
 * <rocm/trace/trace.h>, which maps them to each transport. These are the
 * numbers a session passes. Information: every event except the hot paths;
 * verbose: functions called in tight loops (HSA signal and queue-index
 * functions). */

#define ROCM_TRACE_ETW_LEVEL_INFORMATION 4
#define ROCM_TRACE_ETW_LEVEL_VERBOSE 5
#define ROCM_TRACE_LTTNG_LEVEL_INFORMATION 6 /* TRACE_INFO */
#define ROCM_TRACE_LTTNG_LEVEL_VERBOSE 14    /* TRACE_DEBUG */

/* ---------------------------------------------------------------- arguments
 *
 * How much of each API call's arguments a process records, set from the
 * ROCM_TRACE_CAPTURE environment variable (none, scalars, full; default full).
 * Events keep their fields at every level; what is not captured is recorded
 * as 0 or an empty string. scalars drops strings and the values a call wrote
 * through output pointers, so no data the caller passed or received by
 * address leaves the process; none also drops scalars and addresses, keeping
 * only return values. */

#define ROCM_TRACE_CAPTURE_NONE 0
#define ROCM_TRACE_CAPTURE_SCALARS 1
#define ROCM_TRACE_CAPTURE_FULL 2

/* String fields record at most this many bytes, except URIs and kernel
 * names. */
#define ROCM_TRACE_STRING_CAP 256

#endif /* ROCM_TRACE_VALUES_H */
