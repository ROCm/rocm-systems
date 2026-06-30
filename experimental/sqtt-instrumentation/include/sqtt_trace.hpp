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

#include <cstdint>

// ============================================================================
// SQTT Trace Marker Header
//
// Provides device-side functions to emit s_ttracedata markers into the SQTT
// trace buffer.
//
// All instrumentation is compile-time gated:
//   - SQTT_ENABLED=0 (default): all functions compile to nothing
//   - SQTT_ENABLED=1: real instrumentation emitted
//
// Scope filtering (SIMD/CU/WG) is handled by the LLVM pass plugin via
// environment variables (SQTT_SCOPE_SIMD, SQTT_SCOPE_CU, SQTT_SCOPE_WG).
// The pass wraps all s_ttracedata calls with scope checks automatically.
//
// Markers can be left in production source code permanently with zero cost
// when SQTT_ENABLED is not set.
//
// --- Address Trace Protocol ---
//
// When SQTT_TRACE_ADDRESSES=memory|lds|memory,lds is set at compile time,
// the pass plugin emits per-lane virtual addresses for every qualifying
// memory operation. Each address trace block in the s_ttracedata stream:
//
//   s_ttracedata  <header>       -- point marker (addr_trace_load/store/lds_load/lds_store)
//   s_ttracedata  exec_lo        -- EXEC mask low 32 bits
//   s_ttracedata  exec_hi        -- EXEC mask high 32 bits (0 on wave32)
//   s_ttracedata  lane0_addr_lo  -- readlane loop: address low 32 bits
//   s_ttracedata  lane0_addr_hi  -- address high 32 bits (omitted for LDS)
//   ...                          -- repeat for all lanes (wave_size from ISA target)
//
// Address widths:
//   memory (global/flat, AS 0/1): 64-bit, 2 tokens per lane
//   lds (AS 3):                   32-bit, 1 token per lane
//   private/scratch (AS 5):       always excluded
//
// Wave sizes:
//   gfx9 (CDNA):   64 lanes
//   gfx10+ (RDNA): 32 lanes
//
// The header marker ID maps to a funcmap entry with an "addr_trace_" prefix.
// The decoder uses this prefix to enter address capture mode and reads
// exactly wave_size lanes of address data.  The wave size is stored in
// the funcmap as a W:N entry (derived from the ISA target: gfx9=64, gfx10+=32).
// ============================================================================

// ---------------------------------------------------------------------------
// Master enable switch
// ---------------------------------------------------------------------------
#ifndef SQTT_ENABLED
#    define SQTT_ENABLED 0
#endif

// ---------------------------------------------------------------------------
// Marker ordering
// ---------------------------------------------------------------------------
// All numeric markers emit a sched_barrier(0) on each side of the
// s_ttracedata to anchor it against backend scheduler reordering. The pass
// plugin additionally plants a stronger reorder boundary around every
// s_ttracedata it sees in IR; that boundary is selected by the env var
// SQTT_MEM_BARRIER (see SPEC).

// ---------------------------------------------------------------------------
// ID encoding conventions
// ---------------------------------------------------------------------------
// Legacy markers share the same 2-bit flag layout:
//
//   Bit  0:      exit previous scope (pop top)
//   Bit  1:      enter scope (push)
//   Bits [7:2]:  6-bit ID   (s_ttracedata_imm, IDs 0-63)
//   Bits [31:2]: 30-bit ID  (s_ttracedata, IDs 0-1G)
//
// Decoding (works for both):
//   exit_prev = val & 1
//   is_enter  = (val >> 1) & 1
//   id        = val >> 2
//
// On gfx10+ targets, IDs 1-63 may use s_ttracedata_imm (8-bit immediate,
// faster). Larger IDs fall back to s_ttracedata (32-bit m0). When the pass
// enables gfx12 shader-clock packing, marker headers are rewritten to full
// s_ttracedata and the funcmap records the non-legacy layout with M: metadata.
//
// Value 0x0 is a no-op. Exit markers encode as value 1 before any pass-side
// target-specific rewrite.
//
// The marker type (function, user, barrier, memory) is determined by
// looking up the ID in the .sqtt_funcmap section, not from encoding bits.
// IDs are allocated dynamically from a unified pool (no reserved ranges).

#define SQTT_FLAG_EXIT_PREV (1u)      // bit 0: exit previous scope
#define SQTT_FLAG_ENTER     (1u << 1) // bit 1: entering scope

// ============================================================================
// Device-side implementation
// ============================================================================
#if defined(__AMDGCN__) || defined(__HIP_DEVICE_COMPILE__)

// ---------------------------------------------------------------------------
// When SQTT_ENABLED=0: everything is a no-op
// ---------------------------------------------------------------------------
#    if !SQTT_ENABLED

static __device__ __forceinline__ void sqtt_marker_enter(uint32_t) {}
static __device__ __forceinline__ void sqtt_marker_enter(const char*) {}
static __device__ __forceinline__ void sqtt_marker_exit(uint32_t) {}
static __device__ __forceinline__ void sqtt_marker_exit(const char*) {}
static __device__ __forceinline__ void sqtt_marker_point(uint32_t) {}
static __device__ __forceinline__ void sqtt_marker_point(const char*) {}
static __device__ __forceinline__ void sqtt_marker_data(const char*, uint32_t) {}

#    else // SQTT_ENABLED

// ---------------------------------------------------------------------------
// Core marker emission
// ---------------------------------------------------------------------------

// Sentinels for named markers -- the pass plugin replaces calls to these
// with s_ttracedata + scope checks. Intentionally undefined: linker error
// if the pass plugin is not loaded.
extern "C" __device__ void __sqtt_named_marker_enter(const char*);
extern "C" __device__ void __sqtt_named_marker_exit(const char*);
extern "C" __device__ void __sqtt_named_marker_point(const char*);
extern "C" __device__ void __sqtt_named_marker_data(const char*, uint32_t);

// --- Named (string) variants ---

static __device__ __forceinline__ void sqtt_marker_enter(const char* name) { __sqtt_named_marker_enter(name); }

static __device__ __forceinline__ void sqtt_marker_exit(const char* name) { __sqtt_named_marker_exit(name); }

static __device__ __forceinline__ void sqtt_marker_point(const char* name) { __sqtt_named_marker_point(name); }

static __device__ __forceinline__ void sqtt_marker_data(const char* name, uint32_t data)
{
    __sqtt_named_marker_data(name, data);
}

// --- Numeric variants ---
//
// Each call is bracketed by sched_barrier(0). The pass plugin separately
// plants the SQTT_MEM_BARRIER-controlled reorder boundary (none/asm/fence)
// around every s_ttracedata it observes in IR -- there is no per-call
// preprocessor knob.

static __device__ __forceinline__ void sqtt_marker_enter(uint32_t data)
{
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_ttracedata(static_cast<int>((data << 2) | SQTT_FLAG_ENTER));
    __builtin_amdgcn_sched_barrier(0);
}

static __device__ __forceinline__ void sqtt_marker_exit(uint32_t data)
{
    (void) data; // exit always pops top scope, ID not needed
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_ttracedata(static_cast<int>(SQTT_FLAG_EXIT_PREV));
    __builtin_amdgcn_sched_barrier(0);
}

static __device__ __forceinline__ void sqtt_marker_point(uint32_t data)
{
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_ttracedata(static_cast<int>(data << 2));
    __builtin_amdgcn_sched_barrier(0);
}

#    endif // SQTT_ENABLED

#endif // __AMDGCN__ || __HIP_DEVICE_COMPILE__

// ============================================================================
// Host-side utilities
// ============================================================================
#if !defined(__AMDGCN__) && !defined(__HIP_DEVICE_COMPILE__)

#    include <cstdlib>
#    include <cstring>

namespace sqtt
{

// Parse a hex or decimal value from an env var. Returns default_val if unset.
// Supports "0x..." hex prefix and "-1" as shorthand for 0xFFFFFFFF.
inline uint32_t parse_env_mask(const char* name, uint32_t default_val = 0xFFFFFFFFu)
{
    const char* val = std::getenv(name);
    if (!val || val[0] == '\0') return default_val;

    // -1 means all bits set
    if (std::strcmp(val, "-1") == 0) return 0xFFFFFFFFu;

    char* end = nullptr;
    unsigned long v = std::strtoul(val, &end, 0); // auto-detect hex/dec/oct
    if (end == val) return default_val;
    return static_cast<uint32_t>(v);
}

// Read scope config from environment variables.
struct ScopeConfig
{
    uint32_t wave_mask;
    uint32_t simd_mask;
    uint32_t cu_mask;
    uint32_t wg_mask;

    static ScopeConfig from_env()
    {
        return {
            parse_env_mask("SQTT_SCOPE_WAVE", 0xFFFFFFFFu),
            parse_env_mask("SQTT_SCOPE_SIMD", 0xFu),
            parse_env_mask("SQTT_SCOPE_CU", 0x3u),
            parse_env_mask("SQTT_SCOPE_WG", 0xFFFFFFFFu),
        };
    }
};

} // namespace sqtt

#endif // host side
