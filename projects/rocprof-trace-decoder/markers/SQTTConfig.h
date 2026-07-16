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
#include <cstdlib>
#include <cstring>

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

// GETREG_IMMED encoding: (size_minus_1 << 11) | (offset << 6) | register_id
constexpr uint32_t GETREG_IMMED(uint32_t sz_m1, uint32_t off, uint32_t reg) { return (sz_m1 << 11) | (off << 6) | reg; }

// GCN/CDNA (gfx9): HW_ID register = 4
constexpr uint32_t GFX9_HWREG_WAVE = GETREG_IMMED(3, 0, 4); // WAVE_ID [3:0], 4 bits
constexpr uint32_t GFX9_HWREG_SIMD = GETREG_IMMED(1, 4, 4); // SIMD_ID [5:4], 2 bits
constexpr uint32_t GFX9_HWREG_CU = GETREG_IMMED(3, 8, 4);   // CU_ID [11:8], 4 bits
constexpr uint32_t GFX9_HWREG_WG = GETREG_IMMED(3, 16, 4);  // TG_ID [19:16], 4 bits

// RDNA (gfx10/11/12): HW_ID1=23, HW_ID2=24
constexpr uint32_t RDNA_HWREG_WAVE = GETREG_IMMED(4, 0, 23); // WAVE_ID [4:0], 5 bits
constexpr uint32_t RDNA_HWREG_SIMD = GETREG_IMMED(1, 8, 23); // SIMD_ID [9:8], 2 bits
constexpr uint32_t RDNA_HWREG_CU = GETREG_IMMED(3, 10, 23);  // WGP_ID [13:10], 4 bits
constexpr uint32_t RDNA_HWREG_WG = GETREG_IMMED(4, 16, 24);  // WG_ID [20:16], 5 bits

// Maximum useful mask per HW field (covers all valid IDs)
constexpr uint32_t FULL_WAVE_MASK = 0xFFFFFFFF; // up to 32 waves
constexpr uint32_t FULL_SIMD_MASK = 0xF;        // up to 4 SIMDs
constexpr uint32_t FULL_CU_MASK = 0xFFFF;       // up to 16 CUs/WGPs
constexpr uint32_t FULL_WG_MASK = 0xFFFFFFFF;   // up to 32 WGs

// Bit flags for marker encoding (low 2 bits)
//
//   Bit  0:      exit previous scope (pop top)
//   Bit  1:      enter scope (push)
//   Bits [7:2]:  6-bit ID   (s_ttracedata_imm, IDs 0-63)
//   Bits [31:2]: 30-bit ID  (s_ttracedata, IDs 0-1G)
//
// The marker type (function, user, barrier, memory) is determined by
// looking up the ID in the .sqtt_funcmap section, not from encoding bits.
constexpr uint32_t FLAG_EXIT_PREV = 1u;  // bit 0: exit previous scope
constexpr uint32_t FLAG_ENTER = 1u << 1; // bit 1: entering scope
constexpr uint32_t FLAG_MASK = 0x3;      // all flag bits

// Encode a marker value for s_ttracedata / s_ttracedata_imm
inline uint32_t encodeMarker(uint32_t id, bool enter, bool exit_prev)
{
    uint32_t val = (id << 2);
    if (exit_prev) val |= FLAG_EXIT_PREV;
    if (enter) val |= FLAG_ENTER;
    return val;
}

// Can this encoded marker value fit in s_ttracedata_imm (8-bit)?
inline bool canUseImm(uint32_t encoded) { return encoded <= 0xFF; }

enum class CostMode
{
    InstructionCount,
    WeightedCost
};

// SQTT_MEM_BARRIER selects the strength of the reordering boundary planted
// around every trace marker.
//
//   None:       no fence/clobber. Only the cheap sched_barrier(0) hints
//               survive. Fastest kernel; markers may drift in LDS-pipelined
//               regions.
//   AsmClobber: empty inline asm with "~{memory}" — IR/MIR-level memory
//               reorder constraint, no machine code.
//   Fence:      fence syncscope("workgroup") acq_rel before AND after the
//               marker, tagged as AMDGPU local/LDS synchronization. Preserves
//               the compiler-visible marker boundary while avoiding global
//               cache invalidation for marker-only fences. Default.
enum class MemBarrierMode
{
    None,
    AsmClobber,
    Fence
};

struct SQTTConfig
{
    bool InstrumentBarriers = false;
    CostMode Mode = CostMode::InstructionCount;
    unsigned FunctionThreshold = 0; // 0 = disabled
    unsigned MemoryChunkSize = 0;   // 0 = disabled; otherwise N ops per marker
    unsigned MemoryMaxGap = 0;      // M: max non-memory instructions between ops
    uint32_t WaveMask = 0xFFFFFFFF; // default: all waves (0-31)
    uint32_t SimdMask = 0xF;        // default: all 4 SIMDs
    uint32_t CuMask = 0x3;          // default: CU 0-1
    uint32_t WgMask = 0xFFFFFFFF;   // default: all WGs (0-31)
    MemBarrierMode MemBarrier = MemBarrierMode::Fence;
    bool TraceMemoryAddrs = false; // trace global/buffer/flat addresses
    bool TraceLDSAddrs = false;    // trace LDS addresses
    unsigned ShaderClockBits = 0; // opt in to clock packing explicitly
    unsigned ShaderClockShift = 4;

    bool hasAddressTracing() const { return TraceMemoryAddrs || TraceLDSAddrs; }

    bool needsScopeCheck() const
    {
        return (WaveMask & FULL_WAVE_MASK) != FULL_WAVE_MASK || (SimdMask & FULL_SIMD_MASK) != FULL_SIMD_MASK ||
               (CuMask & FULL_CU_MASK) != FULL_CU_MASK || (WgMask & FULL_WG_MASK) != FULL_WG_MASK;
    }

    static uint32_t parseEnvMask(const char* name, uint32_t def = 0xFFFFFFFF)
    {
        const char* v = std::getenv(name);
        if (!v || v[0] == '\0') return def;
        if (std::strcmp(v, "-1") == 0) return 0xFFFFFFFF;
        char* end = nullptr;
        unsigned long val = std::strtoul(v, &end, 0);
        if (end == v || *end != '\0')
        {
            llvm::errs() << "SQTT: warning: invalid value for " << name << "='" << v << "', using default\n";
            return def;
        }
        return static_cast<uint32_t>(val);
    }

    static bool parseEnvBool(const char* name, bool def)
    {
        const char* v = std::getenv(name);
        if (!v || v[0] == '\0') return def;
        llvm::StringRef s(v);
        return s.equals_insensitive("1") || s.equals_insensitive("y") || s.equals_insensitive("yes") ||
               s.equals_insensitive("true") || s.equals_insensitive("on");
    }

    static MemBarrierMode parseEnvMemBarrier(const char* name, MemBarrierMode def)
    {
        const char* v = std::getenv(name);
        if (!v || v[0] == '\0') return def;
        llvm::StringRef s(v);
        // Numeric: 0=None, 1=AsmClobber, 2=Fence
        if (s == "0") return MemBarrierMode::None;
        if (s == "1") return MemBarrierMode::AsmClobber;
        if (s == "2") return MemBarrierMode::Fence;
        // Named (case-insensitive)
        if (s.equals_insensitive("none") || s.equals_insensitive("off")) return MemBarrierMode::None;
        if (s.equals_insensitive("asm") || s.equals_insensitive("compiler") || s.equals_insensitive("clobber"))
            return MemBarrierMode::AsmClobber;
        if (s.equals_insensitive("fence") || s.equals_insensitive("on") || s.equals_insensitive("hw"))
            return MemBarrierMode::Fence;
        llvm::errs() << "SQTT: warning: invalid value for " << name << "='" << v << "', expected one of "
                     << "{none|asm|fence|0|1|2}, using default\n";
        return def;
    }

    static unsigned parseEnvUnsigned(const char* name, unsigned def)
    {
        const char* v = std::getenv(name);
        if (!v || v[0] == '\0') return def;
        llvm::StringRef s(v);
        unsigned out = 0;
        if (s.getAsInteger(10, out))
        {
            llvm::errs() << "SQTT: warning: invalid value for " << name << "='" << v << "', using default\n";
            return def;
        }
        return out;
    }

    static SQTTConfig fromEnvironment()
    {
        SQTTConfig c;
        c.InstrumentBarriers = parseEnvBool("SQTT_INSTRUMENT_BARRIERS", false);
        c.MemBarrier = parseEnvMemBarrier("SQTT_MEM_BARRIER", MemBarrierMode::Fence);
        c.WaveMask = parseEnvMask("SQTT_SCOPE_WAVE", 0xFFFFFFFF);
        c.SimdMask = parseEnvMask("SQTT_SCOPE_SIMD", 0xF);
        c.CuMask = parseEnvMask("SQTT_SCOPE_CU", 0x3);
        c.WgMask = parseEnvMask("SQTT_SCOPE_WG", 0xFFFFFFFF);
        c.ShaderClockBits = parseEnvUnsigned("SQTT_SHADER_CLOCK_BITS", 0);
        c.ShaderClockShift = parseEnvUnsigned("SQTT_SHADER_CLOCK_SHIFT", 4);

        const char* funcEnv = std::getenv("SQTT_INSTRUMENT_FUNCTIONS");
        if (funcEnv && funcEnv[0] != '\0')
        {
            llvm::StringRef s(funcEnv);
            if (s.consume_front("cost:"))
                c.Mode = CostMode::WeightedCost;
            s.getAsInteger(10, c.FunctionThreshold);
        }

        // SQTT_INSTRUMENT_MEMORY=N:M  (N=chunk size, M=max gap)
        const char* memEnv = std::getenv("SQTT_INSTRUMENT_MEMORY");
        if (memEnv && memEnv[0] != '\0')
        {
            llvm::StringRef s(memEnv);
            llvm::StringRef nStr, mStr;
            std::tie(nStr, mStr) = s.split(':');
            unsigned n = 0, m = 0;
            if (!nStr.getAsInteger(10, n) && !mStr.empty() && !mStr.getAsInteger(10, m) && n > 0)
            {
                c.MemoryChunkSize = n;
                c.MemoryMaxGap = m;
            }
            else
            {
                llvm::errs() << "SQTT: warning: invalid SQTT_INSTRUMENT_MEMORY "
                                "format '"
                             << memEnv << "', expected N:M\n";
            }
        }

        // SQTT_TRACE_ADDRESSES=memory|lds|memory,lds
        const char* addrEnv = std::getenv("SQTT_TRACE_ADDRESSES");
        if (addrEnv && addrEnv[0] != '\0')
        {
            llvm::StringRef s(addrEnv);
            llvm::SmallVector<llvm::StringRef, 2> Parts;
            s.split(Parts, ',');
            for (auto& p : Parts)
            {
                llvm::StringRef t = p.trim();
                if (t == "memory")
                    c.TraceMemoryAddrs = true;
                else if (t == "lds")
                    c.TraceLDSAddrs = true;
                else
                    llvm::errs() << "SQTT: warning: unknown SQTT_TRACE_ADDRESSES "
                                    "category '"
                                 << t << "'\n";
            }
            if (c.hasAddressTracing() && c.MemoryChunkSize)
            {
                llvm::errs() << "SQTT: error: SQTT_TRACE_ADDRESSES and "
                                "SQTT_INSTRUMENT_MEMORY are mutually exclusive\n";
                c.TraceMemoryAddrs = c.TraceLDSAddrs = false;
            }
        }

        return c;
    }
};
