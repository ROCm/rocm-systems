#pragma once

#include <cstdint>
#include "trace_decoder_types.h"

inline uint64_t get_sa_wgp(uint64_t sa, uint64_t wgp)
{
    return (sa << ROCPROFILER_TRACE_DECODER_CU_SA_SHIFT) + (wgp & ROCPROFILER_TRACE_DECODER_CU_WGP_MASK);
}

enum RdnaType
{
    UNKNOWN = 0,
    VALU_INST,
    IMM_ONE,
    IMMEDIATE,
    WAVE_READY,
    NEW_PC_GFX10,
    WAVE_END,
    WAVE_START,
    WAVE_START_EXT,
    WAVE_ALLOC,
    SHADER_DATA,
    SHADER_DATA_SHORT,
    UTIL_COUNTER_GFX10,
    TIME,
    NOP,
    MISC_GFX10,
    EVENT,
    EVENT_SYNC,
    REG,
    REG_INIT,
    TIMESTAMP,
    HEADER,
    INST,
    LAST_GFX10_TYPE = INST,
    UTIL_COUNTER_GFX11,
    LAST_GFX11_TYPE = UTIL_COUNTER_GFX11,
    // New GFX12
    EXEC_POPCOUNT1,
    EXEC_POPCOUNT3,
    NEW_PC_GFX12,
    LAST_GFX12_TYPE = NEW_PC_GFX12,
    NAVI_TYPE_LAST
};

struct wstart_type_common
{
    uint64_t tm      : 2;
    uint64_t sa      : 1;
    uint64_t simd    : 2;
    uint64_t wgp     : 4;
    uint64_t wid     : 5;
    uint64_t pipe    : 2;
    uint64_t me      : 1;
    uint64_t count   : 10;
    uint64_t isExt   : 1;
    uint64_t wgid    : 5;
    uint64_t last    : 1;
    uint64_t dynvgpr : 1;

    uint64_t SACU() const { return get_sa_wgp(sa, wgp); }
    uint64_t getGPULocation() const { return (SACU() << 7) | (simd << 5) | wid; };

#ifdef SQTT_LOGGING
    std::stringstream print() const
    {
        std::stringstream ss;
        ss << "wgp:" << wgp << " simd:" << simd << " wid:" << wid << " sa:" << sa << " me:" << me << " pipe:" << pipe;
        return ss;
    }
    const char* typestr() const { return "WAVE_START"; };
#endif
};
