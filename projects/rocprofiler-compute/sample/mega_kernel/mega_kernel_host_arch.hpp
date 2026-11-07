// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Host-side architecture string → category (keeps detection in one table).

#pragma once

#include <cstring>

namespace mega_kernel_host
{

// Order: longer gfx tokens before shorter prefixes (e.g. gfx942 before gfx94).
inline void
detect_architecture(const char* arch, char* out_name, size_t out_name_sz, int* out_type)
{
    if(out_name_sz > 0)
    {
        std::strncpy(out_name, arch, out_name_sz - 1);
        out_name[out_name_sz - 1] = '\0';
    }
    *out_type = 0;

    static const struct
    {
        const char* needle;
        int         typ;
    } map[] = {
        {"gfx1250", 7},
        {"gfx1153", 6}, {"gfx1152", 6}, {"gfx1151", 6}, {"gfx1150", 6},
        {"gfx1201", 5}, {"gfx1200", 4},
        {"gfx950", 3},
        {"gfx942", 2},
        {"gfx90a", 1}, {"gfx908", 1},
    };

    for(const auto& e : map)
    {
        if(std::strstr(arch, e.needle) != nullptr)
        {
            *out_type = e.typ;
            break;
        }
    }
}

/**
 * Whether a failed host memory check is the known gfx90a counter-collection
 * artifact. Under rocprofiler-sdk counter collection on gfx90a the host copy
 * of the output buffer can hold stale values while the ISA path tests pass.
 * @param arch_name GCN arch name, such as "gfx90a:sramecc+:xnack-".
 * @param counter_collection ROCPROF_COUNTER_COLLECTION value, or nullptr.
 * @return true only for gfx90a with counter collection on ("1").
 */
inline bool
memory_verify_failure_is_counter_artifact(const char* arch_name,
                                          const char* counter_collection)
{
    return arch_name != nullptr && counter_collection != nullptr &&
           std::strstr(arch_name, "gfx90a") != nullptr &&
           std::strcmp(counter_collection, "1") == 0;
}

inline const char*
arch_description(int arch_type)
{
    switch(arch_type)
    {
        case 1: return "MI100/MI250 (CDNA1-2/gfx908|gfx90a)";
        case 2: return "MI300/MI300X (CDNA3/gfx942)";
        case 3: return "MI350/MI355X (CDNA4/gfx950)";
        case 4: return "RX 9060 series (RDNA4/gfx1200)";
        case 5: return "RX 9070 XT/9070 (RDNA4/gfx1201)";
        case 6:
            return "Strix/Strix Halo/Krackan (RDNA3.5/gfx115x APU)";
        case 7: return "gfx1250";
        default: return "Unknown Architecture";
    }
}

}  // namespace mega_kernel_host
