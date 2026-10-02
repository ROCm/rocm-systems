// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstdint>

namespace pm4_builder
{
struct Gfx11TraceTarget
{
    int      wgp = -1;
    uint32_t sa  = 0;
};

// The GFX11 DRM map folds SEs 4-7 into columns 2-3 of rows 0-3.
// Two CU bits represent each physical WGP. A zero entry can mean an
// entirely harvested shader array, so only trust a complete topology.
class Gfx11TraceTopology
{
public:
    Gfx11TraceTopology() = default;

    Gfx11TraceTopology(const uint32_t (&bitmap)[4][4],
                       uint32_t se_count,
                       uint32_t sa_count,
                       uint32_t active_cus)
    {
        if(se_count == 0 || se_count > 8 || sa_count == 0 || sa_count > 2) return;
        uint32_t counted = 0;
        for(uint32_t se = 0; se < se_count; ++se)
            for(uint32_t sa = 0; sa < sa_count; ++sa)
            {
                uint32_t bits  = bitmap[se % 4][sa + (se / 4) * 2];
                masks_[se][sa] = bits;
                for(; bits != 0; bits &= bits - 1)
                    ++counted;
            }
        available_ = active_cus != 0 && counted == active_cus;
    }

    bool available() const { return available_; }

    Gfx11TraceTarget select(uint32_t se, uint32_t requested_wgp) const
    {
        if(!available_ || se >= masks_.size()) return {};
        if(requested_wgp < 16 && (masks_[se][0] & (3u << (2 * requested_wgp))) != 0)
            return {static_cast<int>(requested_wgp), 0};
        for(uint32_t sa = 0; sa < 2; ++sa)
            for(uint32_t wgp = 0; wgp < 16; ++wgp)
                if((masks_[se][sa] & (3u << (2 * wgp))) != 0) return {static_cast<int>(wgp), sa};
        return {};
    }

private:
    bool                                   available_ = false;
    std::array<std::array<uint32_t, 2>, 8> masks_{};
};
}  // namespace pm4_builder
