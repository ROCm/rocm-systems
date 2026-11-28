// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "addc/mca/decoder.hpp"
#include "addc/mca/registers.hpp"

#include <cstdint>

namespace addc::mca::detail
{

inline uint32_t gpu_decode_afid(McaRegisters regs) noexcept
{
    const uint16_t hw_id = regs.ipid.hardware_id;
    const uint16_t mca_ty = regs.ipid.mca_type;
    const uint8_t ext = regs.status.error_code_ext;
    const auto& s = regs.status;

    uint8_t sev; // 0 = corrected, 1 = fatal, 2 = non-fatal
    if (s.poison)
        sev = 2;
    else if (s.pcc)
        sev = 1;
    else if (s.uc && s.tcc)
        sev = 1;
    else if (s.uc && !s.tcc)
        sev = 2;
    else if (!s.uc && !s.tcc && s.deferred)
        sev = 2;
    else if (!s.uc && !s.tcc && !s.deferred)
        sev = 0;
    else
        return kAfidSentinel;

    if (hw_id == 0x50 && mca_ty == 0x0)
    {
        return sev == 0 ? 17u : 18u;
    }

    if ((hw_id == 0x259 && mca_ty == 0x0) || (hw_id == 0x267 && mca_ty == 0x0))
    {
        return sev == 0 ? 15u : 16u;
    }

    if (hw_id == 0xFF && mca_ty == 0x1 && ext == 0x3F)
    {
        return sev == 0 ? 15u : 16u;
    }

    if (hw_id == 0x96 && mca_ty == 0x0)
    {
        bool is_hbm_ext =
            (ext == 0x0 || ext == 0x1 || ext == 0x4 || ext == 0x5 ||
             ext == 0x9 || ext == 0xB || ext == 0xF);
        if (is_hbm_ext)
        {
            if (sev == 0)
                return 24u;
            if (ext == 0xB || ext == 0xF)
                return sev == 1 ? 21u : 23u;
            if (ext == 0x0)
                return sev == 1 ? 20u : 22u;
            return 25u;
        }
    }

    if (sev == 1)
    {
        bool is_wdt = (hw_id == 0xB0 && mca_ty == 0x5 && ext == 0x0) ||
                      (hw_id == 0x2E && mca_ty == 0x1 && ext == 0x5);
        if (is_wdt)
            return 27u;
    }

    if (sev == 1)
    {
        bool is_hwa =
            (hw_id == 0x2E && mca_ty == 0x2 && ext == 0x11) ||
            (hw_id == 0xB0 && mca_ty == 0x3 && ext == 0x9) ||
            (hw_id == 0xB0 && mca_ty == 0x5 && ext == 0xB) ||
            (hw_id == 0xB0 && mca_ty == 0x6 && ext == 0x6) ||
            (hw_id == 0xB0 && mca_ty == 0x1 && ext == 0xE) ||
            (hw_id == 0xB0 && mca_ty == 0x2 && ext == 0x3) ||
            (hw_id == 0xB0 && mca_ty == 0x7 && ext == 0x7) ||
            (hw_id == 0xB0 && mca_ty == 0x10 && ext == 0x16) ||
            (hw_id == 0x2E && mca_ty == 0x1 && ext == 0x0);
        if (is_hwa)
            return 26u;
    }

    if (sev == 0)
        return 29u;
    if (sev == 2)
        return 28u;
    return 30u;
}

} // namespace addc::mca::detail
