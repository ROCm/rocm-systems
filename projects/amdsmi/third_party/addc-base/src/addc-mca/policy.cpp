// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "addc/mca/policy.hpp"

namespace addc::mca
{

bool is_bad_page_retirement(std::string_view project,
                            bool error_threshold_exceeded,
                            McaRegisters registers) noexcept
{
    if (!error_threshold_exceeded)
    {
        return false;
    }
    if (!project.starts_with("mi300") && !project.starts_with("mi350"))
    {
        return false;
    }

    const bool raw_source_matches =
        registers.ipid.hardware_id == 0x96U && registers.ipid.mca_type == 0x0U;
    if (!raw_source_matches)
    {
        return false;
    }

    const uint8_t error_code_ext = registers.status.error_code_ext;
    return error_code_ext == 0x0U || error_code_ext == 0x0fU;
}

} // namespace addc::mca
