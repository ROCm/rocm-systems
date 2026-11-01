// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include <string>
#include <vector>

namespace addc
{

struct CperErrorSummaryEntry
{
    std::string fru_id;
    std::string fru_text;
    int afid = 0;
    std::string additional_context;

    bool operator==(const CperErrorSummaryEntry&) const = default;
};

using CperErrorSummary = std::vector<CperErrorSummaryEntry>;

} // namespace addc
