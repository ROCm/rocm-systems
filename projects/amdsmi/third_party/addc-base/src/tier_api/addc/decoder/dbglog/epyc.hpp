// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "addc/decoder/dbglog/decoder.hpp"

namespace addc::decoder
{

[[nodiscard]] std::vector<DecodedDbgLog> decode_epyc_dbglog(
    const std::vector<addc::pipeline::DbgLogGroup>& groups);

} // namespace addc::decoder
