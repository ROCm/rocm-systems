// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "addc/decoder/dbglog/epyc.hpp"

namespace addc::decoder
{

std::vector<DecodedDbgLog> decode_epyc_dbglog(
    const std::vector<addc::pipeline::DbgLogGroup>& groups)
{
    auto decoded = decode_generic_dbglog(groups);
    for (std::size_t index = 0U; index < decoded.size(); ++index)
    {
        decoded[index].afid =
            addc::afid_list(25508 + static_cast<int>(groups[index].block_id));
    }
    return decoded;
}

} // namespace addc::decoder
