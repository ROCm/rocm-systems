// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "section_helpers.hpp"

#include <string>

namespace addc::pipeline::detail
{

inline void parse_mtl_body(AmdEpycMtlSection& sec, const nlohmann::json& body)
{
    if (const auto v = json_str(body, "loggingEnabled"))
    {
        try
        {
            sec.logging_enabled =
                static_cast<uint32_t>(std::stoul(*v, nullptr, 16));
        }
        catch (...)
        {}
    }
    if (const auto v = json_str(body, "tailOffset"))
    {
        try
        {
            sec.tail_offset =
                static_cast<uint16_t>(std::stoul(*v, nullptr, 16));
        }
        catch (...)
        {}
    }
    if (const auto v = json_u64(body, "entries"))
        sec.entries = static_cast<uint16_t>(*v);
    if (const auto v = json_str(body, "logVersion"))
    {
        try
        {
            sec.log_version =
                static_cast<uint32_t>(std::stoul(*v, nullptr, 16));
        }
        catch (...)
        {}
    }
    if (const auto v = json_str(body, "usecTimestamp"))
    {
        try
        {
            sec.usec_timestamp =
                static_cast<uint32_t>(std::stoul(*v, nullptr, 16));
        }
        catch (...)
        {}
    }
    if (auto bytes = json_bytes(body, "mtlRawData"))
        sec.raw_data = std::move(*bytes);
}

} // namespace addc::pipeline::detail
