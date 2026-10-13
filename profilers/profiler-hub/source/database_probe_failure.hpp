// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "profiler-hub/c/profiler_hub_types.h"
#include "profiler-hub/cpp/database_probe.hpp"

#include <stdexcept>
#include <utility>

namespace profiler_hub
{

/** @brief A probe that did not produce a supported trace. */
class database_probe_failure : public std::runtime_error
{
public:
    explicit database_probe_failure(database_probe_result_t result)
    : std::runtime_error(result.message.empty() ? "database probe failed"
                                                : result.message)
    , m_result{ std::move(result) }
    {}

    [[nodiscard]] const database_probe_result_t& result() const { return m_result; }

private:
    database_probe_result_t m_result;
};

[[nodiscard]] inline ph_result_t
to_ph_result(probe_status_t status)
{
    switch(status)
    {
        case probe_status_t::supported: return PH_RESULT_SUCCESS;
        case probe_status_t::unknown_format: return PH_RESULT_UNSUPPORTED_DATABASE;
        case probe_status_t::metadata_missing: return PH_RESULT_METADATA_MISSING;
        case probe_status_t::unsupported_version:
            return PH_RESULT_UNSUPPORTED_SCHEMA_VERSION;
        case probe_status_t::file_not_found:
        case probe_status_t::unreadable:
        case probe_status_t::metadata_malformed:
        case probe_status_t::ambiguous_trace: return PH_RESULT_INVALID_DATABASE;
    }
    return PH_RESULT_INVALID_DATABASE;
}

}  // namespace profiler_hub
