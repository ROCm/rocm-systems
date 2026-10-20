// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "profiler-hub/cpp/reader_types.hpp"

#include <stdexcept>
#include <utility>

namespace profiler_hub
{

/** @brief A probe that did not produce a supported trace. */
class file_probe_failure : public std::runtime_error
{
public:
    explicit file_probe_failure(reader_types::file_probe_result_t result)
    : std::runtime_error(result.message.empty() ? "file probe failed" : result.message)
    , m_result{ std::move(result) }
    {}

    [[nodiscard]] const reader_types::file_probe_result_t& result() const
    {
        return m_result;
    }

private:
    reader_types::file_probe_result_t m_result;
};

}  // namespace profiler_hub
