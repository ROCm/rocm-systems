// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "profiler-hub/cpp/reader.hpp"

#include <memory>
#include <string_view>

namespace profiler_hub::common
{

/**
 * @brief Owns one SQLite connection (via reader_t). Holds nothing else --
 *        track/agent/node metadata lookups are the caller's concern, done
 *        against this connection's own reader() when needed.
 */
class connection
{
public:
    explicit connection(std::string_view file_path);
    ~connection() = default;

    connection(const connection&)            = delete;
    connection& operator=(const connection&) = delete;
    connection(connection&&)                 = delete;
    connection& operator=(connection&&)      = delete;

    [[nodiscard]] profiler_hub::reader_t& reader() { return *m_reader; }

private:
    std::unique_ptr<profiler_hub::reader_t> m_reader;
};

}  // namespace profiler_hub::common
