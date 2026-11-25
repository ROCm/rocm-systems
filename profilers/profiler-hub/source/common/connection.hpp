// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "profiler-hub/cpp/reader.hpp"

#include <memory>
#include <string_view>

namespace profiler_hub
{
struct reader_catalog_t;
}  // namespace profiler_hub

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
    // Pooled/shared-catalog construction: `catalog` (non-null) is shared
    // with sibling connections in the same pool; see connection_pool.
    // `format` is the probed schema and version and selects the reader.
    connection(std::string_view                                  file_path,
               std::shared_ptr<profiler_hub::reader_catalog_t>   catalog,
               std::string_view                                  uuid,
               const profiler_hub::reader_types::trace_format_t& format);

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
