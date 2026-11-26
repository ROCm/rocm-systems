// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "track_row_reader.hpp"

#include <memory>
#include <type_traits>
#include <utility>

namespace profiler_hub
{

/** Hands out readers backed by trace connections; the production source wraps the
 * connection pool. */
class connection_source
{
public:
    connection_source()          = default;
    virtual ~connection_source() = default;

    connection_source(const connection_source&)            = delete;
    connection_source& operator=(const connection_source&) = delete;
    connection_source(connection_source&&)                 = delete;
    connection_source& operator=(connection_source&&)      = delete;

    /** Blocks until a reader is available. */
    [[nodiscard]] virtual std::unique_ptr<track_row_reader> acquire() = 0;

    /** Returns nullptr instead of blocking when no reader is available. */
    [[nodiscard]] virtual std::unique_ptr<track_row_reader> try_acquire() = 0;
};

/** Runs @p fn on the calling thread with an acquired reader and releases it afterwards.
 */
template <typename Fn>
[[nodiscard]] auto
with_reader(connection_source& source,
            Fn&&               fn) -> std::invoke_result_t<Fn, track_row_reader&>
{
    const auto reader = source.acquire();
    return std::forward<Fn>(fn)(*reader);
}

}  // namespace profiler_hub
