// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <functional>
#include <optional>
#include <string>

namespace profiler_hub
{

struct reader_catalog_t;

namespace common
{
class connection_pool;
class thread_pool;
}  // namespace common

/** @brief Receives the text of each phase of catalog building. The fraction is
 *         set only when the phase has a real one. The callback may run on a
 *         worker thread. */
using catalog_progress_fn =
    std::function<void(const std::string&, std::optional<double>)>;

void
populate_reader_catalog(common::thread_pool&       workers,
                        common::connection_pool&   connections,
                        reader_catalog_t&          catalog,
                        const catalog_progress_fn& progress = {});

}  // namespace profiler_hub
