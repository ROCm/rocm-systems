// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "common/connection_pool.hpp"
#include "connection_source.hpp"

namespace profiler_hub
{

/** Serves readers from the connections of a connection_pool. The pool must outlive the
 * source and its readers. */
class pooled_connection_source final : public connection_source
{
public:
    explicit pooled_connection_source(common::connection_pool& pool) noexcept;

    [[nodiscard]] std::unique_ptr<track_row_reader> acquire() override;
    [[nodiscard]] std::unique_ptr<track_row_reader> try_acquire() override;

private:
    common::connection_pool& m_pool;
};

}  // namespace profiler_hub
