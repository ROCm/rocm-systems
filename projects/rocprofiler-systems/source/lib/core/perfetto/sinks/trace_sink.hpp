// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <vector>

namespace rocprofsys::core
{
// Passed to basic_cached_perfetto_engine::start() as a shared_ptr; the engine
// keeps only a weak_ptr, so it never extends the sink's lifetime. The caller
// must keep the sink alive until stop() returns. on_source_drained() and
// finalize() are invoked synchronously, from within stop(), on whichever
// thread calls it.
class trace_sink
{
public:
    trace_sink()          = default;
    virtual ~trace_sink() = default;

    trace_sink(const trace_sink&)            = delete;
    trace_sink& operator=(const trace_sink&) = delete;
    trace_sink(trace_sink&&)                 = delete;
    trace_sink& operator=(trace_sink&&)      = delete;

    virtual void on_source_drained(int source_id, std::vector<char> bytes) = 0;
    virtual void finalize()                                                = 0;
};
}  // namespace rocprofsys::core
