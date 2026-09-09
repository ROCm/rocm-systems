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
class trace_sink_interface
{
public:
    trace_sink_interface()          = default;
    virtual ~trace_sink_interface() = default;

    trace_sink_interface(const trace_sink_interface&)            = delete;
    trace_sink_interface& operator=(const trace_sink_interface&) = delete;
    trace_sink_interface(trace_sink_interface&&)                 = delete;
    trace_sink_interface& operator=(trace_sink_interface&&)      = delete;

    virtual void on_source_drained(int source_id, std::vector<char> bytes) = 0;
    virtual void finalize()                                                = 0;
};
}  // namespace rocprofsys::core
