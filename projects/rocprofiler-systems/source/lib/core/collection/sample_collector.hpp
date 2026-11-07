// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/collection/sink_interface.hpp"
#include "core/collection/source_control.hpp"
#include "core/collection/source_interface.hpp"

#include <memory>
#include <mutex>
#include <vector>

namespace rocprofsys::collection
{
/** Calls invalid in the current state of a source are ignored for it. The collector holds
 * no global state and installs no fork handlers: the owner may leak it in a forked child.
 */
class sample_collector
{
public:
    explicit sample_collector(sink_interface& out);
    ~sample_collector();

    sample_collector(const sample_collector&)            = delete;
    sample_collector& operator=(const sample_collector&) = delete;

    void add(std::unique_ptr<source_interface> src);

    void start();

    void pause();
    void resume();
    void stop();

private:
    std::vector<source_control*> snapshot_controls();

    void start_source(source_control& control);
    void pause_source(source_control& control);
    void resume_source(source_control& control);
    void stop_source(source_control& control);

    void transition(source_control& control, source_state old_state,
                    source_state new_state, void (source_interface::*step)());

    template <typename Fn>
    void try_call(source_control& control, Fn&& func) noexcept;

    void disable(source_control& control, const char* what) noexcept;

    sink_interface& m_sink;

    std::mutex                                   m_registry_mutex;
    std::vector<std::unique_ptr<source_control>> m_controls;
};
}  // namespace rocprofsys::collection
