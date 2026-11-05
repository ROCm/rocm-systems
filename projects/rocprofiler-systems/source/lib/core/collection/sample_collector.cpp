// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/collection/sample_collector.hpp"
#include <vector>

#include "core/collection/record.hpp"
#include "core/collection/sink_interface.hpp"
#include "core/collection/source_control.hpp"
#include "core/collection/source_interface.hpp"
#include "logger/debug.hpp"

#include <exception>
#include <memory>
#include <mutex>
#include <utility>

namespace rocprofsys::collection
{
sample_collector::sample_collector(sink_interface& out)
: m_sink(out)
{}

sample_collector::~sample_collector() { stop(); }

void
sample_collector::add(std::unique_ptr<source_interface> src)
{
    const std::lock_guard lock(m_registry_mutex);
    const auto            src_id = static_cast<source_id>(m_controls.size());
    m_controls.push_back(std::make_unique<source_control>(std::move(src), src_id));
}

std::vector<source_control*>
sample_collector::snapshot_controls()
{
    const std::lock_guard        lock(m_registry_mutex);
    std::vector<source_control*> result;
    result.reserve(m_controls.size());
    for(const auto& control : m_controls)
    {
        result.push_back(control.get());
    }
    return result;
}

void
sample_collector::start()
{
    for(auto* control : snapshot_controls())
    {
        start_source(*control);
    }
}

void
sample_collector::start_source(source_control& control)
{
    const std::lock_guard lock(control.mutex);
    if(control.state != source_state::created)
    {
        return;
    }

    if(!control.described)
    {
        try_call(control, [&] { control.src->describe(control.id, m_sink); });
        if(control.state == source_state::failed)
        {
            return;
        }
        control.described = true;
    }
    transition(control, source_state::created, source_state::running,
               &source_interface::start);
}

void
sample_collector::pause_source(source_control& control)
{
    const std::lock_guard lock(control.mutex);
    transition(control, source_state::running, source_state::paused,
               &source_interface::pause);
}

void
sample_collector::resume_source(source_control& control)
{
    const std::lock_guard lock(control.mutex);
    transition(control, source_state::paused, source_state::running,
               &source_interface::resume);
}

void
sample_collector::stop_source(source_control& control)
{
    const std::lock_guard lock(control.mutex);
    switch(control.state)
    {
        case source_state::running:
        case source_state::paused:
            transition(control, control.state, source_state::stopped,
                       &source_interface::stop);
            break;
        case source_state::failed:
            try_call(control, [&] { control.src->stop(); });
            control.state = source_state::stopped;
            break;
        case source_state::created: control.state = source_state::stopped; break;
        case source_state::stopped: break;
    }
}

void
sample_collector::pause()
{
    for(auto* control : snapshot_controls())
    {
        pause_source(*control);
    }
}

void
sample_collector::resume()
{
    for(auto* control : snapshot_controls())
    {
        resume_source(*control);
    }
}

void
sample_collector::stop()
{
    for(auto* control : snapshot_controls())
    {
        stop_source(*control);
    }
}

void
sample_collector::transition(source_control& control, source_state old_state,
                             source_state new_state, void (source_interface::*step)())
{
    if(control.state != old_state)
    {
        return;
    }
    try_call(control, [&] { (control.src.get()->*step)(); });
    if(control.state != source_state::failed)
    {
        control.state = new_state;
    }
}

template <typename Fn>
void
sample_collector::try_call(source_control& control, Fn&& func) noexcept
{
    try
    {
        func();
    } catch(const std::exception& error)
    {
        disable(control, error.what());
    }
}

void
sample_collector::disable(source_control& control, const char* what) noexcept
{
    control.state = source_state::failed;
    LOG_ERROR("collection: source '{}' disabled: {}", control.src->name(), what);
}
}  // namespace rocprofsys::collection
