// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/collection/fork_gate.hpp"

#include <mutex>

namespace rocprofsys::collection
{
namespace
{
// Keyed to the gate: a pass on another gate, or the gate a forked child abandoned, does
// not count for this one.
/** The gate the calling thread holds a pass on. */
thread_local const fork_gate* g_pass_gate = nullptr;
/** The gate whose count block() on this thread removed its own pass from. */
thread_local const fork_gate* g_discounted_gate = nullptr;
}  // namespace

fork_gate::pass::pass(fork_gate& gate)
: m_gate(gate)
, m_entered(gate.try_enter())
{}

fork_gate::pass::~pass()
{
    if(m_entered)
    {
        m_gate.leave();
    }
}

void
fork_gate::block()
{
    std::unique_lock lock(m_mutex);
    if(g_pass_gate == this && g_discounted_gate != this)
    {
        --m_polls;
        g_discounted_gate = this;
    }
    ++m_forks;
    m_idle.wait(lock, [this] { return m_polls == 0; });
}

void
fork_gate::unblock()
{
    const std::lock_guard lock(m_mutex);
    --m_forks;
    if(g_discounted_gate == this)
    {
        ++m_polls;
        g_discounted_gate = nullptr;
    }
}

bool
fork_gate::try_enter()
{
    const std::lock_guard lock(m_mutex);
    if(m_forks > 0)
    {
        return false;
    }
    ++m_polls;
    g_pass_gate = this;
    return true;
}

void
fork_gate::leave()
{
    bool fork_waiting = false;
    {
        const std::lock_guard lock(m_mutex);
        --m_polls;
        g_pass_gate  = nullptr;
        fork_waiting = m_forks > 0;
    }
    if(fork_waiting)
    {
        m_idle.notify_all();
    }
}
}  // namespace rocprofsys::collection
