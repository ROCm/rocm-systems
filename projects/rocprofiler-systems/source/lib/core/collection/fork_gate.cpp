// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/collection/fork_gate.hpp"

#include <mutex>

namespace rocprofsys::collection
{
namespace
{
/** The innermost gate the calling thread holds passes on, and how many. Keyed to the
 * gate: a pass on another gate, or the gate a forked child abandoned, does not count
 * for this one. */
struct held_passes
{
    const fork_gate* gate  = nullptr;
    int              depth = 0;
};
thread_local held_passes g_held;
/** The gate whose count block() on this thread removed its own passes from. */
thread_local const fork_gate* g_discounted_gate  = nullptr;
thread_local int              g_discounted_depth = 0;

void
hold_one_more(const fork_gate* gate)
{
    g_held = { .gate = gate, .depth = g_held.gate == gate ? g_held.depth + 1 : 1 };
}
}  // namespace

fork_gate::pass::pass(fork_gate& gate)
: m_gate(gate)
, m_outer_gate(g_held.gate)
, m_outer_depth(g_held.depth)
, m_entered(gate.try_enter())
{}

fork_gate::pass::pass(fork_gate& gate, wait_t)
: m_gate(gate)
, m_outer_gate(g_held.gate)
, m_outer_depth(g_held.depth)
{
    gate.enter();
    m_entered = true;
}

fork_gate::pass::~pass()
{
    if(m_entered)
    {
        m_gate.leave(m_outer_gate, m_outer_depth);
    }
}

void
fork_gate::block()
{
    std::unique_lock lock(m_mutex);
    if(g_held.gate == this && g_discounted_gate != this)
    {
        m_polls -= g_held.depth;
        g_discounted_gate  = this;
        g_discounted_depth = g_held.depth;
    }
    ++m_forks;
    m_idle.wait(lock, [this] { return m_polls == 0; });
}

void
fork_gate::unblock()
{
    bool forks_done = false;
    {
        const std::lock_guard lock(m_mutex);
        --m_forks;
        if(g_discounted_gate == this)
        {
            m_polls += g_discounted_depth;
            g_discounted_gate  = nullptr;
            g_discounted_depth = 0;
        }
        forks_done = m_forks == 0;
    }
    if(forks_done)
    {
        m_idle.notify_all();
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
    hold_one_more(this);
    return true;
}

void
fork_gate::enter()
{
    std::unique_lock lock(m_mutex);
    m_idle.wait(lock, [this] { return m_forks == 0 || g_held.gate == this; });
    ++m_polls;
    hold_one_more(this);
}

void
fork_gate::leave(const fork_gate* outer_gate, int outer_depth)
{
    bool fork_waiting = false;
    {
        const std::lock_guard lock(m_mutex);
        --m_polls;
        g_held       = { .gate = outer_gate, .depth = outer_depth };
        fork_waiting = m_forks > 0;
    }
    if(fork_waiting)
    {
        m_idle.notify_all();
    }
}
}  // namespace rocprofsys::collection
