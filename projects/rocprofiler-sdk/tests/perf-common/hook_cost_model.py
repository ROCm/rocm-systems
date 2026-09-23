# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""Executable specification of the queue-hook cost and semantics models.

These are deliberately small, dependency-free mirrors of decisions that live in C++ and
that no GPU-free test can otherwise reach:

  * the per-dispatch context-gate cost   -- hsa/queue.cpp (WriteInterceptor)
  * context_array_t inline capacity      -- context/context.hpp:213
  * the replay snapshot/restore window   -- kernel_replay/memory_snapshot.cpp
  * serialization refcount semantics     -- hsa/queue_controller.{hpp,cpp}
  * per-pass localized context control   -- kernel_replay/local_context.hpp

The point is not to re-implement the SDK. It is to pin the *documented* semantics and
cost ceilings so that changing them is a deliberate, reviewed act rather than a silent
side effect. Where a number came from a measurement rather than from the source, the
docstring says so.
"""

from typing import Iterable, Optional, Set

# ---------------------------------------------------------------------------
# Context gate
# ---------------------------------------------------------------------------

# context_array_t is common::container::small_vector<const context*>, whose default
# inline element count is derived from a 64-byte preferred object size and a 16-byte
# header: (64 - 16) / 8 == 6. Verified by compiling the real header.
CONTEXT_ARRAY_INLINE_CAPACITY = 6

# Measured at -O2 on x86-64 for the get_active_contexts() pattern (atomic early-out,
# per-slot acquire load, filter predicate, small_vector accumulation). Ceilings are set
# well above the measurement so that ordinary machine-to-machine variation does not
# redden CI; they exist to catch order-of-magnitude regressions, not jitter.
GATE_IDLE_CEILING_NS = 25.0
GATE_PER_CONTEXT_CEILING_NS = 20.0


def gate_allocates(active_contexts: int) -> bool:
    """True when one get_active_contexts() walk must reach the heap.

    Below the inline capacity the small_vector stays on the stack, which is what keeps
    the interceptor gate allocation-free for realistic tool configurations.
    """
    return active_contexts > CONTEXT_ARRAY_INLINE_CAPACITY


def gate_ceiling_ns(active_contexts: int, walks: int = 1) -> float:
    """Upper bound on the per-dispatch-batch cost of `walks` context-gate walks."""
    if walks < 1:
        raise ValueError("walks must be >= 1")
    if active_contexts < 0:
        raise ValueError("active_contexts must be >= 0")
    per_walk = GATE_IDLE_CEILING_NS + active_contexts * GATE_PER_CONTEXT_CEILING_NS
    return walks * per_walk


# ---------------------------------------------------------------------------
# Kernel replay window
# ---------------------------------------------------------------------------

# Effective host<->device copy bandwidth, GB/s. Conservative for PCIe Gen4 x16.
DEFAULT_COPY_BW_GBPS = 45.0


def replay_window_seconds(
    resident_gb: float,
    passes: int,
    bw_gbps: float = DEFAULT_COPY_BW_GBPS,
    kernel_seconds: float = 0.0,
) -> float:
    """Wall-clock cost of replaying ONE dispatch.

    snap() copies every tracked device allocation to the host once; restore() copies it
    back between passes. The final pass deliberately skips restore so the application
    observes the memory state it expects, hence (passes - 1).
    """
    if passes < 1:
        raise ValueError("passes must be >= 1")
    if bw_gbps <= 0.0:
        raise ValueError("bw_gbps must be > 0")
    snap = resident_gb / bw_gbps
    restores = (passes - 1) * resident_gb / bw_gbps
    return snap + restores + passes * kernel_seconds


def replay_host_ram_gb(resident_gb: float) -> float:
    """Host memory the snapshot needs. snap() holds a full host copy, not a diff."""
    return resident_gb


# ---------------------------------------------------------------------------
# Serialization refcount  (hsa/queue_controller.hpp:155)
# ---------------------------------------------------------------------------


def serialization_enabled(all_count: int, per_agent_count: int) -> bool:
    """enabled(agent) == (all + per_agent[agent]) > 0."""
    return (all_count + per_agent_count) > 0


def serialization_concurrency_fraction(total_agents: int, scoped_agents: int) -> float:
    """Fraction of a multi-GPU machine left running concurrently.

    A context with an empty agent set claims every agent, which is the pre-per-agent
    behaviour and yields 0.0.
    """
    if total_agents < 1:
        raise ValueError("total_agents must be >= 1")
    if not 0 <= scoped_agents <= total_agents:
        raise ValueError("scoped_agents must be within [0, total_agents]")
    if scoped_agents == 0:  # empty set == "all agents"
        return 0.0
    return (total_agents - scoped_agents) / total_agents


def collects_on(agents: Optional[Iterable], agent) -> bool:
    """context.hpp: `agents.empty() || agents.count(agent_id) > 0`."""
    agent_set: Set = set(agents or ())
    return not agent_set or agent in agent_set


def contexts_conflict(lhs_agents: Optional[Iterable], rhs_agents: Optional[Iterable]) -> bool:
    """Two same-service contexts conflict when their agent sets intersect.

    An empty set claims every agent, so it conflicts with anything.
    """
    l: Set = set(lhs_agents or ())
    r: Set = set(rhs_agents or ())
    if not l or not r:
        return True
    return bool(l & r)


# ---------------------------------------------------------------------------
# Per-pass localized context control  (kernel_replay/local_context.hpp)
# ---------------------------------------------------------------------------

# PC sampling is agent-wide and deliberately does not consult local_context_override().
OVERRIDE_AWARE_SERVICES = ("counters", "spm", "thread_trace")
OVERRIDE_BLIND_SERVICES = ("pc_sampling",)


def effective_collection(service: str, globally_enabled: bool, override: Optional[bool]) -> bool:
    """Whether `service` collects on a dispatch in the current replay pass.

    The override is AND-ed with the global state: a local start can only undo a local
    stop, never promote a globally stopped context. `None` means no override recorded.
    """
    if service in OVERRIDE_BLIND_SERVICES:
        return globally_enabled
    if override is None:
        return globally_enabled
    return globally_enabled and override
