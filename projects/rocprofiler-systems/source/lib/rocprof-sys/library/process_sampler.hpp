// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>

namespace rocprofsys::collection
{
class source_interface;
}  // namespace rocprofsys::collection

namespace rocprofsys::process_sampler
{
/**
 * Adds a source that cannot exist when setup() runs (SDK perf counters). Used at once
 * when the sampler runs (started paused if paused), by the next setup() otherwise; only
 * once.
 */
void
add_source(std::unique_ptr<collection::source_interface> source);

/** (Re)starts the sampler with the default sources and the queued ones. */
void
setup();

/** Stops every poll thread. Called on a poll thread (a source that exits the process),
 * it marks the sampler finalized and waits for the other sources' polls in flight,
 * without joining any thread. */
void
shutdown();

/** Destroys the sampler and its providers. A no-op on a poll thread. */
void
post_process();

/** Any thread; never waits for setup() or a poll. */
void
pause();

/** Any thread; never waits for setup() or a poll. */
void
resume();

/** Before fork(), any thread including a poll thread: waits for the running polls and
 * keeps new ones from starting until unblock(). */
void
block();

/** After fork() in the parent. No AMD SMI call. */
void
unblock();

/** Forked child: abandons the whole state (its threads do not exist here) and starts a
 * fresh, idle one. No AMD SMI or provider call. */
void
postfork_child();
}  // namespace rocprofsys::process_sampler
