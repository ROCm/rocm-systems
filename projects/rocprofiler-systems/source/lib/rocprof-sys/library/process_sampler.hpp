// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/collection/scheduler.hpp"
#include "core/collection/source_interface.hpp"

#include <chrono>
#include <functional>
#include <memory>

namespace rocprofsys::process_sampler
{
/** Creates a source for the sampler: its polling interval and the factory of its
 * scheduler. */
using source_factory = std::function<std::unique_ptr<collection::source_interface>(
    std::chrono::nanoseconds interval, collection::scheduler_factory<> make_scheduler)>;

/**
 * Adds a source that cannot exist when `setup()` runs (it needs the SDK, which is
 * configured later). Used at once when the sampler runs, by the next `setup()` otherwise;
 * either way only once.
 */
void
add_source(source_factory factory);

void
setup();

void
shutdown();

void
post_process();

void
pause();

void
resume();

/** Before a fork: waits for the polls that are running and keeps new ones from starting
 * until `unblock()`. */
void
block();

void
unblock();

/** Forked child: drops the collector without destroying it (its thread does not exist
 * here). */
void
postfork_child();
}  // namespace rocprofsys::process_sampler
