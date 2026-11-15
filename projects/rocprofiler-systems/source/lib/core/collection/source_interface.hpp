// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/collection/sink_interface.hpp"

#include <cstdint>
#include <string_view>

namespace rocprofsys::collection
{
/** A sampled data source (CPU, GPU, NIC, hipFile, GPU perf counters): it describes its
 * devices once and is polled on a thread the collector owns. It has no thread, clock or
 * lifecycle of its own.
 *
 * Threads: describe() runs on the thread that calls sample_collector::start(), poll() on
 * the source's own poll thread, release() on the thread that calls stop() after the poll
 * thread has ended. The three are never called concurrently. */
class source_interface
{
public:
    virtual ~source_interface() = default;

    /** At most 10 characters: the poll thread is named "omni.<name>". */
    [[nodiscard]] virtual std::string_view name() const noexcept = 0;

    /** Once, before the first poll: enumerates devices and registers one schema per
     * device that has columns.
     * @return false when no device has a column; the source then holds nothing and is
     * never polled */
    [[nodiscard]] virtual bool describe(sink_interface& out) = 0;

    /** Reads every device that has not failed and writes one row per device. A throw
     * ends polling of this source only. */
    virtual void poll(std::uint64_t timeline_ns, sink_interface& out) = 0;

    /** After the poll thread has stopped: drops the devices and the provider. */
    virtual void release() {}
};
}  // namespace rocprofsys::collection
