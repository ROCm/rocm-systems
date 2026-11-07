// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/collection/record.hpp"
#include "core/collection/sink_interface.hpp"

#include <string_view>

namespace rocprofsys::collection
{
/** Lifecycle calls on one source are serialized by the collector; one that throws
 * disables the source. Threads owned by a source must catch everything themselves. */
class source_interface
{
public:
    virtual ~source_interface() = default;

    [[nodiscard]] virtual std::string_view name() const noexcept = 0;

    /** Once, before the first start(): enumerate the devices and register the schemas. */
    virtual void describe(source_id src_id, sink_interface& out) = 0;

    virtual void start() = 0;

    /** Stop capturing until resume(). Must not wait for a poll that is running. */
    virtual void pause() = 0;

    /** Resume capturing and write the gap marker of the pause. */
    virtual void resume() = 0;

    /** Synchronous: no sink call happens after it returns. */
    virtual void stop() = 0;
};
}  // namespace rocprofsys::collection
