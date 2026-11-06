// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

/// Call-stack sampling with cpu-snap (ROCPROFSYS_USE_EXPERIMENTAL_CALLSTACK_SAMPLER). The
/// process has one sampler; its encoded stream goes into the trace cache and is decoded
/// in post-processing.
namespace rocprofsys::cpu_snap
{
/**
 * @brief Creates the sampler from the settings and starts it.
 *
 * Call after trace-cache storage has started: the sampler stores its data there.
 * @throws cpu_snap::error When cpu-snap cannot sample, e.g. the application already
 *         handles cpu-snap's real-time signal.
 */
void
setup();

/// Pauses sampling; no-op without a running sampler. Callable from any thread.
void
pause() noexcept;

/// Resumes sampling; no-op without a running sampler. Callable from any thread.
void
resume() noexcept;

/// Stops sampling and flushes the remaining data into the trace cache. Call before
/// the trace cache shuts down.
void
stop();

/// Logs the sampler's statistics and releases it. Call after stop().
void
post_process();
}  // namespace rocprofsys::cpu_snap
