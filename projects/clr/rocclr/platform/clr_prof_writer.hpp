/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @file clr_prof_writer.hpp
 * @brief Built-in CLR trace writer.
 *
 * Activated by setting the environment variable CLR_TRACE_OUTPUT before
 * launching the application:
 *
 *   CLR_TRACE_OUTPUT=trace.json      → Chrome Trace Event Format (JSON)
 *   CLR_TRACE_OUTPUT=trace.perfetto  → Perfetto proto binary (future)
 *
 * The writer registers itself as a clr_prof subscriber during hipInit and
 * flushes the file on process exit.  Zero overhead when the env var is unset.
 */

#pragma once

namespace amd::clr_prof {

/**
 * Initialize the built-in trace writer if CLR_TRACE_OUTPUT is set.
 * Called once from hipInit.  Idempotent.
 */
void WriterInit();

/**
 * Flush and close the trace file.  Called from hipTearDown / atexit.
 * Safe to call multiple times (subsequent calls are no-ops).
 */
void WriterFini();

}  // namespace amd::clr_prof
