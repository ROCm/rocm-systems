/* Copyright (c) 2022 Advanced Micro Devices, Inc.

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

#pragma once

#include "roctracer.h"

#include <optional>

namespace roctracer {

// Start a new correlation ID region and push it onto the thread local stack. Correlation ID
// regions are nested and per-thread.
activity_correlation_id_t CorrelationIdPush();

// Stop the current correlation ID region and pop it from the thread local stack.
void CorrelationIdPop();

// Return the ID currently active correlation ID region, or 0 if no regin is active.
activity_correlation_id_t CorrelationId();

// Start a new external correlation ID region for the given \p external_id. As for the internal
// correlation ID regions, external correlation ID regions are nested and per-thread.
void ExternalCorrelationIdPush(activity_correlation_id_t external_id);

// Stop the current external correlation ID region and return the external_id used to start the
// region. Return a nullopt if no region was active.
std::optional<activity_correlation_id_t> ExternalCorrelationIdPop();

// Return the current external correlation ID or nullopt is no region is active.
std::optional<activity_correlation_id_t> ExternalCorrelationId();

}  // namespace roctracer