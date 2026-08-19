/*
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

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
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#ifndef RCCL_DECISION_H_
#define RCCL_DECISION_H_

#include <cstdint>

// Single, self-contained description of which implementation RCCL selected for a
// collective. This is the one source of truth that both the dispatch path
// (ncclXxx_impl / taskAppend) and the reporting path (rcclGetCollImplInfo, used
// by rccl-tests) consume, so perf numbers are always attributed to the backend
// that actually ran.
//
// `algo` uses the unified identifier space: a native NCCL_ALGO_* value for the
// standard ring/tree/pat kernel path, or an rcclAddonAlgos_t value (Direct,
// Symmetric, CE, DDA, ...) for an RCCL-specific backend. Name it via
// rcclGetAlgoName(). Kept as int (not the enum) so this header stays free of the
// enum's dependencies and can be included by low-level headers like info.h.
struct rcclCollDecision {
  int algo;             // NCCL_ALGO_* or rcclAddonAlgos_t
  int protocol;         // NCCL_PROTO_*
  uint32_t nMaxChannels; // reporting: channels for the kernel path (0 = N/A)
  // Runtime bits computed once at the decision point and carried into
  // taskAppend() so it never recomputes graph-capture state for AllReduce.
  bool ceCapturing;
  bool ceArGraphAllowed;
};

#endif // RCCL_DECISION_H_
