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
// Symmetric, CE, DDA, GIN-SDMA, ...) for an RCCL-specific backend. Name it via
// rcclGetAlgoName(). Kept as int (not the enum) so this header stays free of the
// enum's dependencies and can be included by low-level headers like info.h.
struct rcclCollDecision {
  int algo;             // NCCL_ALGO_* or rcclAddonAlgos_t
  int protocol;         // NCCL_PROTO_*
  uint32_t nMaxChannels; // reporting: channels for the kernel path (0 = N/A)
  // Runtime bits computed once at the decision point and carried into
  // taskAppend() so it never recomputes graph-capture state.
  bool ceCapturing;
  bool ceArGraphAllowed;
};


// Defined here (not in rccl_common.h) so this header stays free of
// rccl_common.h's heavy device-header chain
// (sym_kernels.h -> gin_scratch.h -> gin_tmp.h).
// rccl_common.h includes rccl_decision.h, so all its users get these types.
typedef uint32_t rcclBackendMask_t;
enum : rcclBackendMask_t {
  RCCL_BACKEND_GIN_SDMA  = 1u << 0,
  RCCL_BACKEND_SYMMETRIC = 1u << 1,
  RCCL_BACKEND_CE        = 1u << 2,
  RCCL_BACKEND_DDA       = 1u << 3,
  RCCL_BACKEND_KERNEL    = 1u << 4,
  RCCL_BACKEND_ALL       = 0x1fu,
};

// Returns the preferred backend mask for this communicator given the active
// CTAPolicy and any force-CE env vars.  RCCL_BACKEND_ALL means "no preference"
// (zero overhead on the default path).  Defined in rccl_wrap.cc.
struct ncclComm;
rcclBackendMask_t rcclPreferredBackends(const struct ncclComm* comm);

// Working state for the single-pass preferred/fallback backend search used by
// rcclSelectXxx().  Each selector initialises its own instance; the preferred
// mask may differ per collective based on what backends are available for it.
struct rcclCandSearch {
  rcclBackendMask_t  preferred;
  rcclCollDecision*  decision;
  rcclCollDecision   bestFallback;
  bool               bestFallbackFound;
  bool               bestPreferredFound;
};

// Record one eligible candidate into the search state.  Called in priority
// order; stops updating once both slots are filled.
// inWindow: true when the candidate is within its tuning size window.
//   - bestFallback is only set when inWindow=true (fallback must be a good default).
//   - bestPreferred is set when the candidate matches the preferred mask AND either
//     (a) preferred is explicit (not ALL, so a deliberate policy choice), or
//     (b) the candidate is in its tuning window.
//   Existing callers omit inWindow and get true by default (no behaviour change).
static inline void rcclCandSearchRecord(rcclCandSearch& s,
                                        rcclBackendMask_t family,
                                        const rcclCollDecision& cand,
                                        bool inWindow = true) {
  if (inWindow && !s.bestFallbackFound) {
    s.bestFallback      = cand;
    s.bestFallbackFound = true;
  }
  const bool qualifiesAsPreferred = (s.preferred & family) &&
                                    (s.preferred != RCCL_BACKEND_ALL || inWindow);
  if (!s.bestPreferredFound && qualifiesAsPreferred) {
    *s.decision          = cand;
    s.bestPreferredFound = true;
  }
}

#endif // RCCL_DECISION_H_
