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

/// \file syncPolicy.h
/// \brief Shared completion policy for the global<->LDS transfer primitives.
///
/// asyncLoadToLDS()/asyncStoreFromLDS() have two interchangeable implementations --
/// the async-to/from-LDS builtins (asyncCopy.h) and the tensor data mover
/// (tdmCopy.h).  Both take this policy as their first template argument so the two
/// are drop-in replacements for one another.

#ifndef __TDM_SYNC_POLICY_H
#define __TDM_SYNC_POLICY_H

#include <cstdint>

/// Whether a transfer primitive drains before returning (Sync) or leaves the
/// transfer in flight for the caller to wait on later (Async).
enum struct SyncPolicy : uint32_t {
  Async,
  Sync,
};

constexpr SyncPolicy DEFAULT_SYNC_POLICY = SyncPolicy::Async;

#endif // __TDM_SYNC_POLICY_H
