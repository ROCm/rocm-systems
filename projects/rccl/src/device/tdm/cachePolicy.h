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

/// \file cachePolicy.h
/// \brief Shared cache-policy encoding for the TDM copy libraries.
///
/// Both the tensor-data-mover implementation (tdmCopy.h) and the async-to/from-LDS
/// implementation (asyncCopy.h) accept the same compile-time cache policy so their
/// public APIs stay identical.  A CachePolicy packs a temporal hint together with a
/// memory scope into the integer immediate the memory instructions expect.
///
/// A round-trip copy (global -> LDS -> global) touches two different global buffers,
/// which in a collective are typically not equally distant: the source is usually
/// local HBM while the destination is a peer's memory reached over the interconnect.
/// The copy entry points therefore take TWO policies -- localCp for the read leg and
/// remoteCp for the write leg -- so the near side can use a cheaper scope than the
/// far side.  The one-way global<->LDS primitives touch a single global buffer and
/// keep a single policy for it.

#ifndef __TDM_CACHE_POLICY_H
#define __TDM_CACHE_POLICY_H

#include <cstdint>

#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif

using CachePolicy = uint32_t;

enum struct MemScope : uint32_t {
  WGP =
    0, // Workgroup processor scope - warps running on the same WGP should be able to see the effect of the operation
  SE, // Shader engine a.k.a cluster scope
  DEV, // Device scope
  SYS, // System scope
};

// Near cache is L1 data cache, far cache is L2.  The first four values apply the same temporality to both caches.  
// The last four values have different policies for the near cache and far cache.
enum struct TemporalHint : uint32_t {
  RT = 0, // Regular temporal (nothing special)
  NT, // Not temporal
  HT, // High temporal
  LU, // Last use
  NT_RT, // Not temporal near cache, regular temporal far cache
  RT_NT, // Regular temporal near cache, not temporal far cache
  NT_HT, // Not temporal near cache, high temporal far cache
  NT_WB, // Non temporal near cache, writeback far cache, only valid for stores, not loads
};

__host__ __device__ constexpr CachePolicy createCachePolicy(TemporalHint temporal, MemScope scope) noexcept {
  return static_cast<CachePolicy>(scope) << 3 | static_cast<CachePolicy>(temporal);
}

static_assert(createCachePolicy(TemporalHint::RT, MemScope::WGP) == 0);
static_assert(createCachePolicy(TemporalHint::NT, MemScope::WGP) == 1);
static_assert(createCachePolicy(TemporalHint::HT, MemScope::WGP) == 2);
static_assert(createCachePolicy(TemporalHint::LU, MemScope::WGP) == 3);
static_assert(createCachePolicy(TemporalHint::NT_RT, MemScope::WGP) == 4);
static_assert(createCachePolicy(TemporalHint::RT_NT, MemScope::WGP) == 5);
static_assert(createCachePolicy(TemporalHint::NT_HT, MemScope::WGP) == 6);
static_assert(createCachePolicy(TemporalHint::RT, MemScope::SE) == 8);
static_assert(createCachePolicy(TemporalHint::NT, MemScope::SE) == 9);
static_assert(createCachePolicy(TemporalHint::HT, MemScope::SE) == 10);
static_assert(createCachePolicy(TemporalHint::LU, MemScope::SE) == 11);
static_assert(createCachePolicy(TemporalHint::NT_RT, MemScope::SE) == 12);
static_assert(createCachePolicy(TemporalHint::RT_NT, MemScope::SE) == 13);
static_assert(createCachePolicy(TemporalHint::NT_HT, MemScope::SE) == 14);
static_assert(createCachePolicy(TemporalHint::RT, MemScope::DEV) == 16);
static_assert(createCachePolicy(TemporalHint::NT, MemScope::DEV) == 17);
static_assert(createCachePolicy(TemporalHint::HT, MemScope::DEV) == 18);
static_assert(createCachePolicy(TemporalHint::LU, MemScope::DEV) == 19);
static_assert(createCachePolicy(TemporalHint::NT_RT, MemScope::DEV) == 20);
static_assert(createCachePolicy(TemporalHint::RT_NT, MemScope::DEV) == 21);
static_assert(createCachePolicy(TemporalHint::NT_HT, MemScope::DEV) == 22);
static_assert(createCachePolicy(TemporalHint::RT, MemScope::SYS) == 24);
static_assert(createCachePolicy(TemporalHint::NT, MemScope::SYS) == 25);
static_assert(createCachePolicy(TemporalHint::HT, MemScope::SYS) == 26);
static_assert(createCachePolicy(TemporalHint::LU, MemScope::SYS) == 27);
static_assert(createCachePolicy(TemporalHint::NT_RT, MemScope::SYS) == 28);
static_assert(createCachePolicy(TemporalHint::RT_NT, MemScope::SYS) == 29);
static_assert(createCachePolicy(TemporalHint::NT_HT, MemScope::SYS) == 30);

constexpr CachePolicy DEFAULT_CACHE_POLICY = createCachePolicy(TemporalHint::RT, MemScope::SYS);

#endif // __TDM_CACHE_POLICY_H
