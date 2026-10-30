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

/// \file uniform.h
/// \brief Wave-uniformity hints for device code.
///
/// Device functions often take arguments that are wave-uniform by contract, but the
/// compiler cannot prove it when the caller derives them from threadIdx (e.g. a
/// per-warp offset). Without that proof every loop bound, switch and wait count
/// downstream becomes a per-lane value with EXEC-masked control flow, and each SGPR
/// operand (such as a TDM descriptor field) is copied over with its own readfirstlane.
/// Applying uniform() at entry restores the proof and keeps that work scalar.
///
/// uniform() returns lane 0's value, so it is only correct for values that really are
/// the same across the active lanes of the wave.

#ifndef RCCL_DEVICE_UNIFORM_H_
#define RCCL_DEVICE_UNIFORM_H_

#include <cstdint>

#ifndef __device__
#define __device__
#endif

namespace wave {

__device__ inline uint32_t uniform(uint32_t v) {
  return __builtin_amdgcn_readfirstlane(v);
}
__device__ inline uint64_t uniform(uint64_t v) {
  const uint32_t lo = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(v));
  const uint32_t hi = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(v >> 32));
  return (static_cast<uint64_t>(hi) << 32) | lo;
}
template <typename T>
__device__ inline T* uniform(T* p) {
  return reinterpret_cast<T*>(uniform(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(p))));
}

} // namespace wave

#endif // RCCL_DEVICE_UNIFORM_H_
