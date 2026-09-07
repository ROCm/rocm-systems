////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2014-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Developed by:
//
//                 AMD Research and AMD HSA Software Development
//
//                 Advanced Micro Devices, Inc.
//
//                 www.amd.com
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal with the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
//  - Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimers.
//  - Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimers in
//    the documentation and/or other materials provided with the distribution.
//  - Neither the names of Advanced Micro Devices, Inc,
//    nor the names of its contributors may be used to endorse or promote
//    products derived from this Software without specific prior written
//    permission.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
// OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
// ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS WITH THE SOFTWARE.
//
////////////////////////////////////////////////////////////////////////////////

#ifndef HSA_RUNTIME_CORE_INC_WAIT_EVENT_AGE_H_
#define HSA_RUNTIME_CORE_INC_WAIT_EVENT_AGE_H_

#include <cstddef>
#include <cstdint>

namespace rocr {
namespace core {

// Size in bytes of the per-signal "event_age" scratch buffer used by
// Signal::WaitMultiple / Signal::WaitAnyExceptions. Exactly one uint64_t is
// stored per unique signal (see the memset and the `event_age[i] = 1` loop), so
// the buffer must reserve sizeof(uint64_t) per unique signal. Centralising the
// size here keeps the allocation and the memset from disagreeing.
inline constexpr std::size_t WaitEventAgeBytes(uint32_t unique_evts) {
  return static_cast<std::size_t>(unique_evts) * sizeof(uint64_t);
}

}  // namespace core
}  // namespace rocr

#endif  // HSA_RUNTIME_CORE_INC_WAIT_EVENT_AGE_H_
