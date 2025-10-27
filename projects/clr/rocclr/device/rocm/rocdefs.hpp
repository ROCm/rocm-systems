/* Copyright (c) 2008 - 2021 Advanced Micro Devices, Inc.

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
namespace amd::roc {

//! Alignment restriction for the pinned memory
static constexpr size_t PinnedMemoryAlignment = 4 * Ki;

//! Specific defines for images for Dynamic Parallelism
static constexpr uint DeviceQueueMaskSize = 32;

//! Set to match the number of pipes, which is 8.
static constexpr uint kMaxAsyncQueues = 8;

constexpr bool kSkipCpuWait = true;

enum HwQueueEngine : uint32_t {
  Compute = 0,
  SdmaRead = 1,
  SdmaWrite = 2,
  SdmaIntra = 3,
  SdmaInter = 4,
  Unknown = 5
};

}  // namespace amd::roc
