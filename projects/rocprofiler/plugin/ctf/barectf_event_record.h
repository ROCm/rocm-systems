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

#ifndef PLUGIN_CTF_BARECTF_EVENT_RECORD_H
#define PLUGIN_CTF_BARECTF_EVENT_RECORD_H

#include <memory>
#include <cstdint>

struct barectf_default_ctx;

namespace rocm_ctf {

// Abstract base class of any barectf event record.
//
// A concrete event record class must implement Write() which must call
// a corresponding barectf tracing function.
//
// `CtxT` is the specific type of the barectf context which Write()
// receives.
template <typename CtxT> class BarectfEventRecord {
 protected:
  // Builds a barectf event record having the clock value `clock_val`.
  explicit BarectfEventRecord(const std::uint64_t clock_val) noexcept : clock_val_{clock_val} {}

 public:
  // Shared pointer to const barectf event record.
  using SP = std::shared_ptr<const BarectfEventRecord>;

  virtual ~BarectfEventRecord() = default;

  // Disabled copy operations to make this class simpler.
  BarectfEventRecord(const BarectfEventRecord&) = delete;
  BarectfEventRecord& operator=(const BarectfEventRecord&) = delete;

  // Clock value of this event record.
  std::uint64_t GetClockVal() const noexcept { return clock_val_; }

  // Calls a corresponding barectf tracing function using the barectf
  // context `barectf_ctx`.
  virtual void Write(CtxT& barectf_ctx) const = 0;

 private:
  // Clock value.
  std::uint64_t clock_val_;
};

}  // namespace rocm_ctf

#endif  // PLUGIN_CTF_BARECTF_EVENT_RECORD_H
