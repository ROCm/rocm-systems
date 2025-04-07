////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2014-2020, Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef AMD_HSA_LOCKS_HPP
#define AMD_HSA_LOCKS_HPP

#include <condition_variable>
#include <cstddef>
#include <mutex>

namespace rocr {
namespace amd {
namespace hsa {
namespace common {

template<typename LockType>
class ReaderLockGuard final {
public:
  explicit ReaderLockGuard(LockType &lock):
    lock_(lock)
  {
    lock_.ReaderLock();
  }

  ~ReaderLockGuard()
  {
    lock_.ReaderUnlock();
  }

private:
  ReaderLockGuard(const ReaderLockGuard&);
  ReaderLockGuard& operator=(const ReaderLockGuard&);

  LockType &lock_;
};

template<typename LockType>
class WriterLockGuard final {
public:
  explicit WriterLockGuard(LockType &lock):
    lock_(lock)
  {
    lock_.WriterLock();
  }

  ~WriterLockGuard()
  {
    lock_.WriterUnlock();
  }

private:
  WriterLockGuard(const WriterLockGuard&);
  WriterLockGuard& operator=(const WriterLockGuard&);

  LockType &lock_;
};

class ReaderWriterLock final {
public:
  ReaderWriterLock():
    readers_count_(0), writers_count_(0), writers_waiting_(0) {}

  ~ReaderWriterLock() {}

  void ReaderLock();

  void ReaderUnlock();

  void WriterLock();

  void WriterUnlock();

private:
  ReaderWriterLock(const ReaderWriterLock&);
  ReaderWriterLock& operator=(const ReaderWriterLock&);

  size_t readers_count_;
  size_t writers_count_;
  size_t writers_waiting_;
  std::mutex internal_lock_;
  std::condition_variable_any readers_condition_;
  std::condition_variable_any writers_condition_;
};

} // namespace common
} // namespace hsa
} // namespace amd
} // namespace rocr

#endif // AMD_HSA_LOCKS_HPP
