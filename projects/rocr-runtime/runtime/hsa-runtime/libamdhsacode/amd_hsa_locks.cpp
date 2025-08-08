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

#include "amd_hsa_locks.hpp"

namespace rocr {
namespace amd {
namespace hsa {
namespace common {

void ReaderWriterLock::ReaderLock()
{
  internal_lock_.lock();
  while (0 < writers_count_) {
    readers_condition_.wait(internal_lock_);
  }
  readers_count_ += 1;
  internal_lock_.unlock();
}

void ReaderWriterLock::ReaderUnlock()
{
  internal_lock_.lock();
  readers_count_ -= 1;
  if (0 == readers_count_ && 0 < writers_waiting_) {
    writers_condition_.notify_one();
  }
  internal_lock_.unlock();
}

void ReaderWriterLock::WriterLock()
{
  internal_lock_.lock();
  writers_waiting_ += 1;
  while (0 < readers_count_ || 0 < writers_count_) {
    writers_condition_.wait(internal_lock_);
  }
  writers_count_ += 1;
  writers_waiting_ -= 1;
  internal_lock_.unlock();
}

void ReaderWriterLock::WriterUnlock()
{
  internal_lock_.lock();
  writers_count_ -= 1;
  if (0 < writers_waiting_) {
    writers_condition_.notify_one();
  }
  readers_condition_.notify_all();
  internal_lock_.unlock();
}

} // namespace common
} // namespace hsa
} // namespace amd
} // namespace rocr
