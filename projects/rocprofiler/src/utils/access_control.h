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

#ifndef SRC_UTILS_ACCESS_CONTROL_H_
#define SRC_UTILS_ACCESS_CONTROL_H_

#include <atomic>
#include <iostream>
#include <mutex>

// TODO(aelwazir): To be implemented

// type for reference counter
typedef std::atomic<uint64_t> ROCProfilerRefCount;

// type for access/release control flag
typedef std::atomic<bool> ROCProfilerARControl;

namespace rocprofiler {

// the access/release control flag can be shared or exclusive
static ROCProfilerARControl kARShared = ATOMIC_VAR_INIT(0);
static ROCProfilerARControl kARExclusive = ATOMIC_VAR_INIT(1);

/*-------------------------------------------------------------------------*/
/**
 * @Synopsis A class for defining access mode for any resource and
 * maintaining the refernce count for accesses to the resource.
 * All the resources must include this class object for corectness
 */
/*--------------------------------------------------------------------------*/
class AccessControl {
 public:
  //!< constructor that defaults the RefCount and ARControl
  AccessControl(ROCProfilerRefCount rrc = ATOMIC_VAR_INIT(0),
                ROCProfilerARControl rac = ATOMIC_VAR_INIT(kARShared.load()));

  //!< naive destructor
  ~AccessControl();

  bool Acquire(ROCProfilerARControl access_mode);
  void Release(ROCProfilerARControl access_mode);

 private:
  ROCProfilerRefCount rcount_;      //!< refernce count, every successful access
                                    //! increments it
  ROCProfilerARControl arcontrol_;  //!< shared access or exclusive?

  std::mutex accessmutex_;
};

//<!
// AccessControl::AccessControl(ROCProfilerRefCount rrc, ROCProfilerARControl rac)
// {
//   rcount_ = rrc.load();
//   arcontrol_ = rac.load();
// }

//<!
AccessControl::~AccessControl() {
  // we cannot decrement reference count here, we can only hope it is 0
#ifdef ROCPROFILER_DEBUG_PRINT
  int zero = 0;
  if (rcount_.compare_exchange_strong(zero&, 0, std::memory_order_acq_rel))
    cout << endl
         << __FILE__ << __FUNC__ << __LINE__
         << ": reference count not"
            " zero"
         << endl;
#endif
}

#if 0
//!< three steps, 1) get scoped lock, 2) check if it is shared to shared
//!< acquire, and 3) if it is exclusive, check if refcount is 0
bool AccessControl::Acquire(ROCProfilerARControl rac) {
  // first, ensure we have a lock_guard
  std::lock_guard<std::mutex> lg(accessmutex_);
  // we should always expect the current value to be shared
  ROCProfilerARControl rac_expected = kARShared.load();
  // if what we have is what we expect then we are done
  if (rac.load(std::memory_order_relaxed) == rac_expected.load()) {
    // rcount_.store(std::memory_order_acq_rel,
    //       rcount_load(std::memory_order_acq_rel)+1));
    return true;
  }
  // now, we have to switch from shared to exclusive, but we need the refcount
  // to be 0. If it is 0, return true, otherwise false
  // int zero = 0;
  // return rcount_.compare_exchange_strong(zero&, 0,
  //    std::memory_order_acq_rel);
}
#endif
//!< always exchange to kARShared and reduce refcount
// void AccessControl::Release(

}  // namespace rocprofiler

#endif  // SRC_UTILS_ACCESS_CONTROL_H_
