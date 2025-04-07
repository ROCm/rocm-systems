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

// Library of syncronization primitives - to be added to as needed.

#ifndef HSA_RUNTIME_CORE_UTIL_LOCKS_H_
#define HSA_RUNTIME_CORE_UTIL_LOCKS_H_

#include "utils.h"
#include "os.h"

namespace rocr {

class HybridMutex {
 public:
  HybridMutex():lock_(0) { 
    sem_ = os::CreateSemaphore(); 
  }

  ~HybridMutex() { 
    os::DestroySemaphore(sem_); 
  }

  bool Try() {
    int old = 0;
    return lock_.compare_exchange_strong(old, 1);
  }

  bool Acquire() {
    int cnt = maxSpinIterPause + maxSpinIterYield;

    int old = 0;
    while (!lock_.compare_exchange_strong(old, 1)) {
      cnt--;
      if (cnt > maxSpinIterPause) {
        _mm_pause();
      } else if (cnt-- > maxSpinIterYield) {
        os::YieldThread();
      } else {
        os::WaitSemaphore(sem_);
        cnt = maxSpinIterPause + maxSpinIterYield;
      }
      old = 0;
    }
    return true;
  }

  void Release() {
    int old = 1;
    if (lock_.compare_exchange_strong(old, 0))
      os::PostSemaphore(sem_);
  }

 private:
  std::atomic<int> lock_;
  os::Semaphore sem_;
  const uint32_t maxSpinIterPause = 55;
  const uint32_t maxSpinIterYield = 55;

  /// @brief: Disable copiable and assignable ability.
  DISALLOW_COPY_AND_ASSIGN(HybridMutex);
};


/// @brief: a class represents a kernel mutex.
/// Uses the kernel's scheduler to keep the waiting thread from being scheduled
/// until the lock is released (Best for long waits, though anything using
/// a kernel object is a long wait).
class KernelMutex {
 public:
  KernelMutex() { lock_ = os::CreateMutex(); }
  ~KernelMutex() { os::DestroyMutex(lock_); }

  bool Try() { return os::TryAcquireMutex(lock_); }
  bool Acquire() { return os::AcquireMutex(lock_); }
  void Release() { os::ReleaseMutex(lock_); }

 private:
  os::Mutex lock_;

  /// @brief: Disable copiable and assignable ability.
  DISALLOW_COPY_AND_ASSIGN(KernelMutex);
};

/// @brief: represents a spin lock.
/// For very short hold durations on the order of the thread scheduling
/// quanta or less.
class SpinMutex {
 public:
  SpinMutex() { lock_ = 0; }

  bool Try() {
    int old = 0;
    return lock_.compare_exchange_strong(old, 1);
  }
  bool Acquire() {
    int old = 0;
    while (!lock_.compare_exchange_strong(old, 1))
	{
		old=0;
    os::YieldThread();
	}
    return true;
  }
  void Release() { lock_ = 0; }

 private:
  std::atomic<int> lock_;

  /// @brief: Disable copiable and assignable ability.
  DISALLOW_COPY_AND_ASSIGN(SpinMutex);
};

class KernelEvent {
 public:
  KernelEvent() { evt_ = os::CreateOsEvent(true, true); }
  ~KernelEvent() { os::DestroyOsEvent(evt_); }

  bool IsSet() { return os::WaitForOsEvent(evt_, 0)==0; }
  bool WaitForSet() { return os::WaitForOsEvent(evt_, 0xFFFFFFFF)==0; }
  void Set() { os::SetOsEvent(evt_); }
  void Reset() { os::ResetOsEvent(evt_); }

 private:
  os::EventHandle evt_;

  /// @brief: Disable copiable and assignable ability.
  DISALLOW_COPY_AND_ASSIGN(KernelEvent);
};

/// @brief: represents a yielding shared mutex.
/// aka read/write mutex
class KernelSharedMutex {
 public:
  /// @brief: Interfaces ScopedAcquire to shared operations.
  class Shared {
   public:
    explicit Shared(KernelSharedMutex* lock) : lock_(lock) {}
    bool Try() { return lock_->TryShared(); }
    bool Acquire() { return lock_->AcquireShared(); }
    void Release() { lock_->ReleaseShared(); }

   private:
    KernelSharedMutex* lock_;
  };

  KernelSharedMutex() { lock_ = os::CreateSharedMutex(); }
  ~KernelSharedMutex() { os::DestroySharedMutex(lock_); }

  // Exclusive mode operations
  bool Try() { return os::TryAcquireSharedMutex(lock_); }
  bool Acquire() { return os::AcquireSharedMutex(lock_); }
  void Release() { os::ReleaseSharedMutex(lock_); }

  // Shared mode operations
  bool TryShared() { return os::TrySharedAcquireSharedMutex(lock_); }
  bool AcquireShared() { return os::SharedAcquireSharedMutex(lock_); }
  void ReleaseShared() { os::SharedReleaseSharedMutex(lock_); }

  // Return shared operations interface
  Shared shared() { return Shared(this); }

 private:
  os::SharedMutex lock_;

  /// @brief: Disable copiable and assignable ability.
  DISALLOW_COPY_AND_ASSIGN(KernelSharedMutex);
};

/// @brief: Type trait to identify mutex types
template <class T> class isMutex {
 public:
  enum { value = false };
};
template <> class isMutex<HybridMutex> {
 public:
  enum { value = true };
};
template <> class isMutex<KernelMutex> {
 public:
  enum { value = true };
};
template <> class isMutex<SpinMutex> {
 public:
  enum { value = true };
};
template <> class isMutex<KernelSharedMutex> {
 public:
  enum { value = true };
};

/// @brief: A class behaves as a lock in a scope. When trying to enter into the
/// critical section, creat a object of this class. After the control path goes
/// out of the scope, it will release the lock automatically.
template <class LockType> class ScopedAcquire {
 public:
  /// @brief: When constructing, acquire the lock.
  /// @param: lock(Input), pointer to an existing lock.
  explicit ScopedAcquire(LockType* lock) : lock_(lock), doRelease(true) {
    static_assert(isMutex<LockType>::value, "ScopedAcquire requires a mutex type.");
    lock_.Acquire();
  }
  explicit ScopedAcquire(LockType lock) : lock_(lock), doRelease(true) {
    static_assert(!isMutex<LockType>::value, "Mutex types are not copyable.");
    lock_.Acquire();
  }

  /// @brief: when destructing, release the lock.
  ~ScopedAcquire() {
    if (doRelease) lock_.Release();
  }

  /// @brief: Release the lock early.  Avoid using when possible.
  void Release() {
    lock_.Release();
    doRelease = false;
  }

 private:
  /// @brief: Adapts between pointers to mutex types and mutex pointer types.
  template <class T, bool B> class container {
   public:
    container(T* lock) : lock_(lock) {}
    __forceinline bool Acquire() { return lock_->Acquire(); }
    __forceinline void Release() { return lock_->Release(); }

   private:
    T* lock_;
  };

  /// @brief: Specialization for mutex pointer types.
  template <class T> class container<T, false> {
   public:
    container(T lock) : lock_(lock) {}
    __forceinline bool Acquire() { return lock_.Acquire(); }
    __forceinline void Release() { return lock_.Release(); }

   private:
    T lock_;
  };

  container<LockType, isMutex<LockType>::value> lock_;
  bool doRelease;

  /// @brief: Disable copiable and assignable ability.
  DISALLOW_COPY_AND_ASSIGN(ScopedAcquire);
};

}  // namespace rocr

#endif  // HSA_RUNTIME_CORE_SUTIL_LOCKS_H_
