/* Copyright (c) 2010 - 2021 Advanced Micro Devices, Inc.

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

#ifndef OCL_THREAD_H
#define OCL_THREAD_H

//!
//! \file Thread.h
//!

#ifdef _WIN32
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0501
#endif

#include "windows.h"
#else
#include "pthread.h"
#endif

//! Entry point for the thread
//! prototype of the entry point in windows
typedef void* (*oclThreadFunc)(void*);

namespace OCLutil {
//! \class Lock
//! \brief Provides a wrapper for locking primitives used to
//!  synchronize _CPU_ threads.
//!
//! Common usage would be:
//!
//!    OCL::Lock lock;
//!
//!    ....
//!
//!    // Critical section begins
//!
//!    lock.lock();
//!
//!    .....
//!
//!    // Critical section ends
//!
//!    lock.unlock();
//!

class Lock {
 public:
  //! Constructor for OCLLock
  Lock();

  //! Destructor for OCLLock
  ~Lock();

  //! Try to acquire the lock, if available continue, else wait on the lock
  void lock();

  //! Try to acquire the lock, if available, hold it, else continue doing
  //! something else
  bool tryLock();

  //! Unlock the lock and return
  void unlock();

 private:
  /////////////////////////////////////////////////////////////
  //!
  //! Private data members and methods
  //!

  //! System specific synchronization primitive
#ifdef _WIN32
  CRITICAL_SECTION _cs;
#else
  pthread_mutex_t _lock;
#endif
};

//////////////////////////////////////////////////////////////
//!
//! \class Thread
//! \brief Provides a wrapper for creating a _CPU_ thread.
//!
//! This class provides a simple wrapper to a CPU thread/
//! The class name might be a bit confusing, esp considering
//! the GPU has it's own threads as well.
//!
class Thread {
 public:
  //! Thread constructor and destructor. Note that the thread is
  //! NOT created in the constructor. The thread creation takes
  //! place in the create method
  Thread();

  ~Thread();

  //! Wrapper for pthread_create. Pass the thread's entry
  //! point and data to be passed to the routine
  bool create(oclThreadFunc func, void* arg);

  //! Wrapper for pthread_join. The calling thread
  //! will wait until _this_ thread exits
  bool join();

  //! Get the thread data passed by the application
  void* getData() { return _data; }

  //! Get the thread ID
  static unsigned int getID();

 private:
  /////////////////////////////////////////////////////////////
  //!
  //! Private data members and methods
  //!

#ifdef _WIN32
  //!  store the handle
  HANDLE _tid;

  unsigned int _ID;
#else
  pthread_t _tid;

  pthread_attr_t _attr;
#endif

  void* _data;
};
};  // namespace OCLutil
#endif
