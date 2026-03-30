/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

//!
//! \file OCLThread.cpp
//!

#include <stdio.h>
#include <stdlib.h>

#include "OCL/Thread.h"
#ifdef _WIN32
#include <process.h>
#endif

//! pack the function pointer and data inside this struct
typedef struct __argsToThreadFunc {
  oclThreadFunc func;
  void* data;

} argsToThreadFunc;

#ifdef _WIN32
//! Windows thread callback - invokes the callback set by
//! the application in OCLThread constructor
unsigned _stdcall win32ThreadFunc(void* args) {
  argsToThreadFunc* ptr = (argsToThreadFunc*)args;
  OCLutil::Thread* obj = (OCLutil::Thread*)ptr->data;
  ptr->func(obj->getData());
  delete args;
  return 0;
}
#endif

////////////////////////////////////////////////////////////////////
//!
//! Constructor for OCLLock
//!
OCLutil::Lock::Lock() {
#ifdef _WIN32
  InitializeCriticalSection(&_cs);
#else
  pthread_mutex_init(&_lock, NULL);
#endif
}

////////////////////////////////////////////////////////////////////
//!
//! Destructor for OCLLock
//!
OCLutil::Lock::~Lock() {
#ifdef _WIN32
  DeleteCriticalSection(&_cs);
#else
  pthread_mutex_destroy(&_lock);
#endif
}

//////////////////////////////////////////////////////////////
//!
//! Try to acquire the lock, wait for the lock if unavailable
//! else hold the lock and enter the protected area
//!
void OCLutil::Lock::lock() {
#ifdef _WIN32
  EnterCriticalSection(&_cs);
#else
  pthread_mutex_lock(&_lock);
#endif
}

//////////////////////////////////////////////////////////////
//!
//! Try to acquire the lock, if unavailable the function returns
//! false and returns true if available(enters the critical
//! section as well in this case).
//!
bool OCLutil::Lock::tryLock() {
#ifdef _WIN32
  return (TryEnterCriticalSection(&_cs) != 0);
#else
  return !((bool)pthread_mutex_trylock(&_lock));
#endif
}

//////////////////////////////////////////////////////////////
//!
//! Unlock the lock
//!
void OCLutil::Lock::unlock() {
#ifdef _WIN32
  LeaveCriticalSection(&_cs);
#else
  pthread_mutex_unlock(&_lock);
#endif
}

////////////////////////////////////////////////////////////////////
//!
//! Constructor for OCLThread
//!
OCLutil::Thread::Thread() : _tid(0), _data(0) {
#ifdef _WIN32
  _ID = 0;
#else
#endif
}

////////////////////////////////////////////////////////////////////
//!
//! Destructor for OCLLock
//!
OCLutil::Thread::~Thread() {
#ifdef _WIN32
  CloseHandle(_tid);
#else
#endif
}

//////////////////////////////////////////////////////////////
//!
//! Create a new thread and return the status of the operation
//!
bool OCLutil::Thread::create(oclThreadFunc func, void* arg) {
  // Save the data internally
  _data = arg;

  unsigned int retVal;

  bool verbose = getenv("VERBOSE") != NULL;

#ifdef _WIN32
  // Setup the callback struct for thread function and pass to the
  // begin thread routine
  // xxx The following struct is allocated but never freed!!!!
  argsToThreadFunc* args = new argsToThreadFunc;
  args->func = func;
  args->data = this;

  _tid = (HANDLE)_beginthreadex(NULL, 0, win32ThreadFunc, args, 0, &retVal);

  if (verbose) {
    printf("Thread handle value = %p\n", _tid);

    printf("Done creating thread. Thread id value = %u\n", retVal);
  }
#else
  //! Now create the thread with pointer to self as the data
  retVal = pthread_create(&_tid, NULL, func, arg);

  if (verbose)
    printf("Done creating thread. Ret value %d, Self = %u\n", retVal, (unsigned int)pthread_self());
#endif

  if (retVal != 0) return false;

  return true;
}

//////////////////////////////////////////////////////////////
//!
//! Return the thread ID for the current OCLThread
//!
unsigned int OCLutil::Thread::getID() {
#ifdef _WIN32
  return GetCurrentThreadId();
  // Type cast the thread handle to unsigned in and send it over
#else
  return (unsigned int)pthread_self();
#endif
}

//////////////////////////////////////////////////////////////
//!
//! Wait for this thread to join
//!
bool OCLutil::Thread::join() {
#ifdef _WIN32
  DWORD rc = WaitForSingleObject(_tid, INFINITE);

  if (rc == WAIT_FAILED) {
    printf("Bad call to function(invalid handle?)\n");
  }
#else
  int rc = pthread_join(_tid, NULL);
#endif

  if (rc != 0) return false;

  return true;
}
