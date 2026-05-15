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

#include "thread/thread.hpp"
#include "thread/monitor.hpp"
#include "os/os.hpp"

#if defined(_WIN32) || defined(__CYGWIN__)
#include <windows.h>
#endif  // _WIN32

namespace amd {

void Thread::create() {
  selfSuspendLock_ = new Monitor();
  data_ = NULL;
  handle_ = NULL;
  setState(CREATED);
}

Thread::Thread(const std::string& name, size_t stackSize, bool spawn)
    : handle_(NULL), name_(name), stackSize_(stackSize) {
  create();

  if (!spawn) return;

  if ((handle_ = Os::createOsThread(this))) {
    // Now we need to wait for Thread::main() to report back.
    ScopedLock sl(selfSuspendLock_);
    // Wait for main() to update state as INITIALIZED
    while (state() != Thread::INITIALIZED) {
      selfSuspendLock_->wait();
    }
  }
}

Thread::~Thread() {
#if defined(_WIN32)
  if (handle_ != NULL) {
    ::CloseHandle((HANDLE)handle_);
  }
#endif
  delete selfSuspendLock_;
}

void* Thread::main() {
#ifdef DEBUG
  Os::setCurrentThreadName(name().c_str());
#endif  // DEBUG
  Os::currentStackInfo(&stackBase_, &stackSize_);
  setCurrent();

  // Notify the parent thread that we are up and running.
  {
    ScopedLock sl(selfSuspendLock_);
    // Notify parent thread that the state is update as INITIALIZED
    setState(INITIALIZED);
    selfSuspendLock_->notify();
  }

  // Now we need to wait for Thread::start() to report back.
  {
    ScopedLock sl(selfSuspendLock_);
    // wait for start() to update state as RUNNABLE
    while (state() != Thread::RUNNABLE) {
      selfSuspendLock_->wait();
    }
  }


  if (state() == RUNNABLE) {
    run(data_);
  }

  setState(FINISHED);
  return NULL;
}

bool Thread::start(void* data) {
  if (state() != INITIALIZED) {
    return false;
  }

  data_ = data;
  // Notify the thread that the parent thread are up and running.
  {
    ScopedLock sl(selfSuspendLock_);
    // Notify main() that the state is updated as RUNNABLE
    setState(RUNNABLE);
    selfSuspendLock_->notify();
  }

  return true;
}

void Thread::resume() {
  ScopedLock sl(selfSuspendLock_);
  selfSuspendLock_->notify();
}

#if defined(__linux__)

namespace details {

__thread Thread* thread_ __attribute__((tls_model("initial-exec")));

}  // namespace details

void Thread::registerStack(address base, address top) {
  // Nothing to do.
}

void Thread::setCurrent() { details::thread_ = this; }

#elif defined(_WIN32)

namespace details {

#if defined(USE_DECLSPEC_THREAD)
__declspec(thread) Thread* thread_;
#else   // !USE_DECLSPEC_THREAD
DWORD threadIndex_ = TlsAlloc();
#endif  // !USE_DECLSPEC_THREAD

}  // namespace details

void Thread::registerStack(address base, address top) {
  // Nothing to do.
}

void Thread::setCurrent() {
#if defined(USE_DECLSPEC_THREAD)
  details::thread_ = this;
#else   // !USE_DECLSPEC_THREAD
  TlsSetValue(details::threadIndex_, this);
#endif  // !USE_DECLSPEC_THREAD
}

#endif

bool Thread::init() {
  static bool initialized_ = false;

  // We could use InitOnceExecuteOnce/pthread_once here:
  if (initialized_) {
    return true;
  }
  initialized_ = true;

  return true;
}

void Thread::tearDown() {
#if defined(_WIN32) && !defined(USE_DECLSPEC_THREAD)
  if (details::threadIndex_ != TLS_OUT_OF_INDEXES) {
    TlsFree(threadIndex_);
  }
#endif  // _WIN32 && !USE_DECLSPEC_THREAD
}

}  // namespace amd
