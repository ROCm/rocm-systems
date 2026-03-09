/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef THREAD_HPP_
#define THREAD_HPP_

#include "top.hpp"
#include "os/os.hpp"

#include <string>

#if defined(_WIN32)
#define USE_DECLSPEC_THREAD 1
#if !defined(USE_DECLSPEC_THREAD)
#include <windows.h>
#endif /*!USE_DECLSPEC_THREAD*/
#endif /*_WIN32*/

namespace amd {

/*! \addtogroup Threads Threading package
 *  @{
 *
 *  \addtogroup OsThread Native Threads
 *  @{
 */

class Monitor;

class Thread : public HeapObject {
  friend const void* Os::createOsThread(Thread*);

 public:
  enum ThreadState { CREATED, INITIALIZED, RUNNABLE, SUSPENDED, FINISHED, FAILED };

 private:
  //! System thread handle.
  const void* handle_;
  //! The thread's name.
  const std::string name_;
  //! Current running state.
  volatile ThreadState state_;
  //! The argument passed to run()
  void* data_;

  Monitor* selfSuspendLock_;  //!< For self suspend/resume.

 protected:
  address stackBase_;  //!< Main stack base.
  size_t stackSize_;   //!< Main stack size.

 private:
  /*! \brief The start wrapper for all newly create threads.
   *  This is called from the pthread_create start_thread.
   */
  static void* entry(Thread* thread);

  /*! \brief Thread main (called from the main function).
   *  Setup the thread for running and wait for the semaphore to be signaled.
   */
  void* main();

  //! The entry point for this thread.
  virtual void run(void* data) = 0;

 protected:
  //! Bring this thread to the created state.
  void create();

  //! Set the current thread state.
  void setState(ThreadState state) { state_ = state; }

  //! Set the thread-local _thread variable (used by current()).
  void setCurrent();

  //! Register the given memory region as a valid stack.
  void registerStack(address base, address top);

  /*! \brief Construct a new thread.
   *  If \a spawn is false, do not create a new OS thread, instead,
   *  bind to the currently running on.
   */
  explicit Thread(const std::string& name, size_t stackSize = 0 /*use system default*/,
                  bool spawn = true /* create a new Os::thread */);

 public:
  //! Return the currently running thread instance.
  static inline Thread* current();

  //! Initialize the OsThread package.
  static bool init();

  //! Tear down the OsThread package.
  static void tearDown();

  //! Destroy this thread.
  virtual ~Thread();

  //! Return the thread's name
  const std::string& name() const { return name_; }

  //! Get the system thread handle.
  const void* handle() const { return handle_; }

  //! Start the thread execution
  bool start(void* data = NULL);

  //! Resume the thread
  void resume();

  //! Return true is this is the host thread.
  virtual bool isHostThread() const { return false; }

  //! Return true if this is a worker thread.
  virtual bool isWorkerThread() const { return false; }

  //! Get the current thread state.
  ThreadState state() const { return state_; }

  //! Return this thread's stack base.
  address stackBase() const { return stackBase_; }
  //! Return this thread's stack size.
  size_t stackSize() const { return stackSize_; }
  //! Return this thread's stack bottom.
  address stackBottom() const { return stackBase() - stackSize(); }
};

/*! @}
 *  @}
 */

namespace details {

#if defined(__linux__)

extern __thread Thread* thread_ __attribute__((tls_model("initial-exec")));

static inline Thread* currentThread() { return thread_; }

#elif defined(_WIN32)

#if defined(USE_DECLSPEC_THREAD)
extern __declspec(thread) Thread* thread_;
#else   // !USE_DECLSPEC_THREAD
extern DWORD threadIndex_;
#endif  // !USE_DECLSPEC_THREAD

static inline Thread* currentThread() {
#if defined(USE_DECLSPEC_THREAD)
  return thread_;
#else   // !USE_DECLSPEC_THREAD
  return (Thread*)TlsGetValue(threadIndex_);
#endif  // !USE_DECLSPEC_THREAD
}

#endif  // _WIN32

}  // namespace details

inline Thread* Thread::current() { return details::currentThread(); }

}  // namespace amd

#endif /*THREAD_HPP_*/
