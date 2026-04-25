/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef RUNTIME_HPP_
#define RUNTIME_HPP_

#include <functional>
#include "top.hpp"
#include "thread/thread.hpp"

namespace amd {

/*! \addtogroup Runtime The OpenCL Runtime
 *  @{
 */

class Runtime : AllStatic {
  static volatile int pid_;  //!< Process ID for this runtime initialization
  static volatile bool initialized_;
  static bool LibraryDetached;

 public:
  //! Return true if the OpencCL runtime is already initialized
  inline static bool initialized() { return initialized_; }

  //! Return PID if the OCL/HIP runtime was initialized in the process
  inline static int pid() { return pid_; }

  //! Initialize the OpenCL runtime.
  static bool init();

  //! Tear down the runtime.
  static void tearDown();

  //! Return true if the Runtime is still single-threaded.
  static bool singleThreaded() { return !initialized(); }

  //! Return whether the library is detached by OS
  static bool isLibraryDetached() { return LibraryDetached; }

  //! Set the library has been detached.
  static void setLibraryDetached() { LibraryDetached = true; }
};

/*@}*/

class RuntimeTearDown : public HeapObject {
 public:
  using TearDownCallback = std::function<void()>;

  RuntimeTearDown() {}
  ~RuntimeTearDown();

  static void RegisterObject(ReferenceCountedObject* obj);
  static void RegisterTearDownCallback(const std::string& msg, TearDownCallback func);

 private:
  static std::vector<ReferenceCountedObject*> external_;
  static std::vector<std::pair<std::string, TearDownCallback>> tear_down_funcs_;
};

}  // namespace amd

#endif /*RUNTIME_HPP_*/
