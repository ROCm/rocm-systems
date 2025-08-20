/* Copyright (c) 2008 - 2022 Advanced Micro Devices, Inc.

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

#ifndef RUNTIME_HPP_
#define RUNTIME_HPP_

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
  static std::vector<ReferenceCountedObject*> external_;

 public:
  RuntimeTearDown() {}
  ~RuntimeTearDown();

  static void RegisterObject(ReferenceCountedObject* obj);
};

}  // namespace amd

#endif /*RUNTIME_HPP_*/
