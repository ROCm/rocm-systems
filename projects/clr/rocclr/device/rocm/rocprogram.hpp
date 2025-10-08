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

#pragma once

#include <string>
#include <sstream>
#include <fstream>
#include <iostream>
#include "rocdevice.hpp"

//! \namespace amd::roc HSA Device Implementation
namespace amd::roc {

//! \class empty program
class Program : public device::Program {
  friend class ClBinary;

 public:
  //! Default constructor
  Program(roc::NullDevice& device, amd::Program& owner);
  //! Default destructor
  ~Program();

  // Initialize Binary for GPU (used only for clCreateProgramWithBinary()).
  virtual bool initClBinary(char* binaryIn, size_t size);

  //! Return a typecasted GPU device
  const NullDevice& rocNullDevice() const { return static_cast<const NullDevice&>(device()); }

  //! Return a typecasted GPU device
  const Device& rocDevice() const {
    assert(!isNull());
    return static_cast<const Device&>(device());
  }

  hsa_executable_t hsaExecutable() const {
    assert(!isNull());
    return hsaExecutable_;
  }

  virtual bool createGlobalVarObj(amd::Memory** amd_mem_obj, void** device_pptr, size_t* bytes,
                                  const char* global_name) const;

 protected:
  //! Disable default copy constructor
  Program(const Program&) = delete;
  //! Disable operator=
  Program& operator=(const Program&) = delete;

  virtual bool defineGlobalVar(const char* name, void* dptr);

  bool createBinary(amd::option::Options* options) override final;

  bool createKernels(void* binary, size_t binSize, bool useUniformWorkGroupSize,
                     bool internalKernel) override final;

  bool setKernels(void* binary, size_t binSize, amd::Os::FileDesc fdesc = amd::Os::FDescInit(),
                  size_t foffset = 0, std::string uri = std::string()) override final;
 protected:
  /* HSA executable */
  hsa_executable_t hsaExecutable_;                //!< Handle to HSA executable
  hsa_code_object_reader_t hsaCodeObjectReader_;  //!< Handle to HSA code reader
};

/*@}*/  // namespace amd::roc
}  // namespace amd::roc

