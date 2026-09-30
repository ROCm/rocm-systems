/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

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
                                  const char* global_name) const override;

 protected:
  //! Disable default copy constructor
  Program(const Program&) = delete;
  //! Disable operator=
  Program& operator=(const Program&) = delete;

  virtual bool defineGlobalVar(const char* name, void* dptr) override;

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

