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
#include <future>
#include <mutex>
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

  //! Launch the HotSwap code-object preparation (retarget) on a background
  //! thread using the binary already stored in clBinary(). setKernels() waits
  //! on the result instead of retargeting synchronously on the loading thread.
  void prepareCodeObjectAsync() override;

  //! Block until an in-flight prepareCodeObjectAsync() has finished so the
  //! source bytes backing this program can be safely released.
  void waitForCodeObjectPrepare() override;

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

  //! Create the source reader and (when HotSwap applies) the prepared reader
  //! from the given code object. Runs at most once (guarded by readerOnce_) and
  //! is safe to call from a background thread. Records its outcome in
  //! prepareStatus_/readerReady_ and appends any error to buildLog_.
  void prepareCodeObjectReader(void* binary, size_t binSize, amd::Os::FileDesc fdesc,
                               size_t foffset, std::string uri);

 protected:
  /* HSA executable */
  hsa_executable_t hsaExecutable_;                //!< Handle to HSA executable
  hsa_code_object_reader_t hsaCodeObjectReader_;  //!< Handle to HSA code reader
  //!< Handle to the prepared (hotswap-retargeted) reader, when one was produced
  //!< by the loader prepare API. Owns the retargeted artifact and is loaded in
  //!< place of hsaCodeObjectReader_; released with the program.
  hsa_code_object_reader_t preparedCodeObjectReader_;

  //!< Future for an asynchronous prepareCodeObjectReader() kicked off by
  //!< prepareCodeObjectAsync(). Valid only when async preparation was started.
  std::future<void> prepareFuture_;
  //!< Guards prepareCodeObjectReader() so the readers are created exactly once
  //!< regardless of whether the async future or setKernels() runs it first.
  std::once_flag readerOnce_;
  //!< Set once the readers have been created (successfully or not).
  bool readerReady_ = false;
  //!< Status of the reader creation / preparation step.
  hsa_status_t prepareStatus_ = HSA_STATUS_SUCCESS;
};

/*@}*/  // namespace amd::roc
}  // namespace amd::roc

