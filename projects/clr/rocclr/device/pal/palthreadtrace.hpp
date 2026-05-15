/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "top.hpp"
#include "device/device.hpp"
#include "device/pal/paldevice.hpp"
#include "palPerfExperiment.h"

#include <vector>
namespace amd::pal {

class VirtualGPU;

class PalThreadTraceReference : public amd::ReferenceCountedObject {
 public:
  static PalThreadTraceReference* Create(VirtualGPU& gpu);

  //! Default constructor
  PalThreadTraceReference(VirtualGPU& gpu  //!< Virtual GPU device object
                          )
      : gpu_(gpu), perfExp_(nullptr), layout_(nullptr), memory_(nullptr) {}

  //! Get PAL thread race object
  Pal::IPerfExperiment* iPerf() const { return perfExp_; }

  //! Returns the virtual GPU device
  const VirtualGPU& gpu() const { return gpu_; }

  //! Prepare for execution
  bool finalize();

  //! Copy ThreadTrace capture to User Buffer
  void copyToUserBuffer(Memory* dstMem, uint seIndex);

 protected:
  //! Default destructor
  ~PalThreadTraceReference();

 private:
  //! Disable copy constructor
  PalThreadTraceReference(const PalThreadTraceReference&);

  //! Disable operator=
  PalThreadTraceReference& operator=(const PalThreadTraceReference&);

  VirtualGPU& gpu_;                 //!< The virtual GPU device object
  Pal::IPerfExperiment* perfExp_;   //!< PAL performance experiment object
  Pal::ThreadTraceLayout* layout_;  //!< Layout of the result
  Memory* memory_;                  //!< Memory bound to PerfExperiment
};

//! ThreadTrace implementation on GPU
class ThreadTrace : public device::ThreadTrace {
 public:
  //! Constructor for the GPU ThreadTrace object
  ThreadTrace(Device& device,                            //!< A GPU device object
              PalThreadTraceReference* palRef,           //!< Reference ThreadTrace
              const std::vector<amd::Memory*>& memObjs,  //!< ThreadTrace memory objects
              uint numSe                                 //!< Number of Shader Engines
              )
      : gpuDevice_(device), palRef_(palRef), numSe_(numSe), memObj_(memObjs) {}

  //! Destructor for the GPU ThreadTrace object
  virtual ~ThreadTrace();

  //! Creates the current object4
  bool create();

  // Populate ThreadTrace memory with PerfExperiment memory
  void populateUserMemory();

  //! Returns the specific information about the thread trace object
  bool info(uint infoType,  //!< The type of returned information
            uint* info,     //!< The returned information
            uint infoSize   //!< The size of returned information
  ) const;

  //! Set isNewBufferBinded_ to true/false if new buffer was binded/unbinded respectively
  void setNewBufferBinded(bool isNewBufferBinded) {}

  //! Returns the GPU device, associated with the current object
  const Device& dev() const { return gpuDevice_; }

  //! Returns the virtual GPU device
  const VirtualGPU& gpu() const { return palRef_->gpu(); }

  //! Get PAL thread trace object
  Pal::IPerfExperiment* iPerf() const { return palRef_->iPerf(); }

 private:
  //! Disable default copy constructor
  ThreadTrace(const ThreadTrace&);

  //! Disable default operator=
  ThreadTrace& operator=(const ThreadTrace&);

  const Device& gpuDevice_;           //!< The backend device
  PalThreadTraceReference* palRef_;   //!< Reference ThreadTrace
  uint numSe_;                        //!< Number of Shader Engines
  std::vector<amd::Memory*> memObj_;  //!< ThreadTrace memory objects
};

}  // namespace amd::pal
