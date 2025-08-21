/* Copyright (c) 2015 - 2021 Advanced Micro Devices, Inc.

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

#include "top.hpp"
#include "device/device.hpp"
#include "device/pal/paldevice.hpp"
#include "palPerfExperiment.h"

namespace amd::pal {

enum class PCIndexSelect : uint {
  None = 0,      ///< no index
  Instance,      ///< index by block instance -- (1 to 1 mapping with PAL)
  ShaderEngine,  ///< index by shader engine (sum = 1 value for all PAL Instances)
  ShaderArray,   ///< index by shader array (sum = 1 value for each shader array)
  ComputeUnit,   ///< index by compute unit
};

class VirtualGPU;

class PalCounterReference : public amd::ReferenceCountedObject {
 public:
  static PalCounterReference* Create(VirtualGPU& gpu);

  //! Default constructor
  PalCounterReference(VirtualGPU& gpu  //!< Virtual GPU device object
                      )
      : gpu_(gpu),
        perfExp_(nullptr),
        layout_(nullptr),
        memory_(nullptr),
        cpuAddr_(nullptr),
        numExpCounters_(0) {}

  //! Get PAL counter
  Pal::IPerfExperiment* iPerf() const { return perfExp_; }

  //! Returns the virtual GPU device
  const VirtualGPU& gpu() const { return gpu_; }

  //! Prepare for execution
  bool finalize();

  //! Returns the PAL counter results
  uint64_t result(const std::vector<int>& index);

  //! Get the latest Experiment Counter index
  uint getPalCounterIndex() { return numExpCounters_++; };

 protected:
  //! Default destructor
  ~PalCounterReference();

 private:
  //! Disable copy constructor
  PalCounterReference(const PalCounterReference&);

  //! Disable operator=
  PalCounterReference& operator=(const PalCounterReference&);

  VirtualGPU& gpu_;                   //!< The virtual GPU device object
  Pal::IPerfExperiment* perfExp_;     //!< PAL performance experiment object
  Pal::GlobalCounterLayout* layout_;  //!< Layout of the result
  Memory* memory_;                    //!< Memory used by PAL performance experiment
  void* cpuAddr_;                     //!< CPU address of memory_
  uint numExpCounters_;               //!< Number of Experiment Counter created
};

//! Performance counter implementation on GPU
class PerfCounter : public device::PerfCounter {
 public:
  //! The performance counter info
  struct Info : public amd::EmbeddedObject {
    uint blockIndex_;            //!< Index of the block to configure
    uint counterIndex_;          //!< Index of the hardware counter
    uint eventIndex_;            //!< Event you wish to count with the counter
    PCIndexSelect indexSelect_;  //!< IndexSelect type of the counter
  };

  //! Constructor for the GPU PerfCounter object
  PerfCounter(const Device& device,         //!< A GPU device object
              PalCounterReference* palRef,  //!< Counter Reference
              uint32_t blockIndex,          //!< HW block index
              uint32_t counterIndex,        //!< Counter index within the block
              uint32_t eventIndex)          //!< Event index for profiling
      : gpuDevice_(device), palRef_(palRef) {
    info_.blockIndex_ = blockIndex;
    info_.counterIndex_ = counterIndex;
    info_.eventIndex_ = eventIndex;
    convertInfo();
  }

  //! Destructor for the GPU PerfCounter object
  virtual ~PerfCounter();

  //! Creates the current object
  bool create();

  //! Returns the specific information about the counter
  uint64_t getInfo(uint64_t infoType  //!< The type of returned information
  ) const;

  //! Returns the GPU device, associated with the current object
  const Device& dev() const { return gpuDevice_; }

  //! Returns the virtual GPU device
  const VirtualGPU& gpu() const { return palRef_->gpu(); }

  //! Returns the PAL performance counter descriptor
  const Info* info() const { return &info_; }

  //! Returns the Info structure for performance counter
  Pal::IPerfExperiment* iPerf() const { return palRef_->iPerf(); }

 private:
  //! Disable default copy constructor
  PerfCounter(const PerfCounter&);

  //! Disable default operator=
  PerfCounter& operator=(const PerfCounter&);

  //! Convert info from ORCA to PAL
  void convertInfo();

  const Device& gpuDevice_;      //!< The backend device
  PalCounterReference* palRef_;  //!< Reference counter
  Info info_;                    //!< The info structure for perfcounter
  std::vector<int> index_;       //!< Counter index in the PAL container
};

}  // namespace amd::pal
