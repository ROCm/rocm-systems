/* Copyright (c) 2010 - 2021 Advanced Micro Devices, Inc.

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

/*! \addtogroup HSA OCL Stub Implementation
 *  @{
 */

//! HSA OCL STUB Implementation
namespace amd::roc {

//! Device settings
class Settings : public device::Settings {
 public:
  enum Hmm : uint32_t {
    EnableSystemMemory = 0x01,    //!< Forces system memory preference by default
    EnableMallocPrefetch = 0x02,  //!< Skips default prefetch after allocation
    EnableSvmTracking = 0x04,     //!< Enables SW SVM tracking
    EnableDebugSvm = 0x08         //!< Extra debug flag (reserved for runtime developers)
  };

  union {
    struct {
      uint doublePrecision_ : 1;       //!< Enables double precision support
      uint enableLocalMemory_ : 1;     //!< Enable GPUVM memory
      uint enableNCMode_ : 1;          //!< Enable Non Coherent mode for system memory
      uint imageDMA_ : 1;              //!< Enable direct image DMA transfers
      uint imageBufferWar_ : 1;        //!< Image buffer workaround for Gfx10
      uint cpu_wait_for_signal_ : 1;   //!< Wait for HSA signal on CPU
      uint system_scope_signal_ : 1;   //!< HSA signal is visibile to the entire system
      uint fgs_kernel_arg_ : 1;        //!< Use fine grain kernel arg segment
      uint barrier_value_packet_ : 1;  //!< Barrier value packet functionality
      uint dynamic_queues_ : 1;        //!< Dynamic queues management
      uint blocking_blit_ : 1;         //!< Blit ops can be blocking on CPU
      uint reserved_ : 21;
    };
    uint value_;
  };

  //! Default max workgroup size for 1D
  int maxWorkGroupSize_;

  //! Preferred workgroup size
  uint preferredWorkGroupSize_;

  uint kernargPoolSize_;
  uint numDeviceEvents_;  //!< The number of device events
  uint numWaitEvents_;    //!< The number of wait events for device enqueue

  size_t xferBufSize_;        //!< Transfer buffer size for image copy optimization
  size_t pinnedXferSize_;     //!< Pinned buffer size for transfer
  size_t pinnedMinXferSize_;  //!< Minimal buffer size for pinned transfer

  size_t sdmaCopyThreshold_;   //!< Use SDMA to copy above this size
  size_t sdma_p2p_threshold_;  //!< Use SDMA in P2P above this size

  uint32_t hmmFlags_;       //!< HMM functionality control flags
  uint32_t limit_blit_wg_;  //!< The number of workgroups for blit execution

  //! Default constructor
  Settings();

  //! Creates settings
  bool create(bool fullProfile, const amd::Isa& isa, bool enableXNACK, bool coop_groups = false,
              bool isXgmi = false, bool hasValidHDPFlush = true);

 private:
  //! Disable copy constructor
  Settings(const Settings&);

  //! Disable assignment
  Settings& operator=(const Settings&);

  //! Overrides current settings based on registry/environment
  void override();

  //! Determine how kernel arguments should be implemented given ASIC (host
  //! memory, device memory, device memory with memory ordering workaround)
  void setKernelArgImpl(const amd::Isa& isa, bool isXgmi, bool hasValidHDPFlush);
};

/*@}*/  // namespace amd::roc
}  // namespace amd::roc

