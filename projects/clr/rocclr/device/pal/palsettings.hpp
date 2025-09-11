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
#include "library.hpp"
#include "palDevice.h"

/*! \addtogroup pal PAL Resource Implementation
 *  @{
 */

//! PAL Device Implementation
namespace amd::pal {

//! Device settings
class Settings : public device::Settings {
 public:
  //! Debug GPU flags
  enum DebugGpuFlags {
    CheckForILSource = 0x00000001,
    StubCLPrograms = 0x00000002,  //!< Enables OpenCL programs stubbing
    LockGlobalMemory = 0x00000004,
  };

  enum BlitEngineType {
    BlitEngineDefault = 0x00000000,
    BlitEngineHost = 0x00000001,
    BlitEngineCAL = 0x00000002,
    BlitEngineKernel = 0x00000003,
  };

  enum HostMemFlags {
    HostMemDisable = 0x00000000,
    HostMemBuffer = 0x00000001,
    HostMemImage = 0x00000002,
  };

  union {
    struct {
      uint remoteAlloc_ : 1;          //!< Allocate remote memory for the heap
      uint stagedXferRead_ : 1;       //!< Uses a staged buffer read
      uint stagedXferWrite_ : 1;      //!< Uses a staged buffer write
      uint disablePersistent_ : 1;    //!< Disables using persistent memory for staging
      uint imageSupport_ : 1;         //!< Report images support
      uint doublePrecision_ : 1;      //!< Enables double precision support
      uint use64BitPtr_ : 1;          //!< Use 64bit pointers on GPU
      uint force32BitOcl20_ : 1;      //!< Force 32bit apps to take CLANG/HSAIL path on GPU
      uint imageDMA_ : 1;             //!< Enable direct image DMA transfers
      uint threadTraceEnable_ : 1;    //!< Thread trace enable
      uint svmAtomics_ : 1;           //!< SVM device atomics
      uint svmFineGrainSystem_ : 1;   //!< SVM fine grain system support
      uint useDeviceQueue_ : 1;       //!< Submit to separate device queue
      uint rgpSqttWaitIdle_ : 1;      //!< Wait for idle after SQTT trace
      uint rgpSqttForceDisable_ : 1;  //!< Disables SQTT
      uint enableHwP2P_ : 1;          //!< Forces HW P2P path for testing
      uint imageBufferWar_ : 1;       //!< Image buffer workaround for Gfx10
      uint disableSdma_ : 1;          //!< Disable SDMA support
      uint alwaysResident_ : 1;       //!< Make resources resident at allocation time
      uint reserved_ : 13;
    };
    uint value_;
  };

  uint oclVersion_;              //!< Reported OpenCL version support
  uint debugFlags_;              //!< Debug GPU flags
  uint hwLDSSize_;               //!< HW local data store size
  uint maxWorkGroupSize_;        //!< Requested workgroup size for this device
  uint preferredWorkGroupSize_;  //!< Requested preferred workgroup size for this device
  uint blitEngine_;              //!< Blit engine type
  uint cacheLineSize_;           //!< Cache line size in bytes
  uint cacheSize_;               //!< L1 cache size in bytes
  uint numComputeRings_;         //!< 0 - disabled, 1 , 2,.. - the number of compute rings
  uint numDeviceEvents_;         //!< The number of device events
  uint numWaitEvents_;           //!< The number of wait events for device enqueue
  uint hostMemDirectAccess_;     //!< Enables direct access to the host memory
  uint numScratchWavesPerCu_;    //!< Maximum number of waves when scratch is enabled
  size_t xferBufSize_;           //!< Transfer buffer size for image copy optimization
  size_t pinnedXferSize_;        //!< Pinned buffer size for transfer
  size_t pinnedMinXferSize_;     //!< Minimal buffer size for pinned transfer
  size_t cpDmaCopySizeMax_;      //!< Threshold for CP DMA path in copy
  size_t resourceCacheSize_;     //!< Resource cache size in MB
  size_t numMemDependencies_;    //!< The array size for memory dependencies tracking
  uint64_t maxAllocSize_;        //!< Maximum single allocation size
  uint rgpSqttDispCount_;        //!< The number of dispatches captured in SQTT
  uint maxCmdBuffers_;           //!< Maximum number of command buffers allocated per queue
  uint mallPolicy_;              //!< 0 - default, 1 - always bypass, 2 - always put

  uint64_t subAllocationMinSize_;    //!< Minimum size allowed for suballocations
  uint64_t subAllocationMaxSize_;    //!< Maximum size allowed with suballocations
  uint64_t subAllocationChunkSize_;  //!< Chunk size for suballocaitons

  amd::LibrarySelector libSelector_;  //!< Select linking libraries for compiler

  size_t prepinnedMinSize_;  //!< minimal memory size for prepinned transfer
  uint32_t limit_blit_wg_;   //!< The number of workgroups for blit execution

  //! Default constructor
  Settings();

  //! Creates settings
  bool create(const Pal::DeviceProperties& palProp,       //!< PAL  device properties
              const Pal::GpuMemoryHeapProperties* heaps,  //!< PAL heap settings
              const Pal::WorkStationCaps& wscaps,         //!< PAL  workstation settings
              const amd::Isa& isa,                        //!< XNACK is enabled on this device
              bool reportAsOCL12Device = false            //!< Report As OpenCL1.2 Device
  );

 private:
  //! Disable copy constructor
  Settings(const Settings&);

  //! Disable assignment
  Settings& operator=(const Settings&);

  //! Overrides current settings based on registry/environment
  void override();

  using KernelArgImpl = device::Settings::KernelArgImpl;
};

/*@}*/  // namespace amd::pal
}  // namespace amd::pal
