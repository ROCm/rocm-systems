/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "device/device.hpp"
#include "utils/macros.hpp"
#include "platform/command.hpp"
#include "platform/program.hpp"
#include "platform/kernel.hpp"
#include "platform/sampler.hpp"
#include "device/pal/paldevice.hpp"
#include "device/pal/palvirtual.hpp"
#include "amd_hsa_kernel_code.h"
#include "AMDHSAKernelDescriptor.h"
#include "device/pal/palprintf.hpp"
#include "hsa.h"

namespace amd {
namespace hsa {
namespace loader {
class Symbol;
}  // namespace loader
namespace code {
namespace Kernel {
class Metadata;
}  // namespace Kernel
}  // namespace code
}  // namespace hsa
}  // namespace amd

//! \namespace amd::pal PAL Device Implementation
namespace amd::pal {

class VirtualGPU;
class Device;
class NullDevice;
class Program;

/*! \addtogroup pal PAL Device Implementation
 *  @{
 */
class Kernel : public device::Kernel {
 public:
  Kernel(const std::string& name, pal::Program* prog, bool internalKernel);

  virtual ~Kernel();

  //! Initializes the metadata required for this kernel,
  bool init();

  //! Setup after code object loading
  bool postLoad();

  //! Returns PAL, possibly null, device object, associated with this kernel.
  const NullDevice& palNullDevice() const { return reinterpret_cast<const NullDevice&>(dev_); }

  //! Returns PAL device object, associated with this kernel which must not be the null device.
  const Device& palDevice() const {
    assert(dev_.isOnline());
    return reinterpret_cast<const Device&>(dev_);
  }

  //! Returns HSA program associated with this kernel
  const pal::Program& prog() const;

  //! Returns LDS size used in this kernel
  uint32_t ldsSize() const { return WorkgroupGroupSegmentByteSize(); }

  //! Returns pointer on CPU to AQL kernel descriptor info
  const llvm::amdhsa::kernel_descriptor_t* cpuAqlKd() const { return &akd_; }

  //! Returns memory object with AQL code
  uint64_t gpuAqlCode() const { return code_; }

  //! Returns size of AQL code
  size_t aqlCodeSize() const { return codeSize_; }

  //! Returns the size of argument buffer
  size_t argsBufferSize() const { return kernargSegmentByteSize_; }

  //! Returns spill reg size per workitem
  uint32_t spillSegSize() const { return workGroupInfo_.privateMemSize_; }

  //! Returns AQL packet in CPU memory
  //! if the kernel arguments were successfully loaded, otherwise NULL
  std::pair<hsa_kernel_dispatch_packet_t* /* packet address */, uint64_t /* packet id */>
  loadArguments(VirtualGPU& gpu,                     //!< Running GPU context
                const amd::Kernel& kernel,           //!< AMD kernel object
                const amd::NDRangeContainer& sizes,  //!< NDrange container
                const_address params,                //!< Application arguments for the kernel
                size_t ldsAddress,                   //!< LDS address that includes all arguments.
                uint64_t vmDefQueue,                 //!< GPU VM default queue pointer
                uint64_t* vmParentWrap               //!< GPU VM parent aql wrap object
  ) const;

  //! Returns the kernel index in the program
  uint index() const { return index_; }

  //! Get the kernel descriptor and copy the code object from the program CPU segment
  bool setKernelDescriptor(amd::hsa::loader::Symbol* sym, llvm::amdhsa::kernel_descriptor_t* akd);

 private:
  //! Disable copy constructor
  Kernel(const pal::Kernel&);

  //! Disable operator=
  Kernel& operator=(const pal::Kernel&);

 protected:
  //! Get the kernel code and copy the code object from the program CPU segment
  bool setKernelCode(amd::hsa::loader::Symbol* sym, amd_kernel_code_t* akc);

  //! Set up the workgroup info based on the kernel metadata
  void setWorkGroupInfo(const uint32_t privateSegmentSize, const uint32_t groupSegmentSize,
                        const uint16_t numSGPRs, const uint16_t numVGPRs);

  llvm::amdhsa::kernel_descriptor_t akd_;  //!< AQL kernel descriptor on CPU, used by LC
  uint index_;                             //!< Kernel index in the program
  uint64_t code_;                          //!< GPU memory pointer to the kernel
  size_t codeSize_;                        //!< Size of ISA code
};

/*@}*/  // namespace amd::pal
}  // namespace amd::pal
