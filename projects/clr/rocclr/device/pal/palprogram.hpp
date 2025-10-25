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

#include "device/pal/palkernel.hpp"
#include "amd_hsa_loader.hpp"

namespace amd {
namespace option {
class Options;
}  // namespace option
namespace hsa {
namespace loader {
class Loader;
class Executable;
class Context;
}  // namespace loader
}  // namespace hsa
}  // namespace amd

//! \namespace amd::pal PAL Device Implementation
namespace amd::pal {

/*! \addtogroup pal PAL Device Implementation
 *  @{
 */

using namespace amd::hsa::loader;
class HSAILProgram;

class Segment : public amd::HeapObject {
 public:
  Segment();
  ~Segment();

  //! Allocates a segment
  bool alloc(HSAILProgram& prog, amdgpu_hsa_elf_segment_t segment, size_t size, size_t align,
             bool zero);

  //! Copies data from host to the segment
  void copy(size_t offset, const void* src, size_t size);

  //! Segment freeze
  bool freeze(bool destroySysmem);

  //! Returns address for GPU access in the segment
  uint64_t gpuAddress(size_t offset) const { return gpuAccess_->vmAddress() + offset; }

  bool gpuAddressOffset(uint64_t offAddr, size_t* offset);

  //! Returns address for CPU access in the segment
  void* cpuAddress(size_t offset) const {
    return ((cpuAccess_ != nullptr) ? cpuAccess_->data() : cpuMem_) + offset;
  }

  void DestroyCpuAccess();

 private:
  Memory* gpuAccess_;  //!< GPU memory for segment access
  Memory* cpuAccess_;  //!< CPU memory for segment (backing store)
  address cpuMem_;     //!< CPU memory for segment without GPU direct access (backing store)
};

class PALHSALoaderContext final : public hsa::loader::Context {
 public:
  PALHSALoaderContext(HSAILProgram* program) : program_(program) {}

  virtual ~PALHSALoaderContext() {}

  hsa_isa_t IsaFromName(const char* name) override;

  bool IsaSupportedByAgent(hsa_agent_t agent, hsa_isa_t isa) override;

  void* SegmentAlloc(amdgpu_hsa_elf_segment_t segment, hsa_agent_t agent, size_t size, size_t align,
                     bool zero) override;

  bool SegmentCopy(amdgpu_hsa_elf_segment_t segment, hsa_agent_t agent, void* dst, size_t offset,
                   const void* src, size_t size) override;

  void SegmentFree(amdgpu_hsa_elf_segment_t segment, hsa_agent_t agent, void* seg,
                   size_t size = 0) override;

  void* SegmentAddress(amdgpu_hsa_elf_segment_t segment, hsa_agent_t agent, void* seg,
                       size_t offset) override;

  void* SegmentHostAddress(amdgpu_hsa_elf_segment_t segment, hsa_agent_t agent, void* seg,
                           size_t offset) override;

  bool SegmentFreeze(amdgpu_hsa_elf_segment_t segment, hsa_agent_t agent, void* seg,
                     size_t size) override;

  bool ImageExtensionSupported() override { return false; }

  hsa_status_t ImageCreate(hsa_agent_t agent, hsa_access_permission_t image_permission,
                           const hsa_ext_image_descriptor_t* image_descriptor,
                           const void* image_data, hsa_ext_image_t* image_handle) override {
    // not supported
    assert(false);
    return HSA_STATUS_ERROR;
  }

  hsa_status_t ImageDestroy(hsa_agent_t agent, hsa_ext_image_t image_handle) override {
    // not supported
    assert(false);
    return HSA_STATUS_ERROR;
  }

  hsa_status_t SamplerCreate(hsa_agent_t agent,
                             const hsa_ext_sampler_descriptor_t* sampler_descriptor,
                             hsa_ext_sampler_t* sampler_handle) override;

  //! All samplers are owned by HSAILProgram and are deleted in its destructor.
  hsa_status_t SamplerDestroy(hsa_agent_t agent, hsa_ext_sampler_t sampler_handle) override;

 private:
  PALHSALoaderContext(const PALHSALoaderContext& c);
  PALHSALoaderContext& operator=(const PALHSALoaderContext& c);

  pal::HSAILProgram* program_;
};

//! \class HSAIL program
class HSAILProgram : public device::Program {
  friend class ClBinary;

 public:
  //! Default constructor
  HSAILProgram(Device& device, amd::Program& owner);
  HSAILProgram(NullDevice& device, amd::Program& owner);
  //! Default destructor
  virtual ~HSAILProgram();

  void addGlobalStore(Memory* mem) { globalStores_.push_back(mem); }

  void setCodeObjects(Segment* seg, Memory* codeGpu, address codeCpu) {
    codeSegGpu_ = codeGpu;
    codeSegment_ = seg;
  }

  const std::vector<Memory*>& globalStores() const { return globalStores_; }

  //! Return a typecasted PAL null device.
  pal::NullDevice& palNullDevice() {
    return const_cast<pal::NullDevice&>(static_cast<const pal::NullDevice&>(device()));
  }

  //! Return a typecasted PAL device. The device must not be the NullDevice.
  pal::Device& palDevice() {
    return const_cast<pal::Device&>(static_cast<const pal::Device&>(device()));
  }

  //! Returns GPU kernel table
  const Memory* kernelTable() const { return kernels_; }

  //! Adds all kernels to the mem handle lists
  void fillResListWithKernels(VirtualGPU& gpu) const;

  //! Returns the maximum number of scratch regs used in the program
  uint maxScratchRegs() const { return maxScratchRegs_; }

  //! Returns the maximum number of VGPR(s) used in the program
  uint maxVgprs() const { return maxVgprs_; }

  //! Add internal static sampler
  void addSampler(Sampler* sampler) { staticSamplers_.push_back(sampler); }

  //! Returns TRUE if the program contains static samplers
  bool isStaticSampler() const { return (staticSamplers_.size() != 0); }

  //! Returns code segement on GPU
  const Memory& codeSegGpu() const { return *codeSegGpu_; }

  //! Returns CPU address for a kernel
  uint64_t findHostKernelAddress(uint64_t devAddr) const {
    return loader_->FindHostAddress(devAddr);
  }

  //! Get symbol by name
  amd::hsa::loader::Symbol* getSymbol(const char* symbol_name, const hsa_agent_t* agent) const {
    return executable_->GetSymbol(symbol_name, agent);
  }

  //! Returns API hash value of the program for RGP thread trace
  uint64_t ApiHash() const { return apiHash_; }

 protected:
  bool saveBinaryAndSetType(type_t type);

  virtual bool createBinary(amd::option::Options* options);

#if defined(WITH_COMPILER_LIB)
  virtual const aclTargetInfo& info();
#endif
  virtual bool createKernels(void* binary, size_t binSize, bool useUniformWorkGroupSize,
                             bool internalKernel) override;

  virtual bool setKernels(void* binary, size_t binSize,
                          amd::Os::FileDesc fdesc = amd::Os::FDescInit(), size_t foffset = 0,
                          std::string uri = std::string()) override;

  //! Destroys CPU allocations in the code segment
  void DestroySegmentCpuAccess() const {
    if (codeSegment_ != nullptr) {
      codeSegment_->DestroyCpuAccess();
    }
  }

  virtual bool defineGlobalVar(const char* name, void* dptr);
  virtual bool createGlobalVarObj(amd::Memory** amd_mem_obj, void** dptr, size_t* bytes,
                                  const char* globalName) const;

 private:
  //! Disable default copy constructor
  HSAILProgram(const HSAILProgram&);

  //! Disable operator=
  HSAILProgram& operator=(const HSAILProgram&);

 protected:
  //! Allocate kernel table
  bool allocKernelTable();

  void* rawBinary_;                    //!< Pointer to the raw binary
  std::vector<Memory*> globalStores_;  //!< Global memory for the program
  Memory* kernels_;                    //!< Table with kernel object pointers
  Memory* codeSegGpu_;                 //!< GPU memory with code objects
  Segment* codeSegment_;               //!< Pointer to the code segment for this program
  uint maxScratchRegs_;                //!< Maximum number of scratch regs used
                                       //!< in the program by individual kernel
  uint maxVgprs_;                      //!< Maximum number of VGPR(s) used
                                       //!< in the program by individual kernel
  uint64_t apiHash_ = 0;               //!< API hash value for RGP thread trace

  std::list<Sampler*> staticSamplers_;  //!< List od internal static samplers

  amd::hsa::loader::Loader* loader_;          //!< Loader object
  amd::hsa::loader::Executable* executable_;  //!< Executable for HSA Loader
  PALHSALoaderContext loaderContext_;         //!< Context for HSA Loader
};

//! \class Lightning Compiler Program
class LightningProgram : public HSAILProgram {
 public:
  LightningProgram(NullDevice& device, amd::Program& owner) : HSAILProgram(device, owner) {
    isLC_ = true;
    isHIP_ = (owner.language() == amd::Program::HIP);
  }

  LightningProgram(Device& device, amd::Program& owner) : HSAILProgram(device, owner) {
    isLC_ = true;
    isHIP_ = (owner.language() == amd::Program::HIP);
  }
  virtual ~LightningProgram() {}
  uint64_t GetTrapHandlerAddress() const;

 protected:
  virtual bool createKernels(void* binary, size_t binSize, bool useUniformWorkGroupSize,
                             bool internalKernel) override;

  virtual bool setKernels(void* binary, size_t binSize,
                          amd::Os::FileDesc fdesc = amd::Os::FDescInit(), size_t foffset = 0,
                          std::string uri = std::string()) override;

  virtual bool createBinary(amd::option::Options* options) override;
};

/*@}*/  // namespace amd::pal
}  // namespace amd::pal
