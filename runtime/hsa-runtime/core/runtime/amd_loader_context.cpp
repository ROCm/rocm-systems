////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
// 
// Copyright (c) 2014-2020, Advanced Micro Devices, Inc. All rights reserved.
// 
// Developed by:
// 
//                 AMD Research and AMD HSA Software Development
// 
//                 Advanced Micro Devices, Inc.
// 
//                 www.amd.com
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal with the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
// 
//  - Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimers.
//  - Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimers in
//    the documentation and/or other materials provided with the distribution.
//  - Neither the names of Advanced Micro Devices, Inc,
//    nor the names of its contributors may be used to endorse or promote
//    products derived from this Software without specific prior written
//    permission.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
// OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
// ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS WITH THE SOFTWARE.
//
////////////////////////////////////////////////////////////////////////////////

#include "core/inc/amd_loader_context.hpp"

#include <algorithm>
#include <cassert>
#include <cstring>

#include "core/inc/amd_gpu_agent.h"
#include "core/inc/amd_memory_region.h"
#include "core/util/os.h"

#include <cstdlib>
#include <utility>
#include "core/inc/hsa_internal.h"
#include "core/util/utils.h"
#include "inc/hsa_ext_amd.h"

#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#else
#include <sys/mman.h>
#endif

namespace rocr {
namespace {

class SegmentMemory {
public:
  virtual ~SegmentMemory() {}
  virtual void* Address(size_t offset = 0) const = 0;
  virtual void* HostAddress(size_t offset = 0) const = 0;
  virtual bool Allocated() const = 0;
  virtual bool Allocate(size_t size, size_t align, bool zero) = 0;
  virtual bool Copy(size_t offset, const void *src, size_t size) = 0;
  virtual void Free() = 0;
  virtual bool Freeze() = 0;

protected:
  SegmentMemory() {}

private:
  SegmentMemory(const SegmentMemory&);
  SegmentMemory& operator=(const SegmentMemory&);
};

class MallocedMemory final: public SegmentMemory {
public:
  MallocedMemory(): SegmentMemory(), ptr_(nullptr), size_(0) {}
  ~MallocedMemory() {}

  void* Address(size_t offset = 0) const override
    { assert(this->Allocated()); return (char*)ptr_ + offset; }
  void* HostAddress(size_t offset = 0) const override
    { return this->Address(offset); }
  bool Allocated() const override
    { return nullptr != ptr_; }

  bool Allocate(size_t size, size_t align, bool zero) override;
  bool Copy(size_t offset, const void *src, size_t size) override;
  void Free() override;
  bool Freeze() override;

private:
  MallocedMemory(const MallocedMemory&);
  MallocedMemory& operator=(const MallocedMemory&);

  void *ptr_;
  size_t size_;
};

bool MallocedMemory::Allocate(size_t size, size_t align, bool zero)
{
  assert(!this->Allocated());
  assert(0 < size);
  assert(0 < align && 0 == (align & (align - 1)));
  ptr_ = _aligned_malloc(size, align);
  if (nullptr == ptr_) {
    return false;
  }
  if (HSA_STATUS_SUCCESS != HSA::hsa_memory_register(ptr_, size)) {
    _aligned_free(ptr_);
    ptr_ = nullptr;
    return false;
  }
  if (zero) {
    memset(ptr_, 0x0, size);
  }
  size_ = size;
  return true;
}

bool MallocedMemory::Copy(size_t offset, const void *src, size_t size)
{
  assert(this->Allocated());
  assert(nullptr != src);
  assert(0 < size);
  memcpy(this->Address(offset), src, size);
  return true;
}

void MallocedMemory::Free()
{
  assert(this->Allocated());
  HSA::hsa_memory_deregister(ptr_, size_);
  _aligned_free(ptr_);
  ptr_ = nullptr;
  size_ = 0;
}

bool MallocedMemory::Freeze()
{
  assert(this->Allocated());
  return true;
}

class MappedMemory final: public SegmentMemory {
public:
  MappedMemory(): SegmentMemory(), ptr_(nullptr), size_(0) {}
  ~MappedMemory() {}

  void* Address(size_t offset = 0) const override
    { assert(this->Allocated()); return (char*)ptr_ + offset; }
  void* HostAddress(size_t offset = 0) const override
    { return this->Address(offset); }
  bool Allocated() const override
    { return nullptr != ptr_; }

  bool Allocate(size_t size, size_t align, bool zero) override;
  bool Copy(size_t offset, const void *src, size_t size) override;
  void Free() override;
  bool Freeze() override;

private:
  MappedMemory(const MappedMemory&);
  MappedMemory& operator=(const MappedMemory&);

  void *ptr_;
  size_t size_;
};

bool MappedMemory::Allocate(size_t size, size_t align, bool zero)
{
  assert(!this->Allocated());
  assert(0 < size);
  assert(0 < align && 0 == (align & (align - 1)));
#if defined(_WIN32) || defined(_WIN64)
  ptr_ = (void*)VirtualAlloc(nullptr, size, MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE);
#else
  ptr_ =
    mmap(nullptr, size, PROT_EXEC | PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_NORESERVE | MAP_PRIVATE, -1, 0);
#endif // _WIN32 || _WIN64
  if (nullptr == ptr_) {
    return false;
  }
  assert(0 == ((uintptr_t)ptr_) % align);
  if (HSA_STATUS_SUCCESS != HSA::hsa_memory_register(ptr_, size)) {
#if defined(_WIN32) || defined(_WIN64)
    VirtualFree(ptr_, size, MEM_DECOMMIT);
    VirtualFree(ptr_, 0, MEM_RELEASE);
#else
    munmap(ptr_, size);
#endif // _WIN32 || _WIN64
    ptr_ = nullptr;
    return false;
  }
  if (zero) {
    memset(ptr_, 0x0, size);
  }
  size_ = size;
  return true;
}

bool MappedMemory::Copy(size_t offset, const void *src, size_t size)
{
  assert(this->Allocated());
  assert(nullptr != src);
  assert(0 < size);
  memcpy(this->Address(offset), src, size);
  return true;
}

void MappedMemory::Free()
{
  assert(this->Allocated());
  HSA::hsa_memory_deregister(ptr_, size_);
#if defined(_WIN32) || defined(_WIN64)
  VirtualFree(ptr_, size_, MEM_DECOMMIT);
  VirtualFree(ptr_, 0, MEM_RELEASE);
#else
  munmap(ptr_, size_);
#endif // _WIN32 || _WIN64
  ptr_ = nullptr;
  size_ = 0;
}

bool MappedMemory::Freeze()
{
  assert(this->Allocated());
  return true;
}

class RegionMemory final: public SegmentMemory {
public:
 static const core::MemoryRegion* AgentLocal(hsa_agent_t agent, bool is_code);
 static const core::MemoryRegion* System(bool is_code);

 RegionMemory(const core::MemoryRegion* region, bool is_code)
     : SegmentMemory(),
       region_(region),
       ptr_(nullptr),
       host_ptr_(nullptr),
       size_(0),
       is_code_(is_code) {}
 ~RegionMemory() {}

 void* Address(size_t offset = 0) const override {
   assert(this->Allocated());
   return (char*)ptr_ + offset; }
  void* HostAddress(size_t offset = 0) const override
    { assert(this->Allocated()); return (char*)host_ptr_ + offset; }
  bool Allocated() const override
    { return nullptr != ptr_; }

  bool Allocate(size_t size, size_t align, bool zero) override;
  bool Copy(size_t offset, const void *src, size_t size) override;
  void Free() override;
  bool Freeze() override;

private:
  RegionMemory(const RegionMemory&);
  RegionMemory& operator=(const RegionMemory&);

  const core::MemoryRegion* region_;
  void *ptr_;
  void *host_ptr_;
  size_t size_;
  bool is_code_;
};

const core::MemoryRegion* RegionMemory::AgentLocal(hsa_agent_t agent, bool is_code) {
  AMD::GpuAgent *amd_agent = (AMD::GpuAgent*)core::Agent::Convert(agent);
  assert(amd_agent->device_type() == core::Agent::kAmdGpuDevice && "Invalid agent type.");
  auto agent_local_region =
      std::find_if(amd_agent->regions().begin(), amd_agent->regions().end(),
                   [&](const core::MemoryRegion* region) {
                     const AMD::MemoryRegion* amd_region = (const AMD::MemoryRegion*)region;
                     return amd_region->IsLocalMemory() && (!amd_region->fine_grain());
                   });
  return agent_local_region == amd_agent->regions().end() ? nullptr : *agent_local_region;
}

const core::MemoryRegion* RegionMemory::System(bool is_code) {
  if (is_code)
    return core::Runtime::runtime_singleton_->system_regions_coarse()[0];
  else
    return core::Runtime::runtime_singleton_->system_regions_fine()[0];
}

bool RegionMemory::Allocate(size_t size, size_t align, bool zero) {
  assert(!this->Allocated());
  assert(0 < size);
  assert(0 < align && 0 == (align & (align - 1)));
  core::MemoryRegion::AllocateFlags flags = core::MemoryRegion::AllocateNoFlags;
  if (is_code_) flags = core::MemoryRegion::AllocateExecutable;
  if (HSA_STATUS_SUCCESS !=
      core::Runtime::runtime_singleton_->AllocateMemory(region_, size, flags, &ptr_)) {
    ptr_ = nullptr;
    return false;
  }
  assert(0 == ((uintptr_t)ptr_) % align);
  if (HSA_STATUS_SUCCESS !=
      core::Runtime::runtime_singleton_->AllocateMemory(
          RegionMemory::System(false), size, core::MemoryRegion::AllocateNoFlags, &host_ptr_)) {
    HSA::hsa_memory_free(ptr_);
    ptr_ = nullptr;
    host_ptr_ = nullptr;
    return false;
  }
  if (zero) {
    memset(host_ptr_, 0x0, size);
  }
  size_ = size;
  return true;
}

bool RegionMemory::Copy(size_t offset, const void* src, size_t size) {
  assert(this->Allocated() && nullptr != host_ptr_);
  assert(nullptr != src);
  assert(0 < size);
  memcpy((char*)host_ptr_ + offset, src, size);
  return true;
}

void RegionMemory::Free()
{
  assert(this->Allocated());
  HSA::hsa_memory_free(ptr_);
  if (nullptr != host_ptr_) {
    HSA::hsa_memory_free(host_ptr_);
  }
  ptr_ = nullptr;
  host_ptr_ = nullptr;
  size_ = 0;
}

bool RegionMemory::Freeze() {
  assert(this->Allocated() && nullptr != host_ptr_);

  core::Agent* agent = region_->owner();

  const size_t& code_object_dmacopy_size =
    core::Runtime::runtime_singleton_->flag().co_dmacopy_size();

  const bool isGpuDevice = (agent->device_type() == core::Agent::kAmdGpuDevice);
  const bool isLargeBarDisabled = isGpuDevice && !reinterpret_cast<AMD::GpuAgent*>(agent)->LargeBarEnabled();
  const bool shouldDmaCopy = isGpuDevice && (isLargeBarDisabled || size_ > code_object_dmacopy_size);

  if (shouldDmaCopy) {
      if (HSA_STATUS_SUCCESS != agent->DmaCopy(ptr_, host_ptr_, size_)) return false;
  } else {
      memcpy(ptr_, host_ptr_, size_);
      if (is_code_ && isGpuDevice)
        reinterpret_cast<AMD::GpuAgent*>(agent)->PcieWcFlush(ptr_, size_);
  }

  // Invalidate agent caches if needed
  if (is_code_ && isGpuDevice)
      reinterpret_cast<AMD::GpuAgent*>(agent)->InvalidateCodeCaches(ptr_, size_);

  return true;
}

}  // namespace anonymous
namespace amd {

hsa_isa_t LoaderContext::IsaFromName(const char *name) {
  assert(name);

  hsa_status_t hsa_status = HSA_STATUS_SUCCESS;
  hsa_isa_t isa_handle;
  isa_handle.handle = 0;

  hsa_status = HSA::hsa_isa_from_name(name, &isa_handle);
  if (HSA_STATUS_SUCCESS != hsa_status) {
    isa_handle.handle = 0;
    return isa_handle;
  }

  return isa_handle;
}

bool LoaderContext::IsaSupportedByAgent(hsa_agent_t agent,
                                        hsa_isa_t code_object_isa,
                                        unsigned codeGenericVersion) {
  struct callBackData {
    std::pair<hsa_isa_t, bool> comparison_data;
    const unsigned int codeGenericV;
  } cbData = {{code_object_isa, false}, codeGenericVersion};

  auto IsIsaEquivalent = [](hsa_isa_t agent_isa_h, void *data) {
    assert(data);

    struct callBackData *inOutCB = reinterpret_cast<decltype(&cbData)>(data);

    std::pair<hsa_isa_t, bool> *data_pair = &inOutCB->comparison_data;
    const unsigned int codeGenericV = inOutCB->codeGenericV;

    assert(data_pair);
    assert(!data_pair->second);

    const core::Isa *agent_isa = core::Isa::Object(agent_isa_h);
    assert(agent_isa);
    const core::Isa *code_object_isa = core::Isa::Object(data_pair->first);
    assert(code_object_isa);

    data_pair->second = core::Isa::IsCompatible(*code_object_isa, *agent_isa, codeGenericV);
    return data_pair->second ? HSA_STATUS_INFO_BREAK : HSA_STATUS_SUCCESS;
  };

  hsa_status_t status = HSA::hsa_agent_iterate_isas(agent, IsIsaEquivalent, &cbData);
  if (status != HSA_STATUS_SUCCESS && status != HSA_STATUS_INFO_BREAK) {
    return false;
  }
  return cbData.comparison_data.second;
}

void* LoaderContext::SegmentAlloc(amdgpu_hsa_elf_segment_t segment,
                                  hsa_agent_t agent,
                                  size_t size,
                                  size_t align,
                                  bool zero)
{
  assert(0 < size);
  assert(0 < align && 0 == (align & (align - 1)));

  hsa_profile_t agent_profile;
  if (HSA_STATUS_SUCCESS !=
      HSA::hsa_agent_get_info(agent, HSA_AGENT_INFO_PROFILE, &agent_profile)) {
    return nullptr;
  }

  SegmentMemory *mem = nullptr;
  switch (segment) {
  case AMDGPU_HSA_SEGMENT_GLOBAL_AGENT:
  case AMDGPU_HSA_SEGMENT_READONLY_AGENT: {
    switch (agent_profile) {
    case HSA_PROFILE_BASE:
      mem = new (std::nothrow) RegionMemory(RegionMemory::AgentLocal(agent, false), false);
      break;
    case HSA_PROFILE_FULL:
      mem = new (std::nothrow) RegionMemory(RegionMemory::System(false), false);
      break;
    default:
      assert(false);
    }
    break;
  }
  case AMDGPU_HSA_SEGMENT_GLOBAL_PROGRAM: {
    mem = new (std::nothrow) RegionMemory(RegionMemory::System(false), false);
    break;
  }
  case AMDGPU_HSA_SEGMENT_CODE_AGENT: {
    switch (agent_profile) {
    case HSA_PROFILE_BASE:
      mem = new (std::nothrow) RegionMemory(RegionMemory::AgentLocal(agent, true), true);
      break;
    case HSA_PROFILE_FULL:
      mem = new (std::nothrow) MappedMemory();
      break;
    default:
      assert(false);
    }
    break;
  }
  default:
    assert(false);
  }

  if (nullptr == mem) {
    return nullptr;
  }

  if (!mem->Allocate(size, align, zero)) {
    delete mem;
    return nullptr;
  }

  return mem;
}

bool LoaderContext::SegmentCopy(amdgpu_hsa_elf_segment_t segment, // not used.
                                hsa_agent_t agent,                // not used.
                                void* dst,
                                size_t offset,
                                const void* src,
                                size_t size)
{
  assert(nullptr != dst);
  return ((SegmentMemory*)dst)->Copy(offset, src, size);
}

void LoaderContext::SegmentFree(amdgpu_hsa_elf_segment_t segment, // not used.
                                hsa_agent_t agent,                // not used.
                                void* seg,
                                size_t size)                      // not used.
{
  assert(nullptr != seg);
  SegmentMemory *mem = (SegmentMemory*)seg;
  mem->Free();
  delete mem;
  mem = nullptr;
}

void* LoaderContext::SegmentAddress(amdgpu_hsa_elf_segment_t segment, // not used.
                                    hsa_agent_t agent,                // not used.
                                    void* seg,
                                    size_t offset)
{
  assert(nullptr != seg);
  return ((SegmentMemory*)seg)->Address(offset);
}

void* LoaderContext::SegmentHostAddress(amdgpu_hsa_elf_segment_t segment, // not used.
                                        hsa_agent_t agent,                // not used.
                                        void* seg,
                                        size_t offset)
{
  assert(nullptr != seg);
  return ((SegmentMemory*)seg)->HostAddress(offset);
}

bool LoaderContext::SegmentFreeze(amdgpu_hsa_elf_segment_t segment, // not used.
                                  hsa_agent_t agent,                // not used.
                                  void* seg,
                                  size_t size)                      // not used.
{
  assert(nullptr != seg);
  return ((SegmentMemory*)seg)->Freeze();
}

bool LoaderContext::ImageExtensionSupported() {
  hsa_status_t hsa_status = HSA_STATUS_SUCCESS;
  bool result = false;

  hsa_status =
      HSA::hsa_system_extension_supported(HSA_EXTENSION_IMAGES, 1, 0, &result);
  if (HSA_STATUS_SUCCESS != hsa_status) {
    return false;
  }

  return result;
}

hsa_status_t LoaderContext::ImageCreate(
    hsa_agent_t agent, hsa_access_permission_t image_permission,
    const hsa_ext_image_descriptor_t *image_descriptor, const void *image_data,
    hsa_ext_image_t *image_handle) {
  assert(agent.handle);
  assert(image_descriptor);
  assert(image_data);
  assert(image_handle);

  assert(ImageExtensionSupported());

  return hsa_ext_image_create(agent, image_descriptor, image_data,
                              image_permission, image_handle);
}

hsa_status_t LoaderContext::ImageDestroy(hsa_agent_t agent,
                                         hsa_ext_image_t image_handle) {
  assert(agent.handle);
  assert(image_handle.handle);

  assert(ImageExtensionSupported());

  return hsa_ext_image_destroy(agent, image_handle);
}

hsa_status_t LoaderContext::SamplerCreate(
    hsa_agent_t agent, const hsa_ext_sampler_descriptor_t *sampler_descriptor,
    hsa_ext_sampler_t *sampler_handle) {
  assert(agent.handle);
  assert(sampler_descriptor);
  assert(sampler_handle);

  assert(ImageExtensionSupported());

  return hsa_ext_sampler_create(agent, sampler_descriptor, sampler_handle);
}

hsa_status_t LoaderContext::SamplerDestroy(hsa_agent_t agent,
                                           hsa_ext_sampler_t sampler_handle) {
  assert(agent.handle);
  assert(sampler_handle.handle);

  assert(ImageExtensionSupported());

  return hsa_ext_sampler_destroy(agent, sampler_handle);
}

}  // namespace amd
}  // namespace rocr
