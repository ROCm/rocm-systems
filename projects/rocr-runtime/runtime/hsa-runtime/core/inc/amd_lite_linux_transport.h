////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
////////////////////////////////////////////////////////////////////////////////

#ifndef HSA_RUNTIME_CORE_INC_AMD_LITE_LINUX_TRANSPORT_H_
#define HSA_RUNTIME_CORE_INC_AMD_LITE_LINUX_TRANSPORT_H_

#include <cstddef>
#include <cstdint>
#include <string>

#include "core/inc/amd_lite_direct_queue.h"

namespace rocr {
namespace AMD {
namespace lite {

struct LinuxLiteBuffer {
  uint64_t handle = 0;
  uint64_t size = 0;
  uint64_t bus_addr = 0;
  uint64_t gpu_addr = 0;
  uint64_t mmap_offset = 0;
  void* cpu = nullptr;
};

class LinuxAmdgpuLiteTransport final : public DirectQueuePlatform {
 public:
  explicit LinuxAmdgpuLiteTransport(std::string devnode = "/dev/amdgpu_lite0");
  ~LinuxAmdgpuLiteTransport() override;

  LinuxAmdgpuLiteTransport(const LinuxAmdgpuLiteTransport&) = delete;
  LinuxAmdgpuLiteTransport& operator=(const LinuxAmdgpuLiteTransport&) = delete;

  hsa_status_t Open();
  void Close();

  bool is_open() const { return fd_ >= 0; }
  uint16_t vendor_id() const { return vendor_id_; }
  uint16_t device_id() const { return device_id_; }
  uint64_t framebuffer_base() const { return framebuffer_base_; }
  uint64_t vram_size() const { return vram_size_; }
  uint64_t visible_vram_size() const { return visible_vram_size_; }

  hsa_status_t AllocGtt(uint64_t size, LinuxLiteBuffer* buffer) const;
  hsa_status_t FreeGtt(LinuxLiteBuffer* buffer) const;
  hsa_status_t MapGttToGpu(LinuxLiteBuffer* buffer,
                           uint64_t requested_gpu_va = 0) const;
  hsa_status_t UnmapGpu(const LinuxLiteBuffer& buffer) const;
  hsa_status_t AllocVram(uint64_t size, LinuxLiteBuffer* buffer) const;
  hsa_status_t FreeVram(LinuxLiteBuffer* buffer) const;

 private:
  hsa_status_t EnsureDoorbellAperture() const override;
  hsa_status_t ReadMmio32(uint32_t base, uint32_t reg,
                          uint32_t* value) const override;
  hsa_status_t WriteMmio32(uint32_t base, uint32_t reg,
                           uint32_t value) const override;
  hsa_status_t ZeroGpuMemory(uint64_t offset, uint64_t size) const override;
  hsa_status_t WriteGpuMemory32(uint64_t offset, uint32_t value) const override;
  void* GpuMemoryCpuPointer(uint64_t offset) const override;
  volatile uint64_t* DoorbellCpuPointer(uint32_t doorbell_index) const override;
  void SleepUs(uint32_t usec) const override;

  hsa_status_t MapBar(uint32_t bar_index, uint64_t bar_size,
                      void** cpu, uint64_t* mapped_size) const;
  void UnmapBars();

  std::string devnode_;
  int fd_ = -1;

  uint16_t vendor_id_ = 0;
  uint16_t device_id_ = 0;
  uint64_t framebuffer_base_ = 0;
  uint64_t vram_size_ = 0;
  uint64_t visible_vram_size_ = 0;
  uint32_t mmio_bar_index_ = 0;
  uint32_t vram_bar_index_ = 0;
  uint32_t doorbell_bar_index_ = 0;

  void* mmio_bar_ = nullptr;
  uint64_t mmio_bar_size_ = 0;
  void* doorbell_bar_ = nullptr;
  uint64_t doorbell_bar_size_ = 0;
  void* vram_bar_ = nullptr;
  uint64_t vram_bar_size_ = 0;
};

}  // namespace lite
}  // namespace AMD
}  // namespace rocr

#endif  // HSA_RUNTIME_CORE_INC_AMD_LITE_LINUX_TRANSPORT_H_
