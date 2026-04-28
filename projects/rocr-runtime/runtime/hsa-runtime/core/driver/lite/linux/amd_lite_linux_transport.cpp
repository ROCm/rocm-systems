////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
////////////////////////////////////////////////////////////////////////////////

#include "core/inc/amd_lite_linux_transport.h"

#include <errno.h>
#include <fcntl.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <utility>

#include "core/driver/lite/linux/amdgpu_lite_uapi.h"

namespace rocr {
namespace AMD {
namespace lite {
namespace {

constexpr uint32_t kGcBase0 = 0x1260;
constexpr uint32_t kNbioBase2 = 0xD20;

constexpr uint32_t regRCC_DEV0_EPF0_RCC_DOORBELL_APER_EN = 0x00c0;
constexpr uint32_t regGDC_S2A0_S2A_DOORBELL_ENTRY_0_CTRL = 0x01cb;
constexpr uint32_t regGDC_S2A0_S2A_DOORBELL_ENTRY_3_CTRL = 0x01ce;
constexpr uint32_t regCP_MEC_DOORBELL_RANGE_LOWER = 0x1dfc;
constexpr uint32_t regCP_MEC_DOORBELL_RANGE_UPPER = 0x1dfd;

bool RangeFits(uint64_t offset, uint64_t size, uint64_t limit) {
  return offset <= limit && size <= limit - offset;
}

hsa_status_t PosixStatus(int err) {
  return err == ENOMEM ? HSA_STATUS_ERROR_OUT_OF_RESOURCES : HSA_STATUS_ERROR;
}

int RetryIoctl(int fd, unsigned long request, void* arg) {
  int ret = -1;
  do {
    ret = ioctl(fd, request, arg);
  } while (ret == -1 && (errno == EINTR || errno == EAGAIN));
  return ret;
}

}  // namespace

LinuxAmdgpuLiteTransport::LinuxAmdgpuLiteTransport(std::string devnode)
    : devnode_(std::move(devnode)) {}

LinuxAmdgpuLiteTransport::~LinuxAmdgpuLiteTransport() { Close(); }

hsa_status_t LinuxAmdgpuLiteTransport::Open() {
  if (fd_ >= 0) return HSA_STATUS_SUCCESS;

  int fd = open(devnode_.c_str(), O_RDWR | O_CLOEXEC);
  if (fd < 0) return PosixStatus(errno);
  fd_ = fd;

  amdgpu_lite_get_info info = {};
  if (RetryIoctl(fd_, AMDGPU_LITE_IOC_GET_INFO, &info) != 0) {
    const hsa_status_t status = PosixStatus(errno);
    Close();
    return status;
  }

  if (info.mmio_bar_index >= info.num_bars ||
      info.doorbell_bar_index >= info.num_bars ||
      info.vram_bar_index >= info.num_bars) {
    Close();
    return HSA_STATUS_ERROR;
  }

  vendor_id_ = info.vendor_id;
  device_id_ = info.device_id;
  vram_size_ = info.vram_size;
  visible_vram_size_ = info.visible_vram_size;
  mmio_bar_index_ = info.mmio_bar_index;
  vram_bar_index_ = info.vram_bar_index;
  doorbell_bar_index_ = info.doorbell_bar_index;
  framebuffer_base_ = info.bars[info.vram_bar_index].phys_addr;

  hsa_status_t status =
      MapBar(mmio_bar_index_, info.bars[mmio_bar_index_].size, &mmio_bar_,
             &mmio_bar_size_);
  if (status == HSA_STATUS_SUCCESS) {
    status = MapBar(doorbell_bar_index_, info.bars[doorbell_bar_index_].size,
                    &doorbell_bar_, &doorbell_bar_size_);
  }
  if (status == HSA_STATUS_SUCCESS) {
    status = MapBar(vram_bar_index_, info.bars[vram_bar_index_].size,
                    &vram_bar_, &vram_bar_size_);
  }
  if (status != HSA_STATUS_SUCCESS) {
    Close();
    return status;
  }

  return HSA_STATUS_SUCCESS;
}

void LinuxAmdgpuLiteTransport::Close() {
  UnmapBars();
  if (fd_ >= 0) {
    close(fd_);
    fd_ = -1;
  }
  vendor_id_ = 0;
  device_id_ = 0;
  framebuffer_base_ = 0;
  vram_size_ = 0;
  visible_vram_size_ = 0;
}

hsa_status_t LinuxAmdgpuLiteTransport::AllocGtt(uint64_t size,
                                                LinuxLiteBuffer* buffer) const {
  if (fd_ < 0 || size == 0 || buffer == nullptr) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  amdgpu_lite_alloc_gtt args = {};
  args.size = size;
  if (RetryIoctl(fd_, AMDGPU_LITE_IOC_ALLOC_GTT, &args) != 0) {
    return PosixStatus(errno);
  }

  void* cpu = mmap(nullptr, args.size, PROT_READ | PROT_WRITE, MAP_SHARED,
                   fd_, static_cast<off_t>(args.mmap_offset));
  if (cpu == MAP_FAILED) {
    const hsa_status_t status = PosixStatus(errno);
    amdgpu_lite_free_gtt free_args = {};
    free_args.handle = args.handle;
    RetryIoctl(fd_, AMDGPU_LITE_IOC_FREE_GTT, &free_args);
    return status;
  }

  *buffer = {};
  buffer->handle = args.handle;
  buffer->size = args.size;
  buffer->bus_addr = args.bus_addr;
  buffer->mmap_offset = args.mmap_offset;
  buffer->cpu = cpu;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t LinuxAmdgpuLiteTransport::FreeGtt(LinuxLiteBuffer* buffer) const {
  if (fd_ < 0 || buffer == nullptr) return HSA_STATUS_ERROR_INVALID_ARGUMENT;

  hsa_status_t status = HSA_STATUS_SUCCESS;
  if (buffer->cpu != nullptr) {
    if (munmap(buffer->cpu, buffer->size) != 0) status = PosixStatus(errno);
  }
  if (buffer->handle != 0) {
    amdgpu_lite_free_gtt args = {};
    args.handle = buffer->handle;
    if (RetryIoctl(fd_, AMDGPU_LITE_IOC_FREE_GTT, &args) != 0 &&
        status == HSA_STATUS_SUCCESS) {
      status = PosixStatus(errno);
    }
  }

  *buffer = {};
  return status;
}

hsa_status_t LinuxAmdgpuLiteTransport::MapGttToGpu(
    LinuxLiteBuffer* buffer, uint64_t requested_gpu_va) const {
  if (fd_ < 0 || buffer == nullptr || buffer->handle == 0 || buffer->size == 0) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  amdgpu_lite_map_gpu args = {};
  args.handle = buffer->handle;
  args.gpu_va = requested_gpu_va;
  args.size = buffer->size;
  if (RetryIoctl(fd_, AMDGPU_LITE_IOC_MAP_GPU, &args) != 0) {
    return PosixStatus(errno);
  }

  buffer->gpu_addr = args.mapped_gpu_va;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t LinuxAmdgpuLiteTransport::UnmapGpu(
    const LinuxLiteBuffer& buffer) const {
  if (fd_ < 0 || buffer.gpu_addr == 0 || buffer.size == 0) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  amdgpu_lite_unmap_gpu args = {};
  args.gpu_va = buffer.gpu_addr;
  args.size = buffer.size;
  return RetryIoctl(fd_, AMDGPU_LITE_IOC_UNMAP_GPU, &args) == 0
             ? HSA_STATUS_SUCCESS
             : PosixStatus(errno);
}

hsa_status_t LinuxAmdgpuLiteTransport::AllocVram(
    uint64_t size, LinuxLiteBuffer* buffer) const {
  if (fd_ < 0 || size == 0 || buffer == nullptr) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  amdgpu_lite_alloc_vram args = {};
  args.size = size;
  if (RetryIoctl(fd_, AMDGPU_LITE_IOC_ALLOC_VRAM, &args) != 0) {
    return PosixStatus(errno);
  }

  void* cpu = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED,
                   fd_, static_cast<off_t>(args.mmap_offset));
  if (cpu == MAP_FAILED) {
    const hsa_status_t status = PosixStatus(errno);
    amdgpu_lite_free_vram free_args = {};
    free_args.handle = args.handle;
    RetryIoctl(fd_, AMDGPU_LITE_IOC_FREE_VRAM, &free_args);
    return status;
  }

  *buffer = {};
  buffer->handle = args.handle;
  buffer->size = size;
  buffer->gpu_addr = args.gpu_addr;
  buffer->mmap_offset = args.mmap_offset;
  buffer->cpu = cpu;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t LinuxAmdgpuLiteTransport::FreeVram(LinuxLiteBuffer* buffer) const {
  if (fd_ < 0 || buffer == nullptr) return HSA_STATUS_ERROR_INVALID_ARGUMENT;

  hsa_status_t status = HSA_STATUS_SUCCESS;
  if (buffer->cpu != nullptr) {
    if (munmap(buffer->cpu, buffer->size) != 0) status = PosixStatus(errno);
  }
  if (buffer->handle != 0) {
    amdgpu_lite_free_vram args = {};
    args.handle = buffer->handle;
    if (RetryIoctl(fd_, AMDGPU_LITE_IOC_FREE_VRAM, &args) != 0 &&
        status == HSA_STATUS_SUCCESS) {
      status = PosixStatus(errno);
    }
  }

  *buffer = {};
  return status;
}

hsa_status_t LinuxAmdgpuLiteTransport::EnsureDoorbellAperture() const {
  hsa_status_t status =
      WriteMmio32(kNbioBase2, regRCC_DEV0_EPF0_RCC_DOORBELL_APER_EN, 1);
  if (status != HSA_STATUS_SUCCESS) return status;
  status = WriteMmio32(kNbioBase2, regGDC_S2A0_S2A_DOORBELL_ENTRY_0_CTRL,
                       (1u << 0) | (3u << 1) | (3u << 28));
  if (status != HSA_STATUS_SUCCESS) return status;
  status = WriteMmio32(kNbioBase2, regGDC_S2A0_S2A_DOORBELL_ENTRY_3_CTRL,
                       (1u << 0) | (6u << 1) | (3u << 28));
  if (status != HSA_STATUS_SUCCESS) return status;
  status = WriteMmio32(kGcBase0, regCP_MEC_DOORBELL_RANGE_LOWER, 0);
  if (status != HSA_STATUS_SUCCESS) return status;
  return WriteMmio32(kGcBase0, regCP_MEC_DOORBELL_RANGE_UPPER,
                     (0x8Au * 2u) << 2);
}

hsa_status_t LinuxAmdgpuLiteTransport::ReadMmio32(uint32_t base, uint32_t reg,
                                                  uint32_t* value) const {
  if (mmio_bar_ == nullptr || value == nullptr) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
  const uint64_t byte_offset = (static_cast<uint64_t>(base) + reg) *
                               sizeof(uint32_t);
  if (!RangeFits(byte_offset, sizeof(uint32_t), mmio_bar_size_)) {
    return HSA_STATUS_ERROR_INVALID_ALLOCATION;
  }

  auto* ptr = reinterpret_cast<volatile uint32_t*>(
      static_cast<char*>(mmio_bar_) + byte_offset);
  *value = *ptr;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t LinuxAmdgpuLiteTransport::WriteMmio32(uint32_t base, uint32_t reg,
                                                   uint32_t value) const {
  if (mmio_bar_ == nullptr) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  const uint64_t byte_offset = (static_cast<uint64_t>(base) + reg) *
                               sizeof(uint32_t);
  if (!RangeFits(byte_offset, sizeof(uint32_t), mmio_bar_size_)) {
    return HSA_STATUS_ERROR_INVALID_ALLOCATION;
  }

  auto* ptr = reinterpret_cast<volatile uint32_t*>(
      static_cast<char*>(mmio_bar_) + byte_offset);
  *ptr = value;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t LinuxAmdgpuLiteTransport::ZeroGpuMemory(uint64_t offset,
                                                     uint64_t size) const {
  for (uint64_t i = 0; i < size; i += sizeof(uint32_t)) {
    hsa_status_t status = WriteGpuMemory32(offset + i, 0);
    if (status != HSA_STATUS_SUCCESS) return status;
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t LinuxAmdgpuLiteTransport::WriteGpuMemory32(uint64_t offset,
                                                        uint32_t value) const {
  if (vram_bar_ == nullptr ||
      !RangeFits(offset, sizeof(uint32_t), vram_bar_size_)) {
    return HSA_STATUS_ERROR_INVALID_ALLOCATION;
  }

  auto* ptr = reinterpret_cast<volatile uint32_t*>(
      static_cast<char*>(vram_bar_) + offset);
  *ptr = value;
  return HSA_STATUS_SUCCESS;
}

void* LinuxAmdgpuLiteTransport::GpuMemoryCpuPointer(uint64_t offset) const {
  if (vram_bar_ == nullptr || offset >= vram_bar_size_) return nullptr;
  return static_cast<char*>(vram_bar_) + offset;
}

volatile uint64_t* LinuxAmdgpuLiteTransport::DoorbellCpuPointer(
    uint32_t doorbell_index) const {
  const uint64_t byte_offset =
      static_cast<uint64_t>(doorbell_index) * sizeof(uint32_t);
  if (doorbell_bar_ == nullptr ||
      !RangeFits(byte_offset, sizeof(uint64_t), doorbell_bar_size_)) {
    return nullptr;
  }
  return reinterpret_cast<volatile uint64_t*>(
      static_cast<char*>(doorbell_bar_) + byte_offset);
}

void LinuxAmdgpuLiteTransport::SleepUs(uint32_t usec) const { usleep(usec); }

hsa_status_t LinuxAmdgpuLiteTransport::MapBar(uint32_t bar_index,
                                              uint64_t bar_size, void** cpu,
                                              uint64_t* mapped_size) const {
  if (fd_ < 0 || cpu == nullptr || mapped_size == nullptr || bar_size == 0) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  amdgpu_lite_map_bar args = {};
  args.bar_index = bar_index;
  args.offset = 0;
  args.size = 0;
  if (RetryIoctl(fd_, AMDGPU_LITE_IOC_MAP_BAR, &args) != 0) {
    return PosixStatus(errno);
  }

  void* ptr = mmap(nullptr, bar_size, PROT_READ | PROT_WRITE, MAP_SHARED,
                   fd_, static_cast<off_t>(args.mmap_offset));
  if (ptr == MAP_FAILED) return PosixStatus(errno);

  *cpu = ptr;
  *mapped_size = bar_size;
  return HSA_STATUS_SUCCESS;
}

void LinuxAmdgpuLiteTransport::UnmapBars() {
  if (vram_bar_ != nullptr) {
    munmap(vram_bar_, vram_bar_size_);
    vram_bar_ = nullptr;
    vram_bar_size_ = 0;
  }
  if (doorbell_bar_ != nullptr) {
    munmap(doorbell_bar_, doorbell_bar_size_);
    doorbell_bar_ = nullptr;
    doorbell_bar_size_ = 0;
  }
  if (mmio_bar_ != nullptr) {
    munmap(mmio_bar_, mmio_bar_size_);
    mmio_bar_ = nullptr;
    mmio_bar_size_ = 0;
  }
}

}  // namespace lite
}  // namespace AMD
}  // namespace rocr
