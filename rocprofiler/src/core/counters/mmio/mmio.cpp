/* Copyright (c) 2023 Advanced Micro Devices, Inc.

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

#include "mmio.h"
#include <cstdint>
#include "../../../utils/helper.h"
#include "../../../utils/libpci_helper.h"
#include "pcie_perfmon_registers_mi200.h"
#include "df_perfmon_registers_mi200.h"

namespace rocprofiler {

namespace mmio {

void PrintFunctionPhase(const char* function_name, int phase) {
#if defined(DEBUG_TRACE)
  if (phase == 0)
    std::clog << function_name << "() START" << std::endl;
  else
    std::clog << function_name << "() END" << std::endl;
#endif
}

void PrintRegisterData(uint32_t& index_value, uint32_t& data_value, const char* function_name,
                       int phase) {
#if defined(DEBUG_TRACE)
  if (phase == 0) {
    PrintFunctionPhase(function_name, phase);
    std::clog << "Old (index, data) : " << std::hex << index_value << " " << data_value
              << std::endl;
  } else {
    std::clog << "New (index, data) : " << std::hex << index_value << " " << data_value
              << std::endl;
    PrintFunctionPhase(function_name, phase);
  }
#endif
}

MMIO::MMIO(const HSAAgentInfo& info)
    : agent_info_(&info), pci_memory_(nullptr), type_(DEFAULT_MMAP) {
  const auto pci_domain = agent_info_->GetDeviceInfo().getPCIDomain();
  const auto pci_location_id = agent_info_->GetDeviceInfo().getPCILocationID();

  pci_device_ =
      GetPciAccessLibApi()->pci_device_find_by_slot(pci_domain, pci_location_id >> 8, pci_location_id & 0xFF, 0);
  if (!pci_device_ || GetPciAccessLibApi()->pci_device_probe(pci_device_)) fatal("failed to probe the GPU device\n");

  // Look for a region between 256KB and 4096KB, 32-bit, non IO, and non prefetchable.
  for (size_t region = 0; region < sizeof(pci_device::regions) / sizeof(pci_device::regions[0]);
       ++region){
    if (pci_device_->regions[region].is_64 == 0 &&
        pci_device_->regions[region].is_prefetchable == 0 &&
        pci_device_->regions[region].is_IO == 0 &&
        pci_device_->regions[region].size >= (256UL * 1024) &&
        pci_device_->regions[region].size <= (4096UL * 1024)) {
      pci_memory_size_ = pci_device_->regions[region].size;
      int err = GetPciAccessLibApi()->pci_device_map_range(pci_device_, pci_device_->regions[region].base_addr,
                                     pci_device_->regions[region].size, PCI_DEV_MAP_FLAG_WRITABLE,
                                     (void**)&pci_memory_);
      if (err) fatal("failed to map the registers. Error code: %d\n", err);
    }
  }

  if (pci_memory_ == nullptr) fatal("could not find the pci memory address\n");

  SetIndexDataRegisters(INDIRECT_REG_INDEX, INDIRECT_REG_DATA);
}

MMIO::~MMIO() {
  if (pci_memory_) {
    int err = GetPciAccessLibApi()->pci_device_unmap_range(pci_device_, pci_memory_, pci_memory_size_);
    if (err) warning("failed to unmap the pci memory. Error code: %d\n", err);
  }
}

bool MMIO::RegisterWriteAPI(uint32_t reg_offset, uint32_t value) {
  // access the mmap
  // write register offset to index register 0x38 of index/data pair (indirect addressing)
  // write register bits to data register 0x3c of index/data pair (indirect addressing)

  // std::lock_guard<std::mutex> lock(mutex_);
  PrintRegisterData(*index_reg_addr, *data_reg_addr, __FUNCTION__, 0);

  // TODO: should work only if map is created

  *index_reg_addr = reg_offset;
  *data_reg_addr = value;

  PrintRegisterData(*index_reg_addr, *data_reg_addr, __FUNCTION__, 1);
  return true;
}

bool MMIO::RegisterReadAPI(uint32_t reg_offset, uint32_t& value) {
  // access the mmap
  // write register offset to index register 0x38 of index/data pair (indirect addressing)
  // read register bits to data register 0x3c of index/data pair (indirect addressing)

  // std::lock_guard<std::mutex> lock(mutex_);
  PrintRegisterData(*index_reg_addr, *data_reg_addr, __FUNCTION__, 0);

  // TODO: should work only if map is created

  *index_reg_addr = reg_offset;
  // TODO: add delay here??
  value = *data_reg_addr;

  PrintRegisterData(*index_reg_addr, *data_reg_addr, __FUNCTION__, 1);
  return true;
}


MMIO* MMIOManager::CreateMMIO(mmap_type_t type, const HSAAgentInfo& info) {
  MMIO* mmio = nullptr;
  switch (type) {
    case PCIE_PERFMON: {
      mmio = GetMMIOInstance(type, info);
      if (mmio == nullptr) {
        mmio = dynamic_cast<MMIO*>(new PciePerfmonMMIO(info));
        AddInstance(mmio);
      }
      break;
    }
    case DF_PERFMON: {
      mmio = GetMMIOInstance(type, info);
      if (mmio == nullptr) {
        mmio = dynamic_cast<MMIO*>(new DFPerfmonMMIO(info));
        AddInstance(mmio);
      }
      break;
    }
    case UMC_PERFMON: {
      break;
    }
    case DEFAULT_MMAP: {
      break;
    }
  }
  return mmio;
}

MMIO* MMIOManager::GetMMIOInstance(mmap_type_t type, const HSAAgentInfo& info) {
  MMIO* mmio = nullptr;
  auto it = mmio_instances_.find(info.getHandle());
  if (it != mmio_instances_.end()) {
    for (auto& mmio_instance : it->second) {
      if (mmio_instance->Type() == type) {
        mmio = mmio_instance;
      }
    }
  }
  return mmio;
}

void MMIOManager::AddInstance(MMIO* in_mmio_instance) {
  uint64_t handle = in_mmio_instance->GetAgentInfo().getHandle();
  mmio_instances_[handle].push_back(in_mmio_instance);
}

void MMIOManager::DestroyMMIOInstance(MMIO* in_mmio_instance) {
  if (in_mmio_instance == nullptr) return;

  uint64_t handle = in_mmio_instance->GetAgentInfo().getHandle();
  auto it = mmio_instances_.find(handle);
  if (it != mmio_instances_.end()) {
    auto& mmio_array = it->second;
    // find instance in the array and remove it from the array
    mmio_array.erase(std::remove(mmio_array.begin(), mmio_array.end(), in_mmio_instance),
                     mmio_array.end());
  }
  delete in_mmio_instance;
}


std::map<decltype(hsa_agent_t::handle), std::vector<MMIO*>> MMIOManager::mmio_instances_;


}  // namespace mmio

}  // namespace rocprofiler
