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

#ifndef SRC_CORE_NON_GFXIP_COUNTERS_MMIO_H
#define SRC_CORE_NON_GFXIP_COUNTERS_MMIO_H

#include <hsa/hsa.h>
#include "src/core/hardware/hsa_info.h"
#include "src/core/hsa/hsa_support.h"

#include <pciaccess.h>
#include <mutex>

#include <iostream>
#include <unistd.h>
#include <sstream>

namespace rocprofiler {

namespace mmio {

#define FUNCTION_START() PrintFunctionPhase(__FUNCTION__, 0)
#define FUNCTION_END() PrintFunctionPhase(__FUNCTION__, 1)

// uncomment below to see register write sequences
// #define DEBUG_TRACE = 1

void PrintFunctionPhase(const char* function_name, int phase);
void PrintRegisterData(uint32_t& index_value, uint32_t& data_value, const char* function_name,
                       int phase);

// Index/Data registers
const static uint32_t INDIRECT_REG_INDEX = 0x38;
const static uint32_t INDIRECT_REG_DATA = 0x3c;

typedef enum { DEFAULT_MMAP, DF_PERFMON, UMC_PERFMON, PCIE_PERFMON } mmap_type_t;

class MMIOManager;

class MMIO {
 public:
  virtual bool RegisterWriteAPI(uint32_t reg_offset, uint32_t value);
  virtual bool RegisterReadAPI(uint32_t reg_offset, uint32_t& value);
  virtual void SetIndexDataRegisters(const uint32_t index_reg, const uint32_t data_reg) {
    index_reg_addr = (uint32_t*)((char*)pci_memory_ + index_reg);
    data_reg_addr = (uint32_t*)((char*)pci_memory_ + data_reg);
  }

  MMIO(MMIO& other) = delete;
  void operator=(const MMIO&) = delete;
  virtual ~MMIO();
  friend class MMIOManager;

  const HSAAgentInfo& GetAgentInfo() { return *agent_info_; }
  mmap_type_t Type() { return type_; }

 protected:
  MMIO(const HSAAgentInfo& info);

  // default constructor; helpful for derived classes
  // which want to setup mmio construction differently
  MMIO() { type_ = DEFAULT_MMAP; };

  const HSAAgentInfo* agent_info_;
  struct pci_device* pci_device_;
  size_t pci_memory_size_;
  uint32_t* pci_memory_;
  mmap_type_t type_;

  uint32_t* index_reg_addr;
  uint32_t* data_reg_addr;
};

// PciePerfmonMMIO has same mmio setup approach as
// done in MMIO class
class PciePerfmonMMIO : public MMIO {
 public:
  friend class MMIOManager;

 protected:
  PciePerfmonMMIO(const HSAAgentInfo& info) : MMIO(info) { type_ = PCIE_PERFMON; };
};

// DFPerfmonMMIO has same mmio setup approach as
// done in MMIO class
class DFPerfmonMMIO : public MMIO {
 public:
  friend class MMIOManager;

 protected:
  DFPerfmonMMIO( const HSAAgentInfo& info) : MMIO(info) { type_ = DF_PERFMON; };
};
/*
    Class to manage mmio for UMC/DF/PCIe etc.
    The mmio approach for the different IPs may
    be same or different. For eg: UMC and PCIe share
    the same mmio and index/data registers
*/
class MMIOManager {
 public:
  static MMIO* CreateMMIO(mmap_type_t type,  const HSAAgentInfo& info);
  static MMIO* GetMMIOInstance(mmap_type_t type, const HSAAgentInfo& info);
  static void DestroyMMIOInstance(MMIO* instance);

 private:
  static void AddInstance(MMIO* instance);
  static std::map<decltype(hsa_agent_t::handle), std::vector<MMIO*>> mmio_instances_;
};


}  // namespace mmio

}  // namespace rocprofiler


#endif