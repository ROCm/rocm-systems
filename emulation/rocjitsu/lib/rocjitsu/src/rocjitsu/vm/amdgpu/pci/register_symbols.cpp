// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/pci/register_symbols.h"

#include "rocjitsu/vm/amdgpu/pci/mmio_registers.h"

#include <utility>

namespace rocjitsu {

uint64_t RegisterSymbols::make_key(int bar, uint64_t byte_offset) {
  return (static_cast<uint64_t>(static_cast<uint32_t>(bar)) << 56) |
         (byte_offset & 0x00ffffffffffffffULL);
}

void RegisterSymbols::add(int bar, uint64_t byte_offset, std::string name) {
  names_.insert_or_assign(make_key(bar, byte_offset), std::move(name));
}

void RegisterSymbols::add_dword(int bar, uint64_t dword_index, std::string name) {
  add(bar, dword_index * 4, std::move(name));
}

std::string_view RegisterSymbols::lookup(int bar, uint64_t byte_offset) const {
  const auto it = names_.find(make_key(bar, byte_offset));
  if (it == names_.end()) {
    return {};
  }
  return it->second;
}

void add_pre_discovery_symbols(RegisterSymbols &symbols, int bar) {
  const auto name_register = [&symbols, bar](MmioRegister reg, std::string name) {
    symbols.add(bar, byte_offset_of(reg), std::move(name));
  };
  name_register(MmioRegister::MmIndex, "MM_INDEX");
  name_register(MmioRegister::MmData, "MM_DATA");
  name_register(MmioRegister::MmIndexHi, "MM_INDEX_HI");
  name_register(MmioRegister::DriverScratch0, "DRIVER_SCRATCH_0");
  name_register(MmioRegister::DriverScratch1, "DRIVER_SCRATCH_1");
  name_register(MmioRegister::DriverScratch2, "DRIVER_SCRATCH_2");
  name_register(MmioRegister::RccConfigMemsize, "RCC_CONFIG_MEMSIZE");
  name_register(MmioRegister::RccIovFuncIdentifier, "RCC_IOV_FUNC_IDENTIFIER");
  name_register(MmioRegister::Mp0SmnC2pmsg33, "MP0_SMN_C2PMSG_33");
  name_register(MmioRegister::IpDiscoveryVersion, "IP_DISCOVERY_VERSION");
}

} // namespace rocjitsu
