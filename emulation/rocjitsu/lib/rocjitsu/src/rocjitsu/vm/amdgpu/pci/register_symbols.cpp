// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/pci/register_symbols.h"

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
  symbols.add_dword(bar, 0x0, "MM_INDEX");
  symbols.add_dword(bar, 0x1, "MM_DATA");
  symbols.add_dword(bar, 0x6, "MM_INDEX_HI");
  symbols.add_dword(bar, 0x94, "DRIVER_SCRATCH_0");
  symbols.add_dword(bar, 0x95, "DRIVER_SCRATCH_1");
  symbols.add_dword(bar, 0x96, "DRIVER_SCRATCH_2");
  symbols.add_dword(bar, 0xde3, "RCC_CONFIG_MEMSIZE");
  symbols.add_dword(bar, 0x16061, "MP0_SMN_C2PMSG_33");
  symbols.add_dword(bar, 0x16a00, "IP_DISCOVERY_VERSION");
}

} // namespace rocjitsu
