// Copyright (c) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/risc_v/hart_state.h"

namespace rocjitsu {
namespace risc_v {

static thread_local Memory *tl_memory = nullptr;

Memory *current_memory() { return tl_memory; }

void set_current_memory(Memory *m) { tl_memory = m; }

uint64_t HartState::read_csr(uint16_t addr) const {
  auto it = csrs.find(addr);
  return (it != csrs.end()) ? it->second : 0;
}

void HartState::write_csr(uint16_t addr, uint64_t val) { csrs[addr] = val; }

} // namespace risc_v
} // namespace rocjitsu
