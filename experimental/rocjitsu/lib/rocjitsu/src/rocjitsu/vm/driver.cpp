// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/driver.h"

#include "rocjitsu/vm/virtual_machine.h"
#include "simdojo/sim/simulation.h"

namespace rocjitsu {

Driver::Driver(VirtualMachine &vm) : vm_(vm) {}

void Driver::submit(amdgpu::DispatchPacket packet, uint32_t xcd_idx) {
  vm_.soc()->xcd(xcd_idx)->command_processor()->submit(std::move(packet));
}

void Driver::close() { vm_.engine()->request_exit("driver closed"); }

} // namespace rocjitsu
