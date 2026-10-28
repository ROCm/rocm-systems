// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file rj_vm_impl.h
/// @brief Private definition of rj_vm_t. Internal to the library.

#ifndef ROCJITSU_VM_RJ_VM_IMPL_H_
#define ROCJITSU_VM_RJ_VM_IMPL_H_

#include "rocjitsu/config/config_loader.h"
#include "rocjitsu/refcount.h"
#include "rocjitsu/vm/virtual_machine.h"

#include "simdojo/sim/simulation.h"

#include <atomic>
#include <memory>

struct rj_vm_t : rocjitsu::RefCounted {
  std::unique_ptr<simdojo::SimulationEngine> engine;
  simdojo::SimulationEngine::Config engine_config{};
  rocjitsu::config::LoadedConfig loaded;
  rocjitsu::SoC *soc = nullptr;
  rocjitsu::VirtualMachine *vm = nullptr;
  std::atomic<bool> plugin_group_active{false};
  // Intrusive process-lifetime retry link used only if an interposer VM's
  // checked destruction refuses final dirty writeback after its last owner
  // reference has gone away. This is private to the implementation, not C ABI.
  rj_vm_t *pending_destroy_next = nullptr;
};

#endif // ROCJITSU_VM_RJ_VM_IMPL_H_
