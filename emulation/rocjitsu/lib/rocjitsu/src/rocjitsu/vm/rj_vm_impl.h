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
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace rocjitsu::detail {

/// Resolve the functional CU-dispatch width for each SoC.
///
/// A nonzero configured value is a per-SoC width. Zero selects an automatic
/// host-wide budget, capped at 32, which is divided as evenly as possible
/// across the SoCs. Every SoC retains a minimum width of one (serial dispatch).
std::vector<uint32_t> resolve_cpu_dispatch_thread_budgets(uint32_t configured_threads,
                                                          uint32_t hardware_threads,
                                                          size_t soc_count);

} // namespace rocjitsu::detail

struct rj_vm_t : rocjitsu::RefCounted {
  std::unique_ptr<simdojo::SimulationEngine> engine;
  simdojo::SimulationEngine::Config engine_config{};
  rocjitsu::config::LoadedConfig loaded;
  rocjitsu::SoC *soc = nullptr;
  rocjitsu::VirtualMachine *vm = nullptr;
  std::atomic<bool> plugin_group_active{false};
};

#endif // ROCJITSU_VM_RJ_VM_IMPL_H_
