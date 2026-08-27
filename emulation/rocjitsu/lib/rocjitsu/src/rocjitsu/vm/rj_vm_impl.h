// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file rj_vm_impl.h
/// @brief Private definition of rj_vm_t. Internal to the library.

#ifndef ROCJITSU_VM_RJ_VM_IMPL_H_
#define ROCJITSU_VM_RJ_VM_IMPL_H_

#include "rocjitsu/config/config_loader.h"
#include "rocjitsu/refcount.h"
#include "rocjitsu/vm/timing/collector.h"
#include "rocjitsu/vm/timing/timing_plane.h"
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
  /// @brief The timed machine, when the config asked for one.
  ///
  /// @details Ordering here is load bearing. The simulated clock holds a raw
  /// pointer to the plane while it is installed, and guest threads read that
  /// clock from inside ioctls with none of this object's locks held, so the
  /// clock must be pointed away from the plane before the plane is destroyed.
  /// The collector is destroyed before the plane because it calls into it.
  std::unique_ptr<rocjitsu::timing::TimingPlane> timing_plane;
  std::unique_ptr<rocjitsu::timing::TimingCollector> timing_collector;
};

#endif // ROCJITSU_VM_RJ_VM_IMPL_H_
