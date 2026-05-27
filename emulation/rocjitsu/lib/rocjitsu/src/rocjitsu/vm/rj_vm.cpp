// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/rj_vm.h"

#include "embedded_schema.h"
#include "rocjitsu/config/checkpoint.h"
#include "rocjitsu/config/config_loader.h"
#include "rocjitsu/refcount.h"
#include "rocjitsu/vm/soc.h"

#include "simdojo/sim/simulation.h"

#include <memory>
#include <stdexcept>

using namespace rocjitsu;

/// @brief Concrete definition of the opaque rj_vm_t handle.
struct rj_vm_t : RefCounted {
  std::unique_ptr<simdojo::SimulationEngine> engine;
  simdojo::SimulationEngine::Config engine_config{};
  SoC *soc = nullptr; ///< Cached pointer into topology root.
};

namespace {

/// @brief Create an rj_vm_t from a LoadedConfig.
rj_status_t create_from_loaded(config::LoadedConfig &loaded, rj_vm_t **handle) {
  if (!loaded.soc())
    return ROCJITSU_STATUS_ERROR;

  auto s = std::make_unique<rj_vm_t>();
  s->engine_config = loaded.engine_config;
  s->soc = loaded.soc();
  s->engine = std::make_unique<simdojo::SimulationEngine>(loaded.engine_config);
  s->engine->topology().set_root(loaded.take_root());
  loaded.wire_links(s->engine->topology());
  s->engine->build();

  *handle = s.release();
  return ROCJITSU_STATUS_SUCCESS;
}

} // namespace

rj_status_t rj_vm_create(const char *json_path, rj_vm_t **vm) {
  if (!json_path || !vm)
    return ROCJITSU_STATUS_INVALID_ARGUMENT;
  try {
    auto loaded = config::load_config(json_path, rocjitsu::kEmbeddedSchema);
    return create_from_loaded(loaded, vm);
  } catch (const std::exception &) {
    return ROCJITSU_STATUS_INVALID_FILE;
  }
}

rj_status_t rj_vm_create_from_string(const char *json, rj_vm_t **vm) {
  if (!json || !vm)
    return ROCJITSU_STATUS_INVALID_ARGUMENT;
  try {
    auto loaded = config::load_config_from_string(json, rocjitsu::kEmbeddedSchema);
    return create_from_loaded(loaded, vm);
  } catch (const std::exception &) {
    return ROCJITSU_STATUS_INVALID_ARGUMENT;
  }
}

void rj_vm_retain(rj_vm_t *vm) {
  if (vm)
    vm->retain();
}

void rj_vm_release(rj_vm_t *vm) {
  if (!vm)
    return;
  if (vm->release())
    delete vm;
}

void rj_vm_destroy(rj_vm_t *vm) {
  if (!vm)
    return;
  if (vm->destroy())
    delete vm;
}

rj_status_t rj_vm_step(rj_vm_t *vm, int *active) {
  if (!vm)
    return ROCJITSU_STATUS_INVALID_ARGUMENT;
  if (!vm->soc)
    return ROCJITSU_STATUS_ERROR;

  bool any_active = vm->engine->step();
  if (active)
    *active = any_active ? 1 : 0;
  return ROCJITSU_STATUS_SUCCESS;
}

rj_status_t rj_vm_run(rj_vm_t *vm, uint64_t *ticks_executed) {
  if (!vm)
    return ROCJITSU_STATUS_INVALID_ARGUMENT;
  if (!vm->soc)
    return ROCJITSU_STATUS_ERROR;

  auto exit = vm->engine->run();

  if (ticks_executed)
    *ticks_executed = exit.tick;

  vm->engine->shutdown();
  return (exit.code == 0) ? ROCJITSU_STATUS_SUCCESS : ROCJITSU_STATUS_ERROR;
}

rj_status_t rj_vm_save_checkpoint(const rj_vm_t *vm, const char *path, uint64_t tick) {
  if (!vm || !path)
    return ROCJITSU_STATUS_INVALID_ARGUMENT;
  if (!vm->soc)
    return ROCJITSU_STATUS_ERROR;
  try {
    config::save_checkpoint(path, *vm->soc, tick, vm->engine_config);
    return ROCJITSU_STATUS_SUCCESS;
  } catch (const std::exception &) {
    return ROCJITSU_STATUS_ERROR;
  }
}

rj_status_t rj_vm_restore_checkpoint(const char *path, rj_vm_t **vm) {
  if (!path || !vm)
    return ROCJITSU_STATUS_INVALID_ARGUMENT;
  try {
    auto loaded = config::restore_checkpoint(path);
    return create_from_loaded(loaded, vm);
  } catch (const std::exception &) {
    return ROCJITSU_STATUS_INVALID_FILE;
  }
}
