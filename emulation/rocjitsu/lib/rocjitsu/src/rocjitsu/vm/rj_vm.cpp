// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/rj_vm.h"

#include "embedded_schema.h"
#include "rocjitsu/config/checkpoint.h"
#include "rocjitsu/kmd/linux/simulated_driver.h"
#include "rocjitsu/vm/rj_vm_impl.h"
#include "rocjitsu/vm/soc.h"

#include "rocjitsu/base/rj_compiler.h"
RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include "linux/uapi/kfd_ioctl.h"
RJ_DIAGNOSTIC_POP

#include <cstring>
#include <memory>
#include <stdexcept>
#include <sys/ioctl.h>

using namespace rocjitsu;

namespace {

rj_status_t create_from_loaded(config::LoadedConfig &loaded, rj_vm_mode_t mode, rj_vm_t **handle) {
  if (!loaded.soc())
    return ROCJITSU_STATUS_ERROR;

  auto s = std::make_unique<rj_vm_t>();
  s->soc = loaded.soc();
  auto num_xcds = s->soc->num_xcds();

  bool serve = (mode == RJ_VM_MODE_LOCAL || mode == RJ_VM_MODE_DAEMON);
  bool daemon = (mode == RJ_VM_MODE_DAEMON);
  if (serve) {
    loaded.engine_config.max_ticks = 0;
    loaded.engine_config.await_primaries = true;
  }

  s->engine_config = loaded.engine_config;
  s->engine = std::make_unique<simdojo::SimulationEngine>(loaded.engine_config);

  if (loaded.num_gpus > 1 && !loaded.extra_gpu_builds.empty()) {
    std::vector<std::unique_ptr<SoC>> socs;
    std::vector<uint32_t> gpu_ids;

    auto root0 = loaded.take_root();
    root0.release();
    socs.push_back(std::unique_ptr<SoC>(s->soc));
    gpu_ids.push_back(loaded.devices.empty() ? 0 : loaded.devices[0].gpu_id);

    for (size_t i = 0; i < loaded.extra_gpu_builds.size(); ++i) {
      auto &eb = loaded.extra_gpu_builds[i];
      auto *extra_soc = dynamic_cast<SoC *>(eb.root.get());
      if (!extra_soc)
        continue;
      eb.root.release();
      socs.push_back(std::unique_ptr<SoC>(extra_soc));
      gpu_ids.push_back(i + 1 < loaded.devices.size() ? loaded.devices[i + 1].gpu_id : 0);
    }

    auto vm_ptr = std::make_unique<VirtualMachine>(std::move(socs), std::move(gpu_ids), daemon);
    s->vm = vm_ptr.get();
    s->engine->topology().set_root(std::move(vm_ptr));
    loaded.wire_links(s->engine->topology());
    s->soc->wire_backing(s->engine->topology());
    for (auto &eb : loaded.extra_gpu_builds) {
      s->engine->topology().wire_links(eb.link_specs, loaded.exec_mode);
      auto *extra_soc = dynamic_cast<SoC *>(s->vm->soc());
      if (extra_soc)
        extra_soc->wire_backing(s->engine->topology());
    }
  } else {
    auto root = loaded.take_root();
    root.release();
    auto vm_ptr = std::make_unique<VirtualMachine>(std::unique_ptr<SoC>(s->soc), daemon);
    s->vm = vm_ptr.get();
    s->engine->topology().set_root(std::move(vm_ptr));
    loaded.wire_links(s->engine->topology());
    s->soc->wire_backing(s->engine->topology());
  }
  s->engine->build();

  if (serve) {
    s->engine->register_as_primary();
    if (loaded.num_gpus > 1 && !loaded.devices.empty())
      s->vm->driver()->setup_topology(loaded.devices, num_xcds);
    else
      s->vm->driver()->setup_topology(loaded.device, num_xcds);
    s->vm->driver()->open();
  }

  s->loaded = std::move(loaded);
  *handle = s.release();
  return ROCJITSU_STATUS_SUCCESS;
}

void reconstruct_embedded_pointers(uint32_t cmd, void *arg, size_t arg_size, size_t total_size) {
  if (total_size <= arg_size)
    return;
  auto *extra = static_cast<uint8_t *>(arg) + arg_size;
  switch (cmd) {
  case AMDKFD_IOC_WAIT_EVENTS: {
    auto *args = static_cast<kfd_ioctl_wait_events_args *>(arg);
    args->events_ptr = reinterpret_cast<uint64_t>(extra);
    break;
  }
  case AMDKFD_IOC_MAP_MEMORY_TO_GPU:
  case AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU: {
    auto *args = static_cast<kfd_ioctl_map_memory_to_gpu_args *>(arg);
    args->device_ids_array_ptr = reinterpret_cast<uint64_t>(extra);
    break;
  }
  case AMDKFD_IOC_GET_PROCESS_APERTURES_NEW: {
    auto *args = static_cast<kfd_ioctl_get_process_apertures_new_args *>(arg);
    args->kfd_process_device_apertures_ptr = reinterpret_cast<uint64_t>(extra);
    break;
  }
  default:
    break;
  }
}

} // namespace

rj_status_t rj_vm_create(const char *json_path, rj_vm_mode_t mode, rj_vm_t **vm) {
  if (!json_path || !vm)
    return ROCJITSU_STATUS_INVALID_ARGUMENT;
  try {
    auto loaded = config::load_config(json_path, rocjitsu::kEmbeddedSchema);
    return create_from_loaded(loaded, mode, vm);
  } catch (const std::exception &) {
    return ROCJITSU_STATUS_INVALID_FILE;
  }
}

rj_status_t rj_vm_create_from_string(const char *json, rj_vm_mode_t mode, rj_vm_t **vm) {
  if (!json || !vm)
    return ROCJITSU_STATUS_INVALID_ARGUMENT;
  try {
    auto loaded = config::load_config_from_string(json, rocjitsu::kEmbeddedSchema);
    return create_from_loaded(loaded, mode, vm);
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

void rj_vm_request_exit(rj_vm_t *vm, const char *reason) {
  if (!vm || !vm->engine)
    return;
  vm->engine->request_exit(reason ? reason : "shutdown");
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
    return create_from_loaded(loaded, RJ_VM_MODE_DEFAULT, vm);
  } catch (const std::exception &) {
    return ROCJITSU_STATUS_INVALID_FILE;
  }
}

namespace {

rj_status_t execute_impl(SimulatedDriver *driver, uint32_t process_id, rj_vm_cmd_t *cmd) {
  auto arg_size = _IOC_SIZE(cmd->cmd);
  reconstruct_embedded_pointers(cmd->cmd, cmd->buf, arg_size, cmd->buf_size);

  cmd->result = driver->ioctl(process_id, cmd->cmd, cmd->buf);
  cmd->shared_handle = -1;

  if (cmd->cmd == AMDKFD_IOC_ALLOC_MEMORY_OF_GPU && cmd->result == 0) {
    auto *alloc_args = static_cast<kfd_ioctl_alloc_memory_of_gpu_args *>(cmd->buf);
    cmd->shared_handle =
        driver->get_mmap_memfd(process_id, static_cast<off_t>(alloc_args->mmap_offset));
  } else if (cmd->cmd == AMDKFD_IOC_IPC_IMPORT_HANDLE && cmd->result == 0) {
    auto *import_args = static_cast<kfd_ioctl_ipc_import_handle_args *>(cmd->buf);
    cmd->shared_handle =
        driver->get_mmap_memfd(process_id, static_cast<off_t>(import_args->mmap_offset));
  }

  return ROCJITSU_STATUS_SUCCESS;
}

} // namespace

rj_status_t rj_vm_execute(rj_vm_t *vm, rj_vm_cmd_t *cmd) {
  if (!vm || !cmd || !vm->vm || !vm->vm->driver())
    return ROCJITSU_STATUS_INVALID_ARGUMENT;
  auto *driver = vm->vm->driver();
  return execute_impl(driver, driver->local_process_id(), cmd);
}

rj_status_t rj_vm_execute_as(rj_vm_t *vm, uint32_t process_id, rj_vm_cmd_t *cmd) {
  if (!vm || !cmd || !vm->vm || !vm->vm->driver())
    return ROCJITSU_STATUS_INVALID_ARGUMENT;
  return execute_impl(vm->vm->driver(), process_id, cmd);
}

rj_status_t rj_vm_device_open(rj_vm_t *vm, uint32_t *process_id) {
  if (!vm || !vm->vm || !vm->vm->driver())
    return ROCJITSU_STATUS_INVALID_ARGUMENT;
  auto *drv = dynamic_cast<SimulatedDriver *>(vm->vm->driver());
  if (!drv)
    return ROCJITSU_STATUS_ERROR;
  uint32_t pid = drv->open_process();
  if (pid == 0)
    return ROCJITSU_STATUS_ERROR;
  if (process_id)
    *process_id = pid;
  return ROCJITSU_STATUS_SUCCESS;
}

rj_status_t rj_vm_device_close(rj_vm_t *vm, uint32_t process_id) {
  if (!vm || !vm->vm || !vm->vm->driver())
    return ROCJITSU_STATUS_INVALID_ARGUMENT;
  if (process_id == 0)
    vm->vm->driver()->close();
  else
    vm->vm->driver()->close(process_id);
  return ROCJITSU_STATUS_SUCCESS;
}

rj_status_t rj_vm_device_map(rj_vm_t *vm, rj_vm_map_t *map) {
  if (!vm || !map || !vm->vm || !vm->vm->driver())
    return ROCJITSU_STATUS_INVALID_ARGUMENT;
  auto *driver = vm->vm->driver();
  auto *result = driver->mmap(reinterpret_cast<void *>(map->addr), static_cast<size_t>(map->length),
                              static_cast<int>(map->prot), static_cast<int>(map->flags),
                              static_cast<off_t>(map->offset));
  map->mapped_addr = reinterpret_cast<uint64_t>(result);
  return ROCJITSU_STATUS_SUCCESS;
}

rj_status_t rj_vm_device_map_as(rj_vm_t *vm, uint32_t process_id, rj_vm_map_t *map) {
  if (!vm || !map || !vm->vm || !vm->vm->driver())
    return ROCJITSU_STATUS_INVALID_ARGUMENT;
  auto *result = vm->vm->driver()->mmap(
      process_id, reinterpret_cast<void *>(map->addr), static_cast<size_t>(map->length),
      static_cast<int>(map->prot), static_cast<int>(map->flags), static_cast<off_t>(map->offset));
  map->mapped_addr = reinterpret_cast<uint64_t>(result);
  return ROCJITSU_STATUS_SUCCESS;
}

rj_status_t rj_vm_device_unmap(rj_vm_t *vm, rj_vm_unmap_t *unmap) {
  if (!vm || !unmap || !vm->vm || !vm->vm->driver())
    return ROCJITSU_STATUS_INVALID_ARGUMENT;
  vm->vm->driver()->munmap(reinterpret_cast<void *>(unmap->addr),
                           static_cast<size_t>(unmap->length));
  return ROCJITSU_STATUS_SUCCESS;
}

rj_status_t rj_vm_device_unmap_as(rj_vm_t *vm, uint32_t process_id, rj_vm_unmap_t *unmap) {
  if (!vm || !unmap || !vm->vm || !vm->vm->driver())
    return ROCJITSU_STATUS_INVALID_ARGUMENT;
  vm->vm->driver()->munmap(process_id, reinterpret_cast<void *>(unmap->addr),
                           static_cast<size_t>(unmap->length));
  return ROCJITSU_STATUS_SUCCESS;
}

rj_status_t rj_vm_gpu_id(rj_vm_t *vm, uint32_t *gpu_id) {
  if (!vm || !gpu_id || !vm->vm || !vm->vm->driver())
    return ROCJITSU_STATUS_INVALID_ARGUMENT;
  *gpu_id = vm->vm->driver()->gpu_id();
  return ROCJITSU_STATUS_SUCCESS;
}

rj_status_t rj_vm_topology_path(rj_vm_t *vm, const char **path) {
  if (!vm || !path || !vm->vm || !vm->vm->driver())
    return ROCJITSU_STATUS_INVALID_ARGUMENT;
  static thread_local std::string cached_path;
  cached_path = vm->vm->driver()->topology_path();
  *path = cached_path.c_str();
  return ROCJITSU_STATUS_SUCCESS;
}

rj_status_t rj_vm_drm_path(rj_vm_t *vm, const char **path) {
  if (!vm || !path || !vm->vm || !vm->vm->driver())
    return ROCJITSU_STATUS_INVALID_ARGUMENT;
  static thread_local std::string cached_drm;
  cached_drm = vm->vm->driver()->topology().drm_path();
  *path = cached_drm.c_str();
  return ROCJITSU_STATUS_SUCCESS;
}

rj_status_t rj_vm_get_shared_mem(rj_vm_t *vm, int64_t offset, rj_handle_t *handle) {
  if (!vm || !handle || !vm->vm || !vm->vm->driver())
    return ROCJITSU_STATUS_INVALID_ARGUMENT;
  *handle = vm->vm->driver()->get_mmap_memfd(static_cast<off_t>(offset));
  return ROCJITSU_STATUS_SUCCESS;
}

rj_status_t rj_vm_get_shared_mem_as(rj_vm_t *vm, uint32_t process_id, int64_t offset,
                                    rj_handle_t *handle) {
  if (!vm || !handle || !vm->vm || !vm->vm->driver())
    return ROCJITSU_STATUS_INVALID_ARGUMENT;
  *handle = vm->vm->driver()->get_mmap_memfd(process_id, static_cast<off_t>(offset));
  return ROCJITSU_STATUS_SUCCESS;
}
