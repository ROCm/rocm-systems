// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/rj_vm.h"

#include "embedded_schema.h"
#include "rocjitsu/config/checkpoint.h"
#include "rocjitsu/kmd/linux/simulated_kfd.h"
#include "rocjitsu/vm/amdgpu/partitioning.h"
#include "rocjitsu/vm/plugins/plugin_loader.h"
#include "rocjitsu/vm/rj_vm_impl.h"
#include "rocjitsu/vm/soc.h"

#include "rocjitsu/base/rj_compiler.h"
#include "util/log.h"
RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include "linux/uapi/kfd_ioctl.h"
RJ_DIAGNOSTIC_POP

#include <cerrno>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <vector>

using namespace rocjitsu;

namespace {

void shutdown_plugin_group(rj_vm_t *vm) {
  // Host API preconditions keep this outside the simulation-callback interval:
  // either execution has not started, or the engine workers have stopped and
  // joined. callback_mutex_ is intentionally not lifecycle synchronization.
  if (vm && vm->soc && vm->plugin_group_active.exchange(false, std::memory_order_acq_rel))
    vm->soc->plugin_group().onShutdown();
}

rj_status_t create_from_loaded(config::LoadedConfig &loaded, rj_vm_mode_t mode, rj_vm_t **handle) {
  if (!loaded.soc())
    return ROCJITSU_STATUS_ERROR;

  auto s = std::make_unique<rj_vm_t>();
  s->soc = loaded.soc();
  auto num_xcds = s->soc->num_xcds();
  std::vector<SoC *> partition_socs;
  partition_socs.reserve(loaded.extra_gpu_builds.size() + 1);
  partition_socs.push_back(s->soc);
  if (loaded.num_gpus > 1) {
    for (auto &eb : loaded.extra_gpu_builds) {
      if (auto *extra_soc = dynamic_cast<SoC *>(eb.root.get()))
        partition_socs.push_back(extra_soc);
    }
  }
  // XCD partitions (config num_threads): run each XCD on its own engine
  // partition/thread so the XCDs execute concurrently across their separate L2s.
  const uint32_t num_threads_requested = loaded.engine_config.num_threads;
  const uint32_t num_threads_used =
      amdgpu::clamp_xcd_partition_count(partition_socs, num_threads_requested);
  if (num_threads_used != num_threads_requested)
    util::Logger::warn("num_threads clamped: requested=", num_threads_requested,
                       ", effective=", num_threads_used);
  loaded.engine_config.num_threads = num_threads_used;

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

    auto prefix_specs = [](std::vector<simdojo::LinkSpec> &specs, const std::string &pfx) {
      for (auto &ls : specs) {
        ls.src = pfx + ls.src;
        ls.dst = pfx + ls.dst;
      }
    };
    prefix_specs(loaded.build_result.link_specs, "gpu0.");
    loaded.wire_links(s->engine->topology());
    s->soc->wire_backing(s->engine->topology());
    for (size_t i = 0; i < loaded.extra_gpu_builds.size(); ++i) {
      auto &eb = loaded.extra_gpu_builds[i];
      prefix_specs(eb.link_specs, "gpu" + std::to_string(i + 1) + ".");
      s->engine->topology().wire_links(eb.link_specs, loaded.exec_mode);
      auto *extra_soc = s->vm->soc(static_cast<uint32_t>(i + 1));
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
  if (num_threads_used > 1 && !amdgpu::partition_topology_by_xcds(
                                  s->engine->topology(), partition_socs, num_threads_used)) {
    throw std::invalid_argument("multi-threaded VM requires at least one XCD");
  }
  s->engine->create();

  // dispatch_wf() cannot enqueue work while a restored component tree is
  // detached from an engine. Once create() has assigned partitions and event
  // queues, resume any resident waves reconstructed from a checkpoint. The
  // schedule_work() guards make this a no-op for ordinary idle configurations.
  for (uint32_t i = 0; i < s->vm->num_socs(); ++i) {
    for (auto *cu : s->vm->soc(i)->all_cus()) {
      if (!cu->is_idle())
        cu->schedule_work();
    }
  }

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

bool reconstruct_embedded_pointers(uint32_t cmd, void *arg, size_t arg_size, size_t total_size) {
  if (total_size < arg_size)
    return false;
  auto *extra = static_cast<uint8_t *>(arg) + arg_size;
  const size_t inline_size = total_size - arg_size;
  auto has_entries = [inline_size](size_t count, size_t entry_size) {
    return entry_size == 0 || count <= inline_size / entry_size;
  };
  switch (cmd) {
  case AMDKFD_IOC_WAIT_EVENTS: {
    auto *args = static_cast<kfd_ioctl_wait_events_args *>(arg);
    if (!has_entries(args->num_events, sizeof(kfd_event_data)))
      return false;
    if (args->num_events > 0)
      args->events_ptr = reinterpret_cast<uint64_t>(extra);
    break;
  }
  case AMDKFD_IOC_MAP_MEMORY_TO_GPU:
  case AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU: {
    auto *args = static_cast<kfd_ioctl_map_memory_to_gpu_args *>(arg);
    if (!has_entries(args->n_devices, sizeof(uint32_t)))
      return false;
    if (args->n_devices > 0)
      args->device_ids_array_ptr = reinterpret_cast<uint64_t>(extra);
    break;
  }
  case AMDKFD_IOC_GET_PROCESS_APERTURES_NEW: {
    auto *args = static_cast<kfd_ioctl_get_process_apertures_new_args *>(arg);
    if (!has_entries(args->num_of_nodes, sizeof(kfd_process_device_apertures)))
      return false;
    if (args->num_of_nodes > 0)
      args->kfd_process_device_apertures_ptr = reinterpret_cast<uint64_t>(extra);
    break;
  }
  case AMDKFD_IOC_DBG_TRAP: {
    auto *args = static_cast<kfd_ioctl_dbg_trap_args *>(arg);
    switch (args->op) {
    case KFD_IOC_DBG_TRAP_ENABLE: {
      const size_t required =
          std::min(static_cast<size_t>(args->enable.rinfo_size), sizeof(kfd_runtime_info));
      if (required > inline_size)
        return false;
      if (required > 0)
        args->enable.rinfo_ptr = reinterpret_cast<uint64_t>(extra);
      break;
    }
    case KFD_IOC_DBG_TRAP_GET_DEVICE_SNAPSHOT:
      if (!has_entries(args->device_snapshot.num_devices, args->device_snapshot.entry_size))
        return false;
      if (args->device_snapshot.num_devices > 0 && args->device_snapshot.snapshot_buf_ptr != 0)
        args->device_snapshot.snapshot_buf_ptr = reinterpret_cast<uint64_t>(extra);
      break;
    case KFD_IOC_DBG_TRAP_GET_QUEUE_SNAPSHOT:
      if (!has_entries(args->queue_snapshot.num_queues, args->queue_snapshot.entry_size))
        return false;
      if (args->queue_snapshot.num_queues > 0 && args->queue_snapshot.snapshot_buf_ptr != 0)
        args->queue_snapshot.snapshot_buf_ptr = reinterpret_cast<uint64_t>(extra);
      break;
    case KFD_IOC_DBG_TRAP_QUERY_EXCEPTION_INFO:
      if (args->query_exception_info.info_size > inline_size)
        return false;
      if (args->query_exception_info.info_size > 0 && args->query_exception_info.info_ptr != 0)
        args->query_exception_info.info_ptr = reinterpret_cast<uint64_t>(extra);
      break;
    case KFD_IOC_DBG_TRAP_SUSPEND_QUEUES:
      if (!has_entries(args->suspend_queues.num_queues, sizeof(uint32_t)))
        return false;
      if (args->suspend_queues.num_queues > 0 && args->suspend_queues.queue_array_ptr != 0)
        args->suspend_queues.queue_array_ptr = reinterpret_cast<uint64_t>(extra);
      break;
    case KFD_IOC_DBG_TRAP_RESUME_QUEUES:
      if (!has_entries(args->resume_queues.num_queues, sizeof(uint32_t)))
        return false;
      if (args->resume_queues.num_queues > 0 && args->resume_queues.queue_array_ptr != 0)
        args->resume_queues.queue_array_ptr = reinterpret_cast<uint64_t>(extra);
      break;
    default:
      break;
    }
    break;
  }
  default:
    break;
  }
  return true;
}

} // namespace

rj_status_t rj_vm_create(const char *json_path, rj_vm_mode_t mode, rj_vm_t **vm) {
  if (!json_path || !vm)
    return ROCJITSU_STATUS_INVALID_ARGUMENT;
  try {
    auto loaded = config::load_config(json_path, rocjitsu::kEmbeddedSchema);
    return create_from_loaded(loaded, mode, vm);
  } catch (const std::exception &e) {
    util::Logger::warn("rj_vm_create failed: ", e.what());
    return ROCJITSU_STATUS_INVALID_FILE;
  }
}

rj_status_t rj_vm_create_from_string(const char *json, rj_vm_mode_t mode, rj_vm_t **vm) {
  if (!json || !vm)
    return ROCJITSU_STATUS_INVALID_ARGUMENT;
  try {
    auto loaded = config::load_config_from_string(json, rocjitsu::kEmbeddedSchema);
    return create_from_loaded(loaded, mode, vm);
  } catch (const std::exception &e) {
    util::Logger::warn("rj_vm_create_from_string failed: ", e.what());
    return ROCJITSU_STATUS_INVALID_ARGUMENT;
  }
}

rj_status_t rj_vm_load_plugins(rj_vm_t *vm, const char *config_json, const char *plugin_dir) {
  if (!vm || !config_json)
    return ROCJITSU_STATUS_INVALID_ARGUMENT;
  if (!vm->soc)
    return ROCJITSU_STATUS_ERROR;
  try {
    auto group = PluginLoader::configure_plugin_group(config_json, plugin_dir ? plugin_dir : "");
    // rj_vm_load_plugins() is a pre-run operation, so replacing and initializing
    // the group cannot overlap simulation callbacks.
    shutdown_plugin_group(vm);
    vm->soc->set_plugin_group(group);
    group->onInit();
    vm->plugin_group_active.store(true, std::memory_order_release);
    return ROCJITSU_STATUS_SUCCESS;
  } catch (const std::exception &) {
    return ROCJITSU_STATUS_ERROR;
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
  // The API requires an asynchronous host to stop and join rj_vm_run() first.
  shutdown_plugin_group(vm);
  if (vm->destroy())
    delete vm;
}

rj_status_t rj_vm_step(rj_vm_t *vm, int *active) {
  if (!vm)
    return ROCJITSU_STATUS_INVALID_ARGUMENT;
  if (!vm->soc)
    return ROCJITSU_STATUS_ERROR;
  if (vm->engine_config.num_threads != 1)
    return ROCJITSU_STATUS_UNSUPPORTED;

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

  // shutdown() joins all engine workers before plugin state is torn down.
  vm->engine->shutdown();
  shutdown_plugin_group(vm);
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

rj_status_t execute_impl(SimulatedKfd *driver, uint32_t process_id, rj_vm_cmd_t *cmd) {
  auto arg_size = _IOC_SIZE(cmd->cmd);
  if (!reconstruct_embedded_pointers(cmd->cmd, cmd->buf, arg_size, cmd->buf_size)) {
    cmd->result = -EINVAL;
    cmd->shared_handle = -1;
    return ROCJITSU_STATUS_SUCCESS;
  }

  // For DBG_TRAP ENABLE the debugger's notifier pipe arrives as an SCM_RIGHTS
  // fd in cmd->in_handle (already in the daemon's fd space). Substitute it for
  // the client-side fd number in the payload so the driver signals the right
  // pipe when a wave stops (the kernel receives the same fd via the ioctl). On
  // success the debug session takes ownership (cmd->in_handle cleared so the
  // transport does not close it); otherwise the transport reclaims it. Only
  // daemon mode transfers the fd; local mode passes the debugger's own fd
  // through the interposer and leaves cmd->in_handle at -1.
  //
  // The client-supplied dbg_fd is a number in the *client's* fd table and is
  // never trusted in the daemon's namespace. Overwrite it unconditionally for
  // ENABLE: with the transferred fd when one arrived via SCM_RIGHTS, otherwise
  // with KFD_INVALID_FD so the handler's fcntl() check rejects it. Leaving the
  // client's integer in place would let a client that omits the ancillary fd
  // point the session at an arbitrary daemon-owned descriptor (a confused-deputy
  // fd substitution).
  bool adopting_notifier = false;
  if (driver->daemon_mode() && cmd->cmd == AMDKFD_IOC_DBG_TRAP) {
    auto *dbg = static_cast<kfd_ioctl_dbg_trap_args *>(cmd->buf);
    if (dbg->op == KFD_IOC_DBG_TRAP_ENABLE) {
      if (cmd->in_handle >= 0) {
        dbg->enable.dbg_fd = static_cast<uint32_t>(cmd->in_handle);
        adopting_notifier = true;
      } else {
        dbg->enable.dbg_fd = KFD_INVALID_FD;
      }
    }
  }

  cmd->result =
      driver->ioctl(process_id, cmd->cmd, cmd->buf, &cmd->in_mem_handle, cmd->in_proc_handle);
  cmd->shared_handle = -1;
  if (adopting_notifier && cmd->result == 0)
    cmd->in_handle = -1;

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
  if (!vm || !cmd || !cmd->buf || !vm->vm || !vm->vm->driver())
    return ROCJITSU_STATUS_INVALID_ARGUMENT;
  auto *driver = vm->vm->driver();
  return execute_impl(driver, driver->local_process_id(), cmd);
}

rj_status_t rj_vm_execute_as(rj_vm_t *vm, uint32_t process_id, rj_vm_cmd_t *cmd) {
  if (!vm || !cmd || !cmd->buf || !vm->vm || !vm->vm->driver())
    return ROCJITSU_STATUS_INVALID_ARGUMENT;
  return execute_impl(vm->vm->driver(), process_id, cmd);
}

rj_status_t rj_vm_device_open(rj_vm_t *vm, rj_client_pid_t client_pid, uint32_t *process_id) {
  if (!vm || !vm->vm || !vm->vm->driver())
    return ROCJITSU_STATUS_INVALID_ARGUMENT;
  auto *drv = dynamic_cast<SimulatedKfd *>(vm->vm->driver());
  if (!drv)
    return ROCJITSU_STATUS_ERROR;
  // client_pid == 0 (local mode) maps to SimulatedKfd::open_process()'s
  // default; a nonzero client_pid enables daemon-mode process reuse and
  // cross-process memory access. Narrow the fixed-width public type to the
  // platform pid_t at the Linux daemon boundary.
  uint32_t pid = drv->open_process(static_cast<pid_t>(client_pid));
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

rj_status_t rj_vm_close_all_devices(rj_vm_t *vm) {
  if (!vm || !vm->vm || !vm->vm->driver())
    return ROCJITSU_STATUS_INVALID_ARGUMENT;
  vm->vm->driver()->close_all_processes();
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
  // Capture errno HERE, immediately after the driver mmap, before any bookkeeping
  // syscall on the way back out can clobber it. Callers (the daemon RPC path) relay
  // this to the client rather than reading their own errno across the API boundary.
  map->map_errno = (result == MAP_FAILED) ? errno : 0;
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

rj_status_t rj_vm_gpu_info(rj_vm_t *vm, rj_vm_gpu_info_t *info) {
  if (!vm || !info || !vm->vm || !vm->vm->driver())
    return ROCJITSU_STATUS_INVALID_ARGUMENT;

  const auto &gpu = vm->vm->driver()->topology().gpu_info();
  *info = {};
  info->present = 1;
  info->gpu_id = gpu.gpu_id;
  info->gfx_target_version = gpu.gfx_target_version;
  info->vendor_id = gpu.vendor_id;
  info->device_id = gpu.device_id;
  info->family_id = gpu.family_id;
  info->unique_id = gpu.unique_id;
  info->location_id = gpu.location_id;
  info->domain = gpu.domain;
  info->hive_id = gpu.hive_id;
  info->drm_render_minor = gpu.drm_render_minor;
  info->revision_id = gpu.revision_id;
  info->pci_revision_id = gpu.pci_revision_id;
  info->simd_count = gpu.simd_count;
  info->max_waves_per_simd = gpu.max_waves_per_simd;
  info->num_shader_engines = gpu.num_shader_engines;
  info->num_shader_arrays_per_engine = gpu.num_shader_arrays_per_engine;
  info->num_cu_per_sh = gpu.num_cu_per_sh;
  info->simd_per_cu = gpu.simd_per_cu;
  info->wave_front_size = gpu.wave_front_size;
  info->num_xcc = gpu.num_xcc;
  info->max_slots_scratch_cu = gpu.max_slots_scratch_cu;
  info->local_mem_size = gpu.local_mem_size;
  info->vram_type = gpu.vram_type;
  info->lds_size_kb = gpu.lds_size_kb;
  info->mem_width = gpu.mem_width;
  info->mem_clk_max = gpu.mem_clk_max;
  info->l1_size_kb = gpu.l1_size_kb;
  info->l1_line_size = gpu.l1_line_size;
  info->l1_assoc = gpu.l1_assoc;
  info->l2_size_kb = gpu.l2_size_kb;
  info->l2_line_size = gpu.l2_line_size;
  info->l2_assoc = gpu.l2_assoc;
  info->num_sdma_engines = gpu.num_sdma_engines;
  info->num_sdma_xgmi_engines = gpu.num_sdma_xgmi_engines;
  info->num_cp_queues = gpu.num_cp_queues;
  info->max_engine_clk_fcompute = gpu.max_engine_clk_fcompute;
  info->capability = gpu.capability;
  info->capability2 = gpu.capability2;
  info->debug_prop = gpu.debug_prop;
  info->fw_version = gpu.fw_version;
  info->sdma_fw_version = gpu.sdma_fw_version;

  size_t name_len = gpu.marketing_name.size();
  if (name_len >= sizeof(info->marketing_name))
    name_len = sizeof(info->marketing_name) - 1;
  std::memcpy(info->marketing_name, gpu.marketing_name.data(), name_len);
  info->marketing_name[name_len] = '\0';
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
