// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "hsa/hsa_api_trace.h"
#include "rocjitsu/base/rj_compiler.h"
#include "rocjitsu/code/consan/encoding.h"
#include "rocjitsu/code/consan/image/artifact.h"
#include "rocjitsu/code/consan/target.h"
#include "rocjitsu/hooks/consan/api.h"
#include "rocjitsu/hooks/consan/runtime_core.h"
#include "rocjitsu/hooks/consan/testing.h"

#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fcntl.h>
#include <memory>
#include <mutex>
#include <sys/random.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <unordered_map>

namespace {

using namespace rocjitsu::consan;
using Blob = std::shared_ptr<std::vector<uint8_t>>;

// Preserve the previously admitted AMD table prefix. Shared headers may append
// APIs absent from the running ROCR, so do not require or copy their full size.
constexpr size_t kAmdTablePrefixBytes =
    offsetof(AmdExtTable, hsa_amd_external_semaphore_handle_close_fn) +
    sizeof(AmdExtTable::hsa_amd_external_semaphore_handle_close_fn);

// A ROCR-tool diagnostic must survive log-groups-OFF and initialization failures.
void diagnostic(std::string_view message) {
  constexpr std::string_view prefix = "[consan] ";
  for (const std::string_view part : {prefix, message, std::string_view("\n")}) {
    size_t offset = 0;
    while (offset < part.size()) {
      const ssize_t written = ::write(STDERR_FILENO, part.data() + offset, part.size() - offset);
      if (written < 0 && errno == EINTR)
        continue;
      if (written <= 0)
        return;
      offset += static_cast<size_t>(written);
    }
  }
}

struct Signal {
  bool dedicated = false;
  bool used = false;
  bool pending = false;
};

// Admitted code/binding across launches, retained through executable lifetime.
// Launch snapshots lease the immutable facts independently of this object.
struct Module {
  hsa_executable_t executable{};
  hsa_agent_t agent{};
  hsa_executable_symbol_t symbol{};
  uint64_t kernel_object = 0;
  LoadedArtifact artifact;
  std::shared_ptr<const Manifest> facts;
};

// Persistent backing capacity, not a launch's effective report length. A lease
// prevents reuse/growth until an independent snapshot succeeds (or submit rolls back).
struct ReportSlot {
  void *memory = nullptr;
  size_t capacity = 0;
  uint32_t generation = 0;
  bool leased = false;
};

// One execution's HSA resources and lifecycle. Rollback/observer cleanup releases
// the transient handles; slot indexes a separate report allocation lease.
struct Launch {
  hsa_signal_t proxy{};
  hsa_signal_t original{};
  void *wrapper = nullptr;
  void *context = nullptr;
  uint32_t slot = 0;
  std::unique_ptr<DispatchLifecycle> lifecycle;
};

class Runtime {
public:
  Runtime(const CoreApiTable &core, const AmdExtTable &amd, const char *artifacts,
          const char *capture)
      : core_(core), amd_{}, artifact_path_(artifacts), capture_path_(capture) {
    std::memcpy(&amd_, &amd, kAmdTablePrefixBytes);
  }

  const CoreApiTable &core() const { return core_; }
  const AmdExtTable &amd() const { return amd_; }

  hsa_status_t reader_memory(const void *bytes, size_t size, hsa_code_object_reader_t *reader) {
    if (bytes == nullptr || reader == nullptr || size == 0 || size > ContractLimits{}.image_bytes)
      return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    auto owned = std::make_shared<std::vector<uint8_t>>(static_cast<const uint8_t *>(bytes),
                                                        static_cast<const uint8_t *>(bytes) + size);
    const hsa_status_t status =
        core_.hsa_code_object_reader_create_from_memory_fn(owned->data(), owned->size(), reader);
    if (status == HSA_STATUS_SUCCESS) {
      const auto destroy = [this](hsa_code_object_reader_t *created) {
        (void)core_.hsa_code_object_reader_destroy_fn(*created);
        created->handle = 0;
      };
      std::unique_ptr<hsa_code_object_reader_t, decltype(destroy)> rollback(reader, destroy);
      std::lock_guard lock(mutex_);
      readers_.emplace(reader->handle, owned);
      (void)rollback.release();
    }
    return status;
  }

  hsa_status_t reader_file(hsa_file_t file, hsa_code_object_reader_t *reader) {
    struct stat info {};
    if (::fstat(file, &info) != 0 || info.st_size <= 0 ||
        uint64_t(info.st_size) > ContractLimits{}.image_bytes)
      return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    std::vector<uint8_t> bytes(static_cast<size_t>(info.st_size));
    size_t offset = 0;
    while (offset < bytes.size()) {
      const ssize_t count = ::pread(file, bytes.data() + offset, bytes.size() - offset, offset);
      if (count < 0 && errno == EINTR)
        continue;
      if (count <= 0)
        return HSA_STATUS_ERROR_INVALID_ARGUMENT;
      offset += static_cast<size_t>(count);
    }
    return reader_memory(bytes.data(), bytes.size(), reader);
  }

  hsa_status_t reader_destroy(hsa_code_object_reader_t reader) {
    const hsa_status_t status = core_.hsa_code_object_reader_destroy_fn(reader);
    if (status == HSA_STATUS_SUCCESS) {
      std::lock_guard lock(mutex_);
      readers_.erase(reader.handle);
    }
    return status;
  }

  hsa_status_t load(hsa_executable_t executable, hsa_agent_t agent, hsa_code_object_reader_t reader,
                    const char *options, hsa_loaded_code_object_t *loaded) {
    Blob original;
    {
      std::lock_guard lock(mutex_);
      const auto found = readers_.find(reader.handle);
      if (closed_ || module_ever_ || loading_ || found == readers_.end())
        return reject_locked("load requires a tracked original reader and one admitted module");
      original = found->second;
      loading_ = true;
    }
    auto module = std::make_shared<Module>();
    ContractError error;
    const std::string target = agent_target(agent);
    auto artifact = load_artifact(artifact_path_, *original, target, &error);
    if (artifact.failed()) {
      std::lock_guard lock(mutex_);
      loading_ = false;
      return reject_locked(error.message);
    }
    module->artifact = std::move(artifact.value());
    module->facts = std::make_shared<const Manifest>(module->artifact.manifest);
    module->executable = executable;
    module->agent = agent;
    hsa_code_object_reader_t prepared{};
    hsa_status_t status = core_.hsa_code_object_reader_create_from_memory_fn(
        module->artifact.prepared.data(), module->artifact.prepared.size(), &prepared);
    if (status == HSA_STATUS_SUCCESS) {
      status = core_.hsa_executable_load_agent_code_object_fn(executable, agent, prepared, options,
                                                              loaded);
      const hsa_status_t destroyed = core_.hsa_code_object_reader_destroy_fn(prepared);
      if (status == HSA_STATUS_SUCCESS && destroyed != HSA_STATUS_SUCCESS)
        status = destroyed;
    }
    std::lock_guard lock(mutex_);
    loading_ = false;
    // Keep prepared storage through executable destruction even when loader
    // failure has partially installed state in the application's executable.
    module_ = std::move(module);
    module_ever_ = true;
    if (status != HSA_STATUS_SUCCESS)
      fail_locked(RuntimeStatus::InvalidFacts, "HSA refused the matched prepared image");
    return status;
  }

  hsa_status_t freeze(hsa_executable_t executable, const char *options) {
    hsa_status_t status = core_.hsa_executable_freeze_fn(executable, options);
    if (status != HSA_STATUS_SUCCESS)
      return status;
    std::shared_ptr<Module> module;
    {
      std::lock_guard lock(mutex_);
      if (!module_ || module_->executable.handle != executable.handle || closed_)
        return reject_locked("freeze has no admitted ConSan module");
      module = module_;
    }
    // LLVM AMDHSA's .amdhsa_kernel convention, already required by shared
    // metadata admission; arbitrary descriptor symbol naming is not admitted.
    const std::string symbol = module->facts->entry.name + ".kd";
    hsa_executable_symbol_t resolved_symbol{};
    uint64_t kernel_object = 0;
    status = core_.hsa_executable_get_symbol_by_name_fn(executable, symbol.c_str(), &module->agent,
                                                        &resolved_symbol);
    uint32_t kernarg = 0, group = 0, private_bytes = 0;
    if (status == HSA_STATUS_SUCCESS)
      status = core_.hsa_executable_symbol_get_info_fn(
          resolved_symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &kernel_object);
    if (status == HSA_STATUS_SUCCESS)
      status = core_.hsa_executable_symbol_get_info_fn(
          resolved_symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE, &kernarg);
    if (status == HSA_STATUS_SUCCESS)
      status = core_.hsa_executable_symbol_get_info_fn(
          resolved_symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE, &group);
    if (status == HSA_STATUS_SUCCESS)
      status = core_.hsa_executable_symbol_get_info_fn(
          resolved_symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE, &private_bytes);
    const auto layout = plan_launch(*module->facts, {{1, 1, 1}, {1, 1, 1}}, group, private_bytes);
    std::lock_guard lock(mutex_);
    // Symbol queries run outside the lock. Finish or executable destruction may
    // have revoked this admission before the binding and workers are installed.
    if (closed_ || module_ != module)
      return reject_locked("freeze admission ended during HSA symbol queries");
    if (status != HSA_STATUS_SUCCESS || layout.failed() || kernel_object == 0 ||
        kernarg != layout.value().wrapper_bytes ||
        group != module->facts->entry.prepared_group_bytes)
      return reject_locked("loaded HSA symbol/resources do not match prepared facts");
    module->symbol = resolved_symbol;
    module->kernel_object = kernel_object;
    if (!started_ && !start_locked(module->facts))
      return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
    return HSA_STATUS_SUCCESS;
  }

  hsa_status_t symbol_info(hsa_executable_symbol_t symbol, hsa_executable_symbol_info_t attribute,
                           void *value) {
    {
      std::lock_guard lock(mutex_);
      if (module_ && symbol.handle == module_->symbol.handle && value != nullptr) {
        if (attribute == HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE) {
          std::memcpy(value, &module_->facts->entry.original_kernarg_bytes, sizeof(uint32_t));
          return HSA_STATUS_SUCCESS;
        }
        if (attribute == HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_ALIGNMENT) {
          std::memcpy(value, &module_->facts->entry.kernarg_alignment, sizeof(uint32_t));
          return HSA_STATUS_SUCCESS;
        }
      }
    }
    return core_.hsa_executable_symbol_get_info_fn(symbol, attribute, value);
  }

  hsa_status_t executable_destroy(hsa_executable_t executable) {
    {
      std::lock_guard lock(mutex_);
      if (module_ && module_->executable.handle == executable.handle && active_gpu_)
        return HSA_STATUS_ERROR_INVALID_EXECUTABLE;
    }
    const hsa_status_t status = core_.hsa_executable_destroy_fn(executable);
    if (status == HSA_STATUS_SUCCESS) {
      std::lock_guard lock(mutex_);
      if (module_ && module_->executable.handle == executable.handle)
        module_.reset();
    }
    return status;
  }

  hsa_status_t queue_create(hsa_agent_t agent, uint32_t size, hsa_queue_type32_t type,
                            void (*callback)(hsa_status_t, hsa_queue_t *, void *), void *data,
                            uint32_t private_bytes, uint32_t group, hsa_queue_t **queue) {
    {
      std::lock_guard lock(mutex_);
      if (!started_ || closed_ || queue_ever_ || !module_ || agent.handle != module_->agent.handle)
        return reject_locked("MVP requires one queue created after the prepared module is frozen");
      queue_ever_ = true;
    }
    hsa_status_t status = amd_.hsa_amd_queue_intercept_create_fn(agent, size, type, callback, data,
                                                                 private_bytes, group, queue);
    if (status != HSA_STATUS_SUCCESS)
      return status;
    status = amd_.hsa_amd_queue_intercept_register_fn(
        *queue,
        [](const void *packets, uint64_t count, uint64_t, void *context,
           hsa_amd_queue_intercept_packet_writer writer) noexcept {
          auto *runtime = static_cast<Runtime *>(context);
          runtime->intercept(packets, count, writer);
        },
        this);
    if (status == HSA_STATUS_SUCCESS)
      status = amd_.hsa_amd_profiling_set_profiler_enabled_fn(*queue, 0);
    if (status != HSA_STATUS_SUCCESS) {
      (void)core_.hsa_queue_destroy_fn(*queue);
      *queue = nullptr;
      return status;
    }
    std::lock_guard lock(mutex_);
    queue_ = *queue;
    return HSA_STATUS_SUCCESS;
  }

  hsa_status_t queue_destroy(hsa_queue_t *queue) {
    {
      std::lock_guard lock(mutex_);
      if (queue == queue_ && active_gpu_)
        return HSA_STATUS_ERROR_INVALID_QUEUE;
    }
    const hsa_status_t status = core_.hsa_queue_destroy_fn(queue);
    if (status == HSA_STATUS_SUCCESS) {
      std::lock_guard lock(mutex_);
      if (queue_ == queue)
        queue_ = nullptr;
    }
    return status;
  }

  hsa_status_t signal_create(hsa_signal_value_t value, uint32_t consumers,
                             const hsa_agent_t *agents, hsa_signal_t *signal) {
    const hsa_status_t status = core_.hsa_signal_create_fn(value, consumers, agents, signal);
    if (status == HSA_STATUS_SUCCESS) {
      std::lock_guard lock(mutex_);
      signals_[signal->handle] = {.dedicated = value == 1};
    }
    return status;
  }

  hsa_status_t signal_destroy(hsa_signal_t signal) {
    {
      std::lock_guard lock(mutex_);
      const auto found = signals_.find(signal.handle);
      if (found != signals_.end() && found->second.pending)
        return HSA_STATUS_ERROR_INVALID_SIGNAL;
    }
    const hsa_status_t status = core_.hsa_signal_destroy_fn(signal);
    if (status == HSA_STATUS_SUCCESS) {
      std::lock_guard lock(mutex_);
      signals_.erase(signal.handle);
    }
    return status;
  }

  int drain() {
    std::unique_lock lock(mutex_);
    const uint64_t target = statistics_.submitted;
    if (!condition_.wait_for(lock, std::chrono::seconds(30),
                             [&] { return processed_ >= target; })) {
      fail_locked(RuntimeStatus::PendingCompletion,
                  "drain has no completion/copy proof; GPU resources remain retained");
      return static_cast<int>(status_);
    }
    return static_cast<int>(status_);
  }

  int finish() {
    // Serialize the whole drain/join/publication operation. The observer uses
    // mutex_, so it remains able to forward completion while finish waits.
    std::lock_guard finishing(finish_mutex_);
    {
      std::lock_guard lock(mutex_);
      if (finished_)
        return static_cast<int>(status_);
      closed_ = true;
      if (!started_) {
        if (output_ >= 0) {
          const int closed = ::close(output_);
          output_ = -1;
          if (closed != 0)
            fail_locked(RuntimeStatus::CloseFailed, "failed partial capture close");
        }
        finished_ = true;
        if (status_ == RuntimeStatus::Success)
          fail_locked(RuntimeStatus::InvalidFacts, "finish has no admitted artifact/capture");
        return static_cast<int>(status_);
      }
    }
    (void)drain();
    {
      std::lock_guard lock(mutex_);
      if (processed_ != statistics_.submitted || active_gpu_)
        return static_cast<int>(status_);
      stop_ = true;
      condition_.notify_all();
    }
    if (observer_.joinable())
      observer_.join();
    if (collector_.joinable())
      collector_.join();
    else {
      writer_.cancel(status_);
      (void)writer_.finish();
    }
    std::lock_guard lock(mutex_);
    for (ReportSlot &slot : slots_) {
      if (slot.memory != nullptr) {
        (void)amd_.hsa_amd_memory_pool_free_fn(slot.memory);
        slot.memory = nullptr;
      }
    }
    finished_ = true;
    return static_cast<int>(status_);
  }

  bool finished() {
    std::lock_guard lock(mutex_);
    return finished_;
  }
  rj_consan_statistics statistics() {
    std::lock_guard lock(mutex_);
    return statistics_;
  }

  bool abandon_unload() {
    {
      std::lock_guard lock(mutex_);
      if (active_gpu_ || processed_ != statistics_.submitted)
        return false;
      status_ = RuntimeStatus::InvalidTransition;
      closed_ = stop_ = true;
      condition_.notify_all();
    }
    if (observer_.joinable())
      observer_.join();
    if (collector_.joinable())
      collector_.join();
    else if (output_ >= 0) {
      const int closed = ::close(output_);
      output_ = -1;
      if (closed != 0)
        diagnostic("failed abandoned partial capture close");
    }
    return true;
  }

private:
  std::string agent_target(hsa_agent_t agent) {
    hsa_device_type_t type{};
    hsa_isa_t isa{};
    uint32_t length = 0;
    if (core_.hsa_agent_get_info_fn(agent, HSA_AGENT_INFO_DEVICE, &type) != HSA_STATUS_SUCCESS ||
        type != HSA_DEVICE_TYPE_GPU ||
        core_.hsa_agent_get_info_fn(agent, HSA_AGENT_INFO_ISA, &isa) != HSA_STATUS_SUCCESS ||
        core_.hsa_isa_get_info_alt_fn(isa, HSA_ISA_INFO_NAME_LENGTH, &length) !=
            HSA_STATUS_SUCCESS ||
        length == 0 || length > 256)
      return {};
    std::vector<char> name(length);
    if (core_.hsa_isa_get_info_alt_fn(isa, HSA_ISA_INFO_NAME, name.data()) != HSA_STATUS_SUCCESS ||
        name.back() != '\0')
      return {};
    constexpr std::string_view prefix = "amdgcn-amd-amdhsa--";
    std::string_view isa_name(name.data());
    if (!isa_name.starts_with(prefix))
      return {};
    isa_name.remove_prefix(prefix.size());
    isa_name = isa_name.substr(0, isa_name.find(':'));
    const auto *target = target_profile(isa_name);
    return target == nullptr ? std::string{} : std::string(target->name);
  }

  void fail_locked(RuntimeStatus status, std::string_view message) {
    if (status_ == RuntimeStatus::Success)
      status_ = status;
    diagnostic(message);
  }
  hsa_status_t reject_locked(std::string_view message) {
    fail_locked(RuntimeStatus::InvalidFacts, message);
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  bool find_pool_locked() {
    const hsa_status_t status = core_.hsa_iterate_agents_fn(
        [](hsa_agent_t agent, void *context) {
          auto &runtime = *static_cast<Runtime *>(context);
          hsa_device_type_t type{};
          if (runtime.core_.hsa_agent_get_info_fn(agent, HSA_AGENT_INFO_DEVICE, &type) !=
                  HSA_STATUS_SUCCESS ||
              type != HSA_DEVICE_TYPE_CPU)
            return HSA_STATUS_SUCCESS;
          runtime.amd_.hsa_amd_agent_iterate_memory_pools_fn(
              agent,
              [](hsa_amd_memory_pool_t pool, void *data) {
                auto &owner = *static_cast<Runtime *>(data);
                uint32_t segment = UINT32_MAX, flags = 0;
                bool allowed = false;
                if (owner.amd_.hsa_amd_memory_pool_get_info_fn(
                        pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &segment) == HSA_STATUS_SUCCESS &&
                    segment == 0 &&
                    owner.amd_.hsa_amd_memory_pool_get_info_fn(
                        pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &flags) ==
                        HSA_STATUS_SUCCESS &&
                    owner.amd_.hsa_amd_memory_pool_get_info_fn(
                        pool, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED, &allowed) ==
                        HSA_STATUS_SUCCESS &&
                    allowed &&
                    (flags & (HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED |
                              HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT)) ==
                        (HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED |
                         HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT)) {
                  owner.pool_ = pool;
                  return HSA_STATUS_INFO_BREAK;
                }
                return HSA_STATUS_SUCCESS;
              },
              &runtime);
          return runtime.pool_.handle != 0 ? HSA_STATUS_INFO_BREAK : HSA_STATUS_SUCCESS;
        },
        this);
    return (status == HSA_STATUS_SUCCESS || status == HSA_STATUS_INFO_BREAK) && pool_.handle != 0;
  }

  bool allocate_locked(size_t bytes, void **memory) {
    if (amd_.hsa_amd_memory_pool_allocate_fn(pool_, bytes, 0, memory) != HSA_STATUS_SUCCESS)
      return false;
    if (amd_.hsa_amd_agents_allow_access_fn(1, &module_->agent, nullptr, *memory) !=
        HSA_STATUS_SUCCESS) {
      (void)amd_.hsa_amd_memory_pool_free_fn(*memory);
      *memory = nullptr;
      return false;
    }
    return true;
  }

  bool host_arguments_locked(const void *pointer, size_t size) {
    hsa_amd_pointer_info_t info{};
    info.size = sizeof(info);
    if (pointer == nullptr || size > CaptureLimits{}.argument_bytes ||
        amd_.hsa_amd_pointer_info_fn(pointer, &info, nullptr, nullptr, nullptr) !=
            HSA_STATUS_SUCCESS ||
        info.size < offsetof(hsa_amd_pointer_info_t, global_flags) + sizeof(info.global_flags) ||
        info.type != HSA_EXT_POINTER_TYPE_HSA || info.hostBaseAddress == nullptr ||
        info.agentBaseAddress != info.hostBaseAddress ||
        (info.global_flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED) == 0)
      return false;
    const uintptr_t base = reinterpret_cast<uintptr_t>(info.hostBaseAddress);
    const uintptr_t address = reinterpret_cast<uintptr_t>(pointer);
    return address >= base && address - base <= info.sizeInBytes &&
           size <= info.sizeInBytes - (address - base);
  }

  bool start_locked(std::shared_ptr<const Manifest> facts) {
    if (!find_pool_locked()) {
      fail_locked(RuntimeStatus::InvalidFacts, "CPU fine-grained kernarg pool unavailable");
      return false;
    }
    CaptureHeader header;
    header.manifest = *facts;
    const auto encoded_facts = encode_manifest(*facts);
    if (encoded_facts.failed()) {
      fail_locked(RuntimeStatus::InvalidFacts, "cannot encode capture facts");
      return false;
    }
    planned_capture_bytes_ = 64 + encoded_facts.value().size() + 64;
    size_t offset = 0;
    while (offset < header.run_identity.size()) {
      const ssize_t count =
          ::getrandom(header.run_identity.data() + offset, header.run_identity.size() - offset, 0);
      if (count < 0 && errno == EINTR)
        continue;
      if (count <= 0) {
        fail_locked(RuntimeStatus::InvalidFacts, "cannot obtain a capture run identity");
        return false;
      }
      offset += static_cast<size_t>(count);
    }
    partial_path_ = capture_path_ + ".partial." + std::to_string(::getpid());
    output_ = ::open(partial_path_.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    if (output_ < 0) {
      fail_locked(RuntimeStatus::WriteFailed, "cannot create the partial capture file");
      return false;
    }
    CaptureSink sink{
        .write = [this](std::span<const uint8_t> bytes) -> int64_t {
          ssize_t count;
          do {
            count = ::write(output_, bytes.data(), bytes.size());
          } while (count < 0 && errno == EINTR);
          return count;
        },
        .close =
            [this] {
              const bool flushed = ::fsync(output_) == 0;
              const bool closed = ::close(output_) == 0;
              output_ = -1;
              return flushed && closed;
            },
        .publish =
            [this] {
              // link publishes without overwriting an existing user file. Only the
              // already-closed complete inode can receive the final capture name.
              if (::link(partial_path_.c_str(), capture_path_.c_str()) != 0)
                return false;
              (void)::unlink(partial_path_.c_str());
              return true;
            },
    };
    const RuntimeStatus status = writer_.start(header, std::move(sink));
    if (status != RuntimeStatus::Success) {
      writer_.cancel(status);
      (void)writer_.finish();
      fail_locked(status, "capture prefix write failed");
      return false;
    }
    started_ = true;
    // Uncaught exceptions terminate the process: merely stopping a worker
    // cannot recover an interrupted ownership handoff. Status failures remain
    // handled by the loops without requiring exception recovery.
    observer_ = std::thread([this] { observer_loop(); });
    collector_ = std::thread([this] { collector_loop(); });
    return true;
  }

  void fail_dispatch_completion_locked(const hsa_kernel_dispatch_packet_t &packet) {
    if ((packet.header & 0xff) != HSA_PACKET_TYPE_KERNEL_DISPATCH)
      return;
    const auto found = signals_.find(packet.completion_signal.handle);
    if (found != signals_.end() && !found->second.pending)
      core_.hsa_signal_store_screlease_fn(packet.completion_signal, -1);
  }

  void reject_packet_locked(const hsa_kernel_dispatch_packet_t &packet, std::string_view message) {
    fail_locked(RuntimeStatus::InvalidFacts, message);
    fail_dispatch_completion_locked(packet);
  }

  void release_unsubmitted_locked(Launch &launch) {
    if (launch.proxy.handle)
      (void)core_.hsa_signal_destroy_fn(launch.proxy);
    if (launch.wrapper)
      (void)amd_.hsa_amd_memory_pool_free_fn(launch.wrapper);
    if (launch.context)
      (void)amd_.hsa_amd_memory_pool_free_fn(launch.context);
    if (launch.slot)
      slots_[launch.slot - 1].leased = false;
  }

  void intercept(const void *packets, uint64_t count,
                 hsa_amd_queue_intercept_packet_writer writer) {
    if (packets == nullptr || writer == nullptr || count == 0)
      return;
    hsa_kernel_dispatch_packet_t packet{};
    std::memcpy(&packet, packets, sizeof(packet));
    const uint16_t type = packet.header & 0xff;
    if (count == 1 && (type == HSA_PACKET_TYPE_BARRIER_AND || type == HSA_PACKET_TYPE_BARRIER_OR)) {
      writer(packets, count);
      return;
    }
    if (count != 1 || type != HSA_PACKET_TYPE_KERNEL_DISPATCH) {
      std::lock_guard lock(mutex_);
      fail_locked(RuntimeStatus::InvalidFacts, "only individual kernel dispatches are admitted");
      const auto *bytes = static_cast<const uint8_t *>(packets);
      for (uint64_t index = 0; index < count; ++index) {
        std::memcpy(&packet, bytes + index * sizeof(packet), sizeof(packet));
        fail_dispatch_completion_locked(packet);
      }
      return;
    }
    auto launch = std::make_shared<Launch>();
    {
      std::lock_guard lock(mutex_);
      const auto release = [this](Launch *unsubmitted) {
        release_unsubmitted_locked(*unsubmitted);
      };
      // The shared Launch owns its lifetime; this guard owns only rollback of
      // acquired HSA resources until the complete packet becomes publishable.
      std::unique_ptr<Launch, decltype(release)> rollback(launch.get(), release);
      // Admit the binding, nonoverlap, dedicated signal, and bounded launch
      // before allocating resources or exposing any packet to HSA.
      const auto signal = signals_.find(packet.completion_signal.handle);
      if (packet.setup != 1 || closed_ || status_ != RuntimeStatus::Success ||
          statistics_.submitted >= CaptureLimits{}.executions || !module_ ||
          packet.kernel_object != module_->kernel_object || active_gpu_ ||
          signal == signals_.end() || !signal->second.dedicated || signal->second.used ||
          core_.hsa_signal_load_scacquire_fn(packet.completion_signal) != 1) {
        reject_packet_locked(
            packet,
            "packet violates binding, single-queue/nonoverlap or dedicated-signal requirements");
        return;
      }
      const Manifest &facts = *module_->facts;
      const LaunchDimensions dimensions{
          {packet.grid_size_x, packet.grid_size_y, packet.grid_size_z},
          {packet.workgroup_size_x, packet.workgroup_size_y, packet.workgroup_size_z}};
      const auto layout =
          plan_launch(facts, dimensions, packet.group_segment_size, packet.private_segment_size);
      if (layout.failed() ||
          !host_arguments_locked(packet.kernarg_address, facts.entry.original_kernarg_bytes)) {
        reject_packet_locked(
            packet,
            "unsupported geometry/segments or non-fine-grained original kernarg allocation");
        return;
      }
      const uint64_t capture_bytes =
          48 + 144 + uint64_t{facts.entry.original_kernarg_bytes} + layout.value().report_bytes;
      if (capture_bytes > CaptureLimits{}.file_bytes - planned_capture_bytes_) {
        reject_packet_locked(packet, "capture budget exhausted before submission");
        return;
      }
      // Prepare sufficient idle report capacity and per-launch wrapper/context
      // plus proxy signal. Rollback owns all unsubmitted acquisitions.
      for (size_t index = 0; index < slots_.size(); ++index) {
        if (!slots_[index].leased) {
          launch->slot = static_cast<uint32_t>(index + 1);
          break;
        }
      }
      if (launch->slot == 0 && slots_.size() < 16) {
        ReportSlot slot;
        slot.capacity = layout.value().report_bytes;
        if (allocate_locked(slot.capacity, &slot.memory)) {
          const auto free = [this](void *memory) {
            (void)amd_.hsa_amd_memory_pool_free_fn(memory);
          };
          std::unique_ptr<void, decltype(free)> uninstalled(slot.memory, free);
          slots_.push_back(slot);
          (void)uninstalled.release();
          launch->slot = static_cast<uint32_t>(slots_.size());
        }
      }
      if (launch->slot == 0) {
        reject_packet_locked(packet,
                             "report allocation/backlog admission failed before submission");
        return;
      }
      ReportSlot &slot = slots_[launch->slot - 1];
      const bool reused = slot.generation != 0 && slot.capacity >= layout.value().report_bytes;
      if (slot.capacity < layout.value().report_bytes) {
        void *replacement = nullptr;
        if (!allocate_locked(layout.value().report_bytes, &replacement)) {
          reject_packet_locked(packet, "report growth allocation failed before submission");
          return;
        }
        // Only an unleased slot reaches here: completion and an independent
        // snapshot already permit replacement. Keep the old allocation on failure.
        (void)amd_.hsa_amd_memory_pool_free_fn(slot.memory);
        slot.memory = replacement;
        slot.capacity = layout.value().report_bytes;
      }
      slot.leased = true;
      if (slot.generation == UINT32_MAX || (reinterpret_cast<uintptr_t>(slot.memory) % 64) != 0 ||
          !allocate_locked(layout.value().wrapper_bytes, &launch->wrapper) ||
          !allocate_locked(kContextBytes, &launch->context) ||
          core_.hsa_signal_create_fn(1, 0, nullptr, &launch->proxy) != HSA_STATUS_SUCCESS) {
        reject_packet_locked(packet, "wrapper/context/proxy allocation failed before submission");
        return;
      }
      const uint64_t serial = statistics_.submitted + 1;
      const auto transport = initialize_transport(facts, dimensions, layout.value().group_bytes,
                                                  layout.value().private_bytes, serial,
                                                  reinterpret_cast<uint64_t>(slot.memory));
      if (transport.failed()) {
        reject_packet_locked(packet, "canonical transport initialization failed before submission");
        return;
      }
      std::memcpy(slot.memory, transport.value().report.data(), transport.value().report.size());
      std::memcpy(launch->context, transport.value().context.data(), kContextBytes);
      auto wrapper =
          std::span(static_cast<uint8_t *>(launch->wrapper), layout.value().wrapper_bytes);
      const std::optional<rocjitsu::KernargExtensionLayout> kernarg =
          rocjitsu::make_kernarg_extension_layout(facts.entry.original_kernarg_bytes,
                                                  kContextPointerPayloads);
      const uint64_t context_pointer = reinterpret_cast<uint64_t>(launch->context);
      const std::array<rocjitsu::KernargExtensionPayloadWrite, 1> payloads = {
          {{.data = &context_pointer, .size = sizeof(context_pointer)}}};
      if (!kernarg || !rocjitsu::write_kernarg_extension_wrapper(
                          wrapper, *kernarg, packet.kernarg_address,
                          reinterpret_cast<uint64_t>(packet.kernarg_address), payloads)) {
        reject_packet_locked(packet, "kernarg wrapper initialization failed before submission");
        return;
      }
      // Bind execution identity and original arguments to an empty report
      // envelope. Lifecycle admission does not itself submit the HSA packet.
      CapturedExecution envelope;
      envelope.serial = serial;
      envelope.dimensions = dimensions;
      envelope.group_bytes = layout.value().group_bytes;
      envelope.private_bytes = layout.value().private_bytes;
      envelope.storage_slot = launch->slot;
      envelope.storage_generation = ++slot.generation;
      envelope.completion = 0;
      envelope.context = transport.value().context;
      envelope.arguments.assign(wrapper.begin(),
                                wrapper.begin() + facts.entry.original_kernarg_bytes);
      launch->original = packet.completion_signal;
      launch->lifecycle = std::make_unique<DispatchLifecycle>(module_->facts, std::move(envelope));
      if (launch->lifecycle->submitted() != RuntimeStatus::Success) {
        reject_packet_locked(packet, "dispatch ownership initialization failed before submission");
        return;
      }
      packet.kernarg_address = launch->wrapper;
      packet.completion_signal = launch->proxy;
      packet.group_segment_size = layout.value().group_bytes;
      packet.private_segment_size = layout.value().private_bytes;
      constexpr uint16_t fences = (3u << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE) |
                                  (3u << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE);
      packet.header = (packet.header & ~fences) |
                      (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE) |
                      (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE);
      pending_.push_back(launch);
      signal->second.used = signal->second.pending = true;
      active_gpu_ = true;
      ++statistics_.submitted;
      planned_capture_bytes_ += capture_bytes;
      if (reused)
        ++statistics_.reused;
      (void)rollback.release();
      condition_.notify_all();
    }
    std::atomic_thread_fence(std::memory_order_release);
    // This is the HSA packet writer, not the collector's capture-file writer.
    // Resources, lifecycle, and observer ownership are ready before publication.
    writer(&packet, 1);
  }

  void observer_loop() {
    for (;;) {
      std::shared_ptr<Launch> launch;
      {
        std::unique_lock lock(mutex_);
        condition_.wait(lock, [&] { return stop_ || !pending_.empty(); });
        if (stop_ && pending_.empty())
          return;
        launch = pending_.front();
      }
      const hsa_signal_value_t observed = core_.hsa_signal_wait_scacquire_fn(
          launch->proxy, HSA_SIGNAL_CONDITION_LT, 1, 1000000, HSA_WAIT_STATE_BLOCKED);
      if (observed != 0)
        continue; // A timeout/fault is never completion/visibility proof.
      {
        std::lock_guard lock(mutex_);
        const RuntimeStatus status = launch->lifecycle->observe(true, true, [&] {
          core_.hsa_signal_subtract_screlease_fn(launch->original, 1);
          return true;
        });
        if (status != RuntimeStatus::Success)
          fail_locked(status, "completion forwarding failed");
        signals_[launch->original.handle].pending = false;
        active_gpu_ = false;
        ++statistics_.forwarded;
        pending_.pop_front();
        // The collector may take this reference before the cleanup below.
        // Our local shared_ptr keeps Launch alive while both workers use it;
        // it does not synchronize mutable lifecycle state.
        completed_.push_back(launch);
        condition_.notify_all();
      }
      (void)core_.hsa_signal_destroy_fn(launch->proxy);
      (void)amd_.hsa_amd_memory_pool_free_fn(launch->wrapper);
      (void)amd_.hsa_amd_memory_pool_free_fn(launch->context);
    }
  }

  void collector_loop() {
    for (;;) {
      std::shared_ptr<Launch> launch;
      {
        std::unique_lock lock(mutex_);
        condition_.wait(lock, [&] { return stop_ || !completed_.empty(); });
        if (stop_ && completed_.empty())
          break;
        launch = completed_.front();
        completed_.pop_front();
      }
      OwnedSnapshot snapshot;
      void *source = nullptr;
      {
        std::lock_guard lock(mutex_);
        source = slots_[launch->slot - 1].memory;
      }
      const RuntimeStatus copied = launch->lifecycle->snapshot(
          [source](std::span<uint8_t> bytes) {
            std::memcpy(bytes.data(), source, bytes.size());
            return CopyResult{RuntimeStatus::Success, bytes.size()};
          },
          snapshot);
      {
        std::lock_guard lock(mutex_);
        if (copied == RuntimeStatus::Success) {
          slots_[launch->slot - 1].leased = false;
          ++statistics_.copied;
        } else
          fail_locked(copied, "independent host snapshot failed; slice remains retained");
      }
      // The slice can be leased again before this blocking write. The writer
      // receives only independent bytes and the original immutable facts lease.
      const RuntimeStatus written =
          copied == RuntimeStatus::Success ? writer_.append(snapshot) : copied;
      {
        std::lock_guard lock(mutex_);
        if (written == RuntimeStatus::Success)
          ++statistics_.written;
        else
          fail_locked(written, "capture execution write failed");
        ++processed_;
        condition_.notify_all();
      }
    }
    RuntimeStatus status;
    {
      std::lock_guard lock(mutex_);
      status = status_;
    }
    if (status != RuntimeStatus::Success)
      writer_.cancel(status);
    const RuntimeStatus closed = writer_.finish();
    if (closed != RuntimeStatus::Success) {
      std::lock_guard lock(mutex_);
      fail_locked(closed, "capture End/close/publication failed");
    }
  }

  // Saved, unhooked function tables used for forwarding and conditional restore.
  CoreApiTable core_;
  AmdExtTable amd_;
  std::string artifact_path_, capture_path_, partial_path_;
  std::mutex mutex_;
  std::mutex finish_mutex_;
  std::condition_variable condition_;
  std::unordered_map<uint64_t, Blob> readers_;
  std::unordered_map<uint64_t, Signal> signals_;
  std::shared_ptr<Module> module_;
  hsa_amd_memory_pool_t pool_{};
  hsa_queue_t *queue_ = nullptr;
  std::vector<ReportSlot> slots_;
  std::deque<std::shared_ptr<Launch>> pending_, completed_;
  std::thread observer_, collector_;
  SnapshotWriter writer_;
  rj_consan_statistics statistics_{};
  // Collector-handled launches, including failed snapshot/write attempts.
  uint64_t processed_ = 0;
  // Cumulative file reservation: prefix, End and all admitted execution frames,
  // including those not yet written. Storage reuse does not release this budget.
  uint64_t planned_capture_bytes_ = 0;
  RuntimeStatus status_ = RuntimeStatus::Success;
  // closed_ ends admission; stop_ requests worker exit; finished_ ends the finish
  // operation. Whether that operation succeeded is retained separately in status_.
  bool loading_ = false, started_ = false, closed_ = false, stop_ = false, finished_ = false,
       active_gpu_ = false, queue_ever_ = false, module_ever_ = false;
  int output_ = -1;
};

std::unique_ptr<Runtime> runtime;
// Non-owning ROCR core/AMD-extension tables, valid through OnUnload. Runtime
// keeps original copies; these pointers identify the active tables we patched.
CoreApiTable *installed_core = nullptr;
AmdExtTable *installed_amd = nullptr;

// Ordinary errors retain their status paths. Unexpected C++ exceptions are
// fatal rather than escaping into ROCR or implying a recoverable runtime state.
template <typename Function> hsa_status_t boundary(Function &&function) noexcept {
  return function();
}

hsa_status_t HSA_API reader_memory(const void *bytes, size_t size,
                                   hsa_code_object_reader_t *reader) {
  return boundary([&] { return runtime->reader_memory(bytes, size, reader); });
}
hsa_status_t HSA_API reader_file(hsa_file_t file, hsa_code_object_reader_t *reader) {
  return boundary([&] { return runtime->reader_file(file, reader); });
}
hsa_status_t HSA_API reader_destroy(hsa_code_object_reader_t reader) {
  return boundary([&] { return runtime->reader_destroy(reader); });
}
hsa_status_t HSA_API load(hsa_executable_t executable, hsa_agent_t agent,
                          hsa_code_object_reader_t reader, const char *options,
                          hsa_loaded_code_object_t *loaded) {
  return boundary([&] { return runtime->load(executable, agent, reader, options, loaded); });
}
hsa_status_t HSA_API freeze(hsa_executable_t executable, const char *options) {
  return boundary([&] { return runtime->freeze(executable, options); });
}
hsa_status_t HSA_API symbol_info(hsa_executable_symbol_t symbol,
                                 hsa_executable_symbol_info_t attribute, void *value) {
  return boundary([&] { return runtime->symbol_info(symbol, attribute, value); });
}
hsa_status_t HSA_API executable_destroy(hsa_executable_t executable) {
  return boundary([&] { return runtime->executable_destroy(executable); });
}
hsa_status_t HSA_API queue_create(hsa_agent_t agent, uint32_t size, hsa_queue_type32_t type,
                                  void (*callback)(hsa_status_t, hsa_queue_t *, void *), void *data,
                                  uint32_t private_bytes, uint32_t group, hsa_queue_t **queue) {
  return boundary([&] {
    return runtime->queue_create(agent, size, type, callback, data, private_bytes, group, queue);
  });
}
hsa_status_t HSA_API queue_destroy(hsa_queue_t *queue) {
  return boundary([&] { return runtime->queue_destroy(queue); });
}
hsa_status_t HSA_API signal_create(hsa_signal_value_t value, uint32_t consumers,
                                   const hsa_agent_t *agents, hsa_signal_t *signal) {
  return boundary([&] { return runtime->signal_create(value, consumers, agents, signal); });
}
hsa_status_t HSA_API signal_destroy(hsa_signal_t signal) {
  return boundary([&] { return runtime->signal_destroy(signal); });
}
hsa_status_t HSA_API profiling(hsa_queue_t *, int enabled) {
  if (enabled != 0)
    diagnostic("ConSan MVP requires profiling disabled");
  return enabled == 0 ? HSA_STATUS_SUCCESS : HSA_STATUS_ERROR_INVALID_ARGUMENT;
}
hsa_status_t HSA_API shutdown() noexcept {
  if (!runtime->finished()) {
    diagnostic("call rj_consan_finish before hsa_shut_down");
    return HSA_STATUS_ERROR;
  }
  return runtime->core().hsa_shut_down_fn();
}
hsa_status_t HSA_API deprecated_load(hsa_executable_t, hsa_agent_t, hsa_code_object_t,
                                     const char *) {
  diagnostic("ConSan MVP requires the code-object reader load API");
  return HSA_STATUS_ERROR_INVALID_ARGUMENT;
}
hsa_status_t HSA_API program_load(hsa_executable_t, hsa_code_object_reader_t, const char *,
                                  hsa_loaded_code_object_t *) {
  diagnostic("ConSan MVP does not admit program code objects");
  return HSA_STATUS_ERROR_INVALID_ARGUMENT;
}
hsa_status_t HSA_API intercept_create(hsa_agent_t, uint32_t, hsa_queue_type32_t,
                                      void (*)(hsa_status_t, hsa_queue_t *, void *), void *,
                                      uint32_t, uint32_t, hsa_queue_t **) {
  diagnostic("create the controlled ConSan queue through hsa_queue_create");
  return HSA_STATUS_ERROR_INVALID_ARGUMENT;
}
hsa_status_t HSA_API intercept_register(hsa_queue_t *, hsa_amd_queue_intercept_handler, void *) {
  diagnostic("ConSan owns the controlled queue's packet interceptor");
  return HSA_STATUS_ERROR_INVALID_ARGUMENT;
}

bool install(HsaApiTable *table) {
  if (runtime || table == nullptr || table->core_ == nullptr || table->amd_ext_ == nullptr ||
      table->core_->version.minor_id < sizeof(CoreApiTable) ||
      table->amd_ext_->version.minor_id < kAmdTablePrefixBytes)
    return false;
  const auto &core = *table->core_;
  const auto &amd = *table->amd_ext_;
  if (!core.hsa_shut_down_fn || !core.hsa_code_object_reader_create_from_memory_fn ||
      !core.hsa_code_object_reader_destroy_fn || !core.hsa_executable_load_agent_code_object_fn ||
      !core.hsa_executable_freeze_fn || !core.hsa_executable_destroy_fn ||
      !core.hsa_executable_symbol_get_info_fn || !core.hsa_executable_get_symbol_by_name_fn ||
      !core.hsa_queue_destroy_fn || !core.hsa_signal_create_fn || !core.hsa_signal_destroy_fn ||
      !core.hsa_signal_load_scacquire_fn || !core.hsa_signal_store_screlease_fn ||
      !core.hsa_signal_subtract_screlease_fn || !core.hsa_signal_wait_scacquire_fn ||
      !core.hsa_iterate_agents_fn || !core.hsa_agent_get_info_fn || !core.hsa_isa_get_info_alt_fn ||
      !amd.hsa_amd_agent_iterate_memory_pools_fn || !amd.hsa_amd_memory_pool_get_info_fn ||
      !amd.hsa_amd_memory_pool_allocate_fn || !amd.hsa_amd_memory_pool_free_fn ||
      !amd.hsa_amd_agents_allow_access_fn || !amd.hsa_amd_pointer_info_fn ||
      !amd.hsa_amd_queue_intercept_create_fn || !amd.hsa_amd_queue_intercept_register_fn ||
      !amd.hsa_amd_profiling_set_profiler_enabled_fn)
    return false;
  const char *artifacts = std::getenv("CONSAN_ARTIFACT_DIR");
  const char *capture = std::getenv("CONSAN_CAPTURE_PATH");
  if (!artifacts || !*artifacts || !capture || !*capture) {
    diagnostic("CONSAN_ARTIFACT_DIR and CONSAN_CAPTURE_PATH are required");
    return false;
  }
  runtime = std::make_unique<Runtime>(core, amd, artifacts, capture);
  auto &patched = *table->core_;
  patched.hsa_code_object_reader_create_from_memory_fn = reader_memory;
  patched.hsa_code_object_reader_create_from_file_fn = reader_file;
  patched.hsa_code_object_reader_destroy_fn = reader_destroy;
  patched.hsa_executable_load_agent_code_object_fn = load;
  patched.hsa_executable_load_code_object_fn = deprecated_load;
  patched.hsa_executable_load_program_code_object_fn = program_load;
  patched.hsa_executable_freeze_fn = freeze;
  patched.hsa_executable_symbol_get_info_fn = symbol_info;
  patched.hsa_executable_destroy_fn = executable_destroy;
  patched.hsa_queue_create_fn = queue_create;
  patched.hsa_queue_destroy_fn = queue_destroy;
  patched.hsa_signal_create_fn = signal_create;
  patched.hsa_signal_destroy_fn = signal_destroy;
  patched.hsa_shut_down_fn = shutdown;
  table->amd_ext_->hsa_amd_profiling_set_profiler_enabled_fn = profiling;
  table->amd_ext_->hsa_amd_queue_intercept_create_fn = intercept_create;
  table->amd_ext_->hsa_amd_queue_intercept_register_fn = intercept_register;
  installed_core = table->core_;
  installed_amd = table->amd_ext_;
  return true;
}

} // namespace

extern "C" RJ_API_EXPORT bool OnLoad(HsaApiTable *table, uint64_t, uint64_t,
                                     const char *const *) noexcept {
  return install(table);
}

extern "C" RJ_API_EXPORT int rj_consan_drain() noexcept {
  return runtime ? runtime->drain() : int(RuntimeStatus::InvalidTransition);
}
extern "C" RJ_API_EXPORT int rj_consan_finish() noexcept {
  return runtime ? runtime->finish() : int(RuntimeStatus::InvalidTransition);
}
extern "C" RJ_API_EXPORT rj_consan_statistics rj_consan_get_statistics() noexcept {
  return runtime ? runtime->statistics() : rj_consan_statistics{};
}

extern "C" RJ_API_EXPORT void OnUnload() noexcept {
  if (!runtime)
    return;
  const bool retain = !runtime->finished();
  if (retain) {
    diagnostic("ConSan unloaded without finish; capture is unpublished and HSA allocations remain "
               "retained");
    if (!runtime->abandon_unload()) {
      diagnostic("cannot unload ConSan code while GPU/collector work still uses it");
      std::abort();
    }
  }
  if (installed_core && installed_amd) {
    // ROCR owns these tables through OnUnload. Restore only our installed
    // functions, without replacing hooks another tool installed after ours.
    auto &core = *installed_core;
    const auto &saved = runtime->core();
#define RJ_CONSAN_RESTORE_HOOK(table, original, field, hook)                                       \
  do {                                                                                             \
    if ((table).field == (hook))                                                                   \
      (table).field = (original).field;                                                            \
  } while (false)
    RJ_CONSAN_RESTORE_HOOK(core, saved, hsa_code_object_reader_create_from_memory_fn,
                           reader_memory);
    RJ_CONSAN_RESTORE_HOOK(core, saved, hsa_code_object_reader_create_from_file_fn, reader_file);
    RJ_CONSAN_RESTORE_HOOK(core, saved, hsa_code_object_reader_destroy_fn, reader_destroy);
    RJ_CONSAN_RESTORE_HOOK(core, saved, hsa_executable_load_agent_code_object_fn, load);
    RJ_CONSAN_RESTORE_HOOK(core, saved, hsa_executable_load_code_object_fn, deprecated_load);
    RJ_CONSAN_RESTORE_HOOK(core, saved, hsa_executable_load_program_code_object_fn, program_load);
    RJ_CONSAN_RESTORE_HOOK(core, saved, hsa_executable_freeze_fn, freeze);
    RJ_CONSAN_RESTORE_HOOK(core, saved, hsa_executable_symbol_get_info_fn, symbol_info);
    RJ_CONSAN_RESTORE_HOOK(core, saved, hsa_executable_destroy_fn, executable_destroy);
    RJ_CONSAN_RESTORE_HOOK(core, saved, hsa_queue_create_fn, queue_create);
    RJ_CONSAN_RESTORE_HOOK(core, saved, hsa_queue_destroy_fn, queue_destroy);
    RJ_CONSAN_RESTORE_HOOK(core, saved, hsa_signal_create_fn, signal_create);
    RJ_CONSAN_RESTORE_HOOK(core, saved, hsa_signal_destroy_fn, signal_destroy);
    RJ_CONSAN_RESTORE_HOOK(core, saved, hsa_shut_down_fn, shutdown);
    auto &amd = *installed_amd;
    const auto &original = runtime->amd();
    RJ_CONSAN_RESTORE_HOOK(amd, original, hsa_amd_profiling_set_profiler_enabled_fn, profiling);
    RJ_CONSAN_RESTORE_HOOK(amd, original, hsa_amd_queue_intercept_create_fn, intercept_create);
    RJ_CONSAN_RESTORE_HOOK(amd, original, hsa_amd_queue_intercept_register_fn, intercept_register);
#undef RJ_CONSAN_RESTORE_HOOK
  }
  if (retain)
    (void)runtime.release();
  else
    runtime.reset();
  installed_core = nullptr;
  installed_amd = nullptr;
}
