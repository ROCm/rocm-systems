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
#include "util/diagnostic.h"

#include <algorithm>
#include <atomic>
#include <bit>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fcntl.h>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <sys/random.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <unordered_map>

namespace {

using rocjitsu::FailureOr;
using namespace rocjitsu::consan;
using Blob = std::shared_ptr<std::vector<uint8_t>>;

// Preserve the previously admitted AMD table prefix. Shared headers may append
// APIs absent from the running ROCR, so do not require or copy their full size.
constexpr size_t kAmdTablePrefixBytes =
    offsetof(AmdExtTable, hsa_amd_external_semaphore_handle_close_fn) +
    sizeof(AmdExtTable::hsa_amd_external_semaphore_handle_close_fn);
constexpr size_t kNativeQueueTableBytes =
    offsetof(AmdExtTable, hsa_amd_queue_create_fn) + sizeof(AmdExtTable::hsa_amd_queue_create_fn);

// A ROCR-tool diagnostic must survive log-groups-OFF and initialization failures.
void diagnostic(std::string_view message) {
  constexpr std::string_view prefix = "[consan] ";
  for (const std::string_view part : {prefix, message, std::string_view("\n")}) {
    size_t offset = 0;
    while (offset < part.size()) {
      const ssize_t written = ::write(STDERR_FILENO, part.data() + offset, part.size() - offset);
      if (written < 0 && errno == EINTR) {
        continue;
      }
      if (written <= 0) {
        return;
      }
      offset += static_cast<size_t>(written);
    }
  }
}

struct Signal {
  bool dedicated = false;
  bool used = false;
  bool pending = false;
};

struct EntryBinding {
  EntryKey key = 0;
  EntryDisposition disposition = EntryDisposition::Native;
  EntryFacts binding;
  hsa_executable_symbol_t symbol = {};
  uint64_t kernel_object = 0;
  uint32_t kernarg_alignment = 0;
  std::shared_ptr<const Manifest> facts;
};

// Admitted code/binding across launches, retained through executable lifetime.
// Launch snapshots lease the immutable facts independently of this object.
struct Module {
  hsa_executable_t executable{};
  hsa_agent_t agent{};
  LoadedArtifact artifact;
  std::shared_ptr<const ImageFacts> facts;
  std::shared_ptr<const LoadedNativeImage> native_image;
  std::vector<EntryBinding> entries;
  bool closing = false;
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
  bool native = false;
  // Protected by Runtime::mutex_. Admission can observe GPU completion while
  // the observer still owns a wait on the application's original signal.
  bool native_completion_visible = false;
  bool application_native = false;
  uint64_t sequence = 0;
  uint64_t serial = 0;
  hsa_queue_t *queue = nullptr;
  std::shared_ptr<const Module> module;
  std::unique_ptr<DispatchLifecycle> lifecycle;
};

class Runtime;
struct QueueState {
  Runtime *owner = nullptr;
  hsa_queue_t *queue = nullptr;
  hsa_agent_t agent = {};
  uint32_t maximum_workgroup = 0;
  std::array<uint16_t, 3> maximum_dimensions = {};
  bool closing = false;
};

class Runtime {
public:
  Runtime(const CoreApiTable &core, const AmdExtTable &amd, const char *artifacts,
          const char *capture, const Digest &expected_profile, bool application_profile,
          std::vector<std::shared_ptr<const LoadedNativeImage>> native_images)
      : core_(core), amd_{}, artifact_path_(artifacts), capture_path_(capture),
        expected_profile_(expected_profile), application_profile_(application_profile),
        native_images_(std::move(native_images)) {
    std::memcpy(&amd_, &amd, std::min(sizeof(amd_), size_t{amd.version.minor_id}));
  }

  const CoreApiTable &core() const { return core_; }
  const AmdExtTable &amd() const { return amd_; }
  bool application_profile() const { return application_profile_; }

  // The lifecycle hooks serialize these reference updates and original HSA
  // calls outside mutex_: final shutdown synchronously reenters OnUnload.
  hsa_status_t initialize() {
    const hsa_status_t status = core_.hsa_init_fn();
    if (status == HSA_STATUS_SUCCESS)
      ++hsa_references_;
    return status;
  }

  bool final_hsa_reference() const { return hsa_references_ == 1; }
  void release_hsa_reference() { --hsa_references_; }

  void begin_shutdown() {
    std::lock_guard lock(mutex_);
    ++shutdown_calls_;
  }

  bool end_shutdown() {
    std::lock_guard lock(mutex_);
    --shutdown_calls_;
    return shutdown_calls_ == 0 && unload_seen_;
  }

  bool defer_unload() {
    std::lock_guard lock(mutex_);
    unload_seen_ = true;
    return shutdown_calls_ != 0;
  }

  hsa_status_t profiling(hsa_queue_t *queue, int enabled) {
    if (!application_profile_) {
      if (enabled != 0)
        diagnostic("ConSan MVP requires profiling disabled");
      return enabled == 0 ? HSA_STATUS_SUCCESS : HSA_STATUS_ERROR_INVALID_ARGUMENT;
    }
    {
      std::lock_guard lock(mutex_);
      if (closed_ ||
          !std::ranges::any_of(queues_, [queue](const std::unique_ptr<QueueState> &state) {
            return state->queue == queue && !state->closing;
          }))
        return HSA_STATUS_ERROR_INVALID_QUEUE;
    }
    return amd_.hsa_amd_profiling_set_profiler_enabled_fn(queue, enabled);
  }

  bool queue_cleanup_failed() {
    std::lock_guard lock(mutex_);
    return queue_cleanup_failed_;
  }

  hsa_status_t reader_memory(const void *bytes, size_t size, hsa_code_object_reader_t *reader) {
    if (bytes == nullptr || reader == nullptr || size == 0 || size > ContractLimits{}.image_bytes) {
      return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    }
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
        uint64_t(info.st_size) > ContractLimits{}.image_bytes) {
      return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    }
    std::vector<uint8_t> bytes(static_cast<size_t>(info.st_size));
    size_t offset = 0;
    while (offset < bytes.size()) {
      const ssize_t count = ::pread(file, bytes.data() + offset, bytes.size() - offset, offset);
      if (count < 0 && errno == EINTR) {
        continue;
      }
      if (count <= 0) {
        return HSA_STATUS_ERROR_INVALID_ARGUMENT;
      }
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
      const std::unordered_map<uint64_t, Blob>::iterator found = readers_.find(reader.handle);
      if (closed_ || status_ != RuntimeStatus::Success || (!application_profile_ && module_ever_) ||
          loading_ || find_module_locked(executable) != nullptr || native_modules_.size() >= 16 ||
          found == readers_.end()) {
        return reject_locked("load requires a tracked original reader and one admitted module");
      }
      original = found->second;
      loading_ = true;
    }
    auto module = std::make_shared<Module>();
    util::StringDiagnostic artifact_diagnostic;
    const std::string target = agent_target(agent);
    const Digest original_digest = digest(*original);
    for (const std::shared_ptr<const LoadedNativeImage> &native : native_images_) {
      if (native->inventory.identity.sha256 == original_digest && native->original == *original) {
        module->native_image = native;
        break;
      }
    }
    if (module->native_image) {
      const TargetProfile *profile = target_profile(target);
      if (profile == nullptr || profile->id != module->native_image->inventory.target) {
        std::lock_guard lock(mutex_);
        loading_ = false;
        return reject_locked("known native image does not match the actual GPU agent");
      }
    } else {
      {
        std::lock_guard lock(mutex_);
        if (module_ever_ || artifact_path_.empty()) {
          loading_ = false;
          return reject_locked("image is not an allowed native image or the single selected image");
        }
      }
      FailureOr<LoadedArtifact> artifact = load_artifact(
          artifact_path_, *original, target, expected_profile_, artifact_diagnostic.emitter());
      if (artifact.failed()) {
        std::lock_guard lock(mutex_);
        loading_ = false;
        return reject_locked(artifact_diagnostic.message());
      }
      module->artifact = std::move(artifact.value());
      module->facts = std::make_shared<const ImageFacts>(std::move(module->artifact.facts));
    }
    module->executable = executable;
    module->agent = agent;
    hsa_code_object_reader_t prepared{};
    const std::vector<uint8_t> &image =
        module->native_image ? module->native_image->original : module->artifact.prepared;
    hsa_status_t status =
        core_.hsa_code_object_reader_create_from_memory_fn(image.data(), image.size(), &prepared);
    if (status == HSA_STATUS_SUCCESS) {
      status = core_.hsa_executable_load_agent_code_object_fn(executable, agent, prepared, options,
                                                              loaded);
      const hsa_status_t destroyed = core_.hsa_code_object_reader_destroy_fn(prepared);
      if (status == HSA_STATUS_SUCCESS && destroyed != HSA_STATUS_SUCCESS) {
        status = destroyed;
      }
    }
    std::lock_guard lock(mutex_);
    loading_ = false;
    // Keep prepared storage through executable destruction even when loader
    // failure has partially installed state in the application's executable.
    if (module->native_image)
      native_modules_.push_back(std::move(module));
    else {
      module_ = std::move(module);
      module_ever_ = true;
    }
    if (status != HSA_STATUS_SUCCESS) {
      fail_locked(RuntimeStatus::InvalidFacts, "HSA refused the matched prepared image");
    }
    return status;
  }

  hsa_status_t freeze(hsa_executable_t executable, const char *options) {
    std::shared_ptr<Module> module;
    {
      std::lock_guard lock(mutex_);
      module = find_module_locked(executable);
      if (!module || closed_ || status_ != RuntimeStatus::Success) {
        return reject_locked("freeze has no active admitted ConSan module");
      }
    }
    // HSA may retain or partially freeze the prepared storage even on failure.
    // The local lease pins it while external HSA work runs without our mutex.
    hsa_status_t status = core_.hsa_executable_freeze_fn(executable, options);
    if (status != HSA_STATUS_SUCCESS) {
      std::lock_guard lock(mutex_);
      module->entries.clear();
      fail_locked(RuntimeStatus::InvalidFacts, "HSA refused to freeze the matched prepared image");
      return status;
    }
    // Validate resources without inventing geometry, then publish all entry
    // bindings together before any dispatch can use them.
    std::vector<EntryBinding> bindings;
    std::set<uint64_t> objects;
    std::set<uint64_t> symbols;
    bool valid = true;
    const std::vector<ImageEntryFacts> &inventory =
        module->facts ? module->facts->entries : module->native_image->inventory.entries;
    for (const ImageEntryFacts &entry : inventory) {
      EntryBinding binding = {.key = entry.key,
                              .disposition = entry.disposition,
                              .binding = entry.binding,
                              .symbol = {},
                              .kernel_object = 0,
                              .kernarg_alignment = 0,
                              .facts = {}};
      const std::string symbol = entry.binding.name + ".kd";
      status = core_.hsa_executable_get_symbol_by_name_fn(executable, symbol.c_str(),
                                                          &module->agent, &binding.symbol);
      uint32_t kernarg = 0, group = 0, private_bytes = 0;
      if (status == HSA_STATUS_SUCCESS) {
        status = core_.hsa_executable_symbol_get_info_fn(
            binding.symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &binding.kernel_object);
      }
      if (status == HSA_STATUS_SUCCESS) {
        status = core_.hsa_executable_symbol_get_info_fn(
            binding.symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE, &kernarg);
      }
      if (status == HSA_STATUS_SUCCESS) {
        status = core_.hsa_executable_symbol_get_info_fn(
            binding.symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_ALIGNMENT,
            &binding.kernarg_alignment);
      }
      if (status == HSA_STATUS_SUCCESS) {
        status = core_.hsa_executable_symbol_get_info_fn(
            binding.symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE, &group);
      }
      if (status == HSA_STATUS_SUCCESS) {
        status = core_.hsa_executable_symbol_get_info_fn(
            binding.symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE, &private_bytes);
      }
      uint32_t expected_kernarg = entry.binding.original_kernarg_bytes;
      if (entry.selected) {
        binding.facts = std::shared_ptr<const Manifest>(module->facts, &*entry.selected);
        const std::optional<rocjitsu::KernargExtensionLayout> wrapper =
            rocjitsu::make_kernarg_extension_layout(entry.binding.original_kernarg_bytes,
                                                    kContextPointerPayloads);
        if (!wrapper) {
          valid = false;
          break;
        }
        expected_kernarg = wrapper->wrapper_size;
      }
      if (status != HSA_STATUS_SUCCESS || binding.symbol.handle == 0 ||
          binding.kernel_object == 0 || !objects.insert(binding.kernel_object).second ||
          !symbols.insert(binding.symbol.handle).second || kernarg != expected_kernarg ||
          !std::has_single_bit(binding.kernarg_alignment) || binding.kernarg_alignment < 8 ||
          group != entry.binding.prepared_group_bytes ||
          private_bytes != entry.binding.prepared_private_bytes) {
        valid = false;
        break;
      }
      binding.kernarg_alignment =
          std::max(binding.kernarg_alignment, entry.binding.kernarg_alignment);
      bindings.push_back(std::move(binding));
    }
    std::lock_guard lock(mutex_);
    // Queries call external HSA functions, so lifetime admission is rechecked.
    if (closed_ || status_ != RuntimeStatus::Success || find_module_locked(executable) != module)
      return reject_locked("freeze admission ended during HSA symbol queries");
    if (!valid || bindings.size() != inventory.size())
      return reject_locked("loaded HSA symbols/resources do not match complete image facts");
    const auto collides = [&](const std::shared_ptr<Module> &other) {
      return other != module && std::ranges::any_of(other->entries, [&](const EntryBinding &entry) {
               return objects.contains(entry.kernel_object) ||
                      symbols.contains(entry.symbol.handle);
             });
    };
    if ((module_ && collides(module_)) || std::ranges::any_of(native_modules_, collides))
      return reject_locked("loaded entry identity collides with another admitted module");
    if (module->facts && !started_ && !start_locked(*module->facts)) {
      return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
    }
    module->entries = std::move(bindings);
    return HSA_STATUS_SUCCESS;
  }

  hsa_status_t symbol_info(hsa_executable_symbol_t symbol, hsa_executable_symbol_info_t attribute,
                           void *value) {
    {
      std::lock_guard lock(mutex_);
      if (module_ && value != nullptr) {
        for (const EntryBinding &entry : module_->entries) {
          if (symbol.handle != entry.symbol.handle) {
            continue;
          }
          // ROCR's v5 API alignment is independent of the metadata note.
          // Forward that query instead of projecting the note's alignment.
          const uint32_t *original = nullptr;
          if (attribute == HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE) {
            original = &entry.binding.original_kernarg_bytes;
          } else if (attribute == HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE) {
            original = &entry.binding.original_group_bytes;
          } else if (attribute == HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE) {
            original = &entry.binding.original_private_bytes;
          }
          if (original != nullptr) {
            std::memcpy(value, original, sizeof(uint32_t));
            return HSA_STATUS_SUCCESS;
          }
          break;
        }
      }
    }
    return core_.hsa_executable_symbol_get_info_fn(symbol, attribute, value);
  }

  hsa_status_t executable_destroy(hsa_executable_t executable) {
    {
      std::unique_lock lock(mutex_);
      if (application_profile_) {
        const std::shared_ptr<Module> module = find_module_locked(executable);
        if (module) {
          module->closing = true;
          if (!condition_.wait_for(lock, std::chrono::seconds(30), [&] {
                return !std::ranges::any_of(pending_, [&](const std::shared_ptr<Launch> &launch) {
                  return launch->module == module;
                });
              }))
            return HSA_STATUS_ERROR_INVALID_EXECUTABLE;
        }
      }
      if (module_ && module_->executable.handle == executable.handle && active_gpu_) {
        return HSA_STATUS_ERROR_INVALID_EXECUTABLE;
      }
    }
    const hsa_status_t status = core_.hsa_executable_destroy_fn(executable);
    if (status == HSA_STATUS_SUCCESS) {
      std::lock_guard lock(mutex_);
      if (module_ && module_->executable.handle == executable.handle) {
        module_.reset();
      }
      std::erase_if(native_modules_, [executable](const std::shared_ptr<Module> &module) {
        return module->executable.handle == executable.handle;
      });
    }
    return status;
  }

  hsa_status_t queue_create(hsa_agent_t agent, uint32_t size, hsa_queue_type32_t type,
                            void (*callback)(hsa_status_t, hsa_queue_t *, void *), void *data,
                            uint32_t private_bytes, uint32_t group, hsa_queue_t **queue) {
    {
      std::lock_guard lock(mutex_);
      if (closed_ || status_ != RuntimeStatus::Success ||
          (!application_profile_ &&
           (!started_ || queue_ever_ || !module_ || agent.handle != module_->agent.handle))) {
        return reject_locked("MVP requires one queue created after the prepared module is frozen");
      }
      queue_ever_ = true;
      ++queue_creations_;
    }
    std::unique_ptr<QueueState> context =
        std::make_unique<QueueState>(QueueState{.owner = this, .agent = agent});
    if (application_profile_ &&
        (core_.hsa_agent_get_info_fn(agent, HSA_AGENT_INFO_WORKGROUP_MAX_SIZE,
                                     &context->maximum_workgroup) != HSA_STATUS_SUCCESS ||
         core_.hsa_agent_get_info_fn(agent, HSA_AGENT_INFO_WORKGROUP_MAX_DIM,
                                     context->maximum_dimensions.data()) != HSA_STATUS_SUCCESS)) {
      std::lock_guard lock(mutex_);
      --queue_creations_;
      condition_.notify_all();
      return reject_locked("cannot query native queue hardware workgroup limits");
    }
    hsa_status_t status = amd_.hsa_amd_queue_intercept_create_fn(agent, size, type, callback, data,
                                                                 private_bytes, group, queue);
    if (status != HSA_STATUS_SUCCESS) {
      std::lock_guard lock(mutex_);
      --queue_creations_;
      condition_.notify_all();
      return status;
    }
    context->queue = *queue;
    status = amd_.hsa_amd_queue_intercept_register_fn(
        *queue,
        [](const void *packets, uint64_t count, uint64_t, void *context,
           hsa_amd_queue_intercept_packet_writer writer) noexcept {
          QueueState *binding = static_cast<QueueState *>(context);
          binding->owner->intercept(*binding, packets, count, writer);
        },
        context.get());
    if (status == HSA_STATUS_SUCCESS && !application_profile_)
      status = amd_.hsa_amd_profiling_set_profiler_enabled_fn(*queue, 0);
    if (status == HSA_STATUS_SUCCESS) {
      std::lock_guard lock(mutex_);
      if (!closed_) {
        --queue_creations_;
        queues_.push_back(std::move(context));
        condition_.notify_all();
        return HSA_STATUS_SUCCESS;
      }
      status = HSA_STATUS_ERROR_INVALID_QUEUE_CREATION;
    }
    if (status != HSA_STATUS_SUCCESS) {
      const hsa_status_t destroyed = core_.hsa_queue_destroy_fn(*queue);
      std::lock_guard lock(mutex_);
      --queue_creations_;
      if (destroyed != HSA_STATUS_SUCCESS) {
        queue_cleanup_failed_ = true;
        queues_.push_back(std::move(context));
        fail_locked(RuntimeStatus::InvalidTransition,
                    "failed to destroy a partially registered queue");
      }
      *queue = nullptr;
      condition_.notify_all();
      return status;
    }
    return status;
  }

  hsa_status_t native_queue_create(hsa_agent_t agent, hsa_amd_queue_create_desc_t *descs,
                                   uint32_t count) {
    {
      std::lock_guard lock(mutex_);
      if (closed_)
        return HSA_STATUS_ERROR_INVALID_QUEUE_CREATION;
      ++native_queue_calls_;
    }
    const auto release = [this](Runtime *) {
      std::lock_guard lock(mutex_);
      --native_queue_calls_;
      condition_.notify_all();
    };
    std::unique_ptr<Runtime, decltype(release)> reservation(this, release);
    const hsa_status_t original = amd_.hsa_amd_queue_create_fn(agent, descs, count);
    if (descs == nullptr)
      return original;
    hsa_status_t first_error = HSA_STATUS_SUCCESS;
    for (uint32_t index = 0; index < count; ++index) {
      hsa_amd_queue_create_desc_t &desc = descs[index];
      hsa_queue_t *plain = desc.queue;
      if (plain == nullptr)
        continue;
      hsa_status_t status = HSA_STATUS_ERROR_INVALID_QUEUE_CREATION;
      hsa_queue_t *replacement = nullptr;
      if (desc.engine_type == HSA_AMD_QUEUE_ENGINE_COMPUTE && desc.flags == 0) {
        const hsa_amd_compute_queue_params_t &compute = desc.engine.compute;
        status = queue_create(agent, desc.queue_size_bytes / sizeof(hsa_kernel_dispatch_packet_t),
                              compute.type, desc.callback, desc.callback_data,
                              compute.private_segment_size, UINT32_MAX, &replacement);
        if (status == HSA_STATUS_SUCCESS && desc.priority != HSA_AMD_QUEUE_PRIORITY_NORMAL)
          status = amd_.hsa_amd_queue_set_priority_fn != nullptr
                       ? amd_.hsa_amd_queue_set_priority_fn(replacement, desc.priority)
                       : HSA_STATUS_ERROR_INVALID_QUEUE_CREATION;
        if (status == HSA_STATUS_SUCCESS && compute.cu_mask_count != 0)
          status = amd_.hsa_amd_queue_cu_set_mask_fn != nullptr
                       ? amd_.hsa_amd_queue_cu_set_mask_fn(replacement, compute.cu_mask_count,
                                                           compute.cu_mask)
                       : HSA_STATUS_ERROR_INVALID_QUEUE_CREATION;
      }
      const hsa_status_t destroyed = core_.hsa_queue_destroy_fn(plain);
      if (destroyed != HSA_STATUS_SUCCESS) {
        std::lock_guard lock(mutex_);
        queue_cleanup_failed_ = true;
        fail_locked(RuntimeStatus::InvalidTransition, "failed to retire an original native queue");
        if (status == HSA_STATUS_SUCCESS)
          status = destroyed;
      }
      if (status == HSA_STATUS_SUCCESS) {
        desc.queue = replacement;
      } else {
        if (replacement != nullptr)
          (void)queue_destroy(replacement);
        desc.queue = nullptr;
        if (first_error == HSA_STATUS_SUCCESS)
          first_error = status;
      }
    }
    return original != HSA_STATUS_SUCCESS ? original : first_error;
  }

  hsa_status_t queue_destroy(hsa_queue_t *queue) {
    {
      std::unique_lock lock(mutex_);
      if (application_profile_) {
        for (const std::unique_ptr<QueueState> &state : queues_)
          if (state->queue == queue)
            state->closing = true;
        if (!condition_.wait_for(lock, std::chrono::seconds(30), [&] {
              return !std::ranges::any_of(pending_, [queue](const std::shared_ptr<Launch> &launch) {
                return launch->queue == queue;
              });
            }))
          return HSA_STATUS_ERROR_INVALID_QUEUE;
      }
      if (active_gpu_ &&
          std::ranges::any_of(queues_, [queue](const std::unique_ptr<QueueState> &binding) {
            return binding->queue == queue;
          })) {
        return HSA_STATUS_ERROR_INVALID_QUEUE;
      }
    }
    const hsa_status_t status = core_.hsa_queue_destroy_fn(queue);
    if (status == HSA_STATUS_SUCCESS) {
      std::lock_guard lock(mutex_);
      std::erase_if(queues_, [queue](const std::unique_ptr<QueueState> &binding) {
        return binding->queue == queue;
      });
    }
    return status;
  }

  hsa_status_t signal_create(hsa_signal_value_t value, uint32_t consumers,
                             const hsa_agent_t *agents, hsa_signal_t *signal) {
    return record_signal(core_.hsa_signal_create_fn(value, consumers, agents, signal), value,
                         signal);
  }

  hsa_status_t amd_signal_create(hsa_signal_value_t value, uint32_t consumers,
                                 const hsa_agent_t *agents, uint64_t attributes,
                                 hsa_signal_t *signal) {
    return record_signal(
        amd_.hsa_amd_signal_create_fn(value, consumers, agents, attributes, signal), value, signal);
  }

  hsa_status_t signal_destroy(hsa_signal_t signal) {
    {
      std::lock_guard lock(mutex_);
      const std::unordered_map<uint64_t, Signal>::iterator found = signals_.find(signal.handle);
      if (found != signals_.end() && found->second.pending) {
        return HSA_STATUS_ERROR_INVALID_SIGNAL;
      }
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
    const uint64_t target_sequence = next_sequence_;
    if (!condition_.wait_for(lock, std::chrono::seconds(30), [&] {
          return processed_ >= target &&
                 !std::ranges::any_of(pending_,
                                      [target_sequence](const std::shared_ptr<Launch> &launch) {
                                        return launch->sequence <= target_sequence;
                                      });
        })) {
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
      std::unique_lock lock(mutex_);
      if (finished_) {
        return static_cast<int>(status_);
      }
      closed_ = true;
      condition_.wait(lock, [&] {
        return queue_creations_ == 0 && native_queue_calls_ == 0 && callbacks_ == 0;
      });
      if (!started_ && !workers_started_) {
        if (output_ >= 0) {
          const int closed = ::close(output_);
          output_ = -1;
          if (closed != 0) {
            fail_locked(RuntimeStatus::CloseFailed, "failed partial capture close");
          }
        }
        finished_ = true;
        if (status_ == RuntimeStatus::Success) {
          fail_locked(RuntimeStatus::InvalidFacts, "finish has no admitted artifact/capture");
        }
        return static_cast<int>(status_);
      }
      if (!started_ && !artifact_path_.empty())
        fail_locked(RuntimeStatus::InvalidFacts, "selected artifact never started its capture");
    }
    (void)drain();
    {
      std::lock_guard lock(mutex_);
      if (processed_ != statistics_.submitted ||
          statistics_.native_completed != statistics_.native_submitted || active_gpu_) {
        return static_cast<int>(status_);
      }
      stop_ = true;
      condition_.notify_all();
    }
    if (observer_.joinable()) {
      observer_.join();
    }
    if (collector_.joinable()) {
      collector_.join();
    } else if (started_) {
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
      if (active_gpu_ || processed_ != statistics_.submitted ||
          statistics_.native_completed != statistics_.native_submitted) {
        return false;
      }
      status_ = RuntimeStatus::InvalidTransition;
      closed_ = stop_ = true;
      condition_.notify_all();
    }
    if (observer_.joinable()) {
      observer_.join();
    }
    if (collector_.joinable()) {
      collector_.join();
    } else if (output_ >= 0) {
      const int closed = ::close(output_);
      output_ = -1;
      if (closed != 0) {
        diagnostic("failed abandoned partial capture close");
      }
    }
    return true;
  }

private:
  std::shared_ptr<Module> find_module_locked(hsa_executable_t executable) {
    if (module_ && module_->executable.handle == executable.handle)
      return module_;
    for (const std::shared_ptr<Module> &module : native_modules_)
      if (module->executable.handle == executable.handle)
        return module;
    return nullptr;
  }

  hsa_status_t record_signal(hsa_status_t status, hsa_signal_value_t value,
                             const hsa_signal_t *signal) {
    if (status == HSA_STATUS_SUCCESS) {
      std::lock_guard lock(mutex_);
      signals_[signal->handle] = {.dedicated = value == 1};
    }
    return status;
  }

  std::string agent_target(hsa_agent_t agent) {
    hsa_device_type_t type{};
    hsa_isa_t isa{};
    uint32_t length = 0;
    if (core_.hsa_agent_get_info_fn(agent, HSA_AGENT_INFO_DEVICE, &type) != HSA_STATUS_SUCCESS ||
        type != HSA_DEVICE_TYPE_GPU ||
        core_.hsa_agent_get_info_fn(agent, HSA_AGENT_INFO_ISA, &isa) != HSA_STATUS_SUCCESS ||
        core_.hsa_isa_get_info_alt_fn(isa, HSA_ISA_INFO_NAME_LENGTH, &length) !=
            HSA_STATUS_SUCCESS ||
        length == 0 || length > 256) {
      return {};
    }
    std::vector<char> name(length);
    if (core_.hsa_isa_get_info_alt_fn(isa, HSA_ISA_INFO_NAME, name.data()) != HSA_STATUS_SUCCESS ||
        name.back() != '\0') {
      return {};
    }
    constexpr std::string_view prefix = "amdgcn-amd-amdhsa--";
    std::string_view isa_name(name.data());
    if (!isa_name.starts_with(prefix)) {
      return {};
    }
    isa_name.remove_prefix(prefix.size());
    isa_name = isa_name.substr(0, isa_name.find(':'));
    const TargetProfile *target = target_profile(isa_name);
    return target == nullptr ? std::string{} : std::string(target->name);
  }

  void fail_locked(RuntimeStatus status, std::string_view message) {
    if (status_ == RuntimeStatus::Success) {
      status_ = status;
    }
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
              type != HSA_DEVICE_TYPE_CPU) {
            return HSA_STATUS_SUCCESS;
          }
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
    if (amd_.hsa_amd_memory_pool_allocate_fn(pool_, bytes, 0, memory) != HSA_STATUS_SUCCESS) {
      return false;
    }
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
        info.agentBaseAddress != info.hostBaseAddress) {
      return false;
    }
    // The application owns its original kernargs before publication. CPU-mapped
    // coarse HSA allocations are readable here too; only the new wrapper/report
    // transport must be fine-grained. Unknown, locked, unmapped and aliased
    // pointers never acquire host-read permission through this path.
    const uint32_t allowed_grain =
        HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED |
        (application_profile_ ? HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED : 0);
    if ((info.global_flags & allowed_grain) == 0) {
      return false;
    }
    const uintptr_t base = reinterpret_cast<uintptr_t>(info.hostBaseAddress);
    const uintptr_t address = reinterpret_cast<uintptr_t>(pointer);
    return address >= base && address - base <= info.sizeInBytes &&
           size <= info.sizeInBytes - (address - base);
  }

  bool start_locked(const ImageFacts &facts) {
    if (!find_pool_locked()) {
      fail_locked(RuntimeStatus::InvalidFacts, "CPU fine-grained kernarg pool unavailable");
      return false;
    }
    CaptureHeader header;
    header.image = facts;
    header.execution_conditions = capture_execution_conditions(facts);
    const FailureOr<std::vector<uint8_t>> encoded_facts = encode_image_facts(facts);
    if (encoded_facts.failed()) {
      fail_locked(RuntimeStatus::InvalidFacts, "cannot encode capture facts");
      return false;
    }
    planned_capture_bytes_ = 64 + encoded_facts.value().size() + 64;
    size_t offset = 0;
    while (offset < header.run_identity.size()) {
      const ssize_t count =
          ::getrandom(header.run_identity.data() + offset, header.run_identity.size() - offset, 0);
      if (count < 0 && errno == EINTR) {
        continue;
      }
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
              if (::link(partial_path_.c_str(), capture_path_.c_str()) != 0) {
                return false;
              }
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
    start_observer_locked();
    collector_ = std::thread([this] { collector_loop(); });
    return true;
  }

  void start_observer_locked() {
    if (!workers_started_) {
      workers_started_ = true;
      observer_ = std::thread([this] { observer_loop(); });
    }
  }

  void fail_packet_completion_locked(const hsa_kernel_dispatch_packet_t &packet) {
    hsa_signal_t completion{};
    switch (packet.header & 0xff) {
    case HSA_PACKET_TYPE_KERNEL_DISPATCH:
      completion = packet.completion_signal;
      break;
    case HSA_PACKET_TYPE_BARRIER_AND: {
      hsa_barrier_and_packet_t barrier{};
      std::memcpy(&barrier, &packet, sizeof(barrier));
      completion = barrier.completion_signal;
      break;
    }
    case HSA_PACKET_TYPE_BARRIER_OR: {
      hsa_barrier_or_packet_t barrier{};
      std::memcpy(&barrier, &packet, sizeof(barrier));
      completion = barrier.completion_signal;
      break;
    }
    default:
      return;
    }
    const std::unordered_map<uint64_t, Signal>::iterator found = signals_.find(completion.handle);
    if (found != signals_.end() && !found->second.pending) {
      core_.hsa_signal_store_screlease_fn(completion, -1);
    }
  }

  void reject_packet_locked(const hsa_kernel_dispatch_packet_t &packet, std::string_view message) {
    fail_locked(RuntimeStatus::InvalidFacts, message);
    fail_packet_completion_locked(packet);
  }

  void release_unsubmitted_locked(Launch &launch) {
    if (launch.proxy.handle) {
      (void)core_.hsa_signal_destroy_fn(launch.proxy);
    }
    if (launch.wrapper) {
      (void)amd_.hsa_amd_memory_pool_free_fn(launch.wrapper);
    }
    if (launch.context) {
      (void)amd_.hsa_amd_memory_pool_free_fn(launch.context);
    }
    if (launch.slot) {
      slots_[launch.slot - 1].leased = false;
    }
  }

  bool prepare_selected_locked(const std::shared_ptr<Launch> &launch, const EntryBinding *binding,
                               hsa_kernel_dispatch_packet_t &packet,
                               const LaunchDimensions &dimensions) {
    if (launch->module != module_) {
      reject_packet_locked(packet, "selected dispatch does not own the prepared image");
      return false;
    }
    const Manifest &facts = *binding->facts;
    const FailureOr<LaunchLayout> layout =
        plan_launch(facts, dimensions, packet.group_segment_size, packet.private_segment_size);
    if (layout.failed() ||
        (reinterpret_cast<uintptr_t>(packet.kernarg_address) % binding->kernarg_alignment) != 0 ||
        !host_arguments_locked(packet.kernarg_address, facts.entry.original_kernarg_bytes)) {
      reject_packet_locked(
          packet, "unsupported geometry/segments or inaccessible original kernarg allocation");
      return false;
    }
    const uint64_t capture_bytes =
        48 + 144 + uint64_t{facts.entry.original_kernarg_bytes} + layout.value().report_bytes;
    if (capture_bytes > CaptureLimits{}.file_bytes - planned_capture_bytes_) {
      reject_packet_locked(packet, "capture budget exhausted before submission");
      return false;
    }
    // Prepare sufficient idle report capacity and per-launch wrapper/context
    // plus proxy signal. Rollback owns all unsubmitted acquisitions.
    for (size_t index = 0; index < slots_.size(); ++index) {
      if (!slots_[index].leased) {
        launch->slot = static_cast<uint32_t>(index + 1);
        break;
      }
    }
    if (launch->slot == 0 &&
        slots_.size() < (application_profile_ ? CaptureLimits{}.executions : 16)) {
      ReportSlot slot;
      slot.capacity = layout.value().report_bytes;
      if (allocate_locked(slot.capacity, &slot.memory)) {
        const auto free = [this](void *memory) { (void)amd_.hsa_amd_memory_pool_free_fn(memory); };
        std::unique_ptr<void, decltype(free)> uninstalled(slot.memory, free);
        slots_.push_back(slot);
        (void)uninstalled.release();
        launch->slot = static_cast<uint32_t>(slots_.size());
      }
    }
    if (launch->slot == 0) {
      reject_packet_locked(packet, "report allocation/backlog admission failed before submission");
      return false;
    }
    ReportSlot &slot = slots_[launch->slot - 1];
    const bool reused = slot.generation != 0 && slot.capacity >= layout.value().report_bytes;
    if (slot.capacity < layout.value().report_bytes) {
      void *replacement = nullptr;
      if (!allocate_locked(layout.value().report_bytes, &replacement)) {
        reject_packet_locked(packet, "report growth allocation failed before submission");
        return false;
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
        reinterpret_cast<uintptr_t>(launch->wrapper) % binding->kernarg_alignment != 0 ||
        !allocate_locked(kContextBytes, &launch->context) ||
        core_.hsa_signal_create_fn(1, 0, nullptr, &launch->proxy) != HSA_STATUS_SUCCESS) {
      reject_packet_locked(packet, "wrapper/context/proxy allocation failed before submission");
      return false;
    }
    const uint64_t serial = statistics_.submitted + 1;
    launch->serial = serial;
    const FailureOr<TransportImage> transport = initialize_transport(
        facts, dimensions, layout.value().group_bytes, layout.value().private_bytes, serial,
        reinterpret_cast<uint64_t>(slot.memory));
    if (transport.failed()) {
      reject_packet_locked(packet, "canonical transport initialization failed before submission");
      return false;
    }
    std::memcpy(slot.memory, transport.value().report.data(), transport.value().report.size());
    std::memcpy(launch->context, transport.value().context.data(), kContextBytes);
    std::span<uint8_t> wrapper =
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
      return false;
    }
    // Bind execution identity and original arguments to an empty report
    // envelope. Lifecycle admission does not itself submit the HSA packet.
    CapturedExecution envelope;
    envelope.serial = serial;
    envelope.entry_key = binding->key;
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
    launch->lifecycle = std::make_unique<DispatchLifecycle>(binding->facts, std::move(envelope));
    if (launch->lifecycle->submitted() != RuntimeStatus::Success) {
      reject_packet_locked(packet, "dispatch ownership initialization failed before submission");
      return false;
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
    ++statistics_.submitted;
    planned_capture_bytes_ += capture_bytes;
    if (reused)
      ++statistics_.reused;
    condition_.notify_all();
    return true;
  }

  void intercept_application(QueueState &queue, const void *packets, uint64_t count,
                             hsa_amd_queue_intercept_packet_writer writer) {
    {
      std::lock_guard lock(mutex_);
      ++callbacks_;
    }
    const auto leave = [this](Runtime *) {
      std::lock_guard lock(mutex_);
      --callbacks_;
      condition_.notify_all();
    };
    std::unique_ptr<Runtime, decltype(leave)> callback(this, leave);
    if (count > 1) {
      // ROCR can coalesce adjacent packets into one callback. Keep their
      // publication order and individual ownership; the enclosing callback
      // remains live through the whole batch for finish's publication fence.
      const uint8_t *bytes = static_cast<const uint8_t *>(packets);
      for (uint64_t index = 0; index < count; ++index)
        intercept_application(queue, bytes + index * sizeof(hsa_kernel_dispatch_packet_t), 1,
                              writer);
      return;
    }
    hsa_kernel_dispatch_packet_t packet = {};
    std::memcpy(&packet, packets, sizeof(packet));
    const uint16_t type = packet.header & 0xff;
    if (type == HSA_PACKET_TYPE_BARRIER_AND || type == HSA_PACKET_TYPE_BARRIER_OR) {
      {
        std::lock_guard lock(mutex_);
        if (closed_ || queue.closing || status_ != RuntimeStatus::Success) {
          fail_packet_completion_locked(packet);
          return;
        }
      }
      writer(packets, 1);
      return;
    }
    hsa_amd_ext_kernel_dispatch_packet_t extended = {};
    bool is_extended = false;
    bool valid = type == HSA_PACKET_TYPE_KERNEL_DISPATCH;
    if (type == HSA_PACKET_TYPE_VENDOR_SPECIFIC) {
      std::memcpy(&extended, packets, sizeof(extended));
      is_extended = true;
      valid =
          extended.amd_format == HSA_AMD_PACKET_TYPE_EXT_KERNEL_DISPATCH && extended.reserved0 == 0;
      packet.setup = extended.setup;
      const std::array<uint32_t, 3> clusters = {extended.cluster_count_x, extended.cluster_count_y,
                                                extended.cluster_count_z};
      const std::array<uint32_t, 3> sizes = {extended.cluster_size_x, extended.cluster_size_y,
                                             extended.cluster_size_z};
      const std::array<uint32_t, 3> workgroup = {packet.workgroup_size_x, packet.workgroup_size_y,
                                                 packet.workgroup_size_z};
      std::array<uint32_t, 3> grid = {};
      for (size_t dimension = 0; dimension < 3; ++dimension) {
        const uint64_t extent =
            uint64_t{clusters[dimension]} * sizes[dimension] * workgroup[dimension];
        if (clusters[dimension] == 0 || sizes[dimension] == 0 || extent > UINT32_MAX)
          valid = false;
        grid[dimension] = static_cast<uint32_t>(extent);
      }
      packet.grid_size_x = grid[0];
      packet.grid_size_y = grid[1];
      packet.grid_size_z = grid[2];
    }
    auto launch = std::make_shared<Launch>();
    {
      std::lock_guard lock(mutex_);
      std::shared_ptr<Module> owner;
      const EntryBinding *binding = nullptr;
      const auto find = [&](const std::shared_ptr<Module> &module) {
        if (!module || module->closing || module->agent.handle != queue.agent.handle)
          return;
        for (const EntryBinding &entry : module->entries)
          if (entry.kernel_object == packet.kernel_object) {
            owner = module;
            binding = &entry;
          }
      };
      find(module_);
      for (const std::shared_ptr<Module> &module : native_modules_)
        find(module);
      const std::array<uint32_t, 3> workgroup = {packet.workgroup_size_x, packet.workgroup_size_y,
                                                 packet.workgroup_size_z};
      const std::array<uint32_t, 3> grid = {packet.grid_size_x, packet.grid_size_y,
                                            packet.grid_size_z};
      const uint64_t threads = uint64_t{workgroup[0]} * workgroup[1] * workgroup[2];
      valid = valid && packet.setup >= 1 && packet.setup <= 3 && threads != 0 &&
              threads <= queue.maximum_workgroup;
      for (size_t dimension = 0; dimension < 3; ++dimension)
        valid = valid && workgroup[dimension] != 0 && grid[dimension] != 0 &&
                workgroup[dimension] <= queue.maximum_dimensions[dimension] &&
                (dimension < packet.setup || (workgroup[dimension] == 1 && grid[dimension] == 1));
      if (!valid || closed_ || queue.closing || status_ != RuntimeStatus::Success ||
          statistics_.submitted + statistics_.native_submitted >= CaptureLimits{}.executions ||
          binding == nullptr || packet.completion_signal.handle != 0) {
        fail_locked(RuntimeStatus::InvalidFacts, "application native packet violates binding, "
                                                 "format, geometry or null-completion profile");
        return;
      }
      const EntryFacts &entry = binding->binding;
      // Native ROCR dispatches can exceed the metadata flat hint (the runtime
      // helper corpus does so). Preserve that behavior within actual hardware
      // limits; required shape remains an ABI constraint, not a detector limit.
      if (packet.private_segment_size != entry.original_private_bytes ||
          packet.group_segment_size < entry.original_group_bytes ||
          packet.group_segment_size > 65536 ||
          (entry.required_workgroup_size[0] != 0 && workgroup != entry.required_workgroup_size) ||
          (entry.original_kernarg_bytes != 0 && packet.kernarg_address == nullptr) ||
          reinterpret_cast<uintptr_t>(packet.kernarg_address) % binding->kernarg_alignment != 0) {
        fail_locked(RuntimeStatus::InvalidFacts, "application native packet violates original ABI");
        return;
      }
      // Trace rows currently use TTMP9 as the global workgroup X identity.
      // Format 3 supplies a cluster coordinate there for nonunit clusters.
      if (binding->disposition == EntryDisposition::Selected && is_extended &&
          (extended.cluster_size_x != 1 || extended.cluster_size_y != 1 ||
           extended.cluster_size_z != 1)) {
        reject_packet_locked(packet, "application execution tracing requires unit cluster sizes");
        return;
      }
      launch->module = std::move(owner);
      launch->queue = queue.queue;
      launch->sequence = ++next_sequence_;
      if (binding->disposition == EntryDisposition::Selected) {
        const auto release = [this](Launch *unsubmitted) {
          release_unsubmitted_locked(*unsubmitted);
        };
        std::unique_ptr<Launch, decltype(release)> rollback(launch.get(), release);
        if (!binding->facts || binding->facts->scope != ScopeKind::ExecutionTrace) {
          reject_packet_locked(packet,
                               "application selected dispatch requires execution-trace facts");
          return;
        }
        if (!prepare_selected_locked(launch, binding, packet,
                                     {.grid = grid, .workgroup = workgroup}))
          return;
        (void)rollback.release();
      } else {
        if (core_.hsa_signal_create_fn(1, 0, nullptr, &launch->proxy) != HSA_STATUS_SUCCESS) {
          fail_locked(RuntimeStatus::InvalidFacts,
                      "application native completion allocation failed");
          return;
        }
        launch->native = launch->application_native = true;
        pending_.push_back(launch);
        ++statistics_.native_submitted;
        start_observer_locked();
        condition_.notify_all();
      }
    }
    // Preserve the packet representation and dependencies. Selected transport
    // also wraps kernargs and strengthens fences for system-visible reports.
    if (is_extended) {
      extended.header = packet.header;
      extended.kernarg_address = packet.kernarg_address;
      extended.group_segment_size = packet.group_segment_size;
      extended.private_segment_size = packet.private_segment_size;
      extended.completion_signal = launch->proxy;
      writer(&extended, 1);
    } else {
      packet.completion_signal = launch->proxy;
      writer(&packet, 1);
    }
  }

  void intercept(QueueState &queue, const void *packets, uint64_t count,
                 hsa_amd_queue_intercept_packet_writer writer) {
    if (packets == nullptr || writer == nullptr || count == 0) {
      return;
    }
    if (application_profile_) {
      intercept_application(queue, packets, count, writer);
      return;
    }
    hsa_kernel_dispatch_packet_t packet{};
    std::memcpy(&packet, packets, sizeof(packet));
    const uint16_t type = packet.header & 0xff;
    if (count == 1 && (type == HSA_PACKET_TYPE_BARRIER_AND || type == HSA_PACKET_TYPE_BARRIER_OR)) {
      {
        std::lock_guard lock(mutex_);
        if (status_ != RuntimeStatus::Success) {
          fail_packet_completion_locked(packet);
          return;
        }
      }
      writer(packets, count);
      return;
    }
    if (count != 1 || type != HSA_PACKET_TYPE_KERNEL_DISPATCH) {
      const auto *bytes = static_cast<const uint8_t *>(packets);
      bool rejected = false;
      {
        std::lock_guard lock(mutex_);
        rejected = status_ != RuntimeStatus::Success;
        for (uint64_t index = 0; index < count; ++index) {
          std::memcpy(&packet, bytes + index * sizeof(packet), sizeof(packet));
          const uint16_t packet_type = packet.header & 0xff;
          if (packet_type != HSA_PACKET_TYPE_BARRIER_AND &&
              packet_type != HSA_PACKET_TYPE_BARRIER_OR) {
            if (!rejected) {
              fail_locked(RuntimeStatus::InvalidFacts,
                          "only individual kernel dispatches are admitted");
              rejected = true;
            }
          }
        }
        if (rejected) {
          // A GPU barrier waits for dependency value zero; a dispatch failure
          // value of -1 cannot satisfy it. Fail the discarded barriers too,
          // while preserving signals owned by already pending dispatches.
          for (uint64_t index = 0; index < count; ++index) {
            std::memcpy(&packet, bytes + index * sizeof(packet), sizeof(packet));
            fail_packet_completion_locked(packet);
          }
        }
      }
      if (rejected) {
        return;
      }
      // A barrier-only batch retains the individual-barrier passthrough
      // behavior. Call downstream tools outside the runtime mutex.
      for (uint64_t index = 0; index < count; ++index) {
        const uint8_t *current = bytes + index * sizeof(packet);
        std::memcpy(&packet, current, sizeof(packet));
        const uint16_t packet_type = packet.header & 0xff;
        if (packet_type == HSA_PACKET_TYPE_BARRIER_AND ||
            packet_type == HSA_PACKET_TYPE_BARRIER_OR) {
          writer(current, 1);
        }
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
      const std::unordered_map<uint64_t, Signal>::iterator signal =
          signals_.find(packet.completion_signal.handle);
      if (active_gpu_ && !pending_.empty()) {
        const std::shared_ptr<Launch> &active = pending_.back();
        if (active->native && core_.hsa_signal_load_scacquire_fn(active->original) == 0) {
          active->native_completion_visible = true;
          active_gpu_ = false;
        }
      }
      const EntryBinding *binding = nullptr;
      if (module_) {
        for (const EntryBinding &candidate : module_->entries) {
          if (candidate.kernel_object == packet.kernel_object) {
            binding = &candidate;
            break;
          }
        }
      }
      if (packet.setup != 1 || closed_ || status_ != RuntimeStatus::Success ||
          statistics_.submitted + statistics_.native_submitted >= CaptureLimits{}.executions ||
          binding == nullptr || active_gpu_ || signal == signals_.end() ||
          !signal->second.dedicated || signal->second.used ||
          core_.hsa_signal_load_scacquire_fn(packet.completion_signal) != 1) {
        reject_packet_locked(
            packet,
            "packet violates binding, single-queue/nonoverlap or dedicated-signal requirements");
        return;
      }
      const LaunchDimensions dimensions = {
          {packet.grid_size_x, packet.grid_size_y, packet.grid_size_z},
          {packet.workgroup_size_x, packet.workgroup_size_y, packet.workgroup_size_z}};
      launch->module = module_;
      launch->queue = queue.queue;
      launch->sequence = ++next_sequence_;
      if (binding->disposition == EntryDisposition::Native) {
        const EntryFacts &entry = binding->binding;
        const uint16_t acquire = (packet.header >> HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE) & 3;
        const uint16_t release_scope =
            (packet.header >> HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE) & 3;
        if (dimensions.grid != dimensions.workgroup || dimensions.workgroup[0] == 0 ||
            dimensions.workgroup[0] > entry.maximum_workgroup_size ||
            dimensions.workgroup[1] != 1 || dimensions.workgroup[2] != 1 ||
            (entry.required_workgroup_size[0] != 0 &&
             dimensions.workgroup != entry.required_workgroup_size) ||
            packet.private_segment_size != entry.original_private_bytes ||
            packet.group_segment_size < entry.original_group_bytes ||
            packet.group_segment_size > 65536 || acquire != HSA_FENCE_SCOPE_SYSTEM ||
            release_scope != HSA_FENCE_SCOPE_SYSTEM ||
            (reinterpret_cast<uintptr_t>(packet.kernarg_address) % binding->kernarg_alignment) !=
                0 ||
            !host_arguments_locked(packet.kernarg_address, entry.original_kernarg_bytes)) {
          reject_packet_locked(packet,
                               "native packet violates original ABI or controlled launch profile");
          return;
        }
        launch->native = true;
        launch->original = packet.completion_signal;
        pending_.push_back(launch);
        signal->second.used = signal->second.pending = true;
        active_gpu_ = true;
        ++statistics_.native_submitted;
        (void)rollback.release();
        condition_.notify_all();
      } else {
        if (!prepare_selected_locked(launch, binding, packet, dimensions))
          return;
        signal->second.used = signal->second.pending = true;
        active_gpu_ = true;
        (void)rollback.release();
        condition_.notify_all();
      }
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
        if (stop_ && pending_.empty()) {
          return;
        }
        launch = pending_.front();
      }
      const hsa_signal_value_t observed = core_.hsa_signal_wait_scacquire_fn(
          launch->native && !launch->application_native ? launch->original : launch->proxy,
          HSA_SIGNAL_CONDITION_LT, 1, 1000000, HSA_WAIT_STATE_BLOCKED);
      if (observed != 0) {
        std::lock_guard lock(mutex_);
        // Controlled admission relies on the newest launch remaining at back().
        // Application launches may complete out of order, including selected traces.
        if (application_profile_)
          std::rotate(pending_.begin(), pending_.begin() + 1, pending_.end());
        continue; // A timeout/fault is never completion/visibility proof.
      }
      if (launch->native) {
        const hsa_status_t destroyed = launch->application_native
                                           ? core_.hsa_signal_destroy_fn(launch->proxy)
                                           : HSA_STATUS_SUCCESS;
        std::lock_guard lock(mutex_);
        if (destroyed != HSA_STATUS_SUCCESS)
          fail_locked(RuntimeStatus::InvalidTransition,
                      "native private completion signal cleanup failed");
        if (!launch->application_native) {
          signals_[launch->original.handle].pending = false;
          // Admission may already have observed this completion and published
          // another launch. Keep that launch's nonoverlap guard intact.
          if (!launch->native_completion_visible)
            active_gpu_ = false;
        }
        ++statistics_.native_completed;
        pending_.pop_front();
        condition_.notify_all();
        continue;
      }
      void *source = nullptr;
      RuntimeStatus observed_status;
      {
        std::lock_guard lock(mutex_);
        observed_status = launch->lifecycle->observe(true, true, [&] {
          if (launch->original.handle != 0)
            core_.hsa_signal_subtract_screlease_fn(launch->original, 1);
          return true;
        });
        if (observed_status != RuntimeStatus::Success)
          fail_locked(observed_status, "completion forwarding failed");
        if (launch->original.handle != 0) {
          signals_[launch->original.handle].pending = false;
          ++statistics_.forwarded;
        }
        if (!application_profile_)
          active_gpu_ = false;
        source = slots_[launch->slot - 1].memory;
      }
      // Snapshot before allowing slot reuse, independently of file I/O. The
      // observer alone mutates this lifecycle after admission. Original args
      // were copied before publication and are never read from the app again.
      OwnedSnapshot snapshot;
      const RuntimeStatus copied =
          observed_status == RuntimeStatus::Success
              ? launch->lifecycle->snapshot(
                    [source](std::span<uint8_t> bytes) {
                      std::memcpy(bytes.data(), source, bytes.size());
                      return CopyResult{RuntimeStatus::Success, bytes.size()};
                    },
                    snapshot)
              : observed_status;
      const hsa_status_t signal_destroyed = core_.hsa_signal_destroy_fn(launch->proxy);
      const hsa_status_t wrapper_freed = amd_.hsa_amd_memory_pool_free_fn(launch->wrapper);
      const hsa_status_t context_freed = amd_.hsa_amd_memory_pool_free_fn(launch->context);
      {
        std::lock_guard lock(mutex_);
        if (copied == RuntimeStatus::Success) {
          slots_[launch->slot - 1].leased = false;
          ++statistics_.copied;
          completed_.emplace(launch->serial, std::move(snapshot));
        } else {
          fail_locked(copied, "independent host snapshot failed; slice remains retained");
          completed_.emplace(launch->serial, std::nullopt);
        }
        if (signal_destroyed != HSA_STATUS_SUCCESS || wrapper_freed != HSA_STATUS_SUCCESS ||
            context_freed != HSA_STATUS_SUCCESS)
          fail_locked(RuntimeStatus::InvalidTransition,
                      "selected completion resource cleanup failed");
        pending_.pop_front();
        condition_.notify_all();
      }
    }
  }

  void collector_loop() {
    uint64_t next_serial = 1;
    for (;;) {
      std::optional<OwnedSnapshot> snapshot;
      {
        std::unique_lock lock(mutex_);
        condition_.wait(lock, [&] { return stop_ || completed_.contains(next_serial); });
        const std::map<uint64_t, std::optional<OwnedSnapshot>>::iterator found =
            completed_.find(next_serial);
        if (found == completed_.end()) {
          if (stop_)
            break;
          continue;
        }
        snapshot = std::move(found->second);
        completed_.erase(found);
      }
      // Completion may arrive out of order. The stream and its storage
      // generations retain submission order without blocking observation/copy.
      if (!snapshot)
        writer_.cancel(RuntimeStatus::CopyFailed);
      const RuntimeStatus written =
          snapshot ? writer_.append(*snapshot) : RuntimeStatus::CopyFailed;
      {
        std::lock_guard lock(mutex_);
        if (written == RuntimeStatus::Success) {
          ++statistics_.written;
        } else {
          fail_locked(written, "capture execution write failed");
        }
        ++processed_;
        ++next_serial;
        condition_.notify_all();
      }
    }
    RuntimeStatus status;
    {
      std::lock_guard lock(mutex_);
      status = status_;
    }
    if (status != RuntimeStatus::Success) {
      writer_.cancel(status);
    }
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
  Digest expected_profile_{};
  const bool application_profile_;
  const std::vector<std::shared_ptr<const LoadedNativeImage>> native_images_;
  std::mutex mutex_;
  std::mutex finish_mutex_;
  std::condition_variable condition_;
  std::unordered_map<uint64_t, Blob> readers_;
  std::unordered_map<uint64_t, Signal> signals_;
  std::shared_ptr<Module> module_;
  std::vector<std::shared_ptr<Module>> native_modules_;
  hsa_amd_memory_pool_t pool_{};
  std::vector<std::unique_ptr<QueueState>> queues_;
  uint32_t queue_creations_ = 0;
  // Enclosing AMD create calls remain live through replacement queue cleanup.
  uint32_t native_queue_calls_ = 0;
  uint32_t callbacks_ = 0;
  uint32_t shutdown_calls_ = 0;
  // ROCR acquires the installing hsa_init reference before calling OnLoad.
  uint32_t hsa_references_ = 1;
  bool unload_seen_ = false;
  uint64_t next_sequence_ = 0;
  bool workers_started_ = false;
  bool queue_cleanup_failed_ = false;
  std::vector<ReportSlot> slots_;
  std::deque<std::shared_ptr<Launch>> pending_;
  std::map<uint64_t, std::optional<OwnedSnapshot>> completed_;
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

// HIP's earlier-registered exit handlers still call these hooks before ROCR
// unloads its tools. An automatic C++ owner would destroy Runtime too early.
// OnUnload owns deletion, deferred through any enclosing HSA shutdown call.
Runtime *runtime = nullptr;
// Outlive Runtime: final shutdown can delete it while this lock is held.
std::mutex hsa_lifecycle_mutex;
// ROCR resets this process-lifetime table in place after unloading its tools.
// An init already waiting on the lifecycle lock must use its current function,
// since a saved tool function may have been unmapped by the preceding shutdown.
std::atomic<CoreApiTable *> lifecycle_core_table = nullptr;
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
hsa_status_t HSA_API native_queue_create(hsa_agent_t agent, hsa_amd_queue_create_desc_t *descs,
                                         uint32_t count) {
  return boundary([&] { return runtime->native_queue_create(agent, descs, count); });
}
hsa_status_t HSA_API signal_create(hsa_signal_value_t value, uint32_t consumers,
                                   const hsa_agent_t *agents, hsa_signal_t *signal) {
  return boundary([&] { return runtime->signal_create(value, consumers, agents, signal); });
}
hsa_status_t HSA_API amd_signal_create(hsa_signal_value_t value, uint32_t consumers,
                                       const hsa_agent_t *agents, uint64_t attributes,
                                       hsa_signal_t *signal) {
  return boundary(
      [&] { return runtime->amd_signal_create(value, consumers, agents, attributes, signal); });
}
hsa_status_t HSA_API signal_destroy(hsa_signal_t signal) {
  return boundary([&] { return runtime->signal_destroy(signal); });
}
hsa_status_t HSA_API profiling(hsa_queue_t *queue, int enabled) {
  return boundary([&] { return runtime->profiling(queue, enabled); });
}
hsa_status_t HSA_API initialize() noexcept {
  std::lock_guard lock(hsa_lifecycle_mutex);
  if (runtime)
    return runtime->initialize();
  const CoreApiTable *core = lifecycle_core_table.load(std::memory_order_acquire);
  if (core == nullptr || core->hsa_init_fn == nullptr || core->hsa_init_fn == initialize)
    return HSA_STATUS_ERROR_NOT_INITIALIZED;
  return core->hsa_init_fn();
}
hsa_status_t HSA_API shutdown() noexcept {
  std::lock_guard lock(hsa_lifecycle_mutex);
  Runtime *active = runtime;
  if (active == nullptr) {
    const CoreApiTable *core = lifecycle_core_table.load(std::memory_order_acquire);
    if (core == nullptr || core->hsa_shut_down_fn == nullptr || core->hsa_shut_down_fn == shutdown)
      return HSA_STATUS_ERROR_NOT_INITIALIZED;
    return core->hsa_shut_down_fn();
  }
  if (active->application_profile() && active->final_hsa_reference() && !active->finished() &&
      active->finish() != 0)
    return HSA_STATUS_ERROR;
  if (!active->application_profile() && !active->finished()) {
    diagnostic("call rj_consan_finish before hsa_shut_down");
    return HSA_STATUS_ERROR;
  }
  active->begin_shutdown();
  const hsa_status_t status = active->core().hsa_shut_down_fn();
  if (status == HSA_STATUS_SUCCESS && active->application_profile())
    active->release_hsa_reference();
  if (active->end_shutdown()) {
    runtime = nullptr;
    delete active;
  }
  return status;
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
      table->amd_ext_->version.minor_id < kAmdTablePrefixBytes) {
    return false;
  }
  const CoreApiTable &core = *table->core_;
  const AmdExtTable &amd = *table->amd_ext_;
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
      !amd.hsa_amd_profiling_set_profiler_enabled_fn) {
    return false;
  }
  const char *artifacts = std::getenv("RJ_CONSAN_ARTIFACT_DIR");
  const char *capture = std::getenv("RJ_CONSAN_CAPTURE_PATH");
  const char *application = std::getenv("RJ_CONSAN_APPLICATION_PROFILE");
  if (application != nullptr && std::strcmp(application, "1") != 0) {
    diagnostic("RJ_CONSAN_APPLICATION_PROFILE must be 1 when configured");
    return false;
  }
  if (application != nullptr && core.hsa_init_fn == nullptr) {
    diagnostic("application profile requires the HSA initialization API");
    return false;
  }
  std::vector<std::shared_ptr<const LoadedNativeImage>> native_images;
  const char *native_paths = std::getenv("RJ_CONSAN_NATIVE_IMAGES");
  if (native_paths != nullptr) {
    if (application == nullptr || *native_paths == '\0') {
      diagnostic("native image configuration requires the application profile");
      return false;
    }
    const std::string_view paths(native_paths);
    uint64_t total_bytes = 0;
    size_t offset = 0;
    for (;;) {
      const size_t separator = paths.find(':', offset);
      const std::string_view path = paths.substr(
          offset, separator == std::string_view::npos ? separator : separator - offset);
      if (path.empty() || native_images.size() >= 16) {
        diagnostic("native image list is empty or exceeds its image budget");
        return false;
      }
      util::StringDiagnostic native_diagnostic;
      FailureOr<LoadedNativeImage> loaded =
          load_native_image(std::string(path), native_diagnostic.emitter());
      if (loaded.failed()) {
        diagnostic(native_diagnostic.message());
        return false;
      }
      const uint64_t bytes = loaded.value().original.size();
      if (bytes > ContractLimits{}.image_bytes - total_bytes ||
          std::ranges::any_of(
              native_images, [&](const std::shared_ptr<const LoadedNativeImage> &image) {
                return image->inventory.identity == loaded.value().inventory.identity;
              })) {
        diagnostic("native images are duplicate or exceed their aggregate byte budget");
        return false;
      }
      total_bytes += bytes;
      native_images.push_back(std::make_shared<const LoadedNativeImage>(std::move(loaded.value())));
      if (separator == std::string_view::npos)
        break;
      offset = separator + 1;
    }
  }
  const bool selected = artifacts != nullptr && *artifacts != '\0';
  if (!selected && native_images.empty()) {
    diagnostic("configure RJ_CONSAN_ARTIFACT_DIR or application-profile RJ_CONSAN_NATIVE_IMAGES");
    return false;
  }
  if (selected && (capture == nullptr || *capture == '\0')) {
    diagnostic("RJ_CONSAN_CAPTURE_PATH is required for a selected artifact");
    return false;
  }
  const char *profile = std::getenv("RJ_CONSAN_SELECTION_PROFILE");
  Digest expected_profile{};
  if (selected && (profile == nullptr || std::strlen(profile) != expected_profile.size() * 2)) {
    diagnostic("RJ_CONSAN_SELECTION_PROFILE must contain a 64-digit expected profile");
    return false;
  }
  const auto hex_digit = [](char value) -> int {
    if (value >= '0' && value <= '9') {
      return value - '0';
    }
    if (value >= 'a' && value <= 'f') {
      return value - 'a' + 10;
    }
    if (value >= 'A' && value <= 'F') {
      return value - 'A' + 10;
    }
    return -1;
  };
  for (size_t index = 0; selected && index < expected_profile.size(); ++index) {
    const int high = hex_digit(profile[index * 2]);
    const int low = hex_digit(profile[index * 2 + 1]);
    if (high < 0 || low < 0) {
      diagnostic("RJ_CONSAN_SELECTION_PROFILE contains a non-hexadecimal digit");
      return false;
    }
    expected_profile[index] = static_cast<uint8_t>((high << 4) | low);
  }
  runtime = new Runtime(core, amd, selected ? artifacts : "", capture ? capture : "",
                        expected_profile, application != nullptr, std::move(native_images));
  CoreApiTable &patched = *table->core_;
  lifecycle_core_table.store(&patched, std::memory_order_release);
  if (application != nullptr)
    patched.hsa_init_fn = initialize;
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
  if (amd.hsa_amd_signal_create_fn) {
    table->amd_ext_->hsa_amd_signal_create_fn = amd_signal_create;
  }
  table->amd_ext_->hsa_amd_profiling_set_profiler_enabled_fn = profiling;
  table->amd_ext_->hsa_amd_queue_intercept_create_fn = intercept_create;
  table->amd_ext_->hsa_amd_queue_intercept_register_fn = intercept_register;
  if (amd.version.minor_id >= kNativeQueueTableBytes && amd.hsa_amd_queue_create_fn != nullptr)
    table->amd_ext_->hsa_amd_queue_create_fn = native_queue_create;
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
  if (!runtime) {
    return;
  }
  if (runtime->queue_cleanup_failed()) {
    diagnostic("cannot unload ConSan while a partially destroyed queue may call its interceptor");
    std::abort();
  }
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
    CoreApiTable &core = *installed_core;
    const CoreApiTable &saved = runtime->core();
#define RJ_CONSAN_RESTORE_HOOK(table, original, field, hook)                                       \
  do {                                                                                             \
    if ((table).field == (hook)) {                                                                 \
      (table).field = (original).field;                                                            \
    }                                                                                              \
  } while (false)
    RJ_CONSAN_RESTORE_HOOK(core, saved, hsa_init_fn, initialize);
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
    AmdExtTable &amd = *installed_amd;
    const AmdExtTable &original = runtime->amd();
    RJ_CONSAN_RESTORE_HOOK(amd, original, hsa_amd_signal_create_fn, amd_signal_create);
    RJ_CONSAN_RESTORE_HOOK(amd, original, hsa_amd_profiling_set_profiler_enabled_fn, profiling);
    RJ_CONSAN_RESTORE_HOOK(amd, original, hsa_amd_queue_intercept_create_fn, intercept_create);
    RJ_CONSAN_RESTORE_HOOK(amd, original, hsa_amd_queue_intercept_register_fn, intercept_register);
    if (original.hsa_amd_queue_create_fn != nullptr)
      RJ_CONSAN_RESTORE_HOOK(amd, original, hsa_amd_queue_create_fn, native_queue_create);
#undef RJ_CONSAN_RESTORE_HOOK
  }
  if (!runtime->defer_unload()) {
    Runtime *unloaded = runtime;
    runtime = nullptr;
    if (!retain)
      delete unloaded;
  }
  installed_core = nullptr;
  installed_amd = nullptr;
}
