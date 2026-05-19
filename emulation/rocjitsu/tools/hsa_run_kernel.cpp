// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "tools/hsa_run_kernel.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <unordered_map>

#if defined(RJ_HAS_HSA_RUNTIME)
#include "rocjitsu/base/rj_compiler.h"
RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
RJ_DIAGNOSTIC_POP
#endif

namespace rocjitsu::tools {

namespace {

constexpr int kCodeObjectError = 2;
constexpr int kHsaError = 3;
constexpr int kTimeoutError = 4;
constexpr int kOutputCopyError = 5;

void add_error(ToolResult<HsaRunOutput> &result, int exit_code, std::string message) {
  result.errors.push_back({exit_code, std::move(message)});
}

[[nodiscard]] bool read_file(const std::string &path, std::vector<uint8_t> &bytes,
                             std::string &error) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in) {
    error = "failed to open file: " + path;
    return false;
  }

  const auto size = in.tellg();
  if (size < 0) {
    error = "failed to determine file size: " + path;
    return false;
  }

  bytes.resize(static_cast<size_t>(size));
  in.seekg(0, std::ios::beg);
  in.read(reinterpret_cast<char *>(bytes.data()), size);
  if (!in) {
    error = "failed to read file: " + path;
    return false;
  }

  return true;
}

[[nodiscard]] size_t patch_size(const KernelArgPatch &patch) {
  switch (patch.kind) {
  case KernelArgKind::Pointer:
  case KernelArgKind::U64:
    return sizeof(uint64_t);
  case KernelArgKind::U32:
  case KernelArgKind::I32:
  case KernelArgKind::F32:
    return sizeof(uint32_t);
  case KernelArgKind::RawBytes:
    return patch.bytes.size();
  }
  return 0;
}

[[nodiscard]] size_t required_kernarg_size(const HsaRunOptions &options) {
  size_t required = std::max(options.kernarg_size, options.kernarg_template.size());
  for (const auto &patch : options.arg_patches)
    required = std::max(required, patch.offset + patch_size(patch));
  return required;
}

#if defined(RJ_HAS_HSA_RUNTIME)

[[nodiscard]] std::string hsa_error(hsa_status_t status, std::string_view context) {
  const char *text = nullptr;
  hsa_status_string(status, &text);
  std::string message(context);
  message += ": ";
  message += text ? text : "unknown HSA error";
  return message;
}

struct HsaRuntime {
  bool initialized = false;

  ~HsaRuntime() {
    if (initialized)
      hsa_shut_down();
  }
};

struct HsaReader {
  hsa_code_object_reader_t reader{};
  bool valid = false;

  ~HsaReader() {
    if (valid)
      hsa_code_object_reader_destroy(reader);
  }
};

struct HsaExecutable {
  hsa_executable_t executable{};
  bool valid = false;

  ~HsaExecutable() {
    if (valid)
      hsa_executable_destroy(executable);
  }
};

struct HsaQueue {
  hsa_queue_t *queue = nullptr;

  ~HsaQueue() {
    if (queue != nullptr)
      hsa_queue_destroy(queue);
  }
};

struct HsaSignal {
  hsa_signal_t signal{};
  bool valid = false;

  ~HsaSignal() {
    if (valid)
      hsa_signal_destroy(signal);
  }
};

struct DeviceAllocation {
  void *ptr = nullptr;

  DeviceAllocation() = default;
  DeviceAllocation(const DeviceAllocation &) = delete;
  DeviceAllocation &operator=(const DeviceAllocation &) = delete;

  DeviceAllocation(DeviceAllocation &&other) noexcept : ptr(other.ptr) { other.ptr = nullptr; }

  DeviceAllocation &operator=(DeviceAllocation &&other) noexcept {
    if (this == &other)
      return *this;
    reset();
    ptr = other.ptr;
    other.ptr = nullptr;
    return *this;
  }

  ~DeviceAllocation() { reset(); }

  void reset() {
    if (ptr != nullptr) {
      hsa_amd_memory_pool_free(ptr);
      ptr = nullptr;
    }
  }
};

struct RuntimeBuffer {
  std::string name;
  size_t size = 0;
  bool copy_output = false;
  DeviceAllocation allocation;
};

[[nodiscard]] hsa_agent_t find_cpu_agent() {
  hsa_agent_t cpu{};
  hsa_iterate_agents(
      [](hsa_agent_t agent, void *data) -> hsa_status_t {
        hsa_device_type_t type{};
        hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type);
        if (type == HSA_DEVICE_TYPE_CPU) {
          *static_cast<hsa_agent_t *>(data) = agent;
          return HSA_STATUS_INFO_BREAK;
        }
        return HSA_STATUS_SUCCESS;
      },
      &cpu);
  return cpu;
}

[[nodiscard]] hsa_agent_t find_gpu_agent(uint32_t requested_index) {
  struct Ctx {
    uint32_t requested = 0;
    uint32_t seen = 0;
    hsa_agent_t gpu{};
  } ctx{requested_index, 0, {}};

  hsa_iterate_agents(
      [](hsa_agent_t agent, void *data) -> hsa_status_t {
        auto *ctx = static_cast<Ctx *>(data);
        hsa_device_type_t type{};
        hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type);
        if (type != HSA_DEVICE_TYPE_GPU)
          return HSA_STATUS_SUCCESS;

        if (ctx->seen == ctx->requested) {
          ctx->gpu = agent;
          return HSA_STATUS_INFO_BREAK;
        }
        ++ctx->seen;
        return HSA_STATUS_SUCCESS;
      },
      &ctx);

  return ctx.gpu;
}

[[nodiscard]] std::string agent_isa_name(hsa_agent_t gpu) {
  hsa_isa_t isa{};
  if (hsa_agent_get_info(gpu, HSA_AGENT_INFO_ISA, &isa) != HSA_STATUS_SUCCESS)
    return {};

  char name[128] = {};
  if (hsa_isa_get_info_alt(isa, HSA_ISA_INFO_NAME, name) != HSA_STATUS_SUCCESS)
    return {};

  return name;
}

[[nodiscard]] hsa_amd_memory_pool_t find_pool(hsa_agent_t agent, hsa_amd_segment_t segment,
                                              bool host_accessible) {
  struct Ctx {
    hsa_amd_segment_t segment;
    bool host_accessible;
    hsa_amd_memory_pool_t pool{};
  } ctx{segment, host_accessible, {}};

  hsa_amd_agent_iterate_memory_pools(
      agent,
      [](hsa_amd_memory_pool_t pool, void *data) -> hsa_status_t {
        auto *ctx = static_cast<Ctx *>(data);
        hsa_amd_segment_t segment{};
        hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &segment);
        if (segment != ctx->segment)
          return HSA_STATUS_SUCCESS;

        if (ctx->host_accessible) {
          bool accessible = false;
          hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_ACCESSIBLE_BY_ALL,
                                       &accessible);
          if (!accessible)
            return HSA_STATUS_SUCCESS;
        }

        ctx->pool = pool;
        return HSA_STATUS_INFO_BREAK;
      },
      &ctx);

  return ctx.pool;
}

[[nodiscard]] bool allocate_pool(hsa_amd_memory_pool_t pool, size_t size, DeviceAllocation &alloc,
                                 ToolResult<HsaRunOutput> &result, std::string_view context) {
  void *ptr = nullptr;
  const hsa_status_t status = hsa_amd_memory_pool_allocate(pool, size, 0, &ptr);
  if (status != HSA_STATUS_SUCCESS || ptr == nullptr) {
    add_error(result, kHsaError, hsa_error(status, context));
    return false;
  }
  alloc.ptr = ptr;
  return true;
}

[[nodiscard]] RuntimeBuffer *find_buffer(std::vector<RuntimeBuffer> &buffers,
                                         const std::string &name) {
  auto it = std::ranges::find_if(buffers, [&](const RuntimeBuffer &buffer) {
    return buffer.name == name;
  });
  return it == buffers.end() ? nullptr : &*it;
}

template <typename T> void write_scalar(uint8_t *dst, T value) {
  std::memcpy(dst, &value, sizeof(T));
}

[[nodiscard]] bool apply_arg_patch(const KernelArgPatch &patch,
                                   std::vector<RuntimeBuffer> &buffers, uint8_t *kernarg,
                                   size_t kernarg_size, ToolResult<HsaRunOutput> &result) {
  const size_t size = patch_size(patch);
  if (patch.offset + size > kernarg_size) {
    add_error(result, kHsaError, "kernel argument patch exceeds kernarg allocation");
    return false;
  }

  uint8_t *dst = kernarg + patch.offset;
  switch (patch.kind) {
  case KernelArgKind::Pointer: {
    RuntimeBuffer *buffer = find_buffer(buffers, patch.buffer_name);
    if (buffer == nullptr) {
      add_error(result, kHsaError, "unknown buffer in pointer argument: " + patch.buffer_name);
      return false;
    }
    // AMDHSA kernargs store device pointers as 64-bit addresses.
    write_scalar<uint64_t>(dst, reinterpret_cast<uint64_t>(buffer->allocation.ptr));
    return true;
  }
  case KernelArgKind::U32: {
    uint32_t value = 0;
    std::memcpy(&value, patch.bytes.data(), sizeof(value));
    write_scalar<uint32_t>(dst, value);
    return true;
  }
  case KernelArgKind::U64: {
    uint64_t value = 0;
    std::memcpy(&value, patch.bytes.data(), sizeof(value));
    write_scalar<uint64_t>(dst, value);
    return true;
  }
  case KernelArgKind::I32: {
    int32_t value = 0;
    std::memcpy(&value, patch.bytes.data(), sizeof(value));
    write_scalar<int32_t>(dst, value);
    return true;
  }
  case KernelArgKind::F32: {
    float value = 0.0f;
    std::memcpy(&value, patch.bytes.data(), sizeof(value));
    write_scalar<float>(dst, value);
    return true;
  }
  case KernelArgKind::RawBytes:
    std::memcpy(dst, patch.bytes.data(), patch.bytes.size());
    return true;
  }

  add_error(result, kHsaError, "unknown kernel argument patch kind");
  return false;
}

[[nodiscard]] uint16_t dispatch_dimensions(const HsaRunOptions &options) {
  if (options.grid_z > 1 || options.workgroup_z > 1)
    return 3;
  if (options.grid_y > 1 || options.workgroup_y > 1)
    return 2;
  return 1;
}

#endif // defined(RJ_HAS_HSA_RUNTIME)

} // namespace

ToolResult<HsaRunOutput> run_hsa_kernel(const HsaRunOptions &options) {
  ToolResult<HsaRunOutput> result;

#if !defined(RJ_HAS_HSA_RUNTIME)
  (void)options;
  add_error(result, kHsaError, "rj_hsa_run was built without ROCm HSA runtime support");
  return result;
#else
  if (options.kernel_name.empty() && options.kernel_symbol.empty()) {
    add_error(result, kCodeObjectError, "kernel name or kernel symbol is required");
    return result;
  }

  std::vector<uint8_t> owned_code_object;
  std::span<const uint8_t> code_object(options.code_object_bytes.data(),
                                       options.code_object_bytes.size());
  if (code_object.empty()) {
    std::string error;
    if (!read_file(options.code_object_path, owned_code_object, error)) {
      add_error(result, kCodeObjectError, error);
      return result;
    }
    code_object = std::span<const uint8_t>(owned_code_object.data(), owned_code_object.size());
  }

  if (code_object.empty()) {
    add_error(result, kCodeObjectError, "code object is empty");
    return result;
  }

  HsaRuntime runtime;
  hsa_status_t status = hsa_init();
  if (status != HSA_STATUS_SUCCESS) {
    add_error(result, kHsaError, hsa_error(status, "hsa_init failed"));
    return result;
  }
  runtime.initialized = true;

  hsa_agent_t gpu = find_gpu_agent(options.agent_index);
  if (gpu.handle == 0) {
    add_error(result, kHsaError, "no GPU agent found");
    return result;
  }

  hsa_agent_t cpu = find_cpu_agent();
  if (cpu.handle == 0) {
    add_error(result, kHsaError, "no CPU agent found");
    return result;
  }

  result.value.agent_isa = agent_isa_name(gpu);
  if (!options.require_agent_isa.empty() &&
      result.value.agent_isa.find(options.require_agent_isa) == std::string::npos) {
    add_error(result, kHsaError,
              "selected GPU ISA '" + result.value.agent_isa + "' does not match required text '" +
                  options.require_agent_isa + "'");
    return result;
  }

  HsaReader reader;
  status = hsa_code_object_reader_create_from_memory(code_object.data(), code_object.size(),
                                                     &reader.reader);
  if (status != HSA_STATUS_SUCCESS) {
    add_error(result, kCodeObjectError, hsa_error(status, "failed to create code object reader"));
    return result;
  }
  reader.valid = true;

  HsaExecutable executable;
  status = hsa_executable_create_alt(HSA_PROFILE_FULL, HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT,
                                     nullptr, &executable.executable);
  if (status != HSA_STATUS_SUCCESS) {
    add_error(result, kHsaError, hsa_error(status, "failed to create HSA executable"));
    return result;
  }
  executable.valid = true;

  status = hsa_executable_load_agent_code_object(executable.executable, gpu, reader.reader,
                                                 nullptr, nullptr);
  if (status != HSA_STATUS_SUCCESS) {
    add_error(result, kHsaError, hsa_error(status, "failed to load code object for GPU agent"));
    return result;
  }

  status = hsa_executable_freeze(executable.executable, nullptr);
  if (status != HSA_STATUS_SUCCESS) {
    add_error(result, kHsaError, hsa_error(status, "failed to freeze HSA executable"));
    return result;
  }

  const std::string symbol_name =
      options.kernel_symbol.empty() ? options.kernel_name + ".kd" : options.kernel_symbol;
  hsa_executable_symbol_t symbol{};
  status =
      hsa_executable_get_symbol_by_name(executable.executable, symbol_name.c_str(), &gpu, &symbol);
  if (status != HSA_STATUS_SUCCESS) {
    add_error(result, kCodeObjectError, hsa_error(status, "failed to resolve kernel symbol"));
    return result;
  }

  uint64_t kernel_object = 0;
  status = hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT,
                                          &kernel_object);
  if (status != HSA_STATUS_SUCCESS || kernel_object == 0) {
    add_error(result, kCodeObjectError, hsa_error(status, "failed to read kernel object handle"));
    return result;
  }

  hsa_amd_memory_pool_t gpu_pool = find_pool(gpu, HSA_AMD_SEGMENT_GLOBAL, false);
  hsa_amd_memory_pool_t kernarg_pool = find_pool(cpu, HSA_AMD_SEGMENT_GLOBAL, true);
  if (gpu_pool.handle == 0 || kernarg_pool.handle == 0) {
    add_error(result, kHsaError, "failed to find required HSA memory pools");
    return result;
  }

  hsa_agent_t agents[] = {cpu, gpu};
  std::vector<RuntimeBuffer> buffers;
  buffers.reserve(options.buffers.size());
  for (const auto &spec : options.buffers) {
    if (spec.name.empty() || spec.size == 0) {
      add_error(result, kHsaError, "buffer name and nonzero size are required");
      return result;
    }
    if (spec.input.size() > spec.size) {
      add_error(result, kHsaError, "input data is larger than buffer: " + spec.name);
      return result;
    }

    RuntimeBuffer buffer;
    buffer.name = spec.name;
    buffer.size = spec.size;
    buffer.copy_output = spec.copy_output;
    if (!allocate_pool(gpu_pool, spec.size, buffer.allocation, result,
                       "failed to allocate GPU buffer")) {
      return result;
    }

    status = hsa_amd_agents_allow_access(2, agents, nullptr, buffer.allocation.ptr);
    if (status != HSA_STATUS_SUCCESS) {
      add_error(result, kHsaError, hsa_error(status, "failed to allow buffer agent access"));
      return result;
    }

    if (!spec.input.empty()) {
      status = hsa_memory_copy(buffer.allocation.ptr, spec.input.data(), spec.input.size());
      if (status != HSA_STATUS_SUCCESS) {
        add_error(result, kHsaError, hsa_error(status, "failed to copy input buffer"));
        return result;
      }
    } else if (spec.zero_fill) {
      // Device allocations are often not CPU-mapped on dGPUs; zero through HSA
      // instead of touching the pointer directly.
      std::vector<uint8_t> zeros(spec.size, 0);
      status = hsa_memory_copy(buffer.allocation.ptr, zeros.data(), zeros.size());
      if (status != HSA_STATUS_SUCCESS) {
        add_error(result, kHsaError, hsa_error(status, "failed to zero buffer"));
        return result;
      }
    }

    buffers.push_back(std::move(buffer));
  }

  const size_t kernarg_size = required_kernarg_size(options);
  DeviceAllocation kernarg;
  if (kernarg_size != 0) {
    if (!allocate_pool(kernarg_pool, kernarg_size, kernarg, result,
                       "failed to allocate kernarg buffer")) {
      return result;
    }

    status = hsa_amd_agents_allow_access(2, agents, nullptr, kernarg.ptr);
    if (status != HSA_STATUS_SUCCESS) {
      add_error(result, kHsaError, hsa_error(status, "failed to allow kernarg agent access"));
      return result;
    }

    std::memset(kernarg.ptr, 0, kernarg_size);
    if (!options.kernarg_template.empty())
      std::memcpy(kernarg.ptr, options.kernarg_template.data(), options.kernarg_template.size());

    for (const auto &patch : options.arg_patches) {
      if (!apply_arg_patch(patch, buffers, static_cast<uint8_t *>(kernarg.ptr), kernarg_size,
                           result)) {
        return result;
      }
    }
  } else if (!options.arg_patches.empty()) {
    add_error(result, kHsaError, "kernel argument patches require a kernarg allocation");
    return result;
  }

  uint32_t queue_size = 0;
  status = hsa_agent_get_info(gpu, HSA_AGENT_INFO_QUEUE_MAX_SIZE, &queue_size);
  if (status != HSA_STATUS_SUCCESS || queue_size == 0) {
    add_error(result, kHsaError, hsa_error(status, "failed to query queue size"));
    return result;
  }

  HsaQueue queue;
  status = hsa_queue_create(gpu, queue_size, HSA_QUEUE_TYPE_SINGLE, nullptr, nullptr, UINT32_MAX,
                            UINT32_MAX, &queue.queue);
  if (status != HSA_STATUS_SUCCESS) {
    add_error(result, kHsaError, hsa_error(status, "failed to create HSA queue"));
    return result;
  }

  HsaSignal signal;
  status = hsa_signal_create(1, 0, nullptr, &signal.signal);
  if (status != HSA_STATUS_SUCCESS) {
    add_error(result, kHsaError, hsa_error(status, "failed to create completion signal"));
    return result;
  }
  signal.valid = true;

  const uint64_t write_index = hsa_queue_add_write_index_relaxed(queue.queue, 1);
  auto *packet = static_cast<hsa_kernel_dispatch_packet_t *>(queue.queue->base_address) +
                 (write_index & (queue.queue->size - 1));

  std::memset(packet, 0, sizeof(*packet));
  packet->setup = dispatch_dimensions(options);
  packet->workgroup_size_x = options.workgroup_x;
  packet->workgroup_size_y = options.workgroup_y;
  packet->workgroup_size_z = options.workgroup_z;
  packet->grid_size_x = options.grid_x;
  packet->grid_size_y = options.grid_y;
  packet->grid_size_z = options.grid_z;
  packet->kernel_object = kernel_object;
  packet->kernarg_address = kernarg.ptr;
  packet->completion_signal = signal.signal;

  uint16_t header = HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE;
  header |= 1 << HSA_PACKET_HEADER_BARRIER;
  header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE;
  header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE;
  __atomic_store_n(reinterpret_cast<uint16_t *>(packet), header, __ATOMIC_RELEASE);

  const auto start = std::chrono::steady_clock::now();
  hsa_signal_store_relaxed(queue.queue->doorbell_signal, write_index);

  const uint64_t timeout_ns = options.timeout_ms * 1'000'000ULL;
  hsa_signal_value_t wait_value =
      hsa_signal_wait_scacquire(signal.signal, HSA_SIGNAL_CONDITION_LT, 1, timeout_ns,
                                HSA_WAIT_STATE_BLOCKED);
  const auto stop = std::chrono::steady_clock::now();
  result.value.elapsed_ns =
      static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start)
                                .count());

  if (wait_value != 0) {
    add_error(result, kTimeoutError, "kernel dispatch timed out or failed");
    return result;
  }

  for (const auto &buffer : buffers) {
    if (!buffer.copy_output)
      continue;
    HsaBufferOutput copied;
    copied.name = buffer.name;
    copied.bytes.resize(buffer.size);
    status = hsa_memory_copy(copied.bytes.data(), buffer.allocation.ptr, buffer.size);
    if (status != HSA_STATUS_SUCCESS) {
      add_error(result, kOutputCopyError, hsa_error(status, "failed to copy output buffer"));
      return result;
    }
    result.value.outputs.push_back(std::move(copied));
  }

  return result;
#endif
}

} // namespace rocjitsu::tools
