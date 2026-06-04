#include "rocjitsu_fuzzer/afl_runtime.h"

#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/code/amdgpu_elf_reader.h"
#include "rocjitsu/code/hsa_code_object_reader_rewriter.h"

#include <hip/hip_runtime_api.h>
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wgnu-anonymous-struct"
#pragma clang diagnostic ignored "-Wnested-anon-types"
#endif
#include <hsa/hsa.h>
#if defined(__clang__)
#pragma clang diagnostic pop
#endif

#include <dlfcn.h>
#include <errno.h>
#include <string.h>
#include <sys/shm.h>

#include <algorithm>
#include <cstring>
#include <limits>
#include <mutex>
#include <new>
#include <optional>
#include <span>
#include <stdio.h>
#include <stdlib.h>
#include <unordered_map>
#include <vector>

namespace {

using rocjitsu::EI_CLASS;
using rocjitsu::EI_MAGIC;
using rocjitsu::EI_MAGIC_SIZE;
using rocjitsu::Elf64_Ehdr;
using rocjitsu::Elf64_Phdr;
using rocjitsu::Elf64_Shdr;
using rocjitsu::ELFCLASS64;
using rocjitsu::fuzzer::afl::kCoverageSlots;
using rocjitsu::fuzzer::afl::kDeviceStart;

constexpr hipError_t kHipErrorRuntimeUnavailable = static_cast<hipError_t>(999);
constexpr size_t kDeviceCounterBytes = sizeof(uint32_t) * kCoverageSlots;

using hipModuleLoadData_t = hipError_t (*)(hipModule_t *, const void *);
using hipModuleUnload_t = hipError_t (*)(hipModule_t);
using hipModuleGetFunction_t = hipError_t (*)(hipFunction_t *, hipModule_t, const char *);
using hipModuleLaunchKernel_t = hipError_t (*)(hipFunction_t, unsigned int, unsigned int,
                                               unsigned int, unsigned int, unsigned int,
                                               unsigned int, unsigned int, hipStream_t, void **,
                                               void **);
using hipDeviceSynchronize_t = hipError_t (*)();
using hipMemcpy_t = hipError_t (*)(void *, const void *, size_t, hipMemcpyKind);
using hipMalloc_t = hipError_t (*)(void **, size_t);
using hipMemset_t = hipError_t (*)(void *, int, size_t);
using hipFree_t = hipError_t (*)(void *);
using hipGetErrorString_t = const char *(*)(hipError_t);
using hsa_code_object_reader_create_from_memory_t = hsa_status_t (*)(const void *, size_t,
                                                                     hsa_code_object_reader_t *);
using hsa_code_object_reader_create_from_file_t = hsa_status_t (*)(hsa_file_t,
                                                                   hsa_code_object_reader_t *);
using hsa_code_object_reader_destroy_t = hsa_status_t (*)(hsa_code_object_reader_t);
using hsa_ven_amd_loader_code_object_reader_create_from_file_with_offset_size_t =
    hsa_status_t (*)(hsa_file_t, size_t, size_t, hsa_code_object_reader_t *);

hipModuleLoadData_t real_hipModuleLoadData = nullptr;
hipModuleUnload_t real_hipModuleUnload = nullptr;
hipModuleGetFunction_t real_hipModuleGetFunction = nullptr;
hipModuleLaunchKernel_t real_hipModuleLaunchKernel = nullptr;
hipDeviceSynchronize_t real_hipDeviceSynchronize = nullptr;
hipMemcpy_t real_hipMemcpy = nullptr;
hipMalloc_t real_hipMalloc = nullptr;
hipMemset_t real_hipMemset = nullptr;
hipFree_t real_hipFree = nullptr;
hipGetErrorString_t real_hipGetErrorString = nullptr;
hsa_code_object_reader_create_from_memory_t real_hsa_code_object_reader_create_from_memory =
    nullptr;
hsa_code_object_reader_create_from_file_t real_hsa_code_object_reader_create_from_file = nullptr;
hsa_code_object_reader_destroy_t real_hsa_code_object_reader_destroy = nullptr;
hsa_ven_amd_loader_code_object_reader_create_from_file_with_offset_size_t
    real_hsa_ven_amd_loader_code_object_reader_create_from_file_with_offset_size = nullptr;

std::once_flag g_symbol_once;
void *g_explicit_hip_runtime = nullptr;
void *g_explicit_hsa_runtime = nullptr;
thread_local uint32_t g_intercept_depth = 0;

std::mutex g_runtime_mutex;
uint8_t *g_afl_area = nullptr;
int g_afl_shm_id = -1;
void *g_device_counters = nullptr;
std::vector<uint32_t> g_host_counters;
rocjitsu::HsaCodeObjectReaderRewriter g_hsa_reader_rewriter;
std::unordered_map<hipModule_t, std::vector<uint8_t>> g_rewritten_hip_modules;

bool env_flag(const char *name) {
  const char *value = getenv(name);
  return value != nullptr && value[0] != '\0' && value[0] != '0';
}

template <typename T> T load_symbol(const char *name) {
  return reinterpret_cast<T>(dlsym(RTLD_NEXT, name));
}

bool verbose_enabled();

template <typename T> T load_hip_symbol(const char *name) {
  if (g_explicit_hip_runtime != nullptr)
    return reinterpret_cast<T>(dlsym(g_explicit_hip_runtime, name));
  return load_symbol<T>(name);
}

template <typename T> T load_hsa_symbol(const char *name) {
  if (g_explicit_hsa_runtime != nullptr)
    return reinterpret_cast<T>(dlsym(g_explicit_hsa_runtime, name));
  return load_symbol<T>(name);
}

void resolve_symbols() {
  std::call_once(g_symbol_once, [] {
    const char *runtime_path = getenv("ROCJITSU_AFL_HIP_RUNTIME_PATH");
    if (runtime_path != nullptr && runtime_path[0] != '\0') {
      g_explicit_hip_runtime = dlopen(runtime_path, RTLD_LAZY | RTLD_LOCAL);
      if (g_explicit_hip_runtime == nullptr && verbose_enabled())
        fprintf(stderr, "rocjitsu-afl: failed to load HIP runtime '%s': %s\n", runtime_path,
                dlerror());
    }

    const char *hsa_runtime_path = getenv("ROCJITSU_AFL_HSA_RUNTIME_PATH");
    if (hsa_runtime_path != nullptr && hsa_runtime_path[0] != '\0') {
      g_explicit_hsa_runtime = dlopen(hsa_runtime_path, RTLD_LAZY | RTLD_LOCAL);
      if (g_explicit_hsa_runtime == nullptr && verbose_enabled())
        fprintf(stderr, "rocjitsu-afl: failed to load HSA runtime '%s': %s\n", hsa_runtime_path,
                dlerror());
    }

    real_hipModuleLoadData = load_hip_symbol<hipModuleLoadData_t>("hipModuleLoadData");
    real_hipModuleUnload = load_hip_symbol<hipModuleUnload_t>("hipModuleUnload");
    real_hipModuleGetFunction = load_hip_symbol<hipModuleGetFunction_t>("hipModuleGetFunction");
    real_hipModuleLaunchKernel = load_hip_symbol<hipModuleLaunchKernel_t>("hipModuleLaunchKernel");
    real_hipDeviceSynchronize = load_hip_symbol<hipDeviceSynchronize_t>("hipDeviceSynchronize");
    real_hipMemcpy = load_hip_symbol<hipMemcpy_t>("hipMemcpy");
    real_hipMalloc = load_hip_symbol<hipMalloc_t>("hipMalloc");
    real_hipMemset = load_hip_symbol<hipMemset_t>("hipMemset");
    real_hipFree = load_hip_symbol<hipFree_t>("hipFree");
    real_hipGetErrorString = load_hip_symbol<hipGetErrorString_t>("hipGetErrorString");
    real_hsa_code_object_reader_create_from_memory =
        load_hsa_symbol<hsa_code_object_reader_create_from_memory_t>(
            "hsa_code_object_reader_create_from_memory");
    real_hsa_code_object_reader_create_from_file =
        load_hsa_symbol<hsa_code_object_reader_create_from_file_t>(
            "hsa_code_object_reader_create_from_file");
    real_hsa_code_object_reader_destroy =
        load_hsa_symbol<hsa_code_object_reader_destroy_t>("hsa_code_object_reader_destroy");
    real_hsa_ven_amd_loader_code_object_reader_create_from_file_with_offset_size =
        load_hsa_symbol<hsa_ven_amd_loader_code_object_reader_create_from_file_with_offset_size_t>(
            "hsa_ven_amd_loader_code_object_reader_create_from_file_with_offset_size");
    g_hsa_reader_rewriter.set_api(real_hsa_code_object_reader_create_from_memory,
                                  real_hsa_code_object_reader_destroy);
  });
}

bool interception_bypassed() { return g_intercept_depth != 0; }

struct ScopedInterceptionBypass {
  ScopedInterceptionBypass() { ++g_intercept_depth; }
  ScopedInterceptionBypass(const ScopedInterceptionBypass &) = delete;
  ScopedInterceptionBypass &operator=(const ScopedInterceptionBypass &) = delete;
  ~ScopedInterceptionBypass() { --g_intercept_depth; }
};

bool verbose_enabled() { return env_flag("ROCJITSU_AFL_VERBOSE"); }

const char *hip_error_string(hipError_t status) {
  if (real_hipGetErrorString != nullptr)
    return real_hipGetErrorString(status);
  return "HIP runtime unavailable";
}

bool parse_shm_id(int *id) {
  const char *value = getenv("__AFL_SHM_ID");
  if (value == nullptr || value[0] == '\0')
    return false;

  errno = 0;
  char *end = nullptr;
  const long parsed = strtol(value, &end, 10);
  if (errno != 0 || end == value || *end != '\0' || parsed < 0)
    return false;

  *id = static_cast<int>(parsed);
  return true;
}

bool ensure_afl_map_locked() {
  int shm_id = -1;
  if (!parse_shm_id(&shm_id))
    return false;

  if (g_afl_area != nullptr && g_afl_shm_id == shm_id)
    return true;

  if (g_afl_area != nullptr)
    shmdt(g_afl_area);

  g_afl_area = static_cast<uint8_t *>(shmat(shm_id, nullptr, 0));
  if (g_afl_area == reinterpret_cast<uint8_t *>(-1)) {
    g_afl_area = nullptr;
    g_afl_shm_id = -1;
    if (verbose_enabled())
      fprintf(stderr, "rocjitsu-afl: failed to attach AFL shared memory id %d\n", shm_id);
    return false;
  }

  g_afl_shm_id = shm_id;
  return true;
}

hipError_t ensure_device_counters_locked() {
  if (g_device_counters != nullptr)
    return hipSuccess;
  if (real_hipMalloc == nullptr || real_hipMemset == nullptr)
    return kHipErrorRuntimeUnavailable;

  hipError_t status = real_hipMalloc(&g_device_counters, kDeviceCounterBytes);
  if (status != hipSuccess)
    return status;

  status = real_hipMemset(g_device_counters, 0, kDeviceCounterBytes);
  if (status != hipSuccess) {
    if (real_hipFree != nullptr)
      static_cast<void>(real_hipFree(g_device_counters));
    g_device_counters = nullptr;
    return status;
  }

  g_host_counters.assign(kCoverageSlots, 0);
  return hipSuccess;
}

uint8_t saturating_add(uint8_t value, uint32_t delta) {
  const uint32_t sum = static_cast<uint32_t>(value) + delta;
  return static_cast<uint8_t>(std::min<uint32_t>(sum, 255));
}

bool checked_add(uint64_t lhs, uint64_t rhs, uint64_t *out) {
  if (out == nullptr || lhs > std::numeric_limits<uint64_t>::max() - rhs)
    return false;
  *out = lhs + rhs;
  return true;
}

bool checked_multiply(uint64_t lhs, uint64_t rhs, uint64_t *out) {
  if (out == nullptr || (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs))
    return false;
  *out = lhs * rhs;
  return true;
}

std::optional<size_t> infer_raw_elf_image_size(const void *image) {
  if (image == nullptr)
    return std::nullopt;

  Elf64_Ehdr ehdr{};
  std::memcpy(&ehdr, image, sizeof(ehdr));
  if (std::memcmp(ehdr.e_ident, EI_MAGIC, EI_MAGIC_SIZE) != 0 ||
      ehdr.e_ident[EI_CLASS] != ELFCLASS64 || ehdr.e_shentsize != sizeof(Elf64_Shdr) ||
      ehdr.e_shnum == 0)
    return std::nullopt;

  uint64_t section_headers_bytes = 0;
  uint64_t section_headers_end = 0;
  if (!checked_multiply(ehdr.e_shnum, sizeof(Elf64_Shdr), &section_headers_bytes) ||
      !checked_add(ehdr.e_shoff, section_headers_bytes, &section_headers_end))
    return std::nullopt;

  uint64_t program_headers_end = 0;
  if (ehdr.e_phnum != 0) {
    if (ehdr.e_phentsize != sizeof(Elf64_Phdr))
      return std::nullopt;

    uint64_t program_headers_bytes = 0;
    if (!checked_multiply(ehdr.e_phnum, sizeof(Elf64_Phdr), &program_headers_bytes) ||
        !checked_add(ehdr.e_phoff, program_headers_bytes, &program_headers_end))
      return std::nullopt;
  }

  const uint64_t inferred_size = std::max(section_headers_end, program_headers_end);
  constexpr uint64_t kMaxModuleImageBytes = 256ull * 1024ull * 1024ull;
  if (inferred_size == 0 || inferred_size > kMaxModuleImageBytes ||
      inferred_size > std::numeric_limits<size_t>::max())
    return std::nullopt;

  return static_cast<size_t>(inferred_size);
}

std::optional<uint64_t> ensure_device_counter_pointer() {
  std::lock_guard<std::mutex> lock(g_runtime_mutex);
  const hipError_t status = ensure_device_counters_locked();
  if (status != hipSuccess) {
    if (verbose_enabled())
      fprintf(stderr, "rocjitsu-afl: HSA reader rewrite disabled: %s\n", hip_error_string(status));
    return std::nullopt;
  }
  return reinterpret_cast<uint64_t>(g_device_counters);
}

std::optional<std::vector<uint8_t>> try_rewrite_hsa_image(std::span<const uint8_t> image, void *) {
  if (!rocjitsu::is_supported_amdgpu_elf(image))
    return std::nullopt;

  const auto state_pointer = ensure_device_counter_pointer();
  if (!state_pointer.has_value())
    return std::nullopt;

  std::vector<uint8_t> rewritten = rocjitsu::patch_amdgpu_elf_kernel_entries(image, *state_pointer);
  if (rewritten.size() == image.size() &&
      std::equal(rewritten.begin(), rewritten.end(), image.begin(), image.end()))
    return std::nullopt;
  return rewritten;
}

std::optional<std::vector<uint8_t>> try_rewrite_hip_module_image(const void *image) {
  const auto image_size = infer_raw_elf_image_size(image);
  if (!image_size.has_value())
    return std::nullopt;
  return try_rewrite_hsa_image(
      std::span<const uint8_t>(static_cast<const uint8_t *>(image), *image_size), nullptr);
}

} // namespace

extern "C" {

hipError_t hipModuleLoadData(hipModule_t *module, const void *image) {
  resolve_symbols();
  if (real_hipModuleLoadData == nullptr)
    return kHipErrorRuntimeUnavailable;
  if (interception_bypassed())
    return real_hipModuleLoadData(module, image);
  if (module == nullptr)
    return real_hipModuleLoadData(module, image);
  ScopedInterceptionBypass bypass;
  try {
    auto rewritten = try_rewrite_hip_module_image(image);
    if (!rewritten.has_value())
      return real_hipModuleLoadData(module, image);

    hipModule_t loaded_module = nullptr;
    hipError_t status = real_hipModuleLoadData(&loaded_module, rewritten->data());
    if (status != hipSuccess)
      return status;

    if (module != nullptr)
      *module = loaded_module;

    std::lock_guard<std::mutex> lock(g_runtime_mutex);
    g_rewritten_hip_modules[loaded_module] = std::move(*rewritten);
    return hipSuccess;
  } catch (const std::bad_alloc &) {
    return kHipErrorRuntimeUnavailable;
  }
}

hipError_t hipModuleUnload(hipModule_t module) {
  resolve_symbols();
  if (real_hipModuleUnload == nullptr)
    return kHipErrorRuntimeUnavailable;
  if (interception_bypassed())
    return real_hipModuleUnload(module);
  ScopedInterceptionBypass bypass;
  const hipError_t status = real_hipModuleUnload(module);
  if (status == hipSuccess) {
    std::lock_guard<std::mutex> lock(g_runtime_mutex);
    g_rewritten_hip_modules.erase(module);
  }
  return status;
}

hipError_t hipModuleGetFunction(hipFunction_t *function, hipModule_t module, const char *kname) {
  resolve_symbols();
  if (real_hipModuleGetFunction == nullptr)
    return kHipErrorRuntimeUnavailable;
  if (interception_bypassed())
    return real_hipModuleGetFunction(function, module, kname);
  ScopedInterceptionBypass bypass;
  return real_hipModuleGetFunction(function, module, kname);
}

hipError_t hipModuleLaunchKernel(hipFunction_t f, unsigned int grid_dim_x, unsigned int grid_dim_y,
                                 unsigned int grid_dim_z, unsigned int block_dim_x,
                                 unsigned int block_dim_y, unsigned int block_dim_z,
                                 unsigned int shared_mem_bytes, hipStream_t stream,
                                 void **kernel_params, void **extra) {
  resolve_symbols();
  if (real_hipModuleLaunchKernel == nullptr)
    return kHipErrorRuntimeUnavailable;
  if (interception_bypassed()) {
    return real_hipModuleLaunchKernel(f, grid_dim_x, grid_dim_y, grid_dim_z, block_dim_x,
                                      block_dim_y, block_dim_z, shared_mem_bytes, stream,
                                      kernel_params, extra);
  }
  ScopedInterceptionBypass bypass;
  return real_hipModuleLaunchKernel(f, grid_dim_x, grid_dim_y, grid_dim_z, block_dim_x, block_dim_y,
                                    block_dim_z, shared_mem_bytes, stream, kernel_params, extra);
}

hipError_t hipDeviceSynchronize() {
  resolve_symbols();
  if (real_hipDeviceSynchronize == nullptr)
    return kHipErrorRuntimeUnavailable;
  if (interception_bypassed())
    return real_hipDeviceSynchronize();
  ScopedInterceptionBypass bypass;
  return real_hipDeviceSynchronize();
}

hipError_t hipMemcpy(void *dst, const void *src, size_t size_bytes, hipMemcpyKind kind) {
  resolve_symbols();
  if (real_hipMemcpy == nullptr)
    return kHipErrorRuntimeUnavailable;
  if (interception_bypassed())
    return real_hipMemcpy(dst, src, size_bytes, kind);
  ScopedInterceptionBypass bypass;
  return real_hipMemcpy(dst, src, size_bytes, kind);
}

hsa_status_t
hsa_code_object_reader_create_from_memory(const void *code_object, size_t size,
                                          hsa_code_object_reader_t *code_object_reader) {
  resolve_symbols();
  if (real_hsa_code_object_reader_create_from_memory == nullptr)
    return HSA_STATUS_ERROR_NOT_INITIALIZED;
  if (interception_bypassed() || code_object == nullptr || size == 0 ||
      code_object_reader == nullptr)
    return real_hsa_code_object_reader_create_from_memory(code_object, size, code_object_reader);

  ScopedInterceptionBypass bypass;
  return g_hsa_reader_rewriter.create_from_memory(code_object, size, code_object_reader,
                                                  try_rewrite_hsa_image, nullptr);
}

hsa_status_t hsa_code_object_reader_create_from_file(hsa_file_t file,
                                                     hsa_code_object_reader_t *code_object_reader) {
  resolve_symbols();
  if (real_hsa_code_object_reader_create_from_file == nullptr)
    return HSA_STATUS_ERROR_NOT_INITIALIZED;
  if (interception_bypassed() || code_object_reader == nullptr)
    return real_hsa_code_object_reader_create_from_file(file, code_object_reader);

  ScopedInterceptionBypass bypass;
  return g_hsa_reader_rewriter.create_from_file(file, code_object_reader,
                                                real_hsa_code_object_reader_create_from_file,
                                                try_rewrite_hsa_image, nullptr);
}

hsa_status_t hsa_ven_amd_loader_code_object_reader_create_from_file_with_offset_size(
    hsa_file_t file, size_t offset, size_t size, hsa_code_object_reader_t *code_object_reader) {
  resolve_symbols();
  if (real_hsa_ven_amd_loader_code_object_reader_create_from_file_with_offset_size == nullptr)
    return HSA_STATUS_ERROR_NOT_INITIALIZED;
  if (interception_bypassed() || code_object_reader == nullptr) {
    return real_hsa_ven_amd_loader_code_object_reader_create_from_file_with_offset_size(
        file, offset, size, code_object_reader);
  }

  ScopedInterceptionBypass bypass;
  return g_hsa_reader_rewriter.create_from_file_with_offset_size(
      file, offset, size, code_object_reader,
      real_hsa_ven_amd_loader_code_object_reader_create_from_file_with_offset_size,
      try_rewrite_hsa_image, nullptr);
}

hsa_status_t hsa_code_object_reader_destroy(hsa_code_object_reader_t code_object_reader) {
  resolve_symbols();
  if (real_hsa_code_object_reader_destroy == nullptr)
    return HSA_STATUS_ERROR_NOT_INITIALIZED;
  if (interception_bypassed())
    return g_hsa_reader_rewriter.destroy(code_object_reader);
  ScopedInterceptionBypass bypass;
  return g_hsa_reader_rewriter.destroy(code_object_reader);
}

int rocjitsu_afl_persistent_begin() {
  resolve_symbols();
  std::lock_guard<std::mutex> lock(g_runtime_mutex);

  if (!ensure_afl_map_locked())
    return 0;

  const hipError_t status = ensure_device_counters_locked();
  if (status != hipSuccess) {
    fprintf(stderr, "rocjitsu-afl: failed to allocate device counters: %s\n",
            hip_error_string(status));
    return static_cast<int>(status);
  }

  if (real_hipMemset == nullptr)
    return static_cast<int>(kHipErrorRuntimeUnavailable);

  const hipError_t memset_status = real_hipMemset(g_device_counters, 0, kDeviceCounterBytes);
  if (memset_status != hipSuccess) {
    fprintf(stderr, "rocjitsu-afl: failed to reset device counters: %s\n",
            hip_error_string(memset_status));
    return static_cast<int>(memset_status);
  }

  if (verbose_enabled())
    fprintf(stderr, "rocjitsu-afl: persistent iteration begin\n");
  return 0;
}

int rocjitsu_afl_persistent_end() {
  resolve_symbols();
  if (real_hipDeviceSynchronize == nullptr)
    return static_cast<int>(kHipErrorRuntimeUnavailable);

  ScopedInterceptionBypass bypass;
  const hipError_t status = real_hipDeviceSynchronize();
  if (status != hipSuccess) {
    fprintf(stderr, "rocjitsu-afl: hipDeviceSynchronize failed: %s\n", hip_error_string(status));
    return static_cast<int>(status);
  }

  std::lock_guard<std::mutex> lock(g_runtime_mutex);
  if (ensure_afl_map_locked() && g_device_counters != nullptr) {
    if (real_hipMemcpy == nullptr)
      return static_cast<int>(kHipErrorRuntimeUnavailable);

    if (g_host_counters.size() != kCoverageSlots)
      g_host_counters.assign(kCoverageSlots, 0);

    const hipError_t copy_status = real_hipMemcpy(g_host_counters.data(), g_device_counters,
                                                  kDeviceCounterBytes, hipMemcpyDeviceToHost);
    if (copy_status != hipSuccess) {
      fprintf(stderr, "rocjitsu-afl: failed to copy device counters: %s\n",
              hip_error_string(copy_status));
      return static_cast<int>(copy_status);
    }

    for (uint32_t slot = 0; slot < kCoverageSlots; ++slot) {
      const uint32_t delta = g_host_counters[slot];
      if (delta != 0)
        g_afl_area[kDeviceStart + slot] = saturating_add(g_afl_area[kDeviceStart + slot], delta);
    }
  }

  if (verbose_enabled())
    fprintf(stderr, "rocjitsu-afl: persistent iteration end\n");
  return 0;
}

} // extern "C"
