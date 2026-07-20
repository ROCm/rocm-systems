// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file rj_comgr_hotswap.cpp
/// @brief Minimal COMGR HotSwap ABI backed by rocjitsu DBT.
///
/// ROCr's HotSwap path resolves six COMGR entry points from one explicitly
/// opened shared object.  This library implements only that narrow contract;
/// it is not a replacement for general-purpose COMGR compilation APIs.

#include "rocjitsu/base/rj_compiler.h"
#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/code/dbt/binary_translator.h"
#include "rocjitsu/code/dbt/processor_revision.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <string>
#include <string_view>
#include <sys/file.h>
#include <sys/stat.h>
#include <unordered_set>
#include <utility>
#include <vector>

#include <unistd.h>

namespace {

// Keep these ABI mirrors in sync with amd_comgr.h. ROCr mirrors the same
// declarations locally so neither side needs a build-time COMGR dependency.
enum ComgrStatus : int {
  kComgrStatusSuccess = 0,
  kComgrStatusError = 1,
  kComgrStatusInvalidArgument = 2,
  kComgrStatusOutOfResources = 3,
};

enum ComgrDataKind : int {
  kComgrDataKindExecutable = 0x8,
};

struct ComgrData {
  uint64_t handle;
};

struct DataObject {
  int kind = 0;
  std::vector<uint8_t> bytes;
};

std::mutex g_data_mutex;
std::unordered_set<DataObject *> g_data_objects;
std::atomic<uint64_t> g_dump_sequence{0};
std::atomic<uint64_t> g_cache_temp_sequence{0};

// Bump this whenever the B0 -> A0 translation policy or serialized output
// changes incompatibly. The input and ISA names are also part of the key.
constexpr std::string_view kCacheSchema =
    "rocjitsu-comgr-gfx1250-b0-a0-v3-tensor-mask";

[[nodiscard]] bool env_flag_enabled(const char *name) {
  const char *value = std::getenv(name);
  if (value == nullptr || value[0] == '\0')
    return false;
  std::string normalized(value);
  std::transform(normalized.begin(), normalized.end(), normalized.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return normalized != "0" && normalized != "off" && normalized != "false" &&
         normalized != "no";
}

template <typename... Args> void verbose_log(const char *format, Args... args) {
  if (!env_flag_enabled("HSA_HOTSWAP_VERBOSE"))
    return;
  std::fputs("[rocjitsu-comgr] ", stderr);
  if constexpr (sizeof...(Args) == 0)
    std::fputs(format, stderr);
  else
    std::fprintf(stderr, format, args...);
}

void dump_failed_input(const DataObject &object) {
  const char *directory = std::getenv("ROCJITSU_HOTSWAP_DUMP_DIR");
  if (directory == nullptr || directory[0] == '\0')
    return;

  const uint64_t sequence = g_dump_sequence.fetch_add(1, std::memory_order_relaxed);
  char path[4096];
  const int length = std::snprintf(path, sizeof(path), "%s/rocjitsu-hotswap-%ld-%llu.hsaco",
                                   directory, static_cast<long>(getpid()),
                                   static_cast<unsigned long long>(sequence));
  if (length < 0 || static_cast<size_t>(length) >= sizeof(path)) {
    verbose_log("failed-input dump path is too long\n");
    return;
  }

  FILE *file = std::fopen(path, "wb");
  if (file == nullptr) {
    verbose_log("could not open failed-input dump %s\n", path);
    return;
  }
  const size_t written = std::fwrite(object.bytes.data(), 1, object.bytes.size(), file);
  const int close_status = std::fclose(file);
  if (written != object.bytes.size() || close_status != 0) {
    verbose_log("failed to write complete failed-input dump %s\n", path);
    return;
  }
  verbose_log("dumped failed input to %s\n", path);
}

[[nodiscard]] DataObject *lookup_data(ComgrData data) {
  auto *object = reinterpret_cast<DataObject *>(static_cast<uintptr_t>(data.handle));
  if (object == nullptr)
    return nullptr;
  std::lock_guard lock(g_data_mutex);
  return g_data_objects.contains(object) ? object : nullptr;
}

[[nodiscard]] ComgrStatus allocate_data(int kind, ComgrData *data) {
  if (data == nullptr || kind != kComgrDataKindExecutable)
    return kComgrStatusInvalidArgument;

  auto object = std::unique_ptr<DataObject>(new (std::nothrow) DataObject());
  if (!object)
    return kComgrStatusOutOfResources;
  object->kind = kind;
  {
    std::lock_guard lock(g_data_mutex);
    try {
      g_data_objects.insert(object.get());
    } catch (const std::bad_alloc &) {
      return kComgrStatusOutOfResources;
    }
  }
  data->handle = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(object.release()));
  return kComgrStatusSuccess;
}

class CacheLock {
public:
  CacheLock() = default;
  CacheLock(const CacheLock &) = delete;
  CacheLock &operator=(const CacheLock &) = delete;
  CacheLock(CacheLock &&other) noexcept : fd_(std::exchange(other.fd_, -1)) {}
  CacheLock &operator=(CacheLock &&other) noexcept {
    if (this != &other) {
      close();
      fd_ = std::exchange(other.fd_, -1);
    }
    return *this;
  }
  ~CacheLock() { close(); }

  [[nodiscard]] static std::optional<CacheLock> acquire(const std::string &path) {
    const int fd = ::open(path.c_str(), O_CREAT | O_RDWR | O_CLOEXEC, 0600);
    if (fd < 0)
      return std::nullopt;
    if (::flock(fd, LOCK_EX) != 0) {
      ::close(fd);
      return std::nullopt;
    }
    CacheLock lock;
    lock.fd_ = fd;
    return lock;
  }

private:
  void close() {
    if (fd_ < 0)
      return;
    (void)::flock(fd_, LOCK_UN);
    (void)::close(fd_);
    fd_ = -1;
  }

  int fd_ = -1;
};

struct CachePaths {
  std::string object;
  std::string lock;
};

// A stable 128-bit non-cryptographic content fingerprint. Cache entries are
// validated as gfx1250 code objects before use; the cache is not a trust
// boundary, and 128 bits makes accidental collisions negligible.
class CacheFingerprint {
public:
  void add(const void *data, size_t size) {
    const auto *bytes = static_cast<const uint8_t *>(data);
    for (size_t i = 0; i < size; ++i) {
      first_ ^= bytes[i];
      first_ *= 1099511628211ULL;
      second_ ^= static_cast<uint64_t>(bytes[i]) + 0x9e3779b97f4a7c15ULL;
      second_ *= 14029467366897019727ULL;
      second_ ^= second_ >> 29;
    }
  }

  void add(std::string_view value) {
    const uint64_t size = value.size();
    add(&size, sizeof(size));
    add(value.data(), value.size());
  }

  [[nodiscard]] std::string hex() const {
    const uint64_t first = avalanche(first_);
    const uint64_t second = avalanche(second_ ^ first);
    char text[33];
    std::snprintf(text, sizeof(text), "%016llx%016llx",
                  static_cast<unsigned long long>(first),
                  static_cast<unsigned long long>(second));
    return text;
  }

private:
  [[nodiscard]] static uint64_t avalanche(uint64_t value) {
    value ^= value >> 30;
    value *= 0xbf58476d1ce4e5b9ULL;
    value ^= value >> 27;
    value *= 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
  }

  uint64_t first_ = 1469598103934665603ULL;
  uint64_t second_ = 0x6a09e667f3bcc909ULL;
};

[[nodiscard]] std::optional<CachePaths>
cache_paths(const DataObject &input, std::string_view source_isa,
            std::string_view target_isa) {
  const char *directory = std::getenv("HSA_HOTSWAP_CACHE_DIR");
  if (directory == nullptr || directory[0] == '\0')
    return std::nullopt;

  struct stat directory_stat {};
  if (::stat(directory, &directory_stat) != 0 || !S_ISDIR(directory_stat.st_mode)) {
    verbose_log("cache directory is unavailable: %s\n", directory);
    return std::nullopt;
  }

  CacheFingerprint fingerprint;
  fingerprint.add(kCacheSchema);
  fingerprint.add(source_isa);
  fingerprint.add(target_isa);
  const uint64_t input_size = input.bytes.size();
  fingerprint.add(&input_size, sizeof(input_size));
  fingerprint.add(input.bytes.data(), input.bytes.size());

  std::string prefix(directory);
  if (!prefix.empty() && prefix.back() != '/')
    prefix.push_back('/');
  prefix += "rocjitsu-hotswap-";
  prefix += fingerprint.hex();
  return CachePaths{prefix + ".hsaco", prefix + ".lock"};
}

[[nodiscard]] bool read_all(int fd, uint8_t *bytes, size_t size) {
  size_t offset = 0;
  while (offset != size) {
    const ssize_t count = ::read(fd, bytes + offset, size - offset);
    if (count > 0) {
      offset += static_cast<size_t>(count);
      continue;
    }
    if (count < 0 && errno == EINTR)
      continue;
    return false;
  }
  return true;
}

[[nodiscard]] bool write_all(int fd, const uint8_t *bytes, size_t size) {
  size_t offset = 0;
  while (offset != size) {
    const ssize_t count = ::write(fd, bytes + offset, size - offset);
    if (count > 0) {
      offset += static_cast<size_t>(count);
      continue;
    }
    if (count < 0 && errno == EINTR)
      continue;
    return false;
  }
  return true;
}

[[nodiscard]] std::optional<std::vector<uint8_t>>
load_cached_translation(const std::string &path) {
  const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0)
    return std::nullopt;

  struct stat file_stat {};
  if (::fstat(fd, &file_stat) != 0 || !S_ISREG(file_stat.st_mode) ||
      file_stat.st_size <= 0 ||
      static_cast<uint64_t>(file_stat.st_size) >
          static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
    (void)::close(fd);
    return std::nullopt;
  }

  std::vector<uint8_t> bytes;
  try {
    bytes.resize(static_cast<size_t>(file_stat.st_size));
  } catch (const std::bad_alloc &) {
    (void)::close(fd);
    return std::nullopt;
  }
  const bool read_ok = read_all(fd, bytes.data(), bytes.size());
  (void)::close(fd);
  if (!read_ok)
    return std::nullopt;

  rocjitsu::AmdGpuCodeObject object(bytes.data(), bytes.size());
  if (!object.is_valid() || object.target_id() != ROCJITSU_CODE_TARGET_GFX1250)
    return std::nullopt;
  return bytes;
}

void store_cached_translation(const std::string &path,
                              const std::vector<uint8_t> &bytes) {
  const uint64_t sequence =
      g_cache_temp_sequence.fetch_add(1, std::memory_order_relaxed);
  const std::string temporary =
      path + ".tmp-" + std::to_string(static_cast<long>(getpid())) + "-" +
      std::to_string(sequence);
  const int fd = ::open(temporary.c_str(), O_CREAT | O_EXCL | O_WRONLY | O_CLOEXEC, 0600);
  if (fd < 0) {
    verbose_log("could not create cache temporary %s\n", temporary.c_str());
    return;
  }

  const bool write_ok = write_all(fd, bytes.data(), bytes.size());
  const bool sync_ok = write_ok && ::fsync(fd) == 0;
  const bool close_ok = ::close(fd) == 0;
  if (!write_ok || !sync_ok || !close_ok || ::rename(temporary.c_str(), path.c_str()) != 0) {
    (void)::unlink(temporary.c_str());
    verbose_log("could not commit cache entry %s\n", path.c_str());
    return;
  }
  verbose_log("stored %zu translated bytes in cache %s\n", bytes.size(), path.c_str());
}

[[nodiscard]] bool is_gfx1250_b0_to_a0(std::string_view source, std::string_view target) {
  constexpr std::string_view kProcessor = "--gfx1250";
  constexpr std::string_view kB0 = ":gfx1250-b0-specific+";
  constexpr std::string_view kA0 = ":gfx1250-b0-specific-";
  return source.find(kProcessor) != std::string_view::npos &&
         target.find(kProcessor) != std::string_view::npos && source.find(kB0) != source.npos &&
         target.find(kA0) != target.npos;
}

[[nodiscard]] bool is_gfx1250_already_a0(std::string_view source, std::string_view target) {
  constexpr std::string_view kProcessor = "--gfx1250";
  constexpr std::string_view kA0 = ":gfx1250-b0-specific-";
  return source.find(kProcessor) != std::string_view::npos &&
         target.find(kProcessor) != std::string_view::npos && source.find(kA0) != source.npos &&
         target.find(kA0) != target.npos;
}

[[nodiscard]] std::string isa_name_from_elf(const DataObject &object) {
  using namespace rocjitsu;
  if (object.bytes.size() < sizeof(Elf64_Ehdr))
    return {};
  Elf64_Ehdr header{};
  std::memcpy(&header, object.bytes.data(), sizeof(header));
  if (std::memcmp(header.e_ident, EI_MAGIC, EI_MAGIC_SIZE) != 0 ||
      header.e_ident[EI_CLASS] != ELFCLASS64 ||
      header.e_ident[EI_OSABI] != ELFOSABI_AMDGPU_HSA || header.e_machine != EM_AMDGPU)
    return {};
  const char *processor = elf_mach_name(header.e_flags);
  if (std::string_view(processor) == "unknown")
    return {};
  return std::string("amdgcn-amd-amdhsa--") + processor;
}

void print_diagnostics(const std::vector<rocjitsu::TranslationDiagnostic> &diagnostics) {
  if (!env_flag_enabled("HSA_HOTSWAP_VERBOSE"))
    return;
  for (const auto &diagnostic : diagnostics) {
    std::fprintf(stderr,
                 "[rocjitsu-comgr] diagnostic kind=%d severity=%d offset=0x%llx mnemonic=%s: %s\n",
                 static_cast<int>(diagnostic.kind), static_cast<int>(diagnostic.severity),
                 static_cast<unsigned long long>(diagnostic.guest_offset.value_or(0)),
                 diagnostic.mnemonic.empty() ? "<none>" : diagnostic.mnemonic.c_str(),
                 diagnostic.message.c_str());
  }
}

} // namespace

extern "C" RJ_API_EXPORT int amd_comgr_create_data(int kind, ComgrData *data) {
  return allocate_data(kind, data);
}

extern "C" RJ_API_EXPORT int amd_comgr_release_data(ComgrData data) {
  auto *object = reinterpret_cast<DataObject *>(static_cast<uintptr_t>(data.handle));
  if (object == nullptr)
    return kComgrStatusInvalidArgument;
  {
    std::lock_guard lock(g_data_mutex);
    auto it = g_data_objects.find(object);
    if (it == g_data_objects.end())
      return kComgrStatusInvalidArgument;
    g_data_objects.erase(it);
  }
  delete object;
  return kComgrStatusSuccess;
}

namespace {

[[nodiscard]] ComgrStatus return_output_bytes(std::vector<uint8_t> bytes,
                                              ComgrData *output) {
  ComgrData result{};
  ComgrStatus status = allocate_data(kComgrDataKindExecutable, &result);
  if (status != kComgrStatusSuccess)
    return status;
  DataObject *result_object = lookup_data(result);
  if (result_object == nullptr) {
    (void)amd_comgr_release_data(result);
    return kComgrStatusError;
  }
  result_object->bytes = std::move(bytes);
  *output = result;
  return kComgrStatusSuccess;
}

} // namespace

extern "C" RJ_API_EXPORT int amd_comgr_set_data(ComgrData data, size_t size, const char *bytes) {
  DataObject *object = lookup_data(data);
  if (object == nullptr || size == 0 || bytes == nullptr)
    return kComgrStatusInvalidArgument;
  try {
    object->bytes.assign(reinterpret_cast<const uint8_t *>(bytes),
                         reinterpret_cast<const uint8_t *>(bytes) + size);
  } catch (const std::bad_alloc &) {
    return kComgrStatusOutOfResources;
  }
  return kComgrStatusSuccess;
}

extern "C" RJ_API_EXPORT int amd_comgr_get_data(ComgrData data, size_t *size, char *bytes) {
  DataObject *object = lookup_data(data);
  if (object == nullptr || size == nullptr || object->bytes.empty())
    return kComgrStatusInvalidArgument;
  const size_t required = object->bytes.size();
  if (bytes != nullptr)
    std::memcpy(bytes, object->bytes.data(), std::min(*size, required));
  *size = required;
  return kComgrStatusSuccess;
}

extern "C" RJ_API_EXPORT int amd_comgr_get_data_isa_name(ComgrData data, size_t *size,
                                                          char *isa_name) {
  DataObject *object = lookup_data(data);
  if (object == nullptr || size == nullptr || object->kind != kComgrDataKindExecutable)
    return kComgrStatusInvalidArgument;
  const std::string isa = isa_name_from_elf(*object);
  if (isa.empty())
    return kComgrStatusError;
  const size_t required = isa.size() + 1;
  if (isa_name != nullptr && *size != 0)
    std::memcpy(isa_name, isa.c_str(), std::min(*size, required));
  *size = required;
  return kComgrStatusSuccess;
}

extern "C" RJ_API_EXPORT int amd_comgr_hotswap_rewrite(ComgrData input,
                                                        const char *source_isa_name,
                                                        const char *target_isa_name,
                                                        ComgrData *output) {
  DataObject *input_object = lookup_data(input);
  if (input_object == nullptr || input_object->kind != kComgrDataKindExecutable ||
      input_object->bytes.empty() || source_isa_name == nullptr || target_isa_name == nullptr ||
      output == nullptr)
    return kComgrStatusInvalidArgument;
  output->handle = 0;

  const bool needs_b0_to_a0 = is_gfx1250_b0_to_a0(source_isa_name, target_isa_name);
  const bool already_a0 = is_gfx1250_already_a0(source_isa_name, target_isa_name);
  if (!needs_b0_to_a0 && !already_a0) {
    verbose_log("rejecting unsupported rewrite %s -> %s\n", source_isa_name, target_isa_name);
    return kComgrStatusInvalidArgument;
  }

  try {
    rocjitsu::AmdGpuCodeObject source(input_object->bytes.data(), input_object->bytes.size());
    if (!source.is_valid() || source.target_id() != ROCJITSU_CODE_TARGET_GFX1250) {
      verbose_log("input is not a valid gfx1250 HSA code object\n");
      return kComgrStatusInvalidArgument;
    }

    if (already_a0) {
      verbose_log("input is already gfx1250 A0-compatible (%s); leaving %zu bytes unchanged\n",
                  source_isa_name, input_object->bytes.size());
      return return_output_bytes(input_object->bytes, output);
    }

    const std::optional<CachePaths> paths =
        cache_paths(*input_object, source_isa_name, target_isa_name);
    std::optional<CacheLock> cache_lock;
    if (paths) {
      if (auto cached = load_cached_translation(paths->object)) {
        verbose_log("cache hit: loaded %zu translated bytes from %s\n", cached->size(),
                    paths->object.c_str());
        return return_output_bytes(std::move(*cached), output);
      }

      verbose_log("cache miss: waiting for %s\n", paths->lock.c_str());
      cache_lock = CacheLock::acquire(paths->lock);
      if (!cache_lock) {
        verbose_log("could not lock cache entry %s; translating without cache\n",
                    paths->lock.c_str());
      } else if (auto cached = load_cached_translation(paths->object)) {
        verbose_log("cache hit after wait: loaded %zu translated bytes from %s\n",
                    cached->size(), paths->object.c_str());
        return return_output_bytes(std::move(*cached), output);
      } else if (::access(paths->object.c_str(), F_OK) == 0) {
        verbose_log("removing invalid cache entry %s\n", paths->object.c_str());
        (void)::unlink(paths->object.c_str());
      }
    }

    verbose_log("confirmed gfx1250 B0 input (%s); translating to A0 (%s)\n", source_isa_name,
                target_isa_name);

    rocjitsu::BinaryTranslatorOptions options;
    options.input_revision = rocjitsu::ProcessorRevision::Gfx1250B0;
    options.output_revision = rocjitsu::ProcessorRevision::Gfx1250A0;
    // Large framework code objects can include kernels that the current model
    // never dispatches. Keep the object loadable and redirect only an
    // unsupported kernel to the target-ISA trap stub; the translator reports a
    // KernelSkipped warning with its symbol and failure reason.
    options.skip_failed_kernels = true;
    rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250,
                                          ROCJITSU_CODE_ARCH_GFX1250,
                                          rocjitsu::EF_AMDGPU_MACH_AMDGCN_GFX1250, options);
    rocjitsu::TranslatedCodeObject translated = translator.translate(source);
    print_diagnostics(translated.diagnostics);
    if (!translated.ok() || translated.elf_bytes.empty()) {
      dump_failed_input(*input_object);
      verbose_log("B0 -> A0 translation failed\n");
      return kComgrStatusError;
    }

    verbose_log("translated %zu input bytes to %zu output bytes\n", input_object->bytes.size(),
                translated.elf_bytes.size());
    if (paths && cache_lock)
      store_cached_translation(paths->object, translated.elf_bytes);
    return return_output_bytes(std::move(translated.elf_bytes), output);
  } catch (const std::bad_alloc &) {
    return kComgrStatusOutOfResources;
  } catch (const std::exception &error) {
    dump_failed_input(*input_object);
    verbose_log("translation threw: %s\n", error.what());
    return kComgrStatusError;
  } catch (...) {
    dump_failed_input(*input_object);
    verbose_log("translation threw an unknown exception\n");
    return kComgrStatusError;
  }
}
