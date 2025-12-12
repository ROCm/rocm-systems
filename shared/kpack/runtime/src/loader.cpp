// Copyright (c) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <msgpack.hpp>
#include <string>
#include <vector>

#include "isa_target_match.h"
#include "kpack_internal.h"
#include "rocm_kpack/kpack.h"

namespace {

// Environment variable names
constexpr const char* ENV_KPACK_PATH = "ROCM_KPACK_PATH";
constexpr const char* ENV_KPACK_PATH_PREFIX = "ROCM_KPACK_PATH_PREFIX";
constexpr const char* ENV_KPACK_DISABLE = "ROCM_KPACK_DISABLE";
constexpr const char* ENV_KPACK_DEBUG = "ROCM_KPACK_DEBUG";

// Debug logging helper - uses cache's debug flag
#define KPACK_DEBUG(cache, ...)                    \
  do {                                             \
    if ((cache) && (cache)->debug) {               \
      std::fprintf(stderr, "kpack: " __VA_ARGS__); \
      std::fprintf(stderr, "\n");                  \
    }                                              \
  } while (0)

// Re-exported from isa_target_match.h for use in this TU.
using kpack::for_each_compatible_target;
using kpack::strip_target_prefix;

// Helper to find key in msgpack map
msgpack::object* find_key(const msgpack::object_map& map, const char* key) {
  for (size_t i = 0; i < map.size; ++i) {
    const auto& kv = map.ptr[i];
    if (kv.key.type == msgpack::type::STR) {
      if (std::strncmp(kv.key.via.str.ptr, key, kv.key.via.str.size) == 0 &&
          key[kv.key.via.str.size] == '\0') {
        return const_cast<msgpack::object*>(&kv.val);
      }
    }
  }
  return nullptr;
}

// Parse HIPK metadata msgpack
// Structure: {"kernel_name": "...", "kpack_search_paths": ["...", ...]}
kpack_error_t parse_hipk_metadata(const void* data, size_t max_size,
                                  std::string& kernel_name,
                                  std::vector<std::string>& search_paths) {
  // We don't know the exact size of the msgpack data, so we try to unpack
  // and let msgpack determine the boundaries
  msgpack::object_handle oh;
  try {
    oh = msgpack::unpack(static_cast<const char*>(data), max_size);
  } catch (...) {
    return KPACK_ERROR_INVALID_METADATA;
  }

  msgpack::object obj = oh.get();
  if (obj.type != msgpack::type::MAP) {
    return KPACK_ERROR_INVALID_METADATA;
  }

  const auto& map = obj.via.map;

  // Extract kernel_name (required)
  auto* val = find_key(map, "kernel_name");
  if (!val || val->type != msgpack::type::STR) {
    return KPACK_ERROR_INVALID_METADATA;
  }
  kernel_name = std::string(val->via.str.ptr, val->via.str.size);

  // Extract kpack_search_paths (required)
  val = find_key(map, "kpack_search_paths");
  if (!val || val->type != msgpack::type::ARRAY) {
    return KPACK_ERROR_INVALID_METADATA;
  }

  const auto& arr = val->via.array;
  search_paths.reserve(arr.size);
  for (size_t i = 0; i < arr.size; ++i) {
    if (arr.ptr[i].type == msgpack::type::STR) {
      search_paths.emplace_back(arr.ptr[i].via.str.ptr,
                                arr.ptr[i].via.str.size);
    }
  }

  if (search_paths.empty()) {
    return KPACK_ERROR_INVALID_METADATA;
  }

  return KPACK_SUCCESS;
}

// Split a path string by separator (colon on Linux, semicolon on Windows)
std::vector<std::string> split_path_list(const char* path_list) {
  std::vector<std::string> paths;
  if (path_list == nullptr || path_list[0] == '\0') {
    return paths;
  }

#ifdef _WIN32
  const char separator = ';';
#else
  const char separator = ':';
#endif

  std::string current;
  for (const char* p = path_list; *p != '\0'; ++p) {
    if (*p == separator) {
      if (!current.empty()) {
        paths.push_back(current);
        current.clear();
      }
    } else {
      current += *p;
    }
  }
  if (!current.empty()) {
    paths.push_back(current);
  }
  return paths;
}

// Resolve a relative path against a base directory
// All std::filesystem operations wrapped in try/catch to prevent exceptions
// crossing C API boundary
std::string resolve_path(const std::string& base_path,
                         const std::string& relative_path) {
  namespace fs = std::filesystem;

  try {
    // If relative_path is absolute, use it directly
    fs::path rel(relative_path);
    if (rel.is_absolute()) {
      return relative_path;
    }

    // Get directory of base_path
    fs::path base_dir = fs::path(base_path).parent_path();

    // Resolve relative path
    fs::path resolved = base_dir / rel;

    // Normalize (resolve .., ., etc.)
    return fs::weakly_canonical(resolved).string();
  } catch (...) {
    // If any filesystem operation fails, return relative_path as-is
    // (caller will fail to open, which is acceptable)
    return relative_path;
  }
}

// Check if a file exists (exception-safe)
bool file_exists(const std::string& path) {
  namespace fs = std::filesystem;
  try {
    std::error_code ec;
    return fs::exists(path, ec) && fs::is_regular_file(path, ec);
  } catch (...) {
    return false;
  }
}

// Placeholder token in search paths that gets replaced with the GPU
// architecture at runtime. The build tools embed paths like:
//   ../.kpack/foo_@GFXARCH@.kpack
// and this loader expands @GFXARCH@ for each requested architecture.
constexpr char kGfxArchPlaceholder[] = "@GFXARCH@";
constexpr size_t kGfxArchPlaceholderLen = sizeof(kGfxArchPlaceholder) - 1;

// Get canonical path for cache key (exception-safe)
std::string get_canonical_path(const std::string& path) {
  namespace fs = std::filesystem;
  try {
    return fs::canonical(path).string();
  } catch (...) {
    // If canonicalization fails, use the path as-is
    return path;
  }
}

}  // namespace

extern "C" {

kpack_error_t kpack_cache_create(kpack_cache_t* cache) {
  if (!cache) {
    return KPACK_ERROR_INVALID_ARGUMENT;
  }

  kpack_cache* c = new (std::nothrow) kpack_cache();
  if (!c) {
    return KPACK_ERROR_OUT_OF_MEMORY;
  }

  // Resolve all environment variables ONCE at creation time
  // This is thread-safe because we do it before returning the cache handle

  // Check if disabled
  const char* disable_env = std::getenv(ENV_KPACK_DISABLE);
  c->disabled = (disable_env != nullptr && disable_env[0] != '\0' &&
                 disable_env[0] != '0');

  // Check if debug enabled
  const char* debug_env = std::getenv(ENV_KPACK_DEBUG);
  c->debug =
      (debug_env != nullptr && debug_env[0] != '\0' && debug_env[0] != '0');

  // Parse path override
  const char* path_override = std::getenv(ENV_KPACK_PATH);
  if (path_override != nullptr && path_override[0] != '\0') {
    c->env_path_override = split_path_list(path_override);
  }

  // Parse path prefix
  const char* path_prefix = std::getenv(ENV_KPACK_PATH_PREFIX);
  if (path_prefix != nullptr && path_prefix[0] != '\0') {
    c->env_path_prefix = split_path_list(path_prefix);
  }

  KPACK_DEBUG(c,
              "cache created: disabled=%d, debug=%d, override_paths=%zu, "
              "prefix_paths=%zu",
              c->disabled, c->debug, c->env_path_override.size(),
              c->env_path_prefix.size());

  *cache = c;
  return KPACK_SUCCESS;
}

void kpack_cache_destroy(kpack_cache_t cache) {
  if (!cache) {
    return;
  }

  // Close all cached archives
  // No lock needed - caller must ensure no other threads are using the cache
  for (auto& pair : cache->archives) {
    kpack_close(pair.second);
  }

  delete cache;
}

kpack_error_t kpack_load_code_object(kpack_cache_t cache,
                                     const void* hipk_metadata,
                                     const char* binary_path, uint32_t co_index,
                                     const char* const* arch_list,
                                     size_t arch_count, void** code_object_out,
                                     size_t* code_object_size_out) {
  // Validate arguments
  if (!cache || !hipk_metadata || !binary_path || !arch_list ||
      arch_count == 0 || !code_object_out || !code_object_size_out) {
    return KPACK_ERROR_INVALID_ARGUMENT;
  }

  // Check if kpack is disabled (resolved at cache creation time)
  if (cache->disabled) {
    KPACK_DEBUG(cache, "kpack disabled via %s", ENV_KPACK_DISABLE);
    return KPACK_ERROR_NOT_IMPLEMENTED;
  }

  // Parse HIPK metadata
  std::string kernel_name;
  std::vector<std::string> embedded_search_paths;

  // Use a reasonable max size for metadata parsing
  // msgpack will stop at actual end of data
  constexpr size_t MAX_METADATA_SIZE = 64 * 1024;
  kpack_error_t err = parse_hipk_metadata(hipk_metadata, MAX_METADATA_SIZE,
                                          kernel_name, embedded_search_paths);
  if (err != KPACK_SUCCESS) {
    KPACK_DEBUG(cache, "failed to parse HIPK metadata");
    return err;
  }

  // Construct lookup key: kernel_name with index suffix
  // The TOC always uses indexed entries: "lib/foo.so#0", "lib/foo.so#1", etc.
  // Even single-TU binaries use #0 for consistency.
  std::string lookup_key = kernel_name + "#" + std::to_string(co_index);

  KPACK_DEBUG(cache,
              "parsed HIPK metadata: kernel_name='%s', co_index=%u, "
              "lookup_key='%s', %zu search paths",
              kernel_name.c_str(), co_index, lookup_key.c_str(),
              embedded_search_paths.size());

  // Build final list of .kpack archive paths.
  // Override/prefix paths are direct .kpack paths.
  // Embedded paths containing @GFXARCH@ are expanded for each requested
  // architecture. Other embedded paths are used as direct .kpack paths.
  std::vector<std::string> kpack_paths;

  if (!cache->env_path_override.empty()) {
    // Use override exclusively (direct .kpack paths)
    kpack_paths = cache->env_path_override;
    KPACK_DEBUG(cache, "using %s override: %zu paths", ENV_KPACK_PATH,
                kpack_paths.size());
  } else {
    // Prepend prefix paths (direct .kpack paths)
    if (!cache->env_path_prefix.empty()) {
      kpack_paths.insert(kpack_paths.end(), cache->env_path_prefix.begin(),
                         cache->env_path_prefix.end());
      KPACK_DEBUG(cache, "prepending %zu paths from %s",
                  cache->env_path_prefix.size(), ENV_KPACK_PATH_PREFIX);
    }

    // Resolve embedded search paths
    for (const auto& rel_path : embedded_search_paths) {
      std::string resolved = resolve_path(binary_path, rel_path);

      // Check for @GFXARCH@ pattern — expand for each requested architecture
      // and all compatible feature-flag subsets (most specific first).
      auto placeholder_pos = resolved.find(kGfxArchPlaceholder);
      if (placeholder_pos != std::string::npos) {
        for (size_t i = 0; i < arch_count; ++i) {
          if (!arch_list[i]) continue;
          for_each_compatible_target(
              arch_list[i], [&](const std::string& target) {
                std::string expanded = resolved;
                expanded.replace(placeholder_pos, kGfxArchPlaceholderLen,
                                 target);
                KPACK_DEBUG(cache, "expanded pattern: %s -> %s",
                            rel_path.c_str(), expanded.c_str());
                kpack_paths.push_back(expanded);
                return false;  // continue — collect all, check existence later
              });
        }
      } else {
        // Direct .kpack path
        KPACK_DEBUG(cache, "direct path: %s -> %s", rel_path.c_str(),
                    resolved.c_str());
        kpack_paths.push_back(resolved);
      }
    }
  }

  // Open/cache archives and build architecture index
  // Lock for archive map access
  std::vector<std::string> valid_archive_paths;
  {
    std::lock_guard<std::mutex> lock(cache->archive_mutex);

    for (const auto& path : kpack_paths) {
      std::string canonical = get_canonical_path(path);

      // Check if already cached
      if (cache->archives.count(canonical) > 0) {
        valid_archive_paths.push_back(canonical);
        continue;
      }

      // Try to open
      if (!file_exists(path)) {
        KPACK_DEBUG(cache, "archive not found: %s", path.c_str());
        continue;
      }

      kpack_archive_t archive = nullptr;
      err = kpack_open(path.c_str(), &archive);
      if (err != KPACK_SUCCESS) {
        KPACK_DEBUG(cache, "failed to open archive: %s (error %d)",
                    path.c_str(), err);
        continue;
      }

      KPACK_DEBUG(cache, "opened and cached archive: %s", path.c_str());

      // Cache the archive
      cache->archives[canonical] = archive;

      // Build architecture index for this archive
      std::set<std::string>& archs = cache->archive_archs[canonical];
      size_t arch_count_in_archive = 0;
      kpack_get_architecture_count(archive, &arch_count_in_archive);
      for (size_t i = 0; i < arch_count_in_archive; ++i) {
        const char* arch = nullptr;
        if (kpack_get_architecture(archive, i, &arch) == KPACK_SUCCESS) {
          archs.insert(arch);
          KPACK_DEBUG(cache, "  architecture: %s", arch);
        }
      }

      valid_archive_paths.push_back(canonical);
    }
  }

  if (valid_archive_paths.empty()) {
    KPACK_DEBUG(cache, "no valid archives found in %zu kpack paths",
                kpack_paths.size());
    return KPACK_ERROR_ARCHIVE_NOT_FOUND;
  }

  // CORRECT SEARCH: arch-first, then archive
  // For each architecture in priority order, search all archives
  void* kernel_data = nullptr;
  size_t kernel_size = 0;

  for (size_t i = 0; i < arch_count; ++i) {
    const char* arch = arch_list[i];
    if (!arch) {
      continue;
    }

    // Try compatible targets (power set of feature flags, most specific first)
    // to find an archive that has a matching architecture.
    KPACK_DEBUG(cache, "trying architecture: %s", arch);

    kpack_archive_t archive = nullptr;
    std::string matched_arch;

    // Find archive containing a compatible architecture
    // Lock only for cache lookup, release before kernel fetch
    {
      std::lock_guard<std::mutex> lock(cache->archive_mutex);

      for_each_compatible_target(arch, [&](const std::string& target) {
        for (const auto& archive_path : valid_archive_paths) {
          auto arch_it = cache->archive_archs.find(archive_path);
          if (arch_it == cache->archive_archs.end() ||
              arch_it->second.count(target) == 0) {
            continue;
          }

          KPACK_DEBUG(cache, "  archive %s has architecture %s",
                      archive_path.c_str(), target.c_str());

          auto archive_it = cache->archives.find(archive_path);
          if (archive_it != cache->archives.end()) {
            archive = archive_it->second;
            matched_arch = target;
            return true;  // found — stop searching
          }
        }
        return false;  // try next compatible target
      });
    }  // Release cache->archive_mutex before kernel fetch

    if (!archive) {
      continue;
    }

    // Fetch kernel using the matched architecture (which may be a subset of
    // the agent's full ISA, e.g., bare "gfx942" for a release build).
    err = kpack_get_kernel(archive, lookup_key.c_str(), matched_arch.c_str(),
                           &kernel_data, &kernel_size);
    if (err == KPACK_SUCCESS) {
      KPACK_DEBUG(cache, "  found kernel: %zu bytes", kernel_size);
      break;
    }
    // Archive was found with matching architecture — the kernel either
    // exists or it doesn't. Trying a different ISA won't help, so return
    // the error directly (KERNEL_NOT_FOUND or unexpected error).
    KPACK_DEBUG(cache, "  kernel not found in this archive (error %d)", err);
    return err;
  }

  if (!kernel_data) {
    KPACK_DEBUG(cache, "no archive with compatible architecture found");
    return KPACK_ERROR_ARCH_NOT_FOUND;
  }

  // kpack_get_kernel() already allocated the data, just pass it through
  *code_object_out = kernel_data;
  *code_object_size_out = kernel_size;

  KPACK_DEBUG(cache, "loaded code object: %zu bytes", kernel_size);

  return KPACK_SUCCESS;
}

void kpack_free_code_object(void* code_object) { std::free(code_object); }

kpack_error_t kpack_enumerate_architectures(const char* archive_path,
                                            kpack_arch_callback_t callback,
                                            void* user_data) {
  // Validate arguments
  if (!archive_path || !callback) {
    return KPACK_ERROR_INVALID_ARGUMENT;
  }

  // Open archive
  kpack_archive_t archive = nullptr;
  kpack_error_t err = kpack_open(archive_path, &archive);
  if (err != KPACK_SUCCESS) {
    return err;
  }

  // Get architecture count
  size_t count = 0;
  err = kpack_get_architecture_count(archive, &count);
  if (err != KPACK_SUCCESS) {
    kpack_close(archive);
    return err;
  }

  // Enumerate architectures
  for (size_t i = 0; i < count; ++i) {
    const char* arch = nullptr;
    err = kpack_get_architecture(archive, i, &arch);
    if (err != KPACK_SUCCESS) {
      kpack_close(archive);
      return err;
    }

    // Invoke callback
    bool should_continue = callback(arch, user_data);
    if (!should_continue) {
      break;
    }
  }

  kpack_close(archive);
  return KPACK_SUCCESS;
}

}  // extern "C"
