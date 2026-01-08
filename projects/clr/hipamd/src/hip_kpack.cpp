/*
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include "hip_kpack.hpp"
#include "hip_internal.hpp"
#include <msgpack.hpp>
#include <rocm_kpack/kpack.h>
#include <cstring>
#include <sstream>

namespace hip {

// Global kpack loader instance
static std::unique_ptr<KpackLoader> g_kpack_loader;

KpackLoader::KpackLoader() : cache_(nullptr), enabled_(false), debug_(false) {
  // Check if kpack is disabled via environment variable
  const char* disable_env = std::getenv("ROCM_KPACK_DISABLE");
  if (disable_env && std::atoi(disable_env) != 0) {
    LogInfo("Kpack loading disabled via ROCM_KPACK_DISABLE");
    enabled_ = false;
    return;
  }

  // Check if debug logging is enabled
  const char* debug_env = std::getenv("ROCM_KPACK_DEBUG");
  debug_ = (debug_env && std::atoi(debug_env) != 0);

  // Create kpack cache
  kpack_error_t err = kpack_cache_create(&cache_);
  if (err != KPACK_SUCCESS) {
    LogError("Failed to create kpack cache, kpack loading disabled");
    enabled_ = false;
    cache_ = nullptr;
    return;
  }

  enabled_ = true;
  if (debug_) {
    LogInfo("Kpack loader initialized successfully");
  }
}

KpackLoader::~KpackLoader() {
  if (cache_) {
    kpack_cache_destroy(cache_);
    cache_ = nullptr;
  }
}

KpackLoader& GetKpackLoader() {
  if (!g_kpack_loader) {
    g_kpack_loader = std::make_unique<KpackLoader>();
  }
  return *g_kpack_loader;
}

hipError_t KpackLoader::ParseKpackMetadata(const void* metadata_ptr, size_t max_size,
                                             KpackMetadata& metadata) {
  if (!metadata_ptr) {
    return hipErrorInvalidValue;
  }

  try {
    // Parse MessagePack data
    msgpack::object_handle oh = msgpack::unpack(static_cast<const char*>(metadata_ptr), max_size);
    msgpack::object obj = oh.get();

    // Convert to map
    std::map<std::string, msgpack::object> map;
    obj.convert(map);

    // Extract kernel_name
    if (map.find("kernel_name") != map.end()) {
      metadata.kernel_name = map["kernel_name"].as<std::string>();
    } else {
      LogError("Kpack metadata missing 'kernel_name' field");
      return hipErrorInvalidKernelFile;
    }

    // Extract kpack_search_paths
    if (map.find("kpack_search_paths") != map.end()) {
      metadata.kpack_search_paths = map["kpack_search_paths"].as<std::vector<std::string>>();
    } else {
      LogError("Kpack metadata missing 'kpack_search_paths' field");
      return hipErrorInvalidKernelFile;
    }

    if (debug_) {
      LogInfo("Parsed kpack metadata: kernel_name='%s', %zu search paths",
              metadata.kernel_name.c_str(), metadata.kpack_search_paths.size());
    }

    return hipSuccess;
  } catch (const std::exception& e) {
    LogError("Failed to parse kpack metadata: %s", e.what());
    return hipErrorInvalidKernelFile;
  }
}

hipError_t KpackLoader::DiscoverBinaryPath(const void* address_in_binary,
                                             std::string& binary_path) {
  char path_buffer[4096];
  kpack_error_t err =
      kpack_discover_binary_path(address_in_binary, path_buffer, sizeof(path_buffer), nullptr);

  if (err != KPACK_SUCCESS) {
    LogError("Failed to discover binary path for address %p", address_in_binary);
    return hipErrorInvalidKernelFile;
  }

  binary_path = std::string(path_buffer);
  if (debug_) {
    LogInfo("Discovered binary path: %s", binary_path.c_str());
  }
  return hipSuccess;
}

hipError_t KpackLoader::LoadCodeObject(const KpackMetadata& metadata,
                                         const std::string& binary_path,
                                         const std::vector<std::string>& arch_list,
                                         void** code_object_out, size_t* code_object_size_out) {
  if (!enabled_ || !cache_) {
    return hipErrorNotSupported;
  }

  if (!code_object_out || !code_object_size_out) {
    return hipErrorInvalidValue;
  }

  if (arch_list.empty()) {
    LogError("Empty architecture list for kpack loading");
    return hipErrorInvalidValue;
  }

  // Parse hipk metadata (MessagePack format)
  // For now, we construct a simple MessagePack representation
  // In reality, this should be passed directly from the .rocm_kpack_ref section
  msgpack::sbuffer sbuf;
  msgpack::packer<msgpack::sbuffer> packer(&sbuf);
  packer.pack_map(2);
  packer.pack("kernel_name");
  packer.pack(metadata.kernel_name);
  packer.pack("kpack_search_paths");
  packer.pack_array(metadata.kpack_search_paths.size());
  for (const auto& path : metadata.kpack_search_paths) {
    packer.pack(path);
  }

  // Convert arch_list to C array
  std::vector<const char*> arch_ptrs;
  for (const auto& arch : arch_list) {
    arch_ptrs.push_back(arch.c_str());
  }

  // Call kpack loader
  kpack_error_t err =
      kpack_load_code_object(cache_, sbuf.data(), binary_path.c_str(), arch_ptrs.data(),
                             arch_ptrs.size(), code_object_out, code_object_size_out);

  if (err == KPACK_ERROR_ARCH_NOT_FOUND) {
    if (debug_) {
      std::string arch_list_str;
      for (const auto& arch : arch_list) {
        if (!arch_list_str.empty()) arch_list_str += ", ";
        arch_list_str += arch;
      }
      LogInfo("No matching architecture found in kpack archives");
      LogInfo("  Searched for: %s", arch_list_str.c_str());
    }
    return hipErrorNoBinaryForGpu;
  } else if (err == KPACK_ERROR_ARCHIVE_NOT_FOUND) {
    // Archive file physically missing on disk
    LogError("Kpack archive file not found");
    LogError("  Binary: %s", metadata.kernel_name.c_str());
    LogError("  Searched paths:");
    for (const auto& path : metadata.kpack_search_paths) {
      LogError("    - %s", path.c_str());
    }
    if (debug_) {
      LogError("  This usually indicates:");
      LogError("    1. Incomplete kpack installation");
      LogError("    2. Architecture-specific package not installed");
      LogError("    3. Incorrect ROCM_KPACK_PATH");
    }
    return hipErrorFileNotFound;
  } else if (err == KPACK_ERROR_INVALID_METADATA) {
    LogError("Invalid kpack metadata format");
    return hipErrorInvalidKernelFile;
  } else if (err != KPACK_SUCCESS) {
    LogError("Kpack load failed with error code: %d", err);
    return hipErrorInvalidKernelFile;
  }

  if (debug_) {
    LogInfo("Successfully loaded code object from kpack: %zu bytes", *code_object_size_out);
  }

  return hipSuccess;
}

void KpackLoader::FreeCodeObject(void* code_object) {
  if (code_object) {
    kpack_free_code_object(code_object);
  }
}

bool KpackLoader::IsEnabled() const { return enabled_; }

// Architecture fallback logic
std::vector<std::string> BuildArchFallbackList(const std::string& device_arch) {
  std::vector<std::string> fallback_list;

  // Add the full architecture string first
  fallback_list.push_back(device_arch);

  // Parse features (xnack, sramecc)
  std::string base_arch = device_arch;
  std::vector<std::string> features;
  size_t pos = device_arch.find(':');
  if (pos != std::string::npos) {
    base_arch = device_arch.substr(0, pos);
    std::string feature_str = device_arch.substr(pos + 1);
    // Split features by ':'
    std::stringstream ss(feature_str);
    std::string feature;
    while (std::getline(ss, feature, ':')) {
      features.push_back(feature);
    }
  }

  // Add base architecture without features
  if (!features.empty()) {
    fallback_list.push_back(base_arch);
  }

  // Map to family-level generic (e.g., gfx110x-generic, gfx94x-generic)
  static const std::unordered_map<std::string, std::string> family_generic_map = {
      {"gfx900", "gfx90x-generic"},   {"gfx902", "gfx90x-generic"},
      {"gfx904", "gfx90x-generic"},   {"gfx906", "gfx90x-generic"},
      {"gfx908", "gfx90x-generic"},   {"gfx909", "gfx90x-generic"},
      {"gfx90a", "gfx90x-generic"},   {"gfx90c", "gfx90x-generic"},
      {"gfx940", "gfx94x-generic"},   {"gfx941", "gfx94x-generic"},
      {"gfx942", "gfx94x-generic"},   {"gfx950", "gfx94x-generic"},
      {"gfx1010", "gfx101x-generic"}, {"gfx1011", "gfx101x-generic"},
      {"gfx1012", "gfx101x-generic"}, {"gfx1013", "gfx101x-generic"},
      {"gfx1030", "gfx103x-generic"}, {"gfx1031", "gfx103x-generic"},
      {"gfx1032", "gfx103x-generic"}, {"gfx1033", "gfx103x-generic"},
      {"gfx1034", "gfx103x-generic"}, {"gfx1035", "gfx103x-generic"},
      {"gfx1036", "gfx103x-generic"}, {"gfx1100", "gfx110x-generic"},
      {"gfx1101", "gfx110x-generic"}, {"gfx1102", "gfx110x-generic"},
      {"gfx1103", "gfx110x-generic"}, {"gfx1150", "gfx115x-generic"},
      {"gfx1151", "gfx115x-generic"}, {"gfx1152", "gfx115x-generic"},
      {"gfx1153", "gfx115x-generic"}, {"gfx1200", "gfx120x-generic"},
      {"gfx1201", "gfx120x-generic"},
  };

  // Map to generation-level generic (e.g., gfx11-generic, gfx9-4-generic)
  static const std::unordered_map<std::string, std::string> generation_generic_map = {
      {"gfx900", "gfx9-generic"},     {"gfx902", "gfx9-generic"},
      {"gfx904", "gfx9-generic"},     {"gfx906", "gfx9-generic"},
      {"gfx908", "gfx9-generic"},     {"gfx909", "gfx9-generic"},
      {"gfx90a", "gfx9-generic"},     {"gfx90c", "gfx9-generic"},
      {"gfx940", "gfx9-4-generic"},   {"gfx941", "gfx9-4-generic"},
      {"gfx942", "gfx9-4-generic"},   {"gfx950", "gfx9-4-generic"},
      {"gfx1010", "gfx10-1-generic"}, {"gfx1011", "gfx10-1-generic"},
      {"gfx1012", "gfx10-1-generic"}, {"gfx1013", "gfx10-1-generic"},
      {"gfx1030", "gfx10-3-generic"}, {"gfx1031", "gfx10-3-generic"},
      {"gfx1032", "gfx10-3-generic"}, {"gfx1033", "gfx10-3-generic"},
      {"gfx1034", "gfx10-3-generic"}, {"gfx1035", "gfx10-3-generic"},
      {"gfx1036", "gfx10-3-generic"}, {"gfx1100", "gfx11-generic"},
      {"gfx1101", "gfx11-generic"},   {"gfx1102", "gfx11-generic"},
      {"gfx1103", "gfx11-generic"},   {"gfx1150", "gfx11-generic"},
      {"gfx1151", "gfx11-generic"},   {"gfx1152", "gfx11-generic"},
      {"gfx1153", "gfx11-generic"},   {"gfx1200", "gfx12-generic"},
      {"gfx1201", "gfx12-generic"},
  };

  // Add family-level generic (e.g., gfx110x-generic)
  auto family_it = family_generic_map.find(base_arch);
  if (family_it != family_generic_map.end()) {
    const std::string& family_generic = family_it->second;

    // Add family generic with features
    if (!features.empty()) {
      std::string family_with_features = family_generic;
      for (const auto& feature : features) {
        family_with_features += ":" + feature;
      }
      fallback_list.push_back(family_with_features);
    }

    // Add family generic without features
    fallback_list.push_back(family_generic);
  }

  // Add generation-level generic (e.g., gfx11-generic, gfx9-4-generic)
  auto gen_it = generation_generic_map.find(base_arch);
  if (gen_it != generation_generic_map.end()) {
    const std::string& generation_generic = gen_it->second;

    // Add generation generic with features
    if (!features.empty()) {
      std::string gen_with_features = generation_generic;
      for (const auto& feature : features) {
        gen_with_features += ":" + feature;
      }
      fallback_list.push_back(gen_with_features);
    }

    // Add generation generic without features
    fallback_list.push_back(generation_generic);
  }

  return fallback_list;
}

}  // namespace hip
