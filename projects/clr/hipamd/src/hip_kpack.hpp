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

#ifndef HIP_KPACK_HPP
#define HIP_KPACK_HPP

#include <string>
#include <vector>
#include <memory>
#include "hip/hip_runtime.h"

// Forward declare rocm_kpack types
typedef struct kpack_cache* kpack_cache_t;

namespace hip {

// Magic constants for fat binary detection
constexpr unsigned kHipFatMAGIC = 0x48495046;  // "HIPF" - Traditional fat binary
constexpr unsigned kHipKpackMAGIC = 0x4B504948;  // "HIPK" - Kpack external reference

// Structure for parsed kpack metadata
struct KpackMetadata {
  std::string kernel_name;                   // Binary name for TOC lookup
  std::vector<std::string> kpack_search_paths;  // Paths to .kpack archives
};

// Kpack loader class for managing kpack cache and loading code objects
class KpackLoader {
 public:
  KpackLoader();
  ~KpackLoader();

  // Parse MessagePack metadata from .rocm_kpack_ref section
  hipError_t ParseKpackMetadata(const void* metadata_ptr, size_t max_size,
                                 KpackMetadata& metadata);

  // Discover the binary path containing the given address
  // Used to resolve relative paths in kpack metadata
  hipError_t DiscoverBinaryPath(const void* address_in_binary, std::string& binary_path);

  // Load code object from kpack archives for the given architectures
  // Tries each architecture in arch_list until one is found
  // Returns allocated code object (caller must free)
  hipError_t LoadCodeObject(const KpackMetadata& metadata, const std::string& binary_path,
                             const std::vector<std::string>& arch_list, void** code_object_out,
                             size_t* code_object_size_out);

  // Free a code object returned by LoadCodeObject
  void FreeCodeObject(void* code_object);

  // Check if kpack loading is enabled (controlled by env vars)
  bool IsEnabled() const;

 private:
  kpack_cache_t cache_;  // Kpack cache handle
  bool enabled_;         // Whether kpack is enabled
  bool debug_;           // Debug logging enabled
};

// Global kpack loader instance
KpackLoader& GetKpackLoader();

// Helper: Build architecture fallback list for a given device
// E.g., for gfx942:xnack+:sramecc+ returns:
//   ["gfx942:xnack+:sramecc+", "gfx942:xnack+", "gfx942", "gfx9-4-generic:xnack+", "gfx9-4-generic"]
std::vector<std::string> BuildArchFallbackList(const std::string& device_arch);

}  // namespace hip

#endif  // HIP_KPACK_HPP
