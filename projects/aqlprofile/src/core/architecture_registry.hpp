// MIT License
//
// Copyright (c) 2017-2025 Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef SRC_CORE_ARCHITECTURE_REGISTRY_HPP_
#define SRC_CORE_ARCHITECTURE_REGISTRY_HPP_

#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>

#include "core/hardware_architecture.hpp"

namespace aql_profile {

/// Singleton registry for hardware architectures
/// Maps gfxip strings to architecture instances
/// Replaces the hardcoded GetGpuId() function and switch-based factory creation
class ArchitectureRegistry {
 public:
  /// Get the singleton instance
  static ArchitectureRegistry& Instance();

  /// Register an architecture implementation
  /// The registry takes ownership of the architecture
  void Register(const std::string& gfxip_prefix,
                std::unique_ptr<HardwareArchitecture> architecture);

  /// Look up architecture by gfxip string
  /// Uses prefix matching (e.g., "gfx908" matches "gfx908" or "gfx90")
  /// Returns nullptr if not found
  const HardwareArchitecture* Lookup(std::string_view gfxip) const;

  /// Get architecture by exact gfxip match
  /// Returns nullptr if not found
  const HardwareArchitecture* GetExact(const std::string& gfxip) const;

  /// Check if an architecture is registered
  bool IsRegistered(std::string_view gfxip) const {
    return Lookup(gfxip) != nullptr;
  }

  /// Get all registered gfxip prefixes
  std::vector<std::string> GetRegisteredPrefixes() const;

  /// Clear all registered architectures (mainly for testing)
  void Clear();

 private:
  ArchitectureRegistry() = default;
  ~ArchitectureRegistry() = default;

  // Non-copyable, non-movable
  ArchitectureRegistry(const ArchitectureRegistry&) = delete;
  ArchitectureRegistry& operator=(const ArchitectureRegistry&) = delete;

  mutable std::mutex mutex_;
  // Map from gfxip prefix to architecture
  // Ordered by key length (descending) to match longest prefix first
  std::map<std::string, std::unique_ptr<HardwareArchitecture>,
           std::function<bool(const std::string&, const std::string&)>> architectures_{
      [](const std::string& a, const std::string& b) {
        // Sort by length descending, then lexicographically
        // This ensures "gfx908" is checked before "gfx90"
        if (a.length() != b.length()) return a.length() > b.length();
        return a < b;
      }};
};

}  // namespace aql_profile

#endif  // SRC_CORE_ARCHITECTURE_REGISTRY_HPP_
