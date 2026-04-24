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

#include "core/architecture_registry.hpp"

#include <algorithm>
#include <stdexcept>

namespace aql_profile {

ArchitectureRegistry& ArchitectureRegistry::Instance() {
  static ArchitectureRegistry instance;
  return instance;
}

void ArchitectureRegistry::Register(const std::string& gfxip_prefix,
                                     std::unique_ptr<HardwareArchitecture> architecture) {
  if (!architecture) {
    throw std::invalid_argument("Cannot register null architecture");
  }

  std::lock_guard<std::mutex> lock(mutex_);
  architectures_[gfxip_prefix] = std::move(architecture);
}

const HardwareArchitecture* ArchitectureRegistry::Lookup(std::string_view gfxip) const {
  std::lock_guard<std::mutex> lock(mutex_);

  // Try prefix matching - architectures_ is ordered by key length descending
  // so we check longest prefixes first
  for (const auto& [prefix, arch] : architectures_) {
    if (gfxip.rfind(prefix, 0) == 0) {  // starts_with
      return arch.get();
    }
  }

  return nullptr;
}

const HardwareArchitecture* ArchitectureRegistry::GetExact(const std::string& gfxip) const {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = architectures_.find(gfxip);
  if (it != architectures_.end()) {
    return it->second.get();
  }

  return nullptr;
}

std::vector<std::string> ArchitectureRegistry::GetRegisteredPrefixes() const {
  std::lock_guard<std::mutex> lock(mutex_);

  std::vector<std::string> prefixes;
  prefixes.reserve(architectures_.size());

  for (const auto& [prefix, _] : architectures_) {
    prefixes.push_back(prefix);
  }

  return prefixes;
}

void ArchitectureRegistry::Clear() {
  std::lock_guard<std::mutex> lock(mutex_);
  architectures_.clear();
}

}  // namespace aql_profile
