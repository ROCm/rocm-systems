// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file shader_engine.h
/// @brief Shader engine containing an array of compute units.

#ifndef ROCJITSU_VM_AMDGPU_SHADER_ENGINE_H_
#define ROCJITSU_VM_AMDGPU_SHADER_ENGINE_H_

#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/spi.h"

#include "simdojo/sim/component.h"

#include <cassert>
#include <cstdint>
#include <string>
#include <vector>

namespace rocjitsu {
namespace amdgpu {

/// @brief A shader engine containing compute units.
///
/// @details Mirrors the AMDGPU hardware hierarchy where each shader engine is an
/// array of compute units within an XCD. The command processor lives at
/// the XCD level; the shader engine owns only its CU array. Purely
/// structural - CUs are activated by the command processor through ports.
class ShaderEngine : public simdojo::CompositeComponent {
public:
  /// @brief Configuration for a shader engine.
  struct Config {
    uint32_t num_compute_units;           ///< Number of CUs in this SE.
    ComputeUnitCore::Config compute_unit; ///< Configuration applied to each CU.
  };

  /// @brief Construct a shader engine from configuration.
  /// @param name Human-readable name (e.g., "se0").
  /// @param config Shader engine configuration.
  /// @param arch ISA architecture for CU creation.
  /// @param memory Shared GPU memory (not owned).
  /// @param l2 Shared L2 cache (owned by the XCD).
  /// @param exec_mode Execution mode for CU creation.
  ShaderEngine(std::string name, const Config &config, rj_code_arch_t arch, GpuMemory *memory,
               L2Cache *l2, simdojo::ExecMode exec_mode = simdojo::ExecMode::FUNCTIONAL);

  /// @brief Construct an empty shader engine (children added externally by the config loader).
  explicit ShaderEngine(std::string name) : simdojo::CompositeComponent(std::move(name)) {
    set_weight(0);
  }

  void add_compute_unit(ComputeUnitCore *cu) { cus_.push_back(cu); }

  /// @brief Set the execution plugin group on all CUs (shared ownership).
  void set_plugin_group(std::shared_ptr<ExecutionPluginGroup> pg) {
    for (auto *cu : cus_)
      cu->set_plugin_group(pg);
  }

  /// @brief Return the number of compute units.
  /// @returns Number of CUs in this shader engine.
  uint32_t num_compute_units() const { return static_cast<uint32_t>(cus_.size()); }

  /// @brief Access a compute unit by index.
  /// @param idx Zero-based CU index.
  /// @returns Pointer to the compute unit.
  ComputeUnitCore *compute_unit(uint32_t idx) {
    assert(idx < cus_.size());
    return cus_[idx];
  }
  /// @returns Const pointer to the compute unit.
  const ComputeUnitCore *compute_unit(uint32_t idx) const {
    assert(idx < cus_.size());
    return cus_[idx];
  }

  /// @brief Return all compute units.
  /// @returns Const reference to the vector of CU pointers.
  const std::vector<ComputeUnitCore *> &compute_units() const { return cus_; }

  /// @brief Get or create the SPI for this shader engine.
  ShaderProcessorInput &spi() {
    if (!spi_)
      spi_ = std::make_unique<ShaderProcessorInput>(cus_);
    return *spi_;
  }

private:
  std::vector<ComputeUnitCore *> cus_;
  std::unique_ptr<ShaderProcessorInput> spi_;
};

} // namespace amdgpu
} // namespace rocjitsu

#endif // ROCJITSU_VM_AMDGPU_SHADER_ENGINE_H_
