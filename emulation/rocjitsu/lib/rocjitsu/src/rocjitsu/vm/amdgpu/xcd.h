// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file xcd.h
/// @brief Accelerator Complex Die (XCD) containing shader engines and a command processor.

#ifndef ROCJITSU_VM_AMDGPU_XCD_H_
#define ROCJITSU_VM_AMDGPU_XCD_H_

#include "rocjitsu/vm/amdgpu/command_processor.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/shader_engine.h"

#include "simdojo/sim/component.h"
#include "simdojo/sim/exec_mode.h"

#include <cassert>
#include <cstdint>
#include <string>
#include <vector>

namespace rocjitsu {
namespace amdgpu {

/// @brief Accelerator Complex Die containing shader engines and a command processor.
///
/// @details Mirrors the CDNA hardware hierarchy where each XCD is an independent
/// compute chiplet with its own command processor and shader engine array.
/// The command processor dispatches work to CUs across all shader engines.
class Xcd : public simdojo::CompositeComponent {
public:
  /// @brief Configuration for an Accelerator Complex Die (XCD).
  struct Config {
    uint32_t num_shader_engines;        ///< Number of shader engines in this XCD.
    ShaderEngine::Config shader_engine; ///< Config applied to each SE.
  };

  /// @brief Construct an XCD from configuration.
  /// @param name Human-readable name (e.g., "xcd0").
  /// @param config XCD configuration parameters.
  /// @param arch ISA architecture.
  /// @param memory Shared GPU memory for instruction fetch (not owned).
  /// @param exec_mode Execution mode for CU creation.
  Xcd(std::string name, const Config &config, rj_code_arch_t arch, GpuMemory *memory,
      simdojo::ExecMode exec_mode = simdojo::ExecMode::FUNCTIONAL);

  /// @brief Construct an empty XCD (children added externally by the config loader).
  explicit Xcd(std::string name) : simdojo::CompositeComponent(std::move(name)) { set_weight(0); }

  /// @brief Set typed child pointers after the builder populates children.
  void set_command_processor(CommandProcessor *cp) { cp_ = cp; }
  void set_l2_cache(L2Cache *l2) { l2_cache_ = l2; }
  void add_shader_engine(ShaderEngine *se) { shader_engines_.push_back(se); }

  /// @brief Set flat-address-space aperture boundaries on all CUs via their SPIs.
  void set_apertures(uint64_t shared_base, uint64_t shared_limit, uint64_t private_base,
                     uint64_t private_limit) {
    for (auto *se : shader_engines_)
      se->spi().set_apertures(shared_base, shared_limit, private_base, private_limit);
  }

  /// @brief Set the execution plugin group on CP and all CUs (shared ownership).
  void set_plugin_group(std::shared_ptr<ExecutionPluginGroup> pg) {
    cp_->set_plugin_group(pg);
    for (auto *se : shader_engines_)
      se->set_plugin_group(pg);
  }

  /// @brief Wire topology links between CP→CU and CU→L2.
  ///
  /// @details L2→HBM/fabric wiring is done by SoC::initialize() since the
  /// destination depends on the IOD topology.
  void initialize() override;

  /// @brief Return the command processor.
  /// @returns Pointer to the command processor.
  CommandProcessor *command_processor() { return cp_; }
  /// @returns Const pointer to the command processor.
  const CommandProcessor *command_processor() const { return cp_; }

  /// @brief Return the number of shader engines.
  /// @returns Number of shader engines in this XCD.
  uint32_t num_shader_engines() const { return static_cast<uint32_t>(shader_engines_.size()); }

  /// @brief Access a shader engine by index.
  /// @param idx Zero-based shader engine index.
  /// @returns Pointer to the shader engine.
  ShaderEngine *shader_engine(uint32_t idx) {
    assert(idx < shader_engines_.size());
    return shader_engines_[idx];
  }
  /// @returns Const pointer to the shader engine.
  const ShaderEngine *shader_engine(uint32_t idx) const {
    assert(idx < shader_engines_.size());
    return shader_engines_[idx];
  }

  /// @brief Return all shader engines.
  /// @returns Const reference to the vector of shader engine pointers.
  const std::vector<ShaderEngine *> &shader_engines() const { return shader_engines_; }

  /// @brief Return the shared L2 cache component.
  /// @returns Pointer to the L2 cache.
  L2Cache *l2_cache() { return l2_cache_; }
  /// @returns Const pointer to the L2 cache.
  const L2Cache *l2_cache() const { return l2_cache_; }

private:
  simdojo::ExecMode exec_mode_;
  CommandProcessor *cp_ = nullptr;
  L2Cache *l2_cache_ = nullptr;
  std::vector<ShaderEngine *> shader_engines_;
};

} // namespace amdgpu
} // namespace rocjitsu

#endif // ROCJITSU_VM_AMDGPU_XCD_H_
