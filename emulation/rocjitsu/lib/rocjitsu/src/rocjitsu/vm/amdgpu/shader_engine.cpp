// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/shader_engine.h"

#include <memory>
#include <string>

namespace rocjitsu {
namespace amdgpu {

ShaderEngine::ShaderEngine(std::string name, const Config &config, rj_code_arch_t arch,
                           GpuMemory *memory, L2Cache *l2, simdojo::ExecMode exec_mode)
    : simdojo::CompositeComponent(std::move(name)) {
  set_weight(0); // Structural container, not a work-producing component.
  auto se_name = this->name();

  // Inject architecture into the CU config.
  auto cu_config = config.compute_unit;
  cu_config.arch = arch;

  // Create compute units, sharing the XCD's L2 cache.
  for (uint32_t i = 0; i < config.num_compute_units; ++i) {
    auto cu = ComputeUnitCore::create(se_name + ".cu" + std::to_string(i), cu_config, memory, l2,
                                      exec_mode);
    cus_.push_back(static_cast<ComputeUnitCore *>(cu.get()));
    add_child(std::move(cu));
  }
}

} // namespace amdgpu
} // namespace rocjitsu
