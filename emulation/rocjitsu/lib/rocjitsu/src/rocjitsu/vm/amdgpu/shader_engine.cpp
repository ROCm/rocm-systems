// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/shader_engine.h"

#include "util/diagnostic.h"

#include <memory>
#include <string>

namespace rocjitsu {
namespace amdgpu {

std::unique_ptr<ShaderEngine> ShaderEngine::create(std::string name, const Config &config,
                                                   rj_code_arch_t arch, GpuMemory *memory,
                                                   L2Cache *l2, simdojo::ExecMode exec_mode,
                                                   util::DiagnosticEmitter emit_error) {
  auto shader_engine = std::make_unique<ShaderEngine>(std::move(name));
  if (shader_engine->configure(config, arch, memory, l2, exec_mode, emit_error).failed())
    return nullptr;
  return shader_engine;
}

util::Result ShaderEngine::configure(const Config &config, rj_code_arch_t arch, GpuMemory *memory,
                                     L2Cache *l2, simdojo::ExecMode exec_mode,
                                     util::DiagnosticEmitter emit_error) {
  set_weight(0); // Structural container, not a work-producing component.
  auto se_name = this->name();

  // Inject architecture into the CU config.
  auto cu_config = config.compute_unit;
  cu_config.arch = arch;

  // Create compute units, sharing the XCD's L2 cache.
  for (uint32_t i = 0; i < config.num_compute_units; ++i) {
    auto cu = ComputeUnitCore::create(se_name + ".cu" + std::to_string(i), cu_config, memory, l2,
                                      exec_mode, emit_error);
    if (!cu)
      return util::Result::failure();
    cus_.push_back(static_cast<ComputeUnitCore *>(cu.get()));
    add_child(std::move(cu));
  }
  return util::Result::success();
}

} // namespace amdgpu
} // namespace rocjitsu
