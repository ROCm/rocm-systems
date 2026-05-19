// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file hsa_run_kernel.h
/// @brief Internal entry point for the rj_hsa_run CLI and tests.

#ifndef ROCJITSU_TOOLS_HSA_RUN_KERNEL_H_
#define ROCJITSU_TOOLS_HSA_RUN_KERNEL_H_

#include "tools/tool_result.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace rocjitsu::tools {

struct HsaBufferSpec {
  std::string name;
  size_t size = 0;
  std::vector<uint8_t> input;
  bool zero_fill = false;
  bool copy_output = false;
};

enum class KernelArgKind {
  Pointer,
  U32,
  U64,
  I32,
  F32,
  RawBytes,
};

struct KernelArgPatch {
  size_t offset = 0;
  KernelArgKind kind = KernelArgKind::RawBytes;
  std::string buffer_name;
  std::vector<uint8_t> bytes;
};

struct HsaRunOptions {
  std::string code_object_path;
  std::vector<uint8_t> code_object_bytes;

  std::string kernel_name;
  std::string kernel_symbol;

  uint32_t grid_x = 1;
  uint32_t grid_y = 1;
  uint32_t grid_z = 1;
  uint16_t workgroup_x = 1;
  uint16_t workgroup_y = 1;
  uint16_t workgroup_z = 1;

  std::string require_agent_isa;
  uint32_t agent_index = 0;
  uint64_t timeout_ms = 5000;

  size_t kernarg_size = 0;
  std::vector<uint8_t> kernarg_template;
  std::vector<HsaBufferSpec> buffers;
  std::vector<KernelArgPatch> arg_patches;
};

struct HsaBufferOutput {
  std::string name;
  std::vector<uint8_t> bytes;
};

struct HsaRunOutput {
  std::string agent_isa;
  uint64_t elapsed_ns = 0;
  std::vector<HsaBufferOutput> outputs;
};

/// @brief Load a code object through HSA, dispatch one kernel, and copy outputs.
///
/// When ROCm HSA is unavailable at build time this function is still present,
/// but returns a structured error. That lets non-HSA builds compile tests that
/// are skipped by configuration or runtime environment.
[[nodiscard]] ToolResult<HsaRunOutput> run_hsa_kernel(const HsaRunOptions &options);

} // namespace rocjitsu::tools

#endif // ROCJITSU_TOOLS_HSA_RUN_KERNEL_H_
