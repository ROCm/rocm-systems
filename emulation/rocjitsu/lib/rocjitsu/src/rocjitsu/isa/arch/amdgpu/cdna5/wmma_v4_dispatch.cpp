// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/isa/arch/amdgpu/cdna5/wmma_v4_provider.h"

#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/execution_backend.h"
#include "rocjitsu/isa/arch/amdgpu/shared/x86_v4_provider.h"
#include "util/simd.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <stdexcept>
#include <string_view>

namespace rocjitsu::cdna5 {
namespace {

constexpr size_t kCallbackCount = static_cast<size_t>(InstructionExecutionId::Count);
using CallbackTable = std::array<Instruction::ExecuteFn, kCallbackCount>;

enum class BackendRequest { Auto, V3, V4 };

BackendRequest backend_request() {
  const char *raw = std::getenv("RJ_CDNA5_WMMA_BACKEND");
  if (!raw || std::string_view(raw) == "auto")
    return BackendRequest::Auto;
  if (std::string_view(raw) == "v3")
    return BackendRequest::V3;
  if (std::string_view(raw) == "v4")
    return BackendRequest::V4;
  throw std::invalid_argument("RJ_CDNA5_WMMA_BACKEND must be auto, v3, or v4");
}

struct SelectedBackend {
  CallbackTable callbacks{};
  IsaExecutionBackend backend{};
  bool enabled = false;

  SelectedBackend() {
    const IsaExecutionBackend &baseline = execution_backend();
    if (baseline.instruction_callback_count != callbacks.size())
      throw std::logic_error("CDNA5 callback count does not match generated execution IDs");
    std::copy_n(baseline.instruction_callbacks, callbacks.size(), callbacks.begin());
    backend = {callbacks.data(), callbacks.size(), baseline.operand_backend};

    if (util::force_scalar())
      return;
    const BackendRequest request = backend_request();
    if (request == BackendRequest::V3)
      return;
    const auto *provider = resolve_x86_v4_provider(request == BackendRequest::V4);
    if (!provider)
      return;

    constexpr std::array ids{
        InstructionExecutionId::VWmmaF3216x16x32F16Vop3p,
        InstructionExecutionId::VWmmaF1616x16x32F16Vop3p,
        InstructionExecutionId::VWmmaF3216x16x32Bf16Vop3p,
        InstructionExecutionId::VWmmaBf1616x16x32Bf16Vop3p,
        InstructionExecutionId::VWmmaBf16f3216x16x32Bf16Vop3p,
    };
    static_assert(ids.size() == kWmmaV4FormCount);
    provider->bind_wmma_backend(&baseline);
    for (size_t i = 0; i < ids.size(); ++i)
      callbacks[static_cast<size_t>(ids[i])] = provider->wmma_callbacks[i];
    enabled = true;
  }
};

} // namespace

const IsaExecutionBackend &selected_wmma_backend() {
  static const SelectedBackend selected;
  return selected.enabled ? selected.backend : execution_backend();
}

} // namespace rocjitsu::cdna5

extern "C" __attribute__((visibility("hidden"))) const rocjitsu::IsaExecutionBackend *
rj_cdna5_wmma_shared_backend() {
  return &rocjitsu::cdna5::selected_wmma_backend();
}
