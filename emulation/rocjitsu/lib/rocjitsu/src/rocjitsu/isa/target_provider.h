// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file target_provider.h
/// @brief Helper for target-owned static ISA provider definitions.

#ifndef ROCJITSU_ISA_TARGET_PROVIDER_H_
#define ROCJITSU_ISA_TARGET_PROVIDER_H_

#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/target_registry.h"

#include <memory>
#include <utility>

namespace rocjitsu {

/// @brief Add a target descriptor using the decoder for @p Isa.
template <typename Isa>
[[nodiscard]] IsaTargetRegistryError
add_isa_target(IsaTargetRegistry &registry, const IsaTargetDescription &description,
               const IsaExecutionBackend *execution_backend = nullptr) {
  std::vector<std::string> aliases;
  aliases.reserve(description.aliases.size());
  for (std::string_view alias : description.aliases)
    aliases.emplace_back(alias);

  std::vector<IsaGpuTargetBinding> gpu_targets;
  gpu_targets.reserve(description.gpu_targets.size());
  for (const IsaGpuTargetDescription &gpu_target : description.gpu_targets) {
    gpu_targets.push_back({
        .public_id = gpu_target.public_id,
        .code_object_id = std::string(gpu_target.code_object_id),
        .elf_machine = gpu_target.elf_machine,
    });
  }

  return registry.add({
      .id = std::string(description.id),
      .aliases = std::move(aliases),
      .architecture_id = description.architecture_id,
      .gpu_targets = std::move(gpu_targets),
      .decoder_factory = +[](const IsaExecutionBackend *backend) -> std::unique_ptr<Decoder> {
        return std::make_unique<IsaDecoder<Isa>>(backend);
      },
      .supports_execution = description.supports_execution,
      .execution_backend = execution_backend,
  });
}

} // namespace rocjitsu

#endif // ROCJITSU_ISA_TARGET_PROVIDER_H_
