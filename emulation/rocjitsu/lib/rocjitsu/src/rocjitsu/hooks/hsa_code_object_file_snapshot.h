// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_HOOKS_HSA_CODE_OBJECT_FILE_SNAPSHOT_H
#define ROCJITSU_HOOKS_HSA_CODE_OBJECT_FILE_SNAPSHOT_H

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace rocjitsu {

using CodeObjectFileSnapshot = std::shared_ptr<const std::vector<uint8_t>>;

[[nodiscard]] CodeObjectFileSnapshot snapshot_code_object_file(int file) noexcept;

[[nodiscard]] CodeObjectFileSnapshot snapshot_code_object_file_range(int file, size_t offset,
                                                                     size_t size) noexcept;

} // namespace rocjitsu

#endif // ROCJITSU_HOOKS_HSA_CODE_OBJECT_FILE_SNAPSHOT_H
