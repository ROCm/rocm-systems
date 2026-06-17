//===- hotswap_platform_io.hpp - Platform-specific file I/O shims ---------===//
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ROCR_HOTSWAP_PLATFORM_IO_HPP
#define ROCR_HOTSWAP_PLATFORM_IO_HPP

#include <cstddef>
#include <cstdint>
#include <hsa.h>

namespace rocr::hotswap::platform_io {

using file_pos_t = uint64_t;

bool read_all(hsa_file_t file, void *buf, size_t count);
bool get_file_bounds(hsa_file_t file, file_pos_t *saved_pos,
                     file_pos_t *file_size);
void restore_file_pos(hsa_file_t file, file_pos_t pos);

} // namespace rocr::hotswap::platform_io

#endif // ROCR_HOTSWAP_PLATFORM_IO_HPP
