// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/isa/isa_family.h"

#include <ostream>

namespace rocjitsu {

std::ostream &operator<<(std::ostream &stream, IsaFamily family) {
  return stream << static_cast<int32_t>(family);
}

} // namespace rocjitsu
