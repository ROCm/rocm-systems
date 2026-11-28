// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "addc/mca/decoder.hpp"

namespace addc::mca::base::mi300x
{

[[nodiscard]] uint32_t decode_afid(McaRegisters regs);
[[nodiscard]] McaResult decode(const BankCapture& bank);

} // namespace addc::mca::base::mi300x
