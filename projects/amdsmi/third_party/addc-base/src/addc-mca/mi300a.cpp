// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "addc/mca/mi300a.hpp"

#include "mi300_gpu_afid.hpp"

namespace addc::mca::base::mi300a
{

uint32_t decode_afid(McaRegisters regs)
{
    return detail::gpu_decode_afid(regs);
}

McaResult decode(const BankCapture& bank)
{
    return decode_common(bank, decode_afid(bank.regs));
}

} // namespace addc::mca::base::mi300a
