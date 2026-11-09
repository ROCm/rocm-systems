// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/base/rj_compiler.h"
#include "rocjitsu/isa/operand.h"

extern "C" RJ_API_EXPORT void rj_probe_operand_delegate(rocjitsu::Operand *operand,
                                                        rocjitsu::Operand *delegate, void *context,
                                                        void (*execute)(void *, bool)) {
  rocjitsu::ScopedOperandDelegate binding(*operand, delegate);
  execute(context, rocjitsu::ScopedOperandDelegate::active());
}
