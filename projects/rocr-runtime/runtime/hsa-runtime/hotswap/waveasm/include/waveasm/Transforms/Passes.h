// Copyright 2026 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef WaveASM_TRANSFORMS_PASSES_H
#define WaveASM_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace waveasm {

#define GEN_PASS_DECL
#include "waveasm/Transforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "waveasm/Transforms/Passes.h.inc"

} // namespace waveasm

#endif // WaveASM_TRANSFORMS_PASSES_H
