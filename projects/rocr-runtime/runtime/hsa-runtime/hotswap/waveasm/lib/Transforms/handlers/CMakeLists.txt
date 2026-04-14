# Copyright 2025 The Wave Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Handler files for per-dialect MLIR to WaveASM translation
# These files implement operation handlers organized by source MLIR dialect.

set(HANDLERS_SRCS
  AffineHandlers.cpp
  AMDGPUHandlers.cpp
  ArithHandlers.cpp
  GPUHandlers.cpp
  HandlerUtils.cpp
  MathHandlers.cpp
  MemRefHandlers.cpp
  SCFHandlers.cpp
  VectorHandlers.cpp
)

# Export handler sources to parent scope for inclusion in the main library
set(HANDLERS_SRCS ${HANDLERS_SRCS} PARENT_SCOPE)
