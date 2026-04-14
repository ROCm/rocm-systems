# Copyright 2025 The Wave Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_mlir_dialect_library(MLIRWaveASMDialect
  WaveASMAttrs.cpp
  WaveASMControlFlowOps.cpp
  WaveASMDialect.cpp
  WaveASMOps.cpp
  WaveASMTypes.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/waveasm

  DEPENDS
  MLIRWaveASMIncGen

  LINK_LIBS PUBLIC
  MLIRControlFlowInterfaces
  MLIRFunctionInterfaces
  MLIRIR
  MLIRLoopLikeInterface
  MLIRSupport
)
