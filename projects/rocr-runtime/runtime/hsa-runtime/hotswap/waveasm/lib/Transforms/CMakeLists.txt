# Copyright 2025 The Wave Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Include handler subdirectory to get HANDLERS_SRCS
add_subdirectory(handlers)

# Prepend handlers/ path to each handler source file
set(HANDLERS_FULL_PATHS)
foreach(src ${HANDLERS_SRCS})
  list(APPEND HANDLERS_FULL_PATHS handlers/${src})
endforeach()

add_mlir_dialect_library(MLIRWaveASMTransforms
  ArithLegalization.cpp
  AssemblyEmitter.cpp
  BufferLoadStrengthReduction.cpp
  ExtractScalarization.cpp
  HazardMitigation.cpp
  LinearScanPass.cpp
  LinearScanRegAlloc.cpp
  LiteralMaterialization.cpp
  Liveness.cpp
  LoopAddressPromotion.cpp
  M0RedundancyElimination.cpp
  MemoryOffsetOptimization.cpp
  MetadataEmitter.cpp
  Peephole.cpp
  RegionBuilder.cpp
  ScalePackElimination.cpp
  SCCSpillReload.cpp
  SCCVerifier.cpp
  ScopedCSE.cpp
  Ticketing.cpp
  TranslateFromLLVMDialect.cpp
  TranslateFromMLIR.cpp
  VGPRCompaction.cpp
  ${HANDLERS_FULL_PATHS}

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/waveasm

  DEPENDS
  MLIRWaveASMPassesIncGen

  LINK_LIBS PUBLIC
  MLIRAffineDialect
  MLIRAMDGPUDialect
  MLIRArithDialect
  MLIRFuncDialect
  MLIRGPUDialect
  MLIRIR
  MLIRLLVMDialect
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRPass
  MLIRROCDLDialect
  MLIRSCFDialect
  MLIRSupport
  MLIRVectorDialect
  MLIRWaveASMDialect
  MLIRWaveASMTarget
)
