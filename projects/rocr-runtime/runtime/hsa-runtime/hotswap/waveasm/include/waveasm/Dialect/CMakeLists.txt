# Copyright 2025 The Wave Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#-------------------------------------------------------------------------------
# Dialect TableGen
#-------------------------------------------------------------------------------

set(LLVM_TARGET_DEFINITIONS WaveASMDialect.td)
mlir_tablegen(WaveASMDialect.h.inc -gen-dialect-decls)
mlir_tablegen(WaveASMDialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(MLIRWaveASMDialectIncGen)


#-------------------------------------------------------------------------------
# Types TableGen
#-------------------------------------------------------------------------------

set(LLVM_TARGET_DEFINITIONS WaveASMTypes.td)
mlir_tablegen(WaveASMTypes.h.inc -gen-typedef-decls)
mlir_tablegen(WaveASMTypes.cpp.inc -gen-typedef-defs)
mlir_tablegen(WaveASMEnums.h.inc -gen-enum-decls)
mlir_tablegen(WaveASMEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRWaveASMTypesIncGen)

#-------------------------------------------------------------------------------
# Attributes TableGen
#-------------------------------------------------------------------------------

set(LLVM_TARGET_DEFINITIONS WaveASMAttrs.td)
mlir_tablegen(WaveASMAttrs.h.inc -gen-attrdef-decls)
mlir_tablegen(WaveASMAttrs.cpp.inc -gen-attrdef-defs)
mlir_tablegen(WaveASMAttrEnums.h.inc -gen-enum-decls)
mlir_tablegen(WaveASMAttrEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRWaveASMAttrsIncGen)

#-------------------------------------------------------------------------------
# Interfaces TableGen
#-------------------------------------------------------------------------------

set(LLVM_TARGET_DEFINITIONS WaveASMInterfaces.td)
mlir_tablegen(WaveASMAttrInterfaces.h.inc -gen-attr-interface-decls)
mlir_tablegen(WaveASMAttrInterfaces.cpp.inc -gen-attr-interface-defs)
mlir_tablegen(WaveASMOpInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(WaveASMOpInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(MLIRWaveASMInterfacesIncGen)

#-------------------------------------------------------------------------------
# Operations TableGen
#-------------------------------------------------------------------------------

set(LLVM_TARGET_DEFINITIONS WaveASMOps.td)
mlir_tablegen(WaveASMOps.h.inc -gen-op-decls)
mlir_tablegen(WaveASMOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRWaveASMOpsIncGen)

#-------------------------------------------------------------------------------
# Combined Target
#-------------------------------------------------------------------------------

add_custom_target(MLIRWaveASMIncGen DEPENDS
  MLIRWaveASMDialectIncGen
  MLIRWaveASMTypesIncGen
  MLIRWaveASMAttrsIncGen
  MLIRWaveASMOpsIncGen
)
