// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef WaveASM_DIALECT_WAVEASMDIALECT
#define WaveASM_DIALECT_WAVEASMDIALECT

include "mlir/IR/DialectBase.td"

def WaveASMDialect : Dialect {
  let name = "waveasm";
  let summary = "Pure SSA dialect for AMDGCN assembly generation";

  let description = [{
    The WAVEASM dialect provides a pure SSA representation for GPU kernels
    with typed operations for each AMDGCN instruction. It enables whole-program
    register allocation and direct assembly emission.

    ## Design Philosophy

    Each GPU instruction is represented as its own MLIR operation (not a string).
    This provides:
    - Type safety per instruction
    - Implicit def-use chains from the SSA graph
    - Standard MLIR optimizations (DCE, CSE) work out of the box
    - IDE autocomplete and validation

    ## Two-Phase IR

    1. **Virtual IR (Pre-Allocation)**: Uses virtual registers (`!waveasm.vreg`,
       `!waveasm.sreg`) as SSA values. Each instruction produces results that
       flow through the graph.

    2. **Physical IR (Post-Allocation)**: After linear scan register allocation,
       virtual registers are replaced with physical registers (`!waveasm.pvreg`,
       `!waveasm.psreg`).

    ## Tied Operands (In-Place Updates)

    For instructions like MFMA where the accumulator is both read and written,
    the `TiedResult` trait indicates that the result must be allocated to the
    same physical register as a specific operand. This is invisible in the IR
    but enforced during register allocation.

    Example:
    ```mlir
    // The accumulator is tied - %acc_out will be same physical reg as %acc_in
    %acc_out = waveasm.v_mfma_f32_16x16x16_f16 %a, %b, %acc_in
        : !waveasm.vreg<2>, !waveasm.vreg<2>, !waveasm.vreg<4> -> !waveasm.vreg<4>
    ```

    ## Supported Targets

    - gfx942: AMD CDNA3 (MI300 series)
    - gfx950: AMD CDNA3+ (future MI series)
    - gfx1250: AMD RDNA4
  }];

  let cppNamespace = "::waveasm";

  let useDefaultAttributePrinterParser = 1;
  let useDefaultTypePrinterParser = 1;

  let extraClassDeclaration = [{
    // Name of the target attribute on kernel programs
    static constexpr ::llvm::StringLiteral kTargetAttrName = "waveasm.target";

    // Name of the ABI attribute on kernel programs
    static constexpr ::llvm::StringLiteral kABIAttrName = "waveasm.abi";

    /// Original kernel name before symbol mangling.
    static ::llvm::StringRef getKernelNameAttrName() {
      return "kernel_name";
    }

    // Registration functions
    void registerAttributes();
    void registerTypes();
  }];
}

#endif // WaveASM_DIALECT_WAVEASMDIALECT
