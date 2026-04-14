# Copyright 2025 The Wave Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
End-to-end testing utilities for C++ WaveASM backend.

This module provides utilities to:
1. Capture MLIR IR from Wave kernels
2. Compile MLIR using the C++ waveasm-translate tool
3. Assemble to GPU binary using clang++
4. Execute and validate against PyTorch

Usage:
    from waveasm_e2e import WaveASMCompiler, compile_and_run

    # Capture MLIR from wave kernel
    mlir_text = capture_wave_mlir(options, kernel_func)

    # Compile using C++ backend
    compiler = WaveASMCompiler(target="gfx942")
    asm_text = compiler.compile_mlir_to_asm(mlir_text)
    binary_path = compiler.assemble_to_binary(asm_text)

    # Execute
    run_kernel(binary_path, inputs, outputs, options)
"""

import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, List, Tuple

import torch

from wave_lang.kernel.wave.utils.mlir_analysis import (
    extract_func_from_stream_mlir,
)
from wave_lang.kernel.wave.utils.run_utils import get_dynamic_stride_args
from wave_lang.kernel.wave.utils.classes import Failure, Result, Success


@dataclass
class CompilationResult:
    """Result of compilation through C++ WaveASM backend."""

    mlir_text: str
    asm_text: str
    binary_path: Optional[Path]
    success: bool
    error_message: Optional[str] = None

    def get_kernel_name(self) -> Optional[str]:
        """Extract kernel function name from generated assembly."""
        import re

        # Look for .globl directive which marks the kernel entry point
        match = re.search(r"\.globl\s+(\w+)", self.asm_text)
        if match:
            return match.group(1)
        return None

    def get_lds_size(self) -> int:
        """Extract LDS (shared memory) size from kernel descriptor."""
        import re

        # Look for .amdhsa_group_segment_fixed_size directive
        match = re.search(r"\.amdhsa_group_segment_fixed_size\s+(\d+)", self.asm_text)
        if match:
            return int(match.group(1))
        return 0

    def get_max_flat_workgroup_size(self) -> int:
        """Extract max flat workgroup size from YAML metadata."""
        import re

        # Look for .max_flat_workgroup_size in YAML metadata
        match = re.search(r"\.max_flat_workgroup_size:\s+(\d+)", self.asm_text)
        if match:
            return int(match.group(1))
        return 64  # Default to 1 wave


def get_waveasm_translate_path() -> Path:
    """Get path to waveasm-translate executable."""
    # Check environment variable first.
    if "WAVEASM_TRANSLATE" in os.environ:
        return Path(os.environ["WAVEASM_TRANSLATE"])

    # Try detect_waveasm (pip-installed binary).
    try:
        from wave_lang.support.detect_waveasm import get_waveasm_translate

        return Path(get_waveasm_translate())
    except (ImportError, RuntimeError):
        pass

    # Fallback: look in waveasm build directory.
    script_dir = Path(__file__).parent
    default_path = (
        script_dir.parent.parent
        / "build"
        / "tools"
        / "waveasm-translate"
        / "waveasm-translate"
    )
    if default_path.exists():
        return default_path

    # Try to find it in PATH.
    try:
        result = subprocess.run(
            ["which", "waveasm-translate"], capture_output=True, text=True
        )
        if result.returncode == 0:
            return Path(result.stdout.strip())
    except Exception:
        pass

    raise FileNotFoundError(
        "waveasm-translate not found. Set WAVEASM_TRANSLATE environment variable "
        "or build waveasm project."
    )


def get_clang_path() -> str:
    """Get path to clang++ for assembly compilation."""
    try:
        from wave_lang.support.detect_waveasm import get_clang

        return get_clang()
    except (ImportError, RuntimeError):
        pass

    # Fallback: look in PATH.
    try:
        result = subprocess.run(["which", "clang++"], capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass

    raise FileNotFoundError(
        "clang++ not found. Build with WAVE_BUILD_WAVEASM=1 or ensure clang++ is in PATH."
    )


class WaveASMCompiler:
    """Compiler for MLIR -> Assembly using C++ WaveASM backend."""

    def __init__(
        self,
        target: str = "gfx942",
        codeobj: str = "5",
        keep_temp_files: bool = False,
    ):
        self.target = target
        self.codeobj = codeobj
        self.keep_temp_files = keep_temp_files
        self.waveasm_translate = get_waveasm_translate_path()
        self.clang = get_clang_path()
        self._temp_dir: Optional[Path] = None

    def _get_temp_dir(self) -> Path:
        """Get or create temp directory for intermediate files."""
        if self._temp_dir is None:
            self._temp_dir = Path(tempfile.mkdtemp(prefix="waveasm_e2e_"))
        return self._temp_dir

    def compile_mlir_to_asm(
        self,
        mlir_text: str,
        workgroup_size: Optional[Tuple[int, int, int]] = None,
        ticketed_waitcnt: bool = True,
    ) -> Tuple[bool, str, str]:
        """
        Compile MLIR to AMDGCN assembly using C++ waveasm-translate.

        Args:
            mlir_text: MLIR module text
            workgroup_size: Optional workgroup size tuple (x, y, z)
            ticketed_waitcnt: Enable ticket-based waitcnt/barrier insertion

        Returns:
            Tuple of (success, asm_text_or_error, stderr)
        """
        temp_dir = self._get_temp_dir()
        mlir_file = temp_dir / "input.mlir"
        asm_file = temp_dir / "output.s"

        # Write MLIR to temp file
        mlir_file.write_text(mlir_text)

        # Run waveasm-translate with full pipeline.
        ticketed = "true" if ticketed_waitcnt else "false"
        cmd = [
            str(self.waveasm_translate),
            f"--target={self.target}",
            "--mlir-cse",  # Pre-translation MLIR CSE for redundant index elimination.
            "--waveasm-scoped-cse",  # Scoped CSE.
            "--waveasm-peephole",  # Peephole optimizations (fuse lshl+add, etc.).
            "--waveasm-scale-pack-elimination",
            "--loop-invariant-code-motion",  # Hoist loop-invariant VALU address ops.
            "--waveasm-m0-redundancy-elim",  # Eliminate redundant M0 writes.
            "--waveasm-buffer-load-strength-reduction",
            "--waveasm-memory-offset-opt",  # Fold constant addresses into offset fields.
            "--canonicalize",  # Clean up dead instructions from offset opt.
            "--waveasm-scoped-cse",  # Re-deduplicate after offset folding.
            "--waveasm-loop-address-promotion",
            "--waveasm-linear-scan=max-vgprs=256 max-agprs=256",  # Register allocation.
            f"--waveasm-insert-waitcnt=ticketed-waitcnt={ticketed}",  # Insert waits.
            f"--waveasm-hazard-mitigation=target={self.target}",  # Handle hazards.
            "--emit-assembly",
        ]

        # Add workgroup size if specified
        if workgroup_size:
            cmd.extend(
                [
                    f"--workgroup-size-x={workgroup_size[0]}",
                    f"--workgroup-size-y={workgroup_size[1]}",
                    f"--workgroup-size-z={workgroup_size[2]}",
                ]
            )

        cmd.append(str(mlir_file))

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.returncode != 0:
                return False, result.stderr, result.stderr

            # The output is printed to stdout
            asm_text = result.stdout

            # Also save to file
            asm_file.write_text(asm_text)

            return True, asm_text, result.stderr

        except subprocess.TimeoutExpired:
            return False, "Compilation timed out", ""
        except Exception as e:
            return False, str(e), ""

    def assemble_to_binary(self, asm_text: str) -> Result[Path]:
        """Assemble AMDGCN assembly to GPU binary using clang++."""
        temp_dir = self._get_temp_dir()
        asm_file = temp_dir / "kernel.s"
        obj_file = temp_dir / "kernel.o"
        hsaco_file = temp_dir / "kernel.hsaco"

        # Write assembly to file
        asm_file.write_text(asm_text)

        # Step 1: Assemble to object file
        compile_cmd = [
            self.clang,
            "-x",
            "assembler",
            "-target",
            "amdgcn-amd-amdhsa",
            f"-mcode-object-version={self.codeobj}",
            f"-mcpu={self.target}",
            "-mwavefrontsize64",
            "-c",
            str(asm_file),
            "-o",
            str(obj_file),
        ]

        try:
            result = subprocess.run(
                compile_cmd,
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.returncode != 0:
                return Failure(f"Assembly failed: {result.stderr}")

            # Step 2: Link to HSACO
            link_cmd = [
                self.clang,
                "-target",
                "amdgcn-amd-amdhsa",
                "-Xlinker",
                "--build-id=sha1",
                "-o",
                str(hsaco_file),
                str(obj_file),
            ]

            result = subprocess.run(
                link_cmd,
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.returncode != 0:
                return Failure(f"Linking failed: {result.stderr}")

            return Success(hsaco_file)

        except subprocess.TimeoutExpired:
            return Failure("Assembly/linking timed out")
        except Exception as e:
            return Failure(str(e))

    def compile_full(
        self,
        mlir_text: str,
        workgroup_size: Optional[Tuple[int, int, int]] = None,
    ) -> CompilationResult:
        """
        Full compilation pipeline: MLIR -> ASM -> Binary.

        Args:
            mlir_text: MLIR module text
            workgroup_size: Optional workgroup size tuple (x, y, z)

        Returns:
            CompilationResult with all intermediate results
        """
        # Step 1: MLIR -> ASM
        success, asm_or_error, stderr = self.compile_mlir_to_asm(
            mlir_text, workgroup_size
        )

        if not success:
            return CompilationResult(
                mlir_text=mlir_text,
                asm_text="",
                binary_path=None,
                success=False,
                error_message=f"MLIR->ASM failed: {asm_or_error}",
            )

        asm_text = asm_or_error

        # Step 2: ASM -> Binary
        asm_result = self.assemble_to_binary(asm_text)

        if not asm_result:
            return CompilationResult(
                mlir_text=mlir_text,
                asm_text=asm_text,
                binary_path=None,
                success=False,
                error_message=f"ASM->Binary failed: {asm_result.error}",
            )

        assert isinstance(asm_result, Success)
        return CompilationResult(
            mlir_text=mlir_text,
            asm_text=asm_text,
            binary_path=asm_result.value,
            success=True,
        )

    def cleanup(self):
        """Clean up temporary files."""
        if not self.keep_temp_files and self._temp_dir is not None:
            import shutil

            try:
                shutil.rmtree(self._temp_dir)
            except Exception:
                pass
            self._temp_dir = None


@dataclass
class CapturedKernelInfo:
    """Information captured from Wave compilation for kernel launch."""

    mlir_text: str
    kernel_name: str
    workgroup_size: Tuple[int, int, int]
    grid_size: Tuple[int, int, int]
    lds_size: int = 0


def capture_wave_mlir(options, kernel_func) -> str:
    """
    Capture the MLIR IR from a Wave kernel before ASM codegen.

    This runs the Wave compilation pipeline up to MLIR emission,
    then extracts the func.func from the stream.executable wrapper.

    Args:
        options: WaveCompileOptions
        kernel_func: Decorated wave kernel function

    Returns:
        MLIR module text with just the kernel func.func
    """
    from wave_lang.kernel._support.indexing import IndexingContext
    from wave_lang.kernel.wave.compile import (
        _trace_launchable_and_get_kernel_signature,
    )

    # Must run within IndexingContext like wave_compile does
    with IndexingContext() as idxc:
        idxc.set_subs(options.subs)

        # Initialize kernel constraints (same as wave_compile)
        kernel_func.initialize_wave_constraints()
        kernel_func.initialize_symbolic_constraints()
        kernel_func.initialize_workgroup_constraints()

        # Trace and get MLIR
        result = _trace_launchable_and_get_kernel_signature(kernel_func, options)
        # Result is a tuple: (mb, trace, exe, kernel_sig, entrypoint_name, options, ...)
        mb = result[0]

        # Get full MLIR text
        full_mlir_text = mb.module_op.get_asm(
            enable_debug_info=False,
            use_local_scope=options.use_local_scope,
        )

    # Extract just the func.func from stream.executable wrapper
    # This avoids duplicate symbol issues and stream dialect parsing problems
    mlir_text = extract_func_from_stream_mlir(full_mlir_text)

    return mlir_text


def capture_wave_kernel_info(
    options,
    kernel_func,
    schedule=None,
    dynamic_values: Optional[Dict] = None,
) -> CapturedKernelInfo:
    """
    Capture MLIR and kernel launch info from Wave compilation.

    This extracts all information needed for kernel launch by using
    options.kernel_launch_info which is populated by _trace_launchable_and_get_kernel_signature.

    Args:
        options: WaveCompileOptions
        kernel_func: Decorated wave kernel function
        schedule: Optional WaveSchedule to apply during compilation
        dynamic_values: Optional dict mapping dynamic symbols to their concrete
            values. Used for grid computation when symbols are not in
            options.subs (i.e. truly dynamic shapes).

    Returns:
        CapturedKernelInfo with all launch information
    """
    from wave_lang.kernel._support.indexing import IndexingContext
    from wave_lang.kernel.wave.compile import (
        _trace_launchable_and_get_kernel_signature,
    )
    from wave_lang.support.ir_imports import Context, Module, func_d
    from wave_lang.kernel.wave.utils.mlir_analysis import (
        walk_ops_recursively,
        should_skip_function,
    )

    # Must run within IndexingContext like wave_compile does
    with IndexingContext() as idxc:
        idxc.set_subs(options.subs)

        # Initialize kernel constraints (same as wave_compile)
        kernel_func.initialize_wave_constraints()
        kernel_func.initialize_symbolic_constraints()
        kernel_func.initialize_workgroup_constraints()

        # Trace and get MLIR - this populates options.kernel_launch_info
        result = _trace_launchable_and_get_kernel_signature(
            kernel_func, options, schedule=schedule
        )
        mb = result[0]

        # Apply opsel optimization: replace scalar extract+bitcast scale
        # chains on scaled_mfma ops with vector-level bitcast and opsel.
        # This must run before MLIR is handed to the C++ backend, matching
        # what compile.py does at line 823.
        from wave_lang.kernel.wave.opsel_scaled_mfma import (
            apply_opsel_scaled_mfma,
        )
        from wave_lang.kernel.wave.utils.compile_utils import (
            canonicalize_module,
        )

        apply_opsel_scaled_mfma(mb.module_op)
        if options.canonicalize:
            canonicalize_module(mb.module_op)

        # Get full MLIR text (Python bindings can parse stream dialect)
        full_mlir_text = mb.module_op.get_asm(
            enable_debug_info=False,
            use_local_scope=options.use_local_scope,
        )

        # Get launch info directly from options - populated by _trace_launchable
        launch_info = options.kernel_launch_info
        blocks = launch_info.blocks if launch_info.blocks else [64, 1, 1]
        lds_size = launch_info.shared_memory_bytes
        kernel_name = launch_info.func_name

        # Compute grid dimensions - launch_info.grid is a sympy.lambdify lambda.
        # It expects arguments in the same order they were defined in compile.py:
        #   grid_symbols = bound_scalar_symbols.keys() + dynamic_symbols
        # and is called with a single list: grid_fn(list_of_values).
        #
        # sympy.lambdify may reference `math.floor`/`math.ceil` in the generated
        # lambda body but the math module may be missing from its __globals__
        # when invoked outside the lambdify call-site.  Inject it defensively.
        import math

        launch_info.grid.__globals__.setdefault("math", math)
        dynamic_syms = list(getattr(options, "dynamic_symbols", None) or [])
        grid_symbols = list(kernel_func.bound_scalar_symbols.keys()) + dynamic_syms
        grid_values = []
        dv = dynamic_values or {}
        for sym in grid_symbols:
            if sym in options.subs:
                grid_values.append(options.subs[sym])
            elif sym in dv:
                grid_values.append(dv[sym])
            else:
                raise ValueError(
                    f"Grid symbol {sym} not found in options.subs or "
                    f"dynamic_values. "
                    f"Available subs: {list(options.subs.keys())}, "
                    f"dynamic_values: {list(dv.keys())}"
                )
        grid = launch_info.grid(grid_values)
        grid = tuple(int(x) for x in grid)

    # Extract func.func from stream wrapper for C++ backend
    extracted_funcs = []

    with Context() as ctx:
        ctx.allow_unregistered_dialects = True
        module = Module.parse(full_mlir_text)

        for fn in walk_ops_recursively(module.operation):
            if not isinstance(fn, func_d.FuncOp):
                continue

            # Skip non-kernel functions
            if should_skip_function(fn):
                continue

            # Extract the function in generic form for C++ backend
            extracted_funcs.append(fn.get_asm(print_generic_op_form=True))
            break  # Found the kernel

    if not extracted_funcs:
        raise ValueError("No kernel function found in MLIR")

    # Wrap extracted function in a module for C++ backend
    mlir_text = "module {\n" + "\n".join(extracted_funcs) + "\n}\n"

    info = CapturedKernelInfo(
        mlir_text=mlir_text,
        kernel_name=kernel_name,
        workgroup_size=tuple(blocks),
        grid_size=grid,
        lds_size=lds_size,
    )

    # Debug output
    import os

    if os.environ.get("WAVEASM_DEBUG"):
        print(f"\n=== Captured Kernel Info ===")
        print(f"  kernel_name: {info.kernel_name}")
        print(f"  workgroup_size: {info.workgroup_size}")
        print(f"  grid_size: {info.grid_size}")
        print(f"  lds_size: {info.lds_size}")
        print(f"============================\n")

    return info


def run_with_wave_runtime(
    binary_path: Path,
    inputs: List[torch.Tensor],
    outputs: List[torch.Tensor],
    grid: Tuple[int, int, int],
    block: Tuple[int, int, int],
    shared_memory_bytes: int = 0,
    func_name: str = "isolated_benchmark",
    dynamic_dims: Optional[List[int]] = None,
):
    """
    Execute a compiled GPU binary using wave_runtime.

    Args:
        binary_path: Path to .hsaco file
        inputs: List of input tensors
        outputs: List of output tensors
        grid: Grid dimensions (x, y, z)
        block: Block dimensions (x, y, z)
        shared_memory_bytes: Shared memory size
        func_name: Function name in the binary (default: "isolated_benchmark")
        dynamic_dims: Optional list of concrete values for dynamic dimension
            symbols, passed as additional kernel arguments.
    """
    import wave_runtime  # type: ignore[import-not-found]

    # Initialize HIP functions
    wave_runtime.load_hip_functions()

    # Load GPU function from binary
    gpu_binary, gpu_func = wave_runtime.load_binary(str(binary_path), func_name)

    # Create launch info (using positional args - the API expects 12 int args)
    stream = torch.cuda.current_stream().cuda_stream
    kernel_launch_info = wave_runtime.KernelLaunchInfo(
        stream,  # arg0: stream
        gpu_func,  # arg1: gpu_func
        shared_memory_bytes,  # arg2: shared_memory_bytes
        grid[0],  # arg3: grid_dim_x
        grid[1],  # arg4: grid_dim_y
        grid[2],  # arg5: grid_dim_z
        block[0],  # arg6: block_dim_x
        block[1],  # arg7: block_dim_y
        block[2],  # arg8: block_dim_z
        1,  # arg9: cluster_dim_x
        1,  # arg10: cluster_dim_y
        1,  # arg11: cluster_dim_z
    )

    # Prepare kernel arguments
    all_tensors = inputs + outputs
    kern_args = [tensor.data_ptr() for tensor in all_tensors]
    kernel_args = wave_runtime.Int64Vector(kern_args)

    dyn_dims = wave_runtime.Int64Vector(dynamic_dims or [])

    # Prepare dynamic stride arguments
    stride_args = get_dynamic_stride_args(all_tensors)

    # Launch
    wave_runtime.launch(kernel_launch_info, kernel_args, dyn_dims, [], stride_args)

    # Sync
    torch.cuda.synchronize()
