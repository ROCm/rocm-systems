# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""
Mega Kernel Test Suite

This test suite uses hipRTC to compile and run the comprehensive mega_kernel
that exercises most GPU instructions and features across different AMD GPU
architectures (CDNA >= MI200, RDNA 3.5 Strix/Strix Halo gfx1150/gfx1151,
and RDNA4 RX 9070 XT).

Not registered in CTest (see CMakeqLists.txt under "Utils tests"): run manually, e.g.
``pytest tests/test_mega_kernel.py``, when HIP/hipRTC and a matching GPU are available.

"""

import os
import sys
from ctypes import (
    Structure,
    byref,
    c_double,
    c_float,
    c_int,
    c_ulonglong,
    c_void_p,
    cast,
    sizeof,
)
from pathlib import Path

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import hip.hip as hip
import hip.hiprtc as hiprtc


# Test result structure matching the kernel's TestResults struct
class TestResults(Structure):
    _fields_ = [
        # Warp operations
        ("warp_shuffle_passed", c_int),
        ("warp_ballot_passed", c_int),
        ("warp_permute_passed", c_int),
        ("warp_reduce_passed", c_int),
        # Floating point conversions
        ("fp8_convert_passed", c_int),
        ("bf8_convert_passed", c_int),
        ("fp16_convert_passed", c_int),
        ("bf16_convert_passed", c_int),
        # Arithmetic operations
        ("fp32_arith_passed", c_int),
        ("fp64_arith_passed", c_int),
        ("int32_arith_passed", c_int),
        ("int64_arith_passed", c_int),
        # Packed operations
        ("pk_f16_passed", c_int),
        ("pk_bf16_passed", c_int),
        ("pk_add_passed", c_int),
        # Atomic operations
        ("atomic_add_f32_passed", c_int),
        ("atomic_add_f64_passed", c_int),
        ("atomic_min_f32_passed", c_int),
        ("atomic_max_f32_passed", c_int),
        ("atomic_int_passed", c_int),
        ("lds_atomic_passed", c_int),
        # Transcendental
        ("trans_sin_passed", c_int),
        ("trans_cos_passed", c_int),
        ("trans_exp_passed", c_int),
        ("trans_log_passed", c_int),
        ("trans_sqrt_passed", c_int),
        ("trans_rcp_passed", c_int),
        ("trans_rsq_passed", c_int),
        # Memory operations
        ("global_load_passed", c_int),
        ("global_store_passed", c_int),
        ("lds_load_passed", c_int),
        ("lds_store_passed", c_int),
        # DOT products
        ("dot4_passed", c_int),
        ("dot8_passed", c_int),
        # MFMA operations
        ("mfma_f32_passed", c_int),
        ("mfma_f64_passed", c_int),
        ("mfma_f16_passed", c_int),
        ("mfma_bf16_passed", c_int),
        ("mfma_i8_passed", c_int),
        ("mfma_f8_passed", c_int),
        # Dual-issue VALU
        ("dual_issue_passed", c_int),
        # Lane/thread info
        ("lane_id_passed", c_int),
        ("wavefront_passed", c_int),
        # Async LDS operations
        ("async_lds_load_passed", c_int),
        ("async_lds_store_passed", c_int),
        # TDM operations
        ("tdm_load_passed", c_int),
        ("tdm_store_passed", c_int),
        # Cluster group operations
        ("cluster_barrier_passed", c_int),
        ("cluster_info_passed", c_int),
        # WMMA operations
        ("wmma_f16_passed", c_int),
        ("wmma_bf16_passed", c_int),
        ("wmma_i8_passed", c_int),
        # Cooperative Atomic operations
        ("coop_atomic_load_passed", c_int),
        ("coop_atomic_store_passed", c_int),
        # VMEM operations
        ("vmem_flat_passed", c_int),
        ("vmem_global_passed", c_int),
        ("vmem_buffer_passed", c_int),
        ("vmem_scratch_passed", c_int),
        ("vmem_lds_passed", c_int),
        ("vmem_tex_load_passed", c_int),
        ("vmem_tex_store_passed", c_int),
        # Atomic operations (inline ASM)
        ("atomic_global_int_passed", c_int),
        ("atomic_global_f32_passed", c_int),
        ("atomic_global_f64_passed", c_int),
        ("atomic_flat_int_passed", c_int),
        ("atomic_ds_int_passed", c_int),
        ("atomic_ds_f32_passed", c_int),
        ("atomic_ds_f64_passed", c_int),
        ("atomic_pk_f16_passed", c_int),
        ("atomic_pk_bf16_passed", c_int),
        ("atomic_cas_passed", c_int),
        # Totals
        ("total_passed", c_int),
        ("total_failed", c_int),
    ]


def load_kernel_source(kernel_path: str) -> str:
    """Load the mega_kernel.hip source code"""
    with open(kernel_path, "r") as f:
        return f.read()


def _is_strix_rdna35(arch: str) -> bool:
    """gfx1150/gfx1151/gfx1152 — same gate as mega_kernel/main.cpp g_arch_type == 7."""
    base = arch.split(":", 1)[0]
    return base in ("gfx1150", "gfx1151", "gfx1152")


def compile_mega_kernel(kernel_source: str, arch: str) -> tuple:
    """
    Compile the mega kernel using hipRTC for the target architecture

    Returns:
        tuple: (module, kernel_function)
    """
    print(f"Compiling mega_kernel for {arch}...")

    # Create hipRTC program
    prog = hiprtc.hiprtcCreateProgram(kernel_source, "mega_kernel.hip")

    # Add kernel name expression for name mangling
    kernel_name = "gpu_mega_kernel"
    hiprtc.hiprtcAddNameExpression(prog, kernel_name)

    # Prepare compile options with HIP include paths

    rocm_path = os.getenv("ROCM_PATH", "/opt/rocm")
    compile_options = [
        f"-I{rocm_path}/include",
        f"-I{rocm_path}/include/hip",
        f"--offload-arch={arch}",
    ]

    # Convert to ctypes array
    from ctypes import c_char_p

    options_array = (c_char_p * len(compile_options))()
    for i, opt in enumerate(compile_options):
        options_array[i] = opt.encode("utf-8")

    try:
        # Compile the program with options

        res = hiprtc._lib.hiprtcCompileProgram(
            prog.handle, len(compile_options), options_array
        )
        # Check compilation status and raise error on failure
        if res != 0:
            log = hiprtc.hiprtcGetProgramLog(prog)
            print(f"Compilation failed for {arch}:")
            print(f"Program log:\n{log}")
            raise hiprtc.HIPRTCError(res)
    except hiprtc.HIPRTCError:
        # Re-raise if already a HIPRTCError (from the check above or elsewhere)
        raise

    # Always print compilation log for debugging
    log = hiprtc.hiprtcGetProgramLog(prog)
    if log and log.strip():
        print(f"Compilation log for {arch}:")
        print(log)

    # Get compiled code
    code = hiprtc.hiprtcGetCode(prog)

    # Load module
    module = hip.hipModuleLoadData(code)

    # Get kernel function
    func = hip.hipModuleGetFunction(module, kernel_name)

    print(f"✓ Compilation successful for {arch}")

    return module, func


def run_mega_kernel_test(
    device_id: int = 0,
    batch_size: int = 1024,
    block_size: int = 256,
    mfma_mode: int = 0,
) -> TestResults:
    """
    Run the mega kernel test on the specified device

    Args:
        device_id: GPU device ID to test
        batch_size: Number of elements to process
        block_size: Thread block size
        mfma_mode: MFMA test mode (0=both, 1=asm, 2=builtin)

    Returns:
        TestResults structure with test results
    """
    # Check if any HIP devices are available
    device_count = hip.hipGetDeviceCount()
    if device_count == 0:
        # Try to use pytest.skip if available (when running under pytest)
        try:
            import pytest

            pytest.skip("No HIP devices available - skipping mega kernel test")
        except ImportError:
            # Not running under pytest - raise RuntimeError instead
            raise RuntimeError("No HIP devices available - cannot run mega kernel test")

    # Validate device_id is within range
    if device_id >= device_count:
        try:
            import pytest

            pytest.skip(
                f"Device {device_id} not available "
                f"(only {device_count} device(s) found)"
            )
        except ImportError:
            raise RuntimeError(
                f"Device {device_id} not available "
                f"(only {device_count} device(s) found)"
            )

    # Set device
    hip.hipSetDevice(device_id)

    # Get device properties
    props = hip.hipGetDeviceProperties(device_id)
    arch = props.gcnArchName.split(":", 1)[0]

    print("=" * 80)
    print("AMD GPU MEGA KERNEL UNIT TEST")
    print("=" * 80)
    print("Device Information:")
    print(f"  Name:                  {props.name}")
    print(f"  GCN Architecture:      {arch}")
    print(f"  Compute Units:         {props.multiProcessorCount}")
    print(f"  Warp Size:             {props.warpSize}")
    print(f"  Max Threads/Block:     {props.maxThreadsPerBlock}")
    print(f"  Total Global Memory:   {props.totalGlobalMem / (1024**3):.2f} GB")
    print("=" * 80)

    # Load kernel source
    kernel_path = Path(__file__).parent / "mega_kernel" / "mega_kernel.hip"
    if not kernel_path.exists():
        raise FileNotFoundError(f"Kernel source not found: {kernel_path}")

    kernel_source = load_kernel_source(str(kernel_path))

    # Calculate grid dimensions
    grid_size = (batch_size + block_size - 1) // block_size

    # Calculate dynamic shared memory size for test_lds_operations
    # Layout: [float array][int array][double array][atomic counter]
    # - float array: block_size * sizeof(float)
    # - int array: block_size * sizeof(int)
    # - double array: (block_size / 2) * sizeof(double)
    # - atomic counter: sizeof(int) for LDS atomic test
    shared_mem_size = (
        block_size * sizeof(c_float)  # lds_float
        + block_size * sizeof(c_int)  # lds_int
        + (block_size // 2) * sizeof(c_double)  # lds_double
        + sizeof(c_int)  # lds_atomic_counter for LDS atomic test
    )

    mfma_mode_names = ["both", "asm", "builtin"]
    if not isinstance(mfma_mode, int) or not (0 <= mfma_mode < len(mfma_mode_names)):
        raise ValueError(
            f"Invalid mfma_mode {mfma_mode!r}; expected one of 0, 1, or 2 "
            f"corresponding to {mfma_mode_names}."
        )

    module = None
    device_buffers: list[hip.HIPDeviceMemory] = []

    def _device_malloc(size: int) -> hip.HIPDeviceMemory:
        mem = hip.hipMalloc(size)
        device_buffers.append(mem)
        return mem

    tex_id = 0
    surf_id = 0

    try:
        # Compile kernel
        module, func = compile_mega_kernel(kernel_source, arch)

        # Allocate device memory
        print(f"\nAllocating device memory (batch_size={batch_size})...")

        # Results structure
        d_results = _device_malloc(sizeof(TestResults))

        # Global memory buffers for atomic tests
        # Kernel uses global_int[tid % 64 + offset]; offset up to 192
        # Maximum index: 64 + 192 = 256
        GLOBAL_INT_SIZE = 256
        GLOBAL_FLOAT_SIZE = 256
        GLOBAL_DOUBLE_SIZE = 64

        d_global_float = _device_malloc(GLOBAL_FLOAT_SIZE * sizeof(c_float))
        d_global_double = _device_malloc(GLOBAL_DOUBLE_SIZE * sizeof(c_double))
        d_global_int = _device_malloc(GLOBAL_INT_SIZE * sizeof(c_int))

        # Input/output buffers
        d_input_buffer = _device_malloc(batch_size * sizeof(c_float))
        d_output_buffer = _device_malloc(batch_size * sizeof(c_float))

        # Async LDS test buffers
        d_async_lds_src = _device_malloc(batch_size * sizeof(c_float))
        d_async_lds_dst = _device_malloc(batch_size * sizeof(c_float))

        # TDM test buffers
        d_tdm_src = _device_malloc(batch_size * sizeof(c_int))
        d_tdm_dst = _device_malloc(batch_size * sizeof(c_int))

        print(f"Launching kernel: grid={grid_size}, block={block_size}")
        print(
            f"Shared memory size: {shared_mem_size} bytes ({shared_mem_size / 1024:.2f} KB)"
        )
        print(f"MFMA mode: {mfma_mode_names[mfma_mode]}")

        # Init device buffers before launch (cross-block race; kernel skips memset)
        h_results_zero = TestResults()
        hip.hipMemcpyHtoD(d_results, byref(h_results_zero), sizeof(TestResults))
        zeros_f = (c_float * GLOBAL_FLOAT_SIZE)()
        hip.hipMemcpyHtoD(
            d_global_float, byref(zeros_f), GLOBAL_FLOAT_SIZE * sizeof(c_float)
        )
        zeros_d = (c_double * GLOBAL_DOUBLE_SIZE)()
        hip.hipMemcpyHtoD(
            d_global_double, byref(zeros_d), GLOBAL_DOUBLE_SIZE * sizeof(c_double)
        )
        zeros_i = (c_int * GLOBAL_INT_SIZE)()
        hip.hipMemcpyHtoD(d_global_int, byref(zeros_i), GLOBAL_INT_SIZE * sizeof(c_int))
        # Input buffer: same as main.cpp (float)i for memory verification
        h_input = (c_float * batch_size)(*[float(i) for i in range(batch_size)])
        hip.hipMemcpyHtoD(d_input_buffer, byref(h_input), batch_size * sizeof(c_float))
        zeros_out = (c_float * batch_size)()
        hip.hipMemcpyHtoD(
            d_output_buffer, byref(zeros_out), batch_size * sizeof(c_float)
        )
        hip.hipMemcpyHtoD(
            d_async_lds_src, byref(zeros_out), batch_size * sizeof(c_float)
        )
        hip.hipMemcpyHtoD(
            d_async_lds_dst, byref(zeros_out), batch_size * sizeof(c_float)
        )
        zeros_tdm = (c_int * batch_size)()
        hip.hipMemcpyHtoD(d_tdm_src, byref(zeros_tdm), batch_size * sizeof(c_int))
        hip.hipMemcpyHtoD(d_tdm_dst, byref(zeros_tdm), batch_size * sizeof(c_int))

        # Strix (gfx1150/1151/1152): mirror mega_kernel/main.cpp for tex/surf
        # so TEX load/store counters increment under rocprof.
        if _is_strix_rdna35(arch):
            lin_bytes = batch_size * sizeof(c_float)
            res_tex = hip.HIPResourceDesc()
            res_tex.resType = hip.HIP_RESOURCE_TYPE_LINEAR
            res_tex.res.linear.devPtr = d_input_buffer.ptr
            res_tex.res.linear.desc = hip.HIPChannelFormatDesc(
                32, 0, 0, 0, hip.HIP_CHANNEL_FORMAT_KIND_FLOAT
            )
            res_tex.res.linear.sizeInBytes = lin_bytes

            tex_desc = hip.HIPTextureDesc()
            tex_desc.normalizedCoords = 0
            tex_desc.filterMode = hip.HIP_FILTER_MODE_POINT
            tex_desc.addressMode[0] = hip.HIP_ADDRESS_MODE_CLAMP

            tex_id = hip.hipCreateTextureObject(res_tex, tex_desc)

            d_surf_buffer = _device_malloc(lin_bytes)
            hip.hipMemset(d_surf_buffer, 0, lin_bytes)

            res_surf = hip.HIPResourceDesc()
            res_surf.resType = hip.HIP_RESOURCE_TYPE_LINEAR
            res_surf.res.linear.devPtr = d_surf_buffer.ptr
            res_surf.res.linear.desc = hip.HIPChannelFormatDesc(
                32, 0, 0, 0, hip.HIP_CHANNEL_FORMAT_KIND_FLOAT
            )
            res_surf.res.linear.sizeInBytes = lin_bytes

            surf_id = hip.hipCreateSurfaceObject(res_surf)

        tex_obj = c_ulonglong(tex_id)
        surf_obj = c_ulonglong(surf_id)

        # Kernel args must match _mega_kernel in mega_kernel.hip
        args = [
            d_results,
            d_global_float,
            d_global_double,
            d_global_int,
            d_input_buffer,
            d_output_buffer,
            d_async_lds_src,
            d_async_lds_dst,
            c_int(batch_size),
            c_int(mfma_mode),
            tex_obj,
            surf_obj,
        ]

        # Convert arguments to void pointers
        args_converted = []
        for arg in args:
            if isinstance(arg, int):
                args_converted.append(c_int(arg))
            elif isinstance(arg, hip.HIPDeviceMemory):
                args_converted.append(arg.ptr)
            else:
                args_converted.append(arg)

        normalized = [cast(byref(arg), c_void_p) for arg in args_converted]
        args_ptr = (c_void_p * len(args))(*normalized)

        # Launch kernel
        print("\nRunning GPU Mega Kernel...")

        event_start = hip.hipEventCreate()
        event_stop = hip.hipEventCreate()

        hip.hipEventRecord(event_start)

        hip.hipModuleLaunchKernel(
            func,
            grid_size,
            1,
            1,  # grid dimensions
            block_size,
            1,
            1,  # block dimensions
            shared_mem_size,  # dynamic shared memory size
            None,  # stream
            args_ptr,
            None,  # extra
        )

        hip.hipEventRecord(event_stop)
        hip.hipDeviceSynchronize()

        # Get execution time
        exec_time_ms = hip.hipEventElapsedTime(event_start, event_stop)

        # Copy results back to host
        h_results = TestResults()
        hip.hipMemcpyDtoH(byref(h_results), d_results, sizeof(TestResults))

        print(f"✓ Kernel execution completed in {exec_time_ms:.3f} ms")

        return h_results
    finally:
        if tex_id:
            hip.hipDestroyTextureObject(tex_id)
        if surf_id:
            hip.hipDestroySurfaceObject(surf_id)
        for mem in reversed(device_buffers):
            hip.hipFree(mem)
        if module is not None:
            del module


def print_test_results(results: TestResults, arch: str):
    """Print formatted test results"""

    print("\n" + "=" * 80)
    print("TEST RESULTS")
    print("=" * 80)

    # Define test categories
    categories = [
        (
            "Warp/Wave Operations",
            [
                (
                    "Warp Shuffle (shfl/shfl_xor/shfl_up/down)",
                    results.warp_shuffle_passed,
                ),
                ("Warp Ballot (__ballot/__any/__all)", results.warp_ballot_passed),
                ("Warp Permute (ds_bpermute/ds_permute)", results.warp_permute_passed),
                ("Warp Reduce (shuffle-based reduction)", results.warp_reduce_passed),
            ],
        ),
        (
            "Data Type Conversions",
            [
                ("FP8 Conversions (cvt_pk_fp8_f32)", results.fp8_convert_passed),
                ("BF8 Conversions (cvt_pk_bf8_f32)", results.bf8_convert_passed),
                ("FP16 Operations", results.fp16_convert_passed),
                ("BF16 Operations", results.bf16_convert_passed),
            ],
        ),
        (
            "Arithmetic Operations",
            [
                ("FP32 Arithmetic (add/mul/fma)", results.fp32_arith_passed),
                ("FP64 Arithmetic (add/mul/fma)", results.fp64_arith_passed),
                ("INT32 Operations (add/mul/bit ops)", results.int32_arith_passed),
                ("INT64 Operations", results.int64_arith_passed),
            ],
        ),
        (
            "Packed Operations",
            [
                ("Packed FP16 (__hadd2/__hmul2)", results.pk_f16_passed),
                ("Packed BF16", results.pk_bf16_passed),
                ("Packed Add", results.pk_add_passed),
            ],
        ),
        (
            "Basic Atomic Operations (HIP API)",
            [
                ("Atomic Add FP32", results.atomic_add_f32_passed),
                ("Atomic Add FP64 (HW atomic on CDNA)", results.atomic_add_f64_passed),
                ("Atomic Min FP32", results.atomic_min_f32_passed),
                ("Atomic Max FP32", results.atomic_max_f32_passed),
                ("Atomic Integer Ops", results.atomic_int_passed),
                ("LDS Atomic Operations", results.lds_atomic_passed),
            ],
        ),
        (
            "Atomic Operations (Inline ASM)",
            [
                (
                    "Global Integer Atomics (add/sub/min/max)",
                    results.atomic_global_int_passed,
                ),
                ("Global FP32 Atomics (add)", results.atomic_global_f32_passed),
                ("Global FP64 Atomics (add/min/max)", results.atomic_global_f64_passed),
                ("Flat Integer Atomics", results.atomic_flat_int_passed),
                ("DS Integer Atomics (LDS)", results.atomic_ds_int_passed),
                ("DS FP32 Atomics (LDS)", results.atomic_ds_f32_passed),
                ("DS FP64 Atomics (LDS)", results.atomic_ds_f64_passed),
                ("Packed FP16 Atomics", results.atomic_pk_f16_passed),
                ("Packed BF16 Atomics", results.atomic_pk_bf16_passed),
                ("Compare-And-Swap (CAS)", results.atomic_cas_passed),
            ],
        ),
        (
            "Transcendental Functions",
            [
                ("Sine (sinf)", results.trans_sin_passed),
                ("Cosine (cosf)", results.trans_cos_passed),
                ("Exponential (expf)", results.trans_exp_passed),
                ("Logarithm (logf)", results.trans_log_passed),
                ("Square Root (sqrtf)", results.trans_sqrt_passed),
                ("Reciprocal (1.0f/x)", results.trans_rcp_passed),
                ("Reciprocal Sqrt (rsqrtf)", results.trans_rsq_passed),
            ],
        ),
        (
            "Memory Operations",
            [
                ("Global Load (coalesced)", results.global_load_passed),
                ("Global Store (coalesced)", results.global_store_passed),
                ("LDS Load", results.lds_load_passed),
                ("LDS Store", results.lds_store_passed),
            ],
        ),
        (
            "VMEM Operations (Inline ASM)",
            [
                ("Flat Memory (flat_load/store)", results.vmem_flat_passed),
                ("Global Memory (global_load/store)", results.vmem_global_passed),
                ("Buffer Memory (buffer_load/store)", results.vmem_buffer_passed),
                ("Scratch Memory (local variables)", results.vmem_scratch_passed),
                ("LDS Memory (ds_read/write)", results.vmem_lds_passed),
                (
                    "Texture Load (tex1Dfetch / INSTS_TEX_LOAD)",
                    results.vmem_tex_load_passed,
                ),
                (
                    "Texture Store (surf1Dwrite / INSTS_TEX_STORE)",
                    results.vmem_tex_store_passed,
                ),
            ],
        ),
        (
            "DOT Product Operations",
            [
                ("DOT4 (4-element INT8 dot product)", results.dot4_passed),
                ("DOT8 (8-element INT4 dot product)", results.dot8_passed),
            ],
        ),
        (
            "MFMA Operations (CDNA Matrix Cores)",
            [
                ("MFMA FP32 (32x32x2)", results.mfma_f32_passed),
                ("MFMA FP64 (16x16x4)", results.mfma_f64_passed),
                ("MFMA FP16 (32x32x8)", results.mfma_f16_passed),
                ("MFMA BF16 (32x32x4/8)", results.mfma_bf16_passed),
                ("MFMA INT8 (32x32x8/16)", results.mfma_i8_passed),
                ("MFMA FP8 (32x32x16)", results.mfma_f8_passed),
            ],
        ),
        (
            "Dual-Issue VALU (MI350/gfx950 specific)",
            [
                ("Dual-Issue Patterns (FP64+FP32 co-exec)", results.dual_issue_passed),
            ],
        ),
        (
            "Lane/Thread Information",
            [
                ("Lane ID (__builtin_amdgcn_mbcnt)", results.lane_id_passed),
                ("Wavefront Size Detection", results.wavefront_passed),
            ],
        ),
        (
            "Async LDS Operations (MI350 specific)",
            [
                (
                    "Async LDS Load (global_load_async_to_lds)",
                    results.async_lds_load_passed,
                ),
                (
                    "Async LDS Store (global_store_async_from_lds)",
                    results.async_lds_store_passed,
                ),
            ],
        ),
        (
            "TDM (Tensor Data Mover) Operations",
            [
                ("TDM Tensor Load to LDS", results.tdm_load_passed),
                ("TDM Tensor Store from LDS", results.tdm_store_passed),
            ],
        ),
        (
            "Cluster Group Operations",
            [
                (
                    "Cluster Barrier (s_barrier_signal/wait)",
                    results.cluster_barrier_passed,
                ),
                ("Cluster Info (workgroup IDs)", results.cluster_info_passed),
            ],
        ),
        (
            "WMMA Operations (RDNA4 Matrix Cores)",
            [
                ("WMMA FP16 (16x16x16)", results.wmma_f16_passed),
                ("WMMA BF16 (16x16x16)", results.wmma_bf16_passed),
                ("WMMA INT8 (16x16x16)", results.wmma_i8_passed),
            ],
        ),
        (
            "Cooperative Atomic Operations",
            [
                (
                    "Cooperative Atomic Load (32x4B/16x8B/8x16B)",
                    results.coop_atomic_load_passed,
                ),
                (
                    "Cooperative Atomic Store (32x4B/16x8B/8x16B)",
                    results.coop_atomic_store_passed,
                ),
            ],
        ),
    ]

    total_passed = 0
    total_failed = 0
    total_bypassed = 0

    for category_name, tests in categories:
        print(f"\n[Category: {category_name}]")
        for test_name, result in tests:
            if result > 0:
                status = "PASS"
                total_passed += 1
            elif result == 0:
                status = "FAIL"
                total_failed += 1
            else:  # result < 0
                status = "BYPASSED (N/A)"
                total_bypassed += 1

            print(f"  {test_name:50s}: {status:20s} ({result}/1)")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Test Categories Passed:   {total_passed}")
    print(f"Test Categories Failed:   {total_failed}")
    print(f"Test Categories Bypassed: {total_bypassed} (N/A for this architecture)")

    if total_failed == 0:
        print(
            f"\n*** OVERALL: PASSED "
            f"({total_passed}/{total_passed}, {total_bypassed} bypassed) ***"
        )
    else:
        print(
            f"\n*** OVERALL: FAILED "
            f"({total_passed}/{total_passed + total_failed}, "
            f"{total_bypassed} bypassed) ***"
        )

    return total_passed, total_failed, total_bypassed


def test_mega_kernel_gfx90a():
    """Test mega kernel on MI250 (gfx90a)"""
    import pytest

    device_id = 0
    arch = hip.hipGetDeviceProperties(device_id).gcnArchName.split(":", 1)[0]

    if arch != "gfx90a":
        pytest.skip(f"Skipping gfx90a test - device is {arch}")

    results = run_mega_kernel_test(device_id)
    print_test_results(results, arch)

    # Assertions for pytest
    assert results.warp_shuffle_passed > 0, "Warp shuffle test failed"
    assert results.fp32_arith_passed > 0, "FP32 arithmetic test failed"
    assert results.mfma_f32_passed > 0, "MFMA FP32 test failed"


def test_mega_kernel_gfx942():
    """Test mega kernel on MI300 (gfx942)"""
    import pytest

    device_id = 0
    arch = hip.hipGetDeviceProperties(device_id).gcnArchName.split(":", 1)[0]

    if arch != "gfx942":
        pytest.skip(f"Skipping gfx942 test - device is {arch}")

    results = run_mega_kernel_test(device_id)
    print_test_results(results, arch)

    # Assertions
    assert results.fp8_convert_passed > 0, "FP8 conversion test failed"
    assert results.mfma_f8_passed > 0, "MFMA FP8 test failed"


def test_mega_kernel_gfx950():
    """Test mega kernel on MI350 (gfx950)"""
    import pytest

    device_id = 0
    arch = hip.hipGetDeviceProperties(device_id).gcnArchName.split(":", 1)[0]

    if arch != "gfx950":
        pytest.skip(f"Skipping gfx950 test - device is {arch}")

    results = run_mega_kernel_test(device_id)
    print_test_results(results, arch)

    # Assertions
    assert results.async_lds_load_passed > 0, "Async LDS load test failed"
    assert results.dual_issue_passed > 0, "Dual-issue VALU test failed"


def test_mega_kernel_gfx1150():
    """Test mega kernel on Strix/Strix Halo (RDNA 3.5, gfx1150/gfx1151/gfx1152)"""
    import pytest

    device_id = 0
    arch = hip.hipGetDeviceProperties(device_id).gcnArchName.split(":", 1)[0]

    if arch not in ("gfx1150", "gfx1151", "gfx1152"):
        pytest.skip(f"Skipping Strix/RDNA3.5 test - device is {arch}")

    results = run_mega_kernel_test(device_id)
    print_test_results(results, arch)

    # Assertions for RDNA 3.5: WMMA, warp ops, FP32, async LDS
    assert results.warp_shuffle_passed > 0, "Warp shuffle test failed"
    assert results.fp32_arith_passed > 0, "FP32 arithmetic test failed"
    assert results.wmma_f16_passed > 0, "WMMA FP16 test failed"
    assert results.vmem_tex_load_passed > 0, (
        "Texture load (tex1Dfetch / SQ_INSTS_TEX_LOAD) test failed"
    )
    assert results.vmem_tex_store_passed > 0, (
        "Texture store (surf1Dwrite / SQ_INSTS_TEX_STORE) test failed"
    )


def test_mega_kernel_current_device():
    """Test mega kernel on current device (auto-detect architecture)"""
    device_id = 0

    try:
        results = run_mega_kernel_test(device_id)
        arch = hip.hipGetDeviceProperties(device_id).gcnArchName.split(":", 1)[0]
        print_test_results(results, arch)

        # Basic assertion - at least some tests should pass.
        # Compute total passed tests on the host, ignoring bypassed (-1) values.
        total_passed_host = 0
        if hasattr(results, "_fields_"):
            for field_name, _ in results._fields_:
                if field_name.endswith("_passed") and field_name != "total_passed":
                    value = getattr(results, field_name)
                    # Ignore bypassed (-1) and zero; count only positive passes.
                    if isinstance(value, int) and value > 0:
                        total_passed_host += value

        assert total_passed_host > 0, "No tests passed"

    except Exception as e:
        print(f"Test failed with error: {e}")
        raise


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AMD GPU Mega Kernel Unit Test")
    parser.add_argument(
        "-d", "--device", type=int, default=0, help="GPU device ID (default: 0)"
    )
    parser.add_argument(
        "-b", "--batch-size", type=int, default=1024, help="Batch size (default: 1024)"
    )
    parser.add_argument(
        "-t",
        "--block-size",
        type=int,
        default=256,
        help="Thread block size (default: 256)",
    )
    parser.add_argument(
        "-m",
        "--mfma-mode",
        type=str,
        default="both",
        choices=["both", "asm", "builtin"],
        help="MFMA test mode (default: both)",
    )

    args = parser.parse_args()

    # Convert mfma_mode string to int
    mfma_mode_map = {"both": 0, "asm": 1, "builtin": 2}
    mfma_mode = mfma_mode_map[args.mfma_mode]

    try:
        results = run_mega_kernel_test(
            args.device, args.batch_size, args.block_size, mfma_mode
        )
        arch = hip.hipGetDeviceProperties(args.device).gcnArchName.split(":", 1)[0]
        # print_test_results computes host-side pass/fail/bypass totals
        total_passed, total_failed, total_bypassed = print_test_results(results, arch)

        # Exit with appropriate code based on host-computed totals
        sys.exit(0 if total_failed == 0 else 1)

    except Exception as e:
        print(f"\nTest execution failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
