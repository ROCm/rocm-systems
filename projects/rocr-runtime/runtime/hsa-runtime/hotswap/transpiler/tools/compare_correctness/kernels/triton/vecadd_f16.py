"""Triton vecadd kernel for compare_correctness.

The file stem (``vecadd_f16``) is the recipe name and the .co filename root.
AOT-compiled by ``aot_compile.py`` for both gfx942 (wave64) and gfx1250
(wave32).  The compare_correctness harness runs the gfx942 build natively as
the gold and compares the gfx1250 build under legacy / salmon transpilation
paths.

Phase 1 keeps the recipe deliberately minimal — f16 inputs, f16 output, a
single scalar shape dim (``N``), and a handful of sizes chosen to straddle
the BLOCK_SIZE boundary.  Extend the ``default_shapes`` list to widen the
sweep; everything else is wired up automatically.
"""
import triton
import triton.language as tl


@triton.jit
def vecadd_kernel(a_ptr, b_ptr, c_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    a = tl.load(a_ptr + offs, mask=mask)
    b = tl.load(b_ptr + offs, mask=mask)
    tl.store(c_ptr + offs, a + b, mask=mask)


RECIPES = [
    {
        "name": "vecadd_f16",
        "kernel_fn": vecadd_kernel,
        # Symbol the code object exposes — this is the Python function name
        # unless @jit is given an override.
        "kernel_symbol": "vecadd_kernel",
        "signature": {
            "a_ptr": "*fp16",
            "b_ptr": "*fp16",
            "c_ptr": "*fp16",
            "N": "i32",
        },
        "constexprs": {"BLOCK_SIZE": 1024},
        "num_warps": 4,
        # The scalar shape dim the harness sweeps.  Its name ("N") must
        # match a scalar sig arg; the dispatch looks it up to fill the
        # kernarg slot.
        "shape_dim": "N",
        # Mix of "clean" multiples of BLOCK_SIZE and a "ragged" size that
        # exercises the trailing mask path.
        "default_shapes": [1024, 4096, 65536, 100000],
        # Launch grid over BLOCK_SIZE-wide tiles.  Only ceil_div, infix
        # math, integer literals, and identifiers from shape/constexprs
        # are supported (see ExprEval in compare_correctness.cpp).
        "grid": {
            "x": "ceil_div(N, BLOCK_SIZE)",
            "y": "1",
            "z": "1",
        },
        # Buffers populated from deterministic RNG and fed into the kernel
        # as global_buffer args.
        "inputs": [
            {"name": "a_ptr", "dtype": "fp16", "elems": "N"},
            {"name": "b_ptr", "dtype": "fp16", "elems": "N"},
        ],
        # Buffers the harness reads back and compares against the gold.
        "outputs": [
            {"name": "c_ptr", "dtype": "fp16", "elems": "N"},
        ],
        # fp16 add-rounding round-trip is exact for inputs in [-1, 1], but
        # give a little slack to absorb any wave-size-dependent reordering
        # the transpiler paths may introduce.  Tighten if we want stricter
        # verification later.
        "comparator": {"kind": "abs", "tol": 1e-2},
    }
]
