"""`tl.sort` on fp32 with DETERMINISTIC lane-identifiable input.

Each row r is `X[r, c] = c.0`: lane c in col c, so the sort
output tells us precisely which LANE ID each output slot pulled
from.  Under a correct implementation, the descending sort of
[0, 1, ..., 31] is [31, 30, ..., 0].

Salmon's canary_tl_sort_fp32 at N=32 shows TWO SORTED HALVES (cols
0..15 desc, cols 16..31 desc) — clearly a broken final bitonic
merge.  With deterministic input, we can READ OUT which
lanes-ended-up-where, which will let us reverse-engineer the
specific shuffle network that's being emitted vs what the source
expects.

Expected under correct sort-descending of [0..31]:
  output = [31, 30, 29, 28, ..., 1, 0]

Expected (if salmon is producing two sorted halves without final
merge) for the source wave32 bitonic sort:
  output[0..15]  = sorted descending of some 16-element subset S1
  output[16..31] = sorted descending of the other 16-element subset S2
  where S1 ∪ S2 = {0..31}

The two subsets S1, S2 will pinpoint the shape of the broken
shuffle — e.g. S1={1,3,5,...,31} (odd-indexed source lanes),
S2={0,2,4,...,30} (even-indexed) implies the first-pair-swap
step is producing out-of-bounds partner indices that happen
to pull the other half's values.
"""
import triton
import triton.language as tl


@triton.jit
def _canary_tl_sort_fp32_deterministic(
    Y, stride_ym,
    n_rows,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid * BLOCK_M >= n_rows:
        return
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    mask_m = offs_m[:, None] < n_rows

    # Deterministic input: X[r, c] = c (i.e. each column holds its
    # column index as its value).  No tl.load — we synthesise
    # in-kernel to keep the IR minimal.
    x = tl.broadcast_to(offs_n[None, :].to(tl.float32), [BLOCK_M, BLOCK_N])
    y = tl.sort(x, dim=1, descending=True)

    Y_ptrs = Y + offs_m[:, None] * stride_ym + offs_n[None, :]
    tl.store(Y_ptrs, y, mask=mask_m)


RECIPES = [{
    "name": "canary_tl_sort_fp32_deterministic",
    "kernel_fn":     _canary_tl_sort_fp32_deterministic,
    "kernel_symbol": "_canary_tl_sort_fp32_deterministic",
    "signature": {
        "Y":           "*fp32",
        "stride_ym":   "i32",
        "n_rows":      "i32",
    },
    "constexprs": {"BLOCK_M": 32, "BLOCK_N": 32},
    "harness_constants": {"N_COLS": 32},
    "scalar_args": {
        "stride_ym": "N_COLS",
        "n_rows":    "N_ROWS",
    },
    "num_warps": 4,
    "shape_dim": "N_ROWS",
    "default_shapes": [512],
    "grid": {
        "x": "ceil_div(N_ROWS, BLOCK_M)",
        "y": "1",
        "z": "1",
    },
    "inputs": [],
    "outputs": [
        {"name": "Y", "dtype": "fp32", "elems": "N_ROWS * N_COLS"},
    ],
    "comparator": {"kind": "abs", "tol": 0.0},
}]
