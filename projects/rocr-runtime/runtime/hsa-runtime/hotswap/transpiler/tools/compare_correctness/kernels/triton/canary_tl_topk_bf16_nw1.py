"""`tl.topk` probe with num_warps=1 to bisect cross-wave projection.

Companion to `canary_tl_topk_bf16`.  If num_warps=4 is WRONG and
num_warps=1 is match, the bug is in CROSS-WARP composition (each
warp's partial top-k + merge across warps).  If both are WRONG,
the bug is in single-warp `tl.topk` cross-lane reduction (under
wave32->wave64 projection).
"""
from canary_tl_topk_bf16 import RECIPES as _BASE
_r = dict(_BASE[0])  # shallow copy
_r["name"] = "canary_tl_topk_bf16_nw1"
_r["num_warps"] = 1
RECIPES = [_r]
