"""MODE=1 bisect variant with num_warps=1.  If this matches salmon
while num_warps=4 (topk_forward_bisect_m1) is WRONG, the regression
depends on multi-warp wave-size-projection composition — specifically,
the cross-warp portion of the tl.sum reduction is what miscompiles.
If both WRONG, the bug is in a single-warp path.
"""
from topk_forward_bisect import _topk_forward_bisect, _recipe
_r = _recipe("topk_forward_bisect_m1_nw1", 1)
_r["num_warps"] = 1
RECIPES = [_r]
