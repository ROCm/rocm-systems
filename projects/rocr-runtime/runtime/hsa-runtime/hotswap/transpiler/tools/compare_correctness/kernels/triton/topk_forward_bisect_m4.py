"""topk_forward_bisect MODE=4 — see topk_forward_bisect.py for the full
bisection narrative.  Phase-1 AOT requires one recipe per file, so each
MODE lives in its own stem; the shared kernel is imported from the
parent bisect module.
"""
from topk_forward_bisect import _topk_forward_bisect, _recipe
RECIPES = [_recipe("topk_forward_bisect_m4", 4)]
