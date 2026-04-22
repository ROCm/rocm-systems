"""MODE=2 variant with strict (bit-exact) comparator on y_values.

Why this probe exists
=====================

`topk_forward_bf16` (full `_topk_forward` pipeline) shows 93% of rows
with different top-k INDEX sets AND a systematic bias where salmon's
post-softmax Yv[0] is LARGER than native's (mean +0.034, SNR 0.88).
That signature points to a real bug (not bf16 drift) inside
streaming_topk or downstream in softmax/sort.

The existing `topk_forward_bisect_m2` exercises `streaming_topk` but
uses `rel-rms(tol=0.15)` on its Yv output.  Under that tolerance,
salmon CAN pick an entirely different top-k SET and still pass
the Yv verdict (if the picked values have similar RMS magnitude).
That's principled for "soft" correctness at the full pipeline
level (the picked values dominate downstream softmax), but it
hides the question "does streaming_topk pick the SAME SET".

This probe runs the identical kernel body as MODE=2 but ratchets
the comparator on Yv to `abs tol=0.0`: if and only if salmon's
top-4 values are bit-identical to native's (in order), the test
matches.  Since streaming_topk sorts descending by value-key
before returning, ANY divergence in the picked SET (or in the
sort order of near-tied values) will show up as a bit-exact
value mismatch.

Why this is the right bisect — avoiding the v_pk_sub_i16 handler gap
------------------------------------------------------------------

An earlier attempt (topk_forward_bisect_m2_idx) exposed
streaming_topk's y_indices as a separate i16 output so the
comparator could be pinned bit-exact on the indices themselves.
That probe tripped a raise-time handler gap: writing Yi as i16
caused the compiler to emit `v_pk_sub_i16` (VOP3P packed int16
subtraction, gfx10/11+ only) for `N_EXPTS_PAD - indx` — Salmon
doesn't handle that opcode.  The production `topk_forward_bf16`
kernel does NOT emit `v_pk_sub_i16` because its downstream Bits
derivation forces a different (wider) instruction selection
path.  Rather than blocking on the handler gap (which is real,
to file as a separate follow-up), this probe routes around it
by NOT writing Yi — only Yv under the strict tolerance.  The
semantic coverage is identical: streaming_topk returns sorted
(y_values, y_indices); if the index SETS disagree, the VALUES
necessarily disagree (at positions where the chosen column
differs).

Expected verdicts
=================

* salmon=match — streaming_topk's y_values are bit-identical to
  native's.  The bug is then downstream (softmax + Yi write +
  Bits derivation).  Very unlikely given the MODE=2 existing
  measurement of rel-rms ~= 0.072.
* salmon=WRONG k/2048 — streaming_topk's y_values diverge.  The
  bug is inside streaming_topk (tl.topk, tl.bitonic_merge,
  tl.sort, the u32 (value_key << 16) | index_key pack, or the
  -inf poisoning of out-of-range lanes in the first-iteration
  peel).  Pair this probe's verdict with the `topk_forward_bf16`
  max|err| signature to narrow further.

All other shape / constexpr parameters match the shared
`_recipe` helper (see `topk_forward_bisect.py`) so the kernel
body is identical to MODE=2.
"""

from topk_forward_bisect import _recipe


_r = _recipe("topk_forward_bisect_m2_strict", 2)

# Override the comparator from the shared helper's rel-rms(0.15) to
# strict bit-exact.  This isolates "did streaming_topk return
# byte-identical y_values" — anything else is a real divergence in
# the pre-softmax path.
#
# Rationale for strict vs some tighter rel-rms (say 0.01): we're
# bisecting, not reporting a production tolerance.  MODE=2 the bulk
# recipe remains at rel-rms(0.15) for "does the full pipeline
# numerically match within a tolerable band", this sibling asks
# the orthogonal question "does the specific stage we're probing
# produce exactly the native-gold result".  Mixing both into one
# comparator would lose the signal we need.
_r["comparator"] = {"kind": "abs", "tol": 0.0}

RECIPES = [_r]
