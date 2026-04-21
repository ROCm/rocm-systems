# Known expected failures (XFAIL).
#
# These tests run but are expected to fail.  CTest treats them as:
#   - PASS if the test exits non-zero (still failing as expected)
#   - FAIL if the test exits zero   (unexpectedly fixed -- update this file!)
#
# When you fix one of these, remove the corresponding set_tests_properties
# line so the test is expected to pass again.

# 128x128-tile matmul family graduated to expected-pass in commit
# da404faf84 ("V_CMP -> V_CNDMASK per-lane-i1 shadow restores cross-
# widening"). The whole Gfx1250Gpu.Matmul* family now runs bit-exact
# or within f16-rounding-only noise on gfx1250 → gfx942 cross-
# widening. See hotswap/docs/learnings.md for the post-mortem (root
# cause: V_CMP → SGPR → V_CNDMASK wave-mask round trip truncated the
# upper 32 bits of the per-lane compare result when routed through
# `ballotI1ToWidth`'s narrow-store, so target Wave64 lanes 32..63 —
# the lanes holding source wave 3's share of the output tile —
# silently fell out of the downstream `v_cndmask`-driven address
# math in the prologue's LDS-to-VGPR load and read warp 0's A-
# fragment into warp 3's destination VGPRs). The four diagnostic
# probes (RowIdA / RowOnly124 / EvenRows / KStripedRow124) remain as
# positive regression guards for this exact defect.

# Gfx1250Gpu.Softmax graduated from XFAIL to expected-pass once
# the P2 rewrite (v_permlane16 / v_permlanex16 emulation via
# ds_bpermute_b32; see the permlane16/permlanex16 row of
# hotswap/docs/wave-size-translation.md §5.3) landed on top of P5
# (DPP intrinsic lift). The
# kernel's two remaining classifier-blocked sites — a single
# `v_permlanex16_b32` and the DPP pattern — now both surface in the
# classifier trace as `[implemented]`, and the end-to-end test runs
# the raised IR on gfx942 hardware producing bit-exact softmax
# outputs (0 errors, maxErr=0.0). This XFAIL annotation has been
# removed.

# Integration vecadd via runPipelineAllKernels: hipError 719 (unspecified
# launch failure).  Single-kernel path (Gfx1250Gpu.Vecadd) works; the
# merged-HSACO linker path has a bug.
set_tests_properties(Integration.VecaddAllKernels PROPERTIES
  WILL_FAIL TRUE
  LABELS "transpiler;xfail"
)
