# Known expected failures (XFAIL).
#
# These tests run but are expected to fail.  CTest treats them as:
#   - PASS if the test exits non-zero (still failing as expected)
#   - FAIL if the test exits zero   (unexpectedly fixed -- update this file!)
#
# When you fix one of these, remove the corresponding set_tests_properties
# line so the test is expected to pass again.

# 128x128-tile matmul: hipError 700 (illegal memory access) in the
# transpiled large-tile kernel.
set_tests_properties(Gfx1250Gpu.Matmul128x128_1tile PROPERTIES
  WILL_FAIL TRUE
  LABELS "transpiler;xfail"
)

# 128x128-tile matmul at 256x256: hangs (transpiler bug in large-tile).
# Short timeout so it doesn't block the suite.
set_tests_properties(Gfx1250Gpu.Matmul128x128 PROPERTIES
  WILL_FAIL TRUE
  TIMEOUT 30
  LABELS "transpiler;xfail"
)

# Gfx1250Gpu.Softmax graduated from XFAIL to expected-pass once
# CROSS_LANE_SURVEY.md P2 (v_permlane16 / v_permlanex16 emulation
# via ds_bpermute_b32) landed on top of P5 (DPP intrinsic lift). The
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
