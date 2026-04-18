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

# Softmax: the legacy transpiler's own gfx1250 softmax fixture uses
# v_mov_b32_dpp (DppCrossLane / SPE_DESIGN.md §3 Class 2) and
# v_permlanex16_b32 (LaneGroupShuffle / §3 Class 2) per
# gpt-oss-derisking.md §7.2/§7.3. Under the new wave-size classifier
# gate (SPE_DESIGN.md §4, wave_size_obstruction.{hpp,cpp}) these are
# correctly flagged as outcome (c) shuffle-rewrite-pending — the
# pre-classifier raiser accepted them and emitted same-lane fallback
# IR (accidentally correct on the specific all-1.0f softmax input,
# see CROSS_LANE_SURVEY.md for why). Graduating this XFAIL back to
# expected-pass requires CROSS_LANE_SURVEY.md items P5 (DPP intrinsic
# lift) and P2/P3/P4 (permlane16 intrinsic lift) to land; at that
# point the corresponding ObstructionKind sites' `rewriteImplemented`
# bit in wave_size_obstruction.cpp flips to true and the classifier
# accepts the kernel as outcome (b).
set_tests_properties(Gfx1250Gpu.Softmax PROPERTIES
  WILL_FAIL TRUE
  LABELS "transpiler;xfail;wave-size-rewrite-pending"
)

# Integration vecadd via runPipelineAllKernels: hipError 719 (unspecified
# launch failure).  Single-kernel path (Gfx1250Gpu.Vecadd) works; the
# merged-HSACO linker path has a bug.
set_tests_properties(Integration.VecaddAllKernels PROPERTIES
  WILL_FAIL TRUE
  LABELS "transpiler;xfail"
)
