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

# Gfx1250Gpu.Softmax: re-introduced as expected-fail on 2026-04-21
# after a separate investigation into a `runPipeline` overload-
# resolution bug (see pipeline.hpp's block comment on the removed
# 3-string convenience overload) uncovered that every cross-arch
# test in this file was silently failing at the pipeline-launch
# boundary rather than actually running.  The pre-fix state made
# `Gfx1250Gpu.Softmax` exit via `ASSERT_TRUE(result.success)` on an
# empty/error PipelineResult BEFORE any raised IR touched the GPU;
# CTest's WILL_FAIL expectation held trivially.  Post-fix, softmax
# reaches the launch and produces `+inf` for every output element,
# while the max reduction over an fp32 array has no input capable of
# producing `+inf` on a finite input set — so the defect is in the
# lifted max/exp reduction path, not the test harness.  This is a
# pre-existing transpiler bug independent of both the overload fix
# and the FLAT_LOAD SADDR-form fix in handle_flat.cpp; investigation
# deferred to a dedicated pass.
#
# The softmax kernel in compare_correctness's Triton corpus
# (`corpus_softmax_fp32`) is structurally different (different
# reduction tree shape, different BLOCK_SIZE) and fails loudly at
# lift time via the writelane/readlane-post-raise-safety-net
# classifier; that failure mode is documented in
# hotswap/docs/modrep-predicate-chain.md §9 (5) sibling-class note
# and is not the same defect as the one re-exposed here.
set_tests_properties(Gfx1250Gpu.Softmax PROPERTIES
  WILL_FAIL TRUE
  LABELS "transpiler;xfail"
)

# Integration vecadd via runPipelineAllKernels: hipError 719 (unspecified
# launch failure).  Single-kernel path (Gfx1250Gpu.Vecadd) works; the
# merged-HSACO linker path has a bug.
set_tests_properties(Integration.VecaddAllKernels PROPERTIES
  WILL_FAIL TRUE
  LABELS "transpiler;xfail"
)
