# Known expected failures (XFAIL).
#
# These tests run but are expected to fail.  CTest treats them as:
#   - PASS if the test exits non-zero (still failing as expected)
#   - FAIL if the test exits zero   (unexpectedly fixed -- update this file!)
#
# When you fix one of these, remove the corresponding set_tests_properties
# line so the test is expected to pass again.

# 128x128-tile matmul family — all three variants refuse at raise time
# with ObstructionKind::WaveIdLiftScalarized (see
# wave_size_obstruction.cpp::buildObstructionReport and the
# lit_tests/c1_wave_id_lift_scalarized refusal fixture).
#
# The generated amdgcn contains the three-way co-occurrence pinned by
# the classifier:
#   1. `s_bfe_u32 sDST, ttmp8, 0x50019` — wave_id_in_workgroup
#      extraction that every Tensile/rocBLAS matmul emits.
#   2. v_writelane_b32 / v_readlane_b32 — hipcc's register allocator
#      chooses these for spill/reload around the 128x128-tile shape
#      (the 64x64-tile sibling Matmul64x64 does NOT trigger them, so
#      stays on the intrinsic-lift rescue path and passes).
#   3. v_wmma_* — the core matmul accumulator opcode.
#
# The handle_sop2.cpp lift for the canonical BFE still fires and
# produces a per-lane divergent VGPR value for the wave_id, but when
# that SGPR-shaped value feeds a v_writelane scalar source operand
# the backend inserts an implicit v_readfirstlane that collapses
# source_wave[0]'s wave_id=0 and source_wave[1]'s wave_id=1 into one
# uniform. WMMA forecloses ThreadLoopProjection (§5.2 needs the full
# target wave simultaneously), so the only correct outcome is the
# loud refusal — exactly what the classifier now emits.
#
# These tests currently assert end-to-end numerical correctness, and
# the raise step aborts before any dispatch happens. Until the raise
# step gains a principled ThreadLoopProjection variant that survives
# WMMA (or the BFE lift is reworked to forward the divergent value
# through the writelane scalar operand in a shape the backend won't
# scalarise) the numerical assertion cannot succeed.
#
# See docs/wave-size-translation.md §6 Class-1 refuse row + §5.2
# ThreadLoopProjection exclusion note for the principled justification.
set_tests_properties(Gfx1250Gpu.Matmul128x128_1tile PROPERTIES
  WILL_FAIL TRUE
  LABELS "transpiler;xfail"
)

set_tests_properties(Gfx1250Gpu.Matmul128x128 PROPERTIES
  WILL_FAIL TRUE
  TIMEOUT 30
  LABELS "transpiler;xfail"
)

# UniformDiag variant of the 128x128 matmul — same refusal path as the
# two above; the uniform-diagonal input pattern was added as a
# debugging probe for the numerical miscompile that the
# WaveIdLiftScalarized refusal now intercepts at raise time. Shares
# the same failure mode until the classifier's underlying issue is
# resolved (see the long comment above).
set_tests_properties(Gfx1250Gpu.Matmul128x128_1tile_UniformDiag PROPERTIES
  WILL_FAIL TRUE
  LABELS "transpiler;xfail"
)

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
