# RocFuzz coverage baseline, 2026-05-25

This report is generated from the maintained real-library smoke and patch-report gates. It is a coverage-quality baseline, not an AFL crash campaign report; crash and hang findings are reported by `summarize-afl-campaign.py` from `afl-fuzz` output roots.

Source revision: `f6c5bb2b68 rocjitsu: document DBI survey for rocfuzz`
Examples build: `emulation/rocjitsu/fuzzer/examples/build`

## Summary

`Patched sites` is the successfully applied instrumentation count; `Device slots` is the observed runtime delta count. `Policy hashed`, `Policy fixed`, `Degraded branch edges`, `PrevBB sites`, and `PrevBB fallbacks` come from planner accounting in patch reports and can be higher when failed or partially patched events kept diagnostic plan data.

| Example | Status | Coverage mix | Patch events | Patched sites | Policy hashed | Policy fixed | Degraded branch edges | PrevBB sites | PrevBB fallbacks | Device slots | Main low-edge cause | Findings |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `rocblas-sgemm` | device branch coverage | previous-BB + fixed fallback | 4 | 1149 | 382 | 800 | 394 | 382 | 197 | 67 | `cfg_shape_no_branch_edge`=752, ... | not assessed by smoke baseline |
| `rocfft-c2c` | device branch coverage with known skipped payload | previous-BB | 3 | 8 | 8 | 0 | 0 | 8 | 0 | 1 | `other_skip_or_patch_failure`=16, ... | not assessed by smoke baseline |
| `rocrand-uniform` | device branch coverage | fixed branch counters | 2 | 17 | 0 | 17 | 0 | 0 | 0 | 9 | `loader_scoped_kernel_filter`=1708, ... | not assessed by smoke baseline |
| `rocsparse-spmv` | device branch coverage with known skipped payload | fixed branch counters | 3 | 23 | 0 | 23 | 0 | 0 | 0 | 12 | `loader_scoped_kernel_filter`=526, ... | not assessed by smoke baseline |
| `rocsolver-getrf` | device branch coverage | fixed branch counters | 6 | 46 | 0 | 46 | 0 | 0 | 0 | 33 | `loader_scoped_kernel_filter`=137, ... | not assessed by smoke baseline |
| `miopen-activation` | device branch coverage | fixed branch counters | 2 | 8 | 0 | 8 | 0 | 0 | 0 | 2 | `cfg_shape_no_branch_edge`=16 | not assessed by smoke baseline |

## Per-Example Details

### `rocblas-sgemm`

- Status: device branch coverage.
- Coverage mix: previous-BB + fixed fallback.
- Progress: 4/4 patch events succeeded, 1149 edge sites were patched; policy accounting selected hashed sites=382, fixed sites=800, degraded logical branch edges=394, previous-BB branch sites=382, and previous-BB sites degraded to fixed=197 (exec-safety=392, placement=2); 67 device slots changed with counter delta 5632.
- `high-edge fixed diagnostic` (`emulation/rocjitsu/fuzzer/examples/build/high-edge-reports/rocblas_sgemm_high_edge.jsonl`): 2/2 successful patch events, 609 patched sites; policy accounting hashed=0, fixed=609, degraded_branch_edges=0, previous_bb_branch_sites=0, previous_bb_site_fallbacks=0, fixed_fallback_causes=none, 16 device slots, strategies `self-contained-fixed-branch`=2, reasons `forced-skip-entry-env`=2.
- Selected edges in `high-edge fixed diagnostic`: `Cijk_S_GA` cond-branch@0x50 fixed-counter slot=4752 cave=appended-cave bytes=108, `Cijk_S_GA` cond-branch@0x220 fixed-counter slot=29384 cave=appended-cave bytes=108, `Cijk_S_PostGSU12_mod16` branch@0x1424 fixed-counter slot=22999 cave=appended-cave bytes=52, `Cijk_S_PostGSU12_mod16` branch@0x1774 fixed-counter slot=25271 cave=appended-cave bytes=52, ....
- `default hybrid` (`emulation/rocjitsu/fuzzer/examples/build/high-edge-reports/rocblas_sgemm_default_hybrid_high_edge.jsonl`): 2/2 successful patch events, 540 patched sites; policy accounting hashed=382, fixed=191, degraded_branch_edges=394, previous_bb_branch_sites=382, previous_bb_site_fallbacks=197, fixed_fallback_causes=exec-safety=392, placement=2, 51 device slots, strategies `self-contained-hybrid-previous-bb-and-fixed-branch`=2, reasons `entry-liveness-preflight-needs-fixed-registers`=2.
- Selected edges in `default hybrid`: `Cijk_S_GA` cond-branch@0x50 fixed-counter slot=4752 scratch=vgpr exec=forced-lane0 addr=v10 spill_vgpr=v0,v1 cave=appended-cave bytes=416, `Cijk_S_GA` cond-branch@0x220 previous-bb-hash scratch=vgpr+sgpr exec=exec addr=v10 spill_vgpr=v0,v1,v2,v3 spill_sgpr=s0,s1,s2,s3,s4 cave=appended-cave bytes=1212, `Cijk_S_PostGSU12_mod16` branch@0x1424 previous-bb-hash cave=appended-cave bytes=232, `Cijk_S_PostGSU12_mod16` branch@0x1774 previous-bb-hash cave=appended-cave bytes=232, ....
- Low-edge attribution: `cfg_shape_no_branch_edge`=752, `liveness_or_register_pressure`=36, `cave_or_branch_range_pressure`=32.
- Coverage degradation attribution: `EXEC-conditioned previous-BB branch probe is not yet proven safe`=392, `previous-BB trampoline placement failed; fixed counter trampoline fit after no branch-reachable local text cave`=2.
- Sampled skipped instructions: `s_endpgm` for `terminator exits kernel` x140 (0xbfb00000), `vopd_opaque` for `block falls through without a branch terminator` x74 (0xca100080, 0x03040080), `v_fmac_f32_e32` for `block falls through without a branch terminator` x24 (0x56080a05), `v_fma_mix_f32` for `block falls through without a branch terminator` x16 (0xcc200004, 0x14120a05), `s_lshl_b64` for `block falls through without a branch terminator` x12 (0x84ae8502), ....
- Findings: smoke baseline only; no crash/hang campaign findings assessed.
- Limitation: The default path has useful previous-BB hashed Tensile coverage, but more sites still need safer relocation, opaque-instruction modeling, and temporary-register planning.

### `rocfft-c2c`

- Status: device branch coverage with known skipped payload.
- Coverage mix: previous-BB.
- Progress: 2/3 patch events succeeded, 8 edge sites were patched; policy accounting selected hashed sites=8, fixed sites=0, degraded logical branch edges=0, previous-BB branch sites=8, and previous-BB sites degraded to fixed=0 (none); 1 device slots changed with counter delta 4.
- `high-edge FFT/twiddle` (`emulation/rocjitsu/fuzzer/examples/build/high-edge-reports/rocfft_c2c_high_edge.jsonl`): 2/3 successful patch events, 8 patched sites; policy accounting hashed=8, fixed=0, degraded_branch_edges=0, previous_bb_branch_sites=8, previous_bb_site_fallbacks=0, fixed_fallback_causes=none, 1 device slots, strategies `self-contained-previous-bb-branch`=2, `entry-previous-bb-block`=1, reasons `entry-liveness-preflight-needs-fixed-registers`=2, `default-entry-preferred`=1.
- Selected edges in `high-edge FFT/twiddle`: `fft_rtc_back_len_32_factors_8_4_wg...Lds_sp_ip_CI_unitstride_sbrr_dirReg` branch@0x60 previous-bb-hash scratch=vgpr+sgpr exec=exec addr=v5 spill_vgpr=v0,v1,v2,v3 spill_sgpr=s0,s1,s2,s3,s4 cave=appended-cave bytes=604, `fft_rtc_back_len_32_factors_8_4_wg...Lds_sp_ip_CI_unitstride_sbrr_dirReg` branch@0x1ac previous-bb-hash scratch=vgpr+sgpr exec=exec addr=v12 spill_vgpr=v0,v1,v2,v3 spill_sgpr=s0,s1,s2,s3,s4 cave=appended-cave bytes=604, `fft_rtc_back_len_32_factors_8_4_wg...Lds_sp_ip_CI_unitstride_sbrr_dirReg` cond-branch@0x40 previous-bb-hash cave=appended-cave bytes=468, `fft_rtc_back_len_32_factors_8_4_wg...Lds_sp_ip_CI_unitstride_sbrr_dirReg` cond-branch@0xf0 previous-bb-hash scratch=vgpr+sgpr exec=exec addr=v12 spill_vgpr=v0,v1,v2,v3 spill_sgpr=s0,s1,s2,s3,s4 cave=appended-cave bytes=1212, ....
- Low-edge attribution: `other_skip_or_patch_failure`=16, `cfg_shape_no_branch_edge`=13, `unsupported_relocation_or_decode`=12, `known_unsafe_kernel_classifier`=1.
- Sampled skipped instructions: `s_cbranch_execz` for `EXEC-conditioned previous-BB branch requires fixed-counter fallback budget` x2 (0xbfa5ffd3), `vopd_opaque` for `block falls through without a branch terminator` x2 (0xca100104, 0x06040103), `s_endpgm` for `terminator exits kernel` x2 (0xbfb00000), `s_swappc_b64` for `terminator is indirect call and needs return-address-preserving coverage policy` (0xbe9e4952).
- Findings: smoke baseline only; no crash/hang campaign findings assessed.
- Limitation: Generated FFT kernels produce device deltas; twiddle_gen_* is still reported as no-patchable-sites until its descriptor/prologue interaction is understood.

### `rocrand-uniform`

- Status: device branch coverage.
- Coverage mix: fixed branch counters.
- Progress: 2/2 patch events succeeded, 17 edge sites were patched; policy accounting selected hashed sites=0, fixed sites=17, degraded logical branch edges=0, previous-BB branch sites=0, and previous-BB sites degraded to fixed=0 (none); 9 device slots changed with counter delta 55421738.
- `default` (`emulation/rocjitsu/fuzzer/examples/build/rocrand_uniform_report.jsonl`): 2/2 successful patch events, 17 patched sites; policy accounting hashed=0, fixed=17, degraded_branch_edges=0, previous_bb_branch_sites=0, previous_bb_site_fallbacks=0, fixed_fallback_causes=none, 9 device slots, strategies `self-contained-fixed-branch`=2, reasons `loader-context-fixed-branch-preferred`=1, `loader-context-self-contained`=1.
- Selected edges in `default`: `_ZN12rocrand_impl6system6detail17t...13xorwow_engineEjPfmS9_jjmEEEvDpT4_` cond-branch@0x13ae8 fixed-counter slot=26498 cave=local-cave bytes=108, `_ZN12rocrand_impl6system6detail17t...13xorwow_engineEjPfmS9_jjmEEEvDpT4_` cond-branch@0x13bc0 fixed-counter slot=20292 cave=local-cave bytes=108, `_ZN12rocrand_impl6system6detail17t...device13xorwow_engineEjjyyEEEvDpT4_` branch@0x15c fixed-counter slot=2268 cave=local-cave bytes=52, `_ZN12rocrand_impl6system6detail17t...device13xorwow_engineEjjyyEEEvDpT4_` branch@0x3f0 fixed-counter slot=17972 cave=local-cave bytes=52, ....
- Low-edge attribution: `loader_scoped_kernel_filter`=1708, `cfg_shape_no_branch_edge`=9.
- Sampled skipped instructions: `s_endpgm` for `terminator exits kernel` x2 (0xbfb00000), `s_mov_b32` for `block falls through without a branch terminator` (0xbea80080), `v_add_co_ci_u32` for `block falls through without a branch terminator` (0xd5207c0b, 0x01aa1601).
- AFL-visible device showmap: `rocfuzz_example_rocrand_uniform_showmap.a.device` has 9 device tuples, `rocfuzz_example_rocrand_uniform_showmap.b.device` has 9, hashes are different.
- Findings: smoke baseline only; no crash/hang campaign findings assessed.
- Limitation: Coverage is shallow but validates launch-scoped KPACK/HSA-reader coverage and runtime shadow launch redirection.

### `rocsparse-spmv`

- Status: device branch coverage with known skipped payload.
- Coverage mix: fixed branch counters.
- Progress: 2/3 patch events succeeded, 23 edge sites were patched; policy accounting selected hashed sites=0, fixed sites=23, degraded logical branch edges=0, previous-BB branch sites=0, and previous-BB sites degraded to fixed=0 (none); 12 device slots changed with counter delta 1982.
- `default` (`emulation/rocjitsu/fuzzer/examples/build/rocsparse_spmv_report.jsonl`): 2/3 successful patch events, 23 patched sites; policy accounting hashed=0, fixed=23, degraded_branch_edges=0, previous_bb_branch_sites=0, previous_bb_site_fallbacks=0, fixed_fallback_causes=none, 12 device slots, strategies `self-contained-fixed-branch`=3, reasons `loader-context-fixed-branch-preferred`=2, `entry-liveness-preflight-rejected-fixed-registers`=1.
- Selected edges in `default`: `_ZN9rocsparseL12scale_kernelILj256...const_host_device_scalarIT2_EEPT1_b` branch@0x3da0 fixed-counter slot=2238 cave=appended-cave bytes=52, `_ZN9rocsparseL12scale_kernelILj256...const_host_device_scalarIT2_EEPT1_b` cond-branch@0x3d28 fixed-counter slot=3618 cave=appended-cave bytes=108, `_ZN9rocsparseL12scale_kernelILj256...const_host_device_scalarIT2_EEPT1_b` cond-branch@0x3d3c fixed-counter slot=24088 cave=appended-cave bytes=108, `_ZN9rocsparseL12scale_kernelILj256...const_host_device_scalarIT2_EEPT1_b` cond-branch@0x3d60 fixed-counter slot=16898 cave=appended-cave bytes=108, ....
- Low-edge attribution: `loader_scoped_kernel_filter`=526, `cfg_shape_no_branch_edge`=9.
- Sampled skipped instructions: `s_endpgm` for `terminator exits kernel` x3 (0xbfb00000), `s_load_b32` for `block falls through without a branch terminator` x2 (0xf4000102, 0xf8000000).
- AFL-visible device showmap: `rocfuzz_example_rocsparse_spmv_showmap.a.device` has 10 device tuples, `rocfuzz_example_rocsparse_spmv_showmap.b.device` has 12, hashes are different.
- Findings: smoke baseline only; no crash/hang campaign findings assessed.
- Limitation: The retained wrapper uses the non-persistent target because the rocSPARSE handle path is not yet stable across AFL deferred forkserver boundaries.

### `rocsolver-getrf`

- Status: device branch coverage.
- Coverage mix: fixed branch counters.
- Progress: 6/6 patch events succeeded, 46 edge sites were patched; policy accounting selected hashed sites=0, fixed sites=46, degraded logical branch edges=0, previous-BB branch sites=0, and previous-BB sites degraded to fixed=0 (none); 33 device slots changed with counter delta 7847.
- `default` (`emulation/rocjitsu/fuzzer/examples/build/rocsolver_getrf_report.jsonl`): 6/6 successful patch events, 46 patched sites; policy accounting hashed=0, fixed=46, degraded_branch_edges=0, previous_bb_branch_sites=0, previous_bb_site_fallbacks=0, fixed_fallback_causes=none, 33 device slots, strategies `self-contained-fixed-branch`=6, reasons `loader-context-fixed-branch-preferred`=3, `loader-context-self-contained`=3.
- Selected edges in `default`: `_ZN9rocsolver6v33500L10reset_infoIiiiEEvPT_T0_T1_S4_` cond-branch@0x138 fixed-counter slot=21866 cave=appended-cave bytes=108, `_ZN9rocsolver6v33500L10reset_infoIiiiEEvPT_T0_T1_S4_` cond-branch@0x1dc fixed-counter slot=22393 cave=appended-cave bytes=108, `_ZN9rocsolver6v33500L11getf2_iamaxIfiPfEEvT0_T1_lS3_lPS3_` cond-branch@0x334 fixed-counter slot=31424 cave=appended-cave bytes=108, `_ZN9rocsolver6v33500L11getf2_iamaxIfiPfEEvT0_T1_lS3_lPS3_` cond-branch@0x400 fixed-counter slot=23426 cave=appended-cave bytes=108, ....
- Low-edge attribution: `loader_scoped_kernel_filter`=137, `cfg_shape_no_branch_edge`=44.
- Sampled skipped instructions: `s_endpgm` for `terminator exits kernel` x6 (0xbfb00000), `ds_store_b32` for `block falls through without a branch terminator` x2 (0xd8340000, 0x00000100), `global_store_b32` for `block falls through without a branch terminator` x2 (0xee06807c, 0x03000000, ...), `s_mov_b32` for `block falls through without a branch terminator` x2 (0xbe830080).
- Findings: smoke baseline only; no crash/hang campaign findings assessed.
- Limitation: Coverage is wired through short helper kernels; deeper edge identity needs more previous-BB-safe sites.

### `miopen-activation`

- Status: device branch coverage.
- Coverage mix: fixed branch counters.
- Progress: 2/2 patch events succeeded, 8 edge sites were patched; policy accounting selected hashed sites=0, fixed sites=8, degraded logical branch edges=0, previous-BB branch sites=0, and previous-BB sites degraded to fixed=0 (none); 2 device slots changed with counter delta 16.
- `default` (`emulation/rocjitsu/fuzzer/examples/build/miopen_activation_report.jsonl`): 1/1 successful patch events, 4 patched sites; policy accounting hashed=0, fixed=4, degraded_branch_edges=0, previous_bb_branch_sites=0, previous_bb_site_fallbacks=0, fixed_fallback_causes=none, 1 device slots, strategies `self-contained-fixed-branch`=1, reasons `classifier-entry-redirection-unsafe`=1.
- Selected edges in `default`: `MIOpenActiveBwd2DLite` cond-branch@0xb78 fixed-counter slot=14469 cave=appended-cave bytes=108, `MIOpenActiveBwdLite` cond-branch@0x920 fixed-counter slot=24074 cave=appended-cave bytes=108, `MIOpenActiveFwd2DLite` cond-branch@0x474 fixed-counter slot=12160 cave=appended-cave bytes=108, `MIOpenActiveFwdLite` cond-branch@0x20 fixed-counter slot=890 cave=appended-cave bytes=108.
- `high-edge fixed diagnostic` (`emulation/rocjitsu/fuzzer/examples/build/high-edge-reports/miopen_activation_high_edge.jsonl`): 1/1 successful patch events, 4 patched sites; policy accounting hashed=0, fixed=4, degraded_branch_edges=0, previous_bb_branch_sites=0, previous_bb_site_fallbacks=0, fixed_fallback_causes=none, 1 device slots, strategies `self-contained-fixed-branch`=1, reasons `forced-skip-entry-env`=1.
- Selected edges in `high-edge fixed diagnostic`: `MIOpenActiveBwd2DLite` cond-branch@0xb78 fixed-counter slot=14469 cave=appended-cave bytes=108, `MIOpenActiveBwdLite` cond-branch@0x920 fixed-counter slot=24074 cave=appended-cave bytes=108, `MIOpenActiveFwd2DLite` cond-branch@0x474 fixed-counter slot=12160 cave=appended-cave bytes=108, `MIOpenActiveFwdLite` cond-branch@0x20 fixed-counter slot=890 cave=appended-cave bytes=108.
- Low-edge attribution: `cfg_shape_no_branch_edge`=16.
- Sampled skipped instructions: `global_store_b128` for `block falls through without a branch terminator` x8 (0xee07407c, 0x00000000, ...), `s_endpgm` for `terminator exits kernel` x8 (0xbfb00000).
- Findings: smoke baseline only; no crash/hang campaign findings assessed.
- Limitation: Activation kernels are small. The current value is proving that the entry-unsafe MIOpen path uses self-contained fixed branch counters without entry redirection.
