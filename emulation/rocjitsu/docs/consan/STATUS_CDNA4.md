# ConSan CDNA4 (`gfx950`) status

Post-merge physical spot revalidation (2026-09-09) used rocm-systems
`6022be69de07`, hip-moi `f15bf1b96124`, rocjitsu-test-corpus
`c00e52c015f2`, and the TheRock-only runtime under
`/home/benjacob/.venv/lib/python3.12/site-packages/_rocm_sdk_devel`. The
official validator accepted the baseline and all four clean profiles for
`d128-block`, `d128-pressure`, `wmma-attention`, `streamk-arrival`, and
`hip-matmul-m128-n128-k128` on the physical MI350X.
It also accepted baseline, SuperCollider, Sampled, and Inline Shadow for
`tree-atomic-or`. Current hip-moi compiler output changes the complete access
denominators from 128 to 122 for `d128-block`, from 252 to 236 for
`d128-pressure`, and from 58 to 50 for `wmma-attention`. The current Sampled
denominators are 119 barriers for `d128-block`, 28 for `d128-pressure`, and 14
for `wmma-attention`, matching the other MOI engines rather than the older
117, 24, and 12 counts. The initial `tree-atomic-or` Record/Replay attempt was
a reproducible physical and emulator regression: its private-state entry
prologue saved the replacement queue/dispatch preload layout before restoring
the guest ABI, then restored a queue pointer over the guest kernarg pointer.
The corrected transaction captures launch identity, repairs the guest ABI,
and only then preserves borrowed entry SGPRs. The official physical validator
now accepts both baseline and Record/Replay with the exact oracle, zero
diagnostics, complete dynamic evidence, 48/48 accesses, 6/6 barriers, 3/3
atomics, and 5/5 fences. The accepted artifact is
`/tmp/consan-validation-gfx950-tree-rr-entry-abi-fix-20260909-c`. The original
failure artifacts, including the test-corpus row, remain under
`/tmp/consan-validation-gfx950-post-merge-6022be69`; the independent
Record/Replay confirmation is under
`/tmp/consan-validation-gfx950-tree-rr-confirm-6022be69`, and the emulator
result is under `/tmp/consan-validation-gfx950-tree-rr-emulator-6022be69`.

ConSan-only physical and simulator follow-up (2026-09-09): a
direct-HSA two-queue regression now forces equal raw dispatch IDs with the same
kernel across reused and distinct kernarg addresses and repeats the collision
at absolute packet indices zero and one. The pre-fix queue-blind fingerprint
produced one Record/Replay dispatch token and a false conflict; the queue-aware
build keeps the clean Record/Replay and Inline Shadow members diagnostic-free,
and Sampled now requests per-launch identity for access/atomic consumers at
stride one. The paired racy members retain their mode-specific expectations.
The complete ConSan host gate passes 1,573 of 1,575 tests with two benchmark
skips, and the two labeled host CTest contracts pass. The expanded simulator
gate passes 2,926/2,926 in 153.90 seconds,
including the eight-case two-queue matrix and ten filtered gfx950 histogram
cases. HIP repeated-dispatch and graph fixtures now state their completed
launch counts while explicitly classifying their underlying HSA queue count
and raw dispatch-ID reuse as unobserved; Record/Replay simultaneously requires
the corresponding retained dispatch-fingerprint counts. Python validation-tool
discovery passes all 343 tests using the venv Python plus the already-installed
Ubuntu `PyYAML` module; no system ROCm component is used. The two-queue HSACO is
`bfed3c14dbc498f925c76060d6e63f8a0e5608361204264ddd1fdbf4d375d3a0`;
the histogram HSACO is
`e43dc0f0b97075279f0323d8eac5e2579661bca8952a90992256ab7d17e124d9`.
All compilers and ROCm libraries resolved under
`/home/benjacob/.venv/lib/python3.12/site-packages/_rocm_sdk_devel`. An
unsandboxed TheRock `rocminfo` identified the physical device as `gfx950`, AMD
Instinct MI350X. The complete current physical ConSan gate passes 606/606 in
433.87 seconds, including the eight direct-HSA two-queue cases, the strengthened
HIP replay/graph cases, all histogram cases, scalar-pressure/spill coverage,
and the post-instrumentation health probe. The current 64-bit
queue-pointer/dispatch-ID
fingerprint retains the collision and destroyed-queue address-reuse limitation
documented in [VALIDATION.md](VALIDATION.md); Sampled's scalar-pressure literal
fallback is weaker still. Neither may be described as exact identity.

Private-entry ABI corrective qualification (2026-09-09): the complete ConSan
host gate passes 1,574/1,576 with only the two intentional benchmark skips.
A minimal scalar-pressure LDS device kernel that forces the same private
dispatch-identity and lane-backed entry-save path passes under both the gfx950
simulator and the physical MI350X. The neighboring memory-backed private-entry
host test now also requires guest repair before its scalar save, and the CDNA3
and CDNA4 full-window kernarg-preload tests verify that emitted tail reloads use
the restored guest kernarg-pointer SGPR. The focused simulator/physical
dispatch-identity, two-queue, and kernarg-preload matrix passes all 52 cases.

Current-toolchain physical E2E follow-up (2026-09-09) used rocm-systems
`ce48f29a13`, rocjitsu-test-corpus `c00e52c015f2`, and TheRock clang 24 plus
the runtime under
`/home/benjacob/.venv/lib/python3.12/site-packages/_rocm_sdk_devel`.
The canonical validator accepts baseline and every clean profile for
HipKittens BF16 and MXFP8: BF16 has 128/128 accesses and 32/32 barriers in
each MOI engine, while MXFP8 has 96/96 accesses and 5/5 barriers. The
artifacts are
`/tmp/consan-validation-gfx950-physical-hipkittens-bf16-all-20260909-a` and
`/tmp/consan-validation-gfx950-physical-hipkittens-mxfp8-all-20260909-a`.
For the freshly built HIP Stream-K objects, baseline, SuperCollider, and
Record/Replay pass both workloads; Sampled passes the simple workload, but
the two-tile exact oracle finishes with two dynamically incomplete events.
Inline Shadow on simple is rejected before execution because automatic
placement cannot reserve its required EXEC-save SGPR window. Inline Shadow
on two-tile executes the exact oracle with complete static coverage, but
reports 3,272,268 dynamically incomplete events. The Stream-K artifacts are
`/tmp/consan-validation-gfx950-physical-streamk-simple-all-20260909-a` and
`/tmp/consan-validation-gfx950-physical-streamk-two-tile-all-20260909-a`.

Physical regression-fix follow-up (2026-09-09): Inline Shadow now preserves a
spill-aliased scalar VGLOBAL atomic base before materializing its detector
address. The new forced-spill device regression passes in both the gfx950
simulator and on the physical MI350X, and the canonical HIP Stream-K simple
Inline row is exact and complete with zero diagnostics and 32/32 accesses,
3/3 barriers, 2/2 atomics, and 2/2 fences. Sampled now treats a stable pending
acquire owned by another launch/workgroup as a bounded table collision rather
than malformed evidence; its host collision regression passes and the
canonical physical two-tile row is exact and complete with zero diagnostics
and 80/80 accesses, 5/5 barriers, 2/2 atomics, and 2/2 fences. Both accepted
artifacts are under
`/tmp/consan-validation-gfx950-streamk-regression-fixes-20260909`. Inline's
two-tile cell remains yellow: the unmodified external corpus intentionally
runs six full-device dispatches in one executable-owned bounded report
lifetime, so it remains a disclosed capacity stress result rather than a
semantic regression. `rocjitsu-test-corpus` remains unmodified.

Current-toolchain physical production follow-up (2026-09-10) is based on
rocm-systems `e642922cebe8` plus the ConSan fixes in this change, the TheRock
SDK `10.1.0a20260909`, rocjitsu-test-corpus `c00e52c015f2`, IREE test suites
`49f46d6d4370`, and rocm-libraries `512b7a5c1d7e`. On the physical MI350X, the
canonical validator accepts baseline and all four clean profiles for TP1
prefill, TP1 decode/combined, all three TP2 rows, CLIP BF16, rocBLAS SGEMM, and
the refreshed target-native Tensile SGEMM. TP1 has complete 176/176 and
352/352 access coverage; each TP2 row has 544/544 accesses and each
sync-tracking engine has 60/60 barriers; CLIP has 45/45 accesses and 24/24
barriers; rocBLAS has 49,435/49,435 accesses and 4,997/4,997 barriers; and
Tensile has 48/48 accesses and 9/9 barriers. CLIP exposed a physical-only
Sampled failure: its barrier cave carried borrowed VGPR private-spill state
across the guest `s_barrier`. Sampled now executes the guest barrier first and
keeps the full save/probe/restore transaction after it. The forced-spill host
regression passes all 188 Sampled tests, the affected physical Sampled device
slice passes 116/116, and the fixed CLIP matrix is retained under
`/tmp/consan-validation-gfx950-more-e2e-20260910-clip-all-fixed`.

The paired rocBLAS baseline is 328.195 ms; measured ratios are 43.233x for
SuperCollider, 108.281x for Record/Replay, 86.153x for Sampled, and 90.074x for
Inline Shadow. The artifact is
`/tmp/consan-validation-gfx950-more-e2e-20260910-rocblas-overhead`. Current
Tensile v5 required integer `BoundsCheck` and `DirectToLds` schema values, a
one-row positive timing threshold, and a measured physical Inline bound. A
fresh 48-site inventory selected the reviewed target-native `ds_write_b128`
wrong-address mutation at kernel-relative PC `0x13f0`. The exact oracle fails
under all four mutations: SuperCollider, Record/Replay, and Inline Shadow emit
detector-owned diagnostics, while Sampled retains its reviewed miss. All four
policies are accepted with paired pre/post GPU health checks. Clean evidence is
under
`/tmp/consan-validation-gfx950-more-e2e-20260910-tensile-clean-fixed3` and
`/tmp/consan-validation-gfx950-more-e2e-20260910-tensile-inline-fixed4`;
contained fault evidence is under
`/tmp/consan-validation-gfx950-more-e2e-20260910-tensile-fault2`,
`/tmp/consan-validation-gfx950-more-e2e-20260910-tensile-fault-rr-confirm`, and
`/tmp/consan-validation-gfx950-more-e2e-20260910-tensile-fault-inline-confirm`.
The exact gfx950 PyTorch companion
`amd-torch-device-gfx950==2.15.0a0+rocm10.1.0a20260909` is installed in the
TheRock virtual environment. Its runtime identity and uninstrumented baseline
now pass; the validation driver extracts the identity from an explicit sentinel
so unrelated import noise cannot corrupt provenance. The current HipKittens
FP8 checkout still fails its external compile because
`ROTATING_BUFFER_COUNT` is not a constant expression. Neither external source
tree was modified.

Physical Qwen Record/Replay follow-up (2026-09-10): the canonical complete
151,936-logit oracle passes on the MI350X with zero diagnostics, complete static
and dynamic analysis, 844/844 accesses, 40/40 barriers, and 3,806,836 replay
input accesses in 98.85 seconds. The target manifest now retains the same
explicit 900-second bound used by the slower gfx950 emulator diagnostic, so
physical validation no longer needs a command-line timeout override. The
accepted artifact is
`/tmp/consan-validation-gfx950-qwen-rr-physical-20260910-b`; its executed code
object fingerprint is `fnv1a64:fb6a21ba7d0a67b4`.

Physical rocBLAS fault qualification (2026-09-10): a fresh 4,997-site
target-native inventory plus `TENSILE_DB=0x8000` dispatch tracing identified
the exact 64x64x32 MFMA specialization selected by `Square_64x64`. Its first
unconditional `s_barrier`, at kernel-relative `.text+0x518`, follows the
initial cooperative LDS publication and wait and immediately precedes peer
`ds_read_b32` operations. The precommitted exact-one drop is accepted in all
four contained trials: Record/Replay and Inline Shadow emit detector-owned
diagnostics, while SuperCollider and Sampled retain reviewed deterministic-
schedule misses. The exact SGEMM oracle remains correct in every trial, and all
paired pre/post MI350X health checks pass. The accepted artifact is
`/tmp/consan-validation-gfx950-rocblas-fault-20260910-b`; the reviewed spec has
SHA-256 `8ea0132c853b3d35c3ebfa5e139eee970c78bd529113db0f0447f62b6ac55c70`.

The current physical HIP Stream-K two-tile Inline Shadow row was rerun after
the dispatch-bank and regression fixes. Its exact numerical oracle and complete
80/80 access, 5/5 barrier, 2/2 atomic, and 2/2 fence transformation still pass,
while the unmodified runner's six 256-CTA launches leave 1,814,097 dynamically
incomplete events in the executable-lifetime 256-bank exact shadow. This is an
improvement over the earlier 3,272,268-event result but remains a disclosed
capacity limit, not a clean verdict. A separately rejected 32-CTA experiment
kept two CTAs per output tile and passed the oracle, but increased incompleteness
to 92,533,034 because longer per-CTA polling amplified bank collisions; the
canonical full-device command is unchanged. Artifacts are
`/tmp/consan-validation-gfx950-streamk-inline-current-20260910-a` and
`/tmp/consan-validation-gfx950-streamk-inline-grid32-20260910-a`.

Current physical PyTorch refresh (2026-09-10): all four `scatter_reduce`
profiles pass the exact BF16/FP32 oracle with complete static and dynamic
coverage of 27/27 accesses in 6.02--9.43 seconds. For the strengthened
4,096-wide norm/softmax workload, SuperCollider, Sampled, and Inline Shadow
pass the exact norm/CPU-softmax oracle with complete 4,880/4,880 access
coverage; Sampled and Inline Shadow also cover 2,132/2,132 barriers, and Inline
Shadow retains 232 visible events. Record/Replay reaches the same exact oracle
at 33.18 seconds but hits the deliberately short 60-second validation cap
before its final analysis verdict, so that cell remains yellow. Artifacts are
`/tmp/consan-validation-gfx950-scatter-all-physical-20260910-a` and
`/tmp/consan-validation-gfx950-norm-all-physical-20260910-a`.

The same refresh found current-object coverage regressions in three
`torch.mode` profiles. Record/Replay, Sampled, and Inline Shadow preserve the
exact value/index oracle, but respectively cover 26,398/26,426,
25,704/26,426, and 26,314/26,426 accesses; Sampled also covers only
2,659/4,359 barriers. The incomplete clean profiles are localized rather than
unexplained:
Record/Replay has 28 access resource failures, all `no_legal_window`; Inline
Shadow has 112 such access resource failures; Sampled has 22 access resource
failures plus 700 access placement/lowering failures, and 1,652 barrier
resource failures plus 48 barrier placement/lowering failures.
The artifact is `/tmp/consan-validation-gfx950-mode-all-physical-20260910-a`.
The SuperCollider rejection was fixed without relaxing final validation: the
text-relocation proof now records the exact per-descriptor before/after
`COMPUTE_PGM_RSRC1` values when DBT must raise a non-target kernel's SGPR
allocation to place its long-entry pair above the source SGPR extent. Final
validation accepts only those exact proof-owned SGPR-granule changes. The
physical rerun preserves the exact value/index oracle with complete static and
dynamic analysis of 25,366/25,366 accesses in 10.68 seconds; its artifact is
`/tmp/consan-validation-gfx950-mode-sc-relocation-proof-20260910-a`.
`torch.topk`
Record/Replay transforms with complete 231,322/231,322 access and
11,423/11,423 barrier coverage, but the physical run terminates after 188.81
seconds with `HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION` before the exact
oracle. Its artifact is
`/tmp/consan-validation-gfx950-topk-rr-physical-20260910-b`. A follow-up with
the scalar-restore dependency-wait repair was stopped early to honor the short
host window; no verdict is claimed for that interrupted run.

The final short physical regression gate passed the two focused
Record/Replay spill contracts (`DispatchIdSgprPressureForcedSpill` and
`PrivateEntryAbiSpill`) 2/2. A complete 607-row physical matrix was not allowed
to consume the remaining host window: its first four high-pressure
Record/Replay members passed 4/4 before the run was intentionally stopped.

Additional short physical PyTorch refresh (2026-09-10, source
`f3ef6f6273`): all four `torch.histc` profiles pass their exact FP32/FP64 bin
oracles with complete static and dynamic analysis. SuperCollider covers
110/110 accesses in 1.39 seconds; Record/Replay, Sampled, and Inline Shadow
each cover 152/152 accesses and 84/84 barriers in 7.78, 1.44, and 4.35 seconds.
The artifact is
`/tmp/consan-validation-gfx950-histc-all-physical-20260910-a`.
`torch.sort` SuperCollider also passes its exact sorted-value/index oracle with
complete 53,064/53,064 access coverage in 39.13 seconds. Record/Replay reaches
the deliberately short 45-second cap while transforming the main code object,
before the oracle or final analysis verdict; Sampled reaches the same bounded
outcome. Both remain yellow without retries, and Inline Shadow was not started
because it has the same large-object setup cost. Artifacts are
`/tmp/consan-validation-gfx950-sort-sc-physical-20260910-a`,
`/tmp/consan-validation-gfx950-sort-rr-physical-20260910-a`, and
`/tmp/consan-validation-gfx950-sort-sampled-physical-20260910-a`.

Status snapshot: 2026-08-26. All rows execute on the physical gfx950. This balanced clean-tree refresh used source `e355d1479e` and artifacts under `/home/ossci/xx/consan-validation/production-design-revalidation-20260826-*`; the full physical device-test suite passed 587/587. Retained green cells keep their previously accepted paired-overhead and reviewed-fault evidence unless stated otherwise.

Post-snapshot gfx950 emulation revalidation is identified explicitly in the affected cells. Source `1274082957a` on the gfx1201 host completed the SuperCollider clean contracts for `pytorch-torch-mode` with 24,179/24,179, `pytorch-torch-histc` with 137/137 eligible non-atomic LDS accesses, `pytorch-norm-softmax` with 4,820/4,820 accesses, and `pytorch-scatter-reduce` with 27/27 accesses. Artifacts are under `/tmp/consan-validation-gfx950-sc-subword-complete-20260905-f`, `/tmp/consan-validation-gfx950-sc-histc-20260905-a`, `/tmp/consan-validation-gfx950-sc-norm-softmax-20260905-a`, and `/tmp/consan-validation-gfx950-sc-scatter-reduce-20260905-a` respectively. Source `b8cc03b6cac` additionally completed the `pytorch-torch-sort` SuperCollider contract with its exact sorted-values-and-indices oracle and 56,884/56,884 accesses; its artifact is `/tmp/consan-validation-gfx950-sc-sort-split-reloc-fix-20260906-a`. Source `b80527824b0`, incorporating the subsequent large-object scaling work, completed the neighboring `pytorch-torch-topk` SuperCollider contract in 138.39 seconds with its exact oracle and complete static and dynamic coverage of all 239,442 accesses; its artifact is `/tmp/consan-validation-gfx950-supercollider-topk-20260906-b`. Source `ea99608b166` completed the Sharktank `tp1-prefill` SuperCollider contract with its exact numerical oracle and 176/176 accesses after adding exact Direct-to-LDS observation; its artifact is `/tmp/consan-validation-gfx950-sc-tp1-direct-lds-fix-20260906-b`. Source `528236c46fc` then completed both exact `tp1-decode-combined` oracles with 352/352 accesses and all three exact `tp2-family` oracles with 1,632/1,632 accesses; the artifacts are `/tmp/consan-validation-gfx950-sc-tp1-decode-direct-lds-fix-20260906-a` and `/tmp/consan-validation-gfx950-sc-tp2-direct-lds-fix-20260906-b`. Source `6b6823a0678` promotes TP2's proven 300-second execution bound into the target manifest. Source `35f1b0312ef` revalidated the Record/Replay `streamk-arrival` and `tree-atomic-or` contracts as exact and complete; artifacts are `/tmp/consan-validation-gfx950-rr-streamk-arrival-refresh-20260906-b` and `/tmp/consan-validation-gfx950-rr-tree-atomic-refresh-20260906-a`. Source `9cf729906b4` completed the Sampled `tree-atomic-or` contract by routing entry barriers through the epoch-only path and retaining single-role atomic associations over redundant acquire-release halves; `/tmp/consan-validation-gfx950-sampled-tree-atomic-final-20260906-a` records the clean exact run, and `/tmp/consan-validation-gfx950-sampled-streamk-role-regression-20260906-a` records the neighboring clean regression run. The exact-tree nonphysical gate then passed 4,693 rows and had one contention-only timeout among 4,694 registrations; that gfx1250 InlineShadow row passed serially in 43.83 seconds, and `f10f539616b` raises its full-gate allowance from 120 to 240 seconds. Source `4a64b565e59` then made successful lane-narrow InlineShadow acquires persist their advanced automatic-private epoch across the complete wave. That completed the exact `streamk-arrival` and `tree-atomic-or` gfx950 InlineShadow contracts in `/tmp/consan-validation-gfx950-inline-streamk-private-epoch-20260906-a` and `/tmp/consan-validation-gfx950-inline-tree-private-epoch-20260906-a`; the following full nonphysical ConSan gate passed all 4,694 registrations at `-j16`. Source `0815c2081cd` prevents spill-backed InlineShadow scalar state from aliasing scalar ordered-atomic address operands. The Release-built HIP Stream-K simple object now transforms with complete 32/32 access, 3/3 barrier, and 2/2 atomic/fence coverage; a 300-second emulator run remained execution-time-limited before the oracle verdict. The artifact is `/tmp/consan-validation-gfx950-inline-streamk-simple-20260906-d`, and the following full nonphysical ConSan gate passed all 4,695 registrations at `-j16`. Source `8d144fa75cb` then moved fixed-stack, spill-backed scalar preservation from a lane reservoir to private memory, preventing `v_writelane` from corrupting inactive guest lanes that later reconverge. The complete `pytorch-torch-topk` InlineShadow row now passes its exact FP64 and BF16 value/index oracles with zero diagnostics and complete static and dynamic coverage of all 239,730 accesses and 11,423 barriers in 391.97 seconds; its artifact is `/tmp/consan-validation-gfx950-inline-topk-private-scalar-fix-20260906-b`. The following complete nonphysical ConSan gate passed all 4,698 registrations at `-j16` in 274.98 seconds, including 2,918 simulator rows over all five targets. Source `41636fe8b15` also revalidated the complete `pytorch-torch-topk` Sampled contract: exact FP64 and BF16 value/index oracles, zero diagnostics, and complete static and dynamic coverage of all 239,730 accesses and 11,423 barriers in 228.82 seconds. Its artifact is `/tmp/consan-validation-gfx950-sampled-topk-refresh-20260906-a`. Source `8ccdc14fe58` revalidated `pytorch-torch-mode` InlineShadow as exact and complete within the retained 120-second manifest bound, then qualified an exact-one bitonic-stage barrier drop as a reviewed deterministic-schedule miss with the exact oracle preserved. The artifacts are `/tmp/consan-validation-gfx950-inline-mode-manifest-bound-20260906-a` and `/tmp/consan-validation-gfx950-mode-fault-20260906-b`. Source `9aacbf97dc9` revalidated `pytorch-torch-sort` Inline Shadow with its exact sorted-value/index oracle and complete 56,884/56,884 access and 6,032/6,032 barrier coverage in 198.10 seconds; `/tmp/consan-validation-gfx950-inline-sort-refresh-20260906-b` retains the result, and the target manifest now carries a 300-second emulator bound. Source `8e257b0294b` qualifies an exact-one key-load retirement barrier drop in that same executed radix-sort specialization. Inline Shadow produces a detector-owned diagnostic while retaining the exact oracle; Sampled at stride one records a reviewed deterministic-schedule miss with the exact oracle intact. The artifacts are `/tmp/consan-validation-gfx950-sort-inline-fault-20260906-a` and `/tmp/consan-validation-gfx950-sort-sampled-fault-20260906-b`.

Source `962d2e83622` revalidated `pytorch-norm-softmax` Sampled with its exact norm and CPU-softmax oracle, zero forbidden diagnostics, and complete static and dynamic coverage of all 4,820 accesses and 2,096 barriers in 38.59 seconds. Its artifact is `/tmp/consan-validation-gfx950-sampled-norm-softmax-refresh-20260906-a`. Source `ffa2f24cf39` qualified the exact-one FP32 shared-bin initialization barrier drop for `pytorch-torch-histc` Inline Shadow as a reviewed deterministic-schedule miss with both FP32 and FP64 exact oracles preserved. Its artifact is `/tmp/consan-validation-gfx950-histc-inline-fault-20260906-b`. Source `cc29aa68947` qualifies an exact-one FP16 tile-publication barrier drop in the executed 256-thread HIP matmul kernel. Inline Shadow produces a detector-owned diagnostic; SuperCollider records a reviewed reached miss before the mutated runner's oracle. The artifacts are `/tmp/consan-validation-gfx950-hip-matmul-inline-fault-20260906-a` and `/tmp/consan-validation-gfx950-hip-matmul-sc-fault-20260906-b`.

Source `0824ba0f6c9` replaces the DBT rejection for a resource-growing canonical address-taken body with object-level descriptor reconciliation across every possible kernel caller. That lets `pytorch-norm-softmax` Inline Shadow transform and execute its exact norm and CPU-softmax oracle with complete 4,820/4,820 access and 2,096/2,096 barrier coverage, eliminating the previous 41 unsupported access sites. The production cadence still yields no dynamically visible Inline Shadow event, so the cell remains yellow for dynamic qualification. Its artifact is `/tmp/consan-validation-gfx950-inline-norm-softmax-canonical-resources-20260906-a`.

Source `edde624fe95` strengthens the shared exact norm/softmax workload to the smallest reduction width that selects an LDS-backed softmax implementation on both CDNA4 and CDNA5. The prior width selected a register-only gfx950 kernel, so its lack of Inline Shadow evidence was a workload coverage hole rather than a failed probe. At width 4,096, all four gfx950 profiles pass their unchanged exact oracle with complete static and dynamic analysis; Inline Shadow records 232 visible events with 4,820/4,820 accesses and 2,096/2,096 barriers in 36.73 seconds. The all-profile artifact is `/tmp/consan-validation-gfx950-all-norm-softmax-4096-20260906-a`.

The same canonical-body resource reconciliation removes the remaining static Record/Replay gap in `pytorch-torch-topk`: both large code objects transform with zero unsupported sites and complete aggregate coverage of all 239,730 accesses and 11,423 barriers. The emulated workload reached its 900-second diagnostic bound before the exact oracle and teardown verdict, so the cell remains yellow for execution scaling while retaining its earlier exact result. The new artifact is `/tmp/consan-validation-gfx950-rr-topk-canonical-resources-20260906-a`.

Source `b9e3f16c66b` revalidated all three Sharktank TP2 Record/Replay rows under their target-resolved bounded validation cadences. Prefill, decode, and combined each passed their exact two-rank oracle with zero forbidden diagnostics, complete dynamic replay evidence, and complete static coverage of 544/544 accesses and 60/60 barriers. The artifacts are `/tmp/consan-validation-gfx950-rr-tp2-prefill-refresh-20260906-b`, `/tmp/consan-validation-gfx950-rr-tp2-decode-refresh-20260906-a`, and `/tmp/consan-validation-gfx950-rr-tp2-combined-refresh-20260906-a`.

Source `ac9dbdcf0e0` completed the split Sharktank TP2 Inline Shadow prefill row in gfx950 emulation. Its exact two-rank oracle passed with zero forbidden diagnostics and complete static and dynamic coverage of 544/544 accesses and 60/60 barriers in 1,090.86 seconds; `/tmp/consan-validation-gfx950-inline-tp2-prefill-current-20260906-b` retains the evidence. Source `e1b386dbc60` records the measured 1,800-second target bound. Decode and combined remain pending because their substantially heavier emulated execution still requires a scalable qualification path.

Source `ea289c0ff1c` records a 120-second gfx950 emulator bound for the complete `pytorch-torch-sort` Record/Replay row. Under that manifest-resolved bound, the exact sorted-value/index oracle passes with zero diagnostics, 93,554 replay inputs, and complete static and dynamic coverage of all 56,884 accesses and 6,032 barriers in 68.02 seconds. Its artifact is `/tmp/consan-validation-gfx950-rr-sort-manifest-bound-20260906-a`.

Source `c10ef55e8d9` refreshes the Qwen Inline Shadow fault/containment qualification against the current gfx950 object. The reviewed exact-one mutation drops the unconditional publication barrier in `main$async_dispatch_14_matmul_like_5x1024x2048_f32` between replacement-tile LDS stores and the following peer reads. Inline Shadow emits the required detector-owned diagnostic while the independent 151,936-output oracle remains exact; mutation installation, static and dynamic analysis of the filtered kernel, and both paired emulator health checks are complete. Its artifact is `/tmp/consan-validation-gfx950-qwen-inline-fault-20260906-c`.

Source `c3403ee0c1a` revalidated the full Qwen Record/Replay production cadence in gfx950 emulation with a 900-second diagnostic envelope. The workload remained in emulated model execution at the bound and consequently produced neither its exact oracle nor its teardown verdict. The retained exact clean result therefore remains yellow for execution scaling; the current artifact is `/tmp/consan-validation-gfx950-rr-qwen-production-refresh-20260906-a`.

The same source revalidated the HIP Stream-K two-tile Inline Shadow row. Its unchanged canonical command remained in emulated inter-CTA execution at a 600-second diagnostic bound, before the numerical oracle or final analysis verdict. This makes execution scaling the current qualification blocker in addition to the retained incomplete dynamic result; the artifact is `/tmp/consan-validation-gfx950-inline-streamk-two-tile-refresh-20260906-b`.

Fresh gfx950 inventories also qualified the first unconditional prologue tile-publication barriers in the exact HipKittens BF16, FP8, and MXFP8 kernels. BF16 retained the preselected exact-one mutation as a qualified miss in both SuperCollider and Inline Shadow; the accepted contained artifacts are `/tmp/consan-validation-gfx950-hk-bf16-sc-fault-20260906-c` and `/tmp/consan-validation-gfx950-hk-bf16-inline-fault-20260906-c`. FP8 and MXFP8 retained qualified-miss outcomes in SuperCollider and produced Inline Shadow diagnostics for their identical preselected mutations; the accepted FP8 artifacts are `/tmp/consan-validation-gfx950-hk-fp8-sc-fault-20260906-a` and `/tmp/consan-validation-gfx950-hk-fp8-inline-fault-20260906-a`, and the MXFP8 artifacts are `/tmp/consan-validation-gfx950-hk-mxfp8-sc-fault-20260906-c` and `/tmp/consan-validation-gfx950-hk-mxfp8-inline-fault-20260906-a`. Each contained run installed exactly one mutation and passed its paired gfx950 discovery and independent HIP-matmul health checks.

The same procedure qualified the first unconditional initial-tile publication barriers in the exact HIP Stream-K simple and two-tile kernels. SuperCollider retained each preselected exact-one mutation as a qualified miss under the deterministic gfx950 schedule, with paired discovery and independent HIP-matmul checks passing. The artifacts are `/tmp/consan-validation-gfx950-streamk-simple-sc-fault-20260906-a` and `/tmp/consan-validation-gfx950-streamk-two-tile-sc-fault-20260906-a`.

Legend: 🩶 unseen · 🟥 broken before useful evidence · 🟧 below 80% aggregate applicable-site support or another substantial gap · 🟨 timeout-only blocker or at least 80% aggregate applicable-site support · 🟩 accepted workload/profile contract.

| Set | Priority | Workload / validation ID | SuperCollider | Record/Replay | Sampled | Inline Shadow |
|---|---:|---|---|---|---|---|
| Main E2E | P0 | Qwen3-0.6B prefill (`qwen-prefill`) | 🟩 exact; 844/844 accesses | 🟩 physical exact/complete with zero diagnostics; 844/844 accesses, 40/40 barriers, and 3,806,836 replay inputs; 98.85 s | 🟩 exact; 844/844 accesses and 37/37 barriers | 🟩 exact/complete clean row; current-object exact-one publication-barrier fault detected under paired containment |
| Main E2E | P1 | Sharktank TP1 prefill (`tp1-prefill`) | 🟩 physical clean revalidated; exact/complete; 176/176 accesses | 🟩 physical clean revalidated; exact; 176/176 accesses and 31/31 barriers | 🟩 physical clean revalidated; exact; 176/176 accesses and 28/28 barriers | 🟩 physical clean revalidated; exact; 176/176 accesses and 31/31 barriers |
| Main E2E | P1 | Sharktank TP1 decode/combined (`tp1-decode-combined`) | 🟩 physical clean revalidated; exact/complete; 352/352 accesses | 🟩 physical clean revalidated; exact; 352/352 accesses and 62/62 barriers | 🟩 physical clean revalidated; exact; 352/352 accesses and 56/56 barriers | 🟩 physical clean revalidated; exact; 352/352 accesses and 62/62 barriers |
| Main E2E | P2 | Sharktank TP2 prefill/decode/combined (`tp2-family`, `tp2-decode`, `tp2-combined`) | 🟩 physical clean revalidated; three exact oracles; each row has 544/544 accesses | 🟩 physical clean revalidated; three exact oracles; each row has 544/544 accesses and 60/60 barriers | 🟩 physical clean revalidated; three exact oracles; each row has 544/544 accesses and 60/60 barriers | 🟩 physical clean revalidated; three exact oracles with zero forbidden diagnostics; each row has 544/544 accesses and 60/60 barriers |
| Main E2E | P3 | Sharktank CLIP BF16 (`clip-bf16`) | 🟩 physical clean revalidated; exact; 45/45 accesses | 🟩 physical clean revalidated; exact; 45/45 accesses and 24/24 barriers | 🟩 physical clean revalidated after barrier-spill repair; exact; 45/45 accesses and 24/24 barriers | 🟩 physical clean revalidated; exact; 45/45 accesses and 24/24 barriers |
| Main E2E | P4 | hip-moi D128 block (`d128-block`) | 🟩 physical clean revalidated; exact; 122/122 accesses | 🟩 physical clean revalidated; exact; 122/122 accesses and 119/119 barriers | 🟩 physical clean revalidated; exact; 122/122 accesses and 119/119 barriers | 🟩 physical clean revalidated; exact; 122/122 accesses and 119/119 barriers |
| Main E2E | P4 | hip-moi D128 pressure (`d128-pressure`) | 🟩 physical clean revalidated; four exact oracles; 236/236 accesses; retained paired/fault bundle | 🟩 physical clean revalidated; exact; 236/236 accesses and 28/28 barriers; retained paired/fault bundle | 🟩 physical clean revalidated; exact; 236/236 accesses and 28/28 barriers; retained paired/fault bundle | 🟩 physical clean revalidated; exact; 236/236 accesses and 28/28 barriers; retained paired/fault bundle |
| Main E2E | P4 | hip-moi MFMA attention (`wmma-attention`) | 🟩 physical clean revalidated; exact; 50/50 accesses; retained paired/fault bundle | 🟩 physical clean revalidated; exact; 50/50 accesses and 14/14 barriers; retained paired/fault bundle | 🟩 physical clean revalidated; exact; 50/50 accesses and 14/14 barriers; retained paired/fault bundle | 🟩 physical clean revalidated; exact; 50/50 accesses and 14/14 barriers; retained paired/fault bundle |
| Main E2E | P4 | hip-moi Stream-K arrival (`streamk-arrival`) | 🟩 physical clean revalidated; exact; 32/32 accesses | 🟩 physical clean revalidated; exact/complete; 32/32 accesses, 6/6 barriers, 1/1 atomic, 2/2 fences | 🟩 physical clean revalidated; exact/complete; 32/32 accesses, 6/6 barriers, 1/1 atomic, 2/2 fences | 🟩 physical clean revalidated; exact/complete with zero forbidden diagnostics; 32/32 accesses, 6/6 barriers, 1/1 atomic, 2/2 fences |
| Main E2E | P4 | hip-moi tree atomic-OR (`tree-atomic-or`) | 🟩 physical clean revalidated; exact; 48/48 accesses | 🟩 physical clean revalidated after private-entry ABI repair; exact/complete with zero diagnostics; 48/48 accesses, 6/6 barriers, 3/3 atomics, 5/5 fences | 🟩 physical clean revalidated; exact/complete; 48/48 accesses, 6/6 barriers, 3/3 atomics, 5/5 fences | 🟩 physical clean revalidated; exact/complete with zero forbidden diagnostics; 48/48 accesses, 6/6 barriers, 3/3 atomics, 5/5 fences |
| Main E2E | P4 | hip-moi Jakub attention (`jakub-attention`) | 🟩 four exact oracles; 338/338 accesses | 🟩 exact; 338/338 accesses and 35/35 barriers | 🟩 exact; 338/338 accesses and 35/35 barriers | 🟩 exact; 338/338 accesses and 35/35 barriers |
| Test corpus | P0 | HIP matmul 128 cubed (`hip-matmul-m128-n128-k128`) | 🟩 physical clean revalidated; exact; 739/739 accesses; retained qualified exact-one FP16 tile-publication miss | 🟩 physical clean revalidated; exact; 739/739 accesses and 109/109 barriers | 🟩 physical clean revalidated; exact; 739/739 accesses and 109/109 barriers | 🟩 physical clean revalidated; exact; 739/739 accesses and 109/109 barriers; retained exact-one FP16 tile-publication fault diagnosis |
| Test corpus | P0 | HipKittens BF16 (`hipkittens-bf16fp32-16x32`) | 🟩 physical clean revalidated; exact; 128/128 accesses; retained qualified exact-one prologue-publication miss | 🟩 physical clean revalidated; exact/complete with zero diagnostics; 128/128 accesses and 32/32 barriers | 🟩 physical clean revalidated; exact/complete; 128/128 accesses and 32/32 barriers | 🟩 physical clean revalidated; exact/complete with zero forbidden diagnostics; 128/128 accesses and 32/32 barriers; retained qualified exact-one prologue-publication miss |
| Test corpus | P1 | HipKittens FP8 (`hipkittens-fp8fp32-4wave`) | 🟩 exact; 64/64 accesses; qualified exact-one prologue-publication miss | 🟩 exact; 96/96 accesses and 5/5 barriers | 🟩 exact; 96/96 accesses and 5/5 barriers | 🟩 exact; 96/96 accesses and 5/5 barriers; exact-one prologue-publication fault diagnosed |
| Test corpus | P1 | HipKittens MXFP8 (`hipkittens-mxfp8-4wave`) | 🟩 physical clean revalidated; exact; 96/96 accesses; retained qualified exact-one prologue-publication miss | 🟩 physical clean revalidated; exact/complete with zero diagnostics; 96/96 accesses and 5/5 barriers | 🟩 physical clean revalidated; exact/complete; 96/96 accesses and 5/5 barriers | 🟩 physical clean revalidated; exact/complete with zero forbidden diagnostics; 96/96 accesses and 5/5 barriers; retained exact-one prologue-publication fault diagnosis |
| Test corpus | P1 | HIP Stream-K simple (`hip-streamk-simple-m256-n256-k256`) | 🟩 physical clean revalidated; exact; 32/32 accesses; retained qualified exact-one initial-tile publication miss | 🟩 physical clean revalidated; exact/complete with zero diagnostics; 32/32 accesses, 3/3 barriers, 2/2 atomics, and 2/2 fences | 🟩 physical clean revalidated; exact/complete; 32/32 accesses, 3/3 barriers, 2/2 atomics, and 2/2 fences | 🟩 physical clean revalidated; exact/complete with zero diagnostics; 32/32 accesses, 3/3 barriers, 2/2 atomics, and 2/2 fences |
| Test corpus | P1 | HIP Stream-K two-tile (`hip-streamk-two-tile-m256-n256-k256`) | 🟩 physical clean revalidated; exact; 80/80 accesses; retained qualified exact-one initial-tile publication miss | 🟩 physical clean revalidated; exact/complete with zero diagnostics; 80/80 accesses, 5/5 barriers, 2/2 atomics, and 2/2 fences | 🟩 physical clean revalidated; exact/complete with zero diagnostics; 80/80 accesses, 5/5 barriers, 2/2 atomics, and 2/2 fences | 🟨 current physical exact oracle and static coverage complete at 80/80 accesses, 5/5 barriers, 2/2 atomics, and 2/2 fences; six full-device launches exceed the 256-bank executable-lifetime exact shadow with 1,814,097 incomplete events |
| Test corpus | P2 | rocBLAS SGEMM square-64 (`rocblas-sgemm-square-64`) | 🟩 physical exact/complete; 49,435/49,435 accesses; 43.233x paired overhead; qualified exact-one initial-tile publication miss | 🟩 physical exact/complete; 49,435/49,435 accesses and 4,997/4,997 barriers; 108.281x paired overhead; exact-one publication fault diagnosed | 🟩 physical exact/complete; 49,435/49,435 accesses and 4,997/4,997 barriers; 86.153x paired overhead; qualified exact-one publication miss | 🟩 physical exact/complete; 49,435/49,435 accesses and 4,997/4,997 barriers; 90.074x paired overhead; exact-one publication fault diagnosed |
| Tensile | P0 | gfx950 Stream-K SGEMM (`tensile-gfx950-lds-positive`) | 🟩 physical exact/complete; 48/48 accesses and 9/9 barriers; current-object wrong-address fault diagnosed | 🟩 physical exact/complete; 48/48 accesses and 9/9 barriers; current-object wrong-address fault diagnosed | 🟩 physical exact/complete; 48/48 accesses and 9/9 barriers; qualified current-object wrong-address miss | 🟩 physical exact/complete; 48/48 accesses and 9/9 barriers; current-object wrong-address fault diagnosed |
| PyTorch | P0 | `torch.mode` (`pytorch-torch-mode`) | 🟩 physical exact/complete after exact relocation-owned descriptor proof; 25,366/25,366 accesses; 10.68 s | 🟨 physical exact oracle and dynamic analysis complete; static access coverage 26,398/26,426, barriers 4,359/4,359 | 🟨 physical exact oracle and dynamic analysis complete; static coverage 25,704/26,426 accesses and 2,659/4,359 barriers | 🟨 physical exact oracle and dynamic analysis complete; static access coverage 26,314/26,426, barriers 4,359/4,359 |
| PyTorch | P0 | `torch.topk` (`pytorch-torch-topk`) | 🟩 gfx950 emulation exact/complete; 239,442/239,442 accesses; 138.39 s | 🟧 current physical transformation is complete at 231,322/231,322 accesses and 11,423/11,423 barriers, but execution aborts with an HSA memory-aperture violation before the oracle at 188.81 s; short follow-up intentionally stopped without verdict | 🟩 gfx950 emulation exact/complete with zero diagnostics; 239,730/239,730 accesses and 11,423/11,423 barriers; 228.82 s | 🟩 gfx950 emulation exact/complete with zero diagnostics; 239,730/239,730 accesses and 11,423/11,423 barriers; 391.97 s |
| PyTorch | P1 | `torch.sort` (`pytorch-torch-sort`) | 🟩 physical exact/complete; 53,064/53,064 accesses; 39.13 s | 🟨 current physical run reaches the intentional 45-s cap during main-object transformation, before oracle or final analysis; retained prior gfx950-emulation exact/complete result | 🟨 current physical run reaches the intentional 45-s cap during main-object transformation, before oracle or final analysis; retained prior exact/complete and qualified-fault evidence | 🟩 gfx950 emulation exact/complete with zero diagnostics; 56,884/56,884 accesses and 6,032/6,032 barriers; exact-one key-load retirement fault diagnosed; 198.10 s |
| PyTorch | P1 | `torch.histc` (`pytorch-torch-histc`) | 🟩 physical FP32/FP64 exact/complete; 110/110 accesses; 1.39 s | 🟩 physical FP32/FP64 exact/complete with zero diagnostics; 152/152 accesses and 84/84 barriers; 7.78 s | 🟩 physical FP32/FP64 exact/complete; 152/152 accesses and 84/84 barriers; 1.44 s | 🟩 physical FP32/FP64 exact/complete; 152/152 accesses and 84/84 barriers; retained fault evidence; 4.35 s |
| PyTorch | P2 | `scatter_reduce` (`pytorch-scatter-reduce`) | 🟩 physical BF16/FP32 exact/complete; 27/27 accesses; 6.02 s | 🟩 physical BF16/FP32 exact/complete with zero diagnostics; 27/27 accesses; 9.43 s | 🟩 physical BF16/FP32 exact/complete; 27/27 accesses; 7.81 s | 🟩 physical BF16/FP32 exact/complete; 27/27 accesses; retained fault evidence; 8.09 s |
| PyTorch | P2 | norm/softmax (`pytorch-norm-softmax`) | 🟩 physical exact/complete; 4,880/4,880 accesses; 26.54 s | 🟨 physical exact oracle reached at 33.18 s, but the intentionally short 60-s cap expired before the analysis/teardown verdict | 🟩 physical exact/complete; 4,880/4,880 accesses and 2,132/2,132 barriers; 30.95 s | 🟩 physical exact/complete with 232 visible events; 4,880/4,880 accesses and 2,132/2,132 barriers; 39.10 s |
