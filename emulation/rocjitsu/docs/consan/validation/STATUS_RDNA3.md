# ConSan RDNA3 (`gfx1100`) status

The six hip-moi rows below were physically requalified with native `gfx1100`
code objects on a Radeon PRO W7900 on September 30, 2026, after rebasing onto
the accelerated upstream ConSan inventory and analysis implementation.
Baseline, Default, and SuperCollider clean runs passed with complete applicable
coverage. The 15 fault campaigns covered 17 profile or operating-point
outcomes and 136 admitted and reached trials. Each outcome used eight trials;
the table records the selected profile controls, detection counts, and
lower-preset misses needed to interpret each grade.

Reproduce these rows with the maintained
[validation runner](../../../tests/dbi/consan/consan_validation.py), the
[reviewed `gfx1100` fault specification](../../../tests/dbi/consan/consan_validation_faults_gfx1100.json),
and the linked allowlist procedure below. The runner's `manifest` and `explain`
commands provide the current target-native workload commands, profile
environment, correctness oracles, and fault policy.

The validation registry resolves all six hip-moi rows to target-native GFX11
fixtures and fails closed instead of substituting another architecture. A
profile is green only when its clean and reviewed fault campaigns meet the
shared qualification contract.

Start any new revalidation with
[rocprofv3-based allowlist discovery and application](VALIDATION.md#first-step-for-revalidation-generate-and-apply-kernel-allowlists).
Apply this before retrying recorded timeouts or raising deadlines; update their
status only after new runs provide evidence.

Physical evidence uses native `gfx1100` code objects on a matching GPU;
simulator prerequisites use RocJITsu `configs/gfx1100_w7900.json`. Exact counts
belong to the binaries used for qualification and must be refreshed after
relevant source, toolchain, workload, emulator, or runtime changes.

The [native publication tests](../../../tests/dbi/consan/hip_consan_rdna3_test.hip)
check output preservation with and without instrumentation. The aligned
32-bit store case must produce a complete publication trace. Unaligned stores
execute their original instruction because observation through atomic exchange
requires natural alignment; they must set the sticky dropped marker so the
decoder rejects the trace as incomplete. The tests keep the event count below
capacity so overflow detection cannot mask a missing dropped marker.

Shared [color scale](VALIDATION.md#status-colors): 🟩 qualified; 🟨 clean run
established, but fault qualification is pending/below bar or the workload is
outside detector scope; 🟧 clean qualification blocked by prerequisites,
unsupported applicable operations, incomplete coverage/evidence, or a timeout;
🟥 observed correctness or instrumentation failure. Empty/🩶 means unassessed
for this execution target; simulator prerequisites alone do not qualify hardware.

| Set | Priority | Workload / validation ID | Default | SuperCollider |
| --- | ---: | --- | --- | --- |
| Physical compact gate | P0 | native two-wave LDS fixture (`ConSanGfx1100Physical.*`) | 🟨 exact clean output, visible records, zero diagnostics; reviewed conflict and broad campaign missing | 🟨 exact clean/all-sites rows and mutation containment; broad E2E fault campaign missing |
| Simulator prerequisite | P0 | native two-wave LDS fixture (`ConSanGfx1100Sim.*`) | 🩶 prerequisite passes | 🩶 prerequisite passes |
| Broad E2E | P0 | Qwen3-0.6B prefill (`qwen-prefill`) | 🩶 unassessed | 🩶 unassessed |
| Broad E2E | P1 | Sharktank TP1 prefill (`tp1-prefill`) | 🩶 unassessed | 🩶 unassessed |
| Broad E2E | P1 | Sharktank TP1 decode/combined (`tp1-decode-combined`) | 🩶 unassessed | 🩶 unassessed |
| Broad E2E | P2 | Sharktank TP2 family (`tp2-family`) | 🩶 unassessed | 🩶 unassessed |
| Broad E2E | P3 | Sharktank CLIP BF16 (`clip-bf16`) | 🩶 unassessed | 🩶 unassessed |
| Broad E2E | P4 | hip-moi D128 block (`d128-block`) | 🟩 high (lowest passing): clean pass; access 278/278, barrier 138/138; grouped K-publication fault 8/8 (`default` 2/8) | 🟩 sleep=15: clean pass; access 278/278; grouped K-publication fault 8/8 |
| Broad E2E | P4 | hip-moi D128 pressure (`d128-pressure`) | 🟩 high (lowest passing): clean pass; access 503/503, barrier 36/36; grouped K/V-publication fault 8/8 (`default` 4/8) | 🟨 sleep=15: clean pass; access 503/503; grouped K/V-publication fault 0/8 (bar 6/8) |
| Broad E2E | P4 | hip-moi WMMA attention (`wmma-attention`) | 🟩 high (lowest passing): clean pass; access 115/115, barrier 18/18; grouped K/V-publication fault 8/8 (`default` 4/8) | 🟩 sleep=15: clean pass; access 115/115; grouped K/V-publication fault 8/8 |
| Broad E2E | P4 | hip-moi Stream-K arrival (`streamk-arrival`) | 🟩 high: clean pass; access 14/14, barrier 1/1, atomic 1/1; release-order fault 8/8 | 🟨 delay=15: clean pass; access 14/14; release-order fault 0/8 (bar 6/8) |
| Broad E2E | P4 | hip-moi tree atomic-OR (`tree-atomic-or`) | 🟩 high: clean pass; access 18/18, barrier 1/1, atomic 2/2; producer release-order fault 8/8 | 🟨 delay=15: clean pass; access 18/18; producer release-order fault 0/8 (bar 6/8) |
| Broad E2E | P4 | hip-moi Jakub attention (`jakub-attention`) | 🟩 higher (lowest passing): clean pass; access 320/320, barrier 13/13; load-to-compute publication fault 8/8 (`default` 0/8, `high` 1/8) | 🟨 sleep=15: clean pass; access 320/320; load-to-compute publication fault 0/8 (bar 6/8) |
