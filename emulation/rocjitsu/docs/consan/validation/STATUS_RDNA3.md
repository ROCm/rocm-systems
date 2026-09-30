# ConSan RDNA3 (`gfx1100`) status

The six hip-moi rows were requalified through the clean gate in RocJITsu
emulation on September 29, 2026, using native `gfx1100` code objects and
W7900-derived rocprofv3 allowlists. All six uninstrumented emulator baselines
passed. Ten of twelve instrumented clean rows passed; Default reported a
conflict on the Stream-K arrival fixture and timed out on the tree atomic-OR
fixture before its final analysis verdict. This campaign qualifies emulator
behavior only, not physical hardware.

Reproduce the clean rows with the maintained
[validation runner](../../../tests/dbi/consan/consan_validation.py) and the
linked allowlist procedure below. The validation registry resolves all six
hip-moi rows to target-native GFX11 fixtures and fails closed instead of
substituting another architecture. Clean passes remain yellow until reviewed
fault trials qualify the detector. The observed clean conflict is red, and the
clean timeout is orange.

Start any new revalidation with
[rocprofv3-based allowlist discovery and application](VALIDATION.md#first-step-for-revalidation-generate-and-apply-kernel-allowlists).
Apply this before retrying recorded timeouts or raising deadlines; update their
status only after new runs provide evidence.

Physical evidence uses native `gfx1100` code objects on a matching GPU;
simulator prerequisites use RocJITsu `configs/gfx1100_w7900.json`. Exact counts
belong to the binaries used for qualification and must be refreshed after
relevant source, toolchain, workload, emulator, or runtime changes.

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
| Broad E2E | P4 | hip-moi D128 block (`d128-block`) | 🟨 clean pass; access 278/278, barrier 138/138; fault trials pending | 🟨 clean pass; access 278/278; fault trials pending |
| Broad E2E | P4 | hip-moi D128 pressure (`d128-pressure`) | 🟨 clean pass; access 503/503, barrier 36/36; fault trials pending | 🟨 clean pass; access 503/503; fault trials pending |
| Broad E2E | P4 | hip-moi WMMA attention (`wmma-attention`) | 🟨 clean pass; access 115/115, barrier 18/18; fault trials pending | 🟨 clean pass; access 115/115; fault trials pending |
| Broad E2E | P4 | hip-moi Stream-K arrival (`streamk-arrival`) | 🟥 clean numerical pass, but 1 diagnostic/1 conflict; access 32/32, barrier 3/3, atomic 2/2 | 🟨 clean pass; access 32/32; fault trials pending |
| Broad E2E | P4 | hip-moi tree atomic-OR (`tree-atomic-or`) | 🟧 300-second clean timeout before final analysis verdict | 🟨 clean pass; access 32/32; fault trials pending |
| Broad E2E | P4 | hip-moi Jakub attention (`jakub-attention`) | 🟨 clean pass; access 320/320, barrier 13/13; fault trials pending | 🟨 clean pass; access 320/320; fault trials pending |
