# rj_isa_diff P0 Implementation Plan

> Executed inline (superpowers:executing-plans style) in the session that wrote it.
> Each task lists the exact interfaces and tests instead of full code bodies, because
> the author implements the plan immediately and does not hand it off.

**Goal:** Take an old/new gfx950 artifact pair (code object or raw words), locate the changed
straight-line SALU regions, derive an explicit entry/exit contract for each, and fuzz paired
inputs. Each case runs *only* the region, through RocJITsu's existing `execute_instruction`.

**Architecture:** There are two halves, and they share only the case directory format.
- The **diff extractor** decodes each kernel once through `BasicBlock::build_reachable`. It
  computes a precise effect/liveness model over `LocSet`, aligns the two instruction streams,
  classifies every change, and plans conservative single-block regions with contracts.
- The **executor** holds one isolated `ComputeUnitCore`/wave per side and does reset/restore,
  deterministic input generation and constraint checking, then compares the observable and
  frame state. It produces a result model with separated evidence levels, counterexample
  bundles, and replay.

**Scope split (user decision, 2026-11-15):** the LLVM side is a **separate task/plugin**. That
covers ingesting `.s` through LLVM MC and producing real old/new LLVM pairs from controlled
compiler switches. Here, artifacts are AMDGPU code objects or raw instruction words, and the
frontend interface (`ArtifactLoader`) is the plugin seam.

**Tech stack:** C++20, CMake/Ninja, GoogleTest, RocJITsu object libraries (`RJ_TOOL_OBJECT_LIBS`).
Builds and tests run on a remote build host, not on the authoring workstation.

---

## File structure

| file | responsibility |
|---|---|
| `tools/isa_diff/location.{h,cpp}` | `Loc` universe (s0..s101, VCC_LO/HI, EXEC_LO/HI, M0, SCC), `LocSet`, names/parse |
| `tools/isa_diff/effects.{h,cpp}` | P0 allowlist and the per-instruction `Effects` summary (reads, writes, rejection reason), built from operands |
| `tools/isa_diff/json.{h,cpp}` | Minimal JSON value, parser and writer, for case, contract and result files |
| `tools/isa_diff/artifact.{h,cpp}` | `Artifact` (bytes, sha256, kernel range, base pc, descriptor/metadata bytes), loaders for code objects and raw words |
| `tools/isa_diff/kernel_model.{h,cpp}` | `KernelModel`: CFG through `build_reachable`, linear instruction list, backward liveness, PC taint |
| `tools/isa_diff/diff.{h,cpp}` | LCS alignment of the two `KernelModel`s, `Hunk`s, change classification (instruction / control-flow / layout / metadata / unexplained) |
| `tools/isa_diff/contract.{h,cpp}` | `RegionContract` (+ JSON): R_in, constraints C, exit mapping, R_out, frame, the source of every item |
| `tools/isa_diff/region_planner.{h,cpp}` | hunks to `RegionPlan`s: single block, hunk merging, backward context, stop rules, unresolved list |
| `tools/isa_diff/machine_state.{h,cpp}` | `MachineState` (SGPRs, specials, frame) + JSON + diff |
| `tools/isa_diff/region_runner.{h,cpp}` | `RegionRunner`: owns CU/wave; `load(state)`, `run()` with classification, `capture()` |
| `tools/isa_diff/input_policy.{h,cpp}` | `InputPolicy` interface, `BaselinePolicy` (boundary/correlated/random), `ConstraintChecker` |
| `tools/isa_diff/differential.{h,cpp}` | Test loop, comparison, statistics, timing, result model, counterexamples |
| `tools/isa_diff/case_io.{h,cpp}` | Case directory read/write (`case.json`, `results.json`, `counterexamples/*.json`) |
| `tools/isa_diff/rj_isa_diff_main.cpp` | CLI `rj_isa_diff analyze/test/replay` |
| `tests/isa_diff/*.cpp` | Validator tests (§13 table) in the `rj_isa_diff_tests` gtest executable |

---

## Phase A: diff extractor

### Task A1: locations and effects
- `enum class LocKind {Sgpr, VccLo, VccHi, ExecLo, ExecHi, M0, Scc}`; `using LocSet = std::bitset<kNumLocs>`
  with `kNumLocs = 102 + 6`; `std::string loc_name(unsigned)`; `std::optional<unsigned> parse_loc(std::string_view)`.
- `struct Effects { LocSet reads, writes; bool p0_executable; std::string reject_reason; bool reads_pc; bool unknown_effect; }`
  and `Effects classify(const Instruction&)`.
- Classification order, per operand:
  1. fieldless with `to_special_reg_class()` (SCC/VCC/PC/EXEC/M0)
  2. `to_register_ref()` SGPR/EXEC
  3. selector `encoding_value()` 106/107 (VCC by `size_bits`), 124 (M0), 253 (`src_scc`)
  4. `const_value()` constant
  5. anything else is unknown, so the instruction is not P0 and its effect is unknown.
- Tests: decode the probe words and assert reads/writes for `s_add_u32`, `s_addc_u32` (reads SCC),
  `s_cselect_b64 …, vcc` (reads VCC_LO+HI+SCC), `s_mov_b32 vcc_hi` (writes VCC_HI only),
  `s_and_b64 exec` (EXEC pair + SCC). Assert that `s_getpc_b64`, `s_waitcnt`, `s_cbranch_vccz`,
  `s_load_dword` and `v_mov_b32` are rejected with distinct reasons.

### Task A2: artifact ingestion
- `struct Artifact { std::string path, sha256; std::vector<uint8_t> image; std::vector<uint32_t> text_words; uint64_t text_base; uint64_t kernel_begin, kernel_end; std::string kernel; std::vector<uint8_t> descriptor; std::vector<uint8_t> metadata; std::string kind; }`
- `FailureOr<Artifact> load_artifact(path, kernel_name)` sniffs the ELF magic and falls back to raw words.
  `Artifact artifact_from_words(std::vector<uint32_t>, std::string label)` is used by tests.
- Tests: a raw-words round trip, plus an ELF built with `make_minimal_amdgpu_kernel_elf`
  (`tests/amdgpu_elf_test_support.h`) loads its kernel range and descriptor.

### Task A3: kernel model
- `class KernelModel { static FailureOr<KernelModel> build(const Artifact&); std::span<const InstRef> insts(); LocSet live_before(i); LocSet live_after(i); bool pc_tainted_before(i, Loc); block_of(i); bool cfg_complete(); }`,
  where `InstRef { const Instruction *inst; uint64_t pc; std::span<const uint32_t> words; Effects fx; uint32_t block; bool is_leader; }`.
- Liveness transfer:
  - P0 instruction: `live = (live - writes) | reads`.
  - Non-P0 instruction: `live |= reads`, plus EXEC and VCC and M0 when it is not SALU. It never kills.
  - Unknown effect, unknown successor, or call edge: `live = ALL`.
  - Block with no successors (`s_endpgm`): empty.
- Tests:
  - SCC is live after `s_add_u32` when it is followed by `s_cselect` and dead when followed by `s_cmp`.
  - An unknown edge (`s_setpc_b64`) makes everything live.
  - A taint from `s_getpc_b64` propagates through `s_add_u32`.

### Task A4: alignment and classification
- `struct Hunk { uint32_t old_begin, old_end, new_begin, new_end; ChangeKind kind; std::string note; }`
  and `DiffResult diff(const KernelModel&, const KernelModel&, const Artifact&, const Artifact&)`.
- Key: raw words, except direct branches (simm16 masked). Matched branch targets are then
  validated to map to matched instructions, and a mismatch makes it a control-flow hunk.
- Also records:
  - metadata/descriptor differences
  - differing uncovered bytes
  - a layout shift in a kernel that contains PC-dependent instructions.
- Tests: identical input gives zero hunks; one changed instruction gives one instruction
  hunk; a moved instruction gives two hunks; a changed branch target gives a control-flow hunk;
  a descriptor byte change gives a metadata item.

### Task A5: contracts and region planner
- `PlanOptions { max_region_insts=256; max_expansions=64; context_insts=0; merge_same_block=true; }`.
- `struct RegionPlan { id; old/new inst ranges; RegionContract contract; status; reason; }`
  and `PlanResult plan_regions(...)`, which carries regions, unresolved items, and a coverage flag.
- Contract:
  - `entry_equal = live_before_old ∪ live_before_new`, with `canary = complement`.
  - `observable = live_after_old ∪ live_after_new`.
  - `frame` lists the fixed frame fields.
  - `exit` is the logical fall-through, so PCs need not match.
  - `status` is ESTABLISHED when nothing upstream differs and the metadata is identical,
    ASSUMED when the region depends on upstream regions or is manual, and UNRESOLVED when
    the descriptor or metadata changed.
- Stop rules: multi-block, unsupported instruction, PC taint, region too long, expansion
  budget exceeded, or an incomplete CFG at the region. Each one produces a typed reason.
- Tests: hunks in one block merge across common code; hunks in different blocks become two
  regions, with the second depending on the first; a `s_getpc`-tainted region and a VALU region
  are rejected; `context_insts=2` pulls in a common prefix.

## Phase B: executor

### Task B1: machine state and runner
- `struct MachineState { std::array<uint32_t,104> sgpr; uint32_t vcc_lo, vcc_hi, exec_lo, exec_hi, m0; bool scc; Frame frame; }`
  with JSON round trip and `first_difference(a, b, LocSet observable, compare_frame)`.
- `class RegionRunner { static FailureOr<RegionRunner> create(span<const RegionInst>); void load(const MachineState&); RunOutcome run(uint64_t budget); MachineState capture(); }`.
  It owns `GpuMemory`, `L2Cache`, `ComputeUnitCore` and the wave, and decodes the words once at
  creation. Per instruction it sets `pc`, runs `execute_instruction`, checks that `pc` is unchanged,
  checks the UB counter delta, and catches exceptions.
- `enum class RunStatus {Ok, DecodeFailure, MissingExecutionBackend, Unimplemented, InvalidRegisterAccess, HostException, UndefinedBehavior, PcWritten, BudgetExceeded}`.
- Tests:
  - executing `s_add_u32` gives the right result and SCC;
  - load→run→capture is deterministic over 1000 interleaved runs on two runners (no leak);
  - a model-only instruction (execute nulled) is classified, not treated as a no-op;
  - a decode failure is classified.

### Task B2: inputs and constraints
- `struct Sample { std::vector<std::pair<unsigned,uint32_t>> values; }`
- `class InputPolicy { virtual Sample propose(const RegionFeatures&, const RegionContract&, const History&) = 0; virtual void observe(const Feedback&) {} }`
- `class BaselinePolicy : InputPolicy` takes a seed and draws from:
  - boundary values (0, 1, -1, INT_MIN/MAX, single and inverted single bits, 0xAAAA…/0x5555…, shift edges 31/32/63);
  - 64-bit pair boundaries over pair-read locations;
  - equal or complementary operand copies;
  - SCC alternating by case parity and otherwise random.
- `ConstraintChecker::validate(sample, contract) -> std::optional<std::string> reject_reason`
  supports known-zero/known-one masks, unsigned ranges, and equalities.
- Tests: the same seed gives the same samples; contradictory constraints give 0 accepted;
  a range constraint only yields in-range values.

### Task B2b: mandatory boundary sweep (added after review)
- `boundary.{h,cpp}`: named catalogs (integer limits, byte/half edges, bit patterns,
  f32/f16/bf16/f64 special encodings, EXEC lane patterns) and the role-specific value sets.
- `roles.{h,cpp}`: operand roles from the decoded region (shift amount, bit index, S_BFE
  control, sign-extension input, signed edge, two-source pairs, register/constant
  relations) and the set of registers the region reads.
- `build_boundary_sweep()`: uniform, single, pair64, pairs and constraint-edge phases over
  zeros/ones/random bases (projected onto constraints), with both SCC values when SCC is
  an input. De-duplicated and deterministic. The cap is reported, never silent.
- `test_region()`: run the whole sweep before the random phase, counting each phase
  separately; `--cases` counts random cases only.
- Tests: catalog coverage per register, role targeting, an off-by-one against a literal
  caught with `--cases 0`, constraint edges and rejections, truncation reporting.

### Task B3: differential loop, results, counterexamples, replay
- `TestOptions { seed; cases; budget; max_counterexamples; }` and
  `RegionResult test_region(const RegionPlan&, const TestOptions&, InputPolicy&)`.
  For each case:
  1. propose a sample;
  2. validate it;
  3. build old and new initial states (same entry-equal values, per-side canaries, fixed frame);
  4. load, run and capture both;
  5. compare exit status, observable, and frame preservation.
- Result fields: `test_status`, `contract_status`, `witness_scope`, `coverage_scope`, `proof_status=NOT_RUN`,
  counts (attempted/accepted/executed/rejections by reason), timings (decode/plan/setup/old/new/compare),
  executed instruction counts per side, and counterexample paths.
- `Counterexample` JSON:
  - region words, PCs and mnemonics;
  - contract;
  - both initial states, seed and case index;
  - both traces (per-instruction changed locations);
  - the first difference;
  - tool, schema and RJ build identity.
- `replay(case, cex)` rebuilds the runners from the recorded words (no artifact, no LLVM) and
  reports whether the recorded outcome reproduced.
- Tests (see §13 below).

## Phase C: CLI, validator tests, measurement

### Task C1: CLI
- `rj_isa_diff analyze --old A --new B [--kernel K] [--context N] [--max-region N] [--no-merge] --out DIR`
- `rj_isa_diff test --case DIR [--seed S] [--cases N] [--region ID]`
- `rj_isa_diff replay --case DIR --counterexample FILE`
- Uses `rocjitsu/cli.h`. Exit codes: 0 means no mismatch with complete scope, 1 a mismatch,
  2 inconclusive/partial/unsupported, 3 a harness error.

### Task C2: validator tests (handoff §13), in `tests/isa_diff/`

| row | fixture | expected |
|---|---|---|
| identical region | same words | NO_MISMATCH_FOUND; repeat with no leak |
| dead temp differs | `s_mov_b32 s9,…` → `s_mov_b32 s9,…'`, s9 overwritten after | no false mismatch |
| live SCC differs | `s_add_u32` vs `s_or_b32` same value, then `s_cselect` | LOCAL_MISMATCH on SCC; dead-SCC pair with `s_cmp` after is clean |
| live-through clobber | new adds `s_mov_b32 s20, 0`, s20 read later | LOCAL_MISMATCH on s20 |
| 64-bit/half/overlap | `s_lshl_b64 s[4:5], s[4:5]`, `s_mov_b32 vcc_hi`, `s_add_u32 s4, s4, s5` | correct effects and comparison |
| reset/interleave | two runners, same seed twice | identical result JSON |
| constraint-dependent | `s_and_b32 s4,s4,0xff; s_lshr…` vs rewritten one that is valid only when s4<256 | unrestricted: LOCAL_MISMATCH; with `--context` or a constraint: NO_MISMATCH_FOUND, labelled ESTABLISHED or ASSUMED |
| contradictory constraint | known_zero & known_one overlap | INCONCLUSIVE, 0 valid cases |
| dependent hunks | two hunks with common code between, in one block | merged; not two independent passes |
| RA rename, multi-block | renamed regs across a branch | INCONCLUSIVE (mapping), not a pass |
| unknown successor / descriptor change / uncovered hunk | `s_setpc_b64`, descriptor bytes | coverage PARTIAL / NOT_ESTABLISHED |
| EXEC=0 / alias consistency | region `s_and_b64 exec` with EXEC=0 samples | executed, not filtered |
| waitcnt / memory / getpc | regions with these | UNSUPPORTED with reason |
| decode failure / model-only / unsupported opcode | bad words; execute nulled | classified errors |
| liveness change invalidates cache | same region bytes, later consumer reads a dead reg | different contract hash, so a different cache key |
| injected wrong opcode/immediate/operand | synthetic mutations | detected, replayable, labelled `synthetic` |

### Task C3: measurement and report
- Benchmark regions of 1, 4, 16, 64 and 256 instructions inside a larger kernel. Report per-phase
  cost, the executed instruction count against the kernel size, and cases per second.
- `findings.md` and a sample `results.json` go under `tools/isa_diff/docs/`.

## Deferred (separate tasks)
- **LLVM plugin:** a `.s` frontend through LLVM MC (an `ArtifactLoader` implementation), and producing
  real old/new pairs from controlled optimisation switches, plus the end-to-end report on them.
- P1: scalar branches, multi-exit regions, loops, SemaAST/SMT proof. P2: VALU, memory, pending effects, reachability.
