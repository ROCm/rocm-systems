# Website, data, and publication

## Source map

Paths below are relative to `emulation/rocjitsu/website/dashboard/`.

```text
src/main.jsx → src/App.jsx             # start React; coordinate loading and pages
src/components/                       # pages, charts, tables, pickers
src/hooks/useDashboardState.js         # filters, selections, URL/history state
src/data/dashboardData.js             # fetch publication JSON
src/data/dashboardDataAdapter.js      # adapt corpus runs and derive memory-only sets
src/data/runSchema.js                 # corpus version and configuration identity
src/data/selectors.js                  # derive chart/comparison data
src/config/metadata.json              # bundled repository/Beta/canonical-branch settings
scripts/process-dashboard-data.mjs     # local-only consumer inspection
vite.config.js                        # build assets and select the data source
```

This is a static React website. The browser fetches JSON and computes the displayed comparisons; there is no dashboard application server in production.

## Display scaling

Desktop density follows browser-window width in CSS pixels, not the display's
physical pixel count or device pixel ratio. Equal-width windows therefore use
the same text and layout sizes on standard and HiDPI displays. At a 1920 CSS-pixel
width the scale is 0.7392: 14px text renders at about 10.35 CSS pixels. The scale
increases for wider windows; below 900 CSS pixels the interface stays native-sized.
Browser zoom can change the available CSS width and cause reflow. Popups use the
applied body density for inverse coordinate scaling and matching text sizes.

## Website deployment versus data publication

These are separate workflows and Git branches:

| What | Repository-root workflow | Destination |
| --- | --- | --- |
| Website HTML/JS/CSS | `.github/workflows/rocjitsu-publish-website.yml` | `gh-pages:rocjitsu-dashboard/` |
| Benchmark JSON | `.github/workflows/rocjitsu-benchmarks.yml` | `gh-pages-rocjitsu:rocjitsu-dashboard/data/` |

Website publication checks current `develop` against the last published source revision, verifies/builds changed sites, then publishes dashboard `dist/`. Manual publication also requires `develop`. Production assets do not contain benchmark fixtures or measurements. Keep fixture JSON out of both publication destinations. Updating bundled settings requires a website rebuild; publishing new measurements does not.

```text
Website URL:
https://rocm.github.io/rocm-systems/rocjitsu-dashboard/

Production JSON base URL:
https://raw.githubusercontent.com/ROCm/rocm-systems/refs/heads/gh-pages-rocjitsu/rocjitsu-dashboard/data/
```

The base URL comes from `scripts/dashboard-data-source.mjs`; `src/data/publishedDataUrls.js` constructs the index URL. Fixture/local-data modes use `./data/`: fixtures come from `tests/fixtures/data/`, or `DASHBOARD_DATA_DIR` supplies a local directory.

## Data and browser loading

Bundled `src/config/metadata.json` contains repository URL, Beta flag and canonical
branch, not a producer schema version. The data root contains only an index and
corpus run documents. Each run embeds flat measurements and definitions; there
are no external catalog resources.

```text
Fetch index.json → fetch indexed runs → strict JSON parsing
  → adapt corpus schema 1 and derive first-occurrence configuration membership
  → existing selectors → UI
```

Unindexed files are not scanned. The mutable index bypasses cache; immutable runs
can use cache. Reload All cache-busts requests without changing source values.
Failures of required identity/path/core inputs are visible data errors. Optional
metadata does not gate the dataset. The [corpus raw-run schema](https://github.com/ROCm/rocjitsu-test-corpus/blob/develop/benchmarks/raw-run-schema.md)
owns the input format. Nullish provenance values display as an em dash (`—`) in metadata tables;
source nulls stay unchanged. Missing or unusable commit dates disable dependent
chronology without replacing them with execution time. Download JSON preserves
the full parsed `{index,runs}`, including ordered samples, exit codes and
diagnostics, independently of filters.

## Preparing and publishing measurements

The benchmark workflow runs on relevant develop pushes or manual dispatch, not a scheduled trigger. It invokes the corpus `performance-v1.toml` manifest once with both target configs and no target, case, or thread-policy selector, using the manifest's sampling defaults. The selected corpus revision is defined by `CORPUS_SHA` in the [benchmark workflow](../../../../../.github/workflows/rocjitsu-benchmarks.yml). This configuration does not establish native or hosted execution acceptance. The runner selects supported combinations and checkpoints one consolidated `run.json`, alongside per-target/policy diagnostics. Before a side-branch benchmark starts, a lightweight job snapshots `develop` and records the dispatched commit's merge-base with that immutable revision. The optional `pull_request_number` input also records an open same-repository PR when its head branch and SHA match the selected workflow branch; without that input, the run keeps its base but has no PR metadata. Develop runs reject the PR input and omit both fields.

### Isolated preparation before measurement

The workflow creates a fresh owned `${RUNNER_TEMP}/rocjitsu-benchmark-${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT}`. `emulation/rocjitsu/scripts/prepare-benchmarks.py` keeps its venv, builds, wheelhouse, generated artifacts, logs and caches there, outside the runner-owned results directory. It rejects reused attempts, mismatched ownership and symbolic directory paths; cleanup removes only the verified owned attempt. Source checkouts remain unchanged and are checked before and after preparation phases.

Preparation uses Python 3.12, the corpus's exact SDK/benchmark requirement pins, full rocm-libraries source `dd77374194a3ea1a7258cd54f87af6747b6362a5`, and separate native nanobind source `9b3afa9dbdc23641daf26fadef7743e7127ff92f`. Dependency resolution records indexes, reports and package inventories while constraining the benchmark pins. Before builds, preflight requires CMake ≥3.25.2, SDK HIP/LLVMObjectYAML/ROCmCMakeBuildTools, ZLIB and OpenMP. A noneditable rocisa wheel uses an external backend build directory and the discovered installed `nanobind_DIR`; only the exact built wheel is installed into the owned venv. Native adapter and host-test builds instead use the separately pinned disconnected nanobind source. Builds use four jobs.

The explicit `ROCISA_BACKEND=native` and four SDK loader directories (`lib`, `lib/rocm_sysdeps/lib`, `lib/llvm/lib`, `lib/host-math/lib`) are retained. Preparation commands use an owned **writable** controlled working directory, not a source checkout or a read-only empty directory: the native capability probe can emit implicit `a.out`. Only that regular assembler-output filename is permitted there; foreign loader files reject. No upstream/ELF/compiler patch or speculative `GPU_TARGETS` rewrite is made. Wheel import checks record exact extension/library hashes, installed-wheel origin and loader mappings; this does not establish portable relocatability of the upstream wheel.

The helper runs the pinned generator sequentially for the four manifest-selected variants, with the exact selected shapes/targets and generator `CpuThreads=4`. It strips inherited `HCC_AMDGPU_TARGET` and sets that variable only in each generator's copied child environment, using the validated target for that variant. This suppresses the installed HIPCC's demonstrated implicit enumeration path without hard-coding a global gfx950 or leaking a target to unrelated preparation commands or later workflow steps. Command receipts record the effective target, not the whole inherited environment. It validates artifact/configuration hashes and provenance before exporting `PYTHON`, SDK/source paths, `TENSILE_CANDIDATE_ARTIFACTS` and `TENSILE_CANDIDATE_RUNNER`. Failed preparation prevents the measured suite; cases are never dropped and stale artifacts are never reused. Generator threads do not set simulator runtime policy. Local intercepted tests verify orchestration and rejection predicates, **not actual dependency installation, native generation, kernel execution or hosted acceptance**.

The existing 210-minute job ceiling remains: aggregate setup/checkouts/Rocjitsu build 25 minutes, dependency resolution 20, wheel/native preparation 45, sequential generation 45, performance 60, diagnostics/upload/cleanup reserve 15. Step limits and helper phase/process watchdogs fail rather than silently increase these ceilings; they are not completion-time predictions. On timeout the owned process group gets TERM, a bounded grace period, then KILL independently of leader exit; absent groups are handled safely, and failure receipts record whether the leader was reaped. Preparation diagnostics exclude files with symbolic ancestors or resolved paths outside the owned attempt and exclusively create a fresh archive without following or truncating existing output links/files. Preparation diagnostics have a separate attempt-specific artifact, 14-day retention, and an explicit inventory describing truncation/omission (at most 128 files, final 256 KiB per file). Upload handling precedes owned cleanup. Hard cancellation or job termination can prevent both upload and cleanup. Manual `publish_results=false` still disables the entire publisher job, not diagnostic handling.

```text
Prepare pinned dependencies, wheel, native adapter and four generated variants
  → validate preparation provenance and artifacts
  → run supported target/policy combinations in one corpus invocation
  → upload raw result directories as a GitHub Actions artifact
  → publish job downloads that artifact
  → fill publication context only
  → copy the run into the branch-appropriate results directory
  → structurally update index.json once
  → ordinary Git commit + push
```

Publication keeps raw measurements, ordered samples, diagnostics, exit codes and
provenance unchanged. Workflow/Tensile revisions belong in `publication`; source
messages describe the measured SHA. Missing optional base/PR/tool context remains
null. No catalog generation, downstream manifest check or dedicated validator
stage is part of this boundary. Local browser tests are not hosted publication
or release evidence.

## Immutable runs

The same path with identical bytes is reusable; different content at an existing
path is an error. A newly indexed path updates `generatedAt`; an already indexed
path leaves index bytes unchanged. Ordinary Git publication failures remain
visible; no force push or custom retry engine is implied.

Benchmark-set identity is `execution_summary.benchmark_suite`. This publisher
requires an immutable TOML recipe and fixed benchmark-ID membership for each
shared target/thread-policy configuration under that identity. A subset or
expanded recipe is a different benchmark set and requires a new suite ID. Failed
measurements remain members; failure does not remove a benchmark from the set.
This is a requirement on producers, not a guarantee stated by the upstream run
schema. The consumer relies on this contract; publication does not perform
cross-run membership validation.

Set picker dates describe the latest known develop commit using each set, not
execution or creation time.
Unknown URL identities stay explicit until the user chooses an available set or
Shared benchmarks. Export remains unfiltered.

Contract: [corpus run JSON](https://github.com/ROCm/rocjitsu-test-corpus/blob/develop/benchmarks/raw-run-schema.md). Commands: [build and test](build-and-test.md).
