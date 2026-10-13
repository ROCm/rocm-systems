---
myst:
    html_meta:
        "description": "ConSan GPU LDS sanitizer reference for rocJITsu, covering enabling, output format, supported race classes, limitations, and known edge cases."
        "keywords": "rocJITsu, ConSan, LDS, sanitizer, race detection, shared memory, AMD, ROCm, GPU, DBI"
---

# ConSan GPU LDS sanitizer reference

ConSan instruments AMD LDS (shared-memory) behavior by intercepting HSA code-object loads, inspecting final native machine code, and loading a patched replacement when instrumentation is possible. ConSan does not translate code objects between GPU architectures; it patches the final code object for the architecture that executes the kernel. Current live implementation and validation target RDNA4 / gfx1201.

For a hands-on walkthrough, see [Detect an LDS data race with ConSan](/tutorials/consan-detect-lds-race.md). For background on how ConSan fits into the rocJITsu plugin system, see [Execution plugin system](/conceptual/execution-plugins.md).

## Enabling ConSan

ConSan loads through the HSA tools interface. Set `HSA_TOOLS_LIB` to the built hook shared library and select an instrumentation flavor with the `RJ_CONSAN_FLAVOR` environment variable. If `RJ_CONSAN_FLAVOR` is unset, ConSan does not instrument the application.

``` bash
env HSA_TOOLS_LIB="$ROCJITSU_BUILD_DIR/lib/rocjitsu/src/rocjitsu/hooks/librocjitsu_dbi_hooks.so" \
  RJ_CONSAN_FLAVOR=supercollider \
  RJ_CONSAN_LOG=1 \
  ./application
```

### Instrumentation profiles

ConSan exposes two top-level flavors. The MOI flavor contains three engines.

| Selection | Behavior |
| --- | --- |
| `RJ_CONSAN_FLAVOR=supercollider` | Duplicate or read-back supported LDS accesses, delay, compare, and set an automatically allocated non-trapping mismatch marker. |
| `RJ_CONSAN_FLAVOR=moi`, `RJ_CONSAN_MOI_ENGINE=record_replay` | Instrument all admitted supported access, barrier, atomic, and fence sites; allocate an inventory-sized report; replay visible records on the host. |
| `RJ_CONSAN_FLAVOR=moi`, `RJ_CONSAN_MOI_ENGINE=sampled` | Patch all admitted supported sites; use automatic runtime stride 16,384 and offset zero; retain bounded sampled causal windows and synchronization metadata. |
| `RJ_CONSAN_FLAVOR=moi`, `RJ_CONSAN_MOI_ENGINE=inline_shadow` | Publish exact-shadow cells and bounded diagnostics on the GPU; track admitted barriers and atomics. |

`RJ_CONSAN_MOI_ENGINE` defaults to `record_replay` when the flavor is `moi`. Legacy `RJ_CONSAN_MOI_BACKEND=context|sampled_watchpoint` aliases map to `record_replay|sampled` only when the engine selector is absent.

### Core environment variables

| Variable | Default | Meaning |
| --- | --- | --- |
| `RJ_CONSAN_FLAVOR` | unset | Select instrumentation: `supercollider` or `moi`. |
| `RJ_CONSAN_MOI_ENGINE` | `record_replay` | Select the MOI engine: `record_replay`, `sampled`, or `inline_shadow`. |
| `RJ_CONSAN_LOG` | disabled | Enable compact logs at `1`; larger values add inventory detail. |
| `RJ_CONSAN_FAIL_CLOSED` | `0` | Reject unsupported or invalid transformation outcomes instead of loading the original code object. |
| `RJ_CONSAN_REQUIRE_PATCH` | `0` | Reject an applicable code object when no real access, barrier, atomic, or fence instrumentation patch is emitted. |
| `RJ_CONSAN_MAX_PATCHES` | `65536` | Expert upper bound per code object. The ordinary value means "all supported." |
| `RJ_CONSAN_FLAT_PROVENANCE` | `likely` | Admit proven `Group` plus heuristic `MaybeGroup` flat LDS sites (`likely`), or only proven `Group` sites (`strict`). |
| `RJ_CONSAN_DUMP_DIR` | unset | Write original and transformed `.hsaco` objects for inspection. |

## Output format

When `RJ_CONSAN_LOG=1` or higher, ConSan emits structured text records to standard error. The important records at the default log level are:

``` text
ConSan patch end ... outcome=... patches=... modified=...
ConSan summary ... patches=... modified=...
ConSan coverage ... flavor=... engine=... access=... barrier=... atomic=... fence=...
ConSan coverage_site ... kind=... disposition=... outcome=... reason=... lowering_reason=... resource_reason=...
ConSan analysis verdict ... static_complete=... dynamic_complete=...
```

### Coverage site records

MOI emits one typed `coverage_site` record per semantically relevant final-code site. Each record retains:

-   **kind** --- access, barrier, atomic, or fence.
-   **disposition** --- `not_applicable`, `supported`, or `unsupported`.
-   **outcome** --- `patched`, `resource_failed`, or `placement_or_lowering_failed`.
-   **reason** --- stable semantic reason for the disposition.
-   **lowering_reason** --- reason for a lowering failure, if applicable.
-   **resource_reason** --- detailed register-plan reason, if applicable.
-   Container ownership, kernel or function scope, text offset, and mnemonic.

`NotApplicable` records are omitted from output. For each event kind, the aggregate accounting follows:

``` text
discovered = supported + unsupported
supported = selected + expert_limit_omitted
selected = patched + resource_failed + placement_or_lowering_failed
```

### MOI report diagnostics

Inline Shadow emits bounded first-N diagnostics recording:

-   Instruction offset and access kind.
-   Owner identity and epoch.
-   LDS byte range.
-   Current conflict EXEC mask.
-   Visible overflow count.

The prior writer's lane mask is unavailable in the compact exact-shadow word and is reported as unknown or zero.

## Supported race classes

ConSan detects intra-workgroup LDS data races in the following categories:

-   **Read-after-write (RAW)** --- one wave reads an LDS byte while another wave has an outstanding write to the same byte without an intervening barrier.
-   **Write-after-read (WAR)** --- one wave writes an LDS byte while another wave has an outstanding read of the same byte without an intervening barrier.
-   **Write-after-write (WAW)** --- two waves write to the same LDS byte without an intervening barrier.

The conflict predicate, in simplified terms, requires:

-   Same workgroup.
-   Overlapping LDS cell or range.
-   At least one write.
-   Different owner.
-   Same unordered epoch, unless atomic or barrier semantics establish ordering.

## SuperCollider controls

| Variable | Default | Meaning |
| --- | --- | --- |
| `RJ_CONSAN_SC_REPORT_MODE` | `auto` | `auto` owns a non-trapping sticky marker per relevant code object. `trap` is an expert process-disrupting mode. |
| `RJ_CONSAN_REPORT_BUFFER` | unset | Use a caller-owned device-visible 32-bit marker address instead of automatic allocation. |
| `RJ_CONSAN_REPORT_MARKER` | `1` | Value written on mismatch. |
| `RJ_CONSAN_DELAY` | `0` | Delay parameter between the guest access and duplicate or read-back. |
| `RJ_CONSAN_DELAY_MODE` | `nop` | Select `s_nop`, `s_sleep`, or `s_sleep_var` delay lowering. |
| `RJ_CONSAN_DELAY_VAR_SSRC` | `106` | Scalar source encoding used by `sleep_var`. |
| `RJ_CONSAN_CHECK_TRAP_MODE` | `all` | Restrict SuperCollider to native DS (`lds`), admitted flat LDS (`flat`), or both (`all`). |

## MOI report buffer controls

| Variable | Default | Meaning |
| --- | --- | --- |
| `RJ_CONSAN_MOI_AUTO_REPORT_BUFFER_SIZE` | 16 MiB ceiling | Expert cap for HSA-tool-owned allocation. `0` disables automatic allocation. |
| `RJ_CONSAN_MOI_REPORT_BUFFER` | unset | Caller-owned device-visible report buffer address. |
| `RJ_CONSAN_MOI_REPORT_BUFFER_SIZE` | `0` | Size of the caller-owned buffer. |
| `RJ_CONSAN_MOI_REQUIRE_RECORDS` | `0` | At unload, require visible auto-buffer evidence. |
| `RJ_CONSAN_MOI_REQUIRE_DIAGNOSTICS` | `0` | Require at least one diagnostic or conflict. |
| `RJ_CONSAN_MOI_FORBID_DIAGNOSTICS` | `0` | Require zero diagnostics or conflicts. |
| `RJ_CONSAN_MOI_REQUIRE_REPLAY_CONFLICT` | `0` | Record/Replay-only positive guard. |
| `RJ_CONSAN_MOI_FORBID_OVERFLOW` | `0` | Fail if evidence was truly dropped. |

Automatic report buffers are limited to 16 MiB per buffer and 256 MiB of live automatic-report memory per process. The allocator requests exact inventory-derived bytes below those ceilings and does not silently shrink site coverage or disable an event kind to fit. Arithmetic overflow, a ceiling violation, or allocation failure is a typed incomplete outcome.

## MOI event and sampling controls

| Variable | Default | Meaning |
| --- | --- | --- |
| `RJ_CONSAN_MOI_TRACK_BARRIERS` | `1` | Track admitted barrier events. Explicit `0` is an expert override. |
| `RJ_CONSAN_MOI_TRACK_ATOMICS` | `1` | Track admitted atomic and fence ordering evidence. Explicit `0` is an expert override. |
| `RJ_CONSAN_MOI_DYNAMIC_ACCESS_RECORDS` | `0` | Record/Replay per-lane dynamic append (bounded expert tracing). |
| `RJ_CONSAN_MOI_SAMPLE_STRIDE` | `1` | Sampled static site stride. |
| `RJ_CONSAN_MOI_SAMPLE_OFFSET` | `0` | Static residue, smaller than the static stride. |
| `RJ_CONSAN_MOI_RUNTIME_SAMPLE_STRIDE` | `16384` | Expert power-of-two runtime stride (Sampled engine). |
| `RJ_CONSAN_MOI_RUNTIME_SAMPLE_OFFSET` | `0` | Expert runtime residue. |
| `RJ_CONSAN_MOI_SAMPLED_CHECK` | `0` | Enable immediate adjacent-range GPU check in addition to host scanning. |

## Supported instruction coverage

### Native LDS instructions

-   `ds_load_b32`
-   `ds_load_b64`
-   `ds_load_b128`
-   `ds_load_2addr_b32`
-   `ds_load_2addr_b64`
-   `ds_load_2addr_stride64_b32`
-   `ds_load_2addr_stride64_b64`
-   `ds_load_u16_d16`
-   `ds_load_u16_d16_hi`
-   `ds_store_b32`
-   `ds_store_b64`
-   `ds_store_b128`

### Flat LDS instructions (RDNA4 12-byte VFLAT)

-   `flat_load_b32`
-   `flat_load_b64`
-   `flat_load_b128`
-   `flat_store_b32`
-   `flat_store_b64`
-   `flat_store_b128`

## Current limitations

### Scope

-   **Intra-workgroup only** --- ConSan detects races within a single workgroup (missing barriers and insufficient wait-counter synchronization). It does not detect inter-workgroup races, races between dispatches, or host-device synchronization issues.

### Architecture

-   **RDNA4 validation** --- Live native validation is gfx1201 only. The intended native-instrumentation target set is `gfx942`, `gfx950`, `gfx1201`, and `gfx1250`. Other target ISAs need their own encoders, inventories, exact mutation identities, and GPU evidence.

### Coverage

-   ConSan is LDS and shared-memory focused. Selected atomics and fences provide ordering evidence but are not general global-memory race instrumentation.
-   Ordinary global-memory instrumentation is not in scope.
-   Unsupported flat widths such as b8, b16, and b96 are not instrumented.
-   Arbitrary flat accesses with unknown provenance are not instrumented.
-   Atomics are not instrumented as SuperCollider duplicate-access checks.
-   Async copies are not instrumented.

### Engine-specific

-   **SuperCollider** reports redundant-access instability, not causality. A race-free program can legitimately advance another wave between the original and repeated access.
-   **Record/Replay** is a bounded snapshot unless dynamic append is explicitly enabled, and dynamic append is still bounded by its finite report.
-   **Sampled** is probabilistic and can miss races. Clean sampled output is inconclusive about race freedom.
-   **Inline Shadow** has bounded diagnostics and supported-form semantics. It does not provide unbounded tracing or complete ISA coverage.

## Known edge cases

### Spilling

ConSan's instrumentation probes require temporary registers. When no dead registers are available in the kernel's current allocation, ConSan uses descriptor-backed fresh windows or spill-preserved windows. On gfx1201, a standalone VGPR spill backend allocates stable slots, emits `scratch_store/load_b32` batches with conservative split waits, and grows the selected descriptor's fixed private segment. General SGPR and AccVGPR spilling do not exist; unsupported ownership or resource shapes fail explicitly.

### Malformed input

ConSan's transformation is transactional. Every invocation produces exactly one typed outcome:

-   **ModifiedValid** --- replacement bytes are nonempty and independently validated.
-   **Unchanged** --- no replacement bytes produced; the original is loaded.
-   **Unsupported** --- no replacement bytes; fail-open loads the original, fail-closed rejects.
-   **Invalid** --- no replacement bytes; fail-open loads the original, fail-closed rejects.

A non-`ModifiedValid` result with replacement bytes or patches is a contract violation. A `ModifiedValid` result that fails final structural validation is rejected under both fail-open and fail-closed policies.

### Flat and generic LDS classification

Flat support is in scope because compiled HIP helper code can access LDS through flat or generic pointers. ConSan classifies flat sites using a machine-code provenance tracker with the following hints:

-   `Group` --- both 32-bit halves coherently traced from `src_shared_base`.
-   `MaybeGroup` --- only a component, select, or arithmetic chain remains consistent with LDS origin (heuristic, not a proof).
-   `Private`, `MaybePrivate`, `Global`, `Unknown`.

`RJ_CONSAN_FLAT_PROVENANCE=likely` (the default) admits both `Group` and `MaybeGroup`. `strict` admits only `Group`.

