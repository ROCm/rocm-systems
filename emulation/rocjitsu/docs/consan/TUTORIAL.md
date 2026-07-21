# ConSan tutorial

ConSan checks concurrency behavior by patching final native AMD GPU code as it
is loaded. It works through a rocJITsu HSA-tools hook, so the application does
not need to be rebuilt.

## 1. Build ConSan

Build the hook in an existing out-of-source rocJITsu build, then name it:

```sh
cmake --build "$ROCJITSU_BUILD_DIR" --target rocjitsu_dbi_hooks -j4

export CONSAN_HOOK="$ROCJITSU_BUILD_DIR/lib/rocjitsu/src/rocjitsu/hooks/librocjitsu_dbi_hooks.so"
```

## 2. Run an application

Start with one ordinary run:

```sh
env HSA_TOOLS_LIB="$CONSAN_HOOK" \
  RJ_CONSAN_ENABLE=1 \
  RJ_CONSAN_LOG=1 \
  ./application
```

That is the complete ordinary setup. `RJ_CONSAN_ENABLE=1` selects ConSan's
recommended default analysis and its standard settings. ConSan discovers
relevant sites and manages registers, report memory, synchronization tracking,
and other instrumentation resources automatically.

Loading the hook alone is inert; `RJ_CONSAN_ENABLE=1` is what turns ConSan on.
`RJ_CONSAN_LOG=1` adds the compact evidence needed to understand the run.

If the application needs a non-system ROCm distribution, set its library path
before running it:

```sh
export LD_LIBRARY_PATH="$ROCM_DIST_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

## 3. Read the result

At `RJ_CONSAN_LOG=1`, first verify that applicable code was transformed:

```text
ConSan patch end ... outcome=modified-valid ... patches=N modified=true
ConSan coverage ... access=... barrier=... atomic=... fence=...
ConSan analysis verdict ... static_complete=... dynamic_complete=...
```

The default analysis then reports a bounded host-side replay in a line shaped
like:

```text
ConSan MOI auto replay ... diagnostics=N conflict=true|false ...
```

This default is called **MOI Record/Replay**. It instruments admitted memory and
synchronization sites, retains a bounded snapshot of runtime events, and
reconstructs their ordering on the host.

`conflict=true` with an attributed replay diagnostic is positive ConSan
evidence. `conflict=false` means only that the retained bounded snapshot did
not expose a conflict. Complete static-site instrumentation does not guarantee
that every dynamic event survived for replay.

If the application's own correctness checks still pass, ConSan preserved the
result for that run. That does not make a clean ConSan report proof of race
freedom. A timeout, signal, GPU reset, or application failure is not by itself
a ConSan diagnostic.

## 4. Choose a different analysis when useful

(Full document: [FLAVORS.md](FLAVORS.md))

Most investigations should stay with the default. ConSan also offers two other
MOI engines and the complementary SuperCollider flavor.

| Flavor | Engine | Useful positive evidence | Main tradeoff |
| --- | --- | --- | --- |
| MOI (default) | **Record/Replay (default)** | Host replay emits an attributed conflict from its retained snapshot. | Clear, inspectable model, but runtime history is bounded. |
| MOI (default) | **Inline Shadow** | The GPU emits an immediate diagnostic for a supported access. | Strong device-side attribution with more device work. |
| MOI (default) | **Sampled** | A retained statistical campaign emits sampled conflicts. | Lower retained state with probabilistic detection. |
| **SuperCollider** | — | A delayed redundant observation changes the automatic mismatch marker. | Complementary instability signal, not a happens-before diagnosis. |

### Other MOI engines

Inline Shadow and Sampled are engines within the default MOI flavor. Their
commands therefore change only `RJ_CONSAN_MOI_ENGINE`.

Use Inline Shadow when immediate supported-form device-side attribution is
worth more device work:

```sh
env HSA_TOOLS_LIB="$CONSAN_HOOK" \
  RJ_CONSAN_ENABLE=1 \
  RJ_CONSAN_MOI_ENGINE=inline_shadow \
  RJ_CONSAN_LOG=1 \
  ./application
```

Use Sampled for broad statistical campaigns where probabilistic detection is
acceptable:

```sh
env HSA_TOOLS_LIB="$CONSAN_HOOK" \
  RJ_CONSAN_ENABLE=1 \
  RJ_CONSAN_MOI_ENGINE=sampled \
  RJ_CONSAN_LOG=1 \
  ./application
```

### Different flavor: SuperCollider

SuperCollider is a separate ConSan flavor, **not** an MOI engine. Its command
must set `RJ_CONSAN_FLAVOR=supercollider`; do not try to select it through
`RJ_CONSAN_MOI_ENGINE` or omit the flavor setting when adapting an MOI command.
Use it for its complementary delayed redundant-observation signal:

```sh
env HSA_TOOLS_LIB="$CONSAN_HOOK" \
  RJ_CONSAN_ENABLE=1 \
  RJ_CONSAN_FLAVOR=supercollider \
  RJ_CONSAN_LOG=1 \
  ./application
```

All ordinary selections automatically instrument every relevant site they
support and allocate registers and reports. The MOI engines also enable
supported barrier and atomic tracking; Sampled chooses its runtime sampling
parameters automatically.

## 5. Enable ConSan self-checks

An ordinary run reports instrumentation problems in its log. For a focused
test, ConSan can instead treat those problems as failures. These self-checks
prevent an ineffective run from looking reassuringly clean:

| Self-check | What it enforces |
| --- | --- |
| `RJ_CONSAN_FAIL_CLOSED=1` | Do not fall back to the original code when transformation is unsupported or invalid. |
| `RJ_CONSAN_REQUIRE_PATCH=1` | Require a real access, barrier, atomic, or fence instrumentation patch in applicable code. |
| `RJ_CONSAN_MOI_REQUIRE_RECORDS=1` | Require the default MOI analysis to produce visible runtime evidence. |
| `RJ_CONSAN_MOI_FORBID_OVERFLOW=1` | Fail if MOI report capacity is exhausted. |

Enable them alongside the default analysis:

```sh
env HSA_TOOLS_LIB="$CONSAN_HOOK" \
  RJ_CONSAN_ENABLE=1 \
  RJ_CONSAN_LOG=1 \
  RJ_CONSAN_FAIL_CLOSED=1 \
  RJ_CONSAN_REQUIRE_PATCH=1 \
  RJ_CONSAN_MOI_REQUIRE_RECORDS=1 \
  RJ_CONSAN_MOI_FORBID_OVERFLOW=1 \
  ./application
```

These checks describe the health of ConSan instrumentation, not whether the
program is race-free. If a known-correct test should also produce no ConSan
diagnostic, add this separate expected-result assertion before `./application`:

```sh
  RJ_CONSAN_MOI_FORBID_DIAGNOSTICS=1 \
```

Apply self-checks first to a focused test or kernel.
`RJ_CONSAN_FAIL_CLOSED=1` and `RJ_CONSAN_REQUIRE_PATCH=1` can be too strict for
a large application that also loads unsupported or irrelevant helper code.

## 6. Validate detection with fault injection

ConSan can deliberately change one synchronization or memory-ordering operation
in a known program—for example, by dropping or moving a barrier or weakening
an atomic operation. This creates a controlled faulty version of the program
and lets you check whether a ConSan analysis recognizes the resulting problem.

A fault-injection run has two independent results:

- The program's own correctness check shows whether the injected fault changed
  its observable behavior.
- An attributed ConSan report shows whether ConSan detected a concurrency
  problem.

A program can notice bad output without ConSan producing a diagnostic. A hang,
crash, or GPU reset means the experiment did not complete safely; it does not
count as detection.

Fault injection uses the same HSA hook as an ordinary ConSan run. First, ask
ConSan to inventory candidates without changing the code. This example looks
for atomic release operations whose ordering can be weakened:

```sh
env HSA_TOOLS_LIB="$CONSAN_HOOK" \
  RJ_CONSAN_ENABLE=1 \
  RJ_CONSAN_LOG=1 \
  RJ_CONSAN_FAULT_DRY_RUN=1 \
  RJ_CONSAN_FAULT_ATOMIC_WEAKEN_ORDER=1 \
  RJ_CONSAN_FAULT_ATOMIC_ORDER_EDGE=release \
  ./application 2>consan-fault-inventory.log
```

The proposed `ConSan fault plan` names a `primary=` identity. Find the matching
`ConSan fault site` record and review its container, instruction, role, and
other fields. Then copy the complete `primary=` value into a live run:

```sh
env HSA_TOOLS_LIB="$CONSAN_HOOK" \
  RJ_CONSAN_ENABLE=1 \
  RJ_CONSAN_LOG=1 \
  RJ_CONSAN_FAULT_ATOMIC_WEAKEN_ORDER=1 \
  RJ_CONSAN_FAULT_ATOMIC_ORDER_EDGE=release \
  RJ_CONSAN_FAULT_SITE_IDENTITY='IDENTITY_FROM_DRY_RUN' \
  RJ_CONSAN_FAULT_REQUIRE_EXACTLY_ONE=1 \
  ./application
```

ConSan applies the mutation before instrumenting the resulting code, so the
same run both introduces the fault and analyzes it. Verify that `ConSan fault
summary` reports `requested=1 planned=1 applied=1`; otherwise the intended
positive control was not created. Site identities describe an exact native
binary and should be rediscovered after rebuilding the program or ConSan.

An injected fault can make a program hang or destabilize the GPU. Run these
experiments one at a time. Check `rocminfo` and a small known-good program
before and after each experiment, and avoid unrelated concurrent GPU work. Give
the application an appropriate external timeout so a deliberately broken
synchronization operation cannot hang indefinitely.

See the fault-injection controls in [USAGE.md](USAGE.md) for other mutation
families and their additional selectors.

## Troubleshooting

No ConSan logs:

- verify `HSA_TOOLS_LIB` names the newly built hook;
- verify `RJ_CONSAN_ENABLE=1` is present in the application environment;
- verify the process uses an HSA runtime that honors HSA tools; and
- set `RJ_CONSAN_LOG=1` explicitly.

`modified=false` or zero patches:

- inspect `ConSan coverage_site` reasons at a higher log level;
- distinguish an unsupported instruction from a register-resource or patch
  placement failure; and
- try a smaller test that isolates the kernel you expected ConSan to instrument.

MOI reports no visible records:

- confirm an automatic report was planned and allocated;
- inspect the logged required and allocated byte counts;
- remember that automatic Sampled selection can retain no event in a short run;
  and
- use `RJ_CONSAN_MOI_REQUIRE_RECORDS=1` only when the program is expected to
  execute an instrumented site.

The GPU becomes unhealthy:

- stop launching GPU work;
- do not count the failed run as a ConSan detection; and
- recover the device and run a small uninstrumented program successfully before
  continuing.

## Next documents

- [USAGE.md](USAGE.md): complete public controls and result interpretation.
- [FLAVORS.md](FLAVORS.md): detailed comparison of the available analyses.
- [MALFORMED_INPUT.md](MALFORMED_INPUT.md): the optional malformed-barrier
  guard and its safety boundary.
- [DESIGN.md](DESIGN.md): implementation details for readers who want to go
  beyond the user interface.
