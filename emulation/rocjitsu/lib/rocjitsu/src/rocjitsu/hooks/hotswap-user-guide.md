# gfx1250 B0-to-A0 HotSwap: user guide

`libhsa_hotswap_rocjitsu.so` translates a gfx1250 **B0** code object to a gfx1250
**A0** code object. ROCr loads it as an ordinary HSA tool; it patches the code-object
load entry points in the API table and rewrites each B0 object to an
A0-executable form on its way into the loader. Nothing else in the stack is
aware of it, and no rebuild of the application is required.

A0 and B0 share one machine identity (`amdgcn-amd-amdhsa--gfx1250`), so the
compiler cannot tell them apart and the loader would otherwise hand B0-only
encodings to an A0 part.

## When it engages

The hook only translates when **every** one of these holds. If any fails, loads
take the untouched path and translation never runs:

1. Linux. On other platforms `Runtime::LoadHotswapTool()` reports
   `HSA_STATUS_ERROR_NOT_SUPPORTED`.
2. `HSA_HOTSWAP_DISABLE` is unset or set to an off value (see below).
3. At `hsa_init` time at least one agent is named `gfx1250` **and** reports
   `HSA_AMD_AGENT_INFO_ASIC_REVISION == 0`. With no such agent ROCr skips the
   dlopen entirely — on a B0-only or non-gfx1250 machine the hook is never
   loaded, so it cannot be the cause of a problem there.
4. `libhsa_hotswap_rocjitsu.so` resolves, either next to `libhsa-runtime64.so`
   or on the default search path.

Once loaded, each individual load is still decided on its own:

| Situation | What happens |
| --- | --- |
| Agent is not gfx1250, or is gfx1250 B0 or later | Forwarded untouched |
| Agent stepping cannot be determined | `HSA_STATUS_ERROR_INVALID_AGENT` |
| Object is not gfx1250 | Loaded as-is |
| gfx1250 object onto an A0 agent | Translated B0 to A0 |
| Agent-less program-scope load, A0 possibly present | `HSA_STATUS_ERROR_INCOMPATIBLE_ARGUMENTS` |

Translation is a pure function of the input bytes and is memoized per source, so
a library registered once per device is translated once. A refusal is remembered
only when the translator returns `INVALID_CODE_OBJECT` — a verdict on the bytes.
Environmental failures such as out-of-memory are not remembered, so they do not
turn one bad moment into a permanent refusal.

## Environment variables

| Variable | Read by | Effect |
| --- | --- | --- |
| `HSA_HOTSWAP_DISABLE` | ROCr | Stops the tool being loaded at all. **Tolerant parsing**: unset, empty, `0`, `off`, `false`, `no`, `n`, `f` (any case) all mean "load it". Anything else disables. |
| `HSA_HOTSWAP_VERBOSE` | the hook | Diagnostic logging to stderr. **Strict parsing**: only unset, empty, or `0` are off. |
| `HSA_HOTSWAP_DUMP_SOURCE` | the hook | Write the source object to disk when translation is refused. Strict parsing, as above. |
| `HSA_HOTSWAP_DUMP_DIR` | the hook | Where those artifacts go. Falls back to `TMPDIR`, then `/tmp`. |

The two parsings genuinely differ, and it is the most common way to mislead
yourself here. `HSA_HOTSWAP_DISABLE=false` leaves hotswap **on**, while
`HSA_HOTSWAP_VERBOSE=false` turns verbose logging **on**, because the hook only
special-cases `0`. Use `1` and `0` and the distinction never bites.

Errors are always reported regardless of `HSA_HOTSWAP_VERBOSE`; the flag governs
what else is reported, never what is done.

## Reading the logs

Every line is prefixed `[hsa-hotswap-rj]` on stderr. There are two shapes.

A per-load summary:

```
[hsa-hotswap-rj] eager translation source_id=fnv1a64:3f2a91c07b5e4d18 \
  input_revision=b0 output_revision=a0 outcome=translated changed=42 \
  input_bytes=1048576 output_bytes=1049216 translation_status=0 status=0
```

- `source_id` — content hash of the input. The handle for a specific object;
  quote it in bug reports and use it to correlate lines across a noisy run.
- `outcome` — one of:
  - `translated` — rewritten and loaded.
  - `translation_failed` — the translator refused these bytes.
  - `output_copy_failed` — translation worked, the process could not allocate a
    copy of the result. Environmental, not a verdict on the object.
  - `reused` / `reused_failure` — served from the memo. Both are
    verbose-only, because on a real workload almost every load is a reuse.
- `changed` — instructions rewritten. **`changed=0` with
  `outcome=translated` means the object needed nothing**, which is a useful
  negative result when you are deciding whether hotswap is involved at all.
- `translation_status` / `status` — the `rj_status_t` and the `hsa_status_t`
  handed back to the caller.

This line is printed automatically on failure. On success it only appears under
`HSA_HOTSWAP_VERBOSE=1`. One wrinkle worth knowing: a failure served from the
memo — `reused_failure`, the same bad object loading a second time — is *not*
reprinted unless verbose is on, so a run can fail repeatedly having logged the
reason once. If the log seems to under-count the failures, turn verbose on.

A per-instruction diagnostic, emitted when the translator refuses:

```
[hsa-hotswap-rj] error: translation diagnostic source_id=fnv1a64:3f2a91c07b5e4d18 \
  severity=error kind=translator-expand-failed guest_offset=.text+0x1240 \
  mnemonic=s_barrier_signal_isfirst message=...
```

- `kind=translator-expand-missing` — no rule exists for that opcode yet.
- `kind=translator-expand-failed` — a rule exists but rejected this encoding.
- `guest_offset` / `mnemonic` — the offending instruction. Disassemble the
  source object at that offset to see it in context.
- `required=` replaces `message=` on follow-up lines that state what a fix
  needs.

## Capturing a refused object

```bash
export HSA_HOTSWAP_DUMP_SOURCE=1
export HSA_HOTSWAP_DUMP_DIR=/tmp/hotswap-dumps
```

Artifacts land as
`rocjitsu-gfx1250-b0-to-a0-<source_id>-XXXXXX.elf`, so the filename ties back to
the `source_id` in the log. Capture is opt-in because these objects are large —
RCCL's gfx1250 image is 212 MiB — and because a failed translation is not
memoized, so the same bytes would otherwise be rewritten on every load.

Limits worth knowing before you rely on it: one artifact per distinct source, at
most 32 distinct sources per process, and an out-of-resources failure is never
captured — copying a large input right after an allocation failure adds pressure
to a system that just ran out, and the bytes are not what failed.

## Is it a hotswap bug?

Work down this list; each step is cheap and rules out a class.

**1. Did the hook even load?** On a machine with no gfx1250 A0 agent it never
does, and it cannot be your problem. Confirm with `HSA_HOTSWAP_VERBOSE=1` — no
`[hsa-hotswap-rj]` output at all means it is not in the picture.

**2. Bisect with the kill switch.**

```bash
HSA_HOTSWAP_DISABLE=1 <your workload>
```

Read the result carefully, because it is easy to over-read:

- **Fails identically with the hook off** — not a hotswap bug. Look elsewhere.
- **Passes with the hook off** — hotswap is implicated, *but* confirm the
  workload actually needed translation before concluding that. If its objects
  translate with `changed=0`, the hook is a bystander and you have found a
  timing or ordering difference instead.
- **Fails differently with the hook off** — expected, and not informative on its
  own. An untranslated B0 object on an A0 part may fault or produce wrong
  answers. Do not read this as evidence either way.

**3. Separate refusal from miscompilation.** These need different owners.

- *Refused* — the load fails with `HSA_STATUS_ERROR_INVALID_CODE_OBJECT` and you
  get an `outcome=translation_failed` line plus diagnostics. The `kind`,
  `mnemonic`, and `guest_offset` name the instruction the translator will not
  handle. That is a gap in the B0-to-A0 rules; report it with those fields.
- *Wrong answers* — the load succeeds, `outcome=translated`, `changed>0`, and
  results are incorrect. That is a bad rule rather than a missing one. Capture
  the input with `HSA_HOTSWAP_DUMP_SOURCE=1` and report it with the `source_id`
  and the artifact.

**4. Narrow to one object.** Every log line carries a `source_id`, so in a
process loading many objects you can tell which one changed behavior, and
whether the one you suspect was translated at all.

A report is actionable when it carries the `source_id`, the full
`outcome=` line, any diagnostic lines, and — for wrong answers — the dumped
artifact.
