---
name: hrr-capture-setup
description: >-
  Records a failing HIP or ROCm GPU workload into an HRR archive that can be
  replayed and debugged later, and checks that archive is readable before anyone
  is asked to send it. Use when an AMD GPU workload crashes, hangs, aborts with a
  memory access fault, or returns wrong or NaN results and the cause is not known
  yet; when the question is what to send AMD to reproduce a GPU failure; or when
  asked to turn on recording, capture or tracing of a workload for later replay.
  Applies to any process that uses the HIP runtime, whatever it is: a training
  or inference run, a simulation, a benchmark, a compiled application. Do NOT
  activate for performance questions, or when an archive already exists and the
  task is to replay or triage it, which is decode-and-triage.
---

# Record a failing workload

Capture writes every HIP call a process makes, plus the host buffers those calls
carry, into an archive directory. That archive replays on another machine
without the application, its source or its data, which is what makes a failure
someone else can look at.

This skill goes from a workload that misbehaves to an archive that has been
checked. Triaging the archive afterwards is a different skill,
[decode-and-triage](../../../hipamd/src/hrr/skills/decode-and-triage/SKILL.md).

Run `scripts/hrr_capture.sh` for every step below; paths are relative to this
skill's own directory. It needs Linux, bash, GNU coreutils and `python3`, and
runs `scripts/inspect_archive.py` itself, so invoke that directly only when a
machine-readable report is wanted. `scripts/test_inspect_archive.py` is the unit
suite for the inspector, run with `pytest`, not part of the workflow.

## References

- `references/workload-shapes.md` — how a workload that spawns children, has to
  be signalled to stop, or manages memory inside one allocation changes what
  comes out. **Load when** the workload is anything other than a single process
  that runs to completion.

## When this is the wrong tool

Say so rather than capturing anyway:

- **The workload is slow, not wrong.** Capture makes a run slower and does not
  measure it. Use the ROCm profilers.
- **The failure is above the GPU runtime**, in the application's own logic
  rather than in what it asks the GPU to do. Only HIP calls are recorded.
- **The failure never reproduces.** Capture records a run that happens; it
  cannot record one that does not.

## 1. Preflight

Run this **in the same environment as the workload**, meaning inside the
container and with the same user as the run itself:

```bash
scripts/hrr_capture.sh preflight --output /data/captures/run.hrr
```

It answers three questions that otherwise fail silently, hours later:

- **Which HIP runtime will the workload load, and does it have capture built
  in?** Capture is compiled into `libamdhip64`, so a runtime built without it
  ignores the environment variable, reports nothing, and leaves an empty
  directory. The script lists every `libamdhip64` visible in load order and
  marks which ones can capture. An application that ships its own runtime uses
  that one rather than the ROCm install, which is the most common reason a
  capture comes back empty.
- **Will the archive survive?** A path on the container's own writable layer
  disappears with the container. It has to be a bind mount from the host.
- **Is there room?** Archives reach tens of gigabytes for a workload of any
  size, and capture writes until the disk fills.

If the runtime that will load has no capture, put one that does in front with
`LD_PRELOAD`, and preflight again rather than assuming it worked.

## 2. Run

```bash
scripts/hrr_capture.sh run --output /data/captures/run.hrr -- \
    <the command that already reproduces the failure, unchanged>
```

Whatever command already reproduces the failure goes after `--`, unchanged. The
script preflights, sets `HIP_HRR_CAPTURE_OUTPUT`, runs the command, and reports
the archive afterwards. The workload's own exit status is preserved and passed
back.

Three things worth knowing before the run:

- **A crash is the good case.** A workload that dies mid-run still produces a
  readable archive; the capture writer flushes periodically and finalizes from
  the crash handler.
- **Only successful HIP calls are recorded.** An application that handles a
  failed call and retries will not replay down the same branch.
- **The device mask is left alone.** This skill never picks a GPU. Capture is
  meant to record the run being reproduced, so run it the way it
  normally runs.

Two flags change what `run` does, both off by default: `--skip-preflight` runs
the command with no checks at all, and `--force` runs it even when preflight
failed for any reason, including too little disk or a path that will not outlive
the container. `--min-free-gb N` moves the space threshold.

A workload that spawns children, one that has to be stopped rather than
finishing on its own, or one that manages its own memory inside a single
allocation, each behave differently here. Read
[references/workload-shapes.md](references/workload-shapes.md) before capturing
any of those.

## 3. Verify before sending

```bash
scripts/hrr_capture.sh verify --output /data/captures/run.hrr \
    --playback /path/to/hrr-playback
```

This reads the archive's manifests, so it needs no GPU, and cross-checks against
`hrr-playback --info`. Add `--json` for a machine-readable report, or
`--no-playback` to report from the manifests alone. Without `--playback` it
takes the first `hrr-playback` on `PATH`,
which is often the wrong one: the reader has to match the runtime that captured,
or it refuses the archive with `Version mismatch: file=N reader=M`. That means
the reader is wrong, not the archive. Read the result like this:

Every run ends in one of four verdicts:

- `Verdict: recorded` — events were written. The archive is worth keeping.
- `Verdict: empty` — process directories exist but hold no events. The processes
  opened a capture and recorded nothing.
- `Verdict: nothing captured` — the directory exists with no process directory
  in it, so no process ever opened a capture. The runtime was almost certainly
  not capturing: go back to preflight.
- `Verdict: no archive` — the directory was never created, so capture never
  started. Usually a workload that died before its first HIP call.

Each process directory also reports its state:

- `state: complete` — it shut down cleanly and wrote its trailer.
- `state: incomplete` — no trailer, so the process died. Expected for a crash,
  and still readable.
- `state: not finalized` — events exist but no manifest, so it was killed before
  it could finalize. The counts then come from `--info`.

An archive with several `pid-*` directories is one archive. Send or replay the
whole directory; the processes belong together.

## 4. Share it, knowing what is in it

The recorded buffers are the real ones, in the clear: whatever the workload put
on the GPU and read back is in the archive. There is no scrubbing or
randomisation option today, so an archive is exactly as sensitive as the data
the workload handled, and where it may go has to be agreed before it is
uploaded.

## Then triage it

With an archive in hand, hand off to
[decode-and-triage](../../../hipamd/src/hrr/skills/decode-and-triage/SKILL.md), which replays it and reports
what failed.
