---
name: enable-recording
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

Capture writes every HIP call a process makes, plus the host buffers those calls carry, into an archive directory. That archive replays on another machine without the application, its source or its data, which is what makes a failure someone else can look at.

This skill goes from a workload that misbehaves to an archive that has been checked. Triaging the archive afterwards is a different skill, [decode-and-triage](../decode-and-triage/SKILL.md).

Run `scripts/hrr_capture.sh` for every step below; paths are relative to this skill's own directory. It needs Linux, bash, GNU coreutils and `python3`, and runs `scripts/inspect_archive.py` itself, so invoke that directly only when a machine-readable report is wanted. `tests/` holds the inspector's unit suite and the recorded tool output it reads, run with `pytest`, and is not part of the workflow.

## Prerequisites

- **A ROCm build whose `libamdhip64` has HRR compiled in.** Capture is part of the runtime, not a separate package, and no ROCm version number tells you whether a given build has it. Preflight is the test: it greps each candidate runtime for `HIP_HRR_CAPTURE_OUTPUT` and marks it `capture: yes` or `capture: NO`.
- **Linux.** Capture on Windows exists but this skill's scripts do not run there.
- **CDNA data-centre parts.** Everything here was exercised on `gfx950` (MI350X). Nothing in it is architecture-specific, and nothing has been tried on RDNA, so on a consumer part treat a failure as unproven rather than as a bug.
- **Disk, in tens of gigabytes.** Archives are large and capture writes until the disk fills. Preflight refuses below 50 GiB free; `--min-free-gb` moves that for a short run.
- **A matching `hrr-playback` if you want the archive cross-checked**, which is optional. The sibling skill's `../decode-and-triage/scripts/ensure_playback.sh` finds one or builds it.

## Environment variables that change what you get

Each of these changes the result silently, and the first two are the ones that waste an afternoon.

- **`HIP_HRR_CAPTURE_OUTPUT`**: switches capture on and names the archive. A runtime without HRR ignores it and reports nothing, which is why preflight exists. Let `run` set it rather than exporting it yourself, because a variable exported in the shell that starts a container does not reach the process inside it.
- **`HIP_VISIBLE_DEVICES`**: changes which devices the workload sees and therefore what gets recorded. This skill never sets it. HIP's ordering is not `rocm-smi`'s, so device 4 here is not necessarily GPU 4 there; if you pin a device, confirm which one you got.
- **`LD_PRELOAD` and `LD_LIBRARY_PATH`**: decide which runtime loads. `LD_PRELOAD` wins over everything, including a binary's own `DT_RPATH`; `LD_LIBRARY_PATH` is the one that loses to it. Winning the link is not the same as working, which is why preloading a runtime into a framework wheel breaks it. See step 1.
- **`ROCM_PATH`**: where `verify` looks for `hrr-playback` when `--playback` is not given. The first reader found is often the wrong one.
- **`HIP_HRR_DEBUG_ARGS`**: turns on capture's own argument logging, and raises the runtime's log level for the whole process, so the workload's output changes shape.
- **`PYTORCH_HIP_ALLOC_CONF`**: changes how PyTorch sub-allocates, which changes what HIP sees and therefore what the archive holds. Capture what the failing run uses, not a tidier configuration.

## References

- `references/workload-shapes.md`: how a workload that spawns children, has to be signalled to stop, or manages memory inside one allocation changes what comes out. **Load when** the workload is anything other than a single process that runs to completion.

## When this is the wrong tool

Say so rather than capturing anyway:

- **The workload is slow, not wrong.** Capture makes a run slower and does not measure it. Use the ROCm profilers.
- **The failure is above the GPU runtime**, in the application's own logic rather than in what it asks the GPU to do. Only HIP calls are recorded.
- **The failure never reproduces.** Capture records a run that happens; it cannot record one that does not.

## 1. Preflight

Run this **in the same environment as the workload**, meaning inside the container and with the same user as the run itself, and pass the workload command after `--`:

```bash
scripts/hrr_capture.sh preflight --output /data/captures/run.hrr -- \
    <the command that already reproduces the failure>
```

The command matters. A binary's own `DT_RPATH` is searched before `LD_LIBRARY_PATH`, and no search of the environment can see it, so given the command preflight asks the loader which runtime will bind rather than guessing from what is visible. Without it the check still runs and is still worth running, but it answers a weaker question and says so.

It answers three questions that otherwise fail silently, hours later:

- **Which HIP runtime will the workload load, and does it have capture built in?** Capture is compiled into `libamdhip64`, so a runtime built without it ignores the environment variable, reports nothing, and creates no archive at all: `verify` afterwards reports `no archive`, which is easy to mistake for a workload that died early. The script lists every `libamdhip64` visible in load order, marks which ones can capture, and names the one the command actually binds. An application that ships its own runtime uses that one rather than the ROCm install, which is the most common reason a capture comes back empty. An interpreter is the awkward case: `python3` links no HIP runtime itself, and the one a framework loads later cannot be resolved this way, so for a Python workload read the bundled-package rows of the listing.
- **Will the archive survive?** A path on the container's own writable layer disappears with the container. It has to be a bind mount from the host.
- **Is there room?** Archives reach tens of gigabytes for a workload of any size, and capture writes until the disk fills.

If the runtime that will load has no capture, the fix depends on what loads it. For a compiled binary, put a capture-capable runtime in front with `LD_PRELOAD`. For a framework that carries its own ROCm in a wheel, PyTorch and vLLM being the common ones, `LD_PRELOAD` does not work: it resolves against the wheel's older HSA and the framework ends up reporting no GPU at all. There the capture-capable runtime has to be installed into the wheel's own lib directory, with the matching `libhsa-runtime64` from the same build. Both routes are in [references/workload-shapes.md](references/workload-shapes.md). Either way, preflight again rather than assuming it worked.

## 2. Run

```bash
scripts/hrr_capture.sh run --output /data/captures/run.hrr -- \
    <the command that already reproduces the failure, unchanged>
```

Whatever command already reproduces the failure goes after `--`, unchanged. The script preflights, sets `HIP_HRR_CAPTURE_OUTPUT`, runs the command, and reports the archive afterwards. The workload's own exit status is preserved and passed back, which also means a run that succeeds while capturing nothing exits zero: the verdict, not the exit status, says whether capture worked.

**Run this inside the container, not around it.** The script sets the environment variable on the command it launches, and a variable set in the shell that starts a container does not cross into it. Wrapping `docker run` therefore produces exactly the empty-capture symptom this skill exists to prevent. Enter the container first, with the skill directory and the output path both visible inside it, and run the three steps there. A container running as root leaves the archive owned by root on the host, where the person who owns the directory can read it and not move it, so `chown` it back afterwards.

Three things worth knowing before the run:

- **A crash is the good case.** A workload that dies mid-run still produces a readable archive: the writer flushes periodically, and it finalizes from the runtime's own handler for `SIGSEGV`, `SIGABRT`, `SIGBUS`, `SIGILL` and `SIGFPE`. Nothing else is covered, so a `docker stop`, a `timeout`, or a launcher killing its ranks leaves `state: not finalized` instead. That archive is still readable and `hrr-playback --repair` gives it a clean trailer.
- **Only successful HIP calls are recorded, with a few exceptions.** An application that handles a failed call and retries will not replay down the same branch. A handful of hand-written shims record regardless of the result, among them the memory-pool and array creation calls.
- **The device mask is left alone.** This skill never picks a GPU. Capture is meant to record the run being reproduced, so run it the way it normally runs.

Two flags change what `run` does, both off by default: `--skip-preflight` runs the command with no checks at all, and `--force` runs it even when preflight failed for any reason, including too little disk or a path that will not outlive the container. `--force` is also the only way past an output directory that already holds a capture, and it then adds this run to it, so the two become one archive nobody can separate afterwards: give `--output` a new path instead unless merging is what you want. `--min-free-gb N` moves the space threshold.

A workload that spawns children, one that has to be stopped rather than finishing on its own, or one that manages its own memory inside a single allocation, each behave differently here. Read [references/workload-shapes.md](references/workload-shapes.md) before capturing any of those.

## 3. Verify before sending

```bash
scripts/hrr_capture.sh verify --output /data/captures/run.hrr \
    --playback /path/to/hrr-playback
```

This reads the archive's manifests, so it needs no GPU, and cross-checks against `hrr-playback --info`. Add `--json` for a machine-readable report, or `--no-playback` to report from the manifests alone, which checks what the capture writer wrote rather than that a reader can decode it. Without `--playback` it takes the first `hrr-playback` on `PATH`, which is often the wrong one: the reader has to match the runtime that captured, or it refuses the archive with `Version mismatch: file=N reader=M`. That means the reader is wrong, not the archive.

A reader has libraries of its own and may not start at all, commonly inside a container that has no ROCm install to fall back on. That is a failed cross-check rather than a bad archive: the report says so and exits 3, and the archive is untouched by it. A reader that starts and then refuses the archive exits 3 for the same reason, since the archive still went unread. No reader anywhere on the machine is not a failure: that is the manifests-only path and it exits 0.

If you are scripting a gate, run `verify` as its own step. `run` returns the workload's exit status, by design, so it cannot also report what the check found. `verify` exits 0 when the archive holds events, 1 for the three verdicts that mean it does not, 2 for a usage mistake, and 3 when a cross-check was asked for and could not run.

A refusal is the good case. A reader from a neighbouring build can do something worse: it accepts the archive and decodes the event names against its own API table, so the counts are right, the names are wrong, and nothing reports an error. From format v5 the runtime API ids are append-only, so that is a risk for older archives and for compiler-table events rather than for every event in every archive. The report therefore names the reader it used and its version, and says so when the pairing cannot be established. If the API mix it prints does not look like the workload that ran, distrust the reader before the archive.

Every run ends in one of four verdicts:

- `Verdict: recorded`: events were written. The archive is worth keeping.
- `Verdict: empty`: process directories exist but hold no events. The processes opened a capture and recorded nothing.
- `Verdict: nothing captured`: the directory exists with no process directory in it, so no process ever opened a capture. The directory came from somewhere else, an earlier attempt or a `mkdir`, and the variable never reached the process that used the GPU: go back to preflight.
- `Verdict: no archive`: the directory was never created, so capture never started. Either the runtime that loaded has no capture compiled in, which is the common case, or the workload died before its first HIP call. Preflight with the command tells the two apart.

Each process directory also reports its state:

- `state: complete`: it shut down cleanly and wrote its trailer.
- `state: incomplete`: no trailer. Usually the process died, which is expected for a crash and still readable. It also appears when capture itself dropped an event it could not fit in the wire format, and that archive has a hole in the middle rather than a truncated tail. The workload's own stderr tells the two apart: capture prints `marking the capture INCOMPLETE` in the second case.
- `state: not finalized`: events exist but no manifest, so it was killed before it could finalize. The counts then come from `--info`.

An archive with several `pid-*` directories is one archive. Send or replay the whole directory; the processes belong together.

## 4. Share it, knowing what is in it

**There is no scrubbing, anonymisation or randomisation option today.** Say that plainly before anyone asks whether the archive can be sanitised, because the answer shapes what they do next and there is no flag that changes it.

What the archive holds, per process, and it is worth reading out to whoever has to approve the transfer:

- **`events.bin`**: every successful HIP call the process made, with its arguments: sizes, device addresses, stream handles, kernel names.
- **`blobs/`**: the host buffers those calls carried, byte for byte and in the clear. For a serving stack that means prompts and generated text; for training, the data and the weights that were copied to the device.
- **`code_objects/`**: the compiled GPU kernels the process loaded, which is the application's own binary code and a separate disclosure question from customer data.
- **`manifest.json`**: process and parent process ids, event and blob counts, the clean-shutdown flag, the HIP and comgr versions that captured, and the device's name, architecture, PCI location and UUID.
- **`regions/*.hrrr`**: the framework allocator's own layout, when a producer ran alongside capture.
- **`writer_state.json`**: the checkpoint cursor, present only while a capture is mid-flight.

Failed HIP calls are not recorded, and neither is anything that never crossed a HIP call. The archive is exactly as sensitive as the data the workload handled, so where it may go has to be agreed before it is uploaded.

`verify --json` gives that inventory in a form you can attach to the request. For a multi-process archive, `hrr-playback --info` against the archive root prints a table covering every process at once. The cross-check in `verify` reports on one process, which is enough to judge the capture and not enough for a disclosure inventory.

If the content is the obstacle, three things exist and none of them is scrubbing. Re-capture the same failure driven by synthetic input, if it still reproduces. Replay the archive yourself and send only the finding, which is a few lines and carries no buffers. Or agree the handling of a production-data transfer before uploading, because that is what this is.

## 5. Then replay it, here, now

**Do this before sending anything anywhere.** The machine that captured is the one machine whose runtime is guaranteed to match the archive, so a replay started here cannot hit the reader-pairing problem that makes every later attempt awkward. It is also the fastest way to learn that the archive is worth sending: a replay that reproduces the fault is proof, where a verdict of `recorded` is only a promise.

Offer it as soon as `verify` says `recorded`, and hand off to [decode-and-triage](../decode-and-triage/SKILL.md) on the archive just made:

```bash
../decode-and-triage/scripts/triage_archive.sh --archive /data/captures/run.hrr/pid-<pid>
```

Start with the process that did not shut down cleanly, `incomplete` or `not
finalized`, since that is the one that failed. A `not finalized` archive gets a
clean trailer from `hrr-playback --repair` first.

That skill replays the archive and reports what failed: fault class, faulting kernel, fault address, failing event. If no `hrr-playback` is around, its `scripts/ensure_playback.sh` finds one or builds it, which is also the answer to "where do I get the matching reader" in step 3.

One rule for a multi-process archive, since the two skills sound contradictory and are not: **send it whole, replay one process at a time.** The processes belong together and the directory is the unit of transfer, while triage takes a single `pid-*` directory.
