# Workload shapes that change how a capture behaves

The main flow in [SKILL.md](../SKILL.md) wraps whatever command already
reproduces the failure and does not care what that command is. A few properties
of a workload do change what comes out, though, and they are worth recognising
before blaming the capture.

## The application may bring its own HIP runtime

An application that ships a copy of `libamdhip64` alongside itself loads that
copy, not the one in the ROCm install, because its own libraries are found
first. So a ROCm install with capture built in does not help if the bundled
copy has none. This is common wherever the runtime is delivered with the
application rather than by the system package manager.

`hrr_capture.sh preflight` lists every runtime it can see, in load order, and
marks which of them can capture. To put a capture-capable one in front:

```bash
export LD_PRELOAD=/path/to/libamdhip64.so.7
```

`LD_PRELOAD` wins over both a bundled copy and `LD_LIBRARY_PATH`. Preflight
again afterwards rather than assuming the preload took effect.

## One process or several

Every process that uses HIP writes its own `pid-<pid>/` sub-archive under the
same output directory, so the shape of the archive follows the shape of the
run. A single process leaves one directory. A workload that forks or spawns
children leaves one per child that touches the GPU, and the child doing the GPU
work is usually the interesting one while the parent only sets things up.

The whole directory is the archive. Sending one `pid-*` out of it loses the
context the rest carries.

## Stopping a long-running workload so the archive finalizes

The clean trailer is written when the process exits normally. A process
terminated with a signal it cannot handle is stopped outright, so its archive
ends without a trailer.

That archive is not lost: the writer flushes periodically, the reader keeps
every complete record and stops at a torn tail, and `hrr-playback --repair`
rewrites it with a clean trailer. But when the failure has already happened and
the run only needs to be ended, stopping the workload from the inside, so that it
returns from its own main loop, gives a complete archive with no repair step.

## Memory the capture cannot see

Some applications take one large allocation from HIP and then hand out pieces
of it in their own code. Those sub-allocations never reach a HIP API, so the
capture cannot see them, and a write that goes out of bounds while staying
inside the original allocation faults neither at capture time nor at replay. A
workload whose symptom is corrupted values rather than a fault may be hitting
exactly this.

## Cost

Capture adds host CPU work and disk writes to every HIP call, and hashes the
host buffers that memory copies carry. The cost is a property of the workload
rather than a constant: a workload that issues many small calls pays far more
than one that issues few large ones, so measure it for the workload in question
rather than quoting a single figure. It is not a setting to leave on in
production.

If the archive grows faster than the disk can take it, writing to a different
filesystem is the cheap experiment that separates an I/O limit from contention
inside the writer.
