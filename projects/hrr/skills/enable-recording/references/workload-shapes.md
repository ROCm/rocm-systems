# Workload shapes that change how a capture behaves

The main flow in [SKILL.md](../SKILL.md) wraps whatever command already reproduces the failure and does not care what that command is. A few properties of a workload do change what comes out, though, and they are worth recognising before blaming the capture.

## The application may bring its own HIP runtime

An application that ships a copy of `libamdhip64` alongside itself loads that copy, not the one in the ROCm install, because its own libraries are found first. So a ROCm install with capture built in does not help if the bundled copy has none. This is common wherever the runtime is delivered with the application rather than by the system package manager.

`hrr_capture.sh preflight` lists every runtime it can see, in load order, and marks which of them can capture. Give it the workload command after `--`: a binary's own `DT_RPATH` is searched before `LD_LIBRARY_PATH` and no search of the environment can see it, so with the command preflight asks the loader instead of guessing.

Which remedy works depends on what loads the runtime, and the difference is not cosmetic.

**A compiled binary that links `libamdhip64` itself.** Put a capture-capable runtime in front:

```bash
export LD_PRELOAD=/path/to/libamdhip64.so.7
```

`LD_PRELOAD` wins over both a bundled copy and `LD_LIBRARY_PATH`. Preflight again afterwards rather than assuming the preload took effect.

**A Python framework that carries its own ROCm, such as a PyTorch or vLLM wheel.** `LD_PRELOAD` does not work here, and it fails in a way that looks like broken hardware: the preloaded runtime resolves against the wheel's older HSA and dies on a missing symbol, or it initialises far enough to enumerate the GPU and then fails code-object initialisation, after which the framework reports no GPU at all. Winning the link is not enough, because the runtime also has to reach a compatible `comgr` and the device bitcode that sit beside it in its own SDK tree.

**Find that directory by watching, not by reading.** A wheel commonly ships two SDK directories holding different builds, and the `DT_RUNPATH` on its libraries can name absolute paths that do not exist in the image it ends up running in. It is also searched after `LD_LIBRARY_PATH` rather than before it, so reading it tells you less than it looks. Ask a live process instead:

```bash
python3 -c 'import torch; torch.cuda.init(); print(open("/proc/self/maps").read())' \
  | grep -m2 -E 'libamdhip64|libhsa-runtime64'
```

In-process on purpose: backgrounding the interpreter and reading `/proc/$!/maps`
after a sleep races with its exit, and an empty result then reads as "nothing
loaded" when the truth is "already gone".

The directory in that output is the one to overlay. Reading `readelf -d` on the framework's libraries answers a different question and has pointed at the wrong directory of the two.

What works is installing the capture-capable runtime into the directory the framework actually binds, together with the matching `libhsa-runtime64` from the same build, and leaving `LD_PRELOAD` unset:

```bash
# inside the container, over the framework's own ROCm libraries.
# Move the originals aside first: this edits a shared environment, and there is
# no way back to the stock wheel afterwards short of reinstalling it.
mkdir -p ~/hrr-wheel-backup
mv <wheel-sdk>/lib/libamdhip64.so.7      ~/hrr-wheel-backup/
mv <wheel-sdk>/lib/libhsa-runtime64.so.1 ~/hrr-wheel-backup/
cp <capture-sdk>/lib/libamdhip64.so.7        <wheel-sdk>/lib/
cp <capture-sdk>/lib/libhsa-runtime64.so.1   <wheel-sdk>/lib/
unset LD_PRELOAD
```

Keep the backup outside that lib directory rather than beside the library: a second `libamdhip64` in there is what makes a runtime verdict unreliable in the first place. If the backup is lost, a throwaway virtual environment or `pip install --force-reinstall` gets you back to the stock wheel.

The capture runtime and the framework have to be compatible in the other direction too, and the newest build is not automatically the right one. A HIP build that needs a newer `comgr` than the framework's own will not run it at all, so the pairing is found by trying rather than by version order.

## One process or several

Every process that uses HIP writes its own `pid-<pid>/` sub-archive under the same output directory, so the shape of the archive follows the shape of the run. A single process leaves one directory. A workload that forks or spawns children leaves one per child that touches the GPU, and the child doing the GPU work is usually the interesting one while the parent only sets things up.

The whole directory is the archive. Sending one `pid-*` out of it loses the context the rest carries.

## Stopping a long-running workload so the archive finalizes

The clean trailer is written when the process exits normally. A process terminated with a signal it cannot handle is stopped outright, so its archive ends without a trailer.

That archive is not lost: the writer flushes periodically, the reader keeps every complete record and stops at a torn tail, and `hrr-playback --repair` rewrites it with a clean trailer. But when the failure has already happened and the run only needs to be ended, stopping the workload from the inside, so that it returns from its own main loop, gives a complete archive with no repair step.

## Memory the capture cannot see

Some applications take one large allocation from HIP and then hand out pieces of it in their own code. Those sub-allocations never reach a HIP API, so the capture cannot see them, and a write that goes out of bounds while staying inside the original allocation faults neither at capture time nor at replay. A workload whose symptom is corrupted values rather than a fault may be hitting exactly this.

## Cost

Capture adds host CPU work and disk writes to every HIP call, and hashes the host buffers that memory copies carry. The cost is a property of the workload rather than a constant: a workload that issues many small calls pays far more than one that issues few large ones, so measure it for the workload in question rather than quoting a single figure. It is not a setting to leave on in production.

If the archive grows faster than the disk can take it, writing to a different filesystem is the cheap experiment that separates an I/O limit from contention inside the writer.
