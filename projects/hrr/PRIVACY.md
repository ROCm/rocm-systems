# What an HRR archive contains, and what replaying one means

This page states what a capture archive can and cannot contain, and what a person who
replays one is trusting. It is the customer-facing text of
[AIRUNTIME-2733](https://amd-hub.atlassian.net/browse/AIRUNTIME-2733) (capture privacy and
content control). The engineering threat model is
[AIRUNTIME-2853](https://amd-hub.atlassian.net/browse/AIRUNTIME-2853).

Every task under AIRUNTIME-2733 that changes what an archive holds updates this page in the
same change. `tools/tests/test_privacy_doc.py`, run by the ctest `hrr-generator-rules`,
fails when the manifest fields written by capture and the table below disagree.

## Replay runs GPU code taken from the archive

An archive is untrusted executable input for the tool that replays it. `hrr-playback` loads
the code objects stored in the archive and launches their kernels on the GPU. That code runs
with the replaying process's access to device memory and to any host memory mapped for the
GPU, and nothing in the replay tool can check what it does. The tool refuses malformed
records and caps memory, events and run time, but it cannot make the code itself safe.
Replay maps recorded pointers to the buffers it allocated so that the run can be rebuilt;
this is reconstruction, not a safety boundary. Only a value that matches a recorded
allocation is translated: a pointer argument that matches none is passed as null, and every
other argument byte, including a pointer-sized word inside a struct that matches none, is
passed as recorded, so a kernel can still receive addresses from the original process.

Replay an archive only if you trust whoever produced it, as an unprivileged user, on a
machine and GPU that hold no data you would not hand to that person. A container is a
reasonable boundary. `hrr-playback --info` reads the archive without loading any code and
needs no GPU.

## Capture modes

Capture is disabled by default on AMD platforms at present (see the [README](README.md));
where it is built in, it is armed by `HIP_HRR_CAPTURE_OUTPUT` alone. Today there is one
mode, and it records **contents**: everything in the tables below.

[AIRUNTIME-2880](https://amd-hub.atlassian.net/browse/AIRUNTIME-2880) adds a
**metadata-only** mode as the default and moves contents behind a second variable. The
"Mode" column states which mode records each item once that task has landed. Until then,
every capture is a contents capture.

## What an archive contains

| Item | Where | Mode | Removed or reduced by |
|---|---|---|---|
| Sequence of successful HIP calls, with sizes, flags, device addresses, handles, thread ids | `events.bin` | both | — kept, replay needs it |
| Kernel names, mangled but read back with `c++filt`; they describe the model's architecture and precision | `events.bin` | both | — kept, triage names the failing kernel |
| Event timestamps, from a monotonic clock (not the date or time of day) | `events.bin` | both | — kept |
| Device properties passed to `hipChooseDevice`, the whole `hipDeviceProp_t`: device and architecture names, PCI ids and, in the current layout, the UUID and LUID | `events.bin` | both | — kept, the call's answer depends on them |
| Other structures passed into a HIP call, copied whole, such as copy, memset, graph node, allocation, pool and launch-attribute parameters. Padding, reserved and unused union bytes are copied as the caller left them | `events.bin` | both | — kept, replay needs it |
| Host buffers copied to or from the device, byte for byte, unencrypted. For a serving stack this includes prompts and generated output as token ids, which the model's public tokeniser reverses, and the model weights | `blobs/` | contents | [AIRUNTIME-2880](https://amd-hub.atlassian.net/browse/AIRUNTIME-2880) |
| Host memory registered with `hipHostRegister`: the whole range, byte for byte, as it was at registration, whether or not it is ever copied | `blobs/` | contents | [AIRUNTIME-2880](https://amd-hub.atlassian.net/browse/AIRUNTIME-2880) |
| Kernel argument bytes | `events.bin` | contents | [AIRUNTIME-2880](https://amd-hub.atlassian.net/browse/AIRUNTIME-2880) |
| Code objects the process loaded (the application's own GPU binary code) | `code_objects/`, `blobs/` | contents | [AIRUNTIME-2880](https://amd-hub.atlassian.net/browse/AIRUNTIME-2880) |
| Linker input buffers passed to `hipLinkAddData`, every byte, with their names, including inputs the call rejected and inputs never linked or loaded | `blobs/` | contents | [AIRUNTIME-2880](https://amd-hub.atlassian.net/browse/AIRUNTIME-2880) |
| Whole fat binary bundles, including targets HIP never loads | `blobs/` | contents | [AIRUNTIME-2858](https://amd-hub.atlassian.net/browse/AIRUNTIME-2858): only the code objects HIP loads |
| Debug and comment sections of code objects: absolute build paths, source file names, the exact compiler build | `code_objects/`, `blobs/` | contents | [AIRUNTIME-2881](https://amd-hub.atlassian.net/browse/AIRUNTIME-2881): stripped |
| Framework allocator layout, when a region producer ran beside capture | `regions/` | both | — kept, written only by an opt-in producer |
| Marker that capture is running: on Linux the boot id and the process start time in clock ticks since boot; on Windows the process creation date and time. A normal exit removes it; a process that crashes or is killed leaves it | `active` | both | — kept, region producers read it |

The per-process `manifest.json` also holds `pid`, `parent_pid`, `complete`, `event_count`,
`blob_count`, the HIP and comgr versions, and one block per device:

| Device field | Kept | Removed by |
|---|---|---|
| `name`, `gcn_arch_name`, `compute_capability` | yes | — |
| `total_global_mem`, `multi_processor_count`, `warp_size`, `max_threads_per_block` | yes | — |
| `clock_rate_khz`, `memory_clock_rate_khz`, `memory_bus_width`, `l2_cache_size` | yes | — |
| `integrated`, `managed_memory`, `memory_pools_supported`, `compute_mode` | yes | — |
| `uuid`, `pci`, `luid`, `luid_device_node_mask`: identify the individual GPU and its slot | no | [AIRUNTIME-2881](https://amd-hub.atlassian.net/browse/AIRUNTIME-2881): blanked |

This table covers `manifest.json` only. A `hipChooseDevice` call can carry the same
identifiers in `events.bin`, as listed above.

## Never recorded

Capture does not read the host name, the command line, environment variables or the user
name, and records a date and time of day only in the `active` marker on Windows. A string
reaches the archive only as a kernel name; as a string argument of a HIP call (the path
given to `hipLinkAddFile`, the name given to `hipLinkAddData`, the function name given to
`hipGetProcAddress`); as the name of a device variable the program registers; as a character
array in a structure passed to a HIP call (the device and architecture names given to
`hipChooseDevice`); inside the padding or unused bytes of such a structure; inside a host
buffer the workload copied or registered; inside a linker input buffer, even one the linker
rejected; or inside a code object. A path in such a buffer,
for example a compiler cache path under the user's home directory, is recorded with it.
Memory that never crossed a HIP call is not recorded, and neither is a failed call, apart
from the exceptions listed in [skills/enable-recording/SKILL.md](skills/enable-recording/SKILL.md).

## What no mode can promise

A metadata-only archive with identifiers blanked still names every kernel the workload
ran, which describes the model and the serving stack, and still records every size and
address. It holds no buffer contents and no code, so it cannot reproduce a fault that
depends on values, and it cannot be replayed in full. A contents archive is as sensitive
as the data the workload handled: where it may go has to be agreed before it is sent.

## How the archive is protected on disk

On Linux the archive is readable only by the user who captured it (directories 0700, files
0600). [AIRUNTIME-2847](https://amd-hub.atlassian.net/browse/AIRUNTIME-2847) makes every
write go through the directory descriptors capture checked.
[AIRUNTIME-2857](https://amd-hub.atlassian.net/browse/AIRUNTIME-2857) decides whether
capture must ask for consent before it starts.
[AIRUNTIME-2878](https://amd-hub.atlassian.net/browse/AIRUNTIME-2878) keeps the crash-time
flush without displacing the application's own crash handler. None of these changes what an
archive contains.
