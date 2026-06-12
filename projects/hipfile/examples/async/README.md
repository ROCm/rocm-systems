# `async`

Examples of hipFile's asynchronous, stream-based API
(`hipFileReadAsync` / `hipFileWriteAsync`). Each one seeds an input file with a
deterministic pattern, issues a GPU-mediated read+write round trip on one or
more HIP streams, synchronizes, and verifies the output by FNV-1a hash. They
share the helpers in [`../common`](../common) and print `OK …` on success.

## The examples

| Program | What it shows |
| --- | --- |
| `roundtrip-async` | Async read + write on the **default stream**, a single `hipStreamSynchronize`, then verify. |
| `roundtrip-async-nonblocking-stream` | Same round trip on an explicit `hipStreamNonBlocking` stream (no implicit sync with the legacy default stream). |
| `roundtrip-async-multi-stream` | `NUM_STREAMS` read/write pairs run concurrently, each on its own non-blocking stream covering a distinct file slice. |
| `roundtrip-async-multi-stream-registered` | Same concurrent multi-stream run, but each stream is registered with `hipFileStreamRegister` (fixed-offset / fixed-size / page-aligned hints) so the driver skips per-submission validation. |

All four take the same arguments:

```
PROGRAM READ_FILE WRITE_FILE [GPUID]
```

`READ_FILE` is created/truncated and seeded by the program itself, so it does
**not** need to exist beforehand. `WRITE_FILE` receives the round-tripped
payload. `GPUID` is optional (default `0`). Sizes and stream counts are
compile-time `#define`s documented at the top of each `.cpp`.

## Building

Built in-tree by the parent hipFile project when `AIS_INSTALL_EXAMPLES=ON`
(the default):

```bash
cd rocm-systems/projects/hipfile
cmake -DCMAKE_CXX_COMPILER=amdclang++ -DCMAKE_HIP_PLATFORM=amd \
      -DAIS_INSTALL_EXAMPLES=ON -B build
cmake --build build --parallel
```

The binaries land under `build/examples/async/`.

## Running

Both paths must live on an `O_DIRECT`-capable local filesystem (ext4 mounted
`data=ordered`, or xfs); verify with `/opt/rocm/bin/ais-check`. From
`build/examples/async/`:

```bash
./roundtrip-async                        in.bin out.bin
./roundtrip-async-nonblocking-stream     in.bin out.bin
./roundtrip-async-multi-stream           in.bin out.bin
./roundtrip-async-multi-stream-registered in.bin out.bin
```

### Via ctest

When configured with `-DBUILD_TESTING=ON`, each example is wrapped as a system
test (labels `async;hipfile;system`), with each test getting its own seeded
input path under `AIS_CAPABLE_DIR` (defaults to `/tmp` — point it at an
`O_DIRECT`-capable path):

```bash
ctest --test-dir build -L async --output-on-failure
```
