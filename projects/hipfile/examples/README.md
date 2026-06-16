# hipFile examples

Worked examples of the hipFile API, grouped by what they demonstrate. The
programs verify their results with an FNV-1a hash and print `OK …` on success.
This top-level README consolidates the full program lists, build, and run
instructions for every example directory.

Most examples move data through the GPU and therefore need an AMD GPU supported
by ROCm and source/destination paths on an `O_DIRECT`-capable local filesystem
(ext4 mounted `data=ordered`, or xfs). Verify support with
`/opt/rocm/bin/ais-check`. The `api/` examples are the exception — they only
query the library and need neither a GPU nor an `O_DIRECT` filesystem.

## The directories

| Directory | What's in it |
| --- | --- |
| [`api`](api) | Minimal examples of the non-I/O API — calls that query or configure the library (e.g. `get-version`). No `O_DIRECT` filesystem or file arguments needed. |
| [`basics`](basics) | Small, single-purpose programs that each exercise one facet of the synchronous API: buffer registration, the `O_DIRECT` requirement, chunked reads, device-buffer offsets, sub-region writes, GPU memory types, and a full round trip. |
| [`async`](async) | Examples of the asynchronous, stream-based API (`hipFileReadAsync` / `hipFileWriteAsync`), including single-stream, non-blocking-stream, and concurrent multi-stream round trips. |
| [`aiscp`](aiscp) | A standalone `cp`-like utility built on hipFile (`aiscp SOURCE DEST`). |
| [`common`](common) | Not an example — a small static library (`examples_common`) of helpers shared by `basics` and `async` (alignment math, pattern fill, hashing, file open/register). |
| `out` | Scratch directory for example output files. |

## Building

Most examples are built in-tree by the parent hipFile project when
`AIS_INSTALL_EXAMPLES=ON` (the default):

```bash
cd rocm-systems/projects/hipfile
cmake -DCMAKE_CXX_COMPILER=amdclang++ -DCMAKE_HIP_PLATFORM=amd \
      -DAIS_INSTALL_EXAMPLES=ON -B build
cmake --build build --parallel
```

The binaries land under `build/examples/<directory>/`. The `api` and `aiscp`
directories can also be built standalone against an installed hipFile — see
those sections below.

---

## `api`

Minimal examples of the non-I/O parts of the hipFile API — calls that query or
configure the library rather than move data through the GPU. These do **not**
require an `O_DIRECT`-capable filesystem or even file arguments.

### The examples

| Program | What it shows | Args |
| --- | --- | --- |
| `get-version` | Read the hipFile version both ways: the `HIPFILE_VERSION_*` header macros (compile-time) and `hipFileGetVersion()` (runtime). | none |

### Building

In-tree, the `api` examples are built by the parent hipFile project when
`AIS_INSTALL_EXAMPLES=ON` (the default). Unlike `basics/` and `async/`, these
use the `ais_add_executable` macro and link the `hipfile` target directly.

To build standalone against an installed hipFile, copy `CMakeLists.install.in`
to `CMakeLists.txt` in a scratch copy of the directory — it uses
`find_package(hipfile)` instead of the in-tree macro:

```bash
mkdir -p /tmp/api-example
cp CMakeLists.install.in /tmp/api-example/CMakeLists.txt
cp get-version.cpp /tmp/api-example/
cmake -DCMAKE_PREFIX_PATH="/opt/rocm;/path/to/hipfile" -S /tmp/api-example -B /tmp/api-example/build
cmake --build /tmp/api-example/build
```

### Running

```bash
./get-version
```

Prints the version from the header symbols and from the runtime call. No file
or GPU memory is touched.

---

## `basics`

Small, single-purpose programs that each exercise one facet of the hipFile C
API: buffer registration, the `O_DIRECT` requirement, chunked reads, device
buffer offsets, sub-region writes, the three GPU memory types, and a full
round trip. They drive the API directly from `main()` and use the shared
helpers in [`common`](common) (`open_file`, `hash_file_range`, `align_up`,
`fill_pattern`, …). Every example verifies its result with an FNV-1a hash and
prints `OK …` on success.

### The examples

| Program | What it shows | Args |
| --- | --- | --- |
| `bufregister-write` | Write a GPU buffer registered with `hipFileBufRegister` straight to disk (the fast path). | `OUTPUT [GPUID]` |
| `no-bufregister-write` | Same write, but without registering the buffer — hipFile copies through its internal pool. | `OUTPUT [GPUID]` |
| `no-odirect-write` | Open the file *without* `O_DIRECT`, forcing hipFile's POSIX compat fallback. Needs `HIPFILE_ALLOW_COMPAT_MODE=true`. | `OUTPUT [GPUID]` |
| `iterative-read` | Chunked read into GPU memory where the **host pointer** advances each iteration, then one write. | `INPUT OUTPUT [GPUID]` |
| `iterative-devmem-offset-read` | Same chunked read, but the base device pointer is fixed and the **`buf_offset`** argument advances. | `INPUT OUTPUT [GPUID]` |
| `subregion-write` | Read a whole file, then write only the bytes at/after an offset using a non-zero `buffer_offset`. | `INPUT OUTPUT [GPUID]` |
| `various-mem-rw` | Round-trip a file using device (`1`), managed (`2`), or pinned-host (`3`) memory as the transfer buffer. | `INPUT OUTPUT MODE [GPUID]` |
| `roundtrip-verify` | Write a known pattern, read it back through the GPU, write a copy, and assert both files hash-match. | `CREATED COPIED [GPUID]` |

`GPUID` is optional and defaults to `0`. Payload/chunk sizes are compile-time
`#define`s (e.g. `-DBRW_SIZE=…`, `-DIR_CHUNK_SIZE=…`) documented at the top of
each `.cpp`.

### Building

Built in-tree by the parent hipFile project when `AIS_INSTALL_EXAMPLES=ON`
(the default; see the top-level [Building](#building) section). The binaries
land under `build/examples/basics/`.

### Running

Reads need an existing input file; create one with `dd`:

```bash
dd if=/dev/urandom of=input.bin bs=1M count=1
```

The input and output paths must live on an `O_DIRECT`-capable local
filesystem (ext4 mounted `data=ordered`, or xfs); verify with
`/opt/rocm/bin/ais-check`. Then, from `build/examples/basics/`:

```bash
./bufregister-write            out_bufregister.bin
./no-bufregister-write         out_no_bufregister.bin
HIPFILE_ALLOW_COMPAT_MODE=true ./no-odirect-write out_no_odirect.bin
./iterative-read               input.bin out_iter.bin
./iterative-devmem-offset-read input.bin out_iter_off.bin
./subregion-write              input.bin out_subregion.bin
./various-mem-rw               input.bin out_vmrw.bin 1     # 1=device 2=managed 3=pinned
./roundtrip-verify             rtv_created.bin rtv_copied.bin
```

### Via ctest

When configured with `-DBUILD_TESTING=ON`, each example is wrapped as a system
test (labels `basics;hipfile;system`). The wrappers seed a scratch input and
run under `AIS_CAPABLE_DIR` (defaults to `/tmp` — point it at an
`O_DIRECT`-capable path):

```bash
ctest --test-dir build -L basics --output-on-failure
```

Note: `various-mem-rw`'s `managed` and `pinned` modes are marked `DISABLED` in
the ctest setup, so only the `device` mode runs under ctest. Run the other two
modes by hand.

---

## `async`

Examples of hipFile's asynchronous, stream-based API
(`hipFileReadAsync` / `hipFileWriteAsync`). Each one seeds an input file with a
deterministic pattern, issues a GPU-mediated read+write round trip on one or
more HIP streams, synchronizes, and verifies the output by FNV-1a hash. They
share the helpers in [`common`](common) and print `OK …` on success.

### The examples

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

### Building

Built in-tree by the parent hipFile project when `AIS_INSTALL_EXAMPLES=ON`
(the default; see the top-level [Building](#building) section). The binaries
land under `build/examples/async/`.

### Running

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

---

## `aiscp`

`aiscp` is a simple test program that uses hipFile to copy a file. It works
like the Linux `cp` command:

```
aiscp SOURCE DEST
```

### Building

Create a build directory and use cmake to configure and build the program. You
may need to add the path(s) to ROCm and/or hipFile if they are installed in
non-standard locations:

```bash
cmake -DCMAKE_PREFIX_PATH="/path/to/rocm;/path/to/hipfile" /path/to/aiscp/dir
cmake --build .
```

---

## `common`

Not an example — a small static library (`examples_common`) of helpers shared
by the [`basics`](basics) and [`async`](async) examples. It was pulled out to
remove verbatim duplication; each example still drives the hipFile API directly
in its own `main()` so the example flow stays readable top-to-bottom.

There is nothing to run here. The library is built automatically as a
dependency whenever the examples are built (`AIS_INSTALL_EXAMPLES=ON`).

### What's in it

See [`examples_common.h`](common/examples_common.h) for the full, documented
API. In brief:

| Helper | Purpose |
| --- | --- |
| `BLOCK_ALIGN`, `is_power_of_two`, `align_up` | `O_DIRECT` alignment math — round transfer sizes up to a power-of-two block size. |
| `fill_pattern` | Fill a buffer with a deterministic test pattern (byte `i` = `i & 0xFF`). |
| `hash_buffer` | FNV-1a 64-bit hash of a memory buffer. |
| `hash_file_range` | Read a byte range of a file via plain POSIX I/O and hash it (host-side reference path). |
| `seed_read_file` | Create/truncate a file and write `fill_pattern` bytes to it (no `O_DIRECT`). |
| `verify_files_match` | Hash the first N bytes of two files and compare. |
| `open_file` | `open(2)` a file (caller controls flags) and register it with `hipFileHandleRegister`. |
| `close_file` | Deregister and close a handle opened with `open_file`. |

`open_file` deliberately does **not** add `O_DIRECT` for you: pass it to take
the GPU-direct fast path, or omit it to route through the POSIX compat path
(see `basics/no-odirect-write`).
</content>
</invoke>
