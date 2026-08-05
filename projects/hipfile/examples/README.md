# hipFile examples

This directory contains working examples of the hipFile API, grouped by what
they demonstrate. The `basics` and `async` programs verify their results with
an FNV-1a hash and print `OK …` on success.

Most examples move data through the GPU on hipFile's fast path, which opens
files with `O_DIRECT`; running them as written therefore requires an AMD GPU
supported by ROCm and source/destination paths on an `O_DIRECT`-capable local
filesystem (ext4 mounted `data=ordered`, or xfs). `O_DIRECT` is not a hipFile
requirement, though — files can be registered without it and routed through the
POSIX compat path (see `basics/no-odirect-write`). Verify fast-path support with
`/opt/rocm/bin/ais-check`. The `api/` examples are the exception — they only
query the library and need neither a GPU nor an `O_DIRECT` filesystem.

## The directories

| Directory | What's in it |
| --- | --- |
| [`api`](api) | Minimal examples of the non-I/O API — calls that query or configure the library (e.g. `get-version`). No `O_DIRECT` filesystem or file arguments needed. |
| [`basics`](basics) | Small, single-purpose programs that each exercise one facet of the synchronous API: buffer registration, the `O_DIRECT` fast path vs. compat fallback, chunked reads, device-buffer offsets, sub-region writes, GPU memory types, and a full round trip. |
| [`async`](async) | Examples of the asynchronous, stream-based API (`hipFileReadAsync` / `hipFileWriteAsync`), including single-stream, non-blocking-stream, and concurrent multi-stream round trips. |
| [`aiscp`](aiscp) | A standalone `cp`-like utility built on hipFile (`aiscp SOURCE DEST`). |
| [`common`](common) | Not an example — a small static library (`examples_common`) of helpers shared by `basics` and `async` (alignment math, pattern fill, hashing, file open/register). |

## Building

Each example directory contains a ready-to-use `CMakeLists.txt` that finds the
installed hipFile via `find_package`. To build a directory, configure and build
it in place:

```bash
cd basics
cmake -B build -DCMAKE_PREFIX_PATH="/opt/rocm;/path/to/hipfile"
cmake --build build --parallel
```

This is an out-of-source build, so the example binaries land in `build/`. The
run snippets below assume you have `cd build`'d into that directory first.

Drop the `-DCMAKE_PREFIX_PATH` argument if ROCm and hipFile are installed in
standard locations. If the install tree is read-only, copy the directory (and,
for `basics`/`async`, the sibling [`common`](common) directory) somewhere
writable first.

`basics` and `async` link the shared helpers in [`common`](common) and pull it
in via `add_subdirectory(../common …)`, so keep the `common` directory alongside
them when you build.

### Finding the library at runtime

When you build an example, CMake records the location of `libhipfile.so` (taken
from the same `find_package` result used above) in the example binary, so it
runs in place even when hipFile is installed outside the default library search
path — no `LD_LIBRARY_PATH` needed.

That recorded path is fixed at build time. If you later move the hipFile
installation, the binaries can no longer find the library and fail to start with
`error while loading shared libraries: libhipfile.so.0`. Point the loader at the
new location to recover (or rebuild the examples):

```bash
cd build
LD_LIBRARY_PATH=/new/path/to/hipfile/lib ./bufregister-write out.bin
```

Per-example payload and chunk sizes are compile-time `#define`s (e.g.
`-DBRW_SIZE=…`, `-DIR_CHUNK_SIZE=…`, documented at the top of each `.cpp`). To
build an example with different sizes, pass the corresponding `-D` flag at
configure time.

---

## `api`

Minimal examples of the non-I/O parts of the hipFile API — calls that query or
configure the library rather than move data through the GPU. These do **not**
require an `O_DIRECT`-capable filesystem or even file arguments.

| Program | What it shows | Args |
| --- | --- | --- |
| `get-version` | Read the hipFile version both ways: the `HIPFILE_VERSION_*` header macros (compile-time) and `hipFileGetVersion()` (runtime). | none |

### Running

```bash
cd build
./get-version
```

Prints the version from the header symbols and from the runtime call. No file
or GPU memory is touched.

---

## `basics`

Small, single-purpose programs that each exercise one facet of the hipFile C
API: buffer registration, the `O_DIRECT` fast path vs. compat fallback, chunked
reads, device buffer offsets, sub-region writes, the three GPU memory types, and
a full round trip. They drive the API directly from `main()` and use the shared
helpers in [`common`](common) (`open_file`, `hash_file_range`, `align_up`,
`fill_pattern`, …). Every example verifies its result with an FNV-1a hash and
prints `OK …` on success.

| Program | What it shows | Args |
| --- | --- | --- |
| `bufregister-write` | Write a GPU buffer registered with `hipFileBufRegister` straight to disk (the fast path). | `OUTPUT [GPUID]` |
| `no-bufregister-write` | Same write, but without registering the buffer — hipFile copies through its internal pool. | `OUTPUT [GPUID]` |
| `no-odirect-write` | Register a file opened *without* `O_DIRECT`. On an `O_DIRECT`-capable filesystem hipFile transparently reopens it with `O_DIRECT` and still takes the fast path; only on a filesystem that can't do `O_DIRECT` does it use the POSIX compat fallback (on by default — set `HIPFILE_ALLOW_COMPAT_MODE=true` only if you previously disabled it). | `OUTPUT [GPUID]` |
| `iterative-read` | Chunked read into GPU memory where the **host pointer** advances each iteration, then one write. | `INPUT OUTPUT [GPUID]` |
| `iterative-devmem-offset-read` | Same chunked read, but the base device pointer is fixed and the **`buf_offset`** argument advances. | `INPUT OUTPUT [GPUID]` |
| `subregion-write` | Read a whole file, then write only the bytes at/after an offset using a non-zero `buffer_offset`. | `INPUT OUTPUT [GPUID]` |
| `various-mem-rw` | Round-trip a file using device (`1`), managed (`2`), or pinned-host (`3`) memory as the transfer buffer. | `INPUT OUTPUT MODE [GPUID]` |
| `roundtrip-verify` | Write a known pattern, read it back through the GPU, write a copy, and assert both files hash-match. | `CREATED COPIED [GPUID]` |

`GPUID` is optional and defaults to `0`. Payload/chunk sizes are compile-time
`#define`s (e.g. `-DBRW_SIZE=…`, `-DIR_CHUNK_SIZE=…`) documented at the top of
each `.cpp`.

### Running

Reads need an existing input file; create one with `dd`:

```bash
cd build
dd if=/dev/urandom of=input.bin bs=1M count=1
```

The input and output paths must live on an `O_DIRECT`-capable local
filesystem (ext4 mounted `data=ordered`, or xfs); verify with
`/opt/rocm/bin/ais-check`:

```bash
./bufregister-write            out_bufregister.bin
./no-bufregister-write         out_no_bufregister.bin
./no-odirect-write             out_no_odirect.bin   # hipFile reopens with O_DIRECT; no env var needed
./iterative-read               input.bin out_iter.bin
./iterative-devmem-offset-read input.bin out_iter_off.bin
./subregion-write              input.bin out_subregion.bin
./various-mem-rw               input.bin out_vmrw.bin 1     # 1=device 2=managed 3=pinned
./roundtrip-verify             rtv_created.bin rtv_copied.bin
```

---

## `async`

Examples of hipFile's asynchronous, stream-based API
(`hipFileReadAsync` / `hipFileWriteAsync`). Each one seeds an input file with a
deterministic pattern, issues a GPU-mediated read+write round trip on one or
more HIP streams, synchronizes, and verifies the output by FNV-1a hash. They
share the helpers in [`common`](common) and print `OK …` on success.

> **Note:** The `O_DIRECT` fast path is not currently supported for
> asynchronous I/O — async operations always run through the POSIX compat
> (fallback) path, regardless of whether the file or filesystem is
> `O_DIRECT`-capable. Fast-path async support is planned for the future.

| Program | What it shows |
| --- | --- |
| `roundtrip-async` | Async read + write on the **default stream**, a single `hipStreamSynchronize`, then verify. |
| `roundtrip-async-nonblocking-stream` | Same round trip on an explicit `hipStreamNonBlocking` stream (no implicit sync with the legacy default stream). |
| `roundtrip-async-multi-stream` | `NUM_STREAMS` read/write pairs run concurrently, each on its own non-blocking stream covering a distinct file slice. |
| `roundtrip-async-multi-stream-registered` | Same concurrent multi-stream run, but each stream is registered with `hipFileStreamRegister` (fixed-offset / fixed-size / page-aligned hints) so the driver skips per-submission validation. |

All four take the same arguments:

```text
PROGRAM READ_FILE WRITE_FILE [GPUID]
```

`READ_FILE` is created/truncated and seeded by the program itself, so it does
**not** need to exist beforehand. `WRITE_FILE` receives the round-tripped
payload. `GPUID` is optional (default `0`). Sizes and stream counts are
compile-time `#define`s documented at the top of each `.cpp`.

### Running

Both paths must live on an `O_DIRECT`-capable local filesystem (ext4 mounted
`data=ordered`, or xfs); verify with `/opt/rocm/bin/ais-check`:

```bash
cd build
./roundtrip-async                        in.bin out.bin
./roundtrip-async-nonblocking-stream     in.bin out.bin
./roundtrip-async-multi-stream           in.bin out.bin
./roundtrip-async-multi-stream-registered in.bin out.bin
```

---

## `aiscp`

`aiscp` is a simple test program that uses hipFile to copy a file. It works
like the Linux `cp` command:

```text
aiscp SOURCE DEST
```

---

## `common`

Not an example — a small static library (`examples_common`) of helpers shared
by the [`basics`](basics) and [`async`](async) examples. It was pulled out to
remove verbatim duplication; each example still drives the hipFile API directly
in its own `main()` so the example flow stays readable top-to-bottom.

There is nothing to run or build here on its own. The `basics`/`async`
directories pull it in via `add_subdirectory(../common …)`, so the helper code
is compiled once and shared rather than duplicated into each example. Keep this
directory alongside `basics`/`async` when building them.

### What's in it

See [`examples_common.h`](common/examples_common.h) for the full, documented
API.
