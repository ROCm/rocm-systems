# hipFile examples

Worked examples of the hipFile API, grouped by what they demonstrate. Each
subdirectory has its own README with the full program list, build, and run
instructions. The programs verify their results with an FNV-1a hash and print
`OK …` on success.

Most examples move data through the GPU and therefore need an AMD GPU supported
by ROCm and source/destination paths on an `O_DIRECT`-capable local filesystem
(ext4 mounted `data=ordered`, or xfs). Verify support with
`/opt/rocm/bin/ais-check`. The `api/` examples are the exception — they only
query the library and need neither.

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
`AIS_INSTALL_EXAMPLES=ON` (default):

```bash
cd rocm-systems/projects/hipfile
cmake -DCMAKE_CXX_COMPILER=amdclang++ -DCMAKE_HIP_PLATFORM=amd \
      -DAIS_INSTALL_EXAMPLES=ON -B build
cmake --build build --parallel
```

The binaries land under `build/examples/<directory>/`. See each
subdirectory's README for standalone (out-of-tree) build instructions and for
running the examples under `ctest`.
