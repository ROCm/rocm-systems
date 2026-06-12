# `basics`

Small, single-purpose programs that each exercise one facet of the hipFile C
API: buffer registration, the `O_DIRECT` requirement, chunked reads, device
buffer offsets, sub-region writes, the three GPU memory types, and a full
round trip. They drive the API directly from `main()` and use the shared
helpers in [`../common`](../common) (`open_file`, `hash_file_range`,
`align_up`, `fill_pattern`, …). Every example verifies its result with an
FNV-1a hash and prints `OK …` on success.

## The examples

| Program | What it shows | Args |
| --- | --- | --- |
| `bufregister-write` | Write a GPU buffer registered with `hipFileBufRegister` straight to disk (the fast path). | `OUTPUT [GPUID]` |
| `no-bufregister-write` | Same write, but without registering the buffer — hipFile copies through its internal pool. | `OUTPUT [GPUID]` |
| `no-odirect-write` | Open the file *without* `O_DIRECT`, forcing hipFile's POSIX compat fallback. Needs `HIPFILE_ALLOW_COMPAT_MODE=1`. | `OUTPUT [GPUID]` |
| `iterative-read` | Chunked read into GPU memory where the **host pointer** advances each iteration, then one write. | `INPUT OUTPUT [GPUID]` |
| `iterative-devmem-offset-read` | Same chunked read, but the base device pointer is fixed and the **`buf_offset`** argument advances. | `INPUT OUTPUT [GPUID]` |
| `subregion-write` | Read a whole file, then write only the bytes at/after an offset using a non-zero `buffer_offset`. | `INPUT OUTPUT [GPUID]` |
| `various-mem-rw` | Round-trip a file using device (`1`), managed (`2`), or pinned-host (`3`) memory as the transfer buffer. | `INPUT OUTPUT MODE [GPUID]` |
| `roundtrip-verify` | Write a known pattern, read it back through the GPU, write a copy, and assert both files hash-match. | `CREATED COPIED [GPUID]` |

`GPUID` is optional and defaults to `0`. Payload/chunk sizes are compile-time
`#define`s (e.g. `-DBRW_SIZE=…`, `-DIR_CHUNK_SIZE=…`) documented at the top of
each `.cpp`.

## Building

These are built in-tree by the parent hipFile project when
`AIS_INSTALL_EXAMPLES=ON` (the default):

```bash
cd rocm-systems/projects/hipfile
cmake -DCMAKE_CXX_COMPILER=amdclang++ -DCMAKE_HIP_PLATFORM=amd \
      -DAIS_INSTALL_EXAMPLES=ON -B build
cmake --build build --parallel
```

The binaries land under `build/examples/basics/`.

## Running

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
HIPFILE_ALLOW_COMPAT_MODE=1 ./no-odirect-write out_no_odirect.bin
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
