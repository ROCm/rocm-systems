# `common`

Not an example — a small static library (`examples_common`) of helpers shared
by the [`basics`](../basics) and [`async`](../async) examples. It was pulled out
to remove verbatim duplication; each example still drives the hipFile API
directly in its own `main()` so the example flow stays readable top-to-bottom.

There is nothing to run here. The library is built automatically as a
dependency whenever the examples are built (`AIS_INSTALL_EXAMPLES=ON`).

## What's in it

See [`examples_common.h`](examples_common.h) for the full, documented API. In
brief:

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
