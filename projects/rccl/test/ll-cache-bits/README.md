# RCCL LL Cache-Bit Guard

A [pytest](https://pytest.org) guard that checks the cache-policy bits on the
LL protocol's flag-poll loads in a built `librccl.so`.

## Why

The LL receiver (`readLL` in `src/device/prims_ll.h`) re-reads one 16-byte FIFO
line `{data1, flag1, data2, flag2}` until both flags change. A peer GPU or a NIC
writes that line. If the load can be served from a stale cached copy, the
receiver never sees the new flag and the collective hangs.

The source asks for the right behaviour (`__builtin_nontemporal_load`, or a
system-scope load on some arches), but the compiler picks the bits that end up
in the instruction. A compiler update, a code change, or a new arch can drop
them with no build error. This guard reads the final ISA and fails if that
happens.

## What it checks

1. Pulls the device code for each arch out of `librccl.so`.
2. Disassembles the functions that can hold LL poll loads.
3. Finds every poll load by its shape: a 128-bit load whose two flag words are
   both compared against the same expected value. Cache bits are not used to
   find it, so a poll that lost them is still found.
4. Asserts each poll carries the arch's expected modifiers, e.g. `nt` on
   gfx942.

It also fails if some `ProtoLL` function has no recognisable poll at all, so a
change in code shape can't make the guard pass by finding nothing.

| File | Purpose |
|------|---------|
| `tests/ll_isa.py` | poll-load detector and ROCm LLVM tool wrappers |
| `tests/test_ll_poll_cache_bits.py` | the guard, run against a built `librccl.so` |
| `tests/test_ll_isa_detector.py` | detector unit tests on hand-written ISA (no ROCm needed) |

Arches with an expectation: **gfx942, gfx950**. Other arches in the library
are reported as skipped, not passed.

## Running locally

Needs a built RCCL and the ROCm LLVM tools. No GPU.

```bash
cd projects/rccl/test/ll-cache-bits
python3 -m venv venv && ./venv/bin/pip install -r requirements.txt
./venv/bin/python -m pytest -v -rs
```

| Variable | Default | Description |
|----------|---------|-------------|
| `RCCL_LIB` | `$RCCL_BUILD/librccl.so` | library to inspect |
| `RCCL_BUILD` | `<rccl>/build/release`, else `build/debug` | RCCL build dir |
| `ROCM_PATH` | `/opt/rocm` | ROCm install root |

The whole suite skips if the library or tools are missing.

## When it fails

The failure lists the offending loads and the functions they're in. Find out
why the bits changed (compiler, source, or build flags) and fix that. Do not
just edit `EXPECTED_POLL_MODIFIERS` to match.
