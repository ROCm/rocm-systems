# rocgdb-multiwave

Debug a real **multi-wave** GPU kernel with ROCgdb — in software, with no AMD
GPU required.

Real GPU kernels are almost always multi-wave: a workgroup spans many 64-lane
wavefronts, and there are usually many workgroups. This demo launches one
workgroup of 128 threads — exactly two 64-lane waves — and sets a breakpoint
inside the kernel so **both waves trap together**.

ROCgdb (through rocm-dbgapi) correlates each trapped wave to its dispatch and
workgroup: `info threads` shows both waves of workgroup `(0,0,0)` at positions
`/0` and `/1`, and reading the scratch-resident `local` in each wave returns
that wave's own value (`3` for global thread 0, `451` for global thread 64) —
proving per-wave private memory and workgroup correlation, not one wave's state
smeared across both.

This is the capability that makes the emulated debugger practical for real
kernels rather than single-wave toys.

## Run it

Record (builds a portable mirage + rocjitsu, then captures the `.cast`):

    emulation/mirage/scripts/record_demo.sh emulation/rocjitsu/demos/rocgdb-multiwave.sh

Or run against an already-built `mirage`:

    MIRAGE_BIN=/path/to/mirage bash emulation/rocjitsu/demos/rocgdb-multiwave.sh

Requires `hipcc` and `rocgdb` on `PATH` (from a ROCm install).

See [rocgdb-debugging.md](../docs/rocgdb-debugging.md) for how the emulator
serializes a stable multi-wave control stack and maps per-wave scratch.
