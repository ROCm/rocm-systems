# rocgdb-watchpoint

Catch which GPU wave writes a buffer with a **hardware data watchpoint** — in
software, with no AMD GPU required.

Hardware watchpoints (`watch`/`rwatch`/`awatch`) are how you find the code that
touches a piece of memory without single-stepping the whole program. On the
emulated MI350X, ROCgdb programs one of the GPU's four address-watch registers
and the emulator traps the wave whose global-memory access hits the watched
address — reporting the old and new value and stopping at the exact store.

This demo builds a tiny HIP kernel that adds 1 to every element of a device
buffer, sets a watchpoint on the buffer from GPU context, and continues: the
kernel's `data[i] += 1` store trips the watchpoint (`Old value = 0`,
`New value = 1`), stopped at the source line that wrote it.

## Run it

Record (builds a portable mirage + rocjitsu, then captures the `.cast`):

    emulation/mirage/scripts/record_demo.sh emulation/rocjitsu/demos/rocgdb-watchpoint.sh

Or run against an already-built `mirage`:

    MIRAGE_BIN=/path/to/mirage bash emulation/rocjitsu/demos/rocgdb-watchpoint.sh

Requires `hipcc` and `rocgdb` on `PATH` (from a ROCm install).

## How it works

`break add_one.hip:<launch>` stops on the host at the kernel launch so ROCgdb can
read the device pointer `d` (no hard-coded VA); `watch *(int*)$waddr` programs a
hardware watchpoint on that address. When the GPU wave executes `data[i] += 1`,
the emulator's memory pipeline matches the access against the debugger's watch
registers, raises `TRAPSTS.addr_watch`, and stops the wave — exactly as gfx9
hardware does. See [rocgdb-debugging.md](../docs/rocgdb-debugging.md).
