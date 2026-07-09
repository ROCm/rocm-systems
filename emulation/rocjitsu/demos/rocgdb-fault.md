# rocgdb-fault

Catch a GPU **memory-access fault (SIGSEGV)** with ROCgdb — in software, with no
AMD GPU required.

A wave that dereferences an unmapped device address is the GPU equivalent of a
segfault. On real hardware it raises a memory-access fault that stops the wave;
on the emulated MI350X the memory pipeline detects the unmapped access, raises
`TRAPSTS.xnack_error`, and reports it to rocm-dbgapi as
`WAVE_STOP_REASON_MEMORY_VIOLATION`, which ROCgdb surfaces as **SIGSEGV** at the
faulting instruction.

This demo builds a tiny HIP kernel that stores through a wild pointer
(`0x0000dead0000`) and runs it under `mirage run --gdb`: ROCgdb stops the GPU
wave with a segmentation fault at the offending store, so you can see the PC and
backtrace of the wave that went out of bounds.

## Run it

Record (builds a portable mirage + rocjitsu, then captures the `.cast`):

    emulation/mirage/scripts/record_demo.sh emulation/rocjitsu/demos/rocgdb-fault.sh

Or run against an already-built `mirage`:

    MIRAGE_BIN=/path/to/mirage bash emulation/rocjitsu/demos/rocgdb-fault.sh

Requires `hipcc` and `rocgdb` on `PATH` (from a ROCm install).

See [rocgdb-debugging.md](../docs/rocgdb-debugging.md) for how the emulator maps
GPU exceptions to debugger stop reasons.
