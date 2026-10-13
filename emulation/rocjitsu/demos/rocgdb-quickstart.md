# rocgdb-quickstart

Debug a real GPU kernel with a single command — entirely in software, with no
AMD GPU required.

`mirage run --gdb` boots the emulated MI350X, launches the workload under
ROCgdb, and makes kernel breakpoints pending so `break <kernel>` resolves when
the GPU code object loads at dispatch. This demo builds a tiny HIP kernel and
drives ROCgdb through it: setting a breakpoint on the GPU kernel, running to the
stopped wave, reading its source-level arguments and locals (`info args`,
`print n`, `print data[0]`), and continuing to completion.

It is the fastest way to see the emulated GPU debugger working end to end, and
the same flow (`mirage run --gdb -- ./your_app`) drops you into an interactive
ROCgdb session on your own kernels.

## Run it

Record (builds a portable mirage + rocjitsu, then captures the `.cast`):

    emulation/mirage/scripts/record_demo.sh emulation/rocjitsu/demos/rocgdb-quickstart.sh

Or run against an already-built `mirage`:

    MIRAGE_BIN=/path/to/mirage bash emulation/rocjitsu/demos/rocgdb-quickstart.sh

Requires `hipcc` and `rocgdb` on `PATH` (from a ROCm install).
