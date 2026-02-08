#!/bin/bash
# Simple gdb session - just run gdb interactively

cd /work2/lmeadows/rocm-systems/projects/rccl

export MAX_KERNELS_FOR_TEST=5

# Start gdb with device_linker
gdb --args \
    tools/device_linker/device_linker \
    -o /tmp/test_output.o \
    --dispatcher build/release/device_linker_output/dispatcher_device.elf \
    --host-table build/release/hipify/gensrc/host_table.cpp \
    --target gfx942:xnack-:sramecc+ \
    --input-dir build/release/specialized_objs/

# Then in gdb, run these commands:
# (gdb) break fprintf
# (gdb) condition 1 strstr((char*)$rdi, "invalid abbreviation") != 0 || strstr((char*)$rdi, "valid abbreviations") != 0
# (gdb) run
# (gdb) bt
# (gdb) frame 0
# (gdb) list
