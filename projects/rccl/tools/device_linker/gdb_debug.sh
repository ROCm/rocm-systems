#!/bin/bash
# Script to run device_linker under gdb to find where "invalid abbreviation" error is printed

cd /work2/lmeadows/rocm-systems/projects/rccl

# Run gdb with the device_linker
gdb --args \
    tools/device_linker/device_linker \
    -o /tmp/test_output.o \
    --dispatcher build/release/device_linker_output/dispatcher_device.elf \
    --host-table build/release/hipify/gensrc/host_table.cpp \
    --target gfx942:xnack-:sramecc+ \
    --input-dir build/release/specialized_objs/ \
    --env MAX_KERNELS_FOR_TEST=5

# GDB commands to run:
# (gdb) break fprintf
# (gdb) condition 1 $_streq((char*)$rdi, "stderr") || strstr((char*)$rdi, "invalid abbreviation") != 0
# (gdb) run
# (gdb) bt
# (gdb) frame <number>
# (gdb) list
