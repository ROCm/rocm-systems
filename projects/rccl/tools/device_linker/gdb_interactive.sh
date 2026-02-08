#!/bin/bash
# Interactive gdb session to manually find the error

cd /work2/lmeadows/rocm-systems/projects/rccl

export MAX_KERNELS_FOR_TEST=5

gdb --args \
    tools/device_linker/device_linker \
    -o /tmp/test_output.o \
    --dispatcher build/release/device_linker_output/dispatcher_device.elf \
    --host-table build/release/hipify/gensrc/host_table.cpp \
    --target gfx942:xnack-:sramecc+ \
    --input-dir build/release/specialized_objs/

# In gdb, try these commands:
# (gdb) break llvm::WithColor::error
# (gdb) break llvm::errs
# (gdb) break std::basic_ostream<char>::operator<<
# (gdb) catch throw
# (gdb) run
# When it prints the error, use:
# (gdb) bt
# (gdb) frame <number>
# (gdb) list
