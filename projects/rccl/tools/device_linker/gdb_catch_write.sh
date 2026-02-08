#!/bin/bash
# GDB script using catch syscall to find where error is written

cd /work2/lmeadows/rocm-systems/projects/rccl

export MAX_KERNELS_FOR_TEST=5

cat > /tmp/gdb_commands.txt << 'EOF'
set environment MAX_KERNELS_FOR_TEST=5
# Catch writes to stderr (fd 2)
catch syscall write
condition $bpnum $rdi == 2
commands
  # Check if the write contains our error string
  # We can't easily check the buffer, but we can see the call stack
  bt
  continue
end
run -o /tmp/test_output.o --dispatcher build/release/device_linker_output/dispatcher_device.elf --host-table build/release/hipify/gensrc/host_table.cpp --target gfx942:xnack-:sramecc+ --input-dir build/release/specialized_objs/
EOF

gdb -x /tmp/gdb_commands.txt --args \
    tools/device_linker/device_linker \
    -o /tmp/test_output.o \
    --dispatcher build/release/device_linker_output/dispatcher_device.elf \
    --host-table build/release/hipify/gensrc/host_table.cpp \
    --target gfx942:xnack-:sramecc+ \
    --input-dir build/release/specialized_objs/
