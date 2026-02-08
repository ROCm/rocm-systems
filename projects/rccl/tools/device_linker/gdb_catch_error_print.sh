#!/bin/bash
# Catch when the error message is actually printed to stderr

cd /work2/lmeadows/rocm-systems/projects/rccl

export MAX_KERNELS_FOR_TEST=5

cat > /tmp/gdb_catch_print.txt << 'EOF'
set environment MAX_KERNELS_FOR_TEST=5
set pagination off
set confirm off
set breakpoint pending on

# Catch writes to stderr (file descriptor 2) that contain our error message
# We'll break on the write syscall and check if it's writing our error string
catch syscall write
condition $bpnum $rdi == 2

# Define command to check if this write contains our error
define check_write
  # $rdi = fd, $rsi = buf, $rdx = count
  # We can't easily check the buffer content, but we can see the call stack
  bt 10
  frame 0
  # Try to print the buffer (might not work if optimized)
  x/s $rsi
  continue
end

commands
  check_write
end

run -o /tmp/test_output.o --dispatcher build/release/device_linker_output/dispatcher_device.elf --host-table build/release/hipify/gensrc/host_table.cpp --target gfx942:xnack-:sramecc+ --input-dir build/release/specialized_objs/

# Continue until we see the error or program exits
continue
continue
continue
continue
continue
quit
EOF

echo "Running gdb to catch error message print..."
gdb -batch -x /tmp/gdb_catch_print.txt --args \
    tools/device_linker/device_linker \
    -o /tmp/test_output.o \
    --dispatcher build/release/device_linker_output/dispatcher_device.elf \
    --host-table build/release/hipify/gensrc/host_table.cpp \
    --target gfx942:xnack-:sramecc+ \
    --input-dir build/release/specialized_objs/ 2>&1 | grep -B 5 -A 20 "contains invalid abbreviation\|bt\|frame" | head -100
