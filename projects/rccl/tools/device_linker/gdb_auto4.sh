#!/bin/bash
# Automated gdb script - simpler approach, capture when error occurs

cd /work2/lmeadows/rocm-systems/projects/rccl

export MAX_KERNELS_FOR_TEST=5

cat > /tmp/gdb_auto_commands4.txt << 'EOF'
set environment MAX_KERNELS_FOR_TEST=5
set pagination off
set confirm off
set print elements 200
set print null-stop on
set breakpoint pending on

# Break at the function
break _ZN4llvm9DWARFUnit22tryExtractDIEsIfNeededEb

# Define a command that runs when breakpoint hits
define print_context
  bt 10
  frame 0
  print this
  info locals
  frame 1
  info locals
  frame 2
  info locals
end

# Set command for breakpoint
commands 1
  print_context
  continue
end

run -o /tmp/test_output.o --dispatcher build/release/device_linker_output/dispatcher_device.elf --host-table build/release/hipify/gensrc/host_table.cpp --target gfx942:xnack-:sramecc+ --input-dir build/release/specialized_objs/

# Continue a few times
continue
continue
continue
quit
EOF

echo "Starting gdb (this may take a while)..."
gdb -batch -x /tmp/gdb_auto_commands4.txt --args \
    tools/device_linker/device_linker \
    -o /tmp/test_output.o \
    --dispatcher build/release/device_linker_output/dispatcher_device.elf \
    --host-table build/release/hipify/gensrc/host_table.cpp \
    --target gfx942:xnack-:sramecc+ \
    --input-dir build/release/specialized_objs/ 2>&1 | tee /tmp/gdb_output4.txt

echo ""
echo "=== Extracting key information ==="
# Extract stack traces
grep -B 2 -A 15 "#0.*tryExtractDIEsIfNeeded" /tmp/gdb_output4.txt | head -100
echo ""
# Extract local variables
grep -A 30 "info locals" /tmp/gdb_output4.txt | head -50
