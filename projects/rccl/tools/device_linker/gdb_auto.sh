#!/bin/bash
# Automated gdb script to find where the error occurs

cd /work2/lmeadows/rocm-systems/projects/rccl

export MAX_KERNELS_FOR_TEST=5

cat > /tmp/gdb_auto_commands.txt << 'EOF'
set environment MAX_KERNELS_FOR_TEST=5
set pagination off
set confirm off

# Break at the location where error is detected
break DWARFUnit::tryExtractDIEsIfNeeded

# Run the program
run -o /tmp/test_output.o --dispatcher build/release/device_linker_output/dispatcher_device.elf --host-table build/release/hipify/gensrc/host_table.cpp --target gfx942:xnack-:sramecc+ --input-dir build/release/specialized_objs/

# When it breaks, print information
commands
  echo \n=== STACK TRACE ===\n
  bt
  echo \n=== CURRENT FRAME (DWARFUnit::tryExtractDIEsIfNeeded) ===\n
  frame 0
  echo \n=== THIS POINTER ===\n
  print this
  echo \n=== LOCAL VARIABLES ===\n
  info locals
  echo \n=== GOING UP STACK ===\n
  up
  info locals
  up
  info locals
  echo \n=== CONTINUING ===\n
  continue
end

# Continue until program exits or crashes
continue
quit
EOF

gdb -batch -x /tmp/gdb_auto_commands.txt --args \
    tools/device_linker/device_linker \
    -o /tmp/test_output.o \
    --dispatcher build/release/device_linker_output/dispatcher_device.elf \
    --host-table build/release/hipify/gensrc/host_table.cpp \
    --target gfx942:xnack-:sramecc+ \
    --input-dir build/release/specialized_objs/ 2>&1 | tee /tmp/gdb_output.txt

echo ""
echo "=== GDB output saved to /tmp/gdb_output.txt ==="
echo "Looking for key information..."

grep -A 20 "STACK TRACE" /tmp/gdb_output.txt | head -30
grep -A 10 "LOCAL VARIABLES" /tmp/gdb_output.txt | head -20
