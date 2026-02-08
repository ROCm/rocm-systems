#!/bin/bash
# Automated gdb script - break on error message or use file:line

cd /work2/lmeadows/rocm-systems/projects/rccl

export MAX_KERNELS_FOR_TEST=5

cat > /tmp/gdb_auto_commands2.txt << 'EOF'
set environment MAX_KERNELS_FOR_TEST=5
set pagination off
set confirm off

# Break at the source file and line from the stack trace
# The error happens at DWARFUnit.cpp:511
break /work1/lmeadows/git/aomp23.0/llvm-project/llvm/lib/DebugInfo/DWARF/DWARFUnit.cpp:511

# Alternative: break on any function with "ExtractDIEs" in the name
# rbreak .*ExtractDIEs.*

# Run the program
run -o /tmp/test_output.o --dispatcher build/release/device_linker_output/dispatcher_device.elf --host-table build/release/hipify/gensrc/host_table.cpp --target gfx942:xnack-:sramecc+ --input-dir build/release/specialized_objs/

# When it breaks, print information
commands
  echo \n=== STACK TRACE ===\n
  bt 20
  echo \n=== CURRENT FRAME ===\n
  frame 0
  echo \n=== THIS POINTER ===\n
  print this
  echo \n=== LOCAL VARIABLES ===\n
  info locals
  echo \n=== ARGUMENTS ===\n
  info args
  echo \n=== GOING UP STACK (frame 1) ===\n
  frame 1
  info locals
  echo \n=== GOING UP STACK (frame 2) ===\n
  frame 2
  info locals
  echo \n=== CONTINUING ===\n
  continue
end

# Continue a few times to catch multiple occurrences
continue
continue
continue
quit
EOF

gdb -batch -x /tmp/gdb_auto_commands2.txt --args \
    tools/device_linker/device_linker \
    -o /tmp/test_output.o \
    --dispatcher build/release/device_linker_output/dispatcher_device.elf \
    --host-table build/release/hipify/gensrc/host_table.cpp \
    --target gfx942:xnack-:sramecc+ \
    --input-dir build/release/specialized_objs/ 2>&1 | tee /tmp/gdb_output2.txt

echo ""
echo "=== Key information from gdb ==="
grep -A 30 "STACK TRACE" /tmp/gdb_output2.txt | head -40
grep -A 15 "LOCAL VARIABLES" /tmp/gdb_output2.txt | head -25
