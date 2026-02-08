#!/bin/bash
# Automated gdb script - use pending breakpoints to avoid loading full symbol table

cd /work2/lmeadows/rocm-systems/projects/rccl

export MAX_KERNELS_FOR_TEST=5

cat > /tmp/gdb_auto_commands3.txt << 'EOF'
set environment MAX_KERNELS_FOR_TEST=5
set pagination off
set confirm off
set print elements 0
set print null-stop on

# Allow pending breakpoints (don't fail if symbol not loaded yet)
set breakpoint pending on

# Break at the mangled function name we found
# _ZN4llvm9DWARFUnit22tryExtractDIEsIfNeededEb
break _ZN4llvm9DWARFUnit22tryExtractDIEsIfNeededEb

# Also try breaking at the source file:line (will be pending until file loads)
break DWARFUnit.cpp:511

# Run the program (symbols will load as needed)
run -o /tmp/test_output.o --dispatcher build/release/device_linker_output/dispatcher_device.elf --host-table build/release/hipify/gensrc/host_table.cpp --target gfx942:xnack-:sramecc+ --input-dir build/release/specialized_objs/

# When it breaks, print information
commands
  echo \n=== STACK TRACE (first 15 frames) ===\n
  bt 15
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

echo "Starting gdb (this may take a while to load symbols)..."
gdb -batch -x /tmp/gdb_auto_commands3.txt --args \
    tools/device_linker/device_linker \
    -o /tmp/test_output.o \
    --dispatcher build/release/device_linker_output/dispatcher_device.elf \
    --host-table build/release/hipify/gensrc/host_table.cpp \
    --target gfx942:xnack-:sramecc+ \
    --input-dir build/release/specialized_objs/ 2>&1 | tee /tmp/gdb_output3.txt

echo ""
echo "=== Key information from gdb ==="
grep -A 40 "STACK TRACE" /tmp/gdb_output3.txt | head -50
echo ""
echo "=== Local variables ==="
grep -A 20 "LOCAL VARIABLES" /tmp/gdb_output3.txt | head -30
