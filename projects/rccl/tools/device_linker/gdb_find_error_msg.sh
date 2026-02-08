#!/bin/bash
# Find where the "invalid abbreviation" error message is printed

cd /work2/lmeadows/rocm-systems/projects/rccl

export MAX_KERNELS_FOR_TEST=5

# First, find the string in the library
echo "Searching for error message string in libLLVM.so..."
STRINGS_OUT=$(strings /COD/LATEST/aomp/llvm/lib/libLLVM.so | grep -i "contains invalid\|valid abbreviations" | head -1)
echo "Found: $STRINGS_OUT"

cat > /tmp/gdb_find_msg.txt << 'EOF'
set environment MAX_KERNELS_FOR_TEST=5
set pagination off
set confirm off
set breakpoint pending on

# Break when the error message string is accessed
# We'll break on any function that might print it
break llvm::WithColor::error
break llvm::errs
break llvm::report_fatal_error

# Also try to break when the string is in memory
# (this is trickier, but we can try)

run -o /tmp/test_output.o --dispatcher build/release/device_linker_output/dispatcher_device.elf --host-table build/release/hipify/gensrc/host_table.cpp --target gfx942:xnack-:sramecc+ --input-dir build/release/specialized_objs/

# When it breaks, check if it's printing our error
commands
  bt 10
  frame 0
  # Check if we're in error printing code
  list
  continue
end

continue
continue
continue
quit
EOF

echo "Running gdb to find error message location..."
gdb -batch -x /tmp/gdb_find_msg.txt --args \
    tools/device_linker/device_linker \
    -o /tmp/test_output.o \
    --dispatcher build/release/device_linker_output/dispatcher_device.elf \
    --host-table build/release/hipify/gensrc/host_table.cpp \
    --target gfx942:xnack-:sramecc+ \
    --input-dir build/release/specialized_objs/ 2>&1 | grep -B 5 -A 20 "invalid abbreviation\|valid abbreviations\|hit Breakpoint" | head -100
