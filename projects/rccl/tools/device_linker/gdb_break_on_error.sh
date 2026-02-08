#!/bin/bash
# Break when the actual error message is printed

cd /work2/lmeadows/rocm-systems/projects/rccl

export MAX_KERNELS_FOR_TEST=5

cat > /tmp/gdb_break_error.txt << 'EOF'
set environment MAX_KERNELS_FOR_TEST=5
set pagination off
set confirm off
set breakpoint pending on
set print elements 200

# The error message format is: "DWARF unit at offset 0x%8.8lx contains invalid abbreviation %lu at offset 0x%8.8lx, valid abbreviations are %s"
# This is likely printed via WithColor::error or similar. Let's break on common error printing functions.

# Break on WithColor::error (most likely)
break llvm::WithColor::error

# Also break on the actual printf/format call if we can find it
# The format string contains "contains invalid abbreviation" and "valid abbreviations are"

run -o /tmp/test_output.o --dispatcher build/release/device_linker_output/dispatcher_device.elf --host-table build/release/hipify/gensrc/host_table.cpp --target gfx942:xnack-:sramecc+ --input-dir build/release/specialized_objs/

# When WithColor::error breaks, check the stack to see if it's our error
commands
  # Check the stack to see where we are
  bt 15
  frame 0
  # Check arguments - WithColor::error takes a Twine or similar
  info args
  # Try to print what's being printed
  print $rdi
  # Go up to see the caller
  up
  bt 5
  frame 0
  info locals
  # Continue to see if this is the right error
  continue
end

# Continue until we see the error or program exits
continue
continue
continue
continue
quit
EOF

echo "Running gdb to find where error is printed..."
gdb -batch -x /tmp/gdb_break_error.txt --args \
    tools/device_linker/device_linker \
    -o /tmp/test_output.o \
    --dispatcher build/release/device_linker_output/dispatcher_device.elf \
    --host-table build/release/hipify/gensrc/host_table.cpp \
    --target gfx942:xnack-:sramecc+ \
    --input-dir build/release/specialized_objs/ 2>&1 | tee /tmp/gdb_error_location.txt

echo ""
echo "=== Extracting error location ==="
grep -B 10 -A 30 "hit Breakpoint.*WithColor::error" /tmp/gdb_error_location.txt | grep -A 30 "bt\|frame\|info locals" | head -80
