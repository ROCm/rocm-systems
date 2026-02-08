#!/bin/bash
# GDB script to find where "invalid abbreviation" error comes from

cd /work2/lmeadows/rocm-systems/projects/rccl

export MAX_KERNELS_FOR_TEST=5

cat > /tmp/gdb_find_error.txt << 'EOF'
set environment MAX_KERNELS_FOR_TEST=5

# Break at the location where error is detected (from stack trace)
# The error happens in DWARFUnit::tryExtractDIEsIfNeeded at line 511
break DWARFUnit::tryExtractDIEsIfNeeded

# Also break on the error message if we can find where it's printed
# Since it's not fprintf, try breaking on WithColor or errs()
break llvm::WithColor::error
break llvm::errs

run -o /tmp/test_output.o --dispatcher build/release/device_linker_output/dispatcher_device.elf --host-table build/release/hipify/gensrc/host_table.cpp --target gfx942:xnack-:sramecc+ --input-dir build/release/specialized_objs/

# When it breaks, check what's being read
commands
  bt
  frame 0
  # Print 'this' pointer to see which DWARFUnit
  print this
  # Check if there's a data pointer or buffer
  print *this
  continue
end

# Continue until we see the error
continue
EOF

gdb -x /tmp/gdb_find_error.txt --args \
    tools/device_linker/device_linker \
    -o /tmp/test_output.o \
    --dispatcher build/release/device_linker_output/dispatcher_device.elf \
    --host-table build/release/hipify/gensrc/host_table.cpp \
    --target gfx942:xnack-:sramecc+ \
    --input-dir build/release/specialized_objs/
