#!/bin/bash
# GDB script to find where "invalid abbreviation" error is printed
# The error likely comes from LLVM's error reporting, not fprintf

cd /work2/lmeadows/rocm-systems/projects/rccl

export MAX_KERNELS_FOR_TEST=5

gdb --batch --ex "set environment MAX_KERNELS_FOR_TEST=5" \
    --ex "break llvm::WithColor::error" \
    --ex "break llvm::errs" \
    --ex "break std::basic_ostream<char>::operator<<" \
    --ex "run -o /tmp/test_output.o --dispatcher build/release/device_linker_output/dispatcher_device.elf --host-table build/release/hipify/gensrc/host_table.cpp --target gfx942:xnack-:sramecc+ --input-dir build/release/specialized_objs/" \
    --ex "bt" \
    --ex "frame 0" \
    --ex "list" \
    --args tools/device_linker/device_linker \
    -o /tmp/test_output.o \
    --dispatcher build/release/device_linker_output/dispatcher_device.elf \
    --host-table build/release/hipify/gensrc/host_table.cpp \
    --target gfx942:xnack-:sramecc+ \
    --input-dir build/release/specialized_objs/ 2>&1 | grep -A 20 "Breakpoint\|invalid abbreviation"
