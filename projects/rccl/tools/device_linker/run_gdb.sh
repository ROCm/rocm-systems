#!/bin/bash
# Run device_linker under gdb to find error location
# 
# Required arguments:
#   -o <output.elf>                    - Output ELF file
#   --dispatcher <dispatcher.elf>      - Dispatcher device ELF file
#
# Optional arguments:
#   --host-table <table.cpp>           - Host table file (for funcId mappings)
#   --target <arch>                    - Target architecture (auto-detected if not provided)
#   --input-dir <dir>                  - Directory containing *.device.o files

cd /work2/lmeadows/rocm-systems/projects/rccl

export MAX_KERNELS_FOR_TEST=5

gdb -x tools/device_linker/gdb_commands.txt --args \
    tools/device_linker/device_linker \
    -o /tmp/test_output.o \
    --dispatcher build/release/device_linker_output/dispatcher_device.elf \
    --host-table build/release/hipify/gensrc/host_table.cpp \
    --target gfx942:xnack-:sramecc+ \
    --input-dir build/release/specialized_objs/
