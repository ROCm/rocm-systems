#!/bin/bash

# Simple script to run HSA applications with debugger blocks
# Usage: ./run_with_blocks.sh <application>

if [ $# -ne 1 ]; then
    echo "Usage: $0 <application>"
    echo "Example: $0 ./app1_signal_create"
    exit 1
fi

APP=$1

if [ ! -f "$APP" ]; then
    echo "Error: Application $APP does not exist"
    exit 1
fi

echo "Running $APP with GDB control"
echo "============================================"
echo "When you see 'AT BLOCK', press Enter to continue"
echo "============================================"

# Create a GDB script
cat > /tmp/gdb_commands.txt << EOF
set confirm off
set pagination off
file $APP

define handle_block
  echo >>> Block reached, press Enter to continue...\n
  shell read -p ""
  call app_debugger_continue()
  continue
end

# Set a breakpoint on the block function output
catch syscall write
commands
  silent
  # Check if we're writing "AT BLOCK"
  if \$rdi == 2
    handle_block
  end
  continue
end

run
EOF

gdb -batch -x /tmp/gdb_commands.txt 2>&1 | grep -v "^Catchpoint" | grep -v "^Continuing"

rm -f /tmp/gdb_commands.txt