#!/bin/bash

# Demonstration script for HSA applications with debugger blocks
# This shows the user how to use the GDB control functionality

echo "HSA Debugger Block Demonstration"
echo "================================"
echo ""
echo "This demo shows how to use the debugger block functionality"
echo "that has been added to all 8 HSA applications:"
echo ""
echo "1. app1_signal_create     - Blocks before/after hsa_signal_create"
echo "2. app2_signal_store      - Blocks before/after hsa_signal_store_relaxed"
echo "3. app3_barrier_packet    - Blocks before/after packet submission"
echo "4. app4_dual_barrier      - Blocks before/after second packet submission"
echo "5. app5_cpu_memory_pool   - Blocks before/after memory pool writes"
echo "6. app6_aql_start_packet  - Blocks before/after AQL start packet submission"
echo "7. app7_aql_read_packet   - Blocks before/after AQL read packet submission"
echo "8. app8_aql_stop_packet   - Blocks before/after AQL stop packet submission"
echo ""
echo "Each application will:"
echo "- Run normally until it hits a debugger block"
echo "- Print 'AT BLOCK' to stderr"
echo "- Wait indefinitely until app_debugger_continue() is called"
echo "- Continue execution until the next block (if any)"
echo ""
echo "Usage options:"
echo ""
echo "Option 1: Manual GDB control"
echo "  gdb ./app1_signal_create"
echo "  (gdb) run"
echo "  # Wait for 'AT BLOCK' message"
echo "  (gdb) call app_debugger_continue()"
echo "  # Repeat for each block"
echo ""
echo "Option 2: Python GDB control script (interactive)"
echo "  python3 gdb_control.py ./app1_signal_create"
echo "  # Press Enter when prompted at each block"
echo ""
echo "Option 3: Automated testing script"
echo "  python3 test_blocks.py"
echo "  # Tests all applications automatically"
echo ""
echo "Would you like to try a quick demo? (y/n): "
read -n 1 answer
echo ""

if [[ $answer == "y" || $answer == "Y" ]]; then
    echo ""
    echo "Running app1_signal_create with manual GDB control..."
    echo "Instructions:"
    echo "1. The app will start and hit the first block"
    echo "2. Type: call app_debugger_continue()"
    echo "3. Type: continue"
    echo "4. Repeat when the second block is reached"
    echo "5. Type: quit to exit GDB when done"
    echo ""
    echo "Starting GDB..."
    sleep 2

    gdb -quiet ./app1_signal_create -ex "run"
else
    echo "Demo skipped. You can try the commands above manually."
fi

echo ""
echo "All HSA applications are ready for debugging with block functionality!"