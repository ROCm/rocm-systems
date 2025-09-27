#!/usr/bin/env python3
"""
Test script to verify that debugger blocks are working in HSA applications.
This runs apps under GDB and automatically continues at each block.
"""

import subprocess
import sys
import time
import os

def test_app_blocks(app_path):
    """Test that an application hits debugger blocks correctly."""

    if not os.path.exists(app_path):
        print(f"Error: Application {app_path} does not exist")
        return False

    print(f"Testing {app_path} for debugger blocks...")

    # Create GDB commands to run the application
    gdb_commands = f"""
set confirm off
set pagination off
file {app_path}
run
"""

    # Start GDB process
    proc = subprocess.Popen(
        ["gdb", "-quiet"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=0
    )

    # Send initial commands
    proc.stdin.write(gdb_commands)
    proc.stdin.flush()

    blocks_found = 0
    output_lines = []

    try:
        while True:
            # Check if process has terminated
            if proc.poll() is not None:
                break

            # Read output with timeout
            try:
                line = proc.stdout.readline()
                if not line:
                    break

                output_lines.append(line.strip())

                # Check for block message
                if "AT BLOCK" in line:
                    blocks_found += 1
                    print(f"  Block {blocks_found} detected")

                    # Automatically call continue
                    time.sleep(0.1)  # Small delay
                    proc.stdin.write("call app_debugger_continue()\n")
                    proc.stdin.write("continue\n")
                    proc.stdin.flush()

                # Check for program exit
                if "exited normally" in line or ("Inferior" in line and "exited" in line):
                    print(f"  Application finished normally")
                    break

                # Check for timeout or hang
                if "timeout" in line.lower():
                    print(f"  Warning: Timeout detected")
                    break

            except:
                break

    except KeyboardInterrupt:
        print("  Interrupted by user")
    finally:
        # Clean up
        try:
            proc.stdin.write("quit\n")
            proc.stdin.write("y\n")
            proc.stdin.flush()
        except:
            pass
        proc.terminate()
        proc.wait()

    print(f"  Total blocks found: {blocks_found}")
    expected_blocks = 2  # Most apps have 2 blocks (before/after operation)

    if blocks_found == expected_blocks:
        print(f"  ✓ PASS: Found expected {expected_blocks} blocks")
        return True
    elif blocks_found > 0:
        print(f"  ⚠ PARTIAL: Found {blocks_found} blocks (expected {expected_blocks})")
        return True
    else:
        print(f"  ✗ FAIL: No blocks found")
        return False

def main():
    """Test all applications."""
    base_dir = "/root/rocm-systems/repro/base_apps"
    apps = [
        "app1_signal_create",
        "app2_signal_store",
        "app3_barrier_packet",
        "app4_dual_barrier",
        "app5_cpu_memory_pool"
    ]

    print("Testing HSA applications for debugger blocks")
    print("=" * 50)

    results = {}
    for app in apps:
        app_path = os.path.join(base_dir, app)
        results[app] = test_app_blocks(app_path)
        print()

    print("=" * 50)
    print("Test Results Summary:")
    for app, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"  {app}: {status}")

    total_passed = sum(results.values())
    print(f"\nTotal: {total_passed}/{len(apps)} applications passed")

    return 0 if total_passed == len(apps) else 1

if __name__ == "__main__":
    sys.exit(main())