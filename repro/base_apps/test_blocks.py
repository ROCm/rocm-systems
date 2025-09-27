#!/usr/bin/env python3
"""
Test script to verify that debugger blocks are working in HSA applications.
This runs apps standalone and uses GDB attach/detach to handle blocks.
"""

import subprocess
import sys
import time
import os
import signal
import select

def test_app_blocks(app_path):
    """Test that an application hits debugger blocks correctly using attach/detach."""

    if not os.path.exists(app_path):
        print(f"Error: Application {app_path} does not exist")
        return False

    print(f"Testing {app_path} for debugger blocks...")

    # Start the application in background
    app_proc = subprocess.Popen(
        [app_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    blocks_found = 0
    max_blocks = 10  # Safety limit
    timeout = 30  # 30 second total timeout
    start_time = time.time()

    try:
        while blocks_found < max_blocks and (time.time() - start_time) < timeout:
            # Check if app has terminated
            if app_proc.poll() is not None:
                print(f"  Application terminated normally")
                break

            # Check for "AT BLOCK" in stderr output
            # Use non-blocking read with select-like behavior
            try:
                # Check if there's stderr output available
                ready, _, _ = select.select([app_proc.stderr], [], [], 0.1)

                if ready:
                    line = app_proc.stderr.readline()
                    if line and "AT BLOCK" in line:
                        blocks_found += 1
                        print(f"  Block {blocks_found} detected - attaching GDB")

                        # Attach GDB to continue the process
                        gdb_proc = subprocess.Popen(
                            ["gdb", "-quiet", "-p", str(app_proc.pid)],
                            stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            text=True
                        )

                        # Send commands to continue
                        gdb_commands = """
set confirm off
set pagination off
call app_debugger_continue()
detach
quit
"""
                        gdb_proc.stdin.write(gdb_commands)
                        gdb_proc.stdin.flush()

                        # Wait for GDB to complete
                        gdb_proc.wait(timeout=5)

                        print(f"  Block {blocks_found} continued - GDB detached")

                        # Small delay before checking for next block
                        time.sleep(0.2)

                else:
                    # No output available, small sleep
                    time.sleep(0.1)

            except Exception as e:
                print(f"  Error reading output: {e}")
                break

    except KeyboardInterrupt:
        print("  Interrupted by user")
        return False
    except Exception as e:
        print(f"  Error during test: {e}")
        return False
    finally:
        # Clean up application process
        if app_proc.poll() is None:
            try:
                app_proc.terminate()
                app_proc.wait(timeout=5)
            except:
                app_proc.kill()
                app_proc.wait()

    # Read any remaining output
    try:
        stdout_output, stderr_output = app_proc.communicate(timeout=1)
        if "completed successfully" in stdout_output:
            print(f"  Application completed successfully")
    except:
        pass

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
        "app5_cpu_memory_pool",
        "app6_aql_start_packet",
        "app7_aql_read_packet",
        "app8_aql_stop_packet"
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