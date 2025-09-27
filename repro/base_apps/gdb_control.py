#!/usr/bin/env python3
"""
GDB control script for HSA applications with debugger blocks.
This script runs an application under GDB and controls execution through blocks.
"""

import subprocess
import sys
import time
import select
import os

def run_app_with_gdb(app_path):
    """Run an application under GDB control."""

    if not os.path.exists(app_path):
        print(f"Error: Application {app_path} does not exist")
        return 1

    print(f"Starting {app_path} under GDB control")
    print("=" * 60)

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

    block_count = 0

    try:
        while True:
            # Check if process has terminated
            if proc.poll() is not None:
                break

            # Read output with timeout
            output_line = ""
            while True:
                # Use select to check if data is available
                ready, _, _ = select.select([proc.stdout], [], [], 0.1)
                if ready:
                    char = proc.stdout.read(1)
                    if char == '\n':
                        break
                    output_line += char
                else:
                    break

            if output_line:
                print(output_line)

                # Check for block message
                if "AT BLOCK" in output_line:
                    block_count += 1
                    print(f"\n>>> Block {block_count} reached <<<")
                    print(">>> Press Enter to continue execution...")
                    input()

                    # Call app_debugger_continue and continue execution
                    print(">>> Calling app_debugger_continue()...")
                    proc.stdin.write("call app_debugger_continue()\n")
                    proc.stdin.write("continue\n")
                    proc.stdin.flush()

                # Check for program exit
                if "exited normally" in output_line or "Inferior" in output_line and "exited" in output_line:
                    print("\n>>> Application finished execution")
                    break

                # Check for errors
                if "No debugging symbols found" in output_line:
                    print("\nWarning: No debugging symbols found. Compile with -g flag for better debugging.")

    except KeyboardInterrupt:
        print("\n\n>>> Interrupted by user")
        proc.stdin.write("quit\n")
        proc.stdin.write("y\n")
        proc.stdin.flush()

    finally:
        # Clean up
        proc.terminate()
        proc.wait()

    print("=" * 60)
    print(f"Total blocks encountered: {block_count}")
    return 0

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 gdb_control.py <application>")
        print("Example: python3 gdb_control.py ./app1_signal_create")
        return 1

    app_path = sys.argv[1]
    return run_app_with_gdb(app_path)

if __name__ == "__main__":
    sys.exit(main())