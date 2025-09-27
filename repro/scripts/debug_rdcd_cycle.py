#!/usr/bin/env python3
"""
Automated RDC Debugging Script

This script:
1. Launches rdcd with -u flag
2. Waits 1 second, then launches rdci with specified parameters
3. Monitors rdcd stderr for "waiting for debugger" messages
4. When detected, attaches GDB to rdcd process
5. Waits for user confirmation before calling rocprofiler_debugger_continue()
6. Detaches and waits for the next "waiting for debugger" occurrence
7. Repeats the cycle until terminated
"""

import subprocess
import time
import threading
import queue
import signal
import sys
import os
from typing import Optional

class RDCDebugger:
    def __init__(self):
        self.rdcd_process: Optional[subprocess.Popen] = None
        self.rdci_process: Optional[subprocess.Popen] = None
        self.rdcd_pid: Optional[int] = None
        self.stderr_queue = queue.Queue()
        self.stderr_thread: Optional[threading.Thread] = None
        self.running = True

        # Paths
        self.rdcd_path = "~/worktress/blocking_counter/projects/rdc/build/server/rdcd"
        self.rdci_path = "~/worktress/blocking_counter/projects/rdc/build/rdci/rdci"

    def cleanup(self, signum=None, frame=None):
        """Clean up processes on exit"""
        print("\n[DEBUG] Cleaning up processes...")
        self.running = False

        if self.rdci_process:
            try:
                self.rdci_process.terminate()
                self.rdci_process.wait(timeout=5)
            except:
                self.rdci_process.kill()

        if self.rdcd_process:
            try:
                self.rdcd_process.terminate()
                self.rdcd_process.wait(timeout=5)
            except:
                self.rdcd_process.kill()

        sys.exit(0)

    def stderr_reader(self, process):
        """Thread function to read stderr and put lines in queue"""
        try:
            for line in iter(process.stderr.readline, b''):
                if not self.running:
                    break
                line_str = line.decode('utf-8', errors='ignore').strip()
                if line_str:
                    self.stderr_queue.put(line_str)
        except:
            pass

    def launch_rdcd(self):
        """Launch rdcd process and start monitoring its stderr"""
        print(f"[DEBUG] Launching rdcd: {self.rdcd_path} -u")

        try:
            self.rdcd_process = subprocess.Popen(
                [os.path.expanduser(self.rdcd_path), "-u"],
                stderr=subprocess.PIPE,
                stdout=subprocess.PIPE,
                bufsize=1,
                universal_newlines=False
            )

            self.rdcd_pid = self.rdcd_process.pid
            print(f"[DEBUG] rdcd started with PID: {self.rdcd_pid}")

            # Start stderr monitoring thread
            self.stderr_thread = threading.Thread(
                target=self.stderr_reader,
                args=(self.rdcd_process,),
                daemon=True
            )
            self.stderr_thread.start()

            return True

        except Exception as e:
            print(f"[ERROR] Failed to launch rdcd: {e}")
            return False

    def launch_rdci(self):
        """Launch rdci process after delay"""
        print("[DEBUG] Waiting 1 second before launching rdci...")
        time.sleep(1)

        rdci_cmd = [
            os.path.expanduser(self.rdci_path),
            "dmon", "-u", "-e", "RDC_FI_PROF_EVAL_FLOPS_64", "-i", "0"
        ]

        print(f"[DEBUG] Launching rdci: {' '.join(rdci_cmd)}")

        try:
            self.rdci_process = subprocess.Popen(
                rdci_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            print(f"[DEBUG] rdci started with PID: {self.rdci_process.pid}")
            return True

        except Exception as e:
            print(f"[ERROR] Failed to launch rdci: {e}")
            return False

    def attach_gdb_and_continue(self):
        """Attach GDB to rdcd, wait for user confirmation, call continue function, then detach"""
        if not self.rdcd_pid:
            print("[ERROR] No rdcd PID available for GDB attachment")
            return False

        print(f"\n[DEBUG] Attaching GDB to rdcd process (PID: {self.rdcd_pid})")

        # Create GDB commands
        gdb_commands = f"""
set confirm off
attach {self.rdcd_pid}
call (void)rocprofiler_debugger_continue()
detach
quit
"""

        try:
            # Wait for user confirmation
            print("[PROMPT] Press Enter to execute rocprofiler_debugger_continue() or 'q' to quit: ", end="", flush=True)
            user_input = input().strip().lower()

            if user_input == 'q':
                print("[DEBUG] User requested quit")
                return False

            print("[DEBUG] Executing GDB commands...")

            # Run GDB with commands
            gdb_process = subprocess.Popen(
                ["gdb", "-batch", "-ex", f"attach {self.rdcd_pid}",
                 "-ex", "call (void)rocprofiler_debugger_continue()",
                 "-ex", "detach", "-ex", "quit"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            stdout, stderr = gdb_process.communicate(timeout=30)

            if gdb_process.returncode == 0:
                print("[DEBUG] GDB commands executed successfully")
                print("[DEBUG] Detached from process, waiting for next 'waiting for debugger' message...")
                return True
            else:
                print(f"[ERROR] GDB failed with return code {gdb_process.returncode}")
                if stderr:
                    print(f"[ERROR] GDB stderr: {stderr}")
                return False

        except subprocess.TimeoutExpired:
            print("[ERROR] GDB command timed out")
            gdb_process.kill()
            return False
        except Exception as e:
            print(f"[ERROR] GDB execution failed: {e}")
            return False

    def monitor_and_debug(self):
        """Main loop to monitor stderr and handle debugging"""
        print("[DEBUG] Monitoring rdcd stderr for 'waiting for debugger' messages...")
        print("[DEBUG] Press Ctrl+C to stop\n")

        debugger_count = 0

        while self.running:
            try:
                # Check if processes are still running
                if self.rdcd_process and self.rdcd_process.poll() is not None:
                    print("[WARNING] rdcd process has terminated")
                    break

                # Check stderr queue for new messages
                try:
                    stderr_line = self.stderr_queue.get(timeout=1)
                    print(f"[STDERR] {stderr_line}")

                    # Check for the trigger message
                    if "waiting for debugger" in stderr_line.lower():
                        debugger_count += 1
                        print(f"\n[DETECTED] 'waiting for debugger' message #{debugger_count}")

                        if not self.attach_gdb_and_continue():
                            break

                        print(f"[DEBUG] Continuing to monitor for next occurrence...\n")

                except queue.Empty:
                    continue

            except KeyboardInterrupt:
                print("\n[DEBUG] Received interrupt signal")
                break

        print("[DEBUG] Monitoring stopped")

    def run(self):
        """Main execution function"""
        # Set up signal handlers
        signal.signal(signal.SIGINT, self.cleanup)
        signal.signal(signal.SIGTERM, self.cleanup)

        try:
            # Launch rdcd
            if not self.launch_rdcd():
                return False

            # Launch rdci
            if not self.launch_rdci():
                return False

            # Start monitoring and debugging cycle
            self.monitor_and_debug()

        except Exception as e:
            print(f"[ERROR] Unexpected error: {e}")
        finally:
            self.cleanup()

        return True

def main():
    print("=== RDC Automated Debugging Script ===")
    print("This script will:")
    print("1. Launch rdcd with -u flag")
    print("2. Launch rdci with specified parameters")
    print("3. Monitor for 'waiting for debugger' messages")
    print("4. Attach GDB and wait for your confirmation")
    print("5. Call rocprofiler_debugger_continue() and detach")
    print("6. Repeat the cycle")
    print("\nPress Ctrl+C to stop at any time\n")

    debugger = RDCDebugger()
    debugger.run()

if __name__ == "__main__":
    main()