"""A small test runner, in the shape the C and shell tests already use.

Each test is a function taking the Suite. It reports with check(), and
an exception in it fails that test rather than the run, so the rest
still report. The process exits non-zero if anything failed.

Copyright (C) Advanced Micro Devices, Inc.

SPDX-License-Identifier: MIT
"""

import sys
import time
import traceback

GREEN = "\033[0;32m"
RED = "\033[0;31m"
NC = "\033[0m"


class Suite:
    """Collects results and prints them as the tests run."""

    def __init__(self, title):
        self.title = title
        self.failures = 0
        self.checks = 0
        self.current = None
        print("=" * 60)
        print(f"  {title}")
        print("=" * 60)

    def check(self, condition, fmt, *args):
        """Record one assertion. Returns whether it held."""
        self.checks += 1
        message = fmt % args if args else fmt
        if condition:
            print(f"  {GREEN}✓{NC} {message}")
            return True
        self.failures += 1
        print(f"  {RED}✗{NC} {message}")
        return False

    def check_eq(self, got, want, fmt, *args):
        """Record one equality, printing both values when it fails."""
        message = fmt % args if args else fmt
        if got == want:
            return self.check(True, message)
        return self.check(False, "%s: got %r, expected %r", message, got, want)

    def run(self, name, func, *args, **kwargs):
        """Run one test function, catching anything it raises."""
        print(f"\n--- {name}")
        self.current = name
        started = time.monotonic()
        try:
            func(self, *args, **kwargs)
        except Exception:  # pylint: disable=broad-except
            self.failures += 1
            print(f"  {RED}✗{NC} {name} raised:")
            traceback.print_exc()
        print(f"  ({time.monotonic() - started:.1f}s)")

    def finish(self):
        """Print the summary and exit with the right status."""
        print(f"\n{'=' * 60}")
        if self.failures:
            print(f"  {RED}{self.failures} of {self.checks} checks failed{NC}")
        else:
            print(f"  {GREEN}all {self.checks} checks passed{NC}")
        print("=" * 60)
        sys.exit(1 if self.failures else 0)
