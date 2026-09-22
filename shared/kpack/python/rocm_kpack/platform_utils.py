"""
Platform-specific utilities for rocm-kpack.

This module provides helpers for cross-platform compatibility, particularly
for handling differences between Windows and Unix environments.
"""

import sys


def configure_windows_console() -> None:
    """Configure Windows console for UTF-8 output.

    Windows consoles often use legacy codepages that can't display Unicode
    characters like checkmarks (✓) or crosses (✗). This function reconfigures
    stdout/stderr to use UTF-8 encoding with replacement for unsupported chars.

    Safe to call on any platform - does nothing on non-Windows systems.

    Usage:
        # At the very start of a CLI script, before any output:
        from rocm_kpack.platform_utils import configure_windows_console
        configure_windows_console()

        if __name__ == "__main__":
            main()
    """
    if sys.platform != "win32":
        return

    # Python 3.7+ has reconfigure() method on text streams
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass  # If it fails, continue with default encoding

    if hasattr(sys.stderr, "reconfigure"):
        try:
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
