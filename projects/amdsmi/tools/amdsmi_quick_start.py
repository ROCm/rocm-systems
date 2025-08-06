# Copyright (C) Advanced Micro Devices. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# This script is used to quickly debug and check amdsmi Python APIs.
# This is not meant to serve best practices for development.
# Run this post install with python3 -i quick_start.py

import atexit
import logging
import signal
import sys

# Metrics cache set to 1 by default, uncomment to change
# os.environ["AMDSMI_GPU_METRICS_CACHE_MS"] = "1"

try:
    from amdsmi import *
except ImportError as e:
    print(f"Failed to import 'amdsmi': {e}")
    print("Ensure that the 'amdsmi' library is installed and accessible.")
    raise

from pathlib import Path

sys.path.append(str(Path('/opt/rocm/libexec/')))
sys.path.append(str(Path('/opt/rocm/libexec/amdsmi_cli/')))

try:
    from amdsmi_commands import AMDSMICommands
    from amdsmi_helpers import AMDSMIHelpers
    from amdsmi_logger import AMDSMILogger
    from amdsmi_parser import AMDSMIParser
    import amdsmi_cli_exceptions
    helpers = AMDSMIHelpers()
except ImportError as e:
    print(f"Failed to import amdsmi cli libs: {e}")
    print("Ensure that you have installed amdsmi's package.")


# Make exit & quit work without parens because it's annoying
type(exit).__repr__ = sys.exit

def signal_handler(sig, frame):
    logging.debug(f"Handling signal: {sig}")
    sys.exit(0)

amdsmi_init()
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
atexit.register(amdsmi_shut_down)

gpus = amdsmi_get_processor_handles()
cpus = amdsmi_get_cpusocket_handles()

print(f"gpus variable populated with:{gpus}")
print(f"cpus variable populated with:{cpus}")
