#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
#
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

import logging
import sys
import os

try:
    import argcomplete
except ImportError as e:
    logging.debug(f"Unhandled import error: {e}")
    logging.debug("argcomplete module not found. Autocomplete will not work.")

# from typing import TYPE_CHECKING
# # only used for type checking
# # pyright trips up and cannot find amdsmi scripts without it
# if TYPE_CHECKING:
#     from amdsmi_commands import AMDSMICommands
#     from amdsmi_parser import AMDSMIParser
#     from amdsmi_logger import AMDSMILogger
#     import amdsmi_cli_exceptions
#     from amdsmi import amdsmi_interface
#     from amdsmi import amdsmi_exception

# Set the environment variable for GPU metrics cache duration
cache_ms = os.environ.setdefault("AMDSMI_GPU_METRICS_CACHE_MS", "100")
logging.debug("AMDSMI_GPU_METRICS_CACHE_MS = %sms", cache_ms)

try:
    from amdsmi_init import *
    from amdsmi_helpers import AMDSMIHelpers
    from amdsmi_commands import AMDSMICommands
    from amdsmi_parser import AMDSMIParser
    from amdsmi_logger import AMDSMILogger
    import amdsmi_cli_exceptions
except ImportError:
    current_path = os.path.dirname(os.path.abspath(__file__))
    cli_files_path = f"{current_path}/../libexec/amdsmi_cli"
    sys.path.append(cli_files_path)
    try:
        from amdsmi_init import *
        from amdsmi_helpers import AMDSMIHelpers
        from amdsmi_commands import AMDSMICommands
        from amdsmi_parser import AMDSMIParser
        from amdsmi_logger import AMDSMILogger
        import amdsmi_cli_exceptions
    except ImportError as e:
        print(f"Unhandled import error: {e}")
        print(f"Unable to import amdsmi_cli files. Check {cli_files_path} if they are present.")
        sys.exit(1)

def _print_error(e, destination):
    if destination in ['stdout', 'json', 'csv']:
        print(e)
    else:
        f = open(destination, "w", encoding="utf-8")
        f.write(e)
        f.close()
        print("Error occurred. Result written to " + str(destination) + " file")

def configure_logging_and_execute(args, amd_smi_commands):
    """
    Configures logging based on the provided arguments and executes the subcommand.

    Args:
        args: Parsed command-line arguments.
        amd_smi_commands: Instance of AMDSMICommands.
    """
    # Remove previous log handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # To enable debug logs in AMD SMI library:
    #   set RSMI_LOGGING = 1 for logging to files
    #   set RSMI_LOGGING = 2 for logging to stdout
    #   set RSMI_LOGGING = 3 for logging to stdout and files
    #   set RSMI_LOGGING = 0 to disable logging
    # Files will be located in /var/log/amd_smi_lib/AMD-SMI-lib.log*

    # log string with the following format:
    # loglevel | YYYY-MM-DD HH:MM:SS.ms | filename:line | message
    logging_dict = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }

    time = '%(asctime)s.%(msecs)03d'
    datefmt = '%Y-%m-%d %H:%M:%S'
    logging.basicConfig(format='%(levelname)s | ' + time + ' | %(filename)s:%(lineno)d | %(message)s',
                        level=logging_dict[args.loglevel], datefmt=datefmt)

    # Disable traceback for non-debug log levels
    if args.loglevel == "DEBUG":
        sys.tracebacklimit = 10
    else:
        sys.tracebacklimit = -1

    logging.debug(args)

    # Execute subcommands
    try:
        args.func(args)
    except amdsmi_cli_exceptions.AmdSmiException as e:
        _print_error(str(e), amd_smi_commands.logger.destination)
    except amdsmi_exception.AmdSmiLibraryException as e:
        exc = amdsmi_cli_exceptions.AmdSmiLibraryErrorException(amd_smi_commands.logger.format, e.get_error_code())
        _print_error(str(exc), amd_smi_commands.logger.destination)


if __name__ == "__main__":
    # Disable traceback before possible init errors in AMDSMICommands and AMDSMIParser
    copy_argv = str(sys.argv.copy()).upper()
    if "DEBUG" in copy_argv:
        sys.tracebacklimit = 10
    else:
        sys.tracebacklimit = -1

    amd_smi_helpers = AMDSMIHelpers()
    amd_smi_commands = AMDSMICommands(helpers=amd_smi_helpers)
    amd_smi_parser = AMDSMIParser(amd_smi_commands.version,
                                    amd_smi_commands.list,
                                    amd_smi_commands.static,
                                    amd_smi_commands.firmware,
                                    amd_smi_commands.bad_pages,
                                    amd_smi_commands.metric,
                                    amd_smi_commands.process,
                                    amd_smi_commands.profile,
                                    amd_smi_commands.event,
                                    amd_smi_commands.topology,
                                    amd_smi_commands.set_value,
                                    amd_smi_commands.reset,
                                    amd_smi_commands.monitor,
                                    amd_smi_commands.xgmi,
                                    amd_smi_commands.partition,
                                    amd_smi_commands.ras,
                                    amd_smi_commands.default,
                                    sys_argv=sys.argv,
                                    helpers=amd_smi_helpers)
    try:
        argcomplete.autocomplete(amd_smi_parser)
    except NameError:
        logging.debug("argcomplete module not found. Autocomplete will not work.")

    # Store possible subcommands & aliases for later errors
    valid_commands = amd_smi_parser.possible_commands
    valid_commands += ['--help', '-h']

    # Convert arguments to lowercase, but preserve case for folder path values
    processed_argv = []
    # Arguments that should preserve case
    case_sensitive_args = ['--folder', '--file', '--gpu', '--cpu', '--core', '--profile', '--cper-file']
    case_sensitive_prefixes = ['--folder=', '--file=', '--gpu=', '--cpu=', '--core=', '--profile=', '--cper-file=']

    preserve_case_for_next = False
    for i, arg in enumerate(sys.argv):
        if preserve_case_for_next:
            # Preserve case for the next argument value
            processed_argv.append(arg)
            preserve_case_for_next = False
        elif arg in case_sensitive_args:
            # Convert flag to lowercase but preserve next value
            processed_argv.append(arg.lower())
            preserve_case_for_next = True
        elif any(arg.startswith(prefix) for prefix in case_sensitive_prefixes):
            # Handle --arg=value format, preserve case for the value part
            for prefix in case_sensitive_prefixes:
                if arg.startswith(prefix):
                    flag = prefix.rstrip('=')
                    value = arg[len(prefix):]
                    processed_argv.append(flag.lower() + '=' + value)
                    break
        elif arg.startswith('--') or not arg.startswith('-'):
            # Convert other long options and positional arguments to lowercase
            processed_argv.append(arg.lower())
        else:
            # Preserve case for short options
            processed_argv.append(arg)
    sys.argv = processed_argv
    
    if len(sys.argv) == 1:
        args = amd_smi_parser.parse_args(args=['default'])
    elif sys.tracebacklimit == 10 and (sys.argv[1] == '--loglevel'):
        args = amd_smi_parser.parse_args(args=['default', '--loglevel'] + sys.argv[2:])
    elif sys.argv[1] in valid_commands:
        args = amd_smi_parser.parse_args(args=None)
    else:
        raise amdsmi_cli_exceptions.AmdSmiInvalidSubcommandException(sys.argv[1],amd_smi_commands.logger.destination)

    # Handle command modifiers before subcommand execution
    # human readable is the default output format
    if hasattr(args, 'json') and args.json:
        amd_smi_commands.logger.format = amd_smi_commands.logger.LoggerFormat.json.value
    if hasattr(args, 'csv') and args.csv:
        amd_smi_commands.logger.format = amd_smi_commands.logger.LoggerFormat.csv.value
    if hasattr(args, 'file') and args.file:
        amd_smi_commands.logger.destination = args.file
    configure_logging_and_execute(args, amd_smi_commands)
