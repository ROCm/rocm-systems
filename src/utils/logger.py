##############################################################################
# MIT License
#
# Copyright (c) 2021 - 2025 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

##############################################################################

import logging
import os
import sys
from pathlib import Path

# Define the colors
BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(8)
RESET_SEQ = "\033[0m"
COLOR_SEQ = "\033[%dm"

COLORS = {
    "WARNING": YELLOW,
    "INFO": GREEN,
    "DEBUG": BLUE,
    "CRITICAL": RED,
    "ERROR": RED,
    "TRACE": MAGENTA,
}


def demarcate(function):

    def wrap_function(*args, **kwargs):
        logging.trace("----- [entering function] -> %s()" % (function.__qualname__))
        result = function(*args, **kwargs)
        logging.trace("----- [exiting  function] -> %s()" % function.__qualname__)
        return result

    return wrap_function


def console_error(*argv, exit=True):
    if len(argv) > 1:
        logging.error(f"[{argv[0]}] {argv[1]}")
    else:
        logging.error(f"{argv[0]}")
    if exit:
        sys.exit(1)


def console_log(*argv, indent_level=0):
    indent = ""
    if indent_level >= 1:
        indent = " " * 3 * indent_level + "|-> "  # spaces per indent level

    if len(argv) > 1:
        logging.info(indent + f"[{argv[0]}] {argv[1]}")
    else:
        logging.info(indent + f"{argv[0]}")


def console_debug(*argv):
    if len(argv) > 1:
        logging.debug(f"[{argv[0]}] {argv[1]}")
    else:
        logging.debug(f"{argv[0]}")


def console_warning(*argv):
    if len(argv) > 1:
        logging.warning(f"[{argv[0]}] {argv[1]}")
    else:
        logging.warning(f"{argv[0]}")


def trace_logger(message, *args, **kwargs):
    logging.log(logging.TRACE, message, *args, **kwargs)


# Define the formatter
class ColoredFormatter(logging.Formatter):

    def format(self, record):
        levelname = record.levelname
        if levelname in COLORS:
            levelname_color = (COLOR_SEQ % (30 + COLORS[levelname]) + levelname +
                               RESET_SEQ)
            record.levelname = levelname_color
        return logging.Formatter.format(self, record)


class ColoredFormatterAll(logging.Formatter):

    def format(self, record):
        levelname = record.levelname
        if levelname in COLORS:
            if levelname == "INFO":
                log_fmt = "%(message)s"
            else:
                log_fmt = f"{COLOR_SEQ % (30 + COLORS[levelname])}%(levelname)s: %(message)s{RESET_SEQ}"
            formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


class PlainFormatter(logging.Formatter):

    def format(self, record):
        if record.levelno == logging.ERROR:
            self._style._fmt = "%(levelname)s %(message)s"
        else:
            self._style._fmt = "%(message)s"
        return logging.Formatter.format(self, record)


# Setup console handler - provided as separate function to be called
# prior to argument parsing
def setup_console_handler():
    logging.getLogger().handlers.clear()
    # register a trace level logger
    logging.TRACE = logging.DEBUG - 5
    logging.addLevelName(logging.TRACE, "TRACE")
    setattr(logging, "TRACE", logging.TRACE)
    setattr(logging, "trace", trace_logger)

    color_setting = 1
    if "ROCPROFCOMPUTE_COLOR" in os.environ.keys():
        color_setting = int(os.environ["ROCPROFCOMPUTE_COLOR"])

    if color_setting == 0:
        # non-colored
        formatter = PlainFormatter()
    elif color_setting == 1:
        # colored loglevel and with non-colored message
        formatter = ColoredFormatter("%(levelname)16s %(message)s")
    elif color_setting == 2:
        # non-colored levelname included
        formatter = logging.Formatter("%(levelname)5s %(message)s")
    elif color_setting == 3:
        # no color or levelname for INFO, other log messages entirely in color
        formatter = ColoredFormatterAll()
    else:
        print("Unsupported setting for ROCPROFCOMPUTE_COLOR - set to 0, 1, 2 or 3.")
        sys.exit(1)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.set_name("console")
    logging.getLogger().addHandler(console_handler)


# Setup file handler - enabled in profile mode
def setup_file_handler(loglevel, workload_dir):
    filename = str(Path(workload_dir).joinpath("log.txt"))
    file_handler = logging.FileHandler(filename, "w")
    file_loglevel = min([loglevel, logging.INFO])
    file_handler.setLevel(file_loglevel)
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    logging.getLogger().addHandler(file_handler)


# Setup logger priority - called after argument parsing
def setup_logging_priority(verbosity, quietmode, appmode, guimode):
    # set loglevel based on selected verbosity and quietmode
    levels = [logging.INFO, logging.DEBUG, logging.TRACE]

    if quietmode:
        loglevel = logging.ERROR
    else:
        loglevel = levels[min(verbosity, len(levels) - 1)]  # cap to last level index

    # optional: suppress Werkzeug's messages in analyze GUIs.
    if quietmode and "analyze" in appmode and guimode:
        werkzeug_logger = logging.getLogger("werkzeug")
        werkzeug_logger.setLevel(logging.ERROR)

    # optional: override of default loglevel via env variable which takes precedence
    if "ROCPROFCOMPUTE_LOGLEVEL" in os.environ.keys():
        loglevel = os.environ["ROCPROFCOMPUTE_LOGLEVEL"]
        if loglevel in {"DEBUG", "debug"}:
            loglevel = logging.DEBUG
        elif loglevel in {"TRACE", "trace"}:
            loglevel = logging.TRACE
        elif loglevel in {"INFO", "info"}:
            loglevel = logging.INFO
        elif loglevel in {"ERROR", "error"}:
            loglevel = logging.ERROR
        else:
            print("Ignoring unsupported ROCPROFCOMPUTE_LOGLEVEL setting (%s)" %
                  loglevel)
            sys.exit(1)

    # update console loglevel based on command-line args/env settings
    for handler in logging.getLogger().handlers:
        if handler.get_name() == "console":
            handler.setLevel(loglevel)

    # set global loglevel to min of console/file settings in profile mode
    if appmode == "profile":
        global_loglevel = min([logging.INFO, loglevel])
        logging.getLogger().setLevel(global_loglevel)
    else:
        logging.getLogger().setLevel(loglevel)

    return loglevel
