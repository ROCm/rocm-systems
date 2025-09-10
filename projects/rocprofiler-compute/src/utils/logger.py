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
from typing import Any, Callable, Optional, TypeVar

R = TypeVar("R")

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

# Constants
TRACE_LEVEL = logging.DEBUG - 5

LOG_LEVEL_MAPPING = {
    "DEBUG": logging.DEBUG,
    "debug": logging.DEBUG,
    "TRACE": TRACE_LEVEL,
    "trace": TRACE_LEVEL,
    "INFO": logging.INFO,
    "info": logging.INFO,
    "ERROR": logging.ERROR,
    "error": logging.ERROR,
}


def demarcate(function: Callable[..., R]) -> Callable[..., R]:
    def wrap_function(*args: Any, **kwargs: Any) -> R:
        trace_logger(f"----- [entering function] -> {function.__qualname__}()")
        result = function(*args, **kwargs)
        trace_logger(f"----- [exiting  function] -> {function.__qualname__}()")
        return result

    return wrap_function


def console_error(*argv: Any, exit: bool = True) -> None:
    if len(argv) > 1:
        logging.error(f"[{argv[0]}] {argv[1]}")
    elif len(argv) == 1:
        logging.error(f"{argv[0]}")
    else:
        logging.error("Empty error message")
    if exit:
        sys.exit(1)


def console_log(*argv: Any, indent_level: int = 0) -> None:
    indent = ""
    if indent_level >= 1:
        indent = " " * (3 * indent_level) + "|-> "  # spaces per indent level

    if len(argv) > 1:
        logging.info(indent + f"[{argv[0]}] {argv[1]}")
    elif len(argv) == 1:
        logging.info(indent + f"{argv[0]}")
    else:
        logging.info(indent + "Empty log message")


def console_debug(*argv: Any) -> None:
    if len(argv) > 1:
        logging.debug(f"[{argv[0]}] {argv[1]}")
    elif len(argv) == 1:
        logging.debug(f"{argv[0]}")
    else:
        logging.debug("Empty debug message")


def console_warning(*argv: Any) -> None:
    if len(argv) > 1:
        logging.warning(f"[{argv[0]}] {argv[1]}")
    elif len(argv) == 1:
        logging.warning(f"{argv[0]}")
    else:
        logging.warning("Empty warning message")


def trace_logger(message: str, *args: Any, **kwargs: Any) -> None:
    logging.log(TRACE_LEVEL, message, *args, **kwargs)


# Define the formatter
class ColoredFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        levelname = record.levelname
        if levelname in COLORS:
            levelname_color = (
                COLOR_SEQ % (30 + COLORS[levelname]) + levelname + RESET_SEQ
            )
            record.levelname = levelname_color
        return logging.Formatter.format(self, record)


class ColoredFormatterAll(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        levelname = record.levelname
        if levelname in COLORS:
            if levelname == "INFO":
                log_fmt = "%(message)s"
            else:
                log_fmt = (
                    f"{COLOR_SEQ % (30 + COLORS[levelname])}"
                    f"%(levelname)s: %(message)s{RESET_SEQ}"
                )
            formatter = logging.Formatter(log_fmt)
            return formatter.format(record)
        return super().format(record)


class PlainFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        if record.levelno == logging.ERROR:
            self._style._fmt = "%(levelname)s %(message)s"
        else:
            self._style._fmt = "%(message)s"
        return logging.Formatter.format(self, record)


# Setup console handler - provided as separate function to be called
# prior to argument parsing
def setup_console_handler() -> None:
    logging.getLogger().handlers.clear()
    # register a trace level logger
    logging.addLevelName(TRACE_LEVEL, "TRACE")
    setattr(logging, "TRACE", TRACE_LEVEL)
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
def setup_file_handler(loglevel: int, workload_dir: str) -> None:
    filename = str(Path(workload_dir) / "log.txt")
    file_handler = logging.FileHandler(filename, "w")
    file_loglevel = min([loglevel, logging.INFO])
    file_handler.setLevel(file_loglevel)
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    logging.getLogger().addHandler(file_handler)


# Setup logger priority - called after argument parsing
def setup_logging_priority(
    verbosity: int, quietmode: bool, appmode: str, guimode: Optional[bool] = None
) -> int:
    # set loglevel based on selected verbosity and quietmode
    levels = [logging.INFO, logging.DEBUG, TRACE_LEVEL]

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

        if loglevel in LOG_LEVEL_MAPPING:
            loglevel = LOG_LEVEL_MAPPING[loglevel]
        else:
            print(f"Ignoring unsupported ROCPROFCOMPUTE_LOGLEVEL setting ({loglevel})")
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
