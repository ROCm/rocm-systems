#!/usr/bin/env python3

# MIT License
#
# Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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


import argparse
import os
import sys


class FormatSource(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        os.system(
            "clang-format-11 -i $(find "
            + os.path.dirname(__file__)
            + "/../../samples "
            + os.path.dirname(__file__)
            + "/../../source "
            + os.path.dirname(__file__)
            + '/../../tests -type f -not -path "'
            + os.path.dirname(__file__)
            + "/../../build/*\" | egrep '\.(h|hpp|hh|c|cc|cpp)(|\.in)$')"
        )
        exit(0)


class FormatCMake(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        os.system(
            "cmake-format -i $(find "
            + os.path.dirname(__file__)
            + '/../.. -type f -not -path "'
            + os.path.dirname(__file__)
            + '/../../build/*" -not -path "'
            + os.path.dirname(__file__)
            + "/../../external/*\" | egrep 'CMakeLists.txt|\.cmake$')"
        )
        exit(0)


class FormatPython(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        os.system("black " + os.path.dirname(__file__) + "/../..")
        exit(0)


class FormatAll(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        os.system(
            "clang-format-11 -i $(find "
            + os.path.dirname(__file__)
            + "/../../samples "
            + os.path.dirname(__file__)
            + "/../../source "
            + os.path.dirname(__file__)
            + '/../../tests -type f -not -path "'
            + os.path.dirname(__file__)
            + "/../../build/*\" | egrep '\.(h|hpp|hh|c|cc|cpp)(|\.in)$')"
        )
        os.system(
            "cmake-format -i $(find "
            + os.path.dirname(__file__)
            + '/../.. -type f -not -path "'
            + os.path.dirname(__file__)
            + '/../../build/*" -not -path "'
            + os.path.dirname(__file__)
            + "/../../external/*\" | egrep 'CMakeLists.txt|\.cmake$')"
        )
        os.system("black " + os.path.dirname(__file__) + "/../..")
        exit(0)


class InstallDepsUbuntu(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        os.system(
            "sudo apt-get update; \
            sudo apt-get install -y python3-pip software-properties-common wget curl clang-format-11; \
            python3 -m pip install -U cmake-format; \
            python -m pip install --upgrade pip; \
            python -m pip install black"
        )
        exit(0)


parser = argparse.ArgumentParser(description="ROCProfiler Formatter")
parser.add_argument(
    "-ud",
    "--ubuntu-deps",
    nargs=0,
    help="Install Formatting dependencies",
    action=InstallDepsUbuntu,
)
parser.add_argument(
    "-s", "--source", nargs=0, help="format source files", action=FormatSource
)
parser.add_argument(
    "-c", "--cmake", nargs=0, help="format cmake files", action=FormatCMake
)
parser.add_argument(
    "-p", "--python", nargs=0, help="format python files", action=FormatPython
)
parser.add_argument(
    "-a", "--all", nargs=0, help="format cmake, source and python files", action=FormatAll
)
parser.parse_args()
