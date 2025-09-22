#!/usr/bin/env python3

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
