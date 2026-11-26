#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""ESMI entry points report NOT_SUPPORTED when libamd_smi.so is built without ESMI.

amd_smi.cc builds them only under ``#ifdef ENABLE_ESMI_LIB``, which CMake sets on
x86_64 alone. Elsewhere the generated wrapper leaves them undefined, and the Python
functions must raise AmdSmiLibraryException(NOT_SUPPORTED), not AttributeError.
"""

import enum
import inspect
import re
import types
import unittest
from pathlib import Path
from unittest import mock

from common.common import amdsmi

_ESMI_SOURCES = (Path("include") / "amd_smi" / "amdsmi.h", Path("src") / "amd_smi" / "amd_smi.cc")
_ESMI_GUARD = "#ifdef ENABLE_ESMI_LIB"
_COMMENT = re.compile(r"/\*.*?\*/|//[^\n]*", re.DOTALL)
_TOP_LEVEL_FUNCTION = re.compile(r"^(?!static\b)\w[\w \t*]*?\b(amdsmi_\w+)\s*\(", re.MULTILINE)


def esmi_only_functions(text: str) -> set:
    """Return the functions declared or defined only under ``#ifdef ENABLE_ESMI_LIB``."""
    in_esmi = []
    kept = []
    for line in _COMMENT.sub("", text).splitlines():
        directive = line.strip()
        if directive.startswith("#if"):
            in_esmi.append(directive == _ESMI_GUARD or (bool(in_esmi) and in_esmi[-1]))
        elif directive.startswith(("#else", "#elif")) and in_esmi:
            in_esmi[-1] = len(in_esmi) > 1 and in_esmi[-2]
        elif directive.startswith("#endif") and in_esmi:
            in_esmi.pop()
        elif in_esmi and in_esmi[-1]:
            kept.append(line)
    return set(_TOP_LEVEL_FUNCTION.findall("\n".join(kept)))


def find_esmi_sources() -> list:
    """Locate amdsmi.h and, in the source tree, amd_smi.cc by walking up from this file.

    amd_smi.cc defines some ESMI-only functions that amdsmi.h declares unconditionally;
    it is not installed, so from ``share/amd_smi/tests`` only the header is checked.
    """
    for parent in Path(__file__).resolve().parents:
        found = [parent / rel for rel in _ESMI_SOURCES if (parent / rel).is_file()]
        if found:
            return found
    raise unittest.SkipTest("amdsmi.h not found in the source tree or the install prefix")


def wrapper_without(hidden: set) -> types.ModuleType:
    """amdsmi_wrapper as a build without ESMI sees it: the hidden names are unbound."""
    real = amdsmi.amdsmi_wrapper
    proxy = types.ModuleType(real.__name__)
    proxy.__dict__.update({k: v for k, v in vars(real).items() if k not in hidden})
    return proxy


def placeholder_args(func) -> list:
    """Arguments that pass the function's own type checks before it reaches the library."""
    args = []
    for param in inspect.signature(func).parameters.values():
        if param.default is not inspect.Parameter.empty:
            break
        annotation = param.annotation
        if "handle" in param.name:
            args.append(amdsmi.amdsmi_wrapper.amdsmi_processor_handle(None))
        elif annotation is str:
            args.append("")
        elif isinstance(annotation, type) and issubclass(annotation, enum.Enum):
            args.append(next(iter(annotation)))
        else:
            args.append(0)
    return args


class EsmiOnlyFunctionsTest(unittest.TestCase):
    def test_collects_top_level_names_inside_guard_only(self):
        text = (
            "amdsmi_status_t amdsmi_get_gpu_x(int a);\n"
            "#ifdef ENABLE_ESMI_LIB\n"
            "/* see amdsmi_get_gpu_y() */\n"
            "static amdsmi_status_t amdsmi_helper(int a) { return a; }\n"
            "amdsmi_status_t amdsmi_get_cpu_x(int a) {\n"
            "  return amdsmi_get_gpu_x(a);\n"
            "}\n"
            "#ifdef OTHER\n"
            "amdsmi_status_t amdsmi_get_cpu_y(int a);\n"
            "#endif\n"
            "amdsmi_status_t amdsmi_get_cpu_z(int a);\n"
            "#else\n"
            "amdsmi_status_t amdsmi_get_cpu_stub(int a);\n"
            "#endif\n"
            "amdsmi_status_t amdsmi_get_gpu_z(int a);\n"
        )
        self.assertEqual(
            esmi_only_functions(text), {"amdsmi_get_cpu_x", "amdsmi_get_cpu_y", "amdsmi_get_cpu_z"}
        )


class EsmiEntryPointsWithoutEsmiTest(unittest.TestCase):
    def test_esmi_functions_raise_not_supported(self):
        sources = find_esmi_sources()
        esmi = set()
        for source in sources:
            esmi |= esmi_only_functions(source.read_text(encoding="utf-8", errors="replace"))
        iface = amdsmi.amdsmi_interface
        functions = {name: getattr(iface, name) for name in sorted(esmi) if hasattr(iface, name)}
        self.assertTrue(functions, msg=f"no ESMI-only functions found in {sources}")

        not_supported = amdsmi.amdsmi_wrapper.AMDSMI_STATUS_NOT_SUPPORTED
        with mock.patch.object(iface, "amdsmi_wrapper", wrapper_without(esmi)):
            for name, func in functions.items():
                with self.subTest(name):
                    with self.assertRaises(amdsmi.AmdSmiLibraryException) as ctx:
                        func(*placeholder_args(func))
                    self.assertEqual(ctx.exception.get_error_code(), not_supported)
