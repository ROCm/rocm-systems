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

from abc import ABC, abstractmethod
from pathlib import PurePosixPath


class PatternMatcherInterface(ABC):
    """Strategy interface for matching a pattern against a target string."""

    @abstractmethod
    def matches(self, pattern: str, target: str) -> bool:
        """Return True if *pattern* matches *target*."""


class PurePosixGlobHierarchyMatcher(PatternMatcherInterface):
    """
    Match slash-delimited hierarchy strings using PurePosixPath glob semantics.

    Delegates entirely to PurePosixPath.match() for glob evaluation.
    Normalizations applied before matching:
      "all"        ->  "**"      (match everything)
      leading "/"  ->  stripped  (cosmetic, PurePosixPath treats "/" as anchor)
    """

    @staticmethod
    def normalize_pattern(raw_pattern: str) -> str:
        pattern = raw_pattern.strip()
        if not pattern:
            return ""
        if pattern in ("all", "*"):
            return "**"
        if pattern.startswith("/"):
            pattern = pattern[1:]
        return pattern

    def matches(self, pattern: str, target: str) -> bool:
        if not pattern or not target:
            return False

        glob_pattern = self.normalize_pattern(pattern)
        if not glob_pattern:
            return False

        return PurePosixPath(target).match(glob_pattern)


class PatternMatcherEngine:
    """
    Facade that selects a matching strategy by mode name.

    Supported modes:
      "glob-hierarchy"  -  PurePosixPath glob matching (default)

    Future modes (e.g. "regex", "wildcard-dsl") can be added here without
    changing any call-site code.
    """

    def __init__(self, mode: str = "glob-hierarchy") -> None:
        if mode == "glob-hierarchy":
            self.impl: PatternMatcherInterface = PurePosixGlobHierarchyMatcher()
        else:
            raise ValueError(f"Unsupported pattern matcher mode: {mode}")

    def matches(self, pattern: str, target: str) -> bool:
        return self.impl.matches(pattern, target)
