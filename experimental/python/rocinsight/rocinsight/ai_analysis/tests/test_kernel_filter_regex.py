#!/usr/bin/env python3
###############################################################################
# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc.
###############################################################################

"""
Tests that --kernel-names has been replaced with --kernel-include-regex
across the codebase, and that kernel name values are regex-escaped.
"""

import re
from pathlib import Path

import pytest


def _package_root() -> Path:
    """Return the rocinsight package root directory."""
    here = Path(__file__).resolve()
    # tests/ -> ai_analysis/ -> rocinsight/
    return here.parent.parent.parent


class TestKernelFilterRegex:
    def test_recommendations_uses_regex_flag(self):
        """recommendations.py should use --kernel-include-regex, not --kernel-names."""
        from rocinsight.analysis.recommendations import generate_recommendations

        # Generate a kernel hotspot rec with a dominant kernel
        time_breakdown = {
            "total_kernel_time": 900_000,
            "total_memcpy_time": 0,
            "total_runtime": 1_000_000,
            "kernel_percent": 90.0,
            "memcpy_percent": 0.0,
            "overhead_percent": 10.0,
        }
        hotspots = [
            {
                "name": "my_kernel(float*, int)",
                "calls": 10,
                "total_duration": 900_000,
                "avg_duration": 90_000,
                "percent_of_total": 90.0,
            }
        ]
        recs = generate_recommendations(time_breakdown, hotspots, {})
        # Find the kernel hotspot / compute rec (category varies by counter availability)
        _rule3_cats = {"Kernel Hotspot", "Compute-Bound Kernel", "Mixed Bottleneck Kernel", "Memory-Bound Kernel"}
        kernel_recs = [r for r in recs if r.get("category") in _rule3_cats]
        assert len(kernel_recs) > 0, f"Expected a kernel hotspot recommendation, got categories: {[r.get('category') for r in recs]}"
        rec = kernel_recs[0]
        # Check commands use --kernel-include-regex
        for cmd in rec.get("commands", []):
            full = cmd.get("full_command", "")
            assert "--kernel-names" not in full, (
                f"full_command still uses --kernel-names: {full}"
            )
            if "--kernel-include-regex" in full:
                # The kernel name should be regex-escaped in the command
                assert "float\\*" in full or "float" in full, (
                    "Kernel name with special chars should be regex-escaped"
                )
            for arg in cmd.get("args", []):
                assert arg.get("name") != "--kernel-names", (
                    "arg name still uses --kernel-names"
                )

    def test_kernel_name_regex_escaped(self):
        """Kernel names with regex special chars should be escaped."""
        from rocinsight.analysis.recommendations import generate_recommendations

        time_breakdown = {
            "total_kernel_time": 900_000,
            "total_memcpy_time": 0,
            "total_runtime": 1_000_000,
            "kernel_percent": 90.0,
            "memcpy_percent": 0.0,
            "overhead_percent": 10.0,
        }
        hotspots = [
            {
                "name": "kernel(float*, int)",
                "calls": 10,
                "total_duration": 900_000,
                "avg_duration": 90_000,
                "percent_of_total": 90.0,
            }
        ]
        recs = generate_recommendations(time_breakdown, hotspots, {})
        _rule3_cats = {"Kernel Hotspot", "Compute-Bound Kernel", "Mixed Bottleneck Kernel", "Memory-Bound Kernel"}
        kernel_recs = [r for r in recs if r.get("category") in _rule3_cats]
        assert len(kernel_recs) > 0, "Expected a kernel hotspot recommendation"
        for cmd in kernel_recs[0].get("commands", []):
            for arg in cmd.get("args", []):
                if arg.get("name") == "--kernel-include-regex":
                    val = arg["value"]
                    # re.escape("kernel(float*, int)") should escape parens, asterisk
                    assert "(" not in val or "\\(" in val, (
                        f"Kernel name not regex-escaped: {val}"
                    )

    def test_no_kernel_names_in_llm_guide(self):
        """LLM reference guide should not contain --kernel-names."""
        guide_path = _package_root() / "ai_analysis" / "share" / "llm-reference-guide.md"
        if guide_path.exists():
            content = guide_path.read_text()
            assert "--kernel-names" not in content, (
                "llm-reference-guide.md still contains --kernel-names"
            )

    def test_no_kernel_names_in_docs_guide(self):
        """LLM_REFERENCE_GUIDE.md copy should not contain --kernel-names."""
        docs_path = _package_root() / "ai_analysis" / "docs" / "LLM_REFERENCE_GUIDE.md"
        if docs_path.exists():
            content = docs_path.read_text()
            assert "--kernel-names" not in content, (
                "docs/LLM_REFERENCE_GUIDE.md still contains --kernel-names"
            )

    def test_filter_rec_commands_uses_regex_flag(self):
        """_NON_DATA_ARGS in _filter_rec_commands should include --kernel-include-regex."""
        from rocinsight.analysis.recommendations import _filter_rec_commands

        cmd = {
            "tool": "rocprofv3",
            "description": "test",
            "flags": ["--sys-trace"],
            "args": [
                {"name": "--pmc", "value": "GRBM_COUNT GRBM_GUI_ACTIVE SQ_WAVES"},
                {"name": "--kernel-include-regex", "value": "my_kernel"},
                {"name": "-d", "value": "./output"},
                {"name": "-o", "value": "profile"},
            ],
            "full_command": (
                'rocprofv3 --sys-trace --pmc GRBM_COUNT GRBM_GUI_ACTIVE SQ_WAVES'
                ' --kernel-include-regex "my_kernel" -d ./output -o profile -- ./app'
            ),
        }
        already = frozenset(
            {"--sys-trace", "pmc:GRBM_COUNT", "pmc:GRBM_GUI_ACTIVE", "pmc:SQ_WAVES"}
        )
        result = _filter_rec_commands([cmd], already)
        assert result == [], (
            "--kernel-include-regex should be treated as non-data arg"
        )
