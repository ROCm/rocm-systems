#!/usr/bin/env python3
###############################################################################
# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc.
###############################################################################

"""
Tests for CLI LLM provider choices and output_format deprecation.
"""

import warnings

import pytest


class TestCliLlmProviders:
    def test_llm_choices_include_private(self):
        """--llm choices should include 'private'."""
        from rocinsight.analyze import add_args
        import argparse

        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        sub = subparsers.add_parser("analyze")
        add_args(sub)

        # Parse with --llm private to verify it is accepted
        args = parser.parse_args(["analyze", "--llm", "private"])
        assert args.llm == "private"

    def test_llm_choices_include_local(self):
        """--llm choices should include 'local'."""
        from rocinsight.analyze import add_args
        import argparse

        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        sub = subparsers.add_parser("analyze")
        add_args(sub)

        args = parser.parse_args(["analyze", "--llm", "local"])
        assert args.llm == "local"

    def test_llm_choices_still_include_anthropic(self):
        """--llm choices should still include the original providers."""
        from rocinsight.analyze import add_args
        import argparse

        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        sub = subparsers.add_parser("analyze")
        add_args(sub)

        for provider in ["anthropic", "openai", "claude-code"]:
            args = parser.parse_args(["analyze", "--llm", provider])
            assert args.llm == provider

    def test_llm_invalid_choice_rejected(self):
        """--llm with an invalid choice should fail."""
        from rocinsight.analyze import add_args
        import argparse

        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        sub = subparsers.add_parser("analyze")
        add_args(sub)

        with pytest.raises(SystemExit):
            parser.parse_args(["analyze", "--llm", "invalid-provider"])


class TestOutputFormatDeprecation:
    def test_output_format_none_no_warning(self):
        """Passing output_format=None (default) should not trigger a warning."""
        # We can't easily call analyze_database without a real DB,
        # but we can test the import and type signature
        from rocinsight.ai_analysis.api import analyze_database
        import inspect

        sig = inspect.signature(analyze_database)
        param = sig.parameters["output_format"]
        # Default should be None, not OutputFormat.PYTHON_OBJECT
        assert param.default is None, (
            f"output_format default should be None, got {param.default}"
        )

    def test_output_format_type_annotation_is_any(self):
        """output_format type annotation should be Any (not OutputFormat)."""
        from rocinsight.ai_analysis.api import analyze_database
        import inspect
        from typing import Any

        sig = inspect.signature(analyze_database)
        param = sig.parameters["output_format"]
        assert param.annotation is Any, (
            f"output_format annotation should be Any, got {param.annotation}"
        )

    def test_output_format_enum_still_importable(self):
        """OutputFormat enum should still be importable for backward compat."""
        from rocinsight.ai_analysis.api import OutputFormat

        assert hasattr(OutputFormat, "PYTHON_OBJECT")
        assert hasattr(OutputFormat, "JSON")
