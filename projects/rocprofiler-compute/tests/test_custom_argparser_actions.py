##############################################################################
# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
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

import argparse
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


class TestExperimentalActionBaseActions:
    """Tests for ExperimentalAction with different base_action types and edge cases."""

    def test_base_action_store(self):
        """Test ExperimentalAction with base_action='store'."""
        from argparser import ExperimentalAction

        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--test-store",
            action=ExperimentalAction,
            experimental_enabled=True,
            feature_label="Test Store Feature",
            base_action="store",
            type=str,
            help="Test store action",
        )

        args = parser.parse_args(["--test-store", "myvalue"])
        assert args.test_store == "myvalue"

    def test_base_action_store_with_type_conversion(self):
        """Test ExperimentalAction with base_action='store' and type conversion."""
        from argparser import ExperimentalAction

        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--test-int",
            action=ExperimentalAction,
            experimental_enabled=True,
            feature_label="Test Int Feature",
            base_action="store",
            type=int,
            help="Test store with int type",
        )

        args = parser.parse_args(["--test-int", "42"])
        assert args.test_int == 42
        assert isinstance(args.test_int, int)

    def test_base_action_store_multiple_values(self):
        """Test ExperimentalAction with base_action='store' and nargs='*'."""
        from argparser import ExperimentalAction

        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--test-multi",
            action=ExperimentalAction,
            experimental_enabled=True,
            feature_label="Test Multi Feature",
            base_action="store",
            type=int,
            nargs="*",
            help="Test store with multiple values",
        )

        args = parser.parse_args(["--test-multi", "1", "2", "3"])
        assert args.test_multi == [1, 2, 3]

    def test_base_action_store_const(self):
        """Test ExperimentalAction with base_action='store_const'."""
        from argparser import ExperimentalAction

        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--test-const",
            action=ExperimentalAction,
            experimental_enabled=True,
            feature_label="Test Const Feature",
            base_action="store_const",
            const="CONSTANT_VALUE",
            default="DEFAULT_VALUE",
            help="Test store_const action",
        )

        # Without flag - should use default
        args_default = parser.parse_args([])
        assert args_default.test_const == "DEFAULT_VALUE"

        # With flag - should use const
        args_const = parser.parse_args(["--test-const"])
        assert args_const.test_const == "CONSTANT_VALUE"

    def test_base_action_store_true(self):
        """Test ExperimentalAction with base_action='store_true'."""
        from argparser import ExperimentalAction

        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--test-true",
            action=ExperimentalAction,
            experimental_enabled=True,
            feature_label="Test True Feature",
            base_action="store_true",
            default=False,
            help="Test store_true action",
        )

        # Without flag - should be False
        args_false = parser.parse_args([])
        assert args_false.test_true is False

        # With flag - should be True
        args_true = parser.parse_args(["--test-true"])
        assert args_true.test_true is True

    def test_base_action_store_false(self):
        """Test ExperimentalAction with base_action='store_false'."""
        from argparser import ExperimentalAction

        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--test-false",
            action=ExperimentalAction,
            experimental_enabled=True,
            feature_label="Test False Feature",
            base_action="store_false",
            default=True,
            help="Test store_false action",
        )

        # Without flag - should be True (default)
        args_true = parser.parse_args([])
        assert args_true.test_false is True

        # With flag - should be False
        args_false = parser.parse_args(["--test-false"])
        assert args_false.test_false is False

    def test_base_action_append(self):
        """Test ExperimentalAction with base_action='append'."""
        from argparser import ExperimentalAction

        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--test-append",
            action=ExperimentalAction,
            experimental_enabled=True,
            feature_label="Test Append Feature",
            base_action="append",
            type=str,
            help="Test append action",
        )

        args = parser.parse_args([
            "--test-append",
            "first",
            "--test-append",
            "second",
            "--test-append",
            "third",
        ])
        assert args.test_append == ["first", "second", "third"]

    def test_base_action_append_const(self):
        """Test ExperimentalAction with base_action='append_const'."""
        from argparser import ExperimentalAction

        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--test-append-const",
            action=ExperimentalAction,
            experimental_enabled=True,
            feature_label="Test Append Const Feature",
            base_action="append_const",
            const="ITEM",
            default=[],
            help="Test append_const action",
        )

        # Without flag - empty list
        args_empty = parser.parse_args([])
        assert args_empty.test_append_const == []

        # With flag multiple times
        args_multiple = parser.parse_args([
            "--test-append-const",
            "--test-append-const",
            "--test-append-const",
        ])
        assert args_multiple.test_append_const == ["ITEM", "ITEM", "ITEM"]

    def test_base_action_count(self):
        """Test ExperimentalAction with base_action='count'."""
        from argparser import ExperimentalAction

        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--test-count",
            action=ExperimentalAction,
            experimental_enabled=True,
            feature_label="Test Count Feature",
            base_action="count",
            default=0,
            help="Test count action",
        )

        # No flags - count is 0
        args_zero = parser.parse_args([])
        assert args_zero.test_count == 0

        # Multiple flags - count increases
        args_three = parser.parse_args([
            "--test-count",
            "--test-count",
            "--test-count",
        ])
        assert args_three.test_count == 3

    def test_base_action_extend(self):
        """Test ExperimentalAction with base_action='extend'."""
        from argparser import ExperimentalAction

        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--test-extend",
            action=ExperimentalAction,
            experimental_enabled=True,
            feature_label="Test Extend Feature",
            base_action="extend",
            nargs="+",
            type=str,
            help="Test extend action",
        )

        args = parser.parse_args([
            "--test-extend",
            "a",
            "b",
            "--test-extend",
            "c",
            "d",
        ])
        assert args.test_extend == ["a", "b", "c", "d"]

    def test_base_action_required_error(self):
        """Test that missing base_action raises ValueError."""
        from argparser import ExperimentalAction

        parser = argparse.ArgumentParser()
        with pytest.raises(ValueError, match="base_action is required"):
            parser.add_argument(
                "--test-missing",
                action=ExperimentalAction,
                experimental_enabled=True,
                feature_label="Test Missing Base Action",
                # base_action is intentionally missing
                help="Test missing base_action",
            )

    def test_invalid_base_action_error(self):
        """Test that invalid base_action raises ValueError."""
        from argparser import ExperimentalAction

        parser = argparse.ArgumentParser()
        with pytest.raises(ValueError, match="Unsupported base_action"):
            parser.add_argument(
                "--test-invalid",
                action=ExperimentalAction,
                experimental_enabled=True,
                feature_label="Test Invalid Base Action",
                base_action="invalid_action",
                help="Test invalid base_action",
            )

    def test_experimental_disabled_blocks_usage(self):
        """Test that experimental feature cannot be used
        when experimental_enabled=False."""
        from argparser import ExperimentalAction

        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--test-disabled",
            action=ExperimentalAction,
            experimental_enabled=False,
            feature_label="Test Disabled Feature",
            base_action="store_true",
            help="Test disabled experimental feature",
        )

        # Using the feature should raise an error
        with pytest.raises(SystemExit):
            parser.parse_args(["--test-disabled"])

    def test_experimental_enabled_allows_usage(self):
        """Test that experimental feature can be used when experimental_enabled=True."""
        from argparser import ExperimentalAction

        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--test-enabled",
            action=ExperimentalAction,
            experimental_enabled=True,
            feature_label="Test Enabled Feature",
            base_action="store_true",
            help="Test enabled experimental feature",
        )

        # Using the feature should work
        args = parser.parse_args(["--test-enabled"])
        assert args.test_enabled is True

    def test_help_text_with_leading_whitespace(self):
        """Test that EXPERIMENTAL prefix is added to help text."""
        from argparser import ExperimentalAction

        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--test-whitespace",
            action=ExperimentalAction,
            experimental_enabled=True,
            feature_label="Test Whitespace",
            base_action="store_true",
            help="\t\t\tThis has leading tabs",
        )

        help_text = parser.format_help()
        # EXPERIMENTAL prefix should be added (argparse may reformat whitespace)
        assert "EXPERIMENTAL: This has leading tabs" in help_text

    def test_help_text_suppressed_when_disabled(self):
        """Test that help text is suppressed when experimental_enabled=False."""
        from argparser import ExperimentalAction

        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--test-suppressed",
            action=ExperimentalAction,
            experimental_enabled=False,
            feature_label="Test Suppressed Feature",
            base_action="store_true",
            help="This help should be suppressed",
        )

        help_text = parser.format_help()
        assert "--test-suppressed" not in help_text
        assert "This help should be suppressed" not in help_text

    def test_store_action_with_default(self):
        """Test that default values work correctly with base_action='store'."""
        from argparser import ExperimentalAction

        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--test-default",
            action=ExperimentalAction,
            experimental_enabled=True,
            feature_label="Test Default Feature",
            base_action="store",
            type=str,
            default="default_value",
            help="Test with default",
        )

        # Without flag - should use default
        args = parser.parse_args([])
        assert args.test_default == "default_value"

    def test_store_action_with_choices(self):
        """Test that choices validation works with base_action='store'."""
        from argparser import ExperimentalAction

        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--test-choices",
            action=ExperimentalAction,
            experimental_enabled=True,
            feature_label="Test Choices Feature",
            base_action="store",
            type=str,
            choices=["a", "b", "c"],
            help="Test with choices",
        )

        # Valid choice
        args_valid = parser.parse_args(["--test-choices", "a"])
        assert args_valid.test_choices == "a"

        # Invalid choice should raise error
        with pytest.raises(SystemExit):
            parser.parse_args(["--test-choices", "invalid"])

    def test_nargs_auto_set_for_store_const(self):
        """Test that nargs=0 is automatically set for store_const."""
        from argparser import ExperimentalAction

        parser = argparse.ArgumentParser()
        # Don't specify nargs - it should be auto-set to 0
        parser.add_argument(
            "--test-auto-nargs",
            action=ExperimentalAction,
            experimental_enabled=True,
            feature_label="Test Auto Nargs",
            base_action="store_const",
            const=True,
            help="Test auto nargs setting",
        )

        # Should work without consuming arguments
        args = parser.parse_args(["--test-auto-nargs"])
        assert args.test_auto_nargs is True

    def test_const_auto_set_for_store_true(self):
        """Test that const=True is automatically set for store_true."""
        from argparser import ExperimentalAction

        parser = argparse.ArgumentParser()
        # Don't specify const - it should be auto-set to True
        parser.add_argument(
            "--test-auto-const",
            action=ExperimentalAction,
            experimental_enabled=True,
            feature_label="Test Auto Const",
            base_action="store_true",
            help="Test auto const setting",
        )

        args = parser.parse_args(["--test-auto-const"])
        assert args.test_auto_const is True

    def test_warning_message_displayed(self):
        """Test that experimental warning is shown when feature is used."""
        from argparser import ExperimentalAction

        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--test-warning",
            action=ExperimentalAction,
            experimental_enabled=True,
            feature_label="Test Warning Feature",
            base_action="store_true",
            help="Test warning message",
        )

        # Mock console_warning to verify it's called
        with patch("argparser.console_warning") as mock_warning:
            args = parser.parse_args(["--test-warning"])  # noqa F841
            mock_warning.assert_called_once()
            assert "Test Warning Feature" in str(mock_warning.call_args)
            assert "experimental and may change" in str(mock_warning.call_args)

    def test_error_message_when_disabled(self):
        """Test error message when experimental feature is used
        without --experimental."""
        from argparser import ExperimentalAction

        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--test-error",
            action=ExperimentalAction,
            experimental_enabled=False,
            feature_label="Test Error Feature",
            base_action="store_true",
            help="Test error message",
        )

        # Should exit with error message
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["--test-error"])
        assert exc_info.value.code == 2

    def test_multiple_experimental_features_same_parser(self):
        """Test multiple experimental features in the same parser."""
        from argparser import ExperimentalAction

        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--feature-one",
            action=ExperimentalAction,
            experimental_enabled=True,
            feature_label="Feature One",
            base_action="store_true",
            help="First feature",
        )
        parser.add_argument(
            "--feature-two",
            action=ExperimentalAction,
            experimental_enabled=True,
            feature_label="Feature Two",
            base_action="store",
            type=str,
            help="Second feature",
        )

        args = parser.parse_args(["--feature-one", "--feature-two", "value"])
        assert args.feature_one is True
        assert args.feature_two == "value"

    def test_mixed_enabled_disabled_features(self):
        """Test parser with mix of enabled and disabled experimental features."""
        from argparser import ExperimentalAction

        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--enabled-feature",
            action=ExperimentalAction,
            experimental_enabled=True,
            feature_label="Enabled Feature",
            base_action="store_true",
            help="Enabled feature",
        )
        parser.add_argument(
            "--disabled-feature",
            action=ExperimentalAction,
            experimental_enabled=False,
            feature_label="Disabled Feature",
            base_action="store_true",
            help="Disabled feature",
        )

        # Enabled feature should work
        args = parser.parse_args(["--enabled-feature"])
        assert args.enabled_feature is True

        # Disabled feature should fail
        with pytest.raises(SystemExit):
            parser.parse_args(["--disabled-feature"])

    def test_nargs_plus_with_store(self):
        """Test base_action='store' with nargs='+'."""
        from argparser import ExperimentalAction

        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--test-plus",
            action=ExperimentalAction,
            experimental_enabled=True,
            feature_label="Test Plus Feature",
            base_action="store",
            nargs="+",
            type=int,
            help="Test nargs plus",
        )

        args = parser.parse_args(["--test-plus", "10", "20", "30"])
        assert args.test_plus == [10, 20, 30]

        # Should require at least one argument
        with pytest.raises(SystemExit):
            parser.parse_args(["--test-plus"])

    def test_optional_nargs_with_const(self):
        """Test base_action='store' with nargs='?' and const."""
        from argparser import ExperimentalAction

        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--test-optional",
            action=ExperimentalAction,
            experimental_enabled=True,
            feature_label="Test Optional Feature",
            base_action="store",
            nargs="?",
            const="CONST_VAL",
            default="DEFAULT_VAL",
            help="Test optional nargs",
        )

        # Without flag - use default
        args_default = parser.parse_args([])
        assert args_default.test_optional == "DEFAULT_VAL"

        # With flag but no value - use const
        args_const = parser.parse_args(["--test-optional"])
        assert args_const.test_optional == "CONST_VAL"

        # With flag and value - use provided value
        args_value = parser.parse_args(["--test-optional", "PROVIDED"])
        assert args_value.test_optional == "PROVIDED"

    def test_append_with_nargs(self):
        """Test base_action='append' with nargs='+'."""
        from argparser import ExperimentalAction

        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--test-append-nargs",
            action=ExperimentalAction,
            experimental_enabled=True,
            feature_label="Test Append Nargs",
            base_action="append",
            nargs="+",
            type=int,
            help="Test append with nargs",
        )

        args = parser.parse_args([
            "--test-append-nargs",
            "1",
            "2",
            "--test-append-nargs",
            "3",
            "4",
            "5",
        ])
        # Each invocation appends a list
        assert args.test_append_nargs == [[1, 2], [3, 4, 5]]

    def test_required_experimental_feature(self):
        """Test experimental feature with required=True."""
        from argparser import ExperimentalAction

        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--test-required",
            action=ExperimentalAction,
            experimental_enabled=True,
            feature_label="Test Required Feature",
            base_action="store",
            type=str,
            required=True,
            help="Test required feature",
        )

        # Without the required flag should fail
        with pytest.raises(SystemExit):
            parser.parse_args([])

        # With the flag should work
        args = parser.parse_args(["--test-required", "value"])
        assert args.test_required == "value"

    def test_metavar_display(self):
        """Test that metavar is properly displayed in help."""
        from argparser import ExperimentalAction

        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--test-metavar",
            action=ExperimentalAction,
            experimental_enabled=True,
            feature_label="Test Metavar",
            base_action="store",
            type=str,
            metavar="FILENAME",
            help="Test metavar",
        )

        help_text = parser.format_help()
        assert "FILENAME" in help_text
        assert "EXPERIMENTAL: Test metavar" in help_text
