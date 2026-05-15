#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE.txt for license information
"""
Test Parser Module
Handles command-line argument parsing and test output parsing
"""

import re
import argparse


class ArgumentParserInterface:
    """Command-line argument parser for RCCL test runner"""

    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description="RCCL Test Runner - Execute and manage RCCL unit tests and MPI tests",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Run all tests from config
  %(prog)s -c test_config.json

  # Run specific test
  %(prog)s -c test_config.json --test-name NET_AllTests_2Nodes_ETH

  # Run with verbose output
  %(prog)s -c test_config.json -v

  # Skip build and use existing build
  %(prog)s -c test_config.json --no-build

  # Generate coverage report from existing data
  %(prog)s -c test_config.json --no-build --skip-tests --coverage-report
            """
        )

    def add_arguments(self):
        """Add all command-line arguments"""
        self.parser.add_argument(
            '-c', '--config',
            type=str,
            required=True,
            help="Test configuration file (JSON format)"
        )
        self.parser.add_argument(
            '-v', '--verbose',
            action='store_true',
            help="Enable verbose output (detailed logging)"
        )
        self.parser.add_argument(
            '-o', '--output',
            type=str,
            help="Output directory for logs and reports (default: auto-generated)"
        )
        self.parser.add_argument(
            '--test-name',
            type=str,
            help="Run only specific test by name"
        )
        self.parser.add_argument(
            '--no-build',
            action='store_true',
            help="Skip build step and use existing build artifacts"
        )
        self.parser.add_argument(
            '--skip-tests',
            action='store_true',
            help="Skip test execution (useful with --coverage-report)"
        )
        self.parser.add_argument(
            '--coverage-report',
            action='store_true',
            help="Generate code coverage report from profraw files"
        )
        self.parser.add_argument(
            '--overwrite',
            action='store_true',
            help="Overwrite previous build/log directories (default: append timestamp)"
        )
        self.parser.add_argument(
            '--report-suffix',
            type=str,
            default='',
            help="Suffix for report directory name (default: blank)"
        )

    def parse_arguments(self):
        """Parse command-line arguments"""
        return self.parser.parse_args()

    def process_arguments(self):
        """Process and validate command-line arguments"""
        self.add_arguments()
        args = self.parse_arguments()
        self.handle_arguments(args)
        return args

    def handle_arguments(self, args):
        """Handle and display parsed arguments"""
        if args.verbose:
            print("="*80)
            print("RCCL Test Runner - Configuration")
            print("="*80)
            print(f"Config file:       {args.config}")
            print(f"Verbose mode:      {args.verbose}")
            print(f"Output dir:        {args.output if args.output else 'auto-generated'}")
            print(f"Test name filter:  {args.test_name if args.test_name else 'all tests'}")
            print(f"No build:          {args.no_build}")
            print(f"Skip tests:        {args.skip_tests}")
            print(f"Coverage report:   {args.coverage_report}")
            print(f"Overwrite:         {args.overwrite}")
            print(f"Report suffix:     {args.report_suffix}")
            print("="*80)
            print()


def parse_test_output(output):
    """
    Parse test output and extract results

    Args:
        output: String containing test output

    Returns:
        dict: Parsed test results including pass/fail status
    """
    results = {
        'passed': False,
        'failed': False,
        'skipped': False,
        'tests_run': 0,
        'tests_passed': 0,
        'tests_failed': 0,
        'errors': []
    }

    # Google Test output patterns
    gtest_passed = re.search(r'\[\s*PASSED\s*\]\s*(\d+)\s*test', output)
    gtest_failed = re.search(r'\[\s*FAILED\s*\]\s*(\d+)\s*test', output)
    gtest_run = re.search(r'\[==========\]\s*(\d+)\s*test.*ran', output)

    if gtest_run:
        results['tests_run'] = int(gtest_run.group(1))

    if gtest_passed:
        results['tests_passed'] = int(gtest_passed.group(1))

    if gtest_failed:
        results['tests_failed'] = int(gtest_failed.group(1))
        results['failed'] = True
    else:
        results['passed'] = results['tests_run'] > 0

    # Check for skipped tests
    if 'SKIPPED' in output or 'Skipped' in output:
        results['skipped'] = True

    # Extract error messages
    error_pattern = re.compile(r'(ERROR|FAILED|TIMEOUT).*', re.MULTILINE)
    errors = error_pattern.findall(output)
    results['errors'] = errors[:10]  # Limit to first 10 errors

    return results

