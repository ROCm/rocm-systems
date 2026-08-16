#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE.txt for license information
"""
Test Parser Module
Handles command-line argument parsing and test output parsing
"""

import os
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

  # Run a specific individual test by exact name
  %(prog)s -c test_config.json --test-name CE_AllGather_2Ranks

  # Run multiple individual tests (colon-separated)
  %(prog)s -c test_config.json --test-name CE_AllGather_2Ranks:CE_AlltoAll_2Ranks

  # Run all suites whose name contains 'CE Tests'
  %(prog)s -c test_config.json --suite-name "CE Tests"

  # Run two specific suites by substring (colon-separated)
  %(prog)s -c test_config.json --suite-name "CE Tests - 2-Rank:CE Tests - 4-Rank"

  # Run with verbose output
  %(prog)s -c test_config.json -v

  # Skip build and use existing build
  %(prog)s -c test_config.json --no-build

  # Use a custom build directory
  %(prog)s -c test_config.json --build-dir /tmp/my_rccl_build

  # Generate coverage report from existing data
  %(prog)s -c test_config.json --no-build --skip-tests --coverage-report

  # Run on thor2 system (uses thor2 MPI args from config)
  %(prog)s -c test_config.json --no-build --system thor2
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
            help="Run only tests matching any glob pattern; ':' = OR, '*' = wildcard, '-' prefix = exclude, case-sensitive (e.g. 'P2P_*' = all P2P tests; '*:-SHM*' = all except SHM)"
        )
        self.parser.add_argument(
            '--suite-name',
            type=str,
            help="gtest-style glob filter on suite names: ':' = OR, '*' = wildcard, '-' prefix = exclude, case-sensitive (e.g. 'CE*' runs all CE suites; '*:-NET*' runs all except NET)"
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
            '--build-dir',
            type=str,
            help="Custom build directory path (default: <workdir>/build/debug or build/release)"
        )
        self.parser.add_argument(
            '--report-suffix',
            type=str,
            default='',
            help="Suffix for report directory name (default: blank)"
        )
        self.parser.add_argument(
            '--rerun-failed',
            action='store_true',
            help="Rerun failed tests with additional environment variables from config (rerun_env_variables)"
        )
        self.parser.add_argument(
            '--skip-mpi-check',
            action='store_true',
            help="Skip MPI: removes --enable-mpi-tests from build flags, skips MPI installation check, and skips tests with num_ranks > 1"
        )
        self.parser.add_argument(
            '--stop-on-rerun-failure',
            action='store_true',
            help="Stop testing immediately if a rerun also fails (requires --rerun-failed)"
        )
        self.parser.add_argument(
            '--system',
            type=str,
            default='',
            help="Select system-specific MPI args profile from config (e.g. 'ainic', 'thor2')"
        )
        self.parser.add_argument(
            '--mpi-args',
            type=str,
            default='',
            help="Extra mpirun arguments appended after the config's base/suite/test mpi_args "
                 "(MPI tests only). Also accepts the RCCL_TEST_MPI_ARGS environment variable; "
                 "the CLI flag and env var are both appended (CLI first)."
        )
        self.parser.add_argument(
            '--mpich',
            action='store_true',
            default=False,
            help="Use MPICH syntax (-env) instead of OpenMPI (-x) for passing env vars to mpirun"
        )
        self.parser.add_argument(
            '--emit-results',
            action='store_true',
            default=False,
            help="Emit structured, machine-readable results (JSON/JSONL + a .tar.gz "
                 "snapshot) for the results dashboard. Enables per-test log "
                 "capture so perf (busbw/algbw) numbers can be parsed."
        )
        self.parser.add_argument(
            '--results-dir',
            type=str,
            default='',
            help="Directory for emitted results + tarballs (default: <workspace>/results). "
                 "The hourly dashboard scrape pulls <results-dir>/latest.tar.gz."
        )
        self.parser.add_argument(
            '--run-label',
            type=str,
            default='',
            help="Optional label/tag stored with the emitted run (e.g. 'nightly', a PR number)."
        )
        self.parser.add_argument(
            '--tag',
            action='append',
            default=[],
            metavar='TAG',
            help="Tag to attach to the emitted run for filtering in the dashboard "
                 "(repeatable, e.g. --tag nightly --tag mi300x). Merged with --tags."
        )
        self.parser.add_argument(
            '--tags',
            type=str,
            default='',
            help="Comma-separated tags to attach to the emitted run (merged with --tag)."
        )
        self.parser.add_argument(
            '--db-push',
            action='store_true',
            default=False,
            help="Also push results to PostgreSQL (DSN from RCCL_RESULTS_DSN). Best "
                 "effort: on failure/timeout the run continues and the local tarball "
                 "remains the source of truth. Implies --emit-results."
        )
        self.parser.add_argument(
            '--db-timeout',
            type=int,
            default=10,
            help="PostgreSQL connect + statement timeout in seconds for --db-push (default: 10)."
        )

        # --- init-pipeline execution mode (opt-in; default stays serial) ---
        self.parser.add_argument(
            '--exec-mode',
            choices=['serial', 'init-pipeline'],
            default='serial',
            help="Execution mode. 'serial' (default) runs tests one at a time. "
                 "'init-pipeline' overlaps entries' RCCL device-code init behind a "
                 "READY/GO barrier and executes them one at a time (no co-tenancy). "
                 "Experimental."
        )
        self.parser.add_argument(
            '--init-pool',
            type=int,
            default=2,
            help="init-pipeline only: max entries INITIALIZING+READY at once "
                 "(experimental default 2, ceiling 8). The executing entry sits "
                 "outside this bound, so worst-case live process trees = N+1."
        )
        self.parser.add_argument(
            '--loader-policy',
            choices=['continuous', 'quiescent_exec'],
            default='continuous',
            help="init-pipeline only: 'continuous' (default) keeps warming entries "
                 "while one executes (max overlap); 'quiescent_exec' pauses device "
                 "warmups while a perf/dashboard entry executes (cleaner numbers)."
        )
        self.parser.add_argument(
            '--init-timeout',
            type=float,
            default=600,
            help="init-pipeline only: seconds to wait for an entry to reach READY "
                 "before failing it (default 600). 0 = wait indefinitely -- NOT "
                 "recommended: a stuck/mis-configured handshake then hangs the whole "
                 "run instead of failing that one entry."
        )
        self.parser.add_argument(
            '--release-order',
            choices=['ready', 'config'],
            default='ready',
            help="init-pipeline only: order in which READY entries are executed - "
                 "'ready' (default) as they become ready, 'config' in config order."
        )
        self.parser.add_argument(
            '--fork-sweep-policy',
            choices=['legacy', 'independent'],
            default='legacy',
            help="init-pipeline only: 'legacy' (default) preserves each fork test's "
                 "original nested-loop sub-entry order + fail-fast (serial-equivalent); "
                 "'independent' drops order/fail-fast for maximum overlap (opt-in)."
        )
        self.parser.add_argument(
            '--phase-timings',
            action='store_true',
            default=False,
            help="init-pipeline only: report per-entry/aggregate init/queue-wait/"
                 "execution phase timings."
        )
        self.parser.add_argument(
            '--queue-wait-warn',
            type=float,
            default=0,
            help="init-pipeline only: warn if an entry waits longer than this many "
                 "seconds between READY and GO (0 = disabled)."
        )
        self.parser.add_argument(
            '--go-timeout',
            type=float,
            default=0,
            help="init-pipeline only: bounded GO wait (seconds) passed to each "
                 "pipeline entry (RCCL_TEST_GO_TIMEOUT_SEC). On expiry an entry "
                 "reports GO_TIMEOUT and all MPI ranks leave together, so a dead/"
                 "hung runner cannot strand ranks. 0 = indefinite (rely on the "
                 "liveness pipe / process-group teardown). Use a short value only "
                 "for runner-death injection tests, never in production."
        )
        self.parser.add_argument(
            '--emit-manifest',
            action='store_true',
            default=False,
            help="init-pipeline only: print the side-effect-free classification "
                 "manifest (per-entry profile/provenance/binary/disposition/"
                 "expansion) and exit, without running."
        )
        self.parser.add_argument(
            '--allow-serial-only',
            action='store_true',
            default=False,
            help="init-pipeline only: permit a run that resolves to zero pipeline "
                 "entries (all serial). Without it, a zero-pipeline init-pipeline run "
                 "is a hard error (catches mis-classification)."
        )
        self.parser.add_argument(
            '--expand-sweeps',
            action='store_true',
            default=False,
            help="serial mode only: split fork sweeps into the SAME pinned "
                 "per-configuration sub-entries the init-pipeline uses (run serially, "
                 "no warmup/overlap) and emit per-config records. Produces the "
                 "per-config SERIAL BASELINE to diff against an init-pipeline run "
                 "(see tools/compare_results.py)."
        )

    def validate_init_pipeline(self, args):
        """Validate init-pipeline flags; returns a list of error strings (empty ok)."""
        errors = []
        if args.exec_mode == 'init-pipeline':
            if args.init_pool < 1:
                errors.append(f"--init-pool must be >= 1 (got {args.init_pool})")
            if args.init_pool > 8:
                print(f"WARNING: --init-pool {args.init_pool} exceeds the experimental "
                      f"ceiling of 8; proceed only for a Gate A5 sweep.")
            if args.init_timeout < 0:
                errors.append(f"--init-timeout must be >= 0 (got {args.init_timeout})")
        return errors

    def parse_arguments(self):
        """Parse command-line arguments"""
        return self.parser.parse_args()

    def process_arguments(self):
        """Process and validate command-line arguments"""
        self.add_arguments()
        args = self.parse_arguments()
        errors = self.validate_init_pipeline(args)
        if errors:
            self.parser.error("; ".join(errors))
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
            print(f"Test name filter:  {args.test_name if args.test_name else 'all (glob)'}")
            print(f"Suite name filter: {args.suite_name if args.suite_name else 'all suites'}")
            print(f"No build:          {args.no_build}")
            print(f"Skip tests:        {args.skip_tests}")
            print(f"Coverage report:   {args.coverage_report}")
            print(f"Build dir:         {args.build_dir if args.build_dir else 'default'}")
            print(f"Report suffix:     {args.report_suffix}")
            print(f"Rerun failed:      {args.rerun_failed}")
            print(f"Skip MPI check:    {args.skip_mpi_check}")
            print(f"Stop on rerun fail: {args.stop_on_rerun_failure}")
            print(f"System profile:    {args.system if args.system else 'none (use default MPI args)'}")
            _cli_mpi = getattr(args, 'mpi_args', '') or ''
            _env_mpi = os.environ.get('RCCL_TEST_MPI_ARGS', '')
            _extra_mpi = ' '.join(p for p in (_cli_mpi, _env_mpi) if p)
            print(f"Extra MPI args:    {_extra_mpi if _extra_mpi else 'none'}")
            print(f"MPI implementation: {'mpich' if args.mpich else 'openmpi (default)'}")
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

