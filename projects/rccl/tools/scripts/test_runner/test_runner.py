#!/usr/bin/env python3
"""
RCCL Test Runner
Main script for executing RCCL unit tests and MPI tests with hierarchical configuration
"""

import sys
import os
import json
import logging

from lib.test_parser import ArgumentParserInterface
from lib.test_config import TestConfigProcessor
from lib.test_executor import TestExecutor, glob_filter_matches

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main():
    """Main entry point for test runner"""

    # Parse command-line arguments
    parser_interface = ArgumentParserInterface()
    args = parser_interface.process_arguments()

    # Validate flag combinations
    if args.stop_on_rerun_failure and not args.rerun_failed:
        print("ERROR: --stop-on-rerun-failure requires --rerun-failed to be set")
        if args.verbose:
            print("Exiting: Invalid flag combination")
        sys.exit(1)

    # Validate config file exists
    if not os.path.exists(args.config):
        print(f"ERROR: Configuration file not found: {args.config}")
        if args.verbose:
            print("Exiting: Missing configuration file")
        sys.exit(1)

    try:
        # Load and validate configuration
        if args.verbose:
            print("Loading configuration...")
        config_processor = TestConfigProcessor(args.config)
        config_processor.validate_config()
        print("Configuration loaded and validated")

        # Create test executor
        executor = TestExecutor(config_processor, args)

        # Check environment
        if not executor.check_environment():
            if args.verbose:
                print("Exiting: Environment check failed")
            sys.exit(1)

        # Build RCCL (if not --no-build)
        if not args.no_build:
            if not executor.build_rccl():
                print("ERROR: Build failed")
                if args.verbose:
                    print("Exiting: RCCL build failed")
                sys.exit(1)
            # Build rccl-tests (perf binaries) if the config provides a
            # rccl_tests_build_configuration section; no-op otherwise.
            if not executor.build_rccl_tests():
                print("ERROR: rccl-tests build failed")
                if args.verbose:
                    print("Exiting: rccl-tests build failed")
                sys.exit(1)
        else:
                print("SKIP: Build step skipped (--no-build)")

        # Parse and run test suites
        if args.skip_tests:
            print("SKIP: Test execution skipped (--skip-tests)")
        if not args.skip_tests:
            if args.verbose:
                print("\nParsing test suites...")
            test_suites = config_processor.parse_test_suites()

            if args.verbose:
                print("\nCombined Test Suites (JSON):")
                print(json.dumps(test_suites, indent=2))
                print()
                print(f"Found {len(test_suites)} test suite(s)")

            # Print skip messages for disabled or filtered-out test suites upfront.
            # --suite-name uses gtest-style glob filtering (see glob_filter_matches).
            print()
            for suite in test_suites:
                suite_name = suite["suite_details"]["name"]
                enabled = suite["suite_details"].get("enabled", True)
                if not enabled:
                    print(f"SKIP: Test suite '{suite_name}' is disabled")
                elif args.suite_name and not glob_filter_matches(suite_name, args.suite_name):
                    print(f"SKIP: Test suite '{suite_name}' (does not match --suite-name '{args.suite_name}')")

            # Run only enabled (and name-matched) test suites
            # Note: Reruns happen immediately within run_test_suite() if --rerun-failed is set
            for suite in test_suites:
                suite_name = suite["suite_details"]["name"]
                enabled = suite["suite_details"].get("enabled", True)
                if not enabled:
                    continue
                if args.suite_name and not glob_filter_matches(suite_name, args.suite_name):
                    continue
                executor.run_test_suite(suite)

            # Print summary once at the end
            executor.print_summary()

        # Generate coverage report
        if not args.coverage_report:
            print("\nSKIP: Coverage report not requested (use --coverage-report to enable)")
        executor.generate_coverage_report()

        # Emit structured results for the dashboard (no-op unless
        # --emit-results / --db-push was passed). Coverage is emitted too when a
        # report was generated above.
        executor.emit_results()

        # Return based on results
        if executor.test_results:
            from lib.test_executor import TestResult

            # Count failures from original run
            failed = executor.test_results.count(TestResult.RESULT_FAILED.value)
            timeout = executor.test_results.count(TestResult.RESULT_TIMEOUT.value)

            # Also check rerun results if any
            if executor.rerun_results:
                rerun_failed = executor.rerun_results.count(TestResult.RESULT_FAILED.value)
                rerun_timeout = executor.rerun_results.count(TestResult.RESULT_TIMEOUT.value)

                if rerun_failed > 0 or rerun_timeout > 0:
                    if args.verbose:
                        print(f"Exiting: Tests failed after rerun (original: failed={failed}, timeout={timeout}; rerun: failed={rerun_failed}, timeout={rerun_timeout})")
                    sys.exit(1)
                else:
                    # All reruns passed, but original tests failed - this is a success with caveat
                    if args.verbose:
                        print(f"Exiting: All rerun tests passed (original had {failed} failures and {timeout} timeouts, but reruns succeeded)")
                    sys.exit(0)
            elif failed > 0 or timeout > 0:
                # No reruns, but original tests failed
                if args.verbose:
                    print(f"Exiting: Tests failed (failed={failed}, timeout={timeout})")
                sys.exit(1)

        if args.verbose:
            print("Exiting: Test run completed successfully")
        sys.exit(0)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        if args.verbose:
            print("Exiting: User interrupted execution")
        sys.exit(130)  # Standard exit code for SIGINT
    except Exception as e:
        print(f"\nERROR: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
            print("Exiting: Unhandled exception occurred")
        sys.exit(1)


if __name__ == "__main__":
    main()

