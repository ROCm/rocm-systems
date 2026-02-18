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
from lib.test_executor import TestExecutor

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

    # Validate config file exists
    if not os.path.exists(args.config):
        print(f"ERROR: Configuration file not found: {args.config}")
        if args.verbose:
            print("Exiting: Missing configuration file")
        return

    try:
        # Load and validate configuration
        if args.verbose:
            print("Loading configuration...")
        config_processor = TestConfigProcessor(args.config)
        config_processor.validate_config()

        # Create test executor
        executor = TestExecutor(config_processor, args)

        # Check environment
        if not executor.check_environment():
            if args.verbose:
                print("Exiting: Environment check failed")
            return

        # Build RCCL (if not --no-build)
        if not args.no_build:
            if not executor.build_rccl():
                print("ERROR: Build failed")
                if args.verbose:
                    print("Exiting: RCCL build failed")
                return

        # Parse and run test suites
        if not args.skip_tests:
            if args.verbose:
                print("\nParsing test suites...")
            test_suites = config_processor.parse_test_suites()

            if args.verbose:
                print("\nCombined Test Suites (JSON):")
                print(json.dumps(test_suites, indent=2))
                print()
                print(f"Found {len(test_suites)} test suite(s)")

            # Print skip messages for disabled test suites upfront
            print()
            for suite in test_suites:
                suite_name = suite["suite_details"]["name"]
                enabled = suite["suite_details"].get("enabled", True)
                if not enabled:
                    print(f"SKIP: Test suite '{suite_name}' is disabled")

            # Run only enabled test suites
            all_results = []
            for suite in test_suites:
                enabled = suite["suite_details"].get("enabled", True)
                if enabled:
                    results = executor.run_test_suite(suite)
                    all_results.extend(results)

            # Print summary once at the end
            executor.print_summary()

        # Generate coverage report
        executor.generate_coverage_report()

        # Return based on results
        if executor.test_results:
            from lib.test_executor import TestResult
            failed = executor.test_results.count(TestResult.RESULT_FAILED.value)
            timeout = executor.test_results.count(TestResult.RESULT_TIMEOUT.value)
            if failed > 0 or timeout > 0:
                if args.verbose:
                    print(f"Exiting: Tests failed (failed={failed}, timeout={timeout})")
                return

        if args.verbose:
            print("Exiting: Test run completed successfully")
        return

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        if args.verbose:
            print("Exiting: User interrupted execution")
        return
    except Exception as e:
        print(f"\nERROR: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
            print("Exiting: Unhandled exception occurred")
        return


if __name__ == "__main__":
    main()

