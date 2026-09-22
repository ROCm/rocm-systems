"""
RCCL Test Runner Library
Provides modules for test configuration, parsing, and execution
"""

from .test_config import TestConfigProcessor
from .test_parser import ArgumentParserInterface, parse_test_output
from .test_executor import TestExecutor, ExitCode, TestResult, glob_filter_matches
from .results_emitter import ResultsEmitter, parse_perf_output, parse_coverage_report

__all__ = [
    'TestConfigProcessor',
    'ArgumentParserInterface',
    'parse_test_output',
    'TestExecutor',
    'ExitCode',
    'TestResult',
    'glob_filter_matches',
    'ResultsEmitter',
    'parse_perf_output',
    'parse_coverage_report',
]

__version__ = '1.0.0'

