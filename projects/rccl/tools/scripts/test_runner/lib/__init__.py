"""
RCCL Test Runner Library
Provides modules for test configuration, parsing, and execution
"""

from .test_config import TestConfigProcessor
from .test_parser import ArgumentParserInterface, parse_test_output
from .test_executor import TestExecutor, ExitCode, TestResult

__all__ = [
    'TestConfigProcessor',
    'ArgumentParserInterface',
    'parse_test_output',
    'TestExecutor',
    'ExitCode',
    'TestResult'
]

__version__ = '1.0.0'

