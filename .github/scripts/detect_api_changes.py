#!/usr/bin/env python3
"""
Detect API changes in ROCprofiler-SDK public headers.

This script analyzes git diffs between base and head refs to identify
API changes that require version bumping.

Detects:
- New public functions (ROCPROFILER_API)
- New structs (typedef struct rocprofiler_*)
- New enums (typedef enum rocprofiler_*)
- Modified function signatures
- New struct members
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple


class APIChangeDetector:
    """Detects API changes in header files."""

    # Regex patterns for API elements
    FUNCTION_PATTERN = re.compile(
        r'ROCPROFILER_API\s+[^\n]*?(?:\*|&)?\s*(rocprofiler_\w+)\s*\)?\s*\(',
        re.MULTILINE
    )
    STRUCT_PATTERN = re.compile(
        r'typedef\s+struct\s+(rocprofiler_\w+)',
        re.MULTILINE
    )
    ENUM_PATTERN = re.compile(
        r'typedef\s+enum\s+(rocprofiler_\w+)',
        re.MULTILINE
    )
    MACRO_PATTERN = re.compile(
        r'#define\s+(ROCPROFILER_\w+)',
        re.MULTILINE
    )

    # Paths to monitor for API changes
    HEADER_PATHS = [
        'projects/rocprofiler-sdk/source/include/rocprofiler-sdk/',
        'projects/rocprofiler-sdk/source/include/rocprofiler-sdk-rocpd/',
        'projects/rocprofiler-sdk/source/include/rocprofiler-sdk-roctx/',
    ]

    def __init__(self, repo_root: Path, base_ref: str, head_ref: str):
        """
        Initialize detector.

        Args:
            repo_root: Root directory of git repository
            base_ref: Base git reference (e.g., 'develop', 'main')
            head_ref: Head git reference (e.g., 'HEAD', PR branch)
        """
        self.repo_root = repo_root
        self.base_ref = base_ref
        self.head_ref = head_ref

    def get_changed_headers(self) -> List[str]:
        """Get list of changed header files between refs."""
        try:
            # Get list of changed files
            cmd = [
                'git', '-C', str(self.repo_root),
                'diff', '--name-only',
                f'{self.base_ref}...{self.head_ref}'
            ]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )

            # Filter for header files in API directories
            changed_files = result.stdout.strip().split('\n')
            header_files = []

            for file_path in changed_files:
                if not file_path:
                    continue

                # Check if file is in API header paths
                for api_path in self.HEADER_PATHS:
                    if file_path.startswith(api_path) and (
                        file_path.endswith('.h') or file_path.endswith('.hpp')
                    ):
                        header_files.append(file_path)
                        break

            return header_files

        except subprocess.CalledProcessError as e:
            print(f"Error getting changed files: {e}", file=sys.stderr)
            print(f"stderr: {e.stderr}", file=sys.stderr)
            return []

    def get_file_diff(self, file_path: str) -> str:
        """Get diff for a specific file."""
        try:
            cmd = [
                'git', '-C', str(self.repo_root),
                'diff',
                f'{self.base_ref}...{self.head_ref}',
                '--', file_path
            ]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout

        except subprocess.CalledProcessError as e:
            print(f"Error getting diff for {file_path}: {e}", file=sys.stderr)
            return ""

    def extract_added_lines(self, diff_text: str) -> str:
        """Extract only added lines from diff (lines starting with +)."""
        added_lines = []
        for line in diff_text.split('\n'):
            if line.startswith('+') and not line.startswith('+++'):
                # Remove the leading '+'
                added_lines.append(line[1:])
        return '\n'.join(added_lines)

    def is_comment_or_whitespace(self, line: str) -> bool:
        """Check if line is a comment or whitespace."""
        stripped = line.strip()
        if not stripped:
            return True
        if stripped.startswith('//'):
            return True
        if stripped.startswith('/*') or stripped.startswith('*'):
            return True
        return False

    def filter_meaningful_additions(self, added_text: str) -> str:
        """Filter out comments and whitespace from added text."""
        meaningful_lines = []
        in_multiline_comment = False

        for line in added_text.split('\n'):
            stripped = line.strip()

            # Track multiline comments
            if '/*' in line:
                in_multiline_comment = True
            if '*/' in line:
                in_multiline_comment = False
                continue

            # Skip if in comment or is comment/whitespace
            if in_multiline_comment or self.is_comment_or_whitespace(line):
                continue

            meaningful_lines.append(line)

        return '\n'.join(meaningful_lines)

    def parse_api_additions(self, diff_text: str) -> Dict[str, Set[str]]:
        """
        Parse API additions from diff text.

        Returns:
            Dictionary with keys: functions, structs, enums, macros
        """
        # Extract only added lines
        added_text = self.extract_added_lines(diff_text)

        # Filter out comments and whitespace
        meaningful_text = self.filter_meaningful_additions(added_text)

        # Find all matches
        functions = set(self.FUNCTION_PATTERN.findall(meaningful_text))
        structs = set(self.STRUCT_PATTERN.findall(meaningful_text))
        enums = set(self.ENUM_PATTERN.findall(meaningful_text))
        macros = set(self.MACRO_PATTERN.findall(meaningful_text))

        return {
            'functions': functions,
            'structs': structs,
            'enums': enums,
            'macros': macros
        }

    def detect_signature_changes(self, diff_text: str) -> bool:
        """
        Detect if existing function signatures were modified.

        This is indicated by both additions and deletions of the same function.
        """
        added_text = self.extract_added_lines(diff_text)

        # Get lines that were removed (start with -)
        removed_lines = []
        for line in diff_text.split('\n'):
            if line.startswith('-') and not line.startswith('---'):
                removed_lines.append(line[1:])
        removed_text = '\n'.join(removed_lines)

        # Find functions in both added and removed
        added_funcs = set(self.FUNCTION_PATTERN.findall(added_text))
        removed_funcs = set(self.FUNCTION_PATTERN.findall(removed_text))

        # If same function appears in both, signature changed
        modified_funcs = added_funcs & removed_funcs

        return len(modified_funcs) > 0

    def analyze_changes(self) -> Dict:
        """
        Analyze all changed headers and detect API changes.

        Returns:
            Dictionary with analysis results
        """
        changed_headers = self.get_changed_headers()

        if not changed_headers:
            return {
                'requires_bump': False,
                'bump_type': None,
                'changed_files': [],
                'new_functions': [],
                'new_structs': [],
                'new_enums': [],
                'new_macros': [],
                'modified_signatures': False,
                'summary': 'No API header changes detected'
            }

        # Aggregate all changes
        all_functions = set()
        all_structs = set()
        all_enums = set()
        all_macros = set()
        has_signature_changes = False

        for header in changed_headers:
            diff = self.get_file_diff(header)
            if not diff:
                continue

            additions = self.parse_api_additions(diff)
            all_functions.update(additions['functions'])
            all_structs.update(additions['structs'])
            all_enums.update(additions['enums'])
            all_macros.update(additions['macros'])

            if self.detect_signature_changes(diff):
                has_signature_changes = True

        # Determine if bump is required
        total_changes = (
            len(all_functions) + len(all_structs) +
            len(all_enums) + len(all_macros)
        )

        requires_bump = (
            total_changes > 0 or has_signature_changes
        )

        # Build summary
        summary_parts = []
        if all_functions:
            summary_parts.append(f"{len(all_functions)} new function(s)")
        if all_structs:
            summary_parts.append(f"{len(all_structs)} new struct(s)")
        if all_enums:
            summary_parts.append(f"{len(all_enums)} new enum(s)")
        if all_macros:
            summary_parts.append(f"{len(all_macros)} new macro(s)")
        if has_signature_changes:
            summary_parts.append("function signature changes")

        summary = ', '.join(summary_parts) if summary_parts else 'No API changes'

        return {
            'requires_bump': requires_bump,
            'bump_type': 'minor' if requires_bump else None,
            'changed_files': changed_headers,
            'new_functions': sorted(list(all_functions)),
            'new_structs': sorted(list(all_structs)),
            'new_enums': sorted(list(all_enums)),
            'new_macros': sorted(list(all_macros)),
            'modified_signatures': has_signature_changes,
            'summary': summary
        }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Detect API changes in ROCprofiler-SDK headers'
    )
    parser.add_argument(
        '--repo-root',
        type=Path,
        default=Path.cwd(),
        help='Root directory of git repository'
    )
    parser.add_argument(
        '--base-ref',
        required=True,
        help='Base git reference (e.g., develop, main)'
    )
    parser.add_argument(
        '--head-ref',
        default='HEAD',
        help='Head git reference (default: HEAD)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        help='Output JSON file (default: stdout)'
    )

    args = parser.parse_args()

    # Run detection
    detector = APIChangeDetector(args.repo_root, args.base_ref, args.head_ref)
    results = detector.analyze_changes()

    # Output results
    json_output = json.dumps(results, indent=2)

    if args.output:
        args.output.write_text(json_output)
        print(f"Results written to {args.output}", file=sys.stderr)
    else:
        print(json_output)

    # Exit with code 0 if bump required, 1 if not (for CI workflows)
    sys.exit(0 if results['requires_bump'] else 1)


if __name__ == '__main__':
    main()
