#!/usr/bin/env python3
"""
tools/_secret_scanner.py — Detect secrets in staged files
Used by pre-commit hook for Phase 9 PR 1
"""

import re
import sys
import subprocess
from pathlib import Path

# Secret patterns
SECRET_PATTERNS = [
    # API keys
    (r"sk-ant-api[0-9a-zA-Z_\-]{20,}", "Anthropic API key"),
    (r"sk-proj-[A-Za-z0-9_\-]{20,}", "OpenAI project key"),
    (r"sk-[A-Za-z0-9_\-]{40,}", "Generic secret key"),
    # AWS
    (r"AKIA[0-9A-Z]{16}", "AWS access key"),
    # GitHub
    (r"ghp_[A-Za-z0-9_]{36,255}", "GitHub personal access token"),
    # npm
    (r"npm_[A-Za-z0-9_]{36,255}", "npm token"),
    # PerfXpert-specific
    (r"PERFXPERT_LLM_\w+_KEY=[^\s]+", "PerfXpert LLM key env var"),
    (r"ANTHROPIC_API_KEY=[^\s]+", "Anthropic API key env var"),
    (r"OPENAI_API_KEY=[^\s]+", "OpenAI API key env var"),
]

# Binary file extensions (check for patterns in bytes)
BINARY_EXTENSIONS = {'.db', '.sqlite', '.sqlite3', '.json', '.yaml', '.yml'}


def scan_file_content(file_path, is_binary=False):
    """Scan file content for secrets."""
    try:
        if is_binary:
            with open(file_path, 'rb') as f:
                content = f.read()
                # Check for string patterns in bytes
                for content_str in content.decode('utf-8', errors='ignore'):
                    for pattern, name in SECRET_PATTERNS:
                        if re.search(pattern, content_str):
                            return (True, name)
        else:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                for pattern, name in SECRET_PATTERNS:
                    if re.search(pattern, content):
                        return (True, name)
    except Exception as e:
        pass

    return (False, None)


def get_staged_files():
    """Get list of staged files from git."""
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return []
    return result.stdout.strip().split('\n')


def check_file_size(file_path):
    """Check if file is suspiciously large (>5MB) and binary."""
    try:
        size = Path(file_path).stat().st_size
        return size > 5 * 1024 * 1024  # 5MB threshold
    except:
        return False


def main():
    """Scan staged files for secrets."""
    staged_files = get_staged_files()
    violations = []

    for file_path in staged_files:
        if not Path(file_path).exists():
            continue

        # Check extension
        suffix = Path(file_path).suffix.lower()
        is_binary = suffix in BINARY_EXTENSIONS

        # Check size
        if is_binary and check_file_size(file_path):
            violations.append({
                'file': file_path,
                'reason': 'Binary file > 5MB (potential data leak)',
            })
            continue

        # Scan content
        has_secret, secret_name = scan_file_content(file_path, is_binary)
        if has_secret:
            violations.append({
                'file': file_path,
                'reason': f'Detected: {secret_name}',
            })

    if violations:
        print("❌ Secret scan failed — found exposed secrets:\n", file=sys.stderr)
        for v in violations:
            print(f"  {v['file']}: {v['reason']}", file=sys.stderr)
        print("\nTo fix:", file=sys.stderr)
        print("  1. Remove the secret", file=sys.stderr)
        print("  2. Run: git reset HEAD <file> && git checkout -- <file>", file=sys.stderr)
        print("  3. Consider rotating the exposed secret", file=sys.stderr)
        return 1

    print("✓ Secret scan passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
