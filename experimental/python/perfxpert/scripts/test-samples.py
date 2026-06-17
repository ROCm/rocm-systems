#!/usr/bin/env python3
"""
scripts/test-samples.py — Extract and execute bash/Python samples from Markdown docs
Part of the docs-audit tooling
"""

import os
import re
import subprocess
import sys
import json
from pathlib import Path
from typing import List, Dict, Any

# Python samples that import `perfxpert` need the app root on sys.path.
# This script already lives under `experimental/python/perfxpert/scripts/`,
# so `parent.parent` is the package root we want to prepend.
_PERFXPERT_SRC = Path(__file__).resolve().parent.parent


def _python_env() -> Dict[str, str]:
    """Return a subprocess env with PYTHONPATH pointing at perfxpert/."""
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    parts = [str(_PERFXPERT_SRC)]
    if existing:
        parts.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(parts)
    return env

# Regex patterns for code blocks
BASH_PATTERN = r'```bash\s*\n(.*?)```'
PYTHON_PATTERN = r'```python\s*\n(.*?)```'

# Dangerous commands to skip (destructive, network-dependent)
SKIP_PATTERNS = [
    r'rm\s+-rf',  # Recursive delete
    r'kill',      # Kill processes
    r'reboot',    # Reboot
    r'shutdown',  # Shutdown
    r'curl\s+https?://',  # Network calls
    r'wget',      # Network download
    r'apt-get.*install',  # Package install
    r'pip\s+install',     # Pip install
]


def should_skip_sample(code: str) -> bool:
    """Check if sample has SKIP-SAMPLE or PSEUDOCODE marker."""
    if '# SKIP-SAMPLE' in code or '# PSEUDOCODE' in code:
        return True
    if '#!/' in code and ('SKIP' in code or 'PSEUDOCODE' in code):
        return True
    return False


def should_skip_destructive(code: str) -> bool:
    """Check if code contains dangerous patterns."""
    for pattern in SKIP_PATTERNS:
        if re.search(pattern, code):
            return True
    return False


def extract_samples(doc_path: Path) -> List[Dict[str, Any]]:
    """Extract bash and Python samples from a Markdown file."""
    samples = []

    try:
        content = doc_path.read_text(encoding='utf-8')
    except Exception as e:
        return []

    # Extract bash samples
    for match in re.finditer(BASH_PATTERN, content, re.DOTALL):
        code = match.group(1).strip()
        samples.append({
            'file': str(doc_path),
            'type': 'bash',
            'code': code,
            'line': content[:match.start()].count('\n') + 1,
        })

    # Extract Python samples
    for match in re.finditer(PYTHON_PATTERN, content, re.DOTALL):
        code = match.group(1).strip()
        samples.append({
            'file': str(doc_path),
            'type': 'python',
            'code': code,
            'line': content[:match.start()].count('\n') + 1,
        })

    return samples


def run_bash_sample(code: str) -> Dict[str, Any]:
    """Execute a bash sample."""
    try:
        result = subprocess.run(
            code,
            shell=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return {
            'passed': result.returncode == 0,
            'returncode': result.returncode,
            'stdout': result.stdout[:200],
            'stderr': result.stderr[:200],
        }
    except subprocess.TimeoutExpired:
        return {
            'passed': False,
            'error': 'Timeout (10s)',
        }
    except Exception as e:
        return {
            'passed': False,
            'error': str(e),
        }


def run_python_sample(code: str) -> Dict[str, Any]:
    """Execute a Python sample.

    Runs under a PYTHONPATH that includes the perfxpert source tree
    so `import perfxpert` works in a clean checkout without requiring
    `pip install -e .` first. See `_python_env()` for details.
    """
    try:
        result = subprocess.run(
            ['python3', '-c', code],
            capture_output=True,
            text=True,
            timeout=10,
            env=_python_env(),
        )
        return {
            'passed': result.returncode == 0,
            'returncode': result.returncode,
            'stdout': result.stdout[:200],
            'stderr': result.stderr[:200],
        }
    except subprocess.TimeoutExpired:
        return {
            'passed': False,
            'error': 'Timeout (10s)',
        }
    except Exception as e:
        return {
            'passed': False,
            'error': str(e),
        }


def run_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a sample and return results."""
    code = sample['code']

    # Check for skip markers
    if should_skip_sample(code):
        sample['status'] = 'SKIPPED'
        sample['reason'] = 'Marked with SKIP-SAMPLE or PSEUDOCODE'
        return sample

    # Check for destructive patterns
    if should_skip_destructive(code):
        sample['status'] = 'SKIPPED'
        sample['reason'] = 'Dangerous pattern (rm, kill, install, etc.)'
        return sample

    # Run sample
    if sample['type'] == 'bash':
        result = run_bash_sample(code)
    elif sample['type'] == 'python':
        result = run_python_sample(code)
    else:
        result = {'passed': False, 'error': f"Unknown type: {sample['type']}"}

    sample['status'] = 'PASSED' if result['passed'] else 'FAILED'
    sample['result'] = result
    return sample


def main():
    """Main entry point."""
    # Parse args: optional directory + --strict flag
    strict = '--strict' in sys.argv
    args = [a for a in sys.argv[1:] if a != '--strict']
    search_root = Path(args[0] if args else ".")

    all_samples = []
    failed_count = 0

    for md_file in search_root.rglob("*.md"):
        # Skip hidden and cache dirs
        if any(part.startswith('.') for part in md_file.parts):
            continue
        # Skip legacy ai_analysis tree — being deleted by the agentic refactor
        if 'ai_analysis' in md_file.parts:
            continue
        # Skip the upstream opencode submodule (MIT). Its README translations
        # and bun node_modules contain non-executable bash samples (curl
        # installers, translated docs) that are out of our scope.
        if 'opencode' in md_file.parts or 'node_modules' in md_file.parts:
            continue

        samples = extract_samples(md_file)
        for sample in samples:
            sample = run_sample(sample)
            all_samples.append(sample)
            if sample['status'] == 'FAILED':
                failed_count += 1
                if not strict:
                    print(f"FAIL: {sample['file']}:{sample['line']} ({sample['type']})")
                    if 'result' in sample and 'stderr' in sample['result']:
                        print(f"  {sample['result']['stderr'][:100]}")

    # Output JSON summary
    if not strict:
        print(json.dumps({
            'total': len(all_samples),
            'passed': sum(1 for s in all_samples if s['status'] == 'PASSED'),
            'failed': failed_count,
            'skipped': sum(1 for s in all_samples if s['status'] == 'SKIPPED'),
            'samples': all_samples,
        }, indent=2))
    else:
        # Strict mode: only emit FAIL lines (none if zero failures)
        for s in all_samples:
            if s['status'] == 'FAILED':
                print(f"FAIL: {s['file']}:{s['line']} ({s['type']})")

    return 1 if failed_count > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
