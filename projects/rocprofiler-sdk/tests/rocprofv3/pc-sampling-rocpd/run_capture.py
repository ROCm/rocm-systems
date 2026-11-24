"""Record PC-sampling capture results for dependent validators."""

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys


def run_capture(command, status_file, skip_regexes):
    status_file = Path(status_file)
    status_file.parent.mkdir(parents=True, exist_ok=True)
    status_file.write_text(json.dumps({"status": "running"}), encoding="utf-8")
    patterns = [re.compile(pattern) for pattern in skip_regexes if pattern]
    skipped = False
    try:
        with subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        ) as process:
            for line in process.stdout:
                print(line, end="", flush=True)
                skipped = skipped or any(pattern.search(line) for pattern in patterns)
            returncode = process.wait()
    except OSError:
        status_file.write_text(json.dumps({"status": "failed"}), encoding="utf-8")
        raise

    status = "skipped" if skipped else "passed" if returncode == 0 else "failed"
    status_file.write_text(
        json.dumps({"status": status, "returncode": returncode}), encoding="utf-8"
    )
    return returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status-file", required=True)
    parser.add_argument("--skip-regex", action="append", default=[])
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a capture command is required")
    return run_capture(command, args.status_file, args.skip_regex)


if __name__ == "__main__":
    sys.exit(main())
