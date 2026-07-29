#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

SUMMARY_PATTERN = re.compile(
    r"^  total=(?P<total>\d+) changed=(?P<changed>\d+) shown=(?P<shown>\d+)$",
    re.MULTILINE,
)
SOURCE_ID_PATTERN = re.compile(
    r"^source_code_object_id: (?P<source_id>fnv1a64:[0-9a-f]{16})$",
    re.MULTILINE,
)
RUNTIME_PATTERN = re.compile(
    r"^\[hsa-hotswap-rj\] eager translation "
    r"source_id=(?P<source_id>fnv1a64:[0-9a-f]{16}) "
    r"input_revision=(?P<input_revision>[a-z0-9]+) "
    r"output_revision=(?P<output_revision>[a-z0-9]+) "
    r"outcome=(?P<outcome>[a-z_]+) "
    r"changed=(?P<changed>\d+) "
    r"input_bytes=(?P<input_bytes>\d+) "
    r"output_bytes=(?P<output_bytes>\d+) "
    r"translation_status=(?P<translation_status>-?\d+) "
    r"status=(?P<status>-?\d+)$",
    re.MULTILINE,
)
RUNTIME_PREFIX = "[hsa-hotswap-rj] eager translation "


def main() -> int:
    args = parse_args()
    expectations = load_expectations(args.expectations)
    selected_tests = load_selection(args.selection)
    if set(expectations) != set(selected_tests):
        missing = sorted(set(selected_tests) - set(expectations))
        extra = sorted(set(expectations) - set(selected_tests))
        raise ValueError(
            f"selection and expectations differ: missing={missing}, extra={extra}"
        )
    failures = []

    for test, expected_changed in expectations.items():
        try:
            binary = find_one(args.artifact_directory, f"build/bin/{test}")
            reference_log = find_one(
                args.artifact_directory,
                f"logs/{test}.reference.run.log",
            )
            runtime_log = find_one(
                args.artifact_directory,
                f"logs/{test}.comparison.run.log",
            )
            diff = run_translation(args.translator, binary)
            source_id = parse_source_id(test, diff)
            changed = parse_changed_count(test, diff)
            if changed != expected_changed:
                raise ValueError(
                    f"reported changed={changed}; expected {expected_changed}"
                )

            reference_text = reference_log.read_text(encoding="utf-8")
            if RUNTIME_PREFIX in reference_text:
                raise ValueError(f"{reference_log} unexpectedly activated translation")

            runtime_text = runtime_log.read_text(encoding="utf-8")
            try:
                matching_runtime_records, runtime_translations = (
                    validate_runtime_evidence(
                        runtime_text,
                        source_id,
                        expected_changed,
                        args.expected_runtime_translations,
                    )
                )
            except ValueError as error:
                raise ValueError(f"{runtime_log}: {error}") from error

            output = args.artifact_directory / "translation-diffs" / f"{test}.log"
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(diff, encoding="utf-8")
            print(
                f"{test}: source_id={source_id} changed={changed} "
                f"matching_runtime_records={matching_runtime_records} "
                f"runtime_translations={runtime_translations}"
            )
        except (OSError, RuntimeError, ValueError) as error:
            failures.append(f"{test}: {error}")

    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate offline rewrite counts and fixture-specific runtime "
            "translation evidence for gfx1250 semantic programs."
        )
    )
    parser.add_argument("--translator", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--expectations", type=Path, required=True)
    parser.add_argument("--artifact-directory", type=Path, required=True)
    parser.add_argument("--expected-runtime-translations", type=int, required=True)
    return parser.parse_args()


def load_expectations(path: Path) -> dict[str, int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != 1:
        raise ValueError(f"{path} must use schema 1")
    tests = payload.get("tests")
    if not isinstance(tests, dict) or not tests:
        raise ValueError(f"{path} field 'tests' must be a non-empty object")

    expectations = {}
    for test, entry in tests.items():
        if not isinstance(test, str) or not test:
            raise ValueError(f"{path} test names must be non-empty strings")
        if not isinstance(entry, dict):
            raise ValueError(f"{path} test '{test}' must contain an object")
        changed = entry.get("instructions_requiring_rewrite")
        if isinstance(changed, bool) or not isinstance(changed, int) or changed <= 0:
            raise ValueError(
                f"{path} test '{test}' must require a positive rewrite count"
            )
        expectations[test] = changed
    return expectations


def load_selection(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    tests = payload.get("semantics")
    if (
        not isinstance(tests, list)
        or not tests
        or any(not isinstance(test, str) or not test for test in tests)
    ):
        raise ValueError(f"{path} field 'semantics' must be a non-empty string list")
    if len(tests) != len(set(tests)):
        raise ValueError(f"{path} field 'semantics' contains duplicates")
    return tests


def find_one(root: Path, suffix: str) -> Path:
    matches = [
        path for path in root.rglob(Path(suffix).name) if str(path).endswith(suffix)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"{root} has {len(matches)} paths ending in {suffix!r}; expected one"
        )
    return matches[0]


def run_translation(translator: Path, binary: Path) -> str:
    command = [
        str(translator),
        str(binary),
        "--input-target",
        "gfx1250",
        "--output-target",
        "gfx1250",
        "--input-revision",
        "b0",
        "--output-revision",
        "a0",
        "--code-object-index",
        "0",
        "--output-mode",
        "diff",
    ]
    process = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",
        check=False,
    )
    if process.returncode:
        raise RuntimeError(
            f"offline translation failed with status {process.returncode}: "
            f"{process.stderr.strip()}"
        )
    return process.stdout


def parse_changed_count(test: str, diff: str) -> int:
    matches = list(SUMMARY_PATTERN.finditer(diff))
    if len(matches) != 1:
        raise ValueError(
            f"offline translation for {test} emitted {len(matches)} summaries; "
            "expected one"
        )
    return int(matches[0].group("changed"))


def parse_source_id(test: str, diff: str) -> str:
    matches = list(SOURCE_ID_PATTERN.finditer(diff))
    if len(matches) != 1:
        raise ValueError(
            f"offline translation for {test} emitted {len(matches)} source identities; "
            "expected one"
        )
    return matches[0].group("source_id")


def parse_runtime_records(runtime_text: str) -> list[re.Match[str]]:
    return list(RUNTIME_PATTERN.finditer(runtime_text))


def find_matching_runtime_records(
    runtime_records: list[re.Match[str]],
    source_id: str,
    expected_changed: int,
) -> list[re.Match[str]]:
    return [
        record
        for record in runtime_records
        if record.group("source_id") == source_id
        and record.group("input_revision") == "b0"
        and record.group("output_revision") == "a0"
        and record.group("outcome") == "translated"
        and int(record.group("changed")) == expected_changed
        and int(record.group("translation_status")) == 0
        and int(record.group("status")) == 0
    ]


def successful_runtime_records(
    runtime_records: list[re.Match[str]],
) -> list[re.Match[str]]:
    return [
        record
        for record in runtime_records
        if record.group("input_revision") == "b0"
        and record.group("output_revision") == "a0"
        and record.group("outcome") == "translated"
        and int(record.group("translation_status")) == 0
        and int(record.group("status")) == 0
    ]


def validate_runtime_evidence(
    runtime_text: str,
    source_id: str,
    expected_changed: int,
    expected_runtime_translations: int,
) -> tuple[int, int]:
    runtime_translations = runtime_text.count(RUNTIME_PREFIX)
    runtime_records = parse_runtime_records(runtime_text)
    successful_records = successful_runtime_records(runtime_records)
    matching_records = find_matching_runtime_records(
        successful_records, source_id, expected_changed
    )
    if not matching_records:
        observed = [
            {
                "source_id": record.group("source_id"),
                "input_revision": record.group("input_revision"),
                "output_revision": record.group("output_revision"),
                "outcome": record.group("outcome"),
                "changed": int(record.group("changed")),
                "translation_status": int(record.group("translation_status")),
                "status": int(record.group("status")),
            }
            for record in runtime_records
        ]
        raise ValueError(
            "no successful runtime record matching "
            f"source_id={source_id}, revisions=b0-to-a0, and "
            f"changed={expected_changed}; observed={observed}"
        )
    if (
        runtime_translations != expected_runtime_translations
        or len(runtime_records) != expected_runtime_translations
        or len(successful_records) != expected_runtime_translations
    ):
        raise ValueError(
            f"found {runtime_translations} runtime translations, "
            f"{len(runtime_records)} machine-readable records, and "
            f"{len(successful_records)} successful records; expected "
            f"{expected_runtime_translations} of each"
        )
    return len(matching_records), runtime_translations


if __name__ == "__main__":
    sys.exit(main())
