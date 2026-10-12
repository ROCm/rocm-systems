#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Stage selected Cargo artifacts without teaching CMake Cargo's output layout."""

import argparse
from dataclasses import dataclass
import filecmp
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys


@dataclass(frozen=True)
class Artifact:
    package_id: str
    library: str
    suffix: str
    destination: Path


def copy_changed(source: Path, destination: Path) -> None:
    if not source.is_file() or source.stat().st_size == 0:
        raise RuntimeError(f"Cargo artifact is missing or empty: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists() or not filecmp.cmp(source, destination, shallow=False):
        shutil.copy2(source, destination)


def collect_messages(
    output: str, artifacts: list[Artifact]
) -> tuple[dict[Artifact, Path], dict[tuple[str, str], str]]:
    found: dict[Artifact, Path] = {}
    native_libraries: dict[tuple[str, str], str] = {}
    # Cargo --message-format=json emits one JSON object per stdout line.
    # compiler-artifact provides package/target identity and output filenames;
    # compiler-message wraps rustc diagnostics, including --print=native-static-libs.
    for line in output.splitlines():
        message = json.loads(line)
        if message.get("reason") == "compiler-message":
            diagnostic = message["message"]
            if diagnostic.get("rendered"):
                print(diagnostic["rendered"], end="", file=sys.stderr)
            text = diagnostic["message"]
            if text.startswith("native-static-libs:"):
                key = (message["package_id"], message["target"]["name"])
                native_libraries[key] = text.split(":", 1)[1].strip()
        elif message.get("reason") == "compiler-artifact":
            for artifact in artifacts:
                if (
                    message["package_id"] == artifact.package_id
                    and message["target"]["name"] == artifact.library
                    and not message["profile"]["test"]
                ):
                    paths = [
                        Path(p)
                        for p in message["filenames"]
                        if p.endswith(artifact.suffix)
                    ]
                    if len(paths) != 1:
                        raise RuntimeError(
                            f"Expected one {artifact.suffix} artifact for {artifact.library}: {paths}"
                        )
                    found[artifact] = paths[0]
    return found, native_libraries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument(
        "--artifact",
        nargs=4,
        action="append",
        required=True,
        metavar=("PACKAGE", "LIBRARY", "SUFFIX", "DESTINATION"),
    )
    parser.add_argument(
        "--native-libs",
        nargs=3,
        action="append",
        default=[],
        metavar=("PACKAGE", "LIBRARY", "DESTINATION"),
    )
    parser.add_argument(
        "command", nargs=argparse.REMAINDER, help="Cargo command after --"
    )
    args = parser.parse_args()
    metadata = json.loads(args.metadata.read_text())
    members = set(metadata["workspace_members"])
    packages = {p["name"]: p["id"] for p in metadata["packages"] if p["id"] in members}
    artifacts = [
        Artifact(packages[p], lib, suffix, Path(dest))
        for p, lib, suffix, dest in args.artifact
    ]
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a Cargo command is required after --")
    process = subprocess.run(command, stdout=subprocess.PIPE, text=True, check=False)
    # Render captured compiler diagnostics even when the build fails; never stage
    # partial output from a failed Cargo invocation.
    found, native_libraries = collect_messages(process.stdout, artifacts)
    if process.returncode:
        return process.returncode
    if set(found) != set(artifacts):
        raise RuntimeError(
            f"Cargo did not report all requested artifacts: {set(artifacts) - set(found)}"
        )
    responses: dict[Path, str] = {}
    for package, library, destination in args.native_libs:
        key = (packages[package], library)
        if key not in native_libraries:
            raise RuntimeError(
                f"Cargo did not report native-static-libs for {package}/{library}; check the configured Rust flags"
            )
        # GNU linker response syntax: preserve order and duplication. These must
        # follow the archive so --as-needed does not discard required libraries.
        responses[Path(destination)] = (
            shlex.join(shlex.split(native_libraries[key])) + "\n"
        )
    for artifact, source in found.items():
        copy_changed(source, artifact.destination)
    for path, response in responses.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists() or path.read_text() != response:
            path.write_text(response)
    return 0


if __name__ == "__main__":
    sys.exit(main())
