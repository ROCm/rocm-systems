#!/usr/bin/env python3
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
    parser.add_argument("--native-libs", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Cargo command after --")
    args = parser.parse_args()
    metadata = json.loads(args.metadata.read_text())
    members = set(metadata["workspace_members"])
    packages = {
        p["name"]: p["id"] for p in metadata["packages"] if p["id"] in members
    }
    artifacts = [
        Artifact(packages[p], lib, suffix, Path(dest))
        for p, lib, suffix, dest in args.artifact
    ]
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a Cargo command is required after --")
    found: dict[Artifact, Path] = {}
    native_libraries: str | None = None
    with subprocess.Popen(command, stdout=subprocess.PIPE, text=True) as process:
        assert process.stdout is not None
        for line in process.stdout:
            message = json.loads(line)
            if message.get("reason") == "compiler-message":
                diagnostic = message["message"]
                if diagnostic.get("rendered"):
                    print(diagnostic["rendered"], end="", file=sys.stderr)
                text = diagnostic["message"]
                if message["package_id"] == packages["libamdf"] and text.startswith(
                    "native-static-libs:"
                ):
                    native_libraries = text.split(":", 1)[1].strip()
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
                                f"Expected one {artifact.suffix} artifact "
                                f"for {artifact.library}: {paths}"
                            )
                        found[artifact] = paths[0]
        result = process.wait()
    if result:
        return result
    if set(found) != set(artifacts):
        raise RuntimeError(
            f"Cargo did not report all requested artifacts: {set(artifacts) - set(found)}"
        )
    if native_libraries is None:
        raise RuntimeError(
            "Cargo did not report AMDF's native-static-libs; "
            "check the configured Rust flags"
        )
    for artifact, source in found.items():
        copy_changed(source, artifact.destination)
    # GNU linker response syntax. Pass this after the archive, not as an early
    # link option, so --as-needed does not discard required native libraries.
    response = shlex.join(shlex.split(native_libraries)) + "\n"
    if not args.native_libs.exists() or args.native_libs.read_text() != response:
        args.native_libs.write_text(response)
    return 0


if __name__ == "__main__":
    sys.exit(main())
