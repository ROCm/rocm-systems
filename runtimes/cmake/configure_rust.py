#!/usr/bin/env python3
"""Validate a provisioned Rust toolchain and emit the CMake Cargo invocation."""

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import tomllib

PROFILES = {
    "Debug": "dev",
    "Release": "release",
    "RelWithDebInfo": "relwithdebinfo",
    "MinSizeRel": "minsizerel",
}
HOSTS = {
    "x86_64": "x86_64-unknown-linux-gnu",
    "amd64": "x86_64-unknown-linux-gnu",
    "aarch64": "aarch64-unknown-linux-gnu",
    "arm64": "aarch64-unknown-linux-gnu",
    "ppc64le": "powerpc64le-unknown-linux-gnu",
    "powerpc64le": "powerpc64le-unknown-linux-gnu",
    "ppc64": "powerpc64-unknown-linux-gnu",
    "powerpc64": "powerpc64-unknown-linux-gnu",
    "riscv64": "riscv64gc-unknown-linux-gnu",
}


def validate_host(processor: str, host: str) -> None:
    expected = HOSTS.get(processor.lower())
    if expected is None or host != expected:
        raise ValueError(
            f"Rust host {host} does not match native CMake target {processor}"
        )


def rust_flags(environment: dict[str, str], compiler: str) -> str:
    # Cargo's encoded form separates arguments with ASCII unit separators.
    # Preserve that form; otherwise tokenize the conventional RUSTFLAGS.
    encoded = environment.get("CARGO_ENCODED_RUSTFLAGS")
    flags = encoded.split("\x1f") if encoded else []
    if encoded is None:
        flags = shlex.split(environment.get("RUSTFLAGS", ""))
    # HSA symbol aliases require LLD for cdylibs and Rust test executables.
    flags += [
        "--print=native-static-libs",
        "-C",
        f"linker={compiler}",
        "-C",
        "link-arg=-fuse-ld=lld",
    ]
    return "\x1f".join(flags)


def cmake_set(name: str, values: list[str]) -> str:
    def quote(value: str) -> str:
        for old, new in (("\\", "\\\\"), ('"', '\\"'), ("$", "\\$"), (";", "\\;")):
            value = value.replace(old, new)
        return f'"{value}"'

    return f"set({name} {' '.join(quote(value) for value in values)})\n"


def write_changed(path: Path, content: str) -> None:
    if not path.exists() or path.read_text() != content:
        path.write_text(content)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "source",
        "binary",
        "cargo",
        "rustc",
        "lld",
        "cmake",
        "compiler",
        "cargo-home",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--processor", required=True)
    parser.add_argument("--system", required=True)
    parser.add_argument("--build-type", choices=PROFILES, required=True)
    parser.add_argument("--cargo-config", default="")
    parser.add_argument("--jobs", default="")
    args = parser.parse_args()
    if args.system != "Linux":
        raise ValueError(
            f"Rust linker/environment setup is not implemented for {args.system}"
        )
    if args.jobs and (not args.jobs.isdecimal() or int(args.jobs) < 1):
        raise ValueError("ROCM_RUNTIMES_CARGO_JOBS must be a positive integer")
    if args.cargo_config:
        # Cargo runs in the source tree, while this helper and later builds may
        # run elsewhere. Anchor the input once without dereferencing symlinks.
        args.cargo_config = os.path.abspath(args.cargo_config)
        if not Path(args.cargo_config).is_file():
            raise ValueError(f"Cargo configuration does not exist: {args.cargo_config}")
    with (args.source / "rust-toolchain.toml").open("rb") as stream:
        version = tomllib.load(stream)["toolchain"]["channel"]
    environment = {
        "RUSTUP_TOOLCHAIN": version,
        "RUSTUP_AUTO_INSTALL": "0",
        "RUSTC": str(args.rustc),
        "CARGO_HOME": str(args.cargo_home),
        "CARGO_TARGET_DIR": str(args.binary / "cargo"),
        "CARGO_NET_OFFLINE": "true",
        "PATH": str(args.lld.parent) + os.pathsep + os.environ.get("PATH", ""),
        "CARGO_ENCODED_RUSTFLAGS": rust_flags(dict(os.environ), str(args.compiler)),
    }
    if args.jobs:
        environment["CARGO_BUILD_JOBS"] = args.jobs

    def output(command: list[str | Path]) -> str:
        return subprocess.run(
            command,
            cwd=args.source,
            env=os.environ | environment,
            check=True,
            stdout=subprocess.PIPE,
            text=True,
        ).stdout

    output([args.lld, "--version"])
    rust_info = output([args.rustc, "--version", "--verbose"])
    # rustc --verbose exposes release/host as one "key: value" field per line.
    fields = dict(
        line.split(": ", 1) for line in rust_info.splitlines() if ": " in line
    )
    if fields.get("release") != version:
        raise ValueError(
            f"Provision Rust {version} before configuring; got:\n{rust_info}"
        )
    validate_host(args.processor, fields["host"])
    environment["CARGO_BUILD_TARGET"] = fields["host"]
    cargo_info = output([args.cargo, "--version"])
    if cargo_info.split()[:2] != ["cargo", version]:
        raise ValueError(f"Expected Cargo {version}; got {cargo_info}")
    cargo: list[str | Path] = [args.cargo]
    if args.cargo_config:
        cargo += ["--config", args.cargo_config]
    metadata = output(
        cargo + ["metadata", "--format-version", "1", "--no-deps", "--frozen"]
    )
    json.loads(metadata)  # Validate before emitting the build configuration.
    write_changed(args.binary / "cargo-metadata.json", metadata)
    command = [str(args.cmake), "-E", "env"]
    command += [f"{key}={value}" for key, value in environment.items()]
    command += [str(value) for value in cargo]
    settings = "# Generated by configure_rust.py; do not edit.\n"
    settings += cmake_set("_runtime_cargo_command", command)
    settings += cmake_set("_runtime_cargo_config", [args.cargo_config])
    settings += cmake_set("_runtime_profile", [PROFILES[args.build_type]])
    write_changed(args.binary / "runtime-rust-config.cmake", settings)


if __name__ == "__main__":
    main()
