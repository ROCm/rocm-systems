# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import argparse
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

GO_VERSION = re.compile(r"\bgo(\d+)\.(\d+)")


def parse_args(argv: list) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AMD SMI Go checks without GPU access")
    parser.add_argument("--run", default=".")
    parser.add_argument("--package", default="./...")
    parser.add_argument("--race", action="store_true")
    parser.add_argument("--checkptr", action="store_true")
    parser.add_argument("--cgocheck2", action="store_true")
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--vet", action="store_true")
    actions.add_argument("--build-example", action="store_true")
    parser.add_argument("--native", action="store_true")
    parser.add_argument("--include-dir")
    parser.add_argument("--library-dir")
    parser.add_argument("--cc", default="cc")
    args = parser.parse_args(argv)
    if args.native and (not args.include_dir or not args.library_dir):
        parser.error("--native requires --include-dir and --library-dir")
    if not args.native and args.library_dir:
        parser.error("mock mode does not accept --library-dir")
    return args


def go_cache_dir() -> Path:
    cache = Path(tempfile.gettempdir()) / ("amdsmi-agent-go-cache-" + str(os.getuid()))
    cache.mkdir(mode=0o700, exist_ok=True)
    info = cache.lstat()
    if (
        cache.is_symlink()
        or not cache.is_dir()
        or info.st_uid != os.getuid()
        or info.st_mode & 0o777 != 0o700
    ):
        raise ValueError("Go cache must be a private directory owned by the current user")
    return cache


def make_env(*, root: Path, include_dir: Path, library_dir: Path, cc: str, cgocheck2: bool) -> dict:
    env = os.environ.copy()
    for key in (
        "LD_PRELOAD",
        "LD_AUDIT",
        "LIBRARY_PATH",
        "CPATH",
        "C_INCLUDE_PATH",
        "CPLUS_INCLUDE_PATH",
        "GOFLAGS",
        "GOEXPERIMENT",
        "GODEBUG",
        "GOOS",
        "GOARCH",
        "CGO_CPPFLAGS",
        "CGO_CXXFLAGS",
        "CGO_FFLAGS",
        "CGO_CFLAGS_ALLOW",
        "CGO_CFLAGS_DISALLOW",
        "CGO_LDFLAGS_ALLOW",
        "CGO_LDFLAGS_DISALLOW",
        "GOCACHEPROG",
    ):
        env.pop(key, None)
    env.update(
        {
            "GOENV": "off",
            "GOTOOLCHAIN": "local",
            "GOWORK": "off",
            "GOPROXY": "off",
            "GOSUMDB": "off",
            "GONOPROXY": "none",
            "GOPRIVATE": "",
            "GOVCS": "*:off",
            "CGO_ENABLED": "1",
            "CC": cc,
            "CGO_CFLAGS": shlex.quote("-I" + str(include_dir)),
            "CGO_LDFLAGS": " ".join(
                shlex.quote(arg)
                for arg in ("-L" + str(library_dir), "-Wl,-rpath," + str(library_dir))
            ),
            "LD_LIBRARY_PATH": str(library_dir),
            "GOPATH": str(root / "gopath"),
            "GOCACHE": str(go_cache_dir()),
            "GOMODCACHE": str(root / "go-mod-cache"),
            "GOTMPDIR": str(root / "go-tmp"),
            "GODEBUG": "cgocheck=1",
        }
    )
    (root / "go-tmp").mkdir()
    if cgocheck2:
        env["GOEXPERIMENT"] = "cgocheck2"
    return env


def go_command(*, args: argparse.Namespace, output: Path) -> list:
    action = "vet" if args.vet else "build" if args.build_example else "test"
    command = ["go", action]
    if not args.native:
        command += ["-tags=amdsmi_mock"]
    if args.race:
        command += ["-race"]
    if args.checkptr:
        command += ["-gcflags=all=-d=checkptr=2"]
    if args.build_example:
        command += ["-o", str(output), "./examples/telemetry"]
    elif args.vet:
        command += [args.package]
    else:
        command += ["-count=1", "-timeout=120s", "-run", args.run, "-v", args.package]
    return command


def validate_tools(
    *, args: argparse.Namespace, env: dict, include_dir: Path, library_dir: Path
) -> None:
    if sys.platform != "linux":
        raise ValueError("the Go bindings require Linux")
    for program in ("go", args.cc):
        if shutil.which(program, path=env.get("PATH")) is None:
            raise FileNotFoundError("required executable missing: " + program)
    if not (include_dir / "amd_smi" / "amdsmi.h").is_file():
        raise FileNotFoundError("public AMD SMI header missing")
    if args.native and not (library_dir / "libamd_smi.so").is_file():
        raise FileNotFoundError("native AMD SMI shared library missing")
    result = subprocess.run(
        ["go", "version"],
        env=env,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
    )
    match = GO_VERSION.search(result.stdout)
    minimum = (1, 21) if args.cgocheck2 else (1, 20)
    if not match or tuple(map(int, match.groups())) < minimum:
        raise ValueError("local Go toolchain must be at least " + ".".join(map(str, minimum)))
    print(result.stdout.strip(), flush=True)


def run_fixture(*, project: Path, args: argparse.Namespace) -> None:
    include_dir = Path(args.include_dir).resolve() if args.include_dir else project / "include"
    with tempfile.TemporaryDirectory(prefix="amdsmi-agent-go-") as directory:
        root = Path(directory)
        library_dir = Path(args.library_dir).resolve() if args.native else root
        env = make_env(
            root=root,
            include_dir=include_dir,
            library_dir=library_dir,
            cc=args.cc,
            cgocheck2=args.cgocheck2,
        )
        validate_tools(args=args, env=env, include_dir=include_dir, library_dir=library_dir)
        if not args.native:
            sources = sorted((project / "go" / "amdsmi" / "testdata").glob("mock_*.c"))
            if not sources:
                raise FileNotFoundError("no controlled native fixture sources")
            command = [
                args.cc,
                "-std=gnu11",
                "-fPIC",
                "-shared",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-I" + str(include_dir),
                "-Wl,-soname,libamd_smi.so",
                "-o",
                str(root / "libamd_smi.so"),
            ]
            subprocess.run(command + [str(path) for path in sources], env=env, check=True)
        subprocess.run(
            go_command(args=args, output=root / "telemetry"),
            cwd=project / "go",
            env=env,
            check=True,
        )


def main() -> int:
    args = parse_args(sys.argv[1:])
    try:
        run_fixture(project=Path(__file__).resolve().parents[2], args=args)
    except (FileNotFoundError, ValueError, subprocess.CalledProcessError) as error:
        print(str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
