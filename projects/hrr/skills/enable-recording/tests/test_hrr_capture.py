#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tests for hrr_capture.sh, driven the way the agent drives it.

Run: python3 -m pytest tests/test_hrr_capture.py

Preflight decides from the loader's point of view, so these tests build it a
world it cannot tell from a real one: libamdhip64 files that do or do not carry
the capture variable, a workload, and stand-ins for ldd, readelf, ldconfig and
the interpreter's site scan first on PATH. The verdict then depends on the
test, not on what the host has installed. `run` gets workloads that write an
archive of the real shape, or nothing, or exit with a status of their own.

The launcher parsing, signal forwarding, umask and output-permission cases
are in test_inspect_archive.py.
"""

from __future__ import annotations

import json
import os
import re
import signal
import struct
import subprocess
import sys
from pathlib import Path

import pytest

SKILL = Path(__file__).resolve().parent.parent
SCRIPT = SKILL / "scripts" / "hrr_capture.sh"

sys.path.insert(0, str(SKILL / "scripts"))

import inspect_archive  # noqa: E402

# The script requires GNU find, stat and readlink, and reads /proc. The
# argument and output-path cases below run anywhere; the ones that let it
# search for runtimes are Linux only, which is what CI runs.
linux_only = pytest.mark.skipif(
    sys.platform != "linux", reason="the script's runtime search needs GNU tools and /proc"
)

CAPTURE = b"\x7fELF runtime built with HRR\0HIP_HRR_CAPTURE_OUTPUT\0"
NO_CAPTURE = b"\x7fELF runtime built without HRR\0"

LISTING = re.compile(r"^\[capture\]   (.+?) +(/\S+) \(capture: (yes|NO)\)$")


def _function(name: str) -> str:
    function = re.search(
        rf"^{name}\(\).*?^\}}", SCRIPT.read_text(encoding="utf-8"), re.M | re.S
    )
    assert function, f"{name} is gone from the script"
    return function.group(0)


def _executable(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(0o755)
    return path


def _archive_events() -> bytes:
    """An events.bin the inspector counts as a recording: the file header, one
    whole event header and its payload."""
    header = struct.pack("<IHH", inspect_archive.HEADER_MAGIC, 5, 0)
    payload_length = 4096 - len(header)
    event = struct.pack("<HQQQI2x", 1, 1, 0, 0, payload_length)
    return header + event + b"\0" * (payload_length - inspect_archive.EVENT_HEADER_BYTES)


class World:
    """A host whose HIP runtimes, loader answers and site packages the test
    chooses. Everything the script would otherwise ask the real system is
    answered from files under tmp_path."""

    def __init__(self, tmp_path: Path):
        self.root = tmp_path
        self.stubs = tmp_path / "stubs"
        self.rocm = tmp_path / "rocm"
        (self.rocm / "lib").mkdir(parents=True)
        # `ldd BIN` answers from BIN.ldd and `readelf -d BIN` from BIN.readelf,
        # so a workload's linkage is whatever the test wrote beside it.
        _executable(self.stubs / "ldd", '#!/bin/sh\n[ -f "$1.ldd" ] && cat "$1.ldd"\nexit 0\n')
        _executable(
            self.stubs / "readelf",
            '#!/bin/sh\n[ "$1" = -d ] && [ -f "$2.readelf" ] && cat "$2.readelf"\nexit 0\n',
        )
        _executable(
            self.stubs / "ldconfig",
            '#!/bin/sh\n[ -n "$FAKE_LDCONFIG" ] && cat "$FAKE_LDCONFIG"\nexit 0\n',
        )
        # The site scan feeds the interpreter a program on stdin; everything
        # else, the inspector included, goes to the real Python.
        _executable(
            self.stubs / "python3",
            '#!/bin/sh\n'
            'if [ "$1" = - ]; then\n'
            '  cat >/dev/null\n'
            '  [ -n "$FAKE_SITE" ] && printf "%s\\n" "$FAKE_SITE"\n'
            '  exit 0\n'
            'fi\n'
            f'exec "{sys.executable}" "$@"\n',
        )
        self.env = {
            key: value
            for key, value in os.environ.items()
            if key not in ("LD_PRELOAD", "LD_LIBRARY_PATH", "HRR_PLAYBACK", "HIP_HRR_CAPTURE_OUTPUT")
        }
        self.env.update(PATH=f"{self.stubs}:/usr/bin:/bin", ROCM_PATH=str(self.rocm))

    def lib(self, directory: str | Path, name: str = "libamdhip64.so.7", *, capture: bool) -> Path:
        path = self.root / directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(CAPTURE if capture else NO_CAPTURE)
        return path

    def workload(
        self,
        name: str = "app",
        body: str = "exit 0\n",
        *,
        links: Path | None = None,
        rpath: str | None = None,
        runpath: str | None = None,
        bindir: str = "app/bin",
    ) -> Path:
        path = _executable(self.root / bindir / name, f"#!/bin/sh\n{body}")
        if links:
            Path(f"{path}.ldd").write_text(
                "\tlinux-vdso.so.1 (0x00007fff00000000)\n"
                f"\tlibamdhip64.so.7 => {links} (0x00007f0000000000)\n"
            )
        dynamic = ""
        if rpath:
            dynamic += f" 0x000000000000000f (RPATH)              Library rpath: [{rpath}]\n"
        if runpath:
            dynamic += f" 0x000000000000001d (RUNPATH)            Library runpath: [{runpath}]\n"
        if dynamic:
            Path(f"{path}.readelf").write_text(dynamic)
        return path

    def script(self, *args: str, input: str | None = None) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["bash", str(SCRIPT), *args],
            capture_output=True,
            text=True,
            env=self.env,
            input=input,
            stdin=None if input is not None else subprocess.DEVNULL,
            timeout=120,
        )


@pytest.fixture
def world(tmp_path):
    return World(tmp_path)


def listing(stderr: str) -> list[tuple[str, str, str]]:
    """The candidate table preflight prints: (source, path, yes|NO)."""
    return [m.groups() for m in map(LISTING.match, stderr.splitlines()) if m]


# --- arguments --------------------------------------------------------------


def test_no_verb_is_a_usage_error_and_help_is_not(world):
    bare = world.script()
    assert bare.returncode == 1
    assert "usage:" in bare.stderr

    for flag in ("-h", "--help", "help"):
        result = world.script(flag)
        assert result.returncode == 0, flag
        assert "usage:" in result.stderr

    unknown = world.script("capture")
    assert unknown.returncode == 1
    assert "usage:" in unknown.stderr


@pytest.mark.parametrize("flag", ["--output", "--playback", "--min-free-gb"])
def test_a_flag_without_its_value_says_so(world, flag):
    """It used to fall off the end of `shift 2` and exit 1 in silence, which
    is also verify's "nothing captured" status."""
    result = world.script("preflight", flag)

    assert result.returncode == 1
    assert f"error: {flag} needs a value" in result.stderr


def test_min_free_gb_takes_a_whole_number(world):
    result = world.script("preflight", "--min-free-gb", "ten")

    assert result.returncode == 1
    assert "--min-free-gb takes a whole number of GiB, not 'ten'" in result.stderr


def test_the_workload_has_to_come_after_the_separator(world):
    marker = world.root / "ran"
    app = world.workload(body=f'touch "{marker}"\n')
    result = world.script("run", "--skip-preflight", "--output", str(world.root / "out"), str(app))

    assert result.returncode == 1
    assert f"unexpected argument: {app}. The workload command goes after --" in result.stderr
    assert not marker.exists()


def test_run_and_verify_need_their_required_arguments(world):
    no_output = world.script("run", "--", "true")
    assert no_output.returncode == 1
    assert "--output is required" in no_output.stderr

    no_command = world.script("run", "--output", str(world.root / "out"))
    assert no_command.returncode == 1
    assert "no command given" in no_command.stderr

    verify = world.script("verify")
    assert verify.returncode == 1
    assert "--output is required" in verify.stderr


# --- which runtime binds ----------------------------------------------------


@pytest.mark.parametrize(
    "name, loadable",
    [
        ("libamdhip64.so", True),
        ("libamdhip64.so.7", True),
        ("libamdhip64.so.7.1.70100", True),
        ("/opt/rocm/lib/libamdhip64.so.7", True),
        # Backups left beside the real runtime match a `.so.[0-9]*` glob and
        # were judged as if the loader would pick them.
        ("libamdhip64.so.7.hrr-orig", False),
        ("libamdhip64.so.7.bak", False),
        ("libamdhip64.so.hrr-orig", False),
        ("libamdhip64.so.7x", False),
        ("libamdhip64_static.so", False),
    ],
)
def test_only_names_the_loader_can_ask_for_are_runtimes(name, loadable):
    result = subprocess.run(
        ["bash", "-c", f'{_function("is_runtime_name")}\nis_runtime_name "$1"', "-", name],
        capture_output=True,
    )
    assert (result.returncode == 0) == loadable


def test_the_soname_is_listed_before_the_development_alias(tmp_path):
    """Nothing links `libamdhip64.so`; a verdict pronounced on it is about a
    file no workload loads."""
    for name in ("libamdhip64.so", "libamdhip64.so.7", "libamdhip64.so.7.hrr-orig"):
        (tmp_path / name).write_bytes(NO_CAPTURE)
    result = subprocess.run(
        ["bash", "-c",
         f'{_function("is_runtime_name")}\n{_function("emit_libs")}\nemit_libs src "$1"',
         "-", str(tmp_path)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.splitlines() == [
        f"src|{tmp_path}/libamdhip64.so.7",
        f"src|{tmp_path}/libamdhip64.so",
    ]


@linux_only
def test_rpath_and_runpath_are_told_apart_and_origin_expanded(world):
    """DT_RPATH is searched before LD_LIBRARY_PATH and DT_RUNPATH after it;
    listing both as one put the wrong candidate first."""
    app = world.workload(rpath="$ORIGIN/../rpath:/opt/r2", runpath="${ORIGIN}/../runpath")
    origin = app.parent

    def dirs(tag: str) -> list[str]:
        result = subprocess.run(
            ["bash", "-c", f'{_function("binary_rpath_dirs")}\nbinary_rpath_dirs "$1" "$2"',
             "-", str(app), tag],
            capture_output=True,
            text=True,
            env=world.env,
            check=True,
        )
        return result.stdout.splitlines()

    assert dirs("RPATH") == [f"{origin}/../rpath", "/opt/r2"]
    assert dirs("RUNPATH") == [f"{origin}/../runpath"]


@linux_only
def test_candidates_are_listed_in_the_loaders_order(world):
    preload = world.lib("preload", capture=True)
    world.lib("app/rpath", capture=True)
    world.lib("llp", capture=True)
    world.lib("app/runpath", capture=True)
    world.lib("site/torch/lib", "libamdhip64.so", capture=True)
    ldconfig_lib = world.lib("ldcache", capture=True)
    world.lib("rocm/lib", capture=True)
    (world.root / "ldconfig.out").write_text(
        f"\tlibamdhip64.so.7 (libc6,x86-64) => {ldconfig_lib}\n"
    )
    world.env.update(
        LD_PRELOAD=str(preload),
        LD_LIBRARY_PATH=str(world.root / "llp"),
        FAKE_SITE=str(world.root / "site"),
        FAKE_LDCONFIG=str(world.root / "ldconfig.out"),
    )
    app = world.workload(rpath="$ORIGIN/../rpath", runpath="$ORIGIN/../runpath", bindir="app/bin")

    result = world.script("preflight", "--", str(app))

    assert [source for source, _, _ in listing(result.stderr)] == [
        "LD_PRELOAD",
        "workload RPATH",
        "LD_LIBRARY_PATH",
        "workload RUNPATH",
        "bundled package",
        "ldconfig",
        f"{world.rocm}/lib",
    ]


@linux_only
def test_the_loaders_answer_replaces_the_guess(world):
    """The first runtime on LD_LIBRARY_PATH cannot capture, but the binary
    binds another one, and only that one decides."""
    world.lib("llp", capture=False)
    bound = world.lib("rocm/lib", capture=True)
    world.env["LD_LIBRARY_PATH"] = str(world.root / "llp")
    app = world.workload(links=bound)

    result = world.script("preflight", "--", str(app))

    assert result.returncode == 0, result.stderr
    assert ("resolved", str(bound), "yes") in listing(result.stderr)
    assert f"the runtime that will load is {bound} (via bound by app) and it has capture" in (
        result.stderr
    )


@linux_only
def test_a_bound_runtime_without_capture_is_refused_with_the_remedy(world):
    bound = world.lib("rocm/lib", capture=False)
    app = world.workload(links=bound)

    alone = world.script("preflight", "--", str(app))
    assert alone.returncode == 1
    assert f"the runtime that will load is {bound} (via bound by app) and it has NO capture" in (
        alone.stderr
    )
    assert "use a ROCm build whose libamdhip64 has HRR compiled in" in alone.stderr

    # A capture-capable runtime elsewhere changes the advice, and the advice
    # has to say where LD_PRELOAD stops working.
    world.lib("capture/lib", capture=True)
    world.env["LD_LIBRARY_PATH"] = str(world.root / "capture/lib")
    beside = world.script("preflight", "--", str(app))
    assert beside.returncode == 1
    assert "another libamdhip64 here does have capture" in beside.stderr
    assert "LD_PRELOAD does not work" in beside.stderr


@linux_only
def test_the_binarys_own_rpath_beats_ld_library_path(world):
    """With nothing resolved, the first candidate decides, and a DT_RPATH
    runtime is ahead of everything LD_LIBRARY_PATH names."""
    stale = world.lib("app/rpath", capture=False)
    world.lib("llp", capture=True)
    world.env["LD_LIBRARY_PATH"] = str(world.root / "llp")
    app = world.workload(rpath="$ORIGIN/../rpath")

    result = world.script("preflight", "--", str(app))

    assert result.returncode == 1
    assert f"the runtime that will load is {stale} (via workload RPATH) and it has NO capture" in (
        result.stderr
    )


@linux_only
def test_unresolved_with_a_runtime_that_cannot_capture_in_reach_is_refused(world):
    """The PyTorch case: the interpreter links no runtime, the wheel's own
    cannot capture, and a clean preflight became an empty archive."""
    world.lib("llp", capture=True)
    wheel = world.lib("site/torch/lib", "libamdhip64.so", capture=False)
    world.env.update(LD_LIBRARY_PATH=str(world.root / "llp"), FAKE_SITE=str(world.root / "site"))
    app = world.workload()

    result = world.script("preflight", "--", str(app))

    assert result.returncode == 1
    assert "app links no HIP runtime of its own" in result.stderr
    assert f"UNRESOLVED, and refusing: nothing here was resolved, and {wheel} has no" in result.stderr


@linux_only
def test_unresolved_with_every_runtime_capable_passes_as_a_guess(world):
    world.lib("rocm/lib", capture=True)
    app = world.workload()

    result = world.script("preflight", "--", str(app))

    assert result.returncode == 0, result.stderr
    assert "UNRESOLVED: nothing here was resolved" in result.stderr
    assert "That is a guess, not a verdict." in result.stderr


@linux_only
def test_no_runtime_anywhere_is_refused(world):
    result = world.script("preflight", "--", str(world.workload()))

    assert result.returncode == 1
    assert "no libamdhip64 found at all" in result.stderr


@linux_only
def test_without_a_command_preflight_says_what_it_could_not_check(world):
    world.lib("rocm/lib", capture=True)

    result = world.script("preflight")

    assert result.returncode == 0, result.stderr
    assert "no command given, so this is what is visible here" in result.stderr


@linux_only
def test_one_runtime_under_several_names_is_listed_once(world):
    """`libamdhip64.so`, `.so.7` and `.so.7.x.y` are hardlinks to one file,
    which readlink cannot collapse, and it is reachable from two places."""
    soname = world.lib("rocm/lib", capture=True)
    os.link(soname, soname.with_name("libamdhip64.so"))
    os.link(soname, soname.with_name("libamdhip64.so.7.1.70100"))
    world.env["LD_LIBRARY_PATH"] = str(soname.parent)

    result = world.script("preflight", "--", str(world.workload()))

    assert listing(result.stderr) == [("LD_LIBRARY_PATH", str(soname), "yes")]


@linux_only
def test_a_backup_beside_the_runtime_is_not_a_candidate(world):
    """`libamdhip64.so.7.hrr-orig`, left by the overlay recipe, cannot capture
    and is loaded by nothing; listing it refused a capture that would work."""
    runtime = world.lib("llp", capture=True)
    world.lib("llp", "libamdhip64.so.7.hrr-orig", capture=False)
    world.lib("llp", "libamdhip64.so.7.bak", capture=False)
    world.env["LD_LIBRARY_PATH"] = str(runtime.parent)

    result = world.script("preflight", "--", str(world.workload()))

    assert result.returncode == 0, result.stderr
    assert listing(result.stderr) == [("LD_LIBRARY_PATH", str(runtime), "yes")]


@linux_only
def test_the_workload_is_looked_up_on_the_path_its_launcher_sets(world):
    """`env PATH=/x app` runs /x/app, whatever the caller's PATH finds."""
    bound = world.lib("rocm/lib", capture=True)
    world.workload("job", links=bound, bindir="elsewhere")

    launcher_path = f"{world.root / 'elsewhere'}:{world.env['PATH']}"
    result = world.script("preflight", "--", "env", f"PATH={launcher_path}", "job")

    assert result.returncode == 0, result.stderr
    assert "(via bound by job) and it has capture" in result.stderr


@linux_only
def test_a_launcher_path_without_the_system_directories_does_not_blind_preflight(world):
    """`env PATH=/opt/app/bin app` is the workload's PATH, not preflight's.
    Applied to the whole check, it left grep, ldd and readlink unfound, and a
    runtime that can capture was reported as one that cannot."""
    bound = world.lib("rocm/lib", capture=True)
    world.workload("job", links=bound, bindir="opt/app/bin")

    result = world.script("preflight", "--", "env", f"PATH={world.root / 'opt/app/bin'}", "job")

    assert result.returncode == 0, result.stderr
    assert ("resolved", str(bound), "yes") in listing(result.stderr)


@linux_only
def test_a_console_script_finds_its_interpreter_on_the_launchers_path(world):
    """`#!/usr/bin/env python3` is resolved on the PATH the workload runs
    with, so under `env PATH=/venv/bin serve` it is the venv's Python whose
    packages count."""
    wheel = world.lib("venv/site/torch/lib", "libamdhip64.so", capture=False)
    _executable(
        world.root / "venv/bin/python3",
        f'#!/bin/sh\ncat >/dev/null\necho "{world.root / "venv/site"}"\n',
    )
    world.lib("rocm/lib", capture=True)
    _executable(world.root / "venv/bin/serve", "#!/usr/bin/env python3\nimport sys\n")

    result = world.script("preflight", "--", "env", f"PATH={world.root / 'venv/bin'}", "serve")

    assert ("bundled package", str(wheel), "NO") in listing(result.stderr)
    assert result.returncode == 1


@linux_only
def test_a_console_script_is_scanned_with_its_own_interpreter(world):
    """`vllm` and `torchrun` are text files naming their virtualenv's Python.
    The PATH's python3 knows nothing of that environment's wheels."""
    wheel = world.lib("venv/site/torch/lib", "libamdhip64.so", capture=False)
    _executable(
        world.root / "venv/bin/python3",
        f'#!/bin/sh\ncat >/dev/null\necho "{world.root / "venv/site"}"\n',
    )
    world.lib("rocm/lib", capture=True)
    console = _executable(
        world.root / "venv/bin/serve", f"#!{world.root / 'venv/bin/python3'}\nimport sys\n"
    )

    result = world.script("preflight", "--", str(console))

    assert ("bundled package", str(wheel), "NO") in listing(result.stderr)
    assert result.returncode == 1


@linux_only
@pytest.mark.parametrize("where", ["nowhere", "HRR_PLAYBACK", "ROCM_PATH"])
def test_preflight_says_whether_a_reader_is_here(world, where):
    world.lib("rocm/lib", capture=True)
    if where == "HRR_PLAYBACK":
        world.env["HRR_PLAYBACK"] = str(_executable(world.root / "reader/hrr-playback", "#!/bin/sh\n"))
    elif where == "ROCM_PATH":
        _executable(world.rocm / "bin/hrr-playback", "#!/bin/sh\n")

    result = world.script("preflight")

    # A missing reader is not a preflight failure: the manifests-only verify
    # and replay elsewhere are both documented.
    assert result.returncode == 0, result.stderr
    if where == "nowhere":
        assert "no hrr-playback here" in result.stderr
        assert "decode-and-triage/scripts/ensure_playback.sh" in result.stderr
    else:
        assert "hrr-playback is available" in result.stderr


# --- the output path --------------------------------------------------------


def _check_output(
    out: str, *, fstype: str = "ext4", free_gib: int = 100, min_gib: int = 0, cwd: Path | None = None
) -> subprocess.CompletedProcess:
    """The script's output-path check, with the filesystem and free space the
    test names."""
    stubs = (
        "log() { printf '%s\\n' \"$*\" >&2; }\n"
        "fail() { printf 'error: %s\\n' \"$*\" >&2; exit 1; }\n"
        f"path_fstype() {{ echo {fstype}; }}\n"
        "df() { printf 'Filesystem 1024-blocks Used Available Capacity Mounted\\n"
        f"stub 1 0 {free_gib * 1024 * 1024} 0%% /\\n'; }}\n"
    )
    return subprocess.run(
        ["bash", "-c", f'{stubs}{_function("check_output_path")}\ncheck_output_path "$1" "$2"',
         "-", out, str(min_gib)],
        capture_output=True,
        text=True,
        cwd=cwd,
    )


@pytest.mark.parametrize("fstype", ["overlay", "overlayfs"])
def test_an_output_path_on_the_container_layer_is_refused(tmp_path, fstype):
    result = _check_output(str(tmp_path / "run.hrr"), fstype=fstype)

    assert result.returncode == 1
    assert "the archive dies with" in result.stderr


def test_an_output_path_in_ram_is_warned_about_not_refused(tmp_path):
    result = _check_output(str(tmp_path / "run.hrr"), fstype="tmpfs")

    assert result.returncode == 0
    assert "this path is in RAM" in result.stderr


def test_too_little_free_space_is_refused(tmp_path):
    result = _check_output(str(tmp_path / "run.hrr"), free_gib=10, min_gib=50)
    assert result.returncode == 1
    assert "10 GiB free against the 50 GiB this asks for" in result.stderr

    assert _check_output(str(tmp_path / "run.hrr"), free_gib=10, min_gib=10).returncode == 0


def test_a_relative_output_path_is_warned_about(tmp_path):
    result = _check_output("run.hrr", cwd=tmp_path)

    assert result.returncode == 0
    assert "output path is relative" in result.stderr


@linux_only
@pytest.mark.parametrize(
    "path, expected",
    [
        # `-v /host/captures:/data/captures` under an overlay root.
        ("/data/captures", "ext4"),
        ("/data/captures/run.hrr", "ext4"),
        ("/data", "overlay"),
        # A prefix of the mount point's name is not inside it.
        ("/data/captures-old", "overlay"),
        ("/data/scratch/run.hrr", "tmpfs"),
    ],
)
def test_the_filesystem_is_the_innermost_mount_holding_the_path(tmp_path, path, expected):
    mountinfo = tmp_path / "mountinfo"
    mountinfo.write_text(
        "22 1 0:21 / / rw,relatime - overlay overlay rw,lowerdir=/l\n"
        "30 22 8:1 /host/captures /data/captures rw,noatime master:1 - ext4 /dev/sda1 rw\n"
        "31 22 0:30 / /data/scratch rw shared:5 - tmpfs tmpfs rw\n"
    )
    function = _function("path_fstype").replace("/proc/self/mountinfo", str(mountinfo))
    result = subprocess.run(
        ["bash", "-c", f'{function}\npath_fstype "$1"', "-", path],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == expected


# --- run and verify ---------------------------------------------------------


def _capturing_workload(world: World, *, exit_status: int = 0, record: bool = True) -> Path:
    """A workload that does what libamdhip64 does with capture on: writes
    pid-<pid>/ under HIP_HRR_CAPTURE_OUTPUT. It also notes the variable."""
    events = world.root / "events.bin"
    events.write_bytes(_archive_events())
    body = f'printf "%s" "$HIP_HRR_CAPTURE_OUTPUT" > "{world.root / "seen-output"}"\n'
    if record:
        body += (
            'dir="$HIP_HRR_CAPTURE_OUTPUT/pid-$$"\n'
            'mkdir -p "$dir/blobs"\n'
            f'cp "{events}" "$dir/events.bin"\n'
            "printf '{\"pid\": %s, \"parent_pid\": 1, \"complete\": true, "
            "\"event_count\": 1, \"blob_count\": 0}' $$ > \"$dir/manifest.json\"\n"
        )
    body += f"exit {exit_status}\n"
    return world.workload("capturing", body)


def test_run_points_capture_at_the_output_and_offers_the_replay(world):
    out = world.root / "capture"
    app = _capturing_workload(world)

    result = world.script("run", "--skip-preflight", "--no-playback", "--output", str(out), "--", str(app))

    assert result.returncode == 0, result.stderr
    assert (world.root / "seen-output").read_text() == str(out)
    assert "Verdict: recorded" in result.stdout
    assert f"triage_archive.sh --archive {out}/pid-<pid>" in result.stderr


def test_run_with_nothing_captured_reports_it_and_offers_no_replay(world):
    """`run` returns the workload's status by design; the verdict is what
    says nothing was recorded."""
    out = world.root / "capture"
    app = _capturing_workload(world, record=False)

    result = world.script("run", "--skip-preflight", "--no-playback", "--output", str(out), "--", str(app))

    assert result.returncode == 0
    assert "Verdict: nothing captured" in result.stdout
    assert "triage_archive.sh" not in result.stderr


def test_run_returns_the_workloads_own_status_after_verifying(world):
    """A workload that crashed is the one worth capturing, and a caller
    scripting around it still needs its status."""
    out = world.root / "capture"
    app = _capturing_workload(world, exit_status=7)

    result = world.script("run", "--skip-preflight", "--no-playback", "--output", str(out), "--", str(app))

    assert result.returncode == 7
    assert "workload exited 7" in result.stderr
    assert "a non-zero exit is fine here" in result.stderr
    assert "Verdict: recorded" in result.stdout


def test_run_refuses_to_merge_into_an_existing_capture(world):
    """Capture resumes into what is there, and two attempts then read as one
    multi-process archive."""
    out = world.root / "capture"
    (out / "pid-1").mkdir(parents=True)
    out.chmod(0o700)
    app = _capturing_workload(world)

    refused = world.script("run", "--skip-preflight", "--no-playback", "--output", str(out), "--", str(app))
    assert refused.returncode == 1
    assert f"{out} already holds a capture" in refused.stderr
    assert not (world.root / "seen-output").exists(), "the workload ran anyway"

    forced = world.script(
        "run", "--skip-preflight", "--force", "--no-playback", "--output", str(out), "--", str(app)
    )
    assert forced.returncode == 0, forced.stderr
    assert "this run will be added" in forced.stderr
    assert (world.root / "seen-output").exists()


@linux_only
def test_run_warns_when_an_existing_output_is_readable_by_others(world):
    out = world.root / "capture"
    out.mkdir()
    app = _capturing_workload(world)

    out.chmod(0o755)
    shared = world.script("run", "--skip-preflight", "--no-playback", "--output", str(out), "--", str(app))
    assert "can be read by other users" in shared.stderr

    for entry in out.iterdir():
        subprocess.run(["rm", "-rf", str(entry)], check=True)
    out.chmod(0o700)
    private = world.script("run", "--skip-preflight", "--no-playback", "--output", str(out), "--", str(app))
    assert "can be read by other users" not in private.stderr


def test_run_passes_piped_input_to_the_workload(world):
    got = world.root / "stdin"
    app = world.workload(body=f'cat > "{got}"\n')

    result = world.script(
        "run", "--skip-preflight", "--no-playback", "--output", str(world.root / "capture"), "--", str(app),
        input="prompt from a pipe\n",
    )

    assert result.returncode == 0, result.stderr
    assert got.read_text() == "prompt from a pipe\n"


def test_run_waits_for_what_the_workload_left_running(world):
    """A launcher that returns before its workers leaves them writing the
    archive. Verifying then reads it half-written."""
    done = world.root / "done"
    app = world.workload(body=f'( sleep 1; touch "{done}" ) </dev/null >/dev/null 2>&1 &\nexit 0\n')

    result = world.script("run", "--skip-preflight", "--no-playback", "--output", str(world.root / "capture"), "--", str(app))

    assert result.returncode == 0, result.stderr
    assert "processes it started are still running; waiting" in result.stderr
    assert done.exists(), "the wrapper returned while the workload's children were running"


# Runs a command as a child subreaper that waits for that command only: a
# process orphaned below it is reparented here and never reaped, as under a
# container whose PID 1 is `sleep infinity` or a Python entry point.
NON_REAPING_PARENT = (
    "import ctypes, subprocess, sys\n"
    "PR_SET_CHILD_SUBREAPER = 36\n"
    "assert ctypes.CDLL(None).prctl(PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) == 0\n"
    "sys.exit(subprocess.call(sys.argv[1:]))\n"
)


@linux_only
def test_run_does_not_wait_on_a_child_that_has_already_exited(world):
    """`kill -0` is true of a zombie. A child that outlived the workload and
    then exited stayed one under a parent that never reaps, and the wait for
    the workload's group never ended."""
    done = world.root / "done"
    app = world.workload(body=f'( sleep 1; touch "{done}" ) </dev/null >/dev/null 2>&1 &\nexit 0\n')

    harness = subprocess.Popen(
        [sys.executable, "-c", NON_REAPING_PARENT,
         "bash", str(SCRIPT), "run", "--skip-preflight", "--no-playback",
         "--output", str(world.root / "capture"), "--", str(app)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        env=world.env,
        start_new_session=True,
    )
    try:
        _, stderr = harness.communicate(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(harness.pid, signal.SIGKILL)
        harness.communicate()
        pytest.fail("the wrapper was still waiting 30 s after the workload's child exited")

    assert harness.returncode == 0, stderr
    assert done.exists()
    assert "workload exited 0" in stderr


@linux_only
def test_run_stops_before_the_workload_when_nothing_would_be_recorded(world):
    bound = world.lib("rocm/lib", capture=False)
    marker = world.root / "ran"
    app = world.workload(body=f'touch "{marker}"\nexit 5\n', links=bound)
    out = world.root / "capture"

    refused = world.script("run", "--min-free-gb", "0", "--output", str(out), "--", str(app))
    assert refused.returncode == 1
    assert "preflight failed: no capture-capable runtime" in refused.stderr
    assert not marker.exists()

    forced = world.script("run", "--force", "--min-free-gb", "0", "--no-playback", "--output", str(out), "--", str(app))
    assert forced.returncode == 5
    assert marker.exists()


@linux_only
def test_run_repeats_that_the_runtime_was_not_resolved(world):
    """A guessed preflight pass is not a promise, and the run says so where
    the user is looking."""
    world.lib("rocm/lib", capture=True)
    app = world.workload()

    # --force only so the host's own filesystem cannot fail the output check.
    result = world.script(
        "run", "--force", "--min-free-gb", "0", "--no-playback",
        "--output", str(world.root / "capture"), "--", str(app),
    )

    assert result.returncode == 0, result.stderr
    assert "preflight could not resolve which runtime this workload binds" in result.stderr


def test_verify_reports_an_existing_archive_and_passes_json_through(world):
    out = world.root / "capture"
    pid_dir = out / "pid-42"
    (pid_dir / "blobs").mkdir(parents=True)
    (pid_dir / "events.bin").write_bytes(_archive_events())
    (pid_dir / "manifest.json").write_text(
        json.dumps({"pid": 42, "parent_pid": 1, "complete": True, "event_count": 1, "blob_count": 0})
    )

    result = world.script("verify", "--output", str(out), "--json", "--no-playback")

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["verdict"] == "recorded"

    missing = world.script("verify", "--output", str(world.root / "never"), "--no-playback")
    assert missing.returncode == inspect_archive.EXIT_EMPTY
