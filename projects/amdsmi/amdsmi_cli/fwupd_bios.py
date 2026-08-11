#!/usr/bin/env python3
#
# Copyright (C) Advanced Micro Devices. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""Adapter for the UMA "carveout" BIOS setting exposed through fwupd.

On HP UEFI-HII platforms the amdgpu ``.../device/uma/carveout`` sysfs node is
absent, but the equivalent "Dedicated Graphics Memory" knob is reachable through
fwupd's BIOS-settings interface. The CLI falls back to this adapter when the
kernel node is missing. Everything shells out to ``fwupdmgr`` in JSON mode;
PolicyKit handles privilege, so no explicit ``sudo`` is used.
"""

import json
import os
import re
import shutil
import subprocess
from typing import List, Optional

# Reading BIOS settings works from fwupd 1.8.4; writing the AMD carveout setting
# needs 2.1.1 (Ubuntu 26.04+).
_MIN_READ_VERSION = (1, 8, 4)
_MIN_WRITE_VERSION = (2, 1, 1)

# The knob is matched by its vendor-specific AppStream ``BiosSettingId`` rather
# than a display name, whose semantics are not guaranteed identical across
# vendors. AMD's canonical attribute is preferred and HP's UEFI-HII naming is the
# fallback; order is significant, so the first id present wins.
_CARVEOUT_SETTING_IDS = ("com.amd-gpu.uma_carveout", "com.hp-bioscfg.Dedicated_Graphics_Memory")

# AppStream component id of the fwupd daemon, reported by ``fwupdmgr --version``.
_FWUPD_APPSTREAM_ID = "org.freedesktop.fwupd"

_VERSION_RE = re.compile(r"(\d+)\.(\d+)\.(\d+)")


def _dry_run() -> bool:
    value = os.environ.get("AMDSMI_DRY_RUN", "").strip().lower()
    return value not in ("", "0", "false", "no", "off")


def _parse_version(text: str) -> Optional[tuple]:
    match = _VERSION_RE.search(text or "")
    if match is None:
        return None
    return tuple(int(part) for part in match.groups())


def _run(argv: List[str]) -> Optional[subprocess.CompletedProcess]:
    try:
        return subprocess.run(argv, capture_output=True, text=True, check=False)
    except (OSError, subprocess.SubprocessError):
        return None


def _run_fwupdmgr(args: List[str]) -> Optional[subprocess.CompletedProcess]:
    exe = shutil.which("fwupdmgr")
    if exe is None:
        return None
    return _run([exe, *args])


def _fwupd_version() -> Optional[tuple]:
    result = _run_fwupdmgr(["--version", "--json"])
    if result is None or result.returncode != 0:
        return None
    try:
        payload = json.loads(result.stdout)
    except (ValueError, TypeError):
        return None
    # ``Versions`` reports both the compile-time and runtime fwupd component; the
    # runtime daemon is what actually applies the change, so prefer it.
    runtime_version = None
    compile_version = None
    for entry in payload.get("Versions", []) or []:
        if entry.get("AppstreamId") != _FWUPD_APPSTREAM_ID:
            continue
        parsed = _parse_version(str(entry.get("Version", "")))
        if parsed is None:
            continue
        if entry.get("Type") == "runtime":
            runtime_version = parsed
        elif entry.get("Type") == "compile":
            compile_version = parsed
    return runtime_version or compile_version


def fwupd_available(require_write: bool = False) -> bool:
    """Whether fwupdmgr is present and new enough to read (or write) settings."""
    version = _fwupd_version()
    if version is None:
        return False
    minimum = _MIN_WRITE_VERSION if require_write else _MIN_READ_VERSION
    return version >= minimum


def _find_carveout(settings) -> Optional[dict]:
    # Index settings by their AppStream id (case-insensitive) and return the
    # highest-priority match, so AMD's attribute wins over HP's when both exist.
    by_id = {}
    for setting in settings:
        setting_id = str(setting.get("BiosSettingId", "")).strip().lower()
        if setting_id and setting_id not in by_id:
            by_id[setting_id] = setting
    for candidate in _CARVEOUT_SETTING_IDS:
        setting = by_id.get(candidate.lower())
        if setting is not None:
            return setting
    return None


def get_carveout_setting() -> Optional[dict]:
    """Read the carveout enumeration from fwupd.

    Returns ``{"name", "options", "current_index", "current_value", "read_only"}``
    where ``options`` is ``BiosSettingPossibleValues`` in array order, or ``None``
    when fwupd is unavailable/too old or the setting is absent.
    """
    if not fwupd_available(require_write=False):
        return None
    result = _run_fwupdmgr(["get-bios-settings", "--json"])
    if result is None or result.returncode != 0:
        return None
    try:
        payload = json.loads(result.stdout)
    except (ValueError, TypeError):
        return None
    setting = _find_carveout(payload.get("BiosSettings", []) or [])
    if setting is None:
        return None
    options = [str(value) for value in setting.get("BiosSettingPossibleValues", [])]
    if not options:
        return None
    current_value = str(setting.get("BiosSettingCurrentValue", ""))
    try:
        current_index = options.index(current_value)
    except ValueError:
        current_index = -1
    return {
        "name": str(setting.get("Name", "")),
        "options": options,
        "current_index": current_index,
        "current_value": current_value,
        "read_only": str(setting.get("BiosSettingReadOnly", "False")).strip().lower() == "true",
    }


def set_carveout_setting(index: int) -> None:
    """Set the carveout to ``options[index]`` via fwupd.

    Raises ``ValueError`` for an out-of-range index and ``RuntimeError`` when the
    setting is missing, read-only, fwupd is too old to write, or fwupdmgr fails.
    fwupd's "nothing to do" (value already matches) is treated as success, and
    ``AMDSMI_DRY_RUN`` skips the actual write.
    """
    setting = get_carveout_setting()
    if setting is None:
        raise RuntimeError("fwupd does not expose a UMA carveout BIOS setting")
    options = setting["options"]
    if not 0 <= index < len(options):
        raise ValueError(f"Invalid index {index}. Valid range: 0-{len(options) - 1}")
    if setting["read_only"]:
        raise RuntimeError("The fwupd UMA carveout BIOS setting is read-only")
    if not fwupd_available(require_write=True):
        raise RuntimeError("Writing the BIOS carveout requires fwupd >= 2.1.1 (Ubuntu 26.04+)")
    value = options[index]
    if _dry_run():
        return
    result = _run_fwupdmgr(["set-bios-setting", setting["name"], value])
    if result is None:
        raise RuntimeError("fwupdmgr is not available")
    if result.returncode != 0:
        combined = f"{result.stdout}\n{result.stderr}".lower()
        if "nothing to do" in combined or "nothing-to-do" in combined:
            return
        raise RuntimeError(
            (result.stderr or "").strip() or f"fwupdmgr failed to set '{setting['name']}'"
        )
