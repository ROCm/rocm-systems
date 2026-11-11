#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Guard the permissions the maintainer scripts leave on the amd-smi log path.

The log directory must be root-owned and not writable by other users, on a
fresh install and on an upgrade over a directory left loose by an older
package. do_configureLogrotate() is run for real under bash against a temp
directory, so the tests need no root, package manager or container.
"""

import os
import re
import shlex
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path

# .../unit/system/<this file> -> amdsmi source root, or share/amd_smi when installed
# (tests/python/CMakeLists.txt ships the two templates there).
REPO_ROOT = Path(__file__).resolve().parents[4]

SCRIPTLETS = {"deb": "DEBIAN/postinst.in", "rpm": "RPM/post.in"}

# dpkg honors the #!/bin/bash line; rpm runs the scriptlet text with /bin/sh.
SHELLS = {"deb": "bash", "rpm": "sh"}

LOG_DIR_LITERAL = "/var/log/amd_smi_lib"
LOGROTATE_CONF_LITERAL = "/etc/logrotate.d/amd_smi.conf"

LOG_FILE_NAME = "AMD-SMI-lib.log"

LOG_DIR_NAME = "amd_smi_lib"
LOG_DIR_MODE = 0o755
LOOSE_DIR_MODE = 0o777
LOG_FILE_MODE = 0o644
VICTIM_MODE = 0o600

PLANTED_CONTENT = "planted\n"
FORGED_CONTENT = "forged\n"
HISTORY_CONTENT = "history\n"
VICTIM_CONTENT = "secret\n"

# The space exercises the scriptlet's own quoting of the paths it is given.
TEMP_PREFIX = "amd smi "

# Shell functions prepended to the scriptlet text. They win over PATH lookups,
# so no executable stubs are written. chown records its arguments (the test is
# not root); find maps "-user root" to the uid this run treats as root; chmod
# records its target's mode as it was when chmod ran.
SHIMS = """chown() {
  printf '%s\\n' "$*" >> @CHOWN_CALLS@
}
chmod() {
  for _target; do :; done
  printf '%s %s\\n' "$(stat -c %a -- "$_target" 2>/dev/null)" "$_target" >> @CHMOD_CALLS@
  command chmod "$@"
}
find() {
  _n=$#
  _prev=
  while [ "$_n" -gt 0 ]; do
    _arg=$1
    shift
    if [ "$_prev" = -user ] && [ "$_arg" = root ]; then
      set -- "$@" @UID@
    else
      set -- "$@" "$_arg"
    fi
    _prev=$_arg
    _n=$((_n - 1))
  done
  command find "$@"
}
"""

DEFAULT_PATH = "/usr/bin:/bin"


FUNCTION_START = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*\(\) \{", re.M)


def _configure_function(relative_path: str) -> str:
    # The function embeds a heredoc with its own column-0 "}", so slice up to
    # the next top-level function instead of matching the closing brace.
    path = REPO_ROOT / relative_path
    if not path.is_file():
        raise unittest.SkipTest("{} not found".format(relative_path))
    script = path.read_text(encoding="utf-8")
    starts = list(FUNCTION_START.finditer(script))
    for index, match in enumerate(starts[:-1]):
        if script.startswith("do_configureLogrotate", match.start()):
            return script[match.start() : starts[index + 1].start()]
    raise AssertionError("do_configureLogrotate() not followed by a function in " + relative_path)


def _read_calls(path: Path) -> list:
    if not path.is_file():
        return []
    return path.read_text(encoding="utf-8").splitlines()


def _run_configure(
    function_text: str,
    log_dir: Path,
    scratch: Path,
    shell: str,
    trusted_uid: int = None,
    umask: int = None,
) -> dict:
    """Run the function with its literal paths pointed into a temp tree.

    The paths are re-quoted so a directory name containing a space exercises
    the function's own quoting. SHIMS stand in for chown, find and chmod:
    ``-user root`` maps to ``trusted_uid`` (default: the test user), so files
    the test creates count as root-owned, or as another user's when a different
    uid is passed. Returns the recorded calls: ``chown`` as argument strings,
    ``chmod`` as ``"<mode before> <target>"``.
    """
    if trusted_uid is None:
        trusted_uid = os.getuid()
    conf = scratch / "amd_smi.conf"
    chown_calls = scratch / "chown.calls"
    chmod_calls = scratch / "chmod.calls"
    shims = (
        SHIMS.replace("@CHOWN_CALLS@", shlex.quote(str(chown_calls)))
        .replace("@CHMOD_CALLS@", shlex.quote(str(chmod_calls)))
        .replace("@UID@", str(trusted_uid))
    )

    text = function_text.replace(
        "logPath=" + LOG_DIR_LITERAL, "logPath=" + shlex.quote(str(log_dir))
    ).replace(
        "logrotateConfFile=" + LOGROTATE_CONF_LITERAL, "logrotateConfFile=" + shlex.quote(str(conf))
    )
    if LOG_DIR_LITERAL in text or LOGROTATE_CONF_LITERAL in text:
        raise AssertionError("a system path survived substitution; refusing to run")

    prologue = "" if umask is None else "umask {:03o}\n".format(umask)
    proc = subprocess.run(
        [shell, "-c", prologue + shims + text + "\ndo_configureLogrotate"],
        env={"PATH": os.environ.get("PATH", DEFAULT_PATH)},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
    )
    if proc.returncode != 0:
        raise AssertionError("{} exited {}:\n{}".format(shell, proc.returncode, proc.stdout))
    return {"chown": _read_calls(chown_calls), "chmod": _read_calls(chmod_calls)}


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.lstat().st_mode)


def _is_regular_file(path: Path) -> bool:
    return stat.S_ISREG(path.lstat().st_mode)


class LogDirPermissionsTest(unittest.TestCase):
    def _fresh_tree(self) -> Path:
        tmp = tempfile.TemporaryDirectory(prefix=TEMP_PREFIX)
        self.addCleanup(tmp.cleanup)
        return Path(tmp.name)

    def test_upgrade_repairs_world_writable_log_dir(self):
        for kind, relative_path in SCRIPTLETS.items():
            with self.subTest(scriptlet=kind):
                scratch = self._fresh_tree()
                log_dir = scratch / LOG_DIR_NAME
                log_dir.mkdir()
                log_dir.chmod(LOOSE_DIR_MODE)

                _run_configure(_configure_function(relative_path), log_dir, scratch, SHELLS[kind])

                self.assertEqual(oct(_mode(log_dir)), oct(LOG_DIR_MODE))

    def test_log_file_is_not_writable_by_group_or_others(self):
        for kind, relative_path in SCRIPTLETS.items():
            with self.subTest(scriptlet=kind):
                scratch = self._fresh_tree()
                log_dir = scratch / LOG_DIR_NAME

                _run_configure(_configure_function(relative_path), log_dir, scratch, SHELLS[kind])

                self.assertEqual(oct(_mode(log_dir / LOG_FILE_NAME)), oct(LOG_FILE_MODE))

    def test_log_dir_and_file_are_chowned_to_root(self):
        for kind, relative_path in SCRIPTLETS.items():
            with self.subTest(scriptlet=kind):
                scratch = self._fresh_tree()
                log_dir = scratch / LOG_DIR_NAME

                calls = _run_configure(
                    _configure_function(relative_path), log_dir, scratch, SHELLS[kind]
                )["chown"]

                for target in (log_dir, log_dir / LOG_FILE_NAME):
                    self.assertIn("root:root -- {}".format(target), calls)

    def test_planted_symlink_is_replaced_and_target_left_alone(self):
        for kind, relative_path in SCRIPTLETS.items():
            with self.subTest(scriptlet=kind):
                scratch = self._fresh_tree()
                log_dir = scratch / LOG_DIR_NAME
                log_dir.mkdir()
                victim = scratch / "victim"
                victim.write_text(VICTIM_CONTENT, encoding="utf-8")
                victim.chmod(VICTIM_MODE)
                (log_dir / LOG_FILE_NAME).symlink_to(victim)

                _run_configure(_configure_function(relative_path), log_dir, scratch, SHELLS[kind])

                self.assertEqual(oct(_mode(victim)), oct(VICTIM_MODE))
                self.assertEqual(victim.read_text(encoding="utf-8"), VICTIM_CONTENT)
                self.assertFalse((log_dir / LOG_FILE_NAME).is_symlink())

    def test_hard_linked_log_is_replaced_and_other_link_left_alone(self):
        for kind, relative_path in SCRIPTLETS.items():
            with self.subTest(scriptlet=kind):
                scratch = self._fresh_tree()
                log_dir = scratch / LOG_DIR_NAME
                log_dir.mkdir()
                victim = scratch / "victim"
                victim.write_text(VICTIM_CONTENT, encoding="utf-8")
                victim.chmod(VICTIM_MODE)
                try:
                    os.link(str(victim), str(log_dir / LOG_FILE_NAME))
                except OSError as error:
                    self.skipTest("hard links unavailable here: {}".format(error))

                _run_configure(_configure_function(relative_path), log_dir, scratch, SHELLS[kind])

                self.assertEqual(oct(_mode(victim)), oct(VICTIM_MODE))
                self.assertEqual(victim.read_text(encoding="utf-8"), VICTIM_CONTENT)
                self.assertEqual((log_dir / LOG_FILE_NAME).stat().st_nlink, 1)

    def test_log_not_owned_by_root_is_replaced_and_held_handle_cannot_write(self):
        for kind, relative_path in SCRIPTLETS.items():
            with self.subTest(scriptlet=kind):
                scratch = self._fresh_tree()
                log_dir = scratch / LOG_DIR_NAME
                log_dir.mkdir()
                log_file = log_dir / LOG_FILE_NAME
                log_file.write_text(PLANTED_CONTENT, encoding="utf-8")
                planted_inode = log_file.stat().st_ino

                with log_file.open("a", encoding="utf-8") as held:
                    _run_configure(
                        _configure_function(relative_path),
                        log_dir,
                        scratch,
                        SHELLS[kind],
                        trusted_uid=(os.getuid() + 1),
                    )
                    held.write(FORGED_CONTENT)
                    held.flush()

                self.assertNotEqual(log_file.stat().st_ino, planted_inode)
                self.assertNotIn(FORGED_CONTENT, log_file.read_text(encoding="utf-8"))

    def test_root_owned_log_keeps_its_history(self):
        for kind, relative_path in SCRIPTLETS.items():
            with self.subTest(scriptlet=kind):
                scratch = self._fresh_tree()
                log_dir = scratch / LOG_DIR_NAME
                log_dir.mkdir()
                log_file = log_dir / LOG_FILE_NAME
                log_file.write_text(HISTORY_CONTENT, encoding="utf-8")
                history_inode = log_file.stat().st_ino

                _run_configure(_configure_function(relative_path), log_dir, scratch, SHELLS[kind])

                self.assertEqual(log_file.stat().st_ino, history_inode)
                self.assertEqual(log_file.read_text(encoding="utf-8"), HISTORY_CONTENT)

    def test_directory_at_log_path_is_replaced_by_regular_file(self):
        for kind, relative_path in SCRIPTLETS.items():
            with self.subTest(scriptlet=kind):
                scratch = self._fresh_tree()
                log_dir = scratch / LOG_DIR_NAME
                (log_dir / LOG_FILE_NAME / "nested").mkdir(parents=True)

                _run_configure(_configure_function(relative_path), log_dir, scratch, SHELLS[kind])

                self.assertTrue(_is_regular_file(log_dir / LOG_FILE_NAME))
                self.assertEqual(oct(_mode(log_dir / LOG_FILE_NAME)), oct(LOG_FILE_MODE))

    def test_fifo_at_log_path_is_replaced_by_regular_file(self):
        for kind, relative_path in SCRIPTLETS.items():
            with self.subTest(scriptlet=kind):
                scratch = self._fresh_tree()
                log_dir = scratch / LOG_DIR_NAME
                log_dir.mkdir()
                os.mkfifo(str(log_dir / LOG_FILE_NAME))

                _run_configure(_configure_function(relative_path), log_dir, scratch, SHELLS[kind])

                self.assertTrue(_is_regular_file(log_dir / LOG_FILE_NAME))

    def test_log_file_is_created_no_wider_than_its_final_mode(self):
        for kind, relative_path in SCRIPTLETS.items():
            with self.subTest(scriptlet=kind):
                scratch = self._fresh_tree()
                log_dir = scratch / LOG_DIR_NAME
                log_path = str(log_dir / LOG_FILE_NAME)

                calls = _run_configure(
                    _configure_function(relative_path), log_dir, scratch, SHELLS[kind], umask=0
                )["chmod"]

                modes_before = [
                    int(mode, 8)
                    for mode, target in (call.split(" ", 1) for call in calls)
                    if target == log_path
                ]
                self.assertEqual(len(modes_before), 1)
                self.assertEqual(modes_before[0] & ~LOG_FILE_MODE, 0)

    def test_deb_and_rpm_functions_are_identical(self):
        # Two copies of one function: this keeps them from drifting apart.
        self.assertEqual(
            _configure_function(SCRIPTLETS["deb"]), _configure_function(SCRIPTLETS["rpm"])
        )
