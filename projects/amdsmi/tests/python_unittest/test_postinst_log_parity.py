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
#
# Postinst <-> preun regex parity.
#
# DEBIAN/postinst.in and RPM/post.in WRITE a line of the form
#   "<iso-timestamp> INFO wrote <site>/amdsmi.pth"
# into /var/log/amd_smi_lib/postinst.log. DEBIAN/prerm.in and RPM/preun.in
# READ that log with `grep -oE 'INFO wrote [^ ]+/amdsmi\.pth' | awk '{print $3}'`
# to find which .pth files to delete. If the writer's format and the reader's
# regex ever drift, package upgrade/removal silently leaves orphan .pth files
# behind.

import re
import shutil
import subprocess
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Synthesised lines exactly as the postinst scriptlets emit them
# (`echo "$(date -Iseconds) INFO wrote $site_packages/amdsmi.pth"`).
SAMPLES = [
    # python3.12 dist-packages on Ubuntu/Debian
    "2026-04-28T07:55:00+00:00 INFO wrote /usr/local/lib/python3.12/dist-packages/amdsmi.pth",
    # python3.9 site-packages on RHEL/AlmaLinux
    "2026-05-01T12:00:00+00:00 INFO wrote /usr/lib/python3.9/site-packages/amdsmi.pth",
    # alt-python3 on /usr/local
    "2026-05-02T09:30:15-04:00 INFO wrote /usr/local/lib/python3.11/dist-packages/amdsmi.pth",
    # interpreter line that must NOT match (parser would treat path as $3)
    "2026-04-28T07:55:00+00:00 INFO interpreter /usr/bin/python3",
]


def _grep_pipeline(line):
    """Run the EXACT grep + awk shell pipeline used in preun.in/prerm.in."""
    proc = subprocess.run(
        # `grep -oE 'INFO wrote [^ ]+/amdsmi\.pth' | awk '{print $3}'`
        ["bash", "-c", r"""grep -oE 'INFO wrote [^ ]+/amdsmi\.pth' | awk '{print $3}'"""],
        input=line + "\n",
        capture_output=True,
        text=True,
        timeout=5,
    )
    return proc.stdout.strip()


class PostinstLogParityTest(unittest.TestCase):
    """The .pth path round-trips through writer + reader unchanged."""

    @unittest.skipUnless(
        shutil.which("bash") and shutil.which("grep") and shutil.which("awk"),
        "bash/grep/awk required to exercise preun pipeline",
    )
    def test_pipeline_extracts_pth_path(self):
        for line in SAMPLES[:3]:
            with self.subTest(line=line):
                want = line.rsplit(" ", 1)[-1]
                got = _grep_pipeline(line)
                self.assertEqual(
                    got, want, "preun pipeline did not extract %r from %r" % (want, line)
                )

    @unittest.skipUnless(shutil.which("bash"), "bash required")
    def test_pipeline_rejects_interpreter_line(self):
        # The interpreter line must not match: extracting the interpreter
        # path here would let preun rm /usr/bin/python3 (catastrophic).
        got = _grep_pipeline(SAMPLES[3])
        self.assertEqual(
            got, "", "interpreter log line MUST NOT be matched by the .pth regex; got %r" % got
        )

    def test_writer_format_present_in_both_scriptlets(self):
        # Sanity: the shared template (used by both postinst.in scriptlets
        # via @INSTALL_AMDSMI_PYTHON_LIB_BODY@ substitution) contains the
        # exact echo format the parser depends on. Also confirm both
        # postinst.in templates still reference the shared body so a
        # future refactor that drops the @INSTALL_AMDSMI_PYTHON_LIB_BODY@
        # placeholder is caught.
        shared = (REPO_ROOT / "cmake_modules" / "install_amdsmi_python_lib.sh.in").read_text()
        deb = (REPO_ROOT / "DEBIAN" / "postinst.in").read_text()
        rpm = (REPO_ROOT / "RPM" / "post.in").read_text()
        token = "INFO wrote "
        self.assertIn(token, shared, "shared install template lost the 'INFO wrote' marker")
        for label, text in (("DEBIAN/postinst.in", deb), ("RPM/post.in", rpm)):
            self.assertIn(
                "@INSTALL_AMDSMI_PYTHON_LIB_BODY@",
                text,
                "%s no longer references the shared install template" % label,
            )
        # Reader uses an equivalent regex. Match tolerantly so that purely
        # cosmetic shell-formatting changes (single vs double quotes, extra
        # whitespace) do not falsely flag a contract break -- the
        # round-trip pipeline test above is the real correctness gate.
        reader = re.compile(r"""grep\s+-oE\s+['"]INFO\s+wrote\s+\[\^\s\]\+/amdsmi\\\.pth['"]""")
        deb_pre = (REPO_ROOT / "DEBIAN" / "prerm.in").read_text()
        rpm_pre = (REPO_ROOT / "RPM" / "preun.in").read_text()
        self.assertRegex(deb_pre, reader)
        self.assertRegex(rpm_pre, reader)

    def test_python_regex_equivalent(self):
        # Pure-Python equivalent of the bash regex -- runs even without bash
        # on the host (some CI runners) and pins the contract independently.
        rx = re.compile(r"INFO wrote ([^ ]+/amdsmi\.pth)")
        for line in SAMPLES[:3]:
            want = line.rsplit(" ", 1)[-1]
            m = rx.search(line)
            self.assertIsNotNone(m, "regex did not match: %r" % line)
            self.assertEqual(m.group(1), want)
        self.assertIsNone(rx.search(SAMPLES[3]))


if __name__ == "__main__":
    unittest.main()
