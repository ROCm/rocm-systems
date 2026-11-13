#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Reports the BIOS and GPU state of an LSTT CI host.

Run it by hand on the host and in CI (.github/workflows/lstt-hw-health.yml)
with the same arguments, and the two results are the same.

This repository is public, so the output is kept minimal: the host name, the
BIOS version, and pass or fail for the module and GPU checks. The amdgpu
version and the output of rocminfo and docker are never read into the output.
"""

import argparse
import os
import socket
import subprocess
import sys

DEFAULT_IMAGE = "rocm/dev-ubuntu-22.04:7.2"
BIOS_FILE = "/sys/class/dmi/id/bios_version"
CHECKS = ("host", "bios", "driver", "gpu")


class Report:
    def __init__(self):
        self.failed = False
        summary = os.environ.get("GITHUB_STEP_SUMMARY")
        self._summary = open(summary, "a") if summary else None
        self._annotate = os.environ.get("GITHUB_ACTIONS") == "true"

    def line(self, text):
        print(text, flush=True)
        if self._summary:
            self._summary.write(text + "\n")
            self._summary.flush()

    def fail(self, line, error):
        self.failed = True
        self.line(line)
        print(f"::error::{error}" if self._annotate else f"ERROR: {error}", flush=True)


def check_host(report, host):
    got = socket.gethostname().split(".")[0]
    if got != host:
        report.fail(
            f"- host: ran on {got}, not {host}",
            f"This row is for {host} but the job ran on {got}. Check that the "
            f"runner labelled {host} is the one on that machine.",
        )
        return False
    report.line(f"- host: {host}")
    return True


def check_bios(report):
    with open(BIOS_FILE) as f:
        report.line(f"- BIOS: {f.read().strip()}")


def check_driver(report):
    # Test the module's directory, not its version file: only the DKMS driver
    # exports a version, so the inbox driver would otherwise look not loaded.
    if not os.path.isdir("/sys/module/amdgpu"):
        report.fail(
            "- amdgpu: not loaded",
            "The amdgpu module is not loaded, so no GPU test can run on this host.",
        )
        return
    report.line("- amdgpu: loaded")


def check_gpu(report, gfx, image):
    def fail(msg):
        report.fail(f"- GPU: {msg}", msg)

    if not os.path.exists("/dev/kfd"):
        return fail("/dev/kfd is missing on the host, so no container can reach the GPU")
    if not os.path.isdir("/dev/dri"):
        return fail("/dev/dri is missing on the host")
    try:
        proc = subprocess.run(
            [
                "docker", "run", "--rm",
                "--device=/dev/kfd", "--device=/dev/dri", "--group-add", "video",
                image, "rocminfo",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except FileNotFoundError:
        return fail("docker is not installed on the host")
    rc = proc.returncode
    if rc == 125:
        return fail("docker could not start the container (daemon down, or the image could not be pulled)")
    if rc in (126, 127):
        return fail(f"the container started but could not run rocminfo (exit {rc})")
    if rc != 0:
        return fail(f"rocminfo failed inside the container (exit {rc}): a broken ROCm stack or driver")
    if gfx not in proc.stdout:
        return fail(f"rocminfo ran but did not report {gfx}: a different ASIC, or the GPU is not visible")
    report.line(f"- GPU: {gfx} visible")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--host", required=True, help="expected short hostname")
    p.add_argument("--gfx", required=True, help="expected GPU target, e.g. gfx1151")
    p.add_argument("--image", default=DEFAULT_IMAGE, help="container image used to run rocminfo")
    args = p.parse_args()

    report = Report()
    report.line(f"### {args.host}")
    host_ok = check_host(report, args.host)
    # A run reports every layer, but not on the wrong host: its values would
    # be attributed to another machine.
    if host_ok:
        check_bios(report)
        check_driver(report)
        check_gpu(report, args.gfx, args.image)
    return 1 if report.failed else 0


if __name__ == "__main__":
    sys.exit(main())
