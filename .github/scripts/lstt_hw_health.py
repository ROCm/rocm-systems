#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Reports the BIOS and GPU state of an LSTT CI host.

Run it by hand on the host and in CI (.github/workflows/lstt-hw-health.yml)
with the same arguments, and the two results are the same.

This repository is public, so the output is kept minimal: a generic id for the
host, the BIOS version, and pass or fail for the module and GPU checks. The real
host name is never printed. When running by hand, set EXPECT_HOSTNAME to the
machine's short host name to check that you are on the machine you mean; the
check is skipped when it is not set, as in CI, where the runner labels pick the
host. The amdgpu version and the output of rocminfo and docker are never read
into the output.

With --bmc it also checks that the host's BMC answers over IPMI. Only
reachability is reported: the BMC firmware version is never read into the output.
"""

import argparse
import os
import shutil
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


def check_host(report):
    expected = os.environ.get("EXPECT_HOSTNAME", "").strip()
    if not expected:
        return True
    if socket.gethostname().split(".")[0] != expected:
        report.fail(
            "- host identity: not the expected machine",
            "The job ran on a different machine than this row expects. Check the "
            "runner labels.",
        )
        return False
    report.line("- host identity: matches")
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


def check_bmc(report):
    def fail(msg):
        report.fail(f"- BMC: {msg}", msg)

    if not any(
        os.path.exists(d) for d in ("/dev/ipmi0", "/dev/ipmi/0", "/dev/ipmidev/0")
    ):
        return fail(
            "no IPMI device on the host (/dev/ipmi0), so it cannot reach its BMC in-band"
        )
    try:
        proc = subprocess.run(
            ["sudo", "-n", "ipmitool", "mc", "info"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=60,
        )
    except FileNotFoundError:
        return fail("sudo or ipmitool is not installed on the host")
    except subprocess.TimeoutExpired:
        return fail(
            "ipmitool did not answer within 60 seconds: the BMC is not responding"
        )
    if proc.returncode != 0:
        return fail(
            f"`sudo -n ipmitool mc info` failed (exit {proc.returncode}): no passwordless "
            "sudo rule for the runner user, or the BMC is not responding"
        )
    report.line("- BMC: reachable")


def check_gpu(report, gfx, image):
    def fail(msg):
        report.fail(f"- GPU: {msg}", msg)

    if not os.path.exists("/dev/kfd"):
        return fail("/dev/kfd is missing, so the GPU is not reachable")
    if not os.path.isdir("/dev/dri"):
        return fail("/dev/dri is missing")
    # In CI the job already runs in the ROCm container, so rocminfo is on PATH.
    # Locally on a host without it, run it in the same image through docker.
    if shutil.which("rocminfo"):
        cmd = ["rocminfo"]
    else:
        cmd = [
            "docker",
            "run",
            "--rm",
            "--device=/dev/kfd",
            "--device=/dev/dri",
            "--group-add",
            "video",
            image,
            "rocminfo",
        ]
    try:
        proc = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True
        )
    except FileNotFoundError:
        return fail("neither rocminfo nor docker is installed")
    rc = proc.returncode
    if rc == 125:
        return fail(
            "docker could not start the container (daemon down, or the image could not be pulled)"
        )
    if rc in (126, 127):
        return fail(f"rocminfo could not run (exit {rc})")
    if rc != 0:
        return fail(f"rocminfo failed (exit {rc}): a broken ROCm stack or driver")
    if gfx not in proc.stdout:
        return fail(
            f"rocminfo ran but did not report {gfx}: a different ASIC, or the GPU is not visible"
        )
    report.line(f"- GPU: {gfx} visible")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--id", required=True, help="generic id of the host, used in the output"
    )
    p.add_argument("--gfx", required=True, help="expected GPU target, e.g. gfx1151")
    p.add_argument(
        "--image", default=DEFAULT_IMAGE, help="container image used to run rocminfo"
    )
    p.add_argument(
        "--bmc", action="store_true", help="also check that the BMC answers over IPMI"
    )
    p.add_argument(
        "--bmc-only",
        action="store_true",
        help="check only the BMC (it needs the host, not the ROCm container)",
    )
    args = p.parse_args()

    report = Report()
    report.line(f"### {args.id}")
    host_ok = check_host(report)
    # A run reports every layer, but not on the wrong host: its values would
    # be attributed to another machine.
    if host_ok and args.bmc_only:
        check_bmc(report)
    elif host_ok:
        check_bios(report)
        check_driver(report)
        if args.bmc:
            check_bmc(report)
        check_gpu(report, args.gfx, args.image)
    return 1 if report.failed else 0


if __name__ == "__main__":
    sys.exit(main())
