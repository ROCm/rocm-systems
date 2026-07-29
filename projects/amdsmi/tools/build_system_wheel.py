#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Assemble a pure-python amdsmi wheel (py3-none-any) using only the stdlib.

No pip/setuptools/wheel are needed at build time. The wheel contains the amdsmi
package .py files only (no bundled .so); the wrapper loads the system
libamd_smi.so via the dynamic linker at runtime. The DEB/RPM post-install
scriptlet installs this wheel into the target interpreter's site-packages.
"""

import argparse
import base64
import hashlib
import os
import re
import sys
import zipfile


def _normalize_version(version: str) -> str:
    # The local segment of a PEP 440 version (after '+') may only contain
    # alphanumerics and dots. A dirty git describe yields e.g. "af02525-dirty";
    # collapse any other run to a dot so the wheel filename and dist-info stay
    # valid (this mirrors what pip/setuptools would emit).
    if "+" in version:
        release, local = version.split("+", 1)
        local = re.sub(r"[^A-Za-z0-9.]+", ".", local).strip(".")
        return release + "+" + local if local else release
    return version


def _record_hash(data: bytes) -> str:
    digest = hashlib.sha256(data).digest()
    return "sha256=" + base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--package-dir", required=True, help="directory containing the 'amdsmi' package"
    )
    parser.add_argument("--version", required=True, help="wheel version string")
    parser.add_argument("--output-dir", required=True, help="directory to write the .whl into")
    args = parser.parse_args()

    version = _normalize_version(args.version)
    pkg_root = os.path.abspath(args.package_dir)
    amdsmi_dir = os.path.join(pkg_root, "amdsmi")
    if not os.path.isdir(amdsmi_dir):
        sys.exit("no amdsmi/ package under {}".format(pkg_root))

    distinfo = "amdsmi-{}.dist-info".format(version)
    records = []
    os.makedirs(args.output_dir, exist_ok=True)
    wheel_path = os.path.join(args.output_dir, "amdsmi-{}-py3-none-any.whl".format(version))

    with zipfile.ZipFile(wheel_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(amdsmi_dir):
            dirs[:] = sorted(d for d in dirs if d != "__pycache__")
            for name in sorted(files):
                # A pure-python wheel: never ship a bundled .so or stale bytecode.
                if name.endswith((".so", ".pyc")):
                    continue
                abs_path = os.path.join(root, name)
                arcname = os.path.relpath(abs_path, pkg_root)
                with open(abs_path, "rb") as handle:
                    data = handle.read()
                zf.writestr(arcname, data)
                records.append((arcname, _record_hash(data), len(data)))

        metadata = (
            "Metadata-Version: 2.1\n"
            "Name: amdsmi\n"
            "Version: {}\n"
            "Summary: AMDSMI Python LIB - AMD GPU Monitoring Library\n"
            "Author: AMD\n"
            "Author-email: amd-smi.support@amd.com\n"
            "Requires-Python: >=3.6\n"
            "Classifier: Programming Language :: Python :: 3\n"
        ).format(version)
        wheel_meta = (
            "Wheel-Version: 1.0\n"
            "Generator: amdsmi-build-system-wheel\n"
            "Root-Is-Purelib: true\n"
            "Tag: py3-none-any\n"
        )
        top_level = "amdsmi\n"

        for arcname, content in (
            (distinfo + "/METADATA", metadata),
            (distinfo + "/WHEEL", wheel_meta),
            (distinfo + "/top_level.txt", top_level),
        ):
            data = content.encode("utf-8")
            zf.writestr(arcname, data)
            records.append((arcname, _record_hash(data), len(data)))

        record_body = "".join("{},{},{}\n".format(n, h, s) for n, h, s in records)
        record_body += "{}/RECORD,,\n".format(distinfo)
        zf.writestr(distinfo + "/RECORD", record_body.encode("utf-8"))

    print(wheel_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
