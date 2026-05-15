#!/usr/bin/env python3
##############################################################################
# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

##############################################################################

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
# .../rocprofiler-compute/tools/config_management

REPO_ROOT = SCRIPT_DIR.parents[1]
# .../rocprofiler-compute

TOOLS_DIR = SCRIPT_DIR

SOC_ROOT = REPO_ROOT / "src" / "rocprof_compute_soc"
ANALYSIS_CONFIGS = SOC_ROOT / "analysis_configs"

TEMPLATE_FILE = ANALYSIS_CONFIGS / "gfx9_config_template.yaml"
HASH_JSON = REPO_ROOT / "src" / "utils" / ".config_hashes.json"
BACKUP_DIR = SCRIPT_DIR / "backups"

PYTHON = sys.executable

VERIFY_SCRIPT = TOOLS_DIR / "verify_against_config_template.py"
PARSE_TEMPLATE_SCRIPT = TOOLS_DIR / "parse_config_template.py"
GENERATE_DELTAS_SCRIPT = TOOLS_DIR / "generate_config_deltas.py"
APPLY_DELTAS_SCRIPT = TOOLS_DIR / "apply_config_deltas.py"
HASH_CHECKER_SCRIPT = REPO_ROOT / "src" / "utils" / "hash_checker.py"
HASH_MANAGER_SCRIPT = TOOLS_DIR / "hash_manager.py"


def run(cmd):
    print("\n$", " ".join(str(c) for c in cmd))
    return subprocess.run(cmd, cwd=str(REPO_ROOT)).returncode


def fatal(msg):
    print(f"\nFATAL: {msg}")
    sys.exit(1)


def confirm(prompt):
    ans = input(f"{prompt} [y/N]: ").strip().lower()
    return ans in ("y", "yes")


def backup(paths):
    BACKUP_DIR.mkdir(exist_ok=True)
    backup_path = BACKUP_DIR / f"backup_{int(time.time())}"
    backup_path.mkdir()

    for p in paths:
        if not p.exists():
            continue
        dest = backup_path / p.name
        if p.is_dir():
            shutil.copytree(p, dest)
        else:
            shutil.copy2(p, dest)

    print(f"\nBackup created at {backup_path}")
    return backup_path


def restore(backup_path, paths):
    print("\nRestoring from backup...")
    for p in paths:
        src = backup_path / p.name
        if not src.exists():
            continue
        if p.exists():
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink()
        if src.is_dir():
            shutil.copytree(src, p)
        else:
            shutil.copy2(src, p)
    print("Restore complete.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ci", action="store_true")
    parser.add_argument("--hash-only", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--edit-existing", action="store_true")
    parser.add_argument("--promote", metavar="ARCH")
    args = parser.parse_args()

    # --------------------------------------------------------
    # CI / HASH-ONLY PATH (non-mutating)
    # --------------------------------------------------------
    if args.ci or args.hash_only:
        if not HASH_CHECKER_SCRIPT.exists():
            fatal("hash_checker.py not found")
        sys.exit(run([PYTHON, HASH_CHECKER_SCRIPT]))

    # --------------------------------------------------------
    # HARD PREFLIGHT (STRUCTURAL VALIDATION) for all non-hash paths
    # --------------------------------------------------------
    if not VERIFY_SCRIPT.exists():
        fatal("verify_against_config_template.py not found")

    rc = run([PYTHON, VERIFY_SCRIPT, ANALYSIS_CONFIGS, TEMPLATE_FILE])
    if rc != 0:
        fatal("Template / architecture verification failed")

    # --------------------------------------------------------
    # VALIDATE-ONLY
    # --------------------------------------------------------
    if args.validate_only:
        print("\nValidation successful.")
        sys.exit(0)

    # --------------------------------------------------------
    # EDIT EXISTING ARCHITECTURE (helpers only; no template/hash updates)
    # --------------------------------------------------------
    if args.edit_existing:
        print("\nEdit existing architecture mode.")

        choice = input(
            "\nChoose:\n  1) Generate delta\n  2) Apply delta\n  3) Exit\nSelect: "
        ).strip()

        if choice == "1":
            base = input("Base arch dir (absolute or relative to repo root): ").strip()
            new = input("New  arch dir: ").strip()
            out = input("Output delta yaml: ").strip()
            sys.exit(run([PYTHON, GENERATE_DELTAS_SCRIPT, base, new, out]))

        if choice == "2":
            base = input("Base arch dir: ").strip()
            delta = input("Delta yaml: ").strip()
            out = input("Output dir: ").strip()

            rc = run([PYTHON, APPLY_DELTAS_SCRIPT, base, delta, out])
            if rc != 0:
                sys.exit(rc)

            # Re-verify after apply
            rc = run([PYTHON, VERIFY_SCRIPT, ANALYSIS_CONFIGS, TEMPLATE_FILE])
            sys.exit(rc)

        sys.exit(0)

    # --------------------------------------------------------
    # PROMOTE NEW LATEST ARCHITECTURE (mutating, with rollback)
    # --------------------------------------------------------
    if args.promote:
        new_latest = args.promote
        new_arch_dir = ANALYSIS_CONFIGS / new_latest

        if not new_arch_dir.is_dir():
            fatal(f"Architecture directory not found: {new_arch_dir}")

        if not confirm(
            f"Promote {new_latest} to latest? "
            "This will update template, regenerate deltas, and update hashes."
        ):
            sys.exit(0)

        # Back up the things we mutate
        backup_path = backup([ANALYSIS_CONFIGS, TEMPLATE_FILE, HASH_JSON])

        try:
            # 1) Update template
            if not PARSE_TEMPLATE_SCRIPT.exists():
                raise RuntimeError("parse_config_template.py not found")

            rc = run([
                PYTHON,
                PARSE_TEMPLATE_SCRIPT,
                new_arch_dir,
                TEMPLATE_FILE,
                "--latest-arch",
                new_latest,
            ])
            if rc != 0:
                raise RuntimeError("Failed to update template")

            # 2) Regenerate deltas for all other archs
            for arch_dir in sorted(ANALYSIS_CONFIGS.iterdir()):
                if not arch_dir.is_dir():
                    continue
                if arch_dir.name == new_latest:
                    continue

                delta_dir = arch_dir / "config_delta"
                delta_dir.mkdir(exist_ok=True)
                out_delta = delta_dir / f"{new_latest}_diff.yaml"

                rc = run([
                    PYTHON,
                    GENERATE_DELTAS_SCRIPT,
                    arch_dir,
                    new_arch_dir,
                    out_delta,
                ])
                if rc != 0:
                    raise RuntimeError(f"Delta generation failed for {arch_dir.name}")

                for f in delta_dir.glob("*_diff.yaml"):
                    if f.name != f"{new_latest}_diff.yaml":
                        f.unlink()

            # 3) Re-verify everything against updated template
            rc = run([PYTHON, VERIFY_SCRIPT, ANALYSIS_CONFIGS, TEMPLATE_FILE])
            if rc != 0:
                raise RuntimeError("Post-promotion verification failed")

            # 4) Now update the hash DB to the new steady state.
            #    Promotion touched many delta files, so compute-all is the safest.
            if not HASH_MANAGER_SCRIPT.exists():
                raise RuntimeError("hash_manager.py not found")

            rc = run([
                PYTHON,
                HASH_MANAGER_SCRIPT,
                "--compute-all",
                ANALYSIS_CONFIGS,
                HASH_JSON,
            ])
            if rc != 0:
                raise RuntimeError("Hash DB update failed (--compute-all)")

            # 5) run hash_checker
            rc = run([PYTHON, HASH_CHECKER_SCRIPT])
            if rc != 0:
                raise RuntimeError(
                    "Hash consistency check failed (after hash DB update)"
                )

            print(f"\nSUCCESS: {new_latest} promoted to latest.")
            sys.exit(0)

        except Exception as e:
            print(f"\nERROR: {e}")
            restore(backup_path, [ANALYSIS_CONFIGS, TEMPLATE_FILE, HASH_JSON])
            sys.exit(1)

    # --------------------------------------------------------
    # NO INTENT PROVIDED
    # --------------------------------------------------------
    print(
        "\nNo workflow selected.\n"
        "Use one of:\n"
        "  --validate-only\n"
        "  --edit-existing\n"
        "  --promote gfxXYZ\n"
        "  --hash-only / --ci\n"
    )
    sys.exit(0)


if __name__ == "__main__":
    main()
