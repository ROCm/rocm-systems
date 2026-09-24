"""Shared utilities for RCCL CI test scripts (JAX, PyTorch, MADEngine)."""

import json
import logging
import os
import re
import smtplib
import sys
import urllib.request
import xml.etree.ElementTree as ET
from email.mime.text import MIMEText
from pathlib import Path

log = logging.getLogger(__name__)

SMTP_SERVERS = ["smtp.amd.com", "aussmtp.amd.com", "mail.amd.com", "localhost"]


def find_pip_sdk_lib_dirs() -> list[Path]:
    """Return library directories from pip-installed ROCm SDK packages."""
    site = Path(sys.prefix) / "lib"
    dirs = sorted(d for d in site.rglob("_rocm_sdk_*/lib") if d.is_dir())
    if not dirs:
        dirs = sorted(d for d in Path(sys.prefix).rglob("_rocm_sdk_*/lib") if d.is_dir())
    return dirs


def override_bundled_rccl(rccl_lib_dir: Path) -> None:
    """Replace the pip-bundled librccl.so with the CI-built version.

    LD_LIBRARY_PATH alone is insufficient because pip-installed wheels
    embed DT_RPATH/DT_RUNPATH in their shared libraries, and the dynamic
    linker resolves these before LD_LIBRARY_PATH.  LD_PRELOAD causes
    LLVM symbol conflicts with PyTorch.  Instead, we physically replace
    the bundled library files so the wheel's own load path resolves to
    the CI-built version.
    """
    import shutil

    ci_rccl = rccl_lib_dir.resolve() / "librccl.so"
    if not ci_rccl.exists():
        raise FileNotFoundError(
            f"CI-built librccl.so not found at {ci_rccl} — check artifact fetch"
        )
    ci_size = ci_rccl.stat().st_size
    log.info("CI-built RCCL: %s (%d bytes)", ci_rccl, ci_size)

    bundled = []
    for d in find_pip_sdk_lib_dirs():
        bundled.extend(
            f for f in d.glob("librccl.so*") if ".pip-original" not in f.name
        )
    if not bundled:
        bundled = [
            f for f in Path(sys.prefix).rglob("librccl.so*")
            if ".pip-original" not in f.name
        ]

    replaced = 0
    for target in bundled:
        if target.is_symlink():
            continue
        backup = target.with_suffix(target.suffix + ".pip-original")
        if not backup.exists():
            shutil.copy2(str(target), str(backup))
            log.info("Backed up original: %s", backup)
        shutil.copy2(str(ci_rccl), str(target))
        log.info("Replaced %s (%d bytes)", target, target.stat().st_size)
        replaced += 1

    if replaced == 0:
        log.error("No bundled librccl.so found to replace — tests would run against pip RCCL")
        sys.exit(1)
    else:
        log.info("Replaced %d bundled librccl.so file(s) with CI-built version", replaced)


def override_bundled_hip_runtime(artifact_lib_dir: Path) -> None:
    """Replace pip-bundled libamdhip64.so with TheRock's version.

    The CI-built librccl.so contains device code objects compiled by
    TheRock's LLVM/HIP compiler.  At test time, the pip-installed HIP
    runtime (libamdhip64.so) parses these code objects.  If the two
    versions differ, kernel symbol resolution fails inside
    Function::BuildKernel → guarantee(symbol != nullptr) → abort().

    Replacing the pip HIP runtime with TheRock's version ensures the
    runtime that loads code objects matches the compiler that produced
    them.  TheRock's libamdhip64.so is already present in the artifact
    directory from the core-hip_lib fetch.
    """
    import shutil

    ci_hip = artifact_lib_dir.resolve() / "libamdhip64.so"
    if not ci_hip.exists():
        log.warning(
            "TheRock libamdhip64.so not found at %s — skipping HIP runtime override",
            ci_hip,
        )
        return
    ci_size = ci_hip.stat().st_size
    log.info("TheRock HIP runtime: %s (%d bytes)", ci_hip, ci_size)

    bundled = []
    for d in find_pip_sdk_lib_dirs():
        bundled.extend(
            f for f in d.glob("libamdhip64.so*") if ".pip-original" not in f.name
        )
    if not bundled:
        bundled = [
            f for f in Path(sys.prefix).rglob("libamdhip64.so*")
            if ".pip-original" not in f.name
        ]

    replaced = 0
    for target in bundled:
        if target.is_symlink():
            continue
        backup = target.with_suffix(target.suffix + ".pip-original")
        if not backup.exists():
            shutil.copy2(str(target), str(backup))
            log.info("Backed up original: %s", backup)
        shutil.copy2(str(ci_hip), str(target))
        log.info("Replaced %s (%d bytes)", target, target.stat().st_size)
        replaced += 1

    if replaced == 0:
        log.warning(
            "No bundled libamdhip64.so found — HIP runtime override skipped"
        )
    else:
        log.info(
            "Replaced %d bundled libamdhip64.so file(s) with TheRock version",
            replaced,
        )


def setup_kpack_device_code(artifact_dir: Path) -> None:
    """Replace pip-bundled kpack archives with CI-built versions.

    TheRock's kpack pipeline strips device code from shared libraries into
    separate .kpack archive files.  override_bundled_rccl() replaces pip's
    librccl.so with the CI-built version; the kpack archives must also be
    replaced so the embedded HIPK search paths resolve to CI-built device
    code.

    We copy CI-built .kpack archives over the matching pip-bundled ones
    rather than using ROCM_KPACK_PATH or ROCM_KPACK_PATH_PREFIX.  Both
    env vars inject CI kpack archives into every library's search list,
    and the kpack loader stops at the first architecture-matching archive
    (even if the kernel belongs to a different archive).  This causes
    hipErrorInvalidImage when RCCL archives shadow PyTorch's.

    By replacing in-place, each library's embedded search paths resolve
    to the correct archive without cross-contamination.
    """
    import shutil

    ci_kpacks = sorted(f for f in artifact_dir.rglob("*.kpack") if f.is_file())
    if not ci_kpacks:
        log.info("No .kpack files in %s — device code may be embedded", artifact_dir)
        return

    for f in ci_kpacks:
        log.info("CI kpack archive: %s (%d bytes)", f.name, f.stat().st_size)

    pip_kpack_dirs = _find_pip_kpack_dirs()
    if not pip_kpack_dirs:
        log.warning("No pip .kpack directories found — cannot replace kpack archives")
        return

    replaced = 0
    for ci_kpack in ci_kpacks:
        for pip_dir in pip_kpack_dirs:
            target = pip_dir / ci_kpack.name
            if not target.exists():
                continue
            backup = target.with_suffix(target.suffix + ".pip-original")
            if not backup.exists():
                shutil.copy2(str(target), str(backup))
                log.info("Backed up original: %s", backup)
            shutil.copy2(str(ci_kpack), str(target))
            log.info("Replaced kpack: %s (%d bytes)", target, ci_kpack.stat().st_size)
            replaced += 1

    if replaced > 0:
        log.info("Replaced %d kpack archive(s) with CI-built versions", replaced)
    else:
        log.info("No matching pip kpack archives found to replace")



def _find_pip_kpack_dirs() -> list[Path]:
    """Return .kpack directories from pip-installed ROCm/torch packages."""
    dirs: list[Path] = []
    site_lib = Path(sys.prefix) / "lib"
    for d in sorted(site_lib.rglob(".kpack")):
        if d.is_dir():
            dirs.append(d)
    if not dirs:
        for d in sorted(Path(sys.prefix).rglob(".kpack")):
            if d.is_dir():
                dirs.append(d)
    return dirs


def quarantine_rocm_sysdeps(artifact_lib_dir: Path) -> None:
    """Remove TheRock-bundled libamd_smi and rocm_sysdeps from artifact dir.

    The CI-built librccl.so retains RPATH from the TheRock build tree.
    When resolved, it loads libamd_smi.so from the artifact directory,
    which transitively loads librocm_sysdeps_nl_genl_3.so.200 via its
    own RUNPATH ($ORIGIN/rocm_sysdeps/lib/).  That library's destructor
    calls genl_unregister_family and crashes (SIGSEGV) in containers.

    Quarantining both libamd_smi and rocm_sysdeps forces the linker to
    fall through the RPATH miss and resolve libamd_smi from the pip
    environment instead.  reconcile_soname_versions() must run BEFORE
    this function so that pip dirs already have compatibility symlinks.
    """
    sysdeps_dir = artifact_lib_dir / "rocm_sysdeps"
    if sysdeps_dir.is_dir():
        sysdeps_lib = sysdeps_dir / "lib"
        if sysdeps_lib.is_dir():
            for f in sorted(sysdeps_lib.glob("librocm_sysdeps_nl_genl*")):
                q = f.parent / (f.name + ".quarantined")
                f.rename(q)
                log.info("Quarantined: %s", f)
        else:
            quarantined = artifact_lib_dir / "rocm_sysdeps.quarantined"
            sysdeps_dir.rename(quarantined)
            log.info("Quarantined: %s -> %s", sysdeps_dir, quarantined)

    for f in sorted(artifact_lib_dir.glob("libamd_smi*")):
        q = f.parent / (f.name + ".quarantined")
        f.rename(q)
        log.info("Quarantined: %s", f.name)


_SONAME_RE = re.compile(r"^(lib.+\.so)\.(\d+)$")


def reconcile_soname_versions(lib_dirs: list[Path]) -> None:
    """Create symlinks to resolve soname version mismatches across lib dirs.

    CI-built and pip-installed libraries may ship different soname versions
    of the same library (e.g., CI artifacts have libamd_smi.so.26 while the
    pip wheel ships libamd_smi.so.27).  This causes dlopen failures when one
    side references the other's version — whether via ELF DT_NEEDED or
    Python-level ctypes.CDLL preloading (rocm_sdk.preload_libraries).

    For every library base name (e.g. libamd_smi.so) that appears with
    different version suffixes across the directories, we create symlinks
    in each directory so that every observed version resolves everywhere.
    """
    per_dir: dict[Path, dict[str, dict[str, Path]]] = {}
    all_versions: dict[str, set[str]] = {}

    for d in lib_dirs:
        if not d.is_dir():
            continue
        bases: dict[str, dict[str, Path]] = {}
        for f in d.iterdir():
            if f.name.endswith(".pip-original"):
                continue
            m = _SONAME_RE.match(f.name)
            if m:
                base, ver = m.group(1), m.group(2)
                bases.setdefault(base, {})[ver] = f
                all_versions.setdefault(base, set()).add(ver)
        per_dir[d] = bases

    created = 0
    for base, versions in sorted(all_versions.items()):
        if len(versions) < 2:
            continue
        for d, bases in per_dir.items():
            dir_versions = bases.get(base, {})
            if not dir_versions:
                continue
            target_ver = max(dir_versions)
            target_name = f"{base}.{target_ver}"
            for ver in versions:
                if ver in dir_versions:
                    continue
                symlink = d / f"{base}.{ver}"
                if symlink.exists() or symlink.is_symlink():
                    continue
                symlink.symlink_to(target_name)
                log.info("Symlink: %s.%s -> %s (in %s)", base, ver, target_name, d)
                created += 1

    if created:
        log.info("Created %d compatibility symlink(s) for soname mismatches", created)
    else:
        log.info("No soname version mismatches found")


def find_rccl_library(artifact_dir: Path) -> Path:
    """Find librccl.so in the artifact directory tree."""
    matches = list(artifact_dir.rglob("librccl.so"))
    if not matches:
        so_files = list(artifact_dir.rglob("*.so"))[:20]
        log.error("librccl.so not found in %s", artifact_dir)
        log.error("Shared libraries found: %s", [str(f) for f in so_files])
        sys.exit(1)
    lib_path = matches[0].resolve()
    log.info("Found librccl.so at: %s", lib_path)
    return lib_path


def verify_rccl_override(rccl_lib_dir: Path) -> None:
    """Verify that the CI-built librccl.so exists on disk."""
    ci_rccl = rccl_lib_dir.resolve() / "librccl.so"
    if not ci_rccl.exists():
        log.error("CI-built librccl.so not found at %s", ci_rccl)
        sys.exit(1)
    log.info("CI-built RCCL: %s (%d bytes)", ci_rccl, ci_rccl.stat().st_size)


def parse_junit_xml(xml_path: Path) -> dict:
    """Parse JUnit XML and return structured results."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    passed_tests = []
    failed_tests = []
    skipped_tests = []
    error_details = []
    tests_run = 0
    failures = 0
    errors = 0

    for suite in root.iter("testsuite"):
        tests_run += int(suite.get("tests", 0))
        failures += int(suite.get("failures", 0))
        errors += int(suite.get("errors", 0))

    for tc in root.iter("testcase"):
        name = tc.get("name", "")
        time_s = tc.get("time", "")
        duration = f"{float(time_s):.2f}s" if time_s else ""

        failure = tc.find("failure")
        error = tc.find("error")
        skipped = tc.find("skipped")
        if failure is not None:
            failed_tests.append(name)
            error_details.append(
                f"FAILED: {name}\n  {failure.get('message', '')}"
            )
        elif error is not None:
            failed_tests.append(name)
            error_details.append(
                f"ERROR: {name}\n  {error.get('message', '')}"
            )
        elif skipped is not None:
            skipped_tests.append(name)
        else:
            passed_tests.append((name, duration))

    return {
        "passed": passed_tests,
        "failed": failed_tests,
        "skipped": skipped_tests,
        "error_details": error_details,
        "tests_run": tests_run,
        "failures": failures,
        "errors": errors,
    }


def write_github_summary(report: str) -> None:
    """Write report to GITHUB_STEP_SUMMARY if available."""
    summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_file:
        with open(summary_file, "a") as f:
            f.write("```\n")
            f.write(report)
            f.write("\n```\n")
        log.info("Summary written to GITHUB_STEP_SUMMARY")


def set_github_output(key: str, value: str) -> None:
    """Write a key=value pair to GITHUB_OUTPUT if available."""
    output_file = os.environ.get("GITHUB_OUTPUT")
    if output_file:
        with open(output_file, "a") as f:
            f.write(f"{key}={value}\n")


def send_email_report(
    report: str, recipient: str, status: str, subject_prefix: str
) -> None:
    """Send the summary report via email."""
    subject = f"{subject_prefix}: {status}"
    msg = MIMEText(report)
    msg["Subject"] = subject
    msg["From"] = "rccl-ci@amd.com"
    msg["To"] = recipient

    for server in SMTP_SERVERS:
        try:
            with smtplib.SMTP(server, timeout=10) as s:
                s.sendmail(msg["From"], [recipient], msg.as_string())
            log.info("Email sent to %s via %s", recipient, server)
            return
        except Exception as e:
            log.debug("SMTP %s failed: %s", server, e)
            continue
    log.warning(
        "Could not send email to %s (tried: %s)", recipient, ", ".join(SMTP_SERVERS)
    )


def send_teams_webhook(
    report: str, webhook_url: str, status: str, subject_prefix: str
) -> None:
    """Send the summary report to a Microsoft Teams channel via webhook."""
    color = "Good" if status == "PASSED" else "Attention"
    run_url = os.environ.get("GITHUB_SERVER_URL", "")
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    run_id = os.environ.get("GITHUB_RUN_ID", "")
    actions_url = f"{run_url}/{repo}/actions/runs/{run_id}" if run_url else ""

    facts = [{"title": "Status", "value": status}]
    if actions_url:
        facts.append({"title": "Run", "value": f"[View]({actions_url})"})

    body = [
        {
            "type": "TextBlock",
            "text": f"{subject_prefix}: {status}",
            "weight": "Bolder",
            "size": "Medium",
            "color": color,
        },
        {"type": "FactSet", "facts": facts},
        {
            "type": "TextBlock",
            "text": report,
            "wrap": True,
            "fontType": "Monospace",
            "size": "Small",
        },
    ]

    payload = {
        "type": "message",
        "attachments": [
            {
                "contentType": "application/vnd.microsoft.card.adaptive",
                "content": {
                    "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
                    "type": "AdaptiveCard",
                    "version": "1.4",
                    "body": body,
                },
            }
        ],
    }

    try:
        req = urllib.request.Request(
            webhook_url,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            log.info("Teams webhook sent (HTTP %d)", resp.status)
    except Exception as e:
        log.warning("Failed to send Teams webhook: %s", e)
