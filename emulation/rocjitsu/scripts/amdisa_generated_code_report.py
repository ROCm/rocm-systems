#!/usr/bin/env python3
"""Generate a Markdown report for amdisa-generated source changes."""

from __future__ import annotations

import argparse
import datetime as dt
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence


ISA_SPECS = (
    ("cdna1", "amdgpu_isa_cdna1.xml"),
    ("cdna2", "amdgpu_isa_cdna2.xml"),
    ("cdna3", "amdgpu_isa_cdna3.xml"),
    ("cdna4", "amdgpu_isa_cdna4.xml"),
    ("rdna1", "amdgpu_isa_rdna1.xml"),
    ("rdna2", "amdgpu_isa_rdna2.xml"),
    ("rdna3", "amdgpu_isa_rdna3.xml"),
    ("rdna3_5", "amdgpu_isa_rdna3_5.xml"),
    ("rdna4", "amdgpu_isa_rdna4.xml"),
)

STATUS_LABELS = {
    "A": "added",
    "C": "copied",
    "D": "deleted",
    "M": "modified",
    "R": "renamed",
    "T": "type changed",
    "U": "unmerged",
    "??": "untracked",
}


def run_command(
    command: Sequence[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    capture: bool = False,
) -> str:
    kwargs: dict[str, object] = {
        "cwd": cwd,
        "env": env,
        "text": True,
        "check": True,
    }
    if capture:
        kwargs["stdout"] = subprocess.PIPE
        kwargs["stderr"] = subprocess.PIPE

    try:
        completed = subprocess.run(list(command), **kwargs)
    except subprocess.CalledProcessError as error:
        if capture:
            if error.stdout:
                print(error.stdout, file=sys.stderr, end="")
            if error.stderr:
                print(error.stderr, file=sys.stderr, end="")
        raise

    if capture:
        return completed.stdout
    return ""


def discover_repo_root() -> Path:
    script_root_guess = Path(__file__).resolve().parents[3]
    output = run_command(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=script_root_guess,
        capture=True,
    )
    return Path(output.strip()).resolve()


def relative_to_repo(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(path)


def shell_join(command: Sequence[str]) -> str:
    return shlex.join(str(part) for part in command)


def validate_mrisa_dir(mrisa_dir: Path) -> None:
    missing = [filename for _, filename in ISA_SPECS if not (mrisa_dir / filename).is_file()]
    if missing:
        formatted = "\n".join(f"  - {filename}" for filename in missing)
        raise SystemExit(f"Missing MR ISA XML files in {mrisa_dir}:\n{formatted}")


def create_output_dirs(generated_dir: Path) -> None:
    for arch, _ in ISA_SPECS:
        (generated_dir / arch).mkdir(parents=True, exist_ok=True)
    (generated_dir / "shared").mkdir(parents=True, exist_ok=True)


def run_direct_generator(
    *,
    rocjitsu_root: Path,
    mrisa_dir: Path,
    generated_dir: Path,
    python_executable: str,
) -> list[list[str]]:
    validate_mrisa_dir(mrisa_dir)
    create_output_dirs(generated_dir)
    env = os.environ.copy()
    python_path = str(rocjitsu_root / "lib" / "python")
    if env.get("PYTHONPATH"):
        python_path = python_path + os.pathsep + env["PYTHONPATH"]
    env["PYTHONPATH"] = python_path

    multi_args = [f"{name}:{mrisa_dir / filename}" for name, filename in ISA_SPECS]
    command = [
        python_executable,
        "-m",
        "amdisa",
        "--gen-all",
        "--gen-shared-execute",
        "-o",
        str(generated_dir),
        "--multi",
        *multi_args,
    ]
    run_command(command, cwd=rocjitsu_root, env=env)
    return [command]


def run_cmake_generator(
    *,
    repo_root: Path,
    rocjitsu_root: Path,
    mrisa_dir: Path,
    build_dir: Path,
    cmake_executable: str,
) -> list[list[str]]:
    validate_mrisa_dir(mrisa_dir)
    build_dir.mkdir(parents=True, exist_ok=True)
    stamp = build_dir / "generated" / "amdisa_amdgpu_isa_codegen.stamp"
    if stamp.exists():
        stamp.unlink()

    configure_command = [
        cmake_executable,
        "-S",
        str(rocjitsu_root),
        "-B",
        str(build_dir),
        "-DRJ_BUILD_GUI=OFF",
        "-DBUILD_TESTING=OFF",
        f"-DRJ_AMDGPU_ISA_DIR={mrisa_dir}",
        f"-DFETCHCONTENT_BASE_DIR={build_dir / '_deps'}",
    ]
    if shutil.which("ninja") or shutil.which("ninja-build"):
        configure_command.extend(["-G", "Ninja"])

    build_command = [
        cmake_executable,
        "--build",
        str(build_dir),
        "--target",
        "rocjitsu_generate_amdgpu_isa",
    ]

    run_command(configure_command, cwd=repo_root)
    run_command(build_command, cwd=repo_root)
    return [configure_command, build_command]


def run_command_for_status(command: Sequence[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def git_diff_no_index(repo_root: Path, args: Sequence[str]) -> str:
    completed = run_command_for_status(["git", "diff", "--no-index", *args], cwd=repo_root)
    if completed.returncode not in (0, 1):
        if completed.stdout:
            print(completed.stdout, file=sys.stderr, end="")
        if completed.stderr:
            print(completed.stderr, file=sys.stderr, end="")
        raise subprocess.CalledProcessError(
            completed.returncode,
            completed.args,
            output=completed.stdout,
            stderr=completed.stderr,
        )
    return completed.stdout


def rel_to_any(path: str, roots: Sequence[Path]) -> str:
    raw = Path(path)
    try:
        resolved = raw.resolve()
    except OSError:
        resolved = raw
    for root in roots:
        try:
            return resolved.relative_to(root.resolve()).as_posix()
        except ValueError:
            continue
    return raw.as_posix()


def parse_no_index_status(output: str, base_dir: Path, current_dir: Path) -> dict[str, str]:
    statuses: dict[str, str] = {}
    for line in output.splitlines():
        if not line:
            continue
        parts = line.split("\t")
        status = parts[0]
        if status.startswith("R") and len(parts) >= 3:
            path = rel_to_any(parts[2], (base_dir, current_dir))
        elif len(parts) >= 2:
            path = rel_to_any(parts[1], (base_dir, current_dir))
        else:
            continue
        statuses[path] = status[0]
    return statuses


def parse_no_index_numstat(output: str, base_dir: Path, current_dir: Path) -> dict[str, tuple[int | None, int | None]]:
    stats: dict[str, tuple[int | None, int | None]] = {}
    for line in output.splitlines():
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        added_raw, deleted_raw, raw_path = parts[0], parts[1], parts[-1]
        added = int(added_raw) if added_raw.isdigit() else None
        deleted = int(deleted_raw) if deleted_raw.isdigit() else None
        path = rel_to_any(raw_path, (base_dir, current_dir))
        stats[path] = (added, deleted)
    return stats


def build_ref_compare_report(
    *,
    repo_root: Path,
    rocjitsu_root: Path,
    mrisa_dir: Path,
    base_ref: str,
    python_executable: str,
    patch_lines: int,
) -> tuple[str, bool]:
    generated_at = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    commands: list[list[str]] = []

    with tempfile.TemporaryDirectory(prefix="amdisa-codegen-report-") as temp_dir_raw:
        temp_dir = Path(temp_dir_raw)
        base_worktree = temp_dir / "base-worktree"
        base_generated = temp_dir / "base-generated"
        current_generated = temp_dir / "current-generated"

        run_command(["git", "worktree", "add", "--detach", str(base_worktree), base_ref], cwd=repo_root)
        try:
            rocjitsu_rel = rocjitsu_root.resolve().relative_to(repo_root.resolve())
            base_rocjitsu_root = base_worktree / rocjitsu_rel
            try:
                mrisa_rel = mrisa_dir.resolve().relative_to(repo_root.resolve())
                base_mrisa_dir = base_worktree / mrisa_rel
                mrisa_display = mrisa_rel.as_posix()
            except ValueError:
                base_mrisa_dir = mrisa_dir
                mrisa_display = str(mrisa_dir)

            commands.extend(run_direct_generator(
                rocjitsu_root=base_rocjitsu_root,
                mrisa_dir=base_mrisa_dir,
                generated_dir=base_generated,
                python_executable=python_executable,
            ))
            commands.extend(run_direct_generator(
                rocjitsu_root=rocjitsu_root,
                mrisa_dir=mrisa_dir,
                generated_dir=current_generated,
                python_executable=python_executable,
            ))

            status_output = git_diff_no_index(
                repo_root,
                ["--name-status", str(base_generated), str(current_generated)],
            )
            numstat_output = git_diff_no_index(
                repo_root,
                ["--numstat", str(base_generated), str(current_generated)],
            )
            diffstat = git_diff_no_index(
                repo_root,
                ["--stat", str(base_generated), str(current_generated)],
            ).strip()
            patch = git_diff_no_index(
                repo_root,
                ["--color=never", str(base_generated), str(current_generated)],
            )
        finally:
            run_command(["git", "worktree", "remove", "--force", str(base_worktree)], cwd=repo_root)

    statuses = parse_no_index_status(status_output, base_generated, current_generated)
    numstats = parse_no_index_numstat(numstat_output, base_generated, current_generated)
    changed_paths = sorted(set(statuses) | set(numstats))

    total_added = 0
    total_deleted = 0
    rows: list[tuple[str, str, int | None, int | None]] = []
    for path in changed_paths:
        added, deleted = numstats.get(path, (None, None))
        if isinstance(added, int):
            total_added += added
        if isinstance(deleted, int):
            total_deleted += deleted
        rows.append((path, statuses.get(path, "M"), added, deleted))

    report: list[str] = [
        "# amdisa Generated Code Report",
        "",
        f"Generated at: {generated_at}",
        f"Base ref: `{base_ref}`",
        f"MR ISA directory: `{mrisa_display}`",
        "Comparison mode: generated output from base ref vs generated output from this checkout",
        "",
        "## Commands",
        "",
    ]
    for command in commands:
        report.append(f"```bash\n{shell_join(command)}\n```")
    report.append("")

    if not changed_paths:
        report.extend([
            "## Summary",
            "",
            "No generated AMDGPU ISA source changes were detected.",
            "",
        ])
        return "\n".join(report), False

    report.extend([
        "## Summary",
        "",
        f"Files changed: {len(changed_paths)}",
        f"Lines added: {total_added}",
        f"Lines deleted: {total_deleted}",
        "",
    ])
    if diffstat:
        report.extend(["## Diffstat", "", "```text", diffstat, "```", ""])

    report.extend([
        "## Changed Files",
        "",
        "| Status | Additions | Deletions | Path |",
        "|---|---:|---:|---|",
    ])
    for path, status, added, deleted in rows:
        added_text = str(added) if added is not None else "-"
        deleted_text = str(deleted) if deleted is not None else "-"
        report.append(
            f"| {status_label(status)} | {added_text} | {deleted_text} | "
            f"`{markdown_escape_table(path)}` |"
        )
    report.append("")

    report.extend(["## Patch Preview", "", "```diff"])
    report.append(truncate_lines(patch, patch_lines).rstrip() if patch else "No textual diff available.")
    report.extend(["```", ""])
    return "\n".join(report), True


def git_capture(repo_root: Path, args: Sequence[str]) -> str:
    return run_command(["git", *args], cwd=repo_root, capture=True)


def parse_status(repo_root: Path, generated_rel: str) -> dict[str, str]:
    output = git_capture(repo_root, ["status", "--porcelain=v1", "--", generated_rel])
    statuses: dict[str, str] = {}
    for line in output.splitlines():
        if not line:
            continue
        raw_status = line[:2]
        raw_path = line[3:]
        path = raw_path.split(" -> ")[-1]
        status = "??" if raw_status == "??" else raw_status.strip()
        statuses[path] = status
    return statuses


def parse_numstat(repo_root: Path, generated_rel: str) -> dict[str, tuple[int | None, int | None]]:
    output = git_capture(repo_root, ["diff", "--numstat", "HEAD", "--", generated_rel])
    stats: dict[str, tuple[int | None, int | None]] = {}
    for line in output.splitlines():
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        added_raw, deleted_raw, path = parts[0], parts[1], parts[-1]
        added = int(added_raw) if added_raw.isdigit() else None
        deleted = int(deleted_raw) if deleted_raw.isdigit() else None
        stats[path] = (added, deleted)
    return stats


def count_untracked_lines(repo_root: Path, path: str) -> int | None:
    file_path = repo_root / path
    if not file_path.is_file():
        return None
    try:
        return len(file_path.read_text(errors="replace").splitlines())
    except OSError:
        return None


def status_label(status: str) -> str:
    if status in STATUS_LABELS:
        return STATUS_LABELS[status]
    labels = [STATUS_LABELS.get(character, character) for character in status]
    return ", ".join(labels)


def truncate_lines(text: str, limit: int) -> str:
    if limit <= 0:
        return text
    lines = text.splitlines()
    if len(lines) <= limit:
        return text
    kept = lines[:limit]
    kept.append(f"... truncated {len(lines) - limit} lines ...")
    return "\n".join(kept)


def read_untracked_preview(repo_root: Path, path: str, limit: int) -> str:
    file_path = repo_root / path
    try:
        lines = file_path.read_text(errors="replace").splitlines()
    except OSError as error:
        return f"Unable to read {path}: {error}"
    if limit > 0 and len(lines) > limit:
        lines = lines[:limit] + [f"... truncated {len(lines) - limit} lines ..."]
    return "\n".join(f"+{line}" for line in lines)


def markdown_escape_table(text: str) -> str:
    return text.replace("|", "\\|")


def build_report(
    *,
    repo_root: Path,
    mrisa_dir: Path,
    generated_dir: Path,
    commands: list[list[str]],
    generator_mode: str,
    generation_skipped: bool,
    patch_lines: int,
) -> tuple[str, bool]:
    generated_rel = relative_to_repo(generated_dir, repo_root)
    mrisa_rel = relative_to_repo(mrisa_dir, repo_root)
    statuses = parse_status(repo_root, generated_rel)
    numstats = parse_numstat(repo_root, generated_rel)

    changed_paths = sorted(set(statuses) | set(numstats))
    rows: list[tuple[str, str, int | None, int | None]] = []
    total_added = 0
    total_deleted = 0
    for path in changed_paths:
        status = statuses.get(path, "M")
        added, deleted = numstats.get(path, (None, None))
        if status == "??" and added is None:
            added = count_untracked_lines(repo_root, path)
            deleted = 0
        if isinstance(added, int):
            total_added += added
        if isinstance(deleted, int):
            total_deleted += deleted
        rows.append((path, status, added, deleted))

    generated_at = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    report: list[str] = [
        "# amdisa Generated Code Report",
        "",
        f"Generated at: {generated_at}",
        f"Generator mode: `{generator_mode}`",
        f"Generation step: {'skipped' if generation_skipped else 'ran'}",
        f"MR ISA directory: `{mrisa_rel}`",
        f"Generated source directory: `{generated_rel}`",
        "",
    ]

    if commands:
        report.append("## Commands")
        report.append("")
        for command in commands:
            report.append(f"```bash\n{shell_join(command)}\n```")
        report.append("")

    if not changed_paths:
        report.append("## Summary")
        report.append("")
        report.append("No generated AMDGPU ISA source changes were detected.")
        report.append("")
        return "\n".join(report), False

    report.extend([
        "## Summary",
        "",
        f"Files changed: {len(changed_paths)}",
        f"Lines added: {total_added}",
        f"Lines deleted: {total_deleted}",
        "",
    ])

    diffstat = git_capture(repo_root, ["diff", "--stat", "HEAD", "--", generated_rel]).strip()
    if diffstat:
        report.extend(["## Diffstat", "", "```text", diffstat, "```", ""])

    report.extend([
        "## Changed Files",
        "",
        "| Status | Additions | Deletions | Path |",
        "|---|---:|---:|---|",
    ])
    for path, status, added, deleted in rows:
        added_text = str(added) if added is not None else "-"
        deleted_text = str(deleted) if deleted is not None else "-"
        report.append(
            f"| {status_label(status)} | {added_text} | {deleted_text} | "
            f"`{markdown_escape_table(path)}` |"
        )
    report.append("")

    report.extend(["## Patch Preview", ""])
    for path, status, added, deleted in rows:
        added_text = str(added) if added is not None else "-"
        deleted_text = str(deleted) if deleted is not None else "-"
        report.append(
            f"<details><summary>{markdown_escape_table(path)} "
            f"(+{added_text}/-{deleted_text})</summary>"
        )
        report.append("")
        if status == "??":
            preview = read_untracked_preview(repo_root, path, patch_lines)
        else:
            preview = git_capture(repo_root, ["diff", "--color=never", "HEAD", "--", path])
            preview = truncate_lines(preview, patch_lines)
        report.append("```diff")
        report.append(preview.rstrip() if preview else "No textual diff available.")
        report.append("```")
        report.append("")
        report.append("</details>")
        report.append("")

    return "\n".join(report), True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run AMDGPU ISA codegen and report generated source changes."
    )
    parser.add_argument("--repo-root", type=Path, help="Repository root. Defaults to git root.")
    parser.add_argument(
        "--rocjitsu-root",
        type=Path,
        help="rocjitsu project root. Defaults to emulation/rocjitsu under the repo root.",
    )
    parser.add_argument(
        "--mrisa-dir",
        type=Path,
        help="Directory containing amdgpu_isa_*.xml. Defaults to shared/machine-readable-isa/isa.",
    )
    parser.add_argument(
        "--generated-dir",
        type=Path,
        help="Generated AMDGPU ISA source directory. Defaults to the checked-in rocjitsu source tree.",
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        help="CMake build directory used with --generator cmake.",
    )
    parser.add_argument(
        "--generator",
        choices=("cmake", "direct"),
        default="cmake",
        help="How to run codegen before reporting. Defaults to the CMake build target.",
    )
    parser.add_argument(
        "--skip-generate",
        action="store_true",
        help="Only report current generated source changes; do not run codegen first.",
    )
    parser.add_argument(
        "--base-ref",
        help="Generate both this checkout and the given git ref, then compare their generated output.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable for --generator direct.",
    )
    parser.add_argument(
        "--cmake",
        default=shutil.which("cmake") or "cmake",
        help="CMake executable for --generator cmake.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write Markdown report to this file. Defaults to stdout.",
    )
    parser.add_argument(
        "--patch-lines",
        type=int,
        default=300,
        help="Maximum diff lines per file in the patch preview. Use 0 for no limit.",
    )
    parser.add_argument(
        "--fail-on-changes",
        action="store_true",
        help="Exit with status 2 when generated source changes are detected.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve() if args.repo_root else discover_repo_root()
    rocjitsu_root = (
        args.rocjitsu_root.resolve()
        if args.rocjitsu_root
        else repo_root / "emulation" / "rocjitsu"
    )
    mrisa_dir = (
        args.mrisa_dir.resolve()
        if args.mrisa_dir
        else repo_root / "shared" / "machine-readable-isa" / "isa"
    )
    build_dir = (
        args.build_dir.resolve()
        if args.build_dir
        else rocjitsu_root / "build" / "amdisa-codegen-report"
    )
    generated_dir = (
        args.generated_dir.resolve()
        if args.generated_dir
        else rocjitsu_root / "lib" / "rocjitsu" / "src" / "rocjitsu" / "isa" / "arch" / "amdgpu"
    )

    if args.base_ref:
        if args.skip_generate:
            raise SystemExit("--base-ref cannot be combined with --skip-generate")
        report, has_changes = build_ref_compare_report(
            repo_root=repo_root,
            rocjitsu_root=rocjitsu_root,
            mrisa_dir=mrisa_dir,
            base_ref=args.base_ref,
            python_executable=args.python,
            patch_lines=args.patch_lines,
        )
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(report + "\n")
            print(f"Wrote {args.output}")
        else:
            print(report)
        if has_changes:
            print("Generated AMDGPU ISA source changes were detected.")
        else:
            print("No generated AMDGPU ISA source changes detected.")
        if has_changes and args.fail_on_changes:
            return 2
        return 0

    commands: list[list[str]] = []
    if not args.skip_generate:
        if args.generator == "cmake":
            commands = run_cmake_generator(
                repo_root=repo_root,
                rocjitsu_root=rocjitsu_root,
                mrisa_dir=mrisa_dir,
                build_dir=build_dir,
                cmake_executable=args.cmake,
            )
        else:
            commands = run_direct_generator(
                rocjitsu_root=rocjitsu_root,
                mrisa_dir=mrisa_dir,
                generated_dir=generated_dir,
                python_executable=args.python,
            )

    report, has_changes = build_report(
        repo_root=repo_root,
        mrisa_dir=mrisa_dir,
        generated_dir=generated_dir,
        commands=commands,
        generator_mode=args.generator,
        generation_skipped=args.skip_generate,
        patch_lines=args.patch_lines,
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n")
        print(f"Wrote {args.output}")
    else:
        print(report)

    if has_changes:
        print("Generated AMDGPU ISA source changes were detected.")
    else:
        print("No generated AMDGPU ISA source changes detected.")

    if has_changes and args.fail_on_changes:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
