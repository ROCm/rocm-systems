"""Stage HIP sources for the native Windows build without changing the source tree."""

import argparse
import concurrent.futures
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile


SOURCE_SUFFIXES = {".h", ".hpp", ".cuh", ".cc", ".cpp", ".cu", ".c"}
COPY_UNCHANGED = {
    "src/include/nccl_device/hip_compat.h",
    "src/include/proxy_trace/proxy_trace.h",
}
STATE_FILE = ".rccl-stage-state.json"


class StagingError(RuntimeError):
    pass


@dataclass(frozen=True)
class Tools:
    perl: Path
    bash: Path
    hipify: Path


def file_digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_tool(override, names, candidates=()):
    """Honor an override, then PATH, then installation-derived candidates."""
    if override:
        found = shutil.which(str(override))
        path = Path(found) if found else Path(override)
        if path.is_file():
            return path.resolve()
        raise StagingError(f"Requested tool does not exist: {override}")
    for name in names:
        found = shutil.which(name)
        if found:
            path = Path(found).resolve()
            # Windows' bash.exe launchers start WSL, not a native shell.
            if path.as_posix().lower().endswith(("/windows/system32/bash.exe",
                                                "/windows/sysnative/bash.exe",
                                                "/microsoft/windowsapps/bash.exe")):
                continue
            return path
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise StagingError(f"Cannot find {names[0]}; put it on PATH or set its RCCL tool override.")


def git_tool_directories():
    directories = []
    git = shutil.which("git") or shutil.which("git.exe")
    if git:
        for parent in Path(git).resolve().parents:
            directories.extend((parent / "usr/bin", parent / "bin"))
    for variable in ("ProgramFiles", "ProgramFiles(x86)", "LOCALAPPDATA"):
        if os.environ.get(variable):
            root = Path(os.environ[variable])
            directories.extend((root / "Git/usr/bin", root / "Programs/Git/usr/bin"))
    return list(dict.fromkeys(directories))


def discover_tools(sdk, perl=None, bash=None, hipify=None):
    git_directories = git_tool_directories()
    return Tools(
        resolve_tool(perl or os.environ.get("RCCL_PERL"), ("perl", "perl.exe"),
                     [p / "perl.exe" for p in git_directories]),
        resolve_tool(bash or os.environ.get("RCCL_BASH"), ("bash", "bash.exe"),
                     [p / "bash.exe" for p in git_directories]),
        resolve_tool(hipify or os.environ.get("RCCL_HIPIFY_PERL"), ("hipify-perl",),
                     [sdk / "libexec/hipify/hipify-perl", sdk / "bin/hipify-perl"]),
    )


def run(command, environment):
    result = subprocess.run([str(arg) for arg in command], env=environment,
                            capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode:
        detail = "\n".join(part.strip() for part in (result.stdout, result.stderr) if part.strip())
        raise StagingError(f"Command failed ({result.returncode}): {command[0]}\n{detail}")


def normalize_collision_includes(content, original, source):
    """Undo only nonexistent flat-build aliases whose original header exists."""
    search_roots = [original.parent, source / "src/include", source / "src",
                    source / "src/device", source / "src/include/nccl_device",
                    source / "src/transport", source / "src/devcomm"]

    def replace(match):
        name = match[2]
        if any((root / name).is_file() for root in search_roots):
            return match[0]
        canonical = name[:-6] + ".h"
        if any((root / canonical).is_file() for root in search_roots):
            return match[1] + canonical + match[3]
        return match[0]

    return re.sub(r'(#include\s*[<"])([^">\r\n]+_tmp\.h)([">])', replace, content)


def atomic_write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(data)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def managed_output(destination, relative):
    path = (destination / relative).resolve()
    try:
        path.relative_to(destination.resolve())
    except ValueError:
        raise StagingError(f"Invalid staged source path in state file: {relative}") from None
    if not relative.startswith("src/"):
        raise StagingError(f"Invalid staged source path in state file: {relative}")
    return path


def depfile_escape(path):
    return str(path).replace("\\", "/").replace("$", "$$").replace("#", "\\#").replace(" ", "\\ ").replace(":", "\\:")


def stage_sources(source, destination, sdk, tools, *, fault_injection=False, jobs=8,
                  stamp=None, depfile=None, depfile_target=None):
    if depfile and not depfile_target and not stamp:
        raise StagingError("--depfile requires --depfile-target or --stamp")
    source, destination, sdk = (Path(p).resolve() for p in (source, destination, sdk))
    paths = sorted(p for p in (source / "src").rglob("*") if p.is_file() and p.suffix in SOURCE_SUFFIXES)
    if not paths:
        raise StagingError(f"No RCCL source files found under {source / 'src'}")
    transform = source / "cmake/scripts/add_unroll.sh"
    dependencies = [Path(__file__).resolve(), tools.perl, tools.bash, tools.hipify, transform]
    if fault_injection:
        dependencies.append(source / "cmake/scripts/add_faults.sh")
    dependencies.extend(p for p in (sdk / ".info/version", sdk / "bin/.hipVersion") if p.is_file())
    for path in dependencies:
        if not path.is_file():
            raise StagingError(f"Required staging input is missing: {path}")
    relative_names = [p.relative_to(source).as_posix() for p in paths]
    configuration = json.dumps({
        "sdk": str(sdk), "fault_injection": fault_injection, "sources": relative_names,
        "inputs": [(str(p), file_digest(p)) for p in dependencies],
    }, sort_keys=True)
    configuration = hashlib.sha256(configuration.encode()).hexdigest()
    state_path = destination / STATE_FILE
    previous = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {}
    old_files = previous.get("files", {})
    reusable = previous.get("configuration") == configuration
    environment = dict(os.environ)
    environment["PATH"] = os.pathsep.join(dict.fromkeys(
        [str(tools.perl.parent), str(tools.bash.parent)] + environment.get("PATH", "").split(os.pathsep)))

    def stage(path):
        relative = path.relative_to(source).as_posix()
        output = managed_output(destination, relative)
        digest = file_digest(path)
        old = old_files.get(relative, {})
        if reusable and old.get("source") == digest and output.is_file() and old.get("output") == file_digest(output):
            return relative, old, False
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=output.parent, prefix=f".{output.stem}.", suffix=output.suffix, delete=False) as stream:
            temporary = Path(stream.name)
        try:
            if relative in COPY_UNCHANGED:
                temporary.write_bytes(path.read_bytes())
            else:
                run([tools.perl, tools.hipify, "-experimental", "-quiet-warnings", path, "-o", temporary], environment)
                content = temporary.read_text(encoding="utf-8")
                temporary.write_text(normalize_collision_includes(content, path, source), encoding="utf-8")
                if re.search(r"/src/device/.*\.h", output.as_posix()):
                    run([tools.bash, transform, temporary.as_posix()], environment)
                if fault_injection:
                    run([tools.bash, source / "cmake/scripts/add_faults.sh", temporary.as_posix()], environment)
            output_digest = file_digest(temporary)
            if not output.is_file() or file_digest(output) != output_digest:
                os.replace(temporary, output)
            return relative, {"source": digest, "output": output_digest}, True
        except StagingError as error:
            raise StagingError(f"{relative}: {error}") from error
        finally:
            temporary.unlink(missing_ok=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        results = list(pool.map(stage, paths))
    files = {relative: record for relative, record, _ in results}
    for relative in old_files.keys() - files.keys():
        managed_output(destination, relative).unlink(missing_ok=True)
    atomic_write(state_path, (json.dumps({"configuration": configuration, "files": files}, sort_keys=True, indent=2) + "\n").encode())
    if depfile:
        target = Path(depfile_target or stamp).resolve()
        rule = depfile_escape(target) + ": " + " ".join(depfile_escape(p) for p in dependencies + paths) + "\n"
        atomic_write(Path(depfile), rule.encode())
    if stamp:
        Path(stamp).parent.mkdir(parents=True, exist_ok=True)
        Path(stamp).touch()
    changed = sum(changed for _, _, changed in results)
    print(f"Staged {len(paths)} HIP source/header files ({changed} transformed).", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("sdk", type=Path)
    parser.add_argument("--perl")
    parser.add_argument("--bash")
    parser.add_argument("--hipify")
    parser.add_argument("--fault-injection", action="store_true")
    parser.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--stamp", type=Path)
    parser.add_argument("--depfile", type=Path)
    parser.add_argument("--depfile-target", type=Path)
    args = parser.parse_args()
    try:
        tools = discover_tools(args.sdk, args.perl, args.bash, args.hipify)
        stage_sources(args.source, args.destination, args.sdk, tools, fault_injection=args.fault_injection,
                      jobs=args.jobs, stamp=args.stamp, depfile=args.depfile, depfile_target=args.depfile_target)
    except (OSError, ValueError, StagingError) as error:
        parser.exit(1, f"HIP staging failed: {error}\n")


if __name__ == "__main__":
    main()
