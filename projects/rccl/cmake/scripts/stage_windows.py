"""Stage HIP sources with the same hipify/unroll transforms as the Linux build."""
import concurrent.futures
import os
import re
from pathlib import Path
import subprocess
import sys

source, destination, sdk = map(Path, sys.argv[1:])
git_bin = Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Git/usr/bin"
perl, bash = git_bin / "perl.exe", git_bin / "bash.exe"
hipify = sdk / "libexec/hipify/hipify-perl"
env = dict(os.environ, PATH=str(git_bin) + os.pathsep + os.environ["PATH"])

def stage(path):
    relative = path.relative_to(source)
    output = destination / relative
    if output.exists() and output.stat().st_mtime > max(path.stat().st_mtime, Path(__file__).stat().st_mtime):
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(perl), str(hipify), "-experimental", "-quiet-warnings", str(path), "-o", str(output)], check=True, env=env, capture_output=True)
    # The Linux flat-name collision workaround uses *_tmp.h. This staging tree
    # preserves every source subdirectory, so includes keep their original names.
    content = output.read_text(encoding="utf-8")
    content = re.sub(r'(#include\s*[<"][^">\n]*)_tmp(\.h[">])', r'\1\2', content)
    output.write_text(content, encoding="utf-8")
    if relative.as_posix().startswith("src/device/") and path.suffix == ".h":
        subprocess.run([str(bash), str(source / "cmake/scripts/add_unroll.sh"), output.as_posix()], check=True, env=env, capture_output=True)

paths = [p for p in (source / "src").rglob("*") if p.suffix in (".h", ".hpp", ".cuh", ".cc", ".cu", ".c")]
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
    list(pool.map(stage, paths))
print(f"Staged {len(paths)} HIP source and header files.", flush=True)
