"""Retain exact Tensile client inputs for reviewed fault campaigns."""

from __future__ import annotations

import hashlib
from pathlib import Path
import shlex
import time


def digest(path: Path) -> str:
    with path.open('rb') as source:
        hasher = hashlib.sha256()
        for block in iter(lambda: source.read(1024 * 1024), b''):
            hasher.update(block)
        return hasher.hexdigest()


def freeze(work_dir: Path, wrapper: Path, contract: dict) -> dict:
    clients = []
    files = {}
    for script in sorted(work_dir.glob('1_BenchmarkProblems/**/build/run.sh')):
        files[str(script.resolve())] = digest(script)
        for line in script.read_text().splitlines():
            tokens = shlex.split(line)
            if not tokens or tokens[0] != str(wrapper):
                continue
            if len(tokens) != 3 or tokens[1] != '--config-file':
                raise ValueError(f'unsupported Tensile client invocation: {script}')
            ini = Path(tokens[2]).resolve()
            ini.relative_to(work_dir.resolve())
            inputs = [ini]
            keys = []
            for entry in ini.read_text().splitlines():
                key, sep, value = entry.partition('=')
                if sep and key in {'library-file', 'code-object'}:
                    path = Path(value).resolve()
                    path.relative_to(work_dir.resolve())
                    inputs.append(path)
                    keys.append(key)
            if 'library-file' not in keys or 'code-object' not in keys:
                raise ValueError(f'missing client library or code object: {ini}')
            if sum(s.startswith('results-file=') for s in ini.read_text().splitlines()) != 1:
                raise ValueError(f'expected one client results file: {ini}')
            for path in inputs:
                files[str(path)] = digest(path)
            clients.append(str(ini))
    if not clients or len(set(clients)) != len(clients):
        raise ValueError('expected nonempty, distinct retained client invocations')
    return {'schema_version': 1, 'contract': contract, 'work_dir': str(work_dir.resolve()),
            'clients': clients, 'files': files}


def verify(manifest: dict, contract: dict) -> None:
    if manifest.get('schema_version') != 1 or manifest.get('contract') != contract:
        raise ValueError('Tensile replay contract does not match requested workload')
    # Re-discover the invocations as well as hashing all their inputs. This also
    # rejects changed run scripts, removed clients, and newly referenced inputs.
    actual = freeze(Path(manifest['work_dir']), Path(contract['wrapper']), contract)
    if actual != manifest:
        raise ValueError('retained Tensile client inputs changed since review')


def run(manifest: dict, contract: dict, output_dir: Path, environment: dict,
        timeout_seconds: int, run_command) -> tuple[int, str, bool]:
    verify(manifest, contract)
    deadline = time.monotonic() + timeout_seconds
    output = []
    for index, client in enumerate(manifest['clients']):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return 1, ''.join(output), True
        ini = output_dir / f'client-{index}.ini'
        ini.write_text('\n'.join(
            f'results-file={output_dir / f"client-{index}.csv"}'
            if line.startswith('results-file=') else line
            for line in Path(client).read_text().splitlines()) + '\n')
        code, text, timed_out = run_command(
            [contract['wrapper'], '--config-file', str(ini)], environment, remaining)
        output.append(text + '\n')
        if timed_out or code != 0:
            return code, ''.join(output), timed_out
        output.append(f'clientExit=0 (PASS) for retained-client-{index}\n')
    verify(manifest, contract)
    return 0, ''.join(output), False
