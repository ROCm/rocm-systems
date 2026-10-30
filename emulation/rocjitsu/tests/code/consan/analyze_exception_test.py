# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Exercise an unexpected exception in the actual analyzer executable."""

import argparse
import os
from pathlib import Path
import resource
import subprocess
import tempfile


def disable_core_dump():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analyzer', type=Path, required=True)
    parser.add_argument('--injector', type=Path, required=True)
    args = parser.parse_args()
    environment = os.environ.copy()
    injected = environment | {
        'LD_PRELOAD': str(args.injector.resolve())
        + (':' + environment['LD_PRELOAD'] if environment.get('LD_PRELOAD') else '')
    }
    with tempfile.TemporaryDirectory(prefix='consan-analyzer-exception-') as directory:
        capture = Path(directory) / 'input.bin'
        for size, env, fatal in (
            (123457, environment, False),
            (1, injected, False),
            (123457, injected, True),
        ):
            capture.write_bytes(b'\0' * size)
            command = [str(args.analyzer.resolve()), str(capture)]
            result = subprocess.run(
                command,
                env=env,
                capture_output=True,
                timeout=15,
                preexec_fn=disable_core_dump,
                check=False,
            )
            print(
                f'{command}: bytes={size}, injected={env is injected}, exit={result.returncode}'
            )
            print(result.stderr.decode(errors='replace'))
            assert not result.stdout, 'failed input unexpectedly produced a result'
            if fatal:
                assert b'ConSan test: throwing input allocation' in result.stderr
                assert (
                    result.returncode < 0
                ), 'unexpected exception returned a normal CLI status'
                assert b'ConSan analyzer tool failure:' not in result.stderr
            else:
                assert result.returncode == 2, 'normal validation failure changed'
                assert b'capture input error:' in result.stderr
                assert b'ConSan test: throwing input allocation' not in result.stderr


if __name__ == '__main__':
    main()
