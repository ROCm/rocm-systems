# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class RocDecodeSamplesTest(unittest.TestCase):
    def _run_sample(self, tempPath, fakeDecoder, filesPath, resultsPath):
        env = os.environ.copy()
        env['PYTHONPATH'] = str(tempPath)
        env['ROCDECODE_ARGV_LOG'] = str(tempPath / 'argv.json')
        script = Path(__file__).with_name('run_rocDecodeSamples.py')
        return subprocess.run(
            [
                sys.executable,
                str(script),
                '--videodecode_exe',
                str(fakeDecoder),
                '--files_directory',
                str(filesPath),
                '--results_directory',
                str(resultsPath),
                '--check_decode_status',
                '1',
                '--use_ffmpeg_demuxer',
                '0',
            ],
            cwd=tempPath,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

    def test_paths_are_passed_as_literal_arguments(self):
        with tempfile.TemporaryDirectory() as tempDir:
            tempPath = Path(tempDir)
            (tempPath / 'pandas.py').write_text('')
            filesPath = tempPath / 'bundle with spaces'
            filesPath.mkdir()
            suspiciousName = 'video;touch PWNED;.h265'
            inputFile = filesPath / suspiciousName
            inputFile.touch()

            fakeDecoder = tempPath / 'fake decoder'
            fakeDecoder.write_text(
                '#!/usr/bin/env python3\n'
                'import json\n'
                'import os\n'
                'import sys\n'
                'with open(os.environ["ROCDECODE_ARGV_LOG"], "w") as log:\n'
                '    json.dump(sys.argv[1:], log)\n'
                'print("info: Input file: test")\n'
                'print("info: Total pictures decoded: 1")\n'
            )
            fakeDecoder.chmod(0o755)

            resultsPath = tempPath / 'results;touch RESULTS_PWNED;'
            completed = self._run_sample(
                tempPath, fakeDecoder, filesPath, resultsPath
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertFalse((tempPath / 'PWNED').exists())
            self.assertFalse((tempPath / 'RESULTS_PWNED').exists())
            self.assertEqual(
                json.loads((tempPath / 'argv.json').read_text()),
                [
                    '-i',
                    str(inputFile),
                    '-d',
                    '0',
                    '-f',
                    '0',
                    '-no_ffmpeg_demux',
                ],
            )
            outputLog = (
                resultsPath
                / 'rocDecode_videoDecode_results'
                / 'rocDecode_output.log'
            )
            self.assertIn(
                'info: Total pictures decoded: 1',
                outputLog.read_text(),
            )

    def test_decoder_failure_is_propagated(self):
        with tempfile.TemporaryDirectory() as tempDir:
            tempPath = Path(tempDir)
            (tempPath / 'pandas.py').write_text('')
            filesPath = tempPath / 'bundle'
            filesPath.mkdir()
            (filesPath / 'input.h265').touch()

            fakeDecoder = tempPath / 'failing decoder'
            fakeDecoder.write_text('#!/bin/sh\nexit 7\n')
            fakeDecoder.chmod(0o755)

            completed = self._run_sample(
                tempPath, fakeDecoder, filesPath, tempPath / 'results'
            )
            self.assertEqual(completed.returncode, 7)


if __name__ == '__main__':
    unittest.main()
