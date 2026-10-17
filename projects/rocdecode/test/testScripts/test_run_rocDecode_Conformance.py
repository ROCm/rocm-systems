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


class RocDecodeConformanceTest(unittest.TestCase):
    def test_paths_are_passed_as_literal_arguments(self):
        script = Path(__file__).with_name('run_rocDecode_Conformance.py')

        with tempfile.TemporaryDirectory() as tempDir:
            tempPath = Path(tempDir)
            filesPath = tempPath / 'bundle with spaces'
            streamsPath = filesPath / 'Streams'
            md5Path = filesPath / 'MD5'
            streamsPath.mkdir(parents=True)
            md5Path.mkdir()

            suspiciousName = 'video;touch PWNED;.h265'
            streamFile = streamsPath / suspiciousName
            md5File = md5Path / suspiciousName
            streamFile.touch()
            md5File.touch()

            argvLog = tempPath / 'argv.json'
            fakeDecoder = tempPath / 'fake decoder'
            fakeDecoder.write_text(
                '#!/usr/bin/env python3\n'
                'import json\n'
                'import os\n'
                'import sys\n'
                'with open(os.environ["ROCDECODE_ARGV_LOG"], "w") as log:\n'
                '    json.dump(sys.argv[1:], log)\n'
                'print("Input file: test")\n'
                'print("MD5 message digest: test")\n'
                'print("MD5 digest matches the reference MD5 digest")\n'
            )
            fakeDecoder.chmod(0o755)

            resultsPath = tempPath / 'results;touch RESULTS_PWNED;'
            env = os.environ.copy()
            env['ROCDECODE_ARGV_LOG'] = str(argvLog)
            completed = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    '--videodecode_exe',
                    str(fakeDecoder),
                    '--files_directory',
                    str(filesPath),
                    '--results_directory',
                    str(resultsPath),
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
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertFalse((tempPath / 'PWNED').exists())
            self.assertFalse((tempPath / 'RESULTS_PWNED').exists())
            self.assertEqual(
                json.loads(argvLog.read_text()),
                [
                    '-i',
                    str(streamFile),
                    '-no_ffmpeg_demux',
                    '-md5_check',
                    str(md5File),
                    '-d',
                    '0',
                ],
            )

            outputLog = resultsPath / 'rocDecode_videoDecode_results' / 'rocDecode_output.log'
            output = outputLog.read_text()
            self.assertIn('MD5 digest matches the reference MD5 digest', output)
            self.assertIn('MD5 digest matches the reference MD5 digest', completed.stdout)
            self.assertTrue(
                (resultsPath / 'rocDecode_videoDecode_results' / 'rocDecode_conformance.log').is_file()
            )


if __name__ == '__main__':
    unittest.main()
