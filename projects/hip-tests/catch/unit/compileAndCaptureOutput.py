# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

import subprocess
import sys
import unittest
import os

class CompileAndCapture(unittest.TestCase):
  path = None
  expected_error_count = 0
  expected_warning_count = 0
  hip_path = None
  file = None
  error_string = None
  warning_string = None
  platform = None

  def setUp(self):
    self.error_string = 'error:'
    self.warning_string = 'warning:'
    self.assertFalse(self.hip_path == None)
    self.assertFalse(self.path == None)
    self.assertFalse(self.file == None)
    self.assertTrue(self.platform == 'amd' or self.platform == 'nvidia')

  def test(self):
    if os.name == 'nt':
        hipcc = self.hip_path + '/bin/hipcc.bat'
    else:
        hipcc = self.hip_path + '/bin/hipcc'

    compiler_args = [
      hipcc,
      '-I' + self.path + '/../../../include',
      '-c',
      self.path + '/' + self.file,
      ]
    # HIP compiler on AMD platforms has limit of 20 errors, and some negative
    # test cases expect that more errors are detected.
    if (self.platform == 'amd'):
      compiler_args.append('-ferror-limit=200')
    compiler_output = subprocess.run(compiler_args, stderr=subprocess.PIPE)
    # Get the compiler output in the stdout if -V flag is raised during ctest invocation.
    compiler_stderr = compiler_output.stderr.decode('UTF-8')
    print(compiler_stderr)

    error_count = compiler_stderr.count(self.error_string)
    if self.expected_error_count < 0:
      self.assertGreater(error_count, 0)
    else:
      self.assertEqual(error_count, self.expected_error_count)

    warning_count = compiler_stderr.count(self.warning_string)
    if self.expected_warning_count < 0:
      self.assertGreater(warning_count, 0)
    else:
      self.assertEqual(warning_count, self.expected_warning_count)

if __name__ == '__main__':
  try:
    CompileAndCapture.path = sys.argv[1]
  except IndexError:
    CompileAndCapture.path = None

  try:
    CompileAndCapture.platform = sys.argv[2]
  except IndexError:
    CompileAndCapture.platform = None

  try:
    CompileAndCapture.hip_path = sys.argv[3]
  except IndexError:
    CompileAndCapture.hip_path = None

  try:
    CompileAndCapture.file = sys.argv[4]
  except IndexError:
    CompileAndCapture.file = None

  try:
    CompileAndCapture.expected_error_count = int(sys.argv[5])
  except IndexError:
    CompileAndCapture.expected_error_count = 0

  try:
    CompileAndCapture.expected_warning_count = int(sys.argv[6])
  except IndexError:
    CompileAndCapture.expected_warning_count = 0

  # Unittest looks at the same argv's as the __main__ and doesn't know how
  # to handle arguments other than the executable (0). Therefore passing only
  # executable as the argv for unittest module.
  unittest.main(argv=[sys.argv[0]])
