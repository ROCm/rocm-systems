#!/usr/bin/env python3
#
# Copyright (C) Advanced Micro Devices. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import argparse
import datetime
import locale
import subprocess
import sys


version_number = '1.0.0'
build_date = f'{datetime.datetime.now():%b %d %Y}'
verbose_choices = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']


class Util:
    def __init__(self, debug_level='ERROR'):
        # Set local encoding for output
        self.use_encoding = locale.getpreferredencoding()
        self.debug_level_index = verbose_choices.index(debug_level)
        return

    def ConvertStr(self, data_in):
        '''
        Decodes string depending on encoding

        Args:
            data_in (str): Command line argument to run

        Returns:
            (str): Decoded string on success otherwise None on failure
        '''

        data_out = None
        if data_in:
            if self.use_encoding:
                data_out = data_in.encode('utf8').decode()
            else:
                data_out = data_in.decode().strip()

        return data_out

    def Print(self, cond, msg, line_flush=True, line_ending='\n'):
        if isinstance(cond, str) and cond in verbose_choices:
            index = verbose_choices.index(cond)
            if index >= self.debug_level_index:
                print(f'{cond}: {msg}', flush=line_flush, end=line_ending)
        elif cond:
            print(msg, flush=line_flush, end=line_ending)
        return

    def GetFuncName(self, stack_line=3):
        '''
        Function name calling this module

        Args:
            stack_line (int, optional): How far down the stack to get the function name.

        Returns:
            (str or 'Unknown'): Function name
        '''

        try:
            func_name = sys._getframe(stack_line).f_back.f_code.co_name
        except Exception as e:
            func_name = 'Unknown'
            self.Print('EXCEPTION', f'Cannot get function name at stack_line {stack_line}')
        return func_name

    def _RunCmd(self, cmd, use_shell, msg_in, time_out, wait):
        if isinstance(cmd, str):
            cmd = cmd.split()

        rc = 1
        std_out = ''
        std_err = ''
        proc = None

        self.Print('INFO', f'RunCmd {cmd}')

        if not cmd or len(cmd) == 0:
            func_name = self.GetFuncName()
            std_err = f'{func_name}: No command supplied'
            self.Print('ERROR', std_err)
            return (rc, std_out, std_err)

        try:
            std_in = None
            if msg_in:
                std_in = subprocess.PIPE

            proc = subprocess.Popen(cmd, encoding=self.use_encoding, shell=use_shell,
                stdin=std_in, stderr=subprocess.PIPE, stdout=subprocess.PIPE)

            if msg_in:
                if not self.use_encoding:
                    msg_in = msg_in.encode()

            if wait:
                stdout_data, stderr_data = proc.communicate(input=msg_in, timeout=time_out)

                rc = proc.returncode
                std_out = self.ConvertStr(stdout_data)
                std_err = self.ConvertStr(stderr_data)

                self.Print('DEBUG', f'rc={rc}')
                self.Print('DEBUG', f'std_out={std_out}')
                self.Print('DEBUG', f'std_err={std_err}')
            else:
                rc = 0
        except subprocess.TimeoutExpired as e:
            rc = 2
            func_name = self.GetFuncName()
            self.Print('EXCEPTION', f'rc={rc} {func_name}: Timeout: cmd={cmd}')
            if msg_in:
                self.Print('EXCEPTION', f'\tstd_in={msg_in}')
            self.Print('EXCEPTION', f'{e}')

            # Process took longer than expected so terminate cmd and collect output
            proc.kill()
            stdout_data, stderr_data = proc.communicate()

            std_out = self.ConvertStr(stdout_data)
            std_err = self.ConvertStr(stderr_data)

            self.Print('EXCEPTION', f'std_out={std_out}')
            self.Print('EXCEPTION', f'std_err={std_err}')
            self.Print('EXCEPTION', f'{e}')
        except Exception as e:
            rc = 3
            func_name = self.GetFuncName()
            self.Print('EXCEPTION', f'rc={rc} {func_name}: cmd={cmd}')
            if msg_in:
                self.Print('EXCEPTION', f'\tstd_in={msg_in}')
            self.Print('EXCEPTION', f'\tstd_out={std_out}')
            self.Print('EXCEPTION', f'\tstd_err={std_err}')
            self.Print('EXCEPTION', f'{e}')

        return (rc, std_out, std_err, proc)

    def RunCmd(self, cmd, use_shell=False, msg_in=None, time_out=None, wait=True):
        '''
        Run a System Command and return rc, std_out, std_err

        See RunCmdSync
        '''
        rc, std_out, std_err, _ = self._RunCmd(cmd, use_shell, msg_in, time_out, wait)
        return (rc, std_out, std_err)

    def RunCmdSync(self, cmd, use_shell=False, msg_in=None, time_out=None):
        '''
        Run a System Command synchronously and return rc, std_out, std_err

        Args:
            cmd (str): Command line argument to run
            use_shell (bool, optional): When True, run in platforms native shell (access to system shell functions)
            msg_in (str, optional): Used as input into the run command standard pipe
            time_out (int, optional): Number of seconds to wait for call to succeed.  If None, wait until finished.

        Returns:
            (int, str or None, str or None): rc, std_out, std_err.
                | rc is the return code and is zero for success otherwise non-zero
                | std_out is standard out or None
                | std_err is standard error or None

        Example:
            | rc, std_out, std_err = RunCmd('<Some Command>')
            | rc, std_out, std_err = RunCmd('<Some Command>', use_shell=True)
        '''

        rc, std_out, std_err, _ = self._RunCmd(cmd, use_shell, msg_in, time_out, wait=True)
        return (rc, std_out, std_err)

    def RunCmdAsync(self, cmd, use_shell=False, msg_in=None):
        '''
        Run a System Command asynchronously and return rc, std_out, std_err, proc

        Args:
            cmd (str): Command line argument to run
            use_shell (bool, optional): When True, run in platforms native shell (access to system shell functions)
            msg_in (str, optional): Used as input into the run command standard pipe

        Returns:
            (int, str or None, str or None, obj): rc, std_out, std_err, proc.
                | rc is the return code and is zero for success otherwise non-zero
                | std_out is standard out or None
                | std_err is standard error or None
                | proc is process id object

        Example:
            | rc, std_out, std_err, proc = RunCmd('<Some Command>')
            | rc, std_out, std_err, proc = RunCmd('<Some Command>', use_shell=True)
        '''

        rc, std_out, std_err, proc = self._RunCmd(cmd, use_shell, msg_in, time_out=None, wait=False)
        return (rc, std_out, std_err, proc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Utility')
    parser.add_argument('--version', action='version', version=version_number, help='Show version and exit')
    parser.add_argument('--build', action='version', version=build_date, help='Show build and exit')
    parser.add_argument('--verbose', choices=verbose_choices , type=str, default='WARNING',
        help='Level of information to output, default=%(default)s')
    parser.add_argument('--cmd', type=str, default=None, help='Run cmd, default=%(default)s')
    args = parser.parse_args()

    util = Util(args.verbose)

    if args.cmd:
        cmd = args.cmd
    else:
        cmd = 'amd-smi'

    (rc, std_out, std_err) = util.RunCmdSync(cmd)
    print(f'output:{cmd}')
    print(f'\trc={rc}')
    print(f'\tstd_out={std_out}')
    print(f'\tstd_err={std_err}')

    sys.exit(rc)

