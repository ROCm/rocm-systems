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

import ctypes
import json
import os
import stat
import sys

import unittest

import common
import runcmd

amdsmi_path = os.environ.get('AMDSMI_PATH', '/opt/rocm/share/amd_smi')
if not os.path.exists(amdsmi_path):
    raise FileNotFoundError(f'AMDSMI_PATH "{amdsmi_path}" does not exist. Please set the correct path in your environment.')
sys.path.append(amdsmi_path)
try:
    import amdsmi
except ImportError:
    raise ImportError(f'Could not import the "amdsmi" module from "{amdsmi_path}"')

# add watch args

class TestAmdSmiCli(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.common = common.Common(verbose)
        self.util = runcmd.Util('WARNING')
        self.Debug = False

        self.AddCmdMods = True
        self.AddDeviceArgs = True
        self.AddWatchArgs = False
        self.AddLogLevel = '--loglevel DEBUG'
        self.AddLogLevel = ''

        global has_info_printed
        if verbose and has_info_printed is False:
            # Execute the following to print the asic and board info once
            # per test run
            has_info_printed = True
            if self.Debug:
                for i, gpu in enumerate(self.common.processors):
                    msg = f'gpu={i}'
                    self.common.print(msg)
                    msg = f'virtualization mode(gpu={i})'
                    self.common.print(msg, self.common.virt_mode[i])
                    msg = f'asic info(gpu={i})'
                    self.common.print(msg, self.common.asic_info[i])
                    msg = f'board info(gpu={i})'
                    self.common.print(msg, self.common.board_info[i])
                    self.common.print('')

        self.PASS = 0
        self.FAIL = 1
        self.tab = '    '
        self.tmp_filename = '_tmp.log'

        if False:
            # Get GPU identification
            amdsmi.amdsmi_init()
            self.gpu_choices = []
            gpus = ['all']
            for i, gpu in enumerate(self.common.processors):
                gpus.append(i)
                bdf = amdsmi.amdsmi_get_gpu_device_bdf(gpu)
                gpus.append(bdf)
                uuid = amdsmi.amdsmi_get_gpu_device_uuid(gpu)
                gpus.append(uuid)
                self.gpu_choices.append({"BDF": bdf, "UUID": uuid, "Device Handle": gpu})
            amdsmi.amdsmi_shut_down()

        # When parsing, expand each arg with array element
        self.sub_args = \
        {
            'CLOCK': ['SYS','DF','DCEF','SOC','MEM','VCLK0','VCLK1','DCLK0','DCLK1','ALL'],
            'PID': [123],
            'NAME': ['AMD'],
            'GPU': ['all'] + [i for i in range(len(self.common.processors))],
            'FILE': [self.tmp_filename],
            #'LEVEL': ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        }

        # When parsing, ignore these entries as they are abmormal
        self.cmd_arg_exceptions = \
        [
            '--voltage',
        ]

        # When parsing, change these args into something else or add to arg
        self.cmd_arg_changes = \
        [
            '--loglevel',
            '--json',
            '--csv',
            '--ucode-list',
            '--watch',
            '--watch_time',
            '--iterations',
        ]

        return

    def setUp(self):
        # Called before each test by unittest framework
        return

    def tearDown(self):
        # Called after each test by unittest framework
        return

    def FindArgs(self, cmd, match_str):
        if (not match_str) or \
           (not self.AddDeviceArgs and 'Device' in match_str) or \
           (not self.AddWatchArgs and 'Watch' in match_str) or \
           (not self.AddCmdMods and 'Command' in match_str):
            return ['pass']
        (rc, std_out, std_err) = self.util.RunCmdSync(cmd)
        found = False
        options = []
        lines = std_out.split('\n')
        for index, line in enumerate(lines):
            if found:
                if not line:
                    break
                items = line.split()
                for item_index, item in enumerate(items):
                    items[item_index] = item.strip()
                item_index = -1
                if '-h' == items[0][0:2]:
                    # Turn help into command without an option
                    options.append('')
                elif '--' in items[0][0:2]:
                    item_index = 0
                elif len(items) > 1 and '--' == items[1][0:2]:
                    item_index = 1
                elif '-' == items[0][0:1]:
                    item_index = 0
                sub_found = False
                if item_index >= 0:
                    if items[item_index][-1:] == ',':
                        items[item_index] = items[item_index][:-1]
                    if items[item_index] in self.cmd_arg_exceptions:
                        pass
                    elif items[item_index] in self.cmd_arg_changes:
                        sub_found = True
                        if '--ucode-list' == items[item_index]:
                            options.append(f'{items[item_index]}')
                            options.append('--fw-list')
                        elif '--json' == items[item_index] or '--csv' == items[item_index]:
                            options.append(f'{items[item_index]}')
                            options.append(f'{items[item_index]} --file {self.tmp_filename}')
                        elif '--watch_time' == items[item_index] or '--iterations' == items[item_index]:
                            options.append(f'--watch 1 {items[item_index]} 2')
                        elif '--loglevel' == items[item_index]:
                            pass
                        elif '--watch' == items[item_index]:
                            pass
                        else:
                            print('ERROR: bad sub arg {items[item_index]}')
                    elif len(items) > item_index:
                        if items[item_index+1][0:1] == '[': # ]
                            items[item_index+1] = items[item_index+1][1:]
                        sub_arg = items[item_index+1]
                        if sub_arg.isupper() and sub_arg in self.sub_args:
                            sub_found = True
                            for item in self.sub_args[sub_arg]:
                                options.append(f'{items[item_index]} {item}')
                    if not sub_found:
                        # Put in sub_arg if it was not found
                        options.append(items[item_index])
            if match_str in line:
                found = True
        if not options:
            return ['pass']
        return options

    def CreateCmds(self, cmd_name, list1_name, list2_name, list3_name, list4_name):
        cmd = f'amd-smi {cmd_name} --help'
        list1_args = self.FindArgs(cmd, list1_name)
        if self.Debug:
            print(f'{list1_name}: {"*"*80}')
            print(json.dumps(list1_args, sort_keys=False, indent=4), flush=True)
        list2_args = self.FindArgs(cmd, list2_name)
        if self.Debug:
            print(f'{list2_name}: {"*"*80}')
            print(json.dumps(list2_args, sort_keys=False, indent=4), flush=True)
        list3_args = self.FindArgs(cmd, list3_name)
        if self.Debug:
            print(f'{list3_name}: {"*"*80}')
            print(json.dumps(list3_args, sort_keys=False, indent=4), flush=True)
        list4_args = self.FindArgs(cmd, list4_name)
        if self.Debug:
            print(f'{list4_name}: {"*"*80}')
            print(json.dumps(list4_args, sort_keys=False, indent=4), flush=True)

        cmds = []
        cmd = f'amd-smi {cmd_name}'
        for list1_arg in list1_args:
            if list1_arg != 'pass':
                #cmds.append((f'{cmd} {list1_arg}', self.PASS))
                cmds.append((f'{cmd} {list1_arg} {self.AddLogLevel}', self.PASS))
            else:
                list1_arg = ''
            for list2_arg in list2_args:
                if list2_arg != 'pass':
                    cmds.append((f'{cmd} {list1_arg} {list2_arg} {self.AddLogLevel}', self.PASS))
                else:
                    list2_arg = ''
                for list3_arg in list3_args:
                    if list3_arg != 'pass':
                        cmds.append((f'{cmd} {list1_arg} {list2_arg} {list3_arg} {self.AddLogLevel}', self.PASS))
                    else:
                        list3_arg = ''
                    for list4_arg in list4_args:
                        if list4_arg != 'pass':
                            cmds.append((f'{cmd} {list1_arg} {list2_arg} {list3_arg} {list4_arg} {self.AddLogLevel}', self.PASS))

        # Remove extra spaces between arguments
        for index, cmd_cond in enumerate(cmds):
            cmd, cond = cmd_cond
            cmd = cmd.split()
            cmd = ' '.join(cmd).strip()
            cmds[index] = (cmd, cond)
        if self.Debug:
            print(f'cmds: {"*"*80}')
            print(json.dumps(cmds, sort_keys=False, indent=4), flush=True)
        return cmds

    def RunCmds(self, cmds):
        errors = []
        for cmd, cond in cmds:
            if self.Debug:
                print(f'cmd={cmd}')
            (rc, std_out, std_err) = self.util.RunCmdSync(cmd)
            error_code = rc
            if rc and len(std_err):
                items = std_err.split()
                if 'amdsmi_exception' in std_err:
                    # error code from amdsmi library exception
                    for index, item in enumerate(items):
                        if item == 'Error':
                            error_code_str = items[index+4]
                            error_code = error_code_str
                            #break
                else:
                    # error code from amd-smi CLI
                    error_code = items[-1]
                    # Check for parse error 'choice'
                    if 'CRITICAL' in error_code:
                        error_code = 'Bad loglevel'

            msg=f'{cmd:85s}:'
            if '--file' in cmd:
                if not os.path.exists(self.tmp_filename):
                    _msg = f'{msg} Failure: File {self.tmp_filename} does not exist'
                    errors.append(_msg)
                else:
                    with open(self.tmp_filename, 'r') as fin:
                        std_out = fin.read()
                    if not len(std_out):
                        _msg = f'{msg} Failure: File {self.tmp_filename} was empty'
                        errors.append(_msg)
                    os.chmod(self.tmp_filename, stat.S_IWRITE)
                    os.remove(self.tmp_filename)
        
            if rc and cond == self.PASS:
                msg += f' Failure: Received FAIL ({error_code}), expected PASS (0)'
                errors.append(msg)
            elif not rc and cond != self.PASS:
                msg += f' Failure: Received PASS (0), expected FAIL (!0)'
                errors.append(msg)
            else:
                if not rc:
                    expected = 'PASS'
                else:
                    expected = 'FAIL'
                msg += f' Success: Received and Expected {expected} ({error_code})'

            self.common.print(f'{self.tab}{msg}')
            if self.Debug:
                print(f'{self.tab}rc={rc}')
                print(f'{self.tab}error_code={error_code}')
                print(f'{self.tab}std_out={std_out}')
                print(f'{self.tab}std_err={std_err}')
        if len(errors):
            msg = f'\n{self.tab}'.join(errors)
            self.fail(f'Fail:\n{self.tab}{msg}')

    def test_help(self):
        self.common.print_func_name('')
        msg = f'### amd-smi help'
        self.common.print(msg)

        cmd = 'amd-smi --help'
        (rc, std_out, std_err) = self.util.RunCmdSync(cmd)
        lines = std_out.split('\n')
        # Find all available command line args
        cmd_args = []
        found = False
        for line in lines:
            if found:
                if not line:
                    break
                items = line.split()
                cmd_args.append(items[0])
                continue;
            if 'Descriptions' in line:
                found = True

        cmds = [(f'amd-smi --help', self.PASS)]
        for cmd_arg in cmd_args:
            cmds.append((f'amd-smi {cmd_arg} --help', self.PASS))

        self.RunCmds(cmds)
        return

    def test_invalid(self):
        self.common.print_func_name('')
        msg = f'{self.tab}### amd-smi'
        self.common.print(msg)
        cmds = \
        [
            # Test invalid command
            ('amd-smi invalid_cmd', self.FAIL),
            # Test invalid sub command
            ('amd-smi version --invalid', self.FAIL),
            ('amd-smi list --invalid', self.FAIL),
            ('amd-smi static --invalid', self.FAIL),
            ('amd-smi firmware --invalid', self.FAIL),
            ('amd-smi bad_pages --invalid', self.FAIL),
            ('amd-smi metric --invalid', self.FAIL),
            ('amd-smi process --invalid', self.FAIL),
            ('amd-smi event --invalid', self.FAIL),
            ('amd-smi topology --invalid', self.FAIL),
            ('amd-smi set --invalid', self.FAIL),
            ('amd-smi reset --invalid', self.FAIL),
            ('amd-smi monitor --invalid', self.FAIL),
            ('amd-smi xgmi --invalid', self.FAIL),
            ('amd-smi partition --invalid', self.FAIL),
            ('amd-smi ras --invalid', self.FAIL),
            ('amd-smi node --invalid', self.FAIL),
            # Test invalid gpu value
            ('amd-smi version --gpu 0', self.FAIL),
            ('amd-smi version --gpu -1', self.FAIL),
            ('amd-smi version --gpu ALL', self.FAIL),
            (f'amd-smi version --gpu {len(self.common.processors)}', self.FAIL),
            ('amd-smi static --gpu -1', self.FAIL),
            ('amd-smi static --gpu _ALL', self.FAIL),
            (f'amd-smi static --gpu {len(self.common.processors)}', self.FAIL),
            # Test invalid loglevel
            ('amd-smi metric --loglevel BADLEVEL', self.FAIL),
            # Test invalid process PID, NAME
            ('amd-smi process --pid NOT_A_NUMBER', self.FAIL),
            ('amd-smi process --pid', self.FAIL),
            ('amd-smi process --name', self.FAIL),
            # Test invalid watch order
            ('amd-smi monitor --interval 2 --watch 1', self.FAIL),
            ('amd-smi monitor --watch_time 2 --watch 1', self.FAIL),
        ]

        # TODO: Add BDF value for gpu
        # TODO: Add bad BDF value for gpu

        self.RunCmds(cmds)
        return

    def test_default(self):
        self.common.print_func_name('')
        msg = f'{self.tab}### amd-smi'
        self.common.print(msg)

        cmds = \
        [
            ('amd-smi', self.PASS),
        ]

        self.RunCmds(cmds)
        return

    def test_version(self):
        self.common.print_func_name('')
        msg = f'{self.tab}### amd-smi version'
        self.common.print(msg)

        cmds = \
        [
            ('amd-smi version', self.PASS),
            ('amd-smi version --cpu_version', self.PASS),
            ('amd-smi version --gpu_version', self.PASS)
        ]

        self.RunCmds(cmds)
        return

    def test_list(self):
        self.common.print_func_name('')
        msg = f'{self.tab}### amd-smi list'
        self.common.print(msg)

        cmds = self.CreateCmds('list', 'List Arguments:', 'Device Arguments:', 'Command Modifiers:', '')
        self.RunCmds(cmds)
        return

    def test_static(self):
        self.common.print_func_name('')
        msg = f'{self.tab}### amd-smi static'
        self.common.print(msg)

        cmds = self.CreateCmds('static', 'Static Arguments:', 'Device Arguments:', 'Command Modifiers:', '')
        self.RunCmds(cmds)
        return

    def test_firmware(self):
        self.common.print_func_name('')
        msg = f'{self.tab}### amd-smi firmware'
        self.common.print(msg)

        cmds = self.CreateCmds('firmware', 'Firmware Arguments:', 'Device Arguments:', 'Command Modifiers:', '')
        self.RunCmds(cmds)
        cmds = self.CreateCmds('ucode', 'Firmware Arguments:', 'Device Arguments:', 'Command Modifiers:', '')
        self.RunCmds(cmds)
        return

    def test_bad_pages(self):
        self.common.print_func_name('')
        msg = f'{self.tab}### amd-smi bad-pages'
        self.common.print(msg)

        cmds = self.CreateCmds('bad-pages', 'Bad Pages Arguments:', 'Device Arguments:', 'Command Modifiers:', '')
        self.RunCmds(cmds)
        return

    def test_metric(self):
        self.common.print_func_name('')
        msg = f'{self.tab}### amd-smi metric'
        self.common.print(msg)

        cmds = self.CreateCmds('metric', 'Metric arguments:', 'Device Arguments:', 'Command Modifiers:', 'Watch Arguments:')
        self.RunCmds(cmds)
        return

    def test_process(self):
        self.common.print_func_name('')
        msg = f'{self.tab}### amd-smi process'
        self.common.print(msg)

        cmds = self.CreateCmds('process', 'Process arguments:', 'Device Arguments:', 'Command Modifiers:', 'Watch Arguments:')
        self.RunCmds(cmds)
        return

    def test_event(self):
        self.common.print_func_name('')
        msg = f'{self.tab}### amd-smi event'
        self.common.print(msg)

        if self.common.TODO_SKIP_FAIL:
            msg = f'{self.tab}Needs input'
            self.common.print(msg)
            self.skipTest(msg)

        cmds = self.CreateCmds('event', 'Event Arguments:', 'Device Arguments:', 'Command Modifiers:', '')
        self.RunCmds(cmds)
        return

    def test_topology(self):
        self.common.print_func_name('')
        msg = f'{self.tab}### amd-smi topology'
        self.common.print(msg)

        cmds = self.CreateCmds('topology', 'Topology arguments:', 'Device Arguments:', 'Command Modifiers:', '')
        self.RunCmds(cmds)
        return

    def test_set(self):
        self.common.print_func_name('')
        msg = f'{self.tab}### amd-smi set'
        self.common.print(msg)

        if self.common.TODO_SKIP_FAIL:
            msg = f'{self.tab}Needs input, Not Yet Implemented'
            #self.common.print(msg)
            self.skipTest(msg)

        cmds = self.CreateCmds('set', 'Set Arguments:', 'Device Arguments:', 'Command Modifiers:', '')
        self.RunCmds(cmds)
        return

    def test_reset(self):
        self.common.print_func_name('')
        msg = f'{self.tab}### amd-smi reset'
        self.common.print(msg)

        if self.common.TODO_SKIP_FAIL:
            msg = f'{self.tab}Needs Testing, Not Yet Implemented'
            #self.common.print(msg)
            self.skipTest(msg)

        cmds = self.CreateCmds('reset', 'Reset Arguments:', 'Device Arguments:', 'Command Modifiers:', '')
        self.RunCmds(cmds)
        return

    def test_monitor(self):
        self.common.print_func_name('')
        msg = f'{self.tab}### amd-smi monitor'
        self.common.print(msg)

        cmds = self.CreateCmds('monitor', 'Monitor Arguments:', 'Device Arguments:', 'Command Modifiers:', 'Watch Arguments:')
        self.RunCmds(cmds)
        return

    def test_xgmi(self):
        self.common.print_func_name('')
        msg = f'{self.tab}### amd-smi xgmi'
        self.common.print(msg)

        cmds = self.CreateCmds('xgmi', 'XGMI arguments:', 'Device Arguments:', 'Command Modifiers:', '')
        self.RunCmds(cmds)
        return

    def test_partition(self):
        self.common.print_func_name('')
        msg = f'{self.tab}### amd-smi partition'
        self.common.print(msg)

        cmds = self.CreateCmds('partition', 'Partition arguments:', 'Device Arguments:', 'Command Modifiers:', '')
        self.RunCmds(cmds)
        return

    def test_ras(self):
        self.common.print_func_name('')
        msg = f'{self.tab}### amd-smi ras'
        self.common.print(msg)

        if self.common.TODO_SKIP_FAIL:
            msg = f'{self.tab}Needs input, Not Yet Implemented'
            #self.common.print(msg)
            self.skipTest(msg)

        # TODO RAS arguments determine which arguments need to be scanned, CPER or AFID
        cmds = self.CreateCmds('ras', 'RAS arguments:', 'Device Arguments:', 'Command Modifiers:', '')
        self.RunCmds(cmds)
        return

    def test_node(self):
        self.common.print_func_name('')
        msg = f'{self.tab}### amd-smi node'
        self.common.print(msg)

        cmds = self.CreateCmds('node', 'Node arguments:', 'Device Arguments:', 'Command Modifiers:', '')
        self.RunCmds(cmds)
        return


if __name__ == '__main__':
    verbose=1
    if '-q' in sys.argv or '--quiet' in sys.argv:
        verbose=0
    elif '-v' in sys.argv or '--verbose' in sys.argv:
        verbose=2
    has_info_printed = False

    if verbose:
        print('AMD SMI CLI Tests')

    # Detect if ran without sudo or root privileges
    if os.geteuid() != 0:
        print('Warning: Some tests may require elevated privileges (sudo/root) to run completely.\n')
        print('Please relaunch with elevated privileges.\n')
        sys.exit(1)

    runner = unittest.TextTestRunner(verbosity=verbose)
    unittest.main(testRunner=runner)
    sys.exit(0)

