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


class TestAmdSmiCli(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.common = common.Common(verbose)
        self.util = runcmd.Util('WARNING')
        self.Debug = False
        self.ReduceCmds = True
        self.PrintCmdsOnly = False

        self.AddCmdMods = True
        self.AddDeviceArgs = True
        self.AddWatchArgs = True
        self.AddLogLevel = '--loglevel DEBUG'

        # Record starting values
        cmd = 'amd-smi metric --json'
        (rc, data, std_err) = self.util.RunCmdSync(cmd)
        self.metric_data = json.loads(data)

        cmd = 'amd-smi static --json'
        (rc, data, std_err) = self.util.RunCmdSync(cmd)
        self.static_data = json.loads(data)

        cmd = 'amd-smi list --json'
        (rc, data, std_err) = self.util.RunCmdSync(cmd)
        self.list_data = json.loads(data)

        cmd = 'amd-smi partition --current --json'
        (rc, data, std_err) = self.util.RunCmdSync(cmd)
        self.partition_data = json.loads(data)

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
        self.tmp_folder = '_tmp'

        self.openBracket = '['
        self.closeBracket = ']'
        self.openCurlyBrace = '{'
        self.closeCurlyBrace = '}'

        self.gpus = ['all']
        for data in self.list_data:
            self.gpus.append(data['gpu'])
            if data['gpu'] == 0:
                # Only test bdf and uuid when gpu=0
                self.gpus.append(data['bdf'])
                self.gpus.append(data['uuid'])

        # When parsing, expand each arg with array element
        self.sub_args = \
        {
            'CLOCK': ['SYS','DF','DCEF','SOC','MEM','VCLK0','VCLK1','DCLK0','DCLK1','ALL'],
            'PID': [123],
            'NAME': ['AMD'],
            'GPU': self.gpus,
            'FILE': [self.tmp_filename, f'{self.tmp_filename} --overwrite', f'{self.tmp_filename} --append'],
            'SEVERITY': ['nonfatal-uncorrected', 'fatal', 'nonfatal-corrected', 'all'],
            'FOLDER': [self.tmp_folder],
            'FILE_LIMIT': [10],
            #'LEVEL': ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        }

        self.perf_levels = ['AUTO', 'LOW', 'HIGH', 'MANUAL', 'STABLE_STD', 'STABLE_PEAK', 'STABLE_MIN_MCLK', 'STABLE_MIN_SCLK', 'DETERMINISM']
        self.profile_levels = ['CUSTOM_MASK', 'VIDEO_MASK', 'POWER_SAVING_MASK', 'COMPUTE_MASK', 'VR_MASK', 'THREE_D_FULL_SCR_MASK', 'BOOTUP_DEFAULT']
        self.compute_partition_modes = ['SPX', 'DPX', 'TPX', 'QPX', 'CPX']
        self.memory_partition_modes = ['NPS1', 'NPS2', 'NPS4', 'NPS8']
        self.power_types = ['ppt0', 'ppt1']
        self.ptl_formats = ['I8', 'F16', 'BF16', 'F32', 'F64', 'F8', 'VECTOR']
        self.clk_limits = ['SCLK', 'MCLK']
        self.limit_types = ['MIN', 'MAX']
        self.clk_levels = ['SCLK', 'MCLK', 'FCLK', 'SOCCLK', 'PCIE']

        # When parsing, ignore these entries as they are abnormal
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
            '--append',
            '--overwrite',
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
        lines = std_out.split('\n')

        found = False
        options = []
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
                    if 'Set' in match_str or 'Reset' in match_str or 'RAS' in match_str:
                        pass # These require an option
                    else:
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
                        elif '--json' == items[item_index]:
                            options.append(f'{{json}}')
                            options.append(f'{{json_file}}')
                            options.append(f'{{json_file_append}}')
                            options.append(f'{{json_file_overwrite}}')
                        elif '--csv' == items[item_index]:
                            options.append(f'{{csv}}')
                            options.append(f'{{csv_file}}')
                            options.append(f'{{csv_file_append}}')
                            options.append(f'{{csv_file_overwrite}}')
                        elif '--append' == items[item_index] or '--overwrite' == items[item_index]:
                            pass
                        elif '--watch' == items[item_index]:
                            options.append(f'{{watch_time}}')
                            options.append(f'{{watch_iterations}}')
                        elif '--watch_time' == items[item_index] or '--iterations' == items[item_index]:
                            pass
                        elif '--loglevel' == items[item_index]:
                            pass
                        else:
                            print(f'ERROR: bad sub arg {items[item_index]}')
                    elif len(items) > item_index:
                        if items[item_index+1][0:1] == self.openBracket:
                            items[item_index+1] = items[item_index+1][1:]
                        sub_arg = items[item_index+1]
                        # Expand out sub_args
                        if sub_arg.isupper() and sub_arg in self.sub_args:
                            sub_found = True
                            for item in self.sub_args[sub_arg]:
                                options.append(f'{items[item_index]} {item}')
                        elif 'Set' in match_str:
                            if sub_arg == '%':  # arg --fan
                                options.append(f'{items[item_index]} 50%')
                                options.append(f'{items[item_index]} 150')
                            elif sub_arg == 'LEVEL':  # arg --perf-level
                                for perf_level in self.perf_levels:
                                    options.append(f'{items[item_index]} {perf_level}')
                            elif sub_arg == 'PROFILE_LEVEL':  # arg --profile
                                for profile_level in self.profile_levels:
                                    options.append(f'{items[item_index]} {profile_level}')
                            elif sub_arg == 'SCLKMAX':  # arg --perf-determinism
                                options.append(f'{{perf_determinism}}')
                            elif sub_arg == 'TYPE/INDEX':  # arg
                                for compute_partition_mode in self.compute_partition_modes:
                                    options.append(f'{items[item_index]} {compute_partition_mode}')
                            elif sub_arg == 'PARTITION':  # arg --memory-partition
                                for memory_partition_mode in self.memory_partition_modes:
                                    options.append(f'{items[item_index]} {memory_partition_mode}')
                            elif sub_arg == 'WATTS':  # arg --power-cap
                                for power_type in self.power_types:
                                    options.append(f'--power-cap {{min_power}} {power_type}')
                                    options.append(f'--power-cap {{avg_power}} {power_type}')
                                    options.append(f'--power-cap {{max_power}} {power_type}')
                            elif sub_arg == 'POLICY_ID' and 'soc' in items[item_index]:  # arg --soc-pstate
                                options.append(f'{items[item_index]} {{soc_pstate}}')
                            elif sub_arg == 'POLICY_ID' and 'xgmi' in items[item_index]:  # arg --xgmi-plpd
                                options.append(f'{items[item_index]} {{xgmi_plpd}}')
                            elif sub_arg == 'CLK_TYPE' and 'level' in items[item_index]:  # arg --clk-level
                                options.append(f'{items[item_index]} {{clk_level_sclk}}')
                                options.append(f'{items[item_index]} {{clk_level_mclk}}')
                                options.append(f'{items[item_index]} {{clk_level_fclk}}')
                                options.append(f'{items[item_index]} {{clk_level_socclk}}')
                                options.append(f'{items[item_index]} {{clk_level_pcie}}')
                            elif sub_arg == 'STATUS' and 'ptl' in items[item_index]:  # arg --ptl-status
                                options.append(f'{items[item_index]} 0')
                                options.append(f'{items[item_index]} 1')
                                pass
                            elif sub_arg == 'FRMT1,FRMT2':  # arg --ptl-format
                                for fmt1 in self.ptl_formats:
                                    for fmt2 in self.ptl_formats:
                                        if fmt1 == fmt2:
                                            continue
                                        options.append(f'{items[item_index]} {fmt1},{fmt2}')
                            elif sub_arg == 'CLK_TYPE' and 'limit' in items[item_index]:  # arg --clk-limit
                                options.append(f'{items[item_index]} {{clk_limit_sclk_min}}')
                                options.append(f'{items[item_index]} {{clk_limit_sclk_max}}')
                                options.append(f'{items[item_index]} {{clk_limit_mclk_min}}')
                                options.append(f'{items[item_index]} {{clk_limit_mclk_max}}')
                            elif sub_arg == 'STATUS' and 'process' in items[item_index]:  # arg --process-isolation
                                options.append(f'{items[item_index]} 0')
                                options.append(f'{items[item_index]} 1')
                            else:
                                print(f'TODO: set {items[item_index]} sub_arg={sub_arg}  match_str={match_str}')
                    if not sub_found:
                        # Put in sub_arg if it was not found
                        if 'Set' in match_str:
                            pass
                        else:
                            options.append(items[item_index])
            if match_str in line:
                found = True
        if not options:
            return ['pass']
        return options

    def CreateCmds(self, cmd_name, list1_name, list2_name, list3_name, list4_name):
        cmd = f'amd-smi {cmd_name} --help'
        list1_args = self.FindArgs(cmd, list1_name)
        list2_args = self.FindArgs(cmd, list2_name)
        list3_args = self.FindArgs(cmd, list3_name)
        list4_args = self.FindArgs(cmd, list4_name)
        if self.Debug:
            print(f'{list1_name}: {"*"*80}')
            print(json.dumps(list1_args, sort_keys=False, indent=4), flush=True)
            print(f'{list2_name}: {"*"*80}')
            print(json.dumps(list2_args, sort_keys=False, indent=4), flush=True)
            print(f'{list3_name}: {"*"*80}')
            print(json.dumps(list3_args, sort_keys=False, indent=4), flush=True)
            print(f'{list4_name}: {"*"*80}')
            print(json.dumps(list4_args, sort_keys=False, indent=4), flush=True)

        cmds = []
        cmd = f'amd-smi {cmd_name}'
        for list1_arg in list1_args:
            if list1_arg != 'pass':
                cmds.append((f'{cmd} {list1_arg} {self.AddLogLevel}', self.PASS))
                if not list1_arg:
                    cmds.append((f'{cmd} --file {self.tmp_filename} {self.AddLogLevel}', self.PASS))
                    cmds.append((f'{cmd} {{json}} {self.AddLogLevel}', self.PASS))
                    cmds.append((f'{cmd} {{json_file}} {self.AddLogLevel}', self.PASS))
                    cmds.append((f'{cmd} {{json_file_append}} {self.AddLogLevel}', self.PASS))
                    cmds.append((f'{cmd} {{json_file_overwrite}} {self.AddLogLevel}', self.PASS))
                    cmds.append((f'{cmd} {{csv}} {self.AddLogLevel}', self.PASS))
                    cmds.append((f'{cmd} {{csv_file}} {self.AddLogLevel}', self.PASS))
                    cmds.append((f'{cmd} {{csv_file_append}} {self.AddLogLevel}', self.PASS))
                    cmds.append((f'{cmd} {{csv_file_overwrite}} {self.AddLogLevel}', self.PASS))
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

        # Calculate and substitute in dependent values
        # Removes cmds that are invalid
        for index, cmd_cond in enumerate(cmds):
            cmd, cond = cmd_cond
            while self.openCurlyBrace in cmd:
                items = cmd.split()
                # Find gpu index and mark when gpu=0
                gpu_0 = False
                try:
                    i = items.index('--gpu')
                    gpu = items[i+1]
                    if gpu.isdigit():
                        gpu_index = int(gpu)
                        if gpu_index == 0:
                            gpu_0 = True
                    else:
                        gpu_index = 0
                except ValueError:
                    gpu_index = 0

                # Find conditional arguments
                posOpen = cmd.find(self.openCurlyBrace)
                if posOpen < 0:
                    break
                posClose = cmd.find(self.closeCurlyBrace, posOpen)
                if posClose < 0:
                    break
                nameStr = cmd[posOpen:posClose+1]

                if nameStr == '{json}' or 'json_file' in nameStr or \
                   nameStr == '{csv}' or 'csv_file' in nameStr:
                    # For adding file options
                    if nameStr == '{json}':
                        cmd = cmd.replace(nameStr, '--json', 1)
                    elif nameStr == '{json_file}':
                        cmd = cmd.replace(nameStr, f'--json --file {self.tmp_filename}', 1)
                    elif nameStr == '{json_file_append}':
                        cmd = cmd.replace(nameStr, f'--json --file {self.tmp_filename} --append', 1)
                    elif nameStr == '{json_file_overwrite}':
                        cmd = cmd.replace(nameStr, f'--json --file {self.tmp_filename} --overwrite', 1)
                    elif nameStr == '{csv}':
                        cmd = cmd.replace(nameStr, '--csv', 1)
                    elif nameStr == '{csv_file}':
                        cmd = cmd.replace(nameStr, f'--csv --file {self.tmp_filename}', 1)
                    elif nameStr == '{csv_file_append}':
                        cmd = cmd.replace(nameStr, f'--csv --file {self.tmp_filename} --append', 1)
                    elif nameStr == '{csv_file_overwrite}':
                        cmd = cmd.replace(nameStr, f'--csv --file {self.tmp_filename} --overwrite', 1)
                    else:
                        print(f'Error: could not replace json/csv options, {nameStr}  cmd={cmd}')
                        cmd = ''
                elif nameStr == '{watch_time}' or nameStr == '{watch_iterations}':
                    # For adding watch options
                    if nameStr == '{watch_time}':
                        cmd = cmd.replace(nameStr, '--watch 1 --watch_time 2', 1)
                    else:
                        cmd = cmd.replace(nameStr, '--watch 1 --iterations 2', 1)
                elif nameStr == '{min_power}' or nameStr == '{avg_power}' or nameStr == '{max_power}':
                    # For setting --power-cap
                    # Find power_type
                    for power_type in self.power_types:
                        if power_type in cmd:
                            power_type = self.static_data['gpu_data'][gpu_index]['limit'][power_type]
                        else:
                            power_type = 'N/A'
                    if power_type == 'N/A' or power_type['min_power_limit'] == 'N/A' or power_type['max_power_limit'] == 'N/A':
                        cmd = ''
                    else:
                        min_power = power_type['min_power_limit']['value']
                        max_power = power_type['max_power_limit']['value']
                        avg_power = int((min_power + max_power) / 2)
                        if nameStr == '{min_power}':
                            cmd = cmd.replace('{min_power}', str(min_power), 1)
                        elif nameStr == '{avg_power}':
                            cmd = cmd.replace('{avg_power}', str(avg_power), 1)
                        elif nameStr == '{max_power}':
                            cmd = cmd.replace('{max_power}', str(max_power), 1)
                elif nameStr == '{perf_determinism}':
                    clock_sys = self.static_data['gpu_data'][gpu_index]['clock']['sys']
                    if clock_sys != 'N/A' and len(clock_sys['frequency_levels']):
                        num = len(clock_sys['frequency_levels'])
                        level = f'Level {num-1}'
                        clock_freq = int(clock_sys['frequency_levels'][level].split()[0].strip())
                        cmd = cmd.replace('{perf_determinism}', f'--perf-determinism {clock_freq+50}', 1)
                    else:
                        cmd = ''
                elif 'clk_limit' in nameStr:
                    clock = self.metric_data['gpu_data'][gpu_index]['clock']
                    if nameStr == '{clk_limit_sclk_min}':
                        clk_type = 'SCLK'
                        clk_type_name = 'socclk_0'
                        limit_type = 'MIN'
                        clk_limit_name = 'min_clk'
                    elif nameStr == '{clk_limit_sclk_max}':
                        clk_type = 'SCLK'
                        clk_type_name = 'socclk_0'
                        limit_type = 'MAX'
                        clk_limit_name = 'max_clk'
                    elif nameStr == '{clk_limit_mclk_min}':
                        clk_type = 'MCLK'
                        clk_type_name = 'mem_0'
                        limit_type = 'MAX'
                        clk_limit_name = 'min_clk'
                    elif nameStr == '{clk_limit_mclk_max}':
                        clk_type = 'MCLK'
                        clk_type_name = 'mem_0'
                        limit_type = 'MIN'
                        clk_limit_name = 'max_clk'
                    clk_type_limit_name = clock[clk_type_name][clk_limit_name]
                    if type(clk_type_limit_name) is dict:
                        value = clk_type_limit_name['value']
                        cmd = cmd.replace(nameStr, f'{clk_type} {limit_type} {value}', 1)
                    else:
                        cmd = ''
                elif 'clk_level' in nameStr:
                    clock = self.static_data['gpu_data'][gpu_index]['clock']
                    value = -1
                    if nameStr == '{clk_level_sclk}':
                        clk_type = 'SCLK'
                        clk_type_name = 'sys'
                    elif nameStr == '{clk_level_mclk}':
                        clk_type = 'MCLK'
                        clk_type_name = 'mem'
                    elif nameStr == '{clk_level_fclk}':
                        clk_type = 'FCLK'
                        clk_type_name = 'df'
                    elif nameStr == '{clk_level_socclk}':
                        clk_type = 'SOCCLK'
                        clk_type_name = 'soc'
                    elif nameStr == '{clk_level_pcie}':
                        bus = self.static_data['gpu_data'][gpu_index]['bus']
                        clk_type = 'PCIE'
                        pcie_levels = bus['pcie_levels']
                        if type(pcie_levels) is dict:
                            value = len(pcie_levels)
                            if value > 0:
                                value = 0
                    if clk_type != 'PCIE' and value < 0:
                        clk_type_name = clock[clk_type_name]
                        if type(clk_type_name) is dict:
                            current_level = clk_type_name['current_level']
                            freq_levels = clk_type_name['frequency_levels']
                            if current_level == 0:
                                value = len(freq_levels) - 1
                            else:
                                value = 0
                    if value >= 0:
                        cmd = cmd.replace(nameStr, f'{clk_type} {value}', 1)
                    else:
                        cmd = ''
                elif nameStr == '{soc_pstate}':
                    soc_pstate = self.static_data['gpu_data'][gpu_index]['soc_pstate']
                    if type(soc_pstate) is dict:
                        num_supported = int(soc_pstate['num_supported'])
                        if num_supported > 0:
                            current = int(soc_pstate['current_id'])
                            if current == 0:
                                num = num_supported - 1
                            else:
                                num = 0
                            cmd = cmd.replace(nameStr, f'{num}', 1)
                        else:
                            cmd = ''
                    else:
                        cmd = ''
                elif nameStr == '{xgmi_plpd}':
                    xgmi_plpd = self.static_data['gpu_data'][gpu_index]['xgmi_plpd']
                    if type(xgmi_plpd) is dict:
                        num_supported = int(xgmi_plpd['num_supported'])
                        if num_supported > 0:
                            current = int(xgmi_plpd['current_id'])
                            if current == 0:
                                num = num_supported - 1
                            else:
                                num = 0
                            cmd = cmd.replace(nameStr, f'{num}', 1)
                        else:
                            cmd = ''
                    else:
                        cmd = ''
            cmds[index] = (cmd, cond)


        # Pare down commands
        if self.ReduceCmds:
            file_mods = ['--file', '--json', '--csv']
            watch_mods = ['--watch', '--watch_time', '--iterations']

            found_sub_arg = False
            for index, cmd_cond in enumerate(cmds):
                cmd, cond = cmd_cond
                items = cmd.split()

                # Find the first sub_arg
                if not found_sub_arg and len(items) >= 3:
                    sub_arg = items[2]
                    for mod in file_mods + ['--gpu', '--loglevel']:
                        if mod == sub_arg:
                            sub_arg = ''
                            break
                    found_sub_arg = sub_arg

                # No explicit gpu infers a gpu=0
                gpu_index = '0'
                if '--gpu' in cmd:
                    try:
                        i = items.index('--gpu')
                        gpu_index = items[i+1]
                    except ValueError as e:
                        # condition where --gpu is not in the cmd
                        # will get default gpu_index=0
                        pass

                # Remove all --gpu for all sub_args except for the first sub_arg
                if cmd and found_sub_arg:
                    sub_arg = items[2]
                    if sub_arg != found_sub_arg:
                        if '--gpu' in cmd:
                            cmd = ''


                # Remove all file and watch modifiers except for gpu 0
                if cmd and gpu_index != '0':
                    for mod in file_mods + watch_mods:
                        if mod in cmd:
                            cmd = ''
                            break

                # Remove all --file and --watch combinations
                if cmd and '--file' in cmd and '--watch' in cmd:
                    cmd = ''

                # Remove all --watch mod for all sub_args except for the first sub_arg
                if cmd and found_sub_arg and len(items) >= 3:
                    sub_arg = items[2]
                    if sub_arg != found_sub_arg:
                        if '--watch' in cmd:
                            cmd = ''

                # Remove all file mod for all sub_args except for the first sub_arg
                if cmd and found_sub_arg and len(items) >= 3:
                    sub_arg = items[2]
                    if sub_arg != found_sub_arg:
                        for mod in file_mods:
                            if mod in cmd:
                                cmd = ''
                                break

                cmds[index] = (cmd, cond)

        # Remove empty (cmd,cond) arguments
        cmds = [cmd_cond for cmd_cond in cmds if cmd_cond[0] != '']

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
        msg_len = 0
        for cmd, cond in cmds:
            num = len(cmd)
            if num > msg_len:
                msg_len = num
        msg_len += 2
        for cmd, cond in cmds:
            if self.Debug or self.PrintCmdsOnly:
                print(f'cmd={cmd}')
            if self.PrintCmdsOnly:
                continue
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

            msg=f'{cmd:{msg_len}s}:'
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
        return

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
                continue
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

        # Create bad bdf and uuid gpus
        bdf = self.list_data[0]['bdf']
        if bdf[-1] == '0':
            bad_bdf = self.list_data[0]['bdf'][:-1] + '1'
        else:
            bad_bdf = self.list_data[0]['bdf'][:-1] + '0'
        uuid = self.list_data[0]['uuid']
        if uuid[-1] == '0':
            bad_uuid = self.list_data[0]['uuid'][:-1] + '1'
        else:
            bad_uuid = self.list_data[0]['uuid'][:-1] + '0'

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
            ('amd-smi reset', self.FAIL),
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
            (f'amd-smi static --gpu {bad_bdf}', self.FAIL),
            (f'amd-smi static --gpu {self.list_data[0]["bdf"][:-1]}', self.FAIL),
            (f'amd-smi static --gpu {self.list_data[0]["bdf"] + "0"}', self.FAIL),
            (f'amd-smi static --gpu {bad_uuid}', self.FAIL),
            (f'amd-smi static --gpu {self.list_data[0]["uuid"][:-1]}', self.FAIL),
            (f'amd-smi static --gpu {self.list_data[0]["uuid"] + "0"}', self.FAIL),
            # Test invalid loglevel
            ('amd-smi metric --loglevel DDEBUG', self.FAIL),
            ('amd-smi metric --loglevel DEBUGG', self.FAIL),
            ('amd-smi metric --loglevel BADLEVEL', self.FAIL),
            # Test invalid set options
            ('amd-smi set', self.FAIL),
            ('amd-smi set --fan', self.FAIL),
            ('amd-smi set --fan 500', self.FAIL),
            ('amd-smi set --fan 150%', self.FAIL),
            ('amd-smi set --perf-level', self.FAIL),
            ('amd-smi set --perf-level INVALID', self.FAIL),
            ('amd-smi set --profile', self.FAIL),
            ('amd-smi set --profile INVALID', self.FAIL),
            ('amd-smi set --perf-determinism', self.FAIL),
            ('amd-smi set --compute-partition', self.FAIL),
            ('amd-smi set --compute-partition INVALID', self.FAIL),
            ('amd-smi set --memory-partition', self.FAIL),
            ('amd-smi set --memory-partition NPS3', self.FAIL),
            ('amd-smi set --memory-partition INVALID', self.FAIL),
            ('amd-smi set --process-isolation', self.FAIL),
            ('amd-smi set --process-isolation 2', self.FAIL),
            ('amd-smi set --clk-limit', self.FAIL),
            ('amd-smi set --clk-limit INVALID', self.FAIL),
            ('amd-smi set --clk-limit SCLK INVALID', self.FAIL),
            ('amd-smi set --clk-limit MCLK INVALID', self.FAIL),
            ('amd-smi set --clk-limit SCLK MIN', self.FAIL),
            ('amd-smi set --clk-limit MCLK MAX', self.FAIL),
            ('amd-smi set --clk-level SCLK', self.FAIL),
            ('amd-smi set --clk-level SCLK INVALID', self.FAIL),
            ('amd-smi set --clk-level MCLK', self.FAIL),
            ('amd-smi set --clk-level MCLK INVALID', self.FAIL),
            ('amd-smi set --clk-level FCLK', self.FAIL),
            ('amd-smi set --clk-level FCLK INVALID', self.FAIL),
            ('amd-smi set --clk-level SOCCLK', self.FAIL),
            ('amd-smi set --clk-level SOCCLK INVALID', self.FAIL),
            ('amd-smi set --clk-level PCIE', self.FAIL),
            ('amd-smi set --clk-level PCIE INVALID', self.FAIL),
            # Test invalid process PID, NAME
            ('amd-smi process --name', self.FAIL),
            ('amd-smi process --pid', self.FAIL),
            ('amd-smi process --pid NOT_A_NUMBER', self.FAIL),
            # Test invalid ras options
            ('amd-smi ras', self.FAIL),
            ('amd-smi ras --cper INVALID', self.FAIL),
            ('amd-smi ras --cper --severity INVALID', self.FAIL),
            ('amd-smi ras --afid', self.FAIL),
            ('amd-smi ras --afid INVALID', self.FAIL),
            # Test invalid watch order
            ('amd-smi monitor --interval 2 --watch 1', self.FAIL),
            ('amd-smi monitor --watch_time 2 --watch 1', self.FAIL),
        ]

        for index, gpu in enumerate(self.common.processors):
            # Test invalid power-cap values
            cmds.append((f'amd-smi set --power-cap --gpu {index}', self.FAIL))
            for power_type in self.power_types:
                cmds.append((f'amd-smi set --power-cap {power_type} --gpu {index}', self.FAIL))
                _power_type = self.static_data['gpu_data'][index]['limit'][power_type]
                socket_power_limit = _power_type['socket_power_limit']
                if socket_power_limit != 'N/A':
                    min_power = _power_type['min_power_limit']['value']
                    max_power = _power_type['max_power_limit']['value']
                    cmds.append((f'amd-smi set --power-cap {min_power - 1} {power_type} --gpu {index}', self.FAIL))
                    cmds.append((f'amd-smi set --power-cap {max_power + 1} {power_type} --gpu {index}', self.FAIL))
                    cmds.append((f'amd-smi set --power-cap {int(max_power * 1.10)} {power_type} --gpu {index}', self.FAIL))

            # Test invalid soc-pstate values
            soc_pstate = self.static_data['gpu_data'][index]['soc_pstate']
            if soc_pstate != 'N/A':
                cmds.append((f'amd-smi set --soc-pstate --gpu {index}', self.FAIL))
                num_supported = int(soc_pstate['num_supported'])
                cmds.append((f'amd-smi set --soc-pstate {num_supported} --gpu {index}', self.FAIL))

            # Test invalid xgmi-plpd values
            xgmi_plpd = self.static_data['gpu_data'][index]['xgmi_plpd']
            if xgmi_plpd != 'N/A':
                cmds.append((f'amd-smi set --xgmi-plpd --gpu {index}', self.FAIL))
                num_supported = int(xgmi_plpd['num_supported'])
                cmds.append((f'amd-smi set --xgmi-plpd {num_supported} --gpu {index}', self.FAIL))

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

        # TODO allow event commands to be executed
        if not self.PrintCmdsOnly:
            if self.common.TODO_SKIP_FAIL:
                msg = f'{self.tab}Needs input'
                self.common.print(msg)
                self.skipTest(msg)

        # Start process with "amd-smi event"
        # In another process create an event with like "amd-smi reset --gpureset"
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

        # TODO allow set commands to be executed
        if not self.PrintCmdsOnly:
            if self.common.TODO_SKIP_FAIL:
                msg = f'{self.tab}Needs input'
                #self.common.print(msg)
                self.skipTest(msg)

        # Get current settings
        power_profile = {}
        for index, gpu in enumerate(self.common.processors):
            try:
                power_profile[index] = amdsmi.amdsmi_get_gpu_power_profile_presets(gpu, 0)
            except amdsmi.AmdSmiLibraryException as e:
                power_profile[index] = None

        cmds = self.CreateCmds('set', 'Set Arguments:', 'Device Arguments:', 'Command Modifiers:', '')
        self.RunCmds(cmds)

        # Restore starting values
        cmds = []
        for index, gpu in enumerate(self.common.processors):
            # set --fan defaults
            fan_speed = self.metric_data['gpu_data'][index]['fan']['speed']
            if fan_speed != 'N/A':
                cmds.append((f'amd-smi set --fan {fan_speed} --gpu {index}', self.PASS))
        
            # set --perf-level defaults
            perf_level = self.metric_data['gpu_data'][index]['perf_level']
            if perf_level != 'N/A':
                perf_level = perf_level.removeprefix('AMDSMI_DEV_PERF_LEVEL_')
                cmds.append((f'amd-smi set --perf-level {perf_level} --gpu {index}', self.PASS))

            # set --profile defaults
            if power_profile[index]:
                profile = power_profile[index]['current'].removeprefix('AMDSMI_PWR_PROF_PRST_')
                cmds.append((f'amd-smi set --profile {profile} --gpu {index}', self.PASS))

            # set --perf-determinism defaults
            clock_sys = self.static_data['gpu_data'][index]['clock']['sys']
            if clock_sys != 'N/A':
                num = len(clock_sys['frequency_levels'])
                level = f'Level {num-1}'
                clock_freq = int(clock_sys['frequency_levels'][level].split()[0].strip())
                cmds.append((f'amd-smi set --perf-determinism {clock_freq} --gpu {index}', self.PASS))

            # set --compute-partition defaults
            accelerator_type = self.partition_data['current_partition'][index]['accelerator_type']
            if accelerator_type != 'N/A':
                cmds.append((f'amd-smi set --compute-partition {accelerator_type} --gpu {index}', self.PASS))

            # set --memory-partition defaults
            memory_partition = self.partition_data['current_partition'][index]['memory']
            if memory_partition != 'N/A':
                cmds.append((f'amd-smi set --memory-partition {memory_partition} --gpu {index}', self.PASS))
        
            # set --power-cap defaults
            for power_type in self.power_types:
                socket_power_limit = self.static_data['gpu_data'][index]['limit'][power_type]['socket_power_limit']
                if socket_power_limit != 'N/A':
                    socket_power = socket_power_limit['value']
                    cmds.append((f'amd-smi set --power-cap {socket_power} {power_type} --gpu {index}', self.PASS))
        
            # set --soc-pstate defaults
            soc_pstate = self.static_data['gpu_data'][index]['soc_pstate']
            if soc_pstate != 'N/A':
                current = int(soc_pstate['current'])
                cmds.append((f'amd-smi set --soc-pstate {current} --gpu {index}', self.PASS))
        
            # set --xgmi-plpd defaults
            xgmi_plpd = self.static_data['gpu_data'][index]['xgmi_plpd']
            if xgmi_plpd != 'N/A':
                current = int(xgmi_plpd['current'])
                cmds.append((f'amd-smi set --xgmi-plpd {current} --gpu {index}', self.PASS))
        
            # set --ptl-status defaults
            ptl_state = self.static_data['gpu_data'][index]['limit']['ptl_state']
            if ptl_state != 'N/A':
                if ptl_state == 'Disabled':
                    ptl_state_value = 0
                else:
                    ptl_state_value = 1
                cmds.append((f'amd-smi set --ptl-status {ptl_state_value} --gpu {index}', self.PASS))

            # set --ptl-format defaults
            ptl_format = self.static_data['gpu_data'][index]['limit']['ptl_format']
            if ptl_format != 'N/A':
                # TODO: get the right ptl-format
                cmds.append((f'amd-smi set --ptl-format {ptl_format} --gpu {index}', self.PASS))
        
            # set --clk-limit defaults
            clock = self.metric_data['gpu_data'][index]['clock']
            for clk_type in self.clk_limits:
                if clk_type == 'SCLK':
                    clk_type_name = 'socclk_0'
                else:
                    clk_type_name = 'mem_0'
                for limit_type in self.limit_types:
                    if limit_type == 'MIN':
                        clk_limit_name = 'min_clk'
                    else:
                        clk_limit_name = 'max_clk'
                    clk_type_limit_name = clock[clk_type_name][clk_limit_name]
                    if type(clk_type_limit_name) is dict:
                        value = clk_type_limit_name['value']
                        cmds.append((f'amd-smi set --clk-limit {clk_type} {limit_type} {value} --gpu {index}', self.PASS))

            # set --clk-level defaults
            clock = self.static_data['gpu_data'][index]['clock']
            for clk_type in self.clk_levels:
                value = -1
                if clk_type == 'SCLK':
                    clk_type_name = 'sys'
                elif clk_type == 'MCLK':
                    clk_type_name = 'mem'
                elif clk_type == 'FCLK':
                    clk_type_name = 'df'
                elif clk_type == 'SOCCLK':
                    clk_type_name = 'soc'
                else:
                    bus = self.static_data['gpu_data'][index]['bus']
                    pcie_levels = bus['pcie_levels']
                    if type(pcie_levels) is dict:
                        value = len(pcie_levels)
                        if value > 0:
                            value -= 1
                if clk_type != 'PCIE' and value < 0:
                    clk_type_name = clock[clk_type_name]
                    if type(clk_type_name) is dict:
                        current_level = clk_type_name['current_level']
                        value = current_level
                if value >= 0:
                    cmds.append((f'amd-smi set --clk-level {clk_type} {value} --gpu {index}', self.PASS))
            # set --process-isolation defaults
            process_isolation = self.static_data['gpu_data'][index]['process_isolation']
            if process_isolation == 'Disabled':
                process_isolation_value = 0
            else:
                process_isolation_value = 1
            cmds.append((f'amd-smi set --process-isolation {process_isolation_value} --gpu {index}', self.PASS))

        print('Restore Starting Values')
        self.RunCmds(cmds)

        return

    def test_reset(self):
        self.common.print_func_name('')
        msg = f'{self.tab}### amd-smi reset'
        self.common.print(msg)

        # TODO allow reset commands to be executed
        if not self.PrintCmdsOnly:
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

        # TODO Yazen
        # TODO allow event commands to be executed
        if not self.PrintCmdsOnly:
            if self.common.TODO_SKIP_FAIL:
                msg = f'{self.tab}Not Yet Implemented'
                #self.common.print(msg)
                self.skipTest(msg)

        cmds = self.CreateCmds('ras', 'RAS arguments:', 'CPER Arguments', 'Device Arguments:', 'Command Modifiers:')
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

