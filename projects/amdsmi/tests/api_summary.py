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
import os
import pathlib
import sys
import textwrap

version_number = '1.0.0'
build_date = f'{datetime.datetime.now():%b %d %Y}'
verbose_choices = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL', 'ALWAYS']

not_supported_error_codes = \
[
    ( '2', 'AMDSMI_STATUS_NOT_SUPPORTED'),
    ( '3', 'AMDSMI_STATUS_NOT_YET_IMPLEMENTED'),
    ('49', 'AMDSMI_STATUS_NO_HSMP_MSG_SUP')
]

open_parenthesis = '('
close_parenthesis = ')'
colon = ':'

header = 'any C(any, func, unit) py(any, func, unit)'


def Print(cond, msg, end=None):
    if isinstance(cond, str):  # Check verbose level
        if cond in verbose_choices:
            cond_num = verbose_choices.index(cond)
            if cond_num < args.verbose_num:
                return
    elif not cond:  # Check cond
        return

    if end == None:
        print(msg)
    else:
        print(msg, end)
    return


def Find_Root_Path():
    last_path = pathlib.Path('/')
    root_path = pathlib.Path.cwd()
    while(True):
        git_path = root_path / '.git'
        if git_path.exists():
            return root_path / 'projects/amdsmi'
        if root_path == last_path:
            break
        root_path = root_path.parent
    return None


def Check_Inputs(amdsmi, file_names):
    root_path = Find_Root_Path()

    # Check for header file
    amdsmi_path = pathlib.Path(amdsmi)
    if not amdsmi_path.exists():
        if root_path:
            amdsmi_path = root_path / amdsmi_path
            if not amdsmi_path.exists():
                amdsmi_path = None
        else:
            amdsmi_path = None

    # Check for any log files
    file_paths = {}
    missing_file_paths = {}
    for key, file_name in file_names.items():
        file_path = args.log_dir / pathlib.Path(file_name)
        if file_path.exists():
            file_paths[key] = file_path
        else:
            if root_path:
                file_path = root_path / args.log_dir / file_name
                if file_path.exists():
                    file_paths[key] = file_path
                else:
                    missing_file_paths[key] = file_path
            else:
                missing_file_paths[key] = file_path
    return (amdsmi_path, file_paths, missing_file_paths)


def ReadAmdsmiHeader(file_content):
    api_map = {}

    end_block_pos = 0
    while (True):
        start_block_pos = file_content.find('@ingroup', end_block_pos)
        if start_block_pos == -1:
            break

        end_block_pos = file_content.find(f'{close_parenthesis};', start_block_pos)
        if end_block_pos == -1:
            Print('ERROR', f'Could not find func definition end, {start_block_pos}')
            break

        end_pos = file_content.rfind(open_parenthesis, start_block_pos, end_block_pos)
        start_pos = file_content.rfind('amdsmi_', start_block_pos, end_pos)
        func_name = file_content[start_pos:end_pos].strip()
        api_map[func_name] = 0

    return api_map


def Find_Name(line, start, name_start, name_end, cof):
    if start in line:
        pos_start = line.find(start)
        pos_start = line.find(name_start, pos_start)
        if pos_start == -1:
            Print('DEBUG', f'start: Bad "{start}" definition, {line}')
            return None
        pos_end = line.find(name_end, pos_start)
        if pos_end == -1:
            if not cof:
                Print('DEBUG', f'end: Bad "{start}" definition, {line}')
                return None
            pos_end = len(line)

        data = line[pos_start:pos_end].strip()
        return data
    return None


def ReadTestingInput(file_contents):
    # api_map[file_name][func_name] = number_times_called
    api_map = {}
    api_support_map = {}

    for file_name in file_contents:
        api_map[file_name] = {}
        for index, line in enumerate(file_contents[file_name]):
            # Finding API name
            func_name = Find_Name(line, '### amdsmi_', 'amdsmi_', open_parenthesis, False)
            if func_name:
                if func_name in api_map[file_name]:
                    api_map[file_name][func_name] += 1
                else:
                    api_map[file_name][func_name] = 1

                # Finding API that are Not Supported
                line = file_contents[file_name][index+1]
                status = Find_Name(line, 'AMDSMI API Returned', 'AMDSMI_', colon, True)
                if status:
                    if func_name in api_support_map and api_support_map[func_name]:
                        # Once API is supported, it cannot be unsupported
                        api_support_map[func_name] += 1
                        pass
                    else:
                        # API is not supported
                        if any(status in value for value in not_supported_error_codes):
                            api_support_map[func_name] = 0
                else:
                    # API is supported
                    api_support_map[func_name] = 1 

    return (api_map, api_support_map)


def Main(amdsmi, file_names):
    amdsmi_path, file_paths, missing_file_paths = Check_Inputs(amdsmi, file_names)
    if not amdsmi_path:
        Print('WARNING', f'Missing header, {amdsmi}')
    for key, file_path in missing_file_paths.items():
        Print('WARNING', f'Missing file, {key}={file_path}')

    if amdsmi_path:
        Print('INFO', f'Using header, {amdsmi_path}')
    for key, file_path in file_paths.items():
        Print('INFO', f'Using file, {key}={file_path}')

    if not amdsmi_path and not len(file_paths):
        Print('ERROR', 'No header or log files found, exiting script')
        return 1

    # Header file is stored as a string
    amdsmi_content = amdsmi_path.read_text()

    # Each log file is stored as a [string]
    file_contents = {}
    for key, file_name in file_paths.items():
        content = file_name.read_text()
        file_contents[key] = content.split('\n')

    # Read in header input
    amdsmi_map = {}
    if amdsmi_content:
        amdsmi_map = ReadAmdsmiHeader(amdsmi_content)
    num_api = len(amdsmi_map)
    if not num_api:
        num_api = 1  # set so code does not divide by zero
        Print('DEBUG', 'No header APIs found')
    Print('DEBUG', f'amdsmi APIs = {num_api}')
    for func_name in amdsmi_map:
        Print('DEBUG', f'\tfunc: {func_name}')

    # Read in testing inputs
    api_map, api_support_map = ReadTestingInput(file_contents)
    found = False
    for file_name in api_map:
        if len(api_map[file_name]):
            found = True
    if not found:
        Print('ERROR', 'No testing APIs found')
        return 1
    for file_name in api_map:
        Print('DEBUG', f'Tested {file_name} funcs = {len(api_map[file_name])}')
        for func_name in api_map[file_name]:
            Print('DEBUG', f'\t{func_name}()')

    if not args.output_dir.exists():
        args.output_dir.mkdir(parents=True)

    api_summary_support = []
    if len(api_support_map):
        sorted_map = {}
        sorted_keys = sorted(api_support_map.keys())
        for key in sorted_keys:
            sorted_map[key] = api_support_map[key]

        num_supported = 0
        num_not_supported = 0
        for func_name, supported in sorted_map.items():
            if supported:
                num_supported += 1
            else:
                num_not_supported += 1

        api_summary_support.append(f'API Not Supported: {num_not_supported}')
        for func_name, supported in sorted_map.items():
            if not supported:
                api_summary_support.append(f'\t{func_name}()')
        api_summary_support.append(f'API Supported: {num_supported}')
        for func_name, supported in sorted_map.items():
            if supported:
                api_summary_support.append(f'\t{func_name}()')
        api_summary_support.append(f'API Not Supported: {num_not_supported}')
        api_summary_support.append(f'API     Supported: {num_supported}')
        api_summary_support.append(f'API         Total: {len(sorted_map)}')
    api_summary_support = '\n'.join(api_summary_support)
    api_summary_support_txt = pathlib.Path(args.output_dir / '_api_summary_support.txt')
    api_summary_support_txt.write_text(api_summary_support)

    # Initialize
    for func_name in amdsmi_map:
        amdsmi_map[func_name] = {'tested':0, 'c_unit_test':0, 'c_integration':0, 'py_unit_test':0, 'py_integration':0}
    # api_map[file_name][func_name] = number_times_called
    for file_name in api_map:
        for func_name in api_map[file_name]:
            amdsmi_map[func_name]['tested'] += 1
            amdsmi_map[func_name][file_name] = 1

    api_summary = []
    msg = f'API, Tested, c_unit_test, c_integration, py_unit_test, py_integration'
    api_summary.append(msg)
    for func_name, tests_map in amdsmi_map.items():
        msg = f'{func_name}, {tests_map["tested"]}, {tests_map["c_unit_test"]}, {tests_map["c_integration"]}, {tests_map["py_unit_test"]}, {tests_map["py_integration"]}'
        api_summary.append(msg)
    api_summary.append('')
    api_summary = '\n'.join(api_summary)
    api_summary_csv = pathlib.Path(args.output_dir / '_api_summary.csv')
    api_summary_csv.write_text(api_summary)


    c_unit_test_total = 0
    c_integration_total = 0
    c_any_total = 0
    py_unit_test_total = 0
    py_integration_total = 0
    py_any_total = 0
    any_total = 0
    integration_total = 0
    unit_test_total = 0
    for func_name, values in amdsmi_map.items():
        tested = values['tested']
        if tested:
            c_unit_test = values['c_unit_test']
            c_integration = values['c_integration']
            c_any = 0
            if c_unit_test or c_integration:
                c_any = 1
                c_any_total += 1
                if c_unit_test:
                    c_unit_test_total += 1
                if c_integration:
                    c_integration_total += 1

            py_unit_test = values['py_unit_test']
            py_integration = values['py_integration']
            py_any = 0
            if py_unit_test or py_integration:
                py_any = 1
                py_any_total += 1
                if py_unit_test:
                    py_unit_test_total += 1
                if py_integration:
                    py_integration_total += 1

            if c_integration or py_integration:
                integration_total += 1

            if c_unit_test or py_unit_test:
                unit_test_total += 1

            if c_unit_test or c_integration or py_unit_test or py_integration:
                any_total += 1

            if tested == 0:
                Print('DEBUG', f'Missing:', end='')
            else:
                Print('DEBUG', f'  Found:', end='')
            Print('DEBUG', f' {func_name}: C(any={c_any} unit={c_unit_test} int={c_integration}) py(any={py_any} unit={py_unit_test} int={py_integration})')

    # Summary Table
    #
    #API          Test(%)                    Func(%)                  Unit(%)
    #C      c_any_total(XX.X)  c_integration_total(XX.X)  c_unit_test_total(XX.X)
    #Py    py_any_total(XX.X) py_integration_total(XX.X) py_unit_test_total(XX.X)
    #Total    any_total(XX.X)    integration_total(XX.X)    unit_test_total(XX.X)
    #
    #Total API's: <Num>
    api_summary_table = []

    c_any_total_percent = (c_any_total / num_api) * 100
    c_unit_test_total_percent = (c_unit_test_total / num_api) * 100
    c_integration_total_percent = (c_integration_total / num_api) * 100

    py_any_total_percent = (py_any_total / num_api) * 100
    py_unit_test_total_percent = (py_unit_test_total / num_api) * 100
    py_integration_total_percent = (py_integration_total / num_api) * 100

    any_total_percent = (any_total / num_api) * 100
    unit_test_total_percent = (unit_test_total / num_api) * 100
    integration_total_percent = (integration_total / num_api) * 100

    def PrintLine(val1, num1, val2, num2, val3, num3, val4, num4):
        return(f'{val1:^{num1}s} {val2:^{num2}s} {val3:^{num3}s} {val4:^{num4}s}')
    def PrintLine2(val, num, val1a, num1a, val1b, num1b, val1c, num1c, val2a, num2a, val2b, num2b, val2c, num2c, val3a, num3a, val3b, num3b, val3c, num3c):
        return(f'{val:^{num}} {val1a:{num1a}s}{val1b:{num1b}d}({val1c:{num1c}f}) {val2a:{num2a}s}{val2b:{num2b}d}({val2c:{num2c}f}) {val3a:{num3a}s}{val3b:{num3b}d}({val3c:{num3c}f})')

    size_d = 3
    size_f = 4.1
    space1 = 5
    space2 = 1
    space3 = 11
    msg = PrintLine('API', space1, 'Test(%)', space3, 'Unit(%)', space3, 'Func(%)', space3)
    api_summary_table.append(msg)
    msg = PrintLine2('C', space1,
        ' ', space2, c_any_total,         size_d, c_any_total_percent,         size_f,
        ' ', space2, c_unit_test_total,   size_d, c_unit_test_total_percent,   size_f,
        ' ', space2, c_integration_total, size_d, c_integration_total_percent, size_f)
    api_summary_table.append(msg)
    msg = PrintLine2('Py', space1, ' ', space2, py_any_total, size_d, py_any_total_percent, size_f, ' ', space2, py_unit_test_total, size_d, py_unit_test_total_percent, size_f, ' ', space2, py_integration_total, size_d, py_integration_total_percent, size_f)
    api_summary_table.append(msg)
    msg = PrintLine2('Total', space1, ' ', space2, any_total, size_d, any_total_percent, size_f, ' ', space2, unit_test_total, size_d, unit_test_total_percent, size_f, ' ', space2, integration_total, size_d, integration_total_percent, size_f)
    api_summary_table.append(msg)
    api_summary_table.append(f'Num APIs: {num_api}')
    api_summary_table.append('')

    api_summary_table = '\n'.join(api_summary_table)
    api_summary_table_txt = pathlib.Path(args.output_dir / '_api_summary_table.txt')
    api_summary_table_txt.write_text(api_summary_table)

    return 0


def Parse_Command_Line(cmds=None):
    class CalledAction(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            setattr(namespace, self.dest, values)
            setattr(namespace, f'{self.dest}_called', True)

    msg_description = 'Create API coverage report for unit_test.py and integration_test.py tests'
    msg_epilog = 'Example:\n\t%(prog)s --c_unit_test <c_unit_test.log> --py_integration_test <py_integration_test.log>'
    parser = argparse.ArgumentParser(description=msg_description, formatter_class=argparse.RawTextHelpFormatter, epilog=textwrap.dedent(msg_epilog))

    parser_header = parser.add_argument_group('Information')
    parser.add_argument('--version', action='version', version=version_number, help='Show version and exit')
    parser.add_argument('--build', action='version', version=build_date, help='Show build and exit')
    parser.add_argument('--verbose', choices=verbose_choices , type=str, default='WARNING', help='Level of information to output, default=%(default)s')

    parser_header = parser.add_argument_group('Header File')
    parser_header.add_argument('--amdsmi', default='include/amd_smi/amdsmi.h', help='Path to header file, default=%(default)s')
    parser_logs = parser.add_argument_group('Log Files')
    parser_logs.add_argument('--log_dir', default='build', help='Path to where logs exist, default=%(default)s')
    parser_logs.add_argument('--c_unit_test', default='_c_unit_test.log', help='Filename for C unit_test output, default=%(default)s')
    parser_logs.add_argument('--c_integration', default='_amdsmitst.log', help='Filename for C integration_test output, default=%(default)s')
    parser_logs.add_argument('--py_unit_test', default='_unit_test.log', help='Filename for Python unit_test output, default=%(default)s')
    parser_logs.add_argument('--py_integration', default='_integration_test.log', help='Filename for Python integration_test output, default=%(default)s')
    parser_output = parser.add_argument_group('Output File')
    parser_output.add_argument('--output_dir', default='build', action=CalledAction, help='Path to output dir, will create if does not exist, default=%(default)s')

    if cmds:
        args = parser.parse_args(cmds.split())
    else:
        args = parser.parse_args()

    args.verbose_num = verbose_choices.index(args.verbose)

    if getattr(args, 'output_dir_called', False):
        args.output_dir = pathlib.Path(args.output_dir)
    else:
        root_path = Find_Root_Path()
        args.output_dir = root_path / args.output_dir

    args.file_names = {}
    args.file_names['c_unit_test'] = args.c_unit_test
    args.file_names['c_integration'] = args.c_integration
    args.file_names['py_unit_test'] = args.py_unit_test
    args.file_names['py_integration'] = args.py_integration

    return args


if __name__ == '__main__':
    rc = 1
    args = Parse_Command_Line()
    rc = Main(args.amdsmi, args.file_names)
    sys.exit(rc)

