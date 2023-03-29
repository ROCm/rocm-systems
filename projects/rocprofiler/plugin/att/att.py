#!/usr/bin/env python3
import sys
if sys.version_info[0] < 3:
    raise Exception("Must be using Python 3")

import os
import argparse
from pathlib import Path
from struct import *
from ctypes import *
import ctypes
from copy import deepcopy
from trace_view import view_trace, Readable
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

class FileBytesIO:
    def __init__(self, iobytes) -> None:
        self.iobytes = iobytes
        self.seek = 0

    def __len__(self):
        return self.iobytes.getbuffer().nbytes

    def read(self, length=0):
        if length<=0:
            return bytes(self.getbuffer())
        else:
            if self.seek >= len(self):
                self.seek = 0
                return None
            response =  self.iobytes.getbuffer()[self.seek:self.seek+length]
            self.seek += length
            return bytes(response)


COUNTERS_MAX_CAPTURES = 1<<12

class PerfEvent(ctypes.Structure):
    _fields_ = [
        ('time', c_uint64),
        ('event0', c_uint16),
        ('event1', c_uint16),
        ('event2', c_uint16),
        ('event3', c_uint16),
        ('cu', c_uint8),
        ('bank', c_uint8),
    ]
    def toTuple(self):
        return (int(self.time), int(self.event0), int(self.event1),
            int(self.event2), int(self.event3), int(self.cu), int(self.bank))


class CodeWrapped(ctypes.Structure):
    """ Matches CodeWrapped on the python side """
    _fields_ = [('line', ctypes.c_char_p),
                            ('loc', ctypes.c_char_p),
                            ('value', ctypes.c_int),
                            ('to_line', ctypes.c_int),
                            ('index', ctypes.c_int),
                            ('line_num', ctypes.c_int)]


class KvPair(ctypes.Structure):
    """ Matches pair<int, int> = (key, value) on the python side """
    _fields_ = [('key', ctypes.c_int),
                            ('value', ctypes.c_int)]


class ReturnAssemblyInfo(ctypes.Structure):
    """ Matches ReturnAssemblyInfo on the python side """
    _fields_ = [('code', POINTER(CodeWrapped)),
                            ('jumps', POINTER(KvPair)),
                            ('code_len', ctypes.c_int),
                            ('jumps_len', ctypes.c_int)]


class Wave(ctypes.Structure):
    _fields_ = [
        ('simd', ctypes.c_uint64),
        ('wave_id', ctypes.c_uint64),
        ('begin_time', ctypes.c_uint64),  # Begin and end cycle
        ('end_time', ctypes.c_uint64),

        # total VMEM/FLAT/LDS/SMEM instructions issued
        # total issued memory instructions
        ('num_mem_instrs', ctypes.c_uint64),
        # total issued instructions (compute + memory)
        ('num_issued_instrs', ctypes.c_uint64),
        ('num_valu_instrs', ctypes.c_uint64),
        ('num_valu_stalls', ctypes.c_uint64),
        # VMEM Pipeline: instrs and stalls
        ('num_vmem_instrs', ctypes.c_uint64),
        ('num_vmem_stalls', ctypes.c_uint64),
        # FLAT instrs and stalls
        ('num_flat_instrs', ctypes.c_uint64),
        ('num_flat_stalls', ctypes.c_uint64),

        # LDS instr and stalls
        ('num_lds_instrs', ctypes.c_uint64),
        ('num_lds_stalls', ctypes.c_uint64),

        # SCA instrs stalls
        ('num_salu_instrs', ctypes.c_uint64),
        ('num_smem_instrs', ctypes.c_uint64),
        ('num_salu_stalls', ctypes.c_uint64),
        ('num_smem_stalls', ctypes.c_uint64),

        # Branch
        ('num_branch_instrs', ctypes.c_uint64),
        ('num_branch_taken_instrs', ctypes.c_uint64),
        ('num_branch_stalls', ctypes.c_uint64),

        ('timeline_string', ctypes.c_char_p),
        ('instructions_string', ctypes.c_char_p)]


class ReturnInfo(ctypes.Structure):
    _fields_ = [('num_waves', ctypes.c_uint64),
                ('wavedata', POINTER(Wave)),
                ('num_events', ctypes.c_uint64),
                ('perfevents', POINTER(PerfEvent))]

rocprofv2_att_lib = os.getenv('ROCPROFV2_ATT_LIB_PATH')
try: # For build dir
    path_to_parser = os.path.abspath(rocprofv2_att_lib)
    SO = CDLL(path_to_parser)
except: # For installed dir
    path_to_parser = os.path.abspath('/usr/lib/hsa-amd-aqlprofile/librocprofv2_att.so')
    SO = CDLL(path_to_parser)

SO.AnalyseBinary.restype = ReturnInfo
SO.AnalyseBinary.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_bool]
SO.wrapped_parse_binary.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
SO.wrapped_parse_binary.restype = ReturnAssemblyInfo

def parse_binary(filename, kernel=None):
    if kernel is None or kernel == '':
        kernel = ctypes.c_char_p(0)
        print('Parsing all kernels')
    else:
        with open(glob.glob(kernel)[0], 'r') as file:
            kernel = file.readlines()
        print('Parsing kernel:', kernel[0].split(': ')[0])
        kernel = kernel[0].split(': ')[1].split('.kd')[0]
        kernel = str(kernel).encode('utf-8')
    filename = os.path.abspath(str(filename))
    info = SO.wrapped_parse_binary(str(filename).encode('utf-8'), kernel)

    code = []
    for k in range(info.code_len):
        code_entry = info.code[k]

        # copy string memory from C++
        line = deepcopy(code_entry.line.decode("utf-8"))
        loc = deepcopy(code_entry.loc.decode("utf-8"))

        # Transform empty entries back to python's None
        to_line = int(code_entry.to_line) if (code_entry.to_line >= 0) else None
        loc = loc if len(loc) > 0 else None

        code.append((line, int(code_entry.value), to_line, loc,
                    int(code_entry.index), int(code_entry.line_num)))

    jumps = {}
    for k in range(info.jumps_len):
        jumps[info.jumps[k].key] = info.jumps[k].value

    return code, jumps


def getWaves(filename, target_cu, verbose):
    filename = os.path.abspath(str(filename))
    info = SO.AnalyseBinary(filename.encode('utf-8'), target_cu, verbose)

    waves = [info.wavedata[k] for k in range(info.num_waves)]
    events = [deepcopy(info.perfevents[k]) for k in range(info.num_events)]

    for wave in waves:
        wave.timeline = deepcopy(wave.timeline_string.decode("utf-8"))
        wave.instructions = deepcopy(wave.instructions_string.decode("utf-8"))

    return waves, events


def persist(trace_file, SIMD):
    trace = Path(trace_file).name
    simds, waves = [], []
    begin_time, end_time, timeline, instructions = [], [], [], []
    mem_ins, issued_ins, valu_ins, valu_stalls = [], [], [], []
    vmem_ins, vmem_stalls, flat_ins, flat_stalls = [], [], [], []
    lds_ins, lds_stalls, salu_ins, salu_stalls = [], [], [], []
    smem_ins, smem_stalls, br_ins, br_taken_ins, br_stalls = [], [], [], [], []

    for wave in SIMD:
        simds.append(wave.simd)
        waves.append(wave.wave_id)
        begin_time.append(wave.begin_time)
        end_time.append(wave.end_time)
        mem_ins.append(wave.num_mem_instrs)
        issued_ins.append(wave.num_issued_instrs)
        valu_ins.append(wave.num_valu_instrs)
        valu_stalls.append(wave.num_valu_stalls)
        vmem_ins.append(wave.num_vmem_instrs)
        vmem_stalls.append(wave.num_vmem_stalls)
        flat_ins.append(wave.num_flat_instrs)
        flat_stalls.append(wave.num_flat_stalls)
        lds_ins.append(wave.num_lds_instrs)
        lds_stalls.append(wave.num_lds_stalls)
        salu_ins.append(wave.num_salu_instrs)
        salu_stalls.append(wave.num_salu_stalls)
        smem_ins.append(wave.num_smem_instrs)
        smem_stalls.append(wave.num_smem_stalls)
        br_ins.append(wave.num_branch_instrs)
        br_taken_ins.append(wave.num_branch_taken_instrs)
        br_stalls.append(wave.num_branch_stalls)
        timeline.append(wave.timeline)
        instructions.append(wave.instructions)

    #df = pd.DataFrame({
    df = {
        'name': [trace for _ in range(len(begin_time))],
        'id': [i for i in range(len(begin_time))],
        'simd': simds,
        'wave_slot': waves,
        'begin_time': begin_time,
        'end_time': end_time,
        'mem_ins': mem_ins,
        'issued_ins': issued_ins,
        'valu_ins': valu_ins,
        'valu_stalls': valu_stalls,
        'vmem_ins': vmem_ins,
        'vmem_stalls': vmem_stalls,
        'flat_ins': flat_ins,
        'flat_stalls': flat_stalls,
        'lds_ins': lds_ins,
        'lds_stalls': lds_stalls,
        'salu_ins': salu_ins,
        'salu_stalls': salu_stalls,
        'smem_ins': smem_ins,
        'smem_stalls': smem_stalls,
        'br_ins': br_ins,
        'br_taken_ins': br_taken_ins,
        'br_stalls': br_stalls,
        'timeline': timeline,
        'instructions': instructions,
    }#)
    #[print(d) for c, d in df.iterrows()]; quit()
    return df


def mem_max(array):
    mem_dict = {}
    for SE in array:
        for wave in SE:
            for inst in wave:
                try:
                    mem_dict[inst[0]][0] = max(mem_dict[inst[0]][0], inst[1])
                except:
                    mem_dict[inst[0]] = inst[1:]
                assert(mem_dict[inst[0]][1] == inst[2])

    return mem_dict

def lgk(count):
    return 'lgkmcnt({0})'.format(count)
def vmc(count):
    return 'vmcnt({0})'.format(count)
def both_cnt(count):
    return lgk(count)+' '+vmc(count)

def insert_waitcnt(flight_count, assembly_code):
    flight_count = mem_max(flight_count)
    for key in sorted(flight_count):
        line_n = key
        issue_amount, waitcnt_amount, = flight_count[key]
        if 'vmcnt' in assembly_code[line_n] and 'lgkmcnt' in assembly_code[line_n]:
            counter_type = both_cnt
        elif 'vmcnt' in assembly_code[line_n]:
            counter_type = vmc
        elif 'lgkmcnt' in assembly_code[line_n]:
            counter_type = lgk
        else:
            print('Error: Line mismatch')
            exit(-1)

        for count in range(waitcnt_amount+1, issue_amount):
            print('Inserted line: '+str(line_n))
            as_index = line_n - count/(issue_amount+1)
            assembly_code[as_index] = \
                '\ts_waitcnt {0}\t\t; Timing analysis.'.format(counter_type(count))
            as_index += 0.5/(issue_amount+1)
            assembly_code[as_index] = '\ts_nop 0\t\t\t\t\t\t; Counters: '+str(issue_amount)

    return assembly_code


def get_delta_time(events):
    try:
        CUS = [[e.time for e in events if e.cu==k and e.bank==0] for k in range(16)]
        CUS = [np.asarray(c).astype(np.int64) for c in CUS if len(c) > 2]
        return np.min([np.min(abs(c[1:]-c[:-1])) for c in CUS])
    except:
        return 1

def draw_wave_metrics(selections, normalize):
    global EVENTS
    global EVENT_NAMES

    response = Readable({"counters": EVENT_NAMES})

    plt.figure(figsize=(15,3))

    delta_step = 8
    quad_delta_time = max(delta_step,int(0.5+np.min([get_delta_time(events) for events in EVENTS])))
    maxtime = np.max([np.max([e.time for e in events]) for events in EVENTS])/quad_delta_time+1

    if maxtime*delta_step >= COUNTERS_MAX_CAPTURES:
        delta_step = 1
    while maxtime >= COUNTERS_MAX_CAPTURES:
        quad_delta_time *= 2
        maxtime /= 2

    maxtime = int(min(maxtime*delta_step, COUNTERS_MAX_CAPTURES))
    event_timeline = np.zeros((16, maxtime), dtype=np.int32)
    print('Delta:', quad_delta_time)
    print('Max_cycles:', maxtime*quad_delta_time*4//delta_step)

    cycles = 4*quad_delta_time//delta_step*np.arange(maxtime)
    kernel = len(EVENTS)*quad_delta_time

    for events in EVENTS:
        for e in range(len(events)-1):
            bk = events[e].bank*4
            start = events[e].time // (quad_delta_time//delta_step)
            end = start+delta_step
            event_timeline[bk:bk+4, start:end] += np.asarray(events[e].toTuple()[1:5])[:, None]
        start = events[-1].time
        event_timeline[bk:bk+4, start:start+delta_step] += \
            np.asarray(events[-1].toTuple()[1:5])[:, None]

    event_timeline = [np.convolve(e, [kernel for k in range(3)])[1:-1] for e in event_timeline]
    #event_timeline = [e/kernel for e in event_timeline]

    if normalize:
        event_timeline = [100*e/max(e.max(), 1E-5) for e in event_timeline]

    colors = ['blue', 'green', 'gray', 'red', 'orange', 'cyan', 'black', 'darkviolet',
                'yellow', 'darkred', 'pink', 'lime', 'gold', 'tan', 'aqua', 'olive']
    [plt.plot(cycles, e, '-', label=n, color=c)
        for e, n, c, sel in zip(event_timeline, EVENT_NAMES, colors, selections) if sel]

    plt.legend()
    if normalize:
        plt.ylabel('As % of maximum')
    else:
        plt.ylabel('Value')
    plt.subplots_adjust(left=0.05, right=1, top=1, bottom=0.07)

    figure_bytes = BytesIO()
    plt.savefig(figure_bytes, dpi=150)
    return response, FileBytesIO(figure_bytes)


def draw_wave_states(selections, normalize):
    global TIMELINES
    plot_indices = [1, 2, 3, 4]
    STATES = [['Empty', 'Idle', 'Exec', 'Wait', 'Stall'][k] for k in plot_indices]
    colors = [['gray', 'orange', 'green', 'red', 'blue'][k] for k in plot_indices]

    plt.figure(figsize=(15,3))

    maxtime = max([np.max((TIMELINES[k]!=0)*np.arange(0,TIMELINES[k].size)) for k in plot_indices])
    timelines = [deepcopy(TIMELINES[k][:maxtime]) for k in plot_indices]
    timelines = [np.pad(t, [0, maxtime-t.size]) for t in timelines]

    if normalize:
        timelines = np.array(timelines) / np.maximum(np.sum(timelines,0)*1E-2,1E-7)

    kernsize = maxtime//150+1
    trim = max(maxtime//5000,1)
    cycles = np.arange(timelines[0].size)[::trim]

    kernel = np.asarray([np.exp(-abs(10*k/kernsize)) for k in range(-kernsize//2,kernsize//2+1)])
    kernel /= np.sum(kernel)

    timelines = [np.convolve(time, kernel)[kernsize//2:-kernsize//2][::trim] if len(time) > 0 else cycles*0 for time in timelines]

    [plt.plot(cycles, t, label='State '+s, linewidth=1.1, color=c)
        for t, s, c, sel in zip(timelines, STATES, colors, selections) if sel]

    plt.legend()
    if normalize:
        plt.ylabel('Waves state %')
    else:
        plt.ylabel('Waves state total')
    plt.ylim(-1)
    plt.xlim(-maxtime//200, maxtime+maxtime//200+1)
    plt.subplots_adjust(left=0.05, right=1, top=1, bottom=0.07)
    figure_bytes = BytesIO()
    plt.savefig(figure_bytes, dpi=150)
    response = Readable({"counters": STATES})
    return response, FileBytesIO(figure_bytes)


def GeneratePIC(selections=[True for k in range(16)], normalize=True, bScounter=True):
    if bScounter and len(EVENTS) > 0 and np.sum([len(e) for e in EVENTS]) > 32:
        return draw_wave_metrics(selections, normalize)
    else:
        return draw_wave_states(selections, normalize)


if __name__ == "__main__":
    pathenv = os.getenv('OUTPUT_PATH')
    if pathenv is None:
        pathenv = "."
    parser = argparse.ArgumentParser()
    parser.add_argument("assembly_code", help="Path of the assembly code")
    parser.add_argument("--trace_file", help="Filter for trace files", default=None, type=str)
    parser.add_argument("-k", "--att_kernel", help="Kernel file", type=str, default=pathenv+'/*_kernel.txt')
    parser.add_argument("-p", "--ports", help="Server and websocket ports, default: 8000,18000")
    parser.add_argument("--target_cu", help="Collected target CU id{0-15}", type=int, default=None)
    parser.add_argument("-g", "--genasm",
                        help="Generate post-processed asm file at this path", type=str, default="")
    args = parser.parse_args()

    global EVENT_NAMES
    with open(os.getenv("COUNTERS_PATH"), 'r') as f:
        lines = [l.split('//')[0] for l in f.readlines()]

        EVENT_NAMES = []
        clean = lambda x: x.split('=')[1].split(' ')[0].split('\n')[0]
        for line in lines:
            if 'PERFCOUNTER_ID=' in line:
                EVENT_NAMES += ['id: '+clean(line)]
            elif args.target_cu is None and 'att: TARGET_CU' in line:
                args.target_cu = int(clean(line))
                print('Target CU set to:', args.target_cu)
        for line in lines:
            if 'PERFCOUNTER=' in line:
                EVENT_NAMES += [clean(line).split('SQ_')[1].lower()]

    if args.target_cu is None:
        args.target_cu = 1

    # Assembly parsing
    path = Path(args.assembly_code)
    if not path.is_file():
        print("Invalid assembly_code('{0}')!".format(args.assembly_code))
        sys.exit(1)

    att_kernel = glob.glob(args.att_kernel)

    if len(att_kernel) == 0:
        print('Could not find att output kernel:', args.att_kernel)
        exit(1)
    elif len(att_kernel) > 1:
        print('Found multiple kernel matching given filters:')
        for n, k in enumerate(att_kernel):
            print('\t', n, '->', k)

        bValid = False
        while bValid == False:
            try:
                args.att_kernel = att_kernel[int(input("Please select number: "))]
                bValid = True
            except KeyboardInterrupt:
                exit(0)
            except:
                print('Invalid option.')
    else:
        args.att_kernel = att_kernel[0]

    print('Att kernel:', args.att_kernel)
    code, jumps = parse_binary(args.assembly_code, args.att_kernel)

    # Trace Parsing
    if args.trace_file is None:
        filenames = glob.glob(args.att_kernel.split('_kernel.txt')[0]+'*.att')
        assert(len(filenames) > 0)
    else:
        filenames = glob.glob(args.trace_file)

    print('Trace filenames:', filenames)

    DBFILES = []
    global TIMELINES
    global EVENTS
    TIMELINES = [np.zeros(int(1E4),dtype=np.int32) for k in range(5)]
    EVENTS = []

    analysed_filenames = []
    for name in filenames:
        SIMD, perfevents = getWaves(name, args.target_cu, False)
        if len(SIMD) == 0:
            print("Error parsing ", name)
            continue
        analysed_filenames.append(name)
        EVENTS.append(perfevents)
        DBFILES.append( persist(name, SIMD) )
        for wave in SIMD:
            time_acc = 0
            tuples1 = wave.timeline.split('(')
            tuples2 = [t.split(')')[0].split(',') for t in tuples1 if t != '']
            tuples3 = [(int(t[0]),int(t[1])) for t in tuples2]

            for state in tuples3:
                if state[1] > 1E7:
                    print('Warning: Time limit reached for ',state[0], state[1])
                    break

                if time_acc+state[1] > TIMELINES[state[0]].size:
                    TIMELINES[state[0]] = np.hstack([
                        TIMELINES[state[0]],
                        np.zeros_like(TIMELINES[state[0]])
                    ])
                TIMELINES[state[0]][time_acc:time_acc+state[1]] += 1
                time_acc += state[1]

    if args.genasm and len(args.genasm) > 0:
        flight_count = view_trace(args, 0, code, jumps, DBFILES, analysed_filenames, True, None)

        with open(args.assembly_code, 'r') as file:
            lines = file.readlines()
        assembly_code = {l+1.0: lines[l][:-1] for l in range(len(lines))}
        assembly_code = insert_waitcnt(flight_count, assembly_code)

        with open(args.genasm, 'w') as file:
            keys = sorted(assembly_code.keys())
            for k in keys:
                file.write(assembly_code[k]+'\n')
    else:
        view_trace(args, 0, code, jumps, DBFILES, analysed_filenames, False, GeneratePIC)
