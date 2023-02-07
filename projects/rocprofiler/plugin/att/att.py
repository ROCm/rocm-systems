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
from trace_view import view_trace
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
import json

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

try: # For build dir
    path_to_parser = os.path.abspath('/usr/lib/hsa-amd-aqlprofile/librocprofv2_att.so')
    SO = CDLL(path_to_parser)
except: # For installed dir
    path_to_parser = os.path.abspath('/usr/local/lib/hsa-amd-aqlprofile/librocprofv2_att.so')
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
    info = SO.AnalyseBinary(filename.encode('utf-8'), target_cu, verbose)

    waves = [info.wavedata[k] for k in range(info.num_waves)]
    events = [deepcopy(info.perfevents[k]) for k in range(info.num_events)]

    for wave in waves:
        wave.timeline = deepcopy(wave.timeline_string.decode("utf-8"))
        wave.instructions = deepcopy(wave.instructions_string.decode("utf-8"))

    return waves, events


def persist(output_ui, trace_file, SIMD):
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


def Copy_Files(output_ui):
    curpath = os.path.dirname(os.path.abspath(__file__))
    outpath = output_ui+'/ui/'

    os.makedirs(outpath, exist_ok=True)
    os.system('cp '+curpath+'/ui/* '+outpath)


def get_delta_time(events):
    for begin in range(len(events)):
        tg_cu = events[begin].cu
        for e in range(begin+1,len(events)):
            if events[e].cu == tg_cu:
                return events[e].time-events[begin].time
    return 1


def num_cus(EVENTS):
    cus = 0
    for events in EVENTS:
        for e in events:
            cus = max(cus, e.cu)
    return cus+1


def draw_wave_metrics(selections, normalize):
    global PIC_SAVE_FOLDER
    global EVENTS
    global EVENT_NAMES

    #event_names = ['Busy CUs', 'Occupancy', 'Eligible waves', 'Waves waiting']
    with open(PIC_SAVE_FOLDER+'counters.json', 'w') as f:
        f.write(json.dumps({"counters": EVENT_NAMES}))

    plt.figure(figsize=(15,3))

    delta_time = int(0.5+np.mean([get_delta_time(events) for events in EVENTS]))
    maxtime = np.max([np.max([e.time for e in events]) for events in EVENTS])+1
    event_timeline = np.zeros((16, maxtime), dtype=np.int32)
    print('Delta:', delta_time)
    print('Max_cycles:', maxtime)

    kernsize = 2*(delta_time//8)+1
    trim = max(maxtime//5000,1)
    cycles = 4*np.arange(maxtime)[::trim]

    kernel = np.asarray([np.exp(-abs(k/kernsize)**2) for k in range(-kernsize*3,kernsize*3+1)])
    kernel /= np.sum(kernel)*len(EVENTS)*delta_time#*5.12 # SEslots/100%

    for events in EVENTS:
        for e in range(len(events)-1):
            bk = events[e].bank*4
            start = events[e].time
            end = start+delta_time
            event_timeline[bk:bk+4, start:end] += np.asarray(events[e].toTuple()[1:5])[:, None]
        start = events[-1].time
        event_timeline[bk:bk+4, start:start+delta_time] += \
            np.asarray(events[-1].toTuple()[1:5])[:, None]

    event_timeline = [np.convolve(e, kernel)[3*kernsize:-3*kernsize] for e in event_timeline]

    if normalize:
        event_timeline = [100*e/max(e.max(), 1E-5) for e in event_timeline]
    #event_timeline[0] = np.clip(event_timeline[0]*8.5, 0, 100) #CC rate
    
    colors = ['blue', 'green', 'gray', 'red', 'orange', 'cyan', 'black', 'darkviolet',
                'yellow', 'darkred', 'pink', 'lime', 'gold', 'tan', 'aqua', 'olive']
    [plt.plot(cycles, e[::trim], '-', label=n, color=c)
        for e, n, c, sel in zip(event_timeline, EVENT_NAMES, colors, selections) if sel]

    plt.legend()
    if normalize:
        plt.ylabel('As % of maximum')
    else:
        plt.ylabel('Value')
    plt.subplots_adjust(left=0.05, right=1, top=1, bottom=0.07)
    plt.savefig(PIC_SAVE_FOLDER+'timeline.png', dpi=150)
    #plt.show()


def draw_wave_states(selections, normalize):
    global TIMELINES
    global PIC_SAVE_FOLDER
    plot_indices = [1, 2, 3, 4]
    STATES = [['Empty', 'Idle', 'Exec', 'Wait', 'Stall'][k] for k in plot_indices]
    colors = [['gray', 'orange', 'green', 'red', 'blue'][k] for k in plot_indices]

    plt.figure(figsize=(15,3))

    maxtime = max([np.max((TIMELINES[k]!=0)*np.arange(0,TIMELINES[k].size)) for k in plot_indices])
    timelines = [deepcopy(TIMELINES[k][:maxtime]) for k in plot_indices]
    timelines = [np.pad(t, [0, maxtime-t.size]) for t in timelines]

    if normalize:
        timelines = np.array(timelines) / np.maximum(np.sum(timelines,0)*1E-2,1E-7)

    kernsize = maxtime//120+3
    trim = max(maxtime//5000,1)
    cycles = np.arange(timelines[0].size)[::trim]

    kernel = np.asarray([np.exp(-abs(10*k/kernsize)) for k in range(-kernsize//2,kernsize//2+1)])
    kernel /= np.sum(kernel)

    timelines = [np.convolve(time, kernel)[kernsize//2:-kernsize//2][::trim] for time in timelines]

    with open(PIC_SAVE_FOLDER+'counters.json', 'w') as f:
        f.write(json.dumps({"counters": STATES}))

    [plt.plot(cycles, t, label='State '+s, linewidth=1.1, color=c)
        for t, s, c, sel in zip(timelines, STATES, colors, selections) if sel]

    plt.legend()
    if normalize:
        plt.ylabel('Waves state %')
    else:
        plt.ylabel('Waves state total')
    plt.ylim(-1)
    plt.xlim(-maxtime//200, maxtime+maxtime//200)
    plt.subplots_adjust(left=0.05, right=1, top=1, bottom=0.07)
    plt.savefig(PIC_SAVE_FOLDER+'timeline.png', dpi=150)


def GeneratePIC(selections=[True for k in range(4)], normalize=True):
    if len(EVENTS) > 0 and np.sum([len(e) for e in EVENTS]) > 32:
        draw_wave_metrics(selections, normalize)
    else:
        draw_wave_states(selections, normalize)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("assembly_code", help="Path of the assembly code")
    parser.add_argument("--trace_file", help="Filter for trace files", default=None, type=str)
    parser.add_argument("-o", "--output_ui", help="Output Folder", default='/dev/shm/attplugin/')
    parser.add_argument("-k", "--att_kernel", help="Kernel file", type=str, default='*_kernel.txt')
    parser.add_argument("-w", "--wave_id", help="wave id")
    parser.add_argument("-p", "--ports", help="Server and websocket ports, default: 8000,18000")
    parser.add_argument("--target_cu", help="Collected target CU id{0-15}", type=int, default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
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

    Copy_Files(args.output_ui)
    DBFILES = []
    global TIMELINES
    global EVENTS
    TIMELINES = [np.zeros(int(1E4),dtype=np.int32) for k in range(5)]
    EVENTS = []
    for name in filenames:
        SIMD, perfevents = getWaves(name, args.target_cu, args.verbose)
        EVENTS.append(perfevents)
        DBFILES.append( persist(args.output_ui, name, SIMD) )
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
        flight_count = view_trace(args, 0, code, jumps, DBFILES, filenames, True, None)

        with open(args.assembly_code, 'r') as file:
            lines = file.readlines()
        assembly_code = {l+1.0: lines[l][:-1] for l in range(len(lines))}
        assembly_code = insert_waitcnt(flight_count, assembly_code)

        with open(args.genasm, 'w') as file:
            keys = sorted(assembly_code.keys())
            for k in keys:
                file.write(assembly_code[k]+'\n')
    else:
        global PIC_SAVE_FOLDER
        PIC_SAVE_FOLDER = args.output_ui+"/ui/"
        view_trace(args, 0, code, jumps, DBFILES, filenames, False, GeneratePIC)
