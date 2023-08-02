#!/usr/bin/env python3
import sys
if sys.version_info[0] < 3:
    raise Exception("Must be using Python 3")

import os
import argparse
from pathlib import Path
from ctypes import *
import ctypes
from copy import deepcopy
from trace_view import view_trace
import sys
import glob
import numpy as np
from stitch import stitch
import gc

try:
    from mpi4py import MPI
    MPI_IMPORTED = True
except:
    MPI_IMPORTED = False

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

        ('timeline_array', POINTER(ctypes.c_int64)),
        ('instructions_array', POINTER(ctypes.c_int64)),
        ('timeline_size', ctypes.c_uint64),
        ('instructions_size', ctypes.c_uint64)]

class PythonWave:
    def __init__(self, source_wave):
        for property, value in Wave._fields_:
            setattr(self, property, getattr(source_wave, property))
        self.timeline_array = None
        self.instructions_array = None

# Flags :
#   IS_NAVI = 0x1
class ReturnInfo(ctypes.Structure):
    _fields_ = [('num_waves', ctypes.c_uint64),
                ('wavedata', POINTER(Wave)),
                ('num_events', ctypes.c_uint64),
                ('perfevents', POINTER(PerfEvent)),
                ('occupancy', POINTER(ctypes.c_uint64)),
                ('num_occupancy', ctypes.c_uint64),
                ('flags', ctypes.c_uint64)]

rocprofv2_att_lib = os.getenv('ROCPROFV2_ATT_LIB_PATH')
if rocprofv2_att_lib is None:
    print("ATT Lib path not set. Use export ROCPROFV2_ATT_LIB_PATH=/path/to/librocprofv2_att.so")
    quit()
path_to_parser = os.path.abspath(rocprofv2_att_lib)
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

        line = deepcopy(code_entry.line.decode("utf-8"))
        loc = deepcopy(code_entry.loc.decode("utf-8"))

        to_line = int(code_entry.to_line) if (code_entry.to_line >= 0) else None
        loc = loc if len(loc) > 0 else None

        code.append([line, int(code_entry.value), to_line, loc,
                    int(code_entry.index), int(code_entry.line_num), 0, 0]) # hitcount + cycles

    jumps = {}
    for k in range(info.jumps_len):
        jumps[info.jumps[k].key] = info.jumps[k].value

    return code, jumps


def getWaves_binary(name, shader_engine_data_dict, target_cu, depth):
    filename = os.path.abspath(str(name))
    info = SO.AnalyseBinary(filename.encode('utf-8'), target_cu, False)

    waves = [info.wavedata[k] for k in range(info.num_waves)]
    events = [deepcopy(info.perfevents[k]) for k in range(info.num_events)]
    occupancy = [int(info.occupancy[k]) for k in range(int(info.num_occupancy))]
    flags = 'navi' if (info.flags & 0x1) else 'vega'

    wave_slot_count = [[0 for k in range(20)] for j in range(4)]
    waves_python = []
    for wave in waves:
        if wave_slot_count[wave.simd][wave.wave_id] >= depth or wave.instructions_size == 0:
            continue
        wave_slot_count[wave.simd][wave.wave_id] += 1
        pwave = PythonWave(wave)
        pwave.timeline = [(wave.timeline_array[2*k], wave.timeline_array[2*k+1]) for k in range(wave.timeline_size)]
        pwave.instructions = [tuple([wave.instructions_array[4*k+m] for m in range(4)]) for k in range(wave.instructions_size)]
        waves_python.append( pwave )
    shader_engine_data_dict[name] = (waves_python, events, occupancy, flags)


def getWaves_stitch(SIMD, code, jumps, flags, latency_map, hitcount_map):
    for pwave in SIMD:
        pwave.instructions = stitch(pwave.instructions, code, jumps, flags)

        for inst in pwave.instructions[0]:
            hitcount_map[inst[-1]] += 1
            latency_map[inst[-1]] += inst[3]


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
    }
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


def apply_min_event(min_event_time, OCCUPANCY, EVENTS, DBFILES, TIMELINES):
    for n, occ in enumerate(OCCUPANCY):
        OCCUPANCY[n] = [max(min(int((u>>16)-min_event_time)<<16,2**42),0) | (u&0xFFFFF) for u in occ]
    for perf in EVENTS:
        for p in perf:
            p.time -= min_event_time

    for df in DBFILES:
        for T in range(len(df['timeline'])):
            timeline = df['timeline'][T]
            time_acc = 0
            tuples3 = [(0,df['begin_time'][T]-min_event_time)]+[(int(t[0]),int(t[1])) for t in timeline]

            for state in tuples3:
                if state[1] > 1E8:
                    print('Warning: Time limit reached for ',state[0], state[1])
                    break
                if time_acc+state[1] > TIMELINES[state[0]].size:
                    TIMELINES[state[0]] = np.hstack([
                        TIMELINES[state[0]],
                        np.zeros_like(TIMELINES[state[0]])
                    ])
                TIMELINES[state[0]][time_acc:time_acc+state[1]] += 1
                time_acc += state[1]

if __name__ == "__main__":
    comm = None
    mpi_root = True
    if MPI_IMPORTED:
        try:
            comm = MPI.COMM_WORLD
            if comm.Get_size() < 2:
                comm = None
            else:
                mpi_root = comm.Get_rank() == 0
        except:
            print('Could not load MPI')
            comm = None

    pathenv = os.getenv('OUTPUT_PATH')
    if pathenv is None:
        pathenv = "."
    parser = argparse.ArgumentParser()
    parser.add_argument("assembly_code", help="Path to the assembly code. Must be the first parameter.")
    parser.add_argument("--depth", help="Maximum number of parsed waves per slot", default=100, type=int)
    parser.add_argument("--trace_file", help="Filter for trace files", default=None, type=str)
    parser.add_argument("--att_kernel", help="Kernel file",
                        type=str, default=pathenv+'/*_kernel.txt')
    parser.add_argument("--ports", help="Server and websocket ports, default: 8000,18000")
    parser.add_argument("--genasm",
                        help="Generate post-processed asm file at this path", type=str, default="")
    parser.add_argument("--mode", help='''ATT analysis modes:\n
                        off: Only run ATT collection, disable analysis.\n
                        file: dump json files to disk.\n
                        network: Open att server over the network.''', type=str, default="off")
    args = parser.parse_args()

    if args.mode.lower() == 'file':
        args.dumpfiles = True
    elif args.mode.lower() == 'network':
        args.dumpfiles = False
    else:
        print('Skipping analysis.')
        quit()

    with open(os.getenv("COUNTERS_PATH"), 'r') as f:
        lines = [l.split('//')[0] for l in f.readlines()]

        EVENT_NAMES = []
        clean = lambda x: x.split('=')[1].split(' ')[0].split('\n')[0]
        for line in lines:
            if 'PERFCOUNTER_ID=' in line:
                EVENT_NAMES += ['id: '+clean(line)]
            elif 'att: TARGET_CU' in line:
                args.target_cu = int(clean(line))
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
        if mpi_root:
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
        if comm is not None:
            args.att_kernel = comm.bcast(args.att_kernel, root=0)
    else:
        args.att_kernel = att_kernel[0]

    # Trace Parsing
    if args.trace_file is None:
        filenames = glob.glob(args.att_kernel.split('_kernel.txt')[0]+'_*.att')
    else:
        filenames = glob.glob(args.trace_file)
    assert(len(filenames) > 0)

    if comm is not None:
        filenames = filenames[comm.Get_rank()::comm.Get_size()]

    code = jumps = None
    if mpi_root:
        print('Att kernel:', args.att_kernel)
        code, jumps = parse_binary(args.assembly_code, args.att_kernel)

    DBFILES = []
    TIMELINES = [np.zeros(int(1E4),dtype=np.int16) for k in range(5)]
    EVENTS = []
    OCCUPANCY = []
    GFXV = []
    analysed_filenames = []

    shader_engine_data_dict = {}
    for name in filenames:
        getWaves_binary(name, shader_engine_data_dict, args.target_cu, args.depth)

    if comm is not None:
        code = comm.bcast(code, root=0)
        jumps = comm.bcast(jumps, root=0)

    gc.collect()
    latency_map = np.zeros((len(code)), dtype=np.int64)
    hitcount_map = np.zeros((len(code)), dtype=np.int32)
    for name in filenames:
        SIMD, perfevents, occupancy, gfxv = shader_engine_data_dict[name]
        getWaves_stitch(SIMD, code, jumps, gfxv, latency_map, hitcount_map)
        if len(SIMD) == 0:
            print("Error parsing ", name)
            continue
        analysed_filenames.append(name)
        EVENTS.append(perfevents)
        DBFILES.append( persist(name, SIMD) )
        OCCUPANCY.append( occupancy )
        GFXV.append(gfxv)

    gc.collect()
    min_event_time = 2**62
    for df in DBFILES:
        if len(df['begin_time']) > 0:
            min_event_time = min(min_event_time, np.min(df['begin_time']))
    for perf in EVENTS:
        for p in perf:
            min_event_time = min(min_event_time, p.time)
    for occ in OCCUPANCY:
        min_event_time = min(min_event_time, np.min(np.array(occ)>>16))

    gc.collect()
    min_event_time = max(0, min_event_time-32)
    if comm is not None:
        min_event_time = comm.reduce(min_event_time, op=MPI.MIN)
        min_event_time = comm.bcast(min_event_time, root=0)

        apply_min_event(min_event_time, OCCUPANCY, EVENTS, DBFILES, TIMELINES)

        GFXV = comm.gather(GFXV, root=0)
        EVENTS = comm.gather(EVENTS, root=0)
        OCCUPANCY = comm.gather(OCCUPANCY, root=0)
        TIMELINES = comm.gather(TIMELINES, root=0)
        gather_latency_map = comm.gather(latency_map, root=0)
        gather_hitcount_map = comm.gather(hitcount_map, root=0)
        gathered_filenames = comm.gather(analysed_filenames, root=0)

        if mpi_root:
            latency_map *= 0
            hitcount_map *= 0
            for hit, lat in zip(gather_hitcount_map, gather_latency_map):
                hitcount_map += hit
                latency_map += lat
            EVENTS = [e for elem in EVENTS for e in elem]
            OCCUPANCY = [e for elem in OCCUPANCY for e in elem]
            gathered_filenames = [e for elem in gathered_filenames for e in elem]
            gfxv = [e for elem in GFXV for e in elem][0]
    
            TIMELINES_GATHER = TIMELINES
            TIMELINES = [np.zeros((np.max([len(tm[k]) for tm in TIMELINES])), np.int16) for k in range(5)]
            for gather in TIMELINES_GATHER:
                for t, m in zip(TIMELINES, gather):
                    t[:len(m)] += m
            del(TIMELINES_GATHER)
        else: # free up memory
            TIMELINES = []
            OCCUPANCY = []
            EVENTS = []
    else:
        apply_min_event(min_event_time, OCCUPANCY, EVENTS, DBFILES, TIMELINES)
        gathered_filenames = analysed_filenames

    if mpi_root:
        for k in range(len(code)):
            code[k][-2] = int(hitcount_map[k])
            code[k][-1] = int(latency_map[k])

    gc.collect()
    print("Min time:", min_event_time)

    drawinfo = {'TIMELINES':TIMELINES, 'EVENTS':EVENTS, 'EVENT_NAMES':EVENT_NAMES, 'OCCUPANCY': OCCUPANCY, 'ShaderNames': gathered_filenames}
    if args.genasm and len(args.genasm) > 0:
        flight_count = view_trace(args, code, DBFILES, analysed_filenames, True, OCCUPANCY, args.dumpfiles, min_event_time, gfxv, drawinfo, comm, mpi_root)
        with open(args.assembly_code, 'r') as file:
            lines = file.readlines()
        assembly_code = {l+1.0: lines[l][:-1] for l in range(len(lines))}
        assembly_code = insert_waitcnt(flight_count, assembly_code)

        with open(args.genasm, 'w') as file:
            keys = sorted(assembly_code.keys())
            for k in keys:
                file.write(assembly_code[k]+'\n')
    else:
        view_trace(args, code, DBFILES, analysed_filenames, False, OCCUPANCY, args.dumpfiles, min_event_time, gfxv, drawinfo, comm, mpi_root)
