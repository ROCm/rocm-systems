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
import stitch
import gc
from collections import defaultdict
import service

ATT_VERSION = 5

class pcInfo(ctypes.Structure):
    _fields_ = [
        ("addr", c_uint64),
        ("marker_id", c_uint64)
    ]
    def to_v2_pc(self):
        if self.marker_id == 0:
            return self.addr
        return self.addr | (self.marker_id<<service.ID_OFFSET) | (1<<service.HEADER_OFFSET)

class TraceData(ctypes.Structure):
    _fields_ = [
        ("type", c_uint64, 8),
        ("hitcount", c_uint64, 56),
        ("latency", c_uint64),
        ("pc", pcInfo)
    ]

class TraceDataTranslated:
    def __init__(self, inst):
        self.type = inst.type
        self.num_waves = inst.hitcount
        self.cycles = inst.latency
        if self.type == stitch.PCINFO:
            self.cycles = inst.pc.to_v2_pc()

class Trace:
    def __init__(self, traceid, tracesize, instructions_array):
        self.instructions = [TraceDataTranslated(instructions_array[k]) for k in range(tracesize)]
        self.traceid = traceid

class PerfEvent(ctypes.Structure):
    _fields_ = [
        ("time", c_uint64),
        ("event0", c_uint16),
        ("event1", c_uint16),
        ("event2", c_uint16),
        ("event3", c_uint16),
        ("cu", c_uint8),
        ("bank", c_uint8),
    ]

    def toTuple(self):
        return (
            int(self.time),
            int(self.event0),
            int(self.event1),
            int(self.event2),
            int(self.event3),
            int(self.cu),
            int(self.bank),
        )


class CodeWrapped(ctypes.Structure):
    """ Matches CodeWrapped on the python side """
    _fields_ = [('line', ctypes.c_char_p),
                ('loc', ctypes.c_char_p),
                ('to_line', ctypes.c_int),
                ('value', ctypes.c_int),
                ('index', ctypes.c_int),
                ('line_num', ctypes.c_int),
                ('addr', ctypes.c_int64)]


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


class WaveState(ctypes.Structure):
    _fields_ = [
        ("type", c_int32),
        ("duration", c_int32)
    ]

class WaveInstruction(ctypes.Structure):
    _fields_ = [
        ("time", c_int64),
        ("duration", c_int64)
    ]

class Wave(ctypes.Structure):
    _fields_ = [
        ("simd", ctypes.c_uint8),
        ("wave_id", ctypes.c_uint8),
        ("trap_status", ctypes.c_uint8),
        ("reserved", ctypes.c_uint8),

        # total VMEM/FLAT/LDS/SMEM instructions issued
        # VMEM Pipeline: instrs and stalls
        ("num_vmem_instrs", ctypes.c_int),
        ("num_vmem_stalls", ctypes.c_int),
        # FLAT instrs and stalls
        ("num_flat_instrs", ctypes.c_int),
        ("num_flat_stalls", ctypes.c_int),
        # LDS instr and stalls
        ("num_lds_instrs", ctypes.c_int),
        ("num_lds_stalls", ctypes.c_int),
        # SCA instrs stalls
        ("num_salu_instrs", ctypes.c_int),
        ("num_smem_instrs", ctypes.c_int),
        ("num_salu_stalls", ctypes.c_int),
        ("num_smem_stalls", ctypes.c_int),
        # Branch
        ("num_branch_instrs", ctypes.c_int),
        ("num_branch_taken_instrs", ctypes.c_int),
        ("num_branch_stalls", ctypes.c_int),

        # total issued memory instructions
        ("num_mem_instrs", ctypes.c_int),
        # total valus insts and stalls
        ("num_valu_stalls", ctypes.c_int),
        ("num_valu_instrs", ctypes.c_size_t),
        # total issued instructions (compute + memory)
        ("num_issued_instrs", ctypes.c_size_t),
        # Begin and end cycle
        ("begin_time", ctypes.c_int64),
        ("end_time", ctypes.c_int64),
        ("traceid", ctypes.c_int64),

        ("timeline_size", ctypes.c_size_t),
        ("instructions_size", ctypes.c_size_t),
        ("timeline_array", POINTER(WaveState)),
        ("instructions_array", POINTER(WaveInstruction)),
    ]


class PythonWave:
    def __init__(self, sourcew):
        for property, value in Wave._fields_:
            try:
                setattr(self, deepcopy(property), deepcopy(getattr(sourcew, property)))
            except:
                pass

        self.timeline = [
            (int(sourcew.timeline_array[k].type), int(sourcew.timeline_array[k].duration))
            for k in range(self.timeline_size)
        ]
        self.timeline_array = None

        self.instructions = [
            (int(sourcew.instructions_array[k].time), int(sourcew.instructions_array[k].duration))
            for k in range(self.instructions_size)
        ]
        self.instructions_array = None


class ReturnInfo(ctypes.Structure):
    _fields_ = [
        ("flags", ctypes.c_uint64),
        ("binaryID", ctypes.c_uint64),
        ("num_traces", ctypes.c_uint64),
        ("tracesizes", POINTER(ctypes.c_uint64)),
        ("traceIDs", POINTER(ctypes.c_int64)),
        ("tracedata", POINTER(POINTER(TraceData))),

        ("num_events", ctypes.c_uint64),
        ("perfevents", POINTER(PerfEvent)),
        ("occupancy", POINTER(ctypes.c_uint64)),
        ("num_occupancy", ctypes.c_uint64),
        ("kernel_id_addr", POINTER(pcInfo)),
        ("num_kernel_ids", ctypes.c_uint64),

        ("wavedata", POINTER(Wave)),
        ("num_waves", ctypes.c_uint64),
    ]


rocprofv2_att_lib = os.getenv("ROCPROFV2_ATT_LIB_PATH")
if rocprofv2_att_lib is None:
    print(
        "ATT Lib path not set. Use export ROCPROFV2_ATT_LIB_PATH=/path/to/librocprofv2_att.so"
    )
    quit()
path_to_parser = os.path.abspath(rocprofv2_att_lib)
SO = CDLL(path_to_parser)

SO.AnalyseBinary.restype = ReturnInfo
SO.AnalyseBinary.argtypes = [ctypes.c_char_p]
SO.wrapped_parse_binary.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
SO.wrapped_parse_binary.restype = ReturnAssemblyInfo
SO.FreeBinary.argtypes = [ctypes.c_uint64]

def parse_binary(filename, kernel=None):
    if kernel is None or kernel == "":
        kernel = ctypes.c_char_p(0)
    else:
        with open(glob.glob(kernel)[0], "r") as file:
            kernel = file.readlines()
        kernel = kernel[0].split(": ")[1].split(".kd")[0]
        kernel = str(kernel).encode("utf-8")
    filename = os.path.abspath(str(filename))
    info = SO.wrapped_parse_binary(str(filename).encode("utf-8"), kernel)

    code = []
    for k in range(info.code_len):
        code_entry = info.code[k]

        line = deepcopy(code_entry.line.decode("utf-8"))
        loc = deepcopy(code_entry.loc.decode("utf-8"))

        to_line = int(code_entry.to_line) if (code_entry.to_line >= 0) else None
        loc = loc if len(loc) > 0 else None

        # asm, inst_type, addr, loc, index, line_num, hitcount, cycles
        code.append([line, int(code_entry.value), to_line, loc, int(code_entry.index),
                    int(code_entry.line_num), int(code_entry.addr), 0, 0])

    jumps = {}
    for k in range(info.jumps_len):
        jumps[info.jumps[k].key] = info.jumps[k].value

    return code, jumps


def getWaves_binary(name):
    filename = os.path.abspath(str(name))
    info = SO.AnalyseBinary(filename.encode("utf-8"))

    isValid = info.flags & 0x1
    if isValid == 0:
        print('Invalid trace ', name)
        return ([], [], [], [], None, [])
    flags = "navi" if (info.flags & 0x2) else "vega"
    kernel_addr = [info.kernel_id_addr[k].to_v2_pc() for k in range(info.num_kernel_ids)]
    events = [deepcopy(info.perfevents[k]) for k in range(info.num_events)]
    occupancy = [int(info.occupancy[k]) for k in range(int(info.num_occupancy))]

    assert(((info.flags >> 3) & 0x1FFF == ATT_VERSION)) # Check ATT parser version

    traces_python = {}
    for T in range(info.num_traces):
        if info.tracesizes[T] > 2:
            id = info.traceIDs[T]
            traces_python[id] = Trace(id, int(info.tracesizes[T]), info.tracedata[T])

    waves_python = []
    for k in range(info.num_waves):
        if info.wavedata[k].instructions_size > 2:
            waves_python.append(PythonWave(info.wavedata[k]))

    SO.FreeBinary(info.binaryID)

    return (traces_python, waves_python, events, occupancy, flags, kernel_addr, info.flags & 0x4)


def getWaves_stitch(traces, code, jumps, flags, latency_map, hitcount_map, bIsAuto, codeobjservice):
    for id in traces.keys():
        traces[id].instructions = stitch.stitch(traces[id].instructions, code, jumps, flags, bIsAuto, codeobjservice)
        if len(code) > hitcount_map.size:
            hitcount_map = np.pad(hitcount_map, [0,len(code)-hitcount_map.size])
            latency_map = np.pad(latency_map, [0,len(code)-latency_map.size])
        if traces[id].instructions is not None:
            for inst in traces[id].instructions[0]:
                hitcount_map[inst.asmline] += inst.num_waves
                latency_map[inst.asmline] += inst.cycles
    return hitcount_map, latency_map


def persist(trace_file, SIMD, traces):
    trace = Path(trace_file).name
    simds, waves = [], []
    begin_time, end_time, timeline, instructions, trace_ids = [], [], [], [], []
    mem_ins, issued_ins, valu_ins, valu_stalls = [], [], [], []
    vmem_ins, vmem_stalls, flat_ins, flat_stalls = [], [], [], []
    lds_ins, lds_stalls, salu_ins, salu_stalls = [], [], [], []
    smem_ins, smem_stalls, br_ins, br_taken_ins, br_stalls = [], [], [], [], []

    for wave in SIMD:
        try:
            if wave.instructions is None or traces[wave.traceid].instructions is None:
                continue
        except:
            continue
        simds.append(wave.simd)
        waves.append(wave.wave_id)
        begin_time.append(wave.begin_time)
        end_time.append(wave.end_time)
        trace_ids.append(wave.traceid)
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

        cc = 0
        insts = []
        skips = traces[wave.traceid].instructions[-1]
        try:
            for v in traces[wave.traceid].instructions[0]:
                while cc in skips:
                    cc += 1
                t = wave.instructions[cc]
                insts.append((t[0], v.type, 0, t[1], v.asmline))
                cc += 1
        except:
            pass # Incomplete waves
        instructions.append((insts,) + traces[wave.traceid].instructions[1:-1])

    df = {
        "name": [trace for _ in range(len(begin_time))],
        "id": [i for i in range(len(begin_time))],
        "simd": simds,
        "wave_slot": waves,
        "begin_time": begin_time,
        "end_time": end_time,
        "mem_ins": mem_ins,
        "issued_ins": issued_ins,
        "valu_ins": valu_ins,
        "valu_stalls": valu_stalls,
        "vmem_ins": vmem_ins,
        "vmem_stalls": vmem_stalls,
        "flat_ins": flat_ins,
        "flat_stalls": flat_stalls,
        "lds_ins": lds_ins,
        "lds_stalls": lds_stalls,
        "salu_ins": salu_ins,
        "salu_stalls": salu_stalls,
        "smem_ins": smem_ins,
        "smem_stalls": smem_stalls,
        "br_ins": br_ins,
        "br_taken_ins": br_taken_ins,
        "br_stalls": br_stalls,
        "timeline": timeline,
        "instructions": instructions,
        "traceids": trace_ids,
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
                assert mem_dict[inst[0]][1] == inst[2]

    return mem_dict


def lgk(count):
    return "lgkmcnt({0})".format(count)


def vmc(count):
    return "vmcnt({0})".format(count)


def both_cnt(count):
    return lgk(count) + " " + vmc(count)


def insert_waitcnt(flight_count, assembly_code):
    flight_count = mem_max(flight_count)
    for key in sorted(flight_count):
        line_n = key
        (
            issue_amount,
            waitcnt_amount,
        ) = flight_count[key]
        if "vmcnt" in assembly_code[line_n] and "lgkmcnt" in assembly_code[line_n]:
            counter_type = both_cnt
        elif "vmcnt" in assembly_code[line_n]:
            counter_type = vmc
        elif "lgkmcnt" in assembly_code[line_n]:
            counter_type = lgk
        else:
            print("Error: Line mismatch")
            exit(-1)

        for count in range(waitcnt_amount + 1, issue_amount):
            print("Inserted line: " + str(line_n))
            as_index = line_n - count / (issue_amount + 1)
            assembly_code[as_index] = "\ts_waitcnt {0}\t\t; Timing analysis.".format(
                counter_type(count)
            )
            as_index += 0.5 / (issue_amount + 1)
            assembly_code[as_index] = "\ts_nop 0\t\t\t\t\t\t; Counters: " + str(
                issue_amount
            )

    return assembly_code


def gen_timelines(DBFILES):
    TIMELINES = [np.zeros(int(1E6), dtype=np.float32) for k in range(5)]
    TIME_RESOLUTION = 16
    for df in DBFILES:
        for T in range(len(df["timeline"])):
            timeline = df["timeline"][T]
            time_acc = 0
            tuples3 = [(0, df["begin_time"][T])] + [(int(t[0]), int(t[1])) for t in timeline]

            for state in tuples3:
                t_end = (time_acc + state[1])//TIME_RESOLUTION
                if t_end > 1E8:
                    print("Warning: Time limit reached for ", state[0], state[1])
                    break
                elif t_end > TIMELINES[state[0]].size:
                    TIMELINES[state[0]] = np.hstack(
                        [TIMELINES[state[0]], np.zeros_like(TIMELINES[state[0]])]
                    )
                TIMELINES[state[0]][time_acc//TIME_RESOLUTION : t_end] += 1
                time_acc += state[1]
    return TIMELINES


if __name__ == "__main__":
    pathenv = os.getenv("OUTPUT_PATH")
    if pathenv is None:
        pathenv = "."
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "assembly_code", help="Path to the assembly code. Must be the first parameter."
    )
    parser.add_argument(
        "--trace_file", help="Filter for trace files", default=None, type=str
    )
    parser.add_argument(
        "--att_kernel", help="Kernel file", type=str, default=os.path.join(pathenv, "*_kernel.txt")
    )
    parser.add_argument("--ports", help="Server and websocket ports, default: 8000,18000")
    parser.add_argument(
        "--mode",
        help="""ATT analysis modes:\n
                        off: Only run ATT collection, disable analysis.\n
                        file: dump json files to disk.""",
        type=str,
        default="off",
    )
    args = parser.parse_args()
    args.mode = args.mode.lower().split(',')

    CSV_MODE = False
    FILE_MODE = False
    if 'csv' in args.mode:
        CSV_MODE = True
    if 'file' in args.mode:
        FILE_MODE = True

    if not CSV_MODE and not FILE_MODE:
        print("Skipping analysis.")
        quit()

    if os.getenv("COUNTERS_PATH"):
        with open(os.getenv("COUNTERS_PATH"), "r") as f:
            lines = [l.split("//")[0] for l in f.readlines()]

            EVENT_NAMES = []
            clean = lambda x: x.split("=")[1].split(" ")[0].split("\n")[0]
            for line in lines:
                if "PERFCOUNTER_ID=" in line:
                    EVENT_NAMES += ["id: " + clean(line)]
            for line in lines:
                if "PERFCOUNTER=" in line:
                    EVENT_NAMES += [clean(line).split("SQ_")[1].lower()]

    att_kernel_list = glob.glob(args.att_kernel)

    if len(att_kernel_list) == 0:
        print("Could not find att output kernel:", args.att_kernel)
        quit()

    for att_kernel in att_kernel_list:
        print('Parsing:', att_kernel)
        att_kernel_f = []
        with open(att_kernel, 'r') as f:
            for line in f:
                att_kernel_f.append(line.split('\n')[0])

        # returns element index of the file:// or memory:// filepath in the string
        get_path_loc = lambda x: np.sum([len(m) for m in x.split(' ')[:3]])+3
        # adds the OUTPUT_PATH env variable to a filepath if necessary
        add_pathnv = lambda x: x[:get_path_loc(x)] + os.path.join(pathenv, x[get_path_loc(x):])
        # returns the memory address in the string
        get_addr = lambda x: int(x.split(' ')[0][2:], 16)
        # Sets a preference for 'file' paths to be added before 'memory' paths. Sorts by addr.
        get_addr_preference = lambda x: [0 if 'file' in x[get_path_loc(x):] else 1<<60][0]

        # Get the GPU id in the string
        gpu_id = int(att_kernel_f[0].split(' ')[2].split('GPU[')[1].split(']')[0])
        # Eliminame first line in the att_kernel txt file and adds the OUTPUT_PATH as needed
        att_kernel_f = [add_pathnv(p) if '.out' == p[-4:] else p for p in att_kernel_f[1:]]
        # Sorts the list of codeobj by address, with 'file' given preference
        att_kernel_f = sorted(att_kernel_f, key=lambda x: get_addr(x)+get_addr_preference(x))
        assembly_code = deepcopy(args.assembly_code)

        # Assembly parsing
        bIsAuto = False
        if assembly_code.lower().strip() == 'auto':
            bIsAuto = True
            code = [['; Begin ATT ASM', 100, 0, '', 0, 0, 0, 0, 0]]
            jumps = []
        elif not Path(assembly_code).is_file():
            print("Invalid assembly_code('{0}')!".format(assembly_code))
            sys.exit(1)
        else:
            code, jumps = parse_binary(assembly_code, att_kernel)

        # Trace Parsing
        trace_instance_name = att_kernel.split("_kernel.txt")[0]
        if args.trace_file is None:
            filenames = glob.glob(trace_instance_name + "_*.att")
        else:
            filenames = glob.glob(args.trace_file)

        if len(filenames) == 0:
            print("Could not find trace files for", att_kernel)
            continue

        print('Att kernel:', att_kernel)

        DBFILES = []
        EVENTS = []
        OCCUPANCY = []
        GFXV = []
        analysed_filenames = []
        occupancy_filenames = []
        kernel_addr = {}

        latency_map = np.zeros((len(code)), dtype=np.int64)
        hitcount_map = np.zeros((len(code)), dtype=np.int32)

        gc.collect()
        if bIsAuto:
            codeservice = service.CodeobjService(gpu_id, att_kernel_f, SO.classify_asm_line)
        else:
            codeservice = None

        for name in filenames:
            traces, waves, perfevents, occupancy, gfxv, addrs, ftrace = getWaves_binary(name)
            if gfxv is None:
                continue
            if CSV_MODE == False and ftrace == 0:
                print('Generating occupancy information')

            for id, addr in enumerate(addrs):
                kernel_addr[id] = addr
            if len(occupancy) > 1:
                OCCUPANCY.append( occupancy )
                occupancy_filenames.append(name)

            if np.sum([0]+[len(s.instructions) for id, s in traces.items()]) == 0:
                print("No traces from", name)
                continue

            hitcount_map, latency_map = getWaves_stitch(traces, code, jumps, gfxv, latency_map, hitcount_map, bIsAuto, codeservice)

            analysed_filenames.append(name)
            EVENTS.append(perfevents)
            DBFILES.append( persist(name, waves, traces) )
            GFXV.append(gfxv)

        gc.collect()
        for k in range(len(code)):
            code[k][-2] = int(hitcount_map[k])
            code[k][-1] = int(latency_map[k])
            try:
                code[k][-3] = codeservice.ToRawPC(code[k][-3])
            except:
                pass
            if k > 0 and code[k-1][-3] == 0:
                code[k-1][-3] = code[k][-3]

        if CSV_MODE:
            from att_to_csv import dump_csv
            dump_csv(code, trace_instance_name, bIsAuto)

        if FILE_MODE:
            try:
                dispatchNames = {id: codeservice.getSymbolName(addr) for id, addr in kernel_addr.items()}
            except:
                dispatchNames = {id: "#addr"+str(addr) for id, addr in kernel_addr.items()}
            drawinfo = {
                "TIMELINES": gen_timelines(DBFILES),
                "EVENTS": EVENTS,
                "EVENT_NAMES": EVENT_NAMES,
                "OCCUPANCY": OCCUPANCY,
                "ShaderNames": occupancy_filenames,
                "DispatchNames": dispatchNames
            }
            view_trace(
                code,
                DBFILES,
                analysed_filenames,
                0,
                gfxv,
                drawinfo,
                trace_instance_name
            )
