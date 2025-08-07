#!/usr/bin/env python3
import sys

if sys.version_info[0] < 3:
    raise Exception("Must be using Python 3")

import ctypes
from ctypes import *
import os

HEADER_OFFSET = 62
HEADER_MASK = 0x3
ID_OFFSET = 34
ID_MASK = (1<<28)-1
OFFSET_MASK = (1<<ID_OFFSET)-1

pluginpath = '../../../lib/rocprofiler/libatt_plugin.so'
filedir = os.path.dirname(os.path.realpath(__file__))
attplugin = CDLL(os.path.join(filedir, pluginpath))

attplugin.getSymbolName.restype = c_char_p
attplugin.getSymbolName.argtypes = [c_uint64]

class instruction_info_t(ctypes.Structure):
    _fields_ = [('inst', c_char_p),
                ('cpp', c_char_p),
                ('size', c_size_t)]

attplugin.getInstructionFromAddr.restype = instruction_info_t
attplugin.getInstructionFromAddr.argtypes = [c_uint64]

attplugin.getInstructionFromID.restype = instruction_info_t
attplugin.getInstructionFromID.argtypes = [c_uint32, c_uint64]

attplugin.addDecoder.restype = c_int
attplugin.addDecoder.argtypes = [c_char_p, c_uint32, c_uint64, c_uint64, c_uint64]

attplugin.removeDecoder.restype = c_int
attplugin.removeDecoder.argtypes = [c_uint32, c_uint64]

def IsRawPC(addr):
    return addr >> HEADER_OFFSET == 0

def getID(addr):
    return (addr >> ID_OFFSET) & ID_MASK

def getOffset(addr):
    return addr & OFFSET_MASK

class CodeobjInstance:
    def __init__(self, gpu_id, line):
        tokens = line.split(' ')
        self.load_base = int(tokens[0], 16)
        self.memsize = int(tokens[1], 16)
        self.att_id = int(tokens[2])
        self.fpath = tokens[3]

        path = self.fpath.encode('utf-8')
        self.error = attplugin.addDecoder(path, self.att_id, self.load_base, self.memsize, gpu_id)
        if self.error != 0:
            print('Warning: Could not open', line)
            raise

    def release(self):
        attplugin.removeDecoder(self.att_id, self.load_base)


class CodeobjService:
    def __init__(self, gpu_id, att_kernel_txt, cfunc):
        cfunc.restype = ctypes.c_int
        cfunc.argtypes = [ctypes.c_char_p, ctypes.c_size_t]

        self.classifier = cfunc
        self.last_instance = None
        self.services = {}
        for line in att_kernel_txt:
            try:
                if 'memory://' == line[0:len('memory://')]:
                    continue
                service = CodeobjInstance(gpu_id, line)
                self.services[service.att_id] = service
            except:
                pass

    def ToRawPC(self, addr):
        if IsRawPC(addr):
            return addr
        return self.services[getID(addr)].load_base + getOffset(addr)

    def release(self):
        for _, instance in self.services.items():
            instance.release()

    def GetInstruction(self, addr):
        if not IsRawPC(addr):
            return self.GetInstructionFromID(getID(addr), getOffset(addr))
        else:
            return self.GetInstructionFromAddr(addr)

    def GetInstructionFromAddr(self, addr):
        info_inst = attplugin.getInstructionFromAddr(addr)
        if info_inst.size == 0 or info_inst.inst is None:
            return None
        inst = info_inst.inst.decode()
        cpp = info_inst.cpp
        if cpp:
            cpp = cpp.decode()

        while len(inst) and (inst[0] == '\t' or inst[0] == ' '):
            inst = inst[1:]
        while len(inst) and (inst[-1] == '\t' or inst[-1] == ' '):
            inst = inst[:-1]

        return (self.classifier(info_inst.inst, len(inst)), inst, cpp, info_inst.size)

    def GetInstructionFromID(self, id, offset):
        info_inst = attplugin.getInstructionFromID(id, offset)
        if info_inst.size == 0 or info_inst.inst is None:
            return None
        inst = info_inst.inst.decode()
        cpp = info_inst.cpp
        if cpp:
            cpp = cpp.decode()
        else:
            cpp = ''

        while len(inst) and (inst[0] == '\t' or inst[0] == ' '):
            inst = inst[1:]
        while len(inst) and (inst[-1] == '\t' or inst[-1] == ' '):
            inst = inst[:-1]

        return (self.classifier(info_inst.inst, len(inst)), inst, cpp, info_inst.size)

    def getSymbolName(self, addr):
        try:
            name = attplugin.getSymbolName(self.ToRawPC(addr))
            if name:
                return name.decode()
            return "Addr #"+hex(self.ToRawPC(addr))
        except:
            return "Addr #"+hex(addr)
