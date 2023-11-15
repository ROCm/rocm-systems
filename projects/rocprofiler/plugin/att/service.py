import ctypes
from ctypes import *

pluginpath = '/home/giovanni/Desktop/rocprofiler/build/lib/rocprofiler/libatt_plugin.so'

attplugin = ctypes.CDLL(pluginpath)

attplugin.createService.restype = ctypes.c_uint64
attplugin.createService.argtypes = [ctypes.c_char_p, ctypes.c_uint64]
attplugin.deleteService.restype = ctypes.c_int
attplugin.deleteService.argtypes = [ctypes.c_uint64]
attplugin.getInstruction.restype = ctypes.c_char_p
attplugin.getInstruction.argtypes = [ctypes.c_uint64, ctypes.c_uint64]
attplugin.getCppref.restype = ctypes.c_char_p
attplugin.getCppref.argtypes = [ctypes.c_uint64, ctypes.c_uint64]
attplugin.getInstSize.restype = ctypes.c_size_t
attplugin.getInstSize.argtypes = [ctypes.c_uint64, ctypes.c_uint64]
attplugin.getSymbolName.restype = ctypes.c_char_p
attplugin.getSymbolName.argtypes = [ctypes.c_uint64]


class CodeobjInstance:
    def __init__(self, line, classification_func):
        tokens = line.split(' ')
        self.load_base = int(tokens[0], 16)
        self.load_end = self.load_base + int(tokens[1], 16)
        self.att_marker = int(tokens[2])
        self.fpath = tokens[3]

        self.handle = attplugin.createService(self.fpath.encode('utf-8'), self.load_base)
        self.classifier = classification_func

        if self.handle == 0:
            print('Warning: Could not open', line)
            raise

    def release(self):
        attplugin.deleteService(self.handle)

    def inrange(self, addr):
        return addr >= self.load_base and addr < self.load_end+0x1000

    def GetInstruction(self, addr):
        inst = attplugin.getInstruction(self.handle, addr)
        if inst is None:
            return None
        inst = inst.decode()
        while len(inst) and (inst[0] == '\t' or inst[0] == ' '):
            inst = inst[1:]
        while len(inst) and (inst[-1] == '\t' or inst[-1] == ' '):
            inst = inst[:-1]
        cpp = attplugin.getCppref(self.handle, addr)
        if cpp:
            cpp = cpp.decode()
        size = attplugin.getInstSize(self.handle, addr)
        if size and inst:
            return (self.classifier(inst.encode('utf-8'), len(inst)), inst, cpp, size)
        return None


class CodeobjService:
    def __init__(self, att_kernel_txt, cfunc) -> None:
        cfunc.restype = ctypes.c_int
        cfunc.argtypes = [ctypes.c_char_p, ctypes.c_size_t]

        self.last_instance = None
        self.services = []
        for line in att_kernel_txt:
            try:
                if 'memory://' == line[0:len('memory://')]:
                    continue
                self.services.append(CodeobjInstance(line, cfunc))
            except:
                pass

    def release(self):
        for _, _, instance in self.services:
            instance.release()

    def GetInstruction(self, addr):
        if self.last_instance and self.last_instance.inrange(addr):
            return self.last_instance.GetInstruction(addr)

        for instance in self.services:
            if instance.inrange(addr):
                self.last_instance = instance
                return instance.GetInstruction(addr)

        return None

    def getSymbolName(self, addr):
        name = attplugin.getSymbolName(addr)
        if name:
            return name.decode()
        return "Addr #"+hex(addr)
