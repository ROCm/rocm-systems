#!/usr/bin/env python3
import sys

if sys.version_info[0] < 3:
    raise Exception("Must be using Python 3")

from collections import defaultdict
from copy import deepcopy

MAX_STITCHED_TOKENS = 200000000
MAX_FAILED_STITCHES = 256

SKIP = 0
SMEM = 1
SALU = 2
VMEM = 3
FLAT = 4
LDS = 5
VALU = 6
JUMP = 7
NEXT = 8
IMMED = 9
BRANCH = 10
GETPC = 11
SETPC = 12
SWAPPC = 13
LANEIO = 14
PCINFO = 15
WAVE_ENDED = 16
DONT_KNOW = 100

# Keeps track of register states for hipcc-generated assembly
class RegisterWatchList:
    def __init__(self, labels, code, jump_map, insts):
        self.registers = {"v" + str(k): [[] for m in range(64)] for k in range(64)}
        for k in range(128):
            self.registers["s" + str(k)] = []
        self.labels = labels

        self.code = code
        self.jump_map = jump_map
        self.insts = insts

    def jump(self, as_line):
        return self.jump_map[as_line[2]]

    def getcode(self, line):
        return self.code[line], 1

    def getincrement(self, line):
        return 1

    def try_translate(self, tok):
        if tok[0] in ["s"]:
            return self.registers[self.range(tok)[0]]
        elif "@" in tok:
            return self.labels[tok.split("@")[0]] + 1

    def range(self, r):
        reg = r.split(":")
        if len(reg) == 1:
            return reg
        else:
            r0 = reg[0].split("[")
            return [r0[0] + str(k) for k in range(int(r0[1]), int(reg[1][:-1]) + 1)]

    def tokenize(self, line):
        return [
            u for u in [t.split(",")[0].strip() for t in line.split(" ")] if len(u) > 0
        ]

    def getpc(self, line, next_line):
        try:
            dst = line.split(" ")[1].strip()
            label_dests = []
            try:
                label_dests = next_line[0].split(", ")
            except:
                pass
            try:
                label_dests.append(next_line[0].split(", ")[-1].split("@")[0])
            except:
                pass

            for label_dst in label_dests:
                try:
                    cur_label = self.labels[label_dst]
                    for reg in self.range(dst):
                        self.registers[reg] = deepcopy(cur_label)
                except:
                    pass
        except:
            pass

    def swappc(self, line, line_num, inst_num):
        try:
            tokens = self.tokenize(line)
            dst = tokens[1]
            src = tokens[2]

            popped = deepcopy(self.registers[self.range(src)[0]])
            self.registers[self.range(dst)[0]] = line_num + 1
            return popped
        except:
            return -1

    def setpc(self, line, inst_num):
        try:
            src = line.split(' ')[1].strip()
            return deepcopy(self.registers[self.range(src)[0]])
        except:
            return -1

    def scratch(self, line):
        try:
            tokens = self.tokenize(line)
            if "_load" in tokens[0]:
                dst = tokens[1]
                src = tokens[3] + tokens[4]
            else:
                src = tokens[2]
                dst = tokens[3] + tokens[4]
            self.registers[dst] = deepcopy(self.registers[src])
        except:
            pass

    def move(self, line):
        try:
            tokens = self.tokenize(line)
            if tokens[2][0] in ["s", "d"] and tokens[1][0] in ["s", "d"]:
                self.registers[self.range(tokens[1])[0]] = deepcopy(
                    self.registers[self.range(tokens[2])[0]]
                )
        except:
            pass

    def updatelane(self, line):
        tokens = self.tokenize(line)
        try:
            if "v_readlane" in tokens[0]:
                self.registers[tokens[1]] = deepcopy(self.registers[tokens[2]][int(tokens[3])])
            elif "v_writelane" in tokens[0]:
                self.registers[tokens[1]][int(tokens[3])] = deepcopy(self.registers[tokens[2]])
        except:
            pass

    # Matches tokens in reverse order
    def try_match_swapped(self, i, line, increment):
        try:
            return self.insts[i + 1].type == self.code[line][1] and self.insts[i].type == self.code[line + 1][1]
        except:
            return False

# Translates PC values to instructions, for auto captured ISA
class PCTranslator:
    def __init__(self, insts, code, raw_code, reverse_map, codeservice):
        self.codeservice = codeservice

        self.insts = insts
        self.addrmap = {c[-3] : (c, self.codeservice.GetInstruction(c[-3])[3]) for c in code if c[-3] > 0}

        self.code = code
        self.raw_code = raw_code
        self.reverse_map = reverse_map
        self.jump_map = {c[-3] : self.getjump_loc(c) for c in code if c[1] == BRANCH}

    def jump(self, as_line):
        return self.jump_map[as_line[-3]]

    def addsymbol(self, addr):
        if addr in self.addrmap:
            return

        symbol = self.codeservice.getSymbolName(addr)
        if symbol is None:
            symbol = "Unkown symbol at 0x" + hex(addr)

        last_line = self.raw_code[-1]
        newline = ['; ' + symbol, DONT_KNOW, last_line[2], '', last_line[4], last_line[5], 0, 0, 0]
        self.raw_code.append(newline)

    def getcode(self, addr):
        try:
            return self.addrmap[addr]
        except Exception as ex:
            new_inst = self.codeservice.GetInstruction(addr)
            if new_inst and new_inst[3]: # Check returned size > 0
                last_line = self.raw_code[-1]
                newline = [new_inst[1], new_inst[0], len(self.raw_code), new_inst[2], last_line[4]+1, last_line[5]+1, addr, 0, 0]
                if new_inst[0] == BRANCH:
                    self.jump_map[addr] = self.getjump_loc(newline)
                self.addrmap[addr] = (newline, new_inst[3])

                next = len(self.code)
                self.reverse_map[addr] = len(self.raw_code)
                self.raw_code.append(newline)
                self.code.append(newline)
                return newline, new_inst[3]
            else:
                raise ex

    def jump(self, asm_line):
        try:
            return self.jump_map[asm_line[-3]]
        except:
            loc = self.getjump_loc(asm_line)
            self.jump_map[asm_line[-3]] = loc
            return loc

    def getjump_loc(self, asm_line):
        try:
            dest = int(asm_line[0].split(' ')[-1])
            if dest >= 32768: dest -= 65536
            return asm_line[-3] + 4*dest+4
        except:
            return -1

    def getincrement(self, addr):
        return self.getcode(addr)[1]

    def try_translate(self, tok):
        pass
    def range(self, r):
        pass
    def tokenize(self, line):
        pass
    def getpc(self, line, next_line):
        pass
    def swappc(self, line, line_num, inst_index):
        try:
            inst_pos = inst_index+1
            while self.insts[inst_pos].type != PCINFO:
                inst_pos += 1
            return self.getcode(self.insts[inst_pos].cycles)[0][-3]
        except:
            print('SWAPPC warning: Could not find addr for', inst_index, line)
            return -1
    def setpc(self, line, inst_index):
        try:
            inst_pos = inst_index+1
            while self.insts[inst_pos].type != PCINFO:
                inst_pos += 1
            return self.getcode(self.insts[inst_pos].cycles)[0][-3]
        except:
            print('SETPC warning: Could not find addr for', inst_index, line)
            return -1
    def scratch(self, line):
        pass
    def move(self, line):
        pass
    def updatelane(self, line):
        pass

    # Matches tokens in reverse order
    def try_match_swapped(self, i, addr, increment):
        try:
            return  self.insts[i + 1].type == self.getcode(addr)[0][1] and \
                    self.insts[i].type == self.getcode(addr + increment)[0][1]
        except Exception as e:
            return False


def stitch(insts, raw_code, jumps, gfxv, bIsAuto, codeservice):
    bGFX9 = gfxv == 'vega'

    result, i, loopCount = [], 0, defaultdict(int)

    SMEM_INST = []  # scalar memory
    VLMEM_INST = []  # vector memory load
    VSMEM_INST = []  # vector memory store
    FLAT_INST = []
    NUM_SMEM = 0
    NUM_VLMEM = 0
    NUM_VSMEM = 0
    NUM_FLAT = 0
    skipped_immed = 0

    mem_unroll = []
    flight_count = []

    labels = {}
    jump_map = [0]

    # Clean the code and remove comments
    code = [raw_code[0]]
    for c in raw_code[1:]:
        c = list(c)
        c[0] = c[0].split(";")[0].split("//")[0].strip()
        jump_map.append(len(code))

        if c[1] != DONT_KNOW:
            code.append(c)
        elif ":" in c[0]:
            labels[c[0].split(":")[0]] = len(code)

    reverse_map = {}
    if bIsAuto:
        for k, v in enumerate(jump_map):
            try:
                reverse_map[code[v][-3]] = k
            except:
                pass
    else:
        for k, v in enumerate(jump_map):
            reverse_map[v] = k

    jumps = {jump_map[j] + 1: j for j in jumps}

    # Checks if we have guaranteed ordering in memory operations
    smem_ordering = 0
    vlmem_ordering = 0
    vsmem_ordering = 0

    num_failed_stitches = 0
    loops = 0
    maxline = 0

    pcskip = []
    if bIsAuto:
        try:
            firstinst = insts[0]

            if firstinst.type != PCINFO:
                print('Warning: Waves without PCINFO')
                return None
            elif firstinst.cycles == 0:
                print('Info: Some waves started before the trace')
                return None

            watchlist = PCTranslator(insts, code, raw_code, reverse_map, codeservice)
            watchlist.addsymbol(firstinst.cycles)
            line = firstinst.cycles
            lineincrement = watchlist.getincrement(line)
        except KeyError as e:
            print('Warning: Waves from addr', hex(e.args[0]), 'have no codeobj info.')
            for i in range(len(insts)):
                insts[i].asmline = 0
            return [i for k, i in enumerate(insts) if i.type != PCINFO], [], [], [], 1, 0, [k for k, i in enumerate(insts) if i.type == PCINFO]
        except Exception as e:
            print('Unknown error', e)
            return None
    else:
        line = 0
        lineincrement = 1
        watchlist = RegisterWatchList(labels=labels, code=code, jump_map=jump_map, insts=insts)

    N = len(insts)

    while i < N and line >= 0 and loops < MAX_STITCHED_TOKENS:
        if insts[i].type == PCINFO:
            pcskip.append(i)
            i += 1
            continue
        loops += 1

        inst = insts[i]
        try:
            as_line, lineincrement = watchlist.getcode(line)
        except:
            break

        matched = True
        next = line + lineincrement

        if not bIsAuto:
            if '_mov_' in as_line[0]:
                watchlist.move(as_line[0])
            elif 'scratch_' in as_line[0]:
                watchlist.scratch(as_line[0])

        if as_line[1] == DONT_KNOW or (as_line[1] == SKIP and not bGFX9):
            matched = False
        elif as_line[1] == GETPC:
            try:
                watchlist.getpc(as_line[0], watchlist.getcode(next)[0])
                matched = inst.type in [SALU, JUMP]
            except:
                matched = False
        elif as_line[1] == LANEIO:
            watchlist.updatelane(as_line[0])
            matched = inst.type == VALU
        elif as_line[1] == SETPC:
            next = watchlist.setpc(as_line[0], i)
            matched = inst.type in [SALU, JUMP]
            i += 1
            pcskip.append(i)
            while bIsAuto and next < 0 and i+1 < len(insts):
                i += 1
                if insts[i].type == PCINFO:
                    pcskip.append(i)
                    next = watchlist.setpc(as_line[0], i-1)
                else:
                    inst.cycles += insts[i].cycles
            if next < 0:
                print('Jump to unknown location in line', as_line[0])
                break
        elif as_line[1] == SWAPPC:
            matched = inst.type in [SALU, JUMP]
            next = watchlist.swappc(as_line[0], line, i)
            i += 1
            pcskip.append(i)
            while bIsAuto and next < 0 and i+1 < len(insts):
                i += 1
                if insts[i].type == PCINFO:
                    next = watchlist.swappc(as_line[0], line, i-1)
                    pcskip.append(i)
                else:
                    inst.cycles += insts[i].cycles
            if next < 0:
                print('Jump to unknown location in line', as_line[0])
                break
        elif inst.type == as_line[1]:
            if line in jumps:
                loopCount[jumps[line] - 1] += 1
            num_inflight = NUM_FLAT + NUM_SMEM + NUM_VLMEM + NUM_VSMEM

            if inst.type == SMEM or inst.type == LDS:
                smem_ordering = 1 if inst.type == SMEM else smem_ordering
                SMEM_INST.append([reverse_map[line], num_inflight])
                NUM_SMEM += 1
            elif inst.type == VMEM or (inst.type == FLAT and "global_" in as_line[0]):
                inc_ordering = False
                if "flat_" in as_line[0]:
                    inc_ordering = True

                if not bGFX9 and "store" in as_line[0]:
                    VSMEM_INST.append([reverse_map[line], num_inflight])
                    NUM_VSMEM += 1
                    if inc_ordering:
                        vsmem_ordering = 1
                else:
                    VLMEM_INST.append([reverse_map[line], num_inflight])
                    NUM_VLMEM += 1
                    if inc_ordering:
                        vlmem_ordering = 1
            elif inst.type == FLAT:
                smem_ordering = 1
                vlmem_ordering = 1
                vsmem_ordering = 1
                FLAT_INST.append([reverse_map[line], num_inflight])
                NUM_FLAT += 1
            elif inst.type == IMMED and "s_wait" in as_line[0] and not "s_wait_alu" in as_line[0]:
                if "lgkmcnt" in as_line[0] or "dscnt" in as_line[0] or "kmcnt" in as_line[0]:
                    try:
                        wait_N = int(as_line[0].split("lgkmcnt(")[1].split(")")[0])
                    except:
                        wait_N = 0
                    flight_count.append([as_line[5], num_inflight, wait_N])
                    if wait_N == 0:
                        smem_ordering = 0
                    if smem_ordering == 0:
                        offset = len(SMEM_INST) - wait_N
                        mem_unroll.append(
                            [reverse_map[line], SMEM_INST[:offset] + FLAT_INST]
                        )
                        SMEM_INST = SMEM_INST[offset:]
                        NUM_SMEM = len(SMEM_INST)
                        FLAT_INST = []
                        NUM_FLAT = 0
                    else:
                        NUM_SMEM = min(max(wait_N - NUM_FLAT, 0), NUM_SMEM)
                        NUM_FLAT = min(max(wait_N - NUM_SMEM, 0), NUM_FLAT)
                    num_inflight = NUM_FLAT + NUM_SMEM + NUM_VLMEM + NUM_VSMEM

                if "vmcnt" in as_line[0] or "loadcnt" in as_line[0]:
                    try:
                        wait_N = int(as_line[0].split("vmcnt(")[1].split(")")[0])
                    except:
                        wait_N = 0
                    flight_count.append([as_line[5], num_inflight, wait_N])
                    if wait_N == 0:
                        vlmem_ordering = 0
                    if vlmem_ordering == 0:
                        offset = len(VLMEM_INST) - wait_N
                        mem_unroll.append(
                            [reverse_map[line], VLMEM_INST[:offset] + FLAT_INST]
                        )
                        VLMEM_INST = VLMEM_INST[offset:]
                        NUM_VLMEM = len(VLMEM_INST)
                        FLAT_INST = []
                        NUM_FLAT = 0
                    else:
                        NUM_VLMEM = min(max(wait_N - NUM_FLAT, 0), NUM_VLMEM)
                        NUM_FLAT = min(max(wait_N - NUM_VLMEM, 0), NUM_FLAT)
                    num_inflight = NUM_FLAT + NUM_SMEM + NUM_VLMEM + NUM_VSMEM

                if "vscnt" in as_line[0] or (bGFX9 and "vmcnt" in as_line[0]) or "storecnt" in as_line[0]:
                    try:
                        wait_N = int(as_line[0].split('vscnt(')[1].split(')')[0])
                    except:
                        try:
                            wait_N = int(as_line[0].split('vmcnt(')[1].split(')')[0])
                        except:
                            wait_N = 0
                    flight_count.append([as_line[5], num_inflight, wait_N])
                    if wait_N == 0:
                        vsmem_ordering = 0
                    if vsmem_ordering == 0:
                        offset = len(VSMEM_INST) - wait_N
                        mem_unroll.append(
                            [reverse_map[line], VSMEM_INST[:offset] + FLAT_INST]
                        )
                        VSMEM_INST = VSMEM_INST[offset:]
                        NUM_VSMEM = len(VSMEM_INST)
                        FLAT_INST = []
                        NUM_FLAT = 0
                    else:
                        NUM_VSMEM = min(max(wait_N - NUM_FLAT, 0), NUM_VSMEM)
                        NUM_FLAT = min(max(wait_N - NUM_VSMEM, 0), NUM_FLAT)
                    num_inflight = NUM_FLAT + NUM_SMEM + NUM_VLMEM + NUM_VSMEM

        elif inst.type == JUMP and as_line[1] == BRANCH:
            next = watchlist.jump(as_line)
            if next is None or next == 0:
                print("Jump to unknown location!", as_line)
                break
        elif inst.type == NEXT and as_line[1] == BRANCH:
            pass
        else:
            matched = False
            if watchlist.try_match_swapped(i, line, lineincrement):
                temp = insts[i]
                insts[i] = insts[i + 1]
                insts[i + 1] = temp
                next = line
            else:
                hasWait = "s_wait" in as_line[0]
                if hasWait or "_load_" in as_line[0]:
                    if skipped_immed > 0 and hasWait:
                        matched = True
                        skipped_immed -= 1
                    elif 's_waitcnt' in as_line[0] and 'scratch_' not in as_line[0]:
                        print('WARNING: Parsing terminated at:', as_line)
                        break

        if matched or as_line[1] != DONT_KNOW:
            if matched:
                inst.asmline = reverse_map[line]
                result.append(inst)
                i += 1
                num_failed_stitches = 0
            elif not bGFX9 and inst.type == IMMED and line != next and as_line[1] != SKIP:
                skipped_immed += 1
                inst.asmline = reverse_map[line]
                result.append(inst)
                if 's_barrier' in as_line[0]:
                    next = line + lineincrement
                i += 1
            else:
                num_failed_stitches += 1

        maxline = max(reverse_map[line], maxline)
        line = next

    N = max(N, 1)
    if i != N and (insts[i].type == WAVE_ENDED or i == N-1):
        print('Warning - Wave ended.')
    elif i < N:
        print('Warning - Stitching rate: '+str(i * 100 / N)+'% matched', i, ' of ', N)
        try:
            print(line, code[line])
        except:
            pass
    else:
        while line < len(code):
            if "s_endpgm" in code[line]:
                mem_unroll.append(
                    [reverse_map[line], SMEM_INST + VLMEM_INST + VSMEM_INST + FLAT_INST]
                )
                break
            line += 1
        print('Success: Parsed', i, 'tokens')

    return result, loopCount, mem_unroll, flight_count, maxline, len(result), pcskip
