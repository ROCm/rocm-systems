#!/usr/bin/env python3
import sys

if sys.version_info[0] < 3:
    raise Exception("Must be using Python 3")

from collections import defaultdict
from copy import deepcopy

MAX_STITCHED_TOKENS = 100000000
MAX_FAILED_STITCHES = 256
STACK_SIZE_LIMIT = 64

UNKNOWN = 0
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
DONT_KNOW = 100

WaveInstCategory = {
    UNKNOWN: "UNKNOWN",
    SMEM: "SMEM",
    SALU: "SALU",
    VMEM: "VMEM",
    FLAT: "FLAT",
    LDS: "LDS",
    VALU: "VALU",
    JUMP: "JUMP",
    NEXT: "NEXT",
    IMMED: "IMMED",
    JUMP: "JUMP",
    NEXT: "NEXT",
    IMMED: "IMMED",
    BRANCH: "BRANCH",
    GETPC: "GETPC",
    SETPC: "SETPC",
    SWAPPC: "SWAPPC",
    LANEIO: "LANEIO",
    PCINFO: "PCINFO",
    DONT_KNOW: "DONT_KNOW",
}

# Keeps track of register states for hipcc-generated assembly
class RegisterWatchList:
    def __init__(self, labels):
        self.registers = {"v" + str(k): [[] for m in range(64)] for k in range(64)}
        for k in range(64):
            self.registers["s" + str(k)] = []
        self.labels = labels

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
        # print('Get pc:', line)
        try:
            dst = line.split(" ")[1].strip()
            label_dest = next_line.split(", ")[-1].split("@")[0]
            for reg in self.range(dst):
                self.registers[reg].append(deepcopy(self.labels[label_dest]))
        except:
            pass

    def swappc(self, line, line_num, inst_num):
        try:
            tokens = self.tokenize(line)
            dst = tokens[1]
            src = tokens[2]

            popped = self.registers[self.range(src)[0]][-1]
            self.registers[self.range(src)[0]] = self.registers[self.range(src)[0]][:-1]
            self.registers[self.range(dst)[0]].append(line_num + 1)
            return popped
        except:
            return 0

    def setpc(self, line, inst_num):
        try:
            src = line.split(' ')[1].strip()
            popped = self.registers[self.range(src)[0]][-1]
            self.registers[self.range(src)[0]] = self.registers[self.range(src)[0]][:-1]
            return popped
        except:
            return 0

    def scratch(self, line):
        try:
            tokens = self.tokenize(line)
            if "_load" in tokens[0]:
                dst = tokens[1]
                src = tokens[3] + tokens[4]
            else:
                src = tokens[2]
                dst = tokens[3] + tokens[4]
            self.registers[dst] = self.registers[src]
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
                self.registers[tokens[1]].append(
                    self.registers[tokens[2]][int(tokens[3])][-1]
                )
                self.registers[tokens[2]][int(tokens[3])] = self.registers[tokens[2]][
                    int(tokens[3])
                ][:-1]
            elif "v_writelane" in tokens[0]:
                self.registers[tokens[1]][int(tokens[3])].append(
                    self.registers[tokens[2]][-1]
                )
                self.registers[tokens[2]] = self.registers[tokens[2]][-STACK_SIZE_LIMIT:]
        except Exception as e:
            pass

# Translates PC values to instructions, for auto captured ISA
class PCTranslator:
    def __init__(self, code, insts):
        self.code = code
        self.insts = insts
        self.addrmap = {code[m][-3] : m for m in range(len(code))}

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
            loc = self.addrmap[self.insts[inst_index+1][2]]
            return loc
        except:
            print('SWAPPC: Could not find addr', self.insts[inst_index+1][2], 'for', line)
            return -1
    def setpc(self, line, inst_index):
        try:
            loc = self.addrmap[self.insts[inst_index+1][2]]
            return loc
        except:
            print('SETPC: Could not find addr', self.insts[inst_index+1][2], 'for', line)
            return -1
    def scratch(self, line):
        pass
    def move(self, line):
        pass
    def updatelane(self, line):
        pass

# Matches tokens in reverse order
def try_match_swapped(insts, code, i, line):
    return insts[i + 1][1] == code[line][1] and insts[i][1] == code[line + 1][1]


FORK_NAMES = 1
# A successful parsed instruction
class CachedInst:
    def __init__(self, inst, as_line):
        self.inst_type = inst
        self.as_line = as_line
        self.forks = None

# A branch of the parsing tree
class Fork:
    def __init__(self):
        global FORK_NAMES
        self.insts = []
        self.data = None
        self.name = FORK_NAMES
        FORK_NAMES += 1
        # print('Created new fork: ', self.name)

# Try to match sequence "insts" with the branch "fork", starting at position "i"
def move_down_fork(fork, insts, i): #(fork : Fork, insts : list, i : int):
    N = min(len(insts), len(fork.insts))

    while i < N:
        if insts[i][1] == fork.insts[i].inst_type:
            i += 1
        elif i<N-1  and insts[i+1][1] == fork.insts[i].inst_type \
                    and insts[i][1] == fork.insts[i+1].inst_type:
            i += 2
        else:
            return False, i

    if len(fork.insts) != len(insts):
        return False, i

    return True, i


FORK_TREE = Fork()

# Check if there exists a previous wave with the same sequence of instructions executed
def fromDict(insts):
    i = 0
    N = len(insts)
    cur_fork = FORK_TREE
    while i < N:
        tillEnd, final_pos = move_down_fork(cur_fork, insts, i)
        if tillEnd:
            # print('Reached end')
            return True, cur_fork

        i += final_pos

        if i >= len(cur_fork.insts):
            return False, cur_fork

        last_inst = cur_fork.insts[i]
        if last_inst.forks is None:
            last_inst.forks = []

        bMatchFork = False
        for fork in last_inst.forks:
            if fork.insts[0].inst_type == insts[0][1]:
                cur_fork = fork
                bMatchFork = True
                break
        if not bMatchFork:
            cur_fork = Fork()
            last_inst.forks.append(cur_fork)
            return False, cur_fork

    print("Warning: Reached end of loop!")
    return False, cur_fork


def stitch(insts, raw_code, jumps, gfxv, bIsAuto):
    bGFX9 = gfxv == 'vega'

    # Try from cached result from a previous wave that have already been parsed
    dict_sucess, current_fork = fromDict(insts)
    if dict_sucess:
        result, loopCount, mem_unroll, flight_count, maxline, pcsequence = current_fork.data
        # Check if the sequence of measured PC values are equal for cached and new wave
        if len(pcsequence) > 0:
            pcs = [r[2] for r in insts if r[1] == PCINFO]
            if len(pcs) != len(pcsequence):
                dict_sucess = False
            for pc1, pc2 in zip(pcs, pcsequence):
                if pc1 != pc2:
                    dict_sucess = False

    # If successful, use resulting assembly from cache
    if dict_sucess:
        result = [r+(asm[-1],) for r, asm in zip(insts, result)]
        return result, loopCount, mem_unroll, flight_count, maxline, len(result)

    result, i, line, loopCount, N = [], 0, 0, defaultdict(int), len(insts)


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
        if bIsAuto and '; Begin ' == c[0][:len('; Begin ')]:
            if '; Begin <Kernel>' in c[0]:
                line = len(code)
                print('Begin at:', line, c)
        c = list(c)
        c[0] = c[0].split(";")[0].split("//")[0].strip()

        if c[1] != 100:
            code.append(c)
        elif ":" in c[0]:
            labels[c[0].split(":")[0]] = len(code)
        jump_map.append(len(code) - 1)

    reverse_map = []
    for k, v in enumerate(jump_map):
        if v >= len(reverse_map):
            reverse_map.append(k)

    jumps = {jump_map[j] + 1: j for j in jumps}

    # Checks if we have guaranteed ordering in memory operations
    smem_ordering = 0
    vlmem_ordering = 0
    vsmem_ordering = 0

    num_failed_stitches = 0
    loops = 0
    maxline = 0

    watchlist = RegisterWatchList(labels=labels) if not bIsAuto else PCTranslator(code, insts)

    pcsequence = []
    while i < N:
        loops += 1
        if line >= len(code) or loops > MAX_STITCHED_TOKENS \
            or num_failed_stitches > MAX_FAILED_STITCHES:
            break

        maxline = max(reverse_map[line], maxline)
        inst = insts[i]
        as_line = code[line]

        matched = True
        next = line + 1

        if not bIsAuto:
            if '_mov_' in as_line[0]:
                watchlist.move(as_line[0])
            elif 'scratch_' in as_line[0]:
                watchlist.scratch(as_line[0])

        if as_line[1] == GETPC:
            try:
                watchlist.getpc(as_line[0], code[line+1][0])
                matched = inst[1] in [SALU, JUMP]
            except:
                matched = False
        elif as_line[1] == LANEIO:
            watchlist.updatelane(as_line[0])
            matched = inst[1] == VALU
        elif as_line[1] == SETPC:
            next = watchlist.setpc(as_line[0], i)
            matched = inst[1] in [SALU, JUMP]
            if bIsAuto:
                matched = next >= 0
                i += 1
                result.append((insts[i][0], PCINFO, 0, 0, 0))
                pcsequence.append(insts[i][2])
        elif as_line[1] == SWAPPC:
            next = watchlist.swappc(as_line[0], line, i)
            matched = inst[1] in [SALU, JUMP]
            if bIsAuto:
                matched = next >= 0
                i += 1
                result.append((insts[i][0], PCINFO, 0, 0, 0))
                pcsequence.append(insts[i][2])
        elif inst[1] == as_line[1]:
            if line in jumps:
                loopCount[jumps[line] - 1] += 1
            num_inflight = NUM_FLAT + NUM_SMEM + NUM_VLMEM + NUM_VSMEM

            if inst[1] == SMEM or inst[1] == LDS:
                smem_ordering = 1 if inst[1] == SMEM else smem_ordering
                SMEM_INST.append([reverse_map[line], num_inflight])
                NUM_SMEM += 1
            elif inst[1] == VMEM or (inst[1] == FLAT and "global_" in as_line[0]):
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
            elif inst[1] == FLAT:
                smem_ordering = 1
                vlmem_ordering = 1
                vsmem_ordering = 1
                FLAT_INST.append([reverse_map[line], num_inflight])
                NUM_FLAT += 1
            elif inst[1] == IMMED and "s_waitcnt" in as_line[0]:
                if "lgkmcnt" in as_line[0]:
                    wait_N = int(as_line[0].split("lgkmcnt(")[1].split(")")[0])
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

                if "vmcnt" in as_line[0]:
                    wait_N = int(as_line[0].split("vmcnt(")[1].split(")")[0])
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

                if "vscnt" in as_line[0] or (bGFX9 and "vmcnt" in as_line[0]):
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

        elif inst[1] == JUMP and as_line[1] == BRANCH:
            next = jump_map[as_line[2]]
            if next is None or next == 0:
                print("Jump to unknown location!", as_line)
                break
        elif inst[1] == NEXT and as_line[1] == BRANCH:
            next = line + 1
        else:
            matched = False
            next = line + 1
            if i + 1 < N and line + 1 < len(code):
                if try_match_swapped(insts, code, i, line):
                    temp = insts[i]
                    insts[i] = insts[i + 1]
                    insts[i + 1] = temp
                    next = line
                elif "s_waitcnt " in as_line[0] or "_load_" in as_line[0]:
                    if skipped_immed > 0 and "s_waitcnt " in as_line[0]:
                        matched = True
                        skipped_immed -= 1
                    elif 'scratch_' not in as_line[0]:
                        print('Parsing terminated at:', as_line)
                        break

        if matched:
            result.append(inst + (reverse_map[line],))
            i += 1
            num_failed_stitches = 0
        elif not bGFX9 and inst[1] == IMMED and line != next:
            skipped_immed += 1
            result.append(inst + (reverse_map[line],))
            next = line
            i += 1
        else:
            num_failed_stitches += 1
        line = next

    N = max(N, 1)
    if i != N:
        print('Warning - Stitching rate: '+str(i * 100 / N)+'% matched')
        print('Leftovers:', [WaveInstCategory[insts[i+k][1]] for k in range(20) if i+k < len(insts)])
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

    current_fork.insts = [CachedInst(inst[1], inst[-1]) for inst in result]
    current_fork.data = result, loopCount, mem_unroll, flight_count, maxline, pcsequence
    result = [r for r in result if r[1] != PCINFO]
    return result, loopCount, mem_unroll, flight_count, maxline, len(result) if i == N else N
