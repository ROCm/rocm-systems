#!/usr/bin/env python3
import sys
if sys.version_info[0] < 3:
    raise Exception("Must be using Python 3")

import os
import sys
import time
import socket
from pathlib import Path
from struct import *
from collections import defaultdict
import json
import time
import http.server
import socketserver
import socket
import asyncio
import websockets
from multiprocessing import Process, Manager
import numpy as np
from copy import deepcopy
from http import HTTPStatus

class Readable:
    def __init__(self, jsonstring) -> None:
        self.jsonstr = json.dumps(jsonstring)
        self.seek = 0

    def read(self, length=0):
        if length<=0:
            return self.jsonstr
        else:
            if self.seek >= len(self):
                self.seek = 0
                return None
            response =  self.jsonstr[self.seek:self.seek+length]
            self.seek += length
            return bytes(response, 'utf-8')

    def __len__(self):
        return len(self.jsonstr)

STACK_SIZE_LIMIT = 64

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
DONT_KNOW = 100

WaveInstCategory = {
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
    DONT_KNOW: "DONT_KNOW",
}

JSON_GLOBAL_DICTIONARY = {}

class RegisterWatchList:
    def __init__(self, labels) -> None:
        self.registers = {'v'+str(k): [[] for m in range(64)] for k in range(64)}
        self.registers = {**self.registers, **{'s'+str(k): [] for k in range(64)}}
        self.labels = labels

    def try_translate(self, tok):
        if tok[0] in ['s']:
            return self.registers[self.range(tok)[0]]
        elif '@' in tok:
            return self.labels[tok.split('@')[0]]+1

    def range(self, r):
        reg = r.split(':')
        if len(reg) == 1:
            return reg
        else:
            r0 = reg[0].split('[')
            return [r0[0]+str(k) for k in range(int(r0[1]), int(reg[1][:-1])+1)]

    def tokenize(self, line):
        return [u for u in [t.split(',')[0].strip() for t in line.split(' ')] if len(u) > 0]

    def getpc(self, line, next_line):
        #print('Get pc:', line)
        dst = line.split(' ')[1].strip()
        label_dest = next_line.split(', ')[-1].split('@')[0]
        for reg in self.range(dst):
            #print('Setting:', reg, label_dest, self.labels[label_dest])
            self.registers[reg].append(deepcopy(self.labels[label_dest]))

    def swappc(self, line, line_num):
        #print('swappc pc:', line)
        tokens = self.tokenize(line)
        dst = tokens[1]
        src = tokens[2]
        #print('swap to', self.registers[self.range(src)[0]])
        self.registers[self.range(dst)[0]].append(line_num+1)
        popped = self.registers[self.range(src)[0]][-1]
        self.registers[self.range(src)[0]] = self.registers[self.range(src)[0]][:-1]
        return popped

    def setpc(self, line):
        #print('Set pc:', line)
        src = line.split(' ')[1].strip()
        #print('Going to:', self.registers[self.range(src)[0]], src)
        popped = self.registers[self.range(src)[0]][-1]
        self.registers[self.range(src)[0]] = self.registers[self.range(src)[0]][:-1]
        return popped

    def updatelane(self, line):
        tokens = self.tokenize(line)
        try:
            #print('Lane:', tokens)
            if 'v_readlane' in tokens[0]:
                self.registers[tokens[1]].append(self.registers[tokens[2]][int(tokens[3])][-1])
                #print('Writelane value', self.registers[tokens[2]][int(tokens[3])])
                self.registers[tokens[2]][int(tokens[3])] = self.registers[tokens[2]][int(tokens[3])][:-1]
            elif 'v_writelane' in tokens[0]:
                self.registers[tokens[1]][int(tokens[3])].append(self.registers[tokens[2]][-1])
                self.registers[tokens[2]] = self.registers[tokens[2]][-STACK_SIZE_LIMIT:]
                #print('Readlane value', self.registers[tokens[2]])
        except Exception as e:
            #print(e, 'Could not set:', line)
            pass


def try_match_swapped(insts, code, i, line):
    return insts[i+1][1] == code[line][1] and insts[i][1] == code[line+1][1]

def Match(inst_value, code_value):
    if code_value == inst_value:
        return True
    if code_value in [GETPC, SWAPPC, SETPC] and inst_value==SALU:
        return True
    if code_value == BRANCH and inst_value in [JUMP, NEXT]: # TODO: Maybe lets not reorder branches?
        return True
    return False

def get_match_lookahead(insts, code, i, line):
    if try_match_swapped(insts, code, i, line):
        return [i+1, i]
    new_inst_order = []

    allowed_insts = list(range(i, min(i+4, len(insts))))
    for l in range(line, min(line+10, len(code))):
        bMatch = False
        for j in allowed_insts:
            if Match(insts[j][1], code[l][1]):
                new_inst_order.append(j)
                allowed_insts.remove(j)
                bMatch = True
                break
        if bMatch == False:
            break
    if len(new_inst_order):
        new_inst_order += [j for j in list(range(i, max(new_inst_order)+1)) if j not in new_inst_order]
    return new_inst_order

def stitch(insts, raw_code, jumps):
    result, i, line, loopCount, N = [], 0, 0, defaultdict(int), len(insts)
    
    SMEM_INST = []
    VMEM_INST = []
    FLAT_INST = []
    NUM_SMEM = 0
    NUM_VMEM = 0
    NUM_FLAT = 0

    mem_unroll = []
    flight_count = []

    labels = {}
    jump_map = [0]
    code = [raw_code[0]]
    for c in raw_code[1:]:
        c = list(c)
        c[0] = c[0].split(';')[0].split('//')[0].strip()

        if c[1] != 100:
            code.append(c)
        elif ':' in c[0]:
            labels[c[0].split(':')[0]] = len(code)
        jump_map.append(len(code)-1)

    reverse_map = []
    for k, v in enumerate(jump_map):
        if v >= len(reverse_map):
            reverse_map.append(k)

    jumps = {jump_map[j]+1: j for j in jumps}

    smem_ordering = 0
    vmem_ordering = 0
    max_line = 0

    watchlist = RegisterWatchList(labels=labels)

    num_failed_stitches = 0
    MAX_FAILED_STITCHES = 128
    loops = 0
    maxline = 0

    while i < N:
        #print('L', line)
        loops += 1
        if line >= len(code) or loops > 100000 or num_failed_stitches >= MAX_FAILED_STITCHES:
            break

        maxline = max(reverse_map[line], maxline)
        inst = insts[i]

        as_line = code[line]
        max_line = max(max_line, reverse_map[line])

        matched = True
        next = line+1
        if as_line[1] == GETPC: # TODO: @ can put you ahead of label!
            watchlist.getpc(as_line[0], code[line+1][0])
            matched = inst[1] == SALU
        elif as_line[1] == LANEIO:
            watchlist.updatelane(as_line[0])
            matched = inst[1] == VALU
        elif as_line[1] == SETPC:
            next = watchlist.setpc(as_line[0])
            matched = inst[1] == SALU
        elif as_line[1] == SWAPPC:
            next = watchlist.swappc(as_line[0], line)
            #print('Next:', next, code[next])
            matched = inst[1] == SALU
        elif inst[1] == as_line[1]:
            if line in jumps:
                loopCount[jumps[line]-1] += 1  # label is the previous line
            num_inflight = NUM_FLAT + NUM_SMEM + NUM_VMEM

            if inst[1] == SMEM or inst[1] == LDS:
                smem_ordering = 1 if inst[1] == SMEM else smem_ordering
                SMEM_INST.append([reverse_map[line],  num_inflight])
                NUM_SMEM += 1
            elif inst[1] == VMEM or (inst[1] == FLAT and 'global_' in as_line[0]):
                VMEM_INST.append([reverse_map[line],  num_inflight])
                NUM_VMEM += 1
                if 'buffer_' in as_line[0]:
                    #watchlist.LDS_buffer_op(as_line[0])
                    vmem_ordering = 1
            elif inst[1] == FLAT:
                smem_ordering = 1
                vmem_ordering = 1
                FLAT_INST.append([reverse_map[line],  num_inflight])
                NUM_FLAT += 1
            elif inst[1] == IMMED and 'waitcnt' in as_line[0]:
                
                if 'lgkmcnt' in as_line[0]:
                    wait_N = int(as_line[0].split('lgkmcnt(')[1].split(')')[0])
                    flight_count.append([as_line[-1], num_inflight, wait_N])
                    if wait_N == 0:
                        smem_ordering = 0
                    if smem_ordering == 0:
                        offset = len(SMEM_INST)-wait_N
                        mem_unroll.append( [reverse_map[line], SMEM_INST[:offset]+FLAT_INST] )
                        SMEM_INST = SMEM_INST[offset:]
                        NUM_SMEM = len(SMEM_INST)
                        FLAT_INST = []
                        NUM_FLAT = 0
                    else:
                        NUM_SMEM = min(max(wait_N-NUM_FLAT, 0), NUM_SMEM)
                        NUM_FLAT = min(max(wait_N-NUM_SMEM, 0), NUM_FLAT)
                    num_inflight = NUM_FLAT + NUM_SMEM + NUM_VMEM

                if 'vmcnt' in as_line[0]:
                    wait_N = int(as_line[0].split('vmcnt(')[1].split(')')[0])
                    flight_count.append([as_line[-1], num_inflight, wait_N])
                    if wait_N == 0:
                        vmem_ordering = 0
                    if vmem_ordering == 0:
                        offset = len(VMEM_INST)-wait_N
                        mem_unroll.append( [reverse_map[line], VMEM_INST[:offset]+FLAT_INST] )
                        VMEM_INST = VMEM_INST[offset:]
                        NUM_VMEM = len(VMEM_INST)
                        FLAT_INST = []
                        NUM_FLAT = 0
                    else:
                        NUM_VMEM = min(max(wait_N-NUM_FLAT, 0), NUM_VMEM)
                        NUM_FLAT = min(max(wait_N-NUM_VMEM, 0), NUM_FLAT)

        elif inst[1] == JUMP and as_line[1] == BRANCH:
            next = jump_map[as_line[2]]
            if next is None or next == 0:
                print('Jump to unknown location!', as_line)
                break
        elif inst[1] == NEXT and as_line[1] == BRANCH:
            next = line + 1
        else:
            matched = False
            next = line + 1
            if i+1 < N and line+1 < len(code):
                if try_match_swapped(insts, code, i, line):
                    temp = insts[i]
                    insts[i] = insts[i+1]
                    insts[i+1] = temp
                    next = line
                elif 's_waitcnt' in as_line[0] or '_load_' in as_line[0]:
                    print(as_line)
                    break

        if matched:
            new_res = inst + (reverse_map[line],) # (line,)
            result.append(new_res)
            i += 1
            num_failed_stitches = 0
        else:
            num_failed_stitches += 1
        line = next

    N = max(N, 1)
    if len(result) != N:
        print('Warning - Stitching rate: '+str(len(result) * 100 / N)+'% matched')
        print('Leftovers:', [WaveInstCategory[insts[i+k][1]] for k in range(5) if i+k < len(insts)])
        try:
            print(line, code[line])
        except:
            pass
    else:
        while line < len(code):
            if 's_endpgm' in code[line]:
                mem_unroll.append( [reverse_map[line], SMEM_INST+VMEM_INST+FLAT_INST] )
                break
            line += 1

    return result, loopCount, mem_unroll, flight_count, maxline


def get_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(0)
    try:
        hostname = socket.gethostname()
        IPAddr = socket.gethostbyname(hostname)
        s.connect(({IPAddr}, 1))
    except Exception:
        IPAddr = '127.0.0.1'
    finally:
        return IPAddr


IPAddr = get_ip()
PORT, WebSocketPort = 8000, 18000
SP = '\u00A0'

def extract_tuple(content, num):
    vals = content.split(',')
    assert (len(vals) == num)
    last_val = vals[-1][:-1] if vals[-1].endswith(')') else vals[-1]
    vals = [vals[0][1:]] + vals[1:-1] + [last_val]
    return tuple(int(val) for val in vals)


def get_top_n(stitched):
    TOP_N = 10
    by_line_num = defaultdict(lambda: [0, 0, 0])
    for (_, _, s2i, run_time, line_num) in stitched:
        entry = by_line_num[line_num]
        entry[0] += 1
        entry[1] += s2i
        entry[2] += run_time
    top_n = sorted(
        [(line_num, v[0], v[1], v[2])
         for (line_num, v) in by_line_num.items()],
        key=lambda x: x[2] + x[3],
        reverse=True)
    return top_n[:TOP_N]


def wave_info(df, id):
    dic = {
        'Issue': df['issued_ins'][id],
        'Valu': df['valu_ins'][id], 'Valu_stall': df['valu_stalls'][id],
        'Salu': df['salu_ins'][id], 'Salu_stall': df['salu_stalls'][id],
        'Vmem': df['vmem_ins'][id], 'Vmem_stall': df['vmem_stalls'][id],
        'Smem': df['smem_ins'][id], 'Smem_stall': df['smem_stalls'][id],
        'Flat': df['flat_ins'][id], 'Flat_stall': df['flat_stalls'][id],
        'Lds': df['lds_ins'][id], 'Lds_stall': df['lds_stalls'][id],
        'Br': df['br_ins'][id], 'Br_stall': df['br_stalls'][id],
    }
    dic['Issue_stall'] = int(np.sum([dic[key] for key in dic.keys() if '_STALL' in key]))
    return dic


def extract_waves(waves):
    result, slot2seq = [], {}
    for id in waves['id']:
        row = {key: waves[key][id] for key in waves.keys()}

        insts, timeline = [], []
        for x in row['instructions'].split('),'):
            if len(x) > 0:
                insts.append(extract_tuple(x, 4))
        for x in row['timeline'].split('),'):
            if len(x) > 0:
                timeline.append(extract_tuple(x, 2))

        # aggregate per wave slot
        if (row['simd'], row['wave_slot']) in slot2seq:
            slot = result[slot2seq[(row['simd'], row['wave_slot'])]]
            last_end_time = slot[2][-1][-1]
            slot[2] += (row['id'], row['begin_time'], row['end_time']),
            slot[3] += insts
            # filler between waves
            slot[4] += (0, row['begin_time'] - last_end_time),
            slot[4] += timeline
        else:
            slot2seq[row['simd'], row['wave_slot']] = len(result)
            result.append([row['simd'], row['wave_slot'],
                           [(row['id'], row['begin_time'], row['end_time'])],
                           insts,
                           timeline])

    return result



def extract_data(df, se_number, code, jumps):
    if len(df['id']) == 0 or len(df['instructions']) == 0 or len(df['timeline']) == 0:
        return None

    cu_waves = extract_waves(df)
    all_filenames = []
    flight_count = []
    maxgrade = [{df['wave_slot'][wave_id]: -1 for wave_id in df['id']} for k in range(4)]
    non_stitched = [{df['wave_slot'][wave_id]: -1 for wave_id in df['id']} for k in range(4)]

    print('Number of waves:', len(df['id']))

    for wave_id in df['id']:
        if non_stitched[df['simd'][wave_id]][df['wave_slot'][wave_id]] == 0:
            continue
        insts, timeline = [], []
        if len(df['instructions'][wave_id]) == 0 or len(df['timeline'][wave_id]) == 0:
            continue

        for x in df['instructions'][wave_id].split('),'):
            insts.append(extract_tuple(x, 4))
        for x in df['timeline'][wave_id].split('),'):
            timeline.append(extract_tuple(x, 2))

        stitched, loopCount, mem_unroll, count, maxline = stitch(insts, code, jumps)
        srate = len(stitched)**2 / max(len(insts), 1)
        if srate <= maxgrade[df['simd'][wave_id]][df['wave_slot'][wave_id]]:
            continue

        maxgrade[df['simd'][wave_id]][df['wave_slot'][wave_id]] = srate
        non_stitched[df['simd'][wave_id]][df['wave_slot'][wave_id]] = len(insts) - len(stitched)
        flight_count.append(count)

        wave_entry = {
            "id": int(df['id'][wave_id]),
            "simd": int(df['simd'][wave_id]),
            "slot": int(df['wave_slot'][wave_id]),
            "begin": int(df['begin_time'][wave_id]),
            "end": int(df['end_time'][wave_id]),
            "info": wave_info(df, wave_id),
            "instructions": stitched,
            "timeline": timeline,
            "code": code[:maxline+16],
            "waitcnt": mem_unroll
        }
        data_obj = {
            "name": 'SE'.format(se_number),
            "kernel": code[0][0],
            "duration": sum(dur for (_, dur) in timeline),
            "wave": wave_entry,
            "simd_waves": [],
            "cu_waves": cu_waves,
            "loop_count": loopCount,
            "top_n": get_top_n(stitched),
            "websocket_port": WebSocketPort,
            "generation_time": time.ctime()
        }
        if len(data_obj["cu_waves"]) == 0:
            continue

        OUT = 'se'+str(se_number)+'_sm'+str(df['simd'][wave_id])+'_wv'+str(df['wave_slot'][wave_id])+'.json'
        JSON_GLOBAL_DICTIONARY[OUT] = Readable(data_obj)
        all_filenames.append(OUT)

    return flight_count, all_filenames


class NoCacheHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_my_headers()
        http.server.SimpleHTTPRequestHandler.end_headers(self)

    def send_my_headers(self):
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")

    def do_GET(self):
        global PICTURE_CALLBACK
        if 'timeline.png?' in self.path:
            selections = [int(s)!=0 for s in self.path.split('timeline.png?')[1]]
            counters_json, imagebytes = PICTURE_CALLBACK(selections[1:], selections[0])
            JSON_GLOBAL_DICTIONARY['counters.json'] = counters_json
            JSON_GLOBAL_DICTIONARY[self.path.split('/')[-1]] = imagebytes
        if '.json' in self.path or 'timeline.png' in self.path:
            try:
                response_file = JSON_GLOBAL_DICTIONARY[self.path.split('/')[-1]]
                #print(response_file)
            except:
                print('Invalid json request:', self.path)
                self.send_error(HTTPStatus.NOT_FOUND, "File not found")
                #print(JSON_GLOBAL_DICTIONARY.keys())
                return
            self.send_response(HTTPStatus.OK)
            if 'timeline.png' in self.path:
                self.send_header("Content-type", 'image/png')
            else:
                self.send_header("Content-type", 'application/json')
            self.send_header("Content-Length", str(len(response_file)))
            self.send_header("Last-Modified", self.date_time_string(time.time()))
            self.end_headers()
            self.copyfile(response_file, self.wfile)
        elif self.path in ['/', '/styles.css', '/index.html', '/logo.svg']:
            http.server.SimpleHTTPRequestHandler.do_GET(self)
        else:
            print('Invalid request:', self.path)
            self.send_error(HTTPStatus.NOT_FOUND, "File not found")


class RocTCPServer(socketserver.TCPServer):
    def server_bind(self):
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind(self.server_address)


def run_server():
    Handler = NoCacheHTTPRequestHandler
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)),'ui'))
    try:
        with RocTCPServer((IPAddr, PORT), Handler) as httpd:
            httpd.serve_forever()
    except KeyboardInterrupt:
        pass


def fix_space(line):
    line = line.replace(' ', SP)
    line = line.replace('\t', SP*4)
    return line


def WebSocketserver(websocket, path):
    data = websocket.recv()
    print(354, data)
    cpp, ln, _ = data.split(':')
    ln = int(ln)
    HL, EMP = 'highlight', ''
    content = None
    print("loading...")
    try:
        f = open(cpp, 'r', errors='replace')
        content = ''.join('<li class=\"line_'+str(i)+
                str(HL if i==ln else EMP)+'">'+str(i).ljust(5)+fix_space(l)+'</li>'
                          for i, l in enumerate(f.readlines(), 1))
    except FileNotFoundError:
        content = cpp + ' not found!'
    websocket.send(content)


def run_websocket():
    start_server = websockets.serve(WebSocketserver, IPAddr, WebSocketPort)
    try:
        asyncio.get_event_loop().run_until_complete(start_server)
        asyncio.get_event_loop().run_forever()
    except KeyboardInterrupt:
        pass


def assign_ports(ports):
    ps = [int(port) for port in ports.split(',')]
    if ps[0] <= 5000 or ps[1] <= 5000:
        print('Need to have port values > 5000')
        sys.exit(1)
    elif ps[0] == ps[1]:
        print('Can not use the same port for both web server and websocket server: '+ps[0])
        sys.exit(1)
    global IPAddr, PORT, WebSocketPort
    PORT, WebSocketPort = ps[0], ps[1]


def call_picture_callback(return_dict):
    global PICTURE_CALLBACK
    response, imagebytes = PICTURE_CALLBACK()
    return_dict[0] = response
    return_dict[1] = imagebytes


def view_trace(args, wait, code, jumps, dbnames, att_filenames, bReturnLoc, pic_callback):
    global PICTURE_CALLBACK
    PICTURE_CALLBACK = pic_callback
    manager = Manager()
    return_dict = manager.dict()

    pic_thread = Process(target=call_picture_callback, args=(return_dict,))
    pic_thread.start()

    assert(len(dbnames) > 0)
    att_filenames = [Path(f).name for f in att_filenames]
    se_numbers = [int(a.split('_se')[1].split('.att')[0]) for a in att_filenames]
    flight_count = []
    simd_wave_filenames = {}

    for se_number, dbname in zip(se_numbers, dbnames):
        if len(dbname['id']) == 0:
            continue

        count, wv_filenames = extract_data(dbname, se_number, code, jumps)

        if count is not None:
            flight_count.append(count)
            simd_wave_filenames[se_number] = wv_filenames

    if bReturnLoc:
        return flight_count

    for key in simd_wave_filenames.keys():
        wv_array = [[
            int(s.split('_sm')[1].split('_wv')[0]),
            int(s.split('_wv')[1][0]),
            s
        ] for s in simd_wave_filenames[key]]

        wv_dict = {}
        for wv in wv_array:
            try:
                wv_dict[wv[0]][wv[1]] = wv[2]
            except:
                try:
                    wv_dict[wv[0]] = {wv[1]: wv[2]}
                except:
                    exit(-1)

        simd_wave_filenames[key] = wv_dict

    JSON_GLOBAL_DICTIONARY['filenames.json'] = Readable({"filenames": simd_wave_filenames})

    if args.ports:
        assign_ports(args.ports)
    print('serving at ports: {0},{1}'.format(PORT, WebSocketPort))

    if wait == 0:
        try:
            PROCS = [Process(target=run_server), Process(target=run_websocket)]
            if pic_thread is not None:
                pic_thread.join()
                JSON_GLOBAL_DICTIONARY['counters.json'] = return_dict[0]
                JSON_GLOBAL_DICTIONARY['timeline.png'] = return_dict[1]

            for p in PROCS:
                p.start()
            for p in PROCS:
                p.join()
        except KeyboardInterrupt:
            print("Exitting.")
