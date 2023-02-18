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
#import webbrowser
import http.server
import socketserver
import socket
import asyncio
import websockets
from multiprocessing import *
from copy import deepcopy

PORT, WebSocketPort = 8000, 18000
SP = '\u00A0'

RS_TRACE_DEBUG = "RS_TRACE_DEBUG" in os.environ
if RS_TRACE_DEBUG:
    LOG = open('./att_viewer.log', 'w')


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


def debug_log(msg, last=False):
    if RS_TRACE_DEBUG:
        LOG.write(msg)

        if last:
            LOG.close()


def try_match_swapped(insts, code, i, line):
    return insts[i+1][1] == code[line][1] and insts[i][1] == code[line+1][1]


def stitch(insts, code, jumps):
    result, i, line, loopCount, N = [], 0, 0, defaultdict(int), len(insts)
    
    SMEM_INST = []
    VMEM_INST = []
    FLAT_INST = []
    NUM_SMEM = 0
    NUM_VMEM = 0
    NUM_FLAT = 0

    mem_unroll = []
    flight_count = []

    ordering = 0 # 0 for in order, 1 for ongoing flats and 2 for SMEM
    while i < N:
        inst = insts[i]
        if line >= len(code):
            break

        as_line = code[line]
        if inst[1] == as_line[1]:
            if line in jumps:
                loopCount[line-1] += 1  # label is the previous line
            matched, next = True, line + 1

            num_inflight = NUM_FLAT + NUM_SMEM + NUM_VMEM

            if inst[1] == 1 or inst[1] == 5: # SMEM, LDS
                ordering = 2 if inst[1] == 1 else ordering
                SMEM_INST.append([line,  num_inflight])
                NUM_SMEM += 1
            elif inst[1] == 3 or (inst[1] == 4 and 'global_' in as_line[0]): # VMEM R/W
                VMEM_INST.append([line,  num_inflight])
                NUM_VMEM += 1
            elif inst[1] == 4: # FLAT
                ordering = max(ordering, 1)
                FLAT_INST.append([line,  num_inflight])
                NUM_FLAT += 1
            elif inst[1] == 9 and 'waitcnt' in as_line[0]:
                
                if 'lgkmcnt' in as_line[0]:
                    wait_N = int(as_line[0].split('lgkmcnt(')[1].split(')')[0])
                    flight_count.append([as_line[-1], num_inflight, wait_N])
                    if wait_N == 0:
                        ordering = 0
                    if ordering == 0:
                        offset = len(SMEM_INST)-wait_N
                        mem_unroll.append( [line, SMEM_INST[:offset]+FLAT_INST] )
                        SMEM_INST = SMEM_INST[offset:]
                        FLAT_INST = []
                        NUM_FLAT = 0
                        NUM_SMEM = 0
                    else:
                        NUM_SMEM = min(max(wait_N-NUM_FLAT, 0), NUM_SMEM)
                        NUM_FLAT = min(max(wait_N-NUM_SMEM, 0), NUM_FLAT)

                if 'vmcnt' in as_line[0]:
                    wait_N = int(as_line[0].split('vmcnt(')[1].split(')')[0])
                    flight_count.append([as_line[-1], num_inflight, wait_N])
                    if wait_N == 0 and ordering != 2:
                        ordering = 0
                    if ordering == 0:
                        offset = len(VMEM_INST)-wait_N
                        mem_unroll.append( [line, VMEM_INST[:offset]+FLAT_INST] )
                        VMEM_INST = VMEM_INST[offset:]
                        FLAT_INST = []
                        NUM_FLAT = 0
                        NUM_VMEM = 0
                    else:
                        NUM_VMEM = min(max(wait_N-NUM_FLAT, 0), NUM_VMEM)
                        NUM_FLAT = min(max(wait_N-NUM_VMEM, 0), NUM_FLAT)


        elif inst[1] == 7 and as_line[1] == 10:  # jump
            matched, next = True, as_line[2]
        elif inst[1] == 8 and as_line[1] == 10:  # next
            matched, next = True, line + 1
        else:
            # instructions with almost same timestamp swapped
            # if i+1 < N and line+1 < len(code) and inst[0] == insts[i+1][0]:
            matched = False
            next = line + 1
            if i+1 < N and line+1 < len(code):
                if try_match_swapped(insts, code, i, line):
                    #print('Swap:', code[line])
                    #print('For:', code[line+1])
                    #result += (*(insts[i+1]), line),
                    #result += (*inst, line+1),
                    #i, next = i+2, line+2
                    temp = insts[i]
                    insts[i] = insts[i+1]
                    insts[i+1] = temp
                    next = line
                #else:
                #    print('Could not parse tokens:', insts[i], as_line)

        if matched:
            new_res = inst + (line,)
            result.append(new_res)
            i += 1
        line = next

    N = max(N, 1)
    if len(result) != N:
        print('Warning - Stitching rate: '+str(len(result) * 100 / N)+'% matched')

    return result, loopCount, mem_unroll, flight_count


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


def rjust_html(s, n):
    s = str(s)
    return SP * (n-len(s)) + s if len(s) < n else s


def rjust_html_format(msg, n1, inst, n2, n3, stall):
    return str(rjust_html(msg,n1)) + str(rjust_html(inst,n2)) + str(SP*n3) + str(stall)


def wave_info(df, id):
    issued_ins, mem_ins = df['issued_ins'][id], df['mem_ins'][id]
    valu_ins, valu_stalls = df['valu_ins'][id], df['valu_stalls'][id]
    salu_ins, salu_stalls = df['salu_ins'][id], df['salu_stalls'][id]
    vmem_ins, vmem_stalls = df['vmem_ins'][id], df['vmem_stalls'][id]
    smem_ins, smem_stalls = df['smem_ins'][id], df['smem_stalls'][id]
    flat_ins, flat_stalls = df['flat_ins'][id], df['flat_stalls'][id]
    lds_ins, lds_stalls = df['lds_ins'][id], df['lds_stalls'][id]
    br_ins, br_stalls = df['br_ins'][id], df['br_stalls'][id]

    return 'Issued:' + str(rjust_html(issued_ins,8)) + str(SP*2) + 'Mem:' + str(mem_ins) \
    + "-" * 26 + rjust_html_format("VALU:",6,valu_ins,8,4,valu_stalls) \
    + rjust_html_format("SALU:",6,salu_ins,8,4,salu_stalls) \
    + rjust_html_format("VMEM:",6,vmem_ins,8,4,vmem_stalls) \
    + rjust_html_format("SMEM:",6,smem_ins,8,4,smem_stalls) \
    + rjust_html_format("FLAT:",6,flat_ins,8,4,flat_stalls) \
    + rjust_html_format("LDS:",6,lds_ins,8,4,lds_stalls) \
    + rjust_html_format("BR:",6,br_ins,8,4,br_stalls)


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


def extract_data(df, output_ui, se_number, code, jumps):    
    if len(df['id']) == 0 or len(df['instructions']) == 0 or len(df['timeline']) == 0:
        return None

    cu_waves = extract_waves(df)
    all_filenames = []
    flight_count = []

    for wave_id in df['id']:
        insts, timeline = [], []
        if len(df['instructions'][wave_id]) == 0 or len(df['timeline'][wave_id]) == 0:
            continue

        for x in df['instructions'][wave_id].split('),'):
            insts.append(extract_tuple(x, 4))
        for x in df['timeline'][wave_id].split('),'):
            timeline.append(extract_tuple(x, 2))

        stitched, loopCount, mem_unroll, count = stitch(insts, code, jumps)
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
            "code": code,
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

        OUT = output_ui+'/ui/se'+str(se_number)+'_sm'+str(df['simd'][wave_id])+\
                '_wv'+str(df['wave_slot'][wave_id])+'.json'

        with open(OUT, 'w') as f:
            f.write(json.dumps(data_obj))
        all_filenames.append(OUT.split('/')[-1])

    return flight_count, all_filenames

#def open_browser():
#    time.sleep(0.1)
#    webbrowser.open_new_tab('http://{0}:{1}'.format(IPAddr, PORT))


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
            PICTURE_CALLBACK(selections[1:], selections[0])
            #PICTURE_CALLBACK(selections[2:], selections[1], selections[0])
        http.server.SimpleHTTPRequestHandler.do_GET(self)

class RocTCPServer(socketserver.TCPServer):
    def server_bind(self):
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind(self.server_address)


def run_server():
    global RS_HOME
    Handler = NoCacheHTTPRequestHandler

    os.chdir(RS_HOME+'/ui')
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


def view_trace(args, wait, code, jumps, dbnames, att_filenames, bReturnLoc, pic_callback):
    global PICTURE_CALLBACK
    PICTURE_CALLBACK = pic_callback
    pic_thread = Process(target=pic_callback)
    pic_thread.start()

    assert(len(dbnames) > 0)
    global RS_HOME
    output_ui = args.output_ui
    RS_HOME = output_ui

    att_filenames = [Path(f).name for f in att_filenames]
    se_numbers = [int(a.split('_se')[1].split('.att')[0]) for a in att_filenames]
    flight_count = []
    simd_wave_filenames = {}

    for se_number, dbname in zip(se_numbers, dbnames):
        if len(dbname['id']) == 0:
            continue

        count, wv_filenames = extract_data(dbname, output_ui, se_number, code, jumps)

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

    with open(output_ui+'/ui/filenames.json', 'w') as f:
        f.write(json.dumps({"filenames": simd_wave_filenames}))

    if args.ports:
        assign_ports(args.ports)
    print('serving at ports: {0},{1}'.format(PORT, WebSocketPort))

    if wait == 0:
        try:
            PROCS = [Process(target=run_server),
                     #Process(target=open_browser),
                     Process(target=run_websocket)]
            if pic_thread is not None:
                pic_thread.join()

            for p in PROCS:
                p.start()
            for p in PROCS:
                p.join()
        except KeyboardInterrupt:
            print("Exitting.")
