#!/usr/bin/env python3
import sys

if sys.version_info[0] < 3:
    raise Exception("Must be using Python 3")

import os
import sys
import time
import socket
from pathlib import Path
from collections import defaultdict
import http.server
import socketserver
import socket
import asyncio
import websockets
from multiprocessing import Process, Manager
import numpy as np
from http import HTTPStatus
from io import BytesIO
from drawing import Readable, GeneratePIC
from copy import deepcopy

JSON_GLOBAL_DICTIONARY = {}


def get_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(0)
    try:
        hostname = socket.gethostname()
        IPAddr = socket.gethostbyname(hostname)
        s.connect(({IPAddr}, 1))
    except Exception:
        IPAddr = "127.0.0.1"
    finally:
        return IPAddr


IPAddr = get_ip()
PORT, WebSocketPort = 8000, 18000
SP = "\u00A0"


def get_top_n(code):
    TOP_N = 10
    top_n = sorted(deepcopy(code), key=lambda x: x[-1], reverse=True)[:TOP_N]
    return [
        (line_num, hitc, 0, run_time) for _, _, _, _, line_num, _, hitc, run_time in top_n
    ]


def wave_info(df, id):
    dic = {
        "Issue": df["issued_ins"][id],
        "Valu": df["valu_ins"][id],
        "Valu_stall": df["valu_stalls"][id],
        "Salu": df["salu_ins"][id],
        "Salu_stall": df["salu_stalls"][id],
        "Vmem": df["vmem_ins"][id],
        "Vmem_stall": df["vmem_stalls"][id],
        "Smem": df["smem_ins"][id],
        "Smem_stall": df["smem_stalls"][id],
        "Flat": df["flat_ins"][id],
        "Flat_stall": df["flat_stalls"][id],
        "Lds": df["lds_ins"][id],
        "Lds_stall": df["lds_stalls"][id],
        "Br": df["br_ins"][id],
        "Br_stall": df["br_stalls"][id],
    }
    dic["Issue_stall"] = int(np.sum([dic[key] for key in dic.keys() if "_STALL" in key]))
    return dic


def extract_data(df, se_number):
    if len(df["id"]) == 0 or len(df["instructions"]) == 0 or len(df["timeline"]) == 0:
        return None

    wave_filenames = []
    flight_count = []
    wave_slot_count = [
        {df["wave_slot"][wave_id]: 0 for wave_id in df["id"]} for k in range(4)
    ]

    print("Number of waves:", len(df["id"]))
    allwaves_maxline = 0

    for wave_id in df["id"]:
        stitched, loopCount, mem_unroll, count, maxline, num_insts = df["instructions"][
            wave_id
        ]
        timeline = df["timeline"][wave_id]

        if len(stitched) == 0 or len(timeline) == 0 or len(stitched) != num_insts:
            continue

        allwaves_maxline = max(allwaves_maxline, maxline)
        flight_count.append(count)

        wave_entry = {
            "id": int(df["id"][wave_id]),
            "simd": int(df["simd"][wave_id]),
            "slot": int(df["wave_slot"][wave_id]),
            "begin": int(df["begin_time"][wave_id]),
            "end": int(df["end_time"][wave_id]),
            "info": wave_info(df, wave_id),
            "instructions": stitched,
            "timeline": timeline,
            "waitcnt": mem_unroll,
        }
        data_obj = {
            "name": "SE".format(se_number),
            "duration": sum(dur for (_, dur) in timeline),
            "wave": wave_entry,
            "loop_count": loopCount,
            "top_n": [],
            "num_stitched": len(stitched),
            "num_insts": num_insts,
            "websocket_port": WebSocketPort,
            "generation_time": time.ctime(),
        }

        simd_id = df["simd"][wave_id]
        slot_id = df["wave_slot"][wave_id]
        slot_count = wave_slot_count[simd_id][slot_id]
        wave_slot_count[simd_id][slot_id] += 1

        OUT = (
            "se"
            + str(se_number)
            + "_sm"
            + str(simd_id)
            + "_sl"
            + str(slot_id)
            + "_wv"
            + str(slot_count)
            + ".json"
        )
        JSON_GLOBAL_DICTIONARY[OUT] = Readable(data_obj)
        wave_filenames.append((OUT, df["begin_time"][wave_id], df["end_time"][wave_id]))

    data_obj = {
        "name": "SE".format(se_number),
        "websocket_port": WebSocketPort,
        "generation_time": time.ctime(),
    }
    se_filename = None
    if len(wave_filenames) > 0:
        se_filename = "se" + str(se_number) + "_info.json"
        JSON_GLOBAL_DICTIONARY[se_filename] = Readable(data_obj)

    return flight_count, wave_filenames, se_filename, allwaves_maxline


class NoCacheHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_my_headers()
        http.server.SimpleHTTPRequestHandler.end_headers(self)

    def send_my_headers(self):
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")

    def do_GET(self):
        if (
            ".png?" in self.path
            and self.path.split("/")[-1] not in JSON_GLOBAL_DICTIONARY.keys()
        ):
            selections = [int(s) != 0 for s in self.path.split(".png?")[-1]]
            counters_json, imagebytes = GeneratePIC(
                self.drawinfo, selections[1:], selections[0]
            )
            JSON_GLOBAL_DICTIONARY["graph_options.json"] = counters_json
            JSON_GLOBAL_DICTIONARY[self.path.split("/")[-1]] = imagebytes[
                self.path.split("/")[-1].split("?")[0]
            ]

        if ".json" in self.path or ".png" in self.path:
            try:
                response_file = JSON_GLOBAL_DICTIONARY[self.path.split("/")[-1]]
            except:
                print("Invalid json request:", self.path)
                print(JSON_GLOBAL_DICTIONARY.keys())
                self.send_error(HTTPStatus.NOT_FOUND, "File not found")
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Length", str(len(response_file)))
            if ".b" in self.path:
                self.send_header("Content-type", "application/octet-stream")
                response_file = BytesIO(response_file)
            elif "timeline.png" in self.path:
                self.send_header("Content-type", "image/png")
            else:
                self.send_header("Content-type", "application/json")
            self.send_header("Last-Modified", self.date_time_string(time.time()))
            self.end_headers()
            self.copyfile(response_file, self.wfile)
        elif self.path in ["/", "/styles.css", "/index.html", "/logo.svg"]:
            http.server.SimpleHTTPRequestHandler.do_GET(self)
        else:
            print("Invalid request:", self.path)
            self.send_error(HTTPStatus.NOT_FOUND, "File not found")


class RocTCPServer(socketserver.TCPServer):
    def server_bind(self):
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind(self.server_address)


def run_server(drawinfo, trace_instance_name):
    Handler = NoCacheHTTPRequestHandler
    Handler.drawinfo = drawinfo
    os.chdir(trace_instance_name+"_ui/")
    try:
        with RocTCPServer((IPAddr, PORT), Handler) as httpd:
            httpd.serve_forever()
    except KeyboardInterrupt:
        pass


def fix_space(line):
    line = line.replace(" ", SP)
    line = line.replace("\t", SP * 4)
    return line


def WebSocketserver(websocket, path):
    data = websocket.recv()
    cpp, ln, _ = data.split(":")
    ln = int(ln)
    HL, EMP = "highlight", ""
    content = None
    print("loading...")
    try:
        f = open(cpp, "r", errors="replace")
        content = "".join(
            '<li class="line_'
            + str(i)
            + str(HL if i == ln else EMP)
            + '">'
            + str(i).ljust(5)
            + fix_space(l)
            + "</li>"
            for i, l in enumerate(f.readlines(), 1)
        )
    except FileNotFoundError:
        content = cpp + " not found!"
    websocket.send(content)


def run_websocket():
    start_server = websockets.serve(WebSocketserver, IPAddr, WebSocketPort)
    try:
        asyncio.get_event_loop().run_until_complete(start_server)
        asyncio.get_event_loop().run_forever()
    except KeyboardInterrupt:
        pass


def assign_ports(ports):
    ps = [int(port) for port in ports.split(",")]
    if ps[0] <= 5000 or ps[1] <= 5000:
        print("Need to have port values > 5000")
        sys.exit(1)
    elif ps[0] == ps[1]:
        print(
            "Can not use the same port for both web server and websocket server: " + ps[0]
        )
        sys.exit(1)
    global IPAddr, PORT, WebSocketPort
    PORT, WebSocketPort = ps[0], ps[1]


def call_picture_callback(return_dict, drawinfo):
    response, imagebytes = GeneratePIC(drawinfo)
    return_dict["graph_options.json"] = response
    for k, v in imagebytes.items():
        return_dict[k] = v

    for n, m in enumerate(drawinfo["TIMELINES"]):
        return_dict["wstates" + str(n) + ".json"] = Readable(
            {"data": [int(n) for n in list(np.asarray(m))]}
        )
    for n, e in enumerate(drawinfo["EVENTS"]):
        return_dict["se" + str(n) + "_perfcounter.json"] = Readable(
            {"data": [v.toTuple() for v in e]}
        )


def view_trace(
    args,
    code,
    dbnames,
    att_filenames,
    bDumpOnly,
    se_time_begin,
    gfxv,
    drawinfo,
    trace_instance_name
):
    global JSON_GLOBAL_DICTIONARY
    pic_thread = None

    manager = Manager()
    return_dict = manager.dict()
    occ_dict = {str(k): drawinfo["OCCUPANCY"][k] for k in range(len(drawinfo["OCCUPANCY"]))}
    occ_dict['dispatches'] = {}
    for id, name in drawinfo['DispatchNames'].items():
        occ_dict['dispatches'][id] = name
    occ_dict['names'] = drawinfo['ShaderNames']

    JSON_GLOBAL_DICTIONARY["occupancy.json"] = Readable(occ_dict)
    pic_thread = Process(target=call_picture_callback, args=(return_dict, drawinfo))
    pic_thread.start()

    att_filenames = [Path(f).name for f in att_filenames]
    se_numbers = [int(a.split("_se")[1].split(".att")[0]) for a in att_filenames]
    flight_count = []
    simd_wave_filenames = {}
    se_filenames = []

    allse_maxline = 0
    for se_number, dbname in zip(se_numbers, dbnames):
        if len(dbname["id"]) == 0:
            continue

        count, wv_filenames, se_filename, maxline = extract_data(dbname, se_number)
        if se_filename is None:
            continue
        allse_maxline = max(allse_maxline, maxline)
        se_filenames.append(se_filename)

        if count is not None:
            flight_count.append(count)
            simd_wave_filenames[se_number] = wv_filenames

    code_sel = [c[:-3]+c[-2:] for c in code[:allse_maxline+16]]
    JSON_GLOBAL_DICTIONARY['code.json'] = Readable({"code": code_sel, "top_n": get_top_n(code_sel)})

    for key in simd_wave_filenames.keys():
        wv_array = [
            [
                int(s[0].split("_sm")[1].split("_sl")[0]),
                int(s[0].split("_sl")[1].split("_wv")[0]),
                int(s[0].split("_wv")[1].split(".")[0]),
                s,
            ]
            for s in simd_wave_filenames[key]
        ]

        wv_dict = {}
        for wv in wv_array:
            try:
                wv_dict[wv[0]][wv[1]][wv[2]] = wv[3]
            except:
                try:
                    wv_dict[wv[0]][wv[1]] = {wv[2]: wv[3]}
                except:
                    try:
                        wv_dict[wv[0]] = {wv[1]: {wv[2]: wv[3]}}
                    except:
                        pass

        simd_wave_filenames[key] = wv_dict

    JSON_GLOBAL_DICTIONARY["filenames.json"] = Readable(
        {
            "wave_filenames": simd_wave_filenames,
            "se_filenames": se_filenames,
            "global_begin_time": int(se_time_begin),
            "gfxv": gfxv,
        }
    )

    if pic_thread is not None:
        pic_thread.join()
        for k, v in return_dict.items():
            JSON_GLOBAL_DICTIONARY[k] = v

    if bDumpOnly == False:
        JSON_GLOBAL_DICTIONARY["live.json"] = Readable({"live": 1})
        if args.ports:
            assign_ports(args.ports)
        print("serving at ports: {0},{1}".format(PORT, WebSocketPort))
        try:
            PROCS = [
                Process(target=run_server, args=[drawinfo, trace_instance_name]),
                Process(target=run_websocket),
            ]
            for p in PROCS:
                p.start()
            for p in PROCS:
                p.join()
        except KeyboardInterrupt:
            print("Exitting.")
    else:
        os.makedirs(trace_instance_name + "_ui/", exist_ok=True)
        JSON_GLOBAL_DICTIONARY["live.json"] = Readable({"live": 0})
        os.system(
            "cp "
            + os.path.join(os.path.abspath(os.path.dirname(__file__)), "ui")
            + "/* " + trace_instance_name + "_ui/"
        )
        for k, v in JSON_GLOBAL_DICTIONARY.items():
            with open(os.path.join(trace_instance_name+"_ui", k), "w" if ".json" in k else "wb") as f:
                f.write(v.read())
