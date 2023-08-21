#!/usr/bin/env python3
import sys

if sys.version_info[0] < 3:
    raise Exception("Must be using Python 3")

import http.server
import socketserver
import socket
import os
import sys


class NoCacheHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_my_headers()
        http.server.SimpleHTTPRequestHandler.end_headers(self)

    def send_my_headers(self):
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")

    def do_GET(self):
        if ".png?" in self.path:
            self.path = self.path.split(".png?")[0] + ".png"

        http.server.SimpleHTTPRequestHandler.do_GET(self)


class RocTCPServer(socketserver.TCPServer):
    def server_bind(self):
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind(self.server_address)


def run_server():
    Handler = NoCacheHTTPRequestHandler
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "."))
    try:
        with RocTCPServer((IPAddr, PORT), Handler) as httpd:
            httpd.serve_forever()
    except KeyboardInterrupt:
        pass


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
PORT = 8000

if len(sys.argv) > 1:
    PORT = int(sys.argv[1])
print("serving at port: {0}".format(PORT))

try:
    run_server()
except KeyboardInterrupt:
    print("Exitting.")
