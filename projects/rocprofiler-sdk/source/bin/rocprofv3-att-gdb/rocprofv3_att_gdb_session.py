# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Local session transport. No debugger or GPU operations run in this thread."""

import json
import os
import queue
import selectors
import socket
import struct


class Session:
    max_frame = 65536

    def __init__(self, path, session, deliver):
        self.session = session
        self.deliver = deliver
        self.jobs = queue.SimpleQueue()
        self.peers = {}
        self.next_peer = 0
        self.closed = False
        self.selector = selectors.DefaultSelector()
        self.listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.listener.bind(path)
        os.chmod(path, 0o600)
        self.listener.listen(128)
        self.listener.setblocking(False)
        self.reader, self.writer = socket.socketpair()
        self.reader.setblocking(False)
        self.writer.setblocking(False)
        self.selector.register(self.listener, selectors.EVENT_READ)
        self.selector.register(self.reader, selectors.EVENT_READ)

    def submit(self, operation, *args):
        if self.closed:
            return
        self.jobs.put((operation, args))
        try:
            self.writer.send(b"x")
        except BlockingIOError:
            pass  # An unread wakeup already exists.
        except OSError:
            if not self.closed:
                raise

    def send(self, peer, message):
        self.submit("send", peer, message)

    def snapshot(self, tag):
        self.submit("snapshot", tag)

    def close(self):
        self.submit("close")

    def accept(self):
        while True:
            try:
                connection, _ = self.listener.accept()
            except BlockingIOError:
                return
            pid, uid, _ = struct.unpack(
                "3i", connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12)
            )
            if uid != os.getuid() or len(self.peers) >= 1024:
                connection.close()
                continue
            connection.setblocking(False)
            self.next_peer += 1
            peer = dict(
                id=self.next_peer,
                socket=connection,
                pid=pid,
                instance=None,
                input=b"",
                output=b"",
            )
            self.peers[peer["id"]] = peer
            self.selector.register(connection, selectors.EVENT_READ, peer)

    def disconnect(self, peer, reason):
        if self.peers.pop(peer["id"], None) is None:
            return
        self.selector.unregister(peer["socket"])
        peer["socket"].close()
        self.deliver(peer["id"], {"op": "DISCONNECTED", "message": reason})

    def read(self, peer):
        try:
            while True:
                try:
                    data = peer["socket"].recv(4096)
                except BlockingIOError:
                    return
                if not data:
                    self.disconnect(peer, "participant disconnected")
                    return
                peer["input"] += data
                while b"\n" in peer["input"]:
                    frame, peer["input"] = peer["input"].split(b"\n", 1)
                    if len(frame) >= self.max_frame:
                        raise ValueError("oversized control frame")
                    message = json.loads(frame)
                    if (
                        not isinstance(message, dict)
                        or message.get("session") != self.session
                    ):
                        raise ValueError("wrong session")
                    if peer["instance"] is None:
                        if (
                            message.get("op") != "REGISTER"
                            or message.get("version") != 3
                            or message.get("pid") != peer["pid"]
                            or not isinstance(message.get("instance"), str)
                            or not message["instance"]
                        ):
                            raise ValueError("invalid participant registration")
                        peer["instance"] = message["instance"]
                    elif message.get("instance") != peer["instance"]:
                        raise ValueError("wrong worker generation")
                    self.deliver(peer["id"], message)
                if len(peer["input"]) >= self.max_frame:
                    raise ValueError("unterminated control frame")
        except (OSError, ValueError, TypeError) as error:
            self.disconnect(peer, str(error))

    def flush(self, peer):
        try:
            if peer["output"]:
                count = peer["socket"].send(peer["output"])
                peer["output"] = peer["output"][count:]
            events = selectors.EVENT_READ
            if peer["output"]:
                events |= selectors.EVENT_WRITE
            self.selector.modify(peer["socket"], events, peer)
        except BlockingIOError:
            self.selector.modify(
                peer["socket"], selectors.EVENT_READ | selectors.EVENT_WRITE, peer
            )
        except OSError as error:
            self.disconnect(peer, str(error))

    def run(self):
        try:
            while not self.closed:
                for key, events in self.selector.select():
                    if key.fileobj is self.listener:
                        self.accept()
                    elif key.fileobj is self.reader:
                        while True:
                            try:
                                if not self.reader.recv(4096):
                                    break
                            except BlockingIOError:
                                break
                    else:
                        peer = key.data
                        if events & selectors.EVENT_READ:
                            self.read(peer)
                        if events & selectors.EVENT_WRITE and peer["id"] in self.peers:
                            self.flush(peer)
                while not self.jobs.empty():
                    operation, args = self.jobs.get()
                    if operation == "close":
                        self.closed = True
                        break
                    if operation == "snapshot":
                        # Registration is sent before participant initialization returns.
                        # Drain it before posting the membership snapshot to the debugger.
                        self.accept()
                        for peer in list(self.peers.values()):
                            self.read(peer)
                        self.deliver(None, {"op": "SNAPSHOT", "tag": args[0]})
                    elif operation == "send":
                        peer = self.peers.get(args[0])
                        if peer is None:
                            continue
                        message = dict(
                            args[1], session=self.session, instance=peer["instance"]
                        )
                        peer["output"] += (json.dumps(message) + "\n").encode()
                        if len(peer["output"]) > self.max_frame:
                            self.disconnect(
                                peer, "participant is not reading control commands"
                            )
                        else:
                            self.flush(peer)
        finally:
            self.closed = True
            for peer in list(self.peers.values()):
                self.disconnect(peer, "session closed")
            self.listener.close()
            self.reader.close()
            self.writer.close()
            self.selector.close()
