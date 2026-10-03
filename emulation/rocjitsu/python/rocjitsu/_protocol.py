# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Private, inherited-socket protocol. Both endpoints execute trusted user code."""

import io
import pickle
import struct
import sys
import time

import cloudpickle

PROTOCOL_VERSION = (1, cloudpickle.__version__)


class _Pickler(cloudpickle.CloudPickler):
    def reducer_override(self, obj):
        torch = sys.modules.get("torch")
        if torch is not None and isinstance(obj, torch.Tensor):
            if obj.device.type != "cpu":
                raise TypeError(
                    "Pass CPU tensors; GPU tensors must stay inside the session"
                )
            if obj.requires_grad:
                raise TypeError(
                    "Detach CPU tensors before crossing the session boundary"
                )
        return super().reducer_override(obj)


def dumps(value):
    stream = io.BytesIO()
    _Pickler(stream, protocol=pickle.HIGHEST_PROTOCOL).dump(value)
    return stream.getvalue()


def _timeout(sock, deadline):
    if deadline is None:
        sock.settimeout(None)
    else:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("rocjitsu worker timed out")
        sock.settimeout(remaining)


def send(sock, payload, deadline=None):
    _timeout(sock, deadline)
    sock.sendall(struct.pack("!Q", len(payload)))
    _timeout(sock, deadline)
    sock.sendall(payload)


def receive_payload(sock, deadline=None):
    def read(size):
        result = bytearray()
        while len(result) < size:
            _timeout(sock, deadline)
            block = sock.recv(min(size - len(result), 1024 * 1024))
            if not block:
                raise EOFError("rocjitsu worker closed its connection")
            result.extend(block)
        return result

    size = struct.unpack("!Q", read(8))[0]
    return read(size)


def receive(sock, deadline=None):
    return pickle.loads(receive_payload(sock, deadline))
