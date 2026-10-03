# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Private worker entry point; stdout/stderr remain ordinary application streams."""

import os
import pickle
import socket
import sys
import traceback

from ._protocol import PROTOCOL_VERSION, dumps, receive, receive_payload, send


def main():
    with socket.socket(fileno=int(sys.argv[1])) as channel:
        channel.set_inheritable(False)
        send(channel, dumps(("ready", PROTOCOL_VERSION, sys.version_info[:3])))
        paths = receive(channel)
        sys.path[:] = paths + [path for path in sys.path if path not in paths]
        send(channel, dumps(("initialized", os.getpid())))
        while True:
            try:
                payload = receive_payload(channel)
            except EOFError:
                return
            try:
                request = pickle.loads(payload)
                if request is None:
                    return
                function, args, kwargs = request
                result = function(*args, **kwargs)
                # Returning CPU data does not necessarily wait for unrelated streams.
                torch = sys.modules.get("torch")
                if torch is not None and torch.cuda.is_initialized():
                    for device in range(torch.cuda.device_count()):
                        torch.cuda.synchronize(device)
                response = dumps(("result", result))
            except BaseException as error:
                try:
                    message = str(error)
                except BaseException:
                    message = "Exception message could not be formatted"
                response = dumps(
                    ("error", type(error).__name__, message, traceback.format_exc())
                )
            send(channel, response)


if __name__ == "__main__":
    main()
