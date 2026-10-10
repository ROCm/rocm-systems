# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Loaded in ROCgdb's embedded Python interpreter by rocprofv3-att-gdb."""

import argparse
import contextlib
import json
import os
from pathlib import Path
import shlex
import socket
import struct
import sys
import threading
import time

import gdb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rocprofv3_att_gdb_common import parse_duration, parse_skip


def log(message):
    gdb.write(f"[att] {message}\n")


def start_thread(thread):
    # New Python threads must not handle GDB's SIGCHLD/SIGINT.
    with (
        gdb.blocked_signals()
        if hasattr(gdb, "blocked_signals")
        else contextlib.nullcontext()
    ):
        thread.daemon = True
        thread.start()


class Parser(argparse.ArgumentParser):
    def error(self, message):
        raise gdb.GdbError(message)


class Capture:
    control_timeout = 10
    connect_timeout = 10

    def __init__(self, config):
        self.config = config
        self.state = "IDLE"
        self.capture = 0
        self.start = self.end = None
        self.timeout = 0
        self.inferior = gdb.selected_inferior()
        self.held = {}
        self.stops = {}
        self.connection = None
        self.epoch = 0
        self.reader_pid = 0
        self.pending = []
        self.watchdog = None
        self.reason = ""
        self.started = 0
        self.trigger_thread = None
        self.continued = False
        self.timer_requested = False
        self.capture_stopped = False
        gdb.events.stop.connect(self.on_stop)
        gdb.events.cont.connect(self.on_continue)
        gdb.events.exited.connect(self.on_exit)
        if hasattr(gdb.events, "gdb_exiting"):
            gdb.events.gdb_exiting.connect(lambda event: self.close())

    def remove(self, name):
        breakpoint = getattr(self, name)
        if breakpoint is not None and breakpoint.is_valid():
            breakpoint.delete()
        setattr(self, name, None)

    def arm(self, start, stop=None, timeout=0, skip=0):
        if self.state not in ("IDLE", "DONE", "CANCELLED") and not (
            self.state == "ERROR" and not self.inferior.pid
        ):
            raise gdb.GdbError(f"cannot arm while {self.state}; use att cancel first")
        if not stop and not timeout:
            raise gdb.GdbError("provide --stop and/or --timeout")
        if not gdb.parameter("non-stop"):
            raise gdb.GdbError(
                "ATT requires 'set non-stop on' before starting the application"
            )
        self.inferior = gdb.selected_inferior()
        self.remove("start")
        self.remove("end")
        try:
            self.start = gdb.Breakpoint(start)
            self.start.ignore_count = skip
            if stop:
                self.end = gdb.Breakpoint(stop)
                self.end.enabled = False
        except Exception:
            self.remove("start")
            self.remove("end")
            raise
        self.capture += 1
        self.timeout = timeout
        self.held.clear()
        self.stops.clear()
        self.reason = ""
        self.trigger_thread = None
        self.continued = False
        self.timer_requested = False
        self.capture_stopped = False
        self.started = 0
        self.state = "ARMED"
        log(
            f"ARMED capture={self.capture}: {start} -> {stop or 'timeout'}"
            + (f", skip={skip} (start on hit {skip + 1})" if skip else "")
            + (f", timeout={timeout / 1000000:g}ms" if timeout else "")
        )

    def fail(self, message):
        self.state = "ERROR"
        self.reason = message
        self.cancel_watchdog()
        self.remove("start")
        self.remove("end")
        log(
            f"ERROR: {message}. Capture may be incomplete; owned stopped threads remain stopped."
        )
        if self.config["batch"]:
            gdb.execute("quit 1")

    def close(self):
        self.cancel_watchdog()
        self.close_connection()

    def close_connection(self):
        self.epoch += 1
        self.reader_pid = 0
        if self.connection:
            try:
                self.connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self.connection.close()
            self.connection = None

    def cancel_watchdog(self):
        if self.watchdog:
            self.watchdog.cancel()
            self.watchdog = None

    def wait_for_control(self, deadline=None):
        self.cancel_watchdog()
        capture = self.capture
        delay = self.control_timeout
        states = ("STARTING", "ACTIVE", "STOPPING")
        message = f"control did not complete within {self.control_timeout:g}s"
        if deadline is not None:
            delay = max(0, (deadline - time.monotonic_ns()) / 1e9 + delay)
            states = ("ACTIVE",)
            message = f"timed capture did not stop within {self.control_timeout:g}s after its deadline"

        def expired():
            if (
                self.watchdog is watchdog
                and capture == self.capture
                and self.state in states
            ):
                self.fail(
                    message + "; check profiler/runtime locks and GPU queue progress"
                )

        watchdog = threading.Timer(delay, lambda: gdb.post_event(expired))
        self.watchdog = watchdog
        start_thread(watchdog)

    def send(self, message):
        if self.connection is None:
            self.pending.append(message)
            return
        try:
            self.connection.sendall((message + "\n").encode())
        except OSError as error:
            self.fail(f"control connection failed: {error}")

    def on_continue(self, event):
        thread = getattr(event, "inferior_thread", None)
        if (
            self.timeout
            and not self.continued
            and self.trigger_thread is not None
            and thread in (None, self.trigger_thread)
        ):
            self.continued = True
            capture = self.capture
            # Continue events occur inside the resume command. Defer until that command
            # returns, including when a shared user breakpoint needs manual continuation.
            gdb.post_event(lambda: self.arm_timer(capture))
        if thread is None:
            self.stops.clear()
            self.held.clear()
        else:
            self.stops.pop(thread, None)
            self.held.pop(thread, None)
        self.connect_helper()

    def arm_timer(self, capture):
        if (
            capture == self.capture
            and self.state == "ACTIVE"
            and self.timeout
            and self.continued
            and not self.timer_requested
        ):
            self.timer_requested = True
            self.wait_for_control()
            self.send(f"CONTINUED {capture}")

    def connect_helper(self):
        if self.inferior.pid and self.reader_pid != self.inferior.pid:
            self.close_connection()
            self.reader_pid = self.inferior.pid
            epoch, pid = self.epoch, self.reader_pid
            path = self.config["socket"]

            def read_messages():
                connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                try:
                    deadline = time.monotonic() + self.connect_timeout
                    while epoch == self.epoch:
                        try:
                            connection.connect(path)
                            break
                        except (FileNotFoundError, ConnectionRefusedError):
                            if time.monotonic() >= deadline:
                                raise TimeoutError(
                                    "helper did not connect; check its preload and application startup"
                                )
                            time.sleep(
                                0.02
                            )  # Preparation only; no polling once connected.
                    if epoch != self.epoch:
                        return
                    peer = struct.unpack(
                        "3i",
                        connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12),
                    )
                    if peer[0] != pid or peer[1] != os.getuid():
                        raise RuntimeError(
                            "helper belongs to a different process or user"
                        )
                    with connection.makefile("rb") as stream:
                        while epoch == self.epoch:
                            line = stream.readline(4096)
                            if not line:
                                raise EOFError("target control connection closed")
                            message = line.decode().rstrip("\n")
                            gdb.post_event(
                                lambda message=message: self.receive(
                                    epoch, connection, message
                                )
                            )
                except Exception as error:
                    message = str(error)
                    gdb.post_event(lambda: self.connection_error(epoch, message))
                finally:
                    connection.close()

            start_thread(threading.Thread(target=read_messages))

    def connection_error(self, epoch, message):
        if epoch != self.epoch:
            return
        self.close_connection()
        if self.state in ("STARTING", "ACTIVE", "STOPPING"):
            self.fail(message)
        else:
            log(
                f"control unavailable: {message}; will retry on continuation or capture start"
            )

    def receive(self, epoch, connection, message):
        if epoch != self.epoch:
            return
        fields = message.split()
        if not fields:
            return
        if fields[0] == "HELLO":
            if (
                len(fields) != 4
                or fields[1] != "2"
                or int(fields[2]) != self.inferior.pid
            ):
                self.fail("incompatible helper handshake")
                return
            self.connection = connection
            log(f"control worker ready (PID {fields[2]}, TID {fields[3]})")
            pending, self.pending = self.pending, []
            for command in pending:
                self.send(command)
            return
        if fields[0] == "ERROR":
            if len(fields) >= 3 and int(fields[1]) in (0, self.capture):
                self.fail(" ".join(fields[2:]))
            return
        if len(fields) < 3 or int(fields[1]) != self.capture:
            return
        if fields[0] == "STARTED" and self.state in ("STARTING", "STOPPING"):
            self.started = int(fields[2])
            if self.state == "STOPPING":
                return
            self.cancel_watchdog()
            self.state = "ACTIVE"
            self.resume_owned()
            self.arm_timer(self.capture)
            log(f"ACTIVE capture={self.capture}; ROCTx start acknowledged")
        elif fields[0] == "TIMED" and self.state == "ACTIVE" and self.timer_requested:
            self.wait_for_control(int(fields[2]))
        elif fields[0] == "STOPPED" and self.state in ("STARTING", "ACTIVE", "STOPPING"):
            self.capture_stopped = True
            self.cancel_watchdog()
            self.remove("end")
            self.remove("start")
            self.state = "DONE"
            self.reason = self.reason or fields[2]
            elapsed = (int(fields[3]) - self.started) / 1000000 if self.started else 0
            log(
                f"DONE capture={self.capture} reason={self.reason} elapsed={elapsed:.3f}ms; "
                "raw ATT flushed, final output/decoding completes at application exit"
            )
            self.resume_owned()

    def resume_owned(self):
        held, self.held = self.held, {}
        selected = gdb.selected_thread()
        for thread, token in held.items():
            if (
                thread.is_valid()
                and thread.is_stopped()
                and self.stops.get(thread) is token
            ):
                thread.switch()
                gdb.execute("continue &", to_string=True)
        if selected and selected.is_valid():
            selected.switch()

    def on_stop(self, event):
        thread = getattr(event, "inferior_thread", None) or gdb.selected_thread()
        if thread is None or thread.inferior != self.inferior:
            return
        # Continuing or stopping again relinquishes ownership of the previous stop.
        self.stops.pop(thread, None)
        self.held.pop(thread, None)
        if not isinstance(event, gdb.BreakpointEvent):
            return
        hit = list(event.breakpoints)
        capture = self.capture
        start_hit, end_hit = self.start in hit, self.end in hit
        exclusive = all(bp in (self.start, self.end) for bp in hit)
        if start_hit or end_hit:
            token = object()
            self.stops[thread] = token
            gdb.post_event(
                lambda: self.handle_hit(
                    capture, thread, token, start_hit, end_hit, exclusive
                )
            )

    def handle_hit(self, capture, thread, token, start_hit, end_hit, exclusive):
        if (
            capture != self.capture
            or not thread.is_valid()
            or not thread.is_stopped()
            or self.stops.get(thread) is not token
        ):
            return
        if exclusive:
            self.held[thread] = token
        if start_hit and self.state == "ARMED":
            self.trigger_thread = thread
            log(f"start hit={self.start.hit_count}; activating capture={self.capture}")
            self.remove("start")  # Remove every owned location before trace activation.
            if self.end:
                self.end.enabled = True
            self.state = "STARTING"
            self.connect_helper()
            self.wait_for_control()
            self.send(f"START {self.capture} {self.timeout}")
        elif end_hit and self.state in ("STARTING", "ACTIVE", "STOPPING"):
            self.remove("end")
            self.stop_capture("breakpoint")
        elif start_hit and self.state in ("STARTING", "STOPPING", "ERROR"):
            # Several CPU threads can trap before the one-shot patch is removed.
            # Keep them held until the outstanding control operation completes.
            return
        else:
            self.resume_owned()

    def stop_capture(self, reason):
        if self.state == "STOPPING":
            return
        self.reason = reason
        self.state = "STOPPING"
        self.wait_for_control()
        self.send(f"STOP {self.capture}")

    def cancel(self):
        self.remove("start")
        self.remove("end")
        if self.state == "ERROR" and self.capture_stopped:
            self.state = "CANCELLED"
            self.held.clear()
            self.stops.clear()
            log(
                "CANCELLED; failed capture is stopped. Rearm, then continue the application."
            )
        elif self.state in ("STARTING", "ACTIVE", "STOPPING") or (
            self.state == "ERROR" and self.connection
        ):
            self.stop_capture("cancel")
        elif self.state == "ARMED":
            self.state = "CANCELLED"
            log("CANCELLED; no capture started")
        else:
            log(f"nothing to cancel ({self.state})")

    def on_exit(self, event):
        if event.inferior != self.inferior:
            return
        self.close()
        self.trigger_thread = None
        self.held.clear()
        self.stops.clear()
        self.pending.clear()
        self.remove("start")
        self.remove("end")
        try:
            os.unlink(self.config["socket"])
        except FileNotFoundError:
            pass
        success = self.state == "DONE" and getattr(event, "exit_code", 1) == 0
        if not success:
            log(
                f"incomplete capture or application failure: state={self.state}, "
                f"exit={getattr(event, 'exit_code', 'signal')}; check trigger locations and initialization"
            )
            self.state = "ERROR"
        if self.config["batch"]:
            gdb.post_event(lambda: gdb.execute(f"quit {0 if success else 1}"))


class ATT(gdb.Command):
    """ATT capture: att arm --start LOCATION [--skip N] (--stop LOCATION | --timeout 10ms), att run, att status, att cancel."""

    def __init__(self, capture):
        super().__init__("att", gdb.COMMAND_USER)
        self.capture = capture

    def invoke(self, text, from_tty):
        self.dont_repeat()
        arguments = shlex.split(text)
        if not arguments or arguments[0] == "status":
            remaining = (
                self.capture.start.ignore_count
                if self.capture.start and self.capture.start.is_valid()
                else 0
            )
            log(
                f"state={self.capture.state} capture={self.capture.capture} skip_remaining={remaining} reason={self.capture.reason or '-'}"
            )
        elif arguments == ["cancel"]:
            self.capture.cancel()
        elif arguments == ["run"]:
            try:
                gdb.execute("run")
            except gdb.error as error:
                self.capture.fail(f"application launch failed: {error}")
        elif arguments[0] == "arm":
            parser = Parser(prog="att arm", add_help=False, allow_abbrev=False)
            parser.add_argument("--start", required=True)
            parser.add_argument("--skip", type=parse_skip, default=0)
            parser.add_argument("--stop")
            parser.add_argument("--timeout", type=parse_duration, default=0)
            args = parser.parse_args(arguments[1:])
            self.capture.arm(args.start, args.stop, args.timeout, args.skip)
        else:
            raise gdb.GdbError(self.__doc__)


with open(os.environ.pop("ROCPROFV3_GDB_CONFIG")) as stream:
    config = json.load(stream)
gdb.execute("set non-stop on")
gdb.execute("set pagination off")
gdb.execute("set print thread-events off")
gdb.execute("set breakpoint pending on")
gdb.execute("set exec-wrapper " + shlex.join(config["wrapper"]))
gdb.execute("set environment ROCPROFV3_GDB_SOCKET " + config["socket"])
if config["batch"]:
    gdb.execute("set confirm off")
capture = Capture(config)
ATT(capture)
log(
    "CPU-breakpoint ATT prototype; choose a start after GPU initialization. Other application work keeps running."
)
if config["start"]:
    capture.arm(config["start"], config["stop"], config["timeout"], config.get("skip", 0))
else:
    log(
        "Use att arm --start LOCATION --stop LOCATION (or --timeout 10ms), then run. See help att."
    )
