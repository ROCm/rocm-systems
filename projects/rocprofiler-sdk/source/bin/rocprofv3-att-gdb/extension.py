# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""CPU breakpoint triggers for a session of in-process ATT participants."""

import argparse
import contextlib
import json
import os
from pathlib import Path
import shlex
import sys
import threading
import time
import uuid

import gdb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rocprofv3_att_gdb_common import parse_duration, parse_skip
from rocprofv3_att_gdb_session import Session


def log(message):
    gdb.write(f"[att] {message}\n")


def start_thread(thread):
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

    def __init__(self, config):
        self.config = config
        self.state = "IDLE"
        self.capture = 0
        self.start = self.end = None
        self.timeout = 0
        self.held = {}
        self.stops = {}
        self.participants = {}
        self.group = {}
        self.watchdog = None
        self.reason = ""
        self.error = ""
        self.trigger_thread = None
        self.trigger_inferior = None
        self.trigger_pid = 0
        self.continued = False
        self.timer_requested = False
        self.deadline = 0
        self.exit_failed = False
        self.closed = False
        self.session = Session(config["socket"], config["session"], self.deliver)
        start_thread(threading.Thread(target=self.session.run))
        gdb.events.stop.connect(self.on_stop)
        gdb.events.cont.connect(self.on_continue)
        gdb.events.exited.connect(self.on_exit)
        if hasattr(gdb.events, "executable_changed"):
            gdb.events.executable_changed.connect(self.on_exec)
        if hasattr(gdb.events, "gdb_exiting"):
            gdb.events.gdb_exiting.connect(lambda event: self.close())

    def deliver(self, peer, message):
        gdb.post_event(lambda: self.receive(peer, message) if not self.closed else None)

    def close(self):
        self.closed = True
        self.cancel_watchdog()
        self.session.close()

    def remove(self, name):
        breakpoint = getattr(self, name)
        if breakpoint is not None and breakpoint.is_valid():
            breakpoint.delete()
        setattr(self, name, None)

    def arm(self, start, stop=None, timeout=0, skip=0):
        if self.state not in ("IDLE", "DONE", "CANCELLED"):
            raise gdb.GdbError(f"cannot arm while {self.state}; use att cancel first")
        if not stop and not timeout:
            raise gdb.GdbError("provide --stop and/or --timeout")
        if not gdb.parameter("non-stop"):
            raise gdb.GdbError(
                "ATT requires 'set non-stop on' before starting the application"
            )
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
        self.group = {}
        self.reason = self.error = ""
        self.trigger_thread = self.trigger_inferior = None
        self.trigger_pid = 0
        self.continued = self.timer_requested = False
        self.deadline = 0
        self.state = "ARMED"
        log(
            f"ARMED capture={self.capture}: {start} -> {stop or 'timeout'}"
            + (f", skip={skip} (start on hit {skip + 1})" if skip else "")
            + (f", timeout={timeout / 1000000:g}ms" if timeout else "")
        )

    def cancel_watchdog(self):
        if self.watchdog:
            self.watchdog.cancel()
            self.watchdog = None

    def wait_for_control(self, deadline=None):
        self.cancel_watchdog()
        capture = self.capture
        delay = self.control_timeout
        message = f"control did not complete within {delay:g}s"
        if deadline is not None:
            delay += max(0, (deadline - time.monotonic_ns()) / 1e9)
            message = f"timed capture did not stop within {self.control_timeout:g}s after its deadline"

        def expired():
            if self.watchdog is not watchdog or capture != self.capture:
                return
            self.watchdog = None
            if self.state == "ERROR":
                log("cleanup acknowledgement missing; capture outcome remains unknown")
                self.write_manifest()
                if self.config["batch"]:
                    gdb.execute("quit 1")
            else:
                self.fail(
                    message + "; check participant progress and profiler/runtime locks"
                )

        watchdog = threading.Timer(delay, lambda: gdb.post_event(expired))
        self.watchdog = watchdog
        start_thread(watchdog)

    def send_group(self, op, **fields):
        for peer, participant in self.group.items():
            if not participant.get("lost") and not participant.get("stopped"):
                self.session.send(peer, dict(op=op, capture=self.capture, **fields))

    def fail(self, message):
        first_error = not self.error
        if not self.error:
            self.error = message
            log(
                f"ERROR: {message}. Capture may be incomplete; owned stopped threads remain stopped."
            )
        self.state = "ERROR"
        self.remove("start")
        self.remove("end")
        if first_error:
            self.send_group("STOP")
        if self.group and not self.group_finished() and first_error:
            self.wait_for_control()
        elif self.group_finished():
            self.finish_group()

    def begin_group(self):
        self.group = {
            peer: dict(registration=registration)
            for peer, registration in self.participants.items()
        }
        if not self.group:
            self.fail(
                "no ATT participants registered; start after profiler initialization"
            )
            return
        self.wait_for_control()
        self.send_group("START")
        log(
            f"STARTING capture={self.capture} trigger_pid={self.trigger_pid} participants={len(self.group)}"
        )

    def receive(self, peer, message):
        try:
            self.handle_message(peer, message)
        except (KeyError, TypeError, ValueError, gdb.error) as error:
            self.fail(f"invalid participant reply or debugger operation: {error}")

    def handle_message(self, peer, message):
        op = message["op"]
        if op == "REGISTER":
            if peer in self.participants:
                raise ValueError("duplicate registration")
            self.participants[peer] = message
            self.session.send(peer, {"op": "REGISTERED"})
            log(f"control worker ready (PID {message['pid']}, TID {message['tid']})")
            return
        if op == "SNAPSHOT":
            if message["tag"] == ["start", self.capture] and self.state == "STARTING":
                self.begin_group()
            else:
                if message["tag"] == ["exit", self.capture, True]:
                    self.trigger_lost()
                self.maybe_finish_batch()
            return
        if op in ("DISCONNECTED", "BYE"):
            self.participants.pop(peer, None)
            participant = self.group.get(peer)
            if participant is not None and not participant.get("stopped"):
                participant["lost"] = message.get(
                    "message", message.get("reason", "exit")
                )
                self.fail(
                    f"participant PID {participant['registration']['pid']} left during capture"
                )
            self.maybe_finish_batch()
            return
        if peer not in self.group or message.get("capture") != self.capture:
            return  # Old connection/capture replies cannot change a new capture.
        participant = self.group[peer]
        if op == "ERROR":
            participant["error"] = message["message"]
            self.fail(
                f"participant PID {participant['registration']['pid']}: {message['message']}"
            )
        elif op == "STARTED":
            participant["started"] = int(message["timestamp"])
            if self.state == "STARTING" and all(
                p.get("started") for p in self.group.values()
            ):
                self.cancel_watchdog()
                self.state = "ACTIVE"
                self.resume_owned()
                self.arm_timer(self.capture)
                log(
                    f"ACTIVE capture={self.capture}; {len(self.group)} participant start acknowledgement(s)"
                )
        elif op == "TIMED":
            if int(message["deadline"]) != self.deadline:
                raise ValueError("participant acknowledged a different deadline")
            participant["timed"] = True
            if self.state == "ACTIVE" and all(
                p.get("timed") for p in self.group.values()
            ):
                self.wait_for_control(self.deadline)
        elif op == "STOPPED":
            if participant.get("stopped"):
                return
            participant["stopped"] = int(message["timestamp"])
            participant["reason"] = message["reason"]
            if message["reason"] == "timeout":
                if not self.deadline or participant["stopped"] < self.deadline:
                    self.fail("participant timeout preceded the group deadline")
                    return
                if self.state == "ACTIVE":
                    self.stop_capture("timeout")
            elif message["reason"] not in ("request", "exit") and self.state != "ERROR":
                self.fail(f"participant stopped unexpectedly: {message['reason']}")
            if self.group_finished():
                self.finish_group()
        else:
            raise ValueError(f"unknown reply {op}")

    def group_finished(self):
        return all(p.get("stopped") or p.get("lost") for p in self.group.values())

    def finish_group(self):
        if self.state == "DONE":
            return
        self.cancel_watchdog()
        self.remove("start")
        self.remove("end")
        if not self.error:
            self.state = "DONE"
            if not self.reason:
                self.reason = (
                    "exit"
                    if any(p.get("reason") == "exit" for p in self.group.values())
                    else "request"
                )
            log(
                f"DONE capture={self.capture} reason={self.reason or 'request'}; "
                "participant stop acknowledgements received, output/decoding completes at process exit"
            )
        self.write_manifest()
        if not self.error:
            self.resume_owned()
        elif self.config["batch"]:
            gdb.execute("quit 1")
        self.maybe_finish_batch()

    def write_manifest(self):
        directories = {p["registration"].get("directory") for p in self.group.values()}
        manifest = dict(
            session=self.config["session"],
            capture=self.capture,
            trigger_pid=self.trigger_pid,
            deadline=self.deadline,
            state=self.state,
            reason=self.reason,
            error=self.error,
            participants=list(self.group.values()),
        )
        for directory in directories - {None}:
            try:
                path = Path(directory) / f"capture_{self.capture}.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(manifest, indent=2) + "\n")
            except OSError as error:
                log(f"unable to write capture manifest: {error}")

    def on_continue(self, event):
        thread = getattr(event, "inferior_thread", None)
        if self.trigger_thread is not None and thread in (None, self.trigger_thread):
            self.continued = True
            capture = self.capture
            # The continue event is emitted inside the resume command. Defer past its return.
            gdb.post_event(lambda: self.arm_timer(capture))
        if thread is None:
            self.stops.clear()
            self.held.clear()
        else:
            self.stops.pop(thread, None)
            self.held.pop(thread, None)

    def arm_timer(self, capture):
        if (
            capture == self.capture
            and self.state == "ACTIVE"
            and self.timeout
            and self.continued
            and not self.timer_requested
        ):
            self.timer_requested = True
            self.deadline = time.monotonic_ns() + self.timeout
            self.wait_for_control(self.deadline)
            self.send_group("DEADLINE", deadline=self.deadline)

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
        if thread is None:
            return
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
            self.trigger_inferior = thread.inferior
            self.trigger_pid = thread.inferior.pid
            log(f"start hit={self.start.hit_count}; activating capture={self.capture}")
            self.remove("start")
            if self.end:
                self.end.inferior = thread.inferior.num
                self.end.enabled = True
            self.state = "STARTING"
            self.wait_for_control()
            self.session.snapshot(["start", self.capture])
        elif end_hit and self.state in ("STARTING", "ACTIVE", "STOPPING"):
            self.remove("end")
            self.stop_capture("breakpoint")
        elif start_hit and self.state in ("STARTING", "STOPPING", "ERROR"):
            return  # Concurrent start hits remain held through the boundary.
        else:
            self.resume_owned()

    def stop_capture(self, reason):
        if self.state == "STOPPING":
            return
        self.reason = reason
        if self.state == "STARTING" and not self.group:
            self.state = "CANCELLED"
            self.cancel_watchdog()
            self.remove("start")
            self.remove("end")
            self.resume_owned()
            log("CANCELLED; no capture started")
            return
        self.state = "STOPPING"
        self.remove("start")
        self.remove("end")
        self.wait_for_control()
        self.send_group("STOP")
        if self.group_finished():
            self.finish_group()

    def cancel(self):
        self.remove("start")
        self.remove("end")
        if self.state == "ERROR" and self.group_finished():
            self.state = "CANCELLED"
            self.held.clear()
            self.stops.clear()
            log("CANCELLED; rearm, then continue the application")
        elif self.state in ("STARTING", "ACTIVE", "STOPPING", "ERROR"):
            self.stop_capture("cancel")
        elif self.state == "ARMED":
            self.state = "CANCELLED"
            log("CANCELLED; no capture started")
        else:
            log(f"nothing to cancel ({self.state})")

    def on_exec(self, event):
        if (
            self.trigger_inferior
            and self.trigger_inferior.is_valid()
            and event.progspace == self.trigger_inferior.progspace
        ):
            self.trigger_lost()

    def trigger_lost(self):
        if self.state in ("STARTING", "ACTIVE"):
            self.remove("end")
            if not self.timer_requested:
                self.fail("trigger process exited or exec'd before its stop boundary")

    def on_exit(self, event):
        if getattr(event, "exit_code", 1) != 0:
            self.exit_failed = True
        # Drain STOPPED/BYE before deciding whether a process exited during its capture.
        self.session.snapshot(
            ["exit", self.capture, event.inferior == self.trigger_inferior]
        )

    def maybe_finish_batch(self):
        if any(inferior.pid for inferior in gdb.inferiors()) or self.participants:
            return
        success = self.state == "DONE" and not self.exit_failed
        if not success:
            log(
                f"incomplete capture or application failure: state={self.state}; check trigger locations and initialization"
            )
        if self.config["batch"]:
            gdb.post_event(lambda: gdb.execute(f"quit {0 if success else 1}"))


class ATT(gdb.Command):
    """ATT capture: att arm --start LOCATION [--skip N] (--stop LOCATION | --timeout 10ms), att run, att status, att cancel, att session."""

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
                f"state={self.capture.state} capture={self.capture.capture} skip_remaining={remaining} "
                f"participants={len(self.capture.participants)} reason={self.capture.reason or '-'}"
            )
            for registration in self.capture.participants.values():
                log(
                    f"participant pid={registration['pid']} instance={registration['instance']} "
                    f"output={registration.get('output', '-')}"
                )
        elif arguments == ["session"]:
            log(f"session file: {self.capture.config.get('session_file', '-')}")
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
config.setdefault("session", uuid.uuid4().hex)
gdb.execute("set non-stop on")
gdb.execute("set detach-on-fork off")
gdb.execute("set follow-fork-mode parent")
gdb.execute("set follow-exec-mode same")
gdb.execute("set pagination off")
gdb.execute("set print thread-events off")
gdb.execute("set breakpoint pending on")
gdb.execute("set exec-wrapper " + shlex.join(config["wrapper"]))
gdb.execute("set environment ROCPROFV3_GDB_SOCKET " + config["socket"])
gdb.execute("set environment ROCPROFV3_GDB_SESSION " + config["session"])
if config["batch"]:
    gdb.execute("set confirm off")
capture = Capture(config)
ATT(capture)
log(
    "CPU-breakpoint ATT session; descendants share triggers. Other application work keeps running."
)
if config["start"]:
    capture.arm(config["start"], config["stop"], config["timeout"], config.get("skip", 0))
else:
    log(
        "Use att arm --start LOCATION --stop LOCATION (or --timeout 10ms), then run. See help att."
    )
