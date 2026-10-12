#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import os
import queue
import signal
import sys
import threading
import time

from amdsmi import amdsmi_exception, amdsmi_interface


class EventCommands:
    def event(self, args, gpu=None):
        """Get event information for target gpus

        Args:
            args (Namespace): argparser args to pass to subcommand
            gpu (device_handle, optional): device_handle for target device. Defaults to None.

        Return:
            stdout event information for target gpus
        """
        if args.gpu:
            gpu = args.gpu

        if gpu == None:
            args.gpu = self.device_handles

        if not isinstance(args.gpu, list):
            args.gpu = [args.gpu]

        print("EVENT LISTENING:\n")
        print("Press q and hit ENTER when you want to stop.")
        self.stop = False
        result_queue = queue.Queue()
        threads = []
        for device_handle in range(len(args.gpu)):
            x = self.EventListenerThread(
                target=self._event_thread, args=(self, device_handle), result_queue=result_queue
            )
            threads.append(x)
            x.start()

        stdin_reader = threading.Thread(target=self._read_stdin, args=(result_queue,), daemon=True)
        stdin_reader.start()

        previous_sigterm_handler = signal.getsignal(signal.SIGTERM)
        system_exit_exc = None
        signal.signal(signal.SIGTERM, self._event_sigterm_handler)
        try:
            kind, payload = result_queue.get()
            if kind == "exception":
                if isinstance(payload, amdsmi_exception.AmdSmiLibraryException):
                    raise payload
                raise payload
            if kind == "quit":
                print("Escape Sequence Detected; Exiting")
            if kind == "eof" or kind == "interrupt":
                print("Input stream closed or interrupted; Exiting")
        except SystemExit as exc:
            system_exit_exc = exc
        except amdsmi_exception.AmdSmiLibraryException as e:
            raise e
        finally:
            self.stop = True
            event_thread.join()
            signal.signal(signal.SIGTERM, previous_sigterm_handler)

        if system_exit_exc is not None:
            raise system_exit_exc

    def _event_sigterm_handler(self, signum, frame):
        self.stop = True
        raise SystemExit(128 + signum)

    def _read_stdin(self, result_queue):
        try:
            os.set_blocking(sys.stdin.fileno(), False)
        except (AttributeError, OSError, ValueError):
            # stdin is closed or invalid (e.g. 0<&-); nothing to read.
            result_queue.put(("eof", None))
            return

        try:
            while not self.stop:
                try:
                    line = sys.stdin.readline()
                except BlockingIOError:
                    time.sleep(0.1)
                    continue
                except (OSError, ValueError):
                    result_queue.put(("eof", None))
                    return
                except KeyboardInterrupt:
                    result_queue.put(("interrupt", None))
                    return

                if line == "":
                    result_queue.put(("eof", None))
                    return
                if line.strip() == "q":
                    result_queue.put(("quit", None))
                    return
        except Exception as e:  # safety net: always unblock the main thread's get()
            result_queue.put(("exception", e))

    def _event_thread(self, commands, i):
        devices = commands.device_handles
        if len(devices) == 0:
            print("No GPUs on machine")
            return

        # Check that KFD permissions are available
        if not self.group_check_printed:
            self.helpers.check_required_groups()
            self.group_check_printed = True

        listeners = [
            amdsmi_interface.AmdSmiEventReader(device, amdsmi_interface.AmdSmiEvtNotificationType)
            for device in devices
        ]
        values_dict = {}

        try:
            while not self.stop:
                try:
                    # read() is a global poll: amdsmi_get_gpu_event_notification()
                    # returns queued events for ALL registered devices, each tagged
                    # with its own processor_handle. A single listener's read()
                    # therefore drains every GPU, not just listeners[0].
                    events = listeners[0].read(2000)
                except amdsmi_exception.AmdSmiLibraryException as e:
                    if e.err_code != amdsmi_interface.amdsmi_wrapper.AMDSMI_STATUS_NO_DATA:
                        print(e, file=sys.stderr)
                    continue

                for event in events:
                    values_dict["timestamp"] = event["timestamp"]
                    values_dict["event"] = event["event"]
                    # parse message as it's own dictionary
                    message_list = event["message"].split("  ")
                    message_dict = {}
                    for item in message_list:
                        if not item == "":
                            item_list = item.split(": ")
                            message_dict.update({item_list[0]: item_list[1]})
                    values_dict["message"] = message_dict
                    commands.logger.store_output(event["processor_handle"], "values", values_dict)
                    commands.logger.print_output()
            except Exception as e:
                        # Isolate per-event failures (e.g. a malformed message) so
                        # one bad record can't drop its batch siblings; keep the
                        # stdout record stream clean by logging to stderr.
                        print(e, file=sys.stderr)
        finally:
            for listener in listeners:
                listener.stop()

    # Thread class for capture exceptions from event listener threads which will be passed to main thread
    class EventListenerThread(threading.Thread):
        def __init__(self, target, args, result_queue):
            super().__init__()
            self.target = target
            self.args = args
            self.result_queue = result_queue

        def run(self):
            try:
                self.target(*self.args)
            except Exception as e:
                self.result_queue.put(("exception", e))
