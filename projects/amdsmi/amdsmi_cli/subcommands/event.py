#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import queue
import signal
import threading

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
                raise payload
            if kind == "quit":
                print("Escape Sequence Detected; Exiting")
        except SystemExit as exc:
            system_exit_exc = exc
        except amdsmi_exception.AmdSmiLibraryException as e:
            raise e
        finally:
            self.stop = True
            for thread in threads:
                thread.join()
            signal.signal(signal.SIGTERM, previous_sigterm_handler)

        if system_exit_exc is not None:
            raise system_exit_exc

    def _event_sigterm_handler(self, signum, frame):
        self.stop = True
        raise SystemExit(128 + signum)

    def _read_stdin(self, result_queue):
        while True:
            try:
                user_input = input()
            except EOFError:
                result_queue.put(("eof", None))
                return
            except KeyboardInterrupt:
                result_queue.put(("interrupt", None))
                return

            if user_input == "q":
                result_queue.put(("quit", None))
                return

    def _event_thread(self, commands, i):
        devices = commands.device_handles
        if len(devices) == 0:
            print("No GPUs on machine")
            return

        # Check that KFD permissions are available
        if not self.group_check_printed:
            self.helpers.check_required_groups()
            self.group_check_printed = True

        device = devices[i]
        listener = amdsmi_interface.AmdSmiEventReader(
            device, amdsmi_interface.AmdSmiEvtNotificationType
        )
        values_dict = {}

        while not self.stop:
            try:
                events = listener.read(2000)
                for event in events:
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
            except amdsmi_exception.AmdSmiLibraryException as e:
                if e.err_code != amdsmi_interface.amdsmi_wrapper.AMDSMI_STATUS_NO_DATA:
                    print(e)
            except Exception as e:
                print(e)

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
