#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

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
        event_thread = threading.Thread(target=self._event_thread, args=(self, args.gpu))
        event_thread.start()

        previous_sigterm_handler = signal.getsignal(signal.SIGTERM)
        system_exit_exc = None
        signal.signal(signal.SIGTERM, self._event_sigterm_handler)
        try:
            while True:
                try:
                    user_input = input()
                except EOFError:
                    self.stop = True
                    break
                except KeyboardInterrupt:
                    self.stop = True
                    break

                if self.stop:
                    break

                if user_input == "q":
                    print("Escape Sequence Detected; Exiting")
                    self.stop = True
                    break
        except SystemExit as exc:
            system_exit_exc = exc
        finally:
            self.stop = True
            event_thread.join()
            signal.signal(signal.SIGTERM, previous_sigterm_handler)

        if system_exit_exc is not None:
            raise system_exit_exc

    def _event_sigterm_handler(self, signum, frame):
        self.stop = True
        raise SystemExit(128 + signum)

    def _event_thread(self, commands, devices):
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
                        commands.logger.store_event_output(event["processor_handle"], values_dict)
                        commands.logger.print_event_output()
                except amdsmi_exception.AmdSmiLibraryException as e:
                    if e.err_code != amdsmi_interface.amdsmi_wrapper.AMDSMI_STATUS_NO_DATA:
                        print(e)
                except Exception as e:
                    print(e)
        finally:
            for listener in listeners:
                listener.stop()
