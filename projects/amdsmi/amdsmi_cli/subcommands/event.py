#!/usr/bin/env python3
#
# Copyright (C) Advanced Micro Devices. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

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
        threads = []
        for device_handle in range(len(args.gpu)):
            x = threading.Thread(target=self._event_thread, args=(self, device_handle))
            threads.append(x)
            x.start()

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
            for thread in threads:
                thread.join()
            signal.signal(signal.SIGTERM, previous_sigterm_handler)

        if system_exit_exc is not None:
            raise system_exit_exc

    def _event_sigterm_handler(self, signum, frame):
        self.stop = True
        raise SystemExit(128 + signum)

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
