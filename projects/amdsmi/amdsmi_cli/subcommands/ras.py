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

import sys
import time

from amdsmi_cli_exceptions import AmdSmiInvalidCommandException


class RasCommands:
    def ras(
        self,
        args,
        multiple_devices=False,
        gpu=None,
        cper=None,
        afid=None,
        decode=None,
        severity=None,
        folder=None,
        file_limit=None,
        cper_file=None,
        follow=None,
    ):
        """
        Retrieve and process CPER (RAS) entries for a target GPU.

        Expected command (all options only):
        amd-smi ras --cper --severity=nonfatal-uncorrected,fatal --folder <folder_name> --file-limit=1000 --follow

        Since no timestamp is provided on the command line, the function starts from a default cursor of 0.
        The output file name is auto-generated using the timestamp from the CPER header data (converted from
        the header’s "YYYY/MM/DD HH:MM:SS" format), along with the GPU/platform ID and error severity.
        """

        # GPU handle logic.
        if gpu:
            args.gpu = gpu
        if cper:
            args.cper = cper
        if afid:
            args.afid = afid
        if decode:
            args.decode = decode
        if severity:
            args.severity = severity
        if folder:
            args.folder = folder
        if file_limit:
            args.file_limit = file_limit
        if cper_file:
            args.cper_file = cper_file
        if follow:
            args.follow = follow
        if args.gpu == None:
            args.gpu = self.device_handles

        if args.afid:
            if args.cper_file:
                afids = self.helpers.cper_dump_afids(args.cper_file)
                if self.logger.is_json_format():
                    afid_output = {"cper_file": str(args.cper_file), "afids": afids}
                    self.logger.output = afid_output
                    self.logger.print_output()
                else:
                    print(" ".join(map(str, afids)))
                return
            else:
                command = " ".join(sys.argv[1:])
                message = f"Command '{command}' requires '--cper-file'. Run '--help' for more info."
                raise AmdSmiInvalidCommandException(command, self.logger.format, message)

        if not self.group_check_printed:
            self.helpers.check_required_groups()
            self.group_check_printed = True

        if not args.cper:
            return

        if not args.gpu:
            return

        if not isinstance(args.gpu, list):
            args.gpu = [args.gpu]

        args.cursor = [0] * len(args.gpu)

        # Using all the devices given in args.gpu
        # Populate a list of all the primary partition GPU ids (GPU 0, GPU 1, etc)
        partition_warning_flag = True
        primary_partition_gpu_ids = set()  # set of all primary partition GPU ids from arg.gpu
        for device_handle in args.gpu:
            # First get the partition
            partition_id = self.helpers.get_partition_id(device_handle)
            # If there is a single primary partition within args.gpu then we don't need to print the warning
            if partition_id == 0:
                partition_warning_flag = False
                break
            # Then attempt to get the primary GPU id for that partition
            primary_partition_gpu_id = self.helpers.get_primary_partition_gpu_id(device_handle)
            # Add to the set if it's a non-primary partition and we found a valid primary GPU id
            if partition_id != 0 and primary_partition_gpu_id is not None:
                primary_partition_gpu_ids.add(primary_partition_gpu_id)

        if partition_warning_flag:
            # Create a list of the primary partitions
            primary_partitions_str = " ".join(
                f"GPU{gpu_id}" for gpu_id in primary_partition_gpu_ids
            )

            print("WARNING: CPER files are only available on primary partitions")
            if len(primary_partition_gpu_ids) > 1:
                print(f"Try with primary partitions {primary_partitions_str}", end="")
            else:
                print(f"Try with primary partition {primary_partitions_str}", end="")

            print()

        is_json = self.logger.is_json_format()
        while True:
            all_json_rows = []
            for idx, device_handle in enumerate(args.gpu):
                rows = self.helpers.ras_cper(
                    args, device_handle, self.logger, idx, emit_json=not is_json
                )
                all_json_rows.extend(rows)
            if is_json and all_json_rows:
                self.logger.multiple_device_output = all_json_rows
                self.logger.print_output(multiple_device_enabled=True)
            if not args.follow:
                break
            time.sleep(1)
