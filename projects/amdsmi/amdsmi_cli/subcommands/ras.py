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

import logging
import os
import sys
import time
from pathlib import Path

from amdsmi_cli_exceptions import AmdSmiInvalidCommandException, AmdSmiInvalidFilePathException


class RasCommands:
    def _validate_ras_args(self, args):
        """Validate ``ras`` arguments up front, before any driver call or file I/O.

        Runs after argparse has fully populated the namespace, so the mode flags
        (``--cper`` / ``--afid``) are reliable here even though an argparse action
        on ``--folder`` could not see them (the action fires mid-parse, before the
        mode flag may have been read). This is the single place that decides
        whether a ``--folder`` is a write target (``--cper``) or a pre-existing
        read source (``--afid``).
        """
        command = " ".join(sys.argv[1:])

        if args.afid:
            # Require exactly one of --cper-file / --folder under --afid.
            if bool(args.cper_file) == bool(args.folder):
                message = (
                    f"Command '{command}' requires exactly one of"
                    " '--cper-file' or '--folder'. Run '--help' for more info."
                )
                raise AmdSmiInvalidCommandException(command, self.logger.format, message)

            if args.folder:
                folder = Path(args.folder)
                # Refuse to follow a symlinked folder (it could redirect the read
                # into an arbitrary directory).
                if folder.is_symlink():
                    message = (
                        f"Folder '{folder}' is a symlink; refusing to follow it for"
                        " '--afid --folder'."
                    )
                    raise AmdSmiInvalidFilePathException(command, self.logger.format, message)
                if not folder.exists() or not folder.is_dir():
                    message = (
                        f"Folder '{folder}' does not exist or is not a directory."
                        " '--afid --folder' requires a folder of pre-existing CPER records."
                    )
                    raise AmdSmiInvalidFilePathException(command, self.logger.format, message)
        elif args.cper:
            if args.folder:
                folder = Path(args.folder)
                # Refuse to write into a symlinked folder.
                if folder.is_symlink():
                    message = (
                        f"Folder '{folder}' is a symlink; refusing to write CPER files into it."
                    )
                    raise AmdSmiInvalidFilePathException(command, self.logger.format, message)
                # Create the destination up front (and fail early if we cannot),
                # rather than deferring the error to the per-GPU write loop.
                try:
                    folder.mkdir(parents=True, exist_ok=True)
                except OSError as e:
                    message = f"Unable to create or access folder '{folder}': {e}"
                    raise AmdSmiInvalidFilePathException(command, self.logger.format, message)

    def _decode_afid_folder(self, args):
        """Decode AFIDs for every ``*.cper`` in ``args.folder``.

        The folder is already validated (exists, is a directory, not a symlink)
        by :meth:`_validate_ras_args`. Each file is opened with ``O_NOFOLLOW`` so a
        planted symlink (e.g. ``evil.cper`` -> ``/etc/shadow``) cannot be followed
        into an arbitrary-file read; this also closes the check-then-read TOCTOU
        window by rejecting symlinks atomically at ``open()`` time.
        """
        folder = Path(args.folder)
        cper_paths = sorted(folder.glob("*.cper"))
        if not cper_paths:
            command = " ".join(sys.argv[1:])
            message = (
                f"Folder '{folder}' contains no '.cper' files."
                " '--afid --folder' requires a folder of pre-existing CPER records."
            )
            raise AmdSmiInvalidFilePathException(command, self.logger.format, message)

        results = []
        for cper_path in cper_paths:
            try:
                fd = os.open(cper_path, os.O_RDONLY | os.O_NOFOLLOW)
            except OSError as e:
                # Symlink (O_NOFOLLOW -> ELOOP) or otherwise unreadable: skip it.
                # This is not a decode failure, so it is not surfaced as one.
                logging.debug("Skipping symlink/unreadable CPER file %s: %s", cper_path, e)
                continue
            try:
                raw = b""
                while True:
                    chunk = os.read(fd, 65536)
                    if not chunk:
                        break
                    raw += chunk
            finally:
                os.close(fd)

            decode_failed = False
            afids = []
            try:
                afids = self.helpers.cper_dump_afids(raw)
            except Exception as e:
                logging.debug("Failed to decode AFIDs from %s: %s", cper_path, e)
                decode_failed = True
            results.append(
                {"cper_file": str(cper_path), "afids": afids, "decode_failed": decode_failed}
            )

        if self.logger.is_json_format():
            self.logger.multiple_device_output = results
            self.logger.print_output(multiple_device_enabled=True)
        else:
            print(f"{'file_name':<32} list of afids")
            for entry in results:
                if entry["decode_failed"]:
                    afids_str = "decode failed"
                elif entry["afids"]:
                    afids_str = " ".join(map(str, entry["afids"]))
                else:
                    afids_str = "-"
                fname = Path(entry["cper_file"]).name
                print(f"{fname:<32} {afids_str}")

    def ras(
        self,
        args,
        multiple_devices=False,
        gpu=None,
        cper=None,
        afid=None,
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
        if args.gpu is None:
            args.gpu = self.device_handles

        # Validate all arguments before touching the driver or the filesystem.
        self._validate_ras_args(args)

        if args.afid:
            if args.cper_file:
                afids = self.helpers.cper_dump_afids(args.cper_file)
                if self.logger.is_json_format():
                    afid_output = {"cper_file": str(args.cper_file), "afids": afids}
                    self.logger.output = afid_output
                    self.logger.print_output()
                else:
                    print(" ".join(map(str, afids)) if afids else "-")
                return

            # --afid --folder: read-only decode of a pre-existing folder.
            self._decode_afid_folder(args)
            return

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

            self.helpers.cper_print(
                "WARNING: CPER files are only available on primary partitions", self.logger
            )
            if len(primary_partition_gpu_ids) > 1:
                self.helpers.cper_print(
                    f"Try with primary partitions {primary_partitions_str}", self.logger
                )
            else:
                self.helpers.cper_print(
                    f"Try with primary partition {primary_partitions_str}", self.logger
                )

        # One-shot warning, header, and follow prompt (kept out of the per-GPU helper
        # so the helper layer stays re-entrant and free of latching state).
        if not self.logger.is_json_format():
            if not args.folder:
                self.helpers.cper_print(
                    "WARNING: No CPER files will be dumped unless "
                    "--folder=<folder_name> is specified and cper entries exist.",
                    self.logger,
                )
            self.helpers._print_header(args.folder, self.logger)
            if args.follow:
                # Always print to stdout so the user sees the prompt, even with --file.
                print("Press CTRL + C to stop.")

        # Shared 1-indexed counter for generated CPER filenames across all GPUs
        # and follow iterations within this invocation.
        cper_counter = [0]
        is_json = self.logger.is_json_format()
        while True:
            all_json_rows = []
            for idx, device_handle in enumerate(args.gpu):
                rows = self.helpers.ras_cper(
                    args,
                    device_handle,
                    self.logger,
                    idx,
                    emit_inline=not is_json,
                    cper_counter=cper_counter,
                )
                all_json_rows.extend(rows)
            if is_json and all_json_rows:
                self.logger.multiple_device_output = all_json_rows
                self.logger.print_output(multiple_device_enabled=True)
            if not args.follow:
                break
            time.sleep(1)
