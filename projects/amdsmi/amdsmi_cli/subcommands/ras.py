#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import json
import logging
import os
import sys
import time
from pathlib import Path

from amdsmi_cli_exceptions import AmdSmiInvalidCommandException, AmdSmiInvalidFilePathException

from amdsmi import amdsmi_exception


class RasCommands:
    def _print_decode_human(self, cper_file, addc: dict):
        """Print the ADDC decode report in amd-smi human-readable style."""
        source = addc.get("source", {})
        events = addc.get("events", [])

        def _v(val):
            return "N/A" if val is None else str(val)

        def _get(d, *keys):
            for k in keys:
                if not isinstance(d, dict):
                    return None
                d = d.get(k)
            return d

        print(f"CPER_FILE: {Path(cper_file).name}")
        print(f"    SCHEMA_VERSION: {_v(addc.get('schema_version'))}")
        print(f"    TIMESTAMP: {_v(source.get('timestamp'))}")
        print(f"    PLATFORM_ID: {_v(source.get('platform_id'))}")
        print(f"    CREATOR_ID: {_v(source.get('creator_id'))}")

        if not events:
            print("    EVENTS: N/A")
            return

        for i, ev in enumerate(events):
            meta = ev.get("metadata", {})
            plat = ev.get("platform", {})
            report = ev.get("event_report", {})
            analysis = ev.get("analysis", {})

            afids = analysis.get("afid", [])
            afid_str = " ".join(str(a) for a in afids) if afids else "N/A"

            print(f"    EVENT[{i}]:")
            print(f"        SECTION: {_v(meta.get('section_index', i))}")
            print(f"        CONTEXT_TYPE: {_v(meta.get('context_type'))}")
            print(f"        CLASS: {_v(meta.get('event_class'))}")
            print(f"        PLATFORM: {_v((_get(meta, 'addc_version') or {}).get('name'))}")
            print(f"        AFID: {afid_str}")
            print(f"        SEVERITY: {_v(report.get('error_severity'))}")
            print(f"        FRU: {_v(plat.get('fru'))}")
            print(f"        TIMESTAMP: {_v(meta.get('timestamp') or source.get('timestamp'))}")
            print(f"        BANK: {_v(report.get('bank'))}")
            print(f"        ERROR_TYPE: {_v(report.get('error_type'))}")
            print(f"        DESCRIPTION: {_v(report.get('error_description'))}")
            print(f"        INSTANCE: {_v(_get(report, 'error_location', 'instance'))}")
            print(f"        SOCKET: {_v(_get(report, 'error_location', 'socket'))}")
            print(f"        DIE: {_v(_get(report, 'error_location', 'die'))}")
            print(f"        CPU: {_v(_get(report, 'error_location', 'cpu'))}")
            hbm = _get(report, "error_location", "hbm")
            if isinstance(hbm, dict):
                print("        HBM:")
                print(f"            COL: {_v(hbm.get('col'))}")
                print(f"            BANK: {_v(hbm.get('bank'))}")
                print(f"            ROW: {_v(hbm.get('row'))}")
                print(f"            PC: {_v(hbm.get('pc'))}")
                print(f"            SID: {_v(hbm.get('sid'))}")
            else:
                print(f"        HBM: {_v(hbm)}")
            print(f"        SYNDROME: {_v(report.get('error_syndrome'))}")
            print(f"        ADDRESS: {_v(report.get('error_address'))}")
            print(f"        UC: {_v(_get(report, 'misc_status', 'uc'))}")
            print(f"        PCC: {_v(_get(report, 'misc_status', 'pcc'))}")
            print(f"        TCC: {_v(_get(report, 'misc_status', 'tcc'))}")
            print(f"        DEFERRED: {_v(_get(report, 'misc_status', 'deferred'))}")
            print(f"        ADDR_VALID: {_v(_get(report, 'misc_status', 'addr_v'))}")
            print(f"        CECC: {_v(_get(report, 'misc_status', 'cecc'))}")
            print(f"        UECC: {_v(_get(report, 'misc_status', 'uecc'))}")
            print(f"        POISON: {_v(_get(report, 'misc_status', 'poison'))}")
            print(f"        TRANSPARENT: {_v(_get(report, 'misc_status', 'transparent'))}")
            print(f"        SCRUB: {_v(_get(report, 'misc_status', 'scrub'))}")

    def _decode_to_csv_rows(self, cper_file, addc: dict) -> list:
        """Flatten each ADDC event into one CSV row dict."""
        source = addc.get("source", {})
        schema = addc.get("schema_version", "")
        rows = []

        def _s(v):
            return "" if v is None else str(v)

        def _get(d, *keys):
            for k in keys:
                if not isinstance(d, dict):
                    return None
                d = d.get(k)
            return d

        for i, ev in enumerate(addc.get("events", [])):
            meta = ev.get("metadata", {})
            plat = ev.get("platform", {})
            report = ev.get("event_report", {})
            analysis = ev.get("analysis", {})
            afids = analysis.get("afid", [])
            afid_val = ",".join(str(a) for a in afids) if afids else ""
            rows.append(
                {
                    "file_name": Path(cper_file).name,
                    "schema_version": schema,
                    "timestamp": source.get("timestamp", ""),
                    "platform_id": source.get("platform_id", ""),
                    "creator_id": source.get("creator_id", ""),
                    "event_index": i,
                    "section": meta.get("section_index", i),
                    "context_type": meta.get("context_type", ""),
                    "class": meta.get("event_class", ""),
                    "platform": (_get(meta, "addc_version") or {}).get("name", ""),
                    "afid": afid_val,
                    "severity": report.get("error_severity", ""),
                    "fru_id": plat.get("fru_id", ""),
                    "fru_text": plat.get("fru", ""),
                    "bank": _s(report.get("bank")),
                    "error_type": _s(report.get("error_type")),
                    "description": _s(report.get("error_description")),
                    "instance": _s(_get(report, "error_location", "instance")),
                    "socket": _s(_get(report, "error_location", "socket")),
                    "die": _s(_get(report, "error_location", "die")),
                    "cpu": _s(_get(report, "error_location", "cpu")),
                    "hbm": (
                        json.dumps(_get(report, "error_location", "hbm"))
                        if isinstance(_get(report, "error_location", "hbm"), dict)
                        else _s(_get(report, "error_location", "hbm"))
                    ),
                    "syndrome": _s(report.get("error_syndrome")),
                    "address": _s(report.get("error_address")),
                    "uc": _s(_get(report, "misc_status", "uc")),
                    "pcc": _s(_get(report, "misc_status", "pcc")),
                    "tcc": _s(_get(report, "misc_status", "tcc")),
                    "deferred": _s(_get(report, "misc_status", "deferred")),
                    "addr_valid": _s(_get(report, "misc_status", "addr_v")),
                    "cecc": _s(_get(report, "misc_status", "cecc")),
                    "uecc": _s(_get(report, "misc_status", "uecc")),
                    "poison": _s(_get(report, "misc_status", "poison")),
                    "transparent": _s(_get(report, "misc_status", "transparent")),
                    "scrub": _s(_get(report, "misc_status", "scrub")),
                }
            )
        return rows

    def _emit_decode_records(self, cper_file, addc: dict):
        """Dispatch ADDC decode output by format."""
        if self.logger.is_human_readable_format():
            self._print_decode_human(cper_file, addc)
        elif self.logger.is_json_format():
            print(json.dumps(addc, indent=4))
        else:  # CSV
            rows = self._decode_to_csv_rows(cper_file, addc)
            self.logger.multiple_device_output = rows
            self.logger.print_output(multiple_device_enabled=True, emit_empty=True)

    def _build_afid_record(self, cper_file, afids=None, code=None):
        """Shared with ``ras --cper``; see AMDSMIHelpers.build_afid_record."""
        return self.helpers.build_afid_record(cper_file, self.logger.format, afids=afids, code=code)

    def _emit_afid_records(self, records):
        """Render decoded --afid records.

        Human output is a compact ``file_name | list of afids`` table (the
        logger's tabular renderer smushes these wide rows). json/csv keep the
        structured schema (cper_file, afids, status, code, message) so machine
        consumers still get the real AMDSMI_STATUS per file.
        """
        if self.logger.is_human_readable_format():
            name_w = max([len("file_name")] + [len(Path(r["cper_file"]).name) for r in records])
            print(f"{'file_name':<{name_w}}  list of afids")
            for r in records:
                name = Path(r["cper_file"]).name
                print(f"{name:<{name_w}}  {self.helpers.afid_cell(r)}")
        else:
            rows = records
            if self.logger.is_csv_format():
                # csv.DictWriter would render the list as "[1, 2, 3]". JSON keeps the list.
                rows = [{**r, "afids": self.helpers.afid_cell(r)} for r in rows]
            self.logger.multiple_device_output = rows
            # emit_empty keeps a folder whose files were all skipped rendering as
            # `[]` rather than nothing, so stdout stays parseable.
            self.logger.print_output(multiple_device_enabled=True, emit_empty=True)

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
        elif args.decode:
            if not args.cper_file:
                message = f"Command '{command}' requires '--cper-file'. Run '--help' for more info."
                raise AmdSmiInvalidCommandException(command, self.logger.format, message)
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

            try:
                afids = self.helpers.cper_dump_afids(raw)
                results.append(self._build_afid_record(cper_path, afids=afids))
            except amdsmi_exception.AmdSmiLibraryException as e:
                logging.debug("Failed to decode AFIDs from %s: %s", cper_path, e)
                # Records the real library status so the decode failure reaches
                # the exit code. NO_PERM aborts instead, since it is not per-file.
                self.helpers.record_or_raise(e)
                results.append(self._build_afid_record(cper_path, code=e.get_error_code()))

        self._emit_afid_records(results)

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
        if decode:
            args.decode = decode

        # Validate all arguments before touching the driver or the filesystem.
        self._validate_ras_args(args)

        if args.afid:
            if args.cper_file:
                # Read failure -> clean file-path error. Decode failure -> a
                # unified per-file record with its AMDSMI_STATUS (same schema as
                # --folder), and a non-zero exit via the error collector.
                try:
                    afids = self.helpers.cper_dump_afids(args.cper_file)
                    record = self._build_afid_record(args.cper_file, afids=afids)
                except OSError as e:
                    raise AmdSmiInvalidFilePathException(
                        args.cper_file,
                        self.logger.format,
                        f"Unable to read CPER file '{args.cper_file}': {e}",
                    ) from e
                except amdsmi_exception.AmdSmiLibraryException as e:
                    self.helpers.record_or_raise(e)
                    record = self._build_afid_record(args.cper_file, code=e.get_error_code())
                self._emit_afid_records([record])
                return

            # --afid --folder: read-only decode of a pre-existing folder.
            self._decode_afid_folder(args)
            return

        if args.decode:
            try:
                addc = self.helpers.cper_dump_decode(args.cper_file)
            except OSError as e:
                raise AmdSmiInvalidFilePathException(
                    args.cper_file,
                    self.logger.format,
                    f"Unable to read CPER file '{args.cper_file}': {e}",
                ) from e
            self._emit_decode_records(args.cper_file, addc)
            return

        if not self.group_check_printed:
            self.helpers.check_required_groups()
            self.group_check_printed = True

        if not args.cper:
            return

        if getattr(args, "gpu", None) is None:
            args.gpu = self.device_handles

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

        if partition_warning_flag and not self.logger.is_json_format():
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
            if is_json and (all_json_rows or not args.follow):
                self.logger.multiple_device_output = all_json_rows
                self.logger.print_output(multiple_device_enabled=True, emit_empty=True)
            if not args.follow:
                break
            time.sleep(1)
