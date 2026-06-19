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
import time

from amdsmi import amdsmi_exception, amdsmi_interface


class ProcessCommands:
    def process(
        self,
        args,
        multiple_devices=False,
        watching_output=False,
        gpu=None,
        general=None,
        engine=None,
        pid=None,
        name=None,
        watch=None,
        watch_time=None,
        iterations=None,
    ):
        """Get Process Information from the target GPU

        Args:
            args (Namespace): Namespace containing the parsed CLI args
            multiple_devices (bool, optional): True if checking for multiple devices. Defaults to False.
            watching_output (bool, optional): True if watch argument has been set. Defaults to False.
            gpu (device_handle, optional): device_handle for target device. Defaults to None.
            general (bool, optional): Value override for args.general. Defaults to None.
            engine (bool, optional): Value override for args.engine. Defaults to None.
            pid (Positive int, optional): Value override for args.pid. Defaults to None.
            name (str, optional): Value override for args.name. Defaults to None.
            watch (Positive int, optional): Value override for args.watch. Defaults to None.
            watch_time (Positive int, optional): Value override for args.watch_time. Defaults to None.
            iterations (Positive int, optional): Value override for args.iterations. Defaults to None.

        Raises:
            IndexError: Index error if gpu list is empty

        Returns:
            None: Print output via AMDSMILogger to destination
        """
        # Set args.* to passed in arguments
        if gpu:
            args.gpu = gpu
        if general:
            args.general = general
        if engine:
            args.engine = engine
        if pid:
            args.pid = pid
        if name:
            args.name = name
        if watch:
            args.watch = watch
        if watch_time:
            args.watch_time = watch_time
        if iterations:
            args.iterations = iterations

        # Handle No GPU passed
        if args.gpu == None:
            args.gpu = self.device_handles

        # Handle watch logic, will only enter this block once
        if args.watch:
            self.helpers.handle_watch(args=args, subcommand=self.process, logger=self.logger)
            return

        # Handle --sort-by-pid: group process output by PID across all GPUs
        if getattr(args, "sort_by_pid", False):
            handles = args.gpu if isinstance(args.gpu, list) else [args.gpu]
            self._process_sort_by_pid(args, handles, watching_output)
            return

        # Handle multiple GPUs
        if isinstance(args.gpu, list):
            if len(args.gpu) > 1:
                # Deepcopy gpus as recursion will destroy the gpu list
                stored_gpus = []
                for gpu in args.gpu:
                    stored_gpus.append(gpu)

                # Store output from multiple devices
                for device_handle in args.gpu:
                    self.process(
                        args,
                        multiple_devices=True,
                        watching_output=watching_output,
                        gpu=device_handle,
                    )

                # Reload original gpus
                args.gpu = stored_gpus

                # Print multiple device output
                self.logger.print_output(
                    multiple_device_enabled=True, watching_output=watching_output
                )

                # Add output to total watch output and clear multiple device output
                if watching_output:
                    self.logger.store_watch_output(multiple_device_enabled=True)

                    # Flush the watching output
                    self.logger.print_output(
                        multiple_device_enabled=True, watching_output=watching_output
                    )

                return
            elif len(args.gpu) == 1:
                args.gpu = args.gpu[0]
            else:
                raise IndexError("args.gpu should not be an empty list")

        # Get gpu_id for logging
        gpu_id = self.helpers.get_gpu_id_from_device_handle(args.gpu)

        # Populate initial processes
        try:
            process_list = amdsmi_interface.amdsmi_get_gpu_process_list(args.gpu)
        except amdsmi_exception.AmdSmiLibraryException as e:
            logging.debug("Failed to get process list for gpu %s | %s", gpu_id, e.get_error_info())
            raise e

        filtered_process_values = []
        for process_info in process_list:
            process_info = {
                "name": process_info["name"],
                "pid": process_info["pid"],
                "memory_usage": {
                    "gtt_mem": process_info["memory_usage"]["gtt_mem"],
                    "cpu_mem": process_info["memory_usage"]["cpu_mem"],
                    "vram_mem": process_info["memory_usage"]["vram_mem"],
                },
                "mem_usage": process_info["mem"],
                "usage": {
                    "gfx": process_info["engine_usage"]["gfx"],
                    "enc": process_info["engine_usage"]["enc"],
                },
                "sdma_usage": process_info["sdma_usage"],
                "cu_occupancy": process_info["cu_occupancy"],
                "evicted_time": process_info["evicted_time"],
            }

            engine_usage_unit = "ns"
            memory_usage_unit = "B"
            evicted_time_unit = "ms"
            sdma_usage_unit = "us"

            if self.logger.is_human_readable_format():
                process_info["mem_usage"] = self.helpers.convert_bytes_to_readable(
                    process_info["mem_usage"]
                )
                for usage_metric in process_info["memory_usage"]:
                    process_info["memory_usage"][usage_metric] = (
                        self.helpers.convert_bytes_to_readable(
                            process_info["memory_usage"][usage_metric]
                        )
                    )
                memory_usage_unit = ""

            process_info["mem_usage"] = self.helpers.unit_format(
                self.logger, process_info["mem_usage"], memory_usage_unit
            )

            process_info["evicted_time"] = self.helpers.unit_format(
                self.logger, process_info["evicted_time"], evicted_time_unit
            )

            process_info["sdma_usage"] = self.helpers.unit_format(
                self.logger, process_info["sdma_usage"], sdma_usage_unit
            )

            for usage_metric in process_info["usage"]:
                process_info["usage"][usage_metric] = self.helpers.unit_format(
                    self.logger, process_info["usage"][usage_metric], engine_usage_unit
                )

            for usage_metric in process_info["memory_usage"]:
                process_info["memory_usage"][usage_metric] = self.helpers.unit_format(
                    self.logger, process_info["memory_usage"][usage_metric], memory_usage_unit
                )

            filtered_process_values.append({"process_info": process_info})

        if not filtered_process_values:
            process_info = "N/A"
            logging.debug("Failed to detect any process on gpu %s", gpu_id)
            filtered_process_values.append({"process_info": process_info})

        # Arguments will filter the populated processes
        # General and Engine to expose process_info values
        if args.general or args.engine:
            for process_info in filtered_process_values:
                if not process_info["process_info"] == "N/A":
                    if args.general and args.engine:
                        del process_info["process_info"]["memory_usage"]
                    elif args.general:
                        del process_info["process_info"]["memory_usage"]
                        del process_info["process_info"]["usage"]  # Used in engine
                    elif args.engine:
                        del process_info["process_info"]["memory_usage"]
                        del process_info["process_info"]["mem_usage"]  # Used in general

        # Filter out non specified pids
        if args.pid:
            process_pids = []
            for process_info in filtered_process_values:
                if process_info["process_info"] == "N/A":
                    continue
                pid = str(process_info["process_info"]["pid"])
                if str(args.pid) == pid:
                    process_pids.append(process_info)
            filtered_process_values = process_pids

        # Filter out non specified process names
        if args.name:
            process_names = []
            for process_info in filtered_process_values:
                if process_info["process_info"] == "N/A":
                    continue
                process_name = str(process_info["process_info"]["name"]).lower()
                if str(args.name).lower() == process_name:
                    process_names.append(process_info)
            filtered_process_values = process_names

        # If the name or pid args filter processes out then insert an N/A placeholder
        if not filtered_process_values:
            filtered_process_values.append({"process_info": "N/A"})

        logging.debug(f"Process Info for GPU {gpu_id} | {filtered_process_values}")

        for index, process in enumerate(filtered_process_values):
            if process["process_info"] == "N/A":
                filtered_process_values[index]["process_info"] = "No running processes detected"

        if self.logger.is_json_format():
            if watching_output:
                self.logger.store_output(args.gpu, "timestamp", int(time.time()))
            self.logger.store_output(args.gpu, "process_list", filtered_process_values)

        if self.logger.is_human_readable_format():
            if watching_output:
                self.logger.store_output(args.gpu, "timestamp", int(time.time()))
            # When we print out process_info we remove the index
            # The removal is needed only for human readable process format to align with Host
            for index, process in enumerate(filtered_process_values):
                self.logger.store_output(args.gpu, f"process_info_{index}", process["process_info"])

        multiple_devices_csv_override = False
        if self.logger.is_csv_format():
            multiple_devices_csv_override = True
            for process in filtered_process_values:
                if watching_output:
                    self.logger.store_output(args.gpu, "timestamp", int(time.time()))
                self.logger.store_output(args.gpu, "process_info", process["process_info"])
                self.logger.store_multiple_device_output()

        if multiple_devices:
            self.logger.store_multiple_device_output()
            return  # Skip printing when there are multiple devices

        multiple_devices = multiple_devices or multiple_devices_csv_override
        self.logger.print_output(
            multiple_device_enabled=multiple_devices, watching_output=watching_output
        )

        if watching_output:  # End of single gpu add to watch_output
            self.logger.store_watch_output(multiple_device_enabled=multiple_devices)

    def _process_sort_by_pid(self, args, handles, watching_output):
        """Process output grouped by PID instead of GPU."""
        try:
            pid_list = amdsmi_interface.amdsmi_get_gpu_process_list_by_pid(handles)
        except amdsmi_exception.AmdSmiLibraryException as e:
            logging.debug("Failed to get process list by pid | %s", e.get_error_info())
            raise e

        # Apply --pid filter
        if getattr(args, "pid", None):
            pid_list = [p for p in pid_list if p["pid"] == args.pid]

        # Apply --name filter
        if getattr(args, "name", None):
            pid_list = [p for p in pid_list if p["name"].lower() == str(args.name).lower()]

        engine_usage_unit = "ns"
        memory_usage_unit = "B"
        evicted_time_unit = "ms"
        sdma_usage_unit = "us"

        for proc in pid_list:
            for gpu_entry in proc["gpus"]:
                if self.logger.is_human_readable_format():
                    gpu_entry["mem"] = self.helpers.convert_bytes_to_readable(gpu_entry["mem"])
                    for key in ("gtt_mem", "cpu_mem", "vram_mem"):
                        gpu_entry["memory_usage"][key] = self.helpers.convert_bytes_to_readable(
                            gpu_entry["memory_usage"][key]
                        )

                mem_unit = "" if self.logger.is_human_readable_format() else memory_usage_unit
                gpu_entry["mem"] = self.helpers.unit_format(self.logger, gpu_entry["mem"], mem_unit)
                gpu_entry["evicted_time"] = self.helpers.unit_format(
                    self.logger, gpu_entry["evicted_time"], evicted_time_unit
                )
                gpu_entry["sdma_usage"] = self.helpers.unit_format(
                    self.logger, gpu_entry["sdma_usage"], sdma_usage_unit
                )
                for key in gpu_entry["engine_usage"]:
                    gpu_entry["engine_usage"][key] = self.helpers.unit_format(
                        self.logger, gpu_entry["engine_usage"][key], engine_usage_unit
                    )
                for key in gpu_entry["memory_usage"]:
                    gpu_entry["memory_usage"][key] = self.helpers.unit_format(
                        self.logger, gpu_entry["memory_usage"][key], mem_unit
                    )

        if not pid_list:
            pid_list = [
                {
                    "pid": "N/A",
                    "name": "N/A",
                    "gpus": [],
                    "message": "No running processes detected",
                }
            ]

        if self.logger.is_json_format():
            for proc in pid_list:
                self.logger.output = proc
                self.logger.store_multiple_device_output()
            self.logger.print_output(multiple_device_enabled=True)
            return

        if self.logger.is_human_readable_format():
            lines = []
            for proc in pid_list:
                if proc.get("message"):
                    lines.append(proc["message"])
                    continue

                lines.append(f"PID: {proc['pid']}  NAME: {proc['name']}")
                for gpu_entry in proc["gpus"]:
                    gpu_idx = gpu_entry["gpu_index"]
                    parts = [f"GPU: {gpu_idx}"]
                    if isinstance(gpu_entry["mem"], dict):
                        parts.append(f"MEM: {gpu_entry['mem']['value']} {gpu_entry['mem']['unit']}")
                    else:
                        parts.append(f"MEM: {gpu_entry['mem']}")
                    eng = gpu_entry.get("engine_usage", {})
                    if isinstance(eng.get("gfx"), dict):
                        parts.append(f"GFX: {eng['gfx']['value']} {eng['gfx']['unit']}")
                        parts.append(f"ENC: {eng['enc']['value']} {eng['enc']['unit']}")
                    else:
                        parts.append(f"GFX: {eng.get('gfx', 'N/A')}")
                        parts.append(f"ENC: {eng.get('enc', 'N/A')}")
                    lines.append("    " + "  ".join(parts))
                lines.append("")  # blank line between PIDs

            print("\n".join(lines))

            if watching_output:
                self.logger.store_watch_output(multiple_device_enabled=False)
            return

        if self.logger.is_csv_format():
            for proc in pid_list:
                for gpu_entry in proc["gpus"]:
                    row = {"pid": proc["pid"], "name": proc["name"]}
                    row["gpu_index"] = gpu_entry["gpu_index"]
                    row.update(self.logger.flatten_dict(gpu_entry["memory_usage"]))
                    row.update(self.logger.flatten_dict(gpu_entry["engine_usage"]))
                    row["sdma_usage"] = gpu_entry["sdma_usage"]
                    row["cu_occupancy"] = gpu_entry["cu_occupancy"]
                    row["evicted_time"] = gpu_entry["evicted_time"]
                    self.logger.output = row
                    self.logger.store_multiple_device_output()
            self.logger.print_output(multiple_device_enabled=True, watching_output=watching_output)
