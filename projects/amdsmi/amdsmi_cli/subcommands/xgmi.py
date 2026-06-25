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

from amdsmi import amdsmi_exception, amdsmi_interface


class XgmiCommands:
    def xgmi(
        self,
        args,
        multiple_devices=False,
        gpu=None,
        metric=None,
        xgmi_source_status=None,
        xgmi_link_status=None,
    ):
        """Get topology information for target gpus
        params:
            args - argparser args to pass to subcommand
            multiple_devices (bool) - True if checking for multiple devices
            gpu (device_handle) - device_handle for target device
            metric (bool) - Value override for args.metric
            xgmi_source_status (bool) - Value override for args.xgmi_source_status
            xgmi_link_status (bool) - Value override for args.xgmi_link_status

        return:
            Nothing
        """
        # Not supported with partitions

        # Set args.* to passed in arguments
        if gpu:
            args.gpu = gpu
        if metric:
            args.metric = metric
        if xgmi_link_status:
            args.link_status = xgmi_link_status
        if xgmi_source_status:
            args.source_status = xgmi_source_status

        # Handle No GPU passed
        if args.gpu == None:
            args.gpu = self.device_handles

        if not isinstance(args.gpu, list):
            args.gpu = [args.gpu]

        # Handle all args being false
        if not any([args.metric, args.link_status, args.source_status]):
            args.metric = True
            args.link_status = True
            args.source_status = True

        # Clear the table header
        self.logger.table_header = "".rjust(7)

        if not self.group_check_printed:
            self.helpers.check_required_groups()
            self.group_check_printed = True

        (total_socket_count, num_gpu_sockets, num_cpu_sockets) = self.helpers._get_socket_counts()
        logging.debug(
            f"total sockets: {total_socket_count}, gpu sockets: {num_gpu_sockets}, cpu sockets: {num_cpu_sockets}"
        )

        # Populate the possible gpus and their bdfs
        xgmi_values = []
        for gpu in args.gpu:
            primary_partition = self.helpers.is_primary_partition(gpu)
            if not primary_partition:
                logging.debug(f"Skipping xgmi command due to non zero partition {gpu}")
                continue

            logging.debug("check1 device_handle: %s", gpu)
            gpu_id = self.helpers.get_gpu_id_from_device_handle(gpu)
            gpu_bdf = amdsmi_interface.amdsmi_get_gpu_device_bdf(gpu)
            xgmi_values.append({"gpu": gpu_id, "bdf": gpu_bdf})
            # Populate header with just it's gpu_id
            self.logger.table_header += f"GPU{gpu_id}".rjust(13)

        # Cache processor handles for each BDF
        src_gpu_handles = {}
        for dict in xgmi_values:
            try:
                src_gpu_handles[dict["bdf"]] = (
                    amdsmi_interface.amdsmi_get_processor_handle_from_bdf(dict["bdf"])
                )
            except amdsmi_exception.AmdSmiLibraryException as e:
                logging.debug(
                    "Failed to get processor handle for %s | %s", dict["bdf"], e.get_error_info()
                )
                src_gpu_handles[dict["bdf"]] = None
        if args.metric:
            # prepend link metrics header to the table header
            link_metrics_header = (
                "       "
                + "bdf".ljust(14)
                + "bit_rate".ljust(10)
                + "max_bandwidth".ljust(15)
                + "link_type".ljust(11)
            )
            self.logger.table_header = link_metrics_header + self.logger.table_header.strip()

            # Populate dictionary according to format
            for xgmi_dict in xgmi_values:
                src_gpu_id = xgmi_dict["gpu"]
                src_gpu_bdf = xgmi_dict["bdf"]
                src_gpu = src_gpu_handles.get(src_gpu_bdf)
                logging.debug("check2 device_handle: %s", src_gpu)
                # This should be the same order as the check1

                xgmi_dict["link_metrics"] = {
                    "bit_rate": "N/A",
                    "max_bandwidth": "N/A",
                    "link_type": "N/A",
                    "links": [],
                }
                xgmi_metrics_info = {"links": []}

                try:
                    xgmi_metrics_info = amdsmi_interface.amdsmi_get_link_metrics(src_gpu)
                    bitrate = xgmi_metrics_info["links"][0]["bit_rate"]
                    max_bandwidth = xgmi_metrics_info["links"][0]["max_bandwidth"]
                except amdsmi_exception.AmdSmiLibraryException as e:
                    bitrate = "N/A"
                    max_bandwidth = "N/A"
                    logging.debug(
                        "Failed to get bitrate and bandwidth for GPU %s | %s",
                        src_gpu_id,
                        e.get_error_info(),
                    )

                # Populate bitrate and max_bandwidth with units logic
                bw_unit = "Gb/s"
                xgmi_dict["link_metrics"]["bit_rate"] = self.helpers.unit_format(
                    self.logger, bitrate, bw_unit
                )
                xgmi_dict["link_metrics"]["max_bandwidth"] = self.helpers.unit_format(
                    self.logger, max_bandwidth, bw_unit
                )

                # Populate link metrics
                for dest_gpu in args.gpu:
                    primary_partition = self.helpers.is_primary_partition(dest_gpu)
                    if not primary_partition:
                        continue

                    dest_gpu_id = self.helpers.get_gpu_id_from_device_handle(dest_gpu)
                    dest_gpu_bdf = amdsmi_interface.amdsmi_get_gpu_device_bdf(dest_gpu)
                    dest_link_dict = {
                        "gpu": dest_gpu_id,
                        "bdf": dest_gpu_bdf,
                        "read": 0,
                        "write": 0,
                    }

                    found = False
                    for link in xgmi_metrics_info["links"]:
                        if link["bdf"] == dest_gpu_bdf:
                            # Accumulate read/write if multiple links have the same bdf
                            dest_link_dict["read"] += link["read"]
                            dest_link_dict["write"] += link["write"]
                            found = True
                    if not found:
                        dest_link_dict["read"] = "N/A"
                        dest_link_dict["write"] = "N/A"
                    else:
                        data_unit = "KB"
                        if self.logger.is_human_readable_format():
                            dest_link_dict["read"] = self.helpers.convert_bytes_to_readable(
                                dest_link_dict["read"] * 1024, True
                            )
                            dest_link_dict["write"] = self.helpers.convert_bytes_to_readable(
                                dest_link_dict["write"] * 1024, True
                            )
                        else:
                            dest_link_dict["read"] = self.helpers.unit_format(
                                self.logger, dest_link_dict["read"], data_unit
                            )
                            dest_link_dict["write"] = self.helpers.unit_format(
                                self.logger, dest_link_dict["write"], data_unit
                            )

                        try:
                            link_type = amdsmi_interface.amdsmi_topo_get_link_type(
                                src_gpu, dest_gpu
                            )["type"]
                            if xgmi_dict["link_metrics"]["link_type"] != "XGMI" and isinstance(
                                link_type, int
                            ):
                                if (
                                    link_type
                                    == amdsmi_interface.amdsmi_wrapper.AMDSMI_LINK_TYPE_INTERNAL
                                ):
                                    xgmi_dict["link_metrics"]["link_type"] = "UNKNOWN"
                                elif (
                                    link_type
                                    == amdsmi_interface.amdsmi_wrapper.AMDSMI_LINK_TYPE_PCIE
                                ):
                                    xgmi_dict["link_metrics"]["link_type"] = "PCIE"
                                elif (
                                    link_type
                                    == amdsmi_interface.amdsmi_wrapper.AMDSMI_LINK_TYPE_XGMI
                                ):
                                    xgmi_dict["link_metrics"]["link_type"] = "XGMI"
                        except amdsmi_exception.AmdSmiLibraryException as e:
                            logging.debug(
                                "Failed to get link type for %s to %s | %s",
                                self.helpers.get_gpu_id_from_device_handle(src_gpu),
                                self.helpers.get_gpu_id_from_device_handle(dest_gpu),
                                e.get_error_info(),
                            )

                    xgmi_dict["link_metrics"]["links"].append(dest_link_dict)

            # Handle printing for tabular format
            if self.logger.is_human_readable_format():
                # Populate tabular output
                tabular_output = []
                for xgmi_dict in xgmi_values:
                    tabular_output_dict = {}

                    # Create GPU row and add to tabular_output
                    for key, value in xgmi_dict.items():
                        if key == "gpu":
                            tabular_output_dict["gpu#"] = f"GPU{value}"
                        if key == "bdf":
                            tabular_output_dict["bdf"] = value
                        if key == "link_metrics":
                            for link_key, link_value in value.items():
                                if link_key == "bit_rate":
                                    tabular_output_dict["bit_rate"] = link_value
                                if link_key == "max_bandwidth":
                                    tabular_output_dict["max_bandwidth"] = link_value
                                if link_key == "link_type":
                                    tabular_output_dict["link_type"] = link_value
                    tabular_output.append(tabular_output_dict)

                    # Create Read and Write rows and add to tabular_output
                    read_output_dict = {"RW": " Read"}
                    write_output_dict = {"RW": " Write"}
                    for key, value in xgmi_dict.items():
                        if key == "link_metrics":
                            for link_key, link_value in value.items():
                                if link_key == "links":
                                    for link in link_value:
                                        read_output_dict[f"bdf_{link['gpu']}"] = link["read"]
                                        write_output_dict[f"bdf_{link['gpu']}"] = link["write"]
                    tabular_output.append(read_output_dict)
                    tabular_output.append(write_output_dict)

                # Print out the tabular output
                self.logger.multiple_device_output = tabular_output
                self.logger.table_title = "\nLINK METRIC TABLE"
                self.logger.print_output(multiple_device_enabled=True, tabular=True)

            self.logger.multiple_device_output = xgmi_values

            if self.logger.is_csv_format():
                new_output = []
                for elem in self.logger.multiple_device_output:
                    new_output.append(self.logger.flatten_dict(elem, topology_override=True))
                self.logger.multiple_device_output = new_output

            if self.logger.is_json_format():
                self.logger.store_xgmi_metric_json_output.append(xgmi_values)
                if not any([args.link_status, args.source_status]):
                    self.logger.combine_arrays_to_json()
            elif not self.logger.is_human_readable_format():
                self.logger.print_output(multiple_device_enabled=True)

        if args.source_status:
            # Header modification
            self.logger.table_header = "".rjust(7)
            current_header = "     ".ljust(7) + "bdf".ljust(14) + "port_num".ljust(20)
            self.logger.table_header = current_header + self.logger.table_header.strip()
            # Process each GPU
            tabular_output = []
            for xgmi_dict in xgmi_values:
                src_gpu_id = xgmi_dict["gpu"]
                src_gpu_bdf = xgmi_dict["bdf"]
                src_gpu = src_gpu_handles.get(src_gpu_bdf)

                # Populate link statuses
                tabular_output_dict = {
                    "gpu#": f"GPU{src_gpu_id}",
                    "gpu": src_gpu_id,
                    "bdf": src_gpu_bdf,
                    "link_status": "N/A",
                }
                try:
                    link_status = amdsmi_interface.amdsmi_get_gpu_xgmi_link_status(src_gpu)
                    logging.debug(
                        f"GPU(src): {src_gpu_id}, BDF(src): {src_gpu_bdf}, link_status: {link_status}"
                    )
                    tabular_output_dict["link_status"] = link_status["status"]
                except amdsmi_exception.AmdSmiLibraryException as e:
                    logging.debug(
                        "Failed to get XGMI link status for GPU %s | %s",
                        src_gpu_id,
                        e.get_error_info(),
                    )
                    # "N/A" * number of gpu sockets, since we only display in for number of total sockets
                    # These can be CPU or GPU links, so we use total_socket_count
                    tabular_output_dict["link_status"] = ["N/A"] * total_socket_count
                if self.logger.is_human_readable_format():
                    del tabular_output_dict["gpu"]
                else:
                    del tabular_output_dict["gpu#"]
                tabular_output.append(tabular_output_dict)
                if self.logger.is_json_format():
                    self.logger.store_xgmi_source_status_json_output.append(tabular_output_dict)

                # populate link status data for output
                if self.logger.is_human_readable_format():
                    xgmi_dict["link_status"] = tabular_output
            self.logger.multiple_device_output = tabular_output
            self.logger.table_title = "\nGPU LINK PORT STATUS"
            if not self.logger.is_json_format():
                self.logger.print_output(multiple_device_enabled=True, tabular=True)
            self.logger.clear_multiple_devices_output()
            if self.logger.is_json_format():
                if not args.link_status:
                    self.logger.combine_arrays_to_json()

        if args.link_status:
            # XGMI LINK STATUS for src_gpu to dest_gpu
            header = ["       ".ljust(7), "bdf".ljust(14)] + [
                f"GPU{d['gpu']}".ljust(14) for d in xgmi_values
            ]
            self.logger.table_header = "".join(header)
            self.logger.table_title = "\nXGMI LINK STATUS"

            src_link_status_map = {}
            for gpu_dict in xgmi_values:
                src_gpu_id = gpu_dict["gpu"]
                src_gpu_bdf = gpu_dict["bdf"]
                src_gpu = src_gpu_handles.get(src_gpu_bdf)
                try:
                    link_status = amdsmi_interface.amdsmi_get_gpu_xgmi_link_status(src_gpu)
                    src_link_status_map[src_gpu_bdf] = link_status["status"]
                except amdsmi_exception.AmdSmiLibraryException:
                    # "N/A" * number of gpu sockets, since we only display in for number of gpu sockets
                    src_link_status_map[src_gpu_bdf] = ["N/A"] * num_gpu_sockets

            tabular_output = []
            for src_xgmi_dict in xgmi_values:
                src_gpu_id = src_xgmi_dict["gpu"]
                src_gpu_bdf = src_xgmi_dict["bdf"]
                src_gpu = src_gpu_handles.get(src_gpu_bdf)
                try:
                    xgmi_metrics_info = amdsmi_interface.amdsmi_get_link_metrics(src_gpu)
                except amdsmi_exception.AmdSmiLibraryException:
                    xgmi_metrics_info = {"links": []}
                # First column: GPU# + tab + bdf, then status for each dest bdf
                if self.logger.is_human_readable_format():
                    gpu_id_str = f"GPU{src_gpu_id}"
                    row_dict = {"": f"{gpu_id_str.ljust(7)}{src_gpu_bdf.ljust(14)}"}
                else:
                    row_dict = {"gpu": f"GPU{src_gpu_id}", "bdf": src_gpu_bdf}
                json_status = []
                # Cache GPU handles for destination GPUs
                dest_gpu_handles = {
                    dest_xgmi_dict["bdf"]: amdsmi_interface.amdsmi_get_processor_handle_from_bdf(
                        dest_xgmi_dict["bdf"]
                    )
                    for dest_xgmi_dict in xgmi_values
                }
                for dest_xgmi_dict in xgmi_values:
                    dest_gpu_bdf = dest_xgmi_dict["bdf"]
                    dest_gpu = dest_gpu_handles[dest_gpu_bdf]

                    # Find all link indexes in xgmi_metrics_info for this destination
                    link_indexes = []
                    for idx, link in enumerate(xgmi_metrics_info["links"]):
                        if link["bdf"] == dest_gpu_bdf:
                            link_indexes.append(idx)

                    # Use the found link index to get the status if valid
                    if link_indexes and len(link_indexes) <= len(
                        src_link_status_map.get(src_gpu_bdf, [])
                    ):
                        statuses = []
                        for link_idx in link_indexes:
                            if link_idx < len(src_link_status_map[src_gpu_bdf]):
                                status_str = str(src_link_status_map[src_gpu_bdf][link_idx])
                                if status_str != "N/A":
                                    statuses.append(status_str)

                        # Join multiple statuses with "/"
                        if statuses:
                            status = "/".join(statuses)
                        else:
                            status = "N/A"
                    elif dest_gpu_bdf == src_gpu_bdf:
                        status = "SELF"
                    else:
                        status = "N/A"

                    if self.logger.is_human_readable_format():
                        row_dict[dest_gpu_bdf.ljust(14)] = str(status).ljust(14)
                    else:
                        row_dict[dest_gpu_bdf] = status
                    json_status.append(status)
                tabular_output.append(row_dict)
                if self.logger.is_json_format():
                    self.logger.store_xgmi_link_status_json_output.append(
                        {"gpu": src_gpu_id, "bdf": src_gpu_bdf, "link_status": json_status}
                    )

            if not self.logger.is_json_format():
                self.logger.multiple_device_output = tabular_output
                self.logger.print_output(multiple_device_enabled=True, tabular=True)

            self.logger.clear_multiple_devices_output()

            if self.logger.is_json_format():
                self.logger.combine_arrays_to_json()

        if self.logger.is_human_readable_format():
            # Populate the legend output
            legend_parts = [
                "\n\nLegend:",
                "  SELF = Current GPU",
                "  N/A = Not supported",
                "  U / D / X = Link is Up / Down / Disabled",
                "  Read / Write = GPU Metric Accumulated Read / Write",
            ]
            legend_output = "\n".join(legend_parts)

            if self.logger.destination == "stdout":
                print(legend_output)
            else:
                with self.logger.destination.open("a", encoding="utf-8") as output_file:
                    output_file.write(legend_output + "\n")
