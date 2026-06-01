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


class NodeCommands:
    def node(
        self,
        args,
        multiple_devices=False,
        nodes=None,
        power_management=None,
        base_board_temps=None,
        gtt=None,
    ):
        """List node information

        Args:
            args (Namespace): Namespace containing the parsed CLI args
            multiple_devices (bool, optional): True if checking for multiple devices.
                Defaults to False.
            nodes (node_handle, optional): node_handle for target node. Defaults to None.
            power_management (bool, optional): Value override for args.power_management. Defaults to None.
            base_board_temps (bool, optional): Value override for args.base_board_temps. Defaults to None.
            gtt (bool, optional): Value override for args.gtt. Defaults to None.

        Returns:
            None: Print output via AMDSMILogger to destination
        """
        # Set args.* to passed in arguments
        if nodes:
            args.nodes = nodes
        if gtt:
            args.gtt = gtt
        # Store args that are applicable to the current platform
        current_platform_args = ["power_management", "base_board_temps", "gtt"]

        # Check if any node-specific options were passed via command line
        current_platform_values = []
        if args.power_management:
            current_platform_values += [args.power_management]
        if args.base_board_temps:
            current_platform_values += [args.base_board_temps]
        if args.gtt:
            current_platform_values += [args.gtt]

        # If no node options are passed, enable all by default
        if not any(current_platform_values):
            for arg in current_platform_args:
                setattr(args, arg, True)
        if getattr(args, "nodes", None) is None:
            args.nodes = self.node_handle

        if not self.group_check_printed:
            self.helpers.check_required_groups()
            self.group_check_printed = True

        # Initialize variables for both power management and base board temps
        npm_dict = {"limit": "N/A", "status": "N/A", "threshold": "N/A"}
        power_unit = "W"
        limit = "N/A"
        base_board_temp_dict = {}
        gtt_dict = {}

        # Get NPM info
        if args.power_management:
            if args.nodes is not None:
                try:
                    npm_info = amdsmi_interface.amdsmi_get_npm_info(args.nodes)
                except amdsmi_exception.AmdSmiLibraryException as e:
                    logging.debug("amdsmi_get_npm_info failed: %s", e.get_error_info())
                    npm_info = "N/A"
            else:
                logging.debug("No node handle available to query NPM info")
                npm_info = "N/A"

            if isinstance(npm_info, dict):
                limit = npm_info.get("limit", "N/A")
                status = npm_info.get("status", npm_info.get("current", "N/A"))
                ubb_power_threshold = npm_info.get("ubb_power_threshold", "N/A")

                if limit != "N/A":
                    npm_dict["limit"] = limit
                status = (
                    "DISABLED"
                    if status == amdsmi_interface.amdsmi_wrapper.AMDSMI_NPM_STATUS_DISABLED
                    else "ENABLED"
                )
                npm_dict.update({"status": status})
                # Add UBB power threshold if available
                if ubb_power_threshold != "N/A":
                    npm_dict["threshold"] = ubb_power_threshold

        # Get base board temperatures using node_handle
        if args.base_board_temps:
            if args.nodes is not None:
                try:
                    # Get device_handle for OAM_ID 0
                    device_handle = self.helpers.get_oam_0_device_handle()
                    gpu_id = self.helpers.get_gpu_id_from_device_handle(device_handle)
                    base_board_temp_dict = self.helpers.get_base_board_temperatures(
                        device_handle, gpu_id, self.logger
                    )
                except amdsmi_exception.AmdSmiLibraryException as e:
                    logging.debug("Failed to get device handle from node: %s", e.get_error_info())
                    base_board_temp_dict = {}

        # Get GTT (shared GPU memory) information
        if args.gtt:
            try:
                ttm_info = amdsmi_interface.amdsmi_get_ttm_info()
                logging.debug(f"TTM info: {ttm_info}")

                gtt_pages = ttm_info.get("current_pages", 0)
                gtt_gb = self.helpers.pages_to_gb(gtt_pages)

                gtt_dict = {"size_gb": gtt_gb, "size_pages": gtt_pages}
            except amdsmi_exception.AmdSmiLibraryException as e:
                logging.debug("Failed to get GTT info | %s", e.get_error_info())
                gtt_dict = {}

        # Print output
        if self.logger.is_human_readable_format() and self.logger.destination == "stdout":
            node_output = ["NODE:"]
            if args.power_management:
                node_output.append("    POWER_MANAGEMENT:")
                node_output.append(f"        LIMIT: {npm_dict.get('limit', 'N/A')} {power_unit}")
                node_output.append(f"        STATUS: {npm_dict.get('status', 'N/A')}")
                threshold = npm_dict.get("threshold", "N/A")
                node_output.append(f"        THRESHOLD: {threshold} {power_unit}")
            if args.base_board_temps and base_board_temp_dict:
                node_output.append("    BASEBOARD:")
                node_output.append("        TEMPERATURE:")
                for temp_name, temp_value in base_board_temp_dict.items():
                    node_output.append(f"            {temp_name.upper()}: {temp_value}")
            if args.gtt and gtt_dict:
                gtt_gb = gtt_dict.get("size_gb", 0)
                gtt_pages = gtt_dict.get("size_pages", 0)
                node_output.append("    GTT:")
                node_output.append(f"        SIZE: {gtt_gb:.2f} GB ({gtt_pages} pages)")
            print("\n".join(node_output))
        else:
            if self.logger.is_csv_format():
                csv_dict = {}
                if args.power_management:
                    csv_dict["limit"] = npm_dict.get("limit", "N/A")
                    csv_dict["status"] = npm_dict.get("status", "N/A")
                    csv_dict["threshold"] = npm_dict.get("threshold", "N/A")
                if args.base_board_temps and base_board_temp_dict:
                    csv_dict.update(base_board_temp_dict)
                if args.gtt and gtt_dict:
                    csv_dict["gtt_gb"] = gtt_dict.get("size_gb", "N/A")
                    csv_dict["gtt_pages"] = gtt_dict.get("size_pages", "N/A")
                self.logger.output = csv_dict
            else:
                # For JSON and human readable format with file output
                node_output = {}
                if args.power_management:
                    npm_dict["limit"] = self.helpers.unit_format(self.logger, limit, power_unit)
                    threshold = npm_dict.get("threshold", "N/A")
                    if threshold != "N/A":
                        npm_dict["threshold"] = self.helpers.unit_format(
                            self.logger, threshold, power_unit
                        )
                    node_output["power_management"] = npm_dict
                if args.base_board_temps and base_board_temp_dict:
                    node_output["base_board"] = {"temperature": base_board_temp_dict}
                if args.gtt and gtt_dict:
                    node_output["gtt"] = gtt_dict
                self.logger.output = {"node": node_output}
                if multiple_devices:
                    self.logger.store_multiple_device_output()
                    return
            self.logger.print_output()
