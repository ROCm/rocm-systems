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


class BadPagesCommands:
    def bad_pages(
        self,
        args,
        multiple_devices=False,
        gpu=None,
        retired=None,
        pending=None,
        un_res=None,
        hex_format=None,
    ):
        """Get bad pages information for target gpu

        Args:
            args (Namespace): Namespace containing the parsed CLI args
            multiple_devices (bool, optional): True if checking for multiple devices. Defaults to False.
            gpu (device_handle, optional): device_handle for target device. Defaults to None.
            retired (bool, optional) - Value override for args.retired
            pending (bool, optional) - Value override for args.pending/
            un_res (bool, optional) - Value override for args.un_res
            hex_format (bool, optional) - Value override for args.hex

        Raises:
            IndexError: Index error if gpu list is empty

        Returns:
            None: Print output via AMDSMILogger to destination
        """
        # Set args.* to passed in arguments
        if gpu:
            args.gpu = gpu
        if retired:
            args.retired = retired
        if pending:
            args.pending = pending
        if un_res:
            args.un_res = un_res
        if hex_format is not None:
            args.hex = hex_format

        # Handle No GPU passed
        if args.gpu == None:
            args.gpu = self.device_handles

        # Handle multiple GPUs
        handled_multiple_gpus, device_handle = self.helpers.handle_gpus(
            args, self.logger, self.bad_pages
        )
        if handled_multiple_gpus:
            return  # This function is recursive

        args.gpu = device_handle

        # If all arguments are False, the print all bad_page information
        if not any([args.retired, args.pending, args.un_res]):
            args.retired = args.pending = args.un_res = True

        values_dict = {}

        # Get gpu_id for logging
        gpu_id = self.helpers.get_gpu_id_from_device_handle(args.gpu)

        bad_pages_not_found = "No bad pages found."
        try:
            bad_page_info = amdsmi_interface.amdsmi_get_gpu_bad_page_info(args.gpu)
            # If bad_page_info is an empty list overwrite with not found error statement
            if bad_page_info == []:
                bad_page_info = bad_pages_not_found
                bad_page_error = True
            else:
                bad_page_error = False
        except amdsmi_exception.AmdSmiLibraryException as e:
            bad_page_info = "N/A"
            bad_page_error = True
            logging.debug("Failed to get bad page info for gpu %s | %s", gpu_id, e.get_error_info())

        if args.retired:
            if bad_page_error:
                values_dict["retired"] = bad_page_info
            else:
                bad_page_info_output = []
                for bad_page in bad_page_info:
                    if bad_page["status"] == amdsmi_interface.AmdSmiMemoryPageStatus.RESERVED:
                        bad_page_info_entry = {}
                        # Format page address and size based on --hex flag
                        if args.hex:
                            bad_page_info_entry["page_address"] = f"0x{bad_page['page_address']:x}"
                            bad_page_info_entry["page_size"] = f"0x{bad_page['page_size']:x}"
                        else:
                            bad_page_info_entry["page_address"] = bad_page["page_address"]
                            bad_page_info_entry["page_size"] = bad_page["page_size"]
                        status_string = (
                            amdsmi_interface.amdsmi_wrapper.amdsmi_memory_page_status_t__enumvalues[
                                bad_page["status"]
                            ]
                        )
                        bad_page_info_entry["status"] = status_string.replace(
                            "AMDSMI_MEM_PAGE_STATUS_", ""
                        )
                        bad_page_info_output.append(bad_page_info_entry)
                # Remove brackets if there is only one value
                if len(bad_page_info_output) == 1:
                    bad_page_info_output = bad_page_info_output[0]

                if bad_page_info_output == []:
                    values_dict["retired"] = bad_pages_not_found
                else:
                    values_dict["retired"] = bad_page_info_output

        if args.pending:
            if bad_page_error:
                values_dict["pending"] = bad_page_info
            else:
                bad_page_info_output = []
                for bad_page in bad_page_info:
                    if bad_page["status"] == amdsmi_interface.AmdSmiMemoryPageStatus.PENDING:
                        bad_page_info_entry = {}
                        # Format page address and size based on --hex flag
                        if args.hex:
                            bad_page_info_entry["page_address"] = f"0x{bad_page['page_address']:x}"
                            bad_page_info_entry["page_size"] = f"0x{bad_page['page_size']:x}"
                        else:
                            bad_page_info_entry["page_address"] = bad_page["page_address"]
                            bad_page_info_entry["page_size"] = bad_page["page_size"]
                        status_string = (
                            amdsmi_interface.amdsmi_wrapper.amdsmi_memory_page_status_t__enumvalues[
                                bad_page["status"]
                            ]
                        )
                        bad_page_info_entry["status"] = status_string.replace(
                            "AMDSMI_MEM_PAGE_STATUS_", ""
                        )
                        bad_page_info_output.append(bad_page_info_entry)
                # Remove brackets if there is only one value
                if len(bad_page_info_output) == 1:
                    bad_page_info_output = bad_page_info_output[0]

                if bad_page_info_output == []:
                    values_dict["pending"] = bad_pages_not_found
                else:
                    values_dict["pending"] = bad_page_info_output

        if args.un_res:
            if bad_page_error:
                values_dict["un_res"] = bad_page_info
            else:
                bad_page_info_output = []
                for bad_page in bad_page_info:
                    if bad_page["status"] == amdsmi_interface.AmdSmiMemoryPageStatus.UNRESERVABLE:
                        bad_page_info_entry = {}
                        # Format page address and size based on --hex flag
                        if hasattr(args, "hex") and args.hex:
                            bad_page_info_entry["page_address"] = f"0x{bad_page['page_address']:x}"
                            bad_page_info_entry["page_size"] = f"0x{bad_page['page_size']:x}"
                        else:
                            bad_page_info_entry["page_address"] = bad_page["page_address"]
                            bad_page_info_entry["page_size"] = bad_page["page_size"]
                        status_string = (
                            amdsmi_interface.amdsmi_wrapper.amdsmi_memory_page_status_t__enumvalues[
                                bad_page["status"]
                            ]
                        )
                        bad_page_info_entry["status"] = status_string.replace(
                            "AMDSMI_MEM_PAGE_STATUS_", ""
                        )
                        bad_page_info_output.append(bad_page_info_entry)
                # Remove brackets if there is only one value
                if len(bad_page_info_output) == 1:
                    bad_page_info_output = bad_page_info_output[0]

                if bad_page_info_output == []:
                    values_dict["un_res"] = bad_pages_not_found
                else:
                    values_dict["un_res"] = bad_page_info_output

        # Store values in logger.output
        self.logger.store_output(args.gpu, "values", values_dict)

        if multiple_devices:
            self.logger.store_multiple_device_output()
            return  # Skip printing when there are multiple devices

        self.logger.print_output()
