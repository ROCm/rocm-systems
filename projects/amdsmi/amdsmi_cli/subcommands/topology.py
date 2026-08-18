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


class TopologyCommands:
    def topology(
        self,
        args,
        multiple_devices=False,
        gpu=None,
        access=None,
        weight=None,
        hops=None,
        link_type=None,
        numa_bw=None,
        coherent=None,
        atomics=None,
        dma=None,
        bi_dir=None,
    ):
        """Get topology information for target gpus
        params:
            args - argparser args to pass to subcommand
            multiple_devices (bool) - True if checking for multiple devices
            gpu (device_handle) - device_handle for target device
            access (bool) - Value override for args.access
            weight (bool) - Value override for args.weight
            hops (bool) - Value override for args.hops
            type (bool) - Value override for args.type
            numa_bw (bool) - Value override for args.numa_bw
            coherent (bool) - Value override for args.coherent
            atomics (bool) - Value override for args.atomics
            dma (bool) - Value override for args.dma
            bi_dir (bool) - Value override for args.bi_dir
        return:
            Nothing
        """
        # Set args.* to passed in arguments
        if gpu:
            args.gpu = gpu
        if access:
            args.access = access
        if weight:
            args.weight = weight
        if hops:
            args.hops = hops
        if link_type:
            args.link_type = link_type
        if numa_bw:
            args.numa_bw = numa_bw
        if coherent:
            args.coherent = coherent
        if atomics:
            args.atomics = atomics
        if dma:
            args.dma = dma
        if bi_dir:
            args.bi_dir = bi_dir

        # Handle No GPU passed
        if args.gpu == None:
            args.gpu = self.device_handles

        if not isinstance(args.gpu, list):
            args.gpu = [args.gpu]

        # Handle all args being false
        if not any(
            [
                args.access,
                args.weight,
                args.hops,
                args.link_type,
                args.numa_bw,
                args.coherent,
                args.atomics,
                args.dma,
                args.bi_dir,
            ]
        ):
            args.access = args.weight = args.hops = args.link_type = args.numa_bw = (
                args.coherent
            ) = args.atomics = args.dma = args.bi_dir = True

        # Clear the table header
        self.logger.table_header = "".rjust(12)

        if not self.group_check_printed:
            self.helpers.check_required_groups()
            self.group_check_printed = True

        p2p_status_cache = {}

        def get_cached_p2p_status(src_gpu, dest_gpu):
            # Get P2P status with caching to avoid duplicate calls
            src_gpu_id = self.helpers.get_gpu_id_from_device_handle(src_gpu)
            dest_gpu_id = self.helpers.get_gpu_id_from_device_handle(dest_gpu)
            key = (src_gpu_id, dest_gpu_id)

            if key not in p2p_status_cache:
                try:
                    if src_gpu == dest_gpu:
                        p2p_status_cache[key] = {
                            "cap": {
                                "is_iolink_coherent": -1,
                                "is_iolink_atomics_32bit": -1,
                                "is_iolink_atomics_64bit": -1,
                                "is_iolink_dma": -1,
                                "is_iolink_bi_directional": -1,
                            }
                        }
                    else:
                        p2p_status_cache[key] = amdsmi_interface.amdsmi_topo_get_p2p_status(
                            src_gpu, dest_gpu
                        )
                except amdsmi_exception.AmdSmiLibraryException as e:
                    logging.debug(
                        "Failed to get link status for %s to %s | %s",
                        src_gpu_id,
                        dest_gpu_id,
                        e.get_error_info(),
                    )
                    p2p_status_cache[key] = {
                        "cap": {
                            "is_iolink_coherent": -1,
                            "is_iolink_atomics_32bit": -1,
                            "is_iolink_atomics_64bit": -1,
                            "is_iolink_dma": -1,
                            "is_iolink_bi_directional": -1,
                        }
                    }

            return p2p_status_cache[key]

        # Populate the possible gpus
        topo_values = []
        for src_gpu_index, src_gpu in enumerate(args.gpu):
            src_gpu_id = self.helpers.get_gpu_id_from_device_handle(src_gpu)
            topo_values.append({"gpu": src_gpu_id})
            src_gpu_bdf = amdsmi_interface.amdsmi_get_gpu_device_bdf(src_gpu)
            topo_values[src_gpu_index]["bdf"] = src_gpu_bdf
            self.logger.table_header += src_gpu_bdf.rjust(13)

            if not self.logger.is_json_format():
                continue  # below is for JSON format only

            ##########################
            # JSON formatting start  #
            ##########################
            links = []
            # create json obj for data alignment
            #  dest_gpu_links = {
            #         "gpu": GPU #
            #         "bdf": BDF identification
            #         "weight": 0 - self (current node); weight >= 0 correlated with hops (GPU-CPU, GPU-GPU, GPU-CPU-CPU-GPU, etc..)
            #         "link_status": "ENABLED" - devices linked; "DISABLED" - devices not linked; Correlated to access
            #         "link_type": "SELF" - current node, "PCIE", "XGMI", "N/A" - no link,"UNKNOWN" - unidentified link type
            #         "num_hops": num_hops - # of hops between devices
            #         "bandwidth": numa_bw - The NUMA "minimum bandwidth-maximum bandwidth" between src and dest nodes
            #                      "N/A" - self node or not connected devices
            #         "coherent": coherent - Coherent / Non-Coherent io links
            #         "atomics": atomics - 32 and 64-bit atomic io link capability between nodes
            #         "dma": dma - P2P direct memory access (DMA) link capability between nodes
            #         "bi_dir": bi_dir - P2P bi-directional link capability between nodes
            #     }

            for dest_gpu_index, dest_gpu in enumerate(args.gpu):
                link_type = "SELF"
                if src_gpu != dest_gpu:
                    link_type = amdsmi_interface.amdsmi_topo_get_link_type(src_gpu, dest_gpu)[
                        "type"
                    ]
                if isinstance(link_type, int):
                    if link_type == amdsmi_interface.amdsmi_wrapper.AMDSMI_LINK_TYPE_INTERNAL:
                        link_type = "UNKNOWN"
                    elif link_type == amdsmi_interface.amdsmi_wrapper.AMDSMI_LINK_TYPE_PCIE:
                        link_type = "PCIE"
                    elif link_type == amdsmi_interface.amdsmi_wrapper.AMDSMI_LINK_TYPE_XGMI:
                        link_type = "XGMI"
                    else:
                        link_type = "N/A"

                numa_bw = "N/A"
                if src_gpu != dest_gpu:
                    try:
                        bw_dict = amdsmi_interface.amdsmi_get_minmax_bandwidth_between_processors(
                            src_gpu, dest_gpu
                        )
                        numa_bw = f"{bw_dict['min_bandwidth']}-{bw_dict['max_bandwidth']}"
                    except amdsmi_exception.AmdSmiLibraryException as e:
                        logging.debug(
                            "Failed to get min max bandwidth for %s to %s | %s",
                            self.helpers.get_gpu_id_from_device_handle(src_gpu),
                            self.helpers.get_gpu_id_from_device_handle(dest_gpu),
                            e.get_error_info(),
                        )

                weight = 0
                num_hops = 0
                if src_gpu != dest_gpu:
                    weight = amdsmi_interface.amdsmi_topo_get_link_weight(src_gpu, dest_gpu)
                    num_hops = amdsmi_interface.amdsmi_topo_get_link_type(src_gpu, dest_gpu)["hops"]
                link_status = amdsmi_interface.amdsmi_is_P2P_accessible(src_gpu, dest_gpu)
                if link_status:
                    link_status = "ENABLED"
                else:
                    link_status = "DISABLED"

                link_coherent = "SELF"
                link_atomics = "SELF"
                link_dma = "SELF"
                link_bi_dir = "SELF"

                if src_gpu != dest_gpu:
                    try:
                        cap = get_cached_p2p_status(src_gpu, dest_gpu)["cap"]
                        link_coherent = (
                            "C"
                            if cap["is_iolink_coherent"] == 1
                            else "NC"
                            if cap["is_iolink_coherent"] == 0
                            else "N/A"
                        )
                        link_atomics = (
                            "64,32"
                            if cap["is_iolink_atomics_32bit"] == 1
                            and cap["is_iolink_atomics_64bit"] == 1
                            else "32"
                            if cap["is_iolink_atomics_32bit"] == 1
                            else "64"
                            if cap["is_iolink_atomics_64bit"] == 1
                            else "N/A"
                        )
                        link_dma = (
                            "T"
                            if cap["is_iolink_dma"] == 1
                            else "F"
                            if cap["is_iolink_dma"] == 0
                            else "N/A"
                        )
                        link_bi_dir = (
                            "T"
                            if cap["is_iolink_bi_directional"] == 1
                            else "F"
                            if cap["is_iolink_bi_directional"] == 0
                            else "N/A"
                        )
                    except amdsmi_exception.AmdSmiLibraryException as e:
                        logging.debug(
                            "Failed to get link status for %s to %s | %s",
                            self.helpers.get_gpu_id_from_device_handle(src_gpu),
                            self.helpers.get_gpu_id_from_device_handle(dest_gpu),
                            e.get_error_info(),
                        )

                # link_status = amdsmi_is_P2P_accessible(src,dest)
                dest_gpu_links = {
                    "gpu": self.helpers.get_gpu_id_from_device_handle(dest_gpu),
                    "bdf": amdsmi_interface.amdsmi_get_gpu_device_bdf(dest_gpu),
                    "weight": weight,
                    "link_status": link_status,
                    "link_type": link_type,
                    "num_hops": num_hops,
                    "bandwidth": numa_bw,
                    "coherent": link_coherent,
                    "atomics": link_atomics,
                    "dma": link_dma,
                    "bi_dir": link_bi_dir,
                }
                if not args.access:
                    del dest_gpu_links["link_status"]
                if not args.weight:
                    del dest_gpu_links["weight"]
                if not args.link_type:
                    del dest_gpu_links["link_type"]
                if not args.hops:
                    del dest_gpu_links["num_hops"]
                if not args.numa_bw:
                    del dest_gpu_links["bandwidth"]
                if not args.coherent:
                    del dest_gpu_links["coherent"]
                if not args.atomics:
                    del dest_gpu_links["atomics"]
                if not args.dma:
                    del dest_gpu_links["dma"]
                if not args.bi_dir:
                    del dest_gpu_links["bi_dir"]
                links.append(dest_gpu_links)
                dest_end = dest_gpu_index + 1 == len(args.gpu)
                isEndOfSrc = src_gpu_index + 1 == len(args.gpu)
                if dest_end:
                    topo_values[src_gpu_index]["links"] = links
                    continue
            if isEndOfSrc:
                self.logger.multiple_device_output = topo_values
                self.logger.print_output(multiple_device_enabled=True, tabular=True)
                return
            ##########################
            # JSON formatting end    #
            ##########################

        if args.access:
            tabular_output = []
            for src_gpu_index, src_gpu in enumerate(args.gpu):
                src_gpu_bdf = amdsmi_interface.amdsmi_get_gpu_device_bdf(src_gpu)
                if self.logger.is_human_readable_format():
                    tabular_output_dict = {"gpu": f"{src_gpu_bdf} "}
                else:
                    tabular_output_dict = {"gpu": src_gpu_bdf}
                src_gpu_links = {}
                for dest_gpu in args.gpu:
                    dest_gpu_id = self.helpers.get_gpu_id_from_device_handle(dest_gpu)
                    dest_gpu_key = f"gpu_{dest_gpu_id}"

                    try:
                        dest_gpu_link_status = amdsmi_interface.amdsmi_is_P2P_accessible(
                            src_gpu, dest_gpu
                        )
                        if dest_gpu_link_status:
                            src_gpu_links[dest_gpu_key] = "ENABLED"
                        else:
                            src_gpu_links[dest_gpu_key] = "DISABLED"
                    except amdsmi_exception.AmdSmiLibraryException as e:
                        src_gpu_links[dest_gpu_key] = "N/A"
                        logging.debug(
                            "Failed to get link status for %s to %s | %s",
                            self.helpers.get_gpu_id_from_device_handle(src_gpu),
                            self.helpers.get_gpu_id_from_device_handle(dest_gpu),
                            e.get_error_info(),
                        )

                topo_values[src_gpu_index]["link_accessibility"] = src_gpu_links

                tabular_output_dict.update(src_gpu_links)
                tabular_output.append(tabular_output_dict)

            if self.logger.is_human_readable_format():
                self.logger.multiple_device_output = tabular_output
                self.logger.table_title = "ACCESS TABLE"
                self.logger.print_output(multiple_device_enabled=True, tabular=True)

        if args.weight:
            tabular_output = []
            for src_gpu_index, src_gpu in enumerate(args.gpu):
                src_gpu_bdf = amdsmi_interface.amdsmi_get_gpu_device_bdf(src_gpu)
                if self.logger.is_human_readable_format():
                    tabular_output_dict = {"gpu": f"{src_gpu_bdf} "}
                else:
                    tabular_output_dict = {"gpu": src_gpu_bdf}
                src_gpu_weight = {}
                for dest_gpu in args.gpu:
                    dest_gpu_id = self.helpers.get_gpu_id_from_device_handle(dest_gpu)
                    dest_gpu_key = f"gpu_{dest_gpu_id}"

                    if src_gpu == dest_gpu:
                        src_gpu_weight[dest_gpu_key] = 0
                        continue

                    try:
                        dest_gpu_link_weight = amdsmi_interface.amdsmi_topo_get_link_weight(
                            src_gpu, dest_gpu
                        )
                        src_gpu_weight[dest_gpu_key] = dest_gpu_link_weight
                    except amdsmi_exception.AmdSmiLibraryException as e:
                        src_gpu_weight[dest_gpu_key] = "N/A"
                        logging.debug(
                            "Failed to get link weight for %s to %s | %s",
                            self.helpers.get_gpu_id_from_device_handle(src_gpu),
                            self.helpers.get_gpu_id_from_device_handle(dest_gpu),
                            e.get_error_info(),
                        )

                topo_values[src_gpu_index]["weight"] = src_gpu_weight

                tabular_output_dict.update(src_gpu_weight)
                tabular_output.append(tabular_output_dict)

            if self.logger.is_human_readable_format():
                self.logger.multiple_device_output = tabular_output
                self.logger.table_title = "WEIGHT TABLE"
                self.logger.print_output(multiple_device_enabled=True, tabular=True)

        if args.hops:
            tabular_output = []
            for src_gpu_index, src_gpu in enumerate(args.gpu):
                src_gpu_bdf = amdsmi_interface.amdsmi_get_gpu_device_bdf(src_gpu)
                if self.logger.is_human_readable_format():
                    tabular_output_dict = {"gpu": f"{src_gpu_bdf} "}
                else:
                    tabular_output_dict = {"gpu": src_gpu_bdf}
                src_gpu_hops = {}
                for dest_gpu in args.gpu:
                    dest_gpu_id = self.helpers.get_gpu_id_from_device_handle(dest_gpu)
                    dest_gpu_key = f"gpu_{dest_gpu_id}"

                    if src_gpu == dest_gpu:
                        src_gpu_hops[dest_gpu_key] = 0
                        continue

                    try:
                        dest_gpu_hops = amdsmi_interface.amdsmi_topo_get_link_type(
                            src_gpu, dest_gpu
                        )["hops"]
                        src_gpu_hops[dest_gpu_key] = dest_gpu_hops
                    except amdsmi_exception.AmdSmiLibraryException as e:
                        src_gpu_hops[dest_gpu_key] = "N/A"
                        logging.debug(
                            "Failed to get link hops for %s to %s | %s",
                            self.helpers.get_gpu_id_from_device_handle(src_gpu),
                            self.helpers.get_gpu_id_from_device_handle(dest_gpu),
                            e.get_error_info(),
                        )

                topo_values[src_gpu_index]["hops"] = src_gpu_hops

                tabular_output_dict.update(src_gpu_hops)
                tabular_output.append(tabular_output_dict)

            if self.logger.is_human_readable_format():
                self.logger.multiple_device_output = tabular_output
                self.logger.table_title = "HOPS TABLE"
                self.logger.print_output(multiple_device_enabled=True, tabular=True)

        if args.link_type:
            tabular_output = []
            for src_gpu_index, src_gpu in enumerate(args.gpu):
                src_gpu_bdf = amdsmi_interface.amdsmi_get_gpu_device_bdf(src_gpu)
                if self.logger.is_human_readable_format():
                    tabular_output_dict = {"gpu": f"{src_gpu_bdf} "}
                else:
                    tabular_output_dict = {"gpu": src_gpu_bdf}
                src_gpu_link_type = {}
                for dest_gpu in args.gpu:
                    dest_gpu_id = self.helpers.get_gpu_id_from_device_handle(dest_gpu)
                    dest_gpu_key = f"gpu_{dest_gpu_id}"

                    if src_gpu == dest_gpu:
                        src_gpu_link_type[dest_gpu_key] = "SELF"
                        continue
                    try:
                        link_type = amdsmi_interface.amdsmi_topo_get_link_type(src_gpu, dest_gpu)[
                            "type"
                        ]
                        if isinstance(link_type, int):
                            if (
                                link_type
                                == amdsmi_interface.amdsmi_wrapper.AMDSMI_LINK_TYPE_INTERNAL
                            ):
                                src_gpu_link_type[dest_gpu_key] = "UNKNOWN"
                            elif link_type == amdsmi_interface.amdsmi_wrapper.AMDSMI_LINK_TYPE_PCIE:
                                src_gpu_link_type[dest_gpu_key] = "PCIE"
                            elif link_type == amdsmi_interface.amdsmi_wrapper.AMDSMI_LINK_TYPE_XGMI:
                                src_gpu_link_type[dest_gpu_key] = "XGMI"
                    except amdsmi_exception.AmdSmiLibraryException as e:
                        src_gpu_link_type[dest_gpu_key] = "N/A"
                        logging.debug(
                            "Failed to get link type for %s to %s | %s",
                            self.helpers.get_gpu_id_from_device_handle(src_gpu),
                            self.helpers.get_gpu_id_from_device_handle(dest_gpu),
                            e.get_error_info(),
                        )

                topo_values[src_gpu_index]["link_type"] = src_gpu_link_type

                tabular_output_dict.update(src_gpu_link_type)
                tabular_output.append(tabular_output_dict)

            if self.logger.is_human_readable_format():
                self.logger.multiple_device_output = tabular_output
                self.logger.table_title = "LINK TYPE TABLE"
                self.logger.print_output(multiple_device_enabled=True, tabular=True)

        if args.numa_bw:
            tabular_output = []
            for src_gpu_index, src_gpu in enumerate(args.gpu):
                src_gpu_bdf = amdsmi_interface.amdsmi_get_gpu_device_bdf(src_gpu)
                if self.logger.is_human_readable_format():
                    tabular_output_dict = {"gpu": f"{src_gpu_bdf} "}
                else:
                    tabular_output_dict = {"gpu": src_gpu_bdf}
                src_gpu_link_type = {}
                for dest_gpu in args.gpu:
                    dest_gpu_id = self.helpers.get_gpu_id_from_device_handle(dest_gpu)
                    dest_gpu_key = f"gpu_{dest_gpu_id}"

                    if src_gpu == dest_gpu:
                        src_gpu_link_type[dest_gpu_key] = "N/A"
                        continue

                    try:
                        link_type = amdsmi_interface.amdsmi_topo_get_link_type(src_gpu, dest_gpu)[
                            "type"
                        ]
                        if isinstance(link_type, int):
                            if link_type != amdsmi_interface.amdsmi_wrapper.AMDSMI_LINK_TYPE_XGMI:
                                # non_xgmi = True
                                src_gpu_link_type[dest_gpu_key] = "N/A"
                                continue
                    except amdsmi_exception.AmdSmiLibraryException as e:
                        src_gpu_link_type[dest_gpu_key] = "N/A"
                        logging.debug(
                            "Failed to get link type for %s to %s | %s",
                            self.helpers.get_gpu_id_from_device_handle(src_gpu),
                            self.helpers.get_gpu_id_from_device_handle(dest_gpu),
                            e.get_error_info(),
                        )

                    try:
                        bw_dict = amdsmi_interface.amdsmi_get_minmax_bandwidth_between_processors(
                            src_gpu, dest_gpu
                        )
                        src_gpu_link_type[dest_gpu_key] = (
                            f"{bw_dict['min_bandwidth']}-{bw_dict['max_bandwidth']}"
                        )
                    except amdsmi_exception.AmdSmiLibraryException as e:
                        src_gpu_link_type[dest_gpu_key] = e.get_error_info()
                        logging.debug(
                            "Failed to get min max bandwidth for %s to %s | %s",
                            self.helpers.get_gpu_id_from_device_handle(src_gpu),
                            self.helpers.get_gpu_id_from_device_handle(dest_gpu),
                            e.get_error_info(),
                        )

                topo_values[src_gpu_index]["numa_bandwidth"] = src_gpu_link_type

                tabular_output_dict.update(src_gpu_link_type)
                tabular_output.append(tabular_output_dict)

            if self.logger.is_human_readable_format():
                self.logger.multiple_device_output = tabular_output
                self.logger.table_title = "NUMA BW TABLE"
                self.logger.print_output(multiple_device_enabled=True, tabular=True)

        if args.coherent:
            tabular_output = []
            for src_gpu_index, src_gpu in enumerate(args.gpu):
                src_gpu_bdf = amdsmi_interface.amdsmi_get_gpu_device_bdf(src_gpu)
                if self.logger.is_human_readable_format():
                    tabular_output_dict = {"gpu": f"{src_gpu_bdf} "}
                else:
                    tabular_output_dict = {"gpu": src_gpu_bdf}
                src_gpu_coherent = {}
                for dest_gpu in args.gpu:
                    dest_gpu_id = self.helpers.get_gpu_id_from_device_handle(dest_gpu)
                    dest_gpu_key = f"gpu_{dest_gpu_id}"

                    if src_gpu == dest_gpu:
                        src_gpu_coherent[dest_gpu_key] = "SELF"
                        continue
                    try:
                        iolink_coherent = get_cached_p2p_status(src_gpu, dest_gpu)["cap"][
                            "is_iolink_coherent"
                        ]
                        src_gpu_coherent[dest_gpu_key] = (
                            "C" if iolink_coherent == 1 else "NC" if iolink_coherent == 0 else "N/A"
                        )
                    except amdsmi_exception.AmdSmiLibraryException as e:
                        src_gpu_coherent[dest_gpu_key] = "N/A"
                        logging.debug(
                            "Failed to get link coherent for %s to %s | %s",
                            self.helpers.get_gpu_id_from_device_handle(src_gpu),
                            self.helpers.get_gpu_id_from_device_handle(dest_gpu),
                            e.get_error_info(),
                        )

                topo_values[src_gpu_index]["coherent"] = src_gpu_coherent

                tabular_output_dict.update(src_gpu_coherent)
                tabular_output.append(tabular_output_dict)

            if self.logger.is_human_readable_format():
                self.logger.multiple_device_output = tabular_output
                self.logger.table_title = "CACHE COHERENCY TABLE"
                self.logger.print_output(multiple_device_enabled=True, tabular=True)

        if args.atomics:
            tabular_output = []
            for src_gpu_index, src_gpu in enumerate(args.gpu):
                src_gpu_bdf = amdsmi_interface.amdsmi_get_gpu_device_bdf(src_gpu)
                if self.logger.is_human_readable_format():
                    tabular_output_dict = {"gpu": f"{src_gpu_bdf} "}
                else:
                    tabular_output_dict = {"gpu": src_gpu_bdf}
                src_gpu_atomics = {}
                for dest_gpu in args.gpu:
                    dest_gpu_id = self.helpers.get_gpu_id_from_device_handle(dest_gpu)
                    dest_gpu_key = f"gpu_{dest_gpu_id}"

                    if src_gpu == dest_gpu:
                        src_gpu_atomics[dest_gpu_key] = "SELF"
                        continue
                    try:
                        cap = get_cached_p2p_status(src_gpu, dest_gpu)["cap"]
                        src_gpu_atomics[dest_gpu_key] = (
                            "64,32"
                            if cap["is_iolink_atomics_32bit"] == 1
                            and cap["is_iolink_atomics_64bit"] == 1
                            else "32"
                            if cap["is_iolink_atomics_32bit"] == 1
                            else "64"
                            if cap["is_iolink_atomics_64bit"] == 1
                            else "N/A"
                        )
                    except amdsmi_exception.AmdSmiLibraryException as e:
                        src_gpu_atomics[dest_gpu_key] = "N/A"
                        logging.debug(
                            "Failed to get link atomics for %s to %s | %s",
                            self.helpers.get_gpu_id_from_device_handle(src_gpu),
                            self.helpers.get_gpu_id_from_device_handle(dest_gpu),
                            e.get_error_info(),
                        )

                topo_values[src_gpu_index]["atomics"] = src_gpu_atomics

                tabular_output_dict.update(src_gpu_atomics)
                tabular_output.append(tabular_output_dict)

            if self.logger.is_human_readable_format():
                self.logger.multiple_device_output = tabular_output
                self.logger.table_title = "ATOMICS TABLE"
                self.logger.print_output(multiple_device_enabled=True, tabular=True)

        if args.dma:
            tabular_output = []
            for src_gpu_index, src_gpu in enumerate(args.gpu):
                src_gpu_bdf = amdsmi_interface.amdsmi_get_gpu_device_bdf(src_gpu)
                if self.logger.is_human_readable_format():
                    tabular_output_dict = {"gpu": f"{src_gpu_bdf} "}
                else:
                    tabular_output_dict = {"gpu": src_gpu_bdf}
                src_gpu_dma = {}
                for dest_gpu in args.gpu:
                    dest_gpu_id = self.helpers.get_gpu_id_from_device_handle(dest_gpu)
                    dest_gpu_key = f"gpu_{dest_gpu_id}"

                    if src_gpu == dest_gpu:
                        src_gpu_dma[dest_gpu_key] = "SELF"
                        continue
                    try:
                        iolink_dma = get_cached_p2p_status(src_gpu, dest_gpu)["cap"][
                            "is_iolink_dma"
                        ]
                        src_gpu_dma[dest_gpu_key] = (
                            "T" if iolink_dma == 1 else "F" if iolink_dma == 0 else "N/A"
                        )
                    except amdsmi_exception.AmdSmiLibraryException as e:
                        src_gpu_dma[dest_gpu_key] = "N/A"
                        logging.debug(
                            "Failed to get link dma for %s to %s | %s",
                            self.helpers.get_gpu_id_from_device_handle(src_gpu),
                            self.helpers.get_gpu_id_from_device_handle(dest_gpu),
                            e.get_error_info(),
                        )

                topo_values[src_gpu_index]["dma"] = src_gpu_dma

                tabular_output_dict.update(src_gpu_dma)
                tabular_output.append(tabular_output_dict)

            if self.logger.is_human_readable_format():
                self.logger.multiple_device_output = tabular_output
                self.logger.table_title = "DMA TABLE"
                self.logger.print_output(multiple_device_enabled=True, tabular=True)

        if args.bi_dir:
            tabular_output = []
            for src_gpu_index, src_gpu in enumerate(args.gpu):
                src_gpu_bdf = amdsmi_interface.amdsmi_get_gpu_device_bdf(src_gpu)
                if self.logger.is_human_readable_format():
                    tabular_output_dict = {"gpu": f"{src_gpu_bdf} "}
                else:
                    tabular_output_dict = {"gpu": src_gpu_bdf}
                src_gpu_bi_dir = {}
                for dest_gpu in args.gpu:
                    dest_gpu_id = self.helpers.get_gpu_id_from_device_handle(dest_gpu)
                    dest_gpu_key = f"gpu_{dest_gpu_id}"

                    if src_gpu == dest_gpu:
                        src_gpu_bi_dir[dest_gpu_key] = "SELF"
                        continue
                    try:
                        iolink_bi_dir = get_cached_p2p_status(src_gpu, dest_gpu)["cap"][
                            "is_iolink_bi_directional"
                        ]
                        src_gpu_bi_dir[dest_gpu_key] = (
                            "T" if iolink_bi_dir == 1 else "F" if iolink_bi_dir == 0 else "N/A"
                        )
                    except amdsmi_exception.AmdSmiLibraryException as e:
                        src_gpu_bi_dir[dest_gpu_key] = "N/A"
                        logging.debug(
                            "Failed to get link bi-directional for %s to %s | %s",
                            self.helpers.get_gpu_id_from_device_handle(src_gpu),
                            self.helpers.get_gpu_id_from_device_handle(dest_gpu),
                            e.get_error_info(),
                        )

                topo_values[src_gpu_index]["bi_dir"] = src_gpu_bi_dir

                tabular_output_dict.update(src_gpu_bi_dir)
                tabular_output.append(tabular_output_dict)

            if self.logger.is_human_readable_format():
                self.logger.multiple_device_output = tabular_output
                self.logger.table_title = "BI-DIRECTIONAL TABLE"
                self.logger.print_output(multiple_device_enabled=True, tabular=True)

        if self.logger.is_human_readable_format():
            # Populate the legend output
            legend_parts = [
                "\n\nLegend:",
                "  SELF = Current GPU",
                "  ENABLED / DISABLED = Link is enabled or disabled",
                "  N/A = Not supported",
                "  T/F = True / False",
                "  C/NC = Coherent / Non-Coherent io links",
                "  64,32 = 64 bit and 32 bit atomic support",
                "  <BW from>-<BW to>",
            ]
            legend_output = "\n".join(legend_parts)

            if self.logger.destination == "stdout":
                print(legend_output)
            else:
                with self.logger.destination.open("a", encoding="utf-8") as output_file:
                    output_file.write(legend_output + "\n")

        self.logger.multiple_device_output = topo_values

        if self.logger.is_csv_format():
            new_output = []
            for elem in self.logger.multiple_device_output:
                new_output.append(self.logger.flatten_dict(elem, topology_override=True))
            self.logger.multiple_device_output = new_output

        if not self.logger.is_human_readable_format():
            self.logger.print_output(multiple_device_enabled=True)
