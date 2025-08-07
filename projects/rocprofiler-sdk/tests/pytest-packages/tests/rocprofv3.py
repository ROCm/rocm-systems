# MIT License
#
# Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import


def test_perfetto_data(
    pftrace_data,
    json_data,
    categories=(
        "hip",
        "hsa",
        "marker",
        "kernel",
        "memory_copy",
        "memory_allocation",
        "rocdecode_api",
        "rocjpeg_api",
        "counter_collection",
        "scratch_memory",
    ),
):

    mapping = {
        "hip": ("hip_api", "hip_api"),
        "hsa": ("hsa_api", "hsa_api"),
        "marker": ("marker_api", "marker_api"),
        "kernel": ("kernel_dispatch", "kernel_dispatch"),
        "memory_copy": ("memory_copy", "memory_copy"),
        "memory_allocation": ("memory_allocation", "memory_allocation"),
        "rocdecode_api": ("rocdecode_api", "rocdecode_api"),
        "rocjpeg_api": ("rocjpeg_api", "rocjpeg_api"),
        "counter_collection": ("counter_collection", "counter_collection"),
        "scratch_memory": ("scratch_memory", "scratch_memory"),
    }

    # make sure they specified valid categories
    for itr in categories:
        assert itr in mapping.keys()

    for pf_category, js_category in [
        itr for key, itr in mapping.items() if key in categories
    ]:
        _pf_data = pftrace_data.loc[pftrace_data["category"] == pf_category]

        _js_data = []
        if js_category != "counter_collection":
            _js_data = json_data["rocprofiler-sdk-tool"]["buffer_records"][js_category]
        else:
            unique_counter_ids = set()

            counter_info = {}
            for itr in json_data["rocprofiler-sdk-tool"]["counters"]:
                counter_info[itr["id"]["handle"]] = itr

            agent_info = {}
            for itr in json_data["rocprofiler-sdk-tool"]["agents"]:
                agent_info[itr["id"]["handle"]] = itr

            for dispatch_entry in json_data["rocprofiler-sdk-tool"]["callback_records"][
                js_category
            ]:
                counter_records = dispatch_entry["records"]
                agent_id = dispatch_entry["dispatch_data"]["dispatch_info"]["agent_id"][
                    "handle"
                ]
                agent_node_id = agent_info[agent_id]["node_id"]

                for record in counter_records:
                    counter_id = record["counter_id"]["handle"]
                    counter_name = counter_info[counter_id]["name"]
                    unique_counter_ids.add(f"Agent [{agent_node_id}] PMC {counter_name}")

            _js_data = [{"counter_id": id} for id in unique_counter_ids]

        assert len(_pf_data) == len(
            _js_data
        ), f"{pf_category} ({len(_pf_data)}):\n\t{_pf_data}\n{js_category} ({len(_js_data)}):\n\t{_js_data}"


def test_otf2_data(
    otf2_data, json_data, categories=("hip", "hsa", "marker", "kernel", "memory_copy")
):
    def get_operation_name(kind_id, op_id):
        return json_data["rocprofiler-sdk-tool"]["strings"]["buffer_records"][kind_id][
            "operations"
        ][op_id]

    def get_kind_name(kind_id):
        return json_data["rocprofiler-sdk-tool"]["strings"]["buffer_records"][kind_id][
            "kind"
        ]

    mapping = {
        "hip": ("hip_api", "hip_api"),
        "hsa": ("hsa_api", "hsa_api"),
        "marker": ("marker_api", "marker_api"),
        "kernel": ("kernel_dispatch", "kernel_dispatch"),
        "memory_copy": ("memory_copy", "memory_copy"),
        "memory_allocation": ("memory_allocation", "memory_allocation"),
        "rocdecode_api": ("rocdecode_api", "rocdecode_api"),
        "rocjpeg_api": ("rocjpeg_api", "rocjpeg_api"),
    }

    # make sure they specified valid categories
    for itr in categories:
        assert itr in mapping.keys()

    for otf2_category, json_category in [
        itr for key, itr in mapping.items() if key in categories
    ]:
        _otf2_data = otf2_data.loc[otf2_data["category"] == otf2_category]
        _json_data = json_data["rocprofiler-sdk-tool"]["buffer_records"][json_category]

        # we do not encode the roctxMark "regions" in OTF2 because
        # they don't map to the OTF2_REGION_ROLE_FUNCTION well
        if json_category == "marker_api":

            def roctx_mark_filter(val):
                return (
                    None
                    if get_kind_name(val.kind)
                    in ["MARKER_CORE_API", "MARKER_CORE_RANGE_API"]
                    and get_operation_name(val.kind, val.operation) == "roctxMarkA"
                    else val
                )

            _json_data = [itr for itr in _json_data if roctx_mark_filter(itr) is not None]

        assert len(_otf2_data) == len(
            _json_data
        ), f"{otf2_category} ({len(_otf2_data)}):\n\t{_otf2_data}\n{json_category} ({len(_json_data)}):\n\t{_json_data}"


def test_rocpd_data(
    rocpd_data,
    json_data,
    categories=(
        "hip",
        "hsa",
        "marker",
        "kernel",
        "memory_copy",
        "memory_allocation",
        "rocdecode_api",
        "rocjpeg_api",
    ),
):

    mapping = {
        "hip": (
            "hip_api",
            (
                "HIP_COMPILER_API",
                "HIP_COMPILER_API_EXT",
                "HIP_RUNTIME_API",
                "HIP_RUNTIME_API_EXT",
            ),
        ),
        "hsa": (
            "hsa_api",
            (
                "HSA_CORE_API",
                "HSA_AMD_EXT_API",
                "HSA_IMAGE_EXT_API",
                "HSA_FINALIZE_EXT_API",
            ),
        ),
        "marker": (
            "marker_api",
            (
                "MARKER_CORE_API",
                "MARKER_CONTROL_API",
                "MARKER_NAME_API",
                "MARKER_CORE_RANGE_API",
            ),
        ),
        "kernel": ("kernel_dispatch", ("KERNEL_DISPATCH")),
        "memory_copy": ("memory_copy", ("MEMORY_COPY")),
        "memory_allocation": ("memory_allocation", ("MEMORY_ALLOCATION")),
        "rocdecode_api": ("rocdecode_api", ("ROCDECODE_API")),
        "rocjpeg_api": ("rocjpeg_api", ("ROCJPEG_API")),
    }

    view_mapping = {
        "hip_api": "regions",
        "hsa_api": "regions",
        "marker_api": "regions_and_samples",
        "rccl_api": "regions",
        "rocdecode_api": "regions",
        "rocjpeg_api": "regions",
        "kernel_dispatch": "kernels",
        "memory_copy": "memory_copies",
        "memory_allocation": "memory_allocations",
    }

    # make sure they specified valid categories
    for itr in categories:
        assert itr in mapping.keys()

    for js_category, rpd_category in [
        itr for key, itr in mapping.items() if key in categories
    ]:
        _js_data = json_data["rocprofiler-sdk-tool"]["buffer_records"][js_category]
        _rpd_cats = (
            rpd_category if isinstance(rpd_category, (list, tuple)) else [rpd_category]
        )
        _rpd_cond = " OR ".join([f"category = '{itr}'" for itr in _rpd_cats])
        _rpd_query = f"SELECT * FROM {view_mapping[js_category]} WHERE {_rpd_cond}"
        _rpd_data = rocpd_data.execute(_rpd_query).fetchall()

        assert len(_rpd_data) == len(
            _js_data
        ), f"query: {_rpd_query}\n{rpd_category} ({len(_rpd_data)}):\n\t{_rpd_data}\n{js_category} ({len(_js_data)}):\n\t{_js_data}"


def _perform_time_sanity_checks(data):
    """Helper function to perform time sanity checks on data."""
    columns = data[0].keys()
    start_columns = [c for c in columns if "start" in c.lower()]
    end_columns = [c for c in columns if "end" in c.lower()]

    if not start_columns or not end_columns:
        return None, None

    for record in data:
        start_time = record[start_columns[0]]
        end_time = record[end_columns[0]]
        assert int(start_time) >= 0, f"Time error: Start time ({start_time}) < 0)."
        assert int(end_time) >= 0, f"Time error: End time ({end_time}) < 0)."
        assert int(end_time) >= int(
            start_time
        ), f"Time error: End time ({end_time}) < Start time ({start_time})."

    return start_columns[0], end_columns[0]


def _perform_csv_json_match(csv_row, json_row, mapping, json_data):

    def get_nested(d, path):
        """Helper to get nested dict values using dot notation."""
        keys = path.split(".")
        for k in keys:
            if isinstance(d, dict):
                d = d.get(k)
            else:
                return None
        return d

    for csv_key, json_info in mapping.items():
        if json_info is None:
            continue

        csv_value = csv_row[csv_key]

        if csv_key == "Operation":
            json_value = json_data["rocprofiler-sdk-tool"]["strings"]["buffer_records"][
                json_row["kind"]
            ]["operations"][json_row["operation"]]

            assert str(csv_value) in str(
                json_value
            ), f"Mismatch for {csv_key}: CSV={csv_value} JSON={json_value}"
            continue

        if csv_key == "Function":
            json_value = json_data["rocprofiler-sdk-tool"]["strings"]["buffer_records"][
                json_row["kind"]
            ]["operations"][json_row["operation"]]
        else:
            json_path, subkey = json_info
            json_value = get_nested(json_row, json_path)
            if subkey:
                json_value = (
                    json_value.get(subkey) if isinstance(json_value, dict) else None
                )

        assert str(csv_value) == str(
            json_value
        ), f"Mismatch for {csv_key}: CSV={csv_value} JSON={json_value}"


def test_csv_data(
    csv_data,
    json_data,
    categories=(
        "agent",
        "hip",
        "hsa",
        "marker",
        "kernel",
        "memory_copy",
        "memory_allocation",
        "rocdecode_api",
        "rocjpeg_api",
        "counter_collection",
    ),
):

    mapping = {
        "hip": "hip_api",
        "hsa": "hsa_api",
        "marker": "marker_api",
        "kernel": "kernel_dispatch",
        "memory_copy": "memory_copy",
        "memory_allocation": "memory_allocation",
        "rocdecode_api": "rocdecode_api",
        "rocjpeg_api": "rocjpeg_api",
        "counter_collection": "counter_collection",
    }

    keys_mapping = {
        "kernel": {
            "Thread_Id": ("thread_id", None),
            "Correlation_Id": ("correlation_id", "internal"),
            "Start_Timestamp": ("start_timestamp", None),
            "End_Timestamp": ("end_timestamp", None),
            "Queue_Id": ("dispatch_info.queue_id.handle", None),
            "Kernel_Id": ("dispatch_info.kernel_id", None),
            "Dispatch_Id": ("dispatch_info.dispatch_id", None),
            "Stream_Id": ("stream_id.handle", None),
            "Workgroup_Size_X": ("dispatch_info.workgroup_size.x", None),
            "Workgroup_Size_Y": ("dispatch_info.workgroup_size.y", None),
            "Workgroup_Size_Z": ("dispatch_info.workgroup_size.z", None),
            "Grid_Size_X": ("dispatch_info.grid_size.x", None),
            "Grid_Size_Y": ("dispatch_info.grid_size.y", None),
            "Grid_Size_Z": ("dispatch_info.grid_size.z", None),
        },
        "hip": {
            "Function": (),  # Special case
            "Thread_Id": ("thread_id", None),
            "Correlation_Id": ("correlation_id", "internal"),
            "Start_Timestamp": ("start_timestamp", None),
            "End_Timestamp": ("end_timestamp", None),
        },
        "hsa": {
            "Function": (),  # Special case
            "Thread_Id": ("thread_id", None),
            "Correlation_Id": ("correlation_id", "internal"),
            "Start_Timestamp": ("start_timestamp", None),
            "End_Timestamp": ("end_timestamp", None),
        },
        "memory_copy": {
            "Correlation_Id": ("correlation_id", "internal"),
            "Start_Timestamp": ("start_timestamp", None),
            "End_Timestamp": ("end_timestamp", None),
        },
        "memory_allocation": {
            "Operation": (),  # Special case
            "Correlation_Id": ("correlation_id", "internal"),
            "Start_Timestamp": ("start_timestamp", None),
            "End_Timestamp": ("end_timestamp", None),
        },
        "marker": {
            "Thread_Id": ("thread_id", None),
            "Correlation_Id": ("correlation_id", "internal"),
            "Start_Timestamp": ("start_timestamp", None),
            "End_Timestamp": ("end_timestamp", None),
        },
    }

    for data in csv_data:
        filename, _csv_data = data

        file_category = [category for category in categories if category in filename]
        assert len(file_category) > 0, f"{filename} is not a valid csv filename"

        category = file_category[0]

        if category == "counter_collection":
            _js_data = json_data["rocprofiler-sdk-tool"]["callback_records"][category]
        elif category == "agent":
            _js_data = json_data["rocprofiler-sdk-tool"]["agents"]
        else:
            json_records_key = mapping[category]
            _js_data = json_data["rocprofiler-sdk-tool"]["buffer_records"][
                json_records_key
            ]

        assert len(_js_data) == len(
            _csv_data
        ), f"Size mismatch for {category}: JSON size= {len(_js_data)} rows, CSV size= {len(_csv_data)} rows."

        if not _csv_data:
            continue  # Exit if there is no data to validate

        csv_start_col, csv_end_col = _perform_time_sanity_checks(_csv_data)
        json_start_col, json_end_col = _perform_time_sanity_checks(_js_data)

        if None in (csv_start_col, json_start_col, csv_end_col, json_end_col):
            continue

        _csv_data_sorted = sorted(
            _csv_data, key=lambda x: (int(x[csv_start_col]), int(x[csv_end_col]))
        )
        _js_data_sorted = sorted(
            _js_data, key=lambda x: (int(x[json_start_col]), int(x[json_end_col]))
        )

        for a, b in zip(_csv_data_sorted, _js_data_sorted):
            _perform_csv_json_match(a, b, keys_mapping[category], json_data)
