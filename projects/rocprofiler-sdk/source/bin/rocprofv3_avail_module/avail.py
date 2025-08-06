#!/usr/bin/env python3

# MIT License
#
# Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.


import sys
import ctypes
import json


def fatal_error(msg, exit_code=1):
    sys.stderr.write(f"Fatal error: {msg}\n")
    sys.stderr.flush()
    sys.exit(exit_code)


def build_counter_string(obj):
    counter_str = "\n".join(
        ["{:20}: {}".format(key, itr) for key, itr in obj.get_as_dict().items()]
    )
    counter_str = counter_str + "\n"
    for dim in obj.dimensions:
        counter_str = counter_str + dim.__str__()
        counter_str = counter_str + "\n"
    return counter_str


class dimension:

    columns = ["dimension_id", "dimension_name", "dimension_instances"]

    def __init__(self, dimension_id, dimension_name, dimension_instances):
        self.id = dimension_id
        self.name = dimension_name
        self.instances = dimension_instances

    def get_as_dict(self):
        return dict(zip((self.columns), [self.id, self.name, self.instances]))

    def __str__(self):
        dimension = "{:20}: {}\n".format(
            "dimension_name", self.get_as_dict()["dimension_name"]
        )
        dimension += "{:20}: [0:{}]".format(
            "dimension_instances", self.get_as_dict()["dimension_instances"]
        )
        return dimension


class counter:

    columns = ["counter_name", "description"]

    def __init__(
        self,
        counter_handle,
        counter_name,
        counter_description,
        counter_dimensions,
        is_hw_constant,
    ):
        self.name = counter_name
        self.counter_handle = counter_handle
        self.description = counter_description
        self.dimensions = counter_dimensions
        self.is_hw_constant = is_hw_constant

    def get_as_dict(self):
        return dict(zip((self.columns), [self.name, self.description]))

    def __str__(self):
        return "\n".join(
            ["{:20}: {}".format(key, itr) for key, itr in self.get_as_dict().items()]
        )


class derived_counter(counter):

    columns = ["counter_name", "description", "expression"]

    def __init__(
        self,
        counter_handle,
        counter_name,
        counter_description,
        counter_expression,
        counter_dimensions,
        is_hw_constant,
    ):
        super().__init__(
            counter_handle,
            counter_name,
            counter_description,
            counter_dimensions,
            is_hw_constant,
        )
        self.expression = counter_expression

    def get_as_dict(self):
        return dict(zip((self.columns), [self.name, self.description, self.expression]))

    def __str__(self):
        return build_counter_string(self)


class basic_counter(counter):

    columns = ["counter_name", "description", "block"]

    def __init__(
        self,
        counter_handle,
        counter_name,
        counter_description,
        counter_block,
        counter_dimensions,
        is_hw_constant,
    ):
        super().__init__(
            counter_handle,
            counter_name,
            counter_description,
            counter_dimensions,
            is_hw_constant,
        )
        self.block = counter_block

    def get_as_dict(self):
        return dict(zip((self.columns), [self.name, self.description, self.block]))

    def __str__(self):
        return build_counter_string(self)


class pc_config:

    columns = ["method", "unit", "min_interval", "max_interval", "flags"]

    def __init__(self, config_method, config_unit, min_interval, max_interval, flags):

        self.method = self.get_method_string(config_method.value)
        self.unit = self.get_unit_string(config_unit.value)
        self.min_interval = min_interval
        self.max_interval = max_interval
        self.flags = flags

    def __str__(self):

        return "\n".join(
            [
                "   {:20}:{}".format(
                    key,
                    itr if key == "method" or key == "unit" else self.get_value(key, itr),
                )
                for key, itr in self.get_as_dict().items()
            ]
        )

    @staticmethod
    def get_value(key, itr):
        if key == "min_interval" or key == "max_interval":
            return itr.value
        elif key == "flags":
            if itr.value == 1:
                return "interval pow2"
            else:
                return "none"
        else:
            fatal_error("Incorrect key")

    @staticmethod
    def get_method_string(key):
        method_map = {1: "stochastic", 2: "host_trap"}
        return method_map[key]

    @staticmethod
    def get_unit_string(key):
        unit_map = {1: "instructions", 2: "cycle", 3: "time"}
        return unit_map[key]

    def get_as_dict(self):

        return dict(
            zip(
                (self.columns),
                [
                    self.method,
                    self.unit,
                    self.min_interval,
                    self.max_interval,
                    self.flags,
                ],
            )
        )


class loadLibrary:
    libname = None
    c_lib = None


def get_library():

    get_library.libname = None
    if loadLibrary.c_lib is None:
        loadLibrary.c_lib = ctypes.CDLL(loadLibrary.libname)
    return loadLibrary.c_lib


def get_string_value(str_ptr):
    return ctypes.cast(str_ptr, ctypes.c_char_p).value.decode("utf-8")


def get_agent_info(agent_handle):
    lib = get_library()
    lib.agent_info.argtypes = [ctypes.c_ulong, ctypes.POINTER(ctypes.c_char_p)]
    agent_info_str = ctypes.c_char_p()
    lib.agent_info(agent_handle, ctypes.byref(agent_info_str))
    return json.loads(agent_info_str.value.decode("utf-8"))


def get_number_of_counters(agent_handle):
    lib = get_library()
    lib.get_number_of_counters.restype = ctypes.c_ulong
    lib.get_number_of_counters.argtypes = [ctypes.c_ulong]
    return lib.get_number_of_agent_counters(agent_handle)


def get_number_agents():
    lib = get_library()
    lib.get_number_of_agents.restype = ctypes.c_ulong
    return lib.get_number_of_agents()


def get_agent_handles():
    lib = get_library()
    num_agents = get_number_agents()
    lib.agent_handles.argtypes = [ctypes.c_ulong * num_agents, ctypes.c_ulong]
    agent_handles_arr = (ctypes.c_ulong * num_agents)()
    lib.agent_handles(agent_handles_arr, num_agents)
    return list(agent_handles_arr)


def get_agent_info_map():
    agent_info_map = {}
    agents = get_agent_handles()
    for agent in agents:
        agent_info_map[agent] = get_agent_info(agent)

    return agent_info_map


def get_number_of_agent_counters(agent_handle):
    lib = get_library()
    lib.get_number_of_agent_counters.argtypes = [ctypes.c_ulong]
    return lib.get_number_of_agent_counters(agent_handle)


def get_agent_counter_handles(agent_handle):
    lib = get_library()
    num_counters = get_number_of_agent_counters(agent_handle)
    lib.agent_counter_handles.argtypes = [
        ctypes.c_ulong * num_counters,
        ctypes.c_ulong,
        ctypes.c_ulong,
    ]
    counter_handles = (ctypes.c_ulong * num_counters)()
    lib.agent_counter_handles(counter_handles, agent_handle, num_counters)
    return list(counter_handles)


def get_dimensions(counter_handle):
    lib = get_library()
    lib.get_number_of_dimensions.argtypes = [ctypes.c_ulong]
    lib.get_number_of_dimensions.restype = ctypes.c_ulong
    num_dims = lib.get_number_of_dimensions(counter_handle)

    lib.counter_dimension_ids.argtypes = [
        ctypes.c_ulong,
        ctypes.c_ulong * num_dims,
        ctypes.c_uint,
    ]
    dims_ids = (ctypes.c_ulong * num_dims)()
    lib.counter_dimension.argtypes = [
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.POINTER(ctypes.c_char_p),
        ctypes.POINTER(ctypes.c_uint),
    ]
    lib.counter_dimension_ids(counter_handle, dims_ids, num_dims)
    dimensions = []
    for dim_id in list(dims_ids):
        dimension_name = ctypes.c_char_p()
        dimension_instance = ctypes.c_uint()
        lib.counter_dimension(
            counter_handle,
            dim_id,
            ctypes.byref(dimension_name),
            ctypes.byref(dimension_instance),
        )
        dim = dimension(
            dim_id, get_string_value(dimension_name), dimension_instance.value
        )
        dimensions.append(dim)
    return dimensions


def get_counters():
    agent_counters = {}
    agents = get_agent_handles()
    agent_counters = {}
    agent_info_map = get_agent_info_map()
    for agent in agents:
        if agent_info_map[agent]["type"] != 2:
            continue
        agent_counters.setdefault(agent, [])
        counters = get_agent_counter_handles(agent)
        if counters:
            for counter_id in list(counters):
                counter_info = get_counter_info(counter_id)
                agent_counters[agent].append(counter_info)
    return agent_counters


def get_pc_sample_configs():
    agent_pc_sample_config = {}
    agents = get_agent_handles()

    for agent in agents:
        configs = get_pc_sample_config(agent)
        if len(configs) > 0:
            agent_pc_sample_config[agent] = configs

    return agent_pc_sample_config


def get_counter_info(counter_handle):
    lib = get_library()
    lib.counter_info.argtypes = [
        ctypes.c_ulong,
        ctypes.POINTER(ctypes.c_char_p),
        ctypes.POINTER(ctypes.c_char_p),
        ctypes.POINTER(ctypes.c_uint),
        ctypes.POINTER(ctypes.c_uint),
    ]
    counter_name = ctypes.c_char_p()
    counter_description = ctypes.c_char_p()
    is_derived = ctypes.c_uint()
    is_hw_constant = ctypes.c_uint()
    lib.counter_info(
        counter_handle,
        ctypes.byref(counter_name),
        ctypes.byref(counter_description),
        ctypes.byref(is_derived),
        ctypes.byref(is_hw_constant),
    )

    if is_derived.value == 1:
        lib.counter_expression.argtypes = [
            ctypes.c_ulong,
            ctypes.POINTER(ctypes.c_char_p),
        ]
        expression = ctypes.c_char_p()
        lib.counter_expression(counter_handle, ctypes.byref(expression))
        dimensions = get_dimensions(counter_handle)
        return derived_counter(
            counter_handle,
            get_string_value(counter_name),
            get_string_value(counter_description),
            get_string_value(expression),
            dimensions,
            is_hw_constant,
        )

    elif not is_hw_constant.value:
        lib.counter_block.argtypes = [ctypes.c_ulong, ctypes.POINTER(ctypes.c_char_p)]
        block = ctypes.c_char_p()
        lib.counter_block(counter_handle, ctypes.byref(block))
        dimensions = get_dimensions(counter_handle)
        return basic_counter(
            counter_handle,
            get_string_value(counter_name),
            get_string_value(counter_description),
            get_string_value(block),
            dimensions,
            is_hw_constant,
        )
    else:
        return counter(
            counter_handle,
            get_string_value(counter_name),
            get_string_value(counter_description),
            [],
            is_hw_constant.value,
        )


def get_number_of_pc_sample_configs(agent_handle):
    lib = get_library()
    lib.get_number_of_pc_sample_configs.argtypes = [ctypes.c_ulong]
    lib.get_number_of_pc_sample_configs.restype = ctypes.c_ulong
    return lib.get_number_of_pc_sample_configs(agent_handle)


def get_pc_sample_config(agent_handle):
    lib = get_library()
    num_configs = get_number_of_pc_sample_configs(agent_handle)
    lib.pc_sample_config.argtypes = [
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.POINTER(ctypes.c_ulong),
        ctypes.POINTER(ctypes.c_ulong),
        ctypes.POINTER(ctypes.c_ulong),
        ctypes.POINTER(ctypes.c_ulong),
        ctypes.POINTER(ctypes.c_ulong),
    ]
    pc_configs = []

    for config in range(0, num_configs):
        method = (ctypes.c_ulong)()
        unit = (ctypes.c_ulong)()
        max_interval = (ctypes.c_ulong)()
        min_interval = (ctypes.c_ulong)()
        flags = (ctypes.c_ulong)()
        lib.pc_sample_config(
            agent_handle,
            config,
            ctypes.byref(method),
            ctypes.byref(unit),
            ctypes.byref(min_interval),
            ctypes.byref(max_interval),
            flags,
        )
        pc_configs.append(
            pc_config(
                method,
                unit,
                min_interval,
                max_interval,
                flags,
            )
        )
    return pc_configs


def check_pmc(agent_counter):
    lib = get_library()

    def get_counter_names(counter_ids):

        counter_names = []
        for counter_id in counter_ids:
            counter = get_counter_info(counter_id)
            if counter.counter_handle == counter_id:
                counter_names.append(counter.name)
        return counter_names

    def get_agent_name(agent_id):
        agent_info_map = get_agent_info_map()
        for agent, info in agent_info_map.items():
            if agent == agent_id:
                return info["name"]

    for agent, counter_ids in agent_counter.items():
        num_counters = len(counter_ids)
        counters = (ctypes.c_ulong * num_counters)(*counter_ids)
        lib.is_counter_set.argtypes = [
            ctypes.c_ulong * num_counters,
            ctypes.c_ulong,
            ctypes.c_ulong,
        ]
        lib.is_counter_set.restype = ctypes.c_bool
        if lib.is_counter_set(counters, agent, num_counters) is False:
            fatal_error(
                "{} not collected on agent {}".format(
                    " ".join(get_counter_names(counter_ids)), get_agent_name(agent)
                )
            )
    return True
