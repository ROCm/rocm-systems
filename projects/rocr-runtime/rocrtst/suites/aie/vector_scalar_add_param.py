# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

import sys

import numpy as np

from aie.iron import (
    DeviceConfiguration,
    ObjectFifo,
    Program,
    Runtime,
    ScratchpadParameter,
    Worker,
    sync_parameters,
)
from aie.iron.device import NPU2Col1
from aie.iron.controlflow import range_
from aie.dialects.aiex import npu_load_pdi

PROBLEM_SIZE = 1024
MEM_TILE_WIDTH = 64
AIE_TILE_WIDTH = 32

# Full ELF only: a scratchpad parameter reaches the cores through the full ELF's control
# scratchpad, which the PDI + instruction sequence path has no counterpart for.
positional = [a for a in sys.argv[1:] if not a.startswith("--")]
if not positional or positional[0] != "npu2":
    raise ValueError("[ERROR] Expected the device name 'npu2'")


def my_vector_param_add():
    mem_tile_ty = np.ndarray[(MEM_TILE_WIDTH,), np.dtype[np.int32]]
    aie_tile_ty = np.ndarray[(AIE_TILE_WIDTH,), np.dtype[np.int32]]
    all_data_ty = np.ndarray[(PROBLEM_SIZE,), np.dtype[np.int32]]

    of_in0 = ObjectFifo(mem_tile_ty, name="in")
    of_in1 = of_in0.cons().forward(obj_type=aie_tile_ty)

    of_out0 = ObjectFifo(aie_tile_ty, name="out")
    of_out1 = of_out0.cons().forward(obj_type=mem_tile_ty)

    # Set by the host before each dispatch, in the control scratchpad.
    bias = ScratchpadParameter("bias", np.int32)

    # Runs once per dispatch rather than forever: the sequence below restarts the core, and a core
    # that looped would read the parameter for its next pass before the next dispatch had set it.
    def core_body(of_in1, of_out0, bias):
        b = bias.read()
        for _ in range_(PROBLEM_SIZE // AIE_TILE_WIDTH):
            elem_in = of_in1.acquire(1)
            elem_out = of_out0.acquire(1)
            for i in range_(AIE_TILE_WIDTH):
                elem_out[i] = elem_in[i] + b
            of_in1.release(1)
            of_out0.release(1)

    worker = Worker(core_body, fn_args=[of_in1.cons(), of_out0.prod(), bias], while_true=False)

    def sequence(inTensor, outTensor, in_handle, out_handle):
        # Loading the empty device first forces the array to reconfigure, so the core starts afresh
        # every dispatch (see vector_scalar_add.py for why a full ELF loads its own PDI at all).
        npu_load_pdi(device_ref="empty")
        npu_load_pdi(device_ref="main")
        sync_parameters()
        in_handle.fill(inTensor)
        out_handle.drain(outTensor, wait=True)
        # Unloading the design again, as MLIR-AIR's sequences do: on aie2p firmware, without it the
        # next dispatch's sync_parameters delivers this dispatch's value instead of its own.
        npu_load_pdi(device_ref="empty")

    rt = Runtime(
        sequence,
        [all_data_ty, all_data_ty, of_in0.prod(), of_out1.cons()],
        name="sequence",
        implicit_configure=False,
    )

    main = DeviceConfiguration("main", NPU2Col1(), workers=[worker], runtimes=[rt])
    empty = DeviceConfiguration("empty", NPU2Col1())
    return Program.compose([empty, main], entry=rt)


module = my_vector_param_add().resolve_program()
res = module.operation.verify()
if res == True:
    print(module)
else:
    print(res, file=sys.stderr)
    sys.exit(1)
