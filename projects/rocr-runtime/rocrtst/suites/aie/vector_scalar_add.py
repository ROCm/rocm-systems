# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

import sys

import numpy as np

from aie.iron import ObjectFifo, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.dialects.aiex import npu_load_pdi, dma_channel_reset_for

PROBLEM_SIZE = 1024
MEM_TILE_WIDTH = 64
AIE_TILE_WIDTH = 32

# With --full-elf the design is compiled into a standalone ELF instead of an
# xclbin. Nothing then configures the AIE array out of band, so the runtime
# sequence has to load its own PDI (see sequence() below).
FULL_ELF = "--full-elf" in sys.argv[1:]
positional = [a for a in sys.argv[1:] if not a.startswith("--")]

if not positional:
    raise ValueError("[ERROR] Expected a device name ('npu', 'npu2' or 'npu3')")
if positional[0] == "npu":
    from aie.iron.device import NPU1Col1
    dev = NPU1Col1()
elif positional[0] == "npu2":
    from aie.iron.device import NPU2Col1
    dev = NPU2Col1()
elif positional[0] == "npu3":
    from aie.iron.device import NPU3Col1
    dev = NPU3Col1()
else:
    raise ValueError("[ERROR] Device name {} is unknown".format(positional[0]))


def my_vector_bias_add():
    # Define tensor types
    mem_tile_ty = np.ndarray[(MEM_TILE_WIDTH,), np.dtype[np.int32]]
    aie_tile_ty = np.ndarray[(AIE_TILE_WIDTH,), np.dtype[np.int32]]
    all_data_ty = np.ndarray[(PROBLEM_SIZE,), np.dtype[np.int32]]

    # AIE-array data movement with object fifos
    of_in0 = ObjectFifo(mem_tile_ty, name="in")
    of_in1 = of_in0.cons().forward(obj_type=aie_tile_ty)

    of_out0 = ObjectFifo(aie_tile_ty, name="out")
    of_out1 = of_out0.cons().forward(obj_type=mem_tile_ty)

    # Define a compute task to perform
    def core_body(of_in1, of_out0):
        elem_in = of_in1.acquire(1)
        elem_out = of_out0.acquire(1)
        for i in range_(AIE_TILE_WIDTH):
            elem_out[i] = elem_in[i] + 1
        of_in1.release(1)
        of_out0.release(1)

    # Create a worker to run the task
    worker = Worker(core_body, fn_args=[of_in1.cons(), of_out0.prod()])

    # Runtime operations to move data to/from the AIE-array
    def sequence(inTensor, outTensor, in_handle, out_handle):
        # On the xclbin path the driver programs the array from the xclbin's
        # AIE_PARTITION PDI before the sequence runs. A full ELF has no xclbin,
        # so the sequence must load the PDI itself -- without this the cores
        # never start and the closing DMA token wait never retires, which the
        # driver reports as ERT_CMD_STATE_TIMEOUT. On aie4 (npu3) aiecc's control code
        # already configures the array in its own first job, so a load_pdi here would
        # apply the same configuration a second time; on real aie4 hardware that stalls
        # this design's dispatch and corrupts vector_scalar_mul.py's output.
        if FULL_ELF and positional[0] != "npu3":
            npu_load_pdi(device_ref="main")
        # Required on aie4 (npu3): re-arm every objectFifo's DMA channel before this dispatch
        # touches it. The runtime does not gap back-to-back full-ELF dispatches to cover for
        # this; without the reset, the next run's control code reconfigures a channel before the
        # previous run's DMA has drained, shifting output by one memtile double-buffer. aiecc's
        # default pipeline never emits this reset on its own, so every aie4 full-ELF design needs
        # it. See aiex.dma_channel_reset_for's own doc.
        if FULL_ELF and positional[0] == "npu3":
            for fifo in ("in", "in_fwd", "out", "out_fwd"):
                dma_channel_reset_for(fifo)
        in_handle.fill(inTensor)
        out_handle.drain(outTensor, wait=True)

    rt = Runtime(sequence, [all_data_ty, all_data_ty, of_in0.prod(), of_out1.cons()])

    # Place program components (assign them resources on the device) and generate an MLIR module
    return Program(dev, rt, workers=[worker]).resolve_program()


module = my_vector_bias_add()
res = module.operation.verify()
if res == True:
    print(module)
else:
    # Fail the build rather than writing the verifier diagnostic into aie.mlir.
    print(res, file=sys.stderr)
    sys.exit(1)
