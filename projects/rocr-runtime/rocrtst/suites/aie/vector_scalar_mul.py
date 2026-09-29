# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

#
# A second design for the dispatch tests, deliberately different from
# vector_scalar_add.py in the two ways the runtime can see:
#
#   * it multiplies by a compile-time constant instead of adding one, so a test
#     can tell which kernel ran from the output alone, and
#   * it takes a single in/out tensor instead of a separate input and output, so
#     its packets carry one kernarg rather than two.
#
# Those differences are what make it useful for interleaving: two PDIs on one
# queue, with different command shapes, whose results cannot be confused.

import numpy as np
import sys

from aie.iron import ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU1Col1, NPU2Col1
from aie.iron.controlflow import range_
from aie.dialects.aiex import npu_load_pdi

PROBLEM_SIZE = 1024
MEM_TILE_WIDTH = 64
AIE_TILE_WIDTH = 32

# The scale is baked into the design rather than passed as a kernel argument, so the packet
# carries exactly one kernarg. Required rather than defaulted: VSMUL_SCALE in CMakeLists.txt sets
# it for both the build and the tests' expected results, and a default here would be a second copy
# to drift from it.
SCALE = None
for _arg in sys.argv[1:]:
    if _arg.startswith("--scale="):
        SCALE = int(_arg.split("=", 1)[1])
if SCALE is None:
    raise ValueError("[ERROR] Expected --scale=<int> (the build passes VSMUL_SCALE)")

# With --full-elf the design is compiled into a standalone ELF instead of an
# xclbin. Nothing then configures the AIE array out of band, so the runtime
# sequence has to load its own PDI (see sequence() below).
FULL_ELF = "--full-elf" in sys.argv[1:]
positional = [a for a in sys.argv[1:] if not a.startswith("--")]

if not positional:
    raise ValueError("[ERROR] Expected a device name ('npu' or 'npu2')")
if positional[0] == "npu":
    dev = NPU1Col1()
elif positional[0] == "npu2":
    dev = NPU2Col1()
else:
    raise ValueError("[ERROR] Device name {} is unknown".format(positional[0]))


def my_vector_scalar_mul():
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
            elem_out[i] = elem_in[i] * SCALE
        of_in1.release(1)
        of_out0.release(1)

    # Create a worker to run the task
    worker = Worker(core_body, fn_args=[of_in1.cons(), of_out0.prod()])

    # Runtime operations to move data to/from the AIE-array. One tensor serves as
    # both source and destination: the fill and the drain walk it in the same
    # order, so a chunk is only written back after it has been read.
    def sequence(inoutTensor, in_handle, out_handle):
        # On the xclbin path the driver programs the array from the xclbin's
        # AIE_PARTITION PDI before the sequence runs. A full ELF has no xclbin,
        # so the sequence must load the PDI itself -- without this the cores
        # never start and the closing DMA token wait never retires, which the
        # driver reports as ERT_CMD_STATE_TIMEOUT.
        if FULL_ELF:
            npu_load_pdi(device_ref="main")
        in_handle.fill(inoutTensor)
        out_handle.drain(inoutTensor, wait=True)

    rt = Runtime(sequence, [all_data_ty, of_in0.prod(), of_out1.cons()])

    # Place program components (assign them resources on the device) and generate an MLIR module
    return Program(dev, rt, workers=[worker]).resolve_program()


module = my_vector_scalar_mul()
res = module.operation.verify()
if res == True:
    print(module)
else:
    # Fail the build rather than writing the verifier diagnostic into aie.mlir.
    print(res, file=sys.stderr)
    sys.exit(1)
