# MIT License
#
# Copyright (c) 2020 Advanced Micro Devices, Inc. All rights reserved.
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

# Dependencies

# HIP dependency is handled earlier in the project cmake file
# when VerifyCompiler.cmake is included.

# GIT

# Test dependencies

# For downloading, building, and installing required dependencies
include(cmake/DownloadProject.cmake)

message(STATUS "Generating ROCM NetIB... ")

# -------------------------
# Configurable paths
# -------------------------
# Path to RCCL source tree (local clone)
set(RCCL_SRC_DIR "${CMAKE_SOURCE_DIR}" CACHE PATH "Path to RCCL source directory")
# Path to patch file
set(ROCM_NETIB_PATCH_FILE "${CMAKE_SOURCE_DIR}/ext-src/rocm_netib.patch" CACHE FILEPATH "ROCM NETIB Patch file to apply to RCCL")
set(ROCM_NETIB_FILE "${CMAKE_SOURCE_DIR}/src/transport/net_ib_rocm.cc" CACHE FILEPATH "Generated ROCM NETIB file")

# -------------------------
# Find tools
# -------------------------
find_program(PATCH_EXECUTABLE patch)
find_program(SED_EXECUTABLE sed)

execute_process(
  COMMAND ${CMAKE_COMMAND} -E echo "Applying RCCL ROCM NetIB patch... to ${CMAKE_SOURCE_DIR}"
  COMMAND bash -c "patch -p1 -i ${ROCM_NETIB_PATCH_FILE} -o ${ROCM_NETIB_FILE}"
  WORKING_DIRECTORY ${RCCL_SRC_DIR}
)
execute_process(
  COMMAND bash -c "sed -i 's/NCCL_PARAM(Ib/NCCL_PARAM(RocmIb/g' ${ROCM_NETIB_FILE}"
  WORKING_DIRECTORY ${RCCL_SRC_DIR}
)
execute_process(
  COMMAND bash -c "sed -i 's/RCCL_PARAM(Ib/RCCL_PARAM(RocmIb/g' ${ROCM_NETIB_FILE}"
  WORKING_DIRECTORY ${RCCL_SRC_DIR}
)
execute_process(
  COMMAND bash -c "sed -i 's/ncclParamIb/ncclParamRocmIb/g' ${ROCM_NETIB_FILE}"
  WORKING_DIRECTORY ${RCCL_SRC_DIR}
)
execute_process(
  COMMAND bash -c "sed -i 's/rcclParamIb/rcclParamRocmIb/g' ${ROCM_NETIB_FILE}"
  WORKING_DIRECTORY ${RCCL_SRC_DIR}
)
execute_process(
  COMMAND bash -c "sed -i 's/ncclIbMergedDevs/rocmIbMergedDevs/g' ${ROCM_NETIB_FILE}"
  WORKING_DIRECTORY ${RCCL_SRC_DIR}
)
execute_process(
  COMMAND bash -c "sed -i 's/ncclIbDevs/rocmIbDevs/g' ${ROCM_NETIB_FILE}"
  WORKING_DIRECTORY ${RCCL_SRC_DIR}
)
execute_process(
  COMMAND bash -c "sed -i 's/ncclIbLock/rocmIbLock/g' ${ROCM_NETIB_FILE}"
  WORKING_DIRECTORY ${RCCL_SRC_DIR}
)
execute_process(
  COMMAND bash -c "sed -i 's/ibProviderName/rocmIbProviderName/g' ${ROCM_NETIB_FILE}"
  WORKING_DIRECTORY ${RCCL_SRC_DIR}
)
execute_process(
  COMMAND bash -c "sed -i 's/ncclIbAsyncThread/rocmIbAsyncThread/g' ${ROCM_NETIB_FILE}"
  WORKING_DIRECTORY ${RCCL_SRC_DIR}
)
execute_process(
  COMMAND bash -c "sed -i 's/ncclIbGdrSupport/rocmIbGdrSupport/g' ${ROCM_NETIB_FILE}"
  WORKING_DIRECTORY ${RCCL_SRC_DIR}
)
execute_process(
  COMMAND bash -c "sed -i 's/ncclIbDmaBufSupport/rocmIbDmaBufSupport/g' ${ROCM_NETIB_FILE}"
  WORKING_DIRECTORY ${RCCL_SRC_DIR}
)
execute_process(
  COMMAND bash -c "sed -i 's/ncclIbInitCommDevBase/rocmIbInitCommDevBase/g' ${ROCM_NETIB_FILE}"
  WORKING_DIRECTORY ${RCCL_SRC_DIR}
)
execute_process(
  COMMAND bash -c "sed -i 's/ncclIbDestroyBase/rocmIbDestroyBase/g' ${ROCM_NETIB_FILE}"
  WORKING_DIRECTORY ${RCCL_SRC_DIR}
)
execute_process(
  COMMAND bash -c "sed -i 's/ncclIbRtrQp/rocmIbRtrQp/g' ${ROCM_NETIB_FILE}"
  WORKING_DIRECTORY ${RCCL_SRC_DIR}
)
execute_process(
  COMMAND bash -c "sed -i 's/ncclIbRtsQp/rocmIbRtsQp/g' ${ROCM_NETIB_FILE}"
  WORKING_DIRECTORY ${RCCL_SRC_DIR}
)
execute_process(
  COMMAND bash -c "sed -i 's/ForceEnableGdrdma/RocmForceEnableGdrdma/g' ${ROCM_NETIB_FILE}"
  WORKING_DIRECTORY ${RCCL_SRC_DIR}
)
execute_process(
  COMMAND bash -c "sed -i 's/ncclIbCheckVProps/rocmIbCheckVProps/g' ${ROCM_NETIB_FILE}"
  WORKING_DIRECTORY ${RCCL_SRC_DIR}
)
execute_process(
  COMMAND bash -c "sed -i 's/ncclIbGetRequest/rocmIbGetRequest/g' ${ROCM_NETIB_FILE}"
  WORKING_DIRECTORY ${RCCL_SRC_DIR}
)
execute_process(
  COMMAND bash -c "sed -i 's/ncclIbFreeRequest/rocmIbFreeRequest/g' ${ROCM_NETIB_FILE}"
  WORKING_DIRECTORY ${RCCL_SRC_DIR}
)
execute_process(
  COMMAND bash -c "sed -i 's/ncclIbRegMrDmaBufInternal/rocmIbRegMrDmaBufInternal/g' ${ROCM_NETIB_FILE}"
  WORKING_DIRECTORY ${RCCL_SRC_DIR}
)
execute_process(
  COMMAND bash -c "sed -i 's/ncclIbGetNetCommDevBase/rocmIbGetNetCommDevBase/g' ${ROCM_NETIB_FILE}"
  WORKING_DIRECTORY ${RCCL_SRC_DIR}
)
execute_process(
  COMMAND bash -c "sed -i 's/ncclIbDeregMrInternal/rocmIbDeregMrInternal/g' ${ROCM_NETIB_FILE}"
  WORKING_DIRECTORY ${RCCL_SRC_DIR}
)
execute_process(
  COMMAND bash -c "sed -i 's/ncclIbPostFifo/rocmIbPostFifo/g' ${ROCM_NETIB_FILE}"
  WORKING_DIRECTORY ${RCCL_SRC_DIR}
)
execute_process(
  COMMAND bash -c "sed -i 's/reqTypeStr/rocmIbReqTypeStr/g' ${ROCM_NETIB_FILE}"
  WORKING_DIRECTORY ${RCCL_SRC_DIR}
)
execute_process(
  COMMAND bash -c "sed -i 's/rcclNetP2pPolicy/rcclRocmNetP2pPolicy/g' ${ROCM_NETIB_FILE}"
  WORKING_DIRECTORY ${RCCL_SRC_DIR}
)
execute_process(
  COMMAND bash -c "sed -i 's/ncclIbMakeVDeviceInternal/rocmIbMakeVDeviceInternal/g' ${ROCM_NETIB_FILE}"
  WORKING_DIRECTORY ${RCCL_SRC_DIR}
)
execute_process(
  COMMAND bash -c "sed -i 's/ncclIbMakeVDevice/rocmIbMakeVDevice/g' ${ROCM_NETIB_FILE}"
  WORKING_DIRECTORY ${RCCL_SRC_DIR}
)
execute_process(
  COMMAND bash -c "sed -i 's/ncclIbInit/rocmIbInit/g' ${ROCM_NETIB_FILE}"
  WORKING_DIRECTORY ${RCCL_SRC_DIR}
)
execute_process(
  COMMAND bash -c "sed -i 's/ncclIbDevices/rocmIbDevices/g' ${ROCM_NETIB_FILE}"
  WORKING_DIRECTORY ${RCCL_SRC_DIR}
)
execute_process(
  COMMAND bash -c "sed -i 's/ncclIbGetPhysProperties/rocmIbGetPhysProperties/g' ${ROCM_NETIB_FILE}"
  WORKING_DIRECTORY ${RCCL_SRC_DIR}
)
execute_process(
  COMMAND bash -c "sed -i 's/ncclIbGetProperties/rocmIbGetProperties/g' ${ROCM_NETIB_FILE}"
  WORKING_DIRECTORY ${RCCL_SRC_DIR}
)
execute_process(
  COMMAND bash -c "sed -i 's/ncclIbListen\(/rocmIbListen\(/g' ${ROCM_NETIB_FILE}"
  WORKING_DIRECTORY ${RCCL_SRC_DIR}
)
execute_process(
  COMMAND bash -c "sed -i 's/ncclIbListen,/rocmIbListen,/g' ${ROCM_NETIB_FILE}"
  WORKING_DIRECTORY ${RCCL_SRC_DIR}
)
execute_process(
  COMMAND bash -c "sed -i 's/ncclIbConnect\(/rocmIbConnect\(/g' ${ROCM_NETIB_FILE}"
  WORKING_DIRECTORY ${RCCL_SRC_DIR}
)
execute_process(
  COMMAND bash -c "sed -i 's/ncclIbConnect /rocmIbConnect /g' ${ROCM_NETIB_FILE}"
  WORKING_DIRECTORY ${RCCL_SRC_DIR}
)
execute_process(
  COMMAND bash -c "sed -i 's/ncclIbConnect,/rocmIbConnect,/g' ${ROCM_NETIB_FILE}"
  WORKING_DIRECTORY ${RCCL_SRC_DIR}
)
execute_process(
  COMMAND bash -c "sed -i 's/ncclIbAccept/rocmIbAccept/g' ${ROCM_NETIB_FILE}"
  WORKING_DIRECTORY ${RCCL_SRC_DIR}
)
execute_process(
  COMMAND bash -c "sed -i 's/ncclIbTest/rocmIbTest/g' ${ROCM_NETIB_FILE}"
  WORKING_DIRECTORY ${RCCL_SRC_DIR}
)
execute_process(
  COMMAND bash -c "sed -i 's/ncclIbRegMrDmaBuf/rocmIbRegMrDmaBuf/g' ${ROCM_NETIB_FILE}"
  WORKING_DIRECTORY ${RCCL_SRC_DIR}
)
execute_process(
  COMMAND bash -c "sed -i 's/ncclIbRegMr/rocmIbRegMr/g' ${ROCM_NETIB_FILE}"
  WORKING_DIRECTORY ${RCCL_SRC_DIR}
)
execute_process(
  COMMAND bash -c "sed -i 's/ncclIbDeregMr/rocmIbDeregMr/g' ${ROCM_NETIB_FILE}"
  WORKING_DIRECTORY ${RCCL_SRC_DIR}
)
execute_process(
  COMMAND bash -c "sed -i 's/ncclIbIsend/rocmIbIsend/g' ${ROCM_NETIB_FILE}"
  WORKING_DIRECTORY ${RCCL_SRC_DIR}
)
execute_process(
  COMMAND bash -c "sed -i 's/ncclIbIrecv/rocmIbIrecv/g' ${ROCM_NETIB_FILE}"
  WORKING_DIRECTORY ${RCCL_SRC_DIR}
)
execute_process(
  COMMAND bash -c "sed -i 's/ncclIbIflush/rocmIbIflush/g' ${ROCM_NETIB_FILE}"
  WORKING_DIRECTORY ${RCCL_SRC_DIR}
)
execute_process(
  COMMAND bash -c "sed -i 's/ncclIbCloseSend/rocmIbCloseSend/g' ${ROCM_NETIB_FILE}"
  WORKING_DIRECTORY ${RCCL_SRC_DIR}
)
execute_process(
  COMMAND bash -c "sed -i 's/ncclIbCloseRecv/rocmIbCloseRecv/g' ${ROCM_NETIB_FILE}"
  WORKING_DIRECTORY ${RCCL_SRC_DIR}
)
execute_process(
  COMMAND bash -c "sed -i 's/ncclIbCloseListen/rocmIbCloseListen/g' ${ROCM_NETIB_FILE}"
  WORKING_DIRECTORY ${RCCL_SRC_DIR}
)
execute_process(
  COMMAND bash -c "sed -i 's/ncclNetIb/rocmNetIb/g' ${ROCM_NETIB_FILE}"
  WORKING_DIRECTORY ${RCCL_SRC_DIR}
)
execute_process(
  COMMAND bash -c "sed -i 's/ncclIbFinalize/rocmNetIbFinalize/g' ${ROCM_NETIB_FILE}"
  WORKING_DIRECTORY ${RCCL_SRC_DIR}
)
execute_process(
  COMMAND bash -c "sed -i 's/ncclIbSetNetAttr/rocmNetIbSetNetAttr/g' ${ROCM_NETIB_FILE}"
  WORKING_DIRECTORY ${RCCL_SRC_DIR}
)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
