################################################################################
## Copyright (c) 2018-2022 Advanced Micro Devices, Inc.
##
## Permission is hereby granted, free of charge, to any person obtaining a copy
## of this software and associated documentation files (the "Software"), to
## deal in the Software without restriction, including without limitation the
## rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
## sell copies of the Software, and to permit persons to whom the Software is
## furnished to do so, subject to the following conditions:
##
## The above copyright notice and this permission notice shall be included in
## all copies or substantial portions of the Software.
##
## THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
## IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
## FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
## AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
## LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
## FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
## IN THE SOFTWARE.
################################################################################

# Generating tracing primitives
set ( GEN_INC_DIR ${PROJECT_BINARY_DIR}/inc )
set ( GEN_SRC_DIR ${PROJECT_BINARY_DIR}/src )
execute_process ( COMMAND sh -xc "mkdir -p ${GEN_INC_DIR}" )
execute_process ( COMMAND sh -xc "mkdir -p ${GEN_SRC_DIR}" )
execute_process ( COMMAND sh -xc "${CMAKE_C_COMPILER} -E ${HSA_RUNTIME_INC_PATH}/hsa.h > ${GEN_INC_DIR}/hsa_pp.h" )
execute_process ( COMMAND sh -xc "${CMAKE_C_COMPILER} -E ${HSA_RUNTIME_INC_PATH}/hsa_ext_amd.h > ${GEN_INC_DIR}/hsa_ext_amd_pp.h" )
execute_process ( COMMAND sh -xc "python3 ${ROOT_DIR}/script/gen_ostream_ops.py -in ${GEN_INC_DIR}/hsa_pp.h,${GEN_INC_DIR}/hsa_ext_amd_pp.h -out ${GEN_INC_DIR}/hsa_ostream_ops.h" )
execute_process ( COMMAND sh -xc "python3 ${ROOT_DIR}/script/hsaap.py ${PROJECT_BINARY_DIR} ${HSA_RUNTIME_INC_PATH}" )
execute_process ( COMMAND sh -xc "${CMAKE_C_COMPILER} -E ${HSA_KMT_INC_PATH}/hsakmttypes.h > ${GEN_INC_DIR}/hsakmttypes_pp.h" )
execute_process ( COMMAND sh -xc "${CMAKE_C_COMPILER} -E ${HIP_PATH}/include/hip/hip_runtime_api.h ${HIP_DEFINES} -I${HIP_PATH}/include -I${ROCM_ROOT_DIR}/hsa/include > ${GEN_INC_DIR}/hip_runtime_api_pp.h" )
execute_process ( COMMAND sh -xc "python3 ${ROOT_DIR}/script/gen_ostream_ops.py -in ${GEN_INC_DIR}/hip_runtime_api_pp.h -out ${GEN_INC_DIR}/hip_ostream_ops.h" )
execute_process ( COMMAND sh -xc "mkdir ${GEN_INC_DIR}/rocprofiler" )
execute_process ( COMMAND sh -xc "ln -s ${ROOT_DIR}/../rocprofiler/inc/rocprofiler.h ${GEN_INC_DIR}/rocprofiler/rocprofiler.h" )
execute_process ( COMMAND sh -xc "ln -s ${ROOT_DIR}/../rocprofiler/src/core/activity.h ${GEN_INC_DIR}/rocprofiler/activity.h" )

# Build dynamic Library object
set ( TARGET_LIB ${TARGET_NAME} )
set ( LIB_SRC
  ${LIB_DIR}/core/roctracer.cpp
  ${LIB_DIR}/proxy/proxy_queue.cpp
  ${LIB_DIR}/proxy/simple_proxy_queue.cpp
  ${LIB_DIR}/proxy/intercept_queue.cpp
  ${LIB_DIR}/util/hsa_rsrc_factory.cpp
)
add_library ( ${TARGET_LIB} ${LIBRARY_TYPE} ${LIB_SRC} )
target_include_directories ( ${TARGET_LIB} PRIVATE ${LIB_DIR} ${ROOT_DIR} ${ROOT_DIR}/inc ${HSA_RUNTIME_INC_PATH} ${ROCM_INC_PATH} ${HIP_INC_DIR} ${HSA_KMT_INC_PATH} ${GEN_INC_DIR} )
target_link_libraries( ${TARGET_LIB} PRIVATE ${HSA_RUNTIME_LIB} c stdc++ )

# Build ROCTX tracing library
set ( ROCTX_LIB "roctx64" )
set ( ROCTX_LIB_SRC
  ${LIB_DIR}/roctx/roctx.cpp
)
add_library ( ${ROCTX_LIB} SHARED ${ROCTX_LIB_SRC} )
target_include_directories ( ${ROCTX_LIB} PRIVATE ${LIB_DIR} ${ROOT_DIR} ${ROOT_DIR}/inc ${HSA_RUNTIME_INC_PATH} ${GEN_INC_DIR} )
target_link_libraries( ${ROCTX_LIB} PRIVATE c stdc++ )
