# RCCL src/algorithms/dda/CMakeLists.txt
#
# >>> RCCL-DELTA: DDA headers and support files consolidated under src/algorithms/dda/
# (develop commit ae0a12ac88). Listed here for the hipify pipeline instead of the old
# src/include/algorithms/ and flat src/include/dda_* paths.

set(DDA_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/device/CollCommon.h
    ${CMAKE_CURRENT_SOURCE_DIR}/device/CollCommon_ll128.h
    ${CMAKE_CURRENT_SOURCE_DIR}/all_reduce/all_reduce_dda.h
    ${CMAKE_CURRENT_SOURCE_DIR}/all_reduce/all_reduce_dda_ll.h
    ${CMAKE_CURRENT_SOURCE_DIR}/all_reduce/all_reduce_dda_ll_twoshot.h
    ${CMAKE_CURRENT_SOURCE_DIR}/all_reduce/all_reduce_dda_fabric_ll128.h
    ${CMAKE_CURRENT_SOURCE_DIR}/reduce_scatter/reduce_scatter_dda.h
    ${CMAKE_CURRENT_SOURCE_DIR}/all_gather/all_gather_dda.h
    ${CMAKE_CURRENT_SOURCE_DIR}/alltoall/alltoall_dda.h
    ${CMAKE_CURRENT_SOURCE_DIR}/all_reduce/all_reduce_dda_fabric.h
    ${CMAKE_CURRENT_SOURCE_DIR}/reduce_scatter/reduce_scatter_dda_fabric.h
    ${CMAKE_CURRENT_SOURCE_DIR}/all_gather/all_gather_dda_fabric.h
    ${CMAKE_CURRENT_SOURCE_DIR}/all_gather/all_gather_dda_fabric_ll.h
    ${CMAKE_CURRENT_SOURCE_DIR}/all_gather/all_gather_dda_fabric_ll128.h
    ${CMAKE_CURRENT_SOURCE_DIR}/alltoall/alltoall_dda_fabric.h
    ${CMAKE_CURRENT_SOURCE_DIR}/alltoall/alltoall_dda_fabric_ll.h
    ${CMAKE_CURRENT_SOURCE_DIR}/alltoall/alltoall_dda_fabric_ll128.h
    ${CMAKE_CURRENT_SOURCE_DIR}/reduce_scatter/reduce_scatter_dda_fabric_ll.h
    ${CMAKE_CURRENT_SOURCE_DIR}/reduce_scatter/reduce_scatter_dda_fabric_ll128.h
    ${CMAKE_CURRENT_SOURCE_DIR}/all_reduce/dda_all_reduce.h
    ${CMAKE_CURRENT_SOURCE_DIR}/reduce_scatter/dda_reduce_scatter.h
    ${CMAKE_CURRENT_SOURCE_DIR}/all_gather/dda_all_gather.h
    ${CMAKE_CURRENT_SOURCE_DIR}/alltoall/dda_alltoall.h
    ${CMAKE_CURRENT_SOURCE_DIR}/dda_init_detail.h
    ${CMAKE_CURRENT_SOURCE_DIR}/device/device_buffer.h
    ${CMAKE_CURRENT_SOURCE_DIR}/fabric/fabric_init.h
    ${CMAKE_CURRENT_SOURCE_DIR}/fabric/fabric_gpu_barrier.h
    ${CMAKE_CURRENT_SOURCE_DIR}/fabric/fabric_mem_handler.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ipc/ipc_gpu_barrier.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ipc/ipc_init.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ipc/ipc_mem_handler.h
)

set(DDA_HEADERS ${DDA_HEADERS} PARENT_SCOPE)
