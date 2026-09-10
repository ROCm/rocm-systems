# The executable and its two kernel libraries are colocated in one directory so
# the driver can resolve the libraries relative to its own path.
set(CG_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/tests/conjugate_gradient)

# Each kernel is its own shared library: kernels compiled into a single binary
# share one fat binary, and so one code object, which is exactly what this
# sample needs to avoid.
foreach(CG_KERNEL_LIBRARY IN ITEMS cg_spmv cg_update)
    set_source_files_properties(${CG_KERNEL_LIBRARY}.hip PROPERTIES LANGUAGE HIP)
    add_library(${CG_KERNEL_LIBRARY} SHARED ${CG_KERNEL_LIBRARY}.hip)
    target_include_directories(${CG_KERNEL_LIBRARY} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    # PC sampling maps samples back to source lines, which needs debug info.
    target_compile_options(${CG_KERNEL_LIBRARY} PRIVATE -g)
    set_target_properties(
        ${CG_KERNEL_LIBRARY}
        PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CG_OUTPUT_DIRECTORY}
    )
endforeach()

set_source_files_properties(conjugate_gradient.cpp PROPERTIES LANGUAGE HIP)
add_executable(conjugate_gradient conjugate_gradient.cpp)
target_include_directories(conjugate_gradient PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(conjugate_gradient PRIVATE -g)
target_link_libraries(conjugate_gradient PRIVATE ${CMAKE_DL_LIBS})
set_target_properties(
    conjugate_gradient
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CG_OUTPUT_DIRECTORY}
)

# The libraries are opened by name at runtime, not linked, so CMake supplies the
# file names and a target rename cannot desync them from the source.
target_compile_definitions(
    conjugate_gradient
    PRIVATE
        CG_SPMV_LIBRARY="$<TARGET_FILE_NAME:cg_spmv>"
        CG_UPDATE_LIBRARY="$<TARGET_FILE_NAME:cg_update>"
)
add_dependencies(conjugate_gradient cg_spmv cg_update)
