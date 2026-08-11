set(ROCPROFILER_PC_SAMPLING_KERNEL_SOURCES
    pcs_common.cpp
    render_node_fd.cpp
    kfd_backend.cpp
    kgd_backend.cpp
    selector.cpp)

set(ROCPROFILER_PC_SAMPLING_KERNEL_HEADERS
    backend.hpp
    pcs_common.hpp
    render_node_fd.hpp
    kfd_backend.hpp
    selector.hpp)

target_sources(
    rocprofiler-sdk-object-library PRIVATE ${ROCPROFILER_PC_SAMPLING_KERNEL_SOURCES}
                                           ${ROCPROFILER_PC_SAMPLING_KERNEL_HEADERS})
