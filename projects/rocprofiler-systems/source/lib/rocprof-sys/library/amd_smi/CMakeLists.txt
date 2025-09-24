#
set(smi_sources
    ${CMAKE_CURRENT_LIST_DIR}/amd_smi.cpp
)

set(smi_headers
    ${CMAKE_CURRENT_LIST_DIR}/amd_smi.hpp
    ${CMAKE_CURRENT_LIST_DIR}/amd_smi_driver.hpp
    ${CMAKE_CURRENT_LIST_DIR}/common.hpp
    ${CMAKE_CURRENT_LIST_DIR}/processor.hpp
    ${CMAKE_CURRENT_LIST_DIR}/service.hpp
)

target_sources(
    rocprofiler-systems-object-library
    PRIVATE ${smi_sources} ${smi_headers}
)
