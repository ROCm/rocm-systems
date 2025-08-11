# ------------------------------------------------------------------------------#
#
# rocprofiler-systems-avail target
#
# ------------------------------------------------------------------------------#

add_executable(rocprofiler-systems-avail)

target_sources(
    rocprofiler-systems-avail
    PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/avail.cpp
        ${CMAKE_CURRENT_LIST_DIR}/avail.hpp
        ${CMAKE_CURRENT_LIST_DIR}/common.cpp
        ${CMAKE_CURRENT_LIST_DIR}/common.hpp
        ${CMAKE_CURRENT_LIST_DIR}/component_categories.hpp
        ${CMAKE_CURRENT_LIST_DIR}/defines.hpp
        ${CMAKE_CURRENT_LIST_DIR}/enumerated_list.hpp
        ${CMAKE_CURRENT_LIST_DIR}/generate_config.cpp
        ${CMAKE_CURRENT_LIST_DIR}/generate_config.hpp
        ${CMAKE_CURRENT_LIST_DIR}/get_availability.hpp
        ${CMAKE_CURRENT_LIST_DIR}/get_categories.hpp
        ${CMAKE_CURRENT_LIST_DIR}/info_type.cpp
        ${CMAKE_CURRENT_LIST_DIR}/info_type.hpp
)

target_include_directories(rocprofiler-systems-avail PRIVATE ${CMAKE_CURRENT_LIST_DIR})
target_compile_definitions(
    rocprofiler-systems-avail
    PRIVATE ROCPROFSYS_EXTERN_COMPONENTS=0
)
target_link_libraries(
    rocprofiler-systems-avail
    PRIVATE
        rocprofiler-systems::rocprofiler-systems-compile-definitions
        rocprofiler-systems::rocprofiler-systems-interface-library
        rocprofiler-systems::librocprofiler-systems-static
)
set_target_properties(
    rocprofiler-systems-avail
    PROPERTIES
        BUILD_RPATH "\$ORIGIN:\$ORIGIN/../${CMAKE_INSTALL_LIBDIR}"
        INSTALL_RPATH "${ROCPROFSYS_EXE_INSTALL_RPATH}"
        OUTPUT_NAME ${BINARY_NAME_PREFIX}-avail
)

rocprofiler_systems_strip_target(rocprofiler-systems-avail)

install(TARGETS rocprofiler-systems-avail DESTINATION ${CMAKE_INSTALL_BINDIR} OPTIONAL)
