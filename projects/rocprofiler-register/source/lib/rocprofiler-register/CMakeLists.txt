#
#   builds the rocprofiler-register library
#
add_library(rocprofiler-register SHARED)
add_library(rocprofiler-register::rocprofiler-register ALIAS rocprofiler-register)

add_subdirectory(details)

target_sources(rocprofiler-register PRIVATE rocprofiler_register.cpp)

if(ROCPROFILER_REGISTER_BUILD_TESTS)
    # make sure header is C-compatible
    target_sources(rocprofiler-register PRIVATE rocprofiler_register.c)
endif()

target_include_directories(
    rocprofiler-register PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/source
                                 ${PROJECT_BINARY_DIR}/source)

target_link_libraries(
    rocprofiler-register
    PUBLIC rocprofiler-register::headers
    PRIVATE fmt::fmt glog::glog rocprofiler-register::build-flags
            rocprofiler-register::memcheck rocprofiler-register::stdcxxfs
            rocprofiler-register::dl)

set_target_properties(
    rocprofiler-register PROPERTIES OUTPUT_NAME rocprofiler-register)

# WINDOWS-DIVERGENCE: PE/COFF DLLs do not carry SO version metadata in the
# filename — ABI versioning lives in the VS_VERSION_INFO resource block.
# Setting SOVERSION/VERSION on Windows produces an unwanted versioned DLL
# name (e.g. rocprofiler-register-1.dll) that downstream LoadLibrary calls
# would not find.
if(NOT WIN32)
    set_target_properties(
        rocprofiler-register PROPERTIES SOVERSION ${PROJECT_VERSION_MAJOR}
                                        VERSION ${PROJECT_VERSION})
endif()

if(ROCPROFILER_REGISTER_BUILD_DEFAULT_ATTACHMENT)
    target_compile_definitions(rocprofiler-register PRIVATE ROCP_REG_DEFAULT_ATTACHMENT=1)
endif()

# Explicit export-controlling macro: in addition to CMake's automatically-defined
# rocprofiler_register_EXPORTS, set ROCPROFILER_REGISTER_EXPORTS so version.h.in
# can choose between __declspec(dllexport) and __declspec(dllimport) on Windows.
target_compile_definitions(rocprofiler-register PRIVATE ROCPROFILER_REGISTER_EXPORTS)

# Explicit per-artifact destinations so the install layout is correct on
# both Linux and Windows: on Linux the SHARED library and its versioned
# symlinks land in ${CMAKE_INSTALL_LIBDIR}; on Windows the .dll lands in
# ${CMAKE_INSTALL_BINDIR} and the import .lib lands in
# ${CMAKE_INSTALL_LIBDIR}.
install(
    TARGETS rocprofiler-register
    COMPONENT core
    EXPORT ${PROJECT_NAME}-library-targets
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
