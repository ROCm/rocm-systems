#
# Platform-abstraction sources for rocprofiler-register.
#
# loader.hpp / pe_parser.hpp expose a thin platform-neutral API used by the
# rest of the library. Exactly one implementation pair is compiled per
# platform: *_posix.cpp on Linux, *_win.cpp on Windows. pe_parser_posix.cpp
# is a stub since the equivalent metadata on Linux comes from
# /proc/<pid>/maps via loader_posix.cpp::get_segment_addresses.
#

set(rocprofiler_register_platform_headers loader.hpp pe_parser.hpp)

if(WIN32)
    list(APPEND rocprofiler_register_platform_headers encoding_win.hpp)
    set(rocprofiler_register_platform_sources loader_win.cpp pe_parser_win.cpp)
else()
    set(rocprofiler_register_platform_sources loader_posix.cpp pe_parser_posix.cpp)
endif()

target_sources(
    rocprofiler-register PRIVATE ${rocprofiler_register_platform_sources}
                                 ${rocprofiler_register_platform_headers})

if(WIN32)
    # WINDOWS-DIVERGENCE: psapi supplies EnumProcessModulesEx,
    # GetModuleFileNameExW, and GetModuleInformation used by the Win32
    # loader implementation.
    target_link_libraries(rocprofiler-register PRIVATE psapi)
endif()
