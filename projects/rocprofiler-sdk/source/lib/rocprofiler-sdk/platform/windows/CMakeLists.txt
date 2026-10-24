#
# Native Windows D3DKMT + wkmi agent enumeration. Only added to the build when
# WIN32 is true (gated by the parent platform/CMakeLists.txt).
#

# ----------------------------------------------------------------------------------------#
#
# Option: link the AMD-private wkmi for detailed (KFD-parity) topology fields. When OFF
# (or when no prebuilt wkmi is available) the wkmi bridge compiles to a stub returning
# false and the enumerator degrades to the basic D3DKMT-only fields.
#
# ----------------------------------------------------------------------------------------#
option(ROCPROFILER_WINDOWS_USE_WKMI
       "Link the AMD-private wkmi for detailed Windows GPU topology fields" ON)

# ----------------------------------------------------------------------------------------#
#
# INTERFACE target carrying the vendored WDK D3DKMT thunk headers and the wkmi include dir
# + static library. Linked PRIVATE onto the object library below.
#
# The vendored d3dkmt headers are consumed via source-root-relative includes (#include
# "lib/rocprofiler-sdk/platform/windows/external/d3dkmt/...") so they need no extra
# include dir; the dir is added anyway so the short "external/d3dkmt/..." form also
# resolves and to honor the documented design.
#
# ----------------------------------------------------------------------------------------#
add_library(rocprofiler-sdk-wkmi INTERFACE)
add_library(rocprofiler-sdk::rocprofiler-sdk-wkmi ALIAS rocprofiler-sdk-wkmi)

target_include_directories(
    rocprofiler-sdk-wkmi INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}"
                                   "${CMAKE_CURRENT_SOURCE_DIR}/external/d3dkmt")

# Prebuilt wkmi drop. rocprofiler-sdk does NOT build wkmi from source: that needs the
# libhsakmt/hsa-runtime build context, which this project's standalone build never sets
# up. Instead it links rocr-runtime's (libhsakmt's) committed wkmi drop: rocr commits
# projects/rocr-runtime/libhsakmt/src/dxg/wkmi/{wkmi.h,win/dbg/wkmi.lib,win/rel/wkmi.lib}
# which is a sibling of this project in the rocm-systems monorepo. The superbuild can
# override ROCPROFILER_WKMI_ROOT to point at another drop (e.g. an installed layout); a
# standalone build without that tree degrades to the stub. NOTE: the wkmi.lib CRT/runtime
# and compiler MUST match the rocprofiler-sdk Windows build to avoid CRT-mismatch crashes
# at runtime; the build does not (and cannot here) verify that - it is the packager's
# responsibility.
set(_ROCPROFILER_WKMI_ROOT_DEFAULT
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../../rocr-runtime/libhsakmt/src/dxg/wkmi")
set(ROCPROFILER_WKMI_ROOT
    "${_ROCPROFILER_WKMI_ROOT_DEFAULT}"
    CACHE PATH "Root of a prebuilt wkmi drop (expects wkmi.h + win/{dbg,rel}/wkmi.lib)")

# Explicit single-lib override (wins over ROCPROFILER_WKMI_ROOT). Its header is still
# resolved from ROCPROFILER_WKMI_ROOT, so set both when pointing elsewhere.
set(WKMI_PREBUILT_LIB
    ""
    CACHE
        FILEPATH
        "Path to a prebuilt wkmi.lib (overrides the per-config ROCPROFILER_WKMI_ROOT defaults)"
    )

if(ROCPROFILER_WINDOWS_USE_WKMI)
    # rocr's header sits directly in ROCPROFILER_WKMI_ROOT (libhsakmt uses that dir as the
    # include dir), so the include dir is the root itself.
    set(_rocprofiler_wkmi_lib "${WKMI_PREBUILT_LIB}")
    set(_rocprofiler_wkmi_inc "${ROCPROFILER_WKMI_ROOT}")
    if(NOT _rocprofiler_wkmi_lib)
        # Mirror the libhsakmt dbg/rel layout under ROCPROFILER_WKMI_ROOT.
        set(_rocprofiler_wkmi_lib_dbg "${ROCPROFILER_WKMI_ROOT}/win/dbg/wkmi.lib")
        set(_rocprofiler_wkmi_lib_rel "${ROCPROFILER_WKMI_ROOT}/win/rel/wkmi.lib")
        set(_rocprofiler_wkmi_lib
            "$<IF:$<CONFIG:Debug>,${_rocprofiler_wkmi_lib_dbg},${_rocprofiler_wkmi_lib_rel}>"
            )
    endif()

    # Determine whether any prebuilt lib actually exists. Generator expressions cannot be
    # EXISTS-checked, so probe the concrete per-config paths (and the explicit override)
    # directly.
    set(_rocprofiler_have_prebuilt OFF)
    if(WKMI_PREBUILT_LIB AND EXISTS "${WKMI_PREBUILT_LIB}")
        set(_rocprofiler_have_prebuilt ON)
    elseif(EXISTS "${ROCPROFILER_WKMI_ROOT}/win/dbg/wkmi.lib"
           OR EXISTS "${ROCPROFILER_WKMI_ROOT}/win/rel/wkmi.lib")
        set(_rocprofiler_have_prebuilt ON)
    endif()

    if(_rocprofiler_have_prebuilt AND EXISTS "${_rocprofiler_wkmi_inc}/wkmi.h")
        message(
            STATUS
                "rocprofiler-sdk: using prebuilt wkmi.lib from ${ROCPROFILER_WKMI_ROOT}")
        target_link_libraries(rocprofiler-sdk-wkmi INTERFACE "${_rocprofiler_wkmi_lib}")
        target_include_directories(rocprofiler-sdk-wkmi
                                   INTERFACE "${_rocprofiler_wkmi_inc}")
        target_compile_definitions(rocprofiler-sdk-wkmi
                                   INTERFACE ROCPROFILER_WINDOWS_USE_WKMI=1)
        # The prebuilt wkmi.lib is not self-contained: its inline DxcoreLoader::Instance()
        # leaves the DxcoreLoader ctor undefined and never initializes the D3DKMT thunks
        # it calls (that is the host runtime's job in the libhsakmt/hsa-runtime build).
        # Compile wkmi_dxcore_loader.cpp below to supply the ctor and resolve those thunks
        # from gdi32.dll. Only needed for the prebuilt path; the stub does not link this
        # orphaned external.
        set(_rocprofiler_wkmi_use_dxcore_loader ON)
    else()
        # ---- option ON but no prebuilt available: stub ----
        message(
            STATUS
                "rocprofiler-sdk: ROCPROFILER_WINDOWS_USE_WKMI=ON but no prebuilt wkmi found; "
                "compiling wkmi bridge as a stub (D3DKMT-only basic fields)")
        target_compile_definitions(rocprofiler-sdk-wkmi
                                   INTERFACE ROCPROFILER_WINDOWS_USE_WKMI=0)
    endif()
else()
    # ---- option OFF: stub ----
    message(
        STATUS "rocprofiler-sdk: ROCPROFILER_WINDOWS_USE_WKMI=OFF; wkmi bridge is a stub")
    target_compile_definitions(rocprofiler-sdk-wkmi
                               INTERFACE ROCPROFILER_WINDOWS_USE_WKMI=0)
endif()

# ----------------------------------------------------------------------------------------#
#
# Sources owned by this directory.
#
# ----------------------------------------------------------------------------------------#
set(ROCPROFILER_PLATFORM_WINDOWS_HEADERS agent.hpp d3dkmt_loader.hpp wkmi_bridge.hpp)
set(ROCPROFILER_PLATFORM_WINDOWS_SOURCES agent.cpp d3dkmt_loader.cpp wkmi_bridge.cpp)

target_sources(
    rocprofiler-sdk-object-library PRIVATE ${ROCPROFILER_PLATFORM_WINDOWS_SOURCES}
                                           ${ROCPROFILER_PLATFORM_WINDOWS_HEADERS})

# Self-containment translation unit for the prebuilt wkmi.lib path (see the wkmi branch
# above). Compiled only when the prebuilt lib is actually linked.
if(_rocprofiler_wkmi_use_dxcore_loader)
    target_sources(rocprofiler-sdk-object-library PRIVATE wkmi_dxcore_loader.cpp)
endif()

# Carry the wkmi/d3dkmt include dirs, lib, and ROCPROFILER_WINDOWS_USE_WKMI definition
# onto the object library. PRIVATE: this is an internal implementation detail, not part of
# the public rocprofiler-sdk interface.
target_link_libraries(rocprofiler-sdk-object-library
                      PRIVATE rocprofiler-sdk::rocprofiler-sdk-wkmi)

if(ROCPROFILER_BUILD_TESTS)
    add_subdirectory(tests)
endif()
