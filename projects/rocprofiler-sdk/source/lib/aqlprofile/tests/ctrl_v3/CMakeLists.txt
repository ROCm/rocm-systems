#
# ctrl_v3: aqlprofile v2 API PMC test using librocprofiler-sdk-aqlprofile.a
#

rocprofiler_deactivate_clang_tidy()

set(CTRL_V3_DIR ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(
    ctrl_v3
    ${CTRL_V3_DIR}/ctrl_v3.cpp
    ${CTRL_V3_DIR}/simple_convolution/simple_convolution.cpp
    ${CTRL_V3_DIR}/util/hsa_rsrc_factory.cpp)
# ctrl_v3 uses the standalone hsa_rsrc_factory.cpp from the aqlprofile test
# tree (direct HSA calls, no rocprofiler-sdk plugin init path). The rocprofiler-
# sdk static lib also contains hsa_rsrc_factory.cpp; --allow-multiple-definition
# lets the linker pick our copy (listed first) and ignore the lib's version.

target_include_directories(
    ctrl_v3
    PRIVATE
        # For ctrl/, simple_convolution/, util/ relative includes
        ${CTRL_V3_DIR}
        # util/ also needs to be on the path for hsa_rsrc_factory.h -> "hsa_includes.h"
        ${CTRL_V3_DIR}/util
        # For lib/aqlprofile/aqlprofile.hpp -> lib/aqlprofile/aql_profile_v2.h
        ${PROJECT_BINARY_DIR}/source
        ${PROJECT_SOURCE_DIR}/source)

target_link_libraries(
    ctrl_v3
    PRIVATE
        -Wl,--allow-multiple-definition
        -Wl,--start-group
        rocprofiler-sdk::rocprofiler-sdk-aqlprofile
        rocprofiler-sdk::rocprofiler-sdk-common-library
        -Wl,--end-group
        $<BUILD_INTERFACE:rocprofiler-sdk::rocprofiler-sdk-hsa-runtime-nolink>
        hsa-runtime64::hsa-runtime64
        pthread
        dl)

# ---- Build .hsaco kernels via clang ----
find_program(AQLPROFILE_CLANG_EXECUTABLE NAMES clang HINTS "${ROCM_PATH}/llvm/bin" /usr/bin)

if(AQLPROFILE_CLANG_EXECUTABLE)
    # Collect GPU targets from the project's GPU_TARGETS or AMDGPU_TARGETS variable
    if(NOT DEFINED CTRL_V3_GPU_TARGETS)
        if(DEFINED GPU_TARGETS)
            set(CTRL_V3_GPU_TARGETS ${GPU_TARGETS})
        elseif(DEFINED AMDGPU_TARGETS)
            set(CTRL_V3_GPU_TARGETS ${AMDGPU_TARGETS})
        endif()
    endif()

    set(CTRL_V3_HSACO_LIST)
    set(CTRL_V3_HSACO_HEADER_LIST)
    foreach(TARGET_ID ${CTRL_V3_GPU_TARGETS})
        set(HSACO_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${TARGET_ID}_simple_convolution.hsaco")
        set(HSACO_HEADER "${CMAKE_CURRENT_BINARY_DIR}/${TARGET_ID}_simple_convolution_hsaco.h")
        add_custom_command(
            OUTPUT "${HSACO_OUTPUT}"
            COMMAND
                ${AQLPROFILE_CLANG_EXECUTABLE}
                    -O2 -x cl
                    -Xclang -finclude-default-header
                    -cl-denorms-are-zero -cl-std=CL2.0
                    -Wl,--build-id=sha1
                    -target amdgcn-amd-amdhsa
                    -mcpu=${TARGET_ID}
                    -o "${HSACO_OUTPUT}"
                    "${CTRL_V3_DIR}/simple_convolution/simple_convolution.cl"
            DEPENDS "${CTRL_V3_DIR}/simple_convolution/simple_convolution.cl"
            COMMENT "Building ${TARGET_ID}_simple_convolution.hsaco"
            VERBATIM)
        # Convert the .hsaco binary to a C header using a CMake script so
        # we avoid shell-pipe portability issues in add_custom_command.
        string(MAKE_C_IDENTIFIER "${TARGET_ID}_simple_convolution" HSACO_SYM)
        set(HSACO2H_SCRIPT "${CMAKE_CURRENT_BINARY_DIR}/hsaco2h_${TARGET_ID}.cmake")
        file(GENERATE OUTPUT "${HSACO2H_SCRIPT}" CONTENT
"# hsaco2h: convert a binary file to a C byte-array header
file(READ \"\${HSACO}\" raw HEX)
string(LENGTH \"\${raw}\" raw_len)
math(EXPR n_bytes \"\${raw_len} / 2\")
# Prefix every two hex digits with '0x' and append ','.
# CMake's regex engine does not support (?:...) or {n} quantifiers,
# so line-wrapping is skipped — C compilers handle arbitrarily long array lines.
string(REGEX REPLACE \"([0-9a-fA-F][0-9a-fA-F])\" \"0x\\\\1,\" body \"\${raw}\")
file(WRITE \"\${OUT}\"
    \"#pragma once\\n\"
    \"const unsigned char \${SYM}_data[] = {\${body}};\\n\"
    \"const unsigned int \${SYM}_size = \${n_bytes};\\n\")
")
        add_custom_command(
            OUTPUT "${HSACO_HEADER}"
            COMMAND ${CMAKE_COMMAND}
                -D "HSACO=${HSACO_OUTPUT}"
                -D "SYM=${HSACO_SYM}"
                -D "OUT=${HSACO_HEADER}"
                -P "${HSACO2H_SCRIPT}"
            DEPENDS "${HSACO_OUTPUT}" "${HSACO2H_SCRIPT}"
            COMMENT "Generating embedded hsaco header for ${TARGET_ID}"
            VERBATIM)
        list(APPEND CTRL_V3_HSACO_LIST "${HSACO_OUTPUT}")
        list(APPEND CTRL_V3_HSACO_HEADER_LIST "${HSACO_HEADER}")
    endforeach()

    if(CTRL_V3_HSACO_LIST)
        # Generate the umbrella header that includes all per-target headers and
        # provides a lookup function used by ctrl_v3.cpp.
        # Use string(ASCII 59 SC) to get a literal semicolon — \; in CMake strings
        # keeps the backslash in the stored value, which file(WRITE) would emit literally.
        string(ASCII 59 SC)
        set(UMBRELLA_HEADER "${CMAKE_CURRENT_BINARY_DIR}/embedded_hsaco.h")
        file(WRITE "${UMBRELLA_HEADER}" "#pragma once\n#include <cstring>\n\n")
        foreach(HDR ${CTRL_V3_HSACO_HEADER_LIST})
            file(APPEND "${UMBRELLA_HEADER}" "#include \"${HDR}\"\n")
        endforeach()
        file(APPEND "${UMBRELLA_HEADER}"
            "\nstruct EmbeddedHsaco { const unsigned char* data${SC} unsigned int size${SC} }${SC}\n")
        file(APPEND "${UMBRELLA_HEADER}"
            "\ninline EmbeddedHsaco FindEmbeddedHsaco(const char* gfx) {\n")
        foreach(TARGET_ID ${CTRL_V3_GPU_TARGETS})
            string(MAKE_C_IDENTIFIER "${TARGET_ID}_simple_convolution" HSACO_SYM)
            file(APPEND "${UMBRELLA_HEADER}"
                "    if (strcmp(gfx, \"${TARGET_ID}\") == 0) return { ${HSACO_SYM}_data, ${HSACO_SYM}_size }${SC}\n")
        endforeach()
        file(APPEND "${UMBRELLA_HEADER}" "    return { nullptr, 0 }${SC}\n}\n")

        add_custom_target(ctrl_v3_hsaco ALL DEPENDS ${CTRL_V3_HSACO_LIST} ${CTRL_V3_HSACO_HEADER_LIST})
        add_dependencies(ctrl_v3 ctrl_v3_hsaco)
        target_include_directories(ctrl_v3 PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
        target_compile_definitions(ctrl_v3 PRIVATE CTRL_V3_EMBEDDED_HSACO)
    endif()
else()
    message(WARNING "ctrl_v3: clang not found — .hsaco kernels will not be built. "
                    "Set AQLPROFILE_CLANG_EXECUTABLE or ensure clang is on PATH.")
endif()
