#!/bin/bash
# Build the include tree that a configured CMake build would have generated, then print the
# compiler flags for an out-of-tree syntax check. Run from the rocprofiler-sdk project root:
#
#   FLAGS=$(.claude/skills/rocprof-sdk-cpu-only-testing/stub-includes.sh)
#   g++ -std=c++17 -fsyntax-only $FLAGS source/lib/rocprofiler-sdk/<path>.cpp
#
# Nothing here is suitable for running code: the comgr header has no implementation and the
# version macros are zeros. It is enough to compile, and for -Wall -Wextra -Wshadow.

set -eo pipefail

OUT=${OUT:-/tmp/rocprof-sdk-stubs}
SIBLINGS=${SIBLINGS:-..}
ROCR=${ROCR:-$SIBLINGS/rocr-runtime}

if [ ! -d source/lib/rocprofiler-sdk ]; then
    echo "run from the rocprofiler-sdk project root" >&2
    exit 1
fi
for d in "$ROCR/runtime/hsa-runtime/inc" "$SIBLINGS/hip/include" "$SIBLINGS/clr/hipamd/include"; do
    if [ ! -d "$d" ]; then
        echo "missing sibling checkout: $d (override with SIBLINGS= or ROCR=)" >&2
        exit 1
    fi
done

mkdir -p "$OUT"

# CMake configures these from templates. Substituting zeros is safe for a syntax check, but
# note that any header selecting struct layout on a version macro takes the version-0 branch.
while read -r template; do
    case "$template" in
        source/lib/aqlprofile/*) rel=lib/aqlprofile/version.h ;;
        *) rel=${template#source/include/}; rel=${rel%.in} ;;
    esac
    mkdir -p "$OUT/$(dirname "$rel")"
    sed -E -e 's/^#cmakedefine01[[:space:]]+([A-Za-z0-9_]+).*/#define \1 0/' \
        -e 's/^#cmakedefine[[:space:]]+([A-Za-z0-9_]+)(.*)/#define \1 \2/' \
        -e 's/@[A-Za-z0-9_]+@/0/g' "$template" >"$OUT/$rel"
done < <(find source -name '*version*.h.in')

# Sources include HSA as <hsa/...>, but rocr-runtime keeps those headers in inc/. hsa_api_trace.h
# then includes its siblings as <inc/...>, which is why hsa-runtime itself is also on the path.
ln -sfn "$(cd "$ROCR/runtime/hsa-runtime/inc" && pwd)" "$OUT/hsa"

# clr/hipamd/CMakeLists.txt emits this one inline rather than from a .in template.
mkdir -p "$OUT/hip"
cat >"$OUT/hip/hip_version.h" <<'EOF'
#pragma once
#define HIP_VERSION_MAJOR 0
#define HIP_VERSION_MINOR 0
#define HIP_VERSION_PATCH 0
#define HIP_VERSION_GITHASH "0"
#define HIP_VERSION_BUILD_ID 0
#define HIP_VERSION_BUILD_NAME "0"
#define HIP_VERSION                                                                                \
    (HIP_VERSION_MAJOR * 10000000 + HIP_VERSION_MINOR * 100000 + HIP_VERSION_PATCH)
EOF

# Not a version header: comgr is a real API surface, reached by anything that pulls in
# context/context.hpp, and there is no comgr checkout in this repository. Declarations only.
mkdir -p "$OUT/amd_comgr"
cat >"$OUT/amd_comgr/amd_comgr.h" <<'EOF'
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

    typedef enum amd_comgr_status_s
    {
        AMD_COMGR_STATUS_SUCCESS                = 0x0,
        AMD_COMGR_STATUS_ERROR                  = 0x1,
        AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT = 0x2,
        AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES = 0x3
    } amd_comgr_status_t;

    typedef enum amd_comgr_data_kind_s
    {
        AMD_COMGR_DATA_KIND_UNDEF      = 0x0,
        AMD_COMGR_DATA_KIND_EXECUTABLE = 0x6
    } amd_comgr_data_kind_t;

    typedef enum amd_comgr_symbol_type_s
    {
        AMD_COMGR_SYMBOL_TYPE_UNKNOWN = -1,
        AMD_COMGR_SYMBOL_TYPE_NOTYPE  = 0,
        AMD_COMGR_SYMBOL_TYPE_OBJECT  = 1,
        AMD_COMGR_SYMBOL_TYPE_FUNC    = 2
    } amd_comgr_symbol_type_t;

    typedef enum amd_comgr_symbol_info_s
    {
        AMD_COMGR_SYMBOL_INFO_NAME_LENGTH = 0x0,
        AMD_COMGR_SYMBOL_INFO_NAME        = 0x1,
        AMD_COMGR_SYMBOL_INFO_TYPE        = 0x2,
        AMD_COMGR_SYMBOL_INFO_SIZE        = 0x3,
        AMD_COMGR_SYMBOL_INFO_VALUE       = 0x5
    } amd_comgr_symbol_info_t;

    typedef struct amd_comgr_data_s
    {
        uint64_t handle;
    } amd_comgr_data_t;

    typedef struct amd_comgr_symbol_s
    {
        uint64_t handle;
    } amd_comgr_symbol_t;

    typedef struct amd_comgr_disassembly_info_s
    {
        uint64_t handle;
    } amd_comgr_disassembly_info_t;

    amd_comgr_status_t amd_comgr_status_string(amd_comgr_status_t status,
                                              const char**       status_string);
    amd_comgr_status_t amd_comgr_create_data(amd_comgr_data_kind_t kind, amd_comgr_data_t* data);
    amd_comgr_status_t amd_comgr_release_data(amd_comgr_data_t data);
    amd_comgr_status_t amd_comgr_set_data(amd_comgr_data_t data, size_t size, const char* bytes);
    amd_comgr_status_t amd_comgr_get_data_isa_name(amd_comgr_data_t data,
                                                   size_t*          size,
                                                   char*            isa_name);
    amd_comgr_status_t amd_comgr_create_disassembly_info(
        const char* isa_name,
        uint64_t (*read_memory_callback)(uint64_t, char*, uint64_t, void*),
        void (*print_instruction_callback)(const char*, void*),
        void (*print_address_annotation_callback)(uint64_t, void*),
        amd_comgr_disassembly_info_t* disassembly_info);
    amd_comgr_status_t amd_comgr_destroy_disassembly_info(
        amd_comgr_disassembly_info_t disassembly_info);
    amd_comgr_status_t amd_comgr_disassemble_instruction(
        amd_comgr_disassembly_info_t disassembly_info,
        uint64_t                     address,
        void*                        user_data,
        uint64_t*                    size);
    amd_comgr_status_t amd_comgr_iterate_symbols(amd_comgr_data_t data,
                                                 amd_comgr_status_t (*callback)(amd_comgr_symbol_t,
                                                                               void*),
                                                 void* user_data);
    amd_comgr_status_t amd_comgr_symbol_get_info(amd_comgr_symbol_t      symbol,
                                                 amd_comgr_symbol_info_t attribute,
                                                 void*                   value);
    amd_comgr_status_t amd_comgr_map_elf_virtual_address_to_code_object_offset(
        amd_comgr_data_t data,
        uint64_t         elf_virtual_address,
        uint64_t*        code_object_offset,
        uint64_t*        slice_size,
        bool*            nobits);

#ifdef __cplusplus
}
#endif
EOF

echo "-D__HIP_PLATFORM_AMD__ -DFMT_HEADER_ONLY=1" \
    "-I source -I source/include -I $OUT" \
    "-I $ROCR/libhsakmt/include -I $ROCR/runtime/hsa-runtime" \
    "-I $SIBLINGS/hip/include -I $SIBLINGS/clr/hipamd/include" \
    "-I external/abseil-cpp -I external/fmt/include" \
    "-I external/googletest/googletest/include -I external/googletest/googlemock/include"
