# Device Linker Build - Simplified CMakeLists.txt
# 
# Prerequisites: Run regular cmake first to get hipified sources:
#   cmake -B build/release -DBUILD_LOCAL_GPU_TARGET_ONLY=ON -DGPU_TARGETS=gfx942 ..
#
# Then build device linker version:
#   cmake -B build/device_linker -S tools/device_linker -DBASE_BUILD=build/release
#   cmake --build build/device_linker
#
cmake_minimum_required(VERSION 3.21)
project(rccl_device_linker LANGUAGES CXX HIP)

get_filename_component(RCCL_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)

# Require BASE_BUILD to point to existing build with hipified sources (relative to RCCL_ROOT)
set(BASE_BUILD "" CACHE STRING "Path to existing RCCL build relative to RCCL root (e.g., build/release)")
set(BASE_BUILD_FULL "${RCCL_ROOT}/${BASE_BUILD}")
if(NOT BASE_BUILD OR NOT EXISTS "${BASE_BUILD_FULL}/hipify")
  message(FATAL_ERROR "BASE_BUILD must point to existing RCCL build directory with hipified sources. "
          "Got: ${BASE_BUILD_FULL}/hipify "
          "Run: cmake -B build/release .. first, then: cmake -DBASE_BUILD=build/release ...")
endif()

set(HIPIFY_DIR "${BASE_BUILD_FULL}/hipify")
set(GEN_DIR ${CMAKE_CURRENT_BINARY_DIR}/gensrc)
set(DEVICE_LINKER_DIR ${CMAKE_CURRENT_SOURCE_DIR})

message(STATUS "Using hipified sources from: ${HIPIFY_DIR}")

# Find Python
find_package(Python3 REQUIRED COMPONENTS Interpreter)

# GPU target
set(GPU_TARGET "gfx942" CACHE STRING "GPU target architecture")
set(GPU_TARGET_FULL "${GPU_TARGET}:xnack+:sramecc+")
set(BUNDLE_TARGET "hipv4-amdgcn-amd-amdhsa--${GPU_TARGET}:sramecc+:xnack+")

message(STATUS "GPU target: ${GPU_TARGET_FULL}")
message(STATUS "Bundle target: ${BUNDLE_TARGET}")

# Find HIP
find_package(hip REQUIRED)

# Tools
set(HIPCC /opt/rocm/bin/hipcc)
set(LLVM_OBJCOPY /opt/rocm/llvm/bin/llvm-objcopy)
set(CLANG_BUNDLER /opt/rocm/llvm/bin/clang-offload-bundler)

# Use generated files from base build
set(GEN_DIR "${BASE_BUILD_FULL}/hipify/gensrc")
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/specialized)

# Generate specialized kernels (only thing we need to regenerate)
message(STATUS "Generating specialized kernels...")
execute_process(
  COMMAND ${Python3_EXECUTABLE} ${RCCL_ROOT}/src/device/generate_specialized.py 
          ${CMAKE_CURRENT_BINARY_DIR}/specialized ON "" "${GPU_TARGET}"
  WORKING_DIRECTORY ${RCCL_ROOT}
  RESULT_VARIABLE gen_spec_result
  OUTPUT_VARIABLE gen_spec_output
)
if(gen_spec_result)
  message(FATAL_ERROR "generate_specialized.py failed")
endif()
message(STATUS "${gen_spec_output}")

# Collect specialized kernel files
file(GLOB SPECIALIZED_FILES "${CMAKE_CURRENT_BINARY_DIR}/specialized/*.cpp")
list(LENGTH SPECIALIZED_FILES SPEC_COUNT)
message(STATUS "Found ${SPEC_COUNT} specialized kernel files")

# Collect host source files from hipified directory
file(GLOB HOST_SOURCES_CC "${HIPIFY_DIR}/src/*.cc")
file(GLOB HOST_SOURCES_GRAPH "${HIPIFY_DIR}/src/graph/*.cc")
file(GLOB HOST_SOURCES_MISC "${HIPIFY_DIR}/src/misc/*.cc")
file(GLOB HOST_SOURCES_TRANSPORT "${HIPIFY_DIR}/src/transport/*.cc")
file(GLOB HOST_SOURCES_REGISTER "${HIPIFY_DIR}/src/register/*.cc")
file(GLOB HOST_SOURCES_SCHEDULER "${HIPIFY_DIR}/src/scheduler/*.cc")
file(GLOB HOST_SOURCES_RAS "${HIPIFY_DIR}/src/ras/*.cc")

set(HOST_SOURCES 
  ${HOST_SOURCES_CC}
  ${HOST_SOURCES_GRAPH}
  ${HOST_SOURCES_MISC}
  ${HOST_SOURCES_TRANSPORT}
  ${HOST_SOURCES_REGISTER}
  ${HOST_SOURCES_SCHEDULER}
  ${HOST_SOURCES_RAS}
)

# Add generated host_table.cpp and device_table.cpp from base build
list(APPEND HOST_SOURCES "${GEN_DIR}/host_table.cpp")
list(APPEND HOST_SOURCES "${GEN_DIR}/device_table.cpp")

# Create stub for symmetric kernels
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/sym_kernels_stub.cpp"
"#include \"nccl.h\"
extern \"C\" void* ncclSymkGetKernelPtr(int, int, int, ncclDataType_t, ncclRedOp_t) { return nullptr; }
extern \"C\" int ncclSymKernelCount = 0;
extern \"C\" void* ncclSymKernelList[] = {};
")
list(APPEND HOST_SOURCES "${CMAKE_CURRENT_BINARY_DIR}/sym_kernels_stub.cpp")

list(LENGTH HOST_SOURCES HOST_COUNT)
message(STATUS "Found ${HOST_COUNT} host source files")

# Include directories (match main build)
set(INCLUDE_DIRS
  ${BASE_BUILD_FULL}/include
  ${HIPIFY_DIR}/src
  ${HIPIFY_DIR}/src/device
  ${HIPIFY_DIR}/src/device/network/unpack
  ${HIPIFY_DIR}/src/include
  ${HIPIFY_DIR}/src/include/mlx5
  ${HIPIFY_DIR}/src/include/nccl_device
  ${HIPIFY_DIR}/src/include/ionic
  ${HIPIFY_DIR}/src/include/plugin
  ${GEN_DIR}
  ${GEN_DIR}/specialized
  /opt/rocm/include
  /opt/rocm/include/hsa
)

# Compile flags
set(COMMON_FLAGS 
  -fPIC
  -D__HIP_PLATFORM_AMD__
  -DDEVICE_LINKER
  -DRCCL_ROCM_BUILD
  -O3
)

# Step 5: Compile specialized kernels to .o files (NOT linked into rccl)
set(SPECIALIZED_OBJECTS "")
foreach(src ${SPECIALIZED_FILES})
  get_filename_component(name ${src} NAME_WE)
  set(obj ${CMAKE_CURRENT_BINARY_DIR}/specialized_objs/${name}.o)
  list(APPEND SPECIALIZED_OBJECTS ${obj})
  
  # Build include flags string
  set(INC_FLAGS "")
  foreach(dir ${INCLUDE_DIRS})
    set(INC_FLAGS "${INC_FLAGS} -I${dir}")
  endforeach()
  
  add_custom_command(
    OUTPUT ${obj}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/specialized_objs
    COMMAND ${HIPCC} -c -fPIC -fno-gpu-rdc --offload-arch=${GPU_TARGET_FULL}
            -DDEVICE_LINKER -D__HIP_PLATFORM_AMD__ -DENABLE_FAULT_INJECTION
            -I${BASE_BUILD_FULL}/include
            -I${HIPIFY_DIR}/src
            -I${HIPIFY_DIR}/src/device
            -I${HIPIFY_DIR}/src/device/network/unpack
            -I${HIPIFY_DIR}/src/include
            -I${HIPIFY_DIR}/src/include/mlx5
            -I${HIPIFY_DIR}/src/include/nccl_device
            -I${HIPIFY_DIR}/src/include/ionic
            -I${HIPIFY_DIR}/src/include/plugin
            -I${GEN_DIR}
            -I/opt/rocm/include
            -I/opt/rocm/include/hsa
            ${src} -o ${obj}
    DEPENDS ${src}
    COMMENT "Compiling specialized kernel: ${name}"
  )
endforeach()

add_custom_target(specialized_kernels ALL DEPENDS ${SPECIALIZED_OBJECTS})

# Step 6: Compile merged_minimal.hip (the dispatcher)
set(DISPATCHER_OBJ ${CMAKE_CURRENT_BINARY_DIR}/merged_minimal.o)
add_custom_command(
  OUTPUT ${DISPATCHER_OBJ}
  COMMAND ${HIPCC} -c -fPIC -fno-gpu-rdc --offload-arch=${GPU_TARGET_FULL}
          -I${BASE_BUILD_FULL}/include
          -I${HIPIFY_DIR}/src
          -I${HIPIFY_DIR}/src/device
          -I${HIPIFY_DIR}/src/device/network/unpack
          -I${HIPIFY_DIR}/src/include
          -I${HIPIFY_DIR}/src/include/mlx5
          -I${HIPIFY_DIR}/src/include/nccl_device
          -I${HIPIFY_DIR}/src/include/ionic
          -I${HIPIFY_DIR}/src/include/plugin
          -I${GEN_DIR}
          -I/opt/rocm/include
          -I/opt/rocm/include/hsa
          ${DEVICE_LINKER_DIR}/merged_minimal.hip -o ${DISPATCHER_OBJ}
  DEPENDS ${DEVICE_LINKER_DIR}/merged_minimal.hip
  COMMENT "Compiling dispatcher (merged_minimal.hip)"
)

# Step 7: Build skeleton librccl.so (host code + dispatcher, NO specialized device code)
add_library(rccl SHARED ${HOST_SOURCES})
target_include_directories(rccl PRIVATE ${INCLUDE_DIRS})
target_compile_definitions(rccl PRIVATE DEVICE_LINKER __HIP_PLATFORM_AMD__ RCCL_ROCM_BUILD ENABLE_FAULT_INJECTION)
target_compile_options(rccl PRIVATE -fPIC -O3)
target_link_libraries(rccl PRIVATE hip::host pthread dl rt)
target_link_options(rccl PRIVATE ${DISPATCHER_OBJ})

# Make sure dispatcher is built before rccl
add_custom_target(dispatcher DEPENDS ${DISPATCHER_OBJ})
add_dependencies(rccl dispatcher)

# Step 8: Run device linker (after specialized kernels are compiled)
set(MERGED_DEVICE ${CMAKE_CURRENT_BINARY_DIR}/merged_device_final.o)
add_custom_command(
  OUTPUT ${MERGED_DEVICE}
  # Extract device code from dispatcher
  COMMAND ${LLVM_OBJCOPY} --dump-section=.hip_fatbin=${CMAKE_CURRENT_BINARY_DIR}/fatbin.bin ${DISPATCHER_OBJ}
  COMMAND ${CLANG_BUNDLER} --type=bc
          --targets=host-x86_64-unknown-linux-gnu-,${BUNDLE_TARGET}
          --input=${CMAKE_CURRENT_BINARY_DIR}/fatbin.bin
          --output=/dev/null --output=${CMAKE_CURRENT_BINARY_DIR}/minimal_device.o
          --unbundle
  # Run device linker
  COMMAND ${DEVICE_LINKER_DIR}/device_linker
          -o ${MERGED_DEVICE}
          --dispatcher ${CMAKE_CURRENT_BINARY_DIR}/minimal_device.o
          --host-table ${GEN_DIR}/host_table.cpp
          --target ${GPU_TARGET}:sramecc+:xnack+
          --input-dir ${CMAKE_CURRENT_BINARY_DIR}/specialized_objs/
  DEPENDS specialized_kernels ${DISPATCHER_OBJ} ${DEVICE_LINKER_DIR}/device_linker
  COMMENT "Running device linker to create merged device image"
)

add_custom_target(device_link DEPENDS ${MERGED_DEVICE})

# Step 9: Replace .hip_fatbin in librccl.so with merged device code (POST_BUILD)
add_custom_command(
  TARGET rccl POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E echo "Replacing .hip_fatbin with merged device image..."
  COMMAND ${LLVM_OBJCOPY} --update-section .hip_fatbin=${MERGED_DEVICE} $<TARGET_FILE:rccl>
  DEPENDS ${MERGED_DEVICE}
  COMMENT "Injecting merged device code into librccl.so"
)

# Make sure device_link runs before the POST_BUILD step
add_dependencies(rccl device_link)

message(STATUS "Device linker build configured")
message(STATUS "Build with: cmake --build <build_dir>")
message(STATUS "Output: librccl.so with merged device code")
