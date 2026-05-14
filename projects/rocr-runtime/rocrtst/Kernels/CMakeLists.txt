
cmake_minimum_required(VERSION 2.8.0)

#
#  Setup build environment
#
#  1) Setup env var LLVM_DIR and OCL_BITCODE_DIR to point to
#     folders containing relevant libraries seperately
#
#     export LLVM_DIR="Path to Lightning build artifacts"
#
#     export OCL_BITCODE_DIR="Path containing AMDGCN Bitcode libraries"
#
#  2) Make an new folder called build under root folder
#
#     mkdir build
#
#  3) Enter into folder of build, and run CMAKE to generate makefile
#     and make it
#
#     cd build; cmake ..; make
#

if(WIN32)
  message("Windows platform is not supported")
  return()
endif()

#
# Flag to enable / disable verbose output.
#
SET( CMAKE_VERBOSE_MAKEFILE on )

set(PROJECT_NAME "CompileKernels")
project (${PROJECT_NAME})

#
# Validate LLVM related resources are available
#
if (NOT DEFINED ENV{LLVM_DIR})
  message("LLVM_DIR define is not set. Kernels cannot be built.")
  return()
endif()
# No longer using environment variables
# We use find_package to locate device libraries instead


set(CLANG $ENV{LLVM_DIR}/clang)
if (NOT EXISTS ${CLANG})
  message("Path to clang (${CLANG}) is not valid. Is LLVM_DIR defined correctly?")
  return()
endif()

#
# Define Opencl version if it is not defined
#
if (DEFINED ENV{OPENCL_VER})
  set(OPENCL_VER $ENV{OPENCL_VER})
else()
  message("OPENCL_VER define is not set. Using default")
  set(OPENCL_VER "2.0")
endif()

#
# Define list of Target Device types for which to get code objects
#
set(DEV_LIST "gfx803" "gfx900" CACHE STRING "List of Gfx Devices")
set(TARGET_DEV_LIST ${DEV_LIST})
separate_arguments(TARGET_DEV_LIST)

# Maintains a global list of targets to build
set (ROCM_CODEOBJ_LIST "" CACHE INTERNAL ROCM_CODEOBJ_LIST)

#
# Options that are passed along to Clang to enable code object generation
#
set(KERN_SUFFIX "kernels.hsaco")
# Find BITCODE_DIR if not already defined by parent CMakeLists.txt
if(NOT DEFINED BITCODE_DIR)
  # Try to find via AMDDeviceLibs package
  find_package(AMDDeviceLibs QUIET CONFIG)
  
  if(AMDDeviceLibs_FOUND)
    if(TARGET AMDDeviceLibs::oclc_isa_version_803)
      get_target_property(DEVICE_LIBS_LOCATION AMDDeviceLibs::oclc_isa_version_803 IMPORTED_LOCATION)
      get_filename_component(BITCODE_DIR "${DEVICE_LIBS_LOCATION}" DIRECTORY)
    elseif(TARGET AMDDeviceLibs::oclc_isa_version_900)
      get_target_property(DEVICE_LIBS_LOCATION AMDDeviceLibs::oclc_isa_version_900 IMPORTED_LOCATION)
      get_filename_component(BITCODE_DIR "${DEVICE_LIBS_LOCATION}" DIRECTORY)
    else()
      set(AMDDeviceLibs_FOUND FALSE)
    endif()
  endif()
  
  if(NOT AMDDeviceLibs_FOUND)
    # Find LLVM first
    find_package(LLVM QUIET CONFIG)
    
    # Search for bitcode directory
    find_path(BITCODE_DIR 
      NAMES "opencl.bc" "opencl.amdgcn.bc" "oclc_isa_version_803.bc"
      HINTS
        "${LLVM_LIBRARY_DIR}/amdgcn/bitcode"
        "${LLVM_LIBRARY_DIR}/../amdgcn/bitcode"
        "${CMAKE_INSTALL_PREFIX}/amdgcn/bitcode"
        "${CMAKE_INSTALL_PREFIX}/lib/amdgcn/bitcode"
      NO_DEFAULT_PATH
      REQUIRED
    )
  endif()
  
  message(STATUS "Kernels - Found BITCODE_DIR: ${BITCODE_DIR}")
endif()

# Check for old vs new device-libs layout
if(EXISTS "${BITCODE_DIR}/opencl.amdgcn.bc")
  set(BITCODE_ARGS "-nogpulib
    -Xclang -mlink-bitcode-file -Xclang ${BITCODE_DIR}/opencl.amdgcn.bc
    -Xclang -mlink-bitcode-file -Xclang ${BITCODE_DIR}/ockl.amdgcn.bc
    -Xclang -mlink-bitcode-file -Xclang ${BITCODE_DIR}/ocml.amdgcn.bc")
else()
  set(BITCODE_ARGS "--hip-device-lib-path=${BITCODE_DIR}")
endif()

#
# Compiles Opencl kernel into a AMDGcn code object
#
function(CompileKernel KRNL_NAME TARGET_DEV)

  #
  # Bind names for code object file and directory containing it
  set(KERNEL_DIR ${PROJECT_BINARY_DIR}/${TARGET_DEV})
  set(CODEASM_FILE "${KRNL_NAME}_${TARGET_DEV}.asm")
  set(CODEOBJ_FILE "${KRNL_NAME}_${TARGET_DEV}.hsaco")

  #
  # Add target name to a global list of target names. This must
  # be executed before the add_custom_target rule
  set(TARGET_NAME "${KRNL_NAME}_${TARGET_DEV}")
  set(ROCM_CODEOBJ_LIST ${ROCM_CODEOBJ_LIST} ${TARGET_NAME}
                      CACHE INTERNAL ROCM_CODEOBJ_LIST)

  #
  # Build clang arguments into a string and tokenize it into a list
  # The command "separate_arguments" will replace each instance of
  # space char with a semi-colon char. Like any other program clang
  # needs its arguments to be passed in as a list of tokens. The
  # following strings are used to generate a code object and code
  # asm files
  #
  string(CONCAT CODE_ARG_STR "-Xclang -finclude-default-header "
                "-target amdgcn-amdh-amdhsa -mcpu=${TARGET_DEV} "
                "${BITCODE_ARGS} -cl-std=CL${OPENCL_VER} "
                "${PROJECT_SOURCE_DIR}/${CL_FILE} -o ${KERNEL_DIR}/${CODEOBJ_FILE}")
  string(CONCAT ASM_ARG_STR "-S -Xclang -finclude-default-header "
                "-target amdgcn-amdh-amdhsa -mcpu=${TARGET_DEV} "
                "${BITCODE_ARGS} -cl-std=CL${OPENCL_VER} "
                "${PROJECT_SOURCE_DIR}/${CL_FILE} -o ${KERNEL_DIR}/${CODEASM_FILE}")
  set(ASM_ARG_LIST ${ASM_ARG_STR})
  set(CODE_ARG_LIST ${CODE_ARG_STR})
  separate_arguments(ASM_ARG_LIST)
  separate_arguments(CODE_ARG_LIST)

  #
  # Create a custom command to execute associated commands
  # and a target it is associated with
  #
  add_custom_command(OUTPUT ${KERNEL_DIR}/${KNAME_EXE}
                     COMMAND ${CMAKE_COMMAND} -E make_directory ${KERNEL_DIR}
                     COMMAND ${CLANG} ${ASM_ARG_LIST}
                     COMMAND ${CLANG} ${CODE_ARG_LIST}
                     COMMENT "BUILDING KERNEL..." VERBATIM)
  add_custom_target("${TARGET_NAME}" ALL DEPENDS "${KERNEL_DIR}/${KNAME_EXE}")

endfunction(CompileKernel)

function(buildCodeObjects kname)

  # Bind the name of CL file and associate
  # a name for the target
  set(KNAME_EXE "${kname}")
  set(CL_FILE "${kname}_kernel.cl")

  # Iterate through list of target devices
  foreach(tdev ${TARGET_DEV_LIST})
    CompileKernel(${kname} ${tdev})
  endforeach(tdev)

endfunction(buildCodeObjects)

buildCodeObjects("read")
buildCodeObjects("write")
buildCodeObjects("binary_search")

#
# Create a custom target which will build the full suite
# of code objects for all kernels and target device pairs
#
add_custom_target(rocm_code_objs ALL DEPENDS ${ROCM_CODEOBJ_LIST})

