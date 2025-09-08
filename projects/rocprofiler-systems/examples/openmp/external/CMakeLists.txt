# ----------------------------------------------------------------------------------------#
#
# OMPVV (OpenMP Validation and Verification) submodule
#
# * Currently, GNU's libgomp cannot be used to capture traces Only LLVM's LIBOMP
#   can be used
# * amdflang uses LIBOMP 201611 (5.0), but not all features are implemented.
#   Only a subset of tests can be compiled.
# ----------------------------------------------------------------------------------------#

rocprofiler_systems_message(STATUS "Configuring OMPVV...")

# Master branch contains stable releases
rocprofiler_systems_checkout_git_submodule(
  RELATIVE_PATH
  external/ompvv
  WORKING_DIRECTORY
  ${PROJECT_SOURCE_DIR}
  TEST_FILE
  Makefile
  REPO_URL
  https://github.com/OpenMP-Validation-and-Verification/OpenMP_VV.git
  REPO_BRANCH
  "master")

# ---------------------------------------------------------------------------------------#
# Compiler
# ---------------------------------------------------------------------------------------#

# OMPVV requires amdclang++
find_program(
  amdclangpp_EXECUTABLE
  NAMES amdclang++
  HINTS ${ROCM_PATH} ENV ROCM_PATH /opt/rocm
  PATHS ${ROCM_PATH} ENV ROCM_PATH /opt/rocm
  PATH_SUFFIXES bin llvm/bin)
mark_as_advanced(amdclangpp_EXECUTABLE)

if(NOT amdclangpp_EXECUTABLE)
  rocprofiler_systems_message(
    FATAL_ERROR
    "Could not find amdclang++. This is required for the OMPVV tests.")
endif()

find_program(
  amdflang_EXECUTABLE
  NAMES amdflang
  HINTS ${ROCM_PATH} ENV ROCM_PATH /opt/rocm
  PATHS ${ROCM_PATH} ENV ROCM_PATH /opt/rocm
  PATH_SUFFIXES bin llvm/bin)
mark_as_advanced(amdflang_EXECUTABLE)

if(NOT amdflang_EXECUTABLE)
  rocprofiler_systems_message(
    FATAL_ERROR
    "Could not find amdflang. This is required for the OMPVV tests.")
endif()

set(OMPVV_COMPILER
    "${amdflang_EXECUTABLE}"
    CACHE FILEPATH "Fortran compiler used for OMPVV tests")
set(OMPVV_FC "amdflang")

# ---------------------------------------------------------------------------------------#
# Variables
# ---------------------------------------------------------------------------------------#

set(OMPVV_OPENMP_VERSION 5.0)
set(OMPVV_NUM_THREADS_HOST ${ROCPROFSYS_THREAD_COUNT})
set(OMPVV_NUM_TEAMS_DEVICE 8) # default used by ompvv
set(OMPVV_NUM_THREADS_DEVICE 8) # default used by ompvv

# Set the tests to be compiled Tests listed are the ones that compile (excluding
# reduction due to OMPVV test failure) If any tests are added/removed, a
# complete rebuild is required
set(OMPVV_TESTS_TO_COMPILE
    "atomic/test_atomic_acquire_release.F90"
    # "atomic/test_atomic_hint.F90" "atomic/test_atomic_num_hint_device.F90"
    # "atomic/test_atomic_num_hint.F90"
    # "declare_target/test_declare_target_device_type_any.F90"
    # "declare_target/test_declare_target_device_type_host.F90"
    # "parallel_for_simd/test_parallel_for_simd_atomic.F90"
    "requires/test_requires_atomic_default_mem_order_relaxed.F90"
    # "requires/test_requires_atomic_default_mem_order_acq_rel.F90"
    # "requires/test_requires_atomic_default_mem_order_seq_cst.F90"
    # "simd/test_simd_if.F90" "simd/test_simd_nontemporal.F90"
    # "simd/test_simd_order_concurrent.F90"
    # "target_simd/test_target_simd_order_concurrent.F90"
    # "target_teams_distribute_parallel_for/test_target_teams_distribute_parallel_for_collapse.F90"
    # "target_teams_distribute_parallel_for_simd/test_target_teams_distribute_parallel_for_simd_atomic.F90"
    # "task/test_task_detach.F90" "teams/test_team_default_shared.F90"
    # "teams/test_teams.F90" # Bug due to task_detach callback failure
)

set(OMPVV_SRC_PREFIX "tests/${OMPVV_OPENMP_VERSION}")
set(OMPVV_SOURCES_LIST "")
foreach(TEST IN LISTS OMPVV_TESTS_TO_COMPILE)
  list(APPEND OMPVV_SOURCES_LIST "${OMPVV_SRC_PREFIX}/${TEST}")
endforeach()

set(OMPVV_FOFFLOADING "-fopenmp")
foreach(arch IN LISTS DEFAULT_GPU_TARGETS)
  set(OMPVV_FOFFLOADING "${OMPVV_FOFFLOADING} --offload-arch=${arch}")
endforeach()

# ---------------------------------------------------------------------------------------#
#
# modify make.def file
#
# ---------------------------------------------------------------------------------------#

file(READ ${PROJECT_SOURCE_DIR}/external/ompvv/sys/make/make.def
     MAKE_DEF_CONTENT)

# Replace FOFFLOADING
string(
  REGEX
  REPLACE
    "(ifeq \\(\\$\\(OMPVV_USED_FC\\), amdflang\\).*FOFFLOADING[ ]*=[ ]*)-fopenmp --offload-arch=gfx90a"
    "\\1${OMPVV_FOFFLOADING}"
    MAKE_DEF_CONTENT
    "${MAKE_DEF_CONTENT}")
# Remove all -O3 flags to prevent function inlining
string(REGEX REPLACE "-O3[ \t]*" "" MAKE_DEF_CONTENT "${MAKE_DEF_CONTENT}")

file(WRITE ${PROJECT_SOURCE_DIR}/external/ompvv/sys/make/make.def
     "${MAKE_DEF_CONTENT}")

# ---------------------------------------------------------------------------------------#
#
# build
#
# ---------------------------------------------------------------------------------------#

# OMPVV doesn't support building subsets of a test folder. Compile each test
# individually
foreach(test IN LISTS OMPVV_SOURCES_LIST)
  rocprofiler_systems_message(STATUS "Compiling OMPVV test: ${test}")

  execute_process(
    COMMAND
      make FC=${OMPVV_FC} DEVICE_TYPE=amd OMP_VERSION=${OMPVV_OPENMP_VERSION}
      NUM_THREADS_HOST=${OMPVV_NUM_THREADS_HOST}
      NUM_THREADS_DEVICE=${OMPVV_NUM_THREADS_DEVICE}
      NUM_TEAMS_DEVICE=${OMPVV_NUM_TEAMS_DEVICE} "SOURCES=${test}" compile
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/external/ompvv
    RESULT_VARIABLE OMPVV_BUILD_RESULT
    OUTPUT_VARIABLE OMPVV_BUILD_OUTPUT
    ERROR_VARIABLE OMPVV_BUILD_ERROR
    OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_STRIP_TRAILING_WHITESPACE)

  if(NOT OMPVV_BUILD_RESULT EQUAL 0)
    rocprofiler_systems_message(
      FATAL_ERROR "OMPVV build failed for ${test}: ${OMPVV_BUILD_ERROR}")
  endif()
endforeach()

# ---------------------------------------------------------------------------------------#
#
# copy over generated binaries to build folder for tests
#
# ---------------------------------------------------------------------------------------#

set(OMPVV_BIN_SOURCE "${PROJECT_SOURCE_DIR}/external/ompvv/bin")
set(OMPVV_BIN_DEST "${CMAKE_BINARY_DIR}")
set(OMPVV_TESTS "")

file(GLOB OMPVV_BIN_FILES "${OMPVV_BIN_SOURCE}/*")

# Create imported executable - required for rocprofiler_systems_add_test
foreach(OMPVV_BIN_FILE ${OMPVV_BIN_FILES})
  get_filename_component(FILENAME ${OMPVV_BIN_FILE} NAME)

  # For consistent naming with other tests
  string(REGEX REPLACE "\\.F90\\.o$" "" BIN_NAME ${FILENAME})
  string(REPLACE "_" "-" BIN_NAME ${BIN_NAME})
  file(COPY "${OMPVV_BIN_FILE}" DESTINATION "${OMPVV_BIN_DEST}")
  file(RENAME "${OMPVV_BIN_DEST}/${FILENAME}"
       "${OMPVV_BIN_DEST}/openmp-vv-${BIN_NAME}")

  set(TARGET_NAME "openmp-vv-${BIN_NAME}")
  add_executable(${TARGET_NAME} IMPORTED GLOBAL)
  set_target_properties(
    ${TARGET_NAME} PROPERTIES IMPORTED_LOCATION
                              "${CMAKE_BINARY_DIR}/openmp-vv-${BIN_NAME}")

  list(APPEND OMPVV_TESTS ${BIN_NAME})
endforeach()

set(OMPVV_TESTS
    "${OMPVV_TESTS}"
    CACHE STRING "Internal variable used to generate OMPVV CTests" FORCE)

rocprofiler_systems_message(STATUS "Succesfully built and compiled OMPVV tests")
