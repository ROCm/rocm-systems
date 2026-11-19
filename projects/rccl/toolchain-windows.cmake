# Native Windows ROCm toolchain (clang in GCC-compatible driver mode).
if(NOT ROCM_PATH)
  file(TO_CMAKE_PATH "$ENV{HIP_PATH}" ROCM_PATH)
  set(ROCM_PATH "${ROCM_PATH}" CACHE PATH "ROCm SDK root")
endif()
set(CMAKE_CXX_COMPILER "${ROCM_PATH}/lib/llvm/bin/clang++.exe" CACHE FILEPATH "AMD C++ compiler")
set(CMAKE_C_COMPILER "${ROCM_PATH}/lib/llvm/bin/clang.exe" CACHE FILEPATH "AMD C compiler")
list(PREPEND CMAKE_PREFIX_PATH "${ROCM_PATH}")
