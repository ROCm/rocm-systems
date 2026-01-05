
if (DEFINED ENV{ROCM_PATH})
  set(rocm_bin "$ENV{ROCM_PATH}/bin")
else()
  set(ROCM_PATH "/opt/rocm" CACHE PATH "Path to the ROCm installation.")
  set(rocm_bin "/opt/rocm/bin")
endif()

if (NOT DEFINED ENV{CXX})
  set(CMAKE_CXX_COMPILER "${rocm_bin}/amdclang++" CACHE PATH "Path to the C++ compiler")
else()
  set(CMAKE_CXX_COMPILER "$ENV{CXX}" CACHE PATH "Path to the C++ compiler")
endif()

if (NOT DEFINED ENV{CXXFLAGS})
  set(CMAKE_CXX_FLAGS_DEBUG "-g -O1")
  set(CMAKE_CXX_FLAGS_RELEASE "-O3")
endif()

if (NOT DEFINED ENV{CC})
  set(CMAKE_C_COMPILER "${rocm_bin}/amdclang" CACHE PATH "Path to the C compiler")
else()
  set(CMAKE_C_COMPILER "$ENV{CC}" CACHE PATH "Path to the C compiler")
endif()

if (NOT DEFINED ENV{CFLAGS})
  set(CMAKE_C_FLAGS_DEBUG "-g -O1")
  set(CMAKE_C_FLAGS_RELEASE "-O3")
endif()
