# CMake generated Testfile for 
# Source directory: /Users/setupuser/github/TheRock/rocm-systems/projects/hip-remote-client
# Build directory: /Users/setupuser/github/TheRock/rocm-systems/projects/hip-remote-client/build
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test([=[hip_remote_basic]=] "/Users/setupuser/github/TheRock/rocm-systems/projects/hip-remote-client/build/hip_remote_test_basic")
set_tests_properties([=[hip_remote_basic]=] PROPERTIES  ENVIRONMENT "TF_DEBUG=1" LABELS "integration" _BACKTRACE_TRIPLES "/Users/setupuser/github/TheRock/rocm-systems/projects/hip-remote-client/CMakeLists.txt;129;add_test;/Users/setupuser/github/TheRock/rocm-systems/projects/hip-remote-client/CMakeLists.txt;0;")
add_test([=[hip_remote_extended]=] "/Users/setupuser/github/TheRock/rocm-systems/projects/hip-remote-client/build/hip_remote_test_extended")
set_tests_properties([=[hip_remote_extended]=] PROPERTIES  ENVIRONMENT "TF_DEBUG=1" LABELS "integration" _BACKTRACE_TRIPLES "/Users/setupuser/github/TheRock/rocm-systems/projects/hip-remote-client/CMakeLists.txt;146;add_test;/Users/setupuser/github/TheRock/rocm-systems/projects/hip-remote-client/CMakeLists.txt;0;")
add_test([=[hip_remote_phase2]=] "/Users/setupuser/github/TheRock/rocm-systems/projects/hip-remote-client/build/hip_remote_test_phase2")
set_tests_properties([=[hip_remote_phase2]=] PROPERTIES  ENVIRONMENT "TF_DEBUG=1" LABELS "integration" _BACKTRACE_TRIPLES "/Users/setupuser/github/TheRock/rocm-systems/projects/hip-remote-client/CMakeLists.txt;162;add_test;/Users/setupuser/github/TheRock/rocm-systems/projects/hip-remote-client/CMakeLists.txt;0;")
add_test([=[hip_remote_graphs]=] "/Users/setupuser/github/TheRock/rocm-systems/projects/hip-remote-client/build/hip_remote_test_graphs")
set_tests_properties([=[hip_remote_graphs]=] PROPERTIES  ENVIRONMENT "TF_DEBUG=1" LABELS "integration" _BACKTRACE_TRIPLES "/Users/setupuser/github/TheRock/rocm-systems/projects/hip-remote-client/CMakeLists.txt;178;add_test;/Users/setupuser/github/TheRock/rocm-systems/projects/hip-remote-client/CMakeLists.txt;0;")
add_test([=[hip_remote_ipc]=] "/Users/setupuser/github/TheRock/rocm-systems/projects/hip-remote-client/build/hip_remote_test_ipc")
set_tests_properties([=[hip_remote_ipc]=] PROPERTIES  ENVIRONMENT "TF_DEBUG=1" LABELS "integration" _BACKTRACE_TRIPLES "/Users/setupuser/github/TheRock/rocm-systems/projects/hip-remote-client/CMakeLists.txt;194;add_test;/Users/setupuser/github/TheRock/rocm-systems/projects/hip-remote-client/CMakeLists.txt;0;")
add_test([=[hip_remote_mempool]=] "/Users/setupuser/github/TheRock/rocm-systems/projects/hip-remote-client/build/hip_remote_test_mempool")
set_tests_properties([=[hip_remote_mempool]=] PROPERTIES  ENVIRONMENT "TF_DEBUG=1" LABELS "integration" _BACKTRACE_TRIPLES "/Users/setupuser/github/TheRock/rocm-systems/projects/hip-remote-client/CMakeLists.txt;210;add_test;/Users/setupuser/github/TheRock/rocm-systems/projects/hip-remote-client/CMakeLists.txt;0;")
add_test([=[hip_remote_graph_nodes]=] "/Users/setupuser/github/TheRock/rocm-systems/projects/hip-remote-client/build/hip_remote_test_graph_nodes")
set_tests_properties([=[hip_remote_graph_nodes]=] PROPERTIES  ENVIRONMENT "TF_DEBUG=1" LABELS "integration" _BACKTRACE_TRIPLES "/Users/setupuser/github/TheRock/rocm-systems/projects/hip-remote-client/CMakeLists.txt;226;add_test;/Users/setupuser/github/TheRock/rocm-systems/projects/hip-remote-client/CMakeLists.txt;0;")
add_test([=[hip_remote_device_apis]=] "/Users/setupuser/github/TheRock/rocm-systems/projects/hip-remote-client/build/hip_remote_test_device_apis")
set_tests_properties([=[hip_remote_device_apis]=] PROPERTIES  ENVIRONMENT "TF_DEBUG=1" LABELS "integration" _BACKTRACE_TRIPLES "/Users/setupuser/github/TheRock/rocm-systems/projects/hip-remote-client/CMakeLists.txt;242;add_test;/Users/setupuser/github/TheRock/rocm-systems/projects/hip-remote-client/CMakeLists.txt;0;")
add_test([=[hip_remote_memory_apis]=] "/Users/setupuser/github/TheRock/rocm-systems/projects/hip-remote-client/build/hip_remote_test_memory_apis")
set_tests_properties([=[hip_remote_memory_apis]=] PROPERTIES  ENVIRONMENT "TF_DEBUG=1" LABELS "integration" _BACKTRACE_TRIPLES "/Users/setupuser/github/TheRock/rocm-systems/projects/hip-remote-client/CMakeLists.txt;258;add_test;/Users/setupuser/github/TheRock/rocm-systems/projects/hip-remote-client/CMakeLists.txt;0;")
