# Since all the plugins generate binary with the same name, build only one of them
add_subdirectory(example)
# add_subdirectory(ib_sharp)
# add_subdirectory(mock)
# OdinLink TB5 uses a distinct SONAME (librccl-net-odl_tb5.so) and can coexist.
add_subdirectory(odinlink-tb5)
