1. Note on building kfdtest

To build this kfdtest application, the following libraries should be already
installed on the building machine:
libdrm libdrm_amdgpu libhsakmt

If libhsakmt is not installed, but the headers and libraries are present
locally, you can specify its directory by
export LIBHSAKMT_PATH=/path/to/libhsakmt.a
With that, CMake/make will look for the lib at LIBHSAKMT_PATH/libhsakmt.a
Note that this assumes that you will be building kfdtest from the same thunk found in ../..

2. How to run kfdtest

Just run "./run_kfdtest.sh" under the building output folder. You may need
to specify library path through:
export LD_LIBRARY_PATH=/path/to/libhsakmt.a

Note: you can use "run_kfdtest.sh -h" to see more options.
