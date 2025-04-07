

To build the sample, first export the following environment variables:

export ROCR_DIR=<root of RocR install; for RocR includes and libraries>
export OPENCL_DIR=<root of OpenCL install; for required clang and bitcode libs>
export OPENCL_VER=<OpenCL version; e.g., "2.0">
export TARGET_DEVICE=<GPU type; e.g., "gfx803" or "gfx900">

Next, do the following:
mkdir build
cd build
cmake ..

Finally, do the following to build the application and respective kernels:

make
make sample_kernels

