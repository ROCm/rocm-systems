# rocm_smi_lib

### To build library and example:
mkdir -p build

cd build

cmake ..

make

cd ..

The above commands will result in building the library librocm_smi.so and
an example, rocm_smi_ex, which links with this library.

