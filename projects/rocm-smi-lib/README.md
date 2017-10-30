# rocm_smi_lib

C++ Library interface for ROCm-SMI to allow you to monitor/trace GPU system atributes 

- GPU Clocks
- Memory Clocks
- Temperature 
- Fan Speed 

Example application is logging this with performance data to track kerenl execution time with GPU and Memory Clocks 

### To build library and example:
mkdir -p build

cd build

cmake ..

make

cd ..

The above commands will result in building the library librocm_smi.so and
an example, rocm_smi_ex, which links with this library.

