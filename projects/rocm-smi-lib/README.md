# rocm_smi_lib

C++ Library interface for ROCm-SMI to allow you to monitor/trace GPU system atributes 

- GPU Clocks
- Memory Clocks
- GPU Temperature 
- GPU Fan Speed  - If you have active cooled device with a fan. 

Example application is logging performance data like Kerenl execution time  vs GPU Temprature,  GPU and Memory Clocks 

### To build library and example:
mkdir -p build

cd build

cmake ..

make

cd ..

The above commands will result in building the library librocm_smi.so and
an example, rocm_smi_ex, which links with this library.


