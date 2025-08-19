# gpu_arch

- Build the sample using cmake
```
$ mkdir build; cd build
$ cmake .. -DCMAKE_PREFIX_PATH=/opt/rocm
$ make
```
- Execute the sample
```
$ ./gpuarch
success
```

## Note : This sample works on architectures gfx908 and above