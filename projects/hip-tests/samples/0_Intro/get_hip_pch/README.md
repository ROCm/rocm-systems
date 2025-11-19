# get hip pch

Just verify that if HIP is built PCH, the __hipGetPCH function and associated macros are exported.

- Steps to build this sample:
```
$ mkdir build; cd build
$ cmake .. -DCMAKE_PREFIX_PATH=/opt/rocm
$ make
```

- Execute File
```
$ ./get_hip_pch
pch size: 11743288
__hipGetPCH succeeded!
PASSED!
```
