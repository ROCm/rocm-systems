### This will test linking hiprtc::hiprtc interface in cmake

ROCM_PATH is the path where ROCM is installed. default path is /opt/rocm.

I. Build

```
$ mkdir -p build; cd build
$ rm -rf *;
$ CXX=<ROCM_PATH>/llvm/bin/amdclang++ cmake -DCMAKE_PREFIX_PATH=/opt/rocm ..
$ make
```

II. Test

```
$ ./test
SAXPY test completed
```
