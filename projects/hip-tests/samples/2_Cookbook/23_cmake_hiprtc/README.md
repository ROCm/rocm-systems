### This will test linking hiprtc::hiprtc interface in cmake
I. Build

```
$ mkdir -p build; cd build
$ rm -rf *;
$ CXX="$(hipconfig -l)"/amdclang++ cmake -DCMAKE_PREFIX_PATH=/opt/rocm ..
$ make
```

II. Test

```
$ ./test
SAXPY test completed
```
