# GenericTarget.md

- Add hip/bin path to the PATH
```
$ export PATH=$PATH:[MYHIP]/bin
```

- Define environment variable
```
$ export HIP_PATH=[MYHIP]
```

- Create build folder
```
$ cd ~/hip/samples/0_Intro/genericTarget
  mkdir -p build && cd build
```

- Build with cmake
```
  cmake ..
  make
```

- Build without cmake
```

/opt/rocm/bin/hipcc ../square.cpp ../../../../catch/hipTestMain/hip_test_features.cc -I../../../common -I../../../../catch/include --offload-arch=gfx9-generic --offload-arch=gfx9-4-generic --offload-arch=gfx10-1-generic --offload-arch=gfx10-3-generic --offload-arch=gfx11-generic --offload-arch=gfx12-generic -mcode-object-version=6 -w -o squareGenericTarget

/opt/rocm/bin/hipcc ../square1.cpp ../../../../catch/hipTestMain/hip_test_features.cc -I../../../common -I../../../../catch/include --offload-arch=gfx9-generic --offload-arch=gfx9-4-generic:sramecc+:xnack- --offload-arch=gfx9-4-generic:sramecc-:xnack- --offload-arch=gfx9-4-generic:xnack+ --offload-arch=gfx10-1-generic --offload-arch=gfx10-3-generic --offload-arch=gfx11-generic --offload-arch=gfx12-generic -mcode-object-version=6 -w -o squareGenericTargetWithFeatures

/opt/rocm/bin/hipcc ../saxpy.cpp ../../../../catch/hipTestMain/hip_test_features.cc -I../../../common -I../../../../catch/include -o saxpyGenericTarget
```
- Execute tests
```

$ ./squareGenericTarget
info: running on device AMD Radeon RX 6900 XT
info: allocate host mem (  7.63 MB)
info: allocate device mem (  7.63 MB)
info: copy Host2Device
info: launch 'vector_square' kernel
info: copy Device2Host
info: check result
PASSED: generic target!

Best to run on Mi3XX to verify features
$ ./squareGenericTargetWithFeatures
info: running on device AMD Radeon RX 6900 XT
info: allocate host mem (  7.63 MB)
info: allocate device mem (  7.63 MB)
info: copy Host2Device
info: launch 'vector_square' kernel
info: copy Device2Host
info: check result
PASSED: generic targets!

$./saxpyGenericTarget
Find generic target gfx11-generic
SAXPY test passed
```