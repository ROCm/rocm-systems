# RocDecode API Negative tests

This test suite is designed to perform negative testing on all rocDecode APIs. The purpose of these tests is to validate the robustness and error-handling mechanisms of the rocDecode APIs 
by providing invalid inputs, unexpected scenarios, or edge cases to ensure the APIs respond with appropriate error messages or behaviors.

## Prerequisites:

* Install [rocDecode](https://rocm.docs.amd.com/projects/rocDecode/en/latest/install/rocDecode-build-and-install.html)

## Build

```shell
mkdir build && cd build
cmake ../
make -j
```

## Run

```shell
./rocdecodenegativetest
```