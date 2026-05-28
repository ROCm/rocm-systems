# rocJPEG API Negative tests

This test suite is designed to perform negative testing on all rocJPEG APIs. The purpose of these tests is to validate the robustness and error-handling mechanisms of the rocJPEG APIs 
by providing invalid inputs, unexpected scenarios, or edge cases to ensure the APIs respond with appropriate error messages or behaviors.

## Prerequisites:

* Install [rocJPEG](../../README.md#build-and-install-instructions)

## Build

```shell
mkdir build && cd build
cmake ../
make -j
```

## Run

```shell
./rocjpegnegativetest
```