# Using RCCL/RCCL-Tests in a docker environment

## Docker build

Assuming you have docker installed on your system:

### To build the docker image :

By default, the given Dockerfile uses `docker.io/rocm/dev-ubuntu-22.04:latest` as the base docker image, and then installs RCCL (develop branch) and RCCL-Tests (develop branch), targetting `gfx942` GPUs.
```shell
$ docker build -t rccl-tests -f Dockerfile.ubuntu --pull .
```

The base docker image, rccl repo, rccl-tests repo, and GPU targets can be modified using `--build-args` in the `docker build` command above. E.g., to use a different base docker image for the MI250 GPU:
```shell
$ docker build -t rccl-tests -f Dockerfile.ubuntu --build-arg="ROCM_IMAGE_NAME=rocm/dev-ubuntu-20.04" --build-arg="ROCM_IMAGE_TAG=6.2" --build-arg="GPU_TARGETS=gfx90a" --pull .
```

### To start an interactive docker container on a system with AMD GPUs :

```shell
$ docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host --network=host --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -it rccl-tests /bin/bash
```

### To run rccl-tests (all\_reduce\_perf) on 8 AMD GPUs (inside the docker container) :

If using ROCm 6.3.x or earlier
```shell
$ mpirun --allow-run-as-root -np 8 --mca pml ucx --mca btl ^openib -x NCCL_DEBUG=VERSION -x HSA_NO_SCRATCH_RECLAIM=1 /workspace/rccl-tests/build/all_reduce_perf -b 1 -e 16G -f 2 -g 1
```

If using ROCm 6.4.0 or later
```shell
$ mpirun --allow-run-as-root -np 8 --mca pml ucx --mca btl ^openib -x NCCL_DEBUG=VERSION /workspace/rccl-tests/build/all_reduce_perf -b 1 -e 16G -f 2 -g 1
```

For more information on rccl-tests options, refer to the [Usage](https://github.com/ROCm/rccl-tests#usage) section of rccl-tests.


## Copyright

All modifications are copyright (c) 2019-2025 Advanced Micro Devices, Inc. All rights reserved.
