.. meta::
   :description: Instruction on how to install the RCCL library for collective communication primitives using Docker
   :keywords: RCCL, ROCm, library, API, install, Docker

.. _install-docker:

*****************************************
Running RCCL using Docker
*****************************************

To use Docker to run RCCL, Docker must already be installed on the system.
To build the Docker image and run the container, follow these steps.

#. Build the Docker image

   By default, the Dockerfile uses ``docker.io/rocm/dev-ubuntu-22.04:latest`` as the base Docker image.
   It then installs RCCL and rccl-tests (in both cases, it uses the version from the ``develop`` branch).

   Use this command to build the Docker image:

   .. code-block:: shell

      docker build -t rccl-tests -f Dockerfile.ubuntu --pull .

   The base Docker image, rccl repository, rccl-tests repository, and GPU targets can be modified
   by using ``--build-args`` in the ``docker build`` command above. For example, to use a different base Docker image for the MI250 GPU,
   use this command:

   .. code-block:: shell

      docker build -t rccl-tests -f Dockerfile.ubuntu --build-arg="ROCM_IMAGE_NAME=rocm/dev-ubuntu-20.04" --build-arg="ROCM_IMAGE_TAG=6.2" --build-arg="GPU_TARGETS=gfx90a" --pull .

#. Launch an interactive Docker container on a system with AMD GPUs:

   .. code-block:: shell

      docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host --network=host --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -it rccl-tests /bin/bash

To run, for example, the ``all_reduce_perf`` test from rccl-tests on 8 AMD GPUs from inside the Docker container, use this command
for ROCm 6.4.1 or earlier:

.. code-block:: shell

   mpirun --allow-run-as-root -np 8 --mca pml ucx --mca btl ^openib -x NCCL_DEBUG=VERSION -x HSA_NO_SCRATCH_RECLAIM=1 /workspace/rccl-tests/build/all_reduce_perf -b 1 -e 16G -f 2 -g 1

For ROCm 6.4.2 or later, use this command:

.. code-block:: shell

   mpirun --allow-run-as-root -np 8 --mca pml ucx --mca btl ^openib -x NCCL_DEBUG=VERSION /workspace/rccl-tests/build/all_reduce_perf -b 1 -e 16G -f 2 -g 1

For more information on the rccl-tests options, see the `Usage guidelines <https://github.com/ROCm/rccl-tests#usage>`_ in the GitHub repository.
