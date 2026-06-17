Container for rocSHMEM development with MPI support
===================================================

* Source: https://github.com/ROCm/rocm-systems/projects/rocshmem
* Documentation: https://rocm.docs.amd.com/projects/rocSHMEM/en/latest/docker
* Contact: rocshmem-maintainers@amd.com

Building the container
----------------------

```bash
# Run from the projects/rocshmem directory:
docker build -f docker/Dockerfile.ubuntu --tag $USER/rocshmem docker/
```

Build parameters:

* Build a PR branch AND devel in the container: `docker build -f docker/Dockerfile.ubuntu --build-arg PR_NUM=4574 --tag $USER/rocshmem-pr4574 docker/`
* Use a different SSH port: `docker build -f docker/Dockerfile.ubuntu --build-arg SSH_PORT=2223 --tag $USER/rocshmem docker/`

Moving the image to compute nodes:

`for node in node01 node02; do docker save $USER/rocshmem | ssh $node docker load; done`

Adding DeepEP/VLLM
------------------

This dockerfile does not contain, but can easily be adapted to build DeepEP/VLLM.
However, we recommend following up-to-date instructions from https://github.com/ROCm/DeepEP?tab=readme-ov-file#development.

If you still want to adapt this dockerfile:
1. Change the FROM to use a rocm/pytorch base, and uncomment OPTIONAL dependencies.
2. ADD rocm/deepep and RUN setup.py to build it in the docker.

How to run examples
-------------------

1. Sanity check: print IB/RoCE info, notably firmware version

```bash
docker run -it --rm --shm-size 64G --network host --device /dev/dri --device /dev/kfd --device /dev/infiniband --ipc host --group-add video --cap-add SYS_PTRACE --security-opt seccomp=unconfined --privileged \
  # Broadcom Thor2 specific options to import the host bnxt driver libraries (note: Thor2 bnxt GDA requires specific firmware version)
    -v /usr/local/lib/libbnxt_re-rdmav34.so:/usr/lib/x86_64-linux-gnu/libibverbs/libbnxt_re-rdmav34.so  -v /usr/local/lib/libbnxt_re.so:/usr/local/lib/libbnxt_re.so \
  # Pensando AMD Pollara specific options to import host ionic driver libraries (note: Pollara ionic GDA requires specific firmware version)
    -v /sys/class/infiniband:/sys/class/infiniband -v /usr/lib/x86_64-linux-gnu/libionic.so.1:/usr/lib/x86_64-linux-gnu/libionic.so.1 -v /usr/lib/x86_64-linux-gnu/libionic.so:/usr/lib/x86_64-linux-gnu/libionic.so \
    -v /usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so:/usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so \
    -v /etc/libibverbs.d/ionic.driver:/etc/libibverbs.d/ionic.driver \
  $USER/rocshmem ibv_devinfo
```

2. Sanity checks, print hardware as seen by rocSHMEM

```bash
docker run -it --rm --shm-size 64G --network host --device /dev/dri --device /dev/kfd --device /dev/infiniband --ipc host --group-add video --cap-add SYS_PTRACE --security-opt seccomp=unconfined --privileged \
  # Broadcom Thor2 options to import host driver libraries (note: Thor2 bnxt GDA requires specific firmware version)
    -v /usr/local/lib/libbnxt_re-rdmav34.so:/usr/lib/x86_64-linux-gnu/libibverbs/libbnxt_re-rdmav34.so  -v /usr/local/lib/libbnxt_re.so:/usr/local/lib/libbnxt_re.so \
  # Pensando AMD Pollara options to import host driver libraries (note: Pollara ionic GDA requires specific firmware version)
    -v /sys/class/infiniband:/sys/class/infiniband -v /usr/lib/x86_64-linux-gnu/libionic.so.1:/usr/lib/x86_64-linux-gnu/libionic.so.1 -v /usr/lib/x86_64-linux-gnu/libionic.so:/usr/lib/x86_64-linux-gnu/libionic.so \
    -v /usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so:/usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so \
    -v /etc/libibverbs.d/ionic.driver:/etc/libibverbs.d/ionic.driver \
  $USER/rocshmem tools/rocshmem_info
```

3. Launch persistent containers (you can use srun, pdsh, ssh, kubernetes, etc to deploy the containers, we show an example with srun here)

```bash
srun -N2 --ntasks-per-node=1 --gpus=8 \
  docker run -d --rm \
    --shm-size 64G --network host --device /dev/dri --device /dev/kfd --device /dev/infiniband --ipc host --group-add video --cap-add SYS_PTRACE --security-opt seccomp=unconfined --privileged
    # Broadcom Thor2 options to import host driver libraries (note: Thor2 bnxt GDA requires specific firmware version)
      -v /usr/local/lib/libbnxt_re-rdmav34.so:/usr/lib/x86_64-linux-gnu/libibverbs/libbnxt_re-rdmav34.so  -v /usr/local/lib/libbnxt_re.so:/usr/local/lib/libbnxt_re.so \
    # Pensando AMD Pollara options to import host driver libraries (note: Pollara ionic GDA requires specific firmware version)
      -v /sys/class/infiniband:/sys/class/infiniband -v /usr/lib/x86_64-linux-gnu/libionic.so.1:/usr/lib/x86_64-linux-gnu/libionic.so.1 -v /usr/lib/x86_64-linux-gnu/libionic.so:/usr/lib/x86_64-linux-gnu/libionic.so \
      -v /usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so:/usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so \
      -v /etc/libibverbs.d/ionic.driver:/etc/libibverbs.d/ionic.driver \
    --name $USER-rocshmem $USER/rocshmem \
# Default command will run rocshmem_info, then launch SSHD, lets print the outcome.
&& sleep 5 && srun -N 2 --ntasks-per-node=1 -l docker logs $USER-rocshmem
```

4. Run a simple rocshmem_waveput_nbi test with 16 Workgroups, 256 threads per WG in the persistent containers

`srun -n1 docker exec -it -e ROCSHMEM_DEBUG_LEVEL=ENV $USER-rocshmem mpiexec --report-bindings -n 2 tests/functional_tests/rocshmem_functional_tests -a 31 -w 16 -z 256`

5. Run the full rocSHMEM testsuite in the persistent containers

`srun -n1 docker exec -it $USER-rocshmem bash tests/rocshmem_functional_driver.sh /app/build/tests/functional_tests/rocshmem_functional_tests gda logs`

Performance comparison
----------------------

Build image with a PR branch and develop, then run comparison with one command:

```bash
docker build --build-arg PR_NUM=4574 --tag $USER/rocshmem-pr4574 \
  projects/rocshmem/docker/

mkdir -p pr4574-results
docker run --rm \
  --shm-size 64G --network host --device /dev/dri --device /dev/kfd \
  --ipc host --group-add video --cap-add SYS_PTRACE \
  --security-opt seccomp=unconfined --privileged \
  -v "$(pwd)/pr4574-results:/results" \
  $USER/rocshmem-pr4574 perf-compare --suite heatmap --iterations 5
# Plots: ./pr4574-results/heatmap_summary.png
```

See scripts/functional_tests/README-perf_compare.md for all options.

Important directories
---------------------

1. `/app/build1234`: if `--build-arg PR_NUM=1234` is provided, this is a build of the selected PR branch. This is the default working directory.
2. `/app/build`: this is a build of the _develop_ branch.
3. `/app/rocm-systems/projects/rocshmem`: source directory, it is a git repo, you may fetch, checkout and created new build directories as needed.
4. `/root/rocshmem`: the default install directory, if you build DeepEP it will pull rocshmem from that location.
5. `/app/ompi`: build of Open MPI with UCX support as required to run RO backend.
