.. meta::
   :description: Instructions for building and running rocSHMEM using Docker containers.
   :keywords: rocSHMEM, ROCm, Docker, container, build, run, GDA, IPC, RO

.. _docker-rocshmem:

------------------------------------------
Using the rocSHMEM Docker container
------------------------------------------

rocSHMEM provides a Dockerfile in ``docker/Dockerfile.ubuntu`` that builds a
development container with all backends, MPI support, and optional PR testing.

Building the container image
----------------------------

Build the default image from the ``docker/`` directory:

.. code-block:: bash

   cd docker
   docker build --tag $USER/rocshmem -f Dockerfile.ubuntu .

Build arguments
^^^^^^^^^^^^^^^

``PR_NUM``
   Build and test a specific pull request branch alongside the ``develop`` branch.

   .. code-block:: bash

      docker build --build-arg PR_NUM=4574 --tag $USER/rocshmem-pr4574 -f Dockerfile.ubuntu .

   When set, the container includes two build directories:

   * ``/app/build`` -- the ``develop`` branch build
   * ``/app/build<PR_NUM>`` -- the PR branch build (default working directory)

``SSH_PORT``
   Change the SSH port used by the internal SSH server (default: ``2222``).

   .. code-block:: bash

      docker build --build-arg SSH_PORT=2223 --tag $USER/rocshmem .

Base image
^^^^^^^^^^

The default base image is ``rocm/dev-ubuntu-24.04:latest``. Alternative base images
are listed in the Dockerfile comments. To build with DeepEP or VLLM support, use a
``rocm/pytorch`` base image and uncomment the optional dependency sections in the
Dockerfile.

Distributing the image
^^^^^^^^^^^^^^^^^^^^^^

To transfer the image to compute nodes:

.. code-block:: bash

   for node in node01 node02; do
     docker save $USER/rocshmem | ssh $node docker load
   done

Running containers
------------------

All ``docker run`` invocations require device and network access flags. The base
set of flags is:

.. code-block:: bash

   docker run -it --rm \
     --shm-size 64G \
     --network host \
     --device /dev/dri \
     --device /dev/kfd \
     --device /dev/infiniband \
     --ipc host \
     --group-add video \
     --cap-add SYS_PTRACE \
     --security-opt seccomp=unconfined \
     --privileged \
     $USER/rocshmem <command>

.. note::

   GDA backends require vendor-specific host driver libraries to be bind-mounted
   into the container. See :ref:`docker-nic-drivers` for details.

Verifying the setup
^^^^^^^^^^^^^^^^^^^

Print InfiniBand/RoCE device information:

.. code-block:: bash

   docker run -it --rm \
     --shm-size 64G --network host \
     --device /dev/dri --device /dev/kfd --device /dev/infiniband \
     --ipc host --group-add video --cap-add SYS_PTRACE \
     --security-opt seccomp=unconfined --privileged \
     $USER/rocshmem ibv_devinfo

Print rocSHMEM build and hardware information:

.. code-block:: bash

   docker run -it --rm \
     --shm-size 64G --network host \
     --device /dev/dri --device /dev/kfd --device /dev/infiniband \
     --ipc host --group-add video --cap-add SYS_PTRACE \
     --security-opt seccomp=unconfined --privileged \
     $USER/rocshmem tools/rocshmem_info

.. _docker-nic-drivers:

NIC driver bind mounts
^^^^^^^^^^^^^^^^^^^^^^

GDA backends require host NIC driver libraries that match the firmware on the NIC.
These must be bind-mounted into the container at run time.

**Broadcom Thor2:**

.. code-block:: bash

   -v /usr/local/lib/libbnxt_re-rdmav34.so:/usr/lib/x86_64-linux-gnu/libibverbs/libbnxt_re-rdmav34.so \
   -v /usr/local/lib/libbnxt_re.so:/usr/local/lib/libbnxt_re.so

**AMD Pensando Pollara (ionic):**

.. code-block:: bash

   -v /sys/class/infiniband:/sys/class/infiniband \
   -v /usr/lib/x86_64-linux-gnu/libionic.so.1:/usr/lib/x86_64-linux-gnu/libionic.so.1 \
   -v /usr/lib/x86_64-linux-gnu/libionic.so:/usr/lib/x86_64-linux-gnu/libionic.so \
   -v /usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so:/usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so \
   -v /etc/libibverbs.d/ionic.driver:/etc/libibverbs.d/ionic.driver

Launching persistent multi-node containers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The default container command starts ``rocshmem_info`` then launches an SSH daemon,
which enables ``mpiexec`` across containers. Use a job launcher (``srun``,
``pdsh``, Kubernetes, etc.) to deploy containers on multiple nodes:

.. code-block:: bash

   srun -N2 --ntasks-per-node=1 --gpus=8 \
     docker run -d --rm \
       --shm-size 64G --network host \
       --device /dev/dri --device /dev/kfd --device /dev/infiniband \
       --ipc host --group-add video --cap-add SYS_PTRACE \
       --security-opt seccomp=unconfined --privileged \
       --name $USER-rocshmem $USER/rocshmem

.. note::

   All containers must use the same Docker image so they share the same SSH key
   generated during the build. Alternatively, import an invariant SSH key pair by
   uncommenting the ``COPY id_ed25519`` lines in the Dockerfile.

Running tests
^^^^^^^^^^^^^

Run a single functional test (for example, ``rocshmem_waveput_nbi``) with 16
workgroups and 256 threads per workgroup inside persistent containers:

.. code-block:: bash

   srun -n1 docker exec -it -e ROCSHMEM_DEBUG_LEVEL=ENV \
     $USER-rocshmem mpiexec --report-bindings -n 2 \
     tests/functional_tests/rocshmem_functional_tests -a 31 -w 16 -z 256

Run the full functional test suite:

.. code-block:: bash

   srun -n1 docker exec -it $USER-rocshmem \
     bash tests/rocshmem_functional_driver.sh \
     /app/build/tests/functional_tests/rocshmem_functional_tests gda logs

Container directory layout
--------------------------

The container organizes files as follows:

.. list-table::
   :header-rows: 1

   * - Path
     - Description
   * - ``/app/build``
     - Build of the ``develop`` branch
   * - ``/app/build<PR_NUM>``
     - Build of the PR branch (only when ``PR_NUM`` is set)
   * - ``/app/rocm-systems/projects/rocshmem``
     - Source directory (full git repository)
   * - ``/root/rocshmem``
     - Default install prefix
   * - ``/app/ompi``
     - Open MPI build with UCX support

Environment variables
---------------------

The container sets the following environment variables by default:

* ``ROCSHMEM_TEST_UUID=1``
* ``ROCSHMEM_HEAP_SIZE=6442450944`` (6 GiB)
* ``OMPI_MCA_plm_rsh_args=-p<SSH_PORT>``
* ``OMPI_ALLOW_RUN_AS_ROOT=1``

Additional environment variables for specific cluster configurations are available
as comments in the Dockerfile. For a full list of rocSHMEM environment variables,
see :doc:`Environment variables <./api/env_variables>`.
