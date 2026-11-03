# rocSHMEM CI link-worker sweep

Authorized experiment for chat 01a19d9e-f83d-70b3-a860-4648f35b1ae9.

The dedicated experimental workflow builds only rocSHMEM using a pinned public
SDK, then relinks the same prepared objects at widths 1, 2, 4, 8, 16 and 31.
Both `rocshmem_info` and `rocshmem_functional_tests` retain all 31 architectures.
The original product concurrency default stays one.

The job requests `aws-linux-scale-rocm-prod`, records CPU affinity, cgroup quota
and memory, and requires at least 96 visible logical CPUs and no visible quota
below 96. Its subprocesses inherit an affinity restricted to 96 logical CPUs.
This is a CPU-budget experiment, not proof of 96 dedicated physical cores.

Source: rocm-systems a10dca087065a82f15ea89dc7d6fdd3c6bb2c246.
SDK: rocm[devel] 10.3.0a20261102.
Preparation is outside link timers and uses 32 build jobs. The complete stack,
RCCL, math libraries, packages and GPU runtime tests are not built or run.

The sweep checks compile commands, generated-header contents, exact link argv,
architecture coverage, input hashes and output parity. It records sampled process
RSS and rejects unsafe memory expansion. It repeats the best observed eligible
width against width two in alternating order. A winning link-only width does not
establish a whole-project/CI speedup, and simultaneous independent links can
multiply the memory requirement.

Publication is limited to a new experiment branch in ROCm/rocm-systems. No default
branch changes, merges, releases or deployment are included. The workflow has
read-only repository permissions and uploads measurement metadata/logs.
