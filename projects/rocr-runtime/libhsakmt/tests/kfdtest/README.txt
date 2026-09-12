1. Note on building kfdtest

To build this kfdtest application, the following libraries should be already
installed on the building machine:
libdrm libdrm_amdgpu libhsakmt

If libhsakmt is not installed, but the headers and libraries are present
locally, you can specify its directory by
export LIBHSAKMT_PATH=/path/to/libhsakmt.a
With that, CMake/make will look for the lib at LIBHSAKMT_PATH/libhsakmt.a
Note that this assumes that you will be building kfdtest from the same thunk found in ../..

2. How to run kfdtest

Just run "./run_kfdtest.sh" under the building output folder. You may need
to specify library path through:
export LD_LIBRARY_PATH=/path/to/libhsakmt.a

Note: you can use "run_kfdtest.sh -h" to see more options.

3. Compartmentalized runs

"./run_kfdtest.sh -C" runs every test in its own kfdtest process and writes the
kernel messages it produced to
./kfdtest_logs/<timestamp>/node<N>-<asic>/<NNN>.<test>.klog, alongside
kernel-log.full for the whole run. NNN is the execution index, so a sorted
listing reads in run order and shows how far the run got. A test that logged
nothing leaves no .klog. Use -o <dir> for another directory.

A test is killed after --timeout seconds (300 by default, 0 disables) and
reported as TIMEOUT, so one stuck test cannot stall the run.

A test that takes the machine down writes no .klog, since the file appears once
the process returns. Each test is therefore named in the kernel log just before
it runs, recoverable after a panic through pstore or the next boot's journal:

  journalctl -k -b -1 | grep kfdtest-marker: | tail

Naming needs passwordless sudo, not merely a cached credential, which expires
mid-run. Without it the run says so once and carries on; --no-kmsg-marker
disables naming.

-C walks every HSA node, one log directory each. With -c/--concurrentnodes or
-t/--testnodenum all nodes go to a single invocation, as in a normal run, so
there is one pass and one directory.

Kernel messages come from "journalctl --kernel", falling back to dmesg then
passwordless sudo, so the run needs no root where kernel.dmesg_restrict is set
(the default on Ubuntu, Fedora and RHEL). The script prints how to grant access
if none work. journald rate limiting can drop messages during a storm; check
"dmesg" directly if a .klog looks truncated.

Isolating tests changes their behaviour: a failure that depends on state left by
an earlier test will not reproduce, and collecting logs between tests shifts
timing. Use a normal run for sign-off and this mode for triage.

