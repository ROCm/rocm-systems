# ROCr SVM prefetch test completion and query checks

`SvmMemoryTestBasic::TestSVMPrefetch` treated every nonzero completion value
as pending. A negative asynchronous error therefore looped forever. It also
queried a set-only attribute, ignored the query status, and asserted an
unchanged preferred-location value that it had initialized to the GPU.

The test now accepts only zero as successful completion, polls with a
ten-second steady-clock deadline, and checks the status and result of a
`PREFETCH_LOCATION` query initialized to zero. This query checks the API's
recorded prefetch destination; it does not independently prove residency.
The deadline bounds the polling loop, not runtime teardown or native ioctls.

The corrected binary was rebuilt from the existing gfx1201 CI configuration.
Three interposed controls ran serially against libhsa on gfx1201:

- Negative completion: the test reports the asynchronous failure.
- Unchanged positive completion: each prefetch subtest reports timeout after
  ten seconds. Two applicable pools produce two failures in about 20.5 seconds,
  within the process watchdog's 25-second bound.
- Zero completion followed by a failed attribute query: the test reports the
  query failure instead of accepting the preinitialized policy value.

Every control exited with a failing test result. None timed out at the process
watchdog, generated kernel diagnostics, or left GPU handles. These are harness
controls, not successful migration runs. Artifacts are under
`target/rocr-test-fixes/runs/svm-harness-qualified-*`, with the injector,
build commands, and `kernel/svm-harness-health.json` in the same workspace.
