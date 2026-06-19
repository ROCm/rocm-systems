sanitizers
==========

hipFile has CMake support for running various sanitizers. There are
two CMake options that control this:

``AIS_USE_SANITIZERS``

This adds ``-fsanitize=<sanitizers>`` where <sanitizers> includes:

* address
* float-divide-by-zero
* integer
* leak
* local-bounds
* nullability
* undefined
* vptr

``AIS_USE_THREAD_SANITIZER``

This adds ``-fsanitize=thread``, which is not compatible with
``AIS_USE_SANITIZERS``.

NOTE: We currently only support clang's sanitizers. We'll add GNU in
      the future.

We don't set any sanitizer environment variables, but this should
still give you stack traces on failures, though you will need
sanitizer-built ROCm libraries to track anything that was allocated
in the HIP layer. We fetch and build GoogleTest with the sanitizer
options set instead of using the system GoogleTest when the sanitizers
are enabled.
