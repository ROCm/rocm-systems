# hip-host Test Executable

This application makes various calls to the HIP runtime in an application without device code.
Without any device code present, the HIP compiler should not generate any calls to
`__hipRegisterFatBinary`, `__hipRegisterFunction`, etc. Thus, this application makes an explicit
call to `__hipUnregisterFatBinary(nullptr)` in the destructor of a global variable --
which (should) result in the destructor being invoked after rocprofiler-sdk has
finalized. The intention is to trigger the initialization of the `HipCompilerDispatchTable`
after rocprofiler-sdk has finalized. When a rocprofiler-sdk tool is loaded, the output should
have the following message:

> `... registration.cpp:###] rocprofiler-sdk has been finalized, ignoring rocprofiler_set_api_table("hip_compiler", 60400, 0, ..., 1) call`
