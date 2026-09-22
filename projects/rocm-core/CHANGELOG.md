Change Logs for rocm-core

ROCm6.4.1 release
 - Added changelog.debian and copyright for debian packages.
ROCm6.3.0 release
 - Enabled Support for CMAKE Module Config.
ROCm6.1.0 release
 - Added new API to get ROCm install Path (getROCmInstallPath()) at runtime
   using dlinfo of rocm-core Target Library.
ROCm6.0.0 release
 - Added script to convert RUNPATH in libraries and binaries to RPATH.
 - Disabled ROCm file reorg backward compatibility

ROCm5.6.0 release
 - Added initial support for Adress Sanitizer(ASAN) enabled builds.
 - Full support of ASAN will be coming in later release

ROCm5.5.0 release
 - Added module file support for ROCm.
 - From ROCm5.5.0 release ROCm module files can be loaded as rocm/5.5.0
