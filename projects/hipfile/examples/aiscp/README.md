# `aiscp`

`aiscp` is a simple test program that uses hipFile to copy a file.

To build it, create a build directory and use cmake to configure and build
the program. You may need to add the path(s) to ROCm and/or hipFile
if they are installed in non-standard locations.

```
cmake -DCMAKE_PREFIX_PATH="/path/to/rocm;/path/to/hipfile" /path/to/aiscp/dir
cmake --build .
```

Running it is simple. It works like the Linux `cp` command.

`aiscp SOURCE DEST`
