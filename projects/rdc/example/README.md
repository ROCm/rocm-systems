# Examples


### How to compile examples?

***NOTE: You have to have RDC installed somewhere.***

If you have rocm (and RDC) installed under `/opt/rocm` - then you can simply do:

```bash
# same as 'mkdir -p build; cd build; cmake ../; cd ../'
cmake -B build
# same as 'cd build; make; cd ../'
make -C build
```

If you have rocm installed under a different directory, then you will have to
add that path with one of the following ways:

- `cmake -DROCM_DIR=/custom/rocm/path -B build`
- `ROCM_PATH=/custom/rocm/path cmake -B build`

followed by `make -C build`

You can also set ROCM\_PATH environment variable.


### I can't find rdc!

- Is RDC installed?
- Is RDC installed under `/opt/rocm`?
- Can you find `/opt/rocm/lib/cmake/rdc/rdcTargets.cmake`?


### Where is rdc?

```bash
ldd build/diagnostic
```

Look for `librdc_bootstrap.so`


### `diagnostic` is halted, but other examples work

Did you wait long enough?

It takes a while to run. 46 seconds on my machine with 2 GPUs.


### `Couldn't find the platform configure..`

### `Couldn't find the config for the Device...`

That's probably ok. The examples will still run.

Try to `cd` into the config directory and call these examples from there.
