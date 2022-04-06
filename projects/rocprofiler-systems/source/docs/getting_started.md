# Getting Started

```eval_rst
.. toctree::
   :glob:
   :maxdepth: 3

   instrumenting
   runtime
   critical_trace
```

## Configuring Environment

Source the `setup-env.sh` script to prefix the `PATH`, `LD_LIBRARY_PATH`, etc. environment variables:

```bash
source /opt/omnitrace/share/omnitrace/setup-env.sh
```

Alternatively, if environment modules are supported, add the `<prefix>/share/modulefiles` directory to `MODULEPATH` via:

```bash
module use /opt/omnitrace/share/modulefiles
```

> ***Alternatively, the above line can be added to the `${HOME}/.modulerc` file.***

Once omnitrace is in the `MODULEPATH`, omnitrace can be loaded via `module load omnitrace/<VERSION>` and unloaded via `module unload omnitrace/<VERSION>`, e.g.:


```bash
module load omnitrace/1.0.0
module unload omnitrace/1.0.0
```

> ***You may need to also add the path to the ROCm libraries to `LD_LIBRARY_PATH`, e.g. `export LD_LIBRARY_PATH=/opt/rocm/lib:${LD_LIBRARY_PATH}`***

### Validating Environment Configuration

If all the following commands execute successfully with output, then you are ready to use omnitrace:

```bash
which omnitrace
which omnitrace-avail
omnitrace --help
omnitrace-avail --all
```
