# Getting Started

```eval_rst
.. toctree::
   :glob:
   :maxdepth: 4
```

## Quickstart

1. **Launch & Profile the target application with the command line profiler**

    The command line profiler launches the target application, calls the rocProfiler API, and collects profile results for the specified kernels, dispatches, and/or IP blocks.  If not specified, Omniperf will default to collecting all available counters for all kernels/dispatches launched by the user's executable.

    To collect the default set of data for all kernels in the target application, launch, e.g.:
    ```shell
    $ omniperf profile -n vcopy_data -- ./vcopy 1048576 256
    ```
    The app runs, each kernel is launched, and profiling results are generated. By default, results are written to (e.g.,) ./workloads/vcopy_data (configurable via the `-n` argument). To collect all requested profile information, it may be required to replay kernels multiple times.

2. **Customize data collection**

    Options are available to specify for which kernels/metrics data should be collected.
    Note that filtering can be applied either in the profiling or analysis stage, however filtering at during profiling collection will often speed up your overall profiling run time.

    Some common filters include:

    - `-k`/`--kernel` enables filtering kernels by name. `-d`/`--dispatch` enables filtering based on dispatch ID
    - `-b`/`--ipblocks` enables collects metrics for only the specified (one or more) IP Blocks.

    To view available metrics by IP Block you can use the `--list-metrics` argument to view a list of all available metrics organized by IP Block.
    ```shell
    $ omniperf analyze --list-metrics <sys_arch>
    ```

3. **Analyze at the command line**

   After generating a local output folder (./workloads/\<name>), the command line tool can also be used to quickly interface with profiling results. View different metrics derived from your profiled results and get immediate access all metrics organized by IP block.

   If no kernel, dispatch, or ipblock filters are applied at this stage, analysis will be reflective of the entirety of the profiling data.

   To interact with profiling results from a different session, users just provide the workload path.  `-p`/`--path` enables users to analyze existing profiling data in the Omniperf CLI.

4. **Analyze in the Grafana GUI**

   To conduct a more in-depth analysis of profiling results we recommend users utilize the Omniperf Grafana GUI. To interact with profiling results, users must import their data to the MongoDB instance included in the Omniperf dockerfile.

    To interact with Grafana GUI data, stored in the Omniperf DB, users can enter ***database*** mode. For example:
   ```shell
    $ omniperf database --import [CONNECTION OPTIONS]
   ```

## Usage

### Modes
Modes change the fundamental behavior of the Omniperf command line tool. Depending on which mode is chosen, different command line options become available.

- **Profile**: Target application is launched on the local system utilizing AMD’s [ROC Profiler](https://github.com/ROCm-Developer-Tools/rocprofiler). Depending on the profiling options chosen, selected kernels, dispatches, and/or IP Blocks in the application are profiled and results are stored locally in an output folder (./workloads/\<name>).

    ```shell
    $ omniperf profile --help
    ```

- **Analyze**: Profiling data from `-p`/`--path` directory is loaded into the Omniperf CLI analyzer where users have immediate access to profiling results and generated metrics. Metrics are quickly generated from the entirety of your profiled application or a subset you’ve identified through the Omniperf CLI analysis filters.

    To gererate a lightweight GUI interface users can add the `--gui` flag to their analysis command.

    This mode is designed to be a middle ground to the highly detailed Omniperf Grafana GUI and is great for users who want immediate access to an IP Block they’re already familiar with.

    ```shell
    $ omniperf analyze --help
    ```

- **Database**: Our detailed Grafana GUI is built on a MongoDB database. `--import` profiling results to the DB to interact with the workload in Grafana or `--remove` the workload from the DB.

    Connection options will need to be specified. See the [*Grafana
    Analysis*](https://rocm.github.io/omniperf/analysis.html#grafana-gui-import) import section
    for more details on this.

    ```shell
    $ omniperf database --help
    ```

## Basic Operations

Operation | Mode | Required Arguments
:--|:--|:--
Profile a workload | profile | `--name`, `-- <profile_cmd>`
Standalone roofline analysis | profile | `--name`, `--roof-only`, `-- <profile_cmd>`
Import a workload to database | database | `--import`, `--host`, `--username`, `--workload`, `--team`
Remove a workload from database | database | `--remove`, `--host`, `--username`, `--workload`, `--team`
Launch standalone GUI from CLI | analyze | `--path`, `--gui`
Interact with profiling results from CLI | analyze | `--path`
