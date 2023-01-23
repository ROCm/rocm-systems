# High Level Design

```eval_rst
.. toctree::
   :glob:
   :maxdepth: 4
```

The [Omniperf](https://github.com/AMDResearch/omniperf) Tool is architecturally composed of three major components, as shown in the following figure.

- **Omniperf Profiling**: Acquire raw performance counters via application replay based on the [ROC Profiler](https://github.com/ROCm-Developer-Tools/rocprofiler). A set of MI200 specific micro benchmarks are also run to acquire the hierarchical roofline data. 

- **Omniperf Grafana Analyzer**: 
  - *Grafana database import*: All raw performance counters are imported into the backend MongoDB database for Grafana GUI analysis and visualization.
  - *Grafana GUI Analyzer*: A Grafana dashboard is designed to retrieve the raw counters info from the backend database. It also creates the relevant performance metrics and visualization.
- **Omniperf Standalone GUI Analyzer**: A standalone GUI is provided to enable performance analysis without importing data into the backend database.

![Omniperf Architectual Diagram](images/omniperf_architecture.png)

