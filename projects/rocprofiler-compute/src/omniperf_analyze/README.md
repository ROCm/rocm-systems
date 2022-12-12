# Description

omniperf_analyze.py is a post-processing profiling tool with the raw data collected from omniperf.

## Features

- All Omniperf build-in metrics.

- Multiple runs base line comparison.
- Metrics customization: pick up subset of build-in metrics or build your own profiling configuration.
- Kernel, gpu-id, dispatch-id filters.

Run `omniperf analyze -h` for more details.

## Dependency

- pip3 or conda: install astunparse numpy tabulate pandas pyyaml

## Recommended workflow

- Do a comprehensive analysis with Omniperf GUI at the beginning.
- Choose your own customized subset of metrics with "-b" a.k.a. "--filter-metrics", or build your own config follwing [config_template](configs/panel_config_template.yaml).
- Quick optimization iterations and profiling with customized metrics.
- Redo a comprehensive analysis with Omniperf GUI at any milestone or at the end.

## Demo

- Single run
  
  `omniperf analyze -p path/to/profiling/results/`

- List top kernels
  
  `omniperf analyze -p path/to/profiling/results/  --list-kernels`

- List metrics

  `omniperf analyze -p path/to/profiling/results/  --list-metrics gfx90a`

- Customized profiling "system speed of light" and "CS_Busy" only
  
  `omniperf analyze -p path/to/profiling/results/  -b 2  5.1.0`

  NB: People can filter single metric or the whole IP block by its id.
      In this case, 1 is the id for "system speed of light" and 5.1.0 the id for metric "GPU Busy Cycles".

- Multiple runs
  
  `omniperf analyze -p workload1/path/  -p workload2/path/`

- Filter kernels
  
  `omniperf analyze -p workload1/path/ -k 0 -p workload2/path/ -k 0`

## FAQ

- tabulate doesn't print properly
  - export LC_ALL=C.UTF-8
  - export LANG=C.UTF-8

- python ast error: 'Constant' object has no attribute 'kind'
  
  It comes from a bug in the default astunparse 1.6.3 with python 3.8. 
  
  Seems good with python 3.7 and 3.9.
  
  Quick work-around:

  - pip3 uninstall astunparse

  - pip3 install astunparse==1.6.2