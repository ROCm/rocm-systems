# Web-based GUI Analysis

```eval_rst
.. toctree::
   :glob:
   :maxdepth: 4
```

## Features

Omniperf's standalone GUI analyzer is a lightweight web page that can
be generated directly from the command-line. This option is provided
as an alternative for users wanting to explore profiling results
graphically, but without the additional setup requirements or
server-side overhead of Omniperf's detailed [Grafana
interface](https://amdresearch.github.io/omniperf/grafana_analyzer.html#)
option.  The standalone GUI analyzer is provided as simple
[Flask](https://flask.palletsprojects.com/en/2.2.x/) application
allowing users to view results from within a web browser.

```{admonition} Port forwarding

Note that the standalone GUI analyzer publishes a web interface on port 8050 by default.
On production HPC systems where profiling jobs run
under the auspices of a resource manager, additional ssh tunneling
between the desired web browser host (e.g. login node or remote workstation) and compute host may be
required. Alternatively, users may find it more convenient to download
profiled workloads to perform analysis on their local system.
```

## Usage

To launch the standalone GUI, include the `--gui` flag with your desired analysis command. For example:

```bash
$ omniperf analyze -p workloads/vcopy/mi200/ --gui

--------
Analyze
--------

Dash is running on http://0.0.0.0:8050/

 * Serving Flask app 'omniperf_analyze.omniperf_analyze' (lazy loading)
 * Environment: production
   WARNING: This is a development server. Do not use it in a production deployment.
   Use a production WSGI server instead.
 * Debug mode: off
 * Running on all addresses (0.0.0.0)
   WARNING: This is a development server. Do not use it in a production deployment.
 * Running on http://127.0.0.1:8050
 * Running on http://10.228.32.139:8050 (Press CTRL+C to quit)
```

At this point, users can then launch their web browser of choice and
go to http://localhost:8050/ to see an analysis page.



![Standalone GUI Homepage](images/standalone_gui.png)

```{tip}
To launch the web application on a port other than 8050, include an optional port argument:  
`--gui <desired port>`
```

When no filters are applied, users will see five basic sections derived from their application's profiling data:

1. Memory Chart Analysis
2. Empirical Roofline Analysis
3. Top Stats (Top Kernel Statistics)
4. System Info
5. System Speed-of-Light

To dive deeper, use the top drop down menus to isolate particular
kernel(s) or dispatch(s). You will then see the web page update with
metrics specific to the filter you've applied.

Once you have applied a filter, you will also see several additional
sections become available with detailed metrics specific to that area
of AMD hardware. These detailed sections mirror the data displayed in
Omniperf's [Grafana
interface](https://amdresearch.github.io/omniperf/grafana_analyzer.html#).


