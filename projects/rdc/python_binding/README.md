# Quick start
If you do not have the RDC installed, please specify the RDC library path using:

    $ export LD_LIBRARY_PATH=<rdc_libs_path>

Then you can run RdcReader in python_binding folder:

    $ python RdcReader.py

# Prometheus plugin
Install the prometheus_client:

    $ pip install prometheus_client

Start the rdcd with auth and then run plugin to connect to it:

    $ python rdc_prometheus.py

Check the options of the plugin:

    $ python rdc_prometheus.py --help

Verify the plugin is running:

    $ curl localhost:5000

In the managment computer, install the Prometheus from
https://github.com/prometheus/prometheus

Modify the file prometheus_targets.json to add the compute nodes running the plugin.
Start the Prometheus

    $ prometheus --config.file=<full path of the rdc_prometheus_example.yml>

Browse to localhost:9090 in the management computer for metrics from RDC.

