#!/bin/sh
config_dir=$(dirname "$0")
docker run --name prometheus -v $config_dir/prometheus.yml:/prometheus/prometheus.yml:ro --rm -p 127.0.0.1:9090:9090 prom/prometheus --web.enable-otlp-receiver
