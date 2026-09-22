#!/usr/bin/env bash

for archive in data/*.tar.gz; do
  [ -e "$archive" ] || continue  # Skip if no files match
  tar -zxf "$archive"
done
