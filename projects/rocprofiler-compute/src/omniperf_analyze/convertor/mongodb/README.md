# Description

The convert is designed to convert major core metric definitions from Omniperf to omniperf_analyze. It depends on the syntax and usage of Grafana and MongoDB query in Omniperf. We are not intending to write a full parser of Grafana/MongoDB.

Originally, we hope to make it one-stop script, a.k.a, "convert -s 0". However, it seems 2 stages conversion is a good practice.

## Suggested workflow

- `./convert -s 2` to generate all yaml configs from ./modified_query to ./converted with exsiting dashboard json config.
- `mv ./converted ./converted_bak`
- `./convert -s 1` to generate all new original qureies into ./original_query
- 'diff ./original_query ./modified_query', and make minor changes in ./modified_query as your expectation.
- Check the build-in metrics in 00_id_00_build_in_variables.s0_original_query.json manually, which are from Grafana Variables. There are only 2 for now in parser.py build_in_vars.
- Specify the new "input_file_path" in the source code to make sure the dashboard json config to be converted.
- `./convert -s 2` 
- `diff ./converted ./converted_bak` to make sure all the changes as your expectation.
- Copy all metric tables from ./converted to ../../configs/gfx90a/ manually ONE by ONE.
- Run a basic test to verify the config, i.e., `omniperf_analyze.py -d sample/mi200`
- `diff ../../configs/gfx90a/ ../../configs/gfx908/`, and update the gfx908 configs accordingly.
- `diff ../../configs/gfx90a/ ../../configs/gfx906/`, and update the gfx906 configs accordingly.
- Run a basic test to verify the config, i.e., `omniperf_analyze.py -d sample/mi50 -d sample/mi100 -d sample/mi200`
